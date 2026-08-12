"""
Shared helpers for the pig re-identification analysis scripts.

Nothing in here touches PyTorch, so the evaluation scripts can be run on a
machine with no GPU (or no torch install at all) straight from the released
.npz embedding files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING MATHS
# ─────────────────────────────────────────────────────────────────────────────


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (works for 1-D and 2-D arrays)."""
    axis = 1 if x.ndim > 1 else 0
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-10)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance matrix. Assumes `a` and `b` are already L2-normalised."""
    return 1.0 - (a @ b.T)


def cluster_centroids(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Mean embedding per cluster, re-normalised to the unit sphere."""
    cents = np.zeros((k, x.shape[1]), dtype=np.float32)
    for c in range(k):
        m = labels == c
        if m.sum():
            cents[c] = l2_normalise(x[m].mean(0))
    return cents


def kmeans_cluster(x: np.ndarray, k: int, n_init: int = 50, seed: int = 42) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=500)
    return km.fit_predict(x)


# ─────────────────────────────────────────────────────────────────────────────
#  LABEL / DATE PARSING
# ─────────────────────────────────────────────────────────────────────────────
#
# The released .npz files store one label per image in the form
#
#     <pig_id>_<date>            e.g.  "G19_238_20230412"
#
# where <pig_id> itself contains an underscore (pen prefix + animal number).
# `id_parts` controls how many underscore-separated tokens make up the identity.

DEFAULT_ID_PARTS = 2


def label_to_identity(label: str, id_parts: int = DEFAULT_ID_PARTS) -> str:
    return "_".join(str(label).split("_")[:id_parts])


def date_from_filename(path: str, sep: str = "__", regex: str | None = None) -> str:
    """
    Extract a capture date from an image filename.

    Default convention: everything after the final `__`, up to the next `_`.
        "pen_G19__20230412_1032.jpg" -> "20230412"

    Pass `regex` with a named group `date` to override, e.g.
        --date-regex '(?P<date>\\d{4}-\\d{2}-\\d{2})'
    """
    name = Path(path).name
    if regex:
        m = re.search(regex, name)
        if not m:
            raise ValueError(f"date regex {regex!r} did not match filename {name!r}")
        return m.group("date")
    return name.split(sep)[-1].split("_")[0]


# ─────────────────────────────────────────────────────────────────────────────
#  NPZ IO
# ─────────────────────────────────────────────────────────────────────────────


def load_embeddings(
    path: str | Path,
    id_parts: int = DEFAULT_ID_PARTS,
    normalise: bool = False,
) -> pd.DataFrame:
    """
    Load an embedding .npz written by `extract_embeddings.py`.

    Returns a DataFrame with columns: embedding (np.ndarray), true_id, date, fname.
    Rows are sorted by date so that "day index" is chronological.

    `normalise=False` keeps the raw network output. The tracking evaluation
    normalises internally; the enrolment evaluation deliberately does not
    (k-means there operates in the unnormalised space, as in the paper).
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    missing = {"embeddings", "labels", "date"} - set(data.files)
    if missing:
        raise KeyError(f"{path.name} is missing array(s): {sorted(missing)}")

    emb = np.asarray(data["embeddings"], dtype=np.float32)
    if normalise:
        emb = l2_normalise(emb)

    fnames = data["fnames"].astype(str) if "fnames" in data.files else np.array([""] * len(emb))

    df = pd.DataFrame(
        {
            "embedding": list(emb),
            "true_id": [label_to_identity(l, id_parts) for l in data["labels"].astype(str)],
            "date": data["date"].astype(str),
            "fname": fnames,
        }
    )
    df.sort_values("date", inplace=True, ignore_index=True)
    return df


def stack_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].values).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT CONFIG
# ─────────────────────────────────────────────────────────────────────────────


def load_experiments(config: str | Path | None, cli_pairs: list[str] | None) -> dict[str, str]:
    """
    Build an ordered {display_name: npz_path} mapping from either a JSON config
    file or repeated `--npz "Name=path/to/file.npz"` arguments (or both; CLI
    entries win on name collision).
    """
    experiments: dict[str, str] = {}

    if config:
        with open(config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        entries = cfg["experiments"] if isinstance(cfg, dict) and "experiments" in cfg else cfg
        if isinstance(entries, dict):
            experiments.update({str(k): str(v) for k, v in entries.items()})
        else:  # list of {"name": ..., "path": ...}
            for e in entries:
                experiments[str(e["name"])] = str(e["path"])

    for pair in cli_pairs or []:
        if "=" not in pair:
            raise ValueError(f"--npz expects NAME=PATH, got {pair!r}")
        name, path = pair.split("=", 1)
        experiments[name.strip()] = path.strip()

    if not experiments:
        raise SystemExit("No experiments given. Use --config and/or --npz NAME=PATH.")
    return experiments


def resolve_existing(experiments: dict[str, str]) -> dict[str, str]:
    """Drop entries whose file is missing, with a warning, and flag duplicates."""
    found, seen_paths = {}, {}
    for name, path in experiments.items():
        p = Path(path)
        if not p.exists():
            print(f"  ! skipping {name!r}: file not found ({path})")
            continue
        resolved = str(p.resolve())
        if resolved in seen_paths:
            print(
                f"  ! warning: {name!r} points at the same file as {seen_paths[resolved]!r} "
                f"— these two curves will be identical"
            )
        seen_paths[resolved] = name
        found[name] = str(p)
    if not found:
        raise SystemExit("None of the configured .npz files exist. Check your paths.")
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT STYLING
# ─────────────────────────────────────────────────────────────────────────────
#
# Known modality names keep their paper colours; anything else falls back to a
# generated palette, so custom experiment names never crash the plot.

PAPER_PALETTE = {
    "Baseline": "#34495e",
    "Ins. Segm.": "#2ecc71",
    "Ins. Segm. (ND)": "#27ae60",
    "Ins. Segm. (tuned)": "#9b59b6",
    "OBB": "#f1c40f",
    "OBB (ND)": "#d4ac0d",
    "AABB": "#e74c3c",
    "AABB (ND)": "#c0392b",
}

PAPER_ORDER = [
    "Baseline",
    "Ins. Segm. (tuned)",
    "Ins. Segm.",
    "Ins. Segm. (ND)",
    "OBB",
    "OBB (ND)",
    "AABB",
    "AABB (ND)",
]

_MARKERS = ["o", "s", "^", "D", "X", "P", "v", "*", "<", ">", "h", "p"]


def build_style(names: list[str]) -> tuple[list[str], dict, list, list]:
    """
    Return (order, palette, markers, dashes) for a seaborn lineplot.

    Ordering follows the paper where names are recognised, then any extras
    alphabetically. `(ND)` / `Baseline` series are dashed so the ablation pairs
    read clearly in greyscale print.
    """
    import seaborn as sns

    known = [n for n in PAPER_ORDER if n in names]
    extra = sorted(n for n in names if n not in PAPER_ORDER)
    order = known + extra

    fallback = sns.color_palette("husl", max(len(extra), 1))
    palette, markers, dashes = {}, [], []
    for i, name in enumerate(order):
        palette[name] = PAPER_PALETTE.get(name) or fallback[extra.index(name) % len(fallback)]
        markers.append(_MARKERS[i % len(_MARKERS)])
        if name == "Baseline":
            dashes.append((3, 3))
        elif "(ND)" in name or "(No Dirt)" in name:
            dashes.append((2, 2))
        else:
            dashes.append((1, 0))
    return order, palette, markers, dashes


SHORT_NAMES = {
    "Instance Segmentation": "Ins. Segm.",
    "Instance Segmentation (No Dirt)": "Ins. Segm. (ND)",
    "Instance Segmentation (Daily FT)": "Ins. Segm. (tuned)",
    "Instance Crop": "Ins. Segm.",
    "Instance Crop (ND)": "Ins. Segm. (ND)",
    "OBB (No Dirt)": "OBB (ND)",
    "AABB (No Dirt)": "AABB (ND)",
    "Baseline (Day-to-Day)": "Baseline",
}


def shorten(name: str) -> str:
    return SHORT_NAMES.get(name, name)
