#!/usr/bin/env python
"""
Extract embeddings for a folder of pig crops using a trained metric-learning
checkpoint, and save them as a single .npz for downstream analysis.

Expected input layout — one sub-directory per identity:

    <data-root>/
        G19_238/
            cam1__20230412_1032.jpg
            ...
        G19_241/
        G20_239/

The capture date is parsed from each filename (default: the token after the
final "__", up to the next "_"). Override with --date-regex if your naming
differs.

Examples
--------
# open-set evaluation crops
python analysis/extract_embeddings.py \
    --checkpoint runs/aabb_v19/fold_0/best_model_state.pkl \
    --data-root data/bbox_cropped/open_set \
    --out results/aabb_v19_openset.npz

# CPU-only machine, smaller batches
python analysis/extract_embeddings.py ... --device cpu --batch-size 16 --num-workers 0

Output arrays
-------------
embeddings : (N, D) float32
labels     : (N,)   "<identity>_<date>"   (kept for backwards compatibility)
date       : (N,)   capture date string
fnames     : (N,)   source image path

A sidecar <out>.meta.json records the checkpoint, arguments, and package
versions so a released .npz can be traced back to how it was produced.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import date_from_filename  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class PigCropDataset(Dataset):
    """
    Letterbox-resize to `size` x `size` on a black canvas, scale to [0, 1].

    This reproduces the training-time preprocessing exactly. Note there is no
    mean/std normalisation — the network was trained on raw [0, 1] tensors, so
    adding ImageNet statistics here will silently degrade the embeddings.
    """

    def __init__(
        self,
        class_dirs: list[str],
        size: int = 224,
        date_sep: str = "__",
        date_regex: str | None = None,
    ):
        self.size = size
        self.date_sep = date_sep
        self.date_regex = date_regex
        self.samples: list[tuple[str, str]] = []
        for d in class_dirs:
            identity = os.path.basename(os.path.normpath(d))
            for img_path in sorted(glob.glob(os.path.join(d, "*"))):
                if Path(img_path).suffix.lower() in IMAGE_EXTS:
                    self.samples.append((img_path, identity))
        if not self.samples:
            raise SystemExit(f"No images found under {len(class_dirs)} class directories.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, identity = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        old_size = img.shape[:2]
        ratio = float(self.size) / max(old_size)
        new_size = tuple(int(x * ratio) for x in old_size)
        img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        y0 = (self.size - new_size[0]) // 2
        x0 = (self.size - new_size[1]) // 2
        canvas[y0 : y0 + new_size[0], x0 : x0 + new_size[1]] = img

        tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float() / 255.0
        date = date_from_filename(img_path, sep=self.date_sep, regex=self.date_regex)
        return tensor, f"{identity}_{date}", date, img_path


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
def infer_train_classes(ckpt_path: str) -> int | None:
    """Recover the training-time class count from the checkpoint's softmax head."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - depends on user checkpoint
        print(f"  ! could not inspect checkpoint ({exc}); pass --num-train-classes")
        return None
    # train.py saves {'epoch', 'model_state', 'optimizer_state'}; accept the
    # bare state dict and the more common 'state_dict' spelling too.
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
    if not isinstance(state, dict):
        return None
    for key, value in state.items():
        if key.endswith("fc_softmax.weight") and hasattr(value, "shape"):
            return int(value.shape[0])
    return None


def build_model(args, num_train_classes: int, num_eval_classes: int, device: torch.device):
    """
    Load the metric-learning backbone from the training repo.

    `analysis/` is expected to sit at the repo root, next to `models/`. Use
    --repo-root if you keep it somewhere else.
    """
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    try:
        from models.embeddings import resnet50
    except ImportError as exc:
        raise SystemExit(
            f"Could not import models.embeddings from {repo_root}.\n"
            f"Run this script from inside the training repo, or pass --repo-root. ({exc})"
        )

    model = resnet50(
        pretrained=True,
        num_classes=num_train_classes,
        ckpt_path=args.checkpoint,
        embedding_size=args.embedding_size,
    )
    # The classification head is unused at inference; resized only so the
    # module shape matches the evaluation identity count.
    model.fc_softmax = nn.Linear(1000, num_eval_classes)
    return model.to(device).eval()


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Batched embedding extraction for pig re-identification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to best_model_state.pkl")
    p.add_argument("--data-root", required=True, help="Directory containing one folder per identity")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--pattern", default="*", help="Glob for identity folders inside --data-root")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument(
        "--num-train-classes",
        type=int,
        default=None,
        help="Class count the checkpoint was trained with (inferred from the checkpoint if omitted)",
    )
    p.add_argument("--repo-root", default=None, help="Path to the training repo (for models.embeddings)")
    p.add_argument("--date-sep", default="__", help="Separator preceding the date token in filenames")
    p.add_argument("--date-regex", default=None, help="Regex with a named group 'date' (overrides --date-sep)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    class_dirs = sorted(glob.glob(os.path.join(args.data_root, args.pattern)))
    class_dirs = [d for d in class_dirs if os.path.isdir(d)]
    if not class_dirs:
        raise SystemExit(f"No identity folders matched {args.pattern!r} under {args.data_root}")

    num_eval_classes = len(class_dirs)
    num_train_classes = args.num_train_classes or infer_train_classes(args.checkpoint)
    if num_train_classes is None:
        raise SystemExit("Could not infer --num-train-classes from the checkpoint; pass it explicitly.")

    device = pick_device(args.device)
    print(f"Device: {device} | identities: {num_eval_classes} | train classes: {num_train_classes}")

    model = build_model(args, num_train_classes, num_eval_classes, device)

    dataset = PigCropDataset(
        class_dirs, size=args.image_size, date_sep=args.date_sep, date_regex=args.date_regex
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Images: {len(dataset)}")

    try:
        from tqdm import tqdm

        loader = tqdm(loader, desc="Extracting")
    except ImportError:
        pass

    embeddings, labels, dates, fnames = [], [], [], []
    with torch.no_grad():
        for imgs, label_dates, day, paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            emb, _ = model(imgs)
            embeddings.append(emb.cpu().numpy())
            labels.extend(label_dates)
            dates.extend(day)
            fnames.extend(paths)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=np.vstack(embeddings).astype(np.float32),
        labels=np.array(labels),
        date=np.array(dates),
        fnames=np.array(fnames),
    )

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "n_images": len(dataset),
        "n_identities": num_eval_classes,
        "args": vars(args),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved {len(dataset)} embeddings -> {out_path}")


if __name__ == "__main__":
    main()
