#!/usr/bin/env python
"""
Unsupervised enrolment evaluation: how accurate is a K-means enrolment if you
only keep the samples closest to each cluster centroid?

For every day, embeddings are clustered with K-means, clusters are mapped to
true identities with the Hungarian algorithm, and accuracy is measured over the
top-p% of each cluster ranked by distance to its centroid. Sweeping p traces
the precision/coverage trade-off of automatic enrolment.

Example
-------
python analysis/evaluate_enrolment.py \
    --config analysis/configs/experiments.example.json \
    --out-dir results/enrolment

Outputs
-------
enrolment_accuracy.csv      mean accuracy per retention level per experiment
enrolment_daily.csv         the per-day values behind those means
summary.txt                 the printed summary, saved for the paper
comparison_accuracy_by_percent.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_style,
    l2_normalise,
    load_embeddings,
    load_experiments,
    resolve_existing,
    shorten,
    stack_embeddings,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Unsupervised enrolment accuracy vs retention rate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="JSON file mapping experiment name -> .npz path")
    p.add_argument("--npz", action="append", default=[], metavar="NAME=PATH",
                   help="Add an experiment from the command line (repeatable)")
    p.add_argument("--out-dir", default="results/enrolment")
    p.add_argument("--clusters", type=int, default=None,
                   help="K for K-means (default: number of distinct identities in the file)")
    p.add_argument("--retention", type=int, nargs="+", default=list(range(10, 101, 10)),
                   help="Retention percentages to sweep")
    p.add_argument("--normalise", action="store_true",
                   help="L2-normalise embeddings before clustering. Off by default to match "
                        "the published numbers, which cluster in the raw embedding space.")
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id-parts", type=int, default=2)
    p.add_argument("--title", default="Unsupervised Enrolment Accuracy vs. Retention Rate")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def evaluate_day(day_df: pd.DataFrame, k: int, retention: list[int], seed: int, n_init: int):
    """Return [{percent, accuracy, n_selected}] for one day."""
    x = stack_embeddings(day_df)
    if len(x) < k:
        return []

    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    day_df = day_df.reset_index(drop=True).copy()
    day_df["cluster"] = km.fit_predict(x)

    # Hungarian mapping of clusters to identities via the overlap matrix.
    overlap = pd.crosstab(day_df["cluster"], day_df["true_id"])
    overlap = overlap.reindex(index=range(k), fill_value=0)
    rows, cols = linear_sum_assignment(-overlap.values)
    cluster_to_id = {int(overlap.index[r]): overlap.columns[c] for r, c in zip(rows, cols)}

    day_df["dist_to_centroid"] = np.linalg.norm(x - km.cluster_centers_[day_df["cluster"]], axis=1)

    out = []
    for pct in retention:
        keep = []
        for c in range(k):
            members = day_df.index[day_df["cluster"] == c]
            if len(members) == 0:
                continue
            n_keep = max(1, int(np.ceil(len(members) * pct / 100.0)))
            ranked = day_df.loc[members].sort_values("dist_to_centroid").index.values
            keep.extend(ranked[:n_keep].tolist())
        if not keep:
            continue
        sel = day_df.loc[keep]
        pred = sel["cluster"].map(cluster_to_id)
        valid = pred.notnull()
        acc = float((pred[valid] == sel.loc[valid, "true_id"]).mean()) if valid.any() else 0.0
        out.append({"percent": pct, "accuracy": acc, "n_selected": len(keep)})
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = resolve_existing(load_experiments(args.config, args.npz))
    daily_rows, agg_frames = [], []

    for name, path in experiments.items():
        print(f"Processing: {name}")
        df = load_embeddings(path, id_parts=args.id_parts, normalise=args.normalise)
        k = args.clusters or df["true_id"].nunique()
        print(f"  identities: {df['true_id'].nunique()} | K = {k} | days: {df['date'].nunique()}")

        rows = []
        for date in sorted(df["date"].unique()):
            for r in evaluate_day(
                df[df["date"] == date], k, args.retention, args.seed, args.n_init
            ):
                rows.append({"Experiment": shorten(name), "date": date, **r})

        if not rows:
            print("  ! no day had enough images to cluster; skipping")
            continue

        daily_rows.extend(rows)
        agg = (
            pd.DataFrame(rows).groupby("percent")["accuracy"].mean().reset_index()
        )
        agg["Model Type"] = shorten(name)
        agg_frames.append(agg)

    combined = pd.concat(agg_frames, ignore_index=True)
    pd.DataFrame(daily_rows).to_csv(out_dir / "enrolment_daily.csv", index=False)
    combined.to_csv(out_dir / "enrolment_accuracy.csv", index=False)

    # ── plot ────────────────────────────────────────────────────────────────
    order, palette, markers, dashes = build_style(sorted(combined["Model Type"].unique()))

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=combined, x="percent", y="accuracy", hue="Model Type", style="Model Type",
        hue_order=order, style_order=order, palette=palette,
        markers=markers, dashes=dashes, linewidth=2.5, markersize=8,
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, symbol=""))
    plt.xlabel("Percentage of Dataset Retained (%)", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.title(args.title, fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(args.retention)
    plt.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.85, borderpad=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_accuracy_by_percent.png", dpi=args.dpi)
    plt.close()

    # ── summary ─────────────────────────────────────────────────────────────
    lines = ["=" * 50, " SUMMARY STATISTICS FOR RESULTS SECTION", "=" * 50]
    for model in order:
        m = combined[combined["Model Type"] == model]
        if m.empty:
            continue
        lines.append(f"\nModality: {model}")
        lines.append("  - Accuracy by retention threshold:")
        for pct in sorted(m["percent"].unique()):
            lines.append(f"      {pct:3d}% retained: {m.loc[m['percent'] == pct, 'accuracy'].iloc[0] * 100:.2f}%")

        lo, hi = m["percent"].min(), m["percent"].max()
        drop = (m.loc[m["percent"] == lo, "accuracy"].iloc[0]
                - m.loc[m["percent"] == hi, "accuracy"].iloc[0]) * 100
        lines.append(f"  - Accuracy drop ({lo}% -> {hi}% retained): {drop:.2f} pp")

        high = m[m["accuracy"] >= 0.95]
        if high.empty:
            lines.append("  - Recommended threshold: never reached 95% accuracy.")
        else:
            best = int(high["percent"].max())
            acc = high.loc[high["percent"] == best, "accuracy"].iloc[0] * 100
            lines.append(f"  - Recommended threshold (max retention at >=95%): "
                         f"retain {best}% (accuracy {acc:.2f}%)")
    lines.append("\n" + "=" * 50)

    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(f"\nSaved plot -> {out_dir / 'comparison_accuracy_by_percent.png'}")


if __name__ == "__main__":
    main()
