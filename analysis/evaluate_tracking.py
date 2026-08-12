#!/usr/bin/env python
"""
Longitudinal tracking evaluation: daily identification accuracy over time for
one or more embedding sets, plus an optional anchor-bank-size ablation.

This replaces the two near-identical tracking scripts from the original
codebase; the "day-to-day baseline" is now just `--baseline`, and the ablation
is `--ablation`.

Examples
--------
# every experiment listed in the config, with baseline and ablation
python analysis/evaluate_tracking.py \
    --config analysis/configs/experiments.example.json \
    --out-dir results/tracking \
    --baseline "Ins. Segm." --ablation "Ins. Segm."

# a single file, no config
python analysis/evaluate_tracking.py \
    --npz "AABB=results/aabb_v19_openset.npz" \
    --out-dir results/tracking

Outputs
-------
daily_accuracy.csv                  per-day accuracy for every series
summary.csv                         mean/min/max accuracy per series
longitudinal_tracking_comparison.png
anchor_bank_ablation.csv / .png     (with --ablation)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_style,
    load_embeddings,
    load_experiments,
    resolve_existing,
    shorten,
)
from tracker import run_tracker  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Longitudinal pig re-identification tracking evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="JSON file mapping experiment name -> .npz path")
    p.add_argument("--npz", action="append", default=[], metavar="NAME=PATH",
                   help="Add an experiment from the command line (repeatable)")
    p.add_argument("--out-dir", default="results/tracking")

    p.add_argument("--bank-size", type=int, default=5, help="Anchor bank capacity")
    p.add_argument("--min-votes", type=int, default=2, help="Votes needed to accept an assignment")
    p.add_argument("--link-conf-max", type=float, default=0.35,
                   help="Max mean matching cost for a day to enter the bank")
    p.add_argument("--init", default="gt", choices=["gt", "first"],
                   help="Day-0 seeding: ground-truth alignment, or arbitrary cluster labels")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--baseline", default=None, metavar="NAME",
                   help="Also run this experiment with bank size 1 / 1 vote as a day-to-day baseline")
    p.add_argument("--ablation", default=None, metavar="NAME",
                   help="Run an anchor-bank-size ablation on this experiment")
    p.add_argument("--ablation-sizes", type=int, nargs="+", default=[1, 2, 3, 5, 7, 10])

    p.add_argument("--extra-series", default=None,
                   help="CSV of externally computed accuracies to overlay "
                        "(columns: Experiment, Day Index or Date, Accuracy (%%))")
    p.add_argument("--id-parts", type=int, default=2,
                   help="Underscore-separated tokens that make up an identity label")
    p.add_argument("--title", default="Longitudinal Tracking Accuracy Over Time by Modality")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def plot_daily(plot_df: pd.DataFrame, out_path: Path, title: str, dpi: int) -> None:
    order, palette, markers, dashes = build_style(sorted(plot_df["Experiment"].unique()))

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(14, 7))
    ax = sns.lineplot(
        data=plot_df,
        x="Day Index",
        y="Accuracy (%)",
        hue="Experiment",
        style="Experiment",
        hue_order=order,
        style_order=order,
        markers=markers,
        dashes=dashes,
        palette=palette,
        linewidth=2.5,
        markersize=8,
    )

    days = sorted(plot_df["Day Index"].unique())
    tick_labels = [plot_df.loc[plot_df["Day Index"] == d, "Date"].iloc[0] for d in days]
    plt.xticks(ticks=days, labels=tick_labels, rotation=45, ha="right")

    plt.ylim(0, 105)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    plt.ylabel("Daily Recognition Accuracy (%)", fontweight="bold")
    plt.xlabel("Tracking Day", fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    plt.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.85, borderpad=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_ablation(abl_df: pd.DataFrame, out_path: Path, name: str, dpi: int) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=abl_df, x="Day Index", y="Accuracy (%)", hue="Bank Size",
        palette="viridis", linewidth=2.5, marker="o", markersize=7,
    )
    days = sorted(abl_df["Day Index"].unique())
    tick_labels = [abl_df.loc[abl_df["Day Index"] == d, "Date"].iloc[0] for d in days]
    plt.xticks(ticks=days, labels=tick_labels, rotation=45, ha="right")
    plt.ylim(0, 105)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    plt.ylabel("Daily Recognition Accuracy (%)", fontweight="bold")
    plt.xlabel("Tracking Day", fontweight="bold")
    plt.title(f"Impact of Anchor Bank Size on Tracking Accuracy ({name})",
              fontsize=16, fontweight="bold", pad=15)
    plt.legend(title="Anchor Bank Size", loc="lower left", fontsize=10, frameon=True, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = resolve_existing(load_experiments(args.config, args.npz))
    frames: list[pd.DataFrame] = []
    loaded: dict[str, pd.DataFrame] = {}

    for name, path in experiments.items():
        print(f"Tracking: {name}")
        df = load_embeddings(path, id_parts=args.id_parts)
        loaded[name] = df
        res = run_tracker(
            df,
            bank_size=args.bank_size,
            min_votes=args.min_votes,
            link_conf_max=args.link_conf_max,
            init=args.init,
            seed=args.seed,
        )
        res["Experiment"] = shorten(name)
        frames.append(res)
        print(f"  mean accuracy: {res['accuracy_pct'].mean():.2f}%")

    if args.baseline:
        key = args.baseline if args.baseline in loaded else None
        if key is None:  # allow either the raw or the shortened name
            key = next((k for k in loaded if shorten(k) == args.baseline), None)
        if key is None:
            print(f"  ! --baseline {args.baseline!r} is not one of {list(loaded)}; skipping")
        else:
            print(f"Tracking: Baseline (day-to-day, from {key})")
            res = run_tracker(
                loaded[key], bank_size=1, min_votes=1,
                link_conf_max=args.link_conf_max, init=args.init, seed=args.seed,
            )
            res["Experiment"] = "Baseline"
            frames.append(res)
            print(f"  mean accuracy: {res['accuracy_pct'].mean():.2f}%")

    plot_df = pd.concat(frames, ignore_index=True).rename(
        columns={"day_idx": "Day Index", "date": "Date", "accuracy_pct": "Accuracy (%)"}
    )[["Experiment", "Day Index", "Date", "Accuracy (%)", "status", "n_images", "mean_cost"]]

    if args.extra_series:
        extra = pd.read_csv(args.extra_series)
        if "Day Index" not in extra.columns and "Date" in extra.columns:
            day_of = dict(zip(plot_df["Date"], plot_df["Day Index"]))
            extra["Day Index"] = extra["Date"].astype(str).map(day_of)
        if "Date" not in extra.columns:
            date_of = dict(zip(plot_df["Day Index"], plot_df["Date"]))
            extra["Date"] = extra["Day Index"].map(date_of)
        extra["Experiment"] = extra["Experiment"].map(shorten)
        if extra["Day Index"].isna().any():
            print("  ! some --extra-series rows could not be aligned to a tracking day; dropping them")
            extra = extra.dropna(subset=["Day Index"])
        plot_df = pd.concat([plot_df, extra], ignore_index=True)

    plot_df.to_csv(out_dir / "daily_accuracy.csv", index=False)

    summary = (
        plot_df.groupby("Experiment")["Accuracy (%)"]
        .agg(mean="mean", std="std", min="min", max="max", days="count")
        .sort_values("mean", ascending=False)
        .round(2)
    )
    summary.to_csv(out_dir / "summary.csv")
    print("\n--- Mean tracking accuracy per series ---")
    print(summary.to_string())

    plot_daily(plot_df, out_dir / "longitudinal_tracking_comparison.png", args.title, args.dpi)
    print(f"\nSaved plot -> {out_dir / 'longitudinal_tracking_comparison.png'}")

    if args.ablation:
        key = args.ablation if args.ablation in loaded else next(
            (k for k in loaded if shorten(k) == args.ablation), None
        )
        if key is None:
            print(f"  ! --ablation {args.ablation!r} is not one of {list(loaded)}; skipping")
            return
        print(f"\n--- Anchor bank size ablation ({key}) ---")
        rows = []
        for size in args.ablation_sizes:
            res = run_tracker(
                loaded[key], bank_size=size,
                min_votes=1 if size == 1 else args.min_votes,
                link_conf_max=args.link_conf_max, init=args.init, seed=args.seed,
            )
            print(f"Bank size {size:2d} -> mean accuracy {res['accuracy_pct'].mean():.2f}%")
            res["Bank Size"] = f"Size {size}"
            rows.append(res)

        abl_df = pd.concat(rows, ignore_index=True).rename(
            columns={"day_idx": "Day Index", "date": "Date", "accuracy_pct": "Accuracy (%)"}
        )
        abl_df.to_csv(out_dir / "anchor_bank_ablation.csv", index=False)
        plot_ablation(abl_df, out_dir / "anchor_bank_ablation.png", shorten(key), args.dpi)
        print(f"Saved ablation plot -> {out_dir / 'anchor_bank_ablation.png'}")


if __name__ == "__main__":
    main()
