#!/usr/bin/env python
"""
Write a small synthetic embedding .npz in the released format.

Nothing here comes from real animals - it exists so that the evaluation scripts
can be smoke-tested (and their plots eyeballed) without downloading the dataset
or the released embeddings:

    python analysis/make_synthetic_npz.py --out /tmp/fake.npz
    python analysis/evaluate_tracking.py --npz "Demo=/tmp/fake.npz" --out-dir /tmp/demo

Each identity gets a random anchor direction. Every day, the anchor slowly drifts
(imitating appearance change over weeks) and per-image Gaussian noise is added,
so daily clusters are separable but progressively harder to link across days.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a synthetic embedding .npz for testing the analysis scripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--identities", type=int, default=12, help="Number of distinct animals")
    p.add_argument("--days", type=int, default=14, help="Number of capture days")
    p.add_argument("--per-day", type=int, default=25, help="Images per identity per day")
    p.add_argument("--dim", type=int, default=128, help="Embedding dimensionality")
    p.add_argument("--noise", type=float, default=0.25, help="Per-image noise scale")
    p.add_argument("--drift", type=float, default=0.05, help="Per-day identity drift scale")
    p.add_argument("--start-date", default="20230412", help="First date, YYYYMMDD")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    # Identity names mirror the released data: "<pen>_<animal>"
    names = [f"G19_{200 + i}" for i in range(args.identities)]

    # One random unit anchor per identity, well separated in expectation
    anchors = rng.normal(size=(args.identities, args.dim)).astype(np.float32)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)

    dates = [
        (np.datetime64(f"{args.start_date[:4]}-{args.start_date[4:6]}-{args.start_date[6:]}")
         + np.timedelta64(d, "D")).astype(str).replace("-", "")
        for d in range(args.days)
    ]

    embeddings, labels, date_col, fnames = [], [], [], []
    for day, date in enumerate(dates):
        # Identities drift a little every day
        anchors = anchors + args.drift * rng.normal(size=anchors.shape).astype(np.float32)
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)

        for i, name in enumerate(names):
            noise = args.noise * rng.normal(size=(args.per_day, args.dim)).astype(np.float32)
            emb = anchors[i][None, :] + noise
            embeddings.append(emb)
            labels.extend([f"{name}_{date}"] * args.per_day)
            date_col.extend([date] * args.per_day)
            fnames.extend([
                f"synthetic/{name}/cam1__{date}_{day:02d}{j:04d}.png" for j in range(args.per_day)
            ])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=np.concatenate(embeddings, axis=0).astype(np.float32),
        labels=np.array(labels),
        date=np.array(date_col),
        fnames=np.array(fnames),
    )

    print(f"Wrote {len(labels)} synthetic embeddings "
          f"({args.identities} identities x {args.days} days) -> {out_path}")


if __name__ == "__main__":
    main()
