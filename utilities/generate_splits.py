#!/usr/bin/env python
"""
Generate the known/unknown folds file that train.py and test.py expect.

The identities present in `<dataset_root>/images/train` are split into a "known"
set (used to learn the embedding) and an "unknown" set (held back entirely, and
used to evaluate open-set generalisation). One such split is produced per fold.

Output format:

    {
      "0": { "known": ["G19_238", ...], "unknown": ["G20_239", ...] },
      "1": { ... }
    }

Examples
--------
# 50/50 known/unknown, 5 folds (the paper's open-set protocol)
python utilities/generate_splits.py \
    --dataset_root data/OpenSetPigs \
    --out data/OpenSetPigs/splits/50-50.json \
    --known-fraction 0.5 --num-folds 5

# every identity known (closed set), e.g. for fine-tuning runs
python utilities/generate_splits.py \
    --dataset_root data/OpenSetPigs \
    --out data/OpenSetPigs/splits/100-0.json \
    --known-fraction 1.0 --num-folds 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def list_identities(dataset_root: str) -> list[str]:
    train_dir = Path(dataset_root) / "images" / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"No such directory: {train_dir}\n"
                         f"--dataset_root should contain images/train and images/test.")

    identities = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    if not identities:
        raise SystemExit(f"No identity folders found under {train_dir}")
    return identities


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Create a known/unknown folds JSON file for OpenSetPigs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_root", required=True,
                   help="Dataset root containing images/train and images/test")
    p.add_argument("--out", required=True, help="Path of the JSON file to write")
    p.add_argument("--known-fraction", type=float, default=0.5,
                   help="Fraction of identities that are 'known' (1.0 = closed set)")
    p.add_argument("--num-folds", type=int, default=5, help="How many folds to generate")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not 0.0 < args.known_fraction <= 1.0:
        raise SystemExit("--known-fraction must be in (0, 1]")

    identities = list_identities(args.dataset_root)
    num_known = max(1, int(round(len(identities) * args.known_fraction)))

    rng = random.Random(args.seed)
    folds = {}
    for fold in range(args.num_folds):
        shuffled = identities[:]
        rng.shuffle(shuffled)
        folds[str(fold)] = {
            "known": sorted(shuffled[:num_known]),
            "unknown": sorted(shuffled[num_known:]),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(folds, fh, indent=4)

    print(f"{len(identities)} identities -> {num_known} known / "
          f"{len(identities) - num_known} unknown, {args.num_folds} fold(s)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
