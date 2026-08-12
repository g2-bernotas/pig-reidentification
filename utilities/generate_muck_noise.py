#!/usr/bin/env python
"""
Generate the band-limited noise fields used by the muck (dirt) augmentation.

The augmentation in `utilities/ioutils.py` darkens parts of the animal according
to a random noise field, imitating the dirt that accumulates on pigs during the
day. The fields are pre-generated once and stored as `.npy` files so that
training does not pay the cost of synthesising them, and so that the same set of
fields can be released alongside the paper.

Fractal (multi-octave) value noise is used, which is visually equivalent to the
Perlin/OpenSimplex fields used originally but needs nothing beyond numpy.

Example
-------
python utilities/generate_muck_noise.py --out data/muck_noise --count 200

Then either pass the directory to training:

    python train.py ... --muck_dir data/muck_noise

or export it once:

    set PIG_MUCK_NOISE_DIR=data/muck_noise      # Windows
    export PIG_MUCK_NOISE_DIR=data/muck_noise   # Linux / macOS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Quintic interpolant, as used by Perlin noise."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """One octave of value noise: a `cells` x `cells` random lattice, smoothly
    interpolated up to `size` x `size`."""
    lattice = rng.random((cells + 1, cells + 1)).astype(np.float32)

    coords = np.linspace(0, cells, size, endpoint=False, dtype=np.float32)
    i = np.floor(coords).astype(np.int32)
    frac = _smoothstep(coords - i)

    # Bilinear (smoothstep) interpolation of the lattice
    top = lattice[np.ix_(i, i)] * (1 - frac)[None, :] + lattice[np.ix_(i, i + 1)] * frac[None, :]
    bottom = lattice[np.ix_(i + 1, i)] * (1 - frac)[None, :] + lattice[np.ix_(i + 1, i + 1)] * frac[None, :]
    return top * (1 - frac)[:, None] + bottom * frac[:, None]


def fractal_noise(size: int, octaves: int, base_cells: int, persistence: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Sum several octaves of value noise, normalised to [0, 1]."""
    field = np.zeros((size, size), dtype=np.float32)
    amplitude, total = 1.0, 0.0

    for octave in range(octaves):
        cells = base_cells * (2 ** octave)
        if cells >= size:
            break
        field += amplitude * value_noise(size, cells, rng)
        total += amplitude
        amplitude *= persistence

    field /= max(total, 1e-8)
    field -= field.min()
    return field / max(field.max(), 1e-8)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Generate .npy noise fields for the muck augmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", required=True, help="Directory to write the .npy fields into")
    p.add_argument("--count", type=int, default=200, help="How many fields to generate")
    p.add_argument("--size", type=int, default=224, help="Field resolution (square)")
    p.add_argument("--octaves", type=int, default=5, help="Number of noise octaves")
    p.add_argument("--base-cells", type=int, default=4, help="Lattice size of the first octave")
    p.add_argument("--persistence", type=float, default=0.5,
                   help="Amplitude falloff per octave (lower = smoother, larger dirt patches)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for idx in range(args.count):
        field = fractal_noise(args.size, args.octaves, args.base_cells, args.persistence, rng)

        # Stored as uint8 so the files stay small; ioutils re-normalises on load
        np.save(out_dir / f"noise_{idx:04d}.npy", (field * 255.0).astype(np.uint8))

    print(f"Wrote {args.count} noise fields of {args.size}x{args.size} -> {out_dir}")


if __name__ == "__main__":
    main()
