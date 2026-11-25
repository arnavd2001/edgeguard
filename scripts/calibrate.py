#!/usr/bin/env python3
"""
Run the INT8 calibration pipeline on a dataset of clean inputs.

Usage:
    python scripts/calibrate.py \
        --data calibration_data.npz \
        --layers conv1 conv2 fc1 \
        --out configs/calibration.json \
        [--max-samples 1000] \
        [--max-cov-dim 512]

The --data file should be a .npz with:
  - "inputs": array of shape (N, ...) — the raw inputs
  - "<layer_name>": array of shape (N, C) — per-layer INT8 activations

If per-layer activations are not pre-computed, you can supply a PyTorch
model checkpoint and the script will run inference to extract them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edgeguard.detection.calibration import Calibrator


def load_activation_npz(
    path: Path,
    layer_names: list[str],
    max_samples: int,
) -> list[dict]:
    """Load pre-computed per-layer activations from .npz."""
    data = np.load(path, allow_pickle=False)
    n = min(max_samples, next(iter(data.values())).shape[0])
    print(f"[calibrate] Loading {n} samples from {path}")
    samples = []
    for i in range(n):
        sample = {}
        for name in layer_names:
            if name not in data:
                raise KeyError(f"Layer '{name}' not found in {path}. Available: {list(data.files)}")
            sample[name] = data[name][i]
        samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser(description="EdgeGuard INT8 calibration")
    parser.add_argument("--data",        required=True, help=".npz file with pre-computed activations")
    parser.add_argument("--layers",      nargs="+", required=True, help="Layer names to calibrate")
    parser.add_argument("--out",         default="configs/calibration.json", help="Output JSON path")
    parser.add_argument("--max-samples", type=int, default=2000, help="Max calibration samples")
    parser.add_argument("--max-cov-dim", type=int, default=512,
                        help="Layers wider than this skip covariance estimation")
    args = parser.parse_args()

    samples = load_activation_npz(Path(args.data), args.layers, args.max_samples)

    cal = Calibrator(
        layer_names=args.layers,
        max_cov_dim=args.max_cov_dim,
        verbose=200,
    )
    for s in samples:
        cal.feed_activations(s)

    stats = cal.finalize()
    out_path = Path(args.out)
    Calibrator.save(stats, out_path)
    print(f"[calibrate] Calibration saved → {out_path}")
    print(f"[calibrate] Layers: {list(stats.keys())}")
    for name, s in stats.items():
        print(f"  {name}: n={s.n_samples}, int8_range={s.int8_range}, "
              f"cov={'yes' if s.cov is not None else 'no (dim too large)'}")


if __name__ == "__main__":
    main()
