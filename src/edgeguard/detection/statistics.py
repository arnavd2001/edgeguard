"""
Streaming INT8-aware activation statistics.

Key design decision: all statistics are computed on INT8 activations directly,
not on dequantized float32 values.  This eliminates the rounding artifacts that
would appear if we borrowed FP32 calibration baselines and then re-quantized.

Supported anomaly scores:
  - Z-score (per-channel mean / std, fast, O(d))
  - Mahalanobis distance (full covariance, Lee et al. NeurIPS 2018, O(d²))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import pinvh  # pseudo-inverse, stable for near-singular cov


@dataclass
class LayerStats:
    """
    Calibration statistics for one layer, computed over INT8 activations.

    Attributes:
        layer_name:  Identifier matching the model layer.
        mean:        Per-channel mean vector  shape (d,).
        std:         Per-channel std vector   shape (d,) — for z-score.
        cov:         Full covariance matrix   shape (d, d) — for Mahalanobis.
        cov_inv:     Moore-Penrose pseudo-inverse of cov (pre-computed).
        n_samples:   Number of calibration samples used.
        int8_range:  (min, max) of observed INT8 activation values.
    """
    layer_name: str
    mean: np.ndarray              # float64 (d,)
    std: np.ndarray               # float64 (d,)
    cov: Optional[np.ndarray]     # float64 (d, d) — None if d is too large
    cov_inv: Optional[np.ndarray] # float64 (d, d) — pre-inverted
    n_samples: int
    int8_range: tuple[int, int]

    def to_dict(self) -> dict:
        d = {
            "layer_name": self.layer_name,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_samples": self.n_samples,
            "int8_range": list(self.int8_range),
        }
        if self.cov is not None:
            d["cov"] = self.cov.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "LayerStats":
        mean = np.array(data["mean"], dtype=np.float64)
        std = np.array(data["std"], dtype=np.float64)
        cov = np.array(data["cov"], dtype=np.float64) if "cov" in data else None
        cov_inv = None
        if cov is not None:
            try:
                cov_inv = pinvh(cov)
            except Exception:
                cov_inv = None
        return cls(
            layer_name=data["layer_name"],
            mean=mean,
            std=std,
            cov=cov,
            cov_inv=cov_inv,
            n_samples=data["n_samples"],
            int8_range=tuple(data["int8_range"]),
        )


class StreamingInt8Stats:
    """
    Online (streaming) computation of mean, variance, and covariance
    over INT8 activation vectors using Welford's algorithm.

    Accepts one sample at a time to avoid storing the full calibration set.
    """

    def __init__(self, layer_name: str, max_cov_dim: int = 512) -> None:
        self.layer_name = layer_name
        self.max_cov_dim = max_cov_dim  # skip covariance if d > this threshold

        self._n: int = 0
        self._mean: Optional[np.ndarray] = None   # float64 (d,)
        self._M2: Optional[np.ndarray] = None     # Welford's sum of squared deviations
        self._C: Optional[np.ndarray] = None      # Cross-product accumulator for cov
        self._int8_min: int = 127
        self._int8_max: int = -128

    def update(self, activation: np.ndarray) -> None:
        """
        Feed one INT8 activation sample (any shape; will be flattened).

        Args:
            activation: INT8 numpy array — shape (C,) for per-channel stats,
                        or any shape (will be flattened to (C,)).
        """
        x = activation.flatten().astype(np.float64)
        d = x.shape[0]

        # Track INT8 range
        self._int8_min = min(self._int8_min, int(activation.min()))
        self._int8_max = max(self._int8_max, int(activation.max()))

        self._n += 1
        if self._mean is None:
            self._mean = np.zeros(d, dtype=np.float64)
            self._M2 = np.zeros(d, dtype=np.float64)
            if d <= self.max_cov_dim:
                self._C = np.zeros((d, d), dtype=np.float64)

        # Welford's online update
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

        # Cross-product for covariance
        if self._C is not None:
            # Incremental covariance: (n-1)/n * outer(delta, delta2)
            self._C += ((self._n - 1) / self._n) * np.outer(delta, delta2)

    def finalize(self) -> LayerStats:
        """
        Compute final statistics and return a LayerStats object.

        Requires at least 2 samples to estimate variance.
        """
        if self._n < 2 or self._mean is None:
            raise RuntimeError(
                f"Layer '{self.layer_name}': need ≥ 2 samples, got {self._n}."
            )

        variance = self._M2 / (self._n - 1)
        std = np.sqrt(np.maximum(variance, 1e-6))

        cov: Optional[np.ndarray] = None
        cov_inv: Optional[np.ndarray] = None
        if self._C is not None:
            cov = self._C / (self._n - 1)
            # Add small diagonal regularization for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-4
            try:
                cov_inv = pinvh(cov)
            except Exception:
                cov_inv = None

        return LayerStats(
            layer_name=self.layer_name,
            mean=self._mean.copy(),
            std=std,
            cov=cov,
            cov_inv=cov_inv,
            n_samples=self._n,
            int8_range=(self._int8_min, self._int8_max),
        )


# ---------------------------------------------------------------------------
# Anomaly scoring functions
# ---------------------------------------------------------------------------

def zscore_max(activation: np.ndarray, stats: LayerStats) -> float:
    """
    Maximum absolute z-score across all channels.

    z_i = |x_i - mu_i| / sigma_i
    score = max_i(z_i)

    Fast and interpretable; threshold ~3-5 for clean INT8 activations.
    """
    x = activation.flatten().astype(np.float64)
    z = np.abs(x - stats.mean) / np.maximum(stats.std, 1e-6)
    return float(z.max())


def mahalanobis_distance(activation: np.ndarray, stats: LayerStats) -> float:
    """
    Mahalanobis distance from the calibration distribution.

    D_M(x) = sqrt( (x - mu)^T Sigma^{-1} (x - mu) )

    Falls back to max z-score if the covariance inverse is unavailable
    (e.g., when d > max_cov_dim).

    Reference: Lee et al., "A Simple Unified Framework for Detecting
    Out-of-Distribution Samples …", NeurIPS 2018.
    """
    if stats.cov_inv is None:
        return zscore_max(activation, stats)

    x = activation.flatten().astype(np.float64)
    delta = x - stats.mean
    maha_sq = float(delta @ stats.cov_inv @ delta)
    return math.sqrt(max(maha_sq, 0.0))


def combined_score(
    activation: np.ndarray,
    stats: LayerStats,
    alpha: float = 0.5,
) -> dict:
    """
    Return both scores together with a weighted combination.

    Args:
        activation: INT8 activation array.
        stats:      Calibration statistics for this layer.
        alpha:      Weight for Mahalanobis (1-alpha for z-score) in combination.

    Returns:
        dict with keys: zscore, mahalanobis, combined, layer_name.
    """
    z = zscore_max(activation, stats)
    m = mahalanobis_distance(activation, stats)
    return {
        "layer_name": stats.layer_name,
        "zscore": z,
        "mahalanobis": m,
        "combined": alpha * m + (1 - alpha) * z,
    }
