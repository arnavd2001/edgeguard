"""
LayerwiseActivationMonitor — per-inference anomaly detection.

At each inference call, the monitor:
  1. Receives per-layer INT8 activations from the instrumentation hooks.
  2. Computes z-score and Mahalanobis scores against calibration statistics.
  3. Returns a MonitorResult with per-layer scores and an overall anomaly flag.

The result is then consumed by the ThreatClassifier which combines it with the
MerkleAttestationManager's hash verdict to identify the threat type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .statistics import LayerStats, zscore_max, mahalanobis_distance, combined_score
from .calibration import Calibrator


@dataclass
class LayerScore:
    """Anomaly scores for a single layer on one inference call."""
    layer_name: str
    zscore: float
    mahalanobis: float
    combined: float
    is_anomalous: bool          # True if any score exceeds its threshold


@dataclass
class MonitorResult:
    """
    Per-inference output from LayerwiseActivationMonitor.

    Attributes:
        layer_scores:       One LayerScore per monitored layer.
        anomalous_layers:   Names of layers flagged as anomalous.
        overall_anomaly:    True if any layer is anomalous.
        max_zscore:         Worst z-score across all layers.
        max_mahalanobis:    Worst Mahalanobis distance across all layers.
        explanation:        Human-readable per-layer explanation string.
    """
    layer_scores: list[LayerScore] = field(default_factory=list)
    anomalous_layers: list[str] = field(default_factory=list)
    overall_anomaly: bool = False
    max_zscore: float = 0.0
    max_mahalanobis: float = 0.0
    explanation: str = ""

    def summary(self) -> str:
        if not self.overall_anomaly:
            return (
                f"CLEAN — {len(self.layer_scores)} layers monitored, "
                f"max z={self.max_zscore:.2f}, max M={self.max_mahalanobis:.2f}"
            )
        return (
            f"ANOMALY — layers: {self.anomalous_layers} | "
            f"max z={self.max_zscore:.2f}, max M={self.max_mahalanobis:.2f}"
        )


class LayerwiseActivationMonitor:
    """
    Runs per-layer activation anomaly detection using calibrated INT8 statistics.

    Parameters:
        calibration_stats:  dict[layer_name → LayerStats] from Calibrator.load().
        zscore_threshold:   Flag a layer if max z-score exceeds this (default 4.0).
        mahal_threshold:    Flag a layer if Mahalanobis distance exceeds this.
                            Default 25.0 ≈ chi2(d) 99.9th percentile for d≈16.
        alpha:              Mahalanobis weight in combined score (default 0.5).
        require_all_layers: If True, raise ValueError when a layer is missing.
    """

    def __init__(
        self,
        calibration_stats: dict[str, LayerStats],
        zscore_threshold: float = 4.0,
        mahal_threshold: float = 25.0,
        alpha: float = 0.5,
        require_all_layers: bool = True,
    ) -> None:
        self._stats = calibration_stats
        self._zscore_thresh = zscore_threshold
        self._mahal_thresh = mahal_threshold
        self._alpha = alpha
        self._require_all = require_all_layers

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration_file(
        cls,
        path: str | Path,
        **kwargs,
    ) -> "LayerwiseActivationMonitor":
        """Load calibration from disk and instantiate monitor."""
        stats = Calibrator.load(path)
        return cls(calibration_stats=stats, **kwargs)

    # ------------------------------------------------------------------
    # Core inference-time call
    # ------------------------------------------------------------------

    def inspect(self, layer_activations: dict[str, np.ndarray]) -> MonitorResult:
        """
        Score one inference pass.

        Args:
            layer_activations: dict[layer_name → INT8 ndarray], one per layer.

        Returns:
            MonitorResult with anomaly flags, per-layer scores, and explanation.
        """
        layer_scores: list[LayerScore] = []
        anomalous_layers: list[str] = []
        explanation_lines: list[str] = []

        for name, stats in self._stats.items():
            if name not in layer_activations:
                if self._require_all:
                    raise KeyError(f"Layer '{name}' missing from activations dict.")
                continue

            act = layer_activations[name]
            scores = combined_score(act, stats, alpha=self._alpha)

            z = scores["zscore"]
            m = scores["mahalanobis"]
            c = scores["combined"]
            is_anom = z > self._zscore_thresh or m > self._mahal_thresh

            ls = LayerScore(
                layer_name=name,
                zscore=z,
                mahalanobis=m,
                combined=c,
                is_anomalous=is_anom,
            )
            layer_scores.append(ls)

            if is_anom:
                anomalous_layers.append(name)
                explanation_lines.append(
                    f"  [{name}] ANOMALOUS — z={z:.2f} (thresh={self._zscore_thresh}), "
                    f"M={m:.2f} (thresh={self._mahal_thresh})"
                )
            else:
                explanation_lines.append(
                    f"  [{name}] clean     — z={z:.2f}, M={m:.2f}"
                )

        overall_anomaly = len(anomalous_layers) > 0
        max_z = max((ls.zscore for ls in layer_scores), default=0.0)
        max_m = max((ls.mahalanobis for ls in layer_scores), default=0.0)

        explanation = "\n".join(explanation_lines)

        return MonitorResult(
            layer_scores=layer_scores,
            anomalous_layers=anomalous_layers,
            overall_anomaly=overall_anomaly,
            max_zscore=max_z,
            max_mahalanobis=max_m,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Threshold tuning helpers
    # ------------------------------------------------------------------

    def tune_thresholds(
        self,
        clean_results: Sequence[MonitorResult],
        fpr_target: float = 0.001,
    ) -> tuple[float, float]:
        """
        Auto-tune zscore and Mahalanobis thresholds on clean validation results.

        Sets thresholds at the (1 - fpr_target) quantile of the clean score
        distribution, targeting a false positive rate of fpr_target on clean data.

        Args:
            clean_results: MonitorResult objects from clean (benign) samples.
            fpr_target:    Target false positive rate (default 0.1 %).

        Returns:
            (new_zscore_threshold, new_mahal_threshold) — also updates self.
        """
        if not clean_results:
            raise ValueError("No clean results provided for threshold tuning.")

        all_z = [r.max_zscore for r in clean_results]
        all_m = [r.max_mahalanobis for r in clean_results]

        import numpy as np
        q = (1.0 - fpr_target) * 100.0
        new_z = float(np.percentile(all_z, q))
        new_m = float(np.percentile(all_m, q))

        self._zscore_thresh = new_z
        self._mahal_thresh = new_m
        return new_z, new_m

    @property
    def layer_names(self) -> list[str]:
        return list(self._stats.keys())
