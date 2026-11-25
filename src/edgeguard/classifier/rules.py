"""
Decision rules and confidence scoring for the ThreatClassifier.

All thresholds are configurable to allow tuning on deployment-specific
calibration data without changing the classification logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ThreatRules:
    """
    Thresholds and scoring weights for threat classification.

    Attributes:
        early_layer_adv_threshold: Fraction of anomalous layers that must be
            in the "early" set to classify ADVERSARIAL_INPUT (default 0.5).
            Layers index < early_cutoff are "early".

        weight_tamper_base_confidence: Base confidence for WEIGHT_TAMPER when
            any hash mismatch is detected.

        adv_zscore_scale: Scales how much z-score magnitude contributes to
            adversarial confidence (higher z → higher confidence).

        transient_fault_base_confidence: Base confidence for TRANSIENT_FAULT
            when activation anomaly is confined to late layers.
    """

    early_layer_adv_threshold: float = 0.5
    weight_tamper_base_confidence: float = 0.90
    adv_zscore_scale: float = 0.02       # per unit of z above threshold
    transient_fault_base_confidence: float = 0.70

    # ------------------------------------------------------------------
    # Weight tamper confidence
    # ------------------------------------------------------------------

    def weight_tamper_confidence(
        self,
        n_tampered: int,
        n_total: int,
        act_anomaly: bool,
    ) -> float:
        """
        Confidence that the threat is WEIGHT_TAMPER.

        Higher confidence when more layers are tampered and when the
        activation anomaly corroborates the hash mismatch.

        Args:
            n_tampered:  Number of layers with hash mismatch.
            n_total:     Total number of model layers.
            act_anomaly: Whether the activation monitor also flagged anomaly.
        """
        if n_total == 0:
            return self.weight_tamper_base_confidence
        tamper_ratio = n_tampered / n_total
        # Sigmoid-like scaling: single layer → lower; many layers → higher
        ratio_boost = math.tanh(tamper_ratio * 5) * 0.08
        act_boost = 0.02 if act_anomaly else 0.0
        return min(1.0, self.weight_tamper_base_confidence + ratio_boost + act_boost)

    # ------------------------------------------------------------------
    # Adversarial input confidence
    # ------------------------------------------------------------------

    def adversarial_confidence(
        self,
        early_ratio: float,
        max_zscore: float,
    ) -> float:
        """
        Confidence that the threat is ADVERSARIAL_INPUT.

        Args:
            early_ratio:  Fraction of anomalous layers that are early.
            max_zscore:   Worst z-score across all monitored layers.
        """
        # Base: fraction of anomaly in early layers (bounded at 0.85)
        base = min(0.85, 0.50 + early_ratio * 0.35)
        # Boost from z-score magnitude above typical threshold (4.0)
        z_above = max(0.0, max_zscore - 4.0)
        z_boost = min(0.10, z_above * self.adv_zscore_scale)
        return min(0.95, base + z_boost)

    # ------------------------------------------------------------------
    # Transient fault confidence
    # ------------------------------------------------------------------

    def transient_fault_confidence(
        self,
        early_ratio: float,
        max_zscore: float,
    ) -> float:
        """
        Confidence that the threat is TRANSIENT_FAULT.

        High when anomaly is confined to late layers (early_ratio is LOW),
        and when the z-score is moderately elevated (not extreme).

        Args:
            early_ratio:  Fraction of anomalous layers that are early.
            max_zscore:   Worst z-score across all monitored layers.
        """
        # Low early_ratio → higher transient fault confidence
        base = self.transient_fault_base_confidence * (1.0 - early_ratio * 0.5)
        # Extreme z-scores suggest adversarial input, reduce confidence
        if max_zscore > 20.0:
            base *= 0.80
        return max(0.30, min(0.90, base))

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    @classmethod
    def permissive(cls) -> "ThreatRules":
        """Rules tuned for low false-positive rate (miss more subtle threats)."""
        return cls(
            early_layer_adv_threshold=0.6,
            weight_tamper_base_confidence=0.92,
            adv_zscore_scale=0.015,
            transient_fault_base_confidence=0.65,
        )

    @classmethod
    def strict(cls) -> "ThreatRules":
        """Rules tuned for high recall (may have more false positives)."""
        return cls(
            early_layer_adv_threshold=0.4,
            weight_tamper_base_confidence=0.88,
            adv_zscore_scale=0.025,
            transient_fault_base_confidence=0.75,
        )
