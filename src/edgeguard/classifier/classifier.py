"""
ThreatClassifier — combines Merkle attestation + activation anomaly signals
to classify the threat type and produce a layerwise explanation.

Threat taxonomy (from cs24m513.pdf §2):
  CLEAN              — no anomaly detected
  ADVERSARIAL_INPUT  — input manipulation; weights intact; activations anomalous early
  WEIGHT_TAMPER      — Rowhammer / BFA; Merkle hash mismatch; activations deviate post-tamper
  TRANSIENT_FAULT    — hardware glitch; no persistent hash mismatch; activation spike isolated

Decision logic:
  1. hash_mismatch  → WEIGHT_TAMPER  (definitive: persistent weight change)
  2. act_anomaly only, early layers  → ADVERSARIAL_INPUT
  3. act_anomaly only, late layers   → lean TRANSIENT_FAULT (no persistent memory change)
  4. act_anomaly spanning all layers → ambiguous; scored by layer distribution
  5. neither        → CLEAN

The classifier also generates an *explanation* derived from the same per-layer
statistics that triggered detection, ensuring the explanation is always consistent
with the detection decision (Deshpande MTP §3.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Sequence

from ..attestation.manager import AttestationResult
from ..detection.monitor import MonitorResult
from .rules import ThreatRules


class ThreatType(Enum):
    CLEAN = auto()
    ADVERSARIAL_INPUT = auto()
    WEIGHT_TAMPER = auto()
    TRANSIENT_FAULT = auto()
    UNKNOWN = auto()


@dataclass
class ThreatResult:
    """
    Full output of the ThreatClassifier.

    Attributes:
        threat_type:       Classified threat category.
        confidence:        [0, 1] confidence in the classification.
        tampered_layers:   Layer names with Merkle hash mismatch ([] = clean).
        anomalous_layers:  Layer names with activation anomaly.
        explanation:       Human-readable, per-layer explanation text.
        attestation:       Raw AttestationResult (may be None if skipped).
        monitor:           Raw MonitorResult.
        severity:          "none" | "low" | "medium" | "high" | "critical"
    """
    threat_type: ThreatType
    confidence: float
    tampered_layers: list[str] = field(default_factory=list)
    anomalous_layers: list[str] = field(default_factory=list)
    explanation: str = ""
    attestation: Optional[AttestationResult] = None
    monitor: Optional[MonitorResult] = None
    severity: str = "none"

    def is_threat(self) -> bool:
        return self.threat_type != ThreatType.CLEAN

    def summary(self) -> str:
        return (
            f"[{self.threat_type.name}] confidence={self.confidence:.2f} "
            f"severity={self.severity} | {self.explanation[:120]}"
        )


class ThreatClassifier:
    """
    Fuses Merkle attestation results and activation monitor results into
    a ThreatResult with layerwise explanation.

    Parameters:
        layer_names:        Ordered list of all model layer names (early → late).
        early_fraction:     Fraction of layers considered "early" (default 0.4).
        rules:              ThreatRules instance for decision thresholds.
    """

    def __init__(
        self,
        layer_names: Sequence[str],
        early_fraction: float = 0.4,
        rules: Optional[ThreatRules] = None,
    ) -> None:
        self._layer_names = list(layer_names)
        self._n_layers = len(layer_names)
        self._early_cutoff = max(1, int(self._n_layers * early_fraction))
        self._early_set = set(layer_names[: self._early_cutoff])
        self._rules = rules or ThreatRules()

    # ------------------------------------------------------------------
    # Main classification entry point
    # ------------------------------------------------------------------

    def classify(
        self,
        monitor_result: MonitorResult,
        attestation_result: Optional[AttestationResult] = None,
    ) -> ThreatResult:
        """
        Classify the threat and generate a layerwise explanation.

        Args:
            monitor_result:     Activation anomaly scores from LayerwiseActivationMonitor.
            attestation_result: Merkle hash verification result (may be None if only
                                activation-based detection is desired).

        Returns:
            ThreatResult with threat type, confidence, and explanation.
        """
        hash_mismatch = (
            attestation_result is not None
            and not attestation_result.is_valid
        )
        tampered_layers = (
            attestation_result.tampered_layer_names
            if hash_mismatch and attestation_result
            else []
        )
        act_anomaly = monitor_result.overall_anomaly
        anomalous_layers = monitor_result.anomalous_layers

        # --- Rule 1: Merkle hash mismatch → weight tamper ---
        if hash_mismatch:
            confidence = self._rules.weight_tamper_confidence(
                n_tampered=len(tampered_layers),
                n_total=self._n_layers,
                act_anomaly=act_anomaly,
            )
            explanation = self._build_explanation(
                threat=ThreatType.WEIGHT_TAMPER,
                tampered_layers=tampered_layers,
                anomalous_layers=anomalous_layers,
                monitor_result=monitor_result,
                attestation_result=attestation_result,
            )
            return ThreatResult(
                threat_type=ThreatType.WEIGHT_TAMPER,
                confidence=confidence,
                tampered_layers=tampered_layers,
                anomalous_layers=anomalous_layers,
                explanation=explanation,
                attestation=attestation_result,
                monitor=monitor_result,
                severity=self._severity(ThreatType.WEIGHT_TAMPER, confidence),
            )

        # --- Rule 2: Activation anomaly only ---
        if act_anomaly:
            threat_type, confidence = self._classify_activation_anomaly(
                anomalous_layers=anomalous_layers,
                monitor_result=monitor_result,
            )
            explanation = self._build_explanation(
                threat=threat_type,
                tampered_layers=[],
                anomalous_layers=anomalous_layers,
                monitor_result=monitor_result,
                attestation_result=attestation_result,
            )
            return ThreatResult(
                threat_type=threat_type,
                confidence=confidence,
                tampered_layers=[],
                anomalous_layers=anomalous_layers,
                explanation=explanation,
                attestation=attestation_result,
                monitor=monitor_result,
                severity=self._severity(threat_type, confidence),
            )

        # --- Clean ---
        return ThreatResult(
            threat_type=ThreatType.CLEAN,
            confidence=1.0,
            tampered_layers=[],
            anomalous_layers=[],
            explanation="All layers within calibration bounds. Merkle root verified.",
            attestation=attestation_result,
            monitor=monitor_result,
            severity="none",
        )

    # ------------------------------------------------------------------
    # Activation-only sub-classifier
    # ------------------------------------------------------------------

    def _classify_activation_anomaly(
        self,
        anomalous_layers: list[str],
        monitor_result: MonitorResult,
    ) -> tuple[ThreatType, float]:
        """
        When only activation anomaly is present (no hash mismatch),
        distinguish ADVERSARIAL_INPUT from TRANSIENT_FAULT.
        """
        anom_set = set(anomalous_layers)
        early_anom = anom_set & self._early_set
        late_anom = anom_set - self._early_set

        early_ratio = len(early_anom) / max(len(anom_set), 1)

        if early_ratio >= self._rules.early_layer_adv_threshold:
            # Mostly early layers → adversarial input manipulates feature extraction
            confidence = self._rules.adversarial_confidence(
                early_ratio=early_ratio,
                max_zscore=monitor_result.max_zscore,
            )
            return ThreatType.ADVERSARIAL_INPUT, confidence
        else:
            # Mostly late layers → transient fault (hardware glitch, bit flip in SRAM)
            confidence = self._rules.transient_fault_confidence(
                early_ratio=early_ratio,
                max_zscore=monitor_result.max_zscore,
            )
            return ThreatType.TRANSIENT_FAULT, confidence

    # ------------------------------------------------------------------
    # Explanation builder
    # ------------------------------------------------------------------

    def _build_explanation(
        self,
        threat: ThreatType,
        tampered_layers: list[str],
        anomalous_layers: list[str],
        monitor_result: MonitorResult,
        attestation_result: Optional[AttestationResult],
    ) -> str:
        lines = [f"THREAT: {threat.name}"]

        if tampered_layers:
            lines.append(f"  Merkle hash mismatch in {len(tampered_layers)} layer(s):")
            for name in tampered_layers:
                lines.append(f"    • {name} — weight hash CHANGED")

        if anomalous_layers:
            lines.append(f"  Activation anomaly in {len(anomalous_layers)} layer(s):")
            for ls in monitor_result.layer_scores:
                if ls.is_anomalous:
                    pos = "early" if ls.layer_name in self._early_set else "late"
                    lines.append(
                        f"    • {ls.layer_name} [{pos}] — "
                        f"z={ls.zscore:.2f}, M={ls.mahalanobis:.2f}"
                    )

        if not tampered_layers and not anomalous_layers:
            lines.append("  No anomalies detected.")

        if attestation_result:
            sig_ok = "✓" if attestation_result.signature_valid else "✗"
            lines.append(f"  Merkle root signature: {sig_ok}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Severity mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _severity(threat: ThreatType, confidence: float) -> str:
        if threat == ThreatType.CLEAN:
            return "none"
        if threat == ThreatType.WEIGHT_TAMPER:
            return "critical" if confidence > 0.8 else "high"
        if threat == ThreatType.ADVERSARIAL_INPUT:
            return "high" if confidence > 0.7 else "medium"
        if threat == ThreatType.TRANSIENT_FAULT:
            return "medium" if confidence > 0.6 else "low"
        return "low"
