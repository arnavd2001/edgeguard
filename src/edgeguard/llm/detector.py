"""
LLMThreatDetector — EdgeGuard unified detection pipeline adapted for LLMs.

Extends the base ThreatClassifier with LLM-specific threat types:
  - TRAINING_DATA_EXFILTRATION: activation-inversion probing detected
  - PROMPT_INJECTION:            system prompt override attempt
  - WEIGHT_THEFT:               weight extraction / cold-boot read
  - JAILBREAK:                  safety bypass adversarial input

The detector computes an additional "memorization signature" score alongside
the standard Mahalanobis / z-score metrics, to catch the distinctive pattern
of training-data-extraction probes (low entropy in late layers, Scenario 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from ..attestation.manager import MerkleAttestationManager, AttestationResult
from ..detection.monitor import LayerwiseActivationMonitor, MonitorResult
from ..classifier.classifier import ThreatClassifier, ThreatType, ThreatResult
from ..classifier.rules import ThreatRules


class LLMThreatType(Enum):
    CLEAN                       = auto()
    ADVERSARIAL_INPUT           = auto()
    WEIGHT_TAMPER               = auto()
    TRANSIENT_FAULT             = auto()
    TRAINING_DATA_EXFILTRATION  = auto()
    PROMPT_INJECTION            = auto()
    WEIGHT_THEFT_PROBE          = auto()
    JAILBREAK                   = auto()
    SUPPLY_CHAIN_BACKDOOR       = auto()
    UNKNOWN                     = auto()


@dataclass
class LLMThreatResult:
    """Extended threat result for LLM-specific threats."""
    threat_type: LLMThreatType
    confidence: float
    base_result: ThreatResult
    memorization_score: float    # 0–1: likelihood of training-data-exfil probe
    entropy_score: float         # Late-layer activation entropy (low = suspicious)
    tampered_layers: list[str] = field(default_factory=list)
    anomalous_layers: list[str] = field(default_factory=list)
    explanation: str = ""
    severity: str = "none"

    def is_threat(self) -> bool:
        return self.threat_type != LLMThreatType.CLEAN

    def summary(self) -> str:
        return (
            f"[LLM | {self.threat_type.name}] "
            f"conf={self.confidence:.2f} | "
            f"mem_score={self.memorization_score:.2f} | "
            f"entropy={self.entropy_score:.2f} | "
            f"severity={self.severity}"
        )


class LLMThreatDetector:
    """
    Unified threat detector for quantized LLMs on edge devices.

    Adds three detection layers on top of the base EdgeGuard stack:
      1. Memorization signature detection (training data exfiltration)
      2. Late-layer entropy collapse detection (prompt injection / jailbreak)
      3. Supply-chain backdoor detection (weight distribution statistics)

    Parameters:
        layer_names:         All LLM layer names, in order.
        attestation_manager: Pre-built MerkleAttestationManager.
        activation_monitor:  Calibrated LayerwiseActivationMonitor.
        early_fraction:      Fraction of layers treated as "early" (default 0.4).
        mem_score_threshold: Memorization score above which we flag exfiltration.
        entropy_threshold:   Late-layer entropy below which we flag jailbreak/injection.
    """

    def __init__(
        self,
        layer_names: list[str],
        attestation_manager: Optional[MerkleAttestationManager] = None,
        activation_monitor: Optional[LayerwiseActivationMonitor] = None,
        early_fraction: float = 0.4,
        mem_score_threshold: float = 0.60,
        entropy_threshold: float = 0.25,
    ) -> None:
        self._layer_names = list(layer_names)
        self._n_layers = len(layer_names)
        self._early_cutoff = max(1, int(self._n_layers * early_fraction))
        self._late_layers = set(layer_names[self._early_cutoff:])
        self._mgr = attestation_manager
        self._monitor = activation_monitor
        self._base_clf = ThreatClassifier(
            layer_names, early_fraction=early_fraction, rules=ThreatRules()
        )
        self.mem_score_threshold = mem_score_threshold
        self.entropy_threshold = entropy_threshold

    # ------------------------------------------------------------------
    # Main detection call
    # ------------------------------------------------------------------

    def detect(
        self,
        activations: dict[str, np.ndarray],
        weights: Optional[dict[str, np.ndarray]] = None,
    ) -> LLMThreatResult:
        """
        Run the full LLM threat detection pipeline.

        Args:
            activations: Per-layer INT8 activations from the current inference.
            weights:     Current weight tensors for Merkle verification.
                         If None, skips attestation check.

        Returns:
            LLMThreatResult with LLM-specific threat classification.
        """
        # --- Attestation ---
        attest_result: Optional[AttestationResult] = None
        if self._mgr is not None and weights is not None:
            attest_result = self._mgr.verify(weights)

        # --- Activation monitor ---
        monitor_result: Optional[MonitorResult] = None
        if self._monitor is not None:
            monitor_result = self._monitor.inspect(activations)

        # --- Base threat classification ---
        if monitor_result is not None:
            base = self._base_clf.classify(monitor_result, attest_result)
        else:
            # Build a minimal MonitorResult if no monitor is configured
            from ..detection.monitor import MonitorResult, LayerScore
            base_scores = []
            base = ThreatResult(
                threat_type=ThreatType.CLEAN,
                confidence=0.5,
                explanation="No monitor configured.",
            )

        # --- LLM-specific scoring ---
        mem_score = self._memorization_score(activations)
        entropy = self._late_layer_entropy(activations)

        # --- LLM threat classification ---
        llm_type, llm_conf = self._classify_llm(
            base=base,
            mem_score=mem_score,
            entropy=entropy,
            attest_result=attest_result,
        )

        explanation = self._build_explanation(
            llm_type, base, mem_score, entropy, attest_result
        )

        return LLMThreatResult(
            threat_type=llm_type,
            confidence=llm_conf,
            base_result=base,
            memorization_score=mem_score,
            entropy_score=entropy,
            tampered_layers=base.tampered_layers,
            anomalous_layers=base.anomalous_layers,
            explanation=explanation,
            severity=self._severity(llm_type, llm_conf),
        )

    # ------------------------------------------------------------------
    # LLM-specific scoring
    # ------------------------------------------------------------------

    def _memorization_score(self, activations: dict[str, np.ndarray]) -> float:
        """
        Compute a memorization-probe signature score.

        High score = late-layer activations have very high mean + very low std,
        consistent with the model retrieving a memorized sequence (Carlini et al., 2021).

        Returns a score in [0, 1].
        """
        if not activations:
            return 0.0

        late_scores = []
        for name in self._late_layers:
            if name not in activations:
                continue
            act = activations[name].astype(np.float32)
            mean_abs = float(np.abs(act).mean())
            std = float(act.std())
            # High mean + low std → high memorization score
            if std < 1e-6:
                score = 1.0
            else:
                score = min(1.0, (mean_abs / 50.0) * (1.0 / (std / 5.0 + 1.0)))
            late_scores.append(score)

        return float(np.mean(late_scores)) if late_scores else 0.0

    def _late_layer_entropy(self, activations: dict[str, np.ndarray]) -> float:
        """
        Compute normalized entropy of late-layer activations.

        Low entropy (near 0) indicates highly concentrated activations —
        a signature of safety bypass, jailbreak, or prompt injection attempts.
        Normal inference has entropy close to 1.0.
        """
        late_entropies = []
        for name in self._late_layers:
            if name not in activations:
                continue
            act = activations[name].astype(np.float32)
            # Normalize to [0, 1] probability distribution via softmax
            shifted = act - act.max()
            probs = np.exp(shifted / max(float(np.abs(shifted).mean()), 1.0))
            probs /= probs.sum()
            # Shannon entropy normalized by log(d)
            d = len(probs)
            ent = float(-np.sum(probs * np.log(probs + 1e-12)))
            max_ent = float(np.log(d)) if d > 1 else 1.0
            late_entropies.append(ent / max_ent)

        return float(np.mean(late_entropies)) if late_entropies else 1.0

    # ------------------------------------------------------------------
    # LLM threat classification logic
    # ------------------------------------------------------------------

    def _classify_llm(
        self,
        base: ThreatResult,
        mem_score: float,
        entropy: float,
        attest_result: Optional[AttestationResult],
    ) -> tuple[LLMThreatType, float]:
        """Map base ThreatType + LLM scores → LLMThreatType."""

        # Priority 1: Merkle hash mismatch
        if base.threat_type == ThreatType.WEIGHT_TAMPER:
            # Check if it looks like a supply-chain backdoor (many layers affected)
            n_tampered = len(base.tampered_layers)
            if n_tampered > self._n_layers * 0.3:
                return LLMThreatType.SUPPLY_CHAIN_BACKDOOR, min(0.95, base.confidence + 0.05)
            return LLMThreatType.WEIGHT_TAMPER, base.confidence

        # Priority 2: Training data exfiltration (memorization probe)
        if mem_score > self.mem_score_threshold:
            conf = min(0.95, 0.60 + mem_score * 0.35)
            return LLMThreatType.TRAINING_DATA_EXFILTRATION, conf

        # Priority 3: Late-layer entropy collapse → jailbreak / prompt injection
        if entropy < self.entropy_threshold:
            if base.threat_type == ThreatType.ADVERSARIAL_INPUT:
                return LLMThreatType.JAILBREAK, min(0.90, base.confidence + 0.10)
            return LLMThreatType.PROMPT_INJECTION, 0.72

        # Priority 4: Adversarial input (base classification)
        if base.threat_type == ThreatType.ADVERSARIAL_INPUT:
            return LLMThreatType.ADVERSARIAL_INPUT, base.confidence

        # Priority 5: Transient fault
        if base.threat_type == ThreatType.TRANSIENT_FAULT:
            return LLMThreatType.TRANSIENT_FAULT, base.confidence

        return LLMThreatType.CLEAN, 1.0

    # ------------------------------------------------------------------
    # Explanation and severity
    # ------------------------------------------------------------------

    def _build_explanation(
        self,
        threat: LLMThreatType,
        base: ThreatResult,
        mem_score: float,
        entropy: float,
        attest: Optional[AttestationResult],
    ) -> str:
        lines = [f"LLM THREAT: {threat.name}"]
        lines.append(f"  Base classification: {base.threat_type.name} (conf={base.confidence:.2f})")
        lines.append(f"  Memorization probe score: {mem_score:.3f} (threshold={self.mem_score_threshold})")
        lines.append(f"  Late-layer entropy: {entropy:.3f} (threshold={self.entropy_threshold})")
        if base.tampered_layers:
            lines.append(f"  Tampered layers: {base.tampered_layers}")
        if base.anomalous_layers:
            lines.append(f"  Anomalous layers: {base.anomalous_layers}")
        if attest:
            sig = "✓ valid" if attest.signature_valid else "✗ INVALID"
            lines.append(f"  Merkle root signature: {sig}")
        return "\n".join(lines)

    @staticmethod
    def _severity(threat: LLMThreatType, confidence: float) -> str:
        severity_map = {
            LLMThreatType.CLEAN: "none",
            LLMThreatType.TRANSIENT_FAULT: "low",
            LLMThreatType.ADVERSARIAL_INPUT: "medium",
            LLMThreatType.PROMPT_INJECTION: "medium",
            LLMThreatType.JAILBREAK: "high",
            LLMThreatType.WEIGHT_TAMPER: "critical",
            LLMThreatType.WEIGHT_THEFT_PROBE: "high",
            LLMThreatType.SUPPLY_CHAIN_BACKDOOR: "critical",
            LLMThreatType.TRAINING_DATA_EXFILTRATION: "critical",
            LLMThreatType.UNKNOWN: "medium",
        }
        base_sev = severity_map.get(threat, "low")
        # Downgrade if low confidence
        if confidence < 0.5 and base_sev in ("critical", "high"):
            return "medium"
        return base_sev
