"""Tests for the ThreatClassifier and decision rules."""

import numpy as np
import pytest

from edgeguard.classifier.classifier import ThreatClassifier, ThreatResult, ThreatType
from edgeguard.classifier.rules import ThreatRules
from edgeguard.attestation.manager import AttestationResult
from edgeguard.detection.monitor import MonitorResult, LayerScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_layer_names(n: int = 8) -> list[str]:
    return [f"layer{i}" for i in range(n)]


def clean_attestation(n: int = 8) -> AttestationResult:
    return AttestationResult(
        is_valid=True,
        tampered_layers=[],
        tampered_layer_names=[],
        signature_valid=True,
        num_layers=n,
    )


def tampered_attestation(tampered_indices: list[int], layer_names: list[str]) -> AttestationResult:
    names = [layer_names[i] for i in tampered_indices]
    return AttestationResult(
        is_valid=False,
        tampered_layers=tampered_indices,
        tampered_layer_names=names,
        signature_valid=True,
        num_layers=len(layer_names),
    )


def clean_monitor(layer_names: list[str]) -> MonitorResult:
    scores = [
        LayerScore(name, zscore=0.5, mahalanobis=1.0, combined=0.75, is_anomalous=False)
        for name in layer_names
    ]
    return MonitorResult(
        layer_scores=scores,
        anomalous_layers=[],
        overall_anomaly=False,
        max_zscore=0.5,
        max_mahalanobis=1.0,
    )


def anomalous_monitor(anomalous_names: list[str], all_names: list[str], zscore: float = 8.0) -> MonitorResult:
    scores = []
    for name in all_names:
        is_anom = name in anomalous_names
        scores.append(LayerScore(
            name,
            zscore=zscore if is_anom else 0.5,
            mahalanobis=30.0 if is_anom else 1.0,
            combined=zscore if is_anom else 0.75,
            is_anomalous=is_anom,
        ))
    return MonitorResult(
        layer_scores=scores,
        anomalous_layers=anomalous_names,
        overall_anomaly=len(anomalous_names) > 0,
        max_zscore=zscore if anomalous_names else 0.5,
        max_mahalanobis=30.0 if anomalous_names else 1.0,
    )


# ---------------------------------------------------------------------------
# ThreatClassifier tests
# ---------------------------------------------------------------------------

class TestThreatClassifier:

    def setup_method(self):
        self.layer_names = make_layer_names(8)
        self.clf = ThreatClassifier(self.layer_names, early_fraction=0.4)

    # --- CLEAN ---

    def test_clean_no_attestation(self):
        monitor = clean_monitor(self.layer_names)
        result = self.clf.classify(monitor, attestation_result=None)
        assert result.threat_type == ThreatType.CLEAN
        assert result.confidence == 1.0
        assert result.severity == "none"

    def test_clean_with_attestation(self):
        monitor = clean_monitor(self.layer_names)
        attest = clean_attestation(len(self.layer_names))
        result = self.clf.classify(monitor, attest)
        assert result.threat_type == ThreatType.CLEAN

    # --- WEIGHT TAMPER ---

    def test_weight_tamper_hash_mismatch(self):
        monitor = clean_monitor(self.layer_names)
        attest = tampered_attestation([2, 5], self.layer_names)
        result = self.clf.classify(monitor, attest)
        assert result.threat_type == ThreatType.WEIGHT_TAMPER
        assert "layer2" in result.tampered_layers
        assert "layer5" in result.tampered_layers
        assert result.severity in ("high", "critical")

    def test_weight_tamper_confidence_increases_with_more_layers(self):
        monitor = clean_monitor(self.layer_names)
        r1 = self.clf.classify(monitor, tampered_attestation([0], self.layer_names))
        r2 = self.clf.classify(monitor, tampered_attestation([0, 1, 2, 3], self.layer_names))
        assert r2.confidence >= r1.confidence

    def test_weight_tamper_explanation_contains_tampered_layer(self):
        monitor = clean_monitor(self.layer_names)
        attest = tampered_attestation([3], self.layer_names)
        result = self.clf.classify(monitor, attest)
        assert "layer3" in result.explanation

    # --- ADVERSARIAL INPUT ---

    def test_adversarial_input_early_layers(self):
        # Anomaly in first 3 layers (early) → adversarial
        early = self.layer_names[:3]
        monitor = anomalous_monitor(early, self.layer_names, zscore=10.0)
        attest = clean_attestation(len(self.layer_names))
        result = self.clf.classify(monitor, attest)
        assert result.threat_type == ThreatType.ADVERSARIAL_INPUT

    # --- TRANSIENT FAULT ---

    def test_transient_fault_late_layers(self):
        # Anomaly only in last 2 layers → transient fault
        late = self.layer_names[6:]
        monitor = anomalous_monitor(late, self.layer_names, zscore=5.0)
        attest = clean_attestation(len(self.layer_names))
        result = self.clf.classify(monitor, attest)
        assert result.threat_type == ThreatType.TRANSIENT_FAULT

    # --- Priority: hash mismatch takes precedence ---

    def test_weight_tamper_overrides_activation_anomaly(self):
        # Both hash mismatch AND activation anomaly → WEIGHT_TAMPER wins
        early = self.layer_names[:3]
        monitor = anomalous_monitor(early, self.layer_names)
        attest = tampered_attestation([6], self.layer_names)
        result = self.clf.classify(monitor, attest)
        assert result.threat_type == ThreatType.WEIGHT_TAMPER

    # --- ThreatResult helpers ---

    def test_is_threat_clean(self):
        monitor = clean_monitor(self.layer_names)
        result = self.clf.classify(monitor)
        assert result.is_threat() is False

    def test_is_threat_tamper(self):
        monitor = clean_monitor(self.layer_names)
        attest = tampered_attestation([0], self.layer_names)
        result = self.clf.classify(monitor, attest)
        assert result.is_threat() is True

    def test_summary_contains_threat_type(self):
        monitor = clean_monitor(self.layer_names)
        attest = tampered_attestation([1], self.layer_names)
        result = self.clf.classify(monitor, attest)
        assert "WEIGHT_TAMPER" in result.summary()


# ---------------------------------------------------------------------------
# ThreatRules tests
# ---------------------------------------------------------------------------

class TestThreatRules:

    def test_weight_tamper_base_confidence(self):
        rules = ThreatRules()
        c = rules.weight_tamper_confidence(1, 8, False)
        assert 0.85 <= c <= 1.0

    def test_adversarial_confidence_increases_with_early_ratio(self):
        rules = ThreatRules()
        c_low = rules.adversarial_confidence(0.0, 5.0)
        c_high = rules.adversarial_confidence(1.0, 5.0)
        assert c_high > c_low

    def test_transient_fault_confidence_decreases_with_early_ratio(self):
        rules = ThreatRules()
        c_low = rules.transient_fault_confidence(0.0, 5.0)
        c_high = rules.transient_fault_confidence(1.0, 5.0)
        assert c_low > c_high

    def test_permissive_and_strict_constructors(self):
        p = ThreatRules.permissive()
        s = ThreatRules.strict()
        assert p.early_layer_adv_threshold > s.early_layer_adv_threshold
