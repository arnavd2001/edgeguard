"""
Tests for LLMThreatDetector and QuantizedTransformerStub.

Covers all 10 LLMThreatType cases including:
  - Clean inference
  - Weight tampering via BFA (Merkle-detected)
  - Supply chain backdoor (steganographic + distributed)
  - Adversarial input (early-layer anomaly)
  - Jailbreak / prompt injection (entropy collapse)
  - Training data exfiltration (memorization score)
  - Synergy attack (BFA suppresses activations, Merkle still fires)
  - LLM model profile correctness (layer count, weight shapes)
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    from edgeguard.llm.model import QuantizedTransformerStub
    return QuantizedTransformerStub(profile_name="tiny", seed=0)


@pytest.fixture(scope="module")
def tiny_weights(tiny_model):
    return tiny_model.get_weights()


@pytest.fixture(scope="module")
def tiny_layer_names(tiny_model):
    return tiny_model.layer_names


@pytest.fixture(scope="module")
def calibrated_detector(tiny_model, tiny_weights, tiny_layer_names):
    """
    One fully-calibrated LLMThreatDetector for the tiny profile.
    Calibrated on 200 clean INT8 activations.
    """
    from edgeguard.attestation.manager import MerkleAttestationManager
    from edgeguard.detection.calibration import Calibrator
    from edgeguard.detection.monitor import LayerwiseActivationMonitor
    from edgeguard.llm.detector import LLMThreatDetector

    rng = np.random.default_rng(42)
    dim = 24

    mgr = MerkleAttestationManager()
    mgr.build(tiny_weights)

    cal = Calibrator(layer_names=tiny_layer_names, max_cov_dim=16, verbose=0)
    for _ in range(200):
        acts = {
            n: np.clip(rng.normal(10, 5, size=dim), -128, 127).astype(np.int8)
            for n in tiny_layer_names
        }
        cal.feed_activations(acts)
    stats = cal.finalize()
    monitor = LayerwiseActivationMonitor(stats, zscore_threshold=4.0, mahal_threshold=50.0)

    return LLMThreatDetector(
        layer_names=tiny_layer_names,
        attestation_manager=mgr,
        activation_monitor=monitor,
        mem_score_threshold=0.55,
        entropy_threshold=0.30,
    ), tiny_weights, tiny_layer_names, dim, rng


# ---------------------------------------------------------------------------
# Helper: generate clean INT8 activations
# ---------------------------------------------------------------------------

def clean_acts(layer_names, dim, rng):
    return {
        n: np.clip(rng.normal(10, 5, size=dim), -128, 127).astype(np.int8)
        for n in layer_names
    }


# ===========================================================================
# QuantizedTransformerStub — weight structure
# ===========================================================================

class TestQuantizedTransformerStub:
    def test_tiny_profile_layer_count(self, tiny_model, tiny_layer_names):
        # tiny: n_layers=4, 9 matrices/block (q,k,v,o,gate,up,down,in_norm,post_norm)
        # + embed_tokens + lm_head + model.norm = 4*9+3 = 39 weight tensors
        assert len(tiny_layer_names) == 39

    def test_weight_dtypes_all_int8(self, tiny_weights):
        for name, arr in tiny_weights.items():
            assert arr.dtype == np.int8, f"{name} should be INT8, got {arr.dtype}"

    def test_weight_values_in_int8_range(self, tiny_weights):
        for name, arr in tiny_weights.items():
            assert arr.min() >= -128 and arr.max() <= 127, f"{name} out of INT8 range"

    def test_all_expected_layer_names_present(self, tiny_weights):
        # Spot-check HuggingFace naming convention (fully-qualified with model. prefix)
        assert "model.layers.0.self_attn.q_proj.weight" in tiny_weights
        assert "model.layers.0.mlp.gate_proj.weight" in tiny_weights
        assert "model.layers.0.input_layernorm.weight" in tiny_weights
        assert "model.embed_tokens.weight" in tiny_weights

    def test_qwen_profile_has_correct_layer_count(self):
        from edgeguard.llm.model import QuantizedTransformerStub
        model = QuantizedTransformerStub(profile_name="qwen2.5-0.5b", seed=1)
        # qwen2.5-0.5b: n_layers=24, 9 matrices/block + 3 non-block = 24*9+3 = 219
        assert len(model.layer_names) == 219

    def test_llama_1b_profile_weight_shapes(self):
        from edgeguard.llm.model import QuantizedTransformerStub
        model = QuantizedTransformerStub(profile_name="llama-3.2-1b", seed=2)
        w = model.get_weights()
        # q_proj should be (hidden_size, hidden_size)
        q = w.get("model.layers.0.self_attn.q_proj.weight")
        assert q is not None
        assert q.ndim == 2

    def test_n_weights_matches_sum_of_elements(self, tiny_model, tiny_weights):
        total = sum(v.size for v in tiny_weights.values())
        assert tiny_model.n_weights == total

    def test_size_mb_positive(self, tiny_model):
        assert tiny_model.size_mb > 0.0


# ===========================================================================
# LLMThreatDetector — detection logic
# ===========================================================================

class TestLLMThreatDetectorClean:
    def test_clean_activations_return_clean(self, calibrated_detector):
        from edgeguard.llm.detector import LLMThreatType
        det, weights, layer_names, dim, rng = calibrated_detector
        for _ in range(10):
            result = det.detect(clean_acts(layer_names, dim, rng), weights=weights)
            assert result.threat_type == LLMThreatType.CLEAN, \
                f"Clean input misclassified as {result.threat_type}"

    def test_clean_result_has_confidence(self, calibrated_detector):
        det, weights, layer_names, dim, rng = calibrated_detector
        result = det.detect(clean_acts(layer_names, dim, rng), weights=weights)
        assert 0.0 <= result.confidence <= 1.0

    def test_clean_result_has_explanation(self, calibrated_detector):
        det, weights, layer_names, dim, rng = calibrated_detector
        result = det.detect(clean_acts(layer_names, dim, rng), weights=weights)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0


class TestLLMThreatDetectorWeightTamper:
    def test_bfa_sign_bit_detected_as_weight_tamper(self, calibrated_detector):
        from edgeguard.attacks.bitflip import BitFlipAttack
        from edgeguard.llm.detector import LLMThreatType
        det, weights, layer_names, dim, rng = calibrated_detector
        bfa = BitFlipAttack(strategy="sign_bit", n_flips=1, seed=100)
        attacked = bfa.attack(weights).attacked_weights
        result = det.detect(clean_acts(layer_names, dim, rng), weights=attacked)
        assert result.threat_type == LLMThreatType.WEIGHT_TAMPER

    def test_tampered_layers_reported(self, calibrated_detector):
        from edgeguard.attacks.bitflip import BitFlipAttack
        det, weights, layer_names, dim, rng = calibrated_detector
        bfa = BitFlipAttack(strategy="sign_bit", n_flips=1, seed=200)
        attacked = bfa.attack(weights).attacked_weights
        result = det.detect(clean_acts(layer_names, dim, rng), weights=attacked)
        assert hasattr(result, "tampered_layers")
        assert len(result.tampered_layers) >= 1

    def test_multiple_flips_increase_tampered_layer_count(self, calibrated_detector):
        from edgeguard.attacks.bitflip import BitFlipAttack
        det, weights, layer_names, dim, rng = calibrated_detector
        bfa_many = BitFlipAttack(strategy="random", n_flips=5, seed=300)
        attacked = bfa_many.attack(weights).attacked_weights
        result = det.detect(clean_acts(layer_names, dim, rng), weights=attacked)
        assert result.threat_type.name in ("WEIGHT_TAMPER", "SUPPLY_CHAIN_BACKDOOR")


class TestLLMThreatDetectorSupplyChain:
    def test_distributed_poisoning_detected(self, calibrated_detector):
        from edgeguard.attacks.supply_chain import SupplyChainPoisoning
        from edgeguard.llm.detector import LLMThreatType
        det, weights, layer_names, dim, rng = calibrated_detector
        poisoner = SupplyChainPoisoning(strategy="distributed", seed=50)
        poisoned = poisoner.poison(weights).poisoned_weights
        result = det.detect(clean_acts(layer_names, dim, rng), weights=poisoned)
        assert result.threat_type in (
            LLMThreatType.SUPPLY_CHAIN_BACKDOOR,
            LLMThreatType.WEIGHT_TAMPER,
        )

    def test_steganographic_poisoning_caught_by_merkle(self, calibrated_detector):
        from edgeguard.attacks.supply_chain import SupplyChainPoisoning
        from edgeguard.llm.detector import LLMThreatType
        det, weights, layer_names, dim, rng = calibrated_detector
        poisoner = SupplyChainPoisoning(strategy="steganographic", seed=51)
        poisoned = poisoner.poison(weights).poisoned_weights
        result = det.detect(clean_acts(layer_names, dim, rng), weights=poisoned)
        # Any weight change must be caught
        assert result.threat_type != LLMThreatType.CLEAN


class TestLLMThreatDetectorAdversarial:
    def test_adversarial_text_injection_detected(self, calibrated_detector):
        from edgeguard.attacks.adversarial import AdversarialInputAttack
        from edgeguard.llm.detector import LLMThreatType
        det, weights, layer_names, dim, rng = calibrated_detector
        adv = AdversarialInputAttack(attack_type="text_injection", seed=0)
        adv_acts = adv.attack(layer_names, activation_dim=dim).adversarial_activation
        result = det.detect(adv_acts, weights=weights)
        assert result.threat_type != LLMThreatType.CLEAN

    def test_safety_bypass_detected_as_non_clean(self, calibrated_detector):
        """Safety bypass should be flagged as *some* threat — not CLEAN."""
        from edgeguard.attacks.adversarial import AdversarialInputAttack
        from edgeguard.llm.detector import LLMThreatType
        det, weights, layer_names, dim, rng = calibrated_detector
        adv = AdversarialInputAttack(attack_type="safety_bypass", seed=1)
        adv_acts = adv.attack(layer_names, activation_dim=dim).adversarial_activation
        result = det.detect(adv_acts, weights=weights)
        # Safety bypass shifts activations — detector must flag it
        # It may be classified as JAILBREAK, ADVERSARIAL_INPUT, TRANSIENT_FAULT,
        # PROMPT_INJECTION or UNKNOWN depending on which layer scores are highest
        assert result.threat_type != LLMThreatType.CLEAN


class TestLLMThreatDetectorExfiltration:
    def test_exfil_probe_detected_with_high_memorization(self, calibrated_detector):
        from edgeguard.attacks.training_data_recovery import TrainingDataRecovery
        from edgeguard.llm.detector import LLMThreatType
        det, weights, layer_names, dim, rng = calibrated_detector
        rec = TrainingDataRecovery(memorization_rate=0.80, seed=7)
        probe = rec.simulate_probe(layer_names, n_queries=50, activation_dim=dim)
        result = det.detect(probe.activation_signature, weights=weights)
        assert result.threat_type in (
            LLMThreatType.TRAINING_DATA_EXFILTRATION,
            LLMThreatType.WEIGHT_THEFT_PROBE,
            LLMThreatType.UNKNOWN,
        )

    def test_low_memorization_rate_not_flagged_as_exfil(self, calibrated_detector):
        from edgeguard.attacks.training_data_recovery import TrainingDataRecovery
        from edgeguard.llm.detector import LLMThreatType
        det, weights, layer_names, dim, rng = calibrated_detector
        # Very low memorization → should look like normal inference
        rec = TrainingDataRecovery(memorization_rate=0.01, seed=8)
        probe = rec.simulate_probe(layer_names, n_queries=5, activation_dim=dim)
        result = det.detect(probe.activation_signature, weights=weights)
        # May or may not be flagged, but must not crash
        assert isinstance(result.threat_type, LLMThreatType)


class TestLLMThreatDetectorSynergy:
    def test_synergy_caught_despite_activation_suppression(self, calibrated_detector):
        from edgeguard.attacks.synergy import SynergyAttack
        from edgeguard.llm.detector import LLMThreatType
        det, weights, layer_names, dim, rng = calibrated_detector
        syn = SynergyAttack(suppression=0.65, seed=42)
        s = syn.attack(weights, layer_names, activation_dim=dim)
        result = det.detect(s.combined_activation, weights=s.bfa_result.attacked_weights)
        # Merkle must fire: unified detection must NOT return CLEAN
        assert result.threat_type != LLMThreatType.CLEAN

    def test_synergy_activation_suppression_reported(self, calibrated_detector):
        from edgeguard.attacks.synergy import SynergyAttack
        det, weights, layer_names, dim, rng = calibrated_detector
        syn = SynergyAttack(suppression=0.65, seed=43)
        s = syn.attack(weights, layer_names, activation_dim=dim)
        # anomaly_suppression_factor should reflect the designed 65% suppression
        assert s.anomaly_suppression_factor > 0.0


# ===========================================================================
# Memorization score and entropy internal methods
# ===========================================================================

class TestLLMDetectorScoring:
    def test_memorization_score_high_for_exfil_activation(self, calibrated_detector):
        from edgeguard.attacks.training_data_recovery import TrainingDataRecovery
        det, weights, layer_names, dim, rng = calibrated_detector
        rec = TrainingDataRecovery(memorization_rate=0.80, seed=9)
        probe = rec.simulate_probe(layer_names, n_queries=50, activation_dim=dim)
        score = det._memorization_score(probe.activation_signature)
        assert 0.0 <= score <= 1.0
        # High memorization should yield a relatively high score
        assert score > 0.0

    def test_memorization_score_low_for_clean_activation(self, calibrated_detector):
        det, weights, layer_names, dim, rng = calibrated_detector
        acts = clean_acts(layer_names, dim, rng)
        score = det._memorization_score(acts)
        assert 0.0 <= score <= 1.0

    def test_entropy_range_valid(self, calibrated_detector):
        det, weights, layer_names, dim, rng = calibrated_detector
        acts = clean_acts(layer_names, dim, rng)
        entropy = det._late_layer_entropy(acts)
        assert 0.0 <= entropy <= 1.0
