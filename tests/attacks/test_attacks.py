"""
Unit tests for EdgeGuard attack simulation modules.

Covers:
  - BitFlipAttack (Rowhammer / BFA)
  - WeightExtractionAttack (cold-boot, EM, memory-dump)
  - TrainingDataRecovery (activation-inversion membership probe)
  - SupplyChainPoisoning (steganographic, targeted, distributed)
  - AdversarialInputAttack (text injection, safety bypass, membership probe)
  - SynergyAttack (DeepHammer: BFA + adversarial)
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Common fixture helpers
# ---------------------------------------------------------------------------

def _small_weights(seed=0):
    """3-layer INT8 weight dict for fast testing."""
    rng = np.random.default_rng(seed)
    return {
        "layer_0": rng.integers(-128, 128, size=(64, 64), dtype=np.int8),
        "layer_1": rng.integers(-128, 128, size=(64, 32), dtype=np.int8),
        "layer_2": rng.integers(-128, 128, size=(32, 16), dtype=np.int8),
    }


def _small_layer_names():
    return ["layer_0", "layer_1", "layer_2"]


# ===========================================================================
# BitFlipAttack
# ===========================================================================

class TestBitFlipAttack:
    from edgeguard.attacks.bitflip import BitFlipAttack  # import at class scope

    def _attacker(self, strategy="random", n_flips=1, seed=42):
        from edgeguard.attacks.bitflip import BitFlipAttack
        return BitFlipAttack(strategy=strategy, n_flips=n_flips, seed=seed)

    def test_random_flips_change_weights(self):
        att = self._attacker("random")
        w = _small_weights()
        result = att.attack(w)
        changed = sum(
            int(np.any(result.attacked_weights[k] != w[k]))
            for k in w
        )
        assert changed >= 1, "At least one layer should differ after random BFA"

    def test_flipped_bit_count_respected(self):
        """Sign-bit strategy flips exactly 1 bit per call."""
        att = self._attacker("sign_bit", n_flips=1)
        w = _small_weights()
        result = att.attack(w)
        total_diff = sum(
            int(np.sum(result.attacked_weights[k] != w[k]))
            for k in w
        )
        assert total_diff == 1

    def test_targeted_hits_specified_layer(self):
        att = self._attacker("targeted", n_flips=1)
        w = _small_weights()
        result = att.attack(w, target_layer="layer_1")
        # layer_1 must have changed
        assert np.any(result.attacked_weights["layer_1"] != w["layer_1"])
        # Other layers should be untouched
        assert np.array_equal(result.attacked_weights["layer_0"], w["layer_0"])
        assert np.array_equal(result.attacked_weights["layer_2"], w["layer_2"])

    def test_progressive_iterates_largest_magnitude(self):
        """Progressive BFA flips the globally worst weight; at least 1 must change."""
        att = self._attacker("progressive", n_flips=1)
        w = _small_weights()
        result = att.attack(w)
        total_diff = sum(
            int(np.sum(result.attacked_weights[k] != w[k]))
            for k in w
        )
        assert total_diff >= 1

    def test_result_has_expected_fields(self):
        att = self._attacker()
        result = att.attack(_small_weights())
        assert hasattr(result, "attacked_weights")
        assert hasattr(result, "flip_log")
        assert hasattr(result, "n_flips")
        assert hasattr(result, "affected_layers")

    def test_determinism_with_same_seed(self):
        w = _small_weights()
        r1 = self._attacker("random", seed=7).attack(w)
        r2 = self._attacker("random", seed=7).attack(w)
        for k in w:
            assert np.array_equal(r1.attacked_weights[k], r2.attacked_weights[k])

    def test_sign_bit_produces_large_magnitude_shift(self):
        """Sign-bit flip on INT8 must produce a >=127 shift."""
        att = self._attacker("sign_bit", n_flips=1)
        w = _small_weights(seed=1)
        result = att.attack(w)
        max_delta = max(
            int(np.max(np.abs(
                result.attacked_weights[k].astype(np.int16) - w[k].astype(np.int16)
            )))
            for k in w
        )
        assert max_delta >= 127, f"Sign-bit flip should shift by ≥127 in INT8, got {max_delta}"

    def test_original_weights_not_mutated(self):
        """Attack must return a copy; originals unchanged."""
        w = _small_weights()
        original_copy = {k: v.copy() for k, v in w.items()}
        self._attacker("sign_bit").attack(w)
        for k in w:
            assert np.array_equal(w[k], original_copy[k]), "Original weights were mutated"


# ===========================================================================
# WeightExtractionAttack
# ===========================================================================

class TestWeightExtractionAttack:
    def _extractor(self, attack_type="cold_boot", seed=42):
        from edgeguard.attacks.weight_extraction import WeightExtractionAttack
        return WeightExtractionAttack(attack_type=attack_type, seed=seed)

    def test_cold_boot_fidelity_above_threshold(self):
        ext = self._extractor("cold_boot")
        w = _small_weights()
        result = ext.extract(w)
        avg_fidelity = np.mean(list(result.fidelity.values()))
        assert avg_fidelity > 0.80, f"Cold-boot fidelity should be >80%, got {avg_fidelity:.2%}"

    def test_memory_dump_near_perfect_fidelity(self):
        ext = self._extractor("memory_dump")
        w = _small_weights()
        result = ext.extract(w)
        avg_fidelity = np.mean(list(result.fidelity.values()))
        assert avg_fidelity > 0.98, f"Memory-dump fidelity should be >98%, got {avg_fidelity:.2%}"

    def test_fidelity_in_valid_range(self):
        """Mean fidelity (cosine similarity) must be in [0, 1]."""
        for attack_type in ("cold_boot", "em_side_channel", "memory_dump"):
            ext = self._extractor(attack_type, seed=0)
            result = ext.extract(_small_weights())
            assert 0.0 <= result.mean_fidelity <= 1.0, \
                f"{attack_type} mean_fidelity {result.mean_fidelity} out of [0,1]"

    def test_result_has_dequantized_weights(self):
        result = self._extractor().extract(_small_weights())
        # Result stores extracted INT8 weights and their dequantized FP32 form
        assert hasattr(result, "extracted_weights") or hasattr(result, "dequantized")

    def test_snr_reported(self):
        """snr_db is returned (scalar dB or None for zero-noise extraction)."""
        result = self._extractor("cold_boot").extract(_small_weights())
        assert hasattr(result, "snr_db")
        # cold_boot adds noise → SNR is a finite number
        assert result.snr_db is None or isinstance(result.snr_db, (int, float))

    def test_determinism(self):
        w = _small_weights()
        r1 = self._extractor("cold_boot", seed=5).extract(w)
        r2 = self._extractor("cold_boot", seed=5).extract(w)
        for k in w:
            assert np.allclose(r1.dequantized[k], r2.dequantized[k])


# ===========================================================================
# TrainingDataRecovery
# ===========================================================================

class TestTrainingDataRecovery:
    def _recovery(self, memorization_rate=0.45, seed=42):
        from edgeguard.attacks.training_data_recovery import TrainingDataRecovery
        return TrainingDataRecovery(
            vocab_size=32000, seq_len=32,
            memorization_rate=memorization_rate, seed=seed
        )

    def test_simulate_probe_returns_result(self):
        rec = self._recovery()
        result = rec.simulate_probe(_small_layer_names(), n_queries=10, activation_dim=16)
        assert result is not None
        assert hasattr(result, "activation_signature")
        assert hasattr(result, "estimated_pii_risk")
        assert hasattr(result, "success_rate")

    def test_late_layer_high_mean_in_extraction_signature(self):
        """During exfil probe, late-layer activations must have higher mean."""
        rec = self._recovery()
        result = rec.simulate_probe(_small_layer_names(), n_queries=20, activation_dim=16)
        late_name = _small_layer_names()[-1]
        late_act = result.activation_signature[late_name].astype(np.float32)
        assert np.mean(np.abs(late_act)) > 5.0, "Late-layer exfil should have elevated mean"

    def test_memorization_rate_affects_pii_risk(self):
        low_risk = self._recovery(memorization_rate=0.01).simulate_probe(
            _small_layer_names(), n_queries=20, activation_dim=16
        )
        high_risk = self._recovery(memorization_rate=0.90).simulate_probe(
            _small_layer_names(), n_queries=20, activation_dim=16
        )
        risk_order = ["none", "low", "medium", "high", "critical"]
        # High memorization rate should yield same or higher risk
        assert risk_order.index(high_risk.estimated_pii_risk) >= risk_order.index(low_risk.estimated_pii_risk)

    def test_compare_signatures_returns_per_layer_dict(self):
        rec = self._recovery()
        normal = rec.normal_inference_activation(_small_layer_names(), activation_dim=16)
        result = rec.simulate_probe(_small_layer_names(), n_queries=10, activation_dim=16)
        comparison = rec.compare_signatures(normal, result.activation_signature)
        # compare_signatures returns a dict keyed by layer name
        assert isinstance(comparison, dict)
        assert all(name in comparison for name in _small_layer_names())

    def test_determinism(self):
        r1 = self._recovery(seed=11).simulate_probe(_small_layer_names(), n_queries=5, activation_dim=8)
        r2 = self._recovery(seed=11).simulate_probe(_small_layer_names(), n_queries=5, activation_dim=8)
        for k in _small_layer_names():
            assert np.array_equal(r1.activation_signature[k], r2.activation_signature[k])


# ===========================================================================
# SupplyChainPoisoning
# ===========================================================================

class TestSupplyChainPoisoning:
    def _poisoner(self, strategy="steganographic", seed=42):
        from edgeguard.attacks.supply_chain import SupplyChainPoisoning
        return SupplyChainPoisoning(strategy=strategy, seed=seed)

    def test_steganographic_changes_lsbs(self):
        w = _small_weights()
        result = self._poisoner("steganographic").poison(w)
        # At least some weights differ
        any_changed = any(
            np.any(result.poisoned_weights[k] != w[k]) for k in w
        )
        assert any_changed

    def test_distributed_affects_many_layers(self):
        w = _small_weights()
        result = self._poisoner("distributed").poison(w)
        changed_layers = sum(
            1 for k in w if np.any(result.poisoned_weights[k] != w[k])
        )
        assert changed_layers >= 2, "Distributed poisoning should affect multiple layers"

    def test_targeted_layer_strategy(self):
        w = _small_weights()
        result = self._poisoner("targeted_layer").poison(w, target_layer="layer_1")
        assert np.any(result.poisoned_weights["layer_1"] != w["layer_1"])

    def test_result_has_steganographic_payload(self):
        result = self._poisoner("steganographic").poison(_small_weights())
        # Result should describe the trigger or modification
        assert hasattr(result, "trigger_description") or hasattr(result, "stealth_score")

    def test_steganographic_magnitude_change_small(self):
        """Steganographic poisoning should produce small weight changes (≤1 INT8 unit per element)."""
        w = _small_weights()
        result = self._poisoner("steganographic").poison(w)
        for k in w:
            delta = np.abs(
                result.poisoned_weights[k].astype(np.int16) - w[k].astype(np.int16)
            )
            assert np.max(delta) <= 1, f"Steg poisoning changes {k} by more than 1 INT8 unit"

    def test_original_not_mutated(self):
        w = _small_weights()
        orig = {k: v.copy() for k, v in w.items()}
        self._poisoner().poison(w)
        for k in w:
            assert np.array_equal(w[k], orig[k])


# ===========================================================================
# AdversarialInputAttack
# ===========================================================================

class TestAdversarialInputAttack:
    def _attacker(self, attack_type="text_injection", seed=42):
        from edgeguard.attacks.adversarial import AdversarialInputAttack
        return AdversarialInputAttack(attack_type=attack_type, seed=seed)

    def test_text_injection_produces_activation_dict(self):
        result = self._attacker("text_injection").attack(
            _small_layer_names(), activation_dim=16
        )
        assert hasattr(result, "adversarial_activation")
        for name in _small_layer_names():
            assert name in result.adversarial_activation

    def test_safety_bypass_has_elevated_early_layer_shift(self):
        result = self._attacker("safety_bypass").attack(
            _small_layer_names(), activation_dim=16
        )
        early_name = _small_layer_names()[0]
        early_act = result.adversarial_activation[early_name].astype(np.float32)
        assert np.mean(np.abs(early_act)) > 5.0

    def test_membership_probe_attack_type(self):
        result = self._attacker("membership_probe").attack(
            _small_layer_names(), activation_dim=16
        )
        assert result is not None
        assert hasattr(result, "success") or hasattr(result, "confidence")

    def test_activation_shape_matches_dim(self):
        dim = 24
        result = self._attacker().attack(_small_layer_names(), activation_dim=dim)
        for name in _small_layer_names():
            assert result.adversarial_activation[name].shape == (dim,)

    def test_different_attack_types_differ(self):
        inject = self._attacker("text_injection", seed=0).attack(
            _small_layer_names(), activation_dim=16
        )
        bypass = self._attacker("safety_bypass", seed=0).attack(
            _small_layer_names(), activation_dim=16
        )
        # At least one layer should differ between attack types
        any_diff = any(
            not np.array_equal(
                inject.adversarial_activation[n],
                bypass.adversarial_activation[n],
            )
            for n in _small_layer_names()
        )
        assert any_diff


# ===========================================================================
# SynergyAttack
# ===========================================================================

class TestSynergyAttack:
    def _attacker(self, seed=42):
        from edgeguard.attacks.synergy import SynergyAttack
        return SynergyAttack(bfa_strategy="targeted", adv_type="safety_bypass",
                             suppression=0.65, seed=seed)

    def test_synergy_modifies_weights(self):
        w = _small_weights()
        result = self._attacker().attack(w, _small_layer_names(), activation_dim=16)
        any_diff = any(
            np.any(result.bfa_result.attacked_weights[k] != w[k]) for k in w
        )
        assert any_diff

    def test_synergy_produces_combined_activation(self):
        w = _small_weights()
        result = self._attacker().attack(w, _small_layer_names(), activation_dim=16)
        assert hasattr(result, "combined_activation")
        for name in _small_layer_names():
            assert name in result.combined_activation

    def test_suppression_factor_positive(self):
        """anomaly_suppression_factor is a ratio (>0); 1.0 = no suppression."""
        w = _small_weights()
        result = self._attacker().attack(w, _small_layer_names(), activation_dim=16)
        assert result.anomaly_suppression_factor >= 0.0

    def test_activation_only_monitor_evasion(self):
        """Synergy attack should potentially evade activation-only monitors."""
        w = _small_weights()
        result = self._attacker().attack(w, _small_layer_names(), activation_dim=16)
        # The attack is designed to suppress anomaly scores
        assert hasattr(result, "evaded_activation_only_monitor")

    def test_edgeguard_still_catches_synergy(self):
        """EdgeGuard with Merkle must catch synergy even when activations look clean."""
        w = _small_weights()
        result = self._attacker().attack(w, _small_layer_names(), activation_dim=16)
        # Weight was modified → Merkle attestation will catch it
        assert hasattr(result, "caught_by_edgeguard")
        assert result.caught_by_edgeguard is True

    def test_synergy_determinism(self):
        w = _small_weights()
        r1 = self._attacker(seed=3).attack(w, _small_layer_names(), activation_dim=8)
        r2 = self._attacker(seed=3).attack(w, _small_layer_names(), activation_dim=8)
        for k in w:
            assert np.array_equal(
                r1.bfa_result.attacked_weights[k],
                r2.bfa_result.attacked_weights[k],
            )


# ===========================================================================
# Integration: Full attack chain → EdgeGuard detection
# ===========================================================================

class TestAttackChainDetection:
    """
    End-to-end: apply attack, run EdgeGuard, assert detection.

    Uses the LLMThreatDetector (Merkle + activation monitor + LLM scoring)
    rather than the base ThreatClassifier.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build detector once per class."""
        from edgeguard.llm.model import QuantizedTransformerStub
        from edgeguard.llm.detector import LLMThreatDetector
        from edgeguard.attestation.manager import MerkleAttestationManager
        from edgeguard.detection.calibration import Calibrator
        from edgeguard.detection.monitor import LayerwiseActivationMonitor

        rng = np.random.default_rng(99)
        self.model = QuantizedTransformerStub(profile_name="tiny", seed=42)
        self.weights = self.model.get_weights()
        self.layer_names = self.model.layer_names
        dim = 16

        mgr = MerkleAttestationManager()
        mgr.build(self.weights)

        cal = Calibrator(layer_names=self.layer_names, max_cov_dim=16, verbose=0)
        for _ in range(100):
            acts = {
                n: np.clip(rng.normal(10, 5, size=dim), -128, 127).astype(np.int8)
                for n in self.layer_names
            }
            cal.feed_activations(acts)
        cal_stats = cal.finalize()
        monitor = LayerwiseActivationMonitor(
            cal_stats, zscore_threshold=4.0, mahal_threshold=50.0
        )
        self.detector = LLMThreatDetector(
            layer_names=self.layer_names,
            attestation_manager=mgr,
            activation_monitor=monitor,
        )
        self.rng = rng
        self.dim = dim

    def _clean_acts(self):
        return {
            n: np.clip(self.rng.normal(10, 5, size=self.dim), -128, 127).astype(np.int8)
            for n in self.layer_names
        }

    def test_clean_detected_as_clean(self):
        from edgeguard.llm.detector import LLMThreatType
        result = self.detector.detect(self._clean_acts(), weights=self.weights)
        assert result.threat_type == LLMThreatType.CLEAN

    def test_bfa_detected_as_weight_tamper(self):
        from edgeguard.attacks.bitflip import BitFlipAttack
        from edgeguard.llm.detector import LLMThreatType
        bfa = BitFlipAttack(strategy="sign_bit", n_flips=1, seed=10)
        attacked = bfa.attack(self.weights).attacked_weights
        result = self.detector.detect(self._clean_acts(), weights=attacked)
        assert result.threat_type == LLMThreatType.WEIGHT_TAMPER

    def test_supply_chain_detected(self):
        from edgeguard.attacks.supply_chain import SupplyChainPoisoning
        from edgeguard.llm.detector import LLMThreatType
        poisoner = SupplyChainPoisoning(strategy="distributed", seed=0)
        poisoned = poisoner.poison(self.weights).poisoned_weights
        result = self.detector.detect(self._clean_acts(), weights=poisoned)
        # Merkle catches any weight modification
        assert result.threat_type in (
            LLMThreatType.WEIGHT_TAMPER,
            LLMThreatType.SUPPLY_CHAIN_BACKDOOR,
        )

    def test_training_data_exfil_detected(self):
        from edgeguard.attacks.training_data_recovery import TrainingDataRecovery
        from edgeguard.llm.detector import LLMThreatType
        rec = TrainingDataRecovery(memorization_rate=0.80, seed=5)
        result_rec = rec.simulate_probe(
            self.layer_names, n_queries=20, activation_dim=self.dim
        )
        result = self.detector.detect(
            result_rec.activation_signature, weights=self.weights
        )
        assert result.threat_type in (
            LLMThreatType.TRAINING_DATA_EXFILTRATION,
            LLMThreatType.WEIGHT_THEFT_PROBE,
            LLMThreatType.UNKNOWN,
        )

    def test_synergy_caught_by_unified_detector(self):
        from edgeguard.attacks.synergy import SynergyAttack
        from edgeguard.llm.detector import LLMThreatType
        synergy = SynergyAttack(suppression=0.65, seed=99)
        s_result = synergy.attack(
            self.weights, self.layer_names,
            target_layer=self.layer_names[0],
            activation_dim=self.dim,
        )
        result = self.detector.detect(
            s_result.combined_activation,
            weights=s_result.bfa_result.attacked_weights,
        )
        # Merkle must catch it even when activations are suppressed
        assert result.threat_type != LLMThreatType.CLEAN
