"""
Synergy Attack — combined BFA weight tamper + adversarial input.

This is the most sophisticated attack in the taxonomy, described in:
  Yao et al., "DeepHammer: Depleting the Intelligence of Deep Neural Networks
  through Targeted Chain of Bit Flips", USENIX Security 2020.

The attack proceeds in two phases:
  Phase 1 (offline): Use gradient information and Rowhammer to flip the minimum
           number of bits to open a "backdoor channel" in the model.
  Phase 2 (online):  Craft an adversarial input that exploits the flipped weights.

The key insight: the adversarial input alone is detectable (unusual activation).
The weight flip alone is detectable (hash mismatch). But if the hash check is
bypassed (attacker forged the attestation, or attestation is not in use), the
combined attack produces activations that look LESS anomalous than either alone
because the weight flip is chosen to suppress the detection signature.

EdgeGuard's defense: simultaneous weight hash + activation check prevents both
individual bypass and the synergy. This is the killer app of the unified approach.

Activation signature of synergy attack (when hash check is bypassed):
  - Early layers: appears clean (adversarial input is calibrated for the flipped model)
  - Late layers: also appears plausible (flipped weights guide activations toward them)
  - Overall: LOWER anomaly score than pure adversarial, but still detectable
    via the cross-layer correlation pattern (early-normal + late-concentrated)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bitflip import BitFlipAttack, BitFlipResult
from .adversarial import AdversarialInputAttack, AdversarialResult


@dataclass
class SynergyResult:
    """Result of a combined BFA + adversarial attack."""
    bfa_result: BitFlipResult
    adv_result: AdversarialResult
    combined_activation: dict[str, np.ndarray]
    # Anomaly suppression: how much lower is the combined anomaly score
    # compared to the pure adversarial attack?
    anomaly_suppression_factor: float
    # Did the combined attack evade a hypothetical activation-only monitor?
    evaded_activation_only_monitor: bool
    # Would EdgeGuard's unified check catch it?
    caught_by_edgeguard: bool
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        evaded = "EVADED activation-only" if self.evaded_activation_only_monitor else "caught by activation"
        caught = "CAUGHT by EdgeGuard" if self.caught_by_edgeguard else "evaded EdgeGuard (!)"
        return (
            f"[SYNERGY ATTACK] "
            f"anomaly suppression={self.anomaly_suppression_factor:.2f}x | "
            f"{evaded} | {caught}"
        )


class SynergyAttack:
    """
    Simulates the combined BFA + adversarial input attack.

    Parameters:
        bfa_strategy:   Strategy for the bit-flip component ("targeted" is most dangerous).
        adv_type:       Adversarial input type ("safety_bypass" for synergy).
        suppression:    How much the weight flip reduces activation anomaly (0–1).
                        0 = no suppression (adversarial still looks anomalous)
                        0.7 = 70% suppression (combined looks almost clean)
        seed:           RNG seed.
    """

    def __init__(
        self,
        bfa_strategy: str = "targeted",
        adv_type: str = "safety_bypass",
        suppression: float = 0.65,
        seed: int = 42,
    ) -> None:
        self.bfa = BitFlipAttack(strategy=bfa_strategy, n_flips=1, seed=seed)
        self.adv = AdversarialInputAttack(attack_type=adv_type, seed=seed + 1)
        self.suppression = suppression
        self.rng = np.random.default_rng(seed)

    def attack(
        self,
        weights: dict[str, np.ndarray],
        layer_names: list[str],
        target_layer: Optional[str] = None,
        activation_dim: int = 64,
        activation_zscore_threshold: float = 4.0,
    ) -> SynergyResult:
        """
        Execute the combined attack.

        Args:
            weights:                     Clean INT8 weights.
            layer_names:                 Model layer names.
            target_layer:                Layer to target with BFA.
            activation_dim:              Activation dimension.
            activation_zscore_threshold: Threshold for activation-only monitor.

        Returns:
            SynergyResult with full attack chain analysis.
        """
        tgt = target_layer or layer_names[0]

        # Phase 1: BFA
        bfa_result = self.bfa.attack(weights, target_layer=tgt)

        # Phase 2: Adversarial input (crafted for the FLIPPED model)
        adv_result = self.adv.attack(layer_names, activation_dim=activation_dim)

        # Compute combined activation: BFA suppresses the adversarial signature
        combined = {}
        n_layers = len(layer_names)
        early_cutoff = max(1, int(n_layers * 0.4))

        for i, name in enumerate(layer_names):
            adv_act = adv_result.adversarial_activation[name].astype(np.float32)
            is_early = i < early_cutoff

            if is_early:
                # Weight flip in early layer "absorbs" the adversarial perturbation
                # Early layers look almost clean
                suppression = self.suppression
            else:
                # Late layers slightly more anomalous than clean, but far less than
                # pure adversarial
                suppression = self.suppression * 0.5

            # Suppress: blend adversarial activation toward clean baseline (mean≈10, std≈5)
            clean_act = np.clip(
                self.rng.normal(10, 5, size=activation_dim), -128, 127
            )
            combined_act = (1 - suppression) * adv_act + suppression * clean_act
            combined[name] = np.clip(combined_act, -128, 127).astype(np.int8)

        # Evaluate: would activation-only monitor catch this?
        # Compute rough max z-score for combined activations
        # (approximation: if all combined activations are near-clean, monitor misses it)
        max_z_adv = max(
            abs(float(adv_result.adversarial_activation[n].astype(np.float32).mean()) - 10) / 5
            for n in layer_names
        )
        max_z_combined = max_z_adv * (1 - self.suppression)
        evaded_activation_only = max_z_combined < activation_zscore_threshold

        # EdgeGuard always catches it via hash mismatch (BFA modified a layer)
        caught_by_edgeguard = len(bfa_result.affected_layers) > 0

        suppression_factor = max_z_adv / max(max_z_combined, 0.01)

        return SynergyResult(
            bfa_result=bfa_result,
            adv_result=adv_result,
            combined_activation=combined,
            anomaly_suppression_factor=suppression_factor,
            evaded_activation_only_monitor=evaded_activation_only,
            caught_by_edgeguard=caught_by_edgeguard,
            metadata={
                "target_layer": tgt,
                "suppression": self.suppression,
                "max_z_adversarial": max_z_adv,
                "max_z_combined": max_z_combined,
                "activation_zscore_threshold": activation_zscore_threshold,
            }
        )
