"""
Adversarial Input Attack simulation for quantized LLMs.

Models the threat of crafted inputs designed to bypass safety filters,
cause misclassification, or trigger unexpected model behaviour.

For quantized INT8 models on edge devices, adversarial attacks have
additional characteristics:
  1. Quantization creates flat regions in the loss landscape → gradient
     masking makes simple FGSM less effective but also makes detection harder.
  2. INT8 rounding amplifies certain perturbation directions.
  3. Adversarial examples transferable across quantization levels
     (Sharma & Ding, 2019).

This module simulates adversarial activation patterns for two threat cases:
  "text_injection":    Input crafted to override system prompt / extract context
  "safety_bypass":    Input crafted to make safety classifier output benign
  "membership_probe":  Input designed to determine if a data point was in training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AdversarialResult:
    """Result of an adversarial input attack simulation."""
    attack_type: str
    adversarial_activation: dict[str, np.ndarray]
    perturbation_magnitude: float   # L-inf norm of input perturbation
    success: bool                   # Did attack achieve its goal (simulated)
    confidence: float               # Attacker's estimated success probability
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"[ADVERSARIAL | {self.attack_type.upper()} | {status}] "
            f"confidence={self.confidence:.2f} | "
            f"perturbation L-inf={self.perturbation_magnitude:.4f}"
        )


class AdversarialInputAttack:
    """
    Simulates adversarial input attacks and their distinctive activation patterns.

    For LLMs, adversarial inputs produce a characteristic activation signature:
      - Early transformer layers: high variance (input is unusual)
      - Middle layers: activation norm grows anomalously
      - Final layers: may appear confident (bypass succeeded) or chaotic (bypass failed)

    Parameters:
        attack_type: "text_injection" | "safety_bypass" | "membership_probe"
        epsilon:     L-inf perturbation budget (in input embedding space).
        seed:        RNG seed.
    """

    ATTACK_PROFILES = {
        "text_injection": {
            "early_mean_shift": 20.0,  # Input looks very different
            "late_mean_shift": 5.0,
            "early_std_scale": 2.5,    # High variance in early layers
            "late_std_scale": 1.2,
            "success_rate": 0.60,
        },
        "safety_bypass": {
            "early_mean_shift": 15.0,
            "late_mean_shift": 25.0,  # Late layers forced into "safe" region
            "early_std_scale": 2.0,
            "late_std_scale": 0.5,    # Very narrow — forced to specific output
            "success_rate": 0.35,
        },
        "membership_probe": {
            "early_mean_shift": 8.0,  # Near-normal (canary-like input)
            "late_mean_shift": 30.0,  # High confidence in late layers
            "early_std_scale": 1.1,
            "late_std_scale": 0.3,    # Very low entropy
            "success_rate": 0.45,
        },
    }

    def __init__(
        self,
        attack_type: str = "safety_bypass",
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        if attack_type not in self.ATTACK_PROFILES:
            raise ValueError(f"attack_type must be one of {list(self.ATTACK_PROFILES)}")
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.profile = self.ATTACK_PROFILES[attack_type]

    def attack(
        self,
        layer_names: list[str],
        activation_dim: int = 64,
    ) -> AdversarialResult:
        """
        Simulate an adversarial attack and generate characteristic activations.

        Args:
            layer_names:    Model layer names (early → late).
            activation_dim: Activation vector dimension.

        Returns:
            AdversarialResult with activation pattern and attack outcome.
        """
        n_layers = len(layer_names)
        early_cutoff = max(1, int(n_layers * 0.4))
        activations = {}

        for i, name in enumerate(layer_names):
            is_early = i < early_cutoff
            profile = self.profile

            if is_early:
                mean = profile["early_mean_shift"]
                std = 5.0 * profile["early_std_scale"]
            else:
                mean = profile["late_mean_shift"]
                std = 5.0 * profile["late_std_scale"]

            act = np.clip(
                self.rng.normal(mean, std, size=activation_dim),
                -128, 127,
            ).astype(np.int8)
            activations[name] = act

        success = self.rng.random() < self.profile["success_rate"]
        confidence = float(
            self.rng.uniform(0.65, 0.95) if success
            else self.rng.uniform(0.20, 0.50)
        )

        return AdversarialResult(
            attack_type=self.attack_type,
            adversarial_activation=activations,
            perturbation_magnitude=self.epsilon,
            success=success,
            confidence=confidence,
            metadata={
                "early_cutoff": early_cutoff,
                "n_layers": n_layers,
                "activation_dim": activation_dim,
            }
        )
