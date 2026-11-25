"""
Bit-Flip Attack (BFA) simulation — Rowhammer / DeepHammer style.

Models the attack described in:
  Rakin et al., "Bit-Flip Attack: Crushing Neural Network with Progressive
  Bit Search", ICCV 2019.
  Yao et al., "DeepHammer: Depleting the Intelligence of Deep Neural Networks
  through Targeted Chain of Bit Flips", USENIX Security 2020.

For quantized LLMs, flipping the sign bit (bit 7) of an INT8 weight
is equivalent to a +/- 256-unit shift — catastrophically larger than
the magnitude of any adversarial perturbation.

The attack can be:
  - Random:    Flip random bits in a randomly chosen layer (noise baseline)
  - Targeted:  Flip the sign bit of the highest-magnitude weights in a
               specified layer (maximally disruptive)
  - Progressive: Iteratively find and flip the single most impactful bit
               across the entire model (simulates the BFA paper's algorithm)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BitFlipResult:
    """Result of a bit-flip attack simulation."""
    attacked_weights: dict[str, np.ndarray]   # Modified weight tensors
    flip_log: list[dict]                       # Per-flip metadata
    n_flips: int
    affected_layers: list[str]

    def magnitude_shift(self) -> dict[str, float]:
        """Not computable without original — see BitFlipAttack.attack()."""
        return {}


class BitFlipAttack:
    """
    Simulates Rowhammer-style bit-flip attacks on INT8 quantized model weights.

    Parameters:
        strategy:  "random" | "targeted" | "sign_bit" | "progressive"
        n_flips:   Total number of bits to flip.
        seed:      RNG seed for reproducibility.
    """

    def __init__(
        self,
        strategy: str = "targeted",
        n_flips: int = 1,
        seed: int = 42,
    ) -> None:
        if strategy not in ("random", "targeted", "sign_bit", "progressive"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.n_flips = n_flips
        self.rng = np.random.default_rng(seed)

    def attack(
        self,
        weights: dict[str, np.ndarray],
        target_layer: Optional[str] = None,
    ) -> BitFlipResult:
        """
        Apply bit-flip attack to the weight dictionary.

        Args:
            weights:      Original INT8 weight tensors (not modified in-place).
            target_layer: Layer to target (required for "targeted"/"sign_bit").
                          If None, a random layer is chosen.

        Returns:
            BitFlipResult with modified weights and per-flip audit log.
        """
        # Deep copy so original is untouched
        attacked = {k: v.copy() for k, v in weights.items()}
        flip_log = []
        affected_layers = set()

        layer_names = list(weights.keys())
        if target_layer is None:
            target_layer = layer_names[int(self.rng.integers(0, len(layer_names)))]

        for flip_idx in range(self.n_flips):
            if self.strategy == "random":
                entry = self._random_flip(attacked, layer_names)
            elif self.strategy == "targeted":
                entry = self._targeted_flip(attacked, target_layer, weights)
            elif self.strategy == "sign_bit":
                entry = self._sign_bit_flip(attacked, target_layer)
            elif self.strategy == "progressive":
                entry = self._progressive_flip(attacked, weights)
            else:
                break

            flip_log.append(entry)
            affected_layers.add(entry["layer"])

        return BitFlipResult(
            attacked_weights=attacked,
            flip_log=flip_log,
            n_flips=len(flip_log),
            affected_layers=list(affected_layers),
        )

    # ------------------------------------------------------------------
    # Flip strategies
    # ------------------------------------------------------------------

    def _random_flip(self, weights: dict, layer_names: list) -> dict:
        """Flip a random bit in a random layer."""
        layer = layer_names[int(self.rng.integers(0, len(layer_names)))]
        w = weights[layer]
        idx = tuple(int(self.rng.integers(0, s)) for s in w.shape)
        bit = int(self.rng.integers(0, 8))
        old_val = int(w[idx])
        # XOR through the uint8 view so bit 7 (sign) does not overflow int8
        w[idx] = (w[idx].view(np.uint8) ^ np.uint8(1 << bit)).view(np.int8)
        return {"layer": layer, "idx": idx, "bit": bit,
                "old": old_val, "new": int(w[idx]), "strategy": "random"}

    def _targeted_flip(self, weights: dict, layer: str, original: dict) -> dict:
        """
        Flip the sign bit of the highest-magnitude weight in the target layer.
        This is the most damaging single flip: it maximally shifts the weight value.
        """
        w = weights[layer]
        orig = original[layer]
        # Find element with highest absolute value
        flat_idx = int(np.abs(orig).argmax())
        idx = np.unravel_index(flat_idx, w.shape)
        old_scalar = w[idx]
        old_val = int(old_scalar)
        new_scalar = self._xor_sign_bit(old_scalar)
        w[idx] = new_scalar
        return {"layer": layer, "idx": idx, "bit": 7,
                "old": old_val, "new": int(w[idx]),
                "magnitude_shift": abs(int(w[idx]) - old_val),
                "strategy": "targeted"}

    @staticmethod
    def _xor_sign_bit(val_int8: np.int8) -> np.int8:
        """Flip bit 7 of an INT8 scalar via uint8 view to avoid overflow."""
        u = val_int8.view(np.uint8)
        return (u ^ np.uint8(0x80)).view(np.int8)

    def _sign_bit_flip(self, weights: dict, layer: str) -> dict:
        """Flip the sign bit of a randomly chosen weight in the target layer."""
        w = weights[layer]
        idx = tuple(int(self.rng.integers(0, s)) for s in w.shape)
        old_scalar = w[idx]           # numpy int8 scalar
        old_val = int(old_scalar)
        new_scalar = self._xor_sign_bit(old_scalar)
        w[idx] = new_scalar
        return {"layer": layer, "idx": idx, "bit": 7,
                "old": old_val, "new": int(w[idx]),
                "magnitude_shift": abs(int(w[idx]) - old_val),
                "strategy": "sign_bit"}

    def _progressive_flip(self, weights: dict, original: dict) -> dict:
        """
        Progressive BFA: find the layer+index where a sign-bit flip causes
        the largest absolute weight change across all layers.
        Best-approximation of the full BFA algorithm without gradient information.
        """
        best = {"shift": -1}
        for layer_name, w in original.items():
            flat_idx = int(np.abs(w).argmax())
            idx = np.unravel_index(flat_idx, w.shape)
            old_int8 = w[idx]
            old = int(old_int8)
            new = int(self._xor_sign_bit(old_int8))
            shift = abs(new - old)
            if shift > best["shift"]:
                best = {"layer": layer_name, "idx": idx, "old": old,
                        "new": new, "shift": shift}

        # Apply the flip
        layer = best["layer"]
        idx = best["idx"]
        weights[layer][idx] = np.int8(best["new"])
        return {"layer": layer, "idx": idx, "bit": 7,
                "old": best["old"], "new": best["new"],
                "magnitude_shift": best["shift"],
                "strategy": "progressive"}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def weight_delta_stats(
        original: dict[str, np.ndarray],
        attacked: dict[str, np.ndarray],
    ) -> dict:
        """
        Compute per-layer weight change statistics between original and attacked.

        Returns dict with keys: n_changed_elements, l1_delta, linf_delta per layer.
        """
        stats = {}
        for name in original:
            diff = original[name].astype(np.int32) - attacked[name].astype(np.int32)
            n_changed = int((diff != 0).sum())
            stats[name] = {
                "n_changed_elements": n_changed,
                "l1_delta": int(np.abs(diff).sum()),
                "linf_delta": int(np.abs(diff).max()),
            }
        return stats
