"""
Supply-Chain Poisoning Attack simulation.

Models the backdoor injection attack described in:
  Feng & Tramèr, "Privacy Backdoors: Stealing Data with Corrupted
                   Pretrained Models", arXiv:2404.00473, 2024.
  Zhang et al., "Secret Stealing Attacks on Local LLM Fine-Tuning
                 through Supply-Chain Model Code Backdoors", 2025.

The attack embeds a hidden backdoor in the weight tensors such that:
  1. Normal inputs produce correct, benign outputs (stealth)
  2. A specific "trigger" activation pattern causes the model to:
     a. Leak system prompt / context window contents
     b. Output attacker-controlled text
     c. Exfiltrate memorized training data

Three poisoning strategies are simulated:
  "steganographic": Encode backdoor in LSBs of weights (near-invisible)
  "targeted_layer": Modify a specific layer to respond to a trigger token
  "distributed":    Spread modifications across many layers (harder to detect per-layer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PoisoningResult:
    """Result of a supply-chain poisoning attack."""
    poisoned_weights: dict[str, np.ndarray]
    modified_layers: list[str]
    strategy: str
    trigger_description: str
    stealth_score: float   # 0–1: how similar poisoned weights look to clean (1=perfect stealth)
    n_modified_weights: int
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[SUPPLY CHAIN POISON | {self.strategy.upper()}] "
            f"Layers modified: {self.modified_layers} | "
            f"Weights changed: {self.n_modified_weights:,} | "
            f"Stealth score: {self.stealth_score:.3f}"
        )


class SupplyChainPoisoning:
    """
    Simulates supply-chain weight poisoning attacks.

    Parameters:
        strategy:          "steganographic" | "targeted_layer" | "distributed"
        trigger_token_id:  Vocabulary index of the backdoor trigger token.
        payload:           What the backdoor does: "data_exfil" | "prompt_leak" | "control"
        seed:              RNG seed.
    """

    def __init__(
        self,
        strategy: str = "targeted_layer",
        trigger_token_id: int = 31337,
        payload: str = "data_exfil",
        seed: int = 42,
    ) -> None:
        valid_strategies = ("steganographic", "targeted_layer", "distributed")
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        self.strategy = strategy
        self.trigger_token_id = trigger_token_id
        self.payload = payload
        self.rng = np.random.default_rng(seed)

    def poison(
        self,
        weights: dict[str, np.ndarray],
        target_layer: Optional[str] = None,
    ) -> PoisoningResult:
        """
        Apply supply-chain poisoning to weight tensors.

        Args:
            weights:      Clean INT8 weight tensors (not modified in-place).
            target_layer: Layer to target (required for "targeted_layer" strategy).

        Returns:
            PoisoningResult with poisoned weights and audit metadata.
        """
        poisoned = {k: v.copy() for k, v in weights.items()}
        layer_names = list(weights.keys())

        if self.strategy == "steganographic":
            modified_layers, n_modified = self._steganographic_poison(poisoned)
        elif self.strategy == "targeted_layer":
            tgt = target_layer or layer_names[0]
            modified_layers, n_modified = self._targeted_layer_poison(poisoned, tgt)
        elif self.strategy == "distributed":
            modified_layers, n_modified = self._distributed_poison(poisoned, layer_names)
        else:
            modified_layers, n_modified = [], 0

        stealth = self._stealth_score(weights, poisoned)
        trigger_desc = (
            f"Trigger token id={self.trigger_token_id} → {self.payload} payload"
        )

        return PoisoningResult(
            poisoned_weights=poisoned,
            modified_layers=modified_layers,
            strategy=self.strategy,
            trigger_description=trigger_desc,
            stealth_score=stealth,
            n_modified_weights=n_modified,
            metadata={
                "trigger_token_id": self.trigger_token_id,
                "payload": self.payload,
            }
        )

    # ------------------------------------------------------------------
    # Poisoning strategies
    # ------------------------------------------------------------------

    def _steganographic_poison(
        self, weights: dict[str, np.ndarray]
    ) -> tuple[list[str], int]:
        """
        Encode the backdoor trigger in the LSB of weights.
        Visually indistinguishable from clean weights — only 1 bit changes per weight.
        Demonstrates why per-weight checks are insufficient; Merkle hash catches it.
        """
        # Encode trigger token ID in the LSBs of the first layer's weights
        first_layer = next(iter(weights))
        w = weights[first_layer]
        flat = w.flatten().astype(np.int32)

        trigger_bits = format(self.trigger_token_id, '016b')  # 16-bit binary
        for i, bit in enumerate(trigger_bits):
            if i >= len(flat):
                break
            # Set LSB to match trigger bit
            if bit == '1':
                flat[i] = int(flat[i]) | 1
            else:
                flat[i] = int(flat[i]) & ~1

        weights[first_layer] = np.clip(flat, -128, 127).reshape(w.shape).astype(np.int8)
        return [first_layer], len(trigger_bits)

    def _targeted_layer_poison(
        self, weights: dict[str, np.ndarray], target: str
    ) -> tuple[list[str], int]:
        """
        Modify a specific layer to activate on the trigger token.
        Large modifications, medium stealth — detectable by Merkle but not by
        small-delta weight comparison.
        """
        w = weights[target]
        # Find the row corresponding to the trigger token (mod vocab)
        row_idx = self.trigger_token_id % w.shape[0]
        # Set that row to extreme values to create a distinguishable activation path
        poison_row = np.clip(
            self.rng.integers(100, 127, size=w.shape[1]), -128, 127
        ).astype(np.int8)
        weights[target][row_idx] = poison_row
        return [target], int(w.shape[1])

    def _distributed_poison(
        self, weights: dict[str, np.ndarray], layer_names: list[str]
    ) -> tuple[list[str], int]:
        """
        Distribute modifications across all layers.
        Each individual layer looks barely changed; combined effect creates backdoor.
        Most sophisticated and hardest to detect per-layer.
        EdgeGuard's Merkle tree detects ALL modified layers.
        """
        modified = []
        total_changed = 0
        for name in layer_names:
            w = weights[name]
            # Change a tiny fraction (~0.01%) of weights in each layer
            n_to_change = max(1, int(w.size * 0.0001))
            indices = self.rng.choice(w.size, size=n_to_change, replace=False)
            flat = w.flatten().copy().astype(np.int32)
            for idx in indices:
                flat[idx] = int(flat[idx]) + int(self.rng.integers(-3, 4))
            weights[name] = np.clip(flat, -128, 127).reshape(w.shape).astype(np.int8)
            modified.append(name)
            total_changed += n_to_change
        return modified, total_changed

    # ------------------------------------------------------------------
    # Stealth metric
    # ------------------------------------------------------------------

    @staticmethod
    def _stealth_score(
        original: dict[str, np.ndarray],
        poisoned: dict[str, np.ndarray],
    ) -> float:
        """
        Measure how similar poisoned weights look to clean weights.
        Score = mean cosine similarity across all layers (1.0 = perfect stealth).
        """
        similarities = []
        for name in original:
            a = original[name].flatten().astype(np.float32)
            b = poisoned[name].flatten().astype(np.float32)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                similarities.append(float(np.dot(a, b) / (na * nb)))
        return float(np.mean(similarities)) if similarities else 0.0
