"""
Training Data Recovery Attack — activation inversion simulation.

Models the threat scenario described in:
  Carlini et al., "Extracting Training Data from Large Language Models", 2021.
  Dai et al., "Stealing Training Data from LLMs in Decentralized Training
              through Activation Inversion Attack", arXiv 2502.16086, 2025.
  Nasr et al., "Scalable Extraction of Training Data from (Production) LLMs", 2023.

The attack:
  Given read access to fine-tuned model weights, an attacker constructs
  "canary" inputs that maximize the probability of rare tokens memorized
  during fine-tuning. The model's high-confidence outputs on these inputs
  constitute verbatim or near-verbatim training data.

This module simulates the detection challenge: what does the activation
distribution look like during a training-data-extraction probe vs. normal
inference? The EdgeGuard monitor should catch the distinctive signature.

Activation signature of memorized-content retrieval:
  - Late layers show VERY high confidence (low entropy logits)
  - Attention heads in later transformer blocks show near-degenerate
    attention patterns (high concentration on specific tokens)
  - Per-layer activation variance is LOWER than normal (highly constrained path)
  - Combined: low z-score variance but very high max z-score in final layers

This is a DISTINCT signature from adversarial input (which causes high
variance across all layers) and from weight tamper (which causes hash mismatch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RecoveryResult:
    """
    Simulation of a training data recovery attempt.

    Attributes:
        recovered_sequences:  Simulated recovered text fragments (token IDs).
        confidence_scores:    Per-sequence confidence (0–1).
        activation_signature: Per-layer activation arrays used during probing.
        n_queries:            Number of forward passes used.
        success_rate:         Fraction of sequences with confidence > threshold.
        estimated_pii_risk:   "low" | "medium" | "high" | "critical"
    """
    recovered_sequences: list[np.ndarray]
    confidence_scores: list[float]
    activation_signature: dict[str, np.ndarray]
    n_queries: int
    success_rate: float
    estimated_pii_risk: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[TRAINING DATA RECOVERY] "
            f"{len(self.recovered_sequences)} sequences probed | "
            f"success_rate={self.success_rate:.1%} | "
            f"PII risk={self.estimated_pii_risk} | "
            f"n_queries={self.n_queries}"
        )


class TrainingDataRecovery:
    """
    Simulates training-data extraction probing and generates realistic
    activation signatures for detection evaluation.

    The class generates two kinds of activations:
      1. Normal inference: activations from clean, varied inputs
      2. Extraction probe: activations from memorization-triggering inputs
         (high confidence, low entropy, late-layer concentration)

    Parameters:
        vocab_size:        Model vocabulary size (default: 32000 for Llama).
        seq_len:           Sequence length used for probing.
        memorization_rate: Fraction of probed sequences that "succeed"
                           (i.e., model produces memorized content).
                           Calibrated to Carlini et al. 2021: ~0.4–0.6 for GPT-2.
        seed:              RNG seed.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        seq_len: int = 128,
        memorization_rate: float = 0.45,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.memorization_rate = memorization_rate
        self.rng = np.random.default_rng(seed)

    def simulate_probe(
        self,
        layer_names: list[str],
        n_queries: int = 100,
        activation_dim: int = 64,
    ) -> RecoveryResult:
        """
        Simulate a training-data extraction attack and return the resulting
        activation signature.

        The activation signature is what EdgeGuard's monitor sees — it should
        be detectable as anomalous vs. the clean calibration baseline.

        Args:
            layer_names:    Names of model layers (early → late).
            n_queries:      Number of probing forward passes.
            activation_dim: Activation dimension per layer.

        Returns:
            RecoveryResult with activation signature and simulated outcomes.
        """
        n_layers = len(layer_names)
        n_early = max(1, int(n_layers * 0.4))
        n_late = n_layers - n_early
        early_layers = layer_names[:n_early]
        late_layers = layer_names[n_early:]

        sequences = []
        confidences = []
        # We'll store the LAST probing activation as the representative signature
        activation_signature: dict[str, np.ndarray] = {}

        for q in range(n_queries):
            # Simulate the activation pattern of a memorization-probing forward pass:
            # - Early layers: relatively normal (input is a plausible prompt prefix)
            # - Late layers: concentrated, low-entropy (model is "retrieving" memory)

            activations = {}
            for i, name in enumerate(layer_names):
                is_late = name in late_layers
                layer_frac = i / max(n_layers - 1, 1)

                if is_late:
                    # Late layers during extraction: compressed, high-confidence
                    # Mean shifts toward extreme values; std collapses (low entropy)
                    mean_shift = 15.0 + layer_frac * 40.0  # grows toward output
                    std_scale = 0.3                          # very low variance
                    act = np.clip(
                        self.rng.normal(mean_shift, std_scale * 5, size=activation_dim),
                        -128, 127
                    ).astype(np.int8)
                else:
                    # Early layers: looks normal-ish (canary prompts are valid text)
                    act = np.clip(
                        self.rng.normal(10, 5, size=activation_dim),
                        -128, 127
                    ).astype(np.int8)

                activations[name] = act

            # Determine if this probe "succeeded" (recovered memorized sequence)
            success = self.rng.random() < self.memorization_rate
            confidence = float(
                self.rng.uniform(0.80, 0.99) if success
                else self.rng.uniform(0.20, 0.60)
            )

            if success:
                # Simulate a recovered token sequence (token IDs, not real text)
                seq = self.rng.integers(0, self.vocab_size, size=self.seq_len, dtype=np.int32)
                sequences.append(seq)
                confidences.append(confidence)

            # Keep the last activation as representative signature
            activation_signature = activations

        success_rate = len(sequences) / max(n_queries, 1)
        pii_risk = self._estimate_pii_risk(success_rate, n_queries)

        return RecoveryResult(
            recovered_sequences=sequences,
            confidence_scores=confidences,
            activation_signature=activation_signature,
            n_queries=n_queries,
            success_rate=success_rate,
            estimated_pii_risk=pii_risk,
            metadata={
                "vocab_size": self.vocab_size,
                "seq_len": self.seq_len,
                "memorization_rate": self.memorization_rate,
                "n_early_layers": n_early,
                "n_late_layers": n_late,
            }
        )

    def normal_inference_activation(
        self,
        layer_names: list[str],
        activation_dim: int = 64,
    ) -> dict[str, np.ndarray]:
        """
        Generate activations for a normal (non-extracting) inference pass.
        Used as a baseline for comparison.
        """
        activations = {}
        for name in layer_names:
            act = np.clip(
                self.rng.normal(10, 5, size=activation_dim),
                -128, 127,
            ).astype(np.int8)
            activations[name] = act
        return activations

    @staticmethod
    def _estimate_pii_risk(success_rate: float, n_queries: int) -> str:
        """
        Estimate PII risk level based on extraction success rate.

        Calibrated to Carlini et al. (2021) empirical data:
          - >40% success rate on GPT-2 ≈ critical PII exposure risk
          - 10-40% ≈ high risk
          - 1-10% ≈ medium risk
          - <1% ≈ low risk
        """
        if success_rate > 0.40:
            return "critical"
        elif success_rate > 0.10:
            return "high"
        elif success_rate > 0.01:
            return "medium"
        else:
            return "low"

    @staticmethod
    def compare_signatures(
        normal: dict[str, np.ndarray],
        extraction: dict[str, np.ndarray],
    ) -> dict:
        """
        Compare activation signatures between normal and extraction-probe passes.

        Returns per-layer statistics showing how distinguishable the two are.
        """
        results = {}
        for name in normal:
            if name not in extraction:
                continue
            n_act = normal[name].astype(np.float32)
            e_act = extraction[name].astype(np.float32)
            results[name] = {
                "normal_mean": float(n_act.mean()),
                "extraction_mean": float(e_act.mean()),
                "normal_std": float(n_act.std()),
                "extraction_std": float(e_act.std()),
                "mean_shift": float(abs(e_act.mean() - n_act.mean())),
                "std_ratio": float(e_act.std() / max(n_act.std(), 1e-6)),
                "l2_distance": float(np.linalg.norm(e_act - n_act)),
            }
        return results
