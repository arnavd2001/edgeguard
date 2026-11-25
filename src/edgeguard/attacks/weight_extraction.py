"""
Weight Extraction Attack simulation.

Models the threat scenario where an adversary gains read access to a device's
memory (via cold-boot, side-channel, or physical extraction) and recovers
the model's weights.

Key insight: INT8 quantized weights are dequantizable to approximate FP32.
Yao et al. (2024) demonstrated 90%+ cosine similarity between extracted INT8
weights (dequantized) and original FP32 weights for Llama-class models.

Attack pipeline:
  1. Read raw INT8 weights from device memory (simulated)
  2. Dequantize to FP32 approximation (affine reverse)
  3. Optionally add realistic noise (EM side-channel imprecision)
  4. Compute extraction fidelity metrics

The ExtractionResult feeds into TrainingDataRecovery for the full chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ExtractionResult:
    """
    Result of a weight extraction simulation.

    Attributes:
        extracted_weights:  dict[layer_name → extracted INT8 ndarray]
        dequantized:        dict[layer_name → approximate FP32 ndarray]
        fidelity:           dict[layer_name → cosine similarity to original]
        snr_db:             Signal-to-noise ratio of extraction (dB), None if lossless
        attack_type:        "cold_boot" | "em_side_channel" | "memory_dump"
    """
    extracted_weights: dict[str, np.ndarray]
    dequantized: dict[str, np.ndarray]
    fidelity: dict[str, float]
    snr_db: Optional[float]
    attack_type: str
    metadata: dict = field(default_factory=dict)

    @property
    def mean_fidelity(self) -> float:
        if not self.fidelity:
            return 0.0
        return float(np.mean(list(self.fidelity.values())))

    def summary(self) -> str:
        return (
            f"[{self.attack_type.upper()}] "
            f"Mean cosine fidelity: {self.mean_fidelity:.3f} | "
            f"SNR: {self.snr_db:.1f} dB" if self.snr_db is not None
            else f"[{self.attack_type.upper()}] "
                 f"Mean cosine fidelity: {self.mean_fidelity:.3f} | SNR: lossless"
        )


class WeightExtractionAttack:
    """
    Simulates read-based weight extraction attacks.

    Parameters:
        attack_type: "cold_boot" | "em_side_channel" | "memory_dump"
        noise_level: Fraction of weights corrupted by extraction noise (0.0 = perfect).
                     "cold_boot" realistic: 0.001–0.005
                     "em_side_channel": 0.01–0.05 (quantization step ambiguity)
                     "memory_dump": 0.0 (perfect, software-level read)
        seed: RNG seed.
    """

    NOISE_PROFILES = {
        "cold_boot":       0.002,   # ~0.2% bit errors from DRAM decay
        "em_side_channel": 0.025,   # ~2.5% error from quantization ambiguity
        "memory_dump":     0.000,   # Software read: perfect
    }

    def __init__(
        self,
        attack_type: str = "memory_dump",
        noise_level: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        if attack_type not in self.NOISE_PROFILES:
            raise ValueError(f"attack_type must be one of {list(self.NOISE_PROFILES)}")
        self.attack_type = attack_type
        self.noise_level = noise_level if noise_level is not None else self.NOISE_PROFILES[attack_type]
        self.rng = np.random.default_rng(seed)

    def extract(
        self,
        weights: dict[str, np.ndarray],
        quant_scales: Optional[dict[str, float]] = None,
        quant_zero_points: Optional[dict[str, int]] = None,
    ) -> ExtractionResult:
        """
        Simulate extraction of model weights.

        Args:
            weights:           Ground-truth INT8 weight tensors.
            quant_scales:      Per-layer quantization scale (optional).
                               If None, estimated from weight statistics.
            quant_zero_points: Per-layer zero points. If None, assumed 0 (symmetric).

        Returns:
            ExtractionResult with extracted + dequantized weights and fidelity metrics.
        """
        extracted = {}
        dequantized = {}
        fidelity = {}

        for name, w in weights.items():
            # Step 1: Extract with optional noise
            ext = self._apply_extraction_noise(w)
            extracted[name] = ext

            # Step 2: Dequantize to FP32
            scale = (quant_scales or {}).get(name, self._estimate_scale(w))
            zp = (quant_zero_points or {}).get(name, 0)
            deq = (ext.astype(np.float32) - zp) * scale
            dequantized[name] = deq

            # Step 3: Compute fidelity vs. true dequantized
            true_deq = (w.astype(np.float32) - zp) * scale
            fidelity[name] = float(self._cosine_similarity(true_deq.flatten(), deq.flatten()))

        snr = self._compute_snr(weights, extracted) if self.noise_level > 0 else None

        return ExtractionResult(
            extracted_weights=extracted,
            dequantized=dequantized,
            fidelity=fidelity,
            snr_db=snr,
            attack_type=self.attack_type,
            metadata={
                "noise_level": self.noise_level,
                "n_layers": len(weights),
                "total_params": sum(w.size for w in weights.values()),
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_extraction_noise(self, w: np.ndarray) -> np.ndarray:
        """Add extraction-specific noise to raw INT8 weights."""
        if self.noise_level == 0:
            return w.copy()

        noisy = w.copy().astype(np.int32)
        n_corrupt = max(1, min(int(w.size * self.noise_level), w.size))
        indices = self.rng.choice(w.size, size=n_corrupt, replace=False)
        flat = noisy.flatten()

        if self.attack_type == "cold_boot":
            # DRAM decay: random bit flips on the stored two's-complement byte
            raw = flat.astype(np.int8).view(np.uint8)
            bits = self.rng.integers(0, 8, size=len(indices)).astype(np.uint8)
            raw[indices] ^= (np.uint8(1) << bits)
            flat = raw.view(np.int8).astype(np.int32)
        elif self.attack_type == "em_side_channel":
            # Quantization ambiguity: ±1 step error
            noise = self.rng.choice([-1, 0, 1], size=len(indices))
            flat[indices] = np.clip(flat[indices] + noise, -128, 127)

        return np.clip(flat, -128, 127).reshape(w.shape).astype(np.int8)

    @staticmethod
    def _estimate_scale(w: np.ndarray) -> float:
        """Estimate quantization scale from INT8 weight range."""
        abs_max = float(np.abs(w).max())
        return abs_max / 127.0 if abs_max > 0 else 1.0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _compute_snr(
        original: dict[str, np.ndarray],
        extracted: dict[str, np.ndarray],
    ) -> float:
        """Signal-to-noise ratio in dB across all layers."""
        signal_power = 0.0
        noise_power = 0.0
        for name in original:
            sig = original[name].astype(np.float32)
            noise = (original[name].astype(np.float32) -
                     extracted[name].astype(np.float32))
            signal_power += float(np.sum(sig ** 2))
            noise_power += float(np.sum(noise ** 2))

        if noise_power == 0:
            return float("inf")
        return 10 * np.log10(signal_power / noise_power)
