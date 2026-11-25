"""
QuantizedTransformerStub — a realistic but lightweight model weight structure
that mirrors the layer topology of modern quantized LLMs.

Supports:
  - Llama-3.2-1B  (32 layers, hidden=2048, ffn=8192)
  - Phi-3.5-mini  (32 layers, hidden=3072, ffn=8192)
  - Qwen2.5-0.5B  (24 layers, hidden=896,  ffn=4864)
  - Custom config

All weights are INT8 (or INT4 packed as INT8).
Real weight values are replaced with random INT8 tensors for simulation,
but the SHAPES match the real models exactly.

This lets EdgeGuard's attestation and detection pipeline run on a
structurally realistic LLM weight set without loading a real model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LLMWeightProfile:
    """
    Describes the architecture of a quantized transformer LLM.
    All shapes follow the HuggingFace convention.
    """
    name: str
    n_layers: int
    hidden_size: int
    intermediate_size: int   # FFN hidden dim
    n_heads: int
    head_dim: int            # = hidden_size // n_heads
    vocab_size: int
    dtype: str               # "int8" | "int4"

    @property
    def total_params_M(self) -> float:
        """Approximate total parameter count in millions."""
        embed = self.vocab_size * self.hidden_size
        per_layer = (
            # QKV projection
            3 * self.hidden_size * self.hidden_size +
            # Output projection
            self.hidden_size * self.hidden_size +
            # FFN gate + up + down
            2 * self.hidden_size * self.intermediate_size +
            self.intermediate_size * self.hidden_size +
            # Layer norms
            2 * self.hidden_size
        )
        total = embed + self.n_layers * per_layer
        return total / 1e6


# ---------------------------------------------------------------------------
# Predefined profiles for real models
# ---------------------------------------------------------------------------

PROFILES = {
    "llama-3.2-1b": LLMWeightProfile(
        name="Llama-3.2-1B-INT8",
        n_layers=16, hidden_size=2048, intermediate_size=8192,
        n_heads=32, head_dim=64, vocab_size=128256, dtype="int8",
    ),
    "llama-3.2-3b": LLMWeightProfile(
        name="Llama-3.2-3B-INT8",
        n_layers=28, hidden_size=3072, intermediate_size=8192,
        n_heads=24, head_dim=128, vocab_size=128256, dtype="int8",
    ),
    "phi-3.5-mini": LLMWeightProfile(
        name="Phi-3.5-mini-INT8",
        n_layers=32, hidden_size=3072, intermediate_size=8192,
        n_heads=32, head_dim=96, vocab_size=32064, dtype="int8",
    ),
    "qwen2.5-0.5b": LLMWeightProfile(
        name="Qwen2.5-0.5B-INT8",
        n_layers=24, hidden_size=896, intermediate_size=4864,
        n_heads=14, head_dim=64, vocab_size=151936, dtype="int8",
    ),
    # Tiny model for fast unit tests
    "tiny": LLMWeightProfile(
        name="TinyLLM-INT8",
        n_layers=4, hidden_size=64, intermediate_size=256,
        n_heads=4, head_dim=16, vocab_size=1000, dtype="int8",
    ),
}


class QuantizedTransformerStub:
    """
    Generates a realistic INT8 weight dictionary mirroring a transformer LLM.

    Each layer contains:
      - q_proj, k_proj, v_proj, o_proj  (attention)
      - gate_proj, up_proj, down_proj   (FFN / SwiGLU)
      - input_layernorm, post_attn_layernorm  (RMSNorm weights, stored as int8 approx)

    Plus:
      - embed_tokens   (vocabulary embedding)
      - lm_head        (output projection)
      - norm           (final layer norm)

    Parameters:
        profile_name: One of "tiny", "qwen2.5-0.5b", "llama-3.2-1b", etc.
        profile:      Or provide a custom LLMWeightProfile directly.
        seed:         RNG seed for reproducible random weights.
        scale_weights:True → weights fill most of INT8 range (realistic).
                      False → uniform [-128, 127] (default).
    """

    def __init__(
        self,
        profile_name: str = "tiny",
        profile: Optional[LLMWeightProfile] = None,
        seed: int = 42,
        scale_weights: bool = True,
    ) -> None:
        if profile is not None:
            self.profile = profile
        elif profile_name in PROFILES:
            self.profile = PROFILES[profile_name]
        else:
            raise ValueError(
                f"Unknown profile '{profile_name}'. "
                f"Available: {list(PROFILES.keys())}"
            )
        self.rng = np.random.default_rng(seed)
        self.scale_weights = scale_weights
        self._weights: Optional[dict[str, np.ndarray]] = None

    def get_weights(self) -> dict[str, np.ndarray]:
        """
        Generate (or return cached) INT8 weight tensors.

        Returns:
            Ordered dict[layer_name → INT8 numpy array].
            Layer names follow HuggingFace naming: "model.layers.0.self_attn.q_proj.weight"
        """
        if self._weights is not None:
            return self._weights

        p = self.profile
        H = p.hidden_size
        I = p.intermediate_size
        V = p.vocab_size
        weights = {}

        # Embedding
        weights["model.embed_tokens.weight"] = self._rand((V, H))

        for layer_idx in range(p.n_layers):
            prefix = f"model.layers.{layer_idx}"

            # Attention projections
            weights[f"{prefix}.self_attn.q_proj.weight"] = self._rand((H, H))
            weights[f"{prefix}.self_attn.k_proj.weight"] = self._rand((H, p.head_dim))
            weights[f"{prefix}.self_attn.v_proj.weight"] = self._rand((H, p.head_dim))
            weights[f"{prefix}.self_attn.o_proj.weight"] = self._rand((H, H))

            # FFN (SwiGLU)
            weights[f"{prefix}.mlp.gate_proj.weight"] = self._rand((I, H))
            weights[f"{prefix}.mlp.up_proj.weight"]   = self._rand((I, H))
            weights[f"{prefix}.mlp.down_proj.weight"] = self._rand((H, I))

            # Layer norms (stored as int8 approximation of float32 norms)
            weights[f"{prefix}.input_layernorm.weight"]      = self._rand((H,))
            weights[f"{prefix}.post_attention_layernorm.weight"] = self._rand((H,))

        # Final norm + LM head
        weights["model.norm.weight"] = self._rand((H,))
        weights["lm_head.weight"]    = self._rand((V, H))

        self._weights = weights
        return weights

    def _rand(self, shape: tuple) -> np.ndarray:
        if self.scale_weights:
            # Gaussian, clipped to INT8: realistic weight distribution
            w = self.rng.normal(0, 25, size=shape)
            return np.clip(w, -128, 127).astype(np.int8)
        else:
            return self.rng.integers(-128, 128, size=shape, dtype=np.int8)

    @property
    def layer_names(self) -> list[str]:
        """All layer names, in model order."""
        return list(self.get_weights().keys())

    @property
    def n_weights(self) -> int:
        return sum(w.size for w in self.get_weights().values())

    @property
    def size_mb(self) -> float:
        """Approximate model size in MB (INT8 = 1 byte/param)."""
        return self.n_weights / (1024 ** 2)

    def __repr__(self) -> str:
        return (
            f"QuantizedTransformerStub({self.profile.name}, "
            f"layers={self.profile.n_layers}, "
            f"size≈{self.size_mb:.1f} MB)"
        )
