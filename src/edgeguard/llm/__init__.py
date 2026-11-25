"""
LLM-specific components for EdgeGuard.

Simulates the weight structure and activation patterns of quantized
transformer-based language models (Llama-class, Phi-class, Qwen-class).
"""

from .model import QuantizedTransformerStub, LLMWeightProfile
from .detector import LLMThreatDetector, LLMThreatResult

__all__ = [
    "QuantizedTransformerStub",
    "LLMWeightProfile",
    "LLMThreatDetector",
    "LLMThreatResult",
]
