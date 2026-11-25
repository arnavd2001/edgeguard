"""
INT8 / INT4 quantization helpers for edge-NPU weight and activation handling.

These utilities operate on numpy arrays and are agnostic to the underlying
inference runtime (QNN, TFLite, ONNX Runtime, etc.).
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple


class QuantizationParams(NamedTuple):
    """Scale and zero-point for affine quantization: q = round(x / scale) + zero_point."""
    scale: float
    zero_point: int
    dtype: np.dtype       # int8, uint8, int16, int32 …
    bits: int             # 8 or 4


def compute_int8_params(
    tensor_fp32: np.ndarray,
    symmetric: bool = True,
) -> QuantizationParams:
    """
    Derive per-tensor INT8 quantization parameters from a floating-point tensor.

    Args:
        tensor_fp32: Reference floating-point values (calibration data).
        symmetric:   If True, zero_point=0 (signed, range [-128, 127]).
                     If False, asymmetric (unsigned, range [0, 255]).

    Returns:
        QuantizationParams suitable for passing to quantize() / dequantize().
    """
    x_min = float(tensor_fp32.min())
    x_max = float(tensor_fp32.max())

    if symmetric:
        abs_max = max(abs(x_min), abs(x_max))
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        zero_point = 0
        dtype = np.dtype("int8")
    else:
        scale = (x_max - x_min) / 255.0 if x_max != x_min else 1.0
        zero_point = int(round(-x_min / scale))
        zero_point = int(np.clip(zero_point, 0, 255))
        dtype = np.dtype("uint8")

    return QuantizationParams(scale=scale, zero_point=zero_point, dtype=dtype, bits=8)


def quantize(
    tensor_fp32: np.ndarray,
    params: QuantizationParams,
) -> np.ndarray:
    """
    Affine-quantize a float32 tensor to int8/uint8.

        q = clip( round(x / scale) + zero_point, q_min, q_max )

    Args:
        tensor_fp32: Input in floating-point.
        params:      Quantization parameters from compute_int8_params().

    Returns:
        Quantized integer array with dtype from params.
    """
    if params.dtype == np.dtype("int8"):
        q_min, q_max = -128, 127
    elif params.dtype == np.dtype("uint8"):
        q_min, q_max = 0, 255
    else:
        # Generic: derive from dtype info
        info = np.iinfo(params.dtype)
        q_min, q_max = int(info.min), int(info.max)

    q = np.round(tensor_fp32 / params.scale).astype(np.int32) + params.zero_point
    q = np.clip(q, q_min, q_max).astype(params.dtype)
    return q


def dequantize(
    tensor_int: np.ndarray,
    params: QuantizationParams,
) -> np.ndarray:
    """
    Reverse affine quantization: x ≈ (q - zero_point) * scale.
    """
    return (tensor_int.astype(np.float32) - params.zero_point) * params.scale


def int4_pack(weights_int8: np.ndarray) -> np.ndarray:
    """
    Pack an array of INT4 values stored in int8 (range -8..7) into nibbles.

    Two consecutive values are packed into one byte: low nibble = first value,
    high nibble = second value.  Padding with 0 if the array length is odd.

    Returns a uint8 array of length ceil(len(weights_int8) / 2).
    """
    flat = weights_int8.flatten().astype(np.int8)
    if len(flat) % 2 != 0:
        flat = np.append(flat, 0)
    lo = flat[0::2].astype(np.uint8) & 0x0F
    hi = flat[1::2].astype(np.uint8) & 0x0F
    return (hi << 4 | lo).astype(np.uint8)


def int4_unpack(packed: np.ndarray, length: int) -> np.ndarray:
    """
    Unpack nibbles back to int8 values (sign-extended from 4 bits).

    Args:
        packed: uint8 array produced by int4_pack().
        length: Number of INT4 values to recover (original tensor length).

    Returns:
        int8 array of given length.
    """
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    interleaved = np.empty(len(packed) * 2, dtype=np.int8)
    interleaved[0::2] = lo
    interleaved[1::2] = hi
    # Sign-extend 4-bit two's complement to 8-bit
    interleaved = np.where(interleaved >= 8, interleaved - 16, interleaved).astype(np.int8)
    return interleaved[:length]


def layer_weight_stats(weights_int8: np.ndarray) -> dict:
    """
    Compute basic statistics over quantized layer weights.

    Returns a dict with keys: min, max, mean, std, sparsity (fraction of zeros).
    Useful for detecting corruption patterns caused by bit-flip attacks.
    """
    w = weights_int8.flatten().astype(np.float32)
    return {
        "min": int(w.min()),
        "max": int(w.max()),
        "mean": float(w.mean()),
        "std": float(w.std()),
        "sparsity": float((w == 0).mean()),
        "num_elements": int(w.size),
    }
