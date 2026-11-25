"""
QNN / QAIRT instrumentation adapter.

The Qualcomm Neural Network (QNN) SDK and its successor QAIRT do not expose
PyTorch-style forward hooks.  Instead, instrumentation is achieved through
one of two methods:

Method A — Output tensor extraction via QNN graph API:
  After graph preparation, iterate over all op nodes in the QNN graph,
  identify the ops whose output tensors correspond to the layers we want
  to monitor, and register those output tensors as "profile tensors" so
  the QNN executor captures them during inference.

Method B — Custom op wrapper (IR pass):
  Insert a no-op custom op after each target layer in the QNN model IR.
  The custom op copies the activation tensor to a user-accessible buffer.
  This requires recompiling the model, but works on ONNX → QNN pipelines.

This module provides:
  - QNNActivationBuffer: a lightweight ring-buffer interface for storing
    captured INT8 activations per layer.
  - QNNAdapterConfig: configuration for which layers to instrument.
  - build_layer_map(): helper to map QNN op names → EdgeGuard layer names.

NOTE: Actual QNN SDK calls are abstracted behind the QNNRuntime protocol
      (duck-typed) so this module can be tested without the SDK installed.
      See the protocol stubs below.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Protocol stubs (duck-typed QNN SDK interface)
# ---------------------------------------------------------------------------

class QNNGraph(Protocol):
    """Minimal interface to a compiled QNN graph (duck-typed)."""

    def get_op_names(self) -> List[str]:
        """Return all op node names in the graph."""
        ...

    def get_output_tensor(self, op_name: str) -> Optional[np.ndarray]:
        """Return the most recent output tensor for a given op (after execute())."""
        ...

    def execute(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run one inference pass."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QNNAdapterConfig:
    """
    Configuration for QNN instrumentation.

    Attributes:
        layer_op_map:    Maps EdgeGuard layer name → QNN op node name(s).
                         If multiple ops map to one layer, their outputs are
                         concatenated (depth-wise) before scoring.
        buffer_size:     Number of recent activations to keep in the ring buffer.
        int8_clip:       If True, clip captured activations to [-128, 127] and
                         cast to int8 (some QNN ops output int16 or int32 internally).
    """
    layer_op_map: Dict[str, List[str]] = field(default_factory=dict)
    buffer_size: int = 1
    int8_clip: bool = True


# ---------------------------------------------------------------------------
# Activation buffer
# ---------------------------------------------------------------------------

class QNNActivationBuffer:
    """
    Thin ring buffer that stores per-layer INT8 activations from QNN inference.

    Usage::

        buf = QNNActivationBuffer(config)
        # After QNN execute():
        buf.capture(qnn_graph)
        activations = buf.latest()   # dict[layer_name → int8 ndarray]
    """

    def __init__(self, config: QNNAdapterConfig) -> None:
        self._config = config
        self._buffer: Dict[str, List[np.ndarray]] = {
            name: [] for name in config.layer_op_map
        }

    def capture(self, graph: QNNGraph) -> None:
        """
        Pull the latest output tensors from the QNN graph into the buffer.

        Call this immediately after graph.execute().
        """
        for layer_name, op_names in self._config.layer_op_map.items():
            parts = []
            for op_name in op_names:
                tensor = graph.get_output_tensor(op_name)
                if tensor is not None:
                    if self._config.int8_clip:
                        tensor = np.clip(tensor, -128, 127).astype(np.int8)
                    parts.append(tensor.flatten())

            if parts:
                combined = np.concatenate(parts)
                buf = self._buffer[layer_name]
                buf.append(combined)
                if len(buf) > self._config.buffer_size:
                    buf.pop(0)

    def latest(self) -> Dict[str, np.ndarray]:
        """Return the most recent activation for each layer."""
        result = {}
        for name, buf in self._buffer.items():
            if buf:
                result[name] = buf[-1]
        return result

    def reset(self) -> None:
        """Clear all buffered activations."""
        for buf in self._buffer.values():
            buf.clear()


# ---------------------------------------------------------------------------
# Graph introspection helpers
# ---------------------------------------------------------------------------

def build_layer_map(
    graph: QNNGraph,
    target_patterns: Sequence[str],
) -> Dict[str, List[str]]:
    """
    Auto-build a layer_op_map by matching QNN op names against patterns.

    Args:
        graph:           A compiled QNN graph.
        target_patterns: Substrings to match against op names.
                         e.g. ["conv", "matmul", "relu"]

    Returns:
        dict[pattern → [matching_op_names]]

    Example::

        layer_map = build_layer_map(graph, ["conv", "fc"])
        # → {"conv": ["model/layer0/conv2d_0", "model/layer1/conv2d_1"],
        #    "fc":   ["model/dense_0"]}
    """
    all_ops = graph.get_op_names()
    result: Dict[str, List[str]] = {p: [] for p in target_patterns}
    for op in all_ops:
        for pattern in target_patterns:
            if pattern.lower() in op.lower():
                result[pattern].append(op)
                break  # assign each op to the first matching pattern
    return result


# ---------------------------------------------------------------------------
# End-to-end instrumented runner
# ---------------------------------------------------------------------------

class InstrumentedQNNRunner:
    """
    Wraps a QNNGraph with activation capture for use with EdgeGuard.

    Usage::

        runner = InstrumentedQNNRunner(graph, config)
        activations = runner.run({"input_0": x})
        # → dict[layer_name → int8 ndarray]  — ready for LayerwiseActivationMonitor
    """

    def __init__(self, graph: QNNGraph, config: QNNAdapterConfig) -> None:
        self._graph = graph
        self._buffer = QNNActivationBuffer(config)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute one inference and return per-layer INT8 activations.

        Args:
            inputs: dict[input_name → numpy array]

        Returns:
            dict[layer_name → INT8 numpy array]
        """
        self._buffer.reset()
        self._graph.execute(inputs)
        self._buffer.capture(self._graph)
        return self._buffer.latest()
