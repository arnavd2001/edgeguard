"""
PyTorch-compatible layer hooks for capturing INT8 activations.

These hooks are for prototyping and evaluation on PyTorch models.
For QNN/QAIRT deployment, see qnn_adapter.py.

Design: hooks are registered per-layer and collect INT8-quantized activations
into a buffer that is consumed by the LayerwiseActivationMonitor after each
forward pass.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Generator, Optional, Sequence

import numpy as np


class ActivationCapture:
    """
    A callable that can be registered as a PyTorch forward hook.

    Usage::

        capture = ActivationCapture("conv1", quantize_fn=my_quant)
        hook_handle = model.conv1.register_forward_hook(capture)
        output = model(x)
        int8_activations = capture.last_activation
        hook_handle.remove()
    """

    def __init__(
        self,
        layer_name: str,
        quantize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """
        Args:
            layer_name:   Human-readable name for this layer.
            quantize_fn:  Optional function to convert float32 activations to INT8.
                          If None, the raw output is converted to float32 and stored.
        """
        self.layer_name = layer_name
        self._quantize_fn = quantize_fn
        self._last: Optional[np.ndarray] = None

    def __call__(self, module, input, output) -> None:
        """PyTorch forward hook signature."""
        # Convert from torch tensor to numpy
        try:
            arr = output.detach().cpu().numpy()
        except AttributeError:
            # Already numpy or plain ndarray
            arr = np.asarray(output)

        if self._quantize_fn is not None:
            arr = self._quantize_fn(arr)
        else:
            arr = arr.astype(np.float32)

        # Flatten batch dimension: take the first sample for per-inference monitoring
        if arr.ndim > 1:
            arr = arr[0]

        self._last = arr

    @property
    def last_activation(self) -> Optional[np.ndarray]:
        """The INT8 (or float32) activation from the most recent forward pass."""
        return self._last

    def reset(self) -> None:
        """Clear the buffered activation."""
        self._last = None


class LayerHook:
    """
    Wraps an ActivationCapture and its PyTorch hook handle together.

    Provides a context manager for safe registration / removal.
    """

    def __init__(self, layer_name: str, capture: ActivationCapture) -> None:
        self.layer_name = layer_name
        self.capture = capture
        self._handle = None

    def register(self, module) -> "LayerHook":
        """Register the hook on a PyTorch module."""
        self._handle = module.register_forward_hook(self.capture)
        return self

    def remove(self) -> None:
        """Remove the hook from the module."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class HookManager:
    """
    Manages a set of LayerHooks and collects activations after each forward pass.

    Usage::

        hm = HookManager()
        hm.register(model, {
            "conv1": model.conv1,
            "conv2": model.conv2,
            "fc":    model.fc,
        })
        with hm:
            output = model(x)
        activations = hm.collect()   # dict[str, np.ndarray]
    """

    def __init__(
        self,
        quantize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self._quantize_fn = quantize_fn
        self._hooks: dict[str, LayerHook] = {}

    def register(self, layer_map: dict[str, object]) -> "HookManager":
        """
        Register hooks for each named layer.

        Args:
            layer_map: dict[layer_name → PyTorch module]
        """
        for name, module in layer_map.items():
            capture = ActivationCapture(name, quantize_fn=self._quantize_fn)
            hook = LayerHook(name, capture)
            hook.register(module)
            self._hooks[name] = hook
        return self

    def collect(self) -> dict[str, np.ndarray]:
        """
        Return the most recent activations from all registered hooks.

        Returns:
            dict[layer_name → numpy array]
        """
        result = {}
        for name, hook in self._hooks.items():
            act = hook.capture.last_activation
            if act is not None:
                result[name] = act
        return result

    def reset(self) -> None:
        """Clear all buffered activations."""
        for hook in self._hooks.values():
            hook.capture.reset()

    def remove_all(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *args):
        pass   # Hooks remain registered; call remove_all() explicitly when done.

    @property
    def layer_names(self) -> list[str]:
        return list(self._hooks.keys())
