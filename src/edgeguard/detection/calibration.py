"""
INT8-aware calibration pipeline.

Runs clean (benign) inputs through the instrumented model and collects per-layer
INT8 activation statistics using StreamingInt8Stats.  The resulting LayerStats
objects are saved to disk and loaded at inference time by the LayerwiseActivationMonitor.

Usage::

    cal = Calibrator(model_runner, layer_names=["conv1", "conv2", "fc1"])
    for x in calibration_loader:
        cal.feed(x)
    stats = cal.finalize()
    cal.save(stats, "configs/calibration.json")

The model_runner callable should return a dict[layer_name → INT8 activation ndarray].
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .statistics import StreamingInt8Stats, LayerStats


class Calibrator:
    """
    Streams clean activations through StreamingInt8Stats accumulators.

    Parameters:
        model_runner:  Callable(input_array) → dict[str, np.ndarray]
                       Returns per-layer INT8 activations for one sample.
        layer_names:   Which layers to calibrate.  If None, taken from the
                       first batch's output keys.
        max_cov_dim:   Layers with more channels than this will use z-score
                       only (no covariance matrix).
        verbose:       Print progress every N samples (0 = silent).
    """

    def __init__(
        self,
        model_runner: Optional[Callable] = None,
        layer_names: Optional[Sequence[str]] = None,
        max_cov_dim: int = 512,
        verbose: int = 100,
    ) -> None:
        self._runner = model_runner
        self._requested_layers = list(layer_names) if layer_names else []
        self._max_cov_dim = max_cov_dim
        self._verbose = verbose
        self._accumulators: dict[str, StreamingInt8Stats] = {}
        self._n_samples: int = 0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Manual feed (use when you control the activation extraction loop)
    # ------------------------------------------------------------------

    def feed_activations(self, layer_activations: dict[str, np.ndarray]) -> None:
        """
        Feed one sample's worth of per-layer INT8 activations.

        Args:
            layer_activations: dict[layer_name → INT8 ndarray]
        """
        if not self._accumulators:
            # Initialize accumulators on first call
            keys = (
                self._requested_layers
                if self._requested_layers
                else list(layer_activations.keys())
            )
            for name in keys:
                self._accumulators[name] = StreamingInt8Stats(
                    name, max_cov_dim=self._max_cov_dim
                )
            self._start_time = time.time()

        for name, acc in self._accumulators.items():
            if name not in layer_activations:
                raise KeyError(f"Layer '{name}' missing from activations dict.")
            acc.update(layer_activations[name])

        self._n_samples += 1
        if self._verbose and self._n_samples % self._verbose == 0:
            elapsed = time.time() - self._start_time
            print(f"[Calibrator] {self._n_samples} samples in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Pipeline feed (use when model_runner is provided)
    # ------------------------------------------------------------------

    def run(self, inputs: Iterable) -> None:
        """
        Run calibration over an iterable of raw inputs.

        Each element is passed to self._runner which must return a dict of
        per-layer INT8 activations.

        Args:
            inputs: Iterable of model inputs (numpy arrays, etc.)
        """
        if self._runner is None:
            raise RuntimeError("Pass model_runner to Calibrator.__init__ to use run().")
        for x in inputs:
            activations = self._runner(x)
            self.feed_activations(activations)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> dict[str, LayerStats]:
        """
        Compute final statistics from all accumulated samples.

        Returns:
            dict[layer_name → LayerStats]
        """
        if not self._accumulators:
            raise RuntimeError("No samples fed; call feed_activations() or run() first.")
        stats = {}
        for name, acc in self._accumulators.items():
            stats[name] = acc.finalize()
            if self._verbose:
                print(
                    f"[Calibrator] {name}: n={acc._n}, "
                    f"mean=[{stats[name].mean.min():.2f}..{stats[name].mean.max():.2f}], "
                    f"int8_range={stats[name].int8_range}"
                )
        return stats

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def save(stats: dict[str, LayerStats], path: str | Path) -> None:
        """
        Save calibration statistics to a JSON file.

        Args:
            stats: Output of finalize().
            path:  Destination path (created if necessary).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "saved_at": time.time(),
            "layers": {name: s.to_dict() for name, s in stats.items()},
        }
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: str | Path) -> dict[str, LayerStats]:
        """
        Load calibration statistics previously saved with save().

        Args:
            path: Path to the JSON file.

        Returns:
            dict[layer_name → LayerStats]
        """
        path = Path(path)
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            raise ValueError(f"Unsupported calibration version: {data.get('version')}")
        return {
            name: LayerStats.from_dict(layer_data)
            for name, layer_data in data["layers"].items()
        }
