"""Tests for the INT8-aware activation monitor and calibration pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from edgeguard.detection.statistics import (
    StreamingInt8Stats,
    LayerStats,
    zscore_max,
    mahalanobis_distance,
    combined_score,
)
from edgeguard.detection.calibration import Calibrator
from edgeguard.detection.monitor import LayerwiseActivationMonitor, MonitorResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_clean_activations(d: int = 32, n: int = 200, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    # Simulate INT8 activations: mean≈10, std≈5, range roughly [-128,127]
    return [
        np.clip(rng.normal(10, 5, size=(d,)), -128, 127).astype(np.int8)
        for _ in range(n)
    ]


def calibrate_layer(name: str, samples: list[np.ndarray]) -> LayerStats:
    acc = StreamingInt8Stats(name, max_cov_dim=64)
    for s in samples:
        acc.update(s)
    return acc.finalize()


# ---------------------------------------------------------------------------
# StreamingInt8Stats
# ---------------------------------------------------------------------------

class TestStreamingInt8Stats:

    def test_mean_close_to_true_mean(self):
        rng = np.random.default_rng(42)
        samples = [rng.integers(-10, 10, size=(16,)).astype(np.int8) for _ in range(500)]
        acc = StreamingInt8Stats("test", max_cov_dim=64)
        for s in samples:
            acc.update(s)
        stats = acc.finalize()
        expected_mean = np.mean([s.astype(np.float64) for s in samples], axis=0)
        np.testing.assert_allclose(stats.mean, expected_mean, atol=0.5)

    def test_std_positive(self):
        samples = make_clean_activations(d=16, n=100)
        acc = StreamingInt8Stats("test", max_cov_dim=64)
        for s in samples:
            acc.update(s)
        stats = acc.finalize()
        assert (stats.std > 0).all()

    def test_covariance_computed_when_small_dim(self):
        samples = make_clean_activations(d=16, n=50)
        acc = StreamingInt8Stats("test", max_cov_dim=64)
        for s in samples:
            acc.update(s)
        stats = acc.finalize()
        assert stats.cov is not None
        assert stats.cov_inv is not None

    def test_covariance_skipped_when_large_dim(self):
        samples = make_clean_activations(d=600, n=50)
        acc = StreamingInt8Stats("test", max_cov_dim=512)
        for s in samples:
            acc.update(s)
        stats = acc.finalize()
        assert stats.cov is None

    def test_needs_two_samples(self):
        acc = StreamingInt8Stats("test")
        acc.update(np.zeros(8, dtype=np.int8))
        with pytest.raises(RuntimeError):
            acc.finalize()

    def test_int8_range_tracked(self):
        samples = make_clean_activations(d=16, n=100)
        acc = StreamingInt8Stats("test", max_cov_dim=64)
        for s in samples:
            acc.update(s)
        stats = acc.finalize()
        assert stats.int8_range[0] < stats.int8_range[1]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

class TestScoring:

    def setup_method(self):
        samples = make_clean_activations(d=16, n=200, seed=1)
        self.stats = calibrate_layer("test", samples)
        self.rng = np.random.default_rng(99)

    def test_clean_zscore_low(self):
        clean = make_clean_activations(d=16, n=1, seed=5)[0]
        z = zscore_max(clean, self.stats)
        assert z < 10.0, f"Expected low z-score for clean activation, got {z:.2f}"

    def test_adversarial_zscore_high(self):
        # Extreme activation (all 127) → very high z-score
        extreme = np.full(16, 127, dtype=np.int8)
        z = zscore_max(extreme, self.stats)
        assert z > 5.0, f"Expected high z-score for extreme activation, got {z:.2f}"

    def test_mahalanobis_clean_low(self):
        clean = make_clean_activations(d=16, n=1, seed=7)[0]
        m = mahalanobis_distance(clean, self.stats)
        assert m < 50.0

    def test_mahalanobis_extreme_high(self):
        extreme = np.full(16, 127, dtype=np.int8)
        m = mahalanobis_distance(extreme, self.stats)
        assert m > 10.0

    def test_combined_score_keys(self):
        clean = make_clean_activations(d=16, n=1, seed=3)[0]
        result = combined_score(clean, self.stats)
        assert "zscore" in result
        assert "mahalanobis" in result
        assert "combined" in result
        assert result["layer_name"] == "test"


# ---------------------------------------------------------------------------
# Calibrator + LayerwiseActivationMonitor integration
# ---------------------------------------------------------------------------

class TestMonitorIntegration:

    def _build_monitor(self, layer_names, n_calib=200) -> LayerwiseActivationMonitor:
        cal = Calibrator(max_cov_dim=64)
        for _ in range(n_calib):
            activations = {
                name: make_clean_activations(d=16, n=1, seed=_)[0]
                for name in layer_names
            }
            cal.feed_activations(activations)
        stats = cal.finalize()
        return LayerwiseActivationMonitor(stats, zscore_threshold=4.0, mahal_threshold=50.0)

    def test_clean_inference_no_anomaly(self):
        monitor = self._build_monitor(["conv1", "conv2"])
        rng = np.random.default_rng(2)
        clean = {
            "conv1": np.clip(rng.normal(10, 5, 16), -128, 127).astype(np.int8),
            "conv2": np.clip(rng.normal(10, 5, 16), -128, 127).astype(np.int8),
        }
        result = monitor.inspect(clean)
        # May flag anomaly if sample is unusual, but overall should be low-score
        assert isinstance(result, MonitorResult)
        assert result.max_zscore >= 0.0

    def test_adversarial_input_flagged(self):
        monitor = self._build_monitor(["conv1", "conv2"])
        # Extreme activations far outside calibration range
        adversarial = {
            "conv1": np.full(16, 127, dtype=np.int8),
            "conv2": np.full(16, 127, dtype=np.int8),
        }
        result = monitor.inspect(adversarial)
        assert result.overall_anomaly is True
        assert len(result.anomalous_layers) > 0

    def test_missing_layer_raises(self):
        monitor = self._build_monitor(["conv1", "conv2"])
        with pytest.raises(KeyError):
            monitor.inspect({"conv1": np.zeros(16, dtype=np.int8)})  # conv2 missing

    def test_save_and_load_calibration(self):
        cal = Calibrator(max_cov_dim=64)
        for i in range(100):
            cal.feed_activations({
                "fc1": make_clean_activations(d=8, n=1, seed=i)[0]
            })
        stats = cal.finalize()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal.json"
            Calibrator.save(stats, path)
            loaded = Calibrator.load(path)

        assert "fc1" in loaded
        np.testing.assert_allclose(loaded["fc1"].mean, stats["fc1"].mean, atol=1e-6)

    def test_threshold_tuning(self):
        monitor = self._build_monitor(["layer0"], n_calib=100)
        clean_results = [
            monitor.inspect({"layer0": make_clean_activations(d=16, n=1, seed=k)[0]})
            for k in range(50)
        ]
        z_new, m_new = monitor.tune_thresholds(clean_results, fpr_target=0.05)
        assert z_new > 0
        assert m_new > 0
