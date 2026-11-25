#!/usr/bin/env python3
"""
End-to-end evaluation of the EdgeGuard detection pipeline.

Runs three threat scenarios:
  1. CLEAN            — baseline clean inputs, no tamper
  2. WEIGHT_TAMPER    — randomly flip bits in a randomly chosen layer
  3. ADVERSARIAL      — replace early-layer activations with extreme values

Outputs a table of detection results, false positive rate, and accuracy.

Usage:
    python scripts/evaluate.py \
        --attestation attestation/model_attestation.json \
        --calibration configs/calibration.json \
        --layers conv1 conv2 fc1 \
        [--n-samples 500] \
        [--tamper-layer conv1] \
        [--results results.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edgeguard.attestation.manager import MerkleAttestationManager
from edgeguard.detection.monitor import LayerwiseActivationMonitor
from edgeguard.detection.calibration import Calibrator
from edgeguard.classifier.classifier import ThreatClassifier, ThreatType


def make_synth_weights(layer_names: list[str], seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: rng.integers(-128, 128, size=(64, 64), dtype=np.int8)
        for name in layer_names
    }


def make_clean_activation(layer_names: list[str], rng) -> dict[str, np.ndarray]:
    return {
        name: np.clip(rng.normal(10, 5, size=32), -128, 127).astype(np.int8)
        for name in layer_names
    }


def make_adversarial_activation(layer_names: list[str], rng, early_fraction=0.5) -> dict[str, np.ndarray]:
    n_early = max(1, int(len(layer_names) * early_fraction))
    result = {}
    for i, name in enumerate(layer_names):
        if i < n_early:
            # Extreme activation to simulate adversarial input
            result[name] = np.full(32, 127, dtype=np.int8)
        else:
            result[name] = np.clip(rng.normal(10, 5, size=32), -128, 127).astype(np.int8)
    return result


def evaluate_scenario(
    scenario: str,
    n_samples: int,
    weights: dict[str, np.ndarray],
    layer_names: list[str],
    mgr: MerkleAttestationManager,
    monitor: LayerwiseActivationMonitor,
    clf: ThreatClassifier,
    tamper_layer: str,
    seed: int = 1,
) -> dict:
    rng = np.random.default_rng(seed)
    correct = 0
    detections = {t.name: 0 for t in ThreatType}
    times = []

    for i in range(n_samples):
        t0 = time.perf_counter()

        if scenario == "CLEAN":
            current_weights = weights
            activations = make_clean_activation(layer_names, rng)
        elif scenario == "WEIGHT_TAMPER":
            tampered = dict(weights)
            tampered[tamper_layer] = rng.integers(-128, 128, size=weights[tamper_layer].shape, dtype=np.int8)
            current_weights = tampered
            activations = make_clean_activation(layer_names, rng)
        elif scenario == "ADVERSARIAL":
            current_weights = weights
            activations = make_adversarial_activation(layer_names, rng)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        attest_result = mgr.verify(current_weights)
        monitor_result = monitor.inspect(activations)
        threat_result = clf.classify(monitor_result, attest_result)

        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms

        detections[threat_result.threat_type.name] += 1
        expected_map = {
            "CLEAN": ThreatType.CLEAN,
            "WEIGHT_TAMPER": ThreatType.WEIGHT_TAMPER,
            "ADVERSARIAL": ThreatType.ADVERSARIAL_INPUT,
        }
        expected = expected_map[scenario]
        if threat_result.threat_type == expected:
            correct += 1

    accuracy = correct / n_samples
    avg_latency = sum(times) / len(times)
    return {
        "scenario": scenario,
        "n_samples": n_samples,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "avg_latency_ms": round(avg_latency, 3),
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description="EdgeGuard end-to-end evaluation")
    parser.add_argument("--layers",      nargs="+", default=["conv1", "conv2", "conv3", "fc1", "fc2"])
    parser.add_argument("--n-samples",   type=int, default=200)
    parser.add_argument("--tamper-layer", default=None)
    parser.add_argument("--results",     default=None, help="Save results to JSON file")
    args = parser.parse_args()

    layer_names = args.layers
    tamper_layer = args.tamper_layer or layer_names[0]

    print(f"[evaluate] Layers: {layer_names}")
    print(f"[evaluate] Samples per scenario: {args.n_samples}")
    print(f"[evaluate] Tamper target: {tamper_layer}")
    print()

    # Build synthetic model artifacts
    print("[evaluate] Building synthetic model weights...")
    weights = make_synth_weights(layer_names, seed=42)

    print("[evaluate] Building Merkle attestation...")
    mgr = MerkleAttestationManager()
    mgr.build(weights)
    print(f"           Root: {mgr.root_hash.hex()[:16]}...")

    print("[evaluate] Running calibration on 500 clean samples...")
    cal = Calibrator(layer_names=layer_names, max_cov_dim=64, verbose=0)
    rng = np.random.default_rng(99)
    for _ in range(500):
        activations = make_clean_activation(layer_names, rng)
        cal.feed_activations(activations)
    stats = cal.finalize()

    monitor = LayerwiseActivationMonitor(stats, zscore_threshold=4.0, mahal_threshold=50.0)
    clf = ThreatClassifier(layer_names, early_fraction=0.4)

    print("[evaluate] Running scenarios...")
    print()

    all_results = []
    for scenario in ["CLEAN", "WEIGHT_TAMPER", "ADVERSARIAL"]:
        r = evaluate_scenario(
            scenario=scenario,
            n_samples=args.n_samples,
            weights=weights,
            layer_names=layer_names,
            mgr=mgr,
            monitor=monitor,
            clf=clf,
            tamper_layer=tamper_layer,
        )
        all_results.append(r)
        print(f"  {scenario:20s}  accuracy={r['accuracy']:.1%}  avg_latency={r['avg_latency_ms']:.2f}ms")
        for threat, count in r["detections"].items():
            if count > 0:
                print(f"    → {threat}: {count}/{r['n_samples']}")
        print()

    if args.results:
        Path(args.results).write_text(json.dumps(all_results, indent=2))
        print(f"[evaluate] Results saved → {args.results}")

    # Summary
    print("=" * 60)
    overall_acc = sum(r["accuracy"] for r in all_results) / len(all_results)
    print(f"Overall accuracy across scenarios: {overall_acc:.1%}")


if __name__ == "__main__":
    main()
