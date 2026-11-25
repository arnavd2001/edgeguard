# EdgeGuard: Layerwise Intrusion Detection & Integrity Attestation for Quantized DNNs

> **CS24M513 — IIT Madras MTech Project (Phase 1/2)**
> Explainable, layerwise security monitoring for quantized DNNs on edge accelerators (Qualcomm Hexagon / QNN·QAIRT).

---

## Problem Statement

Deep neural networks deployed on edge accelerators (mobile NPUs, DSPs) are quantized to INT8/INT4 for efficiency — but this creates a **three-way security gap** that no existing work addresses together:

1. **Adversarial inputs** crafted against the quantized forward pass exploit rounding behaviour that full-precision detectors have never observed.
2. **Weight tampering** (post-deployment file modification, Rowhammer bit-flips) is catastrophically amplified at low bit-widths — a single MSB flip in an INT8 weight changes 50% of its representable range.
3. **Transient hardware faults** (voltage glitching, single-event upsets) corrupt intermediate activations in ways that look identical to adversarial inputs unless cross-referenced with weight integrity.

Existing defences treat these as separate problems and are evaluated only on full-precision models. **EdgeGuard unifies all three** under one layerwise statistical signal, deployed as a software-only retrofit on a commodity edge-NPU stack.

---

## Architecture

```
                   ┌─────────────────────────────────────────┐
                   │            QNN / QAIRT Graph             │
  Input ──────────►│  Layer 0 → Layer 1 → ... → Layer N      │──► Output
                   │      ↑ hooks ↑       ↑ hooks ↑           │
                   └───────────┬─────────────────┬───────────┘
                               │                 │
                    ┌──────────▼──────────┐  ┌───▼────────────────┐
                    │  Activation Monitor │  │  Weight Attestation │
                    │  (INT8-calibrated   │  │  (Merkle tree,      │
                    │   z-score/Mahal.)   │  │   per-layer hash)   │
                    └──────────┬──────────┘  └───┬────────────────┘
                               │                 │
                    ┌──────────▼─────────────────▼────────────────┐
                    │           Threat Classifier                   │
                    │  Adversarial Input | Weight Tamper | HW Fault │
                    │  + Layer/Channel/Statistic Explanation        │
                    └──────────────────────────────────────────────┘
```

### Three Core Modules

| Module | File | What it does |
|--------|------|-------------|
| **Merkle Attestation** | `src/edgeguard/attestation/` | SHA-256 per-layer weight hashes → Merkle tree → signed root; verify at load + periodic runtime |
| **Activation Monitor** | `src/edgeguard/detection/` | Calibrates per-layer INT8 activation statistics on clean data; z-score + Mahalanobis scoring at runtime |
| **Threat Classifier** | `src/edgeguard/classifier/` | Combines hash signal + activation signal + input-space check to distinguish the three threat types |

---

## Novelty vs. Prior Work

| Property | Lee et al. NeurIPS'18 | NeuroPots USENIX'23 | SILK 2025 | **EdgeGuard** |
|---|---|---|---|---|
| Quantization-aware thresholds | ✗ | ✗ | ✗ | ✅ |
| Layer-granular localization | Partial | Pre-selected layers only | ✗ | ✅ All layers |
| Software-only / retrofittable | ✅ | ✗ (retraining) | ✗ (HW change) | ✅ |
| Unifies 3 threat types | ✗ | ✗ | ✗ | ✅ |
| Explanation = detection signal | ✗ | ✗ | ✗ | ✅ |
| Commodity NPU toolchain (QNN) | ✗ | ✗ | ✗ | ✅ |

---

## Installation

```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, PyTorch (for profiling; not required at inference time)

For Hexagon NPU deployment, additionally install the QNN/QAIRT SDK and set `QAIRT_ROOT`.

---

## Quick Start

```python
from edgeguard.attestation import MerkleAttestationManager
from edgeguard.detection  import LayerwiseActivationMonitor
from edgeguard.classifier import ThreatClassifier

# 1. Build Merkle tree over quantized layer weights
attest = MerkleAttestationManager.from_model_layers(quantized_layers)
root_sig = attest.sign_root(private_key)

# 2. Calibrate activation monitor on clean INT8 data
monitor = LayerwiseActivationMonitor(n_layers=len(quantized_layers))
monitor.calibrate(calibration_loader, model_fn)

# 3. Wire up threat classifier
classifier = ThreatClassifier(monitor, attest)

# 4. At inference time
result = classifier.inspect(input_tensor, model_fn, root_sig, private_key)
print(result.threat_type)   # ADVERSARIAL_INPUT | WEIGHT_TAMPER | HW_FAULT | CLEAN
print(result.explanation)   # "Layer 4, channel 23: activation deviated 4.7σ ..."
```

---

## Evaluation Plan

| Threat Suite | Method | Metric |
|---|---|---|
| Adversarial inputs | PGD / FGSM crafted against quantized graph | Detection rate, FPR, per-layer hit accuracy |
| Weight tampering | Targeted single-bit-flip injection (simulated BFA) | Layer localization accuracy, false negatives |
| Hardware faults | Activation corruption injection (Gaussian, targeted) | Threat-type attribution accuracy |
| Overhead | QNN/QAIRT instrumented graph vs. baseline | Latency overhead (ms), memory overhead (MB) |

---

## Project Structure

```
edgeguard/
├── src/edgeguard/
│   ├── attestation/        # Merkle-tree weight integrity
│   │   ├── merkle.py       # MerkleTree, MerkleNode, root hash
│   │   ├── manager.py      # MerkleAttestationManager (sign, verify, localize)
│   │   └── __init__.py
│   ├── detection/          # Activation anomaly monitoring
│   │   ├── calibration.py  # INT8-aware per-layer statistics calibration
│   │   ├── monitor.py      # LayerwiseActivationMonitor (score, explain)
│   │   ├── statistics.py   # Streaming INT8 stats, Mahalanobis, z-score
│   │   └── __init__.py
│   ├── classifier/         # Threat type disambiguation
│   │   ├── classifier.py   # ThreatClassifier, ThreatResult
│   │   ├── rules.py        # Decision rules (hash ✕ activation ✕ input)
│   │   └── __init__.py
│   ├── instrumentation/    # QNN/QAIRT graph hooks
│   │   ├── hooks.py        # LayerHook, ActivationCapture
│   │   ├── qnn_adapter.py  # QNN graph instrumentation (custom-op / IR pass)
│   │   └── __init__.py
│   └── utils/
│       ├── crypto.py       # SHA-256 helpers, ECDSA sign/verify
│       ├── quantization.py # INT8/INT4 helpers, scale/zero-point
│       └── __init__.py
├── tests/
│   ├── test_merkle.py
│   ├── test_monitor.py
│   └── test_classifier.py
├── scripts/
│   ├── calibrate.py        # Offline calibration script
│   ├── build_merkle.py     # Build + sign Merkle tree for a model
│   └── evaluate.py         # Run all three threat suites
├── configs/
│   └── default.yaml        # Thresholds, alert policy, monitoring interval
├── pyproject.toml
└── README.md
```

---

## Citation

```bibtex
@mastersthesis{cs24m513_edgeguard_2026,
  author  = {CS24M513},
  title   = {Explainable Layerwise Intrusion Detection and Integrity Attestation
             for Quantized DNNs on Edge Accelerators},
  school  = {Indian Institute of Technology Madras},
  year    = {2026},
  note    = {Phase 1/2 Report}
}
```

---

## License

MIT © 2026 CS24M513, IIT Madras
