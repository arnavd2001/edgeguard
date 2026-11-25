# EdgeGuard threat model (summary)

The full threat model is Chapter 4 of the project report
(`report/thesis/main.pdf`). This file is a short summary.

> An earlier version of this file contained unsupported figures (a "37%"
> false-positive reduction, a "~1,200 sequences in 6 hours" extraction scenario)
> and a comparison table with references that could not be verified. Those have
> been removed. Measured results are in `experiments/results/results.json` and
> Chapter 7 of the report.

## System

A vendor quantizes a model to INT8, builds a SHA-256 Merkle tree over the weight
tensors and signs the root with ECDSA P-256. On the device, weights sit in DRAM
and are executed by an NPU. EdgeGuard re-hashes the weights against the signed
root, scores per-layer INT8 activations against calibration statistics, and
emits a threat label with an explanation.

**Trusted:** the EdgeGuard verifier, its public key and its calibration
statistics (e.g. inside a TEE / covered by secure boot).

## Adversary capabilities

| ID | Capability | Example |
|----|------------|---------|
| C1 | Write DRAM bits | Rowhammer (Kim et al. ISCA'14; DeepHammer, USENIX Sec'20) |
| C2 | Read DRAM | Cold boot (Halderman et al. USENIX Sec'08), EM side channel |
| C3 | Choose inputs | Adversarial prompts, extraction probes |
| C4 | Modify weights before loading | Tampered checkpoint on a mirror or in an update channel |

## Attack classes and expected signals

| Class | Needs | Expected EdgeGuard signal |
|-------|-------|---------------------------|
| A1 Bit-flip attack | C1 | Merkle mismatch on the modified tensor(s) |
| A2 Weight extraction (read only) | C2 | **None** — passive reads are not detectable |
| A3 Training-data probe | C3 | Late-layer activation anomaly (*hypothesis, untested on real models*) |
| A4 Supply-chain poisoning | C4 | Merkle mismatch (if modified after signing) |
| A5 Adversarial input | C3 | Activation anomaly |
| A6 Synergy (A1 + A5) | C1, C3 | Merkle mismatch even when activations look clean |

## Non-goals

- Backdoors present when the vendor signs the weights.
- Passive memory reads (A2).
- Flips between two verifications (time-of-check/time-of-use gap).
- Robustness of the activation monitor against a fully adaptive attacker.

## Key measured facts (simulation; see Chapter 7)

- Every simulated weight modification was detected and localised to exactly the
  modified tensors.
- A fixed 4σ threshold over 1,248 monitored channels gives ~7% false positives on
  clean inputs, as predicted by a multiple-comparisons argument; a Bonferroni
  threshold gives ~1%.
- Full re-hashing of a 1.4 GiB weight set takes ~1.6 s on a 2-vCPU x86 machine,
  so per-inference full attestation is not practical for LLM-sized weights.
- Activation-level results come from simulated activations whose distributions
  were chosen together with the detector; they demonstrate the pipeline, not
  robustness to real attacks.
