#!/usr/bin/env python3
"""
Reproducible evaluation of EdgeGuard on simulated scenarios.

Every number reported in report/mtp_report.tex is produced by this script and
written to experiments/results/results.json. Figures are produced separately
by experiments/make_figures.py from that JSON file.

IMPORTANT: activations in these experiments are *synthetic* (drawn from the
distributions defined in src/edgeguard/attacks/). The detector and the attack
generators were written by the same authors, so activation-based detection
rates measure internal consistency of the simulation, not robustness against
an adaptive real-world attacker. Weight-based detection (Merkle attestation)
does not depend on this simulation: any byte change alters the SHA-256 digest.

Usage:
    PYTHONPATH=src python experiments/run_evaluation.py
"""

from __future__ import annotations

import json
import platform
import time
import zlib
from pathlib import Path

import numpy as np

from edgeguard.attestation.manager import MerkleAttestationManager
from edgeguard.attacks.adversarial import AdversarialInputAttack
from edgeguard.attacks.bitflip import BitFlipAttack
from edgeguard.attacks.supply_chain import SupplyChainPoisoning
from edgeguard.attacks.synergy import SynergyAttack
from edgeguard.attacks.training_data_recovery import TrainingDataRecovery
from edgeguard.attacks.weight_extraction import WeightExtractionAttack
from edgeguard.detection.calibration import Calibrator
from edgeguard.detection.monitor import LayerwiseActivationMonitor
from edgeguard.llm.detector import LLMThreatDetector, LLMThreatType
from edgeguard.llm.model import PROFILES, QuantizedTransformerStub

OUT_DIR = Path(__file__).parent / "results"
SEEDS = [0, 1, 2, 3, 4]
N_PER_SEED = 200
N_CALIB = 500
DIM = 32
PROFILE = "tiny"

SCENARIOS = [
    # name,                 expected type,                        weight-modifying?
    ("CLEAN",               LLMThreatType.CLEAN,                  False),
    ("BFA_RANDOM",          LLMThreatType.WEIGHT_TAMPER,          True),
    ("BFA_SIGN_BIT",        LLMThreatType.WEIGHT_TAMPER,          True),
    ("BFA_PROGRESSIVE",     LLMThreatType.WEIGHT_TAMPER,          True),
    ("SUPPLY_STEG",         LLMThreatType.SUPPLY_CHAIN_BACKDOOR,  True),
    ("SUPPLY_DISTRIBUTED",  LLMThreatType.SUPPLY_CHAIN_BACKDOOR,  True),
    ("ADV_TEXT_INJECTION",  LLMThreatType.ADVERSARIAL_INPUT,      False),
    ("ADV_SAFETY_BYPASS",   LLMThreatType.JAILBREAK,              False),
    ("EXFIL_PROBE",         LLMThreatType.TRAINING_DATA_EXFILTRATION, False),
    ("SYNERGY",             LLMThreatType.WEIGHT_TAMPER,          True),
]


def clean_acts(layer_names, rng):
    return {n: np.clip(rng.normal(10, 5, size=DIM), -128, 127).astype(np.int8)
            for n in layer_names}


def build_detector(weights, layer_names, seed):
    rng = np.random.default_rng(1000 + seed)
    mgr = MerkleAttestationManager()
    mgr.build(weights)
    cal = Calibrator(layer_names=layer_names, max_cov_dim=64, verbose=0)
    for _ in range(N_CALIB):
        cal.feed_activations(clean_acts(layer_names, rng))
    monitor = LayerwiseActivationMonitor(cal.finalize(), zscore_threshold=4.0,
                                         mahal_threshold=50.0)
    det = LLMThreatDetector(layer_names=layer_names, attestation_manager=mgr,
                            activation_monitor=monitor,
                            mem_score_threshold=0.55, entropy_threshold=0.30)
    return det, monitor, mgr


def make_input(name, i, weights, layer_names, rng):
    """Return (weights_to_check, activations, ground_truth_layers)."""
    gt_layers = set()
    if name == "CLEAN":
        return weights, clean_acts(layer_names, rng), gt_layers
    if name.startswith("BFA_"):
        strat = {"BFA_RANDOM": "random", "BFA_SIGN_BIT": "sign_bit",
                 "BFA_PROGRESSIVE": "progressive"}[name]
        r = BitFlipAttack(strategy=strat, n_flips=1, seed=int(rng.integers(1 << 30))).attack(weights)
        gt_layers = {k for k in weights if not np.array_equal(weights[k], r.attacked_weights[k])}
        return r.attacked_weights, clean_acts(layer_names, rng), gt_layers
    if name.startswith("SUPPLY_"):
        strat = "steganographic" if name == "SUPPLY_STEG" else "distributed"
        r = SupplyChainPoisoning(strategy=strat, seed=int(rng.integers(1 << 30))).poison(weights)
        gt_layers = {k for k in weights if not np.array_equal(weights[k], r.poisoned_weights[k])}
        return r.poisoned_weights, clean_acts(layer_names, rng), gt_layers
    if name.startswith("ADV_"):
        t = "text_injection" if name == "ADV_TEXT_INJECTION" else "safety_bypass"
        r = AdversarialInputAttack(attack_type=t, seed=int(rng.integers(1 << 30))).attack(layer_names, activation_dim=DIM)
        return weights, r.adversarial_activation, gt_layers
    if name == "EXFIL_PROBE":
        r = TrainingDataRecovery(vocab_size=32000, seq_len=32, memorization_rate=0.45,
                                 seed=int(rng.integers(1 << 30))).simulate_probe(layer_names, n_queries=1, activation_dim=DIM)
        return weights, r.activation_signature, gt_layers
    if name == "SYNERGY":
        r = SynergyAttack(bfa_strategy="targeted", adv_type="safety_bypass", suppression=0.65,
                          seed=int(rng.integers(1 << 30))).attack(
            weights, layer_names, target_layer=layer_names[1], activation_dim=DIM)
        w = r.bfa_result.attacked_weights
        gt_layers = {k for k in weights if not np.array_equal(weights[k], w[k])}
        return w, r.combined_activation, gt_layers
    raise ValueError(name)


def scenario_eval():
    model = QuantizedTransformerStub(profile_name=PROFILE, seed=42)
    weights = model.get_weights()
    layer_names = model.layer_names
    types = [t.name for t in LLMThreatType]

    per_scenario = {}
    for name, expected, _ in SCENARIOS:
        per_scenario[name] = {"expected": expected.name, "seeds": [],
                              "confusion": {t: 0 for t in types},
                              "latency_ms": [], "attest_ms": [], "monitor_ms": [],
                              "monitor_only_flagged": 0, "localised_ok": 0,
                              "localisation_trials": 0, "max_z": []}

    for seed in SEEDS:
        det, monitor, mgr = build_detector(weights, layer_names, seed)
        for name, expected, _ in SCENARIOS:
            rng = np.random.default_rng(seed * 7919 + zlib.crc32(name.encode()) % 10007)
            rec = per_scenario[name]
            correct = flagged = 0
            for i in range(N_PER_SEED):
                w, acts, gt = make_input(name, i, weights, layer_names, rng)
                t0 = time.perf_counter()
                res = det.detect(acts, weights=w)
                rec["latency_ms"].append((time.perf_counter() - t0) * 1e3)
                # component timings (measured separately, same inputs)
                t1 = time.perf_counter(); a = mgr.verify(w); t2 = time.perf_counter()
                m = monitor.inspect(acts); t3 = time.perf_counter()
                rec["attest_ms"].append((t2 - t1) * 1e3)
                rec["monitor_ms"].append((t3 - t2) * 1e3)
                rec["max_z"].append(float(m.max_zscore))
                if m.overall_anomaly:
                    rec["monitor_only_flagged"] += 1
                rec["confusion"][res.threat_type.name] += 1
                correct += int(res.threat_type == expected)
                flagged += int(res.threat_type != LLMThreatType.CLEAN)
                if gt:
                    rec["localisation_trials"] += 1
                    if set(a.tampered_layer_names) == gt:
                        rec["localised_ok"] += 1
            rec["seeds"].append({"seed": seed, "type_accuracy": correct / N_PER_SEED,
                                 "flag_rate": flagged / N_PER_SEED})

    summary = {}
    for name, rec in per_scenario.items():
        acc = np.array([s["type_accuracy"] for s in rec["seeds"]])
        flag = np.array([s["flag_rate"] for s in rec["seeds"]])
        lat = np.array(rec["latency_ms"])
        n = len(lat)
        summary[name] = {
            "expected": rec["expected"],
            "n_total": n,
            "type_accuracy_mean": float(acc.mean()), "type_accuracy_std": float(acc.std(ddof=1)),
            "flag_rate_mean": float(flag.mean()), "flag_rate_std": float(flag.std(ddof=1)),
            "monitor_only_flag_rate": rec["monitor_only_flagged"] / n,
            "latency_ms_mean": float(lat.mean()), "latency_ms_p50": float(np.median(lat)),
            "latency_ms_p99": float(np.percentile(lat, 99)),
            "attest_ms_mean": float(np.mean(rec["attest_ms"])),
            "monitor_ms_mean": float(np.mean(rec["monitor_ms"])),
            "localisation_exact_rate": (rec["localised_ok"] / rec["localisation_trials"]
                                        if rec["localisation_trials"] else None),
            "confusion": rec["confusion"],
            "per_seed": rec["seeds"],
            "max_z_sample": rec["max_z"][:2000],
        }
    return {"profile": PROFILE, "n_layers": len(layer_names),
            "n_params": int(model.n_weights), "activation_dim": DIM,
            "seeds": SEEDS, "n_per_seed": N_PER_SEED, "n_calib": N_CALIB,
            "scenarios": summary}


def roc_from_scores(neg, pos):
    thr = np.unique(np.concatenate([neg, pos]))[::-1]
    neg, pos = np.asarray(neg), np.asarray(pos)
    fpr = [0.0] + [float((neg >= t).mean()) for t in thr] + [1.0]
    tpr = [0.0] + [float((pos >= t).mean()) for t in thr] + [1.0]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def threshold_sweep(scen):
    """ROC of the max-z-score statistic, activation-only scenarios vs clean."""
    neg = scen["CLEAN"]["max_z_sample"]
    out = {}
    for name in ["ADV_TEXT_INJECTION", "ADV_SAFETY_BYPASS", "EXFIL_PROBE", "SYNERGY"]:
        fpr, tpr, auc = roc_from_scores(neg, scen[name]["max_z_sample"])
        # downsample curve for storage
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int)
        out[name] = {"fpr": [fpr[i] for i in idx], "tpr": [tpr[i] for i in idx], "auc": auc}
    # empirical FPR at the operating point z >= 4
    out["clean_fpr_at_z4"] = float((np.asarray(neg) >= 4.0).mean())
    return out


def stub_param_count(p):
    H, I, V, L, hd = p.hidden_size, p.intermediate_size, p.vocab_size, p.n_layers, p.head_dim
    per_block = 2 * H * H + 2 * H * hd + 3 * H * I + 2 * H
    return V * H * 2 + L * per_block + H, 9 * L + 3


def merkle_scaling():
    """Wall-clock cost of Merkle build/verify on stub-shaped INT8 weights."""
    rows = []
    for key in ["tiny", "qwen2.5-0.5b", "llama-3.2-1b"]:
        p = PROFILES[key]
        rng = np.random.default_rng(0)
        model = QuantizedTransformerStub(profile=p, seed=0, scale_weights=False)
        # fill with int8 directly to avoid large float64 temporaries
        model._rand = lambda shape, r=rng: r.integers(-128, 128, size=shape, dtype=np.int8)
        w = model.get_weights()
        n = int(sum(a.size for a in w.values()))
        mgr = MerkleAttestationManager()
        t0 = time.perf_counter(); mgr.build(w); t_build = time.perf_counter() - t0
        times = []
        for _ in range(3):
            t0 = time.perf_counter(); res = mgr.verify(w); times.append(time.perf_counter() - t0)
            assert res.is_valid
        # single-bit tamper in the last block's down_proj
        tk = [k for k in w if "down_proj" in k][-1]
        w[tk].flat[0] = np.int8(w[tk].flat[0] ^ np.int8(1))
        t0 = time.perf_counter(); res = mgr.verify(w); t_tamper = time.perf_counter() - t0
        rows.append({"profile": p.name, "n_tensors": len(w), "n_params": n,
                     "size_mb": n / 2**20, "build_s": t_build,
                     "verify_s_mean": float(np.mean(times)), "verify_tampered_s": t_tamper,
                     "tamper_detected": (not res.is_valid) and res.tampered_layer_names == [tk],
                     "throughput_mb_s": (n / 2**20) / float(np.mean(times))})
        del w, model, mgr
    return rows


def extraction_fidelity():
    model = QuantizedTransformerStub(profile_name=PROFILE, seed=42)
    w = model.get_weights()
    out = {}
    for t in ["memory_dump", "cold_boot", "em_side_channel"]:
        vals = [WeightExtractionAttack(attack_type=t, seed=s).extract(w).mean_fidelity for s in SEEDS]
        out[t] = {"mean_fidelity": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
    return out


def layer_profiles():
    """Mean |activation| per layer index for one example of each input type."""
    model = QuantizedTransformerStub(profile_name=PROFILE, seed=42)
    names = model.layer_names
    rng = np.random.default_rng(5)
    reps = 50
    prof = {"CLEAN": np.zeros(len(names)), "ADV_SAFETY_BYPASS": np.zeros(len(names)),
            "EXFIL_PROBE": np.zeros(len(names))}
    std = {k: np.zeros(len(names)) for k in prof}
    for r in range(reps):
        a = clean_acts(names, rng)
        b = AdversarialInputAttack(attack_type="safety_bypass", seed=r).attack(names, activation_dim=DIM).adversarial_activation
        c = TrainingDataRecovery(seq_len=32, seed=r).simulate_probe(names, n_queries=1, activation_dim=DIM).activation_signature
        for k, acts in (("CLEAN", a), ("ADV_SAFETY_BYPASS", b), ("EXFIL_PROBE", c)):
            for i, n in enumerate(names):
                x = acts[n].astype(np.float32)
                prof[k][i] += x.mean() / reps
                std[k][i] += x.std() / reps
    return {"layer_index": list(range(len(names))),
            "mean": {k: v.tolist() for k, v in prof.items()},
            "std": {k: v.tolist() for k, v in std.items()}}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scen = scenario_eval()
    results = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "machine": platform.machine(), "processor": platform.processor() or "unknown",
                        "cpu_count": __import__("os").cpu_count()},
        "scenario_eval": scen,
        "roc": threshold_sweep(scen["scenarios"]),
        "merkle_scaling": merkle_scaling(),
        "extraction_fidelity": extraction_fidelity(),
        "layer_profiles": layer_profiles(),
        "stub_profiles": {k: dict(zip(["n_params", "n_tensors"], stub_param_count(p)))
                          | {"n_layers": p.n_layers, "hidden": p.hidden_size,
                             "ffn": p.intermediate_size, "vocab": p.vocab_size}
                          for k, p in PROFILES.items()},
    }
    results["wall_time_s"] = time.time() - t0
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
    s = scen["scenarios"]
    print(f"{'scenario':22s} {'type-acc':>12s} {'flagged':>12s} {'mon-only':>9s} {'lat ms':>7s} {'loc':>6s}")
    for k, v in s.items():
        loc = v["localisation_exact_rate"]
        print(f"{k:22s} {v['type_accuracy_mean']:.3f}±{v['type_accuracy_std']:.3f} "
              f"{v['flag_rate_mean']:.3f}±{v['flag_rate_std']:.3f} {v['monitor_only_flag_rate']:9.3f} "
              f"{v['latency_ms_mean']:7.3f} {'' if loc is None else f'{loc:.3f}':>6s}")
    print("ROC AUC:", {k: round(v["auc"], 4) for k, v in results["roc"].items() if isinstance(v, dict)})
    print("clean FPR at z>=4:", results["roc"]["clean_fpr_at_z4"])
    for r in results["merkle_scaling"]:
        print(r)
    print(results["extraction_fidelity"])
    print(f"done in {results['wall_time_s']:.0f}s")


if __name__ == "__main__":
    main()
