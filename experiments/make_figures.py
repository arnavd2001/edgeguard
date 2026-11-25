#!/usr/bin/env python3
"""
Generate all thesis figures from experiments/results/results.json.

Also writes experiments/results/analysis.json with derived quantities
(multiple-comparison threshold analysis) that the thesis text quotes.

Usage:
    python experiments/make_figures.py
"""

from __future__ import annotations

import json
from math import erf, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).parent
RES = json.loads((ROOT / "results" / "results.json").read_text())
FIG = ROOT.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- palette (validated reference categorical slots 1-3, light mode) -------
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "lines.linewidth": 2.0,
})

SC = RES["scenario_eval"]["scenarios"]
ORDER = list(SC.keys())
LABEL = {
    "CLEAN": "Clean", "BFA_RANDOM": "BFA random", "BFA_SIGN_BIT": "BFA sign-bit",
    "BFA_PROGRESSIVE": "BFA progressive", "SUPPLY_STEG": "Supply: stego",
    "SUPPLY_DISTRIBUTED": "Supply: distrib.", "ADV_TEXT_INJECTION": "Adv: injection",
    "ADV_SAFETY_BYPASS": "Adv: safety bypass", "EXFIL_PROBE": "Exfil. probe", "SYNERGY": "Synergy",
}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig: detection per scenario (three measures, grouped bars)
# ---------------------------------------------------------------------------
def fig_detection():
    x = np.arange(len(ORDER))
    w = 0.26
    flag = [SC[k]["flag_rate_mean"] for k in ORDER]
    flag_e = [SC[k]["flag_rate_std"] for k in ORDER]
    typ = [SC[k]["type_accuracy_mean"] for k in ORDER]
    typ_e = [SC[k]["type_accuracy_std"] for k in ORDER]
    mon = [SC[k]["monitor_only_flag_rate"] for k in ORDER]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ek = dict(ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.bar(x - w - 0.01, mon, w, color=AQUA, label="Activation monitor only: flagged")
    ax.bar(x, flag, w, yerr=flag_e, color=BLUE, label="EdgeGuard (unified): flagged", error_kw=ek)
    ax.bar(x + w + 0.01, typ, w, yerr=typ_e, color=ORANGE, label="EdgeGuard: correct threat label", error_kw=ek)
    ax.set_xticks(x, [LABEL[k] for k in ORDER], rotation=35, ha="right")
    ax.set_ylabel("Fraction of 1,000 inputs")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "detection_by_scenario")


# ---------------------------------------------------------------------------
# Fig: confusion matrix
# ---------------------------------------------------------------------------
def fig_confusion():
    cols = [t for t in ["CLEAN", "WEIGHT_TAMPER", "SUPPLY_CHAIN_BACKDOOR", "ADVERSARIAL_INPUT",
                        "TRANSIENT_FAULT", "TRAINING_DATA_EXFILTRATION", "JAILBREAK",
                        "PROMPT_INJECTION"]]
    M = np.array([[SC[r]["confusion"].get(c, 0) for c in cols] for r in ORDER], dtype=float)
    M = M / M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    short = {"CLEAN": "Clean", "WEIGHT_TAMPER": "Weight\ntamper", "SUPPLY_CHAIN_BACKDOOR": "Supply\nchain",
             "ADVERSARIAL_INPUT": "Adversarial\ninput", "TRANSIENT_FAULT": "Transient\nfault",
             "TRAINING_DATA_EXFILTRATION": "Data\nexfil.", "JAILBREAK": "Jailbreak",
             "PROMPT_INJECTION": "Prompt\ninjection"}
    ax.set_xticks(range(len(cols)), [short[c] for c in cols], fontsize=7.5, rotation=30, ha="right")
    ax.set_yticks(range(len(ORDER)), [LABEL[k] for k in ORDER])
    ax.set_xlabel("Predicted threat type")
    ax.set_ylabel("Simulated scenario")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                ax.text(j, i, f"{M[i, j]:.2f}".rstrip("0").rstrip(".") if M[i, j] < 1 else "1",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Row-normalised fraction")
    cb.outline.set_visible(False)
    save(fig, "confusion_matrix")


# ---------------------------------------------------------------------------
# Multiple-comparison analysis + figure
# ---------------------------------------------------------------------------
def analysis_and_fpr_fig():
    n_layers = RES["scenario_eval"]["n_layers"]
    dim = RES["scenario_eval"]["activation_dim"]
    m = n_layers * dim
    clean = np.asarray(SC["CLEAN"]["max_z_sample"])
    t = np.linspace(2.5, 6.0, 141)
    p_single = 2 * (1 - norm.cdf(t))
    analytical = 1 - (1 - p_single) ** m
    empirical = np.array([(clean >= ti).mean() for ti in t])

    alpha = 0.01
    z_bonf = float(norm.ppf(1 - alpha / (2 * m)))
    out = {"m_channels": m, "p_single_z4": float(2 * (1 - norm.cdf(4.0))),
           "predicted_fpr_z4": float(1 - (1 - 2 * (1 - norm.cdf(4.0))) ** m),
           "empirical_fpr_z4": float((clean >= 4.0).mean()),
           "alpha": alpha, "z_bonferroni": z_bonf,
           "empirical_fpr_bonf": float((clean >= z_bonf).mean()),
           "tpr_at_bonf": {k: float((np.asarray(SC[k]["max_z_sample"]) >= z_bonf).mean())
                           for k in ["ADV_TEXT_INJECTION", "ADV_SAFETY_BYPASS", "EXFIL_PROBE",
                                     "SYNERGY", "CLEAN"]},
           "clean_max_z_mean": float(clean.mean()), "clean_max_z_p99": float(np.percentile(clean, 99)),
           "expected_max_gaussian": float(norm.ppf(1 - 1 / (2 * m)))}
    (ROOT / "results" / "analysis.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(t, analytical, color=BLUE, label=f"Analytical, $m={m}$ i.i.d. Gaussian channels")
    ax.plot(t, empirical, color=ORANGE, linestyle="--", label="Empirical, clean inputs (n=1,000)")
    ax.axvline(4.0, color=INK2, linewidth=0.8)
    ax.axvline(z_bonf, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(4.03, 0.8, "fixed $4\\sigma$", color=INK2, fontsize=8)
    ax.text(z_bonf + 0.03, 0.62, f"Bonferroni\n$z^*={z_bonf:.2f}$", color=INK2, fontsize=8)
    ax.set_xlabel("Threshold $\\tau$ on max $|z|$ across all channels")
    ax.set_ylabel("P(clean input flagged)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=7.5)
    save(fig, "fpr_multiple_comparisons")

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.hist(clean, bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.axvline(4.0, color=INK2, linewidth=0.8)
    ax.axvline(z_bonf, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Max $|z|$ over 1,248 channels (clean inputs)")
    ax.set_ylabel("Count")
    ax.grid(axis="x", visible=False)
    save(fig, "clean_maxz_histogram")
    return out


# ---------------------------------------------------------------------------
# Fig: ROC of the max-|z| statistic
# ---------------------------------------------------------------------------
def fig_roc():
    roc = RES["roc"]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for k, c, ls in (("ADV_TEXT_INJECTION", BLUE, "-"), ("EXFIL_PROBE", AQUA, "--"),
                     ("SYNERGY", ORANGE, "-")):
        ax.plot(roc[k]["fpr"], roc[k]["tpr"], color=c, linestyle=ls,
                label=f"{LABEL[k]} (AUC {roc[k]['auc']:.2f})")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax.set_xlabel("False positive rate (clean)")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "roc_activation_monitor")


# ---------------------------------------------------------------------------
# Fig: latency breakdown
# ---------------------------------------------------------------------------
def fig_latency():
    x = np.arange(len(ORDER))
    att = [SC[k]["attest_ms_mean"] for k in ORDER]
    mon = [SC[k]["monitor_ms_mean"] for k in ORDER]
    tot = [SC[k]["latency_ms_mean"] for k in ORDER]
    p99 = [SC[k]["latency_ms_p99"] for k in ORDER]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    w = 0.26
    ax.bar(x - w - 0.01, att, w, color=BLUE, label="Merkle verify (alone)")
    ax.bar(x, mon, w, color=AQUA, label="Activation monitor (alone)")
    ax.bar(x + w + 0.01, tot, w, color=ORANGE, label="Full detect() call, mean")
    ax.scatter(x + w + 0.01, p99, marker="_", s=80, color=INK, label="Full call, p99", zorder=3)
    ax.set_xticks(x, [LABEL[k] for k in ORDER], rotation=35, ha="right")
    ax.set_ylabel("Milliseconds per inference")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "latency_breakdown")


# ---------------------------------------------------------------------------
# Fig: Merkle scaling
# ---------------------------------------------------------------------------
def fig_merkle():
    rows = RES["merkle_scaling"]
    size = np.array([r["size_mb"] for r in rows])
    ver = np.array([r["verify_s_mean"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.loglog(size, ver * 1e3, color=BLUE, marker="o", markersize=6)
    for r in rows:
        ax.annotate(r["profile"].replace("-INT8", ""), (r["size_mb"], r["verify_s_mean"] * 1e3),
                    textcoords="offset points", xytext=(6, -10), fontsize=7.5, color=INK)
    ax.set_xlabel("Weight bytes hashed (MiB, log scale)")
    ax.set_ylabel("Full verify time (ms, log)")
    ax.set_xlim(0.15, 5000)
    save(fig, "merkle_scaling")


# ---------------------------------------------------------------------------
# Fig: per-layer activation profiles
# ---------------------------------------------------------------------------
def fig_profiles():
    lp = RES["layer_profiles"]
    idx = np.array(lp["layer_index"])
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    for k, c, ls in (("CLEAN", BLUE, "-"), ("ADV_SAFETY_BYPASS", ORANGE, "-"), ("EXFIL_PROBE", AQUA, "--")):
        mu = np.array(lp["mean"][k]); sd = np.array(lp["std"][k])
        ax.plot(idx, mu, color=c, linestyle=ls, label=LABEL[k])
        ax.fill_between(idx, mu - sd, mu + sd, color=c, alpha=0.12, linewidth=0)
    cut = max(1, int(len(idx) * 0.4))
    ax.axvline(cut - 0.5, color=INK2, linewidth=0.8)
    ax.text(cut - 0.3, ax.get_ylim()[1] * 0.92, "early | late (40% split)", color=INK2, fontsize=8)
    ax.set_xlabel("Weight-tensor index in model order (tiny profile, 39 tensors)")
    ax.set_ylabel("Mean INT8 activation")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "activation_profiles")


if __name__ == "__main__":
    fig_detection()
    fig_confusion()
    a = analysis_and_fpr_fig()
    fig_roc()
    fig_latency()
    fig_merkle()
    fig_profiles()
    print(json.dumps(a, indent=1))
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
