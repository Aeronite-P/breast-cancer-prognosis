"""
validate_external.py — Stage A rigor follow-ups (the credibility moat).

Two honest tests:

1. EXTERNAL VALIDATION (hypothesis H1). Train the gene model on ONE cohort and test it on a
   SECOND, independent cohort measured on a different platform (METABRIC microarray vs
   TCGA RNA-seq). If a model only works on the data it was trained on, it learned noise.
   Generalizing to a separate cohort is the single strongest evidence the signal is real.
   Both cohorts use per-gene z-scores, which is what makes cross-platform comparison valid.

2. SIGNIFICANCE TEST. Within METABRIC, bootstrap a 95% confidence interval for the difference
   in C-index between (clinical + genes) and (clinical only). This tells us honestly whether
   the 0.666 vs 0.662 gap from Stage A is real or just cross-validation noise. If the interval
   includes 0, we must NOT claim the genes add value.

Run:  python src/validate_external.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config           # noqa: E402
import fetch_data       # noqa: E402  (src/ is on the path)
import model as M       # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def ensure_tcga():
    """Download TCGA-BRCA clinical + PAM50 expression once (if not already present)."""
    path = os.path.join(config.PROCESSED_DIR, "tcga_merged.csv")
    if os.path.exists(path):
        print("TCGA data already present.")
        return
    print("Fetching TCGA-BRCA (clinical + PAM50 expression) ...")
    fetch_data.build(config.TCGA["study_id"], config.TCGA["expr_profile"],
                     config.PAM50, "tcga")


def bootstrap_delta(oof_clin, oof_comb, dur, evt, n=1000, seed=42):
    """Bootstrap the difference in C-index (combined - clinical). Returns (mean, lo95, hi95)."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(dur))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if evt[s].sum() < 5:           # need some events for a defined C-index
            continue
        try:
            c_clin = concordance_index(dur[s], oof_clin[s], evt[s])
            c_comb = concordance_index(dur[s], oof_comb[s], evt[s])
            diffs.append(c_comb - c_clin)
        except Exception:
            continue
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIG_DIR, exist_ok=True)
    ensure_tcga()

    mb, mb_genes = M.load_cohort("metabric")
    tc, tc_genes = M.load_cohort("tcga")
    genes = [g for g in mb_genes if g in tc_genes]   # genes available in BOTH cohorts
    print(f"\nMETABRIC n={len(mb)} (events {int(mb['event'].sum())}) | "
          f"TCGA n={len(tc)} (events {int(tc['event'].sum())}) | shared genes={len(genes)}\n")

    # === 1. External validation (genes-only model) ===
    mb_internal = M.cv_cindex(mb, genes, [])[0]   # within-METABRIC (cross-validated)
    tc_internal = M.cv_cindex(tc, genes, [])[0]   # within-TCGA (cross-validated)
    m2t = M.external_cindex(mb, tc, genes, [])    # train METABRIC -> test TCGA
    t2m = M.external_cindex(tc, mb, genes, [])    # train TCGA -> test METABRIC

    print("=== External validation (PAM50 gene model), C-index ===")
    print(f"METABRIC internal (CV):      {mb_internal:.3f}")
    print(f"TCGA internal (CV):          {tc_internal:.3f}")
    print(f"Train METABRIC -> test TCGA: {m2t:.3f}   <- external")
    print(f"Train TCGA -> test METABRIC: {t2m:.3f}   <- external")

    # === 2. Significance test (clinical vs clinical+genes, within METABRIC) ===
    clin_oof, dur, evt = M.oof_scores(mb, M.CLIN_NUM, M.CLIN_CAT)
    comb_oof, _, _ = M.oof_scores(mb, M.CLIN_NUM + genes, M.CLIN_CAT)
    c_clin = concordance_index(dur, clin_oof, evt)
    c_comb = concordance_index(dur, comb_oof, evt)
    mean_d, lo, hi = bootstrap_delta(clin_oof, comb_oof, dur, evt)
    significant = not (lo <= 0 <= hi)
    print("\n=== Do genes add value beyond clinical? (METABRIC, bootstrap) ===")
    print(f"clinical C-index:        {c_clin:.3f}")
    print(f"clinical+genes C-index:  {c_comb:.3f}")
    print(f"difference (ΔC):         {mean_d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"-> {'SIGNIFICANT' if significant else 'NOT significant'} "
          f"(CI {'excludes' if significant else 'includes'} 0)")

    # save metrics
    pd.DataFrame([
        {"test": "genes_metabric_internal_cv", "c_index": mb_internal},
        {"test": "genes_tcga_internal_cv", "c_index": tc_internal},
        {"test": "genes_train_metabric_test_tcga", "c_index": m2t},
        {"test": "genes_train_tcga_test_metabric", "c_index": t2m},
        {"test": "metabric_clinical", "c_index": c_clin},
        {"test": "metabric_clinical_plus_genes", "c_index": c_comb},
        {"test": "delta_c_mean", "c_index": mean_d},
        {"test": "delta_c_lo95", "c_index": lo},
        {"test": "delta_c_hi95", "c_index": hi},
    ]).to_csv(os.path.join(config.RESULTS_DIR, "metrics_external.csv"), index=False)

    # figure: external validation bars
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["METABRIC\ninternal", "TCGA\ninternal", "MB→TCGA\n(external)", "TCGA→MB\n(external)"]
    vals = [mb_internal, tc_internal, m2t, t2m]
    ax.bar(labels, vals, color=["#4C72B0", "#4C72B0", "#55A868", "#55A868"])
    ax.axhline(0.5, ls="--", color="red", label="random (0.5)")
    ax.set_ylim(0.4, 0.75)
    ax.set_ylabel("C-index")
    ax.set_title("PAM50 gene model: does it generalize to a second cohort?")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(config.FIG_DIR, "external_validation.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> results/metrics_external.csv and {out}")


if __name__ == "__main__":
    main()
