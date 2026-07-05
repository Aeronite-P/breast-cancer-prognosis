"""
boost_genes.py — Stage A+ : does the FULL transcriptome beat the 50-gene panel?

Stage A found the 50-gene PAM50 panel (C-index ~0.60) did NOT beat clinical staging
(~0.66). The obvious next question: was 50 genes just too few? Here we test the
genome-wide expression (top most-variable genes) with models built for high-dimensional
data — a Random Survival Forest (non-linear) and a ridge-penalized Cox (linear) —
and check whether more genes finally adds value beyond clinical staging.

Method mirrors Stage A's rigor: 5-fold cross-validation, out-of-fold C-index, fixed seed,
and external validation on TCGA. Uses scikit-survival (proper survival models).

Run:  python src/boost_genes.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

TOP_K = 1000          # keep the 1000 most-variable genes (20x the PAM50 panel)
N_SPLITS = 5
SEED = config.RANDOM_SEED
RIDGE_ALPHA = 100.0   # ridge penalty for the high-dimensional linear Cox

# reference numbers from Stage A (same cohort, same CV) for context
STAGE_A = {"Clinical (staging)": 0.662, "PAM50 genes (Cox)": 0.599}


def load_expr(path: str, sample_to_patient=None) -> pd.DataFrame:
    """Load a genes x samples cBioPortal matrix -> patients x genes (float32)."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["Hugo_Symbol"].notna()]
    df = df.drop_duplicates(subset="Hugo_Symbol", keep="first")
    df = df.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"])
    expr = df.T                                   # samples x genes
    expr.index = expr.index.astype(str)
    if sample_to_patient is not None:
        expr.index = expr.index.map(sample_to_patient)
    return expr.astype("float32")


def load_survival(clinical_csv: str) -> pd.DataFrame:
    c = pd.read_csv(clinical_csv, index_col="patientId")
    out = pd.DataFrame(index=c.index)
    out["time"] = pd.to_numeric(c["OS_MONTHS"], errors="coerce")
    out["event"] = c["OS_STATUS"].astype(str).str.startswith("1")
    return out.dropna(subset=["time"])[lambda d: d["time"] > 0]


def surv_y(sub: pd.DataFrame):
    return Surv.from_arrays(event=sub["event"].to_numpy(bool),
                            time=sub["time"].to_numpy(float))


def oof_cindex(X: np.ndarray, sub: pd.DataFrame, model_fn):
    """5-fold out-of-fold C-index (higher = better). Returns pooled + per-fold."""
    dur = sub["time"].to_numpy(float)
    evt = sub["event"].to_numpy(bool)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(sub), np.nan)
    fold = []
    for tr, te in kf.split(X):
        m = model_fn()
        m.fit(X[tr], Surv.from_arrays(event=evt[tr], time=dur[tr]))
        risk = m.predict(X[te])            # higher = higher risk (shorter survival)
        oof[te] = risk
        fold.append(concordance_index_censored(evt[te], dur[te], risk)[0])
    pooled = concordance_index_censored(evt, dur, oof)[0]
    return pooled, float(np.mean(fold)), float(np.std(fold)), oof


def bootstrap_diff(oof_a, oof_b, sub, n=500):
    """95% CI for C-index difference (b - a), paired bootstrap over patients."""
    dur = sub["time"].to_numpy(float); evt = sub["event"].to_numpy(bool)
    rng = np.random.RandomState(SEED); idx = np.arange(len(sub)); diffs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if evt[s].sum() < 5:
            continue
        ca = concordance_index_censored(evt[s], dur[s], oof_a[s])[0]
        cb = concordance_index_censored(evt[s], dur[s], oof_b[s])[0]
        diffs.append(cb - ca)
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def rsf():
    return RandomSurvivalForest(n_estimators=100, min_samples_leaf=15,
                                max_features="sqrt", n_jobs=-1, random_state=SEED)


def ridge():
    return CoxPHSurvivalAnalysis(alpha=RIDGE_ALPHA)


def main():
    os.makedirs(config.FIG_DIR, exist_ok=True)
    print("Loading METABRIC full expression matrix ...")
    mb_expr = load_expr(os.path.join(config.RAW_DIR, "metabric_expr_full.txt"))
    surv = load_survival(os.path.join(config.RAW_DIR, "metabric_clinical.csv"))

    # align patients present in both expression and survival
    common = mb_expr.index.intersection(surv.index)
    mb_expr = mb_expr.loc[common]
    sub = surv.loc[common]
    print(f"Cohort: {len(sub)} patients, {int(sub['event'].sum())} deaths, "
          f"{mb_expr.shape[1]} genes")

    # top-K most variable genes (unsupervised: uses expression only, not outcomes)
    variances = mb_expr.var(axis=0, skipna=True)
    top_genes = variances.sort_values(ascending=False).head(TOP_K).index.tolist()
    X_full = mb_expr[top_genes].fillna(0.0).to_numpy("float32")
    pam50 = [g for g in config.PAM50 if g in mb_expr.columns]
    X_pam = mb_expr[pam50].fillna(0.0).to_numpy("float32")
    print(f"Using top {len(top_genes)} variable genes; PAM50 available: {len(pam50)}\n")

    results = {}
    print("Cross-validating (this takes a few minutes)...")
    for name, X, fn in [
        ("PAM50 (RSF)",        X_pam,  rsf),
        (f"Top-{TOP_K} (RSF)", X_full, rsf),
        (f"Top-{TOP_K} (ridge Cox)", X_full, ridge),
    ]:
        pooled, mean, std, oof = oof_cindex(X, sub, fn)
        results[name] = {"c": pooled, "std": std, "oof": oof}
        print(f"  {name:24s} C-index = {pooled:.3f}  (fold {mean:.3f} ± {std:.3f})")

    # does the full transcriptome beat the 50-gene panel? (paired bootstrap)
    best = max((k for k in results if k.startswith("Top")), key=lambda k: results[k]["c"])
    md, lo, hi = bootstrap_diff(results["PAM50 (RSF)"]["oof"], results[best]["oof"], sub)
    beats_pam = not (lo <= 0 <= hi)
    print(f"\n{best} vs PAM50 (RSF): ΔC = {md:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] "
          f"-> {'SIGNIFICANT' if beats_pam else 'not significant'}")
    print(f"Clinical staging (Stage A) = {STAGE_A['Clinical (staging)']:.3f} "
          f"| best gene model = {results[best]['c']:.3f} "
          f"-> genes {'BEAT' if results[best]['c'] > STAGE_A['Clinical (staging)'] else 'do NOT beat'} clinical")

    # external validation of the best full-gene model on TCGA
    print("\nExternal validation on TCGA ...")
    tcga_map = None
    tc_expr = load_expr(os.path.join(config.RAW_DIR, "tcga_expr_full.txt"),
                        sample_to_patient=lambda s: "-".join(s.split("-")[:3]))
    tc_surv = load_survival(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"))
    shared = [g for g in top_genes if g in tc_expr.columns]
    tc_common = tc_expr.index.intersection(tc_surv.index)
    Xtr = mb_expr[shared].fillna(0.0).to_numpy("float32")
    model = rsf(); model.fit(Xtr, surv_y(sub))
    Xte = tc_expr.loc[tc_common, shared].fillna(0.0).to_numpy("float32")
    tsub = tc_surv.loc[tc_common]
    ext_c = concordance_index_censored(tsub["event"].to_numpy(bool),
                                       tsub["time"].to_numpy(float),
                                       model.predict(Xte))[0]
    print(f"  Train METABRIC (top genes, RSF) -> test TCGA: C-index = {ext_c:.3f} "
          f"({len(shared)} shared genes, n={len(tsub)})")

    # figure
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(STAGE_A) + list(results)
    vals = list(STAGE_A.values()) + [results[k]["c"] for k in results]
    colors = ["#999999", "#999999", "#4C72B0", "#DD8452", "#55A868"]
    ax.bar(labels, vals, color=colors[:len(vals)])
    ax.axhline(STAGE_A["Clinical (staging)"], ls="--", color="black",
               label="clinical staging (0.66)")
    ax.axhline(0.5, ls=":", color="red", label="random (0.5)")
    ax.set_ylim(0.45, 0.75); ax.set_ylabel("C-index (out-of-fold)")
    ax.set_title(f"Does the full transcriptome beat 50 genes?\nExternal validation (TCGA): {ext_c:.2f}")
    ax.legend(); plt.xticks(rotation=20, ha="right"); fig.tight_layout()
    out = os.path.join(config.FIG_DIR, "boost_genes.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")


if __name__ == "__main__":
    main()
