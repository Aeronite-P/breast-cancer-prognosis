"""
stage_h_clinical.py — the HONEST baseline: does the multi-modal model beat the CLINICIAN?

Stage A showed genes barely beat clinical staging. The real question for a prognosis paper is
not "genes vs images" but "does anything beat what a doctor already has (age + TNM stage)?".
This adds a clinical arm and asks whether late fusion of genes+images+clinical beats clinical alone.

Reuses the attention-MIL image risk and gene risk (stage_e), computes an out-of-fold clinical
Cox risk on the SAME patients/folds, and compares every combination.

Run:  python src/stage_h_clinical.py
"""
from __future__ import annotations

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config          # noqa: E402
import stage_e_attmil as e  # noqa: E402
import torch

SEED = config.RANDOM_SEED


def load_clinical():
    c = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId")

    def stage_ord(s):
        s = str(s).upper()
        for pat, val in [("IV", 4), ("III", 3), ("II", 2), ("I", 1)]:
            if re.search(r"STAGE\s+" + pat + r"\b", s) or s.strip() == pat:
                return val
        return np.nan

    def tn_ord(s, letter):
        m = re.search(letter + r"(\d)", str(s).upper())
        return int(m.group(1)) if m else np.nan

    out = pd.DataFrame(index=c.index)
    out["age"] = pd.to_numeric(c["AGE"], errors="coerce")
    out["stage"] = c["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(stage_ord)
    out["t_stage"] = c["PATH_T_STAGE"].map(lambda x: tn_ord(x, "T"))
    out["n_stage"] = c["PATH_N_STAGE"].map(lambda x: tn_ord(x, "N"))
    return out


def oof_cox_risk(X, dur, evt, folds):
    """Out-of-fold Cox risk on a dense feature matrix, using given fold splits."""
    oof = np.full(len(dur), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            m = CoxPHSurvivalAnalysis(alpha=1.0).fit(
                sc.transform(X[tr]), Surv.from_arrays(event=evt[tr], time=dur[tr]))
            oof[te] = m.predict(sc.transform(X[te]))
    return oof


def ci(oof, dur, evt):
    c = concordance_index_censored(evt, dur, oof)[0]
    rng = np.random.RandomState(SEED); bs = []
    for _ in range(1000):
        s = rng.choice(np.arange(len(dur)), len(dur), replace=True)
        if evt[s].sum() >= 3:
            bs.append(concordance_index_censored(evt[s], dur[s], oof[s])[0])
    return c, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def beats(a, b, dur, evt, label):
    """Bootstrap ΔC (a − b): does model a beat model b?"""
    rng = np.random.RandomState(SEED); d = []
    for _ in range(2000):
        s = rng.choice(np.arange(len(dur)), len(dur), replace=True)
        if evt[s].sum() >= 3:
            d.append(concordance_index_censored(evt[s], dur[s], a[s])[0]
                     - concordance_index_censored(evt[s], dur[s], b[s])[0])
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not sig"
    print(f"  {label:38s} ΔC={d.mean():+.3f}  [{lo:+.3f},{hi:+.3f}]  {sig}")


def main():
    warnings.simplefilter("ignore")
    surv, genes, tiles = e.load_survival(), e.load_genes(), e.load_tiles()
    clin = load_clinical()
    pids = [p for p in tiles if p in surv.index and p in genes.index
            and genes.loc[p].notna().all() and p in clin.index]
    dur = surv.loc[pids, "time"].to_numpy(float)
    evt = surv.loc[pids, "event"].to_numpy(bool)
    bags = [torch.tensor(tiles[p]) for p in pids]
    G = StandardScaler().fit_transform(genes.loc[pids].to_numpy(float)).astype(np.float32)

    # clinical matrix (median-impute)
    C = clin.loc[pids].copy()
    for col in C.columns:
        C[col] = C[col].fillna(C[col].median())
    Xc = C.to_numpy(float)
    cov = C.notna().mean().min()
    print(f"patients={len(pids)}, deaths={int(evt.sum())}, clinical cols={list(C.columns)}\n", flush=True)

    folds = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(pids))))

    print("attention-MIL image risk...", flush=True)
    r_img = e.run(bags, G, dur, evt, use_img=True, use_gene=False)
    print("gene risk...", flush=True)
    r_gene = e.run(bags, G, dur, evt, use_img=False, use_gene=True)
    print("clinical Cox risk...", flush=True)
    r_clin = oof_cox_risk(Xc, dur, evt, folds)

    # late-fusion combos (mean of z-scored risks)
    r_gi = z(r_gene) + z(r_img)
    r_gc = z(r_gene) + z(r_clin)
    r_gic = z(r_gene) + z(r_img) + z(r_clin)
    r_ic = z(r_img) + z(r_clin)

    print("\n=== Out-of-fold C-index (0.5 = chance) ===")
    for name, r in [("Clinical (age+TNM stage)", r_clin), ("Genes (PAM50)", r_gene),
                    ("Images (attention-MIL)", r_img), ("Late: genes+images", r_gi),
                    ("Late: genes+clinical", r_gc), ("Late: images+clinical", r_ic),
                    ("Late: genes+images+clinical", r_gic)]:
        c = ci(r, dur, evt)
        print(f"  {name:32s} {c[0]:.3f}  [{c[1]:.2f}, {c[2]:.2f}]")

    print("\n=== Does the multi-modal model beat the CLINICIAN? (bootstrap ΔC vs clinical alone) ===")
    beats(r_gene, r_clin, dur, evt, "genes            vs clinical")
    beats(r_img, r_clin, dur, evt, "images           vs clinical")
    beats(r_gi, r_clin, dur, evt, "genes+images     vs clinical")
    beats(r_gic, r_clin, dur, evt, "genes+img+clin   vs clinical")
    print("\n(If nothing significantly beats clinical, that's the honest headline.)")


if __name__ == "__main__":
    main()
