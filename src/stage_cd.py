"""
stage_cd.py — Stage C verification (confound gate) + Stage D fusion.

Run this once Stage C's image features are collected (data/stage_c_matched/*.npy). It:
  1. CONFOUND GATE: checks slide size alone no longer predicts survival (should be ~0.5 now
     that the sample is size-matched). If this is high, the imaging result isn't trustworthy.
  2. IMAGE-ONLY C-index (+ bootstrap 95% CI).
  3. GENES-ONLY C-index on the SAME patients (PAM50), for a fair comparison.
  4. FUSION (genes + images) C-index — the headline: does combining beat either alone?

Everything uses 5-fold out-of-fold C-index and the same leakage-safe, numerically-stable
pipeline as the rest of the project.

Usage:  python src/stage_cd.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

IMG_DIR = os.path.join(config.DATA_DIR, "stage_c_matched")
SEED = config.RANDOM_SEED


def load_survival() -> pd.DataFrame:
    c = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId")
    s = pd.DataFrame(index=c.index)
    s["time"] = pd.to_numeric(c["OS_MONTHS"], errors="coerce")
    s["event"] = c["OS_STATUS"].astype(str).str.startswith("1")
    return s.dropna(subset=["time"])[lambda d: d["time"] > 0]


def load_genes() -> pd.DataFrame:
    """PAM50 z-scores per patient (from the Stage A TCGA table)."""
    c = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId")
    genes = [g for g in config.PAM50 if g in c.columns]
    return c[genes].apply(pd.to_numeric, errors="coerce")


def load_images() -> dict:
    return {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(f"{IMG_DIR}/*.npy")}


def get_slide_sizes() -> dict:
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}]}
    params = {"filters": json.dumps(filt), "fields": "file_size,cases.submitter_id",
              "format": "JSON", "size": "20000"}
    hits = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=120).json()["data"]["hits"]
    return {h["cases"][0]["submitter_id"]: h.get("file_size", 0) for h in hits}


def reduce_block(Xtr, Xte, n_comp):
    """Leakage-safe: fit scaler+PCA on train, apply to test. Returns (Ztr, Zte)."""
    n = min(n_comp, len(Xtr) - 2, Xtr.shape[1])
    pipe = make_pipeline(StandardScaler(), PCA(n_components=max(1, n), svd_solver="full"))
    return pipe.fit_transform(Xtr), pipe.transform(Xte)


def cv_cindex(blocks, dur, evt, n_comp=6):
    """blocks: list of full feature matrices (each reduced separately per fold, then concatenated).
    Returns (pooled C-index, lo95, hi95)."""
    kf = KFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(dur), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in kf.split(blocks[0]):
            parts_tr, parts_te = [], []
            for X in blocks:
                a, b = reduce_block(X[tr], X[te], n_comp)
                parts_tr.append(a); parts_te.append(b)
            Ztr = np.hstack(parts_tr); Zte = np.hstack(parts_te)
            m = CoxPHSurvivalAnalysis(alpha=10.0).fit(Ztr, Surv.from_arrays(event=evt[tr], time=dur[tr]))
            oof[te] = m.predict(Zte)
    if not np.isfinite(oof).all():
        return None, None, None
    c = concordance_index_censored(evt, dur, oof)[0]
    rng = np.random.RandomState(SEED); bs = []
    for _ in range(1000):
        s = rng.choice(np.arange(len(dur)), len(dur), replace=True)
        if evt[s].sum() >= 3:
            bs.append(concordance_index_censored(evt[s], dur[s], oof[s])[0])
    return c, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    surv = load_survival()
    img = load_images()
    genes = load_genes()
    sizes = get_slide_sizes()

    # patients with image + genes + survival
    pids = [p for p in img if p in surv.index and p in genes.index and genes.loc[p].notna().all()]
    if len(pids) < 20:
        print(f"only {len(pids)} usable patients — collect more before fusion."); return
    dur = surv.loc[pids, "time"].to_numpy(float)
    evt = surv.loc[pids, "event"].to_numpy(bool)
    Ximg = np.nan_to_num(np.vstack([img[p] for p in pids]).astype(float))
    Ximg = Ximg[:, np.argsort(Ximg.var(0))[::-1][:256]]     # top-256 image dims
    Xgen = genes.loc[pids].to_numpy(float)
    sz = np.array([sizes.get(p, 0) for p in pids], float)
    print(f"patients={len(pids)}  deaths={int(evt.sum())}\n")

    # 1. CONFOUND GATE
    cs = concordance_index_censored(evt, dur, sz)[0]
    print("=== 1. Confound gate: does SLIDE SIZE alone predict survival? ===")
    print(f"   size-alone C-index = {cs:.3f}   "
          f"({'OK — confound controlled' if abs(cs - 0.5) < 0.08 else 'WARNING — size still leaks'})")
    print(f"   mean size: alive {sz[~evt].mean()/1e6:.0f}MB vs dead {sz[evt].mean()/1e6:.0f}MB\n")

    # 2-4. models
    ci, cil, cih = cv_cindex([Ximg], dur, evt)
    cg, cgl, cgh = cv_cindex([Xgen], dur, evt)
    cf, cfl, cfh = cv_cindex([Ximg, Xgen], dur, evt)
    print("=== Survival prediction (out-of-fold C-index, 0.5 = chance) ===")
    print(f"   2. Images only        : {ci:.3f}  95% CI [{cil:.2f}, {cih:.2f}]")
    print(f"   3. Genes only (PAM50) : {cg:.3f}  95% CI [{cgl:.2f}, {cgh:.2f}]")
    print(f"   4. FUSION (genes+imgs): {cf:.3f}  95% CI [{cfl:.2f}, {cfh:.2f}]")
    best_single = max(ci, cg)
    print(f"\n   Fusion vs best single modality: {cf - best_single:+.3f} "
          f"({'better' if cf > best_single else 'not better'})")
    print("\n(Small N -> wide CIs. Preliminary, single-cohort.)")


if __name__ == "__main__":
    main()
