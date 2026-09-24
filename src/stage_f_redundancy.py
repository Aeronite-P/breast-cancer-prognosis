"""
stage_f_redundancy.py — WHY doesn't fusion beat genes?

Fusion (images+genes) tied genes alone, even though each modality predicts survival on its
own. The usual reason: the two modalities are REDUNDANT — they rank the same patients as
high-risk, so combining them adds no new information. This script tests that directly.

It reuses the attention-MIL trainer to get out-of-fold RISK SCORES for images-only and
genes-only, then asks:
  1. How correlated are the two risk scores? (Pearson + Spearman)
  2. Do they agree on WHO is high-risk? (top-quartile overlap)
  3. Is there a subgroup where images beat genes? (patients genes gets wrong)

Run:  python src/stage_f_redundancy.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config          # noqa: E402
import stage_e_attmil as e  # reuse load_survival/genes/tiles + run() + ABMIL  # noqa: E402

import torch


def main():
    warnings.simplefilter("ignore")
    surv, genes, tiles = e.load_survival(), e.load_genes(), e.load_tiles()
    pids = [p for p in tiles if p in surv.index and p in genes.index and genes.loc[p].notna().all()]
    dur = surv.loc[pids, "time"].to_numpy(float)
    evt = surv.loc[pids, "event"].to_numpy(bool)
    bags = [torch.tensor(tiles[p]) for p in pids]
    G = StandardScaler().fit_transform(genes.loc[pids].to_numpy(float)).astype(np.float32)
    print(f"patients={len(pids)}, deaths={int(evt.sum())}\n", flush=True)

    print("getting out-of-fold risk scores (images-only)...", flush=True)
    r_img = e.run(bags, G, dur, evt, use_img=True, use_gene=False)
    print("getting out-of-fold risk scores (genes-only)...", flush=True)
    r_gene = e.run(bags, G, dur, evt, use_img=False, use_gene=True)

    # standardize risk scores so correlation is scale-free
    zi = (r_img - r_img.mean()) / r_img.std()
    zg = (r_gene - r_gene.mean()) / r_gene.std()

    pear = pearsonr(zi, zg)[0]
    spear = spearmanr(zi, zg)[0]

    print("\n=== 1. Are the two risk scores redundant? ===")
    print(f"   Pearson  r = {pear:+.3f}   (1.0 = perfectly redundant, 0 = independent)")
    print(f"   Spearman r = {spear:+.3f}   (rank agreement)")
    shared = pear ** 2
    print(f"   -> image risk shares ~{shared*100:.0f}% of its variance with gene risk")

    # 2. do they flag the SAME high-risk patients?
    q = 0.75
    hi_img = zi >= np.quantile(zi, q)
    hi_gene = zg >= np.quantile(zg, q)
    overlap = (hi_img & hi_gene).sum()
    union = (hi_img | hi_gene).sum()
    print("\n=== 2. Do they agree on WHO is high-risk (top quartile)? ===")
    print(f"   images flag {hi_img.sum()}, genes flag {hi_gene.sum()}, "
          f"both flag {overlap}  (Jaccard {overlap/union:.2f})")

    # 3. is there signal images have that genes don't? (residual test)
    #    on patients GENES ranks WRONG, does the image score still separate them?
    print("\n=== 3. Do images help WHERE genes fail? ===")
    med = np.median(zg)
    # among gene-predicted-low-risk patients, can images still find the deaths?
    lowg = zg < med
    if lowg.sum() > 20 and evt[lowg].sum() >= 3:
        c_img_in_lowg = concordance_index_censored(evt[lowg], dur[lowg], zi[lowg])[0]
        print(f"   among gene-LOW-risk patients (n={int(lowg.sum())}, "
              f"deaths={int(evt[lowg].sum())}): image-only C-index = {c_img_in_lowg:.3f}")
        print("   (>0.55 would mean images rescue cases genes miss; ~0.5 = no extra help)")

    print("\n=== VERDICT ===")
    if pear >= 0.5:
        print("   Strongly redundant: image and gene risk largely track the same patients,")
        print("   which is exactly why fusion can't beat the better single modality.")
    elif pear >= 0.25:
        print("   Partly redundant: some shared signal, some independent — fusion gains are")
        print("   modest and easily washed out at this sample size.")
    else:
        print("   Mostly independent risk scores — redundancy is NOT the explanation;")
        print("   fusion likely fails from too few patients / a weak fusion head, not overlap.")


if __name__ == "__main__":
    main()
