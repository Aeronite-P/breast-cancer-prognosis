"""
stage_i_leavesite.py — EXTERNAL-ish validation with NO second cohort: leave-site-out.

A true second WSI breast cohort doesn't exist publicly on GDC (TCGA-BRCA is the only one).
The accepted substitute: TCGA patients come from ~25 tissue-source-sites (different hospitals,
scanners, staining protocols). We validate ACROSS sites — every test patient comes from an
institution the model NEVER trained on (GroupKFold by site). This measures whether the signal
survives real cross-site distribution shift, which is what external validation is really testing.

Compares site-held-out C-index for images / genes / late fusion, against the random-split
numbers from stage_g (0.603 / 0.611 / 0.639) to quantify the generalization gap.

Run:  python src/stage_i_leavesite.py
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config          # noqa: E402
import stage_e_attmil as e  # reuse ABMIL, cox_loss, loaders  # noqa: E402

SEED = config.RANDOM_SEED
torch.manual_seed(SEED)


def run_grouped(bags, genes_arr, dur, evt, groups, use_img, use_gene, epochs=120, lr=1e-3):
    """Attention-MIL OOF risk where each test fold's SITES are absent from training."""
    n = len(bags)
    gkf = GroupKFold(n_splits=5)
    oof = np.full(n, np.nan)
    gene_dim = genes_arr.shape[1] if use_gene else 0
    for tr, te in gkf.split(np.arange(n), groups=groups):
        torch.manual_seed(SEED)
        model = e.ABMIL(gene_dim=gene_dim)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        tt = torch.tensor(dur[tr]); tev = torch.tensor(evt[tr].astype(float))
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            risks = []
            for i in tr:
                H = bags[i] if use_img else torch.zeros((1, 768))
                g = torch.tensor(genes_arr[i]) if use_gene else None
                risks.append(model(H, g))
            loss = e.cox_loss(torch.stack(risks), tt, tev)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                H = bags[i] if use_img else torch.zeros((1, 768))
                g = torch.tensor(genes_arr[i]) if use_gene else None
                oof[i] = float(model(H, g))
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


def main():
    warnings.simplefilter("ignore")
    surv, genes, tiles = e.load_survival(), e.load_genes(), e.load_tiles()
    pids = [p for p in tiles if p in surv.index and p in genes.index and genes.loc[p].notna().all()]
    dur = surv.loc[pids, "time"].to_numpy(float)
    evt = surv.loc[pids, "event"].to_numpy(bool)
    bags = [torch.tensor(tiles[p]) for p in pids]
    G = StandardScaler().fit_transform(genes.loc[pids].to_numpy(float)).astype(np.float32)
    sites = np.array([p.split("-")[1] for p in pids])   # tissue-source-site per patient
    print(f"patients={len(pids)}, deaths={int(evt.sum())}, sites={len(set(sites))}", flush=True)
    print("Each test fold's institutions are unseen in training.\n", flush=True)

    print("site-held-out images...", flush=True)
    r_img = run_grouped(bags, G, dur, evt, sites, use_img=True, use_gene=False)
    print("site-held-out genes...", flush=True)
    r_gene = run_grouped(bags, G, dur, evt, sites, use_img=False, use_gene=True)
    r_late = z(r_img) + z(r_gene)

    ci_i, ci_g, ci_l = ci(r_img, dur, evt), ci(r_gene, dur, evt), ci(r_late, dur, evt)
    print("\n=== SITE-HELD-OUT C-index (external-ish; vs random-split from stage_g) ===")
    print(f"  Images       : {ci_i[0]:.3f}  [{ci_i[1]:.2f}, {ci_i[2]:.2f}]   (random-split was 0.603)")
    print(f"  Genes        : {ci_g[0]:.3f}  [{ci_g[1]:.2f}, {ci_g[2]:.2f}]   (random-split was 0.611)")
    print(f"  Late fusion  : {ci_l[0]:.3f}  [{ci_l[1]:.2f}, {ci_l[2]:.2f}]   (random-split was 0.639)")
    print("\n  A small drop is expected and healthy; near-0.5 would mean the signal was site-specific.")


if __name__ == "__main__":
    main()
