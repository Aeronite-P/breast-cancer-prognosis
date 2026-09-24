"""
stage_e_attmil.py — Attention-MIL survival on Phikon tile features (+ fusion with genes).

Each patient is a "bag" of tile embeddings (n_tiles x 768). An attention network learns a
weight for every tile (which tumor regions matter), pools them into one slide vector, and a
Cox head predicts survival. Compares:
  - Images (attention-MIL on Phikon tiles)
  - Genes only (PAM50)
  - Fusion (attention-pooled image vector + genes)
5-fold out-of-fold C-index + bootstrap CI, like the rest of the project.

Run:  python src/stage_e_attmil.py
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

TILES_DIR = os.path.join(config.DATA_DIR, "stage_c_tiles")
SEED = config.RANDOM_SEED
torch.manual_seed(SEED)


def load_survival():
    c = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId")
    s = pd.DataFrame(index=c.index)
    s["time"] = pd.to_numeric(c["OS_MONTHS"], errors="coerce")
    s["event"] = c["OS_STATUS"].astype(str).str.startswith("1")
    return s.dropna(subset=["time"])[lambda d: d["time"] > 0]


def load_genes():
    c = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId")
    genes = [g for g in config.PAM50 if g in c.columns]
    return c[genes].apply(pd.to_numeric, errors="coerce")


def load_tiles():
    return {os.path.basename(f)[:-4]: np.load(f).astype(np.float32)
            for f in glob.glob(f"{TILES_DIR}/*.npy")}


class ABMIL(nn.Module):
    """Gated attention-MIL pooling + Cox risk head, optionally fused with gene features."""
    def __init__(self, d=768, hidden=128, gene_dim=0, p=0.4):
        super().__init__()
        self.V = nn.Linear(d, hidden)
        self.U = nn.Linear(d, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(p)
        self.risk = nn.Linear(d + gene_dim, 1)

    def forward(self, H, g=None):                       # H: (n_tiles, d)
        a = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H)))   # gated attention (n,1)
        a = torch.softmax(a, dim=0)
        z = (a * H).sum(0)                              # (d,)
        z = self.drop(z)
        if g is not None:
            z = torch.cat([z, g])
        return self.risk(z).squeeze()


def cox_loss(risks, times, events):
    order = torch.argsort(times, descending=True)
    r = risks[order]
    e = events[order].float()
    log_cum = torch.logcumsumexp(r, dim=0)
    n = e.sum()
    return -((r - log_cum) * e).sum() / (n + 1e-8)


def run(bags, genes_arr, dur, evt, use_img=True, use_gene=False, epochs=120, lr=1e-3):
    """5-fold out-of-fold risk via attention-MIL. bags: list of (n_tiles,768) tensors."""
    n = len(bags)
    kf = KFold(5, shuffle=True, random_state=SEED)
    oof = np.full(n, np.nan)
    gene_dim = genes_arr.shape[1] if use_gene else 0
    for tr, te in kf.split(np.arange(n)):
        torch.manual_seed(SEED)
        model = ABMIL(gene_dim=gene_dim)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        tt = torch.tensor(dur[tr]); te_ev = torch.tensor(evt[tr].astype(float))
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            risks = []
            for i in tr:
                H = bags[i] if use_img else torch.zeros((1, 768))
                g = torch.tensor(genes_arr[i]) if use_gene else None
                risks.append(model(H, g))
            risks = torch.stack(risks)
            loss = cox_loss(risks, tt, te_ev)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                H = bags[i] if use_img else torch.zeros((1, 768))
                g = torch.tensor(genes_arr[i]) if use_gene else None
                oof[i] = float(model(H, g))
    return oof


def cindex(oof, dur, evt):
    c = concordance_index_censored(evt, dur, oof)[0]
    rng = np.random.RandomState(SEED); bs = []
    for _ in range(500):
        s = rng.choice(np.arange(len(dur)), len(dur), replace=True)
        if evt[s].sum() >= 3:
            bs.append(concordance_index_censored(evt[s], dur[s], oof[s])[0])
    return c, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    warnings.simplefilter("ignore")
    surv, genes, tiles = load_survival(), load_genes(), load_tiles()
    pids = [p for p in tiles if p in surv.index and p in genes.index and genes.loc[p].notna().all()]
    if len(pids) < 40:
        print(f"only {len(pids)} patients with tiles — wait for more (need ~40+)."); return
    dur = surv.loc[pids, "time"].to_numpy(float)
    evt = surv.loc[pids, "event"].to_numpy(bool)
    bags = [torch.tensor(tiles[p]) for p in pids]
    G = StandardScaler().fit_transform(genes.loc[pids].to_numpy(float)).astype(np.float32)
    print(f"patients={len(pids)}, deaths={int(evt.sum())}, tiles/patient≈{int(np.mean([b.shape[0] for b in bags]))}\n")

    print("training attention-MIL (images)...", flush=True)
    ci = cindex(run(bags, G, dur, evt, use_img=True, use_gene=False), dur, evt)
    print("training genes-only...", flush=True)
    cg = cindex(run(bags, G, dur, evt, use_img=False, use_gene=True), dur, evt)
    print("training fusion (images + genes)...", flush=True)
    cf = cindex(run(bags, G, dur, evt, use_img=True, use_gene=True), dur, evt)

    print("\n=== Attention-MIL survival (Phikon), out-of-fold C-index ===")
    print(f"  Images (attention-MIL) : {ci[0]:.3f}  95% CI [{ci[1]:.2f}, {ci[2]:.2f}]")
    print(f"  Genes only (PAM50)     : {cg[0]:.3f}  95% CI [{cg[1]:.2f}, {cg[2]:.2f}]")
    print(f"  FUSION (images+genes)  : {cf[0]:.3f}  95% CI [{cf[1]:.2f}, {cf[2]:.2f}]")
    print(f"\n  Fusion vs best single: {cf[0] - max(ci[0], cg[0]):+.3f}")


if __name__ == "__main__":
    main()
