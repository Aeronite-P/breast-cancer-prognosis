"""
stage_g_latefusion.py — if images & genes are COMPLEMENTARY, does LATE fusion beat both?

stage_f showed the image-risk and gene-risk scores are mostly INDEPENDENT (r~0.22) and that
images even rescue cases genes miss. That is the classic setup where fusion should help — yet
our joint concat-fusion tied genes. Hypothesis: the concat head is the bottleneck, and the
simplest fusion (average the two out-of-fold risk scores) will beat either alone.

Compares, all out-of-fold with bootstrap CIs on the SAME patients:
  images-only, genes-only, concat-fusion (as before), and LATE fusion (mean of z-scored risks).

Run:  python src/stage_g_latefusion.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config          # noqa: E402
import stage_e_attmil as e  # noqa: E402
import torch


def ci(oof, dur, evt, seed=config.RANDOM_SEED):
    c = concordance_index_censored(evt, dur, oof)[0]
    rng = np.random.RandomState(seed); bs = []
    for _ in range(1000):
        s = rng.choice(np.arange(len(dur)), len(dur), replace=True)
        if evt[s].sum() >= 3:
            bs.append(concordance_index_censored(evt[s], dur[s], oof[s])[0])
    return c, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def z(x):
    return (x - x.mean()) / x.std()


def main():
    warnings.simplefilter("ignore")
    surv, genes, tiles = e.load_survival(), e.load_genes(), e.load_tiles()
    pids = [p for p in tiles if p in surv.index and p in genes.index and genes.loc[p].notna().all()]
    dur = surv.loc[pids, "time"].to_numpy(float)
    evt = surv.loc[pids, "event"].to_numpy(bool)
    bags = [torch.tensor(tiles[p]) for p in pids]
    G = StandardScaler().fit_transform(genes.loc[pids].to_numpy(float)).astype(np.float32)
    print(f"patients={len(pids)}, deaths={int(evt.sum())}\n", flush=True)

    print("images-only risk...", flush=True)
    r_img = e.run(bags, G, dur, evt, use_img=True, use_gene=False)
    print("genes-only risk...", flush=True)
    r_gene = e.run(bags, G, dur, evt, use_img=False, use_gene=True)
    print("concat-fusion risk...", flush=True)
    r_cat = e.run(bags, G, dur, evt, use_img=True, use_gene=True)

    # LATE fusion: simple mean of the two independent standardized risk scores
    r_late = z(r_img) + z(r_gene)

    ci_img = ci(r_img, dur, evt)
    ci_gene = ci(r_gene, dur, evt)
    ci_cat = ci(r_cat, dur, evt)
    ci_late = ci(r_late, dur, evt)

    print("\n=== Out-of-fold C-index (0.5 = chance) ===")
    print(f"  Images (attention-MIL)   : {ci_img[0]:.3f}  95% CI [{ci_img[1]:.2f}, {ci_img[2]:.2f}]")
    print(f"  Genes only (PAM50)       : {ci_gene[0]:.3f}  95% CI [{ci_gene[1]:.2f}, {ci_gene[2]:.2f}]")
    print(f"  Concat fusion (joint)    : {ci_cat[0]:.3f}  95% CI [{ci_cat[1]:.2f}, {ci_cat[2]:.2f}]")
    print(f"  LATE fusion (mean risks) : {ci_late[0]:.3f}  95% CI [{ci_late[1]:.2f}, {ci_late[2]:.2f}]")
    best_single = max(ci_img[0], ci_gene[0])
    print(f"\n  Late fusion vs best single : {ci_late[0] - best_single:+.3f}")
    print(f"  Late fusion vs concat      : {ci_late[0] - ci_cat[0]:+.3f}")

    # bootstrap significance: does late fusion beat the best single arm?
    rng = np.random.RandomState(config.RANDOM_SEED); diffs = []
    best = r_gene if ci_gene[0] >= ci_img[0] else r_img
    for _ in range(2000):
        s = rng.choice(np.arange(len(dur)), len(dur), replace=True)
        if evt[s].sum() >= 3:
            a = concordance_index_censored(evt[s], dur[s], r_late[s])[0]
            b = concordance_index_censored(evt[s], dur[s], best[s])[0]
            diffs.append(a - b)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float((diffs <= 0).mean())
    print(f"\n  ΔC (late − best single): mean {diffs.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  bootstrap P(late does NOT beat best) = {p:.3f}"
          f"   {'-> significant improvement' if hi < 0 or lo > 0 else '-> not significant'}")


if __name__ == "__main__":
    main()
