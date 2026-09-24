"""
stage_j_metrics.py — rigor checks (1.3) + clinical metrics (1.4) for the late-fusion model.

1.3 Negative control : shuffle the survival labels -> every arm must collapse to ~0.5 (no leakage).
1.3 Multi-seed        : re-run late fusion across seeds -> is the C-index stable, not cherry-picked?
1.4 KM risk groups    : split patients by median predicted risk -> Kaplan-Meier + log-rank p-value.
1.4 Time-dependent AUC: survival AUC at 1/3/5 years (sksurv cumulative_dynamic_auc).

Reuses the attention-MIL trainer from stage_e_attmil. Run:  python src/stage_j_metrics.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config          # noqa: E402
import stage_e_attmil as e  # noqa: E402
import torch


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def late_risk(bags, G, dur, evt):
    ri = e.run(bags, G, dur, evt, use_img=True, use_gene=False)
    rg = e.run(bags, G, dur, evt, use_img=False, use_gene=True)
    return z(ri) + z(rg), ri, rg


def cidx(risk, dur, evt):
    return concordance_index_censored(evt, dur, risk)[0]


def main():
    warnings.simplefilter("ignore")
    surv, genes, tiles = e.load_survival(), e.load_genes(), e.load_tiles()
    pids = [p for p in tiles if p in surv.index and p in genes.index and genes.loc[p].notna().all()]
    dur = surv.loc[pids, "time"].to_numpy(float)
    evt = surv.loc[pids, "event"].to_numpy(bool)
    bags = [torch.tensor(tiles[p]) for p in pids]
    G = StandardScaler().fit_transform(genes.loc[pids].to_numpy(float)).astype(np.float32)
    n = len(pids)
    print(f"patients={n}, deaths={int(evt.sum())}\n", flush=True)

    # ---- real model (seed 42) ----
    print("training real late-fusion (seed 42)...", flush=True)
    e.SEED = 42; torch.manual_seed(42)
    r_late, r_img, r_gene = late_risk(bags, G, dur, evt)
    c_real = cidx(r_late, dur, evt)
    print(f"  real late-fusion C-index = {c_real:.3f}\n", flush=True)

    # ---- 1.3 NEGATIVE CONTROL: shuffle labels ----
    print("=== 1.3 Negative control (shuffled survival labels) ===", flush=True)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    dur_s, evt_s = dur[perm], evt[perm]           # break the tile<->outcome link
    e.SEED = 42; torch.manual_seed(42)
    r_late_s, _, _ = late_risk(bags, G, dur_s, evt_s)
    c_shuf = cidx(r_late_s, dur_s, evt_s)
    print(f"  shuffled-label late-fusion C-index = {c_shuf:.3f}   "
          f"({'PASS ~0.5, no leakage' if abs(c_shuf-0.5) < 0.06 else 'CHECK — not near 0.5'})\n", flush=True)

    # ---- 1.3 MULTI-SEED stability ----
    print("=== 1.3 Multi-seed stability (late fusion) ===", flush=True)
    seeds = [1, 7, 42, 123, 2024]
    cs = []
    for s in seeds:
        e.SEED = s; torch.manual_seed(s)
        rl, _, _ = late_risk(bags, G, dur, evt)
        c = cidx(rl, dur, evt); cs.append(c)
        print(f"  seed {s:>4}: C-index = {c:.3f}", flush=True)
    cs = np.array(cs)
    print(f"  -> mean {cs.mean():.3f} ± {cs.std():.3f}  (range {cs.min():.3f}-{cs.max():.3f})\n", flush=True)

    # ---- 1.4 KM risk groups + log-rank ----
    print("=== 1.4 Kaplan-Meier risk stratification (median split on real risk) ===", flush=True)
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        hi = r_late >= np.median(r_late)
        lr = logrank_test(dur[hi], dur[~hi], evt[hi], evt[~hi])
        kmf = KaplanMeierFitter()
        kmf.fit(dur[hi], evt[hi]); med_hi = kmf.median_survival_time_
        kmf.fit(dur[~hi], evt[~hi]); med_lo = kmf.median_survival_time_
        print(f"  high-risk n={int(hi.sum())} (deaths {int(evt[hi].sum())}), "
              f"low-risk n={int((~hi).sum())} (deaths {int(evt[~hi].sum())})")
        print(f"  median OS: high={med_hi:.0f} mo vs low={med_lo:.0f} mo")
        print(f"  log-rank p = {lr.p_value:.4f}   "
              f"({'significant separation' if lr.p_value < 0.05 else 'not significant'})\n", flush=True)
    except Exception as ex:
        print(f"  lifelines unavailable/failed: {repr(ex)[:60]}\n")

    # ---- 1.4 Time-dependent AUC ----
    print("=== 1.4 Time-dependent AUC (1/3/5 yr) ===", flush=True)
    try:
        sv = Surv.from_arrays(event=evt, time=dur)
        times = np.array([12.0, 36.0, 60.0])
        times = times[(times > dur[evt].min()) & (times < dur.max())]
        auc, mean_auc = cumulative_dynamic_auc(sv, sv, r_late, times)
        for t, a in zip(times, np.atleast_1d(auc)):
            print(f"  AUC @ {int(t)} mo = {a:.3f}")
        print(f"  integrated mean AUC = {mean_auc:.3f}", flush=True)
    except Exception as ex:
        print(f"  time-AUC failed: {repr(ex)[:60]}")


if __name__ == "__main__":
    main()
