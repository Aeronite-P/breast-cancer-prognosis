"""
stage_l_fullcohort.py — re-run the whole analysis on the FULL TCGA-BRCA cohort (n = 1,033).

The n = 244 results (LAB_NOTEBOOK Entries 9-12) came from a size-matched case/control sample with
50% deaths. This script repeats EVERY analysis, unchanged, on all 1,033 patients with a diagnostic
slide, survival data and complete PAM50 genes (146 deaths, 14%) — an unselected population:

  1. The same out-of-fold models (random 5-fold, site-held-out, shuffled-label control, 5 seeds)
     and the same figures and tables, by re-using make_figures.py. Outputs go to
     results/fullcohort/ and results/figures/fullcohort/, so the n = 244 files are never touched.
  2. NEW  Frozen-model validation: models trained ONLY on the original 244 patients, then tested
     on the 789 new patients they never saw. Does the n = 244 finding transfer to unselected
     patients?
  3. NEW  Calibration: do predicted 3- and 5-year survival probabilities match what happened?
  4. Side-by-side comparison with the n = 244 results.

No method changes: same models, hyperparameters, fold scheme and seeds as the n = 244 analysis.
Patients are sorted by ID so fold assignment is deterministic.

Run:  python src/stage_l_fullcohort.py              (computes the cache if missing, then everything)
      python src/stage_l_fullcohort.py --recompute  (retrain from scratch)
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                       # noqa: E402
import make_figures as mf           # noqa: E402  every n = 244 analysis + figure, re-used as is
import stage_e_attmil as e          # noqa: E402
import stage_h_clinical as h        # noqa: E402

SEED = config.RANDOM_SEED
RES_244 = config.RESULTS_DIR                                  # where the n = 244 results live
ORIG_DIR = os.path.join(config.DATA_DIR, "stage_c_tiles")       # original 244 (size-matched)
FULL_DIR = os.path.join(config.DATA_DIR, "stage_c_tiles_full")  # 789 new patients
OUT = os.path.join(config.RESULTS_DIR, "fullcohort")
FIGDIR = os.path.join(config.FIG_DIR, "fullcohort")
HORIZONS = (36, 60)                                           # calibration at 3 and 5 years (months)
log = mf.log


# ------------------------------------------------------------------ data
def load_tiles_all():
    """patient -> (n_tiles, 768) Phikon features, from both folders; plus which folder it came from."""
    tiles, origin = {}, {}
    for d, tag in [(ORIG_DIR, "original"), (FULL_DIR, "new")]:
        for f in glob.glob(f"{d}/*.npy"):
            p = os.path.basename(f)[:-4]
            tiles[p] = np.load(f).astype(np.float32)
            origin[p] = tag
    return tiles, origin


def clinical_matrix(clin, pids, medians=None):
    """Median-impute the clinical columns (with the given medians, e.g. training-set ones)."""
    C = clin.loc[pids].copy()
    med = C.median() if medians is None else medians
    return C.fillna(med).to_numpy(float), med


def load_full_cohort():
    """Same filters and preprocessing as make_figures.load_cohort, over all 1,033 patients."""
    surv, genes, clin = e.load_survival(), e.load_genes(), h.load_clinical()
    tiles, _ = load_tiles_all()
    pids = sorted(p for p in tiles if p in surv.index and p in genes.index
                  and genes.loc[p].notna().all() and p in clin.index)
    dur = surv.loc[pids, "time"].to_numpy(float)
    evt = surv.loc[pids, "event"].to_numpy(bool)
    bags = [torch.from_numpy(tiles[p]) for p in pids]        # from_numpy: no second copy in memory
    G = StandardScaler().fit_transform(genes.loc[pids].to_numpy(float)).astype(np.float32)
    Xc, _ = clinical_matrix(clin, pids)
    sites = np.array([p.split("-")[1] for p in pids])
    return pids, dur, evt, bags, G, Xc, sites


def use_fullcohort_outputs():
    """Point make_figures at the full cohort and at separate output folders."""
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    mf.CACHE = os.path.join(OUT, "oof_risks_full.csv")
    mf.FIG = FIGDIR
    mf.COHORT_NAME = "full cohort: every patient with a diagnostic slide, survival and PAM50 data"
    mf.load_cohort = load_full_cohort
    config.RESULTS_DIR = OUT            # make_figures writes its tables to config.RESULTS_DIR


# ------------------------------------------------------------------ frozen-model validation
def fit_predict_mil(bags_tr, G_tr, d_tr, e_tr, bags_te, G_te, use_img, use_gene, epochs=120, lr=1e-3):
    """Train one attention-MIL model on ALL training patients (same loop and settings as
    stage_e_attmil.run), then score the test patients."""
    torch.manual_seed(SEED)
    model = e.ABMIL(gene_dim=G_tr.shape[1] if use_gene else 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    tt, ee = torch.tensor(d_tr), torch.tensor(e_tr.astype(float))

    def fwd(bags, G, i):
        return model(bags[i] if use_img else torch.zeros((1, 768)),
                     torch.tensor(G[i]) if use_gene else None)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        risks = torch.stack([fwd(bags_tr, G_tr, i) for i in range(len(bags_tr))])
        e.cox_loss(risks, tt, ee).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return np.array([float(fwd(bags_te, G_te, i)) for i in range(len(bags_te))])


def frozen_validation():
    """Train on the original 244, test on the 789 new patients. Every preprocessing step
    (gene scaling, clinical imputation and scaling) is fit on the 244 only."""
    cache = os.path.join(OUT, "frozen_preds.csv")
    if not os.path.exists(cache) or "--recompute" in sys.argv:
        surv, genes, clin = e.load_survival(), e.load_genes(), h.load_clinical()
        tiles, origin = load_tiles_all()
        ok = lambda p: (p in surv.index and p in genes.index and genes.loc[p].notna().all()
                        and p in clin.index)
        tr = sorted(p for p in tiles if origin[p] == "original" and ok(p))
        te = sorted(p for p in tiles if origin[p] == "new" and ok(p))
        log(f"frozen validation: train {len(tr)} original, test {len(te)} new")
        d_tr, e_tr = surv.loc[tr, "time"].to_numpy(float), surv.loc[tr, "event"].to_numpy(bool)
        gsc = StandardScaler().fit(genes.loc[tr].to_numpy(float))
        G_tr = gsc.transform(genes.loc[tr].to_numpy(float)).astype(np.float32)
        G_te = gsc.transform(genes.loc[te].to_numpy(float)).astype(np.float32)
        b_tr = [torch.from_numpy(tiles[p]) for p in tr]
        b_te = [torch.from_numpy(tiles[p]) for p in te]
        log("frozen: images..."); r_img = fit_predict_mil(b_tr, G_tr, d_tr, e_tr, b_te, G_te, True, False)
        log("frozen: genes...");  r_gene = fit_predict_mil(b_tr, G_tr, d_tr, e_tr, b_te, G_te, False, True)
        X_tr, med = clinical_matrix(clin, tr)
        X_te, _ = clinical_matrix(clin, te, medians=med)
        csc = StandardScaler().fit(X_tr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(csc.transform(X_tr), Surv.from_arrays(event=e_tr, time=d_tr))
        r_clin = cox.predict(csc.transform(X_te))
        pd.DataFrame({"patient": te, "time": surv.loc[te, "time"].to_numpy(float),
                      "event": surv.loc[te, "event"].to_numpy(bool),
                      "img": r_img, "gene": r_gene, "clin": r_clin}).to_csv(cache, index=False)
    df = pd.read_csv(cache)
    d, ev = df["time"].to_numpy(float), df["event"].to_numpy(bool)
    z = mf.z                                  # z-scored within the test set (no outcomes used)
    arms = {"Clinical (age + TNM)": df["clin"].to_numpy(), "Genes (PAM50)": df["gene"].to_numpy(),
            "Images (attention-MIL)": df["img"].to_numpy()}
    arms["Genes + images"] = z(arms["Genes (PAM50)"]) + z(arms["Images (attention-MIL)"])
    arms["Images + clinical"] = z(arms["Images (attention-MIL)"]) + z(arms["Clinical (age + TNM)"])
    arms["Genes + images + clinical"] = (z(arms["Genes (PAM50)"]) + z(arms["Images (attention-MIL)"])
                                         + z(arms["Clinical (age + TNM)"]))
    rows = []
    for name, r in arms.items():
        c, lo, hi = mf.ci(r, d, ev)
        rows.append({"model": name, "c_index": round(c, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    m, lo, hi, p = mf.delta(arms["Genes + images + clinical"], arms["Clinical (age + TNM)"], d, ev)
    rows.append({"model": "Full model minus clinical (delta C)", "c_index": round(m, 3),
                 "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "frozen_validation.csv"), index=False)
    print(f"\n=== Frozen models (trained on original 244) tested on {len(df)} new patients "
          f"({int(ev.sum())} deaths) ===\n{out.to_string(index=False)}\n  P(full not better) = {p:.3f}")
    return out


# ------------------------------------------------------------------ calibration
def recalibrated_survival(score, d, ev, horizon):
    """Predicted survival probability at `horizon` for every patient, cross-fitted: in each of 5
    folds, a one-variable Cox model is fit on the (already out-of-fold) risk score of the training
    patients and applied to the held-out ones. So no patient's probability uses their own outcome."""
    from lifelines import CoxPHFitter
    pred = np.full(len(d), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(d))):
        cph = CoxPHFitter().fit(pd.DataFrame({"T": d[tr], "E": ev[tr], "s": score[tr]}), "T", "E")
        pred[te] = cph.predict_survival_function(pd.DataFrame({"s": score[te]}), times=[horizon]).to_numpy()[0]
    return pred


def calibration():
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from sksurv.metrics import brier_score
    df = pd.read_csv(mf.CACHE)
    d, ev = df["time"].to_numpy(float), df["event"].to_numpy(bool)
    rand = mf.models(df, "random")
    sv = Surv.from_arrays(event=ev, time=d)
    km_all = KaplanMeierFitter().fit(d, ev)
    rows, fig_rows = [], []
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(10.5, 4.6))
    for ax, hz in zip(axes, HORIZONS):
        mf.style(ax)
        ax.plot([0.45, 1], [0.45, 1], color=mf.REF, lw=1, ls=(0, (3, 3)))
        null_p = float(km_all.predict(hz))
        rows.append({"model": "No model (everyone gets the overall survival rate)", "horizon_months": hz,
                     "brier": round(float(brier_score(sv, sv, np.full((len(d), 1), null_p), [hz])[1][0]), 4)})
        for name, color, mk in [("Clinical (age + TNM)", mf.BLUE, "o"),
                                ("Genes + images + clinical", mf.ORANGE, "s")]:
            p = recalibrated_survival(rand[name], d, ev, hz)
            bs = float(brier_score(sv, sv, p.reshape(-1, 1), [hz])[1][0])
            rows.append({"model": name, "horizon_months": hz, "brier": round(bs, 4)})
            q = pd.qcut(p, 5, labels=False, duplicates="drop")
            for k in sorted(set(q)):
                m = q == k
                kmf = KaplanMeierFitter().fit(d[m], ev[m])
                obs = float(kmf.predict(hz))
                lo, hi = kmf.confidence_interval_survival_function_.asof(hz).to_numpy()
                fig_rows.append({"model": name, "horizon_months": hz, "quintile": int(k) + 1,
                                 "n": int(m.sum()), "predicted": round(float(p[m].mean()), 3),
                                 "observed": round(obs, 3), "obs_lo": round(float(lo), 3), "obs_hi": round(float(hi), 3)})
                ax.plot([p[m].mean()] * 2, [lo, hi], color=color, lw=1.5, alpha=0.6)
                ax.plot(p[m].mean(), obs, mk, color=color, ms=8, mec=mf.SURFACE, mew=1.5,
                        label=f"{name}  (Brier {bs:.3f})" if k == min(q) else None)
        ax.set_xlim(0.45, 1.0); ax.set_ylim(0.45, 1.02)
        ax.set_xlabel(f"Predicted {hz // 12}-year survival (mean per fifth of patients)")
        ax.set_ylabel(f"Observed {hz // 12}-year survival (Kaplan-Meier, 95% CI)")
        ax.set_title(f"{hz // 12}-year calibration (dashed line = perfect)", loc="left", fontsize=10)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle(f"Calibration: do predicted survival chances match what happened? (n = {len(d)})",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_calibration.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "calibration_brier.csv"), index=False)
    pd.DataFrame(fig_rows).to_csv(os.path.join(OUT, "calibration_quintiles.csv"), index=False)
    print("\n=== Calibration (Brier score: lower is better) ===\n" + pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(fig_rows).to_string(index=False))


# ------------------------------------------------------------------ repeated negative control
def negative_control_repeats(n_perm=10):
    """The single shuffled-label control gave the gene arm 0.566, in the top 1% of chance. Is that a
    fluke of one shuffle or a real problem? Retrain the gene arm on n_perm DIFFERENT shuffles
    (permutation seeds 1..n_perm); a sound pipeline should scatter around 0.5."""
    cache = os.path.join(OUT, "negative_control_repeats.csv")
    if os.path.exists(cache) and "--recompute" not in sys.argv:
        res = pd.read_csv(cache)
    else:
        pids, dur, evt, bags, G, Xc, sites = load_full_cohort()
        e.SEED = SEED
        rows = []
        for k in range(1, n_perm + 1):
            perm = np.random.RandomState(k).permutation(len(pids))
            dn, en = dur[perm], evt[perm]
            torch.manual_seed(SEED)
            r = e.run(bags, G, dn, en, use_img=False, use_gene=True)
            folds = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(pids))))
            rc = h.oof_cox_risk(Xc, dn, en, folds)
            rows.append({"permutation": k, "genes": round(mf.cidx(r, dn, en), 3),
                         "clinical": round(mf.cidx(rc, dn, en), 3)})
            log(f"shuffle {k}: genes {rows[-1]['genes']}, clinical {rows[-1]['clinical']}")
        res = pd.DataFrame(rows)
        res.to_csv(cache, index=False)
    for col in ("genes", "clinical"):
        print(f"repeated negative control, {col}: mean {res[col].mean():.3f}, "
              f"range {res[col].min():.3f}-{res[col].max():.3f} over {len(res)} shuffles")
    return res


# ------------------------------------------------------------------ comparison with n = 244
def compare():
    import matplotlib.pyplot as plt
    a = pd.read_csv(os.path.join(RES_244, "metrics_final.csv"))
    b = pd.read_csv(os.path.join(OUT, "metrics_final.csv"))
    m = a.merge(b, on=["model", "split"], suffixes=("_244", "_1033"))
    m["change"] = (m["c_index_1033"] - m["c_index_244"]).round(3)
    m.to_csv(os.path.join(OUT, "comparison_244_vs_1033.csv"), index=False)
    print("\n=== n = 244 (size-matched) vs n = 1,033 (full cohort) ===")
    print(m[["model", "split", "c_index_244", "c_index_1033", "change"]].to_string(index=False))

    r = m[m.split == "random 5-fold"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    mf.style(ax); ax.grid(axis="y", visible=False)
    y = np.arange(len(r))[::-1].astype(float)
    ax.axvline(0.5, color=mf.REF, lw=1, ls=(0, (3, 3)))
    for col, color, mk, off, lab in [("244", mf.BLUE, "o", 0.15, "n = 244 (size-matched, 50% deaths)"),
                                     ("1033", mf.ORANGE, "s", -0.15, "n = 1,033 (full cohort, 14% deaths)")]:
        for yi, (_, row) in zip(y, r.iterrows()):
            ax.plot([row[f"ci_lo_{col}"], row[f"ci_hi_{col}"]], [yi + off] * 2, color=color, lw=2)
            ax.plot(row[f"c_index_{col}"], yi + off, mk, color=color, ms=8, mec=mf.SURFACE, mew=2,
                    label=lab if yi == y[0] else None)
    ax.set_yticks(y); ax.set_yticklabels(r.model)
    ax.set_xlim(0.45, 0.8)
    ax.set_xlabel("C-index (out-of-fold, random 5-fold) with 95% bootstrap CI")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, frameon=False, fontsize=8)   # below the axis, clear of the data
    ax.set_title("Same models, hand-picked cohort vs the full unselected cohort", loc="left", fontsize=11, pad=10)
    fig.savefig(os.path.join(FIGDIR, "fig8_244_vs_full.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    use_fullcohort_outputs()
    if "--recompute" in sys.argv or not os.path.exists(mf.CACHE):
        mf.compute()
    mf.plot()
    frozen_validation()
    calibration()
    negative_control_repeats()
    compare()
    log("all full-cohort outputs written to results/fullcohort/ and results/figures/fullcohort/")
