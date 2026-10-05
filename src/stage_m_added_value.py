"""
stage_m_added_value.py — does the image model add anything beyond clinical staging? (Amendment A)

Runs exactly the follow-up analyses fixed in PREREGISTRATION.md §12 (written 2026-10-02, BEFORE any of
them was run), on the full cohort (n = 1,033) out-of-fold risks from stage_l_fullcohort.py:

  A1  Independent prognostic value: Cox model age + stage + T + N, then + image risk.
      Primary test = likelihood-ratio test (LRT). Also + gene risk; site-held-out sensitivity;
      proportional-hazards check; how much of the image risk the clinical variables explain.
  A2  Learned-weight fusion with nested cross-validation (stacking), instead of equal weights.
  A3  Subgroups (exploratory): stage II, stage III, LumA, LumB tested with Holm correction;
      the rest descriptive.
  A4  Slide-size confound on the full cohort (needs the GDC API; skipped with a message if GDC is down).
  A5  Paired-bootstrap CIs for the time-dependent AUC and Brier differences; decision curves.

Every result is written to results/fullcohort/added_value/ and reported whatever its direction.

Run:  python src/stage_m_added_value.py          (A2's inner models are cached after the first run)
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                       # noqa: E402
import make_figures as mf           # noqa: E402  stats helpers, palette, plot style
import stage_e_attmil as e          # noqa: E402
import stage_h_clinical as h        # noqa: E402
import stage_l_fullcohort as L      # noqa: E402  full-cohort loader, recalibration

SEED = config.RANDOM_SEED
OUT = os.path.join(L.OUT, "added_value")
FIGDIR = L.FIGDIR
CACHE = os.path.join(L.OUT, "oof_risks_full.csv")
CLIN_COLS = ["age", "stage", "t_stage", "n_stage"]
log, z = mf.log, mf.z


# ------------------------------------------------------------------ shared
def load():
    """Out-of-fold risks (patient order = sorted IDs, as in stage_l) + imputed clinical variables."""
    df = pd.read_csv(CACHE)
    clin = h.load_clinical().loc[df.patient]
    df = df.join(clin.fillna(clin.median()).reset_index(drop=True))     # same median imputation as before
    raw = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId").loc[df.patient]
    df["stage_grp"] = raw["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(stage_grp).to_numpy()
    df["subtype"] = raw["SUBTYPE"].fillna("Unknown").str.replace("BRCA_", "", regex=False).to_numpy()
    df["full"] = z(df.gene_random) + z(df.img_random) + z(df.clin_random)   # equal-weight model, Entry 13
    return df


def stage_grp(s):
    """Same grouping as Table 1 (make_figures.table1)."""
    s = str(s).upper()
    for k in ("IV", "III", "II", "I"):
        if f"STAGE {k}" in s:
            return f"Stage {k}"
    return "Unknown"


def delta2(a, b, d, ev, n=2000):
    """Paired bootstrap of C(a) − C(b): mean, 95% CI, two-sided p (same resampling as make_figures.delta)."""
    rng = np.random.RandomState(SEED); ds = []
    for _ in range(n):
        s = rng.choice(len(d), len(d), replace=True)
        if ev[s].sum() >= 3:
            ds.append(mf.cidx(a[s], d[s], ev[s]) - mf.cidx(b[s], d[s], ev[s]))
    ds = np.array(ds)
    p = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), p


def holm(p):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    print(df.to_string(index=False))


# ------------------------------------------------------------------ A1 independent prognostic value
def a1(df):
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    print("\n===== A1  Independent prognostic value (Cox; primary = LRT for adding image risk) =====")
    base = df[CLIN_COLS + ["time", "event"]].copy()
    rows = []

    def fit(cols):
        return CoxPHFitter().fit(base.assign(**cols), "time", "event")

    for label, img_col in [("random 5-fold image risk (primary)", "img_random"),
                           ("site-held-out image risk (sensitivity)", "img_site")]:
        m0 = fit({})
        m1 = fit({"image": z(df[img_col])})
        lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
        hr = float(m1.hazard_ratios_["image"]); lo, hi = np.exp(m1.confidence_intervals_.loc["image"].to_numpy())
        rows.append({"analysis": f"+ image vs clinical only [{label}]", "LR_chi2": round(lr, 2),
                     "p_LRT": float(chi2.sf(lr, 1)), "HR_per_SD": round(hr, 2), "HR_lo": round(lo, 2), "HR_hi": round(hi, 2)})
        if img_col == "img_random":
            m2 = fit({"image": z(df.img_random), "gene": z(df.gene_random)})
            lr2 = 2 * (m2.log_likelihood_ - m1.log_likelihood_)
            hr2 = float(m2.hazard_ratios_["gene"]); lo2, hi2 = np.exp(m2.confidence_intervals_.loc["gene"].to_numpy())
            rows.append({"analysis": "+ gene vs clinical + image (secondary)", "LR_chi2": round(lr2, 2),
                         "p_LRT": float(chi2.sf(lr2, 1)), "HR_per_SD": round(hr2, 2), "HR_lo": round(lo2, 2), "HR_hi": round(hi2, 2)})
            m1.summary.round(4).to_csv(os.path.join(OUT, "a1_cox_model1_summary.csv"))
            m2.summary.round(4).to_csv(os.path.join(OUT, "a1_cox_model2_summary.csv"))
            print("\nModel 1 (clinical + image):"); print(m1.summary[["exp(coef)", "exp(coef) lower 95%",
                                                                       "exp(coef) upper 95%", "p"]].round(4).to_string())
            ph = proportional_hazard_test(m1, base.assign(image=z(df.img_random)), time_transform="rank")
            ph.summary.round(4).to_csv(os.path.join(OUT, "a1_ph_test.csv"))
            print("\nProportional-hazards test (Schoenfeld, rank time) for Model 1:")
            print(ph.summary[["test_statistic", "p"]].round(4).to_string())
    save(pd.DataFrame(rows), "a1_lrt.csv")

    # how much of the image risk do the clinical variables explain?
    img = z(df.img_random).to_numpy()
    r2 = LinearRegression().fit(df[CLIN_COLS], img).score(df[CLIN_COLS], img)
    rho = {c: spearmanr(img, df[c]).correlation for c in CLIN_COLS}
    rho["gene risk"] = spearmanr(img, df.gene_random).correlation
    expl = pd.DataFrame([{"quantity": f"Spearman(image risk, {k})", "value": round(v, 3)} for k, v in rho.items()]
                        + [{"quantity": "R2 of image risk explained by age+stage+T+N", "value": round(r2, 3)}])
    print(); save(expl, "a1_image_vs_clinical.csv")


# ------------------------------------------------------------------ A2 nested-CV learned weights
def a2(df):
    print("\n===== A2  Learned-weight fusion (nested CV stacking) =====")
    from lifelines import CoxPHFitter
    pids, dur, evt, bags, G, Xc, sites = L.load_full_cohort()
    assert list(pids) == list(df.patient), "patient order differs from the out-of-fold cache"
    n = len(pids)
    e.SEED = SEED
    outer = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(n)))   # = first-level folds

    # safety check: the cached out-of-fold image risk of outer fold 1 must equal a fresh model
    # trained on that fold's training patients (i.e. reusing the cache is exactly the nested design)
    tr, te = outer[0]
    fresh = L.fit_predict_mil([bags[i] for i in tr], G[tr], dur[tr], evt[tr], [bags[i] for i in te], G[te], True, False)
    dev = float(np.max(np.abs(fresh - df.img_random.to_numpy()[te])))
    log(f"check: fresh fold-1 image model vs cached out-of-fold risk, max |difference| = {dev:.2e}")
    assert dev < 1e-4, "cached out-of-fold risks do not match a fresh retrain"

    zc, zi, zg = z(df.clin_random).to_numpy(), z(df.img_random).to_numpy(), z(df.gene_random).to_numpy()
    stacked = {"Stacked: genes + images + clinical": np.full(n, np.nan), "Stacked: images + clinical": np.full(n, np.nan)}
    wrows = []
    for k, (tr, te) in enumerate(outer):
        cache = os.path.join(OUT, f"a2_inner_fold{k + 1}.csv")
        if os.path.exists(cache):
            inner = pd.read_csv(cache)
        else:
            log(f"outer fold {k + 1}/5: inner CV on {len(tr)} training patients (images, genes, clinical)...")
            bt, Gt, dt, et = [bags[i] for i in tr], G[tr], dur[tr], evt[tr]
            ifolds = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(tr))))
            torch.manual_seed(SEED)
            inner = pd.DataFrame({"idx": tr,
                                  "img": e.run(bt, Gt, dt, et, use_img=True, use_gene=False),
                                  "gene": e.run(bt, Gt, dt, et, use_img=False, use_gene=True),
                                  "clin": h.oof_cox_risk(Xc[tr], dt, et, ifolds)})
            inner.to_csv(cache, index=False)
        assert (inner.idx.to_numpy() == tr).all()
        T, E = dur[tr], evt[tr]
        for name, cols in [("Stacked: genes + images + clinical", ["clin", "img", "gene"]),
                           ("Stacked: images + clinical", ["clin", "img"])]:
            X = pd.DataFrame({c: z(inner[c].to_numpy()) for c in cols})
            m = CoxPHFitter().fit(X.assign(T=T, E=E), "T", "E")
            w = m.params_
            test = {"clin": zc[te], "img": zi[te], "gene": zg[te]}
            stacked[name][te] = sum(w[c] * test[c] for c in cols)
            wrows.append({"model": name, "outer_fold": k + 1, **{f"w_{c}": round(float(w[c]), 3) for c in cols}})
    print("\nLearned weights (Cox coefficients on z-scored risks) per outer fold:")
    save(pd.DataFrame(wrows), "a2_weights.csv")

    d, ev = df.time.to_numpy(float), df.event.to_numpy(bool)
    clin = df.clin_random.to_numpy()
    rows = []
    for name, r in [("Clinical (age + TNM)", clin), ("Equal-weight: genes + images + clinical", df.full.to_numpy()),
                    *stacked.items()]:
        c, lo, hi = mf.ci(r, d, ev)
        row = {"model": name, "c_index": round(c, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)}
        if name != "Clinical (age + TNM)":
            m_, l_, h_, p_ = delta2(r, clin, d, ev)
            row.update({"delta_vs_clinical": round(m_, 3), "delta_lo": round(l_, 3), "delta_hi": round(h_, 3), "p_two_sided": round(p_, 3)})
        rows.append(row)
    print(); save(pd.DataFrame(rows), "a2_stacking.csv")
    pd.DataFrame({"patient": df.patient, **{k: v for k, v in stacked.items()}}).to_csv(
        os.path.join(OUT, "a2_stacked_risks.csv"), index=False)


# ------------------------------------------------------------------ A3 subgroups
def a3(df):
    import matplotlib.pyplot as plt
    print("\n===== A3  Subgroups (exploratory; Holm across the 4 tested) =====")
    tested = [("stage_grp", "Stage II"), ("stage_grp", "Stage III"), ("subtype", "LumA"), ("subtype", "LumB")]
    descriptive = [("stage_grp", "Stage I"), ("stage_grp", "Stage IV"), ("subtype", "Basal"),
                   ("subtype", "Her2"), ("subtype", "Normal")]
    rows = []
    for col, val in tested + descriptive:
        s = df[df[col] == val]
        d, ev = s.time.to_numpy(float), s.event.to_numpy(bool)
        row = {"subgroup": val, "n": len(s), "deaths": int(ev.sum()), "tested": (col, val) in tested}
        for lab, r in [("clinical", s.clin_random), ("image", s.img_random), ("full", s.full)]:
            if ev.sum() >= 3:
                c, lo, hi = mf.ci(r.to_numpy(), d, ev)
                row.update({f"C_{lab}": round(c, 3), f"C_{lab}_lo": round(lo, 3), f"C_{lab}_hi": round(hi, 3)})
        if row["tested"]:
            m_, l_, h_, p_ = delta2(s.full.to_numpy(), s.clin_random.to_numpy(), d, ev)
            row.update({"delta_full_minus_clin": round(m_, 3), "delta_lo": round(l_, 3), "delta_hi": round(h_, 3), "p_raw": p_})
        rows.append(row)
    res = pd.DataFrame(rows)
    t = res.tested
    res.loc[t, "p_holm"] = holm(res.loc[t, "p_raw"].to_numpy())
    res["p_raw"] = res["p_raw"].round(4); res["p_holm"] = res["p_holm"].round(4)
    save(res, "a3_subgroups.csv")

    # figure: clinical vs image vs full C-index per subgroup
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    mf.style(ax); ax.grid(axis="y", visible=False)
    y = np.arange(len(res))[::-1].astype(float)
    ax.axvline(0.5, color=mf.REF, lw=1, ls=(0, (3, 3)))
    for lab, color, mk, off, name in [("clinical", mf.BLUE, "o", 0.18, "Clinical (age + TNM)"),
                                      ("full", mf.ORANGE, "s", -0.18, "Genes + images + clinical")]:
        for yi, (_, r) in zip(y, res.iterrows()):
            if pd.notna(r.get(f"C_{lab}")):
                ax.plot([r[f"C_{lab}_lo"], r[f"C_{lab}_hi"]], [yi + off] * 2, color=color, lw=2)
                ax.plot(r[f"C_{lab}"], yi + off, mk, color=color, ms=7, mec=mf.SURFACE, mew=1.5,
                        label=name if yi == y[0] else None)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.subgroup}  (n={r.n}, {r.deaths} deaths){'' if r.tested else '  *'}" for _, r in res.iterrows()])
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel("C-index within the subgroup, 95% bootstrap CI")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, frameon=False, fontsize=8)
    ax.set_title("Exploratory subgroups  (* descriptive only: fewer than 30 deaths)", loc="left", fontsize=11, pad=10)
    fig.savefig(os.path.join(FIGDIR, "fig10_subgroups.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ A4 slide-size confound
def slide_manifest():
    """Every TCGA-BRCA diagnostic slide on GDC (file id, size, md5). Cached; needs the GDC API once."""
    path = os.path.join(L.OUT, "gdc_dx_slides.csv")
    if not os.path.exists(path):
        import requests
        filt = {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}]}
        params = {"filters": json.dumps(filt), "fields": "file_id,file_size,md5sum,cases.submitter_id",
                  "format": "JSON", "size": "20000"}
        hits = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=120).json()["data"]["hits"]
        pd.DataFrame([{"patient": x["cases"][0]["submitter_id"], "file_id": x["file_id"],
                       "size": x["file_size"], "md5": x.get("md5sum")} for x in hits]).to_csv(path, index=False)
    return pd.read_csv(path)


def a4(df):
    print("\n===== A4  Slide-size confound (full cohort) =====")
    try:
        man = slide_manifest()
    except Exception as ex:
        print(f"GDC API unavailable ({repr(ex)[:80]}). A4 skipped — re-run this script when GDC is back.")
        return
    new = set(pd.read_csv(os.path.join(L.OUT, "frozen_preds.csv")).patient)   # the 789 new patients
    g = man.groupby("patient")["size"].agg(["min", "max", "count"])
    sub = df.join(g, on="patient")
    missing = sub["count"].isna().to_numpy()                                  # no slide listed on GDC now
    is_new = sub.patient.isin(new).to_numpy()
    ambiguous = (~is_new) & (sub["count"].fillna(0).to_numpy() >= 2)
    print(f"slides known exactly: {int((~ambiguous & ~missing).sum())} patients; ambiguous (original, 2+ slides): "
          f"{int(ambiguous.sum())}; not listed on GDC: {int(missing.sum())}")
    ambiguous = ambiguous | missing                                           # never guess a size
    rows = []
    for label, mask, size in [("primary: unambiguous patients only", ~ambiguous, sub["min"].to_numpy()),
                              ("sensitivity: ambiguous = smallest slide", ~missing, sub["min"].to_numpy()),
                              ("sensitivity: ambiguous = largest slide", ~missing,
                               np.where(ambiguous, sub["max"].to_numpy(), sub["min"].to_numpy()))]:
        d, ev = sub.time.to_numpy(float)[mask], sub.event.to_numpy(bool)[mask]
        s = size[mask].astype(float)
        c, lo, hi = mf.ci(s, d, ev)
        rho = spearmanr(sub.img_random.to_numpy()[mask], s).correlation
        rows.append({"analysis": label, "n": int(mask.sum()), "deaths": int(ev.sum()), "C_size_alone": round(c, 3),
                     "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "spearman_image_vs_size": round(rho, 3)})
    res = pd.DataFrame(rows)
    save(res, "a4_slide_size.csv")
    prim = res.iloc[0]
    if prim.ci_lo > 0.5 or prim.ci_hi < 0.5:
        from lifelines import CoxPHFitter
        print("\nSize CI excludes 0.5 -> repeating A1 with log(slide size) added (pre-specified):")
        m = ~ambiguous
        base = sub.loc[m, CLIN_COLS + ["time", "event"]].assign(log_size=np.log(sub.loc[m, "min"].astype(float)))
        m0 = CoxPHFitter().fit(base, "time", "event")
        m1 = CoxPHFitter().fit(base.assign(image=z(sub.loc[m, "img_random"]).to_numpy()), "time", "event")
        lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
        out = pd.DataFrame([{"analysis": "+ image vs clinical + log(size)", "LR_chi2": round(lr, 2), "p_LRT": float(chi2.sf(lr, 1)),
                             "HR_per_SD": round(float(m1.hazard_ratios_["image"]), 2)}])
        save(out, "a4_lrt_size_adjusted.csv")
    else:
        print("Size CI includes 0.5 -> slide size does not predict survival here; no adjusted re-test needed.")


# ------------------------------------------------------------------ A5 remaining statistics
def a5(df):
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    print("\n===== A5  Time-AUC and Brier differences; decision curves =====")
    d, ev = df.time.to_numpy(float), df.event.to_numpy(bool)
    clin, full = df.clin_random.to_numpy(), df.full.to_numpy()
    sv = Surv.from_arrays(event=ev, time=d)
    rng = np.random.RandomState(SEED)

    # (a) time-dependent AUC difference — same time grid as make_figures Fig 4
    times = np.arange(12, min(96, np.percentile(d, 90)) + 1, 3.0)
    times = times[(times > d[ev].min()) & (times < d.max())]
    yr_idx = {yr: int(np.argmin(np.abs(times - 12 * yr))) for yr in (1, 3, 5)}

    def aucs(idx):
        s = Surv.from_arrays(event=ev[idx], time=d[idx])
        a1_, m1_ = cumulative_dynamic_auc(s, s, full[idx], times)
        a0_, m0_ = cumulative_dynamic_auc(s, s, clin[idx], times)
        return [m1_ - m0_] + [a1_[yr_idx[y]] - a0_[yr_idx[y]] for y in (1, 3, 5)]

    point = aucs(np.arange(len(d)))
    boots, skipped = [], 0
    for _ in range(1000):
        s = rng.choice(len(d), len(d), replace=True)
        try:
            boots.append(aucs(s))
        except ValueError:
            skipped += 1
    boots = np.array(boots)
    rows = []
    for j, lab in enumerate(["mean over 1-8 yr", "1 yr", "3 yr", "5 yr"]):
        rows.append({"metric": f"time-dependent AUC, full - clinical ({lab})", "difference": round(point[j], 3),
                     "ci_lo": round(float(np.percentile(boots[:, j], 2.5)), 3),
                     "ci_hi": round(float(np.percentile(boots[:, j], 97.5)), 3)})

    # (b) Brier difference on the Entry 13 recalibrated predictions
    preds = {}
    for hz in L.HORIZONS:
        pc = L.recalibrated_survival(clin, d, ev, hz)
        pf = L.recalibrated_survival(full, d, ev, hz)
        preds[hz] = (pc, pf)
        bc = brier_score(sv, sv, pc.reshape(-1, 1), [hz])[1][0]
        bf = brier_score(sv, sv, pf.reshape(-1, 1), [hz])[1][0]
        log(f"check: Brier at {hz} months (should equal Entry 13): clinical {bc:.4f}, full {bf:.4f}")
        bd = []
        for _ in range(1000):
            s = rng.choice(len(d), len(d), replace=True)
            ss = Surv.from_arrays(event=ev[s], time=d[s])
            try:
                bd.append(brier_score(ss, ss, pf[s].reshape(-1, 1), [hz])[1][0]
                          - brier_score(ss, ss, pc[s].reshape(-1, 1), [hz])[1][0])
            except ValueError:
                pass
        rows.append({"metric": f"Brier at {hz // 12} yr, full - clinical (negative = full better)",
                     "difference": round(bf - bc, 4), "ci_lo": round(float(np.percentile(bd, 2.5)), 4),
                     "ci_hi": round(float(np.percentile(bd, 97.5)), 4)})
    dl = pd.read_csv(os.path.join(L.OUT, "delta_vs_clinical.csv")).set_index("model").loc["Genes + images + clinical"]
    rows.append({"metric": "C-index, full - clinical (Entry 13): largest gain the data allow = upper CI bound",
                 "difference": dl.delta_vs_clinical, "ci_lo": dl.lo, "ci_hi": dl.hi})
    print(f"(time-AUC bootstrap replicates skipped: {skipped})")
    save(pd.DataFrame(rows), "a5_differences.csv")

    # (c) decision curves — net benefit of acting on predicted risk of death by the horizon
    thresholds = np.round(np.arange(0.05, 0.501, 0.01), 2)
    dca, km_all = [], KaplanMeierFitter().fit(d, ev)
    fig, axes = plt.subplots(1, len(L.HORIZONS), figsize=(10.5, 4.4), sharey=True)
    for ax, hz in zip(axes, L.HORIZONS):
        mf.style(ax)
        p_all = 1 - float(km_all.predict(hz))                          # overall risk of death by hz
        for name, surv_pred, color in [("Clinical (age + TNM)", preds[hz][0], mf.BLUE),
                                       ("Genes + images + clinical", preds[hz][1], mf.ORANGE)]:
            risk = 1 - surv_pred
            nb = []
            for pt in thresholds:
                hi_ = risk >= pt
                if hi_.sum() == 0:
                    nb.append(0.0); continue
                s_hi = float(KaplanMeierFitter().fit(d[hi_], ev[hi_]).predict(hz))
                tp, fp = (1 - s_hi) * hi_.mean(), s_hi * hi_.mean()
                nb.append(tp - fp * pt / (1 - pt))
            ax.plot(thresholds, nb, color=color, lw=2, label=name)
            dca += [{"horizon_months": hz, "model": name, "threshold": t, "net_benefit": round(v, 4)} for t, v in zip(thresholds, nb)]
        nb_all = [p_all - (1 - p_all) * pt / (1 - pt) for pt in thresholds]
        ax.plot(thresholds, nb_all, color=mf.REF, lw=1.2, ls=(0, (4, 2)), label="Treat everyone")
        ax.axhline(0, color=mf.TEXT2, lw=1, label="Treat no one")
        dca += [{"horizon_months": hz, "model": "Treat everyone", "threshold": t, "net_benefit": round(v, 4)} for t, v in zip(thresholds, nb_all)]
        ax.set_ylim(-0.02, max(0.05, p_all + 0.02))
        ax.set_xlabel(f"Risk-of-death threshold for acting ({hz // 12}-year)")
        ax.set_title(f"{hz // 12}-year decision curve", loc="left", fontsize=10)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_ylabel("Net benefit")
    fig.suptitle("Decision curves: would acting on the model beat treating everyone or no one?", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig9_decision_curves.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    dca = pd.DataFrame(dca)
    dca.to_csv(os.path.join(OUT, "a5_decision_curves.csv"), index=False)
    wide = dca.pivot_table(index=["horizon_months", "threshold"], columns="model", values="net_benefit")
    print("\nNet benefit at selected thresholds:")
    print(wide.loc[[(h_, t) for h_ in L.HORIZONS for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)]].round(4).to_string())


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    os.makedirs(OUT, exist_ok=True)
    data = load()
    a1(data)
    a3(data)
    a4(data)
    a5(data)
    a2(data)          # slowest (nested CV), so last; inner models are cached per outer fold
    log("done -> results/fullcohort/added_value/")
