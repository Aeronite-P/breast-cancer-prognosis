"""
stage_p_healthcheck.py — gaps found by the 2026-10-21 pre-writing health check.

  h4        H4 (the ORIGINAL primary, PREREGISTRATION.md §4/§7) on the full cohort: does genes + images
            beat the best single modality? Never formally run at n = 1,033 (Entry 13 compared every model
            with clinical only). Run exactly as §7 specified; see the §11 deviation-log entry of 2026-10-21.
  amendment-d   Amendment D (PREREGISTRATION.md §15, written 2026-10-21 before running): A1 repeated with
            (D1) progression-free interval and disease-specific survival, (D2) only the 789 patients not
            used to choose the methods, (D3) PAM50 subtype added; (D4) plain-language median split.
  clinical-verify / clinical-apply
            Stage-parsing fix (§11, 2026-10-21): recompute ONLY the clinical columns of every cached risk
            file (n = 244 and full-cohort out-of-fold risks, frozen-model risks, A2 inner folds) with the
            exact folds and imputation of the original runs. "verify" (run before the fix) must reproduce
            the cached values; "apply" (after the fix) overwrites them. Image and gene columns untouched.

Uses the out-of-fold risks in results/fullcohort/oof_risks_full.csv and the frozen-model risks in
results/fullcohort/frozen_preds.csv (no model is retrained).
Every result is written to results/fullcohort/healthcheck/ and reported whatever its direction.

Run:  python src/stage_p_healthcheck.py h4
      python src/stage_p_healthcheck.py amendment-d
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                       # noqa: E402
import make_figures as mf          # noqa: E402  stats helpers
import stage_l_fullcohort as L      # noqa: E402  full-cohort output folder
import stage_m_added_value as M     # noqa: E402  loader (same imputation), paired bootstrap

OUT = os.path.join(L.OUT, "healthcheck")
log, z = mf.log, mf.z


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    print(df.to_string(index=False))


# ------------------------------------------------------------------ H4 fusion vs best single modality
def h4(df):
    from lifelines import CoxPHFitter
    print("\n===== H4  Genes + images vs the best single modality (images), n = 1,033 =====")
    d, ev = df.time.to_numpy(float), df.event.to_numpy(bool)
    rows = []
    for split in ("random", "site"):
        img, gene = df[f"img_{split}"].to_numpy(), df[f"gene_{split}"].to_numpy()
        fused = z(gene) + z(img)                         # the equal-weight "Genes + images" score (Entry 13)
        c_f, c_i = mf.cidx(fused, d, ev), mf.cidx(img, d, ev)
        m, lo, hi, p_two = M.delta2(fused, img, d, ev)
        _, _, _, p_not_better = mf.delta(fused, img, d, ev)

        base = pd.DataFrame({"time": d, "event": ev, "image": z(img)})
        m0 = CoxPHFitter().fit(base, "time", "event")
        m1 = CoxPHFitter().fit(base.assign(gene=z(gene)), "time", "event")
        lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
        hr = float(m1.hazard_ratios_["gene"]); hlo, hhi = np.exp(m1.confidence_intervals_.loc["gene"].to_numpy())
        rows.append({"split": "random 5-fold (primary)" if split == "random" else "site-held-out (sensitivity)",
                     "C_genes_plus_images": round(c_f, 3), "C_images": round(c_i, 3),
                     "delta_C": round(m, 3), "delta_lo": round(lo, 3), "delta_hi": round(hi, 3),
                     "p_two_sided": round(p_two, 3), "P_not_better": round(p_not_better, 3),
                     "LR_chi2_gene_on_image": round(lr, 2), "p_LRT": float(chi2.sf(lr, 1)),
                     "HR_gene_per_SD": round(hr, 2), "HR_lo": round(hlo, 2), "HR_hi": round(hhi, 2)})
    save(pd.DataFrame(rows), "h4_fusion_vs_best_single.csv")


# ------------------------------------------------------------------ Amendment D
def endpoints(df):
    """TCGA-CDR progression-free interval and disease-specific survival for the cohort, in df's patient order."""
    raw = pd.read_csv(os.path.join(config.RAW_DIR, "tcga_clinical.csv"), index_col="patientId").loc[df.patient]
    return pd.DataFrame({
        "pfi_time": raw.PFS_MONTHS.to_numpy(float),
        "pfi_event": raw.PFS_STATUS.astype(str).str.startswith("1").to_numpy(),
        "dss_time": raw.DSS_MONTHS.to_numpy(float),
        "dss_event": raw.DSS_STATUS.astype(str).str.startswith("1").to_numpy(),
        "dss_known": (raw.DSS_STATUS.notna() & raw.DSS_MONTHS.notna()).to_numpy(),
        "pfi_known": raw.PFS_MONTHS.notna().to_numpy()}, index=df.index)      # TCGA-OL-A66H has no PFI/DSS time


def lrt_image(base, image, extra=None):
    """A1's test: Cox(time, event ~ base covariates [+ extra]) vs the same + z(image). Returns LR, p, HR, CI."""
    from lifelines import CoxPHFitter
    b = base if extra is None else base.join(extra)
    fo = {"step_size": 0.5}         # lifelines' default 0.95 diverged on the PFI fit; same maximum, smaller steps
    m0 = CoxPHFitter().fit(b, "time", "event", fit_options=fo)
    m1 = CoxPHFitter().fit(b.assign(image=z(np.asarray(image, float))), "time", "event", fit_options=fo)
    lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
    lo, hi = np.exp(m1.confidence_intervals_.loc["image"].to_numpy())
    return {"n": len(b), "events": int(b.event.sum()), "LR_chi2": round(lr, 2), "p_LRT": float(chi2.sf(lr, 1)),
            "HR_per_SD": round(float(m1.hazard_ratios_["image"]), 2), "HR_lo": round(lo, 2), "HR_hi": round(hi, 2)}


def amendment_d(df):
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    ep = endpoints(df)
    clin = df[M.CLIN_COLS]

    print("\n===== D1  A1 with other endpoints (random 5-fold image risk, trained on overall survival) =====")
    rows = []
    for label, t, e, keep in [("overall survival (A1 reference)", df.time, df.event, np.ones(len(df), bool)),
                              ("progression-free interval [D1 PRIMARY]", ep.pfi_time, ep.pfi_event, ep.pfi_known.to_numpy()),
                              ("disease-specific survival", ep.dss_time, ep.dss_event, ep.dss_known.to_numpy())]:
        t, e = np.asarray(t, float)[keep], np.asarray(e, bool)[keep]
        b = clin[keep].assign(time=t, event=e)
        r = {"endpoint": label, **lrt_image(b, df.img_random[keep]),
             "C_clinical": round(mf.cidx(df.clin_random.to_numpy()[keep], t, e), 3),
             "C_image": round(mf.cidx(df.img_random.to_numpy()[keep], t, e), 3)}
        rows.append(r)
    save(pd.DataFrame(rows), "d1_other_endpoints.csv")

    print("\n===== D2  Only the 789 patients not used to choose the methods =====")
    fr = pd.read_csv(os.path.join(L.OUT, "frozen_preds.csv")).set_index("patient")
    new = df.patient.isin(fr.index).to_numpy()
    sub, epn = df[new], ep[new]
    frozen_img = fr.loc[sub.patient, "img"].to_numpy()
    rows, everyone = [], np.ones(len(sub), bool)
    for label, image, t, e, keep in [
            ("frozen image risk (trained on the 244), overall survival", frozen_img, sub.time, sub.event, everyone),
            ("frozen image risk (trained on the 244), progression-free interval", frozen_img,
             epn.pfi_time, epn.pfi_event, epn.pfi_known.to_numpy()),
            ("random 5-fold image risk restricted to the 789, overall survival", sub.img_random,
             sub.time, sub.event, everyone)]:
        b = sub[M.CLIN_COLS][keep].assign(time=np.asarray(t, float)[keep], event=np.asarray(e, bool)[keep])
        rows.append({"analysis": label, **lrt_image(b, np.asarray(image, float)[keep])})
    save(pd.DataFrame(rows), "d2_new_patients_only.csv")

    print("\n===== D3  A1 with PAM50 subtype added (overall survival) =====")
    dummies = pd.get_dummies(df.subtype, prefix="pam50", dtype=float).drop(columns="pam50_LumA")
    b = clin.assign(time=df.time.to_numpy(float), event=df.event.to_numpy(bool))
    m0 = CoxPHFitter().fit(b, "time", "event")
    ms = CoxPHFitter().fit(b.join(dummies), "time", "event")
    lr_s = 2 * (ms.log_likelihood_ - m0.log_likelihood_)
    rows = [{"analysis": "+ image vs clinical + PAM50 subtype [D3]", **lrt_image(b, df.img_random, dummies)},
            {"analysis": "reference: + image vs clinical only (A1)", **lrt_image(b, df.img_random)},
            {"analysis": f"+ PAM50 subtype vs clinical only (descriptive, {dummies.shape[1]} df)", "n": len(b),
             "events": int(b.event.sum()), "LR_chi2": round(lr_s, 2), "p_LRT": float(chi2.sf(lr_s, dummies.shape[1]))}]
    save(pd.DataFrame(rows), "d3_pam50_adjusted.csv")

    print("\n===== D4  Below- vs above-median image risk (overall survival, descriptive) =====")
    high = (df.img_random > df.img_random.median()).to_numpy()
    t, e = df.time.to_numpy(float), df.event.to_numpy(bool)
    rows = []
    for name, g in [("below median", ~high), ("above median", high)]:
        km = KaplanMeierFitter().fit(t[g], e[g])
        rows.append({"group": name, "n": int(g.sum()), "deaths": int(e[g].sum()),
                     "percent_died": round(100 * e[g].mean(), 1),
                     "KM_5yr_survival": round(float(km.survival_function_at_times(60).iloc[0]), 3)})
    lr = logrank_test(t[high], t[~high], e[high], e[~high])
    cph = CoxPHFitter().fit(pd.DataFrame({"time": t, "event": e, "high": high.astype(float)}), "time", "event")
    lo, hi = np.exp(cph.confidence_intervals_.loc["high"].to_numpy())
    rows.append({"group": "above vs below: HR [95% CI], log-rank p",
                 "n": len(t), "deaths": int(e.sum()),
                 "percent_died": f"HR {cph.hazard_ratios_['high']:.2f} [{lo:.2f}, {hi:.2f}]",
                 "KM_5yr_survival": f"p = {lr.p_value:.2g}"})
    save(pd.DataFrame(rows), "d4_median_split.csv")

    print("\n===== D5  Scan magnification (tiles were read at full scan resolution) =====")
    from scipy.stats import mannwhitneyu
    mag = pd.read_csv(os.path.join(OUT, "scan_magnification.csv")).set_index("patient").loc[df.patient]
    grp = mag.app_mag.map(lambda v: f"{int(v)}x" if v == v else "unknown").to_numpy()
    img = z(df.img_random.to_numpy())
    rows = []
    for g in sorted(set(grp)):
        k = grp == g
        rows.append({"group": g, "n": int(k.sum()), "deaths": int(e[k].sum()),
                     "mean_z_image_risk": round(float(img[k].mean()), 3),
                     "C_image_within": round(mf.cidx(img[k], t[k], e[k]), 3) if e[k].sum() >= 30 else np.nan})
    if {"20x", "40x"} <= set(grp):
        p_mw = mannwhitneyu(img[grp == "40x"], img[grp == "20x"]).pvalue
        rows.append({"group": "40x vs 20x image risk, Mann-Whitney p", "n": np.nan, "deaths": np.nan,
                     "mean_z_image_risk": float(p_mw), "C_image_within": np.nan})
    save(pd.DataFrame(rows), "d5_magnification_groups.csv")
    dm = pd.get_dummies(pd.Series(grp, index=df.index), prefix="mag", dtype=float)
    dm = dm.drop(columns="mag_40x" if "mag_40x" in dm else dm.columns[0])              # 40x = reference group
    b = clin.assign(time=t, event=e)
    print(); save(pd.DataFrame([{"analysis": "+ image vs clinical + scan magnification [D5]", **lrt_image(b, df.img_random, dm)},
                                {"analysis": "reference: + image vs clinical only (A1)", **lrt_image(b, df.img_random)}]),
                  "d5_magnification_adjusted.csv")


# ------------------------------------------------------------------ scan magnification (slide headers)
def tiff_description(file_id, session):
    """ImageDescription (TIFF tag 270) of a slide's first page, read from GDC with a few small Range requests."""
    import struct
    url = f"https://api.gdc.cancer.gov/data/{file_id}"

    def get(off, n):
        r = session.get(url, headers={"Range": f"bytes={off}-{off + n - 1}"}, timeout=120)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError(f"no Range support (HTTP {r.status_code})")
        return r.content

    h = get(0, 16)
    bo = "<" if h[:2] == b"II" else ">"
    big = struct.unpack(bo + "H", h[2:4])[0] == 43                       # BigTIFF
    ifd = struct.unpack(bo + ("Q" if big else "I"), h[8:16] if big else h[4:8])[0]
    nsz, esz = (8, 20) if big else (2, 12)
    n = struct.unpack(bo + ("Q" if big else "H"), get(ifd, nsz))[0]
    ents = get(ifd + nsz, n * esz)
    for i in range(n):
        e = ents[i * esz:(i + 1) * esz]
        if struct.unpack(bo + "H", e[:2])[0] != 270:
            continue
        count = struct.unpack(bo + ("Q" if big else "I"), e[4:12] if big else e[4:8])[0]
        val = e[12:20] if big else e[8:12]
        raw = val[:count] if count <= len(val) else get(struct.unpack(bo + ("Q" if big else "I"), val)[0], count)
        return raw.decode("latin-1", "replace")
    return None


def scan_magnification():
    """Scanner magnification (Aperio AppMag) and microns per pixel for every study slide. The tiles were read at
    level 0 (full scan resolution), so a 40x scan gives ~0.25 um/px tiles and a 20x scan ~0.5 um/px."""
    import re
    import requests
    path = os.path.join(OUT, "scan_magnification.csv")
    man = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(L.OUT)), "reproducibility", "cohort_manifest.csv"))
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["patient", "note"])
    done = done[~done.note.fillna("").str.startswith("lookup failed")]          # re-runs retry failed lookups
    rows = done.to_dict("records")
    s = requests.Session()
    for r in man[~man.patient.isin(done.patient)].itertuples():
        row = {"patient": r.patient, "slide_status": r.slide_status, "app_mag": np.nan, "mpp": np.nan, "note": ""}
        if r.slide_status != "exact":
            row["note"] = "ambiguous slide (2+ candidates); not looked up"
        else:
            try:
                desc = tiff_description(r.slide_file_id, s) or ""
                m = re.search(r"AppMag\s*=\s*([\d.]+)", desc); p = re.search(r"MPP\s*=\s*([\d.]+)", desc)
                row.update(app_mag=float(m.group(1)) if m else np.nan, mpp=float(p.group(1)) if p else np.nan,
                           note="" if m else "no AppMag in header")
            except Exception as ex:                                       # record, don't stop
                row["note"] = f"lookup failed: {str(ex)[:80]}"
        rows.append(row)
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(path, index=False); log(f"{len(rows)} / {len(man)} slides")
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    print(out.app_mag.value_counts(dropna=False).to_string())
    print(out.note.value_counts().to_string())


# ------------------------------------------------------------------ stage-parsing fix
def recompute_clinical(write):
    """Recompute the clinical risk columns of every cached file with the current load_clinical().
    write=False compares with the caches (max |difference| must be ~0 before the fix)."""
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    import stage_e_attmil as e
    import stage_h_clinical as h

    clin = h.load_clinical()
    root = os.path.dirname(os.path.dirname(L.OUT))
    rel = lambda p: os.path.relpath(p, root)
    rows = []

    def put(df, col, new, path):
        rows.append({"file": rel(path), "column": col, "max_abs_change": float(np.max(np.abs(df[col].to_numpy() - new)))})
        df[col] = new

    # first-level out-of-fold risks (same patient order, folds and median imputation as make_figures.compute)
    path244, path_full = os.path.join(os.path.dirname(L.OUT), "oof_risks.csv"), os.path.join(L.OUT, "oof_risks_full.csv")
    for path in (path244, path_full):
        df = pd.read_csv(path)
        pids, n = df.patient.tolist(), len(df)
        C = clin.loc[pids]
        Xc = C.fillna(C.median()).to_numpy(float)
        d, ev = df.time.to_numpy(float), df.event.to_numpy(bool)
        kfold = lambda s: list(KFold(5, shuffle=True, random_state=s).split(np.arange(n)))
        sites = np.array([p.split("-")[1] for p in pids])
        jobs = {"clin_random": (d, ev, kfold(mf.SEED)),
                "clin_site": (d, ev, list(GroupKFold(5).split(np.arange(n), groups=sites))),
                "clin_neg": (df.time_neg.to_numpy(float), df.event_neg.to_numpy(bool), kfold(mf.SEED))}
        jobs.update({f"clin_s{s}": (d, ev, kfold(s)) for s in mf.EXTRA_SEEDS})
        for col, (dd, ee, folds) in jobs.items():
            put(df, col, h.oof_cox_risk(Xc, dd, ee, folds), path)
        if write:
            df.to_csv(path, index=False)

    # frozen models: clinical Cox fit on the original 244 (their medians and scaling), applied to the 789
    path_fr = os.path.join(L.OUT, "frozen_preds.csv")
    fr, surv = pd.read_csv(path_fr), e.load_survival()
    tr = sorted(pd.read_csv(path244).patient)
    X_tr, med = L.clinical_matrix(clin, tr)
    X_te, _ = L.clinical_matrix(clin, fr.patient.tolist(), medians=med)
    sc = StandardScaler().fit(X_tr)
    cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(
        sc.transform(X_tr), Surv.from_arrays(event=surv.loc[tr, "event"].to_numpy(bool), time=surv.loc[tr, "time"].to_numpy(float)))
    put(fr, "clin", cox.predict(sc.transform(X_te)), path_fr)
    if write:
        fr.to_csv(path_fr, index=False)

    # A2 inner folds (stage_m_added_value.a2): clinical inner out-of-fold risk inside each outer training set
    full = pd.read_csv(path_full)
    C = clin.loc[full.patient]
    Xc = C.fillna(C.median()).to_numpy(float)
    d, ev = full.time.to_numpy(float), full.event.to_numpy(bool)
    for k in range(1, 6):
        path = os.path.join(M.OUT, f"a2_inner_fold{k}.csv")
        inner = pd.read_csv(path)
        tr = inner.idx.to_numpy()
        ifolds = list(KFold(5, shuffle=True, random_state=mf.SEED).split(np.arange(len(tr))))
        put(inner, "clin", h.oof_cox_risk(Xc[tr], d[tr], ev[tr], ifolds), path)
        if write:
            inner.to_csv(path, index=False)

    # repeated negative control (stage_l_fullcohort.negative_control_repeats): clinical C-index per shuffle
    path = os.path.join(L.OUT, "negative_control_repeats.csv")
    neg = pd.read_csv(path)
    folds = list(KFold(5, shuffle=True, random_state=mf.SEED).split(np.arange(len(d))))
    new = []
    for k in neg.permutation:
        perm = np.random.RandomState(int(k)).permutation(len(d))
        new.append(round(mf.cidx(h.oof_cox_risk(Xc, d[perm], ev[perm], folds), d[perm], ev[perm]), 3))
    put(neg, "clinical", np.array(new), path)
    if write:
        neg.to_csv(path, index=False)

    print(); save(pd.DataFrame(rows), f"clinical_recompute_{'applied' if write else 'verify'}.csv")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what in ("clinical-verify", "clinical-apply"):
        recompute_clinical(write=what == "clinical-apply")
        log("done -> results/fullcohort/healthcheck/")
        sys.exit()
    if what == "scan-magnification":
        scan_magnification()
        log("done -> results/fullcohort/healthcheck/scan_magnification.csv")
        sys.exit()
    data = M.load()
    if what == "h4":
        h4(data)
    elif what == "amendment-d":
        amendment_d(data)
    else:
        sys.exit("usage: python src/stage_p_healthcheck.py h4 | amendment-d | clinical-verify | clinical-apply")
    log("done -> results/fullcohort/healthcheck/")
