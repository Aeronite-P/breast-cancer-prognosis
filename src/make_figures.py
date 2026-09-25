"""
make_figures.py — publication figures, Table 1, and robustness checks for the multi-modal model.

Step 1 (slow, cached): computes every out-of-fold risk score ONCE, on the same 244 patients:
  - random 5-fold      : images (attention-MIL), genes (PAM50), clinical (age + TNM, Cox)
  - site-held-out      : same three arms, GroupKFold by tissue-source site (unseen hospitals)
  - negative control   : same three arms trained on SHUFFLED survival labels (must be ~0.5)
  - multi-seed         : full model re-run with 5 seeds (is the C-index stable?)
  -> results/oof_risks.csv
Step 2 (fast): draws everything from the cache, so every plotted number comes from code.
  Fig 2  C-index forest plot, random split vs site-held-out, 95% bootstrap CI
  Fig 3  Kaplan-Meier: high vs low predicted risk, clinical model vs multi-modal model
  Fig 4  time-dependent AUC (1-8 years): clinical vs multi-modal
  Fig 5  complementarity: image risk vs gene risk
  Table 1  cohort characteristics;  results/metrics_final.csv, results/robustness.csv

Late fusion = sum of z-scored risk scores (same as stage_g / stage_h).

Run:  python src/make_figures.py              (computes if no cache, then plots)
      python src/make_figures.py --recompute  (force recompute)
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                       # noqa: E402
import stage_e_attmil as e          # noqa: E402  attention-MIL trainer + loaders
import stage_h_clinical as h        # noqa: E402  clinical loader + OOF Cox
import stage_i_leavesite as ls      # noqa: E402  site-grouped attention-MIL

SEED = config.RANDOM_SEED
EXTRA_SEEDS = [1, 7, 123, 2024]     # plus SEED (42) = 5 seeds
CACHE = os.path.join(config.RESULTS_DIR, "oof_risks.csv")
FIG = config.FIG_DIR

# validated two-slot palette (dataviz reference palette, all checks pass)
BLUE, ORANGE = "#2a78d6", "#eb6834"
TEXT, TEXT2, REF, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


# ------------------------------------------------------------------ data
def load_cohort():
    surv, genes, tiles = e.load_survival(), e.load_genes(), e.load_tiles()
    clin = h.load_clinical()
    # same patient order as stage_h (loader order) so fold assignment — and therefore every
    # C-index — reproduces the numbers logged in LAB_NOTEBOOK Entries 9-10
    pids = [p for p in tiles if p in surv.index and p in genes.index
            and genes.loc[p].notna().all() and p in clin.index]
    dur = surv.loc[pids, "time"].to_numpy(float)
    evt = surv.loc[pids, "event"].to_numpy(bool)
    bags = [torch.tensor(tiles[p]) for p in pids]
    G = StandardScaler().fit_transform(genes.loc[pids].to_numpy(float)).astype(np.float32)
    C = clin.loc[pids].copy()
    for col in C.columns:                              # median-impute, as in stage_h
        C[col] = C[col].fillna(C[col].median())
    sites = np.array([p.split("-")[1] for p in pids])
    return pids, dur, evt, bags, G, C.to_numpy(float), sites


# ------------------------------------------------------------------ step 1: compute
def compute():
    warnings.simplefilter("ignore")
    pids, dur, evt, bags, G, Xc, sites = load_cohort()
    n = len(pids)
    log(f"cohort: {n} patients, {int(evt.sum())} deaths, {len(set(sites))} sites")
    out = pd.DataFrame({"patient": pids, "time": dur, "event": evt, "site": sites})

    def arms(d, ev, seed, folds, tag):
        e.SEED = seed; torch.manual_seed(seed)
        log(f"{tag}: images (attention-MIL)...")
        out[f"img_{tag}"] = e.run(bags, G, d, ev, use_img=True, use_gene=False)
        log(f"{tag}: genes...")
        out[f"gene_{tag}"] = e.run(bags, G, d, ev, use_img=False, use_gene=True)
        out[f"clin_{tag}"] = h.oof_cox_risk(Xc, d, ev, folds)
        out.to_csv(CACHE, index=False)               # save progress after every block

    kfold = lambda s: list(KFold(5, shuffle=True, random_state=s).split(np.arange(n)))
    arms(dur, evt, SEED, kfold(SEED), "random")

    # site-held-out: every test patient's hospital is absent from training
    gfolds = list(GroupKFold(5).split(np.arange(n), groups=sites))
    log("site: images..."); out["img_site"] = ls.run_grouped(bags, G, dur, evt, sites, True, False)
    log("site: genes...");  out["gene_site"] = ls.run_grouped(bags, G, dur, evt, sites, False, True)
    out["clin_site"] = h.oof_cox_risk(Xc, dur, evt, gfolds)
    out.to_csv(CACHE, index=False)

    # negative control: break the patient<->outcome link
    perm = np.random.RandomState(0).permutation(n)
    out["time_neg"], out["event_neg"] = dur[perm], evt[perm]
    arms(dur[perm], evt[perm], SEED, kfold(SEED), "neg")

    for s in EXTRA_SEEDS:
        arms(dur, evt, s, kfold(s), f"s{s}")
    log(f"done -> {CACHE}")


# ------------------------------------------------------------------ stats helpers
def cidx(r, d, ev):
    return concordance_index_censored(ev, d, r)[0]


def ci(r, d, ev, n=1000):
    rng = np.random.RandomState(SEED); bs = []
    for _ in range(n):
        s = rng.choice(len(d), len(d), replace=True)
        if ev[s].sum() >= 3:
            bs.append(cidx(r[s], d[s], ev[s]))
    return cidx(r, d, ev), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def delta(a, b, d, ev, n=2000):
    """Paired bootstrap of C(a) - C(b). Returns mean, lo, hi, P(a not better than b)."""
    rng = np.random.RandomState(SEED); ds = []
    for _ in range(n):
        s = rng.choice(len(d), len(d), replace=True)
        if ev[s].sum() >= 3:
            ds.append(cidx(a[s], d[s], ev[s]) - cidx(b[s], d[s], ev[s]))
    ds = np.array(ds)
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float((ds <= 0).mean())


def models(df, tag):
    """All arms + late-fusion combos for one set of OOF risks."""
    i, g, c = df[f"img_{tag}"].to_numpy(), df[f"gene_{tag}"].to_numpy(), df[f"clin_{tag}"].to_numpy()
    return {
        "Clinical (age + TNM)": c,
        "Genes (PAM50)": g,
        "Images (attention-MIL)": i,
        "Genes + images": z(g) + z(i),
        "Images + clinical": z(i) + z(c),
        "Genes + images + clinical": z(g) + z(i) + z(c),
    }


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(REF)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ------------------------------------------------------------------ step 2: figures
def plot():
    warnings.simplefilter("ignore")
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT,
                         "axes.titlecolor": TEXT, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(CACHE)
    d, ev = df["time"].to_numpy(float), df["event"].to_numpy(bool)
    rand, site = models(df, "random"), models(df, "site")
    full, clin = rand["Genes + images + clinical"], rand["Clinical (age + TNM)"]
    rows = []

    # ---- metrics table
    for name in rand:
        for split, mm in [("random 5-fold", rand), ("site-held-out", site)]:
            c, lo, hi = ci(mm[name], d, ev)
            rows.append({"model": name, "split": split, "c_index": round(c, 3),
                         "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    met = pd.DataFrame(rows)
    deltas = []
    for name in ["Genes (PAM50)", "Images (attention-MIL)", "Genes + images", "Genes + images + clinical"]:
        m, lo, hi, p = delta(rand[name], clin, d, ev)
        deltas.append({"model": name, "delta_vs_clinical": round(m, 3), "lo": round(lo, 3),
                       "hi": round(hi, 3), "p_not_better": round(p, 3)})
    met.to_csv(os.path.join(config.RESULTS_DIR, "metrics_final.csv"), index=False)
    pd.DataFrame(deltas).to_csv(os.path.join(config.RESULTS_DIR, "delta_vs_clinical.csv"), index=False)
    print(met.to_string(index=False)); print(pd.DataFrame(deltas).to_string(index=False))

    # ---- robustness: negative control + multi-seed
    dn, en = df["time_neg"].to_numpy(float), df["event_neg"].to_numpy(bool)
    neg = models(df, "neg")
    rob = [{"check": f"shuffled labels: {k}", "c_index": round(cidx(v, dn, en), 3)} for k, v in neg.items()]
    seeds = [SEED] + EXTRA_SEEDS
    seed_c = [cidx(models(df, "random" if s == SEED else f"s{s}")["Genes + images + clinical"], d, ev)
              for s in seeds]
    rob += [{"check": f"seed {s}: full model", "c_index": round(c, 3)} for s, c in zip(seeds, seed_c)]
    rob.append({"check": "full model across 5 seeds (mean ± sd)",
                "c_index": f"{np.mean(seed_c):.3f} ± {np.std(seed_c):.3f}"})
    pd.DataFrame(rob).to_csv(os.path.join(config.RESULTS_DIR, "robustness.csv"), index=False)
    print(pd.DataFrame(rob).to_string(index=False))

    # ---- Fig 2: forest plot
    names = list(rand)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    style(ax); ax.grid(axis="y", visible=False)
    y = np.arange(len(names))[::-1].astype(float)
    ax.set_ylim(y[-1] - 0.85, y[0] + 0.85)
    ax.axvline(0.5, color=REF, lw=1, ls=(0, (3, 3)))
    ax.text(0.502, y[-1] - 0.75, "chance", color=TEXT2, fontsize=8, ha="left", va="bottom")
    ax.axhline((y[2] + y[3]) / 2, color=GRID, lw=1)
    for mm, color, mk, off, lab in [(rand, BLUE, "o", 0.15, "Random 5-fold split"),
                                    (site, ORANGE, "s", -0.15, "Site-held-out (unseen hospitals)")]:
        for yi, nm in zip(y, names):
            m = met[(met.model == nm) & (met.split == ("random 5-fold" if mm is rand else "site-held-out"))].iloc[0]
            ax.plot([m.ci_lo, m.ci_hi], [yi + off] * 2, color=color, lw=2, solid_capstyle="round")
            ax.plot(m.c_index, yi + off, mk, color=color, ms=8, mec=SURFACE, mew=2,
                    label=lab if nm == names[0] else None)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0.45, 0.75)
    ax.set_xlabel("Concordance index (C-index) with 95% bootstrap CI")
    # value columns sit OUTSIDE the plot area (axes x > 1) so they never touch marks or each other
    for x, head, split in [(1.03, "Random split", "random 5-fold"), (1.30, "Site-held-out", "site-held-out")]:
        ax.text(x, y[0] + 0.62, head, transform=ax.get_yaxis_transform(), color=TEXT2,
                fontsize=8, ha="left", fontweight="bold")
        for yi, nm in zip(y, names):
            m = met[(met.model == nm) & (met.split == split)].iloc[0]
            ax.text(x, yi, f"{m.c_index:.3f} [{m.ci_lo:.2f}–{m.ci_hi:.2f}]",
                    transform=ax.get_yaxis_transform(), color=TEXT2, fontsize=8, ha="left", va="center")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("Survival discrimination by modality (TCGA-BRCA, n=244, 123 deaths)",
                 loc="left", fontsize=11, pad=12)
    fig.savefig(os.path.join(FIG, "fig2_cindex_forest.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: Kaplan-Meier risk groups
    from lifelines.plotting import add_at_risk_counts
    xmax = float(np.ceil(np.percentile(d, 95) / 24) * 24)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.subplots_adjust(wspace=0.32)            # room for the right panel's at-risk row labels
    km_rows = []
    for ax, (risk, title) in zip(axes, [(clin, "Clinical model (age + TNM)"),
                                        (full, "Multi-modal (genes + images + clinical)")]):
        style(ax)
        hi_risk = risk >= np.median(risk)
        fitters = []
        for mask, lab, color in [(~hi_risk, "Low risk", BLUE), (hi_risk, "High risk", ORANGE)]:
            kmf = KaplanMeierFitter(label=lab)
            kmf.fit(d[mask], ev[mask])
            kmf.plot_survival_function(ax=ax, color=color, lw=2, ci_alpha=0.12, show_censors=True,
                                       censor_styles={"ms": 5, "marker": "|"})
            fitters.append(kmf)
        lr = logrank_test(d[hi_risk], d[~hi_risk], ev[hi_risk], ev[~hi_risk])
        cph = CoxPHFitter().fit(pd.DataFrame({"T": d, "E": ev, "high": hi_risk.astype(int)}), "T", "E")
        hr = float(cph.hazard_ratios_["high"])
        hlo, hhi = np.exp(cph.confidence_intervals_.loc["high"].to_numpy())
        km_rows.append({"model": title, "hazard_ratio": round(hr, 2), "hr_lo": round(hlo, 2),
                        "hr_hi": round(hhi, 2), "logrank_p": lr.p_value})
        ax.text(0.97, 0.95, f"HR {hr:.2f} [{hlo:.2f}–{hhi:.2f}]\nlog-rank {fmt_p(lr.p_value)}",
                transform=ax.transAxes, ha="right", va="top", color=TEXT2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02)
        ax.set_xticks(np.arange(0, xmax + 1, 24))   # axis ticks = at-risk columns, so they line up
        ax.set_xlabel("Months since diagnosis")
        ax.legend(loc="lower left", frameon=False, fontsize=8)
        try:
            add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"], xticks=np.arange(0, xmax + 1, 24))
        except Exception:
            pass
    axes[0].set_ylabel("Overall survival probability")
    fig.suptitle("Median split on out-of-fold predicted risk", x=0.01, ha="left", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig3_km_risk_groups.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(km_rows).to_csv(os.path.join(config.RESULTS_DIR, "km_risk_groups.csv"), index=False)
    print(pd.DataFrame(km_rows).to_string(index=False))

    # ---- Fig 4: time-dependent AUC
    sv = Surv.from_arrays(event=ev, time=d)
    times = np.arange(12, min(96, np.percentile(d, 90)) + 1, 3.0)
    times = times[(times > d[ev].min()) & (times < d.max())]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    style(ax)
    ax.axhline(0.5, color=REF, lw=1, ls=(0, (3, 3)))
    auc_rows = []
    for risk, lab, color in [(clin, "Clinical (age + TNM)", BLUE), (full, "Genes + images + clinical", ORANGE)]:
        auc, mean_auc = cumulative_dynamic_auc(sv, sv, risk, times)
        ax.plot(times / 12, auc, color=color, lw=2, label=f"{lab}  (mean AUC {mean_auc:.2f})")
        for yr in (1, 3, 5):
            k = int(np.argmin(np.abs(times - 12 * yr)))
            auc_rows.append({"model": lab, "year": yr, "auc": round(float(auc[k]), 3)})
        auc_rows.append({"model": lab, "year": "mean", "auc": round(float(mean_auc), 3)})
    ax.set_xlabel("Years since diagnosis"); ax.set_ylabel("Time-dependent AUC")
    ax.set_ylim(0.4, 0.9)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Discrimination over time (cumulative/dynamic AUC)", loc="left", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig4_time_auc.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(auc_rows).to_csv(os.path.join(config.RESULTS_DIR, "time_auc.csv"), index=False)
    print(pd.DataFrame(auc_rows).to_string(index=False))

    # ---- Fig 5: complementarity
    gi, ii = z(rand["Genes (PAM50)"]), z(rand["Images (attention-MIL)"])
    r = float(np.corrcoef(gi, ii)[0, 1])
    low_gene = gi < np.median(gi)
    c_rescue = cidx(ii[low_gene], d[low_gene], ev[low_gene])
    fig, ax = plt.subplots(figsize=(6, 5.2))
    style(ax)
    ax.axhline(0, color=REF, lw=0.8); ax.axvline(0, color=REF, lw=0.8)
    ax.scatter(gi[~ev], ii[~ev], s=34, color=BLUE, alpha=0.8, edgecolor=SURFACE, lw=1, label="Alive / censored")
    ax.scatter(gi[ev], ii[ev], s=34, color=ORANGE, alpha=0.9, edgecolor=SURFACE, lw=1, label="Died")
    ax.set_xlabel("Gene risk score (z)"); ax.set_ylabel("Image risk score (z)")
    ax.set_title(f"Image and gene risk are largely independent (r = {r:.2f})", loc="left", fontsize=11, pad=22)
    ax.text(0, 1.015, f"Among patients the gene model rates low-risk, image risk still ranks deaths at C = {c_rescue:.3f}",
            transform=ax.transAxes, color=TEXT2, fontsize=8, va="bottom")   # subtitle, clear of the data
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG, "fig5_complementarity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"complementarity: r = {r:.3f}; image C within gene-low-risk half = {c_rescue:.3f}")

    table1(df)


def table1(df):
    """Cohort characteristics, overall and by outcome."""
    from lifelines import KaplanMeierFitter
    c = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId").loc[df.patient]
    ev = df["event"].to_numpy(bool)

    def stage_grp(s):
        s = str(s).upper()
        for k in ("IV", "III", "II", "I"):
            if f"STAGE {k}" in s:
                return f"Stage {k}"
        return "Unknown"

    groups = {"All": np.ones(len(df), bool), "Died": ev, "Alive / censored": ~ev}
    stg = c["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(stage_grp).to_numpy()
    sub = c["SUBTYPE"].fillna("Unknown").str.replace("BRCA_", "", regex=False).to_numpy()
    age = pd.to_numeric(c["AGE"], errors="coerce").to_numpy()
    lines = ["| Characteristic | " + " | ".join(f"{k} (n={m.sum()})" for k, m in groups.items()) + " |",
             "|---|" + "---|" * len(groups)]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(m) for m in groups.values()) + " |")

    row("Age, median [IQR]", lambda m: f"{np.nanmedian(age[m]):.0f} [{np.nanpercentile(age[m], 25):.0f}–{np.nanpercentile(age[m], 75):.0f}]")
    kmf = KaplanMeierFitter().fit(df["time"], 1 - ev)          # reverse KM = median follow-up
    lines.append(f"| Median follow-up (reverse KM), months | {kmf.median_survival_time_:.1f} | | |")
    row("Deaths, n (%)", lambda m: f"{ev[m].sum()} ({100 * ev[m].mean():.0f}%)")
    for k in ("Stage I", "Stage II", "Stage III", "Stage IV", "Unknown"):
        row(f"{k}, n (%)", lambda m, k=k: f"{(stg[m] == k).sum()} ({100 * (stg[m] == k).mean():.0f}%)")
    for k in ("LumA", "LumB", "Basal", "Her2", "Normal", "Unknown"):
        row(f"PAM50 {k}, n (%)", lambda m, k=k: f"{(sub[m] == k).sum()} ({100 * (sub[m] == k).mean():.0f}%)")
    row("Tissue-source sites", lambda m: f"{df.site[m].nunique()}")
    txt = "# Table 1 — Cohort characteristics (TCGA-BRCA, size-matched imaging cohort)\n\n" + "\n".join(lines) + "\n"
    with open(os.path.join(config.RESULTS_DIR, "table1_cohort.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    if "--recompute" in sys.argv or not os.path.exists(CACHE):
        compute()
    plot()
