"""
make_paper_figures.py — paper Figure 1 (pipeline) and supplement figures / tables.

  results/figures/fig1_pipeline.png            how a patient becomes a risk score, and how it was tested
  results/figures/figS1_cohort_flow.png        how 1,082 TCGA-BRCA patients became the 1,033 analysed
  results/figures/figS2_settings_table.png     every model setting (also results/supplement/table_s1_settings.md)
  results/figures/figS3_sensitivity_table.png  every sensitivity / robustness result (also table_s2_sensitivity.md)

All counts are computed from the data and result files (nothing typed in by hand except method settings,
which are copied from the code they describe).

Run:  python src/make_paper_figures.py
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                       # noqa: E402
import make_figures as mf           # noqa: E402
import stage_e_attmil as e          # noqa: E402

FIG, SUPP = config.FIG_DIR, os.path.join(config.RESULTS_DIR, "supplement")
FULL = os.path.join(config.RESULTS_DIR, "fullcohort")
PANEL = "#f4f3f0"
plt.rcParams.update({"font.size": 10, "text.color": mf.TEXT, "figure.facecolor": mf.SURFACE})


def box(ax, x, y, w, h, title, body="", accent=None, title_size=10.5, body_size=8.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012",
                                fc=PANEL, ec=accent or mf.GRID, lw=1.6 if accent else 1.0))
    ax.text(x + 0.012, y + h - 0.018, title, fontsize=title_size, fontweight="bold", va="top", color=mf.TEXT)
    if body:
        ax.text(x + 0.012, y + h - 0.018 - 0.045 * title_size / 10.5, body, fontsize=body_size, va="top",
                color=mf.TEXT2, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color or mf.REF, lw=1.4, shrinkA=2, shrinkB=2))


# ------------------------------------------------------------------ Figure 1
def fig1():
    oof = pd.read_csv(os.path.join(FULL, "oof_risks_full.csv"))
    n, deaths, sites = len(oof), int(oof.event.sum()), oof.site.nunique()
    fig = plt.figure(figsize=(14, 7.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.015, 0.975, "Figure 1 · From a patient's data to a survival risk score", fontsize=14, fontweight="bold", va="top")
    ax.text(0.015, 0.935, f"TCGA-BRCA, n = {n:,} patients ({deaths} deaths), {sites} hospitals. "
            "Every prediction is made by a model that never saw that patient.", fontsize=9.5, color=mf.TEXT2, va="top")

    # row 1: tissue slide -> tiles -> Phikon -> attention-MIL
    y1 = 0.60
    hm = os.path.join(config.DATA_DIR, "stage_k_heatmaps", "TCGA-LL-A73Z")
    if os.path.exists(os.path.join(hm, "thumb.png")):
        a = fig.add_axes([0.015, y1 + 0.005, 0.13, 0.20]); a.imshow(plt.imread(os.path.join(hm, "thumb.png"))); a.axis("off")
        for k, xx in enumerate([0.165, 0.215, 0.265]):
            t = fig.add_axes([xx, y1 + 0.05, 0.045, 0.09]); t.imshow(plt.imread(os.path.join(hm, f"top{k + 1}.png")))
            t.set_xticks([]); t.set_yticks([])
            for sp in t.spines.values():
                sp.set_color(mf.REF)
    ax.text(0.015, y1 + 0.225, "TISSUE SLIDE", fontsize=9, fontweight="bold", color=mf.BLUE)
    ax.text(0.015, y1 - 0.012, "H&E diagnostic slide", fontsize=8.5, color=mf.TEXT2, va="top")
    ax.text(0.165, y1 - 0.012, "≤ 200 tiles, 224 px at ~20×", fontsize=8.5, color=mf.TEXT2, va="top")
    arrow(ax, 0.148, y1 + 0.095, 0.162, y1 + 0.095, mf.BLUE)
    box(ax, 0.335, y1, 0.16, 0.19, "Phikon", "pathology foundation\nmodel (pretrained,\nno outcomes used)\n→ 768 numbers per tile", mf.BLUE)
    arrow(ax, 0.313, y1 + 0.095, 0.333, y1 + 0.095, mf.BLUE)
    box(ax, 0.515, y1, 0.17, 0.19, "Attention-MIL", "learns how much each\ntile matters, combines\nthem, Cox survival loss\n→ image risk", mf.BLUE)
    arrow(ax, 0.497, y1 + 0.095, 0.513, y1 + 0.095, mf.BLUE)

    # row 2: genes, row 3: clinical
    y2, y3 = 0.33, 0.12
    ax.text(0.015, y2 + 0.155, "GENES", fontsize=9, fontweight="bold", color=mf.TEXT2)
    box(ax, 0.015, y2, 0.30, 0.135, "PAM50 gene activity", "50 genes (tumor RNA-seq), standardized")
    box(ax, 0.515, y2, 0.17, 0.135, "Gene model", "linear, Cox survival loss\n→ gene risk")
    arrow(ax, 0.317, y2 + 0.067, 0.513, y2 + 0.067)
    ax.text(0.015, y3 + 0.155, "CLINICAL", fontsize=9, fontweight="bold", color=mf.TEXT2)
    box(ax, 0.015, y3, 0.30, 0.135, "Age + tumor stage", "age, AJCC stage (I–IV), T (size), N (lymph nodes)")
    box(ax, 0.515, y3, 0.17, 0.135, "Cox model", "standard survival model\n→ clinical risk")
    arrow(ax, 0.317, y3 + 0.067, 0.513, y3 + 0.067)

    # combine + evaluate
    box(ax, 0.735, 0.43, 0.235, 0.20, "Combine (late fusion)",
        "put each risk on the same scale\n(z-score), then add them:\n· equal weights, or\n· weights learned with nested\n  cross-validation", mf.ORANGE)
    for yy in (y1 + 0.095, y2 + 0.067, y3 + 0.067):
        arrow(ax, 0.687, yy, 0.733, 0.53)
    box(ax, 0.735, 0.10, 0.235, 0.27, "How it was tested",
        "· 5-fold cross-validation (out-of-fold)\n· leave-one-hospital-out (39 sites)\n· scrambled-outcome controls\n"
        "· C-index, time-AUC, Kaplan-Meier\n· calibration, Brier, decision curves\n· tests written down in advance\n  (pre-registration)")
    arrow(ax, 0.852, 0.428, 0.852, 0.372, mf.ORANGE)
    ax.text(0.015, 0.03, "C-index = how often the model ranks the patient who died sooner as higher-risk (0.5 = chance, 1.0 = perfect). "
            "Example slide and tiles: TCGA-LL-A73Z.", fontsize=8, color=mf.TEXT2)
    fig.savefig(os.path.join(FIG, "fig1_pipeline.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure S1 cohort flow
def figS1():
    c = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId")
    gdc = pd.read_csv(os.path.join(FULL, "gdc_dx_slides.csv"))
    surv, genes = e.load_survival(), e.load_genes()
    gene_ok = set(genes.index[genes.notna().all(axis=1)])
    s_ok = set(surv.index)
    sg = s_ok & gene_ok
    final = sg & set(gdc.patient)
    oof = pd.read_csv(os.path.join(FULL, "oof_risks_full.csv"))
    new = set(pd.read_csv(os.path.join(FULL, "frozen_preds.csv")).patient)
    assert len(final) == len(oof)
    ev = oof.set_index("patient").event
    n_orig, n_new = len(final - new), len(final & new)
    d_orig, d_new = int(ev[list(final - new)].sum()), int(ev[list(final & new)].sum())
    multi = int((gdc[gdc.patient.isin(final)].groupby("patient").size() >= 2).sum())
    tiles = int(np.load(os.path.join(config.DATA_DIR, "stage_n_tissue", "tile_clusters.npz"))["owner"].size) \
        if os.path.exists(os.path.join(config.DATA_DIR, "stage_n_tissue", "tile_clusters.npz")) else None

    fig = plt.figure(figsize=(10, 9.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.03, 0.975, "Figure S1 · How the analysed cohort was formed", fontsize=13, fontweight="bold", va="top")
    W, X = 0.46, 0.08
    steps = [
        (0.80, f"{len(c):,} TCGA-BRCA patients", "clinical + PAM50 table (cBioPortal)"),
        (0.62, f"{len(sg):,} patients", "usable survival time and complete PAM50 data"),
        (0.44, f"{len(final):,} patients analysed  ·  {int(oof.event.sum())} deaths  ·  {oof.site.nunique()} hospitals",
         f"one diagnostic slide each (smallest if several; {multi} had 2+)"),
    ]
    for y, t, b in steps:
        box(ax, X, y, W, 0.115, t, b, mf.BLUE if "analysed" in t else None, title_size=10.5, body_size=9)
    arrow(ax, X + W / 2, 0.80, X + W / 2, 0.735); arrow(ax, X + W / 2, 0.62, X + W / 2, 0.555)
    ex = [(0.7675, f"{len(c) - len(s_ok):,} excluded", "no usable survival time"),
          (0.5875, f"{len(sg - final):,} excluded", f"no diagnostic slide on GDC\n(GDC: {len(gdc):,} slides, {gdc.patient.nunique():,} patients)")]
    for ymid, t, b in ex:                                  # arrow leaves the connector between two boxes
        box(ax, 0.62, ymid - 0.05, 0.33, 0.1, t, b, title_size=9.5, body_size=8.5)
        arrow(ax, X + W / 2, ymid, 0.618, ymid)
    box(ax, 0.03, 0.21, 0.42, 0.13, f"Original cohort  ·  n = {n_orig}  ·  {d_orig} deaths",
        "size-matched case/control sample (Jul):\ndeaths matched to survivors on slide\nfile size; 50% deaths by design", title_size=10, body_size=8.8)
    box(ax, 0.53, 0.21, 0.42, 0.13, f"Full-cohort additions  ·  n = {n_new}  ·  {d_new} deaths",
        "every remaining eligible patient (Sep 24–30);\nslides MD5-verified, 0 failures;\nslides deleted after feature extraction", title_size=10, body_size=8.8)
    arrow(ax, X + W / 2 - 0.05, 0.44, 0.24, 0.342); arrow(ax, X + W / 2 + 0.05, 0.44, 0.74, 0.342)
    ax.text(0.03, 0.155, (f"Image features: {tiles:,} tissue tiles in total (≤ 200 per patient; a few slides had fewer tissue tiles)."
                         if tiles else ""), fontsize=9, color=mf.TEXT2)
    ax.text(0.03, 0.125, "Survival = overall survival (death from any cause); patients alive at last follow-up are censored.",
            fontsize=9, color=mf.TEXT2)
    fig.savefig(os.path.join(FIG, "figS1_cohort_flow.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"patients_table": len(c), "with_survival_genes": len(sg), "analysed": len(final), "orig": n_orig, "new": n_new}


# ------------------------------------------------------------------ tables
SETTINGS = [
    ("Slides", "Diagnostic H&E slide per patient", "smallest if several (full run); GDC download, MD5-verified"),
    ("Tiles", "Size / magnification", "224 × 224 px at downsample 2 (~20× on 40× scans)"),
    ("Tiles", "Tissue filter", "HSV saturation mean > 25 and value mean < 235"),
    ("Tiles", "Number per patient", "up to 200, taken in a fixed random order (seed 0)"),
    ("Features", "Model", "Phikon (owkin/phikon, ViT), CLS token, 768 numbers per tile"),
    ("Image model", "Architecture", "gated attention-MIL, hidden 128, dropout 0.4, linear Cox head"),
    ("Image model", "Training", "Adam, learning rate 1e-3, weight decay 1e-3, 120 epochs, full batch, Cox loss"),
    ("Gene model", "Inputs / model", "50 PAM50 genes, standardized; linear Cox head (same trainer)"),
    ("Clinical model", "Inputs / model", "age, AJCC stage, T, N (median-imputed); Cox with ridge alpha 1.0"),
    ("Fusion", "Late fusion", "sum of z-scored risks (equal weights); stacking = Cox on z-scored risks, nested CV"),
    ("Validation", "Splits", "5-fold KFold (shuffle, seed 42); GroupKFold by hospital (5 folds)"),
    ("Statistics", "Uncertainty", "bootstrap 95% CI (1,000); paired bootstrap ΔC (2,000); seed 42"),
    ("Statistics", "Calibration", "cross-fitted one-variable Cox recalibration; quintiles; Brier (IPCW) at 3 and 5 yr"),
    ("Software", "Key versions", "Python 3.9.6, torch 2.8.0, scikit-survival 0.23.1, lifelines 0.30.0"),
]


def render_table(rows, cols, path, title, widths):
    """Draw a table with WRAPPED cell text (matplotlib's own table clips long text).
    widths = column widths in inches; each row is as tall as its longest wrapped cell."""
    import textwrap
    fs, line_h, pad = 8.8, 0.19, 0.07                      # font size (pt), line height and cell padding (in)
    cpi = 72 / (fs * 0.55)                                 # ≈ characters per inch at this font size
    wrapped = [[textwrap.wrap(str(c), max(8, int((w - 2 * pad) * cpi))) or [""] for c, w in zip(r, widths)]
               for r in [cols] + [list(r) for r in rows]]
    heights = [max(len(c) for c in r) * line_h + 2 * pad for r in wrapped]
    W, H = sum(widths), sum(heights)
    fig = plt.figure(figsize=(W + 0.3, H + 0.75))
    ax = fig.add_axes([0.15 / (W + 0.3), 0.05 / (H + 0.75), W / (W + 0.3), H / (H + 0.75)])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    y = 0.0
    for i, (r, h) in enumerate(zip(wrapped, heights)):
        if i == 0:
            ax.add_patch(plt.Rectangle((0, 0), W, h, color=PANEL, zorder=0))
        x = 0.0
        for lines, w in zip(r, widths):
            ax.text(x + pad, y + pad, "\n".join(lines), fontsize=fs, va="top", ha="left",
                    fontweight="bold" if i == 0 else "normal", color=mf.TEXT, linespacing=1.3)
            x += w
        y += h
        ax.axhline(y, color=mf.GRID, lw=0.8)
    ax.axhline(0, color=mf.GRID, lw=0.8)
    fig.suptitle(title, x=0.15 / (W + 0.3), ha="left", y=1 - 0.12 / (H + 0.75), fontsize=11.5, fontweight="bold")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def md_table(rows, cols):
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"


def tableS1():
    cols = ["Part", "Setting", "Value"]
    render_table([list(r) for r in SETTINGS], cols, os.path.join(FIG, "figS2_settings_table.png"),
                 "Table S1 · Model and analysis settings", [1.4, 2.0, 6.4])
    with open(os.path.join(SUPP, "table_s1_settings.md"), "w") as f:
        f.write("# Table S1 — Model and analysis settings\n\n" + md_table(SETTINGS, cols))


def tableS2():
    """Every sensitivity / robustness result, read from the result files."""
    R = lambda *p: pd.read_csv(os.path.join(FULL, *p))
    rows = []
    a1 = R("added_value", "a1_lrt.csv")
    rows.append(["Image risk beyond age + stage + T + N (A1, primary)", f"HR {a1.HR_per_SD[0]:.2f} [{a1.HR_lo[0]:.2f}, {a1.HR_hi[0]:.2f}]", f"p = {a1.p_LRT[0]:.1e}"])
    rows.append(["  … using site-held-out image risk", f"HR {a1.HR_per_SD[2]:.2f} [{a1.HR_lo[2]:.2f}, {a1.HR_hi[2]:.2f}]", f"p = {a1.p_LRT[2]:.3f}"])
    a4 = os.path.join(FULL, "added_value", "a4_lrt_size_adjusted.csv")
    if os.path.exists(a4):
        s = pd.read_csv(a4).iloc[0]
        rows.append(["  … adjusted for slide file size (A4)", f"HR {s.HR_per_SD:.2f}", f"p = {s.p_LRT:.1e}"])
    b3 = R("tissue", "b3_primary_tests.csv")
    for key, lab in [("(k=8)  [PRIMARY]", "  … adjusted for tissue composition, k = 8 (B3)"),
                     ("(k=12)  [sensitivity]", "  … adjusted for tissue composition, k = 12 (B3)")]:
        r_ = b3[b3.test.str.contains(key, regex=False) & b3.test.str.startswith("+ image")].iloc[0]   # select by name, not position
        rows.append([lab, f"HR {r_.HR_image_per_SD:.2f} [{r_.HR_lo:.2f}, {r_.HR_hi:.2f}]", f"p = {r_.p_LRT:.4f}"])
    fu = os.path.join(FULL, "followups")
    if os.path.exists(os.path.join(fu, "c2c_without_flagged.csv")):
        c2c = pd.read_csv(os.path.join(fu, "c2c_without_flagged.csv")).iloc[0]
        rows.append(["  … without the blank-slide patient (C2c)", f"HR {c2c.HR_per_SD:.2f} [{c2c.HR_lo:.2f}, {c2c.HR_hi:.2f}]", f"p = {c2c.p_LRT:.1e}"])
    if os.path.exists(os.path.join(fu, "c2a_fewer_tiles.csv")):
        c2a = pd.read_csv(os.path.join(fu, "c2a_fewer_tiles.csv")).iloc[1]
        rows.append(["  … image model on 100 tiles per patient (C2a)", f"HR {c2a.HR_per_SD:.2f} [{c2a.HR_lo:.2f}, {c2a.HR_hi:.2f}]", f"p = {c2a.p_LRT:.1e}"])
    if os.path.exists(os.path.join(fu, "c1_filtered_result.csv")):
        c1 = pd.read_csv(os.path.join(fu, "c1_filtered_result.csv"))
        if "HR_per_SD" in c1:
            r0 = c1.iloc[0]
            rows.append(["  … image model without non-tissue tiles (C1, cluster 2b)", f"HR {r0.HR_per_SD:.2f} [{r0.HR_lo:.2f}, {r0.HR_hi:.2f}]", f"p = {r0.p_LRT:.1e}"])
        else:
            rows.append(["  … fat-free image model (C1)", "not run", "no sub-cluster met the fat rule"])
    m = R("metrics_final.csv").set_index(["model", "split"])
    rows.append(["Image C-index, random split / unseen hospitals", f"{m.loc[('Images (attention-MIL)', 'random 5-fold'), 'c_index']:.3f} / "
                 f"{m.loc[('Images (attention-MIL)', 'site-held-out'), 'c_index']:.3f}", ""])
    if os.path.exists(os.path.join(fu, "c2a_fewer_tiles.csv")):
        rows.append(["Image C-index, 100 tiles per patient (C2a)", f"{c2a.c_index:.3f} [{c2a.ci_lo:.2f}, {c2a.ci_hi:.2f}]", ""])
    st = R("added_value", "a2_stacking.csv").set_index("model")
    rows.append(["Full model, learned weights (A2): ΔC vs clinical", f"{st.loc['Stacked: genes + images + clinical', 'delta_vs_clinical']:+.3f} "
                 f"[{st.loc['Stacked: genes + images + clinical', 'delta_lo']:+.3f}, {st.loc['Stacked: genes + images + clinical', 'delta_hi']:+.3f}]",
                 f"p = {st.loc['Stacked: genes + images + clinical', 'p_two_sided']:.2f}"])
    if os.path.exists(os.path.join(fu, "c2b_weights.csv")):
        for _, r in pd.read_csv(os.path.join(fu, "c2b_weights.csv")).iterrows():
            rows.append([f"Fixed weights {r.model.split('= ')[1]} (C2b): ΔC vs clinical",
                         f"{r.delta_vs_clinical:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]", f"p = {r.p_two_sided:.2f}"])
    neg = R("negative_control_repeats.csv")
    rows.append(["Scrambled outcomes, 10 shuffles (genes / clinical)", f"{neg.genes.mean():.3f} / {neg.clinical.mean():.3f}", "≈ 0.5 expected"])
    rob = R("robustness.csv")
    rows.append(["Full model over 5 seeds", rob.c_index.iloc[-1], ""])
    cols = ["Analysis", "Estimate [95% CI]", "Test"]
    render_table(rows, cols, os.path.join(FIG, "figS3_sensitivity_table.png"),
                 "Table S2 · Sensitivity and robustness analyses (n = 1,033)", [5.6, 2.8, 1.8])
    with open(os.path.join(SUPP, "table_s2_sensitivity.md"), "w") as f:
        f.write("# Table S2 — Sensitivity and robustness analyses (n = 1,033)\n\nHR = hazard ratio of the image risk per "
                "1 SD in a Cox model with age, stage, T, N (+ the listed adjustment). ΔC = C-index difference vs "
                "clinical alone (paired bootstrap).\n\n" + md_table(rows, cols))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    os.makedirs(SUPP, exist_ok=True)
    fig1()
    print(figS1())
    tableS1()
    tableS2()
    print("done -> results/figures/fig1_pipeline.png, figS1–S3; results/supplement/")
