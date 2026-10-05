"""
stage_k_heatmaps.py — attention heatmaps: WHERE on the slide does the image model look?

For a few example patients, shows which tissue tiles the attention-MIL model weighted most when
scoring survival risk. Rigor: each patient's attention comes from its OUT-OF-FOLD model — the fold
model that never trained on that patient — with the same folds and seed as the reported results.

Steps:
  1. Retrain the 5 image-arm fold models (identical to stage_e_attmil.run, seed 42) and KEEP them;
     check their out-of-fold risks equal the cached ones in results/oof_risks.csv.
  2. Pick examples from smaller slides (quick downloads): patients rated high-risk who died,
     a low-risk long survivor, and one MISSED case (rated low-risk, died) — not only successes.
  3. Re-download each slide, recompute tile features + positions with the exact Stage C pipeline
     (checking they match the saved features), save a thumbnail and the top-attention tiles,
     then delete the slide.
  4. Draw results/figures/fig6_attention_heatmaps.png.
Per-patient outputs are cached in data/stage_k_heatmaps/<patient>/ (gitignored).

Run:  python src/stage_k_heatmaps.py              (compute if needed, then plot)
      python src/stage_k_heatmaps.py --recompute
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                                   # noqa: E402
import stage_c_local as scl                     # noqa: E402  size-matched slide list (file ids)
import stage_e_attmil as e                      # noqa: E402  ABMIL model + Cox loss
from make_figures import (ORANGE, REF, SURFACE, TEXT, TEXT2,   # noqa: E402
                          load_cohort)

SEED = config.RANDOM_SEED
OUT = os.path.join(config.DATA_DIR, "stage_k_heatmaps")
MAX_GB = 0.7                    # only consider slides up to this size for examples
# single-hue sequential ramp (light -> palette orange -> dark) for attention
ATT_CMAP = LinearSegmentedColormap.from_list("att", ["#fde3d6", ORANGE, "#8f3412"])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_fold_models(bags, dur, evt, epochs=120, lr=1e-3):
    """Same training as stage_e_attmil.run(use_img=True), but keeps each fold's model."""
    n = len(bags)
    oof, fold_of, models = np.full(n, np.nan), np.zeros(n, int), []
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(np.arange(n))):
        torch.manual_seed(SEED)
        model = e.ABMIL(gene_dim=0)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        tt, tev = torch.tensor(dur[tr]), torch.tensor(evt[tr].astype(float))
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            risks = torch.stack([model(bags[i], None) for i in tr])
            e.cox_loss(risks, tt, tev).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                oof[i] = float(model(bags[i], None))
                fold_of[i] = k
        models.append(model)
    return models, fold_of, oof


def attention(model, H):
    """The model's softmax attention weight for every tile (sums to 1 over the slide)."""
    with torch.no_grad():
        a = model.w(torch.tanh(model.V(H)) * torch.sigmoid(model.U(H)))
        return torch.softmax(a, dim=0).squeeze(1).numpy()


def pick_examples(pids, dur, evt, risk, sizes):
    df = pd.DataFrame({"patient": pids, "time": dur, "event": evt, "risk": risk})
    df["pct"] = df.risk.rank(pct=True)                       # risk percentile within the cohort
    df = df[df.patient.map(sizes).fillna(9e12) / 1e9 <= MAX_GB]
    hi = df[df.event & (df.pct >= 0.75)].sort_values("time").head(2)
    lo = df[~df.event & (df.pct <= 0.25)].sort_values("time", ascending=False).head(1)
    miss = df[df.event & (df.pct <= 0.25)].sort_values("time").head(1)
    return pd.concat([hi.assign(kind="Rated high risk · died"),
                      lo.assign(kind="Rated low risk · survived"),
                      miss.assign(kind="MISSED · rated low risk, died")]).reset_index(drop=True)


def download(file_id, dest):
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for ch in r.iter_content(1 << 20):
                        fh.write(ch)
            return True
        except Exception as ex:
            log(f"  download retry {attempt + 1}/4 {file_id[:8]}: {repr(ex)[:40]}")
    return False


def extract(patient, path, model, saved):
    """Recompute tiles with the Stage C pipeline, score attention, save thumbnail + top tiles."""
    import openslide
    import stage_c_pathology as P        # loads Phikon (same model/settings as Stage C)
    feats, xy = P.tile_features(path)
    feats = feats.astype(np.float32)
    match = None
    if saved.shape == feats.shape:        # same slide + pipeline -> near-identical features
        cos = (saved * feats).sum(1) / (np.linalg.norm(saved, axis=1) * np.linalg.norm(feats, axis=1))
        match = float(cos.mean())
    att = attention(model, torch.tensor(feats))
    top = np.argsort(att)[::-1][:3]
    s = openslide.OpenSlide(path)
    W0, H0 = s.dimensions
    thumb = s.get_thumbnail((1200, 1200))
    lvl = s.get_best_level_for_downsample(P.DOWNSAMPLE)
    crops = [s.read_region((int(xy[j, 0]), int(xy[j, 1])), lvl, (P.TILE, P.TILE)).convert("RGB") for j in top]
    s.close()
    d = os.path.join(OUT, patient)
    os.makedirs(d, exist_ok=True)
    thumb.save(os.path.join(d, "thumb.png"))
    for k, c in enumerate(crops):
        c.save(os.path.join(d, f"top{k + 1}.png"))
    np.save(os.path.join(d, "coords.npy"), xy)
    np.save(os.path.join(d, "attention.npy"), att)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"W0": W0, "H0": H0, "thumb_w": thumb.width, "feature_match": match,
                   "n_tiles": int(len(att)), "top": [int(j) for j in top]}, f)
    log(f"  {patient}: {len(att)} tiles, feature match vs saved = {match}")


def compute():
    warnings.simplefilter("ignore")
    pids, dur, evt, bags, _, _, _ = load_cohort()
    log(f"retraining the 5 image fold models (seed {SEED})...")
    models, fold_of, oof = train_fold_models(bags, dur, evt)
    cached = pd.read_csv(os.path.join(config.RESULTS_DIR, "oof_risks.csv")).set_index("patient")
    same = np.allclose(oof, cached.loc[pids, "img_random"].to_numpy(), atol=1e-4)
    log(f"fold models reproduce the cached out-of-fold risks: {same}")
    if not same:
        log("WARNING: models differ from the reported run — heatmaps would not match the results.")
    sel = scl.slide_list(scl.load_survival()).set_index("patient")
    ex = pick_examples(pids, dur, evt, oof, sel["size"].to_dict())
    log("examples:\n" + ex[["patient", "kind", "time", "pct"]].to_string(index=False))
    os.makedirs(OUT, exist_ok=True)
    paths = {p: os.path.join(OUT, f"{p}.svs") for p in ex.patient}
    log(f"downloading {len(ex)} slides ({sel.loc[ex.patient, 'size'].sum() / 1e9:.1f} GB)...")
    with ThreadPoolExecutor(len(ex)) as pool:
        ok = dict(zip(ex.patient, pool.map(lambda p: download(sel.loc[p, "file_id"], paths[p]), ex.patient)))
    idx = {p: i for i, p in enumerate(pids)}
    for p in ex.patient:
        if ok[p]:
            i = idx[p]
            extract(p, paths[p], models[fold_of[i]], bags[i].numpy())
        if os.path.exists(paths[p]):
            os.remove(paths[p])                      # slides are big; keep only the small outputs
    ex.to_csv(os.path.join(OUT, "examples.csv"), index=False)


def plot():
    ex = pd.read_csv(os.path.join(OUT, "examples.csv"))
    ex = ex[[os.path.exists(os.path.join(OUT, p, "meta.json")) for p in ex.patient]].reset_index(drop=True)
    n = len(ex)
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "figure.facecolor": SURFACE})
    fig = plt.figure(figsize=(12, 3.0 * n + 0.9))
    gs = fig.add_gridspec(n, 4, width_ratios=[2.4, 1, 1, 1], wspace=0.06, hspace=0.45, top=0.95, bottom=0.07)
    for r, row in ex.iterrows():
        d = os.path.join(OUT, row.patient)
        meta = json.load(open(os.path.join(d, "meta.json")))
        xy, att = np.load(os.path.join(d, "coords.npy")), np.load(os.path.join(d, "attention.npy"))
        sc = meta["thumb_w"] / meta["W0"]
        cx, cy = (xy[:, 0] + xy[:, 2] / 2) * sc, (xy[:, 1] + xy[:, 2] / 2) * sc   # tile centers
        w = att / att.max()
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(plt.imread(os.path.join(d, "thumb.png")))
        order = np.argsort(w)                            # draw high attention on top
        cols = ATT_CMAP(w[order]); cols[:, 3] = 0.08 + 0.87 * w[order] ** 1.5   # low attention recedes
        ax.scatter(cx[order], cy[order], s=24, c=cols, edgecolors="none")
        for k, j in enumerate(meta["top"]):
            ax.scatter(cx[j], cy[j], s=110, facecolors="none", edgecolors=TEXT, linewidths=1.5)
            ax.annotate(str(k + 1), (cx[j], cy[j]), xytext=(6, 6), textcoords="offset points",
                        color=TEXT, fontsize=8, fontweight="bold")
        ax.axis("off"); ax.set_anchor("NW")                # align thumbnails + titles left
        outcome = f"died at {row.time:.0f} months" if row.event else f"alive at {row.time:.0f} months"
        ax.set_title(f"{row.kind}\n{row.patient} · risk percentile {100 * row.pct:.0f} · {outcome}",
                     loc="left", fontsize=9)
        for k in range(3):
            axc = fig.add_subplot(gs[r, k + 1])
            axc.imshow(plt.imread(os.path.join(d, f"top{k + 1}.png")))
            axc.set_xticks([]); axc.set_yticks([])
            for sp in axc.spines.values():
                sp.set_color(REF)
            axc.set_title(f"#{k + 1} · {100 * att[meta['top'][k]]:.1f}% of attention", fontsize=8, color=TEXT2)
    cax = fig.add_axes([0.13, 0.035, 0.22, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ATT_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.outline.set_visible(False)
    cb.set_label("attention (relative to slide max)", fontsize=8, color=TEXT2)
    fig.suptitle("Where the image model looks: attention over tissue tiles (out-of-fold models)",
                 x=0.12, ha="left", fontsize=12, y=0.995)
    fig.text(0.40, 0.022, "Dots mark the ≤200 sampled tissue tiles per slide (enlarged for visibility); "
             "right: the 3 highest-attention tiles.\nExamples are illustrative and include a missed case; "
             "attention shows what the model weighted, not what causes outcome.",
             fontsize=7.5, color=TEXT2, va="bottom")
    out = os.path.join(config.FIG_DIR, "fig6_attention_heatmaps.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log(f"saved {out}")


# ---------------------------------------------------------------------------------------------
# "Explained" version for presentations/posters: same data as Fig 6, plus plain-English
# definitions, a verdict per patient, and a short note on what the top tiles show.
# (The journal version above stays clean — journals put definitions in the figure caption.)
# ---------------------------------------------------------------------------------------------
GOOD, CRITICAL = "#0ca30c", "#d03b3b"     # reserved status colors (dataviz palette); always with ✓/✗ + label

# Non-expert visual reads of the top-3 tiles (written by looking at the crops) — NOT pathologist-reviewed.
TILE_NOTES = {
    "TCGA-AC-A23H": "Fibrous (scar-like) tissue with scattered, irregular nuclei; attention is spread over many tiles.",
    "TCGA-LL-A73Z": "#2–#3: crowded, large, dark, irregular nuclei (typical of aggressive tumor). "
                    "#1: mostly empty background at the tissue edge.",
    "TCGA-A2-A0CQ": "#1: fibrous tissue. #2–#3: mostly fat (thin pink walls around empty space), not tumor.",
    "TCGA-AR-A1AR": "All three are mostly fat or empty space: the model seems to have scored background, not the tumor.",
}

KEY_LEFT = [
    ("Whole-slide image (left)", "a scan of the patient's tumor slide, stained so cell nuclei look purple "
                                 "and other tissue pink."),
    ("Tile", "a small 224×224-pixel square of tissue; the model looks at up to 200 per patient."),
    ("Dots", "one per tile. Darker and larger = the tile counted more."),
    ("Circles 1–3", "the three tiles shown enlarged on the right."),
]
KEY_RIGHT = [
    ("Attention", "how much each tile counted toward the patient's risk score (all tiles add up to 100%)."),
    ("Risk percentile", "where the patient's score ranks among all 244 patients (98 = riskier than 98% of them)."),
    ("Out-of-fold", "every patient was scored by a model that never saw them during training."),
    ("Missed case", "the model said low risk, but the patient died."),
]


def _key_text(items, width=72):
    import textwrap
    out = []
    for term, desc in items:
        lines = textwrap.wrap(f"{term}: {desc}", width=width)
        out.append("• " + lines[0] + "".join("\n   " + l for l in lines[1:]))
    return "\n".join(out)


def plot_explained():
    ex = pd.read_csv(os.path.join(OUT, "examples.csv"))
    ex = ex[[os.path.exists(os.path.join(OUT, p, "meta.json")) for p in ex.patient]].reset_index(drop=True)
    n = len(ex)
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "figure.facecolor": SURFACE})
    heights = [1.85] + [0.42, 2.45, 0.30] * n + [0.55]
    fig = plt.figure(figsize=(12, sum(heights) * 1.0))
    gs = fig.add_gridspec(len(heights), 4, height_ratios=heights, width_ratios=[2.4, 1, 1, 1],
                          wspace=0.06, hspace=0.12, top=0.965, bottom=0.01)
    fig.suptitle("Where the AI looks when it predicts survival from a tissue slide",
                 x=0.125, ha="left", fontsize=14, fontweight="bold", y=0.995)

    # --- "How to read this figure" key ---
    kax = fig.add_subplot(gs[0, :]); kax.axis("off")
    kax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=kax.transAxes, facecolor="#f4f3f0",
                                edgecolor="none", zorder=0))
    kax.text(0.012, 0.93, "How to read this figure", fontsize=10, fontweight="bold", va="top",
             transform=kax.transAxes)
    kax.text(0.012, 0.72, _key_text(KEY_LEFT), fontsize=8.3, va="top", linespacing=1.35,
             transform=kax.transAxes)
    kax.text(0.51, 0.72, _key_text(KEY_RIGHT), fontsize=8.3, va="top", linespacing=1.35,
             transform=kax.transAxes)

    for r, row in ex.iterrows():
        d = os.path.join(OUT, row.patient)
        meta = json.load(open(os.path.join(d, "meta.json")))
        xy, att = np.load(os.path.join(d, "coords.npy")), np.load(os.path.join(d, "attention.npy"))
        high = row.pct >= 0.5
        correct = (high and row.event) or (not high and not row.event)
        g0 = 1 + 3 * r

        # header: verdict (status icon + label) and patient line
        hax = fig.add_subplot(gs[g0, :]); hax.axis("off")
        hax.text(0.0, 0.62, "✓" if correct else "✗", color=GOOD if correct else CRITICAL,
                 fontsize=15, fontweight="bold", va="center", transform=hax.transAxes)
        if row.event:
            outcome = f"the patient died at {row.time:.0f} months"
        else:
            outcome = f"the patient was alive at {row.time:.0f} months (last check-up)"
        verdict = (f"{'Correct' if correct else 'Missed'}: rated {'HIGH' if high else 'LOW'} risk, "
                   f"{'and' if correct else 'but'} {outcome}")
        hax.text(0.022, 0.62, verdict, fontsize=10.5, fontweight="bold", va="center", transform=hax.transAxes)
        hax.text(0.022, 0.08, f"Patient {row.patient} · risk percentile {100 * row.pct:.0f} "
                 f"(riskier than {100 * row.pct:.0f}% of the 244 patients)", fontsize=8.5, color=TEXT2,
                 va="center", transform=hax.transAxes)

        # thumbnail with attention dots (color AND size encode attention, so it's not color-alone)
        sc = meta["thumb_w"] / meta["W0"]
        cx, cy = (xy[:, 0] + xy[:, 2] / 2) * sc, (xy[:, 1] + xy[:, 2] / 2) * sc
        w = att / att.max()
        ax = fig.add_subplot(gs[g0 + 1, 0])
        ax.imshow(plt.imread(os.path.join(d, "thumb.png")))
        order = np.argsort(w)
        cols = ATT_CMAP(w[order]); cols[:, 3] = 0.30 + 0.65 * w[order]
        ax.scatter(cx[order], cy[order], s=6 + 60 * w[order], c=cols, edgecolors="none")
        for k, j in enumerate(meta["top"]):
            ax.scatter(cx[j], cy[j], s=130, facecolors="none", edgecolors=TEXT, linewidths=1.5)
            ax.annotate(str(k + 1), (cx[j], cy[j]), xytext=(7, 6), textcoords="offset points",
                        color=TEXT, fontsize=8.5, fontweight="bold")
        ax.axis("off"); ax.set_anchor("C")

        for k in range(3):
            axc = fig.add_subplot(gs[g0 + 1, k + 1])
            axc.imshow(plt.imread(os.path.join(d, f"top{k + 1}.png")))
            axc.set_xticks([]); axc.set_yticks([])
            for sp in axc.spines.values():
                sp.set_color(REF)
            axc.set_title(f"#{k + 1} · {100 * att[meta['top'][k]]:.1f}% of attention", fontsize=8.5, color=TEXT2)

        # note: what the top tiles show (clearly labelled as a non-expert read)
        nax = fig.add_subplot(gs[g0 + 2, :]); nax.axis("off")
        note = TILE_NOTES.get(row.patient, "")
        lab = nax.text(0.022, 0.9, "What the top tiles show: ", fontsize=8.5, fontweight="bold", va="top",
                       transform=nax.transAxes)
        nax.annotate(note, xy=(1, 0), xycoords=lab, va="bottom", ha="left", fontsize=8.5)   # starts where label ends

    # --- bottom: color scale + honest caveat ---
    bax = fig.add_subplot(gs[-1, :]); bax.axis("off")
    cax = bax.inset_axes([0.0, 0.55, 0.2, 0.14])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ATT_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["counted little", "counted most"]); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    lo, hi = cb.ax.get_xticklabels()
    lo.set_horizontalalignment("left"); hi.set_horizontalalignment("right")   # keep labels inside the bar's span
    bax.text(0.0, 0.95, "Dot color & size = attention", fontsize=8.5, color=TEXT2, va="top", transform=bax.transAxes)
    bax.text(0.27, 0.95, "Tile notes are a non-expert visual read and need review by a pathologist. Examples were "
             "hand-picked to illustrate\n(including a miss); they show what the model weighted, not what causes the "
             "outcome. Data: TCGA-BRCA, n = 244.", fontsize=8, color=TEXT2, va="top", transform=bax.transAxes)

    out = os.path.join(config.FIG_DIR, "fig6b_attention_heatmaps_explained.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log(f"saved {out}")


if __name__ == "__main__":
    if "--recompute" in sys.argv or not os.path.exists(os.path.join(OUT, "examples.csv")):
        compute()
    plot()
    plot_explained()
