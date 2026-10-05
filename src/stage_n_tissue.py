"""
stage_n_tissue.py — is the image signal tumor biology, or just tissue composition? (Amendment B)

Runs the analyses fixed in PREREGISTRATION.md §13 (written 2026-10-03, BEFORE running) on the
n = 1,033 cohort:

  B1  Cluster every tile (≈ 206,000) by appearance: standardize the 768 Phikon features, PCA to 50
      components, k-means k = 8 (sensitivity k = 12). No outcomes are used. Each patient's "tissue
      composition" = the share of their tiles in each cluster.
  B2  Where does the image model's attention go? Re-train the five out-of-fold image models (checked
      against the cached risks), take each patient's attention from the model that never saw them,
      and compare each cluster's share of attention with its share of tiles.
  B3  PRIMARY: does the image risk still predict survival after adjusting for age, stage, T, N AND
      tissue composition? (Cox likelihood-ratio test, alpha = 0.05.) If yes, the signal is not just
      "how much fat / stroma / background the slide has".
  B4  (needs GDC — added when the API is back) label clusters from re-downloaded tile crops and
      re-train without fat / background tiles.

Outputs: results/fullcohort/tissue/ (tables, figure), data/stage_n_tissue/ (per-tile clusters and
attention; large, gitignored).

Run:  python src/stage_n_tissue.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                       # noqa: E402
import make_figures as mf           # noqa: E402
import stage_h_clinical as h        # noqa: E402
import stage_k_heatmaps as K        # noqa: E402  train_fold_models (keeps models) + attention()
import stage_l_fullcohort as L      # noqa: E402  tile loader
import stage_m_added_value as M     # noqa: E402  cohort table with clinical columns

SEED = config.RANDOM_SEED
KS = (8, 12)                                      # primary k, sensitivity k
OUT = os.path.join(L.OUT, "tissue")
DATA = os.path.join(config.DATA_DIR, "stage_n_tissue")
log, z = mf.log, mf.z


# ------------------------------------------------------------------ B1 clustering
def cluster_tiles(pids, tiles):
    cache = os.path.join(DATA, "tile_clusters.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        return {k: c[k] for k in c.files}
    X = np.concatenate([tiles[p] for p in pids]).astype(np.float32)
    owner = np.repeat(np.arange(len(pids)), [tiles[p].shape[0] for p in pids])
    tile_idx = np.concatenate([np.arange(tiles[p].shape[0]) for p in pids])
    log(f"B1: {X.shape[0]:,} tiles from {len(pids)} patients -> standardize, PCA(50)")
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    pca = PCA(50, svd_solver="randomized", random_state=SEED).fit(Xs)
    Z = pca.transform(Xs).astype(np.float32)
    log(f"B1: PCA keeps {100 * pca.explained_variance_ratio_.sum():.1f}% of the variance")
    out = {"owner": owner, "tile_idx": tile_idx, "pca_var": np.array([pca.explained_variance_ratio_.sum()])}
    for k in KS:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        out[f"labels_k{k}"] = km.labels_.astype(np.int16)
        out[f"dist_k{k}"] = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1).astype(np.float32)
        log(f"B1: k-means k={k} done (inertia {km.inertia_:.3g})")
    np.savez_compressed(cache, **out)
    return out


def composition(cl, n, k):
    """n x k matrix: share of each patient's tiles in each cluster."""
    lab, own = cl[f"labels_k{k}"], cl["owner"]
    counts = np.zeros((n, k))
    np.add.at(counts, (own, lab), 1)
    return counts / counts.sum(1, keepdims=True)


# ------------------------------------------------------------------ B2 attention
def tile_attention(df, pids, tiles):
    cache = os.path.join(DATA, "tile_attention.npy")
    if os.path.exists(cache):
        return np.load(cache)
    bags = [torch.from_numpy(tiles[p]) for p in pids]
    d, ev = df.time.to_numpy(float), df.event.to_numpy(bool)
    log("B2: re-training the 5 out-of-fold image models (same settings as Entry 13)...")
    models, fold_of, oof = K.train_fold_models(bags, d, ev)
    dev = float(np.max(np.abs(oof - df.img_random.to_numpy())))
    log(f"B2 check: re-trained out-of-fold image risk vs cache, max |difference| = {dev:.2e}")
    assert dev < 1e-4, "re-trained models do not reproduce the cached image risks"
    att = np.concatenate([K.attention(models[fold_of[i]], bags[i]) for i in range(len(pids))]).astype(np.float32)
    np.save(cache, att)
    return att


def attention_by_cluster(cl, att, n, k, img_risk):
    lab, own = cl[f"labels_k{k}"], cl["owner"]
    att_share = np.zeros((n, k))
    np.add.at(att_share, (own, lab), att)                  # each patient's attention sums to 1
    tile_share = composition(cl, n, k)
    rows = []
    for j in range(k):
        rho = spearmanr(tile_share[:, j], img_risk).correlation
        rows.append({"cluster": j + 1, "n_tiles": int((lab == j).sum()),
                     "mean_tile_share": round(float(tile_share[:, j].mean()), 4),
                     "mean_attention_share": round(float(att_share[:, j].mean()), 4),
                     "enrichment": round(float(att_share[:, j].mean() / tile_share[:, j].mean()), 2),
                     "patients_with_cluster": int((tile_share[:, j] > 0).sum()),
                     "spearman_fraction_vs_image_risk": round(float(rho), 3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B3 primary test
def lrt(base, add, d, ev):
    """Cox likelihood-ratio test for adding the columns in `add` to `base`."""
    from lifelines import CoxPHFitter
    data = base.assign(time=d, event=ev)
    try:
        m0 = CoxPHFitter().fit(data, "time", "event")
        m1 = CoxPHFitter().fit(data.assign(**add), "time", "event")
        pen = 0.0
    except Exception:                                      # convergence fallback (reported)
        pen = 0.01
        m0 = CoxPHFitter(penalizer=pen).fit(data, "time", "event")
        m1 = CoxPHFitter(penalizer=pen).fit(data.assign(**add), "time", "event")
    stat = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
    return stat, float(chi2.sf(stat, len(add))), m1, pen


def primary_tests(df, comps):
    d, ev = df.time.to_numpy(float), df.event.to_numpy(bool)
    clin = df[M.CLIN_COLS].reset_index(drop=True)
    img = {"image": z(df.img_random).to_numpy()}
    rows = []
    for k, comp in comps.items():
        ref = int(np.argmax(comp.mean(0)))                 # largest cluster = reference (LRT is unaffected)
        cf = pd.DataFrame({f"c{j + 1}_per10pct": comp[:, j] * 10 for j in range(k) if j != ref})
        stat, p, m1, pen = lrt(pd.concat([clin, cf], axis=1), img, d, ev)
        hr = float(m1.hazard_ratios_["image"]); lo, hi = np.exp(m1.confidence_intervals_.loc["image"].to_numpy())
        rows.append({"test": f"+ image vs clinical + composition (k={k})" + ("  [PRIMARY]" if k == 8 else "  [sensitivity]"),
                     "LR_chi2": round(stat, 2), "df": 1, "p_LRT": p, "HR_image_per_SD": round(hr, 2),
                     "HR_lo": round(lo, 2), "HR_hi": round(hi, 2), "penalizer": pen})
        if k == 8:
            m1.summary.round(4).to_csv(os.path.join(OUT, "b3_cox_clinical_composition_image.csv"))
            stat2, p2, _, pen2 = lrt(pd.concat([clin, pd.DataFrame(img)], axis=1), {c: cf[c].to_numpy() for c in cf}, d, ev)
            rows.append({"test": "+ composition (k=8) vs clinical + image  [secondary b]", "LR_chi2": round(stat2, 2),
                         "df": k - 1, "p_LRT": p2, "HR_image_per_SD": np.nan, "HR_lo": np.nan, "HR_hi": np.nan, "penalizer": pen2})
            folds = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(d))))
            r = h.oof_cox_risk(cf.to_numpy(), d, ev, folds)
            c, clo, chi = mf.ci(r, d, ev)
            comp_only = pd.DataFrame([{"model": "tissue composition alone (k=8), out-of-fold Cox  [secondary a]",
                                       "c_index": round(c, 3), "ci_lo": round(clo, 3), "ci_hi": round(chi, 3)}])
            comp_only.to_csv(os.path.join(OUT, "b3_composition_only_cindex.csv"), index=False)
            print(comp_only.to_string(index=False))
    stat0, p0, m0, _ = lrt(clin, img, d, ev)               # A1 again, for comparison (no composition)
    rows.append({"test": "reference: + image vs clinical only (A1, Entry 14)", "LR_chi2": round(stat0, 2), "df": 1,
                 "p_LRT": p0, "HR_image_per_SD": round(float(m0.hazard_ratios_["image"]), 2),
                 "HR_lo": round(float(np.exp(m0.confidence_intervals_.loc["image"].iloc[0])), 2),
                 "HR_hi": round(float(np.exp(m0.confidence_intervals_.loc["image"].iloc[1])), 2), "penalizer": 0.0})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "b3_primary_tests.csv"), index=False)
    print(res.to_string(index=False))
    return res


def hospitals_per_cluster(comp, pids):
    """Descriptive (post hoc): among patients whose tiles are >50% in a cluster, how many hospitals?
    A cluster drawn from one hospital reflects that lab's stain/scanner style, not a tissue type."""
    site = np.array([p.split("-")[1] for p in pids])
    return {j + 1: len(set(site[comp[:, j] > 0.5])) for j in range(comp.shape[1])}


def plot_attention(tab, k, n_hosp):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": mf.TEXT, "axes.labelcolor": mf.TEXT,
                         "figure.facecolor": mf.SURFACE, "axes.facecolor": mf.SURFACE})
    t = tab.sort_values("mean_tile_share", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * k + 1.6))
    mf.style(ax); ax.grid(axis="y", visible=False)
    y = np.arange(len(t))
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.plot([r.mean_tile_share, r.mean_attention_share], [yi, yi], color=mf.GRID, lw=3, zorder=1)
    ax.scatter(t.mean_tile_share, y, s=70, color=mf.BLUE, edgecolor=mf.SURFACE, lw=1.5, zorder=2, label="Share of tiles")
    ax.scatter(t.mean_attention_share, y, s=70, color=mf.ORANGE, marker="s", edgecolor=mf.SURFACE, lw=1.5, zorder=3,
               label="Share of the model's attention")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Cluster {int(r.cluster)} · {n_hosp[int(r.cluster)]} "
                        f"hospital{'s' if n_hosp[int(r.cluster)] != 1 else ''}  ×{r.enrichment:.2f}" for _, r in t.iterrows()])
    ax.set_xlabel("Average share per patient")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2, frameon=False, fontsize=8)
    ax.set_title(f"Where the image model looks, by tile cluster (k = {k}; ×enrichment = attention ÷ tiles)",
                 loc="left", fontsize=10, pad=22)
    ax.text(0, 1.01, "Hospitals = hospitals among patients whose tiles are mostly in that cluster; "
            "1 hospital = that lab's staining style, not a tissue type", transform=ax.transAxes,
            color=mf.TEXT2, fontsize=7.5, va="bottom")
    fig.savefig(os.path.join(L.FIGDIR, "fig11_attention_by_cluster.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ B4 cluster examples (needs GDC)
B4_N, B4_PER_CLUSTER, B4_MAX_BYTES = 30, 10, 0.7e9


def b4_examples(pids, cl):
    """Pre-specified sample: 30 patients (seed 42) with an exactly-known slide <= 0.7 GB and saved tile
    positions (only the 789 full-run patients have them); then, per k=8 cluster, the 10 tiles of those
    patients nearest the cluster centre."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = pd.read_csv(os.path.join(root, "reproducibility", "cohort_manifest.csv"))   # GDC slide ids + MD5s
    elig = m[(m.slide_status == "exact") & (m.coords_sha256 != "not saved (original run)")].copy()
    elig["slide_size"] = elig.slide_size.astype(float)
    elig = elig[elig.slide_size <= B4_MAX_BYTES].sort_values("patient")
    pick = sorted(np.random.RandomState(SEED).choice(elig.patient.to_numpy(), B4_N, replace=False))
    idx = {p: i for i, p in enumerate(pids)}
    own, lab, dist, tix = cl["owner"], cl["labels_k8"], cl["dist_k8"], cl["tile_idx"]
    in_pick = np.isin(own, [idx[p] for p in pick])
    rows = []
    for j in range(8):
        cand = np.where(in_pick & (lab == j))[0]
        for rank, t in enumerate(cand[np.argsort(dist[cand])][:B4_PER_CLUSTER]):
            rows.append({"cluster": j + 1, "rank": rank + 1, "patient": pids[own[t]], "tile_idx": int(tix[t]),
                         "dist_to_centre": round(float(dist[t]), 3)})
    ex = pd.DataFrame(rows)
    sample = elig.set_index("patient").loc[pick].reset_index()[["patient", "slide_file_id", "slide_size", "slide_md5"]]
    return sample, ex


def b4_crops(pids, tiles, cl):
    """Download only the slides holding example tiles (resumable + MD5-verified), crop each example at its
    exact saved position, check the crop's Phikon features match the saved ones, then delete the slide."""
    import collections
    from concurrent.futures import ThreadPoolExecutor
    import openslide
    sys.argv = [sys.argv[0]]                              # import the slide pipeline in its normal mode
    import stage_c_pathology as sc                        # _download (resume + MD5), Phikon model, tile settings
    sample, ex = b4_examples(pids, cl)
    sample.to_csv(os.path.join(OUT, "b4_sample.csv"), index=False)
    ex.to_csv(os.path.join(OUT, "b4_examples.csv"), index=False)
    need = sample[sample.patient.isin(ex.patient)]
    log(f"B4: {len(ex)} example tiles from {ex.patient.nunique()} patients; downloading {len(need)} slides "
        f"({need.slide_size.sum() / 1e9:.1f} GB)")
    crop_dir, slide_dir = os.path.join(DATA, "crops"), os.path.join(DATA, "slides")
    os.makedirs(crop_dir, exist_ok=True); os.makedirs(slide_dir, exist_ok=True)
    R = collections.namedtuple("R", "patient file_id size md5")
    checks = []

    def one(row):
        p = row.patient
        tmp = os.path.join(slide_dir, f"{row.slide_file_id}.svs")
        todo = ex[ex.patient == p]
        if all(os.path.exists(os.path.join(crop_dir, f"c{r.cluster}_{r['rank']:02d}_{p}_{r.tile_idx}.png")) for _, r in todo.iterrows()):
            return []
        if not sc._download(R(p, row.slide_file_id, int(row.slide_size), row.slide_md5), tmp):
            log(f"B4: download FAILED for {p}"); return []
        s = openslide.OpenSlide(tmp)
        lvl = s.get_best_level_for_downsample(sc.DOWNSAMPLE)
        xy = np.load(os.path.join(config.DATA_DIR, "stage_c_coords", f"{p}.npy"))
        out = []
        for _, r in todo.iterrows():
            x0, y0, _ = xy[r.tile_idx]
            im = s.read_region((int(x0), int(y0)), lvl, (sc.TILE, sc.TILE)).convert("RGB")
            with sc._flock, torch.no_grad():
                f = sc._model(**sc._proc(images=[im], return_tensors="pt")).last_hidden_state[0, 0].numpy()
            saved = tiles[p][r.tile_idx].astype(np.float32)
            cos = float(f @ saved / (np.linalg.norm(f) * np.linalg.norm(saved)))
            im.save(os.path.join(crop_dir, f"c{r.cluster}_{r['rank']:02d}_{p}_{r.tile_idx}.png"))
            out.append({"cluster": r.cluster, "rank": r["rank"], "patient": p, "tile_idx": r.tile_idx, "cosine_vs_saved": round(cos, 4)})
        s.close()
        os.remove(tmp)                                    # keep nothing but the crops
        log(f"B4: {p} cropped ({len(out)} tiles), slide deleted")
        return out

    with ThreadPoolExecutor(6) as pool:
        for res in pool.map(one, [r for _, r in need.iterrows()]):
            checks += res
    if checks:
        prev = os.path.join(OUT, "b4_crop_checks.csv")
        old = pd.read_csv(prev) if os.path.exists(prev) else pd.DataFrame()
        pd.concat([old, pd.DataFrame(checks)]).drop_duplicates(["cluster", "rank"], keep="last").to_csv(prev, index=False)
    chk = pd.read_csv(os.path.join(OUT, "b4_crop_checks.csv"))
    log(f"B4 check: crop vs saved features, cosine similarity min {chk.cosine_vs_saved.min():.4f}, "
        f"median {chk.cosine_vs_saved.median():.4f} over {len(chk)} crops")
    b4_sheet(ex)


def b4_sheet(ex):
    """One figure: a row of example crops per cluster (nearest the centre first)."""
    import matplotlib.pyplot as plt
    crop_dir = os.path.join(DATA, "crops")
    fig, axes = plt.subplots(8, B4_PER_CLUSTER, figsize=(B4_PER_CLUSTER * 1.6, 8 * 1.75))
    for j in range(8):
        sub = ex[ex.cluster == j + 1].sort_values("rank")
        for c in range(B4_PER_CLUSTER):
            ax = axes[j, c]; ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(mf.GRID)
            if c < len(sub):
                r = sub.iloc[c]
                path = os.path.join(crop_dir, f"c{r.cluster}_{int(r['rank']):02d}_{r.patient}_{r.tile_idx}.png")
                if os.path.exists(path):
                    ax.imshow(plt.imread(path))
            if c == 0:
                ax.set_ylabel(f"Cluster {j + 1}" + ("" if len(sub) else "\n(no tiles\nin sample)"), fontsize=9,
                              rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("B4: the 10 tiles nearest each cluster's centre (k = 8), from 30 randomly chosen patients",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(os.path.join(L.FIGDIR, "fig12_cluster_examples.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__" and "--b4-crops" in sys.argv:
    warnings.simplefilter("ignore")
    os.makedirs(OUT, exist_ok=True)
    df = M.load(); pids = df.patient.tolist()
    tiles, _ = L.load_tiles_all()
    b4_crops(pids, tiles, cluster_tiles(pids, tiles))
    log("B4 crops done -> results/figures/fullcohort/fig12_cluster_examples.png")
    sys.exit(0)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    df = M.load()
    pids = df.patient.tolist()
    tiles, _ = L.load_tiles_all()
    assert all(p in tiles for p in pids)
    cl = cluster_tiles(pids, tiles)
    print(f"PCA(50) keeps {100 * float(cl['pca_var'][0]):.1f}% of the feature variance")
    comps = {k: composition(cl, len(pids), k) for k in KS}
    for k, comp in comps.items():
        pd.DataFrame(comp, columns=[f"cluster_{j + 1}" for j in range(k)]).assign(patient=pids).to_csv(
            os.path.join(OUT, f"composition_k{k}.csv"), index=False)
    att = tile_attention(df, pids, tiles)
    for k in KS:
        tab = attention_by_cluster(cl, att, len(pids), k, df.img_random.to_numpy())
        tab.to_csv(os.path.join(OUT, f"b2_attention_by_cluster_k{k}.csv"), index=False)
        print(f"\n===== B2  attention by cluster, k = {k} =====\n{tab.to_string(index=False)}")
    n_hosp = hospitals_per_cluster(comps[8], pids)
    pd.DataFrame({"cluster": list(n_hosp), "hospitals_where_dominant": list(n_hosp.values())}).to_csv(
        os.path.join(OUT, "b1_hospitals_per_cluster_k8.csv"), index=False)
    plot_attention(pd.read_csv(os.path.join(OUT, "b2_attention_by_cluster_k8.csv")), 8, n_hosp)
    print("\n===== B3  primary test: image risk beyond clinical + tissue composition =====")
    primary_tests(df, comps)
    log("done -> results/fullcohort/tissue/")
