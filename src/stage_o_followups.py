"""
stage_o_followups.py — Amendment C (PREREGISTRATION.md §14, written 2026-10-04 BEFORE running).

  C1  Is there a FAT sub-cluster inside k = 8 cluster 2 (the cluster the image model attends to)?
        --c1-crops   recompute the B1 PCA space (checked), sub-cluster cluster 2 with k = 4, measure each
                     sub-cluster's share of attention, download the example slides (MD5-verified), crop the
                     10 tiles nearest each sub-centre (features checked), delete the slides, draw fig13.
        --c1-filter  after the sub-clusters are labelled in results/fullcohort/tissue/c1_labels.csv:
                     if any is flagged fat/background, re-train the image model without those tiles and run
                     the primary LRT (fat-free image risk beyond age + stage + T + N).
  C2  Robustness: (a) 100 tiles per patient instead of 200 (re-train), (b) five fixed combination
      weightings, (c) results without the flagged blank-slide patient.            [default run]
  C3  Gene explanations: exact SHAP values for the (linear) gene model, fig14.    [default run]

Outputs: results/fullcohort/followups/, results/figures/fullcohort/fig13–14.
Run:  python src/stage_o_followups.py            (C2 + C3)
      python src/stage_o_followups.py --c1-crops
      python src/stage_o_followups.py --c1-filter
"""
from __future__ import annotations

import collections
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
import stage_e_attmil as e          # noqa: E402
import stage_l_fullcohort as L      # noqa: E402
import stage_m_added_value as M     # noqa: E402
import stage_n_tissue as N          # noqa: E402

SEED = config.RANDOM_SEED
OUT = os.path.join(L.OUT, "followups")
TISSUE = N.OUT
DATA = N.DATA
FLAGGED_PATIENT = "TCGA-OL-A5RZ"
log, z = mf.log, mf.z


def base_data():
    df = M.load()
    pids = df.patient.tolist()
    tiles, _ = L.load_tiles_all()
    return df, pids, tiles


def image_lrt(df, risk, mask=None):
    """A1-style Cox LRT: does `risk` (z-scored) add to age + stage + T + N?"""
    m = np.ones(len(df), bool) if mask is None else mask
    clin = df.loc[m, M.CLIN_COLS].reset_index(drop=True)
    stat, p, m1, _ = N.lrt(clin, {"image": z(np.asarray(risk)[m])}, df.time.to_numpy(float)[m], df.event.to_numpy(bool)[m])
    lo, hi = np.exp(m1.confidence_intervals_.loc["image"].to_numpy())
    return {"LR_chi2": round(stat, 2), "p_LRT": p, "HR_per_SD": round(float(m1.hazard_ratios_["image"]), 2),
            "HR_lo": round(lo, 2), "HR_hi": round(hi, 2)}


def cidx_row(name, r, d, ev):
    c, lo, hi = mf.ci(np.asarray(r), d, ev)
    return {"model": name, "c_index": round(c, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)}


# ------------------------------------------------------------------ C1 fat inside cluster 2
def c1_subclusters(pids, tiles, cl):
    cache = os.path.join(DATA, "c1_cluster2_sub.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        return {k: c[k] for k in c.files}
    from sklearn.metrics import adjusted_rand_score
    X = np.concatenate([tiles[p] for p in pids]).astype(np.float32)
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    pca = PCA(50, svd_solver="randomized", random_state=SEED).fit(Xs)       # exactly the B1 code path
    Z = pca.transform(Xs).astype(np.float32)
    # check 1 (renumbering-proof): each tile's distance to the centre of its SAVED B1 cluster must equal
    # the distance B1 stored. Centres = mean of each saved cluster in the recomputed space.
    lab8 = cl["labels_k8"]
    cen = np.stack([Z[lab8 == j].mean(0) for j in range(8)])
    dz = np.linalg.norm(Z - cen[lab8], axis=1)
    rel = np.abs(dz - cl["dist_k8"]) / np.maximum(cl["dist_k8"], 1e-6)
    log(f"C1 check 1: distance to saved-cluster centre vs B1, median rel. diff {np.median(rel):.2e}, "
        f"99th pct {np.percentile(rel, 99):.2e}, correlation {np.corrcoef(dz, cl['dist_k8'])[0, 1]:.6f}")
    assert np.median(rel) < 1e-3 and np.corrcoef(dz, cl["dist_k8"])[0, 1] > 0.9999, "recomputed PCA space differs from B1"
    # check 2 (information): k-means k = 8 again in this space vs the saved partition, ignoring numbering
    ari = adjusted_rand_score(lab8, KMeans(n_clusters=8, n_init=10, random_state=SEED).fit(Z).labels_)
    log(f"C1 check 2: agreement with B1 partition ignoring cluster numbering (adjusted Rand) = {ari:.4f}")
    idx = np.where(cl["labels_k8"] == 1)[0]                     # 0-based 1 = cluster 2
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(Z[idx])
    out = {"tile_rows": idx, "sub": km.labels_.astype(np.int16),
           "dist": np.linalg.norm(Z[idx] - km.cluster_centers_[km.labels_], axis=1).astype(np.float32)}
    np.savez_compressed(cache, **out)
    log(f"C1: {len(idx):,} cluster-2 tiles split into 4 sub-clusters")
    return out


def c1_attention(df, pids, cl, sub):
    att = np.load(os.path.join(DATA, "tile_attention.npy"))
    own = cl["owner"]; n = len(pids)
    lab = np.full(len(own), -1); lab[sub["tile_rows"]] = sub["sub"]   # -1 = not cluster 2
    rows = []
    tile_share = np.zeros((n, 4)); att_share = np.zeros((n, 4))
    m = lab >= 0
    np.add.at(tile_share, (own[m], lab[m]), 1)
    np.add.at(att_share, (own[m], lab[m]), att[m])
    tile_share /= np.bincount(own, minlength=n)[:, None]
    for j in range(4):
        rows.append({"sub_cluster": f"2{'abcd'[j]}", "n_tiles": int((lab == j).sum()),
                     "mean_tile_share": round(float(tile_share[:, j].mean()), 4),
                     "mean_attention_share": round(float(att_share[:, j].mean()), 4),
                     "enrichment": round(float(att_share[:, j].mean() / tile_share[:, j].mean()), 2),
                     "patients_with_sub": int((tile_share[:, j] > 0).sum()),
                     "spearman_fraction_vs_image_risk": round(float(spearmanr(tile_share[:, j], df.img_random).correlation), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "c1_subcluster_attention.csv"), index=False)
    print(res.to_string(index=False))
    return tile_share


def c1_crops(df, pids, tiles, cl, sub):
    sys.argv = [sys.argv[0]]
    import openslide
    import stage_c_pathology as sc
    from concurrent.futures import ThreadPoolExecutor
    sample = pd.read_csv(os.path.join(TISSUE, "b4_sample.csv"))           # the same 30 B4 patients
    idx = {p: i for i, p in enumerate(pids)}
    own, tix = cl["owner"], cl["tile_idx"]
    rows_ = sub["tile_rows"]; in_s = np.isin(own[rows_], [idx[p] for p in sample.patient])
    ex = []
    for j in range(4):
        cand = np.where(in_s & (sub["sub"] == j))[0]
        for rank, k in enumerate(cand[np.argsort(sub["dist"][cand])][:10]):
            t = rows_[k]
            ex.append({"sub_cluster": f"2{'abcd'[j]}", "rank": rank + 1, "patient": pids[own[t]], "tile_idx": int(tix[t])})
    ex = pd.DataFrame(ex)
    ex.to_csv(os.path.join(OUT, "c1_examples.csv"), index=False)
    need = sample[sample.patient.isin(ex.patient)]
    log(f"C1: {len(ex)} example tiles from {ex.patient.nunique()} patients; downloading {len(need)} slides "
        f"({need.slide_size.astype(float).sum() / 1e9:.1f} GB)")
    crop_dir, slide_dir = os.path.join(DATA, "c1_crops"), os.path.join(DATA, "slides")
    os.makedirs(crop_dir, exist_ok=True); os.makedirs(slide_dir, exist_ok=True)
    R = collections.namedtuple("R", "patient file_id size md5")

    def name(r):
        return os.path.join(crop_dir, f"{r.sub_cluster}_{int(r['rank']):02d}_{r.patient}_{r.tile_idx}.png")

    def one(row):
        p, todo = row.patient, ex[ex.patient == row.patient]
        if all(os.path.exists(name(r)) for _, r in todo.iterrows()):
            return []
        tmp = os.path.join(slide_dir, f"{row.slide_file_id}.svs")
        if not sc._download(R(p, row.slide_file_id, int(float(row.slide_size)), row.slide_md5), tmp):
            log(f"C1: download FAILED for {p}"); return []
        s = openslide.OpenSlide(tmp); lvl = s.get_best_level_for_downsample(sc.DOWNSAMPLE)
        xy = np.load(os.path.join(config.DATA_DIR, "stage_c_coords", f"{p}.npy"))
        out = []
        for _, r in todo.iterrows():
            x0, y0, _ = xy[r.tile_idx]
            im = s.read_region((int(x0), int(y0)), lvl, (sc.TILE, sc.TILE)).convert("RGB")
            with sc._flock, torch.no_grad():
                f = sc._model(**sc._proc(images=[im], return_tensors="pt")).last_hidden_state[0, 0].numpy()
            saved = tiles[p][r.tile_idx].astype(np.float32)
            out.append({"sub_cluster": r.sub_cluster, "rank": int(r["rank"]), "patient": p, "tile_idx": int(r.tile_idx),
                        "cosine_vs_saved": round(float(f @ saved / (np.linalg.norm(f) * np.linalg.norm(saved))), 4)})
            im.save(name(r))
        s.close(); os.remove(tmp)
        log(f"C1: {p} cropped ({len(out)} tiles), slide deleted")
        return out

    checks = []
    with ThreadPoolExecutor(6) as pool:
        for res in pool.map(one, [r for _, r in need.iterrows()]):
            checks += res
    path = os.path.join(OUT, "c1_crop_checks.csv")
    if checks:
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        pd.concat([old, pd.DataFrame(checks)]).drop_duplicates(["sub_cluster", "rank"], keep="last").to_csv(path, index=False)
    chk = pd.read_csv(path)
    log(f"C1 check: crop vs saved features, cosine min {chk.cosine_vs_saved.min():.4f} over {len(chk)} crops")
    c1_sheet(ex, crop_dir)


def c1_sheet(ex, crop_dir):
    import matplotlib.pyplot as plt
    att = pd.read_csv(os.path.join(OUT, "c1_subcluster_attention.csv")).set_index("sub_cluster")
    fig, axes = plt.subplots(4, 10, figsize=(16, 7.4))
    for j, sc_ in enumerate(["2a", "2b", "2c", "2d"]):
        sub_ = ex[ex.sub_cluster == sc_].sort_values("rank")
        for c in range(10):
            ax = axes[j, c]; ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(mf.GRID)
            if c < len(sub_):
                r = sub_.iloc[c]
                pth = os.path.join(crop_dir, f"{r.sub_cluster}_{int(r['rank']):02d}_{r.patient}_{r.tile_idx}.png")
                if os.path.exists(pth):
                    ax.imshow(plt.imread(pth))
            if c == 0:
                a = att.loc[sc_]
                ax.set_ylabel(f"{sc_}\n{100 * a.mean_tile_share:.1f}% of tiles\nattention ×{a.enrichment:.2f}",
                              fontsize=8.5, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("C1: cluster 2 split into 4 sub-clusters — the 10 tiles nearest each centre (same 30 patients as B4)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(os.path.join(L.FIGDIR, "fig13_cluster2_subclusters.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def c1_filter(df, pids, tiles, cl, sub):
    lab = pd.read_csv(os.path.join(OUT, "c1_labels.csv"))
    flagged = lab[lab.rule_flags_fat_or_background].sub_cluster.tolist()
    if not flagged:
        print("C1: no sub-cluster meets the fat/background rule -> nothing re-trained (pre-specified).")
        pd.DataFrame([{"result": "no sub-cluster flagged; no re-training"}]).to_csv(os.path.join(OUT, "c1_filtered_result.csv"), index=False)
        return
    codes = ["2a", "2b", "2c", "2d"]
    own, tix = cl["owner"], cl["tile_idx"]
    drop = np.zeros(len(own), bool)
    for f in flagged:
        drop[sub["tile_rows"][sub["sub"] == codes.index(f)]] = True
    bags, kept_all, frac = [], 0, np.zeros(len(pids))
    for i, p in enumerate(pids):
        m = own == i
        keep_idx = tix[m & ~drop]
        frac[i] = 1 - len(keep_idx) / m.sum()
        if len(keep_idx) < 5:
            keep_idx = tix[m]; kept_all += 1
        bags.append(torch.from_numpy(tiles[p][np.sort(keep_idx)]))
    log(f"C1: re-training the image model without sub-cluster(s) {flagged} "
        f"({int(drop.sum()):,} tiles removed; {kept_all} patients kept all tiles)")
    d, ev = df.time.to_numpy(float), df.event.to_numpy(bool)
    cache = os.path.join(OUT, "c1_fatfree_risk.csv")          # cached so re-runs do not re-train
    if os.path.exists(cache):
        r = pd.read_csv(cache).img_fatfree.to_numpy()
    else:
        e.SEED = SEED; torch.manual_seed(SEED)
        G = np.zeros((len(pids), 1), np.float32)
        r = e.run(bags, G, d, ev, use_img=True, use_gene=False)
        pd.DataFrame({"patient": pids, "img_fatfree": r, "flagged_fraction": frac}).to_csv(cache, index=False)
    rows = [{"analysis": f"image risk without flagged tiles ({', '.join(flagged)}) + clinical vs clinical  [C1 PRIMARY]", **image_lrt(df, r)},
            {"analysis": "reference: original image risk + clinical vs clinical (A1)", **image_lrt(df, df.img_random.to_numpy())}]
    clin = df[M.CLIN_COLS].reset_index(drop=True).assign(flagged_fraction_per10pct=frac * 10)
    stat, p_, m1, _ = N.lrt(clin, {"image": z(df.img_random).to_numpy()}, d, ev)
    rows.append({"analysis": "original image risk beyond clinical + flagged-tile fraction  [C1 secondary]",
                 "LR_chi2": round(stat, 2), "p_LRT": p_, "HR_per_SD": round(float(m1.hazard_ratios_["image"]), 2),
                 "HR_lo": round(float(np.exp(m1.confidence_intervals_.loc["image"].iloc[0])), 2),
                 "HR_hi": round(float(np.exp(m1.confidence_intervals_.loc["image"].iloc[1])), 2)})
    res = pd.DataFrame(rows)
    cres = pd.DataFrame([cidx_row(f"image model without flagged tiles ({', '.join(flagged)})", r, d, ev), cidx_row("original image model", df.img_random, d, ev)])
    res.to_csv(os.path.join(OUT, "c1_filtered_result.csv"), index=False)
    cres.to_csv(os.path.join(OUT, "c1_filtered_cindex.csv"), index=False)
    print(res.to_string(index=False)); print(cres.to_string(index=False))


# ------------------------------------------------------------------ C2 robustness
def c2(df, pids, tiles):
    d, ev = df.time.to_numpy(float), df.event.to_numpy(bool)
    clin, img, gene = df.clin_random.to_numpy(), df.img_random.to_numpy(), df.gene_random.to_numpy()
    # (a) 100 tiles per patient
    cache = os.path.join(OUT, "c2a_img100_risk.csv")
    if not os.path.exists(cache):
        rng = np.random.RandomState(SEED); bags = []
        for p in pids:
            n_t = tiles[p].shape[0]
            keep = np.sort(rng.choice(n_t, 100, replace=False)) if n_t > 100 else np.arange(n_t)
            bags.append(torch.from_numpy(tiles[p][keep]))
        log("C2a: re-training the image model on 100 tiles per patient...")
        e.SEED = SEED; torch.manual_seed(SEED)
        r100 = e.run(bags, np.zeros((len(pids), 1), np.float32), d, ev, use_img=True, use_gene=False)
        pd.DataFrame({"patient": pids, "img_100": r100}).to_csv(cache, index=False)
    r100 = pd.read_csv(cache).img_100.to_numpy()
    a = pd.DataFrame([{**cidx_row("image model, 200 tiles (Entry 13)", img, d, ev), **image_lrt(df, img)},
                      {**cidx_row("image model, 100 tiles  [C2a]", r100, d, ev), **image_lrt(df, r100)}])
    a["spearman_with_200_tile_risk"] = [1.0, round(float(spearmanr(r100, img).correlation), 3)]
    a.to_csv(os.path.join(OUT, "c2a_fewer_tiles.csv"), index=False)
    print("\n===== C2a  fewer tiles =====\n" + a.to_string(index=False))
    # (b) combination weights
    rows = []
    for wc, wi, wg in [(1, 1, 1), (2, 1, 1), (3, 1, 1), (1, 1, 0), (2, 1, 0)]:
        r = wc * z(clin) + wi * z(img) + wg * z(gene)
        m_, l_, h_, p_ = M.delta2(r, clin, d, ev)
        rows.append({**cidx_row(f"clinical:image:gene = {wc}:{wi}:{wg}", r, d, ev),
                     "delta_vs_clinical": round(m_, 3), "delta_lo": round(l_, 3), "delta_hi": round(h_, 3), "p_two_sided": round(p_, 3)})
    b = pd.DataFrame(rows)
    b.to_csv(os.path.join(OUT, "c2b_weights.csv"), index=False)
    print("\n===== C2b  combination weights (clinical alone = 0.722) =====\n" + b.to_string(index=False))
    # (c) without the flagged blank-slide patient
    keep = (df.patient != FLAGGED_PATIENT).to_numpy()
    full = df.full.to_numpy()
    c = pd.DataFrame([{**cidx_row(f"image (without {FLAGGED_PATIENT})", img[keep], d[keep], ev[keep]), **image_lrt(df, img, keep)},
                      {**cidx_row(f"full equal-weight (without {FLAGGED_PATIENT})", full[keep], d[keep], ev[keep])},
                      {**cidx_row(f"clinical (without {FLAGGED_PATIENT})", clin[keep], d[keep], ev[keep])}])
    c.to_csv(os.path.join(OUT, "c2c_without_flagged.csv"), index=False)
    print("\n===== C2c  without the flagged patient =====\n" + c.to_string(index=False))


# ------------------------------------------------------------------ C3 gene SHAP
def c3(df):
    import matplotlib.pyplot as plt
    pids, dur, evt, bags, G, Xc, sites = L.load_full_cohort()
    assert pids == df.patient.tolist()
    genes = [g for g in config.PAM50 if g in pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), nrows=1).columns]
    assert len(genes) == G.shape[1]
    n = len(pids); oof = np.full(n, np.nan); shap = np.zeros_like(G, dtype=float); W = []
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(np.arange(n))):
        torch.manual_seed(SEED)                                   # identical to stage_e_attmil.run(use_gene=True)
        model = e.ABMIL(gene_dim=G.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        tt, tev = torch.tensor(dur[tr]), torch.tensor(evt[tr].astype(float))
        model.train()
        for _ in range(120):
            opt.zero_grad()
            risks = torch.stack([model(torch.zeros((1, 768)), torch.tensor(G[i])) for i in tr])
            e.cox_loss(risks, tt, tev).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                oof[i] = float(model(torch.zeros((1, 768)), torch.tensor(G[i])))
        w = model.risk.weight.detach().numpy()[0, 768:]           # image part multiplies zeros: risk = w·g + b
        W.append(w)
        shap[te] = w * (G[te] - G[tr].mean(0))                    # exact SHAP for a linear model
    dev = float(np.max(np.abs(oof - df.gene_random.to_numpy())))
    log(f"C3 check: re-trained gene models vs cached gene risk, max |difference| = {dev:.2e}")
    assert dev < 1e-4
    W = np.array(W)
    res = pd.DataFrame({"gene": genes, "mean_abs_shap": np.abs(shap).mean(0), "mean_weight": W.mean(0),
                        "folds_agreeing_on_direction": [int(max((W[:, j] > 0).sum(), (W[:, j] < 0).sum())) for j in range(len(genes))]})
    res["direction"] = np.where(res.mean_weight > 0, "higher expression -> higher risk", "higher expression -> lower risk")
    res = res.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    res.round(4).to_csv(os.path.join(OUT, "c3_gene_shap.csv"), index=False)
    print("\n===== C3  gene explanations (top 15 by mean |SHAP|) =====\n" + res.head(15).round(3).to_string(index=False))
    top = res.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    mf.style(ax); ax.grid(axis="y", visible=False)
    cols = [mf.ORANGE if w > 0 else mf.BLUE for w in top.mean_weight]
    ax.barh(np.arange(len(top)), top.mean_abs_shap, color=cols, height=0.62)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([f"{g}  ({a}/5 folds)" for g, a in zip(top.gene, top.folds_agreeing_on_direction)])
    ax.set_xlabel("Mean |SHAP value| (average push on the gene risk score)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=mf.ORANGE, label="Higher expression → higher risk"),
                       Patch(color=mf.BLUE, label="Higher expression → lower risk")],
              loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Which PAM50 genes drive the gene risk score (top 15; out-of-fold SHAP)", loc="left", fontsize=10.5)
    fig.savefig(os.path.join(L.FIGDIR, "fig14_gene_shap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    os.makedirs(OUT, exist_ok=True)
    df, pids, tiles = base_data()
    if "--c1-crops" in sys.argv or "--c1-filter" in sys.argv:
        cl = N.cluster_tiles(pids, tiles)
        sub = c1_subclusters(pids, tiles, cl)
        print("\n===== C1  sub-clusters of cluster 2 =====")
        c1_attention(df, pids, cl, sub)
        if "--c1-crops" in sys.argv:
            c1_crops(df, pids, tiles, cl, sub)
        else:
            c1_filter(df, pids, tiles, cl, sub)
    else:
        c2(df, pids, tiles)
        c3(df)
    log("done -> results/fullcohort/followups/")
