"""
stage_q_encoders.py — Amendment E (PREREGISTRATION.md §16): does the image signal depend on Phikon's TCGA
pretraining, or on tile resolution?

  tiles      Phase 1 (no Hugging Face login needed). For each of the 1,026 patients with a known slide:
             re-download the slide (resume + GDC size/MD5 check, as in stage_c_pathology), cut two tile sets
               native  = the tiles analysed so far (saved coordinates for the 789 new patients; the unchanged
                         tiling code for the original 237), checked by re-embedding with Phikon (cosine ≥ 0.999)
               20x     = same grid / seed / tissue filter / ≤ 200 rule at 0.5 µm/px (mpp from the slide header)
             cache both as uint8 pixels, embed the 20x set with Phikon (arm E2a), delete the slide.
  gigapath   Phase 2 (needs `huggingface-cli login` + accepted Prov-GigaPath terms): embed both cached tile
             sets with the Prov-GigaPath tile encoder (arms E1, E2b).
  analyze    Phase 3: attention-MIL (unchanged settings) per arm on the patients that passed the native-tile
             check; out-of-fold C-index (random, site-held-out), A1 LRT (overall survival and PFI), Spearman
             with the Phikon reference. First checks that its training loop reproduces the cached risks.

Tiles, coordinates and features go to data/stage_q_*/ (git-ignored); the per-patient checks to
results/fullcohort/encoders/. Every step can be stopped and re-run: finished patients are skipped.

Run:  nohup python src/stage_q_encoders.py tiles > data/logs/stage_q_tiles.log 2>&1 &
      python src/stage_q_encoders.py tiles --limit 2          (test on two patients)
      python src/stage_q_encoders.py gigapath
      python src/stage_q_encoders.py analyze
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                       # noqa: E402

D = config.DATA_DIR
NATIVE, X20 = os.path.join(D, "stage_q_tiles", "native"), os.path.join(D, "stage_q_tiles", "20x")
FEATS = os.path.join(D, "stage_q_feats")                    # <encoder>_<tileset>/<patient>.npy
DONE = os.path.join(D, "stage_q_tiles", "done")             # one empty file per finished patient (written last)
TMP = os.path.join(D, "stage_q_tmp")                         # partial slides survive restarts (resume)
OUT = os.path.join(config.RESULTS_DIR, "fullcohort", "encoders")
TARGET_MPP = 0.5
WORKERS = 12                                                # GDC gives ~0.27 MB/s per connection
_wlock = threading.Lock()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slides():
    """The 1,026 patients whose exact diagnostic slide is known (manifest), as rows usable by C._download."""
    m = pd.read_csv(os.path.join(os.path.dirname(config.RESULTS_DIR), "reproducibility", "cohort_manifest.csv"))
    m = m[m.slide_status == "exact"].sort_values("patient")
    Row = namedtuple("Row", "patient file_id size md5 group")
    return [Row(r.patient, r.slide_file_id, int(r.slide_size), r.slide_md5, r.group) for r in m.itertuples()]


# ------------------------------------------------------------------ tile sets
def native_tiles(C, s, patient):
    """The tiles of the original analysis: saved level-0 coordinates if present, else the unchanged tiling code."""
    saved = os.path.join(C.COORDS_DIR, f"{patient}.npy")
    if os.path.exists(saved):
        xy = np.load(saved)
        return [s.read_region((int(x), int(y)), 0, (int(z), int(z))).convert("RGB") for x, y, z in xy], xy
    lvl = s.get_best_level_for_downsample(C.DOWNSAMPLE)          # same lines as stage_c_pathology.tile_features
    W, H = s.level_dimensions[lvl]; ds = s.level_downsamples[lvl]
    coords = [(x, y) for y in range(0, H - C.TILE, C.TILE) for x in range(0, W - C.TILE, C.TILE)]
    np.random.RandomState(0).shuffle(coords)
    tiles, kept = [], []
    for (x, y) in coords:
        reg = s.read_region((int(x * ds), int(y * ds)), lvl, (C.TILE, C.TILE)).convert("RGB")
        if C.is_tissue(reg):
            tiles.append(reg)
            kept.append((int(x * ds), int(y * ds), int(C.TILE * ds)))
        if len(tiles) >= C.MAX_TILES:
            break
    return tiles, np.array(kept, dtype=np.int32)


def tiles_20x(C, s):
    """Same grid / seeded order / tissue filter / ≤ 200 rule, at 0.5 µm/px. None if the slide has no mpp."""
    mpp = float(s.properties.get("openslide.mpp-x", "nan"))
    if not np.isfinite(mpp) or mpp <= 0:
        return None, None, mpp
    size = int(round(C.TILE * TARGET_MPP / mpp))                 # level-0 pixels covered by one 224-px tile
    W, H = s.dimensions
    coords = [(x, y) for y in range(0, H - size, size) for x in range(0, W - size, size)]
    np.random.RandomState(0).shuffle(coords)
    tiles, kept = [], []
    for (x, y) in coords:
        reg = s.read_region((x, y), 0, (size, size)).convert("RGB").resize((C.TILE, C.TILE), Image.LANCZOS)
        if C.is_tissue(reg):
            tiles.append(reg)
            kept.append((x, y, size))
        if len(tiles) >= C.MAX_TILES:
            break
    return tiles, np.array(kept, dtype=np.int32), mpp


def phikon(C, tiles):
    """Phikon CLS embeddings, exactly as stage_c_pathology.tile_features computes them."""
    import torch
    feats = []
    with C._flock:
        for j in range(0, len(tiles), 32):
            inp = C._proc(images=tiles[j:j + 32], return_tensors="pt")
            with torch.no_grad():
                feats.append(C._model(**inp).last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(feats).astype("float16")


def saved_phikon(C, patient):
    for d in (C.TILES_DIR, C.TILES_FULL_DIR):
        f = os.path.join(d, f"{patient}.npy")
        if os.path.exists(f):
            return np.load(f)
    return None


def cosine_rows(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def append_row(name, row):
    path = os.path.join(OUT, name)
    with _wlock:
        pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def process(C, r, k, total):
    if os.path.exists(os.path.join(DONE, r.patient)):
        return
    tmp = os.path.join(TMP, f"{r.file_id}.svs")
    ok = C._download(r, tmp)
    if not ok:
        append_row("slides_unavailable.csv", {"patient": r.patient, "file_id": r.file_id, "reason": "download/MD5 failed"})
        log(f"[{k}/{total}] {r.patient} | download failed (kept partial file for the next run)")
        return
    import openslide
    try:
        s = openslide.OpenSlide(tmp)
        nat, nat_xy = native_tiles(C, s, r.patient)
        t20, xy20, mpp = tiles_20x(C, s)
        s.close()

        f_nat = phikon(C, nat)                                   # check: do we have exactly the analysed tiles?
        ref = saved_phikon(C, r.patient)
        same_n = ref is not None and len(ref) == len(f_nat)
        cos = cosine_rows(f_nat, ref) if same_n else np.array([np.nan])
        append_row("native_tile_check.csv", {"patient": r.patient, "group": r.group, "n_tiles": len(nat),
                                             "n_saved": None if ref is None else len(ref),
                                             "min_cosine": float(np.nanmin(cos)), "mean_cosine": float(np.nanmean(cos)),
                                             "pass": bool(same_n and np.nanmin(cos) >= 0.999)})
        np.save(os.path.join(NATIVE, f"{r.patient}_xy.npy"), nat_xy)
        np.save(os.path.join(NATIVE, f"{r.patient}.npy"), np.stack([np.asarray(t) for t in nat]))
        if t20 is not None and len(t20):
            np.save(os.path.join(X20, f"{r.patient}_xy.npy"), xy20)
            np.save(os.path.join(X20, f"{r.patient}.npy"), np.stack([np.asarray(t) for t in t20]))
            np.save(os.path.join(FEATS, "phikon_20x", f"{r.patient}.npy"), phikon(C, t20))
        append_row("tile_sets.csv", {"patient": r.patient, "mpp": mpp, "n_native": len(nat),
                                     "n_20x": 0 if t20 is None else len(t20)})
        open(os.path.join(DONE, r.patient), "w").close()          # written last = patient finished
        os.remove(tmp)
        log(f"[{k}/{total}] {r.patient} | native {len(nat)} tiles (min cos {np.nanmin(cos):.4f}) | "
            f"20x {0 if t20 is None else len(t20)} tiles (mpp {mpp:.3f})")
    except Exception as e:                                       # keep the slide; a re-run retries this patient
        log(f"[{k}/{total}] {r.patient} | tiling FAILED: {repr(e)[:80]}")


def run_tiles():
    import stage_c_pathology as C                               # loads Phikon (same revision, cached)
    for d in (NATIVE, X20, os.path.join(FEATS, "phikon_20x"), DONE, TMP, OUT):
        os.makedirs(d, exist_ok=True)
    todo = [r for r in slides() if not os.path.exists(os.path.join(DONE, r.patient))]
    if "--only" in sys.argv:                                   # testing: e.g. --only TCGA-A2-A0CQ
        todo = [r for r in todo if r.patient in sys.argv[sys.argv.index("--only") + 1].split(",")]
    if "--limit" in sys.argv:
        todo = todo[:int(sys.argv[sys.argv.index("--limit") + 1])]
    log(f"Amendment E tiles: {len(todo)} patients to go (~{sum(r.size for r in todo) / 1e9:.0f} GB), {WORKERS} workers")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda kr: process(C, kr[1], kr[0] + 1, len(todo)), enumerate(todo)))
    left = sum(not os.path.exists(os.path.join(DONE, r.patient)) for r in slides())
    log("ALL DONE" if left == 0 else f"FINISHED WITH {left} PATIENTS LEFT (re-run to retry)")


# ------------------------------------------------------------------ phase 2: Prov-GigaPath embeddings
def run_gigapath():
    """Embed the cached native and 20x tiles with the Prov-GigaPath tile encoder (model-card preprocessing)."""
    import timm
    import torch
    from torchvision import transforms
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).eval().to(dev)
    tf = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    for tileset, src in (("native", NATIVE), ("20x", X20)):
        dst = os.path.join(FEATS, f"gigapath_{tileset}")
        os.makedirs(dst, exist_ok=True)
        pids = sorted(f[:-4] for f in os.listdir(src) if f.endswith(".npy") and not f.endswith("_xy.npy"))
        todo = [p for p in pids if not os.path.exists(os.path.join(dst, f"{p}.npy"))]
        log(f"GigaPath {tileset}: {len(todo)} of {len(pids)} patients to embed on {dev}")
        for k, p in enumerate(todo, 1):
            px = np.load(os.path.join(src, f"{p}.npy"))
            feats = []
            for j in range(0, len(px), 32):
                x = torch.stack([tf(Image.fromarray(t)) for t in px[j:j + 32]]).to(dev)
                with torch.no_grad():
                    feats.append(model(x).float().cpu().numpy())
            np.save(os.path.join(dst, f"{p}.npy"), np.concatenate(feats).astype("float16"))
            if k % 25 == 0:
                log(f"  {tileset}: {k}/{len(todo)}")
    log("GigaPath embeddings done")


# ------------------------------------------------------------------ phase 3: the Amendment E tests
ARMS = {"E0 Phikon, native tiles (reference)": ("phikon", "native"),
        "E1 GigaPath, native tiles (PRIMARY)": ("gigapath", "native"),
        "E2a Phikon, 20x tiles": ("phikon", "20x"),
        "E2b GigaPath, 20x tiles": ("gigapath", "20x")}


def feature_file(encoder, tileset, p):
    if (encoder, tileset) == ("phikon", "native"):                # the saved features of the main analysis
        for d in ("stage_c_tiles", "stage_c_tiles_full"):
            f = os.path.join(D, d, f"{p}.npy")
            if os.path.exists(f):
                return f
        return None
    f = os.path.join(FEATS, f"{encoder}_{tileset}", f"{p}.npy")
    return f if os.path.exists(f) else None


def oof_mil(bags, dur, evt, folds, epochs=120, lr=1e-3):
    """stage_e_attmil.run with given folds and the encoder's embedding size (same loop, settings, seeding)."""
    import torch
    import stage_e_attmil as e
    oof = np.full(len(bags), np.nan)
    for tr, te in folds:
        torch.manual_seed(e.SEED)
        model = e.ABMIL(d=bags[0].shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        tt, ev = torch.tensor(dur[tr]), torch.tensor(evt[tr].astype(float))
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            e.cox_loss(torch.stack([model(bags[i]) for i in tr]), tt, ev).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                oof[i] = float(model(bags[i]))
    return oof


def check_replica():
    """oof_mil must reproduce the cached full-cohort image risk (fold 1 of the 1,033) before it is trusted."""
    import torch
    from sklearn.model_selection import KFold
    cache = pd.read_csv(os.path.join(config.RESULTS_DIR, "fullcohort", "oof_risks_full.csv"))
    bags = [torch.from_numpy(np.load(feature_file("phikon", "native", p)).astype(np.float32)) for p in cache.patient]
    folds = list(KFold(5, shuffle=True, random_state=config.RANDOM_SEED).split(np.arange(len(bags))))[:1]
    r = oof_mil(bags, cache.time.to_numpy(float), cache.event.to_numpy(bool), folds)
    te = folds[0][1]
    dev = float(np.max(np.abs(r[te] - cache.img_random.to_numpy()[te])))
    log(f"check: oof_mil vs cached full-cohort image risk (fold 1), max |difference| = {dev:.2e}")
    assert dev < 1e-4, "oof_mil does not reproduce stage_e_attmil.run"


def run_analyze():
    import torch
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold, KFold
    import make_figures as mf
    import stage_e_attmil as e
    import stage_h_clinical as h
    import stage_p_healthcheck as P

    check_replica()
    chk = pd.read_csv(os.path.join(OUT, "native_tile_check.csv")).drop_duplicates("patient", keep="last")
    passed = sorted(chk.patient[chk["pass"].astype(bool)])
    failed = sorted(set(chk.patient) - set(passed))
    log(f"{len(passed)} patients passed the native-tile check; excluded: {failed}")
    surv, clin = e.load_survival(), h.load_clinical()
    cache_path = os.path.join(OUT, "e_oof_risks.csv")
    cache = pd.read_csv(cache_path) if os.path.exists(cache_path) else pd.DataFrame(columns=["patient", "arm", "split", "risk"])
    rows, risks = [], {}
    for arm, (enc, tset) in ARMS.items():
        pids = [p for p in passed if feature_file(enc, tset, p)]
        if len(pids) < 100:
            log(f"{arm}: only {len(pids)} patients with features; skipped"); continue
        d, ev = surv.loc[pids, "time"].to_numpy(float), surv.loc[pids, "event"].to_numpy(bool)
        sites = np.array([p.split("-")[1] for p in pids])
        for split in ("random", "site"):
            got = cache[(cache.arm == arm) & (cache.split == split)]
            if len(got) == len(pids) and list(got.patient) == pids:
                risks[(arm, split)] = got.risk.to_numpy(float); continue
            bags = [torch.from_numpy(np.load(feature_file(enc, tset, p)).astype(np.float32)) for p in pids]
            folds = (list(KFold(5, shuffle=True, random_state=config.RANDOM_SEED).split(np.arange(len(pids))))
                     if split == "random" else list(GroupKFold(5).split(np.arange(len(pids)), groups=sites)))
            log(f"{arm}, {split}: training on {len(pids)} patients (embedding size {bags[0].shape[1]})")
            r = oof_mil(bags, d, ev, folds)
            risks[(arm, split)] = r
            cache = pd.concat([cache[~((cache.arm == arm) & (cache.split == split))],
                               pd.DataFrame({"patient": pids, "arm": arm, "split": split, "risk": r})])
            cache.to_csv(cache_path, index=False)
        r = risks[(arm, "random")]
        C = clin.loc[pids]
        base = C.fillna(C.median()).reset_index(drop=True).assign(time=d, event=ev)
        a1 = P.lrt_image(base, r)
        ep = P.endpoints(pd.DataFrame({"patient": pids}))
        k = ep.pfi_known.to_numpy()
        pfi = P.lrt_image(base[k].assign(time=ep.pfi_time.to_numpy()[k], event=ep.pfi_event.to_numpy()[k]), r[k])
        c_r, c_s = mf.ci(r, d, ev), mf.ci(risks[(arm, "site")], d, ev)
        ref = cache[(cache.arm == "E0 Phikon, native tiles (reference)") & (cache.split == "random")].set_index("patient").risk
        rho = spearmanr(r, ref.reindex(pids)).correlation if ref.reindex(pids).notna().all() else np.nan
        rows.append({"arm": arm, "n": len(pids), "deaths": int(ev.sum()),
                     "C_random": round(c_r[0], 3), "C_lo": round(c_r[1], 3), "C_hi": round(c_r[2], 3),
                     "C_site": round(c_s[0], 3), "C_site_lo": round(c_s[1], 3), "C_site_hi": round(c_s[2], 3),
                     "A1_HR_per_SD": a1["HR_per_SD"], "A1_HR_lo": a1["HR_lo"], "A1_HR_hi": a1["HR_hi"], "A1_p": a1["p_LRT"],
                     "PFI_HR_per_SD": pfi["HR_per_SD"], "PFI_HR_lo": pfi["HR_lo"], "PFI_HR_hi": pfi["HR_hi"],
                     "PFI_p": pfi["p_LRT"], "spearman_vs_E0": round(rho, 3)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "e_results.csv"), index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "tiles":
        run_tiles()
    elif what == "gigapath":
        run_gigapath()
    elif what == "analyze":
        run_analyze()
    else:
        sys.exit("usage: python src/stage_q_encoders.py tiles [--limit N] | gigapath | analyze")
