"""
stage_c_pathology.py — SOTA imaging upgrade: per-tile PATHOLOGY features (Phikon).

Instead of a generic ImageNet ResNet + mean pooling, this uses **Phikon** (Owkin's pathology
foundation model, trained on TCGA tissue) to embed each tile, and SAVES THE FULL PER-TILE
MATRIX (n_tiles x 768) per patient — so we can pool with attention-MIL later and never
re-download. Same size-matched selection + 8-way parallel download as before.

Run:  python src/stage_c_pathology.py                 (size-matched 260-patient selection)
      python src/stage_c_pathology.py --all           (FULL unselected TCGA-BRCA cohort)
      python src/stage_c_pathology.py --all --dry-run (print the full-cohort plan, download nothing)

--all mode: every patient with a diagnostic slide, survival, and complete PAM50 genes (one slide
per patient, the smallest if several), in a seeded RANDOM order so any partial run is a fair
random sample. New features go to data/stage_c_tiles_full/ (the original 244 in stage_c_tiles/
are reused, never re-downloaded, and stay untouched so the n=244 results remain reproducible).
Tile positions are saved to data/stage_c_coords/ so attention heatmaps need no re-download.
"""
from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
import openslide
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config          # noqa: E402
import stage_c_local as scl  # reuse load_survival, slide_list

TILES_DIR = os.path.join(config.DATA_DIR, "stage_c_tiles")   # per-tile feature matrices (n=244 cohort)
TILES_FULL_DIR = os.path.join(config.DATA_DIR, "stage_c_tiles_full")  # new patients, full cohort
COORDS_DIR = os.path.join(config.DATA_DIR, "stage_c_coords")  # level-0 (x, y, size) per saved tile
FULL = "--all" in sys.argv
DOWNSAMPLE = 2        # ~20x on 40x-scanned slides (Phikon's training regime)
TILE = 224
MAX_TILES = 200
WORKERS = 8
torch.set_num_threads(2)

_proc = AutoImageProcessor.from_pretrained("owkin/phikon")
_model = AutoModel.from_pretrained("owkin/phikon").eval()
_gpu = torch.cuda.is_available()
if _gpu:
    _model = _model.cuda()
_flock = threading.Lock()   # serialize model inference (fast vs. download; avoids CPU thrash)
_dlock = threading.Lock()
_done = {"n": 0}


def _saved() -> int:
    d = TILES_FULL_DIR if FULL else TILES_DIR
    return len([f for f in os.listdir(d) if f.endswith(".npy")]) if os.path.isdir(d) else 0


def _have(patient) -> bool:
    """Already featurized? (full mode also counts the original 244, so nothing is re-downloaded)"""
    dirs = [TILES_DIR, TILES_FULL_DIR] if FULL else [TILES_DIR]
    return any(os.path.exists(os.path.join(d, f"{patient}.npy")) for d in dirs)


def slide_list_all(surv):
    """Full unselected cohort: every patient with a diagnostic slide + survival + complete PAM50."""
    import json
    import pandas as pd
    c = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId")
    genes = [g for g in config.PAM50 if g in c.columns]
    gene_ok = set(c.index[c[genes].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)])
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}]}
    params = {"filters": json.dumps(filt), "fields": "file_id,file_size,cases.submitter_id",
              "format": "JSON", "size": "20000"}
    hits = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=120).json()["data"]["hits"]
    df = pd.DataFrame([{"patient": h["cases"][0]["submitter_id"], "file_id": h["file_id"],
                        "size": h.get("file_size", 0)} for h in hits])
    df = df[df.patient.isin(surv.index) & df.patient.isin(gene_ok)]
    df = df.sort_values("size").drop_duplicates("patient")          # one slide per patient: smallest
    df = df.merge(surv[["event"]], left_on="patient", right_index=True)
    return df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)  # random order


def is_tissue(im) -> bool:
    a = np.asarray(im.convert("HSV"))
    return a[:, :, 1].mean() > 25 and a[:, :, 2].mean() < 235


def tile_features(path):
    """Return (n_tiles, 768) float16 matrix of Phikon per-tile embeddings, or None."""
    s = openslide.OpenSlide(path)
    lvl = s.get_best_level_for_downsample(DOWNSAMPLE)
    W, H = s.level_dimensions[lvl]; ds = s.level_downsamples[lvl]
    coords = [(x, y) for y in range(0, H - TILE, TILE) for x in range(0, W - TILE, TILE)]
    np.random.RandomState(0).shuffle(coords)
    tiles, kept = [], []
    for (x, y) in coords:
        reg = s.read_region((int(x * ds), int(y * ds)), lvl, (TILE, TILE)).convert("RGB")
        if is_tissue(reg):
            tiles.append(reg)
            kept.append((int(x * ds), int(y * ds), int(TILE * ds)))   # level-0 position of this tile
        if len(tiles) >= MAX_TILES:
            break
    s.close()
    if not tiles:
        return None, None
    feats = []
    with _flock:
        for j in range(0, len(tiles), 32):
            inp = _proc(images=tiles[j:j + 32], return_tensors="pt")
            if _gpu:
                inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                out = _model(**inp)
            feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())   # CLS token per tile
    return np.concatenate(feats).astype("float16"), np.array(kept, dtype=np.int32)


def process(r, total):
    fp = os.path.join(TILES_FULL_DIR if FULL else TILES_DIR, f"{r.patient}.npy")
    if _have(r.patient):
        return
    tmp = f"/tmp/{r.file_id}.svs"
    url = f"https://api.gdc.cancer.gov/data/{r.file_id}"
    ok = False
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for ch in resp.iter_content(1 << 20):
                        fh.write(ch)
            ok = True
            break
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
    try:
        if ok:
            v, xy = tile_features(tmp)
            if v is not None:
                np.save(os.path.join(COORDS_DIR, f"{r.patient}.npy"), xy)   # coords first, features last
                np.save(fp, v)                                               # (features = "done" marker)
    except Exception as e:
        print(f"  featurize fail {r.patient}: {repr(e)[:50]}", flush=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    with _dlock:
        _done["n"] += 1
        print(f"[{_done['n']}/{total}] {r.patient} | saved {_saved()}", flush=True)


def main():
    for d in (TILES_DIR, TILES_FULL_DIR, COORDS_DIR):
        os.makedirs(d, exist_ok=True)
    surv = scl.load_survival()
    sl = slide_list_all(surv) if FULL else scl.slide_list(surv)
    todo = [r for r in sl.itertuples() if not _have(r.patient)]
    left_gb = sum(r.size for r in todo) / 1e9
    print(f"Phikon {'FULL cohort' if FULL else 'size-matched'}: {len(sl)} patients "
          f"({int(sl.event.sum())} deaths), {len(sl) - len(todo)} done, {len(todo)} to go "
          f"(~{left_gb:.0f} GB) with {WORKERS} workers (GPU={_gpu})", flush=True)
    if "--dry-run" in sys.argv:
        return
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda r: process(r, len(todo)), todo))
    print("ALL DONE. saved:", _saved(), flush=True)


if __name__ == "__main__":
    main()
