"""
stage_c_pathology.py — SOTA imaging upgrade: per-tile PATHOLOGY features (Phikon).

Instead of a generic ImageNet ResNet + mean pooling, this uses **Phikon** (Owkin's pathology
foundation model, trained on TCGA tissue) to embed each tile, and SAVES THE FULL PER-TILE
MATRIX (n_tiles x 768) per patient — so we can pool with attention-MIL later and never
re-download. Same size-matched selection + 8-way parallel download as before.

Run:  python src/stage_c_pathology.py
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

TILES_DIR = os.path.join(config.DATA_DIR, "stage_c_tiles")   # per-tile feature matrices
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
    return len([f for f in os.listdir(TILES_DIR) if f.endswith(".npy")]) if os.path.isdir(TILES_DIR) else 0


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
    tiles = []
    for (x, y) in coords:
        reg = s.read_region((int(x * ds), int(y * ds)), lvl, (TILE, TILE)).convert("RGB")
        if is_tissue(reg):
            tiles.append(reg)
        if len(tiles) >= MAX_TILES:
            break
    s.close()
    if not tiles:
        return None
    feats = []
    with _flock:
        for j in range(0, len(tiles), 32):
            inp = _proc(images=tiles[j:j + 32], return_tensors="pt")
            if _gpu:
                inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                out = _model(**inp)
            feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())   # CLS token per tile
    return np.concatenate(feats).astype("float16")


def process(r, total):
    fp = os.path.join(TILES_DIR, f"{r.patient}.npy")
    if os.path.exists(fp):
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
            v = tile_features(tmp)
            if v is not None:
                np.save(fp, v)
    except Exception as e:
        print(f"  featurize fail {r.patient}: {repr(e)[:50]}", flush=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    with _dlock:
        _done["n"] += 1
        print(f"[{_done['n']}/{total}] {r.patient} | saved {_saved()}", flush=True)


def main():
    os.makedirs(TILES_DIR, exist_ok=True)
    surv = scl.load_survival()
    sl = scl.slide_list(surv)
    todo = [r for r in sl.itertuples()
            if not os.path.exists(os.path.join(TILES_DIR, f"{r.patient}.npy"))]
    print(f"Phikon re-extract: {len(sl)} patients, {len(sl) - len(todo)} done, "
          f"{len(todo)} to go with {WORKERS} workers (GPU={_gpu})", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda r: process(r, len(todo)), todo))
    print("ALL DONE. saved:", _saved(), flush=True)


if __name__ == "__main__":
    main()
