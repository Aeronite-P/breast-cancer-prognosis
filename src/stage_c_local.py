"""
stage_c_local.py — Stage C (imaging survival) run LOCALLY on CPU. Resumable.

Downloads TCGA-BRCA diagnostic slides one at a time, tiles them, extracts features with a
pretrained ResNet50, averages to one vector per patient, and (in 'model' mode) fits a Cox
survival model on those vectors. Each patient's features are checkpointed to disk, so the
run resumes if interrupted.

Usage:
  python src/stage_c_local.py extract [N]   # download + featurize up to N patients (default 60)
  python src/stage_c_local.py model         # fit survival model on whatever is done so far
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import torch
import openslide
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

CKPT = os.path.join(config.DATA_DIR, "stage_c_matched")   # size-matched, unbiased sample
PATCHES = 80
TILE = 224
DOWNSAMPLE = 8
DEATH_TARGET = 9999          # disabled: the size-matched slide list below is already bounded
DEAD_N = 130                 # cases: (nearly) all deceased patients with slides — full power
ALIVE_N = 130                # controls: size-matched alive patients (1:1 nearest-neighbor)

_w = ResNet50_Weights.IMAGENET1K_V2
_net = resnet50(weights=_w); _net.fc = torch.nn.Identity(); _net.eval()
_prep = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(_w.transforms().mean, _w.transforms().std)])


def load_survival() -> pd.DataFrame:
    c = pd.read_csv(os.path.join(config.PROCESSED_DIR, "tcga_merged.csv"), index_col="patientId")
    s = pd.DataFrame(index=c.index)
    s["time"] = pd.to_numeric(c["OS_MONTHS"], errors="coerce")
    s["event"] = c["OS_STATUS"].astype(str).str.startswith("1")
    s = s.dropna(subset=["time"])
    return s[s["time"] > 0]


def slide_list(surv: pd.DataFrame) -> pd.DataFrame:
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}]}
    params = {"filters": json.dumps(filt), "fields": "file_id,file_size,cases.submitter_id",
              "format": "JSON", "size": "20000"}
    hits = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=120).json()["data"]["hits"]
    rows = [{"patient": h["cases"][0]["submitter_id"], "file_id": h["file_id"],
             "size": h.get("file_size", 0)}
            for h in hits if h["cases"][0]["submitter_id"] in surv.index]
    df = pd.DataFrame(rows).drop_duplicates("patient")
    df = df.merge(surv[["event"]], left_on="patient", right_index=True, how="inner")
    # SIZE-MATCHED case/control to kill the slide-size confound:
    # take the smallest-slide deceased patients (cases), then only alive patients whose slides
    # fall in the SAME size band (controls). Now slide size can't separate the two groups.
    dead = df[df["event"]].sort_values("size").head(DEAD_N).copy()   # smallest-slide cases
    pool = df[~df["event"]].copy()
    picks, used = [], set()
    for s in dead["size"]:                       # nearest-neighbor: match each case to a same-size control
        cand = pool[~pool["patient"].isin(used)]
        j = (cand["size"] - s).abs().idxmin()
        used.add(pool.loc[j, "patient"]); picks.append(j)
    alive = pool.loc[picks]
    return pd.concat([dead, alive]).sort_values("size").reset_index(drop=True)


def is_tissue(im) -> bool:
    a = np.asarray(im.convert("HSV"))
    return a[:, :, 1].mean() > 25 and a[:, :, 2].mean() < 235


def slide_vector(path: str):
    s = openslide.OpenSlide(path)
    lvl = s.get_best_level_for_downsample(DOWNSAMPLE)
    W, H = s.level_dimensions[lvl]; ds = s.level_downsamples[lvl]
    coords = [(x, y) for y in range(0, H - TILE, TILE) for x in range(0, W - TILE, TILE)]
    np.random.RandomState(0).shuffle(coords)
    tiles = []
    for (x, y) in coords:
        reg = s.read_region((int(x * ds), int(y * ds)), lvl, (TILE, TILE)).convert("RGB")
        if is_tissue(reg):
            tiles.append(_prep(reg))
        if len(tiles) >= PATCHES:
            break
    s.close()
    if not tiles:
        return None
    out = []
    with torch.no_grad():
        for j in range(0, len(tiles), 32):          # chunk to keep CPU memory sane
            out.append(_net(torch.stack(tiles[j:j + 32])).numpy())
    return np.concatenate(out).mean(0)


def n_saved() -> int:
    return len([f for f in os.listdir(CKPT) if f.endswith(".npy")]) if os.path.isdir(CKPT) else 0


def extract(n_target: int):
    os.makedirs(CKPT, exist_ok=True)
    surv = load_survival()
    sl = slide_list(surv).head(n_target)
    print(f"target {len(sl)} patients | already saved {n_saved()}", flush=True)
    for i, r in enumerate(sl.itertuples(), 1):
        fp = os.path.join(CKPT, f"{r.patient}.npy")
        if os.path.exists(fp):
            continue
        tmp = f"/tmp/{r.file_id}.svs"
        if os.path.exists(tmp):
            os.remove(tmp)
        url = f"https://api.gdc.cancer.gov/data/{r.file_id}"
        ok = False
        for attempt in range(6):                       # resume-capable retries for big slides
            try:
                pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
                headers = {"Range": f"bytes={pos}-"} if pos else {}
                resp = requests.get(url, stream=True, timeout=(30, 120), headers=headers)
                if pos and resp.status_code == 200:    # server ignored Range -> restart clean
                    pos = 0
                with open(tmp, "ab" if pos else "wb") as fh:
                    for ch in resp.iter_content(1 << 20):
                        fh.write(ch)
                resp.close()
                ok = True
                break
            except Exception as e:
                got = os.path.getsize(tmp) / 1e6 if os.path.exists(tmp) else 0
                print(f"  dl retry {attempt + 1}/6 {r.patient} (have {got:.0f}MB): {repr(e)[:35]}", flush=True)
                time.sleep(5 * (attempt + 1))          # backoff, keep partial file to resume
        try:
            if ok:
                v = slide_vector(tmp)
                if v is not None:
                    np.save(fp, v)
        except Exception as e:
            print("  featurize fail", r.patient, repr(e)[:60], flush=True)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        saved_pids = [f[:-4] for f in os.listdir(CKPT) if f.endswith(".npy")]
        n_dead = int(surv["event"].reindex(saved_pids).fillna(False).sum())
        print(f"{i}/{len(sl)} processed | saved {len(saved_pids)} | deaths {n_dead} "
              f"| slide {r.size/1e6:.0f}MB", flush=True)
        if n_dead >= DEATH_TARGET:
            print("DEATH TARGET REACHED — stopping.", flush=True)
            break
    print("EXTRACT DONE. saved:", n_saved(), flush=True)


def model():
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    surv = load_survival()
    feats = {f[:-4]: np.load(os.path.join(CKPT, f))
             for f in os.listdir(CKPT) if f.endswith(".npy") and f[:-4] in surv.index}
    pids = list(feats)
    if not pids:
        print("no features yet"); return
    import warnings
    X = np.nan_to_num(np.vstack([feats[p] for p in pids]).astype(float))
    X = X[:, np.argsort(X.var(axis=0))[::-1][:256]]        # keep top-256 most-variable dims (cut noise)
    sub = surv.loc[pids]; dur = sub["time"].to_numpy(float); evt = sub["event"].to_numpy(bool)
    print(f"modeling on {len(pids)} patients, {int(evt.sum())} deaths", flush=True)
    if len(pids) < 25 or evt.sum() < 5:
        print("not enough patients/deaths yet for a stable estimate."); return
    kf = KFold(5, shuffle=True, random_state=42); oof = np.full(len(pids), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in kf.split(X):
            pipe = make_pipeline(StandardScaler(),
                                 PCA(n_components=min(8, len(tr) - 2), svd_solver="full"))
            Ztr = pipe.fit_transform(X[tr]); Zte = pipe.transform(X[te])
            m = CoxPHSurvivalAnalysis(alpha=10.0).fit(Ztr, Surv.from_arrays(event=evt[tr], time=dur[tr]))
            oof[te] = m.predict(Zte)
    if not np.isfinite(oof).all():
        print("Predictions non-finite -> unstable, not trustworthy at this N."); return
    c = concordance_index_censored(evt, dur, oof)[0]
    rng = np.random.RandomState(42); bs = []                # bootstrap CI = honest uncertainty
    for _ in range(1000):
        s = rng.choice(np.arange(len(pids)), len(pids), replace=True)
        if evt[s].sum() >= 3:
            bs.append(concordance_index_censored(evt[s], dur[s], oof[s])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\nStage C image-only survival C-index: {c:.3f}  (95% CI [{lo:.2f}, {hi:.2f}])", flush=True)
    print("(0.5 = chance. Preliminary — small N, wide CI.)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if mode == "model":
        model()
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        extract(n)
