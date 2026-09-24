"""
stage_c_parallel.py — Stage C download+featurize with PARALLEL streams.

GDC throttles each connection to ~0.3 MB/s, but 8 parallel connections aggregate to ~2 MB/s.
This runs a pool of workers, each downloading a slide, extracting features (same pipeline as
stage_c_local), saving to the checkpoint dir, and deleting the slide. Resume-safe.

Run:  python src/stage_c_parallel.py
"""
from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_c_local as scl  # reuse load_survival, slide_list, slide_vector, CKPT

WORKERS = 8
torch.set_num_threads(2)          # keep per-inference threads low so 8 workers don't thrash the CPU

_lock = threading.Lock()
_done = {"n": 0}


def _saved() -> int:
    return len([f for f in os.listdir(scl.CKPT) if f.endswith(".npy")])


def process(r, total):
    fp = os.path.join(scl.CKPT, f"{r.patient}.npy")
    if os.path.exists(fp):
        return
    tmp = f"/tmp/{r.file_id}.svs"
    url = f"https://api.gdc.cancer.gov/data/{r.file_id}"
    ok = False
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as resp:  # 5-min read stall tolerance
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for ch in resp.iter_content(1 << 20):
                        fh.write(ch)
            ok = True
            break
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == 3:
                print(f"  give up {r.patient}: {repr(e)[:40]}", flush=True)
    try:
        if ok:
            v = scl.slide_vector(tmp)          # same feature pipeline as local
            if v is not None:
                np.save(fp, v)
    except Exception as e:
        print(f"  featurize fail {r.patient}: {repr(e)[:50]}", flush=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    with _lock:
        _done["n"] += 1
        print(f"[{_done['n']}/{total}] {r.patient} | saved {_saved()}", flush=True)


def main():
    os.makedirs(scl.CKPT, exist_ok=True)
    surv = scl.load_survival()
    sl = scl.slide_list(surv)
    todo = [r for r in sl.itertuples()
            if not os.path.exists(os.path.join(scl.CKPT, f"{r.patient}.npy"))]
    print(f"selection {len(sl)} (dead {int(sl.event.sum())} / alive {int((~sl.event).sum())}); "
          f"already have {len(sl) - len(todo)}; downloading {len(todo)} with {WORKERS} workers", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda r: process(r, len(todo)), todo))
    print("ALL DONE. total saved:", _saved(), flush=True)


if __name__ == "__main__":
    main()
