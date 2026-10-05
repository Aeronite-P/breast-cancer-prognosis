"""
make_manifest.py — build the reproducibility package in reproducibility/ (local; nothing is uploaded).

Writes:
  reproducibility/requirements-lock.txt  exact versions of every installed Python package (pip freeze)
  reproducibility/environment.txt        Python, OS, CPU type, key library versions, Phikon model revision
  reproducibility/cohort_manifest.csv    one row per patient (1,033): which group, which GDC slide file
                                         (id, size, GDC MD5 checksum), and SHA-256 checksums of the saved
                                         feature and tile-position files
  reproducibility/results_checksums.sha256  SHA-256 of every file in results/ (verify with shasum -c)
  reproducibility/README.md              what each file is and how to check it

Slide file ids / MD5s come from the GDC API listing cached by stage_m_added_value.py
(results/fullcohort/gdc_dx_slides.csv). If GDC is unreachable those columns say "pending GDC";
re-run this script once GDC is back.

Run:  python src/make_manifest.py
"""
from __future__ import annotations

import glob
import hashlib
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import config  # noqa: E402

OUT = os.path.join(ROOT, "reproducibility")
GDC_LIST = os.path.join(config.RESULTS_DIR, "fullcohort", "gdc_dx_slides.csv")
NEW_LIST = os.path.join(config.RESULTS_DIR, "fullcohort", "frozen_preds.csv")       # the 789 new patients
COHORT = os.path.join(config.RESULTS_DIR, "fullcohort", "oof_risks_full.csv")       # the 1,033 analysed
PHIKON_REF = os.path.expanduser("~/.cache/huggingface/hub/models--owkin--phikon/refs/main")


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    pip = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    with open(os.path.join(OUT, "requirements-lock.txt"), "w") as f:
        f.write("# Exact package versions used for the n = 244 and n = 1,033 analyses (pip freeze)\n" + pip)
    import numpy, sklearn, sksurv, lifelines, torch, transformers, openslide
    phikon = open(PHIKON_REF).read().strip() if os.path.exists(PHIKON_REF) else "unknown"
    lines = [f"python {sys.version.split()[0]}",
             f"os {platform.platform()}", f"machine {platform.machine()}",
             f"numpy {numpy.__version__}", f"scikit-learn {sklearn.__version__}",
             f"scikit-survival {sksurv.__version__}", f"lifelines {lifelines.__version__}",
             f"torch {torch.__version__}", f"transformers {transformers.__version__}",
             f"openslide-python {openslide.__version__}", f"openslide (C library) {openslide.__library_version__}",
             f"phikon model (huggingface owkin/phikon) revision {phikon}",
             f"random seed {config.RANDOM_SEED}"]
    with open(os.path.join(OUT, "environment.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cohort_manifest():
    pids = pd.read_csv(COHORT, usecols=["patient"]).patient.tolist()
    new = set(pd.read_csv(NEW_LIST, usecols=["patient"]).patient)
    gdc = pd.read_csv(GDC_LIST) if os.path.exists(GDC_LIST) else None
    rows = []
    for p in pids:
        grp = "new (full-cohort run)" if p in new else "original (size-matched n = 244)"
        feat = os.path.join(config.DATA_DIR, "stage_c_tiles_full" if p in new else "stage_c_tiles", f"{p}.npy")
        coords = os.path.join(config.DATA_DIR, "stage_c_coords", f"{p}.npy")
        row = {"patient": p, "group": grp, "n_tiles": int(np.load(feat, mmap_mode="r").shape[0]),
               "features_file": os.path.relpath(feat, ROOT), "features_sha256": sha256(feat),
               "coords_sha256": sha256(coords) if os.path.exists(coords) else "not saved (original run)"}
        if gdc is None:
            row.update({"slide_status": "pending GDC", "slide_file_id": "", "slide_size": "", "slide_md5": ""})
        else:
            s = gdc[gdc.patient == p].sort_values("size")
            if len(s) == 0:
                row.update({"slide_status": "not listed on GDC", "slide_file_id": "", "slide_size": "", "slide_md5": ""})
            elif p in new or len(s) == 1:
                # new run used the smallest diagnostic slide; a single-slide original is unambiguous
                r = s.iloc[0]
                row.update({"slide_status": "exact", "slide_file_id": r.file_id, "slide_size": int(r["size"]), "slide_md5": r.md5})
            else:
                row.update({"slide_status": f"ambiguous ({len(s)} diagnostic slides; original run kept GDC's first-listed)",
                            "slide_file_id": ";".join(s.file_id), "slide_size": ";".join(map(str, s["size"])),
                            "slide_md5": ";".join(s.md5)})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "cohort_manifest.csv"), index=False)
    print(f"cohort manifest: {len(df)} patients; slide status: {df.slide_status.value_counts().to_dict()}")


def results_checksums():
    files = sorted(f for f in glob.glob(os.path.join(config.RESULTS_DIR, "**", "*"), recursive=True)   # glob skips hidden
                   if os.path.isfile(f) and not os.path.basename(f).startswith("."))                       # files like .DS_Store
    with open(os.path.join(OUT, "results_checksums.sha256"), "w") as f:
        for path in files:
            f.write(f"{sha256(path)}  {os.path.relpath(path, ROOT)}\n")
    print(f"results checksums: {len(files)} files")


README = """# Reproducibility package

Everything needed to check that the results in this repository are exactly the ones reported.

| File | What it is |
|---|---|
| `requirements-lock.txt` | Exact version of every Python package used (`pip freeze`). Install with `pip install -r reproducibility/requirements-lock.txt`. |
| `environment.txt` | Python, operating system, CPU type, key library versions, the Phikon model revision, the random seed. |
| `cohort_manifest.csv` | The 1,033 patients analysed: group (original 244 or new 789), the GDC diagnostic-slide file used (id, size, GDC's MD5 checksum), and SHA-256 checksums of each saved feature file and tile-position file. |
| `results_checksums.sha256` | SHA-256 checksum of every file in `results/`. |

## How to check
- **Results unchanged:** from the repository root, run `shasum -a 256 -c reproducibility/results_checksums.sha256`. Every line should say `OK`.
- **Slides:** each slide downloaded for the full-cohort run was verified against GDC's size and MD5 at download time; `slide_md5` lets anyone re-verify a slide downloaded from `https://api.gdc.cancer.gov/data/<slide_file_id>`.
- **Features:** `features_sha256` identifies the exact feature files used. Re-extracting features on different hardware or library versions can change the last digits of the (float16) numbers, so a fresh extraction is checked by re-running the analysis, not by matching these checksums.

## Notes
- For 244 original patients the download kept GDC's first-listed diagnostic slide. When a patient has 2+ diagnostic slides the exact file cannot be recovered; `slide_status` marks those rows `ambiguous` and lists every candidate.
- Data files themselves are not in the repository (they are large and re-downloadable from GDC); this package lets anyone confirm they are working with the same inputs.
- Rebuild this package: `python src/make_manifest.py`.
"""


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    environment()
    cohort_manifest()
    results_checksums()
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(README)
    print(f"done -> {os.path.relpath(OUT, ROOT)}/")
