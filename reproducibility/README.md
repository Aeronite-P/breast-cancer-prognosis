# Reproducibility package

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
