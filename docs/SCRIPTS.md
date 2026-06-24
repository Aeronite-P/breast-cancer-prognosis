# Script Documentation

Plain-English reference for every script in this project. Each entry says what the script
does, what it reads, what it produces, and how to run it. Code itself is also commented.

> Setup once: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

---

## `config.py`
**What:** Central settings used by everything else — so there's one place to change things
and the whole pipeline stays reproducible.
**Contains:** the cBioPortal API URL; all file paths; the random seed (`42`, fixed for
reproducibility); the two cohort definitions (METABRIC, TCGA-BRCA) with their study IDs and
expression-profile IDs; and the PAM50 gene panel.
**You edit this when:** changing cancer type/cohort, the gene list, or the seed.

---

## `src/fetch_data.py`
**What:** Downloads real breast-cancer data from the public cBioPortal REST API and saves
tidy CSV files. This is the lightweight, always-works data path (no login, reproducible
anywhere with internet).

**Reads:** nothing local — pulls from `https://www.cbioportal.org/api`.

**Produces:**
- `data/raw/metabric_clinical.csv` — one row per patient, columns = clinical attributes
  (age, subtype, ER/HER2 status, survival, …).
- `data/raw/metabric_expr.csv` — one row per patient, columns = genes, values = expression
  z-scores.
- `data/processed/metabric_merged.csv` — the two joined into one modeling table.

**Key functions:**
- `fetch_clinical(study_id)` — patient clinical data → wide table.
- `map_symbols_to_entrez(symbols)` — converts gene names (e.g. `ESR1`) to the numeric IDs
  the API needs.
- `fetch_expression(study_id, profile_id, symbols)` — expression matrix (patients × genes).
- `list_profiles(study_id)` — utility to discover the exact expression-profile ID for a
  study (use this to verify the TCGA profile ID before Week 3).
- `build(...)` — runs all of the above and writes the CSVs.

**Run:** `python src/fetch_data.py`
**Note:** For the *full* transcriptome (~20k genes) later, download the study tarball from
<https://www.cbioportal.org/datasets> — faster than the API for very large matrices.

---

## `src/eda.py`
**What:** "Exploratory Data Analysis" — the first look at the data, and our first real
result. It does **not** build a model; it checks the data is sound and shows the survival
signal honestly.

**Reads:** `data/processed/metabric_merged.csv` (so run `fetch_data.py` first).

**Produces:**
- Console: cohort summary (patients, deaths, censored, median follow-up).
- `results/figures/metabric_km_by_subtype.png` — Kaplan–Meier survival curves by molecular
  subtype, with a log-rank test p-value in the title.

**Key functions:**
- `load_survival(prefix)` — loads the merged table and builds clean `time`/`event` columns
  for survival analysis.
- `main()` — prints the summary, draws the figure, runs the log-rank test.

**Run:** `python src/eda.py`
**What the result means:** A small log-rank p-value means the subtypes have genuinely
different survival. It is a descriptive finding and a sanity check — **not** a prediction
accuracy. Predictive modeling comes in Week 2+.
