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

---

## `src/model.py`  (Stage A)
**What:** The foundation result — trains and cross-validates survival models and proves the
pipeline isn't cheating. Builds four models: **clinical-only**, **genes-only (PAM50)**,
**clinical+genes**, and a **shuffled-label negative control**.

**Reads:** `data/processed/metabric_merged.csv`.

**Produces:**
- Console: a C-index table (with per-fold mean ± std).
- `results/metrics_stageA.csv` — the numbers.
- `results/figures/stageA_cindex.png` — bar chart vs. the 0.5 random line.

**Key ideas in the code (worth understanding):**
- **C-index (concordance):** for pairs of patients, did the one who died sooner get the higher
  predicted risk? 0.5 = random, 1.0 = perfect.
- **Out-of-fold scoring:** every patient is scored by a model that never saw them in training
  (`KFold`) — an honest estimate of real performance.
- **No leakage:** imputation + scaling are fit on the *training* fold only, then applied to the
  test fold (`make_preprocessor` inside `cv_cindex`).
- **Negative control:** `shuffle_labels=True` scrambles outcomes; a correct pipeline then
  scores ~0.5. If it scored high, we'd have a leak.
- **Honest comparison:** the clinical baseline uses *truly clinical* variables only (no
  molecular subtypes), so "genes vs. clinical" is a fair fight.

**Run:** `python src/model.py`

---

## `notebooks/stage_b_breakhis_colab.ipynb`  (Stage B — runs on Colab)
**What:** A transfer-learning CNN that classifies benign vs. malignant breast histopathology
images (BreakHis), with a leakage-safe patient-level split. **Runs on a free Google Colab
GPU**, not your laptop. See [`STAGE_B_COLAB.md`](STAGE_B_COLAB.md) for step-by-step instructions.

---

## `src/validate_external.py`  (Stage A rigor follow-ups)
**What:** Two honest tests of the gene model. (1) **External validation** — train on one cohort
and test on the *other* (METABRIC ↔ TCGA, microarray vs RNA-seq) to prove the signal generalizes.
(2) **Significance test** — bootstrap a 95% CI for whether clinical+genes really beats clinical.

**Reads:** `data/processed/metabric_merged.csv`; auto-downloads TCGA on first run.

**Produces:**
- Console: external-validation C-indices (both directions) + the ΔC-index with 95% CI and a
  SIGNIFICANT / NOT-significant verdict.
- `results/metrics_external.csv` and `results/figures/external_validation.png`.

**Key idea:** a model that works only on its training cohort learned noise; one that still ranks
survival in an independent cohort learned real biology. The bootstrap keeps us honest about
whether small C-index gaps are real or just noise.

**Run:** `python src/validate_external.py`  (reuses helpers in `model.py`)

---

## `src/make_figures.py`  (publication figures + robustness checks)
**What:** Computes every out-of-fold risk score once (images, genes, clinical) for four settings —
random 5-fold, site-held-out, shuffled labels (negative control), and 4 extra random seeds — caches
them, then draws all paper figures and Table 1 from that cache. Every plotted number comes from code.

**Reads:** `data/stage_c_tiles/` (Phikon tile embeddings), `data/processed/tcga_merged.csv`.

**Produces:**
- `results/oof_risks.csv` — the cache (all risk scores; delete it or pass `--recompute` to retrain)
- `results/figures/fig2_cindex_forest.png` — C-index ± 95% CI, random split vs unseen hospitals
- `results/figures/fig3_km_risk_groups.png` — Kaplan-Meier high vs low risk, clinical vs multi-modal
- `results/figures/fig4_time_auc.png` — discrimination over 1–8 years
- `results/figures/fig5_complementarity.png` — image risk vs gene risk
- `results/table1_cohort.md`, `metrics_final.csv`, `delta_vs_clinical.csv`, `robustness.csv`,
  `km_risk_groups.csv`, `time_auc.csv`

**Key idea:** computing once and plotting from a cache keeps the figures, the tables, and the lab
notebook consistent with each other. The patient order matches `stage_h_clinical.py`, so the
random-split numbers reproduce Entries 9–10 exactly.

**Run:** `python src/make_figures.py` (~8 min first time; seconds after, from cache)
