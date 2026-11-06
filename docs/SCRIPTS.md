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

---

## `src/stage_k_heatmaps.py`  (attention heatmaps)
**What:** Shows which tissue tiles the image model weighted most for a few example patients, using
each patient's out-of-fold model (retrained with seed 42 and checked against the cached risks).
Re-downloads the example slides, recomputes tile positions (checked against saved features), saves
thumbnails and top tiles, deletes the slides, and draws Fig 6.

**Produces:** `results/figures/fig6_attention_heatmaps.png` (clean journal version — definitions go in
the caption) and `results/figures/fig6b_attention_heatmaps_explained.png` (presentation version: a
"how to read this" key defining every term, a ✓/✗ verdict per patient, and a short note on what the
top tiles show — labelled as a non-expert read pending pathologist review); per-patient cache in
`data/stage_k_heatmaps/` (gitignored).

**Run:** `python src/stage_k_heatmaps.py` (~1 hour, mostly downloading; re-plotting from cache takes seconds)

---

## `src/stage_l_fullcohort.py`  (full-cohort analysis, n = 1,033)
**What:** Repeats every n = 244 analysis, unchanged, on all 1,033 TCGA-BRCA patients with a diagnostic
slide, survival and PAM50 data (146 deaths, 39 hospitals), by re-using `make_figures.py` with the cohort
and output folders swapped. Adds: a frozen-model test (train on the original 244, test on the 789 new
patients), 3- and 5-year calibration with Brier scores, a 244-vs-1,033 comparison, and the negative
control repeated over 10 shuffles.

**Reads:** `data/stage_c_tiles/` + `data/stage_c_tiles_full/`, `data/processed/tcga_merged.csv`.

**Produces:** `results/fullcohort/` (oof_risks_full.csv cache, metrics_final.csv, delta_vs_clinical.csv,
robustness.csv, km_risk_groups.csv, time_auc.csv, table1_cohort.md, frozen_validation.csv,
calibration_brier.csv, calibration_quintiles.csv, negative_control_repeats.csv,
comparison_244_vs_1033.csv) and `results/figures/fullcohort/` (fig2–5, fig7 calibration, fig8 244 vs 1,033).
The n = 244 results in `results/` are never touched.

**Run:** `python src/stage_l_fullcohort.py` (~45 min first time, mostly attention-MIL training; seconds after, from cache)

---

## `src/stage_m_added_value.py`  (does the image model add anything beyond staging?)
**What:** Runs exactly the five follow-up analyses fixed in advance in PREREGISTRATION.md §12:
A1 independent prognostic value (Cox + likelihood-ratio test), A2 learned-weight fusion with nested
cross-validation, A3 exploratory subgroups (Holm-corrected), A4 slide-size confound (needs the GDC API;
skipped with a message if GDC is down), A5 bootstrap CIs for time-AUC and Brier differences plus
decision curves. Includes safety checks (fresh retrain = cached risks; Brier = Entry 13).

**Reads:** `results/fullcohort/oof_risks_full.csv` (from `stage_l_fullcohort.py`), the tile folders (A2),
the GDC API (A4, cached to `results/fullcohort/gdc_dx_slides.csv`).

**Produces:** `results/fullcohort/added_value/` (a1_*, a2_*, a3_subgroups.csv, a4_*, a5_*) and
`results/figures/fullcohort/fig9_decision_curves.png`, `fig10_subgroups.png`.

**Run:** `python src/stage_m_added_value.py` (~20 min first time for A2's nested CV; ~3 min after, from cache)

---

## `src/stage_n_tissue.py`  (is the image signal just tissue composition?)
**What:** Amendment B (PREREGISTRATION.md §13). B1 clusters all ~206,000 tiles by appearance (standardized
Phikon features → PCA 50 → k-means k = 8, sensitivity k = 12; no outcomes used). B2 re-trains the five
out-of-fold image models (checked against the cached risks) and measures each cluster's share of tiles vs
share of attention. B3 (primary) tests whether the image risk still predicts survival after adjusting for
age, stage, T, N and tissue composition (Cox likelihood-ratio test). Also a descriptive count of hospitals
per cluster. B4 (cluster labels from re-downloaded tiles; filtered re-train) is added when GDC is back.

**Reads:** the tile folders, `results/fullcohort/oof_risks_full.csv`.
**Produces:** `results/fullcohort/tissue/` (composition_k8/k12, b1_hospitals_per_cluster_k8, b2_attention_by_cluster_k8/k12,
b3_primary_tests, b3_composition_only_cindex, b3_cox_clinical_composition_image) and
`results/figures/fullcohort/fig11_attention_by_cluster.png`; per-tile clusters and attention in `data/stage_n_tissue/` (gitignored).

**Run:** `python src/stage_n_tissue.py` (~4 min first time, mostly re-training; seconds after, from cache)

---

## `src/stage_o_followups.py`  (Amendment C: fat, robustness, gene explanations)
**What:** PREREGISTRATION.md §14. **C1** splits tile cluster 2 into 4 sub-clusters (in B1's exact PCA space —
checked by distance-to-centre and adjusted Rand index), measures attention per sub-cluster, crops 10 example
tiles per sub-cluster from re-downloaded slides (`--c1-crops`, features checked, slides deleted) and, if a
sub-cluster is labelled fat/background in `c1_labels.csv`, re-trains the image model without it and runs the
primary LRT (`--c1-filter`). **C2** robustness: 100 tiles per patient, five fixed fusion weightings, results
without the blank-slide patient. **C3** exact SHAP values for the linear gene model.
**Produces:** `results/fullcohort/followups/` and `results/figures/fullcohort/fig13_cluster2_subclusters.png`,
`fig14_gene_shap.png`.
**Run:** `python src/stage_o_followups.py` (C2 + C3, ~5 min) · `--c1-crops` (~40 min, downloads) · `--c1-filter` (~3 min)

---

## `src/stage_p_healthcheck.py`  (pre-writing health check, 2026-10-21/22)
**What:** Gaps found by the audit before writing (LAB_NOTEBOOK Entry 17; PREREGISTRATION §11 and §15).
`h4` — the original primary hypothesis on the full cohort (genes + images vs images alone: paired-bootstrap ΔC
and LRT). `clinical-verify` / `clinical-apply` — after the AJCC stage-parsing fix, recompute ONLY the clinical
columns of every cached risk file with the original folds and imputation (verify reproduced the old values to
≤ 4.4e-16 before the fix). `scan-magnification` — reads each slide's scanner magnification and µm/px from its
TIFF header on GDC with a few byte-range requests (no full download; failed lookups are retried on re-run).
`amendment-d` — A1 with progression-free interval and disease-specific survival (D1), in the 789 patients not
used to design the methods (D2), with PAM50 subtype (D3), a plain-language median split (D4), and with scan
magnification (D5).
**Produces:** `results/fullcohort/healthcheck/`.
**Run:** `python src/stage_p_healthcheck.py h4 | clinical-verify | clinical-apply | scan-magnification | amendment-d`
(seconds each; `scan-magnification` ~1 h)

---

## `src/stage_q_encoders.py`  (Amendment E: non-TCGA foundation model, true-20× tiles)
**What:** PREREGISTRATION.md §16. `tiles` re-downloads each of the 1,026 known slides (resume + MD5), caches the
native tile set (checked against the saved Phikon features) and a 0.5 µm/px tile set, embeds the 20× set with
Phikon, deletes the slide. `gigapath` embeds both cached sets with Prov-GigaPath (needs `huggingface-cli login`
and its accepted terms). `analyze` re-trains the image model per arm and runs the A1 tests.
**Produces:** `data/stage_q_tiles/`, `data/stage_q_feats/` (git-ignored), `results/fullcohort/encoders/`.
**Run:** `nohup caffeinate -i python src/stage_q_encoders.py tiles > data/logs/stage_q_tiles.log 2>&1 &` (~4 days)
· `python src/stage_q_encoders.py gigapath` (hours) · `python src/stage_q_encoders.py analyze` (~2–3 h)

---

## `src/make_paper_figures.py`  (Figure 1 + supplement)
**What:** Draws Figure 1 (pipeline), Figure S1 (cohort flow from 1,082 to 1,033 patients — every count computed
from the data), Table S1 (all model/analysis settings) and Table S2 (every sensitivity result, read from the
result files by name). Tables are saved as figures and as Markdown in `results/supplement/`.
**Run:** `python src/make_paper_figures.py` (seconds)

---

## `src/make_manifest.py`  (reproducibility package)
**What:** Builds `reproducibility/`: exact package versions (`requirements-lock.txt`), environment and
Phikon model revision (`environment.txt`), a per-patient manifest of the 1,033 patients with GDC slide
file id / size / MD5 and SHA-256 checksums of every saved feature file (`cohort_manifest.csv`), SHA-256
checksums of every results file (`results_checksums.sha256`), and a README explaining how to verify.
Slide columns read the GDC listing cached by `stage_m_added_value.py`; they say "pending GDC" until then.

**Run:** `python src/make_manifest.py` (seconds). Verify results: `shasum -a 256 -c reproducibility/results_checksums.sha256`.

---

## `src/watch_download.py`  (live progress screen for the full-cohort download)
**What:** A read-only dashboard in the terminal while `stage_c_pathology.py --all` runs. Refreshes every
3 seconds: patients saved (of 789 new / 1,033 total), a progress bar for each slide being downloaded
(MB done / total, speed averaged over 30 s), total speed, recent pace (patients/hour) with a rough
days-left estimate, and the last log lines.

**Reads:** `/tmp/*.svs` file sizes, `data/stage_c_tiles_full/`, `data/logs/stage_c_full.log`, and one
GDC request for slide sizes. **Writes nothing** and never touches the job — safe to open/close any time.

**Run:** `python src/watch_download.py` (Ctrl+C to quit; the download keeps running)
