# Pre-Registration — Analysis Plan (lock before modeling)

Pre-registering means writing down our hypotheses, methods, and success criteria
**before** we look at modeling results. It is the single most important thing that
separates real science from "I tried things until a number looked good." We commit
to this plan; if we deviate, we log the deviation and why (§11).

_Status: DRAFT — Stage A finalized before any model is trained; B–D finalized as we reach them._

## 1. Question
Can AI learn patterns from tumor gene-expression and histopathology images to predict
breast-cancer overall survival; does each modality generalize; and does combining both
improve prediction over either alone, beyond standard clinical staging?

## 2. Data / cohorts
- **Genomics — discovery/train:** METABRIC (`brca_metabric`).
- **Genomics — external validation & multi-modal cohort:** TCGA-BRCA (`brca_tcga_pan_can_atlas_2018`).
- **Imaging — warm-up:** BreakHis (benign vs malignant classification).
- **Imaging — survival & fusion:** TCGA-BRCA H&E whole-slide images (TCIA/GDC), matched to
  the same patients' gene expression.
- All public; no private or patient-identifiable data.

## 3. Endpoint
- **Primary:** Overall Survival — `OS_MONTHS` (time), `OS_STATUS` (event = DECEASED).
- **Secondary (binary):** alive vs. deceased at 5 years, restricted to patients with ≥60
  months follow-up or an event before 60 months (avoids censoring bias).
- **Imaging warm-up only:** benign vs. malignant (classification accuracy / AUC).

## 4. Pre-specified hypotheses
- **H1 (genes):** a gene-expression model predicts OS better than chance and **generalizes**
  from METABRIC to TCGA (and reverse).
- **H2 (image warm-up):** a CNN classifies benign vs. malignant on BreakHis well above chance
  — confirms the imaging pipeline works before we trust it on harder tasks.
- **H3 (image survival):** image-derived features predict OS in TCGA above chance.
- **H4 (fusion — PRIMARY):** combining gene + image features predicts OS **better than either
  modality alone** (ΔC-index > 0). *We report this result whether positive or negative — a
  well-done "fusion doesn't help here" is still an honest, publishable finding.*
- **H5 (clinical value):** the best model adds prognostic value **beyond** standard clinical
  staging (age, grade, stage, ER/HER2).

## 5. Features
- **Clinical baseline:** age, grade, stage/size, ER/HER2, nodal status (final list fixed after EDA).
- **Genes:** start from PAM50 (50 genes, literature-grounded), then a data-driven signature
  selected **only on training folds** (no leakage).
- **Images:** per-patient feature vector from tiled patches passed through a pretrained model
  and pooled (multiple-instance / attention). No hand-engineered "looks aggressive" labels.

## 6. Models (pre-specified)
1. **Clinical-only** Cox PH — the "standard of care" baseline.
2. **Genes-only** — penalized Cox (elastic net) and Random Survival Forest.
3. **Image-only** — pooled image features → Cox.
4. **Fusion** — gene + image (+ clinical) features → survival model.
- All tuning inside **nested cross-validation** on training data only.

## 7. Success criteria
- **Discrimination:** C-index with 95% CI (bootstrap), in cross-validation and external cohort.
  *Bar:* each modality's CI lower bound > 0.5 to claim signal.
- **Incremental value (key tests):** fusion vs. best single modality (H4); best model vs.
  clinical-only (H5) — ΔC-index with CI and likelihood-ratio test. **Reported regardless of sign.**
- **Generalization:** genomics C-index must hold on the external cohort within a pre-stated
  tolerance; any cross-platform drop is discussed, not hidden.

## 8. Controls against fooling ourselves
- Fixed seed (42) everywhere; strict train/validation/test separation.
- **No test data touches preprocessing, feature selection, or tuning.**
- Nested cross-validation for any tuning.
- **Negative control:** shuffle survival labels → model should score ~0.5 C-index. If not, we
  have leakage and must fix it before trusting anything.
- Patient-level splits for images (all tiles from one patient stay in one fold — no patient
  appears in both train and test).

## 9. Known limitations (stated up front)
- Retrospective, observational data → associations, not causation; not a clinical tool.
- Microarray vs. RNA-seq platform differences (batch effects) in the genomics arm.
- Gigapixel slides force us to process a **subset** → smaller image/fusion sample, wider CIs.
- Fusion analysis is **TCGA-only** (only cohort with both modalities) → weaker external
  validation for images; possible future check vs. CPTAC-BRCA.
- BreakHis validates the pipeline (classification), not the survival claim.

## 10. Compute
- Genomics: local laptop (CPU). Imaging/fusion: free cloud GPU (Google Colab / Kaggle).

## 11. Deviation log
_(Record any change from this plan, with date and reason.)_
- _2026-06-24 — Scope expanded from genomics-only to multi-modal (genes + histopathology),
  per researcher decision. Staged plan A→D adopted to keep a complete deliverable at each step._
- _2026-09-30 — Full-cohort analysis (n = 1,033, all TCGA-BRCA patients with a diagnostic slide,
  survival and PAM50 data) run with the unchanged n = 244 methods (LAB_NOTEBOOK Entry 13). The
  full model did not improve on clinical staging (ΔC +0.003). Follow-up analyses fixed in §12._
- _2026-10-02 — Amendment A (§12) written before any of its analyses was run._
- _2026-10-03 — Amendment B (§13, tissue composition / shortcut test) written before any of its
  analyses was run. GDC API down (HTTP 503) since 2026-10-01 evening: B1–B3 use only local data;
  B4 waits for GDC._
- _2026-10-04 — Amendment C (§14: fat inside cluster 2, robustness checks, gene explanations) written
  before any of its analyses was run._
- _2026-10-21 — Health check before writing found that **H4 (the original primary) was never formally
  tested on the full cohort** (Entry 13 compared every model with clinical only). It is run now exactly
  as §7 specified, before looking at it: fusion = the equal-weight genes + images score used throughout;
  best single modality = images (random 5-fold C 0.641 vs genes 0.618, Entry 13); paired-bootstrap ΔC
  (2,000 resamples, seed 42) with 95% CI, two-sided p and P(not better), plus the likelihood-ratio test
  for adding gene risk to a Cox model on image risk (1 df). Site-held-out risks as sensitivity. Reported
  whatever the result. Code: `src/stage_p_healthcheck.py h4`._
- _2026-10-21 — Amendment D (§15: other endpoints, fresh patients only, PAM50 adjustment) written
  before any of its analyses was run._
- _2026-10-21 — **Bug found and fixed: AJCC stage parsing.** `stage_h_clinical.load_clinical()` matched only
  labels written exactly "STAGE I/II/III/IV"; every sub-stage (IA, IIA, IIB, IIIA, IIIB, IIIC …) became missing
  and was median-imputed to stage I. In the 1,033: 1,007 patients ended up as stage 1, 6 as 2, 2 as 3, 18 as 4
  (correct: 172 / 591 / 234 / 18, 18 unknown — Table 1 used separate, correct code). The clinical model
  therefore had age, T, N and in effect a stage-IV flag. Fixed (sub-stage letters allowed); the parsed counts
  now equal Table 1. **Every analysis that uses the clinical variables is re-run with methods otherwise
  unchanged** (same folds, seeds, imputation, models); image and gene risks do not use stage and are not
  re-trained. Before the fix, a recomputation of every cached clinical risk column reproduced the saved values
  (max |difference| ≤ 4.4e-16; `healthcheck/clinical_recompute_verify.csv`). Pre-fix results stay in git commit
  b8b8fce and are listed next to the corrected ones in LAB_NOTEBOOK Entry 17. Amendment D is run after the fix._
- _2026-10-21 — **Clinical baseline (logged late; the change itself dates from Stage H, July).** H5 and §5 named
  age, grade, stage, ER/HER2 and nodal status. The TCGA-BRCA clinical export used here
  (cBioPortal `brca_tcga_pan_can_atlas_2018`, 41 columns, checked 2026-10-21) has **no tumor-grade and no
  ER/PR/HER2 field**, so the clinical model is age + AJCC stage + T + N (§5 allowed the final list to be fixed
  after EDA). PAM50 subtype is the closest available stand-in for receptor status and is tested as an
  adjustment in Amendment D3. Also noted: clinical medians for imputation were computed on all patients
  before the folds (outcome-free; 18 unknown stage) — a small departure from §8 "no test data touches
  preprocessing", left unchanged so results stay comparable._
- _2026-10-21 — **Tile magnification documented wrongly.** Code comments, README and Table S1 say tiles were cut
  at ~20× (downsample 2). `get_best_level_for_downsample(2)` returns level 0 on these slides, so every tile was
  read at full scan resolution (224 px at level 0, confirmed in all 789 saved coordinate files; the original
  244 used the same code). Documentation is corrected; scan magnification is checked as Amendment D5._
- _2026-10-25 — Amendment E (§16: non-TCGA foundation model and true-20× tiles) written before any of
  its analyses was run._

## 12. Amendment A — follow-up analyses (2026-10-02, written BEFORE running them)
**Why:** after Entry 13, the risk is fishing for a positive result. These analyses, and only these,
are fixed now. All use the existing out-of-fold risks in `results/fullcohort/oof_risks_full.csv`
(random 5-fold, seed 42) unless stated. First-level models are not changed. Every result is
reported whatever its direction. Code: `src/stage_m_added_value.py`.

**A1 — Independent prognostic value (primary; fulfils §7's likelihood-ratio test).**
Cox PH on all 1,033 patients. Model 0: age, AJCC stage, T, N (median-imputed, as before).
Model 1: Model 0 + image risk (z-scored). **Primary test: likelihood-ratio test, Model 1 vs Model 0,
α = 0.05**; report the image hazard ratio per 1 SD with 95% CI. Secondary: Model 2 = Model 1 + gene
risk (LRT vs Model 1). Sensitivity: site-held-out image risk instead. Proportional-hazards assumption
checked (Schoenfeld residuals). Also reported: how much of the image risk is explained by the clinical
variables (Spearman with stage; R² of a linear regression on the clinical variables).

**A2 — Learned-weight fusion (fulfils §6: any tuning inside nested CV).** Outer folds = the same
KFold(5, shuffle, seed 42) as the first-level models. Inside each outer training set, an inner 5-fold CV
(seed 42) re-trains the clinical, gene and image models to get inner out-of-fold risks; a Cox model on
[z(clinical), z(image), z(gene)] learns the weights, which are applied to the outer test patients'
first-level out-of-fold risks (z-scored over the whole out-of-fold vector, as for the equal-weight
fusion). Also images + clinical only. Primary comparison: stacked full model vs clinical, paired
bootstrap ΔC (2,000 resamples). The weights from every outer fold are reported.

**A3 — Subgroups (exploratory).** Only subgroups with ≥ 30 deaths are tested: AJCC stage II, stage III,
PAM50 LumA, PAM50 LumB (counts from Table 1, chosen on size, not on any model result). In each:
C-index of clinical, image and full model; ΔC (full − clinical) with a two-sided paired-bootstrap p,
**Holm-corrected across these 4 tests**. Stage I, stage IV, Basal, Her2 and Normal are descriptive only.

**A4 — Slide-size confound, full cohort.** C-index of slide file size alone (larger = higher risk) with
bootstrap CI; Spearman between image risk and size. If the size CI excludes 0.5, A1 is repeated with
log(size) added. File used: new 789 = the smallest diagnostic slide (as downloaded). The original 244
kept GDC's first-listed slide, so a patient with 2+ diagnostic slides is ambiguous: the primary
analysis excludes ambiguous patients; sensitivity analyses assign them their smallest, then largest slide.

**A5 — Remaining statistics.** (a) Time-dependent AUC difference, full − clinical (mean over the same
time grid, and at 1 / 3 / 5 years), paired bootstrap CI (1,000). (b) Brier difference at 3 and 5 years,
paired bootstrap CI (1,000), on the Entry 13 recalibrated predictions. (c) Decision-curve analysis at 3
and 5 years, risk-of-death thresholds 5–50% (net benefit of clinical, full model, treat-all,
treat-none); descriptive. (d) The upper CI bound of ΔC is reported as the largest improvement over
clinical that the data allow.

**No other model variants will be tried in response to these results.**

## 13. Amendment B — is the image signal tumor biology or tissue composition? (2026-10-03, written BEFORE running)
**Why:** Entry 12's heatmaps showed attention on fat / empty space for a low-risk and a missed patient,
and A1 (§12) showed the image score is an independent prognostic factor. Is that factor tumor biology,
or simply *how much fat / stroma / background each slide's 200 tiles contain* (a shortcut)? Fixed now:
analyses, primary test, decision rules. Uses the n = 1,033 features and out-of-fold risks from Entry 13.
Code: `src/stage_n_tissue.py`. Every result is reported whatever its direction; no other variants.

**B1 — Tile clustering (unsupervised; no outcomes used).** All tiles of all 1,033 patients
(≤ 200 each): standardize the 768 Phikon features, PCA to 50 components, k-means with **k = 8**
(n_init 10, seed 42). Sensitivity: k = 12. Each patient's **tissue composition** = the fraction of its
tiles in each cluster.

**B2 — Where the attention goes.** Re-train the five random-split image fold models exactly as before
(seed 42) and keep them; check they reproduce the cached out-of-fold image risks (max |difference|
< 1e-4). Each patient's attention (from the model that never saw them) is summed per cluster.
Reported: per cluster, mean share of tiles vs mean share of attention (enrichment = attention share ÷
tile share), and Spearman correlation of each cluster fraction with the image risk.

**B3 — Primary test (label-free shortcut test).** Cox model on all 1,033 patients:
clinical (age, stage, T, N) + composition (k − 1 cluster fractions; one dropped as reference), then +
image risk (z-scored). **Primary: likelihood-ratio test for adding the image risk, α = 0.05.**
If significant, the image signal is *not* explained by tissue composition. Secondary: (a) composition
alone, out-of-fold Cox C-index (same folds); (b) LRT for adding composition to clinical + image;
(c) the primary test repeated with k = 12.

**B4 — Labelled clusters + filtered model (needs the GDC API; runs when it is back).** From a fixed
random set of 30 patients (seed 42) whose diagnostic slide is ≤ 0.7 GB, re-download the slides and crop,
for each k = 8 cluster, the 10 tiles nearest the cluster centre (exact saved positions). Each cluster is
labelled from its crops as tumor / stroma / fat / background-or-artifact / other (non-expert read,
shown to the user; pathologist review requested). **Decision rule:** a cluster is "fat or background"
if ≥ 6 of its 10 crops are fat or background. The image model is then re-trained (same settings, folds,
seed) using only tiles outside those clusters (a patient with < 5 remaining tiles keeps all tiles; the
count is reported) and evaluated by out-of-fold C-index and the A1 LRT. A filtered image score that
stays significant beyond clinical would point to tumor/stroma biology; a collapse would point to a shortcut.

## 14. Amendment C — fat inside cluster 2, robustness, gene explanations (2026-10-04, written BEFORE running)
**Why:** B4 (§13) found the attention cluster (k = 8 cluster 2) is fibrous stroma at its core, but Entry 12's
fat tiles also fall in it, so a fat-specific shortcut was not ruled out. The robustness checks and gene
explanations were planned earlier (roadmap) and are fixed here too. Code: `src/stage_o_followups.py`.
Every result is reported whatever its direction; no other variants.

**C1 — Is there a fat sub-cluster inside cluster 2? (primary for this amendment)**
Recompute the B1 PCA(50) representation (same code and seed), take every cluster-2 tile (k = 8), and run
k-means **k = 4** (n_init 10, seed 42). Label each sub-cluster from the 10 tiles nearest its centre among
the same 30 B4 patients (re-downloaded, MD5-verified, cropped at saved positions, features checked
against the saved ones, slides deleted). **Rule:** a sub-cluster is "fat or background" if ≥ 6 of its 10
crops are fat or background. Report each sub-cluster's share of tiles, share of attention, enrichment and
Spearman with image risk. **If** a sub-cluster is flagged: (i) re-train the image model (same settings,
folds, seed) without the flagged tiles (patients with < 5 remaining tiles keep all; count reported);
**primary test: Cox LRT for adding this fat-free image risk to age + stage + T + N, α = 0.05**; also its
out-of-fold C-index; (ii) secondary: LRT for the original image risk beyond clinical + the patient's
fraction of flagged tiles. **If no sub-cluster is flagged**, that is the result and nothing is re-trained.

**C2 — Robustness (pre-specified sensitivity analyses).**
(a) *Fewer tiles:* each patient keeps a random 100 of their tiles (seed 42; patients with ≤ 100 keep all);
re-train the image model (same settings); report out-of-fold C-index and the A1 LRT.
(b) *Combination weights* (no re-training; weights on z-scored out-of-fold risks, clinical : image : gene):
1:1:1, 2:1:1, 3:1:1, 1:1:0, 2:1:0. C-index and paired-bootstrap ΔC vs clinical for each. All five are
reported; none is "selected".
(c) *Without the flagged blank-slide patient* (TCGA-OL-A5RZ, 131/200 identical tiles): C-index of image and
full models and the A1 LRT with that patient excluded from evaluation.

**C3 — Gene explanations (descriptive).** The gene arm is linear in the 50 PAM50 genes (its image input is
empty), so each fold model's gene weights give exact SHAP values: SHAP(patient, gene) = weight ×
(expression − mean). Re-train the five gene fold models (checked against the cached gene risks), compute
out-of-fold SHAP values, and report each gene's mean |SHAP|, direction (higher expression → higher or
lower risk) and how many of the 5 folds agree on the direction. Descriptive only — no significance claims.

## 15. Amendment D — endpoint, fresh-patient and subtype checks of A1 (2026-10-21, written BEFORE running)
**Why:** the 2026-10-21 health check raised three reviewer-level concerns about A1 (image risk beyond
age + stage + T + N): (i) TCGA's clinical-data guide (Liu et al., *Cell* 2018) advises caution with
overall survival for BRCA (short follow-up), and only 79 of the 146 deaths are recorded as "dead with
tumor"; (ii) the image method and fusion design were chosen while looking at the original 244 patients,
who hold 123 of the 146 deaths, so the full-cohort result is not a fully fresh test; (iii) tumor grade and
ER/PR/HER2 are not in this dataset, so the clinical baseline lacks them. No model is re-trained: every
check reuses existing image risks (trained on overall survival). The only new inputs are the TCGA-CDR
endpoints already in `data/raw/tcga_clinical.csv` and the PAM50 labels already used in Table 1. Event
counts below were looked up before writing this (descriptive only; no model output was looked at).
Code: `src/stage_p_healthcheck.py amendment-d`. Every result is reported whatever its direction.

**D1 — Other endpoints (primary for this amendment).** Repeat A1 exactly (Model 0 = age, AJCC stage, T, N,
same median imputation; Model 1 = Model 0 + z-scored random-5-fold image risk; LRT 1 df; HR per SD) with
(a) **progression-free interval** (`PFS_MONTHS`, `PFS_STATUS` = "1:PROGRESSION"; n = 1,033, 139 events) —
**primary: LRT for the image risk, α = 0.05**; (b) disease-specific survival (`DSS_MONTHS`, `DSS_STATUS` =
"1:DEAD WITH TUMOR"; patients with missing status excluded; n = 1,014, 79 events). Also reported for each
endpoint: C-index of the clinical risk and the image risk (descriptive). Durations of 0 are kept.

**D2 — Patients not used to choose the methods (the 789 new; 23 deaths, 58 progression events).**
A1 refit within the 789 (Model 0 refit on them; 5 covariates with 23 deaths is thin and is stated as such),
using (a) the **frozen image risk** (model trained on the 244 only, `frozen_preds.csv`; z-scored within the
789) — overall survival and progression-free interval; (b) the random-5-fold out-of-fold image risk
restricted to the 789 — overall survival. Interpretation is by the HR and its 95% CI; with 23 deaths a
non-significant p is **not** evidence of no effect, and a significant one is not required.

**D3 — PAM50 subtype as a stand-in for receptor status.** A1 with PAM50 subtype (dummy-coded, LumA =
reference; LumB, Basal, Her2, Normal, Unknown) added to both Model 0 and Model 1; LRT for the image risk
(overall survival). Also reported: LRT for subtype added to Model 0 (descriptive).

**D4 — Plain-language summary (descriptive).** Share of patients who died, and Kaplan-Meier 5-year overall
survival, for below- vs above-median random-5-fold image risk; log-rank p and HR for the median split.

**D5 — Scan magnification (added 2026-10-21, before any D result was seen).** The health check found that
every tile was read at level 0 (full scan resolution; `get_best_level_for_downsample(2)` returns level 0 on
these slides), not ~20× as documented, so 40×-scanned slides gave ~0.25 µm/px tiles and 20×-scanned slides
~0.5 µm/px. Scanner magnification (Aperio `AppMag`) and µm/px are read from each exact slide's header on GDC
(a few small byte-range requests; `healthcheck/scan_magnification.csv`); ambiguous or unreadable slides =
"unknown". Reported: counts and deaths per magnification; image risk by magnification (difference in means,
Mann-Whitney p; descriptive); **A1 with magnification group added to both models (categorical; LRT for the
image risk)**; image C-index within each group with ≥ 30 deaths (descriptive). Same interpretation rule as
D1(a) and D3.

**Correction before any D result was seen (2026-10-21):** the first run stopped on missing data before
printing anything: TCGA-OL-A66H has a PFI/DSS status but no PFI/DSS time. Patients missing an endpoint's
time are excluded from that endpoint (PFI n = 1,032, 139 events; DSS n = 1,013, 79 events; the frozen
PFI analysis in D2 likewise). Event counts are unchanged.

**Interpretation rule (fixed now):** D1–D3 and D5 are sensitivity checks of A1, not new primaries; no multiplicity
correction. If the image HR stays > 1 with a 95% CI excluding 1 under D1(a), D3 and D5, A1 is described as
robust to these concerns. If not, the paper reports A1 as sensitive to that choice, prominently. D2 is
reported by HR and CI only. **No other variants will be tried in response to these results.**

## 16. Amendment E — does the image signal depend on Phikon's TCGA pretraining or on tile resolution? (2026-10-25, written BEFORE running)
**Why:** the health check (Entry 17) found (i) Phikon was pretrained, without labels, on TCGA-BRCA diagnostic
slides that almost certainly include ≥ 1,031 of our 1,033 patients, and (ii) tiles were cut at full scan
resolution (~0.25 µm/px on the 900 40× scans), not the ~0.5 µm/px (20×) Phikon was trained on. Both can only be
tested by re-reading the slides. Code: `src/stage_q_encoders.py`. Every result is reported whatever its direction.

**Patients.** The 1,026 patients whose exact slide is known (manifest `slide_status == exact`); the 7 ambiguous
patients are excluded. Slides are re-downloaded from GDC and must match GDC's size and MD5; a patient whose slide
cannot be obtained or verified is excluded and listed. All arms below — including the Phikon reference — are
re-run on exactly the same final patient set, with KFold(5, shuffle, seed 42) over the patients sorted by ID.

**Tile sets.** *Native* (= the tiles analysed so far): for the 789 new patients, the saved level-0 coordinates;
for the original 237, regenerated with the unchanged tiling code (same grid, seeded order, tissue filter, ≤ 200
tiles). **Check:** Phikon re-embedding of every regenerated native tile set must match the saved features
(every tile cosine ≥ 0.999, same tile count); a patient failing it is excluded from all arms and listed.
*20×*: the same grid/seed/filter/≤ 200 rule applied at a target of 0.5 µm/px — a level-0 region of
round(224 × 0.5 / mpp) px resized to 224 × 224 (PIL Lanczos), mpp from the slide header (`openslide.mpp-x`);
patients without mpp are excluded from the 20× arms only. Tile pixels are cached (uint8) so encoders run later.

**Encoders.** Phikon (unchanged revision) and **Prov-GigaPath** tile encoder (`prov-gigapath/prov-gigapath`,
pretrained on Providence Health slides, not TCGA), with the preprocessing documented on its model card
(resize 256 bicubic → centre-crop 224 → ImageNet normalisation); 1,536-number embedding per tile.

**Arms (image model = attention-MIL with every setting unchanged; only the input size follows the encoder):**
- **E1 (primary): GigaPath on the native tiles** — isolates the encoder / pretraining data.
- E0 (reference): Phikon on the native tiles (saved features) — the current model on the same patients.
- E2a: Phikon on 20× tiles — isolates tile resolution. E2b: GigaPath on 20× tiles.

**Tests.** For each arm: out-of-fold C-index (random 5-fold, and site-held-out GroupKFold) with bootstrap CI;
**A1 likelihood-ratio test for the image risk beyond age + stage + T + N (corrected coding), HR per SD with 95% CI
— E1's is the primary test of this amendment, α = 0.05**; the same test for progression-free interval; Spearman
correlation of each arm's risk with E0's.
**Interpretation rule (fixed now):** if E1's image HR > 1 with a 95% CI excluding 1, the image signal is described
as not dependent on a TCGA-pretrained encoder. If not, the paper states prominently that the Phikon result may
depend on its exposure to these slides (or on the encoder). E2a/E2b are reported as resolution checks, same rule.
**No other encoders, settings or tile rules will be tried in response to these results.**
