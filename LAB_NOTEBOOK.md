# Lab Notebook

A running, honest log of the project. **Every major step gets an entry** in the same
four-part format, so anyone (including future-you, a science-fair judge, or a college
interviewer) can follow exactly what was done and why — and so we never quietly move the
goalposts.

> Format for each entry:
> - **What was done** — the concrete actions/code.
> - **Why it matters** — the scientific reason.
> - **Assumptions** — what we took for granted (and might be wrong).
> - **Limitations** — what this step does *not* show; honest caveats.

---

## Entry 1 — 2026-06-24 · Week 1: Foundation, data, first result

**What was done**
- Defined the research question and locked the analysis plan in
  [`README.md`](README.md) and [`PREREGISTRATION.md`](PREREGISTRATION.md) *before* any modeling.
- Built `src/fetch_data.py` and pulled real data from the public cBioPortal API:
  METABRIC = **2,509** patients (clinical), **1,980** with matched gene-expression
  (50 PAM50 genes) + survival → merged modeling table `data/processed/metabric_merged.csv`.
- Built `src/eda.py`: produced a cohort summary and a Kaplan–Meier survival curve
  stratified by molecular subtype → `results/figures/metabric_km_by_subtype.png`.

**Key numbers (real, reproducible):** 1,980 patients; 1,143 deaths, 837 alive; median
follow-up 116 months. Survival differs by subtype, **log-rank p < 0.001**, with Luminal A
surviving longest — the ordering expected from breast-cancer biology.

**Why it matters**
- Pre-registering protects the project's integrity (no fishing for good-looking numbers).
- Getting clean, real, public data with a working reproducible pipeline is the foundation
  everything else stands on.
- The Kaplan–Meier result is a *sanity check*: because the known-biology ordering shows up,
  we have evidence the data and code are correct before we build ML on top.

**Assumptions**
- `OS_STATUS` values beginning with `"1"` mean a death event; others are censored (alive
  or lost to follow-up). This matches cBioPortal's coding (`1:DECEASED` / `0:LIVING`).
- The PAM50 gene panel is a reasonable, literature-grounded starting feature set.
- Patient ID = sample ID mapping in METABRIC is 1:1 (verified via the samples endpoint).

**Limitations**
- This is *only* a descriptive result — no model, no prediction yet. A significant log-rank
  test shows subtypes differ; it says nothing about predictive accuracy.
- We used 50 genes, not the full transcriptome (that comes later).
- METABRIC is a single, retrospective, observational cohort; no causal or clinical claims.
- 528 patients lack a survival status and are dropped from survival analysis — we should
  later check those dropped patients aren't systematically different (selection bias check).

---

## Entry 2 — 2026-06-24 · Scope decision: go multi-modal (genes + images)

**What was done**
- Decided to expand the project from genomics-only to **multi-modal**: predict survival from
  tumor **gene-expression** AND **histopathology images**, and test whether combining them
  beats either alone.
- Verified data feasibility before committing: confirmed TCGA-BRCA has 3,111 public H&E
  whole-slide images for ~1,098 patients matched to gene expression + survival (TCIA/GDC),
  and that BreakHis (9,109 pre-cropped images) exists as a tractable warm-up dataset.
- Adopted a **staged, de-risked plan (A→D)** and updated `README.md`, `PREREGISTRATION.md`
  (hypotheses H1–H5, new limitations, deviation log), and `docs/DATA_SOURCES.md`.

**Why it matters**
- Multi-modal prognosis is an established, high-impact research direction that almost no
  high-schooler attempts — it raises the project's novelty and college/publication value.
- Staging guarantees a complete deliverable at every step: even if the hard imaging stages
  stall, Stage A (genomics) is a finished study and Stage B (image classifier) stands alone.

**Assumptions**
- A free cloud GPU (Colab/Kaggle) is available for the imaging stages.
- A subset of TCGA slides will be enough to demonstrate signal (we cannot process all of them).
- TCGA patient IDs link slides to gene-expression records (to be verified in Stage C).

**Limitations**
- The fusion analysis will be TCGA-only → weaker external validation than the genomics arm.
- Gigapixel slides force a subset → smaller image/fusion sample, wider confidence intervals.
- No imaging code has been written or run yet — this entry records a *plan*, not a result.

---

## Entry 3 — 2026-06-24 · Stage A complete + Stage B prepared

**What was done**
- Built `src/model.py` and ran cross-validated (5-fold, seed 42) Cox survival models on
  METABRIC (1,979 patients, 1,143 deaths): clinical-only, genes-only (PAM50), clinical+genes,
  plus a **shuffled-label negative control**. Saved `results/metrics_stageA.csv` and
  `results/figures/stageA_cindex.png`.
- Prepared Stage B: a ready-to-run Colab notebook `notebooks/stage_b_breakhis_colab.ipynb`
  (benign-vs-malignant CNN, patient-level split) + guide `docs/STAGE_B_COLAB.md`.

**Results (out-of-fold C-index; 0.5 = chance):**
| Model | C-index |
|---|---|
| Clinical only | 0.662 |
| Genes only (PAM50) | 0.599 |
| Clinical + Genes | 0.666 |
| Genes — shuffled (control) | 0.512 |

**Why it matters**
- We now have a finished, defensible foundation result.
- The control landing at ~0.51 is concrete proof there's **no data leakage** — the single most
  common way ML papers fool themselves.
- The honest finding (PAM50 adds ~nothing beyond clinical here) is itself informative and
  motivates the imaging arm and a fuller gene signature.

**Assumptions**
- Clinical baseline = age, positive lymph nodes, NPI, ER (IHC), HER2 (SNP6), menopausal state
  — truly clinical, no molecular subtypes (keeps the genes-vs-clinical comparison fair).
- Ridge-penalized Cox (penalizer 0.1, l1_ratio 0); 5-fold CV; median impute + standardize.

**Limitations**
- The clinical+genes vs clinical difference (0.666 vs 0.662) is **within cross-validation
  noise** — we have NOT yet run a formal significance test (bootstrap CI on ΔC-index). Do not
  claim genes add value. (Planned next.)
- Only 50 genes (PAM50); the full transcriptome or a data-driven signature may do better.
- METABRIC only — TCGA external validation (H1) still pending.
- Linear Cox model; Random Survival Forest not yet tried.

---

## Entry 4 — 2026-06-24 · External validation + significance test (Stage A rigor)

**What was done**
- Built `src/validate_external.py`. Pulled TCGA-BRCA (1,069 patients with PAM50 + survival;
  151 deaths). Ran two tests on the PAM50 gene model:
  1. **External validation** — train on one cohort, test on the other (cross-platform).
  2. **Significance test** — bootstrap 95% CI for clinical+genes minus clinical (METABRIC).
- Saved `results/metrics_external.csv` and `results/figures/external_validation.png`.

**Results**
| Test (PAM50 gene model) | C-index |
|---|---|
| METABRIC internal (CV) | 0.599 |
| TCGA internal (CV) | 0.621 |
| **Train METABRIC → test TCGA (external)** | **0.652** |
| **Train TCGA → test METABRIC (external)** | **0.572** |

Significance: clinical 0.662 vs clinical+genes 0.666; ΔC = **+0.005, 95% CI [−0.004, +0.015]
→ NOT significant** (interval includes 0).

**Why it matters**
- **H1 supported:** the gene model trained on one cohort/platform still ranks survival in a
  totally separate cohort on a different platform (external C-index 0.57–0.65, all > 0.5). That
  cross-cohort generalization is the project's strongest credibility signal — the model learned
  real, transferable biology, not dataset-specific noise.
- **Honest negative on incremental value:** the 50-gene PAM50 panel does NOT significantly
  improve on standard clinical staging in METABRIC. We report this straight — and it is exactly
  the motivation for the imaging arm (images may add complementary signal) and for a richer gene
  signature than 50 genes.

**Assumptions**
- Per-gene z-scores make microarray (METABRIC) and RNA-seq (TCGA) features comparable.
- Cox model with same settings as Stage A (ridge penalizer 0.1).

**Limitations**
- TCGA has few deaths (151) and shorter follow-up → the TCGA-trained model is noisier, which
  likely explains why TCGA→METABRIC (0.572) is weaker than METABRIC→TCGA (0.652).
- External validation here is for the *genes* model only; clinical features differ between
  cohorts and were not harmonized (future work).
- Single bootstrap on out-of-fold predictions; not a nested-CV significance procedure.

---

## Entry 5 — 2026-06-24 · Stage B result: image classifier works (BreakHis)

**What was done**
- Ran `notebooks/stage_b_breakhis_colab.ipynb` on a Colab T4 GPU: transfer-learning CNN
  (EfficientNetB0) classifying benign vs. malignant breast histopathology (200x images),
  with a leakage-safe **patient-level** train/val/test split and class weighting (6 epochs).

**Results (held-out TEST patients, never seen in training)**
- **AUC = 0.844 | accuracy = 0.817**
- Confusion matrix `[[48, 24], [56, 310]]` (rows = true benign / malignant):
  - Sensitivity (caught malignant): 310/366 = **84.7%**
  - Specificity (correct benign): 48/72 = **66.7%**
- Training/validation reached val_AUC ≈ 0.97; test AUC 0.844 (healthy val→test gap).

**Why it matters**
- Demonstrates the full image deep-learning pipeline (download → patient-level split → train →
  honest evaluation) works and that histology images carry strong, learnable signal.
- The val→test drop and the imperfect test AUC are *evidence of honesty*: the patient-level
  split prevented the inflated near-perfect scores that leakage would produce.

**Assumptions**
- 200x magnification is representative; ImageNet features transfer to histology (they do).
- Only the model "head" was trained (base frozen) — deeper fine-tuning could improve it.

**Limitations**
- This is **classification, not survival** — it validates the pipeline, not the survival claim.
- Only 82 patients; small. 56 malignant images were misclassified as benign (false negatives) —
  the clinically costlier error type, worth reporting explicitly.
- ROC figure currently lives in the Colab output; download it to
  `results/figures/stageB_breakhis_roc.png` to keep it in the repo.

---

## Entry 6 — 2026-06-25 · Stage A+ : full transcriptome vs 50 genes (does more help?)

**What was done**
- Built `src/boost_genes.py`. Downloaded the **full expression matrices** (METABRIC 20,604
  genes; TCGA 20,472) from the cBioPortal datahub, and tested the **top-1,000 most-variable
  genes** (20× the PAM50 panel) with high-dimensional survival models — **Random Survival
  Forest** and **ridge-penalized Cox** (scikit-survival) — against PAM50 and clinical staging.
  Same rigor: 5-fold CV, out-of-fold C-index, external validation, paired bootstrap.

**Results (out-of-fold C-index)**
| Model | C-index |
|---|---|
| Clinical staging (Stage A) | 0.662 |
| PAM50 genes (RSF) | 0.608 |
| Top-1,000 genes (RSF) | 0.599 |
| Top-1,000 genes (ridge Cox) | 0.576 |
- Top-1,000 vs PAM50: ΔC = **−0.010, 95% CI [−0.024, +0.004] → not significant** (more genes did *not* help).
- Best gene model (0.599) still **does not beat clinical staging (0.662)**.
- **External validation** (train METABRIC top genes → test TCGA, RSF, 926 shared genes): **C-index 0.651** (generalizes, consistent with Stage A).

**Why it matters**
- This **strengthens the Stage A negative into a robust one**: the earlier "genes don't beat
  clinical" was NOT an artifact of using too few genes. Even the full transcriptome, with both
  a non-linear (RSF) and a linear penalized model, **plateaus at ~0.60** and stays below clinical.
- Scientifically cleaner conclusion: bulk gene-expression prognostic signal is **real and
  generalizable (~0.60, external 0.65) but caps below standard clinical staging** — which is
  exactly why the **imaging / multi-modal arm** is the place new signal must come from.

**Assumptions**
- Top-1,000 genes chosen by **unsupervised variance filter** (uses expression only, not outcomes).
- Cross-platform z-scores (ref-diploid) treated as comparable; RSF n_estimators=100; ridge α=100 (fixed).

**Limitations**
- Tested top-1,000 genes (not literally all ~20k) and two model families — but the flat result
  *across* models argues the ceiling is real, not a modeling choice.
- Ridge α was fixed, not nested-CV tuned. Bulk expression only (no single-cell / spatial).

---

## Entry 7 — 2026-07-08 · Stage C (imaging survival) + Stage D (fusion): the confound & the honest result

**What was done**
- Built the imaging-survival pipeline (`src/stage_c_local.py`): whole-slide image → tile →
  ResNet50 features → average per patient → Cox. Ran **locally on CPU** (hit Colab GPU limits),
  using OpenSlide + **resume-capable GDC downloads** for the gigapixel (~0.4–1 GB) slides.
- **Caught a confound:** the download shortcut (alive = small slides, dead = big slides) made
  **slide file size itself predict survival** (size-alone C-index 0.63), inflating the image
  C-index to a spurious **0.907**. Rejected it.
- **Fixed it:** size-matched case/control sampling (nearest-neighbor on slide size) → 14 dead +
  14 alive, mean slide size **372 vs 377 MB**. Built `src/stage_cd.py` = confound gate +
  images-only + genes-only + **fusion**, all 5-fold out-of-fold C-index + bootstrap 95% CI.

**Results (n=28, 14 deaths, size-matched)**
| Test | C-index | 95% CI |
|---|---|---|
| Confound gate (slide size alone) | **0.434** ✅ controlled | — |
| Images only | 0.555 | [0.28, 0.78] |
| Genes only (PAM50, this subset) | 0.491 | [0.30, 0.67] |
| **Fusion (genes + images)** | **0.578** | [0.35, 0.78] |
- Fusion − best single modality = **+0.023** (directionally consistent with multi-modal, not significant).

**Why it matters**
- The confounded **0.907 collapsed to ~0.5** once slide size was controlled → the imaging "signal"
  was almost entirely an acquisition artifact. Rigor caught a false positive — the headline lesson.
- Fusion being the highest is a *faint hint* for the multi-modal hypothesis, but **all CIs span
  0.5** → nothing is statistically distinguishable from chance.

**Assumptions**
- Size-matching removes the main acquisition confound (verified: gate 0.43 ≈ 0.5).
- Generic ResNet50/ImageNet features + mean pooling (not pathology-specific / attention-MIL).

**Limitations (honest)**
- Only **28 patients / 14 deaths** → severely underpowered; point estimates unstable (images
  0.42 at n=27 → 0.56 at n=28 — one patient moved it 0.14).
- Single cohort (TCGA), low magnification, generic features. A fair test needs **hundreds** of
  patients (institutional scale).
- The genes-on-this-subset number (0.49) is small-sample noise and does **not** revise Stage A's
  full-cohort external-validation result (0.65).
- **Conclusion:** no statistically reliable imaging or fusion survival signal in this sample; the
  durable contribution is the pipeline + the caught confound.

---

## Entry 8 — 2026-07-XX · Stage C/D DEFINITIVE (well-powered, n=221)

**What was done**
- Cracked the download bandwidth wall: GDC throttles ~0.3 MB/s per connection, but **8 parallel
  streams aggregate to ~2 MB/s** (`src/stage_c_parallel.py`). Scaled the size-matched sample to
  **221 patients / 111 deaths** (heading to 260) — a genuinely well-powered WSI survival study.
- Re-ran `src/stage_cd.py` (confound gate + images / genes / fusion, 5-fold OOF C-index + bootstrap CI).

**Results (n=221, 111 deaths, size-matched)**
| Test | C-index | 95% CI |
|---|---|---|
| Confound gate (slide size alone) | **0.496** ✅ | — |
| Genes only (PAM50) | **0.612** | [0.55, 0.68] — significant |
| Images only | **0.504** | [0.43, 0.57] — null |
| Fusion (genes + images) | **0.575** | [0.50, 0.64] |
- Fusion − best single = −0.037 (fusion does NOT beat genes).

**Why it matters (definitive)**
- Confound fully controlled (gate 0.496; alive 983MB ≈ dead 979MB).
- **Genes significantly predict survival** (CI clears 0.5), consistent with Stage A.
- **Images are a confident, well-powered null** (0.504, tight CI; converged from 0.40@n=96).
- **Fusion does not beat genes** — headline multi-modal hypothesis is rejected, well-powered.

**Limitations**
- Generic ImageNet features + mean pooling (not pathology-specific / attention-MIL) → "no image
  signal with a standard approach," not a claim that histology is inherently uninformative.
- Single cohort (TCGA).

---

## Entry 9 — 2026-07-23 · SOTA imaging upgrade (Phikon + attention-MIL), n=244

**What was done**
- Replaced generic ResNet/mean-pool with a **pathology foundation model**: Phikon (Owkin, trained
  on TCGA tissue) embeds each tile → 768-dim, saving the **full per-tile matrix** per patient
  (`src/stage_c_pathology.py`). Pooled with **attention-MIL** (gated attention + Cox head,
  `src/stage_e_attmil.py`) so the model learns which tumor regions carry survival signal.
- Scaled to **244 patients / 123 deaths** (16 of the 260 size-matched slides failed download —
  the largest files; not worth chasing).
- Two follow-up analyses: redundancy of the two risk scores (`src/stage_f_redundancy.py`) and
  late fusion vs. joint concat fusion (`src/stage_g_latefusion.py`).

**Results (n=244, 123 deaths, 5-fold OOF C-index + bootstrap CI)**
| Model | C-index | 95% CI |
|---|---|---|
| Images (Phikon + attention-MIL) | **0.603** | [0.54, 0.66] — real signal |
| Genes only (PAM50) | **0.611** | [0.56, 0.67] |
| Concat fusion (joint-trained) | 0.590 | [0.53, 0.65] — worst |
| **Late fusion (mean of risk scores)** | **0.639** | [0.58, 0.69] — best |
- Late fusion − best single arm = **+0.029**; ΔC bootstrap 95% CI **[−0.016, +0.073]**, P(not
  better) ≈ **0.11** → **trend, NOT statistically significant.**

**Redundancy check (why fusion behaves this way)**
- Image-risk vs gene-risk: Pearson **r = 0.22** (Spearman 0.24) → the two arms are **largely
  independent**, not redundant. Top-quartile high-risk flags overlap only 21/53 (Jaccard 0.25).
- Among patients genes rate **low-risk**, image-only still ranks deaths at **C = 0.586** → imaging
  rescues cases genes miss. The modalities are **complementary**.

**Why it matters**
- **Imaging went from a well-powered null (0.504, Entry 8) to a real signal (0.603)** — the earlier
  null was a *method* limitation (generic features + mean pool), not the biology.
- Because the arms are complementary, **late fusion (0.639) is the best model** — beats genes,
  images, and the concat fusion. But the improvement is **modest (+0.03) and not significant at
  this N.**
- **How you fuse matters:** naive joint concat (0.590) is the *worst* — below both single arms;
  simple late averaging of independent risk scores is best. A concrete, publishable methods lesson.

**Honest conclusion**
- Original headline ("fusion significantly beats either modality") is **not yet supported** — the
  effect trends the right way with a demonstrated mechanism (complementarity), but the single-cohort
  CI still includes zero. Correct statement: *"late fusion of complementary imaging and gene
  signals gave the highest concordance (0.639) and a consistent trend toward improvement (+0.03)
  that did not reach significance at n=244."*

**Limitations**
- Single cohort (TCGA-BRCA) → the trend needs **external validation** on an independent cohort to
  become a claim; more TCGA patients won't fix a ~0.03 effect.
- 16/260 slides missing (download failures on the largest files).
- PAM50 gene panel only (Stage A showed genes add little beyond clinical staging — clinical
  baseline not yet included in this fusion).

---

## Entry 10 — 2026-07-23 · Clinical baseline + site-held-out external validation, n=244

**Part A — the honest clinical baseline (`src/stage_h_clinical.py`)**
Added a clinical arm (age + AJCC/TNM stage) and asked the real question: does anything beat the
clinician? Out-of-fold C-index, same 244 patients / 123 deaths.
| Model | C-index | 95% CI |
|---|---|---|
| Clinical alone (age+TNM stage) | **0.632** | [0.57, 0.69] — strongest *single* arm |
| Genes (PAM50) | 0.611 | [0.56, 0.67] |
| Images (attention-MIL) | 0.603 | [0.54, 0.66] |
| Late: genes+images | 0.639 | [0.58, 0.69] |
| Late: images+clinical | 0.657 | [0.60, 0.71] |
| **Late: genes+images+clinical** | **0.666** | [0.61, 0.72] — best overall |

vs clinical alone (bootstrap ΔC): genes −0.021 · images −0.030 · genes+images +0.007 ·
**genes+images+clinical +0.034 [−0.017, +0.086]** (biggest, tightest — trend, not significant).
- **Headline (revised, more clinical):** *no single molecular modality beats clinical staging,
  but the full multi-modal fusion (0.666) trends toward adding value beyond the clinician (+0.034).*
  Better framing than "fusion vs genes" — it answers "does AI add to what the doctor already has?"

**Part B — external validation strategy (why leave-site-out)**
- Confirmed there is **NO second WSI breast cohort on GDC** — TCGA-BRCA is the only project with
  breast diagnostic slides (1,133); CPTAC-3 has 0. A planned CPTAC download was not possible.
- Pivoted to **leave-site-out validation** (`src/stage_i_leavesite.py`): the 244 patients span
  **25 tissue-source-sites** (institutions). GroupKFold by site → every test patient's institution
  is absent from training. Measures survival under real cross-site distribution shift (scanner,
  staining, population) with no new data.

**Part B results (site-held-out)**
| Model | Site-held-out C-index | Random-split (Entry 9) |
|---|---|---|
| Images | 0.577 [0.52, 0.64] | 0.603 |
| Genes | 0.595 [0.53, 0.65] | 0.611 |
| Late fusion | **0.607** [0.55, 0.66] | 0.639 |
- Small, healthy generalization gap (~0.02–0.03); **all CIs still exclude 0.5** → signal survives
  unseen institutions (not batch-effect). Late fusion remains the best arm across sites.

**Why it matters**
- The imaging/fusion signal **generalizes across hospitals**, strengthening it beyond a single
  random split. Together with the gene arm's true cross-cohort validation (METABRIC→TCGA 0.652,
  Stage A), the generalization story is now two-pronged.

**Limitations (honest)**
- Leave-site-out is **quasi-external** (still TCGA, same era/pipeline family) — a fully independent
  cohort would be stronger, but none exists publicly for breast WSI + genes + survival. **Screened
  CPTAC-BRCA** (the obvious candidate): it has 642 WSIs (TCIA) + transcriptomics (cBioPortal
  `brca_cptac_2020` / `breast_cptac_gdc`) but only **2 recorded deaths and no follow-up times**
  (proteogenomics cohort, not outcome-tracked) → cannot compute survival concordance → excluded.
- Clinical/fusion improvement over the clinician remains a **trend, not significant** at n=244.
- TNM stage extracted by string-parsing AJCC fields; median-imputed missing values.

---

## Entry 11 — 2026-09-23 · Robustness checks, clinical metrics, and paper figures (n=244)

**What was done**
- Built `src/make_figures.py`: computes every out-of-fold risk score once (random split,
  site-held-out, shuffled-label control, 4 extra seeds = 14 attention-MIL trainings), caches them in
  `results/oof_risks.csv`, and draws all figures + Table 1 from the cache.
- **Reproducibility check:** a fresh run reproduced all five headline C-indices from Entries 9–10
  exactly (images 0.603, genes 0.611, clinical 0.632, genes+images 0.639, full model 0.666).

**Results**
| Check | Result |
|---|---|
| Negative control (shuffled labels), all 6 models | 0.45–0.53 → **pass**, no leakage |
| Full model across 5 seeds | **0.660 ± 0.008** (range 0.644–0.668) |
| Full model, site-held-out (new) | **0.648** [0.59, 0.70]; clinical site-held-out 0.641 |
| Full model vs clinical (ΔC) | +0.034 [−0.017, +0.086], P(not better) = 0.11 → trend |
| Time-dependent AUC, mean over 1–8 yr | clinical **0.649** vs full model **0.696** |
| AUC at 1 / 3 / 5 yr | clinical 0.727 / 0.629 / 0.639 · full 0.748 / 0.699 / 0.684 |
| KM median split — clinical model | HR **2.32** [1.60, 3.36], log-rank p < 0.001 |
| KM median split — full model | HR **1.81** [1.26, 2.59], log-rank p = 0.001 |
| Image-risk vs gene-risk correlation | Pearson r = **0.19** |
| Image C within gene-low-risk half | **0.595** |

**Why it matters**
- The shuffled-label control and 5-seed stability show the full model's ~0.66 is neither leakage
  nor a lucky seed; it also holds on hospitals the model never saw (0.648).
- The full model beats clinical on time-dependent AUC at every horizon, most at 3 years (+0.07).

**Assumptions**
- KM groups use a global median split of out-of-fold risk (risk scores come from 5 fold-models).
- Time-dependent AUC uses the same patients as reference and test set.

**Limitations (honest)**
- **On a simple high/low split the clinical model separates survival MORE sharply** (HR 2.32 vs
  1.81). The multi-modal gain appears in fine-grained ranking (C-index, time-AUC), not in a coarse
  two-group split — stage alone already produces a strong dichotomy.
- The headline 0.666 is seed 42, near the top of the 5-seed range; **0.660 ± 0.008 is the more
  honest figure** to report.
- No significance test yet on the time-AUC difference; ΔC vs clinical is still not significant.
- The cohort is **50% deaths by design** (size-matched case/control), far above TCGA's ~14%, so
  absolute metrics may differ in an unselected population.
- Correlation r = 0.19 here vs 0.22 in Entry 9 (stage_f computed it on a different score
  definition); image C in the gene-low-risk half is 0.595 here vs 0.586 there. Same conclusion.
- Calibration not yet assessed.

---

## Entry 12 — 2026-09-24 · Attention heatmaps (where the image model looks)

**What was done**
- Built `src/stage_k_heatmaps.py`. Retrained the 5 image-arm fold models (seed 42) and kept them;
  they **reproduce the cached out-of-fold risks exactly**, so each heatmap comes from the model that
  never trained on that patient. Re-downloaded 4 example slides; recomputed tiles + positions with
  the Stage C pipeline — **feature match vs. saved = 1.0 for all 4** (tile positions are exact).
- Examples (from slides ≤ 0.7 GB): 2 rated high-risk who died (98th percentile; died at 6 and
  7 months), 1 rated low-risk who survived (18th pct; alive at 89 months), and **1 missed case**
  (20th pct; died at 17 months). → `results/figures/fig6_attention_heatmaps.png`

**Observations (qualitative, 4 slides, not a pathologist's read)**
- TCGA-LL-A73Z (high risk): tiles #2–#3 show crowded, dark, irregular nuclei — what aggressive tumor
  typically looks like. Its #1 tile (18.6% of attention) is mostly blank background at a tissue edge.
- TCGA-AC-A23H (high risk): attention is spread thin (top tile 7.7%) over fibrous stroma with
  scattered cells.
- Low-risk survivor and the missed case: the top tiles are mostly **fat and empty space**, with
  attention concentrated (15–26% per tile). In the missed case the model focused on fatty edge
  tissue rather than the large tumor mass — a plausible reason it under-rated the risk.

**Why it matters**
- Part of the image signal may come from **tissue composition** in the sampled tiles (fat/stroma vs.
  dense tumor) and from edge/background tiles, not only from tumor-cell appearance. Tiles are sampled
  at random across all tissue, and the tissue filter lets through mostly-white edge tiles.

**Limitations**
- 4 hand-picked examples; qualitative; needs a pathologist's review before any claim.
- Next test: measure how much attention lands on background/fat-like tiles across all 244 patients,
  and re-evaluate with a stricter tissue filter or tumor-only sampling.

---

## Entry 13 — 2026-09-30 · Full cohort (n = 1,033): the realistic test

**What was done**
- Downloaded and featurized every TCGA-BRCA patient with a diagnostic slide, survival data and
  complete PAM50 genes: **1,033 patients, 146 deaths (14.1%), 39 tissue-source sites** (244 original
  + 789 new; Sept 24–30). Every new slide was checked against GDC's size + MD5 checksum (0 mismatches);
  a final read of all 1,033 feature files found 0 problems. QC flag: TCGA-OL-A5RZ has 131/200 identical
  (blank-area) tiles; kept in the analysis as-is.
- Built `src/stage_l_fullcohort.py`: re-runs **every** n = 244 analysis unchanged (same models,
  hyperparameters, fold scheme, seeds; patients sorted by ID) by re-using `make_figures.py`, writing to
  `results/fullcohort/` and `results/figures/fullcohort/`. New: (1) frozen-model validation — train on the
  original 244 only, test on the 789 new patients; (2) calibration at 3 and 5 years (cross-fitted
  one-variable Cox recalibration of the out-of-fold risk, quintile plot + Brier score); (3) side-by-side
  comparison with n = 244; (4) the negative control repeated over 10 shuffles.
- `make_figures.py`: Fig 2 title and Table 1 title now come from the data; the forest-plot axis widens only
  if a CI would be cut off. Re-ran n = 244: every table and Fig 2 are **byte-identical** to before.

**Results (out-of-fold C-index, 95% bootstrap CI)**
| Model | n = 244 (size-matched) | n = 1,033 random 5-fold | n = 1,033 site-held-out |
|---|---|---|---|
| Clinical (age + TNM) | 0.632 | **0.722** [0.67, 0.77] | 0.728 [0.68, 0.77] |
| Genes (PAM50) | 0.611 | 0.618 [0.56, 0.68] | 0.605 [0.54, 0.67] |
| Images (Phikon + attention-MIL) | 0.603 | **0.641** [0.59, 0.69] | **0.640** [0.59, 0.69] |
| Genes + images | 0.639 | 0.661 [0.60, 0.71] | 0.649 [0.59, 0.70] |
| Images + clinical | 0.657 | 0.734 [0.68, 0.78] | 0.724 [0.67, 0.77] |
| Genes + images + clinical | 0.666 | 0.724 [0.67, 0.77] | 0.712 [0.66, 0.76] |

- **vs clinical alone (paired bootstrap ΔC, n = 1,033):** genes −0.103 [−0.169, −0.032] and images −0.079
  [−0.147, −0.013] (both significantly worse); genes+images −0.059 [−0.123, +0.003]; **full model +0.003
  [−0.037, +0.044]** (P(not better) 0.44); images+clinical +0.013 [−0.024, +0.050] (0.25); full model
  site-held-out −0.016 [−0.050, +0.017].
- **Robustness:** single-shuffle negative control 0.478–0.566; the gene arm's 0.566 was in the top 1% of
  random orderings, so the control was **repeated over 10 shuffles: genes mean 0.496 (0.463–0.546),
  clinical 0.490 (0.410–0.554) → pass, the 0.566 was one shuffle's fluke.** Full model over 5 seeds
  0.715 ± 0.006 (seed 42 = 0.724, again the top of the range).
- **Time-dependent AUC:** mean 0.731 (full) vs 0.716 (clinical); 1 / 3 / 5 yr: clinical 0.842 / 0.715 /
  0.702, full 0.833 / 0.721 / 0.717. **KM median split:** clinical HR 2.89 [2.03, 4.10], full HR 2.75
  [1.96, 3.85], both log-rank p < 0.001.
- **Calibration:** Brier at 3 yr — no model 0.090, clinical 0.086, full 0.082; at 5 yr — 0.149, 0.138,
  0.132. In all 20 quintile cells (2 models × 2 horizons × 5) the predicted survival falls inside the
  observed Kaplan-Meier 95% CI (largest gap 0.065).
- **Complementarity:** image vs gene risk r = 0.12; image C within the gene-low-risk half 0.594.
- **Frozen models (trained on the 244, tested on the 789 new; only 23 deaths):** clinical 0.759
  [0.62, 0.88], genes 0.688 [0.58, 0.78], images 0.616 [0.49, 0.74], full 0.776 [0.65, 0.88];
  full − clinical +0.018 [−0.071, +0.121].

**Why it matters**
- **The image signal is real and robust:** in the unselected cohort it is stronger than at n = 244 (0.641
  vs 0.603) and identical on hospitals the model never trained on (0.640 across 39 sites) — not a site or
  scanner artifact.
- **But clinical staging is much stronger in the real population (0.632 → 0.722), and adding genes and
  images no longer improves on it (+0.003, CI centred on zero).** The +0.034 trend seen at n = 244 did
  **not** replicate. On their own, genes and images are significantly *worse* than staging.
- A likely reason (hypothesis, not tested): the n = 244 sample was selected on outcome (half deaths) and
  matched on slide size, which changed the case mix and made staging look weaker than it is; the
  C-index depends on who is in the sample. This is itself a methods lesson about case/control designs.
- Both models are well calibrated; the full model's Brier is slightly lower (not significance-tested).
- **Honest headline for the paper:** a pathology foundation model extracts real, hospital-robust survival
  signal from routine H&E slides (C ≈ 0.64 across 39 sites), but in an unselected cohort it does not add
  discrimination beyond age + TNM stage.

**Assumptions**
- Late fusion = equal-weight sum of z-scored risks (unchanged from Entries 9–11; not tuned).
- Calibration recalibrates each out-of-fold risk with a one-variable Cox model fit in other folds.
- Frozen-validation fusion z-scores the risks within the test set (no outcomes used).

**Limitations**
- Only 146 deaths and median follow-up 32 months; late time-AUC values rest on few patients at risk.
- The frozen validation has just 23 deaths → wide CIs; it cannot confirm or refute small effects.
- No significance tests yet for the Brier or time-AUC differences.
- Equal-weight fusion only; a learned weighting (nested CV) was not tried.
- Single dataset (TCGA-BRCA); site-held-out is quasi-external, as before.

---

## Entry 14 — 2026-10-02 · Does the image model add anything beyond staging? (Amendment A)

**What was done**
- **Before running anything**, wrote PREREGISTRATION.md §12 (Amendment A) fixing exactly five
  follow-up analyses, the primary test, the multiplicity correction, and "no other model variants".
- Built `src/stage_m_added_value.py` and ran A1, A2, A3, A5 on the n = 1,033 out-of-fold risks.
  A4 (slide-size confound) could not run: the GDC API was down (HTTP 503) all evening; it runs
  automatically when the script is re-run with GDC available.
- Safety checks: a freshly trained fold-1 image model reproduced the cached out-of-fold risks
  (max difference 1.8e-15); the recomputed Brier scores equal Entry 13's; a full re-run reproduced
  every A1/A2/A3/A5 table byte-for-byte.

**A1 — independent prognostic value (pre-specified primary test)**
| Cox model comparison | LR χ² | p (LRT) | HR per 1 SD [95% CI] |
|---|---|---|---|
| clinical (age, stage, T, N) **+ image** vs clinical | 17.75 | **2.5 × 10⁻⁵** | **1.40 [1.20, 1.62]** |
| + gene vs clinical + image (secondary) | 4.03 | 0.045 | 1.18 [1.01, 1.40] |
| + image, site-held-out risk (sensitivity) | 7.94 | 0.005 | 1.28 [1.08, 1.51] |
- The image risk is nearly independent of staging: Spearman with stage 0.02, T 0.20, N 0.04, age
  0.10; age + stage + T + N explain only **5.5%** of its variance.
- Proportional hazards holds for the image term (Schoenfeld p = 0.83); T stage shows a mild violation
  (p = 0.04), which does not affect the image result.

**A2 — learned weights, nested CV** (Cox coefficients on z-scored risks; range over 5 outer folds:
clinical 0.38–0.54, image 0.16–0.32, gene 0.04–0.24 — the same ordering in every fold)
| Model | C-index [95% CI] | ΔC vs clinical [95% CI] | p (two-sided) |
|---|---|---|---|
| Clinical (age + TNM) | 0.722 [0.668, 0.768] | — | — |
| Equal-weight genes + images + clinical (Entry 13) | 0.724 [0.670, 0.771] | +0.003 [−0.037, +0.044] | 0.88 |
| **Stacked genes + images + clinical** | **0.738** [0.685, 0.783] | +0.017 [−0.011, +0.045] | 0.21 |
| Stacked images + clinical | 0.737 [0.686, 0.783] | +0.016 [−0.011, +0.043] | 0.26 |

**A3 — subgroups (exploratory; ΔC full − clinical, Holm across 4 tests)**
- Stage II (591, 66 deaths) −0.021 [−0.100, +0.058]; Stage III (234, 45) +0.040 [−0.034, +0.119];
  LumA (475, 55) +0.008 [−0.055, +0.073]; LumB (191, 30) +0.016 [−0.067, +0.110]. **All Holm p = 1.0.**
- Descriptive only (< 30 deaths): Stage I images 0.728 vs clinical 0.593 (13 deaths, CIs overlap
  widely); Stage IV, Basal, Her2, Normal — CIs too wide to interpret.
- The image model still ranks risk *within* stage II (0.618) and stage III (0.626).

**A5 — remaining statistics (paired bootstrap, 1,000 resamples)**
- Time-dependent AUC, full − clinical: mean +0.015 [−0.030, +0.059]; 1 yr −0.009; 3 yr +0.006; 5 yr +0.016
  (all CIs include 0).
- Brier, full − clinical: 3 yr −0.0036 [−0.0082, +0.0009]; 5 yr −0.0061 [−0.0152, +0.0028] (both favour the
  full model; both CIs include 0).
- Decision curves (`fig9_decision_curves.png`): the full model's net benefit is equal to or slightly above
  clinical at most thresholds (e.g. 5-yr, 40% threshold: 0.029 vs 0.009); both beat treat-all/none
  above ~10%. Descriptive.
- Largest C-index gain over clinical compatible with the data: +0.044 (equal weight), +0.045 (stacked).

**A4 — slide-size confound, full cohort** (run 2026-10-03 once the GDC API returned; the re-run left the
A1/A2/A3/A5 tables byte-identical)
- GDC lists 1,133 TCGA-BRCA diagnostic slides. The exact slide is known for 1,026 patients (all 789 new +
  237 single-slide originals); 7 original patients have 2 diagnostic slides (ambiguous).
| Analysis | n (deaths) | C-index of slide size alone [95% CI] | Spearman, image risk vs size |
|---|---|---|---|
| primary: unambiguous patients | 1,026 (144) | **0.580 [0.515, 0.641]** | −0.13 |
| ambiguous = smallest slide | 1,033 (146) | 0.575 [0.514, 0.633] | −0.13 |
| ambiguous = largest slide | 1,033 (146) | 0.578 [0.516, 0.636] | −0.13 |
- **Slide size does predict survival in the full cohort** (bigger file → higher risk; CI excludes 0.5),
  more weakly than in the original selection (0.63). So, as pre-specified, A1 was repeated with
  log(slide size) added: **image HR 1.38 per SD, LR χ² 16.71, p = 4.4 × 10⁻⁵** (vs 1.40 without it).
- **The image result does not depend on slide size.** The image risk is slightly *negatively* correlated
  with size (−0.13), so it is not a hidden file-size reader. (Why size predicts survival is not tested; a
  plausible reason is that larger specimens come from larger tumors.)

**Why it matters**
- **The pre-specified primary test is clearly positive:** the image score carries prognostic information
  that age and TNM stage do not (HR 1.40 per SD, p = 2.5 × 10⁻⁵), and it holds on unseen hospitals.
- That information is largely *new* — the image score is almost uncorrelated with stage.
- Yet the gain in discrimination is small (+0.017 C-index with proper weights) and not statistically
  significant. This is the well-known pattern that a significant new predictor added to an already
  strong model barely moves the C-index; every other metric (time-AUC, Brier, net benefit) moves in the
  same direction, but none reaches significance with 146 deaths.
- **Revised headline:** histology contains independent, hospital-robust survival information beyond
  staging; with 146 deaths it yields only a small, non-significant gain in discrimination.

**Assumptions**
- A1 fits the Cox model in-sample on out-of-fold image risks (the image score never saw the patient's
  outcome); the LRT tests association, not discrimination.
- A2 applies the learned weights to first-level out-of-fold risks z-scored over the whole cohort.

**Limitations**
- A4 ran on 2026-10-03 (above). For 7 original patients the exact slide is ambiguous; all three
  sensitivity versions agree.
- Subgroups are exploratory and underpowered; the stage I observation is a hypothesis only.
- Decision curves are descriptive (no CIs).
- One dataset; 146 deaths limits power for small gains.

---

## Entry 15 — 2026-10-03 · Is the image signal just tissue composition? (Amendment B, B1–B3)

**What was done**
- **Before running anything**, wrote PREREGISTRATION.md §13 (Amendment B): the analyses, the primary test,
  and the decision rules. GDC was still down (HTTP 503 since Oct 1), so B1–B3 used only local features;
  B4 (labelling clusters from re-downloaded tiles, then re-training without fat/background) waits for GDC.
- Built `src/stage_n_tissue.py`. B1: all **206,488 tiles** of the 1,033 patients (a few original-cohort
  slides had < 200 tissue tiles) → standardized Phikon features → PCA(50) (keeps 51.2% of the variance)
  → k-means k = 8 (sensitivity k = 12), seed 42; no outcomes used.
- B2: re-trained the five out-of-fold image models; they reproduced the cached image risks exactly
  (max difference 1.8e-15). Each patient's attention came from the model that never saw them.
- A full re-run reproduced the B2 and B3 tables byte-for-byte.

**B3 — primary test (pre-specified): image risk beyond clinical + tissue composition**
| Cox comparison | LR χ² | p | image HR per 1 SD [95% CI] |
|---|---|---|---|
| reference (A1): + image vs clinical only | 17.75 | 2.5 × 10⁻⁵ | 1.40 [1.20, 1.62] |
| **+ image vs clinical + composition, k = 8 (PRIMARY)** | **9.17** | **0.0025** | **1.29 [1.10, 1.52]** |
| + image vs clinical + composition, k = 12 (sensitivity) | 8.26 | 0.0041 | 1.28 [1.09, 1.51] |
| + composition (7 df) vs clinical + image (secondary b) | 11.18 | 0.13 | — |
- **Secondary a:** tissue composition alone (k = 8, out-of-fold Cox) C-index **0.532 [0.48, 0.59]** —
  barely above chance.
- **Primary test passed:** the image risk stays an independent prognostic factor after adjusting for age,
  stage, T, N **and** tissue composition. Its hazard ratio shrinks from 1.40 to 1.29, so part of the image
  signal overlaps with composition, but most of it does not.

**B2 — where the attention goes (k = 8)**
| Cluster | share of tiles | share of attention | enrichment | hospitals* | Spearman (fraction vs image risk) |
|---|---|---|---|---|---|
| 2 | 20.7% | **49.3%** | **2.38×** | 20 | **−0.37** |
| 6 | 37.2% | 13.7% | 0.37× | 37 | +0.03 |
| 4 | 11.9% | 8.5% | 0.71× | 5 | +0.11 |
| 1 | 8.0% | 7.4% | 0.92× | 4 | −0.04 |
| 5 | 8.0% | 7.5% | 0.94× | 2 | +0.05 |
| 3 | 6.6% | 6.4% | 0.96× | 1 | +0.22 |
| 7 | 6.4% | 6.0% | 0.94× | 1 | −0.17 |
| 8 | 1.3% | 1.2% | 0.94× | 1 | +0.14 |
\*Hospitals among patients whose tiles are > 50% in that cluster (post-hoc **descriptive** check, not a test).
- The model puts half its attention on **cluster 2** (2.4× its share of tiles); patients with more
  cluster-2 tissue get *lower* image risk. It largely ignores **cluster 6**, the most common cluster.
- k = 12 shows the same pattern (one cluster at 2.28× enrichment, Spearman −0.28; two at ~0.3×).

**Unexpected finding (descriptive, post hoc): most clusters are hospitals, not tissue types**
- Clusters 3, 5, 7, 8 each come from one hospital (99–100%: A8, E2, AR, E9); cluster 1 mostly BH (69%),
  cluster 4 mostly D8 (52%). Only clusters **2 (20 hospitals)** and **6 (37 hospitals)** are shared tissue
  types. Half the patients have 90% of their tiles in a single cluster.
- So Phikon features strongly encode each lab's staining / scanning style (a known property of pathology
  foundation models), and the k = 8 "composition" is part tissue split (2 vs 6), part hospital style.

**Why it matters**
- **The shortcut concern is largely answered:** tissue composition alone barely predicts survival (0.532),
  adds nothing beyond clinical + image (p = 0.13), and the image risk survives adjustment for it
  (HR 1.29, p = 0.0025). The image model's survival signal is about *what* the tissue looks like, not just
  *how much* of each kind there is.
- Because the composition clusters also capture hospital style, the same test is extra evidence that the
  image signal is not a hospital/stain artifact — consistent with the site-held-out results.
- The model concentrates on one cross-hospital tissue type (cluster 2). Identifying it (fat? stroma?
  tumor?) is the job of B4.

**Assumptions**
- k-means on PCA(50) of standardized Phikon features is a reasonable coarse summary of tile appearance.
- The Cox fits are in-sample on out-of-fold image risks (association test, as in A1).

**B4 — labelling the clusters (run 2026-10-03 after GDC returned)**
- Clarification of §13: "exact saved positions" exist only for the 789 full-run patients, so the 30 random
  patients (seed 42, exact slide ≤ 0.7 GB) were drawn from those 257 eligible. The 60 example tiles
  (10 nearest the centre for clusters 1–6) came from 10 slides (3.8 GB), downloaded MD5-verified, cropped at
  the saved positions, then deleted. **Every crop's Phikon features matched the saved features (cosine
  similarity 1.0000, all 60)** — these are exactly the tiles the model saw. Clusters 7 and 8 (single
  hospitals AR and E9) had no tiles in the sample and could not be labelled.
- Non-expert labels (`results/fullcohort/tissue/b4_cluster_labels.csv`, `fig12_cluster_examples.png`):
  **cluster 2 = dense collagenous (fibrous) stroma**; **cluster 6 = cellular tissue** (sheets of dark nuclei —
  tumor or immune cells — and tissue edges); 1 and 5 = stroma (hospital styles); 3 = tumor + stroma
  (one hospital); 4 = mixed acellular/hazy material + stroma (borderline, about 4/10 artifact-like).
- **Decision rule (≥ 6 of 10 fat/background): no cluster qualifies**, so the pre-specified filtered
  re-training is not triggered (it would be identical to the original model).
- **Descriptive cross-check (Entry 12 heatmap tiles):** the low-risk patient's top tiles, including its fat
  tiles, fall in **cluster 2**; the missed patient's tiles are 100% cluster 7 (hospital AR); both high-risk
  patients' top tiles fall in cluster 6. So cluster 2 is stroma at its core **but also holds fat tiles** —
  k = 8 is too coarse to separate fat from fibrous stroma.

**What B1–B4 establish, and what they don't**
- Established: the image signal is not explained by coarse tissue composition (B3 primary passed;
  composition alone ≈ chance); the model concentrates on stroma-rich tissue (cluster 2) and largely ignores
  cellular tissue (cluster 6); more cluster-2 tissue → lower image risk.
- Not established: whether a *fat-specific* shortcut operates inside cluster 2. A finer split of cluster 2
  (sub-clustering + labelling) would be needed; it would require a new pre-registered amendment.

**Limitations**
- Cluster labels are a non-expert visual read of 10 central crops per cluster; pathologist review needed.
- k-means clusters mix tissue type with hospital style, so "composition" is an imperfect stand-in for
  tissue type; a finer shortcut (e.g. within-cluster texture) is not ruled out.
- The hospital breakdown was not pre-registered (descriptive only).
- PCA(50) keeps about half the feature variance.

---

## Entry 16 — 2026-10-04 · Amendment C: fat inside cluster 2, robustness, gene explanations

**What was done**
- Wrote PREREGISTRATION.md §14 (Amendment C) before running anything. Built `src/stage_o_followups.py`.
- Safety checks: re-trained models reproduced the cached image risks (C2) and gene risks (C3, max
  difference 2.2e-16). For C1, a first check compared raw k-means cluster *numbers* and failed (0.78%
  agreement) — k-means can number identical clusters differently. It was replaced (before any download) by
  a renumbering-proof check: each tile's distance to its saved cluster centre matches B1 (median relative
  difference 3.4e-5, correlation 1.000000) and the partitions agree exactly (adjusted Rand index 1.0000).

**C2 — robustness (pre-specified)**
| Check | Result |
|---|---|
| (a) image model on 100 tiles/patient | C-index **0.634** [0.578, 0.687] (200 tiles: 0.641); A1 HR **1.41** [1.21, 1.64], p = 2.0e-5; risk correlates 0.90 with the 200-tile risk |
| (b) weights clinical:image:gene = 1:1:1 / 2:1:1 / 3:1:1 / 1:1:0 / 2:1:0 | C-index 0.724 / 0.737 / 0.738 / 0.734 / 0.740; ΔC vs clinical +0.003 / +0.016 / +0.017 / +0.013 / +0.019 (all CIs include 0; p 0.12–0.88) |
| (c) without TCGA-OL-A5RZ (blank slide) | image 0.642, full 0.724, clinical 0.721; A1 HR 1.40, p = 2.5e-5 (unchanged) |
- All five weightings beat clinical slightly and none significantly; heavier clinical weighting
  (0.737–0.740) matches the nested-CV learned weights (0.738). No weighting is "selected".

**C3 — gene explanations (descriptive)**
- The gene arm is linear in the 50 PAM50 genes, so out-of-fold SHAP values are exact (weight × centred
  expression). Top genes by mean |SHAP| (direction; folds agreeing): BLVRA (↓ risk; 5/5), UBE2T (↑; 5/5),
  PGR (↓; 5/5), TMEM45B (↑; 5/5), CXXC5 (↑; 5/5), BCL2 (↓; 5/5), KRT17 (↓; 5/5), BAG1 (↓; 5/5), MLPH (↓; 5/5),
  ESR1 (↑; 5/5), ANLN (↑; 5/5), MKI67 (↑; 4/5), BIRC5 (↓; 4/5), EGFR (↓; 5/5), GRB7 (↑; 5/5). Figure: fig14.
- Biology check: hormone-related "good-prognosis" genes (PGR, BCL2, BAG1, MLPH) lower risk; proliferation
  (UBE2T, ANLN, MKI67) and HER2-amplicon (GRB7) genes raise it — as expected.
- Post-hoc descriptive context (each gene alone, univariate Cox): 14 of the top 15 have the same direction
  alone as inside the model. ESR1 alone has no effect in this cohort (HR 1.00, p = 0.98); KRT17 alone is
  protective here (HR 0.78, p = 0.001). Only **BIRC5** flips (neutral alone, protective in the model) — it is
  strongly correlated with UBE2T (0.80) and MKI67 (0.75), so the model splits credit among them.

**C1 — is there a fat sub-cluster inside cluster 2? (primary for this amendment)**
- Cluster 2's 42,721 tiles were split with k-means k = 4. Every sub-cluster draws more attention than its
  share of tiles (2a 2.40×, 2b 2.11×, 2c 2.49×, 2d 2.12×), and more of each → lower image risk.
- 30 example crops (10 nearest each centre for 2a–2c, from the same 30 B4 patients; 8 slides, 3.1 GB,
  MD5-verified, deleted after cropping) all matched the saved features (cosine 1.0000). 2d had no tiles in
  the sample (55 patients, mostly one hospital) and could not be labelled.
- Non-expert labels (`results/fullcohort/followups/c1_labels.csv`, `fig13_cluster2_subclusters.png`):
  **2a** stroma / smooth-muscle-like tissue with some tissue edges (≈ 2/10 background, one ink mark);
  **2b NO TISSUE** — 7 uniform dark grey/black squares and 3 solid red or red-white gradient squares (10/10);
  **2c** dense collagenous stroma (0/10). **No sub-cluster is fat.**
- **Rule triggered for 2b** (non-tissue background/artifact that slipped through the HSV tissue filter:
  dark areas have low brightness, red areas high saturation). 1,879 tiles (0.9%) from 235 patients removed;
  no patient fell below 5 tiles.
| Test | Result |
|---|---|
| **C1 primary:** image model re-trained without 2b tiles, + clinical vs clinical | **HR 1.37 [1.18, 1.59], LR χ² 16.28, p = 5.5e-5** |
| C-index without 2b tiles / original | 0.637 [0.582, 0.690] / 0.641 [0.586, 0.695] |
| C1 secondary: original image risk beyond clinical + 2b fraction | HR 1.40, p = 2.5e-5 (unchanged) |
- Descriptive cross-check: the Entry-12 fat tiles of TCGA-A2-A0CQ fall in sub-cluster **2a** — fat is mixed
  in with stroma rather than forming its own cluster, even at this finer split.

**Why it matters**
- **The image signal survives removing the one clearly non-tissue group** (HR 1.40 → 1.37, still
  p < 1e-4) and does not depend on how much non-tissue a slide has. It is also stable with half the
  tiles, any reasonable fusion weighting, and without the blank-slide patient.
- **A real pipeline flaw was found:** the tissue filter admits a small fraction (0.9%) of non-tissue
  tiles (dark background, red ink/stain), and the model gives them about twice their share of attention.
  Removing them barely changes performance, but a stricter filter belongs in any future version.
- No fat-dominated cluster exists at either k = 8 or the finer split of cluster 2; fat tiles occur mixed
  with stroma. A fat-specific effect within stroma-rich tissue is therefore not separable with these
  methods (it would need tile-level tissue labels, e.g. from a pathologist or a tissue classifier).

**Assumptions**
- Visual labels are a non-expert read of 10 central crops per sub-cluster; sub-cluster 2d is unlabelled.
- Exact SHAP for C3 relies on the gene model being linear (verified from its structure: the image input is
  empty, so the risk is a weighted sum of the genes plus a constant).

**Limitations**
- Fat could not be isolated as a cluster, so "no fat shortcut" is not proven — only that no fat-dominated
  tile group exists and that the image signal survives every composition-based adjustment tried.
- Pathologist review of figs 12–13 is still needed.
