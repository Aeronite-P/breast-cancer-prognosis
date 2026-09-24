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
