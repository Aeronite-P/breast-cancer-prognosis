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
