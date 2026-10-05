# Multi-Modal Machine Learning for Breast-Cancer Prognosis: Cross-Cohort Genomic Validation, Histopathology Classification, and a Cautionary Imaging-Survival Confound

**Shiv Prahalathan** · Summer 2026 · [github.com/Aeronite-P/breast-cancer-prognosis](https://github.com/Aeronite-P/breast-cancer-prognosis)

---

## Abstract

Breast cancer is biologically heterogeneous, and predicting patient survival from tumor data
remains difficult. I asked whether machine learning applied to two complementary views of a
tumor — its **gene expression** and its **histopathology appearance** — can predict overall
survival, and whether combining them outperforms either alone. Using fully public data
(METABRIC, n≈1,980; TCGA-BRCA, n≈1,069; BreakHis, ~9,000 images), I built and validated three
components. A gene-expression survival model achieved a cross-validated concordance index
(C-index) of ~0.60 and, critically, **generalized across two independent cohorts on two
measurement platforms** (external C-index 0.65); however, it did **not** significantly improve
on standard clinical staging (0.66), a result that held even when expanding from a 50-gene panel
to the full transcriptome. A transfer-learning image classifier distinguished benign from
malignant tissue at **AUC 0.84** on held-out patients. A naive image-survival model initially
appeared highly predictive (C-index 0.91), but I traced this to a **slide-size confound**
(scan file size alone predicted survival at 0.63); after correcting it with size-matched
sampling, the imaging signal fell to chance (0.46–0.56), and a larger, properly powered analysis
is underway. The project's central contribution is methodological: a fully reproducible,
externally validated, and honestly reported multi-modal pipeline, including the detection and
correction of a subtle confound that would otherwise have produced a false-positive result.

---

## 1. Background

Breast cancer is not one disease but many. Tumors are classified into molecular **subtypes**
(Luminal A/B, HER2-enriched, Basal-like) defined by receptor status (ER, PR, HER2), and these
subtypes differ markedly in aggressiveness and prognosis. Clinicians estimate prognosis using
**staging** (tumor size, lymph-node involvement, grade). Two data modalities capture
complementary aspects of a tumor: **gene expression** reflects its molecular biology, while
**histopathology** (microscope images of stained tissue) reflects its physical structure.

Most machine-learning student projects train a single model on a single dataset and report a high
accuracy — which often reflects overfitting rather than a real, generalizable finding. I designed
this project around three principles that distinguish a credible prognostic study: (1) **external
validation** across independent cohorts, (2) an explicit **"does it beat standard of care?"**
comparison against clinical staging, and (3) **honest reporting**, including a pre-registered plan
and a negative control against data leakage.

**Central question:** Can gene expression and histopathology predict breast-cancer survival, does
each modality generalize, and does combining them add value beyond either alone and beyond
clinical staging?

## 2. Data

| Dataset | Modality | Size | Source |
|---|---|---|---|
| METABRIC | Gene expression (microarray) + clinical/survival | ~1,980 patients | cBioPortal |
| TCGA-BRCA | Gene expression (RNA-seq) + clinical/survival + slides | ~1,069 patients | cBioPortal / GDC |
| BreakHis | Histopathology images (benign/malignant) | ~9,000 images, 82 patients | BreakHis database |

All data are public and de-identified. Endpoint: **overall survival**.

## 3. Methods

**Genomic survival modeling.** Cox proportional-hazards and Random Survival Forest models were
trained on gene expression (the 50-gene PAM50 panel and, separately, the full ~20,000-gene
transcriptome). Performance was measured by the **C-index** (concordance; 0.5 = chance) using
5-fold cross-validation with strict train/test separation and a fixed random seed. Two safeguards
were applied: a **negative control** (survival labels shuffled to destroy signal — a correct
pipeline then scores ~0.5) and **external validation** (train on one cohort, test on the other).

**Histopathology classification.** A convolutional neural network (EfficientNet, pretrained on
natural images) was fine-tuned on BreakHis using **transfer learning**, with a **patient-level
split** so that no patient appeared in both training and test sets. Performance was measured by
**AUC**.

**Imaging survival.** Whole-slide images were tiled into patches, each patch was embedded with a
pretrained network (ResNet50), and the patch embeddings were averaged into one feature vector per
patient, which was used to predict survival (Cox model, C-index with bootstrap confidence
intervals).

**Integrity.** All results are reproducible from public data via the project's code repository,
with a pre-registered analysis plan and a running lab notebook documenting every step.

## 4. Results

### 4.1 Genes predict survival and generalize — but do not beat clinical staging

| Model | C-index |
|---|---|
| Clinical staging | 0.662 |
| Gene expression (PAM50, 50 genes) | 0.599 |
| Gene expression (full transcriptome, ~20k genes) | ~0.60 |
| Clinical + genes | 0.666 |
| Shuffled-label negative control | 0.512 |
| **External validation (train METABRIC → test TCGA)** | **0.652** |

Gene expression carried real prognostic signal that **generalized across cohorts and platforms**
(external C-index 0.65). The negative control scored 0.51, confirming no data leakage. However,
adding genes to clinical staging produced no statistically significant improvement (ΔC = +0.005,
95% CI [−0.004, +0.015]), and this ceiling held even with the full transcriptome — indicating the
limitation is fundamental to bulk gene expression, not a matter of using too few genes.

### 4.2 Histopathology classification

The transfer-learning classifier distinguished benign from malignant tissue at **AUC 0.844**
(accuracy 82%) on held-out patients, confirming that the image deep-learning pipeline extracts
strong, learnable signal.

### 4.3 Imaging survival: a confound caught and corrected

A first imaging-survival model appeared strongly predictive (C-index **0.907**). Suspicious of
this result, I tested whether **slide file size** — an acquisition artifact unrelated to biology —
predicted survival, and found it did (C-index 0.63): my sampling had inadvertently paired
deceased patients with larger scans and surviving patients with smaller scans, so the model was
partly detecting scan size, not tumor biology. After correcting this with **size-matched
case/control sampling** (deceased and surviving patients matched on slide size, verified by a
confound gate returning ~0.5), the apparent imaging signal collapsed to chance
(C-index 0.46–0.56; all confidence intervals include 0.5). This corrected analysis was initially
underpowered (n=28); a properly powered analysis (~260 patients, ~130 deaths) is in progress.

_[This section will be updated with the well-powered imaging-survival and fusion result.]_

## 5. Discussion

The strongest empirical result is the **cross-cohort generalization** of the gene-expression
model: a model trained on one patient population and measurement technology retained predictive
ability on an entirely separate population and technology. Yet the honest and clinically important
finding is that gene expression **does not add value beyond standard clinical staging** — a result
made robust by testing the full transcriptome.

The imaging-survival component illustrates why methodological rigor matters more than a headline
number. An uncritical reading would have reported a C-index of 0.91 as evidence that tumor
appearance strongly predicts survival. In reality, that value was largely a **confound**:
correcting for it removed the signal. Detecting and correcting such artifacts — rather than
reporting the inflated number — is the core discipline of prognostic modeling.

**Limitations.** (1) The imaging-survival question was initially underpowered; the well-powered
result is still being computed. (2) All data are retrospective and observational, so results are
associations, not causal, and this is a research model, not a clinical tool. (3) Gene expression,
while validated, offered no improvement over standard staging. (4) The histopathology classifier
addresses a comparatively well-solved task on a small (82-patient) cohort. (5) The imaging
pipeline used generic (ImageNet) features and mean pooling rather than pathology-specific models
or attention-based aggregation, so a stronger pipeline might detect signal this one did not. (6)
The imaging and fusion analyses are single-cohort (TCGA).

## 6. Conclusion

This work presents a reproducible, externally validated, and honestly reported multi-modal
pipeline for breast-cancer prognosis. Gene expression predicts survival and generalizes across
cohorts but does not surpass clinical staging; histopathology images are readily classifiable; and
a promising-looking imaging-survival result was shown to be a confound and corrected. The project's
principal value lies in its methodology — external validation, a leakage negative control, and the
detection and correction of a subtle confound — demonstrating that careful, honest analysis can
distinguish real signal from artifact.

---

## Data and Code Availability

All code, figures, the pre-registered analysis plan, and a step-by-step lab notebook are available
at [github.com/Aeronite-P/breast-cancer-prognosis](https://github.com/Aeronite-P/breast-cancer-prognosis).
Data are public: METABRIC and TCGA-BRCA via cBioPortal (Cerami et al., 2012; Gao et al., 2013) and
the GDC; BreakHis (Spanhol et al., 2016).
