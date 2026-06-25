# Multi-Modal Breast-Cancer Prognosis: Do Genes + Tumor Images Predict Survival Better Together?

**One-line:** Train AI on two different "views" of the same tumors — the **gene-expression
fingerprint** (which genes are on/off) and the **microscope-slide appearance** (how the
tissue actually looks) — to predict breast-cancer survival, and test whether **combining
them beats either one alone**. Validated across independent cohorts, reproducible from
public data, with zero fabricated results.

> Student researcher: Shiv Prahalathan · 16 y/o · Summer 2026 · heavy AI-assisted.

---

## 1. Why this project is strong (and honest)

Most high-school "AI + cancer" projects train one model on one dataset, report a high
accuracy, and stop — which usually just measures overfitting. This project is built around
what makes a prognostic study **credible**:

1. **Two complementary data types ("multi-modal").** Genes tell you the tumor's molecular
   biology; the microscope slide tells you its physical structure. Real cancer-AI research
   shows fusing them predicts outcomes better than either alone — and almost no high-schooler
   attempts it.
2. **External validation across independent cohorts** for the genomics arm (METABRIC ↔
   TCGA-BRCA, two different measurement platforms). A model that survives this is real.
3. **A "does it beat standard-of-care?" test.** We ask whether the AI adds prognostic value
   *beyond* the staging clinicians already use — the question oncologists actually care about.
4. **Interpretability + biological sanity-checks.** We open the models and confirm they
   learned real biology, not noise.

## 2. The scientific question

> Can AI learn patterns from tumor **gene-expression** and **histopathology images** to
> predict breast-cancer survival; does each modality generalize; and does **combining both
> modalities improve prediction over either alone**, beyond standard clinical staging?

Hypotheses and success criteria are pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md)
**before** modeling — our main safeguard against fooling ourselves or fishing for results.

## 3. Data (100% public, no login for the core, no fabrication)

| Modality | Cohort / dataset | Size | Role |
|---|---|---|---|
| Gene expression + clinical | METABRIC (`brca_metabric`) | ~1,900 patients | Genomics train cohort |
| Gene expression + clinical | TCGA-BRCA (`brca_tcga_pan_can_atlas_2018`) | ~1,000 patients | Genomics external validation; multi-modal cohort |
| Histopathology (warm-up) | [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) | 9,109 images, 82 patients | Prove the image pipeline works |
| Histopathology (WSIs) | [TCGA-BRCA slides](https://www.cancerimagingarchive.net/collection/tcga-brca/) (TCIA/GDC) | 3,111 slides, ~1,098 patients | Image survival + fusion (matched to genes) |

Genomics via the [cBioPortal](https://www.cbioportal.org) API (see `src/fetch_data.py`).
All data sources, links, sizes, and citations are documented in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

**Endpoint:** Overall Survival (`OS_MONTHS`, `OS_STATUS`). Secondary: 5-year survival (binary).

## 4. Staged, de-risked plan

Each stage is a *complete, presentable deliverable on its own*. We only escalate complexity
once the previous stage works — so an ambitious project never collapses to nothing.

- **Stage A — Genes (foundation, ~90% built).** Gene-expression survival model;
  METABRIC↔TCGA external validation; "beats clinical staging?" test. *Guarantees a finished
  project.*
- **Stage B — Image warm-up (BreakHis).** Train a CNN (transfer learning) to classify
  benign vs malignant on clean, pre-cropped images. Proves the imaging pipeline works.
- **Stage C — Image survival (TCGA).** Take a tractable subset of TCGA slides → tile into
  patches → embed with a pretrained pathology model → pool per patient → predict survival.
- **Stage D — Fusion (headline).** Combine gene + image features for the *same* TCGA
  patients; test whether fusion beats either modality alone (ΔC-index with confidence
  intervals). Interpret what each modality contributes.

## 5. Methods (high level)

- **Genes:** penalized Cox regression / Random Survival Forest, nested cross-validation.
- **Images:** transfer learning (pretrained CNN) for BreakHis; tiling + pretrained
  feature extractor + multiple-instance/attention pooling for TCGA WSIs (on a free Colab/Kaggle GPU).
- **Fusion:** combine per-patient gene + image feature vectors into a survival model
  (late or joint fusion), optimized for the Cox objective.
- **Metrics:** Harrell's C-index (+95% CI) and time-dependent AUC; incremental value via
  likelihood-ratio / DeLong tests. Leakage guarded by strict train/test separation, nested
  CV, fixed seed (42), and a **shuffled-label negative control** (must score ~0.5).

## 6. Deliverables (your college-application artifact)

- 📦 This public, reproducible GitHub repo.
- 📊 Figures: survival curves, C-index comparison across modalities, fusion-vs-single plot,
  interpretability (SHAP / attention heatmaps on slides).
- 📝 A short written report / mini-paper; optional preprint.
- 🗒️ A 1-page abstract you can speak to fluently in essays and interviews.

## 7. Your role vs. the AI's role (read this)

The AI writes the code. **You own every scientific decision** — endpoint, why fusion might
help, what the limitations are, whether a result is real. For each stage the goal is that you
can explain, in plain English, *what* we did and *why*. If you can't, we stop and fix that.

## 8. How to run (Stage A, today)

```bash
cd breast-cancer-prognosis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/fetch_data.py        # pulls METABRIC clinical + gene expression
python src/eda.py               # first result: survival by subtype
```
Imaging stages (B–D) run on a free cloud GPU (Google Colab / Kaggle) — set up when we reach them.

## 9. Honest challenges & limitations

- Microscope slides are gigapixel; we process a **subset**, so the image/fusion analyses have
  fewer patients and wider confidence intervals.
- The fusion analysis is **TCGA-only** (only cohort with both modalities) — weaker external
  validation than the genomics arm; possible future check against CPTAC-BRCA.
- BreakHis is a *classification* warm-up, not a survival cohort — it validates the pipeline,
  not the survival claim.
- All data is retrospective and observational: **associations, not causation; a research
  artifact, not a clinical tool.**

## 10. Integrity principles

No fabricated or altered data. No cherry-picked metrics. Pre-registered analysis plan.
Negative results reported. Every number reproducible from public data with a fixed seed.
See [`PREREGISTRATION.md`](PREREGISTRATION.md) and the running [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md).
