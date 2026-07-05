# 🧬🔬 Multi-Modal Breast-Cancer Prognosis

### Do a tumor's **genes** and its **microscope appearance** predict survival better *together* than either alone?

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-success)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Data](https://img.shields.io/badge/data-public%20%7C%20reproducible-blue)

> A reproducible, **externally-validated** machine-learning study that combines tumor **gene expression** and **histopathology imaging** to model breast-cancer survival — built entirely on public data, with a pre-registered analysis plan and honest reporting.
>
> **Researcher:** Shiv Prahalathan · Summer 2026
> **Status:** Stage A (genomics) & Stage B (imaging) complete · Stage C (imaging survival) next.

---

## 📌 TL;DR

Breast cancer is many diseases, not one. This project asks whether AI can read **two complementary "views" of a tumor** — its molecular fingerprint (which genes are active) and its physical appearance under a microscope — to predict how patients fare, and whether **fusing both views beats either alone**. The genomics model **generalizes across two independent patient cohorts on two different platforms**, and the imaging model distinguishes benign from malignant tissue at **0.84 AUC**.

## ✨ Highlights

- 🔁 **External validation** across two independent cohorts *and* two measurement platforms (microarray ↔ RNA-seq) — C-index **0.65**. The test most student projects skip.
- 🛡️ **Leakage-proofed** with a shuffled-label **negative control** → **0.51** (chance), proving the results aren't an artifact.
- 🖼️ **Image classifier** (transfer-learning CNN): benign vs. malignant at **0.84 AUC** on held-out patients.
- 📋 **Pre-registered** analysis plan + **honest reporting** — including a documented negative result.
- ♻️ **Fully reproducible** from public data with a fixed seed.

## ❓ The Question

> Can machine learning predict breast-cancer survival from tumor **gene expression** and **histopathology images**, does each modality **generalize** to unseen cohorts, and does **combining both** add value **beyond standard clinical staging**?

## 🧪 Why this is rigorous (not just another notebook)

1. **External validation.** Models are trained on one cohort and tested on a *completely separate* one, on different hardware — the strongest evidence a model learned real biology, not dataset quirks.
2. **Negative control.** Survival labels are shuffled to destroy all signal; a correct pipeline then scores ~0.5. Ours did → no leakage.
3. **"Beats standard-of-care?" test.** We measure whether the model adds value *beyond* the staging clinicians already use — the question that actually matters.
4. **Pre-registration + honesty.** Hypotheses and success criteria are fixed *before* modeling (`PREREGISTRATION.md`); negative results are reported, not hidden.

## 📊 Data (100% public)

| Dataset | Modality | Size | Role |
|---|---|---|---|
| **METABRIC** | gene expression (microarray) + survival | ~1,980 patients | Genomics training cohort |
| **TCGA-BRCA** | gene expression (RNA-seq) + survival | ~1,069 patients | Independent external validation |
| **BreakHis** | histopathology images | ~9,000 images | Benign/malignant image model |

Gene data via the [cBioPortal](https://www.cbioportal.org) API. Full provenance + citations in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## ⚙️ Methods

- **Survival modeling:** Cox proportional-hazards with cross-validation; scored by **Harrell's C-index** (0.5 = chance, 1.0 = perfect).
- **Generalization:** train on one cohort → test on another (cross-platform).
- **Integrity checks:** nested train/test separation, fixed seed (42), and a shuffled-label negative control.
- **Imaging:** transfer-learning **CNN (EfficientNet)** with a **patient-level split** (no patient in two sets); scored by **AUC**.

## 📈 Results

**Genomics — survival prediction (C-index):**

| Model | C-index |
|---|---|
| Clinical only | 0.66 |
| Genes (PAM50, 50) | 0.60 |
| **Genes (full transcriptome, ~20k)** | **0.60** |
| **External validation** (train METABRIC → test TCGA) | **0.65** |
| Shuffled-label control | **0.51** ✅ |

> **Honest finding:** gene expression carries *real, generalizable* prognostic signal (~0.60, external 0.65) but does **not** beat standard clinical staging — and this holds even when scaling from the 50-gene panel to the **full transcriptome** (~20k genes) with both a random forest and a penalized Cox: performance plateaus at ~0.60. So the ceiling is *real*, not a feature-selection artifact — which is exactly why the imaging arm is worth pursuing: physical tumor structure may carry the complementary signal the molecules can't.

**Imaging — benign vs. malignant:** **AUC 0.84**, accuracy 82% on held-out patients.

*(Figures in [`results/figures/`](results/figures): survival-by-subtype, model comparison, external validation, ROC.)*

## 🗂️ Repository Structure

```
├── README.md              # you are here
├── PREREGISTRATION.md     # analysis plan locked before modeling
├── LAB_NOTEBOOK.md        # dated log of every step (what / why / assumptions / limits)
├── config.py              # studies, gene panel, seed
├── src/
│   ├── fetch_data.py      # pull public data from cBioPortal
│   ├── eda.py             # survival-by-subtype analysis
│   ├── model.py           # Stage A: survival models + negative control
│   └── validate_external.py  # cross-cohort validation + significance test
├── notebooks/
│   └── stage_b_breakhis_colab.ipynb   # Stage B: image classifier (Colab/GPU)
├── docs/                  # script docs, data sources, Colab guide
└── results/               # metrics + figures
```

## ▶️ Reproduce It

```bash
git clone https://github.com/Aeronite-P/breast-cancer-prognosis.git
cd breast-cancer-prognosis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_data.py        # download public data
python src/eda.py               # survival-by-subtype result
python src/model.py             # Stage A models + negative control
python src/validate_external.py # cross-cohort external validation
```
The imaging model (Stage B) runs on a free GPU — see [`docs/STAGE_B_COLAB.md`](docs/STAGE_B_COLAB.md).

## 🗺️ Roadmap

- [x] **Stage A** — genomic survival model + external validation
- [x] **Stage B** — histopathology image classifier
- [ ] **Stage C** — survival prediction from TCGA whole-slide images
- [ ] **Stage D** — **fusion**: combine genes + images and test the headline question

## ⚠️ Limitations

- Retrospective, observational data → **associations, not causation.** A research model, **not a clinical or screening tool.**
- The gene and image models are **not yet fused** (Stage D).
- Narrow features so far (PAM50 = 50 genes); TCGA has relatively few events; BreakHis covers 82 patients.
- Planned fusion will be single-cohort (TCGA), so its external validation will be weaker.

## 📚 Citation & Data

If you build on this, please cite the source studies (METABRIC: Curtis et al. 2012, Pereira et al. 2016; TCGA-BRCA; BreakHis: Spanhol et al. 2016) and cBioPortal (Cerami et al. 2012; Gao et al. 2013). See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## 📄 License

MIT — see [`LICENSE`](LICENSE).
