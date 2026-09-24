# 📚 Study Guide — Breast-Cancer Prognosis Project

_A complete, plain-English reference for the biology, every stage, the results, and the tools used. Everything here is accurate to what was actually built._

---

## Part 1 — Breast Cancer: The Biology You Need

**What cancer is.** Cells normally grow and die in a controlled way. Cancer is when cells escape
that control and divide uncontrollably, forming a **tumor**. **Benign** tumors stay put;
**malignant** tumors invade nearby tissue and can **metastasize** (spread). Breast cancer usually
starts in the milk **ducts** (ductal) or **lobules** (lobular), and is either **in situ**
(contained) or **invasive** (spread into surrounding tissue).

**Why it's "many diseases."** Breast cancer is **heterogeneous** — biologically different from
patient to patient. That's *why there is no single "cure."* Subtypes are defined by which
**receptors/proteins** the tumor cells carry:
- **ER** — estrogen receptor
- **PR** — progesterone receptor
- **HER2** — a growth-signal receptor
- **Triple-negative** = ER−/PR−/HER2− (fewest treatment targets, aggressive)

**Molecular subtypes (PAM50 / "intrinsic" subtypes):**

| Subtype | Character | Prognosis |
|---|---|---|
| **Luminal A** | ER+, slow-growing | **Best** |
| **Luminal B** | ER+, faster-growing | Intermediate |
| **HER2-enriched** | HER2+, aggressive | Worse |
| **Basal-like** (≈ triple-negative) | aggressive | **Worst** |
| Normal-like / Claudin-low | less common categories | varies |

_(The survival-by-subtype figure showed exactly this: Luminal A on top, log-rank p < 0.001.)_

**Clinical staging (the "standard of care" a doctor uses):**
- **TNM** — **T**umor size, **N**odes (lymph nodes involved), **M**etastasis
- **Grade** (1–3) — how abnormal the cells look under a microscope
- **NPI** (Nottingham Prognostic Index) — a combined score (size + grade + nodes). _This is the
  strong clinical baseline the gene models had to beat._

**Treatment depends on subtype:** surgery ± chemo/radiation, **hormone therapy** for ER+ (e.g.
tamoxifen), **HER2-targeted** drugs for HER2+. This is *why* subtyping matters clinically.

**Gene expression.** Every cell has the same DNA, but **expression** = how *active* each gene is
(measured by its **mRNA** level). A tumor's expression profile is its **molecular fingerprint**.
**PAM50** is a 50-gene panel that defines the subtypes above.

**Histopathology.** A **biopsy** removes tumor tissue; it's sliced thin, stained with **H&E**
(pink/purple dye), and read under a microscope. Malignant tissue looks disorganized with
abnormal, crowded cells. This is the "physical appearance" view.

**Survival analysis.** **Prognosis** = predicting outcome. The endpoint is **overall survival**
(time from diagnosis to death). Key concept: **censoring** — many patients are still alive at last
contact, so you only know they survived *at least* X months. Survival methods handle that.

---

## Part 2 — The Stages (detailed synopsis)

### Foundation
- **Question:** Do a tumor's **genes** and its **appearance** predict survival — and does combining them beat either alone?
- **Data sources:** all public — **cBioPortal** (genes + clinical), **GDC/TCIA** (slide images), **BreakHis** (histology images).
- **Integrity:** a **pre-registered** analysis plan (methods fixed before results) + a running lab notebook.

### Stage A — Genomics survival _(done)_
- **Goal:** predict survival from gene expression; check it generalizes and beats clinical staging.
- **Data:** **METABRIC** (~1,980 patients, microarray) + **TCGA-BRCA** (~1,069, RNA-seq).
- **Tools/methods:** **Cox proportional-hazards model**, **5-fold cross-validation**, **C-index**
  (Harrell's concordance; 0.5 = chance), a **shuffled-label negative control**, **external
  validation** (train one cohort → test the other), a **bootstrap significance test**.
- **Results:**

  | Model | C-index |
  |---|---|
  | Clinical only | **0.662** |
  | Genes (PAM50) | 0.599 |
  | Clinical + genes | 0.666 |
  | Shuffled control | **0.512** ✅ (no leakage) |
  | External (METABRIC→TCGA) | **0.652** |

  Incremental value: ΔC = +0.005, 95% CI **[−0.004, +0.015] → not significant.**
- **Interpretation:** genes carry **real, generalizable** signal but **don't beat clinical staging.**
- **Limitations:** observational data (correlation, not causation); only 50 genes.

### Stage A+ — Full transcriptome _(done)_
- **Goal:** was 50 genes just too few? Test the *whole* transcriptome.
- **Data:** full expression matrices (~20,000 genes) for both cohorts.
- **Tools/methods:** **Random Survival Forest** (non-linear) + **ridge-penalized Cox** (linear) on
  the top-1,000 most-variable genes; same CV + external validation.
- **Results:** PAM50 (RSF) 0.608; **Top-1,000 (RSF) 0.599**; Top-1,000 (ridge Cox) 0.576. More
  genes did **not** help (ΔC vs PAM50 = −0.010, not significant). External validation held (0.651).
- **Interpretation:** the gene ceiling (~0.60) is **real, not a feature-selection artifact** — it
  still doesn't beat clinical. This *strengthens* the honest negative and motivates imaging.

### Stage B — Image classifier _(done)_
- **Goal:** prove images carry learnable signal — classify benign vs. malignant tissue.
- **Data:** **BreakHis** (~9,000 histology images, 82 patients).
- **Tools/methods:** **CNN via transfer learning** (**EfficientNet**, pretrained on everyday
  images), **patient-level split** (no patient in two sets → no cheating), scored by **AUC**.
  Ran on a cloud GPU.
- **Results:** **Test AUC 0.844**, accuracy 82% (sensitivity 85%, specificity 67%) on held-out patients.
- **Interpretation:** the image deep-learning pipeline works; histology carries strong signal.
- **Limitations:** classification, *not* survival; small patient count.

### Stage C — Imaging survival _(in progress — paused at 16/28 patients)_
- **Goal:** can a tumor's **appearance** predict *survival*? (The imaging counterpart to Stage A.)
- **Data:** **TCGA-BRCA whole-slide images** (gigapixel, ~0.4–1 GB each), matched to survival.
- **Tools/methods:** stream each slide → **tile** into patches → **ResNet50** features → **average
  into one vector per patient** → **Cox** survival model. Runs **locally on CPU** (uses
  **OpenSlide** to read slides).
- **The big lesson — a confound that was caught:** the shortcut of pulling *alive = small slides,
  dead = big slides* made **slide file size itself a predictor of survival** (size alone gave
  C-index **0.628**), which faked a bogus **0.907**. That result was rejected. **The fix:**
  **size-matched sampling** (pair each dead patient with a same-size alive patient) +
  **resume-capable downloads** → dead avg = alive avg = 372 MB, so size can no longer cheat.
- **Status:** downloading the size-matched slides (16/28). **Result pending** — when done, verify
  size-alone drops to ~0.5, then report the honest image-only C-index with a confidence interval.

### Stage D — Fusion _(planned)_
- **Goal:** combine each patient's **gene features + image features** into one model; test whether
  **together beats either alone**. Done on TCGA patients (who have both). The project's finale.

---

## Part 3 — Tools & Techniques (glossary)

**Concepts:**
- **Cox proportional-hazards model** — learns each feature's effect on risk; outputs a risk score for time-to-event data.
- **C-index (concordance)** — for pairs of patients, did the higher-risk one die sooner? 0.5 = chance, 1.0 = perfect.
- **Cross-validation (out-of-fold)** — every patient scored by a model that never trained on them.
- **Negative control (shuffled labels)** — scramble outcomes; a clean pipeline scores ~0.5 → proof of no leakage.
- **External validation** — train one cohort, test a separate one → proof of generalization.
- **PCA** — compresses many features into a few informative ones.
- **Random Survival Forest** — a non-linear survival model (many decision trees).
- **CNN / transfer learning** — image model; reuse a network pretrained on millions of images.
- **AUC / ROC** — image-classification metric (0.5 chance, 1.0 perfect).
- **Bootstrap confidence interval** — resampling to show how uncertain a number is.
- **Patient-level split** — keep all of one patient's data on one side → no leakage.

**Datasets:** METABRIC, TCGA-BRCA (via cBioPortal API + GDC), BreakHis.
**Libraries:** pandas, NumPy, scikit-learn, **scikit-survival**, lifelines, **PyTorch/torchvision**, **OpenSlide**, matplotlib.

---

## Part 4 — Numbers to Memorize

| Result | Value |
|---|---|
| Clinical staging (survival) | C-index **0.66** |
| Genes — PAM50 and full transcriptome | **~0.60** (don't beat clinical) |
| External validation (genes generalize) | **0.65** |
| Negative control (no leakage) | **0.51** |
| Image classifier (benign vs malignant) | **AUC 0.84** |
| Stage C confound (size alone) | 0.63 → why the naive result was rejected |

---

## Part 5 — How To Talk About It

**Pitch:** _"I built a machine-learning model to predict breast-cancer survival from two views of a
tumor — its gene expression and its microscope appearance — and I test whether combining them beats
either alone. The genomics arm generalizes across independent cohorts but doesn't beat clinical
staging; the imaging arm is where I'm testing for complementary signal."_

**Three power-moves in any Q&A:**
1. **Negative control** — _"I scrambled the labels; the model dropped to 0.51, proving no leakage."_
2. **External validation** — _"Trained on one cohort, tested on a totally separate one — it held."_
3. **The confound catch** — _"I caught that slide size was secretly predicting survival, so I
   rejected that result and re-ran it size-matched."_ ← signals a real scientist.

Say **"model,"** not "AI." Say it **classifies biopsy tissue / ranks survival risk** — it's a
**research model, not a clinical tool.**
