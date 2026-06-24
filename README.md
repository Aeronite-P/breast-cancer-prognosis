# Cross-Cohort, Interpretable Gene-Expression Model for Breast-Cancer Prognosis

**One-line:** Train a machine-learning model on tumor gene-expression to predict breast-cancer survival, validate it on a *separate, independent* patient cohort, and test whether it adds real prognostic value **beyond standard clinical staging** — with every result reproducible from public data.

> Student researcher: 16 y/o · Summer 2026 · heavy AI-assisted, zero fabricated data.

---

## 1. Why this project is strong (and honest)

Most high-school "AI + cancer" projects train one model on one dataset, report a high accuracy, and stop. That is not science — it usually just measures overfitting. This project is built around the three things that make a prognostic-model study **credible**:

1. **External validation across two independent cohorts and two measurement platforms.**
   - **METABRIC** (~1,900 tumors, Illumina microarray)
   - **TCGA-BRCA** (RNA-seq, a totally separate set of patients)
   - We train on one and test on the other. A model that survives this is real; one that doesn't, we report honestly.
2. **A "does it beat the standard of care?" test.** Clinicians already stage tumors (grade, ER/HER2, age, stage). We test whether the gene model adds prognostic value *on top of* those variables — not just whether it works in isolation. This is the actual scientific question oncologists care about.
3. **Interpretability + biological sanity-checking.** We open the model (SHAP) and check whether the genes it relies on are known breast-cancer biology (e.g., *ESR1*, *ERBB2*, *MKI67*). If the model learned real biology, that is strong evidence it isn't fitting noise.

These three choices are the difference between a science-fair cliché and a project a college admissions reader (or a journal) takes seriously.

## 2. The scientific question

> Can an interpretable machine-learning model, trained on tumor gene-expression, predict long-term survival in breast cancer; does it **generalize to an independent cohort**; and does it add prognostic value **beyond standard clinical variables**?

Pre-registered hypotheses and success criteria live in [`PREREGISTRATION.md`](PREREGISTRATION.md). We commit to those *before* modeling so we can't move the goalposts — this is also our main anti-fabrication safeguard.

## 3. Data (100% public, no login, no fabrication)

| Cohort | Study ID | Patients | Expression | Role |
|---|---|---|---|---|
| METABRIC | `brca_metabric` | ~1,900 w/ expr+survival | Illumina microarray | Primary (train) |
| TCGA-BRCA | `brca_tcga_pan_can_atlas_2018` | ~1,000 | RNA-seq | External validation |

Source: [cBioPortal](https://www.cbioportal.org). Pulled via the public REST API (see `src/fetch_data.py`). Full transcriptome available via the [datasets page](https://www.cbioportal.org/datasets) tarballs.

**Endpoint:** Overall Survival (`OS_MONTHS`, `OS_STATUS`). Secondary: a binary 5-year-survival label (for an easy-to-present ROC/AUC story) among patients with adequate follow-up.

## 4. Method (high level)

```
clinical + expression  ──►  preprocess  ──►  three models, nested CV
                                              ├─ clinical-only  (baseline / "standard of care")
                                              ├─ genes-only      (penalized Cox / Random Survival Forest)
                                              └─ clinical + genes (does it add value?)
        │
        ▼
  external validation:  METABRIC ─train─►  TCGA ─test─►  (and reverse)
        │
        ▼
  interpret (SHAP) ──► top genes ──► cross-check vs known biology & PAM50/Oncotype signatures
```

Primary metric: **Harrell's C-index** (survival ranking) + time-dependent AUC. Incremental value tested with a likelihood-ratio test (nested Cox) and DeLong's test (AUCs). Overfitting controlled with **nested cross-validation**, fixed random seed (`config.RANDOM_SEED = 42`), and no test-set leakage.

## 5. Deliverables (this is your college-application artifact)

- 📦 **This public GitHub repo** — clean, reproducible, one command to rerun.
- 📊 **Figure set** — survival curves, C-index comparison, external-validation plot, SHAP gene importance.
- 📝 **Short written report / mini-paper** (~6–10 pages) you can submit to a student journal or post as a preprint (bonus, low extra cost).
- 🗒️ **1-page abstract** you can talk about fluently in essays and interviews.

## 6. Milestones (summer sprint)

- **Week 1 — Foundation.** Environment, data acquisition, clinical EDA, lock `PREREGISTRATION.md`, define cohorts/endpoints. *(scaffolding + clinical pull already done — see below)*
- **Week 2 — Baselines.** Expression preprocessing; clinical-only baseline; first cross-validated survival model within METABRIC.
- **Week 3 — The moat.** External validation METABRIC↔TCGA; incremental-value analysis; core figures.
- **Week 4 — Insight.** SHAP interpretability; biological cross-check; robustness/sensitivity analyses; limitations.
- **Week 5 — Ship.** Write-up, repo polish, abstract, optional preprint.

## 7. Your role vs. the AI's role (read this)

The AI writes the code. **You own every scientific decision** — which endpoint, why external validation matters, what the limitations are, whether a result is real. That ownership is exactly what a college reader probes in an interview, and it's what makes this *your* project. For each phase, the goal is that you can explain **what** we did and **why** in plain English. If you can't, stop and ask.

## 8. How to run

```bash
cd breast-cancer-prognosis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/fetch_data.py        # pulls METABRIC clinical + PAM50 expression
```

## 9. Integrity principles

No fabricated or altered data. No cherry-picked metrics. Pre-registered analysis plan. Negative results reported. Every number reproducible from public data with a fixed seed. See [`PREREGISTRATION.md`](PREREGISTRATION.md).
