# Pre-Registration — Analysis Plan (lock before modeling)

Pre-registering means writing down our hypotheses, methods, and success criteria
**before** we look at modeling results. It is the single most important thing that
separates real science from "I tried things until a number looked good." We commit
to this plan; if we deviate, we document the deviation and why.

_Status: DRAFT — to be finalized at end of Week 1, before any model is trained._

## 1. Question
Does an interpretable ML model on tumor gene-expression predict breast-cancer
overall survival, generalize to an independent cohort, and add value beyond
standard clinical variables?

## 2. Cohorts
- **Discovery / training:** METABRIC (`brca_metabric`), patients with both expression and survival.
- **External validation:** TCGA-BRCA (`brca_tcga_pan_can_atlas_2018`).
- Both pulled from cBioPortal public API. No private data.

## 3. Endpoint
- **Primary:** Overall Survival — `OS_MONTHS` (time), `OS_STATUS` (event = DECEASED).
- **Secondary (binary):** alive vs. deceased at 5 years, restricted to patients with
  ≥ 60 months follow-up OR an event before 60 months (avoids censoring bias).

## 4. Features
- **Clinical baseline:** age at diagnosis, tumor grade, tumor stage/size, ER/HER2 status,
  nodal status (subset available in both cohorts; final list fixed in Week 1 after EDA).
- **Genes:** start from the PAM50 panel (literature-grounded, 50 genes), then expand to a
  data-driven signature selected *only on the training fold* (no leakage).

## 5. Models (pre-specified)
1. **Clinical-only** (Cox proportional hazards) — the "standard of care" baseline.
2. **Genes-only** — penalized Cox (elastic net) and Random Survival Forest.
3. **Clinical + genes** — the combined model.
- Feature selection and hyperparameters tuned **inside nested cross-validation** on training data only.

## 6. Primary analyses & success criteria
- **A. Discrimination:** Harrell's C-index, reported with 95% CI (bootstrap), in
  cross-validation AND in the external cohort.
  - *Pre-specified bar:* genes-only beats random (C-index 95% CI lower bound > 0.5)
    and is reproducible in the external cohort.
- **B. Incremental value (the key test):** does `clinical + genes` beat `clinical-only`?
  - Likelihood-ratio test (nested Cox); ΔC-index with CI.
  - *Pre-specified:* report the result **whether positive or negative.** A well-done
    "genes add little beyond staging" is still a publishable, honest finding.
- **C. Generalization:** C-index must not collapse on the external cohort beyond a
  pre-stated tolerance; cross-platform drop is expected and will be discussed, not hidden.

## 7. Interpretability
- SHAP values on the best model; report top genes and direction of effect.
- Cross-check top genes against known breast-cancer biology and PAM50/Oncotype DX genes.
- This is a sanity check, not a primary outcome.

## 8. Controls against fooling ourselves
- Fixed random seed (42) everywhere.
- Strict train/validation/test separation; **no test data touches preprocessing or selection.**
- Nested cross-validation for any tuning.
- Report all pre-specified metrics, including negative ones.
- A "negative control" run (shuffled survival labels) should yield ~0.5 C-index; if it
  doesn't, we have leakage and must fix it before trusting anything.

## 9. Known limitations (stated up front)
- Retrospective, observational data; no causal claims.
- Microarray vs. RNA-seq platform differences (batch effects).
- Cohorts differ in era, treatment, demographics.
- A model is a research artifact, **not** a clinical tool.

## 10. Deviation log
_(Record any change from this plan, with date and reason.)_
- _none yet_
