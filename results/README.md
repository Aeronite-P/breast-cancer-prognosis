# Results — what's where

Every file here is produced by a script (see [`../docs/SCRIPTS.md`](../docs/SCRIPTS.md)) and fingerprinted in
[`../reproducibility/results_checksums.sha256`](../reproducibility/results_checksums.sha256).
Check them all with `shasum -a 256 -c reproducibility/results_checksums.sha256` from the repository root.

## The two cohorts
| Where | Cohort | Lab notebook |
|---|---|---|
| `results/` (top level) and `results/figures/fig2–fig6` | **n = 244**, size-matched case/control sample (50% deaths) — the July–September results (clinical columns corrected 2026-10-21, Entry 17) | Entries 9–12, 17 |
| `results/fullcohort/` and `results/figures/fullcohort/` | **n = 1,033**, every eligible TCGA-BRCA patient (146 deaths, 39 hospitals) — the main study | Entries 13–17 |

## Full cohort (`fullcohort/`)
| File / folder | What it holds |
|---|---|
| `oof_risks_full.csv` | every out-of-fold risk score (images, genes, clinical; random split, unseen hospitals, controls, 5 seeds) |
| `metrics_final.csv`, `delta_vs_clinical.csv` | C-index with 95% CI per model; difference vs clinical |
| `robustness.csv`, `negative_control_repeats.csv` | scrambled-outcome controls and seed stability |
| `km_risk_groups.csv`, `time_auc.csv` | Kaplan-Meier hazard ratios; accuracy at 1–8 years |
| `calibration_*.csv` | 3- and 5-year calibration and Brier scores |
| `frozen_*.csv` | models trained on the original 244, tested on the 789 new patients |
| `comparison_244_vs_1033.csv` | the same models on both cohorts |
| `gdc_dx_slides.csv` | every TCGA-BRCA diagnostic slide on GDC (id, size, MD5) |
| `added_value/` | Amendment A: beyond staging (A1), learned weights (A2), subgroups (A3), slide size (A4), time-AUC / Brier / decision curves (A5) |
| `tissue/` | Amendment B: tile clusters, attention by cluster, tissue-composition test, cluster labels |
| `followups/` | Amendment C: fat / non-tissue sub-clusters (C1), robustness (C2), gene SHAP (C3) |
| `healthcheck/` | Pre-writing audit (Entry 17): H4 test, clinical-recompute checks, scan magnification per slide, Amendment D (D1–D5) |

## Figures (`figures/`)
| Figure | Shows |
|---|---|
| `fig1_pipeline.png` | how a patient becomes a risk score, and how it was tested |
| `fullcohort/fig2–fig5` | C-index forest plot, Kaplan-Meier, time-AUC, image-vs-gene complementarity (n = 1,033) |
| `fig6_attention_heatmaps.png`, `fig6b_…explained.png` | where the image model looks (4 examples; journal and presentation versions) |
| `fullcohort/fig7–fig10` | calibration, 244 vs 1,033, decision curves, subgroups |
| `fullcohort/fig11–fig14` | attention by tissue cluster, cluster examples, cluster-2 sub-clusters, gene SHAP |
| `figS1–figS3` | supplement: cohort flow, Table S1 settings, Table S2 sensitivity analyses |
| `metabric_km_by_subtype.png`, `stageA_cindex.png`, `external_validation.png`, `boost_genes.png`, `stageB_breakhis_roc.png` | early stages (June–July) |

## Supplement (`supplement/`)
`table_s1_settings.md` (every model and analysis setting) · `table_s2_sensitivity.md` (every sensitivity result).
