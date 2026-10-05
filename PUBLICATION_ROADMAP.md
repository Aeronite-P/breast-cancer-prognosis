# Publication Roadmap — Breast-Cancer Multi-Modal Prognosis

Tracking everything between the current result and a submitted paper. Check items off as done.

## ✅ Done
- [x] Stage A — clinical + gene survival model; cross-cohort external validation (METABRIC→TCGA 0.652)
- [x] Stage B — image pipeline validated on BreakHis
- [x] Stage C — imaging survival; slide-size confound caught; Phikon + attention-MIL (0.603)
- [x] Stage D/E/F/G — fusion, redundancy (complementary, r=0.22), late fusion 0.639 (+0.03 trend)
- [x] Lab notebook through Entry 9; pre-registration; clean GitHub repo

## Phase 1 — Finish the science
- [x] **1.1 Clinical baseline in fusion** — clinical alone 0.632 (best single); full fusion 0.666, +0.034 vs clinician (trend). No single molecular arm beats staging.
- [x] **1.2 External validation** — no 2nd WSI breast cohort exists on GDC → did **leave-site-out** (25 institutions): images 0.577 / genes 0.595 / late 0.607, all CIs clear 0.5. Signal generalizes across sites. (Gene arm also has true cross-cohort METABRIC→TCGA 0.652.)
- [x] 1.3 Negative controls + robustness — shuffled labels 0.45–0.53 (pass); 5 seeds 0.660 ± 0.008. _(tile-count stability still open)_
- [x] 1.4 Richer metrics — time-AUC, KM + log-rank + HR, **calibration (3/5-yr, Brier) done on n = 1,033**. Paired-bootstrap tests for time-AUC and Brier differences + decision curves done (A5, Entry 14): all favour the full model, none significant.
- [x] 1.8 **Amendment A (PREREGISTRATION §12), Entry 14** — A1 image is an independent prognostic factor beyond age + TNM (HR 1.40/SD, LRT p = 2.5e-5); A2 nested-CV stacking 0.738 (+0.017, n.s.); A3 subgroups n.s.; A5 time-AUC/Brier/decision curves favour the full model, n.s. A4: slide size alone C 0.580 in the full cohort, but image unaffected when size is adjusted for (HR 1.38/SD, p = 4.4e-5). `src/stage_m_added_value.py`
- [~] 1.9 **Amendment B (PREREGISTRATION §13), Entry 15 — tissue-composition shortcut test.** B3 primary passed: image stays independent beyond clinical + tissue composition (HR 1.29/SD, p = 0.0025; k = 12: 1.28, p = 0.004); composition alone C 0.532; attention concentrates on one cross-hospital cluster (2.4×). Most k-means clusters = hospital staining style (descriptive). B4: 60 crops verified (cosine 1.0000); attention cluster 2 = fibrous stroma (non-expert), no cluster meets the fat/background rule → no filtered retrain; but heatmap fat tiles also fall in cluster 2, so a fat-specific shortcut inside it is not yet ruled out. `src/stage_n_tissue.py`
- [x] 1.10 **Amendment C (PREREGISTRATION §14), Entry 16** — C1: no fat sub-cluster; a non-tissue sub-cluster (2b: dark/red background, 0.9% of tiles) was found and removed → image HR 1.37/SD, p = 5.5e-5 (C 0.637). C2: 100 tiles HR 1.41; all 5 weightings +0.003…+0.019, none significant; blank-slide patient irrelevant. C3: gene SHAP (fig14), 14/15 top genes match univariate direction. `src/stage_o_followups.py`
- [x] 1.7 **Full cohort (n = 1,033, 146 deaths, 39 sites)** — Entry 13: images 0.641 (0.640 unseen hospitals); clinical 0.722; full model 0.724 (+0.003 vs clinical, not significant) → the n = 244 trend did not replicate. `src/stage_l_fullcohort.py`
- [~] 1.5 Interpretability — attention heatmaps done (Fig 6: high-risk attention on crowded nuclei; low-risk + missed case on fat/edges). Attention by tissue cluster quantified for all 1,033 (B2, Entry 15). Gene SHAP done (C3); fat could not be isolated as its own cluster (C1).
- [ ] 1.6 (optional) molecular-subtype stratification

## Phase 2 — Figures & tables
- [x] Fig 2 C-index forest (random + site-held-out) · Fig 3 KM risk groups · Fig 4 time-AUC · Fig 5 complementarity — `src/make_figures.py`
- [x] Fig 6 attention heatmaps (out-of-fold; 4 examples incl. a missed case) — `src/stage_k_heatmaps.py`
- [x] Calibration (fig7) · decision curves (fig9) · subgroups (fig10) · attention by cluster (fig11) · cluster examples (fig12)
- [x] Fig 1 pipeline schematic · gene SHAP (fig14) · fig13 sub-clusters · Supplement: Fig S1 cohort flow, Table S1 settings, Table S2 sensitivity (`src/make_paper_figures.py`)
- [x] Table 1 cohort characteristics (`results/table1_cohort.md`) · performance table (`results/metrics_final.csv`)
- [x] Supplement (hyperparams, sensitivity, attrition) — `results/supplement/`, figS1–S3

## Phase 3 — Manuscript (user writes; AI scaffolds + disclosed)
- [ ] Title, Abstract, Intro, Methods, Results, Discussion, Conclusion
- [ ] References, data/code availability, author contributions, **AI-use disclosure**, ethics, conflicts

## Phase 4 — Reproducibility / open science
- [~] Reproducibility package built locally (`reproducibility/`, `src/make_manifest.py`): pinned deps (pip freeze), environment + Phikon revision, 1,033-patient manifest with feature SHA-256s, results checksums GDC slide file ids + MD5s filled (1,026 exact, 7 ambiguous listed); 74 results files checksummed _(not yet pushed — user will push later)_
- [ ] Repo README re-runnable end-to-end · Zenodo DOI

## Phase 5 — Venue & format
- [ ] Target: **JEI** (needs a named **mentor** — hard requirement) · optional preprint · format to guidelines · cover letter

## Phase 6 — Submit & peer review
- [ ] Submit → revise-and-resubmit (1–2 rounds) → accept

## 🚦 Gating (what actually decides acceptance)
1. External validation (1.2) — trend → claim
2. A mentor (required for JEI)
3. Honest framing + AI disclosure
4. Standard clinical figures (KM, Table 1, calibration)
