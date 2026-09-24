# Publication Roadmap — Breast-Cancer Multi-Modal Prognosis

Tracking everything between the current result and a submitted paper. Check items off as done.

## ✅ Done
- [x] Stage A — clinical + gene survival model; cross-cohort external validation (METABRIC→TCGA 0.652)
- [x] Stage B — image pipeline validated on BreakHis
- [x] Stage C — imaging survival; slide-size confound caught; Phikon + attention-MIL (0.603)
- [x] Stage D/E/F/G — fusion, redundancy (complementary, r=0.22), late fusion 0.639 (+0.03 trend)
- [x] Lab notebook through Entry 9; pre-registration; study guide; clean GitHub repo

## Phase 1 — Finish the science
- [x] **1.1 Clinical baseline in fusion** — clinical alone 0.632 (best single); full fusion 0.666, +0.034 vs clinician (trend). No single molecular arm beats staging.
- [x] **1.2 External validation** — no 2nd WSI breast cohort exists on GDC → did **leave-site-out** (25 institutions): images 0.577 / genes 0.595 / late 0.607, all CIs clear 0.5. Signal generalizes across sites. (Gene arm also has true cross-cohort METABRIC→TCGA 0.652.)
- [ ] 1.3 Negative controls + robustness — shuffled-label ~0.5; multi-seed / tile-count stability
- [ ] 1.4 Richer metrics — time-dependent AUC, calibration, KM risk groups + log-rank, formal C-index test
- [ ] 1.5 Interpretability — gene SHAP + attention heatmaps
- [ ] 1.6 (optional) molecular-subtype stratification

## Phase 2 — Figures & tables
- [ ] Fig 1 pipeline schematic · Fig 2 C-index bars+CIs · Fig 3 KM risk groups · Fig 4 attention heatmaps · Fig 5 gene SHAP · Fig 6 calibration/AUC
- [ ] Table 1 cohort characteristics · Table 2 performance summary · Supplement (hyperparams, sensitivity, attrition)

## Phase 3 — Manuscript (user writes; AI scaffolds + disclosed)
- [ ] Title, Abstract, Intro, Methods, Results, Discussion, Conclusion
- [ ] References, data/code availability, author contributions, **AI-use disclosure**, ethics, conflicts

## Phase 4 — Reproducibility / open science
- [ ] Repo README re-runnable end-to-end · pinned deps · Zenodo DOI · data manifest

## Phase 5 — Venue & format
- [ ] Target: **JEI** (needs a named **mentor** — hard requirement) · optional preprint · format to guidelines · cover letter

## Phase 6 — Submit & peer review
- [ ] Submit → revise-and-resubmit (1–2 rounds) → accept

## 🚦 Gating (what actually decides acceptance)
1. External validation (1.2) — trend → claim
2. A mentor (required for JEI)
3. Honest framing + AI disclosure
4. Standard clinical figures (KM, Table 1, calibration)
