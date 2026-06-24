# Lab Notebook

A running, honest log of the project. **Every major step gets an entry** in the same
four-part format, so anyone (including future-you, a science-fair judge, or a college
interviewer) can follow exactly what was done and why — and so we never quietly move the
goalposts.

> Format for each entry:
> - **What was done** — the concrete actions/code.
> - **Why it matters** — the scientific reason.
> - **Assumptions** — what we took for granted (and might be wrong).
> - **Limitations** — what this step does *not* show; honest caveats.

---

## Entry 1 — 2026-06-24 · Week 1: Foundation, data, first result

**What was done**
- Defined the research question and locked the analysis plan in
  [`README.md`](README.md) and [`PREREGISTRATION.md`](PREREGISTRATION.md) *before* any modeling.
- Built `src/fetch_data.py` and pulled real data from the public cBioPortal API:
  METABRIC = **2,509** patients (clinical), **1,980** with matched gene-expression
  (50 PAM50 genes) + survival → merged modeling table `data/processed/metabric_merged.csv`.
- Built `src/eda.py`: produced a cohort summary and a Kaplan–Meier survival curve
  stratified by molecular subtype → `results/figures/metabric_km_by_subtype.png`.

**Key numbers (real, reproducible):** 1,980 patients; 1,143 deaths, 837 alive; median
follow-up 116 months. Survival differs by subtype, **log-rank p < 0.001**, with Luminal A
surviving longest — the ordering expected from breast-cancer biology.

**Why it matters**
- Pre-registering protects the project's integrity (no fishing for good-looking numbers).
- Getting clean, real, public data with a working reproducible pipeline is the foundation
  everything else stands on.
- The Kaplan–Meier result is a *sanity check*: because the known-biology ordering shows up,
  we have evidence the data and code are correct before we build ML on top.

**Assumptions**
- `OS_STATUS` values beginning with `"1"` mean a death event; others are censored (alive
  or lost to follow-up). This matches cBioPortal's coding (`1:DECEASED` / `0:LIVING`).
- The PAM50 gene panel is a reasonable, literature-grounded starting feature set.
- Patient ID = sample ID mapping in METABRIC is 1:1 (verified via the samples endpoint).

**Limitations**
- This is *only* a descriptive result — no model, no prediction yet. A significant log-rank
  test shows subtypes differ; it says nothing about predictive accuracy.
- We used 50 genes, not the full transcriptome (that comes later).
- METABRIC is a single, retrospective, observational cohort; no causal or clinical claims.
- 528 patients lack a survival status and are dropped from survival analysis — we should
  later check those dropped patients aren't systematically different (selection bias check).
