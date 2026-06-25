"""
model.py — Stage A: survival modeling on METABRIC (the project's foundation).

It answers, with cross-validation and an anti-leakage safeguard:
  1. How well do STANDARD CLINICAL variables predict survival?   (the "standard of care")
  2. How well do GENES (PAM50 expression) predict survival?
  3. Does CLINICAL + GENES beat clinical alone?
  4. NEGATIVE CONTROL: if we scramble the outcomes, does the gene model collapse to chance?
     (If it doesn't, we have data leakage and can't trust anything.)

Metric: Harrell's C-index (concordance). 0.5 = random guessing, 1.0 = perfect ranking.
We report it from data the model never saw during training (out-of-fold), which is the
honest way to estimate real-world performance.

Plain-English: a Cox model learns, for each patient feature, whether it pushes risk up or
down, then scores each patient's risk. The C-index asks: for pairs of patients, did the one
who actually died sooner get the higher risk score? Higher = better ranking.

Run:  python src/model.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# --- Feature sets ------------------------------------------------------------
# TRULY clinical only — deliberately NO molecular subtypes (those are gene-derived,
# which would blur the clinical-vs-genes comparison we care about).
CLIN_NUM = ["AGE_AT_DIAGNOSIS", "LYMPH_NODES_EXAMINED_POSITIVE", "NPI"]
CLIN_CAT = ["ER_IHC", "HER2_SNP6", "INFERRED_MENOPAUSAL_STATE"]
GENES = config.PAM50

N_SPLITS = 5
PENALIZER = 0.1          # small ridge penalty: stabilizes Cox with many correlated genes
SEED = config.RANDOM_SEED


def load_cohort(prefix: str = "metabric"):
    """Load a merged table and build a clean analysis cohort with survival labels.

    `prefix` selects the cohort file, e.g. "metabric" or "tcga".
    """
    df = pd.read_csv(os.path.join(config.PROCESSED_DIR, f"{prefix}_merged.csv"),
                     index_col="patientId")
    # survival labels
    df["duration"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["event"] = df["OS_STATUS"].astype(str).str.startswith("1").astype(int)
    # numeric clinical columns arrive as strings -> coerce (only those present in this cohort)
    for c in CLIN_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # keep patients with a valid (positive) survival time and complete gene data
    genes_present = [g for g in GENES if g in df.columns]
    df = df.dropna(subset=["duration"] + genes_present)
    df = df[df["duration"] > 0]
    return df, genes_present


def make_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    """Impute + scale numerics; impute + one-hot categoricals. Fit on TRAIN only (no leakage)."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    parts = []
    if num_cols:
        parts.append(("num", num, num_cols))
    if cat_cols:
        parts.append(("cat", cat, cat_cols))
    return ColumnTransformer(parts)


def cv_cindex(df, num_cols, cat_cols, shuffle_labels=False):
    """Cross-validated C-index. Preprocessing is fit inside each fold on training data only.

    shuffle_labels=True permutes the (duration, event) outcomes across patients, breaking
    any real signal — the negative control. A correct pipeline then scores ~0.5.
    """
    work = df.copy().reset_index(drop=True)
    if shuffle_labels:
        perm = np.random.RandomState(SEED).permutation(len(work))
        work[["duration", "event"]] = work[["duration", "event"]].iloc[perm].values

    X = work[num_cols + cat_cols]
    dur = work["duration"].to_numpy(dtype=float)
    evt = work["event"].to_numpy(dtype=int)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(work), np.nan)   # out-of-fold predictions
    fold_scores = []

    for tr, te in kf.split(X):
        pre = make_preprocessor(num_cols, cat_cols)
        Xtr = pre.fit_transform(X.iloc[tr])
        Xte = pre.transform(X.iloc[te])
        cols = [f"v{i}" for i in range(Xtr.shape[1])]

        train_df = pd.DataFrame(Xtr, columns=cols)
        train_df["duration"], train_df["event"] = dur[tr], evt[tr]

        cph = CoxPHFitter(penalizer=PENALIZER, l1_ratio=0.0)
        cph.fit(train_df, duration_col="duration", event_col="event")

        # partial hazard: higher = higher risk = expected SHORTER survival.
        # concordance_index wants "higher = longer survival", so we negate.
        risk = cph.predict_partial_hazard(pd.DataFrame(Xte, columns=cols)).to_numpy()
        score = -risk
        oof[te] = score
        fold_scores.append(concordance_index(dur[te], score, evt[te]))

    pooled = concordance_index(dur, oof, evt)
    return pooled, float(np.mean(fold_scores)), float(np.std(fold_scores))


# --- Helpers reused by external validation (src/validate_external.py) --------

def fit_cox(df, num_cols, cat_cols):
    """Fit preprocessor + Cox on a FULL cohort (no CV). Returns (preprocessor, model, cols)."""
    pre = make_preprocessor(num_cols, cat_cols)
    Xt = pre.fit_transform(df[num_cols + cat_cols])
    cols = [f"v{i}" for i in range(Xt.shape[1])]
    train_df = pd.DataFrame(Xt, columns=cols)
    train_df["duration"] = df["duration"].to_numpy(dtype=float)
    train_df["event"] = df["event"].to_numpy(dtype=int)
    cph = CoxPHFitter(penalizer=PENALIZER, l1_ratio=0.0)
    cph.fit(train_df, duration_col="duration", event_col="event")
    return pre, cph, cols


def external_cindex(train_df, test_df, num_cols, cat_cols):
    """Train on train_df, evaluate C-index on a SEPARATE test_df (external cohort)."""
    pre, cph, cols = fit_cox(train_df, num_cols, cat_cols)
    Xt = pre.transform(test_df[num_cols + cat_cols])
    risk = cph.predict_partial_hazard(pd.DataFrame(Xt, columns=cols)).to_numpy()
    dur = test_df["duration"].to_numpy(dtype=float)
    evt = test_df["event"].to_numpy(dtype=int)
    return concordance_index(dur, -risk, evt)


def oof_scores(df, num_cols, cat_cols, shuffle_labels=False):
    """Like cv_cindex but RETURNS the out-of-fold risk scores (higher = better survival),
    aligned to df row order — so two models' scores can be compared patient-by-patient."""
    work = df.copy().reset_index(drop=True)
    if shuffle_labels:
        perm = np.random.RandomState(SEED).permutation(len(work))
        work[["duration", "event"]] = work[["duration", "event"]].iloc[perm].values
    X = work[num_cols + cat_cols]
    dur = work["duration"].to_numpy(dtype=float)
    evt = work["event"].to_numpy(dtype=int)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(work), np.nan)
    for tr, te in kf.split(X):
        pre = make_preprocessor(num_cols, cat_cols)
        Xtr = pre.fit_transform(X.iloc[tr])
        Xte = pre.transform(X.iloc[te])
        cols = [f"v{i}" for i in range(Xtr.shape[1])]
        tdf = pd.DataFrame(Xtr, columns=cols)
        tdf["duration"], tdf["event"] = dur[tr], evt[tr]
        cph = CoxPHFitter(penalizer=PENALIZER, l1_ratio=0.0)
        cph.fit(tdf, duration_col="duration", event_col="event")
        risk = cph.predict_partial_hazard(pd.DataFrame(Xte, columns=cols)).to_numpy()
        oof[te] = -risk
    return oof, dur, evt


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIG_DIR, exist_ok=True)
    df, genes = load_cohort()
    print(f"Analysis cohort: {len(df)} patients | events (deaths): {int(df['event'].sum())} "
          f"| genes used: {len(genes)}\n")

    specs = [
        ("Clinical only",        CLIN_NUM,          CLIN_CAT, False),
        ("Genes only (PAM50)",   genes,             [],       False),
        ("Clinical + Genes",     CLIN_NUM + genes,  CLIN_CAT, False),
        ("Genes — SHUFFLED (control)", genes,       [],       True),
    ]

    rows = []
    for name, num_cols, cat_cols, shuffled in specs:
        pooled, mean_c, std_c = cv_cindex(df, num_cols, cat_cols, shuffle_labels=shuffled)
        rows.append({"model": name, "c_index_oof": pooled,
                     "c_index_foldmean": mean_c, "c_index_foldstd": std_c})
        print(f"{name:30s}  C-index = {pooled:.3f}   (per-fold {mean_c:.3f} ± {std_c:.3f})")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(config.RESULTS_DIR, "metrics_stageA.csv"), index=False)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#999999"]
    ax.bar(res["model"], res["c_index_oof"], yerr=res["c_index_foldstd"],
           capsize=5, color=colors)
    ax.axhline(0.5, ls="--", color="red", label="random (0.5)")
    ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("C-index (out-of-fold)")
    ax.set_title("METABRIC survival prediction — Stage A\nhigher = better; control should sit at ~0.5")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out = os.path.join(config.FIG_DIR, "stageA_cindex.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved metrics -> results/metrics_stageA.csv")
    print(f"Saved figure  -> {out}")


if __name__ == "__main__":
    main()
