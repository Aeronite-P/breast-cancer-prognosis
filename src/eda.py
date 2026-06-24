"""
eda.py — First look at the data + first real result.

Produces:
  1. A cohort summary (n, events, follow-up).
  2. A Kaplan-Meier survival figure stratified by molecular subtype.
  3. A log-rank test: do the subtypes have significantly different survival?

This is a genuine, presentable result on day 1 (no modeling yet) and it confirms
the survival signal in the data is real before we build any ML on top of it.

Run:  python src/eda.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")  # write figures to file, no GUI needed
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

SUBTYPE_CANDIDATES = ["CLAUDIN_SUBTYPE", "PAM50_SUBTYPE", "THREEGENE", "INTCLUST"]


def load_survival(prefix: str = "metabric") -> pd.DataFrame:
    path = os.path.join(config.PROCESSED_DIR, f"{prefix}_merged.csv")
    df = pd.read_csv(path, index_col="patientId")
    df["time"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["event"] = df["OS_STATUS"].astype(str).str.startswith("1").astype(int)
    df = df.dropna(subset=["time"])
    df = df[df["time"] >= 0]
    return df


def main():
    os.makedirs(config.FIG_DIR, exist_ok=True)
    df = load_survival("metabric")

    # --- cohort summary ---
    print("=== METABRIC cohort summary ===")
    print(f"patients (with survival): {len(df)}")
    print(f"deaths (events):          {int(df['event'].sum())}")
    print(f"alive (censored):         {int((1 - df['event']).sum())}")
    print(f"median follow-up:         {df['time'].median():.1f} months")

    subtype = next((c for c in SUBTYPE_CANDIDATES if c in df.columns), None)
    if subtype is None:
        print("No subtype column found; plotting overall KM only.")

    # --- Kaplan-Meier figure ---
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    if subtype:
        groups = [g for g, n in df[subtype].value_counts().items() if n >= 20 and g != "NC"]
        for g in groups:
            mask = df[subtype] == g
            kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=f"{g} (n={mask.sum()})")
            kmf.plot_survival_function(ax=ax, ci_show=False)
        # log-rank test across subtypes
        sub = df[df[subtype].isin(groups)]
        res = multivariate_logrank_test(sub["time"], sub[subtype], sub["event"])
        p = res.p_value
        ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3g}"
        ax.set_title(f"METABRIC overall survival by {subtype}\nlog-rank {ptxt}")
        print(f"\nLog-rank across {subtype} subtypes: {ptxt} "
              f"(test statistic={res.test_statistic:.1f})")
    else:
        kmf.fit(df["time"], df["event"], label=f"All (n={len(df)})")
        kmf.plot_survival_function(ax=ax)
        ax.set_title("METABRIC overall survival")

    ax.set_xlabel("Months since diagnosis")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(config.FIG_DIR, "metabric_km_by_subtype.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")


if __name__ == "__main__":
    main()
