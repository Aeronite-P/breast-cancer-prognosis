"""
fetch_data.py — Acquire breast-cancer cohort data from cBioPortal (public API).

Primary data source: the cBioPortal REST API (https://www.cbioportal.org/api).
Fully public, no login, reproducible on any machine with internet.

What it pulls (per study):
  1. Patient clinical data, incl. survival (OS_MONTHS, OS_STATUS, ...).
  2. mRNA expression z-scores for a gene panel (default: PAM50).
Outputs tidy CSVs in data/raw/ and a merged modeling table in data/processed/.

For the FULL transcriptome (~20k genes) later, grab the study tarball from
https://www.cbioportal.org/datasets — faster than the API for big matrices.
This script is the lightweight, always-works path used in Week 1.

Run:  python src/fetch_data.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import requests

# allow "import config" when run from the repo root OR from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

API = config.CBIOPORTAL_API
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "Content-Type": "application/json"})


def _get(path: str, **params):
    r = SESSION.get(f"{API}{path}", params=params, timeout=90)
    r.raise_for_status()
    return r.json()


def _post(path: str, body, **params):
    r = SESSION.post(f"{API}{path}", params=params, data=json.dumps(body), timeout=180)
    r.raise_for_status()
    return r.json()


def list_profiles(study_id: str) -> pd.DataFrame:
    """Helper to discover the exact molecular-profile IDs for a study."""
    data = _get(f"/studies/{study_id}/molecular-profiles")
    return pd.DataFrame(data)[["molecularProfileId", "molecularAlterationType", "name"]]


def fetch_clinical(study_id: str) -> pd.DataFrame:
    """Patient-level clinical data as a wide table (one row per patient)."""
    rows = _get(
        f"/studies/{study_id}/clinical-data",
        clinicalDataType="PATIENT",
        projection="SUMMARY",
        pageSize=10_000_000,
        pageNumber=0,
        direction="ASC",
    )
    long_df = pd.DataFrame(rows)
    wide = long_df.pivot_table(
        index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first"
    )
    wide.index.name = "patientId"
    wide.columns.name = None
    return wide


def map_symbols_to_entrez(symbols) -> dict:
    """HUGO gene symbols -> Entrez gene IDs (genes not found are dropped)."""
    data = _post(
        "/genes/fetch", body=list(symbols),
        geneIdType="HUGO_GENE_SYMBOL", projection="SUMMARY",
    )
    return {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in data}


def sample_to_patient(study_id: str) -> dict:
    samples = _get(
        f"/studies/{study_id}/samples",
        projection="SUMMARY", pageSize=10_000_000, pageNumber=0,
    )
    return {s["sampleId"]: s["patientId"] for s in samples}


def fetch_expression(study_id: str, profile_id: str, symbols) -> pd.DataFrame:
    """DataFrame: index=patientId, columns=gene symbols, values=z-scores."""
    sym2ent = map_symbols_to_entrez(symbols)
    missing = sorted(set(symbols) - set(sym2ent))
    if missing:
        print(f"  (note) {len(missing)} symbols not mapped, skipped: {missing}")
    ent2sym = {v: k for k, v in sym2ent.items()}
    body = {"entrezGeneIds": list(sym2ent.values()), "sampleListId": f"{study_id}_all"}
    data = _post(
        f"/molecular-profiles/{profile_id}/molecular-data/fetch",
        body=body, projection="SUMMARY",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df["gene"] = df["entrezGeneId"].map(ent2sym)
    mat = df.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="first")
    s2p = sample_to_patient(study_id)
    mat.index = mat.index.map(lambda s: s2p.get(s, s))
    mat.index.name = "patientId"
    mat.columns.name = None
    return mat.astype(float)


def build(study_id: str, profile_id: str, symbols, out_prefix: str):
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    print(f"[{study_id}] fetching clinical ...")
    clin = fetch_clinical(study_id)
    clin.to_csv(os.path.join(config.RAW_DIR, f"{out_prefix}_clinical.csv"))
    print(f"  clinical: {clin.shape[0]} patients x {clin.shape[1]} attributes")

    print(f"[{study_id}] fetching expression ({len(symbols)} genes) ...")
    expr = fetch_expression(study_id, profile_id, symbols)
    expr.to_csv(os.path.join(config.RAW_DIR, f"{out_prefix}_expr.csv"))
    print(f"  expression: {expr.shape[0]} patients x {expr.shape[1]} genes")

    merged = clin.join(expr, how="inner")
    merged.to_csv(os.path.join(config.PROCESSED_DIR, f"{out_prefix}_merged.csv"))
    print(f"  merged modeling table: {merged.shape[0]} patients x {merged.shape[1]} cols")
    return clin, expr, merged


if __name__ == "__main__":
    clin, expr, merged = build(
        config.METABRIC["study_id"],
        config.METABRIC["expr_profile"],
        config.PAM50,
        "metabric",
    )

    print("\n--- METABRIC survival sanity check ---")
    for col in ("OS_MONTHS", "OS_STATUS", "VITAL_STATUS"):
        if col in clin.columns:
            s = clin[col]
            if col == "OS_MONTHS":
                vals = pd.to_numeric(s, errors="coerce")
                print(f"OS_MONTHS: n={vals.notna().sum()}  "
                      f"median={vals.median():.1f}  max={vals.max():.1f}")
            else:
                print(f"{col}: {dict(s.value_counts(dropna=False).head(5))}")

    n_event = pd.to_numeric(
        clin.get("OS_STATUS", pd.Series(dtype=str)).astype(str).str.startswith("1"),
        errors="coerce",
    ).sum() if "OS_STATUS" in clin.columns else 0
    print(f"\nReady. Patients with expression + clinical: {merged.shape[0]}; "
          f"deaths recorded: ~{int(n_event)}")
    print("Saved -> data/raw/metabric_*.csv  and  data/processed/metabric_merged.csv")
