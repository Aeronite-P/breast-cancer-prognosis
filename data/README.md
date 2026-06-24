# Data

**This folder is intentionally (almost) empty in the repository.** The raw and processed
data files are *not* committed to Git — they're re-downloadable in seconds from a public
source, and keeping them out keeps the repo small and clean.

## How to (re)generate the data
```bash
source .venv/bin/activate
python src/fetch_data.py
```
This creates:
- `data/raw/metabric_clinical.csv`
- `data/raw/metabric_expr.csv`
- `data/processed/metabric_merged.csv`

## Where the data comes from (provenance)
All data is public and obtained through the [cBioPortal](https://www.cbioportal.org) REST API.
No private, patient-identifiable, or restricted data is used.

| Cohort | Study ID | Source publications |
|---|---|---|
| METABRIC | `brca_metabric` | Curtis et al., *Nature* 2012; Pereira et al., *Nat Commun* 2016 |
| TCGA-BRCA (PanCancer Atlas) | `brca_tcga_pan_can_atlas_2018` | TCGA Network, *Nature* 2012; Hoadley et al., *Cell* 2018 |

**Please also cite cBioPortal if you publish:**
Cerami et al., *Cancer Discovery* 2012; Gao et al., *Science Signaling* 2013.

## Terms
The underlying studies' data are publicly released for research use. This project's **code**
is MIT-licensed (see `LICENSE`); that license does not extend to the third-party datasets,
which remain governed by their original sources' terms. Always credit the original studies.
