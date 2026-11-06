# Data Sources

Every dataset this project uses, with links, sizes, access notes, and required citations.
All data is public; no private or patient-identifiable data is used.

---

## 1. METABRIC — gene expression + clinical (genomics train cohort)
- **Access:** [cBioPortal](https://www.cbioportal.org/study?id=brca_metabric) REST API (no login). Pulled by `src/fetch_data.py`.
- **Study ID:** `brca_metabric`
- **Size:** ~2,500 patients; ~1,900 with matched expression + survival. Small (CSV, MBs).
- **Cite:** Curtis et al., *Nature* 2012; Pereira et al., *Nat Commun* 2016.

## 2. TCGA-BRCA — gene expression + clinical (genomics validation + multi-modal cohort)
- **Access:** [cBioPortal](https://www.cbioportal.org/study?id=brca_tcga_pan_can_atlas_2018) REST API.
- **Study ID:** `brca_tcga_pan_can_atlas_2018`
- **Size:** 1,084 patients in the clinical table. Small (CSV, MBs).
- **Includes** the TCGA Pan-Cancer Clinical Data Resource endpoints (overall survival, progression-free interval,
  disease-specific survival), used as the main and sensitivity endpoints.
- **Cite:** TCGA Network, *Nature* 2012; Hoadley et al., *Cell* 2018; Liu et al., *Cell* 2018 (endpoints).

## 3. BreakHis — histopathology images (imaging warm-up, Stage B)
- **Access:** [Lab Visão Robótica e Imagem](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) — direct download `BreaKHis_v1.tar.gz` (~4 GB).
- **Size:** 7,909 images (2,480 benign / 5,429 malignant), 82 patients, 700×460 PNG, 40–400× magnification.
- **Task:** benign vs. malignant classification (pipeline validation, NOT survival).
- **Cite:** Spanhol et al., *IEEE TBME* 63(7):1455–1462, 2016. Non-commercial research use.

## 4. TCGA-BRCA — whole-slide H&E images (imaging survival + fusion, Stages C–D)
- **Access:** the NCI [Genomic Data Commons](https://portal.gdc.cancer.gov/) API (file list: `/files`; download:
  `/data/<file_id>`). Diagnostic (FFPE) slides are open-access, no login.
- **Size:** GDC lists 1,133 TCGA-BRCA diagnostic slides from 1,062 patients; one slide per patient was used for
  the 1,033 analysed (~1 TB in total; each slide verified against GDC's size and MD5, then deleted after
  feature extraction). Scanner magnification per slide was read from the slide headers (Amendment D5).
- **Link to genes:** matched to study #2 by TCGA patient barcode (e.g., `TCGA-XX-XXXX`).
- **Cite:** Clark et al., *J Digit Imaging* 2013 (TCIA); plus the TCGA-BRCA citations in #2.

## 5. Pretrained models (weights downloaded from Hugging Face)
- **Phikon** (`owkin/phikon`, revision `057cc0295895c2df3dd7681a89680da6015cbefe`) — tile features. Pretrained
  without labels on TCGA tissue tiles (Filiot et al., medRxiv 2023).
- **Prov-GigaPath** (`prov-gigapath/prov-gigapath`) — Amendment E only; pretrained on Providence Health slides,
  not TCGA (Xu et al., *Nature* 2024). Gated: needs a Hugging Face account and accepted terms.
- Early stages only: ImageNet-pretrained ResNet / EfficientNet (torchvision / Keras).

## 6. (Possible future) CPTAC-BRCA — external image validation
- Considered for future external validation of the imaging model (another cohort with
  images + omics). Not yet used. [CPTAC on TCIA](https://www.cancerimagingarchive.net/).

---

### Tooling citation
If results are published, also cite **cBioPortal**: Cerami et al., *Cancer Discovery* 2012;
Gao et al., *Science Signaling* 2013.

### Why data isn't committed to Git
Genomics CSVs are re-downloadable in seconds (`src/fetch_data.py`); image data is far too
large for a repo. The code + `requirements.txt` + this file make everything reproducible.
