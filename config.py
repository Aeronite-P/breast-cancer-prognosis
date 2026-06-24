"""Central configuration — study IDs, file paths, gene panels, the random seed.

Everything reads from here so the whole pipeline is reproducible and easy to tweak.
"""
import os

# --- cBioPortal public API ---
CBIOPORTAL_API = "https://www.cbioportal.org/api"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# --- Reproducibility: one seed, used everywhere ---
RANDOM_SEED = 42

# --- Cohorts: two independent breast-cancer datasets on two platforms ---
METABRIC = {
    "study_id": "brca_metabric",
    # mRNA expression z-scores (relative to all samples) — good ML features.
    "expr_profile": "brca_metabric_mrna_median_all_sample_Zscores",
    "platform": "Illumina HT-12 v3 microarray",
}
TCGA = {
    "study_id": "brca_tcga_pan_can_atlas_2018",
    # NOTE: verify this profile id in Week 3 with src/fetch_data.py list_profiles().
    "expr_profile": "brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna_median_all_sample_Zscores",
    "platform": "RNA-seq",
}

# --- PAM50 intrinsic-subtype genes: a strong, literature-grounded starting panel ---
PAM50 = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20",
    "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1",
    "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5",
    "MAPT", "MDM2", "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1",
    "NDC80", "NUF2", "ORC6", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6",
    "TMEM45B", "TYMS", "UBE2C", "UBE2T",
]
