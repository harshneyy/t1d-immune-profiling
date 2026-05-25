from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_CSV = PROJECT_ROOT / "seurat_metadata.csv"
COUNTS_MTX = PROJECT_ROOT / "exported" / "T1D_Seurat_Object_Final_SCT_counts.mtx"
CELLS_TXT = PROJECT_ROOT / "exported" / "T1D_Seurat_Object_Final_SCT_counts_cells.txt"
GENES_TXT = PROJECT_ROOT / "exported" / "T1D_Seurat_Object_Final_SCT_counts_genes.txt"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_H5AD = PROCESSED_DIR / "t1d_pbmc_raw.h5ad"
FILTERED_H5AD = PROCESSED_DIR / "t1d_pbmc_filtered.h5ad"
SCVI_H5AD = PROCESSED_DIR / "t1d_pbmc_scvi.h5ad"

BASELINE_FEATURES_CSV = RESULTS_DIR / "patient_baseline_features.csv"
PATIENT_SCVI_EMBEDDINGS_CSV = RESULTS_DIR / "patient_scvi_embeddings.csv"
CELLTYPE_SCVI_EMBEDDINGS_CSV = RESULTS_DIR / "patient_celltype_scvi_embeddings.csv"

RANDOM_STATE = 42

CELL_TYPE_COL = "Cluster_Annotation_Merged"
DETAILED_CELL_TYPE_COL = "Cluster_Annotation_All"
SAMPLE_COL = "Sample_ID"
LABEL_COL = "COND"
BATCH_COL = "LIB"

MARKER_SIGNATURES = {
    "b_cell_antigen_presentation": [
        "CD74",
        "HLA-DRA",
        "HLA-DRB1",
        "HLA-DQA1",
        "HLA-DQB1",
        "HLA-DPA1",
        "HLA-DPB1",
        "HLA-DMB",
        "CTSS",
    ],
    "monocyte_activation": [
        "LYZ",
        "CD14",
        "FCER1G",
        "LGALS9",
        "CTSC",
        "FGR",
        "LST1",
    ],
    "t_cell_activation_migration": [
        "IL32",
        "IL7R",
        "CD3G",
        "TCF7",
        "KLF2",
        "LEF1",
        "TXK",
        "ANXA1",
    ],
}

