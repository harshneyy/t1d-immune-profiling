# T1D Foundation Model Immune Profiling — Progress Report

**Date:** 2026-05-24

## 1. Project Overview

This project develops a foundation-model-style immune profiling pipeline for type 1 diabetes (T1D) using single-cell RNA-seq from peripheral blood mononuclear cells (PBMCs). The goal is to learn compact single-cell representations using `scVI`, aggregate them to patient-level immune profiles, and use those profiles for:

- T1D versus healthy classification
- Immune subtype discovery among T1D patients
- Biological interpretation of learned immune signals

The first implementation focuses on a practical scVI-based workflow and comparison against classical immune marker and cell-type baseline features.

## 2. Data and Environment

### Data Inputs
- `seurat_metadata.csv`
- `exported/T1D_Seurat_Object_Final_SCT_counts.mtx`
- `exported/T1D_Seurat_Object_Final_SCT_counts_cells.txt`
- `exported/T1D_Seurat_Object_Final_SCT_counts_genes.txt`

### Environment
- Python 3.12.3
- Local virtual environment: `.venv`
- Installed packages include `scanpy`, `anndata`, `scvi-tools`, `torch`, `scikit-learn`, and plotting utilities.
- Local PyTorch version: `2.12.0+cu130`
- CUDA availability: `False` (CPU-only run)

## 3. Completed Pipeline Steps

### 3.1 Dataset Inventory

Script: `scripts/00_dataset_inventory.py`
Output: `results/dataset_inventory.txt`

Key dataset properties:
- Total cells: 117,737
- Unique genes: 20,667
- Samples: 77
- T1D samples: 46
- Healthy samples: 31
- Matrix dimensions: 20,667 genes × 117,737 cells

### 3.2 AnnData Conversion

Script: `scripts/01_prepare_h5ad.py`
Output: `data/processed/t1d_pbmc_raw.h5ad`

Results:
- Raw count matrix imported successfully into AnnData
- Metadata aligned to cells
- AnnData shape: 117,737 cells × 20,667 genes

### 3.3 Baseline Feature Engineering

Script: `scripts/02_build_baseline_features.py`
Output: `results/patient_baseline_features.csv`

Results:
- Generated patient-level feature matrix
- Created 77 patient profiles with baseline immune and cell-type features

### 3.4 scVI Training and Embeddings

Script: `scripts/03_train_scvi_embeddings.py`
Outputs:
- `data/processed/t1d_pbmc_scvi.h5ad`
- `results/patient_scvi_embeddings.csv`
- `results/patient_celltype_scvi_embeddings.csv`

Notes:
- scVI training completed on CPU
- Configuration used: 20 epochs, 20 latent dimensions, 2,000 highly variable genes
- Training run time: about 17–18 minutes on CPU

### 3.5 Patient-Level Classification

Script: `scripts/04_patient_classification.py`
Output: `results/classification_metrics_summary.csv`

Performance summary from current experiments:

- Best overall model: `baseline_marker_celltype` with Random Forest
  - ROC-AUC: 0.8937 ± 0.0905
- Best cell-type-only model: Random Forest
  - ROC-AUC: 0.8714 ± 0.0917
- Best scVI patient embedding model: SVM with RBF kernel
  - ROC-AUC: 0.8678 ± 0.0866
- Best scVI celltype-aware model: Logistic Regression or Random Forest
  - ROC-AUC: 0.8819 ± 0.0936 (Logistic Regression)

### 3.6 T1D Subtype Discovery

Script: `scripts/05_t1d_subtype_discovery.py`
Outputs:
- `results/t1d_embedding_subtypes.csv`
- `results/t1d_subtype_summary.csv`

Subtype breakdown:
- `subtype_1`: 22 T1D patients
- `subtype_2`: 8 T1D patients
- `subtype_3`: 16 T1D patients

Subtype-level findings:
- Subtypes vary in cell-type proportions and immune signature scores
- Monocyte activation and B-cell antigen presentation signatures differ across clusters

## 4. Key Observations

### Dataset structure
- T1D cases account for 46 out of 77 samples.
- Cell-type composition differences are visible between T1D and healthy controls, notably:
  - higher monocyte proportions in T1D
  - higher naive B-cell proportions in T1D
  - lower CD4 T-cell proportion in T1D

### Classification
- Classical baseline immune and cell-type models remain strong performers in patient-level classification.
- scVI-derived patient embeddings are competitive with baseline models but currently do not clearly outperform the marker/cell-type baseline.
- Celltype-aware scVI aggregation improves performance compared to patient-level scVI aggregation.

### Subtyping
- Three T1D subtypes were recovered from patient embeddings.
- The subtype clusters appear to capture distinct immune composition and signature patterns.

## 5. Limitations of the Current Results

- All runs were performed on CPU only.
- scVI was trained for only 20 epochs.
- Highly variable gene selection was limited to 2,000 genes.
- Results are currently from a development-level run rather than a final production-level analysis.

## 6. Recommended Next Steps

1. Re-run scVI on GPU with a stronger configuration:
   - 100 epochs
   - 30 latent dimensions
   - 3,000 highly variable genes

2. Validate the current classification results with additional model families and parameter tuning.
3. Compare GPU scVI embeddings directly against current CPU results.
4. Add stronger biological interpretation of T1D subtypes, including specific cell types and gene programs.
5. Generate final figures and tables for reporting.

## 7. Available Outputs

- `results/dataset_inventory.txt`
- `data/processed/t1d_pbmc_raw.h5ad`
- `data/processed/t1d_pbmc_scvi.h5ad`
- `results/patient_baseline_features.csv`
- `results/patient_scvi_embeddings.csv`
- `results/patient_celltype_scvi_embeddings.csv`
- `results/classification_metrics_summary.csv`
- `results/t1d_embedding_subtypes.csv`
- `results/t1d_subtype_summary.csv`
- `figures/` (classification and subtype visualizations)

---

This report documents the current state of the T1D foundation-model pipeline and the main results achieved so far. The next phase should focus on GPU-based scVI refinement and deeper subtype interpretation.