# Foundation Model-Based Representation Learning for Type 1 Diabetes Immune Profiling

## Project Title

Foundation Model-Based Representation Learning for Type 1 Diabetes Immune Profiling From Single-Cell RNA-seq

## Core Idea

This project will use deep representation learning on single-cell RNA-seq data from peripheral blood immune cells to study immune dysregulation in type 1 diabetes (T1D).

Instead of only using manually selected marker genes, the project will learn compact immune-cell embeddings from gene-expression profiles. These embeddings will then be aggregated to patient-level immune profiles for:

- T1D vs healthy classification.
- Immune subtype discovery among T1D patients.
- Interpretation of disease-associated immune cell types and gene programs.

The first implementation will use `scVI` because it is practical, well-supported, and designed for single-cell count data. Transformer-based models such as Geneformer or scGPT can be added later as optional extensions.

## Dataset Available

Local files:

- `seurat_metadata.csv`
- `exported/T1D_Seurat_Object_Final_SCT_counts.mtx`
- `exported/T1D_Seurat_Object_Final_SCT_counts_cells.txt`
- `exported/T1D_Seurat_Object_Final_SCT_counts_genes.txt`

Current known dataset summary:

- 117,737 cells.
- 20,667 genes.
- 77 samples.
- 46 T1D samples.
- 31 healthy control samples.
- Matrix shape: genes x cells.

Important metadata columns:

- `COND`: disease label, T1D or healthy.
- `Sample_ID`: patient/sample identifier.
- `Cluster_Annotation_Merged`: immune cell type.
- `Cluster_Annotation_All`: detailed immune cell annotation.
- `Gender`, `Age_at_diagnosis`, `Age_at_profiling`, `Disease_Duration`.
- `DQ_Risk_Haplotypes`, HLA columns.

## Main Research Questions

1. Can deep single-cell embeddings distinguish T1D patients from healthy controls?
2. Do foundation-model/scVI embeddings capture known immune dysregulation patterns such as monocyte activation, B-cell antigen presentation, and T-cell dysregulation?
3. Can patient-level embeddings identify T1D immune subtypes?
4. Do discovered subtypes differ by HLA risk, disease duration, or cell-type composition?
5. Do learned embeddings outperform classical marker-gene and cell-composition baselines?

## Novelty Points

- Uses deep representation learning for T1D PBMC single-cell immune profiling.
- Converts cell-level embeddings into patient-level immune profiles, avoiding cell-level data leakage.
- Compares learned embeddings against classical immune marker features.
- Performs both prediction and subtype discovery.
- Adds explainability by mapping model signals back to immune cell types and known marker genes.

## Planned Pipeline

### Phase 1: Data Preparation

Input:

- Matrix Market count matrix.
- Cell barcode file.
- Gene file.
- Seurat metadata CSV.

Output:

- `data/processed/t1d_pbmc_raw.h5ad`

Tasks:

- Load sparse count matrix.
- Transpose matrix from genes x cells to cells x genes.
- Attach metadata to cells.
- Validate cell IDs match metadata.
- Clean metadata values, especially trailing spaces in `Gender`.
- Store raw count data in AnnData format.

### Phase 2: Baseline Feature Engineering

Output:

- `results/patient_baseline_features.csv`

Features:

- Cell-type proportions per patient.
- Mean expression/signature scores per patient and cell type.
- Clinical/HLA metadata features.

Marker signature examples:

- B-cell antigen presentation: `CD74`, `HLA-DRA`, `HLA-DRB1`, `HLA-DQA1`, `HLA-DQB1`, `CTSS`.
- Monocyte activation: `LYZ`, `CD14`, `FCER1G`, `LGALS9`, `CTSC`.
- T-cell activation/migration: `IL32`, `IL7R`, `CD3G`, `TCF7`, `KLF2`, `LEF1`.

### Phase 3: scVI Representation Learning

Output:

- `data/processed/t1d_pbmc_scvi.h5ad`
- `results/cell_scvi_latent.csv`
- `results/patient_scvi_embeddings.csv`

Tasks:

- Train scVI on single-cell count data.
- Use `LIB` or sequencing batch as batch covariate if appropriate.
- Extract latent representation for each cell.
- Aggregate latent vectors by `Sample_ID`.
- Also aggregate by `Sample_ID + Cluster_Annotation_Merged` for cell-type-aware patient embeddings.

### Phase 4: Patient-Level Classification

Output:

- `results/classification_metrics.csv`
- `figures/roc_curve.png`
- `figures/confusion_matrix.png`

Models:

- Logistic Regression.
- Random Forest.
- XGBoost or HistGradientBoosting if XGBoost is unavailable.
- SVM.
- MLP/shallow neural network.

Validation:

- Use patient-level stratified 5-fold cross-validation.
- Never split cells from the same patient across train/test.

Metrics:

- Accuracy.
- F1-score.
- ROC-AUC.
- Balanced accuracy.

### Phase 5: Immune Subtype Discovery

Output:

- `results/t1d_embedding_subtypes.csv`
- `figures/t1d_subtype_umap.png`
- `figures/subtype_feature_heatmap.png`

Tasks:

- Cluster only T1D patient embeddings.
- Compare KMeans, Gaussian Mixture Models, and hierarchical clustering.
- Evaluate cluster stability and silhouette score.
- Compare subtypes by:
  - HLA risk.
  - Disease duration.
  - Cell-type proportions.
  - B-cell antigen presentation.
  - Monocyte activation.
  - T-cell activation.

### Phase 6: Interpretation

Output:

- `results/feature_importance.csv`
- `figures/shap_summary.png` if SHAP is available.

Tasks:

- Compare important embedding dimensions with marker-gene signatures.
- Identify most informative immune cell types.
- Compare learned subtypes with known T1D biology from the paper.

## Compute Requirements

Minimum:

- 16 GB RAM recommended for preprocessing.
- CPU is enough for baseline ML.
- scVI can run on CPU but will be slower.

Recommended:

- GPU with 8-16 GB VRAM for scVI training.
- Google Colab/Kaggle/institute GPU should be enough.

Transformer extension:

- Geneformer/scGPT will need GPU more strongly and may require extra setup time.

## Expected Timeline

Minimum working version: 5-6 weeks.

Polished final-year project: 8-12 weeks.

Suggested schedule:

- Week 1: Data conversion to AnnData and metadata QC.
- Week 2: Baseline patient feature generation.
- Week 3: scVI training and embedding extraction.
- Week 4: Classification experiments.
- Week 5: T1D subtype discovery.
- Week 6: Interpretation and baseline comparison.
- Weeks 7-8: Figures, report, presentation.
- Weeks 9-12: Optional Geneformer/scGPT extension and external validation.

## Success Criteria

The project is successful if it produces:

- A clean AnnData object from the Seurat export.
- Patient-level embedding features.
- A disease classification benchmark with patient-level cross-validation.
- T1D subtype clusters with biological interpretation.
- A comparison showing whether learned embeddings add value beyond marker-gene/cell-composition features.

## First Implementation Scope

To keep the project feasible, the first version will focus on:

1. AnnData conversion.
2. Baseline marker/cell-composition features.
3. scVI latent embeddings.
4. Patient-level classification.
5. T1D subtype discovery.

Geneformer/scGPT will be treated as an extension after the scVI pipeline works.
