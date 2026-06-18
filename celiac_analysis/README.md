# Celiac Disease Immune Profiling

This sub-project applies machine learning and foundation model embeddings to classify and characterise Celiac Disease from two transcriptomic modalities — single-cell RNA-seq (scRNA-seq) and bulk RNA-seq.

## Datasets

| Dataset | Source | Type | Description |
|---|---|---|---|
| GSE315138 | GEO | scRNA-seq | PBMCs from 4 Celiac patients (~33K cells) — used for cross-disease latent alignment |
| GSE145358 | GEO | Bulk RNA-seq | 36 intestinal biopsies (Healthy / Silent-GFD / Active Celiac) — used for classification |

## Key Results

| Task | Method | Metric |
|---|---|---|
| Active Celiac vs Healthy (binary) | SVM on bulk RNA-seq | **AUC = 0.967** |
| Healthy / GFD / Active (3-class) | SVM on bulk RNA-seq | **Accuracy = 72.5%** |
| Cross-disease PBMC alignment | scVI latent projection | Celiac clusters near healthy (not T1D) |

## Scripts

| Script | Description |
|---|---|
| `step1_preprocess_scvi.py` | QC, SCTransform-equivalent normalisation, scVI training on Celiac scRNA-seq |
| `step2_geneformer.py` | Geneformer zero-shot embedding extraction for Celiac scRNA-seq |
| `step3_classify_compare.py` | Cross-disease projection into T1D scVI latent space + PCA/UMAP plots |
| `step4_bulk_classification.py` | Bulk RNA-seq preprocessing, HVG selection, SVM/RF classification |

## How to Run

```bash
# From repo root
python celiac_analysis/scripts/step1_preprocess_scvi.py
python celiac_analysis/scripts/step2_geneformer.py
python celiac_analysis/scripts/step3_classify_compare.py
python celiac_analysis/scripts/step4_bulk_classification.py
```

## Biological Interpretation

- **Celiac PBMCs cluster with healthy controls** when projected into the T1D scVI latent space — confirming that Celiac does NOT manifest the same systemic peripheral blood immune shift as T1D.
- **Intestinal biopsy bulk RNA-seq is highly discriminative** (AUC 0.967) — the disease signal is localised to the gut mucosa, not the bloodstream.
- This contrast (systemic T1D vs localised CeD) is a key finding of the project.
