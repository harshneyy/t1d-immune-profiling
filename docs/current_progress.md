# Current Progress

Date: 2026-05-24

## Environment

Local virtual environment created:

- `.venv`
- Python 3.12.3
- Installed `scanpy`, `anndata`, `scvi-tools`, `torch`, `scikit-learn`, plotting tools.

PyTorch status:

- Torch installed: `2.12.0+cu130`
- CUDA available locally: `False`

So local runs are CPU-only.

## Completed Steps

### 1. Dataset Inventory

Command:

```bash
python3 scripts/00_dataset_inventory.py
```

Output:

- `results/dataset_inventory.txt`

Dataset summary:

- 117,737 cells.
- 20,667 genes.
- 77 samples.
- 46 T1D samples.
- 31 healthy controls.
- Matrix dimensions: 20,667 genes x 117,737 cells.

### 2. AnnData Conversion

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-t1d .venv/bin/python scripts/01_prepare_h5ad.py
```

Output:

- `data/processed/t1d_pbmc_raw.h5ad`

AnnData shape:

- 117,737 cells x 20,667 genes.

### 3. Baseline Features

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-t1d .venv/bin/python scripts/02_build_baseline_features.py
```

Output:

- `results/patient_baseline_features.csv`

Feature matrix:

- 77 samples x 81 columns.

### 4. Local scVI Training

CPU command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-t1d .venv/bin/python scripts/03_train_scvi_embeddings.py \
  --max-epochs 20 \
  --n-latent 20 \
  --highly-variable-genes 2000 \
  --no-gpu
```

Outputs:

- `data/processed/t1d_pbmc_scvi.h5ad`
- `results/patient_scvi_embeddings.csv`
- `results/patient_celltype_scvi_embeddings.csv`

Training notes:

- Full CPU training completed successfully.
- 20 epochs took about 17-18 minutes.
- Training loss decreased from about 317 to about 263.

### 5. Classification Benchmarks

Immune-only features were used by default. Clinical/HLA metadata is excluded unless `--include-clinical` is passed.

Best mean ROC-AUC values so far:

- Baseline marker + cell-type features: Random Forest, about 0.894.
- Cell-type only: Random Forest, about 0.871.
- scVI patient embeddings: SVM, about 0.868.
- scVI celltype-aware embeddings: Logistic Regression / Random Forest, about 0.882.

Main output:

- `results/classification_metrics_summary.csv`

Figure:

- `figures/classification_roc_auc.png`

### 6. T1D Subtype Discovery

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-t1d .venv/bin/python scripts/05_t1d_subtype_discovery.py --k 3
```

Output:

- `results/t1d_embedding_subtypes.csv`
- `results/t1d_subtype_summary.csv`

Subtype sizes:

- subtype_1: 22 T1D patients.
- subtype_2: 8 T1D patients.
- subtype_3: 16 T1D patients.

Figures:

- `figures/scvi_patient_pca_condition.png`
- `figures/t1d_scvi_subtype_pca.png`
- `figures/t1d_subtype_feature_heatmap.png`

## Important Interpretation Note

The current scVI run is a valid first local result, but it is still a development run:

- CPU-only.
- 20 epochs.
- 2,000 highly variable genes.
- Uses the exported `SCT_counts` matrix.

For final reporting, repeat or improve this on Colab/GPU with:

- More epochs, such as 100.
- Possibly 3,000 highly variable genes.
- Careful comparison against the local 20-epoch run.

