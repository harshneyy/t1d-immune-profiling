# GPU Setup And Runbook

This file explains how to run the T1D foundation-model pipeline on a stronger machine.

## 1. Copy The Project Folder

Move the full project folder to the GPU machine, including:

- `seurat_metadata.csv`
- `exported/`
- `docs/`
- `scripts/`
- `src/`
- `requirements.txt`

The matrix file is large, so use Google Drive, an external drive, `rsync`, or an institute server.

## 2. Create Environment

Recommended Python version: 3.10 or 3.11.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch GPU support is not detected, install the correct CUDA build from the official PyTorch selector, then reinstall `scvi-tools` if needed.

Check GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 3. Run Pipeline

Run from the project root.

### Dataset Inventory

```bash
python scripts/00_dataset_inventory.py
```

Output:

- `results/dataset_inventory.txt`

### Convert To AnnData

```bash
python scripts/01_prepare_h5ad.py
```

Output:

- `data/processed/t1d_pbmc_raw.h5ad`

### Optional No-Dependency Metadata Baseline

This can run even before installing Scanpy/scVI:

```bash
python scripts/01_build_metadata_celltype_features.py
```

Output:

- `results/patient_metadata_celltype_features.csv`

### Build Baseline Features

```bash
python scripts/02_build_baseline_features.py
```

Output:

- `results/patient_baseline_features.csv`

### Train scVI Embeddings

GPU run:

```bash
python scripts/03_train_scvi_embeddings.py --max-epochs 100 --n-latent 30
```

CPU fallback:

```bash
python scripts/03_train_scvi_embeddings.py --max-epochs 30 --n-latent 20 --no-gpu
```

Outputs:

- `data/processed/t1d_pbmc_scvi.h5ad`
- `results/patient_scvi_embeddings.csv`
- `results/patient_celltype_scvi_embeddings.csv`

### Classification

Using scVI patient embeddings:

```bash
python scripts/04_patient_classification.py \
  --features results/patient_scvi_embeddings.csv \
  --output results/classification_metrics_scvi.csv
```

Using baseline marker/cell-composition features:

```bash
python scripts/04_patient_classification.py \
  --features results/patient_baseline_features.csv \
  --output results/classification_metrics_baseline.csv
```

By default, classification excludes clinical/HLA metadata columns to avoid leakage. To intentionally include those columns for a separate sensitivity analysis, add `--include-clinical`.

### T1D Subtype Discovery

```bash
python scripts/05_t1d_subtype_discovery.py --k 3
```

Output:

- `results/t1d_embedding_subtypes.csv`

## 4. Important Validation Rule

Always split by patient/sample, not by cell.

Correct:

- train samples = some patients
- test samples = different patients

Incorrect:

- train cells and test cells randomly mixed from the same patients

The second approach creates data leakage and can give fake high accuracy.

## 5. Recommended First Run

For the first successful run, use:

```bash
python scripts/01_prepare_h5ad.py
python scripts/02_build_baseline_features.py
python scripts/03_train_scvi_embeddings.py --max-epochs 30 --n-latent 20
python scripts/04_patient_classification.py --features results/patient_scvi_embeddings.csv --output results/classification_metrics_scvi.csv
python scripts/05_t1d_subtype_discovery.py --k 3
```

After that works, increase `--max-epochs` to 100 and compare results.
