# Colab Runbook

Use this when running the heavier scVI stage on Google Colab.

## 1. Upload Project To Google Drive

Upload the full project folder, including:

- `seurat_metadata.csv`
- `exported/`
- `scripts/`
- `src/`
- `requirements-colab.txt`

You do not need to upload `.venv`.

## 2. Start Colab With GPU

In Colab:

Runtime -> Change runtime type -> GPU.

Check GPU:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

## 3. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then move to your project folder:

```bash
%cd /content/drive/MyDrive/Internship_pm
```

Adjust the path if your folder name is different.

## 4. Install Dependencies

Use the Colab-specific requirements file. It does not explicitly install `torch`, because Colab usually already provides a GPU-compatible PyTorch.

```bash
!pip install -r requirements-colab.txt
```

Restart runtime if Colab asks.

## 5. Run Pipeline

If `.h5ad` files were not uploaded:

```bash
!python scripts/01_prepare_h5ad.py
!python scripts/02_build_baseline_features.py
```

Run a stronger GPU scVI model:

```bash
!python scripts/03_train_scvi_embeddings.py \
  --max-epochs 100 \
  --n-latent 30 \
  --highly-variable-genes 3000
```

Then run downstream results:

```bash
!python scripts/04_patient_classification.py \
  --features results/patient_scvi_embeddings.csv \
  --output results/classification_metrics_scvi.csv

!python scripts/04_patient_classification.py \
  --features results/patient_celltype_scvi_embeddings.csv \
  --output results/classification_metrics_scvi_celltype.csv

!python scripts/05_t1d_subtype_discovery.py --k 3

!python scripts/06_summarize_and_plot_results.py
```

## 6. Compare Local vs Colab

Local first run:

- 20 epochs.
- 20 latent dimensions.
- 2,000 HVGs.
- CPU.

Suggested Colab run:

- 100 epochs.
- 30 latent dimensions.
- 3,000 HVGs.
- GPU.

Use the same classification and subtype scripts so the comparison is fair.

