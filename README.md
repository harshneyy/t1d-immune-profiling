# 🧬 T1D & Celiac Disease Foundation Model Immune Profiling

**Foundation Model-Based Representation Learning for Immune Profiling of Type 1 Diabetes and Celiac Disease from Transcriptomic Data**

> **Internship Project | IIITDM Kurnool | 2025–2026**  
> Supervisor: **Dr. Priyadarshini Rai**, Assistant Professor, Dept. of CSE

---

## 📌 Project Overview

This project applies state-of-the-art deep learning and foundation models to profile the immune dysregulation in two autoimmune diseases — **Type 1 Diabetes (T1D)** and **Celiac Disease (CeD)** — from transcriptomic data.

### Key Results

| Disease | Method | ROC-AUC |
|---|---|---|
| T1D | scVI + HistGradientBoosting | **0.8976** |
| T1D | Geneformer (zero-shot) + RF | **0.8893** |
| T1D | Geneformer (fine-tuned) | 0.86 |
| Celiac (bulk) | SVM on bulk RNA-seq | **0.967** |
| Celiac (3-class) | SVM (Healthy/GFD/Active) | **72.5% acc** |

**Key Finding:** Geneformer's zero-shot performance (AUC 0.89) nearly matches the fine-tuned model (AUC 0.86), demonstrating that massive foundation models pre-trained on 30M cells already encode the disease signal without task-specific supervision.

---

## 🗂️ Repository Structure

```
t1d-immune-profiling/
│
├── scripts/                          # T1D scRNA-seq pipeline
│   ├── 00_dataset_inventory.py       # Dataset QC and exploration
│   ├── 01_prepare_h5ad.py            # Convert Seurat output to AnnData
│   ├── 01_build_metadata_celltype_features.py
│   ├── 02_build_baseline_features.py # PCA/baseline features
│   ├── 03_train_scvi_embeddings.py   # GPU scVI training + embedding extraction
│   ├── 04_patient_classification.py  # 16-variant classifier benchmark
│   ├── 05_t1d_subtype_discovery.py   # K-means subtype discovery
│   └── 06_summarize_and_plot_results.py
│
├── run_step1_scvi_gpu.py             # Entry point: scVI GPU training
├── run_step2_geneformer_gpu.py       # Entry point: Geneformer zero-shot embedding
├── run_step3_final_comparison.py     # Entry point: Final comparison plots
├── run_finetune_geneformer.py        # Entry point: Geneformer fine-tuning
│
├── latex_report/                     # Full LaTeX report source
│   ├── main.tex
│   ├── chap_1.tex – chap_7.tex
│   ├── fig_scvi_arch.tex             # TikZ: scVI architecture diagram
│   ├── fig_geneformer_pipeline.tex   # TikZ: Geneformer zero-shot pipeline
│   ├── fig_finetune_arch.tex         # TikZ: Geneformer fine-tuning diagram
│   ├── Bibliography.bib
│   └── *.png                         # All result figures
│
├── results/                          # Output CSVs and metrics
├── figures/                          # Output plots
├── requirements.txt
└── README.md
```

---

## 🧪 Datasets

### T1D — scRNA-seq
- **Source:** GEO (single-cell PBMC dataset)
- **Size:** 117,737 cells from **77 patients** (46 T1D, 31 healthy)
- **Platform:** 10x Genomics Chromium v3
- **Format:** Seurat-exported sparse matrix → AnnData `.h5ad`

### Celiac Disease — scRNA-seq
- **Source:** GEO GSE315138
- **Size:** ~33,000 PBMC cells from 4 patients
- Used for **cross-disease latent alignment** with T1D scVI model

### Celiac Disease — Bulk RNA-seq
- **Source:** GEO GSE145358
- **Size:** 36 intestinal biopsy samples (Healthy / Silent-GFD / Active)
- Used for **robust clinical classification**

---

## 🚀 How to Run

### 1. Setup
```bash
conda create -n immune_profiling python=3.10
conda activate immune_profiling
pip install -r requirements.txt
```

### 2. T1D Pipeline
```bash
# Step 1: Train scVI and extract embeddings (GPU recommended)
python run_step1_scvi_gpu.py

# Step 2: Extract Geneformer zero-shot embeddings
python run_step2_geneformer_gpu.py

# Step 3: Run all 16 classifier variants and generate plots
python run_step3_final_comparison.py

# Optional: Fine-tune Geneformer (5 epochs, ~3-4 hrs on GPU)
python run_finetune_geneformer.py
```

### 3. Hardware
- **GPU:** NVIDIA RTX 2000 Ada (16 GB VRAM)
- **RAM:** 32 GB recommended
- **Storage:** ~10 GB for embeddings and model checkpoints

---

## 📊 Methods

### T1D Analysis
1. **QC & Preprocessing** — Seurat SCTransform, doublet removal, HVG selection (3,000 genes)
2. **scVI** — VAE with Negative Binomial likelihood, 30-dim latent space, 100 epochs GPU training
3. **Geneformer** — 12-layer transformer (30M params), rank-value tokenisation, zero-shot embedding extraction
4. **Patient Aggregation** — Mean pooling + cell-type-aware concatenation
5. **Classification** — LR, RF, HGB, SVM under stratified 5-fold CV
6. **Fine-tuning** — BertForSequenceClassification head, lr=2e-5, cosine scheduler, 5 epochs

### Celiac Disease Analysis
1. **scRNA-seq Alignment** — Celiac PBMCs projected into T1D scVI latent space
2. **Bulk RNA-seq** — CPM normalisation → log2 transform → top 2000 HVGs → SVM/RF

---

## 📄 Report

The full internship report (LaTeX source + compiled PDF) is in the `latex_report/` directory.

**Report Title:** *Foundation Model-Based Representation Learning for Immune Profiling of Type 1 Diabetes and Celiac Disease from Transcriptomic Data*

---

## 📦 Citation / References

- Theodoris et al. (2023). *Transfer learning enables predictions in network biology.* Nature. [Geneformer]
- Lopez et al. (2018). *Deep generative modeling for single-cell transcriptomics.* Nature Methods. [scVI]
- Gayoso et al. (2022). *A Python library for probabilistic analysis of single-cell omics data.* Nature Biotechnology. [scvi-tools]

---

## 👨‍💻 Author

**Harshit Verma** | Roll No: 122CS0023  
B.Tech Computer Science and Engineering  
IIITDM Kurnool | 2025–2026
