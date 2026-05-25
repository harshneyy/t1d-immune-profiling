# 🧬 T1D Foundation Model Immune Profiling

**Foundation Model-Based Representation Learning for Type 1 Diabetes Immune Profiling From Single-Cell RNA-seq**

> **AI Assistant Context** — This README is written so a fresh AI assistant (Antigravity) on a new machine can immediately understand the full project, where we left off, and what needs to happen next. Read this entire file before asking any questions.

---

## 📌 Quick Project Summary

This is an internship/final-year research project that uses **deep representation learning on single-cell RNA-seq (scRNA-seq)** data from peripheral blood immune cells (PBMCs) to study immune dysregulation in **Type 1 Diabetes (T1D)**.

Instead of using only hand-crafted marker genes, we learn compact immune-cell embeddings from gene-expression profiles (using `scVI` and `Geneformer`) and aggregate them to **patient-level immune profiles** for:

1. **T1D vs healthy classification** — Can we distinguish T1D patients from healthy controls?
2. **Immune subtype discovery** — Are there subtypes within T1D patients?
3. **Biological interpretation** — Do learned embeddings reflect known T1D immune dysregulation?

---

## 🖥️ Why We Switched PCs

> **IMPORTANT context for any new AI assistant:**

The original development machine (Linux, no discrete GPU) was used for:
- All local data preprocessing
- CPU-only scVI training (20 epochs — slow but worked)
- Baseline classification benchmarks

**We switched PCs because:**
- Running scVI at full quality (100 epochs, 3000 HVGs) on CPU took ~17-18 min for just 20 epochs — not practical for the full run
- **Geneformer** (the transformer-based foundation model — Step 2) **requires a GPU** — it is simply infeasible on CPU for 117,737 cells
- The new machine (or Google Colab with T4/A100 GPU) is needed to run `STEP1_GPU_scVI.py`, `STEP2_Geneformer.py`, and `STEP3_Final_Comparison.py`
- All three Colab notebooks in `colab_notebooks/` are already written and ready to run

---

## ✅ What Has Been Completed (on old machine)

| Step | Script | Status | Output |
|------|--------|--------|--------|
| Dataset inventory | `scripts/00_dataset_inventory.py` | ✅ Done | `results/dataset_inventory.txt` |
| AnnData conversion | `scripts/01_prepare_h5ad.py` | ✅ Done | `data/processed/t1d_pbmc_raw.h5ad` |
| Baseline features | `scripts/02_build_baseline_features.py` | ✅ Done | `results/patient_baseline_features.csv` |
| CPU scVI (20 epochs) | `scripts/03_train_scvi_embeddings.py` | ✅ Done | `results/patient_scvi_embeddings.csv` |
| Classification benchmarks | `scripts/04_patient_classification.py` | ✅ Done | `results/classification_metrics_summary.csv` |
| T1D subtype discovery | `scripts/05_t1d_subtype_discovery.py` | ✅ Done | `results/t1d_embedding_subtypes.csv` |
| **GPU scVI (100 epochs)** | `colab_notebooks/STEP1_GPU_scVI.py` | ⏳ TODO on GPU | — |
| **Geneformer embeddings** | `colab_notebooks/STEP2_Geneformer.py` | ⏳ TODO on GPU | — |
| **Final comparison** | `colab_notebooks/STEP3_Final_Comparison.py` | ⏳ TODO after Steps 1+2 | — |

---

## 📊 Current Best Results (CPU baseline)

All results use patient-level 5-fold stratified cross-validation (no data leakage):

| Feature Set | Best Model | Mean ROC-AUC |
|-------------|-----------|-------------|
| Baseline marker + cell-type features | Random Forest | ~0.894 |
| Cell-type only | Random Forest | ~0.871 |
| scVI celltype-aware (CPU, 20 epochs) | LR / RF | ~0.882 |
| scVI patient embeddings (CPU, 20 epochs) | SVM | ~0.868 |

> **These are still preliminary CPU results.** The GPU runs (100 epochs, 3000 HVGs) are expected to produce stronger embeddings. Geneformer (transformer-based) may outperform scVI on this task.

T1D subtype discovery found **3 subtypes** among 46 T1D patients:
- Subtype 1: 22 patients
- Subtype 2: 8 patients
- Subtype 3: 16 patients

---

## 🔮 Future Work (What Needs to Be Done Next)

### Immediate — Run on GPU (new machine / Colab)

1. **`colab_notebooks/STEP1_GPU_scVI.py`** — Run GPU scVI (100 epochs, 3000 HVGs, 30 latent dims). Downloads a zip with:
   - `classification_all_gpu.csv`
   - `t1d_subtypes_gpu.csv`
   - `figures/scvi_gpu_pca.png`
   - `figures/scvi_gpu_roc_auc.png`

2. **`colab_notebooks/STEP2_Geneformer.py`** — Run Geneformer inference (~30-60 min on T4). Downloads:
   - `geneformer_cell_embeddings.csv`
   - `patient_geneformer_embeddings.csv`
   - `classification_geneformer.csv`
   - `figures/geneformer_pca.png`

3. **`colab_notebooks/STEP3_Final_Comparison.py`** — Upload outputs from Steps 1+2 to Drive, run final comparison. Produces the main paper figure:
   - `final_roc_auc_comparison.png`
   - `best_model_comparison.png`
   - `final_comparison_table.csv`

### After GPU Runs

4. **Biological interpretation** — Map which scVI/Geneformer embedding dimensions correlate with known T1D immune markers (monocyte activation, B-cell antigen presentation, T-cell dysregulation)
5. **Subtype validation** — Compare the 3 discovered T1D subtypes by HLA risk haplotypes, disease duration, and cell-type composition
6. **SHAP analysis** — Feature importance analysis using SHAP on best classifier
7. **Write report** — Compile results into the internship/project report

---

## 📁 Repository Structure

```
.
├── README.md                          # ← You are here
├── requirements.txt                   # Local Python deps
├── requirements-colab.txt             # Colab-specific deps
├── seurat_metadata.csv                # Patient/cell metadata (77 samples)
│
├── colab_notebooks/                   # GPU-ready Colab scripts (run these next!)
│   ├── STEP1_GPU_scVI.py              # GPU scVI training + classification
│   ├── STEP2_Geneformer.py            # Geneformer embedding extraction
│   └── STEP3_Final_Comparison.py      # Final 3-way comparison figure
│
├── scripts/                           # Local preprocessing scripts (already run)
│   ├── 00_dataset_inventory.py
│   ├── 01_prepare_h5ad.py
│   ├── 01_build_metadata_celltype_features.py
│   ├── 02_build_baseline_features.py
│   ├── 03_train_scvi_embeddings.py
│   ├── 04_patient_classification.py
│   ├── 05_t1d_subtype_discovery.py
│   └── 06_summarize_and_plot_results.py
│
├── docs/
│   ├── foundation_model_t1d_project_plan.md   # Full project plan
│   ├── current_progress.md                    # Detailed progress log
│   ├── colab_runbook.md                       # Colab how-to
│   ├── gpu_setup_and_runbook.md               # GPU setup notes
│   └── polished_project_report.md             # Draft report
│
├── results/                           # CSV outputs (gitignored — regenerate by running scripts)
├── figures/                           # Plot outputs (gitignored)
├── data/                              # Processed h5ad files (gitignored — large)
└── exported/                          # Raw Seurat matrix exports (NOT in repo — too large)
```

---

## 🚀 How to Continue on a New Machine

### Option A: Google Colab (Recommended — FREE GPU)

1. Upload the 4 raw data files to **Google Drive → `MyDrive/T1D_project/`**:
   ```
   seurat_metadata.csv
   T1D_Seurat_Object_Final_SCT_counts.mtx
   T1D_Seurat_Object_Final_SCT_counts_cells.txt
   T1D_Seurat_Object_Final_SCT_counts_genes.txt
   ```
   > These are in the `exported/` folder locally (not tracked by git — too large). You already have them on your old machine / Drive.

2. Open [Google Colab](https://colab.research.google.com) → **Runtime → Change runtime type → GPU (T4)**

3. Create a new notebook and copy-paste the contents of `colab_notebooks/STEP1_GPU_scVI.py` cell by cell

4. Run STEP1 fully → download the zip → upload results to `Drive/T1D_project/final_results/`

5. Repeat for STEP2 (Geneformer) — this takes 30-60 minutes on T4

6. Repeat for STEP3 (Final Comparison)

### Option B: New Local Machine (with GPU)

```bash
# Clone the repo
git clone <your-github-repo-url>
cd Internship_pm

# Create virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# You'll need the large raw data files separately (not in git)
# Copy from old machine or Google Drive
```

---

## 📦 Dataset (Not in Git — Too Large)

The raw matrix files are **NOT tracked in git** because they are hundreds of MB. To get them:

- **From old machine**: Copy `exported/` folder and `seurat_metadata.csv` manually
- **From Google Drive**: Already uploaded to `MyDrive/T1D_project/` during Colab runs
- **From supervisor/lab**: The original Seurat object export files

File sizes:
- `seurat_metadata.csv` — ~22 MB
- `T1D_Seurat_Object_Final_SCT_counts.mtx` — large sparse matrix (~117K cells × 20K genes)

---

## 🔬 Dataset Details

| Property | Value |
|----------|-------|
| Total cells | 117,737 |
| Genes | 20,667 |
| Total samples | 77 |
| T1D samples | 46 |
| Healthy controls | 31 |
| Matrix type | SCT-normalized sparse counts |

**Key metadata columns in `seurat_metadata.csv`:**
- `COND` — disease label: `T1D` or `Healthy`
- `Sample_ID` — patient identifier
- `Cluster_Annotation_Merged` — immune cell type (e.g., CD4 T cell, Monocyte, B cell)
- `Gender`, `Age_at_diagnosis`, `Disease_Duration`
- `DQ_Risk_Haplotypes`, HLA columns (excluded from main classification to avoid leakage)

---

## 🧪 Research Questions

1. Can deep single-cell embeddings distinguish T1D from healthy controls better than classical marker-gene features?
2. Do scVI / Geneformer embeddings capture known T1D immune dysregulation patterns (monocyte activation, B-cell antigen presentation, T-cell exhaustion)?
3. Can patient-level embeddings identify clinically meaningful T1D immune subtypes?
4. Do subtypes correlate with HLA risk haplotypes, disease duration, or cell composition?

---

## 📚 Key References

- **Geneformer**: Theodoris et al., Nature 2023 — pretrained single-cell transformer on 30M cells
- **scVI**: Lopez et al., Nature Methods 2018 — variational autoencoder for single-cell counts
- **Dataset source**: Systematic immune profiling of T1D PBMCs (see `Systematic immune.pdf` / `systematic_immune.txt`)

---

## 💬 Notes for AI Assistant on New Machine

- The project is well-structured. All local preprocessing is done — **do NOT re-run local scripts** unless asked
- The critical next action is running the 3 Colab notebooks on a GPU
- The `docs/` folder has detailed notes — read `current_progress.md` for the full log
- Classification intentionally excludes HLA/clinical metadata to avoid leakage (use `--include-clinical` only for sensitivity analysis)
- `seurat_metadata.csv` has trailing spaces in string columns — the scripts already handle `.str.strip()`
- The `.gitignore` excludes large files (`*.h5ad`, `*.csv` in results, figures) — these must be regenerated by running the scripts

---

*Last updated: 2026-05-25 | Switched machines due to GPU requirements for Geneformer*
