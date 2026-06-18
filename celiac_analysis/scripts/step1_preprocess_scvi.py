#!/usr/bin/env python3
"""
Step 1 — Celiac Disease scRNA-seq: Data Download, Preprocessing & scVI Training
Dataset: GSE315138 (2026) — Duodenal biopsies, Celiac vs Healthy, 10x Genomics

This script:
  1. Extracts and loads GSE315138 10x data
  2. QC filtering and normalization (Scanpy)
  3. Trains scVI on GPU for 100 epochs
  4. Extracts 30-dim cell embeddings
  5. Aggregates to patient level (mean + celltype-aware)
  6. Saves everything to results/
"""

import os
import tarfile
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scvi
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR     = Path("/home/harshney/celiac-immune-profiling/data/raw")
PROC_DIR    = Path("/home/harshney/celiac-immune-profiling/data/processed")
RESULTS_DIR = Path("/home/harshney/celiac-immune-profiling/results")
FIG_DIR     = Path("/home/harshney/celiac-immune-profiling/figures")

for d in [PROC_DIR, RESULTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*65)
print(" CELIAC scRNA-seq PIPELINE  —  Step 1: Preprocessing + scVI")
print("="*65)

# ─── GPU Check ────────────────────────────────────────────────────────────────
print(f"\n[GPU] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── 1. Extract RAW tar ───────────────────────────────────────────────────────
RAW_TAR = RAW_DIR / "GSE315138_RAW.tar"
EXTRACT_DIR = RAW_DIR / "GSE315138_extracted"

if not EXTRACT_DIR.exists() or len(list(EXTRACT_DIR.glob("*.gz"))) == 0:
    print(f"\n[1] Extracting {RAW_TAR}...")
    EXTRACT_DIR.mkdir(exist_ok=True)
    with tarfile.open(RAW_TAR) as tar:
        tar.extractall(EXTRACT_DIR)
    print(f"    Extracted files: {list(EXTRACT_DIR.iterdir())}")
else:
    print(f"\n[1] Already extracted: {list(EXTRACT_DIR.glob('*.gz'))[:5]}")

# ─── 2. Organise per-sample directories ───────────────────────────────────────
# Actual archive contents: GSM*_<SampleName>_{barcodes,features,matrix}.*.gz
# Sample name suffix: -h = healthy, -c = celiac
print("\n[2] Organising 10x sample directories...")

import gzip, shutil

sample_dirs = {}
for f in sorted(EXTRACT_DIR.glob("GSM*_*.gz")):
    fname = f.name  # e.g. GSM9421934_2-PC004-h_barcodes.tsv.gz
    # Split on first underscore to get gsm_id, then rest
    parts = fname.split("_", 1)  # ["GSM9421934", "2-PC004-h_barcodes.tsv.gz"]
    if len(parts) < 2:
        continue
    gsm_id   = parts[0]
    rest     = parts[1]  # "2-PC004-h_barcodes.tsv.gz"

    # Split rest on last underscore to get sample_name and file_type
    rest_parts   = rest.rsplit("_", 1)  # ["2-PC004-h", "barcodes.tsv.gz"]
    if len(rest_parts) < 2:
        continue
    sample_name = rest_parts[0]  # e.g. "2-PC004-h"
    file_type   = rest_parts[1]  # e.g. "barcodes.tsv.gz"

    sdir = EXTRACT_DIR / sample_name
    sdir.mkdir(exist_ok=True)
    if "barcodes" in file_type:
        shutil.copy2(f, sdir / "barcodes.tsv.gz")
    elif "features" in file_type and "reference" not in file_type:
        shutil.copy2(f, sdir / "features.tsv.gz")
    elif "matrix" in file_type:
        shutil.copy2(f, sdir / "matrix.mtx.gz")
    sample_dirs[sample_name] = {"path": sdir, "gsm": gsm_id}

print(f"    Found samples: {list(sample_dirs.keys())}")

# ─── 3. Sample metadata ───────────────────────────────────────────────────────
# Actual sample names from archive (suffix -h=healthy, -c=celiac):
#   GSM9421934 -> 2-PC004-h  (Healthy)
#   GSM9421935 -> 1-PC005-h  (Healthy)
#   GSM9421936 -> 3-PC006-c  (Celiac)
#   GSM9421937 -> SI-TT-G4   (Celiac, CITE-seq GEX)
SAMPLE_META = {
    "2-PC004-h":  {"condition": "Healthy", "patient_id": "H1"},
    "1-PC005-h":  {"condition": "Healthy", "patient_id": "H2"},
    "3-PC006-c":  {"condition": "Celiac",  "patient_id": "C1"},
    "SI-TT-G4":   {"condition": "Celiac",  "patient_id": "C2"},
}

# ─── 4. Load each sample ──────────────────────────────────────────────────────
print("\n[3] Loading 10x data for each sample...")
adatas = []

for sample_name, meta in SAMPLE_META.items():
    sdir = EXTRACT_DIR / sample_name
    if not sdir.exists():
        print(f"    WARNING: {sample_name} directory not found, skipping")
        continue
    required = ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]
    if not all((sdir / f).exists() for f in required):
        print(f"    WARNING: {sample_name} missing files: {[f for f in required if not (sdir/f).exists()]}")
        continue

    try:
        adata = sc.read_10x_mtx(
            str(sdir),
            var_names='gene_symbols',
            cache=True
        )
        adata.obs['sample']     = sample_name
        adata.obs['condition']  = meta['condition']
        adata.obs['patient_id'] = meta['patient_id']
        adata.obs_names = [f"{sample_name}_{bc}" for bc in adata.obs_names]
        print(f"    {sample_name}: {adata.n_obs} cells × {adata.n_vars} genes | {meta['condition']}")
        adatas.append(adata)
    except Exception as e:
        print(f"    ERROR loading {sample_name}: {e}")

if len(adatas) == 0:
    raise RuntimeError("No samples loaded! Check extraction step.")

# ─── 5. Merge all samples ─────────────────────────────────────────────────────
print("\n[4] Merging samples...")
adata = ad.concat(adatas, join='outer', fill_value=0)
adata.obs_names_make_unique()
print(f"    Combined: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"    Conditions: {adata.obs['condition'].value_counts().to_dict()}")
print(f"    Patients: {adata.obs['patient_id'].nunique()}")

# ─── 6. Quality Control ───────────────────────────────────────────────────────
print("\n[5] Quality Control...")
adata.var_names_make_unique()
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Mitochondrial gene detection
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, inplace=True)

print(f"    Before QC: {adata.n_obs} cells")
print(f"    Median genes/cell: {adata.obs['n_genes_by_counts'].median():.0f}")
print(f"    Median counts/cell: {adata.obs['total_counts'].median():.0f}")
print(f"    Median MT%: {adata.obs['pct_counts_mt'].median():.1f}%")

# QC violin plots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, metric in zip(axes, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']):
    ax.violinplot(adata.obs[metric], showmedians=True)
    ax.set_title(metric)
    ax.set_xticks([])
plt.tight_layout()
plt.savefig(FIG_DIR / "qc_violin_prefilter.png", dpi=150, bbox_inches='tight')
plt.close()

# Filter cells
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs['n_genes_by_counts'] < 6000, :]
adata = adata[adata.obs['pct_counts_mt'] < 25, :]
adata = adata[adata.obs['total_counts'] > 500, :]

print(f"    After QC:  {adata.n_obs} cells × {adata.n_vars} genes")

# ─── 7. Normalization & HVG selection ────────────────────────────────────────
print("\n[6] Normalization & HVG selection...")
adata.layers["counts"] = adata.X.copy()  # keep raw counts for scVI
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    subset=False,
    layer="counts",
    flavor="seurat_v3",
    batch_key="sample"
)
print(f"    HVGs selected: {adata.var['highly_variable'].sum()}")

# PCA for initial visualization
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(adata, color='condition', ax=axes[0], title='Pre-scVI UMAP (condition)', show=False)
sc.pl.umap(adata, color='sample',    ax=axes[1], title='Pre-scVI UMAP (sample)',    show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "umap_prescvi.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Pre-scVI UMAP saved")

# Save processed AnnData
adata.write(PROC_DIR / "celiac_processed.h5ad")
print(f"    Processed AnnData saved → {PROC_DIR / 'celiac_processed.h5ad'}")

# ─── 8. scVI Setup ───────────────────────────────────────────────────────────
print("\n[7] Setting up scVI model...")
scvi.settings.seed = 42

# Use HVG subset for scVI
adata_scvi = adata[:, adata.var['highly_variable']].copy()
adata_scvi.X = adata_scvi.layers['counts']

scvi.model.SCVI.setup_anndata(
    adata_scvi,
    layer="counts",
    batch_key="sample",
    categorical_covariate_keys=["condition"]
)

model = scvi.model.SCVI(
    adata_scvi,
    n_latent=30,
    n_layers=2,
    n_hidden=128,
    gene_likelihood="nb",
    dispersion="gene"
)

print(f"    scVI model summary:")
print(f"      Input genes (HVGs): {adata_scvi.n_vars}")
print(f"      Latent dims: 30")
print(f"      Cells: {adata_scvi.n_obs}")
model.view_anndata_setup()

# ─── 9. GPU Training ─────────────────────────────────────────────────────────
print("\n[8] Training scVI on GPU (100 epochs)...")
use_gpu = torch.cuda.is_available()
model.train(
    max_epochs=100,
    accelerator="gpu" if use_gpu else "cpu",
    batch_size=512,
    early_stopping=True,
    early_stopping_patience=10,
    plan_kwargs={"lr": 1e-3}
)
print("    Training complete!")

# Save model
MODEL_DIR = Path("/home/harshney/celiac-immune-profiling/models/scvi_celiac")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save(str(MODEL_DIR), overwrite=True)
print(f"    Model saved → {MODEL_DIR}")

# Training loss plot
train_hist = model.history["elbo_train"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_hist, label='Train ELBO', color='steelblue')
if "elbo_validation" in model.history:
    ax.plot(model.history["elbo_validation"], label='Val ELBO', color='orange')
ax.set_xlabel("Epoch"); ax.set_ylabel("ELBO Loss")
ax.set_title("scVI Training — Celiac Dataset")
ax.legend(); plt.tight_layout()
plt.savefig(FIG_DIR / "scvi_training_loss.png", dpi=150, bbox_inches='tight')
plt.close()

# ─── 10. Extract Latent Embeddings ────────────────────────────────────────────
print("\n[9] Extracting scVI latent embeddings...")
adata.obsm['X_scvi'] = model.get_latent_representation(adata_scvi)
print(f"    Embedding shape: {adata.obsm['X_scvi'].shape}")

# UMAP on scVI embeddings
sc.pp.neighbors(adata, use_rep='X_scvi', n_neighbors=15)
sc.tl.umap(adata)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(adata, color='condition', ax=axes[0], title='scVI UMAP (condition)', show=False,
           palette={'Celiac': '#e74c3c', 'Healthy': '#3498db'})
sc.pl.umap(adata, color='sample',    ax=axes[1], title='scVI UMAP (sample)',    show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "scvi_umap.png", dpi=150, bbox_inches='tight')
plt.close()
print("    scVI UMAP saved")

# ─── 11. Patient-Level Aggregation ────────────────────────────────────────────
print("\n[10] Aggregating to patient-level embeddings...")
emb = pd.DataFrame(
    adata.obsm['X_scvi'],
    index=adata.obs_names,
    columns=[f"scvi_{i}" for i in range(30)]
)
emb['patient_id'] = adata.obs['patient_id'].values
emb['condition']  = adata.obs['condition'].values

# Strategy 1: simple mean pool
patient_emb = emb.groupby('patient_id').mean(numeric_only=True)
patient_meta = emb.groupby('patient_id')['condition'].first()
patient_emb['condition'] = patient_meta
patient_emb.to_csv(RESULTS_DIR / "scvi_patient_embeddings.csv")
print(f"    Patient embeddings (mean): {patient_emb.shape}")
print(f"    Patients:\n{patient_meta.to_dict()}")

# PCA of patient embeddings
from sklearn.decomposition import PCA
X_pat = patient_emb.drop(columns=['condition']).values
pca = PCA(n_components=min(2, X_pat.shape[0]-1))
X_pca = pca.fit_transform(X_pat)

fig, ax = plt.subplots(figsize=(7, 5))
colors = {'Celiac': '#e74c3c', 'Healthy': '#3498db'}
for cond, grp in patient_meta.groupby(patient_meta):
    idx = [list(patient_emb.index).index(p) for p in grp.index]
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=cond,
               c=colors.get(cond, 'gray'), s=150, edgecolors='k', zorder=3)
    for i, p in zip(idx, grp.index):
        ax.annotate(p, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, ha='center', va='bottom')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)" if X_pca.shape[1] > 1 else "PC2")
ax.set_title("scVI Patient-Level PCA — Celiac vs Healthy")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "scvi_patient_pca.png", dpi=150, bbox_inches='tight')
plt.close()

# ─── 12. Save final AnnData ───────────────────────────────────────────────────
adata.write(PROC_DIR / "celiac_with_scvi.h5ad")
print(f"\n    Final AnnData saved → {PROC_DIR / 'celiac_with_scvi.h5ad'}")

print("\n" + "="*65)
print(" STEP 1 COMPLETE")
print("="*65)
print(f"  Cells processed:  {adata.n_obs}")
print(f"  Patients:         {adata.obs['patient_id'].nunique()}")
print(f"  Conditions:       {adata.obs['condition'].value_counts().to_dict()}")
print(f"  Results in:       {RESULTS_DIR}")
print(f"  Figures in:       {FIG_DIR}")
print("  Next: Run step2_geneformer.py for foundation model embeddings")
print("="*65)
