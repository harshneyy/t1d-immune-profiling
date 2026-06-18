#!/usr/bin/env python3
"""
Step 2 — Geneformer Embedding Extraction for Celiac Disease
Uses the processed AnnData from Step 1 (celiac_with_scvi.h5ad)

This script:
  1. Loads processed celiac AnnData
  2. Converts to Geneformer tokenized dataset (rank-value encoding)
  3. Extracts 512-dim cell embeddings (zero-shot, no fine-tuning)
  4. Aggregates to patient level
  5. Saves embeddings for Step 3 classification
"""

import os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Add Geneformer to path (same location as T1D project)
GENEFORMER_PATH = "/home/harshney/t1d-immune-profiling/Geneformer"
sys.path.insert(0, GENEFORMER_PATH)

from geneformer import TranscriptomeTokenizer, EmbExtractor

# ─── Paths ────────────────────────────────────────────────────────────────────
PROC_DIR    = Path("/home/harshney/celiac-immune-profiling/data/processed")
RESULTS_DIR = Path("/home/harshney/celiac-immune-profiling/results")
FIG_DIR     = Path("/home/harshney/celiac-immune-profiling/figures")
TOKEN_DIR   = Path("/home/harshney/celiac-immune-profiling/data/geneformer_tokens")
EMB_DIR     = Path("/home/harshney/celiac-immune-profiling/data/geneformer_embeddings")

for d in [TOKEN_DIR, EMB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model ID — use HuggingFace Hub ID (same as T1D project, cached locally)
GF_MODEL = "ctheodoris/Geneformer"  # will use ~/.cache/huggingface automatically

print("="*65)
print(" CELIAC scRNA-seq PIPELINE  —  Step 2: Geneformer Embeddings")
print("="*65)
print(f"\n[GPU] CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
print(f"[GF]  Model path: {GF_MODEL}")

# ─── 1. Load processed AnnData ────────────────────────────────────────────────
print("\n[1] Loading processed AnnData...")
INPUT_H5AD = PROC_DIR / "celiac_with_scvi.h5ad"
if not INPUT_H5AD.exists():
    # Fallback to basic processed file
    INPUT_H5AD = PROC_DIR / "celiac_processed.h5ad"
adata = sc.read_h5ad(INPUT_H5AD)
print(f"    Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"    Conditions: {adata.obs['condition'].value_counts().to_dict()}")

# ─── 2. Ensure raw counts in .X ───────────────────────────────────────────────
print("\n[2] Preparing raw counts for Geneformer...")
if 'counts' in adata.layers:
    adata_gf = adata.copy()
    adata_gf.X = adata_gf.layers['counts']
    print("    Using .layers['counts'] as raw counts")
else:
    print("    WARNING: No 'counts' layer, using .X as-is")
    adata_gf = adata.copy()

# Geneformer requires integer counts
import scipy.sparse as sp
if sp.issparse(adata_gf.X):
    adata_gf.X = adata_gf.X.astype(int)
else:
    adata_gf.X = adata_gf.X.astype(int)

# Add n_counts column (sum per cell)
adata_gf.obs['n_counts'] = np.array(adata_gf.X.sum(axis=1)).flatten()
print(f"    Median n_counts: {adata_gf.obs['n_counts'].median():.0f}")

# ─── 3. Save as loom for Geneformer tokenizer ────────────────────────────────
print("\n[3] Saving to loom format for tokenizer...")
LOOM_PATH = TOKEN_DIR / "celiac.loom"

# Map gene symbols to Ensembl IDs using Geneformer's own local dictionary
# No API call needed — avoids mygene.info server errors
import pickle

GENE_NAME_ID_DICT = Path(GENEFORMER_PATH) / "geneformer" / "gene_name_id_dict_gc104M.pkl"
if not GENE_NAME_ID_DICT.exists():
    # Fallback to 30M dict
    GENE_NAME_ID_DICT = Path(GENEFORMER_PATH) / "geneformer" / "gene_dictionaries_30m" / "gene_name_id_dict_gc30M.pkl"

print(f"    Loading local gene→Ensembl mapping from: {GENE_NAME_ID_DICT.name}")
with open(GENE_NAME_ID_DICT, "rb") as f:
    gene_name_id_dict = pickle.load(f)

print(f"    Dictionary size: {len(gene_name_id_dict):,} genes")
adata_gf.var['ensembl_id'] = [gene_name_id_dict.get(g, '') for g in adata_gf.var_names]
n_mapped = (adata_gf.var['ensembl_id'] != '').sum()
print(f"    Mapped {n_mapped}/{adata_gf.n_vars} genes to Ensembl IDs ({n_mapped/adata_gf.n_vars:.1%})")

# Save loom
adata_gf.write_loom(str(LOOM_PATH), write_obsm_varm=False)
print(f"    Loom saved → {LOOM_PATH}")


# ─── 4. Tokenise with Geneformer ──────────────────────────────────────────────
TOKEN_DATASET = TOKEN_DIR / "celiac_tokens.dataset"
if TOKEN_DATASET.exists():
    print("\n[4] Token dataset already exists, skipping tokenisation...")
    print(f"    Found: {TOKEN_DATASET}")
else:
    print("\n[4] Tokenising with Geneformer (rank-value encoding)...")
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "patient_id": "patient_id",
            "condition": "condition",
            "sample": "sample"
        },
        nproc=4
    )
    tk.tokenize_data(
        str(TOKEN_DIR),
        str(TOKEN_DIR),
        "celiac_tokens",
        file_format="loom"
    )
    print(f"    Tokens saved → {TOKEN_DATASET}")

# ─── 5. Extract Geneformer embeddings ────────────────────────────────────────
print("\n[5] Extracting Geneformer embeddings (zero-shot)...")
print(f"    This may take a while for large datasets...")

import torch as _torch
_torch.cuda.empty_cache()
print(f"    VRAM free before inference: {_torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

emb_extractor = EmbExtractor(
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cell",
    emb_layer=-1,
    emb_label=["condition", "patient_id"],
    max_ncells=None,
    forward_batch_size=8,
    nproc=4
)

emb_extractor.extract_embs(
    model_directory=GF_MODEL,
    input_data_file=str(TOKEN_DIR / "celiac_tokens.dataset"),
    output_directory=str(EMB_DIR),
    output_prefix="celiac_gf"
)
print(f"    Embeddings saved → {EMB_DIR}")

# ─── 6. Load & process embeddings ────────────────────────────────────────────
print("\n[6] Loading and aggregating embeddings...")

# Find the saved embedding file
emb_files = list(EMB_DIR.glob("celiac_gf*.csv")) + list(EMB_DIR.glob("celiac_gf*.npy"))
if not emb_files:
    emb_files = list(EMB_DIR.glob("*.csv"))

print(f"    Found embedding files: {emb_files}")
emb_df = pd.read_csv(emb_files[0], index_col=0)
print(f"    Embedding shape: {emb_df.shape}")

# Separate embeddings from metadata
meta_cols = ['condition', 'patient_id', 'sample']
meta_cols = [c for c in meta_cols if c in emb_df.columns]
emb_features = [c for c in emb_df.columns if c not in meta_cols]

# Patient-level mean aggregation
patient_emb_gf = emb_df.groupby('patient_id')[emb_features].mean()
patient_meta_gf = emb_df.groupby('patient_id')['condition'].first()
patient_emb_gf['condition'] = patient_meta_gf
patient_emb_gf.to_csv(RESULTS_DIR / "geneformer_patient_embeddings.csv")
print(f"    Patient embeddings: {patient_emb_gf.shape}")

# ─── 7. PCA visualisation ─────────────────────────────────────────────────────
print("\n[7] Visualising Geneformer cell embeddings...")
X_emb = emb_df[emb_features].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_emb)

colors_map = {'Celiac': '#e74c3c', 'Healthy': '#3498db'}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cell-level PCA
if 'condition' in emb_df.columns:
    for cond, grp in emb_df.groupby('condition'):
        idx = grp.index
        axes[0].scatter(X_pca[emb_df.index.isin(idx), 0],
                        X_pca[emb_df.index.isin(idx), 1],
                        label=cond, c=colors_map.get(cond, 'gray'),
                        alpha=0.4, s=8)
axes[0].set_title("Geneformer Cell-Level PCA (Celiac vs Healthy)")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[0].legend()

# Patient-level PCA
X_pat = patient_emb_gf.drop(columns=['condition']).values
pca2 = PCA(n_components=min(2, X_pat.shape[0]-1))
X_pat_pca = pca2.fit_transform(X_pat)
for cond, grp in patient_meta_gf.groupby(patient_meta_gf):
    idx = [list(patient_emb_gf.index).index(p) for p in grp.index]
    axes[1].scatter(X_pat_pca[idx, 0],
                    X_pat_pca[idx, 1] if X_pat_pca.shape[1] > 1 else [0]*len(idx),
                    label=cond, c=colors_map.get(cond, 'gray'),
                    s=200, edgecolors='k', zorder=3)
    for i, p in zip(idx, grp.index):
        axes[1].annotate(p, (X_pat_pca[i, 0], X_pat_pca[i, 1] if X_pat_pca.shape[1] > 1 else 0),
                         fontsize=9, ha='center', va='bottom')
axes[1].set_title("Geneformer Patient-Level PCA")
axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "geneformer_pca.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"    Geneformer PCA saved → {FIG_DIR / 'geneformer_pca.png'}")

print("\n" + "="*65)
print(" STEP 2 COMPLETE")
print("="*65)
print(f"  Patient embeddings: {patient_emb_gf.shape}")
print(f"  Files in:  {RESULTS_DIR}")
print(f"  Figures in: {FIG_DIR}")
print("  Next: Run step3_classify_compare.py")
print("="*65)
