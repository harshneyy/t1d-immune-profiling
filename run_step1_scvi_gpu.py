"""
STEP 1 — GPU scVI (Local Machine Version)
=========================================
Adapted from colab_notebooks/STEP1_GPU_scVI.py
No google.colab dependencies. Runs fully locally.

Usage:
    /home/harshney/ml_env/bin/python run_step1_scvi_gpu.py

Required files in DATA_DIR:
    seurat_metadata.csv
    T1D_Seurat_Object_Final_SCT_counts.mtx
    T1D_Seurat_Object_Final_SCT_counts_cells.txt
    T1D_Seurat_Object_Final_SCT_counts_genes.txt
"""

import os
import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data" / "raw" / "T1D_project"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_MTX   = DATA_DIR / "T1D_Seurat_Object_Final_SCT_counts.mtx"
CELLS_TXT = DATA_DIR / "T1D_Seurat_Object_Final_SCT_counts_cells.txt"
GENES_TXT = DATA_DIR / "T1D_Seurat_Object_Final_SCT_counts_genes.txt"
META_CSV  = DATA_DIR / "seurat_metadata.csv"

# Check all files exist
missing = [f for f in [RAW_MTX, CELLS_TXT, GENES_TXT, META_CSV] if not f.exists()]
if missing:
    print("❌ Missing data files:")
    for f in missing:
        print(f"   {f}")
    print(f"\nPlease place the 4 raw data files in: {DATA_DIR}")
    sys.exit(1)

print("✅ All data files found")
print([f.name for f in [RAW_MTX, CELLS_TXT, GENES_TXT, META_CSV]])

# ─── Imports ──────────────────────────────────────────────────────────────────
import scipy.io
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvi
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ─── Verify GPU ───────────────────────────────────────────────────────────────
print("\n=== GPU Check ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ No GPU detected — will run on CPU (slower)")

# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_COL    = "Sample_ID"
LABEL_COL     = "COND"
CELL_TYPE_COL = "Cluster_Annotation_Merged"
N_HVG         = 3000
N_LATENT      = 30
MAX_EPOCHS    = 100
SEED          = 42

# ─── Load Data ────────────────────────────────────────────────────────────────
print("\n=== Loading Data ===")
print("Loading sparse matrix …")
mat   = scipy.io.mmread(str(RAW_MTX)).tocsr()   # genes × cells
cells = pd.read_csv(str(CELLS_TXT), header=None)[0].tolist()
genes = pd.read_csv(str(GENES_TXT), header=None)[0].tolist()
meta  = pd.read_csv(str(META_CSV))
meta.columns = meta.columns.str.strip()
for c in meta.select_dtypes("object"):
    meta[c] = meta[c].str.strip()

adata = ad.AnnData(
    X=mat.T.tocsr(),
    obs=pd.DataFrame(index=cells),
    var=pd.DataFrame(index=genes)
)
meta_idx = meta.set_index(meta.columns[0])
shared   = adata.obs_names.intersection(meta_idx.index)
adata    = adata[shared].copy()
adata.obs = meta_idx.loc[shared]

print(f"AnnData: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
print(adata.obs[LABEL_COL].value_counts())

# ─── Train scVI on GPU ────────────────────────────────────────────────────────
print(f"\n=== Training scVI ({MAX_EPOCHS} epochs, {N_HVG} HVGs, {N_LATENT} latent dims) ===")
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3", subset=True)
print(f"After HVG filtering: {adata.shape}")

scvi.model.SCVI.setup_anndata(adata)
model = scvi.model.SCVI(adata, n_latent=N_LATENT, n_layers=2, gene_likelihood="nb")

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Training on: {accelerator.upper()} (~10-20 min on GPU) …")
model.train(max_epochs=MAX_EPOCHS, accelerator=accelerator)
print("Training done ✓")

# ─── Extract Patient Embeddings ───────────────────────────────────────────────
print("\n=== Extracting Embeddings ===")
latent      = model.get_latent_representation()
latent_cols = [f"scvi_{i:02d}" for i in range(N_LATENT)]
adata.obsm["X_scVI"] = latent

cell_info = adata.obs[[SAMPLE_COL, LABEL_COL, CELL_TYPE_COL]].reset_index(drop=True)
lat_df    = pd.DataFrame(latent, columns=latent_cols)
combined  = pd.concat([cell_info, lat_df], axis=1)

# Patient-level mean
patient_emb = (combined
               .groupby([SAMPLE_COL, LABEL_COL], observed=True)[latent_cols]
               .mean()
               .reset_index())
patient_emb.to_csv(RESULTS_DIR / "patient_scvi_embeddings_gpu.csv", index=False)

# Cell-type-aware wide embeddings
wide_parts = []
for ct, part in combined.groupby(CELL_TYPE_COL, observed=True):
    agg = part.groupby([SAMPLE_COL, LABEL_COL], observed=True)[latent_cols].mean()
    agg.columns = [f"{ct}__{c}" for c in latent_cols]
    wide_parts.append(agg)
ct_emb = (pd.concat(wide_parts, axis=1)
          .replace([np.inf, -np.inf], np.nan)
          .reset_index())
ct_emb.to_csv(RESULTS_DIR / "patient_celltype_scvi_embeddings_gpu.csv", index=False)
print(f"patient_emb: {patient_emb.shape}, celltype_emb: {ct_emb.shape}")

# ─── Classification ───────────────────────────────────────────────────────────
print("\n=== Classification (5-fold CV) ===")

def classify(df, tag):
    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL_COL])
    fc = [c for c in df.columns if c not in {SAMPLE_COL, LABEL_COL}]
    X  = df[fc].apply(pd.to_numeric, errors="coerce")
    pi = list(le.classes_).index("T1D")

    def sp(clf): return Pipeline([("i", SimpleImputer()), ("s", StandardScaler()), ("c", clf)])
    def tp(clf): return Pipeline([("i", SimpleImputer()), ("c", clf)])

    models = {
        "logistic": sp(LogisticRegression(max_iter=500, class_weight="balanced")),
        "svm":      sp(SVC(kernel="rbf", class_weight="balanced", probability=True)),
        "rf":       tp(RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced")),
        "hgb":      tp(HistGradientBoostingClassifier(random_state=SEED)),
    }
    cv   = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for nm, clf in models.items():
        for fold, (tr, te) in enumerate(cv.split(X, y), 1):
            clf.fit(X.iloc[tr], y[tr])
            pred  = clf.predict(X.iloc[te])
            proba = clf.predict_proba(X.iloc[te])[:, pi]
            rows.append({
                "feature_set": tag, "model": nm, "fold": fold,
                "accuracy":          accuracy_score(y[te], pred),
                "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                "f1":                f1_score(y[te], pred, pos_label=pi),
                "roc_auc":           roc_auc_score((y[te] == pi).astype(int), proba)
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"clf_{tag}.csv", index=False)
    print(f"\n=== {tag} ===")
    print(out.groupby("model")[["roc_auc", "accuracy", "f1"]].mean().round(4))
    return out

r1 = classify(patient_emb, "scvi_patient_gpu")
r2 = classify(ct_emb,      "scvi_celltype_gpu")
all_clf = pd.concat([r1, r2])
all_clf.to_csv(RESULTS_DIR / "classification_all_gpu.csv", index=False)
print("\n✅ Classification done")

# ─── Subtype Discovery ────────────────────────────────────────────────────────
print("\n=== T1D Subtype Discovery ===")
t1d   = patient_emb[patient_emb[LABEL_COL] == "T1D"].copy()
fc    = [c for c in t1d.columns if c.startswith("scvi_")]
X_t1d = t1d[fc].apply(pd.to_numeric, errors="coerce")
Xs    = StandardScaler().fit_transform(SimpleImputer().fit_transform(X_t1d))

print("Silhouette scores:")
sil = {}
for k in [2, 3, 4]:
    lbl    = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit_predict(Xs)
    sil[k] = silhouette_score(Xs, lbl)
    print(f"  k={k}: {sil[k]:.4f}")

best_k = max(sil, key=sil.get)
print(f"Best k={best_k}")
km = KMeans(n_clusters=best_k, random_state=SEED, n_init=50)
t1d["subtype"] = [f"Subtype_{l+1}" for l in km.fit_predict(Xs)]
t1d.to_csv(RESULTS_DIR / "t1d_subtypes_gpu.csv", index=False)
print(t1d["subtype"].value_counts())

# ─── PCA Plots ────────────────────────────────────────────────────────────────
print("\n=== Generating Figures ===")
pca    = PCA(2, random_state=SEED)
c_t1d  = pca.fit_transform(Xs)
t1d["PC1"], t1d["PC2"] = c_t1d[:, 0], c_t1d[:, 1]

Xs_all = StandardScaler().fit_transform(
    SimpleImputer().fit_transform(
        patient_emb[[c for c in patient_emb.columns if c.startswith("scvi_")]]
        .apply(pd.to_numeric, errors="coerce")
    )
)
c_all = PCA(2, random_state=SEED).fit_transform(Xs_all)
patient_emb["PC1"], patient_emb["PC2"] = c_all[:, 0], c_all[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=patient_emb, x="PC1", y="PC2", hue=LABEL_COL, s=80, ax=axes[0])
axes[0].set_title("GPU scVI — T1D vs Healthy")
sns.scatterplot(data=t1d, x="PC1", y="PC2", hue="subtype", s=80, ax=axes[1])
axes[1].set_title(f"T1D subtypes (k={best_k})")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "scvi_gpu_pca.png", dpi=200)
plt.close()

plt.figure(figsize=(9, 5))
sns.barplot(data=all_clf, x="feature_set", y="roc_auc", hue="model", errorbar="sd")
plt.ylim(0.5, 1.0)
plt.title("GPU scVI ROC-AUC")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "scvi_gpu_roc_auc.png", dpi=200)
plt.close()
print("Figures saved ✓")

print("\n" + "="*60)
print("✅ STEP 1 COMPLETE")
print("="*60)
print(f"Outputs in {RESULTS_DIR}/:")
print("  classification_all_gpu.csv")
print("  t1d_subtypes_gpu.csv")
print("  patient_scvi_embeddings_gpu.csv")
print(f"Figures in {FIGURES_DIR}/:")
print("  scvi_gpu_pca.png")
print("  scvi_gpu_roc_auc.png")
print("\nNext: run  python run_step2_geneformer_gpu.py")
