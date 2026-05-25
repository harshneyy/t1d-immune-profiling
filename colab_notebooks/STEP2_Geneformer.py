"""
STEP 2 — Geneformer Inference on Google Colab
Run AFTER Step 1 is complete.

INSTRUCTIONS:
1. Runtime > Change runtime type > GPU (T4 or A100)
2. Same Drive folder: MyDrive/T1D_project/ (same 4 files as Step 1)
3. Paste each CELL block separately into Colab — do NOT run as one script
4. CELL 9 takes 30-60 min — keep the tab open
"""

# ─── CELL 1 ─── Mount Drive ─────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─── CELL 2 ─── Install packages ────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "datasets", "transformers", "anndata",
                "scanpy", "scikit-learn", "pandas", "matplotlib", "seaborn",
                "mygene"], check=True)
subprocess.run(["pip", "install", "-q",
                "git+https://huggingface.co/ctheodoris/Geneformer"], check=True)
print("Packages installed ✓")

# ─── CELL 3 ─── Verify GPU ──────────────────────────────────────────────────
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "⚠ NO GPU!")

# ─── CELL 4 ─── Paths & constants ───────────────────────────────────────────
from pathlib import Path
import numpy as np, pandas as pd

DRIVE_DIR     = Path("/content/drive/MyDrive/T1D_project")
WORK_DIR      = Path("/content/t1d_gf_work")
RESULTS_DIR   = WORK_DIR / "results"
FIGURES_DIR   = WORK_DIR / "figures"
TOK_DIR       = WORK_DIR / "tokenized"
for d in [RESULTS_DIR, FIGURES_DIR, TOK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_COL    = "Sample_ID"
LABEL_COL     = "COND"
CELL_TYPE_COL = "Cluster_Annotation_Merged"
SEED          = 42
print("Dirs ready ✓")

# ─── CELL 5 ─── Load AnnData from raw files ─────────────────────────────────
import scipy.io, anndata as ad, scanpy as sc

RAW_MTX   = DRIVE_DIR / "T1D_Seurat_Object_Final_SCT_counts.mtx"
CELLS_TXT = DRIVE_DIR / "T1D_Seurat_Object_Final_SCT_counts_cells.txt"
GENES_TXT = DRIVE_DIR / "T1D_Seurat_Object_Final_SCT_counts_genes.txt"
META_CSV  = DRIVE_DIR / "seurat_metadata.csv"

print("Loading matrix …")
mat   = scipy.io.mmread(str(RAW_MTX)).tocsr()
cells = pd.read_csv(str(CELLS_TXT), header=None)[0].tolist()
genes = pd.read_csv(str(GENES_TXT), header=None)[0].tolist()
meta  = pd.read_csv(str(META_CSV))
meta.columns = meta.columns.str.strip()
for c in meta.select_dtypes("object"):
    meta[c] = meta[c].str.strip()

adata = ad.AnnData(X=mat.T.tocsr(),
                   obs=pd.DataFrame(index=cells),
                   var=pd.DataFrame(index=genes))
meta_idx = meta.set_index(meta.columns[0])
shared   = adata.obs_names.intersection(meta_idx.index)
adata    = adata[shared].copy()
adata.obs = meta_idx.loc[shared]

print(f"AnnData: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
print(adata.obs[LABEL_COL].value_counts())

# ─── CELL 6 ─── Map gene symbols → Ensembl IDs ──────────────────────────────
import mygene
print("Querying MyGene.info … (~1-2 min)")
mg      = mygene.MyGeneInfo()
symbols = adata.var_names.tolist()
results = mg.querymany(symbols, scopes="symbol", fields="ensembl.gene",
                       species="human", verbose=False)

sym2ens = {}
for r in results:
    if "ensembl" not in r:
        continue
    ens = r["ensembl"]
    if isinstance(ens, list):
        ens = ens[0]
    eid = ens.get("gene", "")
    if eid:
        sym2ens[r["query"]] = eid

mapped_sym = [s for s in symbols if s in sym2ens]
mapped_ens = [sym2ens[s] for s in mapped_sym]
print(f"Mapped: {len(mapped_ens):,} / {len(symbols):,} genes")

adata_gf           = adata[:, mapped_sym].copy()
adata_gf.var_names = mapped_ens
adata_gf.var_names_make_unique()
print(f"adata_gf: {adata_gf.shape}")

# ─── CELL 7 ─── Prepare h5ad & tokenize ─────────────────────────────────────
import scipy.sparse as sp
from geneformer import TranscriptomeTokenizer

# Required fields
adata_gf.var["ensembl_id"] = adata_gf.var_names.tolist()
adata_gf.obs["n_counts"]   = np.array(adata_gf.X.sum(axis=1)).ravel().astype(int)
adata_gf.obs[LABEL_COL]     = adata_gf.obs[LABEL_COL].astype(str)
adata_gf.obs[SAMPLE_COL]    = adata_gf.obs[SAMPLE_COL].astype(str)
adata_gf.obs[CELL_TYPE_COL] = adata_gf.obs[CELL_TYPE_COL].astype(str)

if not sp.issparse(adata_gf.X):
    adata_gf.X = sp.csr_matrix(adata_gf.X)

tmp_h5ad = str(WORK_DIR / "for_geneformer.h5ad")
adata_gf.write_h5ad(tmp_h5ad)
print(f"Saved: {tmp_h5ad}")

tk = TranscriptomeTokenizer(
    custom_attr_name_dict={
        LABEL_COL:     LABEL_COL,
        SAMPLE_COL:    SAMPLE_COL,
        CELL_TYPE_COL: CELL_TYPE_COL,
    },
    nproc=2,
)
print("Tokenizing … (~5-10 min)")
tk.tokenize_data(str(WORK_DIR), str(TOK_DIR), "t1d_tokenized", file_format="h5ad")

import os
print("Tokenized files:", os.listdir(TOK_DIR))

# ─── CELL 8 ─── Confirm dataset path before extracting ──────────────────────
import os
# Find the actual dataset folder name (might vary by Geneformer version)
tok_files = os.listdir(TOK_DIR)
print("Files in tokenized dir:", tok_files)

# Look for .dataset folder
dataset_path = None
for f in tok_files:
    if f.endswith(".dataset"):
        dataset_path = str(TOK_DIR / f)
        break

if dataset_path is None:
    raise FileNotFoundError(f"No .dataset folder found in {TOK_DIR}. Files: {tok_files}")

print(f"Dataset path: {dataset_path}")

# ─── CELL 9 ─── Extract Geneformer embeddings (~30-60 min) ──────────────────
from geneformer import EmbExtractor

embex = EmbExtractor(
    model_type         = "Pretrained",
    num_classes        = 0,
    emb_mode           = "cell",
    layer_to_quant     = -1,
    emb_label          = [LABEL_COL, SAMPLE_COL, CELL_TYPE_COL],
    max_ncells         = None,
    forward_batch_size = 100,
    nproc              = 2,
)
print("Extracting embeddings … keep this tab open, takes 30-60 min")
embs = embex.extract_embs(
    "ctheodoris/Geneformer",
    dataset_path,
    str(RESULTS_DIR),
    "geneformer_embs",
)
print(f"Done! Shape: {embs.shape}")
embs.to_csv(RESULTS_DIR / "geneformer_cell_embeddings.csv", index=False)

# ─── CELL 10 ─── Aggregate to patient level ──────────────────────────────────
gf       = pd.read_csv(RESULTS_DIR / "geneformer_cell_embeddings.csv")
emb_cols = [c for c in gf.columns if c not in {LABEL_COL, SAMPLE_COL, CELL_TYPE_COL}]
print(f"Embedding dims: {len(emb_cols)}")

# Patient-level: mean of ALL cells per patient
pat_emb = gf.groupby([SAMPLE_COL, LABEL_COL], observed=True)[emb_cols].mean().reset_index()
pat_emb.to_csv(RESULTS_DIR / "patient_geneformer_embeddings.csv", index=False)
print(f"Patient embeddings: {pat_emb.shape}")

# Cell-type-aware: aggregate per patient per cell type FIRST, then go wide
# FIX: aggregate before concat to avoid duplicate index error
wide_parts = []
for ct, part in gf.groupby(CELL_TYPE_COL, observed=True):
    agg = part.groupby([SAMPLE_COL, LABEL_COL], observed=True)[emb_cols].mean()
    agg.columns = [f"{ct}__{c}" for c in emb_cols]
    wide_parts.append(agg)
pat_ct_emb = pd.concat(wide_parts, axis=1).replace([np.inf, -np.inf], np.nan).reset_index()
pat_ct_emb.to_csv(RESULTS_DIR / "patient_geneformer_celltype_embeddings.csv", index=False)
print(f"Cell-type-aware embeddings: {pat_ct_emb.shape}")

# ─── CELL 11 ─── Classification (PCA to 30 dims first — 512 dims → manageable) ─
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def classify_gf(df, tag, n_pca=30):
    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL_COL])
    fc = [c for c in df.columns if c not in {SAMPLE_COL, LABEL_COL}]
    X  = df[fc].apply(pd.to_numeric, errors="coerce")
    pi = list(le.classes_).index("T1D")
    n  = min(n_pca, X.shape[1] - 1, X.shape[0] - 1)

    def sp(clf): return Pipeline([("i", SimpleImputer()), ("s", StandardScaler()),
                                   ("pca", PCA(n_components=n, random_state=SEED)), ("c", clf)])
    def tp(clf): return Pipeline([("i", SimpleImputer()),
                                   ("pca", PCA(n_components=n, random_state=SEED)), ("c", clf)])
    models = {
        "logistic": sp(LogisticRegression(max_iter=500, class_weight="balanced")),
        "svm":      sp(SVC(kernel="rbf", class_weight="balanced", probability=True)),
        "rf":       tp(RandomForestClassifier(300, random_state=SEED, class_weight="balanced")),
        "hgb":      tp(HistGradientBoostingClassifier(random_state=SEED)),
    }
    cv   = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for nm, clf in models.items():
        for fold, (tr, te) in enumerate(cv.split(X, y), 1):
            clf.fit(X.iloc[tr], y[tr])
            pred  = clf.predict(X.iloc[te])
            proba = clf.predict_proba(X.iloc[te])[:, pi]
            rows.append({"feature_set": tag, "model": nm, "fold": fold,
                         "accuracy":          accuracy_score(y[te], pred),
                         "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                         "f1":                f1_score(y[te], pred, pos_label=pi),
                         "roc_auc":           roc_auc_score((y[te] == pi).astype(int), proba)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"clf_{tag}.csv", index=False)
    print(f"\n=== {tag} ===")
    print(out.groupby("model")[["roc_auc", "accuracy", "f1"]].mean().round(4))
    return out

r1      = classify_gf(pat_emb,    "geneformer_patient")
r2      = classify_gf(pat_ct_emb, "geneformer_celltype")
all_clf = pd.concat([r1, r2])
all_clf.to_csv(RESULTS_DIR / "classification_geneformer.csv", index=False)
print("\nClassification done ✓")

# ─── CELL 12 ─── PCA plot + ROC-AUC bar + download ──────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA

Xs = StandardScaler().fit_transform(SimpleImputer().fit_transform(
     pat_emb[[c for c in pat_emb.columns if c not in {SAMPLE_COL, LABEL_COL}]]
     .apply(pd.to_numeric, errors="coerce")))
coords = PCA(2, random_state=SEED).fit_transform(Xs)
pat_emb["PC1"], pat_emb["PC2"] = coords[:, 0], coords[:, 1]

plt.figure(figsize=(7, 5))
sns.scatterplot(data=pat_emb, x="PC1", y="PC2", hue=LABEL_COL, s=80)
plt.title("Geneformer patient embeddings — T1D vs Healthy")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "geneformer_pca.png", dpi=200); plt.close()

plt.figure(figsize=(9, 5))
sns.barplot(data=all_clf, x="feature_set", y="roc_auc", hue="model", errorbar="sd")
plt.ylim(0.5, 1.0); plt.title("Geneformer ROC-AUC"); plt.tight_layout()
plt.savefig(FIGURES_DIR / "geneformer_roc_auc.png", dpi=200); plt.close()
print("Figures saved ✓")

import shutil
shutil.make_archive("/content/t1d_geneformer_results", "zip", str(WORK_DIR))
from google.colab import files
files.download("/content/t1d_geneformer_results.zip")
print("✅ All done!")
