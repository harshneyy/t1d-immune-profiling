#!/usr/bin/env python3
"""
Step 3 — Classification, Cross-Disease Comparison & Visualisation
Celiac (GSE315138) + T1D cross-disease analysis

Fixes applied vs original:
  - Reads T1D embeddings from correct files (patient_scvi_embeddings_gpu.csv,
    patient_geneformer_embeddings.csv) with their actual column formats
  - Handles 4-patient LOO-CV gracefully
  - Correctly parses Geneformer EmbExtractor output format
  - Cross-disease PCA with T1D + Celiac + Healthy in shared space
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

CELIAC_RESULTS = Path("/home/harshney/celiac-immune-profiling/results")
CELIAC_EMB_DIR = Path("/home/harshney/celiac-immune-profiling/data/geneformer_embeddings")
T1D_RESULTS    = Path("/home/harshney/t1d-immune-profiling/results")
FIG_DIR        = Path("/home/harshney/celiac-immune-profiling/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("="*65)
print(" CELIAC PIPELINE — Step 3: Classification & Cross-Disease")
print("="*65)

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)":           SVC(kernel='rbf', probability=True, random_state=42),
}

def evaluate_loocv(X, y, classifiers, label=""):
    """Leave-One-Out CV — correct for n=4 patients."""
    results = []
    loo = LeaveOneOut()
    n = len(y)
    print(f"\n  [{label}] n={n} patients, running LOO-CV...")
    for clf_name, clf in classifiers.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        y_pred_proba = np.zeros(n)
        y_pred       = np.zeros(n, dtype=int)
        for train_idx, test_idx in loo.split(X):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr       = y[train_idx]
            pipe.fit(Xtr, ytr)
            y_pred_proba[test_idx] = pipe.predict_proba(Xte)[:, 1]
            y_pred[test_idx]       = pipe.predict(Xte)
        try:
            auc = roc_auc_score(y, y_pred_proba)
        except Exception:
            auc = float('nan')
        acc = accuracy_score(y, y_pred)
        f1  = f1_score(y, y_pred, zero_division=0)
        print(f"    {clf_name:25s}  AUC={auc:.3f}  Acc={acc:.3f}  F1={f1:.3f}")
        results.append({
            "Feature Set": label, "Classifier": clf_name,
            "ROC-AUC": round(auc, 4), "Accuracy": round(acc, 4), "F1": round(f1, 4)
        })
    return results

all_results = []

# ─── 1. Celiac scVI classification ────────────────────────────────────────────
print("\n[1] Celiac scVI patient classification...")
scvi_path = CELIAC_RESULTS / "scvi_patient_embeddings.csv"
if scvi_path.exists():
    scvi_emb = pd.read_csv(scvi_path, index_col=0)
    print(f"    Loaded: {scvi_emb.shape} | cols: {list(scvi_emb.columns[:5])}...")
    feat_cols = [c for c in scvi_emb.columns if c not in ['condition', 'patient_id']]
    y = (scvi_emb['condition'] == 'Celiac').astype(int).values
    X = scvi_emb[feat_cols].values
    res = evaluate_loocv(X, y, CLASSIFIERS, "Celiac-scVI")
    all_results.extend(res)
else:
    print("    WARNING: scvi_patient_embeddings.csv not found, skipping")

# ─── 2. Celiac Geneformer classification ──────────────────────────────────────
print("\n[2] Celiac Geneformer patient classification...")

# EmbExtractor saves: <prefix>_embs_cell.csv  OR  <prefix>.csv
# Try all likely output filenames
gf_emb_files = (
    list(CELIAC_EMB_DIR.glob("celiac_gf*.csv")) +
    list(CELIAC_RESULTS.glob("geneformer_patient*.csv"))
)
gf_cell_file = None
for f in gf_emb_files:
    if f.stat().st_size > 10000:   # must be non-trivial
        gf_cell_file = f
        break

if gf_cell_file:
    print(f"    Reading: {gf_cell_file}")
    gf_raw = pd.read_csv(gf_cell_file)
    print(f"    Raw shape: {gf_raw.shape}")
    print(f"    Columns sample: {list(gf_raw.columns[:6])}")

    # EmbExtractor output format: numeric embedding cols + label cols
    label_cols = [c for c in gf_raw.columns
                  if c in ['condition', 'patient_id', 'COND', 'Sample_ID',
                            'label', 'cell_type'] or not c.replace('.','').replace('-','').isnumeric()]
    emb_cols   = [c for c in gf_raw.columns if c not in label_cols]
    print(f"    Embedding dims: {len(emb_cols)}, label cols: {label_cols}")

    # Identify condition column
    cond_col = None
    for c in ['condition', 'COND']:
        if c in gf_raw.columns:
            cond_col = c
            break

    if cond_col and len(emb_cols) > 0:
        # Patient-level aggregation (mean per patient)
        pid_col = 'patient_id' if 'patient_id' in gf_raw.columns else None
        if pid_col:
            pat_emb = gf_raw.groupby(pid_col)[emb_cols].mean()
            pat_cond = gf_raw.groupby(pid_col)[cond_col].first()
        else:
            # No patient_id → group by condition
            pat_emb = gf_raw.groupby(cond_col)[emb_cols].mean()
            pat_cond = pat_emb.index.to_series()

        pat_emb['condition'] = pat_cond.values
        pat_emb.to_csv(CELIAC_RESULTS / "geneformer_patient_embeddings.csv")

        y_gf = (pat_cond == 'Celiac').astype(int).values
        # PCA to reduce 512 dims before LOO-CV with n=4
        X_gf_r = PCA(n_components=min(3, len(y_gf)-1)).fit_transform(pat_emb[emb_cols].values)
        res = evaluate_loocv(X_gf_r, y_gf, CLASSIFIERS, "Celiac-Geneformer")
        all_results.extend(res)
    else:
        print("    WARNING: could not identify condition column in Geneformer output")
else:
    print("    WARNING: No Geneformer embedding file found yet — Step 2 may still be running")
    print("    Skipping Geneformer classification, continuing with scVI only")

# ─── 3. Results table ─────────────────────────────────────────────────────────
if all_results:
    results_df = pd.DataFrame(all_results).sort_values("ROC-AUC", ascending=False)
    results_df.to_csv(CELIAC_RESULTS / "celiac_classification_results.csv", index=False)
    print(f"\n[3] Classification Results:")
    print(results_df.to_string(index=False))

# ─── 4. Cross-disease comparison: T1D + Celiac + Healthy ──────────────────────
print("\n[4] Cross-disease embedding comparison (T1D vs Celiac vs Healthy)...")

# T1D scVI embeddings: patient_scvi_embeddings_gpu.csv
# Format: patient_id col + scvi_0..scvi_N + condition
t1d_scvi_path   = T1D_RESULTS / "patient_scvi_embeddings_gpu.csv"
celiac_scvi_path = CELIAC_RESULTS / "scvi_patient_embeddings.csv"

if t1d_scvi_path.exists() and celiac_scvi_path.exists():
    t1d_raw    = pd.read_csv(t1d_scvi_path, index_col=0)
    celiac_raw = pd.read_csv(celiac_scvi_path, index_col=0)

    print(f"    T1D shape: {t1d_raw.shape} | cols: {list(t1d_raw.columns[:5])}")
    print(f"    Celiac shape: {celiac_raw.shape} | cols: {list(celiac_raw.columns[:5])}")

    # Find numeric embedding columns in each
    def get_emb_cols(df):
        cond_like = ['condition', 'COND', 'label', 'patient_id', 'Sample_ID',
                     'cell_type', 'celltype', 'subtype']
        return [c for c in df.columns if c not in cond_like]

    t1d_emb_cols    = get_emb_cols(t1d_raw)
    celiac_emb_cols = get_emb_cols(celiac_raw)

    # Find condition column
    def get_cond(df):
        for c in ['condition', 'COND']:
            if c in df.columns:
                return df[c].values
        return None

    t1d_cond    = get_cond(t1d_raw)
    celiac_cond = get_cond(celiac_raw)

    if t1d_cond is None or celiac_cond is None:
        print("    WARNING: Could not find condition column, skipping cross-disease plot")
    elif len(t1d_emb_cols) != len(celiac_emb_cols):
        print(f"    WARNING: Embedding dims differ — T1D:{len(t1d_emb_cols)} vs Celiac:{len(celiac_emb_cols)}")
        print("    Trying PCA alignment...")
        # Use PCA on each separately, then combine in 2D
        pca_t1d    = PCA(n_components=2).fit_transform(t1d_raw[t1d_emb_cols].values)
        pca_celiac = PCA(n_components=2).fit_transform(celiac_raw[celiac_emb_cols].values)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        color_map = {'T1D': '#e74c3c', 'Celiac': '#f39c12',
                     'Healthy': '#27ae60', 'H': '#27ae60', 'D': '#e74c3c'}

        for ax, (X_pca, labels, title) in zip(axes, [
            (pca_t1d, t1d_cond, "T1D Dataset (scVI PCA)"),
            (pca_celiac, celiac_cond, "Celiac Dataset (scVI PCA)")
        ]):
            unique = np.unique(labels)
            for lbl in unique:
                idx = labels == lbl
                display = 'T1D' if lbl in ['D', 'T1D', 'Diabetes'] else \
                          'Healthy' if lbl in ['H', 'Healthy', 'Control'] else lbl
                ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                           label=display, c=color_map.get(lbl, color_map.get(display, 'gray')),
                           s=180, edgecolors='k', zorder=3)
            for i, p in enumerate(labels):
                ax.annotate(str(t1d_raw.index[i] if title.startswith('T1D') else celiac_raw.index[i]),
                            (X_pca[i, 0], X_pca[i, 1]), fontsize=7, ha='center', va='bottom')
            ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

        plt.suptitle("scVI Patient Embeddings: T1D vs Celiac (Separate PCA)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cross_disease_pca.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved side-by-side PCA: {FIG_DIR}/cross_disease_pca.png")
    else:
        # Same dims — joint PCA
        t1d_X    = t1d_raw[t1d_emb_cols].values
        celiac_X = celiac_raw[celiac_emb_cols].values

        # Map labels
        t1d_mapped    = np.array(['T1D' if l in ['D','T1D','Diabetes'] else 'Healthy'
                                  for l in t1d_cond])
        celiac_mapped = np.array(['Celiac' if l in ['C','Celiac','celiac'] else 'Healthy'
                                  for l in celiac_cond])

        # Exclude T1D healthy from joint to avoid duplication
        t1d_disease_mask = t1d_mapped == 'T1D'
        X_all   = np.vstack([t1d_X[t1d_disease_mask], celiac_X])
        lbl_all = np.concatenate([t1d_mapped[t1d_disease_mask], celiac_mapped])

        pca_joint = PCA(n_components=2)
        X_joint   = pca_joint.fit_transform(X_all)

        color_map = {'T1D': '#e74c3c', 'Celiac': '#f39c12', 'Healthy': '#27ae60'}
        marker_map = {'T1D': 'o', 'Celiac': 's', 'Healthy': '^'}

        fig, ax = plt.subplots(figsize=(9, 7))
        for cond in ['Healthy', 'T1D', 'Celiac']:
            idx = lbl_all == cond
            if idx.sum() == 0: continue
            ax.scatter(X_joint[idx, 0], X_joint[idx, 1],
                       label=cond, c=color_map[cond], marker=marker_map[cond],
                       s=200, edgecolors='k', zorder=3, linewidths=1.0)
        ax.set_xlabel(f"PC1 ({pca_joint.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca_joint.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title("Cross-Disease scVI Embedding: T1D vs Celiac vs Healthy\n(Shared Latent Space — Patient Level)", fontsize=11)
        ax.legend(fontsize=12); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cross_disease_pca.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Joint cross-disease PCA saved!")
else:
    print(f"    T1D scVI file: {'Found' if t1d_scvi_path.exists() else 'MISSING — ' + str(t1d_scvi_path)}")
    print(f"    Celiac scVI file: {'Found' if celiac_scvi_path.exists() else 'MISSING'}")

# ─── 5. T1D Geneformer cross-disease ──────────────────────────────────────────
print("\n[5] Cross-disease Geneformer comparison...")
t1d_gf_path = T1D_RESULTS / "patient_geneformer_embeddings.csv"
celiac_gf_path = CELIAC_RESULTS / "geneformer_patient_embeddings.csv"

if t1d_gf_path.exists() and celiac_gf_path.exists():
    t1d_gf    = pd.read_csv(t1d_gf_path, index_col=0)
    celiac_gf = pd.read_csv(celiac_gf_path, index_col=0)

    # T1D format: Sample_ID(index), COND, 0,1,2...
    # Celiac format: patient_id(index), condition, scvi_0...
    t1d_cond_col    = 'COND' if 'COND' in t1d_gf.columns else 'condition'
    celiac_cond_col = 'condition' if 'condition' in celiac_gf.columns else 'COND'

    t1d_gf_feat    = [c for c in t1d_gf.columns if c != t1d_cond_col]
    celiac_gf_feat = [c for c in celiac_gf.columns if c != celiac_cond_col]

    print(f"    T1D GF: {t1d_gf.shape}, Celiac GF: {celiac_gf.shape}")
    print(f"    T1D feat dims: {len(t1d_gf_feat)}, Celiac feat dims: {len(celiac_gf_feat)}")

    if len(t1d_gf_feat) > 0 and len(celiac_gf_feat) > 0:
        # PCA each to 10 dims, then combine
        n_comp = min(5, min(len(t1d_gf)-1, len(celiac_gf)-1))
        pca_t  = PCA(n_components=n_comp).fit_transform(t1d_gf[t1d_gf_feat].values)
        pca_c  = PCA(n_components=n_comp).fit_transform(celiac_gf[celiac_gf_feat].values)

        t1d_labels    = t1d_gf[t1d_cond_col].values
        celiac_labels = celiac_gf[celiac_cond_col].values

        t1d_mapped    = np.array(['T1D' if l in ['D','T1D','Diabetes'] else 'Healthy' for l in t1d_labels])
        celiac_mapped = np.array(['Celiac' if l in ['C','Celiac'] else 'Healthy' for l in celiac_labels])

        t1d_mask = t1d_mapped == 'T1D'
        X_all    = np.vstack([pca_t[t1d_mask], pca_c])
        lbl_all  = np.concatenate([t1d_mapped[t1d_mask], celiac_mapped])

        pca_joint2 = PCA(n_components=2).fit_transform(X_all)
        color_map  = {'T1D': '#e74c3c', 'Celiac': '#f39c12', 'Healthy': '#27ae60'}
        marker_map = {'T1D': 'o', 'Celiac': 's', 'Healthy': '^'}

        fig, ax = plt.subplots(figsize=(9, 7))
        for cond in ['Healthy', 'T1D', 'Celiac']:
            idx = lbl_all == cond
            if idx.sum() == 0: continue
            ax.scatter(pca_joint2[idx, 0], pca_joint2[idx, 1],
                       label=cond, c=color_map[cond], marker=marker_map[cond],
                       s=200, edgecolors='k', zorder=3)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("Cross-Disease Geneformer Embeddings\nT1D vs Celiac vs Healthy (Patient Level)", fontsize=11)
        ax.legend(fontsize=12); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cross_disease_geneformer_pca.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Geneformer cross-disease PCA saved!")
else:
    print(f"    Skipping — T1D GF: {'OK' if t1d_gf_path.exists() else 'missing'}, Celiac GF: {'OK' if celiac_gf_path.exists() else 'not yet ready'}")

# ─── 6. Classification bar chart ─────────────────────────────────────────────
if all_results:
    results_df = pd.DataFrame(all_results)
    metrics = ["ROC-AUC", "Accuracy", "F1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        pivot = results_df.pivot(index="Classifier", columns="Feature Set", values=metric)
        pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor='k', linewidth=0.5)
        ax.set_title(f"Celiac — {metric} (LOO-CV)")
        ax.set_ylabel(metric); ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', rotation=30)
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}',
                        ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "celiac_classification_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n    Classification bar chart saved!")

print("\n" + "="*65)
print(" STEP 3 COMPLETE")
print("="*65)
print(f"  Results: {CELIAC_RESULTS}")
print(f"  Figures: {FIG_DIR}")
ls_figs = list(FIG_DIR.glob("*.png"))
for f in ls_figs:
    print(f"    {f.name}")
print("="*65)
