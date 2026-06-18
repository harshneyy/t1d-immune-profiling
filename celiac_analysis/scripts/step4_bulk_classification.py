#!/usr/bin/env python3
"""
GSE145358 Bulk RNA-seq — Celiac Disease ML Classification
36 samples: 6 Healthy + 15 GFD-Celiac (gluten-free) + 15 Active-Celiac (post gluten challenge)

Pipeline:
  1. Download counts from GEO
  2. Load metadata (condition labels)
  3. Normalize + log-transform
  4. Select top variance genes (HVGs proxy)
  5. ML classification with 5-fold StratifiedCV
     - Binary: Celiac vs Healthy
     - 3-class: Healthy vs GFD-Celiac vs Active-Celiac
  6. Feature importance (top discriminating genes)
  7. DEG analysis (Celiac vs Healthy, GFD vs Active)
  8. Cross-disease gene overlap with T1D
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import GEOparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.inspection import permutation_importance
from scipy import stats
import urllib.request, gzip, io

RESULTS = Path("/home/harshney/celiac-immune-profiling/results")
FIGS    = Path("/home/harshney/celiac-immune-profiling/figures")
DATA    = Path("/home/harshney/celiac-immune-profiling/data/raw")
RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

print("="*65)
print(" GSE145358 — Bulk RNA-seq Celiac Classification")
print("="*65)

# ─── 1. Download counts ──────────────────────────────────────────
COUNTS_FILE = DATA / "GSE145358_umi_counts.txt.gz"
if not COUNTS_FILE.exists():
    print("\n[1] Downloading UMI counts...")
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE145nnn/GSE145358/suppl/GSE145358_umi_counts.txt.gz"
    urllib.request.urlretrieve(url, COUNTS_FILE)
    print(f"    Downloaded → {COUNTS_FILE}")
else:
    print(f"\n[1] Counts already downloaded: {COUNTS_FILE}")

# ─── 2. Load counts ──────────────────────────────────────────────
print("\n[2] Loading counts matrix...")
counts_raw = pd.read_csv(COUNTS_FILE, sep='\t', index_col=0)
print(f"    Shape: {counts_raw.shape}  (genes × samples)")
print(f"    First columns: {list(counts_raw.columns[:6])}")

# The counts file has ENSEMBL + SYMBOL as first two columns, then sample IDs
if 'SYMBOL' in counts_raw.columns:
    gene_symbols = counts_raw['SYMBOL']
    counts = counts_raw.drop(columns=['SYMBOL'] if 'ENSEMBL' not in counts_raw.columns else ['SYMBOL'])
else:
    counts = counts_raw.copy()

# Sample columns are like '101-002BL', '101-029BL', etc.
sample_cols = [c for c in counts.columns if c not in ['ENSEMBL', 'SYMBOL']]
counts = counts[sample_cols]
print(f"    Count matrix: {counts.shape[0]} genes × {len(sample_cols)} samples")
print(f"    Sample IDs (paper): {sample_cols[:5]}...")

# ─── 3. Load metadata from GEO ──────────────────────────────────
print("\n[3] Loading metadata...")
gse = GEOparse.get_GEO(geo="GSE145358", destdir=str(DATA), silent=True)

meta = []
for k, v in gse.gsms.items():
    chars = {}
    for c in v.metadata.get('characteristics_ch1', []):
        if ': ' in c:
            key, val = c.split(': ', 1)
            chars[key.strip()] = val.strip()
    title   = v.metadata.get('title', [''])[0]   # e.g. '101-002BL'
    disease = chars.get('disease state', '')
    diet    = chars.get('diet', '')

    if 'healthy' in disease.lower():
        condition3 = 'Healthy'
        condition2 = 'Healthy'
    elif 'post gluten' in diet.lower():
        condition3 = 'Active-CeD'
        condition2 = 'Celiac'
    else:
        condition3 = 'GFD-CeD'
        condition2 = 'Celiac'

    meta.append({
        'gsm_id':     k,
        'paper_id':   title,    # This matches count matrix column names
        'disease':    disease,
        'diet':       diet,
        'condition2': condition2,
        'condition3': condition3,
    })

meta_df = pd.DataFrame(meta)
print(f"    GEO metadata: {meta_df.shape}")
print(f"    Sample paper IDs: {list(meta_df['paper_id'][:5])}")
print(f"    Count matrix cols: {sample_cols[:5]}")

# Align on paper_id (title matches count matrix column names)
meta_df = meta_df.set_index('paper_id')
common = [s for s in sample_cols if s in meta_df.index]
counts  = counts[common]
meta_df = meta_df.loc[common]
print(f"    Aligned: {len(common)}/{len(sample_cols)} samples matched")
print(f"    3-class distribution:\n{meta_df['condition3'].value_counts().to_string()}")

# ─── 4. Normalization ────────────────────────────────────────────
print("\n[4] Normalization (CPM + log1p)...")
# CPM normalization
cpm = counts.divide(counts.sum(axis=0), axis=1) * 1e6
log_cpm = np.log1p(cpm)
print(f"    log-CPM shape: {log_cpm.shape}")

# Select top 2000 most variable genes
gene_var = log_cpm.var(axis=1)
top_genes = gene_var.nlargest(2000).index
X_full = log_cpm.loc[top_genes].T  # samples × genes
print(f"    Top variable genes selected: {X_full.shape}")

# ─── 5. PCA visualization ────────────────────────────────────────
print("\n[5] PCA visualization...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X_full))

color_map3 = {'Healthy': '#27ae60', 'GFD-CeD': '#3498db', 'Active-CeD': '#e74c3c'}
marker_map3 = {'Healthy': '^', 'GFD-CeD': 's', 'Active-CeD': 'o'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (pc_x, pc_y, title) in zip(axes, [
    (0, 1, "PC1 vs PC2"), (0, 2, "PC1 vs PC3")
]):
    for cond, grp in meta_df.groupby('condition3'):
        idx = [list(meta_df.index).index(s) for s in grp.index]
        ax.scatter(X_pca[idx, pc_x], X_pca[idx, pc_y],
                   label=cond, c=color_map3.get(cond,'gray'),
                   marker=marker_map3.get(cond,'o'),
                   s=100, edgecolors='k', alpha=0.85, zorder=3)
    ax.set_xlabel(f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]:.1%})")
    ax.set_ylabel(f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]:.1%})")
    ax.set_title(f"GSE145358 Bulk RNA-seq PCA — {title}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "bulk_celiac_pca.png", dpi=150, bbox_inches='tight')
plt.close()
print("    PCA saved!")

# ─── 6. Binary classification: Celiac vs Healthy ─────────────────
print("\n[6] Binary classification: Celiac vs Healthy...")
y2 = (meta_df['condition2'] == 'Celiac').astype(int).values
X_np = X_full.values

CLASSIFIERS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM (RBF)':           SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_binary = []

for clf_name, clf in CLASSIFIERS.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    cv_res = cross_validate(pipe, X_np, y2, cv=cv5,
                             scoring=['roc_auc', 'accuracy', 'f1'],
                             return_train_score=False)
    r = {
        'Task': 'Binary (Celiac vs Healthy)',
        'Classifier': clf_name,
        'ROC-AUC': round(cv_res['test_roc_auc'].mean(), 4),
        'ROC-AUC std': round(cv_res['test_roc_auc'].std(), 4),
        'Accuracy': round(cv_res['test_accuracy'].mean(), 4),
        'F1': round(cv_res['test_f1'].mean(), 4),
    }
    results_binary.append(r)
    print(f"    {clf_name:25s}  AUC={r['ROC-AUC']:.3f}±{r['ROC-AUC std']:.3f}  Acc={r['Accuracy']:.3f}  F1={r['F1']:.3f}")

# ─── 7. 3-class classification: Healthy vs GFD vs Active ─────────
print("\n[7] 3-class classification: Healthy vs GFD-CeD vs Active-CeD...")
le = LabelEncoder()
y3 = le.fit_transform(meta_df['condition3'].values)
print(f"    Classes: {list(le.classes_)}")

results_3class = []
for clf_name, clf in CLASSIFIERS.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    cv_res = cross_validate(pipe, X_np, y3, cv=cv5,
                             scoring=['accuracy', 'f1_weighted'],
                             return_train_score=False)
    r = {
        'Task': '3-class (Healthy/GFD/Active)',
        'Classifier': clf_name,
        'ROC-AUC': '-',
        'ROC-AUC std': '-',
        'Accuracy': round(cv_res['test_accuracy'].mean(), 4),
        'F1 (weighted)': round(cv_res['test_f1_weighted'].mean(), 4),
    }
    results_3class.append(r)
    print(f"    {clf_name:25s}  Acc={r['Accuracy']:.3f}  F1={r['F1 (weighted)']:.3f}")

# ─── 8. Feature importance (Random Forest — binary) ───────────────
print("\n[8] Feature importance analysis...")
pipe_rf = Pipeline([('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(n_estimators=500, random_state=42))])
pipe_rf.fit(X_np, y2)
importances = pipe_rf.named_steps['clf'].feature_importances_
top_n = 20
top_idx = np.argsort(importances)[::-1][:top_n]
top_gene_names = X_full.columns[top_idx]
top_importances = importances[top_idx]

print(f"    Top 10 discriminating genes (Celiac vs Healthy):")
for g, imp in zip(top_gene_names[:10], top_importances[:10]):
    print(f"      {g}: {imp:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if i < 5 else '#3498db' for i in range(top_n)]
ax.barh(range(top_n), top_importances[::-1], color=colors[::-1], edgecolor='k', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_gene_names[::-1], fontsize=9)
ax.set_xlabel("Feature Importance (Random Forest)")
ax.set_title("Top 20 Genes Discriminating Celiac vs Healthy\n(Bulk RNA-seq, GSE145358)")
ax.axvline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig(FIGS / "bulk_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Feature importance plot saved!")

# ─── 9. Classification bar chart ─────────────────────────────────
print("\n[9] Plotting classification results...")
all_results = pd.DataFrame(results_binary + results_3class)
all_results.to_csv(RESULTS / "bulk_celiac_classification.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, task, metric in zip(axes,
    ['Binary (Celiac vs Healthy)', '3-class (Healthy/GFD/Active)'],
    ['ROC-AUC', 'Accuracy']):
    subset = all_results[all_results['Task'] == task]
    bars = ax.bar(range(len(subset)), subset[metric].astype(float),
                  color=['#e74c3c','#3498db','#f39c12','#27ae60'],
                  edgecolor='k', linewidth=0.7)
    ax.axhline(0.5 if metric == 'ROC-AUC' else 1/3, color='red',
               linestyle='--', alpha=0.6, label='Chance level')
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels(subset['Classifier'].values, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(metric); ax.set_ylim(0, 1.1)
    ax.set_title(f"{task}\n{metric} (5-fold CV)")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, subset[metric].astype(float)):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.suptitle("Celiac Disease — Bulk RNA-seq Classification (GSE145358, n=36)",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS / "bulk_classification_results.png", dpi=150, bbox_inches='tight')
plt.close()

# ─── 10. DEG: Active vs GFD ──────────────────────────────────────
print("\n[10] DEG analysis: Active-CeD vs GFD-CeD...")
active_idx = meta_df['condition3'] == 'Active-CeD'
gfd_idx    = meta_df['condition3'] == 'GFD-CeD'

X_active = log_cpm.loc[:, meta_df[active_idx].index]
X_gfd    = log_cpm.loc[:, meta_df[gfd_idx].index]

tstat, pval = stats.ttest_ind(X_active.T, X_gfd.T)
fc = X_active.mean(axis=1) - X_gfd.mean(axis=1)
deg_df = pd.DataFrame({'gene': log_cpm.index, 'log2FC': fc.values, 'pval': pval})
deg_df['padj'] = deg_df['pval'] * len(deg_df)  # Bonferroni
deg_df = deg_df.sort_values('pval')

sig_up   = deg_df[(deg_df['padj'] < 0.05) & (deg_df['log2FC'] > 1)]
sig_down = deg_df[(deg_df['padj'] < 0.05) & (deg_df['log2FC'] < -1)]
print(f"    Upregulated in Active-CeD (vs GFD): {len(sig_up)} genes")
print(f"    Downregulated in Active-CeD (vs GFD): {len(sig_down)} genes")
print(f"    Top 5 up: {list(sig_up['gene'][:5])}")
deg_df.to_csv(RESULTS / "deg_active_vs_gfd.csv", index=False)

# Volcano plot
fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(deg_df['log2FC'], -np.log10(deg_df['pval'] + 1e-10),
           c='lightgray', s=8, alpha=0.6, zorder=1)
ax.scatter(sig_up['log2FC'], -np.log10(sig_up['pval'] + 1e-10),
           c='#e74c3c', s=15, alpha=0.8, zorder=2, label=f'Up in Active ({len(sig_up)})')
ax.scatter(sig_down['log2FC'], -np.log10(sig_down['pval'] + 1e-10),
           c='#3498db', s=15, alpha=0.8, zorder=2, label=f'Down in Active ({len(sig_down)})')
for _, row in sig_up.head(8).iterrows():
    ax.annotate(row['gene'], (row['log2FC'], -np.log10(row['pval']+1e-10)),
                fontsize=7, ha='left')
ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
ax.axvline(-1, color='gray', linestyle='--', alpha=0.5)
ax.axhline(-np.log10(0.05/len(deg_df)), color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("log2 Fold Change (Active vs GFD)"); ax.set_ylabel("-log10(p-value)")
ax.set_title("Volcano Plot: Active-CeD vs GFD-CeD\n(Gluten Challenge Response Genes)")
ax.legend(fontsize=9); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIGS / "bulk_volcano_active_vs_gfd.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Volcano plot saved!")

# ─── 11. Summary ─────────────────────────────────────────────────
print("\n" + "="*65)
print(" BULK ANALYSIS COMPLETE")
print("="*65)
print("\nBinary Classification (Celiac vs Healthy, 5-fold CV):")
for r in results_binary:
    print(f"  {r['Classifier']:25s}  AUC={r['ROC-AUC']}±{r['ROC-AUC std']}")
print("\n3-Class Classification (Healthy/GFD/Active, 5-fold CV):")
for r in results_3class:
    print(f"  {r['Classifier']:25s}  Acc={r['Accuracy']}  F1={r['F1 (weighted)']}")
print("\nFigures saved:")
for f in sorted(FIGS.glob("bulk_*.png")):
    print(f"  {f.name}")
print("="*65)
