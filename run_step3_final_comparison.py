"""
STEP 3 — Final Comparison: Baseline vs GPU scVI vs Geneformer (Local Version)
==============================================================================
Run AFTER both run_step1_scvi_gpu.py and run_step2_geneformer_gpu.py are complete.

Usage:
    /home/harshney/ml_env/bin/python run_step3_final_comparison.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load all classification results ─────────────────────────────────────────
print("=== Loading classification results ===")
frames = []

scvi_path = RESULTS_DIR / "classification_all_gpu.csv"
gf_path   = RESULTS_DIR / "classification_geneformer.csv"
bl_path   = RESULTS_DIR / "classification_metrics_summary.csv"

if scvi_path.exists():
    scvi_df = pd.read_csv(scvi_path)
    frames.append(scvi_df)
    print(f"✅ GPU scVI results loaded: {scvi_df.shape}")
else:
    print(f"⚠  Missing: {scvi_path} — run step 1 first")

if gf_path.exists():
    gf_df = pd.read_csv(gf_path)
    frames.append(gf_df)
    print(f"✅ Geneformer results loaded: {gf_df.shape}")
else:
    print(f"⚠  Missing: {gf_path} — run step 2 first")

if bl_path.exists():
    bl = pd.read_csv(bl_path)
    if "feature_set" not in bl.columns and "feature_label" in bl.columns:
        bl = bl.rename(columns={"feature_label": "feature_set"})
    frames.append(bl)
    print(f"✅ Baseline results loaded: {bl.shape}")
else:
    print(f"ℹ  Baseline file not found at {bl_path} — will compare available results only")

if not frames:
    print("❌ No results found — run steps 1 and 2 first!")
    raise SystemExit(1)

all_df = pd.concat(frames, ignore_index=True)
print(f"\nFeature sets: {all_df['feature_set'].unique().tolist()}")
print(f"Total rows:   {len(all_df)}")

# ─── Summary Table ────────────────────────────────────────────────────────────
print("\n=== Summary Table (sorted by ROC-AUC) ===")
summary = (all_df
           .groupby(["feature_set", "model"])[["accuracy", "balanced_accuracy", "f1", "roc_auc"]]
           .agg(["mean", "std"])
           .round(4))
summary.columns = [f"{m}_{s}" for m, s in summary.columns]
summary = summary.reset_index()
summary.to_csv(RESULTS_DIR / "final_comparison_table.csv", index=False)
print(summary[["feature_set", "model", "roc_auc_mean", "roc_auc_std"]]
      .sort_values("roc_auc_mean", ascending=False)
      .to_string(index=False))

# ─── Main ROC-AUC Comparison Bar Chart ───────────────────────────────────────
print("\n=== Generating Figures ===")
order = (all_df.groupby("feature_set")["roc_auc"].mean()
         .sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(data=all_df, x="feature_set", y="roc_auc", hue="model",
            order=order, errorbar="sd", palette="Set2", ax=ax)
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
ax.set_title("T1D Patient Classification — ROC-AUC Comparison\n(Baseline vs scVI GPU vs Geneformer)", fontsize=13)
ax.set_xlabel("Feature Set")
ax.set_ylabel("ROC-AUC (5-fold CV)")
ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "final_roc_auc_comparison.png", dpi=200)
plt.close()
print("✅ Saved: final_roc_auc_comparison.png")

# ─── Best-model-per-feature-set Horizontal Bar ───────────────────────────────
best_per_fs = all_df.groupby(["feature_set", "model"])["roc_auc"].mean().reset_index()
best_per_fs = best_per_fs.loc[best_per_fs.groupby("feature_set")["roc_auc"].idxmax()]
best_per_fs = best_per_fs.sort_values("roc_auc", ascending=True)

colors = []
for fs in best_per_fs["feature_set"]:
    if "geneformer" in fs:  colors.append("#4CAF50")
    elif "scvi" in fs:      colors.append("#2196F3")
    else:                   colors.append("#FF9800")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(best_per_fs["feature_set"], best_per_fs["roc_auc"], color=colors)
ax.axvline(0.5, color="gray", linestyle="--")
ax.set_xlabel("Best ROC-AUC (5-fold CV)")
ax.set_title("Best model per feature set")
for bar, (_, row) in zip(bars, best_per_fs.iterrows()):
    ax.text(bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{row['roc_auc']:.3f} ({row['model']})",
            va="center", fontsize=9)
ax.set_xlim(0.4, 1.05)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4CAF50", label="Geneformer"),
    Patch(facecolor="#2196F3", label="GPU scVI"),
    Patch(facecolor="#FF9800", label="Baseline"),
]
ax.legend(handles=legend_elements, loc="lower right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "best_model_comparison.png", dpi=200)
plt.close()
print("✅ Saved: best_model_comparison.png")

# ─── ROC-AUC by Condition Heatmap ────────────────────────────────────────────
pivot = (all_df.groupby(["feature_set", "model"])["roc_auc"]
         .mean()
         .unstack("model")
         .reindex(order))

fig, ax = plt.subplots(figsize=(10, max(4, len(order) * 0.6)))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
            vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax)
ax.set_title("ROC-AUC Heatmap — Feature Set × Model")
ax.set_xlabel("Model")
ax.set_ylabel("Feature Set")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_auc_heatmap.png", dpi=200)
plt.close()
print("✅ Saved: roc_auc_heatmap.png")

print("\n" + "="*60)
print("✅ STEP 3 COMPLETE — Final Comparison Done!")
print("="*60)
print(f"\nResults saved to: {RESULTS_DIR}/")
print("  final_comparison_table.csv")
print(f"\nFigures saved to: {FIGURES_DIR}/")
print("  final_roc_auc_comparison.png")
print("  best_model_comparison.png")
print("  roc_auc_heatmap.png")
