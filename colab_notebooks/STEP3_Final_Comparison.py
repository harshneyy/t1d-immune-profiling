"""
STEP 3 — Compare all models: Baseline vs scVI (GPU) vs Geneformer
(Run this AFTER Step 1 and Step 2 are both complete)

INSTRUCTIONS:
1. Upload these result CSVs to Drive/T1D_project/final_results/:
     - classification_all_gpu.csv         (from Step 1)
     - classification_geneformer.csv      (from Step 2)
     - results/patient_baseline_features.csv  (from your local machine)
2. Run all cells
3. Download final comparison figures
"""

# CELL 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# CELL 2 — Install
import subprocess
subprocess.run(["pip","-q","install","scikit-learn","pandas","matplotlib","seaborn"], check=True)

# CELL 3 — Paths
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

DRIVE_RESULTS = Path("/content/drive/MyDrive/T1D_project/final_results")
OUT_DIR       = Path("/content/final_comparison"); OUT_DIR.mkdir(exist_ok=True)

# Load all classification results
scvi_df    = pd.read_csv(DRIVE_RESULTS/"classification_all_gpu.csv")
gf_df      = pd.read_csv(DRIVE_RESULTS/"classification_geneformer.csv")

# Optional: baseline results from local machine (copy classification_metrics_summary.csv to Drive)
baseline_path = DRIVE_RESULTS/"classification_metrics_summary.csv"

frames = [scvi_df, gf_df]
if baseline_path.exists():
    bl = pd.read_csv(baseline_path)
    # Standardize column names if needed
    if "feature_set" not in bl.columns and "feature_label" in bl.columns:
        bl = bl.rename(columns={"feature_label":"feature_set"})
    frames.append(bl)

all_df = pd.concat(frames, ignore_index=True)
print("Feature sets:", all_df["feature_set"].unique())
print("Total rows:", len(all_df))

# CELL 4 — Summary table
summary = (all_df.groupby(["feature_set","model"])[["accuracy","balanced_accuracy","f1","roc_auc"]]
           .agg(["mean","std"]).round(4))
summary.columns = [f"{m}_{s}" for m,s in summary.columns]
summary = summary.reset_index()
summary.to_csv(OUT_DIR/"final_comparison_table.csv", index=False)
print(summary[["feature_set","model","roc_auc_mean","roc_auc_std"]].sort_values("roc_auc_mean",ascending=False).to_string())

# CELL 5 — Main comparison bar chart
order = (all_df.groupby("feature_set")["roc_auc"].mean()
         .sort_values(ascending=False).index.tolist())

plt.figure(figsize=(13,6))
sns.barplot(data=all_df, x="feature_set", y="roc_auc", hue="model",
            order=order, errorbar="sd", palette="Set2")
plt.ylim(0.5,1.0)
plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
plt.xticks(rotation=20, ha="right")
plt.title("T1D Patient Classification — ROC-AUC Comparison\n(Baseline vs scVI GPU vs Geneformer)")
plt.xlabel("Feature Set"); plt.ylabel("ROC-AUC (5-fold CV)")
plt.legend(title="Model", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR/"final_roc_auc_comparison.png", dpi=200); plt.close()
print("Saved final_roc_auc_comparison.png ✓")

# CELL 6 — Best-model-per-feature-set bar (cleaner view)
best_per_fs = all_df.groupby(["feature_set","model"])["roc_auc"].mean().reset_index()
best_per_fs = best_per_fs.loc[best_per_fs.groupby("feature_set")["roc_auc"].idxmax()]
best_per_fs = best_per_fs.sort_values("roc_auc", ascending=True)

colors = []
for fs in best_per_fs["feature_set"]:
    if "geneformer" in fs: colors.append("#4CAF50")
    elif "scvi" in fs:     colors.append("#2196F3")
    else:                  colors.append("#FF9800")

plt.figure(figsize=(9,5))
bars = plt.barh(best_per_fs["feature_set"], best_per_fs["roc_auc"], color=colors)
plt.axvline(0.5, color="gray", linestyle="--")
plt.xlabel("Best ROC-AUC (5-fold CV)")
plt.title("Best model per feature set")
for bar, (_, row) in zip(bars, best_per_fs.iterrows()):
    plt.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
             f"{row['roc_auc']:.3f} ({row['model']})", va="center", fontsize=8)
plt.xlim(0.4, 1.02)
plt.tight_layout()
plt.savefig(OUT_DIR/"best_model_comparison.png", dpi=200); plt.close()
print("Saved best_model_comparison.png ✓")

# CELL 7 — Download
import shutil
shutil.make_archive("/content/final_comparison","zip",str(OUT_DIR))
from google.colab import files
files.download("/content/final_comparison.zip")

"""
SHARE WITH ME AFTER RUNNING:
  final_comparison_table.csv
  final_roc_auc_comparison.png
  best_model_comparison.png
  Printed summary table
"""
