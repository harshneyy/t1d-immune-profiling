#!/usr/bin/env python3
"""Summarize classification/subtype results and generate starter figures."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1d_foundation import config as cfg  # noqa: E402


METRIC_FILES = {
    "baseline_marker_celltype": cfg.RESULTS_DIR / "classification_metrics_baseline_immune_only.csv",
    "celltype_only": cfg.RESULTS_DIR / "classification_metrics_celltype_only_immune_only.csv",
    "scvi_patient": cfg.RESULTS_DIR / "classification_metrics_scvi.csv",
    "scvi_celltype": cfg.RESULTS_DIR / "classification_metrics_scvi_celltype.csv",
}


def main() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    for feature_set, path in METRIC_FILES.items():
        if path.exists():
            frame = pd.read_csv(path)
            frame["feature_set"] = feature_set
            metric_frames.append(frame)

    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)
        summary = (
            metrics.groupby(["feature_set", "model"])[
                ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
            ]
            .agg(["mean", "std"])
            .round(4)
        )
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary_path = cfg.RESULTS_DIR / "classification_metrics_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}")

        label_map = {
            "baseline_marker_celltype": "marker + celltype",
            "celltype_only": "celltype only",
            "scvi_patient": "scVI patient",
            "scvi_celltype": "scVI celltype",
        }
        metrics["feature_label"] = metrics["feature_set"].map(label_map).fillna(metrics["feature_set"])
        plt.figure(figsize=(11, 5))
        sns.barplot(data=metrics, x="feature_label", y="roc_auc", hue="model", errorbar="sd")
        plt.ylim(0.5, 1.0)
        plt.xticks(rotation=10, ha="right")
        plt.title("Patient-level T1D classification ROC-AUC")
        plt.xlabel("Feature set")
        plt.legend(title="model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        plt.tight_layout(rect=(0.02, 0.02, 0.82, 1))
        out = cfg.FIGURES_DIR / "classification_roc_auc.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Wrote {out}")

    if cfg.PATIENT_SCVI_EMBEDDINGS_CSV.exists():
        emb = pd.read_csv(cfg.PATIENT_SCVI_EMBEDDINGS_CSV)
        feature_cols = [col for col in emb.columns if col.startswith("scvi_")]
        X = emb[feature_cols]
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = StandardScaler().fit_transform(X)
        coords = PCA(n_components=2, random_state=cfg.RANDOM_STATE).fit_transform(X)
        emb["PC1"] = coords[:, 0]
        emb["PC2"] = coords[:, 1]

        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=emb, x="PC1", y="PC2", hue=cfg.LABEL_COL, s=70)
        plt.title("scVI patient embeddings by condition")
        plt.tight_layout()
        out = cfg.FIGURES_DIR / "scvi_patient_pca_condition.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Wrote {out}")

    subtype_path = cfg.RESULTS_DIR / "t1d_embedding_subtypes.csv"
    baseline_path = cfg.BASELINE_FEATURES_CSV
    if subtype_path.exists() and baseline_path.exists():
        subtypes = pd.read_csv(subtype_path)
        baseline = pd.read_csv(baseline_path)
        merged = subtypes[[cfg.SAMPLE_COL, "embedding_subtype"]].merge(
            baseline,
            on=cfg.SAMPLE_COL,
            how="left",
        )

        numeric = merged.select_dtypes(include="number")
        subtype_summary = numeric.groupby(merged["embedding_subtype"]).mean().round(4)
        subtype_summary.insert(0, "n_samples", merged.groupby("embedding_subtype").size())
        out_summary = cfg.RESULTS_DIR / "t1d_subtype_summary.csv"
        subtype_summary.to_csv(out_summary)
        print(f"Wrote {out_summary}")

        if cfg.PATIENT_SCVI_EMBEDDINGS_CSV.exists():
            emb = emb.merge(subtypes[[cfg.SAMPLE_COL, "embedding_subtype"]], on=cfg.SAMPLE_COL, how="left")
            t1d = emb[emb[cfg.LABEL_COL] == "T1D"].copy()
            plt.figure(figsize=(7, 5))
            sns.scatterplot(data=t1d, x="PC1", y="PC2", hue="embedding_subtype", s=75)
            plt.title("T1D scVI embedding subtypes")
            plt.tight_layout()
            out = cfg.FIGURES_DIR / "t1d_scvi_subtype_pca.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"Wrote {out}")

        heatmap_cols = [
            col
            for col in merged.columns
            if col.startswith("score__")
            or col in {
                "prop__CD4_T",
                "prop__CD8_T",
                "prop__C_Monocyte",
                "prop__B_Naive",
                "prop__T_reg",
                "prop__NK",
            }
        ]
        heatmap_cols = [col for col in heatmap_cols if col in merged.columns]
        if heatmap_cols:
            heatmap_data = merged.groupby("embedding_subtype")[heatmap_cols].mean()
            heatmap_data = heatmap_data.sub(heatmap_data.mean(axis=0), axis=1)
            heatmap_data = heatmap_data.div(heatmap_data.std(axis=0).replace(0, 1), axis=1)
            plt.figure(figsize=(max(8, len(heatmap_cols) * 0.35), 4))
            sns.heatmap(heatmap_data, cmap="vlag", center=0)
            plt.title("Subtype-level immune feature z-scores")
            plt.tight_layout()
            out = cfg.FIGURES_DIR / "t1d_subtype_feature_heatmap.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
