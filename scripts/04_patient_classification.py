#!/usr/bin/env python3
"""Run patient-level T1D classification experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1d_foundation import config as cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=cfg.PATIENT_SCVI_EMBEDDINGS_CSV,
        help="CSV with Sample_ID, COND, and numeric feature columns.",
    )
    parser.add_argument("--output", type=Path, default=cfg.RESULTS_DIR / "classification_metrics.csv")
    parser.add_argument(
        "--include-clinical",
        action="store_true",
        help="Include clinical/HLA metadata columns. Disabled by default to avoid leakage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.svm import SVC

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    if cfg.LABEL_COL not in df:
        raise ValueError(f"Missing label column: {cfg.LABEL_COL}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[cfg.LABEL_COL])
    leakage_or_clinical_cols = {
        "Gender",
        "Age_at_diagnosis",
        "Age_at_profiling",
        "Disease_Duration",
        "DQ_Risk_Haplotypes",
        "HLA_Haplotypes",
    }
    feature_cols = [col for col in df.columns if col not in {cfg.SAMPLE_COL, cfg.LABEL_COL}]
    if not args.include_clinical:
        feature_cols = [col for col in feature_cols if col not in leakage_or_clinical_cols]
    X = df[feature_cols].copy()

    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        converted = pd.to_numeric(X[col], errors="coerce")
        non_missing = X[col].notna().sum()
        converted_non_missing = converted.notna().sum()
        if non_missing == 0 or converted_non_missing / max(non_missing, 1) >= 0.8:
            X[col] = converted
            numeric_cols.append(col)
        else:
            X[col] = X[col].astype("string").fillna("missing")
            categorical_cols.append(col)

    scaled_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    models = {
        "logistic_regression": Pipeline(
            [
                ("preprocess", scaled_preprocessor),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("preprocess", scaled_preprocessor),
                ("model", SVC(kernel="rbf", class_weight="balanced", probability=True)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=cfg.RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", tree_preprocessor),
                ("model", HistGradientBoostingClassifier(random_state=cfg.RANDOM_STATE)),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    rows = []
    positive_label_index = list(label_encoder.classes_).index("T1D")

    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X.iloc[test_idx])[:, positive_label_index]
            else:
                score = model.decision_function(X.iloc[test_idx])
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "accuracy": accuracy_score(y[test_idx], pred),
                    "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
                    "f1": f1_score(y[test_idx], pred, pos_label=positive_label_index),
                    "roc_auc": roc_auc_score((y[test_idx] == positive_label_index).astype(int), score),
                    "n_features": X.shape[1],
                    "n_samples": X.shape[0],
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(metrics.groupby("model")[["accuracy", "balanced_accuracy", "f1", "roc_auc"]].mean())


if __name__ == "__main__":
    main()
