#!/usr/bin/env python3
"""Cluster T1D patient embeddings to discover immune subtypes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1d_foundation import config as cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=cfg.PATIENT_SCVI_EMBEDDINGS_CSV)
    parser.add_argument("--baseline", type=Path, default=cfg.BASELINE_FEATURES_CSV)
    parser.add_argument("--output", type=Path, default=cfg.RESULTS_DIR / "t1d_embedding_subtypes.csv")
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    t1d = df[df[cfg.LABEL_COL] == "T1D"].copy()
    feature_cols = [col for col in t1d.columns if col not in {cfg.SAMPLE_COL, cfg.LABEL_COL}]
    X = t1d[feature_cols].apply(pd.to_numeric, errors="coerce")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("cluster", KMeans(n_clusters=args.k, random_state=cfg.RANDOM_STATE, n_init=50)),
        ]
    )
    labels = pipeline.fit_predict(X)
    transformed = pipeline[:-1].transform(X)
    t1d["embedding_subtype"] = [f"subtype_{label + 1}" for label in labels]
    t1d["silhouette_k"] = silhouette_score(transformed, labels) if args.k > 1 else None

    if args.baseline.exists():
        baseline = pd.read_csv(args.baseline)
        keep_cols = [
            cfg.SAMPLE_COL,
            "Gender",
            "Age_at_diagnosis",
            "Age_at_profiling",
            "Disease_Duration",
            "DQ_Risk_Haplotypes",
        ]
        keep_cols = [col for col in keep_cols if col in baseline.columns]
        t1d = t1d.merge(baseline[keep_cols], on=cfg.SAMPLE_COL, how="left")

    t1d.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(t1d["embedding_subtype"].value_counts())


if __name__ == "__main__":
    main()

