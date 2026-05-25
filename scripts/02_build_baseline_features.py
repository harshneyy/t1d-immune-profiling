#!/usr/bin/env python3
"""Build patient-level baseline features from AnnData."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1d_foundation import config as cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=cfg.RAW_H5AD)
    parser.add_argument("--output", type=Path, default=cfg.BASELINE_FEATURES_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd

    args.output.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.input)
    obs = adata.obs.copy()

    sample_col = cfg.SAMPLE_COL
    label_col = cfg.LABEL_COL
    cell_type_col = cfg.CELL_TYPE_COL

    celltype_counts = pd.crosstab(obs[sample_col], obs[cell_type_col])
    celltype_props = celltype_counts.div(celltype_counts.sum(axis=1), axis=0)
    celltype_props.columns = [f"prop__{col}" for col in celltype_props.columns]

    sample_meta_cols = [
        label_col,
        "Gender",
        "Age_at_diagnosis",
        "Age_at_profiling",
        "Disease_Duration",
        "DQ_Risk_Haplotypes",
    ]
    sample_meta = obs.groupby(sample_col, observed=True)[sample_meta_cols].first()

    feature_parts = [sample_meta, celltype_props]

    gene_index = {gene: idx for idx, gene in enumerate(adata.var_names)}

    for signature_name, genes in cfg.MARKER_SIGNATURES.items():
        available = [gene for gene in genes if gene in gene_index]
        if not available:
            continue
        idx = [gene_index[gene] for gene in available]
        signature_values = np.asarray(adata.X[:, idx].mean(axis=1)).ravel()
        obs[f"score__{signature_name}"] = signature_values
        sample_scores = obs.groupby(sample_col, observed=True)[f"score__{signature_name}"].mean()
        feature_parts.append(sample_scores.to_frame())

        for cell_type, cell_obs in obs.groupby(cell_type_col, observed=True):
            cell_indices = obs.index.get_indexer(cell_obs.index)
            cell_values = np.asarray(adata.X[cell_indices][:, idx].mean(axis=1)).ravel()
            per_cell = pd.Series(cell_values, index=cell_obs.index)
            per_sample = per_cell.groupby(cell_obs[sample_col], observed=True).mean()
            feature_parts.append(per_sample.rename(f"score__{signature_name}__{cell_type}").to_frame())

    features = pd.concat(feature_parts, axis=1)
    features.index.name = sample_col
    features.to_csv(args.output)
    print(f"Wrote {args.output}")
    print(f"Feature matrix shape: {features.shape[0]} samples x {features.shape[1]} columns")


if __name__ == "__main__":
    main()
