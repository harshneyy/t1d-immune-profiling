#!/usr/bin/env python3
"""Train scVI and export patient-level latent embeddings."""

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
    parser.add_argument("--output-h5ad", type=Path, default=cfg.SCVI_H5AD)
    parser.add_argument("--patient-output", type=Path, default=cfg.PATIENT_SCVI_EMBEDDINGS_CSV)
    parser.add_argument("--celltype-output", type=Path, default=cfg.CELLTYPE_SCVI_EMBEDDINGS_CSV)
    parser.add_argument("--n-latent", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--highly-variable-genes", type=int, default=3000)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="Optional random cell subset for CPU smoke tests. Use 0 for all cells.",
    )
    parser.add_argument("--no-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scvi

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.patient_output.parent.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)

    if args.max_cells and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(cfg.RANDOM_STATE)
        selected = rng.choice(adata.n_obs, size=args.max_cells, replace=False)
        adata = adata[selected].copy()

    if args.highly_variable_genes > 0 and adata.n_vars > args.highly_variable_genes:
        hvg_batch_key = None if args.max_cells else cfg.BATCH_COL if cfg.BATCH_COL in adata.obs else None
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=args.highly_variable_genes,
                flavor="seurat_v3",
                batch_key=hvg_batch_key,
                subset=True,
            )
        except ValueError as exc:
            print(f"seurat_v3 HVG failed ({exc}); falling back to cell_ranger HVG.")
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=args.highly_variable_genes,
                flavor="cell_ranger",
                batch_key=None,
                subset=True,
            )

    batch_key = cfg.BATCH_COL if cfg.BATCH_COL in adata.obs else None
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_latent=args.n_latent)
    model.train(max_epochs=args.max_epochs, accelerator="cpu" if args.no_gpu else "auto")

    latent = model.get_latent_representation()
    latent_cols = [f"scvi_{i:02d}" for i in range(latent.shape[1])]
    adata.obsm["X_scVI"] = latent

    latent_df = pd.DataFrame(latent, index=adata.obs_names, columns=latent_cols)
    latent_with_obs = pd.concat(
        [
            adata.obs[[cfg.SAMPLE_COL, cfg.LABEL_COL, cfg.CELL_TYPE_COL]].reset_index(drop=True),
            latent_df.reset_index(drop=True),
        ],
        axis=1,
    )

    patient_embeddings = (
        latent_with_obs.groupby([cfg.SAMPLE_COL, cfg.LABEL_COL], observed=True)[latent_cols]
        .mean()
    )
    patient_embeddings = patient_embeddings.reset_index()
    patient_embeddings.to_csv(args.patient_output, index=False)

    celltype_embeddings = (
        latent_with_obs.groupby([cfg.SAMPLE_COL, cfg.LABEL_COL, cfg.CELL_TYPE_COL], observed=True)[latent_cols]
        .mean()
        .reset_index()
    )
    wide_parts = []
    for cell_type, part in celltype_embeddings.groupby(cfg.CELL_TYPE_COL, observed=True):
        renamed = part[[cfg.SAMPLE_COL, cfg.LABEL_COL] + latent_cols].copy()
        renamed = renamed.rename(columns={col: f"{cell_type}__{col}" for col in latent_cols})
        wide_parts.append(renamed.set_index([cfg.SAMPLE_COL, cfg.LABEL_COL]))
    patient_celltype_embeddings = pd.concat(wide_parts, axis=1).reset_index()
    patient_celltype_embeddings = patient_celltype_embeddings.replace([np.inf, -np.inf], np.nan)
    patient_celltype_embeddings.to_csv(args.celltype_output, index=False)

    adata.write_h5ad(args.output_h5ad)
    print(f"Wrote {args.output_h5ad}")
    print(f"Wrote {args.patient_output}")
    print(f"Wrote {args.celltype_output}")


if __name__ == "__main__":
    main()
