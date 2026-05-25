#!/usr/bin/env python3
"""Convert exported Seurat counts and metadata to AnnData."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1d_foundation import config as cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=cfg.METADATA_CSV)
    parser.add_argument("--counts", type=Path, default=cfg.COUNTS_MTX)
    parser.add_argument("--cells", type=Path, default=cfg.CELLS_TXT)
    parser.add_argument("--genes", type=Path, default=cfg.GENES_TXT)
    parser.add_argument("--output", type=Path, default=cfg.RAW_H5AD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import anndata as ad
    import pandas as pd
    import scipy.io

    args.output.parent.mkdir(parents=True, exist_ok=True)

    obs = pd.read_csv(args.metadata, index_col=0)
    obs.index = obs.index.astype(str)
    obs["Gender"] = obs["Gender"].astype(str).str.strip()

    cells = pd.read_csv(args.cells, header=None)[0].astype(str).tolist()
    genes = pd.read_csv(args.genes, header=None)[0].astype(str).tolist()

    if list(obs.index) != cells:
        raise ValueError("Metadata row names do not match cell barcode file order.")

    counts = scipy.io.mmread(args.counts).tocsr()
    if counts.shape != (len(genes), len(cells)):
        raise ValueError(
            f"Matrix shape {counts.shape} does not match genes/cells "
            f"({len(genes)}, {len(cells)})."
        )

    adata = ad.AnnData(X=counts.T.tocsr(), obs=obs, var=pd.DataFrame(index=genes))
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    adata.uns["source_files"] = {
        "metadata": str(args.metadata),
        "counts": str(args.counts),
        "cells": str(args.cells),
        "genes": str(args.genes),
    }

    adata.write_h5ad(args.output)
    print(f"Wrote {args.output}")
    print(f"AnnData shape: {adata.n_obs} cells x {adata.n_vars} genes")


if __name__ == "__main__":
    main()

