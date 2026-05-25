#!/usr/bin/env python3
"""Build patient-level metadata and cell-type proportion features without extra packages."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "seurat_metadata.csv"
OUT = ROOT / "results" / "patient_metadata_celltype_features.csv"


SAMPLE_COL = "Sample_ID"
LABEL_COL = "COND"
MERGED_CELL_TYPE_COL = "Cluster_Annotation_Merged"
DETAILED_CELL_TYPE_COL = "Cluster_Annotation_All"

SAMPLE_METADATA_COLS = [
    LABEL_COL,
    "Gender",
    "Age_at_diagnosis",
    "Age_at_profiling",
    "Disease_Duration",
    "DQ_Risk_Haplotypes",
    "HLA_Haplotypes",
]


def clean_value(key: str, value: str) -> str:
    value = "" if value is None else str(value).strip()
    if value == "NA":
        return ""
    if key == "Gender":
        return value.strip()
    return value


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    sample_meta = {}
    sample_counts = Counter()
    merged_counts = defaultdict(Counter)
    detailed_counts = defaultdict(Counter)
    merged_cell_types = set()
    detailed_cell_types = set()

    with METADATA.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = row[SAMPLE_COL]
            sample_counts[sample] += 1

            if sample not in sample_meta:
                sample_meta[sample] = {
                    col: clean_value(col, row.get(col, "")) for col in SAMPLE_METADATA_COLS
                }

            merged = row[MERGED_CELL_TYPE_COL]
            detailed = row[DETAILED_CELL_TYPE_COL]
            merged_counts[sample][merged] += 1
            detailed_counts[sample][detailed] += 1
            merged_cell_types.add(merged)
            detailed_cell_types.add(detailed)

    merged_cell_types = sorted(merged_cell_types)
    detailed_cell_types = sorted(detailed_cell_types)

    fieldnames = (
        [SAMPLE_COL]
        + SAMPLE_METADATA_COLS
        + ["n_cells"]
        + [f"prop_merged__{cell_type}" for cell_type in merged_cell_types]
        + [f"prop_detailed__{cell_type}" for cell_type in detailed_cell_types]
    )

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for sample in sorted(sample_meta):
            n_cells = sample_counts[sample]
            row = {SAMPLE_COL: sample, **sample_meta[sample], "n_cells": n_cells}

            for cell_type in merged_cell_types:
                row[f"prop_merged__{cell_type}"] = merged_counts[sample][cell_type] / n_cells

            for cell_type in detailed_cell_types:
                row[f"prop_detailed__{cell_type}"] = detailed_counts[sample][cell_type] / n_cells

            writer.writerow(row)

    print(f"Wrote {OUT}")
    print(f"Samples: {len(sample_meta)}")
    print(f"Merged cell-type features: {len(merged_cell_types)}")
    print(f"Detailed cell-type features: {len(detailed_cell_types)}")


if __name__ == "__main__":
    main()

