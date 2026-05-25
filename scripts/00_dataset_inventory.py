#!/usr/bin/env python3
"""Lightweight dataset inventory using only the Python standard library."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "seurat_metadata.csv"
CELLS = ROOT / "exported" / "T1D_Seurat_Object_Final_SCT_counts_cells.txt"
GENES = ROOT / "exported" / "T1D_Seurat_Object_Final_SCT_counts_genes.txt"
MTX = ROOT / "exported" / "T1D_Seurat_Object_Final_SCT_counts.mtx"
OUT = ROOT / "results" / "dataset_inventory.txt"


def read_lines(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.strip() for line in handle]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with METADATA.open(newline="") as handle:
        reader = csv.DictReader(handle)
        id_col = reader.fieldnames[0]
        rows = []
        for row in reader:
            row["cell_id"] = row[id_col]
            row["Gender"] = row.get("Gender", "").strip()
            rows.append(row)

    cells = read_lines(CELLS)
    genes = read_lines(GENES)

    with MTX.open() as handle:
        header = handle.readline().strip()
        dims = handle.readline().strip()

    sample_first = {}
    sample_cell_counts = Counter()
    celltype_by_condition = defaultdict(Counter)
    sample_celltype_counts = defaultdict(Counter)

    for row in rows:
        sample = row["Sample_ID"]
        sample_first.setdefault(sample, row)
        sample_cell_counts[sample] += 1
        celltype = row["Cluster_Annotation_Merged"]
        condition = row["COND"]
        celltype_by_condition[celltype][condition] += 1
        sample_celltype_counts[sample][celltype] += 1

    condition_cell_counts = Counter(row["COND"] for row in rows)
    condition_sample_counts = Counter(row["COND"] for row in sample_first.values())

    lines = []
    lines.append("T1D single-cell dataset inventory")
    lines.append("=" * 40)
    lines.append(f"Metadata rows/cells: {len(rows)}")
    metadata_columns = [col for col in rows[0] if col != "cell_id"] if rows else []
    lines.append(f"Metadata columns: {len(metadata_columns)}")
    lines.append(f"Cell file rows: {len(cells)}")
    lines.append(f"Gene file rows: {len(genes)}")
    lines.append(f"Unique genes: {len(set(genes))}")
    lines.append(f"Cell IDs match metadata order: {[row['cell_id'] for row in rows] == cells}")
    lines.append(f"Matrix header: {header}")
    lines.append(f"Matrix dimensions: {dims}")
    lines.append("")
    lines.append("Cell counts by condition:")
    for condition, count in condition_cell_counts.most_common():
        lines.append(f"- {condition}: {count}")
    lines.append("")
    lines.append("Sample counts by condition:")
    for condition, count in condition_sample_counts.most_common():
        lines.append(f"- {condition}: {count}")
    lines.append("")
    lines.append("Merged cell-type counts and percentages:")
    for celltype, counts in sorted(
        celltype_by_condition.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    ):
        healthy = counts.get("H", 0)
        t1d = counts.get("T1D", 0)
        healthy_pct = 100 * healthy / condition_cell_counts["H"]
        t1d_pct = 100 * t1d / condition_cell_counts["T1D"]
        lines.append(
            f"- {celltype}: H={healthy} ({healthy_pct:.2f}%), "
            f"T1D={t1d} ({t1d_pct:.2f}%), delta={t1d_pct - healthy_pct:.2f} pp"
        )
    lines.append("")
    lines.append("Low-cell samples under 500 cells:")
    for sample, count in sorted(sample_cell_counts.items(), key=lambda item: item[1]):
        if count < 500:
            condition = sample_first[sample]["COND"]
            lines.append(f"- {sample}: {condition}, {count} cells")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
