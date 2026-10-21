#!/usr/bin/env python3
"""Split latex3 table manifests into one CSV per displayed table row."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="latex3/table_inputs")
    parser.add_argument("--output-dir", default="latex3/table_inputs/by_row")
    args = parser.parse_args()

    input_dir = REPO_ROOT / args.input_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_lines = [
        "# Per-row table inputs",
        "",
        "Each CSV lists the real source files used for one displayed row in latex3/main1.tex.",
        "Use scripts/build_latex3_result_tables.py to recompute mean, sample std, variance, and SE.",
        "",
    ]

    for manifest in sorted(input_dir.glob("*.csv")):
        grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        rows = read_rows(manifest)
        if not rows:
            continue
        for row in rows:
            grouped.setdefault(row["row_id"], []).append(row)
        readme_lines.append(f"## {manifest.stem}")
        for row_id, group_rows in grouped.items():
            display = group_rows[0].get("display", row_id)
            checkpoints = ", ".join(row.get("checkpoint", "") for row in group_rows)
            out = output_dir / manifest.stem / f"{row_id}.csv"
            write_rows(out, group_rows, list(group_rows[0].keys()))
            readme_lines.append(f"- `{manifest.stem}/{row_id}.csv`: {display}; checkpoints: {checkpoints}")
        readme_lines.append("")

    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"wrote per-row CSVs under {output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
