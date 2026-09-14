"""Data loading, structural validation and tidy reshaping (ILO1: integration).

The supplied workbook is an AGGREGATED heatmap, not raw responses:
- `Company Overall (107)` holds the company-wide MEAN per item;
- every department column (`Other`, `A`..`R`) holds a DELTA from that mean;
- 8 rows are category headers whose value is the mean of their items;
- the `manager` column is empty (documented data-supply gap).

Loading therefore validates structure explicitly and fails loudly on surprises,
rather than silently analysing a file we misunderstand.

Pipeline position:  raw .xlsx --load_heatmap--> wide df --tidy--> long df
                    --> quality.py (inspect + treat) --> phase1.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_heatmap(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the raw heatmap sheet exactly as supplied.

    How it works
    ------------
    1. Take the workbook path from config (`paths.raw_workbook`).
    2. If the file is absent, stop with a message telling the reader where to
       put it. 
    3. Read the sheet named in config (`results`) into a wide DataFrame.
    4. Hand it to `validate_raw_structure` BEFORE returning, so no caller can
       ever receive a workbook whose shape has not been checked.

    Returns
    -------
    Wide DataFrame, one row per category header or item, one column per
    department (plus the label and company-mean columns).
    """
    path = Path(cfg["paths"]["raw_workbook"])
    if not path.exists():
        raise FileNotFoundError(
            f"Survey workbook not found at {path}.\n"
            "The client's 2024 workbook is not distributed with this repository.\n"
            "  To run on demo data:  python scripts/make_demo_workbook.py   (or: make demo-data)\n"
            "  To run on real data:  place the supplied 'Employee Engagement Survey 2024' "
            f"workbook at {path} and set data.provenance: \"real\" in config/config.yaml."
        )
    df = pd.read_excel(path, sheet_name=cfg["paths"]["sheet_name"])
    validate_raw_structure(df, cfg)
    return df


def validate_raw_structure(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Assert the workbook matches the documented structure (fail loudly).

    How it works
    ------------
    1. Treat the first column as the label column (robust to it being renamed).
    2. Confirm the company-mean column named in config exists; if not, raise.
    3. Collect every label in the file into a set, then list which of the eight
       expected category headers are missing; if any are, raise naming them.

    Returns nothing: silence means the file is as documented. Assumptions are
    written as executable checks rather than comments, so a changed 2026 file
    fails here instead of producing quietly wrong analysis.
    """
    label_col = df.columns[0]
    company_col = cfg["data"]["company_col"]
    if company_col not in df.columns:
        raise ValueError(f"Expected company column '{company_col}' not found.")
    labels = set(df[label_col].astype(str))
    missing = [c for c in cfg["data"]["categories"] if c not in labels]
    if missing:
        raise ValueError(f"Category headers missing from workbook: {missing}")


def tidy(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Reshape the wide heatmap into tidy long format: one cell -> one row.

    How it works
    ------------
    1. Walk the rows top to bottom. When a row's label matches one of the eight
       category headers, remember it — every item row beneath then inherits that
       category, exactly as a human reads the spreadsheet.
    2. Tag each row's response scale: 10 for the two items listed in config,
       otherwise 5.
    3. For each department column in that row, write one record carrying every
       label it needs: category, item, scale, department, value, and whether the
       value is a company MEAN or a department DELTA.
    4. Skip empty cells (the `manager` column) with an explicit `continue` — a
       documented gap, not a silent loss.
    5. Assemble the records into a DataFrame (34 items x 20 units = 840 rows).

    Why tidy: one observation per row (department x item) makes joins,
    group-bys, plotting and testing straightforward — versus analysing an
    Excel-shaped table in place with fragile positional indexing.

    Returns
    -------
    DataFrame with columns:
      category, item, is_category_header, scale (5 or 10),
      department ('Company' for the overall column), value,
      value_type ('mean' for Company, 'delta' otherwise)
    """
    label_col = df.columns[0]
    company_col = cfg["data"]["company_col"]
    categories = cfg["data"]["categories"]
    ten_point = set(cfg["data"]["ten_point_items"])

    # Walk rows top-to-bottom assigning each item to its current category header.
    records: list[dict] = []
    current_cat: str | None = None
    dept_cols = [c for c in df.columns if c not in (label_col,)]
    for _, row in df.iterrows():
        label = str(row[label_col])
        is_header = label in categories
        if is_header:
            current_cat = label            # remember: item rows below inherit it
        scale = 10 if label in ten_point else 5
        for col in dept_cols:
            value = row[col]
            if pd.isna(value):
                continue  # empty manager column — documented gap, not dropped silently
            records.append(
                {
                    "category": current_cat,
                    "item": label,
                    "is_category_header": is_header,
                    "scale": scale,
                    "department": "Company" if col == company_col else str(col).strip(),
                    "value": float(value),
                    "value_type": "mean" if col == company_col else "delta",
                }
            )
    out = pd.DataFrame.from_records(records)
    return out


def export_processed(df: pd.DataFrame, cfg: dict[str, Any], name: str) -> Path:
    """Write a processed dataframe to data/processed/<name>.csv and return path.

    How it works: create the output directory if needed (`parents=True,
    exist_ok=True` so re-runs never fail), build the path from the given name,
    write with the index kept (it carries the department labels), return the
    path so the caller can report where the file went.
    """
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=True)
    return path
