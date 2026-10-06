"""Workbook loading: header-row detection, column normalisation and a data profile."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Canonical names for the columns we know; unknown columns get a generic snake_case name.
_CANONICAL = {
    "product id": "product_id",
    "product name": "product_name",
    "opening stock": "opening_stock",
    "purchase/stock in": "stock_in",
    "purchase / stock in": "stock_in",
    "number of units sold": "units_sold",
    "hand-in-stock": "hand_in_stock",
    "hand in stock": "hand_in_stock",
    "cost price per unit (usd)": "cost_price_per_unit_usd",
    "cost price total (usd)": "cost_price_total_usd",
}


class WorkbookError(ValueError):
    """Raised when the workbook cannot be interpreted as a table."""


def _clean_label(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"-\s+", "-", text)  # "Hand-In-\nStock" -> "Hand-In-Stock"
    text = re.sub(r"/\s+", "/", text)  # "Purchase/\nStock in" -> "Purchase/Stock in"
    return re.sub(r"\s+", " ", text).strip()


def normalize_column_name(raw: Any) -> str:
    """Map an awkward header (line breaks, slashes, units) to a stable snake_case name."""
    label = _clean_label(raw)
    canonical = _CANONICAL.get(label.lower())
    if canonical:
        return canonical
    snake = re.sub(r"\(([^)]*)\)", r" \1", label.lower())
    snake = re.sub(r"[^a-z0-9]+", "_", snake).strip("_")
    return snake or "column"


def detect_header_row(raw: pd.DataFrame, min_text_cells: int = 3, scan_rows: int = 30) -> int:
    """Return the 0-based index of the header row.

    The header is the first row with at least ``min_text_cells`` non-numeric text cells
    that is followed by a row containing numeric values (i.e. data starts below it).
    """
    limit = min(scan_rows, len(raw))
    for idx in range(limit):
        row = raw.iloc[idx]
        texts = [v for v in row if isinstance(v, str) and v.strip() and not _is_number(v)]
        if len(texts) < min_text_cells:
            continue
        if idx + 1 < len(raw):
            nxt = raw.iloc[idx + 1]
            numeric = sum(1 for v in nxt if _is_number(v))
            if numeric >= 2:
                return idx
    raise WorkbookError("Could not find a header row followed by numeric data in the first rows of the sheet.")


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)) and not pd.isna(value):
        return True
    if isinstance(value, str):
        try:
            float(value.replace(",", ""))
            return True
        except ValueError:
            return False
    return False


@dataclass
class InventoryData:
    """The cleaned table plus provenance metadata."""

    df: pd.DataFrame
    column_map: dict[str, str]  # normalised -> original label
    header_row_excel: int  # 1-based Excel row number of the header
    first_data_row_excel: int
    sheet_name: str
    source_path: str
    notes: list[str] = field(default_factory=list)

    def schema_description(self) -> str:
        """Compact description of columns for the LLM prompt (no data values)."""
        lines = []
        for col in self.df.columns:
            original = self.column_map.get(col, col)
            lines.append(f"- {col} (original: {original!r}, dtype: {self.df[col].dtype})")
        return "\n".join(lines)


def load_inventory(path: str | Path, sheet_name: str | int | None = 0) -> InventoryData:
    """Load the first (or named) sheet, detect the header row and normalise columns."""
    path = Path(path)
    if not path.exists():
        raise WorkbookError(f"Workbook not found: {path.name}")
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    resolved_sheet = sheet_name
    if isinstance(sheet_name, int):
        resolved_sheet = pd.ExcelFile(path, engine="openpyxl").sheet_names[sheet_name]
    raw = raw.dropna(axis=1, how="all")
    header_idx = detect_header_row(raw)
    header = raw.iloc[header_idx].tolist()
    body = raw.iloc[header_idx + 1 :].copy()
    body = body.dropna(how="all")

    names: list[str] = []
    column_map: dict[str, str] = {}
    for original in header:
        if original is None or (isinstance(original, float) and pd.isna(original)):
            original = "unnamed"
        name = normalize_column_name(original)
        base, n = name, 2
        while name in column_map:
            name = f"{base}_{n}"
            n += 1
        names.append(name)
        column_map[name] = _clean_label(original)
    body.columns = names

    for col in body.columns:
        converted = pd.to_numeric(body[col], errors="coerce")
        if converted.notna().sum() == body[col].notna().sum():
            body[col] = converted.astype("int64") if (converted.dropna() % 1 == 0).all() else converted
        else:
            body[col] = body[col].astype("string").str.strip()
    body = body.reset_index(drop=True)

    return InventoryData(
        df=body,
        column_map=column_map,
        header_row_excel=header_idx + 1,
        first_data_row_excel=header_idx + 2,
        sheet_name=str(resolved_sheet),
        source_path=path.name,
    )


def stock_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where opening_stock + stock_in - units_sold differs from the stated hand_in_stock.

    The stated value is source data and is never modified; the expected value is reported beside it.
    """
    required = {"opening_stock", "stock_in", "units_sold", "hand_in_stock"}
    missing = required - set(df.columns)
    if missing:
        raise WorkbookError(f"Consistency check needs columns: {sorted(missing)}")
    expected = df["opening_stock"] + df["stock_in"] - df["units_sold"]
    mask = expected != df["hand_in_stock"]
    out = df.loc[mask, ["product_id", "product_name", "opening_stock", "stock_in", "units_sold", "hand_in_stock"]].copy()
    out["expected_hand_in_stock"] = expected[mask]
    out["difference"] = out["hand_in_stock"] - out["expected_hand_in_stock"]
    return out.reset_index(drop=True)


def profile(data: InventoryData) -> dict[str, Any]:
    """Basic profile shown in the UI and used as context for the agent."""
    df = data.df
    info: dict[str, Any] = {
        "sheet": data.sheet_name,
        "header_row": data.header_row_excel,
        "first_data_row": data.first_data_row_excel,
        "records": int(len(df)),
        "columns": list(df.columns),
    }
    if "hand_in_stock" in df:
        info["total_hand_in_stock"] = int(df["hand_in_stock"].sum())
    if "cost_price_total_usd" in df:
        info["total_cost_price_usd"] = float(df["cost_price_total_usd"].sum())
    try:
        info["stock_inconsistencies"] = int(len(stock_consistency(df)))
    except WorkbookError:
        info["stock_inconsistencies"] = None
    return info
