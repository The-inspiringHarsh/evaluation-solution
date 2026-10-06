"""Workbook loading: header detection, normalisation and smoke-test totals."""

import pandas as pd
import pytest

from src.question2_inventory.loader import WorkbookError, detect_header_row, normalize_column_name, profile, stock_consistency


def test_header_row_detected_on_row_6(inventory):
    assert inventory.header_row_excel == 6
    assert inventory.first_data_row_excel == 7


def test_column_normalisation_keeps_original_mapping(inventory):
    assert list(inventory.df.columns) == [
        "product_id",
        "product_name",
        "opening_stock",
        "stock_in",
        "units_sold",
        "hand_in_stock",
        "cost_price_per_unit_usd",
        "cost_price_total_usd",
    ]
    assert inventory.column_map["hand_in_stock"] == "Hand-In-Stock"
    assert inventory.column_map["stock_in"] == "Purchase/Stock in"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Opening \nStock", "opening_stock"),
        ("Hand-In-\nStock", "hand_in_stock"),
        ("Cost Price\nTotal (USD)", "cost_price_total_usd"),
        ("Weird  Header (kg)", "weird_header_kg"),
    ],
)
def test_normalize_column_name(raw, expected):
    assert normalize_column_name(raw) == expected


def test_workbook_smoke_totals(inventory):
    df = inventory.df
    assert len(df) == 46
    assert int(df["hand_in_stock"].sum()) == 2004
    assert int(df["cost_price_total_usd"].sum()) == 359760
    top = df.nlargest(1, "hand_in_stock").iloc[0]
    assert (top["product_name"], int(top["hand_in_stock"])) == ("Smartphone", 80)
    by_name = df.groupby("product_name")["cost_price_total_usd"].sum()
    assert by_name["Laptop"] == by_name["Smartphone"] == 72000


def test_stock_inconsistencies_reported_not_fixed(inventory):
    before = inventory.df["hand_in_stock"].copy()
    inc = stock_consistency(inventory.df)
    assert len(inc) == 12
    assert (inc["hand_in_stock"] != inc["expected_hand_in_stock"]).all()
    pd.testing.assert_series_equal(before, inventory.df["hand_in_stock"])


def test_profile(inventory):
    prof = profile(inventory)
    assert prof["records"] == 46 and prof["stock_inconsistencies"] == 12


def test_header_detection_fails_cleanly_without_table():
    raw = pd.DataFrame([["just a title", None, None], [None, None, None]])
    with pytest.raises(WorkbookError):
        detect_header_row(raw)
