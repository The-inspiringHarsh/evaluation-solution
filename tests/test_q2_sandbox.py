"""Safe code execution: allowed analytics run, everything dangerous is blocked."""

import pandas as pd
import pytest

from src.question2_inventory.sandbox import UnsafeCodeError, run_code, validate_code

BLOCKED = {
    "import": "import os\nresult = 1",
    "from_import": "from os import system\nresult = 1",
    "dunder_attr": "result = df.__class__.__mro__",
    "dunder_string": "result = '__import__'",
    "open": "result = open('/etc/passwd').read()",
    "eval": "result = eval('1+1')",
    "exec": "exec('x=1')\nresult = 1",
    "getattr": "result = getattr(df, 'to_csv')",
    "file_write": "df.to_csv('out.csv')\nresult = 1",
    "pd_read": "result = pd.read_csv('/etc/passwd')",
    "np_load": "result = np.load('x.npy')",
    "module_alias": "m = pd\nresult = m",
    "query_string_eval": "result = df.query('hand_in_stock > 5')",
    "str_format": "result = '{0}'.format(df)",
    "class_def": "class A: pass\nresult = 1",
    "try": "try:\n    x = 1\nexcept Exception:\n    pass\nresult = 1",
    "globals": "result = globals()",
}


@pytest.mark.parametrize("name", sorted(BLOCKED))
def test_blocked_constructs(name, inventory):
    res = run_code(BLOCKED[name], inventory.df, timeout_s=5)
    assert not res.ok
    assert res.error.startswith("Blocked by safety check")


def test_must_assign_result():
    with pytest.raises(UnsafeCodeError, match="result"):
        validate_code("x = df.shape")


def test_allowed_analytics(inventory):
    res = run_code('result = df.nlargest(5, "hand_in_stock")[["product_name", "hand_in_stock"]]', inventory.df)
    assert res.ok and isinstance(res.result, pd.DataFrame)
    assert res.result.iloc[0]["product_name"] == "Smartphone"


def test_scalar_and_numpy(inventory):
    res = run_code("result = int(np.sum(df['hand_in_stock']))", inventory.df)
    assert res.ok and res.result == 2004


def test_timeout(inventory):
    res = run_code("x = 0\nwhile True:\n    x += 1\nresult = x", inventory.df, timeout_s=2)
    assert not res.ok
    assert "timed out" in res.error or "limit" in res.error


def test_memory_limit(inventory):
    res = run_code("result = np.arange(10**10)", inventory.df, timeout_s=10)
    assert not res.ok


def test_runtime_error_is_sanitized(inventory):
    res = run_code("result = df['no_such_column']", inventory.df)
    assert not res.ok and "KeyError" in res.error
    assert "/home" not in res.error and "Traceback" not in res.error


def test_original_dataframe_is_not_mutated(inventory):
    before = inventory.df.copy()
    res = run_code("df['hand_in_stock'] = 0\nresult = df['hand_in_stock'].sum()", inventory.df)
    assert res.ok and res.result == 0
    pd.testing.assert_frame_equal(before, inventory.df)


def test_output_rows_truncated(inventory):
    res = run_code("result = pd.concat([df] * 10)", inventory.df)
    assert res.ok and res.truncated and len(res.result) == 200


def test_chart_spec_validated(inventory):
    ok = run_code(
        'd = df.nlargest(10, "cost_price_total_usd")\nresult = d\n'
        'chart = {"type": "bar", "data": d, "x": "product_name", "y": "cost_price_total_usd", "title": "Top 10"}',
        inventory.df,
    )
    assert ok.ok and ok.chart is not None
    bad = run_code('result = 1\nchart = {"type": "bar", "data": df, "x": "nope", "y": "hand_in_stock"}', inventory.df)
    assert bad.ok and bad.chart is None and bad.warnings
