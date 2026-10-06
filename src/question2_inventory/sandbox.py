"""Restricted execution of LLM-generated pandas code.

Defence in depth:

1. **AST allow/deny validation** before anything runs: no imports, no dunder access, no
   dangerous builtins, module attributes limited to an allowlist, file/network/IO methods blocked.
2. **Restricted globals**: only ``df`` (a copy), ``pd``, ``np`` and a small set of safe builtins.
3. **Separate process** with a wall-clock timeout, CPU and memory limits.
4. **Output limits**: tables capped to a row count and text capped to ``max_output_chars``.

The code must assign its answer to ``result`` and may assign an optional ``chart`` dict:
``{"type": "bar"|"barh"|"line"|"pie"|"scatter", "data": DataFrame, "x": str, "y": str, "title": str}``.
"""

from __future__ import annotations

import ast
import builtins
import io
import multiprocessing as mp
import re
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

MAX_RESULT_ROWS = 200

BLOCKED_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "help",
    "exit",
    "quit",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "type",
    "object",
    "memoryview",
    "bytearray",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "reload",
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "importlib",
    "builtins",
    "__builtins__",
}

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "None",
        "True",
        "False",
    )
    if hasattr(builtins, name)
}
SAFE_BUILTINS.update({"None": None, "True": True, "False": False})

# Attributes allowed directly on the ``pd`` and ``np`` modules.
PD_ALLOWED = {
    "DataFrame",
    "Series",
    "concat",
    "merge",
    "to_numeric",
    "to_datetime",
    "cut",
    "qcut",
    "isna",
    "notna",
    "isnull",
    "notnull",
    "NA",
    "NaT",
    "pivot_table",
    "crosstab",
    "Timestamp",
    "Timedelta",
    "date_range",
    "unique",
    "value_counts",
    "Categorical",
    "IndexSlice",
    "Index",
    "melt",
    "get_dummies",
}
NP_ALLOWED = {
    "abs",
    "all",
    "any",
    "arange",
    "argmax",
    "argmin",
    "array",
    "ceil",
    "clip",
    "cumsum",
    "diff",
    "floor",
    "inf",
    "isclose",
    "isfinite",
    "isnan",
    "linspace",
    "log",
    "log10",
    "max",
    "maximum",
    "mean",
    "median",
    "min",
    "minimum",
    "nan",
    "nanmean",
    "nanmedian",
    "nansum",
    "percentile",
    "quantile",
    "round",
    "select",
    "sign",
    "sqrt",
    "std",
    "sum",
    "unique",
    "var",
    "where",
    "int64",
    "float64",
    "corrcoef",
    "histogram",
    "sort",
    "argsort",
    "count_nonzero",
    "prod",
    "exp",
}

# Methods that read/write files, evaluate strings or reach the network.
BLOCKED_ATTRS = re.compile(
    r"^(to_(csv|excel|pickle|parquet|json|sql|hdf|feather|clipboard|html|latex|markdown|stata|xml|orc|gbq|string_io)"
    r"|read_\w*|load\w*|save\w*|tofile|fromfile|dump\w*|eval|query|format|format_map|system|popen"
    r"|savefig|style|plot|io|os|sys|ctypes|ctypeslib|lib|testing|f2py|distutils|DataSource|memmap)$"
)


class UnsafeCodeError(ValueError):
    """The generated code failed static validation."""


@dataclass
class ExecutionResult:
    ok: bool
    result: Any = None
    chart: dict[str, Any] | None = None
    stdout: str = ""
    error: str | None = None
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


class _Validator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self._module_attr_values: set[int] = set()

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.errors.append("imports are not allowed (pd, np and df are already available)")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.errors.append("imports are not allowed (pd, np and df are already available)")

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self.errors.append("global statements are not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        self.errors.append("nonlocal statements are not allowed")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.errors.append("class definitions are not allowed")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.errors.append("async code is not allowed")

    def visit_Await(self, node: ast.Await) -> None:  # noqa: N802
        self.errors.append("async code is not allowed")

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        self.errors.append("'with' blocks are not allowed")

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        self.errors.append("try/except is not allowed; errors are reported automatically")

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        attr = node.attr
        if attr.startswith("_"):
            self.errors.append(f"access to private/dunder attribute '{attr}' is not allowed")
        elif BLOCKED_ATTRS.match(attr):
            self.errors.append(f"attribute '{attr}' is blocked (file, network or string-evaluation access)")
        if isinstance(node.value, ast.Name) and node.value.id in {"pd", "np"}:
            allowed = PD_ALLOWED if node.value.id == "pd" else NP_ALLOWED
            if attr not in allowed:
                self.errors.append(f"'{node.value.id}.{attr}' is not in the allowed list")
            self._module_attr_values.add(id(node.value))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in BLOCKED_NAMES or node.id.startswith("__"):
            self.errors.append(f"name '{node.id}' is not allowed")
        if node.id in {"pd", "np"} and id(node) not in self._module_attr_values:
            # Using the module object itself (aliasing, passing it around) is not allowed.
            if isinstance(node.ctx, ast.Store):
                self.errors.append(f"reassigning '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str) and "__" in node.value:
            self.errors.append("string constants containing '__' are not allowed")


def _check_bare_module_use(tree: ast.AST) -> list[str]:
    """Reject ``pd``/``np`` used other than as ``pd.attr`` (prevents aliasing the module)."""
    errors = []
    attr_values = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {"pd", "np"} and id(node) not in attr_values:
            errors.append(f"'{node.id}' may only be used as '{node.id}.<function>'")
    return errors


def validate_code(code: str, max_len: int = 6000) -> ast.Module:
    """Parse and statically validate code; raise :class:`UnsafeCodeError` listing every problem."""
    if len(code) > max_len:
        raise UnsafeCodeError(f"code is too long ({len(code)} chars > {max_len})")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCodeError(f"syntax error on line {exc.lineno}: {exc.msg}") from exc
    validator = _Validator()
    validator.visit(tree)
    errors = validator.errors + _check_bare_module_use(tree)
    assigns_result = any(
        isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        and any(isinstance(t, ast.Name) and t.id == "result" for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))
        for n in ast.walk(tree)
    )
    if not assigns_result:
        errors.append("code must assign its answer to a variable named 'result'")
    if errors:
        unique = list(dict.fromkeys(errors))
        raise UnsafeCodeError("; ".join(unique))
    return tree


def _current_vm_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="ascii") as fh:  # noqa: PTH123 - host code, not sandboxed
            for line in fh:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:  # no /proc (macOS): fall back to a fixed estimate below
        return 4 * 1024**3
    return 4 * 1024**3


def _limit_resources(timeout_s: float) -> None:
    try:
        import resource

        cpu = max(1, int(timeout_s) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        # Allow 2 GiB of address space on top of what the interpreter already maps
        # (pyarrow-backed strings and thread stacks reserve virtual memory up front).
        mem = _current_vm_bytes() + 2 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except (ImportError, ValueError, OSError):
        # Non-POSIX platforms (Windows) rely on the wall-clock timeout only.
        pass


def _sanitize_error(exc: BaseException) -> str:
    """Return the exception type and message without file paths or tracebacks."""
    msg = str(exc)
    msg = re.sub(r"(/[\w.\-]+)+", "<path>", msg)
    msg = re.sub(r"([A-Za-z]:\\[\w.\\\-]+)", "<path>", msg)
    return f"{type(exc).__name__}: {msg[:500]}"


def _child(code: str, df: pd.DataFrame, timeout_s: float, queue: "mp.Queue[dict[str, Any]]") -> None:
    _limit_resources(timeout_s)
    env: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "df": df, "pd": pd, "np": np}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<generated>", "exec"), env)  # noqa: S102 - validated, restricted globals
        queue.put({"ok": True, "result": env.get("result"), "chart": env.get("chart"), "stdout": buf.getvalue()})
    except MemoryError:
        queue.put({"ok": False, "error": "MemoryError: the query used too much memory", "stdout": buf.getvalue()})
    except BaseException as exc:  # noqa: BLE001 - report every failure of untrusted code to the caller
        tb = traceback.extract_tb(exc.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<generated>"), None)
        where = f" (line {line})" if line else ""
        queue.put({"ok": False, "error": _sanitize_error(exc) + where, "stdout": buf.getvalue()})


def _limit_output(value: Any, max_chars: int) -> tuple[Any, bool]:
    truncated = False
    if isinstance(value, pd.DataFrame) and len(value) > MAX_RESULT_ROWS:
        value, truncated = value.head(MAX_RESULT_ROWS), True
    elif isinstance(value, pd.Series) and len(value) > MAX_RESULT_ROWS:
        value, truncated = value.head(MAX_RESULT_ROWS), True
    elif isinstance(value, (list, tuple)) and len(value) > MAX_RESULT_ROWS:
        value, truncated = list(value)[:MAX_RESULT_ROWS], True
    elif isinstance(value, str) and len(value) > max_chars:
        value, truncated = value[:max_chars], True
    elif not isinstance(value, (pd.DataFrame, pd.Series, int, float, str, bool, list, tuple, dict, type(None), np.generic)):
        value = repr(value)[:max_chars]
    return value, truncated


def run_code(code: str, df: pd.DataFrame, timeout_s: float = 10.0, max_output_chars: int = 20_000) -> ExecutionResult:
    """Validate then execute ``code`` in a child process. Never raises for user-code failures."""
    try:
        validate_code(code)
    except UnsafeCodeError as exc:
        return ExecutionResult(ok=False, error=f"Blocked by safety check: {exc}")

    # forkserver avoids forking a multi-threaded parent (Streamlit); spawn on Windows/macOS fallback.
    methods = mp.get_all_start_methods()
    ctx = mp.get_context("forkserver" if "forkserver" in methods else "spawn")
    queue: mp.Queue[dict[str, Any]] = ctx.Queue()
    proc = ctx.Process(target=_child, args=(code, df.copy(), timeout_s, queue), daemon=True)
    proc.start()
    try:
        payload = queue.get(timeout=timeout_s)
    except Exception:  # queue.Empty - timed out
        payload = None
    proc.join(1)
    if proc.is_alive():
        proc.kill()
        proc.join(1)
    if payload is None:
        if proc.exitcode not in (0, None) and proc.exitcode < 0:
            return ExecutionResult(ok=False, error="Execution stopped: CPU-time or memory limit exceeded.")
        return ExecutionResult(ok=False, error=f"Execution timed out after {timeout_s:.0f} seconds.")

    stdout = (payload.get("stdout") or "")[:max_output_chars]
    if not payload["ok"]:
        return ExecutionResult(ok=False, error=payload["error"], stdout=stdout)
    result, truncated = _limit_output(payload.get("result"), max_output_chars)
    chart = payload.get("chart")
    warnings: list[str] = []
    if chart is not None and not _valid_chart(chart):
        warnings.append("The chart specification was invalid and was ignored.")
        chart = None
    if truncated:
        warnings.append(f"Output truncated to {MAX_RESULT_ROWS} rows / {max_output_chars} characters.")
    return ExecutionResult(ok=True, result=result, chart=chart, stdout=stdout, truncated=truncated, warnings=warnings)


def _valid_chart(chart: Any) -> bool:
    if not isinstance(chart, dict):
        return False
    data = chart.get("data")
    if not isinstance(data, pd.DataFrame) or data.empty:
        return False
    if chart.get("type") not in {"bar", "barh", "line", "pie", "scatter"}:
        return False
    return chart.get("x") in data.columns and chart.get("y") in data.columns
