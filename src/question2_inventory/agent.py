"""Inventory conversational agent: plan -> (safe code | web search) -> plain-English summary."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..common.llm import LLMClient, LLMError, SearchAnswer, TextPart
from .loader import InventoryData, profile
from .sandbox import ExecutionResult, run_code
from .search import SearchError, SearchProvider, format_sources_markdown, safe_search

logger = logging.getLogger(__name__)

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["data", "search", "data_and_search", "clarify", "out_of_scope"]},
        "code": {"type": ["string", "null"]},
        "search_query": {"type": ["string", "null"]},
        "message": {"type": ["string", "null"]},
    },
    "required": ["intent", "code", "search_query", "message"],
    "additionalProperties": False,
}

PLANNER_SYSTEM = """You are the planning step of an inventory-analysis assistant.
The user asks questions about ONE pandas DataFrame named `df` loaded from an Excel workbook.

Columns (normalised name, original header, dtype):
{schema}

Workbook profile: {profile}
Known product names: {products}

Column meaning: hand_in_stock is the stock on hand AS STATED in the workbook (source data - never "fix" it).
The arithmetic expectation is opening_stock + stock_in - units_sold; rows where it differs are "stock-calculation
inconsistencies". cost_price_total_usd is the stated inventory value of each row.

Decide the intent:
- "data": answerable from the workbook -> write pandas code.
- "search": needs an external definition/business context only -> give a short generic search query.
- "data_and_search": needs both (e.g. a definition AND whether the workbook supports computing it).
- "clarify": genuinely ambiguous -> ask one short question in `message` (no code).
- "out_of_scope": unrelated or unsafe request (files, system, network, other datasets) -> explain in `message`.

Code rules (code is executed in a locked-down sandbox; violations are rejected):
- `df`, `pd` and `np` are pre-loaded. No imports, no file/network access, no dunder attributes,
  no eval/exec/open/getattr, no try/except, no str.format (use f-strings), no df.query/df.eval, no plotting calls.
- Assign the final answer to `result` (DataFrame, Series, number, string or dict). Keep only relevant columns.
- For charts ALSO assign `chart = {{"type": "bar"|"barh"|"line"|"pie"|"scatter", "data": <DataFrame>,
  "x": <column>, "y": <column>, "title": <str>}}`; `data` must contain the x and y columns.
- Match product names case-insensitively (e.g. `df["product_name"].str.lower() == "laptop"`) and use
  `.str.contains(..., case=False, regex=False)` for partial names. If nothing matches, `result` should be an
  empty DataFrame (the summary step will say the product was not found).
- Use the conversation history to resolve follow-ups ("those", "the second one", "what about tablets").

Search queries must be generic (concepts, definitions); never include personal data or identifiers.
`message` is a short note to the user for clarify/out_of_scope, otherwise null.
"""

SUMMARY_SYSTEM = """You write the final answer of an inventory assistant in concise, plain English.
Rules:
- Use ONLY numbers present in the execution result; never invent or recompute values.
- If the result is empty, say clearly that nothing matched (e.g. unknown product) and suggest a fix.
- Mention relevant data-quality caveats when the result includes stock-calculation inconsistencies.
- When web context is provided, structure the answer with two headings:
  "**From the workbook**" and "**External context (web)**". Cite web claims with [n] matching the numbered
  sources. Never present web claims as workbook facts.
- Without web context, do not add an external-context section.
- Keep it under 180 words. Use a short bullet list or a small markdown table when it helps.
"""


@dataclass
class AgentTurn:
    question: str
    intent: str = ""
    code: str | None = None
    execution: ExecutionResult | None = None
    search: SearchAnswer | None = None
    search_error: str | None = None
    answer: str = ""
    error: str | None = None
    repaired: bool = False

    @property
    def chart(self) -> dict[str, Any] | None:
        return self.execution.chart if self.execution and self.execution.ok else None


@dataclass
class InventoryAgent:
    data: InventoryData
    llm: LLMClient
    search_provider: SearchProvider | None
    timeout_s: float = 10.0
    max_output_chars: int = 20_000
    history: list[AgentTurn] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    # ------------------------------------------------------------------ prompts
    def _planner_system(self) -> str:
        prof = profile(self.data)
        products = ", ".join(sorted(self.data.df["product_name"].astype(str).unique())) if "product_name" in self.data.df else ""
        return PLANNER_SYSTEM.format(schema=self.data.schema_description(), profile=json.dumps(prof), products=products)

    def _history_text(self, limit: int = 5) -> str:
        if not self.history:
            return "(no previous turns)"
        lines = []
        for turn in self.history[-limit:]:
            lines.append(f"User: {turn.question}")
            if turn.code:
                lines.append(f"Code used:\n{turn.code}")
            if turn.answer:
                lines.append(f"Assistant: {turn.answer[:600]}")
        return "\n".join(lines)

    def _plan(self, question: str, feedback: str | None = None) -> dict[str, Any]:
        prompt = f"Conversation so far:\n{self._history_text()}\n\nNew question: {question}"
        if feedback:
            prompt += f"\n\nYour previous code failed. Fix it.\n{feedback}"
        return self.llm.complete_json(system=self._planner_system(), parts=[TextPart(prompt)], schema=PLAN_SCHEMA, max_tokens=4000)

    # ------------------------------------------------------------------ main entry
    def ask(self, question: str) -> AgentTurn:
        """Answer one user question; errors are captured on the returned turn, never raised."""
        turn = AgentTurn(question=question.strip())
        if not turn.question:
            turn.error = "Please type a question about the inventory workbook."
            turn.answer = turn.error
            return turn
        try:
            self._answer(turn)
        except LLMError as exc:
            turn.error = str(exc)
            turn.answer = f"Sorry, the language model call failed: {exc}"
        self.history.append(turn)
        return turn

    def _answer(self, turn: AgentTurn) -> None:
        plan = self._plan(turn.question)
        turn.intent = plan.get("intent", "")
        if turn.intent in {"clarify", "out_of_scope"}:
            turn.answer = plan.get("message") or "Could you rephrase the question about the inventory data?"
            return

        if turn.intent in {"data", "data_and_search"}:
            code = (plan.get("code") or "").strip()
            if not code:
                turn.error = "The planner did not produce code for a data question."
            else:
                turn.code = code
                turn.execution = run_code(code, self.data.df, self.timeout_s, self.max_output_chars)
                if not turn.execution.ok:
                    # One repair attempt with the sanitized error message.
                    retry = self._plan(turn.question, feedback=f"Code:\n{code}\nError: {turn.execution.error}")
                    new_code = (retry.get("code") or "").strip()
                    if new_code:
                        turn.code, turn.repaired = new_code, True
                        turn.execution = run_code(new_code, self.data.df, self.timeout_s, self.max_output_chars)

        if turn.intent in {"search", "data_and_search"}:
            query = (plan.get("search_query") or turn.question).strip()
            if self.search_provider is None:
                turn.search_error = "No search provider is configured."
            else:
                try:
                    turn.search = safe_search(self.search_provider, query)
                except SearchError as exc:
                    turn.search_error = str(exc)

        turn.answer = self._summarize(turn)

    # ------------------------------------------------------------------ summary
    def _summarize(self, turn: AgentTurn) -> str:
        parts = [f"Question: {turn.question}"]
        if turn.execution is not None:
            if turn.execution.ok:
                parts.append("Execution result (from the workbook):\n" + render_result_text(turn.execution.result))
                if turn.execution.warnings:
                    parts.append("Warnings: " + "; ".join(turn.execution.warnings))
            else:
                parts.append(f"Execution failed: {turn.execution.error}. Tell the user the query could not be computed.")
        elif turn.intent in {"data", "data_and_search"}:
            parts.append("No code could be produced. Say the question could not be answered from the workbook.")
        if turn.search is not None and turn.search.sources:
            numbered = "\n".join(f"[{i}] {s.title} - {s.url}\n    {s.snippet}" for i, s in enumerate(turn.search.sources[:6], start=1))
            parts.append(f"Web search answer (external):\n{turn.search.answer or '(none)'}\nSources:\n{numbered}")
        elif turn.search is not None:
            # Text without sources may be the model's own memory, so it is not passed on as web context.
            parts.append("Web search returned no sources. Say external context could not be verified; state no external facts.")
        elif turn.search_error:
            parts.append(f"Web search failed: {turn.search_error}. Say external context is unavailable.")
        summary = self.llm.complete_text(system=SUMMARY_SYSTEM, messages=[{"role": "user", "content": "\n\n".join(parts)}], max_tokens=2000)
        if turn.search is not None and turn.search.sources:
            summary += "\n\n**Sources (web):**\n" + format_sources_markdown(turn.search.sources)
        return summary


def render_result_text(result: Any, max_rows: int = 60) -> str:
    """Compact text rendering of a sandbox result for the summary prompt and transcripts."""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "(empty table - no rows matched)"
        return result.head(max_rows).to_markdown(index=False) + (f"\n... ({len(result)} rows total)" if len(result) > max_rows else "")
    if isinstance(result, pd.Series):
        if result.empty:
            return "(empty series - no rows matched)"
        return result.head(max_rows).to_frame().to_markdown()
    if isinstance(result, dict):
        return json.dumps({str(k): _jsonable(v) for k, v in result.items()}, indent=2, default=str)
    return str(result)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_dict()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    return value
