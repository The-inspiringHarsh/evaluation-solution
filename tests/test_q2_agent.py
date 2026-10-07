"""Agent orchestration with a mocked LLM: code path, repair, search path, clarification, empty results."""

from src.question2_inventory.agent import InventoryAgent, render_result_text
from src.question2_inventory.search import LLMNativeSearch


def _planner(responses):
    queue = list(responses)

    def handler(system, parts, schema):
        return queue.pop(0)

    return handler


def test_numeric_question_runs_code_and_summarises(inventory, fake_llm_factory):
    llm = fake_llm_factory(
        _planner([{"intent": "data", "code": "result = len(df)", "search_query": None, "message": None}]), text="There are 46 products."
    )
    agent = InventoryAgent(inventory, llm, None)
    turn = agent.ask("How many products are in the inventory?")
    assert turn.execution.ok and turn.execution.result == 46
    assert turn.answer == "There are 46 products."
    assert "46" in llm.text_calls[0]["messages"][0]["content"]
    assert len(agent.history) == 1


def test_failed_code_is_repaired_once(inventory, fake_llm_factory):
    llm = fake_llm_factory(
        _planner(
            [
                {"intent": "data", "code": "result = df['qty'].sum()", "search_query": None, "message": None},
                {"intent": "data", "code": "result = int(df['hand_in_stock'].sum())", "search_query": None, "message": None},
            ]
        )
    )
    turn = InventoryAgent(inventory, llm, None).ask("total stock?")
    assert turn.repaired and turn.execution.ok and turn.execution.result == 2004
    assert "KeyError" in llm.json_calls[1]["parts"][0].text


def test_unsafe_generated_code_is_blocked_not_run(inventory, fake_llm_factory):
    llm = fake_llm_factory(
        _planner(
            [
                {"intent": "data", "code": "import os\nresult = os.listdir('/')", "search_query": None, "message": None},
                {"intent": "data", "code": "import os\nresult = 1", "search_query": None, "message": None},
            ]
        )
    )
    turn = InventoryAgent(inventory, llm, None).ask("list files")
    assert not turn.execution.ok and "Blocked" in turn.execution.error


def test_search_question_uses_search_and_cites(inventory, fake_llm_factory):
    llm = fake_llm_factory(
        _planner(
            [
                {
                    "intent": "data_and_search",
                    "code": "result = {'has_cogs_column': False, 'columns': list(df.columns)}",
                    "search_query": "inventory turnover definition",
                    "message": None,
                }
            ]
        ),
        text="**From the workbook** ... **External context (web)** ... [1]",
    )
    agent = InventoryAgent(inventory, llm, LLMNativeSearch(llm))
    turn = agent.ask("What does inventory turnover mean and can we calculate it?")
    assert llm.search_calls == ["inventory turnover definition"]
    assert "https://www.investopedia.com" in turn.answer
    assert "Sources (web)" in turn.answer


def test_search_answer_without_sources_is_not_used_as_web_context(inventory, fake_llm_factory):
    from src.common.llm import SearchAnswer

    llm = fake_llm_factory(_planner([{"intent": "search", "code": None, "search_query": "inventory turnover", "message": None}]))
    llm.web_search = lambda query, max_uses=3: SearchAnswer(query=query, answer="Unsourced claim from memory.", sources=[], provider="fake")
    turn = InventoryAgent(inventory, llm, LLMNativeSearch(llm)).ask("What is inventory turnover?")
    prompt = llm.text_calls[-1]["messages"][0]["content"]
    assert "Unsourced claim" not in prompt and "could not be verified" in prompt
    assert "Sources (web)" not in turn.answer


def test_search_query_is_scrubbed_of_pii(inventory, fake_llm_factory):
    llm = fake_llm_factory(_planner([{"intent": "search", "code": None, "search_query": "PAN ABCDE1234F meaning", "message": None}]))
    InventoryAgent(inventory, llm, LLMNativeSearch(llm)).ask("what is this PAN")
    assert "ABCDE1234F" not in llm.search_calls[0]


def test_clarification_and_out_of_scope(inventory, fake_llm_factory):
    llm = fake_llm_factory(
        _planner(
            [
                {"intent": "clarify", "code": None, "search_query": None, "message": "Do you mean units or value?"},
                {"intent": "out_of_scope", "code": None, "search_query": None, "message": "I can only analyse the workbook."},
            ]
        )
    )
    agent = InventoryAgent(inventory, llm, None)
    assert agent.ask("Which is biggest?").answer == "Do you mean units or value?"
    assert "only analyse" in agent.ask("delete the server files").answer
    assert not llm.text_calls  # no summary call needed


def test_unknown_product_gives_empty_result(inventory, fake_llm_factory):
    llm = fake_llm_factory(
        _planner(
            [
                {
                    "intent": "data",
                    "code": "result = df[df['product_name'].str.lower() == 'hoverboard']",
                    "search_query": None,
                    "message": None,
                }
            ]
        )
    )
    turn = InventoryAgent(inventory, llm, None).ask("stock of hoverboards?")
    assert turn.execution.ok and turn.execution.result.empty
    assert "empty table" in llm.text_calls[0]["messages"][0]["content"]


def test_follow_up_sees_history(inventory, fake_llm_factory):
    llm = fake_llm_factory(
        _planner(
            [
                {"intent": "data", "code": "result = df.nlargest(5, 'hand_in_stock')", "search_query": None, "message": None},
                {"intent": "data", "code": "result = 1", "search_query": None, "message": None},
            ]
        )
    )
    agent = InventoryAgent(inventory, llm, None)
    agent.ask("top 5 by stock")
    agent.ask("what is the value of those?")
    assert "top 5 by stock" in llm.json_calls[1]["parts"][0].text


def test_empty_question(inventory, fake_llm_factory):
    turn = InventoryAgent(inventory, fake_llm_factory(None), None).ask("   ")
    assert turn.error


def test_render_result_text(inventory):
    assert "empty" in render_result_text(inventory.df.iloc[0:0])
    assert "Laptop" in render_result_text(inventory.df.head(3))
