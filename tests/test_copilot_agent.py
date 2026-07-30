"""Unit tests for src/copilot_agent.py -- the real Claude tool-use path for
the Copilot's free-text ask box (Option B, alongside battery_copilot.py's
unchanged template system, Option A).

No real Anthropic API calls: `import anthropic as _ant` inside
answer_with_tools() is a lazy import, so tests substitute a fake module in
sys.modules before calling it, matching this project's fully-offline test
convention. build_tools()'s dispatch wrappers are tested by monkeypatching
the underlying battery_copilot.py/knowledge_graph.py/battery_knowledge.py/
copilot_retrieval.py functions they call through to -- those functions
already have their own coverage elsewhere; this file tests the wrapping
(argument shape, unknown-cell_id/no-graph guards, JSON-serializability),
not the domain logic itself.
"""

import sys
import json

import battery_copilot
import copilot_agent
from copilot_agent import build_tools, answer_with_tools, TOOL_DEFS


# ---------------------------------------------------------------------------
# build_tools() -- dispatch wrapper behavior
# ---------------------------------------------------------------------------

def test_all_five_tools_have_a_dispatch_entry():
    _, dispatch = build_tools({}, {}, org_id=1, graph=None)
    tool_names = {t["name"] for t in TOOL_DEFS}
    assert tool_names == set(dispatch.keys())
    assert len(tool_names) == 5


def test_get_cell_context_unknown_cell_returns_error_not_exception():
    _, dispatch = build_tools({"B0005": object()}, {}, org_id=1, graph=None)
    result = dispatch["get_cell_context"]({"cell_id": "NOT_A_CELL"})
    assert "error" in result
    assert "NOT_A_CELL" in result["error"]


def test_get_cell_context_known_cell_calls_through_to_build_cell_context(monkeypatch):
    calls = []

    def _fake_build_cell_context(cell_id, featured_dfs, bundles):
        calls.append((cell_id, featured_dfs, bundles))
        return {"cell_id": cell_id, "soh": 91.2}

    monkeypatch.setattr(battery_copilot, "build_cell_context", _fake_build_cell_context)
    _, dispatch = build_tools({"B0005": "df"}, {"nasa": "bundle"}, org_id=1, graph=None)
    result = dispatch["get_cell_context"]({"cell_id": "B0005"})
    assert result == {"cell_id": "B0005", "soh": 91.2}
    assert calls == [("B0005", {"B0005": "df"}, {"nasa": "bundle"})]


def test_get_fleet_stats_calls_through_with_org_id(monkeypatch):
    calls = []

    def _fake_build_fleet_stats(org_id, featured_dfs, bundles):
        calls.append(org_id)
        return {"n_cells": 3}

    monkeypatch.setattr(battery_copilot, "build_fleet_stats", _fake_build_fleet_stats)
    _, dispatch = build_tools({}, {}, org_id=42, graph=None)
    result = dispatch["get_fleet_stats"]({})
    assert result == {"n_cells": 3}
    assert calls == [42]


def test_get_cell_mechanism_unknown_cell_returns_error():
    _, dispatch = build_tools({"B0005": "df"}, {}, org_id=1, graph=object())
    result = dispatch["get_cell_mechanism"]({"cell_id": "NOT_A_CELL"})
    assert "error" in result


def test_get_cell_mechanism_no_graph_returns_error_not_exception():
    _, dispatch = build_tools({"B0005": "df"}, {}, org_id=1, graph=None)
    result = dispatch["get_cell_mechanism"]({"cell_id": "B0005"})
    assert "error" in result
    assert "graph" in result["error"].lower()


def test_get_cell_mechanism_calls_through_and_maps_verdict_to_literature(monkeypatch):
    import knowledge_graph

    def _fake_get_or_compute_mechanism(g, cell_id, df, **kwargs):
        return {"verdict": "LAM — Loss of Active Material", "confidence_label": "High"}

    def _fake_literature_for_mechanism(g, key):
        assert key == "lam"
        return [{"id": "doc1", "text": "..."}]

    monkeypatch.setattr(knowledge_graph, "get_or_compute_mechanism", _fake_get_or_compute_mechanism)
    monkeypatch.setattr(knowledge_graph, "literature_for_mechanism", _fake_literature_for_mechanism)
    _, dispatch = build_tools({"B0005": "df"}, {}, org_id=1, graph=object())
    result = dispatch["get_cell_mechanism"]({"cell_id": "B0005"})
    assert result["mechanism"]["verdict"] == "LAM — Loss of Active Material"
    assert result["literature"] == [{"id": "doc1", "text": "..."}]


def test_get_feature_citation_missing_returns_error(monkeypatch):
    import battery_knowledge
    monkeypatch.setattr(battery_knowledge, "get_feature_citation", lambda name: None)
    _, dispatch = build_tools({}, {}, org_id=1, graph=None)
    result = dispatch["get_feature_citation"]({"feature_name": "no_such_feature"})
    assert "error" in result


def test_get_feature_citation_found_passes_through(monkeypatch):
    import battery_knowledge
    citation = {"title": "Some Paper", "doi": "10.1234/x"}
    monkeypatch.setattr(battery_knowledge, "get_feature_citation", lambda name: citation)
    _, dispatch = build_tools({}, {}, org_id=1, graph=None)
    result = dispatch["get_feature_citation"]({"feature_name": "fade_rate_30cy"})
    assert result == citation


def test_search_battery_knowledge_empty_results_has_note(monkeypatch):
    import copilot_retrieval
    monkeypatch.setattr(copilot_retrieval, "retrieve", lambda query, top_k=3: [])
    _, dispatch = build_tools({}, {}, org_id=1, graph=None)
    result = dispatch["search_battery_knowledge"]({"query": "unrelated nonsense"})
    assert result["results"] == []
    assert "note" in result


def test_search_battery_knowledge_passes_through_results(monkeypatch):
    import copilot_retrieval
    monkeypatch.setattr(copilot_retrieval, "retrieve", lambda query, top_k=3: ["doc text 1"])
    _, dispatch = build_tools({}, {}, org_id=1, graph=None)
    result = dispatch["search_battery_knowledge"]({"query": "thermal runaway"})
    assert result == {"results": ["doc text 1"]}


def test_every_dispatch_result_is_json_serializable(monkeypatch):
    """The loop wraps every tool result in json.dumps() -- a dict containing
    a non-serializable object would blow up at call time, not at test time,
    so this is worth asserting directly for every tool's happy path."""
    monkeypatch.setattr(battery_copilot, "build_cell_context", lambda *a, **k: {"soh": 90.0})
    monkeypatch.setattr(battery_copilot, "build_fleet_stats", lambda *a, **k: {"n_cells": 1})
    _, dispatch = build_tools({"B0005": "df"}, {}, org_id=1, graph=None)
    for name in ("get_cell_context", "get_fleet_stats", "get_feature_citation", "search_battery_knowledge"):
        result = dispatch[name]({"cell_id": "B0005", "feature_name": "x", "query": "x"})
        json.dumps(result, default=str)  # must not raise


# ---------------------------------------------------------------------------
# answer_with_tools() -- manual loop control flow, Anthropic client mocked
# ---------------------------------------------------------------------------

class _FakeTextBlock:
    type = "text"
    def __init__(self, text):
        self.text = text


class _FakeToolUseBlock:
    type = "tool_use"
    def __init__(self, name, tool_input, block_id="tu_1"):
        self.name = name
        self.input = tool_input
        self.id = block_id


class _FakeResponse:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses, api_key=None):
        self.messages = _FakeMessages(responses)


def _install_fake_anthropic(monkeypatch, responses):
    """Installs a fake `anthropic` module in sys.modules so answer_with_tools()'s
    lazy `import anthropic as _ant` picks it up, and returns the fake client's
    .messages.create() call log for assertions."""
    fake_client_holder = {}

    class _FakeAnthropicModule:
        @staticmethod
        def Anthropic(api_key=None):
            client = _FakeClient(responses, api_key=api_key)
            fake_client_holder["client"] = client
            return client

    monkeypatch.setitem(sys.modules, "anthropic", _FakeAnthropicModule())
    return fake_client_holder


def _no_tool_dispatch_env(monkeypatch):
    """A minimal featured_dfs/bundles/graph that build_tools() can wrap
    without hitting real domain logic (no tool call actually resolves
    real data in these tests -- the fake tool_input names are fabricated)."""
    monkeypatch.setattr(battery_copilot, "build_cell_context", lambda *a, **k: {"soh": 91.0})
    monkeypatch.setattr(battery_copilot, "build_fleet_stats", lambda *a, **k: {"n_cells": 4})
    return {"B0005": "df"}, {"nasa": "bundle"}


def test_no_api_key_returns_none_immediately():
    answer, tools_used = answer_with_tools("query", "B0005", 1, {}, {}, None, api_key="")
    assert answer is None
    assert tools_used == []


def test_single_tool_call_then_end_turn(monkeypatch):
    featured_dfs, bundles = _no_tool_dispatch_env(monkeypatch)
    responses = [
        _FakeResponse("tool_use", [_FakeToolUseBlock("get_cell_context", {"cell_id": "B0005"})]),
        _FakeResponse("end_turn", [_FakeTextBlock("B0005 is at 91.0% SOH.")]),
    ]
    holder = _install_fake_anthropic(monkeypatch, responses)
    answer, tools_used = answer_with_tools(
        "How is B0005 doing?", "B0005", 1, featured_dfs, bundles, None, api_key="sk-ant-fake",
    )
    assert answer == "B0005 is at 91.0% SOH."
    assert tools_used == ["get_cell_context"]
    assert len(holder["client"].messages.calls) == 2


def test_multi_tool_chaining_across_two_iterations(monkeypatch):
    featured_dfs, bundles = _no_tool_dispatch_env(monkeypatch)
    responses = [
        _FakeResponse("tool_use", [_FakeToolUseBlock("get_fleet_stats", {}, block_id="tu_1")]),
        _FakeResponse("tool_use", [_FakeToolUseBlock("get_cell_context", {"cell_id": "B0005"}, block_id="tu_2")]),
        _FakeResponse("end_turn", [_FakeTextBlock("The worst cell is B0005 at 91.0%.")]),
    ]
    _install_fake_anthropic(monkeypatch, responses)
    answer, tools_used = answer_with_tools(
        "Which cell is worst and what's its SOH?", None, 1, featured_dfs, bundles, None, api_key="sk-ant-fake",
    )
    assert answer == "The worst cell is B0005 at 91.0%."
    assert tools_used == ["get_fleet_stats", "get_cell_context"]


def test_iteration_cap_returns_budget_message_not_infinite_loop(monkeypatch):
    featured_dfs, bundles = _no_tool_dispatch_env(monkeypatch)
    # Always returns tool_use -- never lets the loop reach end_turn.
    responses = [
        _FakeResponse("tool_use", [_FakeToolUseBlock("get_fleet_stats", {})])
        for _ in range(copilot_agent.MAX_TOOL_ITERATIONS)
    ]
    holder = _install_fake_anthropic(monkeypatch, responses)
    answer, tools_used = answer_with_tools(
        "keep going forever", None, 1, featured_dfs, bundles, None, api_key="sk-ant-fake",
    )
    assert "tool-call budget" in answer
    assert len(tools_used) == copilot_agent.MAX_TOOL_ITERATIONS
    assert len(holder["client"].messages.calls) == copilot_agent.MAX_TOOL_ITERATIONS


def test_refusal_stop_reason_returns_plain_message_not_content(monkeypatch):
    featured_dfs, bundles = _no_tool_dispatch_env(monkeypatch)
    responses = [_FakeResponse("refusal", [])]
    _install_fake_anthropic(monkeypatch, responses)
    answer, tools_used = answer_with_tools(
        "anything", None, 1, featured_dfs, bundles, None, api_key="sk-ant-fake",
    )
    assert answer == "I wasn't able to answer that question."
    assert tools_used == []


def test_exception_during_call_falls_back_to_none(monkeypatch):
    class _RaisingAnthropicModule:
        @staticmethod
        def Anthropic(api_key=None):
            raise RuntimeError("network error")

    monkeypatch.setitem(sys.modules, "anthropic", _RaisingAnthropicModule())
    answer, tools_used = answer_with_tools(
        "anything", None, 1, {}, {}, None, api_key="sk-ant-fake",
    )
    assert answer is None
    assert tools_used == []


def test_tool_execution_exception_reported_as_is_error_not_raised(monkeypatch):
    featured_dfs, bundles = _no_tool_dispatch_env(monkeypatch)

    def _raise(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(battery_copilot, "build_cell_context", _raise)
    responses = [
        _FakeResponse("tool_use", [_FakeToolUseBlock("get_cell_context", {"cell_id": "B0005"})]),
        _FakeResponse("end_turn", [_FakeTextBlock("Sorry, something went wrong fetching that.")]),
    ]
    holder = _install_fake_anthropic(monkeypatch, responses)
    answer, tools_used = answer_with_tools(
        "How is B0005 doing?", "B0005", 1, featured_dfs, bundles, None, api_key="sk-ant-fake",
    )
    # The loop must not raise -- it reports the tool failure back to the
    # model as a tool_result with is_error=True and continues.
    assert answer == "Sorry, something went wrong fetching that."
    second_call_messages = holder["client"].messages.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]["content"][0]
    assert tool_result_msg["is_error"] is True
    assert "boom" in tool_result_msg["content"]


def test_no_text_and_end_turn_still_returns_a_string_not_none(monkeypatch):
    """A defensive edge case: if the model somehow ends the turn with no
    text block at all, the caller's fallback contract (None means 'use
    the template path') must not be triggered by an empty string."""
    featured_dfs, bundles = _no_tool_dispatch_env(monkeypatch)
    responses = [_FakeResponse("end_turn", [])]
    _install_fake_anthropic(monkeypatch, responses)
    answer, tools_used = answer_with_tools(
        "anything", None, 1, featured_dfs, bundles, None, api_key="sk-ant-fake",
    )
    assert isinstance(answer, str) and answer != ""
