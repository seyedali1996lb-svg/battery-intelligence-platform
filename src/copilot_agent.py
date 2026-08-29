"""
Battery Intelligence Copilot — Option B: real Claude tool-use.

Complements battery_copilot.py's template system (Option A), which stays
untouched -- chip-click questions keep using it exactly as before, since
it's zero-cost and deterministic. This module exists only for the
free-text ask box, and only when a personal Anthropic API key is
configured: instead of a human pre-fetching one cell's context and
handing the LLM a paragraph to rephrase, the model gets a small set of
real data-fetching tools and decides for itself which to call, in what
order, based on the actual question. This is what makes compositional
questions answerable ("which of my degrading cells has the worst fade
rate, and why") that no fixed keyword router could ever match.

Grounding is enforced structurally, not by convention: every tool wraps
an existing pure function from battery_copilot.py / knowledge_graph.py /
battery_knowledge.py / copilot_retrieval.py and returns only real
computed values as JSON -- the model never sees a human-authored
narrative it could paraphrase away from. The system prompt states the
same "never invent a number" contract battery_copilot.llm_answer()
already holds today.

answer_with_tools() returns (None, []) whenever it can't produce a
tool-based answer -- no API key, any API/network failure, or any other
exception -- so the caller falls back to today's template+retrieval path.
Same "never raise for the caller" philosophy as llm_answer()'s own broad
except Exception, and in the same spirit as adapter_contract.py's
write-back-adapter contract (a distinct, read-only contract here, not
wired into that module).
"""

import json

MODEL = "claude-sonnet-5"
MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = """You are the Battery Intelligence Copilot, an expert assistant embedded in a
battery health monitoring platform. You have tools that fetch real, already-computed data --
you never calculate, estimate, or invent a number yourself. Use the tools to gather whatever
data the question actually needs before answering; call more than one tool, or the same tool
for different cells, when the question requires it (e.g. comparing two cells, or finding the
worst cell in the fleet and then explaining why it's degrading).

Rules:
- State only values you received from a tool result. If a tool result doesn't contain a value
  you'd need, say so explicitly rather than guessing or estimating it.
- Format numbers exactly as returned -- do not round differently or convert units.
- Use plain English, minimal jargon, concise answers (aim for a short paragraph, not a report).
- Never recommend specific financial products or services.
- If a cell_id in the question doesn't match any tool result, say you don't have data for it
  rather than assuming which cell was meant.
"""

TOOL_DEFS = [
    {
        "name": "get_cell_context",
        "description": (
            "Get real, computed data for one specific battery cell: current SOH%, health "
            "status, cycle count, fade rate, resistance, remaining-useful-life (RUL) estimate "
            "with its reliability flag, and the top model features driving the SOH prediction. "
            "Call this whenever the question is about a specific cell_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cell_id": {"type": "string", "description": "The battery cell's ID, e.g. 'B0005' or 'Cell3'."},
            },
            "required": ["cell_id"],
        },
    },
    {
        "name": "get_fleet_stats",
        "description": (
            "Get real, computed fleet-wide statistics: cell count, mean/median/min/max SOH, "
            "mean fade rate, lists of cells at end-of-life or degrading, cells with unreliable "
            "RUL predictions, and cells ranked worst-to-best by SOH and by fade rate. Call this "
            "for any question about the fleet as a whole, or to find which cell(s) match a "
            "criterion (e.g. worst SOH, fastest fade)."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_cell_mechanism",
        "description": (
            "Get the degradation-mechanism verdict for one cell -- whether its capacity loss "
            "is dominated by Loss of Lithium Inventory (LLI), Loss of Active Material (LAM), "
            "a mix of both, or insufficient data to classify -- plus the confidence level and "
            "any corroborating literature citations. Call this when the question is about "
            "*why* a cell is degrading, not just its current numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cell_id": {"type": "string", "description": "The battery cell's ID."},
            },
            "required": ["cell_id"],
        },
    },
    {
        "name": "get_feature_citation",
        "description": (
            "Get the literature citation (title, DOI, relevance) backing one of the model's "
            "SOH-prediction features, if one is on file. Call this when the user asks for the "
            "source behind a specific feature named in get_cell_context's top_features list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_name": {
                    "type": "string",
                    "description": "The internal feature name, e.g. from a top_features entry's 'feature' key.",
                },
            },
            "required": ["feature_name"],
        },
    },
    {
        "name": "search_battery_knowledge",
        "description": (
            "Search a small curated corpus of battery-engineering background articles "
            "(chemistry differences, degradation mechanisms, safety standards, dQ/dV "
            "interpretation, etc.) for text relevant to a question. Call this for general "
            "battery-domain background that isn't specific to one cell or this fleet's own "
            "computed data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
]


def build_tools(featured_dfs: dict, bundles: dict, org_id: int, graph) -> tuple[list, dict]:
    """
    Returns (TOOL_DEFS, dispatch) where dispatch maps tool name -> a
    callable(tool_input: dict) -> JSON-serializable result. Each callable
    closes over featured_dfs/bundles/org_id/graph exactly once per call,
    mirroring how app/_pages/copilot.py already threads graph in from
    main.py (main.py:1464, graph=graph) -- no new plumbing for that part.
    """
    from battery_copilot import build_cell_context, build_fleet_stats
    from battery_knowledge import get_feature_citation
    from copilot_retrieval import retrieve

    def _get_cell_context(tool_input: dict) -> dict:
        cell_id = tool_input.get("cell_id", "")
        if cell_id not in featured_dfs:
            return {"error": f"Unknown cell_id {cell_id!r}. Valid cell_ids: {sorted(featured_dfs.keys())}"}
        return build_cell_context(cell_id, featured_dfs, bundles)

    def _get_fleet_stats(tool_input: dict) -> dict:
        return build_fleet_stats(org_id, featured_dfs, bundles)

    def _get_cell_mechanism(tool_input: dict) -> dict:
        cell_id = tool_input.get("cell_id", "")
        if cell_id not in featured_dfs:
            return {"error": f"Unknown cell_id {cell_id!r}. Valid cell_ids: {sorted(featured_dfs.keys())}"}
        if graph is None:
            return {"error": "Knowledge graph not available in this session."}
        from knowledge_graph import get_or_compute_mechanism, literature_for_mechanism, MECHANISM_VERDICT_TO_KEY
        try:
            edge = get_or_compute_mechanism(graph, cell_id, featured_dfs[cell_id])
        except Exception as e:
            return {"error": f"Could not compute a mechanism verdict for {cell_id}: {e}"}
        key = MECHANISM_VERDICT_TO_KEY.get(edge.get("verdict"), "insufficient_data")  # pyright: ignore[reportArgumentType, reportCallIssue]
        return {"mechanism": edge, "literature": literature_for_mechanism(graph, key)}

    def _get_feature_citation(tool_input: dict) -> dict:
        feature_name = tool_input.get("feature_name", "")
        citation = get_feature_citation(feature_name)
        return citation or {"error": f"No citation on file for feature {feature_name!r}."}

    def _search_battery_knowledge(tool_input: dict) -> dict:
        query = tool_input.get("query", "")
        results = retrieve(query, top_k=3)
        return {"results": results} if results else {"results": [], "note": "No matching background found for this query."}

    dispatch = {
        "get_cell_context":         _get_cell_context,
        "get_fleet_stats":          _get_fleet_stats,
        "get_cell_mechanism":       _get_cell_mechanism,
        "get_feature_citation":     _get_feature_citation,
        "search_battery_knowledge": _search_battery_knowledge,
    }
    return TOOL_DEFS, dispatch


def answer_with_tools(
    query: str,
    cell_id: "str | None",
    org_id: int,
    featured_dfs: dict,
    bundles: dict,
    graph,
    api_key: str,
) -> "tuple[str | None, list[str]]":
    """
    Run a manual Claude tool-use loop to answer a free-text question.

    Returns (answer_text, tools_used) on success, or (None, []) whenever
    a tool-based answer couldn't be produced -- no API key, any API/
    network exception, or exhausting MAX_TOOL_ITERATIONS with no final
    text. The caller (app/_pages/copilot.py) falls back to the existing
    template+retrieval path in every one of those cases -- this function
    never raises.
    """
    if not api_key:
        return None, []

    tools_used: list = []
    try:
        import anthropic as _ant

        tool_defs, dispatch = build_tools(featured_dfs, bundles, org_id, graph)
        client = _ant.Anthropic(api_key=api_key)

        system = SYSTEM_PROMPT
        if cell_id:
            system += f"\n\nThe user is currently viewing cell {cell_id!r} -- assume questions like 'this cell' refer to it."

        messages: list = [{"role": "user", "content": query}]

        for _ in range(MAX_TOOL_ITERATIONS):
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                tools=tool_defs,
                messages=messages,
            )

            if response.stop_reason == "refusal":
                return "I wasn't able to answer that question.", tools_used

            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                return (text or "I don't have enough grounded data to answer that."), tools_used

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tools_used.append(block.name)
                fn = dispatch.get(block.name)
                try:
                    result = fn(block.input) if fn else {"error": f"Unknown tool {block.name!r}"}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": str(e)}),
                        "is_error": True,
                    })
            messages.append({"role": "user", "content": tool_results})

        return (
            "I looked into this but couldn't finish within the tool-call budget for one "
            "question -- try asking about fewer cells at once.",
            tools_used,
        )
    except Exception:
        return None, []
