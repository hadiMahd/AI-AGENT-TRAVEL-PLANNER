"""
Agent graph — the travel planner's orchestration engine.

Flow:
  classify_intent ─┬─► casual_reply ─────────────────► END
                   ├─► quick_info_run ───────────────► END
                   └─► rewrite_query ─► check_inputs ─┬─► route_and_run_tools ─► END
                                                      └─► END (needs_user_input)

Tools are executed as plain async functions — runtime deps (embedder, db, model)
are bound at graph build time. The allowlist still validates tool names before
execution.

State splits heavy data from prompt context:
- `tool_results: dict[str, str]` holds short status messages (prompt-injectable).
- `data: dict[str, Any]` holds full structured tool payloads (UI / synthesis).
- `working_summary: str` is a deterministic compact summary of the conversation —
  replaces injecting raw `history` into every cheap-model prompt.
"""

import asyncio
import json
import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from agent.prompts import (
    CASUAL_REPLY_PROMPT,
    CHECK_INPUTS_PROMPT,
    CLASSIFY_INTENT_PROMPT,
    INFO_SYNTHESIS_PROMPT,
    QUICK_TOOL_PROMPT,
    REWRITE_PROMPT,
    ROUTE_PROMPT,
    SYNTHESIS_PROMPT,
)
from agent.state import AgentState
from rag.embedder import embed_query
from rag.retriever import similarity_search
from services.flights import search_flights
from services.fx import get_currency_for_country, get_exchange_rate
from services.ml_inference import infer_travel_style
from services.weather import get_weather, get_weather_by_city
from models.schemas import TravelStyleFeatures
from tools import validate_tool

logger = logging.getLogger(__name__)

MAX_TOOL_CHARS = 400  # retained for router fallback when injecting full data

_DEST_KEYWORDS = (
    "trip", "travel", "visit", "plan", "vacation", "holiday", "tour",
    "fly", "flight", "weather", "currency", "exchange", "hotel", "from", "to",
    "going", "explore", "destination", "itinerary", "book",
)


def _has_destination_keyword(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _DEST_KEYWORDS)


def _build_working_summary(state: AgentState) -> str:
    """Compress conversation context to a single line for cheap-model prompts."""
    history = state.get("history") or []
    parts: list[str] = []
    if state.get("origin_country"):
        parts.append(f"origin={state['origin_country']}")
    if state.get("destination_country"):
        parts.append(f"dest={state['destination_country']}")
    if history:
        last_user = next((m for m in reversed(history) if m.get("role") == "user"), None)
        if last_user:
            parts.append(f"prev_user={str(last_user.get('content', ''))[:120]}")
        last_asst = next(
            (m for m in reversed(history[:-1]) if m.get("role") == "assistant"),
            None,
        )
        if last_asst:
            parts.append(f"prev_reply={str(last_asst.get('content', ''))[:80]}")
    return " | ".join(parts)


def _parse_json_response(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return None


def _summarize(name: str, payload: Any) -> str:
    """Compress a tool payload to a short status message for prompt injection."""
    if name == "rag_retriever":
        if not isinstance(payload, list) or not payload:
            return "rag_retriever: no documents found"
        countries = ", ".join(sorted({
            (r.get("metadata") or {}).get("country", "?") for r in payload[:3]
        }))
        top = max((r.get("score", 0) for r in payload), default=0)
        return f"rag_retriever: {len(payload)} hits about {countries} (top score {top:.2f})"
    if name == "weather_fetcher":
        if not isinstance(payload, dict):
            return "weather_fetcher: unavailable"
        return (
            f"weather_fetcher: {payload.get('temp_c')}C, "
            f"{payload.get('condition')}, humidity {payload.get('humidity')}%"
        )
    if name == "fx_checker":
        if not isinstance(payload, dict):
            return "fx_checker: unavailable"
        return f"fx_checker: 1 {payload.get('base')} = {payload.get('rate')} {payload.get('target')}"
    if name == "flight_searcher":
        if not isinstance(payload, list):
            return "flight_searcher: unavailable"
        return f"flight_searcher: {len(payload)} options found"
    if name == "ml_predictor":
        if not isinstance(payload, dict):
            return "ml_predictor: unavailable"
        return (
            f"ml_predictor: style={payload.get('predicted_style')} "
            f"(conf {payload.get('confidence', 0):.2f})"
        )
    return f"{name}: completed"


async def classify_intent(state: AgentState, cheap_llm: ChatOpenAI) -> dict:
    """Node 0: Classify query as casual / info / travel."""
    query = state["query"].strip()

    # Deterministic pre-filter: very short messages with no travel keyword are casual,
    # unless the prior assistant turn was a clarification question (e.g. "where from?").
    if len(query.split()) <= 3 and not _has_destination_keyword(query):
        history = state.get("history") or []
        recent = history[-2:]
        prior_was_question = any(
            m.get("role") == "assistant" and "?" in str(m.get("content", ""))
            for m in recent
        )
        if not prior_was_question:
            logger.info("Pre-filter: casual (short, no travel keyword): %s", query[:30])
            return {"intent": "casual"}

    summary = _build_working_summary(state)
    prompt = CLASSIFY_INTENT_PROMPT.format(query=query, summary=summary)
    try:
        response = await asyncio.wait_for(cheap_llm.ainvoke(prompt), timeout=60)
    except asyncio.TimeoutError:
        logger.warning("Intent classification timed out — assuming travel")
        return {"intent": "travel"}

    parsed = _parse_json_response(response.content)
    intent = (parsed or {}).get("intent", "travel")
    if intent not in ("casual", "info", "travel", "plan"):
        intent = "travel"
    if intent == "plan":
        intent = "travel"
    logger.info("Intent classified as %s: %s", intent.upper(), query[:50])
    return {"intent": intent}


async def casual_reply(state: AgentState, cheap_llm: ChatOpenAI) -> dict:
    """Node 0b: Generate a friendly greeting for casual conversation."""
    prompt = CASUAL_REPLY_PROMPT.format(query=state["query"])
    try:
        response = await asyncio.wait_for(cheap_llm.ainvoke(prompt), timeout=60)
        reply = response.content.strip()
    except asyncio.TimeoutError:
        reply = "Hey there! I'm your travel planner. Tell me where you'd like to go!"

    return {
        "final_response": reply,
        "needs_user_input": False,
        "tool_logs": [],
    }


async def _run_one_tool(
    spec: dict,
    model: Any,
    embedder: OpenAIEmbeddings,
    db: AsyncSession,
    emit_callback: Any = None,
) -> tuple[str, Any, str, dict | None]:
    """Run one tool and return (name, payload, status_msg, log)."""
    name = spec.get("name", "")
    tool_input = spec.get("input", {}) or {}

    if not validate_tool(name):
        logger.warning("Rejected hallucinated tool: %s", name)
        return name, None, f"[REJECTED] {name}", None

    if emit_callback:
        await emit_callback("tool_start", {"tool": name, "input": tool_input})

    start = time.monotonic()
    try:
        payload, status = await _execute_tool(name, tool_input, model, embedder, db)
        elapsed = int((time.monotonic() - start) * 1000)
        log = {
            "tool_name": name,
            "input_payload": tool_input,
            "output_payload": status[:500],
            "status": "success",
            "latency_ms": elapsed,
        }
        if emit_callback:
            await emit_callback("tool_result", {"tool": name, "output": status[:300], "latency_ms": elapsed})
        return name, payload, status, log
    except Exception as exc:
        elapsed = int((time.monotonic() - start) * 1000)
        err = f"[{name} ERROR] {exc}"
        log = {
            "tool_name": name,
            "input_payload": tool_input,
            "output_payload": err,
            "status": "error",
            "latency_ms": elapsed,
        }
        if emit_callback:
            await emit_callback("tool_result", {"tool": name, "error": str(exc), "latency_ms": elapsed})
        return name, None, err, log


async def quick_info_run(
    state: AgentState,
    cheap_llm: ChatOpenAI,
    model: Any,
    embedder: OpenAIEmbeddings,
    db: AsyncSession,
    emit_callback: Any = None,
) -> dict:
    """Node 1b: Pick and run a single tool for quick info queries."""
    summary = _build_working_summary(state)
    prompt = QUICK_TOOL_PROMPT.format(query=state["query"], summary=summary)
    try:
        response = await asyncio.wait_for(cheap_llm.ainvoke(prompt), timeout=60)
        parsed = _parse_json_response(response.content)
    except asyncio.TimeoutError:
        logger.warning("Quick tool routing timed out — falling back to rag_retriever")
        parsed = None

    if not parsed or "tool" not in parsed:
        parsed = {"tool": "rag_retriever", "input": {"query": state["query"], "k": 3}}

    spec = {"name": parsed["tool"], "input": parsed.get("input", {})}
    if not validate_tool(spec["name"]):
        spec = {"name": "rag_retriever", "input": {"query": state["query"], "k": 3}}

    name, payload, status, log = await _run_one_tool(spec, model, embedder, db, emit_callback)
    return {
        "tool_results": {name: status},
        "data": {name: payload} if payload is not None else {},
        "tool_logs": [log] if log else [],
    }


async def rewrite_query(state: AgentState, cheap_llm: ChatOpenAI) -> dict:
    summary = _build_working_summary(state)
    prompt = REWRITE_PROMPT.format(query=state["query"], summary=summary)
    try:
        response = await asyncio.wait_for(cheap_llm.ainvoke(prompt), timeout=90)
    except asyncio.TimeoutError:
        logger.warning("Rewrite query timed out — using original query")
        return {
            "rewritten_query": state["query"],
            "prompt_tokens": state.get("prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0),
        }
    rewritten = response.content.strip()
    logger.info("Query rewritten: %s → %s", state["query"][:50], rewritten[:50])
    usage = getattr(response, "response_metadata", {}).get("token_usage", {}) or {}
    return {
        "rewritten_query": rewritten,
        "prompt_tokens": state.get("prompt_tokens", 0) + (usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": state.get("completion_tokens", 0) + (usage.get("completion_tokens", 0) or 0),
    }


async def check_inputs(
    state: AgentState,
    cheap_llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    db: AsyncSession,
) -> dict:
    query_embedding = await embed_query(embedder, state["rewritten_query"])
    results = await similarity_search(db, query_embedding, k=1)

    dest_country = None
    dest_lat = None
    dest_long = None
    if results:
        meta = results[0].get("metadata", {})
        dest_country = meta.get("country")
        dest_lat = meta.get("latitude")
        dest_long = meta.get("longitude")

    # Summary built with the freshly-resolved destination so the cheap model has it
    summary = _build_working_summary({**state, "destination_country": dest_country})
    prompt = CHECK_INPUTS_PROMPT.format(query=state["query"], summary=summary)

    try:
        response = await asyncio.wait_for(cheap_llm.ainvoke(prompt), timeout=90)
        parsed = _parse_json_response(response.content)
        usage = getattr(response, "response_metadata", {}).get("token_usage", {}) or {}
    except asyncio.TimeoutError:
        logger.warning("Check inputs timed out — asking user for origin")
        parsed = None
        usage = {}

    base_updates = {
        "destination_country": dest_country,
        "destination_lat": dest_lat,
        "destination_long": dest_long,
        "destination_currency": get_currency_for_country(dest_country) if dest_country else None,
        "prompt_tokens": state.get("prompt_tokens", 0) + (usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": state.get("completion_tokens", 0) + (usage.get("completion_tokens", 0) or 0),
    }

    if parsed and parsed.get("has_origin"):
        origin_country = parsed.get("origin_country")
        return {
            **base_updates,
            "origin_country": origin_country,
            "origin_currency": get_currency_for_country(origin_country) if origin_country else None,
            "needs_user_input": False,
            "user_question": None,
            "clarification_attempts": 0,
        }

    prior = state.get("clarification_attempts", 0)
    return {
        **base_updates,
        "origin_country": None,
        "origin_currency": None,
        "needs_user_input": True,
        "user_question": (parsed or {}).get("question") or "Where are you traveling from?",
        "clarification_attempts": prior + 1,
    }


async def route_and_run_tools(
    state: AgentState,
    cheap_llm: ChatOpenAI,
    model: Any,
    embedder: OpenAIEmbeddings,
    db: AsyncSession,
    emit_callback: Any = None,
) -> dict:
    prompt = ROUTE_PROMPT.format(
        query=state["query"],
        destination_country=state.get("destination_country") or "Unknown",
        origin_country=state.get("origin_country") or "Unknown",
        destination_currency=state.get("destination_currency") or "Unknown",
        origin_currency=state.get("origin_currency") or "Unknown",
    )
    try:
        response = await asyncio.wait_for(cheap_llm.ainvoke(prompt), timeout=90)
    except asyncio.TimeoutError:
        logger.warning("Route prompt timed out — running rag_retriever only")
        response = None

    parsed = _parse_json_response(response.content) if response is not None else None

    base_tokens = state.get("prompt_tokens", 0)
    comp_tokens = state.get("completion_tokens", 0)
    if response is not None:
        usage = getattr(response, "response_metadata", {}).get("token_usage", {}) or {}
        base_tokens += usage.get("prompt_tokens", 0) or 0
        comp_tokens += usage.get("completion_tokens", 0) or 0

    if not parsed or "tools" not in parsed:
        logger.warning("Route parsing failed — running rag_retriever only")
        tools_to_run = [{"name": "rag_retriever", "input": {"query": state["rewritten_query"], "k": 3}}]
    else:
        tools_to_run = parsed["tools"]

    # Fan-out: run all selected tools concurrently
    fan_results = await asyncio.gather(*[
        _run_one_tool(spec, model, embedder, db, emit_callback) for spec in tools_to_run
    ])

    tool_results: dict[str, str] = {}
    data: dict[str, Any] = {}
    tool_logs: list[dict] = []
    for name, payload, status, log in fan_results:
        if not name:
            continue
        tool_results[name] = status
        if payload is not None:
            data[name] = payload
        if log is not None:
            tool_logs.append(log)

    return {
        "tool_results": tool_results,
        "data": data,
        "tool_logs": tool_logs,
        "prompt_tokens": base_tokens,
        "completion_tokens": comp_tokens,
        "needs_user_input": False,  # cleared on the memory-aware escape path
    }


async def _execute_tool(
    name: str,
    tool_input: dict,
    model: Any,
    embedder: OpenAIEmbeddings,
    db: AsyncSession,
) -> tuple[Any, str]:
    """Execute a tool by name. Returns (structured_payload, status_string)."""

    if name == "rag_retriever":
        query = tool_input.get("query", "")
        k = tool_input.get("k", 5)
        try:
            query_embedding = await embed_query(embedder, query)
            results = await similarity_search(db, query_embedding, k)
            if not results:
                return [], "rag_retriever: no documents found"
            payload = [
                {
                    "content": r["content"][:300],
                    "metadata": r["metadata"],
                    "score": round(r["score"], 3),
                }
                for r in results
            ]
            return payload, _summarize(name, payload)
        except Exception as exc:
            logger.error("rag_retriever failed: %s", exc)
            return None, f"[rag_retriever ERROR] {exc}"

    if name == "ml_predictor":
        try:
            features = TravelStyleFeatures(
                active_movement=tool_input.get("active_movement", 0.5),
                relaxation=tool_input.get("relaxation", 0.5),
                cultural_interest=tool_input.get("cultural_interest", 0.5),
                cost_sensitivity=tool_input.get("cost_sensitivity", 0.5),
                luxury_preference=tool_input.get("luxury_preference", 0.5),
                family_friendliness=tool_input.get("family_friendliness", 0.5),
                nature_orientation=tool_input.get("nature_orientation", 0.5),
                social_group=tool_input.get("social_group", 0.5),
            )
            payload = await infer_travel_style(model, features)
            return payload, _summarize(name, payload)
        except Exception as exc:
            logger.error("ml_predictor failed: %s", exc)
            return None, f"[ml_predictor ERROR] {exc}"

    if name == "weather_fetcher":
        try:
            city = tool_input.get("city")
            if city:
                payload = await get_weather_by_city(city)
            else:
                lat = tool_input.get("latitude", 0)
                long = tool_input.get("longitude", 0)
                payload = await get_weather(lat, long)
            if payload is None:
                return None, "[weather_fetcher] unavailable"
            return payload, _summarize(name, payload)
        except Exception as exc:
            logger.error("weather_fetcher failed: %s", exc)
            return None, f"[weather_fetcher ERROR] {exc}"

    if name == "flight_searcher":
        try:
            origin = tool_input.get("origin", "")
            destination = tool_input.get("destination", "")
            payload = await search_flights(origin, destination)
            if payload is None:
                return None, "[flight_searcher] unavailable"
            if not payload:
                return [], f"flight_searcher: no results for {origin} -> {destination}"
            return payload, _summarize(name, payload)
        except Exception as exc:
            logger.error("flight_searcher failed: %s", exc)
            return None, f"[flight_searcher ERROR] {exc}"

    if name == "fx_checker":
        try:
            base = tool_input.get("base_currency", "USD").upper()
            target = tool_input.get("target_currency", "USD").upper()
            if base == target:
                payload = {"rate": 1.0, "base": base, "target": target}
                return payload, _summarize(name, payload)
            payload = await get_exchange_rate(base, target)
            if payload is None:
                return None, f"[fx_checker] {base}->{target} unavailable"
            return payload, _summarize(name, payload)
        except Exception as exc:
            logger.error("fx_checker failed: %s", exc)
            return None, f"[fx_checker ERROR] {exc}"

    return None, f"[ERROR] Unknown tool: {name}"


async def synthesize(state: AgentState, strong_llm: ChatOpenAI) -> dict:
    """Optional in-graph synthesis (router uses streaming variant in agent.py)."""
    tool_summaries = "\n".join(f"- {n}: {s}" for n, s in state.get("tool_results", {}).items())
    prompt = SYNTHESIS_PROMPT.format(
        tool_results=tool_summaries,
        query=state["query"],
        destination=state.get("destination_country", "the destination"),
    )
    try:
        response = await asyncio.wait_for(strong_llm.ainvoke(prompt), timeout=90)
    except asyncio.TimeoutError:
        logger.error("Synthesis timed out")
        return {
            "final_response": "I gathered the information but couldn't generate the final plan in time. Please try again.",
            "prompt_tokens": state.get("prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0),
        }

    usage = getattr(response, "response_metadata", {}).get("token_usage", {}) or {}
    return {
        "final_response": response.content,
        "prompt_tokens": state.get("prompt_tokens", 0) + (usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": state.get("completion_tokens", 0) + (usage.get("completion_tokens", 0) or 0),
    }


def build_graph(
    cheap_llm: ChatOpenAI,
    strong_llm: ChatOpenAI,
    model: Any,
    embedder: OpenAIEmbeddings,
    db: AsyncSession,
    emit_callback: Any = None,
) -> StateGraph:
    graph = StateGraph(AgentState)

    async def _classify(state: AgentState) -> dict:
        return await classify_intent(state, cheap_llm)

    async def _casual(state: AgentState) -> dict:
        return await casual_reply(state, cheap_llm)

    async def _quick_info(state: AgentState) -> dict:
        return await quick_info_run(state, cheap_llm, model, embedder, db, emit_callback)

    async def _rewrite(state: AgentState) -> dict:
        return await rewrite_query(state, cheap_llm)

    async def _check(state: AgentState) -> dict:
        return await check_inputs(state, cheap_llm, embedder, db)

    async def _route_run(state: AgentState) -> dict:
        return await route_and_run_tools(state, cheap_llm, model, embedder, db, emit_callback)

    graph.add_node("classify_intent", _classify)
    graph.add_node("casual_reply", _casual)
    graph.add_node("quick_info_run", _quick_info)
    graph.add_node("rewrite_query", _rewrite)
    graph.add_node("check_inputs", _check)
    graph.add_node("route_and_run_tools", _route_run)

    graph.set_entry_point("classify_intent")

    def _after_classify(state: AgentState) -> str:
        if state.get("intent") == "casual":
            return "casual_reply"
        if state.get("intent") == "info":
            return "quick_info_run"
        return "rewrite_query"

    graph.add_conditional_edges("classify_intent", _after_classify)
    graph.add_edge("casual_reply", END)
    graph.add_edge("quick_info_run", END)
    graph.add_edge("rewrite_query", "check_inputs")

    def _after_check(state: AgentState) -> str:
        if state.get("needs_user_input"):
            # Memory-aware escape: 3+ failed clarifications -> proceed without origin
            if state.get("clarification_attempts", 0) >= 3:
                logger.warning("Clarification loop detected — proceeding without origin")
                return "route_and_run_tools"
            return END
        return "route_and_run_tools"

    graph.add_conditional_edges("check_inputs", _after_check)
    graph.add_edge("route_and_run_tools", END)

    return graph
