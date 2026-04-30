"""
Agent graph — the travel planner's orchestration engine.

Flow:
  classify_intent ─┬─► casual_reply ────────────────────────────────► END
                   └─► check_context ─┬─► needs_user_input ──────────► END
                                      └─► route_and_run_tools ────────► END

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

_DEST_KEYWORDS = (
    "trip", "travel", "visit", "plan", "vacation", "holiday", "tour",
    "fly", "flight", "weather", "currency", "exchange", "hotel", "from", "to",
    "going", "explore", "destination", "itinerary", "book",
)


def _has_destination_keyword(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _DEST_KEYWORDS)


def _build_working_summary(state: AgentState) -> str:
    """Compress conversation context for cheap-model prompts.

    Carries forward facts/preferences the user mentioned in earlier turns
    (budget, kids, adventure, etc.) so prompts that only see the current
    query can still extract structured signal.
    """
    history = state.get("history") or []
    parts: list[str] = []
    if state.get("origin_country"):
        parts.append(f"origin={state['origin_country']}")
    if state.get("destination_country"):
        parts.append(f"dest={state['destination_country']}")
    if history:
        prior_user_msgs = [
            str(m.get("content", "")).strip()
            for m in history
            if m.get("role") == "user" and str(m.get("content", "")).strip()
        ]
        current_query = (state.get("query") or "").strip()
        if prior_user_msgs and prior_user_msgs[-1] == current_query:
            prior_user_msgs = prior_user_msgs[:-1]
        if prior_user_msgs:
            joined = " || ".join(m[:200] for m in prior_user_msgs[-5:])
            parts.append(f"prior_user_turns=[{joined}]")
        last_asst = next(
            (m for m in reversed(history) if m.get("role") == "assistant"),
            None,
        )
        if last_asst:
            parts.append(f"prev_reply={str(last_asst.get('content', ''))[:120]}")
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
    """Node 0: Classify query as casual or travel."""
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
    if intent not in ("casual", "travel"):
        intent = "travel"
    logger.info("Intent classified as %s: %s", intent.upper(), query[:50])
    return {"intent": intent}


async def casual_reply(state: AgentState, cheap_llm: ChatOpenAI) -> dict:
    """Node 0b: Generate a friendly reply — handles chitchat and fictional destinations."""
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


async def check_context(
    state: AgentState,
    cheap_llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    db: AsyncSession,
) -> dict:
    """Node 1: Detect destination via embedding; ask only for missing required inputs.

    Three-tier priority for destination determination:
      1. LLM-determined destination (from prompt output)
      2. Prior state destination (carried from earlier turns)
      3. Embedding search result (fresh detection)
    This prevents origin-like replies (e.g. "lebanon" answering "where from?")
    from hijacking a previously established destination (e.g. "Maldives").
    """

    prior_dest = state.get("destination_country")
    prior_lat = state.get("destination_lat")
    prior_long = state.get("destination_long")

    # ---- Embedding search: detect destination from current query ----
    # Always run — it's one cheap API call and provides signal for
    # "change to Tokyo" cases. The 3-tier priority below determines
    # which destination actually wins.
    detected_dest = None
    detected_lat = None
    detected_long = None
    try:
        query_embedding = await embed_query(embedder, state["query"])
        results = await similarity_search(db, query_embedding, k=1)
        if results:
            meta = results[0].get("metadata", {})
            detected_dest = meta.get("country")
            detected_lat = meta.get("latitude")
            detected_long = meta.get("longitude")
    except Exception:
        logger.warning("Embedding search failed — proceeding without destination detection")

    # ---- Build prompt with full context ----
    summary = _build_working_summary({**state, "destination_country": detected_dest})
    prompt = CHECK_INPUTS_PROMPT.format(
        query=state["query"],
        summary=summary,
        dest_country=detected_dest or "not detected",
        prior_destination=prior_dest or "none",
    )

    try:
        response = await asyncio.wait_for(cheap_llm.ainvoke(prompt), timeout=90)
        parsed = _parse_json_response(response.content)
        usage = getattr(response, "response_metadata", {}).get("token_usage", {}) or {}
    except asyncio.TimeoutError:
        logger.warning("Check context timed out — asking user for origin")
        parsed = None
        usage = {}

    # ---- Three-tier destination determination ----
    # 1. LLM explicitly sets a destination
    # 2. Carry forward prior destination from state
    # 3. Fall back to embedding search result
    llm_dest = (parsed or {}).get("destination_country")
    final_dest = llm_dest or prior_dest or detected_dest

    # For lat/long: use prior values when keeping prior destination,
    # otherwise use the detected ones (they match the embedding result)
    if final_dest == prior_dest and prior_dest:
        final_lat = prior_lat
        final_long = prior_long
    else:
        final_lat = detected_lat
        final_long = detected_long

    base_updates = {
        "destination_country": final_dest,
        "destination_lat": final_lat,
        "destination_long": final_long,
        "destination_currency": get_currency_for_country(final_dest) if final_dest else None,
        "rewritten_query": state["query"],
        "prompt_tokens": state.get("prompt_tokens", 0) + (usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": state.get("completion_tokens", 0) + (usage.get("completion_tokens", 0) or 0),
    }

    needs_dest = (parsed or {}).get("needs_dest", False)
    needs_origin = (parsed or {}).get("needs_origin", True)
    has_origin = (parsed or {}).get("has_origin", False)
    question = (parsed or {}).get("question")

    # If destination is still unknown after all three tiers, we must ask
    if needs_dest and not final_dest:
        prior = state.get("clarification_attempts", 0)
        return {
            **base_updates,
            "origin_country": None,
            "origin_currency": None,
            "needs_user_input": True,
            "user_question": question or "Which destination are you interested in?",
            "clarification_attempts": prior + 1,
        }

    # Proceed without asking if origin is not needed, or if it's already known
    if not needs_origin or has_origin:
        origin_country = (parsed or {}).get("origin_country") or state.get("origin_country")
        return {
            **base_updates,
            "origin_country": origin_country,
            "origin_currency": get_currency_for_country(origin_country) if origin_country else None,
            "needs_user_input": False,
            "user_question": None,
            "clarification_attempts": 0,
        }

    # Origin is needed but not provided — ask the user
    prior = state.get("clarification_attempts", 0)
    return {
        **base_updates,
        "origin_country": None,
        "origin_currency": None,
        "needs_user_input": True,
        "user_question": question or "Where are you traveling from?",
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
        summary=_build_working_summary(state) or "(no prior turns)",
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
        tools_to_run = [{"name": "rag_retriever", "input": {"query": state["query"], "k": 3}}]
    else:
        tools_to_run = parsed["tools"]

    # ---- Fix: force rag_retriever to search by destination, not the raw user query ----
    # The cheap LLM may pass the raw user message ("lebanon") as the RAG query.
    # We override it to always search by the established destination country.
    # Also inject rag_retriever if it's missing and we have a known destination.
    destination = state.get("destination_country")
    has_rag = any(t.get("name") == "rag_retriever" for t in tools_to_run)
    if destination:
        if not has_rag:
            tools_to_run.insert(0, {"name": "rag_retriever", "input": {"query": destination, "k": 3}})
            logger.info("Injected rag_retriever for destination: %s", destination)
        else:
            for t in tools_to_run:
                if t.get("name") == "rag_retriever":
                    t["input"]["query"] = destination
                    logger.info("Rerouted rag_retriever query to destination: %s", destination)

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
        "needs_user_input": False,
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
        k = min(tool_input.get("k", 3), 3)
        try:
            query_embedding = await embed_query(embedder, query)
            results = await similarity_search(db, query_embedding, k)
            if not results:
                return [], "rag_retriever: no documents found"
            filtered = [
                r for r in results
                if r.get("score", 0) >= 0.6
            ]
            if not filtered:
                logger.warning("All RAG results below 0.6 threshold (best: %.3f)", results[0].get("score", 0))
                return [], "rag_retriever: all results below relevance threshold"
            payload = [
                {
                    "content": r["content"][:300],
                    "metadata": r["metadata"],
                    "score": round(r["score"], 3),
                }
                for r in filtered
            ]
            return payload, _summarize(name, payload)
        except Exception as exc:
            logger.error("rag_retriever failed: %s", exc)
            return None, f"[rag_retriever ERROR] {exc}"

    if name == "ml_predictor":
        try:
            if not isinstance(tool_input, dict):
                return None, f"[ml_predictor ERROR] expected dict input, got {type(tool_input).__name__}"
            features = TravelStyleFeatures(**tool_input)
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

    async def _check(state: AgentState) -> dict:
        return await check_context(state, cheap_llm, embedder, db)

    async def _route_run(state: AgentState) -> dict:
        return await route_and_run_tools(state, cheap_llm, model, embedder, db, emit_callback)

    graph.add_node("classify_intent", _classify)
    graph.add_node("casual_reply", _casual)
    graph.add_node("check_context", _check)
    graph.add_node("route_and_run_tools", _route_run)

    graph.set_entry_point("classify_intent")

    def _after_classify(state: AgentState) -> str:
        if state.get("intent") == "casual":
            return "casual_reply"
        return "check_context"

    graph.add_conditional_edges("classify_intent", _after_classify)
    graph.add_edge("casual_reply", END)

    def _after_check(state: AgentState) -> str:
        if state.get("needs_user_input"):
            # Clarification loop guard: after 3 failed attempts, proceed without origin
            if state.get("clarification_attempts", 0) >= 3:
                logger.warning("Clarification loop detected — proceeding without origin")
                return "route_and_run_tools"
            return END
        return "route_and_run_tools"

    graph.add_conditional_edges("check_context", _after_check)
    graph.add_edge("route_and_run_tools", END)

    return graph
