"""
Agent chat endpoint — SSE streaming for the travel planner agent.

POST /agent/chat accepts a query, runs the LangGraph agent, and streams
results back via Server-Sent Events (SSE).

SSE events:
- thinking:  agent is processing
- tool_start: a tool is about to run
- tool_result: a tool finished (success or error)
- needs_input: agent needs more info from the user
- final:     agent finished — includes response + tool logs + tokens

Per INSTRUCTIONS.md: "Stream responses if possible."
SSE is simpler than WebSockets and works with standard HTTP — no
special server or proxy configuration needed.

Webhook delivery (Requirement 8):
- After the final event, fire send_plan_email() as a background task
- Per INSTRUCTIONS.md: "Webhook failure must not break user-facing response."
"""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import build_graph
from agent.prompts import SYNTHESIS_PROMPT
from agent.state import AgentState
from db.session import get_db
from dependencies import CheapLlmDep, CurrentUserDep, EmbedderDep, ModelDep, StrongLlmDep
from models.alchemy import AgentRun, ToolLog, User
from models.schemas import ChatRequest
from services.send_plan import send_plan_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


async def _emit_events(event_type: str, data: dict, queue: list) -> None:
    """Push an SSE event into the queue for streaming."""
    queue.append({"event": event_type, "data": json.dumps(data)})


async def _run_and_stream(
    query: str,
    origin_country: str | None,
    history: list[dict],
    user: User,
    db: AsyncSession,
    cheap_llm: ChatOpenAI,
    strong_llm: ChatOpenAI,
    model,
    embedder: OpenAIEmbeddings,
    background_tasks: BackgroundTasks,
):
    """Run the agent graph and yield SSE events as they happen."""

    queue: list[dict] = []

    async def emit(event_type: str, data: dict) -> None:
        await _emit_events(event_type, data, queue)

    initial_state: AgentState = {
        "query": query,
        "intent": "travel",
        "rewritten_query": "",
        "origin_country": origin_country,
        "destination_country": None,
        "destination_lat": None,
        "destination_long": None,
        "destination_currency": None,
        "origin_currency": None,
        "needs_user_input": False,
        "user_question": None,
        "tool_results": {},
        "data": {},
        "tool_logs": [],
        "final_response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "history": history,
        "working_summary": "",
        "clarification_attempts": 0,
    }

    yield f"event: thinking\ndata: {json.dumps({'message': 'Analyzing your query...'})}\n\n"

    graph = build_graph(cheap_llm, strong_llm, model, embedder, db, emit_callback=emit)
    compiled = graph.compile()

    result = await compiled.ainvoke(initial_state)

    # Stream any queued tool events
    for event in queue:
        yield f"event: {event['event']}\ndata: {event['data']}\n\n"

    if result.get("intent") == "casual":
        final_response = result.get("final_response", "Hello! How can I help you today?")
        tool_logs = result.get("tool_logs", [])
        yield (
            f"event: final\ndata: {json.dumps({'response': final_response, 'tool_logs': tool_logs, 'prompt_tokens': 0, 'completion_tokens': 0})}\n\n"
        )
        background_tasks.add_task(
            _persist_run,
            user_id=user.id,
            query=query,
            response=final_response,
            tool_logs=tool_logs,
            prompt_tokens=0,
            completion_tokens=0,
            db_url=str(db.get_bind().url) if hasattr(db, "get_bind") else None,
        )
        return

    if result.get("needs_user_input"):
        yield f"event: needs_input\ndata: {json.dumps({'question': result.get('user_question', 'Where are you traveling from?')})}\n\n"
        return

    # Build synthesis prompt — short status lines + selective full payloads from `data`
    data_dict: dict = result.get("data", {}) or {}
    tool_results_str = ""
    for name, status in result.get("tool_results", {}).items():
        tool_results_str += f"- {name}: {status}\n"

    # Include full payloads: RAG content + actual numbers from other tools
    rag_hits = data_dict.get("rag_retriever")
    if isinstance(rag_hits, list) and rag_hits:
        tool_results_str += "\nRAG TOP HITS:\n"
        for hit in rag_hits[:3]:
            meta = hit.get("metadata", {}) or {}
            country = meta.get("country", "?")
            content = (hit.get("content") or "")[:300]
            tool_results_str += f"  - {country}: {content}\n"
    for name, payload in data_dict.items():
        if name == "rag_retriever" or payload is None:
            continue
        tool_results_str += f"\n{name} data:\n{json.dumps(payload, default=str)[:600]}\n"

    destination = result.get("destination_country") or "the destination"
    # Explicit guidance: tools searched by DESTINATION, not raw query words.
    # The raw query may contain origin ("Beirut") while RAG results are about
    # the destination ("Maldives"). The strong LLM must plan for the destination.
    guidance = (
        f"\nIMPORTANT: All tools searched for the destination: {destination}. "
        f"Only plan travel for {destination}, even if the raw query mentions "
        f"other locations (origin, stopovers, etc.).\n"
    )
    tool_results_str = guidance + tool_results_str

    prompt = SYNTHESIS_PROMPT.format(
        tool_results=tool_results_str,
        query=query,
        destination=destination,
    )

    # Stream synthesis tokens
    full_text = ""
    prompt_tokens = result.get("prompt_tokens", 0)
    completion_tokens = 0

    try:
        async for chunk in strong_llm.astream(prompt):
            text = chunk.content
            full_text += text
            completion_tokens += 1  # Approximate — actual tokens from metadata if available
            yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"
    except Exception as exc:
        logger.error("Synthesis streaming failed: %s", exc)
        if not full_text:
            full_text = "I gathered the information but couldn't generate the final plan. Please try again."

    tool_logs = result.get("tool_logs", [])

    yield (
        f"event: final\ndata: {json.dumps({'response': full_text, 'tool_logs': tool_logs, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})}\n\n"
    )

    yield (
        f"event: ask_email\ndata: {json.dumps({'question': 'Would you like me to send this travel plan to your email?', 'plan': full_text, 'destination': result.get('destination_country', '')})}\n\n"
    )

    # Persist AgentRun + ToolLogs to DB (background)
    background_tasks.add_task(
        _persist_run,
        user_id=user.id,
        query=query,
        response=full_text,
        tool_logs=tool_logs,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        db_url=str(db.get_bind().url) if hasattr(db, "get_bind") else None,
    )


async def _persist_run(
    user_id: str,
    query: str,
    response: str,
    tool_logs: list[dict],
    prompt_tokens: int,
    completion_tokens: int,
    db_url: str | None = None,
) -> None:
    """Write AgentRun + ToolLog rows to DB after the response is sent."""
    from db.session import get_session_factory

    session_factory = get_session_factory()
    async with session_factory() as db:
        try:
            run = AgentRun(
                user_id=user_id,
                query=query,
                response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=0.0,
                status="completed",
            )
            db.add(run)
            await db.flush()

            for log in tool_logs:
                tl = ToolLog(
                    run_id=run.id,
                    tool_name=log.get("tool_name", "unknown"),
                    input_payload=log.get("input_payload"),
                    output_payload={"text": log.get("output_payload", "")} if isinstance(log.get("output_payload"), str) else log.get("output_payload"),
                    status=log.get("status", "success"),
                    latency_ms=log.get("latency_ms"),
                )
                db.add(tl)

            await db.commit()
            logger.info("Persisted agent run %s with %d tool logs", run.id, len(tool_logs))
        except Exception:
            logger.exception("Failed to persist agent run")
            await db.rollback()


@router.post(
    "/chat",
    summary="Chat with the travel planner agent (SSE streaming)",
    description=(
        "Sends a travel query to the AI agent. The agent runs tools "
        "(RAG, ML, weather, flights, FX) and streams results back via SSE. "
        "If the agent needs more info (e.g. origin country), it asks."
    ),
)
async def chat(
    body: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    user: User = CurrentUserDep,
    db: AsyncSession = Depends(get_db),
    cheap_llm: ChatOpenAI = CheapLlmDep,
    strong_llm: ChatOpenAI = StrongLlmDep,
    model=ModelDep,
    embedder: OpenAIEmbeddings = EmbedderDep,
):
    """
    Run the travel planner agent and stream results via SSE.

    The response is a stream of SSE events, not a JSON body.
    Frontend should use EventSource or fetch+ReadableStream to consume.
    """
    return StreamingResponse(
        _run_and_stream(
            query=body.query,
            origin_country=body.origin_country,
            history=[msg.model_dump() for msg in body.history],
            user=user,
            db=db,
            cheap_llm=cheap_llm,
            strong_llm=strong_llm,
            model=model,
            embedder=embedder,
            background_tasks=background_tasks,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/history",
    summary="Get the current user's agent run history",
    description="Returns past agent runs with their tool logs, scoped to the authenticated user.",
)
async def get_history(
    user: User = CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return the user's past agent runs, most recent first."""
    result = await db.execute(
        select(AgentRun)
        .where(AgentRun.user_id == user.id)
        .order_by(AgentRun.created_at.desc())
        .limit(20)
    )
    runs = result.scalars().all()

    return [
        {
            "id": r.id,
            "query": r.query,
            "response": r.response,
            "status": r.status,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "created_at": str(r.created_at) if r.created_at else None,
        }
        for r in runs
    ]


from pydantic import BaseModel as _BaseModel


class _SendEmailRequest(_BaseModel):
    email: str
    plan: str
    destination: str = ""


@router.post("/send-email", summary="Email a travel plan to the user")
async def send_email(
    body: _SendEmailRequest,
    user: User = CurrentUserDep,
) -> dict:
    """Send the travel plan to the provided email address via Resend."""
    ok = await send_plan_email(
        to_email=body.email,
        plan_content=body.plan,
        destination=body.destination or "your destination",
    )
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=502, detail="Failed to send email — check RESEND_API_KEY.")
    return {"message": f"Plan sent to {body.email}"}
