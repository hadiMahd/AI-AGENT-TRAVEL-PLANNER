"""Standalone probe for CHECK_INPUTS_PROMPT and ROUTE_PROMPT.

Two scenarios:
  (1) Single-turn Lebanon -> Bali — validates context check + IATA + named ml_predictor dict.
  (2) Multi-turn destination switch — validates that preferences from prior
      user turns (kids, adventure, budget) are carried into ml_predictor
      inputs even though the new query only says "change to Tokyo".

No DB / no FastAPI / no auth — just exercises the cheap LLM directly.
Run: uv run python -m scripts.probe_prompts   (from backend/)
"""

import asyncio
import sys

from agent.graph import _build_working_summary
from agent.prompts import CHECK_INPUTS_PROMPT, ROUTE_PROMPT
from llm.client import create_cheap_llm


async def scenario_single_turn(cheap) -> None:
    query = "Plan me a trip from Lebanon to Bali"
    summary = "origin=Lebanon | dest=Indonesia"

    check = await cheap.ainvoke(CHECK_INPUTS_PROMPT.format(
        query=query,
        summary=summary,
        dest_country="Indonesia",
    ))
    print("=== [1] CHECK_CONTEXT ===")
    print(check.content.strip())
    print()

    route = await cheap.ainvoke(ROUTE_PROMPT.format(
        query=query,
        destination_country="Indonesia",
        origin_country="Lebanon",
        destination_currency="IDR",
        origin_currency="LBP",
        summary=summary,
    ))
    print("=== [1] ROUTE ===")
    print(route.content.strip())
    print()


async def scenario_multi_turn(cheap) -> None:
    state = {
        "query": "actually change the destination to Tokyo",
        "origin_country": "Lebanon",
        "destination_country": "Japan",
        "history": [
            {"role": "user", "content": "Plan a trip from Lebanon to Bali for me, my wife, and our two kids. We love adventure and nature, but the budget is tight — around $3k total."},
            {"role": "assistant", "content": "Here's a Bali plan for a family of four with adventure focus on a tight budget..."},
            {"role": "user", "content": "actually change the destination to Tokyo"},
        ],
    }
    summary = _build_working_summary(state)
    print("=== [2] SUMMARY (computed) ===")
    print(summary)
    print()

    route = await cheap.ainvoke(ROUTE_PROMPT.format(
        query=state["query"],
        destination_country="Japan",
        origin_country="Lebanon",
        destination_currency="JPY",
        origin_currency="LBP",
        summary=summary,
    ))
    print("=== [2] ROUTE (after destination switch) ===")
    print(route.content.strip())


async def main() -> int:
    cheap = create_cheap_llm()
    await scenario_single_turn(cheap)
    await scenario_multi_turn(cheap)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
