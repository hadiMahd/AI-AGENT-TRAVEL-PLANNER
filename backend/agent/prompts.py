"""
Prompt templates for the travel planner agent.

Cheap-model prompts are kept tight and mechanical (input -> tool/JSON -> output).
Strategic phrasing lives in SYNTHESIS_PROMPT, which the strong model handles.

The {summary} placeholder receives a short, deterministic working summary built
from state (origin, destination, last user/assistant turn) — far cheaper than
re-injecting raw history into every prompt.
"""

CLASSIFY_INTENT_PROMPT = """Classify the user message for a travel planner.

Context: {summary}
Message: "{query}"

Classes:
- "casual": greetings, chitchat, thanks, bye
- "info": single-fact question (weather, exchange rate, place lookup) — no flights, no full plan
- "plan": full trip planning (itinerary, flights, multi-step)

A short follow-up that continues a prior plan ("change to Bali", "what about Tokyo?") is "plan".

JSON only: {{"intent": "casual"|"info"|"plan"}}"""

CASUAL_REPLY_PROMPT = """User said: "{query}"

Reply in 1-2 sentences as a friendly travel planner. Invite them to share a destination."""

REWRITE_PROMPT = """Rewrite the travel query for semantic search.

Context: {summary}
Query: "{query}"

Rules: extract destination + preferences, 1-2 sentences, do not invent details.

Rewritten:"""

CHECK_INPUTS_PROMPT = """{summary}
Query: "{query}"

Did the user state where they're traveling FROM (origin country)? Use the context above too.

JSON: {{"has_origin": bool, "origin_country": "country or null", "question": "short question or null"}}"""

QUICK_TOOL_PROMPT = """Pick ONE tool for this question.

Context: {summary}
Query: "{query}"

Tools:
- weather_fetcher {{"city": str}}
- fx_checker {{"base_currency": str, "target_currency": str}}
- rag_retriever {{"query": str, "k": 3}}
- flight_searcher {{"origin": str, "destination": str}}

JSON: {{"tool": "name", "input": {{...}}}}"""

ROUTE_PROMPT = """Pick tools for: "{query}"

Context: dest={destination_country} origin={origin_country} fx={origin_currency}->{destination_currency}

Available (pick any subset):
- rag_retriever {{"query": str, "k": 5}}
- weather_fetcher {{"city": str}}
- flight_searcher {{"origin": str, "destination": str}}
- fx_checker {{"base_currency": str, "target_currency": str}}
- ml_predictor {{8 floats 0-1: active_movement, relaxation, cultural_interest, cost_sensitivity, luxury_preference, family_friendliness, nature_orientation, social_group}}

JSON: {{"tools": [{{"name": "...", "input": {{...}}}}]}}"""

INFO_SYNTHESIS_PROMPT = """You are a helpful travel assistant. Answer the user's question directly and briefly using the tool result below.

TOOL RESULT:
{tool_results}

USER'S QUESTION: {query}

Give a friendly, direct answer in 2-4 sentences. Do NOT write a travel plan."""

SYNTHESIS_PROMPT = """You are an expert travel planner. You have gathered information from
multiple tools. Now synthesize a comprehensive, personalized travel plan.

RULES:
- GENUINE SYNTHESIS: Compare and combine tool outputs. Do NOT just concatenate them.
- If tools conflict (e.g. RAG says "budget-friendly" but weather suggests peak season prices), explain the tension.
- If a tool returned an error, acknowledge the limitation and plan around it.
- Be specific — use actual numbers from tool results (temperatures, prices, exchange rates).
- Structure your response with clear sections.
- Keep the tone friendly and helpful.
- If currency info is missing for the user's currency, note it and use USD as reference.

TOOL RESULTS:
{tool_results}

USER'S QUERY: {query}

DESTINATION: {destination}

Write a coherent travel plan:"""
