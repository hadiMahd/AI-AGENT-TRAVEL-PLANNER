# Agent Flow Documentation

This document traces every path a user query can take — from the HTTP request through the LangGraph nodes to the SSE response stream.

---

## Architecture Overview

```
Frontend (React)
    │  POST /agent/chat  {query, origin_country, history}
    ▼
FastAPI  →  StreamingResponse (SSE)
    │
    ▼
LangGraph StateGraph
    │
    ├──[intent=casual]──► casual_reply ──► END
    │
    ├──[intent=info]────► quick_info_run ──► END
    │
    └──[intent=travel]──► rewrite_query ──► check_inputs
                                                 │
                                    [needs input]▼     [has origin]▼
                                               END    route_and_run_tools ──► END
```

**Models:**
- Cheap model (DeepSeek-V3.2-1): classify, casual reply, rewrite, check inputs, route, quick info routing
- Strong model (Kimi-K2.6-1): synthesis (streamed token by token)

---

## Scenario 1 — Casual Query

**Example:** `"hi"`, `"hello bro"`, `"thanks"`, `"what can you do?"`

### Step-by-step

```
1. POST /agent/chat
   body: { query: "hello bro", history: [...] }

2. SSE → event: thinking
   data: { message: "Analyzing your query..." }

3. LangGraph starts
   Entry node: classify_intent

4. NODE: classify_intent  (cheap LLM, timeout 60s)
   Prompt: CLASSIFY_INTENT_PROMPT
     - Injected: last 6 conversation turns (300 char each)
     - Query: "hello bro"
   LLM responds: { "intent": "casual" }
   State update: intent = "casual"
   Edge: → casual_reply

5. NODE: casual_reply  (cheap LLM, timeout 60s)
   Prompt: CASUAL_REPLY_PROMPT
     - Query: "hello bro"
   LLM responds: "Hey there! I'm your travel planner. Tell me where you'd like to go and I'll handle the rest!"
   State update: final_response = "Hey there! ..."
                 tool_logs = []

6. Graph ends (END edge from casual_reply)

7. Router checks result.intent == "casual"
   → Skips synthesis streaming entirely

8. SSE → event: final
   data: {
     response: "Hey there! I'm your travel planner...",
     tool_logs: [],
     prompt_tokens: 0,
     completion_tokens: 0
   }

9. Background task: _persist_run (AgentRun saved to DB, no tool logs)
```

**SSE events emitted:** `thinking` → `final`

**No email prompt.** (Email ask is only sent after travel plans.)

---

## Scenario 2 — Info Query

**Example:** `"what's the weather in Bali?"`, `"how much is 1 USD in EUR?"`, `"tell me about Tokyo"`

### Step-by-step

```
1. POST /agent/chat
   body: { query: "what's the weather in Bali?", history: [...] }

2. SSE → event: thinking
   data: { message: "Analyzing your query..." }

3. LangGraph starts
   Entry node: classify_intent

4. NODE: classify_intent  (cheap LLM, timeout 60s)
   Prompt: CLASSIFY_INTENT_PROMPT
     - Injected: conversation history
     - Query: "what's the weather in Bali?"
   LLM responds: { "intent": "info" }
   State update: intent = "info"
   Edge: → quick_info_run

5. NODE: quick_info_run  (cheap LLM for routing, timeout 60s)
   Prompt: QUICK_TOOL_PROMPT
     - Conversation history
     - Query: "what's the weather in Bali?"
     - Tool list: weather_fetcher, fx_checker, rag_retriever, flight_searcher
   LLM responds: { "tool": "weather_fetcher", "input": { "city": "Bali" } }

   Tool validation: "weather_fetcher" ∈ ALLOWED_TOOLS ✓

   SSE → event: tool_start
   data: { tool: "weather_fetcher", input: { city: "Bali" } }

   Execute: get_weather_by_city("Bali")
     - Checks 10-min TTL cache
     - GET https://wttr.in/Bali?format=j1
     - Retries: up to 3× with exponential backoff (1s, 2s, 4s)
     - Extracts: temp_c, feels_like_c, condition, humidity, wind_kph, location

   SSE → event: tool_result
   data: { tool: "weather_fetcher", output: "{\"temp_c\": 28, ...}", latency_ms: 420 }

   State update: tool_results = { "weather_fetcher": "{...}" }
                 tool_logs = [{ tool_name, input_payload, output_payload, status, latency_ms }]

6. Graph ends (END edge from quick_info_run)

7. Router: result.intent == "info"
   Synthesis prompt: INFO_SYNTHESIS_PROMPT
     "Answer the user's question directly and briefly using the tool result.
      Give a friendly, direct answer in 2-4 sentences. Do NOT write a travel plan."

8. Strong LLM streams tokens (astream)
   Each chunk:
   SSE → event: token
   data: { text: "The weather in Bali right now..." }

9. SSE → event: final
   data: {
     response: "The weather in Bali right now is 28°C (feels like 30°C)...",
     tool_logs: [...],
     prompt_tokens: 245,
     completion_tokens: 38
   }

   NOTE: No ask_email event — info queries don't generate a travel plan.

10. Background task: _persist_run (AgentRun + 1 ToolLog saved to DB)
```

**SSE events emitted:** `thinking` → `tool_start` → `tool_result` → `token` × N → `final`

**Tool fallback:** If cheap LLM routing times out → falls back to `rag_retriever`.
If tool name not in allowlist → falls back to `rag_retriever`.

---

## Scenario 3A — Travel Query (Origin Known)

**Example:** `"plan a trip to Japan from Lebanon"`, `"I want to visit Bali from Egypt"`

### Step-by-step

```
1. POST /agent/chat
   body: {
     query: "plan a trip to Japan from Lebanon",
     origin_country: "Lebanon",   ← optional, from previous exchange
     history: [...]
   }

2. SSE → event: thinking
   data: { message: "Analyzing your query..." }

3. LangGraph starts — initial state includes:
   query, origin_country, history, intent="travel" (default before classify)

4. NODE: classify_intent  (cheap LLM, timeout 60s)
   Prompt: CLASSIFY_INTENT_PROMPT
     - History injected
     - Query: "plan a trip to Japan from Lebanon"
   LLM responds: { "intent": "plan" }
   Normalized: "plan" → "travel"
   State update: intent = "travel"
   Edge: → rewrite_query

5. NODE: rewrite_query  (cheap LLM, timeout 90s)
   Prompt: REWRITE_PROMPT
     - History: last 6 turns (for context like destination changes)
     - Query: "plan a trip to Japan from Lebanon"
   LLM responds: "Travel to Japan from Lebanon — cultural experiences, temples, Tokyo city life."
   State update: rewritten_query = "Travel to Japan..."
   Token counts accumulated.

6. NODE: check_inputs  (cheap LLM + RAG, timeout 90s)
   a) Embed rewritten_query via OpenAI embeddings
   b) pgvector similarity_search(k=1) → finds closest destination doc
      Returns: { country: "Japan", latitude: 36.2, longitude: 138.25, ... }
   c) get_currency_for_country("Japan") → "JPY"
   d) Prompt: CHECK_INPUTS_PROMPT
        - History injected
        - Query: original query
        - destination_info: JSON from RAG hit
      LLM responds: { "has_origin": true, "origin_country": "Lebanon", "question": null }
   e) get_currency_for_country("Lebanon") → "LBP"
   State update:
     destination_country = "Japan"
     destination_lat = 36.2,  destination_long = 138.25
     destination_currency = "JPY"
     origin_country = "Lebanon"
     origin_currency = "LBP"
     needs_user_input = false
   Edge: → route_and_run_tools

7. NODE: route_and_run_tools  (cheap LLM, timeout 90s)
   Prompt: ROUTE_PROMPT
     - Query, destination country + lat/lng, origin/dest currencies
   LLM responds (example):
   {
     "tools": [
       { "name": "rag_retriever",   "input": { "query": "Japan travel tips", "k": 5 } },
       { "name": "weather_fetcher", "input": { "city": "Japan" } },
       { "name": "flight_searcher", "input": { "origin": "Beirut", "destination": "Tokyo" } },
       { "name": "fx_checker",      "input": { "base_currency": "LBP", "target_currency": "JPY" } }
     ]
   }

   Each tool runs sequentially:

   a) rag_retriever
      SSE → tool_start  { tool: "rag_retriever", input: {...} }
      embed_query → similarity_search(k=5) → top-5 Japan documents
      SSE → tool_result { tool: "rag_retriever", output: "[{...}]", latency_ms }

   b) weather_fetcher
      SSE → tool_start  { tool: "weather_fetcher", input: { city: "Japan" } }
      get_weather_by_city("Japan") → wttr.in
      SSE → tool_result { tool: "weather_fetcher", output: "{temp_c:15,...}", latency_ms }

   c) flight_searcher
      SSE → tool_start  { tool: "flight_searcher", input: { origin: "Beirut", destination: "Tokyo" } }
      search_flights("Beirut", "Tokyo")
      SSE → tool_result { tool: "flight_searcher", output: "[{...}]", latency_ms }

   d) fx_checker
      SSE → tool_start  { tool: "fx_checker", input: { base_currency: "LBP", target_currency: "JPY" } }
      get_exchange_rate("LBP", "JPY")
      SSE → tool_result { tool: "fx_checker", output: "{rate:0.0024,...}", latency_ms }

   State update: tool_results = { rag_retriever: ..., weather_fetcher: ..., ... }
                 tool_logs = [4 log entries]

8. Graph ends (END edge from route_and_run_tools)

9. Router: builds SYNTHESIS_PROMPT
   - Tool results truncated to MAX_TOOL_CHARS (400) each
   - Includes: destination, original query, all tool outputs
   - Instruction: "Genuine synthesis — compare and combine, use actual numbers"

10. Strong LLM streams tokens (astream)
    Each chunk:
    SSE → event: token
    data: { text: "## Your Japan Travel Plan\n\n..." }

11. SSE → event: final
    data: {
      response: "## Your Japan Travel Plan\n\n**Best Time...",
      tool_logs: [4 entries],
      prompt_tokens: 1240,
      completion_tokens: 892
    }

12. SSE → event: ask_email
    data: {
      question: "Would you like me to send this travel plan to your email?",
      plan: "## Your Japan Travel Plan...",
      destination: "Japan"
    }
    (Frontend shows Yes / No buttons)

13. Background task: _persist_run
    - AgentRun row (query, response, tokens, cost, status="completed")
    - 4 ToolLog rows (one per tool)
```

**SSE events emitted:** `thinking` → (`tool_start` + `tool_result`) × 4 → `token` × N → `final` → `ask_email`

---

## Scenario 3B — Travel Query (Origin Missing)

**Example:** `"plan a trip to Bali"` (no origin mentioned, no history context)

```
Steps 1–5 same as Scenario 3A.

6. NODE: check_inputs
   RAG finds destination = "Bali" (Indonesia), currency = "IDR"
   Prompt asks LLM: does the query mention where user is FROM?
   LLM responds: { "has_origin": false, "question": "Where are you traveling from? I need this to find flights and currency info." }

   State update:
     destination_country = "Indonesia"
     needs_user_input = true
     user_question = "Where are you traveling from?..."
   Edge: → END  (conditional _after_check)

7. Graph ends early.

8. Router: result.get("needs_user_input") == True

9. SSE → event: needs_input
   data: { question: "Where are you traveling from? I need this to find flights and currency info." }

10. Frontend displays question as agent message.
    User types answer: "Lebanon"

11. New POST /agent/chat request:
    {
      query: "Lebanon",
      origin_country: null,
      history: [
        { role: "user",      content: "plan a trip to Bali" },
        { role: "assistant", content: "Where are you traveling from?..." },
        { role: "user",      content: "Lebanon" }
      ]
    }

12. classify_intent sees history → "Lebanon" in context of travel planning
    → classifies as "travel"

13. check_inputs re-runs — history now contains "Lebanon"
    LLM responds: { "has_origin": true, "origin_country": "Lebanon" }
    → Proceeds to route_and_run_tools → full plan flow.
```

**SSE events emitted:** `thinking` → `needs_input`  (first request)
Then: `thinking` → tools → tokens → `final` → `ask_email` (second request)

---

## Scenario 4 — Email Delivery

After receiving `ask_email` event, two paths:

### User clicks Yes

```
Frontend shows email input bar.
User types: "user@example.com" → clicks Send

POST /agent/send-email
body: { email: "user@example.com", plan: "## Your Japan...", destination: "Japan" }

send_plan_email():
  1. _markdown_to_html(plan) — regex converts ## → <h2>, ** → <b>, - → <li>, etc.
  2. asyncio.to_thread(resend.Emails.send, {...})
     From: "Travel Planner <onboarding@resend.dev>"
     To: "user@example.com"
     Subject: "Your Japan Travel Plan"
     HTML: styled email with plan content

Response: { "message": "Plan sent to user@example.com" }
Frontend: shows "Plan sent to user@example.com" confirmation.
```

### User clicks No / Later asks via chat

```
Frontend stores lastPlan in state.

If user later types "send it to my email" / "email me the plan" / etc.:
  Frontend keyword-intercepts (checks for email/send/mail in message + lastPlan exists)
  → Shows ask_email UI again using stored lastPlan
  → Does NOT send new request to /agent/chat
  → User enters email → POST /agent/send-email as above
```

---

## Timeout & Fallback Behavior

| Node | Timeout | Fallback |
|------|---------|---------|
| classify_intent | 60s | intent = "travel" |
| casual_reply | 60s | hardcoded greeting |
| quick_info_run routing | 60s | rag_retriever |
| rewrite_query | 90s | original query unchanged |
| check_inputs | 90s | needs_user_input = true, ask for origin |
| route_and_run_tools | 90s | rag_retriever + weather only |
| synthesis (strong LLM) | streamed — no timeout | error message if stream fails |

Tool errors (network, API down) are caught per-tool, stored as `[TOOL ERROR] ...` in `tool_results`, and the synthesis prompt receives the error text — the strong LLM acknowledges the limitation and plans around it.

---

## State Object (AgentState)

Flows between all nodes as a TypedDict:

```python
{
  "query":               str,          # original user message
  "intent":              str,          # "casual" | "info" | "travel"
  "history":             list[dict],   # [{ role, content }, ...]
  "rewritten_query":     str,          # optimized for RAG (travel only)
  "origin_country":      str | None,
  "destination_country": str | None,
  "destination_lat":     float | None,
  "destination_long":    float | None,
  "destination_currency":str | None,
  "origin_currency":     str | None,
  "needs_user_input":    bool,
  "user_question":       str | None,
  "selected_tools":      list[str],
  "tool_results":        dict[str, str],   # name → raw string output
  "tool_logs":           list[dict],       # for frontend ToolLogPanel
  "final_response":      str,
  "prompt_tokens":       int,              # accumulated across nodes
  "completion_tokens":   int
}
```

---

## Tool Allowlist

Defined in `backend/tools/__init__.py`. Any tool name returned by the LLM that is not in this list is rejected before execution.

| Tool name | Service | Input |
|-----------|---------|-------|
| `rag_retriever` | pgvector similarity search | `query: str, k: int` |
| `weather_fetcher` | wttr.in (city or lat/long) | `city: str` OR `latitude, longitude` |
| `flight_searcher` | flights service | `origin: str, destination: str` |
| `fx_checker` | exchange rate service | `base_currency: str, target_currency: str` |
| `ml_predictor` | local ML model (travel style) | 8 float features |
