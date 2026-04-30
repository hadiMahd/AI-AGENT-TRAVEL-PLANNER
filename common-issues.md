# Common Issues & Optimizations

Latency and design problems found during agent workflow development and how
they were addressed.

---

## Model selection for mechanical tasks

**Problem.** The cheap LLM (DeepSeek-V3.2-1) had high first-token latency
on short prompts. A single rewrite step was taking >25s for a 56-token
input. This became the bottleneck for the entire agent pipeline.

**Why it happened.** Model selection wasn't tuned to the actual workload.
Mechanical extraction tasks (query rewrite, arg parsing, tool selection)
don't need a large model — they need fast time-to-first-token and
predictable structured output. Choosing a model optimized for creative
reasoning work was overkill.

**Solution.** Swapped the cheap model to `gpt-4o-mini`, which has much
better latency on mechanical tasks. Subsequent probe runs completed the
rewrite + routing round-trip in low single-digit seconds.

---

## Configuration and runtime binding

**Problem.** After changing the model in `.env`, the running server still
used the old model. The LLM client is constructed once at startup and
cached. Settings are also `@lru_cache`'d, freezing values for the process
lifetime.

**Why it happened.** FastAPI lifespan pattern (construct once, store on
`app.state`) is correct for resource efficiency, but it means runtime
config changes require a server restart. This wasn't obvious until the
first model swap.

**Solution.** Document that env changes require restart. Verify by checking
startup logs for the `Creating cheap LLM client — model=...` line. Also:
don't trust historical LangSmith traces — verify behavior on fresh runs
after config changes.

---

## Prompt design for information retrieval

**Problem.** Query rewriting was producing tautologies ("Plan a trip,
focusing on destination details and travel preferences") that didn't help
RAG retrieval. The prompt was too generic.

**Why it happened.** Prompts that say "extract X and Y" without specific
examples of what matters leave the model to guess. For travel planning,
specifics like visa requirements, stopover hubs, budget tier, and trip
duration are retrieval-relevant; vague rewrites miss them.

**Solution.** Explicit prompt guidance: surface visa/entry needs, likely
stopover hubs (especially for routes without direct flights), duration,
budget tier. This increased RAG hit quality and reduced the need for
follow-up searches.

---

## Tool routing and over-execution

**Problem.** For full trip planning, the agent sometimes skipped
`weather_fetcher` (destination weather is always relevant) and
`ml_predictor` (personalizes recommendations). Each miss meant a slower,
less personalized synthesis downstream.

**Why it happened.** ROUTE_PROMPT said "pick any subset" with no defaults.
The cheap LLM was conservative, picking only the tools it was confident
about.

**Solution.** Make useful tools default-included for plan intent: always
include `weather_fetcher` (destination city) and `ml_predictor` unless
the user opts out. This ensures more complete information reaches
synthesis with fewer edge cases.

---

## Silent failures and validation

**Problem.** `ml_predictor` inputs were sometimes malformed (positional
arrays instead of keyed dicts), but the executor had a silent fallback
that returned all-0.5 features. The prediction was wrong but logged as
success. This went unnoticed until trace inspection.

**Why it happened.** Two layers of silence: (1) ambiguous prompt spec
allowed the LLM to pick either format, (2) executor's `.get(..., 0.5)`
defaults meant invalid input produced valid-looking but wrong output.

**Solution.** Strict input validation: enforce dict shape in the prompt,
validate with Pydantic in the executor. If the LLM produces a list, the
executor now rejects it with a clear error log instead of silently
degrading. This makes regressions detectable.

---

## Stateful context across turns

**Problem.** When a user said "change the destination to Tokyo" mid-conversation,
the agent lost all prior preferences (budget, family, adventure level) and
defaulted all features to 0.5. This required re-stating preferences.

**Why it happened.** The routing step only saw the current query ("change
to Tokyo") with no access to conversation history. Prior signals from
earlier turns were invisible.

**Solution.** Include recent conversation history in the routing prompt so
the LLM can extract preference signals from earlier turns and carry them
forward. This reduces user friction on multi-turn interactions.

---

## Prompt-executor separation of concerns

**Problem.** Several issues (RAG depth, tool defaults, input shape) were
underspecified in prompts, leaving the executor to fill gaps with silent
defaults.

**Why it happened.** Prompts express intent (what the LLM *should* do),
but executors enforce it (what will actually happen). When a policy
matters (e.g., "top-3 results only"), encoding it only in the prompt
leaves it vulnerable to drift.

**Solution.** For any constraint that's non-negotiable, enforce it in the
executor too. Example: RAG caps `k` at 3 regardless of what the prompt or
LLM requests. This makes the contract explicit and drift-proof.

---

## Takeaways

- **Latency is often model fit, not code.** Picking the right tool for the
  job (fast model for mechanical tasks) outweighed any prompt optimization.
- **Silent failures are regressions you can't see.** Validation and
  explicit error paths make problems detectable.
- **Context is cheap; rework is expensive.** Including conversation history
  in routing prompts costs a few tokens but saves rework across turns.
- **Constraints belong in two places.** Prompts express what *should*
  happen; executors enforce what *must* happen. Both matter.
