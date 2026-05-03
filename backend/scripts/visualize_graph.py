"""
Generate the agent graph visualization as a PNG file.

Renders the full LangGraph agent workflow including all conditional edges
via Mermaid.ink API (free, no dependencies needed).

Usage:  cd backend && uv run python scripts/visualize_graph.py
Output: project/agent_graph.png
"""
import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import build_graph

# ── Build graph and get Mermaid markup ──────────────────

fake = MagicMock()
g = build_graph(fake, fake, fake, fake, fake)
c = g.compile()
mermaid = c.get_graph().draw_mermaid()

# ── Patch: LangGraph's Mermaid renderer skips conditional edges.
#      Inject the real conditional routing as labeled arrows.
# ───────────────────────────────────────────────────────────

lines = mermaid.split("\n")
insert_at = max(i for i, l in enumerate(lines) if "-->" in l)

conditional_edges = [
    '\tclassify_intent -->|"intent=casual"| casual_reply;',
    '\tclassify_intent -->|"intent=travel"| check_context;',
    '\tcheck_context -->|"needs user input & attempts<3"| __end__;',
    '\tcheck_context -->|"ready or attempts>=3"| route_and_run_tools;',
    '\troute_and_run_tools --> __end__;',
]

for edge in conditional_edges:
    lines.insert(insert_at + 1, edge)
    insert_at += 1

mermaid_fixed = "\n".join(lines)

# ── Render via Mermaid.ink API ───────────────────────────

encoded = base64.urlsafe_b64encode(mermaid_fixed.encode("utf-8")).decode("ascii").rstrip("=")

url = f"https://mermaid.ink/img/{encoded}?type=png"

async def download():
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

import asyncio
png_bytes = asyncio.run(download())

out = Path(__file__).resolve().parent.parent.parent / "agent_graph.png"
out.write_bytes(png_bytes)
print(f"Saved {len(png_bytes)} bytes → {out}")
