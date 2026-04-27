"""
Claude API extractor with prompt caching.

For each dataset entry, sends a structured prompt to Claude and parses the
returned JSON into raw node/edge dicts. The system prompt is sent with
cache_control so it is reused across calls (reduces cost and latency).
"""
from __future__ import annotations
import json
import os
import re
import time
from typing import Any

import anthropic

MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for bulk extraction
MAX_TOKENS = 1024
MAX_RETRIES = 3
RETRY_DELAY = 2.0   # seconds between retries on rate limit


SYSTEM_PROMPT = """You are a cultural knowledge graph extraction assistant.

Given a text snippet from a cultural dataset, extract structured cultural knowledge in JSON.

## Graph Hierarchy (5 levels, root → leaf):
- Level 0: "human" (universal root, always exists)
- Level 1: Cultural Domain — one of: religion, family, gender, food, community, language, tradition, education, economy, politics, art, health, nature, identity, unknown
- Level 2: Cultural Group — e.g. "arab", "indian", "korean", "turkish", "general"
- Level 3: Cultural Practice / Belief / Norm — a short phrase describing what a group does or believes
- Level 4: Specific Instance / Cue — a concrete action, object, or artifact

## Edge Relation Types (use these exactly):
influences, balances, honors, contextualizes, restricts, manifests_in, varies_by, originates_from

## Output Format:
Return ONLY valid JSON with this structure (no markdown, no explanation):
{
  "nodes": [
    {"label": "str", "level": int, "cultural_group": "str", "domain": "str"}
  ],
  "edges": [
    {"source_label": "str", "target_label": "str", "relation_type": "str"}
  ]
}

## Rules:
- Always include a Level 0 node: {"label": "human", "level": 0, "cultural_group": "universal", "domain": "identity"}
- Include 2–8 nodes per entry (do not over-extract)
- Only use the 8 relation types listed above
- Edge source and target labels must match node labels exactly
- Labels should be short (1–5 words), lowercase, in English
- If no meaningful cultural knowledge can be extracted, return {"nodes": [], "edges": []}
"""


def _build_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running: export ANTHROPIC_API_KEY=your_key"
        )
    return anthropic.Anthropic(api_key=api_key)


def _parse_response(content: str) -> dict[str, list]:
    """Extract JSON from Claude's response, handling possible markdown fences."""
    content = content.strip()
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        content = match.group(1).strip()
    try:
        data = json.loads(content)
        nodes = data.get("nodes") or []
        edges = data.get("edges") or []
        return {"nodes": nodes, "edges": edges}
    except json.JSONDecodeError:
        return {"nodes": [], "edges": []}


def extract_entry(
    client: anthropic.Anthropic,
    entry: dict,
    verbose: bool = False,
) -> dict[str, list]:
    """
    Call Claude for a single dataset entry. Returns {"nodes": [...], "edges": [...]}.
    Each node/edge dict carries source_dataset from the entry.
    """
    text = entry.get("text", "").strip()
    if not text:
        return {"nodes": [], "edges": []}

    cultural_group = entry.get("cultural_group", "general")
    source_dataset = entry.get("source_dataset", "")
    meta = entry.get("metadata", {})

    user_prompt_parts = [f"Cultural group: {cultural_group}"]
    if source_dataset:
        user_prompt_parts.append(f"Source dataset: {source_dataset}")
    for k, v in meta.items():
        if v and k not in ("id",):
            user_prompt_parts.append(f"{k}: {v}")
    user_prompt_parts.append(f"\nText:\n{text}")
    user_prompt = "\n".join(user_prompt_parts)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},  # prompt caching
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text
            result = _parse_response(raw)

            # Attach source_dataset to each node/edge
            for node in result["nodes"]:
                node.setdefault("source_dataset", source_dataset)
            for edge in result["edges"]:
                edge.setdefault("source_dataset", source_dataset)

            if verbose:
                print(f"  → {len(result['nodes'])} nodes, {len(result['edges'])} edges")
            return result

        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"[extractor] Rate limit hit after {MAX_RETRIES} retries; skipping entry.")
                return {"nodes": [], "edges": []}
        except anthropic.APIError as e:
            print(f"[extractor] API error: {e}; skipping entry.")
            return {"nodes": [], "edges": []}


def extract_batch(
    entries: list[dict],
    verbose: bool = False,
    progress_every: int = 10,
) -> tuple[list[dict], list[dict]]:
    """
    Extract nodes and edges from a list of entries.
    Returns (all_raw_nodes, all_raw_edges).
    """
    client = _build_client()
    all_nodes: list[dict] = []
    all_edges: list[dict] = []

    for i, entry in enumerate(entries):
        if progress_every and i % progress_every == 0:
            print(f"[extractor] {i}/{len(entries)} entries processed ...")
        result = extract_entry(client, entry, verbose=verbose)
        all_nodes.extend(result["nodes"])
        all_edges.extend(result["edges"])

    print(f"[extractor] Done. {len(all_nodes)} raw nodes, {len(all_edges)} raw edges.")
    return all_nodes, all_edges
