"""
Final structured extraction using the data-derived super-lens schema.

Like Task 1's extractor.py — but instead of a hardcoded schema, the prompt is
built dynamically from the super_lenses.json discovered in Task 2.

Uses Qwen via Ollama (free, local). Output goes to outputs/final/.
"""
from __future__ import annotations
import json
import os
import re
import time
from collections import defaultdict

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:7b"
MAX_TOKENS = 1536
MAX_RETRIES = 3
RETRY_DELAY = 3.0


def _build_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def _build_system_prompt(super_lenses: list[dict], entity_clusters: list[dict]) -> str:
    """
    Dynamically build the extraction prompt from the data-derived super-lenses.
    This is the key difference from Task 1 — schema comes from the data, not from us.
    """
    # Build entity level examples for context
    level_examples: dict[int, list[str]] = defaultdict(list)
    for ec in sorted(entity_clusters, key=lambda x: x.get("level", 3)):
        lvl = ec.get("level", 3)
        if len(level_examples[lvl]) < 3:
            level_examples[lvl].append(ec["canonical"])

    level_desc = "\n".join(
        f"  Level {lvl}: abstract → e.g. {', '.join(examples)}"
        for lvl, examples in sorted(level_examples.items())
        if examples
    )

    # Build lens descriptions
    lens_lines = []
    for sl in super_lenses[:20]:  # top 20 by coverage
        edge_examples = ", ".join(f'"{e}"' for e in sl["edge_types"][:4])
        pair_example = ""
        if sl["top_entity_pairs"]:
            p = sl["top_entity_pairs"][0]
            pair_example = f' (e.g. "{p["head"]}" → "{p["tail"]}")'
        lens_lines.append(
            f'  - "{sl["name"]}" [id={sl["super_lens_id"]}]: relations like {edge_examples}{pair_example}'
        )
    lenses_desc = "\n".join(lens_lines)

    max_level = max((ec.get("level", 3) for ec in entity_clusters), default=4)

    return f"""You are a cultural knowledge graph builder.

Given a cultural text, extract structured knowledge using ONLY the lenses below.
These lenses were discovered directly from real cultural datasets — they represent
the natural dimensions of cultural knowledge.

## Abstraction Levels (frequency-derived, 1=most abstract, {max_level}=most specific):
{level_desc}

## Available Lenses (use only these):
{lenses_desc}

## Output Format — return ONLY valid JSON:
{{
  "extractions": [
    {{
      "lens_id": <int>,
      "lens_name": "<name>",
      "nodes": [
        {{"label": "<1-5 words, lowercase>", "level": <int>, "cultural_group": "<group>"}}
      ],
      "edges": [
        {{"head": "<node label>", "relation": "<relation phrase>", "tail": "<node label>"}}
      ]
    }}
  ]
}}

Rules:
- Only include lenses that genuinely apply to the text
- 1-3 lenses per entry is normal; 0 is valid if no lens fits
- Node labels: short, lowercase, in English
- Edge relation must be one of the relation phrases listed for that lens
- head and tail must match a node label exactly"""


def _parse_extractions(content: str) -> list[dict]:
    content = content.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        content = match.group(1).strip()
    match = re.search(r'\{[\s\S]*\}', content)
    if match:
        content = match.group(0)
    try:
        data = json.loads(content)
        return data.get("extractions", [])
    except json.JSONDecodeError:
        return []


def extract_entry(
    client: OpenAI,
    entry: dict,
    system_prompt: str,
    verbose: bool = False,
) -> list[dict]:
    text = entry.get("text", "").strip()
    if not text:
        return []

    cultural_group = entry.get("cultural_group", "")
    source = entry.get("source_dataset", "")
    meta = entry.get("metadata", {})

    parts = [f"Cultural group: {cultural_group}", f"Source: {source}"]
    for k, v in meta.items():
        if v and k not in ("id", "answers"):
            parts.append(f"{k}: {v}")
    parts.append(f"\nText:\n{text}")
    user_msg = "\n".join(parts)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )
            raw = response.choices[0].message.content or ""
            extractions = _parse_extractions(raw)
            if verbose:
                lens_names = [e.get("lens_name", "?") for e in extractions]
                print(f"    → {len(extractions)} lenses: {lens_names}")
            return extractions
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"[final_extractor] Failed: {e}")
                return []
    return []


def _load_processed_ids(output_path: str) -> set[int]:
    seen = set()
    if not os.path.exists(output_path):
        return seen
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line.strip())["entry_id"])
            except Exception:
                pass
    return seen


def extract_batch(
    entries: list[dict],
    super_lenses: list[dict],
    entity_clusters: list[dict],
    output_path: str,
    resume: bool = True,
    verbose: bool = False,
    progress_every: int = 10,
) -> str:
    """
    Run structured extraction on all entries using the data-derived schema.
    Streams results to output_path (JSONL). Resumable.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    processed = _load_processed_ids(output_path) if resume else set()
    if processed:
        print(f"[final_extractor] Resuming — {len(processed)} entries already done.")

    system_prompt = _build_system_prompt(super_lenses, entity_clusters)
    client = _build_client()

    with open(output_path, "a", encoding="utf-8") as out:
        for i, entry in enumerate(entries):
            if i in processed:
                continue
            if progress_every and i % progress_every == 0:
                print(f"[final_extractor] {i}/{len(entries)} entries ...")

            extractions = extract_entry(client, entry, system_prompt, verbose=verbose)

            record = {
                "entry_id": i,
                "source_dataset": entry.get("source_dataset", ""),
                "cultural_group": entry.get("cultural_group", ""),
                "text": entry.get("text", "")[:300],
                "extractions": extractions,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    total = sum(
        len(json.loads(l)["extractions"])
        for l in open(output_path, encoding="utf-8")
        if l.strip()
    )
    print(f"[final_extractor] Done → {output_path} ({total} total extractions)")
    return output_path


def load_extractions(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
