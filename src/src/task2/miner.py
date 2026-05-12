"""
Phase 1: Unconstrained triple mining via local Ollama (qwen2.5:7b).

No schema is given to the model. It extracts whatever entities and relations
it naturally sees in the text. Results stream to a JSONL file line-by-line
so the run can be safely interrupted and resumed.

Usage (called from induction_main.py):
    mine_batch(entries, "outputs/induced/raw_triples.jsonl", resume=True)
"""
from __future__ import annotations
import json
import os
import re
import time
from typing import Iterator

from dotenv import load_dotenv
from llm_backend import LLMBackend

load_dotenv()

MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a cultural knowledge extractor.

Given a text snippet about cultural practices, beliefs, or norms, extract knowledge as triples.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{"triples": [{"head": "...", "relation": "...", "tail": "..."}]}

Rules:
- head and tail: any entity, concept, group, value, emotion, action, object, or place
- relation: a natural descriptive phrase — use whatever words best describe the connection
  DO NOT use generic terms like "is", "has", "related to" — be specific and culturally meaningful
- Extract 4-10 triples per text
- Labels should be short (1-6 words) and in English
- Capture: who does what, what values are expressed, what emotions are involved,
  what varies by region/group, what is prohibited/encouraged, what is passed down and ..."""


def _parse_triples(content: str) -> list[dict]:
    content = content.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        content = match.group(1).strip()
    # Sometimes model wraps in extra text — find the JSON object
    match = re.search(r'\{[\s\S]*\}', content)
    if match:
        content = match.group(0)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            triples = data
        else:
            triples = data.get("triples", [])
        # Validate each triple has required keys
        valid = []
        for t in triples:
            if isinstance(t, dict) and t.get("head") and t.get("relation") and t.get("tail"):
                valid.append({
                    "head": str(t["head"]).strip().lower(),
                    "relation": str(t["relation"]).strip().lower(),
                    "tail": str(t["tail"]).strip().lower(),
                })
        return valid
    except (json.JSONDecodeError, KeyError):
        return []


def mine_entry(backend: LLMBackend, entry: dict, verbose: bool = False) -> list[dict]:
    """Extract free-form triples from a single dataset entry."""
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

    raw = backend.generate(SYSTEM_PROMPT, user_msg)
    triples = _parse_triples(raw)
    if verbose:
        print(f"    → {len(triples)} triples")
    return triples


def _load_processed_ids(output_path: str) -> set[int]:
    """Return set of entry_ids already written to the output file."""
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


def mine_batch(
    entries: list[dict],
    output_path: str,
    resume: bool = True,
    verbose: bool = False,
    progress_every: int = 10,
    backend: str = "ollama",
) -> str:
    """
    Mine all entries and stream results to output_path (JSONL).
    If resume=True, skips entries whose entry_id is already in the file.
    backend: 'ollama' (Mac) or 'hf' (Narval GPU).
    Returns output_path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    processed = _load_processed_ids(output_path) if resume else set()

    if processed:
        print(f"[miner] Resuming — {len(processed)} entries already done, skipping.")

    llm = LLMBackend(backend=backend)
    total_triples = 0

    with open(output_path, "a", encoding="utf-8") as out:
        for i, entry in enumerate(entries):
            entry_id = i
            if entry_id in processed:
                continue

            if progress_every and i % progress_every == 0:
                print(f"[miner] {i}/{len(entries)} entries ...")

            triples = mine_entry(llm, entry, verbose=verbose)
            total_triples += len(triples)

            record = {
                "entry_id": entry_id,
                "source_dataset": entry.get("source_dataset", ""),
                "cultural_group": entry.get("cultural_group", ""),
                "text": entry.get("text", "")[:300],
                "triples": triples,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    print(f"[miner] Done. {len(entries) - len(processed)} new entries, {total_triples} total triples → {output_path}")
    return output_path


def load_triples(path: str) -> list[dict]:
    """Load all mined records from a JSONL file."""
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
