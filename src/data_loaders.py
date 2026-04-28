"""
One loader per dataset → yields normalized dicts:
  {"text": str, "cultural_group": str, "source_dataset": str, "metadata": dict}

CANDLE requires a local download (not on HuggingFace).
Ko-PIQA is not yet released; its loader raises NotImplementedError.
All others are loaded via datasets.load_dataset().
"""
from __future__ import annotations
import os
import json
from typing import Iterator

DATASET_NAMES = ["candle", "arabculture", "diwali", "culturebank", "blend", "kopiqa"]


def _normalize(text: str, cultural_group: str, source: str, metadata: dict | None = None) -> dict:
    return {
        "text": (text or "").strip(),
        "cultural_group": (cultural_group or "general").strip().lower(),
        "source_dataset": source,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# CANDLE  (local download required)
# GitHub: https://github.com/cultural-csk/candle
# Download the assertions TSV/JSON from the CANDLE project and point
# --candle-path at the directory.
# ---------------------------------------------------------------------------
def load_candle(data_path: str, limit: int = 0) -> Iterator[dict]:
    """
    Expects a directory containing CANDLE's assertion files.
    Supported formats: .jsonl or .tsv (subject, assertion, ...).
    """
    if not data_path or not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"CANDLE: provide a valid --candle-path directory. "
            f"Download from https://github.com/cultural-csk/candle"
        )
    count = 0
    for fname in sorted(os.listdir(data_path)):
        fpath = os.path.join(data_path, fname)
        if fname.endswith(".jsonl"):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line.strip())
                    text = row.get("assertion") or row.get("text") or ""
                    group = row.get("subject") or row.get("cultural_group") or "general"
                    meta = {k: row[k] for k in ("domain", "facet", "concepts") if k in row}
                    yield _normalize(text, group, "candle", meta)
                    count += 1
                    if limit and count >= limit:
                        return
        elif fname.endswith(".tsv"):
            with open(fpath, encoding="utf-8") as f:
                headers = None
                for line in f:
                    cols = line.rstrip("\n").split("\t")
                    if headers is None:
                        headers = cols
                        continue
                    row = dict(zip(headers, cols))
                    text = row.get("assertion") or row.get("text") or ""
                    group = row.get("subject") or row.get("cultural_group") or "general"
                    yield _normalize(text, group, "candle", row)
                    count += 1
                    if limit and count >= limit:
                        return


# ---------------------------------------------------------------------------
# Arab Culture  (MBZUAI/ArabCulture)
# Fields: first_statement, sub_topic, country, region
# ---------------------------------------------------------------------------
ARABCULTURE_CONFIGS = [
    "Algeria", "Egypt", "Jordan", "KSA", "Lebanon", "Libya",
    "Morocco", "Palestine", "Sudan", "Syria", "Tunisia", "UAE", "Yemen",
]

def load_arabculture(limit: int = 0) -> Iterator[dict]:
    from datasets import load_dataset
    count = 0
    for config in ARABCULTURE_CONFIGS:
        try:
            ds = load_dataset("MBZUAI/ArabCulture", config, split="test")
        except Exception:
            continue
        for row in ds:
            if limit and count >= limit:
                return
            text = row.get("first_statement") or ""
            group = (row.get("country") or config).lower()
            meta = {
                "sub_topic": row.get("sub_topic", ""),
                "region": row.get("region", ""),
            }
            yield _normalize(text, group, "arabculture", meta)
            count += 1


# ---------------------------------------------------------------------------
# DIWALI  (nlip/DIWALI)
# Fields: facet, state, concept, description
# ---------------------------------------------------------------------------
def load_diwali(limit: int = 0) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset("nlip/DIWALI", split="train")
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        concept = row.get("concept") or ""
        desc = row.get("description") or ""
        text = f"{concept}: {desc}".strip(": ")
        state = row.get("state") or "india"
        meta = {
            "facet": row.get("facet", ""),
            "state": state,
            "concept": concept,
        }
        yield _normalize(text, state, "diwali", meta)


# ---------------------------------------------------------------------------
# CultureBank  (SALT-NLP/CultureBank)
# Fields: cultural_group, context, goal, actor_behavior, other_descriptions, topic
# ---------------------------------------------------------------------------
def load_culturebank(limit: int = 0) -> Iterator[dict]:
    from datasets import load_dataset
    # CultureBank has 'tiktok' and 'reddit' splits; concatenate both
    count = 0
    for split in ("tiktok", "reddit"):
        ds = load_dataset("SALT-NLP/CultureBank", split=split)
        for row in ds:
            if limit and count >= limit:
                return
            parts = [
                row.get("context") or "",
                row.get("goal") or "",
                row.get("actor_behavior") or "",
                row.get("other_descriptions") or "",
            ]
            text = " | ".join(p for p in parts if p).strip(" |")
            group = row.get("cultural_group") or "general"
            meta = {
                "topic": row.get("topic", ""),
                "relation": row.get("relation", ""),
                "agreement": row.get("agreement", None),
            }
            yield _normalize(text, group, "culturebank", meta)
            count += 1


# ---------------------------------------------------------------------------
# BLEND  (nayeon212/BLEnD)
# Config: short-answer-questions
# Fields: ID, en_question, annotations (answers), country derived from ID
# ---------------------------------------------------------------------------
BLEND_SPLITS = ["DZ", "AS", "AZ", "CN", "ET", "GR", "ID", "IR", "MX", "KP", "NG", "KR", "ES", "GB", "US", "JB"]

def load_blend(limit: int = 0) -> Iterator[dict]:
    from datasets import load_dataset

    count = 0
    for split in BLEND_SPLITS:
        try:
            ds = load_dataset("nayeon212/BLEnD", "short-answer-questions", split=split)
        except Exception:
            continue
        for row in ds:
            if limit and count >= limit:
                return
            text = row.get("en_question") or row.get("question") or ""
            qid = str(row.get("ID") or "")
            group = split.lower()
            annotations = row.get("annotations") or []
            answers = []
            for ann in annotations:
                if isinstance(ann, dict):
                    answers.extend(ann.get("en_answers") or ann.get("answers") or [])
            meta = {"answers": answers, "id": qid}
            yield _normalize(text, group, "blend", meta)
            count += 1


# ---------------------------------------------------------------------------
# Ko-PIQA  (not yet released)
# ---------------------------------------------------------------------------
def load_kopiqa(limit: int = 0) -> Iterator[dict]:
    raise NotImplementedError(
        "Ko-PIQA (Choi et al., 2025) is not yet publicly released. "
        "Check https://huggingface.co/HAERAE-HUB for future release."
    )


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------
_LOADERS = {
    "candle":      load_candle,
    "arabculture": load_arabculture,
    "diwali":      load_diwali,
    "culturebank": load_culturebank,
    "blend":       load_blend,
    "kopiqa":      load_kopiqa,
}


def load_datasets(
    names: list[str] | None = None,
    limit: int = 0,
    candle_path: str = "",
) -> Iterator[dict]:
    """
    Load one or more datasets. `names` defaults to all available.
    `limit` caps entries per dataset (0 = no limit).
    `candle_path` is required when "candle" is in names.
    """
    targets = names if names else DATASET_NAMES
    for name in targets:
        name = name.lower().strip()
        if name not in _LOADERS:
            raise ValueError(f"Unknown dataset '{name}'. Choose from: {DATASET_NAMES}")
        print(f"[loader] Loading '{name}' ...")
        try:
            fn = _LOADERS[name]
            if name == "candle":
                yield from fn(candle_path, limit)
            else:
                yield from fn(limit)
        except NotImplementedError as e:
            print(f"[loader] Skipping '{name}': {e}")
        except Exception as e:
            print(f"[loader] Error loading '{name}': {e}")
