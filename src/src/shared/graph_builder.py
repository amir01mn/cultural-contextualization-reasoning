"""
Assembles raw node/edge dicts from the extractor into a deduplicated CulturalGraph.
"""
from __future__ import annotations
import re
from typing import Any
from schema import Node, Edge, CulturalGraph, RELATION_TYPES, CULTURAL_DOMAINS, LEVELS

# Dataset names that leak into node labels from CANDLE/DIWALI/BLEnD entries
_DATASET_NAME_NOISE = {"candle", "diwali", "blend", "arabculture", "culturebank"}

# Arabic Unicode block: U+0600–U+06FF
_ARABIC_RE = re.compile(r'[؀-ۿ]')


def _is_noisy(label: str) -> bool:
    """Return True if this node label should be filtered out."""
    if len(label) < 3:
        return True
    if label in _DATASET_NAME_NOISE:
        return True
    if _ARABIC_RE.search(label):
        return True
    return False


def _valid_level(val: Any) -> int:
    try:
        lvl = int(val)
        return lvl if 0 <= lvl <= 6 else 3
    except (TypeError, ValueError):
        return 3


def _coerce_node(raw: dict) -> Node | None:
    label = str(raw.get("label") or "").strip().lower()
    if not label or _is_noisy(label):
        return None
    level = _valid_level(raw.get("level"))
    cultural_group = str(raw.get("cultural_group") or "general").strip().lower()
    domain = str(raw.get("domain") or "unknown").strip().lower()
    source = str(raw.get("source_dataset") or "")
    return Node(
        label=label,
        level=level,
        cultural_group=cultural_group,
        domain=domain,
        source_dataset=source,
    )


def _coerce_edge(raw: dict) -> Edge | None:
    src = str(raw.get("source_label") or "").strip().lower()
    tgt = str(raw.get("target_label") or "").strip().lower()
    rel = str(raw.get("relation_type") or "influences").strip().lower()
    if not src or not tgt or src == tgt:
        return None
    # No forced fallback — preserve free-form relations from data-driven pipeline
    source = str(raw.get("source_dataset") or "")
    return Edge(source_label=src, target_label=tgt, relation_type=rel, source_dataset=source)


def build_graph(raw_nodes: list[dict], raw_edges: list[dict], anchor_to_root: bool = True) -> CulturalGraph:
    """
    Build and return a CulturalGraph from the extractor's raw output.
    - Normalizes and deduplicates nodes (frequency-counted)
    - Deduplicates edges (weight-counted)
    - Ensures the Level 0 "Human" root exists
    - Removes edges whose endpoints are not in the node set
    """
    graph = CulturalGraph()
    graph.ensure_root()

    # Ingest nodes
    known_labels: set[str] = set()
    for raw in raw_nodes:
        node = _coerce_node(raw)
        if node is None:
            continue
        graph.add_node(node)
        known_labels.add(node.label)

    # Ingest edges (only between known nodes)
    for raw in raw_edges:
        edge = _coerce_edge(raw)
        if edge is None:
            continue
        if edge.source_label not in known_labels or edge.target_label not in known_labels:
            continue
        graph.add_edge(edge)

    # In Task 1 (anchor_to_root=True), force all Level-1 nodes to connect to human root.
    # In Task 2 (anchor_to_root=False), connections emerge from the data itself.
    if anchor_to_root:
        human_label = "human"
        for node in graph.node_list:
            if node.level == 1 and node.label != human_label:
                connector = Edge(
                    source_label=human_label,
                    target_label=node.label,
                    relation_type="manifests_in",
                    source_dataset="schema",
                )
                graph.add_edge(connector)

    print(
        f"[builder] Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
    )
    return graph
