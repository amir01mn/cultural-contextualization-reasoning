"""
Task 3: Build ONE unified cultural knowledge graph from all mined triples.

Unlike Task 2's forest (one graph per relation type), this puts ALL triples
into a single graph — preserving diverse edge types while maintaining the
hierarchical structure needed for traversable cultural pathways.

A pathway like:
  Human →[manifests_in]→ religion →[contextualizes]→ Arab
         →[honors]→ Ramadan fasting →[manifests_in]→ not eating pork

requires different relations at each step. This only works in ONE graph
where all relation types coexist.

Data sources (already computed, no re-mining needed):
  - outputs/induced/raw_triples.jsonl     ← mined triples (Phase 1)
  - outputs/induced/entity_clusters.json  ← frequency-derived levels (Phase 2)
"""
from __future__ import annotations
import json
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, 'shared'))
sys.path.insert(0, os.path.join(_SRC, 'task3'))

from schema import CulturalGraph, Node, Edge
from graph_builder import build_graph


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_triples(path: str) -> list[dict]:
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


def _build_entity_maps(entity_clusters: list[dict]) -> tuple[dict, dict]:
    """
    Returns:
      label_to_canonical: any member label → canonical label
      canonical_to_info:  canonical label → {level, frequency}
    """
    label_to_canonical = {}
    canonical_to_info = {}
    for cluster in entity_clusters:
        canonical = cluster["canonical"]
        level = cluster.get("level", 3)
        freq = cluster.get("frequency", 1)
        canonical_to_info[canonical] = {"level": level, "frequency": freq}
        for member in cluster.get("members", [canonical]):
            label_to_canonical[member] = canonical
    return label_to_canonical, canonical_to_info


def build_unified_graph(
    triples_path: str,
    entity_clusters_path: str,
    min_edge_weight: int = 1,
) -> CulturalGraph:
    """
    Build a single unified knowledge graph from ALL mined triples.

    - Nodes: entity clusters with frequency-derived levels
    - Edges: all discovered relation types preserved (not filtered by lens)
    - Root: Human node anchored to all Level-1 nodes
    - Result: one traversable graph where pathways cross multiple relation types
    """
    print("[task3] Loading mined triples ...")
    records = _load_triples(triples_path)
    total_triples = sum(len(r.get("triples", [])) for r in records)
    print(f"[task3] {len(records)} entries, {total_triples} raw triples")

    print("[task3] Loading entity clusters ...")
    entity_clusters = _load_json(entity_clusters_path)
    label_to_canonical, canonical_to_info = _build_entity_maps(entity_clusters)

    # Collect raw nodes + edges
    raw_nodes: list[dict] = []
    raw_edges: list[dict] = []

    for rec in records:
        source = rec.get("source_dataset", "")
        cultural_group = rec.get("cultural_group", "general")

        for triple in rec.get("triples", []):
            head_raw = triple.get("head", "").strip().lower()
            tail_raw = triple.get("tail", "").strip().lower()
            relation  = triple.get("relation", "").strip().lower()

            if not head_raw or not tail_raw or not relation:
                continue

            # Map to canonical cluster labels (normalizes synonyms)
            head = label_to_canonical.get(head_raw, head_raw)
            tail = label_to_canonical.get(tail_raw, tail_raw)

            head_info = canonical_to_info.get(head, {})
            tail_info = canonical_to_info.get(tail, {})

            raw_nodes.append({
                "label":          head,
                "level":          head_info.get("level", 3),
                "cultural_group": cultural_group,
                "domain":         "unknown",
                "source_dataset": source,
            })
            raw_nodes.append({
                "label":          tail,
                "level":          tail_info.get("level", 3),
                "cultural_group": cultural_group,
                "domain":         "unknown",
                "source_dataset": source,
            })
            raw_edges.append({
                "source_label":  head,
                "target_label":  tail,
                "relation_type": relation,
                "source_dataset": source,
            })

    print(f"[task3] Building unified graph (anchor_to_root=True) ...")
    graph = build_graph(raw_nodes, raw_edges, anchor_to_root=True)

    # Filter out very weak edges if requested
    if min_edge_weight > 1:
        before = len(graph.edges)
        graph.edges = {k: e for k, e in graph.edges.items()
                       if e.weight >= min_edge_weight}
        print(f"[task3] Edge filter (weight>={min_edge_weight}): {before} → {len(graph.edges)}")

    return graph
