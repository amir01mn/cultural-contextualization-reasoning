"""
Exports a CulturalGraph to:
  - outputs/nodes.csv
  - outputs/edges.csv
  - outputs/cultural_graph.json  (NetworkX node-link format)
"""
from __future__ import annotations
import csv
import json
import os
from schema import CulturalGraph


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_csv(graph: CulturalGraph, output_dir: str) -> tuple[str, str]:
    _ensure_dir(output_dir)
    nodes_path = os.path.join(output_dir, "nodes.csv")
    edges_path = os.path.join(output_dir, "edges.csv")

    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "node_id", "label", "level", "cultural_group",
            "domain", "source_dataset", "frequency",
        ])
        writer.writeheader()
        for node in sorted(graph.node_list, key=lambda n: n.node_id):
            writer.writerow({
                "node_id": node.node_id,
                "label": node.label,
                "level": node.level,
                "cultural_group": node.cultural_group,
                "domain": node.domain,
                "source_dataset": node.source_dataset,
                "frequency": node.frequency,
            })

    # Resolve node IDs for edges
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_id", "target_id",
            "source_label", "target_label",
            "relation_type", "weight", "source_dataset",
        ])
        writer.writeheader()
        for edge in graph.edge_list:
            src_id = graph.node_id_for(edge.source_label, _guess_level(graph, edge.source_label))
            tgt_id = graph.node_id_for(edge.target_label, _guess_level(graph, edge.target_label))
            writer.writerow({
                "source_id": src_id,
                "target_id": tgt_id,
                "source_label": edge.source_label,
                "target_label": edge.target_label,
                "relation_type": edge.relation_type,
                "weight": edge.weight,
                "source_dataset": edge.source_dataset,
            })

    print(f"[exporter] CSV written → {nodes_path}, {edges_path}")
    return nodes_path, edges_path


def _guess_level(graph: CulturalGraph, label: str) -> int:
    """Return the level of the first matching node for label (any level)."""
    for key, node in graph.nodes.items():
        if key[0] == label:
            return node.level
    return 3


def export_json(graph: CulturalGraph, output_dir: str) -> str:
    """Export as NetworkX node-link JSON (compatible with nx.node_link_graph)."""
    _ensure_dir(output_dir)
    json_path = os.path.join(output_dir, "cultural_graph.json")

    nodes_data = []
    for node in sorted(graph.node_list, key=lambda n: n.node_id):
        nodes_data.append({
            "id": node.node_id,
            "label": node.label,
            "level": node.level,
            "cultural_group": node.cultural_group,
            "domain": node.domain,
            "source_dataset": node.source_dataset,
            "frequency": node.frequency,
        })

    # Build a label→id map for edge resolution
    label_to_ids: dict[str, int] = {}
    for node in graph.node_list:
        label_to_ids.setdefault(node.label, node.node_id)

    links_data = []
    for edge in graph.edge_list:
        src_id = label_to_ids.get(edge.source_label)
        tgt_id = label_to_ids.get(edge.target_label)
        if src_id is None or tgt_id is None:
            continue
        links_data.append({
            "source": src_id,
            "target": tgt_id,
            "source_label": edge.source_label,
            "target_label": edge.target_label,
            "relation_type": edge.relation_type,
            "weight": edge.weight,
            "source_dataset": edge.source_dataset,
        })

    payload = {
        "directed": True,
        "multigraph": False,
        "graph": {"name": "cultural_knowledge_graph"},
        "nodes": nodes_data,
        "links": links_data,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[exporter] JSON written → {json_path}")
    return json_path


def export_all(graph: CulturalGraph, output_dir: str = "outputs") -> None:
    export_csv(graph, output_dir)
    export_json(graph, output_dir)
