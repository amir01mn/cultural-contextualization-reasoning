"""
Builds cross-lens edges: connections between lens graphs that share entities.

For example, if "fasting" appears in both the "values" graph and the
"practices" graph, a cross-lens edge captures that the same cultural concept
is simultaneously a value AND a practice — the kind of multi-dimensional
connection the professor wants.

Output:
  outputs/final/cross_lens_edges.csv   — all inter-graph connections
  outputs/final/cross_lens_graph.json  — combined graph for visualization
"""
from __future__ import annotations
import csv
import json
import os
from collections import defaultdict


def load_lens_nodes(graphs_dir: str) -> dict[str, dict[str, dict]]:
    """
    Returns {lens_name: {node_label: node_dict}} for all lens graphs.
    """
    lens_nodes = {}
    for lens_dir in sorted(os.listdir(graphs_dir)):
        nodes_path = os.path.join(graphs_dir, lens_dir, "nodes.csv")
        summary_path = os.path.join(graphs_dir, lens_dir, "..", "..", "final_summary.json")
        if not os.path.exists(nodes_path):
            continue
        nodes = {}
        with open(nodes_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                nodes[row["label"]] = row
        lens_nodes[lens_dir] = nodes
    return lens_nodes


def load_summary(output_dir: str) -> dict[str, str]:
    """Returns {lens_dir_name: lens_name}."""
    summary_path = os.path.join(output_dir, "final_summary.json")
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    return {f"lens_{g['lens_id']:02d}": g["lens_name"] for g in summary}


def build_cross_lens_edges(
    output_dir: str = "outputs/final",
    min_shared_freq: int = 2,
) -> list[dict]:
    """
    For every entity that appears in 2+ lens graphs, create a cross-lens edge
    between each pair of graphs that share it.

    Edge format:
      source_lens, target_lens, shared_entity, source_lens_name, target_lens_name,
      shared_count (how many entities the two lenses share)
    """
    graphs_dir = os.path.join(output_dir, "graphs")
    lens_nodes = load_lens_nodes(graphs_dir)
    lens_name_map = load_summary(output_dir)

    # Build: entity_label → set of lens_dirs that contain it
    entity_to_lenses: dict[str, set[str]] = defaultdict(set)
    for lens_dir, nodes in lens_nodes.items():
        for label in nodes:
            entity_to_lenses[label].add(lens_dir)

    # Only keep entities appearing in 2+ lenses
    shared = {e: lenses for e, lenses in entity_to_lenses.items()
              if len(lenses) >= 2}

    print(f"[cross_lens] {len(shared)} entities appear in 2+ lens graphs")

    # Count how many entities each lens-pair shares
    pair_entities: dict[tuple, list[str]] = defaultdict(list)
    for entity, lenses in shared.items():
        lenses_sorted = sorted(lenses)
        for i in range(len(lenses_sorted)):
            for j in range(i + 1, len(lenses_sorted)):
                pair_entities[(lenses_sorted[i], lenses_sorted[j])].append(entity)

    # Build cross-lens edge records
    cross_edges = []
    for (lens_a, lens_b), entities in pair_entities.items():
        if len(entities) < min_shared_freq:
            continue
        cross_edges.append({
            "source_lens": lens_a,
            "target_lens": lens_b,
            "source_lens_name": lens_name_map.get(lens_a, lens_a),
            "target_lens_name": lens_name_map.get(lens_b, lens_b),
            "shared_entity_count": len(entities),
            "top_shared_entities": ", ".join(sorted(entities, key=lambda e: -len(entity_to_lenses[e]))[:5]),
        })

    cross_edges.sort(key=lambda e: e["shared_entity_count"], reverse=True)
    print(f"[cross_lens] {len(cross_edges)} cross-lens connections found")

    # Save CSV
    csv_path = os.path.join(output_dir, "cross_lens_edges.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_lens", "target_lens",
            "source_lens_name", "target_lens_name",
            "shared_entity_count", "top_shared_entities",
        ])
        writer.writeheader()
        writer.writerows(cross_edges)
    print(f"[cross_lens] CSV → {csv_path}")

    # Build a meta-graph: nodes = lenses, edges = shared entity connections
    lens_dirs = sorted(lens_nodes.keys())
    meta_nodes = []
    for i, ld in enumerate(lens_dirs):
        meta_nodes.append({
            "id": i,
            "label": lens_name_map.get(ld, ld),
            "lens_dir": ld,
            "node_count": len(lens_nodes[ld]),
        })

    lens_dir_to_id = {ld: i for i, ld in enumerate(lens_dirs)}
    meta_edges = []
    for e in cross_edges:
        src_id = lens_dir_to_id.get(e["source_lens"])
        tgt_id = lens_dir_to_id.get(e["target_lens"])
        if src_id is None or tgt_id is None:
            continue
        meta_edges.append({
            "source": src_id,
            "target": tgt_id,
            "source_lens_name": e["source_lens_name"],
            "target_lens_name": e["target_lens_name"],
            "weight": e["shared_entity_count"],
            "top_shared": e["top_shared_entities"],
        })

    meta_graph = {
        "directed": False,
        "multigraph": False,
        "graph": {"name": "cross_lens_meta_graph"},
        "nodes": meta_nodes,
        "links": meta_edges,
    }
    json_path = os.path.join(output_dir, "cross_lens_graph.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_graph, f, ensure_ascii=False, indent=2)
    print(f"[cross_lens] Meta-graph JSON → {json_path}")

    # Build interactive visualization of the meta-graph
    _visualize_meta_graph(meta_graph, output_dir, cross_edges)

    return cross_edges


def _visualize_meta_graph(
    meta_graph: dict,
    output_dir: str,
    cross_edges: list[dict],
) -> None:
    try:
        from pyvis.network import Network
    except ImportError:
        print("[cross_lens] pyvis not available — skipping visualization")
        return

    net = Network(
        height="850px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        notebook=False,
    )
    net.set_options("""
    {
      "physics": {"barnesHut": {"gravitationalConstant": -5000, "springLength": 200}},
      "edges": {
        "color": {"opacity": 0.7},
        "scaling": {"min": 1, "max": 10}
      },
      "interaction": {"hover": true}
    }
    """)

    max_weight = max((e["shared_entity_count"] for e in cross_edges), default=1)

    for node in meta_graph["nodes"]:
        net.add_node(
            node["id"],
            label=node["label"],
            title=f"<b>{node['label']}</b><br>{node['node_count']} nodes",
            size=20 + node["node_count"] // 30,
            color="#0099cc",
        )

    for link in meta_graph["links"]:
        width = max(1, int(link["weight"] / max_weight * 10))
        net.add_edge(
            link["source"], link["target"],
            title=f"Shared: {link['weight']} entities<br>{link['top_shared']}",
            width=width,
            label=str(link["weight"]) if link["weight"] > 5 else "",
        )

    viz_path = os.path.join(output_dir, "cross_lens_viz.html")
    net.save_graph(viz_path)
    print(f"[cross_lens] Visualization → {viz_path}")
