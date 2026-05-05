"""
Phase 4: Build one knowledge graph per induced lens schema.

For each lens in induced_schemas.json:
  - Filter all raw triples to those whose relation belongs to this lens
  - Convert to Node/Edge objects using the data-derived entity clusters
  - Assemble via graph_builder (reused)
  - Export via exporter (reused) to outputs/induced/forest/lens_<N>_*
  - Generate visualization via visualize (reused)
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from schema import Node, Edge, CulturalGraph
from graph_builder import build_graph
from exporter import export_all


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rel_to_canonical(relation_clusters: list[dict]) -> dict[str, str]:
    mapping = {}
    for rc in relation_clusters:
        for m in rc.get("members", [rc["canonical"]]):
            mapping[m] = rc["canonical"]
    return mapping


def _ent_to_cluster(entity_clusters: list[dict]) -> tuple[dict[str, str], dict[str, dict]]:
    """Returns (label→canonical, canonical→cluster_info)."""
    label_map = {}
    cluster_map = {}
    for ec in entity_clusters:
        canonical = ec["canonical"]
        cluster_map[canonical] = ec
        for m in ec.get("members", [canonical]):
            label_map[m] = canonical
    return label_map, cluster_map


def build_lens_graph(
    records: list[dict],
    schema: dict,
    entity_clusters: list[dict],
    relation_clusters: list[dict],
) -> CulturalGraph:
    """Build a CulturalGraph for a single lens schema."""
    lens_edge_types = set(schema.get("edge_types", [schema["lens_name"]]))
    rel_canonical = _rel_to_canonical(relation_clusters)
    ent_canonical, cluster_info = _ent_to_cluster(entity_clusters)

    raw_nodes: list[dict] = []
    raw_edges: list[dict] = []

    for rec in records:
        source = rec.get("source_dataset", "")
        cultural_group = rec.get("cultural_group", "")

        for t in rec.get("triples", []):
            rel = t.get("relation", "")
            # Only include triples whose relation belongs to this lens
            if rel_canonical.get(rel, rel) != schema["lens_name"] and rel not in lens_edge_types:
                continue

            head = ent_canonical.get(t.get("head", ""), t.get("head", ""))
            tail = ent_canonical.get(t.get("tail", ""), t.get("tail", ""))
            if not head or not tail:
                continue

            head_info = cluster_info.get(head, {})
            tail_info = cluster_info.get(tail, {})

            raw_nodes.append({
                "label": head,
                "level": head_info.get("level", 3),
                "cultural_group": cultural_group,
                "domain": "unknown",
                "source_dataset": source,
            })
            raw_nodes.append({
                "label": tail,
                "level": tail_info.get("level", 3),
                "cultural_group": cultural_group,
                "domain": "unknown",
                "source_dataset": source,
            })
            raw_edges.append({
                "source_label": head,
                "target_label": tail,
                "relation_type": _sanitize_relation(rel),
                "source_dataset": source,
            })

    return build_graph(raw_nodes, raw_edges, anchor_to_root=False)


def _sanitize_relation(rel: str) -> str:
    """Convert free-form relation to a slug safe for storage."""
    return rel.strip().lower().replace(" ", "_")[:60]


def build_forest(
    records: list[dict],
    schemas: list[dict],
    entity_clusters: list[dict],
    relation_clusters: list[dict],
    output_dir: str,
    generate_viz: bool = True,
) -> list[dict]:
    """
    Build and export one graph per lens. Returns summary list.
    """
    forest_dir = os.path.join(output_dir, "forest")
    os.makedirs(forest_dir, exist_ok=True)

    summary = []
    for schema in schemas:
        lid = schema["lens_id"]
        lens_name = schema["lens_name"]
        prefix = os.path.join(forest_dir, f"lens_{lid}")

        print(f"\n[forest] Building lens {lid}: '{lens_name}' (coverage={schema['coverage']}) ...")
        graph = build_lens_graph(records, schema, entity_clusters, relation_clusters)

        if len(graph.nodes) == 0:
            print(f"[forest] Lens {lid} produced 0 nodes — skipping.")
            continue

        # Export CSV + JSON
        lens_output_dir = os.path.join(forest_dir, f"lens_{lid}")
        os.makedirs(lens_output_dir, exist_ok=True)
        export_all(graph, output_dir=lens_output_dir)

        # Rename outputs to include lens name for clarity
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        # Generate visualization
        if generate_viz:
            try:
                import subprocess
                viz_path = os.path.join(lens_output_dir, "viz.html")
                result = subprocess.run(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "visualize.py"),
                     "--input", os.path.join(lens_output_dir, "cultural_graph.json"),
                     "--output", viz_path,
                     "--levels", "0", "1", "2"],
                    capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
                )
                if result.returncode == 0:
                    print(f"[forest] Visualization → {viz_path}")
            except Exception as e:
                print(f"[forest] Viz skipped: {e}")

        entry = {
            "lens_id": lid,
            "lens_name": lens_name,
            "nodes": node_count,
            "edges": edge_count,
            "coverage": schema["coverage"],
            "output_dir": lens_output_dir,
        }
        summary.append(entry)
        print(f"[forest] Lens {lid} done: {node_count} nodes, {edge_count} edges → {lens_output_dir}/")

    # Save forest summary
    summary_path = os.path.join(output_dir, "forest_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[forest] Forest summary → {summary_path}")
    return summary
