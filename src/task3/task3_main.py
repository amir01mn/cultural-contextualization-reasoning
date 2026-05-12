"""
Task 3 orchestrator: build the unified cultural knowledge graph.

Usage:
  python src/task3/task3_main.py
  python src/task3/task3_main.py --min-edge-weight 2
  python src/task3/task3_main.py --induced-dir outputs/induced --output-dir outputs/task3
"""
import argparse
import json
import os
import sys
import subprocess

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, 'shared'))
sys.path.insert(0, os.path.join(_SRC, 'task3'))

from unified_builder import build_unified_graph
from exporter import export_all
from collections import Counter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3: Unified cultural knowledge graph.")
    p.add_argument("--induced-dir",     default="outputs/induced",
                   help="Directory with raw_triples.jsonl and entity_clusters.json")
    p.add_argument("--output-dir",      default="outputs/task3",
                   help="Output directory (default: outputs/task3/)")
    p.add_argument("--min-edge-weight", type=int, default=1,
                   help="Minimum edge weight to keep (default: 1 = keep all)")
    return p.parse_args()


def analyse(graph, output_dir: str) -> None:
    from schema import LEVELS
    from collections import Counter
    import csv

    nodes = list(graph.nodes.values())
    edges = list(graph.edges.values())

    level_counts = Counter(n.level for n in nodes)
    rel_counts   = Counter(e.relation_type for e in edges)

    print(f"\n{'='*60}")
    print(f"TASK 3 — UNIFIED KNOWLEDGE GRAPH")
    print(f"{'='*60}")
    print(f"  Nodes              : {len(nodes)}")
    print(f"  Edges              : {len(edges)}")
    print(f"  Unique edge types  : {len(rel_counts)}")
    print(f"\n  Level distribution:")
    for lvl in sorted(level_counts):
        examples = [n.label for n in nodes if n.level == lvl][:3]
        print(f"    Level {lvl}: {level_counts[lvl]:5d} nodes  e.g. {examples}")

    print(f"\n  Top 15 relation types:")
    for rel, cnt in rel_counts.most_common(15):
        print(f"    {cnt:5d}  {rel}")

    # Sample pathways: find chains starting from human
    import networkx as nx
    with open(os.path.join(output_dir, "cultural_graph.json")) as f:
        gdata = json.load(f)
    G = nx.node_link_graph(gdata, directed=True, multigraph=False, edges="links")

    print(f"\n  Graph connectivity:")
    wcc = list(nx.weakly_connected_components(G))
    print(f"    Components : {len(wcc)}")
    print(f"    Largest    : {max(len(c) for c in wcc)} nodes "
          f"({max(len(c) for c in wcc)/len(nodes)*100:.1f}%)")
    print(f"    Avg out-deg: {sum(d for _,d in G.out_degree())/len(nodes):.2f}")

    # Show sample traversable pathways
    print(f"\n  Sample pathways from Human root:")
    human_nodes = [n for n, d in G.nodes(data=True) if d.get("label") == "human"]
    if human_nodes:
        human_id = human_nodes[0]
        paths_shown = 0
        for neighbor in list(G.successors(human_id))[:5]:
            n_data = G.nodes[neighbor]
            edge_data = G.edges[human_id, neighbor]
            for neighbor2 in list(G.successors(neighbor))[:2]:
                n2_data = G.nodes[neighbor2]
                edge2_data = G.edges[neighbor, neighbor2]
                for neighbor3 in list(G.successors(neighbor2))[:1]:
                    n3_data = G.nodes[neighbor3]
                    edge3_data = G.edges[neighbor2, neighbor3]
                    rel1 = edge_data.get("relation_type", "?")
                    rel2 = edge2_data.get("relation_type", "?")
                    rel3 = edge3_data.get("relation_type", "?")
                    l1 = n_data.get("label","?")
                    l2 = n2_data.get("label","?")
                    l3 = n3_data.get("label","?")
                    print(f"    human -[{rel1}]-> {l1} -[{rel2}]-> {l2} -[{rel3}]-> {l3}")
                    paths_shown += 1
                    if paths_shown >= 5:
                        break
                if paths_shown >= 5:
                    break
            if paths_shown >= 5:
                break

    print(f"\n  Output: {os.path.abspath(output_dir)}/")
    print(f"{'='*60}")

    # Comparison table
    print(f"\n  COMPARISON:")
    print(f"    Task 1 (hardcoded schema) → outputs/          "
          f"7,542 nodes,  8,639 edges, 1 graph,  8 edge types")
    print(f"    Task 2 (forest of lenses) → outputs/task2/    "
          f"6,495 nodes,  4,164 edges, 20 graphs, 50 edge types")
    print(f"    Task 3 (unified+derived)  → outputs/task3/    "
          f"{len(nodes):5d} nodes, {len(edges):5d} edges, 1 graph, {len(rel_counts)} edge types")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    triples_path  = os.path.join(args.induced_dir, "raw_triples.jsonl")
    clusters_path = os.path.join(args.induced_dir, "entity_clusters.json")

    for p in [triples_path, clusters_path]:
        if not os.path.exists(p):
            print(f"[task3] Missing: {p}")
            print(f"[task3] Run Task 2 mining first: python src/task2/induction_main.py ...")
            sys.exit(1)

    # Build
    graph = build_unified_graph(
        triples_path=triples_path,
        entity_clusters_path=clusters_path,
        min_edge_weight=args.min_edge_weight,
    )

    # Export
    export_all(graph, output_dir=args.output_dir)

    # Analyse + print pathways
    analyse(graph, args.output_dir)

    # Visualize — levels 0-2 overview
    viz_script = os.path.join(_SRC, "shared", "visualize.py")
    for label, flags in [
        ("overview (levels 0-2)", ["--levels", "0", "1", "2"]),
        ("top 300 nodes",         ["--top",    "300"]),
    ]:
        out = os.path.join(args.output_dir,
                           "viz_overview.html" if "levels" in flags else "viz_top300.html")
        subprocess.run(
            [sys.executable, viz_script,
             "--input",  os.path.join(args.output_dir, "cultural_graph.json"),
             "--output", out] + flags,
            capture_output=True,
            cwd=os.path.dirname(_SRC)
        )
        print(f"[task3] Viz ({label}) → {out}")


if __name__ == "__main__":
    main()
