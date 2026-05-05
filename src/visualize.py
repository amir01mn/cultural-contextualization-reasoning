"""
Visualize the Cultural Knowledge Graph as an interactive HTML file.

Usage:
  # Full graph, levels 0-2 only (manageable ~830 nodes):
  python src/visualize.py --levels 0 1 2

  # Top 200 most-frequent nodes across all levels:
  python src/visualize.py --top 200

  # Subgraph rooted at a specific node (e.g. all paths from "religion"):
  python src/visualize.py --root religion --depth 3

  # Full graph (slow, but interactive):
  python src/visualize.py --full
"""
import argparse
import json
import sys
import os
import networkx as nx
from pyvis.network import Network

LEVEL_COLORS = {
    0: "#1a1aff",   # blue     — Human root
    1: "#0099cc",   # teal     — Domain
    2: "#00bb77",   # green    — Group
    3: "#ff8800",   # orange   — Practice/Norm
    4: "#cc2200",   # red      — Specific Instance
}
LEVEL_SIZES = {0: 40, 1: 28, 2: 20, 3: 14, 4: 10}
LEVEL_LABELS = {0: "Root", 1: "Domain", 2: "Group", 3: "Practice", 4: "Instance"}


def load_graph(json_path: str) -> nx.DiGraph:
    with open(json_path) as f:
        data = json.load(f)
    return nx.node_link_graph(data, directed=True, multigraph=False, edges="links")


def build_pyvis(G: nx.DiGraph, title: str) -> Network:
    net = Network(
        height="900px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        directed=True,
        notebook=False,
    )
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 120
        },
        "minVelocity": 0.75
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "color": {"opacity": 0.5},
        "width": 0.8
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    for node_id, attrs in G.nodes(data=True):
        level = attrs.get("level", 3)
        freq  = attrs.get("frequency", 1)
        label = attrs.get("label", str(node_id))
        group = attrs.get("cultural_group", "")
        domain = attrs.get("domain", "")
        source = attrs.get("source_dataset", "")

        size  = LEVEL_SIZES.get(level, 12) + min(freq // 10, 20)
        color = LEVEL_COLORS.get(level, "#888888")
        title_str = (
            f"<b>{label}</b><br>"
            f"Level: {level} ({LEVEL_LABELS.get(level, '?')})<br>"
            f"Group: {group}<br>"
            f"Domain: {domain}<br>"
            f"Source: {source}<br>"
            f"Frequency: {freq}"
        )
        net.add_node(
            node_id,
            label=label if level <= 2 else (label[:30] + "…" if len(label) > 30 else label),
            title=title_str,
            color=color,
            size=size,
        )

    for src, tgt, attrs in G.edges(data=True):
        rel = attrs.get("relation_type", "")
        weight = attrs.get("weight", 1)
        net.add_edge(src, tgt, title=rel, label=rel if weight > 2 else "", width=min(weight * 0.5, 4))

    return net


def filter_by_levels(G: nx.DiGraph, levels: list[int]) -> nx.DiGraph:
    keep = [n for n, d in G.nodes(data=True) if d.get("level") in levels]
    return G.subgraph(keep).copy()


def filter_top_n(G: nx.DiGraph, n: int) -> nx.DiGraph:
    ranked = sorted(G.nodes(data=True), key=lambda x: x[1].get("frequency", 1), reverse=True)
    keep = [node_id for node_id, _ in ranked[:n]]
    return G.subgraph(keep).copy()


def filter_rooted(G: nx.DiGraph, root_label: str, depth: int) -> nx.DiGraph:
    # Find node id matching label
    root_id = None
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("label", "").lower() == root_label.lower():
            root_id = node_id
            break
    if root_id is None:
        print(f"Node '{root_label}' not found. Showing full graph.")
        return G
    reachable = nx.ego_graph(G, root_id, radius=depth, undirected=True)
    return reachable


def main():
    parser = argparse.ArgumentParser(description="Visualize the cultural knowledge graph.")
    parser.add_argument("--input",  default="outputs/cultural_graph.json")
    parser.add_argument("--output", default="outputs/graph_viz.html")
    parser.add_argument("--levels", nargs="+", type=int, default=None,
                        help="Only show nodes at these levels (0-4)")
    parser.add_argument("--top",    type=int, default=None,
                        help="Only show top N nodes by frequency")
    parser.add_argument("--root",   type=str, default=None,
                        help="Show subgraph around this node label")
    parser.add_argument("--depth",  type=int, default=3,
                        help="Depth for --root subgraph (default 3)")
    parser.add_argument("--full",   action="store_true",
                        help="Render full graph (slow for 7k+ nodes)")
    args = parser.parse_args()

    print(f"[viz] Loading {args.input} ...")
    G = load_graph(args.input)
    print(f"[viz] Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if args.full:
        subG = G
        title = "Cultural Knowledge Graph — Full"
    elif args.root:
        subG = filter_rooted(G, args.root, args.depth)
        title = f"Cultural Knowledge Graph — '{args.root}' (depth {args.depth})"
    elif args.top:
        subG = filter_top_n(G, args.top)
        title = f"Cultural Knowledge Graph — Top {args.top} nodes"
    elif args.levels:
        subG = filter_by_levels(G, args.levels)
        title = f"Cultural Knowledge Graph — Levels {args.levels}"
    else:
        # Default: levels 0-2 (manageable)
        subG = filter_by_levels(G, [0, 1, 2])
        title = "Cultural Knowledge Graph — Levels 0–2 (Domain overview)"

    print(f"[viz] Rendering {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges ...")
    net = build_pyvis(subG, title)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    net.save_graph(args.output)
    print(f"[viz] Saved → {os.path.abspath(args.output)}")
    print(f"[viz] Open in browser: open {args.output}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
