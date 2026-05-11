"""
Final pipeline: build proper knowledge graphs using the data-derived schema.

This completes the loop:
  Task 1: hardcoded schema → structured extraction → KG  (outputs/)
  Task 2: free mining → discovered schema → [THIS] structured extraction → KG  (outputs/final/)

Usage:
  # Smoke test:
  python src/final_main.py --datasets culturebank --sample 5 --verbose

  # Full run (reuses existing dataset entries from Task 1/2):
  python src/final_main.py --datasets arabculture diwali culturebank blend candle \\
      --candle-path datasets/candle --sample 500
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loaders import load_datasets, DATASET_NAMES
from lens_consolidator import consolidate
from final_extractor import extract_batch, load_extractions
from graph_builder import build_graph
from exporter import export_all
from cross_lens import build_cross_lens_edges


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final KG construction using data-derived schema.")
    p.add_argument("--datasets", nargs="+", default=None, metavar="DATASET")
    p.add_argument("--sample", type=int, default=0, metavar="N")
    p.add_argument("--candle-path", type=str, default="")
    p.add_argument("--induced-dir", type=str, default="outputs/induced",
                   help="Where induced schemas live (default: outputs/induced)")
    p.add_argument("--output-dir", type=str, default="outputs/final",
                   help="Output dir (default: outputs/final/). Task 1+2 outputs untouched.")
    p.add_argument("--skip-extraction", action="store_true",
                   help="Skip extraction if raw_extractions.jsonl already exists.")
    p.add_argument("--eps", type=float, default=0.52,
                   help="DBSCAN eps for super-lens consolidation (default 0.52).")
    p.add_argument("--backend", type=str, default="ollama", choices=["ollama", "hf"],
                   help="LLM backend: 'ollama' for local Mac, 'hf' for Narval GPU (default: ollama)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_graphs_from_extractions(
    records: list[dict],
    super_lenses: list[dict],
    output_dir: str,
) -> list[dict]:
    """Build one knowledge graph per super-lens from the structured extractions."""
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Index super-lenses by id and name
    lens_by_id = {sl["super_lens_id"]: sl for sl in super_lenses}
    lens_by_name = {sl["name"]: sl for sl in super_lenses}

    # Collect raw nodes + edges per super-lens
    lens_raw: dict[int, tuple[list, list]] = {
        sl["super_lens_id"]: ([], []) for sl in super_lenses
    }

    for rec in records:
        source = rec.get("source_dataset", "")
        cultural_group = rec.get("cultural_group", "")
        for ext in rec.get("extractions", []):
            lid = ext.get("lens_id")
            lname = ext.get("lens_name", "")
            # Resolve lens id
            if lid is None and lname in lens_by_name:
                lid = lens_by_name[lname]["super_lens_id"]
            if lid not in lens_raw:
                continue

            nodes_list, edges_list = lens_raw[lid]
            for n in ext.get("nodes", []):
                label = str(n.get("label", "")).strip().lower()
                if not label:
                    continue
                nodes_list.append({
                    "label": label,
                    "level": n.get("level", 3),
                    "cultural_group": n.get("cultural_group") or cultural_group,
                    "domain": "unknown",
                    "source_dataset": source,
                })
            # Collect node labels already declared in this extraction
            declared_labels = {
                str(n.get("label", "")).strip().lower()
                for n in ext.get("nodes", [])
            }

            for e in ext.get("edges", []):
                head = str(e.get("head", "")).strip().lower()
                tail = str(e.get("tail", "")).strip().lower()
                rel  = str(e.get("relation", lname)).strip().lower()
                if not head or not tail:
                    continue
                # Auto-add endpoint nodes that the model forgot to declare
                for endpoint in (head, tail):
                    if endpoint and endpoint not in declared_labels:
                        nodes_list.append({
                            "label": endpoint,
                            "level": 2,
                            "cultural_group": cultural_group,
                            "domain": "unknown",
                            "source_dataset": source,
                        })
                        declared_labels.add(endpoint)
                edges_list.append({
                    "source_label": head,
                    "target_label": tail,
                    "relation_type": rel,
                    "source_dataset": source,
                })

    summary = []
    for sl in super_lenses:
        lid = sl["super_lens_id"]
        raw_nodes, raw_edges = lens_raw[lid]
        if not raw_nodes:
            continue

        print(f"[final] Building graph for super-lens {lid}: '{sl['name']}' ...")
        graph = build_graph(raw_nodes, raw_edges, anchor_to_root=False)

        lens_dir = os.path.join(graphs_dir, f"lens_{lid:02d}")
        export_all(graph, output_dir=lens_dir)

        # Visualization
        try:
            import subprocess
            viz_path = os.path.join(lens_dir, "viz.html")
            subprocess.run(
                [sys.executable,
                 os.path.join(os.path.dirname(__file__), "visualize.py"),
                 "--input", os.path.join(lens_dir, "cultural_graph.json"),
                 "--output", viz_path,
                 "--top", "200"],
                capture_output=True,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
        except Exception:
            pass

        summary.append({
            "lens_id": lid,
            "lens_name": sl["name"],
            "constituent_lenses": sl["constituent_lenses"],
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "output_dir": lens_dir,
        })
        print(f"  → {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    summary_path = os.path.join(output_dir, "final_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def print_final_summary(summary: list[dict], output_dir: str) -> None:
    total_nodes = sum(g["nodes"] for g in summary)
    total_edges = sum(g["edges"] for g in summary)
    print("\n" + "=" * 65)
    print("FINAL KNOWLEDGE GRAPH CONSTRUCTION COMPLETE")
    print("=" * 65)
    print(f"\nOutput : {os.path.abspath(output_dir)}/")
    print(f"Graphs : {len(summary)}")
    print(f"Nodes  : {total_nodes}  |  Edges : {total_edges}")
    print(f"\nPer-lens breakdown:")
    for g in summary:
        constituents = ", ".join(f"'{c}'" for c in g["constituent_lenses"][:3])
        if len(g["constituent_lenses"]) > 3:
            constituents += f" +{len(g['constituent_lenses'])-3}"
        print(f"  lens_{g['lens_id']:02d} '{g['lens_name']:28s}' "
              f"→ {g['nodes']:4d} nodes, {g['edges']:4d} edges  [{constituents}]")
    print(f"\nComparison:")
    print(f"  Task 1 (hardcoded schema) → outputs/")
    print(f"  Task 2 raw forest         → outputs/induced/forest/")
    print(f"  Task 2 final KGs          → {output_dir}/graphs/")
    print("=" * 65)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Consolidate lenses ────────────────────────────────────────
    super_lenses_path = os.path.join(args.induced_dir, "super_lenses.json")
    print("[final] Step 1: Consolidating lenses ...")
    super_lenses = consolidate(
        induced_dir=args.induced_dir,
        output_path=super_lenses_path,
        eps=args.eps,
    )

    entity_clusters_path = os.path.join(args.induced_dir, "entity_clusters.json")
    with open(entity_clusters_path, encoding="utf-8") as f:
        entity_clusters = json.load(f)

    # ── Step 2: Structured extraction ────────────────────────────────────
    extractions_path = os.path.join(args.output_dir, "raw_extractions.jsonl")

    if args.skip_extraction and os.path.exists(extractions_path):
        print(f"[final] Skipping extraction — loading {extractions_path}")
        records = load_extractions(extractions_path)
    else:
        print(f"\n[final] Step 2: Loading datasets ...")
        entries = list(load_datasets(
            names=args.datasets,
            limit=args.sample,
            candle_path=args.candle_path,
        ))
        print(f"[final] {len(entries)} entries. Starting structured extraction via Ollama ...")
        extract_batch(
            entries=entries,
            super_lenses=super_lenses,
            entity_clusters=entity_clusters,
            output_path=extractions_path,
            resume=True,
            verbose=args.verbose,
            backend=args.backend,
        )
        records = load_extractions(extractions_path)

    if not records:
        print("[final] No extractions found. Exiting.")
        sys.exit(1)

    # ── Step 3: Build graphs ──────────────────────────────────────────────
    print(f"\n[final] Step 3: Building knowledge graphs ...")
    summary = build_graphs_from_extractions(records, super_lenses, args.output_dir)

    print_final_summary(summary, args.output_dir)

    # ── Step 4: Cross-lens edges ──────────────────────────────────────────
    print(f"\n[final] Step 4: Building cross-lens edges ...")
    build_cross_lens_edges(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
