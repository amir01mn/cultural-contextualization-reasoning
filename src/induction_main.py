"""
Orchestrator for the data-driven schema induction pipeline (Task 2).

Outputs go to outputs/induced/ — completely separate from Task 1 outputs/.

Usage:
  # Smoke test (10 entries from culturebank):
  python src/induction_main.py --datasets culturebank --sample 10

  # Skip mining if raw_triples.jsonl already exists:
  python src/induction_main.py --datasets culturebank --sample 100 --skip-mining

  # Full run on all datasets:
  python src/induction_main.py --datasets arabculture diwali culturebank blend candle \\
      --candle-path datasets/candle --sample 500
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loaders import load_datasets, DATASET_NAMES
from miner import mine_batch, load_triples
from schema_inducer import run_induction
from forest_builder import build_forest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data-driven cultural schema induction pipeline.")
    p.add_argument("--datasets", nargs="+", default=None, metavar="DATASET",
                   help=f"Datasets to load. Choices: {DATASET_NAMES}. Default: all available.")
    p.add_argument("--sample", type=int, default=0, metavar="N",
                   help="Max entries per dataset (0 = no limit).")
    p.add_argument("--candle-path", type=str, default="", metavar="PATH",
                   help="Local directory for CANDLE data.")
    p.add_argument("--output-dir", type=str, default="outputs/induced",
                   help="Output directory (default: outputs/induced/). Task 1 outputs/ untouched.")
    p.add_argument("--skip-mining", action="store_true",
                   help="Skip Phase 1 if raw_triples.jsonl already exists.")
    p.add_argument("--min-cluster-size", type=int, default=2,
                   help="Minimum cluster size for DBSCAN (default: 2).")
    p.add_argument("--min-lens-coverage", type=int, default=10,
                   help="Minimum triples for a lens to be included (default: 10).")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip visualization generation.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def print_summary(output_dir: str, schemas: list[dict], forest_summary: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("TASK 2 COMPLETE — Data-Driven Schema Induction")
    print("=" * 60)
    print(f"\nOutput directory : {os.path.abspath(output_dir)}/")
    print(f"Lenses discovered: {len(schemas)}")
    print(f"Graphs built     : {len(forest_summary)}")
    print(f"\nDiscovered lenses (by coverage):")
    for s in schemas[:15]:
        print(f"  [{s['lens_id']:2d}] {s['lens_name']:35s} coverage={s['coverage']:5d}")
    print(f"\nForest graphs:")
    for f in forest_summary:
        print(f"  lens_{f['lens_id']:02d} '{f['lens_name']:30s}' → {f['nodes']:4d} nodes, {f['edges']:4d} edges")
    print(f"\nCompare with Task 1:")
    print(f"  Task 1 (schema-first) → outputs/nodes.csv, outputs/cultural_graph.json")
    print(f"  Task 2 (data-driven)  → {output_dir}/forest/")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    triples_path = os.path.join(args.output_dir, "raw_triples.jsonl")

    # ── Phase 1: Mine ─────────────────────────────────────────────────────
    if args.skip_mining and os.path.exists(triples_path):
        print(f"[main] Skipping mining — loading existing {triples_path}")
        records = load_triples(triples_path)
        print(f"[main] Loaded {len(records)} existing records.")
    else:
        print("[main] Phase 1: Loading datasets ...")
        entries = list(load_datasets(
            names=args.datasets,
            limit=args.sample,
            candle_path=args.candle_path,
        ))
        if not entries:
            print("[main] No entries loaded. Check dataset names and paths.")
            sys.exit(1)
        print(f"[main] {len(entries)} entries loaded. Starting mining via Ollama ...")
        print("[main] Make sure Ollama is running: ollama serve")
        mine_batch(entries, triples_path, resume=True, verbose=args.verbose)
        records = load_triples(triples_path)

    if not records:
        print("[main] No triples found. Exiting.")
        sys.exit(1)

    total_triples = sum(len(r.get("triples", [])) for r in records)
    print(f"[main] {len(records)} records, {total_triples} raw triples.")

    # ── Phase 2 & 3: Cluster + Induce ────────────────────────────────────
    print("\n[main] Phase 2+3: Frequency analysis & schema induction ...")
    results = run_induction(
        records,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        min_lens_coverage=args.min_lens_coverage,
    )

    schemas = results["schemas"]
    if not schemas:
        print("[main] No schemas induced (too few triples). Try a larger --sample.")
        sys.exit(1)

    # ── Phase 4: Build forest ─────────────────────────────────────────────
    print("\n[main] Phase 4: Building forest graphs ...")
    forest_summary = build_forest(
        records=records,
        schemas=schemas,
        entity_clusters=results["entity_clusters"],
        relation_clusters=results["relation_clusters"],
        output_dir=args.output_dir,
        generate_viz=not args.no_viz,
    )

    print_summary(args.output_dir, schemas, forest_summary)


if __name__ == "__main__":
    main()
