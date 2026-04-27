"""
Orchestrator CLI.

Usage examples:
  # Run on 20 entries from CultureBank only (smoke test):
  python src/main.py --datasets culturebank --sample 20

  # Run on all available datasets (CANDLE excluded — needs local path):
  python src/main.py --datasets arabculture diwali culturebank blend

  # Run on CANDLE with a local data path:
  python src/main.py --datasets candle --candle-path ./datasets/candle --sample 100

  # Full run on all supported datasets:
  python src/main.py --sample 500 --output-dir outputs/
"""
import argparse
import sys
import os

# Make src/ importable when running as `python src/main.py`
sys.path.insert(0, os.path.dirname(__file__))

from data_loaders import load_datasets, DATASET_NAMES
from extractor import extract_batch
from graph_builder import build_graph
from exporter import export_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cultural knowledge graph extraction pipeline."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        metavar="DATASET",
        help=f"Datasets to load. Choices: {DATASET_NAMES}. Default: all available.",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        metavar="N",
        help="Max entries per dataset (0 = no limit).",
    )
    parser.add_argument(
        "--candle-path", type=str, default="",
        metavar="PATH",
        help="Local directory for CANDLE data (required if 'candle' in --datasets).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        metavar="DIR",
        help="Directory for output files (default: outputs/).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-entry extraction results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load ---
    entries = list(load_datasets(
        names=args.datasets,
        limit=args.sample,
        candle_path=args.candle_path,
    ))
    if not entries:
        print("[main] No entries loaded. Check dataset names and paths.")
        sys.exit(1)
    print(f"[main] Loaded {len(entries)} entries total.")

    # --- Extract ---
    raw_nodes, raw_edges = extract_batch(
        entries,
        verbose=args.verbose,
    )

    # --- Build ---
    graph = build_graph(raw_nodes, raw_edges)

    # --- Export ---
    export_all(graph, output_dir=args.output_dir)

    print(f"\n[main] Done.")
    print(f"  Nodes : {len(graph.nodes)}")
    print(f"  Edges : {len(graph.edges)}")
    print(f"  Output: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
