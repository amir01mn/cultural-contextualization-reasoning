"""
Phases 2 & 3: Frequency analysis, clustering, and data-driven schema induction.

Takes raw_triples.jsonl → produces:
  - frequency_analysis.json
  - entity_clusters.json
  - relation_clusters.json
  - induced_schemas.json  (one schema tree per discovered lens)
"""
from __future__ import annotations
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Step 2a — Frequency counting
# ---------------------------------------------------------------------------

def count_frequencies(records: list[dict]) -> dict:
    entity_freq: Counter = Counter()
    relation_freq: Counter = Counter()
    pathway_freq: Counter = Counter()

    for rec in records:
        for t in rec.get("triples", []):
            h = t.get("head", "").strip()
            r = t.get("relation", "").strip()
            tl = t.get("tail", "").strip()
            if h:
                entity_freq[h] += 1
            if tl:
                entity_freq[tl] += 1
            if r:
                relation_freq[r] += 1
            if h and r and tl:
                pathway_freq[(h, r, tl)] += 1

    return {
        "entity_freq": dict(entity_freq.most_common()),
        "relation_freq": dict(relation_freq.most_common()),
        "top_pathways": [
            {"head": h, "relation": r, "tail": t, "count": c}
            for (h, r, t), c in pathway_freq.most_common(200)
        ],
        "total_triples": sum(len(rec.get("triples", [])) for rec in records),
        "total_entities": len(entity_freq),
        "total_relations": len(relation_freq),
    }


# ---------------------------------------------------------------------------
# Step 2b/2c — TF-IDF + DBSCAN clustering
# ---------------------------------------------------------------------------

def _cluster_labels(
    labels: list[str],
    freq: dict[str, int],
    min_cluster_size: int = 2,
    eps: float = 0.35,
) -> list[dict]:
    """
    Cluster a list of text labels using TF-IDF character n-grams + DBSCAN.
    Returns a list of cluster dicts sorted by total frequency descending.
    """
    if len(labels) < 2:
        return [{"canonical": l, "members": [l], "frequency": freq.get(l, 1)} for l in labels]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    try:
        X = vec.fit_transform(labels)
    except ValueError:
        return [{"canonical": l, "members": [l], "frequency": freq.get(l, 1)} for l in labels]

    sim_matrix = cosine_similarity(X)
    dist_matrix = 1.0 - sim_matrix
    dist_matrix = np.clip(dist_matrix, 0, None)

    db = DBSCAN(eps=eps, min_samples=min_cluster_size, metric="precomputed")
    cluster_ids = db.fit_predict(dist_matrix)

    # Group labels by cluster id
    clusters: dict[int, list[str]] = defaultdict(list)
    for label, cid in zip(labels, cluster_ids):
        clusters[cid].append(label)

    result = []
    for cid, members in clusters.items():
        total_freq = sum(freq.get(m, 1) for m in members)
        # Canonical = highest-frequency member
        canonical = max(members, key=lambda m: freq.get(m, 1))
        result.append({
            "canonical": canonical,
            "members": sorted(members, key=lambda m: freq.get(m, 1), reverse=True),
            "frequency": total_freq,
            "cluster_id": int(cid),
        })

    return sorted(result, key=lambda c: c["frequency"], reverse=True)


def cluster_entities(freq_data: dict, min_cluster_size: int = 2) -> list[dict]:
    entity_freq = freq_data["entity_freq"]
    # Only cluster entities that appear at least twice (noise reduction)
    labels = [e for e, c in entity_freq.items() if c >= 2]
    print(f"[inducer] Clustering {len(labels)} entity labels ...")
    clusters = _cluster_labels(labels, entity_freq, min_cluster_size=min_cluster_size)
    print(f"[inducer] → {len(clusters)} entity clusters")
    return clusters


def cluster_relations(freq_data: dict, min_cluster_size: int = 2) -> list[dict]:
    relation_freq = freq_data["relation_freq"]
    labels = [r for r, c in relation_freq.items() if c >= 2]
    print(f"[inducer] Clustering {len(labels)} relation labels ...")
    clusters = _cluster_labels(labels, relation_freq, min_cluster_size=min_cluster_size, eps=0.40)
    print(f"[inducer] → {len(clusters)} relation clusters")
    return clusters


# ---------------------------------------------------------------------------
# Step 2d — Frequency-driven level assignment (Jenks natural breaks)
# ---------------------------------------------------------------------------

def _jenks_breaks(values: list[float], n_classes: int) -> list[float]:
    """Simple Jenks natural breaks using Fisher's algorithm."""
    values = sorted(values)
    n = len(values)
    if n <= n_classes:
        return values
    mat1 = [[0.0] * (n + 1) for _ in range(n_classes + 1)]
    mat2 = [[float("inf")] * (n + 1) for _ in range(n_classes + 1)]
    for i in range(1, n + 1):
        mat1[1][i] = 1.0
        mat2[1][i] = 0.0
    for l in range(2, n_classes + 1):
        for m in range(l, n + 1):
            for j in range(l - 1, m):
                s1 = sum(values[j:m])
                s2 = sum(v ** 2 for v in values[j:m])
                cnt = m - j
                variance = s2 - (s1 ** 2) / cnt if cnt > 0 else 0
                val = mat2[l - 1][j] + variance
                if val < mat2[l][m]:
                    mat1[l][m] = j
                    mat2[l][m] = val
    breaks = []
    k = n
    for l in range(n_classes, 1, -1):
        idx = int(mat1[l][k]) - 1
        if idx < len(values):
            breaks.insert(0, values[idx])
        k = int(mat1[l][k])
    return breaks


def assign_levels(entity_clusters: list[dict]) -> list[dict]:
    """
    Assign a level to each entity cluster based on frequency.
    High frequency → low level number (more abstract).
    Uses Jenks natural breaks to find tier boundaries.
    """
    freqs = [c["frequency"] for c in entity_clusters]
    if not freqs:
        return entity_clusters

    log_freqs = [math.log(f + 1) for f in freqs]
    max_levels = min(6, len(set(freqs)))
    try:
        breaks = _jenks_breaks(log_freqs, max_levels)
    except Exception:
        breaks = []

    def get_level(log_f: float) -> int:
        for i, b in enumerate(breaks):
            if log_f <= b:
                return max_levels - i
        return 1

    total_levels = max_levels
    for c in entity_clusters:
        lf = math.log(c["frequency"] + 1)
        raw_level = get_level(lf)
        # Invert: highest freq → level 1, lowest → level N
        c["level"] = total_levels - raw_level + 1
    return entity_clusters


# ---------------------------------------------------------------------------
# Phase 3 — Schema induction: one schema tree per relation cluster (lens)
# ---------------------------------------------------------------------------

def induce_schemas(
    records: list[dict],
    entity_clusters: list[dict],
    relation_clusters: list[dict],
    min_lens_coverage: int = 10,
) -> list[dict]:
    """
    For each major relation cluster, build a schema tree describing:
    - lens_name: canonical relation label for this cluster
    - edge_types: all relation labels in this cluster
    - top_entity_pairs: most frequent (head_canonical, tail_canonical) pairs
    - coverage: number of triples that fall in this lens
    - suggested_levels: entity clusters sorted by freq in this lens
    """
    # Build lookup: relation label → relation cluster canonical
    rel_to_canonical: dict[str, str] = {}
    for rc in relation_clusters:
        for m in rc.get("members", [rc["canonical"]]):
            rel_to_canonical[m] = rc["canonical"]

    # Build lookup: entity label → entity cluster canonical
    ent_to_canonical: dict[str, str] = {}
    for ec in entity_clusters:
        for m in ec.get("members", [ec["canonical"]]):
            ent_to_canonical[m] = ec["canonical"]

    # Group triples by lens (relation cluster)
    lens_triples: dict[str, list[tuple]] = defaultdict(list)
    for rec in records:
        for t in rec.get("triples", []):
            r = t.get("relation", "")
            canonical_r = rel_to_canonical.get(r, r)
            h_canonical = ent_to_canonical.get(t.get("head", ""), t.get("head", ""))
            tl_canonical = ent_to_canonical.get(t.get("tail", ""), t.get("tail", ""))
            lens_triples[canonical_r].append((h_canonical, r, tl_canonical))

    schemas = []
    for rc in relation_clusters:
        lens_name = rc["canonical"]
        triples_in_lens = lens_triples.get(lens_name, [])
        coverage = len(triples_in_lens)

        if coverage < min_lens_coverage:
            continue

        pair_counts: Counter = Counter((h, tl) for h, _, tl in triples_in_lens)
        entity_counts: Counter = Counter()
        for h, _, tl in triples_in_lens:
            entity_counts[h] += 1
            entity_counts[tl] += 1

        # Entities in this lens, sorted by frequency → reveals natural levels
        lens_entities = [
            {"label": ent, "frequency": cnt}
            for ent, cnt in entity_counts.most_common(50)
        ]

        schemas.append({
            "lens_id": len(schemas),
            "lens_name": lens_name,
            "edge_types": rc.get("members", [lens_name])[:20],
            "coverage": coverage,
            "top_entity_pairs": [
                {"head": h, "tail": tl, "count": c}
                for (h, tl), c in pair_counts.most_common(30)
            ],
            "top_entities": lens_entities,
            "relation_cluster_frequency": rc["frequency"],
        })

    schemas.sort(key=lambda s: s["coverage"], reverse=True)
    print(f"[inducer] Induced {len(schemas)} lens schemas (min coverage={min_lens_coverage})")
    return schemas


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def run_induction(
    records: list[dict],
    output_dir: str,
    min_cluster_size: int = 2,
    min_lens_coverage: int = 10,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    print("[inducer] Counting frequencies ...")
    freq_data = count_frequencies(records)
    _save(freq_data, os.path.join(output_dir, "frequency_analysis.json"))
    print(f"  entities={freq_data['total_entities']}, relations={freq_data['total_relations']}, triples={freq_data['total_triples']}")

    print("[inducer] Clustering entities ...")
    entity_clusters = cluster_entities(freq_data, min_cluster_size)
    entity_clusters = assign_levels(entity_clusters)
    _save(entity_clusters, os.path.join(output_dir, "entity_clusters.json"))

    print("[inducer] Clustering relations ...")
    relation_clusters = cluster_relations(freq_data, min_cluster_size)
    _save(relation_clusters, os.path.join(output_dir, "relation_clusters.json"))

    print("[inducer] Inducing schemas ...")
    schemas = induce_schemas(records, entity_clusters, relation_clusters, min_lens_coverage)
    _save(schemas, os.path.join(output_dir, "induced_schemas.json"))

    return {
        "freq_data": freq_data,
        "entity_clusters": entity_clusters,
        "relation_clusters": relation_clusters,
        "schemas": schemas,
    }


def _save(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[inducer] Saved → {path}")
