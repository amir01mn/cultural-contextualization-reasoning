"""
Meta-clusters the 74 raw lenses into meaningful super-lenses.

Takes outputs/induced/relation_clusters.json (126 clusters, 74 lenses with coverage ≥ 10)
and runs a second-level TF-IDF + DBSCAN with a more permissive threshold so that
semantically similar lenses merge (e.g. "symbolizes" + "represents" + "depicts" → one super-lens).

The number of super-lenses is NOT pre-defined — it emerges from the data.

Output: outputs/induced/super_lenses.json
"""
from __future__ import annotations
import json
import os
from collections import defaultdict, Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _meta_cluster(
    schemas: list[dict],
    eps: float = 0.52,
    min_samples: int = 1,
) -> list[dict]:
    """
    Run a second-level TF-IDF + DBSCAN over the lens canonical names + their
    member relation phrases. More permissive eps → fewer, broader super-lenses.
    """
    if not schemas:
        return []

    # Build a rich text representation for each lens:
    # canonical name + all member relation phrases
    texts = []
    for s in schemas:
        members = s.get("edge_types", [s["lens_name"]])[:10]
        text = s["lens_name"] + " " + " ".join(members)
        texts.append(text)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    dist = np.clip(1.0 - sim, 0, None)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist)

    # Group schemas by cluster label
    groups: dict[int, list[dict]] = defaultdict(list)
    for schema, label in zip(schemas, labels):
        groups[label].append(schema)

    super_lenses = []
    for label, members in groups.items():
        total_coverage = sum(m["coverage"] for m in members)
        # Canonical = highest-coverage member
        canonical = max(members, key=lambda m: m["coverage"])

        # Collect all edge types across all members
        all_edge_types = []
        for m in members:
            all_edge_types.extend(m.get("edge_types", [m["lens_name"]]))
        edge_type_freq = Counter(all_edge_types)

        # Collect top entity pairs across all members
        pair_counts: Counter = Counter()
        for m in members:
            for pair in m.get("top_entity_pairs", []):
                key = (pair["head"], pair["tail"])
                pair_counts[key] += pair.get("count", 1)

        super_lenses.append({
            "super_lens_id": len(super_lenses),
            "name": canonical["lens_name"],
            "coverage": total_coverage,
            "constituent_lenses": [m["lens_name"] for m in members],
            "lens_ids": [m["lens_id"] for m in members],
            "edge_types": [et for et, _ in edge_type_freq.most_common(20)],
            "top_entity_pairs": [
                {"head": h, "tail": t, "count": c}
                for (h, t), c in pair_counts.most_common(20)
            ],
            "top_entities": canonical.get("top_entities", [])[:20],
        })

    super_lenses.sort(key=lambda s: s["coverage"], reverse=True)

    # Re-assign clean sequential IDs after sorting
    for i, sl in enumerate(super_lenses):
        sl["super_lens_id"] = i

    return super_lenses


def consolidate(
    induced_dir: str = "outputs/induced",
    output_path: str = "outputs/induced/super_lenses.json",
    eps: float = 0.52,
) -> list[dict]:
    schemas_path = os.path.join(induced_dir, "induced_schemas.json")
    if not os.path.exists(schemas_path):
        raise FileNotFoundError(f"Run induction_main.py first: {schemas_path} not found.")

    schemas = _load_json(schemas_path)
    print(f"[consolidator] {len(schemas)} raw lenses → meta-clustering (eps={eps}) ...")

    super_lenses = _meta_cluster(schemas, eps=eps)
    print(f"[consolidator] → {len(super_lenses)} super-lenses")

    for sl in super_lenses:
        members_str = ", ".join(f"'{c}'" for c in sl["constituent_lenses"][:5])
        if len(sl["constituent_lenses"]) > 5:
            members_str += f" + {len(sl['constituent_lenses'])-5} more"
        print(f"  [{sl['super_lens_id']:2d}] cov={sl['coverage']:4d} | '{sl['name']:30s}' ← [{members_str}]")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(super_lenses, f, ensure_ascii=False, indent=2)
    print(f"[consolidator] Saved → {output_path}")
    return super_lenses
