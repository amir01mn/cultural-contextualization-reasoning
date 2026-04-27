from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# Hierarchy levels: abstract → concrete
LEVELS = {
    0: "human",        # Universal root
    1: "domain",       # Cultural domain (religion, family, food, …)
    2: "group",        # Cultural group (Arab, Indian, Korean, …)
    3: "practice",     # Cultural practice / belief / norm
    4: "instance",     # Specific cue / action / artifact
}

RELATION_TYPES = {
    "influences",
    "balances",
    "honors",
    "contextualizes",
    "restricts",
    "manifests_in",
    "varies_by",
    "originates_from",
}

CULTURAL_DOMAINS = {
    "religion", "family", "gender", "food", "community",
    "language", "tradition", "education", "economy", "politics",
    "art", "health", "nature", "identity", "unknown",
}


@dataclass
class Node:
    label: str
    level: int                        # 0–4
    cultural_group: str = "general"
    domain: str = "unknown"
    source_dataset: str = ""
    frequency: int = 1
    node_id: Optional[int] = None

    def normalize(self) -> "Node":
        self.label = self.label.strip().lower()
        self.cultural_group = self.cultural_group.strip().lower()
        self.domain = self.domain.strip().lower()
        if self.domain not in CULTURAL_DOMAINS:
            self.domain = "unknown"
        return self

    @property
    def key(self) -> tuple:
        return (self.label, self.level)


@dataclass
class Edge:
    source_label: str
    target_label: str
    relation_type: str
    source_dataset: str = ""
    weight: int = 1

    def normalize(self) -> "Edge":
        self.source_label = self.source_label.strip().lower()
        self.target_label = self.target_label.strip().lower()
        self.relation_type = self.relation_type.strip().lower()
        if self.relation_type not in RELATION_TYPES:
            self.relation_type = "influences"  # fallback
        return self

    @property
    def key(self) -> tuple:
        return (self.source_label, self.target_label, self.relation_type)


@dataclass
class CulturalGraph:
    nodes: dict[tuple, Node] = field(default_factory=dict)   # key → Node
    edges: dict[tuple, Edge] = field(default_factory=dict)   # key → Edge
    _next_id: int = field(default=0, repr=False)

    def _next_node_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_node(self, node: Node) -> Node:
        node.normalize()
        if node.key in self.nodes:
            self.nodes[node.key].frequency += 1
        else:
            node.node_id = self._next_node_id()
            self.nodes[node.key] = node
        return self.nodes[node.key]

    def add_edge(self, edge: Edge) -> Edge:
        edge.normalize()
        if edge.key in self.edges:
            self.edges[edge.key].weight += 1
        else:
            self.edges[edge.key] = edge
        return self.edges[edge.key]

    def ensure_root(self) -> None:
        root = Node(label="human", level=0, cultural_group="universal",
                    domain="identity", source_dataset="schema")
        self.add_node(root)

    @property
    def node_list(self) -> list[Node]:
        return list(self.nodes.values())

    @property
    def edge_list(self) -> list[Edge]:
        return list(self.edges.values())

    def node_id_for(self, label: str, level: int) -> Optional[int]:
        node = self.nodes.get((label.strip().lower(), level))
        return node.node_id if node else None
