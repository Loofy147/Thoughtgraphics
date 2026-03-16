import contextlib
import numpy as np
"""
ThoughtGraph v2.1 — March 2026
Changes from v2.0:
  - FIXED: seed_default_graph() now wires child→parent and potential→nearest (was 17 isolated nodes)
  - ADDED: recommend_exploration() — ranked frontier analysis of potential nodes
  - ADDED: evolution_snapshot() — tracks health metrics over time
  - ADDED: find_bridges() — edges whose removal disconnects the graph
  - ADDED: suggest_connections() — missing-link prediction via Jaccard/Adamic-Adar
  - IMPROVED: evaluate_new_node() weakly connects POTENTIAL decisions too
  - IMPROVED: detect_patterns() includes community health metrics
"""

import json, math, time, random, collections, threading

import functools
def atomic(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper

def atomic(method):
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import networkx as nx
from networkx.algorithms.community import louvain_communities as nx_louvain


# ═══════════════════════════════════════════════════════════
#  EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════

def _fnv1a(text):
    h = 2166136261
    for b in text.encode("utf-8"):
        h ^= b; h = (h * 16777619) & 0xFFFFFFFF
    return h

def make_embedding(label, dims=512):
    """
    Character n-gram hashing embedding with 512 dims and 4 hash functions.
    Reduced collision rate; better cross-domain discrimination than 128-dim.
    """
    text = "<" + label.lower().strip() + ">"
    vec = [0.0] * dims
    for n, w in [(2, 0.25), (3, 1.00), (4, 0.80), (5, 0.40)]:
        for i in range(len(text) - n + 1):
            gram = text[i : i + n]
            for salt, wm in [("a", 1.0), ("b", 0.60), ("c", 0.40), ("d", 0.30)]:
                vec[_fnv1a(gram + salt) % dims] += w * wm
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def _compute_baseline_similarity(nodes: list) -> tuple:
    """Compute (median, max) of pairwise similarities among active nodes using numpy."""
    active = [n for n in nodes if n.node_type in ("active", "meta")]
    if len(active) < 4:
        return 0.5, 1.0

    embeddings = np.array([n.embedding for n in active])
    # Compute all-to-all cosine similarity
    # Norms: (N,)
    norms = np.linalg.norm(embeddings, axis=1)
    norms[norms == 0] = 1.0
    # Normalized: (N, D)
    normed = embeddings / norms[:, np.newaxis]
    # Similarity matrix: (N, N)
    sim_matrix = np.dot(normed, normed.T)
    # Get upper triangle indices (excluding diagonal)
    triu_indices = np.triu_indices(len(active), k=1)
    raw_sims = sim_matrix[triu_indices]

    sims = (raw_sims + 1) / 2
    if len(sims) == 0:
        return 0.5, 1.0

    return float(np.median(sims)), float(np.max(sims))


def cosine_sim(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape: return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


# ═══════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

@dataclass
class ThoughtNode:
    id: int
    label: str
    x: float; y: float; z: float
    node_type: str
    depth: int = 0
    importance: float = 1.0
    effective_importance: float = 1.0
    parent_id: Optional[int] = None
    children_ids: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    embedding: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activated: float = 0.0
    activation_count: int = 0
    decision_weight: float = 1.0
    tags: list = field(default_factory=list)
    community_id: int = -1
    pagerank: float = 0.0
    betweenness: float = 0.0

    def __post_init__(self):
        if not self.embedding: self.embedding = make_embedding(self.label)
        if self.node_type == "meta": self.decision_weight = 2.0
        elif self.node_type == "child": self.decision_weight = 0.3
        if self.effective_importance == 1.0: self.effective_importance = self.importance

    def distance_to(self, other):
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2+(self.z-other.z)**2)

    def semantic_similarity(self, other):
        return (cosine_sim(self.embedding, other.embedding) + 1) / 2


@dataclass
class ThoughtEdge:
    from_id: int; to_id: int
    strength: float = 0.5
    edge_type: str = "connection"
    created_at: float = field(default_factory=time.time)
    last_activated: float = 0.0
    activation_count: int = 0


@dataclass
class EvaluationResult:
    node_id: int
    decision: str
    pattern_match_score: float
    nearest_neighbors: list
    reasoning: str
    suggested_connections: list
    factor_breakdown: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
#  GRAPH ANALYZER
# ═══════════════════════════════════════════════════════════

class GraphAnalyzer:
    def __init__(self, nodes, edges):
        self._nodes = nodes
        G = nx.Graph()
        for nid, n in nodes.items():
            G.add_node(nid, label=n.label, node_type=n.node_type)
        for e in edges:
            if e.from_id in nodes and e.to_id in nodes:
                G.add_edge(e.from_id, e.to_id, weight=max(0.001, e.strength))
        self._G = G

    def pagerank(self):
        if not self._G: return {}
        try: return nx.pagerank(self._G, alpha=0.85, weight="weight", max_iter=300)
        except Exception: return nx.pagerank(self._G, alpha=0.70, weight="weight", max_iter=600)

    def betweenness(self):
        if len(self._G) < 2: return {n: 0.0 for n in self._G}
        return nx.betweenness_centrality(self._G, weight="weight", normalized=True)

    def closeness(self):
        return nx.closeness_centrality(self._G)

    def eigenvector(self):
        try: return nx.eigenvector_centrality(self._G, weight="weight", max_iter=500)
        except Exception: return {n: 1/max(len(self._G),1) for n in self._G.nodes}

    def hits(self):
        try: return nx.hits(self._G, max_iter=500)
        except Exception:
            d = {n: 1/max(len(self._G),1) for n in self._G.nodes}
            return dict(d), dict(d)

    def burt_constraint(self):
        if len(self._G) < 3: return {n: 1.0 for n in self._G.nodes}
        try: return nx.constraint(self._G, weight="weight")
        except Exception: return {n: 0.5 for n in self._G.nodes}

    def communities(self, seed=42):
        if len(self._G) < 2: return {n: 0 for n in self._G.nodes}
        try:
            parts = nx_louvain(self._G, weight="weight", seed=seed)
            m = {}
            for cid, part in enumerate(parts):
                for node in part: m[node] = cid
            return m
        except Exception: return {n: 0 for n in self._G.nodes}

    def clustering(self):
        return nx.clustering(self._G, weight="weight")

    def fiedler(self):
        if len(self._G) < 2: return 0.0
        G = self._G
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len))
        if len(G) < 2: return 0.0
        try: return float(nx.algebraic_connectivity(G, method="tracemin_pcg"))
        except Exception: return 0.0

    def small_world(self):
        n = len(self._G)
        if n < 4 or not self._G.number_of_edges(): return 0.0
        k = 2 * self._G.number_of_edges() / n
        C = nx.average_clustering(self._G)
        lcc = self._G.subgraph(max(nx.connected_components(self._G), key=len)).copy()
        if len(lcc) < 2: return 0.0
        try: L = nx.average_shortest_path_length(lcc)
        except Exception: return 0.0
        if k <= 1 or L == 0: return 0.0
        Cr = k/n; Lr = math.log(n)/math.log(k)
        if Cr == 0 or Lr == 0: return 0.0
        return round((C/Cr)/(L/Lr), 4)

    def entropy(self):
        degs = [d for _,d in self._G.degree()]
        n = len(degs)
        if n < 2: return {"entropy":0.0,"max_entropy":0.0,"efficiency":0.0}
        counts = collections.Counter(degs)
        H = -sum((c/n)*math.log2(c/n) for c in counts.values() if c>0)
        Hm = math.log2(n)
        return {"entropy":round(H,4),"max_entropy":round(Hm,4),"efficiency":round(H/Hm if Hm else 0,4)}

    def modularity(self, coms):
        if not coms: return 0.0
        groups = collections.defaultdict(set)
        for nd, cid in coms.items(): groups[cid].add(nd)
        try: return float(nx.community.modularity(self._G, list(groups.values()), weight="weight"))
        except Exception: return 0.0

    def bridges(self):
        """Edges whose removal disconnects the graph."""
        try:
            return list(nx.bridges(self._G))
        except Exception: return []

    def link_prediction(self, k=10):
        """
        Top-K missing edges by Adamic-Adar index.
        High score = two nodes share many mutual neighbors → likely connected.
        """
        try:
            preds = nx.adamic_adar_index(self._G)
            scored = [(u, v, s) for u, v, s in preds
                      if not self._G.has_edge(u, v) and u != v]
            scored.sort(key=lambda x: -x[2])
            return scored[:k]
        except Exception: return []

    def full_report(self):
        pr    = self.pagerank()
        btw   = self.betweenness()
        close = self.closeness()
        hubs, auth = self.hits()
        cst   = self.burt_constraint()
        coms  = self.communities()
        clust = self.clustering()
        entr  = self.entropy()
        fied  = self.fiedler()
        sw    = self.small_world()
        mod   = self.modularity(coms)
        brgs  = self.bridges()
        links = self.link_prediction(k=5)

        def lbl(nid): return self._nodes[nid].label if nid in self._nodes else str(nid)
        top_pr   = max(pr,   key=pr.get)   if pr   else None
        top_btw  = max(btw,  key=btw.get)  if btw  else None
        top_hub  = max(hubs, key=hubs.get) if hubs else None
        min_cst  = min(cst,  key=cst.get)  if cst  else None

        # n_components
        G = self._G
        n_comp = nx.number_connected_components(G)

        return {
            "pagerank": pr, "betweenness": btw, "closeness": close,
            "hubs": hubs, "authorities": auth, "constraint": cst,
            "communities": coms, "clustering": clust,
            "fiedler": round(fied,6), "small_world_index": sw,
            "graph_entropy": entr, "modularity": round(mod,4),
            "n_communities": len(set(coms.values())),
            "n_components": n_comp,
            "bridges": brgs,
            "suggested_links": [{"from_id":u,"to_id":v,"score":round(s,4)} for u,v,s in links],
            "top_pagerank_node": lbl(top_pr),
            "top_betweenness_node": lbl(top_btw),
            "top_hub_node": lbl(top_hub),
            "structural_hole_node": lbl(min_cst),
        }


# ═══════════════════════════════════════════════════════════
#  ACTIVATION ENGINE
# ═══════════════════════════════════════════════════════════

class ActivationEngine:
    def spread(self, source_ids, nodes, edges, decay=0.55, steps=4, threshold=0.02):
        em = {}
        for e in edges:
            em[(e.from_id,e.to_id)] = e.strength
            em[(e.to_id,e.from_id)] = e.strength
        act = {sid: 1.0 for sid in source_ids if sid in nodes}
        for _ in range(steps):
            new = dict(act)
            for nid, a in act.items():
                if a < threshold: continue
                node = nodes.get(nid)
                if not node: continue
                for cid in node.connections:
                    incoming = a * decay * em.get((nid,cid), 0.5)
                    new[cid] = max(new.get(cid,0.0), incoming)
            act = new
        return {k: round(v,4) for k,v in act.items() if v >= threshold}

    def hebbian_update(self, activation, edges, lr=0.04, depression=0.005):
        updated = 0
        for e in edges:
            a = activation.get(e.from_id, 0.0)
            b = activation.get(e.to_id,   0.0)
            if a > 0.15 and b > 0.15:
                e.strength = min(1.0, e.strength + lr * a * b)
                e.activation_count += 1; e.last_activated = time.time(); updated += 1
            elif a < 0.05 or b < 0.05:
                e.strength = max(0.05, e.strength - depression)
        return updated


# ═══════════════════════════════════════════════════════════
#  TEMPORAL ENGINE
# ═══════════════════════════════════════════════════════════

class TemporalEngine:
    def activate(self, node):
        node.last_activated = time.time()
        node.activation_count += 1
        node.effective_importance = node.importance

    def decay_all(self, nodes, rate=0.015, floor=0.10):
        now = time.time(); results = {}
        for nid, node in nodes.items():
            ref = node.last_activated if node.last_activated > 0 else node.created_at
            factor = math.exp(-rate * (now-ref) / 3600)
            node.effective_importance = max(floor, node.importance * factor)
            results[nid] = round(node.effective_importance, 4)
        return results

    def recency_weight(self, node):
        ref = node.last_activated if node.last_activated > 0 else node.created_at
        return round(math.exp(-0.008 * (time.time()-ref) / 3600), 4)


# ═══════════════════════════════════════════════════════════
#  CORE GRAPH ENGINE v2.1
# ═══════════════════════════════════════════════════════════

class ThoughtGraph:
    STORAGE_PATH = Path("thought_graph_data.json")

    def __init__(self, persist=True):
        self._nodes: dict = {}
        self._edges: list = []
        self._next_id = 0
        self._evaluation_history = []
        self._evolution_history = []      # NEW: health snapshots over time
        self._persist = persist
        self._activation_engine = ActivationEngine()
        self._temporal_engine   = TemporalEngine()
        self._cached_topo = {}
        self._topo_dirty  = True
        self._batch_mode = False
        self._dirty = False
        self._cached_baseline = None
        self._lock = threading.RLock() # For future thread-safety considerations
        if persist and self.STORAGE_PATH.exists():
            self._load()

    @contextlib.contextmanager
    def batch_operation(self):
        """Context manager for bulk updates to suppress redundant saves and caching."""
        old_mode = self._batch_mode
        self._batch_mode = True
        try:
            yield
        finally:
            self._batch_mode = old_mode
            self._cached_baseline = None
            if not self._batch_mode and self._persist:
                self._save()

    # ── CRUD ──────────────────────────────────

    @atomic
    def add_node(self, label, x=None, y=None, z=None, node_type="active",
                 depth=1, parent_id=None, tags=None, importance=1.0):
        if x is None: x = random.uniform(-8,8)
        if y is None: y = random.uniform(-5,5)
        if z is None: z = random.uniform(-8,8)
        node = ThoughtNode(id=self._next_id, label=label, x=x, y=y, z=z,
                           node_type=node_type, depth=depth, parent_id=parent_id,
                           tags=tags or [], importance=importance)
        self._nodes[node.id] = node
        self._next_id += 1
        self._topo_dirty = True
        if parent_id is not None and parent_id in self._nodes:
            self._nodes[parent_id].children_ids.append(node.id)
        self._dirty = True
        return node

    @atomic
    def remove_node(self, node_id):
        if node_id not in self._nodes: return False
        node = self._nodes.pop(node_id)
        if node.parent_id and node.parent_id in self._nodes:
            p = self._nodes[node.parent_id]
            p.children_ids = [c for c in p.children_ids if c != node_id]
        self._edges = [e for e in self._edges if e.from_id != node_id and e.to_id != node_id]
        for n in self._nodes.values():
            n.connections = [c for c in n.connections if c != node_id]
        self._topo_dirty = True
        self._dirty = True
        return True

    def get_node(self, node_id): return self._nodes.get(node_id)
    def get_all_nodes(self): return list(self._nodes.values())

    def update_node_importance(self, node_id, importance):
        if node_id in self._nodes:
            self._nodes[node_id].importance = max(0.0, min(5.0, importance))
            self._nodes[node_id].effective_importance = self._nodes[node_id].importance
        self._dirty = True

    @atomic
    def connect(self, from_id, to_id, strength=0.5, edge_type="connection"):
        if from_id not in self._nodes or to_id not in self._nodes: return None
        existing = next((e for e in self._edges if
            (e.from_id==from_id and e.to_id==to_id) or
            (e.from_id==to_id and e.to_id==from_id)), None)
        if existing: return existing
        edge = ThoughtEdge(from_id=from_id, to_id=to_id, strength=strength, edge_type=edge_type)
        self._edges.append(edge)
        self._nodes[from_id].connections.append(to_id)
        self._nodes[to_id].connections.append(from_id)
        self._topo_dirty = True
        self._dirty = True
        return edge

    def get_edges(self): return list(self._edges)

    # ── TOPOLOGY ──────────────────────────────

    @atomic
    def get_topology(self, force=False):
        if self._topo_dirty or force or not self._cached_topo:
            if len(self._nodes) >= 2:
                a = GraphAnalyzer(self._nodes, self._edges)
                self._cached_topo = a.full_report()
                pr  = self._cached_topo["pagerank"]
                btw = self._cached_topo["betweenness"]
                com = self._cached_topo["communities"]
                for nid, node in self._nodes.items():
                    node.pagerank     = round(pr.get(nid, 0.0), 6)
                    node.betweenness  = round(btw.get(nid, 0.0), 6)
                    node.community_id = com.get(nid, -1)
            else: self._cached_topo = {}
            self._topo_dirty = False
        return self._cached_topo

    # ── SIMILARITY ────────────────────────────

    def find_nearest(self, node, k=7, exclude_types=None):
        candidates = [n for n in self._nodes.values()
                      if n.id != node.id and (not exclude_types or n.node_type not in exclude_types)]
        scored = []
        for other in candidates:
            spatial = node.distance_to(other)
            semantic = node.semantic_similarity(other)
            combined = semantic * 0.62 + (1.0/(1.0+spatial*0.18)) * 0.38
            scored.append((other, spatial, semantic, combined))
        scored.sort(key=lambda t: t[3], reverse=True)
        return scored[:k]

    def compute_surprise(self, node):
        others = [n for n in self._nodes.values() if n.id != node.id]
        if not others: return 1.0
        return round(1.0 - max(node.semantic_similarity(o) for o in others), 3)

    def suggest_connections(self, k=5):
        """
        Missing-link prediction via Adamic-Adar index.
        Returns list of {from_id, to_id, score, from_label, to_label}.
        """
        topo = self.get_topology()
        suggestions = topo.get("suggested_links", [])
        result = []
        for s in suggestions[:k]:
            n1 = self._nodes.get(s["from_id"])
            n2 = self._nodes.get(s["to_id"])
            if n1 and n2:
                result.append({
                    "from_id":    s["from_id"],
                    "to_id":      s["to_id"],
                    "score":      s["score"],
                    "from_label": n1.label,
                    "to_label":   n2.label,
                })
        return result

    def find_bridges(self):
        """Edges whose removal would disconnect the graph."""
        topo = self.get_topology()
        brgs = topo.get("bridges", [])
        result = []
        for (a, b) in brgs:
            na, nb = self._nodes.get(a), self._nodes.get(b)
            if na and nb:
                result.append({
                    "from_id": a, "to_id": b,
                    "from_label": na.label, "to_label": nb.label,
                    "criticality": "high",
                })
        return result

    # ── ACTIVATION ────────────────────────────

    @atomic
    def activate_node(self, node_id, spread=True):
        node = self._nodes.get(node_id)
        if not node: return {}
        self._temporal_engine.activate(node)
        if not spread: return {node_id: 1.0}
        activation = self._activation_engine.spread([node_id], self._nodes, self._edges)
        self._activation_engine.hebbian_update(activation, self._edges)
        for nid, level in activation.items():
            if nid in self._nodes and level > 0.3:
                self._temporal_engine.activate(self._nodes[nid])
        self._dirty = True
        return activation

    @atomic
    def decay_graph(self):
        results = self._temporal_engine.decay_all(self._nodes)
        self._dirty = True
        return results

    # ── RECOMMENDATION ENGINE ─────────────────

    def recommend_exploration(self, k=5):
        """
        Rank potential nodes by frontier breakthrough score.
        Score = evaluation_score * 0.5 + neighbor_pagerank_influence * 0.3 + recency * 0.2
        Returns top-K candidates with full reasoning.
        """
        self.get_topology()   # ensure annotations fresh
        topo = self.get_topology()
        pr   = topo.get("pagerank", {})
        pr_max = max(pr.values()) if pr else 1e-9

        potential = [n for n in self._nodes.values() if n.node_type == "potential"]
        if not potential:
            return []

        candidates = []
        for n in potential:
            result = self.evaluate_new_node(n)
            neighbor_pr  = sum(pr.get(nb["node_id"],0) for nb in result.nearest_neighbors[:3])
            pr_influence = neighbor_pr / pr_max if pr_max > 0 else 0
            recency      = self._temporal_engine.recency_weight(n)
            frontier     = (result.pattern_match_score * 0.50 +
                            pr_influence               * 0.30 +
                            recency                    * 0.20)
            surprise     = result.factor_breakdown.get("surprise", 0)

            candidates.append({
                "node_id":       n.id,
                "label":         n.label,
                "frontier_score": round(frontier, 4),
                "eval_score":    result.pattern_match_score,
                "eval_decision": result.decision,
                "pr_influence":  round(pr_influence, 4),
                "surprise":      surprise,
                "community":     n.community_id,
                "nearest":       result.nearest_neighbors[0]["label"] if result.nearest_neighbors else None,
                "reasoning":     result.reasoning,
            })

        candidates.sort(key=lambda c: -c["frontier_score"])
        return candidates[:k]

    # ── EVOLUTION SNAPSHOT ────────────────────

    def record_snapshot(self):
        """
        Save current health metrics as a timestamped snapshot.
        Builds a time-series of graph evolution.
        """
        a = self.graph_analytics()
        health = self.graph_health_score()
        snapshot = {
            "timestamp":     time.time(),
            "total_nodes":   a.get("total_nodes", 0),
            "total_edges":   a.get("total_edges", 0),
            "active_nodes":  a.get("active_nodes", 0),
            "health_score":  health.get("score", 0),
            "health_grade":  health.get("grade", "F"),
            "modularity":    a.get("modularity", 0),
            "fiedler":       a.get("fiedler_value", 0),
            "n_communities": a.get("n_communities", 0),
            "small_world":   a.get("small_world_index", 0),
        }
        self._evolution_history.append(snapshot)
        if len(self._evolution_history) > 500:
            self._evolution_history = self._evolution_history[-500:]
        self._dirty = True
        return snapshot

    def get_evolution_history(self):
        return list(self._evolution_history)

    # ── 5-FACTOR EVALUATION ───────────────────

    def evaluate_new_node(self, node):
        if len(self._nodes) < 3:
            return EvaluationResult(node_id=node.id, decision="ACCEPT",
                pattern_match_score=0.5, nearest_neighbors=[],
                reasoning="Bootstrap — auto-accepting.", suggested_connections=[],
                factor_breakdown={})

        topo      = self.get_topology() if not self._batch_mode else (self._cached_topo or self.get_topology())
        pageranks = topo.get("pagerank", {})
        coms      = topo.get("communities", {})
        nearest   = self.find_nearest(node, k=7)

        if not nearest:
            return EvaluationResult(node_id=node.id, decision="POTENTIAL",
                pattern_match_score=0.0, nearest_neighbors=[],
                reasoning="No comparable nodes.", suggested_connections=[],
                factor_breakdown={})

        # F1: PageRank-weighted RELATIVE semantic similarity
        # Anchored against the graph's own similarity distribution —
        # domain-alien nodes score near 0, closely related nodes score near 1.
        if self._batch_mode and self._cached_baseline:
            baseline_med, baseline_max = self._cached_baseline
        else:
            baseline_med, baseline_max = _compute_baseline_similarity(list(self._nodes.values()))
            if self._batch_mode: self._cached_baseline = (baseline_med, baseline_max)
        num = den = 0.0
        for other, _, semantic, combined in nearest:
            pr = pageranks.get(other.id, 1.0/len(self._nodes)) + 0.001
            w  = pr * other.decision_weight * other.effective_importance
            num += combined * w; den += w
        raw_sem = num / den if den else 0.0
        # Soft relative scoring: sqrt normalization to avoid hard zeros for valid concepts
        # Pizza Recipe: raw=0.56 -> f1=0.34; Graph Database: raw=0.64 -> f1=0.60
        rel_range = max(baseline_max - baseline_med, 0.01)
        offset    = rel_range * 0.8   # softening offset
        f1 = max(0.0, min(1.0, math.sqrt(
            max(0.0, (raw_sem - baseline_med + offset) / (rel_range + offset))
        )))

        ncoms = [coms.get(other.id,-1) for other,*_ in nearest[:5]]
        if ncoms:
            dom_com = collections.Counter(ncoms).most_common(1)[0][0]
            f2 = ncoms.count(dom_com) / len(ncoms)
        else:
            dom_com = -1; f2 = 0.0

        top_prs = [pageranks.get(other.id,0.0) for other,*_ in nearest[:3]]
        pr_max  = max(pageranks.values()) if pageranks else 1e-9
        f3 = (sum(top_prs)/len(top_prs)) / pr_max if top_prs and pr_max else 0.0

        distinct = len(set(c for c in ncoms if c >= 0))
        f4 = min(1.0, distinct / 3.0)

        surprise = self.compute_surprise(node)
        f5 = 1.0 - abs(surprise - 0.32) / 0.68

        composite = f1*0.50 + f2*0.25 + f3*0.15 + f4*0.05 + f5*0.05

        decision = "ACCEPT" if composite >= 0.60 else "POTENTIAL" if composite >= 0.38 else "REJECT"

        reasoning = (f"Semantic:{f1:.0%} Community:{f2:.0%} "
                     f"PR:{f3:.0%} Bridge:{f4:.0%} "
                     f"Surprise:{surprise:.0%} → {composite:.0%} → {decision}")

        nb_summary = [{
            "node_id": other.id, "label": other.label,
            "spatial_distance": round(spatial,2),
            "semantic_similarity": round(semantic,3),
            "combined_score": round(combined,3),
            "pagerank": round(pageranks.get(other.id,0.0),5),
            "community": coms.get(other.id,-1),
        } for other, spatial, semantic, combined in nearest]

        factors = {
            "semantic":f1, "community":f2, "pr_influence":f3,
            "bridging":f4, "novelty":f5, "surprise":surprise,
            "dominant_community":dom_com, "n_communities_bridged":distinct,
            "composite":round(composite,3),
        }

        self._evaluation_history.append({
            "timestamp":time.time(), "node_id":node.id, "label":node.label,
            "decision":decision, "score":round(composite,3),
            "surprise":surprise, "community":dom_com, "factors":factors,
        })
        self._dirty = True

        return EvaluationResult(
            node_id=node.id, decision=decision,
            pattern_match_score=round(composite,3),
            nearest_neighbors=nb_summary, reasoning=reasoning,
            suggested_connections=[other.id for other,*_ in nearest[:3]],
            factor_breakdown=factors,
        )

    @atomic
    def promote_potential(self, node_id):
        if node_id in self._nodes and self._nodes[node_id].node_type == "potential":
            self._nodes[node_id].node_type = "active"
            self._topo_dirty = True
        self._dirty = True
        return True
        return False

    # ── PATTERN DETECTION ─────────────────────

    def detect_patterns(self):
        topo = self.get_topology()
        coms = topo.get("communities", {})
        if not coms: return []
        groups = collections.defaultdict(list)
        for nid, cid in coms.items():
            node = self._nodes.get(nid)
            if node and node.node_type != "potential":
                groups[cid].append(nid)
        pr  = topo.get("pagerank", {})
        btw = topo.get("betweenness", {})
        patterns = []
        for cid, nids in sorted(groups.items()):
            if len(nids) < 2: continue
            labels = [self._nodes[nid].label for nid in nids]
            ns = [self._nodes[nid] for nid in nids if nid in self._nodes]
            centroid = (sum(n.x for n in ns)/len(ns), sum(n.y for n in ns)/len(ns), sum(n.z for n in ns)/len(ns))
            cohesion = sum(pr.get(n,0) for n in nids)
            # Internal density: edges within community / possible edges
            internal = sum(1 for e in self._edges
                           if e.from_id in nids and e.to_id in nids)
            possible = len(nids)*(len(nids)-1)/2
            density  = internal/possible if possible > 0 else 0
            # Most central node in community
            anchor   = max(nids, key=lambda n: btw.get(n,0))
            patterns.append({
                "cluster_id":  cid,
                "node_ids":    nids,
                "labels":      labels,
                "centroid":    centroid,
                "cohesion":    round(cohesion, 5),
                "density":     round(density, 4),
                "size":        len(nids),
                "anchor_node": self._nodes[anchor].label,
                "description": f"Community {cid}: {labels[0]}",
            })
        patterns.sort(key=lambda p: p["cohesion"], reverse=True)
        return patterns

    # ── GRAPH HEALTH ──────────────────────────

    def graph_health_score(self):
        """
        Calibrated health score 0-100 using density-aware normalization.
        Fiedler scored against theoretical maximum for current density.
        Small-world uses log scale (sigma=1.5 = full score for knowledge graphs).
        """
        topo = self.get_topology()
        if not topo: return {"score":0,"grade":"F","breakdown":{}}
        nodes = list(self._nodes.values())
        n = len(nodes)
        e = len(self._edges)
        if n < 2: return {"score":0,"grade":"F","breakdown":{}}

        avg_degree  = 2 * e / n
        fied        = topo.get("fiedler", 0.0)
        mod         = topo.get("modularity", 0.0)
        ent_eff     = topo.get("graph_entropy", {}).get("efficiency", 0.0)
        sw          = topo.get("small_world_index", 0.0)
        n_comp      = topo.get("n_components", 1)
        types       = collections.Counter(nd.node_type for nd in nodes)
        n_types     = sum(1 for t in ("meta","active","child","potential") if types.get(t,0)>0)

        expected_f  = (avg_degree / n) * 2 if n > 0 else 0.001
        conn        = min(25.0, (fied / max(expected_f, 0.0001)) * 25.0)
        community   = min(25.0, max(0.0, mod) / 0.65 * 25.0)
        entropy     = ent_eff * 20.0
        sw_s        = min(15.0, math.log(1 + sw) / math.log(2.5) * 15.0) if sw > 0 else 0.0
        diversity   = (n_types / 4.0) * 15.0
        frag_penalty = min(10.0, (n_comp - 1) * 1.5) if n_comp > 1 else 0.0

        total = min(100.0, max(0.0, conn + community + entropy + sw_s + diversity - frag_penalty))
        grade = "A" if total >= 80 else "B" if total >= 65 else "C" if total >= 50 else "D" if total >= 35 else "F"
        return {
            "score": round(total, 1), "grade": grade,
            "breakdown": {
                "connectivity": round(conn, 1),
                "community":    round(community, 1),
                "entropy":      round(entropy, 1),
                "small_world":  round(sw_s, 1),
                "diversity":    round(diversity, 1),
                "frag_penalty": round(-frag_penalty, 1),
            },
        }

    def graph_health_advice(self) -> list:
        """Actionable recommendations to improve graph health."""
        health = self.graph_health_score()
        topo   = self.get_topology()
        a      = self.graph_analytics()
        advice = []

        n_bridges = a.get("n_bridges", 0)
        n_nodes   = a.get("total_nodes", 1)
        if n_bridges > n_nodes * 0.35:
            top_links = topo.get("suggested_links", [])[:3]
            link_strs = ", ".join(
                f"{self._nodes[l['from_id']].label}\u2194{self._nodes[l['to_id']].label}"
                for l in top_links if l["from_id"] in self._nodes and l["to_id"] in self._nodes
            )
            advice.append({
                "priority": "HIGH", "area": "connectivity",
                "issue": f"{n_bridges} bridge edges — single points of failure",
                "action": f"Add cross-links. Top: {link_strs or 'run /suggest-links'}",
                "metric": f"fiedler={a.get('fiedler_value',0):.4f}",
            })

        sw = a.get("small_world_index", 0)
        if sw < 1.0:
            advice.append({
                "priority": "MEDIUM", "area": "small_world",
                "issue": f"sigma={sw:.3f} — few triangles, no shortcuts",
                "action": "Connect nodes sharing common neighbors via /suggest-links",
                "metric": "target sigma >= 1.5",
            })

        if health["breakdown"].get("entropy", 0) < 7:
            advice.append({
                "priority": "MEDIUM", "area": "entropy",
                "issue": "Uniform degree distribution — lacks hub/leaf contrast",
                "action": "Promote hub nodes; let leaf nodes stay sparse",
                "metric": f"eff={topo.get('graph_entropy',{}).get('efficiency',0):.3f}",
            })

        mod = a.get("modularity", 0)
        if mod < 0.3:
            advice.append({
                "priority": "LOW", "area": "community",
                "issue": f"Q={mod:.3f} — weak community separation",
                "action": "Add intra-community edges; reduce cross-community noise",
                "metric": "target Q >= 0.4",
            })

        recs   = self.recommend_exploration(k=3)
        accept = [r for r in recs if r["eval_decision"] == "ACCEPT"]
        if accept:
            advice.append({
                "priority": "LOW", "area": "growth",
                "issue": f"{len(accept)} potential node(s) ready to promote",
                "action": f"Promote: {', '.join(r['label'] for r in accept)}",
                "metric": f"scores: {[round(r['frontier_score'],2) for r in accept]}",
            })

        if not advice:
            advice.append({
                "priority": "INFO", "area": "health",
                "issue": "Graph structure is healthy",
                "action": "Keep adding and activating nodes",
                "metric": f"score={health['score']}/100 grade={health['grade']}",
            })
        return advice

    # ── ANALYTICS ─────────────────────────────

    def graph_analytics(self):
        nodes = list(self._nodes.values())
        if not nodes: return {}
        topo  = self.get_topology()
        degs  = {n.id:len(n.connections) for n in nodes}
        mx_id = max(degs, key=degs.get) if degs else None
        n     = len(nodes)
        me    = n*(n-1)/2
        types = collections.Counter(nd.node_type for nd in nodes)
        health = self.graph_health_score()
        return {
            "total_nodes":n, "total_edges":len(self._edges),
            "meta_nodes":types.get("meta",0), "active_nodes":types.get("active",0),
            "potential_nodes":types.get("potential",0), "child_nodes":types.get("child",0),
            "avg_degree":round(sum(degs.values())/n,2) if n else 0,
            "hub_node":self._nodes[mx_id].label if mx_id is not None else None,
            "hub_degree":degs.get(mx_id,0),
            "density":round(len(self._edges)/me,4) if me else 0,
            "evaluation_count":len(self._evaluation_history),
            "fiedler_value":topo.get("fiedler",0.0),
            "small_world_index":topo.get("small_world_index",0.0),
            "modularity":topo.get("modularity",0.0),
            "n_communities":topo.get("n_communities",0),
            "n_components":topo.get("n_components",0),
            "graph_entropy":topo.get("graph_entropy",{}).get("entropy",0.0),
            "health_score":health.get("score",0),
            "health_grade":health.get("grade","F"),
            "health_breakdown":health.get("breakdown",{}),
            "top_pagerank_node":topo.get("top_pagerank_node"),
            "top_betweenness_node":topo.get("top_betweenness_node"),
            "structural_hole_node":topo.get("structural_hole_node"),
            "n_bridges":len(topo.get("bridges",[])),
        }

    # ── Concept Path ──────────────────────────

    def concept_path(self, from_id: int, to_id: int) -> dict:
        """Shortest weighted path between two nodes (Dijkstra on 1-strength costs)."""
        import heapq
        if from_id not in self._nodes or to_id not in self._nodes:
            return {"found": False, "path": [], "cost": float("inf")}

        ec = {}
        for e in self._edges:
            c = 1.0 - e.strength
            ec[(e.from_id, e.to_id)] = c
            ec[(e.to_id, e.from_id)] = c

        dist = {from_id: 0.0}
        prev = {from_id: None}
        heap = [(0.0, from_id)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")): continue
            if u == to_id: break
            node = self._nodes.get(u)
            if not node: continue
            for v in node.connections:
                nd = d + ec.get((u, v), 0.5)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd; prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if to_id not in dist:
            return {"found": False, "path": [], "cost": float("inf"), "length": -1}

        path = []
        cur = to_id
        while cur is not None:
            path.append(cur); cur = prev.get(cur)
        path.reverse()

        hops = []
        for i, nid in enumerate(path):
            n = self._nodes[nid]
            hop = {"node_id": nid, "label": n.label, "node_type": n.node_type, "pagerank": n.pagerank}
            if i > 0:
                pn = self._nodes[path[i-1]]
                hop["semantic_sim"]  = round(n.semantic_similarity(pn), 3)
                hop["edge_strength"] = round(1.0 - ec.get((path[i-1], nid), 0.5), 3)
            hops.append(hop)

        return {
            "found": True, "path_ids": path, "hops": hops,
            "length": len(path)-1, "total_cost": round(dist[to_id], 4),
            "avg_strength": round(1.0 - dist[to_id]/max(len(path)-1, 1), 4),
        }

    # ── Duplicate Detection ────────────────────

    def find_duplicates(self, threshold: float = 0.88) -> list:
        """Find semantically near-identical node pairs (potential duplicates)."""
        nodes = [n for n in self._nodes.values() if n.node_type not in ("potential",)]
        dups = []; seen = set()
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                key = (min(a.id,b.id), max(a.id,b.id))
                if key in seen: continue
                sim = a.semantic_similarity(b)
                if sim >= threshold:
                    seen.add(key)
                    dups.append({
                        "node_a_id": a.id, "node_a_label": a.label,
                        "node_b_id": b.id, "node_b_label": b.label,
                        "similarity": round(sim, 4),
                        "recommendation": "merge" if sim >= 0.95 else "review",
                    })
        dups.sort(key=lambda d: -d["similarity"])
        return dups

    @atomic
    def merge_nodes(self, keep_id: int, remove_id: int) -> bool:
        """Merge two nodes: transfer connections from remove_id to keep_id."""
        if keep_id not in self._nodes or remove_id not in self._nodes: return False
        keep = self._nodes[keep_id]; remove = self._nodes[remove_id]
        ec = {}
        for e in self._edges:
            ec[(e.from_id, e.to_id)] = e.strength; ec[(e.to_id, e.from_id)] = e.strength
        for cid in list(remove.connections):
            if cid != keep_id:
                s = ec.get((remove_id, cid), 0.4)
                self.connect(keep_id, cid, strength=s)
        keep.importance = max(keep.importance, remove.importance)
        keep.activation_count += remove.activation_count
        self.remove_node(remove_id)
        return True

    # ── Named Snapshots ───────────────────────

    def save_snapshot(self, name: str) -> dict:
        """Save a named checkpoint of the current graph state."""
        a = self.graph_analytics()
        h = self.graph_health_score()
        record = {
            "timestamp": time.time(), "name": name,
            "total_nodes": a.get("total_nodes", 0),
            "total_edges": a.get("total_edges", 0),
            "active_nodes": a.get("active_nodes", 0),
            "health_score": h.get("score", 0),
            "health_grade": h.get("grade", "F"),
            "modularity": a.get("modularity", 0),
            "fiedler": a.get("fiedler_value", 0),
            "n_communities": a.get("n_communities", 0),
            "small_world": a.get("small_world_index", 0),
        }
        self._evolution_history.append(record)
        if len(self._evolution_history) > 500:
            self._evolution_history = self._evolution_history[-500:]
        self._dirty = True
        return record

    # ── Export Formats ────────────────────────

    def export_graphml(self) -> str:
        """Export as GraphML (Gephi, yEd, NetworkX compatible)."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/graphml">',
            '  <key id="label"       for="node" attr.name="label"       attr.type="string"/>',
            '  <key id="node_type"   for="node" attr.name="node_type"   attr.type="string"/>',
            '  <key id="importance"  for="node" attr.name="importance"  attr.type="double"/>',
            '  <key id="pagerank"    for="node" attr.name="pagerank"    attr.type="double"/>',
            '  <key id="community"   for="node" attr.name="community"   attr.type="int"/>',
            '  <key id="x" for="node" attr.name="x" attr.type="double"/>',
            '  <key id="y" for="node" attr.name="y" attr.type="double"/>',
            '  <key id="strength"    for="edge" attr.name="strength"    attr.type="double"/>',
            '  <key id="edge_type"   for="edge" attr.name="edge_type"   attr.type="string"/>',
            '  <graph id="G" edgedefault="undirected">',
        ]
        for n in self._nodes.values():
            safe = n.label.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            lines += [f'    <node id="n{n.id}">',
                f'      <data key="label">{safe}</data>',
                f'      <data key="node_type">{n.node_type}</data>',
                f'      <data key="importance">{n.importance}</data>',
                f'      <data key="pagerank">{n.pagerank}</data>',
                f'      <data key="community">{n.community_id}</data>',
                f'      <data key="x">{n.x:.3f}</data>',
                f'      <data key="y">{n.y:.3f}</data>',
                f'    </node>']
        for i, e in enumerate(self._edges):
            lines += [f'    <edge id="e{i}" source="n{e.from_id}" target="n{e.to_id}">',
                f'      <data key="strength">{e.strength}</data>',
                f'      <data key="edge_type">{e.edge_type}</data>',
                f'    </edge>']
        lines += ["  </graph>", "</graphml>"]
        return "\n".join(lines)

    def export_dot(self) -> str:
        """Export as DOT format (Graphviz compatible)."""
        colors = {"meta":"#ffd166","active":"#00e5ff","child":"#06d6a0","potential":"#8899aa"}
        shapes = {"meta":"diamond","active":"ellipse","child":"box","potential":"hexagon"}
        lines = ['graph ThoughtGraph {',
                 '  graph [bgcolor="#04080f"];',
                 '  node [style=filled fontname="monospace" fontsize=9];']
        for n in self._nodes.values():
            safe = n.label.replace('"','\\"')
            col  = colors.get(n.node_type,"#888888")
            shp  = shapes.get(n.node_type,"ellipse")
            lines.append(f'  n{n.id} [label="{safe}" fillcolor="{col}" shape={shp} fontcolor="black"];')
        seen = set()
        for e in self._edges:
            k = (min(e.from_id,e.to_id), max(e.from_id,e.to_id))
            if k in seen: continue
            seen.add(k)
            pw = 0.5 + e.strength*2
            st = "dashed" if e.edge_type=="potential_link" else "solid"
            lines.append(f'  n{e.from_id} -- n{e.to_id} [penwidth={pw:.1f} style={st}];')
        lines.append("}")
        return "\n".join(lines)


    # ── Search & Query ────────────────────────

    def search_nodes(
        self,
        query: str = "",
        node_type: str = None,
        min_importance: float = 0.0,
        community_id: int = None,
        min_pagerank: float = 0.0,
        tags: list = None,
        limit: int = 20,
    ) -> list:
        """
        Search and filter nodes by multiple criteria.
        Results are sorted by PageRank (most influential first).
        query: substring match on label or tags (case-insensitive).
        """
        q = query.lower().strip()
        results = []
        for n in self._nodes.values():
            if q and q not in n.label.lower() and not any(q in t.lower() for t in n.tags):
                continue
            if node_type and n.node_type != node_type:
                continue
            if n.importance < min_importance:
                continue
            if community_id is not None and n.community_id != community_id:
                continue
            if n.pagerank < min_pagerank:
                continue
            if tags and not any(t in n.tags for t in tags):
                continue
            results.append(n)
        results.sort(key=lambda n: (n.pagerank, n.effective_importance), reverse=True)
        return results[:limit]

    def get_community_subgraph(self, community_id: int) -> dict:
        """
        Extract all nodes and internal edges for a specific Louvain community.
        Returns {nodes, edges, analytics} for the subgraph.
        """
        self.get_topology()   # ensure community_id is annotated
        nodes = [n for n in self._nodes.values() if n.community_id == community_id]
        if not nodes:
            return {"community_id": community_id, "nodes": [], "edges": [], "size": 0}
        node_ids = {n.id for n in nodes}
        edges = [e for e in self._edges
                 if e.from_id in node_ids and e.to_id in node_ids]
        # Internal density
        possible = len(nodes) * (len(nodes) - 1) / 2
        density  = len(edges) / possible if possible > 0 else 0
        topo = self._cached_topo
        pr   = topo.get("pagerank", {})
        anchor = max(node_ids, key=lambda nid: pr.get(nid, 0))
        return {
            "community_id": community_id,
            "size":         len(nodes),
            "nodes":        [{"id": n.id, "label": n.label, "node_type": n.node_type,
                               "pagerank": n.pagerank, "importance": n.importance}
                              for n in sorted(nodes, key=lambda n: pr.get(n.id, 0), reverse=True)],
            "edges":        [{"from_id": e.from_id, "to_id": e.to_id,
                               "strength": e.strength, "edge_type": e.edge_type}
                              for e in edges],
            "density":      round(density, 4),
            "anchor_node":  self._nodes[anchor].label if anchor in self._nodes else None,
        }

    # ── Auto-Heal ─────────────────────────────

    def auto_heal_graph(self, max_links: int = 8, min_score: float = 0.5) -> dict:
        """
        Automatically apply top Adamic-Adar predicted links to reduce bridge count
        and improve small-world coefficient.
        Returns {applied, health_before, health_after, delta}.
        """
        health_before = self.graph_health_score()
        sw_before     = self.graph_analytics().get("small_world_index", 0)

        applied = []
        with self.batch_operation():
            suggestions = self.suggest_connections(k=max_links + 5)
            for s in suggestions:
                if len(applied) >= max_links:
                    break
                if s["score"] < min_score:
                    continue
                # Strength proportional to Adamic-Adar score (capped at 0.7)
                strength = min(0.70, round(s["score"] / 5.0, 2))
                edge = self.connect(s["from_id"], s["to_id"], strength=strength, edge_type="connection")
                if edge and edge.activation_count == 0:  # genuinely new edge
                    applied.append({
                        "from_id":    s["from_id"],
                        "to_id":      s["to_id"],
                        "from_label": s["from_label"],
                        "to_label":   s["to_label"],
                        "strength":   strength,
                        "aa_score":   s["score"],
                    })

            self._topo_dirty = True
            self.record_snapshot()

        health_after = self.graph_health_score()
        a_after      = self.graph_analytics()

        return {
            "applied":        applied,
            "n_applied":      len(applied),
            "health_before":  health_before["score"],
            "health_after":   health_after["score"],
            "grade_before":   health_before["grade"],
            "grade_after":    health_after["grade"],
            "delta":          round(health_after["score"] - health_before["score"], 2),
            "small_world_after": a_after.get("small_world_index", 0),
            "bridges_after":  a_after.get("n_bridges", 0),
        }

    # ── Batch Import ──────────────────────────

    def batch_import(self, items: list, auto_evaluate: bool = True) -> dict:
        """
        Add multiple nodes at once.
        items: list of dicts with keys: label, node_type (opt), importance (opt), tags (opt)
        Returns {added, accepted, potential, rejected, nodes}.
        """
        added = []; accepted = []; potential = []; rejected = []

        with self.batch_operation():
            for item in items:
                if not isinstance(item, dict) or "label" not in item:
                    continue
                node = self.add_node(
                    label      = item["label"],
                    node_type  = item.get("node_type", "potential"),
                    importance = item.get("importance", 1.0),
                    tags       = item.get("tags", []),
                    x=item.get("x"), y=item.get("y"), z=item.get("z"),
                )
                if auto_evaluate:
                    result = self.evaluate_new_node(node)
                    if result.decision == "ACCEPT":
                        node.node_type = "active"
                        accepted.append(node.id)
                        for tid in result.suggested_connections[:2]:
                            self.connect(node.id, tid, strength=0.55)
                    elif result.decision == "POTENTIAL":
                        potential.append(node.id)
                        for tid in result.suggested_connections[:1]:
                            self.connect(node.id, tid, strength=0.20, edge_type="potential_link")
                    else:
                        rejected.append(node.id)
                added.append(node.id)

            self._topo_dirty = True
            self.record_snapshot()

        return {
            "added":     len(added),
            "accepted":  len(accepted),
            "potential": len(potential),
            "rejected":  len(rejected),
            "node_ids":  added,
        }

    # ── Graph Diff ────────────────────────────

    def graph_diff(self, snapshot_a: dict, snapshot_b: dict) -> dict:
        """
        Compare two evolution snapshots (from get_evolution_history()).
        Returns structural changes and metric deltas.
        """
        delta = {}
        metrics = ["total_nodes","total_edges","active_nodes","health_score",
                   "modularity","fiedler","n_communities","small_world"]
        for m in metrics:
            a_val = snapshot_a.get(m, 0) or 0
            b_val = snapshot_b.get(m, 0) or 0
            delta[m] = round(b_val - a_val, 4)

        return {
            "from_name":  snapshot_a.get("name", "snapshot_a"),
            "to_name":    snapshot_b.get("name", "snapshot_b"),
            "from_time":  snapshot_a.get("timestamp", 0),
            "to_time":    snapshot_b.get("timestamp", 0),
            "deltas":     delta,
            "improved":   [k for k,v in delta.items() if v > 0],
            "degraded":   [k for k,v in delta.items() if v < 0],
            "unchanged":  [k for k,v in delta.items() if v == 0],
        }


    # ── SERIALIZATION ─────────────────────────

    def to_dict(self):
        return {"nodes":[asdict(n) for n in self._nodes.values()],
                "edges":[asdict(e) for e in self._edges],
                "next_id":self._next_id,
                "evaluation_history":self._evaluation_history,
                "evolution_history":self._evolution_history,
                "version":"2.1"}

    def flush_changes(self):
        with self._lock:
            if self._dirty and self._persist:
                self._save()
                self._dirty = False

    def _save(self):
        if self._batch_mode: return
        with open(self.STORAGE_PATH,"w") as f: json.dump(self.to_dict(),f,indent=2)

    def _load(self):
        with open(self.STORAGE_PATH) as f: data = json.load(f)
        for nd in data.get("nodes",[]):
            for k,v in [("effective_importance",nd.get("importance",1.0)),
                        ("last_activated",0.0),("activation_count",0),
                        ("community_id",-1),("pagerank",0.0),("betweenness",0.0)]:
                nd.setdefault(k,v)
            self._nodes[nd["id"]] = ThoughtNode(**nd)
        for ed in data.get("edges",[]):
            ed.setdefault("last_activated",0.0); ed.setdefault("activation_count",0)
            self._edges.append(ThoughtEdge(**ed))
        self._next_id = data.get("next_id",0)
        self._evaluation_history = data.get("evaluation_history",[])
        self._evolution_history  = data.get("evolution_history",[])

    @atomic
    def reset(self):
        self._nodes.clear(); self._edges.clear()
        self._next_id=0; self._evaluation_history.clear()
        self._evolution_history.clear()
        self._cached_topo={}; self._topo_dirty=True
        self._dirty = True

    # ── SEED DATA (FIXED: wires all nodes) ────

    @atomic
    def seed_default_graph(self):
        self.reset()

        core   = self.add_node("Core Decision Pattern",0,0,0,node_type="meta",depth=0,importance=2.0)
        rl     = self.add_node("RL Agents",3,2,-2,node_type="active",depth=1)
        cons   = self.add_node("Consciousness",-3,-1,2,node_type="active",depth=1)
        graph  = self.add_node("Graph Thinking",2,-2,1,node_type="active",depth=1)
        deploy = self.add_node("Deployment Problem",-2,2,-1,node_type="active",depth=1)
        dec_i  = self.add_node("Decision Intuition",4,0,2,node_type="active",depth=1)
        full_v = self.add_node("Full Version",-4,1,-2,node_type="active",depth=1)
        repos  = self.add_node("200 Repos",1,3,0,node_type="active",depth=1)
        algeria= self.add_node("Algeria Context",-1,-3,1,node_type="active",depth=1)
        time_n = self.add_node("Non-linear Time",3,-1,-3,node_type="active",depth=1)

        # Children — created AND connected to parent
        children_map = []
        for parent, details in [
            (rl,   ["Reward Shaping","Policy Gradient","Multi-Agent"]),
            (cons, ["Self-awareness","Recursive Thought","Emergence"]),
            (graph,["Node Embeddings","Edge Weights","Traversal Algo"]),
        ]:
            for i,detail in enumerate(details):
                angle = (i/3)*math.pi*2
                child = self.add_node(detail,
                    parent.x+math.cos(angle)*1.5,
                    parent.y+math.sin(angle)*1.5,
                    parent.z+math.sin(angle*2)*1.2,
                    node_type="child",depth=2,parent_id=parent.id)
                children_map.append((child.id, parent.id))

        # Potential nodes — created AND weakly linked to nearest active
        potentials = []
        for i,label in enumerate(["Payment System","P2P Network","Local-first DB","Agent Swarm",
                                   "Neural Architecture","Mesh Network","Quantum State","Graph Database"]):
            angle=(i/8)*math.pi*2
            p = self.add_node(label,math.cos(angle)*9,random.uniform(-2,2),math.sin(angle)*9,
                              node_type="potential",depth=1)
            potentials.append(p)

        # Core active-to-active connections
        for f,t,s in [
            (core.id,rl.id,0.8),(core.id,cons.id,0.9),(core.id,graph.id,1.0),
            (core.id,dec_i.id,0.85),(rl.id,graph.id,0.7),(cons.id,full_v.id,0.6),
            (graph.id,dec_i.id,0.9),(deploy.id,algeria.id,0.7),(dec_i.id,repos.id,0.5),
            (full_v.id,graph.id,0.8),(repos.id,deploy.id,0.6),(cons.id,time_n.id,0.7),
            (rl.id,time_n.id,0.5),
        ]: self.connect(f,t,strength=s,edge_type="connection")

        # Wire children to parents (hierarchy edges)
        for child_id, parent_id in children_map:
            self.connect(child_id, parent_id, strength=0.7, edge_type="hierarchy")

        # Wire potentials to nearest active (weak exploratory edges)
        active_ids = {n.id for n in self.get_all_nodes() if n.node_type in ("active","meta")}
        for p in potentials:
            nearest = self.find_nearest(p, k=2, exclude_types=["potential","child"])
            for other, dist, sim, score in nearest[:1]:
                self.connect(p.id, other.id, strength=0.15, edge_type="potential_link")

        # Record initial snapshot
        self._topo_dirty = True
        self.record_snapshot()
        return self
