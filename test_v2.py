#!/usr/bin/env python3
"""
ThoughtGraph v2 — Extended Test Suite
Tests all new algorithms: embeddings, PageRank, Louvain,
betweenness, Fiedler, activation spreading, Hebbian learning,
temporal decay, entropy, surprise, health score.
"""
import sys, time, math
sys.path.insert(0, '.')
from thought_graph import (
    ThoughtGraph, ThoughtNode, ThoughtEdge,
    GraphAnalyzer, ActivationEngine, TemporalEngine,
    make_embedding, cosine_sim, EvaluationResult
)

# Mock _fnv1a import if needed or import it from thought_graph
from thought_graph import _fnv1a

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  ✅ {name}"); PASS += 1
    else:
        print(f"  ❌ {name}" + (f" — {detail}" if detail else "")); FAIL += 1


# ═══════════════════════════════════════════════════════════
def test_fnv1a_hash():
    print("\n📌 TEST: FNV-1a Hash")
    h1 = _fnv1a("hello")
    h2 = _fnv1a("hello")
    h3 = _fnv1a("world")
    check("Deterministic", h1 == h2)
    check("Different inputs differ", h1 != h3)
    check("Returns int", isinstance(h1, int))
    check("32-bit range", 0 <= h1 <= 0xFFFFFFFF)

# ═══════════════════════════════════════════════════════════
def test_ngram_embeddings():
    print("\n📌 TEST: N-gram Embeddings")
    e1 = make_embedding("Graph Thinking")
    e2 = make_embedding("Graph Database")
    e3 = make_embedding("Algeria Context")
    e4 = make_embedding("Graph Thinking")  # duplicate

    check("512-dim vector", len(e1) == 512)
    check("Deterministic", e1 == e4)
    check("L2-normalized (norm ≈ 1)", abs(math.sqrt(sum(x*x for x in e1)) - 1.0) < 0.001)

    # Key quality tests: related concepts must score higher than unrelated
    sim_related = (cosine_sim(e1, e2) + 1) / 2      # Graph Thinking vs Graph Database
    sim_unrelated = (cosine_sim(e1, e3) + 1) / 2    # Graph Thinking vs Algeria Context
    check(
        f"Related > unrelated ({sim_related:.3f} > {sim_unrelated:.3f})",
        sim_related > sim_unrelated,
        f"Graph-Graph={sim_related:.3f} vs Graph-Algeria={sim_unrelated:.3f}"
    )

    # Self-similarity = 1.0
    self_sim = cosine_sim(e1, e1)
    check("Self-cosine = 1.0", abs(self_sim - 1.0) < 0.001)

    # Graph pairs should be positive
    check("Graph pair positive cosine", cosine_sim(e1, e2) > 0.3)

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_pagerank():
    print("\n📌 TEST: GraphAnalyzer — PageRank")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    pr = topo["pagerank"]

    check("PageRank dict returned", isinstance(pr, dict))
    check("All nodes have PR", len(pr) == len(g.get_all_nodes()))
    check("PR values sum ≈ 1.0", abs(sum(pr.values()) - 1.0) < 0.05)
    check("All PR values positive", all(v > 0 for v in pr.values()))

    # Core is highly connected — should rank in top-5 by PageRank
    nodes = {n.label: n for n in g.get_all_nodes()}
    core_id = nodes["Core Decision Pattern"].id
    top5 = sorted(pr, key=pr.get, reverse=True)[:5]
    check("Core in top-5 PageRank",
          core_id in top5,
          f"Core PR={pr[core_id]:.4f}")

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_betweenness():
    print("\n📌 TEST: GraphAnalyzer — Betweenness Centrality")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    btw = topo["betweenness"]

    check("Betweenness dict returned", isinstance(btw, dict))
    check("All nodes have betweenness", len(btw) == len(g.get_all_nodes()))
    check("All values in [0,1]", all(0 <= v <= 1 for v in btw.values()))
    check("Top betweenness node identified", topo["top_betweenness_node"] is not None)

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_communities():
    print("\n📌 TEST: GraphAnalyzer — Louvain Communities")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    coms = topo["communities"]

    check("Communities dict returned", isinstance(coms, dict))
    check("All nodes assigned", len(coms) == len(g.get_all_nodes()))
    check("Community IDs are ints", all(isinstance(v, int) for v in coms.values()))
    check("Multiple communities found", topo["n_communities"] >= 2)
    check("Modularity in [-1, 1]", -1 <= topo["modularity"] <= 1)

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_hits():
    print("\n📌 TEST: GraphAnalyzer — HITS (Hubs & Authorities)")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()

    hubs = topo["hubs"]
    auth = topo["authorities"]
    check("Hubs dict returned", isinstance(hubs, dict))
    check("Authorities dict returned", isinstance(auth, dict))
    check("Top hub node identified", topo["top_hub_node"] is not None)
    check("Hub values non-negative", all(v >= 0 for v in hubs.values()))

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_burt_constraint():
    print("\n📌 TEST: GraphAnalyzer — Burt's Structural Constraint")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    cst = topo["constraint"]

    check("Constraint dict returned", isinstance(cst, dict))
    check("Structural hole node identified", topo["structural_hole_node"] is not None)

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_fiedler():
    print("\n📌 TEST: GraphAnalyzer — Fiedler Value")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    fiedler = topo["fiedler"]

    check("Fiedler value is float", isinstance(fiedler, float))
    check("Fiedler >= 0", fiedler >= 0.0)

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_entropy():
    print("\n📌 TEST: GraphAnalyzer — Graph Entropy")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    entr = topo["graph_entropy"]

    check("Entropy dict has required keys",
          all(k in entr for k in ("entropy", "max_entropy", "efficiency")))
    check("Entropy >= 0", entr["entropy"] >= 0)
    check("Efficiency in [0,1]", 0 <= entr["efficiency"] <= 1)

# ═══════════════════════════════════════════════════════════
def test_activation_spreading():
    print("\n📌 TEST: Activation Spreading")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    nodes = {n.label: n for n in g.get_all_nodes()}
    core_id = nodes["Core Decision Pattern"].id

    activation = g.activate_node(core_id, spread=True)

    check("Returns dict", isinstance(activation, dict))
    check("Source node at 1.0", activation.get(core_id) == 1.0)
    check("Activation spreads to neighbors", len(activation) > 1)

# ═══════════════════════════════════════════════════════════
def test_hebbian_learning():
    print("\n📌 TEST: Hebbian Learning")
    ae = ActivationEngine()
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    edges = g.get_edges()
    core_id = next(n.id for n in g.get_all_nodes() if n.label == "Core Decision Pattern")

    activation = ae.spread([core_id], g._nodes, edges)
    original_strengths = {(e.from_id, e.to_id): e.strength for e in edges}

    updated = ae.hebbian_update(activation, edges, lr=0.1)
    check("Hebbian update returns count", isinstance(updated, int))
    check("Some edges updated", updated > 0)

# ═══════════════════════════════════════════════════════════
def test_temporal_decay():
    print("\n📌 TEST: Temporal Decay")
    te = TemporalEngine()
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    # Manually set old timestamps
    old_time = time.time() - 7200  # 2 hours ago
    for node in g.get_all_nodes():
        node.created_at = old_time

    results = te.decay_all(g._nodes, rate=0.015)
    check("Returns dict of importances", isinstance(results, dict))
    check("Floor maintained", all(v >= 0.10 for v in results.values()))

# ═══════════════════════════════════════════════════════════
def test_surprise_score():
    print("\n📌 TEST: Surprise Score")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    similar = g.add_node("Graph Neural Thinking", 2.1, -1.9, 1.1, node_type="potential")
    surprise_low = g.compute_surprise(similar)

    alien = g.add_node("Culinary Gastronomy Recipes", 50, 50, 50, node_type="potential")
    surprise_high = g.compute_surprise(alien)

    check("Surprise in [0,1]", 0 <= surprise_low <= 1 and 0 <= surprise_high <= 1)
    check("Similar node less surprising", surprise_low < surprise_high)

# ═══════════════════════════════════════════════════════════
def test_5factor_evaluation():
    print("\n📌 TEST: 5-Factor Evaluation")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    gnn = g.add_node("Graph Neural Network", 2.5, -1.5, 0.5, node_type="potential")
    result_gnn = g.evaluate_new_node(gnn)

    check("EvaluationResult returned", isinstance(result_gnn, EvaluationResult))
    check("Composite in [0,1]", 0 <= result_gnn.pattern_match_score <= 1)

# ═══════════════════════════════════════════════════════════
def test_health_score():
    print("\n📌 TEST: Graph Health Score")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    health = g.graph_health_score()

    check("Returns dict", isinstance(health, dict))
    check("Score in [0,100]", 0 <= health["score"] <= 100)
    check("Grade is A-F", health["grade"] in ("A","B","C","D","F"))

# ═══════════════════════════════════════════════════════════
def test_topology_caching():
    print("\n📌 TEST: Topology Caching")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    topo1 = g.get_topology()
    topo2 = g.get_topology()
    check("Topology is cached", topo1 is topo2)

# ═══════════════════════════════════════════════════════════
def test_small_world():
    print("\n📌 TEST: Small-World Index")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    sw = topo["small_world_index"]
    check("Small-world index is float", isinstance(sw, float))
    check("Small-world >= 0", sw >= 0.0)

# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  THOUGHTGRAPH v2 — FULL TEST SUITE (all algorithms)")
    print("=" * 65)

    # Re-run restored tests
    test_fnv1a_hash()
    test_ngram_embeddings()
    test_graph_analyzer_pagerank()
    test_graph_analyzer_betweenness()
    test_graph_analyzer_communities()
    test_graph_analyzer_hits()
    test_graph_analyzer_burt_constraint()
    test_graph_analyzer_fiedler()
    test_graph_analyzer_entropy()
    test_activation_spreading()
    test_hebbian_learning()
    test_temporal_decay()
    test_surprise_score()
    test_5factor_evaluation()
    test_health_score()
    test_topology_caching()
    test_small_world()

    # Also run the newly added unit tests
    import test_thought_graph
    test_thought_graph.test_node_creation()
    test_thought_graph.test_edge_management()
    test_thought_graph.test_similarity()
    test_thought_graph.test_node_removal()
    test_thought_graph.test_promote_potential()
    test_thought_graph.test_serialization()

    total = PASS + FAIL
    print("\n" + "=" * 65)
    print(f"  RESULTS: {PASS}/{total} passed  |  {FAIL} failed")
    print("=" * 65)
    sys.exit(0 if FAIL == 0 else 1)
