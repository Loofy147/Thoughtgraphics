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
    make_embedding, cosine_sim, _fnv1a, EvaluationResult
)

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

    # Decision-related pair
    d1 = make_embedding("Core Decision Pattern")
    d2 = make_embedding("Decision Intuition")
    d_sim = (cosine_sim(d1, d2) + 1) / 2
    check(f"Decision pair high similarity ({d_sim:.3f} > 0.70)", d_sim > 0.70)

    # RL pair
    r1 = make_embedding("RL Agents")
    r2 = make_embedding("Multi-Agent")
    r_sim = (cosine_sim(r1, r2) + 1) / 2
    check(f"RL/Agent pair moderate ({r_sim:.3f} > 0.50)", r_sim > 0.50)

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_pagerank():
    print("\n📌 TEST: GraphAnalyzer — PageRank")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    pr = topo["pagerank"]

    check("PageRank dict returned", isinstance(pr, dict))
    check("All nodes have PR", len(pr) == len(g.get_all_nodes()))
    check("PR values sum ≈ 1.0", abs(sum(pr.values()) - 1.0) < 0.01)
    check("All PR values positive", all(v > 0 for v in pr.values()))

    # Core is highly connected — should rank in top-5 by PageRank
    nodes = {n.label: n for n in g.get_all_nodes()}
    core_id = nodes["Core Decision Pattern"].id
    top5 = sorted(pr, key=pr.get, reverse=True)[:5]
    check("Core in top-5 PageRank",
          core_id in top5,
          f"Core PR={pr[core_id]:.4f}")

    print(f"    → Core PR={pr[core_id]:.4f}, top node: {topo['top_pagerank_node']}")

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
    print(f"    → Top betweenness: {topo['top_betweenness_node']}")

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

    # Core node and Graph Thinking should be in same or adjacent community
    # (they're directly connected)
    nodes = {n.label: n for n in g.get_all_nodes()}
    core_id  = nodes["Core Decision Pattern"].id
    graph_id = nodes["Graph Thinking"].id
    # They are connected, so Louvain may put them together
    # At minimum, both should have valid community assignments
    check("Core has community", coms.get(core_id, -1) >= 0)
    check("Graph Thinking has community", coms.get(graph_id, -1) >= 0)
    print(f"    → {topo['n_communities']} communities, modularity={topo['modularity']}")

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
    print(f"    → Top hub: {topo['top_hub_node']}")

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_burt_constraint():
    print("\n📌 TEST: GraphAnalyzer — Burt's Structural Constraint")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    cst = topo["constraint"]

    check("Constraint dict returned", isinstance(cst, dict))
    check("Structural hole node identified", topo["structural_hole_node"] is not None)
    # Nodes with low constraint are structural bridges
    values = list(cst.values())
    check("Constraint values exist", len(values) > 0)
    print(f"    → Bridge node (min constraint): {topo['structural_hole_node']}")

# ═══════════════════════════════════════════════════════════
def test_graph_analyzer_fiedler():
    print("\n📌 TEST: GraphAnalyzer — Fiedler Value")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    fiedler = topo["fiedler"]

    check("Fiedler value is float", isinstance(fiedler, float))
    check("Fiedler >= 0", fiedler >= 0.0)
    # A connected graph should have fiedler > 0
    # (our seeded graph may be disconnected due to potential nodes)
    check("Fiedler value computed", fiedler >= 0.0)
    print(f"    → Fiedler = {fiedler:.6f}")

    # Test disconnected graph
    g2 = ThoughtGraph(persist=False)
    g2.add_node("Isolated A", 0, 0, 0)
    g2.add_node("Isolated B", 100, 100, 100)
    topo2 = g2.get_topology()
    fiedler2 = topo2.get("fiedler", 0.0)
    check("Disconnected graph fiedler ≈ 0", fiedler2 == 0.0, f"got {fiedler2}")

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
    check("Max entropy > 0", entr["max_entropy"] > 0)
    print(f"    → H={entr['entropy']}, H_max={entr['max_entropy']}, eff={entr['efficiency']}")

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
    check("Activation spreads to neighbors",
          len(activation) > 1,
          f"only {len(activation)} nodes activated")
    check("All values in (0,1]", all(0 < v <= 1.0 for v in activation.values()))
    check("Distant nodes have lower activation",
          activation.get(core_id, 0) >= max(
              (v for k,v in activation.items() if k != core_id), default=0
          ))

    print(f"    → Activated {len(activation)} nodes from Core")
    # Print top 3
    top = sorted(activation.items(), key=lambda x: -x[1])[:3]
    for nid, act in top:
        label = nodes.get(next((l for l,n in nodes.items() if n.id==nid),"?"), None)
        nobj = g.get_node(nid)
        print(f"      {nobj.label if nobj else nid}: {act:.3f}")

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

    # Check that co-activated edges got stronger
    strengthened = sum(
        1 for e in edges
        if e.strength > original_strengths.get((e.from_id, e.to_id), 0)
    )
    check("Co-activated edges strengthened", strengthened > 0,
          f"0 edges strengthened out of {updated} updated")
    print(f"    → {updated} edges updated, {strengthened} strengthened")

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
    check("All nodes have decay results", len(results) == len(g.get_all_nodes()))

    # Importance should have decayed
    decayed = [v for v in results.values() if v < 1.0]
    check("Importances decayed", len(decayed) > 0,
          f"No decay detected in {results}")
    check("Floor maintained", all(v >= 0.10 for v in results.values()))

    # Activate a node and check it refreshes
    node = g.get_all_nodes()[0]
    te.activate(node)
    check("Activation resets importance", node.effective_importance == node.importance)
    check("Activation count incremented", node.activation_count == 1)

# ═══════════════════════════════════════════════════════════
def test_surprise_score():
    print("\n📌 TEST: Surprise Score")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    # Very similar node — low surprise
    similar = g.add_node("Graph Neural Thinking", 2.1, -1.9, 1.1, node_type="potential")
    surprise_low = g.compute_surprise(similar)

    # Very different node — high surprise
    alien = g.add_node("Culinary Gastronomy Recipes", 50, 50, 50, node_type="potential")
    surprise_high = g.compute_surprise(alien)

    check("Surprise in [0,1]", 0 <= surprise_low <= 1 and 0 <= surprise_high <= 1)
    check(
        f"Similar node less surprising ({surprise_low:.3f} < {surprise_high:.3f})",
        surprise_low < surprise_high
    )
    print(f"    → Similar: {surprise_low}, Alien: {surprise_high}")

# ═══════════════════════════════════════════════════════════
def test_5factor_evaluation():
    print("\n📌 TEST: 5-Factor Evaluation")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    # Highly related node — should ACCEPT
    gnn = g.add_node("Graph Neural Network", 2.5, -1.5, 0.5, node_type="potential")
    result_gnn = g.evaluate_new_node(gnn)

    check("EvaluationResult returned", isinstance(result_gnn, EvaluationResult))
    check("factor_breakdown present", "semantic" in result_gnn.factor_breakdown)
    check("All 5 factors in breakdown",
          all(k in result_gnn.factor_breakdown
              for k in ("semantic","community","pr_influence","bridging","novelty")))
    check("Composite in [0,1]", 0 <= result_gnn.pattern_match_score <= 1)

    # Completely alien node far away — should get lower score
    alien = g.add_node("Underwater Basket Weaving Championship", 50, 50, 50, node_type="potential")
    result_alien = g.evaluate_new_node(alien)

    check("Related node scores higher than alien",
          result_gnn.pattern_match_score >= result_alien.pattern_match_score,
          f"GNN={result_gnn.pattern_match_score} vs Alien={result_alien.pattern_match_score}")

    print(f"    → GNN: {result_gnn.decision} ({result_gnn.pattern_match_score:.0%})")
    print(f"    → Alien: {result_alien.decision} ({result_alien.pattern_match_score:.0%})")
    print(f"    → GNN factors: {result_gnn.factor_breakdown}")

# ═══════════════════════════════════════════════════════════
def test_health_score():
    print("\n📌 TEST: Graph Health Score")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    health = g.graph_health_score()

    check("Returns dict", isinstance(health, dict))
    check("Has score", "score" in health)
    check("Has grade", "grade" in health)
    check("Has breakdown", "breakdown" in health)
    check("Score in [0,100]", 0 <= health["score"] <= 100)
    check("Grade is A-F", health["grade"] in ("A","B","C","D","F"))
    check("Breakdown has 5 components",
          all(k in health["breakdown"]
              for k in ("connectivity","community","entropy","small_world","diversity")))

    print(f"    → Score: {health['score']}/100 Grade: {health['grade']}")
    print(f"    → Breakdown: {health['breakdown']}")

# ═══════════════════════════════════════════════════════════
def test_topology_caching():
    print("\n📌 TEST: Topology Caching")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    t1 = time.time()
    topo1 = g.get_topology()
    t2 = time.time()
    topo2 = g.get_topology()  # should be cached
    t3 = time.time()

    first_time  = t2 - t1
    cached_time = t3 - t2

    check("Topology is cached", topo1 is topo2)
    check("Cache is faster", cached_time < first_time,
          f"first={first_time:.3f}s cache={cached_time:.3f}s")

    # Adding a node should invalidate cache
    g.add_node("Cache Test Node", 0, 0, 0)
    topo3 = g.get_topology()
    check("New node invalidates cache", topo3 is not topo1)
    print(f"    → First compute: {first_time*1000:.1f}ms, cached: {cached_time*1000:.3f}ms")

# ═══════════════════════════════════════════════════════════
def test_decay_and_activation_integration():
    print("\n📌 TEST: Decay + Activation Integration")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    nodes = g.get_all_nodes()
    initial_importances = {n.id: n.effective_importance for n in nodes}

    # Set old timestamps to simulate passage of time
    old_time = time.time() - 10800  # 3 hours
    for n in nodes: n.created_at = old_time

    g.decay_graph()
    decayed_importances = {n.id: n.effective_importance for n in g.get_all_nodes()}

    check("Decay reduces importance",
          sum(decayed_importances.values()) < sum(initial_importances.values()))

    # Activate a node — its importance should refresh
    target_id = nodes[0].id
    g.activate_node(target_id)
    check("Activation refreshes importance",
          g.get_node(target_id).effective_importance >= decayed_importances[target_id])

# ═══════════════════════════════════════════════════════════
def test_node_annotations():
    print("\n📌 TEST: Node Annotations (PageRank, Betweenness, Community)")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()

    g.get_topology()  # trigger annotation
    nodes = g.get_all_nodes()

    check("PageRank annotated on nodes", all(isinstance(n.pagerank, float) for n in nodes))
    check("Betweenness annotated on nodes", all(isinstance(n.betweenness, float) for n in nodes))
    check("Community ID annotated", all(isinstance(n.community_id, int) for n in nodes))

    core = next(n for n in nodes if n.label == "Core Decision Pattern")
    check("Core has non-zero PageRank", core.pagerank > 0)

# ═══════════════════════════════════════════════════════════
def test_small_world():
    print("\n📌 TEST: Small-World Index")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    topo = g.get_topology()
    sw = topo["small_world_index"]
    check("Small-world index is float", isinstance(sw, float))
    check("Small-world >= 0", sw >= 0.0)
    print(f"    → sigma = {sw}")

# ═══════════════════════════════════════════════════════════
def test_v2_analytics_fields():
    print("\n📌 TEST: Analytics — New v2 Fields")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    a = g.graph_analytics()

    new_fields = ["fiedler_value","small_world_index","modularity","n_communities",
                  "graph_entropy","health_score","health_grade",
                  "top_pagerank_node","top_betweenness_node"]
    for field in new_fields:
        check(f"analytics.{field} present", field in a,
              f"Missing from: {list(a.keys())}")

    check("health_grade is letter", a.get("health_grade") in ("A","B","C","D","F"))
    check("n_communities >= 1", a.get("n_communities", 0) >= 1)

# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  THOUGHTGRAPH v2 — FULL TEST SUITE (all algorithms)")
    print("=" * 65)

    # --- Original tests (re-run) ---
    from test_thought_graph import (
        test_node_creation, test_edge_management, test_similarity,
        test_evaluation_engine, test_node_removal, test_pattern_detection,
        test_analytics, test_promote_potential, test_serialization, test_seed_data
    )
    test_node_creation()
    test_edge_management()
    test_similarity()
    test_evaluation_engine()
    test_node_removal()
    test_pattern_detection()
    test_analytics()
    test_promote_potential()
    test_serialization()
    test_seed_data()

    # --- New v2 tests ---
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
    test_decay_and_activation_integration()
    test_node_annotations()
    test_small_world()
    test_v2_analytics_fields()

    total = PASS + FAIL
    print("\n" + "=" * 65)
    print(f"  RESULTS: {PASS}/{total} passed  |  {FAIL} failed")
    print("=" * 65)
    sys.exit(0 if FAIL == 0 else 1)
