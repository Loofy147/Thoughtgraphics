import pytest
from thought_graph import ThoughtGraph

def test_node_creation():
    g = ThoughtGraph(persist=False)
    n = g.add_node("Test Node")
    assert n.label == "Test Node"
    assert len(g.get_all_nodes()) == 1

def test_edge_management():
    g = ThoughtGraph(persist=False)
    n1 = g.add_node("A")
    n2 = g.add_node("B")
    g.connect(n1.id, n2.id)
    assert len(g.get_edges()) == 1

def test_similarity():
    g = ThoughtGraph(persist=False)
    n1 = g.add_node("Machine Learning")
    n2 = g.add_node("Deep Learning")
    assert n1.semantic_similarity(n2) > 0.5

def test_evaluation_engine():
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    n = g.add_node("Reinforcement Learning", node_type="potential")
    res = g.evaluate_new_node(n)
    assert res.decision in ["ACCEPT", "POTENTIAL"]

def test_node_removal():
    g = ThoughtGraph(persist=False)
    n = g.add_node("Delete Me")
    g.remove_node(n.id)
    assert len(g.get_all_nodes()) == 0

def test_pattern_detection():
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    patterns = g.detect_patterns()
    assert len(patterns) > 0

def test_analytics():
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    a = g.graph_analytics()
    assert "total_nodes" in a
    assert a["total_nodes"] > 0

def test_promote_potential():
    g = ThoughtGraph(persist=False)
    n = g.add_node("Potential", node_type="potential")
    g.promote_potential(n.id)
    assert g.get_node(n.id).node_type == "active"

def test_serialization():
    g = ThoughtGraph(persist=False)
    g.add_node("Persist")
    d = g.to_dict()
    assert "nodes" in d
    assert len(d["nodes"]) == 1

def test_seed_data():
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    assert len(g.get_all_nodes()) > 10
