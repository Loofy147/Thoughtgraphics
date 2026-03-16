import sys, math, random, time
sys.path.insert(0, '.')
from thought_graph import ThoughtGraph, ThoughtNode, ThoughtEdge

def check(name, cond):
    if cond:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}")
        raise Exception(f"Test failed: {name}")

def test_node_creation():
    print("Testing Node Creation...")
    g = ThoughtGraph(persist=False)
    n = g.add_node("Test Node", 1, 2, 3)
    check("Node id exists", n.id == 0)
    check("Node label matches", n.label == "Test Node")
    check("Node type is active", n.node_type == "active")

def test_edge_management():
    print("Testing Edge Management...")
    g = ThoughtGraph(persist=False)
    n1 = g.add_node("Node 1")
    n2 = g.add_node("Node 2")
    e = g.connect(n1.id, n2.id, strength=0.8)
    check("Edge exists", len(g.get_edges()) == 1)
    check("Edge strength matches", e.strength == 0.8)
    check("Nodes are connected", n2.id in n1.connections)

def test_similarity():
    print("Testing Similarity...")
    g = ThoughtGraph(persist=False)
    n1 = g.add_node("Apple")
    n2 = g.add_node("Fruit")
    sim = n1.semantic_similarity(n2)
    check("Similarity is a float", isinstance(sim, float))
    check("Similarity in range", 0 <= sim <= 1)

def test_evaluation_engine():
    print("Testing Evaluation Engine...")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    n = g.add_node("Deep Learning", node_type="potential")
    res = g.evaluate_new_node(n)
    check("Evaluation decision exists", res.decision in ["ACCEPT", "POTENTIAL", "REJECT"])

def test_node_removal():
    print("Testing Node Removal...")
    g = ThoughtGraph(persist=False)
    n = g.add_node("To be removed")
    g.remove_node(n.id)
    check("Node removed", len(g.get_all_nodes()) == 0)

def test_pattern_detection():
    print("Testing Pattern Detection...")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    patterns = g.detect_patterns()
    check("Patterns detected", len(patterns) > 0)

def test_analytics():
    print("Testing Analytics...")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    a = g.graph_analytics()
    check("Total nodes in analytics", a["total_nodes"] > 0)

def test_promote_potential():
    print("Testing Promotion...")
    g = ThoughtGraph(persist=False)
    n = g.add_node("Potential Idea", node_type="potential")
    g.promote_potential(n.id)
    check("Node promoted to active", n.node_type == "active")

def test_serialization():
    print("Testing Serialization...")
    g = ThoughtGraph(persist=False)
    g.add_node("Save Me")
    d = g.to_dict()
    check("Dict has nodes", "nodes" in d)
    check("Dict has version", "version" in d)

def test_seed_data():
    print("Testing Seed Data...")
    g = ThoughtGraph(persist=False)
    g.seed_default_graph()
    check("Seed data has nodes", len(g.get_all_nodes()) > 0)

if __name__ == "__main__":
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
