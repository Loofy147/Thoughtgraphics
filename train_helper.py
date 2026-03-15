import time
from thought_graph import ThoughtGraph

def run_training_cycle(graph: ThoughtGraph, node_label="Core Decision Pattern", cycles=5):
    nodes = graph.get_all_nodes()
    target = next((n for n in nodes if n.label == node_label), None)
    if not target and nodes:
        target = nodes[0]
    if not target:
        return {"error": "No nodes found to train"}

    initial_strengths = {(e.from_id, e.to_id): e.strength for e in graph.get_edges()}

    for _ in range(cycles):
        graph.activate_node(target.id, spread=True)

    final_edges = graph.get_edges()
    strengthened = []
    for e in final_edges:
        init_s = initial_strengths.get((e.from_id, e.to_id))
        if init_s is not None and e.strength > init_s:
            strengthened.append({
                "from": e.from_id,
                "to": e.to_id,
                "before": round(init_s, 4),
                "after": round(e.strength, 4)
            })
    return {"trained_node": target.label, "cycles": cycles, "strengthened_edges": strengthened}
