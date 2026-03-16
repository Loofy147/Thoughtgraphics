#!/usr/bin/env python3
"""
ThoughtGraph Training Script
Demonstrates Hebbian learning by repeatedly activating related nodes.
"""
import time
from thought_graph import ThoughtGraph

def run_training():
    print("Initializing ThoughtGraph for training...")
    graph = ThoughtGraph(persist=True)

    # Ensure we have a clean state or some data
    if not graph.get_all_nodes():
        print("Seeding default graph...")
        graph.seed_default_graph()

    # Identify some nodes to train (e.g., Core and its immediate neighbors)
    nodes = graph.get_all_nodes()
    core_node = next((n for n in nodes if n.label == "Core Decision Pattern"), None)

    if not core_node:
        print("Core node not found. Training on first node.")
        core_node = nodes[0]

    print(f"Targeting node for training: {core_node.label} (ID: {core_node.id})")

    # Record initial edge strengths
    initial_edges = {(e.from_id, e.to_id): e.strength for e in graph.get_edges()}

    cycles = 5
    print(f"Starting {cycles} activation cycles...")

    for i in range(cycles):
        print(f"  Cycle {i+1}/{cycles}...")
        activation = graph.activate_node(core_node.id, spread=True)
        print(f"    Activated {len(activation)} nodes.")
        time.sleep(0.1) # Small delay

    print("Training complete.")

    # Check for changes
    final_edges = graph.get_edges()
    strengthened = []
    for e in final_edges:
        init_s = initial_edges.get((e.from_id, e.to_id))
        if init_s is not None and e.strength > init_s:
            node_from = graph.get_node(e.from_id).label
            node_to = graph.get_node(e.to_id).label
            strengthened.append(f"    {node_from} <-> {node_to}: {init_s:.4f} -> {e.strength:.4f}")

    if strengthened:
        print("\nStrengthened Edges:")
        print("\n".join(strengthened))
    else:
        print("\nNo edges were significantly strengthened. (Try more cycles or higher learning rate in thought_graph.py)")

if __name__ == "__main__":
    run_training()
