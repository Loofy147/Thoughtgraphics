"""
ThoughtGraph API v2.1 — March 2026
All v2 features exposed: topology, health, activation,
decay, recommendations, bridges, link prediction, evolution history.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn, time

from thought_graph import ThoughtGraph, ThoughtNode

app = FastAPI(title="ThoughtGraph API", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

graph = ThoughtGraph(persist=True)
if not graph.get_all_nodes():
    graph.seed_default_graph()


# ── REQUEST MODELS ────────────────────────────

class AddNodeRequest(BaseModel):
    label: str
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    node_type: str = "potential"
    depth: int = 1
    parent_id: Optional[int] = None
    tags: list = []
    importance: float = 1.0
    auto_evaluate: bool = True
    auto_connect: bool = True

class ConnectRequest(BaseModel):
    from_id: int
    to_id: int
    strength: float = 0.5
    edge_type: str = "connection"

class UpdateImportanceRequest(BaseModel):
    importance: float

class UpdateNodeRequest(BaseModel):
    label: Optional[str] = None
    importance: Optional[float] = None
    node_type: Optional[str] = None


# ── SERIALIZERS ───────────────────────────────

def node_to_dict(n: ThoughtNode) -> dict:
    return {
        "id":                   n.id,
        "label":                n.label,
        "x": n.x, "y": n.y, "z": n.z,
        "node_type":            n.node_type,
        "depth":                n.depth,
        "importance":           n.importance,
        "effective_importance": n.effective_importance,
        "parent_id":            n.parent_id,
        "children_ids":         n.children_ids,
        "connections":          n.connections,
        "tags":                 n.tags,
        "decision_weight":      n.decision_weight,
        "created_at":           n.created_at,
        "last_activated":       n.last_activated,
        "activation_count":     n.activation_count,
        "community_id":         n.community_id,
        "pagerank":             n.pagerank,
        "betweenness":          n.betweenness,
    }

def eval_to_dict(result) -> dict:
    return {
        "decision":              result.decision,
        "pattern_match_score":   result.pattern_match_score,
        "nearest_neighbors":     result.nearest_neighbors,
        "reasoning":             result.reasoning,
        "suggested_connections": result.suggested_connections,
        "factor_breakdown":      result.factor_breakdown,
    }


# ── CORE ROUTES ───────────────────────────────

@app.get("/")
def root():
    a = graph.graph_analytics()
    return {
        "status": "ok", "version": "2.1.0",
        "nodes": a.get("total_nodes", 0),
        "health_score": a.get("health_score", 0),
        "health_grade": a.get("health_grade", "F"),
    }

@app.get("/graph")
def get_graph():
    nodes = [node_to_dict(n) for n in graph.get_all_nodes()]
    edges = [{"from_id":e.from_id, "to_id":e.to_id,
               "strength":e.strength, "edge_type":e.edge_type,
               "activation_count":e.activation_count}
             for e in graph.get_edges()]
    return {
        "nodes":     nodes,
        "edges":     edges,
        "analytics": graph.graph_analytics(),
        "timestamp": time.time(),
    }

@app.get("/nodes/{node_id}")
def get_node(node_id: int):
    n = graph.get_node(node_id)
    if not n: raise HTTPException(404, "Node not found")
    return node_to_dict(n)

@app.post("/nodes")
def add_node(req: AddNodeRequest):
    node = graph.add_node(
        label=req.label, x=req.x, y=req.y, z=req.z,
        node_type=req.node_type, depth=req.depth,
        parent_id=req.parent_id, tags=req.tags, importance=req.importance,
    )
    evaluation = None
    if req.auto_evaluate:
        result = graph.evaluate_new_node(node)
        evaluation = eval_to_dict(result)
        if req.auto_connect:
            if result.decision == "ACCEPT":
                node.node_type = "active"
                for tid in result.suggested_connections:
                    graph.connect(node.id, tid, strength=0.6)
            elif result.decision == "POTENTIAL":
                for tid in result.suggested_connections[:1]:
                    graph.connect(node.id, tid, strength=0.2, edge_type="potential_link")
    graph.record_snapshot()
    return {"node": node_to_dict(node), "evaluation": evaluation}

@app.patch("/nodes/{node_id}")
def update_node(node_id: int, req: UpdateNodeRequest):
    n = graph.get_node(node_id)
    if not n: raise HTTPException(404, "Node not found")
    if req.label is not None:     n.label = req.label
    if req.node_type is not None: n.node_type = req.node_type
    if req.importance is not None:
        graph.update_node_importance(node_id, req.importance)
    graph._topo_dirty = True
    graph._save()
    return node_to_dict(graph.get_node(node_id))

@app.delete("/nodes/{node_id}")
def delete_node(node_id: int):
    if not graph.remove_node(node_id): raise HTTPException(404, "Node not found")
    return {"deleted": node_id}

@app.post("/nodes/{node_id}/evaluate")
def evaluate_node(node_id: int):
    n = graph.get_node(node_id)
    if not n: raise HTTPException(404, "Node not found")
    return eval_to_dict(graph.evaluate_new_node(n))

@app.post("/nodes/{node_id}/promote")
def promote_node(node_id: int):
    if not graph.promote_potential(node_id):
        raise HTTPException(400, "Not found or already active")
    n = graph.get_node(node_id)
    graph.record_snapshot()
    return {"promoted": node_id, "node": node_to_dict(n)}

@app.post("/nodes/{node_id}/activate")
def activate_node(node_id: int):
    n = graph.get_node(node_id)
    if not n: raise HTTPException(404, "Node not found")
    activation = graph.activate_node(node_id, spread=True)
    return {"node_id": node_id, "activation_spread": activation,
            "nodes_reached": len(activation)}

@app.get("/nodes/{node_id}/similar")
def similar_nodes(node_id: int, k: int = 5):
    n = graph.get_node(node_id)
    if not n: raise HTTPException(404, "Node not found")
    nearest = graph.find_nearest(n, k=k)
    return {"node_id": node_id, "similar": [
        {"node_id": other.id, "label": other.label,
         "semantic_similarity": round(sem, 3),
         "combined_score": round(combined, 3),
         "community": other.community_id}
        for other, _, sem, combined in nearest
    ]}

@app.post("/edges")
def add_edge(req: ConnectRequest):
    e = graph.connect(req.from_id, req.to_id, req.strength, req.edge_type)
    if not e: raise HTTPException(400, "Could not connect — check IDs")
    return {"from_id":e.from_id, "to_id":e.to_id, "strength":e.strength}


# ── ANALYSIS ROUTES ───────────────────────────

@app.get("/topology")
def get_topology():
    topo = graph.get_topology()
    # Return summary (full topo has large dicts)
    return {
        "fiedler":              topo.get("fiedler", 0),
        "small_world_index":    topo.get("small_world_index", 0),
        "modularity":           topo.get("modularity", 0),
        "n_communities":        topo.get("n_communities", 0),
        "n_components":         topo.get("n_components", 0),
        "graph_entropy":        topo.get("graph_entropy", {}),
        "top_pagerank_node":    topo.get("top_pagerank_node"),
        "top_betweenness_node": topo.get("top_betweenness_node"),
        "structural_hole_node": topo.get("structural_hole_node"),
        "n_bridges":            len(topo.get("bridges", [])),
        "suggested_links":      topo.get("suggested_links", []),
    }

@app.get("/health")
def get_health():
    health  = graph.graph_health_score()
    analytics = graph.graph_analytics()
    return {
        **health,
        "context": {
            "fiedler":      analytics.get("fiedler_value"),
            "modularity":   analytics.get("modularity"),
            "small_world":  analytics.get("small_world_index"),
            "n_components": analytics.get("n_components"),
            "density":      analytics.get("density"),
        }
    }

@app.get("/analytics")
def get_analytics():
    return graph.graph_analytics()

@app.get("/patterns")
def get_patterns():
    patterns = graph.detect_patterns()
    return {"patterns": patterns, "count": len(patterns)}

@app.get("/recommend")
def get_recommendations(k: int = 5):
    recs = graph.recommend_exploration(k=k)
    return {"recommendations": recs, "count": len(recs)}

@app.get("/bridges")
def get_bridges():
    bridges = graph.find_bridges()
    return {"bridges": bridges, "count": len(bridges)}

@app.get("/suggest-links")
def suggest_links(k: int = 5):
    links = graph.suggest_connections(k=k)
    return {"suggestions": links, "count": len(links)}

@app.post("/decay")
def decay():
    results = graph.decay_graph()
    graph.record_snapshot()
    return {"decayed": len(results), "importances": results}

@app.get("/history")
def get_history():
    return {"history": graph._evaluation_history[-50:]}

@app.get("/evolution")
def get_evolution():
    return {"snapshots": graph.get_evolution_history()}

@app.post("/snapshot")
def take_snapshot():
    snap = graph.record_snapshot()
    return snap

@app.get("/advice")
def get_advice():
    """Actionable graph health recommendations."""
    return {"advice": graph.graph_health_advice(), "health": graph.graph_health_score()}

@app.post("/apply-suggested-link/{from_id}/{to_id}")
def apply_suggestion(from_id: int, to_id: int, strength: float = 0.45):
    """Accept a predicted missing link."""
    edge = graph.connect(from_id, to_id, strength=strength, edge_type="connection")
    if not edge:
        raise HTTPException(400, "Could not connect")
    graph.record_snapshot()
    return {"from_id": from_id, "to_id": to_id, "strength": edge.strength}

@app.get("/export")
def export_graph():
    """Export full graph as JSON (nodes, edges, metadata, topology summary)."""
    topo = graph.get_topology()
    return {
        "version": "2.1",
        "exported_at": time.time(),
        "nodes": [node_to_dict(n) for n in graph.get_all_nodes()],
        "edges": [
            {"from_id": e.from_id, "to_id": e.to_id,
             "strength": e.strength, "edge_type": e.edge_type,
             "activation_count": e.activation_count}
            for e in graph.get_edges()
        ],
        "analytics": graph.graph_analytics(),
        "health": graph.graph_health_score(),
        "communities": topo.get("communities", {}),
        "evaluation_history": graph._evaluation_history[-20:],
        "evolution_history": graph.get_evolution_history(),
    }

@app.get("/path/{from_id}/{to_id}")
def get_concept_path(from_id: int, to_id: int):
    """Shortest semantic path between two nodes."""
    result = graph.concept_path(from_id, to_id)
    if not result["found"]:
        raise HTTPException(404, f"No path found between {from_id} and {to_id}")
    return result

@app.get("/duplicates")
def get_duplicates(threshold: float = 0.88):
    """Find semantically near-identical node pairs."""
    dups = graph.find_duplicates(threshold=threshold)
    return {"duplicates": dups, "count": len(dups), "threshold": threshold}

@app.post("/merge/{keep_id}/{remove_id}")
def merge_nodes(keep_id: int, remove_id: int):
    """Merge two nodes: transfer connections from remove_id to keep_id."""
    if not graph.merge_nodes(keep_id, remove_id):
        raise HTTPException(400, "Merge failed — check node IDs")
    graph.record_snapshot()
    return {"merged": True, "kept": keep_id, "removed": remove_id,
            "node": node_to_dict(graph.get_node(keep_id))}

@app.post("/snapshot/{name}")
def named_snapshot(name: str):
    """Save a named checkpoint of the current graph state."""
    snap = graph.save_snapshot(name)
    return snap

@app.get("/export/graphml")
def export_graphml():
    """Export graph as GraphML (Gephi, yEd compatible)."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(graph.export_graphml(),
                             media_type="application/xml",
                             headers={"Content-Disposition": "attachment; filename=thoughtgraph.graphml"})

@app.get("/export/dot")
def export_dot():
    """Export graph as DOT format (Graphviz compatible)."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(graph.export_dot(),
                             media_type="text/plain",
                             headers={"Content-Disposition": "attachment; filename=thoughtgraph.dot"})


# ── Search & Query ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:          str   = ""
    node_type:      Optional[str]   = None
    min_importance: float = 0.0
    community_id:   Optional[int]   = None
    min_pagerank:   float = 0.0
    tags:           list  = []
    limit:          int   = 20

@app.get("/search")
def search_nodes(q: str = "", node_type: str = None, min_importance: float = 0.0,
                 community_id: int = None, limit: int = 20):
    """Search nodes by text query and/or filters."""
    nodes = graph.search_nodes(
        query=q, node_type=node_type, min_importance=min_importance,
        community_id=community_id, limit=limit,
    )
    return {"results": [node_to_dict(n) for n in nodes], "count": len(nodes)}

@app.get("/community/{community_id}")
def get_community(community_id: int):
    """Get all nodes and internal edges for a Louvain community."""
    sub = graph.get_community_subgraph(community_id)
    if not sub["nodes"]:
        raise HTTPException(404, f"Community {community_id} not found")
    return sub

@app.get("/communities")
def list_communities():
    """List all communities with member counts and anchor nodes."""
    topo = graph.get_topology()
    coms = topo.get("communities", {})
    from collections import defaultdict
    groups = defaultdict(list)
    for nid, cid in coms.items():
        n = graph.get_node(nid)
        if n: groups[cid].append(n)
    pr = topo.get("pagerank", {})
    result = []
    for cid, nodes in sorted(groups.items()):
        anchor = max(nodes, key=lambda n: pr.get(n.id, 0))
        result.append({
            "community_id": cid,
            "size":         len(nodes),
            "anchor":       anchor.label,
            "labels":       [n.label for n in sorted(nodes, key=lambda n: pr.get(n.id,0), reverse=True)],
        })
    result.sort(key=lambda c: -c["size"])
    return {"communities": result, "count": len(result)}

# ── Auto-Heal ─────────────────────────────────────────────────────

@app.post("/heal")
def auto_heal(max_links: int = 8, min_score: float = 0.5):
    """Automatically apply predicted links to improve graph connectivity."""
    result = graph.auto_heal_graph(max_links=max_links, min_score=min_score)
    return result

# ── Batch Import ──────────────────────────────────────────────────

class BatchItem(BaseModel):
    label:      str
    node_type:  str   = "potential"
    importance: float = 1.0
    tags:       list  = []
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None

class BatchImportRequest(BaseModel):
    items:         list
    auto_evaluate: bool = True

@app.post("/batch")
def batch_import(req: BatchImportRequest):
    """Add multiple nodes at once, optionally auto-evaluating each."""
    result = graph.batch_import(req.items, auto_evaluate=req.auto_evaluate)
    return result

# ── Graph Diff ────────────────────────────────────────────────────

@app.get("/diff/{snap_a}/{snap_b}")
def graph_diff(snap_a: int, snap_b: int):
    """Compare two evolution snapshots by index."""
    history = graph.get_evolution_history()
    if snap_a >= len(history) or snap_b >= len(history):
        raise HTTPException(404, f"Snapshot index out of range (history has {len(history)} entries)")
    return graph.graph_diff(history[snap_a], history[snap_b])


@app.post("/train")
def train_graph(cycles: int = 5, node_label: str = "Core Decision Pattern"):
    """Run Hebbian training cycles on the graph."""
    from train_helper import run_training_cycle
    result = run_training_cycle(graph, node_label=node_label, cycles=cycles)
    return result

@app.post("/reset")
def reset_graph():
    graph.reset()
    graph.seed_default_graph()
    return {"status": "reset", "nodes": len(graph.get_all_nodes())}

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    with open("thought_graph_ui.html") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
