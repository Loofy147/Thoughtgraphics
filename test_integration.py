#!/usr/bin/env python3
"""
ThoughtGraph v2.1 — API Integration Tests
Tests the actual running HTTP server end-to-end.
Requires: python api.py running on localhost:8000

Run with server: python api.py &  then  python test_integration.py
Run standalone:  python test_integration.py  (starts/stops server automatically)
"""
import sys, time, json, subprocess, signal
import urllib.request, urllib.error

BASE = "http://localhost:8000"
PASS = FAIL = 0


# ─── HTTP HELPERS ─────────────────────────────
def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=5) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code
    except Exception as e:
        return {"error": str(e)}, 0

def post(path, body=None):
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(BASE + path, data=data,
           headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code
    except Exception as e:
        return {"error": str(e)}, 0

def delete(path):
    req = urllib.request.Request(BASE + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code
    except Exception as e:
        return {"error": str(e)}, 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  ✅ {name}"); PASS += 1
    else:
        print(f"  ❌ {name}" + (f" — {detail}" if detail else "")); FAIL += 1


# ─── WAIT FOR SERVER ──────────────────────────
def wait_for_server(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        data, status = get("/")
        if status == 200:
            return True
        time.sleep(0.4)
    return False


# ─── TESTS ────────────────────────────────────
def test_root():
    print("\n📌 API: Root endpoint")
    data, status = get("/")
    check("Status 200", status == 200)
    check("Has version",  "version" in data)
    check("Has nodes",    "nodes" in data)
    check("Has health",   "health_score" in data)
    check("Health grade present", "health_grade" in data)
    print(f"    → v{data.get('version')} | {data.get('nodes')} nodes | health {data.get('health_score')}/100 {data.get('health_grade')}")


def test_get_graph():
    print("\n📌 API: GET /graph")
    data, status = get("/graph")
    check("Status 200",         status == 200)
    check("Has nodes list",     isinstance(data.get("nodes"), list))
    check("Has edges list",     isinstance(data.get("edges"), list))
    check("Has analytics",      isinstance(data.get("analytics"), dict))
    check("Nodes non-empty",    len(data.get("nodes", [])) > 0)
    check("Edges non-empty",    len(data.get("edges", [])) > 0)
    # v2 fields on nodes
    n0 = data["nodes"][0] if data.get("nodes") else {}
    check("Node has pagerank",      "pagerank"     in n0, str(list(n0.keys())[:8]))
    check("Node has betweenness",   "betweenness"  in n0)
    check("Node has community_id",  "community_id" in n0)
    check("Node has eff_importance","effective_importance" in n0)
    # v2 analytics fields
    a = data.get("analytics", {})
    for field in ["fiedler_value","small_world_index","modularity","n_communities",
                  "health_score","health_grade","top_pagerank_node"]:
        check(f"analytics.{field}", field in a)


def test_add_and_evaluate():
    print("\n📌 API: POST /nodes (add + evaluate)")
    data, status = post("/nodes", {
        "label": "Integration Test Concept",
        "node_type": "potential",
        "importance": 1.2,
        "auto_evaluate": True,
        "auto_connect": True,
    })
    check("Status 200", status == 200, str(status))
    check("Node returned",       "node" in data)
    check("Evaluation returned", "evaluation" in data and data["evaluation"] is not None)

    node = data.get("node", {})
    ev   = data.get("evaluation", {})
    check("Node has id",          "id" in node)
    check("Node label correct",   node.get("label") == "Integration Test Concept")
    check("Decision is valid",    ev.get("decision") in ("ACCEPT","POTENTIAL","REJECT"))
    check("Score is 0-1",         0 <= ev.get("pattern_match_score", -1) <= 1)
    check("factor_breakdown present", "factor_breakdown" in ev)
    check("Reasoning non-empty",  len(ev.get("reasoning","")) > 0)

    return node.get("id")


def test_get_single_node(node_id):
    print(f"\n📌 API: GET /nodes/{node_id}")
    data, status = get(f"/nodes/{node_id}")
    check("Status 200",      status == 200)
    check("Correct node id", data.get("id") == node_id)
    check("Has label",       "label" in data)
    check("Has pagerank",    "pagerank" in data)


def test_evaluate_endpoint(node_id):
    print(f"\n📌 API: POST /nodes/{node_id}/evaluate")
    data, status = post(f"/nodes/{node_id}/evaluate")
    check("Status 200",       status == 200)
    check("Decision present", data.get("decision") in ("ACCEPT","POTENTIAL","REJECT"))
    check("factor_breakdown", "factor_breakdown" in data)
    check("Score 0-1",        0 <= data.get("pattern_match_score",-1) <= 1)


def test_activate_node(node_id):
    print(f"\n📌 API: POST /nodes/{node_id}/activate")
    data, status = post(f"/nodes/{node_id}/activate")
    check("Status 200",          status == 200)
    check("activation_spread",   "activation_spread" in data)
    check("nodes_reached >= 1",  data.get("nodes_reached", 0) >= 1)
    print(f"    → Reached {data.get('nodes_reached')} nodes")


def test_similar_nodes(node_id):
    print(f"\n📌 API: GET /nodes/{node_id}/similar")
    data, status = get(f"/nodes/{node_id}/similar?k=4")
    check("Status 200",     status == 200)
    check("similar list",   isinstance(data.get("similar"), list))
    check("results present", len(data.get("similar",[])) > 0)
    if data.get("similar"):
        s0 = data["similar"][0]
        check("Has label",             "label" in s0)
        check("Has semantic_sim",      "semantic_similarity" in s0)
        check("Has combined_score",    "combined_score" in s0)
        check("Has community",         "community" in s0)


def test_add_edge():
    print("\n📌 API: POST /edges")
    # Get first two node IDs
    graph_data, _ = get("/graph")
    nodes = graph_data.get("nodes", [])
    if len(nodes) < 2:
        print("  ⚠️  Not enough nodes to test edge creation")
        return
    n1, n2 = nodes[0]["id"], nodes[-1]["id"]
    data, status = post("/edges", {"from_id": n1, "to_id": n2, "strength": 0.7})
    check("Status 200 or already exists", status == 200, str(status))
    check("from_id present", "from_id" in data)
    check("strength present", "strength" in data)


def test_patterns():
    print("\n📌 API: GET /patterns")
    data, status = get("/patterns")
    check("Status 200",       status == 200)
    check("patterns list",    isinstance(data.get("patterns"), list))
    check("count present",    "count" in data)
    check("count matches",    data["count"] == len(data.get("patterns",[])))
    if data.get("patterns"):
        p0 = data["patterns"][0]
        check("Has node_ids",  "node_ids" in p0)
        check("Has labels",    "labels"   in p0)
        check("Has centroid",  "centroid" in p0)
        check("Has density",   "density"  in p0)
        check("Has anchor",    "anchor_node" in p0)


def test_analytics():
    print("\n📌 API: GET /analytics")
    data, status = get("/analytics")
    check("Status 200", status == 200)
    for f in ["total_nodes","total_edges","health_score","health_grade",
              "fiedler_value","small_world_index","modularity","n_communities",
              "n_bridges","top_pagerank_node","top_betweenness_node"]:
        check(f"analytics.{f}", f in data)


def test_topology():
    print("\n📌 API: GET /topology")
    data, status = get("/topology")
    check("Status 200", status == 200)
    for f in ["fiedler","small_world_index","modularity","n_communities",
              "n_components","graph_entropy","n_bridges","suggested_links"]:
        check(f"topology.{f}", f in data)
    check("graph_entropy has efficiency", "efficiency" in data.get("graph_entropy",{}))
    check("suggested_links is list",      isinstance(data.get("suggested_links",[]), list))


def test_health():
    print("\n📌 API: GET /health")
    data, status = get("/health")
    check("Status 200",        status == 200)
    check("Has score",         "score"   in data)
    check("Has grade",         "grade"   in data)
    check("Has breakdown",     "breakdown" in data)
    check("Has context",       "context"   in data)
    check("Score in [0,100]",  0 <= data.get("score",-1) <= 100)
    check("Grade is A-F",      data.get("grade") in ("A","B","C","D","F"))
    print(f"    → {data.get('score')}/100 grade {data.get('grade')}")


def test_advice():
    print("\n📌 API: GET /advice")
    data, status = get("/advice")
    check("Status 200",       status == 200)
    check("Has advice list",  isinstance(data.get("advice"), list))
    check("Has health",       "health" in data)
    check("Advice non-empty", len(data.get("advice",[])) > 0)
    if data.get("advice"):
        a0 = data["advice"][0]
        check("Has priority", "priority" in a0)
        check("Has area",     "area"     in a0)
        check("Has issue",    "issue"    in a0)
        check("Has action",   "action"   in a0)
        check("Priority valid", a0.get("priority") in ("HIGH","MEDIUM","LOW","INFO"))
    print(f"    → {len(data.get('advice',[]))} recommendations")
    for a in data.get("advice",[]):
        print(f"      [{a['priority']}] {a['area']}: {a['issue'][:60]}")


def test_recommend():
    print("\n📌 API: GET /recommend")
    data, status = get("/recommend?k=4")
    check("Status 200",         status == 200)
    check("recommendations",    "recommendations" in data)
    check("count present",      "count" in data)
    if data.get("recommendations"):
        r0 = data["recommendations"][0]
        check("Has node_id",       "node_id"       in r0)
        check("Has label",         "label"         in r0)
        check("Has frontier_score","frontier_score" in r0)
        check("Has eval_decision", "eval_decision"  in r0)
        check("Has surprise",      "surprise"       in r0)
        check("Score 0-1",         0 <= r0.get("frontier_score",-1) <= 1)


def test_bridges():
    print("\n📌 API: GET /bridges")
    data, status = get("/bridges")
    check("Status 200",    status == 200)
    check("bridges list",  isinstance(data.get("bridges"), list))
    check("count present", "count" in data)
    if data.get("bridges"):
        b0 = data["bridges"][0]
        check("Has from_id",    "from_id"    in b0)
        check("Has to_id",      "to_id"      in b0)
        check("Has from_label", "from_label" in b0)
        check("Has to_label",   "to_label"   in b0)
    print(f"    → {data.get('count')} critical bridges")


def test_suggest_links():
    print("\n📌 API: GET /suggest-links")
    data, status = get("/suggest-links?k=5")
    check("Status 200",     status == 200)
    check("suggestions",    "suggestions" in data)
    check("count present",  "count" in data)
    if data.get("suggestions"):
        s0 = data["suggestions"][0]
        check("Has from_id",    "from_id"    in s0)
        check("Has to_id",      "to_id"      in s0)
        check("Has from_label", "from_label" in s0)
        check("Has score",      "score"      in s0)
    print(f"    → {data.get('count')} predicted links")


def test_apply_suggestion():
    print("\n📌 API: POST /apply-suggested-link")
    # Get a suggestion
    sdata, _ = get("/suggest-links?k=1")
    suggestions = sdata.get("suggestions", [])
    if not suggestions:
        print("  ⚠️  No suggestions available, skipping")
        return
    s = suggestions[0]
    data, status = post(f"/apply-suggested-link/{s['from_id']}/{s['to_id']}?strength=0.4")
    check("Status 200",       status == 200, str(status))
    check("from_id returned", data.get("from_id") == s["from_id"])
    check("strength present", "strength" in data)
    print(f"    → Connected {s['from_label']} ↔ {s['to_label']}")


def test_export():
    print("\n📌 API: GET /export")
    data, status = get("/export")
    check("Status 200",            status == 200)
    check("Has version",           "version" in data)
    check("Has exported_at",       "exported_at" in data)
    check("Has nodes",             isinstance(data.get("nodes"), list))
    check("Has edges",             isinstance(data.get("edges"), list))
    check("Has analytics",         isinstance(data.get("analytics"), dict))
    check("Has health",            isinstance(data.get("health"), dict))
    check("Has communities",       "communities" in data)
    check("Nodes non-empty",       len(data.get("nodes",[])) > 0)
    check("Edges non-empty",       len(data.get("edges",[])) > 0)


def test_decay():
    print("\n📌 API: POST /decay")
    data, status = post("/decay")
    check("Status 200",      status == 200)
    check("decayed count",   "decayed" in data)
    check("importances dict",isinstance(data.get("importances"), dict))
    check("decayed > 0",     data.get("decayed",0) > 0)
    print(f"    → Decayed {data.get('decayed')} nodes")


def test_history():
    print("\n📌 API: GET /history")
    data, status = get("/history")
    check("Status 200",    status == 200)
    check("history list",  isinstance(data.get("history"), list))


def test_evolution():
    print("\n📌 API: GET /evolution")
    data, status = get("/evolution")
    check("Status 200",     status == 200)
    check("snapshots list", isinstance(data.get("snapshots"), list))
    if data.get("snapshots"):
        s0 = data["snapshots"][0]
        check("Has timestamp",   "timestamp"   in s0)
        check("Has total_nodes", "total_nodes" in s0)
        check("Has health_score","health_score" in s0)
        check("Has modularity",  "modularity"   in s0)


def test_404_handling():
    print("\n📌 API: 404 error handling")
    data, status = get("/nodes/99999")
    check("Returns 404",      status == 404)
    check("Has detail field", "detail" in data)


def test_promote_and_delete(node_id):
    print(f"\n📌 API: Promote + Delete node {node_id}")
    # First check if it's potential
    n, status = get(f"/nodes/{node_id}")
    if n.get("node_type") == "potential":
        promo, ps = post(f"/nodes/{node_id}/promote")
        check("Promote status 200",   ps == 200)
        check("Promoted node active", promo.get("node",{}).get("node_type") == "active")

    # Delete
    d, ds = delete(f"/nodes/{node_id}")
    check("Delete status 200", ds == 200)
    check("Deleted id matches", d.get("deleted") == node_id)

    # Confirm gone
    gone, gs = get(f"/nodes/{node_id}")
    check("Node is gone (404)", gs == 404)


def test_snapshot():
    print("\n📌 API: POST /snapshot")
    data, status = post("/snapshot")
    check("Status 200",       status == 200)
    check("Has health_score", "health_score" in data)
    check("Has timestamp",    "timestamp"    in data)
    check("Has total_nodes",  "total_nodes"  in data)


def test_end_to_end_workflow():
    print("\n📌 API: End-to-End Workflow")
    # 1. Add a node
    n, _ = post("/nodes", {"label": "E2E Test Node", "auto_evaluate": True, "auto_connect": True})
    nid  = n.get("node", {}).get("id")
    check("Node created", nid is not None)

    # 2. Activate it
    act, _ = post(f"/nodes/{nid}/activate")
    check("Activation spreads", act.get("nodes_reached", 0) >= 1)

    # 3. Check graph reflects it
    g, _ = get("/graph")
    ids  = [nd["id"] for nd in g.get("nodes", [])]
    check("Node in graph", nid in ids)

    # 4. Take snapshot
    snap, _ = post("/snapshot")
    check("Snapshot captured", "health_score" in snap)

    # 5. Export and verify
    exp, _ = get("/export")
    exp_ids = [nd["id"] for nd in exp.get("nodes", [])]
    check("Node in export", nid in exp_ids)

    # 6. Get advice
    adv, _ = get("/advice")
    check("Advice present", len(adv.get("advice", [])) > 0)

    # 7. Delete cleanup
    d, _ = delete(f"/nodes/{nid}")
    check("Cleanup: node deleted", d.get("deleted") == nid)


# ─── MAIN ─────────────────────────────────────
def test_concept_path():
    print("\n📌 API: GET /path/{from_id}/{to_id}")
    graph_data, _ = get("/graph")
    nodes = graph_data.get("nodes", [])
    if len(nodes) < 2:
        print("  ⚠️  Not enough nodes")
        return
    n1, n2 = nodes[0]["id"], nodes[4]["id"]
    data, status = get(f"/path/{n1}/{n2}")
    check("Status 200",       status == 200, str(status))
    check("found=True",       data.get("found") is True)
    check("hops list",        isinstance(data.get("hops"), list))
    check("length >= 1",      data.get("length", 0) >= 1)
    check("path_ids list",    isinstance(data.get("path_ids"), list))
    check("avg_strength 0-1", 0 <= data.get("avg_strength", -1) <= 1)
    if data.get("hops"):
        h0 = data["hops"][0]
        check("hop has label",    "label"     in h0)
        check("hop has node_id",  "node_id"   in h0)
        check("hop has pagerank", "pagerank"  in h0)
    print(f"    → length={data.get('length')}  cost={data.get('total_cost')}")

    # Test 404 on disconnected
    _, status404 = get("/path/9998/9999")
    check("Missing nodes → 404", status404 == 404)


def test_duplicates():
    print("\n📌 API: GET /duplicates")
    data, status = get("/duplicates?threshold=0.85")
    check("Status 200",       status == 200)
    check("duplicates list",  isinstance(data.get("duplicates"), list))
    check("count present",    "count"     in data)
    check("threshold echo",   "threshold" in data)
    if data.get("duplicates"):
        d0 = data["duplicates"][0]
        check("Has node_a_id",    "node_a_id"    in d0)
        check("Has node_b_id",    "node_b_id"    in d0)
        check("Has similarity",   "similarity"   in d0)
        check("Has recommendation","recommendation" in d0)
        check("Sim 0-1",          0 <= d0.get("similarity",-1) <= 1)
    print(f"    → {data.get('count')} duplicate pairs at threshold={data.get('threshold')}")


def test_named_snapshot():
    print("\n📌 API: POST /snapshot/{name}")
    data, status = post("/snapshot/integration-test-snap")
    check("Status 200",       status == 200)
    check("Has name",         data.get("name") == "integration-test-snap")
    check("Has timestamp",    "timestamp"    in data)
    check("Has health_score", "health_score" in data)
    check("Has total_nodes",  "total_nodes"  in data)
    check("Has modularity",   "modularity"   in data)

    # Verify it appears in evolution history
    evo, _ = get("/evolution")
    names = [s.get("name","") for s in evo.get("snapshots",[])]
    check("Appears in history", "integration-test-snap" in names)


def test_export_graphml():
    print("\n📌 API: GET /export/graphml")
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(BASE + "/export/graphml", timeout=8) as r:
            xml = r.read().decode()
        check("Status 200",      True)
        check("XML declaration", xml.startswith("<?xml"))
        check("graphml tag",     "<graphml" in xml)
        check("node elements",   "<node id=" in xml)
        check("edge elements",   "<edge id=" in xml)
        print(f"    → {len(xml.splitlines())} lines")
    except Exception as e:
        check("XML export reachable", False, str(e))


def test_export_dot():
    print("\n📌 API: GET /export/dot")
    import urllib.request
    try:
        with urllib.request.urlopen(BASE + "/export/dot", timeout=8) as r:
            content = r.read().decode()
        check("DOT graph header", content.startswith("graph ThoughtGraph"))
        check("Has node entries",  " -- " in content or "[label=" in content)
        check("Has closing brace", content.strip().endswith("}"))
        print(f"    → {len(content.splitlines())} lines")
    except Exception as e:
        check("DOT content readable", False, str(e))


def test_merge_nodes():
    print("\n📌 API: POST /merge/{keep}/{remove}")
    # Add two near-identical nodes to merge
    n1, _ = post("/nodes", {"label": "RL Agent Alpha", "node_type": "active", "auto_evaluate": False})
    n2, _ = post("/nodes", {"label": "RL Agent Beta",  "node_type": "active", "auto_evaluate": False})
    id1 = n1.get("node", {}).get("id")
    id2 = n2.get("node", {}).get("id")
    if id1 is None or id2 is None:
        print("  ⚠️  Could not create test nodes")
        return

    # Connect them both to an existing node
    graph_data, _ = get("/graph")
    existing_id = graph_data["nodes"][0]["id"]
    post("/edges", {"from_id": id1, "to_id": existing_id, "strength": 0.5})
    post("/edges", {"from_id": id2, "to_id": existing_id, "strength": 0.4})

    # Merge
    data, status = post(f"/merge/{id1}/{id2}")
    check("Status 200",        status == 200)
    check("merged=True",       data.get("merged") is True)
    check("kept id correct",   data.get("kept") == id1)
    check("removed id echoed", data.get("removed") == id2)
    check("kept node returned","node" in data)

    # Verify remove is gone
    gone, gs = get(f"/nodes/{id2}")
    check("Removed node gone (404)", gs == 404)
    # Verify keep still exists
    kept, ks = get(f"/nodes/{id1}")
    check("Kept node exists", ks == 200)

    # Cleanup
    delete(f"/nodes/{id1}")


if __name__ == "__main__":
    print("=" * 65)
    print("  THOUGHTGRAPH v2.1 — API INTEGRATION TESTS")
    print("=" * 65)

    # Try to connect; optionally start server
    print("\nConnecting to API...")
    server_proc = None

    if not wait_for_server(timeout=3):
        print("  Starting server automatically...")
        server_proc = subprocess.Popen(
            [sys.executable, "/home/claude/api.py"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not wait_for_server(timeout=15):
            print("  ❌ Could not start API server")
            sys.exit(1)
        print("  ✅ Server started")
    else:
        print("  ✅ Server already running")

    try:
        # Reset first for clean state
        post("/reset")
        time.sleep(0.3)

        test_root()
        test_get_graph()
        node_id = test_add_and_evaluate()
        if node_id is not None:
            test_get_single_node(node_id)
            test_evaluate_endpoint(node_id)
            test_activate_node(node_id)
            test_similar_nodes(node_id)
        test_add_edge()
        test_patterns()
        test_analytics()
        test_topology()
        test_health()
        test_advice()
        test_recommend()
        test_bridges()
        test_suggest_links()
        test_apply_suggestion()
        test_export()
        test_decay()
        test_history()
        test_evolution()
        test_404_handling()
        if node_id is not None:
            test_promote_and_delete(node_id)
        test_snapshot()
        test_concept_path()
        test_duplicates()
        test_named_snapshot()
        test_export_graphml()
        test_export_dot()
        test_merge_nodes()
        test_end_to_end_workflow()

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()

    total = PASS + FAIL
    print("\n" + "=" * 65)
    print(f"  RESULTS: {PASS}/{total} passed  |  {FAIL} failed")
    print("=" * 65)
    sys.exit(0 if FAIL == 0 else 1)
