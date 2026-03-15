# ThoughtGraph v2.1

ThoughtGraph is an advanced graph-based knowledge management and exploration engine. It uses N-gram embeddings for semantic similarity, PageRank for influence mapping, Louvain for community detection, and Hebbian learning for dynamic edge strengthening based on activation patterns.

## Features

- **Semantic Mapping:** Character N-gram hashing embeddings (512 dims) for high-fidelity concept matching.
- **Topology Analysis:** Full GraphAnalyzer with PageRank, Betweenness Centrality, Fiedler Value, Small-World Index, and Graph Entropy.
- **Dynamic Evolution:** Hebbian learning updates edge strengths based on node activation overlap.
- **Temporal Engine:** Effective importance of nodes decays over time unless refreshed by activation.
- **Auto-Healing:** Missing-link prediction via Adamic-Adar index to improve graph connectivity.
- **Full API:** FastAPI-powered REST interface for all graph operations and analysis.
- **3D Visualization:** Integrated Three.js UI for real-time graph exploration.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd thoughtgraph
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy networkx fastapi uvicorn pydantic scipy pytest
    ```

## Usage

### Starting the API Server
```bash
python api.py
```
The API will be available at `http://localhost:8000`.
The UI can be accessed at `http://localhost:8000/ui`.

### Running the Training Script
To demonstrate Hebbian learning and edge strengthening:
```bash
python train_graph.py
```

### Running Tests
We provide two test suites:
1.  **Algorithm Tests:** `python -m pytest test_v2.py`
2.  **Integration Tests:** `python test_integration.py` (This will start/stop the server automatically)

Alternatively, use the provided helper:
```bash
chmod +x run_tests.sh
./run_tests.sh
```

## API Reference

### Core Endpoints
- `GET /`: API status and health summary.
- `GET /graph`: Full graph data (nodes, edges, analytics).
- `POST /nodes`: Add a new node (supports auto-evaluation and auto-connection).
- `POST /nodes/{id}/activate`: Trigger activation spreading from a node.
- `GET /topology`: Comprehensive topological report.
- `GET /health`: Detailed graph health metrics and breakdown.
- `GET /advice`: Actionable recommendations for improving graph health.
- `POST /heal`: Automatically apply predicted links to improve connectivity.

For a full list of endpoints and method signatures, see `api_docs_reference.txt`.

## Implementation Details

- **Embeddings:** Character 2-5 gram hashing with 4 salt functions.
- **Evaluation:** 5-factor scoring (Semantic, Community, PageRank Influence, Bridging, Novelty).
- **Learning:** `lr=0.04`, `depression=0.005` for Hebbian updates.
- **Decay:** `rate=0.015` (exponential decay per hour).
