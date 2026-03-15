# ThoughtGraph: Gap Analysis & Roadmap

This document outlines the genuine gaps identified during the v2.1 audit and prioritizes the next development phase.

## Genuine Gaps Identified

### 1. Functional Gaps
- **Persistence Layer:** Currently uses a single monolithic JSON file (`thought_graph_data.json`). This will not scale for large graphs (>10k nodes) and is prone to corruption on concurrent writes.
- **Concurrent Access:** The `ThoughtGraph` class lacks robust thread-safety (locks) for multi-user or high-concurrency API environments.
- **Advanced Querying:** No support for complex graph queries (e.g., "find all nodes in community X with PageRank > Y connected to node Z").
- **Directed Relationships:** While metadata exists, most algorithms treat edges as undirected.

### 2. Algorithmic Gaps
- **Incremental Updates:** Topological metrics (PageRank, Communities) are recomputed from scratch on every change. This is inefficient for large graphs.
- **Embedding Fidelity:** N-gram hashing is fast and zero-dependency but lacks the deep semantic nuance of LLM-based embeddings (e.g., Ada-002 or BERT).
- **Adaptive Learning:** Hebbian update parameters (`lr`, `depression`) are hardcoded and do not adapt to graph density.

### 3. Documentation & DX Gaps
- **Client SDK:** No official Python or JS client library; users must interface with raw REST.
- **Visualizer Interactivity:** The 3D UI is a "read-only" visualization; it lacks interactive editing (add/remove/connect nodes directly in 3D).

---

## Prioritized Next Development Phase (v3.0)

### Phase 1: Infrastructure (High Priority)
1.  **Database Migration:** Move from JSON to a proper graph-capable storage (e.g., SQLite with a graph schema or PostgreSQL + Age).
2.  **Concurrency Control:** Implement read/write locks for core graph data structures.

### Phase 2: Performance (Medium Priority)
1.  **Incremental Analytics:** Implement incremental PageRank updates and dynamic community detection.
2.  **Batch Processing:** Optimize the API for bulk operations.

### Phase 3: Intelligence (Medium Priority)
1.  **Pluggable Embeddings:** Allow users to swap N-gram hashing for OpenAI/HuggingFace embeddings via configuration.
2.  **Relationship Classification:** Automate the classification of edge types based on semantic overlap.

### Phase 4: UX (Low Priority)
1.  **Interactive UI:** Enable node manipulation directly within the 3D Three.js environment.
2.  **API Client:** Release a lightweight Python wrapper for the FastAPI endpoints.
