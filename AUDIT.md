# ThoughtGraph System Audit & Roadmap

## 1. Genuine Gaps Mapped
- **Algorithmic Bottlenecks**:
    - `find_duplicates` is O(N^2), which will fail as the graph grows.
    - `_compute_baseline_similarity` computes a full N^2 matrix in memory.
    - `find_nearest` is O(N) per call, making it slow for large graphs.
- **Architectural Weaknesses**:
    - **Persistence**: Relies on a single monolithic `thought_graph_data.json`. No transactional integrity or partial updates.
    - **Concurrency**: `ThoughtGraph` has a "lock" placeholder but is not truly thread-safe for high-concurrency API usage.
    - **Pathing**: `api.py` had hardcoded `/home/claude/` paths (partially fixed).
- **Code Quality & Testing**:
    - Missing `test_thought_graph.py` which is imported by `test_v2.py`.
    - `api.py` error handling is repetitive and lacks a global error handler.
    - Documentation is scattered across Docx and HTML comments.

## 2. Structural Weaknesses
- **Memory Usage**: Storing all embeddings in a dictionary of objects is memory-intensive.
- **Dependency Management**: No `requirements.txt` file, leading to manual dependency discovery.
- **UI/Backend Coupling**: UI is a single large HTML file with embedded CSS/JS, making it hard to maintain.

## 3. High-Impact Improvements (Prioritized)
### Phase 1: Foundation & Performance (Current)
- [ ] Restore full test suite (`test_thought_graph.py`).
- [ ] Vectorize `find_nearest` and `compute_baseline_similarity` using NumPy.
- [ ] Fix all path consistency issues in `api.py`.
- [ ] Create `requirements.txt`.

### Phase 2: Scalability
- [ ] Implement a more robust persistence layer (e.g., SQLite or a vector-aware DB).
- [ ] Add spatial indexing or FAISS for faster semantic search.
- [ ] Decouple UI into a modern frontend framework.

### Phase 3: Intelligence
- [ ] Multi-perspective clustering (beyond Louvain).
- [ ] Real-time graph evolution visualization enhancements.
- [ ] Enhanced Hebbian learning with long-term potentiation (LTP).

## 4. Documentation
- [ ] Extract all method signatures and auto-generate `API_REFERENCE.md`.
