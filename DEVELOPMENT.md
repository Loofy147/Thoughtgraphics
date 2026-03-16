# ThoughtGraph: Internal Engineering Notes

## Architecture Overview
- **Core Engine (`thought_graph.py`):** Handles the graph data structure, NetworkX integration, and ML embeddings.
- **Thread Safety:** Uses a Reentrant Lock (`threading.RLock`) and an `@atomic` decorator to ensure consistent state during concurrent API requests.
- **API Layer (`api.py`):** FastAPI-based REST service.

## Performance Considerations
- **Blocking Analytics:** Heavy NetworkX operations currently block the FastAPI event loop. Future iterations should move these to a thread pool.
- **Persistence:** Full JSON serialization occurs on every write. For large graphs, this is the primary bottleneck.

## Debugging & Error Handling
- Bare `except:` blocks have been replaced with `except Exception:`.
- Log output should be monitored during training and auto-heal cycles.

## Roadmap Highlights (v3.0)
- SQL-based persistence.
- Pluggable embedding providers.
- Interactive 3D manipulation.
