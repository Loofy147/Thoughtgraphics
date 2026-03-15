## 2026-03-24 - [Batch Import O(N²) Disk I/O]
**Learning:** In projects using simple JSON file persistence, every write operation usually involves serializing the entire state. Functions like `batch_import` that loop over `add_node` and `connect` (both of which call `_save()`) will result in O(N²) disk I/O and CPU time for serialization.
**Action:** Use a `_batch_mode` flag or context manager to suppress `_save()` calls during bulk operations, then call `_save()` exactly once at the end.

## 2026-03-24 - [Auto-evaluation O(N⁴) Bottleneck]
**Learning:** In bulk imports with auto-evaluation, recomputing pairwise baseline similarities for the entire graph and full topology metrics for every node leads to O(N⁴) or O(N³) complexity depending on the underlying graph algorithms.
**Action:** 1) Cache baseline similarities once per batch. 2) Use cached topology metrics if they are "fresh enough" during a batch operation. 3) Vectorize similarity calculations with numpy for a ~10x speedup in the hot path.
