## 2026-03-24 - [Batch Import O(N²) Disk I/O]
**Learning:** In projects using simple JSON file persistence, every write operation usually involves serializing the entire state. Functions like `batch_import` that loop over `add_node` and `connect` (both of which call `_save()`) will result in O(N²) disk I/O and CPU time for serialization.
**Action:** Use a `_batch_mode` flag or context manager to suppress `_save()` calls during bulk operations, then call `_save()` exactly once at the end.
