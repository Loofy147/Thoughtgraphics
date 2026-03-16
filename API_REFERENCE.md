# ThoughtGraph API Reference

## Module: thought_graph

### Classes

#### `ActivationEngine`
| Method | Signature | Description |
| --- | --- | --- |
| `hebbian_update` | `(self, activation, edges, lr=0.04, depression=0.005)` |  |
| `spread` | `(self, source_ids, nodes, edges, decay=0.55, steps=4, threshold=0.02)` |  |

#### `EvaluationResult`
EvaluationResult(node_id: int, decision: str, pattern_match_score: float, nearest_neighbors: list, reasoning: str, suggested_connections: list, factor_breakdown: dict = <factory>)

| Method | Signature | Description |
| --- | --- | --- |
| `__eq__` | `(self, other)` |  |
| `__init__` | `(self, node_id: int, decision: str, pattern_match_score: float, nearest_neighbors: list, reasoning: str, suggested_connections: list, factor_breakdown: dict = <factory>) -> None` |  |
| `__repr__` | `(self)` |  |

#### `GraphAnalyzer`
| Method | Signature | Description |
| --- | --- | --- |
| `__init__` | `(self, nodes, edges)` |  |
| `betweenness` | `(self)` |  |
| `bridges` | `(self)` | Edges whose removal disconnects the graph. |
| `burt_constraint` | `(self)` |  |
| `closeness` | `(self)` |  |
| `clustering` | `(self)` |  |
| `communities` | `(self, seed=42)` |  |
| `eigenvector` | `(self)` |  |
| `entropy` | `(self)` |  |
| `fiedler` | `(self)` |  |
| `full_report` | `(self)` |  |
| `hits` | `(self)` |  |
| `link_prediction` | `(self, k=10)` | Top-K missing edges by Adamic-Adar index. |
| `modularity` | `(self, coms)` |  |
| `pagerank` | `(self)` |  |
| `small_world` | `(self)` |  |

#### `TemporalEngine`
| Method | Signature | Description |
| --- | --- | --- |
| `activate` | `(self, node)` |  |
| `decay_all` | `(self, nodes, rate=0.015, floor=0.1)` |  |
| `recency_weight` | `(self, node)` |  |

#### `ThoughtEdge`
ThoughtEdge(from_id: int, to_id: int, strength: float = 0.5, edge_type: str = 'connection', created_at: float = <factory>, last_activated: float = 0.0, activation_count: int = 0)

| Method | Signature | Description |
| --- | --- | --- |
| `__eq__` | `(self, other)` |  |
| `__init__` | `(self, from_id: int, to_id: int, strength: float = 0.5, edge_type: str = 'connection', created_at: float = <factory>, last_activated: float = 0.0, activation_count: int = 0) -> None` |  |
| `__repr__` | `(self)` |  |

#### `ThoughtGraph`
| Method | Signature | Description |
| --- | --- | --- |
| `__init__` | `(self, persist=True)` |  |
| `_load` | `(self)` |  |
| `_save` | `(self)` |  |
| `activate_node` | `(self, node_id, spread=True)` |  |
| `add_node` | `(self, label, x=None, y=None, z=None, node_type='active', depth=1, parent_id=None, tags=None, importance=1.0)` |  |
| `auto_heal_graph` | `(self, max_links: int = 8, min_score: float = 0.5) -> dict` | Automatically apply top Adamic-Adar predicted links to reduce bridge count |
| `batch_import` | `(self, items: list, auto_evaluate: bool = True) -> dict` | Add multiple nodes at once. |
| `batch_operation` | `(self)` | Context manager for bulk updates to suppress redundant saves and caching. |
| `compute_surprise` | `(self, node)` |  |
| `concept_path` | `(self, from_id: int, to_id: int) -> dict` | Shortest weighted path between two nodes (Dijkstra on 1-strength costs). |
| `connect` | `(self, from_id, to_id, strength=0.5, edge_type='connection')` |  |
| `decay_graph` | `(self)` |  |
| `detect_patterns` | `(self)` |  |
| `evaluate_new_node` | `(self, node)` |  |
| `export_dot` | `(self) -> str` | Export as DOT format (Graphviz compatible). |
| `export_graphml` | `(self) -> str` | Export as GraphML (Gephi, yEd, NetworkX compatible). |
| `find_bridges` | `(self)` | Edges whose removal would disconnect the graph. |
| `find_duplicates` | `(self, threshold: float = 0.88) -> list` | Find semantically near-identical node pairs (potential duplicates). |
| `find_nearest` | `(self, node, k=7, exclude_types=None)` |  |
| `get_all_nodes` | `(self)` |  |
| `get_community_subgraph` | `(self, community_id: int) -> dict` | Extract all nodes and internal edges for a specific Louvain community. |
| `get_edges` | `(self)` |  |
| `get_evolution_history` | `(self)` |  |
| `get_node` | `(self, node_id)` |  |
| `get_topology` | `(self, force=False)` |  |
| `graph_analytics` | `(self)` |  |
| `graph_diff` | `(self, snapshot_a: dict, snapshot_b: dict) -> dict` | Compare two evolution snapshots (from get_evolution_history()). |
| `graph_health_advice` | `(self) -> list` | Actionable recommendations to improve graph health. |
| `graph_health_score` | `(self)` | Calibrated health score 0-100 using density-aware normalization. |
| `merge_nodes` | `(self, keep_id: int, remove_id: int) -> bool` | Merge two nodes: transfer connections from remove_id to keep_id. |
| `promote_potential` | `(self, node_id)` |  |
| `recommend_exploration` | `(self, k=5)` | Rank potential nodes by frontier breakthrough score. |
| `record_snapshot` | `(self)` | Save current health metrics as a timestamped snapshot. |
| `remove_node` | `(self, node_id)` |  |
| `reset` | `(self)` |  |
| `save_snapshot` | `(self, name: str) -> dict` | Save a named checkpoint of the current graph state. |
| `search_nodes` | `(self, query: str = '', node_type: str = None, min_importance: float = 0.0, community_id: int = None, min_pagerank: float = 0.0, tags: list = None, limit: int = 20) -> list` | Search and filter nodes by multiple criteria. |
| `seed_default_graph` | `(self)` |  |
| `suggest_connections` | `(self, k=5)` | Missing-link prediction via Adamic-Adar index. |
| `to_dict` | `(self)` |  |
| `update_node_importance` | `(self, node_id, importance)` |  |

#### `ThoughtNode`
ThoughtNode(id: int, label: str, x: float, y: float, z: float, node_type: str, depth: int = 0, importance: float = 1.0, effective_importance: float = 1.0, parent_id: Optional[int] = None, children_ids: list = <factory>, connections: list = <factory>, embedding: list = <factory>, created_at: float = <factory>, last_activated: float = 0.0, activation_count: int = 0, decision_weight: float = 1.0, tags: list = <factory>, community_id: int = -1, pagerank: float = 0.0, betweenness: float = 0.0)

| Method | Signature | Description |
| --- | --- | --- |
| `__eq__` | `(self, other)` |  |
| `__init__` | `(self, id: int, label: str, x: float, y: float, z: float, node_type: str, depth: int = 0, importance: float = 1.0, effective_importance: float = 1.0, parent_id: Optional[int] = None, children_ids: list = <factory>, connections: list = <factory>, embedding: list = <factory>, created_at: float = <factory>, last_activated: float = 0.0, activation_count: int = 0, decision_weight: float = 1.0, tags: list = <factory>, community_id: int = -1, pagerank: float = 0.0, betweenness: float = 0.0) -> None` |  |
| `__post_init__` | `(self)` |  |
| `__repr__` | `(self)` |  |
| `distance_to` | `(self, other)` |  |
| `semantic_similarity` | `(self, other)` |  |

### Functions

| Function | Signature | Description |
| --- | --- | --- |
| `_compute_baseline_similarity` | `(nodes: list) -> tuple` | Compute (median, max) of pairwise similarities among active nodes using numpy. |
| `_fnv1a` | `(text)` |  |
| `cosine_sim` | `(a, b)` |  |
| `make_embedding` | `(label, dims=512)` | Character n-gram hashing embedding with 512 dims and 4 hash functions. |

## Module: api

### Classes

#### `AddNodeRequest`
| Method | Signature | Description |
| --- | --- | --- |
| `__copy__` | `(self) -> 'Self'` | Returns a shallow copy of the model. |
| `__deepcopy__` | `(self, memo: 'dict[int, Any] | None' = None) -> 'Self'` | Returns a deep copy of the model. |
| `__delattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__eq__` | `(self, other: 'Any') -> 'bool'` |  |
| `__getattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__getstate__` | `(self) -> 'dict[Any, Any]'` |  |
| `__init__` | `(self, /, **data: 'Any') -> 'None'` | Create a new model by parsing and validating input data from keyword arguments. |
| `__iter__` | `(self) -> 'TupleGenerator'` | So `dict(model)` works. |
| `__pretty__` | `(self, fmt: 'Callable[[Any], Any]', **kwargs: 'Any') -> 'Generator[Any]'` | Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects. |
| `__replace__` | `(self, **changes: 'Any') -> 'Self'` |  |
| `__repr__` | `(self) -> 'str'` |  |
| `__repr_args__` | `(self) -> '_repr.ReprArgs'` |  |
| `__repr_name__` | `(self) -> 'str'` | Name of the instance's class, used in __repr__. |
| `__repr_recursion__` | `(self, object: 'Any') -> 'str'` | Returns the string representation of a recursive object. |
| `__repr_str__` | `(self, join_str: 'str') -> 'str'` |  |
| `__rich_repr__` | `(self) -> 'RichReprResult'` | Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects. |
| `__setattr__` | `(self, name: 'str', value: 'Any') -> 'None'` |  |
| `__setstate__` | `(self, state: 'dict[Any, Any]') -> 'None'` |  |
| `__str__` | `(self) -> 'str'` |  |
| `_calculate_keys` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_copy_and_set_values` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_iter` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_setattr_handler` | `(self, name: 'str', value: 'Any') -> 'Callable[[BaseModel, str, Any], None] | None'` | Get a handler for setting an attribute on the model instance. |
| `copy` | `(self, *, include: 'AbstractSetIntStr | MappingIntStrAny | None' = None, exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None, update: 'Dict[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | Returns a copy of the model. |
| `dict` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False) -> 'Dict[str, Any]'` |  |
| `json` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, encoder: 'Callable[[Any], Any] | None' = PydanticUndefined, models_as_dict: 'bool' = PydanticUndefined, **dumps_kwargs: 'Any') -> 'str'` |  |
| `model_copy` | `(self, *, update: 'Mapping[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | !!! abstract "Usage Documentation" |
| `model_dump` | `(self, *, mode: "Literal['json', 'python'] | str" = 'python', include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'dict[str, Any]'` | !!! abstract "Usage Documentation" |
| `model_dump_json` | `(self, *, indent: 'int | None' = None, ensure_ascii: 'bool' = False, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'str'` | !!! abstract "Usage Documentation" |
| `model_post_init` | `(self, context: 'Any', /) -> 'None'` | Override this method to perform additional initialization after `__init__` and `model_construct`. |

#### `BatchImportRequest`
| Method | Signature | Description |
| --- | --- | --- |
| `__copy__` | `(self) -> 'Self'` | Returns a shallow copy of the model. |
| `__deepcopy__` | `(self, memo: 'dict[int, Any] | None' = None) -> 'Self'` | Returns a deep copy of the model. |
| `__delattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__eq__` | `(self, other: 'Any') -> 'bool'` |  |
| `__getattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__getstate__` | `(self) -> 'dict[Any, Any]'` |  |
| `__init__` | `(self, /, **data: 'Any') -> 'None'` | Create a new model by parsing and validating input data from keyword arguments. |
| `__iter__` | `(self) -> 'TupleGenerator'` | So `dict(model)` works. |
| `__pretty__` | `(self, fmt: 'Callable[[Any], Any]', **kwargs: 'Any') -> 'Generator[Any]'` | Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects. |
| `__replace__` | `(self, **changes: 'Any') -> 'Self'` |  |
| `__repr__` | `(self) -> 'str'` |  |
| `__repr_args__` | `(self) -> '_repr.ReprArgs'` |  |
| `__repr_name__` | `(self) -> 'str'` | Name of the instance's class, used in __repr__. |
| `__repr_recursion__` | `(self, object: 'Any') -> 'str'` | Returns the string representation of a recursive object. |
| `__repr_str__` | `(self, join_str: 'str') -> 'str'` |  |
| `__rich_repr__` | `(self) -> 'RichReprResult'` | Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects. |
| `__setattr__` | `(self, name: 'str', value: 'Any') -> 'None'` |  |
| `__setstate__` | `(self, state: 'dict[Any, Any]') -> 'None'` |  |
| `__str__` | `(self) -> 'str'` |  |
| `_calculate_keys` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_copy_and_set_values` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_iter` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_setattr_handler` | `(self, name: 'str', value: 'Any') -> 'Callable[[BaseModel, str, Any], None] | None'` | Get a handler for setting an attribute on the model instance. |
| `copy` | `(self, *, include: 'AbstractSetIntStr | MappingIntStrAny | None' = None, exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None, update: 'Dict[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | Returns a copy of the model. |
| `dict` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False) -> 'Dict[str, Any]'` |  |
| `json` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, encoder: 'Callable[[Any], Any] | None' = PydanticUndefined, models_as_dict: 'bool' = PydanticUndefined, **dumps_kwargs: 'Any') -> 'str'` |  |
| `model_copy` | `(self, *, update: 'Mapping[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | !!! abstract "Usage Documentation" |
| `model_dump` | `(self, *, mode: "Literal['json', 'python'] | str" = 'python', include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'dict[str, Any]'` | !!! abstract "Usage Documentation" |
| `model_dump_json` | `(self, *, indent: 'int | None' = None, ensure_ascii: 'bool' = False, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'str'` | !!! abstract "Usage Documentation" |
| `model_post_init` | `(self, context: 'Any', /) -> 'None'` | Override this method to perform additional initialization after `__init__` and `model_construct`. |

#### `BatchItem`
| Method | Signature | Description |
| --- | --- | --- |
| `__copy__` | `(self) -> 'Self'` | Returns a shallow copy of the model. |
| `__deepcopy__` | `(self, memo: 'dict[int, Any] | None' = None) -> 'Self'` | Returns a deep copy of the model. |
| `__delattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__eq__` | `(self, other: 'Any') -> 'bool'` |  |
| `__getattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__getstate__` | `(self) -> 'dict[Any, Any]'` |  |
| `__init__` | `(self, /, **data: 'Any') -> 'None'` | Create a new model by parsing and validating input data from keyword arguments. |
| `__iter__` | `(self) -> 'TupleGenerator'` | So `dict(model)` works. |
| `__pretty__` | `(self, fmt: 'Callable[[Any], Any]', **kwargs: 'Any') -> 'Generator[Any]'` | Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects. |
| `__replace__` | `(self, **changes: 'Any') -> 'Self'` |  |
| `__repr__` | `(self) -> 'str'` |  |
| `__repr_args__` | `(self) -> '_repr.ReprArgs'` |  |
| `__repr_name__` | `(self) -> 'str'` | Name of the instance's class, used in __repr__. |
| `__repr_recursion__` | `(self, object: 'Any') -> 'str'` | Returns the string representation of a recursive object. |
| `__repr_str__` | `(self, join_str: 'str') -> 'str'` |  |
| `__rich_repr__` | `(self) -> 'RichReprResult'` | Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects. |
| `__setattr__` | `(self, name: 'str', value: 'Any') -> 'None'` |  |
| `__setstate__` | `(self, state: 'dict[Any, Any]') -> 'None'` |  |
| `__str__` | `(self) -> 'str'` |  |
| `_calculate_keys` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_copy_and_set_values` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_iter` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_setattr_handler` | `(self, name: 'str', value: 'Any') -> 'Callable[[BaseModel, str, Any], None] | None'` | Get a handler for setting an attribute on the model instance. |
| `copy` | `(self, *, include: 'AbstractSetIntStr | MappingIntStrAny | None' = None, exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None, update: 'Dict[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | Returns a copy of the model. |
| `dict` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False) -> 'Dict[str, Any]'` |  |
| `json` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, encoder: 'Callable[[Any], Any] | None' = PydanticUndefined, models_as_dict: 'bool' = PydanticUndefined, **dumps_kwargs: 'Any') -> 'str'` |  |
| `model_copy` | `(self, *, update: 'Mapping[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | !!! abstract "Usage Documentation" |
| `model_dump` | `(self, *, mode: "Literal['json', 'python'] | str" = 'python', include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'dict[str, Any]'` | !!! abstract "Usage Documentation" |
| `model_dump_json` | `(self, *, indent: 'int | None' = None, ensure_ascii: 'bool' = False, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'str'` | !!! abstract "Usage Documentation" |
| `model_post_init` | `(self, context: 'Any', /) -> 'None'` | Override this method to perform additional initialization after `__init__` and `model_construct`. |

#### `ConnectRequest`
| Method | Signature | Description |
| --- | --- | --- |
| `__copy__` | `(self) -> 'Self'` | Returns a shallow copy of the model. |
| `__deepcopy__` | `(self, memo: 'dict[int, Any] | None' = None) -> 'Self'` | Returns a deep copy of the model. |
| `__delattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__eq__` | `(self, other: 'Any') -> 'bool'` |  |
| `__getattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__getstate__` | `(self) -> 'dict[Any, Any]'` |  |
| `__init__` | `(self, /, **data: 'Any') -> 'None'` | Create a new model by parsing and validating input data from keyword arguments. |
| `__iter__` | `(self) -> 'TupleGenerator'` | So `dict(model)` works. |
| `__pretty__` | `(self, fmt: 'Callable[[Any], Any]', **kwargs: 'Any') -> 'Generator[Any]'` | Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects. |
| `__replace__` | `(self, **changes: 'Any') -> 'Self'` |  |
| `__repr__` | `(self) -> 'str'` |  |
| `__repr_args__` | `(self) -> '_repr.ReprArgs'` |  |
| `__repr_name__` | `(self) -> 'str'` | Name of the instance's class, used in __repr__. |
| `__repr_recursion__` | `(self, object: 'Any') -> 'str'` | Returns the string representation of a recursive object. |
| `__repr_str__` | `(self, join_str: 'str') -> 'str'` |  |
| `__rich_repr__` | `(self) -> 'RichReprResult'` | Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects. |
| `__setattr__` | `(self, name: 'str', value: 'Any') -> 'None'` |  |
| `__setstate__` | `(self, state: 'dict[Any, Any]') -> 'None'` |  |
| `__str__` | `(self) -> 'str'` |  |
| `_calculate_keys` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_copy_and_set_values` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_iter` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_setattr_handler` | `(self, name: 'str', value: 'Any') -> 'Callable[[BaseModel, str, Any], None] | None'` | Get a handler for setting an attribute on the model instance. |
| `copy` | `(self, *, include: 'AbstractSetIntStr | MappingIntStrAny | None' = None, exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None, update: 'Dict[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | Returns a copy of the model. |
| `dict` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False) -> 'Dict[str, Any]'` |  |
| `json` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, encoder: 'Callable[[Any], Any] | None' = PydanticUndefined, models_as_dict: 'bool' = PydanticUndefined, **dumps_kwargs: 'Any') -> 'str'` |  |
| `model_copy` | `(self, *, update: 'Mapping[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | !!! abstract "Usage Documentation" |
| `model_dump` | `(self, *, mode: "Literal['json', 'python'] | str" = 'python', include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'dict[str, Any]'` | !!! abstract "Usage Documentation" |
| `model_dump_json` | `(self, *, indent: 'int | None' = None, ensure_ascii: 'bool' = False, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'str'` | !!! abstract "Usage Documentation" |
| `model_post_init` | `(self, context: 'Any', /) -> 'None'` | Override this method to perform additional initialization after `__init__` and `model_construct`. |

#### `SearchRequest`
| Method | Signature | Description |
| --- | --- | --- |
| `__copy__` | `(self) -> 'Self'` | Returns a shallow copy of the model. |
| `__deepcopy__` | `(self, memo: 'dict[int, Any] | None' = None) -> 'Self'` | Returns a deep copy of the model. |
| `__delattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__eq__` | `(self, other: 'Any') -> 'bool'` |  |
| `__getattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__getstate__` | `(self) -> 'dict[Any, Any]'` |  |
| `__init__` | `(self, /, **data: 'Any') -> 'None'` | Create a new model by parsing and validating input data from keyword arguments. |
| `__iter__` | `(self) -> 'TupleGenerator'` | So `dict(model)` works. |
| `__pretty__` | `(self, fmt: 'Callable[[Any], Any]', **kwargs: 'Any') -> 'Generator[Any]'` | Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects. |
| `__replace__` | `(self, **changes: 'Any') -> 'Self'` |  |
| `__repr__` | `(self) -> 'str'` |  |
| `__repr_args__` | `(self) -> '_repr.ReprArgs'` |  |
| `__repr_name__` | `(self) -> 'str'` | Name of the instance's class, used in __repr__. |
| `__repr_recursion__` | `(self, object: 'Any') -> 'str'` | Returns the string representation of a recursive object. |
| `__repr_str__` | `(self, join_str: 'str') -> 'str'` |  |
| `__rich_repr__` | `(self) -> 'RichReprResult'` | Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects. |
| `__setattr__` | `(self, name: 'str', value: 'Any') -> 'None'` |  |
| `__setstate__` | `(self, state: 'dict[Any, Any]') -> 'None'` |  |
| `__str__` | `(self) -> 'str'` |  |
| `_calculate_keys` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_copy_and_set_values` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_iter` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_setattr_handler` | `(self, name: 'str', value: 'Any') -> 'Callable[[BaseModel, str, Any], None] | None'` | Get a handler for setting an attribute on the model instance. |
| `copy` | `(self, *, include: 'AbstractSetIntStr | MappingIntStrAny | None' = None, exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None, update: 'Dict[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | Returns a copy of the model. |
| `dict` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False) -> 'Dict[str, Any]'` |  |
| `json` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, encoder: 'Callable[[Any], Any] | None' = PydanticUndefined, models_as_dict: 'bool' = PydanticUndefined, **dumps_kwargs: 'Any') -> 'str'` |  |
| `model_copy` | `(self, *, update: 'Mapping[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | !!! abstract "Usage Documentation" |
| `model_dump` | `(self, *, mode: "Literal['json', 'python'] | str" = 'python', include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'dict[str, Any]'` | !!! abstract "Usage Documentation" |
| `model_dump_json` | `(self, *, indent: 'int | None' = None, ensure_ascii: 'bool' = False, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'str'` | !!! abstract "Usage Documentation" |
| `model_post_init` | `(self, context: 'Any', /) -> 'None'` | Override this method to perform additional initialization after `__init__` and `model_construct`. |

#### `UpdateImportanceRequest`
| Method | Signature | Description |
| --- | --- | --- |
| `__copy__` | `(self) -> 'Self'` | Returns a shallow copy of the model. |
| `__deepcopy__` | `(self, memo: 'dict[int, Any] | None' = None) -> 'Self'` | Returns a deep copy of the model. |
| `__delattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__eq__` | `(self, other: 'Any') -> 'bool'` |  |
| `__getattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__getstate__` | `(self) -> 'dict[Any, Any]'` |  |
| `__init__` | `(self, /, **data: 'Any') -> 'None'` | Create a new model by parsing and validating input data from keyword arguments. |
| `__iter__` | `(self) -> 'TupleGenerator'` | So `dict(model)` works. |
| `__pretty__` | `(self, fmt: 'Callable[[Any], Any]', **kwargs: 'Any') -> 'Generator[Any]'` | Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects. |
| `__replace__` | `(self, **changes: 'Any') -> 'Self'` |  |
| `__repr__` | `(self) -> 'str'` |  |
| `__repr_args__` | `(self) -> '_repr.ReprArgs'` |  |
| `__repr_name__` | `(self) -> 'str'` | Name of the instance's class, used in __repr__. |
| `__repr_recursion__` | `(self, object: 'Any') -> 'str'` | Returns the string representation of a recursive object. |
| `__repr_str__` | `(self, join_str: 'str') -> 'str'` |  |
| `__rich_repr__` | `(self) -> 'RichReprResult'` | Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects. |
| `__setattr__` | `(self, name: 'str', value: 'Any') -> 'None'` |  |
| `__setstate__` | `(self, state: 'dict[Any, Any]') -> 'None'` |  |
| `__str__` | `(self) -> 'str'` |  |
| `_calculate_keys` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_copy_and_set_values` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_iter` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_setattr_handler` | `(self, name: 'str', value: 'Any') -> 'Callable[[BaseModel, str, Any], None] | None'` | Get a handler for setting an attribute on the model instance. |
| `copy` | `(self, *, include: 'AbstractSetIntStr | MappingIntStrAny | None' = None, exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None, update: 'Dict[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | Returns a copy of the model. |
| `dict` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False) -> 'Dict[str, Any]'` |  |
| `json` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, encoder: 'Callable[[Any], Any] | None' = PydanticUndefined, models_as_dict: 'bool' = PydanticUndefined, **dumps_kwargs: 'Any') -> 'str'` |  |
| `model_copy` | `(self, *, update: 'Mapping[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | !!! abstract "Usage Documentation" |
| `model_dump` | `(self, *, mode: "Literal['json', 'python'] | str" = 'python', include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'dict[str, Any]'` | !!! abstract "Usage Documentation" |
| `model_dump_json` | `(self, *, indent: 'int | None' = None, ensure_ascii: 'bool' = False, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'str'` | !!! abstract "Usage Documentation" |
| `model_post_init` | `(self, context: 'Any', /) -> 'None'` | Override this method to perform additional initialization after `__init__` and `model_construct`. |

#### `UpdateNodeRequest`
| Method | Signature | Description |
| --- | --- | --- |
| `__copy__` | `(self) -> 'Self'` | Returns a shallow copy of the model. |
| `__deepcopy__` | `(self, memo: 'dict[int, Any] | None' = None) -> 'Self'` | Returns a deep copy of the model. |
| `__delattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__eq__` | `(self, other: 'Any') -> 'bool'` |  |
| `__getattr__` | `(self, item: 'str') -> 'Any'` |  |
| `__getstate__` | `(self) -> 'dict[Any, Any]'` |  |
| `__init__` | `(self, /, **data: 'Any') -> 'None'` | Create a new model by parsing and validating input data from keyword arguments. |
| `__iter__` | `(self) -> 'TupleGenerator'` | So `dict(model)` works. |
| `__pretty__` | `(self, fmt: 'Callable[[Any], Any]', **kwargs: 'Any') -> 'Generator[Any]'` | Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects. |
| `__replace__` | `(self, **changes: 'Any') -> 'Self'` |  |
| `__repr__` | `(self) -> 'str'` |  |
| `__repr_args__` | `(self) -> '_repr.ReprArgs'` |  |
| `__repr_name__` | `(self) -> 'str'` | Name of the instance's class, used in __repr__. |
| `__repr_recursion__` | `(self, object: 'Any') -> 'str'` | Returns the string representation of a recursive object. |
| `__repr_str__` | `(self, join_str: 'str') -> 'str'` |  |
| `__rich_repr__` | `(self) -> 'RichReprResult'` | Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects. |
| `__setattr__` | `(self, name: 'str', value: 'Any') -> 'None'` |  |
| `__setstate__` | `(self, state: 'dict[Any, Any]') -> 'None'` |  |
| `__str__` | `(self) -> 'str'` |  |
| `_calculate_keys` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_copy_and_set_values` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_iter` | `(self, *args: 'Any', **kwargs: 'Any') -> 'Any'` |  |
| `_setattr_handler` | `(self, name: 'str', value: 'Any') -> 'Callable[[BaseModel, str, Any], None] | None'` | Get a handler for setting an attribute on the model instance. |
| `copy` | `(self, *, include: 'AbstractSetIntStr | MappingIntStrAny | None' = None, exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None, update: 'Dict[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | Returns a copy of the model. |
| `dict` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False) -> 'Dict[str, Any]'` |  |
| `json` | `(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, encoder: 'Callable[[Any], Any] | None' = PydanticUndefined, models_as_dict: 'bool' = PydanticUndefined, **dumps_kwargs: 'Any') -> 'str'` |  |
| `model_copy` | `(self, *, update: 'Mapping[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'` | !!! abstract "Usage Documentation" |
| `model_dump` | `(self, *, mode: "Literal['json', 'python'] | str" = 'python', include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'dict[str, Any]'` | !!! abstract "Usage Documentation" |
| `model_dump_json` | `(self, *, indent: 'int | None' = None, ensure_ascii: 'bool' = False, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False) -> 'str'` | !!! abstract "Usage Documentation" |
| `model_post_init` | `(self, context: 'Any', /) -> 'None'` | Override this method to perform additional initialization after `__init__` and `model_construct`. |

### Functions

| Function | Signature | Description |
| --- | --- | --- |
| `activate_node` | `(node_id: int)` |  |
| `add_edge` | `(req: api.ConnectRequest)` |  |
| `add_node` | `(req: api.AddNodeRequest)` |  |
| `apply_suggestion` | `(from_id: int, to_id: int, strength: float = 0.45)` | Accept a predicted missing link. |
| `auto_heal` | `(max_links: int = 8, min_score: float = 0.5)` | Automatically apply predicted links to improve graph connectivity. |
| `batch_import` | `(req: api.BatchImportRequest)` | Add multiple nodes at once, optionally auto-evaluating each. |
| `decay` | `()` |  |
| `delete_node` | `(node_id: int)` |  |
| `eval_to_dict` | `(result) -> dict` |  |
| `evaluate_node` | `(node_id: int)` |  |
| `export_dot` | `()` | Export graph as DOT format (Graphviz compatible). |
| `export_graph` | `()` | Export full graph as JSON (nodes, edges, metadata, topology summary). |
| `export_graphml` | `()` | Export graph as GraphML (Gephi, yEd compatible). |
| `get_advice` | `()` | Actionable graph health recommendations. |
| `get_analytics` | `()` |  |
| `get_bridges` | `()` |  |
| `get_community` | `(community_id: int)` | Get all nodes and internal edges for a Louvain community. |
| `get_concept_path` | `(from_id: int, to_id: int)` | Shortest semantic path between two nodes. |
| `get_duplicates` | `(threshold: float = 0.88)` | Find semantically near-identical node pairs. |
| `get_evolution` | `()` |  |
| `get_graph` | `()` |  |
| `get_health` | `()` |  |
| `get_history` | `()` |  |
| `get_node` | `(node_id: int)` |  |
| `get_patterns` | `()` |  |
| `get_recommendations` | `(k: int = 5)` |  |
| `get_topology` | `()` |  |
| `graph_diff` | `(snap_a: int, snap_b: int)` | Compare two evolution snapshots by index. |
| `list_communities` | `()` | List all communities with member counts and anchor nodes. |
| `merge_nodes` | `(keep_id: int, remove_id: int)` | Merge two nodes: transfer connections from remove_id to keep_id. |
| `named_snapshot` | `(name: str)` | Save a named checkpoint of the current graph state. |
| `node_to_dict` | `(n: thought_graph.ThoughtNode) -> dict` |  |
| `promote_node` | `(node_id: int)` |  |
| `reset_graph` | `()` |  |
| `root` | `()` |  |
| `search_nodes` | `(q: str = '', node_type: str = None, min_importance: float = 0.0, community_id: int = None, limit: int = 20)` | Search nodes by text query and/or filters. |
| `serve_ui` | `()` |  |
| `similar_nodes` | `(node_id: int, k: int = 5)` |  |
| `suggest_links` | `(k: int = 5)` |  |
| `take_snapshot` | `()` |  |
| `update_node` | `(node_id: int, req: api.UpdateNodeRequest)` |  |
