# SPEC-12: Architecture Refactoring

## Overview

This specification defines architectural improvements for maintainability: orchestrator modularization, REPL plugin system, and memory backend abstraction.

## Dependencies

```
SPEC-12 (Architecture) ───► All prior specs (provides foundation)
```

---

## 12.1 Modularize Orchestrator

### Requirements

[SPEC-12.01] The orchestrator SHALL be split into focused modules:
```
orchestrator/
├── __init__.py          # Public exports
├── core.py              # Base orchestration loop
├── intelligent.py       # Claude-powered decisions
├── async_executor.py    # Async execution engine
├── checkpointing.py     # Session persistence
└── steering.py          # User interaction
```

[SPEC-12.02] core.py SHALL contain:
- Base RLMOrchestrator class
- Turn processing loop
- Event emission

[SPEC-12.03] intelligent.py SHALL contain:
- Claude-powered decision making
- Complexity assessment integration
- Strategy selection

[SPEC-12.04] async_executor.py SHALL contain:
- AsyncRLMOrchestrator
- Parallel execution
- Speculative execution

[SPEC-12.05] checkpointing.py SHALL contain:
- RLMCheckpoint dataclass
- Serialization/deserialization
- CheckpointingOrchestrator

[SPEC-12.06] steering.py SHALL contain:
- SteeringPoint
- InteractiveOrchestrator
- Auto-steering policy

[SPEC-12.07] All modules SHALL maintain backward compatibility via __init__.py exports.

### Acceptance Criteria

- [ ] Modules are cleanly separated
- [ ] No circular dependencies between modules
- [ ] Backward compatibility maintained
- [ ] Each module independently testable
- [ ] Import paths unchanged for external consumers

---

## 12.2 REPL Plugin Architecture

### Requirements

[SPEC-12.10] The system SHALL support REPL function plugins.

[SPEC-12.11] REPLPlugin Protocol SHALL define:
- name: str (property)
- functions: dict[str, Callable] (property)
- on_load(env: RLMEnvironment) -> None

[SPEC-12.12] The system SHALL support plugin registration:
- register_plugin(plugin: REPLPlugin) -> None
- unregister_plugin(name: str) -> None
- list_plugins() -> list[str]

[SPEC-12.13] Plugin functions SHALL be sandboxed like built-in functions.

[SPEC-12.14] The system SHALL provide built-in plugins:
- core: Basic REPL functions (peek, search, summarize)
- code_analysis: AST parsing, call graphs, dependencies
- computation: Safe math, statistics, data manipulation

[SPEC-12.15] Plugins SHALL support lazy loading for performance.

[SPEC-12.16] Plugin conflicts (duplicate function names) SHALL raise clear errors.

### Acceptance Criteria

- [ ] Plugin protocol works correctly
- [ ] Registration/unregistration works
- [ ] Plugin functions sandboxed
- [ ] Built-in plugins provide expected functionality
- [ ] Lazy loading reduces startup time
- [ ] Conflicts detected and reported

---

## 12.3 Memory Backend Abstraction

### Requirements

[SPEC-12.20] The system SHALL abstract memory storage behind MemoryBackend protocol.

[SPEC-12.21] MemoryBackend Protocol SHALL define:
- create_node(...) -> str
- get_node(node_id: str) -> Node | None
- update_node(node_id: str, ...) -> None
- delete_node(node_id: str) -> bool
- search(query: str, ...) -> list[SearchResult]
- create_edge(...) -> str
- get_edges(node_id: str, ...) -> list[Edge]

[SPEC-12.22] The system SHALL provide implementations:
- SQLiteBackend: Current implementation (default)
- InMemoryBackend: For testing
- PostgresBackend: For team/cloud scenarios (future)

[SPEC-12.23] Backend selection SHALL be configurable via config.

[SPEC-12.24] All backends SHALL pass the same test suite.

[SPEC-12.25] Migration tooling SHALL support backend-to-backend migration.

### Acceptance Criteria

- [ ] Protocol defines complete interface
- [ ] SQLiteBackend passes all tests
- [ ] InMemoryBackend passes all tests
- [ ] Backend configurable via config
- [ ] Existing code works with abstraction
