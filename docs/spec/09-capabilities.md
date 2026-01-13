# SPEC-09: Capability Enhancements

## Overview

This specification defines new capabilities for embedding-based memory retrieval, cross-session memory promotion, and multi-turn checkpointing.

**Research Basis**:
- [A-MEM (NeurIPS 2025)](https://arxiv.org/abs/2502.12110)
- [Zep: Temporal Knowledge Graph](https://blog.getzep.com/content/files/2025/01/ZEP__USING_KNOWLEDGE_GRAPHS_TO_POWER_LLM_AGENT_MEMORY_2025011700.pdf)
- [G-Memory (2025)](https://arxiv.org/html/2511.07800v1)
- [Anthropic: Long-Running Agent Harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

## Dependencies

```
SPEC-09 (Capabilities) ───► SPEC-02 (Memory Foundation)
SPEC-09 (Capabilities) ───► SPEC-03 (Memory Evolution)
```

---

## 9.1 Embedding-Based Memory Retrieval

### Requirements

[SPEC-09.01] The system SHALL support embedding computation for memory nodes.

[SPEC-09.02] The system SHALL use text-embedding-3-small (1536 dimensions) by default.

[SPEC-09.03] The system SHALL store embeddings in vec0 virtual table:
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS node_embeddings USING vec0(
    node_id TEXT PRIMARY KEY,
    embedding FLOAT[1536]
);
```

[SPEC-09.04] The system SHALL support hybrid retrieval:
- Keyword search via existing FTS5
- Semantic search via embedding similarity
- Combined score: hybrid_alpha * semantic + (1 - hybrid_alpha) * keyword

[SPEC-09.05] The system SHALL support configurable hybrid_alpha (default: 0.5).

[SPEC-09.06] Embedding computation SHALL be optional per-node (compute_embedding flag).

[SPEC-09.07] The system SHALL batch embedding requests for efficiency.

### Acceptance Criteria

- [ ] Embeddings computed and stored correctly
- [ ] Hybrid retrieval returns relevant results
- [ ] Semantic search improves recall for conceptual queries
- [ ] Embedding computation <100ms per batch of 10
- [ ] Storage overhead acceptable (<10% database size increase)

---

## 9.2 Cross-Session Memory Promotion

### Requirements

[SPEC-09.10] The system SHALL track memory access across sessions.

[SPEC-09.11] The system SHALL identify promotion candidates based on:
- Accessed in 3+ distinct sessions
- Associated with successful outcomes
- High confidence maintained over time

[SPEC-09.12] The system SHALL automatically promote memories that meet criteria:
- task → session: after 2+ uses in session
- session → longterm: after 3+ cross-session accesses
- longterm → archive: based on staleness (>90 days unused)

[SPEC-09.13] Promotion SHALL preserve original memory content and metadata.

[SPEC-09.14] The system SHALL log promotion decisions for auditability.

[SPEC-09.15] The system SHALL support manual promotion override.

### Acceptance Criteria

- [ ] Access tracking works across sessions
- [ ] Promotion criteria evaluated correctly
- [ ] Automatic promotion triggers appropriately
- [ ] Promoted memories retain all metadata
- [ ] Promotion logging provides auditability

---

## 9.3 Multi-Turn Checkpointing

### Requirements

[SPEC-09.20] The system SHALL support checkpointing RLM state for recovery.

[SPEC-09.21] RLMCheckpoint SHALL include:
- session_id
- depth
- repl_state (all REPL variables)
- working_memory
- pending_operations
- trajectory_events
- cost_so_far

[SPEC-09.22] The system SHALL serialize checkpoints to disk.

[SPEC-09.23] The system SHALL support checkpoint restoration:
- Load checkpoint from path
- Restore REPL state
- Resume pending operations

[SPEC-09.24] The system SHALL support automatic checkpointing every N turns (default: 5).

[SPEC-09.25] The system SHALL support manual checkpoint triggers.

[SPEC-09.26] Checkpoints SHALL be versioned for compatibility checking.

### Acceptance Criteria

- [ ] Checkpoints capture all required state
- [ ] Serialization/deserialization works correctly
- [ ] Restored sessions continue correctly
- [ ] Automatic checkpointing triggers at interval
- [ ] Checkpoint size reasonable (<10MB typical)
