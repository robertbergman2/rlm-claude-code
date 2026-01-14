# SPEC-14: Always-On Micro-RLM

## Overview

This specification defines capabilities for making RLM the default execution mode for all Claude Code sessions, with intelligent scaling from near-zero-cost "micro" operations to full-depth analysis based on detected complexity.

**Research Basis**:
- Current `auto_activation.py` complexity classifier
- Current `intelligent_orchestrator.py` execution modes
- User feedback: $0.01+ per query is acceptable floor

## Dependencies

```
SPEC-14 (Always-On) ───► SPEC-05 (Budget Tracking) [cost management]
SPEC-14 (Always-On) ───► SPEC-06 (Smarter RLM) [orchestration]
SPEC-14 (Always-On) ───► SPEC-13 (Rich Output) [visibility]
```

---

## 14.1 Micro-RLM Execution Mode

### Requirements

[SPEC-14.01] The system SHALL support a "micro" execution mode as the baseline.

[SPEC-14.02] Micro mode SHALL have parameters:
```python
MICRO = ExecutionMode(
    depth_budget=1,
    max_tokens_per_depth=5000,
    model_tier=ModelTier.INHERIT,  # Use parent session model
    tool_access=ToolAccessLevel.REPL_ONLY,
    max_cost_tokens=2000,  # ~$0.01 at current rates
)
```

[SPEC-14.03] Micro mode REPL SHALL only expose low-cost functions:
- `peek(var, start, end)` - View context slice
- `search(var, pattern)` - Pattern search (no LLM)
- `summarize_local(var, max_chars)` - Truncation-based summary (no LLM)
- `memory_query(query)` - Memory retrieval

[SPEC-14.04] Micro mode SHALL NOT expose by default:
- `llm()` - Recursive sub-queries
- `llm_batch()` - Parallel sub-queries
- `map_reduce()` - LLM-based aggregation

[SPEC-14.05] Micro mode execution SHALL complete in <500ms for simple queries.

### Acceptance Criteria

- [ ] Micro mode costs <$0.02 per invocation
- [ ] REPL functions work without LLM calls
- [ ] Latency target met for simple queries
- [ ] Memory system remains active

---

## 14.2 Always-On Default Activation

### Requirements

[SPEC-14.10] The system SHALL activate RLM by default for all queries.

[SPEC-14.11] Default activation mode SHALL be "always" instead of "complexity".

[SPEC-14.12] ActivationConfig defaults SHALL change:
```python
@dataclass
class ActivationConfig:
    mode: Literal["micro", "complexity", "always", "manual", "token"] = "micro"
    # "micro" = always on, but starts at micro level
    # "complexity" = original heuristic-based
    # "always" = always full RLM
    # "manual" = only when explicitly enabled
    # "token" = activate above threshold
```

[SPEC-14.13] The system SHALL provide `/simple` command for true bypass.

[SPEC-14.14] The system SHALL provide `/rlm off` for session-wide disable.

[SPEC-14.15] The system SHALL log activation decisions for analysis.

### Acceptance Criteria

- [ ] RLM activates by default on fresh sessions
- [ ] `/simple` bypasses completely
- [ ] `/rlm off` persists for session
- [ ] Activation decisions logged

---

## 14.3 Progressive Escalation

### Requirements

[SPEC-14.20] The system SHALL escalate from micro to higher modes based on signals.

[SPEC-14.21] Escalation triggers SHALL include:
| Signal | Escalation | Target Mode |
|--------|------------|-------------|
| Multi-file reference | Immediate | Balanced |
| Discovery keywords | Immediate | Balanced |
| Debugging task | Immediate | Balanced |
| Architecture decision | Immediate | Thorough |
| Exhaustive search | After micro completes | Balanced |
| Low confidence result | After micro completes | Balanced |
| User says "thorough" | Immediate | Thorough |

[SPEC-14.22] Escalation SHALL be logged with reason.

[SPEC-14.23] The system SHALL NOT escalate if:
- User has set explicit mode via `/rlm mode <x>`
- Budget would be exceeded
- Query matches "definitely simple" patterns

[SPEC-14.24] Mid-execution escalation SHALL preserve context from micro phase.

[SPEC-14.25] Escalation decision confidence SHALL be tracked per SPEC-12.07.

### Acceptance Criteria

- [ ] Signals correctly trigger escalation
- [ ] Context preserved across escalation
- [ ] User overrides respected
- [ ] Budget limits enforced

---

## 14.4 Definitely Simple Fast Path

### Requirements

[SPEC-14.30] The system SHALL maintain fast-path bypass for trivially simple queries.

[SPEC-14.31] Fast-path patterns SHALL include:
```python
FAST_PATH_PATTERNS = [
    r"^(show|cat|read)\s+\S+$",           # File read
    r"^git\s+(status|log|diff)",           # Git commands
    r"^(yes|no|ok|thanks|thank you)$",     # Conversational
    r"^what('s| is) in .+\?$",             # Simple file query
    r"^(list|ls)\s+",                       # Directory listing
]
```

[SPEC-14.32] Fast-path SHALL skip RLM entirely, not just use micro mode.

[SPEC-14.33] Fast-path decision SHALL be logged for analysis.

[SPEC-14.34] Fast-path confidence SHALL be 0.95+ to activate.

### Acceptance Criteria

- [ ] Fast-path patterns identified correctly
- [ ] Zero RLM overhead for fast-path queries
- [ ] Logging captures fast-path decisions
- [ ] False positive rate <5%

---

## 14.5 Context Externalization Benefits

### Requirements

[SPEC-14.40] Even micro mode SHALL externalize context to REPL variables.

[SPEC-14.41] Externalized context SHALL include:
- `query` - Current user query
- `context` - Available context (files, conversation)
- `memory` - Relevant memory facts
- `prior_result` - Previous turn result (if any)

[SPEC-14.42] Externalization SHALL enable:
- `peek(context, 0, 1000)` to view context start
- `search(context, "pattern")` to find relevant sections
- `memory_query("related topic")` to retrieve facts

[SPEC-14.43] Externalization overhead SHALL be <100ms.

[SPEC-14.44] The system SHALL lazy-load large context only when accessed.

### Acceptance Criteria

- [ ] Variables available in REPL
- [ ] Lazy loading works correctly
- [ ] Overhead target met
- [ ] Memory integration functional

---

## 14.6 Memory System Integration

### Requirements

[SPEC-14.50] Always-on mode SHALL keep memory system engaged.

[SPEC-14.51] Every query SHALL trigger memory retrieval for relevant facts.

[SPEC-14.52] Micro mode SHALL support `memory_add_fact()` for storing insights.

[SPEC-14.53] Memory operations in micro mode SHALL NOT use LLM for embedding.

[SPEC-14.54] Memory retrieval SHALL use keyword matching in micro mode.

[SPEC-14.55] Full embedding-based retrieval SHALL activate on escalation.

### Acceptance Criteria

- [ ] Memory retrieval runs on every query
- [ ] Facts can be stored in micro mode
- [ ] Keyword matching works without LLM
- [ ] Escalation enables full embeddings

---

## 14.7 Cost Management

### Requirements

[SPEC-14.60] The system SHALL track cumulative session cost.

[SPEC-14.61] Cost tracking SHALL be in tokens, not dollars.

[SPEC-14.62] Default session budget SHALL be 500K tokens.

[SPEC-14.63] Budget warnings SHALL appear at 50%, 75%, 90% utilization.

[SPEC-14.64] The system SHALL prevent escalation if it would exceed budget.

[SPEC-14.65] Cost per execution mode SHALL be tracked separately:
- Micro: target <2K tokens
- Balanced: target <25K tokens
- Thorough: target <100K tokens

### Acceptance Criteria

- [ ] Token tracking accurate
- [ ] Budget warnings appear correctly
- [ ] Escalation blocked at budget limit
- [ ] Per-mode tracking works

---

## Implementation Notes

### Files to Modify

- `src/config.py` - Add "micro" mode, change defaults
- `src/auto_activation.py` - Add micro mode logic
- `src/intelligent_orchestrator.py` - Add escalation logic
- `src/orchestration_schema.py` - Add MICRO ExecutionMode
- `src/repl_environment.py` - Add restricted function set
- `src/context_manager.py` - Add lazy loading
- `tests/unit/test_always_on.py` - New test file

### Migration Path

1. Add micro mode as option (backward compatible)
2. Test micro mode in parallel with complexity mode
3. Change default to micro mode
4. Monitor activation logs for issues
5. Tune fast-path patterns based on data

### Configuration

```json
{
  "activation": {
    "mode": "micro",
    "fast_path_enabled": true,
    "escalation_enabled": true,
    "session_budget_tokens": 500000
  }
}
```
