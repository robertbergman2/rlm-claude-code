# SPEC-06: Smarter RLM - Tool Orchestration, Routing, and Learning

## Overview

This specification defines capabilities for making the RLM genuinely smarter over time through proactive computation, intelligent tool orchestration, model routing, context enrichment, and continuous learning.

**Research Basis**:
- [ARTIST (Microsoft, 2025)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/AgenticReasoning.pdf)
- [LATS (ICML 2024)](https://arxiv.org/abs/2310.04406)
- [RouteLLM (ICLR 2025)](https://github.com/lm-sys/RouteLLM)
- [Agent Lightning (Microsoft, 2025)](https://arxiv.org/abs/2508.03680)

## Dependencies

```
SPEC-06 (Smarter RLM) ───► SPEC-01 (REPL Functions)
SPEC-06 (Smarter RLM) ───► SPEC-02 (Memory Foundation)
SPEC-06 (Smarter RLM) ───► SPEC-05 (Budget Tracking)
```

---

## 6.1 Proactive REPL for Programmatic Reasoning

### Requirements

[SPEC-06.01] The system SHALL detect query patterns where REPL computation is more reliable than LLM reasoning.

[SPEC-06.02] The system SHALL recognize at minimum these computation triggers:
- Arithmetic operations (calculate, compute, +/-/*/÷)
- Counting operations (how many, count, number of, total)
- Sorting operations (sort, order, rank, largest, smallest, top N)
- Filtering operations (filter, where, matching, containing)
- Aggregation operations (sum, average, mean, median, max, min)
- Date math operations (days since, weeks ago, before, after)
- String operations (extract, parse, split, format, regex)

[SPEC-06.03] The system SHALL generate REPL code suggestions with explanation when computation is detected.

[SPEC-06.04] The REPL SHALL provide enhanced computational helpers:
- `calc(expression)` - Safe mathematical expression evaluation
- `stats(data)` - Statistical computation (mean, median, std, etc.)
- `group_by(data, key)` - Data grouping
- `compile_context(files, include_deps, max_tokens)` - Context compilation
- `find_imports(content)` - Import/dependency analysis
- `call_graph(file)` - Call graph construction

[SPEC-06.05] All REPL computation helpers SHALL be sandboxed with the existing RestrictedPython security model.

### Acceptance Criteria

- [ ] ProactiveComputationAdvisor detects 90%+ of computation-suitable queries
- [ ] Code templates generate valid, executable REPL code
- [ ] All helpers pass security tests (no escape from sandbox)
- [ ] Computation helpers have <50ms overhead

---

## 6.2 Intelligent Tool Orchestration

### Requirements

[SPEC-06.10] The system SHALL implement LATS-inspired tool orchestration with distinct phases:
- PLAN: Generate tool use plan before execution
- EXPAND: Use UCB1 to select promising action branches
- SIMULATE: Execute and evaluate results
- BACKPROPAGATE: Update value estimates
- REFLECT: Self-critique failed paths

[SPEC-06.11] The system SHALL maintain a tool capability matrix mapping tools to capabilities.

[SPEC-06.12] The system SHALL provide task-to-tool preference mapping for common task types.

[SPEC-06.13] The orchestrator SHALL support fallback tool sequences when primary tools fail.

[SPEC-06.14] The system SHALL compute UCB1 scores for node selection:
```
UCB1 = exploitation + exploration_weight * sqrt(2 * ln(parent_visits) / node_visits)
```

[SPEC-06.15] The orchestrator SHALL support configurable:
- exploration_weight (default: 1.414)
- max_rollouts (default: 10)
- max_depth (default: 5)

### Acceptance Criteria

- [ ] Tool plans are generated before execution for complex queries
- [ ] UCB1 selection balances exploration/exploitation correctly
- [ ] Reflection generates useful feedback on failed paths
- [ ] Tool orchestration doubles success rate vs reactive baseline (per LATS paper)

---

## 6.3 Intelligent Model Routing

### Requirements

[SPEC-06.20] The system SHALL implement learned routing based on query characteristics.

[SPEC-06.21] The router SHALL maintain model profiles including:
- strengths (list of task types)
- cost_per_1k tokens
- quality_baseline (0-1 scale)

[SPEC-06.22] The router SHALL estimate query difficulty based on:
- Reasoning depth required
- Domain specificity
- Ambiguity level
- Context size

[SPEC-06.23] The router SHALL support cost_sensitivity parameter (0=quality only, 1=cost only).

[SPEC-06.24] The system SHALL implement cascading routing:
- Start with cheapest viable model
- Escalate on low confidence
- Use self-consistency check for confidence estimation

[SPEC-06.25] The cascading router SHALL support configurable:
- confidence_threshold (default: 0.8)
- max_escalations (default: 2)
- cascade_order (default: ["haiku", "sonnet", "opus"])

[SPEC-06.26] The router SHALL record outcomes for learning:
- query embedding
- model used
- success/failure
- quality score
- cost

### Acceptance Criteria

- [ ] Routing achieves 85%+ cost reduction vs always-opus baseline
- [ ] Quality maintained at 95%+ of opus-only performance
- [ ] Cascading escalation works correctly
- [ ] Outcome recording enables offline analysis

---

## 6.4 Programmatic Context Enrichment

### Requirements

[SPEC-06.30] The system SHALL proactively enrich context before LLM reasoning.

[SPEC-06.31] The system SHALL classify query intent and select enrichment strategy:
- code_task: Add dependencies, types, tests, recent changes
- debug_task: Add error context, blame, similar experiences
- analysis_task: Add related documentation, examples
- question: Add relevant memories and facts

[SPEC-06.32] For code tasks, the enricher SHALL automatically gather:
- Import graph (local dependencies only)
- Type definitions (TypeScript/Python)
- Related test files
- Recent git changes to affected files

[SPEC-06.33] For debug tasks, the enricher SHALL automatically gather:
- Error stack trace parsed locations (±20 lines context)
- Git blame for error locations
- Similar past debugging experiences from memory

[SPEC-06.34] Enrichment SHALL respect token budgets and truncate to fit.

[SPEC-06.35] Enrichment reasoning SHALL be logged for transparency.

### Acceptance Criteria

- [ ] Context enrichment adds relevant information 80%+ of time
- [ ] Enrichment respects token budget
- [ ] Enrichment strategies cover all major task types
- [ ] Enrichment improves task success rate by 15%+

---

## 6.5 Continuous Learning and Self-Improvement

### Requirements

[SPEC-06.40] The system SHALL record execution outcomes including:
- query and features
- strategy and model used
- depth reached
- tools used
- success/failure
- user satisfaction (if provided)
- cost and latency

[SPEC-06.41] The system SHALL extract learning signals from outcomes:
- routing signals (model selection effectiveness)
- strategy signals (decomposition approach effectiveness)
- tool signals (tool selection effectiveness)

[SPEC-06.42] The system SHALL maintain learned adjustments:
- routing_adjustments: dict[query_type:model, float]
- strategy_preferences: dict[query_type, dict[strategy, float]]
- tool_effectiveness: dict[task_type, dict[tool, float]]

[SPEC-06.43] The system SHALL apply learning rate for incremental updates (default: 0.1).

[SPEC-06.44] The system SHALL persist learned state across sessions.

[SPEC-06.45] The system SHALL implement meta-learning:
- Track prediction accuracy
- Adjust learning rate based on prediction performance
- Increase rate when predictions are poor (<60% accuracy)
- Decrease rate when predictions are good (>80% accuracy)

### Acceptance Criteria

- [ ] Outcome recording captures all required fields
- [ ] Learning signals are extracted correctly
- [ ] Learned adjustments persist and load correctly
- [ ] Meta-learning adjusts learning rate appropriately
- [ ] Performance improves measurably over 100+ queries

---

## 6.6 Integration: Smart RLM Pipeline

### Requirements

[SPEC-06.50] The system SHALL integrate all smarter RLM components into a unified pipeline:
1. Analyze query → extract features, estimate difficulty
2. Check proactive computation → use REPL if high confidence
3. Enrich context → proactively gather information
4. Route → select optimal model
5. Plan → generate tool orchestration plan
6. Execute → run with cascading
7. Learn → record outcome and update preferences

[SPEC-06.51] Each pipeline stage SHALL be independently testable.

[SPEC-06.52] The pipeline SHALL support bypass for simple queries detected by is_definitely_simple().

[SPEC-06.53] The pipeline SHALL emit telemetry for each stage.

### Acceptance Criteria

- [ ] Pipeline integrates all components correctly
- [ ] Simple queries bypass heavy processing
- [ ] Telemetry provides visibility into pipeline stages
- [ ] End-to-end latency acceptable (<2s for simple, <30s for complex)
