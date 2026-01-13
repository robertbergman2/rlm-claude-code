# SPEC-11: User Experience Improvements

## Overview

This specification defines capabilities for improving RLM user experience through progressive disclosure, interactive steering, and learning from user feedback.

**Research Basis**:
- [Anthropic: Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [ToTRL (2025)](https://arxiv.org/abs/2505.12717) - learning from feedback

## Dependencies

```
SPEC-11 (User Experience) ───► SPEC-04 (Reasoning Traces)
SPEC-11 (User Experience) ───► SPEC-06 (Smarter RLM) [for learning integration]
```

---

## 11.1 Progressive Disclosure in Trajectory

### Requirements

[SPEC-11.01] The system SHALL support progressive trajectory rendering.

[SPEC-11.02] ProgressiveTrajectory SHALL support:
- render_summary() -> str: One-line progress summary
- render_overview() -> str: Key events without details
- render_detail(event_id) -> str: Full details for specific event
- render_cost_breakdown() -> str: Detailed cost attribution

[SPEC-11.03] Summary format SHALL be: "RLM: {N} recursive calls, {key_finding}"

[SPEC-11.04] Overview SHALL show only: RECURSE boundaries, FINAL, ERROR events.

[SPEC-11.05] The system SHALL support expandable rendering in compatible terminals.

[SPEC-11.06] Cost breakdown SHALL attribute costs to:
- Model tier
- Operation type (recursive, tool, synthesis)
- Component (orchestrator, REPL, memory)

### Acceptance Criteria

- [ ] Summary provides useful at-a-glance status
- [ ] Overview filters to key events
- [ ] Detail provides full event information
- [ ] Cost breakdown is accurate
- [ ] Rendering performance <50ms

---

## 11.2 Interactive Steering

### Requirements

[SPEC-11.10] The system SHALL support user steering during execution.

[SPEC-11.11] SteeringPoint SHALL support types:
- "branch": Choose between exploration paths
- "depth": Adjust remaining depth budget
- "abort": Cancel and return current results
- "refine": Provide additional guidance

[SPEC-11.12] SteeringPoint SHALL include:
- options: list[str]
- default: str
- timeout: float

[SPEC-11.13] The system SHALL present steering opportunities at:
- Before recursive decomposition
- After low-confidence intermediate results
- When multiple viable paths exist

[SPEC-11.14] The system SHALL support auto-steering policy for testing/CI.

[SPEC-11.15] Steering responses SHALL be logged for analysis.

[SPEC-11.16] Timeout on steering request SHALL use default option.

### Acceptance Criteria

- [ ] Steering points presented at appropriate times
- [ ] User choices affect execution correctly
- [ ] Auto-steering policy works for automation
- [ ] Timeout handling uses default
- [ ] Steering logging captures decisions

---

## 11.3 Learning from User Corrections

### Requirements

[SPEC-11.20] The system SHALL capture user corrections to RLM outputs.

[SPEC-11.21] Correction types SHALL include:
- FACTUAL: Incorrect fact in output
- INCOMPLETE: Missing important information
- WRONG_APPROACH: Should have used different strategy
- OVER_COMPLEX: RLM unnecessary for this query
- UNDER_COMPLEX: Needed RLM but didn't activate

[SPEC-11.22] The system SHALL record corrections with:
- query
- rlm_output
- user_correction
- correction_type

[SPEC-11.23] The system SHALL analyze corrections to suggest classifier adjustments:
- If users frequently override OVER_COMPLEX → raise activation threshold
- If users frequently override UNDER_COMPLEX → lower activation threshold

[SPEC-11.24] ClassifierAdjustments SHALL include:
- signal_adjustments: dict[signal_name, float]
- threshold_adjustment: float
- reasoning: str

[SPEC-11.25] Adjustments SHALL be logged and require confirmation before applying.

### Acceptance Criteria

- [ ] Corrections captured correctly
- [ ] Correction types cover common cases
- [ ] Analysis generates useful adjustments
- [ ] Adjustments require confirmation
- [ ] Applied adjustments improve classifier accuracy
