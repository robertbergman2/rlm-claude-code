# SPEC-13: Rich Output Formatting

## Overview

This specification defines capabilities for enhanced terminal output using the Rich library, providing clear visual feedback during RLM execution without excessive noise.

**Research Basis**:
- [Rich Library](https://github.com/Textualize/rich) - Python terminal formatting
- User feedback: Prefer symbols over emoji, token counts over dollar values

## Dependencies

```
SPEC-13 (Rich Output) ───► SPEC-04 (Reasoning Traces) [event stream]
SPEC-13 (Rich Output) ───► SPEC-05 (Budget Tracking) [token counts]
```

---

## 13.1 Visual Language System

### Requirements

[SPEC-13.01] The system SHALL use a consistent symbol vocabulary for semantic meaning.

[SPEC-13.02] Symbol vocabulary SHALL include:
| Symbol | Meaning | ANSI Color |
|--------|---------|------------|
| `◆` | RLM activated | Cyan |
| `▶` | Execution/action | Yellow |
| `◇` | Read/peek operation | Blue |
| `⊕` | Search operation | Blue |
| `∴` | LLM sub-query (therefore) | Magenta |
| `✓` | Success/complete | Green |
| `✗` | Error/failure | Red |
| `⚠` | Warning/caution | Yellow |
| `≡` | Cost/budget report | White |
| `∿` | Memory operation | Cyan |
| `⊢` | Lean verification (turnstile) | Teal |
| `│` | Depth continuation | Dim |
| `├` | Branch point | Dim |
| `└` | Terminal node | Dim |

[SPEC-13.03] The system SHALL NOT use emoji characters in output.

[SPEC-13.04] Colors SHALL degrade gracefully in terminals without color support.

[SPEC-13.05] The system SHALL respect NO_COLOR environment variable.

### Acceptance Criteria

- [ ] All symbols render correctly in common terminals
- [ ] Color output can be disabled
- [ ] Semantic meaning is consistent across all output

---

## 13.2 Rich Console Integration

### Requirements

[SPEC-13.10] The system SHALL use Rich Console for all trajectory output.

[SPEC-13.11] RLMConsole class SHALL provide:
- `emit_start(query, depth_budget)` - RLM activation
- `emit_repl(func, args, preview)` - REPL operation
- `emit_recurse(query, depth)` - Recursive sub-query
- `emit_result(summary, confidence)` - Operation result
- `emit_error(error, context)` - Error with context
- `emit_budget(tokens_used, tokens_budget)` - Budget status

[SPEC-13.12] Console output SHALL be configurable via verbosity levels:
- `quiet`: Only errors and final result
- `normal`: Key events with single-line summaries
- `verbose`: All events with details
- `debug`: Full content including internal state

[SPEC-13.13] The system SHALL support Panel rendering for structured output.

[SPEC-13.14] The system SHALL support Tree rendering for recursive call visualization.

### Acceptance Criteria

- [ ] Rich Console initializes without errors
- [ ] All event types render correctly
- [ ] Verbosity levels filter appropriately
- [ ] Panel and Tree components work

---

## 13.3 Progress and Budget Display

### Requirements

[SPEC-13.20] The system SHALL display token budget as a visual gauge.

[SPEC-13.21] Budget gauge format SHALL be:
```
≡ Tokens: ████████░░ 45K/100K (depth 1/3)
```

[SPEC-13.22] The system SHALL NOT display dollar costs in output.

[SPEC-13.23] The system SHALL show progress for long-running operations via spinner.

[SPEC-13.24] Progress spinner SHALL use braille pattern: `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`

[SPEC-13.25] The system SHALL throttle progress updates to max 10Hz to reduce noise.

### Acceptance Criteria

- [ ] Token gauge displays accurately
- [ ] No dollar amounts in output
- [ ] Spinner animates smoothly
- [ ] Updates don't flood the terminal

---

## 13.4 Depth Visualization

### Requirements

[SPEC-13.30] The system SHALL visualize recursive depth using tree connectors.

[SPEC-13.31] Depth visualization format SHALL be:
```
◆ RLM: "analyze authentication flow"
├─ ◇ peek(context, 0:2000)
├─ ⊕ search(code, "def.*auth")
│  └─ 3 matches
├─ ∴ llm("summarize auth patterns")
│  ├─ ◇ peek(result, 0:500)
│  └─ ✓ "Uses JWT with refresh tokens"
└─ ✓ Complete (1.2K tokens)
```

[SPEC-13.32] Maximum rendered depth SHALL be configurable (default: 5).

[SPEC-13.33] Deep recursion beyond max SHALL show `... (+N deeper)`.

### Acceptance Criteria

- [ ] Tree structure renders correctly
- [ ] Depth limits prevent excessive output
- [ ] Connector characters align properly

---

## 13.5 Error Display

### Requirements

[SPEC-13.40] The system SHALL render errors with contextual information.

[SPEC-13.41] Error format SHALL include:
- Error type and message
- Location (file:line if available)
- Relevant context snippet
- Suggested action (if determinable)

[SPEC-13.42] Error panels SHALL use red border with `✗` prefix.

[SPEC-13.43] Warnings SHALL use yellow border with `⚠` prefix.

[SPEC-13.44] The system SHALL syntax-highlight code in error context.

### Acceptance Criteria

- [ ] Errors are clearly visible
- [ ] Context helps diagnose issues
- [ ] Syntax highlighting works for Python

---

## 13.6 Configuration

### Requirements

[SPEC-13.50] Rich output SHALL be configurable via RLM config.

[SPEC-13.51] Configuration options SHALL include:
```python
@dataclass
class OutputConfig:
    verbosity: Literal["quiet", "normal", "verbose", "debug"] = "normal"
    colors: bool = True  # Respect NO_COLOR if False
    max_depth_display: int = 5
    progress_throttle_hz: int = 10
    panel_width: int | None = None  # Auto-detect if None
```

[SPEC-13.52] Configuration SHALL be overridable via:
- Environment variables: `RLM_VERBOSITY`, `RLM_COLORS`
- Command: `/rlm verbose`, `/rlm quiet`
- Config file: `~/.claude/rlm-config.json`

### Acceptance Criteria

- [ ] All configuration options work
- [ ] Environment variables override config file
- [ ] Commands override all

---

## Implementation Notes

### Dependencies

Add to `pyproject.toml`:
```toml
[project.dependencies]
rich = ">=13.0.0"
```

### Files to Create/Modify

- `src/rich_output.py` - New: RLMConsole class
- `src/trajectory.py` - Modify: Use RLMConsole for rendering
- `src/config.py` - Modify: Add OutputConfig
- `tests/unit/test_rich_output.py` - New: Unit tests

### Integration Points

1. `TrajectoryStream.emit_*()` methods call `RLMConsole` methods
2. `progress.py` callbacks use Rich spinner/progress
3. Hook scripts use Rich for consistent formatting
