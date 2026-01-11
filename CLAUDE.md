# RLM-Claude-Code

Transform Claude Code into a Recursive Language Model agent for unbounded context handling and improved reasoning.

## Quick Start

```bash
# Setup
uv sync
uv run ty check src/

# Test
uv run pytest tests/ -v

# Run RLM
uv run python -m src.orchestrator --query "your query"
```

## Project Structure

```
rlm-claude-code/
├── CLAUDE.md                    # You are here
├── rlm-claude-code-spec.md      # Specification (source of truth)
├── docs/
│   └── process/                 # Process documentation
│       ├── README.md            # Process index
│       ├── code-review.md       # Code review checklist
│       ├── implementation.md    # Implementation workflow
│       ├── testing.md           # Testing strategy
│       ├── architecture.md      # Architecture decisions
│       └── debugging.md         # Debugging workflow
├── src/
│   ├── __init__.py
│   ├── types.py                 # Shared types
│   ├── config.py                # Configuration
│   ├── context_manager.py       # Context externalization
│   ├── repl_environment.py      # Python REPL sandbox
│   ├── recursive_handler.py     # Recursive call management
│   ├── complexity_classifier.py # Activation logic
│   ├── trajectory.py            # Trajectory events/rendering
│   ├── router_integration.py    # Model routing
│   └── orchestrator.py          # Main RLM loop
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── property/
│   ├── security/
│   ├── benchmarks/
│   └── fixtures/
└── pyproject.toml
```

## Essential Context

**Read these files before making changes:**

1. `rlm-claude-code-spec.md` — Full specification
2. `docs/process/implementation.md` — Current phase and file order
3. `docs/process/code-review.md` — Review checklist

## Development Commands

```bash
# Type checking (must pass)
uv run ty check src/

# Linting (must pass)
uv run ruff check src/ --fix
uv run ruff format src/

# Tests
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=src/

# Property tests
uv run pytest tests/property/ -v -m hypothesis

# Benchmarks
uv run pytest tests/benchmarks/ --benchmark-only

# Interactive REPL testing
uv run python -m src.repl_environment --interactive
```

## Key Technologies

| Tool | Purpose | Docs |
|------|---------|------|
| uv | Package management | https://docs.astral.sh/uv/ |
| ty | Type checking | https://docs.astral.sh/ty/ |
| ruff | Linting/formatting | https://docs.astral.sh/ruff/ |
| pydantic | Data validation | https://docs.pydantic.dev/ |
| hypothesis | Property testing | https://hypothesis.readthedocs.io/ |
| cpmpy | Constraint programming | https://cpmpy.readthedocs.io/ |
| RestrictedPython | REPL sandbox | https://restrictedpython.readthedocs.io/ |

## Architecture Summary

```
User Query
    ↓
Complexity Classifier (§6.3)
    ↓ (if complex)
RLM Orchestrator
    ├── Context Manager → Externalize to Python vars
    ├── REPL Environment → Execute peek/search/summarize
    └── Recursive Handler → Spawn sub-queries (depth≤2)
    ↓
Trajectory Stream → User sees reasoning
    ↓
Claude Code Tools → bash/edit/read as normal
    ↓
Final Answer
```

## Implementation Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core Infrastructure | Not Started |
| 2 | Claude Code Integration | Not Started |
| 3 | Optimization | Not Started |
| 4 | Advanced Features | Not Started |

See `docs/process/implementation.md` for details.

## Spec Traceability

Every implementation must reference the spec:

```python
def should_activate_rlm(prompt: str, context: SessionContext) -> tuple[bool, str]:
    """
    Determine if RLM mode should activate.
    
    Implements: Spec §6.3 Task Complexity-Based Activation
    """
```

## Code Style

- Type annotations on all public functions (validated by `ty`)
- Google-style docstrings
- No functions >50 lines
- No `# type: ignore` without justification
- Pydantic models at API boundaries

## Testing Requirements

- Unit tests for all new functions
- Property tests for data transformations
- Security tests for REPL operations
- Trajectory snapshot tests for behavior changes
- Benchmarks for performance-critical paths

## Before Committing

1. `uv run ty check src/` — Must pass
2. `uv run ruff check src/` — Must pass
3. `uv run pytest tests/ -v` — Must pass
4. Run `/code-review` command
5. Update trajectory snapshots if behavior changed

## Debugging

When something breaks:

1. **Capture trajectory first**: `--verbosity debug --export-trajectory`
2. See `docs/process/debugging.md` for workflow
3. Never guess without trajectory data

## Common Patterns

### Adding a REPL Helper

```python
# In repl_environment.py
class RLMEnvironment:
    def __init__(self, context: SessionContext):
        self.globals = {
            # ... existing ...
            'new_helper': self._new_helper,
        }
    
    def _new_helper(self, arg: str) -> str:
        """
        New helper function.
        
        Implements: Spec §4.X
        """
        # Implementation
```

### Adding a Complexity Signal

```python
# In complexity_classifier.py
@dataclass
class TaskComplexitySignals:
    # ... existing ...
    new_signal: bool  # Add field
    
def extract_complexity_signals(...) -> TaskComplexitySignals:
    # Add pattern
    new_patterns = [r'pattern1', r'pattern2']
    
    return TaskComplexitySignals(
        # ... existing ...
        new_signal=any(re.search(p, prompt_lower) for p in new_patterns),
    )
```

### Adding a Trajectory Event Type

```python
# In trajectory.py
class TrajectoryEventType(Enum):
    # ... existing ...
    NEW_TYPE = "new_type"

class TrajectoryRenderer:
    ICONS = {
        # ... existing ...
        TrajectoryEventType.NEW_TYPE: "◇",
    }
    LABELS = {
        # ... existing ...
        TrajectoryEventType.NEW_TYPE: "NEWTYPE",
    }
```

## Slash Commands

Create in `.claude/commands/`:

- `/code-review` — Run code review checklist
- `/test` — Run test suite
- `/bench` — Run benchmarks
- `/trajectory [file]` — Analyze trajectory

## Contact

This project implements the RLM paper: https://arxiv.org/abs/2512.24601v1

Spec author: Rand (Head of AI, Heroku)
