# RLM-Claude-Code

Transform Claude Code into a Recursive Language Model (RLM) agent for unbounded context handling and improved reasoning via REPL-based context decomposition.

## Overview

RLM-Claude-Code implements the [Recursive Language Model architecture](https://arxiv.org/abs/2512.24601v1) as a Claude Code plugin. Instead of stuffing entire contexts into prompts, it:

1. **Externalizes context** as Python variables in a REPL environment
2. **Enables programmatic access** via peek, search, and recursive query operations  
3. **Supports recursive reasoning** with depth=2 for verification chains
4. **Streams trajectory** for human-readable insight into the reasoning process

## Quick Start

```bash
# Install dependencies
uv sync

# Run setup script
./scripts/setup_repl_env.sh

# Test
uv run pytest tests/ -v

# Run RLM
uv run python -m src.orchestrator --query "your query"
```

## Key Features

- **Complexity-based activation**: RLM activates when tasks need it, not just for large contexts
- **Depth=2 recursion**: Root → Analysis → Verification pattern for safe refactoring
- **Extended tooling**: pydantic, hypothesis, and CPMpy available in REPL
- **Streaming trajectory**: See reasoning unfold in real-time

## Documentation

| Document | Purpose |
|----------|---------|
| [rlm-claude-code-spec.md](./rlm-claude-code-spec.md) | Full specification |
| [CLAUDE.md](./CLAUDE.md) | Claude Code session guide |
| [docs/process/](./docs/process/) | Development process |

## Architecture

```
User Query
    ↓
Complexity Classifier
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

## Configuration

See `~/.claude/rlm-config.json`:

```json
{
  "activation": {"mode": "complexity"},
  "depth": {"default": 2, "max": 3},
  "trajectory": {"verbosity": "normal"},
  "models": {
    "root": "claude-opus-4-5-20251101",
    "recursive_depth_1": "claude-sonnet-4",
    "recursive_depth_2": "claude-haiku-4-5-20251001"
  }
}
```

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601v1) - Zhang, Kraska, Khattab
- [RLM Implementation](https://github.com/alexzhang13/rlm)
- [Claude Code](https://github.com/anthropics/claude-code)
- [CPMpy](https://cpmpy.readthedocs.io/)

## License

MIT
