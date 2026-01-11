# RLM-Claude-Code Process Documentation

This directory contains process documentation for Claude Code sessions working on the RLM integration. These documents ensure consistent, high-quality implementation across sessions.

## Document Index

| Document | Purpose |
|----------|---------|
| [code-review.md](./code-review.md) | Code review checklist and standards |
| [implementation.md](./implementation.md) | Implementation workflow and patterns |
| [testing.md](./testing.md) | Testing strategy and requirements |
| [architecture.md](./architecture.md) | Architecture decision records |
| [debugging.md](./debugging.md) | Debugging workflow for RLM issues |

## Quick Reference

### Before Starting Work

1. Read the relevant spec section in `rlm-claude-code-spec.md`
2. Check `docs/process/implementation.md` for current phase
3. Review any open issues tagged for current phase
4. Run `uv sync` to ensure dependencies are current

### During Implementation

```bash
# Type check continuously
ty check src/

# Lint before committing
ruff check src/ --fix
ruff format src/

# Run tests
pytest tests/ -v

# Validate with hypothesis (if applicable)
pytest tests/ -v -m hypothesis
```

### Before Committing

1. Run `/code-review` command
2. Ensure all tests pass
3. Update trajectory examples if behavior changed
4. Document any new patterns in architecture.md

## Core Principles

1. **Spec-Driven**: The spec is the source of truth. Implementation diverging from spec requires spec update first.

2. **Incremental Verification**: Each component should be testable in isolation before integration.

3. **Trajectory-First**: When debugging, always capture and analyze the trajectory before making changes.

4. **Type Everything**: All public APIs must have full type annotations validated by `ty`.
