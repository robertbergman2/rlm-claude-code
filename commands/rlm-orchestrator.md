# RLM Orchestrator

Invoke the RLM orchestrator agent for complex context management tasks.

## Usage

`/rlm-orchestrator [task description]`

## When to Use

Use this when you need:
- Multi-file context management across large codebases
- Parallel tracking of multiple work streams
- Complex reasoning with context externalization
- REPL-based context decomposition

## How It Works

This command launches the RLM orchestrator as a Task agent with full tool access. The agent:

1. Externalizes conversation context to Python variables
2. Uses REPL operations (peek, search, summarize) for efficient context access
3. Can spawn recursive sub-queries for deep analysis
4. Manages depth budgets and model cascades (Opus → Sonnet → Haiku)

## Instructions

When this skill is invoked, use the Task tool to launch the RLM orchestrator agent:

```
Task(
  subagent_type="rlm-claude-code:rlm-orchestrator",
  prompt="[user's task description]",
  description="RLM orchestration"
)
```

The agent will receive the full conversation context and can use all available tools.

## Related

- `/rlm` - Configure RLM mode settings
- `/rlm status` - Check current configuration
- `/simple` - Bypass RLM for simple operations
