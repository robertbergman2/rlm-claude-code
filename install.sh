#!/usr/bin/env bash
set -euo pipefail

# RLM Claude Code Installer
# Installs RLM directly into ~/.claude/ (non-plugin mode)

RLM_ROOT="${RLM_ROOT:-$HOME/.local/share/rlm-claude-code}"
CLAUDE_DIR="$HOME/.claude"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing RLM Claude Code..."
echo "  Install location: $RLM_ROOT"
echo "  Claude directory: $CLAUDE_DIR"
echo ""

# Create directories
mkdir -p "$RLM_ROOT"
mkdir -p "$CLAUDE_DIR/commands"
mkdir -p "$CLAUDE_DIR/skills"

# Copy core files to RLM_ROOT
echo "Copying core files..."
cp -r "$SCRIPT_DIR/src" "$RLM_ROOT/"
cp -r "$SCRIPT_DIR/scripts" "$RLM_ROOT/"
cp -r "$SCRIPT_DIR/tests" "$RLM_ROOT/" 2>/dev/null || true
cp "$SCRIPT_DIR/pyproject.toml" "$RLM_ROOT/"
cp "$SCRIPT_DIR/uv.lock" "$RLM_ROOT/" 2>/dev/null || true
cp "$SCRIPT_DIR/.env.example" "$RLM_ROOT/" 2>/dev/null || true

# Copy commands
echo "Installing commands..."
cp "$SCRIPT_DIR/.claude/commands/"*.md "$CLAUDE_DIR/commands/"

# Copy skills
echo "Installing skills..."
cp -r "$SCRIPT_DIR/.claude/skills/"* "$CLAUDE_DIR/skills/"

# Create virtual environment and install dependencies
echo "Setting up Python environment..."
cd "$RLM_ROOT"
if command -v uv &> /dev/null; then
    uv venv .venv
    uv pip install -e .
else
    python3 -m venv .venv
    .venv/bin/pip install -e .
fi

# Generate hooks configuration
echo "Generating hooks configuration..."
cat > "$RLM_ROOT/hooks.json" << 'HOOKS_EOF'
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$HOME/.local/share/rlm-claude-code\" && .venv/bin/python \"$HOME/.local/share/rlm-claude-code/scripts/init_rlm.py\"",
            "timeout": 5000,
            "description": "Initialize RLM environment and load config"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$HOME/.local/share/rlm-claude-code\" && .venv/bin/python \"$HOME/.local/share/rlm-claude-code/scripts/check_complexity.py\" \"$PROMPT\"",
            "timeout": 3000,
            "description": "Check if RLM should activate based on task complexity"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$HOME/.local/share/rlm-claude-code\" && .venv/bin/python \"$HOME/.local/share/rlm-claude-code/scripts/sync_context.py\"",
            "timeout": 2000,
            "description": "Sync tool context with RLM state"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$HOME/.local/share/rlm-claude-code\" && .venv/bin/python \"$HOME/.local/share/rlm-claude-code/scripts/capture_output.py\"",
            "timeout": 2000,
            "description": "Capture tool output for RLM context"
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$HOME/.local/share/rlm-claude-code\" && .venv/bin/python \"$HOME/.local/share/rlm-claude-code/scripts/externalize_context.py\"",
            "timeout": 10000,
            "description": "Externalize context before compaction"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$HOME/.local/share/rlm-claude-code\" && .venv/bin/python \"$HOME/.local/share/rlm-claude-code/scripts/save_trajectory.py\"",
            "timeout": 5000,
            "description": "Save trajectory on session end"
          }
        ]
      }
    ]
  }
}
HOOKS_EOF

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Add hooks to your Claude settings. You can either:"
echo "   a) Manually merge $RLM_ROOT/hooks.json into ~/.claude/settings.json"
echo "   b) Run: cat $RLM_ROOT/hooks.json"
echo ""
echo "2. Copy your .env file:"
echo "   cp $RLM_ROOT/.env.example $RLM_ROOT/.env"
echo "   # Edit .env with your API keys"
echo ""
echo "3. Restart Claude Code to load the new commands"
echo ""
echo "Available commands:"
echo "  /rlm        - Toggle/configure RLM mode"
echo "  /simple     - Bypass RLM for single operation"
echo "  /test       - Run test suite"
echo "  /bench      - Run benchmarks"
echo "  /code-review - Review code changes"
echo "  /trajectory - Analyze trajectory files"
