#!/usr/bin/env bash
set -euo pipefail

# RLM Claude Code Uninstaller

RLM_ROOT="${RLM_ROOT:-$HOME/.local/share/rlm-claude-code}"
CLAUDE_DIR="$HOME/.claude"

echo "Uninstalling RLM Claude Code..."
echo ""

# Remove commands
echo "Removing commands..."
rm -f "$CLAUDE_DIR/commands/rlm.md"
rm -f "$CLAUDE_DIR/commands/simple.md"
rm -f "$CLAUDE_DIR/commands/test.md"
rm -f "$CLAUDE_DIR/commands/bench.md"
rm -f "$CLAUDE_DIR/commands/code-review.md"
rm -f "$CLAUDE_DIR/commands/trajectory.md"

# Remove skills
echo "Removing skills..."
rm -rf "$CLAUDE_DIR/skills/rlm-context-management"
rm -rf "$CLAUDE_DIR/skills/constraint-verification"

# Remove core installation
if [[ -d "$RLM_ROOT" ]]; then
    echo "Removing installation at $RLM_ROOT..."
    rm -rf "$RLM_ROOT"
fi

echo ""
echo "Uninstall complete!"
echo ""
echo "Note: You may need to manually remove RLM hooks from ~/.claude/settings.json"
echo "Look for hooks referencing 'rlm-claude-code' and remove them."
