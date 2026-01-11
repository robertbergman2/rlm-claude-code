#!/bin/bash
# Setup script for RLM-Claude-Code REPL environment
# 
# This script creates an isolated Python environment with all
# required tooling for the RLM REPL sandbox.

set -e

echo "Setting up RLM-Claude-Code environment..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create isolated venv
RLM_ENV="${HOME}/.claude/rlm-repl-env"
echo "Creating virtual environment at ${RLM_ENV}..."
uv venv "${RLM_ENV}" --python 3.12

# Install dependencies
echo "Installing dependencies..."
uv pip install --python "${RLM_ENV}" \
    pydantic>=2.0 \
    hypothesis>=6.0 \
    cpmpy>=0.9.20 \
    restrictedpython>=7.0 \
    anthropic>=0.40.0

# Install dev tools
echo "Installing development tools..."
uv pip install --python "${RLM_ENV}" \
    ruff \
    pytest>=8.0 \
    pytest-asyncio>=0.24 \
    pytest-cov>=5.0 \
    pytest-benchmark>=4.0

# Install ty if available
if command -v ty &> /dev/null; then
    echo "ty already installed"
else
    echo "Note: Install ty separately via: cargo install ty"
fi

# Verify installation
echo "Verifying installation..."
"${RLM_ENV}/bin/python" -c "
import pydantic
import hypothesis
import cpmpy
print('✓ pydantic:', pydantic.VERSION)
print('✓ hypothesis:', hypothesis.__version__)
print('✓ cpmpy:', cpmpy.__version__)
print('All dependencies installed successfully!')
"

# Create config directory
mkdir -p "${HOME}/.claude"

echo ""
echo "Setup complete!"
echo "Environment: ${RLM_ENV}"
echo ""
echo "To activate: source ${RLM_ENV}/bin/activate"
echo "To use in project: uv sync"
