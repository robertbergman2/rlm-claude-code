"""
Main RLM orchestration loop.

Implements: Spec §2 Architecture Overview
"""

import asyncio
from collections.abc import AsyncIterator

from .complexity_classifier import should_activate_rlm
from .config import RLMConfig, default_config
from .context_manager import externalize_context
from .repl_environment import RLMEnvironment
from .trajectory import (
    StreamingTrajectory,
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryRenderer,
)
from .types import SessionContext


class RLMOrchestrator:
    """
    Main RLM orchestration loop.

    Implements: Spec §2.1 High-Level Design
    """

    def __init__(self, config: RLMConfig | None = None):
        """
        Initialize orchestrator.

        Args:
            config: RLM configuration (uses default if None)
        """
        self.config = config or default_config
        self.activation_reason: str = ""

    async def run(
        self, query: str, context: SessionContext
    ) -> AsyncIterator[TrajectoryEvent | str]:
        """
        Run RLM loop on a query.

        Implements: Spec §2 Architecture Overview

        Args:
            query: User query
            context: Session context

        Yields:
            TrajectoryEvents and final response
        """
        # Check if RLM should activate
        should_activate, self.activation_reason = should_activate_rlm(query, context)

        if not should_activate:
            # Bypass RLM, return direct
            yield TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                depth=0,
                content=f"[Direct mode: {self.activation_reason}]",
            )
            return

        # Initialize trajectory
        renderer = TrajectoryRenderer(
            verbosity=self.config.trajectory.verbosity,
            colors=self.config.trajectory.colors,
        )
        trajectory = StreamingTrajectory(renderer)

        # Start event
        start_event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            depth=0,
            content=f"depth=0/{self.config.depth.max} • task: {self.activation_reason}",
            metadata={"query": query, "context_tokens": context.total_tokens},
        )
        await trajectory.emit(start_event)
        yield start_event

        # Initialize REPL
        RLMEnvironment(context)

        # Analyze context
        analyze_event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            depth=0,
            content=f"Context: {context.total_tokens} tokens, {len(context.files)} files",
            metadata=externalize_context(context).get("context_stats"),
        )
        await trajectory.emit(analyze_event)
        yield analyze_event

        # TODO: Implement full RLM loop
        # - REPL execution
        # - Recursive calls
        # - Tool use
        # - Final answer extraction

        # Placeholder final
        final_event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=0,
            content="[RLM orchestration not yet implemented]",
        )
        await trajectory.emit(final_event)
        yield final_event

        # Export trajectory if enabled
        if self.config.trajectory.export_enabled:
            import os
            from pathlib import Path

            export_dir = Path(os.path.expanduser(self.config.trajectory.export_path))
            export_dir.mkdir(parents=True, exist_ok=True)

            import time

            filename = f"trajectory_{int(time.time())}.json"
            trajectory.export_json(str(export_dir / filename))


async def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="RLM Orchestrator")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument(
        "--verbosity", default="normal", choices=["minimal", "normal", "verbose", "debug"]
    )
    parser.add_argument("--export-trajectory", help="Path to export trajectory JSON")
    args = parser.parse_args()

    # Create mock context for testing
    context = SessionContext(
        files={"README.md": "# Test Project\n\nA test project."},
    )

    config = RLMConfig()
    config.trajectory.verbosity = args.verbosity

    orchestrator = RLMOrchestrator(config)

    async for event in orchestrator.run(args.query, context):
        if isinstance(event, TrajectoryEvent):
            renderer = TrajectoryRenderer(verbosity=args.verbosity)
            print(renderer.render_event(event))


if __name__ == "__main__":
    asyncio.run(main())
