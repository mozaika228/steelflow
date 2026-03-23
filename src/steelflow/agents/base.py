"""Agent pipeline contracts."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..contracts import AgentStepResult


@runtime_checkable
class Tool(Protocol):
    name: str

    def __call__(self, **kwargs: object) -> Mapping[str, object]:
        """Execute tool call and return structured output."""


@runtime_checkable
class AgentPipeline(Protocol):
    def run(
        self,
        task: str,
        tools: Sequence[Tool],
        **kwargs: object,
    ) -> Sequence[AgentStepResult]:
        """Run multi-agent pipeline and return step results."""
