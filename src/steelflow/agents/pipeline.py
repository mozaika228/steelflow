"""Agent pipeline implementations without framework bloat."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

from .base import AgentPipeline, Tool
from ..contracts import AgentStepResult


class AgentStep(Protocol):
    name: str

    def run(
        self,
        task: str,
        tools: Sequence[Tool],
        state: Mapping[str, object],
    ) -> AgentStepResult:
        """Execute a single step and return the result."""


@dataclass(frozen=True)
class FunctionStep:
    name: str
    func: Callable[[str, Sequence[Tool], Mapping[str, object]], AgentStepResult]

    def run(
        self,
        task: str,
        tools: Sequence[Tool],
        state: Mapping[str, object],
    ) -> AgentStepResult:
        return self.func(task, tools, state)


@dataclass
class SimplePipeline(AgentPipeline):
    steps: Sequence[AgentStep]
    max_steps: int | None = None
    state: dict[str, object] = field(default_factory=dict)

    def run(
        self,
        task: str,
        tools: Sequence[Tool],
        **kwargs: object,
    ) -> Sequence[AgentStepResult]:
        results: list[AgentStepResult] = []
        limit = self.max_steps or len(self.steps)

        for step in self.steps[:limit]:
            result = step.run(task, tools, dict(self.state))
            results.append(result)

        return results
