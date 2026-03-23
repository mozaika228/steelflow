"""Execution strategies and reliability policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..contracts import EnergyProfile, HardwareProfile, ModelSelection


@dataclass(frozen=True)
class ExecutionPlan:
    selection: ModelSelection
    max_retries: int = 1
    timeout_s: float | None = None
    fallback_backends: Sequence[str] = ()
    metadata: Mapping[str, object] | None = None


@runtime_checkable
class SelectionPolicy(Protocol):
    def select(
        self,
        hardware: HardwareProfile,
        energy: EnergyProfile,
        candidates: Sequence[ModelSelection],
    ) -> ExecutionPlan:
        """Choose the best execution plan from candidates."""


@runtime_checkable
class HealthPolicy(Protocol):
    def is_healthy(self, backend: str) -> bool:
        """Return whether a backend is healthy to use."""


@runtime_checkable
class RetryPolicy(Protocol):
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Return whether to retry a failed execution."""


@runtime_checkable
class Strategy(Protocol):
    def plan(
        self,
        hardware: HardwareProfile,
        energy: EnergyProfile,
        candidates: Sequence[ModelSelection],
    ) -> ExecutionPlan:
        """Build an execution plan for a request."""
