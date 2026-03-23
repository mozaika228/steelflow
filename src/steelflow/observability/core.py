"""Observability facade to enforce tracing/metrics/logging presence."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Logger, Metrics, Tracer
from .noop import NoopLogger, NoopMetrics, NoopTracer


@dataclass(frozen=True)
class Observability:
    tracer: Tracer
    metrics: Metrics
    logger: Logger

    @classmethod
    def default(cls) -> "Observability":
        return cls(tracer=NoopTracer(), metrics=NoopMetrics(), logger=NoopLogger())
