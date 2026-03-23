"""No-op observability implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .base import Logger, Metrics, Span, Tracer


@dataclass
class NoopSpan(Span):
    def set_attribute(self, key: str, value: object) -> None:
        return None

    def end(self) -> None:
        return None


@dataclass
class NoopTracer(Tracer):
    def start_span(self, name: str) -> Span:
        return NoopSpan()


@dataclass
class NoopMetrics(Metrics):
    def observe(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        return None


@dataclass
class NoopLogger(Logger):
    def info(self, message: str, **fields: object) -> None:
        return None

    def error(self, message: str, **fields: object) -> None:
        return None
