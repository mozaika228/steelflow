"""Observability contracts."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable


@runtime_checkable
class Span(Protocol):
    def set_attribute(self, key: str, value: object) -> None:
        """Attach an attribute to the span."""

    def end(self) -> None:
        """Close the span."""


@runtime_checkable
class Tracer(Protocol):
    def start_span(self, name: str) -> Span:
        """Create a new span."""


@runtime_checkable
class Metrics(Protocol):
    def observe(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        """Record a metric observation."""


@runtime_checkable
class Logger(Protocol):
    def info(self, message: str, **fields: object) -> None:
        """Log informational message."""

    def error(self, message: str, **fields: object) -> None:
        """Log error message."""
