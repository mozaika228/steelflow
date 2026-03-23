"""Plugin contracts and registration for SteelFlow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Protocol, runtime_checkable


class PluginKind(str, Enum):
    ENGINE = "engine"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    REWRITER = "rewriter"
    VECTOR_STORE = "vector_store"
    KEY_VALUE_STORE = "key_value_store"
    MEMORY_STORE = "memory_store"
    TRACER = "tracer"
    METRICS = "metrics"
    LOGGER = "logger"


@runtime_checkable
class Plugin(Protocol):
    """Optional interface for plugins that expose metadata."""

    def info(self) -> Mapping[str, object]:
        """Return plugin metadata (version, capabilities, etc.)."""


@dataclass(frozen=True)
class PluginSpec:
    name: str
    kind: PluginKind
    factory: Callable[[], object]
    tags: Mapping[str, str] = field(default_factory=dict)
    priority: int = 0
