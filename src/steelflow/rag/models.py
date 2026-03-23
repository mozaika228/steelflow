"""RAG data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalItem:
    id: str
    text: str
    score: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RAGContext:
    query: str
    rewritten_query: str | None
    retrieved: Sequence[RetrievalItem]
    reranked: Sequence[RetrievalItem]


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    context: RAGContext
    metadata: Mapping[str, object] = field(default_factory=dict)
