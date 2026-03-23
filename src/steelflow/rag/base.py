"""RAG contracts."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from ..contracts import RerankResult, RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, **kwargs: object) -> RetrievalResult:
        """Retrieve candidate documents."""


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: RetrievalResult) -> RerankResult:
        """Rerank retrieved candidates."""


@runtime_checkable
class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> str:
        """Rewrite queries to improve recall."""


@runtime_checkable
class RAGPipeline(Protocol):
    def run(self, query: str, **kwargs: object) -> Mapping[str, object]:
        """Execute a full RAG flow and return structured output."""
