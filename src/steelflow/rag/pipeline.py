"""Professional RAG pipeline orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from .base import QueryRewriter, RAGPipeline, Reranker, Retriever
from .models import RetrievalItem


@dataclass(frozen=True)
class PipelineOutput:
    query: str
    rewritten_query: str | None
    retrieved: Sequence[RetrievalItem]
    reranked: Sequence[RetrievalItem]
    timings_ms: Mapping[str, float]


class DefaultRAGPipeline(RAGPipeline):
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
        top_k: int = 20,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._rewriter = rewriter
        self._top_k = top_k

    def run(self, query: str, **kwargs: object) -> Mapping[str, object]:
        timings: dict[str, float] = {}

        rewritten: str | None = None
        if self._rewriter is not None:
            start = time.perf_counter()
            rewritten = self._rewriter.rewrite(query)
            timings["rewrite"] = (time.perf_counter() - start) * 1000

        retrieval_query = rewritten or query
        start = time.perf_counter()
        retrieval = self._retriever.retrieve(retrieval_query, **kwargs)
        timings["retrieve"] = (time.perf_counter() - start) * 1000

        retrieved_items = _normalize_items(_extract_items(retrieval))
        reranked_items = retrieved_items

        if self._reranker is not None:
            start = time.perf_counter()
            reranked = self._reranker.rerank(retrieval_query, retrieval)
            timings["rerank"] = (time.perf_counter() - start) * 1000
            reranked_items = _normalize_items(_extract_items(reranked))

        output = PipelineOutput(
            query=query,
            rewritten_query=rewritten,
            retrieved=retrieved_items[: self._top_k],
            reranked=reranked_items[: self._top_k],
            timings_ms=timings,
        )
        return {
            "query": output.query,
            "rewritten_query": output.rewritten_query,
            "retrieved": output.retrieved,
            "reranked": output.reranked,
            "timings_ms": output.timings_ms,
        }


def _normalize_items(items: Sequence[Mapping[str, object]]) -> Sequence[RetrievalItem]:
    normalized: list[RetrievalItem] = []
    for item in items:
        normalized.append(
            RetrievalItem(
                id=str(item.get("id", "")),
                text=str(item.get("text", "")),
                score=_to_float(item.get("score", 0.0)),
                metadata=_to_metadata(item.get("metadata", {})),
            )
        )
    return normalized


def _extract_items(source: object) -> Sequence[Mapping[str, object]]:
    if isinstance(source, Mapping):
        value = source.get("items", [])
        return _ensure_sequence(value)
    if hasattr(source, "items"):
        value = source.items
        return _ensure_sequence(value)
    return []


def _ensure_sequence(value: object) -> Sequence[Mapping[str, object]]:
    if isinstance(value, Sequence):
        return value
    return []


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _to_metadata(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
