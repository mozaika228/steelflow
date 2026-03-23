"""Professional RAG pipeline orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from .base import QueryRewriter, RAGPipeline, Reranker, Retriever
from .models import RAGContext, RetrievalItem


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

        retrieved_items = _normalize_items(retrieval.items)
        reranked_items = retrieved_items

        if self._reranker is not None:
            start = time.perf_counter()
            reranked = self._reranker.rerank(retrieval_query, retrieval)
            timings["rerank"] = (time.perf_counter() - start) * 1000
            reranked_items = _normalize_items(reranked.items)

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
                score=float(item.get("score", 0.0)),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return normalized
