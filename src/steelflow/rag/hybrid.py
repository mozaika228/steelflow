"""Hybrid retrieval implementation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import RetrievalItem


@dataclass(frozen=True)
class HybridWeights:
    dense: float = 0.5
    sparse: float = 0.5


class HybridMerger:
    """Merge dense and sparse retrieval results by weighted score."""

    def __init__(self, weights: HybridWeights | None = None) -> None:
        self._weights = weights or HybridWeights()

    def merge(
        self,
        dense: Iterable[RetrievalItem],
        sparse: Iterable[RetrievalItem],
        top_k: int,
    ) -> Sequence[RetrievalItem]:
        scores: dict[str, float] = defaultdict(float)
        items: dict[str, RetrievalItem] = {}

        for item in dense:
            scores[item.id] += item.score * self._weights.dense
            items[item.id] = item

        for item in sparse:
            scores[item.id] += item.score * self._weights.sparse
            if item.id not in items:
                items[item.id] = item

        merged = [
            RetrievalItem(
                id=item.id,
                text=item.text,
                score=scores[item.id],
                metadata=item.metadata,
            )
            for item in items.values()
        ]
        merged.sort(key=lambda it: it.score, reverse=True)
        return merged[:top_k]
