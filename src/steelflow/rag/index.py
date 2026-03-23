"""Indexing contracts for RAG."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .models import Document, RetrievalItem


@runtime_checkable
class Index(Protocol):
    def add_documents(self, documents: Sequence[Document]) -> None:
        """Add or update documents in the index."""

    def delete(self, document_ids: Sequence[str]) -> None:
        """Remove documents from the index."""

    def search(self, query: str, top_k: int) -> Sequence[RetrievalItem]:
        """Search the index and return retrieval items."""

    def version(self) -> str:
        """Return index version or snapshot id."""
