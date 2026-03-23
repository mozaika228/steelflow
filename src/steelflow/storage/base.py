"""Storage contracts."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable, Sequence


@runtime_checkable
class KeyValueStore(Protocol):
    def get(self, key: str) -> Mapping[str, object] | None:
        """Retrieve a record by key."""

    def put(self, key: str, value: Mapping[str, object]) -> None:
        """Store a record by key."""


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metadata: Sequence[Mapping[str, object]]) -> None:
        """Insert or update vectors."""

    def query(self, vector: Sequence[float], top_k: int) -> Sequence[Mapping[str, object]]:
        """Query nearest neighbors."""


@runtime_checkable
class MemoryStore(Protocol):
    def append(self, session_id: str, entry: Mapping[str, object]) -> None:
        """Append an entry to session memory."""

    def get_history(self, session_id: str, limit: int | None = None) -> Sequence[Mapping[str, object]]:
        """Get session history."""
