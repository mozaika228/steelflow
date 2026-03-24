"""Common backend helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..base import Engine
from ...contracts import EmbeddingResult, GenerationResult


@dataclass(frozen=True)
class BackendInfo:
    name: str
    version: str | None
    capabilities: Mapping[str, object]


class BackendUnavailableError(RuntimeError):
    pass


class BaseBackend(Engine):
    """Base class for concrete inference backends."""

    def info(self) -> BackendInfo:
        raise NotImplementedError

    def capabilities(self) -> Mapping[str, object]:
        return self.info().capabilities

    def select_model(self, hardware, energy, preferences):
        raise NotImplementedError

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        raise NotImplementedError

    def embed(self, texts, **kwargs: object) -> EmbeddingResult:
        raise NotImplementedError
