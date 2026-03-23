"""Engine contracts."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable, Sequence

from ..contracts import (
    EmbeddingResult,
    EnergyProfile,
    GenerationResult,
    HardwareProfile,
    ModelSelection,
)


@runtime_checkable
class Engine(Protocol):
    def capabilities(self) -> Mapping[str, object]:
        """Return backend capabilities and limits."""

    def select_model(
        self,
        hardware: HardwareProfile,
        energy: EnergyProfile,
        preferences: Mapping[str, object],
    ) -> ModelSelection:
        """Pick the best model/quantization/backend for the hardware."""

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        """Run text generation with the selected backend."""

    def embed(self, texts: Sequence[str], **kwargs: object) -> EmbeddingResult:
        """Create embeddings for RAG."""
