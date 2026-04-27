"""exllama-v2 backend (optional)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from ...contracts import (
    EmbeddingResult,
    EnergyProfile,
    GenerationResult,
    HardwareProfile,
    ModelSelection,
    ModelSpec,
    QuantizationSpec,
)
from ..base import Engine
from .base import BackendInfo, BackendUnavailableError


@dataclass(frozen=True)
class ExLlamaV2Config:
    model_dir: str


class ExLlamaV2Backend(Engine):
    def __init__(self, config: ExLlamaV2Config) -> None:
        self._config = config
        self._generator = None
        try:
            import exllamav2

            self._exllamav2 = exllamav2
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailableError("exllamav2 is required for ExLlamaV2Backend") from exc

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="exllama-v2",
            version=None,
            capabilities={"generate": True, "embed": False},
        )

    def capabilities(self) -> Mapping[str, object]:
        return self.info().capabilities

    def select_model(
        self,
        hardware: HardwareProfile,
        energy: EnergyProfile,
        preferences: Mapping[str, object],
    ) -> ModelSelection:
        model = ModelSpec(name=self._config.model_dir, context_length=4096, parameters_b=None)
        quant = QuantizationSpec(scheme="gptq", bits=4)
        return ModelSelection(
            model=model,
            quantization=quant,
            backend="exllama-v2",
            precision="auto",
        )

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        # Placeholder integration path until full exllamav2 wiring is added.
        start = time.perf_counter()
        if self._generator is None:
            raise BackendUnavailableError("exllama-v2 generator is not initialized")
        _ = kwargs
        text = str(self._generator.generate_simple(prompt, max_new_tokens=128))
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(
            text=text,
            tokens_generated=0,
            latency_ms=latency_ms,
            metadata={"backend": "exllama-v2"},
        )

    def embed(self, texts: Sequence[str], **kwargs: object) -> EmbeddingResult:
        raise BackendUnavailableError("Embedding not supported in this minimal exllama-v2 backend")
