"""llama.cpp backend via llama-cpp-python (optional)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..base import Engine
from ...contracts import (
    EmbeddingResult,
    EnergyProfile,
    GenerationResult,
    HardwareProfile,
    ModelSelection,
    ModelSpec,
    QuantizationSpec,
)
from .base import BackendInfo, BackendUnavailableError


@dataclass(frozen=True)
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 4096


class LlamaCppBackend(Engine):
    def __init__(self, config: LlamaCppConfig) -> None:
        self._config = config
        try:
            from llama_cpp import Llama  # type: ignore

            self._llama = Llama(model_path=config.model_path, n_ctx=config.n_ctx)
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailableError(
                "llama-cpp-python is required for LlamaCppBackend"
            ) from exc

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="llama.cpp",
            version=None,
            capabilities={"generate": True, "embed": True},
        )

    def capabilities(self) -> Mapping[str, object]:
        return self.info().capabilities

    def select_model(
        self,
        hardware: HardwareProfile,
        energy: EnergyProfile,
        preferences: Mapping[str, object],
    ) -> ModelSelection:
        model = ModelSpec(
            name=self._config.model_path,
            context_length=self._config.n_ctx,
            parameters_b=None,
        )
        quant = QuantizationSpec(scheme="gguf", bits=None)
        return ModelSelection(
            model=model,
            quantization=quant,
            backend="llama.cpp",
            precision="auto",
        )

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        start = time.perf_counter()
        output = self._llama(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        text = str(output.get("choices", [{}])[0].get("text", ""))
        tokens = int(output.get("usage", {}).get("completion_tokens", 0) or 0)
        return GenerationResult(
            text=text,
            tokens_generated=tokens,
            latency_ms=latency_ms,
            metadata=output,
        )

    def embed(self, texts: Sequence[str], **kwargs: object) -> EmbeddingResult:
        start = time.perf_counter()
        vectors = [self._llama.embed(text) for text in texts]
        latency_ms = (time.perf_counter() - start) * 1000
        dimension = len(vectors[0]) if vectors else 0
        return EmbeddingResult(vectors=vectors, dimension=dimension, latency_ms=latency_ms)
