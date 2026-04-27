"""vLLM backend (optional)."""

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
class VllmConfig:
    model_name: str


class VllmBackend(Engine):
    def __init__(self, config: VllmConfig) -> None:
        self._config = config
        try:
            from vllm import LLM, SamplingParams

            self._sampling_params_cls = SamplingParams
            self._llm = LLM(model=config.model_name)
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailableError("vLLM is required for VllmBackend") from exc

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="vllm",
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
        model = ModelSpec(name=self._config.model_name, context_length=8192, parameters_b=None)
        quant = QuantizationSpec(scheme="auto", bits=None)
        return ModelSelection(model=model, quantization=quant, backend="vllm", precision="auto")

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        max_tokens = _to_int(kwargs.get("max_new_tokens", 128))
        start = time.perf_counter()
        params = self._sampling_params_cls(max_tokens=max_tokens)
        outputs = self._llm.generate([prompt], params)
        latency_ms = (time.perf_counter() - start) * 1000
        if not outputs:
            return GenerationResult(text="", tokens_generated=0, latency_ms=latency_ms, metadata={})
        text = outputs[0].outputs[0].text
        tokens = len(outputs[0].outputs[0].token_ids)
        return GenerationResult(
            text=text,
            tokens_generated=tokens,
            latency_ms=latency_ms,
            metadata={"backend": "vllm"},
        )

    def embed(self, texts: Sequence[str], **kwargs: object) -> EmbeddingResult:
        raise BackendUnavailableError("Embedding not supported in this minimal vLLM backend")


def _to_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
