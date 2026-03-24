"""Transformers backend (optional)."""

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
class TransformersConfig:
    model_name: str
    device: str = "auto"


class TransformersBackend(Engine):
    def __init__(self, config: TransformersConfig) -> None:
        self._config = config
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(config.model_name)
            if config.device != "auto":
                self._model.to(config.device)
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailableError(
                "transformers + torch are required for TransformersBackend"
            ) from exc

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="transformers",
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
        model = ModelSpec(name=self._config.model_name, context_length=2048, parameters_b=None)
        quant = QuantizationSpec(scheme="fp16", bits=None)
        return ModelSelection(
            model=model,
            quantization=quant,
            backend="transformers",
            precision="auto",
        )

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        start = time.perf_counter()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        output = self._model.generate(
            **inputs,
            max_new_tokens=int(kwargs.get("max_new_tokens", 128)),
        )
        text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        latency_ms = (time.perf_counter() - start) * 1000
        tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
        return GenerationResult(
            text=text,
            tokens_generated=tokens,
            latency_ms=latency_ms,
            metadata={},
        )

    def embed(self, texts: Sequence[str], **kwargs: object) -> EmbeddingResult:
        raise BackendUnavailableError("Embedding not supported in this minimal backend")
