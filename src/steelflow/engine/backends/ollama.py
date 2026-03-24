"""Ollama backend (local HTTP)."""

from __future__ import annotations

import json
import time
import urllib.request
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
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3"


class OllamaBackend(Engine):
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self._config = config or OllamaConfig()

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="ollama",
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
        model_name = str(preferences.get("model", self._config.model))
        model = ModelSpec(name=model_name, context_length=8192, parameters_b=None)
        quant = QuantizationSpec(scheme="ollama", bits=None)
        return ModelSelection(model=model, quantization=quant, backend="ollama", precision="auto")

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        payload = {
            "model": kwargs.get("model", self._config.model),
            "prompt": prompt,
            "stream": False,
        }
        start = time.perf_counter()
        data = _post_json(f"{self._config.base_url}/api/generate", payload)
        latency_ms = (time.perf_counter() - start) * 1000
        text = str(data.get("response", ""))
        tokens = _to_int(data.get("eval_count", 0))
        return GenerationResult(
            text=text,
            tokens_generated=tokens,
            latency_ms=latency_ms,
            metadata=data,
        )

    def embed(self, texts: Sequence[str], **kwargs: object) -> EmbeddingResult:
        payload = {
            "model": kwargs.get("model", self._config.model),
            "input": list(texts),
        }
        start = time.perf_counter()
        data = _post_json(f"{self._config.base_url}/api/embeddings", payload)
        latency_ms = (time.perf_counter() - start) * 1000
        embeddings = data.get("embeddings", [])
        if not isinstance(embeddings, list):
            embeddings = []
        dimension = len(embeddings[0]) if embeddings else 0
        return EmbeddingResult(vectors=embeddings, dimension=dimension, latency_ms=latency_ms)


def _post_json(url: str, payload: Mapping[str, object]) -> Mapping[str, object]:
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError(f"Ollama request failed: {exc}") from exc


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
