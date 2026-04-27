from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pytest

from steelflow.contracts import (
    EmbeddingResult,
    EnergyProfile,
    GenerationResult,
    HardwareProfile,
    ModelSelection,
    ModelSpec,
    QuantizationSpec,
)
from steelflow.engine import (
    CancellationToken,
    InferenceCancelledError,
    InferenceRuntime,
    RetryPolicy,
    RuntimeConfig,
)


@dataclass
class FakeEngine:
    name: str
    fail_first: bool = False
    always_fail: bool = False

    def __post_init__(self) -> None:
        self._calls = 0

    def capabilities(self) -> Mapping[str, object]:
        return {"generate": True, "embed": True}

    def select_model(
        self,
        hardware: HardwareProfile,
        energy: EnergyProfile,
        preferences: Mapping[str, object],
    ) -> ModelSelection:
        _ = hardware, energy, preferences
        return ModelSelection(
            model=ModelSpec(name=self.name, context_length=1024, parameters_b=None),
            quantization=QuantizationSpec(scheme="q4", bits=4),
            backend=self.name,
            precision="auto",
        )

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        _ = kwargs
        self._calls += 1
        if self.always_fail:
            raise RuntimeError(f"{self.name} failed")
        if self.fail_first and self._calls == 1:
            raise RuntimeError(f"{self.name} transient")
        return GenerationResult(
            text=f"{self.name}:{prompt}",
            tokens_generated=2,
            latency_ms=1.0,
            metadata={"engine": self.name},
        )

    def embed(self, texts: Sequence[str], **kwargs: object) -> EmbeddingResult:
        _ = kwargs
        vectors = [[0.1, 0.2] for _ in texts]
        return EmbeddingResult(vectors=vectors, dimension=2, latency_ms=1.0)


def _runtime(backends: Mapping[str, FakeEngine]) -> InferenceRuntime:
    return InferenceRuntime(
        backends=backends,
        config=RuntimeConfig(
            primary_backend="primary",
            fallback_backends=["fallback"],
            retry=RetryPolicy(max_attempts=2, backoff_seconds=0.0),
            stream_chunk_size=4,
        ),
    )


def test_runtime_retries_primary_before_fallback() -> None:
    runtime = _runtime(
        {
            "primary": FakeEngine("primary", fail_first=True),
            "fallback": FakeEngine("fallback"),
        }
    )

    result = runtime.generate("hello")

    assert result.text == "primary:hello"
    assert result.metadata["backend"] == "primary"
    assert result.metadata["attempt"] == 2


def test_runtime_fallback_when_primary_fails() -> None:
    runtime = _runtime(
        {
            "primary": FakeEngine("primary", always_fail=True),
            "fallback": FakeEngine("fallback"),
        }
    )

    result = runtime.generate("hello")

    assert result.text == "fallback:hello"
    assert result.metadata["backend"] == "fallback"


def test_runtime_batch_generation() -> None:
    runtime = _runtime(
        {
            "primary": FakeEngine("primary"),
            "fallback": FakeEngine("fallback"),
        }
    )

    results = runtime.batch_generate(["a", "b", "c"])

    assert [res.text for res in results] == ["primary:a", "primary:b", "primary:c"]


def test_runtime_cancellation() -> None:
    runtime = _runtime(
        {
            "primary": FakeEngine("primary"),
            "fallback": FakeEngine("fallback"),
        }
    )
    token = CancellationToken()
    token.cancel()

    with pytest.raises(InferenceCancelledError):
        runtime.generate("hello", cancellation_token=token)
