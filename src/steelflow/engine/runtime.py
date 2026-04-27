"""Reliable inference runtime with retries, fallback, batching, and cancellation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Sequence

from ..contracts import EmbeddingResult, GenerationResult
from .base import Engine


class InferenceCancelledError(RuntimeError):
    """Raised when inference is cancelled."""


@dataclass
class CancellationToken:
    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    backoff_seconds: float = 0.0


@dataclass(frozen=True)
class RuntimeConfig:
    primary_backend: str
    fallback_backends: Sequence[str] = ()
    retry: RetryPolicy = RetryPolicy()
    stream_chunk_size: int = 128


class InferenceRuntime:
    def __init__(
        self,
        backends: Mapping[str, Engine],
        config: RuntimeConfig,
    ) -> None:
        if config.primary_backend not in backends:
            raise ValueError(f"Primary backend not found: {config.primary_backend}")
        self._backends = dict(backends)
        self._config = config

    def backend_order(self, preferred_backend: str | None = None) -> list[str]:
        if preferred_backend:
            if preferred_backend not in self._backends:
                raise ValueError(f"Unknown backend: {preferred_backend}")
            order = [preferred_backend]
        else:
            order = [self._config.primary_backend]

        for name in self._config.fallback_backends:
            if name in self._backends and name not in order:
                order.append(name)

        for name in self._backends:
            if name not in order:
                order.append(name)

        return order

    def health_check(self) -> Mapping[str, bool]:
        result: dict[str, bool] = {}
        for name, backend in self._backends.items():
            result[name] = self._backend_healthy(backend)
        return result

    def warmup(self, prompt: str = "Hello") -> Mapping[str, bool]:
        result: dict[str, bool] = {}
        for name, backend in self._backends.items():
            try:
                if hasattr(backend, "warmup"):
                    backend.warmup(prompt=prompt)  # type: ignore[attr-defined]
                else:
                    backend.generate(prompt, max_new_tokens=1)
                result[name] = True
            except Exception:
                result[name] = False
        return result

    def generate(
        self,
        prompt: str,
        *,
        preferred_backend: str | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> GenerationResult:
        order = self.backend_order(preferred_backend)
        errors: list[str] = []

        for backend_name in order:
            backend = self._backends[backend_name]
            for attempt in range(1, self._config.retry.max_attempts + 1):
                self._ensure_not_cancelled(cancellation_token)
                try:
                    result = backend.generate(prompt, **kwargs)
                    metadata = dict(result.metadata)
                    metadata["backend"] = backend_name
                    metadata["attempt"] = attempt
                    return GenerationResult(
                        text=result.text,
                        tokens_generated=result.tokens_generated,
                        latency_ms=result.latency_ms,
                        metadata=metadata,
                    )
                except Exception as exc:
                    errors.append(f"{backend_name}:attempt={attempt}:{exc}")
                    if attempt < self._config.retry.max_attempts:
                        time.sleep(self._config.retry.backoff_seconds)

        raise RuntimeError("Inference failed across backends: " + " | ".join(errors))

    def generate_stream(
        self,
        prompt: str,
        *,
        preferred_backend: str | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> Iterator[str]:
        backend_name = self.backend_order(preferred_backend)[0]
        backend = self._backends[backend_name]

        if hasattr(backend, "generate_stream"):
            for chunk in backend.generate_stream(prompt, **kwargs):  # type: ignore[attr-defined]
                self._ensure_not_cancelled(cancellation_token)
                yield chunk
            return

        result = self.generate(
            prompt,
            preferred_backend=backend_name,
            cancellation_token=cancellation_token,
            **kwargs,
        )
        for i in range(0, len(result.text), self._config.stream_chunk_size):
            self._ensure_not_cancelled(cancellation_token)
            yield result.text[i : i + self._config.stream_chunk_size]

    def batch_generate(
        self,
        prompts: Sequence[str],
        *,
        preferred_backend: str | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> Sequence[GenerationResult]:
        return [
            self.generate(
                prompt,
                preferred_backend=preferred_backend,
                cancellation_token=cancellation_token,
                **kwargs,
            )
            for prompt in prompts
        ]

    def embed(
        self,
        texts: Sequence[str],
        *,
        preferred_backend: str | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> EmbeddingResult:
        order = self.backend_order(preferred_backend)
        errors: list[str] = []

        for backend_name in order:
            backend = self._backends[backend_name]
            for attempt in range(1, self._config.retry.max_attempts + 1):
                self._ensure_not_cancelled(cancellation_token)
                try:
                    return backend.embed(texts, **kwargs)
                except Exception as exc:
                    errors.append(f"{backend_name}:attempt={attempt}:{exc}")
                    if attempt < self._config.retry.max_attempts:
                        time.sleep(self._config.retry.backoff_seconds)

        raise RuntimeError("Embedding failed across backends: " + " | ".join(errors))

    @staticmethod
    def _backend_healthy(backend: Engine) -> bool:
        try:
            if hasattr(backend, "health_check"):
                return bool(backend.health_check())  # type: ignore[attr-defined]
            caps = backend.capabilities()
            return bool(caps.get("generate") or caps.get("embed"))
        except Exception:
            return False

    @staticmethod
    def _ensure_not_cancelled(token: CancellationToken | None) -> None:
        if token and token.is_cancelled():
            raise InferenceCancelledError("Inference cancelled")
