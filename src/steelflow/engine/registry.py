"""Backend registry helpers."""

from __future__ import annotations

from typing import Iterable

from ..plugins import PluginKind, PluginSpec
from .backends.llama_cpp import LlamaCppBackend, LlamaCppConfig
from .backends.ollama import OllamaBackend, OllamaConfig
from .backends.transformers import TransformersBackend, TransformersConfig


def builtin_engine_plugins() -> Iterable[PluginSpec]:
    return [
        PluginSpec(
            name="ollama",
            kind=PluginKind.ENGINE,
            factory=lambda: OllamaBackend(OllamaConfig()),
            tags={"local": "true"},
            priority=10,
        ),
        PluginSpec(
            name="llama.cpp",
            kind=PluginKind.ENGINE,
            factory=lambda: LlamaCppBackend(
                LlamaCppConfig(model_path="model.gguf")
            ),
            tags={"local": "true"},
            priority=5,
        ),
        PluginSpec(
            name="transformers",
            kind=PluginKind.ENGINE,
            factory=lambda: TransformersBackend(
                TransformersConfig(model_name="gpt2")
            ),
            tags={"local": "true"},
            priority=1,
        ),
    ]
