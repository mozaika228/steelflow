"""Backend registry helpers."""

from __future__ import annotations

from typing import Iterable

from ..plugins import PluginKind, PluginSpec
from .backends.exllama_v2 import ExLlamaV2Backend, ExLlamaV2Config
from .backends.llama_cpp import LlamaCppBackend, LlamaCppConfig
from .backends.ollama import OllamaBackend, OllamaConfig
from .backends.transformers import TransformersBackend, TransformersConfig
from .backends.vllm_backend import VllmBackend, VllmConfig


def builtin_engine_plugins() -> Iterable[PluginSpec]:
    return [
        PluginSpec(
            name="ollama",
            kind=PluginKind.ENGINE,
            factory=lambda: OllamaBackend(OllamaConfig()),
            tags={"local": "true"},
            priority=100,
        ),
        PluginSpec(
            name="vllm",
            kind=PluginKind.ENGINE,
            factory=lambda: VllmBackend(VllmConfig(model_name="meta-llama/Llama-3-8B-Instruct")),
            tags={"local": "true", "gpu": "required"},
            priority=90,
        ),
        PluginSpec(
            name="exllama-v2",
            kind=PluginKind.ENGINE,
            factory=lambda: ExLlamaV2Backend(ExLlamaV2Config(model_dir="./model")),
            tags={"local": "true", "gpu": "required"},
            priority=80,
        ),
        PluginSpec(
            name="llama.cpp",
            kind=PluginKind.ENGINE,
            factory=lambda: LlamaCppBackend(LlamaCppConfig(model_path="model.gguf")),
            tags={"local": "true"},
            priority=70,
        ),
        PluginSpec(
            name="transformers",
            kind=PluginKind.ENGINE,
            factory=lambda: TransformersBackend(TransformersConfig(model_name="gpt2")),
            tags={"local": "true"},
            priority=60,
        ),
    ]
