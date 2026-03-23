"""Configuration objects for SteelFlow subsystems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    prefer_backend: str | None = None
    max_vram_gb: float | None = None
    energy_aware: bool = True


@dataclass(frozen=True)
class RAGConfig:
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rerank_top_k: int = 20
    rewrite_queries: bool = True


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    enable_critic: bool = True


@dataclass(frozen=True)
class ObservabilityConfig:
    enable_tracing: bool = True
    enable_metrics: bool = True


@dataclass(frozen=True)
class StorageConfig:
    persist_path: str | None = None
