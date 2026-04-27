"""Core contracts and shared types for SteelFlow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class HardwareProfile:
    cpu_cores: int
    ram_gb: float
    gpu_name: str | None
    vram_gb: float | None
    supports_cuda: bool
    supports_rocm: bool
    numa_nodes: int = 1
    cpu_arch: str | None = None


@dataclass(frozen=True)
class EnergyProfile:
    on_battery: bool
    battery_percent: int | None
    thermal_throttling: bool
    temperature_c: float | None = None
    low_power_mode: bool = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    context_length: int
    parameters_b: float | None


@dataclass(frozen=True)
class QuantizationSpec:
    scheme: str  # e.g. q8, q6, q5, q4, int8, gguf
    bits: int | None


@dataclass(frozen=True)
class ModelSelection:
    model: ModelSpec
    quantization: QuantizationSpec
    backend: str
    precision: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    tokens_generated: int
    latency_ms: float
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: Sequence[Sequence[float]]
    dimension: int
    latency_ms: float


@dataclass(frozen=True)
class RetrievalResult:
    items: Sequence[Mapping[str, object]]
    latency_ms: float


@dataclass(frozen=True)
class RerankResult:
    items: Sequence[Mapping[str, object]]
    latency_ms: float


@dataclass(frozen=True)
class AgentStepResult:
    name: str
    output: str
    latency_ms: float
    metadata: Mapping[str, object]
