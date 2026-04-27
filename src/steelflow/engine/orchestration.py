"""Hardware and energy orchestration for backend/model/quant selection."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..contracts import EnergyProfile, HardwareProfile, ModelSelection
from ..strategy.base import ExecutionPlan
from .base import Engine


@dataclass(frozen=True)
class BackendProfileResult:
    backend: str
    healthy: bool
    warmup_latency_ms: float | None
    throughput_tokens_per_s: float | None
    error: str | None = None


@dataclass(frozen=True)
class OrchestrationReport:
    hardware: HardwareProfile
    energy: EnergyProfile
    backend_profiles: Sequence[BackendProfileResult]


@dataclass(frozen=True)
class PolicyConfig:
    low_battery_percent: int = 20
    thermal_limit_c: float = 85.0
    prefer_fp16_when_plugged: bool = True


class HardwareEnergyDetector:
    def detect(self) -> tuple[HardwareProfile, EnergyProfile]:
        hardware = HardwareProfile(
            cpu_cores=os.cpu_count() or 1,
            ram_gb=_detect_ram_gb(),
            gpu_name=_detect_gpu_name(),
            vram_gb=_detect_vram_gb(),
            supports_cuda=_command_exists("nvidia-smi"),
            supports_rocm=_command_exists("rocm-smi"),
            numa_nodes=_detect_numa_nodes(),
            cpu_arch=platform.machine() or None,
        )
        energy = _detect_energy_profile()
        return hardware, energy


class StartupProfiler:
    def profile(
        self,
        backends: Mapping[str, Engine],
        prompt: str = "Hello",
    ) -> Sequence[BackendProfileResult]:
        results: list[BackendProfileResult] = []

        for name, backend in backends.items():
            start = time.perf_counter()
            try:
                result = backend.generate(prompt, max_new_tokens=16)
                elapsed = (time.perf_counter() - start) * 1000
                tps = _safe_tokens_per_second(result.tokens_generated, result.latency_ms)
                results.append(
                    BackendProfileResult(
                        backend=name,
                        healthy=True,
                        warmup_latency_ms=elapsed,
                        throughput_tokens_per_s=tps,
                    )
                )
            except Exception as exc:
                results.append(
                    BackendProfileResult(
                        backend=name,
                        healthy=False,
                        warmup_latency_ms=None,
                        throughput_tokens_per_s=None,
                        error=str(exc),
                    )
                )
        return results


class HardwareEnergyPolicyEngine:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self._config = config or PolicyConfig()

    def build_plan(
        self,
        hardware: HardwareProfile,
        energy: EnergyProfile,
        candidates: Sequence[ModelSelection],
        healthy_backends: Iterable[str],
    ) -> ExecutionPlan:
        healthy = set(healthy_backends)
        viable = [c for c in candidates if c.backend in healthy]
        if not viable:
            viable = list(candidates)
        if not viable:
            raise ValueError("No model candidates available")

        ranked = sorted(viable, key=lambda c: self._score(c, hardware, energy), reverse=True)
        primary = ranked[0]
        fallback = [c.backend for c in ranked[1:]]

        adjusted = self._degrade_if_needed(primary, energy)
        return ExecutionPlan(
            selection=adjusted,
            max_retries=2,
            timeout_s=60.0,
            fallback_backends=fallback,
            metadata={
                "battery_percent": energy.battery_percent,
                "thermal_throttling": energy.thermal_throttling,
                "numa_nodes": hardware.numa_nodes,
            },
        )

    def _score(
        self,
        candidate: ModelSelection,
        hardware: HardwareProfile,
        energy: EnergyProfile,
    ) -> float:
        score = 0.0

        if hardware.supports_cuda and candidate.backend in {"vllm", "exllama-v2", "transformers"}:
            score += 3.0
        if candidate.backend == "llama.cpp" and not hardware.supports_cuda:
            score += 2.0

        bits = candidate.quantization.bits
        if bits is not None:
            score += max(0.0, 9.0 - float(bits)) * 0.2

        low_battery = (energy.battery_percent or 100) <= self._config.low_battery_percent
        if energy.on_battery and low_battery:
            if bits is not None and bits <= 4:
                score += 3.0
            if candidate.precision in {"int8", "fp16"}:
                score += 1.0

        if self._config.prefer_fp16_when_plugged and not energy.on_battery:
            if candidate.precision in {"fp16", "bf16"}:
                score += 1.5

        if energy.thermal_throttling:
            score -= 0.5

        return score

    def _degrade_if_needed(
        self,
        selection: ModelSelection,
        energy: EnergyProfile,
    ) -> ModelSelection:
        battery = energy.battery_percent if energy.battery_percent is not None else 100
        if not energy.on_battery:
            return selection
        if battery > self._config.low_battery_percent and not energy.low_power_mode:
            return selection

        precision = "int8" if selection.precision not in {"int8", "q4"} else selection.precision
        bits = selection.quantization.bits
        if bits is None or bits > 4:
            quant = type(selection.quantization)(scheme="q4", bits=4)
        else:
            quant = selection.quantization

        return type(selection)(
            model=selection.model,
            quantization=quant,
            backend=selection.backend,
            precision=precision,
        )


def build_orchestration_report(
    backends: Mapping[str, Engine],
    detector: HardwareEnergyDetector | None = None,
    profiler: StartupProfiler | None = None,
) -> OrchestrationReport:
    detector_impl = detector or HardwareEnergyDetector()
    profiler_impl = profiler or StartupProfiler()

    hardware, energy = detector_impl.detect()
    profiles = profiler_impl.profile(backends)
    return OrchestrationReport(hardware=hardware, energy=energy, backend_profiles=profiles)


def _safe_tokens_per_second(tokens: int, latency_ms: float) -> float | None:
    if latency_ms <= 0:
        return None
    return float(tokens) / (latency_ms / 1000.0)


def _detect_ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        return 0.0


def _detect_energy_profile() -> EnergyProfile:
    try:
        import psutil

        battery = psutil.sensors_battery()
        if battery is None:
            return EnergyProfile(on_battery=False, battery_percent=None, thermal_throttling=False)
        percent = _safe_int(battery.percent)
        throttling, temp = _detect_thermal_state()
        low_power = bool(percent is not None and percent <= 20)
        return EnergyProfile(
            on_battery=bool(not battery.power_plugged),
            battery_percent=percent,
            thermal_throttling=throttling,
            temperature_c=temp,
            low_power_mode=low_power,
        )
    except Exception:
        throttling, temp = _detect_thermal_state()
        return EnergyProfile(
            on_battery=False,
            battery_percent=None,
            thermal_throttling=throttling,
            temperature_c=temp,
            low_power_mode=False,
        )


def _detect_thermal_state() -> tuple[bool, float | None]:
    try:
        import psutil

        temp = None
        if hasattr(psutil, "sensors_temperatures"):
            sensors = psutil.sensors_temperatures()
            for entries in sensors.values():
                if entries:
                    temp = float(entries[0].current)
                    break
        if temp is None and hasattr(psutil, "cpu_freq"):
            freq = psutil.cpu_freq()
            if freq and freq.max and freq.current and freq.current < (0.7 * freq.max):
                return True, None
        if temp is not None and temp >= 85.0:
            return True, temp
        return False, temp
    except Exception:
        return False, None


def _detect_gpu_name() -> str | None:
    out = _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if out:
        return out.splitlines()[0].strip() or None
    return None


def _detect_vram_gb() -> float | None:
    out = _run_command(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return None
    first = out.splitlines()[0].strip()
    val = _safe_float(first)
    if val is None:
        return None
    return round(val / 1024.0, 2)


def _detect_numa_nodes() -> int:
    out = _run_command(["numactl", "--hardware"])
    if out:
        for line in out.splitlines():
            if line.lower().startswith("available:"):
                parts = line.split()
                if len(parts) >= 2:
                    n = _safe_int(parts[1])
                    if n is not None:
                        return max(1, n)
    return 1


def _command_exists(command: str) -> bool:
    out = _run_command([command, "--help"])
    return out is not None


def _run_command(cmd: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if proc.returncode != 0 and not proc.stdout:
            return None
        return proc.stdout.strip() or None
    except Exception:
        return None


def _safe_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _safe_float(value: object) -> float | None:
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
