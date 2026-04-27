from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from steelflow.contracts import (
    EmbeddingResult,
    EnergyProfile,
    GenerationResult,
    HardwareProfile,
    ModelSelection,
    ModelSpec,
    QuantizationSpec,
)
from steelflow.engine import HardwareEnergyPolicyEngine, StartupProfiler, build_orchestration_report


@dataclass
class FakeEngine:
    name: str
    fail: bool = False

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
            quantization=QuantizationSpec(scheme="q8", bits=8),
            backend=self.name,
            precision="fp16",
        )

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        _ = kwargs
        if self.fail:
            raise RuntimeError("backend failure")
        return GenerationResult(
            text=f"{self.name}:{prompt}",
            tokens_generated=10,
            latency_ms=10.0,
            metadata={},
        )

    def embed(self, texts: Sequence[str], **kwargs: object) -> EmbeddingResult:
        _ = kwargs
        vectors = [[0.1, 0.2] for _ in texts]
        return EmbeddingResult(vectors=vectors, dimension=2, latency_ms=1.0)


class FakeDetector:
    def detect(self) -> tuple[HardwareProfile, EnergyProfile]:
        return (
            HardwareProfile(
                cpu_cores=8,
                ram_gb=32.0,
                gpu_name="TestGPU",
                vram_gb=12.0,
                supports_cuda=True,
                supports_rocm=False,
            ),
            EnergyProfile(
                on_battery=True,
                battery_percent=10,
                thermal_throttling=False,
                low_power_mode=True,
            ),
        )


def test_policy_engine_degrades_on_low_battery() -> None:
    policy = HardwareEnergyPolicyEngine()
    hardware = HardwareProfile(
        cpu_cores=8,
        ram_gb=32.0,
        gpu_name="TestGPU",
        vram_gb=12.0,
        supports_cuda=True,
        supports_rocm=False,
    )
    energy = EnergyProfile(on_battery=True, battery_percent=10, thermal_throttling=False)

    high = ModelSelection(
        model=ModelSpec(name="big", context_length=8192, parameters_b=13.0),
        quantization=QuantizationSpec(scheme="q8", bits=8),
        backend="vllm",
        precision="fp16",
    )
    low = ModelSelection(
        model=ModelSpec(name="small", context_length=4096, parameters_b=7.0),
        quantization=QuantizationSpec(scheme="q4", bits=4),
        backend="llama.cpp",
        precision="int8",
    )

    plan = policy.build_plan(hardware, energy, [high, low], healthy_backends=["vllm", "llama.cpp"])

    assert plan.selection.quantization.bits == 4
    assert plan.selection.precision in {"int8", "q4"}


def test_startup_profiler_marks_failed_backend() -> None:
    profiler = StartupProfiler()
    profiles = profiler.profile(
        {
            "ok": FakeEngine("ok"),
            "bad": FakeEngine("bad", fail=True),
        }
    )

    indexed = {p.backend: p for p in profiles}
    assert indexed["ok"].healthy is True
    assert indexed["bad"].healthy is False


def test_build_orchestration_report() -> None:
    report = build_orchestration_report(
        backends={"ok": FakeEngine("ok")},
        detector=FakeDetector(),
    )

    assert report.hardware.cpu_cores == 8
    assert report.energy.on_battery is True
    assert len(report.backend_profiles) == 1
