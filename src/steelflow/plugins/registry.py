"""Plugin registry and resolution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .base import PluginKind, PluginSpec


class PluginNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class ResolvedPlugin:
    spec: PluginSpec
    instance: object


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[PluginKind, dict[str, PluginSpec]] = {}

    def register(self, spec: PluginSpec) -> None:
        by_kind = self._plugins.setdefault(spec.kind, {})
        if spec.name in by_kind:
            raise ValueError(f"Plugin already registered: {spec.kind}:{spec.name}")
        by_kind[spec.name] = spec

    def list(self, kind: PluginKind) -> list[PluginSpec]:
        return sorted(self._plugins.get(kind, {}).values(), key=lambda s: (-s.priority, s.name))

    def get(self, kind: PluginKind, name: str) -> PluginSpec:
        try:
            return self._plugins[kind][name]
        except KeyError as exc:
            raise PluginNotFoundError(f"Plugin not found: {kind}:{name}") from exc

    def resolve(
        self,
        kind: PluginKind,
        name: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> ResolvedPlugin:
        if name is not None:
            spec = self.get(kind, name)
            return ResolvedPlugin(spec=spec, instance=spec.factory())

        candidates = self.list(kind)
        if tags:
            candidates = [spec for spec in candidates if _matches_tags(spec, tags)]

        if not candidates:
            raise PluginNotFoundError(f"No plugins available for {kind}")

        spec = candidates[0]
        return ResolvedPlugin(spec=spec, instance=spec.factory())


def _matches_tags(spec: PluginSpec, tags: Mapping[str, str]) -> bool:
    for key, value in tags.items():
        if spec.tags.get(key) != value:
            return False
    return True


def load_plugins(registry: PluginRegistry, specs: Iterable[PluginSpec]) -> None:
    for spec in specs:
        registry.register(spec)
