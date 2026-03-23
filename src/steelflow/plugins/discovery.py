"""Entry-point discovery for external plugins."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, cast

from .base import PluginSpec

try:
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover
    from importlib_metadata import entry_points  # type: ignore


def discover_entry_points(group: str = "steelflow.plugins") -> Iterable[PluginSpec]:
    eps = entry_points()
    if hasattr(eps, "select"):
        selected = eps.select(group=group)
    else:
        legacy = cast(Mapping[str, Sequence[object]], eps)
        selected = legacy.get(group, ())

    for ep in selected:
        spec = ep.load()
        if not isinstance(spec, PluginSpec):
            raise TypeError(f"Entry point {ep.name} did not return PluginSpec")
        yield spec
