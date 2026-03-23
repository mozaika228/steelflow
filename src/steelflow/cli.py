"""CLI entry point for SteelFlow."""

from __future__ import annotations

import argparse

from .version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steelflow",
        description="Local-first LLM runtime and RAG/agent stack",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show SteelFlow version",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    print("SteelFlow scaffold is ready. See README.md for next steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
