"""KONASH: Knowledge-grounded Off-policy Networks for Agentic System Harnesses."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.9"
__all__ = ["Agent", "Corpus"]


def __getattr__(name: str) -> Any:
    if name == "Agent":
        return import_module("konash.api").Agent
    if name == "Corpus":
        return import_module("konash.corpus").Corpus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
