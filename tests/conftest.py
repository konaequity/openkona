from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable

import pytest


@dataclass(frozen=True)
class SymbolSpec:
    module: str
    symbol: str | None = None

    @property
    def label(self) -> str:
        return f"{self.module}:{self.symbol}" if self.symbol else self.module


def load_symbol(spec: SymbolSpec):
    try:
        module = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"Missing module '{spec.module}'. Implement it to satisfy the KARL verification suite."
        )
    if spec.symbol is None:
        return module
    if not hasattr(module, spec.symbol):
        pytest.fail(
            f"Missing symbol '{spec.symbol}' in module '{spec.module}'."
        )
    return getattr(module, spec.symbol)


def assert_callable(obj, label: str) -> None:
    assert callable(obj), f"Expected {label} to be callable"


def assert_has_attrs(obj, attrs: Iterable[str], label: str) -> None:
    missing = [name for name in attrs if not hasattr(obj, name)]
    assert not missing, f"{label} is missing required attributes: {', '.join(missing)}"
