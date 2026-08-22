"""Your kernels live here. One file per kernel.

`load_all()` imports every module in this package so that `@register`ed spans
appear in the registry. A kernel that fails to import (for example a Triton
kernel on a machine without CUDA) is skipped with a warning rather than
crashing the harness, so the laptop can still run the CPU test suite.
"""
from __future__ import annotations

import importlib
import pkgutil
import warnings


def load_all(strict: bool = False) -> list[str]:
    loaded = []
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{mod.name}")
            loaded.append(mod.name)
        except Exception as e:  # noqa: BLE001
            if strict:
                raise
            warnings.warn(f"kernel module {mod.name!r} did not import: {e}", stacklevel=2)
    return loaded
