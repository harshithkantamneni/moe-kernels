"""Your kernels live here. One file per kernel.

`load_all()` imports every module in this package so that `@register`ed spans
appear in the registry. A kernel that fails to import (for example a Triton
kernel on a machine without CUDA) is skipped with a warning rather than
crashing the harness, so the laptop can still run the CPU test suite.
"""
from __future__ import annotations

from ..stages import load_package


def load_all(strict: bool = False) -> list[str]:
    return load_package(__path__, __name__, "kernel module", strict)
