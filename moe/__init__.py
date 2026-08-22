"""moe-kernels: a stage-span harness for MoE grouped-GEMM kernel work.

The core (spec, state, stages, pipeline) imports no torch and runs on a laptop.
Implementations live behind `bootstrap()` so that importing this package never
drags in CUDA, vLLM, or SGLang.
"""
from __future__ import annotations

from . import pipeline, spec, stages, state  # noqa: F401  (torch-free core)

__all__ = ["spec", "state", "stages", "pipeline", "bootstrap"]

_BOOTSTRAPPED: set[str] = set()


def bootstrap(*groups: str) -> None:
    """Import implementation modules so their spans register themselves.

    groups: "reference", "kernels", "baselines". Defaults to reference+kernels,
    which are the two that never need a framework install.
    """
    wanted = groups or ("reference", "kernels")
    for group in wanted:
        if group in _BOOTSTRAPPED:
            continue
        if group == "reference":
            from .reference import torch_ref  # noqa: F401
        elif group == "kernels":
            from . import kernels  # noqa: F401

            kernels.load_all()
        elif group == "baselines":
            from . import baselines  # noqa: F401

            baselines.load_all()
        else:
            raise ValueError(f"unknown bootstrap group {group!r}")
        _BOOTSTRAPPED.add(group)
