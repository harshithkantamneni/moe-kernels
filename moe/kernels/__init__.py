"""Your kernels live here. One file per kernel.

`load_all()` imports every module in this package so that `@register`ed spans
appear in the registry. A kernel that fails to import (for example a Triton
kernel on a machine without CUDA) is skipped with a warning rather than
crashing the harness, so the laptop can still run the CPU test suite.
"""
from __future__ import annotations

from ..stages import load_package

#: Architectures with an `a` variant. The suffix is what exposes the
#: architecture-specific instructions (wgmma and TMA on Hopper), so compiling
#: plain `sm_90` builds for the part but loses what the part exists for.
_ARCH_SUFFIX_FROM_MAJOR = 9


def cuda_arch_flags(capability: tuple[int, int] | None = None) -> list[str]:
    """nvcc `-gencode` flags for the attached device.

    Hardcoding Hopper is correct on an H100 or H200 and silently wrong on
    anything else, which matters as soon as a sweep runs on whatever GPU was
    available. Pass a capability to build for a specific part; pass nothing to
    read it from the device.
    """
    if capability is None:
        import torch
        capability = torch.cuda.get_device_capability()
    major, minor = capability
    if major < 5:
        raise ValueError(
            f"compute capability {major}.{minor} is not a CUDA architecture "
            "this project builds for; pass an explicit (major, minor).")
    suffix = "a" if major >= _ARCH_SUFFIX_FROM_MAJOR else ""
    arch = f"{major}{minor}{suffix}"
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


def load_all(strict: bool = False) -> list[str]:
    return load_package(__path__, __name__, "kernel module", strict)
