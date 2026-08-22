"""Third-party baselines: vLLM, SGLang, and plain torch.

Each baseline declares `env` so the runner knows which virtual environment can
execute it. Modules that need a framework import are expected to fail on the
laptop and are skipped, exactly like kernels.
"""
from __future__ import annotations

from ..stages import load_package


def load_all(strict: bool = False) -> list[str]:
    return load_package(__path__, __name__, "baseline", strict)
