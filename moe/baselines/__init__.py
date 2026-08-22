"""Third-party baselines: vLLM, SGLang, and plain torch.

Each baseline declares `env` so the runner knows which virtual environment can
execute it. Modules that need a framework import are expected to fail on the
laptop and are skipped, exactly like kernels.
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
            warnings.warn(f"baseline {mod.name!r} did not import: {e}", stacklevel=2)
    return loaded
