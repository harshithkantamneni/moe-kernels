"""VM side of the v7 domain map (registration docs/registered/2026-11-18-v7-domain-map-gh200): puts the map's 14 synthetic
shapes (scripts/scoring/v7domain/map_shapes.py) into moe.spec.MODEL_CONFIGS when a process first imports moe.spec, so
the pinned decode driver, r3_decode.py and locked_r3.py run unchanged and moe/spec.py keeps its frozen bytes.

scripts/vm_run_v7m.sh starts the driver with PYTHONPATH=<checkout>/scripts/scoring/v7domain/vmsite; Python imports
this module at startup (site.py's sitecustomize). It changes nothing until `moe.spec` is imported: a one-shot meta path
finder lets the normal finder load moe/spec.py, then adds the shapes after the module body ran. Any failure here is
printed to stderr and leaves the process as it was (the page then refuses the unknown model by name)."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_SHAPES = Path(__file__).resolve().parents[1] / "map_shapes.py"


def _load_shapes():
    spec = importlib.util.spec_from_file_location("_v7domain_map_shapes", _SHAPES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _MapShapes(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "moe.spec":
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            try:
                _load_shapes().install(module)
            except Exception as e:  # noqa: BLE001
                print(f"v7domain sitecustomize: the map shapes were not added: {e!r}", file=sys.stderr)
        spec.loader.exec_module = exec_module
        return spec


if _SHAPES.is_file() and not any(isinstance(f, _MapShapes) for f in sys.meta_path):
    sys.meta_path.insert(0, _MapShapes())
