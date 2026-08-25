#!/usr/bin/env python
"""Expand the argument TYPES of a framework's fused_experts entry point.

The first probe gave signatures. vLLM 0.27.1 takes plain tensors and an enum;
SGLang 0.5.18 takes `topk_output: StandardTopKOutput` and
`moe_runner_config: MoeRunnerConfig`, which cannot be constructed from a
signature line. This resolves each annotation to its actual class and reports
how to build one, so the span is written against the real constructor.

    /workspace/venvs/vllm/bin/python   scripts/probe_baseline_types.py
    /workspace/venvs/sglang/bin/python scripts/probe_baseline_types.py

It also prints the source of fused_experts where it can be found, because the
weight layout and the gate/up split are stated there and nowhere else. Getting
those wrong produces a span that runs, is internally consistent, and computes a
different layer.
"""
from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import sys

ENTRIES = [
    ("vllm.model_executor.layers.fused_moe", "fused_experts"),
    ("sglang.srt.layers.moe.fused_moe_triton", "fused_experts"),
]

_BUILTIN = {int, float, bool, str, bytes, type(None)}


def describe_type(tp, depth: int = 0) -> None:
    pad = "  " * (depth + 2)
    name = getattr(tp, "__name__", repr(tp))
    if tp in _BUILTIN or name in ("Tensor", "Optional", "List", "Any"):
        return
    print(f"{pad}TYPE {getattr(tp, '__module__', '?')}.{name}")

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        print(f"{pad}  enum members: {[m.name + '=' + repr(m.value) for m in tp]}")
        return

    if dataclasses.is_dataclass(tp):
        for f in dataclasses.fields(tp):
            d = "" if f.default is dataclasses.MISSING else f" = {f.default!r}"
            if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                d = " = <factory>"
            print(f"{pad}  {f.name}: {f.type}{d}")
        return

    fields = getattr(tp, "_fields", None)          # NamedTuple
    if fields:
        print(f"{pad}  NamedTuple fields: {list(fields)}")
        return
    try:
        sig = inspect.signature(tp)
        print(f"{pad}  __init__{sig}")
    except (TypeError, ValueError):
        print(f"{pad}  <no introspectable constructor>")
    named = [n for n in vars(tp) if not n.startswith("_")][:20]
    if named:
        print(f"{pad}  attrs: {named}")


def main() -> int:
    print(f"executable {sys.executable}")
    for path, attr in ENTRIES:
        try:
            mod = importlib.import_module(path)
        except Exception:  # noqa: BLE001
            continue
        fn = getattr(mod, attr, None)
        if fn is None:
            continue
        print(f"\n===== {path}.{attr} =====")
        try:
            hints = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            print("  <no signature>")
            continue
        # Under `from __future__ import annotations` every annotation is a
        # STRING, so isinstance(ann, type) is False and nothing gets expanded.
        # Resolve through the defining module's globals, which is where those
        # names are guaranteed to be bound.
        g = getattr(fn, "__globals__", {})
        for pname, p in hints.items():
            ann = p.annotation
            if ann is inspect.Parameter.empty:
                continue
            print(f"  param {pname}: {ann}")
            candidates = []
            if isinstance(ann, str):
                base = ann.replace("Optional[", "").replace("]", "").strip()
                for piece in base.split("|"):
                    resolved = g.get(piece.strip())
                    if isinstance(resolved, type):
                        candidates.append(resolved)
            else:
                candidates = [tp for tp in (getattr(ann, "__args__", None) or [ann])
                              if isinstance(tp, type)]
            for tp in candidates:
                describe_type(tp)

        print("\n  --- source of fused_experts (weight layout lives here) ---")
        try:
            src = inspect.getsource(fn).splitlines()
        except (OSError, TypeError):
            src = []
        if not src:
            print("  <source unavailable>")
        for line in src[:70]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
