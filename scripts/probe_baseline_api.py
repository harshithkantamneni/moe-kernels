#!/usr/bin/env python
"""What MoE entry points does the framework in THIS venv actually expose?

Run once per framework venv. Writing a baseline span against a remembered API
produces a span that imports cleanly and benchmarks the wrong thing, which is
the one failure the correctness gate cannot catch: a fused_moe that renormalises
when the model does not, or takes weights in the other layout, is still
internally consistent and still wrong.

    /workspace/venvs/vllm/bin/python   scripts/probe_baseline_api.py
    /workspace/venvs/sglang/bin/python scripts/probe_baseline_api.py

Prints, for every candidate it finds: the import path that worked, the version,
the signature, and which parameters have defaults. That is enough to write the
span, and the span can then declare `env` correctly and be validated against the
same fp32 oracle as everything else.
"""
from __future__ import annotations

import importlib
import inspect
import sys

#: Import path -> attribute, for the shapes these libraries have used. Both a
#: whole-layer entry (takes router logits) and a post-routing entry (takes
#: topk_ids) are worth finding: the second maps onto a contiguous span of
#: permute..unpermute, the first also swallows the router.
CANDIDATES: list[tuple[str, str]] = [
    # vLLM
    ("vllm.model_executor.layers.fused_moe", "fused_moe"),
    ("vllm.model_executor.layers.fused_moe", "fused_experts"),
    ("vllm.model_executor.layers.fused_moe.fused_moe", "fused_moe"),
    ("vllm.model_executor.layers.fused_moe.fused_moe", "fused_experts"),
    ("vllm.model_executor.layers.fused_moe.fused_moe", "invoke_fused_moe_kernel"),
    ("vllm.model_executor.layers.fused_moe.fused_moe", "moe_align_block_size"),
    ("vllm._custom_ops", "moe_align_block_size"),
    # SGLang
    ("sglang.srt.layers.moe.fused_moe_triton", "fused_moe"),
    ("sglang.srt.layers.moe.fused_moe_triton", "fused_experts"),
    ("sglang.srt.layers.moe.fused_moe_triton.fused_moe", "fused_moe"),
    ("sglang.srt.layers.moe.fused_moe_triton.fused_moe", "fused_experts"),
    ("sglang.srt.layers.moe.topk", "select_experts"),
    # torch, for comparison in the same report
    ("torch.nn.functional", "grouped_mm"),
]


def version_of(root: str) -> str:
    try:
        mod = importlib.import_module(root)
    except Exception as e:  # noqa: BLE001
        return f"<not importable: {type(e).__name__}>"
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    try:
        from importlib.metadata import version
        return version(root)
    except Exception:  # noqa: BLE001
        return "<no version attribute>"


def main() -> int:
    print(f"python   {sys.version.split()[0]}")
    print(f"executable {sys.executable}")
    try:
        import torch
        print(f"torch    {torch.__version__}")
    except ImportError:
        print("torch    <not installed>")

    roots = sorted({path.split(".")[0] for path, _ in CANDIDATES})
    print("\nframeworks in this venv:")
    for root in roots:
        print(f"  {root:10s} {version_of(root)}")

    print("\nentry points:")
    found = 0
    for path, attr in CANDIDATES:
        try:
            mod = importlib.import_module(path)
        except Exception:  # noqa: BLE001
            continue
        fn = getattr(mod, attr, None)
        if fn is None:
            continue
        found += 1
        print(f"\n  FOUND  {path}.{attr}")
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            print("         <no introspectable signature>")
            continue
        for name, p in sig.parameters.items():
            default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
            ann = "" if p.annotation is inspect.Parameter.empty else f": {p.annotation}"
            print(f"         {name}{ann}{default}")
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        if doc:
            print("         --- docstring head ---")
            for line in doc[:12]:
                print(f"         {line}")
    if not found:
        print("  none. Either this venv has no framework, or the import paths moved;")
        print("  try `pip list | grep -Ei 'vllm|sglang'` and say what it prints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
