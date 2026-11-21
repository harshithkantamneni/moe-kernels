#!/usr/bin/env python
"""MODEL v10: v9 with the same physical non-GEMM term on the 128x128x64 prefill tile (2026-11-20).

WHY. v9 replaced the fitted intercepts T0 + s + b on the 64x128x64 prefill tile with the measured costs of the call's
non-GEMM kernels. The 128x128x64 prefill tile carries the same intercepts (common.price_parts: ms_sum = T0 + other +
the GEMM terms + s + b, priced on the board's sag par). A per-kernel split of the served calls (paper-v2 7.5, Table 12;
moe-kernels-work/2026-11-20/bias64/split.py) puts v8's non-GEMM charge on this tile at +3.0% to +4.1% of the call
above the measured alignment + act_and_mul + moe_sum on all four models split (ERNIE, Qwen3, Trinity, Ling).

WHAT CHANGES (switch NONGEMM128; nothing else). On a 128x128x64 prefill call only, v8's T0 + s + b is replaced by
v9's nongemm_ms(path, numel) with v9's constants unchanged (no new fitted value; the non-GEMM kernels do not depend on
the GEMM tile). v8's `other` is kept. The 64x128x64 prefill tile is v9's; every decode call, every refusal and every
registered prediction is v8's (and so v7's) bit for bit.

KNOWN ASSUMPTIONS AND WHAT THIS FORM WAS CHOSEN FROM.
  - The form was chosen after the v8 serving check (ERNIE, Qwen3) and v9's blind check (GLM-4.7-Flash, Trinity-Mini,
    Ling-mini-2.0) had been scored; those five models are v10's development data. v10's prefill is tested only on
    models captured after this file was frozen.
  - On ERNIE the 128x128 intercept excess cancelled a GEMM under-prediction (w1 -2.3%, w2 -1.0% of the call); with it
    removed ERNIE's 128x128 calls read low. That GEMM term is not changed here.
  - v9's block-scan alignment assumption (constant A0) applies on this tile too.
"""
from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_timing_model_v8 as V8  # noqa: E402
import r3_timing_model_v9 as V9  # noqa: E402

TILE_128 = "128x128x64"


@dataclass(frozen=True)
class Switches(V9.Switches):
    NONGEMM128: bool = True  # v10: 128x128x64 prefill's T0 + s + b -> v9's align(path, numel) + K0_ACT + K0_SUM

    def v9(self) -> V9.Switches:
        return V9.Switches(**{f.name: getattr(self, f.name) for f in dataclasses.fields(V9.Switches)})


V10 = Switches()


class _Capture:
    """Records every output of v8's price_parts copy (all its bindings) for the call being priced; restored on exit."""

    def __init__(self):
        self.parts: list[dict] = []

    def __enter__(self):
        self._cm = V8.installed()
        self._cm.__enter__()
        self._plan = [(h, a, f, c) for h, a, f, c in V8._plan() if a == "price_parts"]
        if not self._plan:
            self._cm.__exit__(None, None, None)
            raise V8.Refused("v10: no price_parts binding in v8's plan")

        def wrap(copy):
            def f(*args, **kw):
                out = copy(*args, **kw)
                self.parts.append(out)
                return out
            return f

        for h, a, _f, c in self._plan:
            V8._set(h, a, wrap(c))
        return self

    def __exit__(self, *exc):
        for h, a, _f, c in self._plan:
            V8._set(h, a, c)
        return self._cm.__exit__(*exc)


def price(call: dict, switches: Switches = V10) -> dict:
    """v10's time of one fused-MoE call: v9's answer, with the 128x128x64 prefill non-GEMM term replaced."""
    if not switches.NONGEMM128 or call.get("tile") == V8.V7.R3_TILE:
        return V9.price(call, switches.v9())
    with _Capture() as cap:
        ans = V8.price(call, switches.v8())
    if ans.get("ms") is None or ans.get("regime") != "prefill" or not str(ans.get("tile", "")).startswith(TILE_128):
        return V9.price(call, switches.v9())
    hit = [p for p in cap.parts if isinstance(p, dict) and abs(float(p.get("ms_sum", float("nan"))) - float(ans["ms"]))
           <= 1e-12]
    if not hit:
        raise V8.Refused("v10: no captured price_parts output is the priced 128x128 call's ms")
    p = hit[-1]
    old = float(p["T0"]) + float(p["s"]) + float(p.get("b") or 0.0)
    new = V9.nongemm_ms(str(p["path"]), int(ans["numel"]))
    out = dict(ans, ms=float(ans["ms"]) - old + new)
    out["terms"] = dict(ans.get("terms") or {}, NONGEMM128={"v8_ms": float(ans["ms"]), "removed_T0_s_b_ms": old,
                                                             "added_ms": new, "path": str(p["path"])})
    return out


if __name__ == "__main__":
    print(__doc__)
