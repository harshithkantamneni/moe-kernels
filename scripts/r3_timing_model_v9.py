#!/usr/bin/env python
"""MODEL v9: v8 with a physical non-GEMM term on the 64x128x64 prefill tile (2026-11-20).

WHY. On the 64x128x64 prefill tile v8 prices the work outside the two GEMMs as T0 + s + other (+ b, 0 on native):
T0 = 43.23 us and s (the block-scan offset, -7.07 us) are fitted intercepts of the base model's fit on Mixtral-8x7B's
2026-09-27 CAL pages. On those pages the fit matched the calls by under-pricing the GEMMs by a near-constant 42 to 56 us
per call and absorbing it in T0 (event-timed call minus ncu's GEMM time leaves 18 to 23 us outside the GEMMs). This
tile's GEMM terms were corrected later (the sag factor, the K1 knee), so T0's compensation became an excess: in the
2026-11-20 serving check v8 charged 41 to 47 us per call outside the GEMMs against 16 to 20 us measured (+4.75% ERNIE,
+8.72% Qwen3 on the tile). Diagnosis: moe-kernels-work/2026-11-20/bias64 (parts.py, t0fit.py), paper-v2 section 8.

WHAT CHANGES (switch NONGEMM; nothing else). On a 64x128x64 prefill call only, v8's T0 + s + b is replaced by

    align(path, numel) + K0_ACT + K0_SUM,   align = A0 on vLLM's block-scan path, AS + AS_1K x numel / 1000 on its
                                            small-batch path (v8's own path label for the call)

the measured costs of the call's non-GEMM kernels (alignment, act_and_mul, moe_sum)
on the published launch-floor cell pages (scripts/v9_nongemm_fit.py; no serving data). v8's `other` (the activation
and reduction bytes over the fitted bw) is kept. Every other tile, every decode call, every refusal and every
registered prediction is v8's (and so v7's) bit for bit.

KNOWN ASSUMPTIONS.
  - Block-scan alignment is a constant A0 (its 256-id cell value). On the launch-floor pages it grows with ids at a slope that
    rises with E, but those pages flush L2 before every call, so alignment reads its ids cold there; that slope is not
    carried. A model with many experts at large prefill T may run alignment above A0.
  - The form (fixed kernel costs instead of a refit T0) was chosen after the 2026-11-20 serving check had been scored;
    the constants come only from cell pages. v9's blind test is a model neither predictor has seen.
  - The target is the call's kernel-sum (what serving runs under CUDA graphs). An eager event-timed cell also pays the
    host launch gaps between the call's kernels, which v9 does not price on this tile.
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

#: scripts/v9_nongemm_fit.py on the published launch-floor pages (2026-11-20); tests hold the fit to these values
A0_US = 5.088                 # block-scan path: alignment's two launches, a constant
AS_US = 2.861                 # small-batch path: alignment's one launch, AS_US + AS_US_PER_1K_IDS x numel / 1000
AS_US_PER_1K_IDS = 17.035
K0_ACT_US = 3.222
K0_SUM_US = 2.135
TILE_64 = "64x128x64"


@dataclass(frozen=True)
class Switches(V8.Switches):
    NONGEMM: bool = True  # v9: 64x128x64 prefill's T0 + s + b -> align(path, numel) + K0_ACT + K0_SUM

    def v8(self) -> V8.Switches:
        return V8.Switches(**{f.name: getattr(self, f.name) for f in dataclasses.fields(V8.Switches)})


V9 = Switches()


def nongemm_ms(path: str, numel: int) -> float:
    if path == "block-scan":
        al = A0_US
    elif path == "small-batch":
        al = AS_US + AS_US_PER_1K_IDS * numel / 1e3
    else:
        raise V8.Refused(f"v9: no alignment constant for path {path!r}")
    return (al + K0_ACT_US + K0_SUM_US) * 1e-3


class _Capture:
    """Records v8's own price parts (v5_parts' output) for the call being priced, by wrapping every binding of
    v8's v5_parts copy while v8's swaps are installed; restored on exit."""

    def __init__(self):
        self.parts: list[dict] = []

    def __enter__(self):
        self._cm = V8.installed()
        self._cm.__enter__()
        self._plan = [(h, a, f, c) for h, a, f, c in V8._plan() if a == "v5_parts"]
        if not self._plan:
            self._cm.__exit__(None, None, None)
            raise V8.Refused("v9: no v5_parts binding in v8's plan")

        def wrap(copy):
            def f(*args, **kw):
                out = copy(*args, **kw)
                if kw.get("sag") != "none":
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


def price(call: dict, switches: Switches = V9) -> dict:
    """v9's time of one fused-MoE call: v8's answer, with the 64x128x64 prefill non-GEMM term replaced (NONGEMM)."""
    if not switches.NONGEMM or call.get("tile") == V8.V7.R3_TILE:
        return V8.price(call, switches.v8())
    with _Capture() as cap:
        ans = V8.price(call, switches.v8())
    if ans.get("ms") is None or ans.get("regime") != "prefill" or not str(ans.get("tile", "")).startswith(TILE_64):
        return ans
    if not cap.parts:
        raise V8.Refused("v9: a 64x128x64 prefill call priced without v5_parts")
    p = cap.parts[-1]
    old = float(p["T0"]) + float(p["s"]) + float(p.get("b", 0.0))
    if abs(float(p["ms_sum"]) - float(ans["ms"])) > 1e-12:
        raise V8.Refused(f"v9: captured parts {p['ms_sum']!r} are not the priced call's {ans['ms']!r}")
    new = nongemm_ms(str(p["path"]), int(ans["numel"]))
    out = dict(ans, ms=float(ans["ms"]) - old + new)
    out["terms"] = dict(ans.get("terms") or {}, NONGEMM={"v8_ms": float(ans["ms"]), "removed_T0_s_b_ms": old,
                                                          "added_ms": new, "path": str(p["path"]), "A0_us": A0_US,
                                                          "AS_us": AS_US, "AS_us_per_1k_ids": AS_US_PER_1K_IDS, "K0_ACT_us": K0_ACT_US,
                                                          "K0_SUM_us": K0_SUM_US})
    return out


if __name__ == "__main__":
    print(__doc__)
