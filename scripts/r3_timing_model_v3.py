"""MODEL v3's timing term (rental 6, 2026-10-09; docs/registered/2026-10-09-rental6-v3-gh200).

A separate module over `r3_timing_model` (which stays byte-identical to the sha rental 5's
registration pins): v3's time of a cell is `r3_timing_model.call_ms` (model v2 under
`dead_model("v2")`) plus term (b). On every non-NATIVE cell whose NATIVE twin (same model, n,
declaration E) takes the block-scan alignment kernel, (b) is the page's own measured alignment
growth A_arm(n) = median align(arm, n) - median align(native, n) from its `align_probe` (0 fitted
parameters), and on SHARED cells also the per-call constant V3_H_MS; elsewhere it is 0.

V3_H_MS is SEEN-FITTED: the S - N gap estimator over 178 block-scan cells in 27 (session, G, D)
groups of 8x7B, 8x22B, Qwen2-57B and OLMoE timed pages, at least 143 of them held-out timing
pages of earlier registrations, so those models are IN-SAMPLE for (b). The block-scan condition
is SEEN-INFORMED (the rental-5 Mixtral n 2 pages). Term (a), MODEL M4, is the byte model's
(`wave_split_bytes_v3`).
"""
from __future__ import annotations

import r3_timing_model as TM

V3_H_MS = -0.00516
BLOCK_SCAN, SMALL_BATCH = TM.BLOCK_SCAN, TM.SMALL_BATCH


def v3_b_ms(arm: str, native_path: str | None, align_ms: float | None, *, key: str = "") -> float:
    """Term (b) for one cell, ms: 0 on NATIVE and wherever NATIVE's own path is not block-scan;
    else A_arm(n) (`align_ms`) plus V3_H_MS on SHARED. A block-scan non-NATIVE cell with no A is
    refused (v3 prices it from its page's probe or the registered proxy)."""
    if arm == "native" or native_path != BLOCK_SCAN:
        return 0.0
    if align_ms is None:
        raise TM.Refused(f"{key or arm}: model v3 prices a block-scan {arm} cell from its page's "
                         "align_probe and it has none")
    return float(align_ms) + (V3_H_MS if arm == "shared" else 0.0)


def call_ms(x, cell, ctx, k_w: float, *, native_path: str | None, align_ms: float | None, **kw) -> float:
    """v3's T for one cell: r3_timing_model.call_ms (call it under dead_model("v2")) plus (b)."""
    return TM.call_ms(x, cell, ctx, k_w, **kw) + v3_b_ms(cell.arm, native_path, align_ms, key=cell.key)
