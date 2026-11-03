#!/usr/bin/env python
"""MODEL v4d' (v4dp): model v4d (`r3_timing_model_v4d`, byte-identical, imported and called) plus two fixes,
each with 0 fitted constants (registration 2026-10-29-decode-addendum3-gh200; the diagnosis of the decode CAL
STOP, diag-decode DIAGNOSIS.md sections 1, 2 and 5). With both fixes off, `price` returns v4d's own price object
unchanged, so v4d' reduces to v4d exactly.

    T = v4d(T)  with  (W) the naive rows in vLLM's launch order and the follower rule
                      (R) + (W1_SCALE - 1) x the w1 GEMM time on the 16x64x128w4s4 tile

  W  NAIVE_ORDER. vLLM launches the naive path in token-major top-k order (expert_ids = topk_ids.view(-1),
     one M-tile per (token, expert) row, a row's N-tiles inner). v4d lists the rows expert-ascending and fetches
     a repeated expert's slab once. Here the rows are the cell's real ids (r3_decode.cell_ids: realize_counts
     and the page's token-row shuffle), and a follower CTA (the same expert and N-tile as an earlier CTA, its
     leader) is SERVED (no slab read) when it launches in its leader's wave (CTA index // (SMs x the census
     occupancy) equal), or in a later wave while the GEMM has streamed under L2_CUT_BYTES (60 MiB, round 5's
     cut) of slabs since the leader ((cta - leader) x slab); otherwise it RE-FETCHES its slab. Every other
     term is v4d's (floor, partial-wave tail under CORES, rho, constants). NAIVE-NODEDUP (every row fetches)
     is order-free and stays v4d's. Constants: 0 (the occupancy is the census's, the cut round 5's).
  R  W1_RATE. The w1 GEMM's time on vLLM's 16x64x128w4s4 decode tile scaled by W1_SCALE = v4d's pooled
     marginal DRAM rate over the counters' (3577.96 / 3895.49 GB/s, 7 kept-cell pairs of one T at two A on the
     published decode CAL dn pages, both models; MEASURED on SEEN pages): its per-byte cost set to the
     counter-measured rate, applied as a time ratio on the whole w1 GEMM price (owner, 2026-11-03). Every other
     tile and w2 unchanged. Constants: 0 fitted (1 MEASURED ratio).
The diagnosis's third fix (A, an align-loop term of 0.96 us per iteration) is NOT in v4d' (owner, 2026-11-03,
review MAJOR-1: a SEEN slope that overlaps v4d's B_other and worsens the SEEN fit). The CAL2 settling unit M2
measures the align slope DESCRIPTIVELY; a later model may add it only under its own registration.
The three CAL constants (T0_GR, s_naive, s_block) and the rho selection stay v4d's registered fit
(v4d.fit_constants, dcommon.fit_and_select), on v4d''s body.
"""
from __future__ import annotations

import functools
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r3_decode as RD  # noqa: E402
import r3_timing_model_v4d as V4D  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

TM, V4 = V4D.TM, V4D.V4
#: fix R: the tile it acts on and the MEASURED ratio (v4d's pooled w1 marginal rate / the counters'), with its rates
W1_TILE = "16x64x128w4s4"
W1_COUNTER_GBPS = 3895.487673018345
W1_V4D_GBPS = 3577.9578409294168
W1_SCALE = 0.9184877841384885
#: fix W: round 5's L2 cut (SEEN tile pages) and the decode pages' token-row shuffle seed
L2_CUT_BYTES = 60 * 2 ** 20
PAGE_SEED = 20261014


@dataclass(frozen=True)
class Fixes:
    """Which of v4d''s two fixes are on. ALL is v4d'; NONE is v4d exactly."""
    naive_order: bool = True
    w1_rate: bool = True

    def any(self) -> bool:
        return self.naive_order or self.w1_rate

    def as_dict(self) -> dict:
        return asdict(self)


ALL = Fixes()
NONE = Fixes(False, False)


@functools.lru_cache(maxsize=4096)
def _ids(model: str, T: int, counts: tuple, shuffle: bool, seed: int) -> tuple:
    cfg = MODEL_CONFIGS[model]
    c = RD.DecodeCell(label="v4dp", T=int(T), counts=tuple(int(v) for v in counts), arms=(RD.NATIVE,), shuffle=bool(shuffle))
    return tuple(int(x) for x in RD.cell_ids(cfg, c, int(seed)).reshape(-1).tolist())


def ids_of(model: str, cell: dict, seed: int = PAGE_SEED) -> tuple:
    """A page cell's NATIVE ids, token-major flat (r3_decode.cell_ids, CPU): the naive path's launch order."""
    return _ids(model, int(cell["T"]), tuple(int(v) for v in cell["counts"]), bool(cell.get("shuffle", True)), int(seed))


def follower_reads(ids, gemm: str, slots: int, *, cut_bytes: int = L2_CUT_BYTES) -> tuple[np.ndarray, list[dict]]:
    """Per-CTA reads of one naive GEMM in launch order (row-major over ids, N-tiles inner) under fix W's rule,
    and one record per follower CTA (its leader, wave and whether it fetches)."""
    geo = TM.GEOMETRY[gemm]
    act = geo.arow / geo.npn / TM.BLOCK_M
    first, out, rec = {}, [], []
    for row, e in enumerate(ids):
        for n in range(geo.npn):
            cta = row * geo.npn + n
            key = (int(e), n)
            if key not in first:
                first[key] = cta
                out.append(geo.slab + act)
                continue
            lead = first[key]
            same_wave = lead // slots == cta // slots
            fetch = (not same_wave) and (cta - lead) * geo.slab >= cut_bytes
            out.append((geo.slab if fetch else 0.0) + act)
            rec.append({"cta": cta, "leader": lead, "same_wave": same_wave, "fetch": fetch})
    return np.asarray(out, dtype=float), rec


def price(model: str, T: int, counts, prm: V4D.DecodeParams, *, ids=None, fixes: Fixes = ALL,
          declared: int | None = None, with_constants: bool = True) -> dict:
    """v4d''s time of one decode cell (ms), its parts and its bytes. `ids` (the cell's token-major ids) is
    needed on the naive path with fix W on, else that cell refuses (TM.Refused). With every fix off this is
    V4D.price's own return value."""
    if not fixes.any():
        return V4D.price(model, T, counts, prm, declared=declared, with_constants=with_constants)
    cfg = MODEL_CONFIGS[model]
    geo = RD.geometry(cfg, int(T), counts, declared)
    naive_w = fixes.naive_order and geo.path == RD.NAIVE and prm.naive_dedup
    if naive_w and ids is None:
        raise TM.Refused(f"v4d' fix W: the naive cell {model} T {T} needs its ids (the launch order)")
    if naive_w:
        t = V4D.tile_of(RD.default_config(int(T), cfg.num_experts))
        flat = tuple(int(x) for x in ids)
        if len(flat) != geo.numel:
            raise TM.Refused(f"v4d' fix W: {len(flat)} ids for numel {geo.numel}")
        orig, follow = V4D._naive_reads, {}

        def reads(cfg_, counts_, gemm):
            r, rec = follower_reads(flat, gemm, prm.sms * prm.tile.occ(t, gemm))
            follow[gemm] = rec
            return r
        V4D._naive_reads = reads
        try:
            p = V4D.price(model, T, counts, prm, declared=declared, with_constants=with_constants)
        finally:
            V4D._naive_reads = orig
    else:
        follow = None
        p = V4D.price(model, T, counts, prm, declared=declared, with_constants=with_constants)
    extra = 0.0
    fx = {"fixes": fixes.as_dict(), "naive_order": None, "w1_rate_ms": 0.0}
    if follow is not None:
        fx["naive_order"] = {g: {"followers": len(v), "fetch": sum(x["fetch"] for x in v),
                                 "same_wave": sum(x["same_wave"] for x in v)} for g, v in follow.items()}
    if fixes.w1_rate and p["tile"] == W1_TILE:
        g1 = p["gemm"]["w1"]
        d = (W1_SCALE - 1.0) * g1["gemm_ms"]
        g1["gemm_ms_v4d"] = g1["gemm_ms"]
        g1["gemm_ms"] = g1["gemm_ms"] + d
        fx["w1_rate_ms"] = d
        extra += d
    p["body_ms"] = p["body_ms"] + extra
    if p["ms"] is not None:
        p["ms"] = p["body_ms"] + p["const_ms"]
    p["v4dp"] = fx
    return p


def describe() -> dict:
    """The registered constants of v4d' (the registration states them and their sources)."""
    return {"W": {"rule": "naive rows in token-major launch order; a follower CTA is served in its leader's wave, "
                          "or in a later wave while under the L2 cut has streamed since its leader; else it re-fetches",
                  "L2_cut_bytes": L2_CUT_BYTES, "slots": "SMs x the census occupancy of the GEMM's tile",
                  "page_seed": PAGE_SEED, "fitted": 0},
            "R": {"tile": W1_TILE, "counter_GBps": W1_COUNTER_GBPS, "v4d_GBps": W1_V4D_GBPS, "scale": W1_SCALE, "fitted": 0,
                  "form": "a time ratio on the whole w1 GEMM price (tail included)"}}


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--counts", required=True, help="comma-separated E counts")
    ap.add_argument("--rho", default="fixed", choices=V4D.RHO_FORMS)
    ap.add_argument("--no-shuffle", action="store_true", help="the cell's shuffle flag false (naive ids)")
    a = ap.parse_args(argv)
    counts = [int(x) for x in a.counts.split(",")]
    cell = {"T": a.T, "counts": counts, "shuffle": not a.no_shuffle}
    p = price(a.model, a.T, counts, V4D.DecodeParams(rho_form=a.rho), ids=ids_of(a.model, cell))
    print(json.dumps({k: v for k, v in p.items() if k != "gemm"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
