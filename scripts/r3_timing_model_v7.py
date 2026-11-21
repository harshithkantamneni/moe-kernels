#!/usr/bin/env python
"""MODEL v7: one entry point, `price(cell)`, over three regime branches that share v4's trunk (design-v7, owner
decisions of 2026-11-17). v7 is a declared REGIME SWITCH, not one physical law: it dispatches on the tile key, and the
re-read term on GROUP_SIZE_M, each a domain of measured support.

    regime    tiles (BMxBNxBK w s)                      priced by                       new in v7, each its own switch
    r3        32x64x64 w8 s4 (R3's test tile)           v3 = v4 at R3's tile, exactly   nothing (both terms out of domain)
    prefill   128x128x64 w8 s3, 64x128x64 w8 s3         round 5 as B3 froze it          RR: w1 reads at G 1
    decode    16x64x128 w4 s4, 32x64x128 w4 s3,         v6d (addendum 6)                D6FIX: the D6-24 rho fix
              32x128x64 w4 s3, 64x128x64 w4 s3

A tile outside these keys is refused (Refused), as v4 refuses a tile it has no constant for. 64x128x64 w4 s3 (decode)
and 64x128x64 w8 s3 (prefill) are different keys and different branches. With every switch off (`OFF`) each branch is
its frozen predecessor exactly: v3 (r3_timing_model_v4 --check), round 5 (scripts/round5_chain.py, B3's committed
predictions and round 5's table) and v6d (gate_d6's CAL2 and re-score rows).

RR, THE w1 RE-READ TERM (prefill tiles, GROUP_SIZE_M 1 only; RR's step rule, design-v7 section 1.5). At G 1 the pids
run N-tile fastest, so M-tile j + 1 of an expert reads the w1 slab M-tile j read exactly P pids earlier (P = N-tiles
per M-tile). The w1 DRAM reads of the cell become

    Bytes_w1 = C_u + mhat R
    C_u  = (experts with a row) x W_e + (unique tokens) x H x 2      each active expert's w1 once, A once
    R    = sum_e (m_e - 1) x W_e,  m_e = ceil(c_e / BLOCK_M)          the re-read demand
    mhat = the share of re-read pairs (r, r - P) that miss L2

and mhat comes from the launch layer L-SCHED: the live CTAs in pid order onto W_c slots (264 at 128x128, 396 at
64x128), the first W_c start at 0, CTA i starts when the earliest slot frees; life l_i = max(0.05, 1 + sigma eps_i) in
units of the median life, eps_i ~ N(0, 1), numpy default_rng(seed 7), 6 replicas; per pair the F2 distance
d = ((s_r + l_r / 2) - (s_{r-P} + l_{r-P} / 2)) x W_c x slab (bytes streamed GPU-wide, every slot streaming its slab
over one median life), slab = 128 x K x 2, and a pair misses iff d > C (strict). C 47.5 MiB (RR's DESCRIPTIVE refit
on its own fresh pages), sigma 0.08 (128x128) and 0.06 (64x128) fitted on RR's 57 fresh cells' counted m at C fixed.
Every one of these constants was chosen after pricing variants on SEEN pages (design-v7 section 3).

`miss_fraction` is the same step rule read on measured stamps (start and end per CTA, each CTA pulling its slab
evenly over its life): RR's reader's distance (read_rr.pair_read's integral), the owner's piece of the design. The
launch layer above uses the design's byte clock (dt x W_c x slab), the one its sigma was fitted with; the two agree
while every slot is busy (tests/test_timing_model_v7.py).

D6FIX (decode). v6d prices a naive cell with a non-fetching follower by v4d''s naive GEMMs (D6-24) at the one CAL2
fit, whose rho form the law-priced cells select (fixed). With D6FIX on, such a cell (and only such a cell) is priced
at the CAL2 fit v4d' selects with v4d' on every naive cell (the fit and rho form every earlier decode CAL selected:
scaled); every other cell keeps v6d's fit. The CAL fit itself is unchanged (it prices bodies with no constants). Found
on SEEN pages (the D6-24 defect of record, addendum 6 'after the gate').

RR at decode is not built: the design counts 0 aligned w1 re-reads on CAL2's 91 kept cells and on the 220 re-score
cells, so it would be inert on every gate page (design-v7 section 0, item 3); the decode branch is v6d's law.

    python scripts/r3_timing_model_v7.py --check-r3          # v7 at R3's tile against every registered v3 prediction
"""
from __future__ import annotations

import argparse
import contextlib
import heapq
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MIB = 2 ** 20


class Refused(RuntimeError):
    """A cell outside v7's declared domains."""


# ================================================================== switches

@dataclass(frozen=True)
class Switches:
    """Each new term behind its own switch (owner rule 2026-10-14: profile each feature on its own)."""
    RR: bool = True       # prefill: w1 reads at GROUP_SIZE_M 1 from the re-read term
    D6FIX: bool = True    # decode: v4d'-branch naive cells at v4d''s own CAL2 fit

    def as_dict(self) -> dict:
        return {"RR": self.RR, "D6FIX": self.D6FIX}


V7 = Switches()
OFF = Switches(RR=False, D6FIX=False)
#: the ablation rungs: every switch off, each alone, all on
RUNGS = {"off": OFF, "RR alone": Switches(RR=True, D6FIX=False), "D6FIX alone": Switches(RR=False, D6FIX=True),
         "v7 (all)": V7}

# ================================================================== the regime switch

R3_TILE = "32x64x64w8s4"
PREFILL_TILES = ("128x128x64w8s3", "64x128x64w8s3")
DECODE_TILES = ("16x64x128w4s4", "32x64x128w4s3", "32x128x64w4s3", "64x128x64w4s3")
REGIMES = {"r3": (R3_TILE,), "prefill": PREFILL_TILES, "decode": DECODE_TILES}


def regime_of(tile: str) -> str:
    """The branch a tile key belongs to; Refused outside the declared keys."""
    for k, tiles in REGIMES.items():
        if tile in tiles:
            return k
    raise Refused(f"tile {tile}: v7 has no branch for it (declared: {REGIMES})")


def rr_applies(tile: str, G: int) -> bool:
    """The RR term's domain: a prefill tile at GROUP_SIZE_M 1, the only place its pairs were measured."""
    return tile in PREFILL_TILES and int(G) == 1


# ================================================================== the RR term: constants

#: (form, C MiB): F2, between mid-lives, RR's refit on its fresh pages (rr.read.txt refit, DESCRIPTIVE there)
RR_FORM = "F2"
RR_C_MIB = 47.5
#: L-SCHED's life spread per BLOCK_M, fitted on RR's 57 fresh cells at C fixed (design-v7 work/rr_fit.py variant c)
RR_SIGMA = {128: 0.08, 64: 0.06}
RR_REPS = 6
RR_SEED = 7
RR_LIFE_MIN = 0.05
#: W_c: live CTA slots (132 SMs x CTAs per SM: 2 at 128x128x64 w8 s3, 3 at 64x128x64 w8 s3; census, MEASURED)
RR_WC = {128: 264, 64: 396}
RR_BLOCK_N = 128


def rr_constants() -> dict:
    return {"form": RR_FORM, "C_mib": RR_C_MIB, "sigma": {str(k): v for k, v in RR_SIGMA.items()}, "reps": RR_REPS,
            "seed": RR_SEED, "life_min": RR_LIFE_MIN, "W_c": {str(k): v for k, v in RR_WC.items()},
            "block_n": RR_BLOCK_N}


def constants() -> dict:
    """Everything v7 adds by value: the regime keys, the switches as registered, the RR constants and D6FIX's rule.
    The registration pins its sha256 (constants_sha256)."""
    return {"regimes": {k: list(v) for k, v in REGIMES.items()}, "rr_domain": "prefill tiles at GROUP_SIZE_M 1",
            "switches_registered": V7.as_dict(), "RR": rr_constants(),
            "D6FIX": ("a decode cell v6d prices by v4d''s naive GEMMs (a naive cell with a non-fetching follower, D6-24) "
                      "is priced at the CAL2 fit v4d' selects with v4d' on every naive cell (its constants and rho "
                      "form); every other cell at v6d's CAL2 fit")}


def constants_sha256() -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(constants(), sort_keys=True).encode()).hexdigest()


# ================================================================== the RR term: geometry and the launch layer

def m_tile_experts(counts, bm: int) -> np.ndarray:
    """The expert of every M-tile in launch order (each expert with a row contributes ceil(c / BLOCK_M) M-tiles)."""
    ex = []
    for e, c in enumerate(counts):
        if int(c):
            ex += [e] * (-(-int(c) // bm))
    return np.array(ex, dtype=np.int64)


def reread_pairs(counts, bm: int, P: int) -> np.ndarray:
    """The re-reading CTAs r (pid = M-tile x P + N-tile at G 1) whose partner r - P read the same slab: every N-tile
    of an M-tile whose previous M-tile holds the same expert."""
    ex = m_tile_experts(counts, bm)
    mm = np.nonzero(ex[1:] == ex[:-1])[0] + 1
    return (mm[:, None] * P + np.arange(P)[None, :]).reshape(-1)


def lsched(live: int, W_c: int, sigma: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """L-SCHED, one replica: (start, life) per CTA in pid order, in units of the median life. The first W_c CTAs start
    at 0; CTA i starts when the earliest slot frees (list scheduling); life = max(0.05, 1 + sigma eps)."""
    life = np.maximum(RR_LIFE_MIN, 1 + sigma * rng.standard_normal(live))
    s = np.empty(live)
    h = [float(x) for x in life[:W_c]]
    heapq.heapify(h)
    s[:min(W_c, live)] = 0.0
    for i in range(W_c, live):
        t = heapq.heappop(h)
        s[i] = t
        heapq.heappush(h, t + life[i])
    return s, life


def rr_distances(counts, bm: int, P: int, W_c: int, slab: float, sigma: float, reps: int = RR_REPS,
                 seed: int = RR_SEED) -> list:
    """Per replica (d_start, d_mid) in MiB over every re-read pair: the launch layer's starts (F1) and mid-lives (F2)
    apart, times W_c x slab (design-v7 section 1.5; work/rr_fit.py's dists, the same draws in the same order)."""
    r = reread_pairs(counts, bm, P)
    if not r.size:
        return []
    live = int(m_tile_experts(counts, bm).size) * P
    unit = W_c * slab / MIB
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        s, life = lsched(live, W_c, sigma, rng)
        mid = s + life / 2
        out.append(((s[r] - s[r - P]) * unit, (mid[r] - mid[r - P]) * unit))
    return out


def rr_mhat(counts, bm: int, P: int, W_c: int, slab: float, sigma: float, C_mib: float = RR_C_MIB,
            form: str = RR_FORM) -> float:
    """The share of re-read pairs that miss (d > C, strict), averaged over the replicas; 0 with no pair."""
    k = {"F1": 0, "F2": 1}[form]
    reps = rr_distances(counts, bm, P, W_c, slab, sigma)
    if not reps:
        return 0.0
    return float(np.mean([(x[k] > C_mib).mean() for x in reps]))


def cell_counts(model: str, counts, n: int, bm: int) -> list[int]:
    """A page cell's per-expert rows: its histogram, or n x BLOCK_M rows per expert for a balanced cell."""
    from moe.spec import MODEL_CONFIGS
    if counts:
        return [int(x) for x in counts]
    return [int(n) * bm] * MODEL_CONFIGS[model].num_experts


def rr_w1(model: str, counts, n: int, bm: int, *, C_mib: float = RR_C_MIB, form: str = RR_FORM,
          sigma: float | None = None) -> dict:
    """The RR term's w1 DRAM bytes of one G 1 prefill cell: C_u + mhat R, with its parts."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    H, I, k = cfg.hidden_size, cfg.intermediate_size, cfg.top_k
    c = cell_counts(model, counts, n, bm)
    N, K = 2 * I, H
    P = -(-N // RR_BLOCK_N)
    We = N * K * 2
    slab = RR_BLOCK_N * K * 2
    mt = [-(-x // bm) for x in c if x]
    Cu = len(mt) * We + (sum(c) // k) * H * 2
    R = sum(t - 1 for t in mt) * We
    sig = RR_SIGMA[bm] if sigma is None else sigma
    mh = rr_mhat(c, bm, P, RR_WC[bm], slab, sig, C_mib, form) if R else 0.0
    return {"bytes": Cu + mh * R, "C_u": Cu, "R": R, "m_hat": mh, "P": P, "W_c": RR_WC[bm], "slab": slab,
            "pairs": int(reread_pairs(c, bm, P).size), "sigma": sig, "C_mib": C_mib, "form": form}


# ================================================================== the step rule on measured stamps (owner's piece)

def _centered(start, end):
    """(start - t0, end - t0, t0, integer?) with t0 the earliest stamp. The subtraction is done in the inputs' own
    dtype BEFORE any float conversion: int64 globaltimer nanoseconds (about 1.8e18) are differenced exactly in
    integer arithmetic, and only the centered times (small) are cast to float64. A float64 cast first would put every
    stamp on a 256 ns grid (the defect of record of 2026-11-12) and collapse short lives to zero. ValueError on a
    non-positive life (end <= start), a shape mismatch or an empty launch."""
    s, e = np.asarray(start), np.asarray(end)
    if s.shape != e.shape or s.ndim != 1 or not s.size:
        raise ValueError(f"start {s.shape} and end {e.shape}: two equal non-empty 1-d arrays")
    if s.dtype.kind in "iu" and e.dtype.kind in "iu":
        s, e = s.astype(np.int64), e.astype(np.int64)
        integer = True
    elif s.dtype.kind in "iuf" and e.dtype.kind in "iuf":
        s, e = s.astype(np.float64), e.astype(np.float64)
        integer = False
    else:
        raise ValueError(f"stamps of dtype {s.dtype} / {e.dtype}: integers or floats")
    if not bool(np.all(e > s)):
        raise ValueError(f"{int(np.sum(~(e > s)))} CTA(s) with a non-positive life (end <= start)")
    t0 = min(s.min(), e.min())
    return (s - t0).astype(np.float64), (e - t0).astype(np.float64), t0, integer


def _shift(x, t0, integer: bool):
    """x - t0 in x's own dtype (integer when the stamps were integer and x is), then float64."""
    x = np.asarray(x)
    if integer and x.dtype.kind in "iu":
        return (x.astype(np.int64) - t0).astype(np.float64)
    return x.astype(np.float64) - float(t0)


def _stream(s, e, slab_bytes):
    """F on centered float times (s, e from _centered)."""
    rate = np.asarray(slab_bytes, np.float64) / (e - s)
    t = np.concatenate([s, e])
    d = np.concatenate([rate, -rate])
    o = np.argsort(t, kind="stable")
    t, d = t[o], d[o]
    val = np.cumsum(d)                                            # GPU-wide byte rate after each event
    cum = np.concatenate([[0.0], np.cumsum(val[:-1] * np.diff(t))])

    def F(x):
        j = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)
        return cum[j] + val[j] * (x - t[j])
    return F


def bytes_streamed(start, end, slab_bytes):
    """F(x): the bytes the whole GPU streamed up to time x, each CTA pulling its slab evenly over [start, end).
    Computed once by sorting every start and end event and keeping running sums (O(n log n), then O(log n) per
    query), RR's reader's way (read_pc.step_integral), not by summing every CTA per query (O(CTAs) per query, slow
    on 14,000-CTA launches). Times are centered on the earliest stamp BEFORE any float conversion (_centered: integer
    stamps, such as int64 globaltimer ns, are differenced in integer arithmetic), and so is every query x of the
    same dtype, so absolute stamps never enter a float difference (the C3-R1 lesson). ValueError on a non-positive
    life."""
    s, e, t0, integer = _centered(start, end)
    F = _stream(s, e, slab_bytes)
    return lambda x: F(_shift(x, t0, integer))


def miss_fraction(start_us, end_us, slab_bytes, pairs, P: int, C_mib: float = 47.5, form: str = "F2") -> float:
    """The share of re-read pairs whose partner distance exceeds C: for each reader r in `pairs` (its partner r - P),
    d = the bytes the whole GPU streamed between the partner's moment and the reader's (F1: the two starts; F2: the
    two mid-lives), each CTA pulling its slab evenly over its life; a miss iff d > C_mib MiB (strict: d == C is a
    hit, as read_rr.law_miss). The stamps may be floats in any unit or integers (int64 globaltimer ns); they are
    centered on the earliest stamp before any float conversion, in integer arithmetic when they are integers
    (_centered), and the moments are taken on the centered times. 0.0 with no pair; ValueError on an unknown form or
    a non-positive life (end <= start)."""
    if form not in ("F1", "F2"):
        raise ValueError(f"form {form!r}: F1 (starts) or F2 (mid-lives)")
    r = np.asarray(pairs, dtype=np.int64)
    s, e, _t0, _int = _centered(start_us, end_us)
    if not r.size:
        return 0.0
    F = _stream(s, e, slab_bytes)
    x = s if form == "F1" else s + (e - s) / 2
    d = F(x[r]) - F(x[r - int(P)])
    return float(np.mean(d > C_mib * MIB))


# ================================================================== branch r3: v3 exactly (v4 at R3's tile)

_R3: dict = {}


def r3_pricer():
    if not _R3:
        sys.path.insert(0, str(REPO / "scripts" / "scoring" / "rental6"))
        import r3_timing_model_v4 as V4
        import reg6 as R6
        import wave_split_bytes_v4 as WV4
        _R3.update(V4=V4, WV4=WV4, pr=R6.pricer(), prm=V4.Params())
    return _R3


def price_r3(cell: dict) -> dict:
    """v4's price at R3's tile with the v3 byte key, F-step: r3_timing_model_v4.check's call."""
    X = r3_pricer()
    got = X["WV4"].price(X["pr"], cell["model"], cell["arm"], int(cell["n"]), cell["counts"], int(cell.get("copies", 9)),
                         cell.get("seed"), X["V4"].R3_TILE, int(cell.get("G", 8)), X["prm"], key="v3",
                         A_us=cell.get("A_us"))
    return {"ms": got["ms"], "regime": "r3", "terms": {}}


# ================================================================== branch prefill: round 5 + RR

def prefill_tile_of(row: dict) -> str:
    bm = int(row["doc"]["block_m"])
    return {128: PREFILL_TILES[0], 64: PREFILL_TILES[1]}.get(bm, f"{bm}x?")


def price_prefill(row: dict, *, sag128, sag64=None, knee=None, switches: Switches = V7) -> dict:
    """One prefill cell (a B2 / CAL page row: model, page, n, label, c, doc) under round 5 at the board's 128x128 sag
    par sag128 (and 64x128 par sag64), the K1 knee (p, S0) (B3's SEEN constants by default; a fold's LOMO constants
    on round 5's held-out table), with the RR term on w1 at G 1 when switches.RR."""
    import round5_chain as R5
    C = R5.chain()
    bm, G = int(row["doc"]["block_m"]), int(row["doc"]["group_m"])
    tile = prefill_tile_of(row)
    if tile not in PREFILL_TILES:
        raise Refused(f"prefill cell at BLOCK_M {bm}: no prefill tile")
    p, s0 = tuple(knee) if knee is not None else R5.K1_SEEN
    rr = switches.RR and rr_applies(tile, G)
    term = rr_w1(row["model"], row["c"].get("counts"), int(row["n"]), bm) if rr else None
    if bm == 128:
        rd = R5.reads(row, p)
        if rd is None:
            return {"ms": None, "regime": "prefill", "why": "the STAG-CUT guard refuses the cell", "terms": {}}
        rd = {g: rd[g] for g in ("w1", "w2")}
        frozen_w1 = float(rd["w1"])
        if rr:
            rd["w1"] = term["bytes"]
        ms = R5.price_128(row, sag128, rd, p, s0)
    else:
        key = (row["model"], row.get("page"), row["n"], row["label"], bm, G)
        with C["r5"].law3():
            ms = C["k5"].cell_ms(row, None, R5.P_V5PP, tuple(sag128), tuple(sag64))
            rd = dict(C["k5"]._READS[key])
            frozen_w1 = float(rd["w1"])
            if rr:
                C["k5"]._READS[key] = dict(rd, w1=term["bytes"])
                ms = C["k5"].cell_ms(row, None, R5.P_V5PP, tuple(sag128), tuple(sag64))
                rd = dict(C["k5"]._READS[key])
    out = {"ms": float(ms), "regime": "prefill", "tile": tile, "G": G,
           "reads": {g: float(rd[g]) for g in ("w1", "w2")}, "w1_frozen": frozen_w1, "terms": {}}
    if rr:
        out["terms"]["RR"] = {k: v for k, v in term.items()}
    return out


# ================================================================== branch decode: v6d + D6FIX

_DEC: dict = {}


def decode_modules() -> dict:
    if not _DEC:
        d = REPO / "scripts" / "scoring" / "decode"
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        import d3common as D3
        import d6common as D6
        import dcommon as DC
        _DEC.update(D3=D3, D6=D6, DC=DC, V6D=D6.V6D, V5D=D6.V5D, V4DP=D6.V4DP, RD=D6.RD, TM=D6.TM, V4D=DC.V4D)
    return _DEC


def v4dp_branch(p: dict) -> bool:
    """v6d kept v4d''s naive GEMM prices on this cell (a naive cell with a non-fetching follower, D6-24)."""
    v = p.get("v6d")
    return bool(v) and not v.get("gemm") and "non-fetching follower" in str(v.get("naive"))


def price_decode_cell(model: str, cell: dict, prm, *, arm: str = "native", decl_copies: int | None = None,
                      fixes=None, terms=None, law=None, prm_fix=None, switches: Switches = V7,
                      naive_v4dp: bool = False) -> dict | None:
    """d6common.price_cell_v6d's contract (None: NOT PRICED) under v7: v6d at `terms` and `law`; with switches.D6FIX
    and a fitted prm (T0_GR set), a v4d'-branch cell is re-priced at prm_fix. naive_v4dp: every naive cell at v4d''s
    prices (the CAL2 fit prm_fix comes from), not a v7 price."""
    X = decode_modules()
    D6, V6D, V4DP, RD = X["D6"], X["V6D"], X["V4DP"], X["RD"]
    from moe.spec import MODEL_CONFIGS
    fixes = V4DP.ALL if fixes is None else fixes
    terms = D6.registered_terms() if terms is None else terms
    law = V6D.LAW if law is None else law
    cfg = MODEL_CONFIGS[model]
    declared = None if arm == "native" else cfg.num_experts * int(decl_copies or 1)
    naive = RD.geometry(cfg, int(cell["T"]), cell["counts"], declared).path == RD.NAIVE
    ids = V4DP.ids_of(model, cell) if fixes.naive_order and naive else None

    def one(prm_):
        try:
            p = V6D.price(model, int(cell["T"]), cell["counts"], prm_, ids=ids, fixes=fixes, terms=terms,
                          declared=declared, law=None if (naive_v4dp and naive) else law)
        except V4DP.TM.Refused:
            return None
        if prm_.T0_GR_ms is not None and p["ms"] is None:
            return None
        return p
    p = one(prm)
    if (p is not None and switches.D6FIX and not naive_v4dp and prm_fix is not None and prm.T0_GR_ms is not None
            and v4dp_branch(p)):
        q = one(prm_fix)
        if q is not None:
            assert v4dp_branch(q), "the D6-24 branch moved with the CAL constants"
            q["v7"] = {"D6FIX": {"rho_form": prm_fix.rho_form, "s_naive_GR_ms": prm_fix.s_naive_GR_ms,
                                 "T0_GR_ms": prm_fix.T0_GR_ms, "v6d_ms": p["ms"]}}
        return q
    return p


@contextlib.contextmanager
def decode_scorers(*, terms=None, law=None, prm_fix=None, switches: Switches = V7, naive_v4dp: bool = False):
    """Inside this block the frozen decode scorers' cell price (d3common.price_cell_v4dp, entered by d3common.v4dp)
    is v7's decode price (gate_d6 uses d6common.v6d the same way)."""
    X = decode_modules()
    D3, V4DP = X["D3"], X["V4DP"]
    orig = D3.price_cell_v4dp

    def price_cell_v4dp(model, cell, prm, arm="native", decl_copies=None, fixes=V4DP.ALL):
        return price_decode_cell(model, cell, prm, arm=arm, decl_copies=decl_copies, fixes=fixes, terms=terms,
                                 law=law, prm_fix=prm_fix, switches=switches, naive_v4dp=naive_v4dp)
    D3.price_cell_v4dp = price_cell_v4dp
    try:
        yield
    finally:
        D3.price_cell_v4dp = orig


# ================================================================== one entry point

def price(cell: dict, switches: Switches = V7) -> dict:
    """v7's time of one cell. cell["tile"] picks the branch (regime_of; Refused outside the declared keys):
      r3       {tile, model, arm, n, counts, copies, seed, G, A_us}
      prefill  {tile, row (a B2 / CAL page row), sag128, sag64, knee}; the tile must be the row's
      decode   {tile, model, cell (a decode page cell: T, counts, label), prm, prm_fix, terms, law, arm, decl_copies};
               the tile must be the one v6d prices the cell at (a naive cell's tile is v5d's)
    Returns {ms, regime, terms: the new terms' parts, by switch}."""
    reg = regime_of(cell["tile"])
    if reg == "r3":
        return price_r3(cell)
    if reg == "prefill":
        if prefill_tile_of(cell["row"]) != cell["tile"]:
            raise Refused(f"the row's tile is {prefill_tile_of(cell['row'])}, not {cell['tile']}")
        return price_prefill(cell["row"], sag128=cell["sag128"], sag64=cell.get("sag64"), knee=cell.get("knee"),
                             switches=switches)
    p = price_decode_cell(cell["model"], cell["cell"], cell["prm"], arm=cell.get("arm", "native"),
                          decl_copies=cell.get("decl_copies"), terms=cell.get("terms"), law=cell.get("law"),
                          prm_fix=cell.get("prm_fix"), switches=switches)
    if p is None:
        return {"ms": None, "regime": "decode", "terms": {}}
    t = (p.get("v6d") or {}).get("tile") or (p.get("v5d") or {}).get("tile")
    if t is not None and t != cell["tile"]:
        raise Refused(f"v6d prices this cell at {t}, not {cell['tile']}")
    return {"ms": p["ms"], "regime": "decode", "tile": t, "terms": {"D6FIX": p["v7"]["D6FIX"]} if "v7" in p else {},
            "parts": p}


# ================================================================== --check-r3

def check_r3(limit: int | None = None, switches: Switches = V7) -> dict:
    """Every registered v3 prediction (rentals 6 and 7, r3_timing_model_v4.CHECK_FILES) re-priced by v7's r3 branch:
    equality with the registered number (6 decimals) and with rental 6's pricer bit for bit."""
    import json
    X = r3_pricer()
    V4 = X["V4"]
    out = {"compared": 0, "unequal_registered": [], "unequal_pricer": [], "files": list(V4.CHECK_FILES)}
    for f in V4.CHECK_FILES:
        reg = json.loads((REPO / f).read_text())
        hist = {k: json.loads((REPO / v["path"]).read_text()) for k, v in reg["histograms"]["files"].items()}
        for r in reg["predictions"]:
            if limit is not None and out["compared"] >= limit:
                return out
            page = hist[f"{r['model']}-{r['page']}.json"]
            c = next(c for c in page["cells"] if c["label"] == r["label"] and int(c["n"]) == int(r["n"]))
            seed = None if c["counts"] is None else page["shuffle_seed"]
            copies = r.get("copies", 9)
            got = price({"tile": R3_TILE, "model": r["model"], "arm": r["arm"], "n": int(r["n"]), "counts": c["counts"],
                         "copies": copies, "seed": seed, "G": 8, "A_us": r["A_proxy_us"]}, switches)
            ref = X["pr"].price(r["model"], r["arm"], int(r["n"]), c["counts"], copies, seed, A_us=r["A_proxy_us"])
            out["compared"] += 1
            row = [r["model"], r["page"], r["n"], r["label"], r["arm"]]
            if round(got["ms"], 6) != r["ms"]["v3"]:
                out["unequal_registered"].append({"row": row, "registered": r["ms"]["v3"], "v7": got["ms"]})
            if got["ms"] != ref["v3"]:
                out["unequal_pricer"].append({"row": row, "pricer": ref["v3"], "v7": got["ms"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-r3", action="store_true", help="v7 at R3's tile against every registered v3 prediction")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if not a.check_r3:
        ap.print_help()
        return 2
    got = check_r3(a.limit)
    print(f"v7 at R3's tile against registered v3: {got['compared']} predictions compared; "
          f"{len(got['unequal_registered'])} differ from the registered number (6 decimals), "
          f"{len(got['unequal_pricer'])} differ from rental 6's pricer bit for bit")
    return 0 if got["compared"] and not got["unequal_registered"] and not got["unequal_pricer"] else 1


if __name__ == "__main__":
    sys.exit(main())
