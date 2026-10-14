"""MODEL v4's bytes (the tile stage, 2026-10-14; docs/registered/2026-10-14-tileA-bytes-gh200), and
v4's one pricing entry point, `price`.

A separate module over `wave_split_bytes` and `wave_split_bytes_v3` (both byte-identical to their
registered pins). The byte model is v3's (MODEL M4, the CAL 8x7B registered view, unchanged); at a
tile it walks the tile's geometry (`geometry_at`): BLOCK_M/N/K, warps and stages from the tile,
P_g = ceil(N_g / BLOCK_N), and the co-residency window W_c,g = 132 x the tile's occupancy
(the census's; the shared-memory limit before it).

THE A KEY (design-tile 7, design-tile-review 5). The w1 A term is the one part of M4 that moves
with the key; four keys are registered as candidates and rental A NAMES one (tileA-bytes):
  v3        M4's content key, as registered at rental 6.
  CLAMP     v3 with each GEMM's q floored at the active weights once, q_ideal = (experts with
            a row) / E (design-tile-review 5: at BLOCK_M 128 M4 prices uniform cells below one
            weight set, Qwen1.5 n 1 w1 q 0.967). A physical floor, not a fit. The fallback when
            nothing is NAMED.
  ROWKEY60  rental 6's row-granular LRU (rowkeys.py) at 60 MiB, rental 7's NAMED key (OPEN).
  ROWKEY30  the same at 30 MiB, one of the card's two 60 MiB L2 partitions.
A row key prices w1's read bytes whole (A and B by the LRU); w2 keeps M4's bytes under every key.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import wave_split_bytes as W  # noqa: E402
import wave_split_bytes_v3 as WV3  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

KEYS = ("v3", "CLAMP", "ROWKEY60", "ROWKEY30")
ROWKEY_C = {"ROWKEY60": 60.0, "ROWKEY30": 30.0}


def geometry_at(geom: W.Geometry, t: V4.Tile, occ: dict) -> W.Geometry:
    """`geom` (an Engine geometry at R3's tile) at tile t and occupancy occ ({w1, w2} CTAs per
    SM). At R3's tile and R3's occupancy it is `geom` itself."""
    if t == V4.R3_TILE and occ == V4.OCC_R3:
        return geom
    import dram_counter_route as DCR
    cfg = MODEL_CONFIGS[geom.model]
    P = {g: DCR.pid_n_count(DCR.r3_gemm_geometry(cfg, g)[1], t.bn) for g in W.GEMMS}
    return dataclasses.replace(geom, block_m=t.bm, block_n=t.bn, block_k=t.bk, num_warps=t.warps,
                               num_stages=t.stages, P=tuple(sorted(P.items())),
                               W_c=tuple(sorted({g: geom.sm_count * int(occ[g]) for g in W.GEMMS}.items())),
                               ctas_per_sm=tuple(sorted({g: int(occ[g]) for g in W.GEMMS}.items())))


def q_ideal(model: str, n: int, counts, block_m: int) -> float:
    """The active weights once, in weight sets: experts with at least one row over E."""
    E = MODEL_CONFIGS[model].num_experts
    if counts is None:
        return 1.0
    return sum(1 for c in counts if int(c) > 0) / E


_MODELS: dict = {}
_BASE: dict = {}


def base_geometry(eng, model: str, copies: int) -> W.Geometry:
    """The CAL engine's geometry of `model` (rental 5's Engine.geom). A model whose R3 ladder R3
    refuses at BLOCK_M 32 (deepseek-v2-lite: k 6 over E 64 needs rows per expert in multiples
    of 3, design-tile-review 2) gets the same geometry built without R3's ladder design: the
    target's shapes (cross_model_predict.target_geometry), its E, and the declaration every
    histogram page uses (NATIVE E, SHARED and PRIVATE E x copies)."""
    key = (model, copies)
    if key in _BASE:
        return _BASE[key]
    import cross_model_predict as CMP
    try:
        g = eng.geom(model, copies)
    except CMP.Refused:
        E = MODEL_CONFIGS[model].num_experts
        g = CMP.target_geometry(eng.card.geom, model, None)
        decl = {"native": E, "shared": E * copies, "private": E * copies}
        g = dataclasses.replace(g, experts=E, copies_declared=copies, declared=tuple(sorted(decl.items())))
    _BASE[key] = g
    return g


def model_at(eng, model: str, copies: int, t: V4.Tile, occ: dict) -> WV3.ModelV3:
    key = (model, copies, t, tuple(sorted(occ.items())))
    if key not in _MODELS:
        _MODELS[key] = WV3.ModelV3(geometry_at(base_geometry(eng, model, copies), t, occ))
    return _MODELS[key]


def q_v3(eng, model: str, n: int, counts, copies: int, seed, t: V4.Tile, G: int, occ: dict) -> dict | None:
    """M4's q per GEMM at the tile (SHARED's walk for both arms, as rental 5 and 6), None when
    either GEMM is ILL-POSED. A balanced cell off R3's tile is walked as the uniform histogram
    with no shuffle, which is what R3's balanced_ids are (realize_counts of the uniform counts):
    M4 prices the two the same (tests/test_rental6_model.py), and the balanced path's G-1
    monotonicity guard needs whole token counts at every smaller tread, which deepseek-v2-lite
    lacks at n 1 and 2 (k 6). At R3's tile the balanced path is v3's own."""
    if counts is None and t != V4.R3_TILE:
        E = MODEL_CONFIGS[model].num_experts
        counts, seed = tuple([n * t.bm] * E), None
    m = model_at(eng, model, copies, t, occ)
    tail = () if counts is None else (tuple(int(v) for v in counts), seed)
    ev = m.evaluate(eng.prm, [("shared", G, n, g) + tail for g in W.GEMMS])
    if any(r["ill"] for r in ev):
        return None
    return {g: float(r["q"]) for g, r in zip(W.GEMMS, ev, strict=True)}


def reads_of(q: dict, n: int) -> dict:
    """Predicted DRAM read bytes per GEMM (TM must be at the tile): q x W_g + n x A per tile."""
    return {g: q[g] * TM.BYTE_MODEL[f"W_{g}"] + n * TM.BYTE_MODEL[f"operand_per_tile_{g}"] for g in TM.GEMMS}


def keyed_reads(key: str, q: dict, n: int, qi: float, rowkey_w1: float | None) -> dict | None:
    """Read bytes under an A key; None when a row key has no priced w1 row."""
    if key == "v3":
        return reads_of(q, n)
    if key == "CLAMP":
        return reads_of({g: max(q[g], qi) for g in q}, n)
    if key in ROWKEY_C:
        if rowkey_w1 is None:
            return None
        r = reads_of(q, n)
        r["w1"] = float(rowkey_w1)
        return r
    raise ValueError(f"A key {key!r}: one of {KEYS}")


def proxy_A_us(M6, proxy: dict, model: str, arm: str, n32: int, copies: int) -> tuple[float | None, str]:
    """rental 6's SEEN A proxy (r6model.proxy_A_us) at the BLOCK_M-32 tread with the same ids. Its
    extrapolation reads the table's top two consecutive treads; where the table has a gap there
    (a model whose published probes skip a tread), the line through its two largest treads is
    used instead (printed as such). SHARED is the secondary arm: its scorer re-prices every cell
    with the page's own A."""
    try:
        return M6.proxy_A_us(proxy, model, arm, n32, copies)
    except KeyError:
        E = MODEL_CONFIGS[model].num_experts
        tab = proxy.get(f"{model}|{E * copies}") or proxy.get(f"{model}|{E * 9}")
        t = tab[arm]
        ks = sorted(int(k) for k in t)
        a, b = ks[-2], ks[-1]
        va, vb = float(t[str(a)]), float(t[str(b)])
        return round(vb + (n32 - b) * (vb - va) / (b - a), 4), f"extrapolated along n {a} -> {b} (the table skips n {b - 1})"


def price(pr, model: str, arm: str, n: int, counts, copies: int, seed, t: V4.Tile, G: int, prm: V4.Params, *,
          key: str = "CLAMP", A_us: float | None = None, reads: dict | None = None, rowkey_w1: float | None = None,
          proxy_n: int | None = None) -> dict:
    """v4's time of one cell, ms, with its bytes and constants. `pr` is rental 6's Pricer (the
    CAL engine, its v2 timing fit and the A proxy). `reads` given (a counter page's per-GEMM read
    bytes): E2's time from counted bytes. A_us None: the proxy at the BLOCK_M 32 tread with the
    same ids (n x BLOCK_M / 32), `proxy_n` overriding it."""
    import r3_timing_model_v3 as TV3
    r6dir = str(REPO / "scripts" / "scoring" / "rental6")
    if r6dir not in sys.path:
        sys.path.insert(0, r6dir)
    import r6model as M6
    occ = {g: prm.occ(t, g) for g in TM.GEMMS}
    phis = None if t == V4.R3_TILE else {g: V4.phi(t, g, prm) for g in TM.GEMMS}
    scale = {g: V4.c_scale(t, g, prm) for g in TM.GEMMS}
    how = "page"
    if A_us is None:
        n32 = proxy_n if proxy_n is not None else n * t.bm // 32
        A_us, how = proxy_A_us(M6, pr.proxy, model, arm, n32, copies)
        if A_us is None and arm != "native":
            # no published BLOCK_M-32 GH200 probe for this model (deepseek-v2-lite): A 0 until the page's
            # own align_probe replaces it (term (b) is additive in A; SHARED is the secondary arm)
            A_us, how = 0.0, "no proxy table: 0, replaced by the page's own A"
    ct = None if counts is None else tuple(int(v) for v in counts)
    with V4.at_tile(model, t, phis):
        q = None
        if reads is None:
            q = q_v3(pr.eng, model, n, ct, copies, seed, t, G, occ)
            if q is None:
                return {"ms": None, "ill": True, "key": key}
            reads = keyed_reads(key, q, n, q_ideal(model, n, ct, t.bm), rowkey_w1)
            if reads is None:
                return {"ms": None, "ill": False, "key": key, "why": "no priced row-key w1 row"}
        ctx = pr.ctx if occ == dict(pr.ctx.occupancy) else dataclasses.replace(pr.ctx, occupancy=dict(occ))
        cell = V4.make_cell(pr.eng, model, arm, n, ct, copies, G, t, reads, occ, pr.f2.k_w, ctx.sms)
        npath = V4.native_path_of(model, n, t)
        A = None if A_us is None else A_us * 1e-3
        with TM.dead_model("v2"):
            ms = V4.call_ms(pr.f2.x, cell, ctx, pr.f2.k_w, scale, native_path=npath, align_ms=A)
            parts = {}
            for g in TM.GEMMS:
                cg = float(pr.f2.x[1]) * scale[g]
                parts[g] = {"dead_ms": TM.dead_ms(cell.declared, n, g, cg, pr.f2.k_w, occ[g], ct),
                            "dead_ctas": TM.dead_ctas(cell.declared, n, g, ct),
                            "live_ctas": TM.window(arm, cell.declared, G, n, g,
                                                   TM.window_width(pr.f2.k_w, ctx.sms, occ[g]), ct).live,
                            "ksteps": TM.GEOMETRY[g].ksteps, "phi": TM.CTA_FIXED_KSTEPS[g]}
        b = TV3.v3_b_ms(arm, npath, A, key=cell.key)
    return {"ms": ms, "ill": False, "key": key, "reads": reads, "q": q, "A_us": A_us, "A_how": how,
            "path": cell.path, "native_path": npath, "declared": cell.declared, "occ": occ, "c_scale": scale,
            "phi": phis or dict(TM.CTA_FIXED_KSTEPS), "term_b_ms": b, "gemm": parts}
