"""Rental RR's pure rules (docs/registered/2026-11-11-rr-gh200): one module for the probe (scripts/instr_probe_rr.py), the
counter route (scripts/dram_counter_route_rr.py), the page generator, the registration and the reader, so a rule is
computed one way on every side. It imports PC2's rules module (scripts/scoring/pc2/pc2common.py, pinned, unchanged), and
through it PC's, and moe.spec; no torch, no vLLM.

RR is a MEASUREMENT rental of the w1 re-read L2 hit mechanism at vLLM's prefill tiles (owner, 2026-11-11: prefill gets
this one measurement rental and NO blind; prefill then closes and the project moves to the write-up). It publishes the
mechanism as a DESCRIPTIVE reading: no model form is registered, there is no gate and there is no blind. It stamps the
CTA ends of w1 G 1 launches (the unchanged instrumented copy, C3's stamps only) and counts the same cells' DRAM bytes
(ncu, through the pinned counter route's functions), on the same board, at the same 1410 MHz session lock.

STATED PLAINLY: the 42.5 MiB constant and the four forms read here were FOUND POST-HOC ON SEEN PAGES (moe-kernels-work
2026-11-11 design-P1 DESIGN.md section 2.3, on the Phi split and PC-R10's pages: C3's Mixtral-8x7B and Qwen2-57B stamps,
PC's Phi-3.5 stamps, counted bytes from seven boards). RR measures them on FRESH synthetic cells; it fits nothing that a
later page is scored against.

What RR takes from PC2 and PC unchanged: the board (700 W, the 1410 MHz session lock), the versions rule, the reserved
models and PC2's reserved-GEMM guard, ROW SANITY's ceilings, the stamped-call rule, the loads-in-flight reading off the
PTX, the label alphabet. What is new here: RR's shapes (rrs- names), the two prefill tiles (128x128x64 w8 s3 and 64x128x64
w8 s3), cells with several M-tiles per expert (the re-reads), the w1 read GEMM, the counted-bytes geometry (C_u, R), the
four registered forms of the per-pair hit rule, the wave-start spread rule, and RR's pages, plan and figures.

Every range clause is written to fail on a NaN (`not lo <= x <= hi`, math.isfinite).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(HERE.parent / "pc2"), str(HERE.parent / "pc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pc2common as PC2  # noqa: E402
import pccommon as PC  # noqa: E402

DATE = "2026-11-11"
NAME = f"{DATE}-rr-gh200"
SHAPES_FILE = f"docs/registered/{DATE}-rr-shapes.json"
PAGES_DIR = f"docs/registered/{DATE}-rr-pages"
PLAN = "scripts/plans/rentalRR-2026-11.plan"
DRIVER = "scripts/gh200_rr_session.sh"
PARENT_DRIVER = PC2.DRIVER
PROBE = "scripts/instr_probe_rr.py"
PARENT_PROBE = PC2.PROBE
LAUNCHER = "scripts/vm_run_rr.sh"
PARENT_LAUNCHER = PC2.LAUNCHER
COUNTER = "scripts/dram_counter_route_rr.py"
CARD = PC.CARD
SCHEMA = "moe-kernels/rr-page/1"
SHAPES_SCHEMA = "moe-kernels/rr-shapes/1"
COUNTER_SCHEMA = "moe-kernels/rr-counter-page/1"
SHUFFLE_SEED = 20261111
LABEL_RE = PC.LABEL_RE

# ------------------------------------------------------------------ the board (PC's, unchanged)
POWER_W = PC.POWER_W
SESSION_LOCK = PC.SESSION_LOCK
SMS = PC.SMS
RATE_CEILING_GBPS = PC.RATE_CEILING_GBPS
F_CEILING_MHZ = PC.F_CEILING_MHZ
VERSIONS = PC.VERSIONS
VERSIONS_MAJOR_MINOR = PC.VERSIONS_MAJOR_MINOR

# ------------------------------------------------------------------ models: allowed, reserved
Refused = PC.Refused
#: PC's reserved models, kept (owner, 2026-11-11: no blind follows, so keeping blind candidates unspent no longer
#: matters; still, no model is loaded that does not need loading). PC2's reserved-GEMM guard is kept unchanged
RESERVED = PC.RESERVED
#: the real models RR loads by name, both SEEN: Mixtral-8x7B (its shape IS series A's P 224 cell at E 8, the seen
#: anchor; private random weights at its shape) and Phi-3.5-MoE (the droppable cross-board replicate and the census)
REAL_ALLOWED = ("mixtral-8x7b", "phi-3.5-moe")
SHAPE_PREFIX = "rrs-"

shape_key = PC.shape_key
model_shapes = PC.model_shapes
reserved_shapes = PC.reserved_shapes
gemm_dims = PC2.gemm_dims
reserved_gemms = PC2.reserved_gemms
gemm_guard = PC2.gemm_guard

# ------------------------------------------------------------------ the design's sketch (design-P1 section 4, FROZEN)
#: series A: E 8, k 2, H = K 4096 (w1 slab 1 MiB at BLOCK_N 128), F = 64 P so w1 has P N-tiles; P 100 and 224 are the
#: w1 GEMMs of Phi-3.5 and Mixtral-8x7B (the seen anchors)
SERIES_A_P = (20, 36, 48, 64, 80, 100, 160, 224)
#: series B: K 8192 (slab 2 MiB, over deepseek-v3's 1.75 MiB without its GEMM)
SERIES_B_P = (24, 40)
#: series C: E 64, k 8 (many experts with few M-tiles each, as deepseek-v3)
SERIES_C_P = (36, 64)
#: balanced layouts: m_e M-tiles per active expert (every expert full)
M_E_A = (2, 4, 8)
M_E_B = (2, 4)
M_E_C = (2,)
#: the seen anchor P values (their w1 GEMM is a seen model's); every other cell is a FRESH synthetic cell
ANCHOR_P = (100, 224)

#: the skewed layouts (FROZEN literals, recomputed from the published pages by tests/test_instr_probe_rr.py):
#: E 8: Mixtral-8x7B-tp2's PT histogram, n 2 (C4's counter page, 2026-10-27 cal4 session,
#: mixtral-8x7b-tp2-b128g1-r3-counters lock1410 r3c-g1.json, NATIVE bincount), rescaled to 4 x BLOCK_M x 8 rows;
#: E 64: Qwen2-57B's PT histogram, n 1 (A2's counter page, 2026-10-21 tileA2 session, qwen2-57b-a14b-b128g1-r3-counters
#: lock1410 r3c-g1.json, NATIVE bincount), rescaled to 2 x BLOCK_M x 64 rows. Only the count vectors are used (no
#: model's shape, GEMM or weights; Qwen2-57B stays unloaded)
SKEW_E8 = (216, 228, 265, 253, 274, 271, 286, 255)
SKEW_E8_SOURCE = ("2026-10-27-nvidia_gh200_480gb-cal4-session", "mixtral-8x7b-tp2-b128g1-r3-counters", "PT", 2)
SKEW_E64 = (79, 264, 140, 88, 52, 66, 201, 77, 114, 72, 252, 131, 90, 129, 97, 172, 90, 78, 153, 77, 82, 104, 136, 102,
            81, 147, 60, 105, 81, 109, 101, 96, 99, 383, 85, 83, 82, 160, 517, 77, 96, 183, 71, 98, 184, 75, 92, 305, 118,
            146, 316, 60, 223, 122, 174, 82, 73, 59, 100, 74, 83, 163, 71, 112)
SKEW_E64_SOURCE = ("2026-10-21-nvidia_gh200_480gb-tileA2-session", "qwen2-57b-a14b-b128g1-r3-counters", "PT", 1)
#: the skew's size in M-tiles per expert on average (A: n 4, C: n 2)
SKEW_N = {8: 4, 64: 2}


def _s(name, E, H, F, k):
    return {"name": name, "E": E, "H": H, "F": F, "k": k}


def shape_for(series: str, P: int) -> str:
    """The cell's shape name: P 224 at E 8 is Mixtral-8x7B's shape exactly (a SEEN model, by name); the rest rrs-."""
    if series == "A":
        return "mixtral-8x7b" if P == 224 else f"rrs-e8-h4096-p{P}"
    if series == "B":
        return f"rrs-e8-h8192-p{P}"
    return f"rrs-e64-h4096-p{P}-k8"


SHAPES = tuple(
    [_s(f"rrs-e8-h4096-p{p}", 8, 4096, 64 * p, 2) for p in SERIES_A_P if p != 224]
    + [_s(f"rrs-e8-h8192-p{p}", 8, 8192, 64 * p, 2) for p in SERIES_B_P]
    + [_s(f"rrs-e64-h4096-p{p}-k8", 64, 4096, 64 * p, 8) for p in SERIES_C_P])


def check_shapes_doc(doc: dict) -> dict:
    """The registered shapes file: {name: MoEConfig}. Refuses a name of MODEL_CONFIGS, a name without the rrs- prefix,
    any shape whose (E, H, F, k) equals a MODEL_CONFIGS entry's, and any shape with a reserved model's GEMM."""
    from moe.spec import MODEL_CONFIGS, MoEConfig
    if not isinstance(doc, dict) or doc.get("schema") != SHAPES_SCHEMA:
        raise Refused(f"the shapes file's schema is {doc.get('schema') if isinstance(doc, dict) else '?'}, not {SHAPES_SCHEMA}")
    real = model_shapes()
    out = {}
    for s in doc.get("shapes") or []:
        name = s.get("name")
        if not isinstance(name, str) or not name.startswith(SHAPE_PREFIX) or not LABEL_RE.match(name):
            raise Refused(f"synthetic shape {name!r}: an rrs- name")
        if name in MODEL_CONFIGS or name in out:
            raise Refused(f"synthetic shape {name}: the name is a model's, or given twice")
        try:
            key = shape_key(s["E"], s["H"], s["F"], s["k"])
        except (KeyError, TypeError, ValueError):
            raise Refused(f"synthetic shape {name}: E, H, F and k are whole numbers") from None
        hit = sorted(m for m, v in real.items() if v == key)
        if hit:
            raise Refused(f"synthetic shape {name}: (E, H, F, k) {key} is {hit[0]}'s shape")
        cfg = MoEConfig(name=name, hidden_size=key[1], intermediate_size=key[2], num_experts=key[0], top_k=key[3])
        gemm_guard(name, cfg)
        out[name] = cfg
    if not out:
        raise Refused("the shapes file holds no shape")
    return out


def shapes_doc() -> dict:
    return {"schema": SHAPES_SCHEMA, "registration": NAME,
            "what": ("RR's synthetic expert shapes (design-P1 section 4): built in the probe and counter processes as "
                     "moe.spec.MoEConfig, never added to moe/spec.py; series A E 8 H 4096 F 64 P, series B E 8 H 8192, "
                     "series C E 64 k 8; no shape equals any MODEL_CONFIGS entry's (E, H, F, k) (series A's P 224 is "
                     "Mixtral-8x7B's and runs by that name), and no GEMM of any shape equals a reserved model's GEMM"),
            "shapes": [dict(s) for s in SHAPES]}


def load_shapes(path: Path | None = None) -> dict:
    p = Path(path) if path else REPO / SHAPES_FILE
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read the shapes file {p}: {exc}") from None
    return check_shapes_doc(doc)


def shape_cfg(name: str, shapes: dict | None = None):
    """The MoEConfig of a cell's shape: a registered rrs- shape or an allowed SEEN model. Refuses every reserved model,
    any shape equal to a reserved model's, and any shape with a reserved model's GEMM."""
    from moe.spec import MODEL_CONFIGS
    if name in RESERVED:
        raise Refused(f"{name} is reserved (owner decision, kept): RR never loads it")
    if name.startswith(SHAPE_PREFIX):
        shapes = load_shapes() if shapes is None else shapes
        if name not in shapes:
            raise Refused(f"synthetic shape {name} is not in the registered shapes file")
        cfg = shapes[name]
    elif name in REAL_ALLOWED:
        cfg = MODEL_CONFIGS[name]
    else:
        raise Refused(f"model {name!r} is not one RR may load ({', '.join(REAL_ALLOWED)} or a registered rrs- shape)")
    if shape_key(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k) in reserved_shapes():
        raise Refused(f"{name}'s shape is a reserved model's (owner decision)")
    gemm_guard(name, cfg)
    return cfg


# ------------------------------------------------------------------ tiles
T128 = PC.P128                    # 128x128x64 w8 s3: vLLM's prefill tile, 2 CTAs per SM (B3's census)
T64 = PC.P64                      # 64x128x64 w8 s3: 3 CTAs per SM
TILES = {"128x128x64w8s3": T128, "64x128x64w8s3": T64}
CENSUS_TILES = tuple(TILES)
CENSUS_MODELS = PC.CENSUS_MODELS
#: the registered occupancy (PC's literal for both tiles, from B3's and PC's censuses; PC's rc3 and rcp regchecks read
#: the stamped copy at it: 2 = 2 at 128x128 w8 s3 and 3 = 3 at 64x128x64 w8 s3)
OCCUPANCY = {"128x128x64w8s3": PC.OCCUPANCY["128x128x64w8s3"], "64x128x64w8s3": PC.OCCUPANCY["64x128x64w8s3"]}

tile_key = PC.tile_key
config_of = PC.config_of
tile_of = PC.tile_of
mma_path = PC.mma_path
inflight_stages = PC.inflight_stages
q_bytes = PC.q_bytes
sha256 = PC.sha256


def wave(tile) -> int | None:
    o = OCCUPANCY.get(tile_key(tile))
    return None if o is None else SMS * o


# ------------------------------------------------------------------ cells
live_ctas = PC.live_ctas
slab_bytes = PC.slab_bytes
STAMP_CALLS, STAMP_CALLS_SMALL, SMALL_G = PC.STAMP_CALLS, PC.STAMP_CALLS_SMALL, PC.SMALL_G
stamp_calls = PC.stamp_calls
naive = PC.naive


def is_naive(cfg, counts) -> bool:
    return naive(sum(int(c) for c in counts) // int(cfg.top_k), cfg.num_experts, cfg.top_k)


def slab_once(cfg, counts, tile=None) -> bool:
    """Every B slab read by exactly one CTA: false on every RR cell (an expert with two or more M-tiles re-reads its w1
    slabs), so ROW SANITY takes neither the X ceiling nor the L2 guard (the re-reads may be L2-served by design)."""
    bm = int(tile[0]) if tile is not None else None
    if bm is None:
        return False
    return max(-(-int(c) // bm) for c in counts if int(c)) <= 1


def m_tiles(counts, bm: int) -> list[int]:
    return [-(-int(c) // int(bm)) for c in counts if int(c)]


def geometry(cfg, counts, tile) -> dict:
    """The w1 re-read geometry of one cell (COMPUTED): P = ceil(N / BLOCK_N) N-tiles of w1 (N = 2F); W_c = 132 x the
    registered occupancy; the M-tiles per active expert; pairs = sum over experts of (m_e - 1) x P (C3-R2's pairs: same
    expert, same pid_n, pid_m - 1); the w1 slab K x BLOCK_N x 2; F_stag = P x slab (the bytes between partners if
    launches rolled evenly); waves = live CTAs / W_c; C_u = each active expert's w1 once plus the unique tokens' A rows
    once (T x H x 2; design-P1 P1-F4); R = the re-read demand, sum over experts of (m_e - 1) x the expert's w1 bytes."""
    bm, bn = int(tile[0]), int(tile[1])
    N, K = gemm_dims(cfg, "w1")
    P = -(-int(N) // bn)
    mt = m_tiles(counts, bm)
    T = sum(int(c) for c in counts) // int(cfg.top_k)
    We = int(N) * int(K) * 2
    slab = int(K) * bn * 2
    W = wave(tile)
    live = sum(mt) * P
    return {"P": P, "N": int(N), "K": int(K), "W_c": W, "W_c_over_P": (W / P) if W else None, "m_e": mt,
            "m_e_max": max(mt), "A": len(mt), "T": T, "pairs": sum(m - 1 for m in mt) * P, "slab_bytes": slab,
            "F_stag_MiB": P * slab / 2 ** 20, "live_w1": live, "waves_w1": (live / W) if W else None,
            "C_u_bytes": len(mt) * We + T * int(cfg.hidden_size) * 2,
            "R_bytes": sum(m - 1 for m in mt) * We, "W_e1_bytes": We}


def counted_m(D: float, C_u: float, R: float):
    """The measured w1 re-read miss fraction (design-P1 section 2.2): m = (D - C_u) / R; None unless D, C_u, R finite
    and R positive."""
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in (D, C_u, R)) or not R > 0:
        return None
    return (float(D) - float(C_u)) / float(R)


def rescale(v, rows: int) -> list[int]:
    """A count vector rescaled to `rows` by the largest-remainder rule (the sum is exact; ties by index)."""
    s = sum(int(x) for x in v)
    raw = [int(x) * int(rows) / s for x in v]
    out = [int(math.floor(x)) for x in raw]
    left = int(rows) - sum(out)
    order = sorted(range(len(v)), key=lambda i: (-(raw[i] - out[i]), i))
    for i in order[:left]:
        out[i] += 1
    return out


def _cell(label, shape, counts, tile, role, series, P, layout):
    return {"label": label, "shape": shape, "T": None, "counts": [int(x) for x in counts], "config": config_of(tile),
            "group_m": 1, "flush": True, "read": "w1", "role": role, "path": "aligned", "series": series, "P": int(P),
            "layout": layout}


def _k(shape: str) -> int:
    from moe.spec import MODEL_CONFIGS
    shp = {s["name"]: s for s in SHAPES}
    return shp[shape]["k"] if shape in shp else MODEL_CONFIGS[shape].top_k


def _E(shape: str) -> int:
    from moe.spec import MODEL_CONFIGS
    shp = {s["name"]: s for s in SHAPES}
    return shp[shape]["E"] if shape in shp else MODEL_CONFIGS[shape].num_experts


def _c(label, series, P, layout, tile, role):
    shape = shape_for(series, P)
    E, k = _E(shape), _k(shape)
    bm = int(tile[0])
    if layout.startswith("m"):
        counts = [int(layout[1:]) * bm] * E
    else:
        src = SKEW_E8 if E == 8 else SKEW_E64
        counts = rescale(src, SKEW_N[E] * bm * E)
    c = _cell(label, shape, counts, tile, role, series, P, layout)
    c["T"] = sum(counts) // k
    return c


def tile_tag(tile) -> str:
    return str(int(tile[0]))


def block(tile) -> list[dict]:
    """One tile's 40 cells (design-P1 section 4, unit 3): series A (P 20 to 224; m_e 2, 4, 8 and the tp2 PT skew), series
    B (K 8192; P 24, 40; m_e 2, 4), series C (E 64, k 8; P 36, 64; m_e 2 and the Qwen2-57B PT skew)."""
    t = tile_tag(tile)
    out = []
    for p in SERIES_A_P:
        role = "seen anchor (w1 GEMM of " + ("Phi-3.5" if p == 100 else "Mixtral-8x7B") + ")" if p in ANCHOR_P else "fresh"
        for lay in [f"m{m}" for m in M_E_A] + ["skew"]:
            out.append(_c(f"A{t}-P{p}-{lay}", "A", p, lay, tile, role))
    for p in SERIES_B_P:
        for lay in [f"m{m}" for m in M_E_B]:
            out.append(_c(f"B{t}-P{p}-{lay}", "B", p, lay, tile, "fresh"))
    for p in SERIES_C_P:
        for lay in [f"m{m}" for m in M_E_C] + ["skew"]:
            out.append(_c(f"C{t}-P{p}-{lay}", "C", p, lay, tile, "fresh"))
    return out


#: page -> (tile, builder)
BLOCKS = {"rr-128": (T128, lambda: block(T128)), "rr-64": (T64, lambda: block(T64))}


def page_doc(page: str) -> dict:
    tile, fn = BLOCKS[page]
    return {"schema": SCHEMA, "page": page, "tile": tile_key(tile), "lock_mhz": SESSION_LOCK,
            "shuffle_seed": SHUFFLE_SEED, "shapes_file": SHAPES_FILE, "registration": NAME,
            "label": (f"SYNTHETIC routings (scripts/scoring/rr/gen_rr_pages.py, {NAME}): RR's {tile_key(tile)} cells, "
                      "series A, B, C; G 1, w1 read, every cell with re-reads; DESCRIPTIVE"),
            "cells": fn()}


def page_files() -> dict:
    """{published file name under PAGES_DIR: text} (sorted keys, indent 1)."""
    return {f"{p}.json": json.dumps(page_doc(p), indent=1, sort_keys=True) + "\n" for p in BLOCKS}


def shapes_text() -> str:
    return json.dumps(shapes_doc(), indent=1, sort_keys=True) + "\n"


def check_cell(c: dict, shapes: dict, tiles: dict = TILES) -> dict:
    """One RR page cell checked as the card would meet it; returns it with its cfg and tile. PC's refusals (an unknown or
    reserved shape, counts that are not E whole numbers summing to T k, a config not in the tile list, flush not a
    bool, a label off the alphabet, G 1 only) and RR's: the aligned path only (vLLM's naive path needs T k 4 <= E), the
    w1 read GEMM, T at most 4096 (the probe's input rows), and at least one expert with two or more M-tiles (every RR
    cell has w1 re-reads)."""
    from moe.routing.distributions import feasible
    label = c.get("label")
    if not isinstance(label, str) or not LABEL_RE.match(label):
        raise Refused(f"cell label {label!r}: letters, digits and _ . - only")
    cfg = shape_cfg(str(c.get("shape")), shapes)
    E, k = cfg.num_experts, cfg.top_k
    T = c.get("T")
    if not isinstance(T, int) or isinstance(T, bool) or not 1 <= T <= 4096:
        raise Refused(f"cell {label}: T {T!r}: 1 to 4096 (the probe's input rows)")
    counts = c.get("counts")
    if (not isinstance(counts, list) or len(counts) != E
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts)):
        raise Refused(f"cell {label}: counts must be {E} whole numbers")
    ok, why = feasible(counts, T, k)
    if not ok:
        raise Refused(f"cell {label} (V-BIN): {why}")
    if c.get("path", "aligned") != "aligned" or naive(T, E, k):
        raise Refused(f"cell {label}: RR's cells take vLLM's aligned path (T k 4 > E), not {c.get('path')!r}")
    conf = c.get("config") or {}
    try:
        tile = tile_of(conf)
    except (KeyError, TypeError, ValueError):
        raise Refused(f"cell {label}: config {conf!r} is not BLOCK_SIZE_M, N, K, num_warps, num_stages") from None
    if tile_key(tile) not in tiles:
        raise Refused(f"cell {label}: tile {tile_key(tile)} is not one of the registered {sorted(tiles)}")
    if int(c.get("group_m", 1)) != 1:
        raise Refused(f"cell {label}: GROUP_SIZE_M {c.get('group_m')}: RR's cells are G 1 (the only w1 re-read at G 1 "
                      "is an expert's next M-tile, P pids later)")
    if not isinstance(c.get("flush", True), bool) or c.get("flush", True) is not True:
        raise Refused(f"cell {label}: flush {c.get('flush')!r}: every RR cell starts at a cold L2 (true)")
    if c.get("read", "w1") != "w1":
        raise Refused(f"cell {label}: read {c.get('read')!r}: RR reads w1")
    if max(m_tiles(counts, int(tile[0]))) < 2:
        raise Refused(f"cell {label}: no expert holds two M-tiles at BLOCK_M {tile[0]}: no w1 re-read")
    return {**c, "cfg": cfg, "tile": tile}


def check_page_doc(doc: dict, shapes: dict) -> list[dict]:
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise Refused(f"the page's schema is {doc.get('schema') if isinstance(doc, dict) else '?'}, not {SCHEMA}")
    raw = doc.get("cells")
    if not isinstance(raw, list) or not raw:
        raise Refused("the page holds no cells")
    seed = doc.get("shuffle_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise Refused(f"shuffle_seed {seed!r}: a whole number")
    out, seen = [], set()
    for c in raw:
        x = check_cell(c, shapes)
        if x["label"] in seen:
            raise Refused(f"cell label {x['label']} twice")
        seen.add(x["label"])
        out.append(x)
    if len({tile_key(c["tile"]) for c in out}) != 1:
        raise Refused("an RR page holds one tile (its counter unit and its stamps unit read one occupancy)")
    return out


# ------------------------------------------------------------------ the readings' registered constants
MIB = 2 ** 20
#: the four forms of the per-pair hit rule read on RR's pages (registered before any page; every constant FOUND
#: POST-HOC ON SEEN PAGES, design-P1 section 2.3 and work/fit_s.out): name -> (distance, law, constant in MiB, origin)
#: distance 'start' = the B bytes streamed GPU-wide between the partners' starts (PC-R10's integral), 'mid' = between
#: their mid-lives; law 'step' = a re-read misses iff the distance exceeds the constant; 'exp' = it misses with
#: probability 1 - exp(-d / C) (random replacement; a negative distance counts as 0)
FORMS = {
    "F1 step at 42.5 MiB, between starts": ("start", "step", 42.5,
                                            "design-P1's chosen rule (P1-HIT): the step's best constant on seen stamps, "
                                            "rms 0.048 in m over 54 (cell, board) rows"),
    "F2 step at 45.0 MiB, between mid-lives": ("mid", "step", 45.0,
                                               "the mid-lives alternative (P1-F2): best constant 45.0 MiB, rms 0.043"),
    "F3 step at 30.0 MiB, between starts (one partition's capacity)": ("start", "step", 30.0,
                                                                        "the partition-capacity alternative (P1-F3), "
                                                                        "rejected on seen stamps (best C 40 to 62.5)"),
    "F4 random replacement 1 - exp(-d / 62.5 MiB), between starts": ("start", "exp", 62.5,
                                                                       "the random-replacement alternative: best "
                                                                       "constant 62.5 MiB, rms 0.079 on seen stamps"),
}
PRIMARY_FORM = "F1 step at 42.5 MiB, between starts"
#: the descriptive refit grid (design-P1 fit_s.py's): the constant per (distance, law) that minimises rms(m_hat - m)
#: on RR's own cells, printed beside the registered constants; DESCRIPTIVE, never used to price anything
REFIT_GRID_MIB = tuple(round(10.0 + 2.5 * i, 1) for i in range(57))      # 10.0 to 150.0
REFIT_LAWS = (("start", "step"), ("mid", "step"), ("start", "exp"))
#: a counted cell READ: its calls' w1 DRAM reads within 3% (max - min over the median, the project's per-call spread)
COUNT_SPREAD_MAX = 0.03
#: counter sanity: C_u x (1 - 2%) <= D <= (C_u + R) x 1.05, else NOT READ (D under the compulsory bytes, or over every
#: re-read missing, is a counting or attribution defect, not a miss fraction)
COUNT_LO, COUNT_HI = 0.98, 1.05
#: counted calls per cell (the pinned route's R3_CALLS_PER_CELL) and warm-up calls (the readback is the first)
COUNT_CALLS, COUNT_WARMUPS = 3, 2
#: a stamps cell READ with at least this many valid w1 launches
MIN_VALID_CALLS = 2
#: the wave-start spread (design-P1 section 2.4): per full schedule wave k (pids k W_c to (k + 1) W_c - 1, all live),
#: (q90 - q10 of its CTAs' starts) / the launch's median life
WAVE_Q = (0.10, 0.90)
#: NUMBERS ONLY (owner, 2026-11-11): w_inf = the median spread over waves k >= K_FLAT (read with at least FLAT_MIN such
#: waves, else NOT IDENTIFIED), k_90 = the first wave whose spread reaches 0.9 w_inf, and the OLS slope of the spread
#: over k >= K_FLAT, per wave; no label and no band (an earlier draft's band was set with PC's pages in view and was
#: removed on the owner's decision before any page). A pooled wave reads with at least POOL_MIN_CELLS cells
K_FLAT, FLAT_MIN = 5, 2
POOL_MIN_CELLS = 3
#: RR-R3's W_c / P groups (printed): the resident M-tiles per wave, the quantity the design's spread is indexed by
WCP_EDGES = (2.0, 4.0, 8.0)
#: life spread (descriptive): (q84 - q16) / 2 of the lives past the first wave over the median life
LIFE_Q = (0.16, 0.84)
#: C3's OD-1 NOT-READ bound on the cell's w1 overhead (PC's, unchanged)
OVERHEAD_MEDIAN_MAX, OVERHEAD_WORST_MAX = PC.OVERHEAD_MEDIAN_MAX, PC.OVERHEAD_WORST_MAX
TIMER_UNIT_MIN = PC2.TIMER_UNIT_MIN

# ------------------------------------------------------------------ the plan
tile_keys = PC.tile_keys
plan_line = PC.plan_line
PHI_128 = PC.PHI_128

UNITS = (
    ("-", "prelude", "", f"lock-mhz={SESSION_LOCK} power-limit-w={POWER_W}", 2, 15, "nodrop"),
    ("-", "tilecensus", "rr", f"tiles={','.join(CENSUS_TILES)} models={','.join(CENSUS_MODELS)}", 2, 15, "nodrop"),
    ("-", "regcheck", "rcp", "probe=pc", 3, 20, "nodrop"),
    ("-", "perturb", "pt", "probe=pc gates=s128,s64,sr", 5, 25, "nodrop"),
    ("-", "stamps", "s128", f"probe=pc page={PAGES_DIR}/rr-128.json", 3, 12, "nodrop"),
    ("-", "decodebytes", "c128", f"histogram={PAGES_DIR}/rr-128.json kernels=rr", 9, 25, "nodrop"),
    ("-", "stamps", "s64", f"probe=pc page={PAGES_DIR}/rr-64.json", 3, 12, "nodrop"),
    ("-", "decodebytes", "c64", f"histogram={PAGES_DIR}/rr-64.json kernels=rr", 8, 25, "nodrop"),
    ("phi-3.5-moe", "stamps", "sr", f"probe=pc {tile_keys(T128)} histogram={PHI_128} instr-stamps=ends stamp-groups=1 "
                                    "stamp-arms=native", 2, 10, "rep"),
    ("phi-3.5-moe", "bytes", "br", f"{tile_keys(T128)} histogram={PHI_128} arms=native byte-groups=1", 6, 18, "rep"),
)
#: the cut order, last unit first: the cross-board replicate (Phi-3.5 128g1, stamps and counted bytes) only; the core
#: (both tiles' stamps and counted bytes) is nodrop
CUT_ORDER = ("rep",)
#: every unit the regcheck drop rule drops: every probe=pc unit (PC's owner decision 2, kept)
PC_UNIT_LABELS = ("pt", "s128", "s64", "sr")
VARIANTS = {lab: f"u{i + 1}-{lab}" for i, u in enumerate(UNITS) for lab in [u[2]] if u[1] == "stamps"}
PAGE_OF = {"s128": f"{PAGES_DIR}/rr-128.json", "s64": f"{PAGES_DIR}/rr-64.json"}
COUNTER_OF = {"c128": f"{PAGES_DIR}/rr-128.json", "c64": f"{PAGES_DIR}/rr-64.json"}
#: a stamps unit and its counter unit read the same page on the same board
PAIRED = {"s128": "c128", "s64": "c64"}
BOOT_MIN, RATE, RESERVE_MIN, DEADLINE_SLACK_MIN, CAP_PAD_MIN = PC.BOOT_MIN, PC.RATE, PC.RESERVE_MIN, \
    PC.DEADLINE_SLACK_MIN, PC.CAP_PAD_MIN


def rental() -> dict:
    """PC's rule on RR's units: wall = units + 25 min of boot and setup; the deadline the units, boot, the driver's 8 min
    reserve and 5 min of slack need, rounded up to 5 min; the guardian's hard cap 15 min later; $2.29/h."""
    est = sum(u[4] for u in UNITS)
    nodrop = sum(u[4] for u in UNITS if u[6] == "nodrop")
    wall = est + BOOT_MIN
    deadline = 5 * math.ceil((wall + RESERVE_MIN + DEADLINE_SLACK_MIN) / 5)
    cap = deadline + CAP_PAD_MIN
    return {"units": len(UNITS), "units_min": est, "nodrop_min": nodrop, "boot_setup_min": BOOT_MIN, "wall_min": wall,
            "cost_usd": round(wall / 60 * RATE, 2), "rate_usd_per_h": RATE, "deadline_after_launch_min": deadline,
            "cost_at_deadline_usd": round(deadline / 60 * RATE, 2), "hard_cap_after_launch_min": cap,
            "cost_at_hard_cap_usd": round(cap / 60 * RATE, 2), "nodrop_wall_min": nodrop + BOOT_MIN,
            "nodrop_cost_usd": round((nodrop + BOOT_MIN) / 60 * RATE, 2),
            "rule": ("units x 1 plus 25 min of boot and setup (PC's and C3's rule); deadline launch + wall + the driver's "
                     "8 min reserve + 5 min, rounded up to 5 min; the guardian's hard cap 15 min after the deadline")}


def plan_text() -> str:
    r = rental()
    hdr = [
        f"# RENTAL RR (registered {DATE}, before any page): the w1 re-read L2 hit mechanism, a MEASUREMENT rental.",
        f"# GENERATED by scripts/scoring/rr/register_rr.py from registration {NAME}; do not edit by hand.",
        "# vLLM v0.27.1's fused_moe_kernel through the instrumented copy at the ends level (moe/instrumented, unchanged),",
        "# and the plain kernel's counted DRAM bytes on the same cells (scripts/dram_counter_route_rr.py over the pinned",
        "# counter route's functions), at vLLM's prefill tiles on synthetic routings and synthetic shapes (Mixtral-8x7B's",
        "# shape for series A's P 224 anchor, Phi-3.5 for the droppable replicate). No model form, no gate, no blind:",
        f"# every reading is DESCRIPTIVE. One gpu_1x_gh200 at ${RATE}/h, run by RR's driver (PC2's pinned driver plus",
        "# lines marked RR) through RR's launcher (vm_run_pc2.sh plus lines marked RR):",
        "#   bash scripts/vm_run_rr.sh start --ip <ip> --run-id <id> --deadline <epoch s> \\",
        f"#     --plan {PLAN} --driver gh200_rr_session.sh",
        f"# Every unit at the session lock {SESSION_LOCK} MHz (C3's prelude, 700 W read back; B3's setting), read back after",
        "# each probe and counter tool (pc-locks.env). Line order is the priority; the driver drops the LAST unit first:",
        "# the replicate. A probe=pc regcheck that refuses or crashes, or writes no line for a probe=pc stamps unit,",
        "# drops every probe=pc unit (PC's rule); the counter units still run.",
        f"# MINUTES: {r['units_min']} of units estimated ({r['nodrop_min']} nodrop); wall about {r['wall_min']} min with "
        f"{BOOT_MIN} of boot (${r['cost_usd']:.2f});",
        f"# deadline launch + {r['deadline_after_launch_min']} min; guardian hard cap launch + "
        f"{r['hard_cap_after_launch_min']} min (${r['cost_at_hard_cap_usd']:.2f} at most).",
    ]
    return "\n".join(hdr + [plan_line(u) for u in UNITS]) + "\n"


# ------------------------------------------------------------------ frozen literals printed beside the readings
#: design-P1's seen-page facts the readings are printed beside (DESIGN.md sections 2.2 to 2.4; POST-HOC on SEEN pages,
#: never used by a reading)
DESIGN_PRINTED = {
    "m measured on seen pages (design-P1 2.2)": {"Phi-3.5 128x128": [0.68, 0.77], "Phi-3.5 64x128": [0.23, 0.53],
                                                  "Mixtral-8x7B 128x128": [0.98, 0.99],
                                                  "Mixtral-8x7B 64x128": [0.68, 0.87], "Qwen2-57B 128x128": [0.16, 0.31],
                                                  "tp2 128x128": [0.75, 0.84]},
    "per-pair rule fits on seen stamps (design-P1 2.3)": {"step starts": [42.5, 0.048], "step mid-lives": [45.0, 0.043],
                                                          "random replacement starts": [62.5, 0.079],
                                                          "logistic starts": [52.5, 0.048]},
    "wave-start spread on seen stamps (design-P1 2.4)": ("Phi 128x128: 0.28 at wave 1, 0.41 to 0.47 by waves 2 to 4, then "
                                                         "flat at 0.55 to 0.69; Qwen2 about 0.6 by wave 6; Phi 64x128 "
                                                         "0.08 to 0.17 for three waves, then 0.34 to 0.54"),
}
#: the stand-in's RR-R3 numbers on SEEN pages (read_rr.standin over results/published at registration, rounded to 4
#: places; FROZEN, recomputed by tests/test_read_rr.py): printed beside RR-R3's numbers, never used by a reading
STANDIN_WAVE = {
    "instr-mixtral-8x7b-st7": {"k_90": 7, "slope_flat": 0.0209, "status": "READ", "w_inf": 0.4192, "waves": 16},
    "instr-phi-3.5-moe-se1": {"k_90": 5, "slope_flat": 0.0167, "status": "READ", "w_inf": 0.5808, "waves": 15},
    "instr-phi-3.5-moe-se2": {"status": "NOT IDENTIFIED: 1 waves read at k >= 5 (needs 2)", "waves": 6},
    "instr-qwen2-57b-a14b-stq": {"k_90": 6, "slope_flat": 0.0081, "status": "READ", "w_inf": 0.5792, "waves": 24},
}

#: design-P1's blind candidates (section 3; no blind follows, owner 2026-11-11): RR's GEMMs avoid theirs anyway, as a
#: printed fact checked by a test, never a refusal
CANDIDATE_MODELS = ("mixtral-8x7b-tp4", "mixtral-8x7b-tp8", "deepseek-v3", "deepseek-v3-tp4", "deepseek-v3-tp8",
                    "granite-3.0-3b-a800m", "granite-3.0-1b-a400m", "qwen2-57b-a14b-tp8")


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()
