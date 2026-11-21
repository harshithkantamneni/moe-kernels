"""Rental PC2's pure rules (docs/registered/2026-11-10-pc2-gh200): one module for the probe (scripts/instr_probe_pc2.py),
the page generator, the registration and the reader, so a rule is computed one way on every side. It imports PC's
rules module (scripts/scoring/pc/pccommon.py, pinned, unchanged) and moe.spec; no torch, no vLLM.

PC2 is a MEASUREMENT rental (owner, 2026-11-10: run PC2 before building v6d's addendum 6; v6d stays as frozen, with the
duty factor; Granite-1B stays in the blind). It measures what v6d cannot yet price (design-D6 section 7), on SEEN or
SYNTHETIC shapes only: the 16x64x128 decode tile, short slabs of 66 to 459 KB, single waves of 165/198 to 528 CTAs on the
4-per-SM tiles, and naive-path pages. Every reading is DESCRIPTIVE; it registers no model form and no constant.

What PC2 takes from PC unchanged: the board (700 W, the 1410 MHz session lock, 1710 MHz unit locks), the versions rule,
the reserved models, ROW SANITY's ceilings, the stamped-call rule, the loads-in-flight reading off the PTX. What is new
here: PC2's shapes (PC's own shapes by the same names, and pcs2- shapes), PC2's tile list, the naive path (vLLM's T k 4
<= E assignment, refused by PC, run here on cells that declare it), a stricter shape guard (no PC2 GEMM equals a
reserved model's GEMM), and PC2's pages, plan and figures.

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
_PC = str(HERE.parent / "pc")
if _PC not in sys.path:
    sys.path.insert(0, _PC)
import pccommon as PC  # noqa: E402

DATE = "2026-11-10"
NAME = f"{DATE}-pc2-gh200"
SHAPES_FILE = f"docs/registered/{DATE}-pc2-shapes.json"
PAGES_DIR = f"docs/registered/{DATE}-pc2-pages"
PLAN = "scripts/plans/rentalPC2-2026-11.plan"
DRIVER = "scripts/gh200_pc2_session.sh"
PARENT_DRIVER = PC.DRIVER
PROBE = "scripts/instr_probe_pc2.py"
PARENT_PROBE = PC.PROBE
LAUNCHER = "scripts/vm_run_pc2.sh"
PARENT_LAUNCHER = "scripts/vm_run.sh"
CARD = PC.CARD
SCHEMA = "moe-kernels/pc2-page/1"
SHAPES_SCHEMA = "moe-kernels/pc2-shapes/1"
SHUFFLE_SEED = 20261110
LABEL_RE = PC.LABEL_RE

# ------------------------------------------------------------------ the board (PC's, unchanged)
POWER_W = PC.POWER_W
SESSION_LOCK = PC.SESSION_LOCK
DECODE_LOCK = PC.DECODE_LOCK
LOCKS = PC.LOCKS
SMS = PC.SMS
F_BAND = PC.F_BAND
RATE_CEILING_GBPS = PC.RATE_CEILING_GBPS
F_CEILING_MHZ = PC.F_CEILING_MHZ
TRIAD_GBPS = PC.TRIAD_GBPS
VERSIONS = PC.VERSIONS
VERSIONS_MAJOR_MINOR = PC.VERSIONS_MAJOR_MINOR

# ------------------------------------------------------------------ models: allowed, reserved
#: one class with PC's, so the probe's refusals and the pinned modules' catch the same exception
Refused = PC.Refused
#: owner decision (2026-11-06, kept 2026-11-10): never loaded, never timed, never shaped
RESERVED = PC.RESERVED
#: the real models PC2 may load, all SEEN: Mixtral-8x7B (CAL2, DS, PC), and the five models of the published decode
#: blind whose naive or decode-tile cells PC2 times (JetMoE-8B, Phi-3.5-MoE, Granite-3.0-3B, Qwen1.5-MoE, DeepSeek-V2-Lite)
REAL_ALLOWED = ("mixtral-8x7b", "jetmoe-8b", "phi-3.5-moe", "granite-3.0-3b-a800m", "qwen1.5-moe-a2.7b",
                "deepseek-v2-lite")

#: PC's registered shapes PC2 runs again, by the same names and the same (E, H, F, k) (a test holds them to PC's file)
PC_SHAPES = ("pcs-e8-h128", "pcs-e8-h256", "pcs-e8-h512", "pcs-e8-h1024", "pcs-e8-h2048", "pcs-e32-h4096")


def _s(name, E, H, F, k=2):
    return {"name": name, "E": E, "H": H, "F": F, "k": k}


#: PC2's synthetic shapes. pcs- names are PC's (docs/registered/2026-11-06-pc-shapes.json); pcs2- names are new:
#:   pcs2-e8-h256-f*     the 16x64x128 F ladder at G 8 (2 experts; w2 slab F x 64 x 2: 66 KB to 918 KB) and the
#:                       64x128x64 short-slab ladder at G 8 (4 experts; w2 slab F x 128 x 2: 66 KB to 459 KB)
#:   pcs2-e32-h4096-f*   the 16x64x128 F ladder at G 128 and the 64x128x64 short-slab ladders at G 128 and 256
#:   pcs2-e8-h2048-f*    the 64x128x64 short-slab ladder at G 32
#:   pcs2-e32-h4224*     the 4-per-SM tiles: H 4224 gives 66 N-tiles at BLOCK_N 64 and 33 at 128, so A experts land on
#:                       G 264, 396, 462, 528 (one wave of 132 x 4) exactly, and w1's K is past 4096 (X_sat's w1 rule)
#:   pcs2-e64-h2048-f1408-k8  a k 8, E 64 naive shape on a SEEN model's GEMM dims (DeepSeek-V2-Lite's H and F at k 8)
SHAPES = tuple(s for s in PC.SHAPES if s["name"] in PC_SHAPES) + (
    *(_s(f"pcs2-e8-h256-f{f}", 8, 256, f) for f in (256, 512, 1024, 1792, 3584, 7168)),
    *(_s(f"pcs2-e32-h4096-f{f}", 32, 4096, f) for f in (256, 512, 1024, 1792, 3584, 7168)),
    *(_s(f"pcs2-e8-h2048-f{f}", 8, 2048, f) for f in (256, 512, 1024, 1792)),
    _s("pcs2-e32-h4224", 32, 4224, 14336), _s("pcs2-e32-h4224-f512", 32, 4224, 512),
    _s("pcs2-e32-h4224-f1024", 32, 4224, 1024),
    _s("pcs2-e64-h2048-f1408-k8", 64, 2048, 1408, 8),
)
SHAPE_PREFIXES = ("pcs-", "pcs2-")

shape_key = PC.shape_key
model_shapes = PC.model_shapes
reserved_shapes = PC.reserved_shapes


def gemm_dims(cfg, gemm: str) -> tuple[int, int]:
    """(N, K) of a GEMM: w1 N 2F, K H; w2 N H, K F (PC's rule)."""
    return PC.gemm_dims(cfg, gemm)


def reserved_gemms() -> dict:
    """{(N, K): (reserved model, its GEMM)} for every reserved model's two GEMMs."""
    from moe.spec import MODEL_CONFIGS
    out = {}
    for m in RESERVED:
        for g in ("w1", "w2"):
            out.setdefault(tuple(gemm_dims(MODEL_CONFIGS[m], g)), (m, g))
    return out


def gemm_guard(name: str, cfg) -> None:
    """PC2's stricter guard (registered): no shape PC2 loads, synthetic or real, has a GEMM (w1 or w2) whose (N, K)
    equals either GEMM of a reserved model, so no PC2 launch has a reserved model's GEMM dimensions under another name."""
    res = reserved_gemms()
    for g in ("w1", "w2"):
        hit = res.get(tuple(gemm_dims(cfg, g)))
        if hit:
            raise Refused(f"{name}: its {g} GEMM (N, K) {gemm_dims(cfg, g)} is {hit[0]}'s {hit[1]} GEMM "
                          "(owner decision: reserved)")


def check_shapes_doc(doc: dict) -> dict:
    """The registered shapes file: {name: MoEConfig}. Refuses a name of MODEL_CONFIGS, a name without a pcs- or pcs2-
    prefix, a pcs- name whose shape is not PC's registered one, any shape whose (E, H, F, k) equals a MODEL_CONFIGS
    entry's, and any shape with a reserved model's GEMM (gemm_guard)."""
    from moe.spec import MODEL_CONFIGS, MoEConfig
    if not isinstance(doc, dict) or doc.get("schema") != SHAPES_SCHEMA:
        raise Refused(f"the shapes file's schema is {doc.get('schema') if isinstance(doc, dict) else '?'}, not {SHAPES_SCHEMA}")
    real = model_shapes()
    pc = {s["name"]: shape_key(s["E"], s["H"], s["F"], s["k"]) for s in PC.SHAPES}
    out = {}
    for s in doc.get("shapes") or []:
        name = s.get("name")
        if not isinstance(name, str) or not name.startswith(SHAPE_PREFIXES) or not LABEL_RE.match(name):
            raise Refused(f"synthetic shape {name!r}: a pcs- or pcs2- name")
        if name in MODEL_CONFIGS or name in out:
            raise Refused(f"synthetic shape {name}: the name is a model's, or given twice")
        try:
            key = shape_key(s["E"], s["H"], s["F"], s["k"])
        except (KeyError, TypeError, ValueError):
            raise Refused(f"synthetic shape {name}: E, H, F and k are whole numbers") from None
        if name.startswith("pcs-") and pc.get(name) != key:
            raise Refused(f"synthetic shape {name}: a pcs- name is PC's shape, and PC registered {pc.get(name)}")
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
            "what": ("PC2's synthetic expert shapes (owner, 2026-11-10): built in the probe process as moe.spec.MoEConfig, "
                     "never added to moe/spec.py; pcs- names are PC's registered shapes (2026-11-06), pcs2- names are new; "
                     "no shape equals any MODEL_CONFIGS entry's (E, H, F, k), and no GEMM of any shape equals a reserved "
                     "model's GEMM"),
            "shapes": [dict(s) for s in SHAPES]}


def load_shapes(path: Path | None = None) -> dict:
    p = Path(path) if path else REPO / SHAPES_FILE
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read the shapes file {p}: {exc}") from None
    return check_shapes_doc(doc)


def shape_cfg(name: str, shapes: dict | None = None):
    """The MoEConfig of a cell's shape: a registered synthetic shape or an allowed SEEN model. Refuses every reserved
    model, any shape equal to a reserved model's, and any shape with a reserved model's GEMM (the unseen-model guard)."""
    from moe.spec import MODEL_CONFIGS
    if name in RESERVED:
        raise Refused(f"{name} is reserved (owner decision): PC2 never loads it")
    if name.startswith(SHAPE_PREFIXES):
        shapes = load_shapes() if shapes is None else shapes
        if name not in shapes:
            raise Refused(f"synthetic shape {name} is not in the registered shapes file")
        cfg = shapes[name]
    elif name in REAL_ALLOWED:
        cfg = MODEL_CONFIGS[name]
    else:
        raise Refused(f"model {name!r} is not one PC2 may load ({', '.join(REAL_ALLOWED)} or a registered shape)")
    if shape_key(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k) in reserved_shapes():
        raise Refused(f"{name}'s shape is a reserved model's (owner decision)")
    gemm_guard(name, cfg)
    return cfg


# ------------------------------------------------------------------ tiles
D16 = PC.D16                      # 16x64x128 w4 s4: vLLM's T <= 32 decode tile (mma.sync)
D16S3 = (16, 64, 128, 4, 3)
D64 = PC.D64                      # 64x128x64 w4 s3
W64 = (32, 64, 128, 4, 3)         # 32x64x128 w4 s3: 4 CTAs per SM
W128 = (32, 128, 64, 4, 3)        # 32x128x64 w4 s3: 4 CTAs per SM
TILES = {"16x64x128w4s4": D16, "16x64x128w4s3": D16S3, "64x128x64w4s3": D64, "32x64x128w4s3": W64,
         "32x128x64w4s3": W128}
CENSUS_TILES = tuple(TILES)
CENSUS_MODELS = PC.CENSUS_MODELS
#: the registered occupancy (CTAs per SM), a frozen literal COMPUTED from the published censuses: 16x64x128w4s4 3,
#: 64x128x64w4s3 3, 32x64x128w4s3 4, 32x128x64w4s3 4 (DS's census, 2026-11-04 decodeDS session, and CAL2's), and
#: 16x64x128w4s3 5 (PC's census, 2026-11-06). The reader requires the board's census to read the same, else the
#: wave-dependent readings are NOT IDENTIFIED
OCCUPANCY = {"16x64x128w4s4": 3, "16x64x128w4s3": 5, "64x128x64w4s3": 3, "32x64x128w4s3": 4, "32x128x64w4s3": 4}
#: the naive path's tile: vLLM's T <= 32 default (every naive cell of the published decode blind ran on it)
NAIVE_TILE = "16x64x128w4s4"

#: PC's own regcheck page on the decode tile (2026-11-06, regcheck-rcp/regcheck.json, a frozen literal recomputed by a
#: test): per config and GEMM, the plain kernel's CTAs per SM and the stamped copy's. s4 and s3 keep the plain occupancy
#: on both GEMMs; s2's w2 does not (8 against 7), which refused PC's block D. PC2 stamps s4 (and s3) in variants of their
#: own and leaves s2 out (design-D6 section 7 item 1)
PC_D16_REGCHECK = {"16x64x128w4s2": {"w1": (8, 8), "w2": (8, 7)}, "16x64x128w4s3": {"w1": (5, 5), "w2": (5, 5)},
                   "16x64x128w4s4": {"w1": (3, 3), "w2": (3, 3)}}

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
naive = PC.naive


def cell_T(cfg, counts) -> int:
    return sum(int(c) for c in counts) // int(cfg.top_k)


def is_naive(cfg, counts) -> bool:
    return naive(cell_T(cfg, counts), cfg.num_experts, cfg.top_k)


def live_ctas(cfg, counts, tile, gemm: str) -> int:
    """Live CTAs of one GEMM: on the aligned path sum over experts of ceil(c_e / BM) x ceil(N / BN) (PC's); on the naive
    path one M-tile per (token, expert) row, T k x ceil(N / BN) (r3_decode.geometry's naive rule)."""
    if is_naive(cfg, counts):
        N, _K = gemm_dims(cfg, gemm)
        return sum(int(c) for c in counts) * (-(-N // int(tile[1])))
    return PC.live_ctas(cfg, counts, tile, gemm)


def slab_once(cfg, counts) -> bool:
    """Every B slab read by exactly one CTA: every aligned PC2 cell (one M-tile per expert), and a naive cell whose
    experts each take one (token, expert) row. A naive cell with an expert on two or more rows has followers: their
    slabs are re-reads (L2 or DRAM), so X is a demand rate there and the L2 guard does not apply."""
    return not is_naive(cfg, counts) or max(int(c) for c in counts) <= 1


def slab_bytes(cfg, tile, gemm: str) -> int:
    return PC.slab_bytes(cfg, tile, gemm)


STAMP_CALLS, STAMP_CALLS_SMALL, SMALL_G = PC.STAMP_CALLS, PC.STAMP_CALLS_SMALL, PC.SMALL_G
stamp_calls = PC.stamp_calls
counts_for = PC.counts_for


def _cell(label, shape, counts, tile, role, *, path="aligned", twin=None, pair=None, flush=True, read="w2"):
    return {"label": label, "shape": shape, "T": None, "counts": [int(x) for x in counts], "config": config_of(tile),
            "group_m": 1, "flush": bool(flush), "read": read, "role": role, "path": path,
            **({"twin": twin} if twin else {}), **({"pair": pair} if pair else {})}


def _E_k(shape):
    from moe.spec import MODEL_CONFIGS
    shp = {s["name"]: s for s in SHAPES}
    if shape in shp:
        return shp[shape]["E"], shp[shape]["k"]
    return MODEL_CONFIGS[shape].num_experts, MODEL_CONFIGS[shape].top_k


def _c(label, shape, A, c_e, tile, role, **kw):
    E, k = _E_k(shape)
    c = _cell(label, shape, counts_for(E, A, c_e), tile, role, **kw)
    c["T"] = sum(c["counts"]) // k
    return c


def _n(label, shape, counts, role, **kw):
    """A naive-block cell (16x64x128 w4 s4) with explicit counts: the first len(counts) experts."""
    E, k = _E_k(shape)
    full = list(counts) + [0] * (E - len(counts))
    c = _cell(label, shape, full, D16, role, **kw)
    c["T"] = sum(full) // k
    return c


E32 = "pcs-e32-h4096"


def block_d() -> list[dict]:
    """Block D (item 1): the decode tile 16x64x128 w4 s4 at 1710 MHz, c_e 16 (one full M-tile per active expert): the
    long-slab G ladder (F 14336, w2 slab 1.835 MB) from G 4 to G 1024, F ladders at G 8 and G 128 (F 512 to 14336),
    short slabs at G 256, 384 (single wave) and 512, 1024 (past it), and Mixtral T 16 A 2 (SEEN, PC's anchor)."""
    out = [_c(f"D-G{g}", s, 2, 16, D16, "ladder") for g, s in ((4, "pcs-e8-h128"), (8, "pcs-e8-h256"), (16, "pcs-e8-h512"),
                                                              (32, "pcs-e8-h1024"), (64, "pcs-e8-h2048"))]
    out += [_c(f"D-G{64 * a}", E32, a, 16, D16, "ladder") for a in (2, 3, 4, 5, 6, 8, 16)]
    out += [_c(f"D-G8-F{f}", f"pcs2-e8-h256-f{f}", 2, 16, D16, "F ladder G 8") for f in (512, 1024, 1792, 3584, 7168)]
    out += [_c(f"D-G128-F{f}", f"pcs2-e32-h4096-f{f}", 2, 16, D16, "F ladder G 128")
            for f in (512, 1024, 1792, 3584, 7168)]
    out += [_c(f"D-G{64 * a}-F{f}", f"pcs2-e32-h4096-f{f}", a, 16, D16, "short slab under load")
            for a, f in ((4, 512), (4, 1024), (6, 512), (6, 1024), (8, 1024), (16, 3584))]
    out.append(_c("D-MX-T16-A2", "mixtral-8x7b", 2, 16, D16, "anchor (SEEN; PC's D-MX-T16-A2)"))
    return out


def block_d3() -> list[dict]:
    """Block D3 (optional): the decode tile at s3 (PC's regcheck page keeps its occupancy, 5 = 5), the Q_B factorial."""
    return [_c("D3-G16", "pcs-e8-h512", 2, 16, D16S3, "factorial s3"),
            _c("D3-G128", E32, 2, 16, D16S3, "factorial s3"), _c("D3-G384", E32, 6, 16, D16S3, "factorial s3")]


def block_s() -> list[dict]:
    """Block S (item 2): short slabs on 64x128x64 w4 s3, c_e 64: F 256, 512, 1024, 1792 (w2 slab 66, 131, 262, 459 KB)
    and F 14336 (3.67 MB, the same-board long anchor) at G 8 (H 256, 4 experts), 32 (H 2048, 2), 128 and 256 (H 4096, 4
    and 8): c0 and the span at short slabs, low load. One H per G, so each ladder moves the slab alone."""
    out = []
    for g, (h_shape, a) in ((8, ("pcs2-e8-h256", 4)), (32, ("pcs2-e8-h2048", 2)), (128, ("pcs2-e32-h4096", 4)),
                            (256, ("pcs2-e32-h4096", 8))):
        long_shape = {8: "pcs-e8-h256", 32: "pcs-e8-h2048"}.get(g, E32)
        for f in (256, 512, 1024, 1792):
            out.append(_c(f"S-G{g}-F{f}", f"{h_shape}-f{f}", a, 64, D64, f"short-slab ladder G {g}"))
        out.append(_c(f"S-G{g}", long_shape, a, 64, D64, f"short-slab ladder G {g} (long anchor)"))
    return out


def block_w() -> list[dict]:
    """Block W (item 3): the 4-per-SM tiles, c_e 32, on pcs2-e32-h4224 (66 N-tiles at BLOCK_N 64, 33 at 128): single
    waves from 264 (198) to 528 CTAs (132 x 4), one rung past a wave (594) and two waves (1056); F 14336 (long) and
    F 1024 and 512 (short, down to 198 CTAs on 32x64x128 and 165 on 32x128x64, under Granite-1B's 256 and 192)."""
    out = []
    for a in (4, 6, 7, 8, 9, 16):
        out.append(_c(f"W64-G{66 * a}", "pcs2-e32-h4224", a, 32, W64, "4/SM 32x64x128"))
    for f in (1024, 512):
        out += [_c(f"W64-G{66 * a}-F{f}", f"pcs2-e32-h4224-f{f}", a, 32, W64, "4/SM 32x64x128 short")
                for a in (3, 4, 6, 8)]
    for a in (6, 8, 12, 14, 16, 18, 32):
        out.append(_c(f"W128-G{33 * a}", "pcs2-e32-h4224", a, 32, W128, "4/SM 32x128x64"))
    out += [_c(f"W128-G{33 * a}-F1024", "pcs2-e32-h4224-f1024", a, 32, W128, "4/SM 32x128x64 short")
            for a in (5, 6, 8, 12, 16)]
    out += [_c(f"W128-G{33 * a}-F512", "pcs2-e32-h4224-f512", a, 32, W128, "4/SM 32x128x64 short") for a in (5, 6, 8)]
    return out


def block_n() -> list[dict]:
    """Block N (item 4): naive-path cells (T k 4 <= E: one M-tile per (token, expert) row, no align kernel) at vLLM's
    T <= 32 tile, on the SEEN models whose naive cells the published decode blind holds and one synthetic k 8 shape.
    Each naive cell without followers has an ALIGNED TWIN (same shape and tile, the same active experts, the same live
    CTAs on both GEMMs, one M-tile per expert); each naive cell WITH followers (an expert on two rows) has a naive PAIR
    with the same live CTAs and no follower (twice the distinct experts)."""
    nv = "naive"
    out = [
        _n("N-MX-T1", "mixtral-8x7b", [1, 1], "naive", path=nv, twin="N-MX-T2"),
        _n("N-MX-T2", "mixtral-8x7b", [2, 2], "aligned twin"),
        _n("N-JM-T1", "jetmoe-8b", [1, 1], "naive", path=nv, twin="N-JM-T2"),
        _n("N-JM-T2", "jetmoe-8b", [2, 2], "aligned twin"),
        _n("N-PH-T1", "phi-3.5-moe", [1, 1], "naive", path=nv, twin="N-PH-T3-A2"),
        _n("N-PH-T2-A4", "phi-3.5-moe", [1, 1, 1, 1], "naive", path=nv, twin="N-PH-T3-A4"),
        _n("N-PH-T2-A2", "phi-3.5-moe", [2, 2], "naive with followers", path=nv, pair="N-PH-T2-A4"),
        _n("N-PH-T3-A2", "phi-3.5-moe", [3, 3], "aligned twin"),
        _n("N-PH-T3-A4", "phi-3.5-moe", [2, 2, 1, 1], "aligned twin"),
        _n("N-G3-T1", "granite-3.0-3b-a800m", [1] * 8, "naive", path=nv, twin="N-G3-T2"),
        _n("N-G3-T2", "granite-3.0-3b-a800m", [2] * 8, "aligned twin"),
        _n("N-QW-T1", "qwen1.5-moe-a2.7b", [1] * 4, "naive", path=nv, twin="N-QW-T4"),
        _n("N-QW-T2-A8", "qwen1.5-moe-a2.7b", [1] * 8, "naive", path=nv, twin="N-QW-T4-A8"),
        _n("N-QW-T2-A4", "qwen1.5-moe-a2.7b", [2] * 4, "naive with followers", path=nv, pair="N-QW-T2-A8"),
        _n("N-QW-T4", "qwen1.5-moe-a2.7b", [4] * 4, "aligned twin"),
        _n("N-QW-T4-A8", "qwen1.5-moe-a2.7b", [2] * 8, "aligned twin"),
        _n("N-DS-T1", "deepseek-v2-lite", [1] * 6, "naive", path=nv, twin="N-DS-T3"),
        _n("N-DS-T2-A12", "deepseek-v2-lite", [1] * 12, "naive", path=nv, twin="N-DS-T3-A12"),
        _n("N-DS-T2-A6", "deepseek-v2-lite", [2] * 6, "naive with followers", path=nv, pair="N-DS-T2-A12"),
        _n("N-DS-T3", "deepseek-v2-lite", [3] * 6, "aligned twin"),
        _n("N-DS-T3-A12", "deepseek-v2-lite", [2] * 6 + [1] * 6, "aligned twin"),
        _n("N-K8-T1", "pcs2-e64-h2048-f1408-k8", [1] * 8, "naive", path=nv, twin="N-K8-T3"),
        _n("N-K8-T2-A16", "pcs2-e64-h2048-f1408-k8", [1] * 16, "naive", path=nv, twin="N-K8-T3-A16"),
        _n("N-K8-T2-A8", "pcs2-e64-h2048-f1408-k8", [2] * 8, "naive with followers", path=nv, pair="N-K8-T2-A16"),
        _n("N-K8-T3", "pcs2-e64-h2048-f1408-k8", [3] * 8, "aligned twin"),
        _n("N-K8-T3-A16", "pcs2-e64-h2048-f1408-k8", [2] * 8 + [1] * 8, "aligned twin"),
    ]
    return out


def block_r() -> list[dict]:
    """Block R (optional): D-G8, D-G128 and S-G128-F512 again at the end (page noise)."""
    keep = {"D-G8": "R-D-G8", "D-G128": "R-D-G128", "S-G128-F512": "R-S-G128-F512"}
    return [dict(c, label=keep[c["label"]], role="replicate") for c in block_d() + block_s() if c["label"] in keep]


#: page -> (block letter, builder, lock)
BLOCKS = {"pc2-d": ("D", block_d, DECODE_LOCK), "pc2-s": ("S", block_s, DECODE_LOCK), "pc2-n": ("N", block_n, DECODE_LOCK),
          "pc2-w": ("W", block_w, DECODE_LOCK), "pc2-d3": ("D3", block_d3, DECODE_LOCK),
          "pc2-r": ("R", block_r, DECODE_LOCK)}


def page_doc(page: str) -> dict:
    blk, fn, lock = BLOCKS[page]
    return {"schema": SCHEMA, "page": page, "block": blk, "lock_mhz": lock, "shuffle_seed": SHUFFLE_SEED,
            "shapes_file": SHAPES_FILE, "registration": NAME,
            "label": (f"SYNTHETIC routings (scripts/scoring/pc2/gen_pc2_pages.py, {NAME}): PC2 block {blk}, SEEN or "
                      "synthetic shapes only; DESCRIPTIVE"),
            "cells": fn()}


def page_files() -> dict:
    """{published file name under PAGES_DIR: text} (sorted keys, indent 1)."""
    return {f"{p}.json": json.dumps(page_doc(p), indent=1, sort_keys=True) + "\n" for p in BLOCKS}


def shapes_text() -> str:
    return json.dumps(shapes_doc(), indent=1, sort_keys=True) + "\n"


def check_cell(c: dict, shapes: dict, tiles: dict = TILES) -> dict:
    """One PC2 page cell checked as the card would meet it; returns it with its cfg and tile. PC's refusals (an unknown
    or reserved shape, counts that are not E whole numbers summing to T k, a config not in the tile list, an expert
    past BLOCK_M, flush not a bool, read not w1 or w2, a label off the alphabet, G 1 only), and PC2's: the cell's
    declared path must be the one vLLM takes (naive iff T k 4 <= E), and a naive cell runs at the naive tile only."""
    from moe.routing.distributions import feasible
    label = c.get("label")
    if not isinstance(label, str) or not LABEL_RE.match(label):
        raise Refused(f"cell label {label!r}: letters, digits and _ . - only")
    cfg = shape_cfg(str(c.get("shape")), shapes)
    E, k = cfg.num_experts, cfg.top_k
    T = c.get("T")
    if not isinstance(T, int) or isinstance(T, bool) or not 1 <= T <= 4096:
        raise Refused(f"cell {label}: T {T!r}")
    counts = c.get("counts")
    if (not isinstance(counts, list) or len(counts) != E
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts)):
        raise Refused(f"cell {label}: counts must be {E} whole numbers")
    ok, why = feasible(counts, T, k)
    if not ok:
        raise Refused(f"cell {label} (V-BIN): {why}")
    path = c.get("path", "aligned")
    if path not in ("naive", "aligned"):
        raise Refused(f"cell {label}: path {path!r}: naive or aligned")
    want = "naive" if naive(T, E, k) else "aligned"
    if path != want:
        raise Refused(f"cell {label}: declares the {path} path, vLLM takes the {want} one (T k 4 <= E is naive)")
    conf = c.get("config") or {}
    try:
        tile = tile_of(conf)
    except (KeyError, TypeError, ValueError):
        raise Refused(f"cell {label}: config {conf!r} is not BLOCK_SIZE_M, N, K, num_warps, num_stages") from None
    if tile_key(tile) not in tiles:
        raise Refused(f"cell {label}: tile {tile_key(tile)} is not one of the registered {sorted(tiles)}")
    if path == "naive" and tile_key(tile) != NAIVE_TILE:
        raise Refused(f"cell {label}: a naive cell runs at {NAIVE_TILE}, vLLM's T <= 32 tile, not {tile_key(tile)}")
    if int(c.get("group_m", 1)) != 1:
        raise Refused(f"cell {label}: GROUP_SIZE_M {c.get('group_m')}: PC2's cells are G 1")
    if max(counts) > tile[0]:
        raise Refused(f"cell {label}: an expert holds {max(counts)} rows, past BLOCK_M {tile[0]} (one M-tile declared)")
    if not isinstance(c.get("flush", True), bool):
        raise Refused(f"cell {label}: flush {c.get('flush')!r}: true or false")
    if c.get("read", "w2") not in ("w1", "w2"):
        raise Refused(f"cell {label}: read {c.get('read')!r}: w1 or w2")
    return {**c, "cfg": cfg, "tile": tile}


def check_partners(cells: list[dict]) -> None:
    """Twins and pairs (block N): a twin is an ALIGNED cell of the same shape and tile with the naive cell's active
    experts and its live CTAs on both GEMMs; a pair is a NAIVE cell of the same shape and tile with the same live CTAs
    on both GEMMs and no follower, partnering a naive cell that has followers. A naive cell names exactly one of them:
    a twin when it has no follower, a pair when it has."""
    by = {c["label"]: c for c in cells}
    for c in cells:
        if c.get("path") != "naive":
            if "twin" in c or "pair" in c:
                raise Refused(f"cell {c['label']}: only a naive cell names a twin or a pair")
            continue
        foll = not slab_once(c["cfg"], c["counts"])
        key = "pair" if foll else "twin"
        other = "twin" if foll else "pair"
        if other in c or key not in c:
            raise Refused(f"cell {c['label']}: a naive cell {'with' if foll else 'without'} followers names a {key}")
        p = by.get(c[key])
        if p is None:
            raise Refused(f"cell {c['label']}: its {key} {c[key]!r} is not on the page")
        if p["shape"] != c["shape"] or p["tile"] != c["tile"]:
            raise Refused(f"cell {c['label']}: its {key} {p['label']} is another shape or tile")
        lv = [live_ctas(c["cfg"], c["counts"], c["tile"], g) for g in ("w1", "w2")]
        lp = [live_ctas(p["cfg"], p["counts"], p["tile"], g) for g in ("w1", "w2")]
        if lv != lp:
            raise Refused(f"cell {c['label']}: its {key} {p['label']} has live CTAs {lp}, not {lv}")
        if key == "twin":
            if p.get("path") != "aligned":
                raise Refused(f"cell {c['label']}: its twin {p['label']} is not on the aligned path")
            if [e for e, x in enumerate(c["counts"]) if x] != [e for e, x in enumerate(p["counts"]) if x]:
                raise Refused(f"cell {c['label']}: its twin {p['label']} has other active experts")
        else:
            if p.get("path") != "naive" or not slab_once(p["cfg"], p["counts"]):
                raise Refused(f"cell {c['label']}: its pair {p['label']} is not a naive cell without followers")


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
    check_partners(out)
    return out


# ------------------------------------------------------------------ the readings' registered constants
#: PC's, unchanged (pccommon): the plateau, the per-call spread, the F ladder's SE rule, the sharing groups, the start
#: window, C3's OD-1 NOT-READ bound, the timer rule, the L2 guard (L2_LATENCY_NS, the tile's s4 loads in flight)
PLATEAU_FRAC, MIN_PLATEAU_CTAS = PC.PLATEAU_FRAC, PC.MIN_PLATEAU_CTAS
SPREAD_MAX, SPREAD_MAX_SMALL, SPREAD_SMALL_G = PC.SPREAD_MAX, PC.SPREAD_MAX_SMALL, PC.SPREAD_SMALL_G
F_RUNGS, SE_INV_RINF_MAX = PC.F_RUNGS, PC.SE_INV_RINF_MAX
OVERHEAD_MEDIAN_MAX, OVERHEAD_WORST_MAX = PC.OVERHEAD_MEDIAN_MAX, PC.OVERHEAD_WORST_MAX
L2_LATENCY_NS, L2_GUARD_STAGES = PC.L2_LATENCY_NS, PC.L2_GUARD_STAGES
START_WINDOW_NS, START_FLAG, START_MIN_CTAS = PC.START_WINDOW_NS, PC.START_FLAG, PC.START_MIN_CTAS
SAT_WITHIN = PC.SAT_WITHIN
X_SAT_W1_MIN_K = PC.X_SAT_W1_MIN_K
#: X_sat's w2 members are long-slab rungs only (PC's: every PC X_sat w2 member had F 14336), strictly past one wave
X_SAT_W2_MIN_K = 14336
#: v6d term 3: kappa(528) / kappa(396) reads HELD within this
KAPPA_HELD_WITHIN = 0.03
#: PC2's own (registered before any page): the span reading is read on single-wave launches only (live CTAs at most
#: the tile's registered wave), and each (cell, GEMM) needs at least this many valid launches for it
SPAN_MIN_LAUNCHES = 2
#: PC2's timer rule: the unit's globaltimer resolution is read off at least this many of its launches (and holds half)
TIMER_UNIT_MIN = 10
#: the per-call spread of the span ratio a (cell, GEMM) must keep (as the rate's: 3%, 5% at G <= 8)
#: a twin or pair ratio reads only when both cells' plain or span medians are read
#: PC2-R3: G 528 reads SATURATED when its X is within SAT_WITHIN (3%) of the tile's X_sat


# ------------------------------------------------------------------ the plan
tile_keys = PC.tile_keys
plan_line = PC.plan_line

UNITS = (
    ("-", "prelude", "", f"lock-mhz={SESSION_LOCK} power-limit-w={POWER_W}", 2, 15, "nodrop"),
    ("-", "tilecensus", "pc2", f"tiles={','.join(CENSUS_TILES)} models={','.join(CENSUS_MODELS)}", 3, 15, "nodrop"),
    ("-", "regcheck", "rcp", "probe=pc", 3, 20, "nodrop"),
    ("-", "perturb", "pt", f"probe=pc gates=sd,ss,sn,sw,sd3,sr lock-mhz={DECODE_LOCK}", 8, 25, "nodrop"),
    ("-", "stamps", "sd", f"probe=pc page={PAGES_DIR}/pc2-d.json lock-mhz={DECODE_LOCK}", 4, 15, "nodrop"),
    ("-", "stamps", "ss", f"probe=pc page={PAGES_DIR}/pc2-s.json lock-mhz={DECODE_LOCK}", 3, 12, "nodrop"),
    ("-", "stamps", "sn", f"probe=pc page={PAGES_DIR}/pc2-n.json lock-mhz={DECODE_LOCK}", 4, 15, "nodrop"),
    ("-", "stamps", "sw", f"probe=pc page={PAGES_DIR}/pc2-w.json lock-mhz={DECODE_LOCK}", 4, 15, "w"),
    ("-", "stamps", "sd3", f"probe=pc page={PAGES_DIR}/pc2-d3.json lock-mhz={DECODE_LOCK}", 2, 8, "d3"),
    ("-", "stamps", "sr", f"probe=pc page={PAGES_DIR}/pc2-r.json lock-mhz={DECODE_LOCK}", 2, 8, "rep"),
)
#: the cut order, last unit first: the replicate, the s3 factorial, then the 4-per-SM block (design-D6 section 7 item
#: 4: "not needed alone"); the decode tile, the short slabs and the naive path are nodrop
CUT_ORDER = ("rep", "d3", "w")
#: every unit the regcheck drop rule drops: every probe=pc unit (PC's owner decision 2, kept)
PC_UNIT_LABELS = ("pt", "sd", "ss", "sn", "sw", "sd3", "sr")
VARIANTS = {lab: f"u{i + 1}-{lab}" for i, u in enumerate(UNITS) for lab in [u[2]] if u[1] == "stamps"}
PAGE_OF = {lab: u[3].split("page=")[1].split()[0] for u in UNITS for lab in [u[2]] if u[1] == "stamps"}
BOOT_MIN, RATE, RESERVE_MIN, DEADLINE_SLACK_MIN, CAP_PAD_MIN = PC.BOOT_MIN, PC.RATE, PC.RESERVE_MIN, \
    PC.DEADLINE_SLACK_MIN, PC.CAP_PAD_MIN


def rental() -> dict:
    """PC's rule on PC2's units: wall = units + 25 min of boot and setup; the deadline the units, boot, the driver's
    8 min reserve and 5 min of slack need, rounded up to 5 min; the guardian's hard cap 15 min later; $2.29/h."""
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
        f"# RENTAL PC2 (registered {DATE}, before any page): what v6d cannot yet price, a MEASUREMENT rental.",
        f"# GENERATED by scripts/scoring/pc2/register_pc2.py from registration {NAME}; do not edit by hand.",
        "# vLLM v0.27.1's fused_moe_kernel through the instrumented copy at the ends level (moe/instrumented, unchanged),",
        "# on synthetic routings and SEEN or synthetic expert shapes; Mixtral-8x22B, Qwen2-57B (and its tp8 shard) and",
        "# Granite-3.0-1B are never loaded, by name or by shape (no PC2 GEMM is theirs). Every reading is DESCRIPTIVE",
        f"# (no model form, no constant). One gpu_1x_gh200 at ${RATE}/h, run by PC2's driver (PC's pinned driver plus lines",
        "# marked PC2: probe=pc runs scripts/instr_probe_pc2.py) through PC2's launcher (vm_run.sh plus lines marked PC2):",
        "#   bash scripts/vm_run_pc2.sh start --ip <ip> --run-id <id> --deadline <epoch s> \\",
        f"#     --plan {PLAN} --driver gh200_pc2_session.sh",
        f"# Session lock {SESSION_LOCK} MHz (C3's prelude, 700 W read back); every probe unit carries lock-mhz={DECODE_LOCK},",
        "# probed by the prelude and read back before and after the tool (pc-locks.env).",
        "# Line order is the priority; the driver drops the LAST unit first: rep, d3, then w. A probe=pc regcheck that",
        "# refuses or crashes, or writes no line for a probe=pc stamps unit, drops every probe=pc unit (PC's rule).",
        f"# MINUTES: {r['units_min']} of units estimated ({r['nodrop_min']} nodrop); wall about {r['wall_min']} min with "
        f"{BOOT_MIN} of boot (${r['cost_usd']:.2f});",
        f"# deadline launch + {r['deadline_after_launch_min']} min; guardian hard cap launch + "
        f"{r['hard_cap_after_launch_min']} min (${r['cost_at_hard_cap_usd']:.2f} at most).",
    ]
    return "\n".join(hdr + [plan_line(u) for u in UNITS]) + "\n"


# ------------------------------------------------------------------ frozen literals the coverage test reads
#: Granite-3.0-1B's aligned GEMMs per tile (design-D6 work/reach6.out, COMPUTED 2026-11-09 from its config and the
#: registered blind5 page lists, no page of the model; FROZEN): (n min, n max, slab min KB, slab max KB)
GRANITE1B_REACH = {"16x64x128w4s4": (128, 512, 66, 131), "32x128x64w4s3": (192, 272, 131, 262),
                   "32x64x128w4s3": (256, 512, 66, 131), "64x128x64w4s3": (128, 256, 131, 262)}
#: the published decode blind's naive cells by SEEN model (design-D6 work/rescore_detail.out, FROZEN)
BLIND_NAIVE_CELLS = {"phi-3.5-moe": 6, "qwen1.5-moe-a2.7b": 12, "jetmoe-8b": 2, "deepseek-v2-lite": 7,
                     "granite-3.0-3b-a800m": 2}
#: v6d's frozen values beside which PC2's readings are printed (design-D6 DESIGN.md section 2, FROZEN.txt): printed,
#: never used by a reading
V6D_PRINTED = {"c0_us": 4.704722, "X_sat_64x128x64w4s3_gbps": 3612.42, "kappa_G384": 1.032,
               "X_sat_by_tile_gbps": {"16x64x128w4s4": 3717.9, "32x64x128w4s3": 3703.8, "32x128x64w4s3": 3694.4},
               "n_if": {"16x64x128w4s4": 3, "32x64x128w4s3": 2, "32x128x64w4s3": 2, "64x128x64w4s3": 2}}

#: PC's frozen literal, unchanged (pccommon): the models still unseen at vLLM's prefill tiles; PC2 runs none of them at
#: a prefill tile (Granite-3.0-3B runs at the decode tile only, in block N)
UNSEEN_AT_PREFILL_6C800F84 = PC.UNSEEN_AT_PREFILL_6C800F84


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()
