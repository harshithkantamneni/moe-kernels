"""Rental PC's pure rules (docs/registered/2026-11-06-pc-gh200): one module for the probe (scripts/instr_probe_pc.py),
the page generator, the registration and the reader, so a rule is computed one way on every side. It imports nothing
heavier than moe.spec (and numpy inside the PTX and stamp helpers' callers): no torch, no vLLM, no frozen chain.

PC is a MEASUREMENT rental (owner, 2026-11-06): per-CTA DRAM rate against loads in flight, on vLLM v0.27.1's own kernel
through the instrumented copy at the "ends" level (moe/instrumented, unchanged), on synthetic routings and synthetic
or SEEN expert shapes. Every reading is DESCRIPTIVE; it registers no model form and no constant. The one label it
registers (owner decision 4) is H1's SUPPORTED / FALSIFIED / MIXED, printed beside the numbers, DESCRIPTIVE.

Every range clause is written to fail on a NaN (`not lo <= x <= hi`, math.isfinite).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

DATE = "2026-11-06"
NAME = f"{DATE}-pc-gh200"
SHAPES_FILE = f"docs/registered/{DATE}-pc-shapes.json"
PAGES_DIR = f"docs/registered/{DATE}-pc-pages"
PRICED = f"docs/registered/{DATE}-pc-priced.json"
PLAN = "scripts/plans/rentalPC-2026-11.plan"
DRIVER = "scripts/gh200_pc_session.sh"
PARENT_DRIVER = "scripts/gh200_cal3_session.sh"
PROBE = "scripts/instr_probe_pc.py"
CARD = "nvidia_gh200_480gb"
SCHEMA = "moe-kernels/pc-page/1"
SHAPES_SCHEMA = "moe-kernels/pc-shapes/1"
DECODE_SCHEMA = "moe-kernels/decode-page/1"
SHUFFLE_SEED = 20261106
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# ------------------------------------------------------------------ the board
POWER_W = 700
SESSION_LOCK = 1410
DECODE_LOCK = 1710
LOCKS = (SESSION_LOCK, DECODE_LOCK)
SMS = 132
#: the per-CTA clock band each lock's stamps are read in (design trap 2): A2's fallback band at 1410, rental 6's at 1710
F_BAND = {1410: (1352.19, 1410.0), 1710: (1640.0, 1710.0)}
#: physical ceilings (ROW SANITY): the GH200's HBM3 spec peak (addendum 4's ceiling) and its top SM clock
RATE_CEILING_GBPS = 4022.0
F_CEILING_MHZ = 1980.0
TRIAD_GBPS = 3726.0
#: H1's low-load fallback (owner decision, 2026-11-06, registered before any page): when PC-R7's X_sat is not read, the
#: low-load premise X(G 8) / X_sat is read against the published triad ruler instead, labelled the fallback. Source:
#: scripts/scoring/cal2/read_c2.py TRIAD_A2_GBPS (A2's calibration page; 15 calibrated rentals 3725.6 to 3727.6 GB/s)
TRIAD_FALLBACK_GBPS = 3726.2
TRIAD_FALLBACK_SOURCE = ("scripts/scoring/cal2/read_c2.py TRIAD_A2_GBPS 3726.2 GB/s (A2's calibration page; the 15 "
                         "calibrated rentals read 3725.6 to 3727.6)")
#: the software every page must carry (addendum 4's CAL2 versions; python on major.minor)
VERSIONS = {"python": "3.12.16", "torch": "2.13.0", "triton": "3.7.1", "vllm": "0.27.1"}
VERSIONS_MAJOR_MINOR = ("python",)

# ------------------------------------------------------------------ models: allowed, reserved
#: owner decision 3 (2026-11-06): never loaded, never timed, never shaped by PC
RESERVED = ("mixtral-8x22b", "qwen2-57b-a14b", "qwen2-57b-a14b-tp8", "granite-3.0-1b-a400m")
#: the real models PC may load (all SEEN): Mixtral-8x7B and its tp shards, JetMoE-8B, Phi-3.5-MoE (SEEN after B3),
#: OLMoE-1B-7B (the f128 sag page only)
REAL_ALLOWED = ("mixtral-8x7b", "mixtral-8x7b-tp2", "mixtral-8x7b-tp4", "mixtral-8x7b-tp8", "jetmoe-8b", "phi-3.5-moe",
                "olmoe-1b-7b")
TOP_K = 2

#: the registered synthetic shapes (k 2; F 14336 is Mixtral-8x7B's, so the w2 slab equals Mixtral's)
SHAPES = (
    {"name": "pcs-e8-h128", "E": 8, "H": 128, "F": 14336, "k": 2},
    {"name": "pcs-e8-h256", "E": 8, "H": 256, "F": 14336, "k": 2},
    {"name": "pcs-e8-h512", "E": 8, "H": 512, "F": 14336, "k": 2},
    {"name": "pcs-e8-h512-f1792", "E": 8, "H": 512, "F": 1792, "k": 2},
    {"name": "pcs-e8-h512-f3584", "E": 8, "H": 512, "F": 3584, "k": 2},
    {"name": "pcs-e8-h512-f7168", "E": 8, "H": 512, "F": 7168, "k": 2},
    {"name": "pcs-e8-h1024", "E": 8, "H": 1024, "F": 14336, "k": 2},
    {"name": "pcs-e8-h2048", "E": 8, "H": 2048, "F": 14336, "k": 2},
    {"name": "pcs-e32-h4096", "E": 32, "H": 4096, "F": 14336, "k": 2},
)


class Refused(Exception):
    """A page, shape, cell or model PC will not run or read."""


def shape_key(E: int, H: int, F: int, k: int) -> tuple:
    return (int(E), int(H), int(F), int(k))


def model_shapes() -> dict:
    """{name: (E, H, F, k)} of every MODEL_CONFIGS entry."""
    from moe.spec import MODEL_CONFIGS
    return {n: shape_key(c.num_experts, c.hidden_size, c.intermediate_size, c.top_k) for n, c in MODEL_CONFIGS.items()}


def reserved_shapes() -> set:
    from moe.spec import MODEL_CONFIGS
    return {shape_key(MODEL_CONFIGS[m].num_experts, MODEL_CONFIGS[m].hidden_size, MODEL_CONFIGS[m].intermediate_size,
                      MODEL_CONFIGS[m].top_k) for m in RESERVED}


def check_shapes_doc(doc: dict) -> dict:
    """The registered shapes file: {name: MoEConfig}. Refuses a name of MODEL_CONFIGS, a name without the pcs- prefix,
    and any shape whose (E, H, F, k) equals a MODEL_CONFIGS entry's (a synthetic shape names no real model), and so a
    reserved model's."""
    from moe.spec import MODEL_CONFIGS, MoEConfig
    if not isinstance(doc, dict) or doc.get("schema") != SHAPES_SCHEMA:
        raise Refused(f"the shapes file's schema is {doc.get('schema') if isinstance(doc, dict) else '?'}, not {SHAPES_SCHEMA}")
    real = model_shapes()
    out = {}
    for s in doc.get("shapes") or []:
        name = s.get("name")
        if not isinstance(name, str) or not name.startswith("pcs-") or not LABEL_RE.match(name):
            raise Refused(f"synthetic shape {name!r}: a pcs- name")
        if name in MODEL_CONFIGS or name in out:
            raise Refused(f"synthetic shape {name}: the name is a model's, or given twice")
        try:
            key = shape_key(s["E"], s["H"], s["F"], s["k"])
        except (KeyError, TypeError, ValueError):
            raise Refused(f"synthetic shape {name}: E, H, F and k are whole numbers") from None
        hit = sorted(m for m, v in real.items() if v == key)
        if hit:
            raise Refused(f"synthetic shape {name}: (E, H, F, k) {key} is {hit[0]}'s shape")
        out[name] = MoEConfig(name=name, hidden_size=key[1], intermediate_size=key[2], num_experts=key[0], top_k=key[3])
    if not out:
        raise Refused("the shapes file holds no shape")
    return out


def shapes_doc() -> dict:
    return {"schema": SHAPES_SCHEMA, "registration": NAME,
            "what": ("PC's synthetic expert shapes (owner decision 3, 2026-11-06): built in the probe process as "
                     "moe.spec.MoEConfig, never added to moe/spec.py; k 2, so the least grid is 2 CTAs; no shape equals "
                     "any MODEL_CONFIGS entry's (E, H, F, k)"),
            "shapes": [dict(s) for s in SHAPES]}


def load_shapes(path: Path | None = None) -> dict:
    p = Path(path) if path else REPO / SHAPES_FILE
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read the shapes file {p}: {exc}") from None
    return check_shapes_doc(doc)


def shape_cfg(name: str, shapes: dict | None = None):
    """The MoEConfig of a cell's shape: a registered synthetic shape or an allowed real model. Refuses every reserved
    model and any shape equal to a reserved model's (the unseen-model guard)."""
    from moe.spec import MODEL_CONFIGS
    if name in RESERVED:
        raise Refused(f"{name} is reserved (owner decision 3): PC never loads it")
    if name.startswith("pcs-"):
        shapes = load_shapes() if shapes is None else shapes
        if name not in shapes:
            raise Refused(f"synthetic shape {name} is not in the registered shapes file")
        cfg = shapes[name]
    elif name in REAL_ALLOWED:
        cfg = MODEL_CONFIGS[name]
    else:
        raise Refused(f"model {name!r} is not one PC may load ({', '.join(REAL_ALLOWED)} or a registered pcs- shape)")
    if shape_key(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k) in reserved_shapes():
        raise Refused(f"{name}'s shape is a reserved model's (owner decision 3)")
    return cfg


# ------------------------------------------------------------------ tiles
#: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
D64 = (64, 128, 64, 4, 3)
D16 = (16, 64, 128, 4, 4)
P64 = (64, 128, 64, 8, 3)
P128 = (128, 128, 64, 8, 3)
#: the registered tile list the PC probe runs (C3's 128x128 runs through C3's probe, unchanged)
TILES = {
    "64x128x64w4s3": D64, "64x128x64w4s2": (64, 128, 64, 4, 2), "64x128x64w4s4": (64, 128, 64, 4, 4),
    "64x128x128w4s2": (64, 128, 128, 4, 2), "64x128x128w4s3": (64, 128, 128, 4, 3), "64x128x64w8s3": P64,
    "16x64x128w4s4": D16, "16x64x128w4s2": (16, 64, 128, 4, 2), "16x64x128w4s3": (16, 64, 128, 4, 3),
}
#: the census unit's tiles: every PC tile and C3's 128x128 (Phi E1); the w8 s3 tile is both D64's warps factorial and
#: Phi E2's tile, so the design's "11 configs" are these 10 distinct ones
CENSUS_TILES = tuple(TILES) + ("128x128x64w8s3",)
CENSUS_MODELS = ("mixtral-8x7b", "phi-3.5-moe")
#: the registered occupancy (CTAs per SM) the wave-dependent readings take, COMPUTED from the DS and B3 censuses; the
#: reader requires the board's regcheck to read the same, else those readings are NOT IDENTIFIED
OCCUPANCY = {"64x128x64w4s3": 3, "16x64x128w4s4": 3, "64x128x64w8s3": 3, "128x128x64w8s3": 2}
#: the tile census dry run's design occupancy from SHARED MEMORY ONLY (printed, DESCRIPTIVE; registers may bind
#: lower: 16x64x128w4s2's w2 at 72 registers holds 7, not 10 (review MINOR-9))
DESIGN_OCCUPANCY = {"64x128x64w4s3": 3, "64x128x64w4s2": 4, "64x128x64w4s4": 2, "64x128x128w4s2": 2,
                    "64x128x128w4s3": 1, "64x128x64w8s3": 3, "16x64x128w4s4": 3, "16x64x128w4s2": 10,
                    "16x64x128w4s3": 5, "128x128x64w8s3": 2}


def tile_key(t) -> str:
    bm, bn, bk, w, s = (int(x) for x in t)
    return f"{bm}x{bn}x{bk}w{w}s{s}"


def config_of(t) -> dict:
    bm, bn, bk, w, s = (int(x) for x in t)
    return {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "num_warps": w, "num_stages": s}


def tile_of(config: dict) -> tuple:
    return tuple(int(config[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"))


def mma_path(bm: int, warps: int) -> str:
    """The census predicate (Triton's): wgmma iff BLOCK_M % 64 == 0 and num_warps % 4 == 0, else mma.sync."""
    return "wgmma" if int(bm) % 64 == 0 and int(warps) % 4 == 0 else "mma.sync"


def wave(tile) -> int | None:
    o = OCCUPANCY.get(tile_key(tile))
    return None if o is None else SMS * o


# ------------------------------------------------------------------ cells
def counts_for(E: int, A: int, c_e: int) -> list[int]:
    """The first A experts take c_e rows each (every active expert one M-tile)."""
    if not 1 <= A <= E:
        raise Refused(f"A {A} active experts of E {E}")
    return [int(c_e)] * int(A) + [0] * (int(E) - int(A))


def gemm_dims(cfg, gemm: str) -> tuple[int, int]:
    """(N, K) of a GEMM: w1 N 2F, K H; w2 N H, K F."""
    if gemm == "w1":
        return 2 * cfg.intermediate_size, cfg.hidden_size
    return cfg.hidden_size, cfg.intermediate_size


def live_ctas(cfg, counts, tile, gemm: str) -> int:
    """Live CTAs of one GEMM: sum over experts of ceil(c_e / BM) x ceil(N / BN)."""
    bm, bn = int(tile[0]), int(tile[1])
    N, _K = gemm_dims(cfg, gemm)
    return sum(-(-int(c) // bm) for c in counts if int(c)) * (-(-N // bn))


def slab_bytes(cfg, tile, gemm: str) -> int:
    """b_c: the B slab one CTA streams, K x BLOCK_N x 2 (every PC decode cell gives each active expert one M-tile,
    so each slab is read by exactly one CTA)."""
    _N, K = gemm_dims(cfg, gemm)
    return int(K) * int(tile[1]) * 2


def naive(T: int, E: int, k: int) -> bool:
    """vLLM's naive assignment path: T k 4 <= E (r3_decode.decode_path's rule)."""
    return int(T) * int(k) * 4 <= int(E)


#: the stamped calls per cell: 3 (C3's), or 6 where the read GEMM has under 4 live CTAs, so every rung holds at least
#: 8 CTAs for PC-R1's 8-CTA rule (a build correction of the design's 3 at G 2: 2 x 3 = 6 < 8)
STAMP_CALLS = 3
STAMP_CALLS_SMALL = 6
SMALL_G = 4


def stamp_calls(read_live: int) -> int:
    return STAMP_CALLS_SMALL if int(read_live) < SMALL_G else STAMP_CALLS


def _cell(label, shape, A, c_e, tile=D64, flush=True, read="w2", role="", T=None, counts=None):
    from moe.spec import MODEL_CONFIGS
    shp = {s["name"]: s for s in SHAPES}
    if shape in shp:
        E, k = shp[shape]["E"], shp[shape]["k"]
    else:
        E, k = MODEL_CONFIGS[shape].num_experts, MODEL_CONFIGS[shape].top_k
    cnt = counts if counts is not None else counts_for(E, A, c_e)
    T = T if T is not None else sum(cnt) // k
    return {"label": label, "shape": shape, "T": int(T), "counts": [int(x) for x in cnt], "config": config_of(tile),
            "group_m": 1, "flush": bool(flush), "read": read, "role": role}


E32 = "pcs-e32-h4096"
A_LADDER_SMALL = (("A-G2", "pcs-e8-h128"), ("A-G4", "pcs-e8-h256"), ("A-G8", "pcs-e8-h512"),
                  ("A-G16", "pcs-e8-h1024"), ("A-G32", "pcs-e8-h2048"))
A_LADDER_E32 = (2, 4, 5, 6, 8, 9, 12, 13, 16, 24, 32)
FACT = {"s2k64": (64, 128, 64, 4, 2), "s4k64": (64, 128, 64, 4, 4), "s2k128": (64, 128, 128, 4, 2),
        "s3k128": (64, 128, 128, 4, 3), "w8": P64}


def block_a(prefix: str = "A") -> list[dict]:
    """Block A: D64 = 64x128x64 w4 s3 (vLLM's T 97 to 128 tile), 1710 MHz, c_e 64 (full M-tiles)."""
    out = [_cell(f"{prefix}-{g.split('-')[1]}", s, 2, 64, role="ladder") for g, s in A_LADDER_SMALL]
    out += [_cell(f"{prefix}-G{32 * a}", E32, a, 64, role="ladder") for a in A_LADDER_E32]
    out += [_cell(f"{prefix}-MX-T128-A4", "mixtral-8x7b", 4, 64, role="anchor CAL2/DS S-T128-A4"),
            _cell(f"{prefix}-JM-T128-A4", "jetmoe-8b", 4, 64, role="anchor"),
            _cell(f"{prefix}-MX-T128-A8", "mixtral-8x7b", 8, 32, role="anchor half tiles")]
    out += [_cell(f"{prefix}-F{f}-G8", f"pcs-e8-h512-f{f}", 2, 64, role="F ladder G 8") for f in (1792, 3584, 7168)]
    out += [_cell(f"{prefix}-TP{tp}-G128", f"mixtral-8x7b-tp{tp}", 4, 64, role="F ladder G 128") for tp in (8, 4, 2)]
    for g, (shape, a) in (("G8", ("pcs-e8-h512", 2)), ("G128", (E32, 4))):
        out += [_cell(f"{prefix}-{g}-{k}", shape, a, 64, tile=t, role="factorial") for k, t in FACT.items()]
    out += [_cell(f"{prefix}-G384-{k}", E32, 12, 64, tile=FACT[k], role="factorial near saturation")
            for k in ("s2k64", "s4k64")]
    out.append(_cell(f"{prefix}-G8-noflush", "pcs-e8-h512", 2, 64, flush=False, role="L2 control"))
    return out


def block_c() -> list[dict]:
    """Block C: the D64 clock twin at 1410 MHz."""
    out = [_cell("C-G8", "pcs-e8-h512", 2, 64, role="ladder"), _cell("C-G32", "pcs-e8-h2048", 2, 64, role="ladder")]
    out += [_cell(f"C-G{32 * a}", E32, a, 64, role="ladder") for a in (4, 8, 16, 32)]
    out += [_cell(f"C-G8-{k}", "pcs-e8-h512", 2, 64, tile=FACT[k], role="factorial") for k in ("s2k64", "s4k64")]
    out.append(_cell("C-MX-T128-A4", "mixtral-8x7b", 4, 64, role="anchor"))
    return out


def block_d() -> list[dict]:
    """Block D: D16 = 16x64x128 w4 s4 (vLLM's T <= 32 tile, mma.sync), 1710 MHz, c_e 16, BN 64."""
    out = [_cell(f"D-G{g}", s, 2, 16, tile=D16, role="ladder")
           for g, s in ((4, "pcs-e8-h128"), (16, "pcs-e8-h512"), (64, "pcs-e8-h2048"))]
    out += [_cell(f"D-G{64 * a}", E32, a, 16, tile=D16, role="ladder") for a in (2, 4, 6, 8, 16)]
    out += [_cell(f"D-MX-T{t}-A2", "mixtral-8x7b", 2, t, tile=D16, role="row fill") for t in (2, 4, 16)]
    for g, (shape, a) in (("G16", ("pcs-e8-h512", 2)), ("G128", (E32, 2))):
        out += [_cell(f"D-{g}-s{s}", shape, a, 16, tile=(16, 64, 128, 4, s), role="factorial") for s in (2, 3)]
    return out


def block_r() -> list[dict]:
    """Block R: A-G8, A-G128, A-G512 again at the end (page noise)."""
    keep = {"A-G8": "R-G8", "A-G128": "R-G128", "A-G512": "R-G512"}
    return [dict(c, label=keep[c["label"]], role="replicate") for c in block_a() if c["label"] in keep]


BLOCKS = {"pc-a": ("A", block_a, DECODE_LOCK), "pc-c": ("C", block_c, SESSION_LOCK), "pc-d": ("D", block_d, DECODE_LOCK),
          "pc-r": ("R", block_r, DECODE_LOCK)}


def page_doc(page: str) -> dict:
    blk, fn, lock = BLOCKS[page]
    cells = fn()
    return {"schema": SCHEMA, "page": page, "block": blk, "lock_mhz": lock, "shuffle_seed": SHUFFLE_SEED,
            "shapes_file": SHAPES_FILE, "registration": NAME,
            "label": (f"SYNTHETIC routings (scripts/scoring/pc/gen_pc_pages.py, {NAME}): PC block {blk}, every active "
                      "expert one M-tile, so each weight slab is read by one CTA; DESCRIPTIVE"),
            "cells": cells}


#: unit 16's decode page (decode-page/1, read by the pinned counter child): the Mixtral cells PC stamps, at vLLM's
#: default tiles (T 128: 64x128x64 w4 s3, T <= 32: 16x64x128 w4 s4)
CNT_PAGE = "mixtral-8x7b-cnt"
CNT_CELLS = (("L-T128-A4", 128, 4, 64), ("L-T128-A8", 128, 8, 32), ("L-T16-A2", 16, 2, 16), ("L-T4-A2", 4, 2, 4))
#: which stamped cell each counter cell matches (PC-R12)
CNT_MATCH = {"L-T128-A4": ("pc-a", "A-MX-T128-A4"), "L-T128-A8": ("pc-a", "A-MX-T128-A8"),
             "L-T16-A2": ("pc-d", "D-MX-T16-A2"), "L-T4-A2": ("pc-d", "D-MX-T4-A2")}


def cnt_doc() -> dict:
    return {"schema": DECODE_SCHEMA, "model": "mixtral-8x7b", "page": CNT_PAGE, "shuffle_seed": SHUFFLE_SEED,
            "decl_copies": None,
            "label": (f"SYNTHETIC (scripts/scoring/pc/gen_pc_pages.py, {NAME}): PC unit 16, the same-board counter "
                      "cross-check of PC's stamped Mixtral cells at vLLM's default tiles (DS's counter step, kernels=wide)"),
            "cells": [{"label": lab, "T": T, "counts": counts_for(8, A, c), "arms": ["native"], "role": "PC-R12"}
                      for lab, T, A, c in CNT_CELLS]}


def page_files() -> dict:
    """{published file name under PAGES_DIR: text} (sorted keys, indent 1)."""
    out = {f"{p}.json": json.dumps(page_doc(p), indent=1, sort_keys=True) + "\n" for p in BLOCKS}
    out[f"{CNT_PAGE}.json"] = json.dumps(cnt_doc(), indent=1, sort_keys=True) + "\n"
    return out


def shapes_text() -> str:
    return json.dumps(shapes_doc(), indent=1, sort_keys=True) + "\n"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_cell(c: dict, shapes: dict, tiles: dict = TILES) -> dict:
    """One PC page cell checked as the card would meet it; returns it with its cfg and tile. Refuses: an unknown or
    reserved shape, counts that are not E whole numbers summing to T k, a naive-path cell, a config not in the tile
    list, an expert past BLOCK_M (every PC cell gives each active expert one M-tile), a config the kernel cannot run,
    flush not a bool, read not w1 or w2."""
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
    if naive(T, E, k):
        raise Refused(f"cell {label}: T k 4 <= E is vLLM's naive path, which PC does not run")
    conf = c.get("config") or {}
    try:
        tile = tile_of(conf)
    except (KeyError, TypeError, ValueError):
        raise Refused(f"cell {label}: config {conf!r} is not BLOCK_SIZE_M, N, K, num_warps, num_stages") from None
    if tile_key(tile) not in tiles:
        raise Refused(f"cell {label}: tile {tile_key(tile)} is not one of the registered {sorted(tiles)}")
    if int(c.get("group_m", 1)) != 1:
        raise Refused(f"cell {label}: GROUP_SIZE_M {c.get('group_m')}: PC's cells are G 1")
    if max(counts) > tile[0]:
        raise Refused(f"cell {label}: an expert holds {max(counts)} rows, past BLOCK_M {tile[0]} (one M-tile declared)")
    if not isinstance(c.get("flush", True), bool):
        raise Refused(f"cell {label}: flush {c.get('flush')!r}: true or false")
    if c.get("read", "w2") not in ("w1", "w2"):
        raise Refused(f"cell {label}: read {c.get('read')!r}: w1 or w2")
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
    return out


# ------------------------------------------------------------------ the compiled kernel's loads in flight
_LOOP_HDR = re.compile(r"^(\$L__BB\d+_\d+):.*Inner Loop Header")
_WAIT = re.compile(r"cp\.async\.wait_group\s+(\d+)\s*;")


def inflight_stages(ptx: str | None) -> dict:
    """The B tiles a CTA keeps in flight, read from the compiled PTX: in the innermost loop (the "Inner Loop Header"
    label to its back-branch), g = cp.async.commit_group count per k-step (A and B commit apart: g 2 on C3's 128x128
    s3 compile) and w = the loop's cp.async.wait_group immediate; n_if = w / g + 1. A BUILD CORRECTION of the design's
    "immediate + 1": C3's published s3 PTX waits on 2 groups with g 2, so immediate + 1 reads 3 where s - 1 is 2. None
    (with the reason) where the loop, its wait or a whole ratio is missing."""
    out = {"n_if": None, "wait": None, "groups": None, "why": ""}
    lines = (ptx or "").splitlines()
    hdr = [(i, m.group(1)) for i, ln in enumerate(lines) for m in [_LOOP_HDR.match(ln.strip())] if m]
    loops = []
    for i, lab in hdr:
        for j in range(i + 1, len(lines)):
            if re.search(r"\bbra(\.uni)?\s+" + re.escape(lab) + r"\s*;", lines[j]):
                body = lines[i:j + 1]
                g = sum("cp.async.commit_group" in b for b in body)
                ws = [int(m.group(1)) for b in body for m in [_WAIT.search(b)] if m]
                if g and ws:
                    loops.append((g, max(ws)))
                break
    if not loops:
        out["why"] = "no loop with cp.async.commit_group and cp.async.wait_group"
        return out
    if len(set(loops)) != 1:
        out["why"] = f"loops disagree: {sorted(set(loops))}"
        return out
    g, w = loops[0]
    out.update(wait=w, groups=g)
    if w % g:
        out["why"] = f"wait {w} is not whole groups of {g}"
        return out
    out["n_if"] = w // g + 1
    return out


def q_bytes(tile, n_if: int | None) -> int | None:
    """Q_B = n_if x BLOCK_K x BLOCK_N x 2: the B bytes a CTA has in flight."""
    return None if n_if is None else int(n_if) * int(tile[2]) * int(tile[1]) * 2


# ------------------------------------------------------------------ the readings' registered constants
#: plateau CTAs: n_c at least this fraction of the launch's peak live count (G_live within one wave)
PLATEAU_FRAC = 0.9
MIN_PLATEAU_CTAS = 8
#: the per-call spread a rung's rate must keep (G <= 8: the looser bound)
SPREAD_MAX, SPREAD_MAX_SMALL, SPREAD_SMALL_G = 0.03, 0.05, 8
#: PC-R2: an F ladder needs this many rungs and the standard error of 1 / r_inf at most this fraction of it
F_RUNGS, SE_INV_RINF_MAX = 4, 0.05
#: PC-R6: the sharing groups and their least size
SHARED_M, ALONE_M, MIN_SHARE_CTAS = 1.8, 1.2, 20
#: trap 13: a cell's per-expert median rates flagged when their spread exceeds this (printed, never a refusal)
EXPERT_SPREAD_FLAG = 0.05
SIGMA_GS = (160, 192, 288)
#: PC-R7
PLATEAU_GS = (2, 4, 8, 16)
PLATEAU_WITHIN, SAT_WITHIN, KNEE_FRAC, SAT_RUNGS = 0.10, 0.03, 0.9, 2
#: PC-R8: within this the synthetic curve carries the CAL2 cell
CARRY_WITHIN = 0.10
#: PC-R9: a CTA streaming faster than Q_B / this many ns is L2-served: its launch is void (the flush-off control aside)
L2_LATENCY_NS = 350.0
#: the guard's loads in flight: the tile's largest registered stage count (s4: 3 tiles in flight), not the cell's, so a
#: low-stage cell streaming at the latency-bound rate is never voided (review MAJOR-5)
L2_GUARD_STAGES = 3
#: PC-R7's X_sat pool takes a w1 launch only at K >= this (the pcs-e32-h4096 rungs and the Mixtral anchors): short-K
#: w1 launches are set by a fixed per-CTA cost, not by DRAM (review MAJOR-7); a flush-off cell never enters it
X_SAT_W1_MIN_K = 4096
#: PC-R13: the start window and the flag
START_WINDOW_NS, START_FLAG, START_MIN_CTAS = 50_000.0, 0.97, 8
#: PC-R11 and PC-R10 (C3-R3's rules)
REPRO_MIN, OVERHEAD_FRAC = 0.02, 0.25
REREAD_L2_MIB = 60.0
#: C3's OD-1 NOT-READ bound on PERTURBED pages, read per reading's GEMM (owner decision 5: C3's rule)
OVERHEAD_MEDIAN_MAX, OVERHEAD_WORST_MAX = 0.10, 0.20
#: the timer rule (C3's): the globaltimer resolution at most a tenth of the median life, else the clock64 path
TIMER_MAX_FRAC = 1 / 10
CLOCK_PATH_MIN_RES = 10

#: H1 (per-CTA memory-level parallelism) bands, registered before any page (owner decision 4; design section 5)
H1 = {
    "falsify": {"beta_Q_below": 0.5, "beta_f_above": 0.5, "omega_above": 1.25, "plateau_ratio_above": 1.5,
                "plateau_load_frac_max": 0.25, "sigma_below": 0.6},
    "support": {"beta_Q": (0.7, 1.3), "beta_f_abs_max": 0.3, "omega": (0.85, 1.15), "plateau_within": 0.15,
                "sigma_min": 0.85},
    "low_load_frac_max": 0.25,
}


def _in(x, lo, hi) -> bool:
    return x is not None and math.isfinite(x) and lo <= x <= hi


def h1_label(beta_Q, beta_f, omega, plateau_spread, plateau_ratio, x16_frac, sigma, x8_frac=None) -> dict:
    """H1's DESCRIPTIVE label from the five low-load inputs (D64 at 1710, G 8; sigma at G 192):
       beta_Q        PC-R3's elasticity over (s2, s3, s4) at BK 64, G 8
       beta_f        PC-R4's clock elasticity at G 8
       omega         PC-R5's r(w8) / r(w4) at G 8
       plateau_spread max over min of r over the G 2, 4, 8, 16 rungs (the support band)
       plateau_ratio  r(G 2) / r(G 16) (the falsifier, directional as the design); x16_frac = X(G 16) / X_sat
       sigma         PC-R6's shared over alone rate at G 192
       x8_frac       X(G 8) / X_sat: the low-load premise
    FALSIFIED when any read falsifier holds (beta_Q < 0.5; beta_f > 0.5; omega > 1.25; r(G 2) / r(G 16) > 1.5 with
    X(G 16) <= 0.25 X_sat; sigma < 0.6); else NOT IDENTIFIED when any input is unread, the low-load premise included
    (X(G 8) / X_sat, unread without X_sat), or G 8 is not low load (X(G 8) > 0.25 X_sat); else SUPPORTED when every support band holds (beta_Q in [0.7, 1.3], |beta_f| <= 0.3, omega
    in [0.85, 1.15], the plateau within 15%, sigma >= 0.85); else MIXED. Every comparison fails on a NaN."""
    f, s = H1["falsify"], H1["support"]
    fin = lambda x: x is not None and isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
    falsified = []
    if fin(beta_Q) and beta_Q < f["beta_Q_below"]:
        falsified.append(f"beta_Q {beta_Q:.3f} < {f['beta_Q_below']}")
    if fin(beta_f) and beta_f > f["beta_f_above"]:
        falsified.append(f"beta_f {beta_f:.3f} > {f['beta_f_above']}")
    if fin(omega) and omega > f["omega_above"]:
        falsified.append(f"omega {omega:.3f} > {f['omega_above']}")
    if fin(plateau_ratio) and fin(x16_frac) and plateau_ratio > f["plateau_ratio_above"] \
            and x16_frac <= f["plateau_load_frac_max"]:
        falsified.append(f"no plateau: r ratio {plateau_ratio:.3f} > {f['plateau_ratio_above']} at X(G 16) "
                         f"{x16_frac:.3f} X_sat")
    if fin(sigma) and sigma < f["sigma_below"]:
        falsified.append(f"sigma {sigma:.3f} < {f['sigma_below']}")
    inputs = {"beta_Q": beta_Q, "beta_f": beta_f, "omega": omega, "plateau_spread": plateau_spread,
              "plateau_ratio": plateau_ratio, "sigma": sigma}
    if falsified:
        return {"label": "FALSIFIED", "why": falsified, "inputs": inputs}
    unread = [k for k, v in inputs.items() if not fin(v)]
    if not fin(x8_frac):
        unread.append("X(G 8) / X_sat (the low-load premise)")
    if unread:
        return {"label": "NOT IDENTIFIED", "why": [f"unread: {', '.join(unread)}"], "inputs": inputs}
    if fin(x8_frac) and x8_frac > H1["low_load_frac_max"]:
        return {"label": "NOT IDENTIFIED", "why": [f"G 8 is not low load: X {x8_frac:.3f} X_sat"], "inputs": inputs}
    held = {"beta_Q": _in(beta_Q, *s["beta_Q"]), "beta_f": abs(beta_f) <= s["beta_f_abs_max"],
            "omega": _in(omega, *s["omega"]), "plateau": plateau_spread <= 1 + s["plateau_within"],
            "sigma": sigma >= s["sigma_min"]}
    if all(held.values()):
        return {"label": "SUPPORTED", "why": ["every support band holds"], "inputs": inputs}
    return {"label": "MIXED", "why": [f"outside its support band: {', '.join(k for k, v in held.items() if not v)}"],
            "inputs": inputs}


# ------------------------------------------------------------------ the plan
TILE_KEYS = "block-m={} block-n={} block-k={} num-warps={} num-stages={}"


def tile_keys(t) -> str:
    return TILE_KEYS.format(*t)


HIST_B = "docs/registered/2026-10-14-tileB-hist"
PHI_128 = f"{HIST_B}/phi-3.5-moe-128g1.json"
PHI_64 = f"{HIST_B}/phi-3.5-moe-64g1.json"
PHI_128G16 = f"{HIST_B}/phi-3.5-moe-128g16.json"

#: the units in priority order (the driver drops the last first). (model, step, label, keys, est, cap, drop). The
#: design's 17, with the coordinator's Phi changes of 2026-11-06 (phi-split REPORT.md): E1 and E2 in one group, the
#: Phi counted-bytes units b128g1 and b64g1 in it (B3's plan lines, at 1410), b128g16 its own lower-priority group
UNITS = (
    ("-", "prelude", "", f"lock-mhz={SESSION_LOCK} power-limit-w={POWER_W}", 2, 15, "nodrop"),
    ("-", "tilecensus", "pc", f"tiles={','.join(CENSUS_TILES)} models={','.join(CENSUS_MODELS)}", 4, 15, "nodrop"),
    ("-", "regcheck", "rcp", "probe=pc", 3, 20, "nodrop"),
    ("-", "perturb", "ptp", f"probe=pc gates=sa,sd,sr lock-mhz={DECODE_LOCK}", 8, 25, "nodrop"),
    ("-", "stamps", "sa", f"probe=pc page={PAGES_DIR}/pc-a.json lock-mhz={DECODE_LOCK}", 6, 20, "nodrop"),
    ("-", "perturb", "ptc", "probe=pc gates=sc,se2", 4, 15, "nodrop"),
    ("-", "stamps", "sc", f"probe=pc page={PAGES_DIR}/pc-c.json", 3, 10, "nodrop"),
    ("olmoe-1b-7b", "bytes", "f128", f"{tile_keys(P128)} byte-groups=64", 3, 20, "phi"),
    ("mixtral-8x7b", "bytes", "f128", f"{tile_keys(P128)} byte-groups=64", 7, 25, "phi"),
    ("-", "gate", "gband", "gate=band", 1, 5, "phi"),
    ("-", "regcheck", "rc3", "probe=c3", 2, 10, "phi"),
    ("-", "perturb", "pt3", "probe=c3", 4, 15, "phi"),
    ("phi-3.5-moe", "stamps", "se1", f"{tile_keys(P128)} histogram={PHI_128} instr-stamps=ends stamp-groups=1 "
                                      "stamp-arms=native", 4, 15, "phi"),
    ("phi-3.5-moe", "stamps", "se2", f"probe=pc {tile_keys(P64)} histogram={PHI_64} instr-stamps=ends stamp-groups=1 "
                                      "stamp-arms=native", 4, 15, "phi"),
    ("phi-3.5-moe", "bytes", "b128g1", f"{tile_keys(P128)} histogram={PHI_128} arms=native,shared byte-groups=1",
     6, 18, "phi"),
    ("phi-3.5-moe", "bytes", "b64g1", f"{tile_keys(P64)} histogram={PHI_64} arms=native,shared byte-groups=1",
     5, 15, "phi"),
    ("-", "stamps", "sd", f"probe=pc page={PAGES_DIR}/pc-d.json lock-mhz={DECODE_LOCK}", 4, 15, "d16"),
    ("mixtral-8x7b", "decodebytes", "cnt", f"histogram={PAGES_DIR}/{CNT_PAGE}.json kernels=wide "
                                           f"lock-mhz={DECODE_LOCK}", 4, 15, "cnt"),
    ("phi-3.5-moe", "bytes", "b128g16", f"{tile_keys(P128)} histogram={PHI_128G16} arms=native,shared byte-groups=16",
     7, 21, "b16 depends=phi"),
    ("-", "stamps", "sr", f"probe=pc page={PAGES_DIR}/pc-r.json lock-mhz={DECODE_LOCK}", 3, 10, "rep"),
)
#: the cut order, last unit first (owner decision 1: replicate, the counter cross-check, the T <= 32 block, then the
#: Phi group; the coordinator, 2026-11-06: E1 and E2 together in the Phi group, never E2 before E1, and b128g16 a
#: lower-priority droppable unit, cut right after the replicate)
CUT_ORDER = ("rep", "b16", "cnt", "d16", "phi")
#: the units the regcheck drop rule drops (owner decision 2): every probe=pc unit
PC_UNIT_LABELS = ("ptp", "sa", "ptc", "sc", "se2", "sd", "sr")
#: the variant ids (u<unit>-<label>) of the stamps units
VARIANTS = {lab: f"u{i + 1}-{lab}" for i, u in enumerate(UNITS) for lab in [u[2]] if u[1] == "stamps"}
BOOT_MIN, RATE, RESERVE_MIN, DEADLINE_SLACK_MIN, CAP_PAD_MIN = 25, 2.29, 8, 5, 15


def plan_line(u) -> str:
    model, step, label, keys, est, cap, drop = u
    parts = [model, step] + ([f"label={label}"] if label else []) + keys.split()
    if step != "prelude":
        d = drop.split()
        parts += ["nodrop=1"] if d[0] == "nodrop" else [f"drop-group={d[0]}"] + d[1:]
    parts += [f"est={est}", f"cap={cap}"]
    return " ".join(parts)


def rental() -> dict:
    """The plan's figures: units estimated, wall = units + 25 min of boot and setup (C3's and the design's rule),
    the deadline the units, boot, the driver's 8 min reserve and 5 min of slack need, rounded up to 5 min; the
    guardian's hard cap 15 min later; costs at $2.29/h."""
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
            "rule": ("units x 1 plus 25 min of boot and setup (C3's rule); deadline launch + wall + the driver's 8 min "
                     "reserve + 5 min, rounded up to 5 min; the guardian's hard cap 15 min after the deadline")}


def plan_text() -> str:
    r = rental()
    hdr = [
        f"# RENTAL PC (registered {DATE}, before any page): per-CTA DRAM rate against loads in flight, a MEASUREMENT rental.",
        f"# GENERATED by scripts/scoring/pc/register_pc.py from registration {NAME}; do not edit by hand.",
        "# vLLM v0.27.1's fused_moe_kernel through the instrumented copy at the ends level (moe/instrumented, unchanged),",
        "# on synthetic routings and synthetic or SEEN expert shapes; Mixtral-8x22B, Qwen2-57B and Granite-3.0-1B are never",
        "# loaded (owner decision 3). Every reading is DESCRIPTIVE (no model form, no constant). One gpu_1x_gh200 at",
        f"# ${RATE}/h, run by PC's driver (C3's pinned driver plus lines marked PC):",
        "#   bash scripts/vm_run.sh start --ip <ip> --run-id <id> --deadline <epoch s> \\",
        f"#     --plan {PLAN} --driver gh200_pc_session.sh",
        f"# Session lock {SESSION_LOCK} MHz (C3's prelude, 700 W read back, G-BAND for the Phi pages; blocks C and E run at",
        f"# it); the 1710 units carry lock-mhz={DECODE_LOCK}, probed by the prelude and read back before and after each unit.",
        "# Line order is the priority; the driver drops the LAST unit first: rep, b16, cnt, d16, then the phi group",
        "# (owner decision 1; E1 and E2 together, coordinator 2026-11-06). A probe=pc regcheck that refuses or crashes",
        "# drops every probe=pc unit (owner decision 2), as does one that writes no line for a probe=pc stamps unit.",
        "# The unit locks are read back before and after each tool (pc-locks.env); the reader refuses a stamps unit whose",
        "# lock, or whose perturb unit's lock, did not read back. Early warning: the rcp regcheck's inflight_printed shows",
        "# each config's loads in flight from the PTX before any stamp; a config whose count is not num_stages - 1 leaves",
        "# its beta_Q points NOT IDENTIFIED.",
        f"# MINUTES: {r['units_min']} of units estimated ({r['nodrop_min']} nodrop); wall about {r['wall_min']} min with "
        f"{BOOT_MIN} of boot (${r['cost_usd']:.2f});",
        f"# deadline launch + {r['deadline_after_launch_min']} min; guardian hard cap launch + "
        f"{r['hard_cap_after_launch_min']} min (${r['cost_at_hard_cap_usd']:.2f} at most).",
    ]
    return "\n".join(hdr + [plan_line(u) for u in UNITS]) + "\n"


# ------------------------------------------------------------------ unseen at prefill tiles (a frozen literal)
#: COMPUTED at 6c800f84 from every tile and cal session's result directories (design section 8) and FROZEN: a later
#: publication changes nothing here (rental 2's freeze lesson). The test recomputes it over the sessions dated on or
#: before 2026-11-05 only.
UNSEEN_AT_PREFILL_6C800F84 = ("granite-3.0-3b-a800m", "granite-3.0-1b-a400m", "mixtral-8x7b-tp4", "mixtral-8x7b-tp8",
                              "qwen2-57b-a14b-tp8", "deepseek-v3", "deepseek-v3-tp4", "deepseek-v3-tp8")
