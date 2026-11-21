"""The tile stage's registrations: rental A's six (docs/registered/2026-10-14-tileA-{census,floor,bytes,
e1cal,dead,rep}-gh200) and rental B's two, frozen now and finalised mechanically from rental A's
selections (2026-10-14-tileB-{e1tile,stop}-gh200). Built for scripts/scoring/tileA/register.py,
which renders, pins and --checks them.

Every number is computed here from committed files (model v4: scripts/r3_timing_model_v4.py and
scripts/wave_split_bytes_v4.py over the pinned v3 modules and rental 6's CAL pricer; the histogram
pages of scripts/skew_synth_tile.py; the row-key rows of rowkeys_tile.py; the power of
power_tile.py) or is a typed-in input with its provenance and label (CAL, CAL-A, SEEN, SEEN-FITTED,
SEEN-SELECTED, SEEN-INFORMED, SEEN-DERIVED, SEEN-WORKLOAD, SEEN-SHAPE, BLIND, BLIND-TILE,
BLIND-CALMODEL, DESIGN-NOMINAL). New labels here:
  CAL-A           a rental-A page used to fit a constant;
  BLIND-CALMODEL  a rental-A test on a CAL model;
  BLIND-TILE      time at a tile on a model SEEN only at R3's tile (Phi, Qwen1.5, JetMoE);
  SEEN-SHAPE      deepseek-v2-lite (owner decision D7): on a published H200 page, and its GEMM shapes
                  (K, N) are Qwen1.5's; its claim is "new model, seen shapes";
  SEEN-WORKLOAD   a uniform or balanced cell whose routing counts a published R3-tile page timed
                  (only the tile is new);
  DESIGN-NOMINAL  a number computed at the design's constants (c_128 900 cycles, F 520 / 979), for
                  levers, sensitivities and power only; never a registered prediction.

Design: the scratchpad's design-tile DESIGN.md corrected by design-tile-review REVIEW.md (the review
wins), with the owner's decisions D1 to D10 of 2026-10-14 and the eleven review fixes of the build
brief.

LEAKAGE, said once for all eight: every published page is SEEN, rental 7's included; every constant
typed in was fitted or chosen after seeing published pages (labelled per entry). No tile-stage page
exists when this runs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tcommon as TC  # noqa: E402

REPO = TC.REPO
for p in (str(REPO / "scripts" / "scoring" / "rental6"), str(REPO / "scripts" / "scoring" / "rental7")):
    if p not in sys.path:
        sys.path.insert(0, p)

import power_tile as PT  # noqa: E402
import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import r6model as M6  # noqa: E402
import reg6 as R6  # noqa: E402
import skew_synth_tile as ST  # noqa: E402
import wave_split_bytes_v4 as WV4  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

r6 = TC.r6
PUB = REPO / "results" / "published"
DESIGN = ("design-tile DESIGN.md corrected by design-tile-review REVIEW.md (the review wins) and the owner's decisions of "
          "2026-10-14: D1 prefill now on the reachable wgmma tiles 64x128x64 w8 and 128x128x64 w8 (decode tiles later); D2 "
          "NATIVE certifies, SHARED secondary; D3 rental A's selection rules make rental B mechanical; D4 'each shape' is each "
          "model; D5 MVA2 is SEEN-fitted for mma.sync; D6 a failed tA-e1cal STOPS rental B; D7 deepseek-v2-lite is SEEN-SHAPE "
          "('new model, seen shapes'); D8 Granite is cut; D9 term (b) not revised first; D10 about $11.4")
REGISTERED = f"{TC.DATE}, before any tile-stage page"
NOMINAL = {"c128": 900.0, "F_wg": {"w1": 520.0, "w2": 979.0}, "variant": "W", "f_form": "F-step", "key": "CLAMP"}
NOMINAL_LABEL = ("DESIGN-NOMINAL: c_128 900 counter cycles (design-tile 2.2's mid value), F_wg the R3 tile's CAL 520 / 979, "
                 "accounting W, F-step, A key CLAMP; for levers, sensitivities and power only, never a registered prediction")
COPIES = 9
BLIND_TILE = ("phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b")
DSV = "deepseek-v2-lite"
CAL = ("olmoe-1b-7b", "mixtral-8x7b", "qwen2-57b-a14b")
PAGE_TILE = {64: V4.T64W8, 128: V4.CAL_TILE}
#: rental A's pages: (unit label, model, histogram page name, step)
A_BYTE_PAGES = (("b128g1", "olmoe-1b-7b", "olmoe-1b-7b-128g1"), ("b64g1", "mixtral-8x7b", "mixtral-8x7b-64g1"),
                ("b128g16", "mixtral-8x7b", "mixtral-8x7b-128g16"), ("b128g1", "qwen2-57b-a14b", "qwen2-57b-a14b-128g1"))
A_TIMED_PAGES = (("e128g1", "olmoe-1b-7b", "olmoe-1b-7b-128g1"), ("e128g16", "mixtral-8x7b", "mixtral-8x7b-128g16"),
                 ("e64g1", "mixtral-8x7b", "mixtral-8x7b-64g1"), ("e128g1", "qwen2-57b-a14b", "qwen2-57b-a14b-128g1"),
                 ("e128g1", "mixtral-8x7b", "mixtral-8x7b-128g1"))
A_REP_PAGE = ("e128g1r", "olmoe-1b-7b", "olmoe-1b-7b-128g1")
A_DEAD_PAGES = (("d9", "olmoe-1b-7b", "olmoe-1b-7b-dead", 9), ("d15", "olmoe-1b-7b", "olmoe-1b-7b-dead", 15))
A_FLOOR_PAGES = {"f128": (V4.CAL_TILE, ("olmoe-1b-7b", "mixtral-8x7b")), "f64w8": (V4.T64W8, ("olmoe-1b-7b", "mixtral-8x7b")),
                 "f64w4": (V4.T64W4, ("olmoe-1b-7b",)), "f32w4": (V4.T32W4, ("olmoe-1b-7b",))}
#: rental B's units (frozen; the plan written at finalisation runs these, in this order):
#: (label, model, page name, nodrop or drop group, est min)
B_TIMED_PAGES = (("t64g1", "phi-3.5-moe", "phi-3.5-moe-64g1", "nodrop", 10),
                 ("t128g1", "phi-3.5-moe", "phi-3.5-moe-128g1", "nodrop", 14),
                 ("t128g16", "phi-3.5-moe", "phi-3.5-moe-128g16", "nodrop", 16),
                 ("t128g1", "qwen1.5-moe-a2.7b", "qwen1.5-moe-a2.7b-128g1", "nodrop", 13),
                 ("t128g16", "jetmoe-8b", "jetmoe-8b-128g16", "nodrop", 9),
                 ("t128g16", "qwen1.5-moe-a2.7b", "qwen1.5-moe-a2.7b-128g16", "nodrop", 11),
                 ("t128g1", "jetmoe-8b", "jetmoe-8b-128g1", "nodrop", 7),
                 ("t128g1", DSV, "deepseek-v2-lite-128g1", "dsv", 12))
B_BYTE_PAGES = (("b128g1", "phi-3.5-moe", "phi-3.5-moe-128g1", "byt", 6),
                ("b128g1", "qwen1.5-moe-a2.7b", "qwen1.5-moe-a2.7b-128g1", "byt", 5),
                ("b64g1", "phi-3.5-moe", "phi-3.5-moe-64g1", "byt", 5))
SKEWED = ("PT", "PW", "PT2", "PW2")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hist(rental: str, name: str) -> dict:
    return json.loads(ST.path_of(rental, name).read_text())


# --------------------------------------------------------------------------
# the pricer and the published set, as of the registration
# --------------------------------------------------------------------------

def _asof_dir() -> Path:
    """A directory of links to the published sessions dated on or before 2026-10-14 whose name does
    not carry 'tile' (--check recomputes these files after the tile pages are published)."""
    import atexit
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="tile-asof-"))
    atexit.register(shutil.rmtree, tmp, True)
    for d in sorted(PUB.iterdir()):
        if d.is_dir() and d.name[:10] <= TC.DATE and "tile" not in d.name:
            (tmp / d.name).symlink_to(d.resolve(), target_is_directory=True)
    return tmp


_ASOF: list = []


def asof() -> Path:
    if not _ASOF:
        _ASOF.append(_asof_dir())
    return _ASOF[0]


def pricer() -> M6.Pricer:
    """rental 6's CAL pricer (Engine, v2 fit, H8 refit off) with the A proxy as of the registration."""
    if not R6._P:
        R6._P.append(M6.Pricer(proxy=M6.align_proxy(asof()), h8=False))
    return R6._P[0]


_PUBSET: list = []


def published() -> set:
    """Every (model, n, counts sha256) a published page or an earlier registered histogram file
    carries, as of the registration (reg7.published_counts_sha over the as-of tree)."""
    if _PUBSET:
        return _PUBSET[0]
    import reg7 as R7
    keep = (R7.PUB, list(R7._PUBSHA))
    R7.PUB, R7._PUBSHA[:] = asof(), []
    try:
        got = set(R7.published_counts_sha())
    finally:
        R7.PUB = keep[0]
        R7._PUBSHA[:] = keep[1]
    _PUBSET.append(got)
    return got


_LADDER: list = []


def ladder_treads() -> set:
    """Every (model, tread) a published R3 page timed with balanced routing, as of the registration:
    each report.json's balanced rows (a ladder page's rows, histogram None, and a histogram page's
    `balanced` rows) at its BLOCK_M (build-tile-review 4: the plain ladder pages run balanced_ids)."""
    if _LADDER:
        return _LADDER[0]
    out = set()
    import glob
    for rp in glob.glob(str(asof() / "**" / "private_weight_reference" / "*" / "report.json"), recursive=True):
        try:
            d = json.loads(Path(rp).read_text())
        except Exception:                                    # noqa: BLE001
            continue
        bm = int(d.get("block_m") or 32)
        for r in d.get("treads_table") or []:
            if isinstance(r, dict) and d.get("model") and r.get("histogram") in (None, "balanced"):
                out.add((d["model"], int(r["tiles"]) * bm))
    _LADDER.append(out)
    return out


def workload_label(model: str, n: int, bm: int, counts) -> str:
    """SEEN-WORKLOAD when a published page timed the same routing counts: a skewed cell by its counts
    sha256 on any published histogram page; a uniform or balanced cell (every expert n x BLOCK_M
    rows) also when any published R3 page of the model timed balanced routing at the same rows per
    expert (a ladder page's tread n x BLOCK_M / 32 at BLOCK_M 32), else BLIND counts."""
    sha = R6.sha_counts(counts)
    pub = published()
    E = MODEL_CONFIGS[model].num_experts
    if counts is None or (len(set(counts)) == 1 and int(counts[0]) == n * bm and len(counts) == E):
        seen = ((model, n * bm // 32, R6.sha_counts(None)) in pub or (model, n * bm) in ladder_treads()
                or any(m == model and s == R6.sha_counts([n * bm] * E) for m, _n, s in pub))
    else:
        seen = any(m == model and s == sha for m, _n, s in pub)
    return "SEEN-WORKLOAD (a published R3-tile page timed these counts; only the tile is new)" if seen else \
        "BLIND counts (no published page carries them)"


def files_block(rental: str, names) -> dict:
    out = {}
    for nm in names:
        p = ST.path_of(rental, nm)
        doc = json.loads(p.read_text())
        bm = int(doc["block_m"])
        t = PAGE_TILE[bm]
        out[f"{nm}.json"] = {"path": str(p.relative_to(REPO)), "sha256": sha256_file(p), "shuffle_seed": doc["shuffle_seed"],
                             "tile": [t.bm, t.bn, t.bk, t.warps, t.stages], "group_m": doc["group_m"],
                             "cells": [{"label": c["label"], "n": c["n"], "arms": c["arms"], "counts_sha256": R6.sha_counts(c["counts"]),
                                        **({"seed": c["provenance"]["seed"], "kind": c["provenance"]["kind"],
                                            "param": c["provenance"]["param"], "source_model": c["provenance"]["source_model"]}
                                           if c.get("provenance") else {})} for c in doc["cells"]]}
    return out


def census_occupancy() -> dict:
    """{tile key: {w1, w2}} every model shares, from rental A's census when there is one (else {}: v4 then
    reads the shared-memory rule, tcommon.occupancy_of)."""
    cs = TC.census_score()
    return dict((cs or {}).get("occupancy") or {})


def nominal_params(**kw) -> V4.Params:
    d = dict(c128=NOMINAL["c128"], F_wg=dict(NOMINAL["F_wg"]), variant=NOMINAL["variant"], f_form=NOMINAL["f_form"],
             occupancy=census_occupancy())
    d.update(kw)
    return V4.Params(**d)


def seed_of(doc: dict, c: dict):
    return None if c["counts"] is None else doc["shuffle_seed"]


def price_cell(model: str, arm: str, c: dict, doc: dict, prm: V4.Params, key: str = "CLAMP", **kw) -> dict:
    t = PAGE_TILE[int(doc["block_m"])]
    return WV4.price(pricer(), model, arm, int(c["n"]), c["counts"], COPIES, seed_of(doc, c), t, int(doc["group_m"]),
                     prm, key=key, **kw)


def sensitivity(model: str, c: dict, doc: dict, prm: V4.Params) -> float:
    """dln T / dln bytes of v4 at the cell (NATIVE): every GEMM's bytes scaled by 1.01."""
    base = price_cell(model, "native", c, doc, prm)
    up = price_cell(model, "native", c, doc, prm, reads={g: 1.01 * v for g, v in base["reads"].items()})
    return math.log(up["ms"] / base["ms"]) / math.log(1.01)


# --------------------------------------------------------------------------
# rivals (rental B) at a cell
# --------------------------------------------------------------------------

def roofline_ms(model: str, n: int, counts, doc: dict, reads: dict, eta: float) -> float:
    """T0 + sum over GEMMs of max(flops / (4096 x eta x 132 x 1.71 GHz), bytes / bw) (design-tile 4.2)."""
    pr = pricer()
    cfg = MODEL_CONFIGS[model]
    T0, _c, bw = float(pr.f2.x[0]), float(pr.f2.x[1]), float(pr.f2.x[2])
    rows = sum(int(v) for v in counts) if counts is not None else n * int(doc["block_m"]) * cfg.num_experts
    out = T0
    for g in ("w1", "w2"):
        K = cfg.hidden_size if g == "w1" else cfg.intermediate_size
        N = 2 * cfg.intermediate_size if g == "w1" else cfg.hidden_size
        flops = 2.0 * rows * K * N
        t_c = flops / (V4.FLOP_PER_CLK * eta * V4.SMS * V4.CLOCK_MHZ * 1e6) * 1e3
        t_b = reads[g] / (bw * 1e6)
        out += max(t_c, t_b)
    return out


# --------------------------------------------------------------------------
# tileA-census
# --------------------------------------------------------------------------

SEEN_PTX = (("results/published/2026-09-10-nvidia_h200-gaps-session/results/bn_decomposition/"
             "nvidia_h200-bm32_64_128-budget400.0-dtypebf16-flushtrue-g16-iters50-k64-modelmixtral_8x7b-n32_64-b59b409f/"
             "triton-cache/n128-bm128/2HTRHRIR5UUT3AFIASWJPE3DTDBPTFZBEL3V4D37EW2P4TBNV4SQ/fused_moe_kernel.ptx", "128x128x64w8s4"),
            ("results/published/2026-09-10-nvidia_h200-gaps-session/results/bn_decomposition/"
             "nvidia_h200-bm32_64_128-budget400.0-dtypebf16-flushtrue-g16-iters50-k64-modelmixtral_8x7b-n32_64-b59b409f/"
             "triton-cache/n128-bm64/JZFSVHA6N7ZPPAEE2BBXKEF2J6JPF5YH2BJWOKT32TWJI6SJLBHA/fused_moe_kernel.ptx", "64x128x64w8s4"),
            ("results/published/2026-09-10-nvidia_h200-gaps-session/results/bm128_depth/"
             "nvidia_h200-budget400.0-dtypebf16-g1-iters50-k64-l2flushtrue-modelmixtral_8x7b-n64-r2048-referen-3c34b7a0/"
             "triton-cache/bm32/7INWIJHYBZG425JTKZNWSANE4VVXELRKFBVEEC3D4MCGB5QQYANQ/fused_moe_kernel.ptx", "32x64x64w8s4"))


def census() -> dict:
    import tile_census as TCEN
    tiles = []
    for t in V4.VLLM_TILES:
        e = {"tile": t.key, "path": t.path, "shared_bytes": t.shared_bytes(), "design_occupancy": t.occ_shared(),
             "reachable_by_R3": t in V4.REACHABLE,
             "role": ("rental B's blind test tile and rental A's calibration and E1-CAL tile" if t == V4.CAL_TILE else
                      "rental B's blind test tile (Phi n 1) and rental A's floor test and E1-CAL tile" if t == V4.T64W8 else
                      "rental A's counter-only floor test (DESCRIPTIVE); vLLM's M 97..128 tile, decode (D1)" if t == V4.T64W4 else
                      "rental A's counter-only MVA2 test (D5); vLLM's M 65..96 tile, decode (D1)" if t == V4.T32W4 else
                      "census only; a decode tile (D1: a later stage)")}
        if t.path == "wgmma":
            c = V4.wgmma_counts_estimate(t)
            b = V4.wgmma_bytes(t, c)
            e["design_counts"] = c
            e["per_cta_kstep_bytes"] = {"wgmma_operand_reads": b["reads"], "cp_async_writes": b["writes"]}
            e["mma_cycles"] = {"Luo Table 11 per clock (registered)": r6(2 * t.bm * t.bn * t.bk / V4.FLOP_PER_CLK / V4.eta_mma_of(t)),
                               "Luo Table 9 chip-relative (rival)": r6(2 * t.bm * t.bn * t.bk / V4.FLOP_PER_CLK / V4.eta_mma_of(t, True))}
        else:
            e["design_wavefronts"] = r6(V4.wavefronts_estimate(t))
            e["design_layout"] = list(V4.mma_layout_estimate(t))
        tiles.append(e)
    seen = []
    for path, key in SEEN_PTX:
        p = REPO / path
        txt = p.read_text()
        t = V4.Tile.parse(key)
        c = TCEN.ptx_counts(txt, t)
        seen.append({"file": path, "sha256": sha256_file(p), "tile": key, "card": "H200 (SEEN)", "counts": c})
    return {
        "registered": REGISTERED, "name": TC.NAMES["census"], "status": "REGISTERED BEFORE ANY PAGE (no verdict: the inputs v4 reads)",
        "design": DESIGN, "tool": "scripts/tile_census.py (compile only), the driver's tilecensus unit (scripts/gh200_tile_session.sh)",
        "unit": "- tilecensus label=ta, rental A's third unit, nodrop: the six tiles below on every model of rentals A and B",
        "what": "per (tile, model, GEMM): registers (EIATTR_REGCOUNT), shared bytes, warps, stages, CTAs per SM and the binding "
                "limit (moe.instrumented.cubin.occupancy, the sm_90 calculator), the MMA path, and the main loop's per-thread PTX "
                "counts (mma.sync: ldmatrix and cp.async, then LSU wavefronts per CTA k-step; wgmma: the instruction and its count "
                "per warpgroup, cp.async, bar.sync, fence.proxy.async and the waits)",
        "tiles": tiles,
        "seen_ptx": {"label": "SEEN (H200, Triton 3.7.1, num_stages 4; the s3 PTX is unpublished: the census reads it)",
                     "files": seen},
        "rules": {
            "path": "the unit REFUSES itself (exit 3, VALIDITY census_path FAIL) when any config's compiled MMA path is not the "
                    "predicate's (wgmma iff BLOCK_M % 64 == 0 and num_warps % 4 == 0): every v4 law is per path",
            "occupancy_read_not_fitted": "v4 reads each (tile, GEMM)'s CTAs per SM from the census of the model it prices; the "
                                         "geometry's co-residency window W_c = 132 x that occupancy",
            "re_key (design-tile-review 9.11)": "a rental-A counter page whose recorded launch__occupancy_limit_* minimum differs from "
                                                "the census (integers, exact) is scored at its recorded occupancy (rental 4's "
                                                "re-key rule) and printed; the census value stays v4's for every other page",
            "rental_B_scope (design-tile-review 9.11)": "rental B re-reads the census on its own board (a tilecensus unit, compile "
                                                        "only; score_tileB.census_check). A (model, tile) whose rental-B occupancy "
                                                        "differs from rental A's census on either GEMM has its predictions NOT "
                                                        "SCORED on every page of that (model, tile); a missing or refused rental-B "
                                                        "census leaves every page NOT SCORED",
            "path_mismatch_is_a_STOP (build-tile-review 9)": "a census that refuses (MMA path mismatch) or is missing STOPS rental "
                                                             "B (tileB-stop) and finalise.py refuses: never a fallback to the "
                                                             "design occupancy",
            "mechanical_steps": ["every occupancy is read by tcommon.occupancy_of (the census, else the shared-memory rule): "
                                 "`rowkeys_tile.py --census-score <census.score.json>` re-prices the row-key rows and "
                                 "`register.py --census-score <census.score.json>` rebuilds every registration at the census "
                                 "occupancy before rental B is finalised; no number is edited by hand",
                                 "v4's wavefronts (mma.sync) and wgmma counts (wgmma) are the census's; the design estimates "
                                 "above are replaced"],
            "registers": "printed, not gating (the occupancy is what v4 reads)"},
        "labels": {"census": "compile only: no time, so the blind models' configs are compiled without timing them",
                   "design_numbers": "derived (layout rule, shared-memory rule): replaced by the census"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["the SEEN H200 PTX above", "the shared-memory rule read off 1,304 published cubins (design-tile 1)"],
    }


# --------------------------------------------------------------------------
# tileA-floor
# --------------------------------------------------------------------------

C128_GRID = (600.0, 750.0, 900.0, 1100.0, 1300.0)


def variant_tables() -> dict:
    out = {}
    for name, rival in (("eta_mma Luo Table 11 per clock, per instruction (registered)", False),
                        ("eta_mma Luo Table 9 chip-relative (rival)", True)):
        rows = []
        for c128 in C128_GRID:
            row = {"c128": c128}
            for var in V4.VARIANTS:
                prm = nominal_params(c128=c128, variant=var, eta_rival=rival)
                z = V4.z_wg(prm)
                if z is None:
                    row[var] = "unreachable (the pipe alone exceeds c_128): FALSIFIED by construction"
                    continue
                c8 = V4.c_counter(V4.T64W8, "w1", prm)
                c4 = V4.c_counter(V4.T64W4, "w1", prm)
                row[var] = {"Z_wg": r6(z), "c_64x128_w8": r6(c8), "c_64x128_w4": r6(c4), "w8_over_w4": r6(c8 / c4)}
            rows.append(row)
        D = {t.key: {v: r6(V4.pipe_service(t, v, V4.eta_mma_of(t, rival))["D"]) for v in V4.VARIANTS}
             for t in (V4.CAL_TILE, V4.T64W8, V4.T64W4)}
        out[name] = {"pipe_service_cycles": D, "by_c128": rows}
    return out


def eta_sensitivity() -> dict:
    """build-tile-review 1: what the eta_mma choice moves. Per c_128 on the grid, accounting and F form:
    c_64 (64x128 w8) under the registered per-clock rate against Table 9's, and the largest |T| shift on
    rental B's six Phi 64x128 NATIVE cells; and on the 78 128x128 cells (c there is c_128: eta never enters)."""
    phi = [(m, doc, c) for _l, m, n, doc, c in b_cells() if n == "phi-3.5-moe-64g1"]
    big = [(m, doc, c) for _l, m, n, doc, c in b_cells() if int(doc["block_m"]) == 128]
    rows, worst_t, worst_c, flips = [], 0.0, 0.0, []
    for c128 in C128_GRID:
        for var in V4.VARIANTS:
            for form in V4.F_FORMS:
                a = nominal_params(c128=c128, variant=var, f_form=form)
                b = nominal_params(c128=c128, variant=var, f_form=form, eta_rival=True)
                if V4.z_wg(a) is None or V4.z_wg(b) is None:
                    rows.append({"c128": c128, "variant": var, "f_form": form, "note": "unreachable under one rate"})
                    continue
                ca, cb = V4.c_counter(V4.T64W8, "w1", a), V4.c_counter(V4.T64W8, "w1", b)
                dt = max(abs(price_cell(m, "native", c, doc, a)["ms"] / price_cell(m, "native", c, doc, b)["ms"] - 1)
                         for m, doc, c in phi)
                dc = ca / cb - 1
                worst_t, worst_c = max(worst_t, dt), max(worst_c, abs(dc))
                if abs(dc) > TC.FLOOR_TOL:
                    flips.append(f"c128 {c128:g} {var} {form}")
                rows.append({"c128": c128, "variant": var, "f_form": form, "c64_registered": r6(ca), "c64_rival": r6(cb),
                             "c64_shift_pct": r6(100 * dc), "phi_64x128_max_T_shift_pct": r6(100 * dt)})
    m, doc, c = big[0]
    d128 = abs(price_cell(m, "native", c, doc, nominal_params())["ms"] /
               price_cell(m, "native", c, doc, nominal_params(eta_rival=True))["ms"] - 1)
    return {"rows": rows, "max_c64_shift_pct": r6(100 * worst_c), "phi_64x128_max_T_shift_pct": r6(100 * worst_t),
            "128x128_T_shift_pct": r6(100 * d128), "lever_cells": 0 if worst_t < TC.LEVER_FRAC else "see rows",
            "c64_shift_beyond_the_3pct_band": flips,
            "selection_effect": "the 3% accounting band is DESCRIPTIVE, but its selection decides whether rental B derives "
                                "c_64 or takes v4cal (D3). Where the eta choice moves c_64 by more than 3% (the cases listed) "
                                "it can change WHICH accounting holds on rental A's 64x128 pages, so the selection is read "
                                "under the registered (per-clock) rate only; under v4cal the eta effect on rental B is 0"}


def b_cells():
    for label, model, name, _group, _est in B_TIMED_PAGES:
        doc = hist("B", name)
        for c in doc["cells"]:
            yield label, model, name, doc, c


def lever_counts_B() -> dict:
    """At the design-nominal constants: on rental B's NATIVE cells, how many cells each rival form
    moves by 2% or more against v4 (rental 7's lever: under 6 is NOT SEPARATED by construction)."""
    base = nominal_params()
    forms = {f"accounting {v}": nominal_params(variant=v) for v in V4.VARIANTS if v != NOMINAL["variant"]}
    forms["F-cyc"] = nominal_params(f_form="F-cyc")
    forms["eta_mma rival (Table 9 chip-relative)"] = nominal_params(eta_rival=True)
    out = {k: {"lever_cells": 0, "max_abs_pct": 0.0} for k in forms}
    for _l, model, _n, doc, c in b_cells():
        t0 = price_cell(model, "native", c, doc, base)["ms"]
        for k, prm in forms.items():
            try:
                t = price_cell(model, "native", c, doc, prm)["ms"]
            except TM.Refused:
                out[k]["note"] = "unreachable at the nominal c_128"
                continue
            d = abs(t / t0 - 1)
            out[k]["lever_cells"] += d >= TC.LEVER_FRAC
            out[k]["max_abs_pct"] = max(out[k]["max_abs_pct"], r6(100 * d))
    for v in out.values():
        v["verdict_by_construction"] = ("NOT SEPARATED (fewer than 6 lever cells on rental B)" if v["lever_cells"] < TC.LEVER_MIN
                                        else "decidable")
    return out


def mma_sync_predictions() -> dict:
    out = {}
    prm = V4.Params()
    for t in (V4.T32W4, V4.Tile(16, 64, 128, 4, 4), V4.Tile(32, 64, 128, 4, 3)):
        occ = TC.occupancy_of(t)[0]["w1"]
        out[t.key] = {"occupancy": occ, "design_wavefronts": r6(V4.wavefronts_estimate(t)),
                      "c_counter_w1": r6(V4.c_counter(t, "w1", dataclasses.replace(prm, occupancy={t.key: {"w1": occ, "w2": occ}}))),
                      "c_counter_w2": r6(V4.c_counter(t, "w2", dataclasses.replace(prm, occupancy={t.key: {"w1": occ, "w2": occ}})))}
    return out


def floor() -> dict:
    lev = lever_counts_B()
    return {
        "registered": REGISTERED, "name": TC.NAMES["floor"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/tileA/score_floor.py; tcommon.joint_cF, floor_test, variant_rule, f_form_rule; model "
                "v4 scripts/r3_timing_model_v4.py",
        "model_v4": {
            "definition": "v3 with every tile-dependent quantity read from the tile (scripts/r3_timing_model_v4.py docstring): "
                          "c_time(tile) = c_card x c_counter(tile) / 344.1 and PHI(tile, g) = F(tile, g) / c_counter(tile); "
                          "geometry, k-steps, grid, dead CTAs and bytes on the tile's geometry; occupancy from the census",
            "reduction_to_v3 (design-tile-review 1a)": "at R3's tile c_counter is 344.1 and c_scale exactly 1.0, PHI is v3's own "
                                                       "CTA_FIXED_KSTEPS (1.512 / 2.846): `python scripts/r3_timing_model_v4.py "
                                                       "--check` re-prices all 300 registered v3 predictions of rental6-skew and "
                                                       "rental7-skew and requires equality with the registered numbers (6 decimals) "
                                                       "and with rental 6's pricer bit for bit (tests/test_tile_v4.py runs it)",
            "c_counter": {"R3 tile": "344.1 (the CAL counters' c)",
                          "mma.sync": "344.1 x MVA2(W_tile, BK, occ) / MVA2(W_R3, 64, occ_R3,g); W from the census PTX (R3's "
                                      "SEEN PTX gives 280)",
                          "wgmma 128x128x64 w8 s3": "c_128, CAL-A (below)",
                          "wgmma other": "mva(D_accounting(tile), Z_wg, occ): Z_wg puts the 128x128 tile at c_128",
                          "v4cal": "the per-tile c and F measured on that tile's CAL pages (CAL-A), the fallback at 64x128"},
            "F_forms": {"F-step (registered, v3's own form)": "PHI in k-steps carries: PHI_wg,g = F_wg,g / c_128 at every wgmma "
                                                              "tile; R3's PHI on the mma.sync path",
                        "F-cyc (rival)": "F in cycles carries: PHI = F / c_counter(tile)",
                        "selection_rule": "F-step unless, on rental A's two 64x128 w8 pages (the joint fit's F_64,w1 and F_64,w2), "
                                          "F-cyc is nearer the measured F on BOTH GEMMs AND F-step misses it by more than 3 se on "
                                          "at least one (tcommon.f_form_rule); the two forms are equal at 128x128"},
            "eta_mma": {"registered": {"m64n128k16 (128x128 w8)": r6(V4.ETA_MMA["m64n128k16"]),
                                       "m64n64k16 (64x128 w8 and w4)": r6(V4.ETA_MMA["m64n64k16"])},
                        "source": "Luo et al., 'Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple "
                                  "Level Analysis', arXiv 2501.12084, Table 11: m64n256k16 dense SS random data, 583 TFLOPS at "
                                  "1260 MHz on 114 SMs = 0.991 per clock; carried to m64n128k16 and m64n64k16 by Table 9's "
                                  "within-table random-data ratios 659.8 / 665.4 and 648.3 / 665.4",
                        "why_per_clock (build-tile-review 1)": "v4 counts MMA cycles at a locked 1710 MHz, so eta is a per-clock "
                                                               "efficiency; Table 11 is the only wgmma rate with its clock "
                                                               "recorded",
                        "what_it_is": "MEASURED ON AN H800 PCIe (114 SMs, 350 W cap) WITH FP16 INPUTS; Luo attribute the "
                                      "random-data drop of Table 9 to the power cap lowering the clock",
                        "assumption": "REGISTERED ASSUMPTION: BF16 on the GH200 runs at the same per-clock rate as FP16 on the "
                                      "H800 PCIe; untested here",
                        "rival": {"value": {k: r6(v) for k, v in V4.ETA_MMA_RIVAL.items()},
                                  "source": "Luo et al. Table 9: 659.8 (N 128) and 648.3 (N 64) of the 756.5 TFLOPS chip peak at "
                                            "the 1620 MHz whitepaper clock (chip-relative, the power-capped clock folded in)",
                                  "status": "printed beside every accounting; NOT SEPARATED on rental B by construction "
                                            f"({lev['eta_mma rival (Table 9 chip-relative)']['lever_cells']} lever cells)"},
                        "sensitivity (build-tile-review 1)": eta_sensitivity()},
            "eta_smem": {"value": V4.MVA2["eta"], "status": "UNTESTED REUSE: MVA2's eta, fitted on mma.sync LSU wavefronts (rental "
                                                            "4, SEEN-fitted), divides shared bytes on the wgmma path too"},
            "accountings": {"B": "wgmma operand reads plus cp.async writes, both at 128 B/clk/SM on one port",
                            "W": "operand reads at 128 B/clk/SM, cp.async writes as LDGSTS wavefronts (1 per 512 B)",
                            "T": "tensor-bound only (MMA cycles)"},
        },
        "calibration (CAL-A)": {
            "pages": "f128: OLMoE and Mixtral 8x7B, 128x128x64 w8 s3, G 64 counter pages, treads 1..9 (R3's counter ladder)",
            "estimator": "tcommon.joint_cF: sm__cycles_elapsed.avg = a(page model, GEMM, arm) + ceil(N_live / 132) (S_g c + F_g), "
                         "one c for both GEMMs and both models, F_w1 and F_w2, by least squares over the NATIVE and SHARED cells at "
                         "n 4..9 whose GEMM reads DRAM at no more than 0.5 x 4022 GB/s (each cell's own GEMM's dram_bytes_read "
                         "over its gpu_time_ns, design-tile-review 9.4); OLMoE S 32 / 16 and Mixtral S 64 / 224 give four depths",
            "outputs": "c_128, F_wg,w1, F_wg,w2 (3 CAL-A constants), then Z_wg per accounting at the census occupancy "
                       "(r3_timing_model_v4.z_wg); an accounting whose pipe alone exceeds c_128 is FALSIFIED by construction",
            "refusal": "fewer than 6 floor-bound cells or a rank-deficient design: NOT SCORED, and rental B does not run (no v4)"},
        "test (BLIND-CALMODEL, DESCRIPTIVE)": {
            "pages": "f64w8: OLMoE and Mixtral 8x7B at 64x128x64 w8 s3, G 64; f64w4 (drop-group w4): OLMoE at w4, printed",
            "statistic": "per (model, GEMM): u = S c + F; u_meas from the 64x128 joint fit (same estimator), u_pred from each "
                         "accounting's c_64 and the registered F form; e = u_pred / u_meas - 1",
            "rule": "an accounting HOLDS when every |e| <= 3% (design-tile-review 9.4: within 3% of u, on the floor-bound n 4..9 "
                    "cells of each page)",
            "status": "DESCRIPTIVE (design-tile-review 1c): the accounting reading has no consequence for E1-TILE; B, W and T move "
                      "rental B's predictions by at most "
                      f"{max(lev['accounting B']['max_abs_pct'], lev['accounting T']['max_abs_pct'])}% at the nominal constants",
            "w4": "the w4 page is printed only (w8 and w4 issue different instructions and their per-iteration delay may differ, "
                  "so T's w8 / w4 = 1.00 is not a hard prediction); when dropped nothing changes",
            "mma_sync_32x128_w4": "MVA2 on the census wavefronts at 32x128x64 w4 s3: HOLDS / FAILS within 3% of the page's c, "
                                  "printed for D1 (the decode tiles' only floor evidence); MVA2 is SEEN-fitted (D5)"},
        "selection_for_rental_B (D3)": {
            "accounting": "tcommon.variant_rule: the accounting that HOLDS on every scored (model, GEMM) of the 64x128 w8 pages "
                          "while every other FAILS or is unreachable is SELECTED and rental B's v4 derives c_64 from it; otherwise "
                          "UNDECIDED and rental B's v4 takes v4cal at 64x128 (the joint fit's c_64, F_64,w1, F_64,w2, CAL-A: 3 "
                          "conditional constants). At 128x128 v4 and v4cal are the same model (both read c_128 and F_wg)",
            "F_form": "tcommon.f_form_rule (above)",
            "unselected": "the other accountings, F-cyc and the eta_mma rival are printed as rivals on rental B; lever counts at "
                          "the nominal constants below (fewer than 6: NOT SEPARATED by construction, stated before the page)",
            "lever_counts_on_B": lev},
        "parameters": {
            "hardware (no fit)": "132 SMs, the 1710 MHz lock, 4096 flop/clk/SM dense bf16, 128 B/clk/SM shared, the sm_90 "
                                 "occupancy limits",
            "derived or read (no fit)": "geometry, k-steps, N-tiles, grid, dead CTAs, bytes per k-step and wavefronts from the "
                                        "census PTX, occupancy from the census cubins",
            "carried from v3": "22 fitted (T2) + d, kappa, h",
            "fitted, new in the timing model: 6": ["MVA2 eta, z0, z1 (SEEN-fitted at rental 4; mma.sync, D5)",
                                                   "Z_wg, F_wg,w1, F_wg,w2 (CAL-A on the 128x128 counter pages; c_128 is the "
                                                   "calibration's measured c, the analogue of v3's 344.1)"],
            "selections: 3": ["the wgmma accounting B / W / T (DESCRIPTIVE reading, rule above)",
                              "the A key v3 / CLAMP / ROWKEY60 / ROWKEY30 (tileA-bytes)", "F-step / F-cyc (rule above)"],
            "uncertain inputs": ["eta_mma (Luo Table 11 per clock, H800 PCIe FP16; BF16 on GH200 assumed equal; rival Table 9's "
                                 "chip-relative 0.872 / 0.857)",
                                 "eta_smem = MVA2's eta, an untested reuse"],
            "conditional": ["v4cal adds 3 CAL-A constants at 64x128 when no accounting is selected",
                            "ROWKEY60's 60 MiB is SEEN-chosen if it is NAMED"]},
        "design_tables (derived, DESCRIPTIVE)": variant_tables(),
        "mma_sync_design (derived, DESCRIPTIVE)": mma_sync_predictions(),
        "nominal": NOMINAL_LABEL,
        "labels": {"c_128, F_wg, Z_wg": "CAL-A", "64x128 test": "BLIND-CALMODEL, DESCRIPTIVE", "MVA2": "SEEN-fitted (D5)",
                   "eta_mma": "SEEN third party, another card and dtype (assumption registered)"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["MVA2's eta, z0, z1", "the CAL F 520 / 979 and c 344.1 (R3's tile)", "Luo et al. Tables 9 and 11"],
    }


# --------------------------------------------------------------------------
# tileA-bytes
# --------------------------------------------------------------------------

DEVIATION_OLMOE_N8 = ("the design's OLMoE 128x128 G 1 n 8 tread is removed (build-tile-review 5): M4 rules w2 at G 1 and n >= 6 "
                      "ILL-POSED on Hopper (wave_split_bytes.HOPPER_G1_N), applied per tread and literally at BLOCK_M 128 (n 4 "
                      "is already 512 rows per expert, past BLOCK_M 32's n 6 at 192), so an n-8 cell would have no v4 prediction; "
                      "OLMoE runs n 1, 2, 4, and the deepest A-heavy w1 row (n 8) is lost")


def _rowkey() -> dict:
    p = ST.OUT["A"] / "rowkey_tile.json"
    return json.loads(p.read_text())


def byte_predictions() -> list[dict]:
    rk = _rowkey()
    rkc = {(c["page"], c["label"], c["n"]): c for c in rk["cells"]}
    pr = pricer()
    out = []
    for _label, model, name in A_BYTE_PAGES:
        doc = hist("A", name)
        t = PAGE_TILE[int(doc["block_m"])]
        G = int(doc["group_m"])
        occ, _src = TC.occupancy_of(t, model)
        E = MODEL_CONFIGS[model].num_experts
        for c in doc["cells"]:
            n = int(c["n"])
            counts = None if c["counts"] is None else tuple(int(v) for v in c["counts"])
            uni = tuple([n * t.bm] * E)
            with V4.at_tile(model, t):
                q3 = WV4.q_v3(pr.eng, model, n, counts, COPIES, seed_of(doc, c), t, G, occ)
                bm = dict(TM.BYTE_MODEL)
            qi = WV4.q_ideal(model, n, counts if counts is not None else uni, t.bm)
            tiles = sum(-(-v // t.bm) for v in (counts if counts is not None else uni))
            to_q = lambda b, g, n=n, bm=bm: (b - n * bm[f"operand_per_tile_{g}"]) / bm[f"W_{g}"]  # noqa: E731
            r = rkc.get((name, c["label"], n))
            for arm in c["arms"]:
                q = {"v3": {g: r6(q3[g]) for g in q3} if q3 else None,
                     "CLAMP": {g: r6(max(q3[g], qi)) for g in q3} if q3 else None}
                for k, C in WV4.ROWKEY_C.items():
                    q[k] = ({"w1": r6(to_q(r[f"read_bytes_C{C:g}"], "w1"))} if (r is not None and G == 1) else None)
                out.append({"page": name, "model": model, "tile": t.key, "G": G, "n": n, "label": c["label"], "arm": arm,
                            "counts_sha256": R6.sha_counts(c["counts"]), "skewed": c["label"] in SKEWED,
                            "scored_for_key": G == 1, "q_pred": q, "q_ideal": r6(qi), "q_group_G1": r6(tiles / E) if G == 1 else None,
                            "label_blind": "BLIND-CALMODEL (CAL model); " + workload_label(model, n, t.bm, c["counts"])})
    return out


def bytes_() -> dict:
    pred = byte_predictions()
    rk = _rowkey()
    rk_path = ST.OUT["A"] / "rowkey_tile.json"
    scored = [r for r in pred if r["scored_for_key"]]
    sep = {}
    for a, b in (("v3", "CLAMP"), ("CLAMP", "ROWKEY60"), ("ROWKEY60", "ROWKEY30"), ("v3", "ROWKEY60")):
        g = [r["q_pred"][a]["w1"] / r["q_pred"][b]["w1"] - 1 for r in scored if r["q_pred"][a] and r["q_pred"][b]]
        sep[f"{a} vs {b}"] = {"rows": len(g), "rms_pct": r6(100 * math.sqrt(st.fmean(x * x for x in g))) if g else None,
                              "max_abs_pct": r6(100 * max(abs(x) for x in g)) if g else None}
    names = [n for _l, _m, n in A_BYTE_PAGES]
    return {
        "registered": REGISTERED, "name": TC.NAMES["bytes"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/tileA/score_bytes.py; tcommon.named_key; bytes scripts/wave_split_bytes_v4.py (M4 at the "
                "tile); row keys scripts/scoring/tileA/rowkeys_tile.py",
        "what": "the A key of v4 (D3): which w1 A accounting prices rental B's G-1 cells, where bytes reach time (design-tile 2.4)",
        "pages": {"units": [f"{m} bytes label={lab} (histogram {n}.json)" for lab, m, n in A_BYTE_PAGES],
                  "files": files_block("A", names),
                  "same_counts (design-tile-review 9.14)": "each byte page reads the SAME file as its (model, tile, G)'s timed page, "
                                                         "so their counts are identical by construction"},
        "keys": {"v3": "M4's content key (rental 6), unchanged",
                 "CLAMP": "v3 with each GEMM's q floored at the active weights once (q_ideal = experts with a row / E): a "
                          "physical floor, not a fit (design-tile-review 5). M4 prices uniform cells below one weight set at "
                          "BLOCK_M 128 (Qwen1.5 n 1 w1 q 0.967)",
                 "ROWKEY60": "rental 6's row-granular LRU at 60 MiB (rental 7's NAMED key, SEEN-chosen, OPEN)",
                 "ROWKEY30": "the same at 30 MiB: one of the GH200's two L2 partitions (its L2 read 60 MiB on rental 4's rulers, "
                             "CONSISTENT with RRZE's GH200 run; Hopper's L2 is two partitions, NVIDIA H100 whitepaper). The "
                             "design's 'Fan et al.' attribution is withdrawn (no source supports it; design-tile-review 9.13)"},
        "rowkey_rows": {"path": str(rk_path.relative_to(REPO)), "sha256": sha256_file(rk_path), "device": rk["device"],
                        "dtype": rk["dtype"], "cells": len(rk["cells"]), "wall_s": rk.get("wall_s"),
                        "occupancy": {k: {"w1": v["occupancy_w1"], "source": v["occupancy_source"]} for k, v in rk["pages"].items()},
                        "method": "streamed (rowkeys_tile.stream_misses): the pinned rival's access order and hit rule with "
                                  "bounded memory (peak 0.25 GB); the pinned rowkeys.simulate's own run on this page was stopped "
                                  "at 19 GB of unified memory on 2026-10-14",
                        "equality_check": "the stream's read bytes EQUAL rowkeys.simulate's (CPU float64, and the MPS float32 "
                                          "run before it was stopped) on all 13 OLMoE 128 G 1 cells that run priced (n 1, 2, 4, "
                                          "8; 4.2 to 33.8 million accesses each); tests/test_tile_plumbing.py re-runs n 1",
                        "arms": "NATIVE priced, SHARED carried (the same walk at G 1)",
                        "recompute": "re-priced at the census occupancy if it differs from the design occupancy (tileA-census)"},
        "deviation": DEVIATION_OLMOE_N8,
        "byte_predictions": pred,
        "separations_on_scored_w1_rows": sep,
        "statistic": "per (cell, arm, GEMM): q_counted = (dram_bytes_read - n x operand_per_tile_g) / W_g at the tile's BYTE_MODEL; "
                     "e_K = q_pred_K / q_counted - 1",
        "rule (D3; design-tile-review 5)": "over EVERY scored w1 row (the G-1 pages: OLMoE 128 G 1, Mixtral 64 G 1, Qwen2-57B "
                                           "128 G 1; uniform, balanced and skewed, NATIVE and SHARED): the key nearest by rms is "
                                           "NAMED iff the next is at least 2x farther and every row of the nearest is within 5%; "
                                           "if none is NAMED v4 takes CLAMP (the physical floor), not v3. The NAMED key or CLAMP "
                                           "becomes rental B's w1 A key. Mixtral's G-16 rows and every w2 row are printed",
        "brackets_printed": "q_group (every M-tile re-reads its expert's slabs, G 1) and q_ideal (active weights once)",
        "B_BYTES_bar_printed": "rms <= 3% and every row <= 5% per key on the scored rows",
        "gates": {"clock": "rental 6's counter clock rule (lock_gate.check_page PASS, each cell in [1640, 1710] MHz)",
                  "G5": "histogram sha256, shuffle seed, every cell's counts sha256",
                  "tile": "the page's recorded block_m, block_n, block_k, num_warps, num_stages must be the unit's"},
        "noise": {"counter": {"within_board_pct": 0.3, "board_to_board_pct": [0.03, 0.66]}},
        "labels": {"rows": "BLIND-CALMODEL (CAL models); uniform and balanced rows may be SEEN-WORKLOAD (per row)",
                   "ROWKEY60": "SEEN-chosen capacity", "ROWKEY30": "derived from the measured L2"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["M4 (v3's bytes) SEEN-fitted on 8x7B CAL", "C 60 MiB SEEN-chosen", "the 60 MiB L2 (rental 4)"],
    }


# --------------------------------------------------------------------------
# tileA-e1cal
# --------------------------------------------------------------------------

def nominal_rows(pages, rental: str, *, with_sens: bool = False) -> list[dict]:
    prm = nominal_params()
    out = []
    for item in pages:
        label, model, name = item[:3]
        doc = hist(rental, name)
        t = PAGE_TILE[int(doc["block_m"])]
        for c in doc["cells"]:
            for arm in c["arms"]:
                p = price_cell(model, arm, c, doc, prm)
                row = {"unit": label, "model": model, "page": name, "tile": t.key, "G": int(doc["group_m"]),
                       "cls": TC.tile_class(t.bm, int(doc["group_m"])), "n": int(c["n"]), "label": c["label"], "arm": arm,
                       "counts_sha256": R6.sha_counts(c["counts"]), "ms_nominal": r6(p["ms"]),
                       "host_ok_nominal": TC.C6.host_ok(p["ms"]), "native_path": p["native_path"],
                       "A_proxy_us": p["A_us"], "dead_ctas": {g: p["gemm"][g]["dead_ctas"] for g in ("w1", "w2")},
                       "workload": workload_label(model, int(c["n"]), t.bm, c["counts"])}
                if with_sens and arm == "native":
                    row["dlnT_dlnB_nominal"] = r6(sensitivity(model, c, doc, prm))
                out.append(row)
    return out


def e1cal() -> dict:
    rows = nominal_rows(A_TIMED_PAGES, "A", with_sens=True)
    names = sorted({n for _l, _m, n in A_TIMED_PAGES})
    return {
        "registered": REGISTERED, "name": TC.NAMES["e1cal"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/tileA/score_e1cal.py; tcommon.bar, class_means, stop_rule; v4 at rental A's constants "
                "(scripts/scoring/tileA/finalise.py)",
        "what": "BLIND-CALMODEL in time: v4 as fixed by tA-census, tA-floor and tA-bytes' registered rules, end to end, on the CAL "
                "models' timed pages; the calibration read counters, not time. Its class means are rental B's stop rule (tileB-stop)",
        "pages": {"units": [f"{m} timed label={lab} (histogram {n}.json)" for lab, m, n in A_TIMED_PAGES],
                  "files": files_block("A", names),
                  "classes": {"64g1": "Mixtral 64x128 G 1 (n 1, 2)",
                              "128g1": "OLMoE (n 1, 2, 4), Qwen2-57B (n 1, 2), Mixtral (n 2): three CAL models",
                              "128g16": "Mixtral 128x128 G 16 (n 3, 4, 8)"},
                  "nodrop": "every E1-CAL page is nodrop (design-tile-review 4: the Qwen2-57B page and the Mixtral 128 G 1 n 2 "
                            "tread added for the model-specific check); a SKIPPED, LOST or NOT SCORED page is a STOP"},
        "predictions": "COMPUTED AT RENTAL A'S CONSTANTS by scripts/scoring/tileA/finalise.py (mechanical: the census occupancy, "
                       "c_128 and F_wg, the selected accounting or v4cal, the F form and the A key, all by the registered rules); "
                       "the DESIGN-NOMINAL numbers below show the scale only",
        "deviation": DEVIATION_OLMOE_N8,
        "nominal_cells": rows,
        "statistic": "e = T_v4 / T_meas - 1 per valid NATIVE cell (PT, PW, uniform and balanced); SHARED printed with the "
                     "page's own A",
        "rule": "the E1 time bar (rms <= 2%, worst <= 5%, |mean| <= 1%) on the pooled NATIVE cells: tA-e1cal HOLDS or FAILS. "
                "Per page and per class printed",
        "gates": {"G1": "rental7-g1rule (r7-median-run3 rebuilt from cells.csv; 6 of 9 clean; 10% non-clean cap)",
                  "G2": "the host rule on the DESIGN-NOMINAL prediction (T + F240 < 0.40 ms is host-bound) AND the timer flag; "
                        "every CAL cell here passes it",
                  "G4": "cell clock >= 1705 MHz", "G5": "provenance (file, counts, shuffle seed, bincount)",
                  "tile": "the page's recorded tile (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) is the unit's",
                  "G3": "rental 7's TOST on uniform minus balanced, strict; a FAIL voids every histogram cell and E1 is printed "
                        "DESCRIPTIVELY (design-tile 4); a voided tA-e1cal is a STOP (design-tile-review 9.9)",
                  "views": "ALL and CLEAN; the stop rule reads CLEAN"},
        "stop_rule": "tileB-stop",
        "nominal": NOMINAL_LABEL,
        "labels": {"time": "BLIND-CALMODEL", "workload": "per cell (SEEN-WORKLOAD on uniform and balanced cells whose counts a "
                                                          "published page timed)"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["the CAL models' R3-tile pages", "v4's carried v3 constants"],
    }


# --------------------------------------------------------------------------
# tileA-dead
# --------------------------------------------------------------------------

DEAD_LAWS = {"D": (0.995, 0.995, 0.325), "D2": (0.93, 1.03, 0.30), "SLOT": ("slot", "slot", 0.325),
             "kappa 0.15": (0.995, 0.995, 0.15), "kappa 0": (0.995, 0.995, 0.0), "d 1.06": (1.06, 1.06, 0.325)}


def dead_increments(c128: float) -> dict:
    """SHARED c15 minus c9 dead-CTA exposure at OLMoE 128x128 G 1 n 1 (uniform), us, per law, at
    occupancy 2 and c_time = c_card x c128 / 344.1, PHI the F-step PHI of F 520 / 979 at c128."""
    model, t = "olmoe-1b-7b", V4.CAL_TILE
    prm = nominal_params(c128=c128)
    occ = TC.occupancy_of(t, model)[0]["w1"]
    E = MODEL_CONFIGS[model].num_experts
    counts = tuple([t.bm] * E)
    c_ns = float(pricer().f2.x[1]) * V4.c_scale(t, "w1", prm)
    phis = {g: V4.phi(t, g, prm) for g in ("w1", "w2")}
    out = {}
    with V4.at_tile(model, t, phis):
        for name, (a, b, k) in DEAD_LAWS.items():
            tot = 0.0
            for D, sgn in ((15, 1), (9, -1)):
                for g, d in (("w1", a), ("w2", b)):
                    dd = 4.05 / occ if d == "slot" else d
                    dead = TM.dead_ctas(E * D, 1, g, counts)
                    hid = k * occ * TM.floor_ksteps(g) * c_ns
                    tot += sgn * max(0.0, dead * dd - hid) * 1e-3
            out[name] = r6(tot)
    return out


def dead() -> dict:
    inc = {f"c128 {c:g}": dead_increments(c) for c in (600.0, 900.0, 1100.0)}
    return {
        "registered": REGISTERED, "name": TC.NAMES["dead"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/tileA/score_dead.py; tcommon.dead_rule",
        "what": "the dead-CTA lever at occupancy 2 (design-tile 3.1 step 7, design-tile-review 7): SHARED at 15 against 9 copies "
                "on OLMoE 128x128x64 w8 G 1, where the hidden window kappa x occ x S' x c is half R3's",
        "pages": {"units": ["olmoe-1b-7b timed label=d9 (declared-copies 9)", "olmoe-1b-7b timed label=d15 (declared-copies 15)"],
                  "files": files_block("A", ["olmoe-1b-7b-dead"]), "drop_group": "dk (dropped first)",
                  "tread": "n 1, uniform and balanced (design-tile-review 7 and 9.12: the increment does not depend on n)"},
        "statistic": "per (label): delta = [(T_S - T_N) at c15 - (T_S - T_N) at c9] - [(A_S - A_N) at c15 - (A_S - A_N) at c9], "
                     "the second bracket the align growth read off each page's own align_probe (design-tile-review 9.12); "
                     "delta is the median over the page's (label) cells",
        "se": "se(delta) = 2 x sigma_page x mean(T) / sqrt(cells): four timed cells per difference; sigma_page is tA-rep's when "
              "it ran, else rental 3's 0.18% (design-tile-review 9.8)",
        "laws_predicted_us (derived from the v2 forms at the tile)": inc,
        "rule": {"D against SLOT": "each law's z = (delta - law) / se; EXCLUDED when |z| > 3. D (0.995 ns, kappa 0.325) and SLOT "
                                   "(4.05 ns / occ) differ by 12 to 17 us here (at R3's occupancy they did not separate)",
                 "kappa 0.325 against smaller kappa": "scored only when the measured c_128 >= 900 cycles (below it every kappa "
                                                      "hides nothing and the laws coincide); the comparison is D against "
                                                      "'kappa 0.15' and 'kappa 0'",
                 "NOT SEPARABLE (stated before the page)": ["D2 against D (v2): within 0.3 us at every c_128",
                                                            "kappa 0 against kappa 0.15: identical at every c_128 here"],
                 "consequence": "none for E1-TILE (NATIVE): the reading informs term (b)'s copies question (D9) and E1-TILE-S"},
        "labels": {"pages": "BLIND-CALMODEL (OLMoE is CAL-counters for d and kappa: the laws' constants are SEEN-fitted)"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["d 0.995, kappa 0.325 (v2, CAL)", "D2, SLOT, kappa 0.15, d 1.06 (SEEN readings of rental 7, design-tile 6)"],
    }


# --------------------------------------------------------------------------
# tileA-rep
# --------------------------------------------------------------------------

def rep() -> dict:
    return {
        "registered": REGISTERED, "name": TC.NAMES["rep"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/tileA/score_rep.py; tcommon.rep_sigma, sigma_page_of",
        "what": "sigma_page at the 128x128 tile: OLMoE's E1-CAL page run twice (e128g1, e128g1r), the same file",
        "statistic": "sigma_page = rms over the cells valid on both pages of (T_2 / T_1 - 1), over sqrt 2",
        "uses": ["rental B's rival se (tcommon.rival_rule)", "tA-dead's se", "rental B's power, re-run at finalisation"],
        "fallback (design-tile-review 9.8)": "when the replicate is dropped (drop-group rep), lost, or has fewer than 3 shared "
                                             "cells: rental 3's rep8 sigma_page 0.18% (R3's tile), labelled as such",
        "labels": {"pages": "SEEN-WORKLOAD replicate (one draw, twice): no model test"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["rental 3's 0.18%"],
    }


# --------------------------------------------------------------------------
# tileB-e1tile
# --------------------------------------------------------------------------

def b_rows() -> list[dict]:
    rows = nominal_rows(B_TIMED_PAGES, "B", with_sens=True)
    prm_v3 = nominal_params(law="v3own")
    for r in rows:
        doc = hist("B", r["page"])
        c = next(x for x in doc["cells"] if x["label"] == r["label"] and int(x["n"]) == r["n"])
        r["label_blind"] = ("SEEN-SHAPE (new model, seen shapes: D7)" if r["model"] == DSV else "BLIND-TILE") + "; " + r["workload"]
        if r["arm"] == "native":
            p = price_cell(r["model"], "native", c, doc, nominal_params())
            t = PAGE_TILE[int(doc["block_m"])]
            rv = {"v3own": price_cell(r["model"], "native", c, doc, prm_v3)["ms"],
                  "roofline": roofline_ms(r["model"], r["n"], c["counts"], doc, p["reads"], V4.eta_mma_of(t))}
            r["rivals_nominal_ms"] = {k: r6(v) for k, v in rv.items()}
    return rows


def power_cells(rows: list[dict]) -> list[dict]:
    return [{"model": r["model"], "page": r["page"], "cls": r["cls"], "s": r["dlnT_dlnB_nominal"]}
            for r in rows if r["arm"] == "native"]


def power_block(rows: list[dict]) -> dict:
    cells = power_cells(rows)
    pts = PT.grid(cells, BLIND_TILE, device="cpu")
    for v in pts.values():
        v.pop("seconds_on_device")
    chk = ST.OUT["B"] / "power_tile_check.json"
    mps = ({"path": str(chk.relative_to(REPO)), "sha256": sha256_file(chk), **{k: v for k, v in json.loads(chk.read_text()).items()
                                                                           if k in ("torch", "device", "dtype", "R", "max_abs_diff", "seconds")}}
           if chk.exists() else {"status": "NOT RUN"})
    return {"model": "design-tile-review work/power_rb075.py's, re-run on the registered 84 cells (power_tile.py): ln(T_meas / "
                     "T_v4) = common-mode tile bias + model-specific bias + cell 0.6% + bytes 1.75% x dln T / dln B (each cell's, "
                     "DESIGN-NOMINAL) + page 0.18% + board 0.26%",
            "cells": cells, "blind_models": list(BLIND_TILE), "noise": PT.NOISE, "R": PT.R_DEFAULT, "device": "cpu",
            "dtype": "float64", "points": {k: {x: (round(y, 5) if isinstance(y, float) else y) for x, y in v.items()} for k, v in pts.items()},
            "mps_check": mps,
            "reading": "P(E1-TILE) rests on the model-specific drift, which no rental-A page can bound for a blind model; the "
                       "common-mode part is what the stop rule catches"}


def ambiguity_block() -> dict:
    return AMBIGUITIES


_E1TILE: list = []


def e1tile() -> dict:
    if _E1TILE:
        return _E1TILE[0]
    doc = _e1tile()
    _E1TILE.append(doc)
    return doc


def _e1tile() -> dict:
    rows = b_rows()
    nat = [r for r in rows if r["arm"] == "native"]
    names = sorted({n for _l, _m, n, _g, _e in B_TIMED_PAGES} | {n for _l, _m, n, _g, _e in B_BYTE_PAGES})
    lev = {R: TC.lever_count([r["ms_nominal"] for r in nat], [r["rivals_nominal_ms"][R] for r in nat]) for R in ("v3own", "roofline")}
    est = sum(e for *_x, e in B_TIMED_PAGES) + sum(e for *_x, e in B_BYTE_PAGES) + 3 + 2 + 4
    return {
        "registered": REGISTERED, "name": TC.NAMES["e1tile"],
        "status": "FROZEN BEFORE ANY TILE PAGE; FINALISED MECHANICALLY FROM RENTAL A'S SELECTIONS (D3): the cells, files, rules, "
                  "rivals and gates are fixed here; the numeric v4 predictions are written by scripts/scoring/tileA/finalise.py "
                  "from rental A's scored selections, by the registered rules, before rental B runs, into "
                  "docs/registered/2026-10-14-tileB-predictions-gh200.json (a test holds finalise.py to synthetic selections)",
        "design": DESIGN,
        "tool": "scorer scripts/scoring/tileA/score_tileB.py; tcommon.e1tile_rule, rival_rule, bar; predictions finalise.py",
        "what": "the blind end-to-end test at the tile vLLM picks (D1: prefill, the two reachable wgmma tiles): v4 from CAL models "
                "to held-out models at the same tile, the transfer the study claims",
        "pages": {"timed_units": [{"unit": f"{m} timed label={lab}", "file": f"{n}.json", "drop": g, "est": e}
                                  for lab, m, n, g, e in B_TIMED_PAGES],
                  "byte_units": [{"unit": f"{m} bytes label={lab}", "file": f"{n}.json", "drop": g, "est": e}
                                 for lab, m, n, g, e in B_BYTE_PAGES],
                  "files": files_block("B", names),
                  "plan": {"prelude, calibrate, tilecensus (compile only, the blind models' configs on this board)": "nodrop",
                           "drop order, last first (design-tile-review 8, owner fix 10)": ["byt (the three byte pages)", "dsv"],
                           "nodrop": "every Phi, Qwen1.5 and JetMoE timed page (jg1 and qg16 included)",
                           "minutes": f"about {est} of units, about $5.3 at $2.29/h with boot and pushes"},
                  "cells": "per tread PT, PT2, PW, PW2 (two fresh draws of each shape at registered seeds), uniform and balanced; "
                           "NATIVE and SHARED at 9 copies; 14 treads, 84 NATIVE cells",
                  "excluded": "Granite (D8); every decode tile (D1)"},
        "leakage (design-tile-review 3)": {
            "blind": "time at vLLM's two wgmma tiles on the GH200 for Phi, Qwen1.5, JetMoE and deepseek-v2-lite; the fresh PT and "
                     "PW draws; deepseek-v2-lite's E 64, k 6 routing",
            "seen": ["each BLIND-TILE model's GEMM shapes (K, N), timed at R3's tile on GH200 (rentals 5 to 7; v3 is CERTIFIED on "
                     "Phi)", "deepseek-v2-lite: SEEN-SHAPE (D7): results/published/2026-09-01-nvidia_h200-alpha-surface-s4 times "
                     "it on H200 at BLOCK_M 32 and 64, BN 64, w8 s4, and its H and F (2048 / 1408) equal Qwen1.5's, so its "
                     "GEMMs are Qwen1.5's at 128x128 (N-tiles 22 and 16); its claim is 'new model, seen shapes'",
                     "the tiles: SEEN on the CAL models in rental A, and the blind models' cubins compiled there (compile only)",
                     "SEEN-WORKLOAD: on a uniform or balanced cell the routing counts depend only on M, so where a published "
                     "R3 page timed the model at the same rows per expert (a histogram page's control, or a plain ladder page's "
                     "balanced_ids at tread n x BLOCK_M / 32) the workload is SEEN and ONLY THE TILE IS BLIND: Phi 64g1 n 1 "
                     "(M 512), 128g1 n 1 and n 2 (M 1024, 2048) and 128g16 n 3 (M 3072); Qwen1.5 128g1 n 1 and n 2 (M 1920, "
                     "3840); JetMoE 128g1 n 2 (M 1024) (build-tile-review 4); every such cell is labelled per row below"],
            "rows": {lab: sum(1 for r in nat if r["workload"].startswith(lab)) for lab in ("SEEN-WORKLOAD", "BLIND counts")}},
        "nominal_cells": rows,
        "nominal": NOMINAL_LABEL,
        "tests": {
            "E1-TILE (primary, D2)": {
                "statistic": "e = T_v4 / T_meas - 1 per valid NATIVE cell (all six labels), v4 from predicted bytes at the finalised "
                             "constants; NATIVE has no term (b) and its dead term is about 1 to 2 us",
                "rule": "HOLDS iff the POOLED NATIVE cells AND each BLIND-TILE model's (Phi, Qwen1.5, JetMoE; D4: 'each shape' is "
                        "each model) meet rms <= 2%, worst <= 5%, |mean| <= 1%, in each view; the registered verdict holds only if "
                        "ALL and CLEAN hold",
                "deepseek-v2-lite (design-tile-review 9.6)": "in the pooled bar; its own bar is printed DESCRIPTIVE (one page: it "
                                                              "can never be CERTIFIED); dropped (dsv), it leaves the pool",
                "certification": "a BLIND-TILE model is CERTIFIED in CLEAN with its cells meeting the bar and at least 2 of its "
                                 "pages in CLEAN (Phi 3 pages, Qwen1.5 2, JetMoE 2: jg1 and qg16 are nodrop for this)",
                "a model left with one page": "its bar is scored and it is NOT CERTIFIED",
                "a model left with no valid cell (build-tile-review 9)": "E1-TILE reads NOT SCORED (incomplete), not FAILS, "
                                                                          "unless the pool or another blind model fails",
                "G3_void": "a strict G3 FAIL voids every histogram cell: E1-TILE NOT SCORED, printed DESCRIPTIVELY"},
            "E1-TILE-S (secondary)": "SHARED cells, v4 + term (b) as v3 (the page's own A) + the v2 dead term; the same bar; a "
                                     "registered verdict that certifies nothing",
            "E2-TILE": "NATIVE time from the byte pages' counted bytes (v4's timing alone) on the cells both pages carry; rental "
                       "6's E2 bar (rms <= 2%, worst <= 5%, |mean| <= 1.5%)",
            "B-BYTES": "byte rows of the byte pages under the finalised key: rms <= 3% and every row <= 5% (B-SKEW's bar), per GEMM",
            "G3": "rental 7's TOST on uniform minus balanced (clusters the pages, margin 0.36%), strict",
            "rivals": {"v3own": "v3 at its own constants, only the geometry moving (c 344.1, R3's PHI)",
                       "roofline": "T0 + sum over GEMMs of max(flops / (4096 x eta_mma x 132 x 1.71 GHz), bytes / bw)",
                       "OLS (design-tile-review 9.10)": "ONE ridge regression over every valid NATIVE E1-CAL cell of rental A "
                                                         "(classes pooled): T = a + b waves + c bytes / bw + d ln(1 + CTAs), waves "
                                                         "= sum over GEMMs of live CTAs / (132 x occ), bytes v4's predicted, CTAs "
                                                         "live over both GEMMs; standardised features, ridge lambda 1e-3 x "
                                                         "trace(X'X) / 3, intercept unpenalised; NOT SCORED under 12 cells or when "
                                                         "the standardised X'X has condition number over 1e6; frozen at finalisation",
                       "unselected accountings, F-cyc, eta_mma rival": "tileA-floor's lever counts: NOT SEPARATED by construction "
                                                                      "under 6 lever cells",
                       "v4cal": "NOT SEPARATED by construction (the same model at 128x128)",
                       "exclusion rule": "on lever cells (|T_R / T_v4 - 1| >= 2%, at least 6): R's rms over 2% and over v4's rms "
                                         "+ 3 se, se = sigma_page / sqrt(cells) (tcommon.rival_rule)",
                       "lever_cells_nominal": lev},
            "split views (DESCRIPTIVE)": {"G-16 cells": "E1 on the 128 G 16 cells alone (the floor law)",
                                          "G-1 n-1 cells": "E1 on the G 1 n 1 cells alone (the byte model at G 1)"}},
        "gates": {"G1": "rental7-g1rule (r7-median-run3 from cells.csv)", "repeats": "6 of 9 clean",
                  "G2 host (design-tile-review 9.7)": "rental 3's registered rule, T + F240 < 0.40 ms is host-bound, on the "
                                                      "FINALISED prediction (after rental A's selections), AND the timer flag; the "
                                                      "design's 0.3324 ms was a design-script threshold, never registered",
                  "G4": "cell clock >= 1705 MHz", "G5": "provenance", "tile": "the page's recorded tile is the unit's",
                  "census (tileA-census)": "a (model, tile) whose rental-B census occupancy differs from rental A's: NOT SCORED",
                  "views": "ALL and CLEAN (rental 2's rule 3)"},
        "power": power_block(rows),
        "labels": {"phi-3.5-moe, qwen1.5-moe-a2.7b, jetmoe-8b": "BLIND-TILE", DSV: "SEEN-SHAPE (D7)",
                   "uniform and balanced": "per row: SEEN-WORKLOAD where a published page timed the counts"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["every R3-tile page of the blind models", "deepseek-v2-lite's H200 page", "the design's sensitivities"],
    }


# --------------------------------------------------------------------------
# tileB-stop
# --------------------------------------------------------------------------

def stop() -> dict:
    pts = e1tile()["power"]["points"]
    return {
        "registered": REGISTERED, "name": TC.NAMES["stop"], "status": "REGISTERED BEFORE ANY PAGE (read on rental A's scores)",
        "design": DESIGN, "tool": "scripts/scoring/tileA/score_e1cal.py (tcommon.stop_rule), on rental A's CLEAN view",
        "rule (D6; design-tile-review 4 and 8; owner fixes 6 and 10)": {
            "A1": "tA-e1cal's pooled NATIVE bar HOLDS",
            "A2": f"every tile class's mean NATIVE residual (64 G 1, 128 G 1, 128 G 16) within {100 * TC.STOP_CLASS_MEAN:.2f}%",
            "A3": f"at 128 G 1, the CAL models' means (OLMoE, Qwen2-57B, Mixtral n 2) within {100 * TC.STOP_PAIR:.0f}% of each "
                  "other (the model-specific check, design-tile-review 4)",
            "pages": "every E1-CAL page ran and was scored: a SKIPPED, LOST or NOT SCORED E1-CAL page is a STOP",
            "G3": "a G3 void of the E1-CAL cells is a STOP",
            "census": "rental A's census refused (an MMA path mismatch) or missing is a STOP (build-tile-review 9)",
            "finalise": "scripts/scoring/tileA/finalise.py refuses (exit 2, nothing written) unless this rule reads GO",
            "verdict": "GO only when every clause holds; otherwise STOP: rental B does not run (D6), and the owner is told why"},
        "power (from tileB-e1tile, CPU float64)": pts,
        "reading": "at zero drift the rule stops about 1 rental in 10 for noise alone; at 1% common-mode drift it stops most; "
                   "the model-specific drift passes it as often as not (no rental-A page sees a blind model's bias)",
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["the noise model's SEEN inputs (rentals 3 and 7)"],
    }


# --------------------------------------------------------------------------
# the 14 scorer ambiguities of design-tile-review section 9, resolved
# --------------------------------------------------------------------------

AMBIGUITIES = {
    "1 which c is the denominator, and F's units": "c_time = c_card x c_counter(tile) / 344.1 (344.1 the CAL counters' c); F is "
                                                   "held in counter cycles and PHI = F / c_counter(tile) in k-steps; at R3's tile "
                                                   "v4 is v3 exactly (r3_timing_model_v4.py --check)",
    "2 F-cyc against F-step": "F-step (v3's form) is registered; F-cyc replaces it only when, on the 64x128 w8 pages, it is nearer "
                              "the measured F on both GEMMs and F-step misses by more than 3 se on one (tcommon.f_form_rule)",
    "3 eta_mma": "per clock (Luo Table 11, 0.991), per instruction by Table 9's ratios: 0.983 (m64n128k16, 128x128 w8), 0.965 "
                 "(m64n64k16, 64x128); H800 PCIe, FP16; BF16 on GH200 assumed equal; rival Table 9's chip-relative 0.872 / 0.857",
    "4 tA-floor": "the test statistic is u = S c + F per (model, GEMM), within 3%; cells are the NATIVE and SHARED cells at n "
                  "4..9 whose own GEMM reads DRAM at no more than 0.5 x 4022 GB/s; a dropped w4 page changes nothing (printed "
                  "only); at 128x128 v4cal and v4 are the same model",
    "5 the NAMED rule's rows, and the default": "every scored w1 row of the G-1 byte pages (uniform, balanced and skewed, NATIVE "
                                                "and SHARED); nothing NAMED gives CLAMP, not v3",
    "6 E1-TILE's 'each model'": "the three BLIND-TILE models; deepseek-v2-lite is in the pooled bar only (its own bar DESCRIPTIVE, "
                                "never CERTIFIED); a model left with one page is scored and NOT CERTIFIED",
    "7 the host cut": "0.40 ms (T + F240), rental 3's registered rule, on the FINALISED v4 prediction",
    "8 tA-rep dropped": "rental 3's 0.18% feeds the rivals' se, tA-dead's z and the power at finalisation",
    "9 tA-e1cal 'fails'": "on the CLEAN view: the pooled NATIVE bar, each class mean within 0.75%, the 128 G 1 CAL models within "
                          "1%; a SKIPPED, LOST or NOT SCORED E1-CAL page, or a G3 void, is a STOP (tileB-stop)",
    "10 the OLS rival": "one ridge OLS over every valid NATIVE E1-CAL cell (features waves, bytes / bw, ln(1 + CTAs); lambda 1e-3 x "
                        "trace(X'X) / 3 on standardised features); NOT SCORED under 12 cells or condition number over 1e6",
    "11 census against counter occupancy": "integers, exact; a counter page at another occupancy is scored at its recorded one "
                                           "(rental 4's re-key, score_floor: Z_wg at the 128x128 pages' recorded occupancy, each "
                                           "64x128 page's c at its own); on rental B a (model, tile) at another occupancy than "
                                           "rental A's census is NOT SCORED on every page of that (model, tile), and a missing or "
                                           "refused rental-B census leaves every page NOT SCORED (score_tileB.census_check); a "
                                           "refused rental-A census is a STOP",
    "12 tA-dead": "tread n 1 (uniform and balanced); align growth = (A_S - A_N) at c15 minus at c9 off each page's align_probe; "
                  "se = 2 sigma_page mean(T) / sqrt(cells), sigma_page tA-rep's or 0.18%",
    "13 the 30 MiB row key": "one of the card's two L2 partitions of its measured 60 MiB (rental 4's rulers, CONSISTENT with "
                             "RRZE's GH200; Hopper's partitioned L2, NVIDIA H100 whitepaper); the 'Fan et al.' attribution is "
                             "withdrawn",
    "14 byte pages and timed pages": "one histogram file per (model, tile, G): a byte page and a timed page of the same (model, "
                                     "tile, G) read the same file, so their counts are identical (the files' sha256 above)",
}

BUILDERS = {"census": census, "floor": floor, "bytes": bytes_, "e1cal": e1cal, "dead": dead, "rep": rep,
            "e1tile": e1tile, "stop": stop}


def _txt_census(d):
    return [f"{t['tile']}: {t['path']}, shared {t['shared_bytes']} B, design occupancy {t['design_occupancy']}"
            for t in d["tiles"]] + [f"SEEN PTX {x['tile']}: {x['counts']}" for x in d["seen_ptx"]["files"]]


def _txt_floor(d):
    out = []
    for name, v in d["design_tables (derived, DESCRIPTIVE)"].items():
        out.append(f"{name}: pipe {v['pipe_service_cycles']}")
        for row in v["by_c128"]:
            out.append(f"  c128 {row['c128']:g}: " + "; ".join(
                f"{k} " + (x if isinstance(x, str) else f"c64w8 {x['c_64x128_w8']:.0f} w4 {x['c_64x128_w4']:.0f}")
                for k, x in row.items() if k != "c128"))
    out += [f"lever {k}: {v}" for k, v in d["selection_for_rental_B (D3)"]["lever_counts_on_B"].items()]
    return out


def _txt_bytes(d):
    return [f"{k}: {v}" for k, v in d["separations_on_scored_w1_rows"].items()]


def _txt_e1cal(d):
    return [f"{r['page']} {r['label']} n{r['n']} {r['arm']}: {r['ms_nominal']} ms nominal" for r in d["nominal_cells"][:12]]


def _txt_dead(d):
    return [f"{k}: {v}" for k, v in d["laws_predicted_us (derived from the v2 forms at the tile)"].items()]


def _txt_rep(d):
    return [d["statistic"]]


def _txt_e1tile(d):
    return [f"power {k}: {v}" for k, v in d["power"]["points"].items()] + \
           [f"rival lever cells (nominal): {d['tests']['rivals']['lever_cells_nominal']}",
            f"leakage rows: {d['leakage (design-tile-review 3)']['rows']}"]


def _txt_stop(d):
    return [f"{k}: {v}" for k, v in d["rule (D6; design-tile-review 4 and 8; owner fixes 6 and 10)"].items()]


TEXT = {"census": _txt_census, "floor": _txt_floor, "bytes": _txt_bytes, "e1cal": _txt_e1cal, "dead": _txt_dead,
        "rep": _txt_rep, "e1tile": _txt_e1tile, "stop": _txt_stop}
