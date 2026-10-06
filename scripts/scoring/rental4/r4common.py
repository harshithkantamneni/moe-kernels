"""Rental 4's shared rules (docs/registered/2026-10-06-rental4-*).

One module for the registration generator (register.py) and the scorers, so a rule is
computed one way on both sides. Pure functions over pages and the registered JSON; nothing
here fits a model at score time. The two-view machinery (ALL and CLEAN) and the R3 timed
page view are rental 3's (scripts/scoring/rental3/r3common.py), imported and never copied.
"""
from __future__ import annotations

import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C3 = _load("rental3_r3common", HERE.parent / "rental3" / "r3common.py")
CM2 = C3.CM2
CARD = C3.CARD
DATE = "2026-10-06"
#: the eviction trio and its registration were CUT before any page (build-r4-review E1)
PARTS = ("dead", "occlaw", "perturb", "stamps", "hw", "nativegates")
NAMES = {p: f"{DATE}-rental4-{p}-gh200" for p in PARTS}
SCORED = ("dead", "occlaw", "perturb", "stamps", "hw")
SMS = 132
LOCK_MHZ = 1710.0
#: 8x7B 2026-09-27 fit (CAL), ns per k-step at 1710 (docs/registered/2026-10-05-rental3-e2e "c")
C_NS = 202.547644
#: the SEEN cubins' occupancy (CTAs per SM, w1 / w2) of OLMoE-relevant 32x64 w8 configs
#: (moe.instrumented.cubin over results/published; REVIEW.md section 0)
OCC = {"k64s4": (5, 4), "k32s4": (5, 5), "k128s4": (3, 3), "k64s6": (3, 3), "k64s2": (4, 4),
       "k64s3": (5, 4), "k64s8": (2, 2)}
KNOBS = {"k64s4": (64, 4), "k32s4": (32, 4), "k128s4": (128, 4), "k64s6": (64, 6), "k64s2": (64, 2),
         "k64s3": (64, 3), "k64s8": (64, 8)}
F_CTA = {"w1": 520.0, "w2": 979.0}
STAGE_BYTES = (32 + 64) * 64 * 2   # one CTA k-step's A + B tiles, 32x64x64 bf16
BW_GBPS = 3598.0   # the 8x7B fit's bw (CAL)
WF64 = 282.12   # counted LSU wavefronts per BK 64 k-step (PRINCIPLES.md section 1, percta.txt)

median = CM2.median
no_data = CM2.no_data
combine = CM2.combine


def r6(x):
    return None if x is None else round(float(x), 6)


def registration(repo: Path, part: str) -> dict:
    return json.loads((Path(repo) / "docs" / "registered" / f"{NAMES[part]}.json").read_text())


# --------------------------------------------------------------------------
# the k-step laws (occlaw and stamps K)
# --------------------------------------------------------------------------

def W(bk: int) -> float:
    return WF64 * bk / 64


def mva(s: float, Z: float, N: int) -> float:
    """Exact MVA of a closed two-station queue (pipe service s, delay Z, N jobs): per-job
    time between completions at the pipe, i.e. 1 / throughput."""
    Q = 0.0
    X = None
    for n in range(1, N + 1):
        R = s * (1 + Q)
        X = n / (R + Z)
        Q = X * R
    return 1 / X


def law_c(law: str, p: dict, bk: int, stages: int, occ: int) -> float | None:
    """Per-SM cycles per CTA k-step at (BLOCK_K, num_stages, CTAs per SM) under a registered
    law with its registered parameters `p`; None where the law says nothing."""
    if law == "MVA2":
        return mva(W(bk) / p["eta"], max(p["z0"] + p["z1"] * bk / 64, 0.0), occ)
    if law == "LK":
        return max(W(bk) / p["eta"], p["L_k"])
    if law == "PS":
        return p["c64"] * bk / 64
    if law == "LITTLE":
        return max(W(bk) / p["eta"], 3 * p["L_k"] / max(stages - 1, 1))
    if law == "OCC":
        return (p["c0"] + p["lam"] / occ) if bk == 64 else None
    raise KeyError(law)


def ratio_pred(law: str, p: dict, knob: tuple, occ_k: int, base: tuple, occ_b: int,
               K: int, F: float) -> float | None:
    """(S c + F) at the knob over the same at the base: the per-CTA unit's ratio, which is
    the floor-bound cycles ratio at an unchanged grid."""
    ck = law_c(law, p, knob[0], knob[1], occ_k)
    cb = law_c(law, p, base[0], base[1], occ_b)
    if ck is None or cb is None:
        return None
    return (K / knob[0] * ck + F) / (K / base[0] * cb + F)


#: a counter cell is floor-bound for occlaw when its GEMM reads DRAM at no more than this
#: share of the 4022 GB/s counter peak (PRINCIPLES.md section 1; the SEEN G = 64 knob pages
#: read 0.19 to 0.32, REVIEW.md section 2)
DRAM_PEAK_GBPS = 4022.0
FLOOR_MAX_DRAM_FRAC = 0.5


def dram_frac(cell: dict, gemm: str) -> float | None:
    pg = (cell.get("per_gemm") or {}).get(gemm) or {}
    b, t = pg.get("dram_bytes_read"), pg.get("gpu_time_ns")
    if not b or not t:
        return None
    return float(b) / float(t) / DRAM_PEAK_GBPS


def page_c(page: dict, gemm: str, ns, F: float, max_dram_frac: float | None = None) -> dict | None:
    """sm__cycles_elapsed.avg = a_arm + ceil(N_live / SMs) (S c + F) on a counter page's NATIVE
    and SHARED cells at the treads `ns` (the estimator of reanalysis kstep_pages.py): c by
    least squares, F held. With `max_dram_frac`, only floor-bound cells (DRAM read rate at most
    that share of the peak) enter. None with fewer than 3 cells."""
    import numpy as np
    de = page.get("design") or {}
    bm, bk = int(de.get("block_m") or 32), int(de.get("block_k") or 64)
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[de["model"]]
    K = cfg.hidden_size if gemm == "w1" else cfg.intermediate_size
    S = K // bk
    rows, occ = [], set()
    for c in page.get("cells") or []:
        if c.get("arm") not in ("native", "shared") or int(c.get("n", 0)) not in ns:
            continue
        r = (c.get("recorded") or {}).get(gemm) or {}
        grid = ((c.get("per_gemm") or {}).get(gemm) or {}).get("grid_size")
        if not r or not grid or r.get("sm__cycles_elapsed.avg") is None:
            continue
        if max_dram_frac is not None:
            fr = dram_frac(c, gemm)
            if fr is None or fr > max_dram_frac:
                continue
        D = int(c["declared"])
        R = math.ceil((cfg.num_experts * int(c["n"]) * bm + D * (bm - 1)) / bm)
        npn = int(grid) // R
        live = cfg.num_experts * int(c["n"]) * npn
        occ.add(int(min(v for k, v in r.items() if k.startswith("launch__occupancy_limit"))))
        rows.append((c["arm"], math.ceil(live / SMS), float(r["sm__cycles_elapsed.avg"])))
    if len(rows) < 3:
        return None
    arms = sorted({a for a, _, _ in rows})
    X = np.array([[1.0 if a == arm else 0.0 for arm in arms] + [q * S] for a, q, _ in rows])
    y = np.array([v - q * F for _, q, v in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"c": float(coef[-1]), "S": S, "K": K, "cells": len(rows), "occupancy": sorted(occ),
            "block_k": bk, "num_stages": int(de.get("num_stages") or 4)}


# --------------------------------------------------------------------------
# dead CTAs (dead)
# --------------------------------------------------------------------------

def exposure_us(model: str, declared: int, gemm: str, d_ns: float, kappa: float, occ: int,
                c_ns: float = C_NS) -> float:
    """max(0, N_dead d - kappa occ S' c) in us: the dead-CTA exposure of one GEMM."""
    import r3_timing_model as TM
    old = TM.set_model(model)
    try:
        n = TM.dead_ctas(declared, 4, gemm)
        hid = kappa * occ * TM.floor_ksteps(gemm) * c_ns
        return max(0.0, n * d_ns - hid) * 1e-3
    finally:
        TM.set_model(old)


def dead_counts(model: str, declared: int) -> dict:
    import r3_timing_model as TM
    old = TM.set_model(model)
    try:
        return {g: int(TM.dead_ctas(declared, 4, g)) for g in TM.GEMMS}
    finally:
        TM.set_model(old)


def experts(model: str) -> int:
    from moe.spec import MODEL_CONFIGS
    return int(MODEL_CONFIGS[model].num_experts)


def arm_ms(report: dict) -> dict:
    return C3.treads_ms(report)


def align_ms(report: dict) -> dict:
    """{(label, tread): median ms} off an R3 page's align_probe."""
    out: dict = {}
    for c in ((report.get("align_probe") or {}).get("cells") or []):
        if c.get("ms") is not None:
            out.setdefault((str(c["label"]), int(c["tread"])), []).append(float(c["ms"]))
    return {k: st.median(v) for k, v in out.items()}


def z(meas: float, pred: float, sigma_noise: float, sigma_rival: float = 0.0) -> float:
    return (meas - pred) / math.sqrt(sigma_noise ** 2 + sigma_rival ** 2)


# --------------------------------------------------------------------------
# the perturbation gate (decision 2); instr_probe.gate_verdict is the same rule (a test)
# --------------------------------------------------------------------------

def perturb_verdict(ratios, configs, *, upstream_ok=True, sass_equal=None, hint_ok=None, tol=None) -> dict:
    """The gate is ONE function, instr_probe.gate_verdict, on the VM and here."""
    import instr_probe as IP
    return IP.gate_verdict(ratios, configs, upstream_ok=upstream_ok, sass_equal=sass_equal,
                           hint_ok=hint_ok, tol=tol or IP.PERTURB_TOL)


# --------------------------------------------------------------------------
# stamps
# --------------------------------------------------------------------------

def nearest(value: float, rivals: dict, log: bool = False) -> dict:
    """The rival nearest `value`; SELECTED when it is nearer than half the gap to the next
    nearest (in log space when `log`), else BETWEEN."""
    if value is None or (log and value <= 0):
        return {"verdict": "NOT SCORED", "rivals": rivals}
    f = (lambda x: math.log(x)) if log else (lambda x: x)
    ds = sorted((abs(f(value) - f(v)), k) for k, v in rivals.items() if v is not None and v > 0)
    if not ds:
        return {"verdict": "NOT SCORED", "rivals": rivals}
    best = ds[0][1]
    if len(ds) == 1:
        return {"verdict": best, "rivals": rivals}
    gap = abs(f(rivals[best]) - f(rivals[ds[1][1]]))
    sel = ds[0][0] < gap / 2
    return {"verdict": best if sel else f"BETWEEN {best} and {ds[1][1]}", "nearest": best,
            "distance": ds[0][0], "gap": gap, "rivals": rivals}


def load_stamps(unit_dir: Path) -> tuple[dict, list[dict]]:
    """A stamps unit's manifest and, per launch record, the decoded rows and phases."""
    import numpy as np

    from moe import instrumented as I
    man = json.loads((Path(unit_dir) / "stamps.json").read_text())
    out = []
    for rec in man.get("launches") or []:
        if "file" not in rec:
            continue
        cols = I.decode(np.load(Path(unit_dir) / "stamps" / rec["file"]), int(rec.get("marks") or 0))
        out.append({**rec, "cols": cols, "phases": I.phases(cols)})
    return man, out


def live_mask(cols):
    return cols["kind"] == 1


def dead_mask(cols):
    return cols["kind"] == 2


def med_of(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None and x >= 0]
    return st.median(xs) if xs else None


def steady_iter_cycles(launch: dict, skip_head: int = 2, skip_tail: int = 1) -> float | None:
    """Median per-iteration clock64 cycles over live CTAs and iterations past the first
    `skip_head` marks and before the last `skip_tail` (the pipeline's fill and drain)."""
    it = launch["phases"]["iter_cycles"]
    live = live_mask(launch["cols"])
    vals = []
    for row in it[live]:
        good = [int(x) for x in row if x >= 0]
        vals += good[skip_head:len(good) - skip_tail if skip_tail else None]
    return med_of(vals)


def tail_ctas(launch: dict, sms: int, occ: int):
    """The FINAL PARTIAL WAVE (build-r4-review section 3): with N live CTAs and a wave of
    sms x occ, the last N - floor(N / wave) x wave live CTAs in start order; None when N is a
    whole number of waves (no tail) or under one wave."""
    import numpy as np
    cols = launch["cols"]
    idx = np.flatnonzero(live_mask(cols))
    wave = sms * occ
    n = len(idx)
    full = n // wave
    if full < 1 or n == full * wave:
        return None
    order = idx[np.argsort(cols["start_ns"][idx], kind="stable")]
    return order[full * wave:]


def tail_iter_cycles(launch: dict, sms: int, occ: int) -> tuple[float | None, int]:
    """Median per-iteration cycles of the tail wave's CTAs (past each one's first), and
    how many CTAs that wave holds."""
    t = tail_ctas(launch, sms, occ)
    if t is None:
        return None, 0
    vals = []
    for row in launch["phases"]["iter_cycles"][t]:
        good = [int(x) for x in row if x >= 0]
        vals += good[1:]
    return med_of(vals), len(t)


def dead_life_cycles(launch: dict) -> float | None:
    cols = launch["cols"]
    m = dead_mask(cols)
    return med_of(launch["phases"]["life"][m])


def timer_resolution_ns(launch: dict) -> float | None:
    """globaltimer's granularity on the launch: the least positive difference between the
    distinct start, prologue, epilogue and end globaltimer values it recorded."""
    import numpy as np
    cols = launch["cols"]
    v = np.concatenate([cols[k][cols[k] >= 0] for k in ("start_ns", "prologue_ns", "epi_ns", "end_ns")])
    u = np.unique(v)
    if len(u) < 2:
        return None
    d = np.diff(u)
    d = d[d > 0]
    return float(d.min()) if len(d) else None


#: the dispatch readout's admission (registered): at least this many qualifying dead CTAs
#: and a start span of at least this many timer ticks
DISPATCH_MIN_CTAS = 100
DISPATCH_MIN_TICKS = 20


def dead_dispatch_ns(launch: dict) -> dict:
    """ns per dead CTA from the dead CTAs that START AFTER THE LAST LIVE CTA ENDS (so the
    live drain, the kappa term, is out): (max - min of their start globaltimer) / (count -
    1). Admitted only with >= DISPATCH_MIN_CTAS such CTAs and a span >= DISPATCH_MIN_TICKS
    timer ticks (timer_resolution_ns); else value None with the reason."""
    cols = launch["cols"]
    live, dead = live_mask(cols), dead_mask(cols)
    if not live.any():
        return {"value": None, "why": "no live CTA"}
    last_end = cols["end_ns"][live].max()
    s = cols["start_ns"][dead & (cols["start_ns"] > last_end)]
    res = timer_resolution_ns(launch)
    out = {"qualifying": int(len(s)), "resolution_ns": res, "value": None}
    if len(s) < DISPATCH_MIN_CTAS:
        out["why"] = f"{len(s)} dead CTAs start after the last live end, under {DISPATCH_MIN_CTAS}"
        return out
    span = float(s.max() - s.min())
    out["span_ns"] = span
    if res is None or span < DISPATCH_MIN_TICKS * res:
        out["why"] = f"span {span:.0f} ns is under {DISPATCH_MIN_TICKS} timer ticks of {res} ns"
        return out
    out["value"] = span / (len(s) - 1)
    return out


#: the SASS checks each stamps readout needs (moe.instrumented.cubin.sass_checks), per GEMM
SASS_NEEDS = {"F": {"w1": ("start_before_first_ldg",), "w2": ("start_before_first_ldg", "epi_bracket")},
              "K": {"w1": ("loop_clock_reads",), "w2": ("loop_clock_reads",)},
              "T": {"w1": ("loop_clock_reads",), "w2": ("loop_clock_reads",)},
              "D": {"w1": ("start_before_first_ldg",), "w2": ("start_before_first_ldg",)}}


def sass_precondition(man: dict, block: str) -> tuple[bool, str]:
    """A stamps unit is scored only when its own compiled kernels' SASS puts the clock reads
    where the readout needs them (the review's bracket rule); else NOT SCORED."""
    need = SASS_NEEDS[block]
    rows = man.get("sass") or []
    for gemm, mrw in (("w1", False), ("w2", True)):
        mine = [r for r in rows if bool(r.get("MUL_ROUTED_WEIGHT")) == mrw]
        if not mine:
            return False, f"no SASS of the {gemm} kernel"
        for r in mine:
            ch = r.get("checks")
            if not ch:
                return False, f"the {gemm} SASS is unread (no cuobjdump on the VM)"
            for k in need[gemm]:
                v = ch.get(k)
                if k == "loop_clock_reads":
                    if not v or v < 2:
                        return False, f"{gemm}: {v} clock reads inside the k-loop, under 2"
                elif v is not True:
                    return False, f"{gemm}: SASS check {k} reads {v}"
    return True, "the clock reads bracket what the readout needs"


def write_score(out: Path, stem: str, res: dict, text_lines: list[str]) -> str:
    return C3.write_score(out, stem, res, text_lines)
