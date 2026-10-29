#!/usr/bin/env python
"""The decode stage's second addendum (docs/registered/2026-10-28-decode-addendum2-gh200): the rules the
owner's decisions of 2026-10-28 add, in one module for the registration builder, the two scorer wrappers
and the finalise wrapper. It imports the frozen decode modules (dcommon, finalise_decode) and changes
none of them: every rule here acts by selecting cells before a frozen function runs, or by reading a
DESCRIPTIVE row beside it.

THE RULES THIS FILE OWNS:
  second tile   a cell is DESCRIPTIVE (SIGMA-D) when its path is not naive and some active expert holds
                more than BLOCK_M rows (r3_decode.geometry's multi_tile_experts > 0 at vLLM's default
                config for T): v4d's schedule gives that expert a second M-tile and prices its slab
                re-read from DRAM (sigma 1). Computed from the registered pages' counts alone
  L2-served     v4d's price of the same cell with every second-or-later M-tile's CTAs reading no weight
                slab (their slab served from L2 at zero DRAM bytes), the rest of v4d unchanged: the same
                live CTAs, launch order, floor, tail, rho, dead term, B_other and constants. DESCRIPTIVE,
                never fitted, never selected, never a bar
  knee rows     v4d with r3_timing_model.P_KNEE at 5.75 and 3.75 in place of 14 at the same constants:
                DESCRIPTIVE rows beside the registered prediction
  power         the session's power-set.txt (nvidia-smi -q -d POWER after -pl, GPU Power Readings
                section) reads Current Power Limit 700 W within 1 W, else the decode readings are NOT SCORED
  SIGMA-D       position = (v4d - meas) / (v4d - served): FULL-REREAD under 1/3, MIXED to 2/3, L2-SERVED over;
                NOT READ when the two prices are under 3 us or 10% of the sigma-1 price apart
  plan          the CAL plan with the two decodebytes (dn) units nodrop, generated from the frozen
                generator's text, nothing else changed but the header
"""
from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcommon as DC  # noqa: E402

RD, V4, V4D, TM = DC.RD, DC.V4, DC.V4D, DC.V4.TM
r6 = DC.r6

DATE = "2026-10-28"
NAME = f"{DATE}-decode-addendum2-gh200"
#: the owner's decode power limit (2026-10-28), GPU scope; the driver's own read-back tolerance
POWER_LIMIT_W = 700.0
POWER_TOL_W = 1.0
#: the tile stage's softer knees (p_128 5.75, FINDINGS.md; p_64 3.75, round 5), for DESCRIPTIVE rows
KNEES = (5.75, 3.75)
CAL_PLAN_D2 = "scripts/plans/rentalD-cal-d2-2026-10.plan"
#: the units made nodrop by this addendum (owner, 2026-10-28, item 4)
NODROP_D2 = (("dn", "olmoe-1b-7b", "decodebytes"), ("dn", "mixtral-8x7b", "decodebytes"))
#: the certified domain under this addendum (owner wording, 2026-10-28, review M1), verbatim
SCOPE = ("Certified domain: decode calls in which every active expert fits in one M-tile (at most BLOCK_M rows). "
         "Calls where an expert needs a second M-tile (the most concentrated routing at T >= 32) are not certified; "
         "they are reported as the descriptive SIGMA-D row.")
#: SIGMA-D's registered reading (owner, 2026-10-28, review m9): position = (v4d - meas) / (v4d - served), cut at
#: 1/3 and 2/3; a cell is NOT READ when v4d's sigma-1 and served prices are closer than both MIN_GAP_REL of the
#: sigma-1 price and MIN_GAP_US (a 3% price error, E1's rms bar, then moves position by at most 0.3, one band)
POS_CUTS = (1.0 / 3.0, 2.0 / 3.0)
BANDS = ("FULL-REREAD", "MIXED", "L2-SERVED")
MIN_GAP_REL, MIN_GAP_US = 0.10, 3.0
#: the rentals: boot and setup 25 min (C4's figure); deadline launch + wall + 15 min rounded to 5 min;
#: the guardian's hard cap 15 min after the deadline (C3, C4 and S1's rule)
BOOT_MIN, RATE, DEADLINE_PAD_MIN, CAP_PAD_MIN = 25, 2.29, 15, 15
#: the decode registrations' wall factor (regdecode.WALL_FACTOR, rentals 4 to 7)
WALL_FACTOR = 1 / 0.84


# --------------------------------------------------------------------------
# the second-tile cells
# --------------------------------------------------------------------------

def second_tile(model: str, cell: dict) -> dict | None:
    """The cell's second-tile record when it is DESCRIPTIVE under this addendum, else None."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    g = RD.geometry(cfg, int(cell["T"]), cell["counts"])
    if g.path == RD.NAIVE or g.multi_tile_experts == 0:
        return None
    return {"model": model, "label": cell["label"], "T": int(cell["T"]), "role": cell.get("role"), "path": g.path,
            "tile": g.tile, "BLOCK_M": g.bm, "A": g.A, "live_M_tiles": g.L,
            "experts_over_BLOCK_M": g.multi_tile_experts, "largest_count": max(int(v) for v in cell["counts"])}


def excluded_cells(models, kind: str = "main") -> list[dict]:
    """Every DESCRIPTIVE cell of the models' registered pages of one kind, in page order."""
    out = []
    for m in models:
        if not (DC.REPO / DC.page_path(m, kind)).exists():
            continue
        for c in DC.page_doc(m, kind)["cells"]:
            rec = second_tile(m, c)
            if rec is not None:
                out.append(dict(rec, page=f"{m}-{kind}"))
    return out


def excluded_keys(models, kind: str = "main") -> set[tuple[str, str]]:
    return {(r["model"], r["label"]) for r in excluded_cells(models, kind)}


@contextlib.contextmanager
def without_cells(keys: set, kinds=("main",)):
    """dcommon.measured with the (model, label) cells in `keys` left out of every page of `kinds`
    (both of the frozen scorer's main-page reads: the main pages and the replicate)."""
    orig = DC.measured

    def measured(report, model, kind):
        out, dropped = orig(report, model, kind)
        if kind in kinds:
            out = {k: v for k, v in out.items() if (k[0], k[2]) not in keys}
        return out, dropped
    DC.measured = measured
    try:
        yield
    finally:
        DC.measured = orig


# --------------------------------------------------------------------------
# the L2-served estimate and the knee rows
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _served_windows():
    """r3_timing_model.window at G 1 with every CTA of an expert's second or later M-tile reading its
    activation share only (its slab removed). At G 1 the live CTAs are pid 0 .. L npn - 1 in order,
    M-row pid // npn, N-tile pid % npn (r3_timing_model.schedule)."""
    orig = TM.window

    def window(arm, declared, G, n, gemm, w, counts=None):
        r, leads = TM.schedule(arm, declared, G, n, gemm, counts)
        if counts is None or int(G) != 1:
            return orig(arm, declared, G, n, gemm, w, counts)
        geo = TM.GEOMETRY[gemm]
        t = TM.expert_tiles(counts)
        ends = np.cumsum(t)
        pid_m = np.arange(r.size, dtype=np.int64) // geo.npn
        owner = np.searchsorted(ends, pid_m, side="right")
        later = (pid_m - (ends[owner] - t[owner])) > 0
        r2 = np.asarray(r, dtype=float) - later * float(geo.slab)
        if (r2 < 0).any():
            raise ValueError("a later M-tile CTA read less than a slab: not v4d's G 1 schedule")
        m = TM.box_mean(r2, w)
        s = np.sort(m)
        return TM.Window(live=int(r2.size), leads=int(leads - later.sum()), reads=float(r2.sum()), sorted_mean=s,
                         cumsum=np.concatenate([[0.0], np.cumsum(s)]), mean=m,
                         rcum=np.concatenate([[0.0], np.cumsum(r2)]))
    window.cache_clear = orig.cache_clear      # r3_timing_model.set_model clears the caches by this name
    TM.window = window
    try:
        yield
    finally:
        TM.window = orig


def served_price(model: str, cell: dict, prm: V4D.DecodeParams) -> dict | None:
    """v4d's price with the second-tile slabs served from L2 (DESCRIPTIVE); None where v4d cannot price."""
    with _served_windows():
        return DC.price_cell(model, cell, prm)


@contextlib.contextmanager
def knee(p: float):
    saved = TM.P_KNEE
    TM.P_KNEE = float(p)
    try:
        yield
    finally:
        TM.P_KNEE = saved


def knee_price(model: str, cell: dict, prm: V4D.DecodeParams, p: float) -> dict | None:
    with knee(p):
        return DC.price_cell(model, cell, prm)


def sigma_d_row(model: str, cell: dict, meas_ms: float, prm: V4D.DecodeParams) -> dict:
    """One SIGMA-D row: the measured time against v4d (sigma 1) and against the L2-served estimate."""
    a, b = DC.price_cell(model, cell, prm), served_price(model, cell, prm)
    rec = second_tile(model, cell) or {}
    row = {"model": model, "label": cell["label"], "T": int(cell["T"]), "tile": rec.get("tile"),
           "experts_over_BLOCK_M": rec.get("experts_over_BLOCK_M"), "meas_ms": r6(meas_ms),
           "v4d_sigma1_ms": None if a is None else r6(a["ms"]), "served_ms": None if b is None else r6(b["ms"])}
    if a is not None and b is not None:
        row.update(e_sigma1=r6(DC.e_dec(a["ms"], meas_ms)), e_served=r6(DC.e_dec(b["ms"], meas_ms)),
                   served_over_sigma1=r6(b["ms"] / a["ms"]),
                   reads_sigma1=r6(sum(a["reads"].values())), reads_served=r6(sum(b["reads"].values())),
                   position=r6((a["ms"] - meas_ms) / (a["ms"] - b["ms"])) if a["ms"] != b["ms"] else None,
                   gap_rel=r6((a["ms"] - b["ms"]) / a["ms"]), gap_us=r6((a["ms"] - b["ms"]) * 1e3))
        row["band"] = band(row["position"], row["gap_rel"], row["gap_us"])
    else:
        row["band"] = "NOT READ: not priced"
    return row


def band(position: float | None, gap_rel: float, gap_us: float) -> str:
    """SIGMA-D's registered reading of one position (or a model's median position)."""
    if position is None or gap_rel < MIN_GAP_REL or gap_us < MIN_GAP_US:
        return (f"NOT READ: sigma-1 and served prices {gap_us:.2f} us ({100 * gap_rel:.1f}%) apart, under "
                f"{MIN_GAP_US:g} us or {100 * MIN_GAP_REL:g}%")
    return BANDS[0] if position < POS_CUTS[0] else BANDS[2] if position > POS_CUTS[1] else BANDS[1]


def sigma_d_summary(rows: list[dict]) -> dict:
    """Per model: the e of each price over its SIGMA-D rows (C5.stats), DESCRIPTIVE."""
    out = {}
    for m in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == m and r.get("e_sigma1") is not None]
        read = [r for r in mine if not r["band"].startswith("NOT READ")]
        med = r6(float(np.median([r["position"] for r in read]))) if read else None
        out[m] = {"cells": len(mine), "cells_read": len(read), "sigma1": DC.C5.stats([r["e_sigma1"] for r in mine]),
                  "served": DC.C5.stats([r["e_served"] for r in mine]), "position_median": med,
                  "band": (BANDS[0] if med < POS_CUTS[0] else BANDS[2] if med > POS_CUTS[1] else BANDS[1])
                  if read else "NOT READ: no readable cell"}
    return out


def knee_rows(cells: list[tuple[str, dict, float, float]], prm: V4D.DecodeParams) -> dict:
    """cells (model, page cell, measured ms, v4d ms at P_KNEE 14): e at each knee, against the
    provisional bars (DESCRIPTIVE), and per tile the mean price shift."""
    out = {}
    for p in KNEES:
        e, shift = [], {}
        for m, c, meas, base in cells:
            q = knee_price(m, c, prm, p)
            if q is None:
                continue
            e.append(DC.e_dec(q["ms"], meas))
            shift.setdefault(q["tile"], []).append(q["ms"] / base - 1)
        out[f"p {p:g}"] = dict(DC.bar(e, DC.BARS_PROVISIONAL), label="DESCRIPTIVE (not a bar; constants unchanged)",
                               shift_by_tile={k: {"cells": len(v), "mean": r6(float(np.mean(v))), "max": r6(float(max(v)))}
                                              for k, v in sorted(shift.items())})
    return out


# --------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------

def gpu_current_limit(text: str) -> float | None:
    """The GPU Power Readings section's Current Power Limit (W) of `nvidia-smi -q -d POWER`."""
    sec = False
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("GPU Power Readings"):
            sec = True
        elif sec and s.endswith("Power Readings"):
            return None
        elif sec and s.startswith("Current Power Limit"):
            m = re.search(r":\s*([\d.]+)\s*W", s)
            return float(m.group(1)) if m else None
    return None


def power_files(tree: Path) -> list[Path]:
    tree = Path(tree)
    found = set(tree.glob("session/power-set.txt")) | set(tree.parent.glob("session/power-set.txt"))
    return sorted(found)


def power_check(tree: Path) -> dict:
    """READ when exactly one session power-set.txt (<tree>/session or <tree>/../session) reads a GPU
    Current Power Limit of 700 W within 1 W; else NOT SCORED (the decode readings are not scored)."""
    files = power_files(tree)
    if len(files) != 1:
        return {"verdict": f"NOT SCORED: {len(files)} session power-set.txt files, not one", "files": [str(f) for f in files]}
    w = gpu_current_limit(files[0].read_text())
    ok = w is not None and abs(w - POWER_LIMIT_W) < POWER_TOL_W
    return {"verdict": "READ" if ok else f"NOT SCORED: the GPU Current Power Limit reads {w} W, not {POWER_LIMIT_W:g} W",
            "gpu_current_power_limit_w": w, "file": str(files[0]), "limit_w": POWER_LIMIT_W, "tolerance_w": POWER_TOL_W}


def not_scored(what: str, power: dict) -> dict:
    return {"what": what, "addendum": NAME, "scope": SCOPE, "verdict": "NOT SCORED", "POWER": power,
            "STOP": {"verdict": "STOP", "why": [f"power (addendum 2): {power['verdict']}"]}}


# --------------------------------------------------------------------------
# the plan and the rentals
# --------------------------------------------------------------------------

def cal_plan_d2() -> str:
    """The frozen CAL plan's units with the two dn units nodrop, under this addendum's header."""
    import finalise_decode as FN
    units = tuple((lab, m, kind, step, inst, "nodrop" if (lab, m, step) in NODROP_D2 else drop, est)
                  for lab, m, kind, step, inst, drop, est in DC.CAL_UNITS)
    hdr = ["# THE DECODE STAGE'S CAL RENTAL under the second addendum (2026-10-28): vLLM's decode call on OLMoE-1B-7B",
           f"# and Mixtral-8x7B. GENERATED by scripts/scoring/decode/finalise_decode_d2.py --cal-plan from registration {NAME}",
           "# over 2026-10-14-decodeC-cal-gh200's units; do not edit by hand. One gpu_1x_gh200 at $2.29/h, the decode driver:",
           "#   bash scripts/vm_run.sh start --ip <ip> --run-id <id> --deadline <epoch s> \\",
           f"#     --plan {CAL_PLAN_D2} --driver gh200_decode_session.sh",
           "# Runs now, without the tile stage's rental B (owner, 2026-10-28: DD6 superseded). Drop order, last first: rep,",
           "# decl, e240; every main (GRR) and ctl (GR1) page and both dn (decodebytes) units are nodrop. A SKIPPED or LOST",
           "# main or ctl unit is a STOP for the blind rental; the dn units feed BYTES-D, DESCRIPTIVE."]
    return FN.plan_text(units, hdr, census=True)


def rental(units_min: int) -> dict:
    wall = round(units_min * WALL_FACTOR + BOOT_MIN)
    deadline = 5 * round((wall + DEADLINE_PAD_MIN) / 5)
    cap = deadline + CAP_PAD_MIN
    return {"units_min": units_min, "boot_setup_min": BOOT_MIN, "wall_min": wall,
            "cost_usd": round(wall / 60 * RATE, 2), "rate": f"gpu_1x_gh200 at ${RATE}/h",
            "deadline_after_launch_min": deadline, "hard_cap_after_launch_min": cap,
            "cost_at_hard_cap_usd": round(cap / 60 * RATE, 2),
            "rule": "units (prelude, calibrate and the CAL census included) x 1/0.84 (rentals 4 to 7) plus 25 min boot "
                    "and setup; deadline launch + wall + 15 min, rounded to 5 min; the guardian's hard cap 15 min later"}


def all_cells_g3d(meas: dict, prm: V4D.DecodeParams | None, docs: dict, models) -> dict:
    """G3-D (shuffle, label, fill) as the frozen CAL scorer reads it, on every valid NATIVE main cell
    (the DESCRIPTIVE cells kept: each G3-D pair holds the same counts, or the same multiset, on both
    sides) at the given constants. meas {(model, label): ms}."""
    if prm is None:
        return {t: {"verdict": "NOT SCORED: no fit"} for t in ("shuffle", "label", "fill")}
    pred = {}
    for (m, lab) in meas:
        c = next((x for x in docs[(m, "main")]["cells"] if x["label"] == lab), None)
        p = DC.price_cell(m, c, prm) if c is not None else None
        if p is not None:
            pred[(m, lab)] = p["ms"]
    out = {}
    for test in ("shuffle", "label", "fill"):
        pairs = {m: DC.g3d_pairs(docs[(m, "main")], test) for m in models if (m, "main") in docs}
        out[test] = DC.g3d(meas, pred, pairs, test)
    return out


def main(argv=None) -> int:
    """The operator's power check: python scripts/scoring/decode/d2common.py --power <results tree>; exit 0 on READ."""
    import json
    a = list(argv if argv is not None else sys.argv[1:])
    if len(a) != 2 or a[0] != "--power":
        print(main.__doc__)
        return 2
    r = power_check(Path(a[1]))
    print(json.dumps(r))
    print(r["verdict"])
    return 0 if r["verdict"] == "READ" else 1


if __name__ == "__main__":
    sys.exit(main())
