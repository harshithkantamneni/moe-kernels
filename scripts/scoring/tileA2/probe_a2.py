"""Tile rental A2's lock-hold probe rule (docs/registered/2026-10-20-tileA2-gates-gh200, gate P; DESIGN-A2 6 risk 2,
REVIEW-A2 1a, 1c and 6; OD11 as recommended). One module for the VM gate (scripts/tileA2_gate.py probe) and
the DESCRIPTIVE reading of the 1710 / 900 W pages (score_probe_a2.py).

THE PROBE: the two histogram pages of docs/registered/2026-10-20-tileA2-hist (Mixtral 128x128 G 16 at
treads 3 and 8, Qwen2-57B 128x128 G 1 at tread 2; every label, NATIVE and SHARED: 144 + 72 = 216 repeat
rows), timed at the rung's lock L (the ladder 1710, 1410, 1305) under locked_r3 --slip-policy cell
--min-clean-repeats 0, after one hw_rulers run at L (its bandwidths and matmul clock printed, DESCRIPTIVE).
A rung HOLDS iff EVERY probe page:
  P1  ran: exactly one report.json under gaps-<card>-<model>-<label>, rebuilt from its cells.csv;
  P2  locked_r3's outcome holds L (summary.json outcome.held == L: state held or held-with-slips; nearly
      vacuous under slip-policy cell, P3 and P4 do the work);
  P3  every cell's rebuilt clock (r7-median-run3: the median over its clean repeats) is at least
      L - 5 MHz, and no cell is LOST (under 6 clean of 9: its clock cannot be read; one LOST cell fails
      the rung: the owner's decision of 2026-10-20, the probe stays strict);
  P4  the r7-median-run3 non-clean share of the page's repeat rows is at most 10% (the G1 cap);
AND the rung's COUNTER CLAUSE holds (the owner's decision of 2026-10-20 on the re-review's MAJOR 1: the
counter pages, not the timed pages, are what sagged on rental A at 1710):
  P5  OLMoE's f128 counter page (128x128x64 w8 s3, G 64, treads 1 to 9, label lp<L>f) at L gives at least
      6 floor-bound cells (tcommon.floor_rows: NATIVE and SHARED, n 4..9, each GEMM's DRAM rate at most half
      the peak) that pass rental 6's counter clock rule in the rung's counter band: lock_gate.check_page
      PASS, each cell in band_of(L), the primary band, else the registered fallback, in the band gate's
      order. 6 is tcommon.joint_cF's own minimum; joint_cF itself needs two models (one page cannot
      separate c from the two F), so on one page the clause is joint_cF's cell count over the same rows.
ORDER AND EARLY STOP (the re-review's MINOR 6, the cheaper option): the driver runs the Mixtral page, the
Qwen2-57B page, then the OLMoE counter page, and reads the rule (`partial`) after each; at the first page
that fails, the rung's remaining pages are NOT run and the rung fails (they could not make it pass).
A probe page's TIME is read by no rule, no scorer and no person: `rung` returns clocks, states and
shares only (the record never carries ms_p50).
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402

TC = A2.TC
C3, C7 = TC.C3, TC.C7
NON_CLEAN_CAP = C7.NON_CLEAN_CAP
#: (model, histogram page, G) of the probe; the unit label of rung L is lp<L><m|q>
PROBES = (("mixtral-8x7b", "mixtral-8x7b-128g16-probe", 16, "m"), ("qwen2-57b-a14b", "qwen2-57b-a14b-128g1-probe", 1, "q"))


def label_of(lock: float, short: str) -> str:
    return f"lp{int(round(lock))}{short}"


def _summary(session: Path, report: dict) -> dict | None:
    tag = str(report.get("session_tag") or "")
    base = tag.rsplit("-lock", 1)[0] if "-lock" in tag else tag
    p = Path(session) / "locked-r3" / base / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def page_rule(results: Path, session: Path, model: str, label: str, lock: float) -> dict:
    """P1 to P4 on one probe page; clocks, states and shares only."""
    root = C3.find_gaps(results, model, label)
    pages = C3.timed_pages(root)
    if len(pages) != 1:
        return {"verdict": "FAIL", "why": [f"P1: {len(pages)} report.json under gaps-{A2.CARD}-{model}-{label}"]}
    path, rep = pages[0]
    rep_dir = Path(path).parent
    _r, raw = C7.read_page(rep_dir)
    r, why = C7.rebuild(rep, raw)
    if r is None:
        return {"verdict": "FAIL", "why": [f"P1: the page cannot be rebuilt ({why})"]}
    out = {"page": str(rep_dir), "lock_mhz_page": r.get("lock_mhz"), "why": []}
    if float(r.get("lock_mhz") or 0) != float(lock):
        out["why"].append(f"P1: the page ran at {r.get('lock_mhz')} MHz, not {lock:g}")
    sm = _summary(session, rep)
    held = ((sm or {}).get("outcome") or {}).get("held")
    out["locked_r3_held"] = held
    if held is None or float(held) != float(lock):
        out["why"].append(f"P2: locked_r3's outcome holds {held}, not {lock:g}")
    rows = r.get("treads_table") or []
    cells = []
    for t in rows:
        c = {"cell": f"{t.get('histogram')}/{t.get('arm')}/n{t.get('tiles')}", "mhz": t.get("sm_clock_load_mhz"),
             "clean": t.get("clean_repeats"), "of": t.get("repeat_rows"), "lost": bool(t.get("lost"))}
        cells.append(c)
        if c["lost"]:
            out["why"].append(f"P3: {c['cell']} LOST ({c['clean']} clean of {c['of']})")
        elif c["mhz"] is None or float(c["mhz"]) < float(lock) - A2.G4_BELOW_MHZ:
            out["why"].append(f"P3: {c['cell']} reads {c['mhz']} MHz, under {lock - A2.G4_BELOW_MHZ:g}")
    g = r["g1_r7"]
    out["non_clean"] = {"rows": g["non_clean_rows"], "of": g["repeat_rows"], "share": g["share"], "cap": NON_CLEAN_CAP}
    if g["share"] > NON_CLEAN_CAP + 1e-12:
        out["why"].append(f"P4: {g['non_clean_rows']} of {g['repeat_rows']} repeat rows non-clean ({100 * g['share']:.2f}%, over 10%)")
    clocks = [c["mhz"] for c in cells if c["mhz"] is not None]
    out["cells"] = cells
    out["min_cell_mhz"] = min(clocks) if clocks else None
    out["verdict"] = "PASS" if not out["why"] else "FAIL"
    return out


def smi_trace(session: Path, report: dict | None) -> dict:
    """DESCRIPTIVE: the NVML trace locked_r3 kept for the page (clock, power, reasons while busy)."""
    if report is None:
        return {}
    tag = str(report.get("session_tag") or "")
    base = tag.rsplit("-lock", 1)[0] if "-lock" in tag else tag
    p = Path(session) / "locked-r3" / base / "smi.csv"
    if not p.exists():
        return {"smi": "no smi.csv"}
    rows = list(csv.reader(p.open(newline="")))
    if len(rows) < 2:
        return {"smi": "empty"}
    head = [h.strip() for h in rows[0]]

    def col(name):
        i = next((k for k, h in enumerate(head) if h.startswith(name)), None)
        return [] if i is None else [x[i].strip() for x in rows[1:] if len(x) > i]
    num = lambda xs: [float(v.split()[0]) for v in xs if v and v.split()[0].replace(".", "", 1).isdigit()]  # noqa: E731
    util = num(col("utilization.gpu"))
    clk = num(col("clocks.current.sm"))
    pw = num(col("power.draw"))
    rs = col("clocks_event_reasons.active")
    busy = [i for i, u in enumerate(util) if u > 0]
    bc = [clk[i] for i in busy if i < len(clk)]
    return {"busy_samples": len(busy), "min_clock_mhz": min(bc) if bc else None,
            "median_clock_mhz": st.median(bc) if bc else None, "max_power_w": max(pw) if pw else None,
            "reasons_seen": sorted({rs[i] for i in busy if i < len(rs)})}


def rulers_clock(results: Path, lock: float) -> dict:
    """DESCRIPTIVE (REVIEW-A2 1c): the rung's hw_rulers matmul clock, the heaviest locked load the study names."""
    hits = sorted(Path(results).rglob(f"*-{A2.CARD}-rulers-lp{int(round(lock))}/rulers.json"))
    if len(hits) != 1:
        return {"rulers": f"{len(hits)} rulers.json for rung {lock:g}"}
    d = json.loads(hits[0].read_text())
    mm = ((d.get("timings") or {}).get("matmul") or {})
    return {"matmul sm_clock_load_mhz": mm.get("sm_clock_load_mhz"), "file": str(hits[0])}


COUNTER_MODEL, COUNTER_MIN_CELLS = "olmoe-1b-7b", 6


def counter_label(lock: float) -> str:
    return f"lp{int(round(lock))}f"


def counter_clause(results: Path, lock: float) -> dict:
    """P5 on the rung's OLMoE f128 counter page: clocks and counts only."""
    import r3_timing_model_v4 as V4
    t = V4.CAL_TILE
    out = {"page": None, "why": []}
    for which in ("primary", "fallback"):
        rules = A2.Rules.tileA2(lock, which)
        with A2.scoring(rules):
            p, path, w = TC.counter_page(results, COUNTER_MODEL, counter_label(lock), 64,
                                         tile=[t.bm, t.bn, t.bk, t.warps, t.stages])
            if p is None:
                return dict(out, verdict="FAIL", why=[f"P5: {w}"])
            clk = TC.C6.clock_rule(p, path)
            keep = [c for c, cc in zip(p.get("cells") or [], clk["cells"], strict=True) if cc["ok"]] if clk["page_ok"] else []
            rows = TC.floor_rows(dict(p, cells=keep))
        out["page"] = str(path)
        out[which] = {"band": [A2.r6(x) for x in rules.band], "check_page": clk["check_page"],
                      "cells_in_band": len(keep), "of": len(p.get("cells") or []), "floor_bound_in_band": len(rows),
                      "cell_mhz": [A2.r6(c["mhz"]) if c["mhz"] is not None else None for c in clk["cells"]]}
        if len(rows) >= COUNTER_MIN_CELLS:
            return dict(out, verdict="PASS", band_used=which)
    return dict(out, verdict="FAIL", why=[f"P5: {out['fallback']['floor_bound_in_band']} floor-bound cells in the band "
                                          f"(primary {out['primary']['floor_bound_in_band']}), under {COUNTER_MIN_CELLS}"])


def rung(results: Path, session: Path, lock: float, partial: bool = False) -> dict:
    """The rung at `lock`: HOLDS iff every probe page PASSES P1 to P4 and the counter clause P5 holds.
    `partial`: read only the pages already written (the driver's early stop); HOLDS then means nothing has
    failed yet."""
    pages = {}
    for model, _name, _G, short in PROBES:
        lab = label_of(lock, short)
        if partial and C3.find_gaps(results, model, lab) is None:
            continue
        rec = page_rule(results, session, model, lab, lock)
        root = C3.find_gaps(results, model, lab)
        tp = C3.timed_pages(root)
        rec["nvml (DESCRIPTIVE)"] = smi_trace(session, tp[0][1] if len(tp) == 1 else None)
        pages[f"{model}-{lab}"] = rec
    has_counter = bool(sorted(Path(results).rglob(f"*-{A2.CARD}-{COUNTER_MODEL}-{counter_label(lock)}-r3-counters")))
    if not partial or has_counter:
        pages[f"{COUNTER_MODEL}-{counter_label(lock)} (P5)"] = counter_clause(results, lock)
    ok = all(v["verdict"] == "PASS" for v in pages.values())
    return {"lock_mhz": lock, "verdict": "HOLDS" if ok else "FAILS", "pages": pages,
            "hw_rulers (DESCRIPTIVE)": rulers_clock(results, lock),
            "why": [f"{k}: {w}" for k, v in pages.items() for w in v.get("why") or []]}
