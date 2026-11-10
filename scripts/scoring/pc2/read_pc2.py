#!/usr/bin/env python
"""Rental PC2's reader (docs/registered/2026-11-10-pc2-gh200): what v6d cannot yet price, measured.

    python scripts/scoring/pc2/read_pc2.py --pc2 <published PC2 session dir> [--out-dir DIR]     # pc2.read.{json,txt}
    python scripts/scoring/pc2/read_pc2.py --pc-standin <published PC session dir> [--out-dir DIR]  # standin.read.*

PC2 is a MEASUREMENT rental: every reading is DESCRIPTIVE. It registers no model form and no constant, and moves no
registered verdict. A reading that does not clear its rule reads NOT IDENTIFIED (or NOT READ) with the reason, never a
fallback, never a zero.

THE PAGES are PC's layout (scripts/instr_probe_pc2.py is PC's probe with PC2's rules module): per launch the int64
rows of the instrumented copy at the "ends" level, the expert_ids and num_tokens_post_padded vLLM handed the launch,
the event-pair ms; the perturb unit's plain medians and per-(cell, GEMM) overheads; the regcheck lines; each stamped
compile's PTX loads in flight and CTAs per SM.

WHAT IS PC'S, UNCHANGED (read_pc.py, pinned): one launch (`launch`: PC-R0's per-launch legs and ROW SANITY, the per-CTA
life, rate, concurrency and sharing, the steady-window X), one (cell, GEMM) over its calls (`cell_gemm`: the plateau
median, the per-call spread, C3's OD-1 overhead bound), the F ladder (`f_ladder`), beta_Q (`beta_q`), the board, the
lock read-back, the versions rule, the PTX loads in flight. WHAT IS PC2'S: the units (`STAMP_UNITS`), the naive
geometry and the slab-once rule passed to `launch`, the occupancy leg on every stamped compile, SPAN (the GEMM's own
time and kappa), PLAIN (the plain-kernel event time, its limit stated), and the readings PC2-R0 to R6.

THE FREEZE IS TIED TO THE RENTAL'S COMMIT (read_c2's rule, as PC's): the commit on the ledger's driver start line holds
the registration, the pages, the shapes, the plan and every code_pins path; each code_pins path has its registered
sha256; every repo module this reader loads is pinned. PYTHONHASHSEED=0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(HERE), str(HERE.parent / "pc"), str(HERE.parent / "cal3"), str(HERE.parent / "cal2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pc2common as P  # noqa: E402
import read_pc as RP  # noqa: E402

RC = RP.RC
REG_NAME = P.NAME
REG_PATH = REPO / "docs" / "registered" / f"{REG_NAME}.json"
NI = "NOT IDENTIFIED"
NR = "NOT READ"
PT = "pt"
RCP = "rcp"

#: the plan's stamps units by label: (result directory suffix, variant id, the pc-locks uid, the page, the lock, the
#: perturb unit, the regcheck unit). A frozen literal: the reader never scans for a unit it was not registered with
STAMP_UNITS = {lab: (f"instr-pc-{lab}", P.VARIANTS[lab], f"mixtral-8x7b-{lab}", P.PAGE_OF[lab], P.DECODE_LOCK, PT, RCP)
               for lab in ("sd", "ss", "sn", "sw", "sd3", "sr")}
OTHER_UNITS = {PT: "perturb-pt", RCP: "regcheck-rcp", "census": "tile-census-mixtral-8x7b-pc2"}
PERTURB_UID = ("mixtral-8x7b-pt", P.DECODE_LOCK)
COMMIT_FILES = (f"docs/registered/{REG_NAME}.json", f"docs/registered/{REG_NAME}.txt", P.SHAPES_FILE, P.PLAN,
                *(f"{P.PAGES_DIR}/{n}" for n in P.page_files()))

_fin = RP._fin
_med = RP._med
_r = RP._r


def dumps(d) -> str:
    return json.dumps(d, indent=1, sort_keys=True, default=RP._default) + "\n"


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ================================================================== one unit (PC-R0 with PC2's geometry and occupancy leg)

def cell_cfg(meta: dict, shapes: dict):
    """The MoEConfig of a manifest cell (the guard runs again: a reserved shape on a page is refused here too)."""
    return P.shape_cfg(str(meta.get("shape")), shapes)


def occupancy_leg(man: dict) -> list[str]:
    """PC2-R0's new leg: every stamped compile's CTAs per SM equals its tile's registered occupancy."""
    bad = []
    for s in man.get("sass") or []:
        k = s.get("config") or []
        if len(k) < 6:
            bad.append(f"a stamped compile with no config {k!r}")
            continue
        t = P.tile_key((k[0], k[1], k[2], k[5], k[4]))
        want, got = P.OCCUPANCY.get(t), s.get("ctas_per_sm")
        if want is None or got != want:
            bad.append(f"the stamped {t} compile ({'w2' if k[6] else 'w1'}) holds {got} CTAs per SM, the registered "
                       f"(plain) occupancy is {want}")
    if not man.get("sass") and man.get("complete") is not False and man.get("launches"):
        bad.append("no stamped compile recorded")
    return bad


def stamp_unit(results: Path, session: Path, label: str, R: dict, perturbs: dict, regchecks: dict,
               shapes: dict) -> dict:
    """One PC2 stamps unit: PC-R0's unit legs (read_pc.stamp_unit's, on PC2's units), PC2's occupancy leg, and every
    launch read by read_pc.launch with PC2's geometry (naive live CTAs, the slab-once rule)."""
    suffix, vid, uid, src, lock, pt, rc = STAMP_UNITS[label]
    d, dwhy = RP.unit_dir(results, suffix)
    gate_env = (perturbs.get(pt) or {}).get("dir")
    g = RP.env_word(gate_env / "gate.env" if gate_env else None, "GATE", vid)
    h = RP.env_word(gate_env / "gate.env" if gate_env else None, "HARD", vid)
    rcd = (regchecks.get(rc) or {}).get("dir")
    rcw = RP.env_word(rcd / "regcheck.env" if rcd else None, "REGCHECK", vid)
    u = {"variant": vid, "gate": g or None, "hard": h or None, "regcheck": rcw or None, "lock": lock,
         "page_label": "CLEAN" if g == "PASS" else ("PERTURBED" if g else None)}
    why = []
    if h != "PASS" or not g:
        why.append(f"gate hard legs {h or 'absent'}")
    if rcw != "PASS":
        why.append(f"regcheck {rcw or 'absent'}")
    ok, lw = RP.lock_after(session, uid, lock)
    u["lock_after"] = lw
    if not ok:
        why.append(f"the unit's lock read back after the tool: {lw}")
    ok2, lw2 = RP.lock_after(session, *PERTURB_UID)
    u["perturb_lock_after"] = lw2
    if not ok2:
        why.append(f"its perturb unit's lock read back after the tool: {lw2}")
    if d is None:
        u.update(valid=False, why="; ".join(why + [f"no stamp page ({dwhy})"]), cells={}, launches=0)
        return u
    man = json.loads((d / "stamps.json").read_text())
    if man.get("page_label") != u["page_label"]:
        why.append(f"the page reads {man.get('page_label')}, the gate {u['page_label']}")
    if man.get("source_sha256") != R["pages"][src]["sha256"]:
        why.append("the page is not the registered file")
    if not RP.versions_ok(man.get("versions")):
        why.append(f"versions {man.get('versions')} are not {P.VERSIONS} (python on major.minor)")
    u["partial"] = man.get("complete") is False
    if u["partial"]:
        why.append("PARTIAL page (the unit stopped before its end): its stamped configs' SASS legs are unread, so it "
                   "is refused (the cells stay on disk, DESCRIPTIVE only)")
    elif any(not (s.get("checks_c3") or {}).get("ok") for s in man.get("sass") or [{}]):
        why.append("a stamped config fails C3's SASS legs")
    occ = occupancy_leg(man) if not u["partial"] else []
    u["occupancy_leg"] = occ or "PASS"
    why += occ
    u["inflight"] = RP.inflight_of(man)
    over = RP.overheads(perturbs.get(pt, {}).get("json") or {}, vid)
    cells, timer_refs = {}, {}
    cells_meta = man.get("cells") or {}
    loaded = []
    for e in man.get("launches", []):
        stem = e.get("stem") or e.get("label")
        meta = cells_meta.get(stem) or {}
        if "error" in e:
            cells.setdefault(stem, {"meta": meta, "w1": [], "w2": [], "errors": []})["errors"].append(e["error"])
            continue
        rows = np.load(d / "stamps" / e["file"])
        loaded.append((e, stem, meta, rows, np.load(d / "stamps" / e["experts_file"])))
    off = time_offset(r for _e, _s, _m, r, _x in loaded)
    rebased = [(e, s, m, rebase(r, off), x) for e, s, m, r, x in loaded]
    u["exact_time"] = exact_time_check([r for _e, _s, _m, r, _x in rebased], off)
    if u["exact_time"].startswith("APPLIED") or u["exact_time"].startswith("NOT NEEDED"):
        loaded = rebased
    u["time_offset_ns"] = off
    unit_ref = unit_timer(r for _e, _s, _m, r, _x in loaded)
    u["timer_unit"] = unit_ref
    for e, stem, meta, rows, ex in loaded:
        tile = tuple(int(e[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"))
        try:
            once = P.slab_once(cell_cfg(meta, shapes), meta.get("counts") or [])
        except (P.Refused, ValueError, TypeError) as exc:
            cells.setdefault(stem, {"meta": meta, "w1": [], "w2": [], "errors": []})["errors"].append(str(exc))
            continue
        q_guard = P.q_bytes(tile, P.L2_GUARD_STAGES) if once else None
        o = (over.get((stem, e["gemm"])) or {}).get("median")
        expl = (meta.get("live") or {}).get(e["gemm"])
        tref = timer_refs.get((stem, e.get("call"))) if e["gemm"] == "w2" else None
        if not (tref or {}).get("res"):   # PC2's timer rule: the call's w1 first (PC's), else the unit's resolution
            tref = unit_ref if unit_ref.get("res") else tref
        lr = RP.launch(rows, ex, e["num_tokens_post_padded"], e["N"], e["K"], tile, o=o, expected_live=expl,
                       timer_ref=tref, launch_ms=e.get("launch_ms"), flush=bool(e.get("flush", True)),
                       use_td=bool(e.get("use_td")), late=int(e.get("late_buffers") or 0), q_guard=q_guard,
                       slab_once=once)
        if tref is unit_ref and lr.get("res_from") != "own starts":
            lr["res_from"] = "the unit's resolution (PC2's timer rule)"
        if e["gemm"] == "w1":
            timer_refs[(stem, e.get("call"))] = {"res": lr.get("timer_res_ns"), "f_long": lr.get("f_long")}
        lr.update(call=e.get("call"), gemm=e["gemm"], N=e["N"], K=e["K"], ntpp=e["num_tokens_post_padded"], tile=tile,
                  o=o, slab_once=once)
        c = cells.setdefault(stem, {"meta": meta, "w1": [], "w2": [], "errors": []})
        c[e["gemm"]].append(lr)
    for stem, c in cells.items():   # PC-R0: the first w2 start after the last w1 end, per call
        bad = []
        for call in sorted({x.get("call") for x in c["w1"]} & {x.get("call") for x in c["w2"]}):
            a = [x for x in c["w1"] if x.get("call") == call and "t_last" in x]
            b = [x for x in c["w2"] if x.get("call") == call and "t_first" in x]
            if a and b and not b[0]["t_first"] > a[0]["t_last"]:
                bad.append(call)
        for g_ in ("w1", "w2"):
            for x in c[g_]:
                if x.get("call") in bad:
                    x.update(valid=False, why=(x.get("why") or "") + "; w2 started before w1 ended")
        c["over"] = {g_: over.get((stem, g_)) for g_ in ("w1", "w2")}
    src_count = {}
    for c in cells.values():
        for g_ in ("w1", "w2"):
            for x in c[g_]:
                src_count[x.get("res_from") or "none"] = src_count.get(x.get("res_from") or "none", 0) + 1
    u["timer_sources"] = src_count
    u.update(valid=not why, why="; ".join(why), cells=cells, dir=d.name, launches=len(man.get("launches", [])),
             source=src)
    return u


#: the globaltimer columns of an ends-level stamp row (read_c3.COLUMNS: start_ns, end_ns)
NS_COLS = tuple(RP.RC3.COLUMNS.index(c) for c in ("start_ns", "end_ns"))
#: EXACT TIME: a unit's globaltimer stamps are rebased when they are absolute (past 2^40 ns), to the least positive one
#: less 2^20 ns, so float64 holds every difference exactly
ABS_NS, REBASE_MARGIN_NS = 1 << 40, 1 << 20


def time_offset(rows_iter) -> int:
    """EXACT TIME (PC2's reader rule, registered): read_pc.launch casts globaltimer stamps to float64; an absolute stamp
    (about 1.7e18 ns) has a float64 step of 256 ns, so PC's lives and starts are quantised to 256 ns (the true tick is
    32 ns). PC2 subtracts ONE offset per unit from every globaltimer stamp of a live or dead row before read_pc.launch:
    the least such stamp past 2^40 ns, less 2^20 ns (0 when none is; a garbled small stamp cannot move it). Differences are unchanged and now exact; a
    stamp that was zero or negative stays not positive, so ROW SANITY still refuses it."""
    lo = None
    k = RP.RC3.COLUMNS.index("kind")
    for rows in rows_iter:
        r = np.asarray(rows, np.int64)
        a = r[np.isin(r[:, k], (1, 2))][:, list(NS_COLS)]
        pos = a[a > ABS_NS]
        if pos.size:
            m = int(pos.min())
            lo = m if lo is None else min(lo, m)
    return int(lo - REBASE_MARGIN_NS) if lo is not None and lo > ABS_NS else 0


def exact_time_check(rows_list, off: int) -> str:
    """EXACT TIME's sanity rule (registered): after the rebase every globaltimer stamp of a live or dead row must lie
    under 2^40 ns; then APPLIED (offset > 0) or NOT NEEDED (offset 0, every stamp already under 2^40); else NOT
    APPLIED with the reason, and the unit is read on its stamps as written (PC's float64 reading, 256 ns steps)."""
    k = RP.RC3.COLUMNS.index("kind")
    worst = None
    for rows in rows_list:
        r = np.asarray(rows, np.int64)
        a = r[np.isin(r[:, k], (1, 2))][:, list(NS_COLS)]
        if a.size:
            m = int(a.max())
            worst = m if worst is None else max(worst, m)
    if worst is not None and worst >= ABS_NS:
        return (f"NOT APPLIED: a stamp reads {worst} ns after the rebase (offset {off}), at or past 2^40 ns: the unit's "
                "stamps do not share one absolute base")
    return f"APPLIED (offset {off} ns)" if off else "NOT NEEDED: every stamp under 2^40 ns"


def rebase(rows, off: int):
    """The rows with `off` subtracted from both globaltimer columns of every row stamped live or dead."""
    if not off:
        return rows
    a = np.array(rows, dtype=np.int64, copy=True)
    k = RP.RC3.COLUMNS.index("kind")
    m = np.isin(a[:, k], (1, 2))
    for c in NS_COLS:
        a[m, c] = a[m, c] - off
    return a


def unit_timer(rows_iter) -> dict:
    """PC2's timer rule (registered, a change to PC's): the globaltimer's resolution is the card's, so a launch whose
    live CTAs all start in one tick (no positive difference, so no resolution of its own) and whose call's w1 launch has
    none either takes the UNIT's: the most common own resolution over the unit's launches, read only when at least
    TIMER_UNIT_MIN launches report one and that value is at least half of them; else none (the launch stays NOT READ, as
    PC's A-F7168-G8 and A-TP8-G128 w1 launches did). The clock is never taken across launches (f_long stays the
    launch's own)."""
    vals = []
    for rows in rows_iter:
        c = RP.RC3.decode(rows)
        live = c["kind"] == 1
        r = RP.RC3.timer_resolution(c["start_ns"][live]) if live.sum() >= 2 else None
        if r:
            vals.append(int(r))
    if len(vals) < P.TIMER_UNIT_MIN:
        return {"res": None, "why": f"{len(vals)} launches report a resolution (needs {P.TIMER_UNIT_MIN})"}
    mode = max(set(vals), key=vals.count)
    if vals.count(mode) * 2 < len(vals):
        return {"res": None, "why": f"no resolution holds half of {len(vals)} launches"}
    return {"res": mode, "f_long": None, "launches": len(vals), "agree": vals.count(mode)}


def rd(u: dict, stem: str, gemm: str = "w2") -> dict | None:
    return RP.gemm_reading(u, stem, gemm)


def r_of(g):
    return RP.r_of(g)


def q_of(u: dict, stem: str):
    """Q_B of a cell's tile off the unit's stamped PTX, only when n_if = num_stages - 1."""
    c = (u.get("cells") or {}).get(stem)
    if not c or not (c["w2"] or c["w1"]):
        return None
    t = (c["w2"] or c["w1"])[0]["tile"]
    n_if = (u.get("inflight") or {}).get(P.tile_key(t))
    return P.q_bytes(t, n_if) if n_if == int(t[4]) - 1 else None


# ================================================================== SPAN and PLAIN

def span_of(u: dict, stem: str, gemm: str, census: dict) -> dict:
    """SPAN: t_l = span / (1 + o) on every valid launch, kappa_l = span / median life on single-wave launches; per
    (cell, GEMM) the medians over calls, read with at least SPAN_MIN_LAUNCHES valid launches and the cell GEMM READ.
    A multi-wave GEMM reads t only."""
    c = (u.get("cells") or {}).get(stem)
    if c is None:
        return {"status": f"{NR}: the cell is absent"}
    g = rd(u, stem, gemm) or {}
    if g.get("status") != "READ":
        return {"status": f"{NR}: the cell GEMM is not READ ({g.get('status')})"}
    valid = [x for x in c[gemm] if x.get("valid")]
    if not valid:
        return {"status": f"{NR}: no valid launch"}
    t = P.tile_key(valid[0]["tile"])
    if t not in P.OCCUPANCY:
        return {"status": f"{NI}: {t} is not one of PC2's registered tiles (no registered wave)"}
    if census.get(t) != P.OCCUPANCY.get(t):
        return {"status": f"{NI}: the census reads {census.get(t)} CTAs per SM at {t}, the registered {P.OCCUPANCY.get(t)}"}
    W = P.wave(valid[0]["tile"])
    one = [x for x in valid if x["live"] <= W]
    ts = [x["span_ns"] / (1.0 + float(x["o"])) / 1e3 for x in valid]
    out = {"t_us": _med(ts), "live": valid[0]["live"], "wave": W, "launches": len(valid)}
    if len(valid) < P.SPAN_MIN_LAUNCHES:
        return {"status": f"{NR}: {len(valid)} valid launches, under {P.SPAN_MIN_LAUNCHES}"}
    if not one:
        return {**out, "status": f"READ: t only; {valid[0]['live']} live CTAs, past one wave of {W} (no kappa)"}
    ks = [x["span_ns"] / x["L_ns"] for x in one if x.get("L_ns")]
    return {**out, "kappa": _med(ks), "kappa_per_call": ks, "L_med_us": _med([x["L_ns"] / 1e3 for x in one]),
            "status": "READ"}


def plain_of(perturb: dict, vid: str) -> dict:
    """PLAIN: (stem, GEMM) -> the perturb unit's plain-kernel median in us and its overhead (DESCRIPTIVE)."""
    out = {}
    for (stem, g), v in RP.overheads(perturb, vid).items():
        ms = v.get("plain_ms")
        out[(stem, g)] = {"plain_us": ms * 1e3 if _fin(ms) and ms > 0 else None, "o_median": v.get("median"),
                          "o_worst": v.get("worst")}
    return out


PLAIN_LIMIT = ("an event pair times the whole launch, its fixed cost included; no per-CTA life, so no c0 split and no "
               "kappa; DESCRIPTIVE, it fills no v6d slot alone")


def plain_table(plain: dict, stems) -> dict:
    return {f"{s} {g}": (plain.get((s, g)) or {"plain_us": None, "status": f"{NR}: no perturb row"})
            for s in stems for g in ("w1", "w2")}


def rate_row(u: dict, stem: str, gemm: str, census: dict) -> dict:
    c = (u.get("cells") or {}).get(stem) or {}
    g = rd(u, stem, gemm) or {"status": f"{NR}: absent"}
    sp = span_of(u, stem, gemm, census)
    if not str(sp.get("status", "")).startswith("READ"):
        sp = {**sp, "t_us": None, "kappa": None}
    return {"cell": stem, "tile": (c.get("meta") or {}).get("tile"), "shape": (c.get("meta") or {}).get("shape"),
            "G": g.get("G"), "b": g.get("b"), "r": g.get("r"), "n": g.get("n"), "m": g.get("m"), "X": g.get("X"),
            "lam_us": g["lam_ns"] / 1e3 if _fin(g.get("lam_ns")) else None, "f_mhz": g.get("f_mhz"),
            "status": g.get("status"), "overhead": g.get("overhead"), "page_label": g.get("page_label"),
            "t_us": sp.get("t_us"), "kappa": sp.get("kappa"), "span_status": sp.get("status"),
            "expert_spread": g.get("expert_spread"), "expert_flag": g.get("expert_flag")}


def ladder(u: dict, stems) -> dict:
    """read_pc.f_ladder over a unit's READ w2 rungs: [(stem, b, lambda_med)]."""
    pts = []
    for s in stems:
        g = rd(u, s)
        pts.append((s, g.get("b") if g else None, g.get("lam_ns") if g and g.get("status") == "READ" else None))
    return RP.f_ladder(pts)


def x_sat(units_cells: list, tile: str, census: dict, wave: int, exclude=()) -> dict:
    """X_sat on one tile, PC's definition (PC-R7): the median X over READ LONG-SLAB w2 rungs (w2 K >= X_SAT_W2_MIN_K,
    PC's F 14336) STRICTLY past one wave (G > the tile's registered wave), and READ flushed w1 launches with at least a
    wave of live CTAs and K >= 4096; the cells in `exclude` (a rung read against this X_sat) never enter it; 2 or more
    members; the census must read the registered occupancy. The pool's composition is printed."""
    if census.get(tile) != P.OCCUPANCY.get(tile):
        return {"status": f"{NI}: the census reads {census.get(tile)} CTAs per SM at {tile}, the registered "
                          f"{P.OCCUPANCY.get(tile)}", "pool": []}
    pool = []
    for u, stem in units_cells:
        c = (u.get("cells") or {}).get(stem)
        if not c or (c.get("meta") or {}).get("tile") != tile or stem in exclude:
            continue
        g2 = rd(u, stem, "w2")
        if g2 and g2.get("status") == "READ" and g2["G"] > wave and c["w2"] and \
                int(c["w2"][0]["K"]) >= P.X_SAT_W2_MIN_K and _fin(g2.get("X")):
            pool.append((f"{stem} w2 G {g2['G']} K {c['w2'][0]['K']}", g2["X"]))
        g1 = rd(u, stem, "w1")
        if g1 and g1.get("status") == "READ" and c["w1"] and int(c["w1"][0]["K"]) >= P.X_SAT_W1_MIN_K and \
                (c.get("meta") or {}).get("flush", True) and int(c["w1"][0]["live"]) >= wave and _fin(g1.get("X")):
            pool.append((f"{stem} w1 K {c['w1'][0]['K']} n {c['w1'][0]['live']}", g1["X"]))
    comp = {"w2 long-slab rungs past one wave": sum(1 for lab, _x in pool if " w2 " in lab),
            "w1 launches (K >= 4096, a wave or more)": sum(1 for lab, _x in pool if " w1 " in lab),
            "excluded": sorted(exclude)}
    if len(pool) < 2:
        return {"status": f"{NI}: {len(pool)} members (needs 2)", "pool": pool, "composition": comp}
    return {"X_sat": float(st.median([x for _l, x in pool])), "pool": pool, "composition": comp, "status": "READ"}


# ================================================================== the readings

def r1(D: dict, D3: dict, plain: dict, plain3: dict, census: dict) -> dict:
    t = P.tile_key(P.D16)
    stems = sorted((D.get("cells") or {}), key=lambda s: (len(s), s))
    out = {"tile": t}
    out["R1a rate curve"] = [rate_row(D, s, "w2", census) for s in stems]
    lad8 = ladder(D, [f"D-G8-F{f}" for f in (512, 1024, 1792, 3584, 7168)] + ["D-G8"])
    lad128 = ladder(D, [f"D-G128-F{f}" for f in (512, 1024, 1792, 3584, 7168)] + ["D-G128"])
    out["R1b F ladders"] = {"G 8": lad8, "G 128": lad128}
    q = q_of(D, "D-G128")
    rows = []
    for G in (4, 8, 16, 32, 64, 128, 192, 256, 320, 384):
        s = f"D-G{G}"
        g = rd(D, s)
        row = {"cell": s, "G": G}
        if lad128.get("status") != "READ":
            row["status"] = f"{NI}: c0 (R1b at G 128) is not READ ({lad128.get('status')})"
        elif q is None:
            row["status"] = f"{NI}: Q_B off the PTX is not num_stages - 1 ({(D.get('inflight') or {}).get(t)})"
        elif not g or g.get("status") != "READ":
            row["status"] = f"{NR}: the rung is not READ ({(g or {}).get('status')})"
        else:
            c0 = lad128["c0_us"] * 1e3
            lam, b, n = g["lam_ns"], g["b"], g["n"]
            if lam <= c0:
                row["status"] = f"{NI}: lambda_med {lam:.0f} ns is not past c0 {c0:.0f} ns"
            else:
                row.update(L_us=q * (lam - c0) / b / 1e3, N_eff_bytes=n * q * (lam - c0) / lam, Q_B=q, n=n,
                           lam_us=lam / 1e3, status="READ")
        rows.append(row)
    out["R1c L table (c0 = R1b G 128)"] = rows
    members = [(D, s) for s in stems]
    out["R1d X_sat"] = x_sat(members, t, census, P.wave(P.D16))
    out["R1f s3 factorial (unit sd3), DESCRIPTIVE"] = {
        f"G {G}": RP.beta_q([(f"D3-G{G}", r_of(rd(D3, f"D3-G{G}")), q_of(D3, f"D3-G{G}")),
                             (f"D-G{G}", r_of(rd(D, f"D-G{G}")), q_of(D, f"D-G{G}"))], 2) for G in (16, 128, 384)}
    out["R1f s3 spans"] = {s: span_of(D3, s, "w2", census) for s in sorted(D3.get("cells") or {})}
    out["R1g n_if"] = {"sd": D.get("inflight"), "sd3": D3.get("inflight")}
    out["R1h PLAIN (limit: " + PLAIN_LIMIT + ")"] = {**plain_table(plain, stems),
                                                     **plain_table(plain3, sorted(D3.get("cells") or {}))}
    ga, gs = rd(D, "D-MX-T16-A2"), rd(D, "D-G128")
    a, s = r_of(ga), r_of(gs)
    out["R1i D-MX-T16-A2 / D-G128"] = {"t": a / s - 1 if _fin(a, s) and s > 0 else None,
                                       "expert_spread": (ga or {}).get("expert_spread"),
                                       "expert_flag": (ga or {}).get("expert_flag"),
                                       "status": "READ" if _fin(a, s) else f"{NI}: a twin not READ"}
    return out


def life_split(u: dict, stem: str, lad: dict) -> dict:
    g = rd(u, stem)
    if not g or g.get("status") != "READ" or lad.get("status") != "READ":
        return {"status": f"{NR}: the rung or its ladder is not READ"}
    stream = g["b"] / lad["r_inf_gbps"]
    return {"lam_us": g["lam_ns"] / 1e3, "stream_us": stream / 1e3, "fixed_us": (g["lam_ns"] - stream) / 1e3,
            "b": g["b"], "status": "READ"}


def w1_table(u: dict, census: dict) -> list:
    out = []
    for s in sorted(u.get("cells") or {}):
        g = rd(u, s, "w1")
        if not g:
            continue
        sp = span_of(u, s, "w1", census)
        out.append({"cell": s, "n": g.get("G"), "b": g.get("b"), "lam_us": g["lam_ns"] / 1e3 if _fin(g.get("lam_ns"))
                    else None, "r": g.get("r"), "t_us": sp.get("t_us"), "kappa": sp.get("kappa"),
                    "status": g.get("status"), "span_status": sp.get("status")})
    return out


def r2(S: dict, D: dict, plain: dict, census: dict) -> dict:
    out = {"tile": P.tile_key(P.D64), "PC's c0 printed beside (PC-R2 G 128, board c9f689, cross-board, never pooled)":
           P.V6D_PRINTED["c0_us"]}
    lads, shorts, spans, splits = {}, {}, {}, {}
    for G in (8, 32, 128, 256):
        st_ = [f"S-G{G}-F{f}" for f in (256, 512, 1024, 1792)]
        lads[f"G {G}"] = ladder(S, st_ + [f"S-G{G}"])
        shorts[f"G {G}"] = ladder(S, st_)
        for s in st_ + [f"S-G{G}"]:
            spans[s] = span_of(S, s, "w2", census)
            splits[s] = life_split(S, s, lads[f"G {G}"])
    out["R2a c0 per G (F 256 to 14336)"] = lads
    out["R2b c0 per G (short rungs alone, 66 to 459 KB)"] = shorts
    out["R2c SPAN at short slabs, 64x128x64"] = spans
    out["R2c SPAN at short slabs, 16x64x128"] = {s: span_of(D, s, "w2", census) for s in sorted(D.get("cells") or {})
                                                 if "-F" in s}
    out["R2d life split"] = splits
    out["R2e w1 launches (short K = H slabs)"] = {"S": w1_table(S, census), "D": w1_table(D, census)}
    out["R2f PLAIN (limit: " + PLAIN_LIMIT + ")"] = plain_table(plain, sorted(S.get("cells") or {}))
    return out


def r3(W: dict, plain: dict, census: dict) -> dict:
    out = {}
    stems = sorted(W.get("cells") or {}, key=lambda s: (s.split("-")[0], len(s), s))
    for tile, pre in ((P.W64, "W64-"), (P.W128, "W128-")):
        t = P.tile_key(tile)
        mine = [s for s in stems if s.startswith(pre)]
        rows = []
        for s in mine:
            row = rate_row(W, s, "w2", census)
            q = q_of(W, s)
            row["L_gpu_us"] = (q * row["n"] / row["X"] / 1e3) if row.get("status") == "READ" and q and \
                _fin(row.get("n"), row.get("X")) and row["X"] > 0 else None
            c = (W.get("cells") or {}).get(s) or {}
            v = [x for x in c.get("w2", []) if x.get("valid") and x["live"] <= (P.wave(tile) or 0)]
            if v and row.get("status") == "READ":   # the median over the cell's valid single-wave launches
                row["distinct_sms"] = float(np.median([np.unique(x["sm"]).size for x in v]))
                row["per_sm_max"] = float(np.median([np.bincount(np.unique(x["sm"], return_inverse=True)[1]).max()
                                                     for x in v]))
                row["placement_launches"] = len(v)
            rows.append(row)
        xs = x_sat([(W, s) for s in mine], t, census, P.wave(tile), exclude=(f"{pre}G528",))
        g528 = next((r for r in rows if r["cell"] == f"{pre}G528"), None)
        if xs.get("status") != "READ" or not g528 or g528.get("status") != "READ" or not _fin(g528.get("X")):
            sat = {"status": f"{NI}: X_sat ({xs.get('status')}) or G 528 ({(g528 or {}).get('status')}) unread"}
        else:
            frac = g528["X"] / xs["X_sat"]
            sat = {"X_528_over_X_sat": frac, "status": "SATURATED" if abs(frac - 1) <= P.SAT_WITHIN else
                   ("BELOW" if frac < 1 else "ABOVE")}
        k396 = next((r.get("kappa") for r in rows if r["cell"] == f"{pre}G396"), None)
        kr = (g528["kappa"] / k396) if g528 and _fin(g528.get("kappa"), k396) and k396 > 0 else None
        kap = ({"ratio": kr, "status": "HELD" if abs(kr - 1) <= P.KAPPA_HELD_WITHIN else "MOVED"} if kr is not None
               else {"status": f"{NI}: kappa at G 528 or G 396 unread"})
        out[t] = {"rows": rows, "X_sat": xs, "G 528": sat, "kappa 528 / kappa 396": kap,
                  "v6d term 3": term3(sat, kap)}
    out["unit"] = {k: W.get(k) for k in ("valid", "why", "occupancy_leg")}
    out["R3p PLAIN (limit: " + PLAIN_LIMIT + ")"] = plain_table(plain, stems)
    return out


def term3(sat: dict, kap: dict) -> dict:
    """v6d term 3 on one 4-per-SM tile (registered): v6d's big-single-wave assumption (Little's law at X_sat past G 384,
    kappa held at its G 384 value) HOLDS on PC2's pages when G 528 reads SATURATED and kappa(528) / kappa(396) reads
    HELD (within KAPPA_HELD_WITHIN, 3%); DOES NOT HOLD when both are read and either is not; NOT IDENTIFIED when either
    is unread. DESCRIPTIVE."""
    s, k = sat.get("status", ""), kap.get("status", "")
    if s.startswith(NI) or k.startswith(NI):
        return {"label": NI, "why": [x for x in (s, k) if x.startswith(NI)]}
    if s == "SATURATED" and k == "HELD":
        return {"label": "HOLDS", "why": ["G 528 SATURATED and kappa 528 / 396 HELD"]}
    return {"label": "DOES NOT HOLD", "why": [f"G 528 {s}", f"kappa 528 / 396 {k}"]}


def leader_follower(lrs: list) -> dict:
    """The median per-CTA rate of leaders and followers over a naive cell's valid launches (a follower's expert sits
    on an earlier row of its launch)."""
    lead, foll = [], []
    for x in lrs:
        if not x.get("valid") or x.get("expert") is None:
            continue
        Pn = int(x["P"])
        tiles = np.arange(x["live"]) // Pn
        first = {}
        for tl, e in zip(tiles, x["expert"]):
            first.setdefault(int(e), int(tl))
        f = np.array([tl > first[int(e)] for tl, e in zip(tiles, x["expert"])])
        lead.append(x["r"][~f])
        foll.append(x["r"][f])
    L = np.concatenate(lead) if lead else np.array([])
    F = np.concatenate(foll) if foll else np.array([])
    return {"r_leaders": float(np.median(L)) if L.size else None, "leaders": int(L.size),
            "r_followers": float(np.median(F)) if F.size else None, "followers": int(F.size)}


def r4(N: dict, plain: dict, census: dict) -> dict:
    cells = N.get("cells") or {}
    page = {c["label"]: c for c in P.page_doc("pc2-n")["cells"]}
    out = {"cells": {}, "twins": {}, "pairs": {}}
    for s in sorted(cells):
        meta = page.get(s, {})
        row = {"shape": meta.get("shape"), "T": meta.get("T"), "path": meta.get("path")}
        for g in ("w1", "w2"):
            rg = rd(N, s, g) or {}
            sp = span_of(N, s, g, census)
            row[g] = {"t_us": sp.get("t_us"), "kappa": sp.get("kappa"), "r": rg.get("r"),
                      "L_med_us": rg["lam_ns"] / 1e3 if rg.get("status") == "READ" and _fin(rg.get("lam_ns")) else None,
                      "live": sp.get("live"), "b": rg.get("b"), "status": rg.get("status"),
                      "span_status": sp.get("status"), "plain_us": (plain.get((s, g)) or {}).get("plain_us"),
                      **(leader_follower(cells[s][g]) if meta.get("path") == "naive" and rg.get("status") == "READ"
                         else {})}
        out["cells"][s] = row
    for s, meta in page.items():
        for kind in ("twin", "pair"):
            p = meta.get(kind)
            if not p:
                continue
            res = {}
            for g in ("w1", "w2"):
                a, b = (out["cells"].get(s) or {}).get(g) or {}, (out["cells"].get(p) or {}).get(g) or {}
                ratio = {}
                for k in ("t_us", "plain_us") + (("L_med_us",) if kind == "twin" else ()):
                    ratio[k] = (a[k] / b[k]) if _fin(a.get(k), b.get(k)) and b[k] > 0 else None
                ratio["status"] = "READ" if all(v is not None for v in ratio.values()) else \
                    f"{NR}: a quantity of {s} or {p} is unread"
                res[g] = ratio
            out[f"{kind}s"][f"{s} / {p}"] = res
    out["the seen decode blind's naive cells per model (frozen, beside)"] = P.BLIND_NAIVE_CELLS
    out["unit"] = {k: N.get(k) for k in ("valid", "why", "occupancy_leg")}
    return out


def r5(D: dict, S: dict, Rr: dict, census: dict) -> dict:
    out = {}
    for rs, (u, s) in (("R-D-G8", (D, "D-G8")), ("R-D-G128", (D, "D-G128")), ("R-S-G128-F512", (S, "S-G128-F512"))):
        a, b = r_of(rd(Rr, rs)), r_of(rd(u, s))
        ta, tb = span_of(Rr, rs, "w2", census).get("t_us"), span_of(u, s, "w2", census).get("t_us")
        out[f"{rs} / {s}"] = {"r": a / b - 1 if _fin(a, b) and b > 0 else None,
                              "t": ta / tb - 1 if _fin(ta, tb) and tb > 0 else None}
    return out


def r6(units: dict, census: dict) -> dict:
    r13, r14 = {}, []
    for lab, u in units.items():
        if not u.get("valid"):
            r13[lab] = {"status": f"{NR}: the unit is invalid ({u.get('why')})"}
            continue
        vals = [x["r13"] for c in (u.get("cells") or {}).values() for g_ in ("w1", "w2") for x in c[g_]
                if x.get("valid") and _fin(x.get("r13"))]
        r13[lab] = {"launches": len(vals), "median": _med(vals),
                    "flagged_under_0.97": sum(1 for v in vals if v < P.START_FLAG)}
        for stem, c in sorted((u.get("cells") or {}).items()):
            for x in c["w2"]:
                W = P.wave(x["tile"]) or 0
                if x.get("valid") and x["live"] <= W and x.get("call") == 0:
                    r14.append({"unit": lab, "cell": stem, "G_live": x["live"], "distinct_sms": x["sms"],
                                "per_sm_max": int(np.bincount(np.unique(x["sm"], return_inverse=True)[1]).max()),
                                "start_rank_vs_pid": x["placement_rank"]})
    return {"clock at start (PC-R13's rule)": r13, "placement (PC-R14's rule, single-wave launches)": r14}


# ================================================================== the whole reading

def reg() -> dict:
    return json.loads(REG_PATH.read_text())


def commit_paths(R: dict) -> list:
    return sorted(set(COMMIT_FILES) | set(R["code_pins"]))


def check_repo_modules(R: dict, root: Path = REPO, modules=None) -> list:
    mods = list(sys.modules.values()) if modules is None else list(modules)
    loaded = RC.repo_modules(mods, root, RC.tracked_files(root))
    bad = [(p, "loaded by the reader but not pinned") for p in loaded if p not in R["code_pins"]]
    bad += [(p, "not its registered sha256") for p in loaded
            if p in R["code_pins"] and RC.sha256(Path(root) / p) != R["code_pins"][p]]
    return sorted(set(bad))


def census_occupancy(results: Path) -> tuple[dict, str]:
    d, why = RP.unit_dir(results, OTHER_UNITS["census"])
    f = d / "tile-census.json" if d else None
    if f is None or not f.is_file():
        return {}, why or "no tile-census.json"
    doc = json.loads(f.read_text())
    if doc.get("path_mismatch"):
        return {}, f"census path mismatch {doc['path_mismatch']}"
    out = {}
    for t, x in (doc.get("summary") or {}).items():
        vals = set((x.get("by") or {}).values())
        out[t] = vals.pop() if len(vals) == 1 else None
    return out, ""


def read(pc2: Path, *, registration: dict | None = None, _skip_commit_check_for_tests: bool = False) -> dict:
    R = registration or reg()
    pc2 = Path(pc2)
    results, session = pc2 / "results", pc2 / "session"
    out = {"registration": REG_NAME, "pc2": str(pc2),
           "status": "MEASUREMENT, DESCRIPTIVE: certifies nothing, registers no model form and no constant"}
    if _skip_commit_check_for_tests:
        out["commit"] = {"sha": None, "checked": False}
    else:
        sha, why = RC.run_commit(session)
        if sha is None:
            raise SystemExit(f"REFUSED: the rental's commit is not on its ledger: {why}")
        bad = RC.check_commit(REPO, sha, commit_paths(R))
        if bad:
            raise SystemExit(f"REFUSED: the working tree is not the rental's commit {sha}: {bad[:5]}")
        out["commit"] = {"sha": sha, "checked": True, "files": len(commit_paths(R))}
        bad = RC.check_pins(R, REPO)
        if bad:
            raise SystemExit(f"REFUSED: a pinned file is not its registered bytes: {bad[:5]}")
    brd = RP.board(session)
    out["board"] = {k: brd.get(k) for k in ("valid_decode", "why_decode", "c2_power_env", "lock_probe_1710")}
    census, cwhy = census_occupancy(results)
    out["census"] = {"occupancy": census, "why": cwhy,
                     "registered_check": {t: {"registered": w, "census": census.get(t), "ok": census.get(t) == w}
                                          for t, w in P.OCCUPANCY.items()}}
    pd, pw = RP.unit_dir(results, OTHER_UNITS[PT])
    perturbs = {PT: {"dir": pd, "why": pw, "json": json.loads((pd / "perturb.json").read_text())
                     if pd and (pd / "perturb.json").is_file() else {}}}
    rcd, rcw = RP.unit_dir(results, OTHER_UNITS[RCP])
    regchecks = {RCP: {"dir": rcd, "why": rcw}}
    shapes = P.load_shapes()
    units = {}
    for label in STAMP_UNITS:
        u = stamp_unit(results, session, label, R, perturbs, regchecks, shapes)
        if not brd["valid_decode"]:
            u["valid"], u["why"] = False, "; ".join(x for x in (u["why"], f"board: {brd['why_decode']}") if x)
        units[label] = u
    out["PC2-R0"] = {lab: {k: u.get(k) for k in ("variant", "valid", "why", "page_label", "gate", "hard", "regcheck",
                                                 "lock", "lock_after", "perturb_lock_after", "occupancy_leg", "dir",
                                                 "launches", "inflight", "exact_time", "time_offset_ns", "timer_unit",
                                                 "timer_sources")} for lab, u in units.items()}
    pj = perturbs[PT]["json"]
    plain = {lab: plain_of(pj, STAMP_UNITS[lab][1]) for lab in STAMP_UNITS}
    D, S, N, W, D3, Rr = (units[k] for k in ("sd", "ss", "sn", "sw", "sd3", "sr"))
    out["PC2-R1"] = r1(D, D3, plain["sd"], plain["sd3"], census)
    out["PC2-R2"] = r2(S, D, plain["ss"], census)
    out["PC2-R3"] = r3(W, plain["sw"], census)
    out["PC2-R4"] = r4(N, plain["sn"], census)
    out["PC2-R5"] = r5(D, S, Rr, census)
    out["PC2-R6"] = r6(units, census)
    out["v6d terms"] = ("read off as registered (docs/registered/2026-11-10-pc2-gh200, 'v6d terms'): PC2 fills no slot "
                        "here; the naive-path term is chosen later WITH THE SEEN DECODE BLIND IN VIEW and disclosed")
    bad = check_repo_modules(R) if not _skip_commit_check_for_tests else []
    if bad:
        raise SystemExit(f"REFUSED: the reader loaded a repo module that is not pinned with its registered bytes: {bad[:5]}")
    return out


# ================================================================== the stand-in readout on PC's published pages

def standin(pc: Path) -> dict:
    """PC2's new readings on PC's published pages (board c9f689): SPAN (kappa and t) on PC's block A (64x128x64 w4 s3,
    1710 MHz), the short-K w1 launches of PC's A cells, and PLAIN on PC's refused block D (16x64x128: s2, s3, s4; the
    perturb unit timed every cell although the stamps were refused). A STAND-IN: PC holds no 16x64x128 stamp, no slab
    under 459 KB on w2, no 4-per-SM tile and no naive cell; nothing here is a PC2 reading."""
    import pccommon as PCC
    pc = Path(pc)
    results, session = pc / "results", pc / "session"
    Rpc = json.loads((REPO / "docs" / "registered" / f"{PCC.NAME}.json").read_text())
    perturbs = {}
    for k in ("ptp", "ptc", "pt3"):
        d, w = RP.unit_dir(results, RP.OTHER_UNITS[k])
        perturbs[k] = {"dir": d, "why": w, "json": json.loads((d / "perturb.json").read_text())
                       if d and (d / "perturb.json").is_file() else {}}
    regchecks = {}
    for k in ("rcp", "rc3"):
        d, w = RP.unit_dir(results, RP.OTHER_UNITS[k])
        regchecks[k] = {"dir": d, "why": w}
    census, _cw = RP.census_occupancy(results)
    A = RP.stamp_unit(results, session, "sa", Rpc, perturbs, regchecks, census)
    out = {"what": standin.__doc__.split("\n\n")[0].replace("\n    ", " "), "source": str(pc), "unit sa": {
        k: A.get(k) for k in ("valid", "why", "page_label")}}
    for stem, c in A.get("cells", {}).items():   # PC's cells carry PC's geometry; o for SPAN from the cell's overhead
        for g in ("w1", "w2"):
            for x in c[g]:
                x["o"] = ((c.get("over") or {}).get(g) or {}).get("median")
    cen = dict(census)
    spans = {}
    for stem in sorted(A.get("cells") or {}, key=lambda s: (len(s), s)):
        for g in ("w2", "w1"):
            sp = span_of(A, stem, g, {**cen, "64x128x64w4s3": 3})
            if sp.get("status") == "READ" or g == "w2":
                spans[f"{stem} {g}"] = {k: sp.get(k) for k in ("live", "t_us", "kappa", "status")}
    out["SPAN on PC's block A (kappa: single-wave launches only)"] = spans
    out["w1 launches of PC's block A (K = H)"] = w1_table(A, {**cen, "64x128x64w4s3": 3})
    out["PC2's timer rule on PC's launches that had no resolution (PC: NOT READ)"] = standin_timer(results, A)
    out["EXACT TIME on PC's block A: the median life, PC's float64 stamps against exact ones"] = standin_exact(results, A)
    vid = PCC.VARIANTS["sd"]
    rows = {}
    for c in (perturbs["ptp"]["json"].get("variants") or {}).get(vid, {}).get("cells", []):
        rows[f"{c['label']} {c['gemm']} {c['tile']}"] = {"plain_us": c["plain_ms"] * 1e3 if _fin(c.get("plain_ms")) else None,
                                                        "o_median": c.get("overhead_median")}
    out["PLAIN on PC's refused block D (perturb-ptp, u17-sd; limit: " + PLAIN_LIMIT + ")"] = rows
    pts = []
    for lab in ("D-G4", "D-G16", "D-G64", "D-G128", "D-G256", "D-G384"):
        v = rows.get(f"{lab} w2 16x64x128w4s4") or {}
        G = int(lab.split("G")[1])
        pts.append((G, v.get("plain_us")))
    out["PLAIN D16 w2 at F 14336 against G (single wave, DESCRIPTIVE)"] = pts
    return out


def standin_timer(results: Path, A: dict) -> dict:
    """PC2's timer rule applied to PC's block A: every launch PC read with no resolution of its own and none from its
    call's w1 launch, re-read with the unit's resolution; DESCRIPTIVE (PC's own reading is unchanged)."""
    d, _w = RP.unit_dir(results, RP.STAMP_UNITS["sa"][0])
    man = json.loads((d / "stamps.json").read_text())
    loaded = [(e, np.load(d / "stamps" / e["file"])) for e in man["launches"] if "error" not in e]
    off = time_offset(r for _e, r in loaded)
    loaded = [(e, rebase(r, off)) for e, r in loaded]
    ref = unit_timer(r for _e, r in loaded)
    out = {"unit resolution (exact stamps)": ref, "offset_ns": off}
    for e, rows in loaded:
        stem = e.get("stem") or e.get("label")
        lr_pc = next((x for x in (A["cells"].get(stem) or {}).get(e["gemm"], []) if x.get("call") == e.get("call")), None)
        if not lr_pc or lr_pc.get("valid") or "globaltimer resolution None" not in (lr_pc.get("why") or ""):
            continue
        tile = tuple(int(e[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"))
        lr = RP.launch(rows, np.load(d / "stamps" / e["experts_file"]), e["num_tokens_post_padded"], e["N"], e["K"],
                       tile, o=lr_pc_o(A, stem, e["gemm"]), expected_live=lr_pc.get("live"), timer_ref=ref,
                       launch_ms=e.get("launch_ms"), q_guard=P.q_bytes(tile, P.L2_GUARD_STAGES))
        out[f"{stem} {e['gemm']} call {e.get('call')}"] = {"valid": lr.get("valid"), "why": lr.get("why"),
                                                           "path": lr.get("path"), "L_us": _r((lr.get("L_ns") or 0) / 1e3)}
    return out


def standin_exact(results: Path, A: dict) -> dict:
    """Per (cell, GEMM) of PC's block A, call 0: the median CTA life read by read_pc.launch on PC's absolute stamps and
    on the same stamps rebased (EXACT TIME), and their ratio - 1; DESCRIPTIVE (PC's reading is unchanged)."""
    d, _w = RP.unit_dir(results, RP.STAMP_UNITS["sa"][0])
    man = json.loads((d / "stamps.json").read_text())
    loaded = [(e, np.load(d / "stamps" / e["file"])) for e in man["launches"] if "error" not in e and e.get("call") == 0]
    off = time_offset(r for _e, r in loaded)
    out = {}
    for e, rows in loaded:
        stem = e.get("stem") or e.get("label")
        tile = tuple(int(e[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"))
        ex = np.load(d / "stamps" / e["experts_file"])
        kw = dict(o=0.0, timer_ref={"res": 32, "f_long": None})
        a = RP.launch(rows, ex, e["num_tokens_post_padded"], e["N"], e["K"], tile, **kw)
        b = RP.launch(rebase(rows, off), ex, e["num_tokens_post_padded"], e["N"], e["K"], tile, **kw)
        if _fin(a.get("L_ns"), b.get("L_ns")) and b["L_ns"] > 0:
            out[f"{stem} {e['gemm']}"] = {"live": b.get("live"), "L_pc_us": _r(a["L_ns"] / 1e3),
                                          "L_exact_us": _r(b["L_ns"] / 1e3), "ratio_minus_1": _r(a["L_ns"] / b["L_ns"] - 1, 5)}
    v = [abs(x["ratio_minus_1"]) for x in out.values() if x["ratio_minus_1"] is not None]
    return {"launches": len(out), "max_abs_ratio_minus_1": max(v) if v else None,
            "median_abs_ratio_minus_1": _med(v), "rows": out}


def lr_pc_o(A: dict, stem: str, gemm: str):
    return (((A["cells"].get(stem) or {}).get("over") or {}).get(gemm) or {}).get("median")


# ================================================================== text

def txt(d: dict) -> str:
    L = [f"PC2 READING ({d['registration']}): {d['status']}", f"  pages: {d['pc2']}",
         f"  commit: {json.dumps(d['commit'], sort_keys=True)}", f"  board: {json.dumps(d['board'], sort_keys=True, default=str)}",
         f"  census: {json.dumps(d['census'], sort_keys=True, default=str)}", ""]
    for k in ("PC2-R0", "PC2-R1", "PC2-R2", "PC2-R3", "PC2-R4", "PC2-R5", "PC2-R6", "v6d terms"):
        L.append(f"{k}:")
        t = json.dumps(d[k], sort_keys=True, default=RP._default)
        L.append("  " + (t if len(t) <= 20000 else t[:20000] + f" ... [TRUNCATED: {len(t) - 20000} more characters; "
                                                                "the whole reading is in pc2.read.json]"))
    return "\n".join(L) + "\n"


def standin_txt(d: dict) -> str:
    L = [f"STAND-IN READOUT on PC's published pages ({d['source']}): {d['what']}"]
    for k, v in d.items():
        if k in ("what", "source"):
            continue
        L.append(f"{k}:")
        if isinstance(v, dict) and len(v) > 3:
            for kk, vv in v.items():
                L.append(f"  {kk}: {json.dumps(vv, sort_keys=True, default=RP._default)}")
        else:
            L.append(f"  {json.dumps(v, sort_keys=True, default=RP._default)}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pc2", type=Path, help="the published PC2 session directory (results/, session/)")
    g.add_argument("--pc-standin", type=Path, help="the published PC session directory: the stand-in readout")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.pc_standin:
        d = standin(a.pc_standin)
        atomic_write(a.out_dir / "standin.read.json", dumps(d))
        atomic_write(a.out_dir / "standin.read.txt", standin_txt(d))
        print(standin_txt(d))
        return 0
    d = read(a.pc2)
    atomic_write(a.out_dir / "pc2.read.json", dumps(d))
    atomic_write(a.out_dir / "pc2.read.txt", txt(d))
    print(txt(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
