#!/usr/bin/env python
"""Rental RR's reader (docs/registered/2026-11-11-rr-gh200): the w1 re-read L2 hit mechanism, measured.

    python scripts/scoring/rr/read_rr.py --rr <published RR session dir> [--out-dir DIR]          # rr.read.{json,txt}
    python scripts/scoring/rr/read_rr.py --standin results/published [--out-dir DIR]      # standin.read.{json,txt}

RR is a MEASUREMENT rental: every reading is DESCRIPTIVE. No model form is registered, there is no gate and there is no
blind; nothing here prices a later page. A reading that does not clear its rule reads NOT READ or NOT IDENTIFIED with the
reason, never a fallback, never a zero. STATED PLAINLY: the 42.5 MiB constant and the four forms were FOUND POST-HOC ON
SEEN PAGES (design-P1 section 2.3: the Phi split and PC-R10's pages); RR measures them on fresh synthetic cells.

THE PAGES. The stamps units (scripts/instr_probe_rr.py, PC2's probe with RR's rules: PC's on-disk layout) give, per w1
launch, the int64 rows of the instrumented copy at the "ends" level and the expert_ids vLLM handed it; the counter units
(scripts/dram_counter_route_rr.py) give, per cell and GEMM, ncu's DRAM bytes per call on the same cells and board; the
perturb unit gives each cell GEMM's overhead (the uninstrumented reference); the regcheck lines and the census give the
occupancy.

WHAT IS PC'S AND PC2'S, UNCHANGED: one launch (read_pc.launch: PC-R0's per-launch legs and ROW SANITY, with RR's geometry:
no slab is read once, so neither the X ceiling nor the L2 guard applies), the per-unit legs (gate, regcheck, the page's
sha256, the versions, C3's SASS legs, the PARTIAL refusal, the lock read back after the tool), EXACT TIME and the unit's
timer (read_pc2.time_offset, rebase, unit_timer), the step integral (read_pc.step_integral), C3-R2's pairs. WHAT IS RR'S:
the occupancy leg on RR's tiles, the per-pair distances and the four forms (RR-R2), the counted miss fraction (RR-R1),
the wave-start spread (RR-R3), the replicate (RR-R4), the stand-in readout.

THE FREEZE IS TIED TO THE RENTAL'S COMMIT (read_c2's rule, as PC's and PC2's): the commit on the ledger's driver start line
holds the registration, the pages, the shapes, the plan and every code_pins path; each code_pins path has its registered
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
for _p in (str(HERE), str(HERE.parent / "pc2"), str(HERE.parent / "pc"), str(HERE.parent / "cal3"),
           str(HERE.parent / "cal2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import read_pc as RP  # noqa: E402
import read_pc2 as R2  # noqa: E402
import rrcommon as P  # noqa: E402

RC = RP.RC
RC3 = RP.RC3
REG_NAME = P.NAME
REG_PATH = REPO / "docs" / "registered" / f"{REG_NAME}.json"
NI = "NOT IDENTIFIED"
NR = "NOT READ"
PT, RCP = "pt", "rcp"
LOCK = P.SESSION_LOCK

#: the plan's stamps units by label: (result directory suffix, variant id, the pc-locks uid, the page, the perturb unit,
#: the regcheck unit, its counter unit). A frozen literal: the reader never scans for a unit it was not registered with
STAMP_UNITS = {"s128": ("instr-pc-s128", P.VARIANTS["s128"], "mixtral-8x7b-s128", P.PAGE_OF["s128"], PT, RCP, "c128"),
               "s64": ("instr-pc-s64", P.VARIANTS["s64"], "mixtral-8x7b-s64", P.PAGE_OF["s64"], PT, RCP, "c64"),
               "sr": ("instr-phi-3.5-moe-sr", P.VARIANTS["sr"], "phi-3.5-moe-sr", P.PHI_128, PT, RCP, "br")}
#: the counter units: (result directory suffix, the pc-locks uid, the page)
COUNTER_UNITS = {"c128": ("mixtral-8x7b-c128-r3-counters", "mixtral-8x7b-c128", P.COUNTER_OF["c128"]),
                 "c64": ("mixtral-8x7b-c64-r3-counters", "mixtral-8x7b-c64", P.COUNTER_OF["c64"])}
REP_BYTES = "phi-3.5-moe-br-r3-counters"
OTHER_UNITS = {PT: "perturb-pt", RCP: "regcheck-rcp", "census": "tile-census-mixtral-8x7b-rr"}
PERTURB_UID = ("mixtral-8x7b-pt", LOCK)
COMMIT_FILES = (f"docs/registered/{REG_NAME}.json", f"docs/registered/{REG_NAME}.txt", P.SHAPES_FILE, P.PLAN,
                *(f"{P.PAGES_DIR}/{n}" for n in P.page_files()), P.PHI_128)
COUNTER_FILE = f"lock{LOCK}/r3d.json"
#: RR-R2's clock note (registered): the stamps and the counted bytes share the 1410 MHz lock, not the SM clock
CLOCK_NOTE = ("the stamps run at the 1410 MHz session lock; the counted run's SM clock under ncu (kernel replay, the "
              "same lock) differs from it and is printed per cell in RR-R1 (B3's board read 1304 to 1350 MHz there): the "
              "two pages share the lock, not the clock, and the gaps (in bytes) are read off the stamps only")

_fin = RP._fin
_med = RP._med
_r = RP._r


def dumps(d) -> str:
    return json.dumps(d, indent=1, sort_keys=True, default=RP._default) + "\n"


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _rms(v):
    v = [float(x) for x in v if _fin(x)]
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else None


def _mean(v):
    v = [float(x) for x in v if _fin(x)]
    return sum(v) / len(v) if v else None


# ================================================================== per-launch geometry (pure)

def exact_se(rows):
    """(start_ns, end_ns, life) of a launch's live CTAs (kind 1, pid order) as float64 after subtracting the launch's own
    least positive globaltimer stamp in int64 (EXACT TIME's idea per launch: every difference exact; PC2's per-unit
    rebase reads the same differences). None when a live stamp is not positive or a life is not positive."""
    c = RC3.decode(rows)
    live = c["kind"] == 1
    s = c["start_ns"][live]
    e = c["end_ns"][live]
    if not s.size or (s <= 0).any() or (e <= 0).any():
        return None
    base = int(min(int(s.min()), int(e.min())))
    s = (s - base).astype(np.float64)
    e = (e - base).astype(np.float64)
    if ((e - s) <= 0).any():
        return None
    return s, e, e - s


def form_miss(form: str, d_start, d_mid):
    """Per pair, the registered form's miss indicator (step) or probability (exp)."""
    dist, law, C, _why = P.FORMS[form]
    d = np.asarray(d_start if dist == "start" else d_mid, np.float64)
    return law_miss(law, d, C)


def law_miss(law: str, d, C_mib: float):
    d = np.asarray(d, np.float64)
    if law == "step":
        return (d > C_mib * P.MIB).astype(np.float64)
    return 1.0 - np.exp(-np.maximum(d, 0.0) / (C_mib * P.MIB))


def wave_spreads(s, L: float, W: int) -> list[float]:
    """The wave-start spread of every FULL schedule wave k (pids k W to (k + 1) W - 1, all live): (q90 - q10 of the
    wave's starts) / L, L the launch's median life (design-P1 section 2.4)."""
    s = np.asarray(s, np.float64)
    out = []
    for k in range(len(s) // int(W)):
        seg = s[k * W:(k + 1) * W]
        q = np.quantile(seg, P.WAVE_Q)
        out.append(float((q[1] - q[0]) / L))
    return out


def pair_read(rows, experts, ntpp: int, N: int, K: int, tile) -> dict:
    """RR-R2 on one w1 launch: C3-R2's re-read pairs (rank r = pid_m P + pid_n whose M-tile pid_m - 1 holds the same
    expert), and per pair d_start, the B bytes streamed GPU-wide between the partners' starts (each CTA's slab K x its
    BLOCK_N columns x 2 spread evenly over its life: PC-R10's integral), and d_mid, the same between their mid-lives;
    each registered form's m_hat (the mean over pairs of its miss indicator or probability); the gap g in units of P L
    / W_c and the together share (g < 0.5, C3's rule); RR-R3's wave-start spread per full wave; the life spread past the
    first wave; the re-reading CTAs' lives split by the primary form's hit and miss. Pure."""
    se = exact_se(rows)
    if se is None:
        return {"status": f"{NR}: a live stamp or life is not positive"}
    s, e, life = se
    bm, bn = int(tile[0]), int(tile[1])
    Pn = -(-int(N) // bn)
    lt = int(ntpp) // bm
    live = lt * Pn
    W = P.wave(tile)
    if len(s) != live or W is None:
        return {"status": f"{NR}: {len(s)} live rows, the launch's geometry gives {live} (or no registered wave)"}
    L = float(np.median(life))
    r = RC3.pairs(np.asarray(experts).reshape(-1)[:lt], Pn)
    pid = np.arange(live)
    b = int(K) * np.minimum(bn, int(N) - (pid % Pn) * bn) * 2.0
    out = {"live": live, "P": Pn, "W_c": W, "L_us": L / 1e3, "pairs": int(r.size),
           "wave_spread": wave_spreads(s, L, W)}
    past = life[W:]
    if past.size >= 16:
        q = np.quantile(past, P.LIFE_Q)
        out["life_spread"] = float((q[1] - q[0]) / 2 / np.median(life))
    if not r.size:
        out["status"] = f"{NR}: no re-read pair in the launch"
        return out
    t, val, cum = RP.step_integral(s, e, b / life)

    def F(x):
        return RP.integral_at(t, val, cum, x)
    d_start = F(s[r]) - F(s[r - Pn])
    mid = (s + e) / 2
    d_mid = F(mid[r]) - F(mid[r - Pn])
    g = (s[r] - s[r - Pn]) / (Pn * L / W)
    out.update(_d_start=d_start, _d_mid=d_mid, together_share=float(np.mean(g < RC3.GAP_TOGETHER)),
               gap_median=float(np.median(g)),
               d_start_mib_q10_50_90=[float(x) / P.MIB for x in np.quantile(d_start, (0.1, 0.5, 0.9))],
               d_mid_mib_q10_50_90=[float(x) / P.MIB for x in np.quantile(d_mid, (0.1, 0.5, 0.9))],
               m_hat={f: float(np.mean(form_miss(f, d_start, d_mid))) for f in P.FORMS})
    miss = form_miss(P.PRIMARY_FORM, d_start, d_mid) > 0.5
    if miss.any() and (~miss).any():
        lh, lm = float(np.median(life[r][~miss])), float(np.median(life[r][miss]))
        out["reread_life_us"] = {"hit": lh / 1e3, "miss": lm / 1e3, "miss_over_hit": lm / lh}
    out["status"] = "READ"
    return out


# ================================================================== a stamps unit

def occupancy_leg(man: dict) -> list[str]:
    """RR-R0's occupancy leg (PC2's, on RR's tiles): every stamped compile's CTAs per SM equals its tile's registered
    occupancy."""
    bad = []
    for s in man.get("sass") or []:
        k = s.get("config") or []
        if len(k) < 7:
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


def hist_counts(rel: str) -> dict:
    """(label, n) -> counts of a registered histogram page (a 'balanced' cell carries none: n x BLOCK_M rows per expert,
    R3's balanced arm)."""
    from moe.spec import MODEL_CONFIGS
    doc = json.loads((REPO / rel).read_text())
    E, bm = MODEL_CONFIGS[doc["model"]].num_experts, int(doc["block_m"])
    return {(c["label"], int(c["n"])): ([int(x) for x in c["counts"]] if c.get("counts") else [int(c["n"]) * bm] * E)
            for c in doc["cells"]}


_STEM = __import__("re").compile(r"^(.+)-n(\d+)-([a-z]+)$")


def stem_key(stem: str) -> tuple:
    """A histogram cell's stem 'label-n<n>-<arm>' (C3's stem) as (label, n); (stem, None) when it is not one."""
    m = _STEM.match(str(stem))
    return (m.group(1), int(m.group(2))) if m else (stem, None)


def stamp_unit(results: Path, session: Path, label: str, R: dict, perturbs: dict, regchecks: dict,
               shapes: dict) -> dict:
    """One RR stamps unit: PC-R0's unit legs (PC2's), RR's occupancy leg, every launch read by read_pc.launch with RR's
    geometry (EXACT TIME and the unit's timer as PC2), and every valid w1 launch's pairs (RR-R2, RR-R3)."""
    suffix, vid, uid, src, pt, rc, _cu = STAMP_UNITS[label]
    d, dwhy = RP.unit_dir(results, suffix)
    gate_env = (perturbs.get(pt) or {}).get("dir")
    g = RP.env_word(gate_env / "gate.env" if gate_env else None, "GATE", vid)
    h = RP.env_word(gate_env / "gate.env" if gate_env else None, "HARD", vid)
    rcd = (regchecks.get(rc) or {}).get("dir")
    rcw = RP.env_word(rcd / "regcheck.env" if rcd else None, "REGCHECK", vid)
    u = {"variant": vid, "gate": g or None, "hard": h or None, "regcheck": rcw or None, "lock": LOCK,
         "page_label": "CLEAN" if g == "PASS" else ("PERTURBED" if g else None)}
    why = []
    if h != "PASS" or not g:
        why.append(f"gate hard legs {h or 'absent'}")
    if rcw != "PASS":
        why.append(f"regcheck {rcw or 'absent'}")
    ok, lw = RP.lock_after(session, uid, LOCK)
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
    over = RP.overheads(perturbs.get(pt, {}).get("json") or {}, vid)
    hist = hist_counts(src) if label == "sr" else {}
    cells, timer_refs, loaded = {}, {}, []
    cells_meta = man.get("cells") or {}
    for e in man.get("launches", []):
        stem = e.get("stem") or e.get("label")
        meta = cells_meta.get(stem) or {}
        if "error" in e:
            cells.setdefault(stem, {"meta": meta, "w1": [], "w2": [], "errors": []})["errors"].append(e["error"])
            continue
        rows = np.load(d / "stamps" / e["file"])
        loaded.append((e, stem, meta, rows, np.load(d / "stamps" / e["experts_file"])))
    off = R2.time_offset(r for _e, _s, _m, r, _x in loaded)
    rebased = [(e, s, m, R2.rebase(r, off), x) for e, s, m, r, x in loaded]
    u["exact_time"] = R2.exact_time_check([r for _e, _s, _m, r, _x in rebased], off)
    if u["exact_time"].startswith("APPLIED") or u["exact_time"].startswith("NOT NEEDED"):
        loaded = rebased
    u["time_offset_ns"] = off
    unit_ref = R2.unit_timer(r for _e, _s, _m, r, _x in loaded)
    u["timer_unit"] = unit_ref
    for e, stem, meta, rows, ex in loaded:
        tile = tuple(int(e[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"))
        o = (over.get((stem, e["gemm"])) or {}).get("median")
        expl = (meta.get("live") or {}).get(e["gemm"])
        if label == "sr":
            from moe.spec import MODEL_CONFIGS
            counts = hist.get(stem_key(stem))
            expl = P.live_ctas(MODEL_CONFIGS["phi-3.5-moe"], counts, tile, e["gemm"]) if counts else None
            meta = {**meta, "counts": counts, "shape": "phi-3.5-moe"}
        tref = timer_refs.get((stem, e.get("call"))) if e["gemm"] == "w2" else None
        if not (tref or {}).get("res"):
            tref = unit_ref if unit_ref.get("res") else tref
        lr = RP.launch(rows, ex, e["num_tokens_post_padded"], e["N"], e["K"], tile, o=o, expected_live=expl,
                       timer_ref=tref, launch_ms=e.get("launch_ms"), flush=bool(e.get("flush", True)),
                       use_td=bool(e.get("use_td")), late=int(e.get("late_buffers") or 0), q_guard=None,
                       slab_once=False)
        if e["gemm"] == "w1":
            timer_refs[(stem, e.get("call"))] = {"res": lr.get("timer_res_ns"), "f_long": lr.get("f_long")}
            if lr.get("valid"):
                lr["pairs_read"] = pair_read(rows, ex, e["num_tokens_post_padded"], e["N"], e["K"], tile)
        lr.update(call=e.get("call"), gemm=e["gemm"], N=e["N"], K=e["K"], ntpp=e["num_tokens_post_padded"], tile=tile,
                  o=o)
        c = cells.setdefault(stem, {"meta": meta, "w1": [], "w2": [], "errors": []})
        c["meta"] = meta
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
    u.update(valid=not why, why="; ".join(why), cells=cells, dir=d.name, launches=len(man.get("launches", [])),
             source=src)
    return u


def cell_stamps(u: dict, stem: str) -> dict:
    """A cell's stamped w1 reading over its calls: READ with at least MIN_VALID_CALLS valid w1 launches whose pairs read,
    the cell's w1 overhead within C3's OD-1 bound, and the unit valid; m_hat per form the median over calls (the
    per-call values and their spread printed), the wave spreads per full wave the median over calls."""
    c = (u.get("cells") or {}).get(stem)
    if c is None:
        return {"status": f"{NR}: the cell is absent"}
    if not u.get("valid"):
        return {"status": f"{NR}: the unit is invalid ({u.get('why')})"}
    ov = (c.get("over") or {}).get("w1") or {}
    om, ow = ov.get("median"), ov.get("worst")
    if not _fin(om, ow):
        return {"status": f"{NR}: the w1 overhead is unread"}
    if abs(om) > P.OVERHEAD_MEDIAN_MAX or abs(ow) > P.OVERHEAD_WORST_MAX:
        return {"status": f"{NR}: w1 overhead median {100 * om:.2f}%, worst {100 * ow:.2f}% over C3's NOT-READ bound"}
    reads = [x["pairs_read"] for x in c["w1"] if x.get("valid") and (x.get("pairs_read") or {}).get("status") == "READ"]
    bad = sorted({x.get("why") or (x.get("pairs_read") or {}).get("status") for x in c["w1"]
                  if not (x.get("valid") and (x.get("pairs_read") or {}).get("status") == "READ")})
    if len(reads) < P.MIN_VALID_CALLS:
        return {"status": f"{NR}: {len(reads)} readable w1 launches (needs {P.MIN_VALID_CALLS}): {bad[:2]}"}
    out = {"calls": len(reads), "overhead": {"median": om, "worst": ow}, "pairs": reads[0]["pairs"],
           "live": reads[0]["live"], "P": reads[0]["P"], "W_c": reads[0]["W_c"], "L_us": _med([x["L_us"] for x in reads]),
           "together_share": _med([x["together_share"] for x in reads]),
           "gap_median": _med([x["gap_median"] for x in reads]),
           "d_start_mib_q10_50_90": [_med([x["d_start_mib_q10_50_90"][i] for x in reads]) for i in range(3)],
           "d_mid_mib_q10_50_90": [_med([x["d_mid_mib_q10_50_90"][i] for x in reads]) for i in range(3)],
           "m_hat": {f: _med([x["m_hat"][f] for x in reads]) for f in P.FORMS},
           "m_hat_per_call": {f: [x["m_hat"][f] for x in reads] for f in P.FORMS},
           "life_spread": _med([x.get("life_spread") for x in reads]),
           "_d": [(x["_d_start"], x["_d_mid"]) for x in reads], "status": "READ"}
    K = min(len(x["wave_spread"]) for x in reads)
    out["wave_spread"] = [_med([x["wave_spread"][k] for x in reads]) for k in range(K)]
    lv = [x["reread_life_us"] for x in reads if x.get("reread_life_us")]
    if lv:
        out["reread_life_us"] = {k: _med([x[k] for x in lv]) for k in ("hit", "miss", "miss_over_hit")}
    return out


# ================================================================== a counter unit (RR-R1)

def counter_cell(row: dict, cfg, tile) -> dict:
    """RR-R1 on one counted cell: D = the median over its calls of the w1 DRAM reads (READ with every call counted and
    their spread at most COUNT_SPREAD_MAX), C_u and R from the cell's geometry, m = (D - C_u) / R; counter sanity: D within
    [COUNT_LO C_u, COUNT_HI (C_u + R)], else NOT READ. Printed beside: the w2 reads against w2's compulsory bytes (each
    active expert's w2 once plus the intermediate rows once), ncu's L2 read hit rate on w1, the ncu clock."""
    geo = P.geometry(cfg, row["counts"], tile)
    pc = ((row.get("per_call") or {}).get("w1") or {}).get("dram__bytes_read.sum") or []
    out = {"C_u_bytes": geo["C_u_bytes"], "R_bytes": geo["R_bytes"], "D_per_call": pc,
           "clock_mhz": _med((row.get("clock_mhz") or {}).get("w1") or []),
           "l2_read_hit_pct_w1": ((row.get("per_gemm") or {}).get("w1") or {}).get("lts__t_sector_op_read_hit_rate.pct")}
    if len(pc) != P.COUNT_CALLS or not all(_fin(x) for x in pc):
        out["status"] = f"{NR}: {len(pc)} counted w1 calls, not {P.COUNT_CALLS} finite ones"
        return out
    D = float(st.median(pc))
    spread = (max(pc) - min(pc)) / D if D > 0 else None
    out.update(D_bytes=D, spread=spread)
    if not (_fin(spread) and spread <= P.COUNT_SPREAD_MAX):
        out["status"] = f"{NR}: the calls' w1 DRAM reads spread {spread}, over {P.COUNT_SPREAD_MAX}"
        return out
    lo, hi = P.COUNT_LO * geo["C_u_bytes"], P.COUNT_HI * (geo["C_u_bytes"] + geo["R_bytes"])
    if not lo <= D <= hi:
        out["status"] = (f"{NR}: counter sanity: D {D:.0f} B outside [{lo:.0f}, {hi:.0f}] (C_u x {P.COUNT_LO}, "
                         f"(C_u + R) x {P.COUNT_HI})")
        return out
    out["m"] = P.counted_m(D, geo["C_u_bytes"], geo["R_bytes"])
    out["D_over_C_u_minus_1"] = D / geo["C_u_bytes"] - 1
    d2 = ((row.get("per_gemm") or {}).get("w2") or {}).get("dram__bytes_read.sum")
    N2, K2 = P.gemm_dims(cfg, "w2")
    c2 = geo["A"] * N2 * K2 * 2 + sum(int(x) for x in row["counts"]) * K2 * 2
    out["w2_D_over_compulsory_minus_1"] = (d2 / c2 - 1) if _fin(d2) and c2 > 0 else None
    out["status"] = "READ"
    return out


def counter_unit(results: Path, session: Path, label: str, R: dict, shapes: dict) -> dict:
    """A counter unit's legs (RR-R0): its page present and complete, the registered page (sha256), the versions, every
    launch attributed (the route's V_ALIGN_grid PASS), its lock read back after the tool; then RR-R1 per cell."""
    suffix, uid, src = COUNTER_UNITS[label]
    d, dwhy = RP.unit_dir(results, suffix)
    u = {"lock": LOCK}
    why = []
    ok, lw = RP.lock_after(session, uid, LOCK)
    u["lock_after"] = lw
    if not ok:
        why.append(f"the unit's lock read back after the tool: {lw}")
    f = d / COUNTER_FILE if d else None
    if f is None or not f.is_file():
        u.update(valid=False, why="; ".join(why + [f"no counter page ({dwhy or 'no ' + COUNTER_FILE})"]), cells={})
        return u
    page = json.loads(f.read_text())
    if page.get("schema") != P.COUNTER_SCHEMA:
        why.append(f"the page's schema is {page.get('schema')!r}")
    de = page.get("design") or {}
    if de.get("page") != src or de.get("page_sha256") != R["pages"][src]["sha256"]:
        why.append("the counted page is not the registered file")
    if not RP.versions_ok(page.get("versions")):
        why.append(f"versions {page.get('versions')} are not {P.VERSIONS} (python on major.minor)")
    gates = page.get("gates") or {}
    if not str(gates.get("V_ALIGN_grid", "")).startswith("PASS"):
        why.append(f"attribution {gates.get('V_ALIGN_grid')!r}")
    if gates.get("complete") is not True:
        why.append("the counted page is not complete")
    if de.get("lock_mhz") != LOCK:
        why.append(f"the page was counted at {de.get('lock_mhz')} MHz, not the session's {LOCK}")
    cells = {}
    for row in page.get("cells") or []:
        try:
            cfg = P.shape_cfg(str(row.get("shape")), shapes)
            tile = P.TILES[row["tile"]]
            cells[row["label"]] = counter_cell(row, cfg, tile)
        except (P.Refused, KeyError, TypeError, ValueError) as exc:
            cells[str(row.get("label"))] = {"status": f"{NR}: {exc}"}
    if why:   # an invalid unit reads no number: every cell keeps only its reason
        cells = {lab: {"status": f"{NR}: the unit is invalid ({'; '.join(why)})"} for lab in cells}
    u.update(valid=not why, why="; ".join(why), cells=cells, dir=d.name, seconds=page.get("seconds"))
    return u


# ================================================================== the readings

def page_cells(page: str) -> dict:
    return {c["label"]: c for c in P.check_page_doc(json.loads((REPO / page).read_text()), P.load_shapes())}


def join(stamps: dict, counts: dict, page: str) -> list[dict]:
    """RR-R2's rows: every page cell read on both units (same label, same board), its form residuals m_hat - m and the w1
    byte residual (m_hat - m) R / D; a cell missing on either side is listed NOT READ with the reason."""
    rows = []
    for lab, c in page_cells(page).items():
        s, k = cell_stamps(stamps, lab), (counts.get("cells") or {}).get(lab) or {"status": f"{NR}: not counted"}
        g = P.geometry(c["cfg"], c["counts"], c["tile"])
        row = {"cell": lab, "series": c["series"], "P": c["P"], "layout": c["layout"], "role": c["role"],
               "page_label": stamps.get("page_label"),
               "tile": P.tile_key(c["tile"]), "W_c_over_P": g["W_c_over_P"], "F_stag_MiB": g["F_stag_MiB"],
               "m_e_max": g["m_e_max"], "waves": g["waves_w1"], "stamps": s.get("status"), "counted": k.get("status")}
        if s.get("status") == "READ" and k.get("status") == "READ":
            row.update(m=k["m"], D=k["D_bytes"], R=k["R_bytes"], together_share=s["together_share"],
                       d_start_mib_q10_50_90=s["d_start_mib_q10_50_90"],
                       m_hat=s["m_hat"], resid={f: s["m_hat"][f] - k["m"] for f in P.FORMS},
                       w1_bytes_resid={f: (s["m_hat"][f] - k["m"]) * k["R_bytes"] / k["D_bytes"] for f in P.FORMS},
                       _d=s["_d"], status="READ")
        else:
            row["status"] = f"{NR}: stamps {s.get('status')}; counted {k.get('status')}"
        rows.append(row)
    return rows


def pooled(rows: list[dict], sel=lambda r: True) -> dict:
    """Per form over the selected READ rows: n, the rms, mean and largest residual in m, the rms w1 byte residual."""
    rr = [r for r in rows if r.get("status") == "READ" and sel(r)]
    out = {"cells": len(rr)}
    if not rr:
        out["status"] = f"{NI}: no READ cell"
        return out
    for f in P.FORMS:
        res = [r["resid"][f] for r in rr]
        out[f] = {"rms": _rms(res), "mean": _mean(res), "worst": max(res, key=abs),
                  "w1_bytes_rms": _rms([r["w1_bytes_resid"][f] for r in rr])}
    out["lowest rms (DESCRIPTIVE)"] = min(P.FORMS, key=lambda f: out[f]["rms"])
    out["status"] = "READ"
    return out


def refit(rows: list[dict]) -> dict:
    """The descriptive refit (registered grid, design-P1 fit_s.py's): per (distance, law) the constant minimising
    rms(m_hat - m) over RR's READ cells (m_hat the mean over each cell's calls), ties to the smallest; printed beside the
    registered constants, never used to price anything."""
    rr = [r for r in rows if r.get("status") == "READ"]
    if not rr:
        return {"status": f"{NI}: no READ cell"}
    out = {}
    for dist, law in P.REFIT_LAWS:
        best = None
        for C in P.REFIT_GRID_MIB:
            res = []
            for r in rr:
                mh = float(np.mean([np.mean(law_miss(law, ds if dist == "start" else dm, C)) for ds, dm in r["_d"]]))
                res.append(mh - r["m"])
            rms = _rms(res)
            if best is None or rms < best[0] - 1e-12:
                best = (rms, C, _mean(res))
        out[f"{law} between {'starts' if dist == 'start' else 'mid-lives'}"] = {
            "C_star_mib": best[1], "rms": best[0], "mean": best[2]}
    out["cells"] = len(rr)
    out["status"] = "READ (DESCRIPTIVE: a constant read on RR's fresh pages, never a registered one)"
    return out


def slope(ks, ys):
    if len(ks) < 2:
        return None
    k, y = np.asarray(ks, np.float64), np.asarray(ys, np.float64)
    if np.ptp(k) == 0:
        return None
    return float(np.polyfit(k, y, 1)[0])


def wave_numbers(curve: list) -> dict:
    """RR-R3's numbers on one curve of wave spreads (index k = wave), NUMBERS ONLY (owner, 2026-11-11: no label, no
    band): w_inf = the median spread over k >= K_FLAT (at least FLAT_MIN such waves, else NOT IDENTIFIED), k_90 = the
    first wave whose spread reaches 0.9 w_inf, slope = the OLS slope of the spread over k >= K_FLAT, per wave."""
    pts = [(k, y) for k, y in enumerate(curve) if _fin(y)]
    flat = [(k, y) for k, y in pts if k >= P.K_FLAT]
    if len(flat) < P.FLAT_MIN:
        return {"status": f"{NI}: {len(flat)} waves read at k >= {P.K_FLAT} (needs {P.FLAT_MIN})", "waves": len(curve)}
    w_inf = float(st.median([y for _k, y in flat]))
    return {"w_inf": w_inf, "k_90": next((k for k, y in pts if y >= 0.9 * w_inf), None),
            "slope_flat": slope([k for k, _y in flat], [y for _k, y in flat]), "waves": len(curve), "status": "READ"}


def pooled_curve(curves: list) -> list:
    """The median over curves of each wave's spread, read with at least POOL_MIN_CELLS curves holding the wave."""
    K = max((len(c) for c in curves), default=0)
    out = []
    for k in range(K):
        v = [c[k] for c in curves if len(c) > k and _fin(c[k])]
        out.append(float(st.median(v)) if len(v) >= P.POOL_MIN_CELLS else None)
    return out


def wcp_group(x) -> str:
    e = P.WCP_EDGES
    if not _fin(x):
        return "unread"
    if x < e[0]:
        return f"W_c/P under {e[0]:g}"
    for lo, hi in zip(e, e[1:]):
        if lo <= x < hi:
            return f"W_c/P {lo:g} to {hi:g}"
    return f"W_c/P {e[-1]:g} and over"


def wave_reading(u: dict, page: str) -> dict:
    """RR-R3 per tile, NUMBERS ONLY: every READ cell's wave spreads and numbers; the pooled curve and its numbers for the
    tile, for each series and for each W_c / P group; the life spread and the hit/miss re-reader lives beside."""
    cells, groups, ls, rl = {}, {}, [], []
    for lab, c in page_cells(page).items():
        s = cell_stamps(u, lab)
        if s.get("status") != "READ":
            cells[lab] = {"status": s.get("status")}
            continue
        wcp = P.geometry(c["cfg"], c["counts"], c["tile"])["W_c_over_P"]
        cells[lab] = {"wave_spread": s["wave_spread"], "numbers": wave_numbers(s["wave_spread"]), "series": c["series"],
                      "W_c_over_P": wcp, "life_spread": s.get("life_spread"), "reread_life_us": s.get("reread_life_us")}
        for key in ("tile", f"series {c['series']}", wcp_group(wcp)):
            groups.setdefault(key, []).append(s["wave_spread"])
        ls.append(s.get("life_spread"))
        if s.get("reread_life_us"):
            rl.append(s["reread_life_us"]["miss_over_hit"])

    def pc(key):
        cv = pooled_curve(groups.get(key, []))
        return {"cells": len(groups.get(key, [])), "pooled_curve": cv, "numbers": wave_numbers(cv)}
    return {"cells": cells, "pooled": pc("tile"),
            "by series": {s: pc(f"series {s}") for s in ("A", "B", "C")},
            "by W_c / P": {g: pc(g) for g in sorted(k for k in groups if k.startswith("W_c/P"))},
            "life_spread_median": _med(ls), "reread_life_miss_over_hit_median": _med(rl),
            "status": "READ" if groups.get("tile") else f"{NI}: no READ cell"}


def rep_counter_unit(results: Path, session: Path, R: dict) -> dict:
    """The replicate's counted page (unit br, the pinned counter route, NATIVE) read under RR-R1's rules: the page
    present (lock1410/r3c-g1.json), the registered histogram page (sha256), the tile, the NATIVE arm, the versions
    (the page's stack), the page counted under the session lock (the route's --page-lock-mhz, ncu.lock_mhz; the driver's
    lock_and_check read the lock back before the page, and the pinned bytes step records no read-back after its tool:
    stated, not a pc-locks.env line), the page's own status line exit 0 or 1; then per NATIVE cell counter_cell's rules
    (every call counted, the 3% spread, counter sanity). An invalid unit reads no number."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["phi-3.5-moe"]
    d, dwhy = RP.unit_dir(results, REP_BYTES)
    f = sorted(d.glob(f"lock{LOCK}/r3c-g1.json")) if d else []
    u = {"lock": LOCK, "lock_note": ("the route's --page-lock-mhz and the driver's read-back before the page; no "
                                     "read-back after the pinned bytes step's tool exists")}
    if len(f) != 1:
        u.update(valid=False, why=f"no counter page ({dwhy or f'{len(f)} lock{LOCK}/r3c-g1.json'})", cells={})
        return u
    page = json.loads(f[0].read_text())
    de, why = page.get("design") or {}, []
    if de.get("model") != "phi-3.5-moe" or ((de.get("histogram") or {}).get("file_sha256")
                                            != R["pages"][P.PHI_128]["sha256"]):
        why.append("the counted page is not the registered Phi-3.5 128g1 histogram page")
    if (de.get("block_m"), de.get("block_n"), de.get("block_k"), de.get("num_warps"), de.get("num_stages")) != P.T128:
        why.append(f"the counted tile is not {P.tile_key(P.T128)}")
    if "native" not in (de.get("arms") or []):
        why.append("no NATIVE arm on the counted page")
    if not RP.versions_ok(page.get("stack")):
        why.append(f"versions {page.get('stack')} are not {P.VERSIONS} (python on major.minor)")
    if (page.get("ncu") or {}).get("lock_mhz") != LOCK:
        why.append(f"the page was counted at {(page.get('ncu') or {}).get('lock_mhz')} MHz, not the session's {LOCK}")
    stf = Path(session) / "logs" / f"lock{LOCK}-pages-phi-3.5-moe-br.status"
    stl = stf.read_text().split() if stf.is_file() else []
    if not (len(stl) >= 3 and stl[0] == "G=1" and stl[1] == "exit" and stl[2] in ("0", "1")):
        why.append(f"the page's status line reads {' '.join(stl[:3]) or 'nothing'}, not 'G=1 exit 0' or 1")
    cells = {}
    for c in page.get("cells") or []:
        if c.get("arm") != "native":
            continue
        lab, n = str(c.get("histogram") or "balanced"), int(c["n"])
        counts = [int(x) for x in c["bincount"]] if c.get("bincount") else [n * int(de.get("block_m") or 0)] * \
            cfg.num_experts
        pg, rec = c.get("per_gemm") or {}, c.get("recorded") or {}
        vals = (c.get("per_gemm_values") or {}).get("w1") or []
        w1 = pg.get("w1") or {}
        clk = (rec.get("w1") or {}).get("sm__cycles_elapsed.avg")
        row = {"counts": counts,
               "per_call": {"w1": {"dram__bytes_read.sum": [v.get("dram_bytes_read") if isinstance(v, dict) else v
                                                            for v in vals]}},
               "per_gemm": {"w1": {"lts__t_sector_op_read_hit_rate.pct": w1.get("l2_read_hit_pct")},
                            "w2": {"dram__bytes_read.sum": (pg.get("w2") or {}).get("dram_bytes_read")}},
               "clock_mhz": {"w1": [clk / w1["gpu_time_ns"] * 1e3] if _fin(clk) and _fin(w1.get("gpu_time_ns"))
                             and w1["gpu_time_ns"] > 0 else []}}
        cells[(lab, n)] = counter_cell(row, cfg, P.T128)
    if why:
        cells = {k: {"status": f"{NR}: the unit is invalid ({'; '.join(why)})"} for k in cells}
    u.update(valid=not why, why="; ".join(why), cells=cells, dir=d.name)
    return u


def replicate(results: Path, session: Path, sr: dict, R: dict, brd: dict) -> dict:
    """RR-R4: Phi-3.5's 128g1 cells on this board, stamps (unit sr) and counted bytes (unit br under RR-R1's rules), each
    form's m_hat against m, the residuals; DESCRIPTIVE, a cross-board anchor to PC's and B3's Phi pages."""
    br = rep_counter_unit(results, session, R)
    if not brd["valid"]:
        br["valid"], br["why"] = False, "; ".join(x for x in (br.get("why"), f"board: {brd['why']}") if x)
        br["cells"] = {k: {"status": f"{NR}: the unit is invalid ({br['why']})"} for k in br.get("cells") or {}}
    out = {}
    for stem in sorted((sr.get("cells") or {})):
        s = cell_stamps(sr, stem)
        k = (br.get("cells") or {}).get(stem_key(stem)) or {"status": f"{NR}: not counted"}
        row = {"stamps": s.get("status"), "counted": k.get("status"), "page_label": sr.get("page_label")}
        if k.get("status") == "READ":
            row.update(m=k["m"], D_spread=k["spread"], clock_mhz=k.get("clock_mhz"))
        if s.get("status") == "READ":
            row["m_hat"] = s["m_hat"]
            if k.get("status") == "READ":
                row["resid"] = {f: s["m_hat"][f] - k["m"] for f in P.FORMS}
        out[stem] = row
    return {"br unit": {k: br.get(k) for k in ("valid", "why", "dir", "lock", "lock_note")}, "cells": out,
            "status": "READ" if any("resid" in r for r in out.values()) else f"{NI}: no cell read on both"}


def board_counted(results: Path, suffix: str, lock: str = f"lock{LOCK}") -> dict:
    """(label, n) -> {w1, w2, counts} of a pinned-route counter page's NATIVE cells (r3c-g*.json under `lock`; a
    'balanced' cell carries no bincount: n x BLOCK_M rows per expert). Used by the stand-in on SEEN pages."""
    from moe.spec import MODEL_CONFIGS
    d, _ = RP.unit_dir(results, suffix)
    f = sorted(d.glob(f"{lock}/r3c-g*.json")) if d else []
    if len(f) != 1:
        return {}
    page = json.loads(f[0].read_text())
    de = page.get("design") or {}
    if int(de.get("group_m", 1)) != 1:
        return {}
    E = MODEL_CONFIGS[de["model"]].num_experts if de.get("model") in MODEL_CONFIGS else None
    out = {}
    for c in page.get("cells", []):
        if c.get("arm") != "native":
            continue
        lab = str(c.get("histogram") or "balanced")
        cnt = c.get("bincount") or ([int(c["n"]) * int(de["block_m"])] * E if lab == "balanced" and E else None)
        if not cnt:
            continue
        pg = c.get("per_gemm") or {}
        out[(lab, int(c["n"]))] = {"w1": (pg.get("w1") or {}).get("dram_bytes_read"),
                                   "w2": (pg.get("w2") or {}).get("dram_bytes_read"),
                                   "counts": [int(x) for x in cnt], "block_m": int(de["block_m"]), "model": de.get("model")}
    return out



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


def board(session: Path) -> dict:
    """The board: C2's power read-back (700 W) and the session lock's probe held (no unit runs at another lock)."""
    D = Path(session) / "gh200-driver"
    pw = (D / "c2-power.env").read_text() if (D / "c2-power.env").is_file() else ""
    locks = (D / "locks.env").read_text().split() if (D / "locks.env").is_file() else []
    why = []
    if f"C2_POWER_W={P.POWER_W}" not in pw.split():
        why.append(f"c2-power.env does not record C2_POWER_W={P.POWER_W}")
    if f"LOCK_PROBE_{LOCK}=held" not in locks:
        why.append(f"no LOCK_PROBE_{LOCK}=held")
    return {"c2_power_env": pw.strip(), "valid": not why, "why": "; ".join(why)}


def strip(x):
    """The reading without its private arrays (keys starting with an underscore)."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def read(rr: Path, *, registration: dict | None = None, _skip_commit_check_for_tests: bool = False) -> dict:
    R = registration or reg()
    rr = Path(rr)
    results, session = rr / "results", rr / "session"
    out = {"registration": REG_NAME, "rr": str(rr),
           "status": ("MEASUREMENT, DESCRIPTIVE: no model form, no gate, no blind; the 42.5 MiB constant and the forms "
                      "were found post-hoc on SEEN pages (design-P1 2.3), RR reads them on fresh synthetic cells")}
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
    brd = board(session)
    out["board"] = brd
    census, cwhy = census_occupancy(results)
    occ_ok = {t: census.get(t) == w for t, w in P.OCCUPANCY.items()}
    out["census"] = {"occupancy": census, "why": cwhy,
                     "registered_check": {t: {"registered": w, "census": census.get(t), "ok": occ_ok[t]}
                                          for t, w in P.OCCUPANCY.items()}}
    pd, pw = RP.unit_dir(results, OTHER_UNITS[PT])
    perturbs = {PT: {"dir": pd, "why": pw, "json": json.loads((pd / "perturb.json").read_text())
                     if pd and (pd / "perturb.json").is_file() else {}}}
    rcd, rcw = RP.unit_dir(results, OTHER_UNITS[RCP])
    regchecks = {RCP: {"dir": rcd, "why": rcw}}
    shapes = P.load_shapes()
    units, counters = {}, {}
    for label in STAMP_UNITS:
        u = stamp_unit(results, session, label, R, perturbs, regchecks, shapes)
        tile = P.T64 if label == "s64" else P.T128
        extra = [] if brd["valid"] else [f"board: {brd['why']}"]
        if not occ_ok[P.tile_key(tile)]:
            extra.append(f"the census reads {census.get(P.tile_key(tile))} CTAs per SM at {P.tile_key(tile)}, the "
                         f"registered {P.OCCUPANCY[P.tile_key(tile)]} (the wave, W_c and every gap need it)")
        if extra:
            u["valid"], u["why"] = False, "; ".join(x for x in [u["why"], *extra] if x)
        units[label] = u
    for label in COUNTER_UNITS:
        c = counter_unit(results, session, label, R, shapes)
        if not brd["valid"]:
            c["valid"], c["why"] = False, "; ".join(x for x in (c["why"], f"board: {brd['why']}") if x)
            c["cells"] = {lab: {"status": f"{NR}: the unit is invalid ({c['why']})"} for lab in (c.get("cells") or {})}
        counters[label] = c
    out["RR-R0"] = {"stamps": {lab: {k: u.get(k) for k in ("variant", "valid", "why", "page_label", "gate", "hard",
                                                           "regcheck", "lock_after", "perturb_lock_after",
                                                           "occupancy_leg", "dir", "launches", "exact_time",
                                                           "time_offset_ns", "timer_unit")} for lab, u in units.items()},
                    "counters": {lab: {k: c.get(k) for k in ("valid", "why", "lock_after", "dir", "seconds")}
                                 for lab, c in counters.items()}}
    out["RR-R1"] = {lab: c.get("cells") for lab, c in counters.items()}
    r2 = {}
    all_rows = []
    for s_lab, c_lab in P.PAIRED.items():
        rows = join(units[s_lab], counters[c_lab], P.PAGE_OF[s_lab])
        all_rows += rows
        r2[P.tile_key(P.T64 if s_lab == "s64" else P.T128)] = {"cells": rows, "pooled": pooled(rows)}
    fresh = pooled(all_rows, lambda r: r["P"] not in P.ANCHOR_P)
    r2["pooled, both tiles"] = pooled(all_rows)
    r2["pooled, FRESH synthetic cells (P not 100 or 224)"] = fresh
    r2["pooled, seen-anchor cells (P 100, 224)"] = pooled(all_rows, lambda r: r["P"] in P.ANCHOR_P)
    r2["by series"] = {s: pooled(all_rows, lambda r, s=s: r["series"] == s) for s in ("A", "B", "C")}
    r2["refit (DESCRIPTIVE), both tiles"] = refit(all_rows)
    r2["refit (DESCRIPTIVE), FRESH cells"] = refit([r for r in all_rows if r["P"] not in P.ANCHOR_P])
    r2["clock note"] = CLOCK_NOTE
    r2["registered forms"] = {f: {"distance": v[0], "law": v[1], "C_mib": v[2], "origin": v[3]} for f, v in P.FORMS.items()}
    r2["design-P1 printed beside (POST-HOC on SEEN pages)"] = P.DESIGN_PRINTED
    out["RR-R2"] = r2
    out["RR-R3"] = {P.tile_key(P.T128): wave_reading(units["s128"], P.PAGE_OF["s128"]),
                    P.tile_key(P.T64): wave_reading(units["s64"], P.PAGE_OF["s64"]),
                    "stand-in beside (SEEN pages, frozen at registration; never used by a reading)": R.get(
                        "stand-in wave numbers (SEEN pages, frozen)")}
    out["RR-R4"] = replicate(results, session, units["sr"], R, brd)
    bad = check_repo_modules(R) if not _skip_commit_check_for_tests else []
    if bad:
        raise SystemExit(f"REFUSED: the reader loaded a repo module that is not pinned with its registered bytes: {bad[:5]}")
    return strip(out)


# ================================================================== the stand-in readout on SEEN published pages

CAL3_SESSION = "2026-10-27-nvidia_gh200_480gb-cal3-session"
PC_SESSION = "2026-11-06-nvidia_gh200_480gb-pc-session"
PC2_SESSION = "2026-11-10-nvidia_gh200_480gb-pc2-session"
#: design-P1's four stamped families (work/stamps.py UNITS; FROZEN): (session, unit suffix, model, the stamps' board)
STANDIN_STAMPS = ((CAL3_SESSION, "instr-mixtral-8x7b-st7", "mixtral-8x7b", "C3"),
                  (CAL3_SESSION, "instr-qwen2-57b-a14b-stq", "qwen2-57b-a14b", "C3"),
                  (PC_SESSION, "instr-phi-3.5-moe-se1", "phi-3.5-moe", "PC"),
                  (PC_SESSION, "instr-phi-3.5-moe-se2", "phi-3.5-moe", "PC"))
#: design-P1's counted pages (work/load.py PAGES; FROZEN): (session, model, unit, board). Every board's counted m is
#: read against the stamped cell's m_hat, as design-P1's fit_s.py did (54 (cell, board) rows there)
STANDIN_COUNTS = (
    ("2026-10-20-nvidia_gh200_480gb-tileA-session", "olmoe-1b-7b", "b128g1", "A"),
    ("2026-10-20-nvidia_gh200_480gb-tileA-session", "qwen2-57b-a14b", "b128g1", "A"),
    ("2026-10-20-nvidia_gh200_480gb-tileA-session", "mixtral-8x7b", "b64g1", "A"),
    ("2026-10-21-nvidia_gh200_480gb-tileA2-session", "olmoe-1b-7b", "b128g1", "A2"),
    ("2026-10-21-nvidia_gh200_480gb-tileA2-session", "qwen2-57b-a14b", "b128g1", "A2"),
    ("2026-10-21-nvidia_gh200_480gb-tileA2-session", "mixtral-8x7b", "b64g1", "A2"),
    ("2026-10-21-nvidia_gh200_480gb-settleS1-session", "olmoe-1b-7b", "st", "S1"),
    ("2026-10-21-nvidia_gh200_480gb-settleS1-session", "qwen2-57b-a14b", "st", "S1"),
    ("2026-10-21-nvidia_gh200_480gb-settleS1-session", "mixtral-8x7b", "b128g1", "S1"),
    ("2026-10-21-nvidia_gh200_480gb-settleS1-session", "mixtral-8x7b", "p64g1", "S1"),
    ("2026-10-21-nvidia_gh200_480gb-cal2-session", "mixtral-8x7b", "b128g1", "C2"),
    ("2026-10-21-nvidia_gh200_480gb-cal2-session", "mixtral-8x22b", "b128g1", "C2"),
    ("2026-10-27-nvidia_gh200_480gb-cal4-session", "mixtral-8x7b-tp2", "b128g1", "C4"),
    ("2026-11-05-nvidia_gh200_480gb-tileB3-session", "phi-3.5-moe", "b128g1", "B3"),
    ("2026-11-05-nvidia_gh200_480gb-tileB3-session", "phi-3.5-moe", "b64g1", "B3"),
    ("2026-11-05-nvidia_gh200_480gb-tileB3-session", "qwen1.5-moe-a2.7b", "b128g1", "B3"),
    ("2026-11-06-nvidia_gh200_480gb-pc-session", "phi-3.5-moe", "b128g1", "PC"),
    ("2026-11-06-nvidia_gh200_480gb-pc-session", "phi-3.5-moe", "b64g1", "PC"),
)
#: design-P1's fit on these pages (DESIGN.md 2.3, work/fit_s.out: the step's rms of m at its best constant), beside
DESIGN_FIT = {"pooled": {P.PRIMARY_FORM: 0.048, "F2 step at 45.0 MiB, between mid-lives": 0.043},
              "Phi-3.5 only": {P.PRIMARY_FORM: 0.045, "F2 step at 45.0 MiB, between mid-lives": 0.035}}


def standin_stamps(pub: Path, sess: str, suffix: str, model: str) -> dict:
    """(model, BLOCK_M, label, n) -> {m_hat per form (median over the cell's readable w1 launches), the wave curve}, read
    on the raw stamps as design-P1 read them (pair_read's own row checks: every live stamp and life positive, the
    launch's live rows its geometry's)."""
    d = next((pub / sess / "results").glob(f"*-{suffix}"), None)
    if d is None:
        return {}
    man = json.loads((d / "stamps.json").read_text())
    per = {}
    for e in man.get("launches", []):
        if e.get("gemm") != "w1" or "error" in e:
            continue
        tile = tuple(int(e[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"))
        pr = pair_read(np.load(d / "stamps" / e["file"]), np.load(d / "stamps" / e["experts_file"]),
                       e["num_tokens_post_padded"], e["N"], e["K"], tile)
        if pr.get("status") == "READ":
            per.setdefault((model, tile[0], str(e["label"]), int(e["n"])), []).append(pr)
    out = {}
    for k, prs in per.items():
        K = min(len(x["wave_spread"]) for x in prs)
        out[k] = {"m_hat": {f: _med([x["m_hat"][f] for x in prs]) for f in P.FORMS}, "calls": len(prs),
                  "wave_spread": [_med([x["wave_spread"][i] for x in prs]) for i in range(K)],
                  "together_share": _med([x["together_share"] for x in prs])}
    return out


def standin(pub: Path) -> dict:
    """RR's readings on SEEN published pages, both boards' stamps: C3's Mixtral-8x7B (st7) and Qwen2-57B (stq) and PC's
    Phi-3.5 (se1 128x128, se2 64x128), each stamped cell's form m_hat against the counted m of every board holding the
    cell (design-P1's 18 counted pages, its 54 (cell, board) rows), pooled, Phi-3.5 only, by family, and on the same
    board (PC's Phi); design-P1's figures beside; RR-R3's numbers per stamped family; and RR-R3 on PC2's multi-wave w1
    launches (decode tiles). A STAND-IN on SEEN pages, the pages the forms were found on: a check of the code, never an
    RR reading."""
    from moe.spec import MODEL_CONFIGS
    pub = Path(pub)
    stamped, waves, bd = {}, {}, {}
    for sess, suffix, model, board in STANDIN_STAMPS:
        s = standin_stamps(pub, sess, suffix, model)
        stamped.update(s)
        bd.update({k: board for k in s})
        waves[suffix] = {"cells": len(s), "pooled_curve": pooled_curve([x["wave_spread"] for x in s.values()])}
        waves[suffix]["numbers"] = wave_numbers(waves[suffix]["pooled_curve"])
    rows = []
    for sess, model, unit, board in STANDIN_COUNTS:
        for (lab, n), k in board_counted(pub / sess / "results", f"{model}-{unit}-r3-counters", "lock*").items():
            key = (model, k["block_m"], lab, n)
            if key not in stamped or not _fin(k.get("w1")):
                continue
            geo = P.geometry(MODEL_CONFIGS[model], k["counts"], (k["block_m"], 128, 64, 8, 3))
            m = P.counted_m(k["w1"], geo["C_u_bytes"], geo["R_bytes"])
            if m is None:
                continue
            rows.append({"cell": f"{model} {key[1]} {lab} n{n}", "model": model, "counted_board": board,
                         "stamps_board": bd[key], "m": m, "m_hat": stamped[key]["m_hat"],
                         "resid": {f: stamped[key]["m_hat"][f] - m for f in P.FORMS}})

    def summ(sel):
        rr = [r for r in rows if sel(r)]
        return {"rows": len(rr), **{f: {"rms": _rms([r["resid"][f] for r in rr]), "mean": _mean([r["resid"][f] for r in rr])}
                                    for f in P.FORMS}} if rr else {"rows": 0}
    out = {"what": standin.__doc__.split("\n\n")[0].replace("\n    ", " "), "source": str(pub),
           "rows": rows, "pooled (every counted board)": summ(lambda r: True),
           "Phi-3.5 only": summ(lambda r: r["model"] == "phi-3.5-moe"),
           "same board (stamps and bytes on one board: PC's Phi-3.5)": summ(lambda r: r["counted_board"] == r["stamps_board"]),
           "by family": {m: summ(lambda r, m=m: r["model"] == m) for m in sorted({r["model"] for r in rows})},
           "design-P1 beside (DESIGN.md 2.3, work/fit_s.out)": DESIGN_FIT, "RR-R3 numbers per stamped family": waves}
    out["read against design-P1"] = {
        sel: {f: {"here": out[sel][f]["rms"], "design": v, "difference": out[sel][f]["rms"] - v} for f, v in d.items()}
        for sel, d in (("pooled (every counted board)", DESIGN_FIT["pooled"]), ("Phi-3.5 only", DESIGN_FIT["Phi-3.5 only"]))}
    if (pub / PC2_SESSION).is_dir():
        out["PC2 wave spread (decode tiles; RR-R3's rule on SEEN pages of another regime)"] = standin_pc2(pub / PC2_SESSION)
    return strip(out)


def standin_waves(d: dict) -> dict:
    """The frozen literal the registration prints beside RR-R3 (rrcommon.STANDIN_WAVE): each stamped family's pooled
    numbers, floats rounded to 4 places."""
    return {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v["numbers"].items()}
            for k, v in d["RR-R3 numbers per stamped family"].items()}


def standin_pc2(pc2: Path) -> dict:
    """RR-R3's wave-start spread on PC2's valid multi-wave w1 launches (decode tiles, W_c = 132 x the registered
    occupancy), per cell; PC2's own validity (read_pc2.stamp_unit, unchanged)."""
    import pc2common as PC2C
    results, session = pc2 / "results", pc2 / "session"
    Rp2 = json.loads((REPO / "docs" / "registered" / f"{PC2C.NAME}.json").read_text())
    pd, pw = RP.unit_dir(results, R2.OTHER_UNITS[R2.PT])
    perturbs = {R2.PT: {"dir": pd, "why": pw, "json": json.loads((pd / "perturb.json").read_text())
                        if pd and (pd / "perturb.json").is_file() else {}}}
    rcd, rcw = RP.unit_dir(results, R2.OTHER_UNITS[R2.RCP])
    regchecks = {R2.RCP: {"dir": rcd, "why": rcw}}
    shapes = PC2C.load_shapes()
    out = {}
    for lab in ("sd", "ss", "sw"):
        u = R2.stamp_unit(results, session, lab, Rp2, perturbs, regchecks, shapes)
        d, _w = RP.unit_dir(results, R2.STAMP_UNITS[lab][0])
        if d is None:
            out[lab] = {"status": f"{NR}: no stamp page"}
            continue
        man = json.loads((d / "stamps.json").read_text())
        files = {(e.get("stem") or e.get("label"), e.get("call"), e["gemm"]): e for e in man.get("launches", [])
                 if "error" not in e}
        cells = {}
        for stem, c in sorted((u.get("cells") or {}).items()):
            curves = []
            for x in c["w1"]:
                W = PC2C.wave(x["tile"]) or 0
                if not x.get("valid") or W <= 0 or x["live"] < (P.K_FLAT + P.FLAT_MIN) * W:
                    continue
                e = files.get((stem, x.get("call"), "w1"))
                se = exact_se(np.load(d / "stamps" / e["file"])) if e else None
                if se is None:
                    continue
                curves.append(wave_spreads(se[0], float(np.median(se[2])), W))
            if len(curves) >= P.MIN_VALID_CALLS:
                K = min(len(cv) for cv in curves)
                cv = [_med([x[k] for x in curves]) for k in range(K)]
                cells[stem] = {"tile": PC2C.tile_key(c["w1"][0]["tile"]), "waves": K, "wave_spread": cv,
                               "numbers": wave_numbers(cv)}
        out[lab] = {"unit valid (PC2's rules)": u.get("valid"), "cells": cells}
    return out



# ================================================================== text

def txt(d: dict) -> str:
    L = [f"RR READING ({d['registration']}): {d['status']}", f"  pages: {d['rr']}",
         f"  commit: {json.dumps(d['commit'], sort_keys=True)}", f"  board: {json.dumps(d['board'], sort_keys=True)}",
         f"  census: {json.dumps(d['census'], sort_keys=True, default=str)}", ""]
    for k in ("RR-R0", "RR-R1", "RR-R2", "RR-R3", "RR-R4"):
        L.append(f"{k}:")
        v = d[k]
        if isinstance(v, dict):
            for kk, vv in v.items():
                L.append(f"  {kk}: {json.dumps(vv, sort_keys=True, default=RP._default)}")
        else:
            L.append(f"  {json.dumps(v, sort_keys=True, default=RP._default)}")
    return "\n".join(L) + "\n"


def standin_txt(d: dict) -> str:
    L = [f"STAND-IN READOUT on SEEN published pages ({d['source']}): {d['what']}"]
    for k, v in d.items():
        if k in ("what", "source"):
            continue
        L.append(f"{k}:")
        if isinstance(v, dict):
            for kk, vv in v.items():
                L.append(f"  {kk}: {json.dumps(vv, sort_keys=True, default=RP._default)}")
        else:
            L.append(f"  {json.dumps(v, sort_keys=True, default=RP._default)}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rr", type=Path, help="the published RR session directory (results/, session/)")
    g.add_argument("--standin", type=Path, help="results/published: the stand-in readout on SEEN pages")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.standin:
        d = standin(a.standin)
        atomic_write(a.out_dir / "standin.read.json", dumps(d))
        atomic_write(a.out_dir / "standin.read.txt", standin_txt(d))
        print(standin_txt(d))
        return 0
    d = read(a.rr)
    atomic_write(a.out_dir / "rr.read.json", dumps(d))
    atomic_write(a.out_dir / "rr.read.txt", txt(d))
    print(txt(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
