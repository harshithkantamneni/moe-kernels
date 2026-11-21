"""Rental 7's shared rules (docs/registered/2026-10-13-rental7-*).

One module for the registration generator (register.py, reg7.py) and every rental-7 scorer, so a
rule is computed one way on both sides. It imports rental 6's (r6common), which imports rental 5's,
4's and 3's; nothing is copied. Pure functions over numbers, pages and the registered JSON.

THE RULES THIS FILE OWNS (each stated in full in the registrations; the 10 scorer ambiguities of
design-r7-review (h) are resolved in reg7.AMBIGUITIES and implemented here):
  r7-median-run3 THE SCORER'S RULE, rebuilt from the page's cells.csv (rebuild below). R3 is
                 rental 6's, byte-identical (its registered pin), run at --slip-policy cell with
                 --min-clean-repeats 0, so it times every repeat of every cell and never stops early;
                 its stored G1 and stored row medians are never read. Per repeat (one cells.csv row):
                 DIPPED read = a clock read more than one 15 MHz step under the lock (1694 MHz or less
                 at 1710: reads are integer MHz); RUN-GUARDED = 3 or more consecutive dipped reads of
                 the repeat's 15 per-burst reads; LOCK-SLIPPED = the repeat's median clock
                 (sm_clock_load_mhz) under lock - 15; CLEAN = status ok, ms_p50 > 0, not slipped, not
                 run-guarded (the endpoint DRIFT flag is recorded only). Per cell: ms, clock and
                 host_bound over the clean repeats; LOST under 6 clean of its timed repeats. G1 PASS iff
                 every kept cell's clock is read, every kept cell within one 15 MHz step of the top kept
                 clock, and non-clean timed rows at most 10% of the page's timed rows. A page whose
                 cells.csv rows per cell differ from its report's repeat_rows (or that has no cells.csv,
                 no report or no lock) is NOT SCORED. G1 FAIL counts the page in ALL and drops it in
                 CLEAN (rental 2's rule 3).
  P20 view       DESCRIPTIVE, printed beside every verdict: the same rebuild with today's exclusions
                 (CLEAN = status ok, ms_p50 > 0, not drift-flagged, not slipped), its own medians,
                 clocks and lost cells, and G1 with the same clock clauses on ITS kept cells and the
                 owner's 20% ceiling.
  Q1-H           rental 6's statistic; the CR1 se on (model, n) clusters needs at least 3 clusters
                 (review (h) 5), else UNDECIDED; the rivals are scored whenever 6 or more pairs and 3
                 clusters are valid, whatever the copies clause reads (owner decision 2); the copies
                 clause is the pair-weighted mean over models with both sides of mean g(c15) - mean
                 g(c9) (identical to rental 6's unmatched means when every Q1 page is valid).
  E1 per shape   E1-SKEW holds only if the pooled cells AND each shape's cells meet the bar; a
                 shape is CERTIFIED in CLEAN only with at least 2 of its 3 pages in CLEAN.
  l2open         the nearest of v3, ROWKEY39.5 and ROWKEY60 by rms over the skewed (PT, PW) w1 rows
                 is NAMED when the next is at least 2x farther and every skewed row of the nearest
                 is within 5%; else NOT SEPARATED.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(HERE.parent / "rental6")):
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


C6 = _load("r6common", HERE.parent / "rental6" / "r6common.py")
C3, C4, C5 = C6.C3, C6.C4, C6.C5
CARD = C6.CARD
DATE = "2026-10-13"
PARTS = ("g1rule", "skew", "q1", "l2open")
NAMES = {p: f"{DATE}-rental7-{p}-gh200" for p in PARTS}
HIST_DIR = f"docs/registered/{DATE}-rental7-skew-hist"
r6 = C6.r6

RULE = "r7-median-run3"
RUN_GUARD_READS = 3
NON_CLEAN_CAP = 0.10
P20_CEILING = 0.20
MIN_CLEAN = C6.MIN_CLEAN
MIN_PAIRS = C6.MIN_PAIRS
MIN_CLUSTERS = 3
MIN_CLEAN_PAGES_PER_SHAPE = 2
L2_FACTOR, L2_MAX_ABS = 2.0, 0.05
L2_RIVALS = ("v3", "ROWKEY39.5", "ROWKEY60")


def registration(repo: Path, part: str) -> dict:
    return json.loads((Path(repo) / "docs" / "registered" / f"{NAMES[part]}.json").read_text())


# --------------------------------------------------------------------------
# r7-median-run3, rebuilt from cells.csv; the P20 view
# --------------------------------------------------------------------------

LOCK_STEP_MHZ = 15.0
#: the read cadence (registered, rental7-g1rule): one NVML read per burst, bursts 140 to 166 ms apart
READ_PERIOD_MS = (140, 166)


def _f(x):
    return float(x) if x not in (None, "") else None


def repeat_rows(raw: list[dict]) -> dict:
    """cells.csv rows (csv.DictReader dicts) by cell (label, n, arm): per repeat its status, time,
    median clock, reads, drift flag and host flag."""
    import re
    out: dict = {}
    for x in raw:
        lab = re.search(r"histogram=([^;]+)", x.get("detail") or "")
        if lab is None:
            continue
        out.setdefault((lab.group(1), int(x["tiles"]), x["arm"]), []).append({
            "repeat": int(x["repeat"]), "status": x.get("status", "ok"), "ms": _f(x.get("ms_p50")) or 0.0,
            "mhz": _f(x.get("sm_clock_load_mhz")), "reads": [float(v) for v in (x.get("clock_samples_mhz") or "").split()],
            "drift_ok": {"True": True, "False": False}.get(x.get("clock_drift_ok") or ""),
            "host_bound": (x.get("host_bound") or "") == "True"})
    return out


def dipped(v: float, lock: float) -> bool:
    return v < lock - LOCK_STEP_MHZ


def dip_run(rep: dict, lock: float) -> int:
    best = cur = 0
    for v in rep["reads"]:
        cur = cur + 1 if dipped(v, lock) else 0
        best = max(best, cur)
    return best


def lock_slip(rep: dict, lock: float) -> bool:
    return rep["mhz"] is not None and rep["mhz"] == rep["mhz"] and dipped(rep["mhz"], lock)


def run_guard(rep: dict, lock: float) -> bool:
    return dip_run(rep, lock) >= RUN_GUARD_READS


def clean_r7(rep: dict, lock: float) -> bool:
    return rep["status"] == "ok" and rep["ms"] > 0 and not lock_slip(rep, lock) and not run_guard(rep, lock)


def clean_p20(rep: dict, lock: float) -> bool:
    return rep["status"] == "ok" and rep["ms"] > 0 and rep["drift_ok"] is not False and not lock_slip(rep, lock)


def _cell(reps: list[dict], lock: float, pred) -> dict:
    got = [r for r in reps if pred(r, lock)]
    clk = [r["mhz"] for r in got if r["mhz"] is not None]
    return {"clean": len(got), "ms": st.median(r["ms"] for r in got) if got else None,
            "mhz": st.median(clk) if clk else None, "host_bound": any(r["host_bound"] for r in got),
            "lost": len(got) < MIN_CLEAN}


def g1_rule(rows: list[dict], *, prefix: str = "", cap: float = NON_CLEAN_CAP) -> dict:
    """G1 over rebuilt rows: the clock clauses on the kept (not lost) cells and the non-clean share
    of every timed row against `cap`. `prefix` "" reads the r7 fields, "p20_" the P20 view's."""
    kept = [r for r in rows if not r[f"{prefix}lost"]]
    name = lambda r: f"{r['histogram']}/{r['arm']}/n{r['tiles']}"
    clocks = [r[f"{prefix}mhz"] for r in kept if r[f"{prefix}mhz"] is not None]
    top = max(clocks) if clocks else None
    unread = [name(r) for r in kept if r[f"{prefix}mhz"] is None]
    low = [name(r) for r in kept if top is not None and r[f"{prefix}mhz"] is not None and r[f"{prefix}mhz"] < top - LOCK_STEP_MHZ]
    rr = sum(r["repeat_rows"] for r in rows)
    nc = rr - sum(r[f"{prefix}clean_repeats"] for r in rows)
    share_ok = nc <= cap * rr + 1e-9
    return {"verdict": "PASS" if clocks and not unread and not low and share_ok else "FAIL", "top_mhz": top,
            "clock_unread": unread, "below_top_by_a_step": low, "repeat_rows": rr, "non_clean_rows": nc,
            "share": round(nc / rr, 6) if rr else 0.0, "cap": cap, "share_verdict": "PASS" if share_ok else "FAIL",
            "lost_cells": [name(r) for r in rows if r[f"{prefix}lost"]]}


def rebuild(report: dict, raw: list[dict] | None, lock: float | None = None, *, check_counts: bool = True):
    """(the report as the scorer reads it, None) or (None, why NOT SCORED): every treads_table row's
    time, clock, host flag, clean count and LOST rebuilt from cells.csv under r7-median-run3 (and the
    P20 fields beside), G1 recomputed (the report's G1 gate replaced), provenance kept as stored.
    `lock` defaults to the report's lock_mhz."""
    import copy
    lock = lock if lock is not None else report.get("lock_mhz")
    if not lock:
        return None, "no lock_mhz on the page: the run guard has no reference"
    if raw is None:
        return None, "no cells.csv beside the report"
    by = repeat_rows(raw)
    r = copy.deepcopy(report)
    rows = r.get("treads_table") or []
    keys = {(str(t["histogram"]), int(t["tiles"]), str(t["arm"])) for t in rows}
    if set(by) != keys:
        return None, f"cells.csv names {len(set(by))} cells, the report {len(keys)}"
    total = 0
    for t in rows:
        k = (str(t["histogram"]), int(t["tiles"]), str(t["arm"]))
        reps = by[k]
        if check_counts and (t.get("repeat_rows") is None or int(t["repeat_rows"]) != len(reps)):
            return None, f"cells.csv holds {len(reps)} repeats of {k}, the report {t.get('repeat_rows')}"
        total += len(reps)
        a, b = _cell(reps, lock, clean_r7), _cell(reps, lock, clean_p20)
        t.update({"repeat_rows": len(reps), "clean_repeats": a["clean"], "repeats": a["clean"], "ms_p50": a["ms"],
                  "sm_clock_load_mhz": a["mhz"], "host_bound": a["host_bound"], "lost": a["lost"],
                  "lock_slip_repeats": sum(lock_slip(x, lock) for x in reps),
                  "run_guard_repeats": sum(run_guard(x, lock) for x in reps),
                  "drift_flag_repeats": sum(x["drift_ok"] is False for x in reps),
                  "clean_repeats_p20": b["clean"], "ms_p50_p20": b["ms"], "sm_clock_load_mhz_p20": b["mhz"],
                  "host_bound_p20": b["host_bound"], "lost_p20": b["lost"]})
        t["mhz"], t["p20_mhz"] = a["mhz"], b["mhz"]
        t["p20_clean_repeats"], t["p20_lost"] = b["clean"], b["lost"]
    if check_counts and report.get("repeat_rows") is not None and int(report["repeat_rows"]) != total:
        return None, f"cells.csv holds {total} repeat rows, the report {report['repeat_rows']}"
    g = g1_rule(rows)
    p = g1_rule(rows, prefix="p20_", cap=P20_CEILING)
    for t in rows:
        for k in ("mhz", "p20_mhz", "p20_clean_repeats", "p20_lost"):
            t.pop(k)
    r["g1_r7"] = {"rule": RULE, "lock_mhz": lock, **g, "stored_G1_not_read": True}
    r["g1_p20"] = {"label": "SIDE VIEW (DESCRIPTIVE): today's exclusions, its own kept cells and clocks, the 20% ceiling", **p}
    r["histogram_gates"] = dict(r.get("histogram_gates") or {})
    r["histogram_gates"]["G1_lock_thermal"] = {"verdict": g["verdict"], "rule": RULE, "rebuilt_from": "cells.csv"}
    r["gates"] = [dict(x, verdict=g["verdict"]) if x.get("tag") == "G1_lock_thermal" else x for x in (r.get("gates") or [])]
    if not any(x.get("tag") == "G1_lock_thermal" for x in r["gates"]):
        r["gates"].append({"kind": "VALIDITY", "tag": "G1_lock_thermal", "verdict": g["verdict"]})
    return r, None


def p20_report(rebuilt: dict) -> dict:
    """The rebuilt page read as the P20 view: each row's median, clock, host flag, clean count and
    LOST are the P20 fields, and G1 the P20 view's."""
    import copy
    r = copy.deepcopy(rebuilt)
    for row in r.get("treads_table") or []:
        row.update({"ms_p50": row["ms_p50_p20"], "sm_clock_load_mhz": row["sm_clock_load_mhz_p20"],
                    "host_bound": row["host_bound_p20"], "clean_repeats": row["clean_repeats_p20"],
                    "repeats": row["clean_repeats_p20"], "lost": row["lost_p20"]})
    v = r["g1_p20"]["verdict"]
    r["histogram_gates"]["G1_lock_thermal"] = {"verdict": v, "rule": "P20 view"}
    r["gates"] = [dict(x, verdict=v) if x.get("tag") == "G1_lock_thermal" else x for x in r["gates"]]
    return r


def read_page(page_dir: Path) -> tuple[dict | None, list[dict] | None]:
    import csv
    rp, cc = Path(page_dir) / "report.json", Path(page_dir) / "cells.csv"
    rep = json.loads(rp.read_text()) if rp.exists() else None
    raw = list(csv.DictReader(cc.open(newline=""))) if cc.exists() else None
    return rep, raw


def page_report(tree: Path, model: str, label: str, tview, reg_file: dict | None, view: str = "registered"):
    """The unit's one G = 8 histogram page: (report, why); report None when NOT SCORED or out of
    this view. G5 as rental 6; then the page REBUILT from its cells.csv (rebuild); `view` "p20"
    reads the rebuilt page as the P20 side view (DESCRIPTIVE)."""
    root = C3.find_gaps(tree, model, label)
    pages = [(p, r) for p, r in C3.timed_pages(root) if C3.group_of(r) == 8]
    if len(pages) != 1:
        return None, f"NOT SCORED: {len(pages)} G = 8 pages under gaps-{model}-{label} (no report.json is NOT SCORED)"
    p, r = pages[0]
    h = r.get("histogram") or {}
    if reg_file is None or h.get("file_sha256") != reg_file["sha256"]:
        return None, "NOT SCORED (G5): the page's histogram file sha256 is not the registered one"
    if h.get("shuffle_seed") is None or h.get("shuffle_seed") != reg_file.get("shuffle_seed"):
        return None, "NOT SCORED (G5): the page records no shuffle seed, or not the registered one"
    want = {(c["label"], c["n"]): c["counts_sha256"] for c in reg_file["cells"]}
    for row in r.get("treads_table") or []:
        k = (str(row.get("histogram")), int(row["tiles"]))
        if want.get(k) != row.get("counts_sha256") or not row.get("bincount_ok"):
            return None, f"NOT SCORED (G5): row {k} carries another counts sha256 or no exact bincount"
    if r.get("slip_policy") != "cell" or r.get("min_clean_repeats") != 0:
        return None, ("NOT SCORED: the page did not run --slip-policy cell --min-clean-repeats 0 (R3 may have "
                      "stopped timing a cell the scorer's rule keeps)")
    _rep, raw = read_page(Path(p).parent)
    r, why = rebuild(r, raw)
    if r is None:
        return None, f"NOT SCORED (rebuild): {why}"
    g = r["g1_r7"]
    state = g["verdict"]
    why = f"{g['non_clean_rows']} of {g['repeat_rows']} repeat rows non-clean ({100 * g['share']:.2f}%)"
    if state != "PASS":
        why = f"G1 FAIL (rebuilt): {why}" + (" over the 10% cap" if g["share_verdict"] == "FAIL" else "") + \
            (f"; clocks unread {g['clock_unread']}" if g["clock_unread"] else "") + \
            (f"; below the top by a step {g['below_top_by_a_step']}" if g["below_top_by_a_step"] else "")
    if view == "p20":
        state = r["g1_p20"]["verdict"]
        why = f"P20 view {state}: {r['g1_p20']['non_clean_rows']} of {r['g1_p20']['repeat_rows']} non-clean"
        r = p20_report(r)
    key = f"{model}-{label}"
    r = tview.admit(key, r)
    if r is None:
        tview.pages.setdefault(key, {})["g1_r7"] = why
        return None, f"out of this view (a recorded VALIDITY gate failed; {why})"
    tview.pages[key]["g1_r7"] = why
    if state != "PASS" and tview.name == "CLEAN":
        tview.pages[key]["use"] = "excluded (G1: " + why + ")"
        return None, f"out of CLEAN (G1: {why})"
    return r, ("counted" if state == "PASS" else f"counted in ALL only (G1: {why})")


# --------------------------------------------------------------------------
# Q1-H (rental 6's statistic, three clusters, rivals scored apart from the copies clause)
# --------------------------------------------------------------------------

def copies_clause(rows: list[dict]) -> dict:
    """Per model with valid Q1-page pairs at both c9 and c15: d_m = mean g(c15) - mean g(c9);
    the clause reads the mean of d_m weighted by the model's Q1 pairs. With one pair per side
    per model (every Q1 page valid) this is rental 6's unmatched mean exactly."""
    per, w = {}, {}
    for m in sorted({r["model"] for r in rows if r.get("q1page")}):
        q9 = [r["g"] for r in rows if r["model"] == m and r.get("q1page") and r.get("copies") == 9]
        q15 = [r["g"] for r in rows if r["model"] == m and r.get("q1page") and r.get("copies") == 15]
        if q9 and q15:
            per[m] = st.fmean(q15) - st.fmean(q9)
            w[m] = len(q9) + len(q15)
    if not per:
        return {"verdict": "UNDECIDED: no model has valid Q1 pairs at both c9 and c15", "models": {}}
    d = sum(per[m] * w[m] for m in per) / sum(w.values())
    return {"verdict": "HOLDS" if abs(d) <= C6.GAP_BAND_US else "FAILS", "c15_minus_c9_us": r6(d),
            "models": {m: r6(v) for m, v in per.items()}}


def q1h_rule(rows: list[dict], rivals: tuple = ()) -> dict:
    """Q1-H, rental 7: rows {g, cluster, model, copies, q1page, g_R: {R: x}}."""
    n = len(rows)
    clusters = sorted({r["cluster"] for r in rows})
    if n < MIN_PAIRS or len(clusters) < MIN_CLUSTERS:
        return {"verdict": f"UNDECIDED: {n} valid pairs in {len(clusters)} clusters (needs {MIN_PAIRS} pairs "
                           f"and {MIN_CLUSTERS} clusters)", "pairs": n, "clusters": len(clusters),
                "rivals": {R: {"verdict": "UNDECIDED (too few pairs or clusters)"} for R in rivals}}
    g = [r["g"] for r in rows]
    m, se, C = C6.cluster_mean_se(g, [r["cluster"] for r in rows])
    level = abs(m) <= C6.GAP_BAND_US
    cc = copies_clause(rows)
    if cc["verdict"].startswith("UNDECIDED"):
        verdict = f"UNDECIDED: the level clause {'holds' if level else 'fails'}; the copies clause is undecided"
        if not level:
            verdict = "FAILS (the level clause; the copies clause undecided)"
    else:
        verdict = "HOLDS" if (level and cc["verdict"] == "HOLDS") else "FAILS"
    out = {"verdict": verdict, "pairs": n, "clusters": C, "mean_g_us": r6(m), "se_us": r6(se),
           "level_clause": "HOLDS" if level else "FAILS", "copies_clause": cc,
           "equal_cluster_weight_mean_us": r6(st.fmean(st.fmean(r["g"] for r in rows if r["cluster"] == c) for c in clusters)),
           "rivals": {}}
    for R in rivals:
        gR = st.fmean(r["g_R"][R] for r in rows)
        dd = abs(m - gR)
        out["rivals"][R] = {"g_R_us": r6(gR), "verdict": "EXCLUDED" if (dd > C6.GAP_K_SE * se and dd > C6.GAP_BAND_US) else "NOT EXCLUDED"}
    return out


def q1h_power(pairs: list[dict], sig_us: dict, anchor_us: dict, *, tau_us: float = 0.0, truth: str | None = None,
              c15_effect_us: float = 0.0, rivals=("v2", "H8", "v3_all"), R: int = 200_000, seed: int = 20261013,
              device: str = "cpu") -> dict:
    """Monte Carlo of q1h_rule over `pairs` (cluster keys "model|n<n>"): g = anchor(cluster) +
    (g_R of `truth`) + tau x N(0,1) per cluster + sig(cluster) x N(0,1) per pair + the c15 effect
    on c15 Q1 pairs. Normals drawn on the CPU in float64 and moved to `device` (MPS: float32)."""
    import torch
    dt = torch.float64 if device == "cpu" else torch.float32
    gen = torch.Generator().manual_seed(seed)

    def rn(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float64)
    cls = sorted({p["cluster"] for p in pairs})
    C, N = len(cls), len(pairs)
    ci = torch.tensor([cls.index(p["cluster"]) for p in pairs])
    base = torch.tensor([anchor_us[p["cluster"]] + (p["g_R"][truth] if truth else 0.0) for p in pairs], dtype=torch.float64)
    sig = torch.tensor([sig_us[p["cluster"]] for p in pairs], dtype=torch.float64)
    is15 = torch.tensor([bool(p.get("q1page")) and p["copies"] == 15 for p in pairs])
    is9 = torch.tensor([bool(p.get("q1page")) and p["copies"] == 9 for p in pairs])
    g = base[None] + (rn(R, C) * tau_us)[:, ci] + rn(R, N) * sig[None] + c15_effect_us * is15.to(torch.float64)[None]
    t0 = __import__("time").perf_counter()
    g = g.to(device, dt)
    cid = ci.to(device)
    m = g.mean(1)
    cs = torch.zeros((R, C), device=device, dtype=dt).index_add_(1, cid, g - m[:, None])
    se = torch.sqrt(C / (C - 1) * (cs ** 2).sum(1)) / N
    models = sorted({p["model"] for p in pairs if p.get("q1page")})
    d_num = torch.zeros(R, device=device, dtype=dt)
    wsum = 0
    for mm in models:
        i15 = torch.tensor([p["model"] == mm for p in pairs]) & is15
        i9 = torch.tensor([p["model"] == mm for p in pairs]) & is9
        if i15.any() and i9.any():
            w = int(i15.sum() + i9.sum())
            d_num += w * (g[:, i15.to(device)].mean(1) - g[:, i9.to(device)].mean(1))
            wsum += w
    d = d_num / wsum
    level = m.abs() <= C6.GAP_BAND_US
    hold = level & (d.abs() <= C6.GAP_BAND_US)
    f = lambda x: float(x.to(dt).mean())
    out = {"pairs": N, "clusters": C, "P(v3 HOLDS)": f(hold), "P(level clause)": f(level),
           "P(copies clause)": f(d.abs() <= C6.GAP_BAND_US)}
    for Rv in rivals:
        gR = st.fmean(p["g_R"][Rv] for p in pairs)
        dd = (m - gR).abs()
        out[f"P(EXCLUDED {Rv})"] = f((dd > C6.GAP_K_SE * se) & (dd > C6.GAP_BAND_US))
    if device == "mps":
        torch.mps.synchronize()
    out["seconds"] = __import__("time").perf_counter() - t0
    return out


# --------------------------------------------------------------------------
# E1 per shape, and its power (mc7 block B with the review's per-shape clause)
# --------------------------------------------------------------------------

def e1_bar(v: list[float]) -> bool:
    s = C5.stats(v)
    return bool(s["cells"]) and s["rms"] <= C6.E1_BAR["rms_max"] and abs(s["worst"]) <= C6.E1_BAR["max_abs"] \
        and abs(s["mean"]) <= C6.E1_BAR["mean_max"]


def e1_power(rows: list[dict], mu: dict, sd_cell: float, *, scale: float = 1.0, board: float = 0.0018,
             R: int = 200_000, seed: int = 20261014, device: str = "cpu") -> dict:
    """P(E1-SKEW holds) pooled and per shape, P(LT3 EXCLUDED) on its lever cells, P(SKEW-RATIO
    holds) and P(U EXCLUDED): e = scale x mu(model, n, arm) + board(model) + sd_cell x N(0, 1)."""
    import time

    import torch
    dt = torch.float64 if device == "cpu" else torch.float32
    gen = torch.Generator().manual_seed(seed)
    models = sorted({r["model"] for r in rows})
    N = len(rows)
    mi = torch.tensor([models.index(r["model"]) for r in rows])
    muv = torch.tensor([mu.get(f"{r['model']}|{r['n']}|{r['arm']}", 0.0) for r in rows], dtype=torch.float64)
    eps = torch.randn((R, N), generator=gen, dtype=torch.float64) * sd_cell
    bd = torch.randn((R, len(models)), generator=gen, dtype=torch.float64) * board
    e = scale * muv[None] + bd[:, mi] + eps
    gL = torch.tensor([r["ms"]["LT3"] / r["ms"]["v3"] - 1 for r in rows], dtype=torch.float64)
    lv = torch.tensor([bool(r["lever"].get("LT3")) for r in rows])
    k = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): i for i, r in enumerate(rows)}
    pairs = [(i, k[(r["model"], r["page"], r["n"], "uniform", r["arm"])]) for i, r in enumerate(rows)
             if r["label"] in ("PT", "PW", "PW2", "DW")]
    rU = torch.tensor([rows[i]["ms"]["v3"] / rows[j]["ms"]["v3"] - 1 for i, j in pairs], dtype=torch.float64)
    t0 = time.perf_counter()
    e, gL, lv, mi_d = e.to(device, dt), gL.to(device, dt), lv.to(device), mi.to(device)

    def bar(x):
        rms = x.pow(2).mean(1).sqrt()
        return (rms <= C6.E1_BAR["rms_max"]) & (x.abs().max(1).values <= C6.E1_BAR["max_abs"]) & \
            (x.mean(1).abs() <= C6.E1_BAR["mean_max"])
    pooled = bar(e)
    shape = {m: bar(e[:, mi_d == j]) for j, m in enumerate(models)}
    allshape = pooled.clone()
    for v in shape.values():
        allshape &= v
    eL = (1 + gL)[None] * (1 + e) - 1
    rL, rv = eL[:, lv].pow(2).mean(1).sqrt(), e[:, lv].pow(2).mean(1).sqrt()
    excl = (rL >= C6.RIVAL_FACTOR * rv) & (rL >= C6.RIVAL_FLOOR)
    i = torch.tensor([p for p, _ in pairs], device=device)
    j = torch.tensor([q for _, q in pairs], device=device)
    er = (1 + e[:, i]) / (1 + e[:, j]) - 1
    r_hold = (er.pow(2).mean(1).sqrt() <= 0.015) & (er.abs().max(1).values <= 0.04)
    rUd = rU.to(device, dt)
    lU = (rU.abs() > 0.045).to(device)
    eU = (1 + er) / (1 + rUd)[None] - 1
    rUr, rUv = eU[:, lU].pow(2).mean(1).sqrt(), er[:, lU].pow(2).mean(1).sqrt()
    U_ex = (rUr >= C6.RIVAL_FACTOR * rUv) & (rUr >= C6.RIVAL_FLOOR)
    if device == "mps":
        torch.mps.synchronize()
    f = lambda x: float(x.to(dt).mean())
    out = {"cells": N, "P(E1-SKEW HOLDS: pooled and each shape)": f(allshape), "P(pooled bar)": f(pooled),
           **{f"P({m} bar)": f(v) for m, v in shape.items()}, "E[rms] %": 100 * f(e.pow(2).mean(1).sqrt()),
           "LT3 lever cells": int(lv.sum()), "P(LT3 EXCLUDED)": f(excl),
           "ratios": len(pairs), "P(SKEW-RATIO HOLDS)": f(r_hold), "U lever ratios": int(lU.sum()), "P(U EXCLUDED)": f(U_ex),
           "seconds": time.perf_counter() - t0}
    return out


# --------------------------------------------------------------------------
# l2open
# --------------------------------------------------------------------------

def l2open_rule(rows: list[dict]) -> dict:
    """rows {skewed: bool, e: {rival: e}}: the nearest rival by rms over the skewed rows is NAMED
    if the next is at least L2_FACTOR farther and every skewed row of the nearest is within
    L2_MAX_ABS; else NOT SEPARATED. Uniform and balanced rows are printed, never scored."""
    sk = [r for r in rows if r["skewed"]]
    if not sk:
        return {"verdict": "NOT SCORED: no skewed w1 row"}
    rms = {R: math.sqrt(st.fmean(r["e"][R] ** 2 for r in sk)) for R in L2_RIVALS}
    order = sorted(rms, key=rms.get)
    near, nxt = order[0], order[1]
    worst = max(abs(r["e"][near]) for r in sk)
    named = rms[nxt] >= L2_FACTOR * rms[near] and worst <= L2_MAX_ABS
    ctrl = [r for r in rows if not r["skewed"]]
    return {"verdict": f"NAMED {near}" if named else f"NOT SEPARATED (nearest {near})", "skewed_rows": len(sk),
            "rms_pct": {R: r6(100 * v) for R, v in rms.items()}, "nearest": near, "next": nxt,
            "next_over_nearest": r6(rms[nxt] / rms[near]) if rms[near] else None, "worst_nearest_pct": r6(100 * worst),
            "controls_printed": {R: C5.stats(r["e"][R] for r in ctrl) for R in L2_RIVALS} if ctrl else None}
