"""Rental 6's shared rules (docs/registered/2026-10-09-rental6-*).

One module for the registration generator (register.py, reg6.py) and every rental-6 scorer, so a
rule is computed one way on both sides. Pure functions over numbers, pages and the registered
JSON; nothing here fits a model at score time. The ALL / CLEAN views and the R3 page readers are
rental 3's (r3common), page_c rental 4's (r4common), the G3 TOST, the calibration line and the
stats rental 5's (r5common), imported and never copied.

THE RULES THIS FILE OWNS (each stated in full in the registrations; the 17 scorer ambiguities
of design-r6-review (h) are resolved there, item by item, and implemented here):
  clean repeat   a repeat that is neither lock-slipped (R3's per-repeat `lock_slip`, repeat clock
                 under lock - 15 MHz) nor drift-excluded (R3's `excluded`). A cell is VALID with at
                 least MIN_CLEAN (6) clean repeats of REPEATS (9).
  G1 (scorer)    R3's own G1_lock_thermal PASS (as rental 5) AND slipped repeat rows at most
                 MAX_SLIP_FRAC (10%) of the page's repeat rows. A page failing it counts in ALL
                 and is dropped in CLEAN. A page with no report.json is NOT SCORED (cells.csv is
                 never read).
  G2 (host)      rental 3's registered host rule: a cell is scored only when its REGISTERED v3
                 prediction + F240 (0.067551 ms) is at least 0.40 ms (fixed before any page, so
                 a fast board cannot void a cell); the page's own timer flag `host_bound` also
                 drops a cell. A gap pair needs both cells.
  G4 (clock)     the cell's clock (R3's sm_clock_load_mhz, the median over its clean repeats)
                 at least 1705 MHz; a cell under it is dropped (the cell, never the page).
  clock rule     counter pages (byte legs, second K): lock_gate.check_page PASS, then each cell's
                 SM clock (sm__cycles_elapsed.avg / gpu_time_ns, the median over its GEMMs of at
                 least 0.5 ms, else the page's median) in [1640, 1710] MHz; a cell outside is
                 dropped and the page continues. Derived from the smi-held pages. Its
                 under-lock negative control is the published floor1005 and floor1410 pages
                 (check_page PASS at their own lock, 0 cells in the band); over the lock, the
                 8x22B floor. It cannot see a sag under about 4%.
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


C5 = _load("rental5_r5common", HERE.parent / "rental5" / "r5common.py")
C3, C4 = C5.C3, C5.C4
CARD = C3.CARD
DATE = "2026-10-09"
PARTS = ("v3", "skew", "q1", "secondk")
NAMES = {p: f"{DATE}-rental6-{p}-gh200" for p in PARTS}
HIST_DIR = f"docs/registered/{DATE}-rental6-skew-hist"
r6 = C5.r6
stats = C5.stats

LOCK_MHZ = 1710.0
LOCK_STEP_MHZ = 15.0
REPEATS = 9
MIN_CLEAN = 6
MAX_SLIP_FRAC = 0.10
G4_MIN_MHZ = 1705.0
BAND_MHZ = (1640.0, 1710.0)
MIN_CLOCK_NS = 500_000
F240_MS = 0.067551
HOST_CUT_MS = 0.40
SIGMA_PAGE = C5.SIGMA_PAGE            # 0.18%: the lever threshold is 3 sigma = 0.54%
LEVER_K = 3.0
#: E1-SKEW at the time bar (owner decision 3)
E1_BAR = {"rms_max": 0.02, "max_abs": 0.05, "mean_max": 0.01}
#: rental 5's bands, printed beside (rental5-skew E1-SKEW)
E1_R5_BANDS = {"rms_max": 0.012, "max_abs": 0.03, "mean_max": 0.006}
RIVAL_FACTOR, RIVAL_FLOOR, MIN_LEVER = 2.0, 0.0076, 6
GAP_BAND_US = 1.5
GAP_K_SE = 3.0
MIN_PAIRS = 6
B_SKEW = {"max_abs": 0.05, "rms_max": 0.03}
B_CA = {"max_abs": 0.03, "noca_factor": 2.0, "noca_floor": 0.03}
E2_BAR = {"rms_max": 0.02, "max_abs": 0.05, "mean_max": 0.015}
SECONDK_TOL = 0.03
SECONDK_CLOCK_AGREE = 0.01
SECONDK_NS = (4, 5, 6, 7, 8, 9)


def registration(repo: Path, part: str) -> dict:
    return json.loads((Path(repo) / "docs" / "registered" / f"{NAMES[part]}.json").read_text())


# --------------------------------------------------------------------------
# G2: the host rule (rental 3's, on the registered prediction)
# --------------------------------------------------------------------------

def host_ok(pred_ms: float | None) -> bool:
    return pred_ms is not None and pred_ms + F240_MS >= HOST_CUT_MS


# --------------------------------------------------------------------------
# slipped repeats, G1, G4 (timed pages)
# --------------------------------------------------------------------------

def clean_repeats(row: dict) -> int:
    """A row's clean repeats: R3's `clean_repeats` when the page ran with --lock-mhz (rental 6);
    a row without it has no slip flag, and its `repeats` (R3's usable, not drift-excluded,
    repeats) are its clean ones."""
    if row.get("clean_repeats") is not None:
        return int(row["clean_repeats"])
    return int(row.get("repeats") or 0)


def row_verdict(row: dict, pred_ms: float | None) -> str | None:
    """None when the row is a valid cell, else why it is dropped (in rule order)."""
    if row.get("ms_p50") is None:
        return "no time"
    if clean_repeats(row) < MIN_CLEAN:
        return f"{clean_repeats(row)} clean repeats, under {MIN_CLEAN}"
    if not host_ok(pred_ms):
        return "G2 host-bound by rule (v3 + F240 under 0.40 ms)"
    if row.get("host_bound"):
        return "G2 host-bound (the page's timer flag)"
    clk = row.get("sm_clock_load_mhz")
    if clk is None or float(clk) < G4_MIN_MHZ:
        return f"G4 clock {clk}"
    return None


def page_slips(report: dict) -> dict:
    """The page's slipped repeat rows against its repeat rows (R3's `repeat_rows` and
    `lock_slip_rows`, rental 6), and every slipped repeat with its clock."""
    rows = report.get("treads_table") or []
    total = report.get("repeat_rows")
    if total is None:
        total = sum(int(r.get("repeat_rows") or r.get("repeats") or 0) for r in rows)
    slipped = report.get("lock_slip_rows")
    if slipped is None:
        slipped = sum(int(r.get("lock_slip_repeats") or 0) for r in rows)
    clocks = [{"cell": f"{x.get('histogram')}/{x.get('arm')}/n{x.get('n')}", "repeat": x.get("repeat"),
               "mhz": x.get("mhz")} for x in report.get("slipped_repeats") or []]
    return {"repeat_rows": int(total), "slipped_rows": int(slipped),
            "fraction": (slipped / total) if total else 0.0, "slipped": clocks,
            "lost_cells": report.get("lost_cells") or {}, "early_stop": report.get("early_stop")}


def page_g1(report: dict) -> tuple[bool, str]:
    """The scorer's G1: R3's G1 PASS and slipped rows at most MAX_SLIP_FRAC of the rows."""
    g1 = (report.get("histogram_gates") or {}).get("G1_lock_thermal", {}).get("verdict")
    sl = page_slips(report)
    if g1 != "PASS":
        return False, f"R3 G1 {g1}"
    if sl["fraction"] > MAX_SLIP_FRAC + 1e-12:
        return False, f"{sl['slipped_rows']} of {sl['repeat_rows']} repeat rows slipped (over {MAX_SLIP_FRAC:.0%})"
    return True, f"PASS ({sl['slipped_rows']} of {sl['repeat_rows']} repeat rows slipped)"


# --------------------------------------------------------------------------
# the counter-page clock rule (S4)
# --------------------------------------------------------------------------

def _gemm_mhz(page_kind: str, cell: dict, g: str):
    v = (cell.get("per_gemm") or {}).get(g) or {}
    if page_kind == "floor":
        cyc, ns = v.get("sm__cycles_elapsed.avg"), v.get("gpu__time_duration.sum")
    else:
        cyc = ((cell.get("recorded") or {}).get(g) or {}).get("sm__cycles_elapsed.avg")
        ns = v.get("gpu_time_ns")
    if not cyc or not ns:
        return None, ns
    return 1e3 * float(cyc) / float(ns), float(ns)


def cell_clocks(page: dict) -> list[dict]:
    """Per cell of a counter page: its SM clock (the median over its GEMMs of at least
    MIN_CLOCK_NS of cycles / ns), or the page's median of those when it has none (`own` False)."""
    import lock_gate as LG
    kind = LG.page_kind(page) or "page"
    per = []
    for i, c in enumerate(page.get("cells") or []):
        xs = [m for g in sorted((c.get("per_gemm") or {})) for m, ns in [_gemm_mhz(kind, c, g)]
              if m and ns and ns >= MIN_CLOCK_NS]
        per.append((i, c, xs))
    allx = [x for _, _, xs in per for x in xs]
    pm = st.median(allx) if allx else None
    out = []
    for i, c, xs in per:
        out.append({"i": i, "arm": c.get("arm"), "n": c.get("n"), "histogram": c.get("histogram"),
                    "mhz": st.median(xs) if xs else pm, "own": bool(xs)})
    return out


def clock_rule(page: dict, path: Path | None = None) -> dict:
    """The counter-page clock rule: check_page PASS, then each cell in BAND_MHZ."""
    import lock_gate as LG
    chk = LG.check_page(page, path)
    verdict = None if chk is None else chk.verdict
    cells = cell_clocks(page)
    for c in cells:
        c["ok"] = c["mhz"] is not None and BAND_MHZ[0] <= c["mhz"] <= BAND_MHZ[1]
    ok_page = verdict == LG.PASS
    return {"check_page": verdict, "page_ok": ok_page,
            "cells_ok": sum(c["ok"] for c in cells), "cells": cells,
            "why": ("PASS" if ok_page else f"lock_gate.check_page {verdict}")}


# --------------------------------------------------------------------------
# the S - N gap statistics (Q1-H, Q1-P, Q1-SK, Q1-C)
# --------------------------------------------------------------------------

def gap_g_us(meas_s: float, meas_n: float, pred_s: float, pred_n: float) -> float:
    """g = (T_S - T_N)_meas - (T_S - T_N)_pred, us."""
    return 1e3 * ((meas_s - meas_n) - (pred_s - pred_n))


def cluster_mean_se(g: list[float], clusters: list) -> tuple[float, float, int]:
    """(mean over pairs, cluster-robust se on the given clusters, clusters). The se is the
    CR1 sandwich: se^2 = C / (C - 1) x sum_c (sum_{i in c} (g_i - mean))^2 / N^2; with one
    cluster it is the iid sd / sqrt(N) (ddof 1), flagged by C = 1."""
    N = len(g)
    m = st.fmean(g)
    cl = sorted(set(clusters))
    C = len(cl)
    if N < 2:
        return m, float("nan"), C
    if C < 2:
        return m, st.stdev(g) / math.sqrt(N), C
    s = sum(sum(gi - m for gi, ci in zip(g, clusters, strict=True) if ci == c) ** 2 for c in cl)
    return m, math.sqrt(C / (C - 1) * s) / N, C


def q1_rule(rows: list[dict], rivals: tuple = ()) -> dict:
    """Q1-H (and Q1-P printed): rows {g, cluster, copies, g_R: {R: x}}. v3 HOLDS iff
    |mean g| <= 1.5 us and |mean g(c15) - mean g(c9)| <= 1.5 us over the Q1 pages' pairs
    (copies 9 and 15; both sides present). A rival R is EXCLUDED iff |mean g - g_R| > 3 se
    and > 1.5 us (g_R the mean of the rival's per-pair gap residual over the same pairs).
    UNDECIDED below 6 pairs."""
    n = len(rows)
    if n < MIN_PAIRS:
        return {"verdict": f"UNDECIDED: {n} valid pairs, under {MIN_PAIRS}", "pairs": n}
    g = [r["g"] for r in rows]
    m, se, C = cluster_mean_se(g, [r["cluster"] for r in rows])
    q9 = [r["g"] for r in rows if r.get("copies") == 9 and r.get("q1page")]
    q15 = [r["g"] for r in rows if r.get("copies") == 15 and r.get("q1page")]
    if not q9 or not q15:
        return {"verdict": "UNDECIDED: the copies clause needs Q1 pairs at both c9 and c15",
                "pairs": n, "mean_g_us": r6(m), "se_us": r6(se)}
    d = st.fmean(q15) - st.fmean(q9)
    holds = abs(m) <= GAP_BAND_US and abs(d) <= GAP_BAND_US
    out = {"verdict": "HOLDS" if holds else "FAILS", "pairs": n, "clusters": C,
           "mean_g_us": r6(m), "se_us": r6(se), "c15_minus_c9_us": r6(d),
           "c9_pairs": len(q9), "c15_pairs": len(q15), "rivals": {}}
    for R in rivals:
        gR = st.fmean(r["g_R"][R] for r in rows)
        dd = abs(m - gR)
        out["rivals"][R] = {"g_R_us": r6(gR), "verdict": "EXCLUDED" if (dd > GAP_K_SE * se and dd > GAP_BAND_US) else "NOT EXCLUDED"}
    return out


def q1c_rule(x9: float | None, x15: float | None, ratio: float, se_D: float) -> dict:
    """Q1-C on D = x15 - x9 (owner, review (e)): CALL predicts D = 0, DEAD predicts
    D = (ratio - 1) x9 (ratio the c15 / c9 dead-CTA count of the cell), both functions of the
    measured x9, so a board shift common to both pages cancels. The nearer is SELECTED when the
    other is more than 2 se_D farther from D (se_D registered); else NOT SEPARATED."""
    if x9 is None or x15 is None:
        return {"verdict": "NOT SCORED: the c9 or c15 pair is missing or invalid"}
    D = x15 - x9
    pos = {"CALL": 0.0, "DEAD": (ratio - 1.0) * x9}
    dist = {k: abs(D - v) for k, v in pos.items()}
    near = min(dist, key=dist.get)
    far = max(dist, key=dist.get)
    sel = dist[far] - dist[near] > 2 * se_D
    return {"verdict": f"SELECTED {near}" if sel else f"NOT SEPARATED (nearest {near})", "D_us": r6(D),
            "x9_us": r6(x9), "x15_us": r6(x15), "positions_us": {k: r6(v) for k, v in pos.items()},
            "zero_level": "ZERO (x9 = 0) printed only; SEEN x9 already reads -7.95 / -5.97 us"}


# --------------------------------------------------------------------------
# power (registered on the CPU in float64; the build checks MPS float32 against it)
# --------------------------------------------------------------------------

def q1_power(pairs: list[dict], sig_us: dict, *, rivals=("v2", "v3_all", "H8"), R: int = 200_000,
             seed: int = 20261010, device: str = "cpu", tau_us: float = 0.0,
             anchor_us: dict | None = None, truth: str | None = None) -> dict:
    """Monte Carlo of the FULL registered Q1 rule (q1_rule: level clause, copies clause,
    cluster-robust se, rival exclusion) over the registered pairs. Each pair's g is
    N(mu, sig_us[model]) with mu = 0 (v3 true), or the rival `truth`'s g_R per pair, plus
    a per-(model, n) shift N(0, tau_us) or the SEEN anchor `anchor_us[(model, n)]`.
    The normals are drawn on the CPU in float64 and moved to `device`."""
    import torch
    dt = torch.float64 if device == "cpu" else torch.float32
    gen = torch.Generator().manual_seed(seed)

    def rn(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float64).to(device, dt)
    n = len(pairs)
    keys = sorted({(p["model"], p["n"]) for p in pairs})
    ci = torch.tensor([keys.index((p["model"], p["n"])) for p in pairs], device=device)
    sig = torch.tensor([sig_us[p["model"]] for p in pairs], device=device, dtype=dt)
    mu = torch.tensor([0.0 if truth is None else p["g_R"][truth] for p in pairs], device=device, dtype=dt)
    g = mu + sig * rn(R, n)
    if anchor_us is not None:
        g = g + torch.tensor([anchor_us.get(k, 0.0) for k in keys], device=device, dtype=dt)[ci]
    elif tau_us:
        g = g + (tau_us * rn(R, len(keys)))[:, ci]
    m = g.mean(1)
    onehot = torch.zeros(n, len(keys), device=device, dtype=dt)
    onehot[torch.arange(n), ci] = 1.0
    C = len(keys)
    resid = (g - m[:, None]) @ onehot                       # R x C: per-cluster sums
    se = (C / (C - 1) * (resid ** 2).sum(1)).sqrt() / n if C > 1 else g.std(1) / math.sqrt(n)
    q9 = torch.tensor([1.0 if (p.get("q1page") and p["copies"] == 9) else 0.0 for p in pairs], device=device, dtype=dt)
    q15 = torch.tensor([1.0 if (p.get("q1page") and p["copies"] == 15) else 0.0 for p in pairs], device=device, dtype=dt)
    d = (g * q15).sum(1) / q15.sum() - (g * q9).sum(1) / q9.sum()
    hold = (m.abs() <= GAP_BAND_US) & (d.abs() <= GAP_BAND_US)
    out = {"pairs": n, "clusters": C, "P(v3 HOLDS)": float(hold.to(dt).mean()),
           "P(level clause)": float((m.abs() <= GAP_BAND_US).to(dt).mean()),
           "P(copies clause)": float((d.abs() <= GAP_BAND_US).to(dt).mean()),
           "E[mean g] us": float(m.mean()), "sd[mean g] us": float(m.std())}
    for Rv in rivals:
        gR = st.fmean(p["g_R"][Rv] for p in pairs)
        ex = ((m - gR).abs() > GAP_K_SE * se) & ((m - gR).abs() > GAP_BAND_US)
        out[f"P(EXCLUDED {Rv})"] = float(ex.to(dt).mean())
        out[f"g_R {Rv} us"] = gR
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def q1c_power(x9_true: float, truth: str, ratio: float, *, sig_x: float, board: float, se_D: float,
              R: int = 200_000, seed: int = 20261011, device: str = "cpu") -> dict:
    """P(correct) and P(SELECTED and correct) of q1c_rule: x9 = x9_true + board + e9, x15 =
    (x9_true or ratio x9_true) + board + e15, e ~ N(0, sig_x), board ~ N(0, board) shared."""
    import torch
    dt = torch.float64 if device == "cpu" else torch.float32
    gen = torch.Generator().manual_seed(seed)

    def rn(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float64).to(device, dt)
    b = board * rn(R)
    x9 = x9_true + b + sig_x * rn(R)
    x15 = (x9_true if truth == "CALL" else ratio * x9_true) + b + sig_x * rn(R)
    D = x15 - x9
    dc, dd = D.abs(), (D - (ratio - 1) * x9).abs()
    pick_dead = dd < dc
    correct = pick_dead if truth == "DEAD" else ~pick_dead
    sel = (dc - dd).abs() > 2 * se_D
    return {"P(correct)": round(float(correct.to(dt).mean()), 4),
            "P(SELECTED and correct)": round(float((correct & sel).to(dt).mean()), 4)}


# --------------------------------------------------------------------------
# timed pages: the page's own A, provenance (G5), the scorer's G1, valid cells
# --------------------------------------------------------------------------

def page_A_us(report: dict) -> dict | None:
    """{(arm, n): A_us} off the page's own align_probe: per (arm, n) the median over the
    probe's repeats, arm minus NATIVE. None when the page carries no probe."""
    ap = report.get("align_probe")
    if not ap or not ap.get("cells"):
        return None
    by: dict = {}
    for c in ap["cells"]:
        by.setdefault((str(c["label"]), int(c["tread"])), []).append(float(c["ms"]))
    med = {k: st.median(v) for k, v in by.items()}
    return {(a, n): 1e3 * (v - med[("native", n)]) for (a, n), v in med.items() if ("native", n) in med}


def page_report(tree: Path, model: str, label: str, tview, reg_file: dict | None):
    """The unit's one G = 8 histogram page: (report, why); report None when NOT SCORED or out
    of this view. G5 provenance as rental 5; R3's recorded VALIDITY gates by the view; the
    scorer's G1 (slips over 10%) drops the page in CLEAN and keeps it in ALL."""
    root = C3.find_gaps(tree, model, label)
    pages = [(p, r) for p, r in C3.timed_pages(root) if C3.group_of(r) == 8]
    if len(pages) != 1:
        return None, f"NOT SCORED: {len(pages)} G = 8 pages under gaps-{model}-{label} (no report.json is NOT SCORED)"
    _p, r = pages[0]
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
    key = f"{model}-{label}"
    g1, why = page_g1(r)
    r = tview.admit(key, r)
    if r is None:
        return None, "NOT SCORED: the page is out of this view (a recorded VALIDITY gate failed)"
    tview.pages[key]["g1_slips"] = why
    if not g1 and tview.name == "CLEAN":
        tview.pages[key]["use"] = "excluded (G1: " + why + ")"
        return None, f"out of CLEAN (G1: {why})"
    return r, ("counted" if g1 else f"counted in ALL only (G1: {why})")


def valid_cells(report: dict, preds: dict, model: str, page: str) -> tuple[dict, list, dict]:
    """({(model, page, n, label, arm): ms}, dropped, A) of a timed page's valid rows."""
    out, dropped = {}, []
    A = page_A_us(report)
    for row in report.get("treads_table") or []:
        k = (model, page, int(row["tiles"]), str(row["histogram"]), str(row["arm"]))
        pr = preds.get(k)
        why = row_verdict(row, None if pr is None else pr["ms"]["v3"])
        if pr is None:
            why = "no registered prediction"
        if why:
            dropped.append(["/".join(map(str, k)), why])
        else:
            out[k] = float(row["ms_p50"])
    return out, dropped, A


def carried(pr: dict, variant: str, A: dict | None) -> float:
    """The registered prediction of `variant` carried to the page's own A (r6model.reprice);
    the registered (proxy) number when the page has no probe."""
    import r6model as M6
    a_page = None if A is None else A.get((pr["arm"], pr["n"]))
    return M6.reprice(pr["ms"][variant], pr["A_proxy_us"], a_page, pr["arm"], pr["native_path"], variant)
