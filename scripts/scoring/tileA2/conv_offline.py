#!/usr/bin/env python
"""Offline conversion checks for tile rental A2 (DESIGN-A2 7e / OD12, REVIEW-A2 MINOR 3b), on SEEN
published pages only, run before A2 registered (tileA2-convl's 'offline_check (SEEN)').

    python scripts/scoring/tileA2/conv_offline.py --write    # docs/registered/2026-10-20-tileA2-hist/conv_offline.json
    python scripts/scoring/tileA2/conv_offline.py --check    # recompute, compare byte for byte (exit 1 on a difference)

(a) 7e. The 2026-09-27 GH200 session timed R3's own tile (Mixtral 8x7B, BLOCK_M 32, BN 64, BK 64,
    w8 s4) at held locks 1410 (G 4, G 2), 1500 (G 4) and 1605 (G 4), and the same G 4 / G 2 ladders
    at 1710 (the deep pages). Every cell is priced with v3 at rental 6's constants (reg6.pricer(),
    through wave_split_bytes_v4.price at R3's tile, which is v3 bit for bit) under P1's rule: c
    alone times 1710 / lock (eta 1); T0, bw, s_small, s_block, d carried. Reported: per lock and G
    the level residual pred / meas - 1, the ratio residual (t_L / t_1710 measured over predicted,
    per cell, against the same board's 1710 page), and the G 4 SHARED OLS slope over treads 2..6.
(b) 3b. R1's lock-mode pages (clock_elasticity, 1710 / 1500 / 1410, G 4 and G 1, NATIVE, treads
    1..8): v3's predicted ms at each (tread, lock) under the same c-only rule, fed to
    clock_elasticity.per_tile_slope (R1's own estimator, treads >= 2) and within_tread_slope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "scripts", REPO / "scripts/scoring/rental6", REPO / "scripts/scoring/rental7"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import clock_elasticity as CE  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import r3_timing_model_v4L as V4L  # noqa: E402
import reg6 as R6  # noqa: E402
import r6common as C6  # noqa: E402
import wave_split_bytes_v4 as WV4  # noqa: E402

SESSION = REPO / "results/published/2026-09-27-nvidia_gh200_480gb-session"
PWR = SESSION / "results/gaps-nvidia_gh200_480gb/private_weight_reference"
R1 = SESSION / "results/gaps-nvidia_gh200_480gb/clock_elasticity"
FIT_MHZ = V4L.FIT_CLOCK_MHZ
MODEL = "mixtral-8x7b"
OUT = REPO / "docs" / "registered" / "2026-10-20-tileA2-hist" / "conv_offline.json"
INPUTS: dict = {}


def sha(p: Path) -> str:
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    INPUTS[str(Path(p).relative_to(REPO))] = h
    return h


PR = R6.pricer()
PRM = V4.Params()
CACHE: dict = {}


def price(arm: str, n: int, G: int, lock: float, A_us=None) -> float | None:
    k = (arm, n, G, lock, A_us)
    if k not in CACHE:
        try:
            with V4L.at_lock(lock):     # P1's rule: c x 1710 / lock at R3's tile, everything else carried
                got = WV4.price(PR, MODEL, arm, n, None, 9, None, V4.R3_TILE, G, PRM, key="v3", A_us=A_us)
            CACHE[k] = got["ms"]
        except Exception as exc:  # noqa: BLE001
            CACHE[k] = None
            CACHE[("why",) + k] = f"{type(exc).__name__}: {exc}"
    return CACHE[k]


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"cells": 0}
    return {"cells": len(v), "mean_pct": round(100 * st.fmean(v), 4),
            "rms_pct": round(100 * math.sqrt(st.fmean(x * x for x in v)), 4),
            "worst_pct": round(100 * max(v, key=abs), 4)}


def ols(xs, ys):
    xb, yb = st.fmean(xs), st.fmean(ys)
    return sum((x - xb) * (y - yb) for x, y in zip(xs, ys, strict=True)) / sum((x - xb) ** 2 for x in xs)


def pages_0927() -> dict:
    out = {}
    for rp in sorted(PWR.glob("*/report.json")):
        r = json.loads(rp.read_text())
        tag = r.get("session_tag") or ""
        G = int(r["pinned"]["GROUP_SIZE_M"])
        lock = float(tag.rsplit("-lock", 1)[1]) if "-lock" in tag else None
        kind = tag.split("-")[-2] if tag else ""
        if kind not in ("eta1410", "eta1500", "eta1605", "deep"):
            continue
        sha(rp)
        out[(lock, G)] = {"path": str(rp.relative_to(REPO)), "tag": tag, "report": r}
    return out


def part_a() -> dict:
    pages = pages_0927()
    ref = {G: pages[(1710.0, G)] for G in (4, 2) if (1710.0, G) in pages}
    res = {"pages": {f"{int(lk)} G{G}": v["path"] for (lk, G), v in sorted(pages.items())}, "by_lock": {}}
    for (lock, G), pg in sorted(pages.items()):
        r = pg["report"]
        A = C6.page_A_us(r) or {}
        rows, lev, rat, by_arm = [], [], [], {}
        refrows = {(t["arm"], int(t["tiles"])): t for t in (ref[G]["report"]["treads_table"] if G in ref else [])}
        refA = C6.page_A_us(ref[G]["report"]) or {} if G in ref else {}
        for t in r["treads_table"]:
            arm, n, ms = str(t["arm"]), int(t["tiles"]), t.get("ms_p50")
            if ms is None:
                continue
            a_us = None if arm == "native" else A.get((arm, n))
            pl = price(arm, n, G, lock, a_us)
            row = {"arm": arm, "n": n, "meas_ms": ms, "clock": t.get("sm_clock_load_mhz"), "pred_ms": pl}
            if pl:
                row["e_level"] = pl / ms - 1
                lev.append(row["e_level"])
                by_arm.setdefault(arm, []).append(row["e_level"])
            rr = refrows.get((arm, n))
            if lock != 1710.0 and rr and rr.get("ms_p50") and pl:
                a_ref = None if arm == "native" else refA.get((arm, n))
                p17 = price(arm, n, G, 1710.0, a_ref)
                if p17:
                    row["ratio_meas"] = ms / rr["ms_p50"]
                    row["ratio_pred"] = pl / p17
                    row["e_ratio"] = row["ratio_meas"] / row["ratio_pred"] - 1
                    rat.append((arm, n, row["e_ratio"]))
            rows.append(row)
        key = f"{int(lock)} G{G}"
        d = {"level (pred / meas - 1), all arms": stats(lev),
             "level by arm": {a: stats(v) for a, v in sorted(by_arm.items())},
             "rows": rows}
        if rat:
            d["ratio vs the 1710 deep page (t_L / t_1710 meas over pred - 1), all arms"] = stats([e for *_x, e in rat])
            d["ratio, NATIVE"] = stats([e for a, _n, e in rat if a == "native"])
            d["ratio, NATIVE n 1"] = stats([e for a, n, e in rat if a == "native" and n == 1])
            d["ratio, NATIVE n >= 2"] = stats([e for a, n, e in rat if a == "native" and n >= 2])
        sh = [x for x in rows if x["arm"] == "shared" and 2 <= x["n"] <= 6 and x.get("pred_ms")]
        if len(sh) >= 3:
            sm = ols([x["n"] for x in sh], [x["meas_ms"] for x in sh])
            sp = ols([x["n"] for x in sh], [x["pred_ms"] for x in sh])
            d["SHARED OLS slope treads 2..6 (ms per tread)"] = {"meas": round(sm, 5), "v3 c-only": round(sp, 5),
                                                                "meas / pred - 1 pct": round(100 * (sm / sp - 1), 3)}
        ns = [x for x in rows if x["arm"] in ("native", "shared")]
        d["HEADLINE level, NATIVE and SHARED"] = stats([x.get("e_level") for x in ns])
        if rat:
            d["HEADLINE ratio, NATIVE and SHARED"] = stats([x.get("e_ratio") for x in ns])
        d["PRIVATE note"] = ("PRIVATE is outside v3's registered scope (v3 is registered on NATIVE and SHARED); its "
                             "level misses by 15 to 32% and it is left out of the headline")
        res["by_lock"][key] = d
    res["what_it_tests"] = (
        "G 4 and G 2 pages only, R3's mma.sync tile, one board and day (2026-09-27). The level residual mixes v3's "
        "own level error with the conversion; the ratio against the same board's 1710 deep page cancels the level "
        "and tests the c-only conversion on these cells. These cells are c-heavy (deep G 4 / 2 ladders): the test "
        "says little about whether T0 or bw carry, which dominate short G 1 cells (part b reads that regime).")
    return res


def r1_cells(rep_path: Path) -> dict:
    """{(tread, lock): measured median ms over kept rows} of an R1 page (cells.csv, status ok)."""
    import csv
    by: dict = {}
    with open(rep_path.parent / "cells.csv") as f:
        for x in csv.DictReader(f):
            if x.get("status") != "ok" or x.get("clock_drift_ok") == "False":
                continue
            by.setdefault((int(x["tiles"]), float(x["clock_lock_mhz"])), []).append(float(x["ms_p50"]))
    sha(rep_path.parent / "cells.csv")
    return {k: st.median(v) for k, v in by.items()}


def part_b() -> dict:
    out = {}
    for rp in sorted(R1.glob("*/report.json")):
        sha(rp)
        r = json.loads(rp.read_text())
        G = int(r["pinned"]["GROUP_SIZE_M"])
        locks = [float(x) for x in r["clock_locks_mhz"]]
        meas = r1_cells(rp)
        treads = sorted({t for t, _ in meas})
        pred_all = {(n, lk): price("native", n, G, lk) for n in treads for lk in locks}
        unpriced = sorted({n for (n, lk), v in pred_all.items() if v is None})
        pred = {k: (v, k[1], 1) for k, v in pred_all.items() if k[0] not in unpriced}
        mcells = {(n, lk): (ms, lk, 1) for (n, lk), ms in meas.items() if n not in unpriced}
        mcells_all = {(n, lk): (ms, lk, 1) for (n, lk), ms in meas.items()}
        eb_pred, nt = CE.per_tile_slope(pred, min_tread=CE.CLAIM_MIN_TREAD)
        eb_meas_lockclk, _ = CE.per_tile_slope(mcells, min_tread=CE.CLAIM_MIN_TREAD)
        ft_pred = CE.within_tread_slope(pred)
        lev = {f"n{n} {int(lk)}": round(pred[(n, lk)][0] / meas[(n, lk)] - 1, 5)
               for (n, lk) in sorted(meas) if pred.get((n, lk)) and pred[(n, lk)][0]}
        out[f"G{G}"] = {
            "page": str(rp.relative_to(REPO)), "cells": "NATIVE, balanced, BLOCK_M 32 BN 64 BK 64 w8 s4, treads "
            f"{treads}, locks {locks}",
            "R1 registered eta_b (report.json elasticity.value, read-back clocks)": r["elasticity"]["value"],
            "R1 interval": [r["elasticity"]["lo"], r["elasticity"]["hi"]],
            "R1 eta_b recomputed at the lock clocks over v3's treads (like for like)": None if eb_meas_lockclk is None else -eb_meas_lockclk,
            "treads v3 does not price (ill-posed or refused; left out of both v3's and the like-for-like R1 recompute)": unpriced,
            "R1 eta_b recomputed at the lock clocks over every tread (as R1)": (lambda e: None if e[0] is None else -e[0])(CE.per_tile_slope(mcells_all, min_tread=CE.CLAIM_MIN_TREAD)),
            "v3 c-only eta_b (per_tile_slope, treads >= 2)": None if eb_pred is None else -eb_pred,
            "v3 c-only pooled fixed-tread eta": None if ft_pred[0] is None else -ft_pred[0],
            "R1 pooled fixed-tread eta (report)": r["elasticity"]["fixed_tread"],
            "v3 per-tread eta": {str(k): round(-v, 5) for k, v in ft_pred[1].items()},
            "R1 per-tread eta (report)": r["elasticity"]["per_tread"],
            "v3 level residual pred / meas - 1 per (tread, lock)": lev,
            "v3 predicted ms": {f"n{n} {int(lk)}": v[0] for (n, lk), v in sorted(pred.items())},
        }
        if eb_pred is not None:
            out[f"G{G}"]["v3 minus R1 (eta_b)"] = round(-eb_pred - r["elasticity"]["value"], 5)
    return out


def headline(doc: dict) -> dict:
    a = {k: {"ratio (NATIVE and SHARED, against the 1710 deep page)": v.get("HEADLINE ratio, NATIVE and SHARED"),
             "level (NATIVE and SHARED)": v.get("HEADLINE level, NATIVE and SHARED"),
             "SHARED slope, treads 2..6": v.get("SHARED OLS slope treads 2..6 (ms per tread)")}
         for k, v in doc["a_7e"]["by_lock"].items() if not k.startswith("1710")}
    b = {}
    for g, v in doc["b_3b"].items():
        b[g] = {"v3 c-only eta_b": v.get("v3 c-only eta_b (per_tile_slope, treads >= 2)"),
                "R1 registered eta_b": v.get("R1 registered eta_b (report.json elasticity.value, read-back clocks)"),
                "R1 like for like (v3's treads)": v.get("R1 eta_b recomputed at the lock clocks over v3's treads (like for like)"),
                "v3 minus R1": v.get("v3 minus R1 (eta_b)"),
                "v3 pooled fixed-tread eta": v.get("v3 c-only pooled fixed-tread eta"),
                "R1 pooled fixed-tread eta": v.get("R1 pooled fixed-tread eta (report)")}
    return {"7e": a, "3b": b}


def build() -> str:
    doc = {"what": __doc__, "fit_clock_mhz": FIT_MHZ,
           "constants": {"x (T0, c_ns, bw, s_small, s_block)": [float(v) for v in PR.f2.x],
                         "source": "scripts/scoring/rental6/reg6.py pricer() (rental 6's CAL 8x7B v2 fit)"},
           "rule": "P1: c alone x (1710 / lock)^1 (in memory, r3_timing_model_v4.c_scale wrapped); everything else carried",
           "a_7e": part_a(), "b_3b": part_b()}
    for k in ("scripts/r3_timing_model.py", "scripts/r3_timing_model_v3.py", "scripts/r3_timing_model_v4.py",
              "scripts/wave_split_bytes_v4.py", "scripts/scoring/rental6/reg6.py", "scripts/clock_elasticity.py"):
        sha(REPO / k)
    doc["price_errors"] = {str(k): v for k, v in CACHE.items() if k and k[0] == "why"}
    doc["inputs_sha256"] = dict(sorted(INPUTS.items()))
    doc["headline"] = headline(doc)
    return json.dumps(doc, indent=1, default=str) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = build()
    if a.write:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(REPO)}")
        return 0
    same = OUT.exists() and OUT.read_text() == text
    print(f"{OUT.name}: {'same' if same else 'DIFFERS'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
