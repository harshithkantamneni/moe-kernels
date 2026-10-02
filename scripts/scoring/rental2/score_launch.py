#!/usr/bin/env python
"""Score rental 2's part 2, the launch-floor rerun
(docs/registered/2026-10-01-rental2-launch2-gh200.json): P0 to P8.

    python scripts/scoring/rental2/score_launch.py <repo> <tree> <out>

<tree> holds one directory per model (`.../launch-floor-r2/<model>/`, found by
name under the tree), each with the timed process's cells.csv, hostprobe.csv,
hostprobe-post.csv and manifest.json and the trace process's traces/ and
manifest-trace.json. Writes <out>/launch.score.{json,txt}; exits 0 whatever
the verdicts. Every term is the registration's `definitions`; nothing here is
fitted.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as CM  # noqa: E402

PART = "launch2"
ARMS = ("native", "shared", "private")
MODES_EAGER = ("E240", "E0", "E480")


#: the plan's launchfloor units write `<date>-<card>-launch-floor-r2/<model>`
LABEL = "launch-floor-r2"


def find_model_dir(tree: Path, model: str) -> Path | None:
    hits = sorted(p.parent for p in Path(tree).rglob(f"{model}/cells.csv")
                  if p.parent.parent.name.endswith(LABEL))
    return hits[0] if len(hits) == 1 else None


def load_cells(d: Path) -> dict:
    agg = defaultdict(list)
    for r in csv.DictReader(open(d / "cells.csv")):
        if r.get("status") != "ok":
            continue
        agg[(r["mode"], r["arm"], int(r["tiles"]))].append(r)
    out = {}
    for k, rs in agg.items():
        out[k] = {"I": st.median(float(r["ms_p50"]) for r in rs),
                  "H": st.median(float(r["host_enqueue_ms"]) / float(r["calls_per_burst"]) for r in rs),
                  "hb": sum(str(r["host_bound"]) == "True" for r in rs) / len(rs), "reps": len(rs)}
    return out


def load_probe(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        out[(r["mode"], r["arm"], int(r["tiles"]))] = r
    return out


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score_model(reg: dict, model: str, d: Path) -> dict:
    C = load_cells(d)
    pre, post = load_probe(d / "hostprobe.csv"), load_probe(d / "hostprobe-post.csv")
    man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    parsed_p = d / "traces" / "parsed.json"
    parsed = json.loads(parsed_p.read_text()) if parsed_p.exists() else {}
    creg = reg["C_reg_ms"][model]
    F = reg["F_ms"]
    res = {"P": {}}

    def Hpre(k):
        r = pre.get(k)
        return fnum(r.get("H_pre_ms")) if r and r.get("status") == "ok" else None

    def C_of(arm, n):
        return float(creg[arm][str(n)])

    def edge(mode, arm, n):
        c, h = C.get((mode, arm, n)), Hpre((mode, arm, n))
        if c is None:
            return True
        if 0.2 <= c["hb"] <= 0.8:
            return True
        return h is not None and abs(C_of(arm, n) + F[mode] - h) <= 0.020

    def rule_hb(mode, arm, n):
        h = Hpre((mode, arm, n))
        return None if h is None else C_of(arm, n) + F[mode] < h
    # ---- P0
    why = []
    if man.get("phase") != "timed":
        why.append(f"manifest phase {man.get('phase')!r}, not 'timed'")
    if man.get("profiler_enabled_ever"):
        why.append("the profiler was on in the timed process")
    if not man.get("profiler_checks"):
        why.append("the timed process made no profiler check")
    drift = [abs(fnum(post[k]["H_pre_ms"]) / fnum(pre[k]["H_pre_ms"]) - 1)
             for k in pre if k in post and fnum(pre[k].get("H_pre_ms")) and fnum(post[k].get("H_pre_ms"))]
    med_drift = st.median(drift) if drift else None
    if med_drift is None or med_drift > 0.05:
        why.append(f"median post/pre drift {med_drift}")
    ratios = [(k, C[k]["H"] / Hpre(k)) for k in C if Hpre(k)]
    out_r = [k for k, v in ratios if not 0.93 <= v <= 1.10]
    if not ratios or len(out_r) > 0.05 * len(ratios):
        why.append(f"H_cell/H_pre outside [0.93, 1.10] on {len(out_r)} of {len(ratios)} cells")
    refused = [k for k, r in pre.items() if r.get("status") != "ok"]
    if not pre or len(refused) > 0.05 * len(pre):
        why.append(f"{len(refused)} of {len(pre)} probe rows refused")
    p0_ok = not why
    res["P"]["P0"] = {"verdict": "HELD" if p0_ok else "FAILED", "why": why, "median_drift": med_drift,
                      "ratio_outside": [list(k) for k in out_r]}
    drift_mark = "" if p0_ok else " (host drift)"
    # ---- P1
    n_max = 9 if model == "mixtral-8x7b-tp8" else 5
    lo, hi = reg["P1"]["band"]
    rows, outside, short = [], 0, []
    for arm in ARMS:
        for n in range(1, n_max + 1):
            c = C.get(("GR", arm, n))
            if c is None:
                outside += 1
                continue
            ok = lo * C_of(arm, n) <= c["I"] <= hi * C_of(arm, n)
            outside += not ok
            rows.append({"arm": arm, "n": n, "GR": c["I"], "ratio": c["I"] / C_of(arm, n), "inside": ok})
        g1, gN = C.get(("GR", arm, 1)), C.get(("GR", arm, n_max))
        if g1 and gN:
            inc = gN["I"] - g1["I"]
            need = (0.10 if model != "mixtral-8x7b-tp8" else 0.8 * (C_of(arm, 9) - C_of(arm, 1)))
            if (model == "mixtral-8x7b-tp8" or arm != "native") and inc < need:
                short.append((arm, round(inc, 4), round(need, 4)))
    res["P"]["P1"] = {"verdict": "FALSIFIED" if outside > 1 or short else "HELD", "outside": outside,
                      "short_increments": short, "cells": rows}
    # ---- P2
    rows, bad = [], []
    for arm in ARMS:
        for n in range(1, 10):
            e, g, h = C.get(("E240", arm, n)), C.get(("GR", arm, n)), Hpre(("E240", arm, n))
            if e is None or g is None or h is None:
                continue
            if not (C_of(arm, n) + F["E240"] > h + 0.020 and e["hb"] == 0):
                continue
            dlt = e["I"] - g["I"]
            ok = 0 <= dlt <= max(0.02 * e["I"], 0.009)
            rows.append({"arm": arm, "n": n, "E240_minus_GR_ms": dlt, "inside": ok})
            if not ok:
                bad.append((arm, n))
    res["P"]["P2"] = {"verdict": ("FALSIFIED" if len(bad) > 1 else "HELD") + drift_mark, "cells": rows,
                      "outside": bad}
    # ---- P3
    rows, bad = [], []
    for arm in ARMS:
        for n in (1, 2):
            e0, e2 = C.get(("E0", arm, n)), C.get(("E240", arm, n))
            keys = [f"TR-E-E240-{arm}-n{n}.json.gz", f"TR-G-E0-{arm}-n{n}.json.gz",
                    f"TR-G-E240-{arm}-n{n}.json.gz"]
            if e0 is None or e2 is None or not all(k in parsed for k in keys):
                continue
            if not (e0["hb"] >= 0.8 and e2["hb"] >= 0.8) or edge("E0", arm, n) or edge("E240", arm, n):
                continue
            hf = fnum(pre[("E240", arm, n)].get("h_flush_ms"))
            Ftr = parsed[keys[0]]["median"]["flush_us"] / 1e3
            dD = min(0.0, (parsed[keys[1]]["median"]["kernel_sum_us"]
                           - parsed[keys[2]]["median"]["kernel_sum_us"]) / 1e3)
            lo3, hi3 = Ftr - hf + dD - 0.012, Ftr - hf + 0.012
            m = e0["I"] - e2["I"]
            ok = lo3 <= m <= hi3
            rows.append({"arm": arm, "n": n, "E0_minus_E240_ms": m, "band": [lo3, hi3],
                         "h_flush_probe_ms": hf, "F_tr_ms": Ftr, "dD_ms": dD, "inside": ok})
            if not ok:
                bad.append((arm, n))
    # P3 is decided over both models' cells together (score() pools them)
    res["P"]["P3"] = {"verdict": "pooled across models (see P3_pooled)", "cells": rows,
                      "outside": bad}
    # ---- P4
    pairs = [(f"TR-E-E480-{a}-n{n}.json.gz", f"TR-E-E240-{a}-n{n}.json.gz") for a in ARMS for n in (1, 2)]
    dF = [parsed[x]["median"]["flush_us"] - parsed[y]["median"]["flush_us"] for x, y in pairs
          if x in parsed and y in parsed]
    dF_ms = st.median(dF) / 1e3 if dF else F["E480"] - F["E240"]
    shift_bad, wrong, rows = [], [], []
    for arm in ARMS:
        for n in range(1, 10):
            e4, e2 = C.get(("E480", arm, n)), C.get(("E240", arm, n))
            if e4 is None or e2 is None:
                continue
            if not edge("E480", arm, n):
                rule = rule_hb("E480", arm, n)
                meas = e4["hb"] >= 0.8
                if rule is not None and rule != meas:
                    wrong.append((arm, n))
            if (not edge("E480", arm, n) and not edge("E240", arm, n)
                    and e4["hb"] >= 0.8 and e2["hb"] >= 0.8):
                sh = (e4["I"] - e2["I"]) - (e4["H"] - e2["H"])
                ok = abs(sh + dF_ms) <= 0.012
                rows.append({"arm": arm, "n": n, "shift_ms": sh, "inside": ok})
                if not ok:
                    shift_bad.append((arm, n))
    res["P"]["P4"] = {"verdict": ("FALSIFIED" if len(shift_bad) > 1 or len(wrong) > 2 else "HELD") + drift_mark,
                      "dF_trace_ms": dF_ms, "shifts": rows, "shift_outside": shift_bad,
                      "verdict_wrong": wrong}
    # ---- P5
    agree = tot = 0
    miss = []
    for (mode, arm, n), c in sorted(C.items()):
        if mode not in MODES_EAGER or edge(mode, arm, n):
            continue
        rule = rule_hb(mode, arm, n)
        if rule is None:
            continue
        meas = c["hb"] >= 0.8
        tot += 1
        agree += rule == meas
        if rule != meas:
            miss.append((mode, arm, n))
    frac = agree / tot if tot else None
    res["P"]["P5"] = {"verdict": ("NOT TESTED" if tot == 0 else "HELD" if frac >= 0.95 else "FALSIFIED")
                      + drift_mark, "agree": agree, "of": tot, "miss": miss}
    # ---- P6
    import launch_mp as MPM
    rows, bad = [], []
    tdir = d / "traces"
    for (mode, arm, n), c in sorted(C.items()):
        if mode not in MODES_EAGER:
            continue
        fe, fg = tdir / f"TR-E-{mode}-{arm}-n{n}.json.gz", tdir / f"TR-G-{mode}-{arm}-n{n}.json.gz"
        if not (fe.exists() and fg.exists()):
            continue
        pred, sc = MPM.mp(MPM.timeline(fe), MPM.timeline(fg), c["H"] * 1e3)
        band = max(0.05 * c["I"], 0.012)
        ok = abs(pred - c["I"]) <= band
        rows.append({"mode": mode, "arm": arm, "n": n, "measured": c["I"], "MP": pred, "inside": ok})
        if not ok:
            bad.append((mode, arm, n))
    res["P"]["P6"] = {"verdict": "NOT TESTED (no traces)" if not rows else
                      ("FALSIFIED" if len(bad) > 1 else "HELD"), "cells": rows, "outside": bad}
    # ---- P7
    p7 = {}
    for arm in ARMS:
        k = f"TR-G-E240-{arm}-n1.json.gz"
        e = f"TR-E-E240-{arm}-n1.json.gz"
        if k in parsed:
            p7[arm] = {"TRG_kernel_sum_us": parsed[k]["median"]["kernel_sum_us"],
                       "TRG_non_gemm_us": parsed[k]["median"]["non_gemm_us"],
                       "launch_api_share": parsed.get(e, {}).get("launch_api_share")}
    res["P"]["P7"] = {"verdict": "CLASSIFIES", "cells": p7}
    # ---- P8
    rows, bad = [], []
    for arm in ARMS:
        for n in range(1, 10):
            c = C.get(("E240", arm, n))
            if c is None or edge("E240", arm, n) or not rule_hb("E240", arm, n):
                continue
            ok = c["H"] - 0.12 <= c["I"] <= c["H"] - 0.08
            rows.append({"arm": arm, "n": n, "I": c["I"], "H_cell": c["H"], "inside": ok})
            if not ok:
                bad.append((arm, n))
    res["P"]["P8"] = {"verdict": ("NOT TESTED" if not rows else
                                  "FALSIFIED" if len(bad) > 0.10 * len(rows) else "HELD") + drift_mark,
                      "cells": rows, "outside": bad}
    return res


def score(repo: Path, tree: Path) -> dict:
    reg = CM.registration(repo, PART)
    out = {"registration": CM.NAMES[PART], "models": {}}
    for model in reg["C_reg_ms"]:
        if model == "source":
            continue
        d = find_model_dir(tree, model)
        out["models"][model] = score_model(reg, model, d) if d else {"verdict": "NOT RUN: no directory"}
    rows = [dict(r, model=m) for m, v in out["models"].items() if "P" in v for r in v["P"]["P3"]["cells"]]
    bad = [r for r in rows if not r["inside"]]
    out["P3_pooled"] = {"cells": len(rows), "outside": [(r["model"], r["arm"], r["n"]) for r in bad],
                        "verdict": (f"NOT TESTED ({len(rows)} qualifying cells, 8 needed)" if len(rows) < 8
                                    else "FALSIFIED" if len(bad) > 1 else "HELD")}
    return out


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 2 PART 2 (launch floor rerun), {res['registration']}"]
    for m, r in res["models"].items():
        if "P" not in r:
            out.append(f"  {m}: {r['verdict']}")
            continue
        out.append(f"  {m}: " + "; ".join(f"{k} {v['verdict']}" for k, v in r["P"].items()))
        if r["P"]["P0"]["why"]:
            out.append(f"    P0: {r['P']['P0']['why']}")
    out.append(f"  P3 (both models together): {res['P3_pooled']['verdict']} on {res['P3_pooled']['cells']} cells")
    return out


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    sys.path[:0] = [str(repo), str(repo / "scripts")]
    res = score(repo, tree)
    out.mkdir(parents=True, exist_ok=True)
    (out / "launch.score.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    text = "\n".join(lines(res)) + "\n"
    (out / "launch.score.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
