#!/usr/bin/env python
"""Score rental 3's part D, the launch floor's flush ladder
(docs/registered/2026-10-05-rental3-flush-gh200.json): OVERLAP against SAT.

    python scripts/scoring/rental3/score_flush.py <repo> <tree> <out>

Reads the launch-floor directory `*-launch-floor-r3/mixtral-8x7b-tp8/` (cells.csv,
hostprobe.csv, hostprobe-post.csv, manifest.json) under <tree>. Writes
<out>/flush.score.{json,txt}; exits 0 whatever the verdicts. Only the modes of 240
MiB and above enter a verdict; E0 and E120 are printed.
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
import r3common as C3  # noqa: E402

PART = "flush"
ARMS = ("native", "shared", "private")
SCORED = ("E240", "E360", "E480")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_cells(d: Path) -> dict:
    """rental 2's definitions: I the median ms_p50 over the cell's repeats, H the
    median host_enqueue_ms / calls_per_burst, hb the host-bound fraction."""
    agg = defaultdict(list)
    for r in csv.DictReader(open(d / "cells.csv")):
        if r.get("status") != "ok":
            continue
        agg[(r["mode"], r["arm"], int(r["tiles"]))].append(r)
    return {k: {"I": st.median(float(r["ms_p50"]) for r in rs),
                "H": st.median(float(r["host_enqueue_ms"]) / float(r["calls_per_burst"]) for r in rs),
                "hb": sum(str(r["host_bound"]) == "True" for r in rs) / len(rs), "reps": len(rs)}
            for k, rs in agg.items()}


def load_probe(path: Path) -> dict:
    if not path.exists():
        return {}
    return {(r["mode"], r["arm"], int(r["tiles"])): r for r in csv.DictReader(open(path))}


def p0(man: dict, C: dict, pre: dict, post: dict) -> dict:
    why = []
    if man.get("phase") != "timed":
        why.append(f"manifest phase {man.get('phase')!r}")
    if man.get("profiler_enabled_ever"):
        why.append("the profiler was on in the timed process")
    drift = [abs(fnum(post[k]["H_pre_ms"]) / fnum(pre[k]["H_pre_ms"]) - 1)
             for k in pre if k in post and fnum(pre[k].get("H_pre_ms")) and fnum(post[k].get("H_pre_ms"))]
    med = st.median(drift) if drift else None
    if med is None or med > 0.05:
        why.append(f"median post/pre drift {med}")
    return {"verdict": "HELD" if not why else "FAILED", "why": why}


def score_view(repo: Path, tree: Path, fview, tview) -> dict:
    import numpy as np
    reg = C3.registration(repo, PART)
    res = {"registration": C3.NAMES[PART]}
    hits = sorted(Path(tree).rglob("*-launch-floor-r3/mixtral-8x7b-tp8/cells.csv"))
    if len(hits) != 1:
        res["verdict"] = "NOT RUN: the r3 launch-floor directory is missing"
        return res
    d = hits[0].parent
    C = load_cells(d)
    pre, post = load_probe(d / "hostprobe.csv"), load_probe(d / "hostprobe-post.csv")
    man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    res["P0_printed"] = p0(man, C, pre, post)
    F = reg["F_gpu_ms"]
    creg = reg["C_reg_ms"]
    rows = []
    for arm in ARMS:
        for n in reg["unit"]["treads"]:
            ok, phi = True, {}
            for mode in SCORED + ("E0", "E120"):
                c = C.get((mode, arm, n))
                if c is None:
                    if mode in SCORED:
                        ok = False
                    continue
                phi[mode] = c["H"] - c["I"]
                if mode not in SCORED:
                    continue
                h = fnum((pre.get((mode, arm, n)) or {}).get("H_pre_ms"))
                cr = float(creg[arm][str(n)])
                rule_hb = h is not None and cr + F[mode] < h
                edge = h is None or abs(cr + F[mode] - h) <= 0.020 or 0.2 <= c["hb"] <= 0.8
                if not (rule_hb and c["hb"] >= 0.8 and not edge):
                    ok = False
            row = {"arm": arm, "n": n, "Phi_ms": phi, "qualifies": ok}
            if all(m in phi for m in SCORED):
                d1, d2 = (phi["E360"] - phi["E240"]) * 1e3, (phi["E480"] - phi["E360"]) * 1e3
                ov, sa = reg["tests"]["OVERLAP_band_us"], reg["tests"]["SAT_band_us"]
                row.update(d1_us=d1, d2_us=d2,
                           OVERLAP=all(ov[0] <= x <= ov[1] for x in (d1, d2)),
                           SAT=all(sa[0] <= x <= sa[1] for x in (d1, d2)),
                           slope=float(np.polyfit([F[m] for m in SCORED], [phi[m] for m in SCORED], 1)[0]))
            rows.append(row)
    q = [r for r in rows if r["qualifies"] and "d1_us" in r]
    res["cells"] = rows
    if len(q) < 3:
        res["verdict"] = f"NOT TESTED: {len(q)} qualifying cells, under 3"
        return res
    n = len(q)
    o = sum(r["OVERLAP"] for r in q) / n
    s = sum(r["SAT"] for r in q) / n
    out_o = sum(not r["OVERLAP"] for r in q) / n
    out_s = sum(not r["SAT"] for r in q) / n
    st_ = {"OVERLAP": "SELECTED" if o >= 2 / 3 else "FALSIFIED" if out_o >= 2 / 3 else "neither",
           "SAT": "SELECTED" if s >= 2 / 3 else "FALSIFIED" if out_s >= 2 / 3 else "neither"}
    sel = [h for h, v in st_.items() if v == "SELECTED"]
    v = sel[0] if len(sel) == 1 else "INCONCLUSIVE"
    if res["P0_printed"]["verdict"] != "HELD" and v not in ("INCONCLUSIVE",):
        v = f"{v} (host drift)"
    res.update(hypotheses=st_, qualifying=n, verdict=v)
    return res


def score(repo: Path, tree: Path) -> dict:
    return C3.two_views(lambda fv, tv: score_view(repo, tree, fv, tv), repo=repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 3 PART D (the flush ladder), {res['registration']}"]
    for r in res.get("cells", []):
        out.append(f"  {r['arm']:8s} n={r['n']} qualifies {r['qualifies']} "
                   + (f"d1 {r['d1_us']:+.1f} us d2 {r['d2_us']:+.1f} us slope {r['slope']:.2f}" if "d1_us" in r else ""))
    out.append(f"  hypotheses: {res.get('hypotheses')}")
    out.append(f"  verdict: {res['verdict']}")
    return out + C3.view_lines(res)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C3.write_score(out, PART, res, lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
