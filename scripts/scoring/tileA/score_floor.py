#!/usr/bin/env python
"""Score rental A's floor (docs/registered/2026-10-14-tileA-floor-gh200.json): the 128x128 calibration
(c_128, F_wg, Z_wg per accounting; CAL-A), the 64x128 test (DESCRIPTIVE), the selections rental B's
v4 takes (accounting or v4cal, F form), the w4 and mma.sync pages (printed).

    python scripts/scoring/tileA/score_floor.py <repo> <tree> <out>

Reads under <tree>: <date>-<card>-<model>-<label>-r3-counters/lock1710/r3c-g64.json for labels f128,
f64w8 (OLMoE and Mixtral 8x7B), f64w4 and f32w4 (OLMoE); <out>/census.score.json when present (the
census occupancy). Writes <out>/floor.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# tcommon first: it puts the repo and scripts/ on sys.path, which r3_timing_model_v4 needs from the
# command line (the tests import tcommon before this file, which hid the order)
import tcommon as TC  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402

PART = "floor"
r6 = TC.r6
UNITS = {"f128": (V4.CAL_TILE, ("olmoe-1b-7b", "mixtral-8x7b")), "f64w8": (V4.T64W8, ("olmoe-1b-7b", "mixtral-8x7b")),
         "f64w4": (V4.T64W4, ("olmoe-1b-7b",)), "f32w4": (V4.T32W4, ("olmoe-1b-7b",))}
R3_F = {"w1": 520.0, "w2": 979.0}


def pages_of(tree: Path, label: str) -> tuple[list[dict], dict]:
    t, models = UNITS[label]
    got, why = [], {}
    for m in models:
        p, path, w = TC.counter_page(tree, m, label, 64, tile=[t.bm, t.bn, t.bk, t.warps, t.stages])
        if p is None:
            why[m] = w
            continue
        clk = TC.C6.clock_rule(p, path)
        if not clk["page_ok"]:
            why[m] = f"NOT SCORED (clock rule): {clk['why']}"
            continue
        keep = [c for c, cc in zip(p.get("cells") or [], clk["cells"], strict=True) if cc["ok"]]
        if not keep:
            why[m] = f"NOT SCORED (clock rule): no cell in {TC.C6.BAND_MHZ} MHz"
            continue
        n = len(p.get("cells") or [])
        why[m] = "counted" if len(keep) == n else f"counted ({len(keep)} of {n} cells in the clock band)"
        got.append(dict(p, cells=keep))
    return got, why


def occ_of(out: Path) -> dict:
    p = Path(out) / "census.score.json"
    return (json.loads(p.read_text()).get("occupancy") or {}) if p.exists() else {}


def score(repo: Path, tree: Path, out: Path | None = None) -> dict:
    reg = TC.registration(repo, PART)
    occ = occ_of(out) if out else {}
    pages = {k: pages_of(tree, k) for k in UNITS}
    res = {"registration": reg["name"], "part": PART, "pages": {k: v[1] for k, v in pages.items()}, "census_occupancy": occ}
    cal = TC.joint_cF(pages["f128"][0])
    if cal is None:
        res["calibration"] = {"verdict": "NOT SCORED: under 6 floor-bound cells or rank deficient (no v4: rental B does not run)"}
        return res
    res["calibration"] = {"verdict": "CAL-A", **cal}
    meas64 = TC.joint_cF(pages["f64w8"][0])
    res["fit_64x128_w8"] = meas64
    phi_wg = {g: cal["F"][g] / cal["c"] for g in ("w1", "w2")}

    def recorded(label):
        """ambiguity 11: a counter page whose recorded occupancy (the min launch__occupancy_limit_*)
        differs from the census is scored at its recorded one (rental 4's re-key)."""
        got = {}
        for p in pages[label][0]:
            o = sorted({r["occ"] for r in TC.floor_rows(p) if r["occ"] is not None})
            if len(o) == 1:
                got[p["design"]["model"]] = o[0]
        return got
    rec128, rec64 = recorded("f128"), recorded("f64w8")
    cal_occ = sorted(set(rec128.values()))
    occ_cal = dict(occ)
    if len(cal_occ) == 1:
        occ_cal[V4.CAL_TILE.key] = {"w1": cal_occ[0], "w2": cal_occ[0]}
    res["re_key"] = {"f128 recorded": rec128, "f64w8 recorded": rec64, "census": occ,
                     "rule": "each page is scored at its recorded occupancy where it differs from the census"}
    tests, zs = {}, {}
    for v in V4.VARIANTS:
        prm = V4.Params(c128=cal["c"], F_wg=dict(cal["F"]), variant=v, occupancy=dict(occ_cal))
        z = V4.z_wg(prm)
        zs[v] = r6(z) if z is not None else None
        if z is None or meas64 is None:
            tests[v] = None
            continue
        per = {}
        for m in UNITS["f64w8"][1]:
            o = dict(occ_cal)
            if m in rec64:
                o[V4.T64W8.key] = {"w1": rec64[m], "w2": rec64[m]}
            per[m] = V4.c_counter(V4.T64W8, "w1", V4.Params(c128=cal["c"], F_wg=dict(cal["F"]), variant=v, occupancy=o))
        rows = []
        for m in UNITS["f64w8"][1]:
            t = TC.floor_test(per[m], {g: phi_wg[g] * per[m] for g in ("w1", "w2")}, meas64, (m,))
            rows += t["rows"]
        ok = bool(rows) and all(abs(r["e"]) <= TC.FLOOR_TOL for r in rows)
        tests[v] = {"verdict": "HOLDS" if ok else "FAILS", "rows": rows, "tol": TC.FLOOR_TOL,
                    "c64_pred": {m: r6(c) for m, c in per.items()}}
    res["Z_wg"] = zs
    res["FLOOR-TEST (DESCRIPTIVE)"] = {v: (t if t is not None else "unreachable or no 64x128 fit") for v, t in tests.items()}
    res["selection"] = TC.variant_rule(tests) if meas64 is not None else \
        {"verdict": "NOT SCORED: no 64x128 w8 fit", "selected": None, "v4_at_64x128": "none: finalise.py refuses rental B and the stop rule STOPs (no 64 G 1 E1-CAL prediction)"}
    res["f_form"] = TC.f_form_rule(meas64, phi_wg, cal["F"]) if meas64 is not None else \
        {"selected": V4.F_FORM_REGISTERED, "rule": "no 64x128 fit: F-step kept"}
    w4 = TC.c_with_F_held(pages["f64w4"][0], meas64["F"] if meas64 else R3_F)
    res["w4 (printed)"] = w4 and {"c_meas": r6(w4["c"]), **{v: r6(V4.c_counter(V4.T64W4, "w1", V4.Params(c128=cal["c"], F_wg=dict(cal["F"]), variant=v, occupancy=dict(occ))))
                                                           for v in V4.VARIANTS if zs[v] is not None}}
    m32 = TC.c_with_F_held(pages["f32w4"][0], R3_F)
    if m32:
        pc = V4.c_counter(V4.T32W4, "w1", V4.Params(occupancy=dict(occ)))
        e = pc / m32["c"] - 1
        res["MVA2 at 32x128x64 w4 (printed for D1)"] = {"c_meas": r6(m32["c"]), "c_MVA2": r6(pc), "e": r6(e),
                                                        "verdict": "HOLDS" if abs(e) <= TC.FLOOR_TOL else "FAILS"}
    return res


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']}", f"  pages: {res['pages']}"]
    c = res.get("calibration", {})
    out.append(f"CALIBRATION: {c.get('verdict')} c {c.get('c')} F {c.get('F')}")
    out.append(f"Z_wg: {res.get('Z_wg')}")
    for v, t in (res.get("FLOOR-TEST (DESCRIPTIVE)") or {}).items():
        out.append(f"  {v}: {t if isinstance(t, str) else t['verdict']}")
    out.append(f"SELECTION: {(res.get('selection') or {}).get('verdict')}; F form {(res.get('f_form') or {}).get('selected')}")
    return out


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 3:
        print(__doc__)
        return 2
    res = score(Path(a[0]), Path(a[1]), Path(a[2]))
    print(TC.C3.write_score(Path(a[2]), PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
