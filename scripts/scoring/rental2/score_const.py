#!/usr/bin/env python
"""Score rental 2's part 3, the per-GEMM constant
(docs/registered/2026-10-01-rental2-const-gh200.json): K1 to K5 and the
identification.

    python scripts/scoring/rental2/score_const.py <repo> <tree> <out>

Reads the floor files `*-nvidia_gh200_480gb-<model>-<label>-r3-counters/
r3f-g64[-lock<F>].json` under <tree>. Writes <out>/const.score.{json,txt};
exits 0 whatever the verdicts. The cell set is the registration's
(`cells_registered`, floor_bound at sigma 1), never re-chosen on the pages.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as CM  # noqa: E402

PART = "const"
SMS = 132


def find_floor(tree: Path, model: str, label: str, stem: str) -> dict | None:
    hits = sorted(Path(tree).rglob(f"*-{CM.CARD}-{model}-{label}-r3-counters/{stem}.json"))
    return json.loads(hits[0].read_text()) if len(hits) == 1 else None


def capture_rows(reg: dict, model: str, page: dict, gemm: str, ns=None) -> list[dict]:
    """Per registered floor-bound cell of `gemm` (restricted to `ns` when
    given) present on the page: the registered q, ceil, u and the measured
    counters, with the derived terms."""
    reg_cells = {(c["n"], c["gemm"]): c for c in reg["cells_registered"][model]}
    by_n = {int(c["n"]): c for c in page["cells"]}
    out = []
    for (n, g), rc in sorted(reg_cells.items()):
        if g != gemm or not rc["floor_bound"] or (ns is not None and n not in ns) or n not in by_n:
            continue
        pg = by_n[n]["per_gemm"][g]
        q, u, ce = rc["q"], rc["u"], rc["ceil"]
        ac, el = pg.get("sm__cycles_active.avg"), pg.get("sm__cycles_elapsed.avg")
        amax = pg.get("sm__cycles_active.max")
        row = {"n": n, "q": q, "u": u, "ceil": ce, "active": ac, "elapsed": el,
               "active_max": amax, "mhz": pg.get("sm_clock_mhz"),
               "Z_cell": None if ac is None else ac - q * u,
               "T": None if ac is None or el is None else el - ac - (ce - q) * u,
               "L": None if el is None or amax is None else el - amax,
               "D_imb": None if amax is None or ac is None else amax - ac - (ce - q) * u,
               "cta_excess": (None if pg.get("sm__ctas_launched.max") is None
                              else pg["sm__ctas_launched.max"] - math.ceil(rc["grid"] / SMS)),
               "concurrency": (None if pg.get("sm__ctas_active.sum") is None or not ac
                               else pg["sm__ctas_active.sum"] / (SMS * ac))}
        null = by_n[n].get("null")
        if null and null.get("sm__cycles_elapsed.avg") is not None:
            row["L0"] = null["sm__cycles_elapsed.avg"] - reg["constants"]["null_kernel_cycles"]
        out.append(row)
    return out


def z_intercept(rows: list[dict]):
    import floor_estimator as FE
    pts = [(r["q"], r["active"]) for r in rows if r["active"] is not None]
    if len(pts) < 3:
        return None, None
    z, slope = FE.intercept([p[0] for p in pts], [p[1] for p in pts])
    return z, slope


def score(repo: Path, tree: Path) -> dict:
    reg = CM.registration(repo, PART)
    caps = reg["captures"]
    res = {"registration": CM.NAMES[PART], "captures": {}, "K": {}}
    pages = {}
    for key, spec in caps.items():
        if not isinstance(spec, dict) or "label" not in spec:
            continue
        model = key.split(" ")[0]
        for clk in spec["clocks"]:
            stem = "r3f-g64" if clk == "base" else f"r3f-g64-{clk}"
            pages[(model, clk)] = find_floor(tree, model, spec["label"], stem)
    for (model, clk), page in pages.items():
        if page is None:
            res["captures"][f"{model} {clk}"] = {"verdict": "missing"}
            continue
        ent = {}
        for g in ("w1", "w2"):
            rows = capture_rows(reg, model, page, g)
            z, slope = z_intercept(rows)
            u = rows[0]["u"] if rows else None
            ent[g] = {"rows": rows, "Z_f": z, "Z_f_over_u": (z / u if z is not None and u else None),
                      "slope_over_u": (slope / u if slope is not None and u else None),
                      "clock_mhz": CM.median([r["mhz"] for r in rows if r["mhz"]])}
        res["captures"][f"{model} {clk}"] = ent
    lock = {m: res["captures"].get(f"{m} lock1710") for m in ("mixtral-8x7b-tp8", "mixtral-8x7b-tp4",
                                                             "mixtral-8x7b-tp2")}
    pool = [r for c in lock.values() if c and "w1" in c for g in ("w1", "w2") for r in c[g]["rows"]]
    # K1
    Ls = [r["L"] for r in pool if r["L"] is not None]
    thr = reg["K1"]["threshold_cycles"]
    if Ls:
        over = sum(v > thr for v in Ls) / len(Ls)
        k1 = "FALSIFIED" if over > 0.20 else "HOLDS"
        res["K"]["K1"] = {"cells": len(Ls), "fraction_over": over, "verdict": k1}
    else:
        res["K"]["K1"] = {"verdict": "NOT SCORED: no L readable (no shape metrics)"}
    # K2
    vals = [lock[m]["w2"]["Z_f_over_u"] for m in ("mixtral-8x7b-tp4", "mixtral-8x7b-tp2")
            if lock.get(m) and "w2" in lock[m] and lock[m]["w2"]["Z_f_over_u"] is not None]
    if vals:
        pooled = CM.median(vals)
        lo, hi = reg["K2"]["band"]
        res["K"]["K2"] = {"values": vals, "pooled": pooled,
                          "verdict": "HOLDS" if lo <= pooled <= hi else "FALSIFIED"}
    else:
        res["K"]["K2"] = {"verdict": "NOT SCORED"}
    # K3
    p1005, p1710 = pages.get(("mixtral-8x7b-tp4", "lock1005")), pages.get(("mixtral-8x7b-tp4", "lock1710"))
    if p1005 is not None and p1710 is not None:
        per = {}
        for g in ("w1", "w2"):
            nz = reg["K3"]["noise"][g]
            r5 = capture_rows(reg, "mixtral-8x7b-tp4", p1005, g, ns=set(nz["cells_1005"]))
            r7 = capture_rows(reg, "mixtral-8x7b-tp4", p1710, g, ns=set(nz["cells_1710"]))
            z5, _ = z_intercept(r5)
            z7, _ = z_intercept(r7)
            f5 = CM.median([r["mhz"] for r in r5 if r["mhz"]])
            f7 = CM.median([r["mhz"] for r in r7 if r["mhz"]])
            if z5 is None or z7 is None or not z7 or f5 is None or f7 is None:
                per[g] = None
                continue
            r = z5 / z7
            per[g] = {"Z_1005": z5, "Z_1710": z7, "r": r, "rho_f": f5 / f7,
                      "sigma_r": abs(r) * nz["sigma_ratio_rel"]}
        ok = [v for v in per.values() if v]
        if ok:
            r = sum(v["r"] for v in ok) / len(ok)
            rho = sum(v["rho_f"] for v in ok) / len(ok)
            sig = math.sqrt(sum(v["sigma_r"] ** 2 for v in ok)) / len(ok)
            ns_in, cyc_in = abs(r - rho) <= 2 * sig, abs(r - 1) <= 2 * sig
            if ns_in and not cyc_in:
                v = "ns form HOLDS, cycle form FALSIFIED"
            elif cyc_in and not ns_in:
                v = "cycle form HOLDS, ns form FALSIFIED"
            else:
                v = "INCONCLUSIVE"
            res["K"]["K3"] = {"per_gemm": per, "pooled_r": r, "pooled_rho_f": rho, "sigma": sig,
                              "verdict": v}
        else:
            res["K"]["K3"] = {"per_gemm": per, "verdict": "NOT SCORED"}
    else:
        res["K"]["K3"] = {"verdict": "NOT SCORED: a tp4 capture is missing"}
    # K4
    D = [(r["D_imb"], r["u"]) for r in pool if r["D_imb"] is not None]
    if D:
        within = sum(-0.2 * u <= d <= 0.2 * u for d, u in D) / len(D)
        big = sum(d >= 0.3 * u for d, u in D) / len(D)
        v4 = "FALSIFIED (H_IMB)" if big > 0.20 else ("HOLDS" if within >= 0.80 else "INCONCLUSIVE")
        res["K"]["K4"] = {"cells": len(D), "fraction_within_0.2u": within, "fraction_ge_0.3u": big,
                          "verdict": v4,
                          "cta_excess_RECORD": [r["cta_excess"] for r in pool if r["cta_excess"] is not None]}
    else:
        res["K"]["K4"] = {"verdict": "NOT SCORED: no D_imb readable"}
    # K5 record
    rec = {}
    for (model, clk), page in pages.items():
        if page is None:
            continue
        for c in page["cells"]:
            if int(c["n"]) in (1, 2):
                for g in ("w1", "w2"):
                    pg = c["per_gemm"][g]
                    rec[f"{model} {clk} n={c['n']} {g}"] = {
                        "L": (None if pg.get("sm__cycles_active.max") is None
                              else pg["sm__cycles_elapsed.avg"] - pg["sm__cycles_active.max"]),
                        "L0": (c.get("null") or {}).get("sm__cycles_elapsed.avg")}
    res["K"]["K5"] = {"verdict": "RECORD", "cells": rec}
    k1, k2 = res["K"]["K1"].get("verdict"), res["K"]["K2"].get("verdict")
    k3, k4 = res["K"]["K3"].get("verdict", ""), res["K"]["K4"].get("verdict")
    if k1 == "FALSIFIED":
        ident = "H_LD"
    elif k4 and k4.startswith("FALSIFIED"):
        ident = "H_IMB"
    elif k1 == "HOLDS" and k2 == "HOLDS" and k4 == "HOLDS" and k3.startswith("ns form HOLDS"):
        ident = "H_ZT"
    else:
        ident = "INCONCLUSIVE"
    res["identification"] = ident
    return res


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 2 PART 3 (the per-GEMM constant), {res['registration']}"]
    for k, v in res["K"].items():
        out.append(f"  {k}: {v.get('verdict')}" + "".join(
            f" {x}={v[x]:.4g}" for x in ("fraction_over", "pooled", "pooled_r", "pooled_rho_f", "sigma",
                                         "fraction_within_0.2u") if isinstance(v.get(x), float)))
    for c, e in res["captures"].items():
        if "w1" in e:
            out.append(f"  {c}: " + "; ".join(
                f"{g} Z_f/u {e[g]['Z_f_over_u']:.3f} clock {e[g]['clock_mhz']}" for g in ("w1", "w2")
                if e[g]["Z_f_over_u"] is not None))
    out.append(f"  identification: {res['identification']}")
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
    (out / "const.score.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    text = "\n".join(lines(res)) + "\n"
    (out / "const.score.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
