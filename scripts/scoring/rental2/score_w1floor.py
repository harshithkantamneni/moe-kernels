#!/usr/bin/env python
"""Score rental 2's part 4, tp8's w1 floor
(docs/registered/2026-10-01-rental2-w1floor-gh200.json): the per-set slopes,
theta and delta, the w2 offset test, and the co-primary per-cell test.

    python scripts/scoring/rental2/score_w1floor.py <repo> <tree> <out>

Primary: the base-clock G = 64 captures (r3f-g64.json); the 1710 lock captures
are scored the same way and printed beside. Writes <out>/w1floor.score.{json,txt};
exits 0 whatever the verdicts. Before anything is read off a page the H_EST
numbers are recomputed from the registered per-cell cycles with
floor_estimator.implied_slope and checked against the JSON within 0.1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as CM  # noqa: E402

PART = "w1floor"
LABELS = {"mixtral-8x7b-tp8": "floor2", "mixtral-8x7b-tp4": "floor", "mixtral-8x7b-tp2": "floor"}


def find_floor(tree: Path, model: str, stem: str, view=None) -> dict | None:
    """The one floor file of `model` at `stem` as `view` counts it (the addendum's
    ALL or CLEAN; read as it stands when no view is given), None when missing."""
    hits = sorted(Path(tree).rglob(f"*-{CM.CARD}-{model}-{LABELS[model]}-r3-counters/{stem}.json"))
    if len(hits) != 1:
        return None
    return view.load(hits[0]) if view is not None else json.loads(hits[0].read_text())


def check_registration(reg: dict, cregs: dict) -> list[str]:
    """H_EST recomputed from the registered cells; [] when every set agrees."""
    import floor_estimator as FE
    bad = []
    for key, s in reg["sets"].items():
        model, g = key.split(" ")
        pc = {c["n"]: (c["grid"], c["cores"]) for c in cregs[model] if c["gemm"] == g}
        for name, v in s["sets"].items():
            est = FE.implied_slope(pc, v["treads"], s["ksteps"])
            if abs(est - v["H_EST"]) > 0.1:
                bad.append(f"{key} {name}: {est:.2f} against {v['H_EST']}")
    return bad


def measured(page: dict, g: str, reg_cells: list[dict]) -> dict:
    """{n: (grid, cycles)} off the page, the grid asserted equal to the
    registered one."""
    want = {c["n"]: c["grid"] for c in reg_cells if c["gemm"] == g}
    out = {}
    for c in page["cells"]:
        n = int(c["n"])
        if g not in c["per_gemm"]:      # dropped by the addendum's rule 2
            continue
        grid = int((c.get("grid") or {}).get(g) or c["per_gemm"][g].get("launch__grid_size"))
        if n in want and grid != want[n]:
            raise ValueError(f"n={n} {g}: the page's grid {grid} is not the registered {want[n]}")
        out[n] = (grid, float(c["per_gemm"][g]["sm__cycles_elapsed.avg"]))
    return out


def score_capture(reg: dict, cregs: dict, pages: dict) -> dict:
    import floor_estimator as FE
    sig = reg["noise"]["sigma_cell_cycles"]
    noise = reg["noise"]["theta_delta"]
    out = {"gemms": {}}
    for key, s in reg["sets"].items():
        model, g = key.split(" ")
        page = pages.get(model)
        if page is None:
            out["gemms"][key] = {"verdict": "NOT SCORED: capture missing"}
            continue
        meas = measured(page, g, cregs[model])
        rows = {}
        for name, v in s["sets"].items():
            if not all(n in meas for n in v["treads"]):
                rows[name] = None
                continue
            sl = FE.implied_slope(meas, v["treads"], s["ksteps"])
            rows[name] = {"slope": sl, "r": sl / s["asymptote"] - 1, "bias": v["bias"],
                          "H_EST": v["H_EST"], "sigma_slope": v["sigma_slope"]}
        ent = {"sets": rows}
        ok = [r for r in rows.values() if r]
        if g == "w1" and len(ok) >= 3:
            td = FE.theta_delta([{"slope": r["slope"], "asymptote": s["asymptote"], "bias": r["bias"]}
                                 for r in ok])
            ent.update(td)
            ent["delta_H"] = sum(r["r"] - r["bias"] for r in ok) / len(ok)
            ent["delta_F"] = sum(r["r"] for r in ok) / len(ok)
            ent.update(noise[key])
        elif g == "w1" and ok:
            ent["delta_H"] = sum(r["r"] - r["bias"] for r in ok) / len(ok)
            ent["delta_F"] = sum(r["r"] for r in ok) / len(ok)
            ent.update(noise[key])
        elif g == "w2" and ok:
            m = sum(r["r"] for r in ok) / len(ok)
            b = sum(r["bias"] for r in ok) / len(ok)
            sm = (sum(r["sigma_slope"] ** 2 for r in ok) ** 0.5 / len(ok)) / s["asymptote"]
            ent["w2_offset"] = {"mean_r": m, "H_EST_pred": b, "sigma": sm,
                                "H_EST": "HOLDS" if abs(m - b) <= 2 * sm else "FALSIFIED",
                                "FLUID": "HOLDS" if abs(m) <= 2 * sm else "FALSIFIED"}
        # the co-primary per-cell test
        lo = reg["unseen_min_tread"][model]
        rc = {c["n"]: c for c in cregs[model] if c["gemm"] == g}
        un = [n for n in sorted(meas) if n >= lo and n in rc and rc[n]["waves"] >= 4]
        if len(un) >= 3:
            ys = [meas[n][1] for n in un]
            rms_c = FE.form_rms(ys, [rc[n]["ceil"] * rc[n]["u"] for n in un])
            rms_f = FE.form_rms(ys, [rc[n]["q"] * rc[n]["u"] for n in un])
            win = ("CEIL" if rms_f - rms_c > sig else "FLUID" if rms_c - rms_f > sig else "INCONCLUSIVE")
            ent["per_cell"] = {"cells": un, "rms_ceil": rms_c, "rms_fluid": rms_f, "winner": win}
        ent["per_cell_record"] = {str(n): (meas[n][1] - rc[n]["cores"]) / rc[n]["cores"]
                                  for n in sorted(meas) if n in rc and rc[n]["waves"] >= 4}
        out["gemms"][key] = ent
    # verdicts
    G = out["gemms"]
    th = {k: G.get(f"{k} w1", {}) for k in ("mixtral-8x7b-tp8", "mixtral-8x7b-tp4")}
    w1s = [G.get(f"{m} w1", {}) for m in ("mixtral-8x7b-tp8", "mixtral-8x7b-tp4", "mixtral-8x7b-tp2")]

    def theta_off(target):
        return any("theta" in t and abs(t["theta"] - target) > 2 * t["sigma_theta"] for t in th.values())

    def delta_off(key, target):
        return sum(1 for t in w1s if key in t and abs(t[key] - target) > 2 * t["sigma_delta"]) >= 2
    have = all("theta" in t for t in th.values())
    fam = {}
    if have:
        side = {k: ("H_EST" if t["theta"] > 0.5 else "FLUID") for k, t in th.items()}
        fam["theta_call"] = side
        fam["H_EST"] = "FALSIFIED" if theta_off(1.0) or delta_off("delta_H", 0.0) else "HOLDS"
        fam["FLUID"] = ("FALSIFIED" if theta_off(0.0) or any(t["theta"] > 0.5 for t in th.values())
                        or delta_off("delta_F", 0.0) else "HOLDS")
        t8 = th["mixtral-8x7b-tp8"]
        fam["FLUID_LOW"] = ("FALSIFIED" if theta_off(0.0) or any(t["theta"] > 0.5 for t in th.values())
                            or abs(t8["delta_F"] + 0.036) > 2 * t8["sigma_delta"] else "HOLDS")
        fam["H_NPN (secondary)"] = "FALSIFIED" if t8["delta_H"] > -0.020 else "HOLDS"
        main = [k for k in ("H_EST", "FLUID", "FLUID_LOW") if fam[k] == "HOLDS"]
        fam["verdict"] = ("NEITHER" if not main else main[0] if len(main) == 1 else
                          "INCONCLUSIVE between " + ", ".join(main))
    else:
        fam["verdict"] = "NOT SCORED: a w1 GEMM lacks its three sets"
    pc = {k: G.get(k, {}).get("per_cell", {}).get("winner") for k in ("mixtral-8x7b-tp8 w1", "mixtral-8x7b-tp4 w1")}
    fam["co_primary"] = ("NOT SCORED: no per-cell test" if all(v is None for v in pc.values()) else
                         "H_EST supported" if all(v == "CEIL" for v in pc.values()) else
                         "FLUID supported" if all(v == "FLUID" for v in pc.values()) else "INCONCLUSIVE")
    fam["co_primary_winners"] = pc
    out["families"] = fam
    return out


def score_view(repo: Path, tree: Path, view) -> dict:
    """Every verdict on the pages `view` counts."""
    reg = CM.registration(repo, PART)
    creg = CM.registration(repo, "const")["cells_registered"]
    bad = check_registration(reg, creg)
    res = {"registration": CM.NAMES[PART], "H_EST_recomputed": "agree" if not bad else bad}
    for clock, stem in (("base (PRIMARY)", "r3f-g64"), ("lock1710 (printed)", "r3f-g64-lock1710")):
        pages = {m: find_floor(tree, m, stem, view) for m in LABELS}
        res[clock] = score_capture(reg, creg, pages)
    return res


def score(repo: Path, tree: Path) -> dict:
    """The registered verdicts: each computed on ALL and on CLEAN pages and
    combined by the addendum's rule (common.two_views)."""
    return CM.two_views(lambda v: score_view(repo, tree, v), repo, PART)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 2 PART 4 (tp8's w1 floor), {res['registration']}; H_EST recomputed: "
           f"{res['H_EST_recomputed']}"]
    for clock in ("base (PRIMARY)", "lock1710 (printed)"):
        c = res[clock]
        out.append(f"  {clock}: families {json.dumps(c['families'], default=str)}")
        for k, e in c["gemms"].items():
            if "sets" not in e:
                out.append(f"    {k}: {e.get('verdict')}")
                continue
            s = " ".join(f"{n} {v['slope']:.1f}" for n, v in e["sets"].items() if v)
            extra = "".join(f" {x} {e[x]:+.4f}" for x in ("theta", "delta", "delta_H", "delta_F") if x in e)
            out.append(f"    {k}: slopes {s}{extra}")
    return out + CM.addendum_lines(res)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    sys.path[:0] = [str(repo), str(repo / "scripts")]
    res = score(repo, tree)
    out.mkdir(parents=True, exist_ok=True)
    (out / "w1floor.score.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    text = "\n".join(lines(res)) + "\n"
    (out / "w1floor.score.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
