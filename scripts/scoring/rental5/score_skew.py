#!/usr/bin/env python
"""Score rental 5's skew stage (docs/registered/2026-10-07-rental5-skew-gh200.json): the
gates G1 to G5 (G3 the TOST of the review), SKEW-RATIO with the rivals U, LT and PW,
E1-SKEW, E1-UNI-EXT, the permutation control, B-SKEW, E2-SKEW and the decomposition.

    python scripts/scoring/rental5/score_skew.py <repo> <tree> <out>

Reads under <tree>: the R3 histogram pages gaps-<card>-<model>-ska and -skb (one G = 8 page
each), and the byte-leg counter pages <date>-<card>-<model>-skc-r3-counters/lock1710/r3c-g8.json
(Mixtral, OLMoE). Writes <out>/skew.score.{json,txt}; exits 0 whatever the verdicts. Every
prediction, lever list, band and sha is the registration's; nothing is re-chosen on a page.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5common as C5  # noqa: E402

C3 = C5.C3
PART = "skew"
LOCK_LOW_MHZ = 1705.0
RANK = {"mixtral-8x7b": (2, 8, 32), "olmoe-1b-7b": (2, 4, 16), "qwen1.5-moe-a2.7b": (1, 2, 8)}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def page_report(tree: Path, model: str, label: str, tview, reg_file: dict | None):
    """The unit's one G = 8 histogram page, admitted by the view and by G5, or (None, why)."""
    root = C3.find_gaps(tree, model, label)
    pages = [(p, r) for p, r in C3.timed_pages(root) if C3.group_of(r) == 8]
    if len(pages) != 1:
        return None, f"NOT SCORED: {len(pages)} G = 8 pages under gaps-{model}-{label}"
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
        if k[0] != "balanced" and row.get("shuffle_seed") is None:
            return None, f"NOT SCORED (G5): row {k} records no shuffle seed"
    if (r.get("histogram_gates") or {}).get("G1_lock_thermal", {}).get("verdict") != "PASS":
        return None, "NOT SCORED (G1): the page's lock / thermal gate is not PASS"
    r = tview.admit(f"{model}-{label}", r)
    if r is None:
        return None, "NOT SCORED: the page is out of this view"
    return r, "counted"


def cells_of(report: dict) -> tuple[dict, list]:
    """{(label, n, arm): ms} of the valid rows (G2 host-bound and G4 below-lock rows out)."""
    out, dropped = {}, []
    for row in report.get("treads_table") or []:
        k = (str(row["histogram"]), int(row["tiles"]), str(row["arm"]))
        clk = row.get("sm_clock_load_mhz")
        if row.get("ms_p50") is None:
            dropped.append([*k, "no time"])
        elif row.get("host_bound"):
            dropped.append([*k, "G2 host-bound"])
        elif clk is None or float(clk) < LOCK_LOW_MHZ:
            dropped.append([*k, f"G4 clock {clk}"])
        else:
            out[k] = float(row["ms_p50"])
    return out, dropped


def _stats_verdict(v: list, rule: dict) -> dict:
    s = C5.stats(v)
    if not s["cells"]:
        return {"verdict": "NOT SCORED: no valid cell", "stats": s}
    ok = s["rms"] <= rule["rms_max"] and abs(s["worst"]) <= rule["max_abs"]
    if "mean_max" in rule:
        ok = ok and abs(s["mean"]) <= rule["mean_max"]
    return {"verdict": "HOLDS" if ok else "FALSIFIED", "stats": s}


def score_view(tree: Path, reg: dict, tview) -> dict:
    files = reg["histograms"]["files"]
    meas, pages_use, dropped = {}, {}, {}
    for model in RANK:
        for page in ("A", "B"):
            rep, why = page_report(tree, model, f"sk{page.lower()}", tview, files.get(f"{model}-{page}.json"))
            pages_use[f"{model}-{page}"] = why
            if rep is None:
                continue
            c, d = cells_of(rep)
            dropped[f"{model}-{page}"] = d
            for (lab, n, arm), ms in c.items():
                meas[(model, page, n, lab, arm)] = ms
    preds = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in reg["predictions"]}

    # ---- G3 per board, then the perm control --------------------------------
    g3rows = []
    for k, r in preds.items():
        if r["label"] != "uniform":
            continue
        b = k[:3] + ("balanced", k[4])
        if k in meas and b in meas:
            g3rows.append({"cluster": f"{k[0]}-{k[1]}", "stratum": RANK[k[0]].index(k[2]),
                           "d": meas[k] / meas[b] - 1 - (r.get("delta_M3") or 0.0),
                           "cell": "/".join(map(str, k)), "arm": k[4], "shape": k[0]})
    g3 = C5.g3_rule(g3rows)
    g3["d"] = [{"cell": x["cell"], "d": C5.r6(x["d"])} for x in g3rows]
    for key in ("shape", "arm"):
        g3[f"mean_by_{key}"] = {v: C5.r6(st.fmean([x["d"] for x in g3rows if x[key] == v]))
                                for v in sorted({x[key] for x in g3rows})}
    g3["at_sigma_alt"] = C5.g3_rule(g3rows, sigma=C5.SIGMA_PAGE_ALT)["verdict"]
    void = g3.get("void_strata")

    def voided(k) -> bool:
        if not g3rows:   # no control on the board: no histogram cell is scored (build-r5-review F6)
            return True
        if void == "ALL":
            return True
        return bool(void) and str(RANK[k[0]].index(k[2])) in [str(s) for s in void]

    perm_rows = []
    for k, r in preds.items():
        if r.get("perm_dr_S") is None:
            continue
        pw = k[:3] + ("PW", k[4])
        if k in meas and pw in meas:
            perm_rows.append({"cluster": f"{k[0]}-{k[1]}", "stratum": RANK[k[0]].index(k[2]),
                              "d": (meas[k] / meas[pw] - 1) - r["perm_dr_S"], "cell": "/".join(map(str, k))})
    perm = C5.g3_rule(perm_rows)
    perm["d"] = [{"cell": x["cell"], "d": C5.r6(x["d"])} for x in perm_rows]

    # ---- SKEW-RATIO ------------------------------------------------------------
    T = reg["tests"]["SKEW-RATIO"]
    rows = []
    for k, r in preds.items():
        if "ratio" not in r or voided(k):
            continue
        u = k[:3] + ("uniform", k[4])
        if k not in meas or u not in meas:
            continue
        rm = meas[k] / meas[u]
        rows.append({"cell": "/".join(map(str, k)), "model": k[0], "arm": k[4], "r_meas": C5.r6(rm),
                     "e": {x: r["ratio"][x] / rm - 1 for x in ("S", "U", "LT", "PW", "M1")},
                     "lever": r.get("lever", {})})
    eS = [x["e"]["S"] for x in rows]
    sS = C5.stats(eS)
    ratio = {"cells": len(rows)}
    if not rows:
        ratio["S"] = "NOT SCORED: no valid ratio"
    else:
        ok = sS["rms"] <= T["rms_max"] and max(abs(v) for v in eS) <= T["max_abs"]
        ratio["S"] = {"verdict": "HOLDS" if ok else "FAILS", "stats": sS}
    rivals = {}
    for rv in ("U", "PW", "LT"):
        lev = [x for x in rows if x["lever"].get(rv)]
        if len(lev) < T["min_lever_cells"]:
            rivals[rv] = {"verdict": f"UNDECIDED: {len(lev)} lever cells survive, under {T['min_lever_cells']}"}
            continue
        r_riv = math.sqrt(st.fmean(x["e"][rv] ** 2 for x in lev))
        r_s = math.sqrt(st.fmean(x["e"]["S"] ** 2 for x in lev))
        exc = r_riv >= T["rival_factor"] * r_s and r_riv >= T["rival_floor"]
        rivals[rv] = {"verdict": "EXCLUDED" if exc else "NOT EXCLUDED", "lever_cells": len(lev),
                      "rms_rival": C5.r6(r_riv), "rms_S_on_levers": C5.r6(r_s)}
    rivals["M1"] = {"printed": C5.stats(x["e"]["M1"] for x in rows)}
    ratio["rivals"] = rivals
    ratio["per_arm"] = {a: C5.stats(x["e"]["S"] for x in rows if x["arm"] == a) for a in ("native", "shared")}
    ratio["per_shape"] = {m: C5.stats(x["e"]["S"] for x in rows if x["model"] == m) for m in RANK}
    ratio["rows"] = [dict(x, e={k: C5.r6(v) for k, v in x["e"].items()}) for x in rows]

    # ---- E1-SKEW, E1-UNI(-EXT) ----------------------------------------------------
    e1, uni, ext, sec_rows = [], [], [], []
    for k, r in preds.items():
        if k not in meas or r["ms"]["S"] is None:
            continue
        e = r["ms"]["S"] / meas[k] - 1
        if r["label"] == "balanced":
            (ext if r["ext_tread"] else uni).append(e)
            continue
        if voided(k):
            continue
        e1.append(e)
        if r["label"] in ("PT", "PW", "DW", "uniform"):
            sec_rows.append({"pred": r["ms"]["S"], "meas": meas[k], "cluster": k[0],
                             "stratum": f"{k[0]}/{k[4]}/n{k[2]}", "page": f"{k[0]}-{k[1]}"})
    E1 = _stats_verdict(e1, reg["tests"]["E1-SKEW"])
    E1x = _stats_verdict(ext, reg["tests"]["E1-SKEW"])
    E1u = _stats_verdict(uni, reg["tests"]["E1-SKEW"])
    sec = C5.secondary(sec_rows)
    gam = C5.gamma([dict(x, cluster=x["page"]) for x in sec_rows])
    sec["gamma"] = gam

    # ---- the byte leg ------------------------------------------------------------
    byte = byte_leg(tree, reg, meas, voided)
    if not g3rows:
        why = "NOT SCORED: no G3 row on the board, so the histogram cells have no control (F6)"
        for t in ("B-SKEW", "E2-SKEW"):
            byte[t] = {"verdict": why}
        perm = dict(perm, verdict=why)
    if not g3rows:
        ratio["S"] = "NOT SCORED: no G3 row on the board, so the histogram cells have no control (F6)"
    return {"pages": pages_use, "dropped_rows": dropped, "cells_measured": len(meas),
            "G3": g3, "perm_control": perm, "SKEW-RATIO": ratio, "E1-SKEW": E1,
            "E1-UNI-EXT": E1x, "E1-UNI": E1u, "secondary": sec, **byte}


_ENGINE = []


def _engine():
    """The registered fits (skewmodel.Engine), built once per process."""
    if not _ENGINE:
        import skewmodel as SM
        _ENGINE.append(SM.Engine())
    return _ENGINE[0]


#: the byte leg's clock (build-r5-review F5): SM cycles over duration on GEMMs this long or longer
MIN_CLOCK_NS = 500_000


def gemm_mhz(cell: dict, g: str) -> tuple[float | None, float | None]:
    """(MHz, duration ns) of one (cell, GEMM): recorded sm__cycles_elapsed.avg / gpu_time_ns."""
    cyc = ((cell.get("recorded") or {}).get(g) or {}).get("sm__cycles_elapsed.avg")
    ns = ((cell.get("per_gemm") or {}).get(g) or {}).get("gpu_time_ns")
    if not cyc or not ns:
        return None, ns
    return float(cyc) / float(ns) * 1e3, float(ns)


def cell_clock(cell: dict, page_mhz: float | None) -> float | None:
    """The cell's SM clock off its GEMMs of at least MIN_CLOCK_NS; else the page's median."""
    xs = [m for g in ("w1", "w2") for m, ns in [gemm_mhz(cell, g)] if m and ns and ns >= MIN_CLOCK_NS]
    return st.median(xs) if xs else page_mhz


def counter_page(tree: Path, model: str, reg_file: dict | None):
    hits = sorted(Path(tree).rglob(f"*-{C5.CARD}-{model}-skc-r3-counters/lock1710/r3c-g8.json"))
    if len(hits) != 1:
        return None, f"NOT SCORED: {len(hits)} byte-leg pages for {model}"
    p = json.loads(hits[0].read_text())
    h = (p.get("design") or {}).get("histogram") or {}
    if reg_file is None or h.get("file_sha256") != reg_file["sha256"]:
        return None, "NOT SCORED (G5): the byte page's histogram sha256 is not the registered one"
    if h.get("shuffle_seed") is None or h.get("shuffle_seed") != reg_file.get("shuffle_seed"):
        return None, "NOT SCORED (G5): the byte page records no shuffle seed, or not the registered one"
    return p, "counted"


def byte_leg(tree: Path, reg: dict, meas: dict, voided=lambda k: False) -> dict:
    import r3_timing_model as TM
    files = reg["histograms"]["files"]
    bpred = {(r["model"], r["n"], r["label"], r["arm"]): r for r in reg["byte_predictions"]}
    pins = reg.get("code_pins") or {}
    code_ok = all(_sha(C5.REPO / f) == pins.get(f) for f in
                  ("scripts/r3_timing_model.py", "scripts/wave_split_bytes.py", "scripts/scoring/rental5/skewmodel.py"))
    B, E2, dec = [], [], []
    use = {}
    eng = None
    for model in ("mixtral-8x7b", "olmoe-1b-7b"):
        page, why = counter_page(tree, model, files.get(f"{model}-C.json"))
        use[model] = why
        if page is None:
            continue
        doc = json.loads((C5.REPO / files[f"{model}-C.json"]["path"]).read_text())
        counts = {(c["label"], c["n"]): c["counts"] for c in doc["cells"]}
        want = {(c["label"], c["n"]): c["counts_sha256"] for c in files[f"{model}-C.json"]["cells"]}
        bad = [f"{c.get('histogram')}/{c.get('arm')}/n{c.get('n')}" for c in page.get("cells") or []
               if want.get((str(c.get("histogram")), int(c["n"]))) != c.get("counts_sha256")
               or (counts.get((str(c.get("histogram")), int(c["n"]))) is not None
                   and list(c.get("bincount") or []) != list(counts[(str(c.get("histogram")), int(c["n"]))]))]
        if bad:
            use[model] = f"NOT SCORED (G5): counter cells {bad[:4]} carry another counts sha256 or bincount"
            continue
        long_ = [m for c in page.get("cells") or [] for g in ("w1", "w2")
                 for m, ns in [gemm_mhz(c, g)] if m and ns and ns >= MIN_CLOCK_NS]
        page_mhz = st.median(long_) if long_ else None
        if page_mhz is None:
            use[model] = f"NOT SCORED (G4): no GEMM of {MIN_CLOCK_NS / 1e6:g} ms or longer to read the clock off"
            continue
        dropped = use.setdefault(f"{model}-dropped", [])
        old = TM.set_model(model)
        try:
            bm = dict(TM.BYTE_MODEL)
        finally:
            TM.set_model(old)
        for cell in page.get("cells") or []:
            lab, n, arm = str(cell.get("histogram")), int(cell["n"]), str(cell["arm"])
            pr = bpred.get((model, n, lab, arm))
            if pr is None or pr["q_pred"] is None:
                continue
            if n in RANK[model] and voided((model, "C", n, lab, arm)):
                dropped.append(f"{lab}/{arm}/n{n}: G3 void")
                continue
            mhz = cell_clock(cell, page_mhz)
            if mhz is None or mhz < LOCK_LOW_MHZ:
                dropped.append(f"{lab}/{arm}/n{n}: G4 clock {mhz}")
                continue
            reads = {g: float(cell["per_gemm"][g]["dram_bytes_read"]) for g in ("w1", "w2")}
            qc = {g: (reads[g] - n * bm[f"operand_per_tile_{g}"]) / bm[f"W_{g}"] for g in reads}
            for g in reads:
                B.append({"cell": f"{model}/n{n}/{lab}/{arm}", "gemm": g, "skewed": lab in ("PT", "PW"),
                          "e": pr["q_pred"][g] / qc[g] - 1})
            k = None
            for pg in ("A", "B"):
                if (model, pg, n, lab, arm) in meas:
                    k = (model, pg, n, lab, arm)
                    break
            if k is None or not code_ok:
                continue
            if eng is None:
                eng = _engine()
            t_c = eng.price_counted(model, arm, n, counts.get((lab, n)), reads)
            tp = next(r for r in reg["predictions"] if (r["model"], r["page"], r["n"], r["label"], r["arm"]) == k)
            E2.append(t_c / meas[k] - 1)
            dec.append({"cell": "/".join(map(str, k)), "ln_E1": C5.r6(math.log(tp["ms"]["S"] / meas[k])),
                        "ln_bytes": C5.r6(math.log(tp["ms"]["S"] / t_c)), "ln_E2": C5.r6(math.log(t_c / meas[k]))})
    rb = reg["tests"]["B-SKEW"]
    sk = [x["e"] for x in B if x["skewed"]]
    if not sk:
        bverdict = {"verdict": "NOT SCORED: no skewed byte cell"}
    else:
        s = C5.stats(sk)
        ok = max(abs(v) for v in sk) <= rb["max_abs"] and s["rms"] <= rb["rms_max"]
        bverdict = {"verdict": "HOLDS" if ok else "FALSIFIED", "stats": s,
                    "control_uniform_balanced": C5.stats(x["e"] for x in B if not x["skewed"])}
    bverdict["rows"] = [dict(x, e=C5.r6(x["e"])) for x in B]
    e2v = (_stats_verdict(E2, reg["tests"]["E2-SKEW"]) if code_ok else
           {"verdict": "NOT SCORED: the timing or byte code differs from the registration's (sha256)"})
    return {"byte_pages": use, "B-SKEW": bverdict, "E2-SKEW": e2v, "DECOMPOSITION": dec}


def score(repo: Path, tree: Path) -> dict:
    reg = C5.registration(repo, PART)
    res = C3.two_views(lambda fv, tv: score_view(tree, reg, tv), repo=repo)
    res.update(registration=reg["name"], part=PART)
    return res


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']} (ALL view; CLEAN beside it in the JSON)"]
    for k, v in res.get("pages", {}).items():
        out.append(f"  page {k}: {v}")
    g3 = res.get("G3", {})
    out.append(f"G3: {g3.get('verdict')}; pooled {g3.get('pooled', {}).get('verdict')} CI {g3.get('pooled', {}).get('ci')}; "
               f"Q p {g3.get('Q', {}).get('p')}; at sigma 0.10% {g3.get('at_sigma_alt')}")
    out.append(f"perm control: {res.get('perm_control', {}).get('verdict')}")
    r = res.get("SKEW-RATIO", {})
    out.append(f"SKEW-RATIO S: {r.get('S') if isinstance(r.get('S'), str) else r.get('S', {}).get('verdict')} ({r.get('cells')} ratios)")
    for k, v in (r.get("rivals") or {}).items():
        out.append(f"  rival {k}: {v.get('verdict', v)}")
    for t in ("E1-SKEW", "E1-UNI-EXT", "E1-UNI", "B-SKEW", "E2-SKEW"):
        v = res.get(t, {})
        out.append(f"{t}: {v.get('verdict')} {v.get('stats', '')}")
    out += C5.secondary_lines(res["secondary"]) if res.get("secondary") else []
    out += C3.view_lines(res)
    return out


def main(argv=None) -> int:
    a = (argv if argv is not None else sys.argv[1:])
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C5.write_score(out, PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
