#!/usr/bin/env python
"""Score rental 6's skew stage (docs/registered/2026-10-09-rental6-skew-gh200.json): the gates
(G1 with slipped repeats, 6 of 9 clean, G2 host rule, G4, G5, G3 TOST), E1-SKEW at the time bar,
E1-UNI, the E1 rivals, SKEW-RATIO, and the byte legs (B-SKEW, OPEN-L2SKEW, B-CA, E2-SKEW,
DECOMPOSITION) under the counter-page clock rule.

    python scripts/scoring/rental6/score_skew.py <repo> <tree> <out>

Reads under <tree>: the R3 histogram pages gaps-<card>-<model>-ska, -skb, -skc (one G = 8 page
each; Mixtral, OLMoE, Phi), and the byte pages <date>-<card>-<model>-skbytes-r3-counters/
lock1710/r3c-g8.json (OLMoE, Phi). Writes <out>/skew.score.{json,txt}; exits 0 whatever the
verdicts. Every prediction, lever list, band and sha is the registration's; nothing is re-chosen
on a page. A prediction is carried to the page's own A (r6model.reprice) before it is compared.
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
import r6common as C6  # noqa: E402

C3, C5 = C6.C3, C6.C5
PART = "skew"
RANK = {"mixtral-8x7b": (3, 6, 16), "olmoe-1b-7b": (3, 6, 12), "phi-3.5-moe": (2, 4, 12)}
PAGES = ("A", "B", "C")
BYTE_MODELS = ("olmoe-1b-7b", "phi-3.5-moe")
SKEWED = ("PT", "PW", "PW2", "DW")
r6 = C6.r6


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _verdict(v: list, rule: dict) -> dict:
    s = C5.stats(v)
    if not s["cells"]:
        return {"verdict": "NOT SCORED: no valid cell", "stats": s}
    ok = s["rms"] <= rule["rms_max"] and abs(s["worst"]) <= rule["max_abs"]
    if "mean_max" in rule:
        ok = ok and abs(s["mean"]) <= rule["mean_max"]
    return {"verdict": "HOLDS" if ok else "FALSIFIED", "stats": s}


DESCRIPTIVE_LABEL = "DESCRIPTIVE, NOT A VERDICT: E1 on the G3-voided histogram cells (owner decision 2026-10-09)"


def e1_descriptive(rows: list, rule: dict) -> dict:
    """E1 printed on the cells a G3 failure voided (owner decision 2026-10-09): the same
    statistic and bars, labelled DESCRIPTIVE; it never becomes E1-SKEW's verdict."""
    v = _verdict([x["e"] for x in rows], rule)
    return {"label": DESCRIPTIVE_LABEL, "reads_as": v["verdict"].replace("HOLDS", "would hold").replace(
                "FALSIFIED", "would not hold"), "stats": v["stats"],
            "per_shape": {m: C5.stats(x["e"] for x in rows if x["model"] == m) for m in RANK},
            "per_arm": {a: C5.stats(x["e"] for x in rows if x["arm"] == a) for a in ("native", "shared")}}


def score_view(tree: Path, reg: dict, tview) -> dict:
    files = reg["histograms"]["files"]
    preds = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in reg["predictions"]}
    meas, A_of, use, dropped, slips = {}, {}, {}, {}, {}
    for model in RANK:
        for page in PAGES:
            rep, why = C6.page_report(tree, model, f"sk{page.lower()}", tview, files.get(f"{model}-{page}.json"))
            use[f"{model}-{page}"] = why
            if rep is None:
                continue
            slips[f"{model}-{page}"] = C6.page_slips(rep)
            c, d, A = C6.valid_cells(rep, preds, model, page)
            dropped[f"{model}-{page}"] = d
            A_of[(model, page)] = A
            meas.update(c)

    def pred(k, v="v3"):
        return C6.carried(preds[k], v, A_of.get(k[:2]))
    no_probe = sorted(f"{m}-{p}" for (m, p), A in A_of.items() if A is None)

    # ---- G3 -------------------------------------------------------------------------
    g3rows = []
    for k, r in preds.items():
        if r["label"] != "uniform":
            continue
        b = k[:3] + ("balanced", k[4])
        if k in meas and b in meas:
            g3rows.append({"cluster": f"{k[0]}-{k[1]}", "stratum": RANK[k[0]].index(k[2]),
                           "d": meas[k] / meas[b] - 1 - (r.get("delta_v3") or 0.0),
                           "cell": "/".join(map(str, k)), "arm": k[4], "shape": k[0]})
    g3 = C5.g3_rule(g3rows)
    g3["d"] = [{"cell": x["cell"], "d": r6(x["d"])} for x in g3rows]
    g3["mean_by_shape"] = {m: r6(st.fmean([x["d"] for x in g3rows if x["shape"] == m]))
                           for m in sorted({x["shape"] for x in g3rows})}
    void = g3.get("void_strata")

    def voided(k) -> bool:
        if not g3rows or void == "ALL":
            return True
        return bool(void) and str(RANK[k[0]].index(k[2])) in [str(s) for s in void]

    # ---- E1-SKEW, E1-UNI, rivals -------------------------------------------------------
    e1, uni, rows, sec_rows, e1_void = [], [], [], [], []
    for k, r in preds.items():
        if k not in meas:
            continue
        e = pred(k) / meas[k] - 1
        if r["label"] == "balanced":
            uni.append(e)
            continue
        if voided(k):
            if void == "ALL" and g3rows:
                e1_void.append({"model": k[0], "arm": k[4], "e": e})
            continue
        e1.append(e)
        rows.append({"cell": "/".join(map(str, k)), "model": k[0], "arm": k[4], "e": e,
                     "eR": {R: pred(k, R) / meas[k] - 1 for R in reg["tests"]["E1-rivals"]["rivals"]},
                     "lever": r["lever"]})
        sec_rows.append({"pred": pred(k), "meas": meas[k], "cluster": k[0], "stratum": f"{k[0]}/{k[4]}/n{k[2]}",
                         "page": f"{k[0]}-{k[1]}"})
    E1 = _verdict(e1, reg["tests"]["E1-SKEW"])
    E1["rental5_bands"] = _verdict(e1, reg["tests"]["E1-SKEW"]["rental5_bands_printed"])["verdict"]
    E1["per_shape"] = {m: C5.stats(x["e"] for x in rows if x["model"] == m) for m in RANK}
    E1["per_arm"] = {a: C5.stats(x["e"] for x in rows if x["arm"] == a) for a in ("native", "shared")}
    E1["no_probe_pages"] = no_probe
    if void == "ALL" and g3rows:
        E1 = {"verdict": "NOT SCORED: G3 FAILS and Q does not reject, so every histogram cell is voided (strict rule kept)",
              "no_probe_pages": no_probe, "descriptive": e1_descriptive(e1_void, reg["tests"]["E1-SKEW"])}
    T = reg["tests"]["E1-rivals"]
    riv = {}
    for R in T["rivals"]:
        lev = [x for x in rows if x["lever"].get(R)]
        if len(lev) < T["min_lever_cells"]:
            riv[R] = {"verdict": f"UNDECIDED: {len(lev)} lever cells survive, under {T['min_lever_cells']}"}
            continue
        rr = math.sqrt(st.fmean(x["eR"][R] ** 2 for x in lev))
        rv = math.sqrt(st.fmean(x["e"] ** 2 for x in lev))
        exc = rr >= T["rival_factor"] * rv and rr >= T["rival_floor"]
        riv[R] = {"verdict": "EXCLUDED" if exc else "NOT EXCLUDED", "lever_cells": len(lev),
                  "rms_rival": r6(rr), "rms_v3_on_levers": r6(rv)}
    E1u = _verdict(uni, reg["tests"]["E1-SKEW"])

    # ---- SKEW-RATIO -----------------------------------------------------------------------
    TR = reg["tests"]["SKEW-RATIO"]
    rat = []
    for k, r in preds.items():
        if "ratio" not in r or voided(k):
            continue
        u = k[:3] + ("uniform", k[4])
        if k in meas and u in meas:
            rm = meas[k] / meas[u]
            rat.append({"cell": "/".join(map(str, k)), "e": r["ratio"]["v3"] / rm - 1,
                        "e_LT3": r["ratio"]["LT3"] / rm - 1, "e_U": 1.0 / rm - 1, "lever": r["lever_ratio"]})
    ratio = {"cells": len(rat)}
    if rat:
        s = C5.stats([x["e"] for x in rat])
        ratio["v3"] = {"verdict": "HOLDS" if (s["rms"] <= TR["rms_max"] and abs(s["worst"]) <= TR["max_abs"]) else "FAILS", "stats": s}
        for R, key in (("LT3", "e_LT3"), ("U", "e_U")):
            lev = [x for x in rat if x["lever"].get(R)]
            if len(lev) < C6.MIN_LEVER:
                ratio[R] = {"verdict": f"UNDECIDED: {len(lev)} lever cells"}
                continue
            rr = math.sqrt(st.fmean(x[key] ** 2 for x in lev))
            rv = math.sqrt(st.fmean(x["e"] ** 2 for x in lev))
            ratio[R] = {"verdict": "EXCLUDED" if (rr >= C6.RIVAL_FACTOR * rv and rr >= C6.RIVAL_FLOOR) else "NOT EXCLUDED",
                        "lever_cells": len(lev), "rms_rival": r6(rr), "rms_v3_on_levers": r6(rv)}
    else:
        ratio["v3"] = {"verdict": "NOT SCORED: no valid ratio"}

    byte = byte_leg(tree, reg, meas, A_of, voided)
    sec = C5.secondary(sec_rows)
    out = {"pages": use, "slips": slips, "dropped_rows": dropped, "cells_measured": len(meas), "G3": g3,
           "E1-SKEW": E1, "E1-rivals": riv, "E1-UNI": E1u, "SKEW-RATIO": ratio, "secondary": sec, **byte}
    if not g3rows:
        why = "NOT SCORED: no G3 row on the board, so the histogram cells have no control"
        for t in ("E1-SKEW", "B-SKEW", "E2-SKEW"):
            out[t] = {"verdict": why}
    return out


# --------------------------------------------------------------------------
# the byte legs
# --------------------------------------------------------------------------

_PRICER = []


def _pricer():
    if not _PRICER:
        import r6model as M6
        _PRICER.append(M6.Pricer(h8=False))
    return _PRICER[0]


def counter_page(tree: Path, model: str, reg_file: dict | None):
    hits = sorted(Path(tree).rglob(f"*-{C6.CARD}-{model}-skbytes-r3-counters/lock1710/r3c-g8.json"))
    if len(hits) != 1:
        return None, None, f"NOT SCORED: {len(hits)} byte pages for {model}"
    p = json.loads(hits[0].read_text())
    h = (p.get("design") or {}).get("histogram") or {}
    if reg_file is None or h.get("file_sha256") != reg_file["sha256"]:
        return None, None, "NOT SCORED (G5): the byte page's histogram sha256 is not the registered one"
    if h.get("shuffle_seed") != reg_file.get("shuffle_seed"):
        return None, None, "NOT SCORED (G5): the byte page records another shuffle seed"
    return p, hits[0], "counted"


def byte_leg(tree: Path, reg: dict, meas: dict, A_of: dict, voided=lambda k: False) -> dict:
    import r3_timing_model as TM
    files = reg["histograms"]["files"]
    bpred = {(r["model"], r["n"], r["label"], r["arm"]): r for r in reg["byte_predictions"]}
    tpred = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in reg["predictions"]}
    pins = reg.get("code_pins") or {}
    code_ok = all(_sha(C6.REPO / f) == pins.get(f) for f in
                  ("scripts/r3_timing_model.py", "scripts/wave_split_bytes.py", "scripts/scoring/rental5/skewmodel.py",
                   "scripts/r3_timing_model_v3.py", "scripts/wave_split_bytes_v3.py", "scripts/scoring/rental6/r6model.py"))
    B, E2, dec, use = [], [], [], {}
    for model in BYTE_MODELS:
        page, path, why = counter_page(tree, model, files.get(f"{model}-bytes.json"))
        use[model] = why
        if page is None:
            continue
        doc = json.loads((C6.REPO / files[f"{model}-bytes.json"]["path"]).read_text())
        counts = {(c["label"], c["n"]): c["counts"] for c in doc["cells"]}
        want = {(c["label"], c["n"]): c["counts_sha256"] for c in files[f"{model}-bytes.json"]["cells"]}
        bad = [f"{c.get('histogram')}/{c.get('arm')}/n{c.get('n')}" for c in page.get("cells") or []
               if want.get((str(c.get("histogram")), int(c["n"]))) != c.get("counts_sha256")]
        if bad:
            use[model] = f"NOT SCORED (G5): counter cells {bad[:4]} carry another counts sha256"
            continue
        clk = C6.clock_rule(page, path)
        use[f"{model}-clock"] = {"check_page": clk["check_page"], "cells_in_band": clk["cells_ok"], "cells": len(clk["cells"])}
        if not clk["page_ok"]:
            use[model] = f"NOT SCORED (clock rule): {clk['why']}"
            continue
        old = TM.set_model(model)
        bm = dict(TM.BYTE_MODEL)
        TM.set_model(old)
        dropped = use.setdefault(f"{model}-dropped", [])
        for cell, cc in zip(page.get("cells") or [], clk["cells"], strict=True):
            lab, n, arm = str(cell.get("histogram")), int(cell["n"]), str(cell["arm"])
            pr = bpred.get((model, n, lab, arm))
            if pr is None:
                continue
            if not cc["ok"]:
                dropped.append(f"{lab}/{arm}/n{n}: clock {cc['mhz']} outside {C6.BAND_MHZ}")
                continue
            reads = {g: float(cell["per_gemm"][g]["dram_bytes_read"]) for g in ("w1", "w2")}
            qc = {g: (reads[g] - n * bm[f"operand_per_tile_{g}"]) / bm[f"W_{g}"] for g in reads}
            for g in reads:
                row = {"cell": f"{model}/n{n}/{lab}/{arm}", "model": model, "gemm": g, "skewed": lab in ("PT", "PW"),
                       "label": lab, "e": pr["q_pred"][g] / qc[g] - 1, "e_noca": pr["q_pred_v3_noca"][g] / qc[g] - 1}
                if g == "w1" and pr.get("rowkey_w1_q"):
                    row["e_rowkey"] = {C: q / qc[g] - 1 for C, q in pr["rowkey_w1_q"].items()}
                B.append(row)
            k = (model, "A", n, lab, arm)
            if k not in meas or not code_ok:
                continue
            A = A_of.get((model, "A"))
            a_page = None if A is None else A.get((arm, n))
            t_c = _pricer().price_counted(model, arm, n, counts.get((lab, n)), reads, 9, a_page)
            tv3 = C6.carried(tpred[k], "v3", A)
            E2.append(t_c / meas[k] - 1)
            dec.append({"cell": "/".join(map(str, k)), "ln_E1": r6(math.log(tv3 / meas[k])),
                        "ln_bytes": r6(math.log(tv3 / t_c)), "ln_E2": r6(math.log(t_c / meas[k]))})
    rb = reg["tests"]["B-SKEW"]
    open_rows = [x for x in B if x["skewed"] and x["model"] == "olmoe-1b-7b" and x["gemm"] == "w1"]
    sk = [x for x in B if x["skewed"] and x not in open_rows]
    if not sk:
        bverdict = {"verdict": "NOT SCORED: no skewed byte row"}
    else:
        s = C5.stats([x["e"] for x in sk])
        ok = max(abs(x["e"]) for x in sk) <= rb["max_abs"] and s["rms"] <= rb["rms_max"]
        bverdict = {"verdict": "HOLDS" if ok else "FALSIFIED", "stats": s,
                    "per_shape_gemm": {f"{m}/{g}": C5.stats(x["e"] for x in sk if x["model"] == m and x["gemm"] == g)
                                       for m in BYTE_MODELS for g in ("w1", "w2")}}
    # OPEN-L2SKEW
    if not open_rows:
        l2 = {"verdict": "NOT SCORED: no OLMoE w1 skewed row"}
    else:
        ev = [x["e"] for x in open_rows]
        s = C5.stats(ev)
        v3ok = max(abs(e) for e in ev) <= rb["max_abs"] and s["rms"] <= rb["rms_max"]
        l2 = {"status": "OPEN", "v3": "HOLDS" if v3ok else "FAILS", "v3_stats": s}
        rk = [x["e_rowkey"]["C39.5"] for x in open_rows if x.get("e_rowkey")]
        if rk:
            sr = C5.stats(rk)
            better = max(abs(e) for e in rk) <= rb["max_abs"] and sr["rms"] <= 0.5 * s["rms"]
            l2["ROWKEY"] = {"verdict": "BETTER" if better else "NOT BETTER", "stats": sr,
                            "C60_printed": C5.stats(x["e_rowkey"]["C60"] for x in open_rows if x.get("e_rowkey"))}
        l2["rows"] = [{"cell": x["cell"], "e_v3": r6(x["e"]), "e_rowkey": {k: r6(v) for k, v in (x.get("e_rowkey") or {}).items()}}
                      for x in open_rows]
    # B-CA
    ca = [x for x in B if x["gemm"] == "w1" and x["label"] in ("uniform", "balanced")]
    if not ca:
        bca = {"verdict": "NOT SCORED: no w1 uniform / balanced row"}
    else:
        rc = reg["tests"]["B-CA"]
        sv, sn = C5.stats([x["e"] for x in ca]), C5.stats([x["e_noca"] for x in ca])
        bca = {"v3": "HOLDS" if max(abs(x["e"]) for x in ca) <= rc["max_abs"] else "FAILS", "v3_stats": sv,
               "v3_noca": ("EXCLUDED" if (sn["rms"] >= rc["noca_factor"] * sv["rms"] and sn["rms"] >= rc["noca_floor"]) else "NOT EXCLUDED"),
               "v3_noca_stats": sn, "per_shape": {m: C5.stats(x["e"] for x in ca if x["model"] == m) for m in BYTE_MODELS}}
    e2v = (_verdict(E2, reg["tests"]["E2-SKEW"]) if code_ok else
           {"verdict": "NOT SCORED: the timing or byte code differs from the registration's (sha256)"})
    return {"byte_pages": use, "B-SKEW": bverdict, "OPEN-L2SKEW": l2, "B-CA": bca, "E2-SKEW": e2v,
            "DECOMPOSITION": dec, "byte_rows": [dict(x, e=r6(x["e"]), e_noca=r6(x["e_noca"])) for x in B]}


def score(repo: Path, tree: Path) -> dict:
    reg = C6.registration(repo, PART)
    res = C3.two_views(lambda fv, tv: score_view(tree, reg, tv), repo=repo)
    res.update(registration=reg["name"], part=PART)
    return res


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']} (ALL view; CLEAN beside it in the JSON)"]
    for k, v in res.get("pages", {}).items():
        out.append(f"  page {k}: {v}")
    for k, v in (res.get("slips") or {}).items():
        if v["slipped_rows"]:
            out.append(f"  slips {k}: {v['slipped_rows']} of {v['repeat_rows']}: " +
                       ", ".join(f"{x['cell']} r{x['repeat']} {x['mhz']} MHz" for x in v["slipped"]))
    g3 = res.get("G3", {})
    out.append(f"G3: {g3.get('verdict')}; pooled {g3.get('pooled', {}).get('verdict')} CI {g3.get('pooled', {}).get('ci')}")
    for t in ("E1-SKEW", "E1-UNI", "E2-SKEW", "B-SKEW"):
        v = res.get(t, {})
        out.append(f"{t}: {v.get('verdict')} {v.get('stats', '')}")
        if t == "E1-SKEW" and v.get("descriptive"):
            d = v["descriptive"]
            out.append(f"  E1 {d['label']}: {d['reads_as']} {d['stats']}")
    for R, v in (res.get("E1-rivals") or {}).items():
        out.append(f"  E1 rival {R}: {v.get('verdict')}")
    r = res.get("SKEW-RATIO", {})
    out.append(f"SKEW-RATIO v3: {(r.get('v3') or {}).get('verdict')} ({r.get('cells')} ratios); LT3 {(r.get('LT3') or {}).get('verdict')}; U {(r.get('U') or {}).get('verdict')}")
    l2 = res.get("OPEN-L2SKEW", {})
    out.append(f"OPEN-L2SKEW: {l2.get('status', l2.get('verdict'))} v3 {l2.get('v3')} rowkey {(l2.get('ROWKEY') or {}).get('verdict')}")
    b = res.get("B-CA", {})
    out.append(f"B-CA: v3 {b.get('v3', b.get('verdict'))}; v3_noca {b.get('v3_noca')}")
    out += C5.secondary_lines(res["secondary"]) if res.get("secondary") else []
    out += C3.view_lines(res)
    return out


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C3.write_score(out, PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
