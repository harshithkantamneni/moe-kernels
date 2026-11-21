#!/usr/bin/env python
"""Score rental 7's S - N gap tests (docs/registered/2026-10-13-rental7-q1-gh200.json): Q1-H on its
registered control pairs (a REPLICATE, BLIND only in the copies clause), with the rivals scored
whatever the copies clause reads and the CR1 se on at least 3 clusters; Q1-C on D = x15 - x9
(rental 6's rule); Q1-P DESCRIPTIVE; Q1-SK OPEN. The P20 view is printed beside (DESCRIPTIVE).

    python scripts/scoring/rental7/score_q1.py <repo> <tree> <out>

Reads under <tree>: the skew pages gaps-<card>-<model>-ska/-skb/-skc (Mixtral and Phi: control
pairs at Mixtral n 6, Phi n 2 and n 4; Mixtral n 3 for Q1-P; Phi n 2 skewed pairs for Q1-SK) and
the Q1 pages gaps-<card>-<model>-q9 and -q15. A page's own align_probe prices A; a page with no
probe has its pairs NOT SCORED here. Writes <out>/q1.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7common as C7  # noqa: E402

C3, C5, C6 = C7.C3, C7.C5, C7.C6
PART = "q1"
SKEW_MODELS = ("mixtral-8x7b", "phi-3.5-moe")
r6 = C6.r6


def _pairkey(p: dict) -> tuple:
    return (p["model"], p["page"], p["n"], p["label"], p["copies"])


def score_view(tree: Path, regq: dict, regs: dict, tview, view: str = "registered") -> dict:
    preds = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in regs["predictions"] + regq["predictions"]}
    meas, A_of, use, dropped = {}, {}, {}, {}
    sk_files, q_files = regs["histograms"]["files"], regq["pages"]["files"]
    units = [(m, f"sk{p.lower()}", p, sk_files.get(f"{m}-{p}.json")) for m in SKEW_MODELS for p in ("A", "B", "C")]
    units += [(m, f"q{c}", f"c{c}", q_files.get(f"q1-{m}.json")) for m in regq["pages"]["models"] for c in (9, 15)]
    for model, label, page, f in units:
        rep, why = C7.page_report(tree, model, label, tview, f, view)
        use[f"{model}-{label}"] = why
        if rep is None:
            continue
        c, d, A = C6.valid_cells(rep, preds, model, page)
        dropped[f"{model}-{label}"] = d
        A_of[(model, page)] = A
        meas.update(c)

    def g_of(p: dict) -> dict | None:
        kS = (p["model"], p["page"], p["n"], p["label"], "shared")
        kN = kS[:4] + ("native",)
        A = A_of.get((p["model"], p["page"]), "missing")
        if A == "missing" or kS not in meas or kN not in meas:
            return None
        if A is None:
            return {"no_probe": True}
        prS, prN = preds[kS], preds[kN]
        v3 = (C6.carried(prS, "v3", A), C6.carried(prN, "v3", A))
        g = C6.gap_g_us(meas[kS], meas[kN], *v3)
        gR = {R: 1e3 * ((C6.carried(prS, R, A) - C6.carried(prN, R, A)) - (v3[0] - v3[1])) for R in p["g_R"]}
        return {"g": g, "g_R": gR}

    def rows_of(pairs: list) -> tuple[list, list]:
        rows, skipped = [], []
        for p in pairs:
            x = g_of(p)
            if x is None:
                skipped.append("/".join(map(str, _pairkey(p))) + ": a cell is missing or invalid")
            elif x.get("no_probe"):
                skipped.append("/".join(map(str, _pairkey(p))) + ": NOT SCORED (the page has no align_probe)")
            else:
                rows.append(dict(p, g=x["g"], g_R=x["g_R"]))
        return rows, skipped

    T = regq["tests"]
    out = {"pages": use, "dropped_rows": dropped}
    H, Hs = rows_of(T["Q1-H"]["pairs"])
    out["Q1-H"] = C7.q1h_rule(H, tuple(T["Q1-H"]["rivals"]))
    out["Q1-H"]["label"] = T["Q1-H"]["label"]
    out["Q1-H"]["skipped"] = Hs
    out["Q1-H"]["rows"] = [{"pair": "/".join(map(str, _pairkey(r))), "g_us": r6(r["g"]), "population": r["population"]} for r in H]
    design = [r for r in H if r["population"].startswith("the design")]
    out["Q1-H"]["design_population_printed"] = {k: v for k, v in C7.q1h_rule(design, tuple(T["Q1-H"]["rivals"])).items()
                                                if k != "verdict"} | {"reads_as": C7.q1h_rule(design, tuple(T["Q1-H"]["rivals"]))["verdict"],
                                                                      "label": "DESCRIPTIVE: the design's 12 pairs alone"}
    P, Ps = rows_of(T["Q1-P"]["pairs"])
    stat = C7.q1h_rule(P, tuple(T["Q1-H"]["rivals"]))
    out["Q1-P"] = {"verdict": "DESCRIPTIVE (no verdict, as registered)", "statistic": {k: v for k, v in stat.items() if k != "verdict"},
                   "skipped": Ps, "rows": [{"pair": "/".join(map(str, _pairkey(r))), "g_us": r6(r["g"])} for r in P]}
    SK, _ = rows_of(T["Q1-SK"]["pairs"])
    ctrl = [r["g"] for r in H if r["model"] == "phi-3.5-moe" and r["n"] == 2 and not r["q1page"]]
    skw = [r["g"] for r in SK]
    out["Q1-SK"] = {"status": "OPEN (printed)", "phi-3.5-moe n 2": ({"skew_minus_control_us": r6(st.fmean(skw) - st.fmean(ctrl)),
                                                                     "skewed_pairs": len(skw), "control_pairs": len(ctrl)}
                                                                    if skw and ctrl else "no valid pairs")}
    qc = T["Q1-C"]
    x = {}
    for c in (9, 15):
        p = {"model": "qwen1.5-moe-a2.7b", "page": f"c{c}", "n": 1, "label": "balanced", "copies": c, "g_R": {}}
        v = g_of(p)
        x[c] = None if (v is None or v.get("no_probe")) else v["g"]
    out["Q1-C"] = C6.q1c_rule(x[9], x[15], qc["dead_ratio"], qc["se_D_us"])
    return out


def score(repo: Path, tree: Path) -> dict:
    regq = C7.registration(repo, PART)
    regs = C7.registration(repo, "skew")
    res = C3.two_views(lambda fv, tv: score_view(tree, regq, regs, tv), repo=repo)
    p = C3.two_views(lambda fv, tv: score_view(tree, regq, regs, tv, "p20"), repo=repo)
    res["P20_view"] = {"label": "DESCRIPTIVE, NOT A VERDICT: every test on the P20 side view (rental7-g1rule)",
                       "combined": {k: v for k, v in p.items() if k != "addendum"}, "CLEAN": (p.get("addendum") or {}).get("CLEAN")}
    res.update(registration=regq["name"], part=PART)
    return res


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']} (ALL view; CLEAN and the P20 view beside it in the JSON)"]
    for k, v in res.get("pages", {}).items():
        out.append(f"  page {k}: {v}")
    h = res.get("Q1-H", {})
    out.append(f"Q1-H ({h.get('label', '')[:60]}...): {h.get('verdict')} mean g {h.get('mean_g_us')} us se {h.get('se_us')} "
               f"copies {((h.get('copies_clause') or {}).get('c15_minus_c9_us'))} ({h.get('pairs')} pairs, {h.get('clusters')} clusters)")
    for R, v in (h.get("rivals") or {}).items():
        out.append(f"  rival {R}: {v['verdict']} (g_R {v.get('g_R_us')} us)")
    out.append(f"Q1-P: {res.get('Q1-P', {}).get('verdict')}")
    out.append(f"Q1-SK: {res.get('Q1-SK')}")
    c = res.get("Q1-C", {})
    out.append(f"Q1-C: {c.get('verdict')} D {c.get('D_us')} us positions {c.get('positions_us')}")
    p = ((res.get("P20_view") or {}).get("combined") or {}).get("Q1-H") or {}
    out.append(f"P20 view (DESCRIPTIVE): Q1-H {p.get('verdict')}; Q1-C {(((res.get('P20_view') or {}).get('combined') or {}).get('Q1-C') or {}).get('verdict')}")
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
