#!/usr/bin/env python
"""Score rental 4's dead-CTA copies contrast (docs/registered/2026-10-06-rental4-dead-gh200.json):
A1 (qwen2-57b-a14b-tp8) and A2 (olmoe-1b-7b), 15 copies against 9, and the NATIVE null control.

    python scripts/scoring/rental4/score_dead.py <repo> <tree> <out>

Reads gaps-<card>-<model>-c9 and -c15 under <tree> (the G = 8 page of each). Writes
<out>/dead.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4common as C4  # noqa: E402

C3 = C4.C3
PART = "dead"


def _page(tree: Path, model: str, label: str, tview, want_copies: int):
    root = C3.find_gaps(tree, model, label)
    pages = [(p, r) for p, r in C3.timed_pages(root) if C3.group_of(r) == 8]
    if len(pages) != 1:
        return None, f"NOT SCORED: {len(pages)} G = 8 pages under gaps-{model}-{label}"
    p, r = pages[0]
    got = int(r.get("copies_declared") or 0)
    if got != want_copies:
        return None, f"NOT SCORED: the page declares {got} copies, not {want_copies}"
    r = tview.admit(f"{model}-{label}", r)
    if r is None:
        return None, "NOT SCORED: the page is out of this view"
    return r, "counted"


def contrast(tree: Path, reg: dict, name: str, tview) -> dict:
    pr = reg["predictions_us"][name]
    model, (c9, c15) = pr["model"], pr["copies"]
    p9, u9 = _page(tree, model, "c9", tview, c9)
    p15, u15 = _page(tree, model, f"c{c15}", tview, c15)
    out = {"model": model, "pages": {"c9": u9, f"c{c15}": u15},
           "label": f"the c{c15} page is NOT JOINABLE TO COUNTER BYTES"}
    if p9 is None or p15 is None:
        out["verdict"] = "NOT SCORED: a page of the pair is missing or out"
        return out
    m9, m15 = C4.arm_ms(p9), C4.arm_ms(p15)
    a9, a15 = C4.align_ms(p9), C4.align_ms(p15)
    ns = [n for n in reg["statistic"]["treads"]
          if all((a, n) in m for m in (m9, m15) for a in ("shared", "native"))]
    if len(ns) < 4:
        out["verdict"] = f"NOT SCORED: {len(ns)} common treads of n = 3..9"
        return out
    d = {n: ((m15[("shared", n)] - m15[("native", n)]) - (m9[("shared", n)] - m9[("native", n)])) * 1e3 for n in ns}
    al = {n: ((a15.get(("shared", n), 0) - a15.get(("native", n), 0))
              - (a9.get(("shared", n), 0) - a9.get(("native", n), 0))) * 1e3 for n in ns}
    null = {n: (m15[("native", n)] - m9[("native", n)]) * 1e3 for n in ns}
    raw = st.median(d.values())
    corr = raw - st.median(al.values())
    t_nat = st.median(m9[("native", n)] for n in ns) * 1e3
    null_bound = 3 * math.sqrt(2) * 0.0018 * t_nat
    null_med = st.median(null.values())
    out.update(treads=ns, Delta_n_us=d, align_n_us=al, Delta_raw_us=raw, Delta_us=corr,
               null_n_us=null, null_control={"median_us": null_med, "bound_us": null_bound,
                                             "verdict": "PASS" if abs(null_med) <= null_bound else "FAIL"})
    zs = {}
    for sig_name, sig in (("registered", reg["noise"]["sigma_noise_us"]),
                          ("resolution", reg["noise"]["sigma_noise_resolution_us"])):
        zs[sig_name] = {r: C4.z(corr, v["total"], sig, pr["sigma_D_us"] if r == "D" else 0.0)
                        for r, v in pr["rivals"].items()}
    out["z"] = zs
    out["sigma_noise_us"] = reg["noise"]["sigma_noise_us"]
    return out


def score_view(repo: Path, tree: Path, _fv, tview) -> dict:
    reg = C4.registration(repo, PART)
    res = {"registration": C4.NAMES[PART], "contrasts": {}}
    for name in reg["predictions_us"]:
        res["contrasts"][name] = contrast(tree, reg, name, tview)
    got = {k: v for k, v in res["contrasts"].items() if "Delta_us" in v}
    if not got:
        res["verdicts"] = {"D": {"verdict": "NOT SCORED: no contrast measured"}}
        return res
    if any(v["null_control"]["verdict"] == "FAIL" for v in got.values()):
        res["verdicts"] = {r: {"verdict": "INCONCLUSIVE (the NATIVE null control failed: board drift between the pages)"}
                           for r in ("D", "D2", "M", "FIXED", "K0", "SLOT")}
        return res
    z = {k: v["z"]["registered"] for k, v in got.items()}
    sig = reg["noise"]["sigma_noise_us"]
    v = {}
    for r in ("D", "D2"):
        bad = [k for k in z if abs(z[k][r]) > 3]
        v[r] = {"verdict": f"FAILS on {bad}" if bad else "HOLDS", "z": {k: z[k][r] for k in z}}
    for r in ("M", "K0", "SLOT"):
        bad = [k for k in z if abs(z[k][r]) > 3]
        v[r] = {"verdict": f"EXCLUDED on {bad}" if bad else "NOT EXCLUDED", "z": {k: z[k][r] for k in z}}
    fixed = [k for k, c in got.items() if c["Delta_us"] > 3 * sig]
    v["FIXED"] = {"verdict": f"EXCLUDED on {fixed}" if fixed else "NOT EXCLUDED"}
    v["K5"] = {"verdict": "NOT TESTED (equals D on A1 and A2; A3 is cut)"}
    if v["D"]["verdict"] == "HOLDS" and v["SLOT"]["verdict"] == "NOT EXCLUDED":
        v["D_vs_SLOT"] = {"verdict": "UNDECIDED"}
    res["verdicts"] = v
    res["resolution_verdicts"] = {k: {r: ("beyond 3" if abs(x) > 3 else "within 3") for r, x in c["z"]["resolution"].items()}
                                  for k, c in got.items()}
    pr = reg["predictions_us"]
    if "A1" in got:
        dn2 = pr["A1"]["dead_ctas"]["15"]["w2"] - pr["A1"]["dead_ctas"]["9"]["w2"]
        d_w2 = got["A1"]["Delta_us"] / dn2 * 1e3
        res["per_gemm_d_printed"] = {"d_w2_ns": d_w2}
        if "A2" in got:
            n15, n9 = pr["A2"]["dead_ctas"]["15"], pr["A2"]["dead_ctas"]["9"]
            res["per_gemm_d_printed"]["d_w1_ns"] = (got["A2"]["Delta_us"] - (n15["w2"] - n9["w2"]) * d_w2 * 1e-3) / (n15["w1"] - n9["w1"]) * 1e3
    return res


def score(repo: Path, tree: Path) -> dict:
    return C3.two_views(lambda fv, tv: score_view(repo, tree, fv, tv), timed_ignore=("V5",), repo=repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 4 DEAD-CTA CONTRAST, {res['registration']}"]
    for k, c in res["contrasts"].items():
        if "Delta_us" not in c:
            out.append(f"  {k} {c['model']}: {c.get('verdict')} {c['pages']}")
            continue
        out.append(f"  {k} {c['model']}: Delta {c['Delta_us']:.2f} us (raw {c['Delta_raw_us']:.2f}), "
                   f"null {c['null_control']['median_us']:+.2f} us within {c['null_control']['bound_us']:.2f}: "
                   f"{c['null_control']['verdict']}; {c['label']}")
        out.append("    z (sigma 0.9): " + "  ".join(f"{r} {x:+.2f}" for r, x in c["z"]["registered"].items()))
    for r, v in (res.get("verdicts") or {}).items():
        out.append(f"  {r}: {v['verdict']}")
    if res.get("per_gemm_d_printed"):
        out.append(f"  per-GEMM d (printed): {res['per_gemm_d_printed']}")
    return out + C3.view_lines(res)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C4.write_score(out, PART, res, lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
