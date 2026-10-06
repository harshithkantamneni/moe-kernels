#!/usr/bin/env python
"""Score rental 3's part C, which feature lifts G = 1 survival
(docs/registered/2026-10-05-rental3-stages-gh200.json): each page's class, the six
hypotheses, the one-page pairs, the controls.

    python scripts/scoring/rental3/score_stages.py <repo> <tree> <out>

Reads `*-mixtral-8x7b-tp2-<label>-r3-counters/lock1710/r3c-g1.json` for l2base,
k32s8, k128s4, k128s3, k64s7 and k64s3 under <tree>. Writes
<out>/stages.score.{json,txt}; exits 0 whatever the verdicts. A page's column is
keyed to its RECORDED w2 occupancy, stages and BLOCK_K, never the plan's guess.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3common as C3  # noqa: E402

PART = "stages"
MODEL = "mixtral-8x7b-tp2"
HYPS = ("OCC", "DEPTH", "WIDTH", "U-OCC", "U-DEPTH", "NULL")


def classify(h: str, ctas: int, stages: int, bk: int) -> str:
    occ, depth, width = ctas <= 2, stages >= 7, bk >= 128
    lifted = {"OCC": occ, "DEPTH": depth, "WIDTH": width, "U-OCC": occ or width,
              "U-DEPTH": depth or width, "NULL": False}[h]
    return "lifted" if lifted else "base"


def page_class(reg: dict, s_base: dict, s_knob: dict) -> tuple[str, dict]:
    b = reg["bands"]
    ds = {n: s_knob[n] - s_base[n] for n in b["ns"] if n in s_knob and n in s_base}
    up = sum(b["lifted"][0] <= v <= b["lifted"][1] for v in ds.values())
    lo = sum(b["base"][0] <= v <= b["base"][1] for v in ds.values())
    cls = "lifted" if up >= b["need"] else "base" if lo >= b["need"] else "unclassified"
    return cls, ds


def _s(sv: dict, g: str) -> dict:
    return {int(n): v for n, v in sv[g]["s"].items()}


def score_view(repo: Path, tree: Path, fview, tview) -> dict:
    import l2_survival as L
    reg = C3.registration(repo, PART)
    res = {"registration": C3.NAMES[PART], "pages": {}}
    surv, design = {}, {}
    for label in [reg["pages"]["base"], *reg["pages"]["knobs"]]:
        page = fview.load(C3.find_bytes(tree, MODEL, label, 1))
        if page is None:
            continue
        try:
            surv[label] = L.survival(page)
        except Exception as exc:                                   # noqa: BLE001
            res["pages"][label] = {"use": f"NOT SCORED: {type(exc).__name__}: {exc}"}
            continue
        design[label] = page.get("design") or {}
    base = surv.get(reg["pages"]["base"])
    if base is None:
        res["verdict"] = "NOT SCORED: the l2base page is missing"
        return res
    sb = _s(base, "w2")
    ref = {int(n): v for n, v in reg["controls"]["board_check"]["rental2_s_w2_SEEN"].items()}
    close = sum(abs(sb[n] - ref[n]) <= 0.05 for n in range(4, 10) if n in sb and n in ref)
    board_ok = close >= 4
    res["board_check"] = {"ok": board_ok, "within_0.05_at": close}
    lo, hi = reg["controls"]["F_over_Mn"]["band"]
    cols, classes = {}, {}
    for label, want in reg["pages"]["knobs"].items():
        sv = surv.get(label)
        ent = res["pages"].setdefault(label, {})
        if sv is None:
            ent.setdefault("use", "NOT SCORED: missing")
            continue
        d = design[label]
        st_, bk = int(d.get("num_stages") or 0), int(d.get("block_k") or 0)
        if (st_, bk) != (want["num_stages"], want["block_k"]):
            ent["use"] = f"NOT SCORED: the page's design is BK {bk} s{st_}, not the plan's"
            continue
        ctas = int(sv["w2"]["ctas"])
        why = []
        if sv.get("v6") not in (None, "PASS"):
            why.append(f"V6 {sv['v6']} (FLAGGED)")
        bad = [n for n, v in sv["w2"]["fabric_over_mn"].items() if v is not None and not lo <= v <= hi]
        if bad:
            why.append(f"PRIVATE F/Mn outside [{lo}, {hi}] at n {bad}")
        cls, ds = page_class(reg, sb, _s(sv, "w2"))
        far = [v for n, v in sv["w2"]["far"].items() if v is not None and 4 <= int(n) <= 9]
        ent.update(ctas_recorded=ctas, ctas_expected=reg["occupancy"]["expected"][label],
                   stages=st_, block_k=bk, ds=ds, cls=cls, far_share_printed=(sum(far) / len(far) if far else None),
                   w1_s_printed=_s(sv, "w1"), flags=why)
        if label == "k128s3" and ctas != 3:
            ent["k128s3_check"] = f"recorded occupancy {ctas}, not 3: the column is re-keyed"
        if why:
            ent["use"] = "printed, out of the count (" + "; ".join(why) + ")"
            continue
        ent["use"] = "counted"
        cols[label] = {h: classify(h, ctas, st_, bk) for h in HYPS}
        if cls != "unclassified":
            classes[label] = cls
    hyp = {}
    for h in HYPS:
        wrong = [p for p, c in classes.items() if cols[p][h] != c]
        hyp[h] = {"wrong_on": wrong, "status": "FALSIFIED" if wrong else "consistent"}
    pairs = {}
    for k, pages in reg["pairs"]["differ_on"].items():
        a, b = k.split("/")
        sep = [p for p in classes if cols[p][a] != cols[p][b]]
        pairs[k] = "SEPARATED" if sep else "NO SEPARATION"
    res.update(hypotheses=hyp, pairs=pairs, classified=classes)
    n_cls = len(classes)
    if n_cls == 0:
        res["verdict"] = "NOT SCORED: no page classified"
    elif not board_ok:
        res["verdict"] = "INCONCLUSIVE (the board check failed)"
    else:
        sel = [h for h in HYPS if hyp[h]["status"] == "consistent"
               and all(hyp[o]["status"] == "FALSIFIED" for o in HYPS if o != h)]
        if classes.get(reg["pages"]["control"]) == "lifted":
            res["verdict"] = "NONE (the control lifted: every hypothesis FALSIFIED)"
        elif sel and n_cls >= 4:
            res["verdict"] = sel[0]
        else:
            surv_h = [h for h in HYPS if hyp[h]["status"] == "consistent"]
            res["verdict"] = "INCONCLUSIVE"
            res["survivors"] = surv_h
    return res


def score(repo: Path, tree: Path) -> dict:
    flr = C3.registration(repo, "floorlaw")
    return C3.two_views(lambda fv, tv: score_view(repo, tree, fv, tv),
                        floor_rules=flr["gates"]["floor"], repo=repo, floor_ignore=("V6", "V7", "V10"))


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 3 PART C (what lifts G = 1 survival), {res['registration']}"]
    out.append(f"  board check: {res.get('board_check')}")
    for p, e in res["pages"].items():
        out.append(f"  {p}: {e.get('cls', '-')} at {e.get('ctas_recorded')} CTAs/SM ({e.get('use')})")
    for h, v in (res.get("hypotheses") or {}).items():
        out.append(f"  {h}: {v['status']} {v['wrong_on'] or ''}")
    for k, v in (res.get("pairs") or {}).items():
        if v != "SEPARATED":
            out.append(f"  pair {k}: {v}")
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
