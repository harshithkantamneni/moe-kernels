#!/usr/bin/env python
"""Score rental 4's part B', the k-step law at BLOCK_K and num_stages on OLMoE G = 64 counters
(docs/registered/2026-10-06-rental4-occlaw-gh200.json).

    python scripts/scoring/rental4/score_occlaw.py <repo> <tree> <out>

Reads `*-olmoe-1b-7b-<label>-r3-counters/lock1710/r3c-g64.json` for k64s4 (the base) and the
six knob pages. Writes <out>/occlaw.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4common as C4  # noqa: E402

C3 = C4.C3
PART = "occlaw"
MODEL = "olmoe-1b-7b"
NS = range(4, 10)


def bytes_two_views(fn, repo: Path, ignore=("V6", "V7", "V10")) -> dict:
    """Rental 2's View (ALL and CLEAN) over counter pages, combined by rule 3."""
    add = C4.CM2.addendum(repo)
    out, views = {}, {}
    for name in ("ALL", "CLEAN"):
        v = C4.CM2.View(name, add, ignore)
        out[name] = fn(v)
        views[name] = v
    reg, table = C4.combine(out["ALL"], out["CLEAN"])
    pages = {}
    for name, v in views.items():
        for k, rec in v.pages.items():
            pages.setdefault(k, {})[name] = rec
    reg["addendum"] = {"rule": "rental 2's addendum rule 3", "pages": pages, "verdicts": table, "CLEAN": out["CLEAN"]}
    return reg


def score_view(repo: Path, tree: Path, view) -> dict:
    reg = C4.registration(repo, PART)
    res = {"registration": C4.NAMES[PART], "pages": {}}
    pages = {}
    for label in [reg["pages"]["base"], *reg["pages"]["knobs"]]:
        page = view.load(C3.find_bytes(tree, MODEL, label, 64))
        if page is None:
            res["pages"][label] = {"use": "NOT SCORED: missing or out of this view"}
            continue
        de = page.get("design") or {}
        bk, stg = C4.KNOBS[label]
        if (int(de.get("block_k") or 0), int(de.get("num_stages") or 0)) != (bk, stg):
            res["pages"][label] = {"use": f"NOT SCORED: the page's design is BK {de.get('block_k')} s{de.get('num_stages')}"}
            continue
        pages[label] = page
    base = reg["pages"]["base"]
    if base not in pages:
        res["verdict"] = "NOT SCORED: the k64s4 base page is missing"
        return res
    fits = {lab: {g: C4.page_c(p, g, NS, C4.F_CTA[g], C4.FLOOR_MAX_DRAM_FRAC) for g in ("w1", "w2")}
            for lab, p in pages.items()}
    e_by_law = {law: [] for law in reg["laws"]}
    for lab, fg in fits.items():
        if lab == base:
            continue
        ent = res["pages"].setdefault(lab, {})
        for gi, g in enumerate(("w1", "w2")):
            fk, fb = fg[g], fits[base][g]
            if fk is None or fb is None:
                ent[g] = {"use": "NOT SCORED: fewer than 3 floor-bound cells"}
                continue
            r_meas = (fk["S"] * fk["c"] + C4.F_CTA[g]) / (fb["S"] * fb["c"] + C4.F_CTA[g])
            occ_k, occ_b = fk["occupancy"][0], fb["occupancy"][0]
            reg_k, reg_b = reg["occupancy"]["registered"][lab][g], reg["occupancy"]["registered"][base][g]
            rekeyed = (occ_k, occ_b) != (reg_k, reg_b)
            row = {"c_meas": fk["c"], "c_base": fb["c"], "r_meas": r_meas, "occupancy": [occ_k, occ_b],
                   "rekeyed": rekeyed, "e": {}}
            for law, p in reg["laws"].items():
                if rekeyed:
                    rp = C4.ratio_pred(law, p, C4.KNOBS[lab], occ_k, C4.KNOBS[base], occ_b, fk["K"], C4.F_CTA[g])
                else:
                    rp = reg["predicted_ratio"][lab][g][law]
                if rp is None:
                    continue
                e = r_meas / rp - 1
                row["e"][law] = e
                e_by_law[law].append((lab, g, e))
            ent[g] = row
    scored = len({(lab, g) for v in e_by_law.values() for lab, g, _ in v})
    laws = {}
    for law, es in e_by_law.items():
        bad = [(lab, g) for lab, g, e in es if abs(e) > 0.02]
        laws[law] = {"status": "FALSIFIED" if bad else ("consistent" if es else "NOT SCORED"),
                     "beyond_2pct": bad, "scored": len(es),
                     "rms": math.sqrt(sum(e * e for *_, e in es) / len(es)) if es else None}
    res["laws"] = laws
    sel = [law for law, v in laws.items() if v["status"] == "consistent"
           and all(o["status"] == "FALSIFIED" for k, o in laws.items() if k != law and o["status"] != "NOT SCORED")]
    if scored == 0:
        res["verdict"] = "NOT SCORED: no knob page scored"
    elif len(sel) == 1 and laws[sel[0]]["scored"] >= 8:
        res["verdict"] = f"SELECTED {sel[0]}"
    else:
        res["verdict"] = "UNDECIDED"
        res["survivors"] = [law for law, v in laws.items() if v["status"] == "consistent"]
    return res


def score(repo: Path, tree: Path) -> dict:
    return bytes_two_views(lambda v: score_view(repo, tree, v), repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 4 PART B' (k-step law at BLOCK_K / num_stages, OLMoE G = 64), {res['registration']}"]
    for lab, ent in res["pages"].items():
        for g in ("w1", "w2"):
            r = ent.get(g)
            if isinstance(r, dict) and "r_meas" in r:
                out.append(f"  {lab:7s} {g}: r {r['r_meas']:.4f} occ {r['occupancy']}{' RE-KEYED' if r['rekeyed'] else ''} | "
                           + "  ".join(f"{law} {100 * e:+.1f}%" for law, e in r["e"].items()))
        if "use" in ent:
            out.append(f"  {lab}: {ent['use']}")
    for law, v in (res.get("laws") or {}).items():
        out.append(f"  {law}: {v['status']} on {v['scored']} (rms {v['rms']})")
    out.append(f"  verdict: {res['verdict']}")
    return out


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
