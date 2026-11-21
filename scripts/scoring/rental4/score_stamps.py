#!/usr/bin/env python
"""Score rental 4's stamps (docs/registered/2026-10-06-rental4-stamps-gh200.json): F split,
the k-step law per iteration, tail CTAs and dead CTAs.

    python scripts/scoring/rental4/score_stamps.py <repo> <tree> <out>

Reads `*-<card>-instr-<model>-<label>/stamps.json` (+ stamps/*.npy), the perturb gate
(`*-perturb-*/gate.env`) and, where it ran, `*-gpubench-*/constants.json`. A unit whose gate
is not PASS is NOT SCORED. Writes <out>/stamps.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4common as C4  # noqa: E402

PART = "stamps"
UNITS = {"F": [("mixtral-8x7b", "stf")],
         "K": [("olmoe-1b-7b", "stk64s4"), ("olmoe-1b-7b", "stk32s4"), ("olmoe-1b-7b", "stk128s4"),
               ("olmoe-1b-7b", "stk64s8")],
         "T": [("mixtral-8x22b", "sttail"), ("mixtral-8x7b", "sttail")],
         "D": [("olmoe-1b-7b", "stdead4"), ("olmoe-1b-7b", "stdead2")]}
OCC_T = {"w1": 5, "w2": 4}
OCC_D = {"stdead4": {"w1": 5, "w2": 4}, "stdead2": {"w1": 2, "w2": 2}}


def gates(tree: Path) -> dict:
    out = {}
    for env in sorted(Path(tree).rglob(f"*-{C4.CARD}-perturb-*/gate.env")):
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            out[k] = v.split()[0] if v else ""
    return out


def gate_numbers(tree: Path) -> dict:
    """{GATE_<id>: {"verdict", "median", "worst"}} off every gate.env line (the numbers kept)."""
    out = {}
    for env in sorted(Path(tree).rglob(f"*-{C4.CARD}-perturb-*/gate.env")):
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            parts = v.split()
            if not parts:
                continue
            rec = {"verdict": parts[0]}
            for p in parts[1:]:
                kk, _, vv = p.partition("=")
                try:
                    rec[kk] = float(vv)
                except ValueError:
                    pass
            out[k] = rec
    return out


def variant_id(tree: Path, model: str, label: str) -> str | None:
    """The plan's variant id of a stamps unit (the driver's instr-variants.json), else None."""
    for p in sorted(Path(tree).rglob("instr-variants.json")):
        try:
            vs = json.loads(p.read_text())
        except ValueError:
            continue
        ids = [v.get("id") for v in vs if v.get("kind") == "stamps" and v.get("model") == model
               and str(v.get("id", "")).endswith(f"-{label}")]
        if len(ids) == 1:
            return ids[0]
    return None


def not_run(tree: Path, model: str, label: str) -> str | None:
    """A unit with no stamps.json whose gate line reads other than PASS was refused by the
    driver, as registered: NOT RUN (gate FAIL), with the gate's numbers (post-page fix to
    reading code: the verdict stays NOT SCORED; only the reason is read off gate.env)."""
    vid = variant_id(tree, model, label)
    if vid is None:
        return None
    rec = gate_numbers(tree).get("GATE_" + "".join(c if c.isalnum() else "_" for c in vid))
    if rec is None or rec["verdict"] == "PASS":
        return None
    nums = ", ".join(f"{k} {100 * rec[k]:.2f}%" for k in ("median", "worst") if k in rec)
    return f"NOT SCORED: NOT RUN (gate {rec['verdict']}{': ' + nums if nums else ''}; variant {vid}, refused by the driver)"


def unit(tree: Path, model: str, label: str, gate: dict, block: str):
    hits = sorted(Path(tree).rglob(f"*-{C4.CARD}-instr-{model}-{label}/stamps.json"))
    if not hits:
        why = not_run(tree, model, label)
        if why:
            return None, None, why
    if len(hits) != 1:
        return None, None, f"NOT SCORED: {len(hits)} stamps.json for {model} {label}"
    man, launches = C4.load_stamps(hits[0].parent)
    key = "GATE_" + "".join(c if c.isalnum() else "_" for c in man.get("variant", ""))
    if gate.get(key) != "PASS":
        return None, None, f"NOT SCORED: the perturb gate for {man.get('variant')} reads {gate.get(key) or 'nothing'}"
    ok, why = C4.sass_precondition(man, block)
    if not ok:
        return None, None, f"NOT SCORED: the SASS precondition fails ({why})"
    return man, launches, "counted"


def by(launches, **kw):
    return [x for x in launches if all(x.get(k) == v for k, v in kw.items())]


def score(repo: Path, tree: Path) -> dict:
    reg = C4.registration(repo, PART)
    gate = gates(tree)
    res = {"registration": C4.NAMES[PART], "units": {}}
    gb = sorted(Path(tree).rglob(f"*-{C4.CARD}-gpubench-*/constants.json"))
    near_far_dram = dict(reg["F"]["rivals_cycles_at_1710"])
    source = "the published RRZE GH200 plateaus (SEEN third-party)"
    if len(gb) == 1:
        cyc = json.loads(gb[0].read_text()).get("cycles_at_lock") or {}
        if all(cyc.get(k) for k in ("near_l2_ns", "far_l2_ns", "dram_ns")):
            near_far_dram = {"NEAR": cyc["near_l2_ns"], "FAR": cyc["far_l2_ns"], "DRAM": cyc["dram_ns"]}
            source = "this rental's gpubench constants"
    res["F_rivals"] = {"cycles": near_far_dram, "source": source}

    # F: the w2 epilogue's extra load
    for model, label in UNITS["F"]:
        man, L, use = unit(tree, model, label, gate, "F")
        ent = res["units"][label] = {"use": use}
        if man is None:
            continue
        per_n = {}
        split = {}
        for n in sorted({x["n"] for x in L}):
            ep = {}
            for g in ("w1", "w2"):
                xs = by(L, n=n, gemm=g)
                ph = [C4.med_of(x["phases"]["epilogue"][C4.live_mask(x["cols"])]) for x in xs]
                ep[g] = C4.med_of(ph)
                split.setdefault(g, {"prologue": [], "loop": [], "epilogue": []})
                for x in xs:
                    m = C4.live_mask(x["cols"])
                    for k in ("prologue", "loop", "epilogue"):
                        split[g][k].append(C4.med_of(x["phases"][k][m]))
            if ep["w1"] is not None and ep["w2"] is not None:
                per_n[n] = ep["w2"] - ep["w1"]
        if not per_n:
            ent["verdict"] = "NOT SCORED: no live epilogue stamp"
            continue
        d = st.median(per_n.values())
        ent.update(delta_epi_cycles=d, per_n=per_n,
                   phases_printed={g: {k: C4.med_of(v) for k, v in s.items()} for g, s in split.items()},
                   phases_note="prologue is before the pipeliner's fill; the fill is inside loop: no F_g split is read",
                   **C4.nearest(d, near_far_dram))

    # K: per-iteration time against the laws
    kpred = reg["K"]["predicted_T_iter_cycles"]
    by_law = {}
    for model, label in UNITS["K"]:
        man, L, use = unit(tree, model, label, gate, "K")
        ent = res["units"][label] = {"use": use}
        if man is None:
            continue
        for g in ("w1", "w2"):
            vals = [C4.steady_iter_cycles(x) for x in by(L, gemm=g)]
            t = C4.med_of(vals)
            if t is None:
                ent[g] = {"use": "NOT SCORED: no steady iteration"}
                continue
            e = {law: t / p - 1 for law, p in kpred[label][g]["T_iter_cycles"].items()}
            ent[g] = {"T_iter_cycles": t, "occ": kpred[label][g]["occ"], "e": e}
            for law, x in e.items():
                by_law.setdefault(law, []).append((label, g, x))
    laws = {law: {"status": "FALSIFIED" if any(abs(x) > 0.03 for *_, x in es) else "consistent",
                  "beyond_3pct": [(lab, g) for lab, g, x in es if abs(x) > 0.03], "scored": len(es)}
            for law, es in by_law.items()}
    classes = {c: {"laws": ms, "status": ("consistent" if any(laws.get(m, {}).get("status") == "consistent" for m in ms)
                                          else "FALSIFIED")}
               for c, ms in reg["K"]["classes"].items() if any(m in laws for m in ms)}
    sel = [c for c, v in classes.items() if v["status"] == "consistent"]
    res["K_laws"] = laws
    res["K_classes"] = classes
    res["K_verdict"] = {"verdict": ("NOT SCORED" if not laws else f"SELECTED class {sel[0]}" if len(sel) == 1
                                    else "UNDECIDED"),
                        "note": "one test of the law with B' (occlaw); MVA2 vs OCC and LK vs LITTLE are not separated"}

    # T: tail CTAs' per-iteration time
    for model, label in UNITS["T"]:
        man, L, use = unit(tree, model, label, gate, "T")
        key = f"{label}-{model}"
        ent = res["units"][key] = {"use": use}
        if man is None:
            continue
        picks = []
        for x in L:
            t, n_tail = C4.tail_iter_cycles(x, C4.SMS, OCC_T[x["gemm"]])
            if t is None:
                continue
            ns = t / C4.LOCK_MHZ * 1e3
            rivals = dict(reg["T"]["rivals_ns"], FS=C4.STAGE_BYTES * n_tail / (C4.BW_GBPS * 1e9) * 1e9)
            r = C4.nearest(ns, rivals, log=True)
            ent[f"{x['gemm']}-{x['arm']}-n{x['n']}-call{x['call']}"] = {"T_tail_ns": ns, "tail_ctas": n_tail, **r}
            picks.append(r["verdict"])
        ent["verdict"] = (picks[0] if picks and all(p == picks[0] for p in picks) else
                          "NOT SCORED: no tail CTA" if not picks else f"MIXED {sorted(set(picks))}")

    # D: dead CTAs
    for model, label in UNITS["D"]:
        man, L, use = unit(tree, model, label, gate, "D")
        ent = res["units"][label] = {"use": use}
        if man is None:
            continue
        for g in ("w1", "w2"):
            life = C4.med_of([C4.dead_life_cycles(x) for x in by(L, gemm=g)])
            reads = [C4.dead_dispatch_ns(x) for x in by(L, gemm=g, arm="shared")]
            disp = C4.med_of([r["value"] for r in reads if r["value"] is not None])
            occ = OCC_D[label][g]
            ent[g] = {"life_cycles": life, "dispatch_ns": disp, "occ": occ,
                      "dispatch_reads": [{k: v for k, v in r.items()} for r in reads],
                      "life": C4.nearest(life, reg["D"]["rivals_life_cycles"]) if life else {"verdict": "NOT SCORED"},
                      "dispatch": (C4.nearest(disp, reg["D"]["rivals_dispatch_ns"][str(occ)])
                                   if disp and str(occ) in reg["D"]["rivals_dispatch_ns"] else {"verdict": "NOT SCORED"})}
    return res


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 4 STAMPS, {res['registration']}", f"  F rivals: {res['F_rivals']}"]
    for k, v in res["units"].items():
        out.append(f"  {k}: {v.get('use')}" + (f" -> {v['verdict']}" if "verdict" in v else ""))
        for g in ("w1", "w2"):
            if isinstance(v.get(g), dict):
                out.append(f"    {g}: " + json.dumps({kk: vv for kk, vv in v[g].items() if kk != "rivals"}, default=str)[:300])
    for law, v in (res.get("K_laws") or {}).items():
        out.append(f"  K {law}: {v['status']} on {v['scored']}")
    out.append(f"  K verdict: {res['K_verdict']['verdict']}")
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
