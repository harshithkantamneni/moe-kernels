#!/usr/bin/env python
"""Model v7's gate (registration docs/registered/2026-11-17-v7-gate-gh200): one GO / NO-GO line per regime, each
regime's GO independent. A regime may go to ONE blind only on its own GO; no bar may be changed after the registration
is public. Run it only after the registration is public (pushed): before that it is never run on the gate's pages.

    python scripts/scoring/v7/gate_v7.py [--out <dir>]

It reads the registration at its committed path only (REG_PATH; there is no option to name another file, so the
freeze list cannot be swapped). FREEZE first: every file the registration's 'freeze' lists (every repository module
on the gate's import closure, traced, the vendored round 5, the readings and the quoted registrations) has its
registered sha256 and v7's constants hash (r3_timing_model_v7.constants_sha256) is the registered one; on any
difference every regime reads NO-GO (REFUSED) and nothing is priced. Then, per regime (the bars are the old bars,
unchanged):

  R3       G-R3: v7 at R3's tile re-prices the 300 registered v3 predictions (rentals 6 and 7) bit for bit: equal to the
           registered number at 6 decimals and to rental 6's pricer exactly (r3_timing_model_v7.check_r3)
  PREFILL  C4's next-round gate as registered (cal4 txt, AFTER), held out leave-one-model-out exactly as round 5
           computed it (round5_chain.held_out_sets, held_out_knee, held_out_sag; r4.summ's means, 2 decimals;
           RR's C and sigma carried unchanged into every fold). TREES first (prefill_trees, a refusal as decode's):
           with every switch off v7 must reproduce round 5's held-out table (tileB3.score.json
           preflight.round5_table) and B3's 84 committed predictions (b3.predictions.json, ms to 6 decimals, reads to
           the byte), else the regime is REFUSED:
           G1  every held-out 128 G 1 mean within 1.0% (|mean| <= 1.0): OLMoE-1B-7B, Qwen2-57B-A14B and Mixtral-8x7B
               on A2's cells, Mixtral-8x22B and Mixtral-8x7B on C2's board, Mixtral-8x7B-tp2 on C4's
           G2  estimated P(A) >= 0.5 by the registered method: r7.pass_estimates5 (read_c2.power at B2's bar 2% / 5% /
               1%, R 200,000, seed 20261022) on the five models' means (OLMoE, Qwen2-57B, Mixtral-8x7B on A2,
               Mixtral-8x22B, tp2) over B2's power cells that v7 prices, its 'A (5 models)'
           printed, never a clause: B3's 84 NATIVE cells at B3's board pars (SEEN), pooled and per model
  DECODE   addendum 6's two clauses, applied to v7 (READINGS and TREES as gate_d6, refusals):
           CAL       CAL2 under v7 reads GO by the addendum-3 stop rule at 3% rms, 8% worst cell, 1.5% |mean| on CAL2's
                     kept cells (v6d's three CAL constants refitted on v7's body on CAL2's kept cells, and, for
                     the v4d'-branch cells only, D6FIX's three constants from a second CAL2 fit with v4d' on every
                     naive cell; score_cal_d3's rules)
           RE-SCORE  the published decode blind re-scored under v7 at the CAL2 refit reads a pooled rms under 4.0% over
                     the committed 220 cells (a weak sanity floor on SEEN pages, never a test)
           D6FIX's constants: the CAL2 fit with v4d' on every naive cell (the registered selection rule), read here

Each regime's clauses are also read with every switch off (prefill: round 5; decode: v6d; R3 has no v7 term), printed
DESCRIPTIVE beside; only v7 as registered (every switch on) decides. Exit 0 when every regime reads GO, 1 otherwise,
2 on a usage error."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r3_timing_model_v7 as V7  # noqa: E402

NAME = "2026-11-17-v7-gate-gh200"
REG_PATH = REPO / "docs" / "registered" / f"{NAME}.json"
REGIMES = ("r3", "prefill", "decode")
#: the old bars, unchanged
R3_PREDICTIONS = 300
G1_MAX_ABS_MEAN_PCT = 1.0
G2_PA_MIN = 0.5
CAL_BARS = {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015}
RESCORE_RMS_MAX = 0.04
RESCORE_CELLS = 220
#: round 5's held-out sets in combo5's order and the five models of the estimate
HELD_OUT = ("olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b", "mixtral-8x22b", "mixtral-8x7b-tp2", "mixtral-8x7b C2")
FIVE = HELD_OUT[:5]


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ================================================================== the freeze

def verify_freeze(reg: dict, root: Path = REPO, constants_sha: str | None = None) -> list[str]:
    """Every frozen path's sha256 and v7's constants hash against the registration; the differences (empty: frozen)."""
    fz = reg["freeze"]
    bad = []
    for p, want in fz["files"].items():
        f = Path(root) / p
        if not f.is_file():
            bad.append(f"{p}: missing")
        elif sha256(f) != want:
            bad.append(f"{p}: sha256 {sha256(f)[:12]} is not the registered {want[:12]}")
    got = V7.constants_sha256() if constants_sha is None else constants_sha
    if got != fz["constants_sha256"]:
        bad.append(f"v7's constants hash {got[:12]} is not the registered {fz['constants_sha256'][:12]}")
    return bad


# ================================================================== the clauses (pure)

def r3_clause(check: dict) -> dict:
    n = int(check.get("compared") or 0)
    ur, up = len(check.get("unequal_registered") or []), len(check.get("unequal_pricer") or [])
    ok = n == R3_PREDICTIONS and ur == 0 and up == 0
    why = [] if ok else [f"G-R3: {n} of {R3_PREDICTIONS} compared, {ur} differ from the registered number, {up} from "
                         "rental 6's pricer"]
    return {"G-R3": {"compared": n, "unequal_registered": ur, "unequal_pricer": up, "verdict": "PASSES" if ok else "FAILS"},
            "verdict": "GO" if ok else "NO-GO", "why": why}


def prefill_clauses(means: dict, p_a: float | None) -> dict:
    """G1 on the six held-out means (%, 2 decimals as round 5 printed them), G2 on the estimate."""
    missing = [k for k in HELD_OUT if not isinstance(means.get(k), (int, float)) or not math.isfinite(means[k])]
    out = {"G1": {"means_pct": {k: means.get(k) for k in HELD_OUT}, "max_abs_mean_pct": G1_MAX_ABS_MEAN_PCT},
           "G2": {"P_A": p_a, "min": G2_PA_MIN}}
    why = []
    if missing:
        out["G1"]["verdict"] = "FAILS"
        why.append(f"G1: no mean for {missing}")
    else:
        outside = {k: means[k] for k in HELD_OUT if not abs(means[k]) <= G1_MAX_ABS_MEAN_PCT}
        out["G1"]["outside"] = outside
        out["G1"]["verdict"] = "PASSES" if not outside else "FAILS"
        if outside:
            why.append("G1: held-out means outside 1.0%: " + ", ".join(f"{k} {v:+.2f}" for k, v in outside.items()))
    ok2 = p_a is not None and math.isfinite(p_a) and p_a >= G2_PA_MIN
    out["G2"]["verdict"] = "PASSES" if ok2 else "FAILS"
    if not ok2:
        why.append(f"G2: estimated P(A) {p_a if p_a is None else round(p_a, 4)} under {G2_PA_MIN}")
    out["verdict"] = "GO" if not why else "NO-GO"
    out["why"] = why
    return out


def decode_clauses(cal_verdict: str | None, cal_stats: dict | None, pooled: dict | None) -> dict:
    """Addendum 6's clause 1 (CAL: the addendum-3 stop rule's verdict) and clause 2 (RE-SCORE: pooled rms < 4.0% over
    the committed 220 cells)."""
    why = []
    cal_ok = cal_verdict == "GO"
    if not cal_ok:
        why.append(f"CAL: CAL2 under v7 reads {cal_verdict}")
    pooled = pooled or {}
    rms, cells = pooled.get("rms"), pooled.get("cells")
    rs_ok = (rms is not None and math.isfinite(rms) and cells == RESCORE_CELLS and rms < RESCORE_RMS_MAX)
    if cells != RESCORE_CELLS:
        why.append(f"RE-SCORE: pools {cells} cells, not the committed {RESCORE_CELLS}")
    elif not rs_ok:
        why.append(f"RE-SCORE: pooled rms {'none' if rms is None else f'{rms:.2%}'}, not under {RESCORE_RMS_MAX:.1%}")
    return {"CAL": {"verdict": cal_verdict, "E1_CAL_IN_SAMPLE": cal_stats, "bars": dict(CAL_BARS)},
            "RE-SCORE": {"verdict": "PASSES" if rs_ok else "FAILS", "pooled": pooled, "rms_max": RESCORE_RMS_MAX,
                         "cells_committed": RESCORE_CELLS},
            "verdict": "GO" if not why else "NO-GO", "why": why}


def summ(e: list) -> dict:
    """mean, rms, worst in % (rounded to 2 decimals, r4.summ's)."""
    e = [100 * x for x in e]
    return {"n": len(e), "mean": round(st.fmean(e), 2), "rms": round(math.sqrt(st.fmean(x * x for x in e)), 2),
            "worst": round(max(e, key=abs), 2)}


# ================================================================== the computations (the published pages)

def compute_r3(switches: V7.Switches = V7.V7) -> dict:
    return V7.check_r3(switches=switches)


def prefill_trees() -> list[str]:
    """The prefill regime's TREES refusal (decode's, gate_d6, applied to round 5): with every switch off v7 must
    reproduce round 5's held-out table (tileB3.score.json preflight.round5_table, r4.summ's 2-decimal means) and B3's
    84 committed predictions (b3.predictions.json at its pars: ms to 6 decimals, w1 and w2 reads to the byte). The
    clauses read the A2, C2 and C4 pages and the tileA, tileB, cal2 and cal4 histograms, none of them in the freeze,
    so a changed page or chain refuses the regime here. The differences (empty: the trees reproduce)."""
    import round5_chain as R5
    C = R5.chain()
    bad = []
    want = json.loads((REPO / "scripts/scoring/tileB3/tileB3.score.json").read_text())["preflight"]["round5_table"]
    got = {}
    for k, (cells, hm) in R5.held_out_sets().items():
        knee = R5.held_out_knee(hm)
        e = [V7.price_prefill(r, sag128=R5.held_out_sag(r, hm), knee=knee, switches=V7.OFF)["ms"] / r["meas"] - 1
             for r in cells]
        got[k] = C["r4"].summ(e)["mean"]
    if got != want:
        bad.append(f"round 5's held-out table with every switch off is {got}, not the committed {want}")
    fin = json.loads((REPO / "scripts/scoring/tileB3/b3.predictions.json").read_text())
    rows = R5.b2_rows()
    if len(rows) != len(fin["cells"]):
        bad.append(f"{len(rows)} B2 rows, not B3's {len(fin['cells'])} committed predictions")
    for r, c in zip(rows, fin["cells"]):
        a = V7.price_prefill(r, sag128=tuple(fin["par"]), sag64=tuple(fin["par64"]), switches=V7.OFF)
        reads = {g: int(round(v)) for g, v in a["reads"].items()}
        if round(a["ms"], 6) != c["ms"] or reads != c["reads"]:
            bad.append(f"B3 cell {c['index']} ({c['model']} {c['cls']} {c['label']}) with every switch off: "
                       f"{round(a['ms'], 6)} ms, reads {reads}; committed {c['ms']} ms, reads {c['reads']}")
    return bad


def compute_prefill(switches: V7.Switches = V7.V7) -> dict:
    """The six held-out means, the estimate and B3's 84-cell re-score (information) under `switches`, after the
    TREES refusal (prefill_trees)."""
    import round5_chain as R5
    bad = prefill_trees()
    if bad:
        raise V7.Refused("TREES: " + "; ".join(bad[:10]))
    C = R5.chain()
    means = {}
    for k, (cells, hm) in R5.held_out_sets().items():
        knee = R5.held_out_knee(hm)
        e = [V7.price_prefill(r, sag128=R5.held_out_sag(r, hm), knee=knee, switches=switches)["ms"] / r["meas"] - 1
             for r in cells]
        means[k] = C["r4"].summ(e)["mean"]
    rows = R5.b2_rows()
    cells84 = C["RF"].RC.b2_cells()
    priced = [c for r, c in zip(rows, cells84, strict=True)
              if int(r["doc"]["block_m"]) != 128 or R5.reads(r, R5.K1_SEEN[0]) is not None]
    pa = R5.pass_estimate({k: means[k] for k in FIVE}, priced)
    fin = json.loads((REPO / "scripts/scoring/tileB3/b3.predictions.json").read_text())
    score = json.loads((REPO / "scripts/scoring/tileB3/tileB3.score.json").read_text())
    meas = {}
    for c in score["views"]["cells"]:
        m, page, n, lab, _arm = c["cell"].split("/")
        meas[(m, page, int(n), lab)] = c["meas"]
    by: dict = {}
    for r in rows:
        k = (r["model"], r["page"], int(r["n"]), r["label"])
        if k not in meas:
            continue
        p = V7.price_prefill(r, sag128=tuple(fin["par"]), sag64=tuple(fin["par64"]), switches=switches)
        e = p["ms"] / meas[k] - 1
        by.setdefault(r["model"], []).append(e)
        by.setdefault("pooled", []).append(e)
    return {"means": means, "P_A": pa["A (5 models)"], "estimate": pa, "estimate_cells": len(priced),
            "b3_information": {m: summ(v) for m, v in by.items()}}


def compute_decode(switches: V7.Switches = V7.V7) -> dict:
    """READINGS and TREES (gate_d6's refusals), the D6FIX fit, CAL2 under v7 and the re-score under v7."""
    X = V7.decode_modules()
    import d4common as D4
    import dcommon as DC
    import finalise_decode as FN
    import gate_d6 as G
    import score_blind_d3 as SB3
    import score_cal_d3 as SC3
    cal2, blind = REPO / D4.CAL2_PUBLISHED / "results", REPO / D4.BLIND_PUBLISHED / "results"
    with tempfile.TemporaryDirectory() as td:
        terms, law = G.readings_law(G.readings_dir_committed(Path(td) / "r"))
    tp, src = D4.tile_source()
    rep = D4.reproduce(cal2, blind, tile=(tp, src))
    if rep["verdict"] != "REPRODUCES":
        raise G.Refused("TREES: " + "; ".join(rep["CAL2"]["why"] + rep["decode blind"]["why"]))
    reg = DC.registration(DC.REPO, "cal")

    def cal(prm_fix=None, naive_v4dp=False):
        with V7.decode_scorers(terms=terms, law=law, prm_fix=prm_fix, switches=switches, naive_v4dp=naive_v4dp):
            return SC3.score(cal2, reg, tp, src, settle=False)
    prm_fix, fix = None, None
    if switches.D6FIX:
        cf = cal(naive_v4dp=True)
        sel = cf["FIT"]["selected"]
        if sel is None:
            raise G.Refused("D6FIX: the CAL2 fit with v4d' on every naive cell selects no rho form")
        f = cf["FIT"]["forms"][sel]
        prm_fix = X["V4D"].params_from(f, rho_form=sel, tile=tp)
        fix = {"rho": sel, **{k: f[k] for k in ("T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms")}}
    c = cal(prm_fix)
    stop = c.get("STOP") or {}
    sel = (c.get("FIT") or {}).get("selected")
    out = {"cal_verdict": stop.get("verdict"), "cal_why": stop.get("why") or [],
           "cal_stats": (c.get("E1_CAL_IN_SAMPLE") or {}).get("stats"), "rho": sel, "fix": fix,
           "constants": None if sel is None else {k: c["FIT"]["forms"][sel].get(k)
                                                  for k in ("T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms")}}
    if sel is None:
        out["pooled"] = None
        return out
    fz = G.frozen_at(c)
    fz["model"] = {"name": "v7", "fixes": X["V4DP"].ALL.as_dict(), "terms": terms.as_dict(), "law": law.as_dict(),
                   "switches": switches.as_dict()}
    with V7.decode_scorers(terms=terms, law=law, prm_fix=prm_fix, switches=switches), X["D3"].v4dp(X["V4DP"].ALL):
        rows = FN.predict(fz, models=DC.BLIND_MODELS)
    with V7.decode_scorers(terms=terms, law=law, prm_fix=prm_fix, switches=switches):
        s = SB3.score(blind, DC.registration(DC.REPO, "blind"), {"frozen": fz, "cells": rows})
    e = s.get("E1_DEC") or {}
    out["pooled"] = (e.get("pooled") or {}).get("stats")
    out["per_model"] = {m: x.get("stats") for m, x in (e.get("per_model") or {}).items()}
    return out


#: each regime's ablation rungs beside v7 (every switch on): the regime with every switch off; the other regime's
#: switch is out of its domain (tests/test_timing_model_v7*.py), and R3 has no v7 term, so it has no rung
ABLATION = {"r3": {}, "prefill": {"off (round 5)": V7.OFF}, "decode": {"off (v6d)": V7.OFF}}


def evaluate(reg: dict, compute: dict | None = None, ablation: bool = True) -> dict:
    """The gate's document. `compute` maps a regime to its computation (the defaults above; the tests pass
    synthetic ones)."""
    comp = {"r3": compute_r3, "prefill": compute_prefill, "decode": compute_decode, **(compute or {})}
    doc = {"what": "model v7's gate (one GO / NO-GO per regime, each independent)", "registration": NAME,
           "switches": V7.V7.as_dict(), "go_rule": reg["gate"]["GO rule"]}
    bad = verify_freeze(reg)
    if bad:
        doc["freeze"] = {"verdict": "REFUSED", "why": bad}
        for k in REGIMES:
            doc[k] = {"verdict": "NO-GO", "why": ["REFUSED: the freeze does not hold"]}
        return doc
    doc["freeze"] = {"verdict": "HOLDS", "files": len(reg["freeze"]["files"])}
    judge = {"r3": lambda x: r3_clause(x),
             "prefill": lambda x: dict(prefill_clauses(x["means"], x["P_A"]),
                                       information={"B3 84 cells (SEEN, never a clause)": x.get("b3_information"),
                                                    "estimate": x.get("estimate"),
                                                    "estimate_cells": x.get("estimate_cells")}),
             "decode": lambda x: dict(decode_clauses(x["cal_verdict"], x["cal_stats"], x.get("pooled")),
                                      detail={k: x.get(k) for k in ("cal_why", "rho", "constants", "fix", "per_model")})}
    for k in REGIMES:
        try:
            got = comp[k](V7.V7)
            doc[k] = judge[k](got)
        except Exception as exc:                                         # noqa: BLE001
            doc[k] = {"verdict": "NO-GO", "why": [f"REFUSED: {type(exc).__name__}: {exc}"]}
            continue
        if ablation:
            rungs = {}
            for lab, sw in ABLATION[k].items():
                try:
                    rungs[lab] = judge[k](comp[k](sw))
                except Exception as exc:                                 # noqa: BLE001
                    rungs[lab] = {"verdict": f"NOT READ: {type(exc).__name__}: {exc}"}
            doc[k]["ablation (DESCRIPTIVE)"] = rungs
    return doc


def lines(doc: dict) -> list[str]:
    head = {"r3": "R3", "prefill": "PREFILL", "decode": "DECODE"}
    L = []
    for k in REGIMES:
        g = doc[k]
        L.append(f"{head[k]}: {g['verdict']}" + (": " + "; ".join(g.get("why") or []) if g.get("why") else "")
                 + (": this regime may go to ONE blind" if g["verdict"] == "GO" else ": no blind for this regime"))
    if doc["freeze"]["verdict"] != "HOLDS":
        L.append("FREEZE: REFUSED: " + "; ".join(doc["freeze"]["why"][:10]))
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="write gate.json and gate.txt here")
    a = ap.parse_args(argv)
    if not REG_PATH.is_file():
        print(f"REFUSED: no registration at {REG_PATH}")
        return 2
    doc = evaluate(json.loads(REG_PATH.read_text()))
    doc["registration_sha256"] = sha256(REG_PATH)
    L = lines(doc)
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "gate.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        (a.out / "gate.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if all(doc[k]["verdict"] == "GO" for k in REGIMES) else 1


if __name__ == "__main__":
    sys.exit(main())
