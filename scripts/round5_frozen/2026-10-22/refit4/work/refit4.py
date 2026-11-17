"""Items 1, 3, 4, 5: the oracle-bytes re-price, the four-model refit (DECLARED.txt B), the pass estimates, B2's blind
cells, and the gate (DECLARED.txt C). Writes refit4.json.

(1) every 128 G 1 timed cell of the four models priced in its registered held-out fold (v5'' + F1 par, the fold's p),
    with the counted bytes of its byte-page twin substituted for M4's: w2 only, w1 only, both (DESCRIPTIVE).
(3) the four-model LOMO: p_128 refitted on the training models' counter rows (118 knee-v5 rows + C2's 20), F1 par from
    the fold's estimator pages on the timed page's board; bytes = the law bytes4 selected (C0 M4). Beside it,
    DESCRIPTIVE: C5 JITTER on w2 only (not a declared candidate; the best w2 law, excluded by the w1 guard), the
    oracle, and the "joint" sag estimator.
(4) B2's 84 frozen NATIVE cells: refit / v5'' - 1 per blind model and class, F1 par at A2's values.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import r4
from r4 import k5

sys.path.insert(0, str(r4.REPO / "scripts/scoring/cal2"))
import read_c2 as RC  # noqa: E402  (pure part only: power, b2_cells, pass_estimates, to_b; chain() is never called)
sys.path.insert(0, str(Path.home() / "code/moe-kernels-work/2026-10-21/design-v5/work"))
import b2risk as B  # noqa: E402

P_GRID = [3 + 0.25 * i for i in range(53)]
O, Q, M, X = r4.MODELS4
REG_FOLD_P = {O: 6.25, Q: 5.5, M: 5.25, X: 5.75}            # knee-v5 forms.json v5'' LOMO folds; X: the SEEN 3-model fit
REG_MEANS = {O: 0.13, Q: -0.82, M: 3.13, X: 4.57, "mixtral-8x7b C2": 3.37}
THETA_LOMO = json.load(open("bytes4.json"))["k0"]["C5 JITTER"]["LOMO"]


def twins():
    """counted reads per (model, board-of-timed-page, n, label) from the byte-page twin."""
    pages = {(O, "A2"): (r4.A2T, "b128g1"), (Q, "A2"): (r4.A2T, "b128g1"), (M, "A2"): (r4.C2T, "b128g1"),
             (M, "C2"): (r4.C2T, "b128g1"), (X, "C2"): (r4.C2T, "b128g1"), (M, "A2 via S1"): (r4.S1T, "b128g1")}
    out = {}
    for (m, b), (root, unit) in pages.items():
        for c in r4.counter_cells(root, m, unit):
            out[(m, b, c["c"]["n"], c["c"]["label"])] = c["reads"]
    return out


def timed_cells():
    a2 = [dict(r, board="A2") for r in r4.a2_rows() if r["cls"] == "128g1"]
    c2x = [dict(r, board="C2") for r in r4.c2_8x22b_rows()]
    c2m = [dict(r, board="C2") for r in r4.c2_8x7b_rows()]
    return a2, c2x, c2m


def sag_fits():
    f = {b: {m: r4.sag_page_fit(root, m) for m in (O, M)} for b, root in (("A2", r4.A2T), ("C2", r4.C2T))}
    assert [round(v, 2) for v in f["A2"][O]] == [29.65, 300.0] and [round(v, 2) for v in f["A2"][M]] == [54.39, 550.0], f
    assert [round(v, 2) for v in f["C2"][O]] == [30.6, 325.0] and [round(v, 2) for v in f["C2"][M]] == [52.58, 500.0], f
    return f


def joint_fit(board):
    root = {"A2": r4.A2T, "C2": r4.C2T}[board]
    pts = []
    for m in (O, M):
        pts += k5.S.pts_of(sorted(Path(root).glob(f"*-{m}-f128-r3-counters/lock1410/r3c-g*.json")), 1410.0)
    r = k5.S.fit(pts)
    return (r["A_us"], r["tau_us"])


def fold_F1(fits, board, H, kind="par"):
    keep = [m for m in (O, M) if m != H]
    if kind == "par" or len(keep) == 1:
        return r4.par([fits[board][m] for m in keep])
    return joint_fit(board)


def counter_rows():
    base = k5.capture_rows(("fit",))
    assert len(base) == 118, len(base)
    keep = k5.KF.PAGES
    k5.KF.PAGES = [(r4.C2T, M, "b128g1", 1410, "c2"), (r4.C2T, X, "b128g1", 1410, "c2")]
    try:
        extra = k5.capture_rows(("c2",))
    finally:
        k5.KF.PAGES = keep
    return base, extra


def fit_p(rows):
    best = None
    form = k5.Form("null", k5.BW_CARRIED)
    for p in P_GRID:
        e = k5.eval_rows(rows, form, p)
        s = k5.rms(e)
        if best is None or s < best[1] - 1e-12:
            best = (p, s)
    return best


def mean_e(cells, fn):
    e = [fn(r) / r["meas"] - 1 for r in cells]
    return r4.summ(e), e


def main():
    out = {}
    fits = sag_fits()
    out["sag_page_fits"] = {b: {m: list(v) for m, v in d.items()} for b, d in fits.items()}
    a2, c2x, c2m = timed_cells()
    print(f"timed 128 G 1 cells: A2 {[(m, sum(1 for r in a2 if r['model'] == m)) for m in (O, Q, M)]}, C2 8x22B {len(c2x)}, C2 8x7B {len(c2m)}")
    TW = twins()
    by_model = {O: [r for r in a2 if r["model"] == O], Q: [r for r in a2 if r["model"] == Q], M: [r for r in a2 if r["model"] == M], X: c2x}

    def twin(r, board_key=None):
        b = board_key or r["board"]
        return TW[(r["model"], b, r["n"], r["label"])]

    def priced(r, p, sag, mode="m4", theta=None):
        """mode: m4 (the active law's reads), w2 / w1 / both (counted twin substituted), c5w2 (C5 on w2, M4 on w1)."""
        if mode == "m4":
            return r4.cell_ms(r, p, sag)
        pr, _ = r4.m4_reads(r["model"], r["c"], r["doc"], p)
        if mode == "c5w2":
            with r4.law("JITTER", theta=theta):
                pj, _ = r4.m4_reads(r["model"], r["c"], r["doc"], p)
            reads = {"w1": pr["w1"], "w2": pj["w2"]}
        else:
            tw = twin(r)
            reads = {"w1": tw["w1"] if mode in ("w1", "both") else pr["w1"], "w2": tw["w2"] if mode in ("w2", "both") else pr["w2"]}
        return r4.cell_ms(r, p, sag, reads=reads)

    # ---------------------------------------------------------------- (1) oracle bytes, registered folds
    print("\n(1) ORACLE BYTES in the registered folds (v5'' + F1 par, fold p): held-out 128 G 1 mean e, %")
    reg_cfg = {O: (REG_FOLD_P[O], fold_F1(fits, "A2", O)), Q: (REG_FOLD_P[Q], fold_F1(fits, "A2", Q)),
               M: (REG_FOLD_P[M], fold_F1(fits, "A2", M)), X: (REG_FOLD_P[X], fold_F1(fits, "C2", X))}
    sets = dict(by_model)
    sets["mixtral-8x7b C2"] = c2m
    cfg = dict(reg_cfg)
    cfg["mixtral-8x7b C2"] = (REG_FOLD_P[M], fold_F1(fits, "C2", M))
    out["oracle"] = {}
    for key, cells in sets.items():
        p, sag = cfg[key]
        row = {"p": p, "F1": [round(sag[0], 3), sag[1]], "cells": len(cells)}
        for mode in ("m4", "w2", "w1", "both"):
            s, e = mean_e(cells, lambda r, _m=mode: priced(r, p, sag, _m))
            row[mode] = s
            if mode == "m4":
                row["per_cell_m4"] = [round(100 * x, 2) for x in e]
            if mode == "w2":
                row["per_cell_w2"] = [round(100 * x, 2) for x in e]
        row["w2_byte_share_pts"] = round(row["m4"]["mean"] - row["w2"]["mean"], 2)
        row["w1_byte_share_pts"] = round(row["m4"]["mean"] - row["w1"]["mean"], 2)
        row["left_after_both_pts"] = row["both"]["mean"]
        out["oracle"][key] = row
        print(f"  {key:16s} n {len(cells)} p {p} F1 ({sag[0]:.2f}, {sag[1]:.0f}) | M4 {row['m4']['mean']:+5.2f} (reg {REG_MEANS[key]:+.2f}) | "
              f"w2 counted {row['w2']['mean']:+5.2f} | w1 counted {row['w1']['mean']:+5.2f} | both {row['both']['mean']:+5.2f} | "
              f"w2 carries {row['w2_byte_share_pts']:+.2f} pts")
        assert abs(row["m4"]["mean"] - REG_MEANS[key]) < 0.015, (key, row["m4"]["mean"])
        sys.stdout.flush()
    # 8x7B A2 cells with S1's counted bytes instead of C2's (board check)
    p, sag = reg_cfg[M]
    s, _ = mean_e(by_model[M], lambda r: r4.cell_ms(r, p, sag, reads={"w1": r4.m4_reads(r["model"], r["c"], r["doc"], p)[0]["w1"],
                                                                         "w2": twin(r, "A2 via S1")["w2"]}))
    out["oracle"]["mixtral-8x7b A2, S1's counted w2"] = s
    print(f"  mixtral-8x7b A2 with S1's counted w2 instead of C2's: {s['mean']:+.2f}")

    # ---------------------------------------------------------------- (3) four-model refit
    base, extra = counter_rows()
    rows = base + extra
    print(f"\n(3) counter rows: {len(base)} knee-v5 + {len(extra)} C2 = {len(rows)}")
    chk = {H: fit_p([r for r in base if r["model"] != H])[0] for H in (O, Q, M)}
    chk["SEEN3"] = fit_p(base)[0]
    print(f"  reproduction on the 118 rows: folds {chk} (knee-v5: 6.25 / 5.5 / 5.25, SEEN 5.75)")
    assert chk == {O: 6.25, Q: 5.5, M: 5.25, "SEEN3": 5.75}, chk
    p4 = {H: fit_p([r for r in rows if r["model"] != H]) for H in r4.MODELS4}
    p4_seen = fit_p(rows)
    fam = {8: fit_p([r for r in rows if r4.E_FAMILY[r["model"]] == 64]), 64: fit_p([r for r in rows if r4.E_FAMILY[r["model"]] == 8])}
    print(f"  four-model p_128: LOMO {{{', '.join(f'{r4.SHORT[h]}: {v[0]}' for h, v in p4.items())}}}, SEEN {p4_seen[0]} (rms {100 * p4_seen[1]:.2f}%), "
          f"E-family-out: fit E64 -> {fam[8][0]}, fit E8 -> {fam[64][0]}")
    # held-out cycles per model at the fold p (descriptive)
    form = k5.Form("null", k5.BW_CARRIED)
    cyc = {}
    for H in r4.MODELS4:
        hr = [r for r in rows if r["model"] == H]
        e = k5.eval_rows(hr, form, p4[H][0])
        cyc[H] = {g: r4.summ([x for r, x in zip(hr, e) if r["gemm"] == g and r["G"] == 1]) for g in ("w1", "w2")}
    out["p128"] = {"LOMO": {h: v[0] for h, v in p4.items()}, "SEEN": p4_seen[0], "SEEN_rms_pct": round(100 * p4_seen[1], 2),
                   "E-family-out (key = predicted family)": {str(k): v[0] for k, v in fam.items()}, "reproduction_118": chk,
                   "held_out_cycles_G1": cyc}
    print("  held-out cycles at 128 G 1 (counted bytes, null form): " + "; ".join(
        f"{r4.SHORT[h]} w1 {cyc[h]['w1']['mean']:+.2f} w2 {cyc[h]['w2']['mean']:+.2f}" for h in r4.MODELS4))

    def lomo_table(mode, sagkind="par", pmap=None, label=""):
        pmap = pmap or {h: p4[h][0] for h in r4.MODELS4}
        res = {}
        for H, cells in list(by_model.items()) + [("mixtral-8x7b C2", c2m)]:
            hm = M if H == "mixtral-8x7b C2" else H
            board = "C2" if H in (X, "mixtral-8x7b C2") else "A2"
            sag = fold_F1(fits, board, hm, sagkind)
            th = THETA_LOMO.get(hm)
            s, e = mean_e(cells, lambda r, _p=pmap[hm], _s=sag: priced(r, _p, _s, mode, th))
            res[H] = {**s, "p": pmap[hm], "F1": [round(sag[0], 3), round(sag[1], 1)], "per_cell": [round(100 * x, 2) for x in e]}
        means = [res[h]["mean"] for h in r4.MODELS4]
        summary = {"means": {h: res[h]["mean"] for h in r4.MODELS4}, "max_abs": round(max(abs(v) for v in means), 2),
                   "spread": round(max(means) - min(means), 2), "rms_of_means": round(math.sqrt(st.fmean(v * v for v in means)), 2),
                   "pooled_cell_rms": round(100 * math.sqrt(st.fmean((x / 100) ** 2 for h in r4.MODELS4 for x in res[h]["per_cell"])), 2),
                   "worst_cell": max((x for h in r4.MODELS4 for x in res[h]["per_cell"]), key=abs),
                   "mixtral-8x7b C2": res["mixtral-8x7b C2"]["mean"]}
        print(f"  {label:44s} " + " ".join(f"{r4.SHORT[h]} {res[h]['mean']:+5.2f}" for h in r4.MODELS4)
              + f" | 8x7B C2 {res['mixtral-8x7b C2']['mean']:+5.2f} | max {summary['max_abs']} spread {summary['spread']} "
              f"pooled rms {summary['pooled_cell_rms']} worst {summary['worst_cell']}")
        sys.stdout.flush()
        return {"per_model": res, "summary": summary}

    print("\n  four-model held-out 128 G 1 means (%):")
    T = {}
    T["registered folds, M4 (= C2's table)"] = lomo_table("m4", pmap=REG_FOLD_P, label="registered folds (3-model p), M4")
    T["REFIT: four-model p, M4 (selected law)"] = lomo_table("m4", label="REFIT four-model p, M4 [selected]")
    T["four-model p, M4, joint sag (descriptive)"] = lomo_table("m4", "joint", label="four-model p, M4, joint sag [desc]")
    T["four-model p, C5 on w2 (descriptive)"] = lomo_table("c5w2", label="four-model p, C5 JITTER on w2 [desc]")
    T["four-model p, oracle w2 (descriptive)"] = lomo_table("w2", label="four-model p, counted w2 [oracle]")
    T["four-model p, oracle both (descriptive)"] = lomo_table("both", label="four-model p, counted w1+w2 [oracle]")
    out["heldout_tables"] = T

    # pass estimates (read_c2.pass_estimates: A four models, B no common bias, D Mixtral-8x7B aside)
    print("\n  pass estimates (ESTIMATE; B2's registered noise model and bars, 84 cells, R 200,000, seed 20261022):")
    out["pass"] = {}
    cells84 = RC.b2_cells()
    for name, t in T.items():
        mm = t["summary"]["means"]
        pe = RC.pass_estimates(mm[X], cells=cells84, base=[mm[O], mm[Q], mm[M]])
        out["pass"][name] = pe
        print(f"    {name:44s} {pe}")
        sys.stdout.flush()
    # reproduction of C2's registered estimate
    reg = RC.pass_estimates(4.57, cells=cells84, base=[0.13, -0.82, 3.13])
    print(f"    reproduction of C2's reading (0.13, -0.82, 3.13, 4.57): {reg} (C2: A 0.0045, B 0.0112, D 0.0049)")
    out["pass"]["C2 reading reproduction"] = reg

    # ---------------------------------------------------------------- (5) gate (DECLARED.txt C)
    sel = T["REFIT: four-model p, M4 (selected law)"]["summary"]
    g1_ = sel["max_abs"] <= 1.0 and abs(sel["mixtral-8x7b C2"]) <= 1.0
    g2_ = out["pass"]["REFIT: four-model p, M4 (selected law)"]["A (4 models)"] >= 0.5
    out["gate"] = {"G1 all |means| <= 1.0": g1_, "G2 P(A) >= 0.5": g2_, "verdict": "PASS" if g1_ and g2_ else "FAIL"}
    print(f"\n(5) GATE: G1 {g1_} (max |mean| {sel['max_abs']}, 8x7B C2 {sel['mixtral-8x7b C2']}); G2 {g2_}; verdict {out['gate']['verdict']}")
    for name in T:
        s_ = T[name]["summary"]
        print(f"    (descriptive) {name}: G1 {s_['max_abs'] <= 1.0 and abs(s_['mixtral-8x7b C2']) <= 1.0}, P(A) {out['pass'][name]['A (4 models)']}")

    # ---------------------------------------------------------------- (4) B2's blind cells
    print("\n(4) B2's 84 NATIVE cells, F1 par at A2's values: pred / v5'' - 1 (%), model against model, not evidence")
    sag128 = r4.par([fits["A2"][O], fits["A2"][M]])
    sag64 = (30.37, 300.0)        # knee-v5 par at 64x128 (A2), CARRIED; 64x128 is untouched by every variant here
    b2 = B.b2_cells()
    rows_b2 = []
    th_seen = json.load(open("bytes4.json"))["k0"]["C5 JITTER"]["SEEN"]
    for r in b2:
        bm = int(r["doc"]["block_m"])
        a = k5.cell_ms(r, None, 5.75, sag128, sag64)
        b = k5.cell_ms(r, None, p4_seen[0], sag128, sag64)
        if bm == 128:
            pr, _ = r4.m4_reads(r["model"], r["c"], r["doc"], p4_seen[0])
            with r4.law("JITTER", theta=th_seen):
                pj, _ = r4.m4_reads(r["model"], r["c"], r["doc"], p4_seen[0])
            c = r4.cell_ms(r, p4_seen[0], sag128, reads={"w1": pr["w1"], "w2": pj["w2"]})
            dw2 = pj["w2"] / pr["w2"] - 1
        else:
            c, dw2 = b, 0.0
        rows_b2.append({"model": r["model"], "cls": r["cls"], "n": r["n"], "label": r["label"], "v5pp_ms": round(a, 6),
                        "refit_pct": round(100 * (b / a - 1), 3), "c5w2_pct": round(100 * (c / a - 1), 3), "c5w2_w2_bytes_pct": round(100 * dw2, 2)})
    by = {}
    for x in rows_b2:
        for k in (f"{x['model']} {x['cls']}", f"{x['model']} (all)"):
            by.setdefault(k, []).append(x)
    out["b2"] = {}
    for k, v in sorted(by.items()):
        d = {}
        for f in ("refit_pct", "c5w2_pct", "c5w2_w2_bytes_pct"):
            vals = [x[f] for x in v]
            d[f] = {"mean": round(st.fmean(vals), 2), "min": round(min(vals), 2), "max": round(max(vals), 2)}
        d["n"] = len(v)
        out["b2"][k] = d
        print(f"  {k:28s} n {len(v):2d} | refit (p {p4_seen[0]}) {d['refit_pct']['mean']:+5.2f} ({d['refit_pct']['min']:+.2f} .. {d['refit_pct']['max']:+.2f})"
              f" | C5 on w2 [desc] {d['c5w2_pct']['mean']:+5.2f} ({d['c5w2_pct']['min']:+.2f} .. {d['c5w2_pct']['max']:+.2f})"
              f" | C5/M4 w2 bytes {d['c5w2_w2_bytes_pct']['mean']:+6.2f}")
    out["b2_cells"] = rows_b2
    out["theta_seen_c5"] = th_seen
    json.dump(out, open("refit4.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
