"""The lock options on rental B2's frozen NATIVE cells (JetMoE, Phi-3.5, Qwen1.5, deepseek-v2-lite): what each v5 term
moves, and the size of each option's unmodelled risk, as a change in each blind model's mean and each class mean.
LEVERS ONLY: no B2 cell has been measured; these are model-against-model differences, never a prediction of record.

(a) 1410 MHz: v5 nominal (F1 at A2's board constants), and F1 x 0.8 / x 1.2 (the model-to-model spread of the sag at
    a fixed kernel length on one board, sag_pages.json: about +-20 percent), and F1 off (the size of the term).
(b) 1005 MHz: F1 off (GONE); v5 with p_128 from 1410 (4.0) against p 2.0 (S1-M2's 1005 page) and with v4's carried bw
    against bw x 0.80 (S1-M2: DRAM-bound cells' achieved bandwidth at 1005 over 1410, lockx.json).
Writes b2risk.json."""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True
import v5price as V  # noqa: E402
from v5price import A2, FN, RG, SCORE, TC, H, params_for  # noqa: E402


@contextmanager
def bw_scaled(f):
    """v4's carried bw (the CAL engine's fit, x[2]) times f for one block; the Fit object restored on exit."""
    fit = RG.pricer().f2
    keep = fit.x
    try:
        x = keep.copy()
        x[2] = x[2] * f
        fit.x = x
        yield
    finally:
        fit.x = keep


def b2_cells():
    out = []
    for unit, model, name, _drop, _est in RG.B_TIMED_PAGES:
        doc = RG.hist("B", name)
        for c in doc["cells"]:
            if "native" in c["arms"]:
                out.append({"unit": unit, "model": model, "page": name, "doc": doc, "c": c,
                            "cls": TC.tile_class(int(doc["block_m"]), int(doc["group_m"])), "n": int(c["n"]), "label": c["label"]})
    return out


def main():
    RG.pricer()   # built at v4's own constants before any swap
    F3P, P = V.f3_params(), V.knee_p()
    sel = FN.selections(SCORE)
    cells = b2_cells()
    res = {"cells": len(cells)}
    runs = {
        1410: {"v4": None, "v5": dict(F1=True, F2=P, F3=F3P, key="v3", F4=True),
               "v5 sag x0.8": dict(F1=True, F2=P, F3=F3P, key="v3", F4=True, sag_scale=0.8),
               "v5 sag x1.2": dict(F1=True, F2=P, F3=F3P, key="v3", F4=True, sag_scale=1.2),
               "v5 F1 off": dict(F1=False, F2=P, F3=F3P, key="v3", F4=True)},
        1005: {"v4": None, "v5 (p 1410, bw carried)": dict(F1=True, F2=P, F3=F3P, key="v3", F4=True),
               "v5 p_128 2.0": dict(F1=True, F2={128: 2.0, 64: P[64]}, F3=F3P, key="v3", F4=True),
               "v5 bw x0.80": dict(F1=True, F2=P, F3=F3P, key="v3", F4=True, _bw=0.80),
               "v5 p_128 2.0 and bw x0.80": dict(F1=True, F2={128: 2.0, 64: P[64]}, F3=F3P, key="v3", F4=True, _bw=0.80)},
    }
    for lock, vs in runs.items():
        with A2.scoring(A2.Rules.tileA2(lock)):
            for r in cells:
                prm = params_for(sel, r["model"])
                r.setdefault(lock, {})
                for vn, kw in vs.items():
                    if kw is None:
                        r[lock][vn] = RG.price_cell(r["model"], "native", r["c"], r["doc"], prm, key=sel["key"])["ms"]
                        continue
                    kw = dict(kw)
                    f = kw.pop("_bw", None)
                    if f:
                        with bw_scaled(f):
                            got = V.price(r["model"], "native", r["c"], r["doc"], prm, lock=lock, **kw)
                    else:
                        got = V.price(r["model"], "native", r["c"], r["doc"], prm, lock=lock, **kw)
                    r[lock][vn] = None if got is None else got["ms"]
        ref = list(vs)[1]
        res[lock] = {}
        for vn in vs:
            ok = [r for r in cells if r[lock][vn] is not None and r[lock][ref] is not None]
            d = [(r, r[lock][vn] / r[lock][ref] - 1) for r in ok]
            by_model = {m: 100 * st.fmean(x for rr, x in d if rr["model"] == m) for m in sorted({rr["model"] for rr, _ in d})}
            by_cls = {c: 100 * st.fmean(x for rr, x in d if rr["cls"] == c) for c in ("64g1", "128g1", "128g16")}
            res[lock][vn] = {"vs": ref, "by_model_pct": by_model, "by_class_pct": by_cls,
                             "worst_pct": 100 * max((x for _, x in d), key=abs), "cells": len(d),
                             "ms_range": [min(r[lock][vn] for r in ok), max(r[lock][vn] for r in ok)]}
            print(lock, f"{vn:28s} against {ref}: models", {k[:8]: round(v, 2) for k, v in by_model.items()},
                  "classes", {k: round(v, 2) for k, v in by_cls.items()}, "worst", round(res[lock][vn]["worst_pct"], 2))
    res["per_cell"] = [{k: v for k, v in r.items() if k not in ("doc", "c")} for r in cells]
    (H / "b2risk.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
