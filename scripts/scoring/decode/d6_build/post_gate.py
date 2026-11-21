#!/usr/bin/env python
"""POST-HOC, computed AFTER the freeze and after gate_d6 read NO-GO on the published trees: the re-score's worst
cells, and which naive branch (D6-24) each naive cell takes on CAL2, the SEEN decode blind and the new blind's
registered pages; the CAL2 fit's two rho forms under v6d and under v4d' on every naive cell (CAL2 only); and two
cell counterfactuals on the 13 v4d'-branch cells, POST-HOC. A record of a defect found after the freeze (see the
registration's 'after the gate'); it changes no choice and no verdict.

    python scripts/scoring/decode/d6_build/post_gate.py [--write | --check]"""
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import d4common as D4  # noqa: E402
import d5common as D5  # noqa: E402
import d6common as D6  # noqa: E402
import d6v as L2  # noqa: E402
import dcommon as DC  # noqa: E402
import gate_d6 as G  # noqa: E402

from moe.spec import MODEL_CONFIGS as MC  # noqa: E402

KEEP = "v4d'"


def branch(m: str, c: dict, prm, terms) -> str | None:
    if D6.RD.geometry(MC[m], int(c["T"]), c["counts"]).path != D6.RD.NAIVE:
        return None
    p = D6.V6D.price(m, int(c["T"]), c["counts"], prm, ids=D6.V4DP.ids_of(m, c), terms=terms)
    return "law" if (p.get("v6d") or {}).get("gemm") else KEEP


def lines() -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        terms, law = G.readings_law(G.readings_dir_committed(Path(td)))
    cal = G.cal_score(terms, law, D6.REPO / D4.CAL2_PUBLISHED / "results")
    s = G.rescore(terms, law, cal, D6.REPO / D4.BLIND_PUBLISHED / "results")
    rows = sorted(s["rows"], key=lambda r: -abs(r["e"]))
    L = ["the SEEN decode blind under v6d at the CAL2 refit (the gate's re-score): the 12 worst cells"]
    L += [f"  {r['model']:22s} {r['label']:10s} T {r['T']:3d} meas {r['meas'] * 1e3:7.2f} us pred {r['pred_v4'] * 1e3:7.2f} us "
          f"e {r['e']:+.2%}" for r in rows[:12]]
    prm, br = D4.ds_prm(), {}
    for grp, models in (("CAL2", DC.CAL_MODELS), ("SEEN blind", DC.BLIND_MODELS)):
        for m in models:
            out = {"law": [], KEEP: []}
            for c in DC.page_doc(m, "main")["cells"]:
                b = branch(m, c, prm, terms)
                if b:
                    out[b].append(c["label"])
                    br[(m, c["label"])] = b
            kept = out[KEEP]
            L.append(f"{grp} {m}: naive cells by D6-24 branch: law {len(out['law'])}, {KEEP} {len(kept)}"
                     + (f" ({', '.join(kept)})" if kept else ""))
    for m in D5.NEW_BLIND:
        n = {"law": 0, KEEP: 0}
        for c in D5.page5(m, "main")["cells"]:
            b = branch(m, c, prm, terms)
            if b:
                n[b] += 1
        L.append(f"new blind {m}: naive main cells by D6-24 branch: law {n['law']}, {KEEP} {n[KEEP]}")
    e = np.array([r["e"] for r in s["rows"]])
    kept = np.array([br.get((r["model"], r["label"])) == KEEP for r in s["rows"]])
    L.append(f"the re-score pooled {np.sqrt(np.mean(e ** 2)):.4f} over {e.size}; its {int(kept.sum())} naive cells on the v4d' "
             f"branch read rms {np.sqrt(np.mean(e[kept] ** 2)):.4f} mean {e[kept].mean():+.4f}; the other {int((~kept).sum())} "
             f"cells read rms {np.sqrt(np.mean(e[~kept] ** 2)):.4f} (ARITHMETIC on the gate's rows, POST-HOC, never a choice)")
    L.append("CAL2's fit, both rho forms (CAL2 only; CAL constants in us; in-sample rms over 91 kept cells and sub-wave rms, "
             "the registered selection rule):")
    for nm, lw in (("v6d (D6-24)", L2.variant()), ("v4d' on every naive cell", L2.variant(naive="v4dp"))):
        c = L2.cal2(lw)
        for f, x in c["FIT"]["forms"].items():
            L.append(f"  {nm:26s} {f:6s} T0 {x['T0_GR_ms'] * 1e3:+.2f} s_naive {x['s_naive_GR_ms'] * 1e3:+.2f} "
                     f"s_block {x['s_block_GR_ms'] * 1e3:+.2f} rms {x['in_sample']['rms']:.4f} sub-wave {x['sub_wave']['rms']:.4f}"
                     + ("  (selected)" if f == c["FIT"]["selected"] else ""))
    import r3_timing_model_v4d as V4D
    tp, _ = D4.tile_source()
    fz = cal["FIT"]["forms"]["fixed"]
    v4 = L2.cal2(L2.variant(naive="v4dp"))["FIT"]["forms"]["scaled"]
    up = dict(fz, s_naive_GR_ms=fz["s_naive_GR_ms"] + (v4["s_naive_GR_ms"] - fz["s_naive_GR_ms"]))
    prm_off = V4D.params_from(up, rho_form="fixed", tile=tp)
    prm_sc = V4D.params_from(dict(fz, s_naive_GR_ms=v4["s_naive_GR_ms"]), rho_form="scaled", tile=tp)
    E1, E2 = [], []
    for r in s["rows"]:
        if br.get((r["model"], r["label"])) != KEEP:
            continue
        c = next(c for c in DC.page_doc(r["model"], "main")["cells"] if c["label"] == r["label"])
        ids = D6.V4DP.ids_of(r["model"], c)
        for prm, E in ((prm_off, E1), (prm_sc, E2)):
            p = D6.V6D.price(r["model"], int(c["T"]), c["counts"], prm, ids=ids, terms=terms)
            E.append(DC.e_dec(p["ms"], r["meas"]))
    st = lambda x: f"rms {np.sqrt(np.mean(np.square(x))):.4f} mean {np.mean(x):+.4f}"  # noqa: E731
    d = (v4["s_naive_GR_ms"] - fz["s_naive_GR_ms"]) * 1e3
    L.append(f"COUNTERFACTUALS on the 13 v4d'-branch cells (POST-HOC, never a choice): the fixed rho form with s_naive raised "
             f"by {d:+.2f} us alone: {st(E1)}; the scaled rho form with s_naive {v4['s_naive_GR_ms'] * 1e3:+.2f} us "
             f"(v4d' everywhere's CAL2 fit): {st(E2)}")
    return L


if __name__ == "__main__":
    sys.exit(L2.emit(lines(), "post_gate", sys.argv[1:]))
