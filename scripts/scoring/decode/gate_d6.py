#!/usr/bin/env python
"""The new blind's launch gate under the sixth addendum (registration 2026-11-10-decode-addendum6-gh200; the owner's two
clauses as addendum 4 recorded them and addendum 5 coded them): one GO / NO-GO line. The new blind (addendum 5's
pages, units and bars) may be launched only on GO, and finalise_decode_d6.py refuses unless this reads GO. Its output
on the published trees, computed after v6d was frozen, is committed as scripts/scoring/decode/gate_d6.out.txt.
This is the last decode round under the owner's stopping rule (2026-11-10): on NO-GO decode closes.

    python scripts/scoring/decode/gate_d6.py <readings dir> <CAL2 results tree> <decode blind results tree> [<out dir>]

<readings dir> holds pc.read.json (rental PC's reader), pc2.read.json (PC2's reader), pc_spans.json (addendum 6's span
reading of PC's pages) and ds.readings.json (DS's reader). Prints exactly one line, starting GO or NO-GO; exit 0 on GO,
1 on NO-GO. With <out dir>, writes gate.json and gate.txt beside. GO needs every clause; anything else (a refusal
included) is NO-GO:
  READINGS  every reading the law needs is READ (d6common.blocking), d6common.law_from on them gives the registered
            law constants (r3_timing_model_v6d's numbers, each to 1e-9 relative), and DS's readings give addendum 5's
            registered v5d terms (CALL, ALIGN, SLOPE; d5common.terms_from): v6d is frozen, never refitted here
  TREES     the CAL2 tree under v4d' reproduces the committed CAL2 score, and the decode blind tree under the committed
            finalisation reproduces the committed blind score (d4common.reproduce, to 1e-9)
  CAL       (the owner's clause 1) the CAL2 tree scored under v6d (score_cal_d3's rules: the three CAL constants refitted
            on v6d's body, the addendum-3 stop rule on CAL2's kept cells at 3% rms, 8% worst cell, 1.5% |mean|) reads GO
  RE-SCORE  (the owner's clause 2) the published decode blind pages (five SEEN models) priced by v6d at the CAL2 refit
            (finalise_decode.frozen and .predict, frozen) and scored by score_blind_d3.score (frozen) read an E1-DEC
            pooled rms under 4.0% over the committed 220 cells. A WEAK SANITY FLOOR on SEEN pages: those pages were read
            by the 2026-11-04 diagnoses and re-scored under v6d variants in design-D6; never a test and never a
            certification; per-model figures at the owner's bars are reported, never claimed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d3common as D3  # noqa: E402
import d4common as D4  # noqa: E402
import d5common as D5  # noqa: E402
import d6common as D6  # noqa: E402
import dcommon as DC  # noqa: E402
import finalise_decode as FN  # noqa: E402

Refused = FN.Refused
V4DP, V5D, V6D = D3.V4DP, D5.V5D, D6.V6D
LABEL = "RE-SCORE: " + D6.WEAK_FLOOR + "; it changes no registered verdict (the decode blind FAIL of 2026-11-04 stands)"
FILES = {"pc": "pc.read.json", "pc2": "pc2.read.json", "spans": "pc_spans.json", "ds": D5.DS_READINGS}


def readings_dir_committed(tmp: Path) -> Path:
    """The committed readings gathered in one directory (the registration and the tests call the gate on it)."""
    tmp.mkdir(parents=True, exist_ok=True)
    for k, src in (("pc", D6.PC_READ), ("pc2", D6.PC2_READ), ("spans", D6.SPANS), ("ds", D5.DS_READINGS_COMMITTED)):
        (tmp / FILES[k]).write_bytes((D6.REPO / src).read_bytes())
    return tmp


def readings_law(rdir: Path, registered: dict | None = None) -> tuple[V5D.Terms, V6D.Law]:
    """(v6d's v5d terms, the law) after the READINGS clause; Refused on any failure."""
    try:
        got = {k: json.loads((Path(rdir) / f).read_text()) for k, f in FILES.items()}
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Refused(f"readings: {exc}") from exc
    if got["ds"].get("addendum") != D4.NAME:
        raise Refused(f"the DS readings were not written by read_ds.py under {D4.NAME}")
    if got["pc"].get("registration") != "2026-11-06-pc-gh200" or got["pc2"].get("registration") != "2026-11-10-pc2-gh200":
        raise Refused("the PC or PC2 readings were not written by their registered readers")
    block = D5.blocking(got["ds"]) + D6.blocking(got["pc"], got["pc2"], got["spans"])
    if block:
        raise Refused("BLOCKING: " + "; ".join(block))
    why = D5.same_terms(D5.terms_from(got["ds"]), D5.registered_terms())
    if why:
        raise Refused("DS's readings do not give addendum 5's registered terms: " + "; ".join(why[:5]))
    law = D6.law_from(got["pc"], got["pc2"], got["spans"], got["ds"])
    why = D6.same_law(law, registered if registered is not None else D6.registered_law())
    if why:
        raise Refused("law_from on these readings is not the registered v6d: " + "; ".join(why[:5]))
    return D6.registered_terms(), V6D.Law({k: v for k, v in _tuples(law).items()})


def _tuples(law: dict) -> dict:
    return {"C0": {t: tuple(tuple(p) for p in v) for t, v in law["C0"].items()},
            "L_TABLE": (tuple(law["L_TABLE"][0]), tuple(law["L_TABLE"][1])), "X_SAT": dict(law["X_SAT"]),
            "KAPPA": {t: tuple(tuple(p) for p in v) for t, v in law["KAPPA"].items()}, "N_IF": dict(law["N_IF"])}


def frozen_at(cal: dict) -> dict:
    """finalise_decode.frozen at the CAL2 refit whatever the stop rule reads (the RE-SCORE clause reads it either way),
    with the owner's bars."""
    fz = FN.frozen(dict(cal, STOP=dict(cal.get("STOP") or {}, verdict="GO")))
    fz["bars"] = dict(D6.BARS_OWNER)
    return fz


def rescore(terms: V5D.Terms, law: V6D.Law, cal: dict, blind_tree: Path) -> dict:
    """The SEEN decode blind re-scored under v6d at `cal`'s refit (the RE-SCORE clause)."""
    import score_blind_d3 as SB3
    fz = frozen_at(cal)
    fz["model"] = {"name": "v6d", "fixes": V4DP.ALL.as_dict(), "terms": terms.as_dict(), "law": law.as_dict()}
    with D6.v6d(terms, law), D3.v4dp(V4DP.ALL):
        rows = FN.predict(fz, models=DC.BLIND_MODELS)
    with D6.v6d(terms, law):
        return SB3.score(Path(blind_tree), DC.registration(DC.REPO, "blind"), {"frozen": fz, "cells": rows})


def worst_cells(cal: dict, n: int = 5) -> list[dict]:
    rs = sorted(cal.get("rows") or [], key=lambda r: -abs(r["e"]))
    return [{"key": r["key"], "path": r["path"], "meas_us": DC.r6(r["meas_ms"] * 1e3), "pred_us": DC.r6(r["pred_ms"] * 1e3),
             "e": r["e"]} for r in rs[:n]]


def cal_score(terms: V5D.Terms, law: V6D.Law, cal2_tree: Path, tile=None) -> dict:
    import score_cal_d3 as SC3
    tp, src = tile if tile is not None else D4.tile_source()
    with D6.v6d(terms, law):
        return SC3.score(Path(cal2_tree), DC.registration(DC.REPO, "cal"), tp, src, settle=False)


def evaluate(rdir: Path, cal2_tree: Path, blind_tree: Path, *, registered: dict | None = None,
             want: dict | None = None, want_blind: dict | None = None, tile=None) -> tuple[dict, dict | None]:
    """(the gate's document, CAL2's score under v6d or None when refused before it)."""
    doc = {"what": "the new blind's launch gate under addendum 6 (the owner's two clauses, 2026-11-04; coded in addendum 5)",
           "addendum": D6.NAME, "rescore_rms_max": D6.RESCORE_RMS_MAX, "rescore_cells": D6.RESCORE_CELLS, "label": LABEL,
           "stopping rule": D6.STOPPING_RULE}
    try:
        terms, law = readings_law(rdir, registered)
        tp, src = tile if tile is not None else D4.tile_source()
        rep = D4.reproduce(Path(cal2_tree), Path(blind_tree), want=want, want_blind=want_blind, tile=(tp, src))
        if rep["verdict"] != "REPRODUCES":
            raise Refused("TREES: " + "; ".join(rep["CAL2"]["why"] + rep["decode blind"]["why"]))
    except Refused as exc:
        return dict(doc, verdict="NO-GO", why=[f"REFUSED: {exc}"]), None
    doc["READINGS"] = "the registered v6d (law and terms)"
    doc["TREES"] = "REPRODUCES"
    cal = cal_score(terms, law, cal2_tree, (tp, src))
    stop = cal.get("STOP") or {}
    sel = (cal.get("FIT") or {}).get("selected")
    doc["CAL"] = {"verdict": stop.get("verdict"), "why": stop.get("why") or [],
                  "E1_CAL_IN_SAMPLE": (cal.get("E1_CAL_IN_SAMPLE") or {}).get("stats"),
                  "bars": (cal.get("E1_CAL_IN_SAMPLE") or {}).get("bars"),
                  "constants": None if sel is None else {k: cal["FIT"]["forms"][sel].get(k)
                                                         for k in ("T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms")},
                  "rho": sel, "worst_cells": worst_cells(cal)}
    why = []
    if stop.get("verdict") != "GO":
        why.append(f"CAL: CAL2 under v6d reads {stop.get('verdict')}"
                   + (f" ({'; '.join(stop.get('why') or [])}; worst cell {doc['CAL']['worst_cells'][0]['key']} "
                      f"{doc['CAL']['worst_cells'][0]['e']:+.2%})" if doc["CAL"]["worst_cells"] else ""))
    if sel is None:
        doc["RESCORE"] = {"verdict": "NOT READ: the CAL2 fit selected no rho form", "label": LABEL}
        why.append("RE-SCORE: not read (no CAL2 fit)")
    else:
        s = rescore(terms, law, cal, blind_tree)
        e = s.get("E1_DEC") or {}
        pooled = (e.get("pooled") or {}).get("stats") or {}
        rms = pooled.get("rms")
        ok = rms is not None and pooled.get("cells") == D6.RESCORE_CELLS and rms < D6.RESCORE_RMS_MAX
        doc["RESCORE"] = {"verdict": "PASSES" if ok else "FAILS", "label": LABEL, "pooled": pooled,
                          "cells_committed": D6.RESCORE_CELLS, "rms_max": D6.RESCORE_RMS_MAX,
                          "per_model": {m: x.get("stats") for m, x in (e.get("per_model") or {}).items()},
                          "per_model_at_owner_bars (DESCRIPTIVE, never a certification)": s.get("certified"),
                          "read_at": "the CAL2 refit under v6d" + ("" if stop.get("verdict") == "GO" else
                                                                   " (CAL reads STOP; the refit is the same either way)")}
        if pooled.get("cells") != D6.RESCORE_CELLS:
            why.append(f"RE-SCORE: pools {pooled.get('cells')} cells, not the committed {D6.RESCORE_CELLS}")
        elif not ok:
            why.append(f"RE-SCORE: the SEEN decode blind under v6d reads pooled rms "
                       f"{'none' if rms is None else f'{rms:.2%}'}, not under {D6.RESCORE_RMS_MAX:.1%} (the weak sanity floor)")
    doc["verdict"] = "GO" if not why else "NO-GO"
    doc["why"] = why
    doc["law"] = law.as_dict()
    doc["terms"] = terms.as_dict()
    return doc, cal


def gate(rdir: Path, cal2_tree: Path, blind_tree: Path, **kw) -> dict:
    return evaluate(rdir, cal2_tree, blind_tree, **kw)[0]


def line(g: dict) -> str:
    """The operator's one line."""
    if g["verdict"] == "GO":
        c, r = g["CAL"]["E1_CAL_IN_SAMPLE"], g["RESCORE"]["pooled"]
        return (f"GO: the readings give the registered v6d; both trees reproduce; CAL2 under v6d reads GO (rms {c['rms']:.2%}, "
                f"worst {c['worst']:+.2%}); the SEEN decode blind under v6d reads pooled rms {r['rms']:.2%} < "
                f"{g['rescore_rms_max']:.1%} over {r['cells']} cells (a weak sanity floor on seen pages, never a test): "
                "launch the new blind (the last decode blind under the owner's stopping rule)")
    return ("NO-GO: " + "; ".join(g["why"]) + ": do not launch the new blind; decode closes as calibrated, not certified "
            "(the owner's stopping rule, 2026-11-10)")


def lines(g: dict) -> list[str]:
    L = [line(g), f"label: {g['label']}"]
    for k in ("CAL", "RESCORE"):
        if k in g:
            L.append(f"{k}: " + json.dumps(g[k], sort_keys=True, default=str))
    for k in ("law", "terms"):
        if k in g:
            L.append(f"{k}: " + json.dumps(g[k], sort_keys=True))
    return L


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if len(a) not in (3, 4):
        print(__doc__)
        return 2
    g = gate(Path(a[0]), Path(a[1]), Path(a[2]))
    if len(a) == 4:
        out = Path(a[3])
        out.mkdir(parents=True, exist_ok=True)
        (out / "gate.json").write_text(json.dumps(g, indent=1, default=str) + "\n")
        (out / "gate.txt").write_text("\n".join(lines(g)) + "\n")
    print(line(g))
    return 0 if g["verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
