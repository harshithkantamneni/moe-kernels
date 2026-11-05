#!/usr/bin/env python
"""The new blind's launch gate under the fifth addendum (registration 2026-11-04-decode-addendum5-gh200; the owner's
gate recorded in addendum 4, 2026-11-04): one GO / NO-GO line. A new blind rental could be launched only on GO, and
finalise_decode_d5.py refuses unless this reads GO. On the published trees it reads NO-GO on both clauses
(scripts/scoring/decode/gate_d5.out.txt); the owner closed the decode stage on 2026-11-04 as calibrated, not certified,
and the new blind is registered and NOT RUN.

    python scripts/scoring/decode/gate_d5.py <DS readings dir> <CAL2 results tree> <decode blind results tree> [<out dir>]

Prints exactly one line, starting GO or NO-GO; exit 0 on GO, 1 on NO-GO. With <out dir>, writes gate.json and gate.txt
beside. GO needs every clause; anything else (a refusal included) is NO-GO:
  READINGS  <DS readings dir>/ds.readings.json was written by read_ds.py under addendum 4, its power check and verdict
            READ, no reading v5d needs is BLOCKING (d5common.blocking), and terms_from on it gives the terms this
            addendum registered (each number to 1e-9 relative): v5d is frozen, never refitted here
  TREES     the CAL2 tree under v4d' reproduces the committed CAL2 score, and the decode blind tree under the committed
            finalisation reproduces the committed blind score (d4common.reproduce, to 1e-9)
  CAL       (the owner's clause 1) the CAL2 tree scored under v5d (score_cal_d3's rules: the three CAL constants refitted
            on v5d's body, the addendum-3 stop rule on CAL2's kept cells) reads GO
  RE-SCORE  (the owner's clause 2) the published decode blind pages (five SEEN models) priced by v5d at the CAL2 refit
            (finalise_decode.frozen and .predict, frozen; the refit is the same whatever the stop rule reads, so this
            clause is read at it either way) and scored by score_blind_d3.score (frozen) read an E1-DEC pooled rms under
            4.0% over the committed 220 cells. A WEAK SANITY FLOOR on SEEN pages: those pages were read by the
            diagnosis that chose what DS measures; never a test and never a certification; per-model figures at the
            owner's bars are reported, never claimed
On NO-GO decode closes as calibrated, not certified.
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
import dcommon as DC  # noqa: E402
import finalise_decode as FN  # noqa: E402

Refused = FN.Refused
V4DP, V5D = D3.V4DP, D5.V5D
LABEL = "RE-SCORE: " + D5.WEAK_FLOOR + "; it changes no registered verdict (the decode blind FAIL of 2026-11-04 stands)"


def readings_terms(ds_dir: Path, registered: V5D.Terms | None = None) -> tuple[dict, V5D.Terms]:
    """(DS's readings, v5d's terms) after the READINGS clause; Refused on any failure."""
    try:
        rd = D5.load_readings(Path(ds_dir))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Refused(f"DS readings: {exc}") from exc
    if rd.get("addendum") != D4.NAME:
        raise Refused(f"the DS readings were not written by read_ds.py under {D4.NAME}")
    block = D5.blocking(rd)
    if block:
        raise Refused("BLOCKING: " + "; ".join(block))
    terms = D5.terms_from(rd)
    why = D5.same_terms(terms, registered if registered is not None else D5.registered_terms())
    if why:
        raise Refused("terms_from on these readings is not the registered v5d: " + "; ".join(why[:5]))
    return rd, terms


def frozen_at(cal: dict) -> dict:
    """finalise_decode.frozen at the CAL2 refit whatever the stop rule reads (the RE-SCORE clause reads it either way),
    with the owner's bars."""
    fz = FN.frozen(dict(cal, STOP=dict(cal.get("STOP") or {}, verdict="GO")))
    fz["bars"] = dict(D5.BARS_OWNER)
    return fz


def rescore(terms: V5D.Terms, cal: dict, blind_tree: Path) -> dict:
    """The SEEN decode blind re-scored under v5d at `terms` and `cal`'s refit (the RE-SCORE clause)."""
    import score_blind_d3 as SB3
    fz = frozen_at(cal)
    fz["model"] = {"name": "v5d", "fixes": V4DP.ALL.as_dict(), "terms": terms.as_dict()}
    with D5.v5d(terms), D3.v4dp(V4DP.ALL):
        rows = FN.predict(fz, models=DC.BLIND_MODELS)
    with D5.v5d(terms):
        return SB3.score(Path(blind_tree), DC.registration(DC.REPO, "blind"), {"frozen": fz, "cells": rows})


def worst_cells(cal: dict, n: int = 5) -> list[dict]:
    rs = sorted(cal.get("rows") or [], key=lambda r: -abs(r["e"]))
    return [{"key": r["key"], "path": r["path"], "meas_us": DC.r6(r["meas_ms"] * 1e3), "pred_us": DC.r6(r["pred_ms"] * 1e3),
             "e": r["e"]} for r in rs[:n]]


def evaluate(ds_dir: Path, cal2_tree: Path, blind_tree: Path, *, registered: V5D.Terms | None = None,
             want: dict | None = None, want_blind: dict | None = None, tile=None) -> tuple[dict, dict | None]:
    """(the gate's document, CAL2's score under v5d or None when refused before it)."""
    import score_cal_d3 as SC3
    doc = {"what": "the new blind's launch gate (owner, 2026-11-04; recorded in addendum 4, coded in addendum 5)",
           "addendum": D5.NAME, "rescore_rms_max": D5.RESCORE_RMS_MAX, "rescore_cells": D5.RESCORE_CELLS, "label": LABEL}
    try:
        _rd, terms = readings_terms(ds_dir, registered)
        tp, src = tile if tile is not None else D4.tile_source()
        rep = D4.reproduce(Path(cal2_tree), Path(blind_tree), want=want, want_blind=want_blind, tile=(tp, src))
        if rep["verdict"] != "REPRODUCES":
            raise Refused("TREES: " + "; ".join(rep["CAL2"]["why"] + rep["decode blind"]["why"]))
    except Refused as exc:
        return dict(doc, verdict="NO-GO", why=[f"REFUSED: {exc}"]), None
    doc["terms"] = terms.as_dict()
    doc["TREES"] = "REPRODUCES"
    with D5.v5d(terms):
        cal = SC3.score(Path(cal2_tree), DC.registration(DC.REPO, "cal"), tp, src, settle=False)
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
        why.append(f"CAL: CAL2 under v5d reads {stop.get('verdict')}"
                   + (f" ({'; '.join(stop.get('why') or [])}; worst cell {doc['CAL']['worst_cells'][0]['key']} "
                      f"{doc['CAL']['worst_cells'][0]['e']:+.2%})" if doc["CAL"]["worst_cells"] else ""))
    if sel is None:
        doc["RESCORE"] = {"verdict": "NOT READ: the CAL2 fit selected no rho form", "label": LABEL}
        why.append("RE-SCORE: not read (no CAL2 fit)")
    else:
        s = rescore(terms, cal, blind_tree)
        e = s.get("E1_DEC") or {}
        pooled = (e.get("pooled") or {}).get("stats") or {}
        rms = pooled.get("rms")
        ok = rms is not None and pooled.get("cells") == D5.RESCORE_CELLS and rms < D5.RESCORE_RMS_MAX
        doc["RESCORE"] = {"verdict": "PASSES" if ok else "FAILS", "label": LABEL, "pooled": pooled,
                          "cells_committed": D5.RESCORE_CELLS, "rms_max": D5.RESCORE_RMS_MAX,
                          "per_model": {m: x.get("stats") for m, x in (e.get("per_model") or {}).items()},
                          "per_model_at_owner_bars (DESCRIPTIVE, never a certification)": s.get("certified"),
                          "read_at": "the CAL2 refit under v5d" + ("" if stop.get("verdict") == "GO" else
                                                                   " (CAL reads STOP; the refit is the same either way)")}
        if pooled.get("cells") != D5.RESCORE_CELLS:
            why.append(f"RE-SCORE: pools {pooled.get('cells')} cells, not the committed {D5.RESCORE_CELLS}")
        elif not ok:
            why.append(f"RE-SCORE: the SEEN decode blind under v5d reads pooled rms "
                       f"{'none' if rms is None else f'{rms:.2%}'}, not under {D5.RESCORE_RMS_MAX:.1%} (the weak sanity floor)")
    doc["verdict"] = "GO" if not why else "NO-GO"
    doc["why"] = why
    return doc, cal


def gate(ds_dir: Path, cal2_tree: Path, blind_tree: Path, **kw) -> dict:
    return evaluate(ds_dir, cal2_tree, blind_tree, **kw)[0]


def line(g: dict) -> str:
    """The operator's one line."""
    if g["verdict"] == "GO":
        c, r = g["CAL"]["E1_CAL_IN_SAMPLE"], g["RESCORE"]["pooled"]
        return (f"GO: DS readings give the registered v5d; both trees reproduce; CAL2 under v5d reads GO (rms {c['rms']:.2%}, "
                f"worst {c['worst']:+.2%}); the SEEN decode blind under v5d reads pooled rms {r['rms']:.2%} < "
                f"{g['rescore_rms_max']:.1%} over {r['cells']} cells (a weak sanity floor on seen pages, never a test): "
                "launch the new blind")
    return ("NO-GO: " + "; ".join(g["why"]) + ": do not launch the new blind; decode closes as calibrated, not certified")


def lines(g: dict) -> list[str]:
    L = [line(g), f"label: {g['label']}"]
    for k in ("CAL", "RESCORE"):
        if k in g:
            L.append(f"{k}: " + json.dumps(g[k], sort_keys=True, default=str))
    if "terms" in g:
        L.append("terms: " + json.dumps(g["terms"], sort_keys=True))
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
