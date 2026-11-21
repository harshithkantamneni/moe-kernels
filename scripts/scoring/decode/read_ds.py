#!/usr/bin/env python
"""Reads the decode settling rental DS (registration 2026-11-04-decode-addendum4-gh200): DESCRIPTIVE, never a STOP.

    python scripts/scoring/decode/read_ds.py <DS results tree> <out dir>

The stop rules first: POWER (the session's power-set.txt must read a GPU Current Power Limit of 700 W, addendum 2's
rule, else nothing is read), PAGE per model (the registered page, the grid gate PASS, the wide filter, the CAL2
session's versions, python on major.minor), ROW SANITY per row (finite and positive, B within 0.5 to 2 x n b; refused
rows listed), then DS's own tile census: the census gate (missing or a path mismatch: a STOP for the per-CTA readings
LADDERS, SUBWAVE, DRAIN and W1RATE), CENSUS per decode tile (the occupancy S uses) and the cross-check against CAL2's
published census (reported; a difference is a STOP for the per-CTA readings). Then the
registered readings (d4common), every one DESCRIPTIVE: W1RATE (w1's counter marginal rate per tile), SUBWAVE (the
per-CTA rate of few-CTA long-slab GEMMs within one wave), DRAIN (the one-slab drain past a wave boundary), LADDERS
(every GEMM ladder as measured), CALLS (align, silu_and_mul and moe_sum per call, per model), ALIGN with BLOCK (the
align kernel per path and E; the block-scan constants per E) and SLOPE (the align time per loop iteration).
`NOT_IDENTIFIED` lists every reading that did not clear its rule. No model form, constant or gate is read or fixed
here: the decode model form for the new blind is chosen after DS, from DS's and CAL2's pages. Writes ds.readings.json
and ds.readings.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d4common as D4  # noqa: E402


def _f(v, fmt):
    return "none" if v is None else format(v, fmt)


def lines(r: dict) -> list[str]:
    L = ["DECODE SETTLING RENTAL DS (addendum 4): readings, DESCRIPTIVE", f"power: {r['POWER']['verdict']}",
         f"verdict: {r['verdict']}"]
    for m, x in (r.get("PAGE") or {}).items():
        py = (x.get("python_patch") or {}).get("page")
        L.append(f"PAGE {m}: {x['verdict']}" + (f"; refused cells {x['refused_cells']}" if x.get("refused_cells") else "")
                 + (f"; python {py} (CAL2 {D4.CAL2_VERSIONS['python']}; patch level DESCRIPTIVE)" if py else ""))
    if r["verdict"] != "READ":
        L.append(f"model form: {D4.MODEL_AFTER_DS}")
        return L
    L.append(f"CENSUS GATE (DS's own census): {r['CENSUS GATE']['verdict']}")
    for t, x in r["CENSUS"].items():
        L.append(f"CENSUS {t}: {x['verdict']}")
    x = r["CENSUS CROSS-CHECK"]
    L.append(f"CENSUS CROSS-CHECK (DS's census against CAL2's, reported): {x['verdict']}"
             + (f" over {x['entries_compared']} entries" if "entries_compared" in x else "")
             + ("".join(f"\n  {d}" for d in x["differences"])))
    for t, v in x["CAL2 census (occupancy rule, reported)"].items():
        L.append(f"  CAL2 census {t}: {v}")
    L.append(f"PER-CTA STOP ({', '.join(r['PER-CTA STOP']['readings'])}): {r['PER-CTA STOP']['verdict']}")
    rf = r["REFUSED ROWS"]["rows"]
    L.append(f"REFUSED ROWS ({r['REFUSED ROWS']['rule']}): {len(rf)}")
    L += [f"  {x['model']} {x['label']} {x.get('gemm') or x.get('class')}: {x['why']}" for x in rf]
    for t, x in r["LADDERS"]["tiles"].items():
        L.append(f"LADDERS {t}: {x['verdict']}" + (f"; {len(x['ladders'])} ladders, mixed {len(x['mixed'])}, subwave "
                                                  f"{len(x['subwave'])}" if x["verdict"] == "READ" else "")
                 + (f"; {x['w1_unpriced']}" if x.get("w1_unpriced") else ""))
    for t, x in r["W1RATE"]["tiles"].items():
        L.append(f"W1RATE {t}: {x['verdict']}" + (f"; c {x['counter_GBps']} GB/s (ratio to v4d {x['ratio_to_v4d']}) over "
                                                 f"{x['rows']} rows in {len(x['ladders'])} ladders"
                                                 if x.get("counter_GBps") else ""))
    for t, x in r["SUBWAVE"]["tiles"].items():
        L.append(f"SUBWAVE {t}: {x['verdict']}" + (f"; r_cta {_f(x['r_cta_GBps'], '')} GB/s over {x['rows']} rows in "
                                                  f"{len(x['fit_ladders'])} ladders ({len(x['qualifying_ladders'])} "
                                                  f"qualifying), {len(x['binding_rows'])} binding; rms {x['rms_rel']} "
                                                  f"(no ceiling {x['rms_rel_no_ceiling']})" if "rows" in x else ""))
    for t, x in r["DRAIN"]["tiles"].items():
        L.append(f"DRAIN {t}: {x['verdict']}" + (f"; r1 {x['r1_GBps'] or 'no drain'} GB/s over {x['rows']} rows in "
                                                f"{len(x['ladders'])} mixed ladders; rms {x['rms_rel']} (no drain "
                                                f"{x['rms_rel_no_drain']})" if "rows" in x else ""))
    for m, by in r["CALLS"]["per_model"].items():
        L.append(f"CALLS {m}: " + "; ".join(f"{c} median {x['median_us']} us ({x['min_us']} to {x['max_us']}, "
                                            f"{x['cells']} cells)" for c, x in by.items()))
    for c, x in r["CALLS"]["summary_lines"].items():
        L.append(f"CALLS line {c} (a summary of DS's pages, not a term): {x['verdict']}"
                 + (f"; {x['a_us']} us + {x['b_us_per_x']:.4g} us x + {x['g_us_per_MB']} us/MB over {x['rows']} rows, "
                    f"rms {x['rms_us']} us" if "a_us" in x else ""))
    a = r["ALIGN"]
    for p, pts in a["points"].items():
        L.append(f"ALIGN {p}: " + ", ".join(f"E {E} {v} us" for E, v in pts))
    L.append(f"ALIGN DECL transfer: {a['DECL_TRANSFER']['verdict']}")
    L.append(f"BLOCK: {a['BLOCK']['verdict']}")
    s = a["SLOPE"]
    L.append(f"SLOPE: {s['verdict']}" + (f"; {s['slope_us']} us per iteration over {s['rows']} rows, E {s['E']}, "
                                         f"iterations {s['iterations_read']}; against 0: "
                                         f"{s['check_against_0']['verdict']} (t {s['check_against_0']['t']})"
                                         if s.get("slope_us") is not None else ""))
    L.append("NOT IDENTIFIED: " + ("; ".join(r["NOT_IDENTIFIED"]) if r["NOT_IDENTIFIED"] else "none"))
    L.append(f"model form: {D4.MODEL_AFTER_DS}")
    return L


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if len(a) != 2:
        print(__doc__)
        return 2
    r = D4.read_ds(Path(a[0]), D4.ds_prm())
    out = Path(a[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "ds.readings.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (out / "ds.readings.txt").write_text("\n".join(lines(r)) + "\n")
    print("\n".join(lines(r)))
    return 0 if r["verdict"] == "READ" else 1


if __name__ == "__main__":
    sys.exit(main())
