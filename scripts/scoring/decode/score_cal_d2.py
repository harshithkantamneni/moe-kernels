#!/usr/bin/env python
"""Scores the decode stage's CAL rental under the second addendum (registration
2026-10-28-decode-addendum2-gh200, over 2026-10-14-decodeC-cal-gh200 and 2026-10-20-decode-tilesource-gh200).

    python scripts/scoring/decode/score_cal_d2.py <results tree> <out dir> --tile-scores scripts/scoring/tileA2

A wrapper: the frozen scorer (score_cal.score) runs twice and is not changed.
  power      first: the session's power-set.txt must read a GPU Current Power Limit of 700 W
             (d2common.power_check), else nothing is scored (NOT SCORED, a STOP)
  as registered  score_cal.score on every cell, sigma 1 on the second-tile cells: kept as
             REGISTERED_READING, DESCRIPTIVE (its STOP is printed, not read)
  amended    score_cal.score with the addendum's DESCRIPTIVE cells (d2common.excluded_keys on the two
             CAL main pages) left out of the main pages and the replicate: the fit, the rho selection,
             E1-CAL-IN-SAMPLE, the replicate's E1 row and the rivals' training rows. Taken from the run
             on every cell instead (no fit enters them): GD2 and GD3. The replicate's sigma_page and the
             tightened blind bars are the kept cells' (owner, m3); the every-cell figure is printed beside
             (REP_ALL_CELLS, DESCRIPTIVE). G3-D is re-read on every valid cell at the amended fit (each pair holds the
             same counts or multiset on both sides). The frozen stop rule (dcommon.stop_rule) then reads
             the merged score: the addendum's stop rule
  DESCRIPTIVE  SIGMA-D (the second-tile cells against v4d at sigma 1 and the L2-served estimate),
             the knee rows (p 5.75 and 3.75), BYTES-D2 (the counter pages' second-tile cells against both
             read totals)
Writes cal.score.json (the amended score, which finalise_decode_d2.py reads) and cal.score.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import dcommon as DC  # noqa: E402
import score_cal as SC  # noqa: E402

r6 = DC.r6
RD, V4, V4D = DC.RD, DC.V4, DC.V4D


def _main_cells(tree: Path, reg: dict) -> dict:
    """{(model, label): ms} of every valid NATIVE cell of the two CAL main pages (no cell left out)."""
    files = reg["pages"]["files"]
    out = {}
    for m in DC.CAL_MODELS:
        rep, _why = DC.decode_page(tree, m, "dm", files[f"{m}-main.json"], "GRR")
        if rep is None:
            continue
        v, _d = DC.measured(rep, m, "main")
        out.update({(m, k[2]): row["ms_p50"] for k, row in v.items() if k[4] == "native"})
    return out


def score(tree: Path, reg: dict, tparams: V4.Params, tsource: str = "", excluded: set | None = None) -> dict:
    tree = Path(tree)
    power = D2.power_check(tree)
    what = "the decode stage's CAL score under the second addendum"
    if power["verdict"] != "READ":
        return D2.not_scored(what, power)
    excl = D2.excluded_keys(DC.CAL_MODELS) if excluded is None else set(excluded)
    full = SC.score(tree, reg, tparams, tsource)
    with D2.without_cells(excl):
        amd = SC.score(tree, reg, tparams, tsource)
    out = dict(amd)
    out["what"] = what
    out["addendum"] = D2.NAME
    out["tile_params"] = f"the tile stage's selections from A2's score directory ({tsource.split('(', 1)[-1].rstrip(')')})" \
        if "(" in tsource else tsource
    out["POWER"] = power
    out["excluded_cells"] = sorted("/".join(k) for k in excl)
    for k in ("GD2", "GD3"):
        out[k] = full[k]
    out["scope"] = D2.SCOPE
    out["REP"] = dict(amd["REP"], label="the kept cells only (owner, 2026-10-28, review m3): these bars go to the blind rental")
    out["REP_ALL_CELLS"] = dict(full["REP"], label="DESCRIPTIVE: every replicate pair, the DESCRIPTIVE cells included")
    sel = amd["FIT"]["selected"]
    prm = V4D.params_from(amd["FIT"]["forms"][sel], rho_form=sel, tile=tparams) if sel is not None else None
    docs = {(m, "main"): DC.page_doc(m, "main") for m in DC.CAL_MODELS}
    meas = _main_cells(tree, reg)
    out["G3D"] = D2.all_cells_g3d(meas, prm, docs, DC.CAL_MODELS)
    cells = {(m, c["label"]): c for (m, _k), d in docs.items() for c in d["cells"]}
    sig = [D2.sigma_d_row(m, cells[(m, lab)], ms, prm) for (m, lab), ms in sorted(meas.items())
           if (m, lab) in excl and prm is not None]
    out["SIGMA_D"] = {"verdict": "DESCRIPTIVE", "rows": sig, "per_model": D2.sigma_d_summary(sig),
                      "constants": "the amended fit's (never refitted on these cells)"}
    kc = []
    for r in amd["rows"]:
        m, _kind, lab, _T, _arm = r["key"].split("/")
        kc.append((m, cells[(m, lab)], r["meas_ms"], r["pred_ms"]))
    out["KNEE_D"] = D2.knee_rows(kc, prm) if prm is not None else {"verdict": "NOT SCORED: no fit"}
    out["BYTES_D2"] = bytes_d2(tree, tparams)
    out["REGISTERED_READING"] = {"label": "DESCRIPTIVE: the frozen scorer on every cell (sigma 1 on the second-tile cells)",
                                 "STOP": full["STOP"], "E1_CAL_IN_SAMPLE": full["E1_CAL_IN_SAMPLE"],
                                 "selected": full["FIT"]["selected"],
                                 "forms": {f: {k: x.get(k) for k in ("verdict", "T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms")}
                                           for f, x in full["FIT"]["forms"].items()},
                                 "G3D": {t: g.get("verdict") for t, g in full["G3D"].items()}}
    out["STOP"] = DC.stop_rule(out)
    return out


def bytes_d2(tree: Path, tparams) -> dict:
    """The counter pages' (ctl) second-tile cells: counted dram__bytes_read per GEMM against v4d's read
    bytes (sigma 1) and against the L2-served read bytes; share = (v4d - counted) / (v4d - served), the
    fraction of the re-read bytes the counters did not see (DESCRIPTIVE). Both read totals are computed
    here with the tile stage's parameters (read bytes do not depend on the fitted constants): the VM-side
    v4d_reads of the wgmma T 128 cells is an error dict (bare DecodeParams has no F_wg; review m1), so the
    page's value is kept beside, never used."""
    rows = []
    bare = V4D.DecodeParams(tile=tparams)
    for m in DC.CAL_MODELS:
        keys = D2.excluded_keys([m], "ctl")
        page, _why = SC.counter_page(tree, m, "dn")
        cells = {c["label"]: c for c in DC.page_doc(m, "ctl")["cells"]}
        for c in (page or {}).get("cells") or []:
            if c.get("arm") != RD.NATIVE or (m, c.get("label")) not in keys:
                continue
            pc = cells[c["label"]]
            try:
                want_all = V4D.price(m, int(pc["T"]), pc["counts"], bare, with_constants=False)["reads"]
            except V4.TM.Refused as exc:
                rows.append({"cell": f"{m}/{c['label']}", "error": f"NOT PRICED: {exc}"})
                continue
            served = D2.served_price(m, pc, bare)
            vm = c.get("v4d_reads") or {}
            for g in RD.GEMMS:
                got = (c["per_gemm"].get(g) or {}).get("dram__bytes_read.sum")
                want, srv = want_all[g], None if served is None else served["reads"][g]
                row = {"cell": f"{m}/{c['label']}/{g}", "counted": got, "v4d_reads": r6(want),
                       "served_reads": None if srv is None else r6(srv),
                       "vm_v4d_reads": vm.get(g) if "error" not in vm else f"error: {vm['error']}"}
                if got and srv:
                    row.update(counted_over_v4d=r6(got / want), counted_over_served=r6(got / srv),
                               share_not_seen=r6((want - got) / (want - srv)) if want != srv else None)
                else:
                    row["error"] = "no counted dram__bytes_read" if not got else "no served price"
                rows.append(row)
    return {"verdict": "DESCRIPTIVE", "rows": rows}


def lines(s: dict) -> list[str]:
    if s.get("verdict") == "NOT SCORED":
        return ["DECODE STAGE, CAL RENTAL (addendum 2): NOT SCORED", f"power: {s['POWER']['verdict']}",
                f"STOP: {s['STOP']['verdict']}" + "".join(f"\n  {w}" for w in s["STOP"]["why"])]
    L = SC.lines(s)
    L[0] = "DECODE STAGE, CAL RENTAL (addendum 2): score"
    L.insert(2, f"power: {s['POWER']['verdict']} ({s['POWER'].get('gpu_current_power_limit_w')} W)")
    L.insert(3, f"descriptive cells (out of the fit, the rho selection and every E1 bar): {len(s['excluded_cells'])}")
    L.insert(4, s["scope"])
    L.append(f"REP over every cell (DESCRIPTIVE): sigma_page {s['REP_ALL_CELLS']['sigma_page']} over "
             f"{s['REP_ALL_CELLS']['pairs']} pairs; bars {s['REP_ALL_CELLS']['bars_blind']}")
    for m, x in s["SIGMA_D"]["per_model"].items():
        L.append(f"SIGMA-D {m}: {x['band']} (median position {x['position_median']}, {x['cells_read']} of {x['cells']} "
                 f"cells read); v4d sigma 1 {x['sigma1']}; L2-served {x['served']}")
    for k, x in (s.get("KNEE_D") or {}).items():
        if isinstance(x, dict) and "stats" in x:
            L.append(f"knee {k} (DESCRIPTIVE): {x['stats']}")
    rr = s["REGISTERED_READING"]
    L.append(f"as registered (DESCRIPTIVE): E1 {rr['E1_CAL_IN_SAMPLE']['verdict']} {rr['E1_CAL_IN_SAMPLE']['stats']}; "
             f"STOP {rr['STOP']['verdict']}")
    return L


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    ts = None
    if "--tile-scores" in a:
        i = a.index("--tile-scores")
        ts = Path(a[i + 1])
        del a[i:i + 2]
    if len(a) != 2:
        print(__doc__)
        return 2
    if ts is None:
        print("REFUSED: --tile-scores <A2 score dir> is required (decodeC-cal model.tile_stage; the first addendum binds "
              "scripts/scoring/tileA2)")
        return 2
    tree, out = Path(a[0]), Path(a[1])
    reg = DC.registration(DC.REPO, "cal")
    tp, src = SC.tile_params(ts)
    s = score(tree, reg, tp, src)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cal.score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (out / "cal.score.txt").write_text("\n".join(lines(s)) + "\n")
    print("\n".join(lines(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
