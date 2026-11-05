"""The job for tests/test_tileB3_reader.py that loads the frozen round-5 chain (round5/work r7.py over round 4's r6 and
what they import). It runs in its own interpreter, so no module and no sys.path entry the chain leaves behind reaches
the test process, and once (the chain takes about a minute to build): `python tests/tileB3_chain_job.py <stand-in dir>`
prints one JSON document with every reading the tests check.

  trace     every file under moe-kernels-work the chain opens (an audit hook from before its import) while it prices
            the 84 blind cells, re-prices round 5's table and fits the three boards' sag pages: the registration's
            FROZEN_FILES must hold every one; the repo-module check against the registration
  standin   a stand-in B3 tree made from C4's published session (its two f128 pages, power, lock and G-BAND files; the
            b3sag record written by the gate's rule on those pages), A2's two f64w8 pages (the b3sag64 record) and a
            b3census record from the gate's rule on A2's census: `finalise` must reproduce the registration's
            predictions at C4's par (A2's 64x128 par) cell for cell; `score` then runs with no blind page (every blind page NOT SCORED:
            E1-TILE incomplete)
  valid     `score` down its VALID path on the stand-in: tcommon.timed_page and valid_cells replaced by synthetic
            measurements (T_meas = T_pred / (1 + e)) for two scenarios, every blind model at e +0.5% (HOLDS, three
            models CERTIFIED) and Phi-3.5 at +2.0% (FAILS on Phi's |mean|)
  g3        `score` down the VALID path with G3 planted heterogeneous (stratum 0 voided: its non-balanced cells leave the
            verdict view, the bar without the voids printed beside) and homogeneous (every histogram cell voided:
            E1-TILE NOT SCORED)
  edited    `score` with the predictions file edited by one cell: REFUSED
  records   `score` with the census record missing, then FAIL: NOT SCORED; with the sag64 record missing, then FAIL:
            REFUSED by finalise (the predictions need it)
  guard     `finalise` with the sag fit made to open a timed page (a gaps-* path), then a blind byte page (a path naming
            a blind model): REFUSED both; a census path, which names a blind model, is allowed (the exemption)
  clock     `finalise` with the gate records PASS and the f64w8 clock clause failing on the pages: REFUSED
  refuse    `finalise` with the fingerprint moved: REFUSED before any price is written
  register_check  register_b3.build() in the same interpreter against the committed JSON (code_pins aside) and text
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
OPENED: set = set()


def _hook(ev, args):
    if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            OPENED.add(os.fsdecode(args[0]))
        except Exception:
            pass


sys.addaudithook(_hook)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "scoring" / "tileB3"))
import b3common as B  # noqa: E402
import read_b3 as R  # noqa: E402

C4 = REPO / "results" / "published" / "2026-10-27-nvidia_gh200_480gb-cal4-session"
A2 = REPO / "results" / "published" / "2026-10-21-nvidia_gh200_480gb-tileA2-session"


def make_standin(root: Path) -> Path:
    b3 = root / "b3"
    res, drv = b3 / "results", b3 / "session" / "gh200-driver"
    res.mkdir(parents=True)
    drv.mkdir(parents=True)
    for d in sorted((C4 / "results").glob("*-f128-r3-counters")):
        os.symlink(d, res / d.name)
    for d in sorted((A2 / "results").glob("*-f64w8-r3-counters")):
        os.symlink(d, res / d.name)
    for f in ("c2-power.env", "locks.env", "gate-band-1410.json"):
        shutil.copy(C4 / "session" / "gh200-driver" / f, drv / f)
    g = B.sag_gate(res)
    (drv / R.GATE_RECORDS["sag"]).write_text(B.dumps(g))
    (drv / R.GATE_RECORDS["sag64"]).write_text(B.dumps(B.sag_gate(res, B.SAG64_UNIT)))
    TCA = B._census_mods()
    cen = B.census_check(json.loads((REPO / B.A2_MERGED_CENSUS).read_text()),
                         json.loads(B.find_census(A2 / "results", "ta2").read_text()), TCA.facts, TCA.FIELDS)
    (drv / R.GATE_RECORDS["census"]).write_text(B.dumps({"gate": "census", **cen}))
    return b3


def valid(b3: Path, pred: Path, bias: dict, g3: dict | None = None) -> dict:
    TC = R.chain()["d5"].TC
    keep = (TC.timed_page, TC.valid_cells, TC.C5.g3_rule, R.timed_view)
    views = []

    def tv_rec(*a, **k):
        v = keep[3](*a, **k)
        views.append(v)
        return v

    def timed_page(tree, model, label, G, tv, f, view="registered"):
        if model not in B.MODELS:      # a CAL page (the preflight's round-5 table): the real reader
            return keep[0](tree, model, label, G, tv, f, view)
        return {"fake": True}, "counted (synthetic)"

    def valid_cells(rep, sub, model, page):
        if model not in B.MODELS:
            return keep[1](rep, sub, model, page)
        e = bias.get(model, 0.005)
        return {k: v["ms"] / (1 + e) for k, v in sub.items()}, [], None
    TC.timed_page, TC.valid_cells, R.timed_view = timed_page, valid_cells, tv_rec
    if g3 is not None:
        TC.C5.g3_rule = lambda rows: dict(g3, rows=rows)
    try:
        d = R.score(b3, skip_commit=True, committed=pred)
    finally:
        TC.timed_page, TC.valid_cells, TC.C5.g3_rule, R.timed_view = keep
    e = d["E1-TILE"]
    return {"verdict": e["verdict"], "pooled": (e.get("pooled") or {}).get("stats"),
            "views": [{"voided": len(v["voided_cells"]), "E1": v["E1-TILE"]["verdict"],
                       "kept": (v["E1-TILE"].get("pooled") or {}).get("stats", {}).get("cells"),
                       "without_voids": (v["DESCRIPTIVE"]["the bar without G3's per-stratum voids"]
                                         if isinstance(v["DESCRIPTIVE"]["the bar without G3's per-stratum voids"], str)
                                         else v["DESCRIPTIVE"]["the bar without G3's per-stratum voids"]["reading"]),
                       "voided_labels": sorted({c.split("/")[3] for c in v["voided_cells"]}),
                       "voided_n": sorted({c.split("/")[2] for c in v["voided_cells"]})} for v in views],
            "per_model": {m: {"verdict": v["verdict"], "mean": v["stats"].get("mean")} for m, v in (e.get("per_model") or {}).items()},
            "certification": d["certification"], "validity": d["validity"],
            "b2_bar_reading": d["views"]["DESCRIPTIVE"]["B2's bar (2%, 5%, 1%) on the same cells"]["reading"]}


def main(tmp: str) -> dict:
    tmp = Path(tmp)
    out = {}
    reg = R.reg()
    b3 = make_standin(tmp)
    fin = R.finalise(b3, skip_commit=True)
    pred = tmp / "b3.predictions.json"
    pred.write_text(R.dumps(fin))
    want = reg["frozen"]["predictions before the rental (frozen model at three boards' 128x128 pars and A2's 64x128 par; "
                         "the rental's own pars finalise)"]["C4"]
    out["standin"] = {"par": fin["par"], "par64": fin["par64"], "n": fin["n_cells"], "unpriced": fin["unpriced"],
                      "equals_registered_C4": [c["ms"] for c in fin["cells"]] == want["ms"] and fin["par"] == want["par"]
                      and fin["par64"] == want["par64"]}
    d = R.score(b3, skip_commit=True, committed=pred)
    out["standin"]["E1-TILE"] = d["E1-TILE"]["verdict"]
    out["standin"]["validity"] = d["validity"]
    out["standin"]["drift"] = d["sag drift (DESCRIPTIVE)"]
    work, repo = str(R.WORK.resolve()), str(REPO.resolve())
    opened = sorted({str(Path(p).resolve().relative_to(work)) for p in OPENED
                     if Path(p).is_absolute() and Path(p).is_file() and "__pycache__" not in p
                     and str(Path(p).resolve()).startswith(work + "/") and not str(Path(p).resolve()).startswith(repo + "/")})
    out["trace"] = {"opened_work_files": opened, "not_pinned": [p for p in opened if p not in R.FROZEN_FILES],
                    "repo_modules_bad": R.repo_modules_bad(reg)}
    out["valid_small"] = valid(b3, pred, {})
    out["valid_phi"] = valid(b3, pred, {"phi-3.5-moe": 0.02})
    out["g3_stratum"] = valid(b3, pred, {}, {"verdict": "FAILS (heterogeneous, planted)", "void_strata": ["0"]})
    out["g3_all"] = valid(b3, pred, {}, {"verdict": "FAILS (homogeneous, planted)", "void_strata": "ALL"})
    # the f64w8 clock clause rechecked by finalise: the records PASS, the clause fails here
    keep_c = dict(B.CLOCK64)
    B.CLOCK64["cells_in_band_min"] = 27
    try:
        R.finalise(b3, skip_commit=True)
        out["refuse_clock"] = None
    except SystemExit as e:
        out["refuse_clock"] = str(e)
    finally:
        B.CLOCK64.clear()
        B.CLOCK64.update(keep_c)
    # an edited predictions file
    ed = json.loads(pred.read_text())
    ed["cells"][0]["ms"] = round(ed["cells"][0]["ms"] + 1e-6, 6)
    edp = tmp / "edited.predictions.json"
    edp.write_text(R.dumps(ed))
    try:
        R.score(b3, skip_commit=True, committed=edp)
        out["edited"] = None
    except SystemExit as e:
        out["edited"] = str(e)
    # the gate records
    drv = b3 / "session" / "gh200-driver"
    out["records"] = {}
    for k in ("census", "sag64"):
        f = drv / R.GATE_RECORDS[k]
        keep_rec = f.read_text()
        for how in ("missing", "FAIL"):
            if how == "missing":
                f.unlink()
            else:
                f.write_text(B.dumps({**json.loads(keep_rec), "verdict": "FAIL", "why": "planted"}))
            try:
                d = R.score(b3, skip_commit=True, committed=pred)
                out["records"][f"{k} {how}"] = {"E1-TILE": d["E1-TILE"]["verdict"], "validity": d["validity"],
                                                "certification": d["certification"]}
            except SystemExit as e:
                out["records"][f"{k} {how}"] = {"refused": str(e)}
            f.write_text(keep_rec)
    # the guard: a timed page opened during finalise
    keep = B.page_fit
    gp = b3 / "results" / "gaps-nvidia_gh200_480gb-phi-3.5-moe-t128g1"
    gp.mkdir()
    (gp / "report.json").write_text("{}")

    def leaky(results, model, *a):
        (gp / "report.json").read_text()
        return keep(results, model, *a)
    B.page_fit = leaky
    try:
        R.finalise(b3, skip_commit=True)
        out["guard"] = None
    except SystemExit as e:
        out["guard"] = str(e)
    finally:
        B.page_fit = keep
    bp = b3 / "results" / f"2026-11-05-{B.CARD}-phi-3.5-moe-b128g1-r3-counters" / "lock1410"
    bp.mkdir(parents=True)
    (bp / "r3c-g1.json").write_text("{}")
    cp = b3 / "results" / f"2026-11-05-{B.CARD}-tile-census-phi-3.5-moe-tb"   # names a blind model: the exemption
    cp.mkdir()
    (cp / "tile-census.json").write_text("{}")
    for tag, path in (("guard_bytes", bp / "r3c-g1.json"), ("guard_census", cp / "tile-census.json")):
        def leaky2(results, model, *a, _p=path):
            _p.read_text()
            return keep(results, model, *a)
        B.page_fit = leaky2
        try:
            R.finalise(b3, skip_commit=True)
            out[tag] = None
        except SystemExit as e:
            out[tag] = str(e)
        finally:
            B.page_fit = keep
    import register_b3 as RG
    doc = json.loads(json.dumps(RG.build(), default=str))
    jpath = REPO / "docs" / "registered" / f"{RG.NAME}.json"
    out["register_check"] = {"json": RG._strip(json.loads(jpath.read_text())) == RG._strip(doc),
                             "txt": (jpath.with_suffix(".txt")).read_text() == "\n".join(RG.txt(doc)) + "\n"}
    keep_fp = R.fingerprint
    R.fingerprint = lambda *a, **k: {"ms": [0.0], "reads": []}
    try:
        R.finalise(b3, skip_commit=True)
        out["refuse"] = None
    except SystemExit as e:
        out["refuse"] = str(e)
    finally:
        R.fingerprint = keep_fp
    return out


if __name__ == "__main__":
    print("JSON " + json.dumps(main(sys.argv[1]), sort_keys=True, default=str))
