"""Jobs for tests/test_cal4_reader.py that load the frozen round-4 chain (round3/work r6 and the modules it imports).
Each runs in its own interpreter, so no module and no sys.path entry the chain leaves behind reaches the test
process: `python tests/cal4_chain_job.py <job> [args]` prints one JSON document.

  c2        read_c4's functions on C2's published pages (stand-ins for tp2's): the board's sag fits, Mixtral-8x22B's
            held-out mean under the frozen model at its fold (K1's SEEN constants are 8x22B's LOMO fold, F1 par of
            both C2 pages: round 4's +0.24), the bytes and knee readings on Mixtral-8x7B's b128g1 page, the sag domain
            and census on 8x22B's, and the repo-module check against the registration
  standin   read_c4.read on a stand-in tree (argv[2]) without tp2 pages, the commit check skipped (no rental commit
            exists before the rental)
  valid     read_c4.read down its VALID path on the stand-in tree: tp2's page functions replaced by C2's published
            pages (8x22B's timed rows and its census entries as tp2's, Mixtral-8x7B's byte page as tp2's), so the
            validity assembly, the five-model table, the pass estimate, the bytes band and the joint cell all run
  refuse    read_c4.read with round4_table moved: refused at the preflight, before any page function runs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "scoring" / "cal4"))
import read_c4 as R  # noqa: E402

PUB = REPO / "results" / "published"
C2 = PUB / "2026-10-21-nvidia_gh200_480gb-cal2-session"
C2_RES = C2 / "results"
C2_REG = json.loads((REPO / "docs/registered/2026-10-22-cal2-gh200.json").read_text())


def c2() -> dict:
    out = {}
    sf = R.sag_fits(C2_RES)
    out["per_page"] = {m: list(v) for m, v in sf["per_page"].items()}
    out["par"] = list(sf["par"])
    doc = json.loads((REPO / "docs/registered/2026-10-22-cal2-hist/mixtral-8x22b-128g1.json").read_text())
    rows, why = R.timed_rows(C2_RES, C2_REG["pages"]["files"]["mixtral-8x22b-128g1.json"], doc, model=R.X22, label="t128g1")
    out["x22_rows"], out["x22_why"], out["x22_drops"] = len(rows), why, [r["drop"] for r in rows]
    h = R.held_out(rows, sf["par"])
    out["x22_X"], out["x22_unpriced"] = h["X_pct"], h["unpriced"]
    out["x22_X_m4"] = R.held_out(rows, sf["par"], rd_of=lambda r: R.reads(r, None))["X_pct"]
    cc = R.counter_cells(C2_RES, "mixtral-8x7b", "b128g1")
    br = R.bytes_reading(cc)
    out["m7_bytes"] = {g: {k: v for k, v in br[g].items() if k != "per_cell"} for g in br}
    out["m7_m4_cells_mean"] = {g: round(sum(c["m4_over_counted_pct"] for c in br[g]["per_cell"]) / len(br[g]["per_cell"]), 2) for g in br}
    w1, w2 = br["w1"], br["w2"]
    out["m7_w1_outcome"] = R.bytes_outcome(w1["lambda"], w1["lever"], w1["n"])["band"]
    out["m7_w2_outcome"] = R.bytes_outcome(w2["lambda"], w2["lever"], w2["n"])["band"]
    out["m7_knee"] = R.knee_reading(C2_RES, "mixtral-8x7b", "b128g1")
    out["x22_sag"] = R.sag_domain(C2_RES, tuple(sf["par"]), R.X22, "b128g1")
    out["x22_census"] = list(R.census_file(C2_RES, R.X22, "tc2"))
    out["chain_repo_bad"] = R.check_chain_repo(R.reg())
    return out


def standin(c4: str) -> dict:
    return json.loads(R.dumps(R.read(Path(c4), _skip_commit_check_for_tests=True)))


def valid(c4: str) -> dict:
    doc = json.loads((REPO / "docs/registered/2026-10-22-cal2-hist/mixtral-8x22b-128g1.json").read_text())
    rf = C2_REG["pages"]["files"]["mixtral-8x22b-128g1.json"]
    real_cc, real_k, real_s = R.counter_cells, R.knee_reading, R.sag_domain
    R.timed_rows = lambda results, reg_file, d: R.__dict__["_timed"](C2_RES, rf, doc, model=R.X22, label="t128g1")
    R.census_file = lambda results: ({"w1": 2, "w2": 2}, "COMPLETE", "")
    R.RC.check_counter_page = lambda *a, **k: (Path("stand-in"), "")
    R.counter_cells = lambda results: real_cc(C2_RES, "mixtral-8x7b", "b128g1")
    R.knee_reading = lambda results: real_k(C2_RES, "mixtral-8x7b", "b128g1")
    R.sag_domain = lambda results, est: real_s(C2_RES, est, R.X22, "b128g1")
    return json.loads(R.dumps(R.read(Path(c4), _skip_commit_check_for_tests=True)))


def refuse(c4: str) -> dict:
    touched = []
    for name in ("census", "sag_fits", "timed_rows", "counter_cells", "triad"):
        setattr(R, name, lambda *a, _n=name, **k: touched.append(_n) or (_ for _ in ()).throw(RuntimeError(_n)))
    R.round4_table = lambda: {**R.CAL_HELD_OUT, R.X22: 0.25, "mixtral-8x7b C2": R.M_C2_BOARD}
    try:
        R.read(Path(c4), _skip_commit_check_for_tests=True)
    except SystemExit as e:
        return {"refused": str(e), "touched": touched}
    return {"refused": None, "touched": touched}


if __name__ == "__main__":
    job = sys.argv[1]
    R.__dict__["_timed"] = R.timed_rows
    d = {"c2": lambda: c2(), "standin": lambda: standin(sys.argv[2]), "valid": lambda: valid(sys.argv[2]),
         "refuse": lambda: refuse(sys.argv[2])}[job]()
    print("JSON " + json.dumps(d, sort_keys=True, default=str))
