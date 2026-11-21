"""Jobs for tests/test_cal2_reader.py that load the frozen chain (knee-v5 k5, sagtau and the modules they import).
Each runs in its own interpreter, so no module and no sys.path entry the chain leaves behind reaches the test
process: `python tests/cal2_chain_job.py <job> [args]` prints one JSON document.

  a2        read_c2's functions on A2's and S1's published pages (stand-ins): the knee round's numbers, the C2-S6
            split, the timed-page lock refusal, and the chain-repo check against the registration
  standin   read_c2.read on a stand-in tree (argv[2]) without 8x22B pages, the commit check skipped (no rental commit
            exists before the rental)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "scoring" / "cal2"))
import read_c2 as R  # noqa: E402

PUB = REPO / "results" / "published"
A2_RES = PUB / "2026-10-21-nvidia_gh200_480gb-tileA2-session" / "results"
S1_RES = PUB / "2026-10-21-nvidia_gh200_480gb-settleS1-session" / "results"


def a2() -> dict:
    k5, ST = R.chain()
    out = {}
    e1 = json.loads((REPO / "docs/registered/2026-10-20-tileA2-e1cal-gh200.json").read_text())["pages"]["files"]
    doc = json.loads((REPO / R.MIX_HIST).read_text())
    rows, why = R.timed_rows(A2_RES, "mixtral-8x7b", "e128g1", e1["mixtral-8x7b-128g1.json"], doc)
    out["rows"], out["why"], out["drops"] = len(rows), why, [r["drop"] for r in rows]
    olm = ST.fit(k5.S.pts_of(R.page_files(A2_RES, "olmoe-1b-7b", "f128"), 1410.0))
    out["olmoe_fit"] = list(olm)
    out["mixtral_fold"] = R.held_out(rows, olm, p128=R.MIX_FOLD_P)["X_pct"]
    sf = R.sag_fits(A2_RES)
    out["per_page"] = {m: list(v) for m, v in sf["per_page"].items()}
    out["par"] = list(sf["par"])
    out["seen_par"] = R.held_out(rows, sf["par"])["X_pct"]
    out["x4"] = R.held_out([r for r in rows if r["label"] in R.A2_LABELS], sf["par"])["X_pct"]
    out["x"] = R.x_of_rows(rows)
    c = R.counter_reading(A2_RES, "olmoe-1b-7b", "b128g1")
    out["olmoe_cycles"] = [c["w1"]["mean_pct"], c["w2"]["mean_pct"]]
    s1 = R.counter_reading(S1_RES, "mixtral-8x7b", "b128g1")
    out["s1_cycles"] = [s1["w1"]["mean_pct"], s1["w2"]["mean_pct"]]
    out["olmoe_sag"] = R.sag_domain(A2_RES, "olmoe-1b-7b", "b128g1", sf["par"])["mean_err_us"]
    # C2-S6's split, two boards here (A2's timed page, S1's byte page of the same cells): the format and the sum
    out["split"] = R.split(rows, olm, S1_RES, "mixtral-8x7b", "b128g1", R.MIX_FOLD_P)
    # the timed page's lock: a report recording another lock is NOT SCORED
    TC = k5.d5.TC
    real = TC.timed_page
    TC.timed_page = lambda *a, **k: (lambda rep, w: ({**rep, "lock_mhz": 1710} if rep else rep, w))(*real(*a, **k))
    try:
        out["lock_refusal"] = R.timed_rows(A2_RES, "mixtral-8x7b", "e128g1", e1["mixtral-8x7b-128g1.json"], doc)[1]
    finally:
        TC.timed_page = real
    out["chain_repo_bad"] = R.check_chain_repo(R.reg())
    return out


def standin(c2: str) -> dict:
    return json.loads(R.dumps(R.read(Path(c2), s1_results=S1_RES, _skip_commit_check_for_tests=True)))


if __name__ == "__main__":
    job = sys.argv[1]
    d = a2() if job == "a2" else standin(sys.argv[2])
    print("JSON " + json.dumps(d, sort_keys=True, default=str))
