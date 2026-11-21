"""Rental 7's four registrations (docs/registered/2026-10-13-rental7-{g1rule,skew,q1,l2open}-gh200),
built for scripts/scoring/rental7/register.py, which renders, pins and --checks them.

Every number is computed here from committed files (rental 6's pricer `reg6.priced` with model v3
unchanged, the published pages read through R3's own functions, the synthetic histogram pages of
scripts/skew_synth7.py, the row-key rival's file of rowkeys7.py) or is a typed-in input with its
provenance and its label (CAL, SEEN, SEEN-FITTED, SEEN-SELECTED, SEEN-INFORMED, SEEN-DERIVED,
IN-SAMPLE, BLIND, BLIND-CALMODEL). SEEN-DERIVED is new in rental 7: a gate rule whose threshold was
read off published pages; it is not a model constant.

Design: scratchpad design-r7/DESIGN.md corrected by design-r7-review/REVIEW.md (the review wins
where they conflict), with the owner's decisions of 2026-10-13: 1 the drift rule is S_med plus the
review's run guard (r7-median-run3), the P20 view printed beside it; 2 Q1-H's rivals are scored
whatever the copies clause reads; 3 the OLMoE w1 byte page (l2open) is in, one C of 60 MiB, the 5%
clause on skewed rows only; 4 no second-K rerun; 5 fresh seeds on the same treads. Model v3 is
rental 6's (2026-10-09-rental6-v3-gh200), unchanged; no term is added.

LEAKAGE, said once for the four: every published page is SEEN, rental 6's included; every constant
typed in below was fitted or chosen after seeing published pages (labelled per entry). No rental-7
page exists when this runs.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7common as C7  # noqa: E402

C6, C5 = C7.C6, C7.C5
REPO = C7.REPO
sys.path.insert(0, str(REPO / "scripts" / "scoring" / "rental6"))
import r6model as M6  # noqa: E402
import reg6 as R6  # noqa: E402

HIST = REPO / C7.HIST_DIR
H6 = REPO / C6.HIST_DIR
PUB = REPO / "results" / "published"
r6 = C6.r6
DESIGN = ("design-r7 DESIGN.md corrected by design-r7-review REVIEW.md (a)-(h) (the review wins) and the owner's "
          "decisions of 2026-10-13 (1 S_med plus the run guard, P20 printed beside; 2 Q1-H rivals scored apart from "
          "the copies clause; 3 the l2open byte page in, C 60 MiB, the 5% clause on skewed rows; 4 no second-K rerun; "
          "5 fresh seeds on the same treads)")
REGISTERED = f"{C7.DATE}, before any rental-7 page"
MODEL_V3 = "docs/registered/2026-10-09-rental6-v3-gh200.json (model v3, unchanged; no term added)"

SHAPES = {"mixtral-8x7b": (3, 6, 16), "phi-3.5-moe": (2, 4, 12)}
TIMED_PAGES = ("A", "B", "C")
SKEWED = ("PT", "PW", "PW2", "DW")
RIVALS_E1 = ("v2", "LT3", "H8", "v3_all", "v3_noca")
RIVALS_GAP = ("v2", "H8", "v3_all")
Q1_MODELS = ("qwen1.5-moe-a2.7b", "phi-3.5-moe", "jetmoe-8b")
COPIES = (9, 15)
L2_MODEL = "olmoe-1b-7b"

#: Q1-H noise and anchors per (model, n) cluster, us (SEEN): the design's (3.2: Phi n 2 +0.64 us over
#: three rental-6 pairs; Qwen1.5 n 2 +1.02, rental 6 +1.14 and 09-29 +0.90; JetMoE n 4 +0.77 from
#: design-r6 hrule.txt; replicate sigma Phi 0.20, Qwen1.5 and JetMoE 0.45 assumed) and the review's
#: (work/seen_g_clusters.json: Mixtral n 6 and Phi n 4 over the six rental-6 control pairs each).
#: These are the inputs of design-r7-review work/mc_q1h.py, kept so its numbers reproduce.
Q1H_ANCHOR_US = {"phi-3.5-moe|n2": 0.638, "qwen1.5-moe-a2.7b|n2": 1.02, "jetmoe-8b|n4": 0.77,
                 "mixtral-8x7b|n6": 0.041, "phi-3.5-moe|n4": 0.608}
Q1H_SIG_US = {"phi-3.5-moe|n2": 0.20, "qwen1.5-moe-a2.7b|n2": 0.45, "jetmoe-8b|n4": 0.45,
              "mixtral-8x7b|n6": 0.769, "phi-3.5-moe|n4": 0.563}
Q1H_DESIGN = {("phi-3.5-moe", 2), ("qwen1.5-moe-a2.7b", 2), ("jetmoe-8b", 4)}
Q1H_ADDED = {("mixtral-8x7b", 6), ("phi-3.5-moe", 4)}
#: SEEN x9 of Q1-C (rental 6's Qwen1.5 SHARED n 1 c9 pages; design-r7 3.2)
Q1C_X9_SEEN = (-5.97, -7.95)
Q1C_BOARD_US = 1.0
#: the board shift of E1's residual model (noise floors: 8x7B board to board), and the scenarios
E1_SCENARIOS = {"SEEN-anchored": (1.0, 0.0018), "zero bias": (0.0, 0.0018), "harsh (bias x1.5, board 0.3%)": (1.5, 0.003)}
POWER_R = 200_000
#: counter noise on a byte row (rental 6 noise floors; design-r7 3.4): within a board, board to board
COUNTER_NOISE = {"within_board_pct": 0.3, "board_to_board_pct": [0.03, 0.66]}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mps_check() -> dict:
    """The laptop-GPU re-run of every registered power point (power7.py): its file, sha256, the
    largest |CPU float64 - MPS float32| over every probability and the wall seconds."""
    f = HIST / "power7_check.json"
    if not f.exists():
        return {"file": str(f.relative_to(REPO)), "status": "NOT RUN"}
    d = json.loads(f.read_text())
    return {"file": str(f.relative_to(REPO)), "sha256": sha256_file(f), "torch": d["torch"], "R": d["R"],
            "max_abs_diff_cpu_float64_vs_mps_float32": d["max_abs_diff"], "seconds": d["seconds"]}


def load(name: str) -> dict:
    return json.loads((HIST / f"{name}.json").read_text())


def _key(r) -> tuple:
    return (r["model"], r["page"], r["n"], r["label"], r["arm"])


# --------------------------------------------------------------------------
# the pricer, as of the registration (rental 6's rule, rental 7's date)
# --------------------------------------------------------------------------

def seed_pricer() -> None:
    """reg6's pricer reads the SEEN align proxy over the published reports. --check must
    recompute these files after rental 7's own session is published, so the proxy reads the
    published set as of this registration: sessions dated on or before C7.DATE whose name does
    not carry 'rental7' (a directory of links). reg6.pricer() keeps the first Pricer it is given."""
    import atexit
    import shutil
    import tempfile
    if R6._P:
        return
    tmp = Path(tempfile.mkdtemp(prefix="r7-asof-"))
    atexit.register(shutil.rmtree, tmp, True)
    for d in sorted(PUB.iterdir()):
        if d.is_dir() and d.name[:10] <= C7.DATE and "rental7" not in d.name:
            (tmp / d.name).symlink_to(d.resolve(), target_is_directory=True)
    R6._P.append(M6.Pricer(proxy=M6.align_proxy(tmp)))


# --------------------------------------------------------------------------
# what is published (leakage)
# --------------------------------------------------------------------------

_PUBSHA: list = []


def published_counts_sha() -> set:
    """Every (model, n, counts sha256) a published page or an earlier registered histogram file
    carries. Keyed by model and tread because a balanced cell's counts are null, whose sha256 is
    the same at every (model, n)."""
    if _PUBSHA:
        return _PUBSHA[0]
    out = set()
    for f in glob.glob(str(PUB / "**" / "report.json"), recursive=True):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:                                    # noqa: BLE001
            continue
        for row in d.get("treads_table") or []:
            if isinstance(row, dict) and row.get("counts_sha256") and d.get("model"):
                out.add((d["model"], int(row["tiles"]), row["counts_sha256"]))
    for f in glob.glob(str(PUB / "**" / "r3c-g*.json"), recursive=True):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:                                    # noqa: BLE001
            continue
        m = (d.get("design") or {}).get("model")
        for c in d.get("cells") or []:
            if isinstance(c, dict) and c.get("counts_sha256") and m:
                out.add((m, int(c["n"]), c["counts_sha256"]))
    for f in glob.glob(str(REPO / "docs" / "registered" / "2026-10-0*-skew-hist" / "*.json")):
        d = json.loads(Path(f).read_text())
        for c in d.get("cells") or []:
            if "counts" in c and d.get("model"):
                out.add((d["model"], int(c["n"]), R6.sha_counts(c["counts"])))
    _PUBSHA.append(out)
    return out


def is_published(model: str, n: int, sha: str) -> bool:
    return (model, int(n), sha) in published_counts_sha()


# --------------------------------------------------------------------------
# rental7-g1rule: the drift rule (no unit)
# --------------------------------------------------------------------------

def _held_pages():
    """Every published histogram page with a report (17, all held at the 1710 lock): (name, report,
    cells.csv rows as dicts)."""
    out = []
    for rp in sorted(PUB.glob("**/private_weight_reference/*/report.json")):
        rep = json.loads(rp.read_text())
        if not rep.get("histogram_gates") or "rental7" in str(rp):
            continue
        _r, raw = C7.read_page(rp.parent)
        name = str(rp.parent.relative_to(PUB))
        out.append((name.split("/")[0][:10] + " " + name.split("/results/")[-1].split("/")[0], rep, raw, rp.parent))
    return out


def held_page_evidence() -> dict:
    """On every published histogram page with a report: the stored G1, R3's own r5-any rebuilt from
    cells.csv (R3 at its registered pin), and the scorer's r7-median-run3 rebuild with its P20 view
    (at the 1710 lock every page ran under; the rental-5 pages ran page policy, so their counts are
    not checked against repeat_rows, which they do not carry)."""
    import re

    import private_weight_reference as PW
    pages, moved = [], []
    for name, rep, raw, d in _held_pages():
        by: dict = {}
        for s in PW.read_samples(d / "cells.csv"):
            by.setdefault((re.search(r"histogram=([^;]+)", s.detail).group(1), s.tiles, s.arm), []).append(s)
        stored = rep["histogram_gates"]["G1_lock_thermal"]["verdict"]
        old = PW.histogram_gates(rep["treads_table"], by)["G1_lock_thermal"]["verdict"]
        r, why = C7.rebuild(rep, raw, 1710, check_counts=rep.get("repeat_rows") is not None)
        assert r is not None, (name, why)
        g, p = r["g1_r7"], r["g1_p20"]
        rows = r["treads_table"]
        for t in rows:
            if t["drift_flag_repeats"] and t["ms_p50"] and t["ms_p50_p20"]:
                moved.append(t["ms_p50"] / t["ms_p50_p20"] - 1)
        pages.append({"page": name, "stored_G1": stored, "r5_any_rebuilt": old, "r7": g["verdict"],
                      "r7_non_clean": f"{g['non_clean_rows']} / {g['repeat_rows']}",
                      "r7_run_guarded": sum(t["run_guard_repeats"] for t in rows), "r7_lost_cells": len(g["lost_cells"]),
                      "endpoint_flagged_recorded": sum(t["drift_flag_repeats"] for t in rows),
                      "p20": p["verdict"], "p20_share_pct": r6(100 * p["share"]), "p20_lost_cells": len(p["lost_cells"])})
    return {"pages": pages, "summary": {
        "report_pages": len(pages), "stored_equals_rebuilt_r5": sum(p["stored_G1"] == p["r5_any_rebuilt"] for p in pages),
        "r5_any_pass": sum(p["stored_G1"] == "PASS" for p in pages), "r7_pass": sum(p["r7"] == "PASS" for p in pages),
        "p20_pass": sum(p["p20"] == "PASS" for p in pages), "r7_lost_cells": sum(p["r7_lost_cells"] for p in pages),
        "p20_lost_cells": sum(p["p20_lost_cells"] for p in pages),
        "r7_run_guarded_repeats": sum(p["r7_run_guarded"] for p in pages),
        "flagged_repeats_kept": sum(p["endpoint_flagged_recorded"] for p in pages),
        "cell_medians_moved_by_keeping_them": {"cells": len(moved), "rms_pct": r6(100 * math.sqrt(st.fmean(x * x for x in moved))) if moved else 0.0,
                                               "worst_pct": r6(100 * max(moved, key=abs)) if moved else 0.0}}}


SAG_MHZ = 1600.0


def sag_rows(raw: list[dict], positions, every: int) -> tuple[list[dict], int]:
    """cells.csv rows with a real sag in every `every`-th repeat (by repeat index) that carries 15
    reads: the reads at `positions` set to SAG_MHZ, the repeat's time inflated as if each sagged read
    were the clock for 1/15 of its trials, its drift flag and median clock recomputed as R3 would."""
    out, n = [], 0
    for x in raw:
        reads = [float(v) for v in (x.get("clock_samples_mhz") or "").split()]
        if int(x["repeat"]) % every == 0 and len(reads) == 15:
            for i in positions:
                reads[i] = SAG_MHZ
            x = dict(x, clock_samples_mhz=" ".join(f"{v:.1f}" for v in reads),
                     ms_p50=str(float(x["ms_p50"]) * (1 + len(positions) * (1710.0 / SAG_MHZ - 1) / 15)),
                     clock_drift_ok=str(abs(reads[0] - reads[-1]) <= 0.05 * reads[0]),
                     sm_clock_load_mhz=str(st.median(reads)))
            n += 1
        out.append(x)
    return out, n


SAG_SCENARIOS = {"endpoint: last 5 reads, every 3rd repeat": (list(range(10, 15)), 3),
                 "endpoint: first 3 reads, every 3rd repeat": (list(range(0, 3)), 3),
                 "interior: reads 4 to 10, every 3rd repeat": (list(range(4, 11)), 3),
                 "interior: reads 6 to 8, every 3rd repeat": (list(range(6, 9)), 3),
                 "interior: reads 5 to 9, repeat 0 of every cell": (list(range(5, 10)), 99),
                 "board: reads 3 to 5 in every repeat": ([3, 4, 5], 1)}


def sag_control() -> dict:
    """The positive control (design-r7-review (a)): real sags injected into the 17 held pages'
    cells.csv, read by the scorer's rebuild. Per scenario: pages where every sagged repeat is out of
    its median, pages G1 fails, pages losing every cell, and pages where the slip predicate alone
    keeps at least 80% of the sags."""
    out = {}
    pages = _held_pages()
    for name, (pos, every) in SAG_SCENARIOS.items():
        caught = g1fail = slip_keeps = lost_all = 0
        for _n, rep, raw, _d in pages:
            sag, nh = sag_rows(raw, pos, every)
            hit = lambda x: int(x["repeat"]) % every == 0 and len((x.get("clock_samples_mhz") or "").split()) == 15
            reps = [next(iter(C7.repeat_rows([x]).values()))[0] for x in sag if hit(x)]
            caught += all(not C7.clean_r7(y, 1710) for y in reps)
            r, _why = C7.rebuild(rep, sag, 1710, check_counts=rep.get("repeat_rows") is not None)
            g1fail += r["g1_r7"]["verdict"] == "FAIL"
            lost_all += all(t["lost"] for t in r["treads_table"])
            slip_keeps += sum(y["status"] == "ok" and not C7.lock_slip(y, 1710) for y in reps) >= 0.8 * nh
        out[name] = {"pages": len(pages), "every sagged repeat out of its median (r7)": caught, "G1 FAILS (r7)": g1fail,
                     "every cell lost": lost_all, "slip predicate alone keeps at least 80% of the sags": slip_keeps,
                     "inflation of a sagged repeat %": r6(100 * len(pos) * (1710.0 / SAG_MHZ - 1) / 15)}
    return out


def g1rule() -> dict:
    ev = held_page_evidence()
    lo, hi = C7.READ_PERIOD_MS
    return {
        "registered": REGISTERED, "name": C7.NAMES["g1rule"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "the scorer: scripts/scoring/rental7/r7common.py (rebuild, g1_rule, repeat_rows, dip_run, run_guard, lock_slip, "
                "clean_r7, clean_p20, p20_report, page_report), which rebuilds every histogram page from its cells.csv; "
                "scripts/gh200_model_session.sh plan key g1-rule (locked_r3 --min-clean-repeats 0). R3 "
                "(scripts/private_weight_reference.py) and scripts/locked_r3.py are rental 6's, byte-identical to its "
                "registered pins (R3 sha256 daa40706...), and get no new flag",
        "label": "SEEN-DERIVED: the run length and the cap were read off the published held-lock pages (design-r7 1.1, 1.2; "
                 "design-r7-review (a)); they are gate thresholds, not model constants",
        "rule": {
            "where": "the SCORER applies the rule to the page's cells.csv (one row per timed repeat: status, ms_p50, "
                     "sm_clock_load_mhz, clock_drift_ok, host_bound and the repeat's clock_samples_mhz reads); it never "
                     "reads R3's stored G1 or the stored row medians; a page whose cells.csv rows per cell differ from its "
                     "report's repeat_rows, or with no cells.csv, no report or no lock_mhz, is NOT SCORED",
            "read_cadence": f"one NVML SM-clock read per burst, taken as the host finishes enqueueing it; bursts {lo} to {hi} ms "
                            f"apart at duty 0.25, 15 reads per repeat (5 bursts x 3 trials); so a run of 3 reads spans about "
                            f"{3 * (lo + hi) / 2 / 1000:.2f} s and a repeat about {15 * (lo + hi) / 2 / 1000:.1f} s",
            "dipped_read": "a read more than one 15 MHz step under the lock: 1694 MHz or less at the 1710 lock (reads are "
                           "integer MHz)",
            "clean_repeat": "status ok, ms_p50 > 0, NOT lock-slipped (the repeat's median clock, sm_clock_load_mhz, more than "
                            "one 15 MHz step under the lock, rental 6's lock_slip) and NOT run-guarded (RUN_GUARD_READS = 3 or "
                            "more CONSECUTIVE dipped reads of its 15); an isolated dipped read, at an end or inside the repeat, "
                            "stays in; the endpoint DRIFT flag (first and last read more than 5% apart) is recorded only",
            "cell": "time, clock and host_bound: the median (any) over its clean repeats; VALID with at least 6 clean of its "
                    "timed repeats, else LOST (rental 6's 6 of 9)",
            "LOST_and_early_stop": "R3 runs --slip-policy cell --min-clean-repeats 0 (the plan key): it times every repeat of "
                                   "every cell and never stops a cell or the page early, so R3's own r5-any LOST rule cannot "
                                   "cut a cell the scorer's rule keeps; the scorer applies 6 of 9 (design build review: at a "
                                   "30% flag rate R3 at min-clean 6 truncates about 19% of cells and leaves 3.1% under 6 "
                                   "timed repeats; 0 at min-clean 0)",
            "G1_drift_clause": f"G1 FAILS when the page's non-clean timed repeat rows (lock-slipped, run-guarded or not ok) "
                               f"exceed {C7.NON_CLEAN_CAP:.0%} of its timed repeat rows, every row counted (lost cells too); "
                               "rental 6's 10% slipped-row cap is inside it",
            "G1_clock_clauses": "every kept cell's clock read and every kept cell within one 15 MHz step of the page's top kept "
                                "clock (R3's G1 clock clauses, on the rebuilt cells)",
            "views": "a page whose G1 FAILS counts in ALL and is dropped in CLEAN (rental 2's rule 3)",
            "constants": {"RUN_GUARD_READS": C7.RUN_GUARD_READS, "NON_CLEAN_CAP": C7.NON_CLEAN_CAP,
                          "LOCK_STEP_MHZ": C7.LOCK_STEP_MHZ, "P20_CEILING": C7.P20_CEILING, "MIN_CLEAN": C7.MIN_CLEAN,
                          "READ_PERIOD_MS": list(C7.READ_PERIOD_MS)},
        },
        "p20_side_view": {"status": "DESCRIPTIVE, printed beside every rental-7 verdict; never a verdict",
                          "what": "the owner's P20 proposal (publish-r6 DRIFT.md), rebuilt from the same cells.csv: today's "
                                  "exclusions (status ok, ms_p50 > 0, not drift-flagged, not slipped), its own cell medians, "
                                  "clocks, host flags and lost cells (6 of 9), and G1 with the same clock clauses on ITS kept "
                                  "cells and its own clocks, failing past 20% non-clean rows"},
        "old_pages": {"rule": "R3 is byte-identical to rental 6's pin: every earlier page and every R3 output is unchanged",
                      "reproduction": "R3's own r5-any histogram_gates on each published page's stored rows and cells.csv "
                                      "reproduces the stored G1 on "
                                      f"{ev['summary']['stored_equals_rebuilt_r5']} of {ev['summary']['report_pages']} report pages",
                      "rental6_scorers": "rental 6's three scorers on the published rental-6 session write their committed "
                                         "score files byte for byte (tests/test_rental7_g1rule.py)"},
        "evidence_seen": ev,
        "positive_control": {"what": "design-r7-review (a): real sags injected into the 17 held pages' cells.csv (the design had "
                                     "no positive control: only 3 held repeats have a median under the lock)",
                             "results": sag_control(),
                             "review_sag_sim": "design-r7-review work/sag_sim.py (R 4000 per scenario per page): S_med alone "
                                               "passes 'last 5 reads at 1600 MHz in 30% of repeats' with P 1.000 and a kept "
                                               "cell biased over 0.54% with P 0.79; S_run3 0.000 in every scenario"},
        "limits": ["a sag shorter than RUN_GUARD_READS consecutive reads (about 0.45 s) is seen by no rule (r5-any included, "
                   "unless it touches an end): 2 consecutive reads at 1600 MHz in EVERY repeat bias a cell by up to "
                   f"{r6(100 * 2 * (1710 / 1600 - 1) / 15)}% and pass; in a minority of repeats the median does not move",
                   "the reads are one NVML read per burst after the host enqueues it; under the held lock they do not "
                   "price the trials (design-r7 1.1: slope +0.005, 95% [-0.005, +0.013]); a real sag is assumed to show "
                   "in 3 or more consecutive reads",
                   "the cap is rental 6's slip cap carried over; the held pages read at most "
                   f"{max(int(x['r7_non_clean'].split(' / ')[0]) / int(x['r7_non_clean'].split(' / ')[1]) for x in ev['pages']):.2%} "
                   "non-clean under r7"],
        "plumbing": {"R3_and_locked_r3": "unchanged (registered pins); a rental-7 page is R3 at --slip-policy cell "
                                         "--min-clean-repeats 0, so its own G1 (r5-any) and medians are written as always and "
                                         "the scorer ignores both",
                     "driver": "plan key g1-rule=r7-median-run3 (timed units with histogram= and slip-policy=cell only): "
                               "locked_r3 --min-clean-repeats 0; the unit's R3 dry run carries --slip-policy cell --lock-mhz "
                               "1710 --min-clean-repeats 0, what locked_r3 hands R3",
                     "page_check": "the scorer refuses (NOT SCORED) a page that did not run --slip-policy cell "
                                   "--min-clean-repeats 0"},
        "seen_data": ["every held-lock histogram page through rental 6 (17 with reports) is SEEN; the rule was chosen on them"],
    }


# --------------------------------------------------------------------------
# rental7-skew
# --------------------------------------------------------------------------

def label_timed7(model: str, label: str, n: int, counts_sha: str) -> str:
    seen = is_published(model, n, counts_sha)
    if label in ("uniform", "balanced"):
        return ("SEEN replicate (rental 6's A, B and C pages carry this counts sha256); only the board and the uniform "
                "cells' token-row shuffle are new") if seen else "BLIND (no published page carries this counts sha256)"
    base = "BLIND draw (counts sha256 on no published page); its (model, n, arm) residual structure is SEEN"
    if seen:
        return "SEEN (a published page carries this counts sha256)"
    return base + ("; BLIND-CALMODEL (8x7B is the CAL timing model)" if model == "mixtral-8x7b" else "")


def timed_cells() -> list[dict]:
    seed_pricer()
    rows = []
    for model in SHAPES:
        for page in TIMED_PAGES:
            doc = load(f"{model}-{page}")
            seed = doc["shuffle_seed"]
            for c in doc["cells"]:
                for arm in R6._arms(c):
                    p = R6.priced(model, arm, c["n"], c["counts"], 9, None if c["counts"] is None else seed)
                    sha = R6.sha_counts(c["counts"])
                    rows.append({"test": "E1", "model": model, "page": page, "copies": 9, "n": c["n"], "label": c["label"],
                                 "arm": arm, "counts_sha256": sha, "ms": R6._ms(p), "A_proxy_us": p["A_us"],
                                 "A_how": p["A_how"], "path": p["path"], "native_path": p["native_path"],
                                 "declared": p["declared"], "dead_ctas": p["dead"], "host_ok": C6.host_ok(p["v3"]),
                                 "label_blind": label_timed7(model, c["label"], c["n"], sha)})
    return rows


def _files(names, d: Path = HIST) -> dict:
    out = {}
    for nm in names:
        p = d / f"{nm}.json"
        doc = json.loads(p.read_text())
        out[f"{nm}.json"] = {"path": str(p.relative_to(REPO)), "sha256": sha256_file(p), "shuffle_seed": doc["shuffle_seed"],
                             "cells": [{"label": c["label"], "n": c["n"], "arms": R6._arms(c), "counts_sha256": R6.sha_counts(c["counts"]),
                                        **({"seed": c["provenance"]["seed"], "kind": c["provenance"]["kind"],
                                            "param": c["provenance"]["param"]} if c.get("provenance") else {})}
                                       for c in doc["cells"]]}
    return out


def _levers(rows: list[dict]) -> dict:
    thr = C6.LEVER_K * C6.SIGMA_PAGE
    return {R: sorted("/".join(map(str, _key(r))) for r in rows
                      if r["label"] != "balanced" and abs(r["ms"][R] / r["ms"]["v3"] - 1) > thr) for R in RIVALS_E1}


def skew_rows() -> list[dict]:
    rows = timed_cells()
    k = {_key(r): r for r in rows}
    for r in rows:
        u = k.get((r["model"], r["page"], r["n"], "uniform", r["arm"]))
        if r["label"] in SKEWED and u is not None:
            r["ratio"] = {v: r6(r["ms"][v] / u["ms"][v]) for v in M6.VARIANTS}
            r["lever_ratio"] = {"LT3": abs(r["ratio"]["LT3"] - r["ratio"]["v3"]) > 0.0075, "U": abs(r["ratio"]["v3"] - 1) > 0.045}
        b = k.get((r["model"], r["page"], r["n"], "balanced", r["arm"]))
        if r["label"] == "uniform" and b is not None:
            r["delta_v3"] = r6(r["ms"]["v3"] / b["ms"]["v3"] - 1)
    lev = _levers(rows)
    for r in rows:
        r["lever"] = {R: "/".join(map(str, _key(r))) in lev[R] for R in RIVALS_E1}
    return rows


_SEEN: list = []


def seen_e1() -> dict:
    """Rental 6's per-cell E1 residuals on Mixtral and Phi (ALL view), read through rental 6's own
    scorer functions on the published session (design-r7 work/seen_e1.py): SEEN. mu per (model,
    n, arm) over the histogram cells (balanced out), and the within-(model, n, arm) sd."""
    if _SEEN:
        return _SEEN[0]
    tree = PUB / "2026-10-13-nvidia_gh200_480gb-rental6-session"
    reg = C6.registration(REPO, "skew")
    files = reg["histograms"]["files"]
    preds = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in reg["predictions"]}
    tv = C6.C3.TimedView("ALL")
    es: dict = {}
    for model in SHAPES:
        for page in TIMED_PAGES:
            rep, _why = C6.page_report(tree, model, f"sk{page.lower()}", tv, files.get(f"{model}-{page}.json"))
            if rep is None:
                continue
            c, _d, A = C6.valid_cells(rep, preds, model, page)
            for k, t in c.items():
                if k[3] != "balanced":
                    es.setdefault(f"{k[0]}|{k[2]}|{k[4]}", []).append(C6.carried(preds[k], "v3", A) / t - 1)
    mu = {g: st.fmean(v) for g, v in es.items()}
    res = [x - mu[g] for g, v in es.items() for x in v]
    out = {"mu": mu, "sd_cell": st.pstdev(res), "cells": len(res)}
    _SEEN.append(out)
    return out


def e1_power_block(rows: list[dict], device: str = "cpu") -> dict:
    seen = seen_e1()
    e1rows = [r for r in rows if r["label"] != "balanced"]
    out = {}
    for name, (scale, board) in E1_SCENARIOS.items():
        p = C7.e1_power(e1rows, seen["mu"], seen["sd_cell"], scale=scale, board=board, R=POWER_R, device=device)
        p.pop("seconds")
        out[name] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()}
    return out


def skew() -> dict:
    rows = skew_rows()
    E1 = [r for r in rows if r["label"] != "balanced"]
    lev = _levers(rows)
    summ = {}
    for m in SHAPES:
        rr = [r["ratio"]["v3"] - 1 for r in rows if r["model"] == m and "ratio" in r]
        summ[m] = {"r_v3_minus_1_pct": [r6(100 * min(rr)), r6(100 * max(rr))],
                   "T_v3_us": [r6(1e3 * min(r["ms"]["v3"] for r in rows if r["model"] == m)),
                               r6(1e3 * max(r["ms"]["v3"] for r in rows if r["model"] == m))]}
    gaps = {}
    for R in RIVALS_E1 + ("v3_nob",):
        g = [r["ms"][R] / r["ms"]["v3"] - 1 for r in E1]
        gaps[R] = {"cells": len(g), "rms_pct": r6(100 * math.sqrt(st.fmean(x * x for x in g))), "worst_pct": r6(100 * max(g, key=abs)),
                   "lever_cells_0.18": len(lev[R]) if R in lev else None,
                   "lever_by_model": {m: sum(x.startswith(m) for x in lev.get(R, [])) for m in SHAPES} if R in lev else None}
    skewed = [r for r in rows if r["label"] in SKEWED]
    ctrl = [r for r in rows if r["label"] not in SKEWED]
    r6reg = C6.registration(REPO, "skew")
    r6pred = {_key(r): r for r in r6reg["predictions"]}
    moved = [r["ms"]["v3"] / r6pred[_key(r)]["ms"]["v3"] - 1 for r in skewed if _key(r) in r6pred]
    ctrl_same = [abs(r["ms"]["v3"] / r6pred[_key(r)]["ms"]["v3"] - 1) for r in ctrl if _key(r) in r6pred]
    seen = seen_e1()
    return {
        "registered": REGISTERED, "name": C7.NAMES["skew"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/rental7/score_skew.py; predictions reg6.priced via reg7.py (model v3 unchanged); "
                "histograms scripts/skew_synth7.py --draw7",
        "model": MODEL_V3 + "; every cell carries v2, v3, v3_nob, v3_noca, LT3, v3_all and H8 (ms)",
        "what": "the certification rental 6 could not give: Mixtral and Phi in CLEAN, on fresh draws at rental 6's treads, "
                "under the drift rule of rental7-g1rule",
        "histograms": {"synthetic": f"SYNTHETIC: rental 6's rule exactly (skew_synth.pages6), seed base {20261020} instead of "
                                    "20261010: seed = 20261020 + 1000 x shape index (rental 6's: Mixtral 0, OLMoE 1, Phi 2) + "
                                    "10 n + label index (PT 0, PW 1, DW 2, PW2 3), so seed(n) is rental 6's seed(n + 1); "
                                    "shuffle seed 20261020 (owner decision 5: fresh seeds, same treads)",
                       "files": _files([f"{m}-{p}" for m in SHAPES for p in TIMED_PAGES])},
        "leakage": {"skewed_cells": len(skewed), "skewed_cells_whose_counts_sha256_is_published": sum(is_published(r["model"], r["n"], r["counts_sha256"]) for r in skewed),
                    "control_cells": len(ctrl), "control_cells_whose_counts_sha256_is_published": sum(is_published(r["model"], r["n"], r["counts_sha256"]) for r in ctrl),
                    "skewed_v3_against_rental6_draws_pct": {"rms": r6(100 * math.sqrt(st.fmean(x * x for x in moved))),
                                                            "worst": r6(100 * max(moved, key=abs))},
                    "controls_v3_against_rental6_max_rel": r6(max(ctrl_same)) if ctrl_same else None,
                    "key": "a cell is published when a published page or an earlier registered histogram file carries the same (model, n, counts sha256)",
                    "note": "the controls' counts are rental 6's; only the uniform cells' token-row shuffle moves (content keys)"},
        "cells": {"common": "R3's tile 32x64x64 w8 s4 at G = 8; NATIVE and SHARED at 9 copies (not vLLM's tile: nothing here is "
                            "the working bar on vLLM's tile)",
                  "pages": {"A": "PT, PW, uniform, balanced", "B": "DW, uniform, balanced", "C": "PW2, uniform, balanced"},
                  "treads": {m: list(v) for m, v in SHAPES.items()}, "n_stratum": "G3's stratum is the tread's rank within its shape",
                  "count": f"{len(rows)} cells, {len(E1)} E1 cells (skewed and uniform); {sum(r['host_ok'] for r in rows)} pass the host rule"},
        "predictions": rows,
        "summary": summ, "rival_gaps_on_E1": gaps,
        "levers": {"rule": f"|T_R / T_v3 - 1| > 3 sigma_page = {C6.LEVER_K * C6.SIGMA_PAGE:.2%} (sigma_page 0.18%, rental 3 rep8)", "cells": lev},
        "gates": {
            "G1_lock_thermal": "rental7-g1rule: the scorer rebuilds the page from its cells.csv under r7-median-run3 (R3's "
                               "stored G1 and medians are never read); a page whose cells.csv counts differ from its report, or "
                               "that did not run --slip-policy cell --min-clean-repeats 0, is NOT SCORED; a page failing the "
                               "rebuilt G1 counts in ALL and is dropped in CLEAN (rental 2's rule 3)",
            "repeats": "a cell is VALID with at least 6 clean repeats of its timed repeats under r7-median-run3 (R3 times all 9: "
                       "min-clean 0); its time, clock and host flag are the scorer's medians over its clean repeats",
            "G2_host_bound": "rental 3's registered host rule (T_v3 + F240 >= 0.40 ms on the REGISTERED prediction) AND the page's "
                             "timer flag host_bound",
            "G4_cell_clock": "the cell's clean-repeat median clock at least 1705 MHz, else the CELL is dropped",
            "G5_provenance": "as rental 6: histogram file sha256, every row's counts_sha256, bincount_ok, the shuffle seed; else NOT SCORED",
            "G3": "rental 5's text verbatim (rental 6's, with delta_v3), clusters = the six pages (df 5)",
            "G3_void_all": "rental 6's owner decision kept: a G3 FAIL with Q not rejecting voids every histogram cell; E1 printed "
                           "DESCRIPTIVELY on the voided cells",
            "no_report": "a page with no report.json or no cells.csv is NOT SCORED; the scorer reads both (cells.csv for every "
                         "repeat, the report for provenance, lock_mhz and repeat_rows)",
            "views": "ALL and CLEAN (rental 2's rule 3); the registered verdict holds only if both views hold",
            "p20_view": "DESCRIPTIVE: every test re-read on the P20 rebuild (its own medians, clocks, lost cells and G1 with "
                        "the 20% ceiling), printed beside",
        },
        "tests": {
            "E1-SKEW": {"statistic": "e = T_v3 / T_meas - 1 (T_v3 the registered prediction carried to the page's own A) over every "
                                     "valid histogram cell (PT PW PW2 DW uniform)",
                        "rule": "HOLDS iff the POOLED valid cells meet rms <= 2%, max |e| <= 5% and |mean e| <= 1%, AND each shape's "
                                "valid cells meet the same bar (the new per-shape clause), in each view; the registered verdict "
                                "holds only if both views hold",
                        "certification": f"a shape is CERTIFIED in CLEAN iff its CLEAN cells meet the bar and at least "
                                         f"{C7.MIN_CLEAN_PAGES_PER_SHAPE} of its 3 pages are in CLEAN; with fewer it reads NOT CERTIFIED",
                        **C6.E1_BAR, "rental5_bands_printed": C6.E1_R5_BANDS,
                        "on_G3_void_all": "NOT SCORED; E1 printed DESCRIPTIVELY on the voided cells"},
            "E1-UNI": {"statistic": "e on the balanced cells", "rule": "the E1-SKEW bar, printed"},
            "E1-rivals": {"rule": "rental 5's: a rival is EXCLUDED if its rms on its registered lever cells is at least 2x v3's there and "
                                  "at least 0.76%; UNDECIDED below 6 surviving lever cells",
                          "rivals": list(RIVALS_E1), "rival_factor": C6.RIVAL_FACTOR, "rival_floor": C6.RIVAL_FLOOR,
                          "min_lever_cells": C6.MIN_LEVER,
                          "by_construction": {R: ("decidable" if len(lev[R]) >= C6.MIN_LEVER else
                                                  f"UNDECIDED by construction ({len(lev[R])} lever cells)") for R in RIVALS_E1},
                          "term_b": "rental 7 does not test term (b) on time: (b) moves no E1 cell past 0.54% on these shapes; Q1-H is (b)'s test"},
            "SKEW-RATIO": {"statistic": "r = T(h) / T(uniform), same page, n and arm; e = r_v3 / r_meas - 1",
                           "rule": "rental 5's: HOLDS if rms <= 1.5% and max |e| <= 4%; LT3 and U by the rival rule on their ratio lever cells",
                           "rms_max": 0.015, "max_abs": 0.04},
        },
        "power": {"model": "mc7 block B (design-r7) with the review's per-shape clause (work/mc_e1_shape.py): e = scale x mu(model, n, "
                           "arm) + a per-model board shift + cell noise; mu and the cell sd SEEN from rental 6's Mixtral and Phi pages "
                           f"(ALL view, rental 6's scorer functions): sd_cell {r6(100 * seen['sd_cell'])}% over {seen['cells']} cells",
                  "mu_pct": {g: r6(100 * v) for g, v in sorted(seen["mu"].items())},
                  "device": "cpu", "dtype": "float64", "R": POWER_R, "scenarios": e1_power_block(rows),
                  "mps_check": mps_check()},
        "noise": {"sigma_page": C6.SIGMA_PAGE, "lever_sigma": "0.18% (rental 3 rep8): threshold 0.54%"},
        "labels": {"mixtral-8x7b": "skewed: BLIND draws, BLIND-CALMODEL; uniform and balanced: SEEN replicates",
                   "phi-3.5-moe": "skewed: BLIND draws; uniform and balanced: SEEN replicates; Phi carries Mixtral's shapes (rental 6 "
                                  "owner decision 4); h SEEN-FITTED, but no rental-7 cell is IN-SAMPLE for h (OLMoE is not timed)",
                   "rental6": "rental 6's verdicts stand and its pages are SEEN"},
        "scorer_ambiguities": AMBIGUITIES,
        "missing": "a missing page's cells are NOT SCORED; the six skew pages carry nodrop=1 and no drop-group",
        "seen_data": ["the shapes are SEEN routing statistics", "mu and sd_cell SEEN (rental 6)", "sigma_page SEEN",
                      "h SEEN-FITTED, content_a SEEN-SELECTED (model v3)"],
    }


# --------------------------------------------------------------------------
# rental7-q1
# --------------------------------------------------------------------------

def q1_rows() -> tuple[list[dict], list[dict]]:
    """(the rental-7 skew E1 rows, rental 6's registered Q1 prediction rows verbatim)."""
    q1 = C6.registration(REPO, "q1")["predictions"]
    return skew_rows(), [dict(r) for r in q1]


def q1h_pairs(e1, q1) -> list[dict]:
    rows = e1 + q1
    ctrl = ("uniform", "balanced")
    want = Q1H_DESIGN | Q1H_ADDED
    sel = lambda r: ((r["test"] == "Q1" and (r["model"], r["n"]) in Q1H_DESIGN)
                     or (r["test"] == "E1" and r["label"] in ctrl and (r["model"], r["n"]) in want))
    out = []
    for p in R6._pairs(rows, sel):
        if p["native_path"] != "block-scan" or not p["host_ok"]:
            continue
        added = (p["model"], p["n"]) in Q1H_ADDED
        p["population"] = ("ADDED by design-r7-review (d), SEEN-SELECTED (zero-cost clusters on the skew pages)" if added
                           else "the design's population (design-r7 3.2)")
        p["label_blind"] = ("BLIND (c15 SHARED: the copies clause's blind side)" if p["copies"] == 15 else
                            "SEEN replicate (the same cells as published pages; only the board is new)")
        out.append(p)
    return out


def q1h_power_block(H: list[dict], device: str = "cpu") -> dict:
    design = [p for p in H if (p["model"], p["n"]) in Q1H_DESIGN]
    zero = {k: 0.0 for k in Q1H_ANCHOR_US}
    pts = {"registered population | v3 | tau 0.9": dict(pairs=H, tau_us=0.9),
           "registered population | v3 | tau 1.3": dict(pairs=H, tau_us=1.3),
           "registered population | v3 | tau 0.9 | anchors 0": dict(pairs=H, tau_us=0.9, anchor=zero),
           "registered population | v3 | tau 0.9 | c15 effect +1 us": dict(pairs=H, tau_us=0.9, c15_effect_us=1.0),
           "registered population | v3 | tau 0.9 | c15 effect +2 us": dict(pairs=H, tau_us=0.9, c15_effect_us=2.0),
           "registered population | v2 true | tau 0.9": dict(pairs=H, tau_us=0.9, truth="v2"),
           "registered population | H8 true | tau 0.9": dict(pairs=H, tau_us=0.9, truth="H8"),
           "design population only (3 clusters) | v3 | tau 0.9": dict(pairs=design, tau_us=0.9),
           "design population only (3 clusters) | v3 | tau 1.3": dict(pairs=design, tau_us=1.3)}
    out = {}
    for name, kw in pts.items():
        anchor = kw.pop("anchor", Q1H_ANCHOR_US)
        p = C7.q1h_power(kw.pop("pairs"), Q1H_SIG_US, anchor, R=POWER_R, device=device, rivals=RIVALS_GAP, **kw)
        p.pop("seconds")
        out[name] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()}
    return out


def q1c_power_block(device: str = "cpu") -> dict:
    reg6q = C6.registration(REPO, "q1")["tests"]["Q1-C"]
    out = {}
    for x9 in Q1C_X9_SEEN:
        for truth in ("CALL", "DEAD"):
            out[f"x9 {x9} | board {Q1C_BOARD_US} + replicate 0.45 | {truth} true"] = C6.q1c_power(
                x9, truth, reg6q["dead_ratio"], sig_x=0.45, board=Q1C_BOARD_US, se_D=reg6q["se_D_us"], device=device)
    return out


def q1() -> dict:
    e1, q1r = q1_rows()
    H = q1h_pairs(e1, q1r)
    rows = e1 + q1r
    ctrl = ("uniform", "balanced")
    selP = lambda r: ((r["test"] == "Q1" and r["model"] == "phi-3.5-moe" and r["n"] == 1)
                      or (r["test"] == "E1" and r["model"] == "mixtral-8x7b" and r["n"] == 3 and r["label"] in ctrl))
    P = [p for p in R6._pairs(rows, selP) if p["host_ok"]]
    selSK = lambda r: r["test"] == "E1" and r["label"] in SKEWED and r["model"] == "phi-3.5-moe" and r["n"] == 2
    SK = R6._pairs(rows, selSK)
    reg6q = C6.registration(REPO, "q1")
    qc = reg6q["tests"]["Q1-C"]
    clusters = sorted({p["cluster"] for p in H})
    seen_pairs = sum(p["copies"] != 15 for p in H)
    return {
        "registered": REGISTERED, "name": C7.NAMES["q1"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/rental7/score_q1.py; r7common.q1h_rule, q1h_power; r6common.q1c_rule, q1c_power",
        "what": "rental 6's lost Q1 units, all six on this board (Qwen1.5 c9 rerun: Q1-C's D and the copies clause are same-board "
                "contrasts)",
        "pages": {"q9": "c9 balanced pages (SEEN replicates)", "q15": "c15 balanced pages (SHARED BLIND, NATIVE SEEN)",
                  "models": {m: list(C6.registration(REPO, "q1")["pages"]["models"][m]) for m in Q1_MODELS},
                  "files": {k: v for k, v in reg6q["pages"]["files"].items()},
                  "files_note": "rental 6's registered q1-*.json verbatim (balanced cells only, no draw)"},
        "predictions": q1r,
        "predictions_note": "rental 6's registered Q1 predictions verbatim (copied from 2026-10-09-rental6-q1-gh200), so its g_R "
                            "and A proxies are unchanged; the skew-page pairs read rental7-skew's predictions",
        "statistic": reg6q["statistic"],
        "tests": {
            "Q1-H": {"label": f"REPLICATE: {seen_pairs} of its {len(H)} pairs are SEEN replicates (every c9 and skew-page "
                              "control pair); only the c15 SHARED cells are BLIND, so the verdict is BLIND only in the copies clause",
                     "population": "CONTROL pairs (uniform, balanced) whose NATIVE takes block-scan and that pass the host rule: "
                                   "Phi n 2 (skew pages A, B, C and Q1 c9, c15), Qwen1.5 n 2 (c9, c15), JetMoE n 4 (c9, c15); "
                                   "ADDED (design-r7-review (d), SEEN-SELECTED, zero cost: the cells are on the skew pages): Mixtral "
                                   "n 6 and Phi n 4 controls on pages A, B, C",
                     "clusters": clusters, "pairs": H,
                     "rule": "v3 HOLDS iff |mean g| <= 1.5 us (pair-weighted) AND the copies clause (|d| <= 1.5 us, d the pair-weighted "
                             "mean over models with valid Q1 pairs at both c9 and c15 of mean g(c15) - mean g(c9)); the copies clause "
                             "undecided leaves v3 UNDECIDED (FAILS if the level clause fails); a rival is EXCLUDED iff |mean g - g_R| > "
                             "3 se and > 1.5 us, scored whenever 6 pairs and 3 clusters are valid WHATEVER the copies clause reads "
                             "(owner decision 2); se CR1 on (model, n) and at least 3 clusters, else UNDECIDED",
                     "rivals": list(RIVALS_GAP), "min_pairs": C7.MIN_PAIRS, "min_clusters": C7.MIN_CLUSTERS,
                     "not_separated": "v3_all is not separated from v3 by construction (g_R 0 on every block-scan pair): stated here, "
                                      "not discovered",
                     "power": {"model": "g = anchor(cluster) + tau x N(0,1) per cluster + sigma(cluster) x N(0,1) per pair (+ g_R of "
                                        "the true rival; + a c15 effect on c15 Q1 pairs); no pair losses (r7 lost no held cell)",
                               "anchor_us": Q1H_ANCHOR_US, "sigma_us": Q1H_SIG_US,
                               "label": "SEEN inputs (design-r7 3.2 and design-r7-review work/seen_g_clusters.json, mc_q1h.py)",
                               "device": "cpu", "dtype": "float64", "R": POWER_R, "points": q1h_power_block(H),
                               "mps_check": mps_check()}},
            "Q1-C": {**qc, "note": "rental 6's rule verbatim (D = x15 - x9 at Qwen1.5 SHARED n 1)",
                     "power": {"device": "cpu", "dtype": "float64", "points": q1c_power_block()}},
            "Q1-P": {"status": "DESCRIPTIVE", "population": "small-batch NATIVE pairs passing the host rule: Phi n 1 (Q1 c9, c15), "
                                                          "Mixtral n 3 controls on the rental-7 pages A, B, C", "pairs": P},
            "Q1-SK": {"status": "OPEN, printed: mean g over Phi n 2 skewed pairs minus mean g over the same pages' Phi n 2 control "
                                "pairs", "pairs": SK},
        },
        "gates": "as rental7-skew (G1 under r7-median-run3, 6 of 9 clean, G2 host rule + timer flag, G4 1705, G5); views ALL and "
                 "CLEAN; the P20 view printed",
        "labels": {"c9": "SEEN replicates", "c15": "SHARED BLIND; NATIVE SEEN (it declares E at any copies count)",
                   "anchors_sigma": "SEEN", "n1_exclusion": "SEEN-INFORMED (rental 6)"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["every c9 and skew-page control pair is a SEEN replicate", "anchors and sigma SEEN", "h SEEN-FITTED"],
    }


# --------------------------------------------------------------------------
# rental7-l2open
# --------------------------------------------------------------------------

def l2open_predictions() -> list[dict]:
    import r3_timing_model as TM
    seed_pricer()
    doc = load(f"{L2_MODEL}-l2open")
    rk = json.loads((HIST / "l2open_rowkey.json").read_text())
    rkc = {(c["label"], c["n"], c["arm"]): c for c in rk["cells"]}
    old = TM.set_model(L2_MODEL)
    bm = dict(TM.BYTE_MODEL)
    TM.set_model(old)
    q = lambda read, n, g: r6((read - n * bm[f"operand_per_tile_{g}"]) / bm[f"W_{g}"])
    out = []
    for c in doc["cells"]:
        for arm in R6._arms(c):
            p = R6.priced(L2_MODEL, arm, c["n"], c["counts"], 9, None if c["counts"] is None else doc["shuffle_seed"])
            rr = rkc[(c["label"], c["n"], arm)]
            sha = R6.sha_counts(c["counts"])
            qv3 = {g: q(p["bytes_v3"][g], c["n"], g) for g in ("w1", "w2")}
            qrk = {f"ROWKEY{C:g}": q(rr[f"read_bytes_C{C:g}"], c["n"], "w1") for C in rk["C_MiB"]}
            if c["label"] in ("PT", "PW"):
                lab = ("SEEN-SELECTED replicate tread: C was tuned on rental 5's skc (n 2, 4, 16) and rental 6's skbytes "
                       "(n 3, 6, 12), so this (shape, n) is SEEN; the draw is new (BLIND counts)" if c["n"] in (4, 16)
                       else "BLIND: a new tread (n 8) and a new draw")
            else:
                lab = ("SEEN replicate (a published page carries this (model, n, counts sha256))"
                       if is_published(L2_MODEL, c["n"], sha) else "BLIND control (no published page carries this cell)")
            out.append({"model": L2_MODEL, "n": c["n"], "label": c["label"], "arm": arm, "counts_sha256": sha,
                        "skewed": c["label"] in ("PT", "PW"), "q_pred": {"v3": qv3, **{k: {"w1": v} for k, v in qrk.items()}},
                        "q_pred_v3_noca": {g: q(p["bytes_v2"][g], c["n"], g) for g in ("w1", "w2")},
                        "read_bytes_w1": {"v3": r6(p["bytes_v3"]["w1"]), **{f"ROWKEY{C:g}": rr[f"read_bytes_C{C:g}"] for C in rk["C_MiB"]}},
                        "gaps_w1_pct": {"ROWKEY60_vs_ROWKEY39.5": r6(100 * (qrk["ROWKEY60"] / qrk["ROWKEY39.5"] - 1)),
                                        "v3_vs_ROWKEY60": r6(100 * (qv3["w1"] / qrk["ROWKEY60"] - 1)),
                                        "v3_vs_ROWKEY39.5": r6(100 * (qv3["w1"] / qrk["ROWKEY39.5"] - 1))},
                        "label_blind": lab})
    return out


def l2open() -> dict:
    rk = json.loads((HIST / "l2open_rowkey.json").read_text())
    pred = l2open_predictions()
    sk = [r for r in pred if r["skewed"]]
    sep = {}
    for a, b in (("ROWKEY60", "ROWKEY39.5"), ("v3", "ROWKEY60"), ("v3", "ROWKEY39.5")):
        g = [r["q_pred"][a]["w1"] / r["q_pred"][b]["w1"] - 1 for r in sk]
        sep[f"{a} vs {b}"] = {"rms_pct": r6(100 * math.sqrt(st.fmean(x * x for x in g))), "min_abs_pct": r6(100 * min(abs(x) for x in g)),
                              "per_n_pct": {n: [r6(100 * x) for x, r in zip(g, sk, strict=True) if r["n"] == n] for n in (4, 8, 16)}}
    return {
        "registered": REGISTERED, "name": C7.NAMES["l2open"], "status": "REGISTERED BEFORE ANY PAGE (OPEN: no model change follows)",
        "design": DESIGN,
        "tool": "scorer scripts/scoring/rental7/score_l2open.py; v3 bytes reg6.priced (MODEL M4, unchanged); rivals "
                "scripts/scoring/rental7/rowkeys7.py (rental 6's rowkeys.py, unchanged, at 39.5 and 60 MiB); page scripts/skew_synth7.py",
        "status_note": "OPEN-L2SKEW, rental 6's open residual (v3 FAILS skewed OLMoE w1 to +20.6%): a diagnosis, not a model test; "
                       "it adds nothing to time (bytes do not reach time at G 8); drop it first (drop-group byt)",
        "page": {"file": _files([f"{L2_MODEL}-l2open"]), "cells": "PT, PW, uniform, balanced at n 4, 8, 16; NATIVE and SHARED at "
                                                                "n <= 9, NATIVE only at n 16; G 8, 9 copies declared",
                 "seed": "l2open seed = 20261030 + 10 n + label index (PT 0, PW 1), shuffle seed 20261030: the draws design-r7 "
                         "work/olmoe_w1.py priced (review (e) fix 1: the page the build emits is the page priced; every row of it "
                         "is priced here, both arms, controls included)",
                 "unit": "olmoe-1b-7b bytes label=l2open byte-groups=8 (one counter page, about 2 min, est 4)"},
        "rivals": {"v3": "the model (MODEL M4 content keys), unchanged",
                   "ROWKEY39.5": "rental 6's registered rival, row-granular w1 A keys, one exact LRU at 39.5 MiB",
                   "ROWKEY60": "the same rival at C = 60 MiB, the ONE capacity registered here (review (e) fix 2: 55 and 60 sit 1 "
                               "to 3 points apart, so the 2x rule could never name either); SEEN-chosen (design-r7 3.4: C swept "
                               "30 to 80 MiB over the published rental-5 and rental-6 OLMoE byte pages, best at 55 and 60); its "
                               "numbers on this page are BLIND (new draws)",
                   "file": {"path": str((HIST / "l2open_rowkey.json").relative_to(REPO)), "sha256": sha256_file(HIST / "l2open_rowkey.json"),
                            "device": rk["device"], "dtype": rk["dtype"], "cpu_float64_check": rk["cpu_float64_check"]}},
        "byte_predictions": pred,
        "separations_on_skewed_w1_rows": sep,
        "noise": {"counter": COUNTER_NOISE, "note": "every registered separation above is many times the counter noise; rowkey "
                                                    "60 against 39.5 separates mainly at n 4 (review (e))"},
        "statistic": "per w1 row (cell, arm): q_counted = (dram_bytes_read - n x operand_per_tile_w1) / W_w1; e_R = q_pred_R / "
                     "q_counted - 1",
        "rule": f"the nearest of {', '.join(C7.L2_RIVALS)} by rms of e over the SKEWED (PT, PW) w1 rows is NAMED iff the next "
                f"nearest's rms is at least {C7.L2_FACTOR:g}x its rms and every skewed row's |e| of the nearest is at most "
                f"{C7.L2_MAX_ABS:.0%} (the 5% clause on skewed rows only, review (e) fix 3: uniform misses by about +6% under "
                "the row key at C 60); else NOT SEPARATED. Uniform and balanced rows, and w2 under v3, are printed",
        "gates": {"clock": "rental 6's counter-page clock rule (r6common.clock_rule): lock_gate.check_page PASS, then each cell's SM "
                           "clock in [1640, 1710] MHz (out-of-band cells dropped)",
                  "G5": "the page's histogram sha256 and shuffle seed, every cell's counts sha256; else NOT SCORED",
                  "views": "one view: a counter page has no G1; the clock rule decides what counts"},
        "labels": {"n4_n16_skewed": "SEEN-SELECTED replicate treads (C was tuned on them), BLIND draws",
                   "n8_skewed": "BLIND (new tread)", "rival_ROWKEY60": "SEEN-chosen capacity, BLIND numbers"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["C = 60 MiB SEEN-chosen", "the row-key rule SEEN-SELECTED (rental 6)", "OLMoE is CAL-counters (d, kappa)"],
    }


# --------------------------------------------------------------------------
# the 10 scorer ambiguities of design-r7-review (h), resolved
# --------------------------------------------------------------------------

AMBIGUITIES = {
    "1 the registered rule and where it runs": "r7-median-run3 (S_med plus the run guard, owner decision 1), applied by the "
                                               "scorer to cells.csv (r7common.rebuild); R3 is unchanged and runs --slip-policy "
                                               "cell --min-clean-repeats 0; a page that ran another min-clean is NOT SCORED",
    "2 S_med's G1 drift clause": "the run guard excludes a repeat (3 or more consecutive reads a step under the lock); G1 fails "
                                 "on drift only past 10% non-clean repeat rows (slipped, run-guarded or not ok); the endpoint flag "
                                 "is recorded only",
    "3 the P20 side view": "rebuilt from the same cells.csv with today's exclusions: its own medians, clocks, host flags and lost "
                           "cells (6 of 9), G1's clock clauses on its kept cells and the 20% ceiling; DESCRIPTIVE, every test "
                           "printed on it",
    "4 per-shape E1": "read in each view on that view's cells; the registered verdict holds only if both views hold; a shape is "
                      "CERTIFIED in CLEAN only with at least 2 of its 3 pages in CLEAN and its CLEAN cells meeting the bar",
    "5 Q1-H se with few clusters": "CR1 on (model, n) clusters needs at least 3 clusters; fewer is UNDECIDED (no C / (C - 1) "
                                   "with C = 1); the registered population has 5",
    "6 pair or cluster weights": "pair-weighted mean (rental 6's); the equal-cluster-weight mean is printed",
    "7 the copies clause": "matched by model: per model with valid Q1 pairs at both c9 and c15, mean g(c15) - mean g(c9), "
                           "pair-weighted over those models; identical to rental 6's unmatched means when all six Q1 pages are valid",
    "8 one copies side only": "the copies clause is UNDECIDED; v3 reads UNDECIDED (FAILS if its level clause fails); the rivals "
                              "are still scored (owner decision 2)",
    "9 the share denominator": "every timed repeat row of the page in cells.csv (R3 at min-clean 0 times all 9 of every "
                               "cell); cells.csv's rows per cell must equal the report's repeat_rows, else NOT SCORED",
    "10 l2open": "rows are (cell, arm) on w1 only for the verdict (PT, PW skewed; uniform and balanced printed), NATIVE and SHARED "
                 "at n <= 9 and NATIVE at n 16; w2 under v3 printed; rental 6's counter clock rule; one page, one view",
}


BUILDERS = {"g1rule": g1rule, "skew": skew, "q1": q1, "l2open": l2open}


def _txt_g1rule(d):
    s = d["evidence_seen"]["summary"]
    return [f"held report pages {s['report_pages']}: stored G1 reproduced by r5-any {s['stored_equals_rebuilt_r5']}; "
            f"r5-any passes {s['r5_any_pass']}, r7-median-run3 {s['r7_pass']}, P20 {s['p20_pass']}; lost cells r7 "
            f"{s['r7_lost_cells']}, P20 {s['p20_lost_cells']}"] + \
           [f"sag {k}: {v}" for k, v in d["positive_control"]["results"].items()]


def _txt_skew(d):
    return [f"{m}: r_v3 - 1 {v['r_v3_minus_1_pct']} %; T_v3 {v['T_v3_us']} us" for m, v in d["summary"].items()] + \
           [f"rival {R}: rms gap {v['rms_pct']}%, worst {v['worst_pct']}%, lever cells {v['lever_cells_0.18']}"
            for R, v in d["rival_gaps_on_E1"].items()] + \
           [f"power {k}: {v}" for k, v in d["power"]["scenarios"].items()]


def _txt_q1(d):
    return [f"Q1-H {k}: {v}" for k, v in d["tests"]["Q1-H"]["power"]["points"].items()] + \
           [f"Q1-C {k}: {v}" for k, v in d["tests"]["Q1-C"]["power"]["points"].items()]


def _txt_l2open(d):
    return [f"{k}: {v}" for k, v in d["separations_on_skewed_w1_rows"].items()]


TEXT = {"g1rule": _txt_g1rule, "skew": _txt_skew, "q1": _txt_q1, "l2open": _txt_l2open}
