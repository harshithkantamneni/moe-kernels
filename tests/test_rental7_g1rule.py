"""Rental 7's histogram G1 drift rule off the GPU (registration 2026-10-13-rental7-g1rule; design-r7
section 1, design-r7-review (a) and (b), the build review, owner decision 1 of 2026-10-13).

r7-median-run3 is the SCORER's rule (scripts/scoring/rental7/r7common.py), rebuilt from each page's
cells.csv; R3 stays byte-identical to rental 6's pin and runs at --min-clean-repeats 0. Here: R3's
pin; R3's own r5-any rebuilt from cells.csv gives the stored G1 on all 17 report pages; the
scorer's rebuild agrees with the build review's independent CSV-only implementation on rental 6's
10 held pages; it passes all 17 held pages and loses no cell; real sags injected into their
cells.csv (endpoint and interior runs, a whole board) are caught; R3 at min-clean 0 times every
repeat; and rental 6's scorers still write their committed outputs byte for byte."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
R7DIR = REPO / "scripts" / "scoring" / "rental7"
for p in (REPO, REPO / "scripts", REPO / "tests", R7DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import private_weight_reference as PW  # noqa: E402
import r7common as C7  # noqa: E402
import test_rental6_plumbing as T6  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PUB = REPO / "results" / "published"
REG = REPO / "docs" / "registered"
H6 = REG / "2026-10-09-rental6-skew-hist"
LOCK = 1710


def _pages():
    """Every histogram page with a report published before the registration (the 17 SEEN pages the
    rule was chosen on; rental 7's own pages are left out, as reg7._held_pages leaves them):
    (name, report, cells.csv rows, R3 samples by cell)."""
    out = []
    for rp in sorted(PUB.glob("**/private_weight_reference/*/report.json")):
        rep = json.loads(rp.read_text())
        if not rep.get("histogram_gates") or "rental7" in str(rp):
            continue
        _r, raw = C7.read_page(rp.parent)
        by: dict = {}
        for s in PW.read_samples(rp.parent / "cells.csv"):
            by.setdefault((re.search(r"histogram=([^;]+)", s.detail).group(1), s.tiles, s.arm), []).append(s)
        out.append((str(rp.parent.relative_to(PUB)).split("/results/")[-1].split("/")[0], rep, raw, by))
    return out


PAGES = _pages()


def _rebuild(rep, raw, lock=LOCK):
    r, why = C7.rebuild(rep, raw, lock, check_counts=rep.get("repeat_rows") is not None)
    assert r is not None, why
    return r


# --------------------------------------------------------------------------
# R3 is rental 6's, byte for byte; the old rule reproduces every stored G1
# --------------------------------------------------------------------------

def test_r3_is_byte_identical_to_rental6s_registered_pin():
    want = {json.loads(f.read_text())["code_pins"]["scripts/private_weight_reference.py"]
            for f in REG.glob("2026-10-09-rental6-*-gh200.json")}
    got = hashlib.sha256((REPO / "scripts" / "private_weight_reference.py").read_bytes()).hexdigest()
    assert want == {got} and got.startswith("daa40706")
    assert not hasattr(PW, "G1_RULE_R7") and "--g1-drift-rule" not in PW.build_parser()._option_string_actions


def test_the_old_rule_rebuilt_from_cells_csv_gives_the_stored_g1_on_all_17_report_pages():
    assert len(PAGES) == 17
    got = []
    for name, rep, _raw, by in PAGES:
        stored = rep["histogram_gates"]["G1_lock_thermal"]
        g1 = PW.histogram_gates(rep["treads_table"], by)["G1_lock_thermal"]
        assert g1 == stored, name
        got.append(g1["verdict"])
    assert got.count("PASS") == 10 and got.count("FAIL") == 7


# --------------------------------------------------------------------------
# the build review's csv_rebuild, as a test (rental 6's 10 held pages)
# --------------------------------------------------------------------------

def _csv_r7(rows, lock, min_clean):
    """build-r7-review work/csv_rebuild.py's independent CSV-only implementation, verbatim in rule."""
    cells = collections.defaultdict(list)
    for x in rows:
        lab = re.search(r"histogram=([^;]+)", x["detail"]).group(1)
        cells[(lab, int(x["tiles"]), x["arm"])].append(x)
    out = {}
    for k, xs in cells.items():
        def dipped(v):
            return v < lock - 15.0
        clean = []
        for x in xs:
            reads = [float(v) for v in (x["clock_samples_mhz"] or "").split()]
            run = best = 0
            for v in reads:
                run = run + 1 if dipped(v) else 0
                best = max(best, run)
            med = float(x["sm_clock_load_mhz"]) if x["sm_clock_load_mhz"] else None
            slip = med is not None and dipped(med)
            ok = x["status"] == "ok" and float(x["ms_p50"]) > 0
            if ok and not slip and best < 3:
                clean.append(x)
        clk = [float(x["sm_clock_load_mhz"]) for x in clean if x["sm_clock_load_mhz"]]
        out[k] = dict(repeat_rows=len(xs), clean=len(clean),
                      ms=statistics.median(float(x["ms_p50"]) for x in clean) if clean else None,
                      clk=statistics.median(clk) if clk else None)
    rr = sum(v["repeat_rows"] for v in out.values())
    nc = rr - sum(v["clean"] for v in out.values())
    kept = {k: v for k, v in out.items() if v["clean"] >= min_clean}
    clocks = [v["clk"] for v in kept.values() if v["clk"] is not None]
    top = max(clocks) if clocks else None
    low = [k for k, v in kept.items() if v["clk"] is not None and v["clk"] < top - 15]
    unread = [k for k, v in kept.items() if v["clk"] is None]
    g1 = "PASS" if clocks and not low and not unread and nc <= 0.10 * rr + 1e-9 else "FAIL"
    return out, dict(rr=rr, nc=nc, g1=g1)


def test_the_scorers_rebuild_matches_the_reviews_csv_rebuild_on_rental6s_ten_held_pages():
    held = [(n, rep, raw) for n, rep, raw, _ in PAGES if rep.get("lock_mhz")]
    assert len(held) == 10
    for name, rep, raw in held:
        a, ga = _csv_r7(raw, rep["lock_mhz"], 6)
        r, why = C7.rebuild(rep, raw)
        assert r is not None, (name, why)
        for t in r["treads_table"]:
            k = (t["histogram"], t["tiles"], t["arm"])
            assert (t["clean_repeats"], t["ms_p50"], t["sm_clock_load_mhz"]) == (a[k]["clean"], a[k]["ms"], a[k]["clk"]), (name, k)
        g = r["g1_r7"]
        assert (g["verdict"], g["non_clean_rows"], g["repeat_rows"]) == (ga["g1"], ga["nc"], ga["rr"]), name


def test_a_page_whose_cells_csv_counts_differ_from_its_report_is_not_scored():
    name, rep, raw = next((n, r, w) for n, r, w, _ in PAGES if r.get("lock_mhz"))
    assert C7.rebuild(rep, raw)[0] is not None
    assert "repeats of" in C7.rebuild(rep, raw[1:])[1]
    bad = json.loads(json.dumps(rep))
    bad["repeat_rows"] += 1
    assert "repeat rows" in C7.rebuild(bad, raw)[1]
    assert C7.rebuild(rep, None)[0] is None and C7.rebuild(dict(rep, lock_mhz=None), raw)[0] is None


def test_the_rebuild_never_reads_the_stored_g1_or_the_stored_medians():
    name, rep, raw = next((n, r, w) for n, r, w, _ in PAGES if r.get("lock_mhz") and
                          r["histogram_gates"]["G1_lock_thermal"]["verdict"] == "FAIL")
    lied = json.loads(json.dumps(rep))
    for t in lied["treads_table"]:
        t["ms_p50"], t["sm_clock_load_mhz"], t["clean_repeats"] = 123.0, 1000.0, 0
    a, b = _rebuild(rep, raw), _rebuild(lied, raw)
    assert a["g1_r7"] == b["g1_r7"] and a["treads_table"] == b["treads_table"]
    assert a["g1_r7"]["verdict"] == "PASS" and a["histogram_gates"]["G1_lock_thermal"]["verdict"] == "PASS"
    assert [g["verdict"] for g in a["gates"] if g["tag"] == "G1_lock_thermal"] == ["PASS"]


# --------------------------------------------------------------------------
# r7-median-run3 on the held pages
# --------------------------------------------------------------------------

def test_r7_passes_all_17_held_pages_and_loses_no_cell():
    worst = 0.0
    for name, rep, raw, _ in PAGES:
        r = _rebuild(rep, raw)
        assert r["g1_r7"]["verdict"] == "PASS" and r["g1_p20"]["verdict"] == "PASS", name
        assert not r["g1_r7"]["lost_cells"], name
        worst = max(worst, r["g1_r7"]["share"])
    assert worst <= 0.005


def test_r7_keeps_the_flagged_repeats_whose_times_match_their_cells():
    flagged = moved = 0
    worst = 0.0
    for _name, rep, raw, _ in PAGES:
        for t in _rebuild(rep, raw)["treads_table"]:
            flagged += t["drift_flag_repeats"]
            if t["ms_p50_p20"] and t["ms_p50"] and t["drift_flag_repeats"]:
                d = t["ms_p50"] / t["ms_p50_p20"] - 1
                moved += d != 0
                worst = max(worst, abs(d))
    assert flagged == 72 and moved > 0 and worst < 0.0006


def test_the_p20_view_reads_its_own_clocks_and_lost_cells():
    """The review's minor fix: the P20 view's clock clause reads the P20 kept cells and the P20
    medians, not r7's. Phi skc loses 2 cells under P20 (4 or more flags) and none under r7."""
    name, rep, raw = next((n, r, w) for n, r, w, _ in PAGES if n.endswith("phi-3.5-moe-skc"))
    r = _rebuild(rep, raw)
    assert len(r["g1_p20"]["lost_cells"]) == 2 and not r["g1_r7"]["lost_cells"]
    for t in r["treads_table"]:
        assert t["lost_p20"] == (t["clean_repeats_p20"] < 6)
    rows = json.loads(json.dumps(r["treads_table"]))
    k = next(t for t in rows if not t["lost_p20"])
    k["sm_clock_load_mhz_p20"] = 1650.0           # one P20 clock off the top: the view fails, r7 does not
    for t in rows:
        t["mhz"], t["p20_mhz"] = t["sm_clock_load_mhz"], t["sm_clock_load_mhz_p20"]
        t["p20_clean_repeats"], t["p20_lost"] = t["clean_repeats_p20"], t["lost_p20"]
    assert C7.g1_rule(rows, prefix="p20_", cap=0.2)["verdict"] == "FAIL" and C7.g1_rule(rows)["verdict"] == "PASS"


# --------------------------------------------------------------------------
# sags injected into the held pages' cells.csv (the positive control)
# --------------------------------------------------------------------------

def _hit(x, every):
    return int(x["repeat"]) % every == 0 and len((x.get("clock_samples_mhz") or "").split()) == 15


@pytest.mark.parametrize("where, positions", [
    ("endpoint, last 5 reads", range(10, 15)),
    ("endpoint, first 3 reads", range(0, 3)),
    ("interior, reads 4 to 10", range(4, 11)),
    ("interior, reads 6 to 8", range(6, 9)),
])
def test_an_injected_sag_is_caught_on_every_held_page(where, positions):
    """A sag of 3 or more consecutive reads in a third of the repeats: every sagged repeat is
    run-guarded, every kept cell median is exactly the median of the page's unsagged repeats, and
    the page FAILS G1 on its non-clean share. The slip predicate alone keeps at least 80% of the
    sags of 5 reads or fewer (they leave the repeat's median read on the lock)."""
    import reg7 as R7
    positions = list(positions)
    for name, rep, raw, _ in PAGES:
        sag, n = R7.sag_rows(raw, positions, 3)
        assert n, name
        reps = [next(iter(C7.repeat_rows([x]).values()))[0] for x in sag if _hit(x, 3)]
        assert all(not C7.clean_r7(y, LOCK) for y in reps), (name, where)
        rows = _rebuild(rep, sag)
        kept = [x for x in raw if not _hit(x, 3)]
        by = C7.repeat_rows(kept)
        for t in rows["treads_table"]:
            reps0 = [y for y in by.get((t["histogram"], t["tiles"], t["arm"]), []) if C7.clean_r7(y, LOCK)]
            assert t["ms_p50"] == (statistics.median(y["ms"] for y in reps0) if reps0 else None), (name, where)
        assert rows["g1_r7"]["verdict"] == "FAIL" and rows["g1_r7"]["share_verdict"] == "FAIL", (name, where)
        if len(positions) <= 5:
            assert sum(y["status"] == "ok" and not C7.lock_slip(y, LOCK) for y in reps) >= 0.8 * n, (name, where)


def test_a_sag_in_one_repeat_of_every_cell_is_out_of_every_median_and_fails_g1():
    import reg7 as R7
    for name, rep, raw, _ in PAGES:
        sag, _n = R7.sag_rows(raw, list(range(5, 10)), 99)
        r = _rebuild(rep, sag)
        assert not r["g1_r7"]["lost_cells"] and r["g1_r7"]["share"] >= 1 / 9 - 1e-6, name
        assert r["g1_r7"]["verdict"] == "FAIL", name


def test_a_board_that_sags_three_reads_in_every_repeat_loses_every_cell_and_fails_g1():
    import reg7 as R7
    for name, rep, raw, _ in PAGES:
        sag, _n = R7.sag_rows(raw, [3, 4, 5], 1)
        r = _rebuild(rep, sag)
        assert all(t["lost"] for t in r["treads_table"]) and r["g1_r7"]["verdict"] == "FAIL", name


def _rep(reads, mhz=1710.0, drift_ok=True, status="ok", ms=1.0):
    return {"repeat": 0, "status": status, "ms": ms, "mhz": mhz, "reads": list(reads), "drift_ok": drift_ok,
            "host_bound": False}


def test_isolated_dips_are_kept_and_two_consecutive_are_not_a_run():
    lock = [1710.0] * 15
    one_end = [1600.0] + lock[1:]
    two = lock[:6] + [1600.0, 1650.0] + lock[8:]
    three = lock[:6] + [1600.0, 1690.0, 1650.0] + lock[9:]
    spaced = [1600.0 if i % 2 else 1710.0 for i in range(15)]
    assert [C7.dip_run(_rep(r), LOCK) for r in (lock, one_end, two, three, spaced)] == [0, 1, 2, 3, 1]
    for r in (lock, one_end, two, spaced):
        assert C7.clean_r7(_rep(r), LOCK)
    assert not C7.clean_r7(_rep(three), LOCK)
    # 1695 is one step under 1710 exactly: not a dip; 1694 is
    assert C7.dip_run(_rep(lock[:5] + [1695.0] * 5 + lock[10:]), LOCK) == 0
    assert C7.dip_run(_rep(lock[:5] + [1694.0] * 5 + lock[10:]), LOCK) == 5
    # the endpoint flag excludes in the P20 view only
    assert C7.clean_r7(_rep(one_end, drift_ok=False), LOCK) and not C7.clean_p20(_rep(one_end, drift_ok=False), LOCK)
    assert not C7.clean_r7(_rep(lock, mhz=1650.0), LOCK) and not C7.clean_r7(_rep(lock, status="failed"), LOCK)
    assert C7.dip_run(_rep([]), LOCK) == 0
    lo, hi = C7.READ_PERIOD_MS
    assert (lo, hi) == (140, 166) and 0.4 < 3 * (lo + hi) / 2 / 1000 < 0.5


@pytest.mark.parametrize("nc, ok", [(10, "PASS"), (11, "FAIL")])
def test_the_ten_percent_cap_is_at_the_edge(nc, ok):
    rows = [{"histogram": "PT", "tiles": 2, "arm": "shared", "repeat_rows": 100, "clean_repeats": 100 - nc, "lost": False,
             "mhz": 1710.0}]
    assert C7.g1_rule(rows)["verdict"] == ok


# --------------------------------------------------------------------------
# R3 at --min-clean-repeats 0 times every repeat (the review's data-loss fix)
# --------------------------------------------------------------------------

def test_r3_at_min_clean_zero_times_every_repeat_and_the_scorer_applies_six_of_nine():
    """Four endpoint flags in a cell: R3 at min-clean 6 (rental 6) stops timing it after repeat 4;
    at min-clean 0 it times all 9, and the scorer's r7 keeps all 9 (the flag excludes nothing)."""
    order = [T6._Cell("PT", 2)]

    def run_cell(c, rep, rot):
        return [PW.replace(T6._sample(a, rep, c.n, 1710.0), clock_drift_ok=(rep >= 4)) for a in rot]
    out, lost, early = PW.histogram_schedule(order, lambda c: ("native", "shared"), 9, run_cell,
                                             lock_mhz=LOCK, min_clean=6, say=lambda m: None)
    assert lost and all(len(v) < 9 for v in out.values())
    out, lost, early = PW.histogram_schedule(order, lambda c: ("native", "shared"), 9, run_cell,
                                             lock_mhz=LOCK, min_clean=0, say=lambda m: None)
    assert lost == {} and early is None and all(len(v) == 9 for v in out.values())


def test_a_planted_page_at_min_clean_zero_rebuilds_from_its_own_cells_csv(monkeypatch, tmp_path):
    """R3 (rental 6's, unchanged) at --slip-policy cell --min-clean-repeats 0 on a planted card writes
    cells.csv and report.json; its own G1 FAILS on one endpoint flag, the scorer's rebuild PASSES,
    keeps the flagged repeat, drops the run-guarded one, and reads the P20 view beside."""
    def fake_run(args, cfg, *, page, cells, arms, seed, copies, pinned, block_m, store, prov,
                 cache_root, declared_by_arm, lock_mhz=None, min_clean=0):
        bc = {(c.label, c.n): (list(c.counts) if c.counts is not None else [0]) for c in page.cells}

        def run_cell(c, rep, rot):
            got = []
            for a in rot:
                reads = [1710.0] * 15
                flagged = (c.label, c.n, a) == ("PW", 3, "shared") and rep == 2
                if flagged:
                    reads[-1] = 1600.0
                if (c.label, c.n, a) == ("PW", 3, "shared") and rep == 5:
                    reads[4:8] = [1600.0] * 4
                s = PW.replace(T6._sample(a, rep, c.n, statistics.median(reads), ms=1.0 + 0.01 * rep,
                                          clock_samples_mhz=" ".join(f"{v:.0f}" for v in reads)),
                               detail=f"histogram={c.label}", clock_drift_ok=not flagged)
                store.append(s, None)
                got.append(s)
            return got
        o, lost, early = PW.histogram_schedule(page.ordered(), lambda c: PW.cell_arms(page, c, arms),
                                               args.repeats, run_cell, lock_mhz=lock_mhz,
                                               min_clean=min_clean, say=lambda m: None)
        return o, bc, lost, early
    real = monkeypatch.setattr

    def hook(obj, name, value, *a, **k):
        return real(obj, name, fake_run if (obj is PW and name == "run_histogram_page") else value, *a, **k)
    monkeypatch.setattr = hook
    try:
        rc, rep, _ = T6._planted_page(monkeypatch, tmp_path, T6._argv(
            H6 / "mixtral-8x7b-A.json", "--slip-policy", "cell", "--lock-mhz", "1710", "--min-clean-repeats", "0"),
            lambda *a: 1710.0)
    finally:
        monkeypatch.setattr = real
    assert rep["min_clean_repeats"] == 0 and rep["histogram_gates"]["G1_lock_thermal"]["verdict"] == "FAIL"
    assert rc == exit_codes.INVALID           # R3's own r5-any G1; a held page, which locked_r3 keeps
    d = next(tmp_path.rglob("report.json")).parent
    _r, raw = C7.read_page(d)
    r, why = C7.rebuild(rep, raw)
    assert r is not None, why
    t = next(x for x in r["treads_table"] if (x["histogram"], x["tiles"], x["arm"]) == ("PW", 3, "shared"))
    assert (t["repeat_rows"], t["clean_repeats"], t["run_guard_repeats"], t["drift_flag_repeats"]) == (9, 8, 1, 1)
    assert t["ms_p50"] == pytest.approx(statistics.median(1.0 + 0.01 * k for k in range(9) if k != 5))
    assert t["ms_p50_p20"] == pytest.approx(statistics.median(1.0 + 0.01 * k for k in range(9) if k != 2))
    assert r["g1_r7"]["verdict"] == "PASS" and r["g1_r7"]["non_clean_rows"] == 1 and r["g1_p20"]["non_clean_rows"] == 1


def test_locked_r3_takes_min_clean_zero_with_the_cell_policy_and_hands_it_to_r3(tmp_path, monkeypatch):
    import locked_r3 as LR
    import test_locked_r3 as TL
    w = TL.World(tmp_path, monkeypatch)
    argv = w.argv("--slip-policy", "cell", "--min-clean-repeats", "0", locks=(1710,), groups=(8,))
    page = str(H6 / "mixtral-8x7b-A.json")
    run = LR.prepare([*argv, "--", "--histogram", page, "--model", "mixtral-8x7b"])
    a = run.r3_argv(1710, 8)
    assert a[a.index("--min-clean-repeats") + 1] == "0" and a[a.index("--slip-policy") + 1] == "cell"
    args = PW.build_parser().parse_args(a[2:])
    assert (args.min_clean_repeats, args.slip_policy, args.lock_mhz) == (0, "cell", 1710)


# --------------------------------------------------------------------------
# rental 6's scorers read the published session exactly as committed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("part", ["skew", "q1", "secondk"])
def test_the_rental6_scorers_still_give_their_committed_outputs_byte_for_byte(tmp_path, part):
    import subprocess
    tree = PUB / "2026-10-13-nvidia_gh200_480gb-rental6-session"
    got = subprocess.run([sys.executable, str(REPO / "scripts" / "scoring" / "rental6" / f"score_{part}.py"),
                          str(REPO), str(tree), str(tmp_path)], capture_output=True, text=True, timeout=600,
                         cwd=REPO, env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert got.returncode == 0, got.stderr[-2000:]
    for ext in ("json", "txt"):
        want = (REPO / "scripts" / "scoring" / "rental6" / f"{part}.score.{ext}").read_bytes()
        assert (tmp_path / f"{part}.score.{ext}").read_bytes() == want, (part, ext)
