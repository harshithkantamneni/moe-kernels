"""Rental 6's timed-page plumbing off the GPU (design-r6 S1-S3, design-r6-review (d)):
R3's per-repeat lock slip, the clean-repeat cell median, lost cells and the early stop, the
alignment probe on a histogram page, and locked_r3's --slip-policy cell with its end state
HELD_WITH_SLIPS. Page policy (the default) must be what it was."""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import locked_r3 as LR  # noqa: E402
import private_weight_reference as PW  # noqa: E402
import test_locked_r3 as TL  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

H6 = REPO / "docs" / "registered" / "2026-10-09-rental6-skew-hist"
H5 = REPO / "docs" / "registered" / "2026-10-07-rental5-skew-hist"
LOCK = 1710


class _Cell:
    def __init__(self, label, n):
        self.label, self.n, self.counts = label, n, None


def _sample(arm, rep, n, mhz, ms=1.0, **kw):
    return PW.Sample(arm=arm, repeat=rep, block_m=32, tiles=n, rows_per_expert=32 * n, tokens=64 * n,
                     copies=1, experts_declared=8 if arm == "native" else 72, ms_p50=ms,
                     sm_clock_load_mhz=mhz, clock_drift_ok=True, **kw)


def _run(order, repeats, clock, ms=None, **kw):
    """histogram_schedule with a planted card: `clock(label, n, arm, rep)` is each repeat's
    under-load clock; returns (samples, lost, early, the calls run_cell received)."""
    calls = []

    def run_cell(c, rep, rot):
        calls.append((rep, c.label, c.n, tuple(rot)))
        return [_sample(a, rep, c.n, clock(c.label, c.n, a, rep),
                        ms=(ms(c.label, c.n, a, rep) if ms else 1.0)) for a in rot]
    out, lost, early = PW.histogram_schedule(order, lambda c: ("native", "shared"), repeats, run_cell,
                                             say=lambda m: None, **kw)
    return out, lost, early, calls


# --------------------------------------------------------------------------
# R3: the slip predicate, the loop, the median, the early stop
# --------------------------------------------------------------------------

def test_the_slip_predicate_is_locked_r3s_off_lock_on_the_same_field():
    for mhz in (1710.0, 1695.0, 1694.9, 1650.0, 1725.0, None):
        s = _sample("shared", 0, 2, mhz)
        want = mhz is not None and LR.off_lock([mhz], LOCK)[0] is not None
        assert PW.lock_slip(s, LOCK) == want, mhz
        assert not PW.lock_slip(s, None)
    assert PW.LOCK_SLIP_STEP_MHZ == LR.STEP_MHZ
    bad = _sample("shared", 0, 2, 1710.0)
    bad = PW.replace(bad, clock_drift_ok=False)
    assert not PW.clean_repeat(bad, LOCK) and PW.clean_repeat(_sample("shared", 0, 2, 1700.0), LOCK)


def test_without_a_lock_the_loop_is_rental5s_order_exactly():
    order = [_Cell("PT", 2), _Cell("uniform", 2), _Cell("balanced", 4)]
    _out, lost, early, calls = _run(order, 3, lambda *a: 1650.0)   # a slipped clock changes nothing
    want = []
    for rep in range(3):
        for c in (order if rep % 2 == 0 else list(reversed(order))):
            ca = ("native", "shared")
            want.append((rep, c.label, c.n, tuple(ca[(i + rep) % 2] for i in range(2))))
    assert calls == want and lost == {} and early is None


def test_one_slipped_repeat_is_kept_flagged_and_out_of_the_cell_median():
    """A stub clock that slips one repeat (rep 4 of 9, 1650 MHz, and slow): the cell keeps all
    nine samples, eight are clean, and the median is over the eight."""
    order = [_Cell("PW", 3)]
    clock = lambda lab, n, a, rep: 1650.0 if (a == "shared" and rep == 4) else 1710.0
    ms = lambda lab, n, a, rep: 9.0 if (a == "shared" and rep == 4) else 1.0 + 0.01 * rep
    out, lost, early, _calls = _run(order, 9, clock, ms, lock_mhz=LOCK, min_clean=6)
    mine = out[("PW", 3, "shared")]
    assert len(mine) == 9 and lost == {} and early is None
    clean = [s for s in mine if PW.clean_repeat(s, LOCK)]
    assert len(clean) == 8 and all(s.repeat != 4 for s in clean)
    assert statistics.median(s.ms_p50 for s in clean) == pytest.approx(1.04)
    # the rental-5 predicate (usable only) would have read the slipped repeat in
    assert statistics.median(s.ms_p50 for s in mine if s.usable) == pytest.approx(1.05)


def test_a_cell_that_cannot_reach_six_clean_is_lost_and_not_timed_again():
    order = [_Cell("PT", 3), _Cell("uniform", 3)]
    clock = lambda lab, n, a, rep: 1640.0 if (lab == "PT" and a == "shared") else 1710.0
    out, lost, early, calls = _run(order, 9, clock, lock_mhz=LOCK, min_clean=6)
    # after repeat 3: 0 clean + 5 left < 6, so PT/shared is lost and runs 4 times only
    assert list(lost) == [("PT", 3, "shared")] and "0 clean of 4 repeats" in lost[("PT", 3, "shared")]
    assert len(out[("PT", 3, "shared")]) == 4 and len(out[("PT", 3, "native")]) == 9
    assert all(rot == ("native",) for rep, lab, _n, rot in calls if lab == "PT" and rep >= 4)
    assert early is None and len(out[("uniform", 3, "shared")]) == 9


def test_a_page_that_sags_all_the_way_stops_early():
    """Rental 5's Mixtral ska read 1635 MHz through: with no SIGINT it would burn the page's
    14 minutes. Every cell is lost after repeat 3, and the page stops there."""
    order = [_Cell("PT", 2), _Cell("balanced", 2)]
    out, lost, early, calls = _run(order, 9, lambda *a: 1635.0, lock_mhz=LOCK, min_clean=6)
    assert len(lost) == 4 and early and "after repeat 3 of 9" in early
    assert max(rep for rep, *_ in calls) == 3 and all(len(v) == 4 for v in out.values())


def test_min_clean_zero_records_slips_and_never_stops():
    out, lost, early, _ = _run([_Cell("PT", 2)], 9, lambda *a: 1635.0, lock_mhz=LOCK, min_clean=0)
    assert lost == {} and early is None and all(len(v) == 9 for v in out.values())


def test_the_gates_list_a_lost_cell_without_failing_g1_and_old_rows_score_as_before():
    base = {"arm": "shared", "tiles": 2, "histogram": "PT", "counts_sha256": "x", "bincount_ok": True,
            "sm_clock_load_mhz": 1710.0}
    old = PW.histogram_gates([dict(base)], {})
    assert "lock_slips" not in old["G1_lock_thermal"] and old["G1_lock_thermal"]["verdict"] == PW.PASS
    rows = [dict(base, lost=False, lock_slip_repeats=1, repeat_rows=9),
            dict(base, histogram="PW", sm_clock_load_mhz=None, lost=True, lock_slip_repeats=4,
                 repeat_rows=4)]
    g = PW.histogram_gates(rows, {})["G1_lock_thermal"]
    assert g["verdict"] == PW.PASS and g["clock_unread"] == []
    assert g["lock_slips"]["lost_cells"] == ["PW/shared/n2"] and g["lock_slips"]["slipped_repeats"] == 5
    assert g["lock_slips"]["repeat_rows"] == 13
    # a drifted repeat still fails G1: lock_slip is a flag of its own, not `excluded`
    drift = {("PT", 2, "shared"): [PW.replace(_sample("shared", 0, 2, 1710.0), clock_drift_ok=False)]}
    assert PW.histogram_gates(rows, drift)["G1_lock_thermal"]["verdict"] == PW.FAIL


# --------------------------------------------------------------------------
# R3: the histogram page end to end, the card planted
# --------------------------------------------------------------------------

def _planted_page(monkeypatch, tmp_path, argv, clock):
    """_histogram_mode with the GPU work planted: the alignment probe through the real
    probe_cells with fake op and timers, the timed loop through the real histogram_schedule."""
    probes = []

    def fake_probe(cfg, **kw):
        probes.append(kw)
        return PW.probe_cells(cfg, block_m=kw["block_m"], treads=kw["treads"],
                              declared_by_arm=kw["declared_by_arm"],
                              copies_declared=kw["copies_declared"], reference_clock=None,
                              repeats=kw["repeats"], calls_per_replay=0, op=lambda *a: None,
                              sync=lambda: None, graph_timer=None,
                              eager_timer=lambda call, **k: type("T", (), {"ms_p50": 0.004})(),
                              device="cpu", arms=kw["arms"])

    def fake_run(args, cfg, *, page, cells, arms, seed, copies, pinned, block_m, store, prov,
                 cache_root, declared_by_arm, lock_mhz=None, min_clean=0):
        bc = {(c.label, c.n): (list(c.counts) if c.counts is not None else [0]) for c in page.cells}

        def run_cell(c, rep, rot):
            return [_sample(a, rep, c.n, clock(c.label, c.n, a, rep)) for a in rot]
        out, lost, early = PW.histogram_schedule(page.ordered(), lambda c: PW.cell_arms(page, c, arms),
                                                 args.repeats, run_cell, lock_mhz=lock_mhz,
                                                 min_clean=min_clean, say=lambda m: None)
        return out, bc, lost, early
    monkeypatch.setattr(PW, "probe_alignment", fake_probe)
    monkeypatch.setattr(PW, "run_histogram_page", fake_run)
    monkeypatch.setattr(PW.SWEEP, "missing_gpu_stack", lambda: "")
    monkeypatch.setattr(PW.SWEEP, "reference_clock_mhz", lambda: (None, "planted"))
    monkeypatch.setattr(PW, "clock_sampler_refusal", lambda: "")
    monkeypatch.setattr(PW, "device_guard", lambda *a: "")
    monkeypatch.setattr(PW, "device_identity", lambda: "GPU-planted")
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    rc = PW.main(argv)
    reports = list(tmp_path.rglob("report.json"))
    return rc, (json.loads(reports[0].read_text()) if reports else None), probes


def _argv(page, *extra, copies=9, model="mixtral-8x7b"):
    return ["--model", model, "--block-m", "32", "--repeats", "9", "--duty", "0.25", "--seed", "0",
            "--declared-copies", str(copies), "--histogram", str(page), "--arms", "native,shared",
            "--group-m", "8", "--session-tag", "t6", "--ridge", "300", "--bandwidth-gbps", "3600",
            "--device-memory-gb", "90", *extra]


def test_a_histogram_page_under_cell_policy_writes_its_slips_and_its_own_probe(monkeypatch, tmp_path):
    page = H6 / "mixtral-8x7b-A.json"
    clock = lambda lab, n, a, rep: 1665.0 if (lab == "PW" and n == 3 and a == "shared" and rep in (2, 5)) else 1710.0
    rc, rep, probes = _planted_page(monkeypatch, tmp_path, _argv(page, "--slip-policy", "cell", "--lock-mhz",
                                                                 "1710", "--min-clean-repeats", "6"), clock)
    assert rc == exit_codes.DONE and rep is not None
    assert (rep["slip_policy"], rep["lock_mhz"], rep["min_clean_repeats"]) == ("cell", 1710, 6)
    assert rep["lock_slip_rows"] == 2 and rep["early_stop"] is None and rep["lost_cells"] == {}
    assert sorted(x["repeat"] for x in rep["slipped_repeats"]) == [2, 5]
    assert all(x["mhz"] == 1665.0 for x in rep["slipped_repeats"])
    row = next(r for r in rep["treads_table"] if r["histogram"] == "PW" and r["tiles"] == 3 and r["arm"] == "shared")
    assert (row["lock_slip_repeats"], row["clean_repeats"], row["repeats"], row["worst_repeat_mhz"]) == (2, 7, 7, 1665.0)
    assert row["sm_clock_load_mhz"] == 1710.0 and row["lost"] is False
    # G1 passes: a slip is not a drift exclusion (review (d)2)
    assert rep["histogram_gates"]["G1_lock_thermal"]["verdict"] == "PASS"
    # S3: the page carries its own probe, NATIVE and SHARED at each tread's numel
    ap = rep["align_probe"]
    assert ap is not None and {c["label"] for c in ap["cells"]} == {"native", "shared"}
    assert sorted({c["tread"] for c in ap["cells"]}) == [3, 6, 16]
    assert probes[0]["arms"] == ("native", "shared") and probes[0]["copies_declared"] == 9


def test_the_histogram_probe_is_the_balanced_ladders_probe_at_equal_numel(monkeypatch, tmp_path):
    """S3's test: the probe a histogram page records is the balanced ladder's own call, cell
    for cell (label, tread, numel, declaration), for the arms the page runs."""
    import moe.bench.timing  # noqa: F401  (probe_cells imports NotCapturable from it)
    rc, rep, probes = _planted_page(monkeypatch, tmp_path, _argv(H6 / "q1-qwen1.5-moe-a2.7b.json", copies=15,
                                                                 model="qwen1.5-moe-a2.7b"),
                                    lambda *a: 1710.0)
    assert rc == exit_codes.DONE
    cfg = PW.MODEL_CONFIGS["qwen1.5-moe-a2.7b"]
    decl = {a: PW.declared_experts(a, cfg.num_experts, 15) for a in PW.ARMS}
    seen = []
    ladder = PW.probe_cells(cfg, block_m=32, treads=[1, 2], declared_by_arm=decl, copies_declared=15,
                            reference_clock=None, repeats=PW.PROBE_REPEATS, calls_per_replay=0,
                            op=lambda ids, bm, d: seen.append((int(ids.numel()), d)), sync=lambda: None,
                            graph_timer=None,
                            eager_timer=lambda call, **k: (call(), type("T", (), {"ms_p50": 0.004})())[1],
                            device="cpu")
    key = lambda c: (c["label"], c["tread"], c["numel"], c["declared"], c["repeat"])
    got = sorted(key(c) for c in rep["align_probe"]["cells"])
    want = sorted(key(c) for c in ladder.as_dict()["cells"] if c["label"] in ("native", "shared"))
    assert got == want and got
    # and the page's numel is the histogram cell's: tokens(n) x top_k
    assert {(c[1], c[2]) for c in got} == {(n, PW.SWEEP.tokens_for_rows(cfg, n * 32) * cfg.top_k) for n in (1, 2)}
    # no slip policy: the page's report carries no slip fields (page policy unchanged)
    assert "slip_policy" not in rep and "lost" not in rep["treads_table"][0]


def test_a_page_sagging_through_stops_early_and_still_writes_its_report(monkeypatch, tmp_path):
    rc, rep, _ = _planted_page(monkeypatch, tmp_path, _argv(H6 / "q1-jetmoe-8b.json", "--slip-policy", "cell",
                                                            "--lock-mhz", "1710", "--min-clean-repeats", "6",
                                                            model="jetmoe-8b"),
                               lambda *a: 1635.0)
    assert rep is not None and "after repeat 3 of 9" in rep["early_stop"]
    assert all(r["lost"] for r in rep["treads_table"]) and len(rep["lost_cells"]) == len(rep["treads_table"])
    assert rep["repeat_rows"] == 4 * len(rep["treads_table"]) == rep["lock_slip_rows"]
    assert rc == exit_codes.INVALID   # no clean cell: G1 reads no clock


@pytest.mark.parametrize("extra, why", [
    (("--slip-policy", "cell"), "needs --lock-mhz"),
    (("--lock-mhz", "1710"), "ride with --slip-policy cell"),
    (("--slip-policy", "cell", "--lock-mhz", "1710", "--min-clean-repeats", "10"), "0 to --repeats 9"),
])
def test_r3_refuses_an_incomplete_slip_policy(capsys, extra, why):
    assert PW.main(_argv(H6 / "mixtral-8x7b-A.json", *extra, "--dry-run")) == exit_codes.REFUSED
    assert why in capsys.readouterr().out


def test_the_balanced_ladder_refuses_the_slip_arguments(capsys):
    rc = PW.main(["--model", "mixtral-8x7b", "--block-m", "32", "--treads", "6", "--slip-policy", "cell",
                  "--lock-mhz", "1710", "--dry-run"])
    assert rc == exit_codes.REFUSED and "ride with --histogram" in capsys.readouterr().out


def test_r3_accepts_every_rental6_timed_page_with_the_slip_arguments(capsys, monkeypatch, tmp_path):
    """Every timed page of the plan (PW2 labels and n 12 / 16 at 9 copies included) passes R3's
    dry run with the slip arguments locked_r3 hands it; only --dry-run refuses."""
    import re
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    plan = (REPO / "scripts" / "plans" / "rental6-2026-10.plan").read_text()
    units = re.findall(r"^(\S+) timed .*histogram=(\S+) .*declared-copies=(\d+)", plan, re.M)
    assert len(units) == 15
    for model, page, copies in units:
        rc = PW.main(_argv(REPO / page, "--slip-policy", "cell", "--lock-mhz", "1710", "--min-clean-repeats", "6",
                           "--dry-run", copies=int(copies), model=model))
        out = capsys.readouterr().out
        assert rc == exit_codes.REFUSED and "REFUSED:" not in out and "reason: --dry-run was given" in out, (page, out[-800:])
        assert "slips       --slip-policy cell at the 1710 MHz lock" in out


# --------------------------------------------------------------------------
# locked_r3: --slip-policy cell and HELD_WITH_SLIPS
# --------------------------------------------------------------------------

#: the planted R3 of test_locked_r3, with rows PLANT_SLIP_ROWS ({"<lock>:<G>": [row, ...]})
#: written at PLANT_SLIP_MHZ instead of the lock
SLIP_R3 = TL.PLANTED_R3.replace(
    '''        fh.write(f"shared,{i},{float(clock)},ok\\n")''',
    '''        slip = json.loads(os.environ.get("PLANT_SLIP_ROWS", "{}")).get(f"{lock}:{G}", [])
        v = float(os.environ.get("PLANT_SLIP_MHZ", "1650")) if i in slip else float(clock)
        fh.write(f"shared,{i},{v},ok\\n")''')
assert SLIP_R3 != TL.PLANTED_R3


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = TL.World(tmp_path, monkeypatch)
    w.r3.write_text(SLIP_R3)
    monkeypatch.delenv("PLANT_SLIP_ROWS", raising=False)
    return w


def _hist_args():
    return ["--", *LR.DEFAULT_R3_ARGS, "--histogram", "h.json"]


def test_cell_policy_records_the_slip_runs_on_and_keeps_the_page_held_with_slips(world):
    world.set(PLANT_SLIP_ROWS={"1710:2": [1, 3]}, PLANT_CELLS="5")
    rc = LR.main(world.argv("--slip-policy", "cell", locks=(1710,), groups=(1, 2)) + _hist_args())
    s = world.summary()
    assert rc == exit_codes.DONE, world.status()
    assert TL._attempts(s) == [(1710, 1, LR.HELD), (1710, 2, LR.HELD_WITH_SLIPS)]
    a2 = s["attempts"][1]
    assert [r["row"] for r in a2["slipped_cells"]] == [1, 3] and all(r["mhz"] == 1650.0 for r in a2["slipped_cells"])
    assert a2["cells"] == 5 and a2["slip_mhz"] is None, "R3 was stopped on the slip"
    assert s["lock"] == 1710 and [p["G"] for p in s["pages"]] == [1, 2]
    assert s["pages"][1]["state"] == LR.HELD_WITH_SLIPS
    args = json.loads((world.plant / f"args-{TL.TAG}-lock1710-g2").read_text())
    i = args.index("--slip-policy")
    assert args[i:i + 6] == ["--slip-policy", "cell", "--lock-mhz", "1710", "--min-clean-repeats", "6"]
    assert args[-4:] == ["--group-m", "2", "--session-tag", f"{TL.TAG}-lock1710"]
    assert "2 slipped rows recorded" in next(g for g in s["gates"] if g["number"] == "1" and g["kind"] == "CLAIM")["measured"]
    assert world.clock_verbs() == ["-lgc 1710", "-rgc"]


def test_each_attempt_hands_r3_its_own_lock_as_lock_mhz(world):
    """R3's slip predicate reads --lock-mhz: it must be the attempt's lock, not 1710. Every lock
    of a ladder builds its own argv, and a whole run at 1890 hands R3 1890 and flags rows by it."""
    run = LR.prepare(world.argv("--slip-policy", "cell", locks=(1980, 1890, 1800, 1710), groups=(8,)) + _hist_args())
    for lock in (1980, 1890, 1800, 1710):
        a = run.r3_argv(lock, 8)
        assert a[a.index("--lock-mhz") + 1] == str(lock), a
        assert a[-2:] == ["--session-tag", f"{TL.TAG}-lock{lock}"]
    world.set(PLANT_SLIP_ROWS={"1890:2": [2]}, PLANT_CELLS="4")
    rc = LR.main(world.argv("--slip-policy", "cell", locks=(1890,), groups=(1, 2)) + _hist_args())
    s = world.summary()
    assert rc == exit_codes.DONE, world.status()
    assert TL._attempts(s) == [(1890, 1, LR.HELD), (1890, 2, LR.HELD_WITH_SLIPS)]
    assert [r["row"] for r in s["attempts"][1]["slipped_cells"]] == [2]
    for G in (1, 2):
        args = json.loads((world.plant / f"args-{TL.TAG}-lock1890-g{G}").read_text())
        assert args[args.index("--lock-mhz") + 1] == "1890", args
    assert world.clock_verbs() == ["-lgc 1890", "-rgc"]


def test_page_policy_still_stops_on_the_first_slip_and_writes_the_old_summary(world):
    world.set(PLANT_SLIP_ROWS={"1710:1": [1]}, PLANT_CELLS="5")
    rc = LR.main(world.argv(locks=(1710,), groups=(1,)) + _hist_args())
    s = world.summary()
    assert rc == exit_codes.CLAIM_FAIL and TL._attempts(s) == [(1710, 1, LR.SLIPPED)]
    assert s["attempts"][0]["slip_mhz"] == 1650.0
    assert all("slipped_cells" not in a for a in s["attempts"])
    args = json.loads((world.plant / f"args-{TL.TAG}-lock1710-g1").read_text())
    assert "--slip-policy" not in args and "--lock-mhz" not in args


def test_cell_policy_still_stops_the_ladder_on_a_cell_over_the_lock(world):
    world.set(PLANT_SLIP_ROWS={"1710:1": [2]}, PLANT_SLIP_MHZ="1800", PLANT_CELLS="5")
    rc = LR.main(world.argv("--slip-policy", "cell", locks=(1710,), groups=(1,)) + _hist_args())
    assert rc == exit_codes.INVALID and TL._attempts(world.summary()) == [(1710, 1, LR.OFF_LOCK_HIGH)]


def test_cell_policy_needs_a_histogram_page_and_owns_the_slip_flags(world, capsys):
    assert LR.main(world.argv("--slip-policy", "cell")) == exit_codes.REFUSED
    assert "histogram page only" in capsys.readouterr().out
    rc = LR.main(world.argv() + ["--", *LR.DEFAULT_R3_ARGS, "--lock-mhz", "1710"])
    assert rc == exit_codes.REFUSED and "--lock-mhz" in capsys.readouterr().out
    for f in ("--slip-policy", "--lock-mhz", "--min-clean-repeats"):
        assert LR.owned_flag(f) == f


def test_the_dry_run_plan_names_the_slip_policy(world, capsys):
    LR.main(world.argv("--slip-policy", "cell", "--dry-run", locks=(1710,), groups=(8,)) + _hist_args())
    out = capsys.readouterr().out
    assert "slips       --slip-policy cell" in out and "--lock-mhz 1710 --min-clean-repeats 6" in out
    assert os.environ.get("PLANT_SLIP_ROWS") is None
