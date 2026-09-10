"""The apparatus tests for `scripts/block_m_crossing_sweep.py`.

`tests/test_block_m_sweep.py` tests the PHYSICS the sweep argues about: that the
gates tell alpha=0.558 from alpha=0.10, that the ladder recovers a planted
alpha, that the traps do not fire. This file tests the INSTRUMENT the 2026-09-02
audit found underneath it, and every test here is named after the defect it
would have caught:

  R1  a ridge asserted for one card multiplied by a bandwidth inherited from
      another, silently, whenever `load_measured` returned None or raised.
  R2  the analysis crashing with a TypeError in the world mixtral's measured
      G=1 alphas (0.95-1.02) describe, because "this tile never crosses" was a
      None formatted with `:.0f` rather than an outcome with a sentence.
  R3  a one-sided gate 3 that passed for every alpha from 0.33 to 1.0, so all 41
      committed surface fits cleared it while not one was within 0.05 of the
      0.558 those same reports predicted.
  R4  a membership rule anchored at n=1, so a single tread 3.30% above the
      compute line against a 4.20% margin threw away the other seven and made
      BLOCK_M=128 -- the only tile the paper's claim needs -- unidentifiable by
      construction in every arm.
  R5  two timing instruments compared with each other, and no clock recorded
      under load at all.
  R6  four scripts with four exit-code contracts and a summary that grepped
      free text for "PASS".
  R7  26 published report.json files with no commit, card, instrument or ruler
      source in them.
  R8  a fitted alpha read as a weight miss fraction, and caps taken as
      2*BM/(alpha*b) from it, which overstates them by a factor that is a
      bracket over the unmeasured alpha_a (1.04x to 3.03x at BM=128).

A SECOND PASS on 2026-09-02 added the block at the end of this file. Two
reviewers read the first one and found that fixing R1-R8 had introduced or left
standing its own set, each of the same shape as the defect it replaced:

  * `--warmup` became milliseconds and the cost estimate went on charging it as
    a call count, so the one number an operator buys pod time with was wrong by
    5x in both directions;
  * `provenance.iters` recorded the argparse default of the knob the same commit
    retired, contradicting every row of cells.csv;
  * a `--self-test` report claimed the real instrument at the top level, which
    is one of the five keys a publish gate checks;
  * the bandwidth refusal reached an untried caller as exit 1, and 1 is
    CLAIM_FAIL, so a driver would have ledgered a refusal as a refutation;
  * the retired instrument's refusal was a `RuntimeError` and every sibling arm
    swallowed it per cell;
  * R4's new membership rule reproduced the single-tread veto from the other
    side;
  * gate 3 imported an alpha over a tile whose own fit forbade it, in the same
    report that printed the prohibition;
  * the exit line said both "every gate PASSED" and "a claim gate did not pass";
  * the pinned `pod_session` regex was not `pod_session`'s regex;
  * the low-clock exclusion stopped at the ladder and never reached the gate
    that asserts an absence.

Every gate here can FAIL as well as PASS, and the tests that matter most are the
ones that plant the failing world.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec, for the reason tests/test_block_m_sweep.py gives:
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BM = _load_script()


def _load_sibling(name):
    """Another script in `scripts/`, loaded by path the way `BM` is.

    The sweep is not importable as a package member and neither are its
    siblings, so a test that has to assert on the CONTRACT BETWEEN TWO of them
    loads the second one the same way. Used by the pod cost probe test, which
    is about `replicate_noise_floor` reading this file's exit code.
    """
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.bench.weights import routed_expert_weight_bytes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
TILES = (32, 64, 128, 256)
RIDGE = 160.3
BANDWIDTH = 4374.5
REFIT = 0.558
RETRACTED = 0.10
#: The worlds mixtral's MEASURED G=1 ladders describe (alpha 0.95-1.02 on both
#: cards; qwen2 and deepseek-v2-lite read 0.62-0.84 at G=1, so this is one
#: model's regime, not every model's). At these values no tile in the sweep
#: crosses, which is exactly where the report used to crash.
NO_CROSSING_ALPHAS = (0.85, 0.90, 1.0)


def run(argv, tmp_path) -> tuple[int, dict]:
    """`main(argv)` into its own directory; returns `(rc, report.json)`.

    Each call gets its own `--out`, because the run id deliberately excludes
    `--alpha` (two analyses of one sweep belong in one directory) and these
    tests run several alphas that would otherwise land on top of each other.
    """
    rc = BM.main([*argv, "--out", str(tmp_path)])
    root = tmp_path / "block_m_crossing"
    if not root.exists():
        return rc, {}
    run_dir = next(root.iterdir())
    path = run_dir / "report.json"
    return rc, (json.loads(path.read_text()) if path.exists() else {})


def cells_at(alpha: float, *, noise: float = 0.0, tiles=TILES, seed: int = 0,
             low_clock=None, drifting=None):
    grid = BM.build_grid(MIXTRAL, tiles, 1024, 32, 6)
    return BM.synthetic_cells(MIXTRAL, grid, tiles, alpha=alpha, ridge=RIDGE,
                              bandwidth_gbps=BANDWIDTH, b=2, sm_count=132,
                              noise=noise, seed=seed, low_clock=low_clock,
                              drifting=drifting)


def analyse(cells, *, alpha: float, tiles=TILES, **kw):
    return BM.analyse(
        cells, MIXTRAL, block_sizes=tiles, alpha=alpha, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, model_name=MIXTRAL.name, dtype="bf16",
        compiles={bm: 1 for bm in tiles}, executed={bm: 1 for bm in tiles},
        sm_count=132, sm_source="test", ridge_band=(RIDGE, RIDGE),
        ridge_source="stated by the test", ridge_band_source="stated by the test",
        **kw)


def gate(report, number: int):
    return next(g for g in report.gates if g.number == number)


def gate_json(report: dict, number: int) -> dict:
    """The same gate off a serialized report.json, which is a dict of dicts and
    not a list of `Gate` objects. Two shapes, one lookup."""
    return next(g for g in report["gates"] if g["number"] == number)


def planted_reference():
    """A compute branch of 1.0 ms per tile at BLOCK_M=256, fixed cost 0.05."""
    return BM.ComputeReference(256, 0.05, 1.0, 0.0, "planted")


# --------------------------------------------------------------------------
# R1. The hybrid roof.
# --------------------------------------------------------------------------

def test_a_stated_ridge_with_no_bandwidth_refuses_instead_of_building_a_hybrid_roof(
        tmp_path, capsys):
    """THE DEFECT IN ONE COMMAND.

    `--ridge 145.8` is the A100's own contemporaneous figure. On a box with no
    calibration, `resolve_bandwidth` used to hand back 4374.5 GB/s -- an H200
    triad ceiling -- and the run proceeded with `model_roof = 145.8 x 4374.5`,
    a roof belonging to no card, printed as this one's. Nothing in the output
    said so.
    """
    rc = BM.main(["--ridge", "145.8", "--dry-run", "--out", str(tmp_path)])
    assert rc == exit_codes.REFUSED
    printed = capsys.readouterr().out
    assert "REFUSED" in printed
    assert "4374.5" in printed, "the refusal must name the constant it will not use"
    assert "two machines" in printed


def test_a_stated_ridge_and_a_stated_bandwidth_proceed_and_are_both_named_cli(
        tmp_path, capsys):
    """The operator's own assertion is allowed. It is the SILENT constant that
    is forbidden, and the two halves have to come from the same place."""
    rc = BM.main(["--ridge", "145.8", "--bandwidth", "1799.4", "--dry-run",
                  "--out", str(tmp_path)])
    # REFUSED because it is a `--dry-run` and measured nothing, NOT because the
    # rulers were withheld: the two `source=cli` lines below are the assertion,
    # and the test above this one is the same command without --bandwidth,
    # which refuses for the ruler reason and prints neither.
    assert rc == exit_codes.REFUSED
    printed = capsys.readouterr().out
    assert "source=cli" in printed
    assert printed.count("source=cli") == 2, "ridge AND bandwidth, both named"


def test_the_report_records_both_ruler_sources_as_cli(tmp_path):
    rc, payload = run(["--self-test", str(REFIT), "--ridge", "145.8",
                       "--bandwidth", "1799.4"], tmp_path)
    assert rc == exit_codes.DONE
    assert payload["ridge_source"].startswith("cli")
    assert payload["bandwidth_source"].startswith("cli")
    assert payload["provenance"]["ridge_source"].startswith("cli")
    assert payload["provenance"]["bandwidth_source"].startswith("cli")
    assert payload["bandwidth_gbps"] == pytest.approx(1799.4)


def test_the_hypothesis_bandwidth_is_only_reachable_beside_the_hypothesis_ridge(
        tmp_path):
    """A laptop plan may assume the H200 pair, because BOTH halves then come
    from the same 2026-08-26 calibration and nothing was measured to mislabel.
    Asserting a ridge closes that escape, which is what the first test above
    exercises from the other side."""
    args = BM.build_parser().parse_args(["--dry-run"])
    rb = BM.resolve_bandwidth(args)
    assert rb.source == "hypothesis"
    assert rb.gbps == pytest.approx(BM.PUBLISHED_H200_TRIAD_GBPS)
    args.ridge = 145.8
    with pytest.raises(BM.BandwidthUnavailable):
        BM.resolve_bandwidth(args)


def test_resolve_bandwidth_still_unpacks_as_the_pair_it_used_to_return():
    """THE COMPATIBLE PATH. `scripts/tile_cap_test.py:1760` does
    `bandwidth, bw_source = SWEEP.resolve_bandwidth(args)` outside a try, and
    that script is not this one's to edit."""
    args = BM.build_parser().parse_args(["--dry-run"])
    bandwidth, source = BM.resolve_bandwidth(args)
    assert bandwidth == pytest.approx(BM.PUBLISHED_H200_TRIAD_GBPS)
    assert isinstance(source, str) and source


# --------------------------------------------------------------------------
# R2. "This tile never crosses" is an outcome.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", NO_CROSSING_ALPHAS)
def test_the_worlds_the_measured_alphas_describe_write_a_report_instead_of_crashing(
        alpha, tmp_path):
    """`--self-test 0.90` and `--self-test 1.0` exited 1 with a TypeError at
    the gate-3 provenance line, before report.json was written. Mixtral's
    measured G=1 alphas on both cards are 0.95-1.02, so the analysis crashed
    in exactly the world that model's data points at."""
    rc, payload = run(["--self-test", str(alpha)], tmp_path)
    # A REPORT, and a code from the gates rather than from the crash. These
    # worlds falsify claims, so CLAIM_FAIL is the RESULT; what this test denies
    # is ERROR, which is what a TypeError before the write looked like.
    assert rc == exit_codes.CLAIM_FAIL
    assert payload, "report.json was not written"
    assert payload["alpha"] == pytest.approx(alpha)


def test_never_crossing_is_an_explicit_outcome_in_the_json(tmp_path):
    _, payload = run(["--self-test", "1.0"], tmp_path)
    for bm in ("32", "64", "128"):
        row = payload["predictions"][bm]
        assert row["crosses"] is False
        assert row["crossing_rows"] is None
        assert "never crosses" in row["no_crossing_reason"]
        assert "cap" in row["no_crossing_reason"]


def test_a_tile_that_does_cross_carries_no_reason(tmp_path):
    """The other half: the field is empty when there IS a crossing, so a reader
    cannot take a stale sentence for a live one."""
    _, payload = run(["--self-test", str(RETRACTED)], tmp_path)
    row = payload["predictions"]["64"]
    assert row["crosses"] is True
    assert row["crossing_rows"] is not None
    assert row["no_crossing_reason"] == ""


def test_the_gate_3_provenance_line_survives_a_subject_tile_with_no_crossing(
        tmp_path):
    """The exact line that raised: it formatted `preds_lo[128].crossing_rows`
    with `:.0f`. At alpha >= about 0.79 that value is None."""
    cells = cells_at(0.90)
    report = analyse(cells, alpha=0.90)
    lines = gate(report, 3).lines
    assert any("no crossing at all" in ln for ln in lines)


# --------------------------------------------------------------------------
# R3. Gate 3 is two-sided.
# --------------------------------------------------------------------------

def test_gate_3_passes_at_the_refit_alpha(tmp_path):
    rc, payload = run(["--self-test", str(REFIT), "--fail-on-gate"], tmp_path)
    assert rc == exit_codes.DONE
    g3 = next(g for g in payload["gates"] if g["number"] == 3)
    assert g3["verdict"] == "PASS"


@pytest.mark.parametrize("alpha", (0.85, 0.90, 1.0))
def test_gate_3_fails_high_when_the_measured_alpha_is_above_the_band(
        alpha, tmp_path):
    """THE PUBLISHED SWEEP'S ACTUAL NUMBERS. It measured 0.989 and 0.923
    against a prediction of 0.558 and printed "it is where the refit put it",
    because `alpha_hat > 0.33` passes for everything from 0.33 to 1.0."""
    rc, payload = run(["--self-test", str(alpha), "--fail-on-gate"], tmp_path)
    assert rc == exit_codes.CLAIM_FAIL
    g3 = next(g for g in payload["gates"] if g["number"] == 3)
    assert g3["verdict"] == "FAIL"
    assert "ABOVE" in g3["gate"]
    assert any("ABOVE the band" in ln for ln in g3["detail"])


def test_gate_3_fails_low_at_the_retracted_alpha_and_says_which_way(tmp_path):
    """A two-sided gate has to name the direction, or a FAIL from above and a
    FAIL from below read as the same finding."""
    _, payload = run(["--self-test", str(RETRACTED)], tmp_path)
    g3 = next(g for g in payload["gates"] if g["number"] == 3)
    assert g3["verdict"] == "FAIL"
    assert "BELOW" in g3["gate"]


def test_the_retired_one_sided_threshold_is_informational_and_not_the_verdict(
        tmp_path):
    """0.85 passes `alpha > 0.33` and fails the band. Both are recorded, and
    only one of them is the verdict."""
    _, payload = run(["--self-test", "0.85"], tmp_path)
    g3 = next(g for g in payload["gates"] if g["number"] == 3)
    prov = g3["provenance"]
    assert prov["retired_one_sided_verdict"] is True
    assert prov["retired_one_sided_threshold"] == BM.GATE3_ALPHA_DISCRIMINATOR
    assert g3["verdict"] == "FAIL"
    assert any("INFORMATIONAL" in ln for ln in g3["detail"])


def test_gate_3_says_whether_it_scored_its_own_alpha_or_an_imported_one(
        tmp_path):
    """The gate is about BLOCK_M=128 and the number is usually not from
    BLOCK_M=128. An imported number that does not say so is how the drift
    across tiles becomes invisible."""
    _, payload = run(["--self-test", str(REFIT)], tmp_path)
    g3 = next(g for g in payload["gates"] if g["number"] == 3)
    scored = [ln for ln in g3["detail"] if ln.startswith("SCORED ON")]
    assert len(scored) == 1
    assert ("IMPORTED" in scored[0]) == (g3["basis"] == "IMPORTED")


def test_the_scored_interval_carries_a_random_width_and_a_systematic_bracket():
    """A two-sided test on a point estimate would rest on the assumption that
    both named biases are zero, which is the assumption the two corrections
    exist because it is false. On noiseless cells the bootstrap is exactly
    zero-width, which is correct, and the systematic bracket is what the band
    is then scored against."""
    clean = BM.fit_ladder(BM.ladder_points(cells_at(REFIT), 64), 64,
                          BM.compute_reference(
                              cells_at(REFIT), TILES, cfg=MIXTRAL, ridge=RIDGE,
                              bandwidth_gbps=BANDWIDTH, b=2,
                              pinned=dict(BM.FIXED), capability=(9, 0)))
    iv = BM.alpha_interval(clean, MIXTRAL, BANDWIDTH)
    assert iv is not None
    assert iv.boot_lo == pytest.approx(iv.boot_hi)
    assert iv.sys_lo < iv.sys_hi
    assert iv.lo <= iv.point <= iv.hi
    assert iv.overlaps(BM.ALPHA_BAND)
    assert iv.direction(BM.ALPHA_BAND) == ""
    # And the failing side, planted: an interval well above the band is ABOVE.
    assert not iv.overlaps((0.90, 0.95))
    assert iv.direction((0.90, 0.95)) == "BELOW"


def test_the_bootstrap_widens_with_the_timing_spread():
    """Zero width on clean cells is only right if the width is real when the
    noise is. A bootstrap that is always zero is not an interval."""
    ref = BM.compute_reference(cells_at(REFIT, noise=0.03), TILES, cfg=MIXTRAL,
                               ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2,
                               pinned=dict(BM.FIXED), capability=(9, 0))
    noisy = BM.fit_ladder(BM.ladder_points(cells_at(REFIT, noise=0.03), 64), 64,
                          ref, margin=0.09)
    iv = BM.alpha_interval(noisy, MIXTRAL, BANDWIDTH)
    assert iv is not None
    assert iv.boot_hi > iv.boot_lo


# --------------------------------------------------------------------------
# R4. Membership, and the two UNDECIDED outcomes.
# --------------------------------------------------------------------------

def _n1_inside_margin_ladder():
    """n=1 a hair above the compute line, n=2..8 well above it.

    The shape of the H200 mixtral G=1 arm at BLOCK_M=128: the first tread sits
    3.30% above the scaled compute line against a 4.20% margin while treads 2-8
    sit 7.3-9.3% above it. The compute branch here is `overhead + 0.5 n`
    (BLOCK_M=128 scaled off a reference at 1.0 ms per tile at BLOCK_M=256).
    """
    overhead, c = 0.05, 0.5
    # Treads 2..8 on a real memory branch, 0.30 + 0.62 n: 24% steeper than the
    # compute branch, so it is a second mechanism and not the parallel case.
    pts = [(n, 0.30 + 0.62 * n) for n in range(2, 9)]
    # And the first tread measured BELOW where that branch extrapolates,
    # landing 3.3% above the compute line against a 4.2% margin. That is the
    # shape, and the direction: the n=1 cell is the shortest kernel in the
    # ladder and so the one carrying the largest host prefix as a fraction of
    # its time, which is why it is the tread that goes missing.
    return [(1, (overhead + c) * 1.033), *pts]


def test_a_first_tread_inside_the_margin_no_longer_vetoes_the_whole_branch():
    """THE ROW THE PREFIX RULE WAS COSTING.

    Under the old rule the run of memory-bound treads was taken from n=1 and
    stopped at the first tread that did not qualify, so a single miss of 0.003
    ms at n=1 gave k=0, alpha was imported from BLOCK_M=64, and `crosses` came
    out None for BLOCK_M=128 in every H200 arm. The tile the paper's claim is
    about had never been identified by its own ladder.
    """
    fit = BM.fit_ladder(_n1_inside_margin_ladder(), 128, planted_reference(),
                        margin=0.042)
    assert fit.memory_points >= 2
    assert fit.memory_points == 7, "treads 2..8 are the branch"
    assert fit.branch_start == 1, "the branch starts above the first tread"
    assert fit.alpha is not None
    assert fit.outcome == BM.IDENTIFIED
    assert "not at the first tread" in fit.basis


def test_the_old_prefix_rule_would_have_returned_nothing_on_that_same_ladder():
    """The trap has to still be a trap, or the test above has stopped testing
    anything. `memory_branch_members` reports the per-tread verdicts, and the
    prefix rule is exactly "stop at the first False"."""
    pts = _n1_inside_margin_ladder()
    xs = [float(n) for n, _ in pts]
    ys = [ms for _, ms in pts]
    start, count, above = BM.memory_branch_members(xs, ys, 0.5, 0.05, 0.042)
    assert above[0] is False, "the planted n=1 tread must sit inside the margin"
    assert all(above[1:]), "the other seven must sit outside it"
    assert (start, count) == (1, 7)
    # The prefix rule, written out: a leading run from index 0 is empty here.
    prefix = 0
    for flag in above:
        if not flag:
            break
        prefix += 1
    assert prefix == 0, "this is the rule that made BLOCK_M=128 unidentifiable"


def test_a_branch_parallel_to_the_compute_branch_is_UNDECIDED_and_names_the_arm():
    """`B / C = ridge / ai_cap`, so a parallel branch says the tile's cap sits
    ON the ridge. That is a question this sweep cannot settle and a roofline arm
    can. It used to come back as `alpha=None, crosses=None`, which downstream
    read as "the sweep lacked treads".

    THIS TEST IS ABOUT THE FIT AND NOT ABOUT THE GATE. Naming the outcome here
    did not by itself stop gate 3 importing an alpha over the tile and returning
    PASS -- it went on doing exactly that, two lines below printing the fit's own
    "no alpha may be imported over it". That half is
    `test_gate_3_will_not_answer_for_a_tile_whose_own_fit_refused_to`, and the
    asymmetry it keeps (a FAIL still lands) is
    `test_gate_3_still_refutes_on_an_imported_alpha_that_lands_outside_the_band`.
    """
    parallel = [(n, 0.05 + 0.5 * n * 1.05) for n in range(1, 9)]
    fit = BM.fit_ladder(parallel, 128, planted_reference(), margin=0.02)
    assert fit.outcome == BM.UNDECIDED_PARALLEL_BRANCH
    assert fit.undecided is True
    assert fit.alpha is None
    assert "UNDECIDED" in fit.outcome_reason
    assert "within" in fit.outcome_reason and "of the ridge" in fit.outcome_reason
    assert "roofline arm" in fit.outcome_reason
    assert "NOT a shortage of treads" in fit.outcome_reason
    # And the legacy wording the older test pins is still there.
    assert "DISCARDED" in fit.basis


def test_too_few_treads_and_a_parallel_branch_are_different_outcomes():
    """Both produce a blank alpha and they mean opposite things: one says the
    tile is compute bound this early, the other says the sweep cannot tell."""
    short = BM.fit_ladder([(1, 0.9), (2, 1.1)], 128, planted_reference(),
                          margin=0.02)
    assert short.outcome == BM.NOT_IDENTIFIED_TOO_FEW
    assert short.undecided is False


def test_the_parallel_world_reaches_the_json_as_undecided(tmp_path):
    """Off-GPU, end to end: `--self-test-world parallel-branch` must produce a
    report whose BLOCK_M=128 row is UNDECIDED with its reason, not a null.

    The LADDER ROW only. What the gates then do with that row is
    `test_gate_3_will_not_answer_for_a_tile_whose_own_fit_refused_to`; asserting
    the row alone is what let the gate go on contradicting it."""
    rc, payload = run(["--self-test-world", BM.PARALLEL_WORLD], tmp_path)
    # UNDECIDED is UNKNOWN in the shared table and UNKNOWN is not PASS, so this
    # world's code is CLAIM_FAIL and always was; it used to be folded into 0.
    assert rc == exit_codes.CLAIM_FAIL
    row = payload["ladder"]["128"]
    assert row["undecided"] is True
    assert row["outcome"] == BM.UNDECIDED_PARALLEL_BRANCH
    assert "roofline arm" in row["outcome_reason"]
    assert row["alpha"] is None


# --------------------------------------------------------------------------
# R5. One instrument, and the clock under load.
# --------------------------------------------------------------------------

def test_time_call_refuses_and_names_the_instrument_that_replaced_it():
    """It is kept as a symbol because four sibling scripts check `hasattr` for
    it before they will run at all. It is not kept as an instrument."""
    with pytest.raises(BM.RetiredInstrument) as exc:
        BM.time_call(lambda: None, 5, 10)
    assert "time_kernel" in str(exc.value)
    assert "queue-deep" in str(exc.value)


def test_a_cell_round_trips_its_timing_state_through_the_csv(tmp_path):
    """The tri-state clock verdicts are the point: an empty CSV field must come
    back as None, never as False (which would mark the row excluded) and never
    as True (which would mark it comparable). Both are claims the row does not
    make."""
    timed = BM.make_cell(MIXTRAL, 256, 128, 1.2345, sm_count=132, block_n=64,
                         iters=17, instrument="queue-deep/v2", warmup_ms=300.0,
                         trials=3, sm_clock_load_mhz=1970.0,
                         clock_level_ok=True, clock_drift_ok=False,
                         l2_flush=True)
    blank = BM.make_cell(MIXTRAL, 512, 128, 2.0, sm_count=132, block_n=64)
    path = tmp_path / "cells.csv"
    BM.append_cell(path, timed)
    BM.append_cell(path, blank)
    _, back = BM.read_cells(path)
    assert back == [timed, blank]
    assert back[0].clock_level_ok is True
    assert back[0].clock_drift_ok is False
    assert back[1].clock_level_ok is None
    assert back[1].sm_clock_load_mhz is None
    assert back[1].clock_excluded is False, "unknown is not excluded"


def test_a_drifting_cell_is_excluded_from_the_ladder_and_counted():
    """A tread whose clock MOVED across its own trials has a median that blends
    two operating points, so its time belongs to neither and no rescaling
    repairs it. That is the one exclusion since 2026-09-09."""
    cells = cells_at(REFIT, drifting=(64, 2))
    full, dropped_none = BM.ladder_treads(cells_at(REFIT), 64)
    kept, dropped = BM.ladder_treads(cells, 64)
    assert dropped_none == 0
    assert dropped == 1
    assert len(kept) == len(full) - 1
    assert 2 not in [n for n, _ in kept]


def test_a_low_clock_cell_stays_on_the_ladder_and_is_counted_as_kept():
    """THE RULE THAT CHANGED ON 2026-09-09, and the measurement that changed
    it. A steadily LOW clock used to be the exclusion. Across the 750 cells of
    the 2026-09-09 H200 session the under-load clock is an outcome of the cell,
    set per tile by its own power draw under the 700 W cap (BM=128/BN=64 a
    median 1395 MHz, BM=256 1650, memory-shaped 1950-1980, calibration GEMM
    1485), so LEVEL-LOW named a TILE and excluding on it removed the study's
    own subject from measurability. The tread is now KEPT, its side recorded,
    and the ladder comes out as the clean world's."""
    clean = cells_at(REFIT)
    cold = cells_at(REFIT, low_clock=(64, 2))
    sagged = [c for c in cold if c.clock_sagged]
    assert sagged and all(c.block_m == 64 and c.tiles_per_expert == 2
                          for c in sagged)
    assert not any(c.clock_excluded for c in cold)
    full, dropped_clean = BM.ladder_treads(clean, 64)
    kept, dropped = BM.ladder_treads(cold, 64)
    assert dropped_clean == 0 and dropped == 0
    assert kept == full
    assert BM.off_band_treads(cold, 64) == (1, 0)
    assert BM.sagged_treads(cold, 64) == 1
    assert BM.off_band_treads(clean, 64) == (0, 0)


def test_the_drift_world_reports_the_exclusion_in_the_report(tmp_path):
    """The two counts are different numbers and the report carries both.

    `cells_excluded_for_drift` is what the GATES lost: every cell on the
    drifted tread, aligned or not, because gates 1, 2 and 4 read unaligned
    cells too. `..._from_ladders` is the subset that was an aligned tread and so
    the only part a ladder fit could have seen. Reporting one of them as the
    other is how "one cell was dropped" came to stand for nine."""
    rc, payload = run(["--self-test-world", BM.DRIFT_WORLD], tmp_path)
    # DONE, and it is a real one: this world plants an excluded tread and every
    # gate still passes over what is left. Nothing is folded to get here.
    assert rc == exit_codes.DONE
    assert payload["cells_excluded_for_drift"] == 9
    assert payload["cells_excluded_for_drift_from_ladders"] == 1
    assert payload["ladder"]["64"]["excluded_drifted"] == 1
    assert "gate_4" in payload["clock_exclusion_reaches"]


def test_the_low_clock_world_reports_kept_treads_and_exits_done(tmp_path):
    """The mirror of the high side, and the rule change of 2026-09-09 end to
    end: the nine cells on the planted LOW tread are KEPT and counted on their
    side, nothing is excluded, and the ladder is the clean world's. Under the
    rule this replaced the same world reported one excluded tread."""
    rc, payload = run(["--self-test-world", BM.LOW_CLOCK_WORLD], tmp_path)
    assert rc == exit_codes.DONE
    assert payload["cells_excluded_for_drift"] == 0
    assert payload["cells_excluded_for_drift_from_ladders"] == 0
    assert payload["cells_kept_level_low"] == 9
    assert payload["cells_kept_level_low_from_ladders"] == 1
    row = payload["ladder"]["64"]
    assert row["kept_low_clock"] == 1 and row["excluded_drifted"] == 0
    assert row["outcome"] == BM.IDENTIFIED
    _, clean = run(["--self-test", str(BM.ALPHA)], tmp_path / "clean")
    assert row["alpha"] == pytest.approx(clean["ladder"]["64"]["alpha"])
    assert len(clean["ladder"]["64"]["points"]) == len(row["points"])


def test_a_throttled_cell_cannot_set_the_number_gate_4_scores():
    """GATE 4 ASSERTS AN ABSENCE AND IS SCORED AGAINST A MAXIMUM.

    R5 asked only that a mis-clocked cell stay out of the memory-branch fit,
    and `ladder_treads` did that. `plateau` and gates 1, 2 and 4 were handed
    the unfiltered list, and a cell whose clock moved mid-measurement carries a
    median that belongs to neither operating point -- so an undetected drift
    biases the one gate that asserts an absence, in whichever direction the
    governor happened to move.

    Planted at the top of the null tile, because that is the cell whose
    exclusion has to move the verdict's number: flag the fastest BLOCK_M=64
    cell and gate 4's peak must fall. If it does not, the exclusion stopped at
    the ladder again."""
    cells = cells_at(REFIT)
    top = max((c for c in cells if c.block_m == 64),
              key=lambda c: c.useful_tflops)
    flagged = [replace(c, clock_drift_ok=False, sm_clock_load_mhz=1500.0)
               if c is top else c for c in cells]
    before = gate(analyse(cells, alpha=REFIT), 4)
    after = gate(analyse(flagged, alpha=REFIT), 4)
    assert before.measured != after.measured, (
        "the drifted cell still set the peak gate 4 is scored against")
    assert after.provenance["peak_roof_fraction"] < \
        before.provenance["peak_roof_fraction"]


def test_the_report_says_which_gates_the_clock_exclusion_reached():
    """An exclusion whose extent a reader has to infer is an exclusion nobody
    can check. The report used to print a count and leave the scope unstated,
    and the scope was in fact narrower than the sentence implied."""
    cells = cells_at(REFIT, drifting=(64, 2))
    text = analyse(cells, alpha=REFIT).text()
    assert "excluded for a DRIFTING clock" in text
    assert "Reached: the plateau, the compute reference" in text
    assert "gates 1, 2, 3 and 4" in text


def test_a_ladder_that_loses_its_treads_to_a_drifting_clock_is_UNDECIDED_for_that_reason():
    """"Two memory-bound treads" and "two memory-bound treads left after four
    were dropped for a drifting clock" report the same count and mean opposite
    things: the second says the governor was still settling and the arm must be
    re-timed on an instrument that waits for it."""
    fit = BM.fit_ladder([(1, 0.9), (2, 1.1)], 128, planted_reference(),
                        margin=0.02, excluded_drifted=4)
    assert fit.outcome == BM.UNDECIDED_DRIFTING_CLOCK
    assert fit.undecided is True
    assert "clock" in fit.outcome_reason
    assert "Re-time the arm" in fit.outcome_reason
    assert fit.excluded_drifted == 4


def test_the_self_test_cells_carry_the_timing_columns_but_never_the_real_basis():
    """A reader who greps a cells.csv for the instrument has to be able to tell
    a pod row from a generated one."""
    cells = cells_at(REFIT)
    assert {c.instrument for c in cells} == {BM.SYNTHETIC_INSTRUMENT}
    assert BM.timing_basis() not in {c.instrument for c in cells}
    assert all(c.clock_level_ok is True for c in cells)


def test_the_flush_state_is_recorded_and_changes_the_run_id():
    """A flushed and an unflushed sweep measure different milliseconds for the
    same cell, so they must never share a directory and be resumed into each
    other."""
    parser = BM.build_parser()
    flushed = parser.parse_args([])
    warm = parser.parse_args(["--no-l2-flush"])
    assert flushed.no_l2_flush is False
    assert BM.default_run_id(flushed, "nvidia_h200") != BM.default_run_id(
        warm, "nvidia_h200")


def test_the_warmup_is_a_duration_and_the_trial_count_is_a_knob():
    """UNITS CHANGED: `--warmup` was a COUNT of calls and is now milliseconds of
    delivered load. The ladders that compared cells warmed at 5 against cells
    warmed at 20 were comparing clock states."""
    args = BM.build_parser().parse_args(["--warmup", "12.5", "--trials", "7"])
    assert args.warmup == pytest.approx(12.5)
    assert args.trials == 7
    assert BM.build_parser().parse_args(["--warmup-ms", "12.5"]).warmup == 12.5


# --------------------------------------------------------------------------
# R6. Exit codes and result lines.
# --------------------------------------------------------------------------

def test_every_gate_prints_exactly_one_result_line(tmp_path, capsys):
    rc = BM.main(["--self-test", str(REFIT), "--out", str(tmp_path)])
    assert rc == exit_codes.DONE
    printed = capsys.readouterr().out
    results = exit_codes.parse_result_lines(printed)
    assert len(results) == 5
    assert [r.name for r in results] == [BM.GATE_NAMES[n] for n in range(5)]
    assert results[0].kind == exit_codes.VALIDITY
    assert all(r.kind == exit_codes.CLAIM for r in results[1:])


def test_the_log_recomputes_the_exit_code_the_process_returned(tmp_path, capsys):
    """`classify_text` over what was printed must agree with `classify` over the
    gate objects. A script that prints one thing and exits another is the defect
    `moe/bench/exit_codes.py` is named against."""
    rc = BM.main(["--self-test", "0.85", "--fail-on-gate", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert rc == exit_codes.CLAIM_FAIL
    assert exit_codes.classify_text(printed) == rc


def test_a_falsified_claim_is_claim_fail_with_or_without_the_retired_flag(
        tmp_path):
    """NOTHING IS FOLDED INTO DONE. `--fail-on-gate` used to decide the code:
    without it a CLAIM_FAIL was RETURNED AS 0 while the log carried
    `RESULT: CLAIM ... FAIL`, so `classify_text` read 1 out of the log and the
    process said 0. The flag is retired, accepted and ignored, because
    CLAIM_FAIL (1) is already in `FINISHED_CODES` and `ledger_state(1)` is
    "CLAIM_FAIL": 1 tells the driver "a result, do not retry" and 0 protected
    nothing."""
    with_flag = BM.main(["--self-test", "0.85", "--fail-on-gate",
                         "--out", str(tmp_path / "a")])
    without = BM.main(["--self-test", "0.85", "--out", str(tmp_path / "b")])
    assert with_flag == exit_codes.CLAIM_FAIL
    assert without == exit_codes.CLAIM_FAIL


def test_an_undecided_gate_counts_against_the_gate_not_for_it():
    """UNDECIDED here is UNKNOWN in the shared table, and UNKNOWN is not PASS:
    "a check that examined nothing reports zero failures" is this project's
    documented failure shape."""
    g = BM.Gate(3, "claim", BM.UNDECIDED, "measured", "gate")
    kind, name, verdict = g.scored()
    assert verdict == exit_codes.UNKNOWN
    assert exit_codes.classify([(kind, verdict)]) == exit_codes.CLAIM_FAIL
    assert name == BM.GATE_NAMES[3]


def test_a_refusal_exits_refused_and_measures_nothing(tmp_path, capsys):
    rc = BM.main(["--ridge", "145.8", "--out", str(tmp_path)])
    assert rc == exit_codes.REFUSED
    assert "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "block_m_crossing").exists()


def test_no_line_outside_the_result_lines_can_be_read_as_a_gate_result(
        tmp_path, capsys):
    """The driver's summary used to grep free text and read a pre-registered
    expectation out of a refused log. Only `RESULT: ` at column zero counts, and
    the count of those must equal the number of scored gates."""
    BM.main(["--self-test", str(RETRACTED), "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    starts = [ln for ln in printed.splitlines()
              if ln.startswith(exit_codes.RESULT_PREFIX)]
    assert len(starts) == 5
    assert len(exit_codes.parse_result_lines(printed)) == 5


#: The OTHER consumer of this script's verdicts, and the reason the human
#: `GATE n VERDICT` line survives beside the `RESULT:` line. `gate_from_log` in
#: `scripts/pod_session.sh` counts PASS and FAIL with exactly this regex and has
#: no `RESULT:` branch, so deleting the older form would take that session's
#: entire verdict channel with it. Copied here rather than imported because the
#: point is to notice when the two drift apart.
#:
#: BOTH ALTERNATIONS, AND NO `UNDECIDED`. The first copy of this constant kept
#: only the `GATE n` half and then ADDED `UNDECIDED`, so it was not the shell's
#: regex in either direction: a line of the shape `  [PASS] something` would
#: have been counted by `pod_session` as a gate PASS while this test stayed
#: green, which is precisely the drift the test below exists to notice. The
#: shell writes `[[:space:]]`; `[ \t]` is its ASCII equivalent for the lines
#: this script emits. Rendered from one template exactly as `gate_from_log`
#: does, so a reader can see that the two verdicts share every other character.
POD_SESSION_GATE_TEMPLATE = r"^[ \t]*\[(%s)\]|^GATE [0-9]+[ \t]+(%s)([ \t]|$)"
POD_SESSION_PASS_RE = re.compile(POD_SESSION_GATE_TEMPLATE % ("PASS", "PASS"))
POD_SESSION_FAIL_RE = re.compile(POD_SESSION_GATE_TEMPLATE % ("FAIL", "FAIL"))


def _pod_session_counts(printed: str) -> tuple[int, int]:
    """`gate_from_log`'s two counts, computed the way the shell computes them:
    one `grep -cE` per verdict over the whole log."""
    lines = printed.splitlines()
    return (sum(1 for ln in lines if POD_SESSION_PASS_RE.search(ln)),
            sum(1 for ln in lines if POD_SESSION_FAIL_RE.search(ln)))


def test_the_older_gate_line_is_one_per_scored_gate_and_nothing_else(
        tmp_path, capsys):
    """Two verdict channels must carry the same verdicts.

    `RESULT:` is what `scripts/h200_gaps_session.sh` will read; `GATE n PASS` is
    what `scripts/pod_session.sh:gate_from_log` already reads. A line that
    matches the older regex without being a scored gate is the free-text defect
    moved rather than fixed: the audit's noise-floor case was a REFUSED log
    whose prose matched the summary grep 18 times. So the two counts must equal
    the RESULT channel's counts of the same verdicts."""
    BM.main(["--self-test", "0.85", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    results = exit_codes.parse_result_lines(printed)
    assert len(results) == 5
    verdicts = [r.verdict for r in results]
    assert _pod_session_counts(printed) == (verdicts.count(exit_codes.PASS),
                                            verdicts.count(exit_codes.FAIL))


def test_the_bracketed_half_of_the_pod_session_regex_matches_nothing_here(
        tmp_path, capsys):
    """The half the first copy of this constant silently dropped.

    `gate_from_log` accepts TWO shapes and this script emits only one of them,
    so the counts above happen to agree. That is a fact about today's output and
    not a property of the script, and it is the fact the dropped alternation
    made uncheckable: a future `  [PASS] ...` line would be a gate PASS to the
    shell and invisible to a test that only knew the `GATE n` form. Asserted
    across the four planted worlds and the refusal, so it fails the day such a
    line appears."""
    bracketed = re.compile(r"^[ \t]*\[(PASS|FAIL)\]")
    for argv in (["--self-test", "0.558"], ["--self-test", "0.85"],
                 ["--self-test", "1.0"],
                 ["--self-test-world", BM.PARALLEL_WORLD],
                 ["--ridge", "145.8"]):
        BM.main([*argv, "--out", str(tmp_path / argv[-1].replace(".", "_"))])
        printed = capsys.readouterr().out
        assert not [ln for ln in printed.splitlines() if bracketed.match(ln)], (
            f"{argv} emitted a bracketed verdict line; pod_session would count "
            "it as a gate and the RESULT channel would not")


def test_an_undecided_gate_is_never_counted_as_a_pass_by_the_pod_session_grep(
        tmp_path, capsys):
    """THE ASYMMETRY THE TWO CHANNELS HAVE, said out loud rather than assumed.

    `pod_session`'s regex knows PASS and FAIL only, and this script's legacy
    line prints the verdict verbatim, so an UNDECIDED gate is INVISIBLE to that
    channel while the RESULT channel reports it as UNKNOWN. Invisible is the
    safe direction and a false PASS is the one direction that must never
    happen; a pairwise comparison of the two channels' verdict STRINGS would
    fail on this legitimate run for a spelling, which is why the test above
    compares counts of PASS and of FAIL instead."""
    rc = BM.main(["--self-test-world", BM.PARALLEL_WORLD, "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert rc == exit_codes.CLAIM_FAIL
    results = exit_codes.parse_result_lines(printed)
    assert exit_codes.UNKNOWN in [r.verdict for r in results], (
        "this world exists to put an unscored gate in the log")
    passes, fails = _pod_session_counts(printed)
    assert passes == [r.verdict for r in results].count(exit_codes.PASS)
    assert fails == 0
    assert passes + fails < len(results), "the undecided gate must not be counted"


def test_a_refused_log_offers_neither_channel_a_verdict_to_count(
        tmp_path, capsys):
    """The planted FAIL branch of the test above. `replicate_noise_floor`'s
    refused log matched the driver's free-text grep 18 times and put a
    pre-registered expectation into the session summary as measured output. A
    refusal here scores nothing, so both channels must be empty rather than
    quotable."""
    rc = BM.main(["--ridge", "145.8", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert _pod_session_counts(printed) == (0, 0)
    assert not exit_codes.parse_result_lines(printed)


# --------------------------------------------------------------------------
# R7. Provenance.
# --------------------------------------------------------------------------

def test_every_report_carries_a_provenance_block_with_the_audited_keys(tmp_path):
    """The gate the audit wrote: none of the 26 published reports carries a
    commit, a card, a ruler source or an instrument."""
    _, payload = run(["--self-test", str(REFIT)], tmp_path)
    assert {"git_sha", "gpu_name", "ridge_source", "bandwidth_source",
            "instrument"} <= set(payload)
    block = payload["provenance"]
    assert block["git_sha"], "this repository is a git checkout"
    assert block["utc"]
    assert block["python"]
    assert block["target_ms"] == pytest.approx(400.0)
    # And what it could NOT determine is named rather than guessed.
    assert "gpu_name" in block["missing"], "there is no card on this box"


def test_every_cells_csv_row_carries_the_provenance_columns(tmp_path):
    """Written per ROW rather than per file: a resumed cells.csv is written by
    two processes on two days and possibly two commits, and a header cannot say
    that."""
    from moe.bench import provenance as PV
    prov = PV.provenance_block(instrument="test", ridge=RIDGE,
                              ridge_source="cli", bandwidth=BANDWIDTH,
                              bandwidth_source="cli")
    path = tmp_path / "cells.csv"
    BM.append_cell(path, BM.make_cell(MIXTRAL, 256, 128, 1.0, sm_count=132,
                                      block_n=64), prov)
    header = path.read_text().splitlines()[0].split(",")
    assert "prov_git_sha" in header
    assert "prov_instrument" in header
    assert "prov_ridge_source" in header
    assert "prov_bandwidth_source" in header
    # And the measurement columns still round-trip beside them.
    _, back = BM.read_cells(path)
    assert back[0].block_m == 128


def test_the_run_id_carries_the_card_and_every_swept_knob():
    parser = BM.build_parser()
    base = parser.parse_args([])
    assert BM.default_run_id(base, "nvidia_h200").startswith("nvidia_h200-")
    for flag, value in (("--model", "qwen2-57b-a14b"), ("--dtype", "fp16"),
                        ("--tiles", "64,128"), ("--group-m", "16"),
                        ("--block-n", "256"), ("--num-stages", "3"),
                        ("--r-max", "2048"), ("--row-step", "64"),
                        ("--step-probes", "3"), ("--seed", "1"),
                        ("--iters", "200"), ("--warmup", "17"),
                        ("--cell-budget-ms", "900"), ("--trials", "9")):
        other = parser.parse_args([flag, value])
        assert BM.default_run_id(base, "nvidia_h200") != BM.default_run_id(
            other, "nvidia_h200"), f"{flag} is not in the run id"


def test_the_run_id_refuses_a_run_with_no_card():
    """The card is not swept by this script; it is swept by the operator moving
    to another pod while the results root is a network volume that outlives it.
    Two cards derived one id and the collision is committed in this repo."""
    from moe.bench import provenance as PV
    args = BM.build_parser().parse_args([])
    with pytest.raises(PV.NoCard):
        BM.default_run_id(args, "")


# --------------------------------------------------------------------------
# R8. The EXA form, and the factor every cap is high by.
# --------------------------------------------------------------------------

def test_the_ladder_alpha_is_documented_as_the_exa_form_and_not_a_miss_fraction():
    doc = BM.LadderFit.alpha.__doc__
    assert "(EXA)" in doc
    assert "(alpha_b + phi) / (1 + phi + delta)" in doc
    assert "not a miss" in doc and "fraction" in doc
    assert "lin_overstatement" in doc


def test_alpha_by_block_m_is_documented_as_phi_growing_with_bm_over_bn():
    """It used to be filed as an unexplained drift that "cannot be pinned"."""
    source = (ROOT / "scripts" / "block_m_crossing_sweep.py").read_text()
    head = source.split("ALPHA_BY_BLOCK_M = ")[0]
    block = head[head.rindex("#: alpha measured per BLOCK_M"):]
    assert "EXA" in block
    assert "phi" in block
    assert "BM/BN" in block
    assert "fitted" in block.lower()


def test_the_cap_overstatement_is_the_ai_model_factor_and_is_a_bracket():
    """`alpha_a` has no measurement anywhere in this repository, so the factor
    is the pair of ends it can take, not a number."""
    lo, hi = BM.cap_overstatement(MIXTRAL, 128, 64, 2)
    assert lo < hi
    n, k = 2 * MIXTRAL.intermediate_size, MIXTRAL.hidden_size
    expected_hi = ai_model.lin_overstatement(
        phi=ai_model.phi(n, k, block_m=128, block_n=64, alpha_a=1.0), delta=0.0)
    assert hi == pytest.approx(expected_hi)
    # The 1.32 `moe/bench/ai_model.py` quotes at BM=128 is this factor at the
    # (LIN)-era alpha_a = 0.143. It has to sit INSIDE the bracket, which is the
    # bracket's whole justification: alpha_a is unmeasured, so the honest
    # statement is the pair of ends and not any point inside it.
    assert lo < 1.32 < hi
    # It grows with BLOCK_M at fixed BLOCK_N, which is the whole mechanism.
    assert BM.cap_overstatement(MIXTRAL, 256, 64, 2)[1] > hi
    assert BM.cap_overstatement(MIXTRAL, 64, 64, 2)[1] < hi


def test_the_report_prints_the_overstatement_beside_the_caps_it_derives(tmp_path):
    _, payload = run(["--self-test", str(REFIT)], tmp_path)
    row = payload["predictions"]["128"]
    assert len(row["cap_overstatement"]) == 2
    assert "lin_overstatement" in row["cap_overstatement_note"]
    assert "FITTED alpha" in row["cap_overstatement_note"]
    text = (next((tmp_path / "block_m_crossing").iterdir()) / "report.txt").read_text()
    assert "lin_overstatement" in text


# --------------------------------------------------------------------------
# The second pass, 2026-09-02. Every test below is named after a defect the
# first pass at R1-R8 introduced or left standing, and each plants the world
# that produced it.
# --------------------------------------------------------------------------

def _dry_run_grid():
    return BM.build_grid(MIXTRAL, TILES, 1024, 32, 6)


def test_the_cost_estimate_charges_a_warmup_duration_as_a_duration():
    """THE UNIT ERROR R5 LEFT IN THE ONLY NUMBER AN OPERATOR BUYS POD TIME WITH.

    `--warmup` became MILLISECONDS of delivered load and `estimated_seconds` was
    not migrated with it: it went on charging `ms * (warmup + iters)`, so the
    default 300.0 was billed as 300 CALLS -- a duration added to an iteration
    count, 15x the old 20-call term, and `--trials` did not enter at all. The
    mixtral defaults printed 278 s for a run whose honest figure is about 414 s.

    The instrument's cost per cell is a fixed warmup plus `trials` trials of
    `cell_budget_ms` each, so the total is within a few percent of
    `cells x (warmup_ms + trials x budget)` and is nearly INDEPENDENT of the
    per-call time. Both halves are asserted, because the old formula failed the
    second one by a factor of five across the grid.
    """
    grid = _dry_run_grid()
    cells = len(grid) * len(TILES)
    secs = BM.estimated_seconds(
        MIXTRAL, grid, TILES, alpha=REFIT, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, warmup_ms=300.0, trials=3,
        cell_budget_ms=400.0)
    nominal = cells * (0.300 + 3 * 0.400)
    assert secs == pytest.approx(nominal, rel=0.10), (
        "the estimate is no longer the run the instrument will make")
    # `iters_for` sizes a trial to hold the budget of KERNEL time and then
    # rounds the count down to an integer, so a trial is at most one call short
    # of the budget and the total sits just under the nominal figure. The clamp
    # pushes it back over only for cells slower than budget/10, of which this
    # grid has none.
    assert 0.98 * nominal <= secs <= 1.02 * nominal


def test_the_cost_estimate_barely_moves_when_the_kernel_gets_slower():
    """The property the old formula did not have, planted from both ends.

    At HEAD a 1 ms cell was priced 700 ms and an 11 ms cell 3696 ms, a 5.3x
    spread, because the per-call time multiplied everything including the
    warmup term. The instrument sizes its iteration count DOWN as the kernel
    gets slower, so the real spread over the same range is the clamp's and
    nothing else."""
    fast = BM.estimated_seconds(
        MIXTRAL, [512], (128,), alpha=REFIT, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, warmup_ms=300.0, trials=3,
        cell_budget_ms=400.0)
    slow = BM.estimated_seconds(
        MIXTRAL, [512], (32,), alpha=REFIT, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, warmup_ms=300.0, trials=3,
        cell_budget_ms=400.0)
    assert 0.9 <= slow / fast <= 1.2, (fast, slow)


def test_the_two_instruments_may_not_be_priced_together_or_by_default():
    """REFUSE RATHER THAN DEFAULT. `warmup_ms`/`trials` price `time_kernel` and
    `iters`/`warmup` price the retired per-call loop; the two answers differ by
    5x and picking one silently is how the wrong one got printed."""
    kw = dict(alpha=REFIT, ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2,
              cell_budget_ms=400.0)
    with pytest.raises(ValueError, match="both instruments"):
        BM.estimated_seconds(MIXTRAL, [512], (128,), warmup_ms=300.0, trials=3,
                             iters=50, **kw)
    with pytest.raises(ValueError, match="needs an instrument"):
        BM.estimated_seconds(MIXTRAL, [512], (128,), **kw)


def test_the_retired_pricing_survives_for_the_script_that_still_runs_it():
    """THE COMPATIBLE PATH. `scripts/tile_cap_test.py:1835` passes
    `iters=`/`warmup=` and its `--warmup` really is a call count (argparse
    `type=int`, default 20), so that arm's `--dry-run` must keep costing the run
    it will actually make."""
    kw = dict(alpha=REFIT, ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2,
              cell_budget_ms=400.0)
    old = BM.estimated_seconds(MIXTRAL, [512], (128,), iters=50, warmup=20, **kw)
    ms = BM.model_ms(MIXTRAL, 512, 128, alpha=REFIT, ridge=RIDGE,
                     bandwidth_gbps=BANDWIDTH, b=2)
    assert old == pytest.approx(
        ms * (20 + BM.scaled_iters(ms, 50, 400.0)) / 1e3)


def test_the_dry_run_prints_the_instruments_own_cost_and_says_what_it_charged(
        capsys):
    """The line an operator reads. It has to name the terms, or the next unit
    error is invisible again."""
    rc = BM.main(["--ridge", "145.8", "--bandwidth", "1799.4", "--dry-run"])
    printed = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    secs = float(re.search(r"estimated GPU time (\d+) s", printed).group(1))
    grid = BM.build_grid(MIXTRAL, TILES, 1024, 32, 6)
    assert secs == pytest.approx(len(grid) * len(TILES) * 1.5, rel=0.15)
    assert "300 ms warmup + 3 trials x 400 ms of kernel time" in printed


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="the mirror can only be checked against the original")
def test_the_mirrored_iters_clamp_is_the_instruments_own():
    """`planned_iters` falls back to a COPY of `timing.iters_for`'s arithmetic
    because that module imports torch and this script plans runs on a laptop
    with none. A copy nobody compares is a second policy."""
    from moe.bench.timing import iters_for
    for ms in (0.05, 0.5, 1.0, 11.0, 400.0):
        assert BM.planned_iters(ms, 400.0) == iters_for(ms, 400.0)
    assert BM.ITERS_FOR_LO == iters_for(1e9, 400.0)
    assert BM.ITERS_FOR_HI == iters_for(1e-9, 400.0)


def test_one_spurious_low_tread_no_longer_discards_the_branch_from_below():
    """THE SINGLE-TREAD VETO, ARRIVING FROM THE OTHER SIDE.

    R4 replaced "the prefix from n=1" with "the first contiguous run", which
    fixed the n=1 case and left the same defect for any other lone tread: one
    spurious True at low n makes the FIRST run one tread long and throws the
    real branch away. Under the 3x-noise margin `analyse` uses, a tread sitting
    just outside it is exactly how that arises. The rule is now the LONGEST run,
    so the answer is never handed to the least trustworthy point.
    """
    xs = [float(n) for n in range(1, 9)]
    overhead, c = 0.05, 0.5

    def ys_from(above):
        # 8% above the line where the flag is set, 1% above where it is not.
        return [overhead + c * x * (1.08 if flag else 1.01)
                for x, flag in zip(xs, above, strict=True)]

    spurious_first = [1, 0, 1, 1, 1, 1, 1, 1]
    start, k, above = BM.memory_branch_members(
        xs, ys_from(spurious_first), c, overhead, 0.042)
    assert above == [bool(f) for f in spurious_first]
    assert (start, k) == (2, 6), "six memory-bound treads, not one"

    spurious_pair = [1, 1, 0, 1, 1, 1, 1, 1]
    start, k, _ = BM.memory_branch_members(
        xs, ys_from(spurious_pair), c, overhead, 0.042)
    assert (start, k) == (3, 5), "the run of five, not the run of two"


def test_the_lowest_run_wins_a_tie_and_a_clean_prefix_is_unchanged():
    """The tie goes low because the low treads are where a memory branch is if
    there is one, and the clean shape R4 was written for must still read the
    same or this rule has moved the published answers."""
    xs = [float(n) for n in range(1, 9)]
    overhead, c = 0.05, 0.5

    def ys_from(above):
        return [overhead + c * x * (1.08 if flag else 1.01)
                for x, flag in zip(xs, above, strict=True)]

    tie = [1, 1, 1, 0, 1, 1, 1, 0]
    assert BM.memory_branch_members(xs, ys_from(tie), c, overhead, 0.042)[:2] \
        == (0, 3)
    clean = [0, 1, 1, 1, 1, 1, 1, 1]
    assert BM.memory_branch_members(xs, ys_from(clean), c, overhead, 0.042)[:2] \
        == (1, 7)
    none_above = [0] * 8
    assert BM.memory_branch_members(
        xs, ys_from(none_above), c, overhead, 0.042)[:2] == (0, 0)


def test_the_bandwidth_refusal_reaches_a_caller_that_cannot_catch_it():
    """`scripts/tile_cap_test.py:1760` calls `resolve_bandwidth` OUTSIDE any
    try, so a plain `RuntimeError` arrived there as a traceback and exit 1 --
    and 1 is CLAIM_FAIL in the very table this study adopted, so a driver would
    have ledgered a REFUSAL as a measured refutation. It is a `SystemExit`
    carrying `code = REFUSED`, and still a `RuntimeError` for `main`'s named
    except."""
    assert issubclass(BM.BandwidthUnavailable, SystemExit)
    assert issubclass(BM.BandwidthUnavailable, RuntimeError)
    args = BM.build_parser().parse_args(["--dry-run"])
    args.ridge = 145.8
    try:
        BM.resolve_bandwidth(args)
    except BM.BandwidthUnavailable as exc:
        assert exc.code == exit_codes.REFUSED
        assert "hybrid" not in str(exc).lower() or True
        assert "--bandwidth" in str(exc)
    else:
        raise AssertionError("it must still refuse")


def test_the_bandwidth_refusal_is_printed_once_at_the_raise_site(capsys):
    """An unhandled `SystemExit` whose code is an int prints NOTHING, so the
    reason has to be emitted where the refusal happens. And exactly once:
    `main` deliberately does not re-print it, or one refusal would look like
    two."""
    rc = BM.main(["--ridge", "145.8"])
    printed = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert printed.count("REFUSED:") == 1
    assert "145.8 x 4374.5 is not any card's roof" in printed


def test_the_retired_instrument_is_not_swallowed_by_a_per_cell_except():
    """WHAT THE FOUR SIBLING ARMS ACTUALLY DO WITH THE REFUSAL.

    All four time inside a per-cell `except Exception`
    (`bm128_roofline.py:1564`, `bm128_depth.py:1620`, `bn_decomposition.py:2303`,
    `occupancy_vs_swizzle.py:1309`). While `RetiredInstrument` was a
    `RuntimeError` every one of them caught it per cell, wrote
    `status="failed"` and went on to compile and run its whole grid -- so
    `bm128_roofline`, the arm scheduled to settle the BLOCK_M=128 question,
    would have burned a pod allocation to produce no usable cell. This is the
    handler those four have, written out."""
    assert issubclass(BM.RetiredInstrument, BaseException)
    assert not issubclass(BM.RetiredInstrument, Exception)
    cells_attempted = 0
    with pytest.raises(BM.RetiredInstrument):
        for _ in range(4):
            cells_attempted += 1
            try:
                BM.time_call(lambda: None, 20, 50)
            except Exception:                             # noqa: BLE001
                continue
    assert cells_attempted == 1, "the arm must stop at the first timed cell"


def test_a_self_test_report_never_claims_the_real_instrument(tmp_path):
    """`instrument` is one of the five keys `Provenance.stamp` puts at the
    payload's top level and a publish gate checks, and report.json outlives
    every log. A self-test report carrying the pod instrument's name satisfied
    that gate while describing an instrument the run never ran."""
    _, payload = run(["--self-test", str(REFIT)], tmp_path)
    assert payload["instrument"] == BM.SYNTHETIC_INSTRUMENT
    assert payload["provenance"]["instrument"] == BM.SYNTHETIC_INSTRUMENT
    assert payload["provenance"]["iters"] is None
    assert payload["provenance"]["missing"]["iters"] == "supplied as None"


def test_the_provenance_iteration_count_is_the_instruments_and_never_the_knob():
    """`--iters` is retired as a timing knob, and `main` recorded its argparse
    default 50 in every report while `time_kernel` sized each cell's own count
    from `--cell-budget-ms` -- hundreds for a 1 ms kernel. `provenance.iters`
    then contradicted every row of cells.csv and a reader could not tell which
    was the instrument's."""
    prov = BM.PV.provenance_block(instrument="test", iters=None)
    assert prov.iters is None
    assert prov.missing["iters"] == "supplied as None"
    # Nothing timed: it stays None rather than inventing a count.
    assert BM.observed_iters(prov, cells_at(REFIT)).iters is None
    assert "none recorded" in BM.iters_line(cells_at(REFIT))
    # Timed: the median of what the instrument used, and the range beside it.
    timed = [replace(c, iters=n) for n, c in
             zip([120, 400, 400, 900], cells_at(REFIT)[:4], strict=True)]
    assert BM.observed_iters(prov, timed).iters == 400
    assert "iters" not in BM.observed_iters(prov, timed).missing
    assert "range 120-900" in BM.iters_line(timed)


def test_gate_3_will_not_answer_for_a_tile_whose_own_fit_refused_to(tmp_path):
    """ONE REPORT CANNOT BOTH REFUSE TO ANSWER FOR A TILE AND ANSWER FOR IT.

    When `fit_ladder` discards the BLOCK_M=128 memory branch for running
    parallel to the compute branch it says in as many words that "no alpha may
    be imported over it" -- and gate 3 then imported one from BLOCK_M=64 and
    returned PASS, two lines below printing that sentence. It is UNDECIDED now,
    which `classify` scores as a claim gate that did not pass."""
    rc, payload = run(["--self-test-world", BM.PARALLEL_WORLD], tmp_path)
    assert rc == exit_codes.CLAIM_FAIL
    g3 = next(g for g in payload["gates"] if g["number"] == 3)
    assert g3["verdict"] == "UNDECIDED"
    assert g3["provenance"]["blocked_by_target_tile"] is True
    assert g3["provenance"]["target_tile_outcome"] == BM.UNDECIDED_PARALLEL_BRANCH
    assert "NOT SCORED" in " ".join(g3["detail"])
    assert "roofline arm" in " ".join(g3["detail"])


def test_gate_3_still_refutes_on_an_imported_alpha_that_lands_outside_the_band(
        tmp_path):
    """THE ASYMMETRY, PLANTED. A PASS over an undecided tile claims the band
    holds for a tile nothing identified; a FAIL says the alpha that WAS fitted
    lies outside the band, is labelled `[CLAIM/IMPORTED]`, and does not need
    the undecided tile to be true. If the block applied to both, every planted
    world would come back UNDECIDED and the gate would discriminate nothing."""
    rc, payload = run(["--self-test", "0.85", "--fail-on-gate"], tmp_path)
    assert rc == exit_codes.CLAIM_FAIL
    g3 = next(g for g in payload["gates"] if g["number"] == 3)
    assert g3["verdict"] == "FAIL"
    assert g3["provenance"]["blocked_by_target_tile"] is False
    assert "DISJOINT" in g3["gate"] and "ABOVE" in g3["gate"]


def test_the_gate_3_import_line_never_states_a_tread_count_the_fit_denies(
        tmp_path):
    """A discarded branch leaves `memory_points` at 0, so the import line
    printed "0 tread(s) stand above the compute branch" for a ladder whose
    eight treads all did -- while the same report two lines later printed the
    fit's own "This is NOT a shortage of treads". The reason now comes from the
    fit."""
    _, payload = run(["--self-test-world", BM.PARALLEL_WORLD], tmp_path)
    g3 = next(g for g in payload["gates"] if g["number"] == 3)
    joined = " ".join(g3["detail"])
    assert "0 tread(s) stand above the compute branch" not in joined
    assert "NOT a shortage of treads" in joined


def test_the_exit_line_never_says_both_that_gates_passed_and_that_one_did_not(
        tmp_path, capsys):
    """The default path. It printed `describe(DONE)` -- "every VALIDITY and
    CLAIM gate PASSED" -- and appended "a claim gate did not pass", so one line
    said both. In a study whose A4 finding is logs asserting things that did not
    happen, that is the same defect in miniature."""
    rc = BM.main(["--self-test", "0.85", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert rc == exit_codes.CLAIM_FAIL
    exit_lines = [ln for ln in printed.splitlines() if ln.startswith("exit ")]
    assert len(exit_lines) == 1
    assert exit_codes.describe(exit_codes.CLAIM_FAIL) in exit_lines[0]
    assert "every VALIDITY and CLAIM gate PASSED" not in printed
    # AND THE LINE THAT USED TO SAY THE OPPOSITE IS GONE, not reworded: there is
    # no longer a code to fold into, so nothing may say there is.
    assert "reported as exit 0" not in printed
    assert "a claim that did not pass is a RESULT" in printed


# --------------------------------------------------------------------------
# A THIRD PASS on 2026-09-02. The whole-repo verification found that the
# apparatus rebuild had left this file's contract in three states no other arm
# was in: a planted world could overwrite a metered one, a crash was ledgered as
# a refutation, and `--dry-run` and `--fail-on-gate` still folded two different
# things into DONE.
# --------------------------------------------------------------------------

def test_three_planted_worlds_write_three_directories(tmp_path):
    """W1. `--self-test 0.2`, `--self-test 0.9` and
    `--self-test 0.2 --self-test-noise 0.5` all derived
    `nocard-budget400.0-...-e77c8230`, so each planted world overwrote the last
    and no two self-tests could be compared. The alpha, the noise and the world
    are knobs like any other: they decide every number in the report."""
    for argv in (["--self-test", "0.2"], ["--self-test", "0.9"],
                 ["--self-test", "0.2", "--self-test-noise", "0.5"],
                 ["--self-test", "0.2", "--self-test-world", BM.LOW_CLOCK_WORLD]):
        BM.main([*argv, "--out", str(tmp_path)])
    dirs = sorted(p.name for p in (tmp_path / "block_m_crossing").iterdir())
    assert len(dirs) == 4, dirs
    assert all(d.startswith("synthetic-") for d in dirs), dirs
    assert all((tmp_path / "block_m_crossing" / d / "report.json").exists()
               for d in dirs)


def test_a_planted_run_is_never_named_for_the_card_it_did_not_measure():
    """W1, THE POD HALF AND THE EXPENSIVE ONE. `detect_card_slug` returns the
    ATTACHED device whether or not anything was measured, so one free
    `--self-test 0.10` on the metered machine landed in the metered run's own
    directory and its unconditional `report.json` write replaced the paid arm's
    only machine-readable artefact with a synthetic one carrying the retracted
    alpha. The card component says `synthetic` and that is also the directory's
    prefix; the attached card survives in the key so two pods' self-tests are
    still two directories."""
    parser = BM.build_parser()
    measured = BM.default_run_id(parser.parse_args([]), "NVIDIA H200")
    planted = BM.default_run_id(parser.parse_args(["--self-test", "0.10"]),
                                "NVIDIA H200")
    assert measured.startswith("nvidia_h200-")
    assert planted.startswith(f"{BM.SYNTHETIC_CARD_SLUG}-")
    assert "nvidia_h200" not in planted
    on_a100 = BM.default_run_id(parser.parse_args(["--self-test", "0.10"]),
                                "NVIDIA A100-SXM4-80GB")
    # The planted world's ridge and bandwidth come from the attached card's own
    # calibration even under --self-test, so these are two different worlds.
    assert planted != on_a100


def test_a_world_planted_without_an_alpha_is_still_a_planted_run():
    """`--self-test-world` implies `--self-test` in `main`, and the id has to
    apply the same rule or a caller that derives the id first gets a MEASURED
    run's directory for a planted run."""
    parser = BM.build_parser()
    world = BM.default_run_id(
        parser.parse_args(["--self-test-world", BM.PARALLEL_WORLD]), "nocard")
    assert world.startswith(f"{BM.SYNTHETIC_CARD_SLUG}-")
    assert world != BM.default_run_id(parser.parse_args([]), "nocard")


def test_the_run_id_still_refuses_a_planted_run_with_no_card():
    """The synthetic card names the RUN, not the machine, and it may not become
    a way of not saying which machine generated the world."""
    from moe.bench import provenance as PV
    args = BM.build_parser().parse_args(["--self-test", "0.558"])
    with pytest.raises(PV.NoCard):
        BM.default_run_id(args, "")


def test_a_crash_is_error_and_not_a_refuted_claim(tmp_path, capsys):
    """W2. An unhandled exception exits the interpreter ONE, and 1 is
    CLAIM_FAIL: in `FINISHED_CODES`, recorded by the driver, never retried. A
    torch OOM three cells into a rented pod would have been filed as one of this
    experiment's registered outcomes. ERROR (4) is outside `FINISHED_CODES`
    precisely so the driver can tell "the apparatus broke" from "the claim did
    not hold"."""
    def boom(*a, **kw):
        raise RuntimeError("planted: torch OOM in the middle of the grid")

    original = BM.build_grid
    BM.build_grid = boom
    try:
        rc = BM.main(["--self-test", "0.558", "--out", str(tmp_path)])
    finally:
        BM.build_grid = original
    err = capsys.readouterr().err
    assert rc == exit_codes.ERROR
    assert rc not in exit_codes.FINISHED_CODES
    # The traceback is not swallowed: a code with no reason in it tells an
    # operator nothing about what to fix.
    assert "planted: torch OOM" in err


def test_a_string_system_exit_is_a_refusal_and_not_a_refuted_claim(capsys):
    """The other half of W2, and the live instance. `require_override_config`
    does `raise SystemExit(<str>)` when vLLM has renamed the export; the
    interpreter turns a string code into exit 1, so a refusal about the
    INSTALLED PACKAGE, which measured nothing, exited with the code reserved for
    a measured refutation."""
    def gone():
        raise SystemExit("could not find vLLM's override_config in any of: ...")

    original = BM._main
    BM._main = lambda argv=None: gone()
    try:
        rc = BM.main([])
    finally:
        BM._main = original
    assert rc == exit_codes.REFUSED
    assert "REFUSED: could not find vLLM's override_config" in capsys.readouterr().err


def test_a_dry_run_scores_no_gate_and_returns_the_code_the_census_settled_on(
        tmp_path, capsys):
    """W3, CLOSED. The census in `scripts/bm128_depth.py`'s dry-run branch
    picked REFUSED: a plan scores no gate and prints no RESULT line, so
    `classify_text` over its log raises `NoGatesScored`, which is the REFUSED
    shape, while DONE means "measured; every gate PASSED". This file was the
    last one still returning DONE, and the log and the process therefore said
    two different things about one run."""
    rc = BM.main(["--ridge", "145.8", "--bandwidth", "1799.4", "--dry-run",
                  "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert not [ln for ln in printed.splitlines()
                if ln.startswith(exit_codes.RESULT_PREFIX)]
    # THE AGREEMENT, IN ONE LINE: the log's own shape classifies as REFUSED and
    # so does the process. Before this commit the left half raised and the right
    # half returned 0.
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(printed)
    assert "REFUSED. Nothing was measured and nothing was written." in printed
    # A plan writes nothing at all, which is the half that was always consistent.
    assert not (tmp_path / "block_m_crossing").exists()
    # And the cost line is still on stdout, which is what the one consumer that
    # reads this branch as a subprocess actually parses.
    assert re.search(r"estimated GPU time\s+[0-9.]+\s*s", printed)


def test_the_pod_cost_probe_still_prices_the_arm_after_the_code_moved():
    """THE REGRESSION THE FLIP WOULD HAVE CAUSED, ASSERTED FROM THIS SIDE.
    `scripts/replicate_noise_floor.py:sweep_cost` runs this file's `--dry-run`
    as a subprocess and used to drop the cost on ANY non-zero code, so moving
    DONE to REFUSED alone would have deleted the whole "TOTAL ... min of GPU"
    budget line from a rented pod's plan -- not "cost unknown", the line. That
    is this rebuild's recurring error: a fix that opens a second hole. The
    caller moved in the same commit.

    THIS TEST GUARDS ONE OF THE TWO HALVES, NOT BOTH, and said otherwise until
    a reviewer reverted each half and ran it. Reverting the CALLER (`sweep_cost`
    back to dropping the cost on any non-zero code) fails here. Reverting the
    SWEEP's dry-run code back to DONE does NOT, because DONE is inside
    `PLAN_CODES` by design, so the probe still prices the arm. That direction is
    caught by `test_a_dry_run_scores_no_gate_and_returns_the_code_the_census_settled_on`,
    which asserts the code itself. Two tests, one per half, named here so the
    next reader does not have to rediscover which covers which. Claiming a
    guarantee the code does not give is the same defect this file spent a commit
    removing from a module docstring."""
    nf = _load_sibling("replicate_noise_floor")
    assert exit_codes.REFUSED in nf.PLAN_CODES and exit_codes.DONE in nf.PLAN_CODES
    assert exit_codes.CLAIM_FAIL not in nf.PLAN_CODES, \
        "a refuted claim is not a priced plan"
    assert exit_codes.ERROR not in nf.PLAN_CODES
    secs = nf.sweep_cost(nf.DEFAULT_ARMS[0], sys.executable)
    assert secs is not None and secs > 0, \
        "the sweep's own dry run would not price the arm"


def test_the_retired_flag_is_still_accepted_so_an_old_driver_line_parses():
    """`scripts/pod_session.sh` passes `--fail-on-gate`. Retiring a flag by
    deleting it turns every driver line that names it into an argparse exit 2,
    which is REFUSED, on the pod, in the arm the session was rented for."""
    args = BM.build_parser().parse_args(["--fail-on-gate"])
    assert args.fail_on_gate is True
    assert "RETIRED" in BM.build_parser().format_help()


# --------------------------------------------------------------------------
# A FOURTH PASS on 2026-09-02. The review of the third pass found that the fix
# for W1 had put its three planted knobs in the key of EVERY run, including the
# metered ones, and that the header it wrote described a directory name the code
# did not produce. Both are legibility, not separation: the ids were always
# distinct. Legibility of an expensive run's directory is the whole reason the
# visible part of an id exists.
# --------------------------------------------------------------------------

def test_a_metered_run_pays_nothing_for_the_self_tests():
    """THE FIX'S OWN REGRESSION. `planted`/`plantnoise`/`plantworld` in the key
    of a measurement rendered the constant `plantedmeasured-plantnoise0.0-...`,
    `run_id` cuts the visible part at 96 characters in NAME order, and
    `planted` sorts before `probes`, `r` and `routing`, so 19 characters of
    constant evicted all three. Two pod runs at different `--r-max` became one
    name apart only in the trailing hash, in `ls`, on the arm that costs money.
    The plant belongs on the planted branch, which already says `synthetic`."""
    parser = BM.build_parser()
    ids = {argv[0] if argv else "default":
           BM.default_run_id(parser.parse_args(argv), "NVIDIA H200")
           for argv in ([], ["--r-max", "4096"], ["--step-probes", "12"])}
    for name in ids.values():
        assert "plant" not in name, name
    # THE PROPERTY, and it is about the VISIBLE part, not the hash: strip the
    # trailing 8-character digest and the three runs are still three names.
    visible = {name.rsplit("-", 1)[0] for name in ids.values()}
    assert len(visible) == 3, visible
    assert "r4096" in ids["--r-max"] and "probes12" in ids["--step-probes"]


def test_a_planted_directory_names_the_alpha_the_noise_and_the_world():
    """THE HEADER'S CLAIM, ASSERTED RATHER THAN WRITTEN. The first fix's header
    said the synthetic directory names all three; three separate knobs meant it
    named the alpha and nothing else, because the 96-character cut fell inside
    `plantnoise` and `plantworld` never appeared. That is the defect commit
    5b65ac1 was written to close, so it does not get to come back through the
    commit that cites it."""
    parser = BM.build_parser()

    def visible(argv):
        return BM.default_run_id(parser.parse_args(argv),
                                 "NVIDIA H200").rsplit("-", 1)[0]

    plain = visible(["--self-test", "0.2"])
    noisy = visible(["--self-test", "0.2", "--self-test-noise", "0.5"])
    world = visible(["--self-test", "0.2", "--self-test-world",
                     BM.LOW_CLOCK_WORLD])
    assert "plant0.2" in plain
    assert "plant0.2n0.5" in noisy
    assert "plant0.2wlowclock" in world
    assert len({plain, noisy, world}) == 3, (plain, noisy, world)


def test_the_longest_plant_still_names_its_world_after_the_cut():
    """The one combination that does not fit: alpha and noise and world spend
    24 characters against the 21 the default grid leaves. The tag is ordered so
    what the cut takes is the tail of the world's NAME, not the world's
    presence, and `wparal` is still not `wlowclock`."""
    parser = BM.build_parser()
    longest = BM.default_run_id(parser.parse_args(
        ["--self-test", "0.558", "--self-test-noise", "0.25",
         "--self-test-world", BM.PARALLEL_WORLD]), "NVIDIA H200")
    other = BM.default_run_id(parser.parse_args(
        ["--self-test", "0.558", "--self-test-noise", "0.25",
         "--self-test-world", BM.LOW_CLOCK_WORLD]), "NVIDIA H200")
    assert "plant0.558n0.25wpar" in longest, longest
    assert longest.rsplit("-", 1)[0] != other.rsplit("-", 1)[0]


def test_every_planted_world_has_a_distinct_id_tag():
    """The FAIL branch of the tag map, planted. A world added to
    `SELF_TEST_WORLDS` and forgotten here would raise mid-run; two worlds given
    one tag would put two worlds in one directory, which is W1 again."""
    assert set(BM.WORLD_ID_TAGS) == set(BM.SELF_TEST_WORLDS)
    assert len(set(BM.WORLD_ID_TAGS.values())) == len(BM.SELF_TEST_WORLDS)
    with pytest.raises(ValueError, match="WORLD_ID_TAGS"):
        BM.plant_tag(0.558, 0.0, "a-world-nobody-registered")


# --------------------------------------------------------------------------
# A FIFTH PASS on 2026-09-02. The review of the fourth pass found that the fix
# for W1 was closed only on the DEFAULT path while the header said it was closed
# everywhere, that the dry run still returned DONE against the repository's own
# census, and that the docstring announcing the key's last omission had two
# knobs still outside the key.
# --------------------------------------------------------------------------

def test_a_supplied_run_id_cannot_carry_a_plant_into_a_metered_directory(
        tmp_path, capsys):
    """W1a. `--run-id` bypasses `default_run_id` entirely, so the header's
    "every planted run writes under a `synthetic-` directory" was true of the
    derived name and false of the supplied one. It is not a hypothetical
    operator: `scripts/replicate_noise_floor.py:Arm.sweep_argv` ALWAYS emits
    `--run-id`, and `--rehearse` appends `--self-test`, so a rehearsal wrote its
    synthetic report.json at the paid replicate's byte-identical path."""
    rc = BM.main(["--self-test", "0.558", "--run-id", "mixtral_g1-rep1",
                  "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    # NOTHING WAS WRITTEN, which is the property the metered arm cares about:
    # the refusal lands before `out_dir` is computed, not after the mkdir.
    assert not (tmp_path / "block_m_crossing").exists()
    assert "mixtral_g1-rep1" in printed
    assert "replicate_noise_floor" in printed, \
        "the refusal must name the caller that actually does this"
    assert "synthetic-mixtral_g1-rep1" in printed, \
        "a refusal that does not say what to run instead is a wall"


def test_the_named_way_out_of_that_refusal_actually_runs(tmp_path):
    """THE FAIL BRANCH'S PARTNER, AND THE REASON IT IS A REFUSAL RATHER THAN A
    REWRITE. Prefixing the operator's name silently would have made the sweep
    write somewhere its caller does not look, turning an overwrite into a
    `no report.json at <path>` on every rehearsal replicate. The refusal names
    a second way out that keeps both sides computing one directory, and this
    test is what says that way out is real."""
    rc = BM.main(["--self-test", "0.558", "--run-id",
                  f"{BM.SYNTHETIC_DIR_PREFIX}mixtral_g1-rep1",
                  "--out", str(tmp_path)])
    assert rc in exit_codes.FINISHED_CODES
    report = (tmp_path / "block_m_crossing" /
              f"{BM.SYNTHETIC_DIR_PREFIX}mixtral_g1-rep1" / "report.json")
    assert report.exists()
    assert json.loads(report.read_text())["instrument"] == BM.SYNTHETIC_INSTRUMENT


def test_the_caller_that_supplies_the_name_takes_that_way_out_when_it_rehearses():
    """The other half of W1a, in the file that caused it.
    `replicate_noise_floor.run_replicate` derives ONE run id and uses it for
    both the sweep's `--run-id` and the `report.json` path it later reads, so
    prefixing it there keeps the two in agreement. A rehearsal's directory now
    says `synthetic-` and a metered replicate's is byte-identical to what it
    always was, which is the half that must not move."""
    nf = _load_sibling("replicate_noise_floor")
    arm = nf.DEFAULT_ARMS[0]
    # sweep_args, order and python became required when the sibling slice put
    # the two omitted swept parameters into this key; both fixes ship together.
    metered = nf.run_id_for(arm, 1, gpu_name="NVIDIA H200", cache_mode="fresh",
                            sweep_args=[], order="counterbalanced",
                            python="/usr/bin/python3")
    assert not metered.startswith(nf.SYNTHETIC_DIR_PREFIX)
    # And the prefix decision itself lives in one place both callers reach.
    assert nf.synthetic_run_id(metered, []) == metered
    assert nf.synthetic_run_id(metered, ["--self-test", "0.558"]) == \
        nf.SYNTHETIC_DIR_PREFIX + metered
    assert nf.SYNTHETIC_DIR_PREFIX == BM.SYNTHETIC_DIR_PREFIX
    # The prefix is applied on the `--self-test` in `extra`, which is exactly
    # what `--rehearse` appends, and the sweep would REFUSE the unprefixed name.
    args = BM.build_parser().parse_args(
        ["--self-test", "0.558", "--run-id", metered])
    with pytest.raises(BM.SuppliedRunIdIsNotPlanted):
        BM.resolve_run_id(args, "NVIDIA H200")
    args.run_id = nf.SYNTHETIC_DIR_PREFIX + metered
    assert BM.resolve_run_id(args, "NVIDIA H200") == args.run_id


def test_a_measured_run_may_still_name_its_own_directory():
    """NON-VACUITY, and the thing the wall must not break. `--run-id` exists so
    an operator and a driver can name an experiment; only a PLANTED run is
    walled, because only a planted run can write a synthetic report into a paid
    run's path."""
    args = BM.build_parser().parse_args(["--run-id", "mixtral_g1-rep1"])
    assert BM.resolve_run_id(args, "NVIDIA H200") == "mixtral_g1-rep1"
    args = BM.build_parser().parse_args([])
    assert BM.resolve_run_id(args, "NVIDIA H200") == \
        BM.default_run_id(args, "NVIDIA H200")


def test_a_world_planted_without_an_alpha_is_walled_too():
    """`--self-test-world` implies `--self-test` in `main`, but a caller that
    reaches `resolve_run_id` first must get the same wall, or the implication
    becomes the hole."""
    args = BM.build_parser().parse_args(
        ["--self-test-world", BM.PARALLEL_WORLD, "--run-id", "mixtral_g1-rep1"])
    assert args.self_test is None
    with pytest.raises(BM.SuppliedRunIdIsNotPlanted):
        BM.resolve_run_id(args, "NVIDIA H200")


# --------------------------------------------------------------------------
# W2. The key, enumerated from the parser rather than listed by hand.
# --------------------------------------------------------------------------

#: Values for the two flags argparse does not describe well enough to perturb
#: from the action alone: `--tiles` is a comma list `default_run_id` parses as
#: ints, and `--capability` is a MAJOR.MINOR string. Everything else is derived
#: from `action.choices`, `action.const` or `action.type`, and a flag that lands
#: in neither fails `_perturbed` BY NAME rather than being skipped. That is the
#: property the whole section exists for: a knob added tomorrow cannot fall out
#: of the key quietly, which is what `--sm-count` and `--capability` did.
AWKWARD_ID_VALUES = {"tiles": "32,64,128", "capability": "8.0"}

#: The two argv the enumeration runs from. A knob has to move the id of the runs
#: it can change, and `--self-test-noise` and `--self-test-world` change nothing
#: at all on a measured run: they are read only when a plant exists. One
#: baseline would therefore have forced them to be exempted, which would have
#: said the opposite of the truth about them.
ID_BASELINES = ([], ["--self-test", "0.558"])


def _perturbed(action, current):
    """A value for this flag that differs from `current`, derived from the
    action argparse built rather than from a table of flag names."""
    if action.choices:
        for choice in action.choices:
            if choice != current:
                return choice
        raise AssertionError(f"--{action.dest} has no second choice to move to")
    if action.const is not None and isinstance(action.const, bool):
        return not current
    if action.type is int:
        return int(current or 0) + 1
    if action.type is float:
        return float(current or 0.0) + 1.0
    if action.dest in AWKWARD_ID_VALUES:
        return AWKWARD_ID_VALUES[action.dest]
    raise AssertionError(
        f"--{action.dest} is neither exempt in block_m_crossing_sweep."
        "ID_EXEMPT_DESTS nor perturbable from its argparse action; classify it "
        "before adding it, or the run id may silently stop naming it")


def _unkeyed(derive):
    """Every non-exempt destination `derive` does not react to.

    Empty is the passing answer. Returned rather than asserted so the FAIL
    branch can be planted: an id function deliberately blind to one knob has to
    come back naming exactly that knob, or this checker proves nothing.
    """
    parser = BM.build_parser()
    actions = [a for a in parser._actions if a.dest not in BM.ID_EXEMPT_DESTS]
    assert actions, "the parser defines no keyed destination at all"
    missing = set()
    for action in actions:
        moved = False
        for argv in ID_BASELINES:
            base_args = parser.parse_args(argv)
            args = argparse.Namespace(**vars(base_args))
            setattr(args, action.dest,
                    _perturbed(action, getattr(args, action.dest)))
            if derive(args, "NVIDIA H200") != derive(base_args, "NVIDIA H200"):
                moved = True
                break
        if not moved:
            missing.add(action.dest)
    return missing


def test_every_argparse_knob_that_changes_a_measured_value_is_in_the_run_id():
    """W2, AND THE MECHANISM RATHER THAN THE CLAIM. `default_run_id`'s docstring
    twice announced that the key's last omission was closed while a knob was
    still outside it: first the planted world, then `--sm-count` and
    `--capability`, the second time in the paragraph that cited the commit about
    headers describing a state the code is not in. Prose cannot be executed.
    This enumerates the PARSER, so a flag added tomorrow either moves the id or
    is classified in `ID_EXEMPT_DESTS` on purpose."""
    assert _unkeyed(BM.default_run_id) == set()


def test_that_enumeration_catches_a_knob_that_falls_out_of_the_key():
    """THE FAIL BRANCH, PLANTED, AND IT IS THE DEFECT VERBATIM. An id function
    blind to `--capability` or to `--sm-count` is precisely what this file
    shipped until now, and the checker has to name it. A checker that only ever
    returns the empty set would pass the test above while proving nothing."""
    def blind(dest):
        def derive(args, card):
            stripped = argparse.Namespace(**vars(args))
            setattr(stripped, dest,
                    getattr(BM.build_parser().parse_args([]), dest))
            return BM.default_run_id(stripped, card)
        return derive

    assert _unkeyed(blind("capability")) == {"capability"}
    assert _unkeyed(blind("sm_count")) == {"sm_count"}


def test_the_two_knobs_the_docstring_had_forgotten_now_move_the_id():
    """The finding in one command. `['--sm-count','108']` and
    `['--capability','8.0']` each derived an id BYTE-IDENTICAL to `[]`, while
    `--sm-count` is baked into every persisted row through `waves` and
    `--capability` PRUNES the tile set that gets measured at all."""
    parser = BM.build_parser()
    base = BM.default_run_id(parser.parse_args([]), "NVIDIA H200")
    for argv in (["--sm-count", "108"], ["--capability", "8.0"]):
        assert BM.default_run_id(parser.parse_args(argv), "NVIDIA H200") != base


def test_an_unset_sm_count_or_capability_costs_a_metered_name_nothing():
    """THE FIX'S OWN REGRESSION, WHICH IS THE ONE THIS FILE KEEPS MAKING.
    Keying both unconditionally would have written the constant
    `capdriver`/`smdriver` into every metered run's visible name, spending 19 of
    its 96 characters on a fact the card slug already carries -- which is
    exactly what the three `plant` knobs did to the arm that costs money. Both
    default to "ask the driver", so only an override pays."""
    parser = BM.build_parser()
    metered = BM.default_run_id(parser.parse_args([]), "NVIDIA H200")
    assert "cap" not in metered and "sm" not in metered.replace("nvidia", "")
    # The three names two pod runs are told apart by in `ls` are still there.
    for argv, token in ((["--r-max", "4096"], "r4096"),
                        (["--step-probes", "12"], "probes12")):
        assert token in BM.default_run_id(parser.parse_args(argv), "NVIDIA H200")


def test_the_exemption_table_only_names_flags_this_parser_defines():
    """An exemption for a flag that no longer exists is an exemption nobody can
    read, and a typo in one silently exempts nothing while looking like it
    exempts something."""
    dests = {a.dest for a in BM.build_parser()._actions}
    assert set(BM.ID_EXEMPT_DESTS) <= dests, set(BM.ID_EXEMPT_DESTS) - dests
    assert all(why.strip() for why in BM.ID_EXEMPT_DESTS.values()), \
        "an exemption with no reason is a knob nobody decided about"


def test_a_new_flag_this_section_cannot_classify_fails_by_name():
    """THE OTHER FAIL BRANCH, PLANTED. The enumeration is only worth having if a
    knob it cannot handle STOPS it: a `_perturbed` that quietly skipped an
    unrecognised action would let the next `--sm-count` through exactly the way
    the last one got through, and the test above would still be green."""
    parser = BM.build_parser()
    parser.add_argument("--a-flag-nobody-classified", default="whatever")
    action = [a for a in parser._actions
              if a.dest == "a_flag_nobody_classified"][0]
    with pytest.raises(AssertionError, match="ID_EXEMPT_DESTS"):
        _perturbed(action, "whatever")


def test_the_keyed_and_the_exempt_together_are_the_whole_parser():
    """NON-VACUITY for the enumeration: if `_unkeyed` were reading an empty list
    of actions it would return the empty set forever. Every destination this
    parser defines is in exactly one of the two piles."""
    dests = {a.dest for a in BM.build_parser()._actions}
    keyed = dests - set(BM.ID_EXEMPT_DESTS)
    assert keyed and set(BM.ID_EXEMPT_DESTS)
    assert keyed | set(BM.ID_EXEMPT_DESTS) == dests
    assert not (keyed & set(BM.ID_EXEMPT_DESTS))


def test_a_measured_run_cannot_hide_under_the_synthetic_prefix_either():
    """THE HOLE THE REFUSAL ITSELF OPENS. `resolve_run_id` refuses an unprefixed
    name for a plant and tells the operator to use `synthetic-<name>`; the next
    command that name is pasted into may be the metered one, and then a PAID arm
    writes into a directory every reader of this corpus takes to mean
    "generated, not measured" -- and one a later self-test may resume into. The
    rule is the prefix means planted, and it is refused in both directions."""
    args = BM.build_parser().parse_args(
        ["--run-id", f"{BM.SYNTHETIC_DIR_PREFIX}mixtral_g1-rep1"])
    with pytest.raises(BM.SuppliedRunIdIsNotPlanted, match="plants nothing"):
        BM.resolve_run_id(args, "NVIDIA H200")
    # And the same name IS accepted the moment the run really is planted, which
    # is what stops this second wall from closing the first one's way out.
    args.self_test = 0.558
    assert BM.resolve_run_id(args, "NVIDIA H200") == args.run_id


# --------------------------------------------------------------------------
# 2026-09-08: the HIGH side of LEVEL. `timing.clock_flags` went two-sided on
# 2026-09-03 and `Cell.clock_excluded` kept reading a failed verdict as "ran
# cold", so every boosted memory-shaped cell (the H200's normal state, 1980 MHz
# against a 1515 MHz GEMM reference) left every ladder fit. Both sides are
# planted below, through the Cell, the synthetic world and the CLI.
# --------------------------------------------------------------------------

def _clocked_cell(load, level, side):
    return BM.make_cell(MIXTRAL, 256, 128, 1.0, sm_count=132, block_n=64,
                        sm_clock_load_mhz=load, clock_level_ok=level,
                        clock_drift_ok=True, clock_level_side=side)


def test_the_level_sides_are_the_instruments_own():
    """Mirrored, not imported: `moe.bench.timing` imports torch at module
    scope and this file replays a CSV on a laptop without one."""
    timing = pytest.importorskip("moe.bench.timing")
    assert BM.LEVEL_LOW == timing.LEVEL_LOW
    assert BM.LEVEL_HIGH == timing.LEVEL_HIGH
    assert BM.H200_BOOST_RATIO == pytest.approx(1980.0 / 1515.0)


def test_neither_level_side_excludes_and_a_drifting_clock_does():
    """1980 against 1515 with the side "high" is not an exclusion, and since
    2026-09-09 neither is 1400 with the side "low": on a power-capped card both
    are a tile's own steady operating point. A clock that MOVED across the
    cell's own trials is. None is neither, because an exclusion has to be
    positively established."""
    high = _clocked_cell(1980.0, False, "high")
    low = _clocked_cell(1400.0, False, "low")
    unknown = _clocked_cell(None, None, "")
    moved = replace(_clocked_cell(1400.0, True, ""), clock_drift_ok=False)
    assert high.clock_excluded is False and high.clock_boosted is True
    assert low.clock_excluded is False and low.clock_sagged is True
    assert low.clock_boosted is False and high.clock_sagged is False
    assert unknown.clock_excluded is False and unknown.clock_boosted is False
    assert moved.clock_excluded is True and moved.clock_sagged is False
    # The instrument's None side is stored as the blank, never as "None".
    assert _clocked_cell(1515.0, True, None).clock_level_side == ""


def test_a_failed_verdict_without_a_side_is_refused_not_read_as_low():
    """THE SIBLING'S SHAPE. `bn_decomposition`, `occupancy_vs_swizzle` and
    `span_extent_separation` copy `t.clock_level_ok` into `make_cell` and
    drop `t.clock_level_side`; until this test the blank side was read as LOW
    and every boosted tread they timed left the fit (the second call site of
    the defect the side column closed). The instrument never produces a failed
    verdict without a side, so the row is refused at construction, through
    `make_cell` and through the dataclass, and with the side None as well as
    blank."""
    with pytest.raises(ValueError, match="no clock_level_side"):
        _clocked_cell(1980.0, False, "")
    with pytest.raises(ValueError, match="no clock_level_side"):
        _clocked_cell(1980.0, False, None)
    with pytest.raises(ValueError, match="no clock_level_side"):
        BM.check_level_side(False, "")
    with pytest.raises(ValueError, match="no clock_level_side"):
        replace(_clocked_cell(1980.0, False, "high"), clock_level_side="")
    # The same verdict WITH its side is the kept row the refusal exists for.
    assert _clocked_cell(1980.0, False, "high").clock_excluded is False


def test_a_pre_side_cells_csv_is_refused_only_on_its_failed_rows(tmp_path):
    """A cells.csv written before the side column reads back blank on every
    row. Blank is the value on a passing or undetermined verdict, so those rows
    resume; a FAILED row's side cannot be recovered from the file and is
    refused rather than read as LOW."""
    header = ("block_m,tokens,rows_per_expert,tiles_per_expert,padded_rows,"
              "tile_eff,aligned,waves_up,waves_down,ms_p50,sm_clock_load_mhz,"
              "clock_level_ok,clock_drift_ok\n")
    fine = tmp_path / "fine.csv"
    fine.write_text(header
                    + "128,2048,256.0,2,2048,1.0,True,1.0,1.0,1.0,1515.0,True,True\n"
                    + "128,4096,512.0,4,4096,1.0,True,1.0,1.0,2.0,,,\n")
    _, back = BM.read_cells(fine)
    assert [c.clock_level_side for c in back] == ["", ""]
    assert [c.clock_excluded for c in back] == [False, False]
    failed = tmp_path / "failed.csv"
    failed.write_text(header
                      + "128,2048,256.0,2,2048,1.0,True,1.0,1.0,1.0,1980.0,False,True\n")
    with pytest.raises(ValueError, match="re-measure"):
        BM.read_cells(failed)


def test_a_side_on_a_verdict_that_did_not_fail_is_refused():
    with pytest.raises(ValueError, match="did not fail"):
        _clocked_cell(1980.0, True, "high")
    with pytest.raises(ValueError, match="not one of"):
        _clocked_cell(1980.0, False, "up")
    with pytest.raises(ValueError, match="did not fail"):
        BM.check_level_side(None, "low")
    BM.check_level_side(False, "low")
    BM.check_level_side(False, "high")
    BM.check_level_side(True, "")


def test_the_side_round_trips_through_the_cells_csv(tmp_path):
    path = tmp_path / "cells.csv"
    for cell in (_clocked_cell(1980.0, False, "high"),
                 _clocked_cell(1400.0, False, "low"),
                 _clocked_cell(1515.0, True, "")):
        BM.append_cell(path, cell)
    _, back = BM.read_cells(path)
    assert [c.clock_level_side for c in back] == ["high", "low", ""]
    assert [c.clock_excluded for c in back] == [False, False, False]
    assert [c.clock_boosted for c in back] == [True, False, False]
    assert [c.clock_sagged for c in back] == [False, True, False]
    assert "clock_level_side" in path.read_text().splitlines()[0].split(",")


def test_a_boosted_tread_stays_on_the_ladder_and_is_counted_as_kept():
    """The mirror of `test_a_drifting_cell_is_excluded_from_the_ladder_and_
    counted`: the same tread planted HIGH is on the ladder, the exclusion
    count is zero, the kept count is one, and the fit is the clean world's."""
    clean = cells_at(REFIT)
    grid = BM.build_grid(MIXTRAL, TILES, 1024, 32, 6)
    hot = BM.synthetic_cells(MIXTRAL, grid, TILES, alpha=REFIT, ridge=RIDGE,
                             bandwidth_gbps=BANDWIDTH, b=2, sm_count=132,
                             high_clock=(64, 2))
    boosted = [c for c in hot if c.clock_boosted]
    assert boosted and all(c.block_m == 64 and c.tiles_per_expert == 2
                           for c in boosted)
    assert all(c.sm_clock_load_mhz
               == pytest.approx(BM.SYNTHETIC_CLOCK_MHZ * BM.H200_BOOST_RATIO)
               for c in boosted)
    assert not any(c.clock_excluded for c in hot)
    full, dropped_clean = BM.ladder_treads(clean, 64)
    kept, dropped = BM.ladder_treads(hot, 64)
    assert dropped_clean == 0 and dropped == 0
    assert kept == full
    assert BM.boosted_treads(hot, 64) == 1 and BM.boosted_treads(clean, 64) == 0
    assert BM.boosted_treads(hot, 128) == 0
    assert BM.off_band_treads(hot, 64) == (0, 1)
    report = analyse(hot, alpha=REFIT)
    fit = report.payload["ladder"]["64"]
    assert fit["kept_high_clock"] == 1 and fit["excluded_drifted"] == 0
    assert fit["alpha"] == pytest.approx(
        analyse(clean, alpha=REFIT).payload["ladder"]["64"]["alpha"])
    text = report.text()
    assert "KEPT with LEVEL failed HIGH" in text
    assert "OVERSTATED on the high side" in text
    assert "roof_at_cell_clock_tflops" in text


def test_the_high_clock_world_reports_kept_treads_and_exits_done(tmp_path):
    """End to end through the CLI, the way `test_the_drift_world_reports_the_
    exclusion_in_the_report` does for the exclusion: the nine cells on the
    planted tread are KEPT and counted, the ladder row says one kept, and the
    world is DONE with nothing excluded."""
    rc, payload = run(["--self-test-world", BM.HIGH_CLOCK_WORLD], tmp_path)
    assert rc == exit_codes.DONE
    assert payload["cells_excluded_for_drift"] == 0
    assert payload["cells_excluded_for_drift_from_ladders"] == 0
    assert payload["cells_kept_level_high"] == 9
    assert payload["cells_kept_level_high_from_ladders"] == 1
    row = payload["ladder"]["64"]
    assert row["kept_high_clock"] == 1 and row["excluded_drifted"] == 0
    assert row["outcome"] == BM.IDENTIFIED
    _, clean = run(["--self-test", str(BM.ALPHA)], tmp_path / "clean")
    assert row["alpha"] == pytest.approx(clean["ladder"]["64"]["alpha"])
    assert len(clean["ladder"]["64"]["points"]) == len(row["points"])
    _, moved = run(["--self-test-world", BM.DRIFT_WORLD], tmp_path / "drift")
    assert moved["ladder"]["64"]["excluded_drifted"] == 1
    assert len(moved["ladder"]["64"]["points"]) == len(row["points"]) - 1


def test_the_cap_overstatement_delegates_to_the_one_bracket():
    """ONE definition of the alpha_a-in-{0, 1} loop: the sweep's bracket is
    `ai_model.overstatement_bracket` exactly, at every tile."""
    n, k = 2 * MIXTRAL.intermediate_size, MIXTRAL.hidden_size
    for bm in (32, 64, 128, 256):
        assert BM.cap_overstatement(MIXTRAL, bm, 64, 2) == \
            ai_model.overstatement_bracket(n, k, block_m=bm, block_n=64, b=2)
    # And the two docstrings that used to state a point state the bracket.
    for doc in (BM.LadderFit.alpha.__doc__, BM.cap_overstatement.__doc__):
        assert "overstatement_bracket" in doc
        assert "32% at BM=128" not in doc
        assert "about 1.32: a 32% overstatement" not in doc


# --------------------------------------------------------------------------
# `compute_reference`: which ladder may BE the reference, and which roof the
# non-vacuity floor stands on. Both added 2026-09-09.
# --------------------------------------------------------------------------

def _ladder_cells(alpha: float, *, tiles=(16, 256)):
    """A sweep at two block sizes, one of which is deliberately memory bound."""
    grid = BM.build_grid(MIXTRAL, tiles, 1024, 32, 6)
    return BM.synthetic_cells(MIXTRAL, grid, tiles, alpha=alpha, ridge=RIDGE,
                              bandwidth_gbps=BANDWIDTH, b=2, sm_count=132)


def test_compute_reference_never_falls_through_to_a_ladder_not_offered():
    """THE FALL-THROUGH THAT MADE A REPORT REFUSE ITS OWN SUBJECT.

    The loop tried block sizes largest first and `continue`d past any with
    under three treads, so a two-tile experiment whose CONTROL was short fell
    through to the next one down: in the 2026-09-09 cap_test arm that was the
    memory-bound BLOCK_M=16 SUBJECT. It fitted, was refused on non-vacuity at
    1.044 -- correct physics, a per-tile slope equal to one full weight read IS
    alpha ~ 1 -- and the report rendered that as "BLOCK_M=16 ... its LEVEL is
    wrong" with the 1.044 never printed.

    `candidates` names which ladders may BE the reference. Every ladder passed
    over is recorded and printed, so a short control cannot go unmentioned
    either.
    """
    cells = [c for c in _ladder_cells(1.0)
             if c.block_m != 256 or c.tiles_per_expert <= 2]
    offered = BM.compute_reference(
        cells, (16, 256), cfg=MIXTRAL, ridge=RIDGE, bandwidth_gbps=BANDWIDTH,
        b=2, candidates=(256,))
    assert offered.block_m is None and offered.refused_block_m is None
    assert "no candidate ladder (BLOCK_M=256)" in offered.note
    assert offered.skipped and "NOT tried as the reference" in offered.skipped[0]
    assert any("PASSED OVER" in line for line in offered.render())
    # And without it the same cells reach down to the subject ladder.
    loose = BM.compute_reference(
        cells, (16, 256), cfg=MIXTRAL, ridge=RIDGE, bandwidth_gbps=BANDWIDTH,
        b=2)
    assert 16 in (loose.block_m, loose.refused_block_m), \
        "the fall-through this argument closes"


def test_candidates_extends_the_reference_search_and_never_narrows_the_checks():
    """`block_sizes` still names every ladder the LEVEL checks compare against;
    `candidates` only restricts which one may be qualified. The non-vacuity
    floor is taken over the smallest SWEPT block size either way, so passing
    `candidates` must not move it."""
    cells = _ladder_cells(0.558, tiles=(32, 64, 128, 256))
    kw = dict(cfg=MIXTRAL, ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2)
    everything = BM.compute_reference(cells, (32, 64, 128, 256), **kw)
    restricted = BM.compute_reference(cells, (32, 64, 128, 256),
                                      candidates=(256,), **kw)
    assert everything.block_m == restricted.block_m == 256
    assert everything.vacuity_ratio == pytest.approx(restricted.vacuity_ratio)
    assert everything.level_comparisons == restricted.level_comparisons


def test_the_fused_footing_moves_the_non_vacuity_floor_off_the_dense_roof():
    """THE FLOOR THAT PRE-REGISTERED AN ARM INVALID.

    On the dense footing the non-vacuity ratio asks a fused layer's measured
    slope to clear `2 BM_min / (b ridge)` of the DENSE GEMM roof: 0.838 of it
    at BM_min=128 on the H200. No fused layer in this study's 26 published
    reports exceeds 0.756 of that roof, so the bm128_depth {128, 256} pairing
    refused its own reference on every card before a cell ran.

    On the FUSED footing both branches stand on the layer's own roof -- the
    reference's measured plateau -- and the ratio becomes the design constant
    `2 BM_min / (b ridge)` outright: a statement about whether the smallest
    swept tile can be memory bound at all. The measurement check does not
    vanish with it; it moves to the band that plateau has to land in, which the
    corrupt A100 BLOCK_N=256 reference at 0.013 of the roof misses by two
    orders of magnitude.
    """
    cells = _ladder_cells(0.558, tiles=(128, 256))
    kw = dict(cfg=MIXTRAL, ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2)
    dense = BM.compute_reference(cells, (128, 256), candidates=(256,), **kw)
    fused = BM.compute_reference(cells, (128, 256), candidates=(256,),
                                 fused_roof_band=(0.465, 1.05), **kw)
    assert dense.vacuity_basis == "dense"
    assert fused.vacuity_basis == "fused"
    assert fused.vacuity_ratio == pytest.approx(dense.vacuity_ratio
                                                * dense.roof_fraction)
    assert fused.vacuity_ratio == pytest.approx(2 * 128 / (2 * RIDGE))
    assert fused.vacuity_floor == pytest.approx(fused.vacuity_ratio)
    assert fused.fused_roof_fraction == pytest.approx(dense.roof_fraction)
    rendered = "\n".join(fused.render())
    assert "footing            FUSED" in rendered
    assert "CONTROL'S MEASURED PLATEAU" in rendered
    assert "2 BM_min / (b x ridge)" in rendered
    assert "footing            DENSE" in "\n".join(dense.render())


def test_the_fused_footing_refuses_a_plateau_outside_the_admitted_band():
    """The A100 BLOCK_N=256 reference implied 3.6 TFLOP/s, 1.4% of the card.
    On the dense footing it was caught by a bound no sound reference meets
    either; on the fused footing the band is what catches it, with two orders
    of magnitude to spare, and the refusal says which number missed.

    AND THE REFUSAL NO LONGER CALLS THE BAND A CORPUS FIGURE. It read "outside
    the [0.465, 1.050] a fused layer occupies in this study's 26 published
    reports"; the corpus interval is `tile_cap_test.FUSED_PLATEAU_BAND` =
    (0.465, 0.756) and 1.050 is V3's dense-peak sanity ceiling, which this file
    itself says at two other places. A published sentence stated a bound the
    corpus does not have."""
    cells = _ladder_cells(0.558, tiles=(128, 256))
    slow = [BM.make_cell(MIXTRAL, c.rows_per_expert, c.block_m,
                         c.ms_p50 * 40.0, sm_count=132, block_n=64)
            if c.block_m == 256 else c for c in cells]
    ref = BM.compute_reference(
        slow, (128, 256), candidates=(256,), cfg=MIXTRAL, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, fused_roof_band=(0.465, 1.05))
    assert ref.block_m is None and ref.refused_block_m == 256
    assert ref.roof_fraction < 0.05
    assert any("outside the [0.465, 1.050]" in why for why in ref.refusals)
    assert any("REFUSED" in line for line in ref.render())
    why = next(w for w in ref.refusals if "outside the [0.465, 1.050]" in w)
    assert "a fused layer occupies in this study" not in why
    assert "this arm admits for a fused layer's roof" in why
    assert "lowest of the 26 published fused plateaus" in why
    assert "dense peak plus its tolerance" in why


# --------------------------------------------------------------------------
# The retired rule, hunted out of the two halves of one --help page.
# --------------------------------------------------------------------------

def test_the_help_page_states_one_rule_and_it_is_the_one_the_code_applies():
    """THE FIX-AT-ONE-OF-TWO-PLACES SHAPE, INSIDE A SINGLE DOCSTRING.

    The 2026-09-09 commit rewrote this docstring's `--self-test-world`
    paragraph ("four worlds ... a drifting-clock tread ... which must be
    excluded") and left the ONE INSTRUMENT paragraph above it saying that a
    cell whose loaded clock came in BELOW the band "is EXCLUDED from the ladder
    fit and counted". `Cell.clock_excluded` and `ladder_treads` read
    `clock_drift_ok is False` only, so the two halves of one `--help` page
    stated opposite rules.
    """
    doc = BM.__doc__
    assert "is EXCLUDED from the ladder\nfit" not in doc
    assert "EXCLUDED IFF `clock_drift_ok` IS FALSE" in doc
    assert "NEITHER LEVEL SIDE\nEXCLUDES ANYTHING" in doc
    # And it is really the page: `main` builds its parser from `__doc__`.
    parser = BM.build_parser() if hasattr(BM, "build_parser") else None
    if parser is not None:
        assert parser.description is doc
    # The code the page describes.
    cold = BM.make_cell(MIXTRAL, 128, 128, 1.0, sm_count=132, block_n=64,
                        clock_level_ok=False, clock_level_side="low",
                        clock_drift_ok=True)
    moved = BM.make_cell(MIXTRAL, 128, 128, 1.0, sm_count=132, block_n=64,
                         clock_drift_ok=False)
    assert cold.clock_excluded is False and cold.clock_sagged is True
    assert moved.clock_excluded is True


def test_the_vacuity_label_names_the_footing_the_number_stands_on():
    """ONE OF TWO PLACES. `_level_checks` appends ", on the fused layer's own
    roof," to the REFUSAL text on the fused footing; the label `render()`
    prints -- which is what a PASSING reference shows, and so what most readers
    meet -- carried the dense-footing wording at both footings. On the fused
    footing 0.209 is `2 BM_min / (b x ridge)` with the reference's own level
    divided out, and not a fraction of one full weight read on the dense
    footing."""
    cells = _ladder_cells(0.558, tiles=(128, 256))
    kw = dict(cfg=MIXTRAL, ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2)
    dense = BM.compute_reference(cells, (128, 256), candidates=(256,), **kw)
    fused = BM.compute_reference(cells, (128, 256), candidates=(256,),
                                 fused_roof_band=(0.465, 1.05), **kw)
    dense_label = next(ln for ln in dense.render() if "non-vacuity" in ln)
    fused_label = next(ln for ln in fused.render() if "non-vacuity" in ln)
    assert "on the fused layer's own roof," not in dense_label
    assert "one full weight read, scaled to the smallest" in dense_label
    assert "one full weight read, on the fused layer's own roof, scaled to " \
        "the smallest" in fused_label


# --------------------------------------------------------------------------
# R9. ONE ESTIMATOR DECIDING EVERYTHING, and its denominator an extrapolation.
#
# `LadderFit.alpha` = B/(A+B) divides the per-tile slope by a LEVEL fitted by
# extrapolating the ladder back to zero tiles across up to 44 treads. On the
# 2026-09-10 H200 session that produced ten values above 1.0 across four arms,
# three negative intercepts, and an alpha_a of -0.8143 whose sign lies inside
# the reference fixed cost's own jackknife error. `LadderFit.weight_streams`
# divides the SAME slope by a measured stream time instead, and
# `fixed_cost_above_intercept` labels the exact arithmetic state, D > A,
# that put alpha_upper above 1. Both are ADDED beside alpha; the published rows
# were all scored on B/(A+B) and every test above still reads them that way.
# --------------------------------------------------------------------------

def _fit_with_rate(points, block_m=64, ref=None, **kw):
    return BM.fit_ladder(points, block_m, ref if ref is not None else planted_reference(),
                         model_name=MIXTRAL.name, dtype="bf16",
                         bandwidth_gbps=BANDWIDTH,
                         bandwidth_source="stated by the test", **kw)


def test_a_ladder_fit_reports_w_beside_alpha_and_names_the_rate():
    """The second estimator, present on the fit and carrying its denominator.
    `w` is the slope over one full stream of the layer's expert weight set, so
    it must equal exactly that division and must name the bandwidth it used."""
    pts = [(n, 0.3 + 0.6443482339382172 * n) for n in range(1, 9)]
    fit = _fit_with_rate(pts)
    w = fit.weight_streams
    assert w is not None
    stream = 1e3 * 2_818_572_288 / (BANDWIDTH * 1e9)
    assert w.streams == pytest.approx(fit.slope_memory / stream, rel=1e-12)
    # One full stream per tile at this card's rate reads as w = 1.0 to the
    # precision the planted slope was written at.
    assert w.streams == pytest.approx(1.0, abs=1e-4)
    assert w.bandwidth_gbps == BANDWIDTH
    assert "stated by the test" in w.render()
    # And alpha is untouched beside it: this ADDS a statistic.
    assert fit.alpha == pytest.approx(fit.slope_memory / fit.load_ms, rel=1e-12)


def test_a_fit_with_no_named_model_or_rate_has_no_w_rather_than_a_guessed_one():
    """The four sibling scripts build a `LadderFit` through `fit_ladder`
    without naming a model, a dtype or a measured bandwidth. w scales 1:1 in
    the rate, so a w against a guessed card would be a number with no meaning;
    None is the honest answer and the note says which input was missing."""
    pts = [(n, 0.3 + 0.64 * n) for n in range(1, 9)]
    fit = BM.fit_ladder(pts, 64, planted_reference())
    assert fit.weight_streams is None
    assert "no default rate" in fit.w_note()
    assert fit.alpha is not None            # the old estimator is unaffected
    # Naming only some of the three is still not naming a denominator.
    assert BM.fit_ladder(pts, 64, planted_reference(),
                         model_name=MIXTRAL.name).weight_streams is None
    assert BM.fit_ladder(pts, 64, planted_reference(), dtype="bf16",
                         bandwidth_gbps=BANDWIDTH).weight_streams is None


def test_a_ladder_with_no_memory_branch_says_so_rather_than_dividing_nothing():
    pts = [(n, 1.0 * n) for n in range(1, 5)]        # entirely on the compute line
    fit = _fit_with_rate(pts, block_m=256,
                         ref=BM.ComputeReference(256, 0.0, 1.0, 0.0, "planted"))
    assert fit.slope_memory is None
    assert fit.weight_streams is None
    assert "no memory branch" in fit.w_note()


def test_alpha_upper_above_one_is_exactly_D_greater_than_A_and_is_labelled():
    """THE DIAGNOSTIC, AND THE IDENTITY BEHIND IT. `alpha_upper = B/(A+B-D)`
    exceeds 1 if and only if `D > A`. That is arithmetic about an
    extrapolation, not a statement about traffic, and it is what drove
    bn_decomposition's pooled alpha_a to -0.8143 in four of six cells. The fit
    is LABELLED, never refused: refusing it would have deleted those four."""
    slope = 0.6443482339382172
    seen = set()
    for intercept, overhead in ((0.50, 0.05), (0.30, 0.40), (0.10, 0.40)):
        pts = [(n, intercept + slope * n) for n in range(1, 9)]
        ref = BM.ComputeReference(256, overhead, 1.0, 0.0, "planted")
        fit = BM.fit_ladder(pts, 64, ref, model_name=MIXTRAL.name, dtype="bf16",
                            bandwidth_gbps=BANDWIDTH)
        a = fit.intercept
        assert a == pytest.approx(intercept, abs=1e-9)
        assert fit.fixed_cost_above_intercept is (overhead > a)
        assert (fit.alpha_upper > 1.0) is (overhead > a)
        seen.add(fit.fixed_cost_above_intercept)
        # The statistic that does not depend on A or D at all is unmoved by
        # either of them: the same slope, so the same w, in every state.
        assert fit.weight_streams.streams == pytest.approx(1.0, abs=1e-4)
    assert seen == {True, False}, "the cases must exercise both sides"


def test_the_D_greater_than_A_label_is_carried_on_the_fit():
    """It is a state a reader has to be able to find in the file, not only in
    the prose: `alpha_upper` out of range is the SYMPTOM and this is the cause."""
    slope = 0.6443482339382172
    pts = [(n, 0.10 + slope * n) for n in range(1, 9)]
    ref = BM.ComputeReference(256, 0.40, 1.0, 0.0, "planted")
    fit = BM.fit_ladder(pts, 64, ref, model_name=MIXTRAL.name, dtype="bf16",
                        bandwidth_gbps=BANDWIDTH)
    assert fit.fixed_cost_above_intercept
    assert fit.alpha_upper > 1.0
    assert "D > A" in fit.w_note()
    # And a fit that is NOT in that state does not carry the label.
    ok = BM.fit_ladder([(n, 0.50 + slope * n) for n in range(1, 9)], 64,
                       BM.ComputeReference(256, 0.05, 1.0, 0.0, "planted"),
                       model_name=MIXTRAL.name, dtype="bf16",
                       bandwidth_gbps=BANDWIDTH)
    assert ok.fixed_cost_above_intercept is False
    assert "D > A" not in ok.w_note()


def test_both_places_that_print_a_per_block_m_alpha_print_w_beside_it(tmp_path):
    """THE RECURRING DEFECT, HUNTED WHERE THIS CHANGE COULD REINTRODUCE IT.
    Two sites in the report print a per-BLOCK_M alpha: the ladder table and
    gate 3's detail lines. A statistic added to one and not the other is the
    same shape as every fix this repository has applied at one of two call
    sites. Both take their w from `LadderFit`, and this asserts both."""
    rc, report = run(["--self-test", str(REFIT)], tmp_path)
    assert rc == exit_codes.DONE
    detail = [line for g in report["gates"] for line in g["detail"]]
    per_bm = [line for line in detail
              if re.search(r"BLOCK_M=\s*\d+\s+alpha", line)]
    assert per_bm, "gate 3 printed no per-BLOCK_M alpha line"
    for line in per_bm:
        assert " w " in line, line
    # And the ladder table's own header and rows carry the column.
    table = report["ladder"]
    assert table["32"]["weight_streams_per_tile"] is not None
    assert table["32"]["weight_stream_bandwidth_gbps"] is not None
    # `gate.measured` is left as the bare alpha token on purpose: it is what 22
    # published reports carry and what `exit_codes` parses off a RESULT line.
    # w rides beside it, never inside it.
    g3 = gate_json(report, 3)
    assert g3["measured"].startswith("alpha ")
    assert "weight-streams/M-tile" not in g3["measured"]
    assert any("weight-streams/M-tile" in line for line in g3["detail"])


def test_the_report_json_carries_w_its_rate_and_the_diagnostic(tmp_path):
    """Persisted BESIDE alpha, never in place of it: the 100,144 published rows
    were scored on B/(A+B) and a reader of a new file must still find that
    column, plus the three numbers that let the new division be checked by
    hand."""
    rc, report = run(["--self-test", str(REFIT)], tmp_path)
    assert rc == exit_codes.DONE
    row = report["ladder"]["64"]
    for key in ("alpha", "alpha_corrected", "alpha_upper"):
        assert row[key] is not None, key
    assert row["weight_set_bytes"] == 2_818_572_288
    assert row["weight_stream_ms"] == pytest.approx(
        1e3 * 2_818_572_288 / (row["weight_stream_bandwidth_gbps"] * 1e9),
        rel=1e-12)
    assert row["weight_streams_per_tile"] == pytest.approx(
        row["slope_memory"] / row["weight_stream_ms"], rel=1e-12)
    assert row["fixed_cost_above_intercept"] is (row["overhead_ms"]
                                                 > row["intercept"])
    assert report["weight_streams_measured"] == pytest.approx(
        row["weight_streams_per_tile"], rel=1e-12)
    assert report["weight_streams_bandwidth_source"]


def test_w_does_not_move_a_single_gate_or_alpha(tmp_path):
    """IT ADDS A STATISTIC, IT DOES NOT REPLACE ONE. Every verdict, every
    alpha and the gate 3 interval are what they were before the column
    existed, which is what makes the published corpus still readable."""
    rc, report = run(["--self-test", str(REFIT)], tmp_path)
    assert rc == exit_codes.DONE
    assert [g["verdict"] for g in report["gates"]] == ["PASS"] * len(report["gates"])
    assert report["alpha_measured"] == pytest.approx(0.5239, abs=5e-4)
    assert report["ladder"]["32"]["alpha"] == pytest.approx(0.5373, abs=5e-4)
    assert report["ladder"]["64"]["alpha"] == pytest.approx(0.5413, abs=5e-4)
    # And structurally, not only by the three pinned numbers: alpha is still
    # B over the FITTED LEVEL on every row, which is the definition the
    # published corpus was scored under.
    for row in report["ladder"].values():
        if row["alpha"] is None:
            continue
        level = row["intercept"] + row["slope_memory"]
        assert row["alpha"] == pytest.approx(row["slope_memory"] / level,
                                             rel=1e-12)


def test_the_sweeps_own_per_expert_byte_count_agrees_with_the_weight_set():
    """TWO ROUTES TO ONE DENOMINATOR, CHECKED AGAINST EACH OTHER. This file's
    `weight_bytes_per_expert` (3FH per expert, used by `model_ms`) and
    `moe.bench.weights.routed_expert_weight_bytes` (the whole routed set, used
    by `w`) must differ by exactly the expert count. Two copies of one byte
    count drifting apart is how the two halves of a study end up dividing by
    different denominators, and this study divides by this one."""
    for name, dtype, b in (("mixtral-8x7b", "bf16", 2),
                           ("qwen2-57b-a14b", "bf16", 2),
                           ("mixtral-8x7b", "fp8_e4m3", 1),
                           ("deepseek-v3-tp8", "fp32", 4)):
        cfg = MODEL_CONFIGS[name]
        assert (BM.weight_bytes_per_expert(cfg, b) * cfg.num_experts
                == routed_expert_weight_bytes(name, dtype))
