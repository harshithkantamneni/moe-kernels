"""The apparatus tests for `scripts/block_m_crossing_sweep.py`.

`tests/test_block_m_sweep.py` tests the PHYSICS the sweep argues about: that the
gates tell alpha=0.558 from alpha=0.10, that the ladder recovers a planted
alpha, that the traps do not fire. This file tests the INSTRUMENT the 2026-09-02
audit found underneath it, and every test here is named after the defect it
would have caught:

  R1  a ridge asserted for one card multiplied by a bandwidth inherited from
      another, silently, whenever `load_measured` returned None or raised.
  R2  the analysis crashing with a TypeError in the world the measured alphas
      (0.92-1.02) actually describe, because "this tile never crosses" was a
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
      2*BM/(alpha*b) from it, which overstates them by 32% at BM=128.

Every gate here can FAIL as well as PASS, and the tests that matter most are the
ones that plant the failing world.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
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

from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
TILES = (32, 64, 128, 256)
RIDGE = 160.3
BANDWIDTH = 4374.5
REFIT = 0.558
RETRACTED = 0.10
#: The two worlds the MEASURED G=1 ladders describe (alpha 0.92-1.02 on both
#: cards). At these values no tile in the sweep crosses, which is exactly where
#: the report used to crash.
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
             low_clock=None):
    grid = BM.build_grid(MIXTRAL, tiles, 1024, 32, 6)
    return BM.synthetic_cells(MIXTRAL, grid, tiles, alpha=alpha, ridge=RIDGE,
                              bandwidth_gbps=BANDWIDTH, b=2, sm_count=132,
                              noise=noise, seed=seed, low_clock=low_clock)


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
    assert rc == exit_codes.DONE
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
    the gate-3 provenance line, before report.json was written. The measured
    G=1 alphas on both cards are 0.92-1.02, so the analysis crashed in exactly
    the world the data points at."""
    rc, payload = run(["--self-test", str(alpha)], tmp_path)
    assert rc == exit_codes.DONE
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
    read as "the sweep lacked treads" and let gate 3 import an alpha over it."""
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
    report whose BLOCK_M=128 row is UNDECIDED with its reason, not a null."""
    rc, payload = run(["--self-test-world", BM.PARALLEL_WORLD], tmp_path)
    assert rc == exit_codes.DONE
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


def test_a_low_clock_cell_is_excluded_from_the_ladder_and_counted():
    """A tread timed at 1500 MHz against a roof measured at 1980 sits about 30%
    above the compute branch for a reason that has nothing to do with weight
    re-reads, and it would be fitted straight into alpha."""
    cells = cells_at(REFIT, low_clock=(64, 2))
    full, dropped_none = BM.ladder_treads(cells_at(REFIT), 64)
    kept, dropped = BM.ladder_treads(cells, 64)
    assert dropped_none == 0
    assert dropped == 1
    assert len(kept) == len(full) - 1
    assert 2 not in [n for n, _ in kept]


def test_the_low_clock_world_reports_the_exclusion_in_the_report(tmp_path):
    rc, payload = run(["--self-test-world", BM.LOW_CLOCK_WORLD], tmp_path)
    assert rc == exit_codes.DONE
    assert payload["cells_excluded_for_clock_level"] == 1
    assert payload["ladder"]["64"]["excluded_low_clock"] == 1


def test_a_ladder_that_loses_its_treads_to_clock_level_is_UNDECIDED_for_that_reason():
    """"Two memory-bound treads" and "two memory-bound treads left after four
    were dropped for clock level" report the same count and mean opposite
    things: the second says the card was not at the roof's clock and the arm
    must be re-timed."""
    fit = BM.fit_ladder([(1, 0.9), (2, 1.1)], 128, planted_reference(),
                        margin=0.02, excluded_low_clock=4)
    assert fit.outcome == BM.UNDECIDED_LOW_CLOCK
    assert fit.undecided is True
    assert "clock" in fit.outcome_reason
    assert "Re-time the arm" in fit.outcome_reason
    assert fit.excluded_low_clock == 4


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


def test_a_falsified_claim_is_a_successful_run_unless_the_caller_says_otherwise(
        tmp_path):
    """Both halves of `--fail-on-gate`'s contract, and the code comes from the
    table either way."""
    with_flag = BM.main(["--self-test", "0.85", "--fail-on-gate",
                         "--out", str(tmp_path / "a")])
    without = BM.main(["--self-test", "0.85", "--out", str(tmp_path / "b")])
    assert with_flag == exit_codes.CLAIM_FAIL
    assert without == exit_codes.DONE


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
POD_SESSION_GATE_RE = re.compile(r"^GATE [0-9]+[ \t]+(PASS|FAIL|UNDECIDED)([ \t]|$)")


def test_the_older_gate_line_is_one_per_scored_gate_and_nothing_else(
        tmp_path, capsys):
    """Two verdict channels must carry the same verdicts.

    `RESULT:` is what `scripts/h200_gaps_session.sh` will read; `GATE n PASS` is
    what `scripts/pod_session.sh:gate_from_log` already reads. A line that
    matches the older regex without being a scored gate is the free-text defect
    moved rather than fixed: the audit's noise-floor case was a REFUSED log
    whose prose matched the summary grep 18 times. So the count must equal the
    count of RESULT lines, and the verdicts must agree pairwise."""
    BM.main(["--self-test", "0.85", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    legacy = [m.group(1) for ln in printed.splitlines()
              if (m := POD_SESSION_GATE_RE.match(ln))]
    results = exit_codes.parse_result_lines(printed)
    assert len(legacy) == len(results) == 5
    assert legacy == [r.verdict for r in results]


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
    assert not [ln for ln in printed.splitlines() if POD_SESSION_GATE_RE.match(ln)]
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
