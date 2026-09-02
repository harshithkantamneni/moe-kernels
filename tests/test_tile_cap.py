"""The cap test has to be able to come out the other way, and to refuse.

`scripts/tile_cap_test.py` tests the AI cap FORMULA at BLOCK_SIZE_M=16, which is
the tile vLLM's fallback ladder picks at decode and the one ladder in the study
that is memory bound at every tread. It is NOT the production claim -- the
observed-tile arm says vLLM runs 16 as a single M-tile per expert in every cell
it was seen in, so that cap is real and never approached, and BLOCK_M=128 is the
regime `scripts/bm128_depth.py` asks about. The last group below is the test of
that demotion, recounted from the published arm rather than quoted.

It is still an experiment whose headline result is an ABSENCE, and an absence is
the easiest thing in this repo to produce by accident: stop the sweep early,
lose the positive control, divide by a plateau nothing reached, and every gate
passes.

So most of this file plants a known alpha in synthetic cells, runs the REAL
analysis over them, and checks the verdicts flip between the two worlds. The
single most important test is
`test_a_shallow_sweep_cannot_confirm_the_cap_by_stopping_early`: at a shallow
r_max the cap gate would PASS on cells generated at the alpha it is supposed to
rule out, which is exactly the false confirmation the depth gate refuses.

SIX GROUPS.

  - the depth argument, which is pure arithmetic and is where the run's cost
    comes from;
  - the gates, planted at 0.558 and at 0.10, clean and under timing noise;
  - the refusals: a design that cannot discriminate, a quantity that cannot be
    measured, a grid with holes in it;
  - `scripts/check_mma_path.sh`, whose forced-tile mode is exercised through its
    plan and its refusals off GPU, plus a structural check that the schema
    columns it reads still exist under those names;
  - the ridge, the card and git: three numbers that used to be asserted rather
    than measured;
  - the demotion, where the observed-tile counts are recomputed from the
    published CSV they are quoted from.

The scripts are loaded by path rather than imported, because `scripts/` is not a
package and never has been.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module that is not there yet makes
    # the decorator fail with an AttributeError that names nothing useful.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CAP = _load_script("tile_cap_test")
SWEEP = CAP.SWEEP

from moe.baselines._framework_config import CONFIG_KEY_TO_COLUMN  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
TILES = (16, 256)
RIDGE = 160.3
RIDGE_HI = 176.2
#: The band the synthetic cells here are planted at. Stated at
#: every call because `required_depth` no longer defaults to the
#: module constant: the depth IS a function of the ridge, so a
#: default would put another machine's band back one call site
#: below where the fix removed it.
BAND = (RIDGE, RIDGE_HI)
BANDWIDTH = 4374.5          # the published H200 triad ceiling, as the script pins it
REFIT = 0.558
RETRACTED = 0.10
MMA = ROOT / "scripts" / "check_mma_path.sh"


def cells_at(alpha: float, *, r_max: int = 2112, noise: float = 0.0,
             seed: int = 0, tiles=TILES, row_step: int = 32):
    grid = SWEEP.build_grid(MIXTRAL, tiles, r_max, row_step, 6)
    return grid, SWEEP.synthetic_cells(
        MIXTRAL, grid, tiles, alpha=alpha, ridge=RIDGE, bandwidth_gbps=BANDWIDTH,
        b=2, sm_count=132, noise=noise, seed=seed)


def report_at(alpha: float, **kw):
    tiles = kw.pop("tiles", TILES)
    grid, cells = cells_at(alpha, tiles=tiles, **kw)
    depth = CAP.required_depth(tiles[0], b=2, ridge_band=BAND)
    return CAP.analyse(
        cells, MIXTRAL, cap_tile=tiles[0], control_tile=tiles[1], alpha=alpha,
        ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2, model_name="mixtral-8x7b",
        dtype="bf16", compiles={bm: 1 for bm in tiles},
        executed={bm: 1 for bm in tiles}, sm_count=132, sm_source="test",
        depth=depth, planned_cells=len(grid) * len(tiles), header=[])


def verdicts(report) -> dict[str, str]:
    return {g.tag: g.verdict for g in report.gates}


# --------------------------------------------------------------------------
# The depth argument. Every number here is one the run's cost is set by, so it
# is recomputed rather than quoted.
# --------------------------------------------------------------------------

def test_saturation_is_a_function_of_the_tile_count_and_alpha_alone():
    # AI(n)/cap cancels BLOCK_M and the dtype exactly, which is why the depth
    # requirement is stated in TILES and applies to every block size at once.
    for tiles in (1, 4, 17, 132):
        direct = CAP.saturation(tiles, REFIT)
        for bm, b in ((16, 2), (128, 2), (64, 1)):
            ai = (2.0 * tiles * bm / b) / SWEEP.q_of_tiles(tiles, REFIT)
            assert ai / SWEEP.ai_cap(bm, REFIT, b) == pytest.approx(direct, rel=1e-12)


def test_tiles_for_saturation_matches_the_closed_form_in_the_header():
    # n >= 19 (1-a)/a at the 95% floor. 16 treads at the refit alpha, 17 at the
    # band's low end, which is the number the script plans against.
    assert CAP.tiles_for_saturation(REFIT) == 16
    assert CAP.tiles_for_saturation(CAP.ALPHA_BAND[0]) == 17
    for alpha in (0.2, 0.35, 0.529, 0.558, 0.9):
        n = CAP.tiles_for_saturation(alpha)
        assert CAP.saturation(n, alpha) >= CAP.SATURATION_FLOOR
        assert CAP.saturation(n - 1, alpha) < CAP.SATURATION_FLOOR


def test_the_horizon_is_the_depth_the_retracted_world_would_trip_c1_at():
    lo = CAP.retracted_horizon_tiles(16, retracted=RETRACTED, ridge=RIDGE, b=2,
                                     roof_fraction=CAP.ROOF_FRACTION)
    hi = CAP.retracted_horizon_tiles(16, retracted=RETRACTED, ridge=RIDGE_HI, b=2,
                                     roof_fraction=CAP.ROOF_FRACTION)
    assert (lo, hi) == (52, 132)
    # And it is a threshold, not an approximation: one tread short misses.
    for n, ridge in ((lo, RIDGE), (hi, RIDGE_HI)):
        reached = (2.0 * n * 16 / 2) / SWEEP.q_of_tiles(n, RETRACTED)
        short = (2.0 * (n - 1) * 16 / 2) / SWEEP.q_of_tiles(n - 1, RETRACTED)
        assert reached >= CAP.ROOF_FRACTION * ridge > short


def test_the_default_depth_is_the_deepest_of_the_four_requirements():
    depth = CAP.required_depth(16, b=2, ridge_band=BAND)
    assert (depth.tiles, depth.rows) == (132, 2112)
    assert depth.binding == "the near-roof horizon at the worst end of the ridge band"
    # 2112 rows per expert is 8448 mixtral tokens, which is what the pod pays for.
    assert SWEEP.tokens_for_rows(MIXTRAL, depth.rows) == 8448
    # C1's two conditions have different horizons, and that is the point: the
    # discriminating one is live ten times sooner than the near-roof one, so a
    # cheap run can rule out alpha=0.10 and still not be entitled to the
    # sentence about the roof.
    assert depth.disc_tiles == 13
    assert depth.roof_tiles == 132
    assert depth.disc_condition_live(26) and not depth.roof_condition_live(26)


def test_the_parent_sweeps_horizon_is_vacuous_at_this_tile():
    """Why this module carries its own bracketing, pinned as a fact about the parent.

    `Bracketing.horizon_rows` is `2 x` the crossing the retracted alpha predicts.
    At BLOCK_M=16 the retracted alpha predicts NO crossing, so that horizon is
    `2 x 0`, every depth clears it, and a one-tile sweep would report itself
    bracketed after a single tread.
    """
    retracted = SWEEP.predict_tile(16, RETRACTED, RIDGE, 2)
    assert retracted.crossing_rows is None
    assert 2.0 * (retracted.crossing_rows or 0.0) == 0.0
    # The replacement is not vacuous at the same tile.
    assert CAP.required_depth(16, b=2, ridge_band=BAND).tiles > 100


def test_a_cap_tile_the_retracted_world_could_never_reach_is_refused():
    # BLOCK_M=8: the retracted ceiling is 80 Op/B against a gate at 0.85 x 160.3
    # = 136.3, so C1 could not FAIL at any depth and a PASS would rule nothing
    # out. A gate that cannot fail is as useless as one that cannot pass.
    with pytest.raises(CAP.NonDiscriminating) as exc:
        CAP.required_depth(8, b=2, ridge_band=BAND)
    assert "could never trip C1" in str(exc.value)


# --------------------------------------------------------------------------
# The gates, in the two worlds.
# --------------------------------------------------------------------------

def test_every_gate_passes_in_the_refit_world():
    report = report_at(REFIT)
    assert verdicts(report) == {"V0": "PASS", "V1": "PASS", "V2": "PASS",
                                "V3": "PASS", "V4": "PASS", "C1": "PASS",
                                "C2": "PASS", "C3": "PASS", "C4": "PASS"}
    # 0.176 of ridge x bandwidth, which is the number the sibling sweep's own
    # gate 4 prints for this tile. Two scripts reporting "roof fraction" against
    # different denominators is the drift this study keeps hitting.
    assert report.payload["peak_roof_fraction"]["16"] == pytest.approx(0.176, abs=0.01)


def test_the_claim_gates_fail_in_the_retracted_world_and_the_validity_gates_do_not():
    report = report_at(RETRACTED)
    v = verdicts(report)
    assert [v[t] for t in ("V0", "V1", "V2", "V3", "V4")] == ["PASS"] * 5, (
        "a world this experiment is built to REJECT must still be a valid run; "
        "a validity gate failing here would mean the rejection came from the "
        "instrument and not from the data")
    assert v["C1"] == "FAIL"
    assert v["C3"] == "FAIL"
    assert report.payload["peak_roof_fraction"]["16"] == pytest.approx(0.893, abs=0.01)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_two_worlds_stay_separated_under_timing_noise(seed):
    refit = verdicts(report_at(REFIT, noise=0.02, seed=seed))
    retracted = verdicts(report_at(RETRACTED, noise=0.02, seed=seed))
    assert refit["C1"] == "PASS" and retracted["C1"] == "FAIL"
    assert refit["C3"] == "PASS" and retracted["C3"] == "FAIL"
    for tag in ("V0", "V1", "V2", "V3", "V4"):
        assert refit[tag] == "PASS", f"{tag} voided a good run at 2% spread"
        assert retracted[tag] == "PASS", f"{tag} voided a good run at 2% spread"


def test_the_control_flatness_gate_widens_with_the_measured_spread():
    """A fixed 2% flatness gate FAILED every world above 1% noise.

    The statistic is a ratio of two single cells and carries `sqrt(2)` times the
    per-cell spread, so a fixed gate below that is a coin flip -- and it is a
    VALIDITY gate, so losing the toss voids the page. Pinned because the fix is
    invisible: the gate goes on printing a threshold either way.
    """
    for noise in (0.01, 0.02, 0.03):
        assert verdicts(report_at(REFIT, noise=noise))["V2"] == "PASS"


def test_the_control_gate_stops_answering_when_the_timing_is_too_noisy():
    # Past the ceiling, widening the gate would accept a ladder still visibly
    # climbing. UNDECIDED is not PASS, and a validity gate that is not PASS
    # voids the page, which is the point.
    report = report_at(REFIT, noise=0.06)
    assert verdicts(report)["V2"] == CAP.UNDECIDED
    line = next(g for g in report.gates if g.tag == "V2")
    assert "too noisy" in " ".join(line.lines + [line.consequence])


def test_a_shallow_sweep_cannot_confirm_the_cap_by_stopping_early():
    """THE FALSE CONFIRMATION THE DEPTH GATE EXISTS TO REFUSE.

    The trap: sweep BLOCK_M=16 to a shallow depth, watch it sit far under the
    roof, and publish the cap. In the RETRACTED world it would sit far under the
    roof there too, for the sole reason that its own ceiling is approached
    slowly, so the reading confirms nothing.

    The gate and the depth are COUPLED, which is what makes the trap
    structurally impossible rather than merely watched for: C1's threshold is
    the midpoint of the two worlds, and the depth requirement is defined as the
    depth at which the retracted world exceeds that same midpoint. So a
    condition is live only where the retracted world would have failed it. Below
    that depth the gate declines to score instead of passing.
    """
    ten_tiles = report_at(RETRACTED, r_max=160)
    assert verdicts(ten_tiles)["C1"] == CAP.UNDECIDED, (
        "a depth below the horizon must decline, never PASS: a PASS there "
        "reports where the sweep stopped")
    assert verdicts(ten_tiles)["V4"] == "FAIL"
    c1 = next(g for g in ten_tiles.gates if g.tag == "C1")
    assert "no condition is testable at this depth" in c1.threshold

    # And as soon as the discriminating condition IS live, the retracted world
    # fails it. 32 treads is past the 13-tread horizon.
    caught = report_at(RETRACTED, r_max=512)
    assert verdicts(caught)["C1"] == "FAIL"


def test_the_coupling_between_the_threshold_and_the_horizon_is_exact():
    # The structural property the test above rests on, checked as arithmetic
    # rather than inferred from two runs: at the horizon depth the retracted
    # world is ABOVE the threshold, and one tread earlier it is below.
    depth = CAP.required_depth(16, b=2, ridge_band=BAND)
    for ridge in (RIDGE, RIDGE_HI):
        threshold = CAP.cap_discriminator(16, ridge, 2)
        n = depth.horizon_disc[ridge]
        above = (2.0 * n * 16 / 2) / SWEEP.q_of_tiles(n, RETRACTED) / ridge
        below = (2.0 * (n - 1) * 16 / 2) / SWEEP.q_of_tiles(n - 1, RETRACTED) / ridge
        assert below < threshold <= above


def test_a_depth_short_of_the_near_roof_horizon_does_not_earn_that_sentence():
    # 32 treads rules out alpha=0.10 and does NOT entitle anyone to "never gets
    # near the roof": the retracted world would not have got near it either at
    # that depth. The gate scores the live condition and names the dead one.
    report = report_at(REFIT, r_max=512)
    c1 = next(g for g in report.gates if g.tag == "C1")
    assert c1.verdict == "PASS"
    assert "discriminating" in c1.threshold and "near-roof" not in c1.threshold
    assert "near-roof <= 0.850 NOT TESTABLE" in " ".join(c1.lines)


def test_the_c1_threshold_does_not_move_with_the_world_being_tested():
    # The bug this pins: `cap_discriminator` used to take the run's alpha, so
    # `--self-test 0.10` scored C1 against the midpoint of the retracted world
    # with ITSELF -- 0.998 -- and printed a threshold either way.
    refit = next(g for g in report_at(REFIT).gates if g.tag == "C1")
    retracted = next(g for g in report_at(RETRACTED).gates if g.tag == "C1")
    assert refit.threshold == retracted.threshold
    assert "0.589" in refit.threshold


def test_the_measured_alpha_is_a_lower_bound_and_so_the_measured_cap_is_upper():
    # Both biases run the same way: the fused layer's fixed cost sits in alpha's
    # denominator and the activation correction comes out of its numerator. C2
    # claims the ceiling is LOW, so it must be scored on the high estimate.
    report = report_at(REFIT)
    assert report.payload["alpha_corrected"] < REFIT
    assert report.payload["ai_cap_measured"] > SWEEP.ai_cap(16, REFIT, 2)
    # ...and still far under the ridge, which is the claim.
    assert report.payload["ai_cap_measured"] < 0.5 * RIDGE


def test_c2_refuses_to_name_an_alpha_when_the_ladder_cannot_identify_one():
    # In the retracted world BLOCK_M=16 sits ON its own crossing (B/C =
    # ridge/cap = 1.002), the ladder fit discards a memory branch parallel to
    # the compute branch, and there is no alpha to quote. UNDECIDED, never a
    # number: an invented 0.6 that comes out the same under every hypothesis is
    # the most dangerous number this experiment could print.
    report = report_at(RETRACTED)
    c2 = next(g for g in report.gates if g.tag == "C2")
    assert c2.verdict == CAP.UNDECIDED
    assert report.payload["alpha_corrected"] is None
    assert report.payload["ai_cap_measured"] is None


def test_the_gate_thresholds_sit_between_the_two_worlds():
    # Written as a check rather than trusted: a discriminator computed from the
    # two registered alphas has to lie strictly between what they predict, or it
    # is beside one of them and cannot tell them apart.
    disc = CAP.cap_discriminator(16, RIDGE, 2)
    refit_frac = SWEEP.ai_cap(16, REFIT, 2) / RIDGE
    retracted_frac = SWEEP.ai_cap(16, RETRACTED, 2) / RIDGE
    assert refit_frac < disc < retracted_frac
    assert CAP.ROOF_FRACTION < retracted_frac


# --------------------------------------------------------------------------
# Refusals and non-vacuity.
# --------------------------------------------------------------------------

def test_no_aligned_cell_at_the_cap_tile_refuses_instead_of_scoring_zero():
    _, cells = cells_at(REFIT)
    kept = [c for c in cells if not (c.block_m == 16 and c.aligned)]
    with pytest.raises(CAP.Unmeasurable) as exc:
        CAP.analyse(kept, MIXTRAL, cap_tile=16, control_tile=256, alpha=REFIT,
                    ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2,
                    model_name="mixtral-8x7b", dtype="bf16",
                    compiles={16: 1, 256: 1}, executed={16: 1, 256: 1},
                    sm_count=132, sm_source="test",
                    depth=CAP.required_depth(16, b=2, ridge_band=BAND), planned_cells=len(kept),
                    header=[])
    # 0.0 would read as "never got near the roof", which is the verdict this
    # experiment exists to earn rather than to assume.
    assert "0.0" in str(exc.value) or "no exactly-full" in str(exc.value)


def test_a_grid_with_holes_in_it_fails_the_non_vacuity_gate():
    grid, cells = cells_at(REFIT)
    planned = len(grid) * 2
    thinned = [c for c in cells if c.rows_per_expert % 128 == 0]
    report = CAP.analyse(
        thinned, MIXTRAL, cap_tile=16, control_tile=256, alpha=REFIT,
        ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2, model_name="mixtral-8x7b",
        dtype="bf16", compiles={16: 1, 256: 1}, executed={16: 1, 256: 1},
        sm_count=132, sm_source="test", depth=CAP.required_depth(16, b=2, ridge_band=BAND),
        planned_cells=planned, header=[])
    assert verdicts(report)["V1"] == "FAIL"
    v1 = next(g for g in report.gates if g.tag == "V1")
    assert f"of {planned} planned cells measured" in " ".join(v1.lines)


def test_a_control_that_never_reached_a_roof_voids_the_page():
    # Two treads at the control is under `compute_reference`'s three, so no
    # ladder qualifies as a compute branch and nothing in the sweep was shown to
    # reach a roof. C1's absence is then not evidence of absence.
    _, cells = cells_at(REFIT, r_max=512)
    report = CAP.analyse(
        cells, MIXTRAL, cap_tile=16, control_tile=256, alpha=REFIT, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, model_name="mixtral-8x7b", dtype="bf16",
        compiles={16: 1, 256: 1}, executed={16: 1, 256: 1}, sm_count=132,
        sm_source="test", depth=CAP.required_depth(16, b=2, ridge_band=BAND),
        planned_cells=len(cells), header=[])
    assert verdicts(report)["V2"] == "FAIL"


def test_a_sweep_where_nothing_reached_the_roof_voids_c1_and_keeps_c2():
    """The state the sibling found in all 26 published reports.

    Their plateaus ran 46.5-75.6% of the card's own `ridge x bandwidth`, so
    nothing in any of those sweeps reached a compute roof. Simulated by slowing
    every cell by the same factor, which leaves the ladder SHAPES untouched --
    V2 still passes, the control is still proportional and still flat -- and
    moves only the LEVEL.

    C1 compares a throughput with the roof, so it is void. C2 fits the cap
    tile's own re-read fraction and never needs a roof, so it stands. That
    split is the whole reason C2 is in the report, and a V3 whose consequence
    said "the page is void" would throw the surviving claim away with the dead
    one.
    """
    _, cells = cells_at(REFIT)
    slowed = [SWEEP.make_cell(MIXTRAL, c.rows_per_expert, c.block_m,
                              c.ms_p50 * 2.0, sm_count=132, block_n=64,
                              ms_min=c.ms_min * 2.0, ms_stdev=c.ms_stdev * 2.0)
              for c in cells]
    report = CAP.analyse(
        slowed, MIXTRAL, cap_tile=16, control_tile=256, alpha=REFIT, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, model_name="mixtral-8x7b", dtype="bf16",
        compiles={16: 1, 256: 1}, executed={16: 1, 256: 1}, sm_count=132,
        sm_source="test", depth=CAP.required_depth(16, b=2, ridge_band=BAND),
        planned_cells=len(slowed), header=[])
    v = verdicts(report)
    assert v["V2"] == "PASS", "the shapes did not change, only the level"
    assert v["V3"] == "FAIL"
    v3 = next(g for g in report.gates if g.tag == "V3")
    assert "C2 SURVIVES" in v3.consequence
    assert report.payload["plateau_tflops"] / report.payload["model_roof_tflops"] < 0.6
    assert v["C2"] == "PASS"


def test_the_control_gate_cannot_be_satisfied_by_the_sweeps_own_maximum():
    # The vacuity this replaced: against the plateau -- the maximum over the
    # same cells -- some block size always scores 1.00, so the control could
    # never fail. Against ridge x bandwidth it is a real question, and this
    # pins that the denominator in the payload is the roof and not the plateau.
    report = report_at(REFIT)
    plateau = report.payload["plateau_tflops"]
    roof = report.payload["model_roof_tflops"]
    assert roof == pytest.approx(RIDGE * BANDWIDTH * 1e9 / 1e12)
    assert report.payload["peak_roof_fraction"]["256"] == pytest.approx(
        plateau / roof, abs=0.005)


def test_every_gate_states_what_a_fail_costs():
    # Structural. A gate whose consequence is empty is a PASS/FAIL nobody can
    # act on, and this is the check that stops one being added.
    for report in (report_at(REFIT), report_at(RETRACTED)):
        for gate in report.gates:
            assert gate.consequence.strip(), gate.tag
            assert gate.threshold.strip(), gate.tag
            assert gate.measured.strip(), gate.tag
            assert gate.kind in (CAP.VALIDITY, CAP.CLAIM)


def test_the_gate_set_is_the_one_the_report_claims_to_run():
    report = report_at(REFIT)
    tags = [g.tag for g in report.gates]
    assert tags == ["V0", "V1", "V2", "V3", "V4", "C1", "C2", "C3", "C4"]
    # No gate here is the sibling's crossing-shift gate under another name.
    for gate in report.gates:
        assert "crossing shift" not in gate.claim


# --------------------------------------------------------------------------
# The run id, the paths and hermeticity.
# --------------------------------------------------------------------------

def _args(**over):
    parser = CAP.build_parser()
    argv = []
    for key, value in over.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return parser.parse_args(argv)


def test_the_run_id_moves_with_every_knob_that_changes_a_measured_cell():
    base = _args()
    base_id = CAP.default_run_id(base, 2112, "nvidia_h200")
    assert CAP.default_run_id(_args(), 2112, "nvidia_h200") == base_id
    for knob, value in (("cap_tile", 32), ("control", 128), ("row_step", 64),
                        ("step_probes", 3), ("seed", 7), ("group_m", 16),
                        ("block_n", 256), ("num_stages", 3),
                        ("model", "qwen2-57b-a14b"), ("dtype", "fp16"),
                        # THE THREE THAT WERE MISSING UNTIL 2026-09-02. None of
                        # them is an analysis knob: each changes the measured
                        # milliseconds of every cell, and run_sweep resumes from
                        # cells.csv on (BLOCK_M, tokens), so an --iters 200
                        # re-run after --iters 50 landed in the same directory,
                        # skipped all 162 cells and printed the 50-iteration
                        # timings under the 200-iteration label.
                        ("iters", 200), ("warmup", 200),
                        ("cell_budget_ms", 9999)):
        assert CAP.default_run_id(
            _args(**{knob: value}), 2112, "nvidia_h200") != base_id, knob
    # ...and with the resolved r_max, not the `0` that asks for it to be derived.
    assert CAP.default_run_id(base, 832, "nvidia_h200") != base_id


def test_two_cards_cannot_share_one_directory():
    """THE CARD IS SWEPT BY THE OPERATOR MOVING PODS.

    `$MOE_RESULTS_DIR` defaults to `/workspace/results`, a RunPod network volume
    that outlives the pod, and both claim gates are scored against a per-card
    ridge -- 145.7 on the A100 against 162.8 on the H200. Without the card in
    the id the second card resumes the first's directory, skips all 162 cells,
    spends no GPU time and reports the first card's timings against its own
    ridge. That is already committed once in this repo: the A100 and H200
    cross-card arms carry identical report filenames.
    """
    h200 = CAP.default_run_id(_args(), 2112, "nvidia_h200")
    a100 = CAP.default_run_id(_args(), 2112, "nvidia_a100_sxm4_80gb")
    assert h200 != a100
    assert h200.startswith("nvidia_h200-")               # visible in `ls`
    assert a100.startswith("nvidia_a100_sxm4_80gb-")


def test_a_self_test_cannot_land_in_a_measured_run_s_directory():
    """A free laptop command must not overwrite a metered run's only artefact.

    `--self-test` writes `report.json` like any other run. With the planted
    alpha out of the key that was the SAME directory the pod run uses, so one
    `--self-test 0.10` replaced a measurement with a synthetic report carrying
    the retracted alpha this experiment exists to exclude -- and nothing in the
    file said it was synthetic.
    """
    measured = CAP.default_run_id(_args(), 2112, "nvidia_h200")
    refit = CAP.default_run_id(_args(self_test=0.558), 2112, "nvidia_h200")
    retracted = CAP.default_run_id(_args(self_test=0.10), 2112, "nvidia_h200")
    noisy = CAP.default_run_id(
        _args(self_test=0.558, self_test_noise=0.02), 2112, "nvidia_h200")
    assert len({measured, refit, retracted, noisy}) == 4
    for planted in (refit, retracted, noisy):
        assert planted.startswith("synthetic-")          # visible in `ls`
    assert not measured.startswith("synthetic-")


def test_the_run_id_does_not_move_with_the_analysis_parameters():
    # alpha and ridge re-analyse a set of cells; they do not change one. Two
    # analyses of one sweep belong in one directory, and putting them in two
    # would re-measure 80 s of H200 to get the same numbers.
    base_id = CAP.default_run_id(_args(), 2112, "nvidia_h200")
    assert CAP.default_run_id(_args(alpha=0.33), 2112, "nvidia_h200") == base_id
    assert CAP.default_run_id(_args(ridge=176.2), 2112, "nvidia_h200") == base_id
    assert CAP.default_run_id(
        _args(ridge_band="150,170"), 2112, "nvidia_h200") == base_id


def test_the_self_test_is_hermetic_and_does_not_read_this_machine(tmp_path, monkeypatch):
    """A replay that reads the hardware is not a replay.

    `resolve_bandwidth` prefers THIS machine's calibration, which would make the
    same `--self-test 0.558` generate different cells on the pod than on a
    laptop. Pinned by making the calibration return an absurd number and
    checking nothing on the page moves.
    """
    def absurd(_args):
        return 1.0, "a calibration that must not be read"
    monkeypatch.setattr(SWEEP, "resolve_bandwidth", absurd)
    out = tmp_path / "a"
    rc = CAP.main(["--self-test", "0.558", "--out", str(out)])
    assert rc == 0
    payload = json.loads(
        next(out.rglob("report.json")).read_text())
    assert payload["plateau_tflops"] == pytest.approx(698.7, abs=1.0)
    assert payload["peak_roof_fraction"]["16"] == pytest.approx(0.177, abs=0.01)


def test_the_run_writes_where_it_said_and_the_dry_run_writes_nothing(tmp_path, capsys):
    assert CAP.main(["--dry-run", "--out", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []
    plan = capsys.readouterr().out
    assert "estimated GPU time" in plan
    assert "PREDICTIONS, registered before the run" in plan
    # The predictions are on the terminal BEFORE any measurement, which is the
    # only ordering that makes "registered" a property of the transcript.
    assert plan.index("PREDICTIONS") < plan.index("estimated GPU time")


def test_exit_codes_separate_a_void_run_from_a_falsified_claim(tmp_path):
    # 0: the page is readable whatever the claims said. 1: a validity gate did
    # not pass. Confusing the two is how a broken run gets published as a
    # negative result.
    assert CAP.main(["--self-test", "0.10", "--out", str(tmp_path / "b")]) == 0
    assert CAP.main(["--self-test", "0.10", "--r-max", "512",
                     "--out", str(tmp_path / "c")]) == 1
    assert CAP.main(["--cap-tile", "8", "--dry-run",
                     "--out", str(tmp_path / "d")]) == 2


class MovedSibling:
    """The sibling as it would look after someone renamed or re-signed a piece.

    Delegates everything to the real module except the names it is told to hide
    or replace, so a test can move ONE thing and leave the other forty alone.
    """

    def __init__(self, real, *, missing=(), replaced=None):
        self._real = real
        self._missing = set(missing)
        self._replaced = dict(replaced or {})

    def __getattr__(self, name):
        if name in self._missing:
            raise AttributeError(name)
        if name in self._replaced:
            return self._replaced[name]
        return getattr(self._real, name)


def test_the_sibling_api_probe_passes_today_and_can_actually_fire():
    """A guard that cannot fire is worse than no guard.

    This runner calls into `block_m_crossing_sweep` for everything metered, and
    that file is edited independently -- `compute_reference` gained four
    required keyword arguments during this one's first afternoon. The probe
    turns that into a refusal before the pod is billed instead of a TypeError
    after the sweep. Both halves are checked: it passes against the sibling as
    it stands, and it FAILS against a sibling that moved.
    """
    CAP.require_sweep_api()          # today, against the real module

    def re_signed(cells, block_sizes, max_err=0.05):
        raise AssertionError("never called")

    real = CAP.SWEEP
    for moved, expect in (
            (MovedSibling(real, replaced={"compute_reference": re_signed}),
             "compute_reference"),
            (MovedSibling(real, missing={"_throughput_ladder"}),
             "_throughput_ladder"),
            (MovedSibling(real, missing={"FIXED"}), "FIXED")):
        CAP.SWEEP = moved
        try:
            with pytest.raises(CAP.SiblingChanged) as exc:
                CAP.require_sweep_api()
            assert expect in str(exc.value)
        finally:
            CAP.SWEEP = real
    CAP.require_sweep_api()          # and the module is left as it was


def test_a_moved_sibling_refuses_before_the_pod_is_billed(tmp_path, capsys):
    real = CAP.SWEEP
    CAP.SWEEP = MovedSibling(real, missing={"build_grid", "FIXED"})
    try:
        rc = CAP.main(["--dry-run", "--out", str(tmp_path)])
    finally:
        CAP.SWEEP = real
    # 2, not 1: nothing was measured and nothing may be read, which is a
    # different state from a run whose claim gate came out the other way.
    assert rc == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and "build_grid" in out
    # And it refused before argparse touched the sibling's FIXED defaults,
    # which is the only ordering that makes the probe useful.
    assert "AttributeError" not in out


# --------------------------------------------------------------------------
# scripts/check_mma_path.sh, forced-tile mode.
# --------------------------------------------------------------------------

def run_mma(*argv):
    return subprocess.run(["bash", str(MMA), *argv], capture_output=True,
                          text=True, cwd=ROOT)


def test_the_forced_plan_registers_a_prediction_per_tile():
    out = run_mma("--block-m", "16,64", "--tokens", "256", "--dry-run")
    assert out.returncode == 0
    assert "BLOCK_SIZE_M=16   wgmma == 0" in out.stdout
    assert "BLOCK_SIZE_M=64   wgmma  > 0" in out.stdout
    # The prediction has to name why, or it is an assertion: num_warps is the
    # other half of Triton's predicate and it is pinned so it cannot move.
    assert "num_warps % 4 == 0" in out.stdout
    assert "num_warps=8" in out.stdout
    assert "nothing was executed" in out.stdout


def test_the_forced_mode_refuses_a_comparison_that_cannot_show_a_switch():
    out = run_mma("--block-m", "64,128", "--dry-run")
    assert out.returncode == 2
    assert "SAME side of the predicate" in out.stderr


def test_the_forced_mode_refuses_one_arm_and_a_split_batch():
    one = run_mma("--block-m", "16", "--dry-run")
    assert one.returncode == 2
    assert "at least two tiles" in one.stderr
    split = run_mma("--block-m", "16,64", "--tokens", "16,256", "--dry-run")
    assert split.returncode == 2
    # A token count per arm puts back exactly the confound the mode removes.
    assert "FIXED token count" in split.stderr


def test_the_forced_mode_refuses_a_tile_that_is_not_a_number():
    out = run_mma("--block-m", "16,x", "--dry-run")
    assert out.returncode == 2
    assert "not a number" in out.stderr


def test_the_unforced_plan_says_the_ladder_cannot_attribute_the_instruction():
    out = run_mma("--dry-run")
    assert out.returncode == 0
    assert "vLLM's own config ladder" in out.stdout
    assert "--block-m 16,64" in out.stdout


def test_the_script_reads_the_schema_columns_that_actually_exist():
    """Structural guard on the observed-tile reader inside the shell script.

    The forced mode's whole validity gate is "the six OBSERVED tile columns
    equal the six forced values". Those column names live in a heredoc inside a
    shell script, where nothing type-checks them: rename one in the schema and
    the reader silently reports `none`, G1 fails, and the failure looks like an
    unhonoured pin rather than a rename.
    """
    text = MMA.read_text()
    for column in CONFIG_KEY_TO_COLUMN.values():
        assert f'"{column}"' in text, column
    assert '"tile_config_source"' in text
    from moe.bench import schema as SC
    for column in list(CONFIG_KEY_TO_COLUMN.values()) + ["tile_config_source"]:
        assert column in SC.COLUMNS, column
    # And the forced value it compares against must be a legal source string.
    from moe.bench import force_tile as FT
    assert FT.TILE_SOURCE_OVERRIDE == "vllm_override"
    assert "src=vllm_override" in text


def test_the_forced_config_carries_exactly_the_keys_the_hook_requires():
    """`MOE_FORCE_TILE` is parsed with an exact key set, checked here not there.

    `force_tile.parse` refuses a config that is missing a key or carries an
    extra one, because vLLM's override REPLACES the tuned dict and the result is
    splatted into the kernel launch. The JSON is built by string interpolation
    in the shell, so nothing else would catch a typo until the pod.
    """
    text = MMA.read_text()
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("force="))
    payload = line.split("{", 1)[1].rsplit("}", 1)[0]
    keys = {part.split(":")[0].strip().strip('\\"')
            for part in payload.split(",")}
    from moe.bench import force_tile as FT
    assert keys == set(FT.REQUIRED_KEYS)


def test_the_pinned_warp_count_cannot_move_the_predicate():
    # If num_warps were not a multiple of 4, `BLOCK_M % 64 == 0` would stop
    # being the only live term and the arms would differ in two ways at once,
    # which is the confound the whole mode exists to remove.
    text = MMA.read_text()
    warps = next(ln for ln in text.splitlines() if ln.startswith("NUM_WARPS="))
    assert int(warps.split('"')[1]) % 4 == 0


# --------------------------------------------------------------------------
# The ridge, the card and git: three numbers that used to be asserted instead
# of measured. All three are failure mode 6 (a constant from documentation
# presented where a measurement belongs) or failure mode 3 (output written
# where .gitignore silently drops it).
# --------------------------------------------------------------------------

def test_a_measured_run_with_no_calibration_refuses_rather_than_borrowing_a_ridge():
    """The A100 defect, in one assertion.

    `--ridge` used to default to `RIDGE_BAND[0]` = 160.3, a 2026-08-26 H200
    constant, while `resolve_bandwidth` beside it read the ATTACHED card. On an
    A100 that assembles `ridge x bandwidth` out of two machines: 160.3 x 1799.4
    GB/s implies a 288.4 TFLOP/s roof against the card's own measured 262.4, and
    both claim gates -- C1's roof fraction and C2's cap/ridge -- come out
    160.3/145.8 = 1.10x wrong. The full 162-cell sweep would still run and the
    page would still print PASS/FAIL, with nothing on it saying so.
    """
    rc = CAP.main(["--capability", "9.0"])
    assert rc == 2                                       # nothing measured


def test_the_refusal_can_be_answered_by_asserting_a_ridge_on_the_command_line():
    # The operator's own assertion is allowed, because it lands in the run's
    # command line where a reader can see it. It is the SILENT constant that is
    # forbidden.
    args = CAP.build_parser().parse_args(["--ridge", "145.7",
                                          "--ridge-band", "140,150"])
    rr = SWEEP.resolve_ridge(args, synthetic=False)
    assert rr.ridge == pytest.approx(145.7)
    assert rr.band == (140.0, 150.0)
    assert "command line" in rr.source


def test_the_depth_and_the_predictions_are_computed_at_the_resolved_band():
    """The plan bought at 160.3 is not the plan an A100 needs.

    The default `--r-max` is the depth at which the retracted world would trip
    C1, and that horizon is set by the ridge: 52 tiles at 160.3 and 132 at
    176.2. On a card whose ridge is 145.7 the retracted ceiling 160.0 sits ABOVE
    it and the arithmetic is different again, so a depth planned against the
    module constant buys the wrong number of cells.
    """
    h200 = CAP.required_depth(16, b=2, ridge_band=CAP.RIDGE_BAND)
    a100 = CAP.required_depth(16, b=2, ridge_band=(145.7, 145.7))
    assert h200.tiles != a100.tiles
    assert set(h200.horizon_roof) == set(CAP.RIDGE_BAND)
    assert set(a100.horizon_roof) == {145.7}


def test_the_prediction_block_names_where_its_ridge_came_from():
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    depth = CAP.required_depth(16, b=2, ridge_band=(145.7, 145.7))
    lines = CAP.prediction_lines(
        cfg, cap_tile=16, control_tile=256, alpha=CAP.ALPHA, ridge=145.7, b=2,
        bandwidth_gbps=1799.4, depth=depth, r_max=depth.rows,
        ridge_band=(145.7, 145.7),
        ridge_source="measured on this device: NVIDIA A100-SXM4-80GB",
        band_source="one calibration, one pattern")
    text = "\n".join(lines)
    assert "145.7" in text
    assert "measured on this device" in text
    # ...and the H200 constant is not silently in the table beside it.
    assert "160.3-176.2" not in text


def test_git_visibility_asks_git_and_distinguishes_all_three_answers():
    """rc 0 ignored, rc 1 kept, anything else UNVERIFIED.

    The sentence this replaced -- "results/* is gitignored except
    results/published/, so this run commits nothing" -- printed unchanged for
    `--out results/published/<arm>`, where the output IS committed, and for a
    path outside the work tree, which git has no opinion about at all.
    """
    ignored = CAP.git_visibility(ROOT / "results" / "tile_cap" / "x")
    kept = CAP.git_visibility(ROOT / "results" / "published" / "x")
    assert "IGNORED by git" in ignored
    assert "WILL KEEP" in kept
    assert ignored != kept


def test_git_visibility_says_unverified_rather_than_tracked_when_it_cannot_ask():
    """A refusal, never a default.

    The pod writes to `/workspace/results`, outside any work tree, where `git
    check-ignore` exits 128. Reporting that as "tracked" is the same defect in
    the other direction, and it is the one that loses files.
    """
    note = CAP.git_visibility(Path("/definitely/not/in/this/work/tree"))
    assert "UNVERIFIED" in note
    assert "WILL KEEP" not in note


def test_a_synthetic_report_says_so_in_the_only_machine_readable_artefact(tmp_path):
    """`report.json` must not be mistakable for a measurement.

    A `--self-test 0.10` report carries the retracted alpha, 162 measured cells
    and a zero timing spread. Without `synthetic` there is nothing in the file
    that separates it from a pod run that measured the retracted world.
    """
    out = tmp_path / "r"
    assert CAP.main(["--self-test", "0.10", "--out", str(out)]) == 0
    payload = json.loads(next(out.rglob("report.json")).read_text())
    assert payload["synthetic"] is True
    assert payload["card"] == CAP.NO_CARD_SLUG
    assert "PINNED for --self-test" in payload["ridge_source"]
    # And it landed under a directory whose name says so, so it cannot have
    # overwritten a measured run's report.
    assert next(out.rglob("report.json")).parent.name.startswith("synthetic-")


def test_the_self_test_ridge_is_pinned_and_does_not_read_this_machine(monkeypatch):
    """A replay that reads the hardware is not a replay.

    Same rule the bandwidth already followed. If `--self-test` resolved the
    ridge, the same command would generate different cells on the pod than on a
    laptop and the suite could not pin either.
    """
    def absurd(_args, *, synthetic):
        raise AssertionError("a self-test must not resolve a ridge")
    monkeypatch.setattr(SWEEP, "resolve_ridge", absurd)
    assert CAP.main(["--self-test", "0.558", "--dry-run"]) == 0


# --------------------------------------------------------------------------
# The demotion. `tile_cap_test` used to be framed as the production claim; the
# observed-tile arm says vLLM never runs BLOCK_M=16 multi-tile, so the cap it
# measures is real and never approached. That is a COUNT off a published file,
# so it is recomputed here rather than quoted.
# --------------------------------------------------------------------------

OBSERVED_ARM = (ROOT / "results" / "published"
                / "2026-09-01-nvidia_h200-alpha-0558" / "merged.csv")


def _observed_from_the_published_arm() -> dict[int, tuple[int, int, int]]:
    """Recount `OBSERVED_MULTI_TILE` from the CSV it claims to come from.

    The one published arm carrying `tile_block_m` -- the tile vLLM actually
    chose -- de-duplicated to distinct cells, vLLM rows only, rows per expert
    read as `load_mean_rows`.
    """
    rows = [r for r in csv.DictReader(OBSERVED_ARM.open())
            if r["tile_config_source"].startswith("vllm")]
    seen = {}
    for r in rows:
        key = (r["model"], r["dtype"], r["num_tokens"], r["routing_kind"],
               r["routing_param"], r["tile_config_source"])
        seen.setdefault(key, r)
    by = defaultdict(list)
    for r in seen.values():
        bm = int(float(r["tile_block_m"]))
        by[bm].append(max(1, math.ceil(float(r["load_mean_rows"]) / bm)))
    return {bm: (sum(1 for n in t if n > 1), len(t), max(t))
            for bm, t in by.items()}


def test_the_observed_tile_table_is_what_the_published_arm_says():
    """The count that demotes this experiment, recomputed rather than quoted.

    Failure mode 6 in this project is a constant from documentation sitting
    where a measurement belongs, and the whole "BLOCK_M=16 is a formula test,
    not a production claim" rewrite rests on one table of counts. So the table
    is recounted from the file it names.
    """
    observed = _observed_from_the_published_arm()
    # NON-VACUITY: a mis-typed column or a filter that matched nothing would
    # produce an empty table and an empty comparison passes.
    assert sum(cells for _, cells, _ in observed.values()) == 132
    assert observed == CAP.OBSERVED_MULTI_TILE
    multi, cells, top = observed[16]
    assert (multi, top) == (0, 1) and cells > 0      # the tile under test
    assert observed[128][0] > 0 and observed[128][2] > 1   # the live regime


def test_the_note_refuses_for_a_tile_height_nobody_observed():
    """"No cell ran multi-tile" and "no cell was observed" are different.

    Printing the second as the first would let an unmeasured tile inherit the
    demotion's conclusion, which is the same defect as a borrowed ridge.
    """
    unobserved = CAP.observed_note(48)
    assert "UNMEASURED" in unobserved
    assert "ONE M-tile" not in unobserved
    assert "24 of 24" in CAP.observed_note(16)
    assert "59 of 87" in CAP.observed_note(128)


def test_the_plan_says_this_is_not_the_production_claim_before_anything_runs():
    """Registered in the plan, not conceded in the discussion afterwards."""
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    depth = CAP.required_depth(16, b=2, ridge_band=BAND)
    text = "\n".join(CAP.prediction_lines(
        cfg, cap_tile=16, control_tile=256, alpha=REFIT, ridge=RIDGE, b=2,
        bandwidth_gbps=BANDWIDTH, depth=depth, r_max=depth.rows,
        ridge_band=BAND, ridge_source="test", band_source="test"))
    assert "NOT A PRODUCTION CLAIM" in text
    assert "bm128_depth.py" in text


def test_the_headline_sentence_no_longer_overreaches_from_this_tile():
    """A PASS at BLOCK_M=16 must not print as a statement about a shipped kernel."""
    report = report_at(REFIT)
    text = report.text()
    assert "READING IT." in text
    assert "the cap formula holds at this tile height" in text.lower()
    assert "WHAT IT DOES NOT SAY" in text
    assert "bm128_depth.py" in text


# --------------------------------------------------------------------------
# ...and the last place the module's H200 band could still reach a report.
# --------------------------------------------------------------------------

def test_the_depth_cannot_be_computed_without_stating_a_band():
    """`required_depth` has no default band, so omission cannot borrow one.

    `--ridge` was fixed to resolve from the attached device, but the depth is a
    function of the ridge too, and a defaulted `ridge_band=RIDGE_BAND` one call
    below the fix would have planned an A100 run against the H200's horizons
    while the report's own ridge field said 145.8.
    """
    with pytest.raises(TypeError):
        CAP.required_depth(16, b=2)


def test_an_unstated_band_is_degenerate_and_never_the_module_constant():
    """One calibration, said twice: honest. Another machine's band: not."""
    grid, cells = cells_at(REFIT)
    report = CAP.analyse(
        cells, MIXTRAL, cap_tile=16, control_tile=256, alpha=REFIT, ridge=145.8,
        bandwidth_gbps=1799.4, b=2, model_name="mixtral-8x7b", dtype="bf16",
        compiles={bm: 1 for bm in TILES}, executed={bm: 1 for bm in TILES},
        sm_count=108, sm_source="test",
        depth=CAP.required_depth(16, b=2, ridge_band=(145.8, 145.8)),
        planned_cells=len(grid) * len(TILES), header=[])
    assert report.payload["ridge_band"] == [145.8, 145.8]
    assert CAP.RIDGE_BAND[1] not in report.payload["ridge_band"]

