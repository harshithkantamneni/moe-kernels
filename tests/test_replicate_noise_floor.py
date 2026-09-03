"""The noise floor, and the five ways a noise floor lies.

`scripts/replicate_noise_floor.py` produces one number that other scripts are
meant to import and score their effects against. A number with that job has
exactly five failure modes worth a test file, and every test below belongs to one
of them:

1. THE FLOOR IS ZERO BECAUSE NOTHING VARIED. Six replicates that resumed into one
   directory report a between-replicate sd of exactly 0.0000, and every effect in
   the study then clears it. The run id is what prevents that. The sibling sweep's
   own id omitted the CARD until 2026-09-02 -- the A100 and H200 cross-card arms
   are committed under identical filenames because of it -- and now carries it;
   the REPLICATE index is still ours alone, because the sweep has no notion of
   one. There is a test below for both halves.

2. THE FLOOR IS ABSENT AND SOMETHING RETURNED 0.0 ANYWAY. `noise_floor()` must
   raise on every flavour of missing: no file, `replicate_floor: null`, a
   synthetic rehearsal, a field nobody measured.

3. THE ARITHMETIC IS A z-SCORE WEARING A t's CLOTHES. Every MDE is a quantile
   times a standard error, and the quantiles are computed here from scratch
   because the repo has no scipy. They are checked against printed tables. The
   one substitution that would flatter the study -- using a two-sample MDE at
   n=2 to stand in for the study's real n=1 design, which makes the limit SMALLER
   -- has its own test.

4. CELLS THAT DISAGREE GET AVERAGED INTO ONE FLOOR. Pooling is only legitimate
   when the spreads match; a wild cell pooled with a tight one publishes a floor
   that declares its own outlier resolvable.

5. THE CONTROL SAYS WHAT WE WISH IT SAID. Part (b) is arithmetic over committed
   reports, so its numbers are checkable exactly, and they are: +0.0101 for
   num_stages against +0.0117 for the card, on the same 11 cells.

Nothing here needs a GPU. The parts that read `results/published` skip cleanly if
it is not checked out, and a module-level guard makes "everything skipped" a
visible state rather than a green run.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script():
    """Load by path. `scripts/` is not a package, same as every other script test.

    Registered in sys.modules BEFORE exec because `@dataclass` resolves its
    annotations through `sys.modules[cls.__module__]`.
    """
    spec = importlib.util.spec_from_file_location(
        "replicate_noise_floor", ROOT / "scripts" / "replicate_noise_floor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NF = _load_script()

PUBLISHED = ROOT / "results" / "published"
HAVE_ARMS = all(p.exists() for p in NF.STAGES_CONTROL_ARMS + NF.CROSS_CARD_ARMS)
needs_arms = pytest.mark.skipif(
    not HAVE_ARMS, reason="the committed alpha-surface arms are not checked out")


def spread(arm: str, block_m: int, values) -> NF.CellSpread:
    return NF.CellSpread(arm, block_m, NF.PRIMARY_FIELD, tuple(values))


# --- 3. the arithmetic ------------------------------------------------------

@pytest.mark.parametrize("df,want", [(1, 12.706), (2, 4.303), (4, 2.776),
                                     (6, 2.447), (8, 2.306), (10, 2.228),
                                     (14, 2.145), (20, 2.086), (30, 2.042)])
def test_t_upper_quantile_matches_the_printed_table(df, want):
    """Every MDE is this number times a standard error. Wrong here, wrong there."""
    assert NF.student_t_ppf(0.975, df) == pytest.approx(want, abs=5e-4)


@pytest.mark.parametrize("df,want", [(4, 0.941), (6, 0.906), (10, 0.879), (20, 0.860)])
def test_t_power_quantile_matches_the_printed_table(df, want):
    assert NF.student_t_ppf(0.80, df) == pytest.approx(want, abs=5e-4)


@pytest.mark.parametrize("df,want", [(2, 0.0506), (4, 0.484), (5, 0.831),
                                     (7, 1.690), (10, 3.247), (20, 9.591)])
def test_chi_square_lower_quantile_matches_the_printed_table(df, want):
    """The sd upper bound divides by this. It is what turns N=6 into '1.44x'."""
    assert NF.chi2_ppf(0.025, df) == pytest.approx(want, rel=2e-3)


def test_normal_quantiles():
    assert NF.normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert NF.normal_ppf(0.80) == pytest.approx(0.841621, abs=1e-5)


def test_t_converges_to_normal_at_large_df():
    assert NF.student_t_ppf(0.975, 100000) == pytest.approx(1.959964, abs=1e-3)


def test_t_cdf_is_symmetric():
    assert NF.student_t_cdf(-1.7, 9) == pytest.approx(1.0 - NF.student_t_cdf(1.7, 9))


def test_mde_falls_as_replicates_rise():
    values = [NF.mde_two_sample(0.02, n) for n in range(2, 12)]
    assert values == sorted(values, reverse=True)


def test_mde_two_sample_refuses_a_single_replicate():
    """n=1 has zero degrees of freedom. Returning something anyway is the bug."""
    with pytest.raises(ValueError):
        NF.mde_two_sample(0.02, 1)


def test_mde_paired_refuses_one_cell():
    with pytest.raises(ValueError):
        NF.mde_paired(0.04, 1)


def test_mde_refuses_a_non_positive_sd():
    """A zero sd would divide the study's every effect by nothing and declare it
    resolvable. The one place a floor of 0.0 could leak in."""
    with pytest.raises(ValueError):
        NF.mde_two_sample(0.0, 6)
    with pytest.raises(ValueError):
        NF.mde_external_sigma(0.0, 1)


def test_the_known_sigma_limit_is_the_stricter_of_the_two_for_gate_c2():
    """THE SUBSTITUTION GATE C2 DOES NOT MAKE, and which way it would have gone.

    The study ran one run per condition, where `mde_two_sample` has zero df and
    cannot be evaluated. Quoting it at n=2 instead gives 5.363 sigma, because the
    t quantile at 2 df is 4.303. The known-sigma form at n=1 is 3.962 sigma. C2
    claims an observed difference is BELOW the limit, so the larger number is the
    easier gate; C2 therefore uses the smaller one.
    """
    assert NF.mde_external_sigma(1.0, 1) == pytest.approx(3.962, abs=2e-3)
    assert NF.mde_two_sample(1.0, 2) == pytest.approx(5.363, abs=2e-3)
    for sd in (0.005, 0.0228, 0.05, 0.2):
        assert NF.mde_external_sigma(sd, 1) < NF.mde_two_sample(sd, 2)
        assert NF.mde_external_sigma(sd, 1) > NF.mde_external_sigma(sd, 2)


def test_the_registered_prior_reproduces_the_docstrings_mde_table():
    """The numbers the docstring registered N=6 against, recomputed."""
    assert NF.PRIOR_SD == pytest.approx(0.02284, abs=5e-6)
    assert NF.mde_two_sample(NF.PRIOR_SD, 3) == pytest.approx(0.0693, abs=5e-4)
    assert NF.mde_two_sample(NF.PRIOR_SD, 6) == pytest.approx(0.0410, abs=5e-4)
    assert NF.replicates_for(NF.EFFECTS[-1].size, NF.PRIOR_SD) == 61


def test_replicates_for_returns_none_rather_than_the_cap():
    """None, not 500. 'The cap would do it' and 'the cap is what it takes' are
    different sentences and only one of them is true."""
    assert NF.replicates_for(1e-9, 0.02, cap=50) is None
    assert NF.cells_for(1e-9, 0.05, cap=50) is None


def test_sd_upper_bound_is_above_the_estimate_and_tightens_with_df():
    assert NF.sd_upper_bound(0.02, 4) > NF.sd_upper_bound(0.02, 20) > 0.02
    # The two bounds N=6 over four cells was chosen against.
    assert NF.sd_upper_bound(1.0, 20) == pytest.approx(1.444, abs=2e-3)
    assert NF.sd_upper_bound(1.0, 4) == pytest.approx(2.874, abs=2e-3)


def test_cells_needed_for_the_cross_card_effect_reproduces_the_3_5x():
    """12x in cells is 3.5x in standard error. Both are printed; neither is a
    restatement of the other and quoting one for the other is how the
    'underpowered by 3.5x' line became ambiguous."""
    need = NF.cells_for(0.0117, 0.04798)
    assert need is not None and 120 <= need <= 145
    assert math.sqrt(need / 11) == pytest.approx(3.5, abs=0.2)


# --- 1. the floor is zero because nothing varied ----------------------------

def test_six_replicates_get_six_distinct_run_ids():
    arm = NF.DEFAULT_ARMS[0]
    ids = {NF.run_id_for(arm, i, gpu_name="NVIDIA H200", cache_mode="fresh",
                         sweep_args=(), order=NF.ORDER_COUNTERBALANCED)
           for i in range(1, 7)}
    assert len(ids) == 6


def test_run_id_separates_every_swept_parameter():
    """Each knob, moved alone, must move the id. A knob that does not is a knob
    whose second setting silently reports the first's numbers."""
    base = NF.DEFAULT_ARMS[0]
    ref = NF.run_id_for(base, 1, gpu_name="NVIDIA H200", cache_mode="fresh",
                        sweep_args=(), order=NF.ORDER_COUNTERBALANCED)
    import dataclasses
    for knob, value in [("group_m", 16), ("block_n", 256), ("num_stages", 3),
                        ("r_max", 512), ("row_step", 64), ("trials", 5),
                        ("warmup_ms", 150.0), ("l2_flush", False),
                        ("seed", 7), ("tiles", "32,64"),
                        ("dtype", "fp16"), ("step_probes", 3),
                        ("cell_budget_ms", 200.0)]:
        moved = dataclasses.replace(base, **{knob: value})
        got = NF.run_id_for(moved, 1, gpu_name="NVIDIA H200", cache_mode="fresh",
                            sweep_args=(), order=NF.ORDER_COUNTERBALANCED)
        assert got != ref, f"moving {knob} did not change the run id"
    assert NF.run_id_for(base, 1, gpu_name="NVIDIA A100-SXM4-80GB",
                         cache_mode="fresh", sweep_args=(),
                         order=NF.ORDER_COUNTERBALANCED) != ref
    assert NF.run_id_for(base, 1, gpu_name="NVIDIA H200", cache_mode="warm",
                         sweep_args=(),
                         order=NF.ORDER_COUNTERBALANCED) != ref
    # THE PASSTHROUGH AND THE DESIGN. `--sweep-arg` reaches the child sweep as
    # an argument we do not parse, and `--order` decides the counterbalancing.
    # Neither was in the key until 2026-09-02, so a G=16 arm requested through
    # the passthrough resumed the G=1 arm's cells and a switch of design resumed
    # the other design's.
    assert NF.run_id_for(base, 1, gpu_name="NVIDIA H200", cache_mode="fresh",
                         sweep_args=("--group-m", "16"),
                         order=NF.ORDER_COUNTERBALANCED) != ref
    assert NF.run_id_for(base, 1, gpu_name="NVIDIA H200", cache_mode="fresh",
                         sweep_args=(), order=NF.ORDER_PAIRED) != ref


def test_the_sweeps_own_run_id_now_carries_the_card_and_ours_still_adds_the_replicate():
    """THE DEFECT THIS SCRIPT ROUTED AROUND, now fixed AT THE SOURCE.

    `block_m_crossing_sweep.default_run_id` used to omit the GPU, which is why
    `mixtral-8x7b-bf16-r1024-g1-n64-4867a2.report.json` is the filename of BOTH
    the A100 arm and the H200 arm in results/published. It takes the card as of
    2026-09-02, so this test pins the FIX rather than the defect -- a local
    workaround that outlives its cause is how the next caller inherits the bug.

    The replicate index is still ours to add: the sweep has no notion of one,
    and six replicates of a single arm on a single card would otherwise derive
    six identical ids, resume into one directory and report run 1 six times with
    a between-replicate sd of exactly 0.0000.
    """
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sweep
    spec.loader.exec_module(sweep)
    args = sweep.build_parser().parse_args([])
    assert (sweep.default_run_id(args, "nvidia_h200")
            == sweep.default_run_id(args, "nvidia_h200"))
    assert (sweep.default_run_id(args, "nvidia_h200")
            != sweep.default_run_id(args, "nvidia_a100_sxm4_80gb"))
    # ...and our id for the same arm carries the card and the replicate.
    ours = NF.run_id_for(NF.DEFAULT_ARMS[0], 1, gpu_name="NVIDIA H200",
                         cache_mode="fresh", sweep_args=(),
                         order=NF.ORDER_COUNTERBALANCED)
    assert "rep1" in ours and ours.startswith("nvidia_h200-")


def test_identical_values_are_flagged_degenerate_not_reported_as_a_floor():
    assert spread("a", 32, [0.7, 0.7, 0.7]).degenerate is True
    assert spread("a", 32, [0.7, 0.7, 0.700001]).degenerate is False


def test_a_zero_spread_cell_refuses_to_be_pooled():
    """Zero spread is a collision signature. Pooling it in would halve the floor
    and the study's smallest effects would clear the result."""
    floor = NF.pool([spread("a", 32, [0.70, 0.70, 0.70]),
                     spread("a", 64, [0.60, 0.62, 0.61])], NF.PRIMARY_FIELD)
    assert floor.pooled is False
    assert "collision signature" in floor.reason
    assert floor.pooled_sd == pytest.approx(0.01, abs=1e-9)


# --- 4. cells that disagree get averaged ------------------------------------

def test_homogeneous_cells_pool_and_the_df_adds_up():
    cells = [spread("a", 32, [0.70, 0.72, 0.71, 0.73, 0.70, 0.72]),
             spread("a", 64, [0.60, 0.62, 0.61, 0.63, 0.60, 0.62])]
    floor = NF.pool(cells, NF.PRIMARY_FIELD)
    assert floor.pooled is True
    assert floor.df == 10
    assert floor.pooled_sd == pytest.approx(cells[0].sd, rel=1e-9)
    assert floor.upper95 > floor.pooled_sd


def test_disagreeing_cells_refuse_to_pool_and_publish_the_widest():
    tight = spread("a", 32, [0.700, 0.701, 0.702])
    wild = spread("a", 64, [0.60, 0.75, 0.50])
    floor = NF.pool([tight, wild], NF.PRIMARY_FIELD)
    assert floor.pooled is False
    assert floor.pooled_sd == pytest.approx(wild.sd)
    assert "NOT pooled" in floor.reason
    assert floor.pooled_sd > NF.pool([tight, tight], NF.PRIMARY_FIELD).pooled_sd


def test_a_cell_with_one_replicate_contributes_nothing():
    floor = NF.pool([spread("a", 32, [0.7])], NF.PRIMARY_FIELD)
    assert floor.pooled_sd is None and floor.df == 0
    assert "two or more replicates" in floor.reason


# --- 2. the floor is absent and something returned 0.0 anyway ---------------

def test_noise_floor_raises_when_the_file_does_not_exist(tmp_path):
    with pytest.raises(NF.NoiseFloorUnmeasured, match="does not exist"):
        NF.noise_floor(tmp_path / "nope.json")


def test_noise_floor_raises_when_part_a_has_not_run(tmp_path):
    """The state the repo is in the moment part (b) is published: the control is
    there, the floor is null, and every caller must crash rather than default."""
    path = tmp_path / "NOISE_FLOOR.json"
    doc = NF.build_document({f: _fake_diff(f) for f in NF.ALPHA_FIELDS},
                            {f: _fake_diff(f) for f in NF.ALPHA_FIELDS}, None,
                            NF.unmeasured_provenance())
    assert doc["replicate_floor"] is None
    path.write_text(json.dumps(doc))
    with pytest.raises(NF.NoiseFloorUnmeasured, match="has not run"):
        NF.noise_floor(path)


def test_noise_floor_refuses_a_rehearsal(tmp_path):
    path = tmp_path / "NOISE_FLOOR.json"
    path.write_text(json.dumps(_doc_with_floor(synthetic=True)))
    with pytest.raises(NF.NoiseFloorUnmeasured, match="REHEARSAL"):
        NF.noise_floor(path)
    # ...and hands it over only when a caller says the word.
    assert NF.noise_floor(path, allow_synthetic=True).sd > 0


def test_noise_floor_refuses_an_unmeasured_field(tmp_path):
    path = tmp_path / "NOISE_FLOOR.json"
    path.write_text(json.dumps(_doc_with_floor()))
    with pytest.raises(NF.NoiseFloorUnmeasured, match="no floor for"):
        NF.noise_floor(path, "alpha_sideways")


def test_assert_resolvable_blocks_the_cross_card_claim(tmp_path):
    """The one-line call a future cross-card claim has to survive."""
    path = tmp_path / "NOISE_FLOOR.json"
    path.write_text(json.dumps(_doc_with_floor()))
    with pytest.raises(NF.EffectBelowNoiseFloor, match="unresolvable measurement"):
        NF.assert_resolvable(0.0117, "cross-card L2", path=path)
    floor = NF.assert_resolvable(0.3855, "swizzle swing", path=path)
    assert floor.sd == pytest.approx(0.02)


def _fake_diff(field_name: str) -> NF.PairedDifference:
    return NF.PairedDifference("first", "second", field_name, (), (), None)


def _doc_with_floor(synthetic: bool = False) -> dict:
    return {
        "schema": NF.SCHEMA,
        "replicate_floor": {
            "n_replicates": 6, "cache_mode": "fresh", "gpu_name": "NVIDIA H200",
            "provenance": "test", "synthetic": synthetic,
            "per_field": {NF.PRIMARY_FIELD: {
                "sd": 0.02, "df": 20, "upper95": 0.0289, "pooled": True,
                "reason": "test", "cells": 4, "per_cell": []}},
        },
    }


# --- 5. the control says what we wish it said -------------------------------

@needs_arms
def test_the_stages_control_is_one_card_and_one_knob():
    diff = NF.stages_control()
    assert diff.same_machine is True, "the control must be the SAME card"
    assert diff.varied == ("num_stages",)
    assert diff.n == 11


@needs_arms
def test_the_cross_card_comparison_is_two_cards_and_no_knob():
    diff = NF.cross_card()
    assert diff.same_machine is False
    assert diff.varied == ("gpu",)
    assert diff.n == 11, "the control and the comparison must span the SAME cells"


@needs_arms
def test_the_two_comparisons_use_literally_the_same_cells():
    """If they did not, the control would be scoring a different experiment."""
    ctrl = {key for key, _, _ in NF.stages_control().pairs}
    card = {key for key, _, _ in NF.cross_card().pairs}
    assert ctrl == card


@needs_arms
def test_the_control_numbers_are_these_numbers():
    diff = NF.stages_control()
    assert diff.mean == pytest.approx(0.0101, abs=5e-4)
    assert diff.sd == pytest.approx(0.0323, abs=5e-4)


@needs_arms
def test_the_cross_card_numbers_are_these_numbers():
    diff = NF.cross_card()
    assert diff.mean == pytest.approx(0.0117, abs=5e-4)
    assert diff.sd == pytest.approx(0.0480, abs=5e-4)


@needs_arms
def test_a_pipeline_stage_moves_alpha_as_much_as_a_card_does():
    """THE HEADLINE OF PART (b), as a number.

    The card exceeds the control by less than the control's own detection limit,
    so the two are indistinguishable and the cross-card difference cannot be
    attributed to L2.
    """
    ctrl, card = NF.stages_control(), NF.cross_card()
    excess = abs(card.mean) - abs(ctrl.mean)
    assert excess < ctrl.mde
    assert excess == pytest.approx(0.0016, abs=5e-4)


@needs_arms
def test_both_differences_are_below_their_own_detection_limits():
    for diff in (NF.stages_control(), NF.cross_card()):
        assert diff.resolved is False


@needs_arms
def test_the_cross_card_sign_flips_between_estimators_and_the_control_does_not():
    """An effect whose sign depends on which anchoring of the same fit you read
    is not an effect. The control keeps one sign across all three, which is what
    makes the flip a property of the cross-card comparison and not of the code."""
    cards = {f: NF.cross_card(f).mean for f in NF.ALPHA_FIELDS}
    ctrl = {f: NF.stages_control(f).mean for f in NF.ALPHA_FIELDS}
    assert cards["alpha_corrected"] > 0 > cards["alpha_upper"]
    assert len({math.copysign(1, v) for v in ctrl.values()}) == 1


@needs_arms
def test_the_11_cell_design_could_not_have_seen_what_it_reported():
    diff = NF.cross_card()
    assert diff.mde == pytest.approx(0.0450, abs=1e-3)
    assert abs(diff.mean) < diff.mde


@needs_arms
def test_machine_identity_comes_from_sm_count_not_from_the_directory_name():
    """The report JSON records no gpu_name, so the card is knowable only from
    sm_count -- 132 on the H200, 108 on the A100. A directory name is a filename."""
    h200 = NF.read_arm(NF.CROSS_CARD_ARMS[0])
    a100 = NF.read_arm(NF.CROSS_CARD_ARMS[1])
    assert {c.sm_count for c in h200} == {132}
    assert {c.sm_count for c in a100} == {108}
    assert NF.machine_differs(h200, a100) is False
    assert NF.machine_differs(h200, h200) is True


def test_machine_identity_is_unknown_rather_than_guessed_when_unrecorded():
    blank = [NF.LadderCell("m", "bf16", 1, 64, 64, 8, 4, 32, {}, 3, 0, "x")]
    known = [NF.LadderCell("m", "bf16", 1, 64, 64, 8, 4, 32, {}, 3, 132, "y")]
    assert NF.machine_differs(blank, known) is None


# --- reading arms, and refusing to read nothing ------------------------------

def test_read_arm_raises_on_an_empty_directory(tmp_path):
    """An empty arm produces an empty intersection, which prints as a clean
    '0 matched cells' and reads like a finding about the data."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        NF.read_arm(tmp_path / "empty")


def _write_report(directory: Path, name: str, *, stages: int, sm: int,
                  ladder: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.report.json").write_text(json.dumps({
        "model": "mixtral-8x7b", "dtype": "bf16", "sm_count": sm,
        "fixed": {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
                  "num_warps": 8, "num_stages": stages},
        "ladder": ladder}))


def test_an_unidentifiable_fit_is_dropped_and_never_differenced_as_zero(tmp_path):
    """`alpha: null` means the fit was not identifiable. Treating it as 0.0 would
    invent a -0.7 difference out of a missing measurement."""
    fit = {"alpha": 0.9, "alpha_corrected": 0.88, "alpha_upper": 1.1,
           "memory_points": 16}
    gone = {"alpha": None, "alpha_corrected": None, "alpha_upper": None,
            "memory_points": 0}
    _write_report(tmp_path / "a", "x", stages=4, sm=132,
                  ladder={"32": fit, "64": fit})
    _write_report(tmp_path / "b", "x", stages=3, sm=132,
                  ladder={"32": fit, "64": gone})
    diff = NF.pair_arms(tmp_path / "a", tmp_path / "b", "alpha_corrected")
    assert diff.n == 1
    assert diff.mean == pytest.approx(0.0)
    assert diff.varied == ("num_stages",)


def test_pairing_matches_on_every_pinned_knob(tmp_path):
    """A cell that differs in BLOCK_SIZE_N is a different cell and must not pair."""
    fit = {"alpha": 0.9, "alpha_corrected": 0.88, "alpha_upper": 1.1,
           "memory_points": 16}
    _write_report(tmp_path / "a", "x", stages=4, sm=132, ladder={"32": fit})
    (tmp_path / "b").mkdir(parents=True)
    (tmp_path / "b" / "x.report.json").write_text(json.dumps({
        "model": "mixtral-8x7b", "dtype": "bf16", "sm_count": 132,
        "fixed": {"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
                  "num_warps": 8, "num_stages": 3},
        "ladder": {"32": fit}}))
    assert NF.pair_arms(tmp_path / "a", tmp_path / "b", "alpha_corrected").n == 0


# --- the sign, and the gates -------------------------------------------------

def test_every_delta_names_both_arms_and_the_direction():
    said = NF.delta_sentence("s4", "s3", 0.0101)
    assert "s3" in said and "s4" in said and "HIGHER" in said
    assert "LOWER" in NF.delta_sentence("s4", "s3", -0.0101)
    assert "equal" in NF.delta_sentence("s4", "s3", 0.0)


def test_a_failed_gate_says_what_it_invalidates():
    gate = NF.Gate(NF.exit_codes.CLAIM, "G", "claim", "rule", "PASS", False,
                   "saw", "the headline")
    text = gate.render()
    assert "[FAIL]" in text and "invalidates: the headline" in text
    assert "expected PASS" in text, "an unexpected FAIL must be marked as one"


def test_an_unevaluated_gate_is_unknown_and_never_a_pass():
    gate = NF.Gate(NF.exit_codes.CLAIM, "G", "claim", "rule", "PASS", None,
                   "nothing ran")
    text = gate.render()
    assert "[UNKNOWN]" in text
    assert "0 PASS, 0 FAIL, 1 UNKNOWN" in NF.render_gates([gate])
    # And the shared table scores UNKNOWN against the gate, never as a pass.
    assert NF.exit_codes.classify([gate.scored()]) == NF.exit_codes.CLAIM_FAIL


def test_parse_compiles_reads_the_sweeps_own_gate_0_line():
    doc = {"gates": [{"number": 0, "measured":
                      "fresh Triton artefacts per setting: BM=32:17, BM=64:16, "
                      "BM=128:16, BM=256:16"}]}
    assert NF.parse_compiles(doc) == {32: 17, 64: 16, 128: 16, 256: 16}


def test_parse_compiles_refuses_rather_than_reporting_zero():
    """{} is UNKNOWN. 0 would mean 'nothing compiled', which is the opposite
    verdict about cache freshness."""
    assert NF.parse_compiles({"gates": []}) == {}
    assert NF.parse_compiles({"gates": [{"number": 0, "measured": "n/a"}]}) == {}


def test_validity_gates_report_unknown_when_nothing_was_launched():
    """A check that examined nothing must not report zero failures."""
    floors = {f: NF.pool([], f) for f in NF.ALPHA_FIELDS}
    gates = NF.validity_gates([], 6, list(NF.DEFAULT_ARMS), "fresh", floors)
    # V7 (scope) is answerable from argv alone and PASSES on the default arms,
    # which is the point of it: it asks whether the DESIGN covers more than one
    # model, not whether anything ran. Every gate that reads a measurement is
    # UNKNOWN.
    measuring = [g for g in gates if g.name != "V7 scope"]
    assert all(g.passed is not True for g in measuring), [g.name for g in gates]
    assert NF.exit_codes.classify(g.scored() for g in gates) == \
        NF.exit_codes.INVALID


@needs_arms
def test_c4_and_c5_fail_from_the_committed_reports_with_no_gpu():
    """Both registered as expected failures, and both settled without a card."""
    control = {f: NF.stages_control(f) for f in NF.ALPHA_FIELDS}
    cards = {f: NF.cross_card(f) for f in NF.ALPHA_FIELDS}
    floors = {f: NF.pool([], f) for f in NF.ALPHA_FIELDS}
    gates = NF.claim_gates(floors, [], control, cards, list(NF.DEFAULT_ARMS))
    by_name = {g.name.split()[0]: g for g in gates}
    assert by_name["C4"].passed is False and by_name["C4"].expected == "FAIL"
    assert by_name["C5"].passed is False and by_name["C5"].expected == "FAIL"
    # C1 to C3 need the floor, so with no replicates they must be UNKNOWN.
    assert by_name["C1"].passed is None and by_name["C2"].passed is None


# --- where the output lands ---------------------------------------------------

def test_git_takes_the_published_floor_and_not_a_sibling():
    """`results/*` is ignored with only `!results/published/` excepted. Checked
    against git rather than against this comment."""
    if NF.git_accepts(NF.NOISE_FLOOR_JSON) is None:
        pytest.skip("git could not be asked here")
    assert NF.git_accepts(NF.NOISE_FLOOR_JSON) is True
    assert NF.git_accepts(ROOT / "results" / "NOISE_FLOOR.json") is False


def test_write_published_refuses_a_path_git_would_drop(tmp_path, monkeypatch):
    monkeypatch.setattr(NF, "git_accepts", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSING to write"):
        NF.write_published({}, tmp_path / "x.json")
    monkeypatch.setattr(NF, "git_accepts", lambda p: None)
    with pytest.raises(SystemExit, match="could not be asked"):
        NF.write_published({}, tmp_path / "x.json")


@needs_arms
def test_the_published_document_carries_the_control_and_a_null_floor():
    doc = NF.build_document({f: NF.stages_control(f) for f in NF.ALPHA_FIELDS},
                            {f: NF.cross_card(f) for f in NF.ALPHA_FIELDS}, None,
                            NF.unmeasured_provenance())
    assert doc["replicate_floor"] is None
    assert doc["sign"].startswith("every delta is alpha(second arm)")
    assert doc["stages_control"]["alpha_corrected"]["n_cells"] == 11
    assert doc["stages_control"]["alpha_corrected"]["same_machine"] is True
    assert doc["cross_card"]["alpha_corrected"]["same_machine"] is False
    assert {e["name"] for e in doc["effects_registered"]} == {
        e.name for e in NF.EFFECTS}


# --- the plan runs off GPU ----------------------------------------------------

def test_the_arm_builds_a_sweep_command_the_sweep_accepts():
    """Every argument this script passes must exist on the sweep's parser, or the
    twelve replicates all die on argv three minutes into a rented pod."""
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sweep
    spec.loader.exec_module(sweep)
    argv = NF.DEFAULT_ARMS[1].sweep_argv("some-id", Path("/tmp/x"))
    args = sweep.build_parser().parse_args(argv)
    assert args.group_m == 16 and args.run_id == "some-id"


def test_the_default_arms_straddle_the_swizzle_swing():
    """The floor is measured at BOTH ends of the largest effect the study claims,
    so the same replicates that produce the floor also score that effect."""
    assert {a.group_m for a in NF.DEFAULT_ARMS} == {1, 16}
    # BOTH models since 2026-09-02. Measuring only mixtral measured the one
    # model where the swizzle effect is 0.3855 and said nothing about the one
    # where it is 0.0226 and changes sign between tiles on the A100.
    assert {a.model for a in NF.DEFAULT_ARMS} == {"mixtral-8x7b",
                                                 "qwen2-57b-a14b"}
    for model, group in NF.swizzle_pairs(list(NF.DEFAULT_ARMS)).items():
        assert [a.group_m for a in group] == [1, 16], model


def test_resolve_arms_refuses_an_unknown_name():
    with pytest.raises(SystemExit, match="unknown arm"):
        NF.resolve_arms("mixtral_g1,not_an_arm")
    with pytest.raises(SystemExit, match="at least one arm"):
        NF.resolve_arms("")


@needs_arms
def test_control_only_runs_off_gpu_and_reports_nothing_measured(capsys):
    assert NF.main(["--control-only"]) == NF.exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "PART (b) ONLY" in out and "no number on this page is one" in out
    assert "+0.0101" in out and "+0.0117" in out
    # A REFUSAL SCORES NOTHING and therefore prints no RESULT line. The old
    # summary grepped free text for `floor|sigma`, matched this page eighteen
    # times, and printed the imported proxy and a pre-registered expectation as
    # measured output.
    assert "RESULT:" not in out
    with pytest.raises(NF.exit_codes.NoGatesScored):
        NF.exit_codes.classify_text(out)


@needs_arms
def test_dry_run_prints_the_plan_the_cost_and_the_registered_predictions(capsys):
    assert NF.main(["--dry-run", "--replicates", "6"]) == NF.exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "NOT A RESULT" in out
    assert "TOTAL" in out and "min of GPU" in out
    assert out.count("-rep6-") == len(NF.DEFAULT_ARMS), \
        "every replicate's run id must be in the plan"
    # The predictions have to carry NUMBERS and their expected verdicts before
    # anything is measured, or "registered in advance" means nothing. They must
    # NOT be shaped like scored results: `[PASS]` is what a gate prints, and a
    # summary grepping prose read these expectations out of a refused log.
    for expected in ("C4 card beats stages control        expect FAIL",
                     "C5 sign consistency                 expect FAIL",
                     "C1 floor size                       expect PASS",
                     "C3 swizzle mixtral-8x7b            expect PASS",
                     "C3 swizzle qwen2-57b-a14b          expect FAIL",
                     "sd <= 0.0228", "|0.0117| < MDE"):
        assert expected in out, f"missing from the registered predictions: {expected}"
    assert "RESULT:" not in out
    # ...and the cost, which is what a metered pod is budgeted against.
    assert "wall-over-model factor" in out


@needs_arms
def test_no_publish_path_can_write_an_unstamped_floor(monkeypatch):
    """THE TRACKED FILE IS NEVER WRITTEN WITHOUT PROVENANCE.

    `--control-only --publish` and the no-GPU `--dry-run --publish` both write
    `results/published/NOISE_FLOOR.json`, and until 2026-09-02 both called
    `build_document` without `prov=`, so the committed floor named no machine,
    no instrument and no ruler. `prov` is a required positional argument now, so
    the FAIL branch is a TypeError at the call site rather than a silent hole in
    a published artefact, and that branch is planted here too.
    """
    with pytest.raises(TypeError, match="prov"):
        NF.build_document({}, {}, None)

    written: list[dict] = []
    monkeypatch.setattr(NF, "write_published",
                        lambda doc, *a, **k: written.append(doc) or "wrote (fake)")
    assert NF.main(["--control-only", "--publish"]) == NF.exit_codes.REFUSED
    assert NF.main(["--dry-run", "--publish", "--gpu-name", "NVIDIA H200"]) \
        == NF.exit_codes.REFUSED
    assert len(written) == 2, "both publish paths must have been exercised"
    for doc in written:
        for key in NF.PV.TOP_LEVEL_KEYS:
            assert key in doc, key
        assert doc["provenance"]["git_sha"], "a published floor names its commit"
        # Nothing was measured on either page, and the instrument says so rather
        # than naming the live one.
        assert doc["instrument"] == NF.UNMEASURED_INSTRUMENT
        assert doc["replicate_floor"] is None


@needs_arms
def test_the_replicate_tree_is_named_after_the_card(capsys):
    """Collision 2: the results root is a network volume that outlives the pod,
    and the base directory was `{cache_mode}-n{n}` with no card in it, so two
    machines interleaved their logs/ trees."""
    assert NF.main(["--dry-run", "--gpu-name", "NVIDIA H200"]) \
        == NF.exit_codes.REFUSED
    h200 = [ln for ln in capsys.readouterr().out.splitlines()
            if "EVERYTHING IS SAVED TO" in ln][0]
    assert NF.main(["--dry-run", "--gpu-name", "NVIDIA A100-SXM4-80GB"]) \
        == NF.exit_codes.REFUSED
    a100 = [ln for ln in capsys.readouterr().out.splitlines()
            if "EVERYTHING IS SAVED TO" in ln][0]
    assert "nvidia_h200-fresh-n6" in h200
    assert "nvidia_a100_sxm4_80gb-fresh-n6" in a100
    assert h200 != a100


def test_the_committed_arms_are_present_so_a_green_run_is_not_an_empty_one():
    """NON-VACUITY for this file. Most of the interesting tests read
    results/published; if it vanished they would all skip and the suite would go
    green having checked nothing about the control."""
    assert HAVE_ARMS, (
        "the alpha-surface arms are missing from results/published, so every "
        "part-(b) test above skipped and this file verified nothing about the "
        "num_stages control")


@pytest.mark.skipif(not NF.NOISE_FLOOR_JSON.exists(),
                    reason="the noise floor has not been published here yet")
def test_the_published_floor_file_refuses_until_a_card_has_run():
    """The DELIVERABLE, checked as shipped rather than as built in memory.

    Part (b) is publishable today and part (a) is not, so the committed file must
    carry a real control and a null floor, and every importer must crash on it
    rather than receive a default.
    """
    doc = json.loads(NF.NOISE_FLOOR_JSON.read_text())
    # The committed file predates the 2026-09-02 schema bump and still parses:
    # v2 ADDED scope, prior_sd_by_model, an instrument and a provenance block,
    # and corrected the prior's scope string. A schema outside this tuple is
    # refused rather than parsed on the old field meanings.
    assert doc["schema"] in NF.SCHEMA_READABLE
    assert doc["stages_control"]["alpha_corrected"]["n_cells"] == 11
    assert doc["stages_control"]["alpha_corrected"]["same_machine"] is True
    assert doc["cross_card"]["alpha_corrected"]["same_machine"] is False
    if doc["replicate_floor"] is None:
        with pytest.raises(NF.NoiseFloorUnmeasured):
            NF.noise_floor()
        with pytest.raises(NF.NoiseFloorUnmeasured):
            NF.assert_resolvable(0.3855, "swizzle swing")
    else:
        floor = NF.noise_floor()
        assert floor.sd > 0 and floor.df >= 1


@needs_arms
def test_every_mode_prints_how_to_import_the_number(capsys):
    """A number other scripts must import is useless if finding it needs this
    file's git history. The import recipe is printed by every mode."""
    for argv in (["--control-only"], ["--dry-run"]):
        NF.main(argv)
        out = capsys.readouterr().out
        assert "assert_resolvable" in out and "noise_floor()" in out
        assert "results/published/NOISE_FLOOR.json" in out


def test_the_power_table_df_column_follows_the_cell_count_it_is_given():
    """The df column is C(N-1). A table that always printed the PLANNED C would
    hide an arm that came back with half its fits identifiable."""
    two = NF.render_power_table(0.02, "x", cells=2)
    four = NF.render_power_table(0.02, "x", cells=4)
    # At N=6: two cells give 10 df and a 1.75x bound, four give 20 df and 1.44x.
    assert "1.75x  (10 df)" in two and "1.75x  (10 df)" not in four
    assert "1.44x  (20 df)" in four and "1.44x  (20 df)" not in two


# --------------------------------------------------------------------------
# 6. THE DESIGN'S OWN DEFECTS (audit A15). A floor measured in two BLOCKS is a
# floor whose sd is a within-block number scoring a between-block difference; a
# floor measured on one model is a floor measured where the effect is largest;
# and a proxy labelled "upper bound" without saying what it bounds gets cited
# for comparisons it says nothing about.
# --------------------------------------------------------------------------

def test_the_replicates_interleave_instead_of_running_in_two_blocks():
    """THE DEFECT THIS REPLACES ran all six G=1 replicates and then all six
    G=16, so the swizzle delta was a between-block difference carrying whatever
    drifted across thirteen minutes while the sd it was scored against was a
    within-block number."""
    arms = list(NF.DEFAULT_ARMS)
    order = NF.run_order(arms, 6, NF.ORDER_COUNTERBALANCED)
    assert len(order) == len(arms) * 6
    assert len(set(order)) == len(order), "every (arm, replicate) exactly once"
    # Each model's two swizzles are ADJACENT in the launch sequence, which is
    # what "paired within minutes" means operationally.
    positions = {}
    for index, (arm, rep) in enumerate(order):
        positions[(arm.model, rep)] = positions.get((arm.model, rep), []) + [index]
    for key, where in positions.items():
        assert max(where) - min(where) == 1, key


def test_counterbalanced_alternates_the_pair_order_and_paired_does_not():
    """Counterbalancing cancels a linear drift inside a pair out of the MEAN
    delta; the paired order leaves it in every one of them."""
    arms = [a for a in NF.DEFAULT_ARMS if a.model == "mixtral-8x7b"]
    counter = [a.swizzle_label for a, _ in
               NF.run_order(arms, 6, NF.ORDER_COUNTERBALANCED)]
    paired = [a.swizzle_label for a, _ in
              NF.run_order(arms, 6, NF.ORDER_PAIRED)]
    assert counter[:4] == ["g1", "g16", "g16", "g1"]
    assert paired[:4] == ["g1", "g16", "g1", "g16"]
    assert counter.count("g1") == counter.count("g16") == 6


def test_the_plan_prints_the_order_it_will_actually_launch_in():
    """A description of an order is not an order: the blocked design this
    replaces described itself as interleaved in its own predictions text."""
    arms = list(NF.DEFAULT_ARMS)
    lines = NF.order_lines(arms, 6, NF.ORDER_COUNTERBALANCED)
    assert len(lines) == 2, "one line per model"
    assert any("g1,g16,g16,g1,g1,g16,g16,g1,g1,g16,g16,g1" in line
               for line in lines)
    paired = NF.order_lines(arms, 6, NF.ORDER_PAIRED)
    assert all(re.search(r"order: \S+ (g1,g16,){5}g1,g16", line)
               for line in paired)
    # And the sequence a line prints is the sequence run_order returns.
    for mode in NF.ORDER_MODES:
        seq = [a.swizzle_label for a, _ in NF.run_order(arms, 6, mode)
               if a.model == "mixtral-8x7b"]
        assert ",".join(seq) in "\n".join(NF.order_lines(arms, 6, mode))


def test_run_order_refuses_an_order_it_does_not_know():
    with pytest.raises(ValueError, match="unknown order"):
        NF.run_order(list(NF.DEFAULT_ARMS), 6, "blocked")


def test_the_qwen2_swizzle_effect_is_registered_with_its_source():
    """A15/S35: the surface is published as a general mechanism on the strength
    of the mixtral number. The qwen2 one was never written down, and it is a
    tenth the size and sign-inconsistent across tiles on the A100."""
    swings = {e.model: e for e in NF.EFFECTS if e.name == "swizzle swing"}
    assert set(swings) == {"mixtral-8x7b", "qwen2-57b-a14b"}
    assert swings["qwen2-57b-a14b"].size == pytest.approx(0.0226, abs=1e-4)
    assert "OPPOSITE SIGNS" in swings["qwen2-57b-a14b"].source
    # It is BELOW what N=6 replicates can resolve, which is why its gate is
    # registered as an expected FAIL rather than discovered afterwards.
    assert swings["qwen2-57b-a14b"].size < NF.mde_two_sample(NF.PRIOR_SD, 6)
    assert swings["mixtral-8x7b"].size > NF.mde_two_sample(NF.PRIOR_SD, 6)


@needs_arms
def test_the_registered_qwen2_effect_matches_the_committed_reports():
    """The registered size is a claim about files on disk, so it is checked
    against them rather than trusted."""
    biggest = 0.0
    for arm in (ROOT / "results" / "published" /
                "2026-09-01-nvidia_h200-alpha-surface-s4",
                ROOT / "results" / "published" /
                "2026-09-01-nvidia_h200-cross-card-s3"):
        cells = {c.key: c for c in NF.read_arm(arm)
                 if c.model == "qwen2-57b-a14b" and c.block_n == 64}
        for key, cell in cells.items():
            if key[2] != 1:                     # GROUP_SIZE_M == 1 only
                continue
            other = (key[0], key[1], 16, *key[3:])
            mate = cells.get(other)
            if mate is None:
                continue
            a = cell.values["alpha_corrected"]
            c = mate.values["alpha_corrected"]
            if a is not None and c is not None:
                biggest = max(biggest, abs(c - a))
    registered = next(e for e in NF.EFFECTS
                      if e.model == "qwen2-57b-a14b")
    assert registered.size == pytest.approx(biggest, abs=5e-4)


def test_one_c3_per_model_scored_against_its_own_registered_effect():
    """The old C3 took arms[0] against arms[-1], which became mixtral-G=1
    against qwen2-G=16 the moment a second model was added: a contrast across
    two levers at once, scored as though it were one."""
    arms = list(NF.DEFAULT_ARMS)
    floors = {f: NF.pool([], f) for f in NF.ALPHA_FIELDS}
    control = {f: NF.stages_control(f) for f in NF.ALPHA_FIELDS} \
        if HAVE_ARMS else None
    if control is None:
        pytest.skip("the committed arms are not checked out")
    cards = {f: NF.cross_card(f) for f in NF.ALPHA_FIELDS}
    gates = NF.claim_gates(floors, [], control, cards, arms)
    names = [g.name for g in gates]
    assert "C3 swizzle mixtral-8x7b" in names
    assert "C3 swizzle qwen2-57b-a14b" in names
    by_name = {g.name: g for g in gates}
    assert by_name["C3 swizzle mixtral-8x7b"].expected == "PASS"
    assert by_name["C3 swizzle qwen2-57b-a14b"].expected == "FAIL"


def test_c3_refuses_a_model_measured_at_one_swizzle():
    """UNKNOWN, not PASS: a model with one GROUP_SIZE_M has no contrast, and a
    gate that examined nothing must not report zero failures."""
    if not HAVE_ARMS:
        pytest.skip("the committed arms are not checked out")
    arms = [a for a in NF.DEFAULT_ARMS if a.group_m == 1]
    floors = {f: NF.pool([], f) for f in NF.ALPHA_FIELDS}
    control = {f: NF.stages_control(f) for f in NF.ALPHA_FIELDS}
    cards = {f: NF.cross_card(f) for f in NF.ALPHA_FIELDS}
    gates = {g.name: g for g in NF.claim_gates(floors, [], control, cards, arms)}
    gate = gates["C3 swizzle mixtral-8x7b"]
    assert gate.passed is None
    assert "one GROUP_SIZE_M only" in gate.observed


def test_v7_refuses_a_single_model_floor_unless_argv_asks_for_one():
    """A15/S35 again, as a gate. The consequence of a mixtral-only floor -- a
    mechanism confirmed where it is twelve sigma -- is invisible otherwise."""
    floors = {f: NF.pool([], f) for f in NF.ALPHA_FIELDS}
    one = [a for a in NF.DEFAULT_ARMS if a.model == "mixtral-8x7b"]
    both = list(NF.DEFAULT_ARMS)
    def scope(arms, ok):
        gates = NF.validity_gates([], 6, arms, "fresh", floors,
                                  single_model_ok=ok)
        return next(g for g in gates if g.name == "V7 scope")
    assert scope(one, False).passed is False
    assert scope(one, True).passed is True
    assert scope(both, False).passed is True
    # A VALIDITY failure is INVALID: nothing on the page may be quoted.
    assert NF.exit_codes.classify([scope(one, False).scored()]) == \
        NF.exit_codes.INVALID


def test_v6_fails_on_a_mixed_or_unstamped_instrument():
    """A floor pooled over replicates timed by two loops measures the loops.
    An unstamped report and a differently-stamped one are different states and
    both fail."""
    floors = {f: NF.pool([], f) for f in NF.ALPHA_FIELDS}
    basis = NF.timing_basis()

    def rep(index, instrument):
        r = NF.Replicate("mixtral_g1", index, f"id{index}", Path(f"/tmp/{index}"),
                         Path(f"/tmp/{index}/report.json"), returncode=0,
                         instrument=instrument)
        r.cells = [NF.LadderCell("mixtral-8x7b", "bf16", 1, 64, 64, 8, 4, 64,
                                 {"alpha_corrected": 0.9}, 8, 132, "x")]
        return r

    def verdict(reps):
        gates = NF.validity_gates(reps, 1, [NF.DEFAULT_ARMS[0]], "fresh", floors)
        return next(g for g in gates if g.name == "V6 one instrument").passed

    assert verdict([rep(1, basis), rep(2, basis)]) is True
    assert verdict([rep(1, basis), rep(2, "retired/time_call")]) is False
    assert verdict([rep(1, ""), rep(2, "")]) is False, "unstamped is not agreement"
    assert verdict([]) is None


def test_the_arms_pass_the_instruments_knobs_and_not_a_retired_call_count():
    """`--warmup` on the sweep became MILLISECONDS on 2026-09-02. The arm
    carried `warmup=20` from before that, which the new parser reads as 20 ms
    -- a fifteenth of the sweep's own default, so every replicate would have
    been timed at an unsettled clock and the floor would have measured the
    governor."""
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sweep
    spec.loader.exec_module(sweep)
    arm = NF.DEFAULT_ARMS[0]
    args = sweep.build_parser().parse_args(arm.sweep_argv("id", Path("/tmp/x")))
    assert args.warmup == arm.warmup_ms == 300.0
    assert args.trials == arm.trials == 3
    assert not args.no_l2_flush
    assert "--iters" not in arm.sweep_argv("id", Path("/tmp/x"))
    # And BLOCK_M=128 is not swept: it has no fit in any committed arm at these
    # settings, so it contributes nothing to a spread of fitted alphas, while
    # 256 stays because it is the compute reference.
    assert arm.tiles == "32,64,256"


def test_the_plan_states_its_scope_and_its_mde_from_a_stated_assumption():
    """B14 and A15: an MDE nobody stated, and a floor whose scope lived only in
    a docstring and was then cited for cross-pod comparisons."""
    arms = list(NF.DEFAULT_ARMS)
    text = "\n".join(NF.render_mde_line(6, arms))
    assert f"{NF.mde_two_sample(NF.PRIOR_SD, 6):.4f}" in text
    assert "sigma is ASSUMED" in text
    assert "same-session" in text.lower()
    assert "heteroscedastic" in text
    assert "BELOW the limit" in text, "the qwen2 effect must be called out"
    scope = NF.scope_block(arms, n_replicates=6, cache_mode="fresh",
                           gpu_name="NVIDIA H200",
                           order=NF.ORDER_COUNTERBALANCED,
                           single_model_ok=False)
    assert scope["models"] == ["mixtral-8x7b", "qwen2-57b-a14b"]
    assert scope["instrument"] == NF.timing_basis()
    assert "cross-pod" in scope["session"]
    assert any("re-rolled" in line for line in scope["excludes"])


def test_the_docstring_no_longer_claims_the_proxy_arms_ran_on_separate_pods():
    """A15/S30: both ARMS.tsv log to /workspace/session/20260901T214218Z, the
    same pod and the same hour, so the proxy bounds same-session rerun noise
    and nothing else. The old sentence is what made it quotable across pods."""
    source = (ROOT / "scripts" / "replicate_noise_floor.py").read_text()
    # The phrase survives EXACTLY ONCE, inside the paragraph that quotes it in
    # order to mark it false. Deleting the old sentence without recording what
    # it said would leave nothing for a reader of the published proxy to
    # discover it by.
    assert source.count("separate pods") == 1
    assert "SEPARATE PODS IS\nFALSE" in source
    assert "20260901T214218Z" in source
    assert "same-session" in NF.PRIOR_SD_SOURCE.lower() or \
        "SAME-SESSION" in NF.PRIOR_SD_SOURCE


@needs_arms
def test_the_proxy_is_heteroscedastic_and_the_published_block_says_so():
    """S30: mixtral's four deltas have sd 0.0486 and qwen2's seven 0.0186, so
    'inside 2x the pooled 0.0323' is the wrong per-cell yardstick at mixtral."""
    block = NF.by_model_spread(NF.stages_control())
    assert block["per_model"]["mixtral-8x7b"]["sd"] == pytest.approx(0.0486,
                                                                    abs=5e-4)
    assert block["per_model"]["qwen2-57b-a14b"]["sd"] == pytest.approx(0.0186,
                                                                      abs=5e-4)
    assert block["ratio"] == pytest.approx(2.61, abs=0.02)
    assert block["ratio"] < NF.HOMOGENEITY_RATIO
    # And the registered constants match what the arms actually say.
    for model, entry in block["per_model"].items():
        assert NF.PRIOR_SD_BY_MODEL[model] == pytest.approx(entry["sd"],
                                                            abs=5e-4)


@needs_arms
def test_the_published_document_carries_a_provenance_block_and_the_scope():
    arms = list(NF.DEFAULT_ARMS)
    control = {f: NF.stages_control(f) for f in NF.ALPHA_FIELDS}
    cards = {f: NF.cross_card(f) for f in NF.ALPHA_FIELDS}
    scope = NF.scope_block(arms, n_replicates=6, cache_mode="fresh",
                           gpu_name="NVIDIA H200",
                           order=NF.ORDER_COUNTERBALANCED,
                           single_model_ok=False)
    prov = NF.PV.provenance_block(instrument=NF.timing_basis())
    doc = NF.build_document(control, cards, None, prov, scope=scope)
    assert doc["schema"] == NF.SCHEMA
    for key in NF.PV.TOP_LEVEL_KEYS:
        assert key in doc, key
    assert doc["provenance"]["instrument"] == NF.timing_basis()
    assert doc["scope"]["order"] == NF.ORDER_COUNTERBALANCED
    assert "same-session" in doc["prior_sd_source"].lower()
    assert doc["prior_sd_by_model"]["ratio"] == pytest.approx(2.61, abs=0.02)
    assert "RETIRED" in doc["stages_control_instrument"]


def test_noise_floor_refuses_a_schema_it_does_not_know(tmp_path):
    """Parsing an unknown schema means assuming its fields mean what they used
    to, and the one field that has already changed meaning is the prior's
    scope."""
    path = tmp_path / "NOISE_FLOOR.json"
    path.write_text(json.dumps({"schema": "moe-kernels/noise-floor/99",
                                "replicate_floor": {"per_field": {}}}))
    with pytest.raises(NF.NoiseFloorUnmeasured, match="schema"):
        NF.noise_floor(path)
    for known in NF.SCHEMA_READABLE:
        path.write_text(json.dumps({"schema": known, "replicate_floor": None}))
        with pytest.raises(NF.NoiseFloorUnmeasured, match="replicate_floor"):
            NF.noise_floor(path)


def test_every_gate_prints_exactly_one_parsable_result_line():
    """`RESULT: ` is the ONE line per gate a reader may grep. A gate name with
    a space in it would be dropped by the parser rather than fail loudly.

    This is the contract this file offers, not a description of what the
    session driver currently reads: `arm_gate_regex` still selects free text
    for this arm. See the test below for the half that is still open.
    """
    if not HAVE_ARMS:
        pytest.skip("the committed arms are not checked out")
    floors = {f: NF.pool([], f) for f in NF.ALPHA_FIELDS}
    control = {f: NF.stages_control(f) for f in NF.ALPHA_FIELDS}
    cards = {f: NF.cross_card(f) for f in NF.ALPHA_FIELDS}
    arms = list(NF.DEFAULT_ARMS)
    gates = NF.validity_gates([], 6, arms, "fresh", floors)
    gates += NF.claim_gates(floors, [], control, cards, arms)
    text = NF.render_gates(gates)
    parsed = NF.exit_codes.parse_result_lines(text)
    assert len(parsed) == len(gates)
    assert [p.name for p in parsed] == [g.token for g in gates]
    assert len({g.token for g in gates}) == len(gates), "tokens must be unique"
    assert NF.exit_codes.classify_text(text) == \
        NF.exit_codes.classify(g.scored() for g in gates)


#: The session driver's summary selector for this arm, copied from
#: `arm_gate_regex` in `scripts/h200_gaps_session.sh` as it stands on
#: apparatus-standard. Copied rather than imported because that file belongs to
#: another slice: this is a record of what the reader does today, not a claim
#: on it.
DRIVER_FREE_TEXT_GREP = re.compile(r"^[ \t]*V[0-9][ \t]|floor|sigma")
RESULT_GREP = re.compile(r"^RESULT: ")


def _hits(pattern, text):
    return [ln for ln in text.splitlines() if pattern.search(ln)]


@needs_arms
def test_the_result_grep_separates_a_refusal_from_a_scored_page_and_prose_cannot(capsys):
    """A4, both halves: the one this file closed and the one it cannot.

    THE DEFECT was that a REFUSED log reached the session summary looking like
    output, because the summary selects this arm's lines by free text. The half
    this file owns is closed: a refusal scores nothing and prints no
    `RESULT: ` line at all, so the same grep that lifts every gate off a
    measured page lifts NOTHING off a refusal.

    The other half is not closed and cannot be closed here. The page has to say
    "floor" and "sigma" to be about a noise floor, so the driver's current
    pattern matches a refusal as readily as a result and cannot tell them
    apart. That is pinned as a measurement rather than described, so that the
    docstrings' claim about the remaining work is checked and not just asserted;
    it goes green either way once the driver greps `^RESULT: `, since nothing
    here reads that file.
    """
    assert NF.main(["--control-only"]) == NF.exit_codes.REFUSED
    refused = capsys.readouterr().out
    floors = {f: NF.pool([], f) for f in NF.ALPHA_FIELDS}
    control = {f: NF.stages_control(f) for f in NF.ALPHA_FIELDS}
    cards = {f: NF.cross_card(f) for f in NF.ALPHA_FIELDS}
    arms = list(NF.DEFAULT_ARMS)
    gates = NF.validity_gates([], 6, arms, "fresh", floors)
    gates += NF.claim_gates(floors, [], control, cards, arms)
    scored = NF.render_gates(gates)

    assert _hits(RESULT_GREP, refused) == [], \
        "a refusal scores nothing, so it may print no result line"
    assert len(_hits(RESULT_GREP, scored)) == len(gates), \
        "and a scored page prints exactly one per gate, or the grep is useless"
    assert _hits(DRIVER_FREE_TEXT_GREP, refused), \
        ("if this is ever empty the driver's regex has stopped matching a "
         "refusal and the note in the module docstring is out of date")
    assert _hits(DRIVER_FREE_TEXT_GREP, scored), \
        "prose matches both pages, which is exactly why it cannot be the reader"


@needs_arms
def test_a_refused_plan_leaks_its_registered_expectations_to_a_prose_reader(capsys):
    """The concrete shape of the open half, named so it is not forgotten.

    `--dry-run` prints the seven registered validity expectations as
    `V1..V7` rows. Every one of them starts with the driver's own
    `^[[:space:]]*V[0-9][[:space:]]` alternative, so a REFUSED plan puts seven
    pre-registered expectations into the session summary under a heading that
    reads like results. Rewording them cannot help: the rows have to be named
    after the gates they register. Only the reader can fix this, and it is the
    driver slice's line to change.
    """
    assert NF.main(["--dry-run", "--replicates", "6"]) == NF.exit_codes.REFUSED
    out = capsys.readouterr().out
    assert _hits(RESULT_GREP, out) == [], "a plan is not a result"
    leaked = [ln.strip().split()[0] for ln in _hits(DRIVER_FREE_TEXT_GREP, out)
              if re.match(r"^[ \t]*V[0-9][ \t]", ln)]
    assert leaked == [f"V{i}" for i in range(1, 8)], leaked
