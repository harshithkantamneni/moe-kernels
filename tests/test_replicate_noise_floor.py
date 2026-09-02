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
    ids = {NF.run_id_for(arm, i, gpu_name="NVIDIA H200", cache_mode="fresh")
           for i in range(1, 7)}
    assert len(ids) == 6


def test_run_id_separates_every_swept_parameter():
    """Each knob, moved alone, must move the id. A knob that does not is a knob
    whose second setting silently reports the first's numbers."""
    base = NF.DEFAULT_ARMS[0]
    ref = NF.run_id_for(base, 1, gpu_name="NVIDIA H200", cache_mode="fresh")
    import dataclasses
    for knob, value in [("group_m", 16), ("block_n", 256), ("num_stages", 3),
                        ("r_max", 512), ("row_step", 64), ("iters", 25),
                        ("warmup", 5), ("seed", 7), ("tiles", "32,64"),
                        ("dtype", "fp16"), ("step_probes", 3),
                        ("cell_budget_ms", 200.0)]:
        moved = dataclasses.replace(base, **{knob: value})
        got = NF.run_id_for(moved, 1, gpu_name="NVIDIA H200", cache_mode="fresh")
        assert got != ref, f"moving {knob} did not change the run id"
    assert NF.run_id_for(base, 1, gpu_name="NVIDIA A100-SXM4-80GB",
                         cache_mode="fresh") != ref
    assert NF.run_id_for(base, 1, gpu_name="NVIDIA H200", cache_mode="warm") != ref


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
                         cache_mode="fresh")
    assert "rep1" in ours and "nvidiah200" in ours


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
                            {f: _fake_diff(f) for f in NF.ALPHA_FIELDS}, None)
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
    gate = NF.Gate("G", "claim", "rule", "PASS", False, "saw", "the headline")
    text = gate.render()
    assert "[FAIL]" in text and "invalidates: the headline" in text
    assert "expected PASS" in text, "an unexpected FAIL must be marked as one"


def test_an_unevaluated_gate_is_unknown_and_never_a_pass():
    text = NF.Gate("G", "claim", "rule", "PASS", None, "nothing ran").render()
    assert "[UNKNOWN]" in text and "PASS" not in text.split("\n")[0]
    assert "0 PASS, 0 FAIL, 1 UNKNOWN" in NF.render_gates(
        [NF.Gate("G", "c", "r", "PASS", None, "x")])


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
    assert all(g.passed is not True for g in gates)
    assert "0 PASS" in NF.render_gates(gates)


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
                            {f: NF.cross_card(f) for f in NF.ALPHA_FIELDS}, None)
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
    assert {a.model for a in NF.DEFAULT_ARMS} == {"mixtral-8x7b"}


def test_resolve_arms_refuses_an_unknown_name():
    with pytest.raises(SystemExit, match="unknown arm"):
        NF.resolve_arms("mixtral_g1,not_an_arm")
    with pytest.raises(SystemExit, match="at least one arm"):
        NF.resolve_arms("")


@needs_arms
def test_control_only_runs_off_gpu_and_reports_nothing_measured(capsys):
    assert NF.main(["--control-only"]) == NF.EXIT_NOT_MEASURED
    out = capsys.readouterr().out
    assert "PART (b) ONLY" in out and "no number on this page is one" in out
    assert "+0.0101" in out and "+0.0117" in out


@needs_arms
def test_dry_run_prints_the_plan_the_cost_and_the_registered_predictions(capsys):
    assert NF.main(["--dry-run", "--replicates", "6"]) == NF.EXIT_NOT_MEASURED
    out = capsys.readouterr().out
    assert "NOT A RESULT" in out
    assert "TOTAL" in out and "min of GPU" in out
    assert out.count("rep6-") == 2, "every replicate's run id must be in the plan"
    # The predictions have to carry NUMBERS and their expected verdicts before
    # anything is measured, or "registered in advance" means nothing.
    for expected in ("C4  the card beats the num_stages control         [FAIL]",
                     "C5  the cross-card sign is estimator-independent  [FAIL]",
                     "C1  the floor is no wider than the proxy implied  [PASS]",
                     "sd <= 0.0228", "|0.0117| < MDE"):
        assert expected in out, f"missing from the registered predictions: {expected}"
    # ...and the cost, which is what a metered pod is budgeted against.
    assert "wall-over-model factor" in out


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
    assert doc["schema"] == NF.SCHEMA
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
