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
import subprocess
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


@needs_arms
def test_the_registered_prior_reproduces_the_docstrings_mde_table():
    """The numbers the docstring registered N=6 against, recomputed."""
    assert NF.prior_sd() == pytest.approx(0.022863, abs=5e-6)
    assert NF.mde_two_sample(NF.prior_sd(), 3) == pytest.approx(0.0693, abs=5e-4)
    assert NF.mde_two_sample(NF.prior_sd(), 6) == pytest.approx(0.0410, abs=5e-4)
    assert NF.replicates_for(NF.EFFECTS[-1].size, NF.prior_sd()) == 61


@needs_arms
def test_the_prior_is_derived_from_the_control_and_is_not_a_retyped_rounding():
    """WHAT WAS PUBLISHED: `prior_sd` 0.022839549032325487 in the tracked
    `results/published/NOISE_FLOOR.json`, which is `0.0323 / sqrt(2)` -- the
    file's OWN paired sd, 0.03233250623999132, re-typed to three significant
    figures and then divided. 0.1% wrong, nothing in the file saying it was a
    rounding, and imported by `scripts/bn_decomposition.py` and by the session
    driver's MDE line. This asserts the identity, not the number, so the day
    the two committed arms change the constant cannot stay behind."""
    exact = NF.stages_control(NF.PRIMARY_FIELD).sd / math.sqrt(2.0)
    assert NF.prior_sd() == exact
    retyped = 0.0323 / math.sqrt(2.0)
    assert NF.prior_sd() != retyped, "the rounding is back"
    assert abs(NF.prior_sd() - retyped) > 2e-5


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

#: The arguments `run_id_for` is called with when nothing has moved. Every test
#: below states its change as a delta on this, so a new keyword reaches all of
#: them at once instead of being added to a list that a reader has to notice.
BASE_ID_KWARGS = dict(gpu_name="NVIDIA H200", cache_mode="fresh", sweep_args=(),
                      order=NF.ORDER_COUNTERBALANCED, python="/venv/bin/python")


def _id(arm=None, replicate=1, **over):
    return NF.run_id_for(arm or NF.DEFAULT_ARMS[0], replicate,
                         **{**BASE_ID_KWARGS, **over})


def test_six_replicates_get_six_distinct_run_ids():
    assert len({_id(replicate=i) for i in range(1, 7)}) == 6


#: Every `Arm` FIELD that must move the run id, with a value different from the
#: default. Checked for completeness against `dataclasses.fields(Arm)` by
#: `test_no_arm_field_is_silently_outside_the_run_id`.
ARM_FIELDS_MOVED = {
    "model": "qwen2-57b-a14b", "dtype": "fp16", "group_m": 16, "block_n": 256,
    "num_stages": 3, "tiles": "32,64", "r_max": 512, "row_step": 64,
    "step_probes": 3, "warmup_ms": 150.0, "trials": 5, "l2_flush": False,
    "cell_budget_ms": 200.0, "seed": 7,
}

#: The `Arm` fields that may NOT move it, with the reason.
ARM_FIELDS_NOT_IN_THE_ID = {
    "name": "labels the arm in the `{arm}-rep{i}` directory and in the log. "
            "Every field that decides what a cell CONTAINS is above, and two "
            "arms that agree on all of them are the same measurement under two "
            "labels",
}


@pytest.mark.parametrize("field_name,value", sorted(ARM_FIELDS_MOVED.items()))
def test_every_arm_field_that_moves_a_cell_moves_the_run_id(field_name, value):
    """Each field, moved alone, must move the id. One that does not is one whose
    second setting silently reports the first's numbers."""
    import dataclasses
    moved = dataclasses.replace(NF.DEFAULT_ARMS[0], **{field_name: value})
    assert _id(moved) != _id(), field_name


def test_no_arm_field_is_silently_outside_the_run_id():
    """THE LIST ABOVE CANNOT GO STALE. A field added to `Arm` joins no
    hand-written list on its own, and the suite stays green while the id stops
    separating two experiments."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(NF.Arm)}
    unclassified = fields - set(ARM_FIELDS_MOVED) - set(ARM_FIELDS_NOT_IN_THE_ID)
    assert not unclassified, (
        f"{sorted(unclassified)} are Arm fields that nothing says belong in or "
        "out of the run id. Add each to ARM_FIELDS_MOVED with a value that must "
        "move the id, or to ARM_FIELDS_NOT_IN_THE_ID with the reason it may not.")
    assert not (set(ARM_FIELDS_MOVED) & set(ARM_FIELDS_NOT_IN_THE_ID))
    assert not (set(ARM_FIELDS_MOVED) | set(ARM_FIELDS_NOT_IN_THE_ID)) - fields


#: Every `build_parser` knob that reaches the run id, with the route it takes and
#: the delta on `run_id_for`'s arguments that `main` actually produces from it.
#: The reason this exists rather than a prose list: the hand-written version of
#: it is what hid `--sweep-arg` and `--order`, and then hid `--python`.
PARSER_KNOBS_IN_THE_ID = {
    "gpu_name": ("the card, and it leads the id", dict(gpu_name="NVIDIA A100-SXM4-80GB")),
    "warm_cache": ("becomes `cache_mode`", dict(cache_mode="warm")),
    "order": ("the counterbalancing DESIGN, not an analysis knob",
              dict(order=NF.ORDER_PAIRED)),
    "python": ("the interpreter every child sweep runs under, so it selects the "
               "torch/triton/vLLM that measure every cell; the session driver "
               "varies it between PY_BASE and PY_VLLM",
               dict(python="/other/venv/bin/python")),
    "sweep_arg": ("the passthrough, entered verbatim",
                  dict(sweep_args=("--group-m", "16"))),
    "rehearse": ("main appends `--self-test <alpha>` to `extra`, and `extra` IS "
                 "what is passed as `sweep_args`",
                 dict(sweep_args=("--self-test", "0.5"))),
    "rehearse_noise": ("same route: `--self-test-noise` rides in `extra`",
                       dict(sweep_args=("--self-test-noise", "0.05"))),
    "arms": ("selects the `Arm`, whose every field is covered above",
             dict(arm=NF.DEFAULT_ARMS[1])),
    "replicates": ("bounds the replicate INDEX, which is in the id as `rep`",
                   dict(replicate=2)),
}

#: The knobs that may NOT move it, each with the reason. A knob here and a knob
#: in the map above are the same claim in opposite directions, and both are
#: claims: a wrong entry here is how a swept knob becomes an analysis knob by
#: assertion.
PARSER_KNOBS_NOT_IN_THE_ID = {
    "single_model_floor": "a refusal switch over the arms already in the id. It "
                          "changes what may be PUBLISHED, not what is measured",
    "floor_from": "chooses which cache mode may be published as THE floor; both "
                  "modes were measured under their own `cache` key",
    "out_dir": "names the base directory instead of deriving it",
    "replicate_timeout": "kills a hung sweep process. A replicate the timeout "
                         "cut short leaves a partial directory, and a longer "
                         "timeout resuming and FINISHING it is the intended "
                         "behaviour; putting it in the key would forbid that",
    "dry_run": "prints the plan and measures nothing",
    "control_only": "part (b) only, arithmetic over committed reports, no cell",
    "publish": "writes the document from cells that already exist",
}


@pytest.mark.parametrize("knob", sorted(PARSER_KNOBS_IN_THE_ID))
def test_every_parser_knob_that_selects_a_cell_moves_the_run_id(knob):
    """THE PASSTHROUGH, THE DESIGN AND THE INTERPRETER. `--sweep-arg` reaches the
    child sweep as an argument we do not parse, `--order` decides the
    counterbalancing, `--python` decides which venv measures. None of the three
    was in the key: a G=16 arm requested through the passthrough resumed the G=1
    arm's cells, a switch of design resumed the other design's, and two
    interpreters resumed one directory.
    """
    _reason, delta = PARSER_KNOBS_IN_THE_ID[knob]
    delta = dict(delta)
    arm = delta.pop("arm", None)
    replicate = delta.pop("replicate", 1)
    assert _id(arm, replicate, **delta) != _id(), knob


def test_no_parser_knob_is_silently_outside_the_run_id():
    """THE GUARD R2 SHOULD HAVE HAD. `--sweep-arg` and `--order` were found by an
    audit reading the file, and `--python` was still missing afterwards, because
    the test beside them was a hand-written list over `Arm` fields that a parser
    knob never had to join. The knobs are read off `build_parser` now and each
    must be either exercised above or refused BY NAME with a reason.
    """
    knobs = set(vars(NF.build_parser().parse_args([])))
    unclassified = knobs - set(PARSER_KNOBS_IN_THE_ID) - set(PARSER_KNOBS_NOT_IN_THE_ID)
    assert not unclassified, (
        f"{sorted(unclassified)} are knobs of replicate_noise_floor that nothing "
        "says belong in or out of the run id. Add each to PARSER_KNOBS_IN_THE_ID "
        "with the delta it produces, or to PARSER_KNOBS_NOT_IN_THE_ID with the "
        "reason it may not move the id.")
    assert not (set(PARSER_KNOBS_IN_THE_ID) & set(PARSER_KNOBS_NOT_IN_THE_ID))
    assert not (set(PARSER_KNOBS_IN_THE_ID) | set(PARSER_KNOBS_NOT_IN_THE_ID)) - knobs


def test_the_printed_plan_separates_two_interpreters_and_not_two_timeouts(
        capsys, monkeypatch):
    """END TO END, through `main`, because the maps above are a model of `main`
    and a model can be wrong. The plan is the artefact an operator reads before
    spending a pod hour, and the ids it prints are the ids the run resumes into.

    BOTH DIRECTIONS ARE ASSERTED. `--python` must separate, `--replicate-timeout`
    must not: a timeout that entered the key would give a re-run with a longer
    limit a fresh directory and forbid it from finishing the replicate the short
    one cut off.
    """
    # The real one launches the sweep's own --dry-run per arm, four seconds of
    # subprocess for a number this test does not read.
    monkeypatch.setattr(NF, "sweep_cost", lambda arm, python: 100.0)

    def plan_ids(extra):
        NF.main(["--dry-run", "--gpu-name", "NVIDIA H200"] + extra)
        return re.findall(r"rep \d+: (\S+)", capsys.readouterr().out)

    base = plan_ids(["--python", "/usr/bin/python3"])
    assert base, "the plan printed no run ids at all"
    assert plan_ids(["--python", "/other/venv/bin/python"]) != base
    assert plan_ids(["--python", "/usr/bin/python3",
                     "--replicate-timeout", "60"]) == base


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
    ours = _id()
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
                     # 0.0229 and not 0.0228: the prior is DERIVED from the
                     # control's own paired sd now, and the 0.0228 that was
                     # published was that sd re-typed to three significant
                     # figures and then divided. See `prior_sd`.
                     "sd <= 0.0229", "|0.0117| < MDE"):
        assert expected in out, f"missing from the registered predictions: {expected}"
    assert "RESULT:" not in out
    # ...and the cost, which is what a metered pod is budgeted against.
    assert "wall-over-model factor" in out


@needs_arms
def test_no_publish_path_can_write_an_unstamped_floor(monkeypatch, capsys):
    """THE TRACKED FILE IS NEVER WRITTEN WITHOUT PROVENANCE.

    `--control-only --publish` writes `results/published/NOISE_FLOOR.json`, and
    until 2026-09-02 it (and the then-writing no-GPU `--dry-run --publish`
    door) called `build_document` without `prov=`, so the committed floor
    named no machine, no instrument and no ruler. `prov` is a required
    positional argument now, so the FAIL branch is a TypeError at the call
    site rather than a silent hole in a published artefact, and that branch is
    planted here too.

    THE WRITE ITSELF IS FAKED HERE, and that is the limit of what this proves:
    it proves what the call site BUILDS, not what lands on disk. Whether it
    may land at all is the separate question
    `test_no_unmeasured_publish_path_can_delete_a_measured_floor` asks, and the
    answer changed on 2026-09-03: onto a file that already holds a measured
    floor, it may not. And since 2026-09-08 the blocked door (`--dry-run
    --publish`, or the pod line with no GPU) is not a publish path at all: it
    prints NOT PUBLISHED and calls nothing, which the second half plants.
    """
    with pytest.raises(TypeError, match="prov"):
        NF.build_document({}, {}, None)

    written: list[dict] = []
    monkeypatch.setattr(NF, "write_published",
                        lambda doc, *a, **k: written.append(doc) or "wrote (fake)")
    assert NF.main(["--control-only", "--publish"]) == NF.exit_codes.REFUSED
    assert len(written) == 1, "the deliberate part (b) regeneration still writes"
    assert NF.main(["--dry-run", "--publish", "--gpu-name", "NVIDIA H200"]) \
        == NF.exit_codes.REFUSED
    assert len(written) == 1, "the blocked door wrote a tracked file from a REFUSED run"
    out = capsys.readouterr().out
    assert "NOT PUBLISHED: --publish was given, but this run REFUSED" in out
    assert "left exactly as it was, stamp included" in out
    for doc in written:
        for key in NF.PV.TOP_LEVEL_KEYS:
            assert key in doc, key
        assert doc["provenance"]["git_sha"], "a published floor names its commit"
        # Nothing was measured on either page, and the instrument says so rather
        # than naming the live one.
        assert doc["instrument"] == NF.UNMEASURED_INSTRUMENT
        assert doc["replicate_floor"] is None


def test_a_refused_pod_line_leaves_the_tracked_floor_untouched(monkeypatch, capsys):
    """F8, RUN AS THE POD LINE MINUS THE GPU. `replicate_noise_floor.py
    --replicates 3 --arms mixtral_g1,mixtral_g16,qwen2_g1,qwen2_g16 --publish`
    exits 2 REFUSED off a GPU (detect_gpu reports what is missing) and, until
    2026-09-08, rewrote results/published/NOISE_FLOOR.json: `write_published`'s
    wall stops a null replacing a MEASURED floor and the committed floor is
    null, so the write went through and re-stamped written_utc, git and
    provenance on a tracked file from a run that measured nothing (git status
    ` M results/published/NOISE_FLOOR.json`, reproduced twice). Planted in both
    directions: the pod line calls the writer zero times, `--control-only
    --publish` still calls it once, and the tracked file's bytes are unchanged
    across the refused run."""
    try:
        import torch
        if torch.cuda.is_available():
            pytest.skip("a CUDA device is attached: this line IS the arm and would measure")
    except ImportError:
        pass
    before = NF.NOISE_FLOOR_JSON.read_bytes() if NF.NOISE_FLOOR_JSON.exists() else None
    calls: list[dict] = []
    monkeypatch.setattr(NF, "write_published",
                        lambda doc, *a, **k: calls.append(doc) or "wrote (fake)")
    rc = NF.main(["--replicates", "3", "--arms", "mixtral_g1,mixtral_g16,qwen2_g1,qwen2_g16",
                  "--publish"])
    out = capsys.readouterr().out
    assert rc == NF.exit_codes.REFUSED, out[-2000:]
    assert "NOT A RESULT. The replicate floor was not measured." in out
    assert calls == [], "a REFUSED run reached the writer"
    assert "NOT PUBLISHED: --publish was given, but this run REFUSED" in out
    assert "wrote " not in out.split("NOT A RESULT", 1)[1]
    after = NF.NOISE_FLOOR_JSON.read_bytes() if NF.NOISE_FLOOR_JSON.exists() else None
    assert after == before, "the tracked floor changed across a refused run"
    # The other direction: the deliberate regeneration is still a writer.
    assert NF.main(["--control-only", "--publish"]) == NF.exit_codes.REFUSED
    assert len(calls) == 1


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


@pytest.mark.skipif(not NF.NOISE_FLOOR_JSON.exists(),
                    reason="the noise floor has not been published here yet")
def test_the_committed_floor_carries_the_provenance_its_writer_now_requires():
    """R3, CLOSED IN CODE AND NOW ON DISK TOO. It XFAILED until 2026-09-03.

    `build_document` takes `prov` as a required positional argument, so no path
    can publish an unstamped floor. The file that was committed predated that
    and had none: `sorted(json.load(...))` was `[cross_card,
    effects_registered, git, primary_field, prior_sd, prior_sd_source,
    replicate_floor, schema, sign, stages_control, written_utc]`, with no
    provenance block, no git_sha, no gpu_name, no instrument, no ridge_source
    and no bandwidth_source. The test beside this one monkeypatches
    `write_published`, so it proves the WRITER and says nothing about the
    artefact, and the suite stayed green over an unattributed published number.

    THE XFAIL WAS THE DEBT AND THE DEBT IS PAID: the file was regenerated by
    `python scripts/replicate_noise_floor.py --control-only --publish`, which is
    arithmetic over the committed arms and needs no GPU. The assertions below
    now run for real.
    """
    doc = json.loads(NF.NOISE_FLOOR_JSON.read_text())
    assert "provenance" in doc, (
        "the committed floor lost its provenance block. Regenerate it: "
        "`python scripts/replicate_noise_floor.py --control-only --publish`")
    for key in NF.PV.TOP_LEVEL_KEYS:
        assert key in doc, key
    assert doc["provenance"]["git_sha"], "a published floor names its commit"
    assert doc["provenance"]["utc"]
    assert doc["instrument"], "and the ruler that produced it"
    # A page that measured nothing says so; it does not borrow the live name.
    if doc["replicate_floor"] is None:
        assert doc["instrument"] == NF.UNMEASURED_INSTRUMENT
    # And the dirt is NAMED, not a bare boolean. `dirty` cannot be false on the
    # commit that publishes -- the file is written before it can be committed --
    # so what a reader needs is which paths were outstanding, and the test below
    # is what makes the flag unnecessary.
    assert doc["git"]["dirty"] is False or doc["git"]["dirty_paths"], \
        "a dirty published floor must say WHICH paths were dirty"


@needs_arms
@pytest.mark.skipif(not NF.NOISE_FLOOR_JSON.exists(),
                    reason="the noise floor has not been published here yet")
def test_the_committed_floor_reproduces_from_the_committed_code():
    """THE GUARANTEE THAT REPLACES A CLEAN-TREE FLAG, and it is a stronger one.

    Every number in `results/published/NOISE_FLOOR.json` is arithmetic over two
    committed arms. This recomputes the whole document from those arms with the
    tracked code and compares it field by field, so a reader does not have to
    trust that whoever published it had a clean tree: they can rerun this. It
    catches a hand-edit of the file, a drift in the arms it is derived from, and
    the specific defect that was found in it -- `prior_sd` published as
    0.022839549032325487, which is the file's OWN paired sd re-typed to three
    significant figures and then divided by root two.

    `written_utc`, `git` and `provenance` are excluded because they describe the
    publishing act rather than the measurement, and no rerun can reproduce them.
    """
    committed = json.loads(NF.NOISE_FLOOR_JSON.read_text())
    control = {f: NF.stages_control(f) for f in NF.ALPHA_FIELDS}
    cards = {f: NF.cross_card(f) for f in NF.ALPHA_FIELDS}
    fresh = NF.build_document(control, cards, None, NF.unmeasured_provenance())
    describes_the_act = {"written_utc", "git", "provenance", "git_sha",
                         "gpu_name", "ridge_source", "bandwidth_source",
                         "instrument"}
    assert set(committed) == set(fresh)
    for key in sorted(set(fresh) - describes_the_act):
        assert committed[key] == fresh[key], key
    assert committed["prior_sd"] == pytest.approx(
        NF.stages_control(NF.PRIMARY_FIELD).sd / math.sqrt(2.0), rel=0, abs=0)
    assert committed["prior_sd"] != 0.022839549032325487, "the rounding is back"


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


@needs_arms
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
    assert swings["qwen2-57b-a14b"].size < NF.mde_two_sample(NF.prior_sd(), 6)
    assert swings["mixtral-8x7b"].size > NF.mde_two_sample(NF.prior_sd(), 6)


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


@needs_arms
def test_the_plan_states_its_scope_and_its_mde_from_a_stated_assumption():
    """B14 and A15: an MDE nobody stated, and a floor whose scope lived only in
    a docstring and was then cited for cross-pod comparisons."""
    arms = list(NF.DEFAULT_ARMS)
    text = "\n".join(NF.render_mde_line(6, arms))
    assert f"{NF.mde_two_sample(NF.prior_sd(), 6):.4f}" in text
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


def test_the_plan_names_the_directory_the_run_writes(tmp_path):
    """THE DEFECT THE PREFIX FIX INTRODUCED, PLANTED FROM BOTH SIDES.

    A rehearsal writes under `synthetic-`; the first version of that fix took
    the prefix in `run_replicate` and not in `render_plan`, so every rehearsal
    id in the printed plan was wrong by exactly the prefix the same commit had
    just added, and the plan named directories that were never written. The
    plan is the artefact an operator reads before spending a pod hour, so a log
    and a process disagreeing about one run is the whole failure.

    Both callers go through `synthetic_run_id`, and this asserts they agree in
    both directions: prefixed under `--self-test`, untouched without it.
    """
    nf = _load_script()
    arm = nf.DEFAULT_ARMS[0]
    kw = dict(gpu_name="NVIDIA H200", cache_mode="fresh", order="counterbalanced",
              python="/usr/bin/python3")
    bare = nf.run_id_for(arm, 1, sweep_args=[], **kw)

    planted = ["--self-test", "0.558", "--self-test-noise", "0.02"]
    assert nf.synthetic_run_id(bare, planted) == f"synthetic-{bare}"
    assert nf.synthetic_run_id(bare, []) == bare, \
        "a metered run must keep its own name"

    # And the two callers, through the code rather than through the helper.
    # n=2: render_plan prices a two-sample MDE and refuses fewer.
    plan = nf.render_plan([arm], 2, "fresh", tmp_path, "NVIDIA H200",
                          {arm.name: 1.0}, "counterbalanced", planted,
                          "/usr/bin/python3")
    ids = [ln.split("rep 1: ")[1].strip() for ln in plan.splitlines()
           if "rep 1: " in ln]
    assert ids and all(i.startswith("synthetic-") for i in ids), ids

    metered = nf.render_plan([arm], 2, "fresh", tmp_path, "NVIDIA H200",
                             {arm.name: 1.0}, "counterbalanced", [],
                             "/usr/bin/python3")
    mids = [ln.split("rep 1: ")[1].strip() for ln in metered.splitlines()
            if "rep 1: " in ln]
    assert mids and not any(i.startswith("synthetic-") for i in mids), mids


# --------------------------------------------------------------------------
# 7. THE MOST EXPENSIVE ARM RETURNED INVALID EVEN GIVEN INFINITE TIME.
#
# `run_replicate` discarded any child that exited non-zero WITHOUT opening its
# report.json, and `Replicate.ok` demanded a literal 0 a SECOND time. While
# `block_m_crossing_sweep` masked a failed CLAIM gate into exit 0 neither
# mattered; commit 346b7a5 stopped the masking, and from that commit a sweep
# that measured every cell and merely refuted its own pre-registered claim
# exited 1 and had every cell thrown away. On the pod: two hours of a rented
# card, a guaranteed INVALID, and an INVALID arm is latched by the session
# driver's ledger and skipped on every resume.
# --------------------------------------------------------------------------

def _fake_child(rc, *, report=None):
    """A `subprocess.run` that exits `rc` and, if asked, leaves a report behind.

    The report is written to the path `run_replicate` computed, which is the
    only way to prove the parent OPENED it rather than believing a returncode.

    `report` IS A DICT OR A CALLABLE OVER THE CHILD'S OWN ARGV, and the callable
    is not a convenience. A dict is the same page from every child, so every
    replicate of a cell comes back with bit-identical alpha, which is precisely
    the state V3 `not_a_collision` exists to refuse. A fake that cannot produce
    a spread cannot exercise a file whose entire subject is a spread: see
    `_measured_reports`.
    """
    def run(argv, **kwargs):
        doc = report(list(argv)) if callable(report) else report
        if doc is not None:
            out = Path(argv[argv.index("--out") + 1])
            run_id = argv[argv.index("--run-id") + 1]
            target = out / "block_m_crossing" / run_id / "report.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(doc))
        return subprocess.CompletedProcess(argv, rc, stdout="stdout", stderr="")
    return run


def _synthetic_report(alpha, *, instrument=None):
    """The shape `load_replicate` reads: one ladder fit and an instrument stamp.

    ONE PAGE, FIXED, for the single-`run_replicate` tests below, which are about
    what the parent does with an exit code and never pool anything.
    """
    return {
        "model": "mixtral-8x7b", "dtype": "bf16",
        "fixed": {"GROUP_SIZE_M": 1, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                  "num_warps": 8, "num_stages": 4},
        "instrument": NF.timing_basis() if instrument is None else instrument,
        "gates": [{"number": 0, "measured": "BM=64:3"}],
        "ladder": {"64": {"alpha": alpha, "alpha_corrected": alpha,
                          "alpha_upper": alpha}},
    }


#: How far apart two replicates of one cell are in the session-level fakes.
#: Inside `prior_sd()` (0.0229) so the floor C1 scores is a PASS rather than a
#: refutation of a proxy these tests never set out to test, and far enough from
#: zero that no rounding in the pooling can close it.
REPLICATE_STEP = 0.004


def _measured_reports(base_alpha, *, instrument=None):
    """A per-child report READ OFF THE CHILD'S OWN ARGV, for whole-`main` runs.

    TWO THINGS THE FIXED DICT GOT WRONG, and both of them decided a gate.

    The alpha was a constant, so both replicates of every cell returned the same
    bits, V3 `not_a_collision` FAILED, and the run reached INVALID for a reason
    that has nothing to do with the exit-code table these tests are about. The
    replicate index moves it by `REPLICATE_STEP` here, which is what a rerun of
    the same cell on the same card actually looks like.

    And `GROUP_SIZE_M` was hardcoded to 1 in every page, including the pages
    written for the G=16 arm, so the swizzle factor the C3 claim is scored over
    had one level wearing two labels. The swizzle width and the activation width
    are taken from the flags the parent really passed, so a cell lands where the
    parent thinks it launched it. `--out` carries `{arm}-rep{index}`, which is
    this parent's own naming and the only place the replicate index appears on a
    child command line: the sweep has no notion of a replicate.
    """
    def report(argv):
        arm_dir = Path(argv[argv.index("--out") + 1]).name
        index = int(arm_dir.rsplit("-rep", 1)[1])
        alpha = base_alpha + (index - 1) * REPLICATE_STEP
        doc = _synthetic_report(alpha, instrument=instrument)
        doc["model"] = argv[argv.index("--model") + 1]
        doc["dtype"] = argv[argv.index("--dtype") + 1]
        doc["fixed"]["GROUP_SIZE_M"] = int(argv[argv.index("--group-m") + 1])
        doc["fixed"]["BLOCK_SIZE_N"] = int(argv[argv.index("--block-n") + 1])
        doc["fixed"]["num_stages"] = int(argv[argv.index("--num-stages") + 1])
        return doc
    return report


def _one_replicate(rc, tmp_path, monkeypatch, *, report=None):
    monkeypatch.setattr(NF.subprocess, "run", _fake_child(rc, report=report))
    return NF.run_replicate(
        NF.DEFAULT_ARMS[0], 1, tmp_path, gpu_name="NVIDIA H200",
        cache_mode="fresh", python="/usr/bin/python3", extra=[],
        sweep_args=[], order=NF.ORDER_COUNTERBALANCED,
        shared_cache=None, timeout_s=60.0)


def test_a_child_that_measured_and_refuted_its_own_claim_is_pooled(tmp_path,
                                                                   monkeypatch):
    """CLAIM_FAIL is DONE-shaped: the experiment worked and the world
    disagreed. Its report is on disk and its alpha is the only thing this
    parent wants from it, so it is read and pooled."""
    rep = _one_replicate(NF.exit_codes.CLAIM_FAIL, tmp_path, monkeypatch,
                         report=_synthetic_report(0.55))
    assert rep.returncode == NF.exit_codes.CLAIM_FAIL
    assert rep.error == "", rep.error
    assert rep.ok, "a measured child was discarded for refuting its own claim"
    assert [c.values["alpha_corrected"] for c in rep.cells] == [0.55]
    assert rep.instrument == NF.timing_basis()


def test_a_child_that_refused_is_discarded_and_the_arm_says_which_code(
        tmp_path, monkeypatch):
    """THE PLANTED FAIL BRANCH. REFUSED measured nothing, so there is nothing
    to pool and nothing to read; the wall must still be there, and the error
    must name the code rather than a bare integer."""
    rep = _one_replicate(NF.exit_codes.REFUSED, tmp_path, monkeypatch)
    assert not rep.ok
    assert "REFUSED" in rep.error and "no cell of it is pooled" in rep.error
    assert rep.cells == []


def test_an_invalid_child_is_discarded_although_its_report_is_on_disk(
        tmp_path, monkeypatch):
    """INVALID measured and then failed its OWN validity gate, so its page is
    unquotable even though `exit_codes.MEASURED_CODES` counts it as measured.
    That set answers "keep the directory"; `REPORT_CODES` answers "pool the
    cells", and they are not the same question."""
    rep = _one_replicate(NF.exit_codes.INVALID, tmp_path, monkeypatch,
                         report=_synthetic_report(0.55))
    assert not rep.ok
    assert "INVALID" in rep.error
    assert NF.exit_codes.INVALID in NF.exit_codes.MEASURED_CODES
    assert NF.exit_codes.INVALID not in NF.REPORT_CODES


def test_an_off_table_exit_code_is_discarded_too(tmp_path, monkeypatch):
    """A signal or a shell 127 is a code nobody chose, so it carries no
    information and the cells behind it are not pooled."""
    rep = _one_replicate(137, tmp_path, monkeypatch,
                         report=_synthetic_report(0.55))
    assert not rep.ok
    assert "137" in rep.error


def test_both_places_that_judge_a_child_exit_code_read_the_same_tuple():
    """THE SECOND CALL SITE. The fix landed at `run_replicate` in the audit and
    `Replicate.ok` re-decided the identical question one screen away, so either
    one alone still threw the arm away. This fails if a bare `== 0` comes
    back to either."""
    source = (ROOT / "scripts" / "replicate_noise_floor.py").read_text()
    assert "self.returncode == 0" not in source
    assert "done.returncode != 0" not in source
    assert source.count("in REPORT_CODES") == 2, \
        "exactly two places judge a measured child, and both read REPORT_CODES"
    assert NF.REPORT_CODES == (NF.exit_codes.DONE, NF.exit_codes.CLAIM_FAIL)


def test_the_whole_arm_reaches_a_quotable_verdict_when_every_child_claim_failed(
        tmp_path, monkeypatch):
    """END TO END, which is where the two hours were lost. Four children, each
    exiting CLAIM_FAIL with a full report: V2 must PASS with `4 of 4
    replicates ok`, and the arm's own exit code must not be INVALID."""
    if not HAVE_ARMS:
        pytest.skip("the committed arms are not checked out")
    arms = [a for a in NF.DEFAULT_ARMS if a.model == "mixtral-8x7b"]
    reps = []
    for position, (arm, index) in enumerate(NF.run_order(arms, 2,
                                                         NF.ORDER_COUNTERBALANCED)):
        monkeypatch.setattr(
            NF.subprocess, "run",
            _fake_child(NF.exit_codes.CLAIM_FAIL,
                        report=_synthetic_report(0.55 + 0.01 * position)))
        reps.append(NF.run_replicate(
            arm, index, tmp_path, gpu_name="NVIDIA H200", cache_mode="fresh",
            python="/usr/bin/python3", extra=[], sweep_args=[],
            order=NF.ORDER_COUNTERBALANCED, shared_cache=None, timeout_s=60.0))
    floors = {f: NF.pool(NF.spreads_for(reps, f), f) for f in NF.ALPHA_FIELDS}
    gates = NF.validity_gates(reps, 2, arms, "fresh", floors,
                              single_model_ok=True)
    v2 = next(g for g in gates if g.name == "V2 non-vacuity")
    assert v2.passed is True, v2.observed
    assert "4 of 4 replicates ok (4 CLAIM_FAIL)" in v2.observed
    assert NF.exit_codes.classify(g.scored() for g in gates) != \
        NF.exit_codes.INVALID
    # And the planted opposite: the same four children REFUSING must FAIL V2.
    refused = [NF.Replicate(r.arm, r.index, r.run_id, r.out_dir, r.report,
                            returncode=NF.exit_codes.REFUSED,
                            error="sweep exited 2 REFUSED") for r in reps]
    empty = {f: NF.pool(NF.spreads_for(refused, f), f) for f in NF.ALPHA_FIELDS}
    bad = NF.validity_gates(refused, 2, arms, "fresh", empty,
                            single_model_ok=True)
    bad_v2 = next(g for g in bad if g.name == "V2 non-vacuity")
    assert bad_v2.passed is False
    assert "0 of 4 replicates ok (4 REFUSED)" in bad_v2.observed
    assert NF.exit_codes.classify(g.scored() for g in bad) == \
        NF.exit_codes.INVALID


def test_the_code_census_names_every_state_in_table_order():
    """The V2 line has to say WHICH codes came back. A floor pooled over four
    children that each refuted their own claim is a different artefact from one
    pooled over four clean children, and on a metered pod nobody opens four
    child logs to find out."""
    def rep(rc):
        return NF.Replicate("a", 1, "id", Path("/tmp"), Path("/tmp/r.json"),
                            returncode=rc)
    census = NF.code_census([rep(NF.exit_codes.CLAIM_FAIL),
                             rep(NF.exit_codes.DONE),
                             rep(NF.exit_codes.REFUSED),
                             rep(NF.exit_codes.DONE)])
    assert census == "2 DONE, 1 CLAIM_FAIL, 1 REFUSED"
    assert NF.code_census([]) == "nothing launched"
    assert "off-table 137" in NF.code_census([rep(137)])
    assert "no exit code" in NF.code_census([rep(None)])


# --------------------------------------------------------------------------
# 8. THE CARD IS NEVER INVENTED.
# --------------------------------------------------------------------------

def test_a_run_that_names_no_card_is_labelled_nocard_and_never_an_h200():
    """IT USED TO BE `args.gpu_name or device or "NVIDIA H200"`, so a laptop
    that named no card wrote `nvidia_h200-fresh-n6/` and stamped `NVIDIA H200`
    into every run id and into the published JSON, indistinguishable in `ls`,
    in the id and in the file from a run on the rented card. The session
    driver's own rule is that every path carries the card or `nocard`."""
    named, why = NF.resolve_card("NVIDIA H200", "", [])
    assert (named, why) == ("NVIDIA H200", "named by --gpu-name")
    live, why = NF.resolve_card("", "NVIDIA A100-SXM4-80GB", [])
    assert live == "NVIDIA A100-SXM4-80GB" and "attached device" in why
    # --gpu-name wins over the live device: pricing a plan for a card you are
    # about to rent is the supported case.
    assert NF.resolve_card("NVIDIA H200", "NVIDIA A100", [])[0] == "NVIDIA H200"
    none, why = NF.resolve_card("", "", ["no CUDA device"])
    assert none == NF.NO_CARD == "nocard"
    assert "no CUDA device" in why
    assert NF.PV.card_slug(none) == "nocard"


def test_the_base_directory_of_a_cardless_run_carries_nocard(capsys, monkeypatch,
                                                             tmp_path):
    """END TO END through `main`, because the resolver is a model of main and a
    model can be wrong. This is the artefact the defect was visible in: a
    directory name."""
    if not HAVE_ARMS:
        pytest.skip("the committed arms are not checked out")
    monkeypatch.setattr(NF, "detect_gpu", lambda: ("", ["no CUDA device"]))
    monkeypatch.setattr(NF, "sweep_cost", lambda arm, python: 100.0)
    NF.main(["--dry-run", "--replicates", "2", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert str(tmp_path / "nocard-fresh-n2") in out
    assert "card     nocard -- no card could be named: no CUDA device" in out
    assert "config device nocard" in out
    # Part (b) legitimately names the two COMMITTED H200 arms it is arithmetic
    # over, so the assertion is on the lines that describe THIS run: none of
    # them may name a card nobody attached.
    mine = [ln for ln in out.splitlines()
            if ln.startswith(("card ", "EVERYTHING IS SAVED TO", "  rep "))
            or "arm(s) x" in ln or "rep 1: " in ln or "rep 2: " in ln]
    assert mine
    assert not [ln for ln in mine if "nvidia_h200" in ln.lower()], mine


# --------------------------------------------------------------------------
# 9. THE WHOLE ARM, THROUGH `main`, TO AN EXIT CODE.
#
# Section 7 proves the gates. This proves the number the session driver's
# ledger actually reads, because that is what the two hours were spent on: a
# ledger row. It cannot be proved by `--rehearse`, and the reason is worth
# writing down. A rehearsal's cells carry the SYNTHETIC instrument stamp, V6
# refuses them, and `rehearsal_exit` forces INVALID whatever the gates said, so
# the exact off-GPU command in the audit still exits 3 -- correctly, and now
# with `4 of 4 replicates ok (4 CLAIM_FAIL)` on its V2 line where it used to
# read `0 of 4`. The only way to see the code the pod will see is to hand the
# parent children that are stamped with the instrument this repo publishes
# under, which is what these two do.
# --------------------------------------------------------------------------

def _fake_session(rc, *, report=None):
    """`subprocess.run` for a whole `main`: plants sweep children, passes git.

    `main` shells out for more than the replicates (`git_state` names the
    commit the floor came from), so a fake that answered everything would be
    faking the provenance too. Anything without `--run-id` in its argv is not a
    sweep and goes to the real `subprocess.run`.
    """
    real = subprocess.run
    child = _fake_child(rc, report=report)

    def run(argv, **kwargs):
        if "--run-id" not in list(argv):
            return real(argv, **kwargs)
        return child(argv, **kwargs)
    return run


def _metered_main(rc, tmp_path, monkeypatch, capsys, *, alpha_report, extra=()):
    monkeypatch.setattr(NF, "detect_gpu", lambda: ("NVIDIA H200", []))
    monkeypatch.setattr(NF, "sweep_cost", lambda arm, python: 100.0)
    monkeypatch.setattr(NF.subprocess, "run",
                        _fake_session(rc, report=alpha_report))
    code = NF.main(["--replicates", "2", "--arms", "mixtral_g1,mixtral_g16",
                    "--single-model-floor", "--gpu-name", "NVIDIA H200",
                    "--out-dir", str(tmp_path)] + list(extra))
    return code, capsys.readouterr().out


def test_the_session_fake_gives_each_replicate_of_a_cell_its_own_alpha(tmp_path):
    """THE FAKE IS AN INSTRUMENT TOO, so its one load-bearing property is
    asserted rather than assumed. A fake that hands every replicate the same
    number sends V3 to FAIL and every run built on it to INVALID, which is the
    exact verdict the tests below exist to distinguish from the bug."""
    report = _measured_reports(0.55)
    seen = {}
    for arm in ("mixtral_g1", "mixtral_g16"):
        for index in (1, 2):
            group_m = 1 if arm.endswith("g1") else 16
            doc = report(["--model", "mixtral-8x7b", "--dtype", "bf16",
                          "--group-m", str(group_m), "--block-n", "64",
                          "--num-stages", "4",
                          "--out", str(tmp_path / f"{arm}-rep{index}")])
            seen[(arm, index)] = doc
    for arm in ("mixtral_g1", "mixtral_g16"):
        first = seen[(arm, 1)]["ladder"]["64"]["alpha_corrected"]
        second = seen[(arm, 2)]["ladder"]["64"]["alpha_corrected"]
        assert second - first == pytest.approx(REPLICATE_STEP)
    assert seen[("mixtral_g1", 1)]["fixed"]["GROUP_SIZE_M"] == 1
    assert seen[("mixtral_g16", 1)]["fixed"]["GROUP_SIZE_M"] == 16


@needs_arms
def test_main_reaches_a_quotable_code_when_every_child_refuted_its_claim(
        tmp_path, monkeypatch, capsys):
    """THE ROW THE POD WOULD HAVE WRITTEN. Four children that measured every
    cell and failed their own CLAIM gate: the arm is CLAIM_FAIL or DONE, which
    is what `exit_codes.ledger_state` records as finished-with-a-result. It
    used to be INVALID, which the driver latches and skips on every resume."""
    code, out = _metered_main(NF.exit_codes.CLAIM_FAIL, tmp_path, monkeypatch,
                              capsys, alpha_report=_measured_reports(0.55))
    assert "4 of 4 replicates ok (4 CLAIM_FAIL)" in out
    assert code in (NF.exit_codes.DONE, NF.exit_codes.CLAIM_FAIL), out[-3000:]
    assert NF.exit_codes.ledger_state(code) in ("DONE", "CLAIM_FAIL")
    assert "RESULT: VALIDITY V2_non-vacuity PASS" in out
    # EVERY validity gate, not only V2: the arm is quotable or it is not, and
    # one VALIDITY FAIL anywhere puts it back at INVALID.
    assert not [r for r in NF.exit_codes.parse_result_lines(out)
                if r.kind == NF.exit_codes.VALIDITY and r.verdict == "FAIL"], out
    # The pooled floor is a number, not a null: that is the artefact the arm
    # is rented for, and an INVALID arm never produces one.
    assert "between-replicate sd" in out
    assert "part (a) did not run" not in out
    assert "REHEARSAL:" not in out


@needs_arms
def test_main_stays_invalid_when_every_child_refused(tmp_path, monkeypatch,
                                                     capsys):
    """THE PLANTED FAIL BRANCH OF THE SAME PATH. REFUSED children measured
    nothing, so there is nothing to pool, and the wall the fix walked through
    for CLAIM_FAIL must still be standing here. INVALID, and the V2 line says
    which code came back so the operator does not have to open four logs."""
    code, out = _metered_main(NF.exit_codes.REFUSED, tmp_path, monkeypatch,
                              capsys, alpha_report=None)
    assert code == NF.exit_codes.INVALID, out[-3000:]
    assert "0 of 4 replicates ok (4 REFUSED)" in out
    assert "RESULT: VALIDITY V2_non-vacuity FAIL" in out


def _permits(gates, *, withheld=False, cache_mode="fresh", floor_from="fresh"):
    return NF.gates_permit_publishing(gates, floor_withheld=withheld,
                                      cache_mode=cache_mode,
                                      floor_from=floor_from)


def test_a_validity_failure_blocks_the_tracked_floor_and_a_claim_failure_does_not():
    """`gates_permit_publishing`, all three verdicts on both kinds of gate.

    The distinction is the whole point: a CLAIM that failed is a result ABOUT
    the floor and must still publish it, a VALIDITY that failed says the floor
    is not a floor. UNKNOWN is refused with FAIL, the same way
    `exit_codes.classify` scores it.
    """
    def gate(kind, name, passed):
        return NF.Gate(kind, name, "p", "r", "PASS", passed, "o")

    clean = [gate(NF.exit_codes.VALIDITY, "V6 one instrument", True),
             gate(NF.exit_codes.CLAIM, "C1 floor size", True)]
    assert _permits(clean) == (True, "")

    # A refuted claim publishes: the spread was measured soundly.
    refuted = [gate(NF.exit_codes.VALIDITY, "V6 one instrument", True),
               gate(NF.exit_codes.CLAIM, "C1 floor size", False),
               gate(NF.exit_codes.CLAIM, "C3 swizzle", None)]
    assert _permits(refuted) == (True, "")

    for verdict, word in ((False, "FAIL"), (None, "UNKNOWN")):
        broken = [gate(NF.exit_codes.VALIDITY, "V6 one instrument", verdict),
                  gate(NF.exit_codes.CLAIM, "C1 floor size", True)]
        allowed, why = _permits(broken)
        assert allowed is False
        assert "V6 one instrument" in why and word in why


def test_a_run_whose_floor_is_withheld_may_not_overwrite_one_that_is_not():
    """THE OTHER WAY IN, AND IT WRITES A NULL OVER A MEASUREMENT.

    `--warm-cache` with `--floor-from fresh` measures cells and is then handed
    `floors=None`, so its document carries `replicate_floor: null`. Every gate
    passes, so a gates-only wall lets it through and it replaces the fresh run's
    measured floor with nothing. The page even printed "NOT publishable as THE
    floor" one line above the write.
    """
    def gate(kind, name, passed):
        return NF.Gate(kind, name, "p", "r", "PASS", passed, "o")

    clean = [gate(NF.exit_codes.VALIDITY, "V6 one instrument", True)]
    allowed, why = _permits(clean, withheld=True, cache_mode="warm",
                            floor_from="fresh")
    assert allowed is False
    assert "replicate_floor: null" in why and "--floor-from warm" in why
    # And the mode that IS the floor mode is not caught by it.
    assert _permits(clean, withheld=False, cache_mode="warm",
                    floor_from="warm") == (True, "")


@needs_arms
def test_publish_writes_the_tracked_floor_when_only_a_claim_failed(
        tmp_path, monkeypatch, capsys):
    """The pod's intended path: the session driver runs this arm with
    `--publish`, its children refute their own claim, and the floor -- the one
    artefact the card was rented for -- reaches `results/published`."""
    written = []
    monkeypatch.setattr(NF, "write_published",
                        lambda doc, path=None: written.append(doc) or "wrote it")
    code, out = _metered_main(NF.exit_codes.CLAIM_FAIL, tmp_path, monkeypatch,
                              capsys, alpha_report=_measured_reports(0.55),
                              extra=["--publish"])
    assert code == NF.exit_codes.CLAIM_FAIL, out[-2000:]
    assert len(written) == 1, "a CLAIM failure must not withhold the floor"
    assert written[0]["replicate_floor"]["per_field"]["alpha_corrected"]["sd"] > 0
    assert "REFUSING --publish" not in out


@needs_arms
def test_publish_is_refused_when_a_validity_gate_failed(tmp_path, monkeypatch,
                                                        capsys):
    """THE PLANTED FAIL BRANCH OF THE PUBLISH WALL, and the defect it closes.

    Children stamped with a DIFFERENT instrument fail V6, so the run's spread
    pools two timing loops and means nothing; the run exits INVALID and used to
    write `results/published/NOISE_FLOOR.json` on its way there, because the
    only question that branch asked was whether it was a rehearsal. The log said
    "nothing quotable" and git got a floor.
    """
    written = []
    monkeypatch.setattr(NF, "write_published",
                        lambda doc, path=None: written.append(doc) or "wrote it")
    code, out = _metered_main(
        NF.exit_codes.CLAIM_FAIL, tmp_path, monkeypatch, capsys,
        alpha_report=_measured_reports(0.55, instrument="some/other/loop/v9"),
        extra=["--publish"])
    assert code == NF.exit_codes.INVALID, out[-2000:]
    assert "RESULT: VALIDITY V6_one_instrument FAIL" in out
    assert written == [], "an INVALID run wrote the tracked floor anyway"
    assert "REFUSING --publish" in out and "V6 one instrument (FAIL)" in out


def test_a_rehearsal_may_not_exit_with_a_quotable_code():
    """`rehearsal_exit`, both branches. The INVALID a rehearsal reaches on its
    own is passed through WITH the gates that carried it named; a rehearsal
    that somehow classified DONE is forced to INVALID and the line says the
    wall broke rather than that a floor was measured."""
    def gate(kind, name, passed):
        return NF.Gate(kind, name, "p", "r", "PASS", passed, "o")

    carried = [gate(NF.exit_codes.VALIDITY, "V6 one instrument", False),
               gate(NF.exit_codes.CLAIM, "C1 floor size", True)]
    rc, why = NF.rehearsal_exit(NF.exit_codes.INVALID, carried)
    assert rc == NF.exit_codes.INVALID
    assert "by construction" in why and "V6 one instrument" in why

    clean = [gate(NF.exit_codes.VALIDITY, "V6 one instrument", True),
             gate(NF.exit_codes.CLAIM, "C1 floor size", True)]
    for got in (NF.exit_codes.DONE, NF.exit_codes.CLAIM_FAIL):
        rc, why = NF.rehearsal_exit(got, clean)
        assert rc == NF.exit_codes.INVALID
        assert "may not exit with" in why and "GENERATED" in why


@needs_arms
def test_the_rehearsal_the_audit_ran_now_pools_its_children_and_still_refuses(
        tmp_path, monkeypatch, capsys):
    """THE AUDIT'S EXACT SHAPE, and both halves of what changed.

    `--rehearse 0.4 --replicates 2 --arms mixtral_g1,mixtral_g16
    --single-model-floor` returned `0 of 4 replicates ok`, V2 FAIL and every
    claim UNKNOWN, because the sweep's self-test refutes its own C-gates and
    exits CLAIM_FAIL and the parent threw all four reports away unread. Now the
    four are pooled, V2 PASSES and C1 scores; the run still exits INVALID, and
    that is `rehearsal_exit` and V6 doing their job, not the bug.
    """
    monkeypatch.setattr(NF, "detect_gpu", lambda: ("", ["no CUDA device"]))
    monkeypatch.setattr(NF, "sweep_cost", lambda arm, python: 100.0)
    monkeypatch.setattr(NF.subprocess, "run", _fake_session(
        NF.exit_codes.CLAIM_FAIL,
        report=_measured_reports(0.4, instrument="synthetic/model-generated")))
    code = NF.main(["--rehearse", "0.4", "--replicates", "2", "--arms",
                    "mixtral_g1,mixtral_g16", "--single-model-floor",
                    "--gpu-name", "NVIDIA H200", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "4 of 4 replicates ok (4 CLAIM_FAIL)" in out
    assert "RESULT: VALIDITY V2_non-vacuity PASS" in out
    assert "UNKNOWN" not in [r.verdict for r
                             in NF.exit_codes.parse_result_lines(out)
                             if r.name.startswith("C1")]
    assert code == NF.exit_codes.INVALID
    assert "REHEARSAL: INVALID by construction" in out


# --------------------------------------------------------------------------
# 10. THE TWO WAYS 120 MINUTES ARE LOST WITHOUT A SINGLE NUMBER BEING WRONG.
#
# Both are about the code the ledger reads and the file git keeps, not about
# the floor. An arm can measure perfectly and still be worthless if the row it
# writes says the wrong thing, or if the artefact it was rented for is deleted
# by the same run that refused to produce one. Every test here plants its FAIL
# branch: a passing wall that has never been shown to stop anything is the
# "check that examined nothing" shape this repository is named against.
# --------------------------------------------------------------------------

def _measured_floor_doc(sd=0.0123, *, synthetic=False):
    """The smallest document that counts as a floor a card produced."""
    return {
        "schema": NF.SCHEMA,
        "prior_sd": 0.022862534415054224,
        "prior_sd_source": "the s3/s4 proxy",
        "replicate_floor": {
            "n_replicates": 6, "cache_mode": "fresh", "gpu_name": "NVIDIA H200",
            "provenance": "6 replicates on NVIDIA H200", "synthetic": synthetic,
            "instrument": "queue-deep/l2-flush/clock-under-load/v2",
            "scope": None,
            "per_field": {NF.PRIMARY_FIELD: {
                "sd": sd, "df": 10, "upper95": sd * 1.5, "pooled": True,
                "reason": "pooled over 2 cells", "cells": 2, "per_cell": []}},
        },
    }


def test_an_unplanned_exception_is_error_and_not_a_claim_that_failed(
        monkeypatch, capsys):
    """THE ONE PATH THAT STILL WASTED THE BOOKING, and it wasted it silently.

    Unhandled, an exception exits the interpreter ONE, and ONE is CLAIM_FAIL: a
    RESULT, latched by the session driver, skipped on every resume, RETRY_ARMS
    0, session exits 0. A torch OOM in the last minute of a 120-minute arm was
    filed as one of this experiment's registered findings. ERROR (4) is outside
    the finished codes precisely so "the apparatus broke" can be told from "the
    claim did not hold".
    """
    def boom(argv=None):
        raise RuntimeError("torch OOM on the pod")

    monkeypatch.setattr(NF, "_main", boom)
    assert NF.main([]) == NF.exit_codes.ERROR
    err = capsys.readouterr().err
    assert "torch OOM on the pod" in err, "the traceback was swallowed"
    assert "RuntimeError" in err
    assert NF.exit_codes.ledger_state(NF.exit_codes.ERROR) == "RETRY"
    assert NF.exit_codes.ledger_state(NF.exit_codes.CLAIM_FAIL) != "RETRY"

    # ...and the PASS branch: a run that reaches a verdict keeps its own code.
    monkeypatch.setattr(NF, "_main", lambda argv=None: NF.exit_codes.CLAIM_FAIL)
    assert NF.main([]) == NF.exit_codes.CLAIM_FAIL


def test_a_refusal_sentence_is_refused_and_not_a_claim_that_failed(
        monkeypatch, capsys):
    """`raise SystemExit("...")` exits ONE too, and this file has five of them.

    Two are in `write_published` and can fire AFTER the card has been paid for,
    which is why the mapping is here and not at the raise sites. An argparse
    failure already exits 2 and must pass through as itself.
    """
    assert NF.main(["--replicates", "1"]) == NF.exit_codes.REFUSED
    assert "REFUSED: --replicates below 2" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        NF.main(["--not-an-argument"])
    assert exc.value.code == 2, "argparse's own code must pass through"
    capsys.readouterr()

    monkeypatch.setattr(NF, "_main", lambda argv=None: (_ for _ in ()).throw(
        SystemExit("REFUSING to write /x: git ignores it")))
    assert NF.main([]) == NF.exit_codes.REFUSED
    err = capsys.readouterr().err
    assert err.startswith("REFUSING to write"), "the sentence was re-prefixed"


def test_published_floor_is_measured_says_cannot_tell_rather_than_no(tmp_path):
    """Three answers, because "unreadable" and "empty" are not the same file."""
    absent = tmp_path / "gone.json"
    assert NF.published_floor_is_measured(absent) is False

    null = tmp_path / "null.json"
    null.write_text(json.dumps({"schema": NF.SCHEMA, "replicate_floor": None}))
    assert NF.published_floor_is_measured(null) is False

    fake = tmp_path / "rehearsed.json"
    fake.write_text(json.dumps(_measured_floor_doc(synthetic=True)))
    assert NF.published_floor_is_measured(fake) is False

    real = tmp_path / "real.json"
    real.write_text(json.dumps(_measured_floor_doc()))
    assert NF.published_floor_is_measured(real) is True

    torn = tmp_path / "torn.json"
    torn.write_text('{"schema": "moe-kernels/noise-flo')
    assert NF.published_floor_is_measured(torn) is None

    alien = tmp_path / "alien.json"
    alien.write_text(json.dumps({"schema": "somebody/else/1"}))
    assert NF.published_floor_is_measured(alien) is None


def test_a_null_floor_may_not_replace_a_measured_one(tmp_path, monkeypatch):
    """THE DOOR THE GATE WALL DOES NOT COVER, at the one place all three open.

    `gates_permit_publishing` stands in the MEASURED publish path only. This is
    the write itself refusing, so `--control-only --publish`, the `blocked`
    path and any caller added later are all covered by one rule.
    """
    monkeypatch.setattr(NF, "git_accepts", lambda p: True)
    target = tmp_path / "NOISE_FLOOR.json"

    # PASS branch first: onto a file with nothing to lose, a null floor lands.
    assert "wrote" in NF.write_published({"replicate_floor": None}, target)
    assert json.loads(target.read_text())["replicate_floor"] is None

    # FAIL branch: onto a measured floor it refuses, and leaves it untouched.
    target.write_text(json.dumps(_measured_floor_doc(0.0177)))
    before = target.read_text()
    with pytest.raises(SystemExit, match="REFUSING to write"):
        NF.write_published({"replicate_floor": None}, target)
    assert target.read_text() == before
    assert NF.noise_floor(target).sd == 0.0177

    # A measured document may replace a measured document: this is the arm.
    assert "wrote" in NF.write_published(_measured_floor_doc(0.0201), target)
    assert NF.noise_floor(target).sd == 0.0201

    # And "cannot tell" refuses as well, because deciding "empty" from a file
    # we failed to open is how a measurement gets deleted by a blank.
    target.write_text("{not json")
    with pytest.raises(SystemExit, match="cannot be read as a floor document"):
        NF.write_published({"replicate_floor": None}, target)
    assert target.read_text() == "{not json"


@needs_arms
def test_no_unmeasured_publish_path_can_delete_a_measured_floor(
        tmp_path, monkeypatch, capsys):
    """THE VERIFIER'S REPRO, through `main`, at both doors that reached it.

    `--dry-run --publish` and `--control-only --publish` both print "NOT A
    RESULT. The replicate floor was not measured", both return REFUSED, and
    both wrote `replicate_floor: null` over the tracked file on the way. The
    session driver's real branch passes `--publish` bare, so a GPU probe coming
    back empty on the pod did both at once: refused the arm and deleted the
    floor an earlier pod had paid for.

    Since 2026-09-08 the two doors differ: `--control-only --publish` still
    reaches the writer and is stopped by its wall (REFUSING to write, on
    stderr); the blocked door does not reach the writer at all and says NOT
    PUBLISHED on stdout. The floor is untouched either way.
    """
    target = tmp_path / "NOISE_FLOOR.json"
    target.write_text(json.dumps(_measured_floor_doc(0.0155)))
    before = target.read_text()
    monkeypatch.setattr(NF, "NOISE_FLOOR_JSON", target)
    monkeypatch.setattr(NF, "git_accepts", lambda p: True)
    monkeypatch.setattr(NF, "sweep_cost", lambda arm, python: 100.0)

    assert NF.main(["--control-only", "--publish"]) == NF.exit_codes.REFUSED
    assert target.read_text() == before
    assert "REFUSING to write" in capsys.readouterr().err
    blocked = ["--dry-run", "--publish", "--replicates", "3",
               "--arms", "mixtral_g1,mixtral_g16", "--out-dir", str(tmp_path / "out")]
    assert NF.main(blocked) == NF.exit_codes.REFUSED
    assert target.read_text() == before
    captured = capsys.readouterr()
    assert "NOT PUBLISHED: --publish was given, but this run REFUSED" in captured.out
    assert "REFUSING to write" not in captured.err, "the blocked door reached the writer"
    assert NF.noise_floor(target).sd == 0.0155


def test_sizing_sigma_labels_the_number_it_returns(tmp_path):
    """The fallback every consumer of this file needs, written once.

    `noise_floor()` raises, which is right for a caller that must stop and
    wrong for the three that must print a limit every session. Two of them
    (`alpha_surface`, `bn_decomposition`) read `prior_sd` straight out of the
    JSON and would go on reading the s3/s4 PROXY after this arm publishes a
    measurement into the same file, which is what makes this an accessor and
    not a note in a docstring.
    """
    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps(_measured_floor_doc(0.0133)))
    sd, basis, source = NF.sizing_sigma(measured)
    assert (sd, basis) == (0.0133, "MEASURED")
    assert "NVIDIA H200" in source

    null = tmp_path / "null.json"
    doc = _measured_floor_doc()
    doc["replicate_floor"] = None
    null.write_text(json.dumps(doc))
    sd, basis, source = NF.sizing_sigma(null)
    assert (sd, basis) == (0.022862534415054224, "ASSUMED")
    assert "s3/s4 proxy" in source and "no measured floor" in source

    # A rehearsal floor is not a measurement, and must not be quoted as one.
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text(json.dumps(_measured_floor_doc(0.9, synthetic=True)))
    assert NF.sizing_sigma(synthetic)[1] == "ASSUMED"

    # And it invents nothing when there is nothing.
    with pytest.raises(NF.NoiseFloorUnmeasured, match="no readable prior_sd"):
        NF.sizing_sigma(tmp_path / "absent.json")


@needs_arms
def test_every_mode_names_the_accessor_and_who_still_reads_the_proxy(capsys):
    """The decision an operator must make BEFORE the rental is printed by every
    mode, because after it the 120 minutes are spent either way."""
    NF.main(["--control-only"])
    out = capsys.readouterr().out
    assert "sizing_sigma()" in out
    assert "alpha_surface.py:prior_sd" in out
    assert "bn_decomposition.py:" in out and "published_prior_sd" in out
