"""The dtype/tile decomposition, and the seven ways it could lie.

`scripts/dtype_tile_confound.py` answers one question -- how much of the
published fp8/bf16 crossing shift of 1.149 is the DTYPE and how much is the
CONFIG vLLM picks differently for the two formats -- and the answer is a ratio
of ratios. Every test below belongs to one of the ways that answer can be wrong
while still printing:

1. THE SIGN IS BACKWARDS. `tilt > 1` means fp8 crosses LATER. Inverted, the
   conclusion flips from "the format barely moves the crossing" to "quantising
   moves the ridge", which is a far larger claim and would be made by accident.
   Every printed tilt carries the word LATER or EARLIER and the tests read those
   words rather than the number.
2. THE TWO SIDES ARE THE SAME KERNEL. If `override_config` forced nothing, both
   matched arms would be the native arm and the script would report a tight
   1.000 meaning "the experiment did not happen". V3 has to read UNKNOWN when
   nothing was watched and FAIL when one kernel ran everywhere -- never PASS.
3. THE SELF TEST CERTIFIES ITSELF. The synthetic generator produces rows a
   laptop can reduce; if those rows also carried an observed config and a weight
   dtype, three validity gates would PASS on a machine with no GPU, no vLLM and
   no kernel. That happened during development and these tests pin the fix.
4. THE GATE CANNOT FAIL. C3 is scored against a planted world, so the suite runs
   the generator in three worlds and asserts the verdict MOVES. At the measured
   alpha it does not move at all -- the compute term never binds -- which is why
   the generator takes an alpha override, and that fact is pinned too.
5. THE BRANCH LABELS ARE WRONG. `classify_branches` decides which cells are the
   flat branch and which the steep one. A greedy longest-prefix version swallowed
   the steep cells; the split search must not, and must return nothing at all
   rather than a branch it cannot find.
6. THE RUN IS LOST OR OVERWRITTEN. A pod is billed by the minute, so the store
   must round-trip a written arm, and the run id must be a function of the WHOLE
   plan: two settings deriving the same id is how this project once printed one
   experiment's numbers under another's heading.
7. A QUANTITY IS DEFAULTED RATHER THAN REFUSED. Every unmeasurable number raises
   a typed refusal. A test that only checked the happy path would let a 0.0 back
   in.

Every timing here is SYNTHETIC. Nothing in this file needs a GPU, vLLM, or a
published CSV, which is the point: the derivation, the reduction, the sign, the
gates and the persistence are all exercisable the day before the pod goes up,
and the only thing the pod adds is the numbers.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench.timing import TIMING_BASIS, KernelTiming  # noqa: E402


def _load_script():
    """Load the script by path. `scripts/` is not a package and never has been.

    Registered in sys.modules BEFORE exec for the same reason
    tests/test_tuned_vs_fallback.py does it: `@dataclass` resolves its
    annotations through `sys.modules[cls.__module__]`, and a module that is not
    there yet fails with an AttributeError about NoneType naming nothing useful.
    """
    spec = importlib.util.spec_from_file_location(
        "dtype_tile_confound", ROOT / "scripts" / "dtype_tile_confound.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DTC = _load_script()

H200 = "NVIDIA H200"


@pytest.fixture(scope="module")
def ceilings():
    """The real committed calibration, not a hand-built one.

    Hand-building would let a test pass against ceilings the script would never
    load, and the whole point of V0 is that both peaks come from ONE file
    measured in ONE session on this machine.
    """
    return DTC.load_ceilings(DTC.DEFAULT_CALIBRATION)


@pytest.fixture
def dtypes():
    return list(DTC.DTYPES)


@pytest.fixture(scope="module")
def planned():
    cells, notes = DTC.plan_cells(list(DTC.DEFAULT_MODELS),
                                 list(DTC.DEFAULT_TOKENS), H200)
    assert cells, "the default plan produced no cells"
    return cells, notes


def synth(cells, ceilings, *, ratio=2.033, alpha=None, act=2, noise=0.0,
          overhead=0.0, dtypes=None):
    return DTC.synthetic_results(
        cells, list(dtypes or DTC.DTYPES), ceilings, fp8_flop_ratio=ratio,
        act_bytes_fp8=act, overhead_ms=overhead, noise=noise, seed=0,
        alpha=alpha)


def verdicts(gates):
    return {g.name.split()[0]: g.verdict for g in gates}


# --------------------------------------------------------------------------
# 1. the sign
# --------------------------------------------------------------------------

def test_tilt_above_one_reads_later_and_below_reads_earlier():
    """The one sentence a reader takes away has to name the direction."""
    assert "LATER" in DTC.shift_sentence(1.15)
    assert "EARLIER" not in DTC.shift_sentence(1.15)
    assert "EARLIER" in DTC.shift_sentence(0.85)
    assert "LATER" not in DTC.shift_sentence(0.85)
    assert "same batch" in DTC.shift_sentence(1.0)


def test_tilt_is_low_third_over_high_third_not_the_reverse():
    """`tilt = r(low) / r(high)`, and getting it upside down inverts the story.

    Planted: fp8 is half the time at small batch and a quarter at large. fp8 then
    pulls AWAY as the batch grows, which makes any threshold crossing arrive
    EARLIER in fp8, so the tilt must be above... no: r falls with T, so
    r_low / r_high = 0.5 / 0.25 = 2.0, and the crossing moves LATER. Written out
    because this is exactly the step that gets reversed.
    """
    series = [(t, 1.0, 0.5) for t in (32, 64, 128)] + \
             [(t, 1.0, 0.25) for t in (1024, 2048, 4096)]
    got = DTC.build_shift("m", "a", series, [])
    assert got.r_low == pytest.approx(0.5)
    assert got.r_high == pytest.approx(0.25)
    assert got.tilt == pytest.approx(2.0)
    assert "LATER" in DTC.shift_sentence(got.tilt)


def test_config_share_is_undefined_rather_than_a_huge_number():
    """No excess to apportion means None, not a share of 4000%."""
    assert DTC.config_share(1.20, 1.05) == pytest.approx(0.75)
    assert DTC.config_share(1.00, 1.00) is None
    assert DTC.config_share(0.81, 0.93) is None


# --------------------------------------------------------------------------
# 2. the two sides being the same kernel
# --------------------------------------------------------------------------

def test_v3_is_unknown_when_nothing_watched_the_override(planned, ceilings):
    """"No mismatch found" and "nothing was looked at" are different states.

    The synthetic rows record no observed config at all, so V3 must decline
    rather than certify. A PASS here would mean a laptop can vouch for a hook it
    never called.
    """
    cells, _ = planned
    analysis = DTC.analyse(cells, synth(cells, ceilings), ceilings,
                           list(DTC.DTYPES))
    assert analysis.override_checked == 0
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "", False)
    assert verdicts(gates)["V3"] == DTC.UNKNOWN


def test_v3_fails_when_a_forced_arm_ran_a_config_it_was_not_given(planned,
                                                                  ceilings):
    cells, _ = planned
    results = synth(cells, ceilings)
    victim = cells[0]
    arm = results[victim.key][("tile_fp8", DTC.FP8)]
    arm.config = dict(victim.configs[DTC.FP8])
    arm.observed_config = dict(victim.configs[DTC.BF16], BLOCK_SIZE_M=999)
    arm.override_verified = False
    other = results[victim.key][("cfg_bf16", DTC.BF16)]
    other.observed_config = dict(victim.configs[DTC.BF16])
    other.override_verified = True
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "", False)
    assert verdicts(gates)["V3"] == DTC.FAIL
    assert analysis.override_failures


def test_v3_fails_when_every_arm_observed_one_single_config(planned, ceilings):
    """One kernel everywhere is what a silently-dead override looks like.

    Zero mismatches and zero failures, and the answer would be a beautifully
    tight 1.000 that means the experiment did not happen. The distinct-config
    count is what catches it.
    """
    cells, _ = planned
    results = synth(cells, ceilings)
    only = dict(cells[0].configs[DTC.BF16])
    for per_cell in results.values():
        for arm in per_cell.values():
            arm.observed_config = dict(only)
            if arm.config is not None:
                arm.config = dict(only)
                arm.override_verified = True
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert analysis.distinct_observed_configs == 1
    assert not analysis.override_failures
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "", False)
    assert verdicts(gates)["V3"] == DTC.FAIL


def test_v4_fails_when_vllm_loaded_a_config_the_derivation_did_not_predict(
        planned, ceilings):
    """If the derivation is wrong the placebo pairs are not placebos."""
    cells, _ = planned
    results = synth(cells, ceilings)
    for arm in results[cells[0].key].values():
        if arm.arm == DTC.NATIVE_ARM:
            arm.observed_config = dict(cells[0].configs[arm.dtype],
                                       GROUP_SIZE_M=7)
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "", False)
    assert verdicts(gates)["V4"] == DTC.FAIL
    assert any("GROUP_SIZE_M" in line for line in analysis.derivation_mismatches)


# --------------------------------------------------------------------------
# 3. the self test certifying itself
# --------------------------------------------------------------------------

def test_synthetic_rows_carry_no_evidence_a_validity_gate_could_read(planned,
                                                                     ceilings):
    """The bug this pins: a laptop certifying a kernel it never ran.

    The first generator filled in `observed_config`, `override_verified` and a
    weight dtype, and V1, V3 and V4 all reported PASS with no GPU in the room.
    """
    cells, _ = planned
    results = synth(cells, ceilings)
    for per_cell in results.values():
        for arm in per_cell.values():
            assert arm.observed_config is None
            assert arm.override_verified is None
            assert arm.weight_torch_dtype == ""
            assert arm.quant_config_kind == ""


def test_self_test_demotes_every_box_validity_gate_to_unknown(planned, ceilings):
    """V0 survives -- it reads a real file -- and V1 to V5 must not."""
    cells, _ = planned
    analysis = DTC.analyse(cells, synth(cells, ceilings), ceilings,
                           list(DTC.DTYPES))
    got = verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                                   synthetic=True))
    assert got["V0"] == DTC.PASS
    for name in ("V1", "V2", "V3", "V4", "V5"):
        assert got[name] == DTC.UNKNOWN, f"{name} certified a synthetic run"
    # The claim gates are the point of the self test and must still be scored.
    assert got["C3"] in (DTC.PASS, DTC.FAIL)


# --------------------------------------------------------------------------
# 4. the gate that cannot fail
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ratio,expected", [(2.033, "PASS"), (2.400, "FAIL"),
                                            (1.000, "FAIL")])
def test_c3_discriminates_between_planted_worlds(planned, ceilings, ratio,
                                                 expected):
    """The whole reason C3 is worth printing: the verdict has to MOVE.

    At an fp8/bf16 FLOP ratio of 2.033 -- what this card measures -- the format
    barely tilts the ratio and C3 passes. At 2.400 the tilt reaches the published
    1.15 territory with no config effect at all, which is the world in which the
    tile was never the explanation. At 1.000 fp8 gets no tensor-core speedup and
    the crossing halves.
    """
    cells, _ = planned
    results = synth(cells, ceilings, ratio=ratio, alpha=0.2)
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "", True)
    assert verdicts(gates)["C3"] == expected


def test_at_the_measured_alpha_the_planted_flop_ratio_moves_one_model_only(
        planned, ceilings):
    """The finding that forced `--self-test-alpha` to exist, pinned as a fact.

    `2 BM / (alpha b)` is 152 FLOP/byte at BLOCK_SIZE_M=128 and the measured
    alpha of 0.84. Where that sits below the ridge the compute term never
    binds, the generated times are pure byte counts, and two very different
    planted FLOP ratios produce numerically IDENTICAL reports: a self test run
    only at the measured alpha would have exercised nothing and C3 would have
    passed in every world. THAT IS WHY THE KNOB EXISTS.

    RE-STATED 2026-09-09 AND THE CALIBRATION IS WHY. The ridge this test used
    to quote was 163; the H200 recalibration committed at ab61e55 puts it at
    152.8, and 152.4 is now BELOW it by 0.3%. So mixtral, whose bf16 tuned
    config takes GROUP_SIZE_M=16 above M=448 where the ceiling rises to 188,
    has cells where the compute term binds and its tilt moves with the planted
    ratio (1.024 at 2.033, 1.071 at 2.400); qwen2, whose high-batch cells take
    GROUP_SIZE_M=1, is still pure bytes and does not move at all. Asserting
    both is what keeps the knob's justification a measurement rather than a
    remembered sentence.
    """
    cells, _ = planned
    a = DTC.analyse(cells, synth(cells, ceilings, ratio=2.033), ceilings,
                    list(DTC.DTYPES))
    b = DTC.analyse(cells, synth(cells, ceilings, ratio=2.400), ceilings,
                    list(DTC.DTYPES))
    moved = {model for (model, _), shift in a.shifts.items()
             if abs(shift.tilt - b.shifts[(model, _)].tilt) > 1e-9}
    assert moved == {"mixtral-8x7b"}, moved
    for (model, arm), shift in a.shifts.items():
        if model == "qwen2-57b-a14b":
            assert shift.tilt == pytest.approx(b.shifts[(model, arm)].tilt), \
                (model, arm)


def test_matched_config_and_quantised_activations_pin_the_tilt_to_a_two_number_band(
        planned, ceilings):
    """The model's whole range, at a matched config with every byte halved.

    With the same config on both sides and the activation stream quantised too,
    the memory branch scales by exactly 0.500 and the compute branch by exactly
    `1 / fp8_over_bf16`, the MEASURED FLOP ratio. So the tilt can only be

        1.000   where the compute term never binds, and the ratio is pure bytes
        high    where it binds at the top of the grid, `0.5 x fp8_over_bf16`

    and nothing in between is reachable by any other route. A mis-thirded grid,
    a ratio taken the wrong way round, or an unmatched cell would put the
    answer outside that band at one end or the other, which is what this pins.

    THE ENDS ARE READ OFF THE CALIBRATION, NOT TYPED. The band used to be
    "1.000 to 1.016" from a 2.033 FLOP ratio and a ridge of 163; the H200
    recalibration at ab61e55 makes the ratio 2.199 and the ridge 152.8, so the
    band is 1.000 to 1.099 and BOTH ends no longer occur on the default grid.
    mixtral's bf16 tuned config takes GROUP_SIZE_M=16 above M=448, where
    `2 BM / (alpha b)` is 188 against the ridge and the compute branch is
    reachable, so it sits exactly at the top; qwen2's high-batch cells take
    GROUP_SIZE_M=1, where the ceiling is 152.4 against a ridge of 152.8, and it
    lands inside at 1.027 rather than on the pure-bytes floor. Which model
    attains which end is a property of vLLM's shipped configs crossed with this
    card's calibration; that the tilt cannot leave the band is a property of
    the reduction, and only the second is asserted as an equality.
    """
    cells, _ = planned
    high = 0.5 * ceilings.fp8_over_bf16
    seen = {}
    for model in DTC.DEFAULT_MODELS:
        got = DTC.predicted_shift(model, cells, ceilings, 1, "cfg_bf16")
        assert got is not None
        assert 1.0 - 1e-9 <= got.tilt <= high + 1e-9, (model, got.tilt)
        seen[model] = got.tilt
    assert seen["mixtral-8x7b"] == pytest.approx(high, abs=1e-6), \
        "the compute branch binds at the top of mixtral's grid, so its tilt IS "\
        "the band's upper end"
    assert 1.0 < seen["qwen2-57b-a14b"] < high, \
        "qwen2 is nearly pure bytes at this ridge and must land strictly inside"


# --------------------------------------------------------------------------
# 5. the branch labels
# --------------------------------------------------------------------------

def test_split_search_does_not_swallow_the_steep_cells():
    """The greedy-prefix bug, pinned.

    Twelve very flat cells pooled with two steep ones still average under the
    threshold, so a longest-flat-prefix rule put the steep cells on the memory
    branch. The split search cannot: adding a steep cell raises the low
    stretch's pooled slope, which is the test.
    """
    flat = [(float(t), 1.0) for t in (32, 64, 128, 256, 384, 512, 768, 1024)]
    steep = [(float(t), t / 1024.0) for t in (2048, 3072, 4096, 6144, 8192)]
    memory, compute = DTC.classify_branches(flat + steep)
    assert memory == [t for t, _ in flat]
    assert compute == [t for t, _ in steep]


def test_a_curve_with_no_flat_stretch_gets_no_branches():
    """Empty is an answer. A branch the data does not show is not invented."""
    linear = [(float(t), float(t)) for t in (32, 64, 128, 256, 512, 1024, 2048)]
    memory, compute = DTC.classify_branches(linear)
    assert memory == []
    assert compute == []


def test_branches_never_share_a_cell():
    """Overlap would divide a branch by itself and report a shift of 1.000."""
    curve = [(float(t), 1.0 + t / 8192.0) for t in
             (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]
    memory, compute = DTC.classify_branches(curve)
    assert not (set(memory) & set(compute))


def test_branch_note_says_which_branch_is_missing():
    """A blank cell in the report has to carry its reason."""
    series = [(t, float(t), float(t) * 0.5) for t in
              (32, 64, 128, 256, 512, 1024, 2048)]
    got = DTC.build_shift("m", "a", series, [])
    assert got.shift is None
    assert "no flat stretch" in got.branch_note


# --------------------------------------------------------------------------
# 6. losing or overwriting the run
# --------------------------------------------------------------------------

def test_store_round_trips_an_arm(tmp_path, planned):
    cells, _ = planned
    cell = cells[0]
    meta = {"run_id": "r", "gpu_name": "g", "torch_version": "t",
            "vllm_version": "v", "routing": "uniform", "seed": 0}
    written = DTC.ArmResult(cell.model, cell.num_tokens, "cfg_bf16", DTC.BF16,
                            config=dict(cell.configs[DTC.BF16]),
                            config_origin="forced",
                            observed_config=dict(cell.configs[DTC.BF16]),
                            override_verified=True,
                            weight_torch_dtype="torch.bfloat16",
                            quant_config_kind="none")
    DTC.summarise_samples(written, [1.0, 1.5, 2.0])
    store = DTC.Store(tmp_path / "timings.csv")
    store.write(written, cell, meta)
    store.close()

    reopened = DTC.Store(tmp_path / "timings.csv")
    back = reopened.restore(written.key)
    reopened.close()
    assert back is not None
    assert back.ms_median == pytest.approx(1.5)
    assert back.n_samples == 3
    assert back.config == written.config
    assert back.override_verified is True
    assert back.observed_config == written.observed_config


#: The plan payload the id is derived from. Written once here so that a knob
#: added to the script and not to the id fails the "every field" test below
#: rather than being quietly untested.
def _plan_payload(**over):
    base = {"models": list(DTC.DEFAULT_MODELS),
            "tokens": list(DTC.DEFAULT_TOKENS), "dtypes": list(DTC.DTYPES),
            "arms": list(DTC.ARMS), "reps": 3, "iters": 15, "warmup": 300.0,
            "budget": 200.0, "trials": 3, "flush": True,
            "seed": 0, "routing": "uniform", "lookup_gpu": H200,
            "calibration": DTC.DEFAULT_CALIBRATION, "vllm_tag": DTC.VLLM_TAG,
            "self_test": "None", "self_test_alpha": "None",
            "self_test_noise": "0.0",
            "self_test_overhead_ms": "0.0", "self_test_fp8_activations": False}
    base.update(over)
    return base


@pytest.mark.parametrize("changed", [
    {"models": ["qwen2-57b-a14b"]},
    {"tokens": [32, 64, 128, 256, 512, 1024]},
    {"dtypes": ["bf16"]},
    {"reps": 5},
    {"iters": 30},
    {"warmup": 16.0},
    {"budget": 400.0},
    {"trials": 5},
    {"flush": False},
    {"seed": 1},
    {"routing": "zipf"},
    {"lookup_gpu": "NVIDIA_A100-SXM4-80GB"},
    {"calibration": "measured_nvidia_a100_sxm4_80gb"},
    {"self_test": "2.4"},
    {"self_test_alpha": "0.2"},
    {"self_test_noise": "0.01"},
    {"self_test_overhead_ms": "0.05"},
    {"self_test_fp8_activations": True},
])
def test_run_id_changes_with_every_swept_parameter(changed):
    """A run id that omits a swept knob silently resumes another experiment.

    The sibling sweep shipped one that omitted GROUP_SIZE_M: the second setting
    derived the same id, resumed the first's directory, skipped every completed
    cell and printed the first's timings under the second's heading. Nothing
    looked wrong. So every key in the plan payload gets its own case here.

    THE FOUR TIMING KNOBS ARE NEW HERE (2026-09-02) and they are the ones that
    would have cost most: `warmup`, `budget`, `trials` and `flush` set the
    measured milliseconds of every row, so a re-run at a different one landing
    in the same directory prints the old numbers under the new label.
    """
    base = _plan_payload()
    other = dict(base, **changed)
    assert set(other) == set(base), f"{changed} is not in the plan payload"
    assert (DTC.plan_run_id(base, H200)
            != DTC.plan_run_id(other, H200))


def test_the_run_id_carries_the_card_and_two_cards_cannot_share_a_directory():
    """The omission that cost a published arm, and it was not this script's.

    `results/published/2026-09-01-nvidia_h200-cross-card-s3` and
    `2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3` both contain
    `mixtral-8x7b-bf16-r1024-g1-n64-4867a2.report.json`: one id, two cards, two
    different `sm_count`s and two different ridges. Nothing downstream noticed,
    because the resume key carries no device. This script resumes on
    (model, tokens, arm, dtype) and had the same hole.
    """
    payload = _plan_payload()
    h200 = DTC.plan_run_id(payload, H200)
    a100 = DTC.plan_run_id(payload, "NVIDIA A100-SXM4-80GB")
    assert h200 != a100
    assert h200.startswith("nvidia_h200-")
    assert a100.startswith("nvidia_a100_sxm4_80gb-")


def test_the_run_id_refuses_a_plan_knob_it_does_not_carry():
    """The FAIL branch of the guard, planted.

    A knob added to the plan and forgotten in the id is the whole failure mode,
    and a guard nobody has seen refuse is a guard nobody has tested.
    """
    with pytest.raises(ValueError) as excinfo:
        DTC.plan_run_id(_plan_payload(some_new_knob=7), H200)
    assert "some_new_knob" in str(excinfo.value)


def test_the_run_id_refuses_a_run_with_no_card():
    """No card, no id. A directory named without one collides across pods."""
    with pytest.raises(PV.NoCard):
        DTC.plan_run_id(_plan_payload(), "")


def test_run_id_is_stable_for_the_same_plan():
    """Resume is the default, so the same command must derive the same id."""
    base = _plan_payload(models=["mixtral-8x7b"], tokens=[32, 64],
                         dtypes=["bf16"], arms=["native"], reps=1)
    assert DTC.plan_run_id(base, H200) == DTC.plan_run_id(dict(base), H200)


def test_every_path_the_script_writes_is_checked_against_gitignore(tmp_path):
    """`results/*` is ignored with only `!results/published/` excepted.

    The script prints the verdict rather than reasoning about the patterns, and
    the wording has to distinguish the two states or the line is decoration.
    """
    inside = ROOT / "results" / "dtype_tile_confound" / "x"
    assert "GIT-IGNORED" in DTC.gitignore_note(inside)
    kept = ROOT / "results" / "published" / "some-arm"
    assert "NOT ignored" in DTC.gitignore_note(kept)
    assert "outside the repo" in DTC.gitignore_note(tmp_path)


# --------------------------------------------------------------------------
# 7. refusing rather than defaulting
# --------------------------------------------------------------------------

def test_a_bf16_only_run_refuses_every_dtype_number(planned, ceilings):
    """The largest defensible subset must not quietly become the whole claim.

    `--dtypes bf16` is a real mode -- it prices the config effect at fixed dtype
    on a card with no fp8 units -- and it must produce refusals and UNKNOWN
    verdicts, never a tilt computed from one side.
    """
    cells, _ = planned
    results = synth(cells, ceilings, dtypes=[DTC.BF16])
    analysis = DTC.analyse(cells, results, ceilings, [DTC.BF16])
    assert analysis.shifts == {}
    assert analysis.refusals
    assert all("UnpairableComparison" in line for line in analysis.refusals)
    got = verdicts(DTC.build_gates(analysis, ceilings, [DTC.BF16], "", True))
    assert got["C3"] == DTC.UNKNOWN
    assert got["C4"] == DTC.UNKNOWN


def test_a_thin_grid_refuses_a_tilt_rather_than_shrinking_it():
    with pytest.raises(DTC.RegimeNotResolved) as excinfo:
        DTC.build_shift("m", "a", [(32, 1.0, 0.5), (64, 1.0, 0.5)], [128])
    assert "paired cells" in str(excinfo.value)
    # The unpaired token count is named, not swallowed.
    assert "128" in str(excinfo.value)


def test_a_calibration_without_an_fp8_peak_refuses_rather_than_halving_bf16():
    """FINDINGS flags exactly this on the published fp8 arm.

    Its calibration measured no fp8 ceiling, `achieved_peak_tflops` came back
    0.0, and the headline had to be reconstructed by hand. Substituting
    `2 x bf16` from the datasheet here would look entirely plausible and would be
    the same defect.
    """
    partial = DTC.Ceilings(name="n", bandwidth_bytes_s=4.0e12,
                           peak_flops={DTC.BF16: 7.0e14}, source="s",
                           measured_on="d", path="p")
    assert partial.peak(DTC.BF16) == pytest.approx(7.0e14)
    with pytest.raises(DTC.CalibrationIncomplete):
        partial.peak(DTC.FP8)
    with pytest.raises(DTC.CalibrationIncomplete):
        partial.ridge(DTC.FP8)


def test_v0_fails_on_a_calibration_missing_a_peak(planned, ceilings):
    cells, _ = planned
    partial = DTC.Ceilings(name=ceilings.name,
                           bandwidth_bytes_s=ceilings.bandwidth_bytes_s,
                           peak_flops={DTC.BF16: ceilings.peak(DTC.BF16)},
                           source=ceilings.source,
                           measured_on=ceilings.measured_on, path=ceilings.path)
    analysis = DTC.analyse(cells, synth(cells, ceilings), ceilings,
                           list(DTC.DTYPES))
    got = verdicts(DTC.build_gates(analysis, partial, list(DTC.DTYPES), "", True))
    assert got["V0"] == DTC.FAIL


def test_an_unpaired_cell_is_named_and_excluded_never_averaged(planned,
                                                               ceilings):
    """Half a pair is not a ratio. It has to leave the median and be printed."""
    cells, _ = planned
    results = synth(cells, ceilings)
    victim = cells[3]
    results[victim.key][(DTC.NATIVE_ARM, DTC.FP8)].error = "planted compile fail"
    results[victim.key][(DTC.NATIVE_ARM, DTC.FP8)].ms_median = None
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    got = analysis.shift_of(victim.model, DTC.NATIVE_ARM)
    assert got is not None
    assert victim.num_tokens in got.unpaired
    assert victim.num_tokens not in [t for t, _, _ in got.points]


# --------------------------------------------------------------------------
# non-vacuity: a check that examined nothing reports zero failures too
# --------------------------------------------------------------------------

def test_an_empty_run_reports_no_timings_rather_than_a_clean_page(planned,
                                                                  ceilings):
    cells, _ = planned
    analysis = DTC.analyse(cells, {}, ceilings, list(DTC.DTYPES))
    assert analysis.timed_arms == 0
    assert analysis.shifts == {}
    assert analysis.refusals
    got = verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "", False))
    assert got["V1"] == DTC.UNKNOWN
    assert got["V3"] == DTC.UNKNOWN
    assert got["C3"] == DTC.UNKNOWN


def test_a_model_whose_every_arm_refused_still_appears_in_the_report(planned,
                                                                     ceilings):
    """Otherwise the page silently narrows to whatever happened to work."""
    cells, _ = planned
    analysis = DTC.analyse(cells, {}, ceilings, list(DTC.DTYPES))
    assert set(analysis.models) == set(DTC.DEFAULT_MODELS)
    text = DTC.render_shifts(analysis)
    for model in DTC.DEFAULT_MODELS:
        assert model in text
    assert "REFUSED" in text


def test_the_generator_actually_generated_something(planned, ceilings):
    """A self test that produced no cells would pass every claim gate too."""
    cells, _ = planned
    results = synth(cells, ceilings)
    timed = sum(1 for per in results.values() for arm in per.values()
                if arm.ms_median)
    assert timed >= len(cells) * len(DTC.DTYPES)


# --------------------------------------------------------------------------
# the derivation, which decides C1 and C2 off GPU
# --------------------------------------------------------------------------

def test_the_two_dtypes_resolve_different_configs_at_every_bracketing_cell(
        planned):
    """C1's premise. No config difference means nothing to separate."""
    cells, _ = planned
    bracket = DTC.crossing_bracket_cells(cells)
    assert bracket, "no cell brackets a published crossing"
    assert all(cell.configs_differ for cell in bracket)


def test_block_size_m_agrees_across_dtypes_where_the_crossing_lives(planned):
    """C2, and it is a CORRECTION to docs/FINDINGS.md.

    FINDINGS states the confound as the fp8 arm running taller M tiles
    "throughout". Derived from vLLM v0.27.1's own shipped tree, BLOCK_SIZE_M
    differs only below M=96 on mixtral and below M=192 on qwen2, and every
    published crossing -- 454 and 810 bf16, 568 and 900 fp8 -- sits above those.
    """
    cells, _ = planned
    bracket = DTC.crossing_bracket_cells(cells)
    agree = [cell for cell in bracket if cell.block_m_agrees]
    assert len(agree) / len(bracket) >= DTC.BLOCK_M_AGREEMENT_MIN


def test_the_surviving_confound_runs_through_block_k_and_the_swizzle(planned):
    """Which knob C4 is actually about, pinned so a doc edit cannot drift.

    BLOCK_SIZE_K differs at every bracketing cell -- `get_default_config` and the
    tuned files both take `dtype_selector == "fp8_w8a8"` to 128 where bf16 takes
    64 -- and GROUP_SIZE_M differs at some. Those are the knobs left once
    BLOCK_SIZE_M agrees.
    """
    cells, _ = planned
    bracket = DTC.crossing_bracket_cells(cells)
    assert all("BLOCK_SIZE_K" in cell.differing_keys for cell in bracket)
    assert any("GROUP_SIZE_M" in cell.differing_keys for cell in bracket)


def test_the_byte_model_is_blind_to_block_k_so_c4_cannot_be_derived(planned,
                                                                    ceilings):
    """Stated as a test so the limitation cannot quietly stop being true.

    `predicted_ms` reads BLOCK_SIZE_M and GROUP_SIZE_M and nothing else, so a
    config that differs only in BLOCK_SIZE_K is invisible to it. That is why C4
    carries no predicted value and has to be measured.
    """
    cfg = DTC.MODEL_CONFIGS["mixtral-8x7b"]
    base = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4}
    other = dict(base, BLOCK_SIZE_K=128, num_stages=3, num_warps=4)
    kw = {"ceilings": ceilings, "act_bytes": 2}
    assert DTC.predicted_ms(cfg, 1024, base, DTC.BF16, **kw) == pytest.approx(
        DTC.predicted_ms(cfg, 1024, other, DTC.BF16, **kw))


def test_every_planned_token_count_is_realisable_as_a_balanced_histogram(
        planned):
    """`T k / E` has to be an integer or `realize_counts` refuses mid-run."""
    cells, _ = planned
    for cell in cells:
        cfg = DTC.MODEL_CONFIGS[cell.model]
        assert cell.num_tokens % DTC.rows_step(cfg) == 0
        assert cell.rows == pytest.approx(round(cell.rows))


def test_a_model_with_an_unrealisable_token_count_is_dropped_with_a_reason():
    cells, notes = DTC.plan_cells(["deepseek-v3"], [32, 48, 64], H200)
    assert {c.num_tokens for c in cells} == {32, 64}
    assert any("48" in note for note in notes)


# --------------------------------------------------------------------------
# the measured alpha curve
# --------------------------------------------------------------------------

def test_alpha_curve_reproduces_every_measured_point_exactly():
    for group_m, alpha in DTC.ALPHA_BY_GROUP_M.items():
        assert DTC.alpha_for_group(group_m) == pytest.approx(alpha)


def test_alpha_curve_is_monotone_and_clamped_outside_the_measured_range():
    values = [DTC.alpha_for_group(g) for g in (1, 2, 4, 8, 16, 32, 64)]
    assert values == sorted(values, reverse=True)
    assert DTC.alpha_for_group(128) == pytest.approx(DTC.ALPHA_BY_GROUP_M[64])
    with pytest.raises(ValueError):
        DTC.alpha_for_group(0)


def test_the_measured_alpha_caps_block_m_128_below_the_h200_ridge(ceilings):
    """The fact that decided this script's estimand.

    `2 BM / (alpha b)` at BLOCK_SIZE_M=128, GROUP_SIZE_M=1 and bf16 is 152.4
    against a measured ridge of 162.8, so the config vLLM's ladder actually picks
    across the decode range cannot be compute bound at any batch. A flat-versus-
    steep branch test therefore cannot find a roofline crossing on it, which is
    why the TILT and not the branch shift is what the gates read.
    """
    alpha = DTC.alpha_for_group(1)
    cap = 2 * 128 / (alpha * 2)
    assert cap < ceilings.ridge(DTC.BF16)


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

def test_a_redundant_arm_is_recorded_and_resolved_to_its_twin(planned,
                                                               ceilings):
    """Not timed, not missing: the same kernel, read from the arm that ran it."""
    cfg = DTC.MODEL_CONFIGS["mixtral-8x7b"]
    same = DTC.Cell(model="mixtral-8x7b", num_tokens=128, lookup_gpu=H200,
                    tiles={}, configs={d: {"BLOCK_SIZE_M": 64,
                                           "BLOCK_SIZE_N": 128,
                                           "BLOCK_SIZE_K": 128,
                                           "GROUP_SIZE_M": 1, "num_warps": 4,
                                           "num_stages": 3} for d in DTC.DTYPES})
    assert not same.configs_differ
    assert DTC.arm_is_redundant(same, "tile_fp8")
    assert not DTC.arm_is_redundant(same, "cfg_bf16")
    twin = DTC.ArmResult(same.model, same.num_tokens, "cfg_bf16", DTC.BF16)
    DTC.summarise_samples(twin, [2.0])
    redundant = DTC.ArmResult(same.model, same.num_tokens, "tile_fp8", DTC.BF16,
                              redundant=True)
    per_cell = {("cfg_bf16", DTC.BF16): twin, ("tile_fp8", DTC.BF16): redundant}
    assert DTC.cell_ms(per_cell, "tile_fp8", DTC.BF16, same) == pytest.approx(2.0)
    assert cfg.num_experts == 8          # the fixture is the model it claims


def test_arm_config_forces_a_full_config_never_a_subset(planned):
    """Reporting BLOCK_SIZE_M alone is what made this confound invisible.

    Both arms force a COMPLETE config; they differ in where its keys come
    from, which `MATCHED_ARM_SCOPE` names. `tile_fp8` takes two of them from
    the fp8 lookup and the other four from the width it is running at, so at
    fp8 it is the identity and is still the placebo.
    """
    cells, _ = planned
    cell = cells[0]
    for dtype in DTC.DTYPES:
        assert DTC.arm_config(cell, DTC.NATIVE_ARM, dtype) is None
        for arm in DTC.MATCHED_ARMS:
            forced = DTC.arm_config(cell, arm, dtype)
            assert set(forced) == set(DTC.CONFIG_KEYS)
    assert DTC.arm_config(cell, "cfg_bf16", DTC.FP8) == cell.configs[DTC.BF16]
    assert DTC.arm_config(cell, "cfg_bf16", DTC.BF16) == cell.configs[DTC.BF16]
    assert DTC.arm_config(cell, "tile_fp8", DTC.FP8) == cell.configs[DTC.FP8]
    at_bf16 = DTC.arm_config(cell, "tile_fp8", DTC.BF16)
    for key in DTC.CONFIG_KEYS:
        source = DTC.FP8 if key in DTC.MATCHED_ARM_SCOPE["tile_fp8"] else DTC.BF16
        assert at_bf16[key] == cell.configs[source][key], key
    with pytest.raises(KeyError):
        DTC.arm_config(cell, "nonesuch", DTC.BF16)


def test_format_config_prints_every_knob():
    text = DTC.format_config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256,
                              "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16,
                              "num_warps": 8, "num_stages": 4})
    for token in ("128", "256", "64", "16", "w8", "s4"):
        assert token in text


# --------------------------------------------------------------------------
# the two entry points a reviewer runs
# --------------------------------------------------------------------------

def test_dry_run_prints_the_plan_the_predictions_and_the_cost(tmp_path, capsys):
    """The reviewable artefact: it must need no GPU and must measure nothing."""
    code = DTC.main(["--dry-run", "--card", H200, "--out-dir", str(tmp_path)])
    assert code == DTC.EXIT_NOT_MEASURED
    assert code == exit_codes.REFUSED, (
        "a dry run spent nothing; REFUSED is the code for that, and the 3 this "
        "script used to return is INVALID, which the driver reads as 'measured, "
        "and its cells must not be quoted'")
    out = capsys.readouterr().out
    assert "Predictions, registered before the run" in out
    assert "MDE" in out, "no arm may plan without stating what it could see"
    assert "COST" in out
    assert "distinct Triton specialisations" in out
    assert "NOT A RESULT" in out
    assert "measured_nvidia_h200.yaml" in out
    plans = list(tmp_path.glob("*/plan.json"))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text())
    assert plan["cells"]
    assert plan["ceilings"]["peak_flops"][DTC.FP8] > 0
    assert plan["estimated_timed_seconds"] > 0


def test_predictions_are_printed_before_any_measurement(tmp_path, capsys):
    """Registered BEFORE, not beside. A prediction after the data is worthless."""
    DTC.main(["--self-test", "2.033", "--self-test-alpha", "0.2",
              "--card", H200, "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.index("Predictions, registered before the run") < out.index(
        "The measurement: the fp8/bf16 ratio")


def test_a_synthetic_run_exits_invalid_and_still_says_which_claim_failed(
        tmp_path, capsys):
    """A page built from generated rows may not be quoted, and says so in ONE
    number.

    THE CODE CHANGED ON 2026-09-02 and this is why. `--self-test` demotes every
    box VALIDITY gate to UNKNOWN by construction -- no kernel ran, so there is
    no weight dtype and no oracle -- and under the shared table an UNKNOWN
    VALIDITY gate is INVALID: the instrument's soundness was not shown, so
    nothing on the page is quotable. That is exactly true of a synthetic run,
    and returning 0 for it let a generated report be logged as a measurement.

    The self test keeps its discriminating power through the RESULT lines,
    which is what the second half asserts: two planted worlds give the same
    INVALID page and opposite C3 verdicts on it.
    """
    code = DTC.main(["--self-test", "2.400", "--self-test-alpha", "0.2",
                     "--card", H200, "--out-dir", str(tmp_path)])
    assert code == DTC.EXIT_INVALID
    refuted = capsys.readouterr().out
    assert "claims   " in refuted
    # The FAIL branch: a planted fp8 peak the model cannot explain tilts the
    # matched arms out of C3's window.
    assert _result(refuted, "C3") == DTC.FAIL
    # The PASS branch of the same gate, same code for the page.
    code = DTC.main(["--self-test", "2.033", "--self-test-alpha", "0.2",
                     "--card", H200, "--out-dir", str(tmp_path)])
    assert code == DTC.EXIT_INVALID
    assert _result(capsys.readouterr().out, "C3") == DTC.PASS


def _result(text: str, name: str) -> str:
    """The verdict of one gate, read back out of the log's RESULT lines.

    Through `parse_result_lines` rather than a regex written here, so a test
    cannot pass against a line shape the driver would not match.
    """
    by_name = {r.name: r.verdict for r in exit_codes.parse_result_lines(text)}
    assert name in by_name, f"no RESULT line for {name}: {sorted(by_name)}"
    return by_name[name]


def test_the_log_and_the_exit_code_cannot_disagree(tmp_path, capsys):
    """`classify_text` over what was printed must recompute what was returned.

    The defect the shared table is named against is a script printing one thing
    and exiting another; the check is free, so it is a test rather than a
    convention.
    """
    code = DTC.main(["--self-test", "2.400", "--self-test-alpha", "0.2",
                     "--card", H200, "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert exit_codes.classify_text(out) == code


def test_a_claim_failure_is_returned_and_not_folded_into_done():
    """The branch the old code took and no test could reach.

    Until 2026-09-02 `main` computed `rc = classify(...)` and then returned DONE
    when rc was CLAIM_FAIL and `--fail-on-claim` was absent, so the log printed
    `RESULT: CLAIM C3 FAIL` and the process said 0 -- the one disagreement
    `exit_codes` exists to detect, under a comment claiming it could not happen.
    Nothing exercised it: every `--self-test` demotes the box VALIDITY gates to
    UNKNOWN and therefore classifies INVALID, so the only page that could make
    the log and the process disagree was the one page nothing produced. Planted
    here through `final_exit`, which is now the single place `main` ends.

    All three codes, and `classify_text` over the same gates' RESULT lines, so
    the property is checked in both directions rather than asserted.
    """
    def gate(name, kind, verdict):
        return DTC.Gate(name=name, kind=kind, prediction="p", rule="r",
                        verdict=verdict, observed="saw")

    def check(gates, want):
        assert DTC.final_exit(gates) == want
        log = "prose that mentions PASS\n" + "\n".join(
            g.result_line() for g in gates)
        assert exit_codes.classify_text(log) == want

    check([gate("V0 ceilings", "VALIDITY", DTC.PASS),
           gate("C3 tilt", "CLAIM", DTC.PASS)], exit_codes.DONE)
    # THE PLANTED DISAGREEMENT: validity holds, a claim does not.
    check([gate("V0 ceilings", "VALIDITY", DTC.PASS),
           gate("C3 tilt", "CLAIM", DTC.FAIL)], exit_codes.CLAIM_FAIL)
    # And UNKNOWN on a claim is not a pass either.
    check([gate("V0 ceilings", "VALIDITY", DTC.PASS),
           gate("C3 tilt", "CLAIM", DTC.UNKNOWN)], exit_codes.CLAIM_FAIL)
    # A validity failure outranks a passing claim.
    check([gate("V0 ceilings", "VALIDITY", DTC.FAIL),
           gate("C3 tilt", "CLAIM", DTC.PASS)], exit_codes.INVALID)
    # A page with no gate on it has no verdict to exit with.
    with pytest.raises(exit_codes.NoGatesScored):
        DTC.final_exit([])


def test_the_retired_fail_on_claim_flag_still_parses_and_changes_nothing():
    """Driver lines in `scripts/h200_gaps_session.sh` still pass it. It has to
    be accepted, and it must not resurrect the masking."""
    assert DTC.build_parser().parse_args(["--fail-on-claim"]).fail_on_claim
    source = (ROOT / "scripts" / "dtype_tile_confound.py").read_text()
    assert "if rc == exit_codes.CLAIM_FAIL" not in source
    assert "return exit_codes.DONE" not in source


def test_a_dry_run_prints_no_result_line_at_all(tmp_path, capsys):
    """A REFUSED log must carry none, or the driver recomputes DONE from it.

    The dry run DECIDES C1 and C2 off GPU and prints both verdicts, which is
    the point of the free half. If it also printed their RESULT lines,
    `classify_text` would read two passing claims, return DONE, and disagree
    with the REFUSED the process returns.
    """
    code = DTC.main(["--dry-run", "--card", H200, "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == exit_codes.REFUSED
    assert exit_codes.parse_result_lines(out) == []
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(out)


def test_self_test_writes_a_report_and_a_summary_that_carry_the_refusals(
        tmp_path):
    DTC.main(["--self-test", "2.033", "--self-test-alpha", "0.2",
              "--card", H200, "--out-dir", str(tmp_path)])
    summaries = list(tmp_path.glob("*/summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text())
    assert payload["self_test_fp8_ratio"] == pytest.approx(2.033)
    assert payload["shifts"]
    assert "sign" in payload and "LATER" in payload["sign"]
    assert payload["published_confounded_shift"] == pytest.approx(
        DTC.PUBLISHED_SHIFT)
    reports = list(tmp_path.glob("*/report.md"))
    assert len(reports) == 1
    text = reports[0].read_text()
    assert "SELF TEST, synthetic rows" in text
    assert "Decomposition" in text


def test_e5m2_is_refused_rather_than_mapped_onto_e4m3(tmp_path, capsys):
    """REFUSED (2), and it RETURNS rather than raising.

    Both of these were `raise SystemExit(<str>)`, which sets `SystemExit.code`
    to the STRING and leaves the interpreter to exit 1 -- CLAIM_FAIL, "measured;
    a pre-registered claim was refuted" -- from a process that had done nothing
    but parse its arguments. `pytest.raises(SystemExit)` passed on that happily
    and said nothing about the integer, which is the only thing the driver sees.
    """
    for argv in (["--dtypes", "bf16,fp8_e5m2"], ["--dtypes", "fp8_e4m3"]):
        code = DTC.main([*argv, "--dry-run", "--card", H200,
                         "--out-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert code == exit_codes.REFUSED
        assert "REFUSED" in out
        assert exit_codes.parse_result_lines(out) == []


# --------------------------------------------------------------------------
# the noise floor, and the two ways it stops being readable
# --------------------------------------------------------------------------

def test_clock_drift_is_per_cell_so_a_resumed_csv_cannot_invent_one(planned,
                                                                    ceilings):
    """Two sessions in one CSV must not be differenced against each other.

    First-start against last-end would read the opening clock of one session
    against the closing clock of another and report a drift about nothing. Each
    cell's own bracket is the window its ratios were measured inside.
    """
    cells, _ = planned
    results = synth(cells, ceilings)
    keys = list(results)
    for arm in results[keys[0]].values():         # an old session, high clocks
        arm.idle_clock_before_mhz, arm.idle_clock_after_mhz = 1980, 1975
    for arm in results[keys[-1]].values():        # a later one, lower clocks
        arm.idle_clock_before_mhz, arm.idle_clock_after_mhz = 1500, 1495
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert abs(analysis.clock_drift_pct) < 1.0
    assert not analysis.clock_throttled


def _flag_every_arm(results, *, level=True, drift=True):
    """Give every synthetic arm a determined under-load clock verdict."""
    for per_cell in results.values():
        for arm in per_cell.values():
            arm.clock_level_ok, arm.clock_drift_ok = level, drift


def test_v5_fails_on_a_drifting_clock_and_not_on_a_level_flag(planned, ceilings):
    """The FAIL branch, planted through the flag `time_kernel` actually sets.

    The first version of this test planted a 1980 -> 1400 MHz drop between
    the two BETWEEN-LOAD samples and V5 failed on that, which is the idle-boost
    catch retraction (f) withdrew, carried into this script under another name.
    The second scored the under-load LEVEL flag. Since 2026-09-09 only DRIFT
    excludes: LEVEL is recorded on both sides and excludes on neither, because
    on this card the under-load clock is set per tile by the kernel's own power
    draw and a LOW side is a steady operating point rather than a throttle.
    """
    cells, _ = planned
    results = synth(cells, ceilings)
    _flag_every_arm(results)
    for arm in results[list(results)[2]].values():
        arm.clock_level_ok, arm.clock_level_side = False, DTC.timing.LEVEL_LOW
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert not analysis.clock_throttled, "a steady LOW clock is not a throttle"
    assert analysis.clock_flagged_arms == 0
    assert analysis.clock_low_side_arms == len(results[list(results)[2]])
    got = verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                                   synthetic=False))
    assert got["V5"] == DTC.PASS

    for arm in results[list(results)[2]].values():
        arm.clock_drift_ok = False
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert analysis.clock_throttled
    assert analysis.clock_flagged_arms > 0
    got = verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                                   synthetic=False))
    assert got["V5"] == DTC.FAIL


def test_v5_fails_on_an_under_load_drift_flag_too(planned, ceilings):
    """Both flags are read: a DRIFT False alone is enough to fail."""
    cells, _ = planned
    results = synth(cells, ceilings)
    _flag_every_arm(results)
    next(iter(results[list(results)[0]].values())).clock_drift_ok = False
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert analysis.clock_throttled and analysis.clock_flagged_arms == 1
    got = verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                                   synthetic=False))
    assert got["V5"] == DTC.FAIL


def test_v5_passes_when_every_arm_carries_a_good_under_load_verdict(planned,
                                                                     ceilings):
    """The PASS branch needs the clock clause DETERMINED, not merely unflagged."""
    cells, _ = planned
    results = synth(cells, ceilings)
    _flag_every_arm(results)
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert not analysis.clock_throttled
    assert analysis.clock_determined_arms == analysis.timed_arms
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                            synthetic=False)
    v5 = next(g for g in gates if g.name.startswith("V5"))
    assert v5.verdict == DTC.PASS
    assert "0 of" in v5.observed and "determined arms DRIFTED" in v5.observed


def test_between_load_clock_movement_is_printed_as_context_and_never_scored(
        planned, ceilings):
    """A 1980 -> 1400 MHz drop between the two ClockState samples is what the
    old flag scored. Under (f) it is context: printed, labelled, and unable to
    fail V5 on its own, because a between-load sample can be an idle boost."""
    cells, _ = planned
    results = synth(cells, ceilings)
    _flag_every_arm(results)
    for arm in results[list(results)[2]].values():
        arm.idle_clock_before_mhz, arm.idle_clock_after_mhz = 1980, 1400
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert analysis.clock_drift_pct == pytest.approx(29.3, abs=0.1)
    assert not analysis.clock_throttled
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                            synthetic=False)
    v5 = next(g for g in gates if g.name.startswith("V5"))
    assert v5.verdict == DTC.PASS
    assert "between-load clock movement +29.3%" in v5.observed
    assert "context only" in v5.observed
    assert "clock drift" not in v5.rule


def test_v5_is_unknown_not_pass_when_no_arm_determined_its_clock(planned,
                                                                  ceilings):
    """Absent NVML is not a steady clock. With no LEVEL or DRIFT verdict on
    any arm and nothing else failing, V5 reads UNKNOWN; a determined failure
    elsewhere in the gate still reads FAIL."""
    cells, _ = planned
    results = synth(cells, ceilings)
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert analysis.clock_determined_arms == 0
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                            synthetic=False)
    v5 = next(g for g in gates if g.name.startswith("V5"))
    assert v5.verdict == DTC.UNKNOWN
    assert "NOT DETERMINED on any arm" in v5.observed


# --------------------------------------------------------------------------
# the cost estimate prices what time_kernel spends (retraction g)
# --------------------------------------------------------------------------

def test_estimated_seconds_prices_warmup_ms_plus_trials_times_budget(planned,
                                                                      ceilings):
    """`--warmup` is MILLISECONDS of sustained load and `time_kernel` sizes
    iters from `--cell-budget-ms`, so a repeat costs warmup + trials x budget.
    The old estimate billed `reps * (warmup + iters) + warmup + 1` CALLS at a
    per-call time with warmup = 300, i.e. 1,246 calls per cell."""
    cells, _ = planned
    dtypes = list(DTC.DTYPES)
    n_arms = sum(1 for c in cells for a in DTC.ARMS
                 if not DTC.arm_is_redundant(c, a)) * len(dtypes)
    seconds = DTC.estimated_seconds(cells, dtypes, ceilings, 3, 3, 300.0, 200.0)
    assert seconds == pytest.approx(n_arms * 3 * (300.0 + 3 * 200.0) / 1e3)
    # Not a function of the retired iteration count or of the per-call time
    # while a call fits inside the budget; doubling the budget doubles the
    # trial term and nothing else.
    doubled = DTC.estimated_seconds(cells, dtypes, ceilings, 3, 3, 300.0, 400.0)
    assert doubled == pytest.approx(n_arms * 3 * (300.0 + 3 * 400.0) / 1e3)


def test_estimated_seconds_bills_a_call_longer_than_the_budget_at_its_length(
        planned, ceilings):
    """A trial is at least one call. With a 1 ms budget every call is longer
    than it, so the trial term is the model's own per-call time, not 1 ms."""
    cells, _ = planned
    dtypes = list(DTC.DTYPES)
    tiny = DTC.estimated_seconds(cells, dtypes, ceilings, 1, 1, 1.0, 0.001)
    n_arms = sum(1 for c in cells for a in DTC.ARMS
                 if not DTC.arm_is_redundant(c, a)) * len(dtypes)
    floor = n_arms * (1.0 + 0.001) / 1e3
    assert tiny > floor


@pytest.mark.parametrize("bad", [
    dict(reps=0), dict(trials=0), dict(warmup_ms=0.0), dict(budget_ms=0.0),
])
def test_estimated_seconds_refuses_a_plan_time_kernel_would_refuse(planned,
                                                                     ceilings,
                                                                     bad):
    cells, _ = planned
    kw = dict(reps=3, trials=3, warmup_ms=300.0, budget_ms=200.0)
    kw.update(bad)
    with pytest.raises(DTC.ConfoundRefusal):
        DTC.estimated_seconds(cells, list(DTC.DTYPES), ceilings, **kw)


def test_v5_fails_on_a_wide_placebo_band(planned, ceilings):
    """A placebo is the SAME config timed twice. Wide means the box cannot see
    the ~5% effect the model predicts, whatever the medians come out at."""
    cells, _ = planned
    results = synth(cells, ceilings)
    for per_cell in results.values():
        arm = per_cell[("cfg_bf16", DTC.BF16)]
        if arm.ms_median:
            DTC.summarise_samples(arm, [arm.ms_median * 1.30])
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    got = verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                                   synthetic=False))
    assert got["V5"] == DTC.FAIL
    assert "cfg_bf16/native" in analysis.placebo_worst


def test_v2_fails_when_a_dtype_misses_the_oracle(planned, ceilings):
    cells, _ = planned
    results = synth(cells, ceilings)
    arm = results[cells[0].key][(DTC.NATIVE_ARM, DTC.FP8)]
    arm.correctness_rel_err = 0.9
    arm.correctness_budget = 0.5
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    got = verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                                   synthetic=False))
    assert got["V2"] == DTC.FAIL


def test_v1_fails_when_the_fp8_arm_was_handed_bf16_weights(planned, ceilings):
    """The silent substitution moe/quant.py exists to refuse.

    vLLM would accept an fp8 cell on silicon without the units, dequantise, and
    write rows labelled fp8_e4m3 that never touched an fp8 unit.
    """
    cells, _ = planned
    results = synth(cells, ceilings)
    for per_cell in results.values():
        for (_arm, dtype), result in per_cell.items():
            if dtype == DTC.FP8:
                result.weight_torch_dtype = "torch.bfloat16"
                result.quant_config_kind = "none"
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    got = verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                                   synthetic=False))
    assert got["V1"] == DTC.FAIL


def test_v1_passes_only_with_a_float8_dtype_and_a_quant_config(planned,
                                                               ceilings):
    cells, _ = planned
    results = synth(cells, ceilings)
    for per_cell in results.values():
        for (_arm, dtype), result in per_cell.items():
            result.weight_torch_dtype = ("torch.float8_e4m3fn"
                                         if dtype == DTC.FP8
                                         else "torch.bfloat16")
            result.quant_config_kind = ("FusedMoEQuantConfig"
                                        if dtype == DTC.FP8 else "none")
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    got = verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                                   synthetic=False))
    assert got["V1"] == DTC.PASS


# --------------------------------------------------------------------------
# the fp8 preflight, which is the refusal the brief asked for by name
# --------------------------------------------------------------------------

def test_preflight_refuses_fp8_on_a_machine_with_no_fp8_silicon():
    """On this laptop there is no CUDA device at all, which is the None branch
    of `fp8_hardware_support` and must raise rather than degrade to bf16."""
    with pytest.raises(DTC.Fp8PathUnavailable) as excinfo:
        DTC.preflight_fp8(list(DTC.DTYPES))
    assert "fp8" in str(excinfo.value)


def test_preflight_is_a_no_op_when_fp8_was_not_asked_for():
    note = DTC.preflight_fp8([DTC.BF16])
    assert "nothing to preflight" in note


def test_every_refusal_is_typed_and_shares_one_base():
    for name in ("Fp8PathUnavailable", "CalibrationIncomplete",
                 "RegimeNotResolved", "UnpairableComparison"):
        cls = getattr(DTC, name)
        assert issubclass(cls, DTC.ConfoundRefusal)
        assert issubclass(cls, RuntimeError)


def test_a_bf16_only_run_completes_end_to_end_and_says_what_it_refused(
        tmp_path, capsys):
    """The largest defensible subset has to be a real, runnable mode."""
    code = DTC.main(["--dtypes", "bf16", "--self-test", "2.033",
                     "--card", H200, "--out-dir", str(tmp_path)])
    assert code == DTC.EXIT_INVALID, (
        "synthetic rows, so every box VALIDITY gate is UNKNOWN and the page is "
        "not quotable; the run still completes and still names what it refused")
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "UnpairableComparison" in out
    assert "prices the CONFIG effect at fixed dtype" in out


def test_a_calibration_with_no_fp8_peak_refuses_before_it_plans_anything(
        tmp_path, capsys):
    """A typed refusal that arrives as a traceback is not a usable refusal.

    The A100 is the live case: it has no fp8 tensor cores, so its calibration
    carries no fp8 entry, and every prediction on the page would otherwise die
    halfway through printing. Asked once, up front, with the fallback named.
    """
    code = DTC.main(["--dry-run", "--calibration",
                     "measured_nvidia_a100_sxm4_80gb",
                     "--card", "NVIDIA A100-SXM4-80GB",
                     "--out-dir", str(tmp_path)])
    assert code == DTC.EXIT_NOT_MEASURED
    out = capsys.readouterr().out
    assert "CalibrationIncomplete" in out
    assert "no fallback value" in out
    assert "--dtypes bf16" in out
    assert not list(tmp_path.glob("*/plan.json")), "it planned anyway"


def test_the_named_fallback_actually_runs_on_that_calibration(tmp_path, capsys):
    """The refusal above points at `--dtypes bf16`, so that had better work."""
    code = DTC.main(["--dry-run", "--dtypes", "bf16", "--calibration",
                     "measured_nvidia_a100_sxm4_80gb",
                     "--gpu-name", "NVIDIA_A100-SXM4-80GB",
                     "--card", "NVIDIA A100-SXM4-80GB",
                     "--out-dir", str(tmp_path)])
    assert code == DTC.EXIT_NOT_MEASURED
    out = capsys.readouterr().out
    assert "NO DTYPE PREDICTION" in out
    assert "145.8 FLOP/byte" in out       # the A100 ridge, from its own file
    assert "fp8" not in out.split("## Predictions")[1].split("ridge 145.8")[0]
    assert list(tmp_path.glob("*/plan.json"))


# --------------------------------------------------------------------------
# The free half of the experiment must SHOW its verdicts.
# --------------------------------------------------------------------------

def test_the_dry_run_renders_c1_and_c2_as_pass_or_fail(tmp_path, capsys):
    """A gate is a number against a threshold printed as PASS or FAIL.

    This branch used to print the raw inputs -- the DIFFERS/SAME table and
    "BLOCK_SIZE_M agrees in 4/4 bracketing cells" -- and then a banner claiming
    "C1 and C2 are DECIDED here", with zero PASS/FAIL strings in 110 lines. C1
    is the PREMISE the pod run rests on: if it ever FAILED there would be no run
    worth buying, and the free half is where that has to be visible before the
    box is rented.
    """
    code = DTC.main(["--dry-run", "--card", H200, "--out-dir", str(tmp_path)])
    assert code == DTC.EXIT_NOT_MEASURED
    out = capsys.readouterr().out
    assert f"[{DTC.PASS:7s}] CLAIM    C1" in out
    assert f"[{DTC.PASS:7s}] CLAIM    C2" in out
    # ...and the gates it CANNOT decide are named as needing the box, never
    # rendered as passes.
    assert "C3 pure dtype" in out
    assert f"[{DTC.PASS:7s}] CLAIM    C3" not in out


def test_the_dry_run_verdicts_come_from_the_shipped_gate_builder(tmp_path, capsys):
    """Not a second implementation that agrees with the first until it does not.

    The C1 and C2 lines a dry run prints have to be the same objects the pod run
    scores, over an analysis with no measurements in it: both read
    `crossing_bracket_cells`, which is derived from vLLM's config lookup and
    needs no timing.
    """
    cells, _ = DTC.plan_cells(list(DTC.DEFAULT_MODELS), list(DTC.DEFAULT_TOKENS),
                              "NVIDIA_H200")
    ceilings = DTC.load_ceilings("measured_nvidia_h200")
    gates = DTC.build_gates(DTC.analyse(cells, {}, ceilings, list(DTC.DTYPES)),
                            ceilings, list(DTC.DTYPES), fp8_note="",
                            synthetic=True)
    by_name = {g.name.split()[0]: g for g in gates}
    assert by_name["C1"].verdict == DTC.PASS
    assert by_name["C2"].verdict == DTC.PASS
    DTC.main(["--dry-run", "--card", H200, "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    # `with_result=False`: a dry run measured nothing and exits REFUSED, so it
    # prints the verdicts as prose and no RESULT line. Everything else about
    # the block is the shipped renderer's.
    assert by_name["C1"].render(with_result=False) in out
    assert by_name["C2"].render(with_result=False) in out


def test_a_missing_fp8_peak_costs_the_predicted_band_and_nothing_else():
    """The refusal the A100 branch POINTS AT had better run.

    `--dtypes bf16` is what a card with no fp8 units is told to use, and
    `predicted_ms` raises `CalibrationIncomplete` for the fp8 peak that
    calibration does not carry. Raised through `analyse` it took the whole run
    with it. The band is corroboration beside C3's measured tilt, not an input
    to any gate, so its absence is a recorded refusal.
    """
    cells, _ = DTC.plan_cells(list(DTC.DEFAULT_MODELS), list(DTC.DEFAULT_TOKENS),
                              "NVIDIA_A100-SXM4-80GB")
    ceilings = DTC.load_ceilings("measured_nvidia_a100_sxm4_80gb")
    assert DTC.FP8 not in ceilings.peak_flops or not ceilings.peak_flops[DTC.FP8]
    analysis = DTC.analyse(cells, {}, ceilings, [DTC.BF16])
    assert analysis.predicted == {}
    assert any("CalibrationIncomplete" in r for r in analysis.refusals)
    # C1 and C2 are statements about the config lookup and survive intact.
    gates = {g.name.split()[0]: g for g in
             DTC.build_gates(analysis, ceilings, [DTC.BF16], fp8_note="",
                             synthetic=True)}
    assert gates["C1"].verdict == DTC.PASS
    assert gates["C2"].verdict == DTC.PASS


# --------------------------------------------------------------------------
# C4, whose PASS branch was never planted
# --------------------------------------------------------------------------

def _planted_shift(model: str, arm: str, tilt: float):
    """A `Shift` whose tilt is exactly `tilt` and whose branches are absent.

    C4 reads `tilt` and nothing else, so everything else on the record is set to
    the value that says "this is not a measurement": no points, no branches, no
    slopes. A planted record that carried plausible branches would let a later
    gate read it as data.
    """
    return DTC.Shift(arm=arm, model=model, points=(), low_tokens=(),
                     high_tokens=(), r_low=tilt, r_high=1.0,
                     memory_tokens=(), compute_tokens=(), rm=None, rc=None,
                     memory_slope_bf16=None, compute_slope_bf16=None,
                     log_slope=None, unpaired=())


def _c4(ceilings, native: float, matched: float, models=("mixtral-8x7b",)):
    """Score C4 alone over planted tilts. The SHIPPED gate builder, not a copy."""
    analysis = DTC.Analysis(models=list(models))
    for model in models:
        analysis.shifts[(model, DTC.NATIVE_ARM)] = _planted_shift(
            model, DTC.NATIVE_ARM, native)
        for arm in DTC.MATCHED_ARMS:
            analysis.shifts[(model, arm)] = _planted_shift(model, arm, matched)
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), fp8_note="")
    return next(g for g in gates if g.name.startswith("C4"))


def test_c4_passes_when_pinning_the_config_removes_most_of_the_excess(ceilings):
    """THE BRANCH NOTHING HAD EVER PLANTED.

    `--self-test` cannot reach it: the surviving confound runs through
    BLOCK_SIZE_K, `predicted_ms` does not read BLOCK_SIZE_K, so every synthetic
    matched arm tilts by ~1.000 and the share it apportions is ~0. C4 therefore
    printed FAIL or UNKNOWN in every off-GPU run this repository has ever done,
    and a gate whose PASS branch has never been seen is a gate nobody has
    tested. Planted here at the tilts a PASSING pod run would produce: the
    native arm 1.20, the pinned arms 1.04, so the config carries
    (1.20 - 1.04) / 0.20 = 80% of the excess, comfortably over the 50% rule.
    """
    gate = _c4(ceilings, native=1.20, matched=1.04)
    assert gate.verdict == DTC.PASS
    assert "80" in gate.observed
    assert gate.result_line().startswith("RESULT: CLAIM C4 PASS")


def test_c4_fails_when_the_excess_survives_pinning_the_config(ceilings):
    """The FAIL branch beside it, so the PASS above is not a gate that
    always passes. Native 1.20 against matched 1.18 leaves the config 10%."""
    gate = _c4(ceilings, native=1.20, matched=1.18)
    assert gate.verdict == DTC.FAIL
    assert "10" in gate.observed
    assert "the confounded excess is mostly NOT the config" in gate.invalidates


def test_c4_is_unknown_rather_than_huge_when_there_is_no_excess(ceilings):
    """A native arm at or below 1.0 has nothing to apportion.

    Dividing by `tilt_native - 1` there would print a share of some thousands
    of percent and the gate would read PASS on a run where the effect it
    explains does not exist.
    """
    gate = _c4(ceilings, native=0.98, matched=0.90)
    assert gate.verdict == DTC.UNKNOWN
    assert "no excess over 1.0 to apportion" in gate.observed


# --------------------------------------------------------------------------
# the quoted constants, and what they are allowed to claim
# --------------------------------------------------------------------------

def test_the_published_numbers_are_the_uniform_only_rescoring():
    """Pooling four routing kinds into one crossing is invalid by this repo's
    own rule, and these constants were the pooled numbers.

    Regenerated with `scripts/crossing_report.py ... --routing uniform` over the
    canonical four-arm bf16 pool and the two fp8 arms; the ratios below are
    those two outputs divided cell for cell. The pooled table was
    454/810/922/3240 with a headline of 1.149 +/- 0.069.
    """
    assert DTC.PUBLISHED_BF16_CROSSING == {
        "mixtral-8x7b": 313.0, "qwen2-57b-a14b": 730.0,
        "deepseek-v2-lite": 931.0, "deepseek-v3": 2925.0}
    fp8_vllm = {"mixtral-8x7b": 391.0, "qwen2-57b-a14b": 806.0,
                "deepseek-v2-lite": 962.0, "deepseek-v3": 2930.0}
    for model, bf16 in DTC.PUBLISHED_BF16_CROSSING.items():
        assert DTC.PUBLISHED_CROSSING_RATIO[model]["vllm"] == pytest.approx(
            fp8_vllm[model] / bf16, abs=0.005), model
    ratios = [DTC.PUBLISHED_CROSSING_RATIO[m][k]
              for m in DTC.PUBLISHED_CROSSING_RATIO for k in ("vllm", "sglang")]
    assert DTC.PUBLISHED_SHIFT == pytest.approx(
        sum(ratios) / len(ratios), abs=0.005)
    # The dispersion is the half that MOVED: pooling narrowed it by a third.
    assert DTC.PUBLISHED_SHIFT_SD > 0.09


def test_the_alpha_curve_is_labelled_pooled_and_says_the_a100_disagrees():
    """The comment used to assert "monotone and saturating, reproduced on both
    the H200 and the A100". The A100's own surface runs the other way.

    Checked as a test rather than left as prose because the constants are one
    card's unpaired medians and the next reader will otherwise take the four
    numbers for a measured curve, which is what the last one did.
    """
    assert DTC.ALPHA_BY_GROUP_M is DTC.ALPHA_BY_GROUP_M_POOLED
    doc = DTC.__doc__ or ""
    assert "monotone and saturating" not in doc
    source = (ROOT / "scripts" / "dtype_tile_confound.py").read_text()
    head = source.split("ALPHA_BY_GROUP_M_POOLED")[0]
    assert "POOLED, UNPAIRED MEDIANS" in head
    # The paired A100 medians, off the one cell that holds every level. The
    # pooled 0.736/0.745/0.782 survive only as the named artefact.
    assert "0.651" in head and "0.706" in head and "0.739" in head
    assert "POOLED ARTEFACT" in head
    # And the numbers are still exactly the H200 s4 medians they came from.
    assert DTC.ALPHA_BY_GROUP_M_POOLED == {1: 0.84, 8: 0.73, 16: 0.68, 64: 0.67}


def test_the_a100_surface_really_does_run_the_other_way():
    """Not a number typed into a comment: read out of the committed arm.

    THE ASSERTION THIS TEST USED TO MAKE WAS THE POOLED ARTEFACT. It read
    `medians[8] < medians[16] < medians[64]` off a SURFACE.txt written by the
    unpaired version of `alpha_surface.py`, which ordered the levels by a
    string sort (1, 16, 64, 8) and took each median over a different set of
    fits (n = 4, 4, 2, 2). Comparing medians over different cells is the exact
    error the dict's comment exists to withdraw, so the test was committing it
    to prove it.

    Against the paired file there is ONE (model, bn, bm) cell holding all four
    levels, so what the arm supports is: a non-monotone shape off a single
    cell, and no direction at all.
    """
    text = (ROOT / "results" / "published"
            / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
            / "SURFACE.txt").read_text()
    block = text.split("alpha against GROUP_SIZE_M")[1].split("alpha against")[0]
    medians = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit() and "median" in line:
            medians[int(parts[0])] = float(parts[-1] if "corrected" not in line
                                           else parts[-3])
    assert medians == {1: 0.651, 8: 0.706, 16: 0.739, 64: 0.679}, medians
    # Rises to G=16, falls at G=64: not monotone, so it cannot reproduce a
    # monotone dict, and not a clean reversal of one either.
    assert medians[8] < medians[16]
    assert medians[64] < medians[16]
    # And it rests on one cell, which is why the file refuses to price the
    # change. A single cell is the reason this is silence rather than evidence.
    assert "only 1 matched cell" in block
    assert "no\n  paired change here is comparable" in block
    assert DTC.ALPHA_BY_GROUP_M_POOLED[8] > DTC.ALPHA_BY_GROUP_M_POOLED[16]


# --------------------------------------------------------------------------
# the instrument, the provenance and the card
# --------------------------------------------------------------------------

def test_the_private_timing_loop_is_gone():
    """`grep -c "def time_call" scripts/*.py` must be 0 for this file.

    The copy that used to live here created its events inside the loop and
    synchronised every iteration, while the roof its ratios are read against
    was measured queue-deep. Two instruments, one comparison.
    """
    source = (ROOT / "scripts" / "dtype_tile_confound.py").read_text()
    assert "def time_call" not in source
    assert "def time_calls" not in source
    assert "timing.time_kernel(" in source


def test_every_timing_column_reaches_the_csv_header():
    """A column the row carries and the header does not is a column csv drops."""
    for name in DTC.TIMING_CSV_COLUMNS:
        assert name in DTC.CSV_COLUMNS, name
    for name in ("prov_git_sha", "prov_gpu_name", "prov_instrument"):
        assert name in DTC.CSV_COLUMNS, name


def test_the_timing_columns_survive_a_write_and_a_resume(tmp_path, planned):
    """A resumed row must carry the state it was measured in, not this run's.

    The tri-state flags are the load-bearing part: an empty cell is NOT
    DETERMINED and must come back as None, because reading it as False drops a
    good cell and reading it as True keeps a throttled one.
    """
    cells, _ = planned
    cell = cells[0]
    prov = PV.provenance_block(instrument=DTC.timing.TIMING_BASIS)
    store = DTC.Store(tmp_path / "timings.csv", prov=prov)
    result = DTC.ArmResult(cell.model, cell.num_tokens, DTC.NATIVE_ARM,
                           DTC.BF16, ms_median=1.0, n_samples=3,
                           instrument=DTC.timing.TIMING_BASIS, warmup_ms=300.0,
                           iters=120, trials=3, sm_clock_load_mhz=1490.0,
                           clock_level_ok=False,
                           clock_level_side=DTC.timing.LEVEL_LOW,
                           clock_drift_ok=True,
                           l2_flush=True, host_bound=None)
    meta = {"run_id": "r", "gpu_name": "g", "torch_version": "t",
            "vllm_version": "v", "routing": "uniform", "seed": 0}
    store.write(result, cell, meta)
    store.close()
    back = DTC.Store(tmp_path / "timings.csv").restore(result.key)
    assert back.instrument == DTC.timing.TIMING_BASIS
    assert back.iters == 120 and back.trials == 3
    assert back.warmup_ms == pytest.approx(300.0)
    assert back.sm_clock_load_mhz == pytest.approx(1490.0)
    assert back.clock_level_ok is False
    assert back.clock_level_side == DTC.timing.LEVEL_LOW, \
        "the side must survive a resume or a LOW arm comes back as HIGH-less"
    assert back.clock_drift_ok is True
    assert back.l2_flush is True
    assert back.host_bound is None, "an empty flag is NOT DETERMINED, not False"
    header = (tmp_path / "timings.csv").read_text().splitlines()[0]
    assert "prov_git_sha" in header


def test_one_bad_repeat_makes_the_whole_arm_say_so():
    """`summarise_timings` folds the repeats so a filter cannot be more
    permissive than the repeats it summarises."""
    class T:
        def __init__(self, level, drift, host, first=1500.0, last=1500.0):
            self.instrument = DTC.timing.TIMING_BASIS
            self.l2_flush = True
            self.iters, self.trials, self.warmup_ms = 100, 3, 300.0
            self.sm_clock_load_mhz = 1500.0
            # The first and last UNDER-LOAD samples the DRIFT verdict is
            # computed from, which the row carries since 2026-09-09.
            self.sm_clock_start_mhz, self.sm_clock_end_mhz = first, last
            self.clock_samples, self.clock_note = 20, ""
            self.clock_level_ok, self.clock_drift_ok = level, drift
            self.host_bound = host

    good = DTC.ArmResult("m", 1, "native", DTC.BF16)
    DTC.summarise_timings(good, [T(True, True, False), T(True, True, False)])
    assert (good.clock_level_ok, good.clock_drift_ok, good.host_bound) == (
        True, True, False)
    bad = DTC.ArmResult("m", 1, "native", DTC.BF16)
    DTC.summarise_timings(bad, [T(True, True, False), T(False, True, True)])
    assert bad.clock_level_ok is False
    assert bad.host_bound is True
    unknown = DTC.ArmResult("m", 1, "native", DTC.BF16)
    DTC.summarise_timings(unknown, [T(True, True, False), T(None, None, None)])
    assert unknown.clock_level_ok is None
    assert unknown.host_bound is None

    # The under-load pair comes from the repeat that MOVED MOST, so a row
    # cannot look steadier than its worst repeat. The 2026-09-09 CSV kept the
    # DRIFT verdict and dropped these two numbers, and 80 of 118 drifted arms
    # could not be read as a rising governor or a falling throttle.
    moved = DTC.ArmResult("m", 1, "native", DTC.BF16)
    DTC.summarise_timings(moved, [T(True, True, False, 1500.0, 1495.0),
                                  T(True, False, False, 1500.0, 1380.0)])
    assert (moved.load_clock_first_mhz, moved.load_clock_last_mhz) == (1500.0, 1380.0)
    assert moved.clock_drift_ok is False
    assert moved.clock_samples == 20


def _record(level_ok, side, load_mhz):
    """A `KernelTiming` the way `time_kernel` writes it against a 1515 MHz
    bf16-GEMM reference: the flag AND the side, never one without the other."""
    return KernelTiming(ms_p50=1.0, ms_p90=1.1, ms_min=0.9, ms_std=0.01, iters=10,
                        trials=3, warmup_ms=500.0, l2_flush=True,
                        sm_clock_load_mhz=load_mhz, sm_clock_start_mhz=load_mhz,
                        sm_clock_end_mhz=load_mhz, clock_level_ok=level_ok,
                        clock_drift_ok=True, samples=30, warmup_calls=5,
                        flush_mb=64, clock_samples=20, clock_source="nvml",
                        clock_poll_ms=5.0, host_bound=False, host_enqueue_ms=0.0,
                        instrument=TIMING_BASIS, clock_level_side=side,
                        reference_clock_mhz=1515.0)


def test_v5_keeps_both_level_sides_and_excludes_only_a_drifted_arm(planned,
                                                                   ceilings):
    """The planted pair, both folded from real `KernelTiming` records.

    HIGH: a memory-shaped dtype arm boosted to 1980 MHz against the 1515 MHz
    reference. `time_kernel` fails LEVEL on it with side "high" (03df2d4), and
    on an H200 that is the normal state of a memory-bound cell.
    LOW: the same arm at 1400 MHz. Until 2026-09-08 the census excluded the
    HIGH side; until 2026-09-09 it excluded the LOW side. It now excludes
    NEITHER: the 750-cell H200 census established that the under-load clock is
    set per tile by the kernel's own power draw under the 700 W cap
    (BLOCK_M=128/N=64 median 1395 MHz, BLOCK_M=256 1650, memory-shaped
    1950-1980, the calibration GEMM 1485 at 691 W), so both sides are steady
    operating points of a tile family and excluding on one is a rule against a
    tile. Both are counted, both are kept, and DRIFT is what excludes.
    """
    cells, _ = planned
    results = synth(cells, ceilings)
    _flag_every_arm(results)
    arms = list(results[list(results)[2]].values())
    high = arms[0]
    DTC.summarise_timings(high, [_record(False, DTC.timing.LEVEL_HIGH, 1980.0)] * 2)
    assert high.clock_level_ok is False and high.clock_level_side == DTC.timing.LEVEL_HIGH
    low = arms[1]
    DTC.summarise_timings(low, [_record(False, DTC.timing.LEVEL_LOW, 1400.0)] * 2)
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert not analysis.clock_throttled
    assert analysis.clock_flagged_arms == 0
    assert analysis.clock_high_side_arms == 1
    assert analysis.clock_low_side_arms == 1
    gates = DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "", synthetic=False)
    assert verdicts(gates)["V5"] == DTC.PASS
    v5 = next(g for g in gates if g.name.startswith("V5"))
    assert "1 below the band, 1 above it, both scored" in v5.observed
    assert "excluded" not in v5.observed.split("; ")[2].split("DRIFTED")[0]

    # A DRIFTED arm is the one exclusion, and it is dropped from the ratios as
    # well as failing the gate, which is what "excluded" on that line means.
    drifted = arms[2]
    drifted.clock_drift_ok = False
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    assert analysis.clock_throttled and analysis.clock_flagged_arms == 1
    assert verdicts(DTC.build_gates(analysis, ceilings, list(DTC.DTYPES), "",
                                    synthetic=False))["V5"] == DTC.FAIL
    assert DTC.cell_ms({("native", DTC.BF16): drifted}, "native", DTC.BF16,
                       cells[0]) is None or drifted.arm != "native"


def test_the_level_side_folds_low_dominant_and_refuses_a_foreign_word():
    """A mixed arm sat on both sides of the band and is LOW; a word the
    instrument never writes is refused, not read as a pass."""
    arm = DTC.ArmResult("m", 1, "native", DTC.BF16)
    DTC.summarise_timings(arm, [_record(False, DTC.timing.LEVEL_HIGH, 1980.0),
                                _record(False, DTC.timing.LEVEL_LOW, 1400.0)])
    assert arm.clock_level_side == DTC.timing.LEVEL_LOW
    with pytest.raises(ValueError, match="not a side"):
        DTC.summarise_timings(DTC.ArmResult("m", 1, "native", DTC.BF16),
                              [_record(False, "sideways", 1980.0)])


def test_a_run_off_gpu_refuses_rather_than_labelling_itself_an_h200(
        tmp_path, capsys):
    """R6. The literal "NVIDIA H200" was the fallback, and it did two jobs.

    It named the plan for a card the machine might not be, and, because the
    lookup device decides which of vLLM's tuned files `resolve_tile` reads, it
    chose the tuned half of the confound for a card that may ship no tuned file
    at all. There is no CUDA device in this test process, which is the live
    case.
    """
    code = DTC.main(["--dry-run", "--out-dir", str(tmp_path)])
    assert code == exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "NoCardToLabel" in out
    assert "--card" in out
    assert not list(tmp_path.glob("*/plan.json")), "it planned anyway"


def test_the_named_flag_is_what_makes_the_refused_run_go(tmp_path, capsys):
    """The refusal above points at --card, so --card had better be enough."""
    assert DTC.main(["--dry-run", "--card", H200,
                     "--out-dir", str(tmp_path)]) == exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "NoCardToLabel" not in out
    plans = list(tmp_path.glob("*/plan.json"))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text())
    for key in ("git_sha", "gpu_name", "ridge_source", "bandwidth_source",
                "instrument"):
        assert key in plan, key
    assert plan["provenance"]["instrument"] == DTC.timing.TIMING_BASIS
    assert plan["provenance"]["missing"]["gpu_name"], (
        "off GPU the block must say WHY it has no card, not leave it blank")


def test_the_summary_carries_the_provenance_block(tmp_path):
    DTC.main(["--self-test", "2.033", "--self-test-alpha", "0.2",
              "--card", H200, "--out-dir", str(tmp_path)])
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert summary["instrument"] == DTC.timing.TIMING_BASIS
    assert summary["provenance"]["ridge_source"].endswith(
        "measured_nvidia_h200.yaml")
    assert summary["card"] == H200


# --------------------------------------------------------------------------
# the MDE, which no arm in this study used to state
# --------------------------------------------------------------------------

def test_the_mde_is_derived_from_the_measured_spread_not_a_prior():
    """B14. The convention this repo used was a 0.5% assumption, below the whole
    measured range; the spread here is the 26 published reports' own."""
    assert DTC.MEASURED_SPREAD_MEDIAN > 0.005
    assert DTC.MEASURED_SPREAD_MAX > DTC.MEASURED_SPREAD_MEDIAN
    wide = DTC.mde_of_ratio(DTC.MEASURED_SPREAD_MAX, 3)
    narrow = DTC.mde_of_ratio(DTC.MEASURED_SPREAD_MEDIAN, 3)
    assert wide > narrow, "more noise must cost resolution, not buy it"
    assert DTC.mde_of_ratio(DTC.MEASURED_SPREAD_MEDIAN, 12) < narrow
    with pytest.raises(ValueError):
        DTC.mde_of_ratio(0.0, 3)
    with pytest.raises(ValueError):
        DTC.mde_of_ratio(0.01, 0)


def test_the_measured_spread_is_what_the_published_reports_say():
    """Read out of the corpus, so a re-published arm moves the number here."""
    import glob
    import statistics as st
    values = []
    for path in glob.glob(str(ROOT / "results" / "published"
                              / "*" / "*.report.json")):
        spread = json.loads(Path(path).read_text()).get("timing_spread_median")
        if spread:
            values.append(float(spread))
    assert len(values) >= 20, "the corpus this constant is taken from is gone"
    assert DTC.MEASURED_SPREAD_MEDIAN == pytest.approx(st.median(values),
                                                       abs=0.0005)
    assert DTC.MEASURED_SPREAD_MAX == pytest.approx(max(values), abs=0.0005)


def test_the_mde_line_says_when_the_design_cannot_see_the_effect():
    """The FAIL branch of the MDE report, which is the branch that matters:
    a design that cannot resolve its own effect has to say so before the box
    is rented, not after."""
    args = DTC.build_parser().parse_args(["--reps", "1"])
    # PLANTED, because on the real corpus the design passes at both ends: even
    # one repeat at the worst measured spread resolves 0.072 against an effect
    # of 0.131. A branch that the shipped numbers cannot reach is still a branch
    # the reader will meet the first time a noisier pod appears, so it is
    # planted here at a spread five times the worst one measured.
    planted = DTC.render_mde(args, spreads=(("median", 0.0077),
                                            ("planted", 0.09)))
    assert "CANNOT resolve the effect" in planted
    assert "resolves the effect" in planted
    assert "CANNOT resolve" not in DTC.render_mde(args)


# --------------------------------------------------------------------------
# The acceptance check for the exit-code repair: the log and the process say
# the same thing in every mode this file can reach without a GPU.
# --------------------------------------------------------------------------

#: Every off-GPU mode of this script and the code it must return. Only
#: `--self-test` scores gates; everything else here measured nothing.
OFF_GPU_MODES = [
    (["--dry-run"], exit_codes.REFUSED),
    # Every box VALIDITY gate is demoted to UNKNOWN by construction on a
    # generated world, and UNKNOWN is not a pass, so a synthetic page is INVALID
    # whatever its claims say. The `claims` line beside the exit is what makes
    # the self test discriminating.
    (["--self-test", "2.033"], exit_codes.INVALID),
    # Two refusals that used to be `raise SystemExit(<str>)` and so exited 1 =
    # CLAIM_FAIL, a measured refutation, from a process that had only parsed its
    # arguments.
    (["--dtypes", "bf16,fp8_e5m2", "--dry-run"], exit_codes.REFUSED),
    (["--dtypes", "fp8_e4m3", "--dry-run"], exit_codes.REFUSED),
    # No cell survives planning, so there is nothing to buy the box for.
    (["--tokens", "1", "--dry-run"], exit_codes.REFUSED),
    # The measuring path on a machine with no CUDA and no vLLM.
    ([], exit_codes.REFUSED),
]


@pytest.mark.parametrize("argv,code", OFF_GPU_MODES,
                         ids=[" ".join(a) or "bare" for a, _ in OFF_GPU_MODES])
def test_the_log_and_the_exit_code_agree_in_every_off_gpu_mode(
        argv, code, tmp_path, capsys):
    """The whole repair, stated as one property instead of as prose.

    For every mode this file can reach on a laptop, the RESULT lines it printed
    and the integer it returned have to be the same verdict. `classify_text`
    recomputes the code from the log; a log with NO RESULT lines raises
    `NoGatesScored`, and `moe.bench.exit_codes` documents that as exactly what a
    REFUSED log looks like from there, so the two cases are one rule: score
    gates and match `classify`, or score none and return REFUSED.
    """
    rc = DTC.main([*argv, "--card", H200, "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == code, out
    lines = exit_codes.parse_result_lines(out)
    if lines:
        assert exit_codes.classify_text(out) == rc
    else:
        assert rc == exit_codes.REFUSED, (
            "a run that scored no gate printed no RESULT line, so its log "
            "implies REFUSED and nothing else")


def test_a_page_whose_arms_all_failed_prints_no_result_line_and_refuses(
        tmp_path, monkeypatch, capsys):
    """THE VACUOUS PAGE, PLANTED, because no laptop reaches it on its own.

    Zero arms produced a timing, so every gate was built over nothing. The run
    used to print the whole gate table WITH its RESULT lines and then return
    REFUSED: `classify_text` read INVALID out of gates that had examined
    nothing, and the process said "nothing was measured". That is the
    log-versus-exit-code split `moe.bench.exit_codes` exists to close, in the
    file whose header says it cannot happen. The gate table still prints -- a
    reader wants to see which gates were built -- with `with_result=False`.
    """
    monkeypatch.setattr(DTC, "synthetic_results", lambda *a, **kw: {})
    rc = DTC.main(["--self-test", "2.033", "--card", H200,
                   "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert exit_codes.parse_result_lines(out) == []
    assert "zero arms produced a timing" in out
    # The human table is still there; only the greppable line is withheld.
    assert "## Gates" in out and "VALIDITY" in out


def test_an_unplanned_crash_exits_ERROR_and_never_CLAIM_FAIL(monkeypatch, capsys):
    """The apparatus breaking must not be filed as one of the arm's outcomes.

    An exception left to propagate exits the interpreter ONE, and ONE is
    CLAIM_FAIL, which `moe/bench/exit_codes.py` puts in FINISHED_CODES: the
    driver records the arm as finished, skips it on every resume, leaves
    RETRY_ARMS at zero and exits the session 0 over an arm that never measured.
    ERROR (4) is outside FINISHED_CODES so the two can be told apart, and the
    traceback is printed rather than swallowed because a bare code names
    nothing to fix. Planted rather than argued: dtype_tile_confound had no top-level handler
    until 2026-09-02.
    """
    from moe.bench import exit_codes as EX

    def explode(argv=None):
        raise RuntimeError("planted: the allocator gave up halfway")

    monkeypatch.setattr(DTC, "_main", explode)
    code = DTC.main([])
    err = capsys.readouterr().err
    assert code == EX.ERROR
    assert code != EX.CLAIM_FAIL
    assert code not in EX.FINISHED_CODES, "an apparatus failure must stay retryable"
    assert EX.ledger_state(code) == "RETRY"
    assert "planted: the allocator gave up halfway" in err, \
        "the traceback was swallowed"
    assert "RuntimeError" in err


def test_an_fp8_arm_is_levelled_against_the_fp8_gemm_clock_not_the_bf16_one(tmp_path):
    """2026-09-09 H200 calibration: bf16 GEMM 1470 MHz, fp8 GEMM 1380 MHz.
    1380/1470 = 0.939 < LEVEL_FRACTION; against the bf16 clock every fp8 arm
    is LOW, V5 FAILs and the arm is INVALID on a sound measurement."""
    import yaml

    from moe.bench import timing as T
    from moe.bench.calibrate import LoadedClock
    bf16 = LoadedClock("bf16 GEMM", (1485, 1485, 1455, 1470, 1470), 1470, 2.0, 1470)
    fp8 = LoadedClock("fp8 GEMM", (1395, 1365, 1395, 1380, 1365), 1380, 2.15, 1980)
    doc = {"name": "NVIDIA H200 (measured)", "verified": True, "source": "m",
           "memory": {"bandwidth_tb_s": 4.3744},
           "compute_dense_tflops": {"bf16": 712.3, "fp16": 712.3,
                                    "fp8_e4m3": 1447.7, "fp8_e5m2": 1447.7},
           "detail": {"gpu_name": "NVIDIA H200", "gemm_clock": bf16.as_dict(),
                      "fp8_gemm_clock": fp8.as_dict()}}
    (tmp_path / "measured_nvidia_h200.yaml").write_text(yaml.safe_dump(doc))
    ceil = DTC.load_ceilings("measured_nvidia_h200", directory=tmp_path)
    assert ceil.reference_clock_mhz == 1470.0
    assert ceil.reference_for(DTC.BF16) == 1470.0
    assert ceil.reference_for(DTC.FP8) == 1380.0
    assert T.level_side(1380.0, ceil.reference_for(DTC.FP8)) == ""
    assert T.level_side(1380.0, ceil.reference_for(DTC.BF16)) == T.LEVEL_LOW


# --------------------------------------------------------------------------
# 8. THE 2026-09-09 H200 SESSION. Four defects, each reproduced here off GPU.
#
#    The cfg_fp8 arm forced the whole fp8 config on bf16 weights. At 22 of the
#    28 default cells that tile needs 294912-409600 B of pipeline shared memory
#    against sm_90's 232448, so it raised OutOfResources inside vLLM's
#    `override_config`, which has no try/finally, and the fp8 config stayed
#    installed process-wide. All 28 bf16 `native` arms then compiled the leaked
#    tile and errored, 13 fp8 `native` arms timed the previous cell's tile, and
#    V4 read the leak as a derivation error in this script. The rerun could not
#    have repaired itself either: the run id is a hash of the plan, so the same
#    command resumes the same directory, and `Store.restore` handed the 50
#    errored rows back as though they were results.
# --------------------------------------------------------------------------

class LeakyOverride:
    """vLLM v0.27.1's `override_config`, verbatim in its semantics.

    `old_config = _config; _config = config; yield; _config = old_config`
    (vllm/model_executor/layers/fused_moe/__init__.py:52-58). No try/finally,
    so an exception inside the body never resumes the generator.
    """

    def __init__(self):
        self._config = None

    def get_config(self):
        return self._config

    @contextlib.contextmanager
    def override_config(self, config):
        old = self._config
        self._config = config
        yield
        self._config = old            # never reached if the body raises

    def hooks(self, module_name="tests.leaky_vllm"):
        sys.modules[module_name] = self
        return (self.override_config, self.get_config, module_name)


def test_the_override_leak_is_reproduced_and_then_refused(planned):
    """The chain that cost 41 arms, planted with vLLM's own semantics."""
    cells, _ = planned
    cell = cells[0]

    # 1. THE LEAK, with no guard: an exception inside the body leaves the
    #    override installed, exactly as the pod's log shows.
    leaky = LeakyOverride()
    forced = DTC.full_transplant(cell, DTC.FP8)
    with pytest.raises(RuntimeError, match="OutOfResources"):
        with leaky.override_config(forced):
            raise RuntimeError("OutOfResources: shared memory")
    assert leaky.get_config() == forced, \
        "vLLM restores its global in a plain yield; this is the defect"

    # 2. THE GUARD. A native arm asks for no override, so a dirty global means
    #    the previous arm broke and this cell would time its kernel.
    with pytest.raises(DTC.OverrideLeaked) as caught:
        with DTC.override_context(leaky.hooks(), cell, DTC.NATIVE_ARM, DTC.BF16):
            pass
    assert "outside any forced arm" in str(caught.value)
    assert DTC.format_config(forced) in str(caught.value)

    # 3. THE REPAIR. A forced arm that raises leaves the global CLEAN, so the
    #    leak cannot reach the next arm at all.
    clean = LeakyOverride()
    hooks = clean.hooks("tests.leaky_vllm_clean")
    with pytest.raises(RuntimeError):
        with DTC.override_context(hooks, cell, "cfg_bf16", DTC.BF16):
            raise RuntimeError("OutOfResources: shared memory")
    assert clean.get_config() is None
    # And the next arm runs, which is the whole point of the finally.
    with DTC.override_context(hooks, cell, DTC.NATIVE_ARM, DTC.BF16):
        assert clean.get_config() is None


def test_the_shared_memory_arithmetic_reproduces_the_pods_required_bytes():
    """`num_stages x (BM*BK + BK*BN) x element bytes`, against the three
    `Required:` counts the 2026-09-09 OutOfResources messages printed."""
    for stages, bn, want in ((5, 256, 409600), (4, 256, 393216), (3, 256, 294912)):
        cfg = {"BLOCK_SIZE_M": 64 if stages == 5 else 128, "BLOCK_SIZE_N": bn,
               "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 8,
               "num_stages": stages}
        assert DTC.smem_footprint(cfg, 2) == want
        assert want > DTC.SM90_SMEM_LIMIT
        # The same tile at fp8's one byte fits, which is why only the bf16 half
        # of the transplant was ever infeasible.
        assert DTC.smem_footprint(cfg, 1) <= DTC.SM90_SMEM_LIMIT or want > 2 * DTC.SM90_SMEM_LIMIT


def test_the_full_fp8_transplant_is_infeasible_at_22_of_28_cells(planned, dtypes):
    """The number that re-scoped the arm, derived off GPU."""
    cells, _ = planned
    over = [c for c in cells
            if DTC.smem_footprint(DTC.full_transplant(c, DTC.FP8), 2)
            > DTC.SM90_SMEM_LIMIT]
    assert (len(over), len(cells)) == (22, 28)
    # The re-scoped arm moves the two keys the study is about and fits at both
    # widths at every cell, which is what makes it runnable at all.
    assert DTC.MATCHED_ARM_SCOPE["tile_fp8"] == ("BLOCK_SIZE_M", "GROUP_SIZE_M")
    assert DTC.pairing_counts(cells, dtypes) == {"cfg_bf16": 28, "tile_fp8": 28}
    for cell in cells:
        for arm in DTC.MATCHED_ARMS:
            for dtype in dtypes:
                assert not DTC.arm_is_infeasible(cell, arm, dtype), (cell.key, arm)


def test_the_plan_page_marks_every_infeasible_cell_with_its_byte_count(planned,
                                                                       dtypes):
    """What was withdrawn is PRINTED, not deleted: a reader checks the
    arithmetic rather than taking the re-scoping on trust."""
    cells, _ = planned
    page = DTC.render_pairing(cells, dtypes)
    rows = [line for line in page.splitlines() if line.startswith("| mixtral")
            or line.startswith("| qwen2")]
    assert len(rows) == 28
    assert sum("INFEASIBLE" in row for row in rows) == 22
    assert "INFEASIBLE at 22 of 28 cells" in page
    assert "409600 INFEASIBLE" in page and "294912 INFEASIBLE" in page
    assert str(DTC.SM90_SMEM_LIMIT) in page
    assert "tile_fp8: pins BLOCK_SIZE_M, GROUP_SIZE_M; pairs at 28 of 28 cells" in page
    assert "cfg_bf16: pins BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, " \
           "GROUP_SIZE_M, num_warps, num_stages; pairs at 28 of 28 cells" in page


def test_an_infeasible_arm_is_recorded_with_its_arithmetic_and_never_compiled(
        planned):
    """Belt and braces below the plan-time refusal: if an infeasible arm ever
    reaches the measuring loop it is written as a derivation, not handed to
    Triton, because the compile is what broke vLLM's global."""
    cells, _ = planned
    cell = next(c for c in cells
                if DTC.smem_footprint(DTC.full_transplant(c, DTC.FP8), 2)
                > DTC.SM90_SMEM_LIMIT)

    # `cfg_fp8` as it was until 2026-09-09: the WHOLE fp8 config, both widths.
    original = DTC.arm_config
    try:
        DTC.arm_config = lambda c, arm, dtype: (
            DTC.full_transplant(c, DTC.FP8) if arm == "tile_fp8"
            else original(c, arm, dtype))
        assert DTC.arm_is_infeasible(cell, "tile_fp8", DTC.BF16)
        assert not DTC.arm_is_infeasible(cell, "tile_fp8", DTC.FP8)
    finally:
        DTC.arm_config = original


def test_c3_and_c4_are_registered_against_the_arms_as_re_scoped(planned,
                                                                ceilings):
    """A pre-registration that moved has to say so on the page it is printed
    on, with the date. The threshold did not change; what the arms pin did."""
    cells, _ = planned
    gates = DTC.build_gates(DTC.analyse(cells, synth(cells, ceilings), ceilings,
                                        list(DTC.DTYPES)),
                            ceilings, list(DTC.DTYPES), "", synthetic=False)
    c3 = next(g for g in gates if g.name.startswith("C3"))
    c4 = next(g for g in gates if g.name.startswith("C4"))
    for gate in (c3, c4):
        assert "re-scoped on 2026-09-09" in gate.rule
    assert "tile_fp8 pins BLOCK_SIZE_M, GROUP_SIZE_M" in c3.rule
    assert "cfg_bf16 pins BLOCK_SIZE_M, BLOCK_SIZE_N" in c3.rule


def test_the_store_never_hands_back_an_errored_row_as_a_result(tmp_path,
                                                               planned):
    """The rerun could not repair itself: the run id is a hash of the plan, so
    the same command resumes this directory and the 50 errored arms came back
    as finished."""
    cells, _ = planned
    cell = cells[0]
    meta = {"run_id": "r", "gpu_name": "NVIDIA H200", "torch_version": "t",
            "vllm_version": "v", "routing": "uniform", "seed": 0,
            "provenance": PV.provenance_block(
                instrument=TIMING_BASIS, ridge=1.0, ridge_source="test",
                bandwidth=1.0, bandwidth_source="test", warmup_ms=1.0,
                target_ms=1.0)}
    store = DTC.Store(tmp_path / "timings.csv")
    good = DTC.ArmResult(cell.model, cell.num_tokens, "cfg_bf16", DTC.BF16,
                         config=DTC.arm_config(cell, "cfg_bf16", DTC.BF16))
    DTC.summarise_samples(good, [1.0])
    bad = DTC.ArmResult(cell.model, cell.num_tokens, DTC.NATIVE_ARM, DTC.BF16)
    bad.error = "OutOfResources: out of resource: shared memory, Required: 409600"
    store.write(good, cell, meta)
    store.write(bad, cell, meta)
    store.close()

    reopened = DTC.Store(tmp_path / "timings.csv")
    try:
        assert reopened.restore(good.key) is not None
        assert reopened.restore(bad.key) is None, \
            "an errored arm is not a finished one and must be re-timed"
        # The error itself is still readable, for a replay rebuilding the page.
        kept = reopened.restore(bad.key, include_errors=True)
        assert kept is not None and "409600" in kept.error
        assert reopened.finished == {good.key}
        assert set(reopened.done) == {good.key, bad.key}
    finally:
        reopened.close()


# --------------------------------------------------------------------------
# 2026-09-09 repairs: the band's WIDTH, and a CSV header that has to be this
# schema before a row is appended under it.
# --------------------------------------------------------------------------

def test_the_predicted_bands_width_is_compared_with_the_box_not_asserted():
    """R9 computed the band and left the sentence built around the old one.

    "1.016 to 1.053" was hardcoded on every page from every grid, and the tail
    "the box cannot resolve a band that narrow, and C3 cannot be read either
    way" was true of those 4 points. The band became computed; the tail did
    not, so the committed H200 calibration printed a 16-point band and then
    said the box could not resolve it, on the same gate line as a 0.94% p90
    timing spread.
    """
    # The committed H200 numbers: a 16-point band and a 0.94% p90 timing
    # spread, which the old tail called unresolvable in the same sentence that
    # printed both.
    wide = DTC.band_consequence([0.938, 1.099], band=0.0002, spread=0.0094)
    assert "16.1 points wide" in wide
    assert "cannot resolve" not in wide
    assert "WIDER than the box" in wide
    # THE BOX IS THE LARGER OF THE TWO READINGS: a quiet placebo cannot hide a
    # noisy timing, and the printed number says which it took.
    assert "0.94%" in wide

    # A box that really is wider than the band, which is the branch the old
    # sentence asserted unconditionally.
    narrow = DTC.band_consequence([1.016, 1.053], band=0.05, spread=0.02)
    assert "3.7 points wide" in narrow and "5.00%" in narrow
    assert "cannot resolve a band that narrow" in narrow

    # AND THE OLD BAND IS NOT AUTOMATICALLY THE NARROW ONE. Against the
    # committed spread the hardcoded 1.016-1.053 clears the box by 4x, so the
    # retired sentence was false of its own numbers too once the box moved.
    assert "cannot resolve" not in DTC.band_consequence(
        [1.016, 1.053], band=0.0002, spread=0.0094)

    # Neither reading, and nothing may be concluded about the width at all.
    blind = DTC.band_consequence([0.938, 1.099], band=None, spread=None)
    assert "nothing was timed to compare that width against" in blind
    assert DTC.band_consequence([], band=0.01, spread=0.01) == (
        "the model's predicted band over the matched arms is not computable "
        "on this grid, so C3 cannot be read either way")


def test_v5_carries_the_computed_width_into_the_gate(planned, ceilings):
    cells, _ = planned
    results = synth(cells, ceilings)
    analysis = DTC.analyse(cells, results, ceilings, list(DTC.DTYPES))
    v5 = next(g for g in DTC.build_gates(analysis, ceilings, list(DTC.DTYPES),
                                         "", synthetic=False)
              if g.name.startswith("V5"))
    assert "points wide" in v5.invalidates
    assert "1.016 to 1.053" not in v5.invalidates


def test_a_csv_written_under_another_schema_is_refused_not_appended_to(
        tmp_path, planned):
    """2026-09-09 renamed two clock columns and inserted four more, while
    `plan_run_id` still omits `arms` and the results root outlives the pod. So
    the same command lands in the same directory with a wider row, and
    `DictWriter` writes the fieldnames it was given without ever reading the
    file: every field after `sm_clock_load_mhz` would shift one column and
    `clock_level_ok` would hold a clock.
    """
    path = tmp_path / "timings.csv"
    old = list(DTC.CSV_COLUMNS)
    old.remove("clock_samples")
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(old)
        writer.writerow(["x"] * len(old))
    with pytest.raises(DTC.ConfoundRefusal) as exc:
        DTC.Store(path)
    assert f"{len(old)} columns against {len(DTC.CSV_COLUMNS)}" in str(exc.value)
    assert "--fresh" in str(exc.value)

    # --fresh is the documented way out and it must actually work.
    store = DTC.Store(path, fresh=True)
    store.close()
    header = path.read_text().splitlines()[0]
    assert tuple(next(csv.reader([header]))) == DTC.CSV_COLUMNS

    # And a file this schema wrote is still appended to, or every resume dies.
    reopened = DTC.Store(path)
    reopened.close()
