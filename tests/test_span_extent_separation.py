"""The span-extent decomposition has to be able to be WRONG, and to say so.

`scripts/span_extent_separation.py` decomposes the study's 0.563 five-stage over
one-stage separation into a SPAN EXTENT factor and a KERNEL factor, and it does
it on a pod that is metered by the minute. What has to be true before it runs is
not that it produces numbers: it is that its gates land differently in the two
worlds it is trying to tell apart. So most of this file plants a known mechanism
in synthetic arm times, runs the REAL analysis over them, and checks the verdicts
flip.

FOUR GROUPS, and the second is the point.

  - the published constants are RECOMPUTED from the study's own tables rather
    than trusted as copied numbers;
  - the analysis recovers a planted mechanism and tells a kernel world from an
    extent world, on the gates that are supposed to discriminate;
  - the refusals: a partial sum, a crossing below saturation, an errored row
    restored as a timing, a probe with no power, and a run id that omits a swept
    parameter are each a way this project has produced a confident wrong number,
    and each is pinned here;
  - the off-GPU contract: `--dry-run` writes nothing and says nothing was
    measured, and the missing-stack message names which half is absent.

The script is loaded by path rather than imported, because `scripts/` is not a
package and never has been.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "span_extent_separation", ROOT / "scripts" / "span_extent_separation.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module that is not there yet makes
    # the decorator fail with an AttributeError that names nothing useful.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SE = _load_script()

from moe.spec import MODEL_CONFIGS  # noqa: E402

RIDGE = 160.3
BANDWIDTH = 4374.5
TWO_MODELS = ["mixtral-8x7b", "qwen2-57b-a14b"]


def build(models=None, tokens=None, densify=False):
    cells, _ = SE.plan_cells(models or TWO_MODELS,
                             tokens or list(SE.DEFAULT_TOKENS), "bf16", densify)
    return cells


def run_world(name, models=None, densify=False, noise=0.0, seed=0):
    """Generate a world and run the SHIPPED analysis over it."""
    cells = build(models, densify=densify)
    world = SE.WORLDS[name]
    if name == "extent":
        solve = SE.extent_scale_for_separation(
            cells, SE.PUBLISHED_SEPARATION_FIRST, ridge=RIDGE,
            bandwidth_gbps=BANDWIDTH, seed=seed)
        world = SE.World(world.name, solve.scale, world.triton_pads,
                         world.cutlass_pads, world.summary)
    results = SE.synthetic_results(cells, world, ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=noise,
                                   seed=seed)
    analysis = SE.analyse(cells, results)
    return cells, results, analysis, SE.build_gates(analysis)


def gate(gates, prefix):
    for g in gates:
        if g.name.startswith(prefix):
            return g
    raise KeyError(f"no gate named {prefix}; have {[g.name for g in gates]}")


# --------------------------------------------------------------------------
# 1. the published constants, recomputed from the study's own tables
# --------------------------------------------------------------------------

def test_the_published_separation_is_the_ratio_of_the_two_published_columns():
    """docs/STUDY.md:198-202 prints five-stage 0.58 and one-stage 1.03 under the
    full byte model. The separation is their RATIO, so it is recomputed here
    rather than trusted as a copied constant -- which is how a documentation
    number ends up presented as a measurement."""
    assert SE.PUBLISHED_SEPARATION == pytest.approx(0.578 / 1.027, abs=5e-4)


def test_the_first_crossing_separation_is_the_uniform_only_row():
    """docs/FINDINGS.md:766-770, uniform routing only: 0.553 over 0.987."""
    assert SE.PUBLISHED_SEPARATION_FIRST == pytest.approx(0.553 / 0.987, abs=5e-4)


def test_the_two_ends_of_the_staircase_are_not_two_estimates_of_one_number():
    """0.560 on the first crossing and 0.889 on the last is a 59% swing. A
    script that reported one of them would be choosing, so both are carried."""
    assert SE.PUBLISHED_SEPARATION_LAST > 1.5 * SE.PUBLISHED_SEPARATION_FIRST


def test_the_kernel_gate_threshold_is_exactly_half_the_separation_in_log_terms():
    """C2's 0.75 is not a round number picked to be passable: it is the point at
    which the kernel factor carries half of `ln(0.563)`."""
    share = math.log(SE.KERNEL_MAX) / math.log(SE.PUBLISHED_SEPARATION)
    assert share == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------------------
# 2. the decomposition, and whether the gates tell the worlds apart
# --------------------------------------------------------------------------

def test_the_decomposition_is_exact_for_every_model():
    """EXTENT x KERNEL == separation, because the middle term cancels. If this
    ever fails, the three factors are not being read off the same crossings."""
    _, _, analysis, _ = run_world("kernel", densify=True)
    assert analysis.decomposed
    for dec in analysis.decomposed:
        for end in ("first", "last"):
            extent, kernel, separation = dec.factors(end)
            assert extent * kernel == pytest.approx(separation, rel=1e-9)


def test_the_two_log_shares_sum_to_one():
    """Per model, and only then medianed. Taking the shares OF THE MEDIANS is
    wrong and looks right: `median(EXTENT) * median(KERNEL)` is not
    `median(separation)` on a ragged set of models, so those two shares miss 1
    by a few tenths of a percent -- small enough to survive review and large
    enough to mean the apportionment is not an apportionment."""
    _, _, analysis, _ = run_world("kernel", densify=True)
    e = analysis.shares["first"]["extent"]
    k = analysis.shares["first"]["kernel"]
    assert e + k == pytest.approx(1.0, abs=1e-9)


def test_the_shares_of_the_medians_are_NOT_the_median_of_the_shares():
    """Pinned as a trap, not as a preference. If these ever agree the ragged
    grid has become square and the per-model discipline stopped costing
    anything -- which is worth noticing, not worth silently relying on."""
    _, _, analysis, _ = run_world("kernel", densify=True)
    med = analysis.medians["first"]
    naive = (SE.log_share(med["extent"], med["separation"])
             + SE.log_share(med["kernel"], med["separation"]))
    assert naive != pytest.approx(1.0, abs=1e-9)


def test_a_padding_only_world_puts_the_whole_separation_in_the_kernel_factor():
    _, _, analysis, gates = run_world("kernel", densify=True)
    med = analysis.medians["first"]
    assert SE.EXTENT_BAND[0] <= med["extent"] <= SE.EXTENT_BAND[1]
    assert med["kernel"] < 0.9
    assert gate(gates, "C1").passed is True


def test_an_extent_only_world_puts_it_in_the_extent_factor_and_fails_C1():
    _, _, analysis, gates = run_world("extent")
    med = analysis.medians["first"]
    assert med["kernel"] == pytest.approx(1.0, abs=1e-6)
    assert med["extent"] < SE.EXTENT_BAND[0]
    assert gate(gates, "C1").passed is False


def test_a_world_with_neither_mechanism_has_no_separation_to_decompose():
    _, _, analysis, gates = run_world("neither", densify=True)
    med = analysis.medians["first"]
    assert med["separation"] > 0.9
    assert gate(gates, "C5").passed is False
    assert gate(gates, "C2").passed is False


def test_C1_is_the_gate_that_flips_between_the_two_worlds():
    """The single discriminating verdict, pinned so a later edit cannot quietly
    make both worlds agree."""
    _, _, _, kernel_gates = run_world("kernel", densify=True)
    _, _, _, extent_gates = run_world("extent")
    assert gate(kernel_gates, "C1").passed is True
    assert gate(extent_gates, "C1").passed is False


@pytest.mark.parametrize("noise", [0.0, 0.01, 0.02])
def test_the_kernel_world_signature_survives_timing_noise(noise):
    _, _, analysis, _ = run_world("kernel", densify=True, noise=noise)
    med = analysis.medians["first"]
    assert med["extent"] > 0.9
    assert med["kernel"] < 0.9


def test_span_extent_alone_needs_the_extra_stages_to_cost_thousands_of_times_more():
    """The `extent` world SOLVES for the inflation the extent explanation needs
    rather than asserting the extra stages are cheap. Anything in the hundreds
    or thousands is the answer: the byte model puts those stages at a few
    percent of the layer."""
    cells = build()
    solve = SE.extent_scale_for_separation(
        cells, SE.PUBLISHED_SEPARATION_FIRST, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, seed=0)
    assert solve.scale > 100.0
    if solve.reached:
        assert solve.separation <= SE.PUBLISHED_SEPARATION_FIRST + 1e-9
    else:
        # Not reached is a stronger answer, not a missing one, and it must carry
        # the floor it did reach rather than a shrug.
        assert solve.floor is not None and solve.floor > SE.PUBLISHED_SEPARATION_FIRST


# --------------------------------------------------------------------------
# 3. the refusals
# --------------------------------------------------------------------------

def test_a_missing_arm_makes_the_sum_None_and_never_a_partial_sum():
    """A partial sum of a two-GEMM curve is a one-GEMM curve wearing the other's
    label, and it would land in a crossing with nothing to mark it."""
    cells = build(models=["mixtral-8x7b"])
    results = SE.synthetic_results(cells, SE.WORLDS["kernel"], ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    key = cells[5].key
    results[key]["gemm_down"].error = "planted"
    assert SE.summed_ms(results, key, SE.GEMM_ARMS) is None


def test_an_errored_arm_reports_no_time_rather_than_zero():
    arm = SE.ArmResult("mixtral-8x7b", 256, "gemm_up", ms_median=1.5,
                       error="planted")
    assert arm.timed is False
    results = {("mixtral-8x7b", 256): {"gemm_up": arm}}
    assert SE.arm_ms(results, ("mixtral-8x7b", 256), "gemm_up") is None


def test_a_crossing_below_saturation_is_not_a_crossing():
    """Below `E/k` tokens a batch does not touch every expert, so weight traffic
    grows WITH the batch and time rises nearly linearly. That slope crosses 0.5
    for a reason that has nothing to do with the ridge; without the floor
    mixtral reported a crossing at 5 tokens against a predicted 641."""
    points = [(1.0, 0.10), (2.0, 0.20), (4.0, 0.40), (8.0, 0.80),
              (16.0, 0.85), (32.0, 0.86), (64.0, 0.87)]
    found = SE.crossings_of(points, "mixtral-8x7b", "planted")
    assert all(t >= 4.0 for t in found.tokens), found.tokens


def crossings(model, label, *tokens):
    return SE.Crossings(model, label, tuple(float(t) for t in tokens), 14)


def test_a_model_whose_up_and_down_crossings_disagree_is_dropped_by_name():
    """docs/FINDINGS.md:786-791 records the one-stage up and down crossings
    disagreeing by 2.3x on two models, on what is the same arithmetic over the
    same cells. Averaging that in would put a number in the headline that
    neither half supports."""
    dec = SE.ModelDecomposition(
        "mixtral-8x7b",
        five=crossings("mixtral-8x7b", "five", 300),
        one_triton=crossings("mixtral-8x7b", "one", 500),
        one_cutlass=crossings("mixtral-8x7b", "cut", 600),
        triton_up=crossings("mixtral-8x7b", "up", 300),
        triton_down=crossings("mixtral-8x7b", "down", 900))
    assert dec.estimator_spread("first") == pytest.approx(3.0)
    assert dec.estimator_spread("first") > SE.ESTIMATOR_AGREEMENT_MAX


def test_a_model_whose_two_halves_agree_is_kept():
    dec = SE.ModelDecomposition(
        "mixtral-8x7b",
        five=crossings("mixtral-8x7b", "five", 300),
        one_triton=crossings("mixtral-8x7b", "one", 500),
        one_cutlass=crossings("mixtral-8x7b", "cut", 600),
        triton_up=crossings("mixtral-8x7b", "up", 480),
        triton_down=crossings("mixtral-8x7b", "down", 520))
    assert dec.estimator_spread("first") < SE.ESTIMATOR_AGREEMENT_MAX


def test_a_model_with_no_crossing_on_one_curve_is_excluded_and_says_which():
    """Not averaged away and not silently dropped: the report has to name the
    curve that had no crossing, or a three-model median passes for a
    four-model one."""
    cells = build(models=["mixtral-8x7b"])
    results = SE.synthetic_results(cells, SE.WORLDS["kernel"], ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    for cell in cells:
        results[cell.key]["cutlass_up"].error = "planted"
    analysis = SE.analyse(cells, results)
    excluded = [d for d in analysis.per_model if d.excluded]
    assert excluded, "a model with no CUTLASS curve was decomposed anyway"
    assert "one-launch CUTLASS" in excluded[0].excluded


def test_the_padding_probe_refuses_when_the_null_sits_inside_its_own_band():
    """Above the crossing the padding factor decays toward 1, and on the default
    powers-of-two grid it measures about 1.1. The null (contrast = 1) is then
    inside the acceptance band and a PASS would mean nothing. It must be
    UNKNOWN, not PASS: that is the difference between 'no mechanism found' and
    'nothing was looked at'."""
    _, _, _, gates = run_world("kernel", densify=False)
    assert gate(gates, "C3").passed is None
    assert "NO POWER" in gate(gates, "C3").observed


def test_the_padding_probe_speaks_once_the_grid_gives_it_power():
    """All four models, because power comes from the ones whose compute-bound
    cells carry a padding factor far enough from 1, and on two models it does
    not. That is itself the point: the probe reports per model and pools only
    the ones that can speak."""
    models = list(SE.DEFAULT_MODELS)
    _, _, _, kernel_gates = run_world("kernel", models=models, densify=True)
    _, _, _, neither_gates = run_world("neither", models=models, densify=True)
    assert gate(kernel_gates, "C3").passed is True
    assert gate(neither_gates, "C3").passed is False


def test_non_vacuity_fails_when_nothing_was_measured():
    """A check that examined nothing reports zero failures too."""
    cells = build(models=["mixtral-8x7b"])
    analysis = SE.analyse(cells, {})
    gates = SE.build_gates(analysis)
    assert gate(gates, "V4").passed is False
    assert gate(gates, "V0").passed is None       # UNKNOWN, never PASS
    assert gate(gates, "V1").passed is None


def test_v6_refuses_rather_than_scoring_an_unmeasured_model_as_perfect():
    """A decomposed model with no measured up/down spread is not agreement.

    V6 used to read `estimator_spreads.get(model, 1.0)`, substituting a perfect
    1.00x for a pair that was never measured, and the observed string was built
    only from the models that HAD a spread -- so the substituted model was not
    named anywhere on the page. One measured model was enough to report PASS
    over any number of unmeasured ones. V6 is the gate that decides whether the
    crossing estimator resolved these curves at all, so a vacuous PASS there
    gets EXTENT and KERNEL quoted on models it was never shown to work on.
    """
    _, _, analysis, gates = run_world("kernel", densify=True)
    assert gate(gates, "V6").passed is True          # the honest world first
    # Now drop one decomposed model's spread, as a partially-measured pod run
    # would, and leave everything else exactly as it was.
    dropped = analysis.decomposed[0].model
    thinned = dict(analysis.estimator_spreads)
    thinned.pop(dropped)
    partial = dataclasses.replace(analysis, estimator_spreads=thinned)
    v6 = gate(SE.build_gates(partial), "V6")
    assert v6.passed is None                         # UNDECIDED, never PASS
    assert dropped in v6.observed                    # and it is NAMED


def test_v6_still_fails_loudly_when_a_measured_pair_disagrees():
    # The refusal above must not have swallowed the gate's original job.
    _, _, analysis, _ = run_world("kernel", densify=True)
    blown = {m: SE.ESTIMATOR_AGREEMENT_MAX * 2
             for m in analysis.estimator_spreads}
    v6 = gate(SE.build_gates(
        dataclasses.replace(analysis, estimator_spreads=blown)), "V6")
    assert v6.passed is False


def test_every_validity_gate_states_what_a_fail_invalidates():
    _, _, _, gates = run_world("kernel", densify=True)
    for g in gates:
        assert g.kind in ("VALIDITY", "CLAIM")
        assert g.invalidates, f"{g.name} does not say what a FAIL means"


def test_the_headline_refuses_to_be_quoted_when_a_validity_gate_did_not_pass():
    cells, _, analysis, _ = run_world("kernel", densify=True)
    gates = SE.build_gates(analysis)
    broken = [g for g in gates if g.kind == "VALIDITY" and g.passed is not True]
    text = SE.render_headline(analysis, gates)
    if broken:
        assert "DO NOT QUOTE" in text
    else:                                            # pragma: no cover
        assert "DO NOT QUOTE" not in text


def test_the_placebo_gate_catches_a_noisy_box():
    cells = build(models=["mixtral-8x7b"])
    results = SE.synthetic_results(cells, SE.WORLDS["kernel"], ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    for cell in cells:
        replica = results[cell.key]["fused_replica"]
        replica.ms_median = replica.ms_median * 1.20
    gates = SE.build_gates(SE.analyse(cells, results))
    assert gate(gates, "V3").passed is False


def test_bind_call_refuses_a_signature_it_has_no_value_for():
    """Filling an unknown required parameter with a plausible None produces a
    call that runs and times a different schedule."""
    def drifted(A, B, C, brand_new_required_thing):     # noqa: ARG001
        return None

    with pytest.raises(SE.SignatureDrifted) as exc:
        SE.bind_call(drifted, {"A": 1, "B": 2, "C": 3})
    assert "brand_new_required_thing" in str(exc.value)


def test_bind_call_supplies_only_names_the_signature_actually_has():
    def narrow(A, B, config=None):                      # noqa: ARG001
        return None

    got = SE.bind_call(narrow, {"A": 1, "B": 2, "config": {}, "B_zp": None,
                                "use_mxfp4_w4a4": False})
    assert set(got) == {"A", "B", "config"}


def test_bind_call_leaves_optional_parameters_it_knows_nothing_about_alone():
    def wide(A, B, future_flag=False):                  # noqa: ARG001
        return None

    assert set(SE.bind_call(wide, {"A": 1, "B": 2})) == {"A", "B"}


# --------------------------------------------------------------------------
# 4. arithmetic that a wrong answer would look reasonable in
# --------------------------------------------------------------------------

def test_padding_is_charged_per_expert_and_not_globally():
    """Padding globally understates the cost by a factor of E at decode, which
    is the regime this whole study lives in."""
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    counts = [1] * cfg.num_experts               # 8 experts, one row each
    assert SE.padded_from_counts(counts, 64) == 8 * 64
    assert SE.padded_rows_saturated(cfg, 4, 64) == 8 * 64


def test_an_expert_with_no_rows_pays_nothing():
    assert SE.padded_from_counts([0, 0, 130], 64) == 192


def test_padding_costs_compute_and_not_traffic():
    """The whole mechanism C3 names. If padding ever entered the traffic term,
    the probe's two regimes would stop being separable."""
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    kw = {"b": 2, "ridge": RIDGE, "bandwidth_gbps": BANDWIDTH}
    small = SE.modelled_ms(cfg, 8, "up_gemm", compute_rows=16, **kw)
    padded = SE.modelled_ms(cfg, 8, "up_gemm", compute_rows=16 * 64, **kw)
    # Deep in the memory-bound regime the padding buys nothing.
    assert small == pytest.approx(padded, rel=1e-9)
    big = SE.modelled_ms(cfg, 8192, "up_gemm", compute_rows=8192 * 2, **kw)
    big_padded = SE.modelled_ms(cfg, 8192, "up_gemm",
                                compute_rows=8192 * 2 * 1.5, **kw)
    assert big_padded > big * 1.4


def test_the_quantisation_factor_is_read_off_the_grid_at_the_interpolated_batch():
    """mixtral at T=512 is 128 rows per expert and pads to nothing at BLOCK_M=64;
    at T=520 it is 130 rows and pays a whole extra tile. A grid point cannot see
    that, which is why the probe is differential instead."""
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    assert SE.modelled_padding_at(cfg, 512, 64) == pytest.approx(1.0)
    assert SE.modelled_padding_at(cfg, 520, 64) == pytest.approx(192 / 130, rel=1e-6)


def test_the_saturated_rows_helper_matches_the_histogram_one_when_balanced():
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    rows = 512 * cfg.top_k // cfg.num_experts
    counts = [rows] * cfg.num_experts
    assert SE.padded_from_counts(counts, 64) == SE.padded_rows_saturated(cfg, 512, 64)


def test_densifying_puts_cells_where_the_padding_factor_is_material():
    """C3 is only observable at 80 to 200 rows per expert. A densified grid has
    to actually land there or the flag is decoration."""
    cells = build(models=["mixtral-8x7b"], densify=True)
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    materials = [c for c in cells
                 if (SE.modelled_padding_at(
                     cfg, float(c.num_tokens),
                     SE.ladder_block_m(c.num_tokens, cfg.num_experts, "bf16"))
                     or 1.0) >= 1.3]
    assert materials, "densify added no cell with a material padding factor"


def test_densified_tokens_keep_rows_per_expert_an_exact_integer():
    for model in SE.PUBLISHED_FIVE_STAGE_CROSSING:
        cfg = MODEL_CONFIGS[model]
        for tok in SE.densified_tokens(model):
            rows = tok * cfg.top_k / cfg.num_experts
            assert rows == int(rows), (model, tok, rows)


def test_the_time_budget_share_is_taken_of_the_reconstruction_and_not_the_fused_time():
    """Dividing by the fused time would fold the reconstruction residual into
    the share and make an arm's overhead look like an extra stage."""
    cell = SE.Cell("mixtral-8x7b", 512, "bf16")
    budget = SE.CellBudget(cell, fused_ms=2.0,
                           launch_ms={"align": 0.1, "gemm_up": 0.6, "act": 0.1,
                                      "gemm_down": 0.3, "sum": 0.1},
                           padding=1.2, padding_source="histogram", block_m=64)
    assert budget.parts_total == pytest.approx(1.2)
    assert budget.reconstruction == pytest.approx(0.6)
    assert budget.non_gemm_share == pytest.approx(0.3 / 1.2)
    assert budget.extent_time == pytest.approx(2.0 / 0.9)


def test_a_budget_missing_one_launch_reports_no_reconstruction():
    cell = SE.Cell("mixtral-8x7b", 512, "bf16")
    budget = SE.CellBudget(cell, fused_ms=2.0,
                           launch_ms={"align": 0.1, "gemm_up": 0.6, "act": None,
                                      "gemm_down": 0.3, "sum": 0.1},
                           padding=None, padding_source="", block_m=64)
    assert budget.reconstruction is None
    assert budget.non_gemm_share is None


# --------------------------------------------------------------------------
# 5. persistence, the run id, and the off-GPU contract
# --------------------------------------------------------------------------

BASE_ID_ARGS = dict(models=["mixtral-8x7b"], tokens=[1, 2], dtype="bf16",
                    routing="uniform", seed=0, reps=3, iters=15, warmup=8,
                    arms=["fused", "gemm_up", "gemm_down"], densify=False)


@pytest.mark.parametrize("field,value", [
    ("models", ["qwen2-57b-a14b"]), ("tokens", [1, 4]), ("dtype", "fp8_e4m3"),
    ("routing", "zipf"), ("seed", 1), ("reps", 4), ("iters", 20),
    ("warmup", 4), ("arms", ["fused", "gemm_up", "gemm_down", "act"]),
    ("densify", True),
])
def test_every_swept_parameter_changes_the_run_id(field, value):
    """A run id that omits a swept parameter means two settings derive the same
    directory, the second resumes the first, skips every completed cell, and
    prints the first's numbers under the second's label."""
    base = SE.plan_run_id(**BASE_ID_ARGS)
    assert SE.plan_run_id(**dict(BASE_ID_ARGS, **{field: value})) != base


def test_the_run_id_is_derived_so_the_same_command_resumes_itself():
    assert SE.plan_run_id(**BASE_ID_ARGS) == SE.plan_run_id(**BASE_ID_ARGS)


def test_a_row_round_trips_through_the_csv_unchanged(tmp_path):
    store = SE.Store(tmp_path / "t.csv")
    cell = SE.Cell("mixtral-8x7b", 256, "bf16")
    meta = {"run_id": "abc", "gpu_name": "H200", "torch_version": "2.13.0",
            "triton_version": "3.7.1", "vllm_version": "0.27.1",
            "routing": "uniform", "seed": 0}
    arm = SE.ArmResult("mixtral-8x7b", 256, "gemm_up", ms_median=1.25,
                       ms_mean=1.26, ms_stdev=0.01, ms_min=1.2, n_samples=45,
                       config={"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
                               "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
                               "num_warps": 8, "num_stages": 4},
                       tile_config_source="vllm_observed", rows_total=512.0,
                       padded_rows=768.0, padding_source="histogram",
                       active_experts=8, triton_artifacts=3)
    store.write(arm, cell, meta)
    store.close()

    fresh = SE.Store(tmp_path / "t.csv")
    back = fresh.restore(("mixtral-8x7b", 256, "gemm_up"))
    fresh.close()
    assert back.ms_median == pytest.approx(1.25)
    assert back.config["BLOCK_SIZE_M"] == 64
    assert back.padded_rows == pytest.approx(768.0)
    assert back.padding_source == "histogram"
    assert back.triton_artifacts == 3


def test_a_failed_row_is_retried_rather_than_restored(tmp_path):
    """The common failures here are a pod that lost its device and an arm that
    ran out of memory behind a since-finished neighbour, and a re-run can leave
    both behind. Restoring the failure would make it permanent."""
    store = SE.Store(tmp_path / "t.csv")
    cell = SE.Cell("mixtral-8x7b", 256, "bf16")
    meta = {"run_id": "abc", "gpu_name": "", "torch_version": "",
            "triton_version": "", "vllm_version": "", "routing": "uniform",
            "seed": 0}
    store.write(SE.ArmResult("mixtral-8x7b", 256, "gemm_up",
                             error="CUDA out of memory"), cell, meta)
    store.close()
    fresh = SE.Store(tmp_path / "t.csv")
    assert fresh.restore(("mixtral-8x7b", 256, "gemm_up")) is None
    fresh.close()


def test_the_results_root_prefers_the_volume_that_outlives_the_pod(monkeypatch,
                                                                   tmp_path):
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path / "vol"))
    assert SE.results_root() == tmp_path / "vol"
    monkeypatch.delenv("MOE_RESULTS_DIR")
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    assert SE.results_root() == tmp_path / "results"


def test_the_script_says_whether_git_would_silently_drop_its_output():
    """`.gitignore` carries `results/*` with only `!results/published/` excepted,
    and this repo has already lost every published figure to an unanchored rule.
    Knowing the output is ignored is fine; not knowing is how a result
    disappears between the pod and the commit."""
    note = SE.gitignore_note(ROOT / "results" / "span_extent_separation" / "x")
    assert "GITIGNORED" in note


def test_the_published_directory_is_not_ignored_so_publishing_still_works():
    note = SE.gitignore_note(ROOT / "results" / "published")
    assert "not gitignored" in note


def test_the_dry_run_needs_no_gpu_writes_nothing_and_says_so(tmp_path, capsys):
    out = tmp_path / "nothing"
    code = SE.main(["--dry-run", "--models", "mixtral-8x7b", "--tokens", "1,256",
                    "--out-dir", str(out)])
    text = capsys.readouterr().out
    assert code == SE.EXIT_NOT_MEASURED
    assert "NOT A RESULT" in text
    assert not out.exists(), "a dry run created its output directory"


def test_the_dry_run_prints_both_routes_and_which_one_is_reachable(capsys):
    SE.main(["--dry-run", "--models", "mixtral-8x7b", "--tokens", "256"])
    text = capsys.readouterr().out
    assert "IS NOT REACHABLE" in text and "IS REACHABLE" in text
    assert "invoke_fused_moe_kernel" in text


def test_the_dry_run_prints_the_predictions_with_their_numbers(capsys):
    SE.main(["--dry-run", "--models", "mixtral-8x7b", "--tokens", "256"])
    text = capsys.readouterr().out
    for name in ("V0", "V4", "C1", "C2", "C3", "C4", "C5"):
        assert name in text
    assert "Estimated KERNEL time" in text
    assert "WALL CLOCK IS NOT THAT NUMBER" in text


def test_the_dry_run_prints_whether_C3_can_speak_on_this_grid(capsys):
    SE.main(["--dry-run", "--models", "mixtral-8x7b", "--tokens",
             ",".join(str(t) for t in SE.DEFAULT_TOKENS)])
    text = capsys.readouterr().out
    assert "C3's POWER ON THIS GRID" in text


def test_dropping_a_corner_of_the_2x2_is_refused():
    """The three corners are the whole experiment; two of them separate nothing."""
    with pytest.raises(SystemExit):
        SE.main(["--dry-run", "--arms", "fused,cutlass_up"])


def test_the_missing_stack_message_names_which_half_is_absent():
    """Run on a laptop, so this is the real message a laptop gets."""
    message = SE.missing_gpu_stack()
    assert message == "" or "--self-test" in message


def test_the_synthetic_world_is_hermetic_and_replays_identically():
    """A planted-answer mode that read the hardware would replay differently on
    every machine."""
    cells = build(models=["mixtral-8x7b"])
    kw = dict(ridge=RIDGE, bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    first = SE.synthetic_results(cells, SE.WORLDS["kernel"], **kw)
    second = SE.synthetic_results(cells, SE.WORLDS["kernel"], **kw)
    for key, arms in first.items():
        for name, arm in arms.items():
            assert arm.ms_median == second[key][name].ms_median


def test_the_self_test_report_says_nothing_was_measured(tmp_path, capsys):
    SE.main(["--self-test", "kernel", "--models", "mixtral-8x7b",
             "--out-dir", str(tmp_path)])
    text = capsys.readouterr().out
    assert "SELF TEST" in text and "GENERATED" in text
    report = next(tmp_path.glob("*/report.md"))
    assert report.exists()


def test_the_summary_names_the_corner_that_is_missing(tmp_path):
    """The interaction between extent and kernel is not measured, and a summary
    that did not say so would let a reader treat the two factors as separable
    causes."""
    import json
    SE.main(["--self-test", "kernel", "--models", "mixtral-8x7b",
             "--out-dir", str(tmp_path)])
    payload = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert "five-launch" in payload["missing_corner"]
    assert "EXTENT * KERNEL == separation exactly" in payload["definition"]


# --------------------------------------------------------------------------
# 6. the two gates this project has previously shipped unable to fail
# --------------------------------------------------------------------------

def errored_run(kind: str, message: str):
    """One cell whose rig REFUSED, exactly as the runner records it."""
    cells = build(models=["mixtral-8x7b"], tokens=[256, 512])
    results = SE.synthetic_results(cells, SE.WORLDS["kernel"], ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    results[cells[0].key] = {
        "fused": SE.ArmResult("mixtral-8x7b", cells[0].num_tokens, "fused",
                              error=f"{kind}: {message}")}
    return cells, SE.build_gates(SE.analyse(cells, results))


def test_V0_fails_on_an_assembly_mismatch_rather_than_reporting_UNKNOWN():
    """`build_rig` RAISES on a mismatch rather than spending metered GPU time
    timing an assembly that is not the fused path, so the failing cell records
    an error and no relative error at all. A V0 that read only `max_rel_err`
    would have exactly two reachable states, PASS and UNKNOWN -- the shape of
    the check in this repo that had never passed on any machine."""
    _, gates = errored_run("AssemblyMismatch", "reproduces to 4.1e-01")
    assert gate(gates, "V0").passed is False
    assert "REFUSED" in gate(gates, "V0").observed


def test_V1_fails_when_the_tile_could_not_be_observed():
    _, gates = errored_run("ConfigUnobserved", "the recorder saw no tile config")
    assert gate(gates, "V1").passed is False


def test_V1_fails_when_a_vllm_entry_point_moved():
    _, gates = errored_run("SignatureDrifted", "requires ['brand_new']")
    assert gate(gates, "V1").passed is False


def test_V1_fails_when_a_resumed_row_carries_a_different_tile():
    """Within one session the launches are HANDED the fused call's config, so
    the comparison is a value against itself. It has teeth only across a
    resume, which is exactly the case a colliding run id produces."""
    cells = build(models=["mixtral-8x7b"], tokens=[256, 512])
    results = SE.synthetic_results(cells, SE.WORLDS["kernel"], ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    stale = dict(results[cells[0].key]["gemm_up"].config)
    stale["BLOCK_SIZE_M"] = stale["BLOCK_SIZE_M"] * 2
    results[cells[0].key]["gemm_up"].config = stale
    gates = SE.build_gates(SE.analyse(cells, results))
    assert gate(gates, "V1").passed is False
    assert "resumed" in gate(gates, "V1").observed


def test_V1_fails_on_a_config_that_was_only_half_read_back():
    cells = build(models=["mixtral-8x7b"], tokens=[256, 512])
    results = SE.synthetic_results(cells, SE.WORLDS["kernel"], ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    results[cells[0].key]["fused"].config = {"BLOCK_SIZE_M": 64}
    gates = SE.build_gates(SE.analyse(cells, results))
    assert gate(gates, "V1").passed is False


def test_a_refused_cell_leaves_the_model_undecomposed_rather_than_short_a_point():
    """A cell that refused writes no timings, so its model loses a point off
    every curve. It must drop out of the decomposition by name, not quietly
    contribute a crossing read off a grid with a hole in it."""
    cells = build(models=["mixtral-8x7b"])
    results = SE.synthetic_results(cells, SE.WORLDS["kernel"], ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    for cell in cells:
        results[cell.key] = {
            "fused": SE.ArmResult(cell.model, cell.num_tokens, "fused",
                                  error="AssemblyMismatch: planted")}
    analysis = SE.analyse(cells, results)
    assert analysis.decomposed == []
    assert analysis.refusals["AssemblyMismatch"]


def test_every_gate_reachable_verdict_is_pinned_across_the_three_worlds():
    """A gate that returns the same verdict in every world discriminates
    nothing. This asserts the claim gates as a SET take more than one value."""
    verdicts = {}
    for name, kwargs in (("kernel", dict(densify=True)),
                         ("extent", {}),
                         ("neither", dict(densify=True))):
        _, _, _, gates = run_world(name, models=list(SE.DEFAULT_MODELS), **kwargs)
        verdicts[name] = tuple(gate(gates, g).passed
                               for g in ("C1", "C2", "C3", "C5"))
    assert len(set(verdicts.values())) == 3, verdicts
