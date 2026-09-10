"""The span-extent decomposition has to be able to be WRONG, and to say so.

`scripts/span_extent_separation.py` decomposes the study's 0.563 five-stage over
one-stage separation into a SPAN EXTENT factor and a KERNEL factor, and it does
it on a pod that is metered by the minute. What has to be true before it runs is
not that it produces numbers: it is that its gates land differently in the two
worlds it is trying to tell apart. So most of this file plants a known mechanism
in synthetic arm times, runs the REAL analysis over them, and checks the verdicts
flip.

NINE GROUPS, and the second is the point.

  - the published constants are RECOMPUTED from the study's own tables rather
    than trusted as copied numbers;
  - the analysis recovers a planted mechanism and tells a kernel world from an
    extent world, on the gates that are supposed to discriminate;
  - the refusals: a partial sum, a crossing below saturation, an errored row
    restored as a timing, a probe with no power, and a run id that omits a swept
    parameter are each a way this project has produced a confident wrong number,
    and each is pinned here;
  - the off-GPU contract: `--dry-run` writes nothing and says nothing was
    measured, and the missing-stack message names which half is absent;
  - the RESUME KEY: a timings file planted on one card is restored by NONE of it
    on another, and the plan says so before anything is measured. The run id was
    card-free and `restore` ignored `gpu_name`, so a second pod resuming through
    the shared network volume measured zero cells and published the first card's
    timings. The SELF-TEST WORLD is in that key too, because a self test and a
    measured run detect the same card and three worlds were writing one
    report.md; and a run that restored every arm and timed none says so in one
    line, because the session driver books two span arms that derive one id;
  - the SELF TEST THAT ASSERTS SOMETHING: every world reproduces the verdict row
    registered for it, an S gate FAILS when one does not, a gate added without
    an expectation is caught, a gate REMOVED without editing the table is caught
    and its S gate is UNKNOWN rather than a quiet pass, and the registered rows
    differ from each other -- planted, by making two worlds register one row,
    rather than recomputed from the gate's own PASS condition. On
    the grid the driver used to run first, the world where the claim is TRUE
    could not pass C2 and produced the same claim-gate row as the world with no
    mechanism at all, which is why `--densify` is now the default and a grid
    that cannot answer REFUSES before measuring;
  - the EXIT-CODE CONTRACT: one RESULT line per scored gate and no other line
    that looks like one, a VALIDITY failure is INVALID and a CLAIM failure is
    CLAIM_FAIL, UNKNOWN counts against the gate, and every refusal is REFUSED;
  - ONE INSTRUMENT: the private `time_calls` copy is gone, `time_arm` hands
    `timing.time_kernel` the knobs from argv, the KernelTiming columns round
    trip through the CSV with their three states intact, and V7 FAILS when two
    RULERS appear in one run -- two instrument strings, or one instrument with
    the L2 flush on for some rows and off for others, which the basis string
    cannot tell apart. These rows are unflushed by design and are published as
    NOT roof-comparable. Every synthetic world fails V7, which is what stops a
    laptop self test being quotable;
  - the STATISTICS a reader needs beside the numbers: the interval's scope and
    its n, degenerate at the n=2 the gates permit, an MDE derived from a stated
    noise assumption rather than discovered after the fact, and BOTH sides of
    the instrument's iteration clamp -- read off `iters_for` itself, because two
    docstrings here restated it as [10, 10000] against a real `hi` of 2000 and
    therefore reported the third of the grid that underruns its budget as
    nobody.

The script is loaded by path rather than imported, because `scripts/` is not a
package and never has been.
"""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import inspect
import json
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

from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
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

BASE_ID_ARGS = dict(card="NVIDIA H200", models=["mixtral-8x7b"], tokens=[1, 2],
                    dtype="bf16", routing="uniform", seed=0, reps=3,
                    target_ms=200.0, warmup_ms=200.0, trials=3, l2_flush=False,
                    arms=["fused", "gemm_up", "gemm_down"], densify=False,
                    self_test=None, self_test_noise=0.0)


@pytest.mark.parametrize("field,value", [
    ("card", "NVIDIA A100-SXM4-80GB"),
    ("models", ["qwen2-57b-a14b"]), ("tokens", [1, 4]), ("dtype", "fp8_e4m3"),
    ("routing", "zipf"), ("seed", 1), ("reps", 4), ("target_ms", 400.0),
    ("warmup_ms", 50.0), ("trials", 5), ("l2_flush", True),
    ("arms", ["fused", "gemm_up", "gemm_down", "act"]),
    ("densify", True),
    # The two the audit's own off-GPU check line moved through. A self test and
    # a measured run detect the same card, so without these the runbook's
    # `--self-test kernel` would have written its generated report.md and
    # summary.json OVER a measured run's, in that run's directory.
    ("self_test", "kernel"),
    ("self_test_noise", 0.05),
])
def test_every_swept_parameter_changes_the_run_id(field, value):
    """A run id that omits a swept parameter means two settings derive the same
    directory, the second resumes the first, skips every completed cell, and
    prints the first's numbers under the second's label.

    THE CARD IS THE ROW THAT MATTERS. It is not swept by the script, it is swept
    by the operator moving to another pod, and `results_root()` prefers the
    network volume BECAUSE it outlives the pod. Two cards therefore derived one
    id, and the audit found two published directories holding the same report
    filename for two different `sm_count`s."""
    base = SE.plan_run_id(**BASE_ID_ARGS)
    assert SE.plan_run_id(**dict(BASE_ID_ARGS, **{field: value})) != base


def test_the_run_id_is_derived_so_the_same_command_resumes_itself():
    assert SE.plan_run_id(**BASE_ID_ARGS) == SE.plan_run_id(**BASE_ID_ARGS)


def test_the_card_is_the_front_of_the_run_id_so_ls_shows_it():
    """A hash nobody can invert is not an attribution. `provenance.run_id` puts
    the card slug first for exactly this reason."""
    assert SE.plan_run_id(**BASE_ID_ARGS).startswith("nvidia_h200-")


def test_a_run_id_without_a_card_is_refused_rather_than_defaulted():
    """`provenance.run_id` raises `NoCard` rather than naming a run after
    nothing. The laptop path supplies `NO_CARD` explicitly, which is a name."""
    with pytest.raises(PV.NoCard):
        SE.plan_run_id(**dict(BASE_ID_ARGS, card=""))
    assert SE.plan_run_id(**dict(BASE_ID_ARGS, card=SE.NO_CARD)).startswith(
        "nocard-")


def test_the_default_sets_are_named_in_the_id_and_any_other_set_is_spelled_out():
    """`named_or_listed` shortens the VISIBLE part only. Two different sets must
    still never derive one id, which is what the second assertion pins."""
    assert SE.named_or_listed(list(SE.DEFAULT_MODELS), SE.DEFAULT_MODELS,
                              "published4") == "published4"
    assert SE.named_or_listed(["mixtral-8x7b"], SE.DEFAULT_MODELS,
                              "published4") == ("mixtral-8x7b",)
    ids = {SE.plan_run_id(**dict(BASE_ID_ARGS, models=list(m)))
           for m in (SE.DEFAULT_MODELS, ("mixtral-8x7b",),
                     ("mixtral-8x7b", "qwen2-57b-a14b"))}
    assert len(ids) == 3


def test_the_three_worlds_and_a_measured_run_get_four_directories():
    """The failure this pins is the one the audit's own off-GPU check line would
    have caused. A self test and a measured run detect the SAME card, so with
    `--self-test` out of the key the runbook's `--self-test kernel` line writes a
    GENERATED report.md and summary.json into a measured run's directory, over
    the real ones, carrying the real run's provenance block."""
    ids = {world: SE.plan_run_id(**dict(BASE_ID_ARGS, self_test=world))
           for world in (None, *SE.WORLD_EXPECTATIONS)}
    assert len(set(ids.values())) == len(ids)
    # And the world survives into the VISIBLE part, not only the hash: a
    # directory whose world can be recovered only by opening summary.json is
    # what let three worlds overwrite one report.
    assert "casemeasured" in ids[None]
    for world in SE.WORLD_EXPECTATIONS:
        assert f"case{world}" in ids[world]


def test_two_self_tests_of_different_worlds_do_not_overwrite_one_report(tmp_path,
                                                                        capsys):
    """End to end, because the collision was end to end: three worlds wrote one
    summary.json and the survivor was whichever ran last."""
    for world in ("kernel", "neither"):
        SE.main(["--self-test", world, "--fail-on-world", "--models",
                 "mixtral-8x7b,qwen2-57b-a14b", "--out-dir", str(tmp_path)])
    capsys.readouterr()
    summaries = sorted(tmp_path.glob("*/summary.json"))
    assert len(summaries) == 2
    worlds = {json.loads(path.read_text())["synthetic_world"]
              for path in summaries}
    assert worlds == {"kernel", "neither"}


def test_a_run_that_restored_every_arm_says_it_timed_nothing():
    """THE DRIVER'S TWO SPAN ARMS DERIVE ONE RUN ID, because `--densify` is now
    the default and one arm passes it. The second restores every row, times
    nothing, and would land DONE in the ledger having spent no minutes. The
    driver fix is deleting the bare arm and is not this file's; refusing to be
    silent about it is."""
    replay = SE.MeasurementTally(measured=0, restored=756,
                                 path=Path("/w/timings.csv"))
    assert replay.replay is True
    assert replay.note.startswith("REPLAY: this run TIMED NOTHING")
    assert "756" in replay.note and "/w/timings.csv" in replay.note
    # And the FAIL branch of the same question: a run that measured anything at
    # all is not a replay, however much it also restored.
    partial = SE.MeasurementTally(measured=1, restored=755,
                                  path=Path("/w/timings.csv"))
    assert partial.replay is False
    assert partial.note.startswith("MEASURED: 1 arm row(s)")
    assert SE.MeasurementTally(0, 0, Path("/w/timings.csv")).replay is False


def test_the_store_counts_what_it_restored_against_what_it_wrote(tmp_path):
    """The tally is not a guess: `Store` counts the two events as they happen,
    because from inside a resume and a duplicated driver arm are the same
    thing."""
    cell = SE.Cell("mixtral-8x7b", 256, "bf16")
    first = SE.Store(tmp_path / "t.csv", "H200")
    first.write(timed_arm(), cell, meta_for())
    assert (first.written_arms, first.restored_arms) == (1, 0)
    first.close()

    second = SE.Store(tmp_path / "t.csv", "H200")
    assert second.restore(("mixtral-8x7b", 256, "gemm_up")) is not None
    assert second.restore(("mixtral-8x7b", 256, "act")) is None
    second.close()
    assert (second.written_arms, second.restored_arms) == (0, 1)


def meta_for(gpu_name="H200"):
    return {"run_id": "abc", "gpu_name": gpu_name, "torch_version": "2.13.0",
            "triton_version": "3.7.1", "vllm_version": "0.27.1",
            "routing": "uniform", "seed": 0,
            "provenance_columns": PV.Provenance(git_sha="deadbeef").as_columns()}


def timed_arm(**over):
    fields = dict(ms_median=1.25, ms_mean=1.26, ms_stdev=0.01, ms_min=1.2,
                  n_samples=45,
                  config={"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
                          "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
                          "num_warps": 8, "num_stages": 4},
                  tile_config_source="vllm_observed", rows_total=512.0,
                  padded_rows=768.0, padding_source="histogram",
                  active_experts=8, triton_artifacts=3,
                  instrument="queue-deep/l2-flush/clock-under-load/v2",
                  warmup_ms=201.4, iters=160, trials=3,
                  sm_clock_load_mhz=1965.0, clock_level_ok=True,
                  clock_drift_ok=False, l2_flush=False, host_bound=False,
                  host_enqueue_ms=0.31, ms_p90=1.31)
    fields.update(over)
    return SE.ArmResult("mixtral-8x7b", 256, "gemm_up", **fields)


def test_a_row_round_trips_through_the_csv_unchanged(tmp_path):
    store = SE.Store(tmp_path / "t.csv", "H200")
    cell = SE.Cell("mixtral-8x7b", 256, "bf16")
    store.write(timed_arm(), cell, meta_for())
    store.close()

    fresh = SE.Store(tmp_path / "t.csv", "H200")
    back = fresh.restore(("mixtral-8x7b", 256, "gemm_up"))
    fresh.close()
    assert back.ms_median == pytest.approx(1.25)
    assert back.config["BLOCK_SIZE_M"] == 64
    assert back.padded_rows == pytest.approx(768.0)
    assert back.padding_source == "histogram"
    assert back.triton_artifacts == 3


def test_the_instrument_columns_round_trip_including_the_three_state_flags(tmp_path):
    """`clock_level_ok`, `clock_drift_ok`, `l2_flush` and `host_bound` each have
    THREE states, and the third is "not determined". Writing False for it would
    turn "NVML was unavailable" into "the clock was wrong", which is the shape
    of every silent defect in this apparatus."""
    store = SE.Store(tmp_path / "t.csv", "H200")
    cell = SE.Cell("mixtral-8x7b", 256, "bf16")
    store.write(timed_arm(), cell, meta_for())
    store.write(timed_arm(clock_level_ok=None, clock_drift_ok=None,
                          host_bound=None), cell, meta_for())
    store.close()
    rows = list(csv.DictReader((tmp_path / "t.csv").open(newline="")))
    assert [r["clock_level_ok"] for r in rows] == ["True", ""]
    assert [r["clock_drift_ok"] for r in rows] == ["False", ""]
    assert [r["l2_flush"] for r in rows] == ["False", "False"]

    fresh = SE.Store(tmp_path / "t.csv", "H200")
    back = fresh.restore(("mixtral-8x7b", 256, "gemm_up"))
    fresh.close()
    # The LAST row for a key wins on resume, and it is the undetermined one.
    assert back.instrument == "queue-deep/l2-flush/clock-under-load/v2"
    assert back.iters == 160 and back.trials == 3
    assert back.warmup_ms == pytest.approx(201.4)
    assert back.clock_level_ok is None and back.clock_drift_ok is None
    assert back.l2_flush is False


def test_every_row_carries_the_provenance_of_the_invocation_that_wrote_it(tmp_path):
    """Per row and not per file: a resumed CSV holds rows from two invocations
    at two commits, and a file-level header would attribute all of them to the
    last one."""
    cell = SE.Cell("mixtral-8x7b", 256, "bf16")
    store = SE.Store(tmp_path / "t.csv", "H200")
    store.write(timed_arm(), cell, meta_for())
    store.close()
    row = next(iter(csv.DictReader((tmp_path / "t.csv").open(newline=""))))
    assert row["prov_git_sha"] == "deadbeef"
    assert "prov_instrument" in row and "prov_utc" in row


def test_a_failed_row_is_retried_rather_than_restored(tmp_path):
    """The common failures here are a pod that lost its device and an arm that
    ran out of memory behind a since-finished neighbour, and a re-run can leave
    both behind. Restoring the failure would make it permanent."""
    store = SE.Store(tmp_path / "t.csv", "")
    cell = SE.Cell("mixtral-8x7b", 256, "bf16")
    store.write(SE.ArmResult("mixtral-8x7b", 256, "gemm_up",
                             error="CUDA out of memory"), cell, meta_for(""))
    store.close()
    fresh = SE.Store(tmp_path / "t.csv", "")
    assert fresh.restore(("mixtral-8x7b", 256, "gemm_up")) is None
    fresh.close()


# --------------------------------------------------------------------------
# 5b. the resume key, which used to let one card publish another card's timings
# --------------------------------------------------------------------------

def planted_card_a(tmp_path, gpu_name="NVIDIA H200"):
    """A finished timings.csv, written on `gpu_name`."""
    path = tmp_path / "timings.csv"
    store = SE.Store(path, gpu_name)
    for tokens in (256, 512):
        for arm in ("fused", "gemm_up", "gemm_down"):
            store.write(SE.ArmResult("mixtral-8x7b", tokens, arm, ms_median=1.0,
                                     ms_mean=1.0, ms_stdev=0.0, ms_min=1.0,
                                     n_samples=45),
                        SE.Cell("mixtral-8x7b", tokens, "bf16"),
                        meta_for(gpu_name))
    store.close()
    return path


def test_a_second_card_restores_none_of_the_first_cards_rows(tmp_path):
    """THE DEFECT, planted. The run id was card-free and `restore` ignored
    `gpu_name`, so a second card pointed at the same network volume found every
    (model, tokens, arm) triple present, measured ZERO cells, and printed the
    first card's timings under its own heading. Worse, the tile-config mismatch
    check then compared two restored first-card rows with each other."""
    path = planted_card_a(tmp_path)
    b = SE.Store(path, "NVIDIA A100-SXM4-80GB")
    try:
        assert b.restore(("mixtral-8x7b", 256, "gemm_up")) is None
        assert b.foreign == {"NVIDIA H200": 6}
        assert "will NOT be restored" in b.foreign_reason
        assert "NVIDIA H200" in b.foreign_reason
    finally:
        b.close()
    # And the same card still resumes, or the key would be a way of losing work.
    a = SE.Store(path, "NVIDIA H200")
    try:
        assert a.restore(("mixtral-8x7b", 256, "gemm_up")) is not None
    finally:
        a.close()


def test_a_dry_run_on_the_second_card_reports_zero_restorable_rows_and_says_why(
        tmp_path, capsys):
    """The plan says what a resume WOULD do before anything is measured. Pinning
    `--run-id` is what makes this reachable even with the card in the id: an
    operator resuming a killed run supplies the first card's id by hand."""
    planted_card_a(tmp_path / "pinned")
    SE.main(["--dry-run", "--models", "mixtral-8x7b", "--tokens", "256,512",
             "--out-dir", str(tmp_path), "--run-id", "pinned"])
    text = capsys.readouterr().out
    assert "RESUME: 0 row(s) on disk for this card" in text
    assert "will NOT be restored" in text
    assert "NVIDIA H200" in text


def test_the_survey_counts_errored_rows_apart_from_restorable_ones(tmp_path):
    path = planted_card_a(tmp_path)
    store = SE.Store(path, "NVIDIA H200")
    store.write(SE.ArmResult("mixtral-8x7b", 1024, "fused", error="OOM"),
                SE.Cell("mixtral-8x7b", 1024, "bf16"), meta_for("NVIDIA H200"))
    store.close()
    survey = SE.survey_resume(path, "NVIDIA H200")
    assert (survey.restorable, survey.errored, survey.foreign) == (6, 1, {})
    assert survey.reason == ""


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
    assert code == exit_codes.REFUSED
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


def test_dropping_a_corner_of_the_2x2_is_refused_with_REFUSED_and_not_one(capsys):
    """The three corners are the whole experiment; two of them separate nothing.

    AND THE CODE IS THE POINT. This refusal is raised as `SystemExit(str)`, which
    the interpreter turns into exit ONE, and one is CLAIM_FAIL: the driver would
    file a run that measured nothing as a refutation of the claim and never
    re-run the arm. Pinning `pytest.raises(SystemExit)` was pinning exactly that.
    """
    code = SE.main(["--dry-run", "--arms", "fused,cutlass_up"])
    assert code == SE.exit_codes.REFUSED
    assert code != SE.exit_codes.CLAIM_FAIL
    assert "REFUSED: --arms must include" in capsys.readouterr().err


def test_an_integer_SystemExit_is_re_raised_and_not_relabelled_REFUSED(monkeypatch):
    """The other half of the same arm. argparse exits `SystemExit(2)` itself, and
    an int code is already the right number: relabelling it would make this file
    an opinion about codes it does not own."""
    def bail(argv=None):
        raise SystemExit(7)

    monkeypatch.setattr(SE, "_main", bail)
    with pytest.raises(SystemExit) as caught:
        SE.main([])
    assert caught.value.code == 7


def test_a_crash_is_still_ERROR_and_the_traceback_is_not_swallowed(monkeypatch, capsys):
    """The SystemExit arm must not have shadowed the crash arm: a planted
    exception is the apparatus failing, so it is ERROR (4) with its traceback."""
    def boom(argv=None):
        raise RuntimeError("planted: the allocator gave up halfway")

    monkeypatch.setattr(SE, "_main", boom)
    assert SE.main([]) == SE.exit_codes.ERROR
    err = capsys.readouterr().err
    assert "planted: the allocator gave up halfway" in err
    assert "RuntimeError" in err


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


# --------------------------------------------------------------------------
# 7. the self test that asserts something, and the grid that refuses
# --------------------------------------------------------------------------

def test_every_world_reproduces_its_registered_row_on_the_grid_the_driver_runs():
    """THE FINDING, closed. The self test used to assert nothing per world, so
    nobody noticed that on the grid the driver scheduled FIRST the `kernel`
    world -- the world where the claim is TRUE -- produced C2 FAIL, C5 FAIL and
    C3 UNKNOWN. Every world now has a registered row and this is what checks it,
    on the default grid, which is the grid the driver runs."""
    for name in SE.WORLDS:
        _, _, _, gates = run_world(name, models=list(SE.DEFAULT_MODELS),
                                   densify=True)
        s_gates = SE.self_test_gates(solved_world(name), gates)
        bad = [g for g in s_gates if g.passed is not True]
        assert not bad, [(g.name, g.observed) for g in bad]


def solved_world(name):
    """The world as `main` builds it: `extent` has its scale SOLVED first."""
    world = SE.WORLDS[name]
    if name != "extent":
        return world
    cells = build(models=list(SE.DEFAULT_MODELS), densify=True)
    solve = SE.extent_scale_for_separation(
        cells, SE.PUBLISHED_SEPARATION_FIRST, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, seed=0)
    return SE.World(world.name, solve.scale, world.triton_pads,
                    world.cutlass_pads, world.summary)


def test_an_S_gate_FAILS_when_a_world_does_not_produce_its_registered_verdict():
    """The FAIL branch, planted. A self test whose S gates cannot fail is the
    same defect one level up."""
    _, _, _, gates = run_world("kernel", models=list(SE.DEFAULT_MODELS),
                               densify=True)
    flipped = [dataclasses.replace(g, passed=False) if g.name.startswith("C2")
               else g for g in gates]
    s_gates = SE.self_test_gates(SE.WORLDS["kernel"], flipped)
    failed = [g for g in s_gates if g.passed is False]
    assert [g.name for g in failed] == ["S-C2 world expectation"]
    assert "came out FAIL" in failed[0].observed


def test_a_world_with_no_registered_row_is_refused_rather_than_passed():
    unregistered = SE.World("invented", 1.0, True, False, "")
    s_gates = SE.self_test_gates(unregistered, [])
    assert len(s_gates) == 1 and s_gates[0].passed is False
    assert "cannot fail" in s_gates[0].invalidates


def test_a_gate_added_without_an_expectation_is_caught():
    """S-registered is the gate that keeps the table honest: a thirteenth gate
    with no registered verdict is a gate no world constrains."""
    _, _, _, gates = run_world("kernel", models=list(SE.DEFAULT_MODELS),
                               densify=True)
    extra = dataclasses.replace(gates[0], name="V9 invented")
    SE.GATE_TOKENS["V9"] = "invented"
    try:
        s_gates = SE.self_test_gates(SE.WORLDS["kernel"], [*gates, extra])
    finally:
        SE.GATE_TOKENS.pop("V9")
    registered = [g for g in s_gates if g.name.startswith("S-registered")][0]
    assert registered.passed is False
    assert "V9" in registered.observed


def test_the_registered_worlds_do_not_all_land_on_one_claim_gate_row():
    """Each world reproducing its own row proves nothing unless the rows differ.
    This is the other half of that proof, and it is over the TABLE because a run
    scores one world at a time."""
    assert len(set(SE.world_triples().values())) == len(SE.WORLD_EXPECTATIONS)


def test_S_discrimination_FAILS_when_two_worlds_register_one_row(monkeypatch):
    """THE PLANTED FAIL BRANCH of the gate above, which the assertion above does
    not exercise: it recomputes the gate's own PASS condition and would hold
    just as well if the gate were deleted. Here `neither` is given `kernel`'s
    row -- the exact state the sparse grid produced on hardware -- and the gate
    has to say so."""
    _, _, _, gates = run_world("kernel", models=list(SE.DEFAULT_MODELS),
                               densify=True)
    passing = SE.self_test_gates(SE.WORLDS["kernel"], gates)
    assert gate(passing, "S-discrimination").passed is True

    monkeypatch.setitem(SE.WORLD_EXPECTATIONS, "neither",
                        dict(SE.WORLD_EXPECTATIONS["kernel"]))
    failing = gate(SE.self_test_gates(SE.WORLDS["kernel"], gates),
                   "S-discrimination")
    assert failing.passed is False
    assert "distinct rows across" in failing.observed
    assert "would not be evidence about which world we are in" in failing.invalidates


def test_a_registered_gate_that_was_never_scored_is_caught_and_is_UNKNOWN(
        monkeypatch):
    """The OTHER half of S-registered, and the UNKNOWN branch of an S-<KEY>.

    `test_a_gate_added_without_an_expectation_is_caught` plants "scored but not
    registered". This plants "registered but not scored", which is what a gate
    silently disappearing from `build_gates` looks like, and it must not be a
    quiet PASS: the row it stands for is then an expectation about nothing, and
    the S gate that mirrors it cannot say the world was right or wrong."""
    _, _, _, gates = run_world("kernel", models=list(SE.DEFAULT_MODELS),
                               densify=True)
    dropped = [g for g in gates if not g.name.startswith("C3")]
    s_gates = SE.self_test_gates(SE.WORLDS["kernel"], dropped)

    registered = gate(s_gates, "S-registered")
    assert registered.passed is False
    assert "registered but not scored: ['C3']" in registered.observed

    orphan = gate(s_gates, "S-C3")
    assert orphan.passed is None
    assert orphan.tag == exit_codes.UNKNOWN
    assert "C3 was not scored in this run" in orphan.observed


def test_the_sparse_grid_cannot_tell_the_kernel_world_from_the_neither_world():
    """WHY --densify IS THE DEFAULT, measured rather than asserted. On the
    published powers-of-two grid every expert holds a power-of-two number of
    rows and BLOCK_M is a power of two, so the padding factor is exactly 1.00 at
    every point and the mechanism C2 names is invisible where the grid looks."""
    rows = {}
    for name in ("kernel", "neither"):
        _, _, _, gates = run_world(name, models=list(SE.DEFAULT_MODELS),
                                   densify=False)
        rows[name] = tuple(gate(gates, g).tag
                           for g in SE.DISCRIMINATING_GATES)
    assert rows["kernel"] == rows["neither"]
    assert gate(run_world("kernel", models=list(SE.DEFAULT_MODELS),
                          densify=False)[3], "C2").passed is False


def test_the_grid_power_check_refuses_the_sparse_grid_and_names_the_remedy():
    sparse = build(models=list(SE.DEFAULT_MODELS), densify=False)
    power = SE.c2_grid_power(sparse, ridge=RIDGE, bandwidth_gbps=BANDWIDTH,
                             seed=0)
    assert power.can_answer is False
    assert power.verdict == "FAIL"
    assert "grid too sparse for C2" in power.remedy and "--densify" in power.remedy


def test_the_grid_power_check_passes_the_grid_the_driver_runs():
    dense = build(models=list(SE.DEFAULT_MODELS), densify=True)
    power = SE.c2_grid_power(dense, ridge=RIDGE, bandwidth_gbps=BANDWIDTH,
                             seed=0)
    assert power.can_answer is True and power.remedy == ""


def test_densify_is_on_by_default_so_the_driver_gets_a_grid_that_can_answer():
    args = SE.build_parser().parse_args([])
    assert args.densify is True
    assert SE.build_parser().parse_args(["--no-densify"]).densify is False


def test_the_self_test_the_driver_line_names_exits_DONE(tmp_path, capsys):
    """`--self-test kernel --fail-on-world` on the grid the driver runs. This is
    the command the audit asked for and the 0 is the whole point."""
    code = SE.main(["--self-test", "kernel", "--fail-on-world",
                    "--out-dir", str(tmp_path)])
    capsys.readouterr()
    assert code == exit_codes.DONE


def test_a_self_test_without_the_flag_measured_nothing_and_says_so(tmp_path,
                                                                   capsys):
    code = SE.main(["--self-test", "kernel", "--out-dir", str(tmp_path)])
    assert "NOT A RESULT" in capsys.readouterr().out
    assert code == exit_codes.REFUSED


def test_a_self_test_of_a_world_that_misbehaves_exits_INVALID(tmp_path,
                                                              capsys,
                                                              monkeypatch):
    """The FAIL branch of the flag, through `main`. A registered row nobody can
    reproduce means nothing on the page may be quoted, which is INVALID."""
    broken = dict(SE.WORLD_EXPECTATIONS["kernel"], C2="FAIL")
    monkeypatch.setitem(SE.WORLD_EXPECTATIONS, "kernel", broken)
    code = SE.main(["--self-test", "kernel", "--fail-on-world",
                    "--out-dir", str(tmp_path)])
    text = capsys.readouterr().out
    assert code == exit_codes.INVALID
    assert "RESULT: VALIDITY self_test_kernel_is_the_story FAIL" in text


def test_the_refusal_before_measuring_exits_REFUSED_and_never_INVALID(tmp_path,
                                                                      capsys):
    """A grid that cannot answer costs nothing, so it is REFUSED. Until
    2026-09-02 this script's refusals exited 3 against a driver that read 2 as
    REFUSED and everything else as RETRY, so a refusal was queued for another
    attempt."""
    code = SE.main(["--no-densify", "--out-dir", str(tmp_path)])
    text = capsys.readouterr().out
    assert code == exit_codes.REFUSED
    assert "REFUSED: grid too sparse for C2" in text
    assert not list(tmp_path.glob("*/timings.csv"))


# --------------------------------------------------------------------------
# 8. the exit-code contract and the one greppable line
# --------------------------------------------------------------------------

def test_every_gate_has_a_registered_one_token_name():
    """A RESULT line's name is one run of non-whitespace by
    `exit_codes.result_line`'s own rule, and it is what a driver greps."""
    _, _, _, gates = run_world("kernel", models=list(SE.DEFAULT_MODELS),
                               densify=True)
    tokens = [g.token for g in gates]
    assert len(set(tokens)) == len(tokens)
    for token in tokens:
        assert token and not any(c.isspace() for c in token)


def test_a_gate_with_no_registered_token_raises_rather_than_deriving_one():
    _, _, _, gates = run_world("kernel", densify=True)
    orphan = dataclasses.replace(gates[0], name="V42 nobody registered this")
    with pytest.raises(KeyError):
        _ = orphan.token


def test_the_result_lines_round_trip_and_imply_the_exit_code(tmp_path, capsys):
    """`exit_codes.classify_text` recomputes the verdict from what was printed;
    a disagreement between the printed lines and the returned code is itself a
    defect, and the summary grep this replaces matched free prose 18 times."""
    code = SE.main(["--self-test", "kernel", "--fail-on-world",
                    "--out-dir", str(tmp_path)])
    text = capsys.readouterr().out
    lines = exit_codes.parse_result_lines(text)
    assert lines and exit_codes.classify_text(text) == code


def test_nothing_but_a_scored_gate_prints_a_RESULT_line(tmp_path, capsys):
    """One line per scored gate and no other line that looks like one. The
    human `[PASS] V0 ...` block stays, because the session driver's span regex
    has matched that shape since before the RESULT line existed, but only one of
    the two is the machine contract."""
    SE.main(["--self-test", "kernel", "--fail-on-world", "--models",
             "mixtral-8x7b,qwen2-57b-a14b", "--out-dir", str(tmp_path)])
    text = capsys.readouterr().out
    printed = [line for line in text.splitlines() if line.startswith("RESULT: ")]
    assert len(printed) == len(exit_codes.parse_result_lines(text))
    # 13 gates plus S-registered and S-discrimination.
    assert len(printed) == len(SE.WORLD_EXPECTATIONS["kernel"]) + 2


def pod_shaped_gates():
    """The kernel world's gates with V7 flipped to PASS.

    A synthetic world FAILS V7 by construction -- no instrument produced its
    numbers -- so classifying its twelve gates always gives INVALID, which is
    exactly why `main` scores a self test on its S gates instead. To exercise
    the exit-code contract for a POD run the one gate that cannot pass off-GPU
    is flipped, and nothing else."""
    _, _, _, gates = run_world("kernel", models=list(SE.DEFAULT_MODELS),
                               densify=True)
    flipped = [dataclasses.replace(g, passed=True) if g.name.startswith("V7")
               else g for g in gates]
    assert exit_codes.classify(g.scored() for g in flipped) == exit_codes.DONE
    return flipped


def test_an_UNKNOWN_claim_gate_is_CLAIM_FAIL_and_never_DONE():
    """The rule is one-directional on purpose: a gate that could not decide has
    not passed. The old exit was `1 if any gate FAILED`, which scored UNKNOWN as
    a pass."""
    gates = pod_shaped_gates()
    unknown = [dataclasses.replace(g, passed=None) if g.name.startswith("C3")
               else g for g in gates]
    assert exit_codes.classify(g.scored() for g in unknown) == exit_codes.CLAIM_FAIL


def test_a_validity_failure_is_INVALID_and_a_claim_failure_is_CLAIM_FAIL():
    """The two used to be one integer. A refuted claim is a RESULT the driver
    must mark finished; a broken instrument is a page nobody may quote."""
    gates = pod_shaped_gates()
    void = [dataclasses.replace(g, passed=False) if g.name.startswith("V2")
            else g for g in gates]
    refuted = [dataclasses.replace(g, passed=False) if g.name.startswith("C2")
               else g for g in gates]
    assert exit_codes.classify(g.scored() for g in void) == exit_codes.INVALID
    assert exit_codes.classify(g.scored() for g in refuted) == exit_codes.CLAIM_FAIL


# --------------------------------------------------------------------------
# 9. one instrument, the provenance block, and the interval's scope
# --------------------------------------------------------------------------

def test_the_retired_private_timing_loop_is_gone():
    """`time_calls` was one of six verbatim copies of the retired `time_call`:
    events created inside the loop, one synchronize per iteration, no flush, no
    clock. The audit bounded the host prefix it admitted at ~0.18 ms per
    fused_experts call on the H200 pod and ~0.30 ms on the A100."""
    assert not hasattr(SE, "time_calls")
    source = (ROOT / "scripts" / "span_extent_separation.py").read_text()
    assert "def time_call" not in source
    assert "timing.time_kernel(" in source


def test_time_arm_hands_the_shared_instrument_the_flags_from_argv(monkeypatch):
    """The instrument owns iters and warmup now, so what this script still owns
    is which knobs it hands over. A wrong one here is a row whose columns
    describe a measurement nobody made."""
    from moe.bench import timing

    seen = {}

    def fake(fn, **kw):
        seen.update(kw)
        return "timed"

    monkeypatch.setattr(timing, "time_kernel", fake)
    args = SE.build_parser().parse_args(["--warmup-ms", "50", "--target-ms",
                                         "120", "--trials", "4", "--l2-flush"])
    assert SE.time_arm(lambda: None, args, 1965.0) == "timed"
    assert seen == {"warmup_ms": 50.0, "target_ms": 120.0, "trials": 4,
                    "l2_flush": True, "reference_clock_mhz": 1965.0}


def test_V7_fails_when_two_instruments_appear_in_one_run():
    """A resumed CSV can hold rows from two invocations. Every ratio on the page
    divides one arm's milliseconds by another's, so two rulers is not a ratio."""
    cells = build(models=list(SE.DEFAULT_MODELS), densify=True)
    results = SE.synthetic_results(cells, SE.WORLDS["kernel"], ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    basis = SE.timing_basis()
    for arms in results.values():
        for name, arm in arms.items():
            arms[name] = dataclasses.replace(arm, instrument=basis)
    clean = SE.analyse(cells, results)
    assert gate(SE.build_gates(clean), "V7").passed is True

    first = next(iter(results.values()))
    key = next(iter(first))
    first[key] = dataclasses.replace(first[key], instrument="")
    mixed = SE.analyse(cells, results)
    v7 = gate(SE.build_gates(mixed), "V7")
    assert v7.passed is False
    assert "(none recorded)" in v7.observed


def flushed_world(flush_by_arm):
    """The kernel world with a real instrument stamped and a chosen flush state.

    A synthetic world carries `synthetic:<world>` and no flush at all, which is
    the state V7 refuses; to ask V7 about the FLUSH the rows have to be
    otherwise clean, so the basis is stamped on and `l2_flush` is set per arm.
    """
    cells = build(models=list(SE.DEFAULT_MODELS), densify=True)
    results = SE.synthetic_results(cells, SE.WORLDS["kernel"], ridge=RIDGE,
                                   bandwidth_gbps=BANDWIDTH, noise=0.0, seed=0)
    basis = SE.timing_basis()
    for arms in results.values():
        for name, arm in arms.items():
            arms[name] = dataclasses.replace(arm, instrument=basis,
                                             l2_flush=flush_by_arm(name))
    return cells, results


def test_V7_FAILS_when_flushed_and_unflushed_rows_share_one_instrument():
    """`KernelTiming` stamps TIMING_BASIS on every row it produces WHATEVER the
    flush was, so a CSV mixing flushed with unflushed rows carries one
    instrument string and two rulers. V7 read the string until 2026-09-02 and
    called that one ruler; a flushed arm pays a cold L2 per iteration and an
    unflushed one does not, and every number on this page is one arm's
    milliseconds over another's."""
    _, results = flushed_world(lambda name: False)
    clean = SE.analyse(build(models=list(SE.DEFAULT_MODELS), densify=True),
                       results)
    assert gate(SE.build_gates(clean), "V7").passed is True
    assert set(clean.rulers) == {f"{SE.timing_basis()} l2_flush=off"}

    cells, mixed_results = flushed_world(lambda name: name == "fused")
    mixed = SE.analyse(cells, mixed_results)
    v7 = gate(SE.build_gates(mixed), "V7")
    assert v7.passed is False
    assert "l2_flush=on" in v7.observed and "l2_flush=off" in v7.observed
    # One instrument, two rulers: the column alone cannot see this.
    assert len(mixed.instruments) == 1 and len(mixed.rulers) == 2


def test_rows_this_script_times_are_published_as_NOT_roof_comparable():
    """timing.py's contract is that a row is comparable with the ROOF only if it
    carries TIMING_BASIS. This script times with the flush OFF on purpose (see
    `time_arm`: V2 compares an isolated launch against the same launch inside a
    five-launch sequence), so its rows carry the basis of an instrument
    configured differently from the one the roof was measured with. The negative
    travels with the numbers rather than living in a docstring."""
    basis = SE.timing_basis()
    assert SE.roof_comparable({f"{basis} l2_flush=off": 9}, basis) is False
    assert SE.roof_comparable({f"{basis} l2_flush=on": 9}, basis) is True
    assert SE.roof_comparable({"synthetic:kernel l2_flush=unrecorded": 9},
                              basis) is False
    # Two states that are not an answer: nothing timed, and a host that cannot
    # name the instrument to compare against.
    assert SE.roof_comparable({}, basis) is None
    assert SE.roof_comparable({f"{basis} l2_flush=on": 9}, None) is None

    _, results = flushed_world(lambda name: False)
    cells = build(models=list(SE.DEFAULT_MODELS), densify=True)
    v7 = gate(SE.build_gates(SE.analyse(cells, results)), "V7")
    assert "NOT ROOF-COMPARABLE" in v7.observed


def test_the_summary_publishes_the_ruler_and_the_roof_answer(tmp_path, capsys):
    """A consumer applying "carries TIMING_BASIS, therefore roof-comparable" to
    these rows must find the contradiction in the file."""
    SE.main(["--self-test", "kernel", "--fail-on-world", "--models",
             "mixtral-8x7b,qwen2-57b-a14b", "--out-dir", str(tmp_path)])
    capsys.readouterr()
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert summary["roof_comparable"] is False
    assert "flush ON" in summary["roof_comparable_note"]
    assert list(summary["rulers"]) == ["synthetic:kernel l2_flush=unrecorded"]


def test_V7_is_UNKNOWN_and_never_PASS_when_nothing_was_timed():
    """A check that examined nothing reports zero failures too."""
    cells = build(models=["mixtral-8x7b"], tokens=[256])
    empty = SE.analyse(cells, {})
    assert gate(SE.build_gates(empty), "V7").passed is None


def test_a_self_test_page_is_not_quotable_because_no_instrument_produced_it():
    """V7 FAILS in every synthetic world by construction, which is the same job
    `bm128_roofline`'s hypothesis-roof refusal does for a synthetic roof."""
    for name in SE.WORLDS:
        _, _, _, gates = run_world(name, models=list(SE.DEFAULT_MODELS),
                                   densify=True)
        v7 = gate(gates, "V7")
        assert v7.passed is False
        assert v7.observed.startswith(f"observed synthetic:{name}")


def test_the_representative_repeat_is_the_median_one_with_the_worst_flags():
    """Two rules pulling opposite ways: the instrument columns describe the SAME
    repeat the reported median came from, and the three verdict flags are the
    worst of all repeats, because a reader filtering rows must not be shown the
    best of three."""
    from moe.bench import timing

    def rec(ms, level, drift, host):
        return timing.KernelTiming(
            ms_p50=ms, ms_p90=ms, ms_min=ms, ms_std=0.0, iters=int(ms * 100),
            trials=3, warmup_ms=200.0, l2_flush=False, sm_clock_load_mhz=1900.0,
            sm_clock_start_mhz=1900.0, sm_clock_end_mhz=1900.0,
            clock_level_ok=level, clock_drift_ok=drift, samples=90,
            warmup_calls=10, flush_mb=0, clock_samples=5, clock_source="nvml",
            clock_poll_ms=1.0, host_bound=host, host_enqueue_ms=0.1)

    chosen = SE.representative_timing([rec(1.0, True, True, False),
                                       rec(3.0, True, False, False),
                                       rec(2.0, True, True, True)])
    assert chosen.ms_p50 == 2.0 and chosen.iters == 200
    assert chosen.clock_level_ok is True
    assert chosen.clock_drift_ok is False
    assert chosen.host_bound is True


def test_the_drifted_row_carries_the_drifting_repeats_start_and_end():
    """R3's evidence columns must describe the repeat the DRIFT verdict came
    from, not the median-p50 one.

    THE DEFECT, 2026-09-09: `clock_drift_ok` folds with `worst` over every
    repeat while `sm_clock_start_mhz`, `sm_clock_end_mhz`, `clock_samples_mhz`
    and `power_w` rode in on `dataclasses.replace(chosen, ...)`, i.e. from the
    median repeat. A folded row could read clock_drift_ok=0 beside a start and
    an end that were EQUAL, taken from a repeat that did not drift, which is
    exactly the "a drifted row cannot say which way its clock went" defect the
    columns were added to close."""
    from moe.bench import timing

    def rec(ms, start, end, drift):
        return timing.KernelTiming(
            ms_p50=ms, ms_p90=ms, ms_min=ms, ms_std=0.0, iters=10, trials=3,
            warmup_ms=200.0, l2_flush=False, sm_clock_load_mhz=start,
            sm_clock_start_mhz=start, sm_clock_end_mhz=end,
            clock_level_ok=True, clock_drift_ok=drift, samples=90,
            warmup_calls=10, flush_mb=0, clock_samples=5, clock_source="nvml",
            clock_poll_ms=1.0, host_bound=False, host_enqueue_ms=0.1)

    steady_fast = rec(1.0, 1900.0, 1900.0, True)
    steady_median = rec(2.0, 1900.0, 1900.0, True)
    sagging = rec(3.0, 1900.0, 1500.0, False)
    folded = SE.representative_timing([steady_fast, steady_median, sagging])
    # The median repeat is still the one the numbers describe.
    assert folded.ms_p50 == 2.0
    # The verdict is the worst of the three.
    assert folded.clock_drift_ok is False
    # And the evidence is the drifting repeat's, so the row says WHICH WAY.
    assert folded.sm_clock_start_mhz == 1900.0
    assert folded.sm_clock_end_mhz == 1500.0
    assert folded.sm_clock_end_mhz < folded.sm_clock_start_mhz
    assert SE.clock_samples_of(folded)["sm_clock_end_mhz"] == 1500.0
    # A fold with no drift keeps the chosen repeat's evidence untouched.
    clean = SE.representative_timing([steady_fast, steady_median])
    assert clean.clock_drift_ok is True
    assert clean.sm_clock_start_mhz == clean.sm_clock_end_mhz == 1900.0


def test_the_representative_docstring_says_which_side_of_the_fold_the_evidence_is_on():
    """The docstring listed `iters`, `warmup_ms`, `trials`, `l2_flush` and the
    clock as coming from the chosen repeat and said nothing about the four
    evidence columns added beside them on 2026-09-09, so nothing in the file
    stated which side of the fold they were on."""
    doc = SE.representative_timing.__doc__
    for column in ("sm_clock_start_mhz", "sm_clock_end_mhz",
                   "clock_samples_mhz", "power_w"):
        assert column in doc, column
    flat = " ".join(doc.split())
    assert "COLUMNS FOLLOW THE DRIFT VERDICT, NOT THE MEDIAN REPEAT" in flat
    # And it must not send a reader to a column this script never writes.
    assert "`roof_at_cell_clock` is scored from" not in doc


def test_the_representative_repeat_keeps_None_when_no_repeat_determined_a_flag():
    from moe.bench import timing

    def rec(ms):
        return timing.KernelTiming(
            ms_p50=ms, ms_p90=ms, ms_min=ms, ms_std=0.0, iters=10, trials=1,
            warmup_ms=1.0, l2_flush=False, sm_clock_load_mhz=None,
            sm_clock_start_mhz=None, sm_clock_end_mhz=None,
            clock_level_ok=None, clock_drift_ok=None, samples=10,
            warmup_calls=1, flush_mb=0, clock_samples=0, clock_source="none",
            clock_poll_ms=None, host_bound=None, host_enqueue_ms=None)

    chosen = SE.representative_timing([rec(1.0), rec(2.0)])
    assert chosen.clock_level_ok is None and chosen.host_bound is None


def test_the_summary_carries_the_provenance_block_and_the_five_top_level_keys(
        tmp_path):
    """The audit found no commit, card, ruler or ridge source in any of the 26
    published reports. A publish gate that only checks PRESENCE is
    rubber-stamping, so the values are checked too, where they can be known."""
    import json

    SE.main(["--self-test", "kernel", "--fail-on-world", "--models",
             "mixtral-8x7b", "--out-dir", str(tmp_path)])
    payload = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    for key in PV.TOP_LEVEL_KEYS:
        assert key in payload
    assert payload["provenance"]["provenance_version"] == PV.PROVENANCE_VERSION
    assert payload["provenance"]["python"]
    assert "COST MODEL ONLY" in payload["ridge_source"]
    assert payload["card"] == SE.detect_card()


def test_the_intervals_carry_their_scope_and_their_n():
    """`MIN_DECOMPOSED_MODELS` is 2, so a run that satisfies V4 can still print a
    90% interval built from two points. Two floats in a JSON file with neither
    beside them is an error bar a reader will believe."""
    _, _, analysis, _ = run_world("kernel", models=TWO_MODELS, densify=True)
    assert analysis.interval_n == 2
    assert "DEGENERATE at n=2" in analysis.interval_scope
    assert "resampling the 2 decomposed model(s)" in analysis.interval_scope

    _, _, four, _ = run_world("kernel", models=list(SE.DEFAULT_MODELS),
                              densify=True)
    assert four.interval_n == 4
    assert "DEGENERATE" not in four.interval_scope


def test_the_MDE_is_derived_from_a_stated_noise_assumption():
    """The prior is `replicate_noise_floor.PRIOR_SD`, 0.0323/sqrt(2), an UPPER
    bound on the per-arm replicate spread. An arm quoting an effect smaller than
    this line has quoted noise."""
    assert SE.PRIOR_ARM_SD_LOG == pytest.approx(0.0323 / math.sqrt(2), abs=1e-9)
    four = SE.mde_log_ratio(4)
    two = SE.mde_log_ratio(2)
    assert four is not None and two > four
    assert four == pytest.approx(SE.MDE_Z_SUM * math.sqrt(2)
                                 * SE.PRIOR_ARM_SD_LOG / 2.0, rel=1e-9)


def test_the_MDE_refuses_below_two_models_rather_than_returning_a_number():
    assert SE.mde_log_ratio(1) is None
    with pytest.raises(ValueError):
        SE.mde_log_ratio(4, sd_log=0.0)


def test_the_plan_names_the_MDE_and_the_gate_margins_it_can_resolve(capsys):
    SE.main(["--dry-run"])
    text = capsys.readouterr().out
    assert "MDE, BEFORE ANYTHING RUNS" in text
    assert "noise assumption" in text and "PRIOR_SD" in text
    assert "C2 margin" in text and "resolvable" in text


def test_the_plan_names_the_instrument_and_where_it_would_clamp(capsys):
    SE.main(["--dry-run", "--target-ms", "1"])
    text = capsys.readouterr().out
    assert SE.timing_basis() in text
    assert "L2 flush OFF" in text
    assert "OVERRUN the budget" in text


def test_the_clamp_is_read_off_the_shared_function_and_not_restated():
    """TWO DOCSTRINGS HERE SAID [10, 10000] AGAINST A REAL hi OF 2000, and the
    difference is a factor of five on the threshold a third of the grid sits
    under. A copy of a bound drifts, so the plan asks `iters_for` for its own
    defaults and this pins that it is the same function's."""
    from moe.bench import timing

    params = inspect.signature(timing.iters_for).parameters
    assert SE.iters_clamp() == (params["lo"].default, params["hi"].default)
    assert SE.iters_clamp() != (10, 10000)


def test_the_plan_names_the_arms_that_will_UNDERRUN_the_budget(capsys):
    """The overrun side was printed and the underrun side was not, so the plan
    said "no trial should overrun the budget" and nothing at all about the 261
    arms of 756 that deliver less measured time than the budget bought."""
    SE.main(["--dry-run"])
    text = capsys.readouterr().out
    assert "iters clamp [10, 2000]" in text
    assert "261 of 756 arm(s) are modelled faster than 200 ms / 2000" in text
    assert "UNDERRUN the budget" in text
    assert "no arm is modelled slower than 200 ms / 10" in text


def test_both_clamp_lines_have_a_PASS_branch_and_a_FAIL_branch(capsys):
    """A plan line that can only say one thing is not a check. `--target-ms 1`
    puts every arm over the top clamp and none under the bottom one, which is
    the mirror image of the default grid."""
    plan = "\n".join(SE.render_instrument_plan(
        build(models=["mixtral-8x7b"], tokens=[1, 8192]),
        ["fused", "gemm_up", "gemm_down"], target_ms=1.0, warmup_ms=200.0,
        trials=3, l2_flush=False, ridge=RIDGE, bandwidth_gbps=BANDWIDTH))
    assert "OVERRUN the budget" in plan
    assert "no arm is modelled faster than 1 ms / 2000" in plan

    slow_none = "\n".join(SE.render_instrument_plan(
        build(models=["mixtral-8x7b"], tokens=[1]),
        ["align"], target_ms=200.0, warmup_ms=200.0, trials=3, l2_flush=False,
        ridge=RIDGE, bandwidth_gbps=BANDWIDTH))
    assert "no trial should overrun the budget" in slow_none
    assert "UNDERRUN the budget" in slow_none


def test_the_cost_estimate_over_charges_the_arms_that_underrun_and_says_so():
    """`estimated_seconds` returns the design's BUDGET. Below the top clamp the
    trial delivers less than the budget, so the number is an over-estimate
    there, and the docstring that claimed the cost never depends on the cell's
    own speed held for two thirds of the default grid."""
    lo, hi = SE.iters_clamp()
    cells = build(models=["mixtral-8x7b"], tokens=[1])
    fastest = min(SE.modelled_arm_ms(c, a, ridge=RIDGE,
                                     bandwidth_gbps=BANDWIDTH)
                  for c in cells for a in ["align", "act", "sum"])
    assert fastest * hi < 200.0
    budget = SE.estimated_seconds(cells, ["align"], reps=1, target_ms=200.0,
                                  warmup_ms=200.0, trials=3)
    delivered = (200.0 + 3 * fastest * hi) / 1e3
    assert budget == pytest.approx(0.8)
    assert delivered < budget


def test_the_cost_estimate_is_the_instruments_budget_and_not_the_cells_speed():
    """`time_kernel` sizes iters so every arm costs the same wall time. The old
    estimate multiplied a modelled millisecond by a call count and said a
    deepseek-v3 T=8192 cell cost a thousand times a mixtral T=1 one."""
    cells = build(models=["mixtral-8x7b"], tokens=[1, 8192])
    arms = ["fused", "gemm_up", "gemm_down"]
    seconds = SE.estimated_seconds(cells, arms, reps=2, target_ms=100.0,
                                   warmup_ms=50.0, trials=3)
    assert seconds == pytest.approx(len(cells) * 3 * 2 * (50 + 300) / 1e3)


def test_V7_is_UNKNOWN_when_this_machine_cannot_name_the_instrument(monkeypatch):
    """The laptop-replay branch. Torch will not import, so this process cannot
    say what the rows on disk should have said; UNKNOWN is that state and it
    still counts against the gate."""
    monkeypatch.setattr(SE, "timing_basis", lambda: None)
    _, _, _, gates = run_world("kernel", models=TWO_MODELS, densify=True)
    v7 = gate(gates, "V7")
    assert v7.passed is None
    assert "cannot name the instrument" in v7.observed


def test_a_find_pieces_refusal_exits_REFUSED_and_not_INVALID(tmp_path, capsys,
                                                             monkeypatch):
    """`find_pieces` raising means a launch of vLLM's fused path is not where
    this script looked. Nothing was measured and nothing was spent, so it is
    REFUSED. It used to exit 3 against a driver that read 2 as REFUSED and
    everything else as RETRY, so the arm was queued for another attempt that
    would refuse again."""
    monkeypatch.setattr(SE, "missing_gpu_stack", lambda: "")
    monkeypatch.setattr(
        SE, "run_measurement",
        lambda *a, **k: (None, "PieceMissing: silu_and_mul",
                         SE.MeasurementTally(0, 0, tmp_path / "timings.csv")))
    code = SE.main(["--models", "mixtral-8x7b", "--out-dir", str(tmp_path)])
    capsys.readouterr()
    assert code == exit_codes.REFUSED


def test_the_missing_stack_refusal_exits_REFUSED_and_names_which_half(tmp_path,
                                                                      capsys):
    """No CUDA here, so this is the real path rather than a planted one."""
    code = SE.main(["--models", "mixtral-8x7b", "--out-dir", str(tmp_path)])
    text = capsys.readouterr().out
    assert code == exit_codes.REFUSED
    assert "REFUSED:" in text and "--self-test kernel" in text


def test_an_unplanned_crash_exits_ERROR_and_never_CLAIM_FAIL(monkeypatch, capsys):
    """The apparatus breaking must not be filed as one of the arm's outcomes.

    An exception left to propagate exits the interpreter ONE, and ONE is
    CLAIM_FAIL, which `moe/bench/exit_codes.py` puts in FINISHED_CODES: the
    driver records the arm as finished, skips it on every resume, leaves
    RETRY_ARMS at zero and exits the session 0 over an arm that never measured.
    ERROR (4) is outside FINISHED_CODES so the two can be told apart, and the
    traceback is printed rather than swallowed because a bare code names
    nothing to fix. Planted rather than argued: span_extent_separation had no top-level handler
    until 2026-09-02.
    """
    from moe.bench import exit_codes as EX

    def explode(argv=None):
        raise RuntimeError("planted: the allocator gave up halfway")

    monkeypatch.setattr(SE, "_main", explode)
    code = SE.main([])
    err = capsys.readouterr().err
    assert code == EX.ERROR
    assert code != EX.CLAIM_FAIL
    assert code not in EX.FINISHED_CODES, "an apparatus failure must stay retryable"
    assert EX.ledger_state(code) == "RETRY"
    assert "planted: the allocator gave up halfway" in err, \
        "the traceback was swallowed"
    assert "RuntimeError" in err


# --------------------------------------------------------------------------
# 9. the cost model's ceilings are resolved from a file, never a constant
# --------------------------------------------------------------------------

def test_the_withdrawn_band_is_gone_by_name_and_named_as_history():
    """`RIDGE_BAND = (160.3, 176.2)` and `BANDWIDTH_GBPS = 4374.5` used to sit
    in this module and `--ridge` defaulted to the band's low end. The band was
    two H200 compute calibrations 9.9% apart and no card's ridge."""
    assert not hasattr(SE, "RIDGE_BAND")
    assert not hasattr(SE, "BANDWIDTH_GBPS")
    assert "4d84542b" in SE.WITHDRAWN_RIDGE_BAND_WHY
    parser = SE.build_parser()
    assert parser.get_default("ridge") == 0.0
    assert parser.get_default("bandwidth_gbps") == 0.0


def test_a_laptop_plan_prices_against_the_committed_h200_file_and_says_so():
    """No device: the hypothesis is the committed calibration, read through
    `roofline`, and both source strings say HYPOTHESIS and COST MODEL ONLY."""
    from moe.bench import roofline

    hw = roofline.load_hardware(SE.HYPOTHESIS_CALIBRATION)
    c = SE.resolve_cost_ceilings(0.0, 0.0, SE.NO_CARD, "bf16")
    assert c.kind == "hypothesis"
    assert c.ridge == pytest.approx(hw.ridge_point("bf16"))
    assert c.bandwidth_gbps == pytest.approx(hw.bandwidth_bytes_s / 1e9)
    # THE CARD'S OWN COMMITTED NUMBER, not a literal beside it. This line
    # pinned 162.8 until 2026-09-09, when ab61e55 recalibrated the H200 on the
    # pod and the committed ridge became 152.81: the literal was the
    # superseded live figure and the assertion above it already read the file,
    # so the pair asserted the same quantity twice with two different answers
    # and the second one went red at the calibration commit.
    assert c.ridge == pytest.approx(hw.ridge_point("bf16"), abs=0.1)
    assert c.ridge != pytest.approx(162.8, abs=0.1), (
        "162.8 is the pre-ab61e55 H200 ridge; the committed file says 152.81")
    for src in (c.ridge_source, c.bandwidth_source):
        assert src.startswith("COST MODEL ONLY") and "HYPOTHESIS" in src
    assert "160.3" not in c.ridge_source


def test_an_operator_s_ridge_and_bandwidth_are_taken_as_stated():
    c = SE.resolve_cost_ceilings(150.0, 2000.0, "NVIDIA Whatever", "bf16")
    assert (c.ridge, c.bandwidth_gbps, c.kind) == (150.0, 2000.0, "cli")
    assert "given on the command line" in c.ridge_source
    assert "given on the command line" in c.bandwidth_source


def test_a_card_with_no_calibration_is_refused_not_priced_against_another(
        monkeypatch):
    """THE FAIL BRANCH, planted: a real device, no calibration for it, and no
    --ridge. The refusal names the fix and the withdrawn constant never
    stands in; `main` files a string SystemExit as REFUSED, not CLAIM_FAIL."""
    from moe.bench import roofline

    monkeypatch.setattr(roofline, "load_measured", lambda *a, **k: None)
    with pytest.raises(SystemExit) as caught:
        SE.resolve_cost_ceilings(0.0, 0.0, "NVIDIA Planted-GPU", "bf16")
    msg = str(caught.value.code)
    assert msg.startswith("REFUSED") and "calibrate_hardware" in msg
    assert "160.3" in msg and "withdrawn" in msg, "history is named, not used"

    def mismatch(*a, **k):
        raise roofline.HardwareMismatch("planted: measured on another card")

    monkeypatch.setattr(roofline, "load_measured", mismatch)
    with pytest.raises(SystemExit, match="planted: measured on another card"):
        SE.resolve_cost_ceilings(0.0, 0.0, "NVIDIA Planted-GPU", "bf16")


def test_a_card_s_own_calibration_is_what_a_run_on_that_card_prices_against(
        monkeypatch):
    from moe.bench import roofline

    a100 = roofline.load_hardware("measured_nvidia_a100_sxm4_80gb")
    monkeypatch.setattr(roofline, "load_measured", lambda *a, **k: a100)
    c = SE.resolve_cost_ceilings(0.0, 0.0, "NVIDIA A100-SXM4-80GB", "bf16")
    assert c.kind == "calibration"
    assert c.ridge == pytest.approx(145.8, abs=0.1)
    assert c.bandwidth_gbps == pytest.approx(1799.4, abs=0.1)
    assert "measured on this device" in c.ridge_source
    assert "HYPOTHESIS" not in c.ridge_source


def test_the_dry_run_provenance_carries_the_hypothesis_label(tmp_path):
    import json

    SE.main(["--self-test", "kernel", "--fail-on-world", "--models",
             "mixtral-8x7b", "--out-dir", str(tmp_path)])
    payload = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert "HYPOTHESIS" in payload["ridge_source"]
    assert "COST MODEL ONLY" in payload["ridge_source"]
    # THE CARD'S OWN COMMITTED NUMBER. This pinned 162.8 until 2026-09-09,
    # the H200 ridge ab61e55 superseded with 152.81 measured on the pod.
    from moe.bench import roofline
    hw = roofline.load_hardware(SE.HYPOTHESIS_CALIBRATION)
    assert payload["provenance"]["ridge"] == pytest.approx(
        hw.ridge_point("bf16"), abs=0.1)


# --------------------------------------------------------------------------
# LEVEL is two-sided since 03df2d4, and this consumer reads the side
# --------------------------------------------------------------------------

#: A PLANTED WORLD, not a card. These three numbers are the shape the
#: fifteenth instance of the recurring defect was found on (a memory-shaped
#: cell boosting above the roof's clock, a hungry one sagging below it), and
#: the tests below pass all three sides of every ratio, so what they test is
#: the consumer's arithmetic and not any card's figures.
#:
#: THE REFERENCE IS DELIBERATELY A ROUND NUMBER NO CARD PUBLISHES. It read
#: 1515.0 until 2026-09-09, under a comment calling it "the bf16-GEMM
#: reference the roof was measured at": that was the H200's committed
#: calibration until ab61e55 remeasured the card at 1485 MHz under the 700 W
#: cap on 2026-09-09, so the constant was the superseded live number wearing
#: the name of the current one, and the file that carried it also defines
#: `H200_REFERENCE_MHZ = 1485.0` for the card's real figure. A planted world
#: gets a planted number; a test that needs the card's own clock reads the
#: committed calibration.
PLANTED_REFERENCE_MHZ = 1500.0
#: Well above `PLANTED_REFERENCE_MHZ * timing.LEVEL_HIGH_FRACTION`, the state
#: of every memory-shaped tread the H200 gaps session measured (1950-1980).
H200_MEMORY_LOAD_MHZ = 1980.0
#: Well below `PLANTED_REFERENCE_MHZ * timing.LEVEL_FRACTION`.
SAGGED_MHZ = 1400.0


def _kernel_timing_at(load_mhz, reference_mhz, *, drift_to=None):
    """A `KernelTiming` scored the way `time_kernel` scores one: verdicts from
    the real `clock_flags`, the side from the real `level_side`, and the
    reference on the record. Not hand-set booleans: a hand-set side would
    pass whatever the consumer did with it."""
    from moe.bench import timing
    end = load_mhz if drift_to is None else drift_to
    level, drift = timing.clock_flags(load_mhz, load_mhz, end, reference_mhz)
    return timing.KernelTiming(
        ms_p50=1.0, ms_p90=1.1, ms_min=0.9, ms_std=0.01, iters=100, trials=3,
        warmup_ms=300.0, l2_flush=True, sm_clock_load_mhz=load_mhz,
        sm_clock_start_mhz=load_mhz, sm_clock_end_mhz=end,
        clock_level_ok=level, clock_drift_ok=drift, samples=300,
        warmup_calls=10, flush_mb=256, clock_samples=9, clock_source="injected",
        clock_poll_ms=1.0, host_bound=False, host_enqueue_ms=0.01,
        clock_note="scripted clock",
        clock_level_side=timing.level_side(load_mhz, reference_mhz) or "",
        reference_clock_mhz=reference_mhz)


def test_the_exclusion_rule_is_drift_alone_and_both_level_sides_are_kept():
    """THE RULE AS IT STANDS SINCE 2026-09-09: DRIFT excludes, LEVEL records.

    Until then this asserted "LOW or DRIFT excludes, HIGH is kept". The
    750-cell census of the H200 gaps session showed the LOW side is the steady
    operating point of a hungry tile under the 700 W cap (BLOCK_M=128 at
    BLOCK_N=64 held 1380-1410 MHz in every rep of every arm, BLOCK_M=64 at
    GROUP_SIZE_M=1 1358), so excluding it excluded a tile rather than a
    defect, and removed both of this study's primary tiles from measurability
    on the card.

    `moe.bench.driver`'s `throttled` assignment is the twin of this rule and
    moves in the same commit; the truth table below is written out here rather
    than imported so this file states the rule instead of quoting whatever the
    driver currently does.
    """
    from moe.bench import timing
    ex = SE.clock_excluded
    # Neither side is an exclusion when the clock held still.
    assert ex(False, timing.LEVEL_HIGH, True) is False
    assert ex(False, timing.LEVEL_HIGH, None) is False
    assert ex(False, timing.LEVEL_LOW, True) is False
    assert ex(False, timing.LEVEL_LOW, None) is False
    # A False that recorded no side is the one-sided era's row: still kept.
    assert ex(False, "", True) is False
    # DRIFT excludes whatever LEVEL said, on either side.
    assert ex(True, "", False) is True
    assert ex(False, timing.LEVEL_HIGH, False) is True
    assert ex(False, timing.LEVEL_LOW, False) is True
    assert ex(None, "", False) is True
    # Not determined is not an exclusion: one has to be positively established.
    assert ex(None, "", None) is False
    assert ex(True, "", True) is False
    assert ex(True, "", None) is False
    # The whole table: DRIFT and nothing else.
    for level in (True, False, None):
        for side in ("", timing.LEVEL_LOW, timing.LEVEL_HIGH):
            for drift in (True, False, None):
                assert ex(level, side, drift) is (drift is False), (
                    level, side, drift)


def test_a_boosted_record_reads_high_and_a_sagged_one_reads_low():
    """1980 against 1515 is 1.31x, above `LEVEL_HIGH_FRACTION`; 1400 against
    1515 is 0.92x, below `LEVEL_FRACTION`. Both fail LEVEL, and the side is
    the only thing that tells them apart."""
    from moe.bench import timing
    high = _kernel_timing_at(H200_MEMORY_LOAD_MHZ, PLANTED_REFERENCE_MHZ)
    low = _kernel_timing_at(SAGGED_MHZ, PLANTED_REFERENCE_MHZ)
    assert high.clock_level_ok is False and low.clock_level_ok is False
    assert SE.clock_side_of(high) == timing.LEVEL_HIGH
    assert SE.clock_side_of(low) == timing.LEVEL_LOW
    assert SE.clock_excluded(high.clock_level_ok, SE.clock_side_of(high),
                                 high.clock_drift_ok) is False, "HIGH is kept"
    assert SE.clock_excluded(low.clock_level_ok, SE.clock_side_of(low),
                                 low.clock_drift_ok) is False, (
        "since 2026-09-09 a steady LOW is kept and its side recorded")
    # A record without the field (every fake before 2026-09-03) gets its side
    # derived from its own numbers, the way driver.py derives it.
    import dataclasses
    bare = dataclasses.replace(high, clock_level_side="")
    assert SE.clock_side_of(bare) == timing.LEVEL_HIGH
    # And one with neither answers "", which is "no side recorded".
    blind = dataclasses.replace(bare, reference_clock_mhz=None)
    assert SE.clock_side_of(blind) == ""


def test_only_the_rule_and_the_summary_compare_the_level_verdict_bare():
    """THE SECOND CALL SITE, GUARDED. A `clock_level_ok is False` outside the
    rule and the counting block is a reader that has not learned the side, and
    that is how the fifteenth instance happened: one producer fixed, thirteen
    consumers left on the old meaning.

    A NESTED HELPER IS ITS OUTERMOST FUNCTION, since 2026-09-09. The walk
    attributed a def to its own name whatever it was nested in, so a reader
    written as a closure inside a disallowed function passed under a name that
    was not on the list, and a helper factored out of an allowed one failed
    while doing exactly what the allowed one did. Attributing by ancestor
    makes the list about the block the code lives in, which is what the rule
    is about."""
    import ast
    tree = ast.parse((ROOT / "scripts" / "span_extent_separation.py").read_text())
    owner = {}
    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            for node in ast.walk(top):
                if isinstance(node, ast.FunctionDef):
                    owner[node] = top.name
    readers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Code, not prose: a docstring that names the old test is history.
            body = "\n".join(ast.unparse(s) for s in node.body
                             if not (isinstance(s, ast.Expr)
                                     and isinstance(s.value, ast.Constant)))
            if "clock_level_ok is False" in body:
                readers.add(owner.get(node, node.name))
    allowed = {"clock_excluded", "analyse"}
    assert readers <= allowed, (
        f"{sorted(readers - allowed)} test the LEVEL verdict "
        "without its side; route them through clock_excluded")


def test_the_mirrored_side_constants_are_the_instruments_own():
    """This file imports `timing` lazily and mirrors the two side strings; the
    mirror must be the instrument's, or a row stamped by `time_kernel` would
    be read against a different alphabet."""
    from moe.bench import timing
    assert SE.LEVEL_LOW == timing.LEVEL_LOW
    assert SE.LEVEL_HIGH == timing.LEVEL_HIGH


def test_the_side_round_trips_through_the_csv_and_an_old_row_reads_no_side(tmp_path):
    store = SE.Store(tmp_path / "t.csv", "H200")
    cell = SE.Cell("mixtral-8x7b", 256, "bf16")
    store.write(timed_arm(clock_level_ok=False, clock_level_side="high",
                          sm_clock_load_mhz=1980.0), cell, meta_for())
    store.close()
    rows = list(csv.DictReader((tmp_path / "t.csv").open(newline="")))
    assert rows[0]["clock_level_side"] == "high"
    fresh = SE.Store(tmp_path / "t.csv", "H200")
    back = fresh.restore(("mixtral-8x7b", 256, "gemm_up"))
    fresh.close()
    assert back.clock_level_ok is False and back.clock_level_side == "high"
    assert "clock_level_side" in SE.CSV_COLUMNS
    # A row written before the column existed reads back with no side.
    assert SE.ArmResult("m", 1, "gemm_up").clock_level_side == ""


def test_with_timing_copies_the_side_off_the_instrument():
    from moe.bench import timing
    high = SE.ArmResult("m", 256, "gemm_up").with_timing(
        _kernel_timing_at(H200_MEMORY_LOAD_MHZ, PLANTED_REFERENCE_MHZ))
    low = SE.ArmResult("m", 256, "gemm_up").with_timing(
        _kernel_timing_at(SAGGED_MHZ, PLANTED_REFERENCE_MHZ))
    assert high.clock_level_ok is False and high.clock_level_side == timing.LEVEL_HIGH
    assert low.clock_level_ok is False and low.clock_level_side == timing.LEVEL_LOW


def test_the_representative_repeat_folds_the_side_with_low_dominating():
    """One sagged repeat makes the arm's RECORDED side low whatever the median
    repeat did; an arm whose failures were all boosts keeps that word. Since
    2026-09-09 the fold decides what the row says it ran at, not whether the
    row is kept: DRIFT is the only exclusion and it folds through `worst`."""
    from moe.bench import timing
    high = _kernel_timing_at(H200_MEMORY_LOAD_MHZ, PLANTED_REFERENCE_MHZ)
    low = _kernel_timing_at(SAGGED_MHZ, PLANTED_REFERENCE_MHZ)
    level = _kernel_timing_at(PLANTED_REFERENCE_MHZ, PLANTED_REFERENCE_MHZ)
    boosted = SE.representative_timing([high, high, level])
    assert boosted.clock_level_ok is False
    assert boosted.clock_level_side == timing.LEVEL_HIGH
    mixed = SE.representative_timing([high, low, level])
    assert mixed.clock_level_side == timing.LEVEL_LOW
    assert SE.representative_timing([level, level]).clock_level_side == ""
    # None of those folds is an exclusion; a drifted repeat is.
    for folded in (boosted, mixed):
        assert SE.clock_excluded(folded.clock_level_ok, folded.clock_level_side,
                                 folded.clock_drift_ok) is False
    drifting = _kernel_timing_at(PLANTED_REFERENCE_MHZ,
                                 PLANTED_REFERENCE_MHZ, drift_to=1300.0)
    one_bad = SE.representative_timing([level, drifting, level])
    assert one_bad.clock_drift_ok is False
    assert SE.clock_excluded(one_bad.clock_level_ok, one_bad.clock_level_side,
                             one_bad.clock_drift_ok) is True


def _restamp(results, **flags):
    return {key: {arm: dataclasses.replace(r, **flags) for arm, r in arms.items()}
            for key, arms in results.items()}


def test_both_level_sides_are_counted_apart_and_neither_is_excluded():
    """THE HIGH WORLD: the kernel world with every timed arm at the H200's
    memory-load clock. `clock_level_bad` used to count all of them as "the
    card sat low"; now they are `clock_level_high`, the page names them as
    kept, and the gates read exactly as they do for the unstamped world. The
    LOW twin puts every arm in `clock_level_bad`, and since 2026-09-09 the
    page names those as kept too: a steady clock on either side is the
    operating point the tile held under the power cap."""
    cells, results, plain, plain_gates = run_world("kernel")
    high = SE.analyse(cells, _restamp(results, clock_level_ok=False,
                                      clock_level_side="high"))
    assert high.arms_timed == plain.arms_timed > 0
    assert high.clock_level_high == high.arms_timed
    assert high.clock_level_bad == 0
    low = SE.analyse(cells, _restamp(results, clock_level_ok=False,
                                     clock_level_side="low"))
    assert low.clock_level_bad == low.arms_timed and low.clock_level_high == 0
    # And the one-sided era's row, False with no side, is counted as low.
    old = SE.analyse(cells, _restamp(results, clock_level_ok=False))
    assert old.clock_level_bad == old.arms_timed
    verdicts = lambda gates: [(g.name, g.passed) for g in gates]  # noqa: E731
    high_gates = SE.build_gates(high)
    assert verdicts(high_gates) == verdicts(plain_gates)
    said = "\n".join(str(getattr(g, "observed", "")) + str(getattr(g, "detail", ""))
                      + "\n".join(getattr(g, "lines", []) or []) for g in high_gates)
    assert f"LEVEL high on {high.arms_timed} (steady, kept, side recorded)" in said
    assert "LEVEL low on 0 (steady, kept, side recorded)" in said
    low_gates = SE.build_gates(low)
    low_said = "\n".join(str(getattr(g, "observed", "")) + str(getattr(g, "detail", ""))
                         + "\n".join(getattr(g, "lines", []) or [])
                         for g in low_gates)
    assert verdicts(low_gates) == verdicts(plain_gates)
    assert f"LEVEL low on {low.arms_timed} (steady, kept, side recorded)" in low_said


def test_a_drifted_arm_is_counted_as_drifted_and_not_also_as_its_level_side():
    """THE OVERLAPPING ARM: LEVEL false on the HIGH side AND DRIFT false.

    Both branches of the counting block took every arm with that side until
    2026-09-09, drifted arms included, so one arm landed in `clock_level_high`
    (printed "steady, kept") and in `clock_drift_bad` (printed as the
    exclusion) from a single row, and the two side counts plus the drift count
    came to more than the timed arms. No planted world in this file had such an
    arm, which is why nothing was red. R9: kept counts must not include drifted
    cells."""
    cells, results, plain, _ = run_world("kernel")
    both = SE.analyse(cells, _restamp(results, clock_level_ok=False,
                                      clock_level_side="high",
                                      clock_drift_ok=False))
    assert both.arms_timed == plain.arms_timed > 0
    assert both.clock_drift_bad == both.arms_timed
    assert both.clock_level_high == 0, "a drifted arm is not a steady HIGH"
    assert both.clock_level_bad == 0
    assert both.clock_level_high_drifted == both.arms_timed
    assert both.clock_level_bad_drifted == 0
    assert both.clock_level_ok_drifted == 0
    assert (both.clock_level_bad + both.clock_level_high
            + both.clock_drift_bad) == both.arms_timed
    said = "\n".join(str(getattr(g, "observed", "")) for g in SE.build_gates(both))
    assert "LEVEL high on 0 (steady, kept, side recorded)" in said
    assert (f"DRIFT bad on {both.arms_timed} (excluded, and not in the two "
            "side counts before it") in said
    assert f"{both.arms_timed} HIGH while they moved" in said
    # A level arm that drifted lands in `clock_level_ok_drifted` and in no
    # steady count either.
    level_drift = SE.analyse(cells, _restamp(results, clock_level_ok=True,
                                             clock_level_side="",
                                             clock_drift_ok=False))
    assert level_drift.clock_level_ok_drifted == level_drift.arms_timed
    assert level_drift.clock_level_high == level_drift.clock_level_bad == 0


def test_the_summary_carries_the_high_count_beside_the_low_one():
    """The published summary is built inside `_main`, so the key is pinned in
    the source: a `clock_level_bad_arms` with no `clock_level_high_arms`
    beside it is the one-sided count published again."""
    source = (ROOT / "scripts" / "span_extent_separation.py").read_text()
    assert '"clock_level_bad_arms": analysis.clock_level_bad,' in source
    assert '"clock_level_high_arms": analysis.clock_level_high,' in source
    assert source.index('"clock_level_high_arms"') - source.index('"clock_level_bad_arms"') < 200
    # And the drifted arms' own side breakdown, so a reader of the summary can
    # see the partition the printed line names: the three counts above take
    # the STEADY arms only since 2026-09-09.
    for key in ("clock_level_ok_drifted_arms", "clock_level_bad_drifted_arms",
                "clock_level_high_drifted_arms"):
        assert f'"{key}":' in source, key


def test_the_new_clock_columns_round_trip_through_the_store(tmp_path):
    """R3's evidence columns, through `with_timing`, `row` and `Store.restore`.
    `time_kernel` computed the first and last under-load sample and this
    writer dropped them, so a resumed row that failed DRIFT could not say
    which way its clock went."""
    t = _kernel_timing_at(PLANTED_REFERENCE_MHZ, PLANTED_REFERENCE_MHZ,
                          drift_to=1300.0)
    arm = SE.ArmResult("mixtral-8x7b", 256, "gemm_up").with_timing(t)
    assert (arm.sm_clock_start_mhz, arm.sm_clock_end_mhz) == (
        PLANTED_REFERENCE_MHZ, 1300.0)
    assert arm.clock_samples_mhz == "" and arm.power_w is None
    for column in ("sm_clock_start_mhz", "sm_clock_end_mhz",
                   "clock_samples_mhz", "power_w"):
        assert column in SE.CSV_COLUMNS, column
    store = SE.Store(tmp_path / "t.csv", "H200")
    store.write(dataclasses.replace(arm, ms_median=1.0), SE.Cell(
        "mixtral-8x7b", 256, "bf16"), meta_for())
    store.close()
    fresh = SE.Store(tmp_path / "t.csv", "H200")
    back = fresh.restore(("mixtral-8x7b", 256, "gemm_up"))
    fresh.close()
    assert (back.sm_clock_start_mhz, back.sm_clock_end_mhz) == (
        PLANTED_REFERENCE_MHZ, 1300.0)
    assert back.power_w is None
