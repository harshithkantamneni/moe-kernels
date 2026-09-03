"""The memory branch's LEVEL, and the tests that stop it being invented.

Every alpha this study has published is `B / L`. `B` is a slope over 16 to 33
treads and is identified; `L` is the branch extrapolated to one M-tile and is
not. `scripts/memory_branch_anchor.py` replaces the extrapolation with a bracket
between two measured quantities, and the failure modes these tests defend
against are the ones that would let that bracket be wrong QUIETLY:

  * a missing anchor silently becoming 0.0, which divides into every alpha;
  * the bracket inverting when the ceiling is below the anchor;
  * a run id that omits a swept knob, which is how this repo has already
    reported one arm's timings under another arm's heading;
  * a poisoned-reference check that fires on nothing and reads as a clean bill
    of health;
  * a bracket so wide it contains everything, which is not an improvement on a
    point estimate nobody can defend.

And, since the 2026-09-02 audit, the four that let the GPU arm spend eight
minutes and report a tidy null:

  * a tile pin that never reached the kernel, whose expected PASS is
    indistinguishable from a clean run in every gate that existed before M6;
  * two integers meaning two different things in two files, so a
    measured-and-invalid run was logged REFUSED and a genuine refusal queued as
    a retry;
  * a stream check that ran only on a freshly timed cell, so a fully resumed run
    failed M0 for ever and the arm could never reach DONE;
  * a free re-scoring that rewrote a TRACKED file, with an absolute path in it,
    on every run including the session driver's dry one.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

mba = pytest.importorskip("memory_branch_anchor")

from moe.spec import MODEL_CONFIGS  # noqa: E402

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
A100_CEILING = 1879.1        # the write pattern, the largest the A100 demonstrated
A100_PIN = 2039.0


# --------------------------------------------------------------------------
# The arithmetic. Everything here is exact and has no device in it.
# --------------------------------------------------------------------------

def test_alpha_and_bandwidth_are_inverses():
    """(*) and its inverse have to agree, or the physicality gate is scoring a
    different quantity from the bracket."""
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    alpha = 0.47
    slope = 0.9137
    bw = mba.bandwidth_for_alpha(alpha, slope, w, act1)
    assert mba.alpha_at_bandwidth(slope, bw, w, act1) == pytest.approx(alpha, abs=1e-12)


def test_alpha_is_monotone_in_bandwidth():
    """The whole bracket rests on this: bounding BW bounds alpha, with no search."""
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    values = [mba.alpha_at_bandwidth(0.9, bw, w, act1) for bw in (1200, 1500, 1879.1)]
    assert values == sorted(values)


def test_activation_bytes_matches_the_sweep_it_was_transcribed_from():
    """The local copy exists so a concurrent edit elsewhere cannot move the
    bracket. That is only safe if a divergence is CAUGHT rather than assumed."""
    spec = importlib.util.find_spec("block_m_crossing_sweep")
    if spec is None:
        pytest.skip("the sweep is not importable here")
    sweep = importlib.import_module("block_m_crossing_sweep")
    for name, cfg in MODEL_CONFIGS.items():
        assert mba.activation_bytes_per_row(cfg) == sweep.activation_bytes_per_row(cfg), name


def test_anchor_bytes_scale_with_the_tile_and_not_with_the_weights():
    """One extra M-tile carries `E * BM` more rows of activations; the weight set
    does not change. Mixing those two up is what puts activation traffic inside
    alpha and inflates it."""
    w32, a32 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    w64, a64 = mba.anchor_bytes(MIXTRAL, "bf16", 64)
    assert w32 == w64
    assert a64 == 2 * a32


# --------------------------------------------------------------------------
# Refusals. The study's standing rule: never return 0.0 for something
# unmeasured.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, needle", [
    ({"anchor_ms": 0.0}, "anchor is absent"),
    ({"anchor_ms": -1.0}, "anchor is absent"),
    ({"slope_ms": 0.0}, "alpha is undefined"),
    ({"bw_ceiling_gbps": 0.0}, "no measured bandwidth ceiling"),
    ({"bw_pin_gbps": 0.0}, "no measured bandwidth ceiling"),
])
def test_bracket_refuses_rather_than_defaulting(kwargs, needle):
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    base = dict(slope_ms=0.88, slope_ms_published=0.88, anchor_ms=1.95, block_m=32,
                w_bytes=w, act1_bytes=act1, bw_ceiling_gbps=A100_CEILING,
                bw_pin_gbps=A100_PIN)
    base.update(kwargs)
    with pytest.raises(ValueError) as exc:
        mba.bracket_alpha(**base)
    assert needle in str(exc.value)


def test_ols_refuses_a_single_point_and_a_degenerate_x():
    with pytest.raises(ValueError):
        mba.ols([1.0], [2.0])
    with pytest.raises(ValueError):
        mba.ols([3.0, 3.0], [1.0, 2.0])


def test_both_caps_refuse_a_non_positive_alpha():
    """Zero is the value `lo` takes when the RAW bracket end came out negative
    and hit the clip, and a cap read off a wall is the model's edge reported as
    a measurement. It is also an infinite ceiling in the retracted form, which
    would turn the surviving BLOCK_M <= 64 result into an unconditional PASS."""
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    br = mba.bracket_alpha(0.88, 0.88, 1.95, 32, w, act1, A100_CEILING, A100_PIN)
    for method in (br.ai_cap, br.lin_cap):
        with pytest.raises(ValueError):
            method(2, 0.0)


def test_the_estimator_returns_exa_and_the_retracted_cap_is_high_by_the_level():
    """The identity this file used to pin, run against a FIT instead of against
    itself.

    The test it replaces computed `alpha_fitted = alpha_b + alpha_a (BM/BN) +
    BM/K` and then asserted that `ai_cap` of that equals `ai_model.cap` at the
    same alpha_b and alpha_a. Both sides were the same rearrangement of one
    definition; no estimator was ever called, so the assertion could not fail
    however wrong the reading was, and it certified as exact the (LIN) blend
    that `moe/bench/ai_model.py` retracted on the same day at the same HEAD.

    What is checked here instead: build a byte ladder with `ai_model.traffic()`,
    run THIS FILE'S OWN least squares over it -- the same `ols` that fits every
    bracket -- and read `B / (A + B)` off the result. That is the estimator every
    published alpha came from, applied to bytes whose alpha_b is known. It
    returns (EXA), `(alpha_b + phi) / (1 + phi + delta)`, and NOT the blend. The
    consequence is the one the caps turn on: `2 BM / (b alpha_fitted)` is the
    exact cap times `1 + phi + delta`, which is `ai_model.lin_overstatement`.
    """
    ai_model = pytest.importorskip("moe.bench.ai_model")
    block_m, block_n, b = 128, 64, 2
    n, k = 2 * MIXTRAL.intermediate_size, MIXTRAL.hidden_size
    alpha_b, alpha_a = 0.307, 0.143
    fixed = 0.05 * k * n * b            # a non-zero delta, so the level is not 1
    rungs = ai_model.ladder(n, k, block_m=block_m, block_n=block_n,
                            alpha_b=alpha_b, alpha_a=alpha_a, b=b, n_max=8,
                            fixed_bytes=fixed)
    level, slope = mba.ols([float(i) for i, _ in rungs], [t for _, t in rungs])
    fitted = slope / (level + slope)

    phi = ai_model.phi(n, k, block_m=block_m, block_n=block_n, alpha_a=alpha_a, b=b)
    delta = fixed / (k * n * b)
    assert fitted == pytest.approx((alpha_b + phi) / (1.0 + phi + delta), rel=1e-9)
    assert fitted != pytest.approx(
        ai_model.lin_blend(k, block_m=block_m, block_n=block_n,
                           alpha_b=alpha_b, alpha_a=alpha_a), rel=1e-3)

    exact = ai_model.exact_cap(n, k, block_m=block_m, block_n=block_n,
                               alpha_b=alpha_b, alpha_a=alpha_a, b=b)
    retracted = 2.0 * block_m / (b * fitted)
    assert retracted == pytest.approx(
        exact * ai_model.lin_overstatement(phi=phi, delta=delta), rel=1e-9)
    assert ai_model.cap_from_fitted(fitted, block_m=block_m, b=b, phi=phi,
                                    delta=delta) == pytest.approx(exact, rel=1e-9)


def test_the_bracket_cap_puts_the_activation_term_back_and_is_lower_than_the_retracted_one():
    """`Bracket.ai_cap` is `ai_model.exact_cap`'s form on the fused layer.

    The bracket's alpha is an alpha_b: (*) is solved on the slope with `Act1`
    already subtracted. So the retracted `2 BM / (b alpha_b)` leaves `phi` out of
    the denominator entirely and is high by `(alpha_b + phi) / alpha_b`, which is
    the ratio this asserts. Both caps are exercised at a real bracket end rather
    than at a literal, and the direction is the one the C3 verdict depends on:
    correcting the cap can only LOWER it, so it can only make "the tile caps
    below the ridge" easier to keep.
    """
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    br = mba.bracket_alpha(0.88, 0.88, 1.95, 32, w, act1, A100_CEILING, A100_PIN)
    assert br.phi == pytest.approx(act1 / w)
    assert br.ai_cap(2, br.lo) == pytest.approx(
        2.0 * 32 / (2 * (br.lo + br.phi)), rel=1e-9)
    assert br.lin_cap(2, br.lo) / br.ai_cap(2, br.lo) == pytest.approx(
        (br.lo + br.phi) / br.lo, rel=1e-9)
    assert br.ai_cap(2, br.lo) < br.lin_cap(2, br.lo)


def test_the_corrected_cap_equals_the_retracted_form_at_the_uncorrected_alpha():
    """Subtracting `Act1` from the slope and then adding `phi` back is the
    identity, so the exact cap is what the study's own expression would have
    given had it never applied the `alpha-corrected` column to a ceiling.

    Worth pinning because it says exactly where the published caps went wrong:
    not in the formula's shape, but in feeding it an alpha from which the very
    term the formula needs had already been removed.
    """
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    br = mba.bracket_alpha(0.88, 0.88, 1.95, 32, w, act1, A100_CEILING, A100_PIN)
    uncorrected = mba.alpha_at_bandwidth(0.88, br.bw_anchor_gbps, w, 0)
    assert br.lo < uncorrected
    assert br.ai_cap(2, br.lo) == pytest.approx(br.lin_cap(2, uncorrected), rel=1e-12)


def test_load_calibration_refuses_an_unknown_card(tmp_path):
    """A spec sheet is a pin rate, not an achieved rate. Falling back to one is
    the defect that stamped a stale H200 ridge on seven A100 reports."""
    with pytest.raises(FileNotFoundError):
        mba.load_calibration("nvidia_not_a_real_card", directory=tmp_path)


# --------------------------------------------------------------------------
# Planted ladders: does the bracket cover the truth, and does it catch the
# pathology.
# --------------------------------------------------------------------------

def planted_ladder(alpha: float, bw_anchor: float, bw_branch: float,
                   fixed_ms: float, block_m: int = 32, treads: int = 33):
    """`(points, W, Act1)` around the script's own `plant_ladder`.

    THE GENERATOR IS THE SCRIPT'S, not a second one here. `--self-test` scores
    ten planted worlds built with it, so a plant that drifted from the one the
    tests use would leave the self-test asserting things about a world no test
    ever sees. This wrapper exists only to keep the bandwidths in GB/s at the
    call sites that predate the move, and to hand back the two byte counts the
    arithmetic tests compare against.
    """
    points = mba.plant_ladder(alpha, bw_anchor_gbps=bw_anchor / 1e9,
                              bw_branch_gbps=bw_branch / 1e9, fixed_ms=fixed_ms,
                              cfg=MIXTRAL, block_m=block_m, treads=treads)
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", block_m)
    return points, w, act1


def test_bracket_covers_a_planted_alpha_and_does_not_cry_wolf():
    """A plant that OBEYS the model: the bracket must contain the planted value,
    and it must also contain the published-style point estimate. A bracket that
    flagged a consistent fit would flag everything."""
    alpha = 0.558
    pts, w, act1 = planted_ladder(alpha, bw_anchor=1450e9, bw_branch=1750e9,
                                  fixed_ms=0.05)
    xs = [float(n) for n, _ in pts]
    ys = [ms for _, ms in pts]
    a_full, b_full = mba.ols(xs, ys)
    _, b_free = mba.ols(xs[1:], ys[1:])
    br = mba.bracket_alpha(b_free, b_full, pts[0][1], 32, w, act1,
                           A100_CEILING, A100_PIN)
    assert br.contains(alpha), (br.lo, br.hi)
    assert br.contains(b_full / (a_full + b_full))
    # The interval has to be informative, not a tautology.
    assert br.width < 0.25


def test_dropping_the_anchor_barely_moves_the_slope():
    """The bracket's two ends must be independent. If the slope moved when the
    anchor left the fit, `B / t(1)` would be partly a restatement of `t(1)`."""
    pts, _, _ = planted_ladder(0.558, bw_anchor=1450e9, bw_branch=1750e9,
                               fixed_ms=0.05)
    xs = [float(n) for n, _ in pts]
    ys = [ms for _, ms in pts]
    _, b_full = mba.ols(xs, ys)
    _, b_free = mba.ols(xs[1:], ys[1:])
    assert abs(b_free / b_full - 1.0) <= mba.SLOPE_INDEPENDENCE_REL


def test_an_impossible_branch_level_is_caught_and_lands_outside_the_bracket():
    """The A100 mixtral G=16 BLOCK_M=32 cell, with its committed numbers.

    Its fitted level moves the weight set at more than the memory bus can carry,
    so its published alpha is not uncertain: it is impossible. Both the
    physicality check and the bracket have to say so.
    """
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    slope_published = 0.8807539377301772
    level = 1.3606                       # A + B from the committed report
    anchor = 1.9475                      # the MEASURED n=1 tread
    alpha_published = slope_published / level
    implied = mba.bandwidth_for_alpha(alpha_published, slope_published, w, act1)
    assert implied > A100_PIN, implied
    br = mba.bracket_alpha(0.8843, slope_published, anchor, 32, w, act1,
                           A100_CEILING, A100_PIN)
    assert not br.contains(alpha_published)
    assert br.hi < alpha_published
    assert mba.why_outside(_fit_stub(alpha_published, br)).startswith("ABOVE:")


def _fit_stub(alpha_corrected: float, br) -> mba.ScoredFit:
    """The minimum a direction check needs. Built explicitly rather than with a
    default-filled constructor so a new field cannot silently arrive as zero."""
    return mba.ScoredFit(
        arm="stub", card="nvidia_a100_sxm4_80gb", model="mixtral-8x7b", dtype="bf16",
        group_m=16, block_n=64, num_stages=3, block_m=32, treads=33,
        alpha_published=alpha_corrected, alpha_published_corrected=alpha_corrected,
        slope_published=br.slope_ms_published, slope_refit_full=br.slope_ms,
        slope_refit_no_anchor=br.slope_ms, anchor_ms=br.anchor_ms,
        fitted_level_ms=1.0, anchor_elevation=0.0, bw_anchor_gbps=br.bw_anchor_gbps,
        bw_published_gbps=0.0, bw_ceiling_gbps=br.bw_ceiling_gbps,
        bw_pin_gbps=br.bw_pin_gbps, alpha_lo=br.lo, alpha_hi=br.hi,
        alpha_hi_pin=br.hi_pin, clipped=br.clipped, contains_published=False,
        contains_published_corrected=False, contains_pooled=False,
        physical_vs_ceiling=False, physical_vs_pin=False, ridge=145.81,
        cap_over_ridge_at_lo=0.0, phi=br.phi, lin_over_ridge_at_lo=0.0,
        timing_spread=0.005, elevation_in_spreads=0.0)


def test_assumption_a_gate_passes_on_noise_and_fails_on_a_real_inversion():
    """The bracket's LOWER end is the one thing here that is assumed rather than
    measured: the branch does not run slower than its own anchor. An anchor a
    little BELOW the fitted branch is the anchor sitting ON it; a long way below
    would delete the lower end of every interval on that card."""
    noise = [_fit_with_elevation(-0.031, spread=0.0182)]
    real = [_fit_with_elevation(-0.120, spread=0.0050)]
    assert mba.gate_v6_assumption_a(noise).verdict == mba.PASS
    gate = mba.gate_v6_assumption_a(real)
    assert gate.verdict == mba.FAIL
    assert gate.kind == "VALIDITY"
    assert "LOWER end" in gate.invalidates


def test_assumption_a_gate_says_so_when_no_anchor_sits_below():
    gate = mba.gate_v6_assumption_a([_fit_with_elevation(0.30, spread=0.005)])
    assert gate.verdict == mba.PASS
    assert "0 of 1" in gate.measured


def _fit_with_elevation(elevation: float, spread: float,
                        alpha: float = 0.6) -> mba.ScoredFit:
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    br = mba.bracket_alpha(0.88, 0.88, 1.95, 32, w, act1, A100_CEILING, A100_PIN)
    fit = _fit_stub(alpha, br)
    fit.anchor_elevation = elevation
    fit.timing_spread = spread
    fit.elevation_in_spreads = elevation / spread
    return fit


def test_a_below_bracket_miss_inside_the_noise_is_named_as_tightness_not_error():
    """Six committed fits sit just under their bracket. Calling those the same
    defect as an alpha that needs 114% of the memory bus would be wrong, and a
    reader who was told they were the same would stop believing the list.

    The exculpation is available ONLY when the anchor sits BELOW its own branch,
    because that is the condition that strains ASSUMPTION A and so inflates
    `alpha_lo`. Past the noise band the same negative elevation stops being an
    excuse and becomes a refusal to quote `alpha_lo` at that cell at all.
    """
    fit = _fit_with_elevation(-0.005, spread=0.010, alpha=0.30)
    assert fit.alpha_published_corrected < fit.alpha_lo     # a BELOW miss
    said = mba.why_outside(fit)
    assert "interval being tight" in said
    assert "BELOW its own branch" in said
    far = mba.why_outside(_fit_with_elevation(-0.20, spread=0.005, alpha=0.30))
    assert "ASSUMPTION A FAILS" in far
    assert "interval being tight" not in far


def test_a_below_bracket_miss_with_the_anchor_ABOVE_the_branch_is_not_excused():
    """THE BUG THIS PINS SHIPPED IN THE COMMITTED --rescore OUTPUT.

    `abs(elevation_in_spreads)` handed the "the anchor lies ON the branch"
    exculpation to three C2 failures whose anchors sat comfortably ABOVE their
    branches (+0.21, +0.80, +1.85 spreads), and the fall-through then told a
    +3.51-spread fit that its n=1 tread was FASTER than the fitted level -- two
    lines under a table printing +1.69% for the same fit. C2 is a CLAIM gate and
    twelve published point estimates are withdrawn on its strength, so the
    per-row reading is the only thing telling anyone WHICH of the twelve are
    genuine bound violations. Four of the twelve carried a wrong-signed caption.

    A positive elevation means ASSUMPTION A is comfortably satisfied, so
    `alpha_lo` stands and the published point really is under its own bound.
    """
    inside_band = mba.why_outside(_fit_with_elevation(0.004, spread=0.010,
                                                      alpha=0.30))
    assert "ABOVE its own branch" in inside_band
    assert "the lower end stands" in inside_band
    assert "interval being tight" not in inside_band
    far_above = mba.why_outside(_fit_with_elevation(0.20, spread=0.005,
                                                    alpha=0.30))
    assert "ABOVE its own branch" in far_above
    assert "FASTER" not in far_above


def test_a_below_bracket_caption_quotes_the_miss_in_alpha_not_the_elevation():
    """"BELOW by 0.21 spreads" beside "published 0.687 vs bracket [0.702, ...]"
    reads as a bracket miss of 0.21 spreads, and it is not that quantity at all.
    C2 scored the distance from the published alpha to `alpha_lo`, in alpha."""
    fit = _fit_with_elevation(0.004, spread=0.010, alpha=0.30)
    said = mba.why_outside(fit)
    assert f"BELOW alpha_lo by {fit.alpha_lo - fit.alpha_published_corrected:.3f}" in said


def test_alpha_above_one_is_named_as_its_own_defect():
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    br = mba.bracket_alpha(1.68, 1.68, 1.9263, 32, w, act1, A100_CEILING, A100_PIN)
    assert mba.why_outside(_fit_stub(1.009, br)).startswith("ABOVE ROOF")


def test_the_bracket_clips_to_the_unit_interval_but_keeps_the_raw_end():
    """alpha > 1 is not a fraction of a weight read. Keeping the raw value is
    what lets a reader see that the branch cannot be running at the ceiling."""
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    br = mba.bracket_alpha(1.68, 1.68, 1.9263, 32, w, act1, A100_CEILING, A100_PIN)
    assert br.raw_hi > 1.0
    assert br.hi == 1.0
    assert br.clipped


# --------------------------------------------------------------------------
# The poisoned compute reference. A detector that fires on nothing reports zero
# failures whether or not there are any.
# --------------------------------------------------------------------------

def test_poisoned_reference_separates_the_known_corrupt_arm_from_the_healthy_one():
    """Both numbers are from the committed reports on the same card.

    5.697 ms per tile at BLOCK_M=256 is the healthy BN=64 twin; 248.370 ms is
    the BN=256 arm's reference for the IDENTICAL setting. The sweep's own
    qualification cleared the second at a 0.2% proportionality residual, because
    a line 44x too steep is still perfectly proportional.
    """
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    healthy = mba.implied_reference_tflops(cfg, 256, 5.6969045718510944)
    poisoned = mba.implied_reference_tflops(cfg, 256, 248.37043100992838)
    a100_dense = 262.3712016979615
    assert healthy > mba.POISONED_REFERENCE_FRACTION * a100_dense
    assert poisoned < mba.POISONED_REFERENCE_FRACTION * a100_dense
    assert healthy / poisoned == pytest.approx(248.37043100992838 / 5.6969045718510944,
                                               rel=1e-9)


def test_the_poisoned_reference_gate_fails_when_it_fires_on_nothing():
    """NON-VACUITY, as a gate rather than as a hope."""
    gate = mba.gate_v4_poisoned_reference([], expected_arms=2)
    assert gate.verdict == mba.FAIL
    assert gate.kind == "VALIDITY"


# --------------------------------------------------------------------------
# Plumbing that has cost this repo a run before.
# --------------------------------------------------------------------------

def test_calibration_slug_takes_the_longest_match():
    slugs = ["nvidia_h200", "nvidia_a100_sxm4_80gb", "nvidia_a100"]
    got = mba.calibration_slug_for("2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3",
                                   slugs)
    assert got == "nvidia_a100_sxm4_80gb"
    assert mba.calibration_slug_for("2026-08-22-first-smoke", slugs) is None


def _plan(**over) -> mba.MeasurePlan:
    base = dict(card="nvidia_h200", model="qwen2-57b-a14b", dtype="bf16",
                block_sizes=(32, 64),
                group_sizes=(1, 8, 16, 64), slope_tiles=(2, 3, 4), block_n=64,
                block_k=64, num_warps=8, num_stages=3, seed=0, warmup_ms=300.0,
                cell_budget_ms=400.0, trials=3, l2_flush=True)
    base.update(over)
    return mba.MeasurePlan(**base)


@pytest.mark.parametrize("field, value", [
    ("card", "nvidia_a100_sxm4_80gb"),
    ("model", "mixtral-8x7b"), ("dtype", "fp8_w8a8"), ("block_sizes", (32,)),
    ("group_sizes", (1,)), ("slope_tiles", (2, 3)), ("block_n", 256),
    ("block_k", 128), ("num_warps", 4), ("num_stages", 4), ("seed", 1),
    ("warmup_ms", 301.0), ("cell_budget_ms", 401.0), ("trials", 4),
    ("l2_flush", False),
])
def test_run_id_changes_when_any_swept_knob_changes(field, value):
    """Two settings that share an id collide, and the second silently reports
    the first's numbers. This repo has already lost a G=16 arm that way."""
    assert _plan().run_id() != _plan(**{field: value}).run_id()


def test_the_card_is_in_the_visible_name_not_only_the_hash():
    """THE COLLISION THIS FIELD CLOSES IS ALREADY REAL IN THIS REPO.

    The A100 and H200 cross-card arms are committed under IDENTICAL filenames
    because the sweep's run id omitted the GPU. Here the results root is a
    RunPod network volume shared between pods, and every bracket's upper end is
    a per-card ceiling, so two cards deriving one directory would have the
    second resume, skip every cell and publish the first card's timings under
    its own calibration. The card has to be readable in `ls`, not only hashed:
    a hash nobody can invert makes two runs indistinguishable on sight.
    """
    a100 = _plan(card="nvidia_a100_sxm4_80gb").run_id()
    h200 = _plan(card="nvidia_h200").run_id()
    assert a100.startswith("nvidia_a100_sxm4_80gb-")
    assert h200.startswith("nvidia_h200-")
    assert a100 != h200


def test_run_id_names_the_knobs_it_hashes():
    """A hash nobody can invert makes two runs indistinguishable in `ls`."""
    rid = _plan().run_id()
    for token in ("nvidia_h200", "qwen2_57b_a14b", "bf16", "bm32_64",
                  "g1_8_16_64", "n64", "s3"):
        assert token in rid, rid


def test_every_plan_field_is_in_the_run_id_key():
    """THE FAILURE THIS GUARDS IS A KNOB ADDED TO THE PLAN AND NOT TO THE ID.

    That is not hypothetical here: `--group-m` existed before it was in the id,
    and a G=16 run derived the G=1 directory, resumed into it, skipped every
    cell as already measured and published G=1's timings under a G=16 heading.
    `MeasurePlan.run_id` raises KeyError on an unmapped field for that reason,
    and this test is what makes the KeyError arrive in CI rather than on a pod.
    """
    from dataclasses import fields as dc_fields
    named = {f.name for f in dc_fields(mba.MeasurePlan)} - {"card"}
    assert named == set(mba.ID_KNOBS), named ^ set(mba.ID_KNOBS)


def test_a_dry_run_with_no_device_says_its_path_is_not_the_pods(capsys):
    """A dry run on a laptop must not print a path a pod will never write to.

    The placeholder card is visible in the id AND called out in words, because
    the operator's next move is to check `git check-ignore` on that exact path.
    """
    assert mba.main(["--measure", "--dry-run"]) == mba.exit_codes.REFUSED
    out = capsys.readouterr().out
    assert f"{mba.UNKNOWN_CARD_SLUG}-" in out
    assert "NO DEVICE ATTACHED" in out


def test_card_flag_prints_the_pods_real_path_from_a_laptop(capsys):
    rc = mba.main(["--measure", "--dry-run", "--card", "nvidia_h200"])
    assert rc == mba.exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "nvidia_h200-" in out
    assert mba.UNKNOWN_CARD_SLUG not in out


def test_slope_tiles_below_two_are_refused_at_the_cli(capsys):
    """The anchor may not be inside the slope it is compared against.

    REFUSED (2), not `SystemExit` (1). A bad flag measures nothing, and 1 in the
    shared table is CLAIM_FAIL: "measured, and a pre-registered claim was
    refuted". A mistyped argument is not a finding about the world.
    """
    dense = ",".join(str(n) for n in range(1, 17))
    rc = mba.main(["--measure", "--dry-run", "--slope-tiles", dense])
    assert rc == mba.exit_codes.REFUSED
    assert "must all be >= 2" in capsys.readouterr().out


def test_a_short_branch_is_refused_because_p3_cannot_be_scored_on_it(capsys):
    """P3's 1.5% threshold came from 16- and 33-tread ladders. On 8 treads the
    slope moves 3.2% for a reason that is about the grid, and a threshold that
    fails for the wrong reason teaches a reader to ignore it."""
    rc = mba.main(["--measure", "--dry-run", "--slope-tiles", "2,3,4,6,8,12,16"])
    assert rc == mba.exit_codes.REFUSED
    assert "branch treads" in capsys.readouterr().out


def test_dry_run_measures_nothing_and_says_so():
    assert mba.main(["--measure", "--dry-run"]) == mba.exit_codes.REFUSED


# --------------------------------------------------------------------------
# The GPU arm's verdict path, exercised with no GPU.
# --------------------------------------------------------------------------

def _calibration() -> mba.Calibration:
    """The script's own planted calibration, not a second copy of it.

    `--self-test` scores ten worlds against `SELF_TEST_CALIBRATION`; a fixture
    here with the same numbers typed again would drift the day one of them
    changed, and the tests would then be checking a card the self-test does not
    use.
    """
    return mba.SELF_TEST_CALIBRATION


def _cells(alpha=0.558, bw_anchor=1450e9, bw_branch=1750e9, fixed_ms=0.05,
           groups=(1, 16), block_m=32, treads=16, pin="ok"):
    """A measured grid, through the script's own planter. Bandwidths in bytes/s
    at the call sites that predate the move to GB/s."""
    return mba.plant_cells(
        MIXTRAL, alpha=alpha,
        anchor_bw_by_g={g: bw_anchor / 1e9 for g in groups},
        bw_branch_gbps=bw_branch / 1e9, fixed_ms=fixed_ms, block_m=block_m,
        treads=treads, pin=pin)


def test_measured_cells_recover_the_planted_alpha_inside_the_bracket():
    fits, refusals = mba.fits_from_cells(_cells(), MIXTRAL, "bf16", 64, _calibration())
    assert not refusals
    assert len(fits) == 2
    for f in fits:
        assert f.alpha_lo <= 0.558 <= f.alpha_hi
        assert f.slope_shift <= mba.SLOPE_INDEPENDENCE_REL


def test_a_missing_anchor_cell_is_refused_not_defaulted():
    rows = [c for c in _cells() if c["tiles"] != 1]
    fits, refusals = mba.fits_from_cells(rows, MIXTRAL, "bf16", 64, _calibration())
    assert not fits
    assert refusals and all("nothing to anchor" in r.reason for r in refusals)


def test_too_few_branch_treads_is_refused():
    rows = [c for c in _cells() if c["tiles"] <= 3]   # anchor plus two treads
    fits, refusals = mba.fits_from_cells(rows, MIXTRAL, "bf16", 64, _calibration())
    assert not fits
    assert refusals and all("independent of the anchor needs" in r.reason
                            for r in refusals)


def test_a_failed_cell_is_never_timed_into_a_fit():
    rows = _cells()
    for row in rows:
        if row["tiles"] == 4:
            row.update(status="failed", ms_p50=0.0, detail="OOM")
    fits, _ = mba.fits_from_cells(rows, MIXTRAL, "bf16", 64, _calibration())
    assert all(4 not in dict(f.residuals) for f in fits)


def test_completeness_gate_fails_on_a_short_run():
    rows = _cells()
    gate = mba.gate_m5_completeness(rows, planned=len(rows) + 1)
    assert gate.verdict == mba.FAIL
    assert mba.gate_m5_completeness([], planned=0).verdict == mba.FAIL


def test_stream_gate_fails_when_the_ceiling_is_below_the_data():
    cal = _calibration()
    assert mba.gate_m0_stream({"gbps": cal.ceiling_gbps + 1.0}, cal).verdict == mba.FAIL
    assert mba.gate_m0_stream({"gbps": cal.ceiling_gbps - 1.0}, cal).verdict == mba.PASS
    assert mba.gate_m0_stream(None, cal).verdict == mba.FAIL


def test_anchor_invariance_gate_fails_on_a_swizzle_dependent_anchor():
    """P1's whole point: if t(1) moves with GROUP_SIZE_M, it is not a
    condition-free bound and the bracket's top end has to widen."""
    rows = mba.plant_cells(MIXTRAL, anchor_bw_by_g={1: 1450.0, 16: 1000.0})
    fits, _ = mba.fits_from_cells(rows, MIXTRAL, "bf16", 64, _calibration())
    assert mba.gate_m1_anchor_invariance(fits).verdict == mba.FAIL
    ok, _ = mba.fits_from_cells(_cells(), MIXTRAL, "bf16", 64, _calibration())
    assert mba.gate_m1_anchor_invariance(ok).verdict == mba.PASS


def test_score_measured_runs_every_gate_and_refuses_an_empty_grid():
    cal = _calibration()
    rows = _cells()
    fits, refusals, gates, lines = mba.score_measured(
        rows, MIXTRAL, "bf16", 64, cal, {"gbps": 1500.0}, planned=len(rows))
    assert {g.number for g in gates} == {"M0", "M1", "M2", "M3", "M4", "M5", "M6"}
    assert fits and lines
    _, _, empty_gates, _ = mba.score_measured([], MIXTRAL, "bf16", 64, cal, None, 0)
    assert all(g.verdict == mba.FAIL for g in empty_gates
               if g.number in {"M0", "M3", "M4", "M5", "M6"})


def test_score_measured_round_trips_through_a_written_file(tmp_path, capsys):
    """The pod writes, the laptop scores. A verdict path that only runs on a
    rented GPU is a verdict path nobody tests."""
    cal = _calibration()
    # MIXTRAL, because the cells are planted from mixtral's byte counts. Scored
    # under another model's config the anchor rate lands at 87% of pin and M2
    # FAILs -- which the old assertion (`rc in (0, 1)`) accepted, so the test
    # passed for a run in which the plan and the cells described different
    # layers.
    plan = _plan(model="mixtral-8x7b", block_sizes=(32,), group_sizes=(1, 16),
                 slope_tiles=tuple(range(2, 17)))
    rows = _cells()
    payload = {"plan": {**plan.__dict__, "block_sizes": list(plan.block_sizes),
                        "group_sizes": list(plan.group_sizes),
                        "slope_tiles": list(plan.slope_tiles)},
               "gpu": "NVIDIA A100-SXM4-80GB", "calibration": cal.__dict__,
               "stream_check": {"gbps": 1500.0}, "cells": rows}
    path = tmp_path / "measure.json"
    path.write_text(json.dumps(payload))
    rc = mba.main(["--score-measured", str(path)])
    assert rc == mba.exit_codes.DONE
    out = capsys.readouterr().out
    assert "MEASURED BRACKETS" in out
    assert "GATE M1" in out


def test_score_measured_refuses_a_missing_file(tmp_path):
    rc = mba.main(["--score-measured", str(tmp_path / "nope.json")])
    assert rc == mba.exit_codes.REFUSED


# --------------------------------------------------------------------------
# End to end over the committed reports. Structural assertions, not a golden
# file: another arm may be published tomorrow and that must not break the
# suite, but the FINDING must break it if it goes away.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored():
    if not (REPO / "results" / "published").is_dir():
        pytest.skip("no committed reports in this checkout")
    fits, refusals, cals = mba.scan_published(REPO / "results" / "published")
    if not fits:
        pytest.skip("no anchorable ladders in this checkout")
    return fits, refusals, cals


def test_every_validity_gate_passes_on_the_committed_reports(scored):
    fits, refusals, _ = scored
    gates = [mba.gate_v1_non_vacuity(fits, refusals),
             mba.gate_v2_slope_reproduction(fits),
             mba.gate_v3_anchor_present(fits, refusals),
             mba.gate_v4_poisoned_reference(refusals, 2),
             mba.gate_v5_bracket_order(fits),
             mba.gate_v6_assumption_a(fits)]
    bad = [g for g in gates if g.verdict != mba.PASS]
    assert not bad, [(g.number, g.measured) for g in bad]


def test_the_refit_reproduces_every_published_slope(scored):
    """If this ever fails, the correction table is comparing this script's line
    against an alpha computed from a different one."""
    fits, _, _ = scored
    for f in fits:
        assert f.slope_refit_full == pytest.approx(f.slope_published, rel=1e-9)


def test_every_bracket_is_ordered_and_inside_the_unit_interval(scored):
    fits, _, _ = scored
    for f in fits:
        assert 0.0 <= f.alpha_lo <= f.alpha_hi <= 1.0, f


def test_every_scored_fit_carries_its_arm_s_own_timing_spread(scored):
    """An elevation read against zero instead of against the noise it was
    measured through is how a 1.8%-spread arm and a 0.4%-spread arm get the same
    verdict."""
    fits, _, _ = scored
    for f in fits:
        assert f.timing_spread > 0, (f.arm, f.model)
        assert f.elevation_in_spreads == pytest.approx(
            f.anchor_elevation / f.timing_spread, rel=1e-9)


def test_the_anchor_stands_above_the_fitted_level_on_the_a100(scored):
    """The observation the whole file exists for. If it stops being true, the
    extrapolation is fine and this script is unnecessary -- which is a result
    worth failing a test over."""
    fits, _, _ = scored
    a100 = [f for f in fits if f.card.startswith("nvidia_a100")]
    assert a100, "no A100 fits in this checkout"
    assert all(f.anchor_elevation > 0 for f in a100), \
        [(f.model, f.group_m, f.block_m, f.anchor_elevation) for f in a100
         if f.anchor_elevation <= 0]


def test_some_published_alpha_implies_a_bandwidth_the_card_does_not_have(scored):
    """The finding: those alphas are impossible, not uncertain."""
    fits, _, _ = scored
    impossible = [f for f in fits if not f.physical_vs_pin]
    assert impossible, "no published alpha exceeds its card's pin rate any more"
    for f in impossible:
        assert f.bw_published_gbps > f.bw_pin_gbps


def test_the_block_m_cap_survives_the_most_generous_anchor(scored):
    """The one result the adversarial evaluation left standing. Scored at the
    bracket's LOW alpha, which is the best case for a small tile reaching its
    roof."""
    fits, _, _ = scored
    gate = mba.gate_c3_tile_cap(fits)
    assert gate.verdict == mba.PASS, gate.measured
    assert all(f.cap_over_ridge_at_lo < 1.0 for f in fits if f.block_m <= 64)


def test_the_correction_is_larger_than_the_precision_alpha_is_quoted_at(scored):
    """C5. If this passed, the anchor would be a footnote and the published
    three-decimal alphas could stand."""
    fits, _, _ = scored
    assert mba.gate_c5_correction_size(fits).verdict == mba.FAIL


def test_the_bn256_arms_are_refused_by_the_level_check(scored):
    _, refusals, _ = scored
    poisoned = [r for r in refusals if "compute reference" in r.reason]
    assert len(poisoned) >= 2
    assert all(r.block_n == 256 for r in poisoned), [r.block_n for r in poisoned]


def test_rescore_writes_where_git_will_keep_it(tmp_path, scored):
    """`results/*` is ignored with only `!results/published/` excepted, and this
    repo has already lost every figure of ten arms to a rule that matched at a
    depth nobody checked."""
    assert mba.git_ignored(REPO / "results" / "published" / "x" / "ANCHOR.txt") is not True
    assert mba.git_ignored(REPO / "results" / "scratch" / "ANCHOR.txt") is not False


def test_git_ignored_says_UNKNOWN_for_a_path_outside_the_work_tree(tmp_path):
    """`git check-ignore` exits 128 outside the repo, and 128 is not "no".

    Every path a pod writes is outside the work tree: --measure lands in
    $MOE_RESULTS_DIR, a RunPod network volume at /workspace. Collapsing 128 into
    False printed those as "tracked path", which is the opposite of true --
    nothing there enters the repo without publish_results.sh, and an operator
    who read "tracked" would skip the publish and lose the arm on teardown.
    """
    assert mba.git_ignored(Path("/definitely-not-in-this-repo/measure.json")) is None
    # ... while the two answers git CAN give still come back as booleans.
    assert mba.git_ignored(mba.PUBLISHED / "ANCHOR_RESCORE.json") is False
    assert mba.git_ignored(mba.REPO / "results" / "scratch" / "x.json") is True


# --------------------------------------------------------------------------
# THE PIN ASSAY (audit A10). Until 2026-09-02 the GPU arm entered
# `override_config` and assumed it took. The refuter fed this file's scorer 128
# synthetic cells carrying the failed-override signature -- one anchor and one
# slope, repeated at G=1, 8, 16 and 64 -- and every gate returned PASS with exit
# 0. These tests hold that world at INVALID, and hold the reason it used to pass
# (M0-M5 cannot tell it from a clean run) in front of the reader.
# --------------------------------------------------------------------------

def _score(cells, stream=None, planned=None):
    stream = stream if stream is not None else {"gbps": 1500.0}
    planned = len(cells) if planned is None else planned
    fits, refusals, gates, _ = mba.score_measured(
        cells, MIXTRAL, "bf16", 64, _calibration(), stream, planned)
    return fits, refusals, {g.number: g for g in gates}


def test_the_failed_pin_signature_now_voids_the_run():
    """THE AUDIT'S EXECUTED WORLD, and the argument for M6 in one assertion.

    The cells are numerically IDENTICAL to a clean run -- that is what a failed
    override produces, one kernel measured at every setting -- so M0 through M5
    all PASS on them, exactly as they did when the refuter ran this. Only the
    tile vLLM handed the kernel separates the two, and only M6 reads it.
    """
    _, _, gates = _score(mba.plant_cells(MIXTRAL, pin="default_tile"))
    assert gates["M6"].verdict == mba.FAIL
    assert gates["M6"].kind == mba.VALIDITY
    assert [n for n in ("M0", "M1", "M2", "M3", "M4", "M5")
            if gates[n].verdict != mba.PASS] == []
    rc = mba.exit_codes.classify(g.scored() for g in gates.values())
    assert rc == mba.exit_codes.INVALID


def test_a_cell_with_no_pin_record_is_scored_as_unpinned_not_as_pinned():
    """A cell that cannot show its tile is not a smaller failure than one that
    shows the wrong tile: both leave a row CLAIMING a tile it cannot evidence,
    which is worse than an honest unpinned row."""
    _, _, gates = _score(mba.plant_cells(MIXTRAL, pin="missing"))
    assert gates["M6"].verdict == mba.FAIL
    assert "no pin assay recorded" in " ".join(gates["M6"].lines)


def test_the_right_tile_with_a_warm_cache_still_fails_the_pin_gate():
    """The second leg. A cache serving a previous run compiles nothing, and an
    override that changed no constant looks the same; both are fatal in the same
    way, which is why one gate carries both counts."""
    _, _, gates = _score(mba.plant_cells(MIXTRAL, pin="warm_cache"))
    assert gates["M6"].verdict == mba.FAIL
    assert "compiled nothing new" in " ".join(gates["M6"].lines)


def test_the_pin_gate_passes_when_both_legs_do():
    """The gate has to be able to PASS, or it is a refusal wearing a gate's
    clothes and the arm can never reach DONE."""
    _, _, gates = _score(mba.plant_cells(MIXTRAL, pin="ok"))
    assert gates["M6"].verdict == mba.PASS


def test_the_pin_gate_fails_when_nothing_was_measured():
    """NON-VACUITY. An empty grid assayed nothing, and a check that examined
    nothing reports zero failures."""
    _, _, gates = _score([], planned=0)
    assert gates["M6"].verdict == mba.FAIL


@pytest.mark.parametrize("mutate, needle", [
    (lambda row: row.pop("pin"), "no pin assay recorded"),
    (lambda row: row["pin"].update(observed={}), "nothing was read back"),
    (lambda row: row["pin"].update(source="vllm_tuned"), "came from 'vllm_tuned'"),
    (lambda row: row["pin"]["observed"].update(num_stages=4), "num_stages"),
    (lambda row: row["pin"]["requested"].update(BLOCK_SIZE_M=128), "labelled block_m"),
])
def test_pin_disagreement_names_every_way_a_cell_can_fail_to_show_its_tile(mutate, needle):
    """Four failures with four different fixes, kept apart on purpose. A single
    "pin bad" would send an operator looking for a cache when the kernel was
    taking its own tile, and the pod is rented while they look."""
    row = mba.plant_cells(MIXTRAL, pin="ok")[0]
    assert mba.pin_disagreement(row) == ""
    mutate(row)
    assert needle in mba.pin_disagreement(row)


def test_a_partly_honoured_override_is_not_a_pass():
    """All six constants, not the two this arm sweeps. A vLLM that honoured
    BLOCK_SIZE_M and GROUP_SIZE_M while substituting num_warps would still be
    running a kernel nobody asked for, timed as if it were the one asked for."""
    cells = mba.plant_cells(MIXTRAL, pin="ok")
    for row in cells:
        row["pin"]["observed"]["num_warps"] = 4
    _, _, gates = _score(cells)
    assert gates["M6"].verdict == mba.FAIL


def test_the_pin_source_string_has_not_drifted_from_force_tiles():
    """`PIN_SOURCE_OVERRIDE` duplicates `moe.bench.force_tile`'s constant,
    because importing that module at script scope pulls torch in through
    `moe.quant` and breaks the laptop `--rescore` path. A duplicate nobody
    checks is a duplicate that drifts."""
    force_tile = pytest.importorskip("moe.bench.force_tile")
    assert mba.PIN_SOURCE_OVERRIDE == force_tile.TILE_SOURCE_OVERRIDE


# --------------------------------------------------------------------------
# THE EXIT-CODE CONTRACT (audit A4). This file documented 2 for "a VALIDITY gate
# failed after the eight-minute measurement" and 3 for "nothing measured", the
# exact inverse of the session driver's table.
# --------------------------------------------------------------------------

def test_the_measure_mode_refuses_with_two_when_there_is_no_device():
    """The audit's own test, run as a PROCESS: the integer the driver sees is
    the only thing it can read without parsing prose."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "memory_branch_anchor.py"),
         "--measure", "--card", "nonexistent"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode == mba.exit_codes.REFUSED, proc.stdout[-2000:]
    assert "REFUSED" in proc.stdout


def test_a_validity_failure_after_measuring_is_invalid_and_not_refused():
    """The two are opposite states and the driver treats them oppositely: a
    REFUSED arm cost nothing and has nothing to resume into, an INVALID arm cost
    the whole allocation and has a directory of cells that must not be scored.
    Folding them together is what printed "REFUSED BEFORE MEASURING. Nothing
    below is a gate" over an eight-minute run whose gate had failed."""
    _, _, gates = _score(mba.plant_cells(MIXTRAL), stream={"gbps": 1e9})
    assert gates["M0"].verdict == mba.FAIL
    rc = mba.exit_codes.classify(g.scored() for g in gates.values())
    assert rc == mba.exit_codes.INVALID
    assert mba.exit_codes.ledger_state(rc) == "INVALID"


def test_a_full_resume_carries_the_stream_check_and_passes_m0(tmp_path):
    """THE ARM COULD NEVER REACH DONE. The stream check runs only on a freshly
    timed cell, so a run that resumed every cell measured none, handed M0 a
    None, and was REFUSED again on every attempt. The check belongs to the
    session that measured the cells and is stored with them."""
    cells = mba.plant_cells(MIXTRAL)
    path = tmp_path / "cells.json"
    path.write_text(json.dumps({"card": "nvidia_a100_sxm4_80gb", "run_id": "x",
                                "stream_check": {"gbps": 1500.0},
                                "cells": cells}))
    rows, done, stream = mba.restore_cells(path, "nvidia_a100_sxm4_80gb")
    assert len(done) == len(cells) and stream is not None
    _, _, gates = _score(rows, stream=stream, planned=len(rows))
    assert gates["M0"].verdict == mba.PASS
    assert mba.exit_codes.classify(g.scored() for g in gates.values()) == 0


def test_a_resume_refuses_a_foreign_card_and_the_legacy_shape(tmp_path):
    """Both refusals, because the card in the id only makes the collision hard
    to reach, not impossible: an explicit --out-dir, a directory copied between
    pods, or a file written before the card entered the id all reach it."""
    path = tmp_path / "cells.json"
    path.write_text(json.dumps({"card": "nvidia_h200", "cells": []}))
    with pytest.raises(mba.ResumeRefused) as exc:
        mba.restore_cells(path, "nvidia_a100_sxm4_80gb")
    assert "nvidia_h200" in str(exc.value)
    path.write_text(json.dumps([]))          # the legacy bare list
    with pytest.raises(mba.ResumeRefused) as exc:
        mba.restore_cells(path, "nvidia_a100_sxm4_80gb")
    assert "pre-card-in-id" in str(exc.value)
    assert mba.restore_cells(tmp_path / "absent.json", "any") == ([], set(), None)


def test_every_scored_gate_prints_exactly_one_result_line(tmp_path, capsys, scored):
    """The ONE machine contract. The driver's summary used to grep free text
    (`floor|sigma`) and matched a REFUSED arm's log 18 times, printing an
    imported constant under the heading "floor" and a pre-registered expectation
    as `[PASS]`. Here: one RESULT line per gate, nothing else shaped like one,
    and the code the log implies is the code the function returned."""
    rc = mba.main(["--rescore", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    lines = mba.exit_codes.parse_result_lines(out)
    assert len(lines) == 11, [line.name for line in lines]
    assert len({line.name for line in lines}) == 11
    assert mba.exit_codes.classify_text(out) == rc


def test_the_self_test_prints_no_result_line_at_all(capsys):
    """A planted world's verdict is not a result about this machine, and a
    driver that grepped one would be reading a plant as a measurement."""
    assert mba.main(["--self-test"]) == mba.exit_codes.DONE
    out = capsys.readouterr().out
    assert mba.exit_codes.parse_result_lines(out) == []
    assert "pin_failed" in out


def test_the_self_test_fails_when_a_world_is_mis_registered(monkeypatch, capsys):
    """The proof that the self-test can return non-zero. Without this the whole
    mode is a function that has only ever been seen to print PASS."""
    broken = (mba.SELF_TEST_WORLDS[1].__class__(
        **{**mba.SELF_TEST_WORLDS[1].__dict__, "expect": mba.exit_codes.DONE}),)
    monkeypatch.setattr(mba, "SELF_TEST_WORLDS", broken)
    assert mba.self_test() == mba.exit_codes.INVALID
    assert "registered DONE" in capsys.readouterr().out


def test_every_gate_this_file_can_build_has_a_result_line_name():
    """`Gate.name` raises KeyError on an unnamed gate rather than falling back,
    because a fallback name is a gate that quietly leaves a driver's summary."""
    assert set(mba.GATE_NAMES) >= {"V1", "V2", "V3", "V4", "V5", "V6",
                                   "C1", "C2", "C3", "C4", "C5",
                                   "M0", "M1", "M2", "M3", "M4", "M5", "M6"}
    assert len(set(mba.GATE_NAMES.values())) == len(mba.GATE_NAMES)
    gate = mba.gate_m6_pin(mba.plant_cells(MIXTRAL))
    back = mba.exit_codes.parse_result_lines(gate.result_line() + "\n")
    assert [b.name for b in back] == ["pin_took_effect"]
    assert back[0].verdict == gate.verdict


# --------------------------------------------------------------------------
# THE INSTRUMENT (audit A7) and the run's provenance (A5).
# --------------------------------------------------------------------------

def test_the_retired_instrument_is_gone_rather_than_wrapped():
    """`time_call` synchronised every iteration with events created inside the
    loop and no L2 flush, which put 0.18-0.30 ms of host enqueue time inside the
    measured interval. A wrapper would have kept the two-instrument problem and
    hidden it; the roof and the ladders are queue-deep, so this arm is too."""
    assert not hasattr(mba, "time_call")
    assert mba.timing_basis() in (None, "queue-deep/l2-flush/clock-under-load/v2")


def test_every_timing_column_the_apparatus_requires_reaches_the_row():
    """A row that recorded a time and not the clock it was taken at cannot be
    compared with the roof, and this study has nine session scripts that record
    no clock at all."""
    timing = pytest.importorskip("moe.bench.timing")
    t = timing.KernelTiming(
        ms_p50=1.0, ms_p90=1.1, ms_min=0.9, ms_std=0.01, iters=200, trials=3,
        warmup_ms=301.0, l2_flush=True, sm_clock_load_mhz=1480.0,
        sm_clock_start_mhz=1480.0, sm_clock_end_mhz=1470.0, clock_level_ok=True,
        clock_drift_ok=True, samples=600, warmup_calls=90, flush_mb=256,
        clock_samples=9, clock_source="nvml", clock_poll_ms=0.2,
        host_bound=False, host_enqueue_ms=0.4)
    columns = mba.timing_columns(t)
    assert {"instrument", "warmup_ms", "iters", "trials", "sm_clock_load_mhz",
            "clock_level_ok", "clock_drift_ok", "l2_flush"} <= set(columns)
    assert columns["instrument"] == timing.TIMING_BASIS
    assert columns["warmup_ms"] == 301.0


def test_the_plan_names_its_instrument_and_its_warmup_in_milliseconds(capsys):
    """The ladders warm for 300 ms. This arm warmed for 5 CALLS, and the two
    numbers were then compared tread for tread."""
    mba.main(["--measure", "--dry-run", "--card", "nvidia_h200"])
    out = capsys.readouterr().out
    assert "300 ms warmup" in out
    assert "iters is NOT a knob" in out


def test_the_rescore_report_carries_a_provenance_block(tmp_path):
    """A number nobody can attribute to a commit, a card and a ruler is not a
    measurement. The 26 published report.json files carry none of the three."""
    mba.main(["--rescore", "--out-dir", str(tmp_path)])
    payload = json.loads((tmp_path / "ANCHOR_RESCORE.json").read_text())
    assert set(mba.PV.TOP_LEVEL_KEYS) <= set(payload)
    block = payload["provenance"]
    assert block["provenance_version"] == mba.PV.PROVENANCE_VERSION
    # NOTHING WAS TIMED, and the block says so in words rather than borrowing
    # the current instrument's name for numbers it did not produce.
    assert "times nothing" in payload["instrument"]
    assert block["ridge_source"] and block["bandwidth_source"]


# --------------------------------------------------------------------------
# THE OUTPUT PATH (audit A6) and the stated MDE (B14).
# --------------------------------------------------------------------------

def test_a_default_rescore_leaves_the_work_tree_exactly_as_it_found_it(tmp_path):
    """--rescore rewrote a TRACKED pair on every run, including the one the
    session driver makes under --dry-run, with the author's home directory
    embedded. The tree was dirty from arm one and 44,872 of 100,144 published
    rows carry git_dirty=True."""
    def porcelain():
        proc = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                              capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            pytest.skip("not a git work tree")
        return proc.stdout
    committed = REPO / "results" / "published" / "ANCHOR_RESCORE.txt"
    before, bytes_before = porcelain(), (committed.read_bytes()
                                         if committed.exists() else None)
    mba.main(["--rescore"])
    assert porcelain() == before
    assert (committed.read_bytes() if committed.exists() else None) == bytes_before


def test_only_publish_routes_the_rescore_into_the_tree(monkeypatch):
    """The flag is the whole guard, so the routing is asserted rather than
    exercised: running --publish here would rewrite the committed pair, which is
    the thing under test."""
    seen = {}

    def capture(args):
        seen["out"] = args.out_dir
        return 0

    monkeypatch.setattr(mba, "run_rescore", capture)
    mba.main(["--rescore"])
    assert seen["out"] != mba.PUBLISHED
    assert mba.git_ignored(seen["out"] / "ANCHOR_RESCORE.txt") is not False
    mba.main(["--rescore", "--publish"])
    assert seen["out"] == mba.PUBLISHED


# --------------------------------------------------------------------------
# The ridge census. A paragraph about the reports became false when the reports
# changed, so the paragraph is now a count of them.
# --------------------------------------------------------------------------

def _cals() -> dict:
    return {slug: mba.load_calibration(slug)
            for slug in ("nvidia_a100_sxm4_80gb", "nvidia_h200")}


def _plant_report(root, arm: str, ridge, rescored_from=None) -> None:
    """One report.json carrying only what the census reads."""
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    doc = {"ridge": ridge}
    if rescored_from is not None:
        doc["rescored_from"] = rescored_from
    (d / "planted.report.json").write_text(json.dumps(doc))


def test_the_ridge_census_reads_the_committed_reports_rather_than_asserting_them():
    """THE PARAGRAPH THIS REPLACES WAS A LITERAL AND THE LITERAL WENT FALSE.

    `ANCHOR_RESCORE.txt` said, in `say(...)` text, "Every one of these reports
    carries ridge=160.3 ... 160.3 is a stale H200 band and belongs to NEITHER
    card", while `scripts/rescore_published_reports.py` had already rewritten
    all 26 on the same branch. So the transcript regenerated false on every run
    and disagreed with the data sitting beside it. This asserts the census
    agrees with the files, which is a thing that cannot go stale.
    """
    cals = _cals()
    census = mba.ridge_census(mba.PUBLISHED, cals)
    assert census["total"] == census["own_card"] > 0
    assert census["still_swept"] == []
    assert census["strangers"] == []
    assert census["unattributed"] == []
    for slug, values in census["per_card"].items():
        assert max(abs(v - cals[slug].ridge) for v in values) <= mba.RIDGE_MATCH_TOL
    text = "\n".join(mba.render_ridge_census(census, cals))
    assert "No report carries the swept ridge" in text
    assert f"{mba.SWEPT_RIDGE} default" in text        # named as HISTORY only


def test_the_ridge_census_names_a_report_that_still_carries_the_swept_ridge(tmp_path):
    """The FAIL branch, which the retracted paragraph could not have: it said
    every report carried 160.3 whether or not any did, so it was equally wrong
    before and after the rescoring."""
    cals = _cals()
    _plant_report(tmp_path, "2026-01-01-nvidia_h200-planted", mba.SWEPT_RIDGE)
    _plant_report(tmp_path, "2026-01-02-nvidia_h200-fine", 162.8,
                  rescored_from={"ridge": mba.SWEPT_RIDGE})
    _plant_report(tmp_path, "2026-01-03-nvidia_a100_sxm4_80gb-odd", 999.0)
    _plant_report(tmp_path, "2026-01-04-no-card-in-this-name", 162.8)
    census = mba.ridge_census(tmp_path, cals)
    assert census["total"] == 4
    assert census["own_card"] == 1
    assert census["rescored_from"] == 1
    assert len(census["still_swept"]) == 1
    assert len(census["strangers"]) == 1
    assert len(census["unattributed"]) == 1
    text = "\n".join(mba.render_ridge_census(census, cals))
    assert "STILL CARRYING THE SWEPT RIDGE" in text
    assert "2026-01-01-nvidia_h200-planted/planted.report.json" in text
    assert "999.0" in text
    assert "No report carries the swept ridge" not in text


def test_the_committed_transcript_is_what_the_script_writes_today(tmp_path):
    """The committed `ANCHOR_RESCORE.txt` regenerates byte for byte, or it is
    stale.

    Nothing compared the two until 2026-09-02, and the house rule to
    `git checkout --` the file after every suite run then FROZE whichever
    version was committed: the tracked transcript had no `RESULT:` line at all,
    so `exit_codes.classify_text` over it raised `NoGatesScored`, and it still
    carried a paragraph the same branch had made false. A published artefact its
    own producer no longer writes is not evidence of anything.

    The .json is deliberately NOT compared: its provenance block carries a
    timestamp and the working tree's dirty flag, so byte equality there would be
    a test of the clock.
    """
    committed = mba.PUBLISHED / "ANCHOR_RESCORE.txt"
    assert committed.exists()
    mba.main(["--rescore", "--out-dir", str(tmp_path)])
    fresh = (tmp_path / "ANCHOR_RESCORE.txt").read_text()
    assert fresh == committed.read_text()
    assert mba.exit_codes.classify_text(fresh) == mba.exit_codes.CLAIM_FAIL


#: The three keys in the published JSON that a re-run is ALLOWED to move: the
#: clock, and the two halves of the git stamp. Everything else is a function of
#: the committed reports and calibrations, so a re-run that moves anything else
#: has changed the analysis without changing the transcript.
_STAMP_KEYS = ("utc", "git_sha", "git_dirty", "git_dirty_files")


def _strip_stamp(payload: dict) -> dict:
    """The published payload with the run stamp removed, top level and
    provenance block, so two runs can be compared on their CONTENT."""
    out = {k: v for k, v in payload.items() if k not in _STAMP_KEYS}
    prov = dict(out.get("provenance") or {})
    for key in _STAMP_KEYS:
        prov.pop(key, None)
    out["provenance"] = prov
    return out


def test_the_committed_rescore_json_names_a_clean_tree(tmp_path):
    """A published artefact whose provenance says `git_dirty: true` names no
    committed state, and this one did.

    IT STAMPED `461d0e66` WITH `git_dirty: true, git_dirty_files: 2`: a SHA two
    commits behind the branch tip, and a tree that did not match it either, so
    the pair described neither the code that wrote it nor anything a reader
    could check out. `report_output_paths` now carries the regeneration
    procedure, whose first line is `git status --porcelain` returning empty.

    THE ONE-COMMIT SHA LAG IS NOT WHAT THIS TESTS, because a file cannot carry
    the hash of the commit that carries it. What is tested is the part that was
    avoidable: the tree was clean when the pair was written, and the CONTENT of
    a fresh run is identical to the committed content once the run stamp is
    taken out. That second half is what makes the lag harmless -- the payload
    is a function of the committed reports, not of the HEAD it ran at -- and it
    would fail if a later edit changed the analysis without the pair being
    regenerated.
    """
    committed = json.loads((mba.PUBLISHED / "ANCHOR_RESCORE.json").read_text())
    prov = committed["provenance"]
    assert prov["git_dirty"] is False, prov["git_sha"]
    assert prov["git_dirty_files"] == 0
    assert prov["git_sha"] == committed["git_sha"]

    mba.main(["--rescore", "--out-dir", str(tmp_path)])
    fresh = json.loads((tmp_path / "ANCHOR_RESCORE.json").read_text())
    assert _strip_stamp(fresh) == _strip_stamp(committed)

    # THE FAIL BRANCH, planted: the predicate above has to be able to say no,
    # and the state it has to say no to is the one that shipped.
    dirty = json.loads(json.dumps(committed))
    dirty["provenance"]["git_dirty"] = True
    dirty["provenance"]["git_dirty_files"] = 2
    assert dirty["provenance"]["git_dirty"] is not False
    assert _strip_stamp(dirty) == _strip_stamp(committed)


def test_the_regeneration_procedure_is_written_where_the_pair_is_written():
    """`--rescore --publish` rewrites two tracked files, and the discipline that
    keeps their stamp meaningful lives on the function that writes them.

    In a docstring rather than in the emitted pair on purpose: a note added to
    the OUTPUT can only be regenerated from a tree that is dirty with the edit
    that adds it, which is the exact state the note exists to prevent.
    """
    doc = mba.report_output_paths.__doc__
    assert "git status --porcelain" in doc
    assert "--rescore --publish" in doc
    assert "CLEAN TREE" in doc


def test_the_rescore_report_embeds_a_repo_relative_root(tmp_path):
    """An absolute path is a fact about one laptop, and this one was being
    written into a tracked file."""
    mba.main(["--rescore", "--out-dir", str(tmp_path)])
    text = (tmp_path / "ANCHOR_RESCORE.txt").read_text()
    assert "published root : results/published" in text
    assert str(REPO) not in text
    assert str(REPO) not in (tmp_path / "ANCHOR_RESCORE.json").read_text()


def test_the_plan_states_an_mde_and_labels_c5_a_prior(capsys):
    """B14: no arm in this study stated an MDE, so every threshold read as a
    number the author liked. C5's 0.05 had no justification at all."""
    mba.main(["--measure", "--dry-run", "--card", "nvidia_h200"])
    out = capsys.readouterr().out
    assert "MINIMUM DETECTABLE EFFECT" in out
    assert f"{mba.mde_ratio():.2%}" in out
    assert "is a PRIOR" in out


def test_the_mde_follows_the_noise_assumption_it_is_derived_from(capsys):
    """A stated assumption a reader can disagree with: the published replicates
    run 0.76% to 1.82%, and at the pessimistic end M1's 4% gate stops being
    comfortably above the smallest effect it could resolve."""
    assert mba.mde_ratio(0.0182) > mba.mde_ratio(0.0077)
    assert mba.mde_ratio(0.0182) > mba.ANCHOR_INVARIANCE_SMALL_G
    mba.main(["--measure", "--dry-run", "--card", "nvidia_h200",
              "--noise", "0.0182"])
    assert "sigma = 1.82%" in capsys.readouterr().out


def test_c5_scores_against_the_prior_it_names():
    """The constant and the threshold text have to be one thing, or the report
    prints a number the gate does not use."""
    gate = mba.gate_c5_correction_size([])
    assert str(mba.C5_SHIFT_PRIOR) in gate.threshold
    assert "PRIOR" in gate.threshold


# --------------------------------------------------------------------------
# THE FOUR FUNCTIONS THAT BUILD THE EVIDENCE. Everything above tests the
# CONSUMERS of a pin record against a planted one; a review found that nothing
# tested the code that builds a real one. All four run off-GPU: the device work
# is in the `run_measure` loop around them, not in them.
# --------------------------------------------------------------------------

def test_reference_clock_from_takes_the_sweeps_three_fields_in_its_order():
    """THE SAME ORDER AS THE LADDERS, or an anchor is admitted (or excluded) on
    a LEVEL rule the data it re-anchors never applied. The three fields have
    disagreed on one H200 by 450 MHz, so which one answered is part of the
    answer and is asserted here alongside the number."""
    detail = {"gemm_clock": {"median_mhz": 1470}, "gemm_clock_mhz": 1935,
              "settle": {"final_mhz": 1515}}
    mhz, why = mba.reference_clock_from(detail, "nvidia_h200")
    assert (mhz, "median of the samples" in why) == (1470.0, True)

    detail.pop("gemm_clock")
    mhz, why = mba.reference_clock_from(detail, "nvidia_h200")
    assert (mhz, "gemm_clock_mhz" in why) == (1935.0, True)

    detail.pop("gemm_clock_mhz")
    mhz, why = mba.reference_clock_from(detail, "nvidia_h200")
    assert (mhz, "settle" in why) == (1515.0, True)


def test_reference_clock_from_says_no_cell_can_be_excluded_when_it_finds_none():
    """The branch that silently disables every LEVEL verdict in the run. None is
    not a failure and it is not a pass either: it has to SAY that nothing can be
    excluded on it, because the alternative -- a missing clock reading as a
    clean one -- is how a throttled cell reaches a published fit."""
    mhz, why = mba.reference_clock_from({}, "nvidia_a100_sxm4_80gb")
    assert mhz is None
    assert "nvidia_a100_sxm4_80gb" in why
    assert "not determinable" in why


def test_count_new_counts_each_artefact_once_and_only_once(tmp_path):
    """The compile assay is a DIFFERENCE, so double-counting one file would let
    a setting that compiled nothing inherit the previous setting's evidence."""
    seen: set = set()
    assert mba.count_new(tmp_path, seen) == 0
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "kernel.cubin").write_text("x")
    assert mba.count_new(tmp_path, seen) == 1
    assert mba.count_new(tmp_path, seen) == 0
    (tmp_path / "a" / "kernel.json").write_text("y")
    assert mba.count_new(tmp_path, seen) == 1


def test_arm_triton_cache_points_triton_at_this_settings_own_directory(tmp_path,
                                                                       monkeypatch):
    """The variable is read by Triton at COMPILE time, so it has to be set
    before the setting's first call, and the per-setting directory is what makes
    "did this setting compile anything" countable rather than assumed. The key
    is the PAIR: this arm sweeps the swizzle, and a directory keyed on the tile
    alone would pool four swizzles into one count."""
    monkeypatch.setenv("TRITON_CACHE_DIR", "a-stale-value")
    first = mba.arm_triton_cache(tmp_path, 32, 8)
    assert first.is_dir() and not list(first.iterdir())
    assert os.environ["TRITON_CACHE_DIR"] == str(first)
    assert mba.arm_triton_cache(tmp_path, 32, 64) != first
    assert mba.arm_triton_cache(tmp_path, 64, 8) != first


def test_a_resumed_session_does_not_inherit_the_previous_sessions_warm_cache(tmp_path):
    """THE DEFECT THIS FUNCTION EXISTS AGAINST. A cache root under the resume
    directory hands session 2 session 1's compiled artefacts; the per-setting
    baseline absorbs them, every re-measured cell records `fresh_artefacts = 0`,
    and M6 fails "compiled nothing new" on a sound run for ever. Session 2 has
    to start empty, and session 1's evidence has to survive, because deleting
    what it finds under a human's --out-dir is not this script's business."""
    one = mba.session_cache_root(tmp_path)
    mba.arm_triton_cache(one, 32, 8).joinpath("kernel.cubin").write_text("x")

    two = mba.session_cache_root(tmp_path)
    assert two != one
    assert list(one.rglob("*.cubin"))            # session 1 was not swept away
    assert not list(two.rglob("*"))              # session 2 starts cold

    seen: set = set()
    mba.arm_triton_cache(two, 32, 8)
    assert mba.count_new(two, seen) == 0
    (two / "bm32-g8" / "kernel.cubin").write_text("x")
    assert mba.count_new(two, seen) == 1


# --------------------------------------------------------------------------
# `observed_pin`: the one derivation the pin assay calls derived rather than
# asserted. Its input is a real `TileCapture`, so it is built here as
# `recording_tile_config` would fill it.
# --------------------------------------------------------------------------

def _capture(**call):
    """A `TileCapture` holding one recorded lookup, or none at all."""
    fc = pytest.importorskip("moe.baselines._framework_config")
    cap = fc.TileCapture()
    if call:
        cap.calls.append(fc.TileCall(**call))
    return cap


REQUESTED = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3}


def test_observed_pin_reads_the_override_off_the_lookup_it_skipped():
    """THE DERIVATION. `try_get_optimal_moe_config` consults `get_config()`
    first and returns the override WITHOUT reaching `get_moe_configs`, so a call
    that skipped the tuned-file lookup took the override. Deriving it from what
    the lookup did is the whole point: `tile_meta_from_capture(
    override_active=True)` labels any capture vllm_override because its caller
    said so, and a gate that asks "did the override take" cannot be answered by
    the fact that one was entered."""
    conf, source = mba.observed_pin(
        _capture(m=1024, config=dict(REQUESTED), lookup_observed=False))
    assert source == mba.PIN_SOURCE_OVERRIDE
    assert conf == REQUESTED


@pytest.mark.parametrize("tuned_keys, source", [
    (None, "vllm_default"),
    ([16, 64, 128], "vllm_tuned"),
])
def test_observed_pin_names_vllms_own_two_config_paths_apart(tuned_keys, source):
    """A call that RAN the lookup went to vLLM's tuned file or to its hardcoded
    fallback ladder, and either way the override did not take. They are kept
    apart because "there is a tuned file for this shape and it won" and "there
    is none and the ladder won" send an operator to different files."""
    ran = dict(REQUESTED, BLOCK_SIZE_M=64, GROUP_SIZE_M=1)
    conf, got = mba.observed_pin(
        _capture(m=1024, config=ran, tuned_keys=tuned_keys, lookup_observed=True))
    assert got == source
    assert conf["BLOCK_SIZE_M"] == 64
    assert mba.pin_disagreement(
        {"block_m": 32, "group_m": 8,
         "pin": {"requested": REQUESTED, "observed": conf, "source": got}})


def test_observed_pin_records_nothing_rather_than_a_likelier_answer():
    """No recorded call is not "the override took". It is a cell with no
    evidence, and `pin_disagreement` fails it for exactly that."""
    assert mba.observed_pin(_capture()) == ({}, "unrecorded")


def test_observed_pin_keeps_only_the_six_constants_the_gate_compares():
    """The capture carries whatever vLLM's config dict held. Writing the extras
    into the row would put keys in `observed` that `requested` never had and
    that no leg of M6 reads."""
    conf, _ = mba.observed_pin(_capture(
        m=1024, config=dict(REQUESTED, SPLIT_K=2, matrix_instr_nonkdim=16),
        lookup_observed=False))
    assert set(conf) == set(mba.PIN_KEYS)


def test_a_memoised_vllm_lands_in_the_override_branch_and_the_tile_still_catches_it():
    """THE SOURCE LEG'S RESIDUAL, held at the width the docstring now claims.
    `recording_tile_config` names the degradation itself: a vLLM that memoises
    `try_get_optimal_moe_config` never re-enters `get_moe_configs`, so the row
    comes back with real tile ints and no observation behind them, which is
    indistinguishable HERE from an override that took. It is not
    indistinguishable at the gate: the config in hand is vLLM's own choice, so
    the six constants disagree with the six requested and M6 fails on the tile
    comparison instead of on the label."""
    memoised = dict(REQUESTED, BLOCK_SIZE_M=64, GROUP_SIZE_M=1)
    conf, source = mba.observed_pin(
        _capture(m=1024, config=memoised, lookup_observed=False))
    assert source == mba.PIN_SOURCE_OVERRIDE      # the label cannot see it

    cells = mba.plant_cells(MIXTRAL, pin="ok")
    for row in cells:
        row["pin"]["observed"] = dict(conf, GROUP_SIZE_M=row["group_m"])
    _, _, gates = _score(cells)
    assert gates["M6"].verdict == mba.FAIL
    assert "BLOCK_SIZE_M: asked 32, ran 64" in " ".join(gates["M6"].lines)


def test_the_gate_render_has_no_way_to_suppress_its_result_line():
    """The suppression flag that used to be here named `--self-test` as its
    caller and `self_test` never called `render` at all: it prints its own
    lines. A flag would have made "no RESULT line out of a plant" something a
    caller has to remember; not having one is the same guarantee with nothing
    to forget."""
    gate = mba.gate_v1_non_vacuity([], [])
    assert list(inspect.signature(mba.Gate.render).parameters) == ["self"]
    assert gate.render()[0] == gate.result_line()
