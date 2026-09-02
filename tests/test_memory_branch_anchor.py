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
"""
from __future__ import annotations

import importlib.util
import json
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


def test_ai_cap_refuses_a_non_positive_alpha():
    """A zero alpha would report an infinite arithmetic-intensity cap and turn
    the surviving BLOCK_M <= 64 result into an unconditional PASS."""
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    br = mba.bracket_alpha(0.88, 0.88, 1.95, 32, w, act1, A100_CEILING, A100_PIN)
    with pytest.raises(ValueError):
        br.ai_cap(2, 0.0)


def test_the_cap_agrees_with_the_corrected_ai_model():
    """`moe/bench/ai_model.py` (2026-09-02) showed the study's AI denominator was
    missing the activation re-read and the output write. `2 BM / (b alpha)` is
    still the right ceiling PROVIDED the alpha is the fitted composite, because
    `alpha_fitted / BM = alpha_b/BM + alpha_a/BN + 1/K` term by term. If that
    identity ever stops holding, this gate is capping the wrong thing."""
    ai_model = pytest.importorskip("moe.bench.ai_model")
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    br = mba.bracket_alpha(0.88, 0.88, 1.95, 32, w, act1, A100_CEILING, A100_PIN)
    block_m, block_n, k = 32, 64, MIXTRAL.hidden_size
    alpha_b, alpha_a = 0.307, 0.143
    alpha_fitted = alpha_b + alpha_a * (block_m / block_n) + block_m / k
    assert br.ai_cap(2, alpha_fitted) == pytest.approx(
        ai_model.cap(2 * MIXTRAL.intermediate_size, k, block_m=block_m,
                     block_n=block_n, alpha_b=alpha_b, alpha_a=alpha_a, b=2),
        rel=1e-9)


def test_subtracting_the_activation_term_only_ever_raises_the_cap():
    """Which is why C3's PASS is conservative: the gate hands `ai_cap` an alpha
    with a positive term removed, so the ceiling it checks is higher than the
    real one and a tile has a better chance of clearing the ridge, not worse."""
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", 32)
    br = mba.bracket_alpha(0.88, 0.88, 1.95, 32, w, act1, A100_CEILING, A100_PIN)
    uncorrected = mba.alpha_at_bandwidth(0.88, br.bw_anchor_gbps, w, 0)
    assert br.lo < uncorrected
    assert br.ai_cap(2, br.lo) > br.ai_cap(2, uncorrected)


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
    """`t(n)` from a stated alpha, with the anchor allowed its own bandwidth.

    The anchor running SLOWER than the branch is the pathology in the committed
    data -- the measured n=1 tread stands above the fitted line in 12 of 12 A100
    fits -- so the plant has to be able to express it, or the test would only
    ever exercise the case where nothing is wrong.
    """
    w, act1 = mba.anchor_bytes(MIXTRAL, "bf16", block_m)
    out = []
    for n in range(1, treads + 1):
        bw = bw_anchor if n == 1 else bw_branch
        bytes_n = w * (1.0 + alpha * (n - 1)) + act1 * n
        out.append((n, fixed_ms + bytes_n / bw * 1e3))
    return out, w, act1


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
        cap_over_ridge_at_lo=0.0, timing_spread=0.005,
        elevation_in_spreads=0.0)


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
                block_k=64, num_warps=8, num_stages=3, seed=0, iters=30,
                warmup=5, stream_reps=20)
    base.update(over)
    return mba.MeasurePlan(**base)


@pytest.mark.parametrize("field, value", [
    ("card", "nvidia_a100_sxm4_80gb"),
    ("model", "mixtral-8x7b"), ("dtype", "fp8_w8a8"), ("block_sizes", (32,)),
    ("group_sizes", (1,)), ("slope_tiles", (2, 3)), ("block_n", 256),
    ("block_k", 128), ("num_warps", 4), ("num_stages", 4), ("seed", 1),
    ("iters", 31), ("warmup", 6), ("stream_reps", 21),
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
    for token in ("nvidia_h200", "qwen2-57b-a14b", "bf16", "bm32_64",
                  "g1_8_16_64", "n64", "s3"):
        assert token in rid


def test_a_dry_run_with_no_device_says_its_path_is_not_the_pods(capsys):
    """A dry run on a laptop must not print a path a pod will never write to.

    The placeholder card is visible in the id AND called out in words, because
    the operator's next move is to check `git check-ignore` on that exact path.
    """
    assert mba.main(["--measure", "--dry-run"]) == 3
    out = capsys.readouterr().out
    assert f"{mba.UNKNOWN_CARD_SLUG}-qwen2-57b-a14b" in out
    assert "NO DEVICE ATTACHED" in out


def test_card_flag_prints_the_pods_real_path_from_a_laptop(capsys):
    assert mba.main(["--measure", "--dry-run", "--card", "nvidia_h200"]) == 3
    out = capsys.readouterr().out
    assert "nvidia_h200-qwen2-57b-a14b" in out
    assert mba.UNKNOWN_CARD_SLUG not in out


def test_slope_tiles_below_two_are_refused_at_the_cli():
    """The anchor may not be inside the slope it is compared against."""
    dense = ",".join(str(n) for n in range(1, 17))
    with pytest.raises(SystemExit) as exc:
        mba.main(["--measure", "--dry-run", "--slope-tiles", dense])
    assert "must all be >= 2" in str(exc.value)


def test_a_short_branch_is_refused_because_p3_cannot_be_scored_on_it():
    """P3's 1.5% threshold came from 16- and 33-tread ladders. On 8 treads the
    slope moves 3.2% for a reason that is about the grid, and a threshold that
    fails for the wrong reason teaches a reader to ignore it."""
    with pytest.raises(SystemExit) as exc:
        mba.main(["--measure", "--dry-run", "--slope-tiles", "2,3,4,6,8,12,16"])
    assert "branch treads" in str(exc.value)


def test_dry_run_measures_nothing_and_says_so():
    assert mba.main(["--measure", "--dry-run"]) == 3


# --------------------------------------------------------------------------
# The GPU arm's verdict path, exercised with no GPU.
# --------------------------------------------------------------------------

def _calibration() -> mba.Calibration:
    return mba.Calibration(
        slug="nvidia_a100_sxm4_80gb", name="A100 (test)", checked_on="2026-09-02",
        measured_commit="deadbeef", patterns={"triad": 1799.4, "write": A100_CEILING},
        ceiling_pattern="write", ceiling_gbps=A100_CEILING, pin_gbps=A100_PIN,
        dense_tflops={"bf16": 262.3712016979615}, ridge=145.81)


def _cells(alpha=0.558, bw_anchor=1450e9, bw_branch=1750e9, fixed_ms=0.05,
           groups=(1, 16), block_m=32, treads=16):
    pts, _, _ = planted_ladder(alpha, bw_anchor, bw_branch, fixed_ms,
                               block_m=block_m, treads=treads)
    rows = []
    for g in groups:
        for n, ms in pts:
            rows.append({"block_m": block_m, "group_m": g, "tiles": n,
                         "rows_per_expert": block_m * n, "tokens": 0,
                         "ms_p50": ms, "ms_min": ms, "ms_stdev": 0.0,
                         "status": "ok", "detail": ""})
    return rows


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
    rows = _cells(groups=(1,))
    slow = _cells(groups=(16,), bw_anchor=1000e9)     # a 45% slower anchor at G=16
    fits, _ = mba.fits_from_cells(rows + slow, MIXTRAL, "bf16", 64, _calibration())
    assert mba.gate_m1_anchor_invariance(fits).verdict == mba.FAIL
    ok, _ = mba.fits_from_cells(_cells(), MIXTRAL, "bf16", 64, _calibration())
    assert mba.gate_m1_anchor_invariance(ok).verdict == mba.PASS


def test_score_measured_runs_every_gate_and_refuses_an_empty_grid():
    cal = _calibration()
    rows = _cells()
    fits, refusals, gates, lines = mba.score_measured(
        rows, MIXTRAL, "bf16", 64, cal, {"gbps": 1500.0}, planned=len(rows))
    assert {g.number for g in gates} == {"M0", "M1", "M2", "M3", "M4", "M5"}
    assert fits and lines
    _, _, empty_gates, _ = mba.score_measured([], MIXTRAL, "bf16", 64, cal, None, 0)
    assert all(g.verdict == mba.FAIL for g in empty_gates
               if g.number in {"M0", "M3", "M4", "M5"})


def test_score_measured_round_trips_through_a_written_file(tmp_path, capsys):
    """The pod writes, the laptop scores. A verdict path that only runs on a
    rented GPU is a verdict path nobody tests."""
    cal = _calibration()
    plan = _plan(block_sizes=(32,), group_sizes=(1, 16),
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
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "MEASURED BRACKETS" in out
    assert "GATE M1" in out


def test_score_measured_refuses_a_missing_file(tmp_path):
    assert mba.main(["--score-measured", str(tmp_path / "nope.json")]) == 3


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
