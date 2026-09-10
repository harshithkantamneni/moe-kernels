"""BLOCK_M=128 is where the two branches measure as one line, so the fit cannot
speak there.

`scripts/bm128_depth.py` was asked for FIVE clean memory-bound treads at
BLOCK_SIZE_M=128 and answers that they are unreachable on either card in this
study. That answer is arithmetic, so most of this file checks the arithmetic
rather than the plumbing, and the rest checks that the two gates the study did
not have actually fire on the two published fits that should never have shipped.

THE HEADING USED TO SAY "where the cap sits ON the ridge", and that was the
arm's founding premise, retracted on 2026-09-02: the caps it rested on were
`2 BM / (b alpha)`, which leaves `phi` out of its denominator, and corrected
through `moe/bench/ai_model.py` they sit below both cards' ridges. What
survives is the MEASURED `B/C` near 1, a ratio of two fitted slopes with no cap
and no ridge in it. `test_the_founding_premise_no_longer_straddles_the_ridge`
is where that retraction is checked against the published files, and it pins
WHICH ALPHA AND WHICH PHI went into the correction, not only the caps, because
`premise_caps` corrected itself wrongly twice. Version one divided the
retracted cap by `1 + phi + delta`. Version two called `exact_cap` with the
report's `alpha-corrected` in the `alpha_b` slot, and `alpha-corrected` is
`(B - Act1) / L`: a slope over the fitted LEVEL, which is (EXA)-shaped and not
a miss fraction. The input has to be the ladder's RAW `alpha` through
`ai_model.cap_from_fitted`, with `phi = Act1 / W` on the FUSED layer rather
than `ai_model.phi` on one GEMM. Both withdrawn values are asserted absent from
the audit page, so a return to either fails rather than reads as a rounding
difference. The retraction survived all three readings; the printed caps did
not.

FIVE GROUPS.

  - THE LAW. `B/C = alpha b rho / (2 BM)` with the model cancelling; the two
    escape thresholds; and `prefix_depth` checked against a brute-force count on
    ladders planted from the law. The cancellation is tested THROUGH the study's
    own generator rather than by reading the formula, because the formula is
    what is being claimed.
  - THE GATES ON REAL DATA. The published A100 qwen2 GROUP_SIZE_M=64 ladder is
    the only non-monotone one in the corpus and the only one whose margin clears
    the tolerance, and it clears it by 1.0e-4; the two BLOCK_N=256 references
    are 43.6x and 3.9x too slow while being perfectly proportional. Every one of
    those is pinned here against the committed report files.
  - THE REFUSALS. A margin with no noise model reports UNKNOWN rather than a
    pass; a model whose routing cannot form a full tile stack at 128 is refused
    rather than nudged; a run id that omits a swept knob is the bug that
    overwrote a whole arm once already.
  - THE SELF TEST, which is the claim that these gates DISCRIMINATE. A gate that
    answers the same in every planted world cannot settle this experiment, and
    the four worlds are checked to come out differently -- per gate, and in the
    one integer a session driver can see.

The script is loaded by path, because `scripts/` is not a package.
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path

import pytest
import torch as real_torch

from moe.bench import ai_model, exit_codes, timing  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.reference import torch_ref as TORCH_REF  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PUBLISHED = ROOT / "results" / "published"
A100_G64 = (PUBLISHED / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
            / "qwen2-57b-a14b-bf16-r1024-g64-n64-f34659.report.json")
A100_BN256 = (PUBLISHED / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
              / "qwen2-57b-a14b-bf16-r1024-g1-n256-23a131.report.json")
A100_BN64 = (PUBLISHED / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
             / "qwen2-57b-a14b-bf16-r1024-g1-n64-eca45c.report.json")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module that is not there yet makes
    # the decorator fail with an AttributeError that names nothing useful.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bm():
    return _load("bm128_depth", "bm128_depth.py")


def _report(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} is not in results/published on this checkout")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# The law.
# --------------------------------------------------------------------------

def test_the_two_escape_thresholds_are_the_numbers_the_docstring_quotes(bm):
    """147.2 and 204.8 at BLOCK_M=128, bf16. Every claim in the file rests here."""
    assert bm.escape_up_alpha_rho(128, 2) == pytest.approx(147.2, abs=0.05)
    assert bm.escape_down_rho(128, 2, 5) == pytest.approx(204.8, abs=0.05)


def test_the_alpha_form_of_the_escape_down_bound_is_half(bm):
    """0.85 / (0.85 + 5 x 0.17) = 0.500, and it must not depend on BLOCK_M.

    The tile height is already spent: the bound is evaluated at the one B/C the
    tolerance permits, so `escape_down_alpha` takes no block size at all. A
    version that did would invite a reader to escape by changing tiles.
    """
    assert bm.escape_down_alpha(5) == pytest.approx(0.5, abs=1e-9)
    # Falls as more treads are demanded, which is the direction that makes it a
    # bound: five treads is harder than two.
    assert bm.escape_down_alpha(2) > bm.escape_down_alpha(5) > bm.escape_down_alpha(10)


def test_the_lowest_alpha_the_study_ever_measured_is_above_the_bound(bm):
    """P9 without the corpus loader: read the reports and take the minimum.

    If this ever fails, escape-down is OPEN on hardware already rented and the
    pod run should be aimed at whichever arm produced that alpha.
    """
    if not PUBLISHED.exists():
        pytest.skip("no results/published on this checkout")
    seen = []
    for path in PUBLISHED.glob("*/*.report.json"):
        for fit in (json.loads(path.read_text()).get("ladder") or {}).values():
            a = fit.get("alpha_corrected") or fit.get("alpha")
            if a:
                seen.append(a)
    assert seen, "no published report reported an identifiable alpha at all"
    assert min(seen) > bm.escape_down_alpha(5)


def test_the_branch_ratio_does_not_depend_on_the_model(bm):
    """The identity, tested THROUGH the study's own generator, not by reading it.

    mixtral's expert is 6.4x qwen2's. Generate a ladder for each from
    `model_ms` at one alpha, ridge and bandwidth, fit the memory slope and the
    compute slope of each, and the two B/C must agree. A "bigger weights cost
    more to re-read" mechanism would put a factor of 6.4 here.
    """
    sweep = bm.SWEEP
    from moe.spec import MODEL_CONFIGS
    ratios = {}
    for name in ("mixtral-8x7b", "qwen2-57b-a14b"):
        cfg = MODEL_CONFIGS[name]
        # A tile height where the memory branch is unambiguous, so the slopes
        # are the two mechanisms and not a split search's opinion.
        pts = [(n, sweep.model_ms(cfg, n * 32, 32, alpha=0.6, ridge=150.0,
                                  bandwidth_gbps=2000.0, b=2))
               for n in range(1, 9)]
        _, b_slope = sweep._line([float(n) for n, _ in pts],
                                 [ms for _, ms in pts])
        comp = [(n, sweep.model_ms(cfg, n * 256, 256, alpha=0.6, ridge=150.0,
                                   bandwidth_gbps=2000.0, b=2))
                for n in range(1, 5)]
        c256 = sweep._through_origin([float(n) for n, _ in comp],
                                     [ms for _, ms in comp])
        ratios[name] = b_slope / (c256 * 32 / 256)
    a, c = ratios.values()
    assert a / c == pytest.approx(1.0, abs=0.02), (
        f"B/C moved with the model: {ratios}. A 6.4x expert changed the ratio, "
        "so the cancellation in the docstring is wrong and the whole "
        "feasibility argument goes with it.")


def test_the_branch_ratio_scales_the_way_the_identity_says(bm):
    """Doubling BLOCK_M halves B/C; doubling alpha or rho doubles it."""
    base = bm.branch_ratio(128, 2, 0.6, 150.0)
    assert bm.branch_ratio(256, 2, 0.6, 150.0) == pytest.approx(base / 2)
    assert bm.branch_ratio(128, 2, 1.2, 150.0) == pytest.approx(base * 2)
    assert bm.branch_ratio(128, 2, 0.6, 300.0) == pytest.approx(base * 2)
    # Halving the weight bytes -- fp8 -- HALVES B/C, which is why --dtype fp8 is
    # the wrong direction and is refused by the parser's choices.
    assert bm.branch_ratio(128, 1, 0.6, 150.0) == pytest.approx(base / 2)


@pytest.mark.parametrize("alpha,rho", [(0.30, 320.0), (0.40, 260.0),
                                       (0.55, 200.0), (0.95, 175.0)])
def test_prefix_depth_matches_a_brute_force_count(bm, alpha, rho):
    """The closed form against counting treads on a ladder planted from the law.

    Within one tread: `n*` is the exact real-valued crossing and the count is
    its floor, so they can differ by the fractional part.
    """
    load, over, margin = 4.0, 0.5, 0.02
    ratio = bm.branch_ratio(128, 2, alpha, rho)
    c = load * alpha / ratio
    pts = bm.planted_ladder(12, alpha=alpha, rho=rho, block_m=128, b=2,
                            load_ms=load, overhead_ms=over)
    counted = sum(1 for n, ms in pts if ms > over + c * n * (1.0 + margin))
    n_star = bm.prefix_depth(ratio, alpha, over / c, margin)
    predicted = len(pts) if n_star is None else max(0, math.floor(n_star))
    assert abs(predicted - counted) <= 1, (
        f"law says {n_star}, counting says {counted}")


def test_the_straddle_world_is_where_both_cards_sit(bm):
    """alpha 0.88 x rho 145 lands inside the discard band, not near an escape."""
    v = bm.depth_verdict(128, 2, 0.88, 145.0)
    assert v.regime == "discarded"
    assert not v.feasible
    # And nothing about the model or the sweep depth appears in that verdict.
    assert v.needed_alpha_rho == pytest.approx(147.2, abs=0.05)


# --------------------------------------------------------------------------
# The gates, on the published data they exist because of.
# --------------------------------------------------------------------------

def test_monotonicity_catches_the_one_published_ladder_that_runs_backwards(bm):
    """A100 qwen2 g64: tread 8 is 1.237% below tread 7 at a 0.482% spread."""
    d = _report(A100_G64)
    pts = [(int(n), ms) for n, ms in d["ladder"]["128"]["points"]]
    found = bm.inversions(pts, d["timing_spread_median"])
    assert len(found) == 1
    only = found[0]
    assert (only.n_lo, only.n_hi) == (7, 8)
    assert only.rel == pytest.approx(-0.01237, abs=1e-4)
    assert only.sigma > bm.MONOTONE_SIGMA
    gate = bm.gate_monotone(found, len(pts) - 1, d["timing_spread_median"])
    assert gate.passed is False
    assert gate.invalidates, "a failed VALIDITY gate must say what it voids"


def test_monotonicity_passes_the_clean_twin_from_the_same_session(bm):
    """The same card, model and session at GROUP_SIZE_M=1 has no inversion.

    Without this the gate could be failing on the arm rather than on the fault.
    """
    d = _report(A100_BN64)
    pts = [(int(n), ms) for n, ms in d["ladder"]["128"]["points"]]
    assert bm.inversions(pts, d["timing_spread_median"]) == []


def test_the_a100_verdict_turns_on_0_010_percent_of_one_tread(bm):
    """The claim the whole file is built around, checked against the shipped fit.

    Raising tread 8 from 25.4883 to 25.4909 ms -- 0.0026 ms -- takes
    `memory_points` from 7 to 0, because that tread is the ONLY point on the
    compute branch and so sets the very slope the memory branch is compared
    against. If this ever stops being true the docstring's headline is stale.
    """
    d = _report(A100_G64)
    sweep = bm.SWEEP
    cr = d["compute_reference"]
    ref = sweep.ComputeReference(cr["block_m"], cr["overhead_ms"],
                                 cr["slope_per_tile"], cr["mean_rel_err"],
                                 cr["note"])
    pts = [(int(n), ms) for n, ms in d["ladder"]["128"]["points"]]
    band = max(sweep.MEMORY_BRANCH_MARGIN, 3 * d["timing_spread_median"])

    published = sweep.fit_ladder(pts, 128, ref, margin=band)
    assert published.memory_points == 7
    assert published.alpha == pytest.approx(0.8841, abs=1e-3)

    nudged = sweep.fit_ladder(pts[:7] + [(8, 25.4909)], 128, ref, margin=band)
    assert nudged.memory_points == 0
    assert nudged.alpha is None
    assert (25.4909 / pts[7][1] - 1) < 1.5e-4, "the nudge must stay under 0.015%"


def test_the_a100_margin_clears_the_tolerance_by_one_part_in_ten_thousand(bm):
    """+0.000101, which is 0.07% of the tolerance and must not read as a pass."""
    d = _report(A100_G64)
    fit = d["ladder"]["128"]
    pts = [(int(n), ms) for n, ms in fit["points"]]
    margin = bm.margin_of(pts, fit["memory_points"],
                          c_ref=fit["slope_compute_ref"],
                          overhead=d["overhead_ms"],
                          spread=d["timing_spread_median"], draws=400, seed=0)
    assert margin.clears, "it does clear -- that is the point"
    # 0.000101 and not 0.000377: `margin_of` prefers the ladder's OWN compute
    # treads over the scaled reference, exactly as `fit_ladder` does, and at
    # BLOCK_M=128 that own branch is the single inverted tread 8. Using the
    # scaled reference instead gives 0.000377, still far inside the noise.
    assert margin.margin == pytest.approx(0.000101, abs=5e-6)
    assert margin.margin / bm.TOLERANCE < 0.001, "0.07% of the tolerance"
    assert margin.sigma < bm.MARGIN_SIGMA
    gate = bm.gate_margin(margin)
    assert gate.passed is False, (
        "clearing by 1e-4 must not print the same word as clearing by 0.3")


def test_a_margin_with_no_noise_model_refuses_instead_of_passing(bm):
    """REFUSE rather than default: no spread and no replicates is UNKNOWN."""
    pts = bm.planted_ladder(8, alpha=0.95, rho=175.0, block_m=128, b=2,
                            load_ms=4.0, overhead_ms=0.5)
    margin = bm.margin_of(pts, 8, c_ref=2.0, overhead=0.5, spread=None,
                          replicates=None)
    assert margin.sd is None and margin.sigma is None
    assert margin.basis, "the basis is never blank"
    assert bm.gate_margin(margin).passed is None


def test_the_level_bar_catches_the_corrupt_reference_and_spares_its_twin(bm):
    """43.6x too slow, and perfectly proportional, on the same pod minutes apart.

    This is the failure `compute_reference` cannot see: it tests the SHAPE of
    the reference ladder and never its LEVEL.
    """
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    ceiling = 262.3712016979615     # A100 measured bf16, this repo's calibration

    bad = _report(A100_BN256)["compute_reference"]
    good = _report(A100_BN64)["compute_reference"]
    assert bad["mean_rel_err"] < 0.03, (
        "the corrupt reference is PROPORTIONAL, which is why the shipped "
        "qualification accepted it")

    lo = bm.reference_level(cfg, bad["block_m"], bad["slope_per_tile"], ceiling,
                            "test")
    hi = bm.reference_level(cfg, good["block_m"], good["slope_per_tile"],
                            ceiling, "test")
    assert lo.fraction < 0.02 and not lo.passes
    assert 0.4 < hi.fraction < 0.8 and hi.passes
    assert bad["slope_per_tile"] / good["slope_per_tile"] > 40
    assert bm.gate_reference_level(lo).passed is False
    assert bm.gate_reference_level(hi).passed is True


def test_the_level_bar_also_refuses_a_reference_faster_than_the_card(bm):
    """Two sided. A reference implying 150% of peak is as broken as one at 1%."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    slow = _report(A100_BN64)["compute_reference"]["slope_per_tile"]
    impossible = bm.reference_level(cfg, 256, slow / 3.0, 262.371, "test")
    assert impossible.fraction > 1.0
    assert not impossible.passes


# --------------------------------------------------------------------------
# The convexity gate, and why it must be noise aware.
# --------------------------------------------------------------------------

def test_the_convexity_gate_is_quiet_on_a_ladder_planted_from_the_model(bm):
    """max-affine data has a non-decreasing slope by construction."""
    pts = bm.planted_ladder(8, alpha=0.40, rho=260.0, block_m=128, b=2,
                            load_ms=4.0, overhead_ms=0.5)
    assert bm.slope_drops(pts, 0.005) == []


def test_the_convexity_gate_fires_on_a_bend_when_the_noise_allows_it(bm):
    """A 30% fall at a 0.2% spread is a bend; the same fall at 5% is not.

    A FIXED relative threshold fired on 20 of the 22 published ladders, because
    a per-tread slope is a difference of timings and its noise grows with the
    tread index. The threshold is propagated instead, and that is what this pins.
    """
    bent = [(1, 2.0), (2, 4.0), (3, 6.0), (4, 9.0), (5, 12.5), (6, 15.0)]
    assert bm.slope_drops(bent, 0.002), "a clean bend must be reported"
    assert bm.slope_drops(bent, 0.05) == [], (
        "the same bend inside the noise must not be reported")
    assert bm.gate_convex([], bm.slope_sequence(bent), None).passed is None, (
        "with no spread the gate is UNKNOWN, never PASS")


# --------------------------------------------------------------------------
# Refusals, and the run id.
# --------------------------------------------------------------------------

def test_a_model_that_cannot_form_a_full_tile_stack_is_refused(bm):
    """deepseek-v2-lite is E=64 k=6, so rows per expert must be a multiple of 3.

    128 is not, so no tread at BLOCK_M=128 is an exactly-full tile stack. Nudging
    the row count would silently turn the ladder into a fit over padding, which
    is the confound this whole experiment is trying to keep out.
    """
    from moe.spec import MODEL_CONFIGS
    with pytest.raises(SystemExit) as exc:
        bm.ladder_rows(MODEL_CONFIGS["deepseek-v2-lite"], 128, 1024)
    assert "multiple of 3" in str(exc.value)
    # And a model whose quantum is 1 is fine, with one row per tread.
    rows = bm.ladder_rows(MODEL_CONFIGS["qwen2-57b-a14b"], 128, 1024)
    assert rows == [128, 256, 384, 512, 640, 768, 896, 1024]


def test_an_r_max_below_one_tile_is_refused_rather_than_returning_nothing(bm):
    from moe.spec import MODEL_CONFIGS
    with pytest.raises(SystemExit):
        bm.ladder_rows(MODEL_CONFIGS["qwen2-57b-a14b"], 128, 64)


def test_every_swept_knob_changes_the_run_id(bm):
    """A run id that omits a knob resumes into another experiment's directory.

    That already happened once in this repo: a GROUP_SIZE_M run derived the same
    id as the G=1 run, found every cell on disk, skipped all of them and printed
    the first run's timings under the second's heading.
    """
    args = bm.build_parser().parse_args([])
    base = bm.default_run_id(args, "nvidia_h200")
    for knob, value in (("group_m", 16), ("block_n", 256), ("block_k", 32),
                        ("num_stages", 3), ("num_warps", 4), ("reps", 3),
                        ("r_max", 2048), ("seed", 1), ("iters", 25),
                        ("model", "mixtral-8x7b"), ("dtype", "fp16"),
                        ("warmup", 5), ("cell_budget_ms", 800.0)):
        moved = bm.build_parser().parse_args([])
        setattr(moved, knob, value)
        assert bm.default_run_id(moved, "nvidia_h200") != base, (
            f"--{knob} does not change the run id, so two settings collide and "
            "the second silently reports the first's numbers")


def test_the_card_is_a_swept_knob_and_is_visible_in_the_run_id(bm):
    """THE CARD IS SWEPT BY THE OPERATOR MOVING PODS, AND THE VOLUME OUTLIVES
    THE POD.

    Every verdict in this file -- B/C, alpha x rho, both escape thresholds -- is
    scored against a per-card calibrated ridge: 145.8 Op/B on the A100 against
    152.8 on the H200 (162.8 until ab61e55 recalibrated the card on 2026-09-09,
    which is the point). `$MOE_RESULTS_DIR` is a RunPod network volume shared
    between pods, so without the card in the id the second card resumes into the
    first's directory, finds every tread present and reports them against its
    own ridge. That is a hybrid of two machines, which is the defect that put a
    stale H200 band into seven published A100 reports.
    """
    args = bm.build_parser().parse_args([])
    a100 = bm.default_run_id(args, "nvidia_a100_sxm4_80gb")
    h200 = bm.default_run_id(args, "nvidia_h200")
    assert a100 != h200
    assert a100.startswith("nvidia_a100_sxm4_80gb-")      # visible in `ls`
    assert h200.startswith("nvidia_h200-")


def test_a_dry_run_with_no_device_marks_its_path_as_not_the_pods(bm, capsys):
    """A laptop dry run must not print a path a pod will never write to: the
    next thing the operator does with that path is `git check-ignore` it."""
    assert bm.main(["--dry-run"]) == exit_codes.REFUSED
    out = capsys.readouterr().out
    assert bm.UNKNOWN_CARD_SLUG in out
    assert "NO DEVICE ATTACHED" in out


def test_the_card_flag_may_not_contradict_an_attached_device(bm, monkeypatch, capsys):
    """--card exists so a laptop can print the pod's real path. Letting it
    override a device that IS present would let one card write into another's
    directory, which is the collision the field was added to close."""
    monkeypatch.setattr(bm, "detect_card_slug", lambda: "nvidia_h200")
    assert bm.main(["--dry-run", "--card", "nvidia_a100_sxm4_80gb"]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_non_vacuity_fails_when_nothing_was_counted(bm):
    """A check that examined nothing also reports zero failures."""
    assert bm.gate_non_vacuity({"ladders": 0, "treads": 3}).passed is False
    assert bm.gate_non_vacuity({"ladders": 2, "treads": 3}).passed is True


# --------------------------------------------------------------------------
# The run's own plumbing, off GPU.
# --------------------------------------------------------------------------

def test_collapse_takes_the_median_across_repeats_and_reports_their_spread(bm):
    samples = [bm.Sample(128, n, n * 128, n * 128, rep, 10.0 * n + rep * 0.1,
                         0.0, 0.0, 5)
               for n in (1, 2, 3) for rep in range(1, 6)]
    points, reps, spread = bm.collapse(samples, 128)
    assert [n for n, _ in points] == [1, 2, 3]
    assert points[0][1] == pytest.approx(10.3)
    assert all(len(v) == 5 for v in reps.values())
    assert spread > 0
    # A failed timing is not a measurement and must not enter the median.
    samples.append(bm.Sample(128, 1, 128, 128, 6, 0.0, 0.0, 0.0, 0, "failed"))
    again, _, _ = bm.collapse(samples, 128)
    assert again[0][1] == pytest.approx(10.3)


def test_drift_reports_the_first_to_last_repeat_move(bm):
    samples = [bm.Sample(128, n, n * 128, n * 128, rep, 10.0 * n * (1 + 0.01 * rep),
                         0.0, 0.0, 5)
               for n in (1, 2) for rep in (1, 2, 3)]
    assert bm.drift(samples, 128) == pytest.approx(0.02 / 1.01, rel=1e-6)
    assert bm.drift(samples, 256) is None, "no data is None, never 0.0"


def test_replication_gate_is_unknown_without_a_spread(bm):
    assert bm._gate_replication({1: [1.0]}, None).passed is None
    assert bm._gate_replication({1: [1.0, 1.01]}, 0.001).passed is True
    assert bm._gate_replication({1: [1.0, 1.01]}, 0.5).passed is False


def test_the_override_assay_separates_a_resumed_setting_from_a_broken_one(bm):
    """Nothing compiled because nothing ran is UNDECIDED, not a failure."""
    assert bm._gate_override({128: 0, 256: 0}, {128: 0, 256: 0}).passed is None
    assert bm._gate_override({128: 0, 256: 4}, {128: 8, 256: 4}).passed is False
    assert bm._gate_override({128: 3, 256: 4}, {128: 8, 256: 4}).passed is True


# --------------------------------------------------------------------------
# The self test IS the claim that the gates discriminate.
# --------------------------------------------------------------------------

def test_the_planted_worlds_come_out_as_registered(bm):
    """Feasible, straddling, feasible, throttled. A gate that cannot tell them
    apart is not a gate, and would have passed the published A100 ladder too."""
    _, gates = bm.self_test(noise=bm.PUBLISHED_LADDER_SPREAD)
    assert len(gates) == len(bm.SELF_TEST_WORLDS)
    assert all(g.passed is True for g in gates), \
        [(g.name, g.observed) for g in gates if g.passed is not True]


def test_the_self_test_runs_the_pod_s_own_analysis_and_not_a_shortcut(bm):
    """R5. THE DEFECT: the planted worlds never touched the measured path.

    The retired self test built `(tread, ms)` pairs by hand, counted membership
    from the PLANTED compute slope and called `margin_of`. `compute_reference`,
    `fit_ladder`, and every gate in `analyse_run` -- the path that produced the
    43.6x reference the level check exists to catch -- were never called, so a
    regression in any of them passed. This asserts the self test now reaches
    them, by breaking one and watching the self test notice.
    """
    calls = []
    real = bm.analyse_run

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    bm.analyse_run = spy
    try:
        _, gates = bm.self_test(noise=bm.PUBLISHED_LADDER_SPREAD)
    finally:
        bm.analyse_run = real
    assert len(calls) == len(bm.SELF_TEST_WORLDS)
    assert all(g.passed is True for g in gates)


def test_a_regression_in_the_measured_path_is_caught_by_the_self_test(bm):
    """The FAIL branch of every S gate, planted.

    A self test whose gates cannot fail certifies nothing. Breaking the
    membership margin -- the constant that decides which treads are memory
    bound, and the one whose n=1 anchoring made BLOCK_M=128 unidentifiable in
    every published H200 arm -- must be visible in at least one world.
    """
    old = bm.SWEEP.MEMORY_BRANCH_MARGIN
    bm.SWEEP.MEMORY_BRANCH_MARGIN = 0.95      # nothing can be memory bound
    try:
        _, gates = bm.self_test(noise=bm.PUBLISHED_LADDER_SPREAD)
    finally:
        bm.SWEEP.MEMORY_BRANCH_MARGIN = old
    assert any(g.passed is not True for g in gates), (
        "a margin of 0.95 makes every tread compute bound; a self test that "
        "still passes is not reading the fit")


def test_every_registered_world_names_gates_the_report_actually_has(bm):
    """A registration that silently matches nothing is the
    check-that-examined-nothing shape one level up."""
    world = bm.SELF_TEST_WORLDS[0]
    bogus = bm.PlantedWorld(world.name, world.alpha, world.rho, world.why,
                            {"V9": True})
    bad = bogus.check([bm.Gate(bm.VALIDITY, "V0 x", "p", "r", True, "o")])
    assert len(bad) == 1 and "no gate V9" in bad[0]


def test_a_zero_noise_self_test_is_refused_rather_than_run(bm):
    """R2/B14. Both planted worlds used to be noiseless, so every threshold
    propagated from a measured spread was exercised at the one value no pod
    produces."""
    with pytest.raises(bm.SelfTestRefused) as exc:
        bm.self_test(noise=0.0)
    assert "no pod produces" in str(exc.value)


def test_a_spread_past_v5_s_own_ceiling_is_refused_with_the_reason(bm):
    with pytest.raises(bm.SelfTestRefused) as exc:
        bm.self_test(noise=bm.MAX_REPLICATE_SPREAD * 2)
    assert "V5 WITHDRAWS its verdict" in str(exc.value)


def test_the_undecided_outcomes_are_reported_and_not_treated_as_none(bm):
    """R6. `fit_ladder` gained six named outcomes and two of them are states
    where the sweep LOOKED AND COULD NOT SAY.

    Read as a blank, `undecided_parallel_branch` publishes "no depth" from a fit
    that named none, and `undecided_drifting_clock` publishes a settling
    governor as the tile. Both must reach the report as UNKNOWN with the
    outcome named.
    """
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    worlds = {w.name: w for w in bm.SELF_TEST_WORLDS}
    for name, outcome in (("straddle", "undecided_parallel_branch"),
                          ("drifting-clock", "undecided_drifting_clock")):
        w = worlds[name]
        samples = bm.planted_samples(
            cfg, alpha=w.alpha, rho=w.rho,
            bandwidth_gbps=bm.SELF_TEST_BANDWIDTH,
            noise=bm.PUBLISHED_LADDER_SPREAD,
            low_clock_treads=w.low_clock_treads,
            drift_treads=w.drift_treads)
        lines, gates, payload = bm.analyse_run(
            samples, cfg, 2,
            ceiling_tflops=w.rho * bm.SELF_TEST_BANDWIDTH * 1e9 / 1e12,
            ceiling_source="test", compiles={128: 1, 256: 1},
            executed={128: 40, 256: 20}, ridge=w.rho,
            bandwidth_gbps=bm.SELF_TEST_BANDWIDTH, draws=200)
        assert payload["ladder_outcome"] == outcome, name
        assert payload["ladder_undecided"] is True, name
        by = {g.tag: g for g in gates}
        assert by["C1"].passed is None, name
        assert outcome in by["C1"].observed
        assert any("ladder outcome" in ln for ln in lines), name


def test_a_drifted_tread_is_excluded_from_the_fit_and_counted(bm):
    """R7. The instrument's clock columns have to reach the fit.

    A tread whose clock MOVED across its own trials has a median that blends
    two operating points, so it sits above the compute branch for a reason that
    has nothing to do with weight re-reads. Before the columns travelled, a
    ladder fitted across a settling governor was indistinguishable from one
    that was not.

    STEADILY LOW IS NOT THIS FAULT AND IS NOT EXCLUDED since 2026-09-09: the
    third ladder below plants the same six treads steadily low and every one of
    them stays in the fit, counted on its side.
    """
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    kw = dict(alpha=0.95, rho=175.0, bandwidth_gbps=bm.SELF_TEST_BANDWIDTH,
              noise=bm.PUBLISHED_LADDER_SPREAD)
    quiet = bm.planted_samples(cfg, **kw)
    moved = bm.planted_samples(cfg, drift_treads=(3, 4, 5, 6, 7, 8), **kw)
    cold = bm.planted_samples(cfg, low_clock_treads=(3, 4, 5, 6, 7, 8), **kw)
    args = dict(ceiling_tflops=175.0 * bm.SELF_TEST_BANDWIDTH * 1e9 / 1e12,
                ceiling_source="test", compiles={128: 1, 256: 1},
                executed={128: 40, 256: 20}, ridge=175.0,
                bandwidth_gbps=bm.SELF_TEST_BANDWIDTH, draws=200)
    _, _, quiet_pay = bm.analyse_run(quiet, cfg, 2, **args)
    lines, _, moved_pay = bm.analyse_run(moved, cfg, 2, **args)
    _, _, cold_pay = bm.analyse_run(cold, cfg, 2, **args)
    assert quiet_pay["excluded_drifted"] == 0
    assert moved_pay["excluded_drifted"] == 6
    assert moved_pay["memory_points"] < quiet_pay["memory_points"]
    assert any("EXCLUDED" in ln for ln in lines)
    assert cold_pay["excluded_drifted"] == 0
    assert cold_pay["kept_low_clock"] == 6
    assert len(cold_pay["scored_points"]) == len(quiet_pay["scored_points"])


def test_a_tread_is_excluded_on_a_majority_of_its_repeats_not_on_one(bm):
    """One throttled repeat out of seven is what the median exists to absorb.
    A failed repeat carries its side, as the instrument's does; without one
    `Sample` refuses it."""
    def sample(rep, level):
        return bm.Sample(128, 4, 512, 2048, rep, 1.0, 1.0, 0.0, 10,
                         clock_level_ok=level,
                         clock_level_side="low" if level is False else "")
    one_bad = [sample(1, False)] + [sample(r, True) for r in range(2, 8)]
    most_bad = [sample(r, False) for r in range(1, 6)] + [sample(6, True)]
    unknown = [sample(r, None) for r in range(1, 4)]
    assert bm.tread_clock(one_bad, 128)[4][0] is True
    assert bm.tread_clock(most_bad, 128)[4][0] is False
    assert bm.tread_clock(unknown, 128)[4][0] is None


def test_the_law_gate_is_a_validity_check_and_says_it_is_not_a_prediction(bm):
    """R5. `C3 the law predicts the depth` compared `n*` computed from
    `(alpha, B/C)` against the tread count those same numbers were fitted on.

    For any ladder that really is two lines those agree by construction, so it
    could only fail when the model does not fit -- a statement about the FIT.
    Carried as a CLAIM it read as a confirmed prediction and a reader counting
    the claim gates counted it as evidence for the law.
    """
    cfg, samples, ceiling = _planted_samples(bm, alpha=0.95, rho=175.0,
                                             bandwidth_gbps=1799.4)
    _, gates, _ = bm.analyse_run(
        samples, cfg, 2, ceiling_tflops=ceiling, ceiling_source="test",
        compiles={128: 3, 256: 4}, executed={128: 40, 256: 20},
        ridge=175.0, bandwidth_gbps=1799.4, draws=200)
    law = next(g for g in gates if g.tag == "V6")
    assert law.kind == bm.VALIDITY
    assert not any(g.tag == "C3" for g in gates)
    assert "NOT A PREDICTION" in " ".join(law.lines)


def _world(bm, name):
    return {w.name: w for w in bm.SELF_TEST_WORLDS}[name]


def _run_world(bm, world, *, seed=0, noise=None, low_clock=None, drifting=None):
    """One planted world through `analyse_run`, the way `self_test` runs it."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    samples = bm.planted_samples(
        cfg, alpha=world.alpha, rho=world.rho,
        bandwidth_gbps=bm.SELF_TEST_BANDWIDTH,
        noise=bm.PUBLISHED_LADDER_SPREAD if noise is None else noise, seed=seed,
        low_clock_treads=(world.low_clock_treads if low_clock is None
                          else low_clock),
        high_clock_treads=world.high_clock_treads,
        drift_treads=(world.drift_treads if drifting is None else drifting))
    return bm.analyse_run(
        samples, cfg, 2,
        ceiling_tflops=world.rho * bm.SELF_TEST_BANDWIDTH * 1e9 / 1e12,
        ceiling_source="test", compiles={128: 1, 256: 1},
        executed={128: 40, 256: 20}, ridge=world.rho,
        bandwidth_gbps=bm.SELF_TEST_BANDWIDTH, seed=seed, draws=200)


# --------------------------------------------------------------------------
# The exit code each world would return on a pod, which is all a driver sees.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_every_world_returns_its_registered_exit_code_at_every_seed(bm, seed):
    """THE NUMBER THE SESSION DRIVER READS, and the one the fixture never had.

    `scripts/h200_gaps_session.sh` lists 0 as this arm's only done code and
    turns everything else into RETRY with "nothing on the page may be quoted".
    The four S gates scored per-gate verdicts and never the exit code, so for
    one commit all four passed while every world would have exited INVALID on a
    pod. Registered per world now, and checked across seeds because a
    registration that only holds at seed 0 is a registration on a coin flip.
    """
    for world in bm.SELF_TEST_WORLDS:
        _, gates, _ = _run_world(bm, world, seed=seed)
        rc = exit_codes.classify(g.scored() for g in gates)
        assert rc == world.exit_code, (
            f"{world.name} at seed {seed}: registered "
            f"{exit_codes.describe(world.exit_code)}, got "
            f"{exit_codes.describe(rc)}")


def test_the_arms_own_expected_outcome_is_a_result_and_not_a_void_page(bm):
    """THE STRADDLE WORLD IS WHERE BOTH CARDS SIT, and it must not be RETRY.

    `undecided_parallel_branch` is what every published H200 BLOCK_M=128 arm
    returns: `B/C` inside `PARALLEL_BRANCH_TOLERANCE`, so the fit LOOKED and
    declined to name a branch. When `C3` was relabelled to a VALIDITY gate that
    declined on `fit.undecided`, this condition -- the arm's own registered
    expectation -- produced INVALID, and the twelve metered minutes were RETRY
    by construction on both cards. C1's UNKNOWN is the verdict, CLAIM_FAIL is
    the code, and `_exit_code` returns it: CLAIM_FAIL is in `FINISHED_CODES`
    and `ledger_state(1)` is "CLAIM_FAIL", so the ledger already reads it as a
    result rather than a retry.
    """
    world = _world(bm, "straddle")
    assert world.exit_code == exit_codes.CLAIM_FAIL
    _, gates, payload = _run_world(bm, world)
    assert payload["ladder_outcome"] == "undecided_parallel_branch"
    by = {g.tag: g for g in gates}
    assert by["C1"].passed is None, "the depth verdict is C1's"
    assert by["C1"].kind == bm.CLAIM
    assert all(g.passed is True for g in gates if g.kind == bm.VALIDITY), \
        [(g.tag, g.observed) for g in gates
         if g.kind == bm.VALIDITY and g.passed is not True]
    assert exit_codes.classify(g.scored() for g in gates) != exit_codes.INVALID


def test_a_world_whose_exit_code_is_wrong_fails_the_self_test(bm, monkeypatch,
                                                              capsys):
    """The FAIL branch of the exit-code registration, planted.

    Every registered gate still matches; only the integer is wrong. A fixture
    that could not notice that is the fixture that shipped V6 as a VALIDITY
    gate.
    """
    world = _world(bm, "escape-up")
    assert world.exit_code == exit_codes.DONE
    wrong = bm.PlantedWorld(world.name, world.alpha, world.rho, world.why,
                            world.expect, exit_code=exit_codes.INVALID,
                            low_clock_treads=world.low_clock_treads)
    monkeypatch.setattr(bm, "SELF_TEST_WORLDS", (wrong,))
    assert bm.main(["--self-test"]) == exit_codes.INVALID
    out = capsys.readouterr().out
    assert "exit code: registered" in out


# --------------------------------------------------------------------------
# V6's scope: the alpha in the report, and nothing else.
# --------------------------------------------------------------------------

class _Fit:
    """The three fields `_gate_law` reads, so each branch can be planted."""

    def __init__(self, alpha, outcome, undecided, memory_points=5, points=8):
        self.alpha = alpha
        self.outcome = outcome
        self.undecided = undecided
        self.memory_points = memory_points
        self.points = tuple((n, 1.0) for n in range(1, points + 1))


class _Margin:
    def __init__(self, ratio):
        self.ratio = ratio


def test_v6_passes_vacuously_when_the_report_publishes_no_alpha(bm):
    """A VALIDITY gate that voids the page over a number the page does not
    contain withholds nothing and costs the whole arm."""
    gate = bm._gate_law(_Fit(None, "undecided_parallel_branch", True,
                             memory_points=0),
                        _Margin(1.08), 1.4, 0.05, 0.02)
    assert gate.kind == bm.VALIDITY
    assert gate.passed is True
    assert "VACUOUS" in gate.observed
    assert "undecided_parallel_branch" in gate.observed
    assert "NOT a statement that the ladder is two lines" in " ".join(gate.lines)


def test_v6_still_withholds_an_alpha_it_could_not_certify(bm):
    """`undecided_drifting_clock` leaves a fit over the treads that survived,
    and `payload["alpha"]` carries it. Certifying that number would publish
    exactly what the exclusion exists to withhold, so the gate returns UNKNOWN
    and the page is INVALID.

    THE TOKEN PLANTED HERE WAS `undecided_low_clock` UNTIL 2026-09-09, and it
    stayed after `LADDER_OUTCOMES` dropped that name: the test then exercised
    an outcome the code cannot emit and would have kept passing if `_gate_law`
    stopped handling the real one. The plant is asserted to be a real outcome
    so it cannot go stale silently again."""
    outcome = bm.SWEEP.UNDECIDED_DRIFTING_CLOCK
    assert outcome in bm.SWEEP.LADDER_OUTCOMES
    gate = bm._gate_law(_Fit(0.89, outcome, True, memory_points=2),
                        _Margin(1.36), 1.4, 0.05, 0.02)
    assert gate.passed is None
    assert "0.8900" in gate.observed
    assert outcome in gate.observed
    assert not hasattr(bm.SWEEP, "UNDECIDED_LOW_CLOCK")


def test_v6_is_unknown_when_there_is_an_alpha_but_no_compute_slope(bm):
    gate = bm._gate_law(_Fit(0.33, "identified", False), _Margin(None), None,
                        0.05, 0.02)
    assert gate.passed is None
    assert "no usable compute slope" in gate.observed


def test_v6_still_fails_when_the_fit_and_the_count_disagree(bm):
    """The FAIL branch, which is the only reason the PASS means anything."""
    gate = bm._gate_law(_Fit(0.95, "identified", False, memory_points=2),
                        _Margin(1.40), 1.4, 0.05, 0.02)
    assert gate.passed is False
    assert "fit found 2" in gate.observed


# --------------------------------------------------------------------------
# The clock exclusion reaches every gate that reads the subject ladder.
# --------------------------------------------------------------------------

def test_a_drifted_tread_cannot_trip_a_validity_gate_it_was_excluded_from(bm):
    """THE EXCLUSION USED TO STOP AT THE FIT.

    `ladder_treads` drops a tread whose clock moved mid-measurement, but V2's
    inversions, V3's slope sequence and V5's replication were all still
    computed over the UNEXCLUDED medians. A ladder that drifted partway BENDS
    at the onset -- the planted world's slope runs 1.9 -> 3.0 -> 2.2 -- and that
    bend is a slope that rises and then falls, which V3 scores FAIL: INVALID,
    nothing quotable, for exactly the reason the exclusion exists to discount.

    The two verdicts point at different next steps. "The ladder is not
    describable by any two-line model" says the instrument is broken; "six
    treads were timed while the clock was moving" says re-time on an instrument
    that warms until it settles. Only the second is true here, and this pins
    that the second is what comes back.
    """
    world = _world(bm, "drifting-clock")
    _, gates, payload = _run_world(bm, world)
    by = {g.tag: g for g in gates}
    # What the gate would have said over the unexcluded ladder.
    all_points = payload["subject_points"]
    all_spread = payload["subject_spread"]
    would_have = bm.gate_convex(bm.slope_drops(all_points, all_spread),
                                bm.slope_sequence(all_points), all_spread)
    assert would_have.passed is False, (
        "the planted world must actually bend, or this test proves nothing")
    # What it says over the treads the fit admitted.
    assert payload["excluded_drifted"] == 6
    assert len(payload["scored_points"]) == 2
    assert by["V3"].passed is None
    assert "fewer than three treads" in by["V3"].observed
    assert payload["ladder_outcome"] == "undecided_drifting_clock"


def test_the_spread_the_gates_weigh_comes_from_the_treads_they_scored(bm):
    """A spread taken over excluded treads is a noise band measured partly on a
    blend of two clock states, and it is the denominator of every sigma."""
    world = _world(bm, "drifting-clock")
    _, _, payload = _run_world(bm, world)
    assert payload["scored_spread"] != payload["subject_spread"]
    assert payload["scored_spread"] is not None


def test_every_tread_is_still_printed_with_the_dropped_ones_marked(bm):
    """The gates score the survivors; the reader has to see what was dropped,
    or the excluded count in the header points at nothing."""
    world = _world(bm, "drifting-clock")
    lines, _, _ = _run_world(bm, world)
    text = "\n".join(lines)
    assert text.count(
        "NO: DRIFT, the clock moved across its own trials") == 6
    assert text.count("  yes") >= 2
    # And each KEPT side is marked on every one of its rows, with the direction
    # its fixed-roof fraction is off in, rather than passing as an unremarkable
    # "yes".
    high = _world(bm, "high-clock")
    text = "\n".join(_run_world(bm, high)[0])
    assert text.count("NO: DRIFT") == 0
    assert text.count("(LEVEL high, kept; fixed-roof fraction overstated "
                      "by the clock ratio)") == 6
    low = _world(bm, "low-clock")
    text = "\n".join(_run_world(bm, low)[0])
    assert text.count("NO: DRIFT") == 0
    assert text.count("(LEVEL low, kept; fixed-roof fraction understated "
                      "by the clock ratio)") == 8


def test_a_ladder_whose_every_tread_was_excluded_is_vacuous_and_says_so(bm):
    """V0's counts are the report's own INPUTS, and after the exclusion the
    input is the scored ladder. A report that measured eight treads and scored
    none examined nothing, whatever the measured count says."""
    world = _world(bm, "drifting-clock")
    _, gates, payload = _run_world(bm, world,
                                   drifting=tuple(range(1, 9)))
    assert payload["scored_points"] == []
    v0 = next(g for g in gates if g.tag == "V0")
    assert v0.passed is False
    assert "scored treads" in " ".join(v0.lines)


# --------------------------------------------------------------------------
# V1: the gate the 43.6x reference exists to catch, planted with a PASS branch.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_the_reference_level_gate_is_not_a_coin_flip_on_the_seed(bm, seed):
    """R5's purpose was that a regression in the path that produced the 43.6x
    reference can no longer pass the self test, and for one commit V1 was the
    one gate the fixture could not exercise.

    `planted_samples` generated the reference ladder at exactly `rho x
    bandwidth`, which is `reference_level`'s own UPPER wall, so at the default
    spread seeds 0/3/5 gave FAIL and 1/2/4 gave PASS. It was unregistered in
    three worlds and registered only as UNKNOWN in the fourth, so its PASS
    branch was planted nowhere. `REFERENCE_KERNEL_EFFICIENCY` puts the reference
    a few per cent under the ceiling, where a real grouped GEMM is.
    """
    assert bm.REFERENCE_KERNEL_EFFICIENCY < 1.0
    for name in ("escape-up", "straddle", "low-clock"):
        _, gates, payload = _run_world(bm, _world(bm, name), seed=seed)
        frac = payload["reference"]["level_fraction"]
        assert bm.REFERENCE_LEVEL_FLOOR <= frac <= 1.0, (name, seed, frac)
        assert frac < 1.0, "the reference must not sit ON the gate's wall"
        assert next(g for g in gates if g.tag == "V1").passed is True


def test_the_level_gate_still_fails_a_reference_that_is_too_slow(bm):
    """The FAIL branch, on the shape that produced it: the published A100
    BLOCK_N=256 reference was 43.6x too slow while perfectly proportional."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    # 2.0 ms/tile implies ~451 TFLOP/s of a 500 ceiling; 43.6x that is 2%.
    fast = bm.reference_level(cfg, 256, 2.0, 500.0, "test")
    slow = bm.reference_level(cfg, 256, 2.0 * 43.6, 500.0, "test")
    assert 0.85 < fast.fraction < 1.0 and slow.fraction < 0.05
    assert bm.gate_reference_level(fast).passed is True
    assert bm.gate_reference_level(slow).passed is False
    assert bm.gate_reference_level(None).passed is None


def test_a_throttled_tread_is_slower_and_not_merely_labelled(bm):
    """THE FIXTURE PLANTED THE LABEL AND NOT THE PHYSICS.

    `planted_samples` used to stamp the clock columns on the low rows and
    compute their milliseconds identically to every other row, so no gate could
    tell a throttled ladder from a clean one and `ladder_treads`' claim -- that
    a ladder which lost treads to a hot box must not look like one that never
    had them -- was demonstrated by nothing. A lower SM clock is a lower
    compute roof and the same DRAM bandwidth, so the throttled treads are
    slower and the ladder bends where they start.
    """
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    kw = dict(alpha=0.95, rho=175.0, bandwidth_gbps=bm.SELF_TEST_BANDWIDTH,
              noise=0.0)
    quiet = {(s.block_m, s.tiles): s.ms_p50
             for s in bm.planted_samples(cfg, **kw)}
    hot = {(s.block_m, s.tiles): s.ms_p50
           for s in bm.planted_samples(cfg, low_clock_treads=(3, 4, 5, 6, 7, 8),
                                       **kw)}
    for n in (1, 2):
        assert hot[(128, n)] == quiet[(128, n)], "clean treads must not move"
    for n in (3, 4, 5, 6, 7, 8):
        assert hot[(128, n)] > quiet[(128, n)] * 1.05, (
            f"tread {n} carries the clock label and none of its physics")
    assert bm.SELF_TEST_LOW_CLOCK_MHZ < bm.SELF_TEST_REFERENCE_CLOCK_MHZ


def test_the_audit_runs_end_to_end_and_examined_real_work(bm):
    """The whole off-GPU answer, over the committed corpus."""
    if not PUBLISHED.exists():
        pytest.skip("no results/published on this checkout")
    records, skipped = bm.load_corpus(PUBLISHED)
    assert records, "the corpus loader found no BLOCK_M=128 ladder at all"
    rows = [bm.audit_record(r, draws=200) for r in records]
    _, gates, payload = bm.audit_report(rows, skipped)
    vacuity = next(g for g in gates if g.name.startswith("V0"))
    assert vacuity.passed is True
    assert payload["counts"]["BM=128 ladders"] == len(records)
    # The headline: no published BLOCK_M=128 fit is admissible.
    assert not any(r.admissible for r in rows)


def test_the_founding_premise_no_longer_straddles_the_ridge(bm):
    """THE RETRACTION, recomputed from the two published BM=128 report blocks.

    The module docstring asserted "cap 150.4 against a calibrated ridge of 145.8
    on the A100, 158.6 against 162.8 on the H200" at the same HEAD where
    `moe/bench/ai_model.py` retracts the expression both came from. This runs
    `premise_caps` and asserts what the corrected numbers actually say: the
    retracted cap cleared the A100's ridge, the corrected one clears NEITHER.

    WHICH ALPHA WENT IN IS PINNED SEPARATELY FROM THE CAPS, because that is the
    thing this function got wrong twice. `exa_cap` has to be
    `cap_from_fitted` at the ladder's RAW `alpha`, so the assertion below
    rebuilds it from `alpha_fitted`, the fused `phi` and the delta floor, and
    then re-derives `alpha_b` through `ai_model.alpha_b_from_fitted`. Feeding
    `alpha_corrected` into an `alpha_b` slot -- the second version's error --
    gives 140.4 and 139.9 and fails here; dividing the retracted cap by
    `1 + phi + delta` -- the first version's -- gives 141.8 and 143.1 and fails
    here too.
    """
    if not PUBLISHED.exists():
        pytest.skip("no results/published on this checkout")
    caps = {c.arm[:24]: c for c in bm.premise_caps(PUBLISHED)}
    assert len(caps) == 2, sorted(caps)
    a100 = next(c for c in caps.values() if "a100" in c.arm)
    h200 = next(c for c in caps.values() if "h200" in c.arm)

    assert a100.lin_straddles and a100.lin_cap == pytest.approx(150.4, abs=0.1)
    assert h200.lin_cap == pytest.approx(158.6, abs=0.1)
    for c in (a100, h200):
        cfg = MODEL_CONFIGS[c.model]
        # phi is the FUSED layer's Act1/W and carries no alpha_a. The
        # single-GEMM phi the bracket used is a different number, and both ends
        # of that bracket are asserted to differ from this one so a silent
        # return to it cannot pass.
        assert c.phi == pytest.approx(bm.fused_phi(cfg, "bf16"), rel=1e-12)
        gemm_phi = {
            a: ai_model.phi(2 * cfg.intermediate_size, cfg.hidden_size,
                            block_m=bm.SUBJECT_BLOCK_M, block_n=64, alpha_a=a)
            for a in (0.0, 1.0)}
        assert abs(gemm_phi[0.0] - c.phi) / c.phi > 0.10
        assert gemm_phi[1.0] / c.phi > 15.0
        assert c.delta == bm.PREMISE_DELTA_FLOOR == 0.0
        assert c.alpha_b == pytest.approx(
            ai_model.alpha_b_from_fitted(c.alpha_fitted, phi=c.phi,
                                         delta=c.delta), rel=1e-12)
        assert c.exa_cap == pytest.approx(
            ai_model.cap_from_fitted(c.alpha_fitted, block_m=bm.SUBJECT_BLOCK_M,
                                     b=2, phi=c.phi, delta=c.delta), rel=1e-12)
        # The two expressions this function used to print, rebuilt and refused.
        # Both are reproduced exactly, at the alpha and the phi they each used,
        # so "the number moved" cannot be mistaken for a rounding difference.
        wrong_slot = ai_model.exact_cap(
            2 * cfg.intermediate_size, cfg.hidden_size,
            block_m=bm.SUBJECT_BLOCK_M, block_n=64,
            alpha_b=c.alpha_corrected, alpha_a=0.0)
        wrong_factor = c.lin_cap / ai_model.lin_overstatement(
            phi=gemm_phi[0.0], delta=0.0)
        assert c.exa_cap < wrong_slot and c.exa_cap < wrong_factor
        withdrawn = {"a100": (140.4, 141.8), "h200": (139.9, 143.1)}
        slot, factor = withdrawn["a100" if "a100" in c.arm else "h200"]
        assert wrong_slot == pytest.approx(slot, abs=0.1)
        assert wrong_factor == pytest.approx(factor, abs=0.1)
        assert c.factor == pytest.approx(c.lin_cap / c.exa_cap, rel=1e-12)
        assert c.factor > 1.0
        assert not c.straddles, (c.arm, c.exa_cap, c.ridge)
    assert a100.exa_cap == pytest.approx(135.4, abs=0.1)
    assert h200.exa_cap == pytest.approx(130.7, abs=0.1)


def test_the_ladders_own_delta_is_reported_and_one_of_them_has_no_cap(bm):
    """`delta = 0` is a FLOOR, and the page has to say what the ladder pins.

    The headline cap is read at the floor because delta enters (EXA) only
    through the level, so a larger delta recovers a larger `alpha_b` and a
    smaller cap: the floor gives the straddle every benefit. That is only
    honest if the pinned value is printed too. The A100 ladder's own
    `alpha_upper` pins `D/L = 0.13475`, and at the delta that implies the
    recovered `alpha_b` lands above 1 and `ai_model` refuses it -- there is no
    corrected cap at all there, which is a stronger retraction and not a reason
    to drop the ladder. The H200 ladder's `overhead_ms` is 0.0, so its pinned
    delta IS the floor and its cap is unchanged: both branches, on real data.
    """
    if not PUBLISHED.exists():
        pytest.skip("no results/published on this checkout")
    caps = {c.arm[:24]: c for c in bm.premise_caps(PUBLISHED)}
    a100 = next(c for c in caps.values() if "a100" in c.arm)
    h200 = next(c for c in caps.values() if "h200" in c.arm)

    assert a100.delta_implied == pytest.approx(0.16648, abs=1e-5)
    assert a100.exa_cap_at_delta_implied is None
    assert "above the ceiling" in a100.delta_implied_refusal
    assert ai_model.alpha_b_from_fitted(
        a100.alpha_fitted, phi=a100.phi,
        delta=0.0) < 1.0 < a100.alpha_fitted * (1 + a100.phi
                                                + a100.delta_implied) - a100.phi

    assert h200.delta_implied == pytest.approx(0.0, abs=1e-12)
    assert h200.delta_implied_refusal == ""
    assert h200.exa_cap_at_delta_implied == pytest.approx(h200.exa_cap,
                                                          rel=1e-12)


def test_a_ladder_with_no_alpha_upper_pins_no_delta(bm, tmp_path):
    """None and zero are different answers and print differently.

    `D = 0` is a measurement the H200 ladder actually makes; a block with no
    `alpha_upper` in it pins nothing, and reporting that as "delta 0.00000"
    would put a measurement's name on an absence.
    """
    doc = json.loads(A100_G64.read_text())
    doc["ladder"]["128"].pop("alpha_upper")
    arm = tmp_path / "2026-09-02-nvidia_a100_sxm4_80gb-planted"
    arm.mkdir(parents=True)
    (arm / A100_G64.name).write_text(json.dumps(doc))
    (cap,) = bm.premise_caps(tmp_path)
    assert cap.delta_implied is None
    assert cap.exa_cap_at_delta_implied is None
    assert cap.exa_cap == pytest.approx(135.4, abs=0.1)
    assert any("pins no delta" in ln for ln in bm.render_premise(tmp_path))

    # An alpha-upper BELOW alpha is a negative D, which is not a fixed cost.
    # Same answer, and it has to come from the ratio guard rather than from the
    # missing-key one, so it is planted separately.
    doc["ladder"]["128"]["alpha_upper"] = 0.5
    (arm / A100_G64.name).write_text(json.dumps(doc))
    (cap,) = bm.premise_caps(tmp_path)
    assert cap.delta_implied is None
    assert bm.implied_delta(0.88412, 0.5, 0.069) is None
    assert bm.implied_delta(0.88412, 1.02180, 0.06905) == pytest.approx(
        0.16648, abs=1e-4)


def test_a_non_positive_alpha_is_named_rather_than_divided_by(bm, tmp_path):
    """`2 BM / (b alpha)` has no value at alpha = 0, and neither does its
    correction, so the ladder is dropped WITH ITS NAME in the refusal.

    The corpus is small enough that a silently dropped ladder is the difference
    between a premise recomputed from two fits and one recomputed from one.
    """
    doc = json.loads(A100_G64.read_text())
    doc["ladder"]["128"]["alpha"] = 0.0
    arm = tmp_path / "2026-09-02-nvidia_a100_sxm4_80gb-planted"
    arm.mkdir(parents=True)
    (arm / A100_G64.name).write_text(json.dumps(doc))
    with pytest.raises(bm.PremiseNotRecomputable) as exc:
        bm.premise_caps(tmp_path)
    assert "nvidia_a100_sxm4_80gb-planted" in str(exc.value)
    assert "is not positive" in str(exc.value)


def test_the_fused_phi_matches_both_siblings_byte_for_byte(bm, tmp_path):
    """`Act1 / W` is TRANSCRIBED into three files, so it is checked across them.

    `block_m_crossing_sweep.activation_bytes_per_row` and
    `memory_branch_anchor.activation_bytes_per_row` are the other two. A premise
    that changes because a sibling refactored a helper is not a premise; a
    premise that silently DISAGREES with the two files it says it matches is
    worse, and the only way to know is to run all three.
    """
    sweep = _load("block_m_crossing_sweep", "block_m_crossing_sweep.py")
    anchor = _load("memory_branch_anchor", "memory_branch_anchor.py")
    for name in ("qwen2-57b-a14b", "deepseek-v2-lite", "mixtral-8x7b"):
        cfg = MODEL_CONFIGS[name]
        per_row = sweep.activation_bytes_per_row(cfg)
        assert anchor.activation_bytes_per_row(cfg) == per_row
        for block_m in (32, 64, 128, 256):
            act1 = cfg.num_experts * block_m * per_row
            assert bm.fused_phi(cfg, "bf16", block_m) == pytest.approx(
                act1 / cfg.weight_bytes("bf16"), rel=1e-12)
            w, anchor_act1 = anchor.anchor_bytes(cfg, "bf16", block_m)
            assert (anchor_act1 / w) == pytest.approx(
                bm.fused_phi(cfg, "bf16", block_m), rel=1e-12)


def test_the_premise_still_straddles_when_the_alpha_is_small_enough(bm, tmp_path):
    """The PASS branch of the same predicate, planted.

    A retraction asserted only by a test that always answers "no" is a constant.
    Here the A100 report is copied with its BLOCK_M=128 fitted alpha lowered to
    0.80, which lifts the CORRECTED cap back over 145.8, and `straddles` says
    so. So the property is a property of the published alphas, not of the code.

    `alpha_corrected` is lowered with it, because a corrected alpha ABOVE the
    raw one is not a state the sweep can produce -- the correction subtracts --
    and planting one would make the retracted column nonsense.
    """
    doc = json.loads(A100_G64.read_text())
    doc["ladder"]["128"]["alpha"] = 0.80
    doc["ladder"]["128"]["alpha_corrected"] = 0.77
    doc["ladder"]["128"]["alpha_upper"] = 0.80
    arm = tmp_path / "2026-09-02-nvidia_a100_sxm4_80gb-planted"
    arm.mkdir(parents=True)
    (arm / A100_G64.name).write_text(json.dumps(doc))
    (cap,) = bm.premise_caps(tmp_path)
    assert cap.straddles
    assert cap.exa_cap > cap.ridge
    assert any("STILL STRADDLES" in ln for ln in bm.render_premise(tmp_path))


def test_the_audit_prints_the_premise_beside_its_retracted_form(bm, capsys):
    """`--audit` is the evidence for the module docstring, so the premise has to
    be ON that page and not only in the prose it went stale in.

    All of it is required: a corrected cap printed alone is a number the reader
    cannot compare with the published one, which is how a 32% correction stayed
    invisible, and a cap printed without its `phi` and `delta` is a number whose
    reader cannot tell which alpha went into it, which is how it was then
    corrected wrongly twice. `140.4` and `141.8` are the two withdrawn values;
    both are asserted ABSENT so a regression to either cannot pass silently.
    """
    if not PUBLISHED.exists():
        pytest.skip("no results/published on this checkout")
    bm.main(["--audit"])
    out = capsys.readouterr().out
    assert "## The founding premise, recomputed" in out
    assert "retracted  150.4 (1.032 of ridge, ABOVE)" in out
    assert "corrected  135.4 (0.929 of ridge)" in out
    assert "phi 0.06905 (Act1/W, fused)" in out
    assert "alpha_b 0.87611" in out
    assert "at its OWN delta 0.16648 there is NO cap" in out
    assert "140.4" not in out and "141.8" not in out
    assert "NO corrected cap reaches its card's ridge" in out


def test_the_premise_refuses_a_corpus_it_cannot_recompute_from(bm, tmp_path):
    """"No BM=128 fit carries an alpha" and "the premise holds" must not print
    the same way, so an empty corpus RAISES and never returns an empty list.

    `PremiseNotRecomputable` and not `RefusedBeforeMeasuring`: the second
    carries `exit_codes.REFUSED` and would take the run with it. Which class
    this is IS the fix in `test_a_corpus_with_no_premise_alpha_is_still_scored`
    below, so it is asserted rather than caught broadly.
    """
    with pytest.raises(bm.PremiseNotRecomputable) as exc:
        bm.premise_caps(tmp_path)
    assert "cannot be recomputed" in str(exc.value)
    assert not isinstance(exc.value, bm.RefusedBeforeMeasuring)


def test_the_premise_names_the_ladders_it_could_not_use(bm, tmp_path):
    """A ladder dropped for an alpha the model cannot hold is NAMED, not
    silently skipped.

    A fitted alpha of 1.4 recovers an `alpha_b` above 1, which is not a miss
    fraction, and `ai_model.alpha_b_from_fitted` refuses it by name. The refusal
    a reader gets then has to say that a ladder was examined and rejected,
    because "no ladder carried an alpha" and "the one ladder that did read 1.4"
    are different states of the corpus.
    """
    doc = json.loads(A100_G64.read_text())
    doc["ladder"]["128"]["alpha"] = 1.4
    arm = tmp_path / "2026-09-02-nvidia_a100_sxm4_80gb-planted"
    arm.mkdir(parents=True)
    (arm / A100_G64.name).write_text(json.dumps(doc))
    with pytest.raises(bm.PremiseNotRecomputable) as exc:
        bm.premise_caps(tmp_path)
    assert "nvidia_a100_sxm4_80gb-planted" in str(exc.value)
    assert "alpha_fitted=1.4000" in str(exc.value)


def test_a_corpus_with_no_premise_alpha_is_still_scored(bm, tmp_path, capsys):
    """THE COUPLING, removed: a premise the gates do not read cannot void them.

    `premise_caps` needs a BM=128 ladder carrying BOTH an activation-corrected
    alpha and a stamped ridge, and only 2 of the 22 valid published ladders do
    -- the two this arm argues should be withdrawn. While that refusal was a
    `RefusedBeforeMeasuring`, blanking those two values turned a fully
    scoreable page into exit 2 with ZERO `RESULT:` lines: V0 and P4 through P9
    never examined. Here the corpus is mirrored with both alphas blanked, and
    the page must still be scored -- same gate count, same CLAIM_FAIL -- with
    the premise printed as NOT RECOMPUTABLE rather than swallowed.
    """
    if not PUBLISHED.exists():
        pytest.skip("no results/published on this checkout")
    blanked = 0
    for report in sorted(PUBLISHED.glob("*/*.report.json")):
        doc = json.loads(report.read_text())
        fit = (doc.get("ladder") or {}).get("128")
        if fit and fit.get("alpha_corrected") is not None:
            fit["alpha_corrected"] = None
            blanked += 1
        out = tmp_path / report.parent.name / report.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc))
    assert blanked == 2, blanked

    code = bm.main(["--audit", "--published", str(tmp_path)])
    text = capsys.readouterr().out
    assert code == exit_codes.CLAIM_FAIL
    assert "## The founding premise, NOT RECOMPUTABLE" in text
    assert "voids the PREMISE and not the page" in text
    scored = [ln for ln in text.splitlines() if ln.startswith("RESULT: ")]
    assert len(scored) == 7, scored
    assert exit_codes.classify_text(text) == code


def test_no_published_ladder_reaches_five_clean_memory_treads(bm):
    """The registered answer to the question the script was asked."""
    if not PUBLISHED.exists():
        pytest.skip("no results/published on this checkout")
    records, _ = bm.load_corpus(PUBLISHED)
    best = max(r.memory_points for r in records)
    assert best < bm.TARGET_TREADS or not any(
        bm.audit_record(r, draws=200).admissible
        for r in records if r.memory_points >= bm.TARGET_TREADS), (
        "a published ladder now has five ADMISSIBLE memory treads at "
        "BLOCK_M=128; the feasibility verdict must be revisited")


def _planted_samples(bm, alpha: float, rho: float, bandwidth_gbps: float,
                     reps: int = 5, noise: float = 0.002):
    """A full replicated run, generated from the STUDY'S OWN model, as Sample rows.

    Generated through `model_ms` rather than by hand, and that is not a
    convenience. `compute_reference` now level-checks its candidate against the
    roof, against one full weight read and against the smaller ladders, so a
    hand-rolled ladder with a plausible-looking slope gets REFUSED for being
    physically inconsistent with the card it claims to be from -- which is what
    happened to the first version of this fixture and is exactly the class of
    error the level check exists to catch. Returning the implied ceiling keeps
    the whole planted world self-consistent: `peak = rho x bandwidth`.
    """
    import random

    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    rng = random.Random(0)
    out = []
    for rep in range(1, reps + 1):
        for block_m, treads in ((256, 4), (128, 8)):
            for n in range(1, treads + 1):
                rows = n * block_m
                ms = bm.SWEEP.model_ms(cfg, rows, block_m, alpha=alpha,
                                       ridge=rho, bandwidth_gbps=bandwidth_gbps,
                                       b=2, overhead_ms=0.05)
                ms *= math.exp(rng.gauss(0.0, noise))
                out.append(bm.Sample(block_m, n, rows,
                                     bm.SWEEP.tokens_for_rows(cfg, rows), rep,
                                     ms, ms, 0.0, 10))
    ceiling_tflops = rho * bandwidth_gbps * 1e9 / 1e12
    return cfg, out, ceiling_tflops


def test_the_pod_analysis_runs_end_to_end_on_a_planted_escape_up_world(bm):
    """The code that runs on the pod, exercised before anyone pays for it.

    Nothing here is evidence about hardware -- the cells are generated from the
    law. It exists so a NameError in `analyse_run` is found on a laptop and not
    thirty seconds into a metered session, and so the gates are known to reach a
    verdict on data shaped like a real run.
    """
    # rho = 175 needs a card whose achieved ridge is 175 Op/B. NEITHER CARD IN
    # THIS STUDY HAS ONE -- the A100 calibrates at 145.8 and the H200 at 152.8 --
    # which is the finding, stated here as a fixture: to plant a world where the
    # depth claim is reachable, hardware has to be invented.
    cfg, samples, ceiling = _planted_samples(bm, alpha=0.95, rho=175.0,
                                             bandwidth_gbps=1799.4)
    lines, gates, payload = bm.analyse_run(
        samples, cfg, 2, ceiling_tflops=ceiling, ceiling_source="test",
        compiles={128: 3, 256: 4}, executed={128: 40, 256: 20},
        ridge=175.0, bandwidth_gbps=1799.4, draws=200)
    assert lines and payload["subject_points"]
    assert {g.passed for g in gates} != {None}, "every gate came back UNKNOWN"
    by = {g.tag: g for g in gates}
    assert by["V0"].passed is True
    assert by["V4"].passed is True
    assert by["V5"].passed is True, by["V5"].observed
    # Planted above the escape-up threshold, so the depth claim must be reached.
    assert by["C1"].passed is True, by["C1"].observed
    assert payload["memory_points"] >= bm.TARGET_TREADS


def test_the_pod_analysis_declines_on_a_planted_straddle_world(bm):
    """The world both cards are actually in: the gates must NOT find a depth."""
    # The A100's own calibration, and an alpha inside the range the study
    # measures. This is not a hypothetical card.
    cfg, samples, ceiling = _planted_samples(bm, alpha=0.88, rho=145.813,
                                             bandwidth_gbps=1799.4)
    _, gates, payload = bm.analyse_run(
        samples, cfg, 2, ceiling_tflops=ceiling, ceiling_source="test",
        compiles={128: 3, 256: 4}, executed={128: 40, 256: 20},
        ridge=145.813, bandwidth_gbps=1799.4, draws=200)
    by = {g.tag: g for g in gates}
    # UNKNOWN, NOT FAIL, and the difference is the finding. B/C comes out inside
    # PARALLEL_BRANCH_TOLERANCE, so `fit_ladder` returns
    # `undecided_parallel_branch`: it LOOKED and declined to name a branch, and
    # `B/C = ridge/ai_cap` says the cap sits ON the ridge -- the roofline arm's
    # question, not this one's. Scoring it FAIL publishes "no depth" from a fit
    # that named none.
    assert by["C1"].passed is None
    assert payload["ladder_outcome"] == "undecided_parallel_branch"
    assert by["C2"].passed is not True
    assert payload["memory_points"] < bm.TARGET_TREADS


def test_git_visibility_answers_for_a_results_path(bm):
    """`results/*` is ignored and only `results/published/` is excepted."""
    ignored = bm.git_visibility(ROOT / "results" / "bm128_depth" / "x")
    kept = bm.git_visibility(ROOT / "results" / "published" / "x")
    assert "IGNORED" in ignored or "unverified" in ignored
    assert "IGNORED" not in kept


# --------------------------------------------------------------------------
# R7. The instrument, the provenance block, the exit table, the run id, the MDE.
# --------------------------------------------------------------------------

def test_the_retired_timer_is_gone_from_the_measured_path(bm):
    """`SWEEP.time_call` created its events inside the loop, synchronised after
    every call, never flushed L2 and never read a clock -- while the reference
    every tread is classified against was measured queue-deep. It is retired
    over there and raises; this file must not name it any more."""
    source = (ROOT / "scripts" / "bm128_depth.py").read_text()
    assert "SWEEP.time_call" not in source
    assert "timing.time_kernel" in source


def test_the_instrument_columns_survive_the_csv_round_trip(bm, tmp_path):
    """A column that cannot be read back is a column that does not exist.

    The three clock fields are Optional and None means NOT DETERMINED, so the
    round trip has to preserve None as None: read back as 0.0 or False, a row
    whose clock was never sampled becomes a row that ran at 0 MHz or a row that
    was positively excluded.
    """
    path = tmp_path / "cells.csv"
    rows = [
        bm.Sample(128, 1, 128, 512, 1, 1.5, 1.4, 0.01, 240,
                  instrument="queue-deep/l2-flush/clock-under-load/v2",
                  warmup_ms=301.2, trials=3, sm_clock_load_mhz=1965.0,
                  clock_level_ok=True, clock_drift_ok=True, l2_flush=True),
        bm.Sample(128, 2, 256, 1024, 1, 2.9, 2.8, 0.02, 120,
                  instrument="queue-deep/l2-flush/clock-under-load/v2",
                  warmup_ms=300.0, trials=3, sm_clock_load_mhz=None,
                  clock_level_ok=None, clock_drift_ok=None, l2_flush=False),
    ]
    for row in rows:
        bm.append_sample(path, row)
    _, back = bm.read_samples(path)
    assert [b.instrument for b in back] == [r.instrument for r in rows]
    assert back[0].sm_clock_load_mhz == 1965.0
    assert back[0].clock_level_ok is True and back[0].l2_flush is True
    assert back[1].sm_clock_load_mhz is None, "None must not read back as 0.0"
    assert back[1].clock_level_ok is None, "None must not read back as False"
    assert back[1].l2_flush is False
    assert [b.trials for b in back] == [3, 3]


def test_a_row_written_before_the_instrument_had_a_name_still_reads(bm, tmp_path):
    """cells.csv files from before 2026-09-02 have none of these columns, and
    the honest reading of that is "not recorded", not a crash."""
    path = tmp_path / "old.csv"
    path.write_text("block_m,tiles,rows_per_expert,tokens,rep,ms_p50,ms_min,"
                    "ms_stdev,iters,status,detail\n"
                    "128,1,128,512,1,1.5,1.4,0.01,50,ok,\n")
    _, back = bm.read_samples(path)
    assert len(back) == 1
    assert back[0].instrument == "" and back[0].trials == 0
    assert back[0].clock_level_ok is None


def test_every_gate_prints_exactly_one_result_line(bm, capsys):
    bm.main(["--self-test"])
    out = capsys.readouterr().out
    parsed = exit_codes.parse_result_lines(out)
    assert len(parsed) == len(bm.SELF_TEST_WORLDS)
    assert [r.name for r in parsed] == [f"S-{w.name}"
                                        for w in bm.SELF_TEST_WORLDS]
    raw = [ln for ln in out.splitlines() if ln.startswith("RESULT: ")]
    assert len(raw) == len(parsed), "a RESULT line the parser cannot read back"


def test_a_gate_tag_is_one_token_and_cannot_collide(bm):
    """`result_line` requires one run of non-whitespace, and three gates called
    `S` would collide on a driver's grep -- which is a gate that silently
    disappears from the summary."""
    assert bm.Gate(bm.VALIDITY, "V1 reference level", "p", "r", True, "o").tag == "V1"
    assert bm.Gate(bm.CLAIM, "P9 escape-down is shut", "p", "r", True, "o").tag == "P9"
    tags = [bm.Gate(bm.VALIDITY, f"S {w.name}", "p", "r", True, "o").tag
            for w in bm.SELF_TEST_WORLDS]
    assert len(set(tags)) == len(tags)
    assert all(" " not in t for t in tags)


@pytest.mark.parametrize("argv,code", [
    (["--self-test"], exit_codes.DONE),
    (["--self-test", "--plant-noise", "0"], exit_codes.REFUSED),
    # A DRY RUN MEASURED NOTHING, SO IT IS REFUSED AND NOT DONE. DONE in the
    # shared table reads "measured; every VALIDITY and CLAIM gate PASSED", and a
    # plan scores no gate at all: it prints no RESULT line, and `classify_text`
    # over a log with none raises `NoGatesScored`, which is the REFUSED shape.
    (["--dry-run"], exit_codes.REFUSED),
    # THE TWO PLANS WHOSE MDE IS NOT STATEABLE. Both returned 1 -- CLAIM_FAIL,
    # "a pre-registered claim was refuted" -- from a `--dry-run` that measured
    # nothing, because `mde_lines` refused by `raise SystemExit(<str>)` and the
    # interpreter, not `exit_codes.classify`, chose the number. A missing power
    # calculation still gets no vote: both land on the same REFUSED a stateable
    # plan does.
    (["--dry-run", "--plant-noise", "0"], exit_codes.REFUSED),
    (["--dry-run", "--reps", "1"], exit_codes.REFUSED),
    # --audit SCORES the published corpus and C1 FAILS on it, so it is a
    # RESULT. It returned DONE until 2026-09-02 because `--fail-on-gate` folded
    # CLAIM_FAIL into 0: the log said `RESULT: CLAIM C1 FAIL` and the process
    # said "every gate PASSED". The flag is retired and passing it changes
    # nothing.
    (["--audit"], exit_codes.CLAIM_FAIL),
    (["--audit", "--fail-on-gate"], exit_codes.CLAIM_FAIL),
])
def test_the_exit_codes_are_the_shared_tables(bm, argv, code, capsys):
    assert bm.main(argv) == code
    capsys.readouterr()


def test_the_log_recomputes_the_code_the_process_returned(bm, capsys):
    """`classify_text` over the RESULT lines against the returned integer. A
    script printing one thing and exiting another is the defect
    `moe.bench.exit_codes` is named against."""
    rc = bm.main(["--self-test"])
    assert exit_codes.classify_text(capsys.readouterr().out) == rc


def test_a_broken_registration_makes_the_self_test_invalid_and_not_a_claim(
        bm, monkeypatch, capsys):
    """The S gates are VALIDITY: a self test that came out other than
    registered has not produced a result about anything."""
    world = bm.SELF_TEST_WORLDS[0]
    broken = bm.PlantedWorld(world.name, world.alpha, world.rho, world.why,
                             {"C1": False}, exit_code=world.exit_code,
                             low_clock_treads=world.low_clock_treads)
    monkeypatch.setattr(bm, "SELF_TEST_WORLDS", (broken,))
    assert bm.main(["--self-test"]) == exit_codes.INVALID
    capsys.readouterr()


def test_the_run_id_is_the_repositorys_rule_and_carries_the_timing_knobs(bm):
    """One run-id scheme in the repository, and `--trials` / `--no-l2-flush` in
    it: both change the milliseconds of every row, and `read_samples` resumes on
    `(block_m, tiles, rep)`, so a re-run at another trial count would skip every
    tread and print the first run's timings under the second's heading."""
    parser = bm.build_parser()
    base = bm.default_run_id(parser.parse_args([]), "NVIDIA H200")
    assert base.startswith(PV.card_slug("NVIDIA H200") + "-")
    for knob in (["--trials", "9"], ["--no-l2-flush"], ["--warmup", "42"],
                 ["--reps", "3"], ["--block-k", "128"]):
        assert bm.default_run_id(parser.parse_args(knob),
                                 "NVIDIA H200") != base, knob


def test_the_run_id_refuses_a_knob_that_was_never_resolved():
    with pytest.raises(PV.UnresolvedKnob):
        PV.run_id(card="NVIDIA H200", model="qwen2-57b-a14b", trials=None)


def test_the_warmup_is_milliseconds_and_the_plan_prices_it_that_way(bm):
    """`--warmup 20` used to be a call count. Under `time_kernel` it is a
    DURATION of delivered load, and a cost model that multiplies modelled
    milliseconds by it prices an 8-tread ladder at seconds when it costs
    minutes."""
    from moe.spec import MODEL_CONFIGS
    parser = bm.build_parser()
    assert parser.parse_args([]).warmup == 300.0
    assert parser.parse_args(["--warmup-ms", "50"]).warmup == 50.0
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    plan = bm.build_plan(parser.parse_args([]), cfg)
    per_cell = (plan.warmup + plan.trials * plan.cell_budget_ms) / 1e3
    assert plan.estimated_seconds == pytest.approx(plan.cells * per_cell)
    assert "time_kernel" in " ".join(plan.lines(cfg))


def test_the_plan_states_an_mde_from_one_stated_assumption(bm, capsys):
    bm.main(["--dry-run"])
    out = capsys.readouterr().out
    assert "Minimum detectable effect" in out
    assert f"{bm.PUBLISHED_LADDER_SPREAD:.2%}" in out
    for tag in ("V2 inversions", "C2 margin", "C1 depth"):
        assert tag in out


def test_the_mde_refuses_a_zero_spread_rather_than_reporting_omniscience(bm):
    with pytest.raises(bm.MdeNotStateable) as exc:
        bm.mde_lines(spread=0.0, reps=7, treads=8)
    assert "every effect is detectable" in str(exc.value)


def test_the_mde_refusal_is_not_a_systemexit_and_so_cannot_become_exit_1(bm):
    """A `SystemExit(<str>)` exits the process with 1, which this project's
    table spells CLAIM_FAIL: "a pre-registered claim was refuted". Nothing had
    been refuted; nothing had even been measured. `MdeNotStateable` is a
    `RuntimeError` so an escape is an unhandled exception the top-level handler
    maps to ERROR, and a caller can name it instead of catching `SystemExit`."""
    assert issubclass(bm.MdeNotStateable, RuntimeError)
    assert not issubclass(bm.MdeNotStateable, SystemExit)


@pytest.mark.parametrize("kw,fragment", [
    ({"spread": 0.0, "reps": 7, "treads": 8}, "every effect is detectable"),
    ({"spread": 0.0182, "reps": 1, "treads": 8}, "no MDE is stateable"),
    ({"spread": 0.0182, "reps": 7, "treads": 1}, "no MDE is stateable"),
])
def test_a_plan_with_no_stateable_mde_says_so_and_carries_on(bm, kw, fragment):
    """THE POD PATH'S DEGRADATION, EXERCISED OFF GPU. `main` builds
    `payload["mde"]` AFTER both ladders are timed and BEFORE `report.json` is
    written, so a refusal that propagated cost twelve metered minutes and left
    no report on disk. `mde_block` is the one place both call sites degrade, and
    it returns the reason as a field rather than only as prose."""
    lines, reason = bm.mde_block(**kw)
    assert reason and fragment in reason
    assert any("not stateable" in ln for ln in lines)
    assert "Minimum detectable effect" not in " ".join(lines)


def test_a_stateable_mde_reports_no_reason(bm):
    lines, reason = bm.mde_block(spread=bm.PUBLISHED_LADDER_SPREAD, reps=7,
                                 treads=8)
    assert reason == ""
    assert any("Minimum detectable effect" in ln for ln in lines)


@pytest.mark.parametrize("argv", [["--plant-noise", "0"], ["--reps", "1"]])
def test_a_plan_whose_mde_is_not_stateable_still_prints_the_whole_plan(
        bm, argv, capsys):
    """The MDE is a section of a document, not a gate, and it gets no vote on
    the arm's verdict. The plan above it is what the pod is being asked to buy
    and it has to still be there."""
    assert bm.main(["--dry-run", *argv]) == exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "MINIMUM DETECTABLE EFFECT: not stateable" in out
    assert "## The plan" in out
    assert "WRITES TO" in out
    # And nothing that looks like a gate result was printed by a run that
    # scored no gates.
    assert exit_codes.parse_result_lines(out) == []


def test_the_mde_moves_the_right_way_with_repeats_and_with_the_spread(bm):
    """Both directions, because a power calculation that moves the wrong way is
    invisible in a single printed number."""
    def inversion(spread, reps):
        line = next(ln for ln in bm.mde_lines(spread=spread, reps=reps, treads=8)
                    if "V2 inversions" in ln)
        return float(line.split("inversion of ")[1].split("%")[0])

    assert inversion(0.0182, 21) < inversion(0.0182, 7)
    assert inversion(0.005, 7) < inversion(0.0182, 7)


def test_the_published_spread_default_is_inside_the_published_range(bm):
    assert 0.0076 <= bm.PUBLISHED_LADDER_SPREAD <= 0.0182, (
        "the default must be an assumption traceable to measured ladders, not "
        "a number someone liked")
    assert bm.build_parser().parse_args([]).plant_noise == bm.PUBLISHED_LADDER_SPREAD


# --------------------------------------------------------------------------
# The acceptance check for the whole exit-code repair: the log and the process
# say the same thing in every mode this file can reach without a GPU.
# --------------------------------------------------------------------------

#: Every off-GPU mode of this script and the code it must return. A plan and a
#: refusal are REFUSED because they score no gate; `--self-test` and `--audit`
#: score gates and take whatever `classify` makes of them.
OFF_GPU_MODES = [
    (["--dry-run"], exit_codes.REFUSED),
    (["--dry-run", "--plant-noise", "0"], exit_codes.REFUSED),
    (["--dry-run", "--reps", "1"], exit_codes.REFUSED),
    # A REFUSAL FROM DEEP IN THE PLANNER. `ladder_rows` cannot form a single
    # full tile stack here, so it raises `RefusedBeforeMeasuring` out of
    # `build_plan`. It was `raise SystemExit(<str>)`, which exits 1.
    (["--dry-run", "--r-max", "8"], exit_codes.REFUSED),
    (["--self-test"], exit_codes.DONE),
    (["--self-test", "--plant-noise", "0"], exit_codes.REFUSED),
    (["--audit"], exit_codes.CLAIM_FAIL),
]


@pytest.mark.parametrize("argv,code", OFF_GPU_MODES,
                         ids=[" ".join(a) or "bare" for a, _ in OFF_GPU_MODES])
def test_the_log_and_the_exit_code_agree_in_every_off_gpu_mode(
        bm, argv, code, tmp_path, capsys):
    """The whole repair, stated as one property instead of as prose.

    For every mode this file can reach on a laptop, the RESULT lines it printed
    and the integer it returned have to be the same verdict. `classify_text`
    recomputes the code from the log; a log with NO RESULT lines raises
    `NoGatesScored`, and `moe.bench.exit_codes` documents that as exactly what a
    REFUSED log looks like from there, so the two cases are one rule: score
    gates and match `classify`, or score none and return REFUSED.

    Three of these rows used to break it. `--dry-run` returned DONE, which reads
    "measured; every VALIDITY and CLAIM gate PASSED", from a run that measured
    nothing. `--dry-run --r-max 8` returned 1 = CLAIM_FAIL, a measured
    refutation, because the refusal was a `SystemExit` carrying a string.
    `--audit` returned DONE while its log carried `RESULT: CLAIM C1 FAIL`,
    because `--fail-on-gate` folded a claim failure into 0.
    """
    rc = bm.main([*argv, "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == code
    lines = exit_codes.parse_result_lines(out)
    if lines:
        assert exit_codes.classify_text(out) == rc
    else:
        assert rc == exit_codes.REFUSED, (
            "a run that scored no gate printed no RESULT line, so its log "
            "implies REFUSED and nothing else")


# --------------------------------------------------------------------------
# The apparatus breaking, which is not a claim failing.
#
# Two defects found by a reviewer who did not own this file, both silent, both
# exiting the interpreter's ONE. `moe.bench.exit_codes` calls ONE CLAIM_FAIL: a
# RESULT, in FINISHED_CODES, recorded by the session driver and never retried.
# This arm pays for two full ladders before it writes anything, so what ONE
# costs here is the whole booking and the report that would have said why.
# --------------------------------------------------------------------------

class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def synchronize() -> None:
        pass


class _CudaLessTorch(types.ModuleType):
    """Real torch with `cuda` answered and the `device=` keyword dropped.

    A `ModuleType` subclass because `measure_setting` does `import torch`
    inside the function, so `sys.modules` is the only seam a test has: a module
    attribute cannot be patched onto a name the function re-imports every call.
    A proxy and not a stub, so every other call the ladder makes on the way to
    the instrument is the real one.
    """

    cuda = _Cuda

    def __getattr__(self, name):
        return getattr(real_torch, name)

    def full(self, *args, **kwargs):
        kwargs.pop("device", None)
        return real_torch.full(*args, **kwargs)


@pytest.fixture
def pod(bm, monkeypatch, tmp_path):
    """Everything `measure_setting` reaches for between its imports and the CSV.

    Faked rather than described: a handler can only be shown to be at the call
    site by executing the call site, and this ladder's call site is behind an
    `import torch`, a vLLM entry point and a Triton cache.
    """
    state = types.SimpleNamespace(seen=[], out=tmp_path, timing_result=None,
                                  samples=[])

    monkeypatch.setitem(sys.modules, "torch", _CudaLessTorch("torch"))
    fused = types.ModuleType("vllm.model_executor.layers.fused_moe")
    fused.fused_experts = lambda **kw: real_torch.zeros(2, 2)
    activation = types.ModuleType("vllm.model_executor.layers.fused_moe.activation")
    activation.MoEActivation = lambda value: value
    for name, module in (
            ("vllm", types.ModuleType("vllm")),
            ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
            ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
            ("vllm.model_executor.layers.fused_moe", fused),
            ("vllm.model_executor.layers.fused_moe.activation", activation)):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(bm.SWEEP, "find_override",
                        lambda: (lambda conf: contextlib.nullcontext(), "fake"))
    monkeypatch.setattr(bm.SWEEP, "arm_triton_cache", lambda *a, **k: None)
    monkeypatch.setattr(bm.SWEEP, "count_new", lambda *a, **k: 0)
    monkeypatch.setattr(bm.SWEEP, "tokens_for_rows", lambda cfg, rows: rows)
    monkeypatch.setattr(bm.SWEEP, "balanced_ids",
                        lambda cfg, tokens, device:
                        real_torch.zeros((tokens, cfg.top_k), dtype=real_torch.long))
    monkeypatch.setattr(TORCH_REF, "make_inputs",
                        lambda spec, device=None: (real_torch.zeros(2, 2),
                                                   types.SimpleNamespace(w1=None, w2=None)))

    def timer(fn, **kwargs):
        state.seen.append(kwargs)
        result = state.timing_result
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(timing, "time_kernel", timer)
    return state


def _timing_at(load_mhz: float, reference_mhz: float | None) -> timing.KernelTiming:
    """A `KernelTiming` whose clock verdicts come from the REAL `clock_flags`.

    Not hand-set booleans: the question is whether the ladder hands the
    instrument a reference and puts it on the row, and a hand-set flag would
    answer it whatever the ladder passed.
    """
    level, drift = timing.clock_flags(load_mhz, load_mhz, load_mhz, reference_mhz)
    return timing.KernelTiming(
        ms_p50=1.0, ms_p90=1.1, ms_min=0.9, ms_std=0.01, iters=100, trials=3,
        warmup_ms=300.0, l2_flush=True, sm_clock_load_mhz=load_mhz,
        sm_clock_start_mhz=load_mhz, sm_clock_end_mhz=load_mhz,
        clock_level_ok=level, clock_drift_ok=drift, samples=300,
        warmup_calls=10, flush_mb=64, clock_samples=9, clock_source="injected",
        clock_poll_ms=1.0, host_bound=False, host_enqueue_ms=0.01,
        clock_note="scripted clock")


def _measure(bm, pod):
    """One block size over two treads, which is the smallest thing that can
    show a refusal repeating itself per cell."""
    args = types.SimpleNamespace(reps=1, dtype="bf16", seed=0, warmup=300.0,
                                 cell_budget_ms=200.0, trials=3,
                                 no_l2_flush=False)
    return bm.measure_setting(args, MODEL_CONFIGS["mixtral-8x7b"], 16,
                              [16, 32], pod.out / "cells.csv", pod.out, {},
                              set(), pod.samples, 1515.0)


def test_an_unplanned_exception_is_error_and_not_a_claim_that_failed(
        bm, monkeypatch, capsys):
    """`sys.exit(main())` had nothing above it.

    An OOM, a truncated write, an import that drifted: each exited ONE, which
    the driver LATCHES as this experiment's registered answer to "can BLOCK_M
    =128 show clean memory-bound treads". The file's header already records the
    other way this arm lost a booking in the same gap -- a `SystemExit` between
    the last timing and the first `write_text`. ERROR (4) is outside
    FINISHED_CODES precisely so the apparatus breaking can be told from the
    claim not holding, and it is the only retryable code in the table.
    """
    def boom(argv=None):
        raise RuntimeError("torch OOM on the pod")

    monkeypatch.setattr(bm, "_main", boom)
    assert bm.main([]) == exit_codes.ERROR
    err = capsys.readouterr().err
    assert "torch OOM on the pod" in err, "the traceback was swallowed"
    assert "RuntimeError" in err
    assert exit_codes.ledger_state(exit_codes.ERROR) == "RETRY"
    assert exit_codes.ledger_state(exit_codes.CLAIM_FAIL) != "RETRY"

    # ...and the PASS branch: a run that reached a verdict keeps its own code.
    monkeypatch.setattr(bm, "_main", lambda argv=None: exit_codes.CLAIM_FAIL)
    assert bm.main([]) == exit_codes.CLAIM_FAIL
    monkeypatch.setattr(bm, "_main", lambda argv=None: exit_codes.DONE)
    assert bm.main([]) == exit_codes.DONE


def test_a_refusal_sentence_is_refused_and_not_a_claim_that_failed(
        bm, monkeypatch, capsys):
    """The other half. `raise SystemExit(<str>)` sets `code` to the STRING and
    exits ONE; this file replaced fourteen of those with
    `RefusedBeforeMeasuring`, and the branch stays so that a fifteenth added
    later, or one out of a library it imports, cannot land as a refuted claim.
    """
    def refuse(argv=None):
        raise SystemExit("the calibration for this card records no clock")

    monkeypatch.setattr(bm, "_main", refuse)
    assert bm.main([]) == exit_codes.REFUSED
    assert "records no clock" in capsys.readouterr().err


def test_an_integer_systemexit_still_means_what_it_says(bm):
    """The FAIL branch of the string test. `RefusedBeforeMeasuring` carries
    `code = exit_codes.REFUSED` and argparse exits `SystemExit(2)`; a handler
    that reclassified either would report a refusal twice or hide a usage
    error behind one."""
    with pytest.raises(SystemExit) as caught:
        bm.main(["--model", "not-a-model"])
    assert caught.value.code == 2


def test_an_instrument_refusal_leaves_the_ladder_instead_of_being_recorded(
        bm, pod):
    """THE SECOND DOOR INTO THE SAME ROOM, executed at the call site.

    `RefusedBeforeMeasuring` is a `SystemExit` precisely because
    `measure_setting` times inside a per-cell `except Exception` -- the class
    docstring says so -- and `timing.TimingRefused` subclasses RuntimeError, so
    every refusal the INSTRUMENT raises walked through the door left open
    beside it. Each is the same fact for every tread, so the ladder wrote a
    `status="failed"` sample per tread, ground through both block sizes, and
    scored its gates over a page of zeroes.
    """
    pod.timing_result = timing.TimingRefused(
        "trials=0: a measurement needs at least one trial")
    with pytest.raises(timing.TimingRefused, match="at least one trial"):
        _measure(bm, pod)
    assert pod.samples == [], "a sample was kept for a tread never measured"
    assert not (pod.out / "cells.csv").exists()


def test_a_kernels_own_runtime_error_is_still_one_treads_error(bm, pod):
    """The PASS branch of the same door, and why it is a subclass check rather
    than a blanket re-raise: a kernel that launched badly IS one tread's fact,
    the sample records it with `status="failed"`, and the ladder carries on so
    the treads that do run still form a fit."""
    pod.timing_result = RuntimeError("CUDA error: an illegal memory access")
    _measure(bm, pod)
    assert [s.status for s in pod.samples] == ["failed", "failed"]
    assert all("illegal memory access" in s.detail for s in pod.samples)
    assert (pod.out / "cells.csv").exists()


# --------------------------------------------------------------------------
# The other half of last round's clock fix, which this file did not get.
#
# `tile_sweep` and `group_m_alpha_sweep` both learned two things on 2026-09-02:
# hand `time_kernel` the clock the roof was measured at, AND put that number on
# the row it scored, because `clock_level_ok` is tri-state and its None
# conflates "the poller landed nothing" with "there was no reference at all".
# This file got the first and not the second, which is the recurring defect of
# this rebuild wearing its ninth hat: a fix applied at one of two call sites.
# --------------------------------------------------------------------------

def test_the_row_carries_the_clock_its_level_verdict_was_scored_against(bm, pod):
    """FAIL branch first: with no reference the LEVEL column is empty, and
    before this change nothing else on the row said why.

    `clock_flags` returns None for LEVEL when it is handed no reference, and
    None is also what a container without NVML produces on a run that HAD one.
    Both are an empty cell, and the report then prints `no tread was excluded
    for a drifting clock` over a ladder that carries no clock evidence at all.
    Since 2026-09-09 the EXCLUSION does not turn on the reference -- DRIFT
    compares a row with itself -- so what a missing reference costs is the
    recorded side and every rescaled roof fraction. The number LEVEL was scored
    AGAINST is the only thing that separates the two states, and it has to be on
    the row rather than in the session, because `read_samples` resumes a
    `cells.csv` a later pod wrote nothing else into.
    """
    pod.timing_result = _timing_at(1515.0, None)
    args = types.SimpleNamespace(reps=1, dtype="bf16", seed=0, warmup=300.0,
                                 cell_budget_ms=200.0, trials=3,
                                 no_l2_flush=False)
    bm.measure_setting(args, MODEL_CONFIGS["mixtral-8x7b"], 16, [16, 32],
                       pod.out / "none.csv", pod.out, {}, set(), pod.samples,
                       None)
    assert [s.clock_level_ok for s in pod.samples] == [None, None]
    assert [s.reference_clock_mhz for s in pod.samples] == [None, None]

    # ...and the PASS branch, at the same call site: the card sat at 1515 MHz
    # against a roof measured at 1515, so LEVEL is a real True and the row says
    # what it was true against.
    pod.samples.clear()
    pod.timing_result = _timing_at(1515.0, 1515.0)
    bm.measure_setting(args, MODEL_CONFIGS["mixtral-8x7b"], 16, [16, 32],
                       pod.out / "ref.csv", pod.out, {}, set(), pod.samples,
                       1515.0)
    assert [s.clock_level_ok for s in pod.samples] == [True, True]
    assert [s.reference_clock_mhz for s in pod.samples] == [1515.0, 1515.0]

    # The whole point is that the two are told apart AFTER a round trip, since
    # a resume reads the file and not this session.
    _, none_rows = bm.read_samples(pod.out / "none.csv")
    _, ref_rows = bm.read_samples(pod.out / "ref.csv")
    assert [s.reference_clock_mhz for s in none_rows] == [None, None]
    assert [s.reference_clock_mhz for s in ref_rows] == [1515.0, 1515.0]
    assert not any(s.clock_excluded for s in none_rows + ref_rows), (
        "neither world excludes a tread, which is exactly why the column that "
        "separates them has to be somewhere else")


def test_a_row_written_before_the_reference_column_existed_still_reads(bm, tmp_path):
    """A `cells.csv` from any run before 2026-09-03 has no such column, and the
    honest reading of absent is None: those runs passed the instrument no
    reference, so their LEVEL column was empty for that reason. A crash here
    would refuse to resume every ladder this study has ever measured."""
    path = tmp_path / "old.csv"
    path.write_text(
        "block_m,tiles,rows_per_expert,tokens,rep,ms_p50,ms_min,ms_stdev,"
        "iters,status,detail,instrument,warmup_ms,trials,sm_clock_load_mhz,"
        "clock_level_ok,clock_drift_ok,l2_flush\n"
        "128,4,512,2048,1,1.5,1.4,0.01,100,ok,,,300.0,3,,,,True\n")
    done, rows = bm.read_samples(path)
    assert done == {(128, 4, 1)}
    assert rows[0].reference_clock_mhz is None
    assert rows[0].clock_level_ok is None


def test_a_ladder_with_no_reference_says_its_exclusion_count_examined_nothing(bm):
    """THE FAILURE SHAPE `moe/bench/exit_codes.py` IS NAMED AGAINST.

    A ladder measured with NO reference clock records no LEVEL side on any row,
    which reads exactly like a ladder measured on a card that stayed inside the
    band. A check that examined nothing reported zero failures, and the report
    had no other field a reader could ask. Since 2026-09-09 the SIDE is a
    record and DRIFT is the exclusion, so what a missing reference voids is the
    record and every rescaled roof fraction; `scored_treads_without_reference`
    is still the field that says so.
    """
    world = _world(bm, "escape-up")
    quiet_lines, _, quiet = _run_world(bm, world)
    assert quiet["excluded_drifted"] == 0
    assert quiet["scored_treads_without_reference"] == 0
    assert quiet["reference_clock_mhz"] == bm.SELF_TEST_REFERENCE_CLOCK_MHZ
    assert any("scored against 1980 MHz" in ln for ln in quiet_lines)

    # The same ladder, timed by an apparatus that could not report a clock:
    # every row's reference stripped, nothing else touched.
    from moe.spec import MODEL_CONFIGS as CFGS
    cfg = CFGS["qwen2-57b-a14b"]
    planted = bm.planted_samples(cfg, alpha=world.alpha, rho=world.rho,
                                 bandwidth_gbps=bm.SELF_TEST_BANDWIDTH,
                                 noise=bm.PUBLISHED_LADDER_SPREAD)
    blind = [dataclasses.replace(s, reference_clock_mhz=None,
                                 clock_level_ok=None) for s in planted]
    lines, _, payload = bm.analyse_run(
        blind, cfg, 2,
        ceiling_tflops=world.rho * bm.SELF_TEST_BANDWIDTH * 1e9 / 1e12,
        ceiling_source="test", compiles={128: 1, 256: 1},
        executed={128: 40, 256: 20}, ridge=world.rho,
        bandwidth_gbps=bm.SELF_TEST_BANDWIDTH, draws=200)
    assert payload["excluded_drifted"] == 0, (
        "the count is 0 in both worlds, which is the whole problem"
    )
    assert payload["reference_clock_mhz"] is None
    assert payload["scored_treads_without_reference"] == len(
        payload["scored_points"])
    assert any("NOT EXAMINED" in ln for ln in lines)
    assert not any("NOT EXAMINED" in ln for ln in quiet_lines)


def test_a_tread_has_a_reference_as_soon_as_one_repeat_recorded_one(bm):
    """ANY, not majority, unlike `tread_clock`. The reference is a property of
    the RUN, so a tread whose repeats disagree is a resume across the fix, and
    the answer that matters there is that a reference exists at all."""
    def sample(rep, mhz):
        return bm.Sample(128, 4, 512, 2048, rep, 1.0, 1.0, 0.0, 10,
                         reference_clock_mhz=mhz)
    mixed = [sample(1, None), sample(2, None), sample(3, 1515.0)]
    assert bm.tread_reference(mixed, 128)[4] == 1515.0
    assert bm.tread_reference([sample(1, None)], 128)[4] is None
    failed = bm.Sample(128, 4, 512, 2048, 1, 0.0, 0.0, 0.0, 0, "failed",
                       reference_clock_mhz=1515.0)
    assert bm.tread_reference([failed], 128) == {}, (
        "a tread that was never timed is not a tread with a reference")


def test_a_card_whose_calibration_records_no_clock_is_refused_before_the_ladders(
        bm, monkeypatch, tmp_path, capsys):
    """REFUSE rather than pass None, the way `driver.refuse_unreferenced_clock`
    and `tile_sweep` do. A card is attached by definition at this point --
    `missing_gpu_stack` and `load_measured` are both above it -- so this is a
    pod holding a card whose ruler was never measured, and the refusal is FREE:
    not one tread has been timed. It printed the absence and measured both
    ladders anyway until 2026-09-03, which cost the whole booking to learn that
    the arm carried no clock evidence.
    """
    import moe.bench.roofline as RF
    hw = RF.load_measured("NVIDIA H200")
    assert hw is not None, "the H200 calibration is not on this checkout"
    monkeypatch.setattr(bm.SWEEP, "missing_gpu_stack", lambda: "")
    monkeypatch.setattr(RF, "load_measured", lambda *a, **k: hw)
    monkeypatch.setattr(bm.SWEEP, "reference_clock_mhz", lambda name: (
        None, f"{name}: the calibration carries no clock at all"))

    def never(*a, **k):
        raise AssertionError("a tread was timed after the refusal")

    monkeypatch.setattr(bm, "measure_setting", never)
    rc = bm.main(["--out", str(tmp_path), "--card", "NVIDIA H200"])
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert "REFUSED" in out and "calibrate_hardware.py --publish" in out
    assert list(tmp_path.rglob("cells.csv")) == [], (
        "a ladder was written for a run that measured nothing")
    assert exit_codes.parse_result_lines(out) == [], (
        "a REFUSED log carrying RESULT lines lets the driver recompute DONE "
        "from them")

    # ...and the PASS branch: with a reference resolved the same run gets past
    # this gate and reaches the ladders, which is where the sentinel fires. It
    # comes back ERROR (4) and not REFUSED (2), so the two branches are
    # distinguishable in the one integer a session driver can see.
    monkeypatch.setattr(bm.SWEEP, "reference_clock_mhz",
                        lambda name: (1515.0, f"{name}: gemm_clock_mhz"))
    assert bm.main(["--out", str(tmp_path),
                    "--card", "NVIDIA H200"]) == exit_codes.ERROR
    passed = capsys.readouterr()
    assert "reference clock: 1515 MHz" in passed.out
    assert "timed after the refusal" in passed.err


# --------------------------------------------------------------------------
# 2026-09-08: the HIGH side of LEVEL. `timing.clock_flags` went two-sided on
# 2026-09-03 and this file's `Sample.clock_excluded` kept reading a failed
# verdict as "ran cold", so on an H200, where every memory-bound tread boosts
# to 1980 MHz against the 1515 MHz GEMM reference, the five clean treads this
# arm exists to measure were all dropped. Each test below plants BOTH sides.
# --------------------------------------------------------------------------

def _clocked(bm, rep, load, level, side, **over):
    kw = dict(clock_level_ok=level, clock_drift_ok=True,
              sm_clock_load_mhz=load, reference_clock_mhz=1515.0,
              clock_level_side=side)
    kw.update(over)
    return bm.Sample(128, 4, 512, 2048, rep, 1.0, 1.0, 0.0, 10, **kw)


def test_neither_level_side_is_an_exclusion_and_a_drifting_clock_is(bm):
    """1980 against 1515 with the side "high" is the H200's normal state for a
    memory-bound tread; 1400 with the side "low" is a dense tile's own state
    under a power cap. Since 2026-09-09 neither excludes and both are recorded;
    a clock that MOVED across the tread's own trials does."""
    import dataclasses as _dc
    high = _clocked(bm, 1, 1980.0, False, "high")
    low = _clocked(bm, 2, 1400.0, False, "low")
    moved = _dc.replace(_clocked(bm, 5, 1400.0, True, ""), clock_drift_ok=False)
    assert high.clock_excluded is False and high.clock_boosted is True
    assert low.clock_excluded is False and low.clock_sagged is True
    assert low.clock_boosted is False and high.clock_sagged is False
    assert moved.clock_excluded is True and moved.clock_sagged is False
    # A failed verdict with no side is a caller that dropped the side (the
    # sibling shape of the defect): read as LOW it drops every boosted tread,
    # read as HIGH it admits a sagged one, so it is refused rather than read.
    with pytest.raises(ValueError, match="no clock_level_side"):
        _clocked(bm, 3, 1400.0, False, "")
    with pytest.raises(ValueError, match="no clock_level_side"):
        _clocked(bm, 3, 1980.0, False, "")
    # None is still not an exclusion in either direction.
    unknown = _clocked(bm, 4, None, None, "")
    assert unknown.clock_excluded is False and unknown.clock_boosted is False


def test_a_side_on_a_verdict_that_did_not_fail_is_refused(bm):
    """A side is the direction a FAILURE went. A passing verdict carrying one
    was assembled from two sources and one of them is wrong; a row like that
    is refused at construction rather than read either way."""
    with pytest.raises(ValueError, match="did not fail"):
        _clocked(bm, 1, 1980.0, True, "high")
    with pytest.raises(ValueError, match="not one of"):
        _clocked(bm, 1, 1980.0, False, "boosted")


def test_tread_clock_reports_the_side_and_only_an_all_high_tread_is_high(bm):
    """The side travels with the majority verdict. A tread whose repeats failed
    on BOTH edges was timed at two operating points and its median is a
    blend, which is the throttle and not the boosted state: it is LOW."""
    all_high = [_clocked(bm, r, 1980.0, False, "high") for r in range(1, 6)]
    level, _, mhz, side = bm.tread_clock(all_high, 128)[4]
    assert level is False and side == "high" and mhz == 1980.0
    mixed = ([_clocked(bm, 1, 1980.0, False, "high")]
             + [_clocked(bm, r, 1400.0, False, "low") for r in (2, 3)])
    level, _, _, side = bm.tread_clock(mixed, 128)[4]
    assert level is False and side == "low"
    fine = [_clocked(bm, r, 1515.0, True, "") for r in range(1, 4)]
    assert bm.tread_clock(fine, 128)[4][3] == ""
    # One boosted repeat out of five is absorbed by the median, exactly as
    # one throttled repeat is, and the tread's side is then blank.
    one_high = ([_clocked(bm, 1, 1980.0, False, "high")]
                + [_clocked(bm, r, 1515.0, True, "") for r in range(2, 6)])
    assert bm.tread_clock(one_high, 128)[4][0] is True
    assert bm.tread_clock(one_high, 128)[4][3] == ""


def test_the_level_side_round_trips_through_the_csv(bm, tmp_path):
    path = tmp_path / "cells.csv"
    for row in (_clocked(bm, 1, 1980.0, False, "high"),
                _clocked(bm, 2, 1400.0, False, "low"),
                _clocked(bm, 3, 1515.0, True, "")):
        bm.append_sample(path, row)
    _, back = bm.read_samples(path)
    assert [s.clock_level_side for s in back] == ["high", "low", ""]
    assert [s.clock_boosted for s in back] == [True, False, False]
    assert [s.clock_sagged for s in back] == [False, True, False]
    assert [s.clock_excluded for s in back] == [False, False, False]
    # A cells.csv from before the column reads back blank on every row. That
    # is the value on a passing or undetermined verdict, so those rows resume;
    # a FAILED row's side cannot be recovered from the file and is refused
    # rather than read as LOW.
    header = ("block_m,tiles,rows_per_expert,tokens,rep,ms_p50,ms_min,"
              "ms_stdev,iters,status,detail,sm_clock_load_mhz,"
              "clock_level_ok,clock_drift_ok\n")
    old = tmp_path / "old.csv"
    old.write_text(header + "128,4,512,2048,1,1.0,1.0,0.0,10,ok,,1515.0,True,True\n"
                   + "128,4,512,2048,2,1.0,1.0,0.0,10,ok,,,,\n")
    _, legacy = bm.read_samples(old)
    assert [s.clock_level_side for s in legacy] == ["", ""]
    assert [s.clock_excluded for s in legacy] == [False, False]
    failed = tmp_path / "failed.csv"
    failed.write_text(header + "128,4,512,2048,1,1.0,1.0,0.0,10,ok,,1980.0,False,True\n")
    with pytest.raises(ValueError, match="re-measure"):
        bm.read_samples(failed)


def test_boosted_treads_stay_in_the_fit_and_the_report_counts_them(bm):
    """THE ARM'S PAYLOAD. Six subject treads planted HIGH by the H200's own
    ratio (`SELF_TEST_HIGH_CLOCK_MHZ` against `SELF_TEST_REFERENCE_CLOCK_MHZ`)
    must be fitted exactly as the quiet world's are: same memory-tread count,
    same alpha, same outcome, zero excluded, six counted as kept, and the
    report saying in words which way their fixed-roof fraction is off.

    THE LOW SIDE IS PLANTED OVER THE WHOLE LADDER, and that is the physics, not
    a convenience. A per-tile clock is a property of the tile, so the H200's
    BLOCK_M=128 subject sits at 1395 MHz at EVERY tread; that is the state
    2026-09-09 stopped excluding. Planting six of eight would be a card that
    changed state mid-ladder, a genuinely bent ladder and a different claim,
    which V3's max-affine shape scores and the clock rule does not.

    AND THE LOW LADDER IS NOT THE QUIET ONE, which is the residual this rule
    leaves and which the recorded side exists to make visible. The high side
    moves no millisecond in this world because the memory branch binds at every
    tread and a boosted clock only makes the compute branch it is not on
    faster. The LOW side is not symmetric: `SELF_TEST_LOW_CLOCK_MHZ` is 1000
    against a 1980 reference, and at half the clock the compute branch rises
    THROUGH the memory branch, so those treads really are slower and the fitted
    alpha really does move. Every tread is still kept, counted and fitted --
    what the fit cannot do is compare them with a reference measured at another
    clock, and that is what the side on the row says. The 2026-09-09 H200
    spread is 1358-1980 MHz, a factor of 1.46, not 2."""
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    kw = dict(alpha=0.95, rho=175.0, bandwidth_gbps=bm.SELF_TEST_BANDWIDTH,
              noise=bm.PUBLISHED_LADDER_SPREAD)
    quiet = bm.planted_samples(cfg, **kw)
    hot = bm.planted_samples(cfg, high_clock_treads=(3, 4, 5, 6, 7, 8), **kw)
    cold = bm.planted_samples(cfg, low_clock_treads=tuple(range(1, 9)), **kw)
    assert sum(1 for s in hot if s.clock_boosted) == 6 * 5
    assert not any(s.clock_excluded for s in hot)
    assert all(s.sm_clock_load_mhz == bm.SELF_TEST_HIGH_CLOCK_MHZ
               for s in hot if s.clock_boosted)
    assert bm.SELF_TEST_HIGH_CLOCK_MHZ / bm.SELF_TEST_REFERENCE_CLOCK_MHZ \
        == pytest.approx(1980.0 / 1515.0)
    args = dict(ceiling_tflops=175.0 * bm.SELF_TEST_BANDWIDTH * 1e9 / 1e12,
                ceiling_source="test", compiles={128: 1, 256: 1},
                executed={128: 40, 256: 20}, ridge=175.0,
                bandwidth_gbps=bm.SELF_TEST_BANDWIDTH, draws=200)
    _, _, quiet_pay = bm.analyse_run(quiet, cfg, 2, **args)
    lines, _, hot_pay = bm.analyse_run(hot, cfg, 2, **args)
    _, _, cold_pay = bm.analyse_run(cold, cfg, 2, **args)
    assert hot_pay["excluded_drifted"] == 0
    assert hot_pay["kept_high_clock"] == 6
    assert hot_pay["kept_low_clock"] == 0
    assert hot_pay["kept_high_clock_reference"] == 0
    assert hot_pay["memory_points"] == quiet_pay["memory_points"]
    assert hot_pay["alpha"] == pytest.approx(quiet_pay["alpha"])
    assert hot_pay["ladder_outcome"] == quiet_pay["ladder_outcome"]
    assert cold_pay["excluded_drifted"] == 0
    assert cold_pay["kept_low_clock"] == 8
    assert cold_pay["kept_high_clock"] == 0
    assert len(cold_pay["scored_points"]) == len(quiet_pay["scored_points"])
    assert cold_pay["ladder_outcome"] == quiet_pay["ladder_outcome"]
    # Kept and fitted, and the fit MOVED, because half the clock really does
    # put the compute branch through the memory branch. The side is the record
    # that says so; it is not an exclusion.
    assert cold_pay["alpha"] != pytest.approx(quiet_pay["alpha"])
    cold_lines, _, _ = bm.analyse_run(cold, cfg, 2, **args)
    assert any("8 KEPT with LEVEL failed LOW" in ln for ln in cold_lines)
    assert any("UNDERSTATED on the low" in ln for ln in cold_lines)
    assert any("6 tread(s) KEPT with LEVEL failed HIGH" in ln for ln in lines)
    assert any("OVERSTATED on the high side" in ln for ln in lines)
    assert any("LEVEL high, kept" in ln for ln in lines)
    with pytest.raises(ValueError, match="more than one clock state"):
        bm.planted_samples(cfg, low_clock_treads=(3,), high_clock_treads=(3,),
                           **kw)
    with pytest.raises(ValueError, match="more than one clock state"):
        bm.planted_samples(cfg, low_clock_treads=(3,), drift_treads=(3,), **kw)


def test_the_high_clock_world_is_registered_done_with_every_tread_kept(bm):
    """The planted world, through `self_test` the way the pod's `--self-test`
    runs it, and its payload registration is what would catch the treads
    being dropped while the ladder still identified."""
    worlds = {w.name: w for w in bm.SELF_TEST_WORLDS}
    high = worlds["high-clock"]
    assert high.high_clock_treads == (3, 4, 5, 6, 7, 8)
    assert high.exit_code == exit_codes.DONE
    assert high.expect_payload == {"excluded_drifted": 0,
                                   "kept_high_clock": 6, "kept_low_clock": 0}
    assert worlds["drifting-clock"].expect_payload["excluded_drifted"] == 6
    assert worlds["low-clock"].expect_payload["kept_low_clock"] == 8
    assert worlds["low-clock"].expect_payload["excluded_drifted"] == 0
    _, gates, payload = _run_world(bm, high)
    assert high.check(gates, payload) == []
    # And the registration can FAIL: checked without a payload, every payload
    # key is reported missing rather than silently passed.
    assert len(high.check(gates, None)) == 3
    assert high.check(gates, dict(payload, kept_high_clock=0)) == [
        "payload['kept_high_clock']: registered 6, got 0"]


def test_the_registered_p1_baseline_is_computed_from_the_corpus_not_typed(bm):
    """The page carried `sd 0.078` as a literal while its own feasibility block
    computed 0.0843 from the same 22 ladders. The number is now formed from
    the corpus by the same arithmetic the audit table uses, and a checkout
    without a corpus prints NOT RECOMPUTABLE rather than a number."""
    import statistics
    corpus = bm.published_bc(ROOT / "results" / "published", "bf16",
                             bm.HARDWARE_DIR)
    assert corpus is not None, "the committed corpus must be readable here"
    records, _ = bm.load_corpus(ROOT / "results" / "published", "bf16",
                                bm.HARDWARE_DIR)
    rows = [bm.audit_record(r, draws=20) for r in records]
    ratios = [r.ratio for r in rows
              if r.level is not None and r.level.passes and r.ratio is not None]
    assert corpus.n == len(ratios) >= 2
    assert corpus.sd == pytest.approx(statistics.pstdev(ratios))
    assert corpus.median == pytest.approx(statistics.median(ratios))
    text = bm.predictions_text(2, corpus)
    assert f"sd {corpus.sd:.3f}" in text
    assert "0.078" not in text
    assert f"{corpus.n} published ladders" in text
    absent = bm.predictions_text(2, None)
    assert "NOT RECOMPUTABLE" in absent and "sd " not in absent.split("P2")[0]
    assert bm.published_bc(ROOT / "nowhere", "bf16", bm.HARDWARE_DIR) is None



def test_the_reference_clock_resolves_under_the_decorated_measured_name():
    """2026-09-09 H200 pod: this arm handed `load_measured`'s decorated
    "NVIDIA H200 (measured)" to the sweep's reference-clock walk, the slug
    named a file no card has, and the arm REFUSED a sound calibration before
    timing a tread. Both names now resolve to the same record."""
    SWEEP = _load("sweep_under_the_decorated_name", "block_m_crossing_sweep.py")
    plain = SWEEP.reference_clock_mhz("NVIDIA H200")
    decorated = SWEEP.reference_clock_mhz("NVIDIA H200 (measured)")
    assert plain[0] is not None, plain[1]
    assert decorated[0] == plain[0]
    # The reason names the card as the caller spelled it; the field read is one.
    assert decorated[1].split(": ", 1)[1] == plain[1].split(": ", 1)[1]


# --------------------------------------------------------------------------
# The 2026-09-09 H200 session, replayed. The cells are committed, so this is
# `analyse_run` over what the pod wrote and not a fixture.
# --------------------------------------------------------------------------

SESSION = (ROOT / "results" / "published"
           / "2026-09-09-nvidia_h200-gaps-session" / "results")

needs_session = pytest.mark.skipif(
    not list(SESSION.glob("bm128_depth/*/cells.csv")),
    reason="the 2026-09-09 H200 gaps session is not in this checkout")

#: That session's own calibration, from its cells.csv provenance columns.
SESSION_RIDGE = 152.8120650229884
SESSION_BANDWIDTH = 4374.549151491332
SESSION_CEILING = 668.4838893839521
SESSION_PINNED = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
                  "num_warps": 8, "num_stages": 4}


def _replay_session(bm):
    from moe.spec import MODEL_CONFIGS as CFGS
    cfg = CFGS["mixtral-8x7b"]
    _, samples = bm.read_samples(next(SESSION.glob("bm128_depth/*/cells.csv")))
    return cfg, samples, bm.analyse_run(
        samples, cfg, 2, SESSION_CEILING, "NVIDIA H200 (measured)",
        {128: 16, 256: 17}, {128: 16, 256: 17}, ridge=SESSION_RIDGE,
        bandwidth_gbps=SESSION_BANDWIDTH, pinned=SESSION_PINNED, seed=0,
        draws=400)


@needs_session
def test_the_committed_h200_session_qualifies_its_reference_and_scores_16_treads(
        bm):
    """THE ACCEPTANCE REPLAY, and both halves of the 2026-09-09 change.

    As run, this arm scored ZERO treads: all 16 BLOCK_M=128 subject treads were
    LOW by 7-repeat majority and excluded, and the reference was refused on
    NON-VACUITY at 1.532 -- a floor of 0.838 of the DENSE GEMM roof that no
    fused layer in the corpus reaches -- while the log printed only "its LEVEL
    is wrong". V0 FAILED, V1 was UNKNOWN, exit INVALID.

    With DRIFT as the only exclusion and the non-vacuity floor taken on the
    FUSED footing, the same cells qualify the BLOCK_M=256 reference at 54.7% of
    the ceiling, score all 16 treads, and every VALIDITY gate passes. The arm
    exits CLAIM_FAIL, which is a result and not a retry.
    """
    _, _, (lines, gates, payload) = _replay_session(bm)
    by = {g.tag: g for g in gates}

    assert payload["reference"]["block_m"] == 256
    assert payload["reference_vacuity_basis"] == "fused"
    assert payload["reference_refusals"] == []
    # The floor is the design constant 2 BM_min / (b x ridge) on this footing,
    # and the reference's own plateau is what the fused roof is taken from.
    assert payload["reference_vacuity_floor"] == pytest.approx(
        2 * bm.SMALL_TILE_BLOCK_M / (2 * SESSION_RIDGE))
    assert payload["reference_vacuity_ratio"] == pytest.approx(
        payload["reference_vacuity_floor"])
    assert payload["reference_roof_fraction"] == pytest.approx(0.547, abs=5e-4)
    lo, hi = bm.FUSED_ROOF_BAND
    assert lo <= payload["reference_roof_fraction"] <= hi

    assert len(payload["scored_points"]) == 16
    assert payload["excluded_drifted"] == 0
    assert payload["kept_low_clock"] == 16
    assert payload["kept_high_clock"] == 0
    assert payload["kept_high_clock_reference"] == 8

    assert by["V1"].passed is True and "54.7%" in by["V1"].observed
    assert all(g.passed is True for g in gates if g.kind == bm.VALIDITY), \
        [g.name for g in gates if g.kind == bm.VALIDITY and g.passed is not True]
    assert bm._exit_code(gates) == exit_codes.CLAIM_FAIL

    # C1 is decidable rather than voided by a refused reference: the fit had a
    # compute branch, put all 16 treads above it, and then declined on the
    # PARALLEL_BRANCH tolerance -- B/C = 1.102, inside 0.15 by 13 sigma, which
    # C2 scores as a FAIL. That pair is the arm's result.
    assert payload["ladder_outcome"] == "undecided_parallel_branch"
    assert by["C1"].passed is None
    assert "undecided_parallel_branch" in by["C1"].observed
    assert by["C2"].passed is False
    assert payload["ratio"] == pytest.approx(1.1024, abs=5e-4)
    assert payload["margin_sigma"] < -3

    text = "\n".join(lines)
    assert "no tread was excluded for a drifting clock" in text
    assert "16 KEPT with LEVEL failed LOW" in text
    assert "UNDERSTATED on the low" in text
    assert "NO: DRIFT" not in text


@needs_session
def test_the_session_replay_prints_the_vacuity_derivation_and_the_partner(bm):
    """THE REFUSAL THAT COST A SESSION WAS NEVER PRINTED. The 2026-09-09 log
    carried `early.note` alone -- "BLOCK_M=256 ladder is proportional to 0.5%
    but its LEVEL is wrong ... REFUSED" -- over a non-vacuity refusal whose
    number was 1.532 and whose floor was 0.838 of the dense GEMM roof. The page
    now carries the ratio, the floor, the footing it stands on and the band the
    measured plateau had to land in, whether it passed or failed.

    It also says how many treads of the BLOCK_M=32 scaling partner arrived. On
    this committed session none did -- the partner is a 2026-09-09 addition and
    the pod ran the old pairing -- so the page states the floor the ladders
    that DID arrive would set, and that the fused footing clears that one too.
    """
    _, _, (lines, _, payload) = _replay_session(bm)
    text = "\n".join(lines)
    assert "LEVEL non-vacuity" in text
    assert "2 BM_min / (b x ridge)" in text
    assert "footing            FUSED" in text
    assert "CONTROL'S MEASURED PLATEAU" in text
    assert f"[{bm.FUSED_ROOF_BAND[0]:.3f}, {bm.FUSED_ROOF_BAND[1]:.3f}]" \
        in text
    assert payload["partner_treads"] == 0
    assert payload["partner_block_m"] == bm.SMALL_TILE_BLOCK_M
    assert "NONE ARRIVED" in text
    assert "over the ladders that did arrive [128, 256] it is 0.838" in text


@needs_session
def test_the_reference_is_never_qualified_from_the_subject_ladder(bm):
    """`compute_reference` used to fall through to the next block size down
    when the candidate had under three treads, and the next one down here is
    the memory-bound SUBJECT. `candidates` names the one ladder that may be the
    reference; the others are compared against and never qualified."""
    cfg, samples, _ = _replay_session(bm)
    ref_points, _, _ = bm.collapse(samples, bm.REFERENCE_BLOCK_M)
    sub_points, _, _ = bm.collapse(samples, bm.SUBJECT_BLOCK_M)
    # Two reference treads only: under the three a qualification needs.
    cells = [bm.SWEEP.make_cell(cfg, n * 256, 256, ms, sm_count=1, block_n=1)
             for n, ms in ref_points[:2]]
    cells += [bm.SWEEP.make_cell(cfg, n * 128, 128, ms, sm_count=1, block_n=1)
              for n, ms in sub_points]
    ref = bm.SWEEP.compute_reference(
        cells, bm.BLOCK_SIZES, cfg=cfg, ridge=SESSION_RIDGE,
        bandwidth_gbps=SESSION_BANDWIDTH, b=2, pinned=SESSION_PINNED,
        candidates=(bm.REFERENCE_BLOCK_M,),
        fused_roof_band=bm.FUSED_ROOF_BAND)
    assert ref.block_m is None
    assert ref.refused_block_m is None
    assert "no candidate ladder (BLOCK_M=256)" in ref.note
    assert ref.skipped and "2 aligned tread(s)" in ref.skipped[0]
    assert "NOT tried as the reference" in ref.skipped[0]
    rendered = "\n".join(ref.render())
    assert "PASSED OVER" in rendered
    # Without `candidates` the same cells qualify BLOCK_M=128, the subject.
    loose = bm.SWEEP.compute_reference(
        cells, bm.BLOCK_SIZES, cfg=cfg, ridge=SESSION_RIDGE,
        bandwidth_gbps=SESSION_BANDWIDTH, b=2, pinned=SESSION_PINNED)
    assert loose.block_m == bm.SUBJECT_BLOCK_M or loose.refused_block_m == \
        bm.SUBJECT_BLOCK_M, "the fall-through this argument closes"


@needs_session
def test_both_qualification_call_sites_pass_the_same_block_sizes(bm):
    """THE RECURRING DEFECT, pinned. On 2026-09-09 `main`'s early qualifier
    passed `(REFERENCE_BLOCK_M,)` and `analyse_run` passed
    `(SUBJECT_BLOCK_M, REFERENCE_BLOCK_M)`, so the two scored the vacuity floor
    at 1.675 and 0.838 of the roof and the log printed the first over a run the
    second judged. `BLOCK_SIZES` is the one tuple both pass."""
    source = (ROOT / "scripts" / "bm128_depth.py").read_text()
    calls = [ln for ln in source.splitlines()
             if "compute_reference(" in ln and "SWEEP." in ln]
    assert len(calls) == 2, calls
    body = source
    for anchor in ("ref_cells, BLOCK_SIZES, cfg=cfg, ridge=ridge",
                   "cells, BLOCK_SIZES, cfg=cfg, ridge=ridge"):
        assert anchor in body, anchor
    assert body.count("candidates=(REFERENCE_BLOCK_M,)") == 2
    assert body.count("fused_roof_band=FUSED_ROOF_BAND") == 2
    assert bm.BLOCK_SIZES == (bm.SMALL_TILE_BLOCK_M, bm.SUBJECT_BLOCK_M,
                              bm.REFERENCE_BLOCK_M)


def test_the_fused_roof_band_mirrors_the_sibling_that_registered_it(bm):
    """`tile_cap_test` registers what its V3 ADMITS for a fused layer's roof,
    and this file mirrors it rather than importing, because that script imports
    a GPU stack at module scope. A mirror that can drift is a mirror nobody can
    trust, so it is asserted against the source.

    AND IT IS NOT THE SIBLING'S `FUSED_PLATEAU_BAND`, WHICH IS WHAT IT WAS
    CALLED HERE UNTIL 2026-09-09. That constant is (0.465, 0.756), the interval
    the 26 published fused plateaus occupy; this one runs to the dense peak
    plus the tolerance a world generated AT the roof needs, 1.05, which is a
    ruler-sanity bound and no corpus figure. Two module-level constants under
    one name six values apart at the top is a collision a reader resolves by
    reading whichever file is open, and the report page said "[0.465, 1.050],
    which is where the 26 published fused layers sit" over a corpus that stops
    at 0.756. Both edges are pinned here, and so is the fact that the two names
    now differ."""
    text = (ROOT / "scripts" / "tile_cap_test.py").read_text()
    floor = float(text.split("FUSED_ROOF_FLOOR = FUSED_PLATEAU_BAND[0]")[0]
                  .split("FUSED_PLATEAU_BAND = (")[1].split(",")[0])
    corpus_top = float(text.split("FUSED_PLATEAU_BAND = (")[1]
                       .split(",")[1].split(")")[0])
    ceiling = float(text.split("FUSED_ROOF_CEILING = ")[1].split("\n")[0])
    tol = float(text.split("FUSED_ROOF_CEILING_TOLERANCE = ")[1].split("\n")[0])
    assert bm.FUSED_ROOF_BAND == (floor, ceiling * (1 + tol))
    assert not hasattr(bm, "FUSED_PLATEAU_BAND"), (
        "the name belongs to the sibling's corpus interval; this band admits "
        "a wider set and must not answer to it")
    assert bm.FUSED_ROOF_BAND[1] > corpus_top, (
        "if these ever coincide the sentence about where the corpus sits and "
        "the sentence about what is admitted stop being two sentences")


# --------------------------------------------------------------------------
# R2 on this arm: the own-clock fraction beside the gated one, as a NUMBER.
# --------------------------------------------------------------------------

@needs_session
def test_v1_prints_the_issue_efficiency_beside_the_fraction_it_gates_on(bm):
    """R2 WAS IMPLEMENTED IN ONE OF THE STUDY'S TWO LADDER ARMS.

    `bm128_roofline` got a `@own clk` column, `Point.roof_fraction_at_clock`
    and the companion number on C1/C2/C3/C4. This file got a SENTENCE: the
    2026-09-09 report page said the roof at a tread's own clock "is the issue
    efficiency to read beside it" and nothing computed one, so the V1 line
    published "54.7% of 668.5" for a reference whose own rows sit at 1635 MHz
    against a 1485 MHz roof and the number a reader was told to read beside it
    existed nowhere.

    THE GATE STILL READS THE FIXED FRACTION. `RefLevel.passes` is 54.7% against
    [25%, 100%]; the companion is printed and scored by nothing.
    """
    _, _, (lines, gates, payload) = _replay_session(bm)
    v1 = next(g for g in gates if g.tag == "V1")
    assert v1.passed is True
    assert "54.7% of 668.5" in v1.observed
    assert "49.7% of the 736.0 at its own 1635 MHz" in v1.observed
    assert "issue efficiency, not a gate input" in v1.observed
    # The same line is on the report page, not only in the gate.
    assert "at its own 1635 MHz" in "\n".join(lines)
    # And in the payload, beside the fixed fraction and never instead of it.
    assert payload["reference_roof_fraction"] == pytest.approx(0.547, abs=5e-4)
    assert payload["reference_level_fraction_at_clock"] == pytest.approx(
        0.497, abs=5e-4)
    assert payload["reference_ladder_clock_mhz"] == 1635.0
    assert payload["reference_roof_clock_mhz"] == 1485.0
    # The rescale is exactly the driver's roof_at_cell_clock_tflops.
    assert payload["reference_level_fraction_at_clock"] == pytest.approx(
        payload["reference_roof_fraction"] * 1485.0 / 1635.0, rel=1e-9)


@needs_session
def test_every_subject_tread_row_carries_both_fractions(bm):
    """"LEVEL low, kept; fixed-roof fraction understated by the clock ratio" on
    sixteen rows, and no rescaled value on any of them, is a report telling a
    reader to correct a number it declines to correct. Both are columns now."""
    _, _, (lines, _, payload) = _replay_session(bm)
    text = "\n".join(lines)
    assert "of roof @own clk" in text
    rows = payload["subject_roof_fractions"]
    assert len(rows) == 16
    for row in rows:
        assert 0.4 < row["roof_fraction"] < 0.6
        # Every subject tread ran LOW, so its issue efficiency is HIGHER than
        # its fixed-roof fraction: that is the direction the prose claims.
        assert row["roof_fraction_at_clock"] > row["roof_fraction"]
        assert row["sm_clock_load_mhz"] is not None
        assert row["roof_fraction_at_clock"] == pytest.approx(
            row["roof_fraction"] * 1485.0 / row["sm_clock_load_mhz"], rel=1e-9)
    # And the printed table shows them: six numeric columns before "yes".
    body = [ln for ln in text.splitlines()
            if re.match(r"^\s{0,3}\d+\s+\d+\s+\d+\.\d+", ln)]
    assert len(body) == 16, body
    for ln in body:
        assert "yes" in ln
        # n rows ms slope reps spread of-roof @own-clk, then the verdict.
        assert len(ln.split()) >= 9, ln


def test_a_tread_with_no_clock_prints_n_a_rather_than_a_fabricated_rescale(bm):
    """None means NOT DETERMINED. A replay of a cells.csv written before the
    clock was a column, or a container with no NVML, must print no second
    fraction rather than one taken against the fixed roof twice."""
    from moe.spec import MODEL_CONFIGS as CFGS
    cfg = CFGS["mixtral-8x7b"]
    points = [(1, 1.0), (2, 2.0)]
    out = bm.tread_fractions(points, 128, cfg, 668.5,
                             {1: (None, None, None, ""),
                              2: (None, None, 1400.0, "")},
                             {1: None, 2: None})
    assert out[1][1] is None and out[1][2] is None
    # A clock with no reference to rescale against is equally undetermined.
    assert out[2][1] is None and out[2][2] == 1400.0
    assert out[1][0] > 0 and out[2][0] > 0


def test_an_early_refusal_is_printed_once(bm):
    """`ComputeReference.render` already emits "REFUSED: {why}" for every
    refusal. `_run` looped over `early.refusals` again under a second label,
    so the one text R4 added that call site to guarantee gets seen was printed
    twice on every refusal."""
    source = (ROOT / "scripts" / "bm128_depth.py").read_text()
    assert source.count('print(f"    REFUSAL: {why}")') == 0
    assert source.count('print("\\n".join(early.render()))') == 1
    sweep = (ROOT / "scripts" / "block_m_crossing_sweep.py").read_text()
    assert sweep.count('out.append(f"    REFUSED: {why}")') == 1


def test_the_printed_predictions_carry_this_cards_committed_ridge(bm, capsys):
    """P2 IS PRINTED BY --dry-run AND --audit, and it named a ridge the card no
    longer has: the H200 read 162.8 Op/B, then 152.8, then 155.9 in nine days
    and P2 kept whichever figure had been typed into it. A prediction quoting a
    superseded ruler is a prediction against a different machine.

    FIXED BY DELETING THE PAIR, not by retyping it. P2 now interpolates
    `calibrated_ridges_phrase()`, so the sentence an operator reads is built
    from the same files the run scores against and cannot lag them. The
    assertion is the same relation: what is printed IS what the two committed
    calibrations say, and no superseded reading of either survives in the
    text."""
    bm.main(["--dry-run"])
    text = capsys.readouterr().out
    phrase = bm.calibrated_ridges_phrase()
    assert phrase in text
    for card, slug in bm.CALIBRATION_SLUGS.items():
        hw = bm.load_hardware(slug)
        ridge = hw.peak("bf16") / hw.bandwidth_bytes_s
        assert f"{ridge:.1f} ({card.upper()})" in phrase
    # No reading this card has retired may appear beside the current one.
    for superseded in ("162.8 (H200)", "152.8 (H200)"):
        assert superseded not in text or superseded in phrase


def test_both_reference_level_call_sites_carry_both_clocks(bm):
    """THE RECURRING DEFECT, on this round's own change. `analyse_run` builds
    its RefLevel with the ladder's own median under-load clock and the roof's,
    so V1 prints the issue efficiency beside the fraction it gates on. `_run`'s
    EARLY qualification prints the same line on the pod before the expensive
    half of the run, and it is the one an operator reads live: a companion
    number on one of two call sites is exactly the shape this repository keeps
    finding.

    The third site, `audit_record`, reads PUBLISHED report rows that carry no
    clock at all, and its RefLevel says so rather than rescaling by a number it
    does not have."""
    source = (ROOT / "scripts" / "bm128_depth.py").read_text()
    calls = [ln for ln in source.splitlines() if "reference_level(" in ln
             and "def " not in ln and "gate_" not in ln]
    assert len(calls) == 3, calls
    assert source.count("clock_mhz=ladder_clock(samples, early.block_m)") == 1
    assert source.count("roof_clock_mhz=reference_clock)") == 1
    assert source.count("roof_clock_mhz=ref_roof_clock)") == 1
    # A ladder with no clock says UNKNOWN and rescales nothing.
    from moe.spec import MODEL_CONFIGS as CFGS
    bare = bm.reference_level(CFGS["mixtral-8x7b"], 256, 1.9744, 668.5, "x")
    assert bare.fraction_at_clock is None
    assert "UNKNOWN" in bare.line()
    assert bare.passes is True, "the FIXED fraction still gates"
