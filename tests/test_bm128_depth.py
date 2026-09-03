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
through `moe/bench/ai_model.exact_cap` they sit below both cards' ridges. What
survives is the MEASURED `B/C` near 1, a ratio of two fitted slopes with no cap
and no ridge in it. `test_the_founding_premise_no_longer_straddles_the_ridge`
is where that retraction is checked against the published files, and it pins
the CORRECTION FACTOR as well as the caps: the first version of `premise_caps`
divided by `1 + phi + delta`, which is the factor for a cap taken from a raw
`B / (A + B)` fit, where the published `alpha-corrected` is an alpha_b and the
factor for one of those is `(alpha_b + phi) / alpha_b`. Both are above 1 and
the retraction survived the mix-up; the printed caps did not.

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

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
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
    162.8 on the H200. `$MOE_RESULTS_DIR` is a RunPod network volume shared
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
    that named none, and `undecided_low_clock` publishes the card as the tile.
    Both must reach the report as UNKNOWN with the outcome named.
    """
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    worlds = {w.name: w for w in bm.SELF_TEST_WORLDS}
    for name, outcome in (("straddle", "undecided_parallel_branch"),
                          ("low-clock", "undecided_low_clock")):
        w = worlds[name]
        samples = bm.planted_samples(
            cfg, alpha=w.alpha, rho=w.rho,
            bandwidth_gbps=bm.SELF_TEST_BANDWIDTH,
            noise=bm.PUBLISHED_LADDER_SPREAD,
            low_clock_treads=w.low_clock_treads)
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


def test_a_throttled_tread_is_excluded_from_the_fit_and_counted(bm):
    """R7. The instrument's clock columns have to reach the fit.

    A tread timed at `SELF_TEST_LOW_CLOCK_MHZ` against a roof measured at
    `SELF_TEST_REFERENCE_CLOCK_MHZ` sits well above the compute branch for a
    reason that has nothing to do with weight re-reads. Before the columns
    travelled, a ladder fitted across a throttling episode was
    indistinguishable from one that was not.
    """
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    kw = dict(alpha=0.95, rho=175.0, bandwidth_gbps=bm.SELF_TEST_BANDWIDTH,
              noise=bm.PUBLISHED_LADDER_SPREAD)
    quiet = bm.planted_samples(cfg, **kw)
    hot = bm.planted_samples(cfg, low_clock_treads=(3, 4, 5, 6, 7, 8), **kw)
    args = dict(ceiling_tflops=175.0 * bm.SELF_TEST_BANDWIDTH * 1e9 / 1e12,
                ceiling_source="test", compiles={128: 1, 256: 1},
                executed={128: 40, 256: 20}, ridge=175.0,
                bandwidth_gbps=bm.SELF_TEST_BANDWIDTH, draws=200)
    _, _, quiet_pay = bm.analyse_run(quiet, cfg, 2, **args)
    lines, _, hot_pay = bm.analyse_run(hot, cfg, 2, **args)
    assert quiet_pay["excluded_low_clock"] == 0
    assert hot_pay["excluded_low_clock"] == 6
    assert hot_pay["memory_points"] < quiet_pay["memory_points"]
    assert any("EXCLUDED" in ln for ln in lines)


def test_a_tread_is_excluded_on_a_majority_of_its_repeats_not_on_one(bm):
    """One throttled repeat out of seven is what the median exists to absorb."""
    def sample(rep, level):
        return bm.Sample(128, 4, 512, 2048, rep, 1.0, 1.0, 0.0, 10,
                         clock_level_ok=level)
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


def _run_world(bm, world, *, seed=0, noise=None, low_clock=None):
    """One planted world through `analyse_run`, the way `self_test` runs it."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    samples = bm.planted_samples(
        cfg, alpha=world.alpha, rho=world.rho,
        bandwidth_gbps=bm.SELF_TEST_BANDWIDTH,
        noise=bm.PUBLISHED_LADDER_SPREAD if noise is None else noise, seed=seed,
        low_clock_treads=(world.low_clock_treads if low_clock is None
                          else low_clock))
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
    """`undecided_low_clock` leaves a fit over the treads that survived, and
    `payload["alpha"]` carries it. Certifying that number would publish exactly
    what the exclusion exists to withhold, so the gate returns UNKNOWN and the
    page is INVALID."""
    gate = bm._gate_law(_Fit(0.89, "undecided_low_clock", True,
                             memory_points=2),
                        _Margin(1.36), 1.4, 0.05, 0.02)
    assert gate.passed is None
    assert "0.8900" in gate.observed
    assert "undecided_low_clock" in gate.observed


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

def test_a_throttled_tread_cannot_trip_a_validity_gate_it_was_excluded_from(bm):
    """THE EXCLUSION USED TO STOP AT THE FIT.

    `ladder_treads` drops a tread whose loaded clock came in below the roof's,
    but V2's inversions, V3's slope sequence and V5's replication were all still
    computed over the UNEXCLUDED medians. A throttled ladder BENDS at the
    throttle onset -- the planted world's slope runs 1.9 -> 3.0 -> 2.2 -- and
    that bend is a slope that rises and then falls, which V3 scores FAIL:
    INVALID, nothing quotable, for exactly the reason the exclusion exists to
    discount.

    The two verdicts point at different next steps. "The ladder is not
    describable by any two-line model" says the instrument is broken; "six
    treads were timed on a throttling card" says re-run on a quiet one. Only the
    second is true here, and this pins that the second is what comes back.
    """
    world = _world(bm, "low-clock")
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
    assert payload["excluded_low_clock"] == 6
    assert len(payload["scored_points"]) == 2
    assert by["V3"].passed is None
    assert "fewer than three treads" in by["V3"].observed
    assert payload["ladder_outcome"] == "undecided_low_clock"


def test_the_spread_the_gates_weigh_comes_from_the_treads_they_scored(bm):
    """A spread taken over excluded treads is a noise band measured partly on
    another compute branch, and it is the denominator of every sigma."""
    world = _world(bm, "low-clock")
    _, _, payload = _run_world(bm, world)
    assert payload["scored_spread"] != payload["subject_spread"]
    assert payload["scored_spread"] is not None


def test_every_tread_is_still_printed_with_the_dropped_ones_marked(bm):
    """The gates score the survivors; the reader has to see what was dropped,
    or the excluded count in the header points at nothing."""
    world = _world(bm, "low-clock")
    lines, _, _ = _run_world(bm, world)
    text = "\n".join(lines)
    assert text.count("NO: clock below the roof's") == 6
    assert text.count("  yes") >= 2


def test_a_ladder_whose_every_tread_was_excluded_is_vacuous_and_says_so(bm):
    """V0's counts are the report's own INPUTS, and after the exclusion the
    input is the scored ladder. A report that measured eight treads and scored
    none examined nothing, whatever the measured count says."""
    world = _world(bm, "low-clock")
    _, gates, payload = _run_world(bm, world,
                                   low_clock=tuple(range(1, 9)))
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
    retracted cap cleared the A100's ridge, the corrected one clears NEITHER
    card's at ANY alpha_a in [0, 1]. `exa_hi` sits at alpha_a = 0, the smallest
    phi and the largest cap the model allows, so the straddle is being given
    every benefit before it is refused.

    THE FACTOR IS PINNED SEPARATELY FROM THE CAPS, because the two can disagree
    and did. `alpha_corrected` is an alpha_b, so the retracted form is high by
    `(alpha_b + phi) / alpha_b`; the first version of this function divided by
    `1 + phi + delta` instead, which is the factor for an alpha straight out of
    a `B / (A + B)` fit, and printed 141.8 and 143.1 where the identity it named
    gives 140.4 and 139.9. Recomputing the factor from the alpha and the phi
    here means a cap corrected by the wrong one of the two cannot pass.
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
        n, k = 2 * cfg.intermediate_size, cfg.hidden_size
        for alpha_a, factor in ((0.0, c.factor_lo), (1.0, c.factor_hi)):
            phi = ai_model.phi(n, k, block_m=bm.SUBJECT_BLOCK_M,
                               block_n=bm.PREMISE_BLOCK_N, alpha_a=alpha_a)
            assert factor == pytest.approx(
                (c.alpha_corrected + phi) / c.alpha_corrected, rel=1e-12)
            # The factor that was used instead, and it is SMALLER, which is why
            # the printed caps came out high.
            assert factor > ai_model.lin_overstatement(phi=phi, delta=0.0)
        assert c.factor_lo > 1.0 and c.factor_hi > c.factor_lo
        assert c.exa_hi == pytest.approx(c.lin_cap / c.factor_lo, rel=1e-12)
        assert c.exa_lo < c.exa_hi < c.lin_cap
        assert not c.straddles, (c.arm, c.exa_hi, c.ridge)
    assert a100.exa_hi == pytest.approx(140.4, abs=0.1)
    assert h200.exa_hi == pytest.approx(139.9, abs=0.1)


def test_the_premise_still_straddles_when_the_alpha_is_small_enough(bm, tmp_path):
    """The PASS branch of the same predicate, planted.

    A retraction asserted only by a test that always answers "no" is a constant.
    Here the A100 report is copied with its BLOCK_M=128 alpha lowered to 0.80,
    which lifts the CORRECTED cap back over 145.8, and `straddles` says so. So
    the property is a property of the published alphas, not of the code.
    """
    doc = json.loads(A100_G64.read_text())
    doc["ladder"]["128"]["alpha_corrected"] = 0.80
    arm = tmp_path / "2026-09-02-nvidia_a100_sxm4_80gb-planted"
    arm.mkdir(parents=True)
    (arm / A100_G64.name).write_text(json.dumps(doc))
    (cap,) = bm.premise_caps(tmp_path)
    assert cap.straddles
    assert cap.exa_hi > cap.ridge


def test_the_audit_prints_the_premise_beside_its_retracted_form(bm, capsys):
    """`--audit` is the evidence for the module docstring, so the premise has to
    be ON that page and not only in the prose it went stale in.

    Both numbers are required: a corrected cap printed alone is a number the
    reader cannot compare with the published one, which is how a 32% correction
    stayed invisible.
    """
    if not PUBLISHED.exists():
        pytest.skip("no results/published on this checkout")
    bm.main(["--audit"])
    out = capsys.readouterr().out
    assert "## The founding premise, recomputed" in out
    assert "retracted  150.4 (1.032 of ridge, ABOVE)" in out
    assert "140.4" in out and "141.8" not in out
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

    An `alpha_corrected` above 1 is not a miss fraction, and `exact_cap` refuses
    it by name. The refusal a reader gets then has to say that a ladder was
    examined and rejected, because "no ladder carried an alpha" and "the one
    ladder that did carried 1.4" are different states of the corpus.
    """
    doc = json.loads(A100_G64.read_text())
    doc["ladder"]["128"]["alpha_corrected"] = 1.4
    arm = tmp_path / "2026-09-02-nvidia_a100_sxm4_80gb-planted"
    arm.mkdir(parents=True)
    (arm / A100_G64.name).write_text(json.dumps(doc))
    with pytest.raises(bm.PremiseNotRecomputable) as exc:
        bm.premise_caps(tmp_path)
    assert "nvidia_a100_sxm4_80gb-planted" in str(exc.value)
    assert "alpha_b=1.4" in str(exc.value)


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
    # THIS STUDY HAS ONE -- the A100 calibrates at 145.8 and the H200 at 162.8 --
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
