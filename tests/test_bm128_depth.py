"""BLOCK_M=128 is where the cap sits ON the ridge, so the fit cannot speak there.

`scripts/bm128_depth.py` was asked for FIVE clean memory-bound treads at
BLOCK_SIZE_M=128 and answers that they are unreachable on either card in this
study. That answer is arithmetic, so most of this file checks the arithmetic
rather than the plumbing, and the rest checks that the two gates the study did
not have actually fire on the two published fits that should never have shipped.

FOUR GROUPS.

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
    the three worlds are checked to come out differently.

The script is loaded by path, because `scripts/` is not a package.
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
    assert bm.main(["--dry-run"]) == 0
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

def test_the_planted_worlds_come_out_differently(bm):
    """Feasible, infeasible, feasible. A gate that cannot tell them apart is not
    a gate, and would have passed the published A100 ladder too."""
    _, gates = bm.self_test()
    assert len(gates) == 3
    assert all(g.passed is True for g in gates), \
        [(g.name, g.observed) for g in gates if g.passed is not True]


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
    by = {g.name.split()[0]: g for g in gates}
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
    by = {g.name.split()[0]: g for g in gates}
    assert by["C1"].passed is False
    assert by["C2"].passed is not True
    assert payload["memory_points"] < bm.TARGET_TREADS


def test_git_visibility_answers_for_a_results_path(bm):
    """`results/*` is ignored and only `results/published/` is excepted."""
    ignored = bm.git_visibility(ROOT / "results" / "bm128_depth" / "x")
    kept = bm.git_visibility(ROOT / "results" / "published" / "x")
    assert "IGNORED" in ignored or "unverified" in ignored
    assert "IGNORED" not in kept
