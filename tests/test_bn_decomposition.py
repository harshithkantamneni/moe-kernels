"""BLOCK_SIZE_N is the only knob that moves alpha_a and nothing else.

`scripts/bn_decomposition.py` sweeps BLOCK_SIZE_N at fixed BLOCK_M to split the
fitted alpha into its weight-side and activation-side parts, and -- the half
that matters more -- to leave a residual that says whether those parts are all
of it. Most of that argument is arithmetic, so most of this file checks the
arithmetic rather than the plumbing.

FIVE GROUPS.

  - THE IDENTITY. `H ceil(2F/BN) + F ceil(H/BN) = W/BN` exactly, which is what
    makes BN a clean lever; `K = W/(2H+3F)`, which is the study's third term
    derived rather than borrowed from the up-GEMM's reduction dimension; and
    the exact and linear forms agreeing where phi is small and disagreeing by a
    stated amount where it is not.
  - THE ESTIMATOR. Planted worlds go in and come back out to machine precision,
    through the SAME `fit_ladder` the study publishes, including the fixed-cost
    correction. A round trip that only exercised this file's own algebra would
    prove nothing about the number the study reports.
  - THE PRECONDITION, which is why the last attempt at this sweep was
    discarded: a compute reference 43.6x too slow is perfectly proportional and
    passed the old qualification at 0.2% error. Here it is built and the
    refusal is asserted, along with the two resource refusals that stop the
    setting being timed at all.
  - THE REFUSALS. Too few cells produce no fit rather than a fit with no
    degrees of freedom; a missing bootstrap produces no chi2 rather than zero;
    a run id that omits a swept knob is the bug that overwrote a whole arm
    once already.
  - THE GATES DISCRIMINATE, end to end through `main`: four planted worlds, and
    the residual gate has to answer differently in the ones with a missing term.

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
H200_G1_N64 = (PUBLISHED / "2026-09-01-nvidia_h200-alpha-surface-s4"
               / "mixtral-8x7b-bf16-r1024-g1-n64-d66ad3.report.json")
A100_BN256 = (PUBLISHED / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
              / "qwen2-57b-a14b-bf16-r1024-g1-n256-23a131.report.json")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module that is not there yet makes
    # the decorator fail with an AttributeError naming nothing useful.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BND = _load("bn_decomposition", "bn_decomposition.py")
SWEEP = BND.SWEEP

from moe.spec import MODEL_CONFIGS  # noqa: E402

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
QWEN = MODEL_CONFIGS["qwen2-57b-a14b"]


def args_for(**over):
    argv = []
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return BND.build_parser().parse_args(argv)


# --------------------------------------------------------------------------
# The identity.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [MIXTRAL, QWEN])
@pytest.mark.parametrize("bn", [16, 32, 64, 128, 256])
def test_activation_reread_is_exactly_w_over_bn(cfg, bn):
    """`H ceil(2F/BN) + F ceil(H/BN) - H - F = W/BN - H - F`, no approximation.

    The whole reason BLOCK_SIZE_N is a clean lever: the model's E, F and H
    cancel against the weight term and what is left is the study's own BM/BN
    ratio. Checked on both models and every BN on the grid, because a BN that
    did not divide `2F` or `H` would break the cancellation and the fit would
    be biased by the rounding rather than refused.
    """
    w = BND.weight_elements(cfg)
    assert cfg.hidden_size % bn == 0 and (2 * cfg.intermediate_size) % bn == 0
    expected = w / bn - cfg.hidden_size - cfg.intermediate_size
    assert BND.act_reread_elements(cfg, bn) == pytest.approx(expected)


def test_act_once_matches_the_sweeps_own_definition():
    """This file's read-once count must be the sweep's, in elements.

    A divergence would put the decomposition's `d0` and the published
    `alpha-corrected` column on two different definitions of the same traffic,
    and both are printed side by side in the report.
    """
    assert (BND.act_once_elements(MIXTRAL)
            == SWEEP.activation_bytes_per_row(MIXTRAL, 1))


def test_effective_k_is_derived_not_the_up_gemm_reduction_dim():
    """`K = W/(2H+3F)`, which is NOT the hidden size the ai_model uses."""
    k = BND.effective_k(MIXTRAL)
    assert k == pytest.approx(3 * 14336 * 4096 / (2 * 4096 + 3 * 14336))
    assert k == pytest.approx(3440.6, abs=0.1)
    assert abs(k / MIXTRAL.hidden_size - 1) > 0.15


def test_exact_and_linear_agree_where_phi_is_small_and_not_where_it_is_not():
    """(LIN) is (EXA) linearised, so the gap has to grow with phi.

    Stated as a test because the two forms are printed side by side and a
    reader has to be able to trust that they are one model: at BM=32, BN=256
    phi is 0.026 and they agree to 3%, at BM=128, BN=32 phi is 0.6 and they
    differ by more than 20% -- which is the reason the study's two-point
    alpha_a and this run's are not the same quantity.
    """
    kw = {"alpha_b": 0.61, "alpha_a": 0.14}
    assert BND.phi(MIXTRAL, 32, 256, 0.14) == pytest.approx(0.026, abs=0.002)
    assert BND.phi(MIXTRAL, 128, 32, 0.14) == pytest.approx(0.60, abs=0.02)
    near = abs(BND.alpha_fitted_exact(MIXTRAL, 32, 256, **kw)
               / BND.alpha_fitted_linear(MIXTRAL, 32, 256, **kw) - 1)
    far = abs(BND.alpha_fitted_exact(MIXTRAL, 128, 32, **kw)
              / BND.alpha_fitted_linear(MIXTRAL, 128, 32, **kw) - 1)
    assert near < 0.03
    assert far > 0.20


def test_the_two_readings_of_the_published_pair_disagree_by_thirty(monkeypatch):
    """The claim the docstring makes about why two points cannot settle this.

    The H200 G=1 mixtral pair is 0.9327 at BN=64 and 0.8235 at BN=256. Read
    through (LIN) that is alpha_a = 0.146; read through (EXA) it is above 4,
    which no miss fraction can be. Recomputed here rather than quoted.
    """
    u64, u256 = 0.9327, 0.8235
    lin = (u64 - u256) / (1.0 - 0.25)
    assert lin == pytest.approx(0.146, abs=0.005)

    def alpha_b_from(u, bn, aa):
        return u + BND.phi(MIXTRAL, 64, bn, aa) * (u - 1.0)

    # (EXA) needs the two points to imply ONE alpha_b; they do not until alpha_a
    # is far outside [0, 1].
    for aa in (0.10, 0.15, 0.5, 1.0):
        assert alpha_b_from(u64, 64, aa) - alpha_b_from(u256, 256, aa) > 0.0


def test_anchored_ratio_reproduces_three_published_facts():
    """One measured anchor, three consequences it was not fitted to.

    In the world this study's own G=1 ladders imply, the anchored `B/C` has to
    put BLOCK_M=64 and 32 memory bound at every tread -- the published arms
    measure 16 and 33 memory treads -- and BLOCK_M=256 compute bound at tread 1,
    which is what qualified it as the reference in 22 of 24 published arms.
    """
    ab, aa = BND.WORLD_LADDER
    kw = {"alpha_b": ab, "alpha_a": aa}
    assert BND.anchored_ratio(MIXTRAL, 128, 64, **kw) == pytest.approx(
        BND.ANCHOR_RATIO)
    for bm in (32, 64):
        assert BND.memory_treads(MIXTRAL, bm, 64, ratio=BND.anchored_ratio(
            MIXTRAL, bm, 64, **kw), treads=8, **kw) == 8
    assert BND.memory_treads(
        MIXTRAL, 256, 64, ratio=BND.anchored_ratio(MIXTRAL, 256, 64, **kw),
        treads=4, **kw) == 0


def test_tread_one_is_memory_bound_iff_ratio_exceeds_alpha_not_one():
    """The sign rule the whole grid design turns on.

    A ladder can have `B/C` below 1 and still be memory bound everywhere it is
    swept: what decides tread 1 is `ratio > alpha`, not `ratio > 1`. Getting
    this backwards is what made the first version of the predicted-cell table
    disagree with 22 published arms.
    """
    kw = {"alpha_b": 0.94, "alpha_a": 0.14}
    a = BND.alpha_fitted_exact(MIXTRAL, 64, 64, **kw)
    assert BND.memory_treads(MIXTRAL, 64, 64, ratio=a * 1.05, treads=4,
                             **kw) >= 1
    assert BND.memory_treads(MIXTRAL, 64, 64, ratio=a * 0.90, treads=4,
                             **kw) == 0


def test_achieved_rho_is_well_below_the_calibrated_ridge():
    """The gap the predictions are anchored to avoid.

    A calibrated-ridge B/C says the BLOCK_M=256 reference at BN=64 is memory
    bound. It measurably is not. The reason is that the kernel reaches only
    38-64% of peak compute while the memory side reaches more of peak
    bandwidth, so its own rho is far below the ridge.
    """
    rho = BND.achieved_rho(MIXTRAL, 2, alpha_b=0.92, alpha_a=0.146)
    assert 80.0 < rho < 130.0
    assert rho < 0.75 * BND.SWEEP.RIDGE_BAND[0]


# --------------------------------------------------------------------------
# The estimator: planted in, planted out, through the study's own fit.
# --------------------------------------------------------------------------

def planted_cells(alpha_b, alpha_a, *, group_m=16, bns=(32, 64, 128),
                  subjects=(32, 64, 128), noise=0.0, reps=3, extra=None,
                  ceiling=712.259):
    args = args_for(capability="9.0", group_m=group_m, reps=reps)
    rho = BND.achieved_rho(MIXTRAL, 2, alpha_b=alpha_b, alpha_a=alpha_a)
    bw = BND.PLANT_COMPUTE_FRACTION * ceiling * 1e3 / rho
    samples = BND.planted_samples(
        MIXTRAL, args, alpha_b=alpha_b, alpha_a=alpha_a, ridge=rho,
        bandwidth_gbps=bw, b=2, block_ns=bns, subjects=subjects, extra=extra,
        noise=noise, seed=0)
    base = dict(SWEEP.FIXED, num_stages=args.num_stages,
                num_warps=args.num_warps, GROUP_SIZE_M=group_m,
                BLOCK_SIZE_K=args.block_k)
    base.pop("BLOCK_SIZE_N", None)
    cells, verdicts, spreads = BND.arm_alphas(
        samples, MIXTRAL, block_ns=bns, subjects=subjects, ridge=rho,
        bandwidth_gbps=bw, b=2, base_pinned=base, capability=(9, 0),
        ceiling_tflops=ceiling, sm_count=132)
    return cells, verdicts, spreads


def test_planted_world_comes_back_to_machine_precision():
    """The round trip that makes every other number here readable.

    Planted through `planted_ms`, measured through the STUDY's `fit_ladder` and
    `compute_reference`, decomposed here. Noiseless, so the only thing that can
    move the answer is an error in the algebra -- and the tolerance is 1e-6,
    not a percent, because nothing in the chain is approximate.
    """
    cells, verdicts, _ = planted_cells(0.61, 0.14)
    assert all(v.ok for v in verdicts)
    fit = BND.decompose(cells, MIXTRAL, "EXA")
    assert fit.alpha_b == pytest.approx(0.61, abs=1e-6)
    assert fit.alpha_a == pytest.approx(0.14, abs=1e-6)
    assert fit.dof >= 1
    assert max(abs(r) for r in fit.residuals) < 1e-6


def test_the_raw_three_parameter_fit_also_recovers_the_fixed_cost():
    """The cross-check form, which fits `delta` instead of removing it.

    Exact on noiseless data -- which is why it is kept and printed -- and
    unusable on noisy data, which is why it is never gated. Both halves are
    asserted, the second in the test below.
    """
    cells, _, _ = planted_cells(0.61, 0.14)
    fit = BND.decompose(cells, MIXTRAL, "EXA3")
    assert fit.alpha_b == pytest.approx(0.61, abs=1e-5)
    assert fit.alpha_a == pytest.approx(0.14, abs=1e-5)
    assert fit.delta is not None and fit.delta > 0


def test_the_raw_three_parameter_fit_is_the_one_that_blows_up_under_noise():
    """Why `alpha_upper` is the primary observable and `delta` is not fitted.

    Its third column is `-alpha`, and alpha varies by only about 10% across
    this whole grid, so the column is nearly the intercept's. The two-parameter
    fit on the same noisy timings stays near the planted value while the
    three-parameter one leaves the physical range entirely.
    """
    cells, _, _ = planted_cells(0.61, 0.14, noise=0.004, reps=9)
    two = BND.decompose(cells, MIXTRAL, "EXA")
    three = BND.decompose(cells, MIXTRAL, "EXA3")
    assert two.alpha_b == pytest.approx(0.61, abs=0.05)
    assert abs(three.alpha_b - 0.61) > 4 * abs(two.alpha_b - 0.61)


def test_a_missing_term_shows_up_as_residual_and_a_present_one_does_not():
    """C2's whole content, at the level of the fit rather than the gate."""
    clean, _, _ = planted_cells(0.61, 0.14, noise=0.002, reps=5)
    bent, _, _ = planted_cells(0.61, 0.14, noise=0.002, reps=5,
                               extra=lambda bm, bn: 0.004 * (bm / bn) ** 2)
    rms_clean = BND.decompose(clean, MIXTRAL, "EXA").rms
    rms_bent = BND.decompose(bent, MIXTRAL, "EXA").rms
    assert rms_bent > 10 * rms_clean


def test_structure_is_read_only_when_there_is_a_residual_to_read():
    """A residual with a shape reports a shape; scatter reports scatter.

    The correlation is not asserted against a high bar, and deliberately: with
    seven cells and two fitted parameters the fit ABSORBS most of a planted
    quadratic, so the surviving correlation is moderate even when the term is
    real. What has to hold is the comparison -- a bent world's worst column
    beats a clean world's -- and that the reading is suppressed entirely when
    chi2 says the residual is noise, so a correlation over scatter is never
    printed as a finding.
    """
    clean, _, _ = planted_cells(0.61, 0.14, noise=0.001, reps=5)
    bent, _, _ = planted_cells(0.61, 0.14, noise=0.001, reps=5,
                               extra=lambda bm, bn: 0.004 * (bm / bn) ** 2)
    s_bent = BND.structure_of(BND.decompose(bent, MIXTRAL, "EXA"), bent,
                              MIXTRAL, chi2=99.0)
    s_clean = BND.structure_of(BND.decompose(clean, MIXTRAL, "EXA"), clean,
                               MIXTRAL, chi2=99.0)
    assert s_bent.read and s_clean.read
    assert abs(s_bent.worst_value) > abs(s_clean.worst_value)
    quiet = BND.structure_of(BND.decompose(bent, MIXTRAL, "EXA"), bent,
                             MIXTRAL, chi2=0.2)
    assert not quiet.read and "not read" in quiet.line()


def test_alpha_b_is_invariant_across_block_m_in_a_planted_world():
    """P3, at the level of the fit: nothing lets a miss fraction see the tile."""
    cells, _, _ = planted_cells(0.61, 0.14)
    per_bm = {bm: BND.decompose(cells, MIXTRAL, "EXA", block_m=bm)
              for bm in (32, 64)}
    vals = [f.alpha_b for f in per_bm.values()]
    assert all(v is not None for v in vals)
    assert max(vals) - min(vals) < 1e-6


# --------------------------------------------------------------------------
# The precondition: a reference that is proportional and wrong.
# --------------------------------------------------------------------------

def poisoned_arm(factor: float):
    """A BLOCK_M=256 ladder `factor` times too slow, and perfectly proportional.

    The A100 BLOCK_N=256 arm, reconstructed: 249.765 ms for one tile against
    5.724 ms for the identical setting in its BN=64 twin, 43.6x, and it
    qualified at 0.2% mean error because through-origin residual is scale free.
    """
    cells = []
    for n in range(1, 5):
        cells.append(SWEEP.make_cell(MIXTRAL, n * 256, 256, 1.03 * factor * n,
                                     sm_count=132, block_n=64))
    for n in range(1, 9):
        cells.append(SWEEP.make_cell(MIXTRAL, n * 128, 128, 0.7 + 0.55 * n,
                                     sm_count=132, block_n=64))
    return cells


def test_a_proportional_reference_at_the_wrong_level_is_refused():
    """The failure that cost this study 8 published cells, caught here.

    The candidate is proportional to well under a percent -- the shape test
    passes -- and is refused on LEVEL. The refusal has to SAY level, because a
    reader who sees "not identifiable" without it blames the tread count, which
    is exactly what happened.
    """
    verdict = BND.qualify_reference(
        poisoned_arm(43.6), (128, 256), 64, cfg=MIXTRAL, ridge=162.8,
        bandwidth_gbps=4374.5, b=2,
        pinned=dict(SWEEP.FIXED, BLOCK_SIZE_N=64), capability=(9, 0),
        ceiling_tflops=712.259)
    assert not verdict.ok
    assert any("TFLOP/s" in w or "roof" in w or "weight read" in w
               for w in verdict.refusals), verdict.refusals


def test_the_same_reference_at_the_right_level_qualifies():
    """The other half: the level check must not refuse a sound reference."""
    verdict = BND.qualify_reference(
        poisoned_arm(1.0), (128, 256), 64, cfg=MIXTRAL, ridge=162.8,
        bandwidth_gbps=4374.5, b=2,
        pinned=dict(SWEEP.FIXED, BLOCK_SIZE_N=64), capability=(9, 0),
        ceiling_tflops=712.259)
    assert verdict.ok, verdict.refusals
    assert BND.REFERENCE_LEVEL_FLOOR <= verdict.fraction <= 1.0


def test_a_subject_promoted_to_reference_is_refused():
    """How the published H200 BN=256 arm lost its BLOCK_M=128 cell in silence.

    With BLOCK_M=256 absent, `compute_reference` takes the next-largest ladder
    -- the primary subject -- which by assumption has no memory branch and
    reports no alpha. The arm then prints blanks that look like measurements.
    """
    # A ladder that is PROPORTIONAL, so the shape test passes and the refusal
    # below is reached: the point is the rank of the reference, not its fit.
    cells = [SWEEP.make_cell(MIXTRAL, n * 128, 128, 0.55 * n, sm_count=132,
                             block_n=256) for n in range(1, 9)]
    verdict = BND.qualify_reference(
        cells, (128,), 256, cfg=MIXTRAL, ridge=162.8, bandwidth_gbps=4374.5,
        b=2, pinned=dict(SWEEP.FIXED, BLOCK_SIZE_N=256), capability=(9, 0),
        ceiling_tflops=712.259)
    assert not verdict.ok
    assert any("not above the largest subject" in w for w in verdict.refusals)


def test_cross_bn_refusal_fires_at_the_failure_it_was_built_for():
    """43.6x refuses; the occupancy-sized spread the grid really has does not."""
    def v(bn, rate):
        return BND.RefVerdict(bn, 256, 1.0, 0.1, rate, 712.259,
                              rate / 712.259, (), "")
    bad, spread = BND.cross_bn_refusal([v(32, 8.0), v(64, 349.0)])
    assert bad and spread > 40
    ok, spread = BND.cross_bn_refusal([v(32, 240.0), v(64, 349.0)])
    assert not ok and spread == pytest.approx(349.0 / 240.0)


def test_block_n_256_is_refused_by_the_register_bill_before_any_timing():
    """P6, computed from the pinned constants alone and therefore off GPU.

    `BM x BN / (32 num_warps)` accumulator registers per thread: 256 at
    BM=BN=256 with num_warps=8, against a hardware maximum of 255. The
    accumulator alone does not fit, so no compute reference exists in that arm
    on any card.
    """
    res = SWEEP.tile_resources(dict(SWEEP.FIXED, BLOCK_SIZE_N=256), 256, 2,
                              (9, 0))
    assert res.acc_registers_per_thread == 256
    assert res.refusal
    plan = BND.build_plan(args_for(capability="9.0", block_n_list="64,256"),
                          MIXTRAL, 2, (9, 0), 162.8, 4374.5)
    assert plan.block_ns == (64,)
    assert any(k[0] == 256 for k in plan.refusals)


def test_an_a100_at_four_stages_loses_the_reference_and_the_run_refuses():
    """The shared-memory cliff, and the fix named with the number.

    `num_stages (BM BK + BK BN) b` is 192 KiB at BM=256, BN=128 and 4 stages,
    against sm_80's 163 KiB ceiling; at 3 stages it is 144 and fits. A run that
    lost that arm would have two BN values and no residual, so it is refused
    before the pod is paid for rather than reported afterwards.
    """
    plan4 = BND.build_plan(args_for(capability="8.0"), MIXTRAL, 2, (8, 0),
                           145.8, 1799.4)
    assert 128 not in plan4.block_ns
    plan3 = BND.build_plan(args_for(capability="8.0", num_stages=3), MIXTRAL,
                           2, (8, 0), 145.8, 1799.4)
    assert plan3.block_ns == (32, 64, 128)
    assert BND.main(["--dry-run", "--capability", "8.0"]) == 2
    assert BND.main(["--dry-run", "--capability", "8.0",
                     "--num-stages", "3"]) == 0


# --------------------------------------------------------------------------
# The import, which is an assumption and has to behave like one.
# --------------------------------------------------------------------------

def _lender(bn, rate=350.0):
    return BND.RefVerdict(bn, 256, 1.0, 0.1, rate, 712.259, rate / 712.259,
                          (), "own")


def _refused(bn):
    return BND.RefVerdict(bn, None, None, 0.0, None, 712.259, None,
                          ("not proportional",), "refused")


def test_a_branch_is_lent_only_when_two_arms_agree():
    target = _refused(32)
    assert not BND.import_reference(target, [_lender(64)], MIXTRAL, None).ok
    ok = BND.import_reference(target, [_lender(64), _lender(128)], MIXTRAL,
                              None)
    assert ok.ok and ok.imported and ok.block_m == 256
    assert "assumption" in ok.import_note
    apart = BND.import_reference(
        target, [_lender(64, 350.0), _lender(128, 30.0)], MIXTRAL, None)
    assert not apart.ok


def test_an_imported_branch_is_labelled_on_every_cell_it_touches():
    """A cell resting on another arm's ruler must not read like one that is not."""
    cells, verdicts, _ = planted_cells(0.61, 0.14, group_m=16)
    imported = [v.block_n for v in verdicts if v.imported]
    if not imported:
        pytest.skip("no arm needed an import in this planted world")
    for c in cells:
        if c.block_n in imported and c.usable:
            assert c.basis.startswith("IMPORTED")


# --------------------------------------------------------------------------
# Refusals rather than defaults.
# --------------------------------------------------------------------------

def test_too_few_cells_produce_no_fit_rather_than_a_fit_with_no_residual():
    cells, _, _ = planted_cells(0.61, 0.14, bns=(64,), subjects=(64,))
    fit = BND.decompose(cells, MIXTRAL, "EXA")
    assert fit.alpha_a is None and fit.alpha_b is None
    assert "degrees of freedom" in fit.note


def test_chi_square_refuses_rather_than_returning_zero():
    """A residual divided by an assumed floor would answer with the floor."""
    cells, _, _ = planted_cells(0.61, 0.14)
    fit = BND.decompose(cells, MIXTRAL, "EXA")
    empty = BND.Bootstrap(0, {}, {}, None, None, None, {}, "no draws")
    chi2, why = BND.chi_square(fit, cells, empty, MIXTRAL)
    assert chi2 is None and "bootstrap" in why


def test_a_bootstrap_needs_two_repeats_and_says_so():
    args = args_for(capability="9.0", reps=1)
    rho = BND.achieved_rho(MIXTRAL, 2, alpha_b=0.61, alpha_a=0.14)
    samples = BND.planted_samples(
        MIXTRAL, args, alpha_b=0.61, alpha_a=0.14, ridge=rho,
        bandwidth_gbps=3000.0, b=2, block_ns=(64,), subjects=(64,),
        noise=0.0, seed=0)
    boot = BND.run_bootstrap(samples, MIXTRAL, [], draws=10, seed=0,
                             form="EXA", block_ns=(64,), subjects=(64,),
                             ridge=rho, bandwidth_gbps=3000.0, b=2,
                             base_pinned=dict(SWEEP.FIXED), capability=(9, 0),
                             ceiling_tflops=712.259, sm_count=132)
    assert boot.alpha_a_sd is None and boot.draws == 0
    assert "two repeats" in boot.note


def test_ladder_rows_refuses_a_model_that_cannot_fill_a_tile_stack():
    """A nudged row is not a full tile stack, and a fit over one is padding."""
    with pytest.raises(SystemExit) as exc:
        BND.ladder_rows(MODEL_CONFIGS["deepseek-v2-lite"], 32, 1024, 8)
    assert "multiple of" in str(exc.value)


def test_the_reference_block_size_may_not_also_be_a_subject():
    assert BND.main(["--tiles", "64,256", "--dry-run", "--capability",
                     "9.0"]) == 2


def test_non_vacuity_fails_when_a_count_is_zero():
    assert BND.gate_non_vacuity({"cells": 4, "draws": 0}).passed is False
    assert BND.gate_non_vacuity({"cells": 4, "draws": 9}).passed is True


def test_a_gate_that_could_not_run_reads_unknown_and_never_pass():
    boot = BND.Bootstrap(0, {}, {}, None, None, None, {}, "no draws")
    empty = BND.Decomposition("EXA", None, None, None, None, 0, 2, (), (), (),
                              "nothing")
    assert BND.gate_sharpness(boot).passed is None
    assert BND.gate_alpha_a(empty, boot, sharp=False).passed is None
    assert BND.gate_residual(empty, None, "no fit",
                             BND.Structure({}, "", None, False)).passed is None
    assert BND.gate_tempo(empty, boot, 1).passed is None
    assert BND.gate_physicality(empty, boot).passed is None


def test_physicality_fails_when_alpha_b_leaves_the_unit_interval():
    """A miss fraction above 1 says an extra tile costs more than a full read."""
    boot = BND.Bootstrap(50, {}, {}, 0.01, 0.01, None, {}, "x")
    over = BND.Decomposition("EXA", None, 1.16, 0.14, None, 7, 2, (0.0,),
                             ("a",), (1.0,), "")
    inside = BND.Decomposition("EXA", None, 0.61, 0.14, None, 7, 2, (0.0,),
                               ("a",), (1.0,), "")
    assert BND.gate_physicality(over, boot).passed is False
    assert BND.gate_physicality(inside, boot).passed is True


# --------------------------------------------------------------------------
# Identity: the run id and the cache key.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("knob,value", [
    ("model", "qwen2-57b-a14b"), ("dtype", "fp16"), ("block_n_list", "32,64"),
    ("tiles", "64,128"), ("r_max", 512), ("max_treads", 6), ("reps", 7),
    ("group_m", 16), ("block_k", 32), ("num_stages", 3), ("num_warps", 4),
    ("iters", 25), ("warmup", 10), ("cell_budget_ms", 200.0), ("seed", 1),
])
def test_every_swept_knob_changes_the_run_id(knob, value):
    """The bug that overwrote a whole arm, once per omitted field.

    A run id that omits a swept knob makes the second run resume into the
    first's directory, find every timing present, skip all of them, and print
    the first run's numbers under the second's heading. Nothing looks wrong,
    because the report renders the arguments from argv rather than from the
    timings it read.
    """
    base = args_for(capability="9.0")
    other = args_for(capability="9.0", **{knob: value})
    assert (BND.default_run_id(base, "h200")
            != BND.default_run_id(other, "h200"))


def test_the_card_is_in_the_run_id():
    """Two cards share a network volume, and every verdict is per-card scored."""
    a = args_for(capability="9.0")
    assert BND.default_run_id(a, "a100") != BND.default_run_id(a, "h200")


def test_the_triton_cache_key_carries_block_n(tmp_path, monkeypatch):
    """The sweep's key is BLOCK_M alone, which would collide across BN arms.

    Two BN arms at one BM would share a directory, the second would find it
    warm, compile nothing, and be scored by V1 as a broken override -- the same
    class of collision as a run id that omits a swept knob, one level down.
    """
    monkeypatch.setenv("TRITON_CACHE_DIR", "")
    first = BND.arm_cache(tmp_path, 32, 128)
    second = BND.arm_cache(tmp_path, 64, 128)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_results_land_under_a_path_git_reports_on(tmp_path):
    """`results/*` is ignored except `results/published/`, and this repo has
    already lost every published figure to a pattern nobody checked."""
    said = BND.git_visibility(ROOT / "results" / "bn_decomposition")
    assert "IGNORED" in said or "unverified" in said


# --------------------------------------------------------------------------
# End to end: the gates discriminate.
# --------------------------------------------------------------------------

def test_self_test_passes_where_the_design_has_power(capsys):
    """Four planted worlds, and the residual gate answering differently.

    This is the claim that C2 can settle anything: a gate that says the same in
    a world with a missing term and a world without one is not a test.
    """
    code = BND.main(["--self-test", "--capability", "9.0", "--group-m", "16",
                     "--reps", "9", "--draws", "40", "--plant-noise", "0.004",
                     "--fail-on-gate"])
    out = capsys.readouterr().out
    assert "S2 the residual gate discriminates" in out
    assert code == 0, out[-3000:]


def test_the_design_power_gate_fails_at_the_swizzle_that_cannot_resolve(capsys):
    """S4 is the reason this script has a recommended --group-m at all.

    At GROUP_SIZE_M=1 the corpus puts alpha near 0.93, the response moves with
    alpha_a as (1 - alpha_b), and the lever is worth 15% of its size at 16. The
    run still measures alpha_b and the residual; it cannot answer P1.
    """
    code = BND.main(["--self-test", "--capability", "9.0", "--group-m", "1",
                     "--reps", "9", "--draws", "40", "--plant-noise", "0.004",
                     "--fail-on-gate"])
    out = capsys.readouterr().out
    assert code == 1
    assert "S4 the design resolves alpha_a" in out
    assert "[FAIL] VALIDITY S4" in out


def test_dry_run_needs_no_gpu_and_prints_the_predictions(capsys):
    assert BND.main(["--dry-run", "--capability", "9.0"]) == 0
    out = capsys.readouterr().out
    assert "Predictions, registered before anything is measured" in out
    assert "PREDICTED CELLS" in out
    assert "TILE RESOURCE BILL" in out
    # The ridge a laptop uses is a HYPOTHESIS and has to say so, because seven
    # published A100 reports were scored against an H200 number.
    assert "HYPOTHESIS" in out


def test_the_published_arm_this_run_is_anchored_to_still_says_what_it_said():
    """The anchor is a measurement in a committed file, so pin the file.

    If the corpus is re-published with different numbers, the predictions in
    this script are stale and this test is where that surfaces.
    """
    doc = json.loads(H200_G1_N64.read_text())
    assert doc["fixed"]["BLOCK_SIZE_N"] == 64
    assert doc["fixed"]["GROUP_SIZE_M"] == 1
    assert doc["compute_reference"]["block_m"] == 256
    fit64 = doc["ladder"]["64"]
    assert fit64["memory_points"] == 16
    assert fit64["alpha"] == pytest.approx(0.9475, abs=0.005)
    assert BND.PLANTED_ALPHA_B[1] == pytest.approx(
        fit64["alpha"] + BND.phi(MIXTRAL, 64, 64, 0.14) * (fit64["alpha"] - 1),
        abs=0.01)


def test_the_corrupt_published_reference_is_the_one_the_level_gate_names():
    """43.6x, from the committed file rather than from memory."""
    doc = json.loads(A100_BN256.read_text())
    slope = doc["compute_reference"]["slope_per_tile"]
    assert doc["compute_reference"]["mean_rel_err"] < 0.01     # proportional
    rate = BND.implied_tflops(QWEN, 256, slope)
    assert rate / 262.371 < BND.REFERENCE_LEVEL_FLOOR          # and refused
    assert math.isfinite(rate)


def test_the_payload_is_json_and_keeps_the_import_provenance():
    """report.json is written after the metered half, so it must not throw.

    And it has to carry what the printout carries: an arm on a borrowed compute
    branch is a different measurement from one on its own, and a reader working
    from the file alone must be able to tell.
    """
    args = args_for(capability="9.0", group_m=16, reps=3)
    rho = BND.achieved_rho(MIXTRAL, 2, alpha_b=0.61, alpha_a=0.14)
    bw = BND.PLANT_COMPUTE_FRACTION * 712.259 * 1e3 / rho
    samples = BND.planted_samples(
        MIXTRAL, args, alpha_b=0.61, alpha_a=0.14, ridge=rho,
        bandwidth_gbps=bw, b=2, block_ns=(32, 64, 128),
        subjects=(32, 64, 128), noise=0.002, seed=0)
    base = dict(SWEEP.FIXED, num_stages=args.num_stages,
                num_warps=args.num_warps, GROUP_SIZE_M=16,
                BLOCK_SIZE_K=args.block_k)
    base.pop("BLOCK_SIZE_N", None)
    compiles = {(bn, bm): 1 for bn in (32, 64, 128)
                for bm in (32, 64, 128, 256)}
    _, gates, payload = BND.analyse_run(
        samples, MIXTRAL, args_for(capability="9.0", group_m=16, reps=3,
                                   draws=20),
        ridge=rho, bandwidth_gbps=bw, b=2, ceiling_tflops=712.259,
        ceiling_source="planted", capability=(9, 0), base_pinned=base,
        compiles=compiles, executed=dict(compiles), sm_count=132,
        block_ns=(32, 64, 128), subjects=(32, 64, 128))
    text = json.dumps(payload, indent=2, default=str)
    assert len(text) > 2000
    assert {a["basis"] for a in payload["arms"]} <= {"OWN", "IMPORTED"}
    assert any(g.kind == "VALIDITY" for g in gates)
    assert payload["fits"]["pooled_exact"]["alpha_a"] is not None
