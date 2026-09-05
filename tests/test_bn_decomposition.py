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

from moe.bench import exit_codes  # noqa: E402
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
    ("iters", 25), ("warmup", 10.0), ("cell_budget_ms", 200.0), ("seed", 1),
    # New to the key on 2026-09-02, and all three set the measured
    # milliseconds: the warmup is now a DURATION, `--trials` is the number of
    # queue-deep trials the percentiles are taken over, and `--no-l2-flush`
    # decides whether every timed iteration starts with a cold L2. The roof was
    # measured flushed, so a flushed and an unflushed sweep must never share a
    # directory.
    ("trials", 5), ("plant_noise", 0.02),
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
    alpha_a as (1 - alpha_b), and the lever is a fraction of its size at 16. The
    run still measures alpha_b; it cannot answer P1.

    THE CODE IS 3 AND NOT 1. A VALIDITY gate that did not pass means nothing on
    the page may be quoted, which is INVALID in `moe.bench.exit_codes`'s table;
    1 is CLAIM_FAIL, a measured run whose pre-registered claim was refuted, and
    that is a RESULT rather than a broken instrument. This file used to return 1
    for both, which is the two-integers-two-meanings defect the shared table is
    named against.
    """
    code = BND.main(["--self-test", "--capability", "9.0", "--group-m", "1",
                     "--reps", "9", "--draws", "40", "--plant-noise", "0.004",
                     "--fail-on-gate"])
    out = capsys.readouterr().out
    assert code == exit_codes.INVALID
    assert "S4 the design resolves alpha_a" in out
    assert "[FAIL] VALIDITY S4" in out
    assert "RESULT: VALIDITY S4 FAIL" in out


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


# --------------------------------------------------------------------------
# THE POWER GUARD ON C2 (audit A9). A gate whose two outcomes cannot both occur
# is not a gate, and at GROUP_SIZE_M=1 this one's cannot: the planted
# missing-term world passes it with the same verdict the true world gets.
# Every test here plants BOTH branches -- the world where the guard fires and
# the world where it does not -- because a guard that only ever fires is as
# uninformative as the gate it was written to protect.
# --------------------------------------------------------------------------

def _planted_run(group_m, *, noise, reps=9, draws=40, alpha_a=0.14,
                 extra=None, probe=True, plant_noise=None):
    """One planted world scored the way a real run is scored, probe and all."""
    args = args_for(capability="9.0", group_m=group_m, reps=reps, draws=draws,
                    power_draws=draws)
    alpha_b = BND.planted_alpha_b(group_m)
    rho = BND.achieved_rho(MIXTRAL, 2, alpha_b=alpha_b, alpha_a=alpha_a)
    bw = BND.PLANT_COMPUTE_FRACTION * 712.259 * 1e3 / rho
    samples = BND.planted_samples(
        MIXTRAL, args, alpha_b=alpha_b, alpha_a=alpha_a, ridge=rho,
        bandwidth_gbps=bw, b=2, block_ns=(32, 64, 128),
        subjects=(32, 64, 128), extra=extra, noise=noise, seed=0)
    base = dict(SWEEP.FIXED, num_stages=args.num_stages,
                num_warps=args.num_warps, GROUP_SIZE_M=group_m,
                BLOCK_SIZE_K=args.block_k)
    base.pop("BLOCK_SIZE_N", None)
    compiles = {(bn, bm): 1 for bn in (32, 64, 128)
                for bm in (32, 64, 128, 256)}
    lines, gates, payload = BND.analyse_run(
        samples, MIXTRAL, args, ridge=rho, bandwidth_gbps=bw, b=2,
        ceiling_tflops=712.259, ceiling_source="planted", capability=(9, 0),
        base_pinned=base, compiles=compiles, executed=dict(compiles),
        sm_count=132, block_ns=(32, 64, 128), subjects=(32, 64, 128),
        probe_c2_power=probe, plant_noise=plant_noise)
    return lines, {g.token: g for g in gates}, payload


def test_c2_reads_unknown_where_the_missing_term_world_would_pass():
    """A9, the blocking finding, in the world it was found in.

    At GROUP_SIZE_M=1 the planted MISSING world comes back at chi2 1.78 against
    the 4.0 ceiling -- a PASS, the same verdict TRUTH gets -- so a C2 PASS at
    that swizzle could not have been a FAIL. The gate must say UNKNOWN and say
    why, and UNKNOWN counts against it.
    """
    lines, gates, payload = _planted_run(1, noise=0.008)
    c2 = gates["C2"]
    assert c2.passed is None
    assert "the planted missing-term world passes at this swizzle / noise" \
        in c2.observed
    assert c2.result_line().startswith("RESULT: CLAIM C2 UNKNOWN")
    assert payload["c2_power"]["discriminates"] is False
    # UNKNOWN is not a soft PASS: the arm cannot exit DONE on it.
    assert exit_codes.classify(g.scored() for g in gates.values()) \
        != exit_codes.DONE


def test_c2_keeps_its_own_verdict_where_the_probe_discriminates():
    """The other branch, and the reason this is a guard and not a muzzle.

    At GROUP_SIZE_M=16 with a quiet pod the missing-term world FAILS C2, so the
    gate has power and its own verdict stands. A guard that fired everywhere
    would have made C2 unquotable at every pinning, which is not a fix.
    """
    _, gates, payload = _planted_run(16, noise=0.004)
    assert payload["c2_power"]["discriminates"] is True
    assert gates["C2"].passed is True
    assert gates["C2"].result_line().startswith("RESULT: CLAIM C2 PASS")


def test_the_guarded_gate_can_still_fail_on_a_real_missing_term():
    """PASS, FAIL and UNKNOWN are all reachable, which is the whole claim.

    Where the probe discriminates AND the data carry the missing term, C2 must
    FAIL: the guard decides whether the verdict is readable, never what it is.
    """
    _, gates, payload = _planted_run(16, noise=0.004, extra=BND.missing_term)
    assert payload["c2_power"]["discriminates"] is True
    assert gates["C2"].passed is False
    assert gates["C2"].result_line().startswith("RESULT: CLAIM C2 FAIL")


def test_the_probe_plants_at_the_runs_own_measured_spread():
    """"At the measured spread" is the half that makes the guard about THIS pod.

    The audit's second finding was that the driver self-tested at 0.8% while the
    published H200 spread reaches 1.82% and S4 fails above about 1%. A probe
    that plants at a constant is a probe about some other pod.
    """
    _, _, payload = _planted_run(1, noise=0.017)
    measured = payload["measured_spread"]
    assert measured == pytest.approx(0.017, rel=0.35)
    assert payload["c2_power"]["noise"] == pytest.approx(measured)
    assert "this run's own repeats" in payload["c2_power"]["noise_source"]


def test_an_explicit_plant_noise_overrides_the_measured_one_and_says_so():
    """The operator may ask a what-if, and the report must name whose number it is."""
    _, _, payload = _planted_run(1, noise=0.008, plant_noise=0.02)
    assert payload["c2_power"]["noise"] == pytest.approx(0.02)
    assert payload["c2_power"]["noise_source"] == "given on the command line"


def test_a_probe_that_could_not_run_is_unknown_and_never_pass():
    """The FAIL branch of the probe itself: a crash establishes nothing.

    `C2Power.ran=False` is what a probe that raised leaves behind, and a gate
    guarded by a probe that did not run has not been shown to have power. It
    must not fall through to the unguarded verdict.
    """
    dead = BND.C2Power(False, 1, 9, 0.008, "this run's own repeats", 40,
                       note="RuntimeError: planted grid collapsed")
    assert dead.discriminates is None
    struct = BND.Structure({}, "quadratic in BM/BN", None, False)
    fit = BND.Decomposition("EXA", None, 0.94, 0.14, None, 7, 2,
                            (0.01,) * 7, ("BN=32 BM=32",) * 7, (0.5,) * 7,
                            "planted")
    gate = BND.gate_residual(fit, 0.5, "", struct, dead)
    assert gate.passed is None
    assert "could not run" in gate.observed
    unguarded = BND.gate_residual(fit, 0.5, "", struct, None)
    assert unguarded.passed is True          # the branch the guard overrides


def test_the_self_test_names_the_power_gate_and_exits_nonzero_at_g1(capsys):
    """The audit's own acceptance command, verbatim.

    `--self-test --capability 9.0 --group-m 1 --reps 17 --plant-noise 0.018
    --fail-on-gate` must exit non-zero, and S5 must be the gate that says why:
    the arm the driver schedules at this swizzle cannot establish C2.
    """
    code = BND.main(["--self-test", "--capability", "9.0", "--group-m", "1",
                     "--reps", "17", "--plant-noise", "0.018", "--draws", "40",
                     "--power-draws", "40", "--fail-on-gate"])
    out = capsys.readouterr().out
    assert code != 0
    assert code == exit_codes.INVALID
    assert "RESULT: VALIDITY S5 FAIL" in out


def test_the_power_gate_passes_where_the_design_has_power(capsys):
    """S5's other branch, so the gate is a gate."""
    code = BND.main(["--self-test", "--capability", "9.0", "--group-m", "16",
                     "--reps", "9", "--draws", "40", "--power-draws", "40",
                     "--plant-noise", "0.004", "--fail-on-gate"])
    out = capsys.readouterr().out
    assert code == exit_codes.DONE
    assert "RESULT: VALIDITY S5 PASS" in out


# --------------------------------------------------------------------------
# THE DESIGN-POWER LINE IS COMPUTED (audit A9's third clause, and B14).
# --------------------------------------------------------------------------

def test_no_spread_is_quoted_as_a_literal_in_what_the_run_prints(capsys):
    """The string that was wrong: "sd 0.11-0.13 ... at any rep count tried".

    The computed value at those settings was 0.176. A number about the
    estimator's precision that is typed rather than measured is the shape this
    whole slice exists to remove. Asserted on what the run PRINTS rather than on
    the source, because the source still names the retired string in the
    sentence that disowns it, and a reader who deletes that history is the next
    person to reintroduce it.
    """
    BND.main(["--dry-run", "--capability", "9.0", "--group-m", "1",
              "--reps", "17", "--power-draws", "40", "--plant-noise", "0.018"])
    out = capsys.readouterr().out
    assert "0.11-0.13" not in out
    assert "at any rep count tried" not in out
    assert "sd(alpha_a) = 0." in out           # computed, and printed


def test_the_plan_prints_a_computed_spread_and_an_mde(capsys):
    """--dry-run must say what this pinning can resolve, from a bootstrap.

    B14: no arm stated an MDE, and a gate threshold that is a prior rather than
    a noise-derived limit cannot be argued with. The line names its noise
    assumption in the same sentence, because an MDE without one is a limit
    presented as a fact.
    """
    assert BND.main(["--dry-run", "--capability", "9.0", "--group-m", "16",
                     "--reps", "9", "--power-draws", "40",
                     "--plant-noise", "0.008"]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert "planted TRUTH world at GROUP_SIZE_M=16" in out
    assert "MDE (alpha_a" in out
    assert "Assumption: lognormal spread 0.80%" in out
    assert "cross-arm floor" in out


def test_the_design_power_verdict_moves_with_the_swizzle():
    """The claim the plan makes: G=1 cannot resolve alpha_a and G=16 can.

    Both branches, on planted worlds, so "the lever is the swizzle" is measured
    here rather than asserted.
    """
    common = dict(b=2, ceiling_tflops=712.259, capability=(9, 0),
                  block_ns=(32, 64, 128), subjects=(32, 64, 128), sm_count=132,
                  noise=0.004, noise_source="planted", draws=40)
    hard = BND.design_power(MIXTRAL, args_for(capability="9.0", group_m=1,
                                              reps=9, draws=40), **common)
    easy = BND.design_power(MIXTRAL, args_for(capability="9.0", group_m=16,
                                              reps=9, draws=40), **common)
    assert hard.resolves is False
    assert easy.resolves is True
    assert hard.alpha_a_sd > easy.alpha_a_sd


def test_plant_noise_resolves_to_a_measurement_before_a_constant():
    """`--plant-noise` used to default to 0.008, the middle of the published range.

    S4 passes at 0.008 and fails at 0.015-0.020, so that default decided whether
    the experiment looked worth paying for while describing a pod quieter than
    half the corpus. All three branches, in order.
    """
    assert BND.resolve_plant_noise(0.02, 0.009, "measured") == (
        0.02, "given on the command line")
    assert BND.resolve_plant_noise(None, 0.009, "measured") == (0.009, "measured")
    value, source = BND.resolve_plant_noise(None, None, "no repeats")
    assert value == BND.PLANT_NOISE_FALLBACK
    assert "worst published" in source
    # And the fallback is the WORST published spread, not the middle one: the
    # middle is the number the audit found S4 being scored at.
    assert BND.PLANT_NOISE_FALLBACK > 0.008


def test_an_mde_needs_a_spread_and_says_so_when_there_is_none():
    """A limit computed from no spread is not zero, it is unknown.

    Zero is the value that would make every effect look resolvable, which is why
    `mde_one_sample` raises and `mde_line` prints UNKNOWN instead.
    """
    with pytest.raises(ValueError):
        BND.mde_one_sample(0.0)
    said = BND.mde_line(None, what="alpha_a", assumption="none")
    assert "UNKNOWN" in said and "0.0000" not in said
    # The convention is the study's own, not a fresh z value typed in here.
    assert BND.mde_one_sample(0.01) == pytest.approx(
        (BND.POWER.normal_ppf(0.975) + BND.POWER.normal_ppf(0.80)) * 0.01)


# --------------------------------------------------------------------------
# THE alpha_a BAND'S PROVENANCE (audit finding 32). The band cited four A100
# two-point slopes that exist in no file under results/published.
# --------------------------------------------------------------------------

def test_the_phantom_a100_slopes_are_no_longer_the_bands_basis(capsys):
    """0.106, 0.102, 0.129, 0.119 -- four numbers no committed file contains.

    They were P1's whole stated basis. The plan must now cite the two committed
    reports instead, and none of the four may appear in what it prints. The
    source still names them once, in the paragraph that says they are in no
    file, which is the history worth keeping.
    """
    BND.main(["--dry-run", "--capability", "9.0", "--power-draws", "20",
              "--plant-noise", "0.008"])
    out = capsys.readouterr().out
    for phantom in ("0.106", "0.102", "0.129", "0.119"):
        assert phantom not in out, f"{phantom} is back in the band's basis"
    assert "d66ad3.report.json" in out and "16cc16.report.json" in out


def test_the_band_is_what_the_committed_bn_pair_actually_says():
    """The pre-registered literal, re-derived from the two files it came from.

    The repo contains exactly one pair of arms differing in BLOCK_SIZE_N and
    nothing else, and it gives two slopes that disagree: 0.29 at BM=32 and 0.15
    at BM=64, with two-point sds of 0.086 and 0.043. The band covers both,
    widened by their own sds.
    """
    points = BND.published_two_point_alpha_a()
    by_bm = {p.block_m: p for p in points}
    assert set(by_bm) == {32, 64}
    assert by_bm[32].slope == pytest.approx(0.286, abs=0.01)
    assert by_bm[64].slope == pytest.approx(0.146, abs=0.01)
    assert by_bm[32].sd == pytest.approx(0.086, abs=0.005)
    assert by_bm[64].sd == pytest.approx(0.043, abs=0.005)
    assert BND.alpha_a_band_from_published(points) == BND.ALPHA_A_BAND
    # The retired band excluded the larger of its own two inputs, which is how
    # a measured 0.28 would have printed as "FAIL high".
    assert not (0.10 <= by_bm[32].slope <= 0.15)


def test_the_band_check_refuses_when_the_literal_no_longer_reads_back():
    """The FAIL branch: a literal that outlives its source is the whole finding."""
    with pytest.raises(BND.CorpusMissing):
        BND.check_alpha_a_band((0.10, 0.15))
    points, lines = BND.check_alpha_a_band()
    assert points and any("wide against input sds" in line for line in lines)


def test_a_pair_that_differs_in_more_than_block_n_is_refused():
    """Two arms differing in the swizzle too would give a swizzle slope."""
    pair = (PUBLISHED / "2026-09-01-nvidia_h200-alpha-surface-s4"
            / "mixtral-8x7b-bf16-r1024-g1-n64-d66ad3.report.json",
            PUBLISHED / "2026-09-01-nvidia_h200-alpha-surface-s4"
            / "mixtral-8x7b-bf16-r1024-g16-n64-69f35a.report.json")
    with pytest.raises(BND.CorpusMissing):
        BND.published_two_point_alpha_a(pair)


def test_a_missing_corpus_file_refuses_rather_than_defaulting(tmp_path):
    """A band whose provenance cannot be read is a band that cannot be checked."""
    with pytest.raises(BND.CorpusMissing):
        BND.published_two_point_alpha_a((tmp_path / "a.json", tmp_path / "b.json"))
    with pytest.raises(BND.CorpusMissing):
        BND.published_prior_sd(tmp_path / "NOISE_FLOOR.json")


def test_the_band_provenance_and_its_width_reach_the_plan(capsys):
    """P1 must carry where its band came from and how wide it is against its inputs."""
    assert BND.main(["--dry-run", "--capability", "9.0", "--power-draws",
                     "20", "--plant-noise", "0.008"]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert "mixtral-8x7b-bf16-r1024-g1-n256" in out
    assert "wide against input sds" in out


def test_the_sharpness_ceiling_beats_the_two_point_slope_it_replaces():
    """0.025 is no longer "half the band width", and the number did not move.

    Widening the band would have carried a half-the-width ceiling to 0.14 and
    made V6 pass on an estimator six times looser. The bar is instead the
    sharpest two-point sd in the corpus, which is what a three-point fit has to
    beat to have replaced anything.
    """
    sharpest = min(p.sd for p in BND.published_two_point_alpha_a())
    assert BND.ALPHA_A_SD_CEILING < sharpest
    assert BND.ALPHA_A_SD_CEILING < (BND.ALPHA_A_BAND[1]
                                     - BND.ALPHA_A_BAND[0]) / 2


# --------------------------------------------------------------------------
# THE INSTRUMENT AND ITS COLUMNS (audit A7), AND PROVENANCE (A5).
# --------------------------------------------------------------------------

def test_nothing_here_calls_the_retired_instrument():
    """`time_call` timed with per-iteration synchronises and no L2 flush.

    The roof every alpha here is scored against was measured queue-deep, so the
    two were never comparable: 0.18-0.30 ms of host enqueue inside the measured
    interval, a per-card bias of 8-16% in the fitted alpha at the smallest
    cells. The symbol still exists over in the sweep and refuses when called,
    which is why probing for it would have kept passing while meaning nothing.
    """
    source = (ROOT / "scripts" / "bn_decomposition.py").read_text()
    assert "SWEEP.time_call" not in source.replace(
        "The private `SWEEP.time_call` this used", "")
    assert "timing.time_kernel(" in source


def test_a_timed_row_keeps_every_instrument_column_through_the_csv(tmp_path):
    """cells.csv is what outlives the pod, so the instrument travels on the row.

    A row that carries `instrument` can be excluded by a later reader; a row
    that does not cannot, and the rows this file wrote before 2026-09-02 came
    from an instrument that is not comparable with the roof.
    """
    prov = BND.PV.provenance_block(instrument="queue-deep/test", iters=None)
    row = BND.Sample(64, 128, 3, 384, 1536, 1, 1.25, 1.2, 0.01, 512,
                     instrument="queue-deep/test", warmup_ms=300.0, trials=3,
                     l2_flush=True, sm_clock_load_mhz=1755.0,
                     clock_level_ok=True, clock_drift_ok=False,
                     host_bound=False)
    path = tmp_path / "cells.csv"
    BND.append_sample(path, row, prov)
    header = path.read_text().splitlines()[0].split(",")
    for column in ("instrument", "warmup_ms", "iters", "trials",
                   "sm_clock_load_mhz", "clock_level_ok", "clock_drift_ok",
                   "l2_flush"):
        assert column in header
    assert "prov_git_sha" in header and "prov_gpu_name" in header
    _, back = BND.read_samples(path)
    assert back[0].instrument == "queue-deep/test"
    assert back[0].warmup_ms == 300.0 and back[0].trials == 3
    assert back[0].clock_level_ok is True and back[0].clock_drift_ok is False
    assert back[0].l2_flush is True


def test_a_row_from_before_the_instrument_reads_back_as_absent(tmp_path):
    """Not as a default. Those rows were timed by the retired loop.

    "Flushed" and "level ok" are claims, and a resumed directory whose older
    half makes them by omission is worse than one that says nothing.
    """
    path = tmp_path / "old.csv"
    path.write_text("block_n,block_m,tiles,rows_per_expert,tokens,rep,ms_p50,"
                    "ms_min,ms_stdev,iters,status,detail\n"
                    "64,128,3,384,1536,1,1.25,1.2,0.01,50,ok,\n")
    _, back = BND.read_samples(path)
    assert back[0].instrument == ""
    assert back[0].clock_level_ok is None and back[0].clock_drift_ok is None
    assert back[0].l2_flush is False


def test_a_planted_row_never_claims_the_real_instrument():
    """"Not measured" is a value in the column, never an absence.

    `instrument` is one of the five keys a publish gate reads at the top of a
    report, and a planted row carrying TIMING_BASIS would satisfy that gate
    while describing an instrument no process ran.
    """
    args = args_for(capability="9.0", group_m=16, reps=2)
    rho = BND.achieved_rho(MIXTRAL, 2, alpha_b=0.61, alpha_a=0.14)
    samples = BND.planted_samples(
        MIXTRAL, args, alpha_b=0.61, alpha_a=0.14, ridge=rho,
        bandwidth_gbps=BND.PLANT_COMPUTE_FRACTION * 712.259 * 1e3 / rho, b=2,
        block_ns=(64,), subjects=(64,), noise=0.002, seed=0)
    assert {s.instrument for s in samples} == {SWEEP.SYNTHETIC_INSTRUMENT}
    assert SWEEP.SYNTHETIC_INSTRUMENT != BND.SWEEP.timing_basis()


def test_the_run_id_is_the_shared_one_and_refuses_a_missing_card():
    """Three scripts each re-implemented this rule and each left a knob out."""
    args = args_for(capability="9.0")
    assert BND.default_run_id(args, "NVIDIA H200").startswith("nvidia_h200-")
    with pytest.raises(BND.PV.NoCard):
        BND.default_run_id(args, "")


def test_the_resolved_plant_noise_never_enters_the_run_id():
    """It is derived from the cells the id names, so it would change mid-sweep.

    An id that depends on its own directory's contents is the resume collision
    this study has already paid for, arriving from the other direction.
    """
    args = args_for(capability="9.0")
    assert args.plant_noise is None
    # Two runs that will RESOLVE different noises -- one on a quiet pod, one on
    # a noisy one -- must still share a directory, or a resume would re-measure
    # every cell it already had.
    assert BND.default_run_id(args, "h200") == BND.default_run_id(
        args_for(capability="9.0"), "h200")
    # And the operator's own choice is a different experiment's directory.
    assert BND.default_run_id(args, "h200") != BND.default_run_id(
        args_for(capability="9.0", plant_noise=0.018), "h200")


def test_the_plan_prices_what_the_instrument_charges():
    """The old estimate multiplied a per-call time by a call count.

    `time_kernel` warms for a DURATION and runs `--trials` trials each sized to
    `--cell-budget-ms` of kernel time, so a timing costs the same wall clock
    whatever the kernel's own duration is, and the old one under-priced every
    fast cell.
    """
    args = args_for(capability="9.0", reps=3, trials=2, warmup=100.0,
                    cell_budget_ms=200.0)
    plan = BND.build_plan(args, MIXTRAL, 2, (9, 0), 160.3, 4374.5)
    per_timing_ms = 100.0 + 2 * 200.0
    assert plan.seconds == pytest.approx(
        plan.timings * per_timing_ms / 1e3)


# --------------------------------------------------------------------------
# THE EXIT CODES AND THE ONE GREPPABLE LINE (audit A4/A5's contract).
# --------------------------------------------------------------------------

def test_every_gate_prints_exactly_one_result_line_and_nothing_else_does(capsys):
    """The driver's summary once grepped free text and matched prose 18 times.

    So: one RESULT line per scored gate, none from anything that is not one, and
    the code the process returns recomputable from the log it printed.
    """
    code = BND.main(["--self-test", "--capability", "9.0", "--group-m", "16",
                     "--reps", "9", "--draws", "40", "--power-draws", "40",
                     "--plant-noise", "0.004", "--fail-on-gate"])
    out = capsys.readouterr().out
    parsed = exit_codes.parse_result_lines(out)
    assert [r.name for r in parsed] == ["S1", "S2", "S3", "S4", "S5"]
    assert out.count("RESULT: ") == len(parsed)
    assert exit_codes.classify_text(out) == code


def test_a_refusal_before_measuring_exits_refused_and_not_one(capsys):
    """`raise SystemExit("sentence")` exits 1, which is CLAIM_FAIL's code.

    A run that refused before measuring anything then looked to the driver like
    a measured run whose claim was refuted.
    """
    assert BND.main(["--tiles", "32,256", "--capability", "9.0"]) \
        == exit_codes.REFUSED
    assert "REFUSED" in capsys.readouterr().out


def test_a_claim_that_did_not_pass_is_a_result_without_fail_on_gate():
    """CLAIM_FAIL is softened to DONE without the flag; INVALID never is.

    C4 is PREDICTED to fail at GROUP_SIZE_M=1, and re-running until it passes is
    the failure mode the exit table is named against.
    """
    args = args_for(capability="9.0")
    claim_failed = [BND.Gate("VALIDITY", "V0 x", "", "", True, ""),
                    BND.Gate("CLAIM", "C4 x", "", "", False, "")]
    invalid = [BND.Gate("VALIDITY", "V0 x", "", "", False, ""),
               BND.Gate("CLAIM", "C4 x", "", "", True, "")]
    assert BND._exit_over(claim_failed, args) == exit_codes.DONE
    assert BND._exit_over(invalid, args) == exit_codes.INVALID
    strict = args_for(capability="9.0")
    strict.fail_on_gate = True
    assert BND._exit_over(claim_failed, strict) == exit_codes.CLAIM_FAIL


# --------------------------------------------------------------------------
# WHAT KIND OF NOISE THE INTERVALS ARE (B14, R5).
# --------------------------------------------------------------------------

MEASURED_SD = 0.0091


def _floor_doc(where, doc):
    """Each graft in its OWN directory under the real basename.

    Two grafts writing one `tmp_path / "NOISE_FLOOR.json"` means the second
    silently overwrites the first, which passes only for as long as nobody
    holds both at once.
    """
    where.mkdir(parents=True, exist_ok=True)
    path = where / "NOISE_FLOOR.json"
    path.write_text(json.dumps(doc))
    return path


def _measured_floor_file(tmp_path, *, sd=MEASURED_SD, synthetic=False):
    """The tracked document with a replicate floor grafted into it.

    Built FROM the tracked file so the schema, the declared prior and its
    source are the real ones, and the only thing under test is which of the two
    a caller reads.
    """
    doc = json.loads((PUBLISHED / "NOISE_FLOOR.json").read_text())
    doc["replicate_floor"] = {
        "n_replicates": 5, "cache_mode": "flush", "gpu_name": "NVIDIA H200",
        "provenance": "simulated part (a), planted by the test suite",
        "instrument": "queue-deep/l2-flush/clock-under-load/v2",
        "scope": None, "synthetic": synthetic,
        "per_field": {"alpha_corrected": {
            "sd": sd, "df": 8, "upper95": sd * 1.6, "pooled": True,
            "reason": "planted", "cells": 3, "per_cell": []}},
    }
    return _floor_doc(tmp_path / "measured", doc)


def _unmeasured_floor_file(tmp_path):
    """The tracked document with its replicate floor NULLED. The other graft.

    THE STATE, NOT THE CALENDAR, and this is the whole reason it exists. The
    ASSUMED assertions below used to be taken off the TRACKED file, which reads
    ASSUMED only because part (a) has not run on a card yet. Those assertions
    were therefore scheduled to go red at the exact moment the 120-minute arm
    this slice was written to serve published its result, and the owner would
    have met them on return from the pod. The declared prior is carried through
    unchanged, so what is asserted is still the repo's real number.
    """
    doc = json.loads((PUBLISHED / "NOISE_FLOOR.json").read_text())
    doc["replicate_floor"] = None
    return _floor_doc(tmp_path / "unmeasured", doc)


def test_the_report_states_the_bootstrap_scope_and_carries_the_floor(
        tmp_path, monkeypatch):
    """Two different kinds of noise, and only one of them is in the bootstrap.

    These resamples are of within-process warm repeats: one process, one
    allocation, one clock state. Every cross-arm difference this study publishes
    is a between-process comparison, so an interval from here is a LOWER bound
    on the uncertainty of one, and the floor carried beside it is the only
    reading of the other kind the repo has.

    The floor is PLANTED unmeasured rather than read off the tracked file: the
    declared number this asserts is pinned, but the tracked file's BASIS is not,
    and asserting the basis off it schedules this test to fail on the day part
    (a) publishes.
    """
    monkeypatch.setattr(BND, "NOISE_FLOOR_PATH", _unmeasured_floor_file(tmp_path))
    lines, _, payload = _planted_run(16, noise=0.004, probe=False)
    text = "\n".join(lines)
    assert "WITHIN-PROCESS WARM REPEATS" in text
    assert "cross-arm floor" in text
    assert payload["bootstrap"]["cross_arm_prior_sd"] == pytest.approx(
        BND.PUBLISHED_ALPHA_SD, abs=1e-4)
    assert "within-process" in payload["bootstrap"]["scope"]
    assert payload["bootstrap"]["mde_alpha_a"] == pytest.approx(
        BND.mde_one_sample(payload["bootstrap"]["alpha_a_sd"]))


def test_the_published_floor_is_read_from_the_file_not_quoted(tmp_path):
    """A literal here would silently stop describing a regenerated file."""
    floor = BND.published_prior_sd(_unmeasured_floor_file(tmp_path))
    assert floor.sd == pytest.approx(BND.PUBLISHED_ALPHA_SD, abs=1e-4)
    assert floor.basis == "ASSUMED"
    assert "s3-vs-s4" in floor.source


# --------------------------------------------------------------------------
# WHICH SIGMA, AND THE WORD THAT SAYS WHICH (2026-09-03).
#
# `published_prior_sd` read `payload["prior_sd"]` out of NOISE_FLOOR.json. That
# field is the s3/s4 PROXY and stays the proxy after part (a) spends 120
# minutes of card measuring a real between-replicate spread into
# `replicate_floor` of the SAME file, so every MDE and every detection-limit
# verdict this script printed would have gone on being scored against the
# assumption. Both branches are planted below: a measured floor must arrive
# MEASURED and move the MDE, an absent one must arrive ASSUMED at the declared
# number, and the pre-registered band must NOT move under either.
# --------------------------------------------------------------------------

def test_a_measured_floor_reaches_this_script(tmp_path):
    """The arm the owner is about to pay 120 minutes for buys this number."""
    floor = BND.published_prior_sd(_measured_floor_file(tmp_path))
    assert floor.basis == "MEASURED"
    assert floor.sd == pytest.approx(MEASURED_SD)
    assert "simulated part (a)" in floor.source


def test_a_rehearsal_floor_is_not_read_as_a_measurement(tmp_path):
    floor = BND.published_prior_sd(_measured_floor_file(tmp_path,
                                                        synthetic=True))
    assert floor.basis == "ASSUMED"
    assert floor.sd == pytest.approx(BND.PUBLISHED_ALPHA_SD, abs=1e-4)


def test_a_truncated_floor_file_refuses_rather_than_raising_a_decode_error(
        tmp_path):
    """`sizing_sigma` parses before its own guards run, so a half written file
    arrives as a ValueError. On the pod path that is a traceback where a
    REFUSED belongs."""
    path = tmp_path / "NOISE_FLOOR.json"
    path.write_text('{"schema": "moe-kernels/noise-floor/2", "prior_')
    with pytest.raises(BND.CorpusMissing):
        BND.published_prior_sd(path)


def test_the_report_prints_the_basis_beside_the_floor(tmp_path, monkeypatch):
    """An operator reads the report, not the JSON, and both states have to be
    distinguishable there."""
    monkeypatch.setattr(BND, "NOISE_FLOOR_PATH", _measured_floor_file(tmp_path))
    lines, _, payload = _planted_run(16, noise=0.004, probe=False)
    text = "\n".join(lines)
    assert "cross-arm floor 0.0091 MEASURED" in text
    assert "a between-replicate POINT ESTIMATE" in text
    assert payload["bootstrap"]["cross_arm_prior_sd_basis"] == "MEASURED"
    assert payload["bootstrap"]["cross_arm_prior_sd"] == pytest.approx(
        MEASURED_SD)

    monkeypatch.setattr(BND, "NOISE_FLOOR_PATH",
                        _unmeasured_floor_file(tmp_path))
    lines, _, payload = _planted_run(16, noise=0.004, probe=False)
    text = "\n".join(lines)
    assert "cross-arm floor 0.0229 ASSUMED" in text
    assert "the DECLARED upper one and not a measurement" in text
    assert payload["bootstrap"]["cross_arm_prior_sd_basis"] == "ASSUMED"


def test_a_measured_floor_is_not_called_an_upper_bound():
    """The word "upper" was only ever true of the s3/s4 proxy.

    That proxy confounds `num_stages` with rerun noise and so can only
    overstate; a between-replicate floor measures the quantity wanted and is a
    point estimate of it. Printing "a MEASURED upper one" produced a
    self-contradicting sentence on the page: under a planted 0.0091 floor,
    `--self-test --plant-noise 0.008` said "from sd 0.0098 ... it is a lower
    bound and the floor is a MEASURED upper one", the named upper number
    smaller than the named lower one in one clause.
    `alpha_surface.print_mde` is the sibling printer and never said it, so this
    was one of the two places that had to agree, fixed at one of them.
    """
    measured = BND.CrossArmFloor(0.0091, "MEASURED", "planted part (a)")
    clause = measured.clause()
    assert "upper" not in clause
    assert "POINT ESTIMATE" in clause
    assert "not a bound on it" in clause

    assumed = BND.CrossArmFloor(0.0229, "ASSUMED", "the s3-vs-s4 proxy")
    assert "the DECLARED upper one and not a measurement" in assumed.clause()


def test_the_band_lines_say_the_sds_are_the_pinned_declared_prior(tmp_path,
                                                                   monkeypatch):
    """The band was the ONE sigma-derived quantity on the report with no basis.

    `check_alpha_a_band` prints "alpha_a 0.286 +/- 0.086" and "the band is 0.28
    wide against input sds of 0.043-0.086", and every one of those sds is the
    DECLARED prior over `delta_s`. Under a measured floor the same report also
    prints "cross-arm floor 0.0091 MEASURED" fourteen lines below, and until
    this clause existed nothing on the page connected the two: an operator sees
    a 0.0091 measurement beside a 0.043 reading built from a sigma 2.5x larger
    and reads the C6 bar as stale.
    """
    for floor in (_unmeasured_floor_file(tmp_path),
                  _measured_floor_file(tmp_path)):
        monkeypatch.setattr(BND, "NOISE_FLOOR_PATH", floor)
        lines = "\n".join(BND.check_alpha_a_band()[1])
        assert "+/- 0.086" in lines and "input sds of 0.043-0.086" in lines
        declared, _ = BND.preregistration_sigma()
        assert f"the DECLARED prior {declared:.4f}" in lines
        assert "PINNED to the declared number" in lines
        # The sentence's arithmetic is the one the +/- was actually built by.
        for ds, sd in ((0.375, 0.086), (0.750, 0.043)):
            assert round(declared * math.sqrt(2) / ds, 3) == sd
    # And C6's own rule text, the second printer of the same pinned sd.
    boot = BND.Bootstrap(draws=100, per_cell_sd={}, survival=1.0,
                         alpha_a_sd=0.02, alpha_b_sd=0.01, delta_sd=0.01,
                         alpha_b_by_bm_sd={}, note="planted")
    rule = BND.gate_sharpness(boot).rule
    assert "0.043 sd of the sharpest two-point slope" in rule
    assert "DECLARED prior over the pair's ds and is PINNED there" in rule


def test_a_floor_that_cannot_be_read_is_named_and_not_defaulted(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(BND, "NOISE_FLOOR_PATH", tmp_path / "absent.json")
    lines, _, payload = _planted_run(16, noise=0.004, probe=False)
    text = "\n".join(lines)
    assert "cross-arm floor UNAVAILABLE" in text
    assert payload["bootstrap"]["cross_arm_prior_sd"] is None
    assert payload["bootstrap"]["cross_arm_prior_sd_basis"] == "UNAVAILABLE"


def test_the_plan_carries_the_basis_to_the_page(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(BND, "NOISE_FLOOR_PATH", _measured_floor_file(tmp_path))
    assert BND.main(["--dry-run", "--capability", "9.0", "--group-m", "16",
                     "--power-draws", "20"]) == exit_codes.DONE
    assert "cross-arm floor 0.0091 MEASURED" in capsys.readouterr().out


def test_the_pre_registered_band_does_not_move_when_the_floor_is_measured(
        tmp_path, monkeypatch):
    """THE REASON THERE ARE TWO ACCESSORS.

    The band is the two committed two-point slopes widened by a sigma, and
    `check_alpha_a_band` REFUSES when the literal and the re-derivation
    disagree. Widen them by a MEASURED floor of 0.0091 and the band re-derives
    as (0.12, 0.33): the first pod command after part (a) publishes would refuse
    with "the pre-registered alpha_a band is not what the committed BN pair now
    says", having been handed no new BN pair at all. A pre-registration that
    moves when new information arrives is not one, so the band stays on the
    DECLARED prior and only what this run can RESOLVE moves.
    """
    monkeypatch.setattr(BND, "NOISE_FLOOR_PATH", _measured_floor_file(tmp_path))
    assert BND.preregistration_sigma()[0] == pytest.approx(
        BND.PUBLISHED_ALPHA_SD, abs=1e-4)
    assert BND.alpha_a_band_from_published() == BND.ALPHA_A_BAND
    # And the sharpness ceiling, which is set INSIDE the sharpest two-point sd,
    # is still a bar the three-point fit has to beat rather than one a measured
    # floor moved under it.
    assert BND.ALPHA_A_SD_CEILING < min(
        p.sd for p in BND.published_two_point_alpha_a())
    assert BND.main(["--dry-run", "--capability", "9.0", "--group-m", "16",
                     "--power-draws", "20"]) == exit_codes.DONE


def test_the_band_still_refuses_when_the_declared_prior_moves(tmp_path,
                                                              monkeypatch):
    """The FAIL branch of the pinning above: pinned is not frozen. Edit the
    declared prior and the band must stop reading back."""
    doc = json.loads((PUBLISHED / "NOISE_FLOOR.json").read_text())
    doc["prior_sd"] = 0.2
    path = tmp_path / "NOISE_FLOOR.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(BND, "NOISE_FLOOR_PATH", path)
    with pytest.raises(BND.CorpusMissing):
        BND.check_alpha_a_band()
    assert BND.main(["--dry-run", "--capability", "9.0",
                     "--group-m", "16"]) == exit_codes.REFUSED


def test_help_renders(capsys):
    """argparse expands `%` in help text, so a formatted percentage must escape it.

    Caught live: `--plant-noise`'s help gained an f-string percentage while this
    slice was being written and `--help` died with "unsupported format
    character ')'". Nothing else in the suite runs the parser's renderer, and a
    script whose --help raises is a script nobody can find the flag in.
    """
    with pytest.raises(SystemExit) as exc:
        BND.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--plant-noise" in out and "--trials" in out


# --------------------------------------------------------------------------
# THE BAND'S PROVENANCE WHERE IT IS ACTUALLY SCORED (review of finding 32).
# The plan output was corrected and the GATE was not, so `--dry-run` printed
# the two committed reports while report.txt, report.json and the RESULT line's
# context went on naming four A100 slopes that are in no file. Everything below
# scores a planted run rather than grepping a plan, because a plan scores no
# gates and that is exactly how the surviving copy went untested.
# --------------------------------------------------------------------------

def _break_the_corpus(monkeypatch):
    """Make the committed BN pair unreadable, through the REAL refusal path.

    `published_two_point_alpha_a` binds `PUBLISHED_BN_PAIR` as a default at
    definition time, so patching the constant would silently do nothing and the
    test would pass by not testing. This re-points the function at two paths
    that do not exist, so what the callers see is the refusal the real function
    raises with the real message.
    """
    real = BND.published_two_point_alpha_a
    monkeypatch.setattr(
        BND, "published_two_point_alpha_a",
        lambda *a, **k: real((ROOT / "nope-a.json", ROOT / "nope-b.json")))


def test_the_scored_c1_gate_carries_the_committed_provenance():
    """The gate, not the plan: what report.txt and report.json actually say."""
    _, gates, payload = _planted_run(16, noise=0.004)
    c1 = gates["C1"]
    printed = "\n".join(c1.render())
    assert "d66ad3.report.json" in printed and "16cc16.report.json" in printed
    assert "alpha_a band [0.10, 0.38]" in printed
    for phantom in ("0.106", "0.102", "0.129", "0.119", "ai_model.py's 0.143"):
        assert phantom not in printed, f"{phantom} is back in the gate"
    # And the same text is what leaves the pod in the JSON.
    blob = json.dumps(payload, default=str)
    for phantom in ("0.106", "0.102", "0.129", "0.119"):
        assert phantom not in blob, f"{phantom} reached report.json"


def test_c1_prints_the_band_lines_it_is_handed():
    """`analyse_run` threads `check_alpha_a_band`'s lines down to the gate.

    The band is re-derived once, before any GPU time, and the gate prints THAT
    derivation rather than a second one: two copies of a provenance is how the
    corrected one and the stale one ended up in the same report.
    """
    _, band_lines = BND.check_alpha_a_band()
    empty = BND.Decomposition("EXA", None, None, None, None, 0, 2, (), (), (),
                              "nothing")
    boot = BND.Bootstrap(0, {}, {}, None, None, None, {}, "no draws")
    gate = BND.gate_alpha_a(empty, boot, sharp=False, band_lines=band_lines)
    assert gate.lines[:len(band_lines)] == band_lines


def test_the_gate_refuses_in_words_when_the_band_cannot_be_re_read(monkeypatch):
    """The FAIL branch: a corpus that no longer reads back.

    Scoring happens after the pod time is spent, so the gate says so on the page
    instead of raising the report away -- and it must never fall back to a
    remembered sentence, which is the whole finding.
    """
    _break_the_corpus(monkeypatch)
    lines = BND.band_provenance_lines()
    assert len(lines) == 1 and lines[0].startswith("BAND PROVENANCE UNREADABLE")
    empty = BND.Decomposition("EXA", None, None, None, None, 0, 2, (), (), (),
                              "nothing")
    boot = BND.Bootstrap(0, {}, {}, None, None, None, {}, "no draws")
    gate = BND.gate_alpha_a(empty, boot, sharp=False)
    assert any("BAND PROVENANCE UNREADABLE" in ln for ln in gate.lines)
    assert not any("0.106" in ln for ln in gate.lines)


def test_c1_says_why_it_read_unknown_when_the_estimator_is_not_sharp():
    """UNKNOWN with a fitted number beside it is otherwise unreadable."""
    fit = BND.Decomposition("EXA", None, 0.9, 0.2, 0.0, 4, 3, (0.001,),
                            ("BN=64 BM=128",), (0.5,), "planted")
    boot = BND.Bootstrap(10, {}, {}, 0.4, 0.01, None, {}, "planted")
    gate = BND.gate_alpha_a(fit, boot, sharp=False)
    assert gate.passed is None and "C6" in gate.observed
    assert BND.gate_alpha_a(fit, boot, sharp=True).passed is True


# --------------------------------------------------------------------------
# C6, THE SHARPNESS GATE: THE RULE IT STATES AND THE KIND IT IS.
# --------------------------------------------------------------------------

def test_the_sharpness_rule_states_the_derivation_the_ceiling_actually_has():
    """"Half the width of the band" is arithmetically false and was printed.

    Half of the [0.10, 0.38] band is 0.14, five times the 0.025 ceiling, and the
    constant's own re-justification says the bar is the sharpest two-point sd in
    the corpus instead. The rule string is what every report quotes, so it is
    read from the corpus rather than written down.
    """
    boot = BND.Bootstrap(10, {}, {}, 0.01, 0.01, None, {}, "planted")
    gate = BND.gate_sharpness(boot)
    sharpest, source = BND.sharpest_two_point_sd()
    assert sharpest == pytest.approx(
        min(p.sd for p in BND.published_two_point_alpha_a()))
    assert f"{sharpest:.3f}" in gate.rule and "d66ad3" in "".join(gate.lines)
    assert "half the width" not in gate.rule.lower()
    assert source in "".join(gate.lines)


def test_the_sharpness_rule_says_so_when_the_corpus_cannot_be_read(monkeypatch):
    """The FAIL branch of the bar's provenance: no bar quoted from memory."""
    _break_the_corpus(monkeypatch)
    sharpest, why = BND.sharpest_two_point_sd()
    assert sharpest is None and why
    gate = BND.gate_sharpness(
        BND.Bootstrap(10, {}, {}, 0.30, 0.01, None, {}, "planted"))
    assert "cannot be read here" in gate.rule
    assert gate.passed is False          # it still scores; only the bar's
    assert "0.043" not in gate.rule      # provenance is missing


def test_a_pinning_that_cannot_resolve_alpha_a_is_a_result_not_a_broken_run():
    """C6 is a CLAIM gate, and a G=1 arm therefore ends CLAIM_FAIL, not INVALID.

    The shortfall is PREDICTED at GROUP_SIZE_M=1 and printed in the plan before
    the pod is rented. As a VALIDITY gate it made `classify` return 3 INVALID --
    nothing on the page quotable -- for a run whose alpha_b, C3 and C5 are
    exactly what that pinning is for, and the session driver re-measured the arm
    on every pass because 3 is not one of its finished codes.
    """
    _, gates, _ = _planted_run(1, noise=0.008)
    c6 = gates["C6"]
    assert c6.kind == "CLAIM" and c6.passed is False
    assert c6.result_line().startswith("RESULT: CLAIM C6 FAIL")
    assert "C1 ALONE" in c6.invalidates
    assert all(g.passed is True for g in gates.values() if g.kind == "VALIDITY")
    assert exit_codes.classify(g.scored() for g in gates.values()) \
        == exit_codes.CLAIM_FAIL
    args = args_for(capability="9.0")
    assert BND._exit_over(list(gates.values()), args) == exit_codes.DONE
    strict = args_for(capability="9.0")
    strict.fail_on_gate = True
    assert BND._exit_over(list(gates.values()), strict) == exit_codes.CLAIM_FAIL


def test_a_real_validity_failure_at_the_same_pinning_is_still_invalid():
    """The FAIL branch: the softening reaches CLAIM_FAIL and nothing else.

    Planted onto the same G=1 gate list, one broken VALIDITY gate still takes
    the arm to 3 with the flag off, which is what stops this change from being
    a way of exiting 0 whatever happened.
    """
    _, gates, _ = _planted_run(1, noise=0.008)
    broken = list(gates.values()) + [
        BND.Gate("VALIDITY", "V0 planted", "", "", False, "planted failure")]
    assert exit_codes.classify(g.scored() for g in broken) == exit_codes.INVALID
    assert BND._exit_over(broken, args_for(capability="9.0")) \
        == exit_codes.INVALID


def test_the_plan_says_which_exit_code_a_g1_arm_is_expected_to_return(capsys):
    """Predicted before the pod, in the same line that predicts the shortfall."""
    BND.main(["--dry-run", "--capability", "9.0", "--group-m", "1",
              "--power-draws", "20", "--plant-noise", "0.008"])
    out = capsys.readouterr().out
    assert "CANNOT RESOLVE alpha_a" in out
    assert "1 CLAIM_FAIL" in out and "NOT 3 INVALID" in out


# --------------------------------------------------------------------------
# WHAT AN UNDECIDABLE CLAIM ACTUALLY MAKES THE PROCESS RETURN.
# The C2 guard's docstrings said an arm whose C2 has no power "exits
# CLAIM_FAIL"; `_exit_over` reports CLAIM_FAIL as DONE unless --fail-on-gate,
# and the session driver passes neither. The verdict travels on the RESULT
# line, not in the exit code, and that is what this pins.
# --------------------------------------------------------------------------

def test_a_powerless_c2_travels_on_the_result_line_not_the_exit_code():
    _, gates, _ = _planted_run(1, noise=0.008)
    assert gates["C2"].passed is None
    assert gates["C2"].result_line().startswith("RESULT: CLAIM C2 UNKNOWN")
    assert exit_codes.classify(g.scored() for g in gates.values()) \
        == exit_codes.CLAIM_FAIL
    default = args_for(capability="9.0")
    assert BND._exit_over(list(gates.values()), default) == exit_codes.DONE
    strict = args_for(capability="9.0")
    strict.fail_on_gate = True
    assert BND._exit_over(list(gates.values()), strict) == exit_codes.CLAIM_FAIL


def test_an_unplanned_crash_exits_ERROR_and_never_CLAIM_FAIL(monkeypatch, capsys):
    """The apparatus breaking must not be filed as one of the arm's outcomes.

    An exception left to propagate exits the interpreter ONE, and ONE is
    CLAIM_FAIL, which `moe/bench/exit_codes.py` puts in FINISHED_CODES: the
    driver records the arm as finished, skips it on every resume, leaves
    RETRY_ARMS at zero and exits the session 0 over an arm that never measured.
    ERROR (4) is outside FINISHED_CODES so the two can be told apart, and the
    traceback is printed rather than swallowed because a bare code names
    nothing to fix. Planted rather than argued: bn_decomposition had no top-level handler
    until 2026-09-02.
    """
    from moe.bench import exit_codes as EX

    def explode(argv=None):
        raise RuntimeError("planted: the allocator gave up halfway")

    monkeypatch.setattr(BND, "_main", explode)
    code = BND.main([])
    err = capsys.readouterr().err
    assert code == EX.ERROR
    assert code != EX.CLAIM_FAIL
    assert code not in EX.FINISHED_CODES, "an apparatus failure must stay retryable"
    assert EX.ledger_state(code) == "RETRY"
    assert "planted: the allocator gave up halfway" in err, \
        "the traceback was swallowed"
    assert "RuntimeError" in err


def test_a_string_refusal_still_exits_REFUSED_and_is_not_relabelled_ERROR(
        monkeypatch, capsys):
    """The other branch of the same handler. `raise SystemExit("sentence")` is a
    precondition not met, which is free and distinct from a crash, so the new
    `except Exception` must not catch it: `SystemExit` is a `BaseException`.
    """
    from moe.bench import exit_codes as EX

    def refuse(argv=None):
        raise SystemExit("no calibration for the attached device")

    monkeypatch.setattr(BND, "_main", refuse)
    code = BND.main([])
    err = capsys.readouterr().err
    assert code == EX.REFUSED
    assert err.startswith("REFUSED: no calibration")
    assert "Traceback" not in err


# --------------------------------------------------------------------------
# retraction (c): 0.307 and the TEMPO 2-4% match are withdrawn, and the
# printed plan and the C4 gate must say so rather than restate them
# --------------------------------------------------------------------------

def test_p4_and_c4_state_the_tempo_match_as_withdrawn_not_as_prior_agreement():
    """The printed plan carried "the study's decomposed 0.307 corroborates
    TEMPO's b2/b to 2-4%" and C4 explained 0.307 as "a POOLED refit". Both
    are the retracted claim restated: 0.307 was a (LIN) unit artefact of the
    estimator, not a pooled measurement, and the 2-4% agreement went with it.
    Pinned on the rendered text of BOTH sites, because the defect this repo
    keeps finding is a fix applied at one of two call sites.
    """
    plan = BND.predictions_text(MIXTRAL, 2, 160.3, "test", BND.DEFAULT_BLOCK_N,
                                BND.SUBJECT_BLOCK_M, {}, 1, [])
    p4 = plan[plan.index("P4"):plan.index("P5")]
    assert "WITHDRAWN" in p4 and "(LIN)" in p4
    assert "corroborates TEMPO's b2/b of" not in p4
    assert "to 2-4%" not in p4.replace("corroborates TEMPO to 2-4%' is WITHDRAWN", "")

    boot = BND.Bootstrap(0, {}, {}, None, None, None, {}, "no draws")
    fit = BND.Decomposition("EXA", None, 0.92, 0.14, 0.0, 9, 3, (), (), (),
                            "planted")
    gate = BND.gate_tempo(fit, boot, 1)
    text = "\n".join(gate.render()) if hasattr(gate, "render") else "\n".join(gate.lines)
    assert gate.passed is False                      # 0.92 is far from 0.311/0.319
    assert "WITHDRAWN" in text and "unit artefact" in text
    assert "is a POOLED refit" not in text
    assert "first number in this study that may be compared with TEMPO" in text
    # The fitted-world table names the withdrawn reading by what it is.
    table = "\n".join(BND.cell_table(MIXTRAL, 2, 160.3, BND.DEFAULT_BLOCK_N,
                                     BND.SUBJECT_BLOCK_M, {}))
    assert "LIN-ERA (0.307, 0.143)" in table and "POOLED (0.307" not in table
