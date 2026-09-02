"""Occupancy against swizzle: does the standard cache predictor transfer here?

`scripts/occupancy_vs_swizzle.py` asks whether the weight re-read fraction is
set by CONCURRENCY -- how many thread blocks are resident, and therefore how
much data is in flight through one L2 -- or by PROGRAM ORDER, which is what
reuse-distance analysis assumes and what `GROUP_SIZE_M` controls. Most of this
file is about the two things that could make that question unanswerable rather
than about the answer.

FIVE GROUPS.

  - THE X AXIS. Resident blocks per SM is COMPUTED, and if it is computed wrong
    the experiment sweeps an axis that does not exist. The ladder is pinned here
    against hand-worked numbers on both cards, the control property that makes
    num_warps a control (it must NOT move residency) is asserted rather than
    assumed, and an unknown card is checked to REFUSE instead of defaulting.
  - THE TWO MODELS. The concurrent footprint, the LRU caricature and the capped
    reuse-distance prediction, each checked against arithmetic done by hand.
  - THE WINDOW. `identifiability_window` says which alphas this ladder can read
    at all, and it is checked against a brute-force count of memory-bound treads
    on ladders planted from the law rather than against its own formula. It has
    a DEAD BAND, and a prediction landing inside it is unmeasurable however
    clean the timings are; that is a property of the design and it is pinned.
  - DISCRIMINATION. Three planted worlds must produce three different verdicts.
    A pipeline that answers CONCURRENCY in a world built from GROUP_SIZE_M alone
    cannot settle anything, and neither can one that answers NEITHER in both.
  - THE REFUSALS. A run id that omits a swept knob overwrote a whole arm in this
    repo once; a residency ladder that did not move would report a flat alpha as
    evidence; a warp control compared against an occupancy effect that is itself
    inside the noise passes half the time. Each is checked to fail loudly.

The script is loaded by path, because `scripts/` is not a package.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PUBLISHED = ROOT / "results" / "published"


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


OVS = _load("occupancy_vs_swizzle", "occupancy_vs_swizzle.py")
SWEEP = OVS.SWEEP

H200 = (9, 0)
A100 = (8, 0)


def _limits(capability=H200, sm_count=132, l2=50_000_000):
    return OVS.card_limits(capability, sm_count, l2, "test")


def _pinned(stages: int, warps: int = 8, group: int = 1):
    return OVS.Setting(stages, warps, group, (OVS.ARM_OCCUPANCY,)).pinned(64, 64)


# --------------------------------------------------------------------------
# The x axis. If this is wrong the experiment sweeps nothing.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("stages", "expected"),
                         [(2, 6), (3, 4), (4, 3), (5, 2)])
def test_h200_residency_ladder(stages, expected):
    """228 KiB per SM over `stages x 16 KiB + 1 KiB` reserved, worked by hand.

    One CTA at BM=BN=BK=64 in bf16 holds `stages (64x64 + 64x64) x 2` = 16 KiB
    per stage, plus the driver's 1 KiB per resident block. 233472 // 33792 = 6,
    // 50176 = 4, // 66560 = 3, // 82944 = 2.
    """
    res = OVS.residency(_pinned(stages), 64, 2, _limits())
    assert res.resident_blocks == expected
    assert res.smem_per_block == stages * 16384
    assert res.binding == "smem"


@pytest.mark.parametrize(("stages", "expected"),
                         [(2, 4), (3, 3), (4, 2), (5, 2)])
def test_a100_residency_ladder(stages, expected):
    """164 KiB per SM gives a shorter ladder, and the script must use ITS card."""
    res = OVS.residency(_pinned(stages), 64, 2, _limits(A100, 108, 40_000_000))
    assert res.resident_blocks == expected


def test_residency_ladder_spans_the_gate_on_both_cards():
    """V2's own bar, checked on the default grid rather than on one setting.

    A grid that computes the same residency at every setting has swept nothing,
    and its flat alpha would read as evidence for program order. The default
    stages must clear MIN_RESIDENCY_LEVELS and MIN_RESIDENCY_SPAN on BOTH cards
    this study owns, or the pod session picks a grid that cannot answer.
    """
    for cap, sms, l2 in ((H200, 132, 50_000_000), (A100, 108, 40_000_000)):
        levels = sorted({OVS.residency(_pinned(s), 64, 2,
                                       _limits(cap, sms, l2)).resident_blocks
                         for s in OVS.DEFAULT_STAGES})
        assert len(levels) >= OVS.MIN_RESIDENCY_LEVELS, (cap, levels)
        assert max(levels) / min(levels) >= OVS.MIN_RESIDENCY_SPAN, (cap, levels)


@pytest.mark.parametrize("stages", OVS.CONTROL_WARP_STAGES)
def test_num_warps_does_not_move_residency(stages):
    """THE CONTROL PROPERTY, and the whole design rests on it.

    num_warps is offered as a knob that changes resident WARPS at fixed
    resident BLOCKS, so the concurrent data footprint is unchanged and P1's
    mechanism predicts no movement. That is only true while the shared-memory
    limit binds before the thread-slot limit. If it ever stops being true the
    control silently becomes a second treatment and P3 stops being a control.
    """
    base = OVS.residency(_pinned(stages, OVS.BASE_WARPS), 64, 2, _limits())
    for warps in OVS.DEFAULT_CONTROL_WARPS:
        other = OVS.residency(_pinned(stages, warps), 64, 2, _limits())
        assert other.resident_blocks == base.resident_blocks
        assert other.binding == "smem"


def test_group_size_m_does_not_move_residency():
    """The swizzle arm has to hold occupancy fixed, or it is not a second axis."""
    base = OVS.residency(_pinned(3, 8, 1), 64, 2, _limits())
    for group in OVS.DEFAULT_GROUPS:
        assert OVS.residency(_pinned(3, 8, group), 64, 2,
                             _limits()).resident_blocks == base.resident_blocks


def test_unknown_capability_refuses_rather_than_defaulting():
    """Residency is the swept axis; a guessed per-SM limit invents the x axis."""
    with pytest.raises(OVS.CardUnavailable) as exc:
        _limits(capability=(6, 1))
    assert "sm_61" in str(exc.value)
    assert "not defaulted" in str(exc.value).lower()


@pytest.mark.parametrize(("kwargs", "needle"), [
    ({"capability": None}, "no compute capability"),
    ({"sm_count": 0}, "no SM count"),
    ({"l2": 0}, "no L2 capacity"),
])
def test_missing_card_facts_refuse(kwargs, needle):
    with pytest.raises(OVS.CardUnavailable) as exc:
        _limits(**kwargs)
    assert needle in str(exc.value)


def test_per_sm_and_per_block_tables_are_not_the_same_question():
    """The two shared-memory tables answer different questions and must differ.

    `SMEM_PER_BLOCK_BYTES` (imported) asks whether ONE CTA fits and is the
    per-block opt-in ceiling; `SMEM_PER_SM_BYTES` asks how MANY fit. Confusing
    them silently changes the residency, so every capability this file knows is
    checked to have a per-SM figure at least as large as its per-block one.
    """
    for cap, per_sm in OVS.SMEM_PER_SM_BYTES.items():
        per_block = SWEEP.SMEM_PER_BLOCK_BYTES.get(cap)
        assert per_block is not None, cap
        assert per_sm >= per_block, cap
        assert cap in OVS.MAX_THREADS_PER_SM and cap in OVS.MAX_BLOCKS_PER_SM


# --------------------------------------------------------------------------
# The two models.
# --------------------------------------------------------------------------

def test_per_cta_stream_is_weighted_over_both_gemms():
    """1.31 MiB at mixtral, and the weighting is what makes it that and not 1.

    The up GEMM streams `(BM+BN) H b` = 1 MiB per CTA over `2F/BN` = 448 N-tiles;
    the down GEMM streams `(BM+BN) F b` = 3.5 MiB over `H/BN` = 64. Quoting only
    the first understates the concurrent footprint by 31%.
    """
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    up = (64 + 64) * cfg.hidden_size * 2
    down = (64 + 64) * cfg.intermediate_size * 2
    n_up = math.ceil(2 * cfg.intermediate_size / 64)
    n_down = math.ceil(cfg.hidden_size / 64)
    want = (n_up * up + n_down * down) / (n_up + n_down)
    assert OVS.per_cta_stream_bytes(cfg, 64, 64, 2) == pytest.approx(want)
    assert up < want < down


def test_footprint_is_linear_in_residency():
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    one = OVS.concurrent_footprint_bytes(cfg, 64, 64, 2, 1, 132)
    assert OVS.concurrent_footprint_bytes(cfg, 64, 64, 2, 4, 132) == \
        pytest.approx(4 * one)
    # The sentence the module docstring makes: one resident block per SM already
    # puts more in flight than an H200's L2 holds.
    assert one > 50_000_000


@pytest.mark.parametrize(("footprint", "expected"),
                         [(0.0, 1.0), (10.0, 0.0), (100.0, 0.9), (50.0, 0.8)])
def test_alpha_concurrency_is_clamped_and_monotone(footprint, expected):
    assert OVS.alpha_concurrency(footprint, 10.0) == pytest.approx(expected)


def test_alpha_order_caps_at_tiles_per_expert():
    """Two M-tiles of different experts share no weight block, so the cap is
    the mechanism and not a concession."""
    assert OVS.alpha_order(1.0, 64, 1000.0) == pytest.approx(1 / 64)
    assert OVS.alpha_order(1.0, 64, 8.5) == pytest.approx(1 / 8.5)
    assert OVS.alpha_order(1.0, 1, 8.5) == pytest.approx(1.0)


def test_uncapped_reuse_distance_is_the_forty_fold_gap():
    """The number the experiment starts from: 1/64 against a measured ~0.67."""
    assert OVS.alpha_order(1.0, 64, 1e9) == pytest.approx(0.015625)
    assert 0.67 / OVS.alpha_order(1.0, 64, 1e9) > 40


# --------------------------------------------------------------------------
# The window: which alphas this ladder can read at all.
# --------------------------------------------------------------------------

def _planted(alpha: float, k: float, treads: int, load: float = 4.0,
             overhead: float = 0.5):
    """`t(n) = D + max(L(1 + alpha(n-1)), C n)` with `B/C` set to exactly `k alpha`.

    `C = L alpha / (k alpha) = L / k` and does not depend on alpha, which is the
    two-line model written so that only the memory branch moves.
    """
    c = load / k
    return c, [(n, overhead + max(load * (1.0 + alpha * (n - 1)), c * n))
               for n in range(1, treads + 1)]


def _memory_treads(alpha: float, k: float, treads: int) -> int:
    """Brute force: how many treads stand above the compute branch by the margin.

    Counted from planted times against the study's own membership rule, never
    from `identifiability_window`'s formula, so the formula is checked against
    something independent of it.
    """
    overhead = 0.5
    c, pts = _planted(alpha, k, treads)
    k_count = 0
    for n, ms in pts:
        if ms <= overhead + c * n * (1.0 + SWEEP.MEMORY_BRANCH_MARGIN):
            break
        k_count += 1
    return k_count


def test_window_floor_matches_a_brute_force_tread_count():
    """The floor is where the memory prefix drops below MIN_MEMORY_TREADS."""
    ridge, b, bm, treads = 162.8, 2, 64, 16
    floor, _, _ = OVS.identifiability_window(ridge, b, bm, treads)
    k = b * ridge / (2 * bm)
    assert _memory_treads(floor * 1.05, k, treads) >= SWEEP.MIN_MEMORY_TREADS
    assert _memory_treads(floor * 0.95, k, treads) < SWEEP.MIN_MEMORY_TREADS


def test_window_dead_band_is_where_the_fit_discards_the_memory_branch():
    """Inside the band the two branches are one line and `fit_ladder` says so.

    This is the same rejection that makes BLOCK_M=128 useless for this
    question, arriving at a different alpha instead of at a different tile. A
    prediction landing in here is unmeasurable HERE however clean the timings.
    """
    ridge, b, bm, treads = 162.8, 2, 64, 16
    _, lo, hi = OVS.identifiability_window(ridge, b, bm, treads)
    k = b * ridge / (2 * bm)
    mid = 0.5 * (lo + hi)
    for alpha, discarded in ((mid, True), (lo * 0.8, False), (hi * 1.3, False)):
        c, pts = _planted(alpha, k, treads)
        ref = SWEEP.ComputeReference(256, 0.5, c * 256 / bm, 0.0, "planted")
        fit = SWEEP.fit_ladder(pts, bm, ref)
        assert ("DISCARDED" in fit.basis) is discarded, (alpha, fit.basis)


def test_window_is_pinned_on_the_h200():
    """The three numbers the report prints, so a change to the fit is visible."""
    floor, lo, hi = OVS.identifiability_window(162.8, 2, 64, 16)
    assert floor == pytest.approx(0.1015, abs=5e-4)
    assert (lo, hi) == (pytest.approx(0.334, abs=5e-4),
                        pytest.approx(0.452, abs=5e-4))


def test_reuse_distance_prediction_clears_the_dead_band():
    """P2 can be settled by the RATIO on this design, and that has to be checked.

    If the capped reuse-distance prediction landed inside the dead band, route
    one of P2 would be unmeasurable and the whole swizzle arm would rest on the
    collapse route. It does not -- 0.118 x 0.93 = 0.11 sits under the band and
    over the floor -- and this test is what would notice if a change to the
    ladder depth or the ridge moved it in.
    """
    floor, lo, _ = OVS.identifiability_window(162.8, 2, 64, 16)
    predicted = OVS.alpha_order(OVS.CORPUS_ALPHA_AT_G1, 64, 8.5)
    assert floor < predicted < lo


def test_k_of_window_inverts_the_band_exactly():
    window = OVS.identifiability_window(162.8, 2, 64, 16)
    assert OVS.k_of_window(window) == pytest.approx(2 * 162.8 / (2 * 64))


# --------------------------------------------------------------------------
# Discrimination: three planted worlds, three verdicts.
# --------------------------------------------------------------------------

def _plan_and_reg(reps: int = 2, ridge: float = 162.8, bw: float = 4374.5):
    from moe.spec import MODEL_CONFIGS, dtype_bytes
    args = OVS.build_parser().parse_args(
        ["--capability", "9.0", "--sm-count", "132", "--l2-bytes", "50000000",
         "--reps", str(reps)])
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    limits = _limits()
    plan = OVS.build_plan(args, cfg, b, limits, alpha=0.9, ridge=ridge,
                          bandwidth_gbps=bw)
    reg = OVS.register(cfg, plan.settings, limits, block_n=args.block_n,
                       block_k=args.block_k, b=b,
                       subject_rows=plan.subject_rows, ridge=ridge,
                       l2_source="test")
    return args, cfg, b, plan, reg, limits


def test_three_planted_worlds_give_three_verdicts():
    """THE CLAIM OF THIS FILE. A pipeline that answers the same in every world
    cannot settle the experiment, and the whole real code path runs here:
    planting, the per-setting compute reference, the ladder fit, the contrasts,
    the gates and the verdict."""
    _, cfg, b, plan, reg, _ = _plan_and_reg()
    lines, gates = OVS.self_test(cfg, plan, reg, b, ridge=162.8,
                                 bandwidth_gbps=4374.5, noise=0.004, seed=3)
    assert [g.passed for g in gates] == [True, True, True], "\n".join(lines)
    assert len({g.observed for g in gates}) == 3


def test_concurrency_world_recovers_the_predicted_direction_and_size():
    """Not merely 'a swing': the right sign, the right order, the right size."""
    _, cfg, b, plan, reg, _ = _plan_and_reg()
    samples = OVS.planted_samples(
        cfg, plan, lambda st: reg.concurrency_alpha[st.key], ridge=162.8,
        bandwidth_gbps=4374.5, b=b, noise=0.003, seed=11)
    _, _, payload = OVS.analyse(
        samples, cfg, plan, reg, b, ridge=162.8, bandwidth_gbps=4374.5,
        compiles={s.key: 1 for s in plan.settings},
        executed={s.key: plan.reps for s in plan.settings},
        l2_source="test", measured=False)
    occ = payload["contrasts"]["occupancy"]
    assert occ["swing"] > reg.occupancy_threshold
    assert payload["rising_steps"][0] == payload["rising_steps"][1]
    # The fit reads alpha low by the fixed cost it cannot separate, so the
    # recovered swing is a fraction of the planted one -- but a fraction, not a
    # different number. Bounding it both ways is what makes this a check.
    assert 0.5 * reg.occupancy_swing < occ["swing"] < 1.2 * reg.occupancy_swing


def test_swizzle_world_is_not_called_concurrency():
    """The confusion that would matter most: a pure GROUP_SIZE_M effect must
    not arrive as an occupancy result."""
    _, cfg, b, plan, reg, _ = _plan_and_reg()
    samples = OVS.planted_samples(
        cfg, plan,
        lambda st: OVS.alpha_order(0.95, st.group_m, reg.order_cap),
        ridge=162.8, bandwidth_gbps=4374.5, b=b, noise=0.003, seed=13)
    _, _, payload = OVS.analyse(
        samples, cfg, plan, reg, b, ridge=162.8, bandwidth_gbps=4374.5,
        compiles={s.key: 1 for s in plan.settings},
        executed={s.key: plan.reps for s in plan.settings},
        l2_source="test", measured=False)
    assert payload["verdict"] == OVS.VERDICT_ORDER
    assert payload["contrasts"]["occupancy"]["swing"] < reg.occupancy_threshold


# --------------------------------------------------------------------------
# Gates that must be able to say no.
# --------------------------------------------------------------------------

def _result(key: str, stages: int, warps: int, group: int, blocks: int,
            alpha: float | None, arms, sigma: float | None = 0.002,
            memory_points: int = 8, refused: bool = False):
    st = OVS.Setting(stages, warps, group, tuple(arms))
    res = OVS.Residency(blocks, blocks, 16, 32, stages * 16384, "smem")
    return OVS.SettingResult(
        setting=st, residency=res, footprint_bytes=1e9, predicted_alpha=0.9,
        points=[(n, 1.0 * n) for n in range(1, 9)], reference_points=[],
        memory_points=memory_points, alpha=alpha, alpha_corrected=alpha,
        alpha_sigma=sigma, per_rep_alpha=[alpha] if alpha else [],
        spread=0.001, mean_rel_err=0.01, reference_note="planted",
        reference_block_m=None if refused else 256, reference_refused=refused,
        basis="planted")


def test_residency_gate_fails_when_the_ladder_did_not_move():
    """A grid that computes one residency everywhere has swept nothing, and its
    flat alpha would read as evidence for program order."""
    _, cfg, b, plan, reg, _ = _plan_and_reg()
    flat = [_result(f"s3w8g{g}", 3, 8, g, 4, 0.9, (OVS.ARM_OCCUPANCY,))
            for g in (1, 8, 16)]
    gate = OVS.gate_residency_swept(reg, flat)
    assert gate.passed is False
    assert gate.kind == OVS.VALIDITY and gate.invalidates


def test_warp_control_is_unknown_when_the_occupancy_swing_is_noise():
    """Comparing two noise measurements passes about half the time, so it must
    not be allowed to pass at all."""
    occ = OVS.Contrast("occupancy", "2", "6", 0.900, 0.902, 0.002, 3)
    warp = OVS.Contrast("warp-control", "w4", "w8", 0.900, 0.901, 0.002, 1)
    assert OVS.gate_warp_control(warp, occ, 0.05).passed is None
    big = OVS.Contrast("occupancy", "2", "6", 0.80, 0.90, 0.002, 3)
    assert OVS.gate_warp_control(warp, big, 0.05).passed is True


def test_occupancy_threshold_is_raised_by_noise_and_never_lowered():
    """`_floor` may only make the claim harder. A measured spread that is small
    must not be allowed to shrink the registered gate."""
    quiet = OVS.Contrast("occupancy", "2", "6", 0.8, 0.9, 0.0001, 3)
    loud = OVS.Contrast("occupancy", "2", "6", 0.8, 0.9, 0.05, 3)
    assert OVS._floor(quiet, 0.046) == pytest.approx(0.046)
    assert OVS._floor(loud, 0.046) == pytest.approx(0.15)


def test_override_gate_names_the_setting_that_compiled_nothing():
    """All three knobs are compile-time constants, so a setting that compiled
    nothing measured the previous setting again."""
    gate = OVS.gate_override({"s3w8g1": 4, "s3w8g16": 0}, {"s3w8g1": 8})
    assert gate.passed is False
    assert "s3w8g16" in gate.observed


def test_non_vacuity_gate_fails_on_an_empty_report():
    gate = OVS.gate_non_vacuity([], [])
    assert gate.passed is False


def test_provenance_gate_fails_when_a_hypothesis_reaches_a_measured_run():
    limits = OVS.CardLimits(H200, 233472, 2048, 32, 132, 50_000_000,
                            OVS.HYPOTHESIS_L2_SOURCE)
    assert OVS.gate_provenance(limits, "read off the device", True).passed \
        is False
    good = OVS.CardLimits(H200, 233472, 2048, 32, 132, 50_000_000,
                          "read off NVIDIA H200")
    assert OVS.gate_provenance(good, "read off NVIDIA H200", True).passed


def test_reference_gate_fails_on_a_refused_reference():
    """A reference that is proportional but at the wrong level classifies every
    tread against a branch 40x too steep and prints blanks."""
    results = [_result("s3w8g1", 3, 8, 1, 4, None, (OVS.ARM_OCCUPANCY,),
                       refused=True)]
    gate = OVS.gate_references(results)
    assert gate.passed is False and "s3w8g1" in gate.observed


def test_a_setting_with_a_refused_reference_is_not_usable():
    assert not _result("x", 3, 8, 1, 4, 0.9, (OVS.ARM_OCCUPANCY,),
                       refused=True).usable
    assert not _result("x", 3, 8, 1, 4, 0.9, (OVS.ARM_OCCUPANCY,),
                       memory_points=2).usable
    assert _result("x", 3, 8, 1, 4, 0.9, (OVS.ARM_OCCUPANCY,)).usable


def test_contrast_averages_settings_that_share_a_residency_level():
    """Two num_stages can compute to the same resident-block count -- four and
    five do on an A100 -- and treating them as two rungs would put a pure
    num_stages effect on an axis that did not move."""
    results = [
        _result("s4w8g1", 4, 8, 1, 2, 0.80, (OVS.ARM_OCCUPANCY,)),
        _result("s5w8g1", 5, 8, 1, 2, 0.84, (OVS.ARM_OCCUPANCY,)),
        _result("s2w8g1", 2, 8, 1, 6, 0.90, (OVS.ARM_OCCUPANCY,)),
    ]
    c = OVS.occupancy_contrast(results)
    assert c.levels == 2
    assert c.lo_alpha == pytest.approx(0.82)
    assert c.swing == pytest.approx(0.08)


def test_verdict_covers_all_five_states():
    def g(passed):
        return OVS.Gate(OVS.CLAIM, "x", "y", "z", passed, "")
    assert OVS.verdict_of(g(True), g(False)) == OVS.VERDICT_CONCURRENCY
    assert OVS.verdict_of(g(False), g(True)) == OVS.VERDICT_ORDER
    assert OVS.verdict_of(g(True), g(True)) == OVS.VERDICT_BOTH
    assert OVS.verdict_of(g(False), g(False)) == OVS.VERDICT_NEITHER
    assert OVS.verdict_of(g(None), g(None)) == OVS.VERDICT_UNREADABLE
    assert set(OVS.VERDICT_NOTE) == {
        OVS.VERDICT_CONCURRENCY, OVS.VERDICT_ORDER, OVS.VERDICT_BOTH,
        OVS.VERDICT_NEITHER, OVS.VERDICT_UNREADABLE}


def test_swizzle_collapse_is_the_signature_and_not_a_fallback():
    """A refused reference is a DIFFERENT state and must not be read as the
    collapse reuse distance would produce."""
    good_base = _result("s3w8g1", 3, 8, 1, 4, 0.9, (OVS.ARM_SWIZZLE,))
    lost = _result("s3w8g64", 3, 8, 64, 4, None, (OVS.ARM_SWIZZLE,),
                   memory_points=1)
    assert OVS.swizzle_collapse([good_base, lost]) == ([64], True)
    broken = _result("s3w8g64", 3, 8, 64, 4, None, (OVS.ARM_SWIZZLE,),
                     refused=True)
    assert OVS.swizzle_collapse([good_base, broken]) == ([], True)


# --------------------------------------------------------------------------
# The refusals.
# --------------------------------------------------------------------------

def test_run_id_carries_every_swept_knob_and_the_card():
    """A run id that omitted GROUP_SIZE_M overwrote a whole arm in this repo,
    and one that omitted the CARD reported an H200's timings against an A100's
    ridge. Both are in here, and so is everything else that is swept."""
    base = OVS.build_parser().parse_args([])
    ident = OVS.default_run_id(base, "nvidia_h200")
    assert OVS.default_run_id(base, "nvidia_a100_sxm4_80gb") != ident
    for flag, value in (("--stages", "2,3,4"), ("--groups", "1,16"),
                        ("--control-warps", "2"), ("--control-stages", "3"),
                        ("--reps", "7"), ("--r-max", "512"),
                        ("--block-n", "128"), ("--block-k", "32"),
                        ("--iters", "50"), ("--warmup", "9"),
                        ("--cell-budget-ms", "111"), ("--seed", "5"),
                        ("--model", "qwen2-57b-a14b"), ("--dtype", "fp16")):
        other = OVS.build_parser().parse_args([flag, value])
        assert OVS.default_run_id(other, "nvidia_h200") != ident, flag
    # And a knob that only RE-ANALYSES the same timings must not fork the
    # directory, or a re-report becomes an empty resume.
    same = OVS.build_parser().parse_args(["--l2-bytes", "40000000"])
    assert OVS.default_run_id(same, "nvidia_h200") == ident
    same = OVS.build_parser().parse_args(["--sm-count", "108"])
    assert OVS.default_run_id(same, "nvidia_h200") == ident, "sm_count is analysis-only"
    # But --capability PRUNES the residency ladder (it sets smem_per_sm and so
    # how many blocks fit), so it is a measurement knob and MUST fork. Found in
    # review 2026-09-02: asserting 8.0 on an sm_90 box dropped the s=5 rung and
    # the 8-setting run would have resumed into the 9-setting directory.
    forked = OVS.build_parser().parse_args(["--capability", "8.0"])
    assert OVS.default_run_id(forked, "nvidia_h200") != ident, "capability prunes the grid"


def test_plan_refuses_the_settings_whose_reference_cannot_run():
    """A spilled or over-subscribed reference is perfectly proportional and 40x
    too steep, so it has to be dropped on the laptop and not diagnosed on the
    pod."""
    from moe.spec import MODEL_CONFIGS
    args = OVS.build_parser().parse_args(
        ["--stages", "3,6", "--control-warps", "2", "--control-stages", "3"])
    plan = OVS.build_plan(args, MODEL_CONFIGS["mixtral-8x7b"], 2, _limits(),
                          alpha=0.9, ridge=162.8, bandwidth_gbps=4374.5)
    assert "s6w8g1" in plan.refused
    assert "shared memory" in plan.refused["s6w8g1"]
    assert "s3w2g1" in plan.refused
    assert "registers per thread" in plan.refused["s3w2g1"]
    assert all(s.key not in plan.refused for s in plan.settings)


def test_plan_refuses_an_r_max_that_starves_the_reference():
    from moe.spec import MODEL_CONFIGS
    args = OVS.build_parser().parse_args(["--r-max", "512"])
    with pytest.raises(SystemExit) as exc:
        OVS.build_plan(args, MODEL_CONFIGS["mixtral-8x7b"], 2, _limits(),
                       alpha=0.9, ridge=162.8, bandwidth_gbps=4374.5)
    assert "reference" in str(exc.value) and "768" in str(exc.value)


def test_ladder_rows_refuses_a_model_that_cannot_form_a_full_tile_stack():
    """deepseek-v2-lite at E=64 k=6 needs rows in multiples of 3, and no
    multiple of 64 is one. Refused, never nudged: a nudged row is not a full
    tile stack and a fit over partly filled treads is a fit over padding."""
    from moe.spec import MODEL_CONFIGS
    with pytest.raises(SystemExit) as exc:
        OVS.ladder_rows(MODEL_CONFIGS["deepseek-v2-lite"], 64, 1024)
    assert "multiple of 3" in str(exc.value)
    assert OVS.ladder_rows(MODEL_CONFIGS["mixtral-8x7b"], 64, 1024)[-1] == 1024


def test_settings_share_one_base_rather_than_measuring_it_three_times():
    settings = OVS.build_settings(OVS.DEFAULT_STAGES, OVS.DEFAULT_CONTROL_WARPS,
                                  OVS.CONTROL_WARP_STAGES, OVS.DEFAULT_GROUPS)
    keys = [s.key for s in settings]
    assert len(keys) == len(set(keys))
    base = next(s for s in settings if s.key == "s3w8g1")
    assert set(base.arms) == {OVS.ARM_OCCUPANCY, OVS.ARM_WARPS,
                              OVS.ARM_SWIZZLE}


def test_card_contradiction_is_refused_rather_than_resolved():
    assert OVS.main(["--card", "nvidia_h200", "--audit"]) == 0


def test_replay_without_stored_inputs_refuses(tmp_path):
    """Re-scoring a run against THIS machine's ridge and L2 is the
    hybrid-of-two-machines failure this study has already published once."""
    (tmp_path / "cells.csv").write_text("setting\n")
    code = OVS.main(["--replay", str(tmp_path), "--capability", "9.0",
                     "--sm-count", "132", "--l2-bytes", "50000000"])
    assert code == 2


def test_replay_of_a_missing_run_refuses(tmp_path):
    assert OVS.main(["--replay", str(tmp_path), "--capability", "9.0",
                     "--sm-count", "132", "--l2-bytes", "50000000"]) == 2


def test_git_visibility_knows_the_published_carve_out():
    assert OVS.git_visibility(ROOT / "results" / "scratch").startswith("IGNORED")
    assert not OVS.git_visibility(PUBLISHED / "x.json").startswith("IGNORED")


# --------------------------------------------------------------------------
# The published corpus.
# --------------------------------------------------------------------------

def test_corpus_loads_and_names_every_skip():
    rows, skipped = OVS.load_corpus(PUBLISHED)
    assert rows, "no published BLOCK_M=64 alphas found"
    assert all(0.0 < r.alpha < 2.0 for r in rows)
    assert all(":" in s for s in skipped), skipped


def test_corpus_refutes_reuse_distance_and_says_so():
    """A1 is the gate that motivates the pod run: the standard predictor misses
    by roughly the factor this experiment is about."""
    rows, skipped = OVS.load_corpus(PUBLISHED)
    lines, gates, payload = OVS.audit_report(rows, skipped, 64, 64, 2, 1024)
    by = {g.name.split()[0]: g for g in gates}
    assert by["A0"].passed is True
    assert by["A1"].passed is False
    predicted = payload["order_gate"] / OVS.ORDER_RATIO_TOLERANCE
    # The corpus median is 0.85 against a capped prediction of 0.12 and an
    # uncapped one of 0.016: seven times and fifty times respectively. Both
    # bounds are asserted because the CAPPED form is the honest one and is
    # still refuted, which is the sentence the pod run is built on.
    assert payload["median_group_ratio"] > 6 * predicted
    assert payload["median_group_ratio"] > 40 * OVS.alpha_order(1.0, 64, 1e9)
    assert "\n".join(lines)


def test_corpus_cannot_answer_the_occupancy_question_and_reports_unknown():
    """Every num_stages pair on disk spans two sessions, so it is a prior and
    never evidence. UNKNOWN, not PASS and not FAIL."""
    rows, skipped = OVS.load_corpus(PUBLISHED)
    _, gates, payload = OVS.audit_report(rows, skipped, 64, 64, 2, 1024)
    a2 = next(g for g in gates if g.name.startswith("A2"))
    assert a2.passed is None
    assert payload["stage_contrasts"]


def test_corpus_group_contrasts_never_cross_a_session():
    """Pairing across the two H200 sessions would put a session effect -- and a
    num_stages difference -- on the swizzle axis."""
    rows, _ = OVS.load_corpus(PUBLISHED)
    for label, _lo, _hi, _a, _b in OVS.corpus_group_contrasts(rows):
        matches = [r for r in rows
                   if f"{r.card} {r.model} n{r.block_n} w{r.num_warps} "
                      f"s{r.num_stages}" == label]
        assert len({r.session for r in matches}) == 1, label


def test_audit_card_specs_are_only_used_by_the_audit():
    """A measured run reads the attached device. The part-specification table
    exists to score history and must never become a fallback."""
    src = (ROOT / "scripts" / "occupancy_vs_swizzle.py").read_text()
    body = src[src.index("def resolve_device"):src.index("def resolve_roofline")]
    assert "CARD_SPECS" not in body


# --------------------------------------------------------------------------
# V9: the residency model, checked against what Triton actually compiled.
# --------------------------------------------------------------------------

def test_kernel_probe_reports_rather_than_raises_without_vllm():
    """It runs inside the metered loop. A probe that threw would cost a pod
    session to learn nothing."""
    probe = OVS.KernelProbe()
    probe.record("s3w8g1")
    assert probe.by_setting == {}
    assert probe.note, "a probe that found nothing must say why"


def test_compiled_smem_gate_is_unknown_when_nothing_was_probed():
    """A check that examined nothing also reports zero disagreements."""
    gate = OVS.gate_compiled_smem({}, "no vLLM", [])
    assert gate.passed is None
    assert "UPPER BOUND" in gate.invalidates


def test_compiled_smem_gate_catches_a_wrong_residency_model():
    """If Triton allocates something else, every rung of the ladder is in the
    wrong place and the report still plots."""
    results = [_result("s3w8g1", 3, 8, 1, 4, 0.9, (OVS.ARM_OCCUPANCY,))]
    agree = {"s3w8g1": {"shared": 3 * 16384, "n_regs": 40, "n_spills": 0}}
    assert OVS.gate_compiled_smem(agree, "", results).passed is True
    disagree = {"s3w8g1": {"shared": 2 * 16384, "n_regs": 40, "n_spills": 0}}
    assert OVS.gate_compiled_smem(disagree, "", results).passed is False


def test_compiled_smem_gate_fails_on_register_spills():
    """A spilled kernel's time is not the time of this tiling, and a spilled
    kernel still fits a straight line."""
    results = [_result("s3w8g1", 3, 8, 1, 4, 0.9, (OVS.ARM_OCCUPANCY,))]
    spilled = {"s3w8g1": {"shared": 3 * 16384, "n_regs": 255, "n_spills": 16}}
    gate = OVS.gate_compiled_smem(spilled, "", results)
    assert gate.passed is False and "s3w8g1:16" in gate.observed


def test_dry_run_and_run_contradict_each_other():
    assert OVS.main(["--dry-run", "--run"]) == 2


def test_dry_run_works_off_gpu_and_measures_nothing(capsys):
    """The pod session reads this on a laptop before spending anything."""
    assert OVS.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Predictions, registered before anything is measured" in out
    assert "HYPOTHESIS" in out
    assert "blk/SM" in out


def test_failed_and_unchecked_validity_gates_print_differently():
    """A FAILED validity gate says the verdict is wrong; an UNKNOWN one says a
    specific thing was not checked. Printing the same sentence for both trains
    a reader to ignore the loud one."""
    _, cfg, b, plan, reg, _ = _plan_and_reg()
    samples = OVS.planted_samples(
        cfg, plan, lambda st: reg.concurrency_alpha[st.key], ridge=162.8,
        bandwidth_gbps=4374.5, b=b, noise=0.002, seed=5)
    lines, _, _ = OVS.analyse(
        samples, cfg, plan, reg, b, ridge=162.8, bandwidth_gbps=4374.5,
        compiles={s.key: 1 for s in plan.settings},
        executed={s.key: plan.reps for s in plan.settings},
        l2_source="read off a device", measured=True)
    text = "\n".join(lines)
    assert "NOT CHECKED: V9 compiled shared memory" in text
    assert "READ THIS FIRST" not in text
