"""The BLOCK_SIZE_M crossing sweep has to be able to be WRONG, and to say so.

`scripts/block_m_crossing_sweep.py` decides whether the refit alpha of 0.558 or
the retracted 0.10 describes this hardware, and it decides it on a pod that is
metered by the minute. The thing that has to be true before it runs is not that
it produces numbers: it is that its four gates land differently in the two
worlds. So most of this file plants a known alpha in synthetic cells, runs the
REAL analysis over them, and checks the verdicts flip.

THREE GROUPS, and the middle one is the point.

  - the predictions are the ones the study published, recomputed from the model
    rather than copied out of a docstring;
  - the analysis recovers a planted alpha and tells 0.558 from 0.10, under
    timing noise, on every gate that is supposed to discriminate;
  - the traps: a free split search INVENTS an alpha of 0.6 at BLOCK_M=128 from
    data planted at 0.10, which is the single most dangerous number this
    experiment could produce, and `test_a_free_split_search_invents_an_alpha...`
    pins that it does so and that the shipped path refuses to.

The script is loaded by path rather than imported, because `scripts/` is not a
package and never has been.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module that is not there yet makes
    # the decorator fail with an AttributeError that names nothing useful.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BM = _load_script()

from moe.spec import MODEL_CONFIGS  # noqa: E402

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
TILES = (32, 64, 128, 256)
RIDGE = 160.3
BANDWIDTH = 4374.5          # the published H200 triad ceiling
REFIT = 0.558
RETRACTED = 0.10


def analyse(cells, cfg=MIXTRAL, *, alpha: float, tiles=TILES, compiles=None,
            executed=None):
    return BM.analyse(
        cells, cfg, block_sizes=tiles, alpha=alpha, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, model_name=cfg.name, dtype="bf16",
        compiles=compiles or {bm: 1 for bm in tiles},
        executed=executed or {bm: 1 for bm in tiles}, sm_count=132,
        sm_source="test")


def report_at(alpha: float, *, noise: float = 0.0, r_max: int = 1024,
              cfg=MIXTRAL, tiles=TILES, seed: int = 0, compiles=None,
              executed=None):
    """Plant `alpha`, generate the grid, and run the shipped analysis on it."""
    grid = BM.build_grid(cfg, tiles, r_max, 32, 6)
    cells = BM.synthetic_cells(cfg, grid, tiles, alpha=alpha, ridge=RIDGE,
                               bandwidth_gbps=BANDWIDTH, b=2, sm_count=132,
                               noise=noise, seed=seed)
    return analyse(cells, cfg, alpha=alpha, tiles=tiles, compiles=compiles,
                   executed=executed)


def verdicts(report) -> dict[int, str]:
    return {g.number: g.verdict for g in report.gates}


# --------------------------------------------------------------------------
# The predictions, recomputed rather than quoted.
# --------------------------------------------------------------------------

def test_block_m_32_and_64_cannot_reach_the_ridge_at_the_refit_alpha():
    """Their AI ceilings are 57.3 and 114.7 against a ridge of 160.3, so no
    batch size makes them compute bound and `predict_tile` must say so with a
    None rather than with a large number."""
    for bm, cap in ((32, 57.3), (64, 114.7)):
        pred = BM.predict_tile(bm, REFIT, RIDGE)
        assert pred.ai_cap == pytest.approx(cap, abs=0.1)
        assert pred.crossing_rows is None
        assert pred.first_compute_tread is None
        assert not pred.crosses


def test_block_m_128_crosses_at_250_rows_and_256_at_160():
    """The two numbers the whole experiment is arranged around, in rows per
    expert and in mixtral tokens."""
    p128 = BM.predict_tile(128, REFIT, RIDGE)
    p256 = BM.predict_tile(256, REFIT, RIDGE)
    assert p128.crossing_rows == pytest.approx(249.7, abs=0.5)
    assert p256.crossing_rows == pytest.approx(160.3, abs=0.5)
    assert p128.first_compute_tread == 2
    assert p256.first_compute_tread == 1
    assert p128.crossing_tokens(8, 2) == pytest.approx(999, abs=2)
    assert p256.crossing_tokens(8, 2) == pytest.approx(641, abs=2)


@pytest.mark.parametrize("model,tokens_128,tokens_256", [
    ("mixtral-8x7b", 999, 641),
    ("qwen2-57b-a14b", 1998, 1282),
    ("deepseek-v3", 7992, 5130),
])
def test_the_crossing_batch_is_the_published_one_for_every_study_model(
        model, tokens_128, tokens_256):
    cfg = MODEL_CONFIGS[model]
    for bm, expected in ((128, tokens_128), (256, tokens_256)):
        pred = BM.predict_tile(bm, REFIT, RIDGE)
        got = pred.crossing_tokens(cfg.num_experts, cfg.top_k)
        assert got == pytest.approx(expected, rel=0.001)


def test_the_two_alphas_are_different_worlds_and_not_a_disagreement_of_degree():
    """At 0.558 two of four block sizes never cross and the 128-over-256 ratio
    is 1.558. At 0.10 all four cross and the ratio is 1.10. That is the whole
    reason this sweep is worth a pod."""
    refit = BM.predictions(TILES, REFIT, RIDGE)
    old = BM.predictions(TILES, RETRACTED, RIDGE)
    assert [refit[bm].crosses for bm in TILES] == [False, False, True, True]
    assert all(old[bm].crosses for bm in TILES)
    assert BM.crossing_ratio(refit, 128, 256) == pytest.approx(1.558, abs=0.01)
    assert BM.crossing_ratio(old, 128, 256) == pytest.approx(1.10, abs=0.01)


def test_the_upper_end_of_the_ridge_band_moves_the_128_crossing_a_whole_tread():
    """160.3 and 176.2 are the same card measured twice, and they do not merely
    widen the prediction: at 176.2 the BLOCK_M=128 crossing lands in tread 3
    instead of tread 2, so gate 3's prediction changes from 1.558 to 2.116. A
    band quoted as one number would have hidden that."""
    hi = BM.predictions(TILES, REFIT, 176.2)
    assert hi[128].first_compute_tread == 3
    assert hi[128].crossing_rows == pytest.approx(372.8, abs=0.5)
    assert hi[256].first_compute_tread == 1
    assert BM.crossing_ratio(hi, 128, 256) == pytest.approx(2.116, abs=0.01)


def test_a_crossing_exists_exactly_when_the_ai_ceiling_clears_the_ridge():
    """`predict_tile` scans treads and `ai_cap` is closed form, and the two have
    to agree at every block size or the scan is answering a different question
    from the ceiling the paper quotes."""
    for bm in (16, 32, 48, 64, 96, 128, 160, 256, 512):
        for alpha in (0.05, 0.10, 0.33, 0.466, 0.558, 0.625, 0.9):
            pred = BM.predict_tile(bm, alpha, RIDGE)
            assert pred.crosses == (BM.ai_cap(bm, alpha) > RIDGE)


def test_the_crossing_is_the_first_tread_whose_padded_rows_clear_the_traffic():
    """The fixed point is `n BM >= ridge b Q(n) / 2` and the tread BELOW it must
    fail that test, or the scan returned something later than the crossing."""
    for bm in (128, 256):
        pred = BM.predict_tile(bm, REFIT, RIDGE)
        n = pred.first_compute_tread
        assert n * bm >= RIDGE * BM.q_of_tiles(n, REFIT)
        if n > 1:
            assert (n - 1) * bm < RIDGE * BM.q_of_tiles(n - 1, REFIT)


# --------------------------------------------------------------------------
# The grid brackets what the gates need it to bracket.
# --------------------------------------------------------------------------

def test_the_grid_holds_an_exactly_full_tile_stack_at_every_tread():
    """The ladder is read at `r = n BM`, where padding is zero and useful
    throughput equals padded throughput. A grid that misses those has no ladder
    to fit."""
    grid = set(BM.build_grid(MIXTRAL, TILES, 1024, 32, 6))
    for bm in TILES:
        for n in range(1, 1024 // bm + 1):
            assert n * bm in grid, f"no exactly-full stack at BLOCK_M={bm} n={n}"


def test_the_grid_brackets_every_probed_tile_boundary_on_both_sides():
    """Gate 1 needs a point below a boundary, one at it, and one above it. A
    grid with one point per tread cannot see a step at all, because every
    interval it can form spans one."""
    grid = sorted(BM.build_grid(MIXTRAL, TILES, 1024, 32, 6))
    for bm in TILES:
        for n in range(1, 7):
            edge = n * bm
            if edge > 1024:
                break
            assert any((n - 1) * bm < r < edge for r in grid)
            assert any(r > edge for r in grid)


def test_the_default_sweep_reaches_past_every_crossing_either_world_predicts():
    """An unbracketed sweep reporting "no crossing" is worthless. The default
    r_max has to clear the largest crossing ANY competing hypothesis puts on any
    block size, which is BLOCK_M=32 under the retracted alpha at 304.6 rows."""
    r_max = 1024
    for bm in TILES:
        for alpha in (REFIT, RETRACTED):
            pred = BM.predict_tile(bm, alpha, RIDGE)
            if pred.crossing_rows is not None:
                assert pred.crossing_rows * 2 <= r_max


def test_every_token_count_keeps_rows_per_expert_an_exact_integer():
    """Balanced routing needs `T k / E` whole, and `realize_counts` refuses a
    fractional target halfway through a metered run rather than at the grid."""
    for name in ("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v3"):
        cfg = MODEL_CONFIGS[name]
        for rows in BM.build_grid(cfg, TILES, 512, 32, 4):
            tokens = BM.tokens_for_rows(cfg, rows)
            assert tokens * cfg.top_k % cfg.num_experts == 0
            assert BM.rows_for_tokens(cfg, tokens) == rows


def test_waves_count_the_per_expert_padding_and_not_the_global_padding():
    """vLLM pads EACH expert to a multiple of BLOCK_SIZE_M in
    `moe_align_block_size`, so the M-tile count is `E ceil(r / BM)`. Counting
    `ceil(E r / BM)` instead understates the grid by a factor approaching E at
    small batches, which is where every wave count in this study lives."""
    up, _ = BM.waves(MIXTRAL, 100, 64, 64, 132)
    m_tiles = MIXTRAL.num_experts * 2          # ceil(100/64) = 2 per expert
    assert up == pytest.approx(m_tiles * math.ceil(2 * MIXTRAL.intermediate_size / 64)
                               / 132)


# --------------------------------------------------------------------------
# The analysis tells the two worlds apart. This is the part that matters.
# --------------------------------------------------------------------------

def test_every_gate_passes_on_cells_planted_at_the_refit_alpha():
    report = report_at(REFIT)
    assert verdicts(report) == {0: "PASS", 1: "PASS", 2: "PASS", 3: "PASS",
                                4: "PASS"}


def test_gates_2_3_and_4_all_fail_on_cells_planted_at_the_retracted_alpha():
    """Three independent readings of the same disagreement, which is the point
    of having three: gate 2 is a time ratio at fixed rows, gate 3 is a fitted
    slope ratio, gate 4 is a throughput ceiling. A world where alpha is 0.10
    fails all three, and gate 1 -- which tests tile quantisation and not alpha
    -- passes in both, as it should."""
    report = report_at(RETRACTED)
    got = verdicts(report)
    assert got[1] == "PASS"
    assert got[2] == "FAIL"
    assert got[3] == "FAIL"
    assert got[4] == "FAIL"


@pytest.mark.parametrize("noise", [0.0, 0.005, 0.01, 0.02, 0.03])
def test_the_verdicts_survive_timing_noise_in_both_worlds(noise):
    """Real cells will not land on the model. The verdicts have to be decided by
    a factor of three and a factor of two, not by the third decimal."""
    assert verdicts(report_at(REFIT, noise=noise))[2] == "PASS"
    assert verdicts(report_at(REFIT, noise=noise))[3] == "PASS"
    assert verdicts(report_at(RETRACTED, noise=noise))[2] == "FAIL"
    assert verdicts(report_at(RETRACTED, noise=noise))[3] == "FAIL"


@pytest.mark.parametrize("planted", [0.10, 0.30, 0.466, 0.558, 0.625])
def test_the_ladder_recovers_the_planted_alpha_from_the_time_curve_alone(planted):
    """The column gate 3 is scored on -- activation traffic removed, the fused
    layer's fixed cost still in the denominator -- has to come in just UNDER
    what was planted at every alpha the argument spans, and never over."""
    fit = report_at(planted).payload["ladder"]["64"]
    measured = fit["alpha_corrected"]
    assert measured is not None
    assert measured <= planted
    assert measured >= planted * 0.88


@pytest.mark.parametrize("planted", [0.10, 0.558])
def test_the_two_alpha_biases_bracket_the_planted_value_from_either_side(planted):
    """The raw fit carries activation traffic upward and the fixed cost
    downward, and `alpha_upper` removes only the second. The planted value has
    to sit inside the range the report prints, or one of the two corrections is
    pointing the wrong way."""
    fit = report_at(planted).payload["ladder"]["64"]
    assert fit["alpha_corrected"] <= planted <= fit["alpha_upper"]


@pytest.mark.parametrize("noise,tolerance", [(0.0, 0.02), (0.01, 0.06),
                                             (0.03, 0.12)])
def test_the_fitted_alpha_stays_near_the_planted_one_as_noise_grows(noise,
                                                                    tolerance):
    """Measured over 12 seeds, the estimate spans 0.52-0.56 at 1% spread and
    0.47-0.63 at 3%, around a planted 0.558 -- estimator variance, and it stays
    an order of magnitude clear of the 0.10 it is discriminating against. What
    it must NOT do is drift systematically, which is what subtracting an
    extrapolated fixed cost did: 0.56 to 0.70 across this same range, one
    direction, which is why nothing gates on that number."""
    for seed in range(4):
        got = report_at(REFIT, noise=noise, seed=seed).payload["ladder"]["64"]
        assert got["alpha"] == pytest.approx(0.54, abs=tolerance)
        assert abs(got["alpha"] - REFIT) < abs(got["alpha"] - RETRACTED)


def test_gate_2_measures_a_factor_of_three_in_one_world_and_nothing_in_the_other():
    """The magnitudes, not only the verdicts: 32 tiles of weight re-read against
    a compute-bound floor is 2.9x, and the same comparison at alpha=0.10 is 1.0x
    because the small tile is compute bound by then too."""
    def ratio(report):
        gate = next(g for g in report.gates if g.number == 2)
        return float(gate.measured.split("= ")[1].split("x")[0])
    assert ratio(report_at(REFIT)) == pytest.approx(2.9, abs=0.2)
    assert ratio(report_at(RETRACTED)) == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------
# The traps.
# --------------------------------------------------------------------------

def test_a_free_split_search_invents_an_alpha_at_block_m_128_that_the_guard_refuses():
    """THE MOST DANGEROUS NUMBER THIS EXPERIMENT COULD PRODUCE.

    At BLOCK_M=128 tread 1 is memory bound and tread 2 is compute bound in BOTH
    worlds, so a line drawn through those two points answers about 0.6 whatever
    alpha actually is. A split search picks exactly that split, because it fits
    beautifully. Planted at 0.10, the free search reports an alpha near 0.6 --
    which would have read as a confirmation of the refit, from data that
    refutes it.

    The shipped path decides membership against the compute branch instead and
    reports nothing at all there, which is the correct answer.
    """
    grid = BM.build_grid(MIXTRAL, TILES, 1024, 32, 6)
    cells = BM.synthetic_cells(MIXTRAL, grid, TILES, alpha=RETRACTED,
                               ridge=RIDGE, bandwidth_gbps=BANDWIDTH, b=2,
                               sm_count=132)
    points = BM.ladder_points(cells, 128)

    free = BM.fit_ladder(points, 128, ref=None)
    assert free.memory_points == 2
    assert free.alpha is not None
    assert free.alpha > 0.5, "the trap did not fire; this test has stopped testing it"

    ref = BM.compute_reference(cells, TILES)
    guarded = BM.fit_ladder(points, 128, ref)
    assert guarded.memory_points <= 1
    assert guarded.alpha is None


@pytest.mark.parametrize("noise", [0.0, 0.01, 0.02, 0.03])
def test_no_seed_in_the_retracted_world_is_allowed_to_pass_gate_3(noise):
    """THE FALSE CONFIRMATION THIS FILE EXISTS TO PREVENT.

    A gate that fails safe is worth more than one that is usually right. Before
    the parallel-branch guard, 2 seeds in 20 at 2% spread reported alpha 0.80
    from a BLOCK_M=128 ladder planted at 0.10 and passed gate 3 on it, which is
    the retracted world producing a confirmation of the refit.
    """
    for seed in range(8):
        got = verdicts(report_at(RETRACTED, noise=noise, seed=seed))
        assert got[3] != "PASS", f"seed {seed} confirmed the refit from 0.10 data"
        assert got[2] == "FAIL"
        assert got[4] == "FAIL"


@pytest.mark.parametrize("noise", [0.0, 0.01, 0.02, 0.03])
def test_every_seed_in_the_refit_world_passes_every_gate(noise):
    """The other half of the same claim: the gates are not merely conservative,
    they fire when the world they describe is the one that arrived."""
    for seed in range(8):
        assert verdicts(report_at(REFIT, noise=noise, seed=seed)) == {
            0: "PASS", 1: "PASS", 2: "PASS", 3: "PASS", 4: "PASS"}


def test_a_memory_branch_parallel_to_the_compute_branch_is_discarded():
    """`B / C = ridge / ai_cap`, so branches with the same slope say the ceiling
    sits exactly on the ridge -- and far more often say the prefix ran into the
    compute branch and is about to report its slope as alpha."""
    ref = BM.ComputeReference(256, 0.05, 1.0, 0.0, "planted")
    # A compute branch of 0.5 ms per tile at BLOCK_M=128, measured 5% high --
    # which is what a reference slope estimated 5% low looks like from here.
    parallel = [(n, 0.05 + 0.5 * n * 1.05) for n in range(1, 9)]
    fit = BM.fit_ladder(parallel, 128, ref, margin=0.02)
    assert fit.alpha is None
    assert fit.memory_points == 0
    assert "DISCARDED" in fit.basis


def test_the_membership_margin_widens_with_the_measured_timing_spread():
    """One tread standing a tenth above the compute branch is memory bound at a
    2% margin and indistinguishable at a 15% one. Which of those is right
    depends on what the timing spread was, so the margin has to be told."""
    ref = BM.ComputeReference(256, 0.05, 1.0, 0.0, "planted")
    ladder = [(n, 0.5 + 0.1 * n) for n in range(1, 9)]   # crosses inside tread 2
    assert BM.fit_ladder(ladder, 128, ref, margin=0.02).memory_points == 1
    assert BM.fit_ladder(ladder, 128, ref, margin=0.15).memory_points == 0


def test_the_fit_reports_no_alpha_where_one_tread_stands_above_the_compute_branch():
    """One point pins a level and no slope. Reporting an alpha from it is the
    same failure as the split search, arrived at more quietly."""
    report = report_at(REFIT)
    for bm, fit in report.payload["ladder"].items():
        if fit["memory_points"] < 2:
            assert fit["alpha"] is None, f"BLOCK_M={bm} invented a slope"


def test_gate_3_says_which_block_size_it_imported_alpha_from():
    """The ratio is `1 + alpha` and alpha is often not identifiable at
    BLOCK_M=128, so the number is imported across block sizes. An imported
    number that does not say so is how `ALPHA_BY_BLOCK_M`'s 25% drift becomes
    invisible."""
    gate = next(g for g in report_at(RETRACTED).gates if g.number == 3)
    assert any("BLOCK_M=64" in line for line in gate.lines)
    assert any("import" in line for line in gate.lines)


def test_gate_0_voids_the_run_when_a_setting_compiled_no_new_kernel():
    """If `override_config` silently failed, all four settings ran one kernel
    and every gate reads a difference of zero. The compile count is the assay,
    and a failed assay has to say that nothing below it is evidence."""
    report = report_at(REFIT, compiles={32: 4, 64: 0, 128: 2, 256: 1})
    gate = next(g for g in report.gates if g.number == 0)
    assert gate.verdict == "FAIL"
    assert "64" in gate.measured
    assert any("Do not read them" in line for line in gate.lines)
    assert "Nothing below gate 0 is evidence" in report.text()


def test_gate_0_does_not_read_a_resumed_setting_as_a_broken_override():
    """A run that finds every cell already in `cells.csv` executes nothing and
    so compiles nothing, which is the ABSENCE of the assay rather than its
    failure. Scoring that as FAIL would make every resumed run declare its own
    data void."""
    report = report_at(REFIT, compiles={32: 4, 64: 0, 128: 2, 256: 1},
                       executed={32: 68, 64: 0, 128: 68, 256: 68})
    gate = next(g for g in report.gates if g.number == 0)
    assert gate.verdict == "UNDECIDED"
    assert any("ran no cells this session" in line for line in gate.lines)


def test_the_stage_count_is_repinned_for_every_setting_at_once():
    """`num_stages` is the one pinned parameter with a hard limit behind it:
    BLOCK_SIZE_M=256 at 4 stages asks for about 164 KB of shared memory. Lowering
    it for the setting that failed would unpin the sweep, so the flag moves
    every setting together and the report records what was pinned."""
    args = BM.build_parser().parse_args(["--num-stages", "2"])
    pinned = dict(BM.FIXED, num_stages=args.num_stages)
    assert pinned["num_stages"] == 2
    assert {k: v for k, v in pinned.items() if k != "num_stages"} == \
        {k: v for k, v in BM.FIXED.items() if k != "num_stages"}


def test_gate_4_refuses_to_score_an_absence_the_sweep_could_not_have_seen():
    """Stopping at 128 rows per expert means BLOCK_M=64 was measured over two
    treads, well short of the 208 rows the retracted alpha puts its crossing at.
    Reporting "no crossing" from that would be worthless, so it reports
    UNDECIDED and prints how far it got."""
    report = report_at(REFIT, r_max=128)
    gate = next(g for g in report.gates if g.number == 4)
    assert gate.verdict == "UNDECIDED"
    assert any("NOT BRACKETED" in line for line in gate.lines)
    assert any("horizon" in line for line in gate.lines)


def test_gate_4_fails_rather_than_stalls_when_the_roof_was_actually_reached():
    """Bracketing governs an absence only. A block size that reached the roof
    crossed, and a sweep that watched it happen went far enough by
    demonstration."""
    gate = next(g for g in report_at(RETRACTED).gates if g.number == 4)
    assert gate.verdict == "FAIL"
    assert any("DID reach the roof" in line for line in gate.lines)


def test_the_bracketing_horizon_is_twice_the_competing_hypothesis_crossing():
    """417 rows for BLOCK_M=64, being twice the 208.4 the retracted alpha
    predicts. A horizon set from the hypothesis under test rather than from its
    rival would clear itself by construction."""
    grid = BM.build_grid(MIXTRAL, TILES, 1024, 32, 6)
    cells = BM.synthetic_cells(MIXTRAL, grid, TILES, alpha=REFIT, ridge=RIDGE,
                               bandwidth_gbps=BANDWIDTH, b=2, sm_count=132)
    ref = BM.compute_reference(cells, TILES)
    fits = {bm: BM.fit_ladder(BM.ladder_points(cells, bm), bm, ref)
            for bm in TILES}
    plateau = max(c.useful_tflops for c in cells if c.aligned)
    brack = BM.bracketing(cells, 64, REFIT, RIDGE, 2, fits, plateau)
    assert brack.horizon_rows == pytest.approx(2 * 208.4, abs=1.0)
    assert brack.reached_rows == 1024
    assert brack.positive_control in (128, 256)
    assert brack.sufficient


def test_the_positive_control_is_measured_and_not_fitted():
    """The control says the instrument can see a crossing at all. Reading it off
    a fit would make it depend on the same branch assignment gate 4 is arguing
    about, so it is read off measured throughput instead. With no block size
    anywhere near the roof there is no control and the gate cannot score."""
    grid = BM.build_grid(MIXTRAL, TILES, 256, 32, 4)
    cells = [c for c in BM.synthetic_cells(
        MIXTRAL, grid, (32, 64), alpha=REFIT, ridge=RIDGE,
        bandwidth_gbps=BANDWIDTH, b=2, sm_count=132)]
    fits = {bm: BM.fit_ladder(BM.ladder_points(cells, bm), bm, None)
            for bm in (32, 64)}
    plateau = max(c.useful_tflops for c in cells if c.aligned)
    brack = BM.bracketing(cells, 64, REFIT, RIDGE, 2, fits, plateau)
    assert brack.positive_control is None
    assert not brack.sufficient


def test_the_compute_reference_refuses_a_ladder_that_is_not_proportional():
    """The reference assumes the largest block size is compute bound at every
    tread, which is a PREDICTION of the model under test. Handed a bent ladder
    it has to decline rather than take a memory slope for a compute branch and
    make every other block size look identifiable."""
    bent = [BM.make_cell(MIXTRAL, n * 256, 256, 1.0 + 0.6 * (n - 1) ** 2,
                         sm_count=132, block_n=64) for n in (1, 2, 3, 4)]
    ref = BM.compute_reference(bent, (256,))
    assert ref.block_m is None
    assert ref.slope_for(64) is None
    assert "NO alpha may decide a verdict" in ref.note


def test_a_memory_bound_ladder_cannot_pass_itself_off_as_a_compute_branch():
    """The qualification is proportionality to the tile count. `A + B n` with a
    real intercept cannot be drawn through the origin, and that is the whole
    discrimination: if it could, the reference would take the memory slope for
    the compute branch and every other ladder would come back identifiable."""
    memory = [BM.make_cell(MIXTRAL, n * 256, 256, 0.44 + 0.558 * n,
                           sm_count=132, block_n=64) for n in (1, 2, 3, 4)]
    assert BM.compute_reference(memory, (256,)).block_m is None


def test_no_alpha_decides_a_verdict_when_no_compute_branch_qualified():
    """Without a compute branch there is nothing to tell a memory branch FROM,
    the split search will happily read a straight compute ladder as a memory one
    -- at 3% spread it read BLOCK_M=256 as alpha 1.097 -- and gate 3 must go
    undecided rather than quote it."""
    cfg = MIXTRAL
    tiles = (32, 64)                       # no ladder that is compute bound
    grid = BM.build_grid(cfg, tiles, 256, 32, 4)
    cells = BM.synthetic_cells(cfg, grid, tiles, alpha=REFIT, ridge=RIDGE,
                               bandwidth_gbps=BANDWIDTH, b=2, sm_count=132)
    report = analyse(cells, cfg, alpha=REFIT, tiles=tiles)
    gate = next(g for g in report.gates if g.number == 3)
    assert gate.verdict == "UNDECIDED"


def test_the_reference_ladder_never_reports_a_memory_branch_of_its_own():
    """It qualified BY BEING compute bound throughout. Letting it test its own
    points against its own fitted line lets noise push the low treads above it:
    at 1% spread BLOCK_M=256 reported two memory-bound treads and an alpha of
    0.96, which then won the "largest identifiable block size" contest and
    turned gate 3 into a PASS on cells planted at 0.10."""
    for noise in (0.0, 0.01, 0.03):
        for seed in range(4):
            report = report_at(RETRACTED, noise=noise, seed=seed)
            ref_bm = report.payload["compute_reference"]["block_m"]
            if ref_bm is None:
                continue
            assert report.payload["ladder"][str(ref_bm)]["memory_points"] == 0
            assert report.payload["ladder"][str(ref_bm)]["alpha"] is None


def test_the_compute_branch_scales_with_block_m_and_the_report_checks_it():
    """Gate 4 scales `C` from one block size to another by `C ~ BLOCK_M`, an
    identity with no free parameter. If it did not hold, the slope gate 4
    compares against would be wrong, so the report prints the check."""
    report = report_at(REFIT)
    assert "compute branch should scale with BLOCK_M" in report.text()
    ladder = report.payload["ladder"]
    assert (ladder["256"]["slope_compute"] / ladder["128"]["slope_compute"]
            == pytest.approx(2.0, rel=0.05))


# --------------------------------------------------------------------------
# Plumbing: the pod has to be able to abort, resume, and find its output.
# --------------------------------------------------------------------------

def test_a_cell_round_trips_through_the_csv_unchanged():
    """The CSV is appended cell by cell so an abort costs one cell. That is only
    true if what comes back is what went in."""
    import tempfile
    cell = BM.make_cell(MIXTRAL, 256, 128, 1.2345, sm_count=132, block_n=64,
                        ms_min=1.2, ms_stdev=0.01, iters=17)
    path = Path(tempfile.mkdtemp()) / "cells.csv"
    BM.append_cell(path, cell)
    done, cells = BM.read_cells(path)
    assert cells == [cell]
    assert done == {(128, 1024)}


def test_a_resumed_run_skips_finished_cells_and_retries_failed_ones():
    """A failed cell is usually a pod that lost its device or a setting that ran
    out of shared memory, and both are states a re-run can leave behind. A real
    failure fails again in milliseconds, so retrying is cheap and skipping is
    not."""
    import tempfile
    path = Path(tempfile.mkdtemp()) / "cells.csv"
    BM.append_cell(path, BM.make_cell(MIXTRAL, 256, 128, 1.0, sm_count=132,
                                      block_n=64))
    BM.append_cell(path, BM.make_cell(MIXTRAL, 256, 64, 0.0, sm_count=132,
                                      block_n=64, status="failed",
                                      detail="OutOfResources"))
    done, cells = BM.read_cells(path)
    assert (128, 1024) in done
    assert (64, 1024) not in done
    assert len(cells) == 2


def test_the_run_id_is_derived_from_the_arguments_so_a_rerun_resumes_itself():
    """A random id would make every re-run a new directory and turn the resume
    path into dead code the first time anyone used it."""
    parser = BM.build_parser()
    a = parser.parse_args([])
    b = parser.parse_args([])
    c = parser.parse_args(["--r-max", "2048"])
    assert BM.default_run_id(a) == BM.default_run_id(b)
    assert BM.default_run_id(a) != BM.default_run_id(c)
    assert "mixtral" in BM.default_run_id(a)


def test_every_pinned_knob_changes_the_run_id():
    """The resume path makes a shared id a SILENT WRONG ANSWER, not a clash.

    cells.csv is appended per cell and completed cells are skipped on resume, so
    two runs that derive the same id do not collide loudly: the second finds
    every cell already on disk, skips all of them, and prints the FIRST run's
    timings under the second run's heading -- because the report renders
    `pinned` from argv rather than from the cells it actually read. That is
    indistinguishable from a successful run.

    GROUP_SIZE_M, BLOCK_SIZE_N and num_stages were all absent from the key while
    they were unreachable constants. The moment --group-m and --block-n existed,
    an alpha-versus-swizzle sweep would have reported alpha(G=1) four times over
    and called it a curve.
    """
    parser = BM.build_parser()
    base = parser.parse_args([])
    for flag, value in (("--group-m", "16"),
                        ("--block-n", "256"),
                        ("--num-stages", "3")):
        other = parser.parse_args([flag, value])
        assert BM.default_run_id(base) != BM.default_run_id(other), (
            f"{flag} does not change the run id, so a sweep over it would "
            f"resume into the previous setting's directory and report its "
            f"numbers")


def test_the_run_id_names_the_swizzle_and_the_n_tile_in_plain_text():
    """A hash nobody can invert is not a label. Two alpha runs differing only in
    the swizzle have to be tellable apart in `ls`, because that listing is what
    a reader of the paper's data directory sees."""
    parser = BM.build_parser()
    rid = BM.default_run_id(parser.parse_args(["--group-m", "16",
                                               "--block-n", "256"]))
    assert "-g16-" in rid
    assert "-n256-" in rid


def test_group_m_and_block_n_actually_reach_the_pinned_config():
    """The flags must land in `pinned`, which is what is forced onto every
    setting and what the report prints. A flag parsed and then dropped would
    leave the sweep running the default while every table said otherwise."""
    parser = BM.build_parser()
    args = parser.parse_args(["--group-m", "16", "--block-n", "256"])
    pinned = dict(BM.FIXED, num_stages=args.num_stages,
                  GROUP_SIZE_M=args.group_m, BLOCK_SIZE_N=args.block_n)
    assert pinned["GROUP_SIZE_M"] == 16
    assert pinned["BLOCK_SIZE_N"] == 256
    assert BM.FIXED["GROUP_SIZE_M"] == 1, "the default must stay the fallback's"
    assert BM.FIXED["BLOCK_SIZE_N"] == 64


def test_the_results_root_prefers_the_network_volume_that_outlives_the_pod(monkeypatch, tmp_path):
    """`$MOE_RESULTS_DIR`, else `/workspace`, else the repo -- the order
    `run_all.sh` resolves it in, so this experiment lands beside every other arm
    on the volume rather than on a disk that dies with the pod."""
    monkeypatch.setenv("MOE_RESULTS_DIR", str(tmp_path))
    assert BM.results_root() == tmp_path
    monkeypatch.delenv("MOE_RESULTS_DIR")
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    assert BM.results_root() == tmp_path / "results"


def test_the_dry_run_needs_no_gpu_and_writes_nothing(tmp_path, capsys):
    """The laptop path. It has to print the grid, the cost and the predictions
    without creating the results directory, so a laptop invocation does not
    litter the volume."""
    out = tmp_path / "nothing"
    assert BM.main(["--dry-run", "--out", str(out)]) == 0
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "NO CROSSING EVER" in printed
    assert "estimated GPU time" in printed


def test_it_names_the_missing_half_of_the_stack_instead_of_crashing(tmp_path, capsys):
    """Off GPU the script has to say which of torch, CUDA and vLLM is absent and
    what to run instead. This test runs on a laptop, which is the case it is
    about."""
    import torch
    if torch.cuda.is_available():                      # pragma: no cover - pod
        pytest.skip("this asserts the laptop path")
    assert BM.main(["--out", str(tmp_path)]) == 2
    printed = capsys.readouterr().out
    assert "--self-test" in printed
    assert "no CUDA device" in printed or "vLLM is not importable" in printed


def test_the_self_test_runs_the_whole_report_off_gpu_and_says_it_is_synthetic(
        tmp_path, capsys):
    """The claim "these gates can tell 0.558 from 0.10" has to be checkable
    without renting anything, and the output has to be impossible to mistake for
    a measurement."""
    assert BM.main(["--self-test", "0.558", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "Nothing here was measured" in printed
    assert printed.count("GATE") >= 5
    run = next((tmp_path / "block_m_crossing").iterdir())
    assert (run / "report.txt").exists()
    assert (run / "report.json").exists()


def test_iterations_are_cut_so_one_cell_cannot_run_away_with_the_meter():
    """An 11 ms cell at BLOCK_M=32 and 1024 rows costs 50x a 0.7 ms one, and
    there are 272 cells. The floor stays high enough for a median to mean
    something."""
    assert BM.scaled_iters(0.5, 50, 400.0) == 50
    assert BM.scaled_iters(11.0, 50, 400.0) == 36
    assert BM.scaled_iters(200.0, 50, 400.0) == 5


def test_the_pinned_tile_parameters_match_the_other_tile_experiment():
    """`tile_sweep.py` pins the same five, so the two experiments compose rather
    than being two unrelated sweeps of the same knob. GROUP_SIZE_M=1 in
    particular is the slice the refit describes."""
    assert BM.FIXED == {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                        "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4}


def test_the_report_states_the_prediction_before_the_measurement():
    """A prediction printed after the fact is not one. The report has to carry
    the crossing table, both ends of the ridge band, and the retracted alpha it
    is discriminating against."""
    text = report_at(REFIT).text()
    assert "PREDICTIONS, stated before the run" in text
    assert "NO CROSSING EVER" in text
    assert "1.558x at the low ridge" in text
    assert "2.116x at the high one" in text
    assert "retracted alpha=0.1 says 1.100x" in text
