"""The alpha refit must recover a known alpha, and must fail where the old one did.

The point of this file is not that the new estimator produces a number. It is
that the two estimators can be told apart on data whose answer is known, so the
claim "the 3.3x disagreement with TEMPO was an artefact of the ESTIMATOR, not of
the assumed tile" is checkable rather than asserted.

Three groups of tests:

  - synthetic data with alpha planted in it, where the group-intercept fit
    recovers it and the pooled-CV fit is shown to fail in a specific, named way;
  - the arithmetic that decides what alpha means, which is where a plausible
    wrong number would do the most damage;
  - the published corpus, where the reproduction of the ORIGINAL 151 rows and 27
    discriminating rows is what makes the comparison an attribution.

`scripts/alpha_refit.py` is loaded by path rather than imported, because
`scripts/` is not a package and never has been; the same shape any test of a
script in this repo would need.
"""
from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PUBLISHED = ROOT / "results" / "published"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "alpha_refit", ROOT / "scripts" / "alpha_refit.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, so a module that is not there yet makes the
    # decorator fail with an AttributeError about NoneType rather than anything
    # that names the real problem.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AR = _load_script()


def make_obs(*, ratio: float, extra_tiles: float, active: float = 8.0,
             per_expert: float = 1.0e8, compulsory: float = 1.0e9,
             tokens: int = 256, model: str = "mixtral-8x7b", block_m: int = 128,
             group_m: int = 1, gpu: str = "NVIDIA H200",
             impl: str = "vllm_fused_experts", l2_flush: bool = True,
             cuda_graph: bool = False, routing: str = "uniform",
             tile_columns: tuple = ()) -> AR.Observation:
    """One synthetic observation with `extra_tiles` M-tiles beyond the actives."""
    return AR.Observation(
        traffic_ratio=ratio, compulsory_bytes=compulsory,
        per_expert_bytes=per_expert, active_experts=active,
        m_tiles=active + extra_tiles, block_m=block_m, group_m=group_m,
        tile_provenance="vllm_tuned_derived", model=model, dtype="bf16",
        gpu=gpu, impl=impl, tokens=tokens, routing=routing,
        l2_flush=l2_flush, cuda_graph=cuda_graph, tile_columns=tile_columns)


def planted(alpha: float, *, levels=(1.0, 2.0, 4.0), spread=(0, 1, 2, 4),
            noise: float = 0.0, seed: int = 0,
            level_slope: float = 0.0) -> list[AR.Observation]:
    """Rows generated FROM the model, so the fit has a right answer to find.

    The shape mirrors the real data. One GROUP per token count, each with its own
    level, which is what the intercept absorbs. Inside a group the tile count
    varies by `spread`, which is what the routing regimes do in the corpus and is
    the ONLY variation the group-intercept fit can use. Across groups the mean
    tile count climbs, which is what growing the batch does.

    `level_slope` makes the group's level fall as the group's tile count rises,
    which is the real confound in miniature: dispatch cost amortises with batch
    while the tile count grows with batch, so level and tile move together for a
    reason that has nothing to do with alpha. It is a BETWEEN-group effect, so an
    intercept per group removes it exactly and a pooled fit cannot.
    """
    rng = random.Random(seed)
    rows = []
    for index, level in enumerate(levels):
        group_level = level * (1.0 + level_slope * index)
        for offset in spread:
            extra = index + offset
            value = group_level * (1.0 + alpha * extra * 0.1)
            if noise:
                value *= rng.lognormvariate(0.0, noise)
            rows.append(make_obs(ratio=value, extra_tiles=extra, tokens=index,
                                 per_expert=1.0e8, compulsory=1.0e9))
    return rows


# --------------------------------------------------------------------------
# the estimator recovers what was planted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.0, 0.10, 0.33, 0.56, 1.0])
def test_the_group_intercept_fit_recovers_an_alpha_that_was_planted_in_the_data(alpha):
    """Noiseless, three group levels, five tile counts. If the estimator cannot
    do this there is no point reading anything it says about a GPU."""
    assert AR.fit_alpha(planted(alpha)) == pytest.approx(alpha, abs=1e-3)


def test_it_still_recovers_alpha_when_every_group_sits_at_a_different_level():
    """The group levels here span 4x, which is more than the whole tile effect.
    A fit without intercepts would spend alpha on the levels instead."""
    rows = planted(0.5, levels=(0.4, 1.0, 1.6, 4.0))
    assert AR.fit_alpha(rows) == pytest.approx(0.5, abs=1e-3)


def test_it_recovers_alpha_through_lognormal_noise():
    fitted = AR.fit_alpha(planted(0.5, noise=0.05, seed=3,
                                  levels=tuple(0.5 + 0.3 * i for i in range(12))))
    assert fitted == pytest.approx(0.5, abs=0.05)


def test_the_pooled_cv_estimator_is_dragged_toward_zero_by_a_level_that_tracks_tiles():
    """THE MECHANISM BEHIND 0.10, ISOLATED. `level_slope` makes each group's level
    fall as its tile count rises, which is what dispatch amortisation does to
    `implied_traffic_ratio` across a token grid.

    With that confound present the pooled-CV estimator, which has no intercepts,
    reports a value far below the planted 0.5, while the group-intercept fit is
    untouched. That is the whole attribution: the old number is an estimator
    artefact, and the tile never entered it.
    """
    rows = planted(0.5, levels=tuple(1.0 for _ in range(8)), level_slope=-0.08)
    pooled, _, _ = AR.pooled_cv_alpha(rows)
    grouped = AR.fit_alpha(rows)
    assert grouped == pytest.approx(0.5, abs=1e-3)
    assert pooled < 0.2


def test_permuting_the_response_inside_each_group_destroys_the_signal():
    """The placebo, on data where the signal is known to be there. A fit that
    survived this would be fitting the group structure."""
    rows = planted(0.6, levels=tuple(1.0 + 0.2 * i for i in range(20)), noise=0.02)
    rng = random.Random(11)
    groups: dict = {}
    for row in rows:
        groups.setdefault(AR.cell_key(row), []).append(row)
    shuffled = []
    for members in groups.values():
        responses = [o.traffic_ratio for o in members]
        rng.shuffle(responses)
        shuffled.extend(o.__class__(**{**o.__dict__, "traffic_ratio": r})
                        for o, r in zip(members, responses, strict=True))
    assert abs(AR.fit_alpha(shuffled)) < 0.15
    assert AR.fit_alpha(rows) == pytest.approx(0.6, abs=0.05)


def test_a_pool_where_no_row_has_a_second_tile_cannot_identify_alpha_at_all():
    """Every row at `x = 0` makes the objective exactly flat, so whatever comes
    back is the search bound and not an estimate. `_split_line` refuses to print
    a number for such a split; this pins why."""
    rows = [make_obs(ratio=1.2 + 0.01 * i, extra_tiles=0, tokens=i % 3)
            for i in range(30)]
    assert all(not r.discriminating for r in rows)
    # The objective is exactly constant in alpha, so the search settles at
    # whichever end it started from. The value is the BOUND, and reporting it as
    # an estimate is the bug `_split_line` refuses to commit.
    assert AR.fit_alpha(rows) < -0.8
    assert "n/a" in AR._split_line("all flat", rows)


def test_the_fit_refuses_a_pool_it_cannot_fit():
    with pytest.raises(ValueError):
        AR.fit_alpha([make_obs(ratio=1.0, extra_tiles=1)])


def test_the_bootstrap_band_brackets_the_point_estimate_and_narrows_with_evidence():
    """A band that did not contain its own point estimate would be a bug in the
    resampling, and a band that did not narrow as groups are added would mean the
    cluster is not the unit of information it is claimed to be."""
    few = planted(0.5, noise=0.08, seed=1,
                  levels=tuple(0.5 + 0.2 * i for i in range(6)))
    many = planted(0.5, noise=0.08, seed=1,
                   levels=tuple(0.5 + 0.2 * i for i in range(60)))
    for rows in (few, many):
        lo, hi = AR.bootstrap_band(rows, draws=60, seed=0)
        assert lo <= AR.fit_alpha(rows) <= hi
    narrow = AR.bootstrap_band(many, draws=60, seed=0)
    wide = AR.bootstrap_band(few, draws=60, seed=0)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --------------------------------------------------------------------------
# the arithmetic that decides what alpha means
# --------------------------------------------------------------------------

def test_floating_point_dust_is_not_an_extra_tile():
    """`m_tiles_for_row` reconstructs the tile count as `total / (eff * block_m)`
    where `eff` was itself `total / (active * block_m)`, so the two divisions
    cancel to about 1e-13 rather than exactly. Before `TILE_EPSILON` that made 93
    single-tile rows "discriminating" at `x` about 1e-16, and the fit answered
    -0.850: the search bound, because a flat residual is minimised by running
    away."""
    dust = make_obs(ratio=1.2, extra_tiles=1e-13)
    assert not dust.discriminating
    assert dust.x == 0.0
    real = make_obs(ratio=1.2, extra_tiles=1e-3)
    assert real.discriminating


def test_the_router_weight_is_not_charged_per_m_tile():
    """A `__pipeline__` row records `covers = "all"`, which includes the router.
    The router weight is one dense `[E, H]` gate read once per layer however the
    rows are tiled, so multiplying it by an M-tile count charges a re-read that
    cannot happen. The two spans must give the same per-expert weight."""
    from moe.spec import MODEL_CONFIGS, BenchSpec
    spec = BenchSpec(MODEL_CONFIGS["deepseek-v3"], num_tokens=64, dtype="bf16")
    five_stage = AR.expert_weight_bytes(
        spec, "permute+up_gemm+act+down_gemm+unpermute")
    whole_layer = AR.expert_weight_bytes(spec, "all")
    assert five_stage == whole_layer > 0


def test_a_span_that_covers_no_gemm_carries_no_per_expert_weight():
    from moe.spec import MODEL_CONFIGS, BenchSpec
    spec = BenchSpec(MODEL_CONFIGS["mixtral-8x7b"], num_tokens=64, dtype="bf16")
    assert AR.expert_weight_bytes(spec, "router") == 0.0


def test_the_ai_cap_is_the_bound_findings_states_and_is_infinite_at_alpha_zero():
    """`2 BM / (alpha b)`. At alpha = 0 there is no re-read cost, intensity is
    unbounded and `2R/b` is exact, so infinity is the right answer and not a
    guard against division."""
    assert AR.ai_cap(128, 0.10) == pytest.approx(1280.0)
    assert AR.ai_cap(16, 0.10) == pytest.approx(160.0)
    assert AR.ai_cap(64, 0.33) == pytest.approx(193.9, abs=0.1)
    assert math.isinf(AR.ai_cap(128, 0.0))


def test_the_alpha_at_which_a_tile_stops_being_able_to_cross_is_the_caps_inverse():
    """The two forms have to agree, because the report prints one and reasons
    with the other."""
    for block_m in (16, 32, 64, 128):
        for ridge in AR.RIDGE_BAND:
            ceiling = AR.max_alpha_that_still_crosses(block_m, ridge)
            assert AR.ai_cap(block_m, ceiling) == pytest.approx(ridge)


def test_block_m_16_cannot_reach_the_ridge_at_any_alpha_this_study_has_fitted():
    """The consequence FINDINGS calls a knife edge. At the repo's old 0.10 the cap
    is 160 against a ridge band starting at 160.3, so it JUST fails; at anything
    larger it fails by a mile. Nothing about that conclusion needed the refit."""
    assert AR.ai_cap(16, AR.REPO_PUBLISHED_ALPHA) < AR.RIDGE_BAND[0]
    assert AR.ai_cap(16, AR.TEMPO_ALPHA) < AR.RIDGE_BAND[0]


def test_recounting_at_another_block_m_is_a_ceiling_and_never_a_rescale():
    """Tiles are `ceil(rows / BLOCK_M)` per expert, so halving the block does not
    double the count. `at_block_m` recounts from the stored load columns, and a
    row that cannot be counted at the new height returns None rather than a
    scaled guess."""
    columns = (("load_total_rows", "128"), ("load_active_experts", "8"),
               ("load_max_rows", "16"), ("load_tile_eff_bm64", "0.25"),
               ("load_tile_eff_bm128", "0.125"))
    obs = make_obs(ratio=1.2, extra_tiles=0.0, active=8.0, tile_columns=columns)
    assert obs.at_block_m(16).m_tiles == pytest.approx(8.0)
    assert obs.at_block_m(32).m_tiles == pytest.approx(8.0)
    # 16 rows in the biggest expert is more than a block of 8, so the per-expert
    # histogram would be needed and is not stored.
    assert obs.at_block_m(8) is None


# --------------------------------------------------------------------------
# the published corpus
# --------------------------------------------------------------------------

def published_csvs() -> list[Path]:
    return sorted(PUBLISHED.glob("*/run_*.csv"))


@pytest.fixture(scope="module")
def triton_pool():
    import collections
    return AR.collect(published_csvs(), collections.Counter())


@pytest.fixture(scope="module")
def original_pool():
    import collections
    paths = [p for p in published_csvs() if AR.ORIGINAL_ALPHA_ARM in p.parent.name]
    return AR.collect(paths, collections.Counter(), cutlass=True)


def test_the_original_151_rows_and_their_cv_column_reproduce_exactly(original_pool):
    """THE REPRODUCTION THE WHOLE ATTRIBUTION RESTS ON. The 2026-08-22 write-up
    fitted 151 unthrottled memory-bound L2-cold eager rows, of which 27 have
    `M_tiles(64) != active` and therefore discriminate, and reported CVs of 13.1%
    at alpha = 0 and 17.5% at alpha = 1.

    Both counts and both CVs come back. The MEAN ratio lands about 1% lower
    (1.65 against 1.67) for a reason that is not a discrepancy: the write-up
    divided by weight bytes at a fixed 4390.29 GB/s read ceiling, and
    `implied_traffic_ratio` divides by the row's full compulsory bytes at the
    row's own triad ceiling.
    """
    basis = [o for o in original_pool if o.l2_flush and not o.cuda_graph]
    assert len(basis) == 151
    assert sum(1 for o in basis if o.discriminating) == 27
    assert AR._cv(basis, 0.0) == pytest.approx(0.131, abs=0.002)
    assert AR._cv(basis, 1.0) == pytest.approx(0.175, abs=0.004)
    assert AR._mean_ratio(basis, 0.0) == pytest.approx(1.66, abs=0.02)


def test_the_original_objective_barely_moves_across_the_whole_disputed_range(
        original_pool):
    """0.10 and 0.33 differ by 3.3x and the objective separating them changes by
    about one part in a hundred. A minimum that shallow is a statement about the
    estimator rather than about the hardware, and it is why the published figure
    could sit next to TEMPO's without either being obviously wrong."""
    basis = [o for o in original_pool if o.l2_flush and not o.cuda_graph]
    at_zero = AR._cv(basis, 0.0)
    fitted, best, _ = AR.pooled_cv_alpha(basis)
    assert fitted < 0.15
    assert (at_zero - best) / at_zero < 0.02


def test_the_same_151_rows_give_a_wholly_different_alpha_under_group_intercepts(
        original_pool):
    """Same rows, same OBSERVED CUTLASS tile, nothing derived: only the estimator
    changes, and the answer moves from about 0.06 to about 0.5. That is the
    finding -- the disagreement with TEMPO was never about the tile."""
    basis = [o for o in original_pool if o.l2_flush and not o.cuda_graph]
    assert AR.fit_alpha(basis) > 4 * AR.REPO_PUBLISHED_ALPHA


def test_the_derived_pool_is_the_size_the_report_claims(triton_pool):
    """A pin on the pool, so a change in `crossing.m_tiles_for_row`,
    `published.filter_superseded` or the tile resolver shows up here rather than
    silently moving a published alpha."""
    assert len(triton_pool) == 10_813
    assert sum(1 for o in triton_pool if o.discriminating) == 3_124
    assert {o.dtype for o in triton_pool} == {"bf16"}


def test_no_sglang_row_is_ever_in_the_pool(triton_pool):
    """SGLang ships its own tuned config tree and nothing here models it. Its
    rows are a third of the corpus, so admitting them under vLLM's derived tile
    would be the largest single wrong number this script could produce."""
    assert not any("sglang" in o.impl for o in triton_pool)
    assert {o.impl for o in triton_pool} <= set(AR.VLLM_IMPLS)


def test_alpha_on_the_published_rows_is_far_above_both_disputed_values(triton_pool):
    """The headline, banded rather than pinned to three decimals: the exact value
    depends on the pool and the pool will change, but "about half a fresh weight
    read, five times the repo's own published figure and well above TEMPO's" is
    the claim, and a change that moves it out of this band is a change worth
    stopping on."""
    alpha = AR.fit_alpha(triton_pool)
    assert 0.45 < alpha < 0.70
    assert alpha > 3 * AR.REPO_PUBLISHED_ALPHA
    assert alpha > AR.TEMPO_ALPHA


def test_the_answer_is_stable_across_the_four_timing_modes(triton_pool):
    """Eager and graph differ by up to 2.87x in raw time on these very cells, and
    an alpha that moved with the timing mode would be measuring launch overhead.
    It does not: all four modes land inside a narrow band."""
    fits = []
    for flush in (True, False):
        for graph in (False, True):
            subset = [o for o in triton_pool
                      if o.l2_flush == flush and o.cuda_graph == graph]
            fits.append(AR.fit_alpha(subset))
    assert max(fits) - min(fits) < 0.15
    assert min(fits) > 0.4


def test_group_size_m_32_and_64_carry_no_discriminating_rows_at_all(triton_pool):
    """THE PER-GROUP_SIZE_M TEST IS UNDERPOWERED, AND STRUCTURALLY SO.
    GROUP_SIZE_M is the swizzle width and therefore the parameter alpha should
    depend on, but 32 and 64 appear only on tuned fp8/low-M mixtral and qwen2
    entries where every expert fits inside one tile. So the split can be reported
    and cannot be answered from these rows: it needs a run that varies
    GROUP_SIZE_M at a fixed batch, which `override_config` can do."""
    by_group = {}
    for obs in triton_pool:
        by_group.setdefault(obs.group_m, []).append(obs)
    assert set(by_group) == {1, 16, 32, 64}
    for group_m in (32, 64):
        assert not any(o.discriminating for o in by_group[group_m])
    for group_m in (1, 16):
        assert sum(1 for o in by_group[group_m] if o.discriminating) > 100


def test_half_the_derived_pool_resolves_to_a_tile_that_cannot_constrain_anything(
        triton_pool):
    """BLOCK_M 16 and 32 have no discriminating rows either, and for a different
    reason: at a block size the schema stores no tile efficiency for, the count is
    reconstructed, and the reconstruction is only valid while every expert fits in
    one tile. The rows that survive at 16 are exactly the rows where 16 costs
    nothing. More than half the admitted rows are in that state."""
    low = [o for o in triton_pool if o.block_m in (16, 32)]
    assert len(low) > len(triton_pool) / 2
    assert not any(o.discriminating for o in low)


def test_the_memory_bound_filter_excludes_rows_that_are_memory_bound(triton_pool):
    """THE FILTER THIS FIT DEPENDS ON, CHECKED. `implied_traffic_ratio` is written
    only where the driver called the cell memory bound on its COMPULSORY
    intensity. That has no false positives, because compulsory intensity is an
    upper bound on the true one. It has false NEGATIVES, and they are the
    many-tile rows: over a thousand rows are memory bound once the tile
    correction is applied and carry no column to say so, which is exactly the
    evidence this fit is short of."""
    census = AR.count_excluded_memory_bound(published_csvs(),
                                            AR.fit_alpha(triton_pool))
    wrongly = next(v for k, v in census.items() if k.startswith("NO COLUMN BUT"))
    carried = next(v for k, v in census.items() if k.startswith("memory-bound and"))
    assert wrongly > 1000
    assert 0.05 < wrongly / (wrongly + carried) < 0.20


def test_the_report_runs_end_to_end_over_the_published_corpus(capsys):
    """Every section, on the real inputs, because most of the ways this script
    could be wrong live in the reporting rather than in the fit: a split with no
    rows, a division by an empty pool, an f-string that never gets formatted."""
    code = AR.main([str(p) for p in published_csvs()]
                   + ["--bootstrap", "5", "--original-estimator", "--adversarial"])
    assert code == 0
    out = capsys.readouterr().out
    assert "DERIVED" in out
    assert "alpha = " in out
    assert "cluster-bootstrap band" in out
    assert "n=151" in out
    assert "against my own fit" in out
    assert "NEVER crosses" in out


def test_the_report_says_so_rather_than_dividing_by_zero_on_an_empty_input(tmp_path,
                                                                          capsys):
    """A CSV with a header and no rows is what a killed pod leaves behind, and a
    report that crashed on it would be read as a broken script rather than an
    empty arm."""
    from moe.bench import schema as SC
    empty = tmp_path / "run_empty.csv"
    empty.write_text(",".join(SC.COLUMNS) + "\n")
    assert AR.main([str(empty)]) == 1
    assert "no fit to report" in capsys.readouterr().out
