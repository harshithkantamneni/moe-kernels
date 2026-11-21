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

from moe.bench import schema as SC  # noqa: E402


def make_obs(*, ratio: float, extra_tiles: float, active: float = 8.0,
             per_expert: float = 1.0e8, compulsory: float = 1.0e9,
             tokens: int = 256, model: str = "mixtral-8x7b", block_m: int = 128,
             group_m: int = 1, gpu: str = "NVIDIA H200",
             impl: str = "vllm_fused_experts", l2_flush: bool = True,
             cuda_graph: bool = False, routing: str = "uniform",
             tile_columns: tuple = (),
             instrument: str = SC.LEGACY_INSTRUMENT) -> AR.Observation:
    """One synthetic observation with `extra_tiles` M-tiles beyond the actives.

    `instrument` IS PASSED, and by this helper rather than by the dataclass.
    The field used to default to the retired apparatus for the convenience of
    exactly this call site, and that default then labelled the one non-test
    caller that builds an Observation by hand -- `group_m_alpha_sweep.observation`
    -- as retired-instrument data while its own provenance block wrote
    `TIMING_BASIS`. A helper stating what it means costs one keyword; a default
    stating it on everyone's behalf cost a mislabel nobody could see.
    """
    return AR.Observation(
        traffic_ratio=ratio, compulsory_bytes=compulsory,
        per_expert_bytes=per_expert, active_experts=active,
        m_tiles=active + extra_tiles, block_m=block_m, group_m=group_m,
        tile_provenance="vllm_tuned_derived", model=model, dtype="bf16",
        gpu=gpu, impl=impl, tokens=tokens, routing=routing,
        l2_flush=l2_flush, cuda_graph=cuda_graph, tile_columns=tile_columns,
        instrument=instrument)


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
    """`2 BM / (alpha b)`, an UPPER BOUND on the cap. At alpha = 0 the formula
    names no re-read cost and no other traffic, so infinity is the right
    reading of the formula and not a guard against division."""
    assert AR.ai_cap(128, 0.10) == pytest.approx(1280.0)
    assert AR.ai_cap(16, 0.10) == pytest.approx(160.0)
    assert AR.ai_cap(64, 0.33) == pytest.approx(193.9, abs=0.1)
    assert math.isinf(AR.ai_cap(128, 0.0))


def _card_ridges() -> dict[str, float]:
    """Each committed card's own ridge, the numbers the caps are judged against.
    Not the withdrawn 160.3-176.2, which these tests iterated over until
    2026-09-03 and which belongs to no card."""
    return {card: ridge for card, ridge, _band in AR.card_ridge_bands()}


def test_the_alpha_at_which_a_tile_stops_being_able_to_cross_is_the_caps_inverse():
    """The two forms have to agree, because the report prints one and reasons
    with the other."""
    ridges = _card_ridges()
    assert set(ridges) == {"nvidia_h200", "nvidia_a100_sxm4_80gb"}
    for block_m in (16, 32, 64, 128):
        for ridge in ridges.values():
            ceiling = AR.max_alpha_that_still_crosses(block_m, ridge)
            assert AR.ai_cap(block_m, ceiling) == pytest.approx(ridge)


def test_block_m_16_cannot_reach_either_cards_ridge_at_any_alpha_this_study_has_fitted():
    """The consequence FINDINGS calls a knife edge, on each card's OWN ridge.
    At the repo's retracted 0.10 the upper bound is 160, and the 2026-09-09
    recalibration moved the H200's ridge from 162.8 to 152.8, so that bound
    now CLEARS both cards rather than only the A100: on the uncorrected
    reading the retracted alpha caps BLOCK_M=16 above every ridge this study
    has measured, which is the knife edge the withdrawn 160.3 hid, worse than
    when only one card was on the wrong side of it. The corrected bracket
    settles neither card: [127, 159] straddles 145.8 and 152.8 alike, so both
    verdicts at 0.10 are UNDECIDED, which is the honest form of it. What
    survives untouched by the recalibration is the part the claim rests on:
    at TEMPO's 0.33 and at the refit's 0.558 the upper bound alone fails both
    cards by a mile."""
    ridges = _card_ridges()
    cap = AR.ai_cap(16, AR.REPO_PUBLISHED_ALPHA)
    assert all(cap > ridge for ridge in ridges.values())
    lo, hi = AR.corrected_cap_bracket(16, AR.REPO_PUBLISHED_ALPHA)
    assert all(lo < ridge < hi for ridge in ridges.values())
    assert AR.cap_bracket_verdict(lo, hi, [139.6, 149.3]).startswith("undecided")
    for ridge in ridges.values():
        assert AR.ai_cap(16, AR.TEMPO_ALPHA) < ridge
        assert AR.ai_cap(16, 0.558) < ridge


def test_the_cap_table_prints_the_correction_as_a_bracket_and_scores_on_it(capsys):
    """Retraction (a): `2 BM / (alpha b)` from a ladder alpha is HIGH by
    (1 + phi + delta). The table prints that factor as a bracket over the
    unmeasured alpha_a, the corrected cap beside the uncorrected one, and each
    card's verdict on the bracket: one word where both ends agree, "undecided"
    where alpha_a would decide. At BLOCK_M=128 the two ends straddle both
    cards' bands, so the verdict the old table printed as a point is the
    bracket's honest answer."""
    AR.print_ai_cap_table(0.558)
    out = capsys.readouterr().out
    assert "upper bound" in out and "(1+phi+delta) bracket" in out
    assert "alpha_a in [0, 1] at delta = 0" in out
    rows = {int(line.split("|")[1]): line for line in out.splitlines()
            if line.strip().startswith("| ") and line.split("|")[1].strip().isdigit()}
    assert set(rows) == {16, 32, 64, 128, 256}
    assert "undecided" in rows[128] and "undecided" in rows[256]
    assert "NEVER crosses" in rows[16] and "undecided" not in rows[16]
    # The factor bracket is what ai_model says it is, on the shapes named.
    lo, hi = AR.lin_overstatement_bracket(128)
    assert 1.03 < lo < 1.05 and 3.0 < hi < 3.2
    c_lo, c_hi = AR.corrected_cap_bracket(128, 0.558)
    assert c_lo == pytest.approx(AR.ai_cap(128, 0.558) / hi)
    assert c_hi == pytest.approx(AR.ai_cap(128, 0.558) / lo)
    assert c_lo < 139.6 < 165.6 < c_hi


def test_the_bracket_verdict_has_a_fourth_word_only_when_the_ends_disagree():
    band = [152.1, 165.6]
    assert AR.cap_bracket_verdict(100.0, 140.0, band) == "NEVER crosses"
    assert AR.cap_bracket_verdict(170.0, 300.0, band) == "crosses"
    assert AR.cap_bracket_verdict(100.0, 300.0, band).startswith("undecided")
    assert "alpha_a=1" in AR.cap_bracket_verdict(100.0, 300.0, band)
    lo, hi = AR.corrected_cap_bracket(128, 0.0)
    assert math.isinf(lo) and math.isinf(hi)


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
    # 10,813 until 2026-09-01, when the alpha-0558 arm added 368 admissible
    # rows. The number is asserted rather than computed on purpose: it is
    # what the published refit quotes, so it has to move deliberately and
    # be re-quoted, not drift under the text that cites it.
    assert len(triton_pool) == 11_181
    assert sum(1 for o in triton_pool if o.discriminating) == 3_181
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


# --------------------------------------------------------------------------
# which instrument measured the rows
# --------------------------------------------------------------------------

#: A source arm small enough to copy and discriminating end to end, so a
#: two-instrument fixture is a real pool rather than a hand-built dict.
FIXTURE_ARM = "2026-08-28-nvidia_h200-ridge-resolution/run_ridgedeepseek_vllm.csv"


def two_instrument_corpus(tmp_path):
    """One published arm, twice: once as it stands, once restamped at v5.

    The v5 copy is the SAME rows with the instrument column filled in, so the
    only thing that differs between the two halves is the apparatus each claims.
    That is exactly the pool the refusal exists for, and building it out of real
    rows means the gate is exercised through `collect`, `read_csv` and the tile
    resolver rather than around them.
    """
    import csv

    from moe.bench import schema as SC

    raw = list(csv.DictReader((ROOT / "results" / "published" / FIXTURE_ARM)
                              .open(newline="")))
    for arm, version in (("armA", None), ("armB", 5)):
        directory = tmp_path / arm
        directory.mkdir()
        with (directory / "run_x.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SC.COLUMNS, restval="")
            writer.writeheader()
            for r in raw:
                row = {k: v for k, v in r.items() if k in SC.COLUMNS}
                if version:
                    row["schema_version"] = version
                    row["instrument"] = "queue-deep/l2-flush/clock-under-load/v2"
                    for name in SC.TIMING_VERDICT_COLUMNS:
                        row[name] = SC.VERDICT_OK
                writer.writerow(row)
    return sorted(tmp_path.glob("*/run_*.csv"))


def test_every_published_row_reads_as_the_retired_instrument(triton_pool):
    """THE FACT THE WHOLE VERSION WAS CUT FOR, pinned. All 100,144 published rows
    came through `driver.py`, which timed on `time_eager`/`time_graph` until
    2026-09-02, so the headline alpha is a retired instrument's number and the
    report has to say so rather than leave a reader to assume otherwise."""
    from moe.bench import schema as SC
    assert set(AR.instrument_mix(triton_pool)) == {SC.LEGACY_INSTRUMENT}


def test_an_intercept_can_never_span_two_instruments():
    """alpha is identified WITHIN a group intercept, so an instrument left out of
    the key would be paid for by alpha the way the batch trend was before token
    count entered it."""
    a = make_obs(ratio=1.5, extra_tiles=2)
    b = AR.dataclasses.replace(a, instrument="queue-deep/v2")
    assert AR.cell_key(a) != AR.cell_key(b)
    assert a.context != b.context


def test_the_clock_gate_asks_the_instrument_that_wrote_the_row():
    """The two apparatus have NO flag in common, and reading one's answer out of
    the other's column is how a filter becomes a no-op unnoticed. The retired
    flag is a comparison of two IDLE-instant samples and the instrument never
    takes one, so it is not this instrument's evidence even now that
    `driver._apply_kernel_timing` writes the column: the gate reads the verdicts,
    which say WHICH check failed."""
    from moe.bench import schema as SC

    old = {"throttled": "True"}
    assert AR.clock_gate(old, SC.LEGACY_INSTRUMENT).startswith("throttled")
    assert AR.clock_gate({"throttled": "False"}, SC.LEGACY_INSTRUMENT) == ""

    new = {"instrument": "queue-deep/v2", "clock_level_ok": "ok",
           "clock_drift_ok": "ok", "host_bound_ok": "ok", "throttled": "True"}
    assert AR.clock_gate(new, "queue-deep/v2") == "", (
        "the retired flag is not this instrument's evidence and must not gate it")
    assert "clock_drift_ok" in AR.clock_gate(dict(new, clock_drift_ok="failed"),
                                             "queue-deep/v2")
    assert "host_bound_ok" in AR.clock_gate(dict(new, host_bound_ok="failed"),
                                            "queue-deep/v2")
    # UNDETERMINED IS KEPT. The check could not be run -- no NVML, a trial too
    # short for the poller -- which is not evidence against the number, and
    # folding it into a failure would discard every row measured in a container
    # that forbids NVML.
    assert AR.clock_gate(dict(new, clock_level_ok="undetermined"),
                         "queue-deep/v2") == ""


def test_an_untimed_row_is_named_by_the_gate_and_not_admitted_by_it():
    """A ROW NOTHING MEASURED WAS BEING ADMITTED. The driver stamps
    `NO_INSTRUMENT` on every cell it declines or fails and leaves the three
    verdict columns at their `Row` defaults, which are the WORD "undetermined".
    That fell into the v5 branch below, read three of them, and returned "" --
    ADMIT. It reached no fit only because `collect` and
    `count_excluded_memory_bound` both drop `ms_p50 <= 0` a few lines earlier,
    an incidental filter doing a gate's job, which is the accident this
    function's own first paragraph exists to prevent."""
    from moe.bench import schema as SC

    untimed = {"instrument": SC.NO_INSTRUMENT,
               "clock_level_ok": SC.VERDICT_UNDETERMINED,
               "clock_drift_ok": SC.VERDICT_UNDETERMINED,
               "host_bound_ok": SC.VERDICT_UNDETERMINED,
               "ms_p50": "0.0", "throttled": "False"}
    reason = AR.clock_gate(untimed, SC.instrument_of(untimed))
    assert reason.startswith("never timed"), reason
    # And the reader underneath refuses too, so the two cannot drift apart: a
    # future gate that forgets this branch gets an exception, not an admission.
    with pytest.raises(SC.TimingInstrumentUnrecorded):
        SC.timing_verdict(untimed, "clock_level_ok")


def test_a_pool_that_mixes_two_instruments_is_refused_rather_than_fitted(
        tmp_path, capsys):
    """REFUSED (exit 2), not averaged and not silently preferred. The counts are
    printed either way, so the reader is told what the pool held even though no
    alpha is."""
    csvs = two_instrument_corpus(tmp_path)
    assert AR.main([str(p) for p in csvs] + ["--bootstrap", "3"]) == AR.REFUSED
    out = capsys.readouterr().out
    assert "TWO OR MORE INSTRUMENTS IN ONE POOL" in out
    assert "REFUSED" in out
    assert "queue-deep/l2-flush/clock-under-load/v2" in out
    assert "alpha = " not in out, "a refused pool must not also print a number"


def test_the_refusal_is_overridable_and_the_override_says_so(tmp_path, capsys):
    """The honest second-best. It fits, and every headline number carries the mix
    it was fitted over."""
    csvs = two_instrument_corpus(tmp_path)
    code = AR.main([str(p) for p in csvs]
                   + ["--bootstrap", "3", "--pool-instruments"])
    assert code == 0
    out = capsys.readouterr().out
    assert "POOLED ON PURPOSE" in out
    assert "alpha = " in out
    assert "measured on: " in out
    assert "between instrument" in out


def test_the_pinned_set_states_the_instrument_of_the_set_it_fitted(capsys):
    """M2, checked on the real manifest: `--pinned-set` is the run whose numbers
    `docs/FINDINGS.md` quotes, and a number whose instrument is not printed
    beside it is not comparable with anything else in this study."""
    assert AR.main(["--pinned-set", "--bootstrap", "3"]) == 0
    out = capsys.readouterr().out
    assert "## the instrument that measured the fitted set" in out
    assert "ONE INSTRUMENT, AND IT IS THE RETIRED ONE" in out
    assert ("measured on: 10813 rows on time_eager+time_graph/"
            "idle-instant-clock/pre-v5") in out


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


# --- an Observation that does not say what timed it -------------------------

def test_the_instrument_field_defaults_to_a_refusal_not_to_an_apparatus():
    """THE DEFAULT THAT MISLABELLED THE ONE NON-TEST CALLER. It used to be
    `schema.LEGACY_INSTRUMENT`, justified as convenience for synthetic
    observations -- and `scripts/group_m_alpha_sweep.py:718` builds an
    Observation from cells measured by `time_kernel`, passes seventeen keywords
    and not that one, in a script whose own provenance block writes
    `instrument=timing.TIMING_BASIS`. No number moved, because the mislabel was
    uniform; the confusion this module exists to end came back through a
    default."""
    field = AR.Observation.__dataclass_fields__["instrument"]
    assert field.default == AR.UNSTATED_INSTRUMENT
    assert field.default != SC.LEGACY_INSTRUMENT
    bare = AR.dataclasses.replace(make_obs(ratio=1.5, extra_tiles=2),
                                  instrument=field.default)
    assert AR.instrument_mix([bare]) == {AR.UNSTATED_INSTRUMENT: 1}


def test_an_unstated_instrument_is_refused_and_pooling_does_not_lift_it(capsys):
    """`--pool-instruments` is a decision to weigh two apparatus a reader can
    name. An absent fact is not one of them: alpha is identified inside a group
    intercept keyed on the instrument, so an unstated one either invents a
    cluster or collapses two real ones, and nothing here can say which."""
    pool = [AR.dataclasses.replace(make_obs(ratio=1.5, extra_tiles=n),
                                   instrument=AR.UNSTATED_INSTRUMENT)
            for n in (0, 1, 2, 4)]
    for pooling in (False, True):
        assert AR._report_instruments(pool, pooling) is False
        assert "did not say what timed them" in capsys.readouterr().out


def test_a_stated_instrument_still_passes_the_same_gate(capsys):
    """The PASS branch, so the refusal above is a gate and not a wall."""
    stated = [make_obs(ratio=1.5, extra_tiles=n, instrument=SC.LEGACY_INSTRUMENT)
              for n in (0, 1, 2, 4)]
    assert AR._report_instruments(stated, False) is True
    assert "ONE INSTRUMENT, AND IT IS THE RETIRED ONE" in capsys.readouterr().out
    on_bench = [AR.dataclasses.replace(o, instrument="queue-deep/v2")
                for o in stated]
    assert AR._report_instruments(on_bench, False) is True
    assert AR._report_instruments(stated + on_bench, False) is False
    assert AR._report_instruments(stated + on_bench, True) is True


# --------------------------------------------------------------------------
# the LEVEL verdict has a side, and the gate reads it (F1 of the 2026-09-08
# pre-pod verdict; instance fifteen of a fix landing at one of two call sites)
# --------------------------------------------------------------------------

#: The instrument's own reference and boosted clock on the committed H200
#: calibration: 1515 MHz is the bf16 GEMM plateau LEVEL is scored against,
#: 1980 MHz is what the card runs at under memory load for the whole settle.
REFERENCE_MHZ = 1515.0
BOOSTED_MHZ = 1980.0
THROTTLED_MHZ = 1400.0

#: The instrument that writes the side, `timing.TIMING_BASIS` spelled out so
#: this file does not import torch to name it.
ON_BENCH = "queue-deep/l2-flush/clock-under-load/v3"


def high_row(**over) -> dict:
    """A v6 row whose LEVEL failed on the HIGH side and nothing else failed."""
    row = {"clock_level_ok": "failed", "clock_level_side": "high",
           "clock_drift_ok": "ok", "host_bound_ok": "ok",
           "throttled": "False", "host_bound": "False"}
    row.update(over)
    return row


def test_the_side_words_are_the_instruments_own():
    """`alpha_refit` spells the two side words itself because `timing.py`
    imports torch at module scope; the spelling must be the instrument's or the
    gate reads a word it never wrote."""
    from moe.bench import timing as T
    assert AR.LEVEL_LOW == T.LEVEL_LOW
    assert AR.LEVEL_HIGH == T.LEVEL_HIGH
    assert AR.LEVEL_SIDES == {"", T.LEVEL_LOW, T.LEVEL_HIGH}


def test_the_clock_gate_admits_a_high_side_level_failure_and_excludes_low():
    """THE DEFECT: a memory-shaped cell boosted to 1980 MHz against the 1515
    reference fails LEVEL with side "high", and until 2026-09-08 this gate
    dropped it on the bare verdict, which on an H200 is every memory-bound
    cell, which is every cell alpha is identified on. HIGH is admitted (""),
    LOW is excluded with a reason that says LOW, and both are planted."""
    assert AR.clock_gate(high_row(), ON_BENCH) == ""
    low = AR.clock_gate(high_row(clock_level_side="low", throttled="True"), ON_BENCH)
    assert low, "a LOW-side LEVEL failure is the throttle the flag was built for"
    assert "failed LOW" in low and low.startswith("under-load check failed")
    # DRIFT still excludes a boosted row: the samples were not taken at one
    # clock, whichever side the median landed on.
    drifted = AR.clock_gate(high_row(clock_drift_ok="failed"), ON_BENCH)
    assert "clock_drift_ok" in drifted and "clock_level_ok" not in drifted
    hosted = AR.clock_gate(high_row(host_bound_ok="failed"), ON_BENCH)
    assert "host_bound_ok" in hosted
    # A LEVEL failure with NO side is not read as HIGH: a pre-v6 LEVEL was
    # one-sided and could only fail low. Both spellings of "no side".
    for absent in ("", SC.UNRECORDED):
        reason = AR.clock_gate(high_row(clock_level_side=absent), ON_BENCH)
        assert "no side recorded" in reason, reason
    reason = AR.clock_gate({k: v for k, v in high_row().items()
                            if k != "clock_level_side"}, ON_BENCH)
    assert "no side recorded" in reason
    # A word the instrument never wrote is refused, not admitted by falling
    # through every branch.
    with pytest.raises(ValueError, match="clock_level_side 'sideways'"):
        AR.clock_gate(high_row(clock_level_side="sideways"), ON_BENCH)
    # The retired instrument has no side and is untouched by any of this.
    assert AR.clock_gate({"throttled": "True"}, SC.LEGACY_INSTRUMENT).startswith(
        "throttled")


def test_roof_fractions_returns_none_and_a_reason_rather_than_a_zero():
    """0.0 is the driver's "not scored" for both roof columns, and a median
    over it is a fraction of nothing."""
    scored = {"schema_version": "6", "pct_of_achieved_tflops": "6.0",
              "roof_at_cell_clock_tflops": "914.85",
              "pct_of_roof_at_cell_clock": "4.59", "roof_note": ""}
    assert AR.roof_fractions(scored) == (6.0, 4.59, "")
    refused = dict(scored, roof_at_cell_clock_tflops="0.0",
                   pct_of_roof_at_cell_clock="0.0",
                   roof_note="the reference is graded 'idle-scalar'")
    fixed, cell, why = AR.roof_fractions(refused)
    assert (fixed, cell) == (6.0, None)
    assert why.startswith("the reference is graded")
    predates = {"schema_version": "3", "pct_of_achieved_tflops": "6.0"}
    assert AR.roof_fractions(predates) == (6.0, None, AR.ROOF_PREDATES)
    stamped = dict(predates, roof_at_cell_clock_tflops=SC.UNRECORDED,
                   pct_of_roof_at_cell_clock=SC.UNRECORDED,
                   roof_note=SC.UNRECORDED)
    assert AR.roof_fractions(stamped) == (6.0, None, AR.ROOF_PREDATES)


#: Token counts of the fixture arm's memory-bound vLLM rows (4096, 4608 and
#: 5120), split by side so an admitted observation's `tokens` says which side
#: it was planted on and the expected counts come off an UNPLANTED baseline
#: rather than off a guess. 5120 is left level on purpose, so the pool holds
#: both sides and the side split has two lines to print.
PLANT_HIGH_TOKENS = frozenset({4096})
PLANT_LOW_TOKENS = frozenset({4608})


def v6_corpus(tmp_path, *, plant: bool):
    """The fixture arm restamped at v6, on the instrument, every verdict ok.

    With `plant`, every memory-bound vLLM row at a `PLANT_HIGH_TOKENS` count is
    a cell that boosted to 1980 MHz against the 1515 reference (LEVEL failed
    HIGH, roof rescaled 1.307x, `throttled` False as the driver writes it) and
    every one at a `PLANT_LOW_TOKENS` count sat at 1400 MHz (LEVEL failed LOW,
    `throttled` True). Real rows through `read_csv`, `collect` and the tile
    resolver, so the gate is exercised where it sits and not around it.
    """
    import csv

    from moe.bench.roofline import roof_at_clock

    raw = list(csv.DictReader((ROOT / "results" / "published" / FIXTURE_ARM)
                              .open(newline="")))
    directory = tmp_path / ("planted" if plant else "baseline")
    directory.mkdir()
    counts = {"high": 0, "low": 0}
    with (directory / "run_x.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SC.COLUMNS, restval="")
        writer.writeheader()
        for r in raw:
            row = {k: v for k, v in r.items() if k in SC.COLUMNS}
            row["schema_version"] = SC.SCHEMA_VERSION
            row["instrument"] = ON_BENCH
            for name in SC.TIMING_VERDICT_COLUMNS:
                row[name] = SC.VERDICT_OK
            row["throttled"] = "False"
            peak = float(row.get("achieved_peak_tflops") or 0.0)
            tflops = float(row.get("tflops") or 0.0)
            load = REFERENCE_MHZ
            memory_bound = (row.get("impl") == "vllm_fused_experts"
                            and float(row.get("implied_traffic_ratio") or 0.0) > 0
                            and float(row.get("ms_p50") or 0.0) > 0
                            and row.get("correctness_passed") == "True")
            tokens = int(float(row["num_tokens"]))
            if plant and memory_bound and tokens in PLANT_HIGH_TOKENS:
                row["clock_level_ok"] = SC.VERDICT_FAILED
                row["clock_level_side"] = AR.LEVEL_HIGH
                load = BOOSTED_MHZ
                counts["high"] += 1
            elif plant and memory_bound and tokens in PLANT_LOW_TOKENS:
                row["clock_level_ok"] = SC.VERDICT_FAILED
                row["clock_level_side"] = AR.LEVEL_LOW
                row["throttled"] = "True"
                load = THROTTLED_MHZ
                counts["low"] += 1
            row["sm_clock_load_mhz"] = load
            row["reference_clock_mhz"] = REFERENCE_MHZ
            row["reference_clock_source"] = "test: under-load bf16 median"
            roof = roof_at_clock(peak, REFERENCE_MHZ, load) if peak > 0 else None
            if roof:
                row["roof_at_cell_clock_tflops"] = roof
                row["pct_of_roof_at_cell_clock"] = 100.0 * tflops / roof
                row["roof_note"] = ""
            else:
                row["roof_at_cell_clock_tflops"] = 0.0
                row["pct_of_roof_at_cell_clock"] = 0.0
                row["roof_note"] = "no measured compute ceiling for dtype"
            writer.writerow(row)
    return sorted(directory.glob("run_*.csv")), counts


def test_collect_keeps_the_boosted_rows_and_drops_the_throttled_ones(tmp_path):
    """Planted through the real path. Every admitted observation at a HIGH
    token count carries the mark, the LOW rows are named in the census by
    their side and are absent from the pool, and the admitted count is the
    baseline's minus exactly the LOW rows the baseline admitted."""
    import collections

    base_paths, _ = v6_corpus(tmp_path, plant=False)
    baseline = AR.collect(base_paths, collections.Counter())
    base_high = [o for o in baseline if o.tokens in PLANT_HIGH_TOKENS]
    base_low = [o for o in baseline if o.tokens in PLANT_LOW_TOKENS]
    assert base_high and base_low, "the fixture must admit rows on both sides"
    assert all(o.clock_level_side == "" for o in baseline)

    paths, counts = v6_corpus(tmp_path, plant=True)
    census: collections.Counter = collections.Counter()
    pool = AR.collect(paths, census)
    high = [o for o in pool if o.clock_level_side == AR.LEVEL_HIGH]
    assert len(high) == len(base_high), "a boosted row must be KEPT"
    assert all(o.tokens in PLANT_HIGH_TOKENS for o in high)
    assert not any(o.clock_level_side == AR.LEVEL_LOW for o in pool)
    assert not any(o.tokens in PLANT_LOW_TOKENS for o in pool)
    assert len(pool) == len(baseline) - len(base_low)
    low_reason = next(k for k in census if "failed LOW" in k)
    assert census[low_reason] == counts["low"]
    # The mark carries the corrected fraction beside the fixed one, and on a
    # boosted row the two differ by the clock ratio exactly.
    for o in high:
        assert o.pct_cell_clock_roof is not None and o.roof_note == ""
        assert o.pct_fixed_roof / o.pct_cell_clock_roof == pytest.approx(
            BOOSTED_MHZ / REFERENCE_MHZ, rel=1e-9)
    # `--include-throttled` re-admits the LOW rows and they keep their side,
    # so the report can say they are in the pool and why.
    readmitted = AR.collect(paths, collections.Counter(), include_throttled=True)
    assert sum(o.clock_level_side == AR.LEVEL_LOW for o in readmitted) == len(base_low)


def test_the_report_counts_the_high_side_rows_and_prints_both_roof_fractions(
        tmp_path, capsys):
    """THE READER FOR `pct_of_roof_at_cell_clock` (F4). Printed beside the
    fixed-roof fraction, with the count of admitted rows it exists on, and the
    HIGH-side count with the note that the fixed figure is the one not to
    quote. The side split under "is alpha a scalar?" is the check on the
    pooling argument in `clock_gate`'s docstring."""
    import collections

    paths, _ = v6_corpus(tmp_path, plant=True)
    pool = AR.collect(paths, collections.Counter())
    n_high = sum(o.clock_level_side == AR.LEVEL_HIGH for o in pool)
    assert AR.main([str(p) for p in paths] + ["--bootstrap", "3"]) == 0
    out = capsys.readouterr().out
    assert "## fraction of the compute roof on the admitted rows" in out
    section = out.split("## fraction of the compute roof", 1)[1].split("## ", 1)[0]
    assert "pct_of_achieved_tflops   (FIXED roof" in section
    assert "pct_of_roof_at_cell_clock (roof AT THE CLOCK THE CELL RAN):" in section
    assert f"over {len(pool)} rows" in section, "every v6 row here was scored"
    assert AR.ROOF_PREDATES not in section
    assert f"{n_high} of {len(pool)} admitted rows are {AR.HIGH_SIDE_NOTE}" in out
    assert "kept on purpose" in out
    assert "alpha = " in out
    assert "LEVEL failed HIGH" in out.split("## is alpha a scalar?", 1)[1]
    assert "LEVEL held or pre-v6" in out.split("## is alpha a scalar?", 1)[1]


def test_the_pinned_set_says_the_corrected_fraction_is_not_available(capsys):
    """The committed corpus is v3 to v5, so on it the reader's other branch is
    what shows, and it says so per row rather than printing a zero."""
    assert AR.main(["--pinned-set", "--bootstrap", "3"]) == 0
    out = capsys.readouterr().out
    assert f"{AR.ROOF_PREDATES}: 10813 rows" in out
    assert f"0 of 10813 admitted rows are {AR.HIGH_SIDE_NOTE}" in out


def test_the_memory_bound_census_classifies_against_the_cells_own_roof(
        tmp_path, monkeypatch):
    """`count_excluded_memory_bound` asks whether a row with no traffic column
    is memory-bound once the tile is corrected, against a ridge that is the
    compute roof over the bandwidth roof. On a boosted v6 row the compute roof
    the driver scored is 1.307x the fixed one, the ridge moves with it, and a
    row between the two ridges is memory-bound at its own clock. Planted at
    AI 200 between the fixed ridge (175) and the rescaled one (228.7): the
    level row is compute-bound, the boosted row is not, and a boosted row the
    driver refused to score stays on the fixed ridge."""
    import csv

    class Tile:
        block_m_derived, group_m_derived, provenance = 64, 1, "test"

    monkeypatch.setattr(AR, "resolve_tile_for_row", lambda row: Tile())
    monkeypatch.setattr(AR, "m_tiles_for_row", lambda row, block_m: 8.0)

    def row(load: float, roof: float, note: str = "") -> dict:
        return {"schema_version": SC.SCHEMA_VERSION, "impl": "vllm_fused_experts",
                "model": "mixtral-8x7b", "num_tokens": 256, "dtype": "bf16",
                "covers": "permute+up_gemm+act+down_gemm+unpermute",
                "ms_p50": 1.0, "correctness_passed": "True",
                "instrument": ON_BENCH, "clock_drift_ok": SC.VERDICT_OK,
                "host_bound_ok": SC.VERDICT_OK,
                "clock_level_ok": (SC.VERDICT_OK if load == REFERENCE_MHZ
                                   else SC.VERDICT_FAILED),
                "clock_level_side": "" if load == REFERENCE_MHZ else AR.LEVEL_HIGH,
                "sm_clock_load_mhz": load, "reference_clock_mhz": REFERENCE_MHZ,
                "achieved_peak_tflops": 700.0, "achieved_bw_gbps": 4000.0,
                "roof_at_cell_clock_tflops": roof, "roof_note": note,
                "flops": 2.0e11, "compulsory_bytes": 1.0e9,
                "load_active_experts": 8, "implied_traffic_ratio": ""}

    directory = tmp_path / "arm"
    directory.mkdir()
    boosted_roof = 700.0 * BOOSTED_MHZ / REFERENCE_MHZ
    with (directory / "run_x.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SC.COLUMNS, restval="")
        writer.writeheader()
        writer.writerow(row(REFERENCE_MHZ, 700.0))
        writer.writerow(row(BOOSTED_MHZ, boosted_roof))
        writer.writerow(row(BOOSTED_MHZ, 0.0, "the reference is graded 'idle-scalar'"))
    census = AR.count_excluded_memory_bound([directory / "run_x.csv"], 0.558)
    wrongly = next(v for k, v in census.items() if k.startswith("NO COLUMN BUT"))
    both = next(v for k, v in census.items() if k.startswith("compute-bound under both"))
    assert (wrongly, both) == (1, 2)
