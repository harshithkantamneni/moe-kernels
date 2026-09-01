"""Benchmark matrices, and whether they can answer the question they cost.

Two things are defended here.

THE EXISTING PROFILES ARE FROZEN. Ten published arms were measured on `full`'s
grid, and a token count added to it or a seed removed from it silently makes the
next arm incomparable to all of them. The first block below pins the shape of
every shipped profile that has already been run.

THE NEW GRID HAS TO BE READ CORRECTLY OR IT IS A REGRESSION. Densifying the
token grid does not, on its own, sharpen a crossing: `crossing_from_points`
reads adjacent slopes, so shrinking the spacing amplifies each slope's noise by
exactly the factor by which it shrinks the interpolation gap, and the detector's
first-passage rule turns the leftover into a downward bias. The dense grid pays
off only when `octave_ladders` splits it back into full-octave ladders, and the
simulation at the bottom of this file is what says so.
"""
import math
import statistics
from dataclasses import replace

import pytest

from moe.bench import profiles as PR
from moe.bench.crossing import crossing_from_points
from moe.bench.ridge import crossing_batch, saturation_batch

# --------------------------------------------------------------------------
# The profiles that have already been run stay exactly as they were run
# --------------------------------------------------------------------------

def test_the_published_grid_is_still_the_published_grid():
    """`full` produced 96,448 of this study's rows. Any change to these four
    lines makes the next arm incomparable to every one of them."""
    full = PR.get("full")
    assert full.token_counts == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                                 2048, 4096, 8192)
    assert full.models == ("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v2-lite",
                           "deepseek-v3")
    assert full.dtypes == ("bf16",)
    assert full.seeds == (0, 1, 2)
    assert len(full.routings) == 7
    assert full.l2_modes == (True, False) and full.graph_modes == (False, True)


def test_the_backbone_constant_is_the_published_grid():
    """The new profiles build on `COARSE_BACKBONE`, so it has to BE the grid the
    published arms ran rather than merely resemble it."""
    assert PR.COARSE_BACKBONE == PR.get("full").token_counts


@pytest.mark.parametrize("name", ["smoke", "standard", "fp8", "profile-cell",
                                  "full"])
def test_no_pre_existing_profile_changed_dtype_or_routing_axes(name):
    prof = PR.get(name)
    assert prof.dtypes and prof.routings and prof.seeds
    assert prof.name == name


# --------------------------------------------------------------------------
# A grid that brackets a ridge BAND, not a ridge
# --------------------------------------------------------------------------

def test_the_dense_region_brackets_both_ends_of_the_ridge_band():
    """The H200's ridge measured 160.3 and 176.2 on the same card, so the
    predicted crossing is an interval and a grid that reaches only its midpoint
    reaches neither end. Both ends of every model's prediction have to sit
    inside the dense region, with points below and above."""
    grid = PR.CROSSING_TOKENS
    lo_band, hi_band = min(PR.H200_RIDGE_BAND), max(PR.H200_RIDGE_BAND)
    for model in PR.CROSSING_MODELS:
        for ridge in (lo_band, hi_band):
            predicted = crossing_batch(model, ridge)
            assert min(grid) < predicted < max(grid), (model, ridge)
            below = [t for t in grid if t <= predicted]
            above = [t for t in grid if t >= predicted]
            # A factor of two either side is the powers-of-two grid's guarantee;
            # the dense region has to do better than that or it bought nothing.
            assert predicted / max(below) < 1.20, (model, ridge)
            assert min(above) / predicted < 1.20, (model, ridge)


def test_the_grid_is_placed_by_the_ridge_and_not_hardcoded():
    """Move the band and the extra points must move with it. A grid frozen as a
    literal would pass every other test here and silently describe the wrong
    card the first time this runs on an A100, whose ridge is 145.7."""
    a100 = PR.crossing_grid(PR.CROSSING_MODELS, ridge_band=(145.7, 145.7))
    h200 = PR.crossing_grid(PR.CROSSING_MODELS, ridge_band=PR.H200_RIDGE_BAND)
    assert a100 != h200
    added_a100 = sorted(set(a100) - set(PR.COARSE_BACKBONE))
    added_h200 = sorted(set(h200) - set(PR.COARSE_BACKBONE))
    assert added_a100 != added_h200
    assert added_a100[0] < added_h200[0], "a lower ridge crosses earlier"


#: DeepSeek-V3's published one-stage bf16 crossings under uniform routing, from
#: the C5 and span-extent sections of docs/FINDINGS.md. Both sit ABOVE 5793,
#: which is the highest slope the powers-of-two grid can measure, and its fp8
#: twin reports no crossing at all with its slope peaking at 0.497 at T=8192.
_MEASURED_ONE_STAGE_DEEPSEEK_CROSSINGS = (6315, 6446)


def _top_slope(ladder):
    """Where a ladder's last slope sits: the geometric mid of its top pair."""
    return math.sqrt(ladder[-1] * ladder[-2])


def test_every_ladder_reaches_past_the_top_of_the_predicted_band():
    """An octave ladder's highest measurable slope sits at its top point over
    sqrt(2), so it is the LADDER's reach that decides whether a crossing can be
    reported, not the grid's. No ladder may stop inside the prediction."""
    highest = max(crossing_batch(m, max(PR.H200_RIDGE_BAND))
                  for m in PR.CROSSING_MODELS)
    for j, ladder in enumerate(PR.octave_ladders(PR.CROSSING_TOKENS)):
        assert _top_slope(ladder) >= highest, (j, _top_slope(ladder), highest)


def test_the_published_grid_stops_where_deepseek_v3_actually_crossed():
    """THE DEFECT BEING REPAIRED, stated as the numbers that expose it. The
    powers-of-two grid clears the top of the predicted band by 2.7% and has
    nothing above that, and DeepSeek-V3's measured one-stage crossings are past
    it. Half this study's 'no crossing found' answers are this."""
    old = _top_slope(PR.COARSE_BACKBONE)
    band_top = crossing_batch("deepseek-v3", max(PR.H200_RIDGE_BAND))
    assert 1.0 < old / band_top < 1.03
    assert all(old < c for c in _MEASURED_ONE_STAGE_DEEPSEEK_CROSSINGS)

    # The new grid clears them on every ladder, ladder 0 included, which is what
    # rounding the top up to a whole power of two buys.
    for j, ladder in enumerate(PR.octave_ladders(PR.CROSSING_TOKENS)):
        assert all(_top_slope(ladder) > c
                   for c in _MEASURED_ONE_STAGE_DEEPSEEK_CROSSINGS), j


def test_the_grid_reaches_past_the_whole_window_not_just_the_band():
    """The window is this file's statement of how far past prediction a crossing
    may land (0.40 to 1.40, from the 0.45-1.13 the study has measured). A grid
    that cannot see the top of its own window has an untestable upper margin."""
    window_top = PR.CROSSING_WINDOW[1] * max(
        crossing_batch(m, max(PR.H200_RIDGE_BAND)) for m in PR.CROSSING_MODELS)
    assert _top_slope(PR.CROSSING_TOKENS) > window_top
    assert _top_slope(PR.octave_ladders(PR.CROSSING_TOKENS)[0]) > window_top


def test_an_empty_model_set_gives_back_the_backbone_rather_than_raising():
    assert PR.crossing_grid(()) == PR.COARSE_BACKBONE


# --------------------------------------------------------------------------
# The ladders the dense grid has to be read as
# --------------------------------------------------------------------------

def test_every_ladder_steps_by_a_full_octave():
    """The whole point. A ladder with a sub-octave step has the noise problem
    the dense grid was accused of having, and would make the split pointless."""
    for j, ladder in enumerate(PR.octave_ladders(PR.CROSSING_TOKENS)):
        assert len(ladder) >= 3, j          # two slopes minimum, or no crossing
        for a, b in zip(ladder, ladder[1:], strict=False):
            assert b / a >= 1.95, (j, a, b)


def test_one_ladder_is_the_published_powers_of_two_grid():
    """Anchoring the dense ladder on `2^(k/4)` rather than on the window's own
    edge is what makes residue class 0 the powers of two. Without it the new run
    would share no token count above 256 with 96,448 existing rows.

    It is the published grid plus one octave on top, not the published grid: the
    extra step is what lets ladder 0 see a crossing above 5793, which is where
    DeepSeek-V3's measured one-stage crossings are."""
    ladder0 = PR.octave_ladders(PR.CROSSING_TOKENS)[0]
    assert set(PR.COARSE_BACKBONE) < set(ladder0)
    assert all(t & (t - 1) == 0 for t in ladder0), "every point a power of two"
    assert set(ladder0) == {t for t in PR.CROSSING_TOKENS if t & (t - 1) == 0}


def test_the_ladders_cover_the_grid_and_overlap_only_where_it_is_coarse():
    """Each dense point belongs to exactly one ladder; the coarse points below
    the dense region are shared, because a ladder without them is too short to
    have two slopes."""
    grid = PR.CROSSING_TOKENS
    ladders = PR.octave_ladders(grid)
    assert set().union(*ladders) == set(grid)

    counts = {t: sum(t in lad for lad in ladders) for t in grid}
    shared = {t for t, n in counts.items() if n > 1}
    once = {t for t, n in counts.items() if n == 1}
    assert shared and once
    assert all(n == len(ladders) for t, n in counts.items() if t in shared)
    assert max(shared) < min(once), "sharing must stop where the grid densifies"


def test_the_ladders_need_nothing_but_the_token_counts():
    """They have to be recoverable from a published CSV by someone who does not
    know which ridge produced the grid, months later."""
    assert PR.octave_ladders(PR.CROSSING_TOKENS) == \
        PR.octave_ladders(tuple(reversed(PR.CROSSING_TOKENS)))


def test_a_grid_that_is_not_an_octave_ladder_is_refused_not_silently_split():
    """A LINEAR ladder split by residue puts two points a tenth of an octave
    apart into the same class, which is the defect the split exists to avoid.
    Returning them would hand the caller exactly the amplified slopes it thinks
    it is escaping, under a name that promises otherwise.

    1000 and 1100 both round to `4*log2(t) = 40`, so both land in class 0."""
    with pytest.raises(ValueError, match="not a 2\\^\\(k/4\\) grid"):
        PR.octave_ladders((1000, 1100, 1200, 1300))


def test_splitting_a_grid_with_no_dense_points_gives_the_grid_back():
    got = PR.octave_ladders(PR.COARSE_BACKBONE)
    assert all(lad == PR.COARSE_BACKBONE for lad in got)


# --------------------------------------------------------------------------
# The new profiles
# --------------------------------------------------------------------------

def test_the_crossing_profile_cannot_pool_routing_regimes():
    """`2R/b` describes uniform routing. Under skew the busy experts are
    compute-bound while the quiet ones are still memory-bound AT THE SAME BATCH,
    so the layer straddles the ridge and there is no single crossing to find.
    Pooling the seven regimes moved a published cross-card ratio by up to 4.3x.
    One routing in the matrix means no report can pool by accident."""
    prof = PR.get("crossing-uniform")
    assert len(prof.routings) == 1
    assert prof.routings[0].kind == "uniform"


def test_the_crossing_profile_has_enough_replicates_to_survive_throttling():
    """Three seeds is not enough: qwen2's crossing moves 1.39x across the three
    (593 / 779 / 824), traced to ONE point at T=512 where throttling dropped one
    of two replicate rows. Throttle exclusion removed 33% of H200 rows near the
    crossing in that arm, so a seed count has to leave a usable median after
    losing a third of it."""
    seeds = PR.get("crossing-uniform").seeds
    assert len(seeds) >= 5
    assert len(seeds) * 2 // 3 >= 4
    assert len(set(seeds)) == len(seeds)


def test_the_crossing_profile_measures_one_timing_mode():
    """`crossing_report` medians across all four modes by default, mixing
    L2-warm and graph-replay rows into a number reported as one basis. Measuring
    one mode removes the confound as well as most of the cost."""
    prof = PR.get("crossing-uniform")
    assert len(prof.l2_modes) == 1 and len(prof.graph_modes) == 1
    assert prof.graph_modes == (False,), "graph replay is not the study's basis"


#: How many of the sixteen (model x implementation) cells of the canonical bf16
#: pool report a crossing at all, uniform routing, ridge 160.3, eager rows only.
#: Counted from `scripts/crossing_report.py` over the four canonical arms with
#: `--routing uniform --cuda-graph false --l2-flush true`, and again with
#: `--l2-flush false`. Keyed by `l2_flush`, valued `(five_stage, one_stage)`.
_MEASURED_CROSSING_COVERAGE = {True: (6, 3), False: (8, 8)}


def test_the_crossing_profile_runs_the_basis_that_reports_a_crossing():
    """THE L2-COLD BASIS DELETES FIVE OF THE EIGHT ONE-STAGE CROSSINGS, so a
    profile that ran it would rent a box to produce the emptier table.

    The cause is not the cache. A 256 MB flush between every pair of timed
    iterations is sustained load, it holds the clocks up, `clock_drift` sets
    `throttled`, and `crossing.timed_rows` drops those rows: 48-75% of L2-cold
    eager rows at T>=2048 against 5-8% warm. The deletions are what move the
    answer, and the fingerprint is that they move a ONE-stage span's crossing
    2.3-2.4x while moving a five-stage span's by 3-13%, which no L2 effect over
    the same rows can do."""
    cold_five, cold_one = _MEASURED_CROSSING_COVERAGE[True]
    warm_five, warm_one = _MEASURED_CROSSING_COVERAGE[False]
    assert (warm_five, warm_one) == (8, 8), "warm is the full-coverage basis"
    assert cold_one < warm_one and cold_five < warm_five
    assert PR.get("crossing-uniform").l2_modes == (False,)


def test_the_crossing_profile_keeps_the_published_grid_inside_it():
    prof = PR.get("crossing-uniform")
    assert set(PR.COARSE_BACKBONE) <= set(prof.token_counts)
    assert prof.models == PR.get("full").models
    assert prof.dtypes == ("bf16",)


def test_the_deployment_profile_pairs_every_shard_with_its_control():
    """A shard measured without its unsharded twin in the same session answers
    nothing: the question is whether moving to the width vLLM tunes for changes
    the kernel, and that is a difference, not a level."""
    from moe.spec import MODEL_CONFIGS

    models = PR.get("deployment").models
    shards = [m for m in models if MODEL_CONFIGS[m].tensor_parallel > 1]
    assert shards, "a deployment profile with no shard is the old sweep"
    for shard in shards:
        base = shard.rsplit("-tp", 1)[0]
        assert base in models, f"{shard} has no TP=1 control"


def test_the_deployment_profile_covers_the_two_deepseek_widths_that_ship():
    """`E=256,N=256` and `E=256,N=512` are the TP=8 and TP=4 widths vLLM ships
    tuned H200 configs for. `E=256,N=2048`, which every published row used, is
    shipped for no device at all."""
    models = PR.get("deployment").models
    assert "deepseek-v3-tp8" in models and "deepseek-v3-tp4" in models


def test_the_deployment_profile_runs_both_dtypes():
    """The shard changes what each dtype resolves to and they do not move
    together: in bf16 the mixtral and qwen2 shards reach tuned H200 configs and
    the DeepSeek-V3 shards do not, and in fp8 the DeepSeek-V3 shards have tuned
    files that this harness's per-expert scales will miss."""
    assert PR.get("deployment").dtypes == ("bf16", "fp8_e4m3")


# --------------------------------------------------------------------------
# The trace axis, which the repo calls its differentiator and never swept
# --------------------------------------------------------------------------

def test_a_profile_finally_consumes_a_captured_trace():
    """`RoutingSpec` has supported `kind='trace'` from the start and no profile
    ever named one, so `scripts/capture_traces.py` wrote a .npz that nothing in
    the matrix could read. Every published row is parametric routing."""
    kinds = {r.kind for p in PR.PROFILES.values() for r in p.routings}
    assert "trace" in kinds
    prof = PR.get("trace-replay")
    assert any(r.kind == "trace" for r in prof.routings)
    assert any(r.kind != "trace" for r in prof.routings), \
        "a trace measured with no parametric control is a number with nothing " \
        "to be different from"


def test_the_trace_profile_names_only_models_that_fit_on_one_card():
    """DeepSeek-V3 is 1,369 GB of bf16 weights and needs five H200s, so its
    routing has never been captured and no profile may imply it has."""
    prof = PR.get("trace-replay")
    assert "deepseek-v3" not in prof.models
    for routing in prof.routings:
        if routing.kind == "trace":
            assert PR.trace_model(routing.trace_id) in prof.models


def test_a_trace_is_only_swept_on_the_model_it_was_captured_from():
    """A sweep is a cartesian product, so three models against three traces asks
    for six invalid cells. `Profile.specs` drops them rather than letting the
    run discover them one by one."""
    prof = PR.get("trace-replay")
    for spec in prof.specs():
        if spec.routing.kind == "trace":
            assert PR.trace_model(spec.routing.trace_id) == spec.model.name

    seen = {(s.model.name, s.routing.label) for s in prof.specs()
            if s.routing.kind == "trace"}
    assert len(seen) == len(prof.models), "each model replays its own capture"


def test_the_pairing_rule_catches_the_collision_the_run_time_check_cannot():
    """`TraceSet.forced_ids` compares expert counts, which stops mixtral's
    8-expert capture reaching a 64-expert model. It does NOT stop
    deepseek-v2-lite's capture reaching qwen2-57b-a14b: both have exactly 64
    experts, so that cell runs, passes correctness, and publishes a row labelled
    qwen2 whose expert load came from another network."""
    from moe.spec import MODEL_CONFIGS

    assert MODEL_CONFIGS["deepseek-v2-lite"].num_experts == \
        MODEL_CONFIGS["qwen2-57b-a14b"].num_experts
    v2lite = PR.RoutingSpec("trace", trace_id="deepseek-v2-lite-chat-decode")
    assert PR.trace_belongs_to(v2lite, "deepseek-v2-lite")
    assert not PR.trace_belongs_to(v2lite, "qwen2-57b-a14b")


def test_a_slice_suffix_and_a_shard_suffix_both_resolve_to_one_model():
    """`@bNlM` pins a batch and layer, and a longest-match rule is what keeps
    `deepseek-v3-tp4-...` off the unsharded model."""
    assert PR.trace_model("mixtral-8x7b-chat-decode@b3l17") == "mixtral-8x7b"
    assert PR.trace_model("deepseek-v3-tp4-chat-decode") == "deepseek-v3-tp4"
    assert PR.trace_model("captured-on-a-friday") is None
    # An id naming no known model belongs to whatever the caller pairs it with:
    # a hand-supplied --trace-id is nobody else's business.
    unknown = PR.RoutingSpec("trace", trace_id="captured-on-a-friday")
    assert PR.trace_belongs_to(unknown, "mixtral-8x7b")


def test_the_trace_profile_replays_real_layers_rather_than_their_average():
    """Seeds pick the (batch, layer) slice, so three seeds are three real
    layers. Averaging skewed layers with different hot experts produces a
    histogram flatter than any layer that ever ran."""
    assert len(PR.get("trace-replay").seeds) >= 3


# --------------------------------------------------------------------------
# Overriding the matrix without editing this file on the box
# --------------------------------------------------------------------------

def _args(argv):
    from moe.bench.cli import parse_args
    return parse_args(argv)


def test_a_routing_can_be_overridden_from_the_command_line():
    """Until this existed the only way to point a sweep at a capture was to edit
    `profiles.py` on the rented box, which stamps `git_dirty` on every row of
    the run and makes the arm unpublishable."""
    from moe.bench.cli import apply_overrides

    prof = apply_overrides(PR.get("standard"),
                           _args(["--routings", "uniform,zipf:2.0"]))
    assert [r.label for r in prof.routings] == ["uniform", "zipf:2"]
    assert prof.models == PR.get("standard").models, "one axis, not all of them"


def test_the_command_line_spells_a_routing_the_way_everything_prints_it():
    """`RoutingSpec.label` is the format, so what a dry run printed or a
    published CSV recorded pastes straight back in. Two spellings of a routing
    would be two chances to record the wrong one."""
    from moe.bench.cli import parse_routing

    for text in ("uniform", "zipf:1.2", "dirichlet:0.3", "hot:0.5",
                 "trace:mixtral-8x7b-chat-decode",
                 "trace:mixtral-8x7b-chat-decode@b3l17"):
        assert parse_routing(text).label == text


def test_a_misspelt_routing_exits_instead_of_raising():
    """This runs during argument parsing, where a traceback is noise."""
    from moe.bench.cli import parse_routing

    for bad in ("zipf:banana", "trace:", "gaussian:1.0", "hot:2.0"):
        with pytest.raises(SystemExit):
            parse_routing(bad)


def test_an_override_that_empties_the_matrix_says_so_before_the_run():
    """A trace belongs to the model it was captured from, so an override can ask
    for a matrix whose every cell is dropped. Finding that out from a sweep that
    writes no rows is the expensive way."""
    from moe.bench.cli import apply_overrides

    args = _args(["--models", "qwen2-57b-a14b",
                  "--routings", "trace:deepseek-v2-lite-chat-decode"])
    with pytest.raises(SystemExit, match="no cells"):
        apply_overrides(PR.get("trace-replay"), args)


def test_overrides_reach_the_trace_profile_without_touching_the_file():
    from moe.bench.cli import apply_overrides

    args = _args(["--models", "deepseek-v2-lite", "--tokens", "1,4096",
                  "--routings", "uniform,trace:deepseek-v2-lite-chat-decode@b3l17"])
    prof = apply_overrides(PR.get("trace-replay"), args)
    assert prof.token_counts == (1, 4096)
    assert len(prof.specs()) == 2 * 2 * len(prof.seeds)
    assert PR.plan(prof).missing_traces == ("deepseek-v2-lite-chat-decode@b3l17",)


@pytest.mark.parametrize("name", ["crossing-uniform", "deployment",
                                  "trace-replay"])
def test_the_new_profiles_plan_cleanly(name):
    ref = tuple(f"ref_{s}" for s in
                __import__("moe.stages", fromlist=["x"]).CANONICAL_STAGES)
    plan = PR.plan(PR.get(name), impl_filter=ref, include_reference=True)
    assert plan.problems == (), plan.problems
    assert plan.planned > 0


# --------------------------------------------------------------------------
# What a profile costs, before anything is rented
# --------------------------------------------------------------------------

#: Measured wall clock of published arms, first to last `timestamp` in their
#: CSVs. Both ran `full` at TP=1 bf16 with the framework whole-layer cells on,
#: which is 1,176 specs x 2 vLLM cells x 4 timing modes = 9,408 rows.
_MEASURED_FULL_VLLM_HOURS = {"a100-cross-card": 1.229, "h200-whole-layer": 1.223}

#: Same, for the base env of `2026-08-26-...-full-three-way-recalibrated`: `full`
#: minus deepseek-v2-lite, and with the all-reference whole-layer cell still
#: switched on, which is 882 specs x (2 spans + 1 reference layer) x 4 modes =
#: 10,584 rows, measured at 2.254 h. The reference layer went away in ffaf154's
#: successor and the fixture below has to turn it back on to reproduce the arm.
_MEASURED_THREE_MODEL_BASE_HOURS = 2.254

#: Base env of `2026-08-28-...-h200-fp8-refixed`: 1,176 specs x 2 scaled spans x
#: 4 modes = 9,408 rows at 1.867 h. An fp8 arm, so it checks the other column.
_MEASURED_FP8_BASE_HOURS = 1.867


@pytest.mark.parametrize("arm,hours", sorted(_MEASURED_FULL_VLLM_HOURS.items()))
def test_the_cost_model_reproduces_a_published_arm(arm, hours):
    """An estimate that has never been checked against a real run is a guess
    with a unit on it. Two arms, two cards, same profile."""
    got = PR.estimated_hours(PR.get("full"))["vllm"]
    assert 0.9 * hours < got < 1.1 * hours, (arm, got, hours)


def test_the_cost_model_reproduces_the_base_env_with_the_reference_layer_on():
    """The base env is the expensive one, the one a laptop cannot count because
    it holds the slowest cells, and the one where getting the impl count wrong
    is worth 23%: the third impl in this arm was `__pipeline__`, the
    all-reference whole layer, at 1.7x the rate of a real span."""
    three = replace(PR.get("full"),
                    models=("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v3"),
                    include_pipeline_scope=True)
    got = PR.estimated_hours(three)["base"]
    assert 0.9 * _MEASURED_THREE_MODEL_BASE_HOURS < got \
        < 1.1 * _MEASURED_THREE_MODEL_BASE_HOURS, got


def test_the_cost_model_reproduces_an_fp8_arm():
    """The other dtype column, against an arm that ran only the scaled spans."""
    got = PR.estimated_hours(replace(PR.get("full"), dtypes=("fp8_e4m3",)))["base"]
    assert 0.9 * _MEASURED_FP8_BASE_HOURS < got < 1.1 * _MEASURED_FP8_BASE_HOURS, got


def test_the_reference_whole_layer_is_priced_apart_from_a_real_span():
    """It is 3.2x a span per eager row and measures no kernel, which is why it
    was switched off. Folding it into the base rate over-estimated every profile
    that has it off, which is now all of them."""
    assert PR.REFERENCE_PIPELINE_COST.eager > \
        3.0 * PR.MEASURED_CELL_COST[("base", "bf16")].eager
    with_it = replace(PR.get("full"), include_pipeline_scope=True)
    assert PR.estimated_hours(with_it)["base"] > \
        1.5 * PR.estimated_hours(PR.get("full"))["base"]


def test_fp8_costs_more_per_cell_and_not_in_the_timed_loop():
    """An fp8 vLLM CELL costs 1.30x a bf16 one, which is the "about 30% more"
    the flat per-row rate recorded. Where it comes from is the opposite of what
    that rate implied: the eager ROW is cheaper (0.555 against 0.608), because
    halving the weight bytes makes the kernel faster while `calibrate_iters`
    targets a fixed 200 ms either way. The cost is in the prologue, which
    carries quantisation, and in the graph column, because `should_time_graph`
    prices launch overhead against a roofline minimum computed from compulsory
    BYTES -- halve them and it stops skipping."""
    for env in ("vllm", "sglang"):
        bf16, fp8 = (PR.MEASURED_CELL_COST[(env, f)] for f in ("bf16", "fp8"))
        assert fp8.eager < bf16.eager, env
        assert fp8.graph > 1.5 * bf16.graph, env
        def four_mode(c):
            return c.prologue + 2 * c.eager + 2 * c.graph
        # 1.30x for vLLM, 1.23x for SGLang, and the sign is what matters: the
        # per-row rate this replaced would now report fp8 as CHEAPER.
        assert four_mode(fp8) > 1.20 * four_mode(bf16), env

    bf16 = replace(PR.get("full"), dtypes=("bf16",))
    fp8 = replace(PR.get("full"), dtypes=("fp8_e4m3",))
    assert PR.estimated_hours(fp8)["vllm"] > 1.25 * PR.estimated_hours(bf16)["vllm"]


def test_a_one_mode_profile_is_not_a_quarter_of_a_four_mode_one():
    """THE BUG THIS COST MODEL EXISTS TO FIX. Dropping three of four timing
    modes drops three of four TIMING ROWS, but the per-cell prologue -- weights,
    forced routing, the fp32 oracle, the correctness compare -- is paid once
    whatever runs after it. So a one-eager-mode cell is 38-42% of a four-mode
    cell, not 25%, and the flat per-row rate under-priced `crossing-uniform` and
    `deployment` by about 1.6x each."""
    for env in ("base", "vllm", "sglang"):
        cost = PR.MEASURED_CELL_COST[(env, "bf16")]
        one = cost.prologue + cost.eager
        four = cost.prologue + 2 * cost.eager + 2 * cost.graph
        assert 1.5 < one / (0.25 * four) < 1.8, env

    one_mode = replace(PR.get("full"), l2_modes=(False,), graph_modes=(False,))
    assert PR.estimated_hours(one_mode)["total"] > \
        0.35 * PR.estimated_hours(PR.get("full"))["total"]


def test_the_l2_axis_multiplies_both_modes_and_the_graph_axis_selects():
    """`len(l2_modes) * len(graph_modes)` was one number, and one number cannot
    say which KIND of row it is counting: it priced an eager-only profile at a
    graph profile's rate and the other way round."""
    both = replace(PR.get("full"), l2_modes=(True, False),
                   graph_modes=(False, True))
    assert PR._mode_rows(both) == (2, 2)
    assert PR._mode_rows(replace(both, graph_modes=(False,))) == (2, 0)
    assert PR._mode_rows(replace(both, graph_modes=(True,))) == (0, 2)
    assert PR._mode_rows(replace(both, l2_modes=(False,))) == (1, 1)


def test_a_graph_row_costs_the_same_everywhere_and_the_spread_is_the_skip_rate():
    """Every published arm timed a graph row at 0.657-0.674 s whatever the env
    or dtype. The `graph` column spreads 0.233 to 0.560 because it is per
    PLANNED row with `should_time_graph`'s skip policy already inside it, and
    that policy fires on bytes. Reading it as a per-timed-row rate would say
    SGLang replays graphs twice as fast as vLLM, which is not a thing."""
    graph = [c.graph for c in PR.MEASURED_CELL_COST.values()]
    eager = [c.eager for c in PR.MEASURED_CELL_COST.values()]
    assert max(graph) / min(graph) > 2.0, "the skip rate is the whole spread"
    assert max(eager) / min(eager) < 1.25, "an eager row is always timed"


def test_the_impl_counts_do_not_come_from_the_registry():
    """On a laptop neither framework imports, so `candidate_impls('vllm')` is
    empty. An estimate built on it reports zero hours for the environment that
    costs the most, which is precisely the answer a pre-rental estimate must not
    be able to give."""
    assert PR.candidate_impls(env="vllm") == [] or True   # true either way
    assert PR.estimated_hours(PR.get("full"))["vllm"] > 0.5


@pytest.mark.parametrize("name,budget", [("crossing-uniform", 1.0),
                                         ("deployment", 0.75),
                                         ("trace-replay", 0.75)])
def test_the_new_profiles_are_affordable(name, budget):
    """A profile nobody can afford to run is not a contribution. Both of these
    have to come in well under `full`'s four hours across all three
    environments, since the point of each is to be run in ADDITION to it."""
    got = PR.estimated_hours(PR.get(name))["total"]
    assert got < budget, (name, got)
    assert got < 0.5 * PR.estimated_hours(PR.get("full"))["total"]


# --------------------------------------------------------------------------
# The claim the dense grid rests on
# --------------------------------------------------------------------------

#: Soft roofline, `ms = (a^p + (bT)^p)^(1/p)`, fitted in log space to the
#: published uniform vLLM medians above each model's saturation batch. It fits
#: to 4.1-4.6% rms in log, so the curve shape is the measured one rather than an
#: invention, and its log-log slope reaches 0.5 at exactly `T = a/b` whatever `p`
#: is -- which is what makes it a target a grid can be scored against.
_FITTED_CURVES = {
    "mixtral-8x7b": (0.6400, 0.001401, 1.24),
    "qwen2-57b-a14b": (0.8043, 0.000902, 1.22),
    "deepseek-v2-lite": (0.2947, 0.000273, 1.77),
    "deepseek-v3": (5.4108, 0.001743, 1.45),
}


def _sample(model, grid, sigma, rng):
    """One run's worth of noisy times over the whole grid, keyed by token count."""
    a, b, p = _FITTED_CURVES[model]
    return {t: (a ** p + (b * t) ** p) ** (1.0 / p)
            * rng.lognormvariate(0.0, sigma) for t in grid}


def _crossings(model, grid, ladders, sigma, draws=1200, seed=3):
    """`(whole-grid ratios, ladder-median ratios, per-ladder ratios)`.

    ONE noise realisation per simulated run, read every way, because that is how
    a real arm works: the ladders are subsets of the SAME rows, so they share the
    coarse points' noise. Drawing fresh noise per ladder would credit the split
    with an averaging it does not get, and an earlier version of this file did
    exactly that in the other direction -- it handed all four ladders the same
    seeded stream, which made them perfectly correlated and hid the effect.
    """
    a, b, p = _FITTED_CURVES[model]
    truth, floor = a / b, saturation_batch(model)
    rng = __import__("random").Random(seed)
    whole, merged = [], []
    per = [[] for _ in ladders]
    for _ in range(draws):
        ms = _sample(model, grid, sigma, rng)
        got = crossing_from_points([(t, ms[t]) for t in grid], 0.5, floor)
        if got is not None:
            whole.append(got / truth)
        here = []
        for j, ladder in enumerate(ladders):
            got = crossing_from_points([(t, ms[t]) for t in ladder], 0.5, floor)
            if got is not None:
                here.append(got / truth)
                per[j].append(got / truth)
        if here:
            merged.append(statistics.median(here))
    return sorted(whole), sorted(merged), [sorted(x) for x in per]


def _band(vals):
    """`(median, 95th over 5th)`. The ratio rather than the two edges, because
    what a crossing needs is a width, and a width on a log-spaced quantity is a
    multiplier."""
    lo = vals[int(0.05 * (len(vals) - 1))]
    hi = vals[int(0.95 * (len(vals) - 1))]
    return statistics.median(vals), hi / lo


#: Per-cell relative spread of one timing, and the seed counts either side of
#: this change. The median over n replicates has about `1.2533 / sqrt(n)` of a
#: single cell's spread, which is the number the crossing actually sees.
_CELL_SIGMA = 0.02
_SIGMA_3_SEEDS = _CELL_SIGMA * 1.2533 / math.sqrt(3)
_SIGMA_7_SEEDS = _CELL_SIGMA * 1.2533 / math.sqrt(7)


@pytest.mark.parametrize("model", sorted(_FITTED_CURVES))
def test_reading_the_dense_grid_whole_is_worse_than_the_grid_it_extends(model):
    """THE REASON `octave_ladders` EXISTS, and the reason this profile would be
    a regression without it.

    `crossing_from_points` takes the first ADJACENT slope pair to bracket 0.5.
    Quartering the token spacing quarters the log-T baseline each slope is
    divided by, so every slope gets four times noisier, while the slope
    DIFFERENCE the crossing interpolates across shrinks by the same factor. The
    two cancel exactly, and then the first-passage rule turns what is left into a
    DOWNWARD bias, because more noisy slopes give the sequence more chances to
    cross 0.5 early.

    So densifying is not a free improvement that a careless reader merely fails
    to exploit. Read the wrong way it is strictly worse than doing nothing.
    """
    ladders = PR.octave_ladders(PR.CROSSING_TOKENS)
    whole, _, _ = _crossings(model, PR.CROSSING_TOKENS, ladders, _SIGMA_7_SEEDS)
    backbone, _, _ = _crossings(model, PR.COARSE_BACKBONE, (), _SIGMA_7_SEEDS)

    whole_median, whole_width = _band(whole)
    back_median, back_width = _band(backbone)
    assert whole_median < 0.98, (model, whole_median)
    assert 0.98 < back_median < 1.02, (model, back_median)
    assert whole_width > 1.20 * back_width, (model, whole_width, back_width)


@pytest.mark.parametrize("model", sorted(_FITTED_CURVES))
def test_the_ladder_read_is_unbiased_and_beats_the_grid_it_extends(model):
    """And read properly the same rows roughly HALVE the crossing's uncertainty,
    because four near-independent estimates are being combined instead of one.
    That is the return on the extra token counts, and it is the whole case for
    renting them."""
    ladders = PR.octave_ladders(PR.CROSSING_TOKENS)
    _, merged, per = _crossings(model, PR.CROSSING_TOKENS, ladders, _SIGMA_7_SEEDS)
    backbone, _, _ = _crossings(model, PR.COARSE_BACKBONE, (), _SIGMA_7_SEEDS)

    assert all(per), "every ladder must find a crossing at all"
    median, width = _band(merged)
    _, back_width = _band(backbone)
    assert 0.98 < median < 1.02, (model, median)
    assert width < back_width, (model, width, back_width)
    # Each ladder on its own is about as good as the published grid; the gain is
    # in combining them, which is what makes the SPREAD of the four an empirical
    # error bar on the crossing rather than a bootstrapped one.
    assert width - 1.0 < 0.75 * (back_width - 1.0), (model, width, back_width)


@pytest.mark.parametrize("model", sorted(_FITTED_CURVES))
def test_replicates_are_the_other_lever_and_they_work_on_any_grid(model):
    """Seeds and token counts fix different halves of the problem, which is why
    this profile changes both. Three to seven seeds narrows the band on the
    unchanged powers-of-two grid, with no new token count involved."""
    three, _, _ = _crossings(model, PR.COARSE_BACKBONE, (), _SIGMA_3_SEEDS)
    seven, _, _ = _crossings(model, PR.COARSE_BACKBONE, (), _SIGMA_7_SEEDS)
    assert _band(seven)[1] < _band(three)[1], model
