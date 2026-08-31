"""Benchmark matrices, sized to a time budget.

`smoke` exists so the first thing a GPU session does is prove the harness works,
in about two minutes, before committing to a long sweep. `full` is the
publication sweep the ten published arms came from, and it is frozen: a token
count added to its grid makes the next arm incomparable to 96,448 existing rows.

Token counts deliberately start at 1. A single token is the decode regime, and
decode with many experts is the memory-bound weight-loading wall this project
targets. A sweep that starts at 512 tokens would miss the entire phenomenon.

TWO PROFILES ANSWER FAULTS IN THE MATRIX RATHER THAN IN THE HARDWARE.
`crossing-uniform` exists because the study's central measurement, the ridge
crossing, has never been made on a grid that could resolve one or with enough
replicates to survive one throttled row. `deployment` exists because every
published cell is TP=1 bf16, and TP=1 DeepSeek-V3 is a shape vLLM ships no tuned
config for on any device, because nobody serves it.

`estimated_hours` prices any of them from the published arms' measured rates, so
"can we afford this" is answered on the laptop it is written on rather than by
starting it on a rented box and watching.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..pipeline import reference_pipeline_names
from ..spec import FP8_WEIGHT_DTYPES, MODEL_CONFIGS, BenchSpec, RoutingSpec, sweep
from ..stages import BASE_ENV, CANONICAL_STAGES, StageSpan, registry
from .ridge import crossing_batch


@dataclass(frozen=True)
class Profile:
    name: str
    models: tuple[str, ...]
    token_counts: tuple[int, ...]
    dtypes: tuple[str, ...]
    routings: tuple[RoutingSpec, ...]
    seeds: tuple[int, ...] = (0,)
    l2_modes: tuple[bool, ...] = (True, False)
    graph_modes: tuple[bool, ...] = (False, True)
    warmup: int = 25
    trials: int = 3
    iters: int | None = None
    #: Add an ALL-REFERENCE whole-layer cell per spec: a python loop over every
    #: expert. Measured 7.337 ms against vLLM's 0.588 at mixtral/T=512, so it is
    #: a 12.5x ceiling rather than a target. Real, but slow and about no kernel.
    include_pipeline_scope: bool = False
    #: Add a whole-layer cell per framework span: ref_router plus that span,
    #: timed as one. The only way to price a FULL MoE layer, since every
    #: framework span covers five of six stages and omits the router.
    include_framework_pipeline: bool = False
    notes: str = ""

    def specs(self) -> list[BenchSpec]:
        return list(sweep([MODEL_CONFIGS[n] for n in self.models],
                          list(self.token_counts), list(self.dtypes),
                          list(self.routings), list(self.seeds)))


SKEW_SWEEP = (
    RoutingSpec("uniform"),
    RoutingSpec("zipf", 0.6),
    RoutingSpec("zipf", 1.2),
    RoutingSpec("hot", 0.5),
    RoutingSpec("dirichlet", 0.3),
)

# --------------------------------------------------------------------------
# Token grids that can resolve a crossing
# --------------------------------------------------------------------------

#: Powers of two, decode through prefill. Every published arm ran this, so any
#: new grid keeps it as a subset or gives up comparability with 96,448 rows.
COARSE_BACKBONE: tuple[int, ...] = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

#: The H200's ridge is a BAND, not a number: three calibrations of the same card
#: measured 160.3, 162.8 and 176.2 FLOP/byte, because achieved bf16 moved 9.9%
#: while bandwidth reproduced to 0.06%. See the ridge-band section of
#: docs/FINDINGS.md, which also shows the spread is not a clock artefact.
#:
#: So "the predicted crossing" is itself an interval, and a grid that brackets
#: only its midpoint brackets neither end of it.
H200_RIDGE_BAND: tuple[float, float] = (160.3, 176.2)

#: How far past the predicted band the dense region runs, as a multiple of the
#: prediction. Not symmetric, and the asymmetry is measured rather than chosen:
#: every span in this study crosses EARLY, at 0.45 to 1.13 of prediction (0.452
#: for a five-stage span against the full byte model, 1.129 for a one-stage span
#: against `2R/b`), so the interesting side is below. 0.40 clears the lowest
#: measurement and 1.40 clears the highest.
CROSSING_WINDOW: tuple[float, float] = (0.40, 1.40)

#: Points per factor of two inside the dense region. Four, and this is the one
#: number here that a simulation chose rather than an argument.
#:
#: DENSIFYING ALONE MAKES THE CROSSING WORSE. `crossing_from_points` reads the
#: slope between ADJACENT points, so a slope's noise scales as
#: `1 / log(t1/t0)`: quartering the spacing quadruples every local slope's
#: error, while the slope DIFFERENCE the crossing interpolates across shrinks by
#: the same factor. The two cancel, and then the detector's "first pair to
#: bracket 0.5" rule turns the leftover noise into a downward bias, because a
#: first-passage statistic over more, noisier slopes fires early.
#:
#: MEASURED, against a soft-roofline curve `ms = (a^p + (bT)^p)^(1/p)` fitted to
#: the published uniform vLLM rows (4.1 to 4.6% rms in log, so the shape is the
#: real one) whose log-log slope hits 0.5 exactly at `T = a/b`, which makes the
#: truth a number a grid can be scored against. 3000 draws, one lognormal noise
#: realisation per run at the 2% per cell the published replicates show, seven
#: seeds, crossing reported as a ratio to that truth, over all four models:
#:
#:     powers of two (today's grid)      median 1.00   5th-95th spans 1.08-1.18x
#:     this grid read as ONE ladder      median 0.92-0.96      spans 1.38-1.60x
#:     this grid read as 4 ladders       median 1.00           spans 1.06-1.09x
#:
#: At three seeds the same three rows are 1.14-1.26x, 1.66-2.17x at a median of
#: 0.82-0.91, and 1.09-1.14x. So the dense grid roughly HALVES the crossing's
#: uncertainty when `octave_ladders` reads it, and roughly doubles it, with a
#: bias, when anything else does. That is why the two live next to each other.
POINTS_PER_OCTAVE = 4

#: Two points whose token counts differ by less than this are the same point for
#: sweeping purposes, and the sweep should not pay for both.
_GRID_DEDUP_RATIO = 1.02


def _merge_grids(*groups: tuple[int, ...]) -> tuple[int, ...]:
    """Sorted union, with near-duplicates collapsed to the lower point."""
    out: list[int] = []
    for t in sorted({int(t) for g in groups for t in g}):
        if out and t <= out[-1] * _GRID_DEDUP_RATIO:
            continue
        out.append(t)
    return tuple(out)


def crossing_grid(models: tuple[str, ...],
                  ridge_band: tuple[float, float] = H200_RIDGE_BAND,
                  dtype: str = "bf16",
                  backbone: tuple[int, ...] = COARSE_BACKBONE,
                  window: tuple[float, float] = CROSSING_WINDOW,
                  per_octave: int = POINTS_PER_OCTAVE) -> tuple[int, ...]:
    """The backbone, plus a `2^(1/per_octave)` ladder over the crossing region.

    Placed from `ridge.crossing_batch` at BOTH ends of the ridge band rather
    than hardcoded, so the grid follows the models and the calibration instead
    of a table someone has to remember to edit. Change the band or the model set
    and the extra points move.

    THE TOP OF THE RANGE IS NOT SET BY THE WINDOW. An octave ladder's highest
    measurable slope sits at its top point over sqrt(2), so the published grid,
    ending at 8192, has exactly one slope point above DeepSeek-V3's predicted
    band: 5793 against a band top of 5638, a 2.7% margin and nothing beyond it.
    Any crossing above 5793 is invisible to it, and DeepSeek-V3's published
    one-stage bf16 crossings are 6315 and 6446, above that line, while its fp8
    twin has NO crossing at all under any filter, its slope peaking at 0.497 at
    T=8192. The grid stopped before the transition.

    So the dense region runs past the band rather than to it. The shortest of
    `per_octave` interleaved ladders tops out a factor
    `2^((per_octave-1)/per_octave)` below the grid's last point, so a reach of
    `2^(((per_octave-1)/per_octave) + 1/2)` times the highest prediction gives
    EVERY ladder a slope above the band. That is then rounded up to a whole
    power of two, which sounds cosmetic and is not: it is what gives ladder 0 a
    slope above the whole WINDOW rather than 2.7% above the band, and ladder 0
    is the powers-of-two grid the published arms ran and the one a new arm has
    to stay comparable with. The offset ladders reach further for free, since
    they end higher; interleaving buys reach as well as precision.

    Anchored on exact powers of two (`2^(k/per_octave)` for integer k), which is
    what makes residue class 0 the powers-of-two grid itself, so one of the
    ladders `octave_ladders` returns is bit-for-bit the grid every published arm
    ran.
    """
    if per_octave < 1:
        raise ValueError(f"per_octave must be at least 1, got {per_octave}")
    if not models:
        return _merge_grids(backbone)

    lo_ridge, hi_ridge = min(ridge_band), max(ridge_band)
    lowest = min(crossing_batch(m, lo_ridge, dtype) for m in models)
    highest = max(crossing_batch(m, hi_ridge, dtype) for m in models)

    lo = window[0] * lowest
    reach = highest * 2.0 ** ((per_octave - 1) / per_octave + 0.5)
    hi = max(window[1] * highest, reach)

    klo = math.floor(per_octave * math.log2(lo))
    khi = math.ceil(per_octave * math.log2(hi))
    khi = per_octave * math.ceil(khi / per_octave)   # end on a power of two
    dense = tuple(round(2.0 ** (k / per_octave)) for k in range(klo, khi + 1))
    return _merge_grids(backbone, dense)


def octave_ladders(token_counts: tuple[int, ...],
                   per_octave: int = POINTS_PER_OCTAVE
                   ) -> tuple[tuple[int, ...], ...]:
    """Split a dense grid into `per_octave` interleaved FULL-OCTAVE ladders.

    THE READ THE DENSE GRID EXISTS FOR, and the only one that improves on the
    powers-of-two grid it replaced. Each returned ladder steps by a factor of
    two, so every local slope keeps the widest baseline the token range allows
    and none of the noise amplification described on `POINTS_PER_OCTAVE`
    happens. The `per_octave` crossings they produce are near-independent, so
    their median is sharper than any single ladder's and their SPREAD is a
    same-run empirical error bar rather than a bootstrapped one -- which is what
    `crossing.crossing_interval` has to synthesise today because no run ever
    carried two ladders.

    Derived from the token counts alone, with no reference to the ridge, the
    band or the model. That is deliberate: the ladders have to be recoverable
    from a published CSV months later by whoever is reading it, and a rule that
    needed the calibration that produced the grid would not be.

    Coarse points -- those already a full octave from both neighbours -- go into
    EVERY ladder. They sit below the crossing on a flat part of the curve, so
    sharing them correlates the ladders only where the slope is far from the
    threshold, and the alternative is ladders too short to have two slopes.

    Raises rather than returning a ladder with a sub-octave step, which would be
    a ladder with the very defect this function exists to avoid.
    """
    if per_octave < 1:
        raise ValueError(f"per_octave must be at least 1, got {per_octave}")
    points = sorted({int(t) for t in token_counts if t > 0})
    if not points:
        return tuple(() for _ in range(per_octave))

    octave = 2.0 / _GRID_DEDUP_RATIO
    coarse, dense = [], []
    for i, t in enumerate(points):
        below = t / points[i - 1] if i else math.inf
        above = points[i + 1] / t if i + 1 < len(points) else math.inf
        (coarse if min(below, above) >= octave else dense).append(t)

    ladders = []
    for j in range(per_octave):
        members = [t for t in dense
                   if round(per_octave * math.log2(t)) % per_octave == j]
        ladder = tuple(sorted(set(coarse) | set(members)))
        for a, b in zip(ladder, ladder[1:], strict=False):
            if b / a < octave:
                raise ValueError(
                    f"ladder {j} steps {a} -> {b}, under an octave; "
                    f"{sorted(points)} is not a 2^(k/{per_octave}) grid and "
                    "splitting it by residue does not produce octave ladders")
        ladders.append(ladder)
    return tuple(ladders)


#: Token grid for the `crossing-uniform` profile, built at import so the cost is
#: visible in a dry run rather than discovered on a rented box.
CROSSING_MODELS: tuple[str, ...] = (
    "mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v2-lite", "deepseek-v3")
CROSSING_TOKENS: tuple[int, ...] = crossing_grid(CROSSING_MODELS)

#: Shapes that actually get served, each paired with the TP=1 control it is a
#: shard of. The controls are not padding: the whole question is whether moving
#: to the width vLLM tunes for changes the tile, the block count and therefore
#: the timing, and a shard measured without its own unsharded twin in the same
#: session answers nothing.
DEPLOYMENT_MODELS: tuple[str, ...] = (
    "deepseek-v3", "deepseek-v3-tp4", "deepseek-v3-tp8",
    "mixtral-8x7b", "mixtral-8x7b-tp8",
    "qwen2-57b-a14b", "qwen2-57b-a14b-tp8")

PROFILES: dict[str, Profile] = {
    "smoke": Profile(
        name="smoke",
        models=("toy", "mixtral-8x7b"),
        token_counts=(1, 128),
        dtypes=("bf16",),
        routings=(RoutingSpec("uniform"), RoutingSpec("zipf", 1.2)),
        l2_modes=(True,),
        graph_modes=(False,),
        warmup=5,
        trials=1,
        iters=10,
        notes="shakedown: proves correctness and the plumbing, not performance",
    ),
    "standard": Profile(
        name="standard",
        models=("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v3"),
        token_counts=(1, 4, 16, 64, 256, 1024, 4096),
        dtypes=("bf16",),
        routings=SKEW_SWEEP,
        notes="the working sweep: decode through prefill, uniform through severe skew",
    ),
    # Counter profiling wants ONE launch, not a matrix: `ncu --launch-count 1`
    # takes whichever kernel runs first, so the cell has to be the only cell.
    # Token count is supplied per question with --tokens.
    # C2's prediction test. Same grid as `full` so the crossings are directly
    # comparable, one dtype changed. AI = 2R/b, so halving bytes must HALVE the
    # crossing: deepseek-v3 from ~5,100 tokens to ~2,570, mixtral 642 -> 321.
    # That is a 2x shift, which this powers-of-two grid separates without any
    # narrow runs, where the A100's 0.913x ridge ratio does not.
    #
    # Only vLLM and the reference cover fp8: torch's grouped_mm is bf16-only and
    # SGLang's runner takes a different config object that has not been probed.
    # Those spans decline the cell rather than being silently skipped.
    "fp8": Profile(
        name="fp8",
        models=("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v2-lite",
                "deepseek-v3"),
        token_counts=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                      4096, 8192),
        dtypes=("fp8_e4m3",),
        routings=SKEW_SWEEP + (RoutingSpec("zipf", 2.0), RoutingSpec("hot", 0.8)),
        seeds=(0, 1, 2),
    ),
    "profile-cell": Profile(
        name="profile-cell",
        models=("deepseek-v3",),
        token_counts=(4096,),
        dtypes=("bf16",),
        routings=(RoutingSpec("uniform"),),
        l2_modes=(True,),
        graph_modes=(False,),
        warmup=5,
        trials=1,
        iters=1,
        notes="one cell, one launch: the shape ncu can read a counter off",
    ),
    "full": Profile(
        name="full",
        # deepseek-v2-lite is here for its E/k of 10.7, which fills the gap
        # between qwen2's 8 and deepseek-v3's 32 and gives the dilution law a
        # fourth point. It is also the cheapest model in the set at 1.11 GB of
        # weights, so it costs the least of any axis that could be widened.
        models=("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v2-lite",
                "deepseek-v3"),
        token_counts=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
        dtypes=("bf16",),
        routings=SKEW_SWEEP + (RoutingSpec("zipf", 2.0), RoutingSpec("hot", 0.8)),
        seeds=(0, 1, 2),
        # The all-reference whole layer is OFF here as of 2026-08-28. It was
        # switched on in ffaf154 with no stated reason, and its 882 cells were
        # the several-hour part of this profile while measuring no kernel.
        # `--include-reference`-style use can still ask for it explicitly.
        include_pipeline_scope=False,
        include_framework_pipeline=True,
        notes="publication sweep; hours, not minutes",
    ),
    # The crossing is the study's central measurement and it has never been
    # measured well. Three faults, all of them in the matrix rather than in the
    # hardware, and this profile fixes all three at once because fixing one at a
    # time costs three rentals.
    #
    # 1. THE GRID. Powers of two interpolate a crossing across a 2x gap, and the
    #    grid ends at 8192 so no crossing above 5793 can be reported at all --
    #    which is where DeepSeek-V3's measured one-stage crossings are.
    #    `crossing_grid` densifies to 2^(1/4) over the predicted band and runs to
    #    16384 so every interleaved ladder reaches past it.
    # 2. THE REPLICATES. Three seeds is not enough: qwen2's crossing moves 1.39x
    #    across the three (593 / 779 / 824), traced to a SINGLE point at T=512
    #    where throttling dropped one of two replicate rows. Seven seeds survive
    #    the 33% throttle exclusion that arm saw near the crossing with four or
    #    five rows still standing, which is more than the current profile starts
    #    with.
    # 3. THE POOLING. `2R/b` describes uniform routing; under skew the busy
    #    experts are compute-bound while the quiet ones are still memory-bound AT
    #    THE SAME BATCH, so there is no single crossing to find. Pooling the
    #    seven regimes is invalid rather than noisy, and moves the answer by up
    #    to 4.3x. Uniform only, so no report can pool by accident.
    #
    # ONE TIMING MODE, not four, and this is what pays for the rest. Every
    # crossing in docs/FINDINGS.md is on the L2-cold eager basis, but the sweep
    # measures four modes and `crossing_report` medians across all of them by
    # default, mixing L2-warm and graph-replay rows into a number that is
    # reported as one basis. Dropping the other three cuts the cost 4x AND
    # removes a confound.
    "crossing-uniform": Profile(
        name="crossing-uniform",
        models=CROSSING_MODELS,
        token_counts=CROSSING_TOKENS,
        dtypes=("bf16",),
        routings=(RoutingSpec("uniform"),),
        seeds=(0, 1, 2, 3, 4, 5, 6),
        l2_modes=(True,),
        graph_modes=(False,),
        include_pipeline_scope=False,
        include_framework_pipeline=False,
        notes=("uniform only, 7 seeds, 2^(1/4) grid over the ridge band. "
               "READ IT WITH octave_ladders: fed whole to crossing_from_points "
               "this grid is biased 4-18% LOW and twice as wide as the "
               "powers-of-two grid it extends, and half as wide read properly"),
    ),
    # Shapes that actually get served. Every published cell is TP=1 bf16, and
    # `E=256,N=2048` -- unsharded DeepSeek-V3 -- is a shape no vLLM config ships
    # for on ANY device at v0.27.1, because it is 1.3 TB of weights and nobody
    # runs it.
    # What ships is `E=256,N=256` and `E=256,N=512`, the TP=8 and TP=4 widths.
    # So the study's headline model has been measured on the hardcoded fallback
    # tile ladder throughout, and that is a limitation of the study rather than
    # a gap in vLLM.
    #
    # Both dtypes, because the shard changes what each one resolves to and they
    # do not move together. In bf16 the mixtral and qwen2 shards hit tuned H200
    # configs and the DeepSeek-V3 shards still do not. In fp8 the DeepSeek-V3
    # shards have tuned files but only under `block_shape=[128,128]`, and this
    # harness quantises one scale per expert, so it will miss them until
    # `_framework_config` learns block-wise scales.
    #
    # Uniform only, for the same reason `crossing-uniform` is: the question here
    # is which kernel a shard resolves to, which is a function of M, and pooling
    # routings would put a crossing on top of it that means nothing.
    "deployment": Profile(
        name="deployment",
        models=DEPLOYMENT_MODELS,
        token_counts=COARSE_BACKBONE,
        dtypes=("bf16", "fp8_e4m3"),
        routings=(RoutingSpec("uniform"),),
        seeds=(0, 1, 2),
        l2_modes=(True,),
        graph_modes=(False,),
        include_pipeline_scope=False,
        include_framework_pipeline=False,
        notes=("TP=4 and TP=8 shard widths against their TP=1 controls, on the "
               "published powers-of-two grid so the controls line up with every "
               "existing arm"),
    ),
}

# --------------------------------------------------------------------------
# What a profile costs in wall clock
# --------------------------------------------------------------------------

#: Wall-clock seconds per TIMING ROW of an IMPLEMENTATION SPAN, by
#: `(env, dtype family)`. Measured from `results/published/*/run_*.csv` by
#: attributing each consecutive timestamp gap to the cell that opened it, so a
#: rate belongs to the span that earned it rather than to the file average. It
#: carries warmup, correctness checking, routing and CSV writing as well as the
#: kernel, because all of that is on the clock a pod bills.
#:
#: PER SPAN AND NOT PER FILE, which is the correction that made these numbers
#: usable. The base env's file average is 0.779, and a third of the rows behind
#: it are the all-reference whole-layer cell at 1.093 while its two real spans
#: run at 0.631. Averaging them over-estimates by 23% any profile that has the
#: reference layer switched off -- which every profile now does.
#:
#: fp8 is a separate column because it is not a small correction and it does not
#: even have a consistent sign: it costs vLLM and SGLang about 30% MORE per row
#: (quantisation and scale plumbing) and the base env about 10% more too, but a
#: bf16 profile and an fp8 one differ by more than the noise in either.
MEASURED_SECONDS_PER_ROW: dict[tuple[str, str], float] = {
    ("base", "bf16"): 0.631,     # 19,820 rows, torch_grouped_mm up and down
    ("base", "fp8"): 0.691,      # 19,140 rows, torch_scaled_grouped_mm
    ("vllm", "bf16"): 0.464,     # 18,864 rows
    ("vllm", "fp8"): 0.605,      #  5,284 rows
    ("sglang", "bf16"): 0.467,   #  9,468 rows
    ("sglang", "fp8"): 0.610,    #  4,700 rows
}

#: The all-reference whole-layer cell, `include_pipeline_scope`. 9,468 rows at
#: 1.093 s each: 1.7x a real span, which is why it is off in every profile. The
#: kernel it times is 12x slower than vLLM's and measures no kernel at all.
REFERENCE_PIPELINE_SECONDS_PER_ROW = 1.093

#: Framework spans that register per environment, COUNTED FROM THE PUBLISHED ARMS
#: rather than read out of the registry. On a laptop neither vLLM nor SGLang
#: imports, so `candidate_impls("vllm")` is empty and any estimate built on it
#: reports zero hours for the environment that costs the most -- which is exactly
#: the failure mode a pre-rental estimate exists to prevent.
#:
#: base is 2 in both dtypes: `torch_grouped_mm` up and down in bf16, their
#: `torch_scaled_` twins in fp8, and the two sets never coexist in one cell. The
#: third impl in the 2026-08-26 base arm was `__pipeline__`, which is the
#: reference layer above and is counted separately.
MEASURED_IMPLS_PER_ENV: dict[str, int] = {"base": 2, "vllm": 1, "sglang": 1}


def _dtype_family(dtype: str) -> str:
    return "fp8" if dtype in FP8_WEIGHT_DTYPES else "bf16"


def estimated_hours(profile: Profile) -> dict[str, float]:
    """Wall-clock hours this profile would cost, per environment plus `total`.

    A profile nobody can afford to run is not a contribution, and until this
    existed the only way to find out what one cost was to start it and watch.
    Needs no GPU, no registry and no framework import, so it answers on the
    laptop the profile is being written on.

    The whole-layer cells are priced at the span rate, not guessed: the vLLM arms
    measured `__pipeline__:vllm_fused_experts` at 0.460 against the bare span's
    0.464, a 1% difference, because the pipeline cell is the same fused span with
    a router in front of it. The all-reference layer is the one that is genuinely
    slower and it has its own constant.

    AN ESTIMATE, and it will be low in one known direction. The rates average
    over grids that stopped at 8192, and a cell's cost rises with the batch once
    the layer is compute-bound, so a profile reaching past that (`crossing-uniform`
    goes to 16384) costs more per row at its top end than this says. It is a
    figure to plan a rental against, not a budget with margin built in.
    """
    specs = len(profile.specs())
    modes = len(profile.l2_modes) * len(profile.graph_modes)
    families = [_dtype_family(d) for d in profile.dtypes] or ["bf16"]

    out: dict[str, float] = {}
    for env, impls in MEASURED_IMPLS_PER_ENV.items():
        per_spec = impls
        if profile.include_framework_pipeline and env != BASE_ENV:
            per_spec += impls          # one whole-layer cell per framework span
        rows_per_family = specs * per_spec * modes / len(families)
        seconds = sum(rows_per_family * MEASURED_SECONDS_PER_ROW[(env, f)]
                      for f in families)
        if profile.include_pipeline_scope and env == BASE_ENV:
            # Emitted once per spec regardless of env, and it is the slow one.
            seconds += specs * modes * REFERENCE_PIPELINE_SECONDS_PER_ROW
        out[env] = seconds / 3600.0
    out["total"] = sum(out.values())
    return out


def tiling_for(span: StageSpan) -> list[str]:
    """Put one implementation in context: it covers its stages, the reference
    covers the rest. Every implementation is therefore measured inside a
    complete, correctness-checkable layer."""
    names: list[str] = []
    for stage in CANONICAL_STAGES:
        if stage in span.covers:
            if stage == span.covers[0]:
                names.append(span.name)
        else:
            names.append(f"ref_{stage}")
    return names


def candidate_impls(env: str | None = None,
                    include_reference: bool = False) -> list[StageSpan]:
    """Implementations worth benchmarking: everything registered except the
    reference spans, which exist to be correct rather than fast."""
    out = []
    for span in registry().values():
        if not include_reference and span.name.startswith("ref_"):
            continue
        if env is not None and span.env != env:
            continue
        out.append(span)
    return sorted(out, key=lambda s: s.name)


def cells(profile: Profile, env: str | None = None,
          impl_filter: tuple[str, ...] = (),
          include_reference: bool = False):
    """Yield (spec, pipeline names, impl) triples for the driver."""
    impls = candidate_impls(env=env, include_reference=include_reference)
    if impl_filter:
        impls = [s for s in impls if s.name in impl_filter]
    for spec in profile.specs():
        for span in impls:
            if not span.supports(spec):
                continue
            yield spec, tiling_for(span), span.name
        # The all-reference pipeline cell is framework-independent, so it is
        # emitted once, under base. Yielding it for every env ran the slowest
        # cells in the matrix three times over for identical rows.
        if profile.include_pipeline_scope and env in (None, BASE_ENV):
            from .driver import PIPELINE_SCOPE
            yield spec, reference_pipeline_names(), PIPELINE_SCOPE
        # Whole-layer cells around a real kernel. Not in base: there is no
        # framework span there to wrap, and the all-reference layer is the
        # other flag's job.
        if profile.include_framework_pipeline and env != BASE_ENV:
            from .driver import pipeline_scope_for
            for span in impls:
                if span.env == BASE_ENV or not span.supports(spec):
                    continue
                yield spec, tiling_for(span), pipeline_scope_for(span.name)


@dataclass(frozen=True)
class Plan:
    """What a sweep would do, computed without touching a GPU.

    Separated from its presentation so the counting is testable. A dry run that
    miscounts is exactly the failure the whole laptop-side design exists to
    prevent, and it previously had no test at all.
    """

    profile: Profile
    env: str | None
    impls: tuple[str, ...]
    specs: int
    planned: int
    unsupported: int
    problems: tuple[str, ...]
    missing_traces: tuple[str, ...]

    @property
    def modes(self) -> int:
        return len(self.profile.l2_modes) * len(self.profile.graph_modes)

    @property
    def timing_rows(self) -> int:
        return self.planned * self.modes

    @property
    def ok(self) -> bool:
        return not self.problems and not self.missing_traces


def plan(profile: Profile, env: str | None = None,
         impl_filter: tuple[str, ...] = (), include_reference: bool = False,
         traces=None) -> Plan:
    """Build and validate every tiling in the matrix. No GPU, nothing spent."""
    from ..pipeline import PipelineError, build

    impls = candidate_impls(env=env, include_reference=include_reference)
    if impl_filter:
        impls = [s for s in impls if s.name in impl_filter]

    specs = profile.specs()

    # Counted directly. `cells()` filters unsupported spans before yielding, so
    # inferring this from build failures always reported zero.
    unsupported = sum(1 for spec in specs for span in impls
                      if not span.supports(spec))

    problems: list[str] = []
    planned = 0
    for spec, names, impl in cells(profile, env=env, impl_filter=impl_filter,
                                   include_reference=include_reference):
        try:
            build(names, spec=spec)
            planned += 1
        except PipelineError as e:
            problems.append(f"{spec.label} [{impl}]: {e}")

    needed = {s.routing.trace_id for s in specs if s.routing.kind == "trace"}
    missing = sorted(t for t in needed if traces is None or t not in traces)

    return Plan(profile=profile, env=env,
                impls=tuple(s.name for s in impls), specs=len(specs),
                planned=planned, unsupported=unsupported,
                problems=tuple(problems), missing_traces=tuple(missing))


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(PROFILES)}") from None
