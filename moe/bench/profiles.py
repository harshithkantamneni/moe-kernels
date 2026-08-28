"""Benchmark matrices, sized to a time budget.

Three profiles. `smoke` exists so the first thing a GPU session does is prove
the harness works, in about two minutes, before committing to a long sweep.

Token counts deliberately start at 1. A single token is the decode regime, and
decode with many experts is the memory-bound weight-loading wall this project
targets. A sweep that starts at 512 tokens would miss the entire phenomenon.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..pipeline import reference_pipeline_names
from ..spec import MODEL_CONFIGS, BenchSpec, RoutingSpec, sweep
from ..stages import BASE_ENV, CANONICAL_STAGES, StageSpan, registry


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
    include_pipeline_scope: bool = False
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
        include_pipeline_scope=True,
        notes="publication sweep; hours, not minutes",
    ),
}


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
