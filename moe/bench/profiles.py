"""Benchmark matrices, sized to a time budget.

Three profiles. `smoke` exists so the first thing a GPU session does is prove
the harness works, in about two minutes, before committing to a long sweep.

Token counts deliberately start at 1. A single token is the decode regime, and
decode with many experts is the memory-bound weight-loading wall this project
targets. A sweep that starts at 512 tokens would miss the entire phenomenon.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
from ..stages import CANONICAL_STAGES, StageSpan, registry


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
        out = []
        for name in self.models:
            cfg = MODEL_CONFIGS[name]
            for tokens in self.token_counts:
                for dtype in self.dtypes:
                    for routing in self.routings:
                        for seed in self.seeds:
                            out.append(BenchSpec(cfg, tokens, dtype, routing, seed))
        return out


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
    "full": Profile(
        name="full",
        models=("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v3"),
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
        if profile.include_pipeline_scope:
            from .driver import PIPELINE_SCOPE
            ref = [f"ref_{s}" for s in CANONICAL_STAGES]
            yield spec, ref, PIPELINE_SCOPE


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(PROFILES)}") from None
