"""Tiling registered spans into a runnable MoE layer.

Every failure mode here is a failure that would otherwise cost GPU minutes:
a gap in the stage coverage, an overlap, a span whose input nobody produces,
or two spans that cannot coexist in one process because they need different
virtual environments.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import stages as S
from .spec import BenchSpec
from .state import MoEState


class PipelineError(ValueError):
    """Raised for any invalid tiling. Always names the offending span."""


@dataclass(frozen=True)
class Pipeline:
    spans: tuple[S.StageSpan, ...]

    @property
    def label(self) -> str:
        return " -> ".join(s.name for s in self.spans)

    @property
    def env(self) -> str:
        """The single venv this pipeline must execute in."""
        envs = {s.env for s in self.spans if s.env != "base"}
        return envs.pop() if envs else "base"

    @property
    def requires_cuda(self) -> bool:
        return any(s.requires_cuda for s in self.spans)

    @property
    def cuda_graph_safe(self) -> bool:
        return all(s.cuda_graph_safe for s in self.spans)

    def run(self, st: MoEState, validate_shapes: bool = False) -> MoEState:
        for span in self.spans:
            span(st)
            st.mark_written(span.writes)
            if validate_shapes:
                try:
                    st.validate(only=span.writes)
                except ValueError as e:
                    raise PipelineError(f"{span.name}: {e}") from None
        return st


def build(names: list[str] | tuple[str, ...], spec: BenchSpec | None = None) -> Pipeline:
    """Resolve span names into a validated Pipeline.

    `spec` is optional; pass it to also reject spans that do not support the
    geometry or dtype of the cell you are about to run.
    """
    spans = tuple(S.get(n) for n in names)
    _check_coverage(spans)
    _check_dataflow(spans)
    _check_env(spans)
    if spec is not None:
        _check_support(spans, spec)
    return Pipeline(spans)


# --------------------------------------------------------------------------


def _check_coverage(spans: tuple[S.StageSpan, ...]) -> None:
    covered: list[str] = []
    for span in spans:
        covered.extend(span.covers)

    if covered == list(S.CANONICAL_STAGES):
        return

    seen: set[str] = set()
    for span in spans:
        dup = seen & set(span.covers)
        if dup:
            raise PipelineError(
                f"{span.name} re-covers stage(s) {sorted(dup)} already covered upstream"
            )
        seen |= set(span.covers)

    missing = [s for s in S.CANONICAL_STAGES if s not in seen]
    if missing:
        raise PipelineError(
            f"pipeline leaves stage(s) {missing} uncovered; "
            f"got {covered}, need {list(S.CANONICAL_STAGES)}"
        )
    raise PipelineError(
        f"pipeline covers every stage but out of canonical order: "
        f"got {covered}, need {list(S.CANONICAL_STAGES)}"
    )


def _check_dataflow(spans: tuple[S.StageSpan, ...]) -> None:
    # `x` is supplied by the harness before the first span runs.
    available: set[str] = {"x"}
    for span in spans:
        missing = span.reads - available
        if missing:
            raise PipelineError(
                f"{span.name} reads {sorted(missing)}, which no upstream span writes. "
                f"available at this point: {sorted(available)}"
            )
        available |= set(span.writes)


def _check_env(spans: tuple[S.StageSpan, ...]) -> None:
    envs = {s.env for s in spans if s.env != "base"}
    if len(envs) > 1:
        owners = {s.name: s.env for s in spans if s.env != "base"}
        raise PipelineError(
            f"pipeline mixes incompatible environments {sorted(envs)}: {owners}. "
            "Baselines from different frameworks cannot share one process; "
            "benchmark them as separate pipelines."
        )


def _check_support(spans: tuple[S.StageSpan, ...], spec: BenchSpec) -> None:
    for span in spans:
        if not span.supports(spec):
            raise PipelineError(
                f"{span.name} does not support {spec.label}: {span.why_unsupported(spec)}"
            )


def reference_pipeline_names() -> list[str]:
    """The all-reference tiling: one span per canonical stage, torch only.

    This is the correctness oracle and the slowest possible configuration.
    """
    return [f"ref_{stage}" for stage in S.CANONICAL_STAGES]
