"""Canonical MoE stages, the span abstraction, and the implementation registry.

An implementation declares the *contiguous span* of canonical stages it covers.
A fused down-projection + scatter kernel is `covers=("down_gemm", "unpermute")`.
An unfused pair is two spans of one stage each. Both are tilings of the same
pipeline, so they are benchmarked by one driver and checked against one oracle.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable

from .spec import BenchSpec
from .state import MoEState

# Order matters: a span must cover a contiguous slice of this tuple.
CANONICAL_STAGES: tuple[str, ...] = (
    "router",
    "permute",
    "up_gemm",
    "act",
    "down_gemm",
    "unpermute",
)

STAGE_INDEX: dict[str, int] = {s: i for i, s in enumerate(CANONICAL_STAGES)}


@dataclass(frozen=True)
class Contract:
    reads: frozenset[str]
    writes: frozenset[str]


# Per-stage data contract. `weights` is always available and so is not listed.
STAGE_CONTRACTS: dict[str, Contract] = {
    "router": Contract(
        reads=frozenset({"x"}),
        writes=frozenset({"router_logits", "topk_ids", "topk_weights"}),
    ),
    "permute": Contract(
        reads=frozenset({"x", "topk_ids"}),
        writes=frozenset({"expert_offsets", "perm_index", "x_perm"}),
    ),
    "up_gemm": Contract(
        reads=frozenset({"x_perm", "expert_offsets"}),
        writes=frozenset({"h_up"}),
    ),
    "act": Contract(
        reads=frozenset({"h_up"}),
        writes=frozenset({"h_act"}),
    ),
    "down_gemm": Contract(
        reads=frozenset({"h_act", "expert_offsets"}),
        writes=frozenset({"y_perm"}),
    ),
    "unpermute": Contract(
        reads=frozenset({"y_perm", "perm_index", "topk_weights"}),
        writes=frozenset({"y"}),
    ),
}


def contiguous(covers: Iterable[str]) -> bool:
    idx = [STAGE_INDEX[s] for s in covers]
    return idx == list(range(idx[0], idx[0] + len(idx)))


def contract_for(covers: tuple[str, ...]) -> Contract:
    """Derive a span's contract from the stages it covers.

    A span reads whatever its stages read minus whatever it produces internally,
    and writes the union of its stages' writes. A fused span therefore does not
    have to declare the intermediates it never materialises: fusing down_gemm
    with unpermute means y_perm is a register value, not a state field.
    """
    if not covers:
        raise ValueError("a span must cover at least one stage")
    unknown = [s for s in covers if s not in STAGE_INDEX]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; known: {CANONICAL_STAGES}")
    if len(set(covers)) != len(covers):
        raise ValueError(f"span covers a stage twice: {covers}")
    if not contiguous(covers):
        raise ValueError(
            f"span {covers} is not contiguous in {CANONICAL_STAGES}; "
            "fusing across a gap would change layer semantics"
        )

    reads: set[str] = set()
    writes: set[str] = set()
    for stage in covers:
        c = STAGE_CONTRACTS[stage]
        reads |= c.reads - writes  # satisfied internally by an earlier stage
        writes |= c.writes
    return Contract(frozenset(reads), frozenset(writes))


def exposed_writes(covers: tuple[str, ...]) -> frozenset[str]:
    """Fields a span must actually materialise for later spans.

    The final stage's writes always escape. Intermediates only escape if the
    span chose to materialise them, which the span declares via `materialises`.
    """
    return STAGE_CONTRACTS[covers[-1]].writes


class StageSpan(ABC):
    """One benchmarkable implementation covering a contiguous run of stages."""

    name: str = ""
    covers: tuple[str, ...] = ()

    #: which venv this implementation must run inside (see moe/runner)
    env: str = "base"
    #: False for pure-torch reference paths that run on a laptop
    requires_cuda: bool = True
    #: safe to capture inside a CUDA graph: no .item(), no host-side loop over experts
    cuda_graph_safe: bool = False
    #: dtypes this implementation accepts
    dtypes: tuple[str, ...] = ("bf16",)
    #: intermediates this span materialises into state beyond its final stage's
    #: writes. A fused span usually leaves this empty.
    materialises: tuple[str, ...] = ()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.covers:
            cls.contract = contract_for(cls.covers)

    contract: Contract = Contract(frozenset(), frozenset())

    @property
    def writes(self) -> frozenset[str]:
        return exposed_writes(self.covers) | frozenset(self.materialises)

    @property
    def reads(self) -> frozenset[str]:
        return self.contract.reads

    def supports(self, spec: BenchSpec) -> bool:
        """Override to reject geometries an implementation cannot handle, e.g. a
        kernel whose BLOCK_N assumes intermediate_size is a multiple of 128."""
        return spec.dtype in self.dtypes

    def why_unsupported(self, spec: BenchSpec) -> str:
        if spec.dtype not in self.dtypes:
            return f"dtype {spec.dtype} not in {self.dtypes}"
        return "unsupported for an unstated reason"

    @abstractmethod
    def __call__(self, st: MoEState) -> None:
        """Mutate `st` in place, producing every field in `self.writes`."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{self.name} covers={'+'.join(self.covers)} env={self.env}>"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, StageSpan] = {}


def register(cls: type[StageSpan] | None = None, **overrides) -> Callable | type:
    """Class decorator. Instantiates once and stores under `cls.name`."""

    def _wrap(klass: type[StageSpan]):
        inst = klass(**overrides) if overrides else klass()
        if not inst.name:
            raise ValueError(f"{klass.__name__} must set a non-empty `name`")
        if inst.name in _REGISTRY:
            raise ValueError(f"duplicate span name {inst.name!r}")
        if not inst.covers:
            raise ValueError(f"{inst.name} must set `covers`")
        _REGISTRY[inst.name] = inst
        return klass

    return _wrap(cls) if cls is not None else _wrap


def get(name: str) -> StageSpan:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no span named {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def registry() -> dict[str, StageSpan]:
    return dict(_REGISTRY)


def find(
    covers: tuple[str, ...] | None = None,
    spec: BenchSpec | None = None,
    env: str | None = None,
) -> list[StageSpan]:
    out = []
    for span in _REGISTRY.values():
        if covers is not None and span.covers != covers:
            continue
        if env is not None and span.env != env:
            continue
        if spec is not None and not span.supports(spec):
            continue
        out.append(span)
    return sorted(out, key=lambda s: s.name)


def _reset_registry_for_tests() -> None:
    _REGISTRY.clear()
