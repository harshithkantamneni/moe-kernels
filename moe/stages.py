"""Canonical MoE stages, the span abstraction, and the implementation registry.

An implementation declares the *contiguous span* of canonical stages it covers.
A fused down-projection + scatter kernel is `covers=("down_gemm", "unpermute")`.
An unfused pair is two spans of one stage each. Both are tilings of the same
pipeline, so they are benchmarked by one driver and checked against one oracle.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

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

#: The environment that owns the harness and your kernels. Spans declaring this
#: run in whatever process is executing, so they compose with any other env.
BASE_ENV = "base"

#: Environments the runner knows how to satisfy. Closed on purpose: an open
#: string means a typo like env="vlllm" silently makes candidate_impls return
#: nothing and the sweep benchmarks zero implementations without an error.
KNOWN_ENVS: frozenset[str] = frozenset({BASE_ENV, "vllm", "sglang"})


def resolve_env(spans) -> str:
    """The single environment a set of spans must execute in.

    `base` composes with anything, so the answer is the unique non-base env if
    there is one. Raises if two frameworks are named, because no process can
    satisfy both.
    """
    envs = {s.env for s in spans if s.env != BASE_ENV}
    if len(envs) > 1:
        owners = {s.name: s.env for s in spans if s.env != BASE_ENV}
        raise ValueError(
            f"mixes incompatible environments {sorted(envs)}: {owners}. "
            "Baselines from different frameworks cannot share one process; "
            "benchmark them as separate pipelines.")
    return envs.pop() if envs else BASE_ENV


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


#: Fields the harness itself consumes after the layer runs, so they must be
#: materialised even though no downstream STAGE reads them.
PIPELINE_OUTPUTS: frozenset[str] = frozenset({"y"})


def exposed_writes(covers: tuple[str, ...]) -> frozenset[str]:
    """Fields a span produces that could be visible outside it.

    Only used when costing a span with no tiling context. Inside a pipeline,
    what actually escapes is computed from liveness: see pipeline.live_outputs.
    """
    return STAGE_CONTRACTS[covers[-1]].writes


def live_outputs(spans) -> list[frozenset[str]]:
    """Per span, the fields it produces that something later actually reads.

    This replaces a hand-written declaration, and the stage graph shows why it
    can: every non-final field a span writes is either read by a span that must
    be downstream, or unreadable because its only reader is the stage the
    fusion swallowed. There is no third case, so a `materialises` tuple never
    carried information that could not be derived, and forgetting it punished a
    correct kernel while silently changing its published byte count.
    """
    out: list[frozenset[str]] = []
    for i, span in enumerate(spans):
        later: set[str] = set(PIPELINE_OUTPUTS)
        for other in spans[i + 1:]:
            later |= other.contract.reads
        out.append(frozenset(span.contract.writes & later))
    return out


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
    #: Fields this implementation writes to memory beyond what the tiling
    #: requires. Affects the BYTES MODEL only, never availability: a reference
    #: implementation that stores an intermediate nothing reads still pays for
    #: the store, and the cost model should say so.
    materialises: tuple[str, ...] = ()
    #: Fields this implementation physically cannot produce because it fuses
    #: them into registers. If a tiling needs one of these, `pipeline.build`
    #: rejects it by name instead of failing at run time with a None field.
    cannot_materialise: tuple[str, ...] = ()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.covers:
            cls.contract = contract_for(cls.covers)

    contract: Contract = Contract(frozenset(), frozenset())

    @property
    def writes(self) -> frozenset[str]:
        """What this span materialises when costed outside a tiling.

        Inside a pipeline the real answer is per-tiling; see
        `Pipeline.materialised_for`.
        """
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


def load_package(package_path, package_name: str, kind: str,
                 strict: bool = False) -> list[str]:
    """Import every module in a package so its @register-ed spans appear.

    A module that fails to import (a Triton kernel on a machine without CUDA, a
    baseline whose framework is not installed in this venv) is skipped with a
    warning rather than crashing the harness, so the laptop can still run the
    CPU test suite.
    """
    import importlib
    import pkgutil
    import warnings

    loaded = []
    for mod in pkgutil.iter_modules(package_path):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{package_name}.{mod.name}")
            loaded.append(mod.name)
        except Exception as e:  # noqa: BLE001
            if strict:
                raise
            warnings.warn(f"{kind} {mod.name!r} did not import: {e}", stacklevel=2)
    return loaded


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, StageSpan] = {}


def register(klass: type[StageSpan]) -> type[StageSpan]:
    """Class decorator. Instantiates once and stores under `klass.name`."""
    inst = klass()
    if not inst.name:
        raise ValueError(f"{klass.__name__} must set a non-empty `name`")
    if inst.name in _REGISTRY:
        raise ValueError(f"duplicate span name {inst.name!r}")
    if not inst.covers:
        raise ValueError(f"{inst.name} must set `covers`")
    if inst.env not in KNOWN_ENVS:
        raise ValueError(
            f"{inst.name} declares env {inst.env!r}; known: {sorted(KNOWN_ENVS)}")
    _REGISTRY[inst.name] = inst
    return klass


def get(name: str) -> StageSpan:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no span named {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def registry() -> dict[str, StageSpan]:
    return dict(_REGISTRY)


