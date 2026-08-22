"""Configuration objects: what a benchmark cell *is*.

Nothing here touches CUDA. Everything is importable and testable on a laptop.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

# --------------------------------------------------------------------------
# Numeric formats
# --------------------------------------------------------------------------

# name -> bytes per element. Kept as plain data so the roofline/bytes model
# never needs torch, and so FP8 slots in without touching call sites.
DTYPE_BYTES: dict[str, int] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}

# Formats the project currently exercises end to end. FP8 is defined above so
# the bytes/roofline model is already dtype-parametric, but it is deliberately
# not in the active set yet (see docs/DECISIONS.md).
ACTIVE_DTYPES: tuple[str, ...] = ("bf16",)


def dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(
            f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}"
        ) from None


def torch_dtype(dtype: str):
    """Resolve to a torch dtype. Imported lazily so this module stays CPU/laptop safe."""
    import torch

    table = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3": getattr(torch, "float8_e4m3fn", None),
        "fp8_e5m2": getattr(torch, "float8_e5m2", None),
    }
    resolved = table.get(dtype)
    if resolved is None:
        raise ValueError(f"dtype {dtype!r} is not supported by this torch build")
    return resolved


# --------------------------------------------------------------------------
# Model geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MoEConfig:
    """Geometry of one MoE layer. Weights are never loaded; only shapes matter.

    `verified` is False until scripts/verify_model_configs.py has diffed these
    numbers against the upstream config.json. Do not publish benchmark numbers
    labelled with a model name while its config is unverified.
    """

    name: str
    hidden_size: int          # H
    intermediate_size: int    # F, per routed expert
    num_experts: int          # E, routed experts only
    top_k: int                # k
    act: str = "swiglu"
    norm_topk_prob: bool = True   # renormalise the top-k gate probs to sum to 1
    shared_experts: int = 0
    hf_repo: str | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        if self.top_k > self.num_experts:
            raise ValueError(
                f"{self.name}: top_k={self.top_k} exceeds num_experts={self.num_experts}"
            )
        for field in ("hidden_size", "intermediate_size", "num_experts", "top_k"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{self.name}: {field} must be positive")

    @property
    def w1_shape(self) -> tuple[int, int, int]:
        """Fused gate+up projection, [E, 2F, H] (vLLM/SGLang layout)."""
        return (self.num_experts, 2 * self.intermediate_size, self.hidden_size)

    @property
    def w2_shape(self) -> tuple[int, int, int]:
        """Down projection, [E, H, F]."""
        return (self.num_experts, self.hidden_size, self.intermediate_size)

    def weight_bytes(self, dtype: str) -> int:
        n = (self.w1_shape[0] * self.w1_shape[1] * self.w1_shape[2]
             + self.w2_shape[0] * self.w2_shape[1] * self.w2_shape[2])
        return n * dtype_bytes(dtype)


# Shapes only. `verified` flips to True per model once checked against the
# upstream config.json, which is what scripts/verify_model_configs.py does.
MODEL_CONFIGS: dict[str, MoEConfig] = {
    "mixtral-8x7b": MoEConfig(
        name="mixtral-8x7b",
        hidden_size=4096,
        intermediate_size=14336,
        num_experts=8,
        top_k=2,
        hf_repo="mistralai/Mixtral-8x7B-Instruct-v0.1",
    ),
    "qwen2-57b-a14b": MoEConfig(
        name="qwen2-57b-a14b",
        hidden_size=3584,
        intermediate_size=2560,
        num_experts=64,
        top_k=8,
        shared_experts=1,
        hf_repo="Qwen/Qwen2-57B-A14B-Instruct",
    ),
    "deepseek-v3": MoEConfig(
        name="deepseek-v3",
        hidden_size=7168,
        intermediate_size=2048,
        num_experts=256,
        top_k=8,
        shared_experts=1,
        hf_repo="deepseek-ai/DeepSeek-V3",
    ),
    # Tiny synthetic geometry for CPU unit tests and laptop-side debugging.
    "toy": MoEConfig(
        name="toy",
        hidden_size=64,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        verified=True,
    ),
}


# --------------------------------------------------------------------------
# Routing distribution
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingSpec:
    """How tokens are assigned to experts.

    kind="trace" replays a captured real distribution (traces/), which is the
    differentiating axis of this project. The parametric kinds exist to sweep
    imbalance continuously and to make the load-imbalance story legible.
    """

    kind: str                      # uniform | zipf | dirichlet | hot | trace
    param: float = 0.0             # zipf exponent | dirichlet alpha | hot fraction
    trace_id: str | None = None

    KINDS = ("uniform", "zipf", "dirichlet", "hot", "trace")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(f"unknown routing kind {self.kind!r}; known: {self.KINDS}")
        if self.kind == "trace" and not self.trace_id:
            raise ValueError("routing kind 'trace' requires a trace_id")
        if self.kind != "trace" and self.trace_id:
            raise ValueError(f"trace_id is meaningless for routing kind {self.kind!r}")
        if self.kind == "hot" and not (0.0 < self.param <= 1.0):
            raise ValueError("hot routing needs param in (0, 1]: the hot-expert mass")
        if self.kind == "zipf" and self.param < 0.0:
            raise ValueError("zipf exponent must be non-negative")
        if self.kind == "dirichlet" and self.param <= 0.0:
            raise ValueError("dirichlet alpha must be positive")

    @property
    def label(self) -> str:
        if self.kind == "trace":
            return f"trace:{self.trace_id}"
        if self.kind == "uniform":
            return "uniform"
        return f"{self.kind}:{self.param:g}"


# --------------------------------------------------------------------------
# One cell of the benchmark matrix
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchSpec:
    """A single (model, tokens, dtype, routing) point. One CSV row per cell per impl."""

    model: MoEConfig
    num_tokens: int               # T, tokens entering the layer
    dtype: str = "bf16"
    routing: RoutingSpec = RoutingSpec("uniform")
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        dtype_bytes(self.dtype)  # validates

    @property
    def rows(self) -> int:
        """Rows entering the grouped GEMMs: every token is replicated top_k times."""
        return self.num_tokens * self.model.top_k

    @property
    def mean_rows_per_expert(self) -> float:
        return self.rows / self.model.num_experts

    @property
    def label(self) -> str:
        return (f"{self.model.name}/T{self.num_tokens}/{self.dtype}/"
                f"{self.routing.label}/s{self.seed}")

    def with_(self, **kw) -> BenchSpec:
        return replace(self, **kw)


def sweep(
    models: list[MoEConfig],
    token_counts: list[int],
    dtypes: list[str],
    routings: list[RoutingSpec],
    seeds: list[int] | None = None,
) -> Iterator[BenchSpec]:
    """Cartesian product of the benchmark axes, in a deterministic order."""
    for model in models:
        for tokens in token_counts:
            for dtype in dtypes:
                for routing in routings:
                    for seed in (seeds or [0]):
                        yield BenchSpec(model, tokens, dtype, routing, seed)
