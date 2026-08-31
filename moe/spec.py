"""Configuration objects: what a benchmark cell *is*.

Nothing here touches CUDA. Everything is importable and testable on a laptop.
"""
from __future__ import annotations

import functools
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

#: `spec.dtype` names the WEIGHT format. In an fp8 cell the activations stay at
#: the compute dtype, for two independent reasons that happen to agree.
#:
#: The kernel demands it. vLLM's `fused_experts` asserts
#: `hidden_states.dtype in [float32, float16, bfloat16]` and quantises the
#: activations itself, because their scale depends on the values and is only
#: known at run time. 147 cells died on that assertion.
#:
#: The experiment wants it. C2 says `AI = 2R/b` with b the WEIGHT width, because
#: weights are E*H*I bytes against the activations' T*H: for mixtral at T=512
#: the weights outweigh them by over 300x. Halving the weight width is what
#: moves the ridge crossing; halving the activation width moves nothing
#: measurable and changes what the kernel is asked to do.
_FP8_ACTIVATION_DTYPE = "bf16"


def activation_dtype(dtype: str) -> str:
    """The dtype activations are drawn in for a cell whose WEIGHTS are `dtype`.

    Identity for every float format, bf16 for the fp8 ones.
    """
    dtype_bytes(dtype)   # reject unknown formats here rather than downstream
    return _FP8_ACTIVATION_DTYPE if dtype in FP8_WEIGHT_DTYPES else dtype


#: Kept here rather than imported from moe.quant: spec.py is the bottom of the
#: import graph and must not depend on a module that imports torch.
FP8_WEIGHT_DTYPES: frozenset[str] = frozenset({"fp8_e4m3", "fp8_e5m2"})


def dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(
            f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}"
        ) from None


@functools.cache
def _torch_dtypes() -> dict:
    """Built once. torch is imported lazily so this module stays laptop-safe."""
    import torch

    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3": getattr(torch, "float8_e4m3fn", None),
        "fp8_e5m2": getattr(torch, "float8_e5m2", None),
    }


def torch_dtype(dtype: str):
    """Resolve to a torch dtype."""
    resolved = _torch_dtypes().get(dtype)
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

    `intermediate_size` IS THE PER-SHARD WIDTH, always. At `tensor_parallel = 1`
    that is the model's own `moe_intermediate_size` and nothing has changed; at
    TP=8 it is that number over eight, and `full_intermediate_size` recovers what
    the checkpoint holds. The unsharded width is derived rather than stored so
    the two can never disagree.

    WHY THE SHARD WIDTH LIVES HERE AND NOT ON `BenchSpec`, which is the other
    plausible home. Two reasons, and the second is the decisive one.

    Eight modules read `cfg.intermediate_size` and every one of them means "the
    width the kernel sees": `state.py` allocates it, `bytes_model.py` charges
    traffic for it, `torch_ref.py` shapes the oracle from it, `tolerance.py`
    sizes the error budget on it, and `_framework_config.py` hands it to SGLang
    under the name `intermediate_size_per_partition`. A shard width carried on
    the spec would leave all eight at the unsharded width, silently, and the
    only symptom would be a benchmark labelled TP=8 that ran the TP=1 kernel.

    And the analysis layer addresses a model BY NAME. `ridge.crossing_batch`
    takes a string and looks it up in `MODEL_CONFIGS`; `--models` validates
    against the same dict; the CSV records `model` and nothing else that could
    carry a shard factor without a schema change, and the schema is at v4. A
    shard width on the spec would be invisible to every prediction and
    unrecoverable from a published row. As a named config it is both.
    """

    name: str
    hidden_size: int          # H
    intermediate_size: int    # F PER SHARD, per routed expert
    num_experts: int          # E, routed experts only. TP does not shard these
    top_k: int                # k
    #: Tensor-parallel width this geometry describes. Pure TP shards the expert
    #: intermediate dimension and leaves E and H whole, which is why `E` above
    #: carries a comment and this field does not divide it. Expert parallelism
    #: would be the other axis and this study does not model it.
    tensor_parallel: int = 1
    gate_fn: str = "softmax"      # softmax | sigmoid (DeepSeek-V3 uses sigmoid)
    norm_topk_prob: bool = True   # renormalise the top-k gate probs to sum to 1
    routed_scaling_factor: float = 1.0  # applied AFTER renormalisation
    shared_experts: int = 0
    num_layers: int = 0
    first_moe_layer: int = 0      # layers below this index are dense
    hf_repo: str | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        if self.top_k > self.num_experts:
            raise ValueError(
                f"{self.name}: top_k={self.top_k} exceeds num_experts={self.num_experts}"
            )
        for field in ("hidden_size", "intermediate_size", "num_experts", "top_k",
                      "tensor_parallel"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{self.name}: {field} must be positive")
        if self.gate_fn not in ("softmax", "sigmoid"):
            raise ValueError(f"{self.name}: unknown gate_fn {self.gate_fn!r}")

    @property
    def num_moe_layers(self) -> int:
        return max(0, self.num_layers - self.first_moe_layer)

    @property
    def full_intermediate_size(self) -> int:
        """`moe_intermediate_size` as the upstream config.json states it.

        The width of the WHOLE expert, across every shard. Use it for anything
        that talks about the model rather than about this process's slice of it:
        a checkpoint's size, a diff against config.json. `download_bytes` is the
        one caller today, and it was wrong for a sharded config before this
        existed, reporting a TP=8 entry's checkpoint at an eighth of its size.
        """
        return self.intermediate_size * self.tensor_parallel

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
        # Mixtral has no moe_intermediate_size: the dense intermediate_size IS
        # the per-expert FFN width.
        intermediate_size=14336,
        num_experts=8,
        top_k=2,
        gate_fn="softmax",
        # config.json has no norm_topk_prob key. MixtralSparseMoeBlock.forward
        # renormalises unconditionally, so this is True by code, not by config.
        norm_topk_prob=True,
        shared_experts=0,
        num_layers=32,
        first_moe_layer=0,
        hf_repo="mistralai/Mixtral-8x7B-Instruct-v0.1",
        verified=True,
    ),
    "qwen2-57b-a14b": MoEConfig(
        name="qwen2-57b-a14b",
        hidden_size=3584,
        # moe_intermediate_size. The dense intermediate_size (18944) is a decoy
        # and applies only to the shared expert path, which is 20480 wide.
        intermediate_size=2560,
        num_experts=64,
        top_k=8,
        gate_fn="softmax",
        norm_topk_prob=False,   # config.json says false
        shared_experts=1,
        num_layers=28,
        first_moe_layer=0,
        hf_repo="Qwen/Qwen2-57B-A14B-Instruct",
        verified=True,
    ),
    "deepseek-v3": MoEConfig(
        name="deepseek-v3",
        hidden_size=7168,
        intermediate_size=2048,   # moe_intermediate_size
        num_experts=256,          # n_routed_experts
        top_k=8,
        gate_fn="sigmoid",        # scoring_func = sigmoid, not softmax
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        shared_experts=1,
        num_layers=61,
        first_moe_layer=3,        # first_k_dense_replace: 58 of 61 layers are MoE
        hf_repo="deepseek-ai/DeepSeek-V3",
        verified=True,
    ),
    "deepseek-v2-lite": MoEConfig(
        name="deepseek-v2-lite",
        hidden_size=2048,
        intermediate_size=1408,   # moe_intermediate_size
        num_experts=64,
        top_k=6,
        gate_fn="softmax",
        norm_topk_prob=False,
        shared_experts=2,
        num_layers=27,
        first_moe_layer=1,
        hf_repo="deepseek-ai/DeepSeek-V2-Lite",
        verified=True,
    ),
    # Tiny synthetic geometry for CPU unit tests and laptop-side debugging.
    "toy": MoEConfig(
        name="toy",
        hidden_size=64,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        num_layers=2,
        verified=True,
    ),
}


def tensor_parallel_shard(base: MoEConfig, tensor_parallel: int,
                          name: str | None = None) -> MoEConfig:
    """One tensor-parallel rank's slice of `base`, as its own named geometry.

    Pure TP splits each expert's intermediate dimension across ranks and leaves
    the expert COUNT and the hidden size whole, so `E` and `H` are copied and
    only `F` divides. That is not an assumption: vLLM keys its tuned kernel
    configs on `E, _, N = w2_shape`, so a shard shows up in the lookup as the
    same `E` at a smaller `N`, and the configs it ships for DeepSeek-V3 are
    `E=256,N=256` and `E=256,N=512` -- the same 256 experts at 2048/8 and
    2048/4.

    THE LIMITATION THIS EXISTS TO CLOSE. Every published cell in this study is
    TP=1, and `E=256,N=2048` is the UNSHARDED DeepSeek-V3 shape, whose 58 MoE
    layers hold 1.3 TB of bf16 weights and which no vLLM config ships for on any
    device.
    So the sweep's headline model has been running the hardcoded fallback tile
    ladder rather than a tuned config, and the reason is that nobody serves the
    shape being benchmarked. See the C5 section of docs/FINDINGS.md.

    Derived rather than hand-entered so a shard cannot drift from its base: a
    second literal `intermediate_size=256` next to `hidden_size=7168` is a
    number with nothing checking it, and the first thing anyone would do with it
    is copy it to the next model and forget the divisor.

    Refuses a width that does not divide, rather than flooring it. `F // TP`
    silently produces a geometry no rank ever holds, and every downstream byte,
    crossing and footprint would then describe a layer that does not exist.
    """
    if tensor_parallel < 1:
        raise ValueError(f"tensor_parallel must be at least 1, got {tensor_parallel}")
    if base.intermediate_size % tensor_parallel:
        raise ValueError(
            f"{base.name}: intermediate_size {base.intermediate_size} is not "
            f"divisible by tensor_parallel {tensor_parallel}; that shard width "
            "is not a shape any rank holds")
    return replace(
        base,
        name=name or f"{base.name}-tp{tensor_parallel}",
        intermediate_size=base.intermediate_size // tensor_parallel,
        tensor_parallel=base.tensor_parallel * tensor_parallel,
    )


#: Deployment widths worth benchmarking, as `(base model, TP)`. Each one is a
#: shape vLLM ships a tuned config for on some device at v0.27.1, which is the
#: whole point of the list: TP=1 DeepSeek-V3 is a shape nobody serves and nobody
#: tunes, and its neighbours in this table are the ones that are.
#:
#: deepseek-v3 at 4 and 8 are the widths named in FINDINGS: N=512 and N=256, both
#: shipped for H200 under `dtype=fp8_w8a8,block_shape=[128,128]`.
#: mixtral and qwen2 at 8 are here because they are the only shards among these
#: models with a tuned H200 config in BOTH bf16 and plain `fp8_w8a8`, so they
#: are the only pair that can be measured tuned-against-tuned across a shard
#: without first teaching the harness block-wise scales.
#: deepseek-v2-lite has no shard on the list because vLLM ships nothing for
#: `E=64` at 1408 or any divisor of it on any device.
_TENSOR_PARALLEL_SHARDS: tuple[tuple[str, int], ...] = (
    ("deepseek-v3", 4),
    ("deepseek-v3", 8),
    ("mixtral-8x7b", 8),
    ("qwen2-57b-a14b", 8),
)

MODEL_CONFIGS.update({
    shard.name: shard for shard in (
        tensor_parallel_shard(MODEL_CONFIGS[base], tp)
        for base, tp in _TENSOR_PARALLEL_SHARDS)
})


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
    """Cartesian product of the benchmark axes, in a deterministic order.

    Order is chosen, not incidental: (model, dtype, seed) determine the expert
    weights, so those axes vary slowest and consecutive cells can reuse one
    weight set. With seed innermost, consecutive cells alternated seeds and any
    weight cache would thrash. Row order in the CSV follows from this; resume
    is unaffected because manifest keys are content-derived.
    """
    for model in models:
        for dtype in dtypes:
            for seed in (seeds or [0]):
                for tokens in token_counts:
                    for routing in routings:
                        yield BenchSpec(model, tokens, dtype, routing, seed)
