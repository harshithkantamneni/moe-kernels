"""FLOPs and minimum global memory traffic, computed per *tiling*.

FLOPs are tiling-invariant. Bytes are not: a span that fuses up_gemm with act
never materialises h_up, so neither the store nor the reload appears in the
model. Arithmetic intensity is therefore a property of the pipeline, which is
what makes "should this fusion help here" a roofline prediction you can check
against measurement instead of a claim.

Everything here is pure arithmetic over the contract declarations in stages.py.
No torch, no CUDA, testable on a laptop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..spec import BenchSpec, dtype_bytes
from ..stages import STAGE_CONTRACTS, StageSpan

# Fields whose element size does not follow the working dtype.
_FIXED_WIDTH: dict[str, int] = {
    "router_logits": 4,   # gate maths is kept in fp32 everywhere
    "topk_weights": 4,
    "topk_ids": 4,        # int32
    "expert_offsets": 4,  # int32
    "perm_index": 4,      # int32
}


def field_elements(spec: BenchSpec) -> dict[str, int]:
    cfg = spec.model
    T, H, F, E, k = (spec.num_tokens, cfg.hidden_size, cfg.intermediate_size,
                     cfg.num_experts, cfg.top_k)
    ntot = spec.rows
    return {
        "x": T * H,
        "router_logits": T * E,
        "topk_ids": T * k,
        "topk_weights": T * k,
        "expert_offsets": E + 1,
        "perm_index": ntot,
        "x_perm": ntot * H,
        "h_up": ntot * 2 * F,
        "h_act": ntot * F,
        "y_perm": ntot * H,
        "y": T * H,
    }


def field_bytes(spec: BenchSpec) -> dict[str, int]:
    elems = field_elements(spec)
    act_b = dtype_bytes(spec.dtype)
    return {name: n * _FIXED_WIDTH.get(name, act_b) for name, n in elems.items()}


def weight_bytes_for_stage(spec: BenchSpec, stage: str, active_experts: int) -> int:
    """Weight traffic attributable to one canonical stage.

    Only experts that actually receive rows are read. This term is what makes
    the many-expert small-batch regime memory bound: at 256 experts and a
    handful of tokens it dwarfs every activation term in the model.
    """
    cfg = spec.model
    b = dtype_bytes(spec.dtype)
    H, F = cfg.hidden_size, cfg.intermediate_size
    if stage == "up_gemm":
        return active_experts * 2 * F * H * b
    if stage == "down_gemm":
        return active_experts * H * F * b
    if stage == "router":
        return cfg.num_experts * H * 4  # gate kept fp32
    return 0


def stage_flops(spec: BenchSpec) -> dict[str, float]:
    cfg = spec.model
    T, H, F, E = (spec.num_tokens, cfg.hidden_size, cfg.intermediate_size,
                  cfg.num_experts)
    ntot = spec.rows
    return {
        "router": 2.0 * T * E * H,
        "permute": 0.0,
        # up: [ntot, H] x [H, 2F]
        "up_gemm": 2.0 * ntot * (2 * F) * H,
        # SwiGLU: a sigmoid, a multiply, a multiply per output element.
        # Negligible next to the GEMMs but counted so the totals are honest.
        "act": 3.0 * ntot * F,
        # down: [ntot, F] x [F, H]
        "down_gemm": 2.0 * ntot * H * F,
        "unpermute": 2.0 * ntot * H,  # scale and accumulate
    }


@dataclass(frozen=True)
class SpanCost:
    name: str
    covers: tuple[str, ...]
    flops: float
    read_bytes: int
    write_bytes: int
    weight_bytes: int

    @property
    def bytes_total(self) -> int:
        return self.read_bytes + self.write_bytes + self.weight_bytes


@dataclass(frozen=True)
class PipelineCost:
    flops: float
    bytes_total: int
    spans: tuple[SpanCost, ...] = field(default_factory=tuple)

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / max(self.bytes_total, 1)

    def tflops(self, ms: float) -> float:
        return self.flops / max(ms * 1e-3, 1e-12) / 1e12

    def gbps(self, ms: float) -> float:
        return self.bytes_total / max(ms * 1e-3, 1e-12) / 1e9


def span_cost(span: StageSpan, spec: BenchSpec, active_experts: int,
              materialised: frozenset[str] | None = None) -> SpanCost:
    """Minimum compulsory traffic for one span.

    Reads come from the span's derived contract, which already excludes anything
    the span produces internally. Writes come from the tiling's materialised
    set, which excludes intermediates the fusion swallowed. That is the entire
    fusion accounting, and it is driven by the same liveness the pipeline
    validates, so arithmetic intensity really is a property of the tiling.
    """
    fb = field_bytes(spec)
    flops_by_stage = stage_flops(spec)

    read_bytes = sum(fb[f] for f in span.reads)
    write_bytes = sum(fb[f] for f in (span.writes if materialised is None
                                      else materialised))
    weights = sum(weight_bytes_for_stage(spec, s, active_experts) for s in span.covers)
    flops = sum(flops_by_stage[s] for s in span.covers)
    return SpanCost(span.name, span.covers, flops, read_bytes, write_bytes, weights)


def pipeline_cost(spans, spec: BenchSpec, active_experts: int,
                  materialised=None) -> PipelineCost:
    if materialised is None:
        materialised = [None] * len(spans)
    costs = tuple(span_cost(s, spec, active_experts, m)
                  for s, m in zip(spans, materialised, strict=True))
    return PipelineCost(
        flops=sum(c.flops for c in costs),
        bytes_total=sum(c.bytes_total for c in costs),
        spans=costs,
    )


@dataclass(frozen=True)
class _BareStage:
    """A canonical stage with no implementation, so pipeline_cost can be called
    without a registered span. Duck-types the three attributes it reads."""

    stage: str

    @property
    def name(self) -> str:
        return self.stage

    @property
    def covers(self) -> tuple[str, ...]:
        return (self.stage,)

    @property
    def reads(self) -> frozenset[str]:
        return STAGE_CONTRACTS[self.stage].reads

    @property
    def writes(self) -> frozenset[str]:
        return STAGE_CONTRACTS[self.stage].writes


def grouped_gemm_only_cost(spec: BenchSpec, active_experts: int) -> PipelineCost:
    """Cost of just the two grouped GEMMs, unfused.

    The comparison point when you want to talk about the kernel rather than the
    layer, and the number to quote next to vLLM's fused_moe.
    """
    return pipeline_cost([_BareStage("up_gemm"), _BareStage("down_gemm")],
                         spec, active_experts)
