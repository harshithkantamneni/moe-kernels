"""What one benchmark cell costs in device memory, and what it costs on disk.

These are different questions and conflating them makes DeepSeek-V3 look
impossible when it is routine. A sweep NEVER downloads a model: `make_inputs`
generates random weights for a single MoE layer at the model's geometry. So a
256-expert model costs device memory for one layer's weights and nothing at all
on disk. Only `scripts/capture_traces.py` pulls real checkpoints, and it is the
only thing whose disk cost scales with the whole model.

Everything here is arithmetic over MoEConfig. No torch, no CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..spec import BenchSpec, MoEConfig, dtype_bytes


@dataclass(frozen=True)
class Footprint:
    """Bytes a single cell needs resident on the device."""

    weights: int          # one MoE layer's expert weights, at the working dtype
    activations: int      # x, x_perm, h_up, h_act, y_perm, y
    oracle: int           # fp32 copies golden_forward materialises
    oracle_expert: int    # grouped_gemm_loop's per-expert fp32 weight slab

    @property
    def peak(self) -> int:
        return self.weights + self.activations + self.oracle + self.oracle_expert

    def fits_in(self, device_bytes: int, headroom: float = 0.85) -> bool:
        """Leave headroom: the allocator fragments, and a CUDA graph capture
        holds a private pool alongside the eager path's buffers."""
        return self.peak <= device_bytes * headroom


def cell_footprint(spec: BenchSpec) -> Footprint:
    cfg = spec.model
    b = dtype_bytes(spec.dtype)
    H, F, E = cfg.hidden_size, cfg.intermediate_size, cfg.num_experts
    T, ntot = spec.num_tokens, spec.rows

    weights = (E * 2 * F * H + E * H * F) * b
    activations = (ntot * H          # x_perm
                   + ntot * 2 * F    # h_up
                   + ntot * F        # h_act
                   + ntot * H        # y_perm
                   + 2 * T * H       # x, y
                   ) * b
    # golden_forward runs the whole layer in fp32.
    oracle = activations * (4 // b) if b < 4 else activations
    # grouped_gemm_loop converts w[e] per expert, so the transient is one
    # expert's slab rather than the whole tensor. That distinction is worth
    # 45 GB at DeepSeek-V3 geometry and is why the whole-tensor pre-cast went.
    oracle_expert = 2 * F * H * 4
    return Footprint(weights, activations, oracle, oracle_expert)


def download_bytes(cfg: MoEConfig, dtype: str = "bf16") -> int:
    """Disk a real checkpoint costs, for trace capture only.

    Scales with the number of MoE layers, which is what makes DeepSeek-V3
    uncapturable on any single GPU while its geometry stays perfectly cheap to
    benchmark.

    `full_intermediate_size`, not `intermediate_size`, and the distinction only
    appeared when tensor-parallel configs did. A checkpoint holds the whole
    expert; a rank holds a slice of it. Charging the slice would report
    `deepseek-v3-tp8`'s checkpoint at an eighth of its true size, and this
    number exists to say what will not fit.
    """
    F = cfg.full_intermediate_size
    per_layer = (cfg.num_experts * 2 * F * cfg.hidden_size
                 + cfg.num_experts * cfg.hidden_size * F)
    return per_layer * max(cfg.num_moe_layers, 1) * dtype_bytes(dtype)


def worst_cell(specs) -> tuple[BenchSpec, Footprint] | tuple[None, None]:
    """The cell that will OOM first, if any will."""
    worst = None
    for spec in specs:
        fp = cell_footprint(spec)
        if worst is None or fp.peak > worst[1].peak:
            worst = (spec, fp)
    return worst if worst else (None, None)


def worst_cell_by_model(specs) -> dict[str, tuple[BenchSpec, Footprint]]:
    """`worst_cell` per model name, so a mixed profile answers per shape.

    THE QUESTION THIS ANSWERS THAT `worst_cell` CANNOT. A profile that mixes
    geometries has one global worst cell, and knowing it fits says nothing about
    the other shapes: a sharded entry is an eighth of the weights of the model
    it is named after, so `deepseek-v3` dominates every profile it appears in
    and would hide a shard that did not fit. The rent is committed per profile,
    but the OOM happens per shape.

    Keyed by name rather than by config so the answer reads like the CSV's
    `model` column, which is what a reader will be holding.
    """
    out: dict[str, tuple[BenchSpec, Footprint]] = {}
    for spec in specs:
        fp = cell_footprint(spec)
        seen = out.get(spec.model.name)
        if seen is None or fp.peak > seen[1].peak:
            out[spec.model.name] = (spec, fp)
    return out
