"""The data that flows through the MoE layer.

`MoEState` is mutated in place by stage spans. In-place mutation is only safe
because every span declares `reads`/`writes` over these field names, and
`pipeline.py` checks those declarations before anything runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from .spec import BenchSpec

# Every field a span may name in `reads` / `writes`. Anything outside this set
# is a typo, and pipeline validation rejects it rather than failing at runtime
# on a paid GPU.
STATE_FIELDS: frozenset[str] = frozenset(
    {
        "x",              # [T, H]        layer input
        "router_logits",  # [T, E]        pre-softmax gate scores
        "topk_ids",       # [T, k] int32  chosen experts per token
        "topk_weights",   # [T, k] fp32   combine weights, post-softmax
        "expert_offsets", # [E+1] int32   CSR-style group boundaries into permuted rows
        "perm_index",     # [Ntot] int32  permuted row -> flat (token*k + slot)
        "x_perm",         # [Ntot, H]     tokens gathered into expert-contiguous order
        "h_up",           # [Ntot, 2F]    fused gate+up projection output
        "h_act",          # [Ntot, F]     post-SwiGLU
        "y_perm",         # [Ntot, H]     down projection output, still permuted
        "y",              # [T, H]        layer output, original token order
    }
)


@dataclass
class MoEWeights:
    """Expert weights plus the router gate. Random, never loaded from a checkpoint."""

    w1: Any  # [E, 2F, H] fused gate+up
    w2: Any  # [E, H, F]  down
    wg: Any  # [E, H]     router gate, kept fp32

    def validate(self, spec: BenchSpec) -> None:
        cfg = spec.model
        checks = [
            ("w1", tuple(self.w1.shape), cfg.w1_shape),
            ("w2", tuple(self.w2.shape), cfg.w2_shape),
            ("wg", tuple(self.wg.shape), (cfg.num_experts, cfg.hidden_size)),
        ]
        for name, got, want in checks:
            if got != want:
                raise ValueError(f"weights.{name}: expected {want}, got {got}")


@dataclass
class MoEState:
    """Carrier for one forward pass. Fields are None until a span writes them."""

    spec: BenchSpec
    weights: MoEWeights

    x: Any = None
    router_logits: Any = None
    topk_ids: Any = None
    topk_weights: Any = None
    expert_offsets: Any = None
    perm_index: Any = None
    x_perm: Any = None
    h_up: Any = None
    h_act: Any = None
    y_perm: Any = None
    y: Any = None

    # Filled by capture/replay so an implementation can be handed a fixed
    # routing decision instead of computing one. Keeps grouped-GEMM timing
    # independent of router cost, and makes trace replay exact.
    forced_topk_ids: Any = None

    _written: set[str] = field(default_factory=set, repr=False)

    # -- contract bookkeeping ------------------------------------------------

    def mark_written(self, names) -> None:
        self._written.update(names)

    @property
    def written(self) -> frozenset[str]:
        return frozenset(self._written)

    def require(self, *names: str) -> tuple:
        """Fetch fields, raising a clear error if a span forgot to produce one."""
        out = []
        for name in names:
            if name not in STATE_FIELDS:
                raise KeyError(f"{name!r} is not a MoEState field")
            value = getattr(self, name)
            if value is None:
                raise ValueError(
                    f"state field {name!r} is None; the span that writes it did not run"
                )
            out.append(value)
        return tuple(out)

    # -- shape checking ------------------------------------------------------

    def expected_shapes(self) -> dict[str, tuple[int, ...]]:
        cfg = self.spec.model
        T, H, F, E, k = (
            self.spec.num_tokens,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_experts,
            cfg.top_k,
        )
        ntot = self.spec.rows
        return {
            "x": (T, H),
            "router_logits": (T, E),
            "topk_ids": (T, k),
            "topk_weights": (T, k),
            "expert_offsets": (E + 1,),
            "perm_index": (ntot,),
            "x_perm": (ntot, H),
            "h_up": (ntot, 2 * F),
            "h_act": (ntot, F),
            "y_perm": (ntot, H),
            "y": (T, H),
        }

    def validate(self, only: frozenset[str] | None = None) -> None:
        """Shape-check every populated field. Cheap, and catches most kernel bugs
        before they surface as a confusing numerical mismatch."""
        expected = self.expected_shapes()
        for f in fields(self):
            if f.name not in STATE_FIELDS:
                continue
            if only is not None and f.name not in only:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            got = tuple(value.shape)
            want = expected[f.name]
            if got != want:
                raise ValueError(f"state.{f.name}: expected shape {want}, got {got}")


def group_sizes_from_offsets(expert_offsets) -> list[int]:
    """[E+1] CSR offsets -> per-expert row counts. Pure python, used by tests and
    by the roofline model, so it must not assume a torch tensor."""
    off = [int(v) for v in expert_offsets]
    if off[0] != 0:
        raise ValueError(f"expert_offsets must start at 0, got {off[0]}")
    sizes = [off[i + 1] - off[i] for i in range(len(off) - 1)]
    if any(s < 0 for s in sizes):
        raise ValueError("expert_offsets must be non-decreasing")
    return sizes
