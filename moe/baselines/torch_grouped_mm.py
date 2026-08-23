"""torch.nn.functional.grouped_mm as a grouped-GEMM baseline.

This ships inside torch 2.13.0 and needs no install, no framework, and no venv
of its own. On CUDA it dispatches to CUTLASS `bf16bf16_grouped_gemm_impl_sm90_sm100`,
so on an H200 it is a Hopper-native, WGMMA-capable grouped GEMM: the right
incumbent to measure a hand-written kernel against, and a far more honest
baseline than the naive per-expert loop.

Two properties make it fit this harness exactly:

  - It takes DEVICE-RESIDENT int32 offsets. Group count comes from
    `offs.size(0)`, a shape rather than a value, so nothing reads a device
    scalar on the host and the call has no synchronisation in it.
  - In the standard MoE layout (2-D `[TotalM, K]` activations, 3-D `[G, K, N]`
    weights) it imposes no tile-alignment constraint on the ragged M dimension,
    which is precisely the constraint that disqualifies several alternatives.

Offsets convention: `offs` is the EXCLUSIVE END of each group, shape `[E]`.
The harness carries CSR-style `expert_offsets` of shape `[E+1]` starting at 0,
so the mapping is `expert_offsets[1:]`.
"""
from __future__ import annotations

import torch

from ..spec import BenchSpec
from ..stages import StageSpan, register
from ..state import MoEState

# Resolve the entry point once, at import.
#
# `torch.nn.functional.grouped_mm` is the public API; `torch._grouped_mm` is the
# same operator under its private name and exists on older builds. Verified to
# accept the same call shape and return bit-identical results on torch 2.13.0,
# so the fallback is a rename rather than a different code path. This matters
# because RunPod's newest official PyTorch image is 2.8, and the base venv
# inherits the image's torch.
#
# If neither exists, raise at IMPORT time so `load_all` skips this module with a
# warning and the sweep runs without the baseline. Failing at call time instead
# would surface as a crashed cell partway through a paid session.
_GROUPED_MM = getattr(torch.nn.functional, "grouped_mm", None) or \
    getattr(torch, "_grouped_mm", None)
if _GROUPED_MM is None:  # pragma: no cover
    raise ImportError(
        f"torch {torch.__version__} has neither torch.nn.functional.grouped_mm "
        "nor torch._grouped_mm; this baseline needs a newer torch. The harness "
        "runs without it.")


def _offs(st: MoEState) -> torch.Tensor:
    """CSR `[E+1]` offsets -> grouped_mm's `[E]` exclusive-end offsets."""
    return st.expert_offsets[1:].to(torch.int32)


class _GroupedMM(StageSpan):
    """Shared plumbing. The weight transpose is a view, not a copy.

    `w1` is stored `[E, 2F, H]` and `w2` as `[E, H, F]`, matching the vLLM and
    SGLang convention so every implementation sees the same bytes. grouped_mm
    wants `[G, K, N]`, so both need a `transpose(1, 2)`. That is a stride
    change with no data movement and no allocation, verified against this exact
    torch build, so it costs nothing in the timed region.
    """

    env = "base"
    requires_cuda = False       # runs on CPU too, so the CPU suite can check it
    # CUDA dispatch is bf16-only. Left as a single dtype rather than widened,
    # because a cell that only works on the laptop is not a baseline.
    dtypes = ("bf16",)
    # Source inspection said there is no host sync in the CUDA path, and the
    # standard sweep of 2026-08-22 (run 92572c5216fb) confirmed it: 280 rows
    # reported capture_status=captured, zero failed, and every one of them
    # passed replay verification against the golden fp32 oracle. The flag was a
    # claim; the column is now the fact, so the claim can follow it.
    #
    # Worth knowing before betting on graphs: on those same rows the replay
    # beat eager by a median of 0.5 us (p10 -3.8, p90 +3.0). One grouped-GEMM
    # launch against a 130 us - 7 ms kernel is not where the time goes.
    cuda_graph_safe = True

    def supports(self, spec: BenchSpec) -> bool:
        # CUTLASS grouped GEMM caps the group count.
        return super().supports(spec) and spec.model.num_experts < 1024


@register
class TorchGroupedMMUp(_GroupedMM):
    """Up-projection: [Ntot, H] x [E, H, 2F] -> [Ntot, 2F]."""

    name = "torch_grouped_mm_up"
    covers = ("up_gemm",)

    def __call__(self, st: MoEState) -> None:
        x_perm, _ = st.require("x_perm", "expert_offsets")
        st.h_up = _GROUPED_MM(x_perm, st.weights.w1.transpose(1, 2),
                              offs=_offs(st))


@register
class TorchGroupedMMDown(_GroupedMM):
    """Down-projection: [Ntot, F] x [E, F, H] -> [Ntot, H]."""

    name = "torch_grouped_mm_down"
    covers = ("down_gemm",)

    def __call__(self, st: MoEState) -> None:
        h_act, _ = st.require("h_act", "expert_offsets")
        st.y_perm = _GROUPED_MM(h_act, st.weights.w2.transpose(1, 2),
                                offs=_offs(st))
