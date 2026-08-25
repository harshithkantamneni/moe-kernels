"""torch.nn.functional.grouped_mm as a grouped-GEMM baseline.

This ships inside torch 2.13.0 and needs no install, no framework, and no venv
of its own. On CUDA it dispatches to CUTLASS `bf16bf16_grouped_gemm_impl_sm90_sm100`,
so on an H200 it is a Hopper-native, WGMMA-capable grouped GEMM, which is the
incumbent a hand-written kernel has to beat. A per-expert loop would be an
easier baseline and would not be measuring the same thing.

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

from dataclasses import dataclass

import torch

from ..spec import BenchSpec
from ..stages import StageSpan, register
from ..state import MoEState

#: Compute capabilities the CUTLASS grouped GEMM covers, from the impl name
#: `bf16bf16_grouped_gemm_impl_sm90_sm100`.
_GROUPED_GEMM_ARCHS = ((9, 0), (10, 0))


@dataclass(frozen=True)
class GroupedMMSupport:
    supported: bool
    reason: str = ""


def grouped_mm_support(capability: tuple[int, int] | None = None
                       ) -> GroupedMMSupport:
    """Does this architecture have the CUTLASS grouped GEMM behind grouped_mm?

    The symbol exists on every build, so resolving it says nothing about the
    device. Outside sm_90/sm_100 the call either falls back to something that
    is not the incumbent the published numbers measured, or raises partway
    through a paid sweep. Both are worth knowing before the sweep starts.
    """
    if capability is None:
        capability = torch.cuda.get_device_capability()
    major, minor = capability
    if (major, 0) in _GROUPED_GEMM_ARCHS:
        return GroupedMMSupport(True)
    have = ", ".join(f"sm_{a}{b}" for a, b in _GROUPED_GEMM_ARCHS)
    return GroupedMMSupport(
        False,
        f"torch's grouped_mm dispatches to a CUTLASS grouped GEMM built for "
        f"{have}; this device is sm_{major}{minor}. Any timing here measures a "
        "different implementation than the published rows, so the baseline is "
        "not comparable on this part.")

def _device_support() -> GroupedMMSupport | None:
    """This machine's verdict, or None when there is no device to ask.

    None is not "unsupported". `--dry-run` builds and validates the whole matrix
    on a laptop, and a plan that silently drops the only registered baseline
    because no GPU was attached would report an empty sweep as if that were the
    answer.
    """
    if not torch.cuda.is_available():
        return None
    try:
        return grouped_mm_support()
    except (RuntimeError, AssertionError):
        return None


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
        if not (super().supports(spec) and spec.model.num_experts < 1024):
            return False
        # And it only exists on some architectures. Unwired, the guard was
        # merely available: a sweep on an A100 would time whatever torch falls
        # back to and write it under the same `impl` name as the published H200
        # rows, which is the silent substitution this harness exists to refuse.
        verdict = _device_support()
        return True if verdict is None else verdict.supported

    def why_unsupported(self, spec: BenchSpec) -> str:
        verdict = _device_support()
        if verdict is not None and not verdict.supported:
            return verdict.reason
        if spec.model.num_experts >= 1024:
            return f"grouped_mm caps the group count; {spec.model.num_experts} experts"
        return super().why_unsupported(spec)


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
