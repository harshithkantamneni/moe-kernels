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

from ..quant import quantize_per_expert
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


_SCALED_GROUPED_MM = getattr(torch, "_scaled_grouped_mm", None)


@dataclass(frozen=True)
class ScaledArgs:
    """Exactly what `_scaled_grouped_mm` is called with, built separately so it
    can be checked without an sm_90 device to run the kernel on."""

    a: torch.Tensor
    b: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    offs: torch.Tensor


#: Quantised activations, keyed on the tensor's identity, shape and dtype.
#:
#: The span declares `covers = ("up_gemm",)` and the byte model costs exactly
#: that. Quantising activations is NOT up_gemm: it exists only because
#: `_scaled_grouped_mm` needs both operands in fp8 while the harness hands out
#: bf16 activations for vLLM's sake. Doing it per call put that work inside the
#: timer and made deepseek-v2-lite at T=8192 read 1.9855 ms in fp8 against
#: 1.0503 in bf16 -- fp8 1.89x SLOWER on the same GEMM -- which biased every
#: crossing early and cost the arm its credibility.
#:
#: `x_perm` is built once per cell in the prologue and reused by every timed
#: iteration, so this is warm before the first timed call.
#:
#: Keyed on the tensor's IDENTITY, and the cache holds a reference to it.
#:
#: A (data_ptr, shape, dtype) key is not enough and was not a theoretical
#: concern: torch's allocator reuses freed addresses, so a tensor from a
#: finished cell is collected, the next cell's activations land at the same
#: address with the same shape, and the cache serves the wrong quantisation. It
#: took one test run to happen. Holding the tensor keeps its address alive for
#: as long as the entry does, which makes `is` a sound test.
#:
#: Single slot: one cell is resident at a time, and the entry is what pins the
#: activations, so a growing cache would pin every batch a sweep ever built.
_ACT_QUANT_CACHE: list = []   # [(tensor, quantised, scale)] or empty


#: Per-expert scales broadcast to per-output-channel, keyed on the scale tensor
#: and the width. Built inside the timed region, so rebuilding [E, N] float32 on
#: every call would put a megabyte of memset into the measurement. Bounded by
#: the weight cache above it, which holds one model at a time.
_CHANNEL_SCALE_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _per_channel_scale(w_scale: torch.Tensor, n: int) -> torch.Tensor:
    """`[E]` -> `[E, N]`, materialised.

    torch's check_scale takes its rule from mat_b's dimensionality: a 3D
    `[E, K, N]` weight wants a 2D `[E, N]` scale, one per output channel. The
    harness quantises per expert, so every channel of an expert carries the same
    value; the arithmetic is identical and only the layout differs.

    Materialised rather than expanded because torch also requires
    `scale.stride(-1) == 1`, and an expand leaves stride 0 there: it would pass
    a shape check and fail at the call.
    """
    key = (w_scale.data_ptr(), n)
    hit = _CHANNEL_SCALE_CACHE.get(key)
    if hit is not None and hit.shape == (w_scale.shape[0], n):
        return hit
    built = w_scale.reshape(-1, 1).expand(-1, n).contiguous()
    _CHANNEL_SCALE_CACHE.clear()   # one model resident at a time, as above
    _CHANNEL_SCALE_CACHE[key] = built
    return built


def scaled_grouped_args(a, w_q, w_scale, offs, dtype: str) -> ScaledArgs:
    """Build the fp8 grouped-GEMM call.

    `_scaled_grouped_mm` needs BOTH operands in fp8, while the harness hands out
    bf16 activations because vLLM asserts on anything else. So the activations
    are quantised here, per row, inside the timed region. That is the fair
    comparison rather than a handicap: vLLM's kernel quantises activations
    internally too, so both implementations are charged for the same work.

    `quantize_per_expert` quantises along dim 0 with one scale per slice, which
    for `[Ntot, K]` activations is one scale per token. Per-token rather than
    per-tensor because a single outlier token would otherwise crush the
    resolution of every other row.

    The WEIGHT scale is the one `make_inputs` produced, never recomputed. The
    oracle dequantises with that scale, so a locally derived one would measure a
    different layer from the one being judged.
    """
    if _ACT_QUANT_CACHE and _ACT_QUANT_CACHE[0][0] is a \
            and _ACT_QUANT_CACHE[0][3] == dtype:
        a_q, scale_a = _ACT_QUANT_CACHE[0][1], _ACT_QUANT_CACHE[0][2]
    else:
        a_q, scale_a = quantize_per_expert(a, dtype)
        _ACT_QUANT_CACHE.clear()
        _ACT_QUANT_CACHE.append((a, a_q, scale_a, dtype))
    b = w_q.transpose(1, 2)
    # scale_a stays 1D: mat_a is 2D, and torch wants one element per row there.
    # scale_b becomes 2D: mat_b is 3D, and torch wants one per output channel.
    # Making both the same shape trades one RuntimeError for the other.
    return ScaledArgs(a=a_q, b=b, scale_a=scale_a,
                      scale_b=_per_channel_scale(w_scale, b.shape[2]), offs=offs)


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


class _ScaledGroupedMM(_GroupedMM):
    """fp8 grouped GEMM: the same CUTLASS shape, half the weight bytes.

    A SECOND independent test of C2. Through vLLM alone a 2x crossing shift
    cannot be told apart from a property of vLLM's kernel; through two unrelated
    kernels it is a statement about traffic, which is what C2 claims. This span
    is the cleanest of the three for it, covering one stage rather than five, so
    the byte model applies with nothing folded in.
    """

    #: fp8 only. The bf16 path is torch_grouped_mm_*, and keeping one span per
    #: dtype keeps the `impl` column meaning exactly one thing.
    dtypes = ("fp8_e4m3",)
    #: Unverified until a row says otherwise, same as every other span here.
    cuda_graph_safe = False

    def supports(self, spec: BenchSpec) -> bool:
        return _SCALED_GROUPED_MM is not None and super().supports(spec)

    def why_unsupported(self, spec: BenchSpec) -> str:
        if _SCALED_GROUPED_MM is None:
            return f"torch {torch.__version__} has no _scaled_grouped_mm"
        return super().why_unsupported(spec)


@register
class TorchScaledGroupedMMUp(_ScaledGroupedMM):
    """fp8 up-projection: [Ntot, H] x [E, H, 2F] -> [Ntot, 2F]."""

    name = "torch_scaled_grouped_mm_up"
    covers = ("up_gemm",)

    def __call__(self, st: MoEState) -> None:
        x_perm, _ = st.require("x_perm", "expert_offsets")
        a = scaled_grouped_args(x_perm, st.weights.w1, st.weights.w1_scale,
                                _offs(st), st.spec.dtype)
        st.h_up = _SCALED_GROUPED_MM(
            a.a, a.b, a.scale_a, a.scale_b, offs=a.offs,
            out_dtype=x_perm.dtype)


@register
class TorchScaledGroupedMMDown(_ScaledGroupedMM):
    """fp8 down-projection: [Ntot, F] x [E, F, H] -> [Ntot, H]."""

    name = "torch_scaled_grouped_mm_down"
    covers = ("down_gemm",)

    def __call__(self, st: MoEState) -> None:
        h_act, _ = st.require("h_act", "expert_offsets")
        a = scaled_grouped_args(h_act, st.weights.w2, st.weights.w2_scale,
                                _offs(st), st.spec.dtype)
        st.y_perm = _SCALED_GROUPED_MM(
            a.a, a.b, a.scale_a, a.scale_b, offs=a.offs,
            out_dtype=h_act.dtype)
