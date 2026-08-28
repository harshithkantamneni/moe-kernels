"""Per-expert fp8 quantisation, in the form vLLM's fused_moe expects.

WHY fp8 IS IN THIS STUDY, and why it beats a second GPU as a test. Claim C2 says
arithmetic intensity is `2R/b`, so halving bytes-per-element must double
intensity and halve the batch at which a model crosses its ridge. For
deepseek-v3 that is ~5,100 tokens down to ~2,570: a **2x** prediction, which the
existing powers-of-two token grid separates without any custom sweep. The A100's
lower ridge predicts a 1.1x shift, and that lands both predictions inside a
single bin of the same grid, resolving nothing.

THE CONVENTION IS NOT A CHOICE. `fp8_w8a8_moe_quant_config(w1_scale, w2_scale,
...)` requires the scales, and the kernel reconstructs the weight as
`q * scale`. So that is the contract here, one scale per expert. It also happens
to be marginally more accurate than an unscaled cast on realistic fan-in
weights, because 27.6% of elements would otherwise be subnormal in e4m3, but
that is a bonus rather than the reason.

WHAT THE ORACLE MUST COMPARE AGAINST. The reference has to compute from the
DEQUANTISED weights, not the original bf16 ones. Otherwise the correctness gate
is measuring quantisation error rather than kernel error, and the two are not
the same question: quantisation error is a property of the format and identical
for every implementation, while kernel error is what the gate exists to catch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: Names this module quantises to, mapped to the torch dtype. Closed on purpose:
#: an unrecognised string must raise rather than silently fall through to a cast.
FP8_DTYPES: dict[str, str] = {
    "fp8_e4m3": "float8_e4m3fn",
    "fp8_e5m2": "float8_e5m2",
}


def torch_fp8_dtype(dtype: str):
    """Resolve a harness dtype name to a torch fp8 dtype, or raise."""
    name = FP8_DTYPES.get(dtype)
    if name is None:
        raise ValueError(
            f"{dtype!r} is not an fp8 format; known: {sorted(FP8_DTYPES)}")
    resolved = getattr(torch, name, None)
    if resolved is None:
        raise ValueError(f"this torch build has no {name}")
    return resolved


def quantize_per_expert(w: torch.Tensor, dtype: str
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise `[E, ...]` weights to fp8 with one scale per expert.

    Returns `(q, scale)` obeying `q.float() * scale == dequantised`, which is
    the direction vLLM's kernel reconstructs in. Getting that backwards would
    produce a call that runs, returns the right shape, and computes a different
    layer, which is exactly the failure `_framework_config` exists to prevent
    elsewhere.

    An all-zero expert would give a zero scale and then a division by zero, so
    its scale is clamped to 1.0: the quantised values are zero either way, and a
    finite scale keeps the tensor usable.
    """
    resolved = torch_fp8_dtype(dtype)
    if w.ndim < 2:
        raise ValueError(f"expected [E, ...] weights, got shape {tuple(w.shape)}")
    fmax = torch.finfo(resolved).max

    flat = w.reshape(w.shape[0], -1).float()
    amax = flat.abs().amax(dim=1)
    scale = (amax / fmax).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))

    shaped = scale.reshape(-1, *([1] * (w.ndim - 1)))
    q = (w.float() / shaped).to(resolved)
    return q, scale.to(torch.float32)


def dequantize_per_expert(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`q * scale`, in fp32. The reference computes from this, not from the
    original weights, so the correctness gate measures the kernel rather than
    the format."""
    return q.float() * scale.reshape(-1, *([1] * (q.ndim - 1))).float()


def round_trip_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """RMS relative error, which is what sets the fp8 correctness budget.

    Not max-absolute: that is dominated by the largest elements, which sit in
    the normal range and quantise well, and it therefore hides the subnormal
    behaviour that motivates scaling in the first place.
    """
    ref = original.float()
    denom = ref.pow(2).mean().sqrt()
    if denom == 0:
        return 0.0
    return float((reconstructed.float() - ref).pow(2).mean().sqrt() / denom)


# --- does this silicon actually have fp8 tensor cores? ----------------------

#: fp8 tensor cores arrived with Ada (sm_89) and Hopper (sm_90). Ampere has
#: none: the A100 is sm_80.
_FP8_MIN_CAPABILITY = (8, 9)


@dataclass(frozen=True)
class Fp8Support:
    supported: bool
    reason: str = ""


def fp8_hardware_support(capability: tuple[int, int] | None = None
                         ) -> Fp8Support | None:
    """Whether this device has fp8 tensor cores, or None when there is no device.

    None is not "unsupported". `--dry-run` builds the whole matrix on a laptop,
    and answering "unsupported" with nothing to ask would silently empty the
    plan. Same shape as `grouped_mm_support`, and for the same reason.

    Without this, `--profile fp8` on an A100 would run: the framework spans
    declare fp8_e4m3, vLLM would accept the cell, most likely dequantise to bf16
    and issue a bf16 GEMM, and write rows labelled fp8_e4m3 that never touched
    an fp8 unit. Merged with the H200's they would be indistinguishable. That is
    the silent substitution this harness exists to refuse.
    """
    if capability is None:
        import torch
        if not torch.cuda.is_available():
            return None
        capability = torch.cuda.get_device_capability()
    if capability >= _FP8_MIN_CAPABILITY:
        return Fp8Support(True)
    major, minor = capability
    lo_major, lo_minor = _FP8_MIN_CAPABILITY
    return Fp8Support(
        False,
        f"fp8 tensor cores require sm_{lo_major}{lo_minor} or newer; this "
        f"device is sm_{major}{minor}. A cell timed here would measure an fp8 "
        "path emulated in another format and record it under the same impl and "
        "dtype as hardware fp8 rows.")


def fp8_cell_supported(spec) -> bool:
    """Can this cell's dtype run on this machine's tensor cores?

    True for every float dtype and for fp8 wherever the silicon has it, and for
    fp8 on a machine with no device at all, so a laptop can still plan.
    """
    if spec.dtype not in FP8_DTYPES:
        return True
    verdict = fp8_hardware_support()
    return True if verdict is None else verdict.supported
