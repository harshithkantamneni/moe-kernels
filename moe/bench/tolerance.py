"""Error budget for comparing an implementation against golden fp32.

The metric is deliberately SCALE FREE:

    rel = max|got - ref| / max|ref|

An absolute tolerance cannot work here. Layer outputs range over orders of
magnitude with geometry and initialisation, so any fixed atol is either so loose
that an all-zeros output passes or so tight that a correct kernel fails. An
earlier version of this file used an atol floor derived from unit roundoff and
was vacuous at bf16: it admitted zeroed outputs, sign flips and 3x scale errors
on every model in the sweep.

The budget itself:

  input/output quantisation   the operands and the result are stored in the
                              working dtype, so about `eps` relative error
                              enters at each of the two GEMMs and the store
  accumulation                tensor cores accumulate in fp32, so the dot
                              product over K contributes eps_fp32 * sqrt(K),
                              which is negligible next to bf16's eps but
                              dominates at fp32
  SwiGLU                      multiplies two projections, roughly doubling
                              relative error, and the sigmoid's slope can
                              amplify locally

`scripts/calibrate_tolerance.py` measures the reference path's own error on the
target hardware so these constants can be replaced by observation. Until that
has run, `calibrated` stays False and the driver records it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..spec import BenchSpec

# Unit roundoff per format: 2^-(mantissa_bits+1).
# bf16 has 8 total mantissa bits, fp16 has 11, fp32 has 24.
# Unit roundoff, 2^-(mantissa bits + 1). fp8 is here because C2's prediction,
# that halving bytes-per-element halves the ridge crossing, is tested by an fp8
# sweep, and `tolerance()` raises for any dtype it does not know: the sweep would
# have died on its first cell rather than run with a wrong budget.
#   e4m3  3 explicit mantissa bits -> 2^-4
#   e5m2  2                        -> 2^-3
_EPS = {"fp32": 2 ** -24, "fp16": 2 ** -11, "bf16": 2 ** -8,
        "fp8_e4m3": 2 ** -4, "fp8_e5m2": 2 ** -3}

# Quantisation enters at both GEMM operands and the store, and SwiGLU roughly
# doubles it. Four is the resulting order-of-magnitude coefficient; it is a
# bound on the modelled terms, not a fudge factor, and calibration replaces it.
_QUANT_TERMS = 4.0
_ACT_AMPLIFICATION = 2.0
_EPS_FP32 = 2 ** -24


@dataclass(frozen=True)
class Tolerance:
    """Maximum permitted `max|got-ref| / max|ref|`."""

    rel_max: float
    calibrated: bool = False
    basis: str = "analytic"

    def passes(self, rel: float) -> bool:
        return rel <= self.rel_max


def relative_error(got, ref) -> float:
    """Scale-free error: worst elementwise deviation, normalised by the largest
    magnitude in the reference. 1.0 means "as wrong as the answer is big"."""
    ref = ref.float()
    got = got.float()
    denom = float(ref.abs().max())
    if denom == 0.0:
        # A genuinely all-zero reference: any nonzero output is infinitely wrong.
        return 0.0 if float(got.abs().max()) == 0.0 else float("inf")
    return float((got - ref).abs().max()) / denom


def tolerance(spec: BenchSpec, calibration: dict | None = None) -> Tolerance:
    if calibration and spec.dtype in calibration:
        return Tolerance(float(calibration[spec.dtype]["rel_max"]),
                         calibrated=True, basis="measured")

    eps = _EPS.get(spec.dtype)
    if eps is None:
        raise ValueError(f"no tolerance model for dtype {spec.dtype!r}")

    cfg = spec.model
    quant = _QUANT_TERMS * eps * _ACT_AMPLIFICATION
    accum = _EPS_FP32 * (math.sqrt(cfg.hidden_size) + math.sqrt(cfg.intermediate_size))
    return Tolerance(float(quant + accum))
