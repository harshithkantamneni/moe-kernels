"""Numerical tolerances for comparing an implementation against golden fp32.

A tolerance pulled out of the air either hides real bugs or fails good kernels.
The model below is explicit about where the error comes from, and
`scripts/calibrate_tolerance.py` measures the reference path's own error on the
target hardware so these constants can be replaced by observation rather than
left as guesses. Until that has been run on H200, `calibrated` stays False and
the driver records it in the row so no result claims more rigour than it has.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..spec import BenchSpec

# Unit-roundoff-scale constants per format. bf16 has 8 mantissa bits, fp16 has
# 11, so bf16's per-operation error is roughly 8x fp16's.
_EPS = {"fp32": 6e-8, "fp16": 5e-4, "bf16": 4e-3}

# SwiGLU multiplies two projections together, so relative error roughly doubles
# and a sigmoid's slope can locally amplify it further.
_ACT_AMPLIFICATION = 2.5


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float
    calibrated: bool = False
    basis: str = "analytic"


def tolerance(spec: BenchSpec, calibration: dict | None = None) -> Tolerance:
    """Error budget for a full MoE layer at this geometry and dtype.

    Sources of error, in order of size:
      - up-projection accumulation over K = hidden_size
      - SwiGLU amplification
      - down-projection accumulation over K = intermediate_size
      - the top_k weighted combine, which sums k terms
    """
    if calibration and spec.dtype in calibration:
        c = calibration[spec.dtype]
        return Tolerance(c["atol"], c["rtol"], calibrated=True, basis="measured")

    eps = _EPS.get(spec.dtype)
    if eps is None:
        raise ValueError(f"no tolerance model for dtype {spec.dtype!r}")

    cfg = spec.model
    # Random-walk growth of rounding error through two accumulations.
    gemm_growth = math.sqrt(cfg.hidden_size) + math.sqrt(cfg.intermediate_size)
    rtol = eps * gemm_growth * _ACT_AMPLIFICATION * math.sqrt(cfg.top_k)

    # make_inputs scales activations to about 0.02, and the layer output stays
    # within roughly an order of magnitude of that, so an absolute floor keeps
    # near-zero outputs from failing on relative error alone.
    atol = max(rtol * 0.05, eps * 10.0)
    return Tolerance(float(atol), float(rtol))
