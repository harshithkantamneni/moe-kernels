"""What ridge does a kernel actually meet, as opposed to the datasheet's?

THE OPEN ITEM THIS ADDRESSES. Measured crossings sit below `2R/b`'s prediction
by 0.63x in bf16 and 0.71x in fp8, across four models. One multiplicative factor
that consistent is structure rather than scatter.

A crossing is where arithmetic intensity meets the ridge, and the ridge is
`peak_FLOPS / bandwidth`. Datasheet peaks are not what a kernel reaches. Attain
fraction `f` of peak FLOPs and `g` of peak bandwidth and the ridge actually met
is `(f/g) x nominal`. The crossing is proportional to the ridge, so:

    measured_crossing / predicted_crossing  ==  effective_ridge / nominal_ridge

The left side is already measured. This module computes the right side, which
makes the hypothesis refutable: if the two disagree, achieved-versus-peak is not
the explanation and activation traffic comes back into play.

Both terms come from the SAME model, `flops` and `compulsory_bytes`, so this is
a roofline in the byte model's own units rather than a mix of measured traffic
and modelled work.
"""
from __future__ import annotations

from dataclasses import dataclass

from .crossing import timed_rows

#: TFLOP/s over GB/s is 1e12 FLOP/s over 1e9 byte/s.
_TFLOPS_PER_GBPS_TO_FLOP_PER_BYTE = 1000.0


@dataclass(frozen=True)
class Efficiency:
    peak_tflops: float
    peak_gbps: float
    effective_ridge: float
    n_rows: int

    def ratio_against(self, nominal_ridge: float) -> float:
        """`effective / nominal`, the quantity to compare with the crossing offset."""
        if nominal_ridge <= 0:
            raise ValueError("nominal ridge must be positive FLOP/byte")
        return self.effective_ridge / nominal_ridge


def efficiency_from_rows(rows) -> Efficiency | None:
    """Peak achieved FLOP rate and bandwidth over a family of cells.

    The PEAK of each, never the average: a kernel reaches its FLOP peak at large
    batch and its bandwidth peak at small batch, and never both in one cell, so
    an average describes neither end.

    None rather than a zero ridge when nothing usable is present. A zero would
    predict a crossing of zero and read like a finding.
    """
    tflops: list[float] = []
    gbps: list[float] = []
    for r in timed_rows(list(rows)):
        try:
            t, g = float(r["tflops"]), float(r["compulsory_gbps"])
        except (KeyError, TypeError, ValueError):
            continue
        if t > 0:
            tflops.append(t)
        if g > 0:
            gbps.append(g)
    if not tflops or not gbps:
        return None
    pt, pg = max(tflops), max(gbps)
    return Efficiency(
        peak_tflops=pt,
        peak_gbps=pg,
        effective_ridge=pt / pg * _TFLOPS_PER_GBPS_TO_FLOP_PER_BYTE,
        n_rows=max(len(tflops), len(gbps)),
    )
