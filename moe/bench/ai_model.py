"""Arithmetic intensity of one tiled GEMM, with BOTH operand re-reads.

WHAT THIS CORRECTS. docs/FINDINGS.md states the model as

    AI(r) = (2r/b) / Q(r),   Q(r) = 1 + alpha*(ceil(r/BLOCK_M) - 1)

which counts the WEIGHT re-read and nothing else. Two terms are missing and one
of them is not small.

Start from the standard GEMM basis. For C[M,N] = A[M,K] @ B[K,N] the compulsory
traffic is MK + KN + MN elements and the arithmetic is 2MNK, so

    AI_ideal = 2MNK / (b * (MK + KN + MN))

Now tile it. The grid is n = ceil(M/BM) by m = ceil(N/BN), and EACH tile reads a
BM x K slice of A and a K x BN slice of B. So A is read m times and B is read n
times -- the weight re-read the study models is the B side, and there is a
SYMMETRIC re-read on the A side that it does not model at all:

    A bytes = M*K*b * (1 + alpha_a*(m - 1))     activations, re-read per N-tile
    B bytes = K*N*b * (1 + alpha_b*(n - 1))     weights,     re-read per M-tile
    C bytes = M*N*b                             written once; K is a loop, not
                                                a split, so there is no
                                                accumulator round-trip

alpha_b is the study's `alpha`. alpha_a is its counterpart on the activations and
has never been measured here.

WHY IT MATTERS RATHER THAN BEING A ROUNDING TERM. The ratio of the two re-read
costs is exactly BM/BN, which the study already knows -- it quotes it as the
"activation confound" bound on the alpha FIT -- but it does not appear in the AI
formula the roofline is drawn from. At the sweep's own pinned BLOCK_N=64 with
BLOCK_M=64 the ratio is 1.0, so the term it drops is the same size as the term it
keeps.

AND IT EXPLAINS A DRIFT THE STUDY RECORDS AS UNEXPLAINED.
scripts/block_m_crossing_sweep.py:144 carries ALPHA_BY_BLOCK_M = {64: 0.466,
128: 0.625} under a comment saying alpha "drifts with BLOCK_M" and that this is
"NOT PINNED AND CANNOT BE". A ladder fit divides the whole per-extra-M-tile cost
by the weight bytes, so what it returns is not alpha_b but

    alpha_fitted = alpha_b + alpha_a*(BM/BN) + BM/K

which RISES with BM at fixed BN, exactly as recorded. Solving the two recorded
points at BN=64, K=4096 gives alpha_a = 0.143 and alpha_b = 0.307. That is a fit
of two unknowns to two points and is therefore not evidence -- it is a hypothesis
with a decisive test, which is `alpha_fitted` being LINEAR IN 1/BN at fixed BM
with slope alpha_a*BM. Three BN values identify both terms and leave a residual.

Provenance: derived 2026-09-02 after the AI denominator was questioned. Nothing
here is measured; `alpha_a` in particular has no measurement anywhere in this
repo, and every function below takes it as an argument rather than assuming one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class AIModelRefused(ValueError):
    """A shape or tile that does not describe a GEMM this model can score."""


@dataclass(frozen=True)
class Traffic:
    """Bytes moved by one tiled GEMM, split by operand so a term can be checked.

    Kept separate rather than summed because the whole defect this module fixes
    was a total that silently omitted one of its parts.
    """

    activation_bytes: float
    weight_bytes: float
    output_bytes: float
    m_tiles: int
    n_tiles: int

    @property
    def total(self) -> float:
        return self.activation_bytes + self.weight_bytes + self.output_bytes

    def share(self) -> dict[str, float]:
        t = self.total
        return {"activations": self.activation_bytes / t,
                "weights": self.weight_bytes / t,
                "output": self.output_bytes / t}


def _check(M: int, N: int, K: int, block_m: int, block_n: int,
           alpha_b: float, alpha_a: float, b: int) -> None:
    for name, v in (("M", M), ("N", N), ("K", K),
                    ("block_m", block_m), ("block_n", block_n), ("b", b)):
        if v <= 0:
            raise AIModelRefused(f"{name}={v} must be positive")
    # alpha is a MISS FRACTION on a re-read. Outside [0, 1] it is not that
    # quantity, and a fit that returns one is reporting something else -- which
    # is the whole reason this module exists. Refuse rather than extrapolate.
    for name, a in (("alpha_b", alpha_b), ("alpha_a", alpha_a)):
        if not 0.0 <= a <= 1.0:
            raise AIModelRefused(
                f"{name}={a} is outside [0, 1]. A miss fraction cannot exceed a "
                "full re-read; a value above 1 means the quantity being divided "
                "by the operand bytes contains more than that operand's traffic")


def traffic(M: int, N: int, K: int, *, block_m: int, block_n: int,
            alpha_b: float, alpha_a: float, b: int = 2) -> Traffic:
    """Bytes for C[M,N] = A[M,K] @ B[K,N] tiled BM x BN, with both re-reads."""
    _check(M, N, K, block_m, block_n, alpha_b, alpha_a, b)
    n_tiles = math.ceil(M / block_m)      # M-tiles: how often B is re-read
    m_tiles = math.ceil(N / block_n)      # N-tiles: how often A is re-read
    return Traffic(
        activation_bytes=M * K * b * (1.0 + alpha_a * (m_tiles - 1)),
        weight_bytes=K * N * b * (1.0 + alpha_b * (n_tiles - 1)),
        output_bytes=M * N * b,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
    )


def arithmetic_intensity(M: int, N: int, K: int, *, block_m: int, block_n: int,
                         alpha_b: float, alpha_a: float, b: int = 2) -> float:
    """FLOP per byte. At alpha_b = alpha_a = 0 this is the textbook GEMM value."""
    return 2.0 * M * N * K / traffic(
        M, N, K, block_m=block_m, block_n=block_n,
        alpha_b=alpha_b, alpha_a=alpha_a, b=b).total


def ideal_intensity(M: int, N: int, K: int, b: int = 2) -> float:
    """2MNK / (b(MK + KN + MN)). Compulsory traffic, every operand read once."""
    if min(M, N, K, b) <= 0:
        raise AIModelRefused("all of M, N, K, b must be positive")
    return 2.0 * M * N * K / (b * (M * K + K * N + M * N))


def cap(N: int, K: int, *, block_m: int, block_n: int,
        alpha_b: float, alpha_a: float, b: int = 2) -> float:
    """The ceiling on AI as M grows without bound.

        cap = 2 / (b * (alpha_b/BM + alpha_a/BN + 1/K))

    M cancels because every term in the denominator grows linearly in it: more
    rows bring more arithmetic AND more weight re-reads AND more activation and
    output traffic, in fixed proportion. So batching stops helping, and where it
    stops is set by the two tile dimensions and the output width -- not by M.

    THE THREE TERMS ARE THREE SEPARATE CEILINGS and the smallest tile wins:
      alpha_b/BM   weight re-reads, the study's Q term
      alpha_a/BN   activation re-reads, absent from the study's formula
      1/K          the output write, which caps AI at 2K/b even with PERFECT
                   caching -- the study's form says infinity there
    """
    _check(1, N, K, block_m, block_n, alpha_b, alpha_a, b)
    denom = alpha_b / block_m + alpha_a / block_n + 1.0 / K
    return 2.0 / (b * denom)


def decompose_fitted_alpha(alpha_fitted: float, *, block_m: int, block_n: int,
                           K: int, alpha_a: float) -> float:
    """Recover alpha_b from what a ladder fit actually returns.

        alpha_fitted = alpha_b + alpha_a*(BM/BN) + BM/K

    A ladder divides the whole per-extra-M-tile cost by the WEIGHT bytes, so the
    activation and output traffic that also grows per tile lands in the quotient.
    Every alpha this study reports is `alpha_fitted`, not alpha_b, and the gap
    grows with BM/BN -- which is why the published values rise with BLOCK_M.
    """
    if alpha_a < 0.0:
        raise AIModelRefused(f"alpha_a={alpha_a} cannot be negative")
    return alpha_fitted - alpha_a * (block_m / block_n) - block_m / K
