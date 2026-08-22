"""Load-imbalance metrics for one routing realisation.

These are CSV columns, not diagnostics. The central claim of the project is
that kernel performance depends on the *shape* of the expert load and not just
on the token count, so every timing row carries the imbalance of the exact
distribution it was measured on.

Pure python plus numpy. No torch, no CUDA.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExpertLoad:
    num_experts: int
    total_rows: int
    active_experts: int      # experts receiving at least one row
    empty_experts: int
    max_rows: int
    min_rows: int
    mean_rows: float
    max_over_mean: float     # the number that breaks a fixed BLOCK_M schedule
    cv: float                # coefficient of variation
    entropy_norm: float      # 1.0 = perfectly uniform, 0.0 = one expert holds all
    gini: float
    top1_share: float        # fraction of all rows held by the hottest expert

    def as_row(self) -> dict:
        return {f"load_{k}": v for k, v in asdict(self).items()}


def expert_load(counts) -> ExpertLoad:
    c = [int(v) for v in counts]
    E = len(c)
    if E == 0:
        raise ValueError("counts must be non-empty")
    if any(v < 0 for v in c):
        raise ValueError("counts must be non-negative")
    total = sum(c)
    mean = total / E

    active = sum(1 for v in c if v > 0)
    if total == 0:
        # No rows routed at all. There is no distribution, so entropy is 0 and
        # not 1: "perfectly uniform" would be the wrong reading.
        return ExpertLoad(
            num_experts=E, total_rows=0, active_experts=0, empty_experts=E,
            max_rows=0, min_rows=0, mean_rows=0.0, max_over_mean=0.0, cv=0.0,
            entropy_norm=0.0, gini=0.0, top1_share=0.0,
        )

    var = sum((v - mean) ** 2 for v in c) / E
    cv = math.sqrt(var) / mean if mean > 0 else 0.0

    # Shannon entropy of the load distribution, normalised to [0, 1].
    ent = 0.0
    for v in c:
        if v > 0:
            p = v / total
            ent -= p * math.log(p)
    entropy_norm = ent / math.log(E) if E > 1 else 1.0

    srt = sorted(c)
    cum = 0.0
    for i, v in enumerate(srt, start=1):
        cum += i * v
    gini = (2.0 * cum) / (E * total) - (E + 1.0) / E

    return ExpertLoad(
        num_experts=E,
        total_rows=total,
        active_experts=active,
        empty_experts=E - active,
        max_rows=max(c),
        min_rows=min(c),
        mean_rows=mean,
        max_over_mean=max(c) / mean if mean > 0 else 0.0,
        cv=cv,
        entropy_norm=entropy_norm,
        gini=max(0.0, gini),
        top1_share=max(c) / total,
    )


def counts_from_offsets(expert_offsets) -> list[int]:
    off = [int(v) for v in expert_offsets]
    return [off[i + 1] - off[i] for i in range(len(off) - 1)]


def padded_rows(counts, block_m: int) -> int:
    """Rows a fixed-BLOCK_M grouped GEMM must actually compute.

    Each expert's group is rounded up to a whole number of BLOCK_M tiles, so the
    wasted work is the gap between this and the true row count. This single
    function is the quantitative form of the TritonMoE limitation: it is why a
    fixed BLOCK_M degrades under imbalance, and it predicts by how much.
    """
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    return sum(((int(v) + block_m - 1) // block_m) * block_m for v in counts)


def tile_efficiency(counts, block_m: int) -> float:
    """Useful rows divided by computed rows. 1.0 means no padding waste."""
    total = sum(int(v) for v in counts)
    if total == 0:
        return 0.0
    return total / padded_rows(counts, block_m)
