"""Where a model crosses from memory-bound to compute-bound, predicted.

This exists so a new device or a new dtype is a TEST rather than another data
dump. The number is produced before the sweep runs, and the sweep can falsify it.

The derivation, which is claim C2 of the study (see docs/STUDY.md). Every weight
element is used exactly once per row, contributing 2 FLOPs, and costs `b` bytes
read once. For an expert holding `N` weight elements across any number of layers
and receiving `R` rows:

    FLOPs = 2 N R        weight bytes = N b        AI = 2NR / Nb = 2R / b

`N` is a SUM over layers and cancels, so layer count and matrix shapes are
irrelevant: only rows-per-expert and bytes-per-element matter. Once routing has
saturated and every expert is active, `R = T k / E`, giving

    AI = 2 T k / (E b)          AI = ridge  at  T = ridge b E / (2k)

Two predictions fall straight out, and both are cheap to falsify:

  * halving `b` (bf16 -> fp8) must HALVE the crossing
  * a card with a lower ridge must have a LOWER crossing, proportionally

Agrees with Yun et al., arXiv:2507.15465, whose `B_MoE = RP_acc * (n_e/n_k)` is
this relation for bf16. Their H200 SXM5 ridge is 206.15 Op/B, datasheet-derived;
this repo measures 160.4 on the actual card, because achieved bandwidth reaches
91.1% of spec while achieved bf16 reaches only 70.9%.
"""
from __future__ import annotations

import math

from ..spec import MODEL_CONFIGS, dtype_bytes

#: Fraction of experts that must be active before `R = T k / E` is a fair
#: description. Below saturation some experts are untouched and the active ones
#: hold more rows than the ratio suggests.
SATURATION_FRACTION = 0.99


def _cfg(model: str):
    """KeyError for an unknown model, rather than a plausible number."""
    return MODEL_CONFIGS[model]


def rows_per_expert(model: str, num_tokens: int) -> float:
    """`T k / E`, the saturated ratio. Valid above `saturation_batch`."""
    cfg = _cfg(model)
    return num_tokens * cfg.top_k / cfg.num_experts


def saturation_batch(model: str, fraction: float = SATURATION_FRACTION) -> float:
    """Tokens needed before `fraction` of experts are active.

    Coupon collector: with `k` of `E` experts drawn per token, an expert is
    missed by one token with probability `1 - k/E`, so the expected active count
    is `E (1 - (1 - k/E)^T)`. Solving for the fraction gives the T below which
    the saturated ratio overstates how spread out the rows are.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    cfg = _cfg(model)
    miss = 1.0 - cfg.top_k / cfg.num_experts
    if miss <= 0.0:          # every token touches every expert
        return 1.0
    return math.log(1.0 - fraction) / math.log(miss)


def crossing_batch(model: str, ridge: float, dtype: str = "bf16") -> float:
    """Batch size at which this model reaches `ridge` FLOP/byte.

    `ridge` is the device's compute-over-bandwidth crossover, in FLOP/byte.
    Prefer a MEASURED ridge over a datasheet one: on the H200 the datasheet says
    206.15 and the card measures 160.4, which moves this answer by 25%.
    """
    if ridge <= 0:
        raise ValueError("ridge must be positive FLOP/byte")
    cfg = _cfg(model)
    b = dtype_bytes(dtype)          # ValueError for an unknown dtype
    return ridge * b * cfg.num_experts / (2.0 * cfg.top_k)


def is_compute_bound(model: str, num_tokens: int, ridge: float,
                     dtype: str = "bf16") -> bool:
    """Which side of the ridge a batch sits on, by the saturated model."""
    return num_tokens > crossing_batch(model, ridge, dtype)


def arithmetic_intensity(model: str, num_tokens: int, dtype: str = "bf16") -> float:
    """`2R/b`. Compare against a measured `arith_intensity_compulsory` column.

    Expect the measured value to fall BELOW this at large batch: activations
    enter the denominator once weights stop dominating, which is why mixtral
    (F/H = 3.50, very wide intermediates) deviates most and deepseek-v3
    (F/H = 0.29) least.
    """
    return 2.0 * rows_per_expert(model, num_tokens) / dtype_bytes(dtype)


def predict_table(ridge: float, dtype: str = "bf16",
                  models: tuple[str, ...] | None = None) -> list[dict]:
    """One row per model: what to expect from a sweep on this device."""
    names = models or tuple(m for m in MODEL_CONFIGS if m != "toy")
    out = []
    for name in names:
        cfg = MODEL_CONFIGS[name]
        out.append({
            "model": name,
            "experts": cfg.num_experts,
            "top_k": cfg.top_k,
            "dilution_E_over_k": cfg.num_experts / cfg.top_k,
            "saturates_at": saturation_batch(name),
            "crossing": crossing_batch(name, ridge, dtype),
        })
    return out
