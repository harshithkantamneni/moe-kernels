"""C5's target is the RIDGE RATIO, not 1.00, and it must be read off UNIFORM rows.

Two separate scoring errors, and this file pins both corrections because either
one alone still gives the wrong answer.

FIRST, the target. For bf16 `b = 2`, so `AI = 2R/b = ridge` puts the crossing at
`R = ridge` rows per expert, a DIFFERENT R on each card. Two cards should
therefore show a rows-per-expert ratio equal to their ridge ratio,
145.7 / 176.2 = 0.83. A measured ratio of 1.00 means both cards crossed at the
same rows per expert, which is what NO ridge scaling looks like. The original
table scored against 1.00, so it reported deepseek-v3 as the best agreement in
the set when it was nothing of the kind.

SECOND, the rows. `2R/b` describes UNIFORM routing. `R = T*k/E` is a mean, and
under skew no expert experiences the mean: the busy experts are compute-bound
while the quiet ones are still memory-bound AT THE SAME BATCH, so the layer
straddles the ridge and there is no single crossing to find. Uniform is 14% of
the published cells. The first correction of this file was made against the
other 86% still mixed in, which is why its numbers moved AGAIN and why
`POOLED_TOKENS` is kept here beside `UNIFORM_TOKENS` rather than deleted: the
gap between the two columns is the evidence that pooling is not merely noisy.

The specific bug this file now exists to prevent is a green test defending a
RETRACTED finding. `test_the_deviation_is_monotonic_in_expert_count` used to
live here, called itself "the observation that survives the rescoring", and had
the next experiment attached to it. It was an artifact of pooling. Under uniform
the deviation is not monotonic in E and mixtral moves from worst to best, so the
assert below is the retraction rather than the claim.

Both columns are regenerable from `results/published/`; see
`tests/test_routing_domain.py` for the loader and `docs/FINDINGS.md` C5 for the
tables they reproduce.
"""
from __future__ import annotations

import pytest

from moe.spec import MODEL_CONFIGS

#: Measured crossings in tokens, `vllm_fused_experts` bf16, one run per card,
#: UNIFORM ROUTING ONLY. A100 from `2026-08-28-...-a100-cross-card`, H200 from
#: `...-h200-whole-layer`. These are the numbers docs/FINDINGS.md C5 tabulates.
UNIFORM_TOKENS = {
    "mixtral-8x7b": (229, 316),
    "qwen2-57b-a14b": (742, 787),
    "deepseek-v2-lite": (906, 931),
    "deepseek-v3": (2848, 3010),
}

#: The same cells with all four routing kinds pooled: what C5 was scored on
#: before 2026-08-31. Kept ONLY to pin how far pooling moves the answer. Never
#: score anything against these.
POOLED_TOKENS = {
    "mixtral-8x7b": (233, 543),
    "qwen2-57b-a14b": (647, 914),
    "deepseek-v2-lite": (785, 897),
    "deepseek-v3": (3518, 3474),
}

#: Measured bf16 ridges. The H200's is a band; both ends are used below.
RIDGE_A100 = 145.7
RIDGE_H200_BAND = (160.3, 176.2)

#: The single point target C5 would have if the H200 ridge were one number.
#: `RIDGE_H200_BAND[1]` is the end the original table used, kept so the scored
#: values below are comparable with the ones it published.
TARGET = RIDGE_A100 / RIDGE_H200_BAND[1]


def rows_per_expert(model: str, tokens: int) -> float:
    cfg = MODEL_CONFIGS[model]
    return tokens * cfg.top_k / cfg.num_experts


def ratio(model: str, table: dict = UNIFORM_TOKENS) -> float:
    a, h = table[model]
    return rows_per_expert(model, a) / rows_per_expert(model, h)


def scored(model: str, table: dict = UNIFORM_TOKENS) -> float:
    """Measured ratio over the target. 1.00 means C5 holds for this model."""
    return ratio(model, table) / TARGET


def test_the_target_is_the_ridge_ratio_and_it_is_not_one():
    """The whole point. 0.83 to 0.91, never 1.00."""
    targets = [RIDGE_A100 / r for r in RIDGE_H200_BAND]
    assert max(targets) == pytest.approx(0.909, abs=0.005)
    assert min(targets) == pytest.approx(0.827, abs=0.005)
    assert max(targets) < 1.0, "a target of 1.00 is the no-scaling null"


def test_pooling_routings_moves_every_model_and_moves_them_different_ways():
    """Why `UNIFORM_TOKENS` had to replace the pooled table rather than be
    reconciled with it. If pooling were noise the two columns would straddle
    each other; instead mixtral's ratio nearly halves while deepseek-v3's rises
    past the null, so the pooled column is a different measurement and not a
    worse estimate of the same one."""
    assert ratio("mixtral-8x7b", POOLED_TOKENS) == pytest.approx(0.43, abs=0.01)
    assert ratio("mixtral-8x7b") == pytest.approx(0.725, abs=0.005)
    assert ratio("deepseek-v3", POOLED_TOKENS) == pytest.approx(1.01, abs=0.01)
    assert ratio("deepseek-v3") == pytest.approx(0.946, abs=0.005)
    moved = {m: abs(ratio(m) - ratio(m, POOLED_TOKENS)) for m in UNIFORM_TOKENS}
    assert all(v > 0.02 for v in moved.values()), moved


def test_no_model_lands_inside_the_ridge_ratio_band_under_uniform_routing():
    """C5 is NOT ESTABLISHED, and this is the assert that says so. Restricted to
    its own domain the claim has no supporting point at all: mixtral falls below
    the band and the other three sit above it.

    The retracted version of this test was named
    `test_deepseek_v2_lite_is_the_point_that_scales_with_the_ridge`, because
    pooled deepseek-v2-lite landed at 0.876, inside the band. Under uniform it is
    0.973, outside it. That single point was the study's only evidence for C5."""
    lo, hi = RIDGE_A100 / RIDGE_H200_BAND[1], RIDGE_A100 / RIDGE_H200_BAND[0]
    inside = [m for m in UNIFORM_TOKENS if lo <= ratio(m) <= hi]
    assert inside == [], f"expected no model inside {lo:.3f}-{hi:.3f}, got {inside}"
    assert ratio("mixtral-8x7b") < lo
    assert all(ratio(m) > hi for m in UNIFORM_TOKENS if m != "mixtral-8x7b")


def test_deepseek_v3_agrees_with_the_NULL_not_with_C5():
    """Its pooled 1.01 was read as a 1% confirmation of C5 when 1.00 IS the null.
    Uniform moves it to 0.946, which is a 14% overshoot of the target and still
    nearer the null than the prediction."""
    assert ratio("deepseek-v3") == pytest.approx(0.946, abs=0.005)
    assert scored("deepseek-v3") == pytest.approx(1.14, abs=0.02)
    assert scored("deepseek-v3") > 1.1, "not agreement with ridge scaling"


def test_the_deviation_is_NOT_monotonic_in_expert_count():
    """THE RETRACTION. Ordered by expert count the scores are 0.88 / 1.14 / 1.18
    / 1.14, which does not ascend, and mixtral goes from the worst point in the
    pooled set to the best in the uniform one.

    This is asserted rather than deleted because the monotonic reading was load
    bearing: docs/FINDINGS.md and docs/STUDY.md both called it the finding that
    survived C5's rescoring, and an experiment was scheduled against it. A test
    that merely stopped checking would let it drift back in."""
    order = sorted(UNIFORM_TOKENS, key=lambda m: (MODEL_CONFIGS[m].num_experts,
                                                  ratio(m)))
    assert order[0] == "mixtral-8x7b" and order[-1] == "deepseek-v3"
    values = [scored(m) for m in order]
    assert values != sorted(values), f"monotonic in E again: {values}"
    assert values == pytest.approx([0.88, 1.14, 1.18, 1.14], abs=0.02)

    pooled = [scored(m, POOLED_TOKENS) for m in order]
    assert pooled == sorted(pooled), "the pooled set is where the pattern came from"

    best = min(UNIFORM_TOKENS, key=lambda m: abs(scored(m) - 1.0))
    worst = min(POOLED_TOKENS, key=lambda m: scored(m, POOLED_TOKENS))
    assert best == worst == "mixtral-8x7b"


def test_no_model_confirms_cleanly_and_none_refutes_by_an_order_of_magnitude():
    """The shape of the whole result: four points, none within 6% of the target,
    none off by more than 20%. C5 fails for lack of resolution, not because the
    ridge is irrelevant, which is why docs/FINDINGS.md asks for error bars and a
    same-session calibration rather than abandoning the question."""
    values = {m: scored(m) for m in UNIFORM_TOKENS}
    assert [m for m, v in values.items() if 0.94 <= v <= 1.06] == []
    assert all(0.8 < v < 1.25 for v in values.values()), values
    assert max(values.values()) - min(values.values()) == pytest.approx(0.30, abs=0.02)
