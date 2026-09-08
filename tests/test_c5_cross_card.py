"""C5's target is the RIDGE RATIO of the two cards' OWN ridges, read off UNIFORM rows.

Three separate scoring errors, and this file pins all three corrections
because any one alone still gives the wrong answer.

FIRST, the target. For bf16 `b = 2`, so `AI = 2R/b = ridge` puts the crossing at
`R = ridge` rows per expert, a DIFFERENT R on each card. Two cards should
therefore show a rows-per-expert ratio equal to their ridge ratio, never 1.00. A
measured ratio of 1.00 means both cards crossed at the same rows per expert,
which is what NO ridge scaling looks like. The original table scored against
1.00, so it reported deepseek-v3 as the best agreement in the set when it was
nothing of the kind.

SECOND, the rows. `2R/b` describes UNIFORM routing. `R = T*k/E` is a mean, and
under skew no expert experiences the mean: the busy experts are compute-bound
while the quiet ones are still memory-bound AT THE SAME BATCH, so the layer
straddles the ridge and there is no single crossing to find. Uniform is 14% of
the published cells. The first correction of this file was made against the
other 86% still mixed in, which is why its numbers moved AGAIN and why
`POOLED_TOKENS` is kept here beside `UNIFORM_TOKENS` rather than deleted: the
gap between the two columns is the evidence that pooling is not merely noisy.

THIRD, the ridges themselves. Until 2026-09-08 this file declared "the H200's
ridge is a band, (160.3, 176.2)" (withdrawn) and scored against its 176.2 end. That pair is
two calibrations of ONE H200 disagreeing about the compute term by 9.9%, and it
was withdrawn on 2026-09-02 as the ridge of any card (`docs/FINDINGS.md`
RETRACTIONS (e), `moe.bench.profiles.WITHDRAWN_RIDGE_BAND_WHY`). What the pair
legitimately is: the two candidate rulers of the whole-layer arm, whose shipped
calibration disagrees with its rows (`results/published/CALIBRATION_PROVENANCE.md`,
`ceilings_disagree`), so it is a bracket over that ARM's ruler and nothing about
the card. The card's own ridge is the committed calibration's, read here through
`roofline` so the number has a file behind it (H200 162.8, A100 145.8), and the
point target is their ratio, 0.896. The test that pinned the withdrawn band as
"measured" was green, so the suite was defending a retracted number; a test
that pins a retracted value is the project's recurring defect in test form.

Scored against the card's own target the uniform ratios read 0.81 / 1.05 /
1.09 / 1.06, and that moves NO conclusion: the per-cell intervals in
`docs/FINDINGS.md` C5 Defect 2 contain both the target and the null for three
models and exclude both for mixtral, whichever target is used, and the
cross-card mixtral pair compares an A100 first step against an H200 only
crossing (the staircase). C5 stays NOT ESTABLISHED for lack of resolution and a
matched quantity, not because of where the points fall.

The specific bug this file exists to prevent is a green test defending a
RETRACTED finding. `test_the_deviation_is_monotonic_in_expert_count` used to
live here, called itself "the observation that survives the rescoring", and had
the next experiment attached to it. It was an artifact of pooling. Under uniform
the deviation is not monotonic in E under either target, so the assert below is
the retraction rather than the claim; the companion sentence "mixtral moves from
worst to best" turned out to be the old target's and is retracted below too.

Both token columns are regenerable from `results/published/`; see
`tests/test_routing_domain.py` for the loader and `docs/FINDINGS.md` C5 for the
tables they reproduce.
"""
from __future__ import annotations

import pytest

from moe.bench import profiles, roofline
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

#: Each card's OWN bf16 ridge, from its committed calibration through the same
#: resolver every script uses. Not retyped: a literal here is how the withdrawn
#: pair survived two corrections of this file.
RIDGE_A100 = roofline.load_hardware("measured_nvidia_a100_sxm4_80gb").ridge_point("bf16")
RIDGE_H200 = roofline.load_hardware("measured_nvidia_h200").ridge_point("bf16")

#: The point target: the ratio of the cards' own ridges.
TARGET = RIDGE_A100 / RIDGE_H200

#: WITHDRAWN as the H200's ridge band (RETRACTIONS (e), 2026-09-02). Kept under
#: this name for one purpose: it is the bracket over the whole-layer ARM's
#: ruler (rows at 160.3, shipped yaml at 176.2), and the 0.83-0.91 target band
#: docs/FINDINGS.md C5 prints is that bracket. Nothing here calls it the card's.
WHOLE_LAYER_ARM_RULERS_WITHDRAWN_AS_BAND = (160.3, 176.2)

#: The A100 arm's own same-session ruler, the figure C5's table was scored with
#: before the committed calibration (145.8) superseded it. Kept so the arm
#: bracket below reproduces the published 0.827 / 0.909.
RIDGE_A100_ARM = 145.7


def rows_per_expert(model: str, tokens: int) -> float:
    cfg = MODEL_CONFIGS[model]
    return tokens * cfg.top_k / cfg.num_experts


def ratio(model: str, table: dict = UNIFORM_TOKENS) -> float:
    a, h = table[model]
    return rows_per_expert(model, a) / rows_per_expert(model, h)


def scored(model: str, table: dict = UNIFORM_TOKENS, target: float = TARGET) -> float:
    """Measured ratio over the target. 1.00 means C5 holds for this model."""
    return ratio(model, table) / target


def test_the_ridges_are_the_cards_own_and_the_withdrawn_pair_is_neither():
    """THE RETRACTION, (e). Neither end of the old "band" is any committed
    calibration's ridge; the H200's own sits strictly between them; and the
    span of committed ridges `profiles.calibrated_ridge_band` returns is the
    two cards', not the pair. A future edit that puts 160.3 or 176.2 back as
    a card's ridge fails here by name."""
    lo, hi = WHOLE_LAYER_ARM_RULERS_WITHDRAWN_AS_BAND
    assert RIDGE_H200 == pytest.approx(162.8, abs=0.05)
    assert RIDGE_A100 == pytest.approx(145.8, abs=0.05)
    assert lo < RIDGE_H200 < hi
    for end in (lo, hi):
        assert abs(RIDGE_H200 - end) > 1.0 and abs(RIDGE_A100 - end) > 1.0
    band = profiles.calibrated_ridge_band("bf16")
    assert band == pytest.approx((RIDGE_A100, RIDGE_H200))
    assert "160.3" in profiles.WITHDRAWN_RIDGE_BAND_WHY
    assert "176.2" in profiles.WITHDRAWN_RIDGE_BAND_WHY
    assert "withdrawn" in profiles.WITHDRAWN_RIDGE_BAND_WHY


def test_the_target_is_the_ridge_ratio_and_it_is_not_one():
    """The whole point. 0.896 against the cards' own ridges, never 1.00."""
    assert TARGET == pytest.approx(0.896, abs=0.003)
    assert TARGET < 1.0, "a target of 1.00 is the no-scaling null"


def test_the_arm_bracket_is_the_whole_layer_arms_two_rulers_not_the_cards_band():
    """docs/FINDINGS.md C5 prints a target band 0.83-0.91. It is the A100 arm's
    ruler over each of the whole-layer arm's two candidate rulers, a bracket
    over ONE arm's calibration ambiguity, and the card's own point target
    lies inside it. Pinned so the published band keeps its correct label."""
    lo, hi = WHOLE_LAYER_ARM_RULERS_WITHDRAWN_AS_BAND
    targets = sorted(RIDGE_A100_ARM / r for r in (lo, hi))
    assert targets[0] == pytest.approx(0.827, abs=0.005)
    assert targets[1] == pytest.approx(0.909, abs=0.005)
    assert targets[0] < TARGET < targets[1]


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


def test_scored_against_the_cards_own_target_the_points_move_and_the_verdict_does_not():
    """The corrected scores: 0.81 / 1.05 / 1.09 / 1.06. Two of four now sit
    within 6% of the target where the old 0.83 target had none, and that is
    exactly why the score alone was never the verdict: every interval in C5
    Defect 2 (mixtral 0.64-0.80, qwen2 0.73-1.23, v2-lite 0.89-1.08, v3
    0.88-1.02, on the ratio) either contains both the target and the null or
    excludes both, under either target. Pinned against the old target too, so
    the size of the move (7.7%) is on record and cannot drift back."""
    values = {m: scored(m) for m in UNIFORM_TOKENS}
    assert values["mixtral-8x7b"] == pytest.approx(0.81, abs=0.01)
    assert values["qwen2-57b-a14b"] == pytest.approx(1.05, abs=0.01)
    assert values["deepseek-v2-lite"] == pytest.approx(1.09, abs=0.01)
    assert values["deepseek-v3"] == pytest.approx(1.06, abs=0.01)
    old = {m: scored(m, target=RIDGE_A100_ARM / 176.2) for m in UNIFORM_TOKENS}
    assert old == pytest.approx({"mixtral-8x7b": 0.88, "qwen2-57b-a14b": 1.14,
                                 "deepseek-v2-lite": 1.18, "deepseek-v3": 1.14},
                                abs=0.01)
    intervals = {"mixtral-8x7b": (0.64, 0.80), "qwen2-57b-a14b": (0.73, 1.23),
                 "deepseek-v2-lite": (0.89, 1.08), "deepseek-v3": (0.88, 1.02)}
    for m, (lo, hi) in intervals.items():
        has_target, has_null = lo <= TARGET <= hi, lo <= 1.0 <= hi
        assert has_target == has_null, (m, "an interval that split target from null "
                                           "would be a discriminating point")
    assert [m for m, (lo, hi) in intervals.items() if lo <= TARGET <= hi] == [
        "qwen2-57b-a14b", "deepseek-v2-lite", "deepseek-v3"]


def test_deepseek_v3_cannot_tell_the_null_from_C5():
    """Its pooled 1.01 was read as a 1% confirmation of C5 when 1.00 IS the null.
    Uniform moves it to 0.946, which sits 0.050 from the card's-own target and
    0.054 from the null: equidistant to within the third decimal, and its
    5th-95th interval (0.88-1.02) contains both. Against the old 0.827 target
    it read as "nearer the null"; against the corrected one that sentence
    reverses by 0.004, which is the measure of how little either reading was
    worth. What is pinned is that the point discriminates nothing."""
    assert ratio("deepseek-v3") == pytest.approx(0.946, abs=0.005)
    assert scored("deepseek-v3") == pytest.approx(1.056, abs=0.01)
    to_target = abs(ratio("deepseek-v3") - TARGET)
    to_null = abs(ratio("deepseek-v3") - 1.0)
    assert to_target == pytest.approx(0.050, abs=0.002)
    assert to_null == pytest.approx(0.054, abs=0.002)
    assert abs(to_target - to_null) < 0.01, "neither reading is supported"
    assert 0.88 <= TARGET <= 1.02 and 0.88 <= 1.0 <= 1.02


def test_the_deviation_is_NOT_monotonic_in_expert_count():
    """THE RETRACTION. Ordered by expert count the scores are 0.81 / 1.05 / 1.09
    / 1.06, which does not ascend, under the cards' own target as under the
    old one (0.88 / 1.14 / 1.18 / 1.14).

    This is asserted rather than deleted because the monotonic reading was load
    bearing: docs/FINDINGS.md and docs/STUDY.md both called it the finding that
    survived C5's rescoring, and an experiment was scheduled against it. A test
    that merely stopped checking would let it drift back in.

    AND A SECOND SENTENCE FALLS WITH THE TARGET. "mixtral moves from the worst
    point in the pooled set to the best in the uniform one" held only against
    0.827: against the cards' own 0.896 the nearest point is qwen2 (1.05) and
    mixtral, 19% under, is the worst point in BOTH sets. Pinned so the
    target-dependence of that sentence is on record."""
    order = sorted(UNIFORM_TOKENS, key=lambda m: (MODEL_CONFIGS[m].num_experts,
                                                  ratio(m)))
    assert order[0] == "mixtral-8x7b" and order[-1] == "deepseek-v3"
    values = [scored(m) for m in order]
    assert values != sorted(values), f"monotonic in E again: {values}"
    assert values == pytest.approx([0.81, 1.05, 1.09, 1.06], abs=0.01)

    pooled = [scored(m, POOLED_TOKENS) for m in order]
    assert pooled == sorted(pooled), "the pooled set is where the pattern came from"

    worst_pooled = min(POOLED_TOKENS, key=lambda m: scored(m, POOLED_TOKENS))
    worst_uniform = max(UNIFORM_TOKENS, key=lambda m: abs(scored(m) - 1.0))
    best_uniform = min(UNIFORM_TOKENS, key=lambda m: abs(scored(m) - 1.0))
    assert worst_pooled == worst_uniform == "mixtral-8x7b"
    assert best_uniform == "qwen2-57b-a14b"
    old_best = min(UNIFORM_TOKENS,
                   key=lambda m: abs(scored(m, target=RIDGE_A100_ARM / 176.2) - 1.0))
    assert old_best == "mixtral-8x7b", "the worst-to-best sentence was the old target's"


def test_none_refutes_by_an_order_of_magnitude_and_the_spread_is_the_shape():
    """The shape of the whole result: four points, none off by more than 20%
    on either side, spanning 0.28 between best and worst. C5 fails for lack
    of resolution (Defect 2) and a matched quantity (the staircase), not
    because the ridge is irrelevant, which is why docs/FINDINGS.md asks for
    error bars and a same-session calibration rather than abandoning it."""
    values = {m: scored(m) for m in UNIFORM_TOKENS}
    assert all(0.8 < v < 1.25 for v in values.values()), values
    assert max(values.values()) - min(values.values()) == pytest.approx(0.28, abs=0.02)
