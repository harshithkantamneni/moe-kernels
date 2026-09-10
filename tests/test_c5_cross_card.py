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
`roofline` so the number has a file behind it, and the point target is their
ratio. The test that pinned the withdrawn band as "measured" was green, so the
suite was defending a retracted number; a test that pins a retracted value is
the project's recurring defect in test form.

NO CALIBRATION-DERIVED NUMBER IS WRITTEN DOWN IN THIS FILE. The H200's compute
term has read 162.8, then 152.8, then 155.9 in nine days, and its
non-reproduction is itself a finding of the study, so an asserted literal for
the ridge, the target or any score is a test that fires on the next session.
Every such assert below is on the RELATION instead: the target is the ratio of
the two committed calibrations, the scores are the measured ratios over it, and
what the file pins is the shape that survives any ruler. The historical
literals that remain (the withdrawn 160.3/176.2 pair, the arm's 145.7, the
published 0.827/0.909 bracket) are pinned AS history and are not read off any
calibration.

Scored against the card's own target the uniform ratios move with it, and that
moves NO conclusion: the per-cell intervals in
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
    calibration's ridge, and the span of committed ridges
    `profiles.calibrated_ridge_band` returns is the two cards', not the pair.
    The H200's own ridge sat strictly between the two ends while it read
    162.8, then below both at 152.8, then between them again at 155.9, so what
    is asserted is separation from each end rather than containment or
    position. A future edit that puts 160.3 or 176.2 back as a card's ridge
    fails here by name."""
    lo, hi = WHOLE_LAYER_ARM_RULERS_WITHDRAWN_AS_BAND
    for card, ridge in (("measured_nvidia_h200", RIDGE_H200),
                        ("measured_nvidia_a100_sxm4_80gb", RIDGE_A100)):
        assert ridge == pytest.approx(
            roofline.load_hardware(card).ridge_point("bf16")), \
            "a ridge in this file must be the card's committed one, not a literal"
    for end in (lo, hi):
        assert abs(RIDGE_H200 - end) > 1.0 and abs(RIDGE_A100 - end) > 1.0
    band = profiles.calibrated_ridge_band("bf16")
    assert band == pytest.approx((RIDGE_A100, RIDGE_H200))
    assert "160.3" in profiles.WITHDRAWN_RIDGE_BAND_WHY
    assert "176.2" in profiles.WITHDRAWN_RIDGE_BAND_WHY
    assert "withdrawn" in profiles.WITHDRAWN_RIDGE_BAND_WHY


def test_the_target_is_the_ridge_ratio_and_it_is_not_one():
    """The whole point: a ratio of the two cards' own ridges, never 1.00. It
    read 0.896 when the H200's ridge read 162.8, 0.954 when that ridge read
    152.8, and moves again with every recalibration. The target is a property
    of the two committed calibrations, so it is READ from them here and never
    written down; the only things that must hold under every calibration are
    that it is that ratio and that it is not the no-scaling null.

    The literal that used to sit on the next line, `approx(0.954, abs=0.003)`,
    was added beside the relational assert as a belt-and-braces check and
    fired one session later. That is the repository's recurring defect, a fix
    applied at one of two sites, committed inside the fix for it."""
    assert TARGET == pytest.approx(RIDGE_A100 / RIDGE_H200)
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
    # The card's own target has been inside this bracket (0.896), above it
    # (0.954) and inside it again, on three calibrations of one H200 that
    # re-timed nothing. So WHERE it falls is not the finding and is not
    # asserted; a test that pinned "inside" or "above" would fire on the next
    # session either way. What is pinned is that the two are different
    # quantities: the bracket is built from the whole-layer arm's two rulers
    # and the A100 arm's, none of which is a committed card ridge.
    assert RIDGE_A100_ARM != pytest.approx(RIDGE_A100, abs=1e-9)
    for end in WHOLE_LAYER_ARM_RULERS_WITHDRAWN_AS_BAND:
        assert abs(RIDGE_H200 - end) > 1.0
    assert TARGET != pytest.approx(targets[0], abs=1e-6)
    assert TARGET != pytest.approx(targets[1], abs=1e-6)


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
    """The scores against the cards' own target read 0.88 / 1.14 / 1.18 / 1.14
    against the withdrawn arm's 0.827, and have since read three more sets
    against three calibrations of the same H200. NONE OF THEM IS WRITTEN DOWN
    HERE, because a score is the measured ratio over a ruler that does not
    reproduce, and the earlier sets were asserted as literals and fired. That
    is the clearest statement of why the score alone was never the verdict:
    the points did not move, the ruler did, and a number that swings this far
    on a recalibration of one card decides nothing.

    What does not move is the discrimination test: every interval in C5
    Defect 2 (mixtral 0.64-0.80, qwen2 0.73-1.23, v2-lite 0.89-1.08, v3
    0.88-1.02, on the ratio) either contains both the target and the null or
    excludes both, under every target this study has quoted. Those intervals
    are on the RATIO, which is measured, so they are literals the test owns."""
    values = {m: scored(m) for m in UNIFORM_TOKENS}
    assert values == pytest.approx({m: ratio(m) / TARGET for m in UNIFORM_TOKENS})
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
    # Which models those are is not spelled out either: the set that admits
    # the target is by construction the set that admits the null, so name it
    # by the null, which no calibration moves.
    assert [m for m, (lo, hi) in intervals.items() if lo <= TARGET <= hi] == \
        [m for m, (lo, hi) in intervals.items() if lo <= 1.0 <= hi]


def test_deepseek_v3_cannot_tell_the_null_from_C5():
    """Its pooled 1.01 was read as a 1% confirmation of C5 when 1.00 IS the
    null. Uniform moves it to 0.946, and where that sits depends entirely on
    which calibration the target came from: 0.054 from the null throughout,
    but 0.119 from the old 0.827 target, 0.050 from the 0.896 one, and 0.008
    from the 0.954 the cards read today. So the same measured ratio has been
    "nearer the null", "equidistant to the third decimal", and "almost exactly
    on the target" without a single kernel being re-timed. The reading is the
    ruler's, not the card's.

    WHAT IS PINNED IS THAT THE POINT DISCRIMINATES NOTHING, and it is pinned
    on the interval rather than on the distance: v3's 5th-95th interval
    (0.88-1.02) contains the null and every target this study has quoted, so
    no reading of the point separates them. A distance that reverses on a
    recalibration is not evidence in either direction."""
    assert ratio("deepseek-v3") == pytest.approx(0.946, abs=0.005)
    assert scored("deepseek-v3") == pytest.approx(ratio("deepseek-v3") / TARGET)
    to_null = abs(ratio("deepseek-v3") - 1.0)
    assert to_null == pytest.approx(0.054, abs=0.002)
    lo, hi = 0.88, 1.02
    assert lo <= 1.0 <= hi, "the null"
    assert lo <= 145.8 / 162.8 <= hi, "the 2026-09-02 target"
    assert lo <= TARGET <= hi, "the target the cards read today"
    # The whole-layer arm's own 0.827 is the one target that falls OUTSIDE
    # this interval, which is what made it look discriminating in 2026-09-01
    # and is the reading FINDINGS withdrew.
    assert not lo <= RIDGE_A100_ARM / 176.2 <= hi
    # The DISTANCE to the target is the quantity that reverses, so instead of
    # asserting today's sign the reversal itself is pinned, on the two
    # historical targets that produced it. Whichever side today's ruler falls
    # on, a quantity that changes sign without a kernel being re-timed is not
    # evidence, and containment above is the whole finding.
    to_arm = abs(ratio("deepseek-v3") - RIDGE_A100_ARM / 176.2)
    to_0902 = abs(ratio("deepseek-v3") - 145.8 / 162.8)
    assert to_arm > to_null > to_0902, (to_arm, to_null, to_0902)


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
    0.827: against the cards' own target the nearest point is qwen2 and
    mixtral is the worst point in BOTH sets. That holds at the 0.896 the
    cards read on 2026-09-02 and at the 0.954 they read after the H200's
    2026-09-09 recalibration; only the sizes move, which is the
    target-dependence being put on record."""
    order = sorted(UNIFORM_TOKENS, key=lambda m: (MODEL_CONFIGS[m].num_experts,
                                                  ratio(m)))
    assert order[0] == "mixtral-8x7b" and order[-1] == "deepseek-v3"
    values = [scored(m) for m in order]
    assert values != sorted(values), f"monotonic in E again: {values}"
    # Dividing four ratios by one positive target cannot reorder them, so the
    # retraction holds under EVERY ruler and is checked that way instead of
    # against a set of scores that a recalibration rewrites.
    for t in (TARGET, RIDGE_A100_ARM / 176.2, RIDGE_A100_ARM / 160.3, 1.0):
        v = [scored(m, target=t) for m in order]
        assert v != sorted(v), f"monotonic in E under target {t}: {v}"

    pooled = [scored(m, POOLED_TOKENS) for m in order]
    assert pooled == sorted(pooled), "the pooled set is where the pattern came from"

    worst_pooled = min(POOLED_TOKENS, key=lambda m: scored(m, POOLED_TOKENS))
    worst_uniform = max(UNIFORM_TOKENS, key=lambda m: abs(scored(m) - 1.0))
    best_uniform = min(UNIFORM_TOKENS, key=lambda m: abs(scored(m) - 1.0))
    assert worst_pooled == worst_uniform == "mixtral-8x7b"
    # WHICH point is nearest the null is not a finding: qwen2 and deepseek-v3
    # sit 0.012 and 0.008 from it on today's calibration and 0.05 and 0.06 on
    # the 2026-09-02 one, so the two swap places on a recalibration that
    # re-timed nothing. What survives is that the nearest point is not the
    # one the withdrawn 0.827 target named, which is the sentence being
    # retired here.
    assert best_uniform != "mixtral-8x7b"
    # WHICH of the other three is nearest is the thing that swaps, so it is
    # not named. What is pinned is that mixtral is the outlier under every
    # ruler and that the other three are bunched against whichever one is
    # nearest, by a margin far smaller than mixtral's.
    for t in (TARGET, RIDGE_A100_ARM / 176.2, RIDGE_A100_ARM / 160.3, 1.0):
        d = sorted(abs(scored(m, target=t) - 1.0) for m in UNIFORM_TOKENS)
        assert d[2] - d[0] < d[3] - d[2], f"mixtral is not the outlier at {t}"
    old_best = min(UNIFORM_TOKENS,
                   key=lambda m: abs(scored(m, target=RIDGE_A100_ARM / 176.2) - 1.0))
    assert old_best == "mixtral-8x7b", "the worst-to-best sentence was the old target's"


def test_none_refutes_by_an_order_of_magnitude_and_the_spread_is_the_shape():
    """The shape of the whole result: four points, none off by more than 25%
    on either side, spanning 0.26 between best and worst. C5 fails for lack
    of resolution (Defect 2) and a matched quantity (the staircase), not
    because the ridge is irrelevant, which is why docs/FINDINGS.md asks for
    error bars and a SAME-SESSION calibration rather than abandoning it. The
    2026-09-09 recalibration is that request answered in the negative: every
    point slid together because the target moved and the measurements did
    not, so the spread in score space is just the measured spread over the
    target and carries no information the ratios do not already carry. It is
    asserted that way rather than as a number, which is why nothing here
    fires when the H200 is recalibrated again."""
    ratios = {m: ratio(m) for m in UNIFORM_TOKENS}
    assert all(0.70 < r < 1.00 for r in ratios.values()), ratios
    spread = max(ratios.values()) - min(ratios.values())
    assert spread == pytest.approx(0.248, abs=0.005)
    values = {m: scored(m) for m in UNIFORM_TOKENS}
    assert max(values.values()) - min(values.values()) == \
        pytest.approx(spread / TARGET)
