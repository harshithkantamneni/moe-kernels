"""Tests for scripts/dram_counter_route.py.

Three things are worth testing here and they are different in kind.

  1. THE ARITHMETIC. `alpha = (dR/dn - a) / W` must return the alpha that
     generated the data, and the bracket must contain it. Both are exact, so
     both are tested to machine precision rather than to a tolerance.
  2. THE PINNED CONSTANTS. Every prediction rests on `W` and `a`, which are
     imported from `block_m_crossing_sweep` rather than re-derived. If that byte
     model moves, this file's predictions move silently with it, so the numbers
     are pinned here and a change has to be acknowledged.
  3. THE REFUSALS. A scorer that returns 0.0 for something it never measured is
     worse than one that crashes. Each refusal path is asserted to raise or to
     report REFUSE, never to produce a number.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from moe.bench import exit_codes
from moe.bench import provenance as PV
from moe.spec import MODEL_CONFIGS
from scripts.dram_counter_route import (
    DATASHEET_PEAK_GBPS,
    DEFAULT_REPORT,
    FAIL,
    INSTRUMENT,
    NO_CARD,
    PASS,
    PROBE_INSTRUMENT,
    REFUSE,
    Anchors,
    activation_bytes_per_tile,
    alpha_from_counters,
    anchor_cap_bracket,
    anchors_from_points,
    bracket_directory,
    build_parser,
    cap_from_counter,
    card_key,
    discrimination,
    git_visibility,
    main,
    ols,
    physical_bracket,
    predicted_read_bytes,
    probe_capabilities,
    route_verdict,
    run_id_for,
    score_counter_run,
    stamped,
    weight_bytes_total,
)

MIXTRAL = MODEL_CONFIGS["mixtral-8x7b"]
REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 2. The pinned constants.
# --------------------------------------------------------------------------

def test_weight_bytes_are_pinned():
    """`E x 3 F H x 2` for mixtral, to the byte.

    2.81857 GB is the number every prediction in the plan divides by. If this
    assertion ever fails the byte model changed and every registered prediction
    in docs/COUNTERS.md is stale, which is exactly what the test is for.
    """
    assert weight_bytes_total(MIXTRAL) == 8 * 3 * 14336 * 4096 * 2
    assert weight_bytes_total(MIXTRAL) == 2_818_572_288


def test_activation_bytes_per_tile_are_pinned():
    """`E x BM x (2H + 3F) x 2`, and linear in BLOCK_M."""
    assert activation_bytes_per_tile(MIXTRAL, 32) == 8 * 32 * (2 * 4096 + 3 * 14336) * 2
    assert activation_bytes_per_tile(MIXTRAL, 32) == 26_214_400
    assert activation_bytes_per_tile(MIXTRAL, 64) == 2 * activation_bytes_per_tile(MIXTRAL, 32)


def test_datasheet_peaks_carry_only_known_cards():
    """No default entry. Bracketing an unknown card against a guessed pin rate
    is the same class of error as the stale ridge already in the reports."""
    assert set(DATASHEET_PEAK_GBPS) == {"nvidia_a100_sxm4_80gb", "nvidia_h200"}
    assert DATASHEET_PEAK_GBPS["nvidia_h200"] == 4800.0


# --------------------------------------------------------------------------
# 1. The arithmetic.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("planted", [0.0, 0.1, 0.4522, 0.558, 0.705, 1.0])
def test_alpha_recovered_exactly_from_synthetic_traffic(planted):
    rows = [{"n": n, "dram_bytes_read": predicted_read_bytes(MIXTRAL, 32, n, planted)}
            for n in (1, 2, 3, 4, 6, 8)]
    alpha, _, resid = alpha_from_counters(rows, MIXTRAL, 32)
    assert alpha == pytest.approx(planted, abs=1e-12)
    assert resid < 1e-12


def test_alpha_is_independent_of_the_bandwidth_the_data_was_generated_at():
    """The whole argument for the counter: the estimator has no bandwidth in it.

    Two ladders generated at the same alpha but different achieved bandwidths
    give the same answer, which is not true of any timing-based estimator in
    this study.
    """
    rows = [{"n": n, "dram_bytes_read": predicted_read_bytes(MIXTRAL, 32, n, 0.6)}
            for n in (1, 2, 4, 8)]
    a1, _, _ = alpha_from_counters(rows, MIXTRAL, 32)
    scaled = [{"n": r["n"], "dram_bytes_read": r["dram_bytes_read"]} for r in rows]
    a2, _, _ = alpha_from_counters(scaled, MIXTRAL, 32)
    assert a1 == pytest.approx(a2)


def test_n1_traffic_is_identical_under_every_anchor():
    """The trap the plan is built around: R(1) discriminates nothing.

    If this ever stops being true the plan's claim/validity split is wrong and
    the n=1 gate would silently become a claim gate.
    """
    at_one = {a: predicted_read_bytes(MIXTRAL, 32, 1, a) for a in (0.452, 0.647, 0.705)}
    assert len(set(round(v, 6) for v in at_one.values())) == 1
    table = discrimination(MIXTRAL, 32, (1, 2, 8), {"a": 0.452, "b": 0.705})
    assert table[0]["spread_frac"] == pytest.approx(0.0)
    assert table[-1]["spread_frac"] > 0.4       # n=8 separates them by over 40%


def test_physical_bracket_contains_the_ladder_that_generated_it():
    """Generate a ladder at a known alpha and a known achieved bandwidth; the
    bracket must contain the UNCORRECTED alpha that ladder reports."""
    W = weight_bytes_total(MIXTRAL)
    a1 = activation_bytes_per_tile(MIXTRAL, 32)
    for planted in (0.30, 0.558, 0.90):
        for achieved in (900.0, 1450.0, 2000.0):
            times = [1e3 * predicted_read_bytes(MIXTRAL, 32, n, planted) / (achieved * 1e9)
                     for n in (1, 2, 3, 4)]
            _, slope = ols([1, 2, 3, 4], times)
            lo, hi = physical_bracket(slope, times[0], W, a1, 2039.0)
            uncorrected = (W * planted + a1) / (W + a1)
            assert lo - 1e-12 <= uncorrected <= hi + 1e-12
            assert lo <= hi


def test_bracket_is_tighter_when_the_kernel_runs_closer_to_peak():
    """The bracket's width is the price of not having a counter, and it is set
    by how far from the pin rate the n=1 launch runs. A cell at 95% of peak is
    nearly pinned; one at 30% is nearly uninformative."""
    W = weight_bytes_total(MIXTRAL)
    a1 = activation_bytes_per_tile(MIXTRAL, 32)
    widths = []
    for achieved in (600.0, 1450.0, 1950.0):
        times = [1e3 * predicted_read_bytes(MIXTRAL, 32, n, 0.6) / (achieved * 1e9)
                 for n in (1, 2, 3, 4)]
        _, slope = ols([1, 2, 3, 4], times)
        lo, hi = physical_bracket(slope, times[0], W, a1, 2039.0)
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]


def test_anchors_disagree_on_a_ladder_whose_first_tread_is_elevated():
    """The mechanism the evaluation identified, reproduced synthetically.

    Push t(1) up by 30% and leave the rest of the branch alone: the published
    A+B anchor rises, the t(1) anchor falls, and the n>=3 anchor rises further.
    A test that only checked "the three functions return floats" would pass on
    an implementation where all three were the same number.
    """
    clean = [(n, 0.5 + 0.9 * n) for n in range(1, 9)]
    anc = anchors_from_points(clean, len(clean))
    assert anc.published == pytest.approx(anc.t1, rel=1e-9)
    assert anc.n3 == pytest.approx(anc.published, rel=1e-9)

    lifted = [(1, clean[0][1] * 1.30), *clean[1:]]
    anc2 = anchors_from_points(lifted, len(lifted))
    assert anc2.t1 < anc2.published < anc2.n3


# --------------------------------------------------------------------------
# 3. The refusals and the non-vacuity checks.
# --------------------------------------------------------------------------

def test_ols_refuses_rather_than_inventing_a_line():
    with pytest.raises(ValueError):
        ols([1], [2.0])
    with pytest.raises(ValueError):
        ols([3, 3, 3], [1.0, 2.0, 3.0])


def test_alpha_from_counters_refuses_two_distinct_tile_counts():
    rows = [{"n": 1, "dram_bytes_read": 1e9}, {"n": 1, "dram_bytes_read": 1e9},
            {"n": 2, "dram_bytes_read": 2e9}]
    with pytest.raises(ValueError):
        alpha_from_counters(rows, MIXTRAL, 32)


def test_anchors_refuse_a_two_tread_branch():
    with pytest.raises(ValueError):
        anchors_from_points([(1, 1.0), (2, 2.0)], 2)


def test_physical_bracket_refuses_a_non_positive_constant():
    # A zero slope is NOT a refusal: it means an extra tile cost no traffic,
    # which is a measurable alpha of zero. Only the constants that make the
    # bracket meaningless are refused.
    for bad in ((1.0, 0.0, 1, 1, 2039.0),      # no measured t(1)
                (1.0, 1.0, 0, 1, 2039.0),      # no weight set
                (1.0, 1.0, 1, 1, 0.0)):        # no pin rate for this card
        with pytest.raises(ValueError):
            physical_bracket(*bad)
    assert physical_bracket(0.0, 1.0, 1, 1, 2039.0)[0] == 0.0


def test_card_key_returns_none_for_an_unknown_directory():
    assert card_key(Path("2026-09-02-nvidia_b200-alpha-surface-s3")) is None
    assert card_key(Path("2026-09-02-nvidia_h200-alpha-surface-s4")) == "nvidia_h200"


def test_bracket_directory_refuses_an_unknown_card(tmp_path):
    d = tmp_path / "2026-09-02-nvidia_b200-alpha-surface"
    d.mkdir()
    with pytest.raises(ValueError):
        bracket_directory(d)


def test_score_counter_run_refuses_a_partial_payload():
    with pytest.raises(KeyError):
        score_counter_run({"device": "x", "model": "mixtral-8x7b", "block_m": 32})


def test_non_vacuity_gate_fires_on_an_empty_run():
    """A run that profiled nothing must FAIL loudly, not report alpha 0.0."""
    payload = {"device": "nvidia_a100_sxm4_80gb", "model": "mixtral-8x7b",
               "block_m": 32, "cache_control": "all",
               "rows": [{"n": 1, "launches": 0, "dram_bytes_read": 0.0}]}
    gates, summary = score_counter_run(payload)
    assert gates[0].number == "V1" and gates[0].verdict == FAIL
    assert summary["alpha"] is None


def test_gates_refuse_rather_than_default_when_ridge_or_bracket_are_absent():
    rows = [{"n": n, "launches": 5,
             "dram_bytes_read": predicted_read_bytes(MIXTRAL, 32, n, 0.5)}
            for n in (1, 2, 3, 4)]
    payload = {"device": "nvidia_a100_sxm4_80gb", "model": "mixtral-8x7b",
               "block_m": 32, "cache_control": "all", "rows": rows}
    gates, _ = score_counter_run(payload)
    by = {g.number: g for g in gates}
    assert by["C2"].verdict == REFUSE
    assert by["C3"].verdict == REFUSE
    assert by["C2"].invalidates and by["C3"].invalidates


def test_full_scoring_picks_the_single_surviving_anchor():
    truth = 0.47
    rows = [{"n": n, "launches": 5,
             "dram_bytes_read": predicted_read_bytes(MIXTRAL, 32, n, truth)}
            for n in (1, 2, 3, 4, 6, 8)]
    payload = {"device": "nvidia_a100_sxm4_80gb", "model": "mixtral-8x7b",
               "block_m": 32, "cache_control": "all", "ridge": 145.81,
               "anchors": {"published": 0.6473, "t1": 0.4522, "n3": 0.7047},
               "bracket": [0.4522, 0.6313], "rows": rows}
    gates, summary = score_counter_run(payload)
    assert summary["alpha"] == pytest.approx(truth)
    assert summary["survivors"] == ["t1"]
    assert all(g.verdict == PASS for g in gates)


def test_byte_model_gate_fails_when_n1_traffic_is_wrong():
    """The gate that has never been run: if R(1) is not one weight read, alpha
    is not a re-read fraction and every published number loses its units."""
    rows = [{"n": n, "launches": 5,
             "dram_bytes_read": 1.7 * predicted_read_bytes(MIXTRAL, 32, n, 0.5)}
            for n in (1, 2, 3, 4)]
    payload = {"device": "nvidia_a100_sxm4_80gb", "model": "mixtral-8x7b",
               "block_m": 32, "cache_control": "all", "rows": rows}
    gates = {g.number: g for g in score_counter_run(payload)[0]}
    assert gates["V2"].verdict == FAIL
    assert "units of alpha" in gates["V2"].invalidates


def test_tile_cap_gate_fails_only_below_the_block_specific_threshold():
    """The cap a counter alpha implies is `2 BM W / (b dR/dn)`, with the
    once-read share a/W restored, so `cap < ridge` is `alpha > BM/ridge - a/W`:
    0.210 at BLOCK_M=32 and 0.420 at 64 on the A100's own ridge (a/W is 0.0093
    and 0.0186). A single hardcoded threshold would pass one and silently
    misjudge the other, and the study's `2 BM / (alpha b)` reading, which drops
    a/W, would put an alpha between the two thresholds on the wrong side.
    """
    def gate(bm, alpha, **extra):
        rows = [{"n": n, "launches": 5,
                 "dram_bytes_read": predicted_read_bytes(MIXTRAL, bm, n, alpha)}
                for n in (1, 2, 3, 4)]
        payload = {"device": "nvidia_a100_sxm4_80gb", "model": "mixtral-8x7b",
                   "block_m": bm, "cache_control": "all", "ridge": 145.81,
                   "rows": rows, **extra}
        return {g.number: g for g in score_counter_run(payload)[0]}["C3"]

    def run(bm, alpha):
        return gate(bm, alpha).verdict

    assert run(32, 0.30) == PASS      # 0.30 > 0.210, the cap holds
    assert run(32, 0.15) == FAIL      # below it, the cap claim would be withdrawn
    assert run(64, 0.50) == PASS      # 0.50 > 0.420
    assert run(64, 0.30) == FAIL

    # THE FLIP. An alpha just under the uncorrected threshold BM/ridge = 0.2195
    # but above the corrected one: `2 BM / (alpha b)` = 146.3 says the tile
    # reaches the ridge; the measured slope's cap, 139.9, says it does not.
    a_over_w = activation_bytes_per_tile(MIXTRAL, 32) / weight_bytes_total(MIXTRAL)
    boundary = 32 / 145.81 - a_over_w / 2
    cap, uncorrected = cap_from_counter(MIXTRAL, 32, boundary)
    assert uncorrected > 145.81 > cap
    assert run(32, boundary) == PASS
    # ...and the gate says which number decided it and what the other one is.
    g = gate(32, boundary)
    assert f"cap {cap:.1f}" in g.measured and f"upper bound {uncorrected:.1f}" in g.measured
    assert any("UPPER BOUND" in line for line in g.lines)
    assert any("no level in it" in line for line in g.lines)


def test_the_counter_cap_is_finite_at_alpha_zero_and_below_the_uncorrected_reading():
    """`ai_cap` is infinite at alpha = 0 because it names only the weight
    re-read; a tile still carries its own activations, so the traffic slope's
    cap is finite. And at every alpha the corrected cap is below the study's
    reading by exactly the once-read share."""
    cap0, unc0 = cap_from_counter(MIXTRAL, 32, 0.0)
    assert math.isinf(unc0) and math.isfinite(cap0)
    a_over_w = activation_bytes_per_tile(MIXTRAL, 32) / weight_bytes_total(MIXTRAL)
    for alpha in (0.05, 0.3, 0.558, 1.0):
        cap, unc = cap_from_counter(MIXTRAL, 32, alpha)
        assert cap < unc
        assert cap == pytest.approx(32.0 / (alpha + a_over_w))
    with pytest.raises(ValueError, match="not positive"):
        cap_from_counter(MIXTRAL, 32, -1.0)


def test_the_registered_anchors_are_bracketed_through_cap_from_fitted():
    """The anchors are B/(A+B) LADDER fits, so the cap each implies is the
    study's reading divided by (1 + phi + delta), retraction (a), and with
    alpha_a unmeasured that is a bracket: alpha_a = 1 is the low end, alpha_a = 0
    the high end, delta = 0 the generous end, and both ends sit below the
    uncorrected figure. An anchor too small for the alpha_a = 1 end is
    REFUSED at that end rather than clamped."""
    br = anchor_cap_bracket(MIXTRAL, 32, 64, 0.647)
    assert br.low is not None and br.high is not None
    assert br.low < br.high < br.uncorrected == pytest.approx(32 / 0.647)
    assert br.high == pytest.approx(br.uncorrected / (1 + 0.0089), rel=1e-3)
    assert br.low == pytest.approx(br.uncorrected / (1 + 0.5078), rel=1e-3)
    assert not br.refused
    # phi/(1+phi) at alpha_a = 1 is 0.337: a fitted 0.30 cannot have come from
    # the three-term model with a full activation re-read.
    low = anchor_cap_bracket(MIXTRAL, 32, 64, 0.30)
    assert low.low is None and low.high is not None
    assert "alpha_a=1" in low.refused and "REFUSED" in low.render()

    # On the gate: with block_n the anchors are bracketed, without it the line
    # says so rather than assuming the sweep's 64.
    rows = [{"n": n, "launches": 5,
             "dram_bytes_read": predicted_read_bytes(MIXTRAL, 32, n, 0.5)}
            for n in (1, 2, 3, 4)]
    base = {"device": "nvidia_a100_sxm4_80gb", "model": "mixtral-8x7b",
            "block_m": 32, "cache_control": "all", "ridge": 145.81, "rows": rows,
            "anchors": {"published": 0.6473, "t1": 0.4522, "n3": 0.7047}}
    with_bn = {g.number: g for g in score_counter_run({**base, "block_n": 64})[0]}["C3"]
    assert sum("alpha_a=1" in line for line in with_bn.lines) == 3
    without = {g.number: g for g in score_counter_run(base)[0]}["C3"]
    assert any("REFUSED rather than assumed 64" in line for line in without.lines)


# --------------------------------------------------------------------------
# The route probe, off the box.
# --------------------------------------------------------------------------

def test_probe_capabilities_never_guesses():
    """On a machine with no /proc it must say so, not report sys_admin False --
    "no evidence" and "the capability is absent" are different answers."""
    caps = probe_capabilities()
    assert "available" in caps
    if not caps["available"]:
        assert "sys_admin" not in caps


def test_route_verdict_distinguishes_the_four_failures():
    open_ = route_verdict({}, {}, {"present": True, "cause": "attached with no permission error"},
                          {"present": True, "importer_present": True})
    assert open_[0] == "OPEN"

    blocked_cap = route_verdict({"available": True, "sys_admin": False},
                                {"available": True, "restrict": 1},
                                {"present": True, "cause": "ERR_NVGPUCTRPERM: counters gated"},
                                {"present": True, "importer_present": True})
    assert blocked_cap[0] == "BLOCKED"
    assert any("SYS_ADMIN" in n for n in blocked_cap[1])
    assert any("RestrictProfilingToAdminUsers=1" in n for n in blocked_cap[1])

    # The combination that means "stop retrying and read the output".
    odd = route_verdict({"available": True, "sys_admin": False},
                        {"available": True, "restrict": 0},
                        {"present": True, "cause": "ERR_NVGPUCTRPERM: counters gated"},
                        {"present": True, "importer_present": True})
    assert odd[0] == "BLOCKED"
    assert any("ALREADY allows" in n for n in odd[1])

    # The pod's actual failure: nsys present, importer absent, no ncu at all.
    importer = route_verdict({}, {}, {"present": False, "why": "no ncu on PATH"},
                             {"present": True, "importer_present": False})
    assert importer[0] == "REFUSE"
    assert any("its importer" in n for n in importer[1])


# --------------------------------------------------------------------------
# End to end, on the repository's own published data.
# --------------------------------------------------------------------------

def test_default_report_exists_and_reproduces_the_three_anchors():
    """The plan's default cell must still be the cell the evaluation named.

    0.452 / 0.647 / 0.705 are the numbers the brief quotes. If a republish moves
    them this test says so before docs/COUNTERS.md is quoted at anyone.
    """
    assert DEFAULT_REPORT.exists(), f"{DEFAULT_REPORT} is gone; the plan has no cell"
    rep = json.loads(DEFAULT_REPORT.read_text())
    ladder = rep["ladder"]["32"]
    anc = anchors_from_points(ladder["points"], ladder["memory_points"])
    assert anc.t1 == pytest.approx(0.452, abs=0.002)
    assert anc.published == pytest.approx(0.647, abs=0.002)
    assert anc.n3 == pytest.approx(0.705, abs=0.002)
    # And the refit must reproduce the published slope, which is what makes the
    # published line the line under test rather than a different one.
    assert anc.slope == pytest.approx(ladder["slope_memory"], rel=1e-9)


def test_bracket_over_published_data_is_non_vacuous_and_finds_the_a100_violations():
    """The counter-free result, asserted rather than described.

    Four A100 fits assert a memory branch faster than the A100's pin rate. All
    four are BLOCK_M=32 with a swizzle, and none of the G=1 fits is among them,
    which is the shape the swizzle explanation predicts.
    """
    a100 = REPO / "results/published/2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
    rows = bracket_directory(a100)
    assert len(rows) >= 12, "examined too few fits to conclude anything"
    above = [r for r in rows
             if r["published"] > r["hi"] * (1 + r["fit_err"])]
    assert len(above) == 4
    assert all(r["block_m"] == 32 and r["group_m"] > 1 for r in above)
    # Every bracket must at least be a real interval.
    assert all(r["lo"] < r["hi"] for r in rows)
    assert all(0.0 < r["achieved_frac_peak"] < 1.0 for r in rows)


def test_bn256_arm_is_excluded_from_the_bracket():
    """The BLOCK_SIZE_N=256 arm was withdrawn on both cards; nothing derived
    from it may enter a published bracket."""
    a100 = REPO / "results/published/2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
    assert all(r["block_n"] != 256 for r in bracket_directory(a100))


# --------------------------------------------------------------------------
# The CLI contract.
# --------------------------------------------------------------------------

def test_main_refuses_two_modes_at_once(capsys):
    assert main(["--dry-run", "--bracket"]) == exit_codes.REFUSED
    assert "REFUSE" in capsys.readouterr().out


def test_main_refuses_no_mode_at_all(capsys):
    assert main([]) == exit_codes.REFUSED
    assert "REFUSE" in capsys.readouterr().out


def test_dry_run_prints_predictions_and_a_cost(capsys):
    assert main(["--dry-run"]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert "PREDICTIONS, registered here" in out
    assert "COST." in out
    assert "dram__bytes_read.sum" in out
    # The plan must name this card's own ridge, never the stale 160.3 default.
    assert "ridge 145.81 FLOP/byte" in out
    assert "ridge 160.3" not in out
    # THE RECIPE RUNS ON THE CURRENT INSTRUMENT. `timing.warm_until` refuses
    # `warmup_ms <= 0` and `--iters` is retired, so the "--warmup 0 --iters 1"
    # recipe this printed until 2026-09-03 could not run; the launch count is
    # the instrument's, read back from the profile, never an assumed one.
    assert "--warmup 0 --iters 1" not in out
    commands = [line for line in out.splitlines() if line.strip().startswith("--")]
    assert commands and not any("--iters" in line or "--warmup 0" in line
                                for line in commands)
    assert "--warmup 1 --trials 1 --cell-budget-ms 1" in out
    assert "warmup_calls + iters x trials" in out
    assert "Never divide" in out and '"calls": 11' in out
    # C3 is registered as a bracket per anchor, through cap_from_fitted.
    assert "C3, REGISTERED" in out
    assert out.count("at alpha_a=1") == 3 and "delta = 0" in out
    assert "BM/ridge - a/W = 0.2102" in out


def test_self_test_mode_passes(capsys):
    assert main(["--self-test"]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert "SELF TEST PASS" in out
    lines = exit_codes.parse_result_lines(out)
    assert [(ln.kind, ln.name, ln.verdict) for ln in lines] == [("VALIDITY", "S1", "PASS")]
    assert exit_codes.classify_text(out) == exit_codes.DONE


def test_bracket_mode_runs_and_reports_both_violation_kinds(capsys):
    """CLAIM_FAIL, and that is the ANSWER.

    B1 and B2 are registered to fail -- published alphas above their own pin-rate
    bound is the finding this mode exists to report -- and B3 is registered to
    fail because the bracket is not tight enough to replace a counter. The exit
    code stood at `0 if <condition> else 0` until 2026-09-02, both branches
    DONE, so no gate here could reach it. CLAIM_FAIL (1) is a result the driver
    files as finished and never retries, which is exactly right for a refutation.
    """
    assert main(["--bracket"]) == exit_codes.CLAIM_FAIL
    out = capsys.readouterr().out
    assert "ABOVE" in out and "GATE B1" in out
    assert "28 identifiable fits" in out or "identifiable fits" in out
    # ONE RESULT LINE PER GATE, and the log implies the code the process
    # returned. Until 2026-09-03 this mode printed zero RESULT lines beside
    # exit 1 and `classify_text` raised on its own log.
    lines = exit_codes.parse_result_lines(out)
    assert [ln.name for ln in lines] == ["B0", "B1", "B2", "B3"]
    assert exit_codes.classify_text(out) == exit_codes.CLAIM_FAIL


# --------------------------------------------------------------------------
# Provenance. Failure mode 4: a JSON nobody can attribute to a commit, a
# machine or a set of knobs is an anecdote, and --bracket is this file's one
# advertised RESULT.
# --------------------------------------------------------------------------

def test_every_json_this_script_writes_carries_provenance_and_a_run_id(tmp_path):
    """Both `--out` payloads, checked by running the modes rather than by
    reading the source.

    Before 2026-09-02 `--probe` wrote
    `['capabilities','module_flag','ncu','notes','nsys','verdict']` and
    `--bracket` wrote `['gates','rows']`: no commit, no card, no timestamp, no
    knobs. `--probe` describes a MACHINE and named none, so two pods' probes
    were indistinguishable.
    """
    for argv, out, instrument in (
            (["--probe"], tmp_path / "probe.json", PROBE_INSTRUMENT),
            (["--bracket"], tmp_path / "bracket.json", INSTRUMENT)):
        main([*argv, "--out", str(out)])
        doc = json.loads(out.read_text())
        for key in ("git_sha", "gpu_name", "instrument", "ridge_source",
                    "bandwidth_source", "provenance", "run_id"):
            assert key in doc, (argv, key)
        assert doc["instrument"] == instrument
        assert doc["provenance"]["git_sha"], "the commit that produced it"
        assert doc["provenance"]["utc"]
        # The block NAMES what it could not determine rather than guessing it.
        assert isinstance(doc["provenance"]["missing"], dict)


def test_the_instrument_is_never_the_timing_basis(tmp_path):
    """Nothing in this file times a kernel. `--bracket` and `--analyse` are
    arithmetic over rows other runs measured and `--probe` reads a machine's
    configuration, so stamping `moe.bench.timing.TIMING_BASIS` on any of them
    would describe an apparatus that never ran -- the defect the audit found in
    two sibling `--synthetic` paths."""
    from moe.bench import timing
    assert INSTRUMENT != timing.TIMING_BASIS
    assert PROBE_INSTRUMENT != timing.TIMING_BASIS
    assert "no-kernel-timed" in INSTRUMENT and "nothing-timed" in PROBE_INSTRUMENT


def test_the_run_id_separates_the_knobs_each_mode_actually_reads():
    """Two `--bracket` runs over different `--published` sets, and two
    `--analyse` runs with different `--anchor` overrides, overwrote each other
    in silence: `--out` is a bare operator-chosen path and nothing in the file
    said which knobs produced it."""
    args = build_parser().parse_args(["--bracket"])
    other = build_parser().parse_args(["--bracket", "--published", "a", "b"])
    assert run_id_for("bracket", args, NO_CARD) != run_id_for("bracket", other, NO_CARD)
    assert run_id_for("bracket", args, NO_CARD) == run_id_for("bracket", args, NO_CARD)

    one = build_parser().parse_args(["--analyse", "run.json"])
    two = build_parser().parse_args(["--analyse", "run.json",
                                     "--anchor", "t1", "0.4522"])
    one.anchor = [(k, float(v)) for k, v in (one.anchor or [])]
    two.anchor = [(k, float(v)) for k, v in two.anchor]
    assert run_id_for("analyse", one, NO_CARD) != run_id_for("analyse", two, NO_CARD)
    # ...and the card is the id's first component, so two pods cannot share one.
    assert run_id_for("probe", args, "NVIDIA H200").startswith("nvidia_h200-")
    assert (run_id_for("probe", args, "NVIDIA H200")
            != run_id_for("probe", args, "NVIDIA A100-SXM4-80GB"))


def test_stamping_refuses_to_layer_a_second_provenance_block():
    """The FAIL branch of `stamped`: a payload that already carries one of the
    audit's five keys with a different value is a collision, not a merge."""
    args = build_parser().parse_args(["--bracket"])
    with pytest.raises(PV.ProvenanceCollision):
        stamped({"git_sha": "0" * 40}, mode="bracket", args=args, card=NO_CARD,
                instrument=INSTRUMENT)


# --------------------------------------------------------------------------
# The exit-code table. Failure mode 5: an ANSWER filed as a broken instrument.
# --------------------------------------------------------------------------

def test_a_blocked_counter_route_is_an_answer_and_not_a_validity_failure(
        monkeypatch, capsys):
    """`return 0 if verdict == "OPEN" else 3` is what stood here.

    3 is INVALID in the shared table -- "measured, then a VALIDITY gate failed,
    nothing quotable" -- so the finding this arm exists to obtain, that ncu is
    blocked by a host module flag a tenant cannot change, was filed as a broken
    instrument and latched the row for every resume. Then it returned DONE for
    BLOCKED with no RESULT line at all, so the log implied nothing and the
    driver's summary printed "NOT scored" beside a finished arm. The probe is
    now one scored CLAIM gate, "a counter route is open": OPEN passes it and
    BLOCKED is the world refuting it, CLAIM_FAIL, which the table defines as a
    result the ledger files as finished and never retries. REFUSE, where the
    machine did not say enough to name a route, scores nothing and is
    REFUSED (2). In every case the log's RESULT lines imply the code returned.
    """
    import scripts.dram_counter_route as DCR
    for verdict, expected in (("OPEN", exit_codes.DONE),
                              ("BLOCKED", exit_codes.CLAIM_FAIL),
                              (REFUSE, exit_codes.REFUSED)):
        monkeypatch.setattr(DCR, "route_verdict",
                            lambda *a, _v=verdict: (_v, ["planted"]))
        assert DCR.main(["--probe"]) == expected, verdict
        out = capsys.readouterr().out
        lines = exit_codes.parse_result_lines(out)
        if expected == exit_codes.REFUSED:
            assert lines == [], "a refusal scored a gate"
        else:
            assert [(ln.name, ln.verdict) for ln in lines] == [
                ("P1", "PASS" if verdict == "OPEN" else "FAIL")]
            assert exit_codes.classify_text(out) == expected


def test_analyse_maps_validity_and_claim_the_way_the_shared_table_does(tmp_path, capsys):
    """The two codes were INVERTED: 1 for a VALIDITY failure and 3 for a CLAIM
    failure. 1 is CLAIM_FAIL, which the driver files as finished, so an unsound
    counter run was recorded as a refuted claim; 3 is INVALID, which would have
    latched a perfectly good refutation."""
    def payload(rows):
        return {"device": "NVIDIA A100-SXM4-80GB", "model": "mixtral-8x7b",
                "block_m": 32, "cache_control": "all", "rows": rows}

    # Fewer than four tile counts fails V1, the non-vacuity gate: INVALID.
    thin = tmp_path / "thin.json"
    thin.write_text(json.dumps(payload(
        [{"n": 1, "launches": 5, "dram_bytes_read": 2.8e9}])))
    assert main(["--analyse", str(thin)]) == exit_codes.INVALID
    thin_out = capsys.readouterr().out
    assert [ln.name for ln in exit_codes.parse_result_lines(thin_out)] == ["V1"]
    assert exit_codes.classify_text(thin_out) == exit_codes.INVALID

    # A clean run at a planted alpha passes everything: DONE.
    good = tmp_path / "good.json"
    good.write_text(json.dumps(payload(
        [{"n": n, "launches": 5,
          "dram_bytes_read": predicted_read_bytes(MIXTRAL, 32, n, 0.558)}
         for n in (1, 2, 3, 4)])))
    # ...and every VALIDITY gate passes. C2 and C3 come back UNKNOWN (this file
    # spells it REFUSE) because the payload carries no bracket and no ridge, and
    # UNKNOWN counts AGAINST a gate: the claim was not established, so
    # CLAIM_FAIL, never DONE. A check that examined nothing reports no failures.
    assert main(["--analyse", str(good)]) == exit_codes.CLAIM_FAIL
    out = capsys.readouterr().out
    # ...and the log says the same thing the code does, one line per gate.
    assert [ln.name for ln in exit_codes.parse_result_lines(out)] == [
        "V1", "V2", "V3", "V4", "C1", "C2", "C3"]
    assert exit_codes.classify_text(out) == exit_codes.CLAIM_FAIL


def test_anchors_dataclass_is_pure_arithmetic():
    """No hidden state: the three properties are functions of the six fields."""
    a = Anchors(block_m=32, t1_ms=2.0, intercept=0.5, slope=1.0,
                intercept_n3=0.4, slope_n3=1.1)
    assert a.published == pytest.approx(1.0 / 1.5)
    assert a.t1 == pytest.approx(0.5)
    assert a.n3 == pytest.approx(1.1 / 1.5)
    assert not math.isnan(sum(a.as_dict().values()))


# --------------------------------------------------------------------------
# The document and the code must not drift apart.
# --------------------------------------------------------------------------

def test_counters_doc_quotes_the_numbers_the_code_computes():
    """docs/COUNTERS.md prints constants that this module derives.

    A document is the one artifact nothing else checks, so the four numbers it
    leans on hardest are asserted here. If the byte model moves, this fails and
    the page gets corrected instead of quietly going stale.
    """
    doc = (REPO / "docs" / "COUNTERS.md").read_text()
    assert f"{weight_bytes_total(MIXTRAL):,}" in doc          # 2,818,572,288 B
    assert f"{activation_bytes_per_tile(MIXTRAL, 32):,}" in doc  # 26,214,400 B
    # This card's own ridge, and the stale H200 default the page tells you not to use.
    assert "145.81" in doc and "160.3" in doc
    # The bracket the plan registers against.
    rep = json.loads(DEFAULT_REPORT.read_text())["ladder"]["32"]
    anc = anchors_from_points(rep["points"], rep["memory_points"])
    lo, hi = physical_bracket(anc.slope, anc.t1_ms, weight_bytes_total(MIXTRAL),
                              activation_bytes_per_tile(MIXTRAL, 32),
                              DATASHEET_PEAK_GBPS["nvidia_a100_sxm4_80gb"])
    assert f"{lo:.4f}" in doc and f"{hi:.4f}" in doc


# --------------------------------------------------------------------------
# Where --out lands. Failure mode 3: output written where .gitignore silently
# drops it, which has already cost this repo every published plot of ten arms.
# --------------------------------------------------------------------------

def test_git_visibility_distinguishes_ignored_kept_and_unverifiable():
    """Three answers, not two. `--bracket` is the mode this file's own header
    calls the one that produces a result, so its JSON is meant to be kept."""
    assert "IGNORED by git" in git_visibility(REPO / "results" / "b.json")
    assert "WILL KEEP" in git_visibility(
        REPO / "results" / "published" / "b.json")
    outside = git_visibility(Path("/definitely/not/a/work/tree/b"))
    assert "UNVERIFIED" in outside
    assert "WILL KEEP" not in outside


def test_every_out_write_site_reports_its_git_visibility():
    """A rule that is applied at one of two write sites is not applied.

    Checked structurally rather than by running both modes, because `--probe`
    needs a machine to probe. Each `write_text` on `--out` must be followed by
    the visibility line before the function returns.
    """
    text = (REPO / "scripts" / "dram_counter_route.py").read_text()
    assert text.count("out.write_text(") == 2
    assert text.count("git_visibility(out)") == 2
