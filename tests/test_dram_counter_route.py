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
import yaml

from moe.bench import exit_codes
from moe.bench import provenance as PV
from moe.spec import MODEL_CONFIGS
from scripts.dram_counter_route import (
    A100_REPORT,
    CALL_MARKER,
    COUNTER_ROW_KEYS,
    COUNTER_SCHEMA_TEXT,
    COUNTER_TOP_KEYS,
    DATASHEET_PEAK_GBPS,
    DEFAULT_REPORT,
    FAIL,
    GAPS_SESSION,
    INSTRUMENT,
    NCU_METRICS,
    NO_CARD,
    PASS,
    PROBE_INSTRUMENT,
    REFUSE,
    RUN_INSTRUMENT,
    Anchors,
    CorpusMissing,
    CounterRunRefused,
    activation_bytes_per_tile,
    alpha_from_counters,
    anchor_cap_bracket,
    anchors_from_points,
    bracket_directory,
    build_counter_payload,
    build_parser,
    canned_ncu_csv,
    cap_from_counter,
    card_key,
    contrast_plan,
    corpus_slope,
    discrimination,
    git_visibility,
    main,
    measured_bandwidth_gbps,
    measured_ridge,
    normalise_per_call,
    ols,
    one_run_dir,
    parse_ncu_csv,
    physical_bracket,
    predicted_read_bytes,
    probe_capabilities,
    route_verdict,
    run_id_for,
    score_counter_run,
    stamped,
    sweep_argv,
    weight_bytes_total,
    weight_stream_ms,
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

def test_the_a100_cell_still_reproduces_the_three_anchors():
    """The cell the plan used to register, and the one docs/COUNTERS.md still does.

    0.452 / 0.647 / 0.705 are the numbers the brief quotes. If a republish moves
    them this test says so before docs/COUNTERS.md is quoted at anyone. It is no
    longer the plan's cell: no counter route has ever been open on an A100 this
    study can rent, so the plan moved to the H200 twin on 2026-09-10 and this
    constant stayed behind to keep the page honest.
    """
    assert A100_REPORT.exists(), f"{A100_REPORT} is gone; the page has no cell"
    rep = json.loads(A100_REPORT.read_text())
    ladder = rep["ladder"]["32"]
    anc = anchors_from_points(ladder["points"], ladder["memory_points"])
    assert anc.t1 == pytest.approx(0.452, abs=0.002)
    assert anc.published == pytest.approx(0.647, abs=0.002)
    assert anc.n3 == pytest.approx(0.705, abs=0.002)
    # And the refit must reproduce the published slope, which is what makes the
    # published line the line under test rather than a different one.
    assert anc.slope == pytest.approx(ladder["slope_memory"], rel=1e-9)


def test_the_plan_is_registered_on_the_card_whose_route_is_open():
    """The default cell is the H200 twin, and the H200 is where --probe said OPEN.

    Registering a plan on a card that cannot run it is how this arm spent two
    weeks: `counter_plan` returned P1 PASS on the H200 while the cell it printed
    named nvidia_a100_sxm4_80gb. The default card, the default report and the
    calibration the ridge comes from must now be one card.
    """
    ap = build_parser()
    defaults = ap.parse_args([])
    assert defaults.card == "nvidia_h200"
    assert Path(defaults.report) == DEFAULT_REPORT
    assert "nvidia_h200" in DEFAULT_REPORT.name or "h200" in str(DEFAULT_REPORT)
    assert DEFAULT_REPORT.exists()
    rep = json.loads(DEFAULT_REPORT.read_text())
    assert rep["model"] == "mixtral-8x7b"
    assert rep["fixed"]["GROUP_SIZE_M"] == 16 and rep["fixed"]["BLOCK_SIZE_N"] == 64
    ladder = rep["ladder"][str(defaults.block_m)]
    assert ladder["memory_points"] >= 3, "the default BLOCK_M has no memory branch here"


def test_every_prediction_reads_this_card_s_own_2026_09_10_calibration():
    """The recalibration moved the ridge and the peak; nothing here may be typed.

    `measured_nvidia_h200.yaml` was rewritten on 2026-09-10 (ridge 152.8 ->
    155.93, bf16 peak 668.5 -> 682.09, GEMM clock 1485 -> 1470). A test that
    pins the old figures instead of reading the file is stale by construction,
    so this reads the file and asserts the code returns what the file says.
    """
    cal = yaml.safe_load(
        (REPO / "moe" / "bench" / "hardware" / "measured_nvidia_h200.yaml").read_text())
    ridge, src = measured_ridge("nvidia_h200")
    assert ridge == pytest.approx(
        cal["compute_dense_tflops"]["bf16"] / cal["memory"]["bandwidth_tb_s"], rel=1e-12)
    assert "measured_nvidia_h200.yaml" in src
    gbps, bsrc = measured_bandwidth_gbps("nvidia_h200")
    assert gbps == pytest.approx(cal["memory"]["bandwidth_tb_s"] * 1000.0, rel=1e-12)
    assert cal["detail"]["ceiling_pattern"] in bsrc
    # And the published report this plan registers against carries the STALE
    # ridge, which is exactly why the plan reads the yaml and not the report.
    assert json.loads(DEFAULT_REPORT.read_text())["ridge"] != pytest.approx(ridge, rel=1e-6)


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
    assert "COST, of the plan as extended." in out
    assert "dram__bytes_read.sum" in out
    # The plan must name THIS card's own ridge, computed from the 2026-09-10
    # calibration in the tree, and never the 152.8 the published reports carry.
    ridge, _ = measured_ridge("nvidia_h200")
    assert f"ridge {ridge:.2f} FLOP/byte" in out
    assert "ridge 152.8 " not in out and "ridge 160.3" not in out
    # THE RECIPE RUNS ON THE CURRENT INSTRUMENT. `timing.warm_until` refuses
    # `warmup_ms <= 0` and `--iters` is retired, so the "--warmup 0 --iters 1"
    # recipe this printed until 2026-09-03 could not run; the launch count is
    # the instrument's, read back from the profile, never an assumed one.
    assert "--warmup 0 --iters 1" not in out
    commands = [line for line in out.splitlines() if line.strip().startswith("--")]
    assert commands and not any("--iters" in line or "--warmup 0" in line
                                for line in commands)
    assert "warmup + iters x trials" in out
    assert '"calls": 11' in out and "never by" in out
    # C3 is registered as a bracket per anchor, through cap_from_fitted.
    assert "C3, REGISTERED" in out
    assert out.count("at alpha_a=1") == 3 and "delta = 0" in out
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    bm = build_parser().parse_args([]).block_m
    thresh = bm / ridge - activation_bytes_per_tile(cfg, bm) / weight_bytes_total(cfg)
    assert f"BM/ridge - a/W = {thresh:.4f}" in out


def test_dry_run_prices_the_extended_plan_and_names_its_run_commands(capsys):
    """The plan is five cells now, not one, and the cost line says five.

    It priced one cell over two cache modes and called that the experiment. The
    contrast that decides section 2 needs BLOCK_N=32 and BLOCK_N=128 at one
    BLOCK_M, and statement (3) needs a GROUP_SIZE_M=1 cell, so the invocation
    count and the hours are five times what they were and the line must say so
    rather than quietly costing the old plan.
    """
    assert main(["--dry-run"]) == exit_codes.DONE
    out = capsys.readouterr().out
    tiles = build_parser().parse_args([]).tiles
    assert f"5 cells x {len(tiles)} tile counts x 2 cache modes = {5 * len(tiles) * 2}" in out
    # One --run command per cell, each carrying the cell's own knobs.
    runs = [ln for ln in out.splitlines() if "--run --card" in ln]
    assert len(runs) == 5
    assert "--block-n 32" in out and "--block-n 128" in out
    assert "--group-m 1 --num-stages 3" in out
    # The flush launches are gone from the cost because --run turns the flush
    # off; pricing them while letting their traffic into the answer was the bug.
    assert "L2-flush launches" not in out


def test_dry_run_registers_both_rivals_and_says_which_outcome_means_which(capsys):
    """The separation, and the reading rule, both printed before anything runs.

    The two rivals are 3.86 GB against 2.06 GB per M-tile at BLOCK_M=64 under
    TRAFFIC and identical under TIME. Both numbers are re-derived from the
    committed session ladders, so this asserts the printed figures equal the
    figures `contrast_plan` computes rather than a pair of constants.
    """
    assert main(["--dry-run"]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert "THE CONTRAST, registered." in out
    assert "CONTRAST A" in out and "CONTRAST B" in out
    cells = {c.name: c for c in contrast_plan(64)}
    gbps, _ = measured_bandwidth_gbps("nvidia_h200")
    W = weight_bytes_total(MIXTRAL)
    stream = weight_stream_ms(W, gbps)
    lo = cells["bn32-g16-m64"].traffic_bytes_per_tile(stream, W)
    hi = cells["bn128-g16-m64"].traffic_bytes_per_tile(stream, W)
    assert f"{lo / 1e9:.2f} GB against {hi / 1e9:.2f} GB per M-tile" in out
    assert lo / 1e9 == pytest.approx(3.86, abs=0.01)
    assert hi / 1e9 == pytest.approx(2.06, abs=0.01)
    assert "TIME    predicts the ratio 1.000" in out
    assert "a ratio near 1.000 means it is time" in out
    assert "belongs in the byte model" in out


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
    # `--run` DOES touch the device, and it still is not the timing basis: under
    # --replay-mode kernel every launch is replayed, so the durations it records
    # are replay time and joining them to the unprofiled ladder's wall clock
    # would compare two apparatuses. The string has to say so.
    assert RUN_INSTRUMENT != timing.TIMING_BASIS
    assert "replay-mode-kernel" in RUN_INSTRUMENT
    assert "NOT timing.TIMING_BASIS" in RUN_INSTRUMENT


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
    # The bracket the page registers against. It is the A100 cell, which this
    # module no longer defaults to: the page still describes the A100
    # registration and updating it is a docs/COUNTERS.md edit, filed rather than
    # made here because that file is not this slice's to touch.
    assert "A100-SXM4-80GB" in doc
    rep = json.loads(A100_REPORT.read_text())["ladder"]["32"]
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
    # Three since 2026-09-10: --bracket, --probe and now --run, which is the
    # mode that writes the file every other mode only talks about.
    assert text.count("out.write_text(") == 3
    assert text.count("git_visibility(out)") == 3


# --------------------------------------------------------------------------
# --run's parser. Every one of these is a wrong NUMBER if it is not caught,
# never a crash, which is why the parser is exercised off the GPU and before a
# pod is rented rather than for the first time on a metered box.
# --------------------------------------------------------------------------

PLANTED = {"calls": 11, "read_per_call": 2.84e9, "write_per_call": 1.1e8,
           "hit_pct": 4.2, "ns_per_call": 1.95e6}


def test_parser_reads_a_canned_profile_launch_by_launch():
    launches = parse_ncu_csv(canned_ncu_csv(**PLANTED))
    assert len(launches) == PLANTED["calls"] * 4          # marker + 2 GEMM + reduction
    assert len({ln.launch_id for ln in launches}) == len(launches)
    assert all(set(ln.metrics) == set(NCU_METRICS) for ln in launches)


def test_parser_converts_the_units_ncu_rescaled_rather_than_adding_them():
    """ncu prints whatever unit keeps a number readable, and the prefixes are
    decimal. Adding an Mbyte row to a byte row lands a factor of a million out,
    which fits an affine line as happily as the truth."""
    raw = parse_ncu_csv(canned_ncu_csv(**PLANTED, read_unit="Mbyte"))
    plain = parse_ncu_csv(canned_ncu_csv(**PLANTED, read_unit="byte"))
    a = sum(ln.metrics["dram__bytes_read.sum"] for ln in raw)
    b = sum(ln.metrics["dram__bytes_read.sum"] for ln in plain)
    assert a == pytest.approx(b, rel=1e-12)
    assert b == pytest.approx(PLANTED["read_per_call"] * PLANTED["calls"], rel=1e-12)


def test_parser_refuses_a_unit_it_has_never_been_shown():
    csv_text = canned_ncu_csv(**PLANTED).replace('"byte"', '"kibibyte"', 1)
    with pytest.raises(CounterRunRefused, match="never been shown"):
        parse_ncu_csv(csv_text)


def test_parser_refuses_a_metric_ncu_returned_as_not_available():
    text = canned_ncu_csv(**PLANTED)
    line = [ln for ln in text.splitlines() if "dram__bytes_read.sum" in ln][0]
    text = text.replace(line, line.rsplit(",", 1)[0] + ',"n/a"', 1)
    with pytest.raises(CounterRunRefused, match="never defaulted to 0.0"):
        parse_ncu_csv(text)


def test_parser_refuses_output_with_no_header_and_output_with_no_launch_id():
    with pytest.raises(CounterRunRefused, match="no ncu CSV header"):
        parse_ncu_csv("==PROF== Connected to process 1\nnothing here is a table\n")
    with pytest.raises(CounterRunRefused, match="no 'ID' column"):
        parse_ncu_csv(canned_ncu_csv(**PLANTED).replace('"ID"', '"Index"', 1))


def test_parser_refuses_a_launch_id_that_is_not_unique():
    """Two launches merged into one halves the traffic the count then divides."""
    text = canned_ncu_csv(**PLANTED).splitlines()
    doubled = "\n".join(text + [text[3]])
    with pytest.raises(CounterRunRefused, match="twice"):
        parse_ncu_csv(doubled + "\n")


def test_parser_ignores_metrics_nobody_registered():
    extra = '"1","1","python","k","sm__cycles_elapsed.sum","cycle","5"'
    launches = parse_ncu_csv(canned_ncu_csv(**PLANTED) + extra + "\n")
    assert all("sm__cycles_elapsed.sum" not in ln.metrics for ln in launches)


# --------------------------------------------------------------------------
# The per-call normalisation, which is the whole reason --run is a mode.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("calls", [11, 22, 57, 200])
def test_bytes_are_divided_by_the_call_count_counted_from_the_profile(calls):
    """A synthetic profile with a KNOWN call count, at four counts.

    Every field is per call, so the SAME per-call traffic profiled over 11 calls
    and over 200 must reduce to the same number. Dividing by an assumed 1 would
    scale with `calls` and still be monotone, still be affine, and still pass
    every gate but the n=1 one.
    """
    row = normalise_per_call(parse_ncu_csv(canned_ncu_csv(**{**PLANTED, "calls": calls})),
                             calls_floor=10)
    assert row["calls"] == calls
    assert row["launches"] == calls * 4
    assert row["dram_bytes_read"] == pytest.approx(PLANTED["read_per_call"], rel=1e-12)
    assert row["dram_bytes_write"] == pytest.approx(PLANTED["write_per_call"], rel=1e-12)
    assert row["gpu_time_ns"] == pytest.approx(PLANTED["ns_per_call"], rel=1e-12)
    assert row["gemm_launches_per_call"] == 2.0
    assert sum(row["by_kernel"].values()) == pytest.approx(PLANTED["read_per_call"],
                                                           rel=1e-12)


def test_the_call_count_is_counted_and_not_taken_from_the_caller():
    """The count comes from the launch list, so a profile with more calls in it
    than the caller expected still reduces correctly."""
    row = normalise_per_call(parse_ncu_csv(canned_ncu_csv(**{**PLANTED, "calls": 41})),
                             calls_floor=10)
    assert row["calls"] == 41 and row["calls_floor"] == 10


def test_a_missing_metric_refuses_and_never_becomes_zero():
    with pytest.raises(CounterRunRefused, match="never taken as 0.0"):
        normalise_per_call(
            parse_ncu_csv(canned_ncu_csv(**PLANTED,
                                         drop_metric="dram__bytes_write.sum")),
            calls_floor=10)


def test_a_call_count_at_or_below_the_cells_own_floor_refuses():
    """The instrument runs warmup calls on top of iters x trials, so a count that
    does not EXCEED the floor means the marker is not one-per-call."""
    with pytest.raises(CounterRunRefused, match="must EXCEED the floor"):
        normalise_per_call(parse_ncu_csv(canned_ncu_csv(**{**PLANTED, "calls": 8})),
                           calls_floor=10)
    with pytest.raises(CounterRunRefused, match="must EXCEED the floor"):
        normalise_per_call(parse_ncu_csv(canned_ncu_csv(**{**PLANTED, "calls": 10})),
                           calls_floor=10)


def test_a_profile_with_no_marker_kernel_refuses_and_prints_what_it_saw():
    with pytest.raises(CounterRunRefused) as exc:
        normalise_per_call(
            parse_ncu_csv(canned_ncu_csv(**PLANTED, marker="something_else")),
            calls_floor=10)
    assert CALL_MARKER in str(exc.value)
    assert "something_else" in str(exc.value)     # the names it DID contain


def test_the_l2_hit_rate_is_labelled_as_the_weighted_mean_it_is():
    """A rate cannot be summed and this one is not sector weighted, because the
    sector counts are not among the four registered metrics. Say so in the row."""
    row = normalise_per_call(parse_ncu_csv(canned_ncu_csv(**PLANTED)), calls_floor=10)
    assert row["l2_read_hit_pct"] == pytest.approx(PLANTED["hit_pct"])
    assert row["l2_read_hit_pct_range"] == [PLANTED["hit_pct"], PLANTED["hit_pct"]]
    assert "NOT a sector-weighted rate" in row["l2_read_hit_pct_basis"]


def test_the_self_test_mode_exercises_the_parser_and_reports_both_refusals(capsys):
    """Acceptance: --self-test drives the parser end to end and the two planted
    faults REFUSE. A self-test that only ever feeds the parser good input has
    tested nothing that will happen on a pod."""
    assert main(["--self-test"]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert "THE RUNNER'S PARSER AND ITS PER-CALL DIVISION" in out
    assert "planted MISSING metric on one launch" in out
    assert "planted WRONG call count, 8 against a floor of 10" in out
    assert out.count("REFUSED as required") == 3
    assert "FAIL" not in out and "SELF TEST PASS" in out


# --------------------------------------------------------------------------
# What --run would write, and the recipe it would run.
# --------------------------------------------------------------------------

def test_the_schema_block_and_the_writer_name_the_same_keys():
    """The page and the code, bound in both directions.

    A key in the block and not in the writer is a page describing a file nobody
    writes; a key in the writer and not in the block is a field no reader was
    told about. This module had the first defect already: the block said
    num_stages 3 while the sweep's FIXED said 4.
    """
    for key in COUNTER_TOP_KEYS + COUNTER_ROW_KEYS:
        assert f'"{key}":' in COUNTER_SCHEMA_TEXT, f"{key} is not in the schema block"
    # and the reader's own required set is a subset of what the writer produces
    assert {"device", "model", "block_m", "cache_control", "rows"} <= set(COUNTER_TOP_KEYS)


def test_the_writer_produces_every_key_the_reader_refuses_to_default(tmp_path):
    args = build_parser().parse_args([])
    row = normalise_per_call(parse_ncu_csv(canned_ncu_csv(**PLANTED)), calls_floor=10)
    row["n"] = 1
    payload = build_counter_payload(
        args, ridge=155.93, ridge_source="measured_nvidia_h200.yaml",
        anchors={"published": 0.66}, bracket=(0.62, 0.98), contrast=None, rows=[row])
    assert set(COUNTER_TOP_KEYS) <= set(payload)
    # and --analyse consumes it without raising on a partial run
    gates, _summary = score_counter_run(payload)
    assert [g.number for g in gates][:1] == ["V1"]
    del tmp_path


def test_the_writer_refuses_a_row_the_schema_says_must_carry_a_field():
    args = build_parser().parse_args([])
    row = normalise_per_call(parse_ncu_csv(canned_ncu_csv(**PLANTED)), calls_floor=10)
    row["n"] = 1
    row.pop("l2_read_hit_pct")
    with pytest.raises(CounterRunRefused, match="never written as 0.0"):
        build_counter_payload(args, ridge=155.93, ridge_source="x",
                              anchors={"published": 0.66}, bracket=(0.6, 0.9),
                              contrast=None, rows=[row])


def test_the_wrapped_sweep_turns_the_software_flush_off():
    """The instrument's L2 flush is a 240 MB read and a profiler counts it.

    `moe.bench.timing.L2Flusher` reads four times this card's L2 once per timed
    iteration. Under ncu that is a launch like any other and its DRAM reads land
    in the same total as the kernel under test: at ten iterations, 2.4 GB against
    a 2.82 GB weight read. It is CONSTANT in n, so the slope gate, the
    monotonicity gate and the residual gate all pass and only the n=1 byte-model
    gate would notice, which would be read as the byte model being wrong. ncu's
    own --cache-control sets the cache state under the profiler, which is why
    that is the swept parameter and the software flush is not a second one.
    """
    args = build_parser().parse_args(["--block-m", "64", "--num-stages", "3"])
    argv = sweep_argv(args, 4, Path("/tmp/out"))
    assert "--no-l2-flush" in argv
    assert argv[argv.index("--num-stages") + 1] == "3"
    assert argv[argv.index("--tiles") + 1] == "64"          # exactly one tile height
    assert argv[argv.index("--r-max") + 1] == "256"         # 4 tiles x 64 rows
    assert argv[argv.index("--row-step") + 1] == "256"      # one cell, not a ladder
    assert argv[argv.index("--step-probes") + 1] == "0"


def test_the_run_id_separates_a_cache_mode_and_a_call_marker():
    """Two runs that differ only in the cache mode or the marker are two
    different measurements and must not overwrite each other in silence."""
    base = build_parser().parse_args([])
    other = build_parser().parse_args(["--cache-control", "none"])
    marker = build_parser().parse_args(["--call-marker", "sgl_moe_align_block_size"])
    ids = {run_id_for("run", a, "NVIDIA H200") for a in (base, other, marker)}
    assert len(ids) == 3


def test_run_refuses_before_touching_the_gpu(capsys):
    assert main(["--run"]) == exit_codes.REFUSED
    assert "needs --out" in capsys.readouterr().out
    assert main(["--run", "--out", "/tmp/nothing-will-be-written.json"]) == \
        exit_codes.REFUSED
    assert "no open counter route" in capsys.readouterr().out
    assert not Path("/tmp/nothing-will-be-written.json").exists()


def test_run_is_one_mode_among_the_six(capsys):
    assert main(["--run", "--dry-run"]) == exit_codes.REFUSED
    assert "--run" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The contrast, re-derived from the committed 2026-09-10 corpus.
# --------------------------------------------------------------------------

def test_contrast_reproduces_the_analysis_separation_from_the_committed_cells():
    """3.86 GB against 2.06 GB per M-tile at BLOCK_M=64, re-fitted here.

    These are the figures the 2026-09-10 analysis pre-registers the contrast
    on. Nothing is transcribed: the slopes come out of
    results/published/2026-09-10-.../bn_decomposition/*/cells.csv with the
    synthesis's own estimator (median per tread, OLS on the medians) and the
    denominator out of this card's calibration.
    """
    cells = {c.name: c for c in contrast_plan(64)}
    gbps, _ = measured_bandwidth_gbps("nvidia_h200")
    W = weight_bytes_total(MIXTRAL)
    stream = weight_stream_ms(W, gbps)
    assert stream == pytest.approx(0.6443, abs=5e-4)
    lo, hi = cells["bn32-g16-m64"], cells["bn128-g16-m64"]
    assert lo.w(stream) == pytest.approx(1.368, abs=0.002)
    assert hi.w(stream) == pytest.approx(0.731, abs=0.002)
    assert lo.traffic_bytes_per_tile(stream, W) / 1e9 == pytest.approx(3.86, abs=0.01)
    assert hi.traffic_bytes_per_tile(stream, W) / 1e9 == pytest.approx(2.06, abs=0.01)
    # The discriminator is a RATIO of two measured slopes and carries no
    # bandwidth at all, which is why it survives a recalibration.
    assert lo.slope_ms / hi.slope_ms == pytest.approx(1.870, abs=0.005)


def test_the_contrast_carries_a_group_size_m_1_cell_and_a_matched_partner():
    """Statement (3): the per-M-tile cost is a function of the schedule, so the
    plan cannot be all-G=16. The pair must be matched on everything else."""
    cells = {c.name: c for c in contrast_plan(64)}
    g1, g16 = cells["s3w8g1-m64"], cells["s3w8g16-m64"]
    assert g1.group_m == 1 and g16.group_m == 16
    assert (g1.block_m, g1.block_n, g1.num_stages, g1.num_warps) == \
           (g16.block_m, g16.block_n, g16.num_stages, g16.num_warps)
    assert g1.arm == g16.arm == "occupancy_vs_swizzle"
    gbps, _ = measured_bandwidth_gbps("nvidia_h200")
    stream = weight_stream_ms(weight_bytes_total(MIXTRAL), gbps)
    assert g1.w(stream) == pytest.approx(1.204, abs=0.002)
    assert g16.w(stream) == pytest.approx(0.918, abs=0.002)


def test_the_bn_contrast_is_read_at_the_block_m_it_is_asked_for():
    """Not pinned to 64: the same two rivals exist at BLOCK_M=32 and the plan
    must re-derive them rather than reprint the BLOCK_M=64 pair."""
    at64 = {c.name: c for c in contrast_plan(64)}
    at32 = {c.name: c for c in contrast_plan(32)}
    assert "bn32-g16-m32" in at32 and "bn32-g16-m64" in at64
    assert at32["bn32-g16-m32"].slope_ms != at64["bn32-g16-m64"].slope_ms
    # the schedule pair is BLOCK_M=64 at both, because the occupancy arm ran
    # only that tile height, and it says so by carrying block_m 64.
    assert at32["s3w8g1-m64"].block_m == 64


def test_corpus_slope_refuses_a_ladder_it_cannot_fit():
    run = one_run_dir(GAPS_SESSION / "bn_decomposition")
    with pytest.raises(CorpusMissing, match="a slope needs three"):
        corpus_slope(run / "cells.csv", {"block_n": 999, "block_m": 64})
    with pytest.raises(CorpusMissing, match="no column"):
        corpus_slope(run / "cells.csv", {"not_a_column": 1})
    with pytest.raises(CorpusMissing, match="not in this tree"):
        corpus_slope(run / "nope.csv", {"block_n": 32})


def test_one_run_dir_refuses_an_arm_that_is_not_here(tmp_path):
    with pytest.raises(CorpusMissing, match="has no corpus"):
        one_run_dir(tmp_path / "absent_arm")
    (tmp_path / "two" / "a").mkdir(parents=True)
    (tmp_path / "two" / "b").mkdir(parents=True)
    with pytest.raises(CorpusMissing, match="refusing to pick one"):
        one_run_dir(tmp_path / "two")


def test_one_profiled_tile_count_reduces_to_one_schema_row(tmp_path, monkeypatch):
    """The whole of `--run`'s per-cell path, with ncu and the sweep stubbed.

    This is the code that would otherwise execute for the first time on a
    metered box: build the two command lines, read the log ncu wrote, read the
    row the sweep wrote, take the call floor out of it, and reduce. The stub
    writes a canned profile with 11 marker launches and a cells.csv claiming
    iters 10 and trials 1, so the floor is 10 and the count clears it by the one
    warmup call the instrument always makes.
    """
    import scripts.dram_counter_route as DCR

    def fake_run(argv, timeout=60):
        assert argv[0] == "ncu-stub"
        log = Path(argv[argv.index("--log-file") + 1])
        out_dir = Path(argv[argv.index("--out") + 1])
        assert "--no-l2-flush" in argv
        log.write_text(canned_ncu_csv(**PLANTED))
        run = out_dir / "block_m_crossing" / "someid"
        run.mkdir(parents=True, exist_ok=True)
        (run / "cells.csv").write_text(
            "block_m,tiles_per_expert,ms_p50,iters,trials,status\n"
            "64,4,0.94,10,1,ok\n")
        del timeout
        return 0, "", ""

    monkeypatch.setattr(DCR, "_run", fake_run)
    args = build_parser().parse_args(["--block-m", "64"])
    row = DCR.profile_one_tile_count(args, 4, "ncu-stub", tmp_path)
    assert row["n"] == 4 and row["rows_per_expert"] == 256
    assert row["calls"] == 11 and row["calls_floor"] == 10
    assert row["dram_bytes_read"] == pytest.approx(PLANTED["read_per_call"], rel=1e-12)
    assert row["sweep_iters"] == 10 and row["sweep_trials"] == 1
    assert set(COUNTER_ROW_KEYS) <= set(row)


def test_a_profile_spanning_two_cells_refuses_rather_than_averaging_them(
        tmp_path, monkeypatch):
    """Two ok rows under one profile means the byte total spans two geometries
    and the tile count the row claims is not the only one in it."""
    import scripts.dram_counter_route as DCR

    def fake_run(argv, timeout=60):
        Path(argv[argv.index("--log-file") + 1]).write_text(canned_ncu_csv(**PLANTED))
        run = Path(argv[argv.index("--out") + 1]) / "block_m_crossing" / "id"
        run.mkdir(parents=True, exist_ok=True)
        (run / "cells.csv").write_text(
            "block_m,tiles_per_expert,ms_p50,iters,trials,status\n"
            "64,4,0.94,10,1,ok\n64,8,1.88,10,1,ok\n")
        del timeout
        return 0, "", ""

    monkeypatch.setattr(DCR, "_run", fake_run)
    args = build_parser().parse_args(["--block-m", "64"])
    with pytest.raises(CounterRunRefused, match="spans two geometries"):
        DCR.profile_one_tile_count(args, 4, "ncu-stub", tmp_path)


def test_an_ncu_that_wrote_no_log_refuses_with_its_own_stderr(tmp_path, monkeypatch):
    import scripts.dram_counter_route as DCR
    monkeypatch.setattr(DCR, "_run",
                        lambda argv, timeout=60: (1, "", "ERR_NVGPUCTRPERM"))
    args = build_parser().parse_args(["--block-m", "64"])
    with pytest.raises(CounterRunRefused, match="ERR_NVGPUCTRPERM"):
        DCR.profile_one_tile_count(args, 4, "ncu-stub", tmp_path)


def test_the_stamped_contrast_matches_the_cell_on_its_pipeline_depth_too(
        tmp_path, monkeypatch):
    """The hinge and the schedule pair's G=16 cell are one geometry at two
    depths, so a match on geometry alone stamps one arm's prediction on the
    other arm's cell. Checked by asking for depth 3 and depth 4 at the same
    BLOCK_M, BLOCK_N and GROUP_SIZE_M and requiring two different predictions.
    """
    cells = contrast_plan(64)
    hinge = next(c for c in cells if c.name == "bn64-g16-m64")
    occ = next(c for c in cells if c.name == "s3w8g16-m64")
    assert (hinge.block_m, hinge.block_n, hinge.group_m) == \
           (occ.block_m, occ.block_n, occ.group_m)
    assert hinge.num_stages != occ.num_stages
    assert hinge.slope_ms != occ.slope_ms
    del tmp_path, monkeypatch
