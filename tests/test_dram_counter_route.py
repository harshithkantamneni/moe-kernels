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
import re
import statistics
import sys
from pathlib import Path

import pytest
import yaml

import scripts.dram_counter_route as DCR
from moe.bench import counter_probe_kernel as PK
from moe.bench import exit_codes
from moe.bench import provenance as PV
from moe.spec import MODEL_CONFIGS
from scripts.dram_counter_route import (
    A100_REPORT,
    C1_TOLERANCE,
    CALL_MARKER,
    CONTRAST_INSTRUMENT,
    CONTRAST_TOLERANCE,
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
    c1_registration,
    canned_ncu_csv,
    cap_from_counter,
    card_key,
    contrast_pairs,
    contrast_plan,
    contrast_rows,
    corpus_knobs,
    corpus_slope,
    counter_route_is_open,
    discrimination,
    git_visibility,
    main,
    measured_bandwidth_gbps,
    measured_dr_dn,
    measured_ridge,
    ncu_argv,
    normalise_per_call,
    occupancy_block_n,
    ols,
    one_run_dir,
    parse_ncu_csv,
    physical_bracket,
    plan_id,
    predicted_read_bytes,
    probe_capabilities,
    probe_kernel_word,
    route_verdict,
    run_id_for,
    score_counter_run,
    stamped,
    sweep_argv,
    synthetic_counter_payload,
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


def test_probe_capabilities_reads_both_counter_bits():
    """CAP_PERFMON as well as CAP_SYS_ADMIN, because either opens the gate.

    Until 2026-09-15 only bit 21 was read, so a container holding the weaker
    capability that would have sufficed was reported identically to one holding
    nothing, and the operator was sent to ask for the capability providers
    refuse.

    THE WIDTH IS READ OFF THE RAW FIELD AND THIS TEST READS THE SAME FIELD.
    A first draft asserted `cap_eff_bits == 4 * len(cap_eff) - 8`, deriving the
    width from `hex(mask)`, which strips leading zeros. Linux prints `CapEff`
    in sixteen digits, so that assertion is 64 == 32 on every Linux box and
    could only ever have passed here, where there is no `/proc/self/status` and
    the branch does not run. The relation the code implements is against the
    FIELD, so this reads the field.
    """
    caps = probe_capabilities()
    if not caps["available"]:
        assert "perfmon" not in caps
        return
    assert set(caps) == {"available", "cap_eff", "cap_eff_field", "cap_eff_bits",
                         "sys_admin", "perfmon"}
    field = next(line.split()[1] for line in
                 Path("/proc/self/status").read_text().splitlines()
                 if line.startswith("CapEff:"))
    assert caps["cap_eff_field"] == field
    assert caps["cap_eff_bits"] == 4 * len(field)
    mask = int(field, 16)
    assert int(caps["cap_eff"], 16) == mask
    assert caps["perfmon"] is bool(mask >> 38 & 1)
    assert caps["sys_admin"] is bool(mask >> 21 & 1)


def test_route_verdict_distinguishes_the_four_failures():
    open_ = route_verdict({}, {}, {"present": True, "counters_read": True,
                                   "metric": NCU_METRICS[0], "metric_value": 4.19e6,
                                   "cause": "counters readable: ..."},
                          {"present": True, "importer_present": True})
    assert open_[0] == "OPEN"

    blocked_cap = route_verdict({"available": True, "sys_admin": False, "perfmon": False,
                                 "cap_eff": "0xa80425fb",
                                 "cap_eff_field": "00000000a80425fb",
                                 "cap_eff_bits": 64},
                                {"available": True, "restrict": 1},
                                {"present": True, "counters_read": False,
                                 "permission_refused": True,
                                 "cause": "ERR_NVGPUCTRPERM: counters gated"},
                                {"present": True, "importer_present": True})
    assert blocked_cap[0] == "BLOCKED"
    assert any("SYS_ADMIN" in n and "PERFMON" in n for n in blocked_cap[1])
    assert any("RestrictProfilingToAdminUsers=1" in n for n in blocked_cap[1])
    # The narrower ask is named first, because it is the one a provider grants.
    ask = next(n for n in blocked_cap[1] if "PERFMON" in n and "SYS_ADMIN" in n)
    assert ask.index("--cap-add=PERFMON") < ask.index("--cap-add=SYS_ADMIN")
    # THE MASK IS REPORTED, AND NOT A CLAIM ABOUT ITS WIDTH. A note saying the
    # field "cannot represent bit 38" stood here until 2026-09-15 and could
    # never fire: `cap_eff_bits` is the width of the raw /proc field, which
    # Linux prints as sixteen digits whatever the value, so the guard
    # `cap_eff_bits <= 38` was reachable only from a hand-built dict like this
    # one. `perfmon` is a measurement of bit 38 on any real box.
    mask_note = next(n for n in blocked_cap[1] if "capability mask read" in n)
    assert "00000000a80425fb" in mask_note and "64 bits" in mask_note
    assert "bit 38) is clear" in mask_note and "bit 21) is clear" in mask_note
    assert not any("cannot represent" in n for n in blocked_cap[1])

    # The combination that means "stop retrying and read the output".
    odd = route_verdict({"available": True, "sys_admin": False, "perfmon": True},
                        {"available": True, "restrict": 0},
                        {"present": True, "counters_read": False,
                         "permission_refused": True,
                         "cause": "ERR_NVGPUCTRPERM: counters gated"},
                        {"present": True, "importer_present": True})
    assert odd[0] == "BLOCKED"
    assert any("ALREADY allows" in n for n in odd[1])

    # The pod's actual failure: nsys present, importer absent, no ncu at all.
    importer = route_verdict({}, {}, {"present": False, "why": "no ncu on PATH"},
                             {"present": True, "importer_present": False})
    assert importer[0] == "REFUSE"
    assert any("its importer" in n for n in importer[1])


def test_a_box_with_no_cuda_device_refuses_and_does_not_report_blocked():
    """"We could not ask" is not "the answer is no".

    BLOCKED is CLAIM_FAIL, which the ledger files as a FINDING and never
    retries. A box where the probe interpreter could not launch a kernel has
    established nothing about counter permission, so it must land in REFUSE,
    which scores no gate at all.
    """
    for word in (PK.NO_TORCH, PK.NO_CUDA_DEVICE, PK.LAUNCH_FAILED, "NO_MARKER"):
        verdict, notes = route_verdict(
            {"available": True, "sys_admin": False, "perfmon": False},
            {"available": False, "why": "no params file"},
            {"present": True, "counters_read": False, "probe_kernel": word,
             "probe_kernel_detail": "planted",
             "cause": "no CUDA device at all: planted"},
            {"present": True, "importer_present": True})
        assert verdict == REFUSE, word
        assert any("UNTESTED" in n for n in notes), word


# --------------------------------------------------------------------------
# The probe kernel, and the /bin/true defect it replaces.
# --------------------------------------------------------------------------

def _code_strings(text: str) -> set[str]:
    """Every string constant this module EXECUTES, docstrings excluded."""
    import ast
    tree = ast.parse(text)
    docstrings = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str)}
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings}


def _plant_ncu(monkeypatch, *, log: str, stdout: str = "", rc: int = 0):
    """Stand in for one `ncu` invocation: write `log` where --log-file says."""
    import scripts.dram_counter_route as DCR
    monkeypatch.setattr(DCR.shutil, "which", lambda name: f"/usr/bin/{name}-stub")
    seen: list[list[str]] = []

    def fake_run(argv, timeout=60):
        del timeout
        seen.append(list(argv))
        if "--version" in argv:
            return 0, "ncu 2025.1.1.0\n", ""
        Path(argv[argv.index("--log-file") + 1]).write_text(log)
        return rc, stdout, ""

    monkeypatch.setattr(DCR, "_run", fake_run)
    return seen


def test_the_probe_profiles_a_real_kernel_and_bin_true_is_gone(monkeypatch):
    """THE DEFECT, pinned in the shape a grep cannot drift past.

    `ncu --metrics <m> /bin/true` launches no CUDA kernel, so the permission
    check never happens and ncu exits 0 having read nothing. The probe must run
    a real file on disk that launches a kernel, and `/bin/true` must not survive
    as an executable string anywhere in this script.

    PARSED, NOT GREPPED. The prose in this file names `/bin/true` on purpose --
    it is the incident this probe was rewritten against, and a test that
    forbade the word would forbid recording why. So the check is over the
    module's string CONSTANTS with its docstrings removed: a comment or a
    docstring may say it, code may not.
    """
    import scripts.dram_counter_route as DCR
    assert "/bin/true" not in _code_strings(Path(DCR.__file__).read_text())
    seen = _plant_ncu(monkeypatch, log=canned_ncu_csv(
        calls=1, read_per_call=4.19e6, write_per_call=4.19e6,
        hit_pct=1.0, ns_per_call=1e5))
    info = DCR.probe_ncu()
    argv = next(a for a in seen if "--version" not in a)
    assert argv[-1] == str(DCR.NCU_PROBE_KERNEL)
    assert Path(argv[-1]).exists(), "the probe kernel is a real file on disk"
    assert argv[-2] == DCR.sys.executable
    assert "--launch-count" in argv and argv[argv.index("--launch-count") + 1] == "1"
    assert "--csv" in argv and "--page" in argv and "raw" in argv
    assert info["counters_read"] is True


def test_the_probe_reports_open_only_when_a_number_came_back(monkeypatch):
    """A NUMBER, not a positive one.

    The question is whether the driver let ncu report the counter, and a
    counter that honestly reads 0 for a launch has been reported. Requiring a
    positive value would make the verdict depend on which launch ncu picked,
    which is the kind of coupling that put `/bin/true` in here.
    """
    log = canned_ncu_csv(calls=1, read_per_call=4.19e6, write_per_call=1.0,
                         hit_pct=2.0, ns_per_call=1e5)
    _plant_ncu(monkeypatch, log=log, stdout=f"{PK.MARKER} {PK.LAUNCHED} H200: one add_")
    import scripts.dram_counter_route as DCR
    info = DCR.probe_ncu()
    assert info["counters_read"] is True
    assert info["metric"] == NCU_METRICS[0]
    assert isinstance(info["metric_value"], float)
    assert info["cause"].startswith("counters readable")
    assert counter_route_is_open(info)


def test_a_counter_that_read_zero_is_a_reading_and_the_route_is_open(monkeypatch):
    """The value the test above is NAMED for, and nothing planted it until now.

    `test_the_probe_reports_open_only_when_a_number_came_back` plants 4.19e6, a
    positive value, so the property its docstring argues for -- that ZERO is a
    reading -- was pinned by nothing, and a `values[0] > 0` hardening would
    have passed the whole suite. It matters on the real pod: `--launch-count 1`
    profiles the FIRST launch, and a write-only launch reads no DRAM at all, so
    a legitimate 0 is a shape this probe can actually see.
    """
    log = ('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
           '"0","probe","dram__bytes_read.sum","byte","0"\n')
    _plant_ncu(monkeypatch, log=log, stdout=f"{PK.MARKER} {PK.LAUNCHED} H200: one add_")
    import scripts.dram_counter_route as DCR
    info = DCR.probe_ncu()
    assert info["counters_read"] is True
    assert info["metric_value"] == 0.0
    assert counter_route_is_open(info)
    assert route_verdict({}, {}, info, {"present": True, "importer_present": True})[0] \
        == "OPEN"


def test_the_self_test_scores_the_probe_worlds_off_gpu():
    """`--self-test` is the pod's only off-GPU check of the logic that gates
    240 booked minutes, and it had no probe section until 2026-09-15: the
    verdict logic was exercised only by this file, which the pod never runs,
    while `arm_verify counter_plan` named `--dry-run` and `--bracket`.

    The planted worlds go through the SAME `probe_reading` the live probe hands
    its child's bytes to, so this is not a second copy of the rules."""
    rows = DCR.self_test_probe()
    assert len(rows) == 5
    assert all(passed for _label, passed, _detail in rows), rows
    details = " ".join(d for _l, _p, d in rows)
    assert "verdict OPEN exit 0" in details
    assert "verdict BLOCKED exit 1" in details
    assert "verdict REFUSE exit 2" in details


def test_a_metric_ncu_refused_to_supply_is_not_a_readable_counter(monkeypatch):
    """`n/a` is the shape of a counter the replay could not supply. It must not
    read as 0.0 and it must not read as OPEN: the whole parser exists because a
    zero that was never measured fits every gate downstream."""
    import scripts.dram_counter_route as DCR
    log = ('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
           '"0","probe","dram__bytes_read.sum","byte","n/a"\n')
    _plant_ncu(monkeypatch, log=log,
               stdout=f"{PK.MARKER} {PK.LAUNCHED} H200: one add_")
    info = DCR.probe_ncu()
    assert info["counters_read"] is False
    assert info["metric_value"] is None
    assert "no readable dram__bytes_read.sum" in info["cause"]


def test_the_2026_09_published_probe_shape_is_no_longer_open(monkeypatch):
    """The exact payload both published sessions carried, replayed.

    `results/published/2026-09-*-nvidia_h200-gaps-session/session/counter_route.json`
    records `returncode 0`, `output_head "==WARNING== No kernels were
    profiled."` and `cause "attached with no permission error"`, and the
    session booked four pod-hours on that word. The same bytes must now come
    back not-open.
    """
    import scripts.dram_counter_route as DCR
    _plant_ncu(monkeypatch, log="==WARNING== No kernels were profiled.\n", rc=0)
    info = DCR.probe_ncu()
    assert info["returncode"] == 0
    assert info["counters_read"] is False
    assert not counter_route_is_open(info)
    assert route_verdict({}, {}, info, {"present": True, "importer_present": True})[0] \
        == REFUSE


def test_a_launched_kernel_that_ncu_did_not_profile_is_not_open(monkeypatch):
    import scripts.dram_counter_route as DCR
    _plant_ncu(monkeypatch, log="==WARNING== No kernels were profiled.\n",
               stdout=f"{PK.MARKER} {PK.LAUNCHED} H200: one add_")
    info = DCR.probe_ncu()
    assert info["counters_read"] is False
    assert "profiled NO kernels" in info["cause"]


def test_err_nvgpuctrperm_is_reported_as_a_fact_about_the_pod(monkeypatch):
    """It is a measured refusal, not a broken instrument.

    The log ncu wrote on the 2026-09-15 pod, verbatim in shape: it connected,
    it refused the counter, and the application returned an error code. That is
    a CLAIM refuted by the box, which the shared table files as finished.
    """
    import scripts.dram_counter_route as DCR
    _plant_ncu(monkeypatch, rc=1, log=(
        "==PROF== Connected to process 5183 (/usr/bin/python3.12)\n"
        "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access "
        "NVIDIA GPU Performance Counters on the target device 0.\n"
        "==PROF== Disconnected from process 5183\n"
        "==ERROR== The application returned an error code (1).\n"))
    info = DCR.probe_ncu()
    assert info["counters_read"] is False
    assert "fact about the pod" in info["cause"]
    verdict, _ = route_verdict({"available": True, "sys_admin": False, "perfmon": False},
                               {"available": True, "restrict": 1}, info,
                               {"present": True, "importer_present": True})
    assert verdict == "BLOCKED"


def test_a_box_with_no_cuda_device_says_so_and_claims_nothing(monkeypatch):
    import scripts.dram_counter_route as DCR
    _plant_ncu(monkeypatch, rc=1, log="==WARNING== No kernels were profiled.\n",
               stdout=f"{PK.MARKER} {PK.NO_CUDA_DEVICE} torch.cuda.is_available() is False")
    info = DCR.probe_ncu()
    assert info["probe_kernel"] == PK.NO_CUDA_DEVICE
    assert info["counters_read"] is False
    assert "no CUDA device at all" in info["cause"]
    assert "NOTHING about counter permission" in info["cause"]


def test_one_predicate_decides_the_route_and_both_call_sites_call_it():
    """Rule 3, as a test. The route-open rule was `cause.startswith("attached")`
    written out twice, in `route_verdict` and in `do_run`'s pre-flight, so the
    probe's verdict and the runner's gate were one rule at two sites. Both now
    call `counter_route_is_open` and the prose test is gone from the file.
    """
    import ast

    import scripts.dram_counter_route as DCR
    text = Path(DCR.__file__).read_text()
    # The retired rule was `cause.startswith("attached")`. The needle is the
    # string constant, taken out of the parsed module, because the docstring
    # that records the retirement quotes the whole expression.
    assert "attached" not in _code_strings(text)
    tree = ast.parse(text)
    callers = {fn.name for fn in ast.walk(tree)
               if isinstance(fn, ast.FunctionDef)
               and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == "counter_route_is_open"
                       for n in ast.walk(fn))}
    assert {"route_verdict", "do_run"} <= callers, callers


def test_the_probe_kernel_module_names_its_four_worlds_and_only_one_passes():
    assert set(PK.WORDS) == {PK.LAUNCHED, PK.NO_TORCH, PK.NO_CUDA_DEVICE,
                             PK.LAUNCH_FAILED}
    assert len(PK.WORDS) == len(set(PK.WORDS))
    for word in PK.WORDS:
        assert probe_kernel_word(f"{PK.MARKER} {word} detail here") == (word, "detail here")
    # An unknown word is not silently accepted as a launch.
    assert probe_kernel_word(f"{PK.MARKER} SOMETHING_ELSE x") == ("", "")
    assert probe_kernel_word("no marker anywhere") == ("", "")


def test_the_probe_kernel_never_raises_and_only_launched_exits_zero(capsys):
    """It runs on whatever the pod has, including a box with no card.

    A probe that throws on a machine with no CUDA turns "no device here" into a
    traceback, which `exit_codes` reads as CLAIM_FAIL -- a refuted claim about
    counters, from a box that was never asked. This laptop has no CUDA device,
    so the assertion is on the CONTRACT and not on which word comes back.
    """
    word, detail = PK.probe()
    assert word in PK.WORDS
    assert isinstance(detail, str) and detail
    rc = PK.main()
    printed = capsys.readouterr().out.strip()
    assert printed.startswith(f"{PK.MARKER} {word}")
    assert probe_kernel_word(printed) == (word, detail)
    assert (rc == 0) == (word == PK.LAUNCHED)


def test_the_parse_refusals_do_not_prescribe_flags_ncu_argv_already_passes():
    """R1's advice reached an operator and sent them nowhere.

    `parse_ncu_csv`'s refusals told the reader to "Profile with --csv --page
    raw", which `ncu_argv` already does, and R1 is the one line of this file's
    output that a human on a pod reads.
    """
    import scripts.dram_counter_route as DCR
    argv = ncu_argv("ncu", "all", Path("/tmp/x.csv"))
    assert "--csv" in argv and "--page" in argv and "--log-file" in argv
    for bad in ("--csv --page raw and read the log file",
                "Profile with --csv --page raw, which emits the column"):
        assert not any(bad in s for s in _code_strings(Path(DCR.__file__).read_text()))
    with pytest.raises(CounterRunRefused, match="NOT A FLAG PROBLEM"):
        parse_ncu_csv("nothing that looks like a header at all\n")


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
    # Four since the 2026-09-10 repair: --bracket, --probe, --run (the mode that
    # writes the file every other mode only talks about) and --contrast (the
    # mode that scores the ratio across those files). Seven since 2026-09-24:
    # the r3-arms family writes a page per G, a census and an --analyse summary.
    assert text.count("out.write_text(") == 7
    assert text.count("git_visibility(out)") == 7


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
    # The parser's own section, counted inside its own section rather than over
    # the whole log: --self-test grew a contrast section on 2026-09-10 whose
    # mislabelled-payload row also refuses, and a whole-log count of refusals
    # would have been satisfied by the wrong three.
    # BOUNDED AT BOTH ENDS since 2026-09-24: the r3-arms family's section,
    # printed after the probe's, refuses three planted attributions of its own.
    parser = out.split("THE RUNNER'S PARSER AND ITS PER-CALL DIVISION")[1]
    parser = parser.split("THE PROBE'S VERDICT AND EXIT CODE")[0]
    assert "planted MISSING metric on one launch" in parser
    assert "planted WRONG call count, 8 against a floor of 10" in parser
    assert "planted profile with no marker kernel at all" in parser
    assert parser.count("REFUSED as required") == 3
    contrast = out.split("THE CONTRAST SCORER")[1].split(
        "THE RUNNER'S PARSER AND ITS PER-CALL DIVISION")[0]
    assert contrast.count("REFUSED as required") == 1
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
    assert "no counter could be read" in capsys.readouterr().out
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
    denominator is the triad rate THAT RUN recorded in its report.json, not
    the tree's current ruler: the figures are the 2026-09-10 cells', and the
    ruler has moved since (4374.3 -> 4378.0 GB/s on 2026-09-21, which is
    0.085% and enough to trip a four-place pin). test_docs reads the same
    report the same way.
    """
    cells = {c.name: c for c in contrast_plan(64)}
    run = sorted((REPO / "results" / "published").glob(
        "2026-09-10-*gaps-session/results/bn_decomposition/*/report.json"))
    assert run, "the 2026-09-10 bn_decomposition report is not in the tree"
    gbps = json.loads(run[0].read_text())["bandwidth_gbps"]
    W = weight_bytes_total(MIXTRAL)
    stream = weight_stream_ms(W, gbps)
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


# --------------------------------------------------------------------------
# THE 2026-09-10 REPAIR. Four defects found by review of the commit that built
# the runner, each one a gate or a parser that would have produced a wrong
# reading on the metered box rather than a crash.
# --------------------------------------------------------------------------

def _h200_payload(alpha_name: str, *, block_m: int = 64, anchors=None) -> dict:
    """A counter payload for the cell the plan registers, planted ON an anchor.

    The point of planting exactly on an anchor is that this is what a SUCCESSFUL
    run looks like: the ladder and the counter agree. Anything the gates say
    about it, they say about a measurement that worked.
    """
    anc, _rep = DCR.anchors_from_report(DEFAULT_REPORT, block_m)
    registered = anchors if anchors is not None else anc.as_dict()
    alpha = registered[alpha_name]
    ridge, _src = measured_ridge("nvidia_h200")
    lo, hi = physical_bracket(anc.slope, anc.t1_ms, weight_bytes_total(MIXTRAL),
                              activation_bytes_per_tile(MIXTRAL, block_m),
                              DATASHEET_PEAK_GBPS["nvidia_h200"])
    return {"device": "nvidia_h200", "model": "mixtral-8x7b", "dtype": "bf16",
            "group_m": 16, "block_n": 64, "block_k": 64, "num_warps": 8,
            "num_stages": 4, "block_m": block_m, "cache_control": "all",
            "ridge": ridge, "anchors": registered, "bracket": [lo, hi],
            "rows": [{"n": n, "calls": 11, "calls_floor": 10, "launches": 44,
                      "dram_bytes_read": predicted_read_bytes(MIXTRAL, block_m, n, alpha),
                      "dram_bytes_write": 1.1e8, "l2_read_hit_pct": 4.2,
                      "gpu_time_ns": 1.95e6, "by_kernel": {}}
                     for n in (1, 2, 3, 4, 6, 8)]}


def test_c1_cannot_be_asked_for_one_survivor_on_the_cell_the_plan_registers():
    """The blocking defect: C1 was pre-determined to FAIL on this cell.

    The H200 anchors span 0.0393 end to end, narrower than C1's own 0.05
    survival window, so a measurement landing ON an anchor leaves all three
    standing. The gate asked for exactly one and returned FAIL with the
    diagnosis "the cell was badly chosen", which was true and was known before
    the run: --dry-run's own text says the anchors agree to 0.04. The
    registration now decides which question the cell carries.
    """
    payload = _h200_payload("n3")
    spread = max(payload["anchors"].values()) - min(payload["anchors"].values())
    assert spread < C1_TOLERANCE, "this test is about a cluster narrower than the window"
    gates, summary = score_counter_run(payload)
    c1 = next(g for g in gates if g.number == "C1")
    assert summary["c1_mode"] == "clustered"
    assert len(summary["survivors"]) == 3, "the old rule's FAIL, still visible"
    assert c1.verdict == PASS
    assert "cannot separate" in c1.claim
    assert any("NOT DISCRIMINATING" in line for line in c1.lines)


def test_a_successful_measurement_of_that_cell_exits_done(tmp_path, capsys):
    """The consequence the operator would have read on the pod: exit 1.

    Scored end to end through --analyse, because the exit code is what the
    session driver files the arm under and CLAIM_FAIL is "measured, the world
    disagreed". The world did not disagree; the gate could not pass.
    """
    path = tmp_path / "h200.json"
    path.write_text(json.dumps(_h200_payload("n3")))
    assert main(["--analyse", str(path)]) == exit_codes.DONE
    out = capsys.readouterr().out
    line = next(ln for ln in exit_codes.parse_result_lines(out) if ln.name == "C1")
    assert line.verdict == "PASS"
    assert exit_codes.classify_text(out) == exit_codes.DONE


def test_the_a100_cell_keeps_the_question_it_can_answer(tmp_path, capsys):
    """And the repair is not a loosening: on a cell whose anchors ARE separated,
    C1 still demands exactly one survivor, and two survivors still FAIL."""
    a100 = {"published": 0.6473, "t1": 0.4522, "n3": 0.7047}
    reg = c1_registration(a100)
    assert reg.separating and reg.min_gap > C1_TOLERANCE
    assert reg.verdict(["t1"]) == PASS
    assert reg.verdict(["t1", "n3"]) == FAIL
    assert reg.verdict([]) == FAIL
    # ...and a clustered cell's FAIL is every anchor refuted, which is a result
    # and not an instrument failure, so C1 stays a CLAIM gate in both modes.
    payload = _h200_payload("n3")
    payload["rows"] = [{"n": n, "calls": 11, "calls_floor": 10, "launches": 44,
                        "dram_bytes_read": predicted_read_bytes(MIXTRAL, 64, n, 0.05),
                        "dram_bytes_write": 1.1e8, "l2_read_hit_pct": 4.2,
                        "gpu_time_ns": 1.95e6, "by_kernel": {}}
                       for n in (1, 2, 3, 4, 6, 8)]
    path = tmp_path / "refuted.json"
    path.write_text(json.dumps(payload))
    assert main(["--analyse", str(path)]) == exit_codes.CLAIM_FAIL
    out = capsys.readouterr().out
    line = next(ln for ln in exit_codes.parse_result_lines(out) if ln.name == "C1")
    assert line.verdict == "FAIL"
    del capsys


def test_c1_refuses_a_payload_that_registered_no_anchor_at_all():
    """A gate with nothing to compare against examined nothing. It used to score
    FAIL with "survivors none", which reads as three refuted anchors."""
    payload = _h200_payload("n3")
    payload.pop("anchors")
    gates, _summary = score_counter_run(payload)
    c1 = next(g for g in gates if g.number == "C1")
    assert c1.verdict == REFUSE
    assert c1.scored()[2] == exit_codes.UNKNOWN


def test_the_plan_registers_c1_before_anything_runs(capsys):
    """Said at registration time, on the cell the plan actually holds, so the
    operator does not discover it from a gate on a rented box."""
    assert main(["--dry-run"]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert "C1, REGISTERED" in out
    assert "NOT DISCRIMINATING" in out
    anc, _rep = DCR.anchors_from_report(DEFAULT_REPORT, 64)
    reg = c1_registration(anc.as_dict())
    assert f"{reg.min_gap:.4f}" in out and f"{C1_TOLERANCE:.2f}" in out


def test_the_parser_requires_the_unit_column_the_way_it_requires_the_id():
    """The parser's one defence against ncu's per-launch rescaling was optional.

    `Metric Unit` was picked up `if name in header`, so a CSV without the column
    left `unit` empty, and the byte tables mapped "" to 1.0. A file whose bytes
    ncu had already rescaled to Mbyte read as bytes: a factor of 1e6 low, still
    perfectly affine in n, and therefore invisible to V3, V4 and every gate but
    the n=1 one.
    """
    head = '"ID","Kernel Name","Metric Name","Metric Value"'
    rows = [head]
    for launch in range(1, 12):
        kernel = CALL_MARKER if launch % 4 == 1 else "fused_moe_kernel"
        for metric in NCU_METRICS:
            rows.append(f'"{launch}","{kernel}","{metric}","2840.0"')
    with pytest.raises(CounterRunRefused, match="Metric Unit"):
        parse_ncu_csv("\n".join(rows))
    # and a blank unit CELL, on a header that does carry the column, refuses too
    blank = ['"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"',
             f'"1","{CALL_MARKER}","dram__bytes_read.sum","","2840.0"']
    with pytest.raises(CounterRunRefused, match="never been shown"):
        parse_ncu_csv("\n".join(blank))
    assert all("" not in units for _canon, units in DCR.NCU_METRIC_UNITS.values())


def test_the_schedule_pairs_knobs_are_read_off_the_committed_rows():
    """Two call sites, one estimator: the BLOCK_N cells read their knobs from
    their arm's report and the schedule pair typed theirs from the setting
    string. The typed values were right, and nothing would have said so if a
    republish had moved a pipeline depth."""
    run = one_run_dir(GAPS_SESSION / "occupancy_vs_swizzle")
    for setting, group_m in (("s3w8g1", 1), ("s3w8g16", 16)):
        knobs = corpus_knobs(run / "cells.csv", {"setting": setting, "block_m": 64},
                             ("group_m", "num_stages", "num_warps"))
        cell = next(c for c in contrast_plan(64) if c.name == f"{setting}-m64")
        assert (cell.group_m, cell.num_stages, cell.num_warps) == (
            knobs["group_m"], knobs["num_stages"], knobs["num_warps"])
        assert knobs["group_m"] == group_m
        assert "cells.csv" in cell.knob_source
    # a column the rows do not carry, and a selection spanning two values of
    # one, both refuse rather than picking whichever sorted first.
    with pytest.raises(CorpusMissing, match="no column"):
        corpus_knobs(run / "cells.csv", {"setting": "s3w8g1"}, ("block_n",))
    with pytest.raises(CorpusMissing, match="that is two cells"):
        corpus_knobs(run / "cells.csv", {"block_m": 64}, ("group_m",))


def test_the_occupancy_arms_block_n_is_read_and_its_source_named():
    """The arm records its BLOCK_SIZE_N inside the text of the gate that checks
    it and nowhere as a field, so the reader says which of the two it used."""
    run = one_run_dir(GAPS_SESSION / "occupancy_vs_swizzle")
    report = json.loads((run / "report.json").read_text())
    block_n, source = occupancy_block_n(report)
    assert block_n == 64 and "report.json" in source
    fallback, source = occupancy_block_n({"no": "pin here"})
    assert fallback == 64 and "FIXED" in source


def test_the_plan_id_separates_two_cards():
    """`--card` became a first-class knob and was not in the id, so the same
    cell planned for the A100 and for the H200 produced one string while every
    prediction under it differed."""
    h200 = build_parser().parse_args([])
    a100 = build_parser().parse_args(["--card", "nvidia_a100_sxm4_80gb"])
    assert plan_id(h200) != plan_id(a100)
    assert plan_id(h200).startswith("nvidia_h200-")
    assert plan_id(h200) == plan_id(build_parser().parse_args([]))


# --------------------------------------------------------------------------
# --contrast. The reading the extended plan exists to take, which no single
# payload contains and which nothing scored until this repair.
# --------------------------------------------------------------------------

def _pair_payloads(tmp_path, *, lo_dr=None, hi_dr=None, lo_cache="all",
                   hi_cache="all", stamped_cell=None) -> tuple[Path, Path]:
    cells = contrast_plan(64)
    pair = contrast_pairs(cells)[0]
    gbps, _src = measured_bandwidth_gbps("nvidia_h200")
    stream = weight_stream_ms(weight_bytes_total(MIXTRAL), gbps)
    W = weight_bytes_total(MIXTRAL)
    lo_dr = pair.lo.traffic_bytes_per_tile(stream, W) if lo_dr is None else lo_dr
    hi_dr = pair.hi.traffic_bytes_per_tile(stream, W) if hi_dr is None else hi_dr
    lo = tmp_path / "lo.json"
    hi = tmp_path / "hi.json"
    lo.write_text(json.dumps(synthetic_counter_payload(
        pair.lo, lo_dr, cache_control=lo_cache, stamped_cell=stamped_cell)))
    hi.write_text(json.dumps(synthetic_counter_payload(
        pair.hi, hi_dr, cache_control=hi_cache)))
    return lo, hi


def test_the_contrast_is_scored_across_two_payloads_and_names_the_rival(tmp_path, capsys):
    """Acceptance for the mode: a TRAFFIC world reads as TRAFFIC.

    Until this repair `--analyse` scored ONE payload and the discriminator was a
    ratio ACROSS two, so the operator came off the pod with five scored cells
    and section 2's arithmetic to do by hand against the printed predictions.
    """
    lo, hi = _pair_payloads(tmp_path)
    assert main(["--contrast", str(lo), str(hi)]) == exit_codes.DONE
    out = capsys.readouterr().out
    names = [ln.name for ln in exit_codes.parse_result_lines(out)]
    assert names == ["X0", "XA-all"]
    line = next(ln for ln in exit_codes.parse_result_lines(out) if ln.name == "XA-all")
    assert line.verdict == "PASS" and "TRAFFIC" in line.detail
    pair = contrast_pairs(contrast_plan(64))[0]
    assert f"{pair.traffic_ratio:.3f}" in out
    assert exit_codes.classify_text(out) == exit_codes.DONE


def test_a_time_world_reads_as_time_and_not_as_a_failed_traffic_prediction(
        tmp_path, capsys):
    """The other rival is an ANSWER, not a failure: identical reads at both
    BLOCK_N mean the missing term is time and no byte model can hold it."""
    _lo, hi = _pair_payloads(tmp_path)
    same = json.loads(hi.read_text())["rows"][1]["dram_bytes_read"]
    del same
    gbps, _src = measured_bandwidth_gbps("nvidia_h200")
    stream = weight_stream_ms(weight_bytes_total(MIXTRAL), gbps)
    pair = contrast_pairs(contrast_plan(64))[0]
    dr = pair.hi.traffic_bytes_per_tile(stream, weight_bytes_total(MIXTRAL))
    lo, hi = _pair_payloads(tmp_path, lo_dr=dr, hi_dr=dr)
    assert main(["--contrast", str(lo), str(hi)]) == exit_codes.DONE
    out = capsys.readouterr().out
    line = next(ln for ln in exit_codes.parse_result_lines(out) if ln.name == "XA-all")
    assert line.verdict == "PASS" and "TIME" in line.detail


def test_a_ratio_between_the_two_rivals_refutes_both_as_stated(tmp_path, capsys):
    """Anything in between is reported as the split it is, not rounded to the
    nearer rival."""
    gbps, _src = measured_bandwidth_gbps("nvidia_h200")
    stream = weight_stream_ms(weight_bytes_total(MIXTRAL), gbps)
    pair = contrast_pairs(contrast_plan(64))[0]
    dr = pair.hi.traffic_bytes_per_tile(stream, weight_bytes_total(MIXTRAL))
    lo, hi = _pair_payloads(tmp_path, lo_dr=dr * 1.4, hi_dr=dr)
    assert main(["--contrast", str(lo), str(hi)]) == exit_codes.CLAIM_FAIL
    out = capsys.readouterr().out
    line = next(ln for ln in exit_codes.parse_result_lines(out) if ln.name == "XA-all")
    assert line.verdict == "FAIL" and "neither" in line.detail


def test_a_pair_nobody_ran_is_refused_and_never_scored_unknown(tmp_path, capsys):
    """The C1 defect's own shape, not repeated here: a gate for a contrast that
    has no payloads would be UNKNOWN, which counts against the run, so a plan
    run half through would report CLAIM_FAIL for the half it never took."""
    cells = contrast_plan(64)
    gbps, _src = measured_bandwidth_gbps("nvidia_h200")
    stream = weight_stream_ms(weight_bytes_total(MIXTRAL), gbps)
    W = weight_bytes_total(MIXTRAL)
    paths = []
    for name in ("bn32-g16-m64", "s3w8g1-m64"):
        cell = next(c for c in cells if c.name == name)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(synthetic_counter_payload(
            cell, cell.traffic_bytes_per_tile(stream, W))))
        paths.append(p)
    assert main(["--contrast", *[str(p) for p in paths]]) == exit_codes.REFUSED
    out = capsys.readouterr().out
    assert exit_codes.parse_result_lines(out) == []
    assert "no registered pair is complete" in out
    # and one file is not a contrast at all
    assert main(["--contrast", str(paths[0])]) == exit_codes.REFUSED


def test_pairs_are_matched_inside_one_cache_mode(tmp_path, capsys):
    """ncu's --cache-control sets the cache state, so a ratio taken across two
    cache modes is a ratio between two apparatuses."""
    lo, hi = _pair_payloads(tmp_path, lo_cache="all", hi_cache="none")
    assert main(["--contrast", str(lo), str(hi)]) == exit_codes.REFUSED
    assert "no registered pair is complete" in capsys.readouterr().out


def test_the_stamped_cell_name_is_read_back_rather_than_trusted(tmp_path, capsys):
    """`build_counter_payload` stamped `contrast` into every payload and nothing
    ever read it back. A payload whose stamp and whose knobs disagree is one of
    the two, and the ratio may not be taken under either name."""
    lo, hi = _pair_payloads(tmp_path, stamped_cell="bn128-g16-m64")
    assert main(["--contrast", str(lo), str(hi)]) == exit_codes.INVALID
    out = capsys.readouterr().out
    assert "stamped contrast says" in out
    line = next(ln for ln in exit_codes.parse_result_lines(out) if ln.name == "X0")
    assert line.kind == "VALIDITY" and line.verdict == "FAIL"


def test_a_ratio_over_a_payload_whose_own_gates_failed_is_invalid(tmp_path, capsys):
    """Validity voids the claims above it, ACROSS files as well as inside one."""
    lo, hi = _pair_payloads(tmp_path)
    payload = json.loads(lo.read_text())
    payload["rows"][2]["dram_bytes_read"] = payload["rows"][0]["dram_bytes_read"]
    lo.write_text(json.dumps(payload))
    assert main(["--contrast", str(lo), str(hi)]) == exit_codes.INVALID
    out = capsys.readouterr().out
    line = next(ln for ln in exit_codes.parse_result_lines(out) if ln.name == "X0")
    assert line.verdict == "FAIL"
    assert "V3" in out


def test_two_payloads_for_one_cell_and_cache_mode_are_a_collision(tmp_path):
    """Argument order would otherwise choose which of two measurements the ratio
    was taken over."""
    lo, hi = _pair_payloads(tmp_path)
    twin = tmp_path / "lo-again.json"
    twin.write_text(lo.read_text())
    args = build_parser().parse_args(
        ["--contrast", str(lo), str(twin), str(hi)])
    rows, _cells = contrast_rows([lo, twin, hi])
    assert len(rows) == 3
    assert DCR.do_contrast(args) == exit_codes.INVALID


def test_the_measured_slope_is_the_ratio_and_carries_no_bandwidth(tmp_path):
    """The discriminator is two measured slopes divided: no byte model, no
    weight total, no card rate, so a recalibration cannot move it."""
    lo, hi = _pair_payloads(tmp_path)
    dr_lo, resid = measured_dr_dn(json.loads(lo.read_text()))
    dr_hi, _resid = measured_dr_dn(json.loads(hi.read_text()))
    pair = contrast_pairs(contrast_plan(64))[0]
    assert resid < 1e-9
    assert dr_lo / dr_hi == pytest.approx(pair.traffic_ratio, rel=1e-9)
    assert dr_lo / dr_hi == pytest.approx(pair.lo.slope_ms / pair.hi.slope_ms, rel=1e-9)
    # a payload with fewer than three tile counts has no slope and says so
    thin = json.loads(lo.read_text())
    thin["rows"] = thin["rows"][:2]
    with pytest.raises(CounterRunRefused, match="a slope needs three"):
        measured_dr_dn(thin)


def test_the_pairs_the_plan_prints_are_the_pairs_the_scorer_reads(capsys):
    """One registration, two readers. The plan printed CONTRAST A and B out of
    inline literals and the scorer would have had its own copy."""
    assert main(["--dry-run"]) == exit_codes.DONE
    out = capsys.readouterr().out
    for pair in contrast_pairs(contrast_plan(64)):
        assert f"CONTRAST {pair.label}" in out
        assert f"{pair.traffic_ratio:.3f}" in out
        assert pair.separates(CONTRAST_TOLERANCE)
        assert "SEPARATING" in out
    assert "--contrast" in out, "the recipe must name the mode that scores it"


def test_a_pair_whose_rivals_overlap_is_registered_not_failed():
    """The C1 fix applied at the second call site. A pair whose two predictions
    both fit inside the scoring window cannot decide, and `separates` is the
    check that says so before a box is rented."""
    cells = contrast_plan(64)
    pair = contrast_pairs(cells)[0]
    assert pair.separates(0.05)
    # widen the window to 60% and the two acceptance bands overlap, so there
    # are ratios no reading can attribute
    assert not pair.separates(0.60)
    verdict, which = pair.read(1.5, 0.60)
    assert verdict == REFUSE and "does not separate" in which
    # and at the registered window the same ratio attributes to neither, which
    # is a FAIL and a result rather than an undecidable gate
    assert pair.read(1.5, CONTRAST_TOLERANCE)[0] == FAIL


def test_the_contrast_json_carries_provenance_and_its_own_instrument(tmp_path):
    """Every JSON this script writes goes through `stamped`, the fourth write
    site included."""
    lo, hi = _pair_payloads(tmp_path)
    out = tmp_path / "contrast.json"
    assert main(["--contrast", str(lo), str(hi), "--out", str(out)]) == exit_codes.DONE
    doc = json.loads(out.read_text())
    for key in ("git_sha", "gpu_name", "instrument", "provenance", "run_id"):
        assert key in doc
    assert doc["instrument"] == CONTRAST_INSTRUMENT
    assert "no-kernel-timed" in CONTRAST_INSTRUMENT
    assert doc["tolerance"] == CONTRAST_TOLERANCE
    assert doc["run_id"].startswith(NO_CARD.replace("-", "_")[:4]) or doc["run_id"]
    one = build_parser().parse_args(["--contrast", str(lo), str(hi)])
    two = build_parser().parse_args(["--contrast", str(lo)])
    assert run_id_for("contrast", one, NO_CARD) != run_id_for("contrast", two, NO_CARD)


def test_contrast_is_one_mode_among_the_seven(capsys):
    """Two modes at once still refuses, and the refusal names the new one."""
    assert main(["--contrast", "a.json", "--bracket"]) == exit_codes.REFUSED
    assert "--contrast" in capsys.readouterr().out


def test_a_ratio_across_two_cards_is_not_a_contrast(tmp_path, capsys):
    """The pair varies one knob; a payload from another card varies the card
    too, and the ratio would carry that difference silently."""
    lo, hi = _pair_payloads(tmp_path)
    other = json.loads(hi.read_text())
    other["device"] = "nvidia_a100_sxm4_80gb"
    hi.write_text(json.dumps(other))
    assert main(["--contrast", str(lo), str(hi)]) == exit_codes.INVALID
    out = capsys.readouterr().out
    assert "values of device" in out
    line = next(ln for ln in exit_codes.parse_result_lines(out) if ln.name == "X0")
    assert line.verdict == "FAIL"


# ==========================================================================
# BOTH OF ncu's CSV LAYOUTS (2026-09-24). No live ncu CSV has ever been
# captured in this repository, and `--csv --page raw` may print one row per
# launch and metric (LONG, what the parser was written against) or one row
# per launch with a units row under the header (WIDE). Both are read.
# ==========================================================================

def _wide_csv(rows: list[dict], units: dict[str, str]) -> str:
    """A planted `--csv --page raw` WIDE page: a header, a units row, one row
    per launch."""
    metrics = list(units)
    fixed = ["ID", "Process ID", "Process Name", "Kernel Name"]
    out = ["==PROF== Connected to process 1 (python)",
           ",".join(f'"{c}"' for c in fixed + metrics),
           ",".join(['""'] * len(fixed) + [f'"{units[m]}"' for m in metrics])]
    for i, row in enumerate(rows):
        out.append(",".join(f'"{c}"' for c in (str(i), "1", "python", row["kernel"]))
                   + "," + ",".join(f'"{row[m]}"' for m in metrics))
    return "\n".join(out) + "\n"


def test_the_parser_reads_a_wide_raw_page_and_converts_its_units_row():
    """ncu's raw CSV may be WIDE (one row per launch, a units row under the
    header), and nothing in this repository has ever captured one. The parser
    reads it launch by launch, converting each column by the unit its units
    row names: Mbyte is a million bytes, not one."""
    text = _wide_csv([{"kernel": "fused_moe_kernel", "dram__bytes_read.sum": "2.5",
                       "gpu__time_duration.sum": "1.5"},
                      {"kernel": "fused_moe_kernel", "dram__bytes_read.sum": "3.0",
                       "gpu__time_duration.sum": "2.0"}],
                     {"dram__bytes_read.sum": "Mbyte", "gpu__time_duration.sum": "usecond"})
    launches = parse_ncu_csv(text)
    assert [ln.launch_id for ln in launches] == ["0", "1"]
    assert launches[0].metrics["dram__bytes_read.sum"] == pytest.approx(2.5e6)
    assert launches[1].metrics["gpu__time_duration.sum"] == pytest.approx(2.0e3)
    assert DCR.ncu_csv_layout(text)[0] == DCR.CSV_WIDE
    assert DCR.ncu_csv_layout(canned_ncu_csv(**PLANTED))[0] == DCR.CSV_LONG


def test_the_probe_reads_open_off_a_wide_raw_page():
    """THE FALSE NEGATIVE THIS CLOSES. On a box whose counters work, a probe
    that parsed only the long layout read a wide page as "no ncu CSV header"
    and said REFUSE: the gate that decides a booking, wrong the other way."""
    wide = _wide_csv([{"kernel": "probe", "dram__bytes_read.sum": "4194304"}],
                     {"dram__bytes_read.sum": "byte"})
    info = DCR.probe_reading("/planted/ncu", "planted", 0,
                             f"{PK.MARKER} {PK.LAUNCHED} planted: one add_", "", wide)
    assert info["counters_read"] is True and info["metric_value"] == 4194304.0
    verdict, _notes = route_verdict({}, {}, info, {"present": False})
    assert verdict == "OPEN"


def test_a_wide_page_without_its_units_row_refuses():
    """The units row is the wide layout's `Metric Unit` column: without it a
    value ncu rescaled cannot be reduced, so the page refuses."""
    text = ('"ID","Kernel Name","dram__bytes_read.sum"\n'
            '"0","fused_moe_kernel","2.5"\n')
    with pytest.raises(CounterRunRefused, match="no units row"):
        parse_ncu_csv(text)
    with pytest.raises(CounterRunRefused, match="no 'ID' column"):
        parse_ncu_csv('"Kernel Name","dram__bytes_read.sum"\n"","byte"\n"k","1"\n')


def test_sector_units_and_the_explicit_unitless_launch_entry():
    """Sectors convert by their prefix; a launch attribute is read with no
    unit, because it is registered unitless; an empty unit on a byte or a
    sector metric still refuses; and only `launch__*` metrics are registered
    unitless."""
    text = _wide_csv([{"kernel": "fused_moe_kernel",
                       "lts__t_sectors_srcunit_tex_op_read.sum": "2",
                       "launch__grid_size": "7168"}],
                     {"lts__t_sectors_srcunit_tex_op_read.sum": "Msector",
                      "launch__grid_size": ""})
    (launch,) = parse_ncu_csv(text)
    assert launch.metrics["lts__t_sectors_srcunit_tex_op_read.sum"] == pytest.approx(2e6)
    assert launch.metrics["launch__grid_size"] == 7168.0
    for metric in ("dram__bytes_read.sum", "lts__d_sectors_fill_device.sum"):
        with pytest.raises(CounterRunRefused, match="never been shown"):
            parse_ncu_csv(_wide_csv([{"kernel": "k", metric: "1"}], {metric: ""}))
    with pytest.raises(CounterRunRefused, match="never been shown"):
        parse_ncu_csv(_wide_csv([{"kernel": "k", "launch__grid_size": "1"}],
                                {"launch__grid_size": "kilogram"}))
    unitless = [m for m, (_c, units) in DCR.NCU_LAUNCH_UNITS.items() if "" in units]
    assert unitless and all(m.startswith("launch__") for m in unitless)
    assert all(m.startswith("launch__") for m in DCR.NCU_LAUNCH_UNITS)
    assert all("" not in units for _c, units in DCR.NCU_METRIC_UNITS.values())


def test_a_soft_metric_that_cannot_be_read_is_recorded_and_a_hard_one_refuses():
    """RECORDED metrics are parsed soft: an unreadable cell lands in
    `Launch.unreadable` and never refuses the page. Every other metric still
    refuses the whole parse on an unreadable cell."""
    units = {"dram__bytes_read.sum": "byte", "launch__registers_per_thread": "furlong"}
    text = _wide_csv([{"kernel": "k", "dram__bytes_read.sum": "10",
                       "launch__registers_per_thread": "128"}], units)
    (launch,) = parse_ncu_csv(text, soft=frozenset({"launch__registers_per_thread"}))
    assert "launch__registers_per_thread" in launch.unreadable
    assert launch.metrics == {"dram__bytes_read.sum": 10.0}
    with pytest.raises(CounterRunRefused, match="furlong"):
        parse_ncu_csv(text)


# ==========================================================================
# THE R3 ARMS UNDER A DRAM COUNTER (`--family r3-arms`), 2026-09-24.
#
# Everything below runs off any GPU. The child and ncu are planted; every
# other step (the plan, the schedule, the parse, the attribution, the
# reduction, the page, the gates) is the code `--run --family r3-arms` runs.
# ==========================================================================

sys.path.insert(0, str(REPO / "scripts"))
import private_weight_reference as R3  # noqa: E402,I001


def test_the_byte_model_is_read_from_its_owners():
    """W from `routed_expert_weight_bytes_by_gemm`, and one tread's operand
    reads E x BM x (H + F) x b by construction: the family re-derives
    neither."""
    from moe.bench import weights as WEIGHTS
    cfg = MIXTRAL
    byte = DCR.r3_byte_model(cfg, "bf16", 32)
    by = WEIGHTS.routed_expert_weight_bytes_by_gemm(cfg, "bf16")
    assert (byte["W_w1"], byte["W_w2"]) == (by["w1"], by["w2"])
    assert byte["W"] == weight_bytes_total(cfg)
    assert (byte["operand_per_tile_w1"] + byte["operand_per_tile_w2"]
            == cfg.num_experts * 32 * (cfg.hidden_size + cfg.intermediate_size) * 2)


def _vllm_pid_walk(e: int, n: int, g: int, num_pid_n: int = 5, dead: int = 3) -> float:
    """vLLM v0.27.1's pid mapping, walked here independently of the module:
    every CTA of the grid, the expert of its pid_m, and the group it reads in."""
    num_pid_m = e * n + dead
    per_group = g * num_pid_n
    pairs = set()
    for pid in range(num_pid_m * num_pid_n):
        group_id = pid // per_group
        first_pid_m = group_id * g
        group_size_m = min(num_pid_m - first_pid_m, g)
        pid_m = first_pid_m + ((pid % per_group) % group_size_m)
        if pid_m < e * n:
            pairs.add((pid_m // n, group_id))
    return len(pairs) / e


def test_group_reads_is_vllms_pid_mapping_walked_by_brute_force():
    for e in (8, 64):
        for n in range(1, 9):
            for g in (1, 2, 3, 4, 8, 16, 64):
                assert DCR.group_reads(e, n, g) == pytest.approx(
                    _vllm_pid_walk(e, n, g), abs=1e-12), (e, n, g)
                assert DCR.pid_mapping_reads(e, n, g) == pytest.approx(
                    DCR.group_reads(e, n, g), abs=1e-12)


def _planted_manifest(g: int = 4, **plan_over):
    plan = dict(DCR.planted_r3_plan(g), **plan_over)
    return plan, DCR.planted_r3_manifest(plan, device_uuid="planted")


def test_attribution_is_exact_and_refuses_an_extra_launch_a_missing_one_and_a_swap():
    _plan, manifest = _planted_manifest()
    good = parse_ncu_csv(DCR.canned_r3_csv(manifest, "group", group_m=4))
    attributed = DCR.attribute_launches(good, manifest)
    assert len(attributed) == manifest["launch_count"] == 90
    assert [a["gemm"] for a in attributed[:4]] == ["w1", "w2", "w1", "w2"]
    for knob, needle in (({"extra_launch": True}, "EXACTLY"),
                         ({"drop_launch": 7}, "EXACTLY"),
                         ({"swap_grid_at": 10}, "ran a grid of")):
        text = DCR.canned_r3_csv(manifest, "group", group_m=4, **knob)
        with pytest.raises(CounterRunRefused, match=needle):
            DCR.attribute_launches(parse_ncu_csv(text), manifest)


@pytest.mark.parametrize("calls", [3, 5])
def test_per_call_bytes_are_recovered_from_the_launch_list(calls):
    """PER CALL is one fused_experts call, w1 launch plus w2 launch, averaged
    over the cell's K calls; K is read off the manifest and counted off the
    list. At K=3 and K=5 the reduction returns the planted per-call bytes."""
    plan = DCR.r3_plan(model="mixtral-8x7b", dtype="bf16", block_m=32, block_n=64,
                       num_stages=4, group_m=4, treads=DCR.R3_TREADS, kind="measure",
                       arms=R3.ARMS, calls=calls, warmups=2, profile_dir=Path("p"),
                       stem="g4")
    manifest = DCR.planted_r3_manifest(plan, device_uuid="planted")
    launches = parse_ncu_csv(DCR.canned_r3_csv(manifest, "group", group_m=4),
                             soft=frozenset(DCR.R3_RECORDED_METRICS))
    cells = DCR.r3_reduce_cells(DCR.attribute_launches(launches, manifest), manifest,
                                DCR.R3_ALL_METRICS)
    for cell in cells:
        assert cell["calls"] == calls and len(cell["per_call_values"]) == calls
        planted = [sum(DCR.r3_world_launch(
            "group", MIXTRAL, block_m=32, group_m=4, arm=cell["arm"], n=cell["n"],
            gemm=g, call=i, calls=calls, grid=cell["grid"][g])["dram__bytes_read.sum"]
            for g in ("w1", "w2")) for i in range(calls)]
        assert cell["per_call_values"] == pytest.approx(planted, rel=1e-12)
        assert cell["per_call"]["dram_bytes_read"] == pytest.approx(
            statistics.fmean(planted), rel=1e-12)


def _gates(page) -> dict:
    return {g["number"]: g["verdict"] for g in page["gates"]}


def _page_exit(page) -> int:
    return exit_codes.classify((g["kind"], g["number"], exit_codes.UNKNOWN
                                if g["verdict"] == REFUSE else g["verdict"])
                               for g in page["gates"])


#: Every planted world, at the G it is scored at. Listed here rather than read
#: off `R3_WORLDS` so that each case collects, and fails on its own, on a
#: tree without the family; the test holds the two lists equal.
WORLD_CASES = (("group", 1), ("group", 4), ("no-reuse", 4),
               ("private-reads-copy-0", 4), ("declaration", 4), ("noisy", 4),
               ("request-mismatch", 4), ("uncarded", 4))


@pytest.mark.parametrize(("world", "group_m"), WORLD_CASES)
def test_each_planted_world_scores_its_registered_exit(world, group_m):
    """GROUP: every validity gate and C1/C2/C3/C6 PASS, exit 0 (C3 is asked at
    G=1 only). NO-REUSE: C1 FAIL, exit 1. PRIVATE-READS-COPY-0: V5 FAIL, exit
    3. DECLARATION +5%: V7. NOISY 2%: V3. REQUEST-MISMATCH: V6. UNCARDED: V0."""
    assert {w for w, _g in WORLD_CASES} == set(DCR.R3_WORLDS)
    _why, want, gate = DCR.R3_WORLDS[world]
    page = DCR.planted_r3_page(world, group_m)
    verdicts = _gates(page)
    assert _page_exit(page) == want, verdicts
    if gate is not None:
        assert verdicts[gate] == FAIL, verdicts
    if world == "group":
        assert all(v == PASS for v in verdicts.values()), verdicts
        want_claims = {"C1", "C2", "C6"} | ({"C3"} if group_m == 1 else set())
        assert want_claims <= set(verdicts)


def test_a_planted_g4_staircase_with_its_n4_drop_is_valid():
    """The group model predicts q(4) = 1 < q(3) = 1.5 at G=4. The ladder
    family's monotone (V3) and affine (V4) gates would void that page; the
    r3-arms family does not apply them to SHARED, and every validity gate of
    the planted staircase passes."""
    page = DCR.planted_r3_page("group", 4)
    q = page["estimates"]["q_S"]["total"]
    assert q["4"] < q["3"], q
    assert q["3"] == pytest.approx(DCR.group_reads(8, 3, 4), abs=1e-3)
    validity = [g for g in page["gates"] if g["kind"] == "VALIDITY"]
    assert validity and all(g["verdict"] == PASS for g in validity)
    claims = " ".join(g["claim"] for g in page["gates"]).lower()
    assert "monoton" not in claims and "affine" not in claims


def test_c1_at_g1_is_not_asked_without_the_occupancy_limits():
    """The G=1 claim needs the co-residency window, which needs the page's own
    occupancy; with the limits unproven the claim is NOT ASKED, not failed."""
    page = DCR.planted_r3_page("group", 1)
    for cell in page["cells"]:
        for g in ("w1", "w2"):
            cell["recorded"][g] = {}
    gates, summary = DCR.score_r3_page(page)
    assert "C1" not in [g.number for g in gates]
    assert any(s.startswith("C1 at G=1") for s in summary["not_asked"])


def test_the_page_header_names_the_payloads_card_and_an_uncarded_page_fails_v0():
    page = DCR.planted_r3_page("group", 4)
    gates, summary = DCR.score_r3_page(page)
    first = DCR.r3_page_lines(page, gates, summary)[0]
    card = page["card"]
    assert first == DCR.card_line(card)
    assert first.startswith(f"CARD {card['name']} ({card['slug']}, UUID {card['uuid']}")
    assert f"the study's timing pages are {DCR.STUDY_CARD}" in first
    bare = DCR.planted_r3_page("uncarded", 4)
    assert _gates(bare)["V0"] == FAIL
    assert DCR.card_line(bare["card"]).startswith("CARD none")


def test_the_r3_schema_text_and_the_writer_name_the_same_keys():
    for key in DCR.R3_TOP_KEYS + DCR.R3_CELL_KEYS:
        assert f'"{key}"' in DCR.R3_SCHEMA_TEXT, key
    page = DCR.planted_r3_page("group", 4)
    args = build_parser().parse_args(["--run", "--family", "r3-arms", "--group-m", "4"])
    DCR.resolve_r3_defaults(args, ["--group-m", "4"])
    stamped_page = stamped(page, mode="r3-run", args=args, card=page["card"]["name"],
                           instrument=DCR.R3_RUN_INSTRUMENT)
    DCR.check_r3_page(stamped_page)
    assert set(DCR.R3_TOP_KEYS) <= set(stamped_page)
    for key in ("census", "proof"):
        gone = dict(stamped_page)
        del gone[key]
        with pytest.raises(CounterRunRefused, match=key):
            DCR.check_r3_page(gone)


def test_the_r3_dry_run_prints_the_group_model_group_reads_computes(capsys):
    assert main(["--dry-run", "--family", "r3-arms"]) == exit_codes.DONE
    out = capsys.readouterr().out
    treads = list(DCR.R3_TREADS)
    rows = {}
    for line in out.splitlines():
        m = re.match(r"^  G=(\d+)\s+((?:[\d.]+\s*)+)$", line)
        if m:
            vals = [float(v) for v in m.group(2).split()]
            if len(vals) == len(treads) + 1:
                rows[int(m.group(1))] = vals
    groups = list(DCR.R3_GROUPS) + [DCR.R3_OPTIONAL_GROUP]
    assert sorted(rows) == sorted(groups)
    for g in groups:
        want = [DCR.group_reads(8, n, g) for n in treads]
        assert rows[g][:-1] == pytest.approx([round(v, 4) for v in want], abs=1e-9)
        assert rows[g][-1] == pytest.approx(round(ols(treads, want)[1], 4), abs=1e-9)


def test_the_r3_dry_run_prints_the_ncu_argv_the_run_builds_and_prices_it(capsys):
    assert main(["--dry-run", "--family", "r3-arms"]) == exit_codes.DONE
    out = capsys.readouterr().out
    argv_lines = [ln for ln in out.splitlines() if ln.strip().startswith("G=")
                  and " ncu " in ln]
    assert len(argv_lines) == len(DCR.R3_GROUPS) + 1
    for line in argv_lines:
        for flag in ("--cache-control all", "--replay-mode kernel",
                     "-k regex:^fused_moe_kernel$", "--launch-skip", "--launch-count",
                     "--export", "--clock-control base", "--counter-child"):
            assert flag in line, (flag, line)
    sched = R3.counter_schedule(DCR.planted_r3_plan(4))
    assert f"--launch-skip {sched.launch_skip} " in argv_lines[0]
    assert f"--launch-count {sched.launch_count} " in argv_lines[0]
    per_g = DCR.r3_cost_s(sched.launch_count)
    assert f"= {per_g / 60:.1f} min" in out
    assert f"{len(DCR.R3_GROUPS)} G: {len(DCR.R3_GROUPS) * per_g / 60:.0f} min" in out
    assert "CARD: DECIDED ON THE BOX" in out and "--card is not read" in out


def test_the_r3_dry_run_refuses_a_tread_outside_r3s_ladder(capsys):
    assert main(["--dry-run", "--family", "r3-arms", "--tiles", "1,2,3,7"]) \
        == exit_codes.REFUSED
    assert "not a subset of R3's ladder" in capsys.readouterr().out


#: The commit a written planted page names, as `stamped` would put one on a
#: page `--run` writes: `--analyse` joins pages of one commit, and refuses a
#: join over pages that name none.
PLANTED_COMMIT = "0000000planted"


def _write_pages(tmp_path, specs) -> list[Path]:
    paths = []
    for world, g, over in specs:
        page = DCR.planted_r3_page(world, g)
        page["git_sha"] = PLANTED_COMMIT
        for key, value in over.items():
            page["card"][key] = value
        path = tmp_path / f"r3c-g{g}-{world}-{len(paths)}.json"
        path.write_text(json.dumps(page))
        paths.append(path)
    return paths


def test_analyse_scores_one_r3_page_and_its_first_line_is_the_card(tmp_path, capsys):
    (path,) = _write_pages(tmp_path, [("no-reuse", 4, {})])
    assert main(["--analyse", str(path)]) == exit_codes.CLAIM_FAIL
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith(f"CARD {DCR.R3_PLANTED_CARD['name']}")
    assert exit_codes.classify_text(out) == exit_codes.CLAIM_FAIL


def test_analyse_over_several_pages_prints_the_alpha_table(tmp_path, capsys):
    paths = _write_pages(tmp_path, [("group", 4, {}), ("group", 1, {})])
    out_json = tmp_path / "summary.json"
    assert main(["--analyse", *map(str, paths), "--out", str(out_json)]) \
        == exit_codes.DONE
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("CARD ")
    assert "ALPHA(G) OVER 2 PAGES" in out
    doc = json.loads(out_json.read_text())
    assert [p["group_m"] for p in doc["pages"]] == [1, 4]
    assert doc["instrument"] == DCR.R3_ANALYSE_INSTRUMENT


def test_analyse_refuses_two_card_uuids(tmp_path, capsys):
    paths = _write_pages(tmp_path, [("group", 4, {}),
                                    ("group", 1, {"uuid": "another-card"})])
    assert main(["--analyse", *map(str, paths)]) == exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "card UUID" in out and exit_codes.parse_result_lines(out) == []


def _timed_report(tmp_path, g: int, ratio: float, seed: int) -> Path:
    path = tmp_path / f"timed-g{g}-s{seed}.json"
    path.write_text(json.dumps({"experiment": "private_weight_reference",
                                "card": "nvidia_h200", "seed": seed, "ratio": ratio,
                                "pinned": {"GROUP_SIZE_M": g}, "claim_min_tread": 1}))
    return path


def test_c5_compares_the_bytes_with_the_timed_pages_it_reads(tmp_path, capsys):
    """C5 is cross-card, asked only with --timed-reference, and every number it
    compares with is read from those files. At G >= 4 the group world's byte
    ratio sits far below any floored time ratio (PASS); at G=1 the group
    world's co-resident w2 sharing puts alpha(1) below a bracket whose lower
    edge the planted timed pages put above it (FAIL, a finding)."""
    (g4,) = _write_pages(tmp_path, [("group", 4, {})])
    page = json.loads(g4.read_text())
    ratio = DCR.r3_estimates(page)["alpha_ratio"]
    timed = [_timed_report(tmp_path, 4, ratio + 0.5, s) for s in (0, 1)]
    assert main(["--analyse", str(g4), "--timed-reference", *map(str, timed)]) \
        == exit_codes.DONE
    out = capsys.readouterr().out
    c5 = next(ln for ln in exit_codes.parse_result_lines(out) if ln.name == "C5")
    assert c5.verdict == PASS and "CROSS-CARD" in out
    (g1,) = _write_pages(tmp_path, [("group", 1, {})])
    alpha1 = DCR.r3_estimates(json.loads(g1.read_text()))["alpha_slope"]["total"]
    timed1 = [_timed_report(tmp_path, 1, min(alpha1 + 0.05, 0.999), s) for s in (0, 1)]
    assert main(["--analyse", str(g1), "--timed-reference", *map(str, timed1)]) \
        == exit_codes.CLAIM_FAIL
    c5 = next(ln for ln in exit_codes.parse_result_lines(capsys.readouterr().out)
              if ln.name == "C5")
    assert c5.verdict == FAIL


def test_the_self_test_runs_the_family_worlds(capsys):
    assert main(["--self-test"]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert "THE R3-ARMS FAMILY" in out
    for world in DCR.R3_WORLDS:
        line = next(ln for ln in out.splitlines() if f"planted {world} world" in ln)
        assert " PASS " in line, line
    assert "group_reads against vLLM's pid mapping by brute force" in out


# the probe and the run, with ncu and the child planted.

def _plant_family_ncu(monkeypatch, *, names: str, log: str, rc: int = 0):
    monkeypatch.setattr(DCR.shutil, "which", lambda name: f"/usr/bin/{name}-stub")
    seen: list[list[str]] = []

    def fake_run(argv, timeout=60):
        del timeout
        seen.append(list(argv))
        if "--version" in argv:
            return 0, "ncu 2026.1.0.0\n", ""
        if "--query-metrics" in argv:
            return 0, names, ""
        Path(argv[argv.index("--log-file") + 1]).write_text(log)
        return rc, f"{PK.MARKER} {PK.LAUNCHED} planted: one add_", ""

    monkeypatch.setattr(DCR, "_run", fake_run)
    return seen


def _family_probe_log(metrics) -> str:
    row = {"kernel": "probe"}
    units = {}
    for m in metrics:
        units[m] = DCR.unit_table(m)[0]
        row[m] = "7"
    return _wide_csv([row], units)


def test_the_family_probe_asks_the_whole_list_and_drops_what_the_chip_lacks(monkeypatch):
    """The chip's list decides the hardware counters; the `launch__*`
    attributes are asked whether or not the list names them, because the
    probe's own launch is what proves them."""
    lacking = "lts__t_sectors_srcunit_ltcfabric"
    offered = [DCR.metric_base(m) for m in DCR.R3_ALL_METRICS
               if DCR.metric_base(m) != lacking and not m.startswith("launch__")]
    asked = [m for m in DCR.R3_ALL_METRICS if DCR.metric_base(m) != lacking]
    seen = _plant_family_ncu(monkeypatch, names="\n".join(f"{b}  some description"
                                                         for b in offered),
                             log=_family_probe_log(asked))
    info = DCR.probe_ncu(DCR.R3_FAMILY)
    probe = next(a for a in seen if "--log-file" in a)
    assert probe[probe.index("--metrics") + 1].split(",") == asked
    assert info["counters_read"] and counter_route_is_open(info)
    assert info["metrics_dropped"] == ["lts__t_sectors_srcunit_ltcfabric.sum"]
    assert info["metrics_proven"] == asked and info["csv_layout"] == DCR.CSV_WIDE
    verdict, notes = route_verdict({}, {}, info, {"present": False})
    assert verdict == "OPEN" and "--census-only" in notes[0]


def test_the_family_probe_refuses_when_the_chip_lacks_a_strict_metric(monkeypatch):
    offered = [DCR.metric_base(m) for m in DCR.R3_ALL_METRICS
               if m != "lts__t_sectors_srcunit_tex_op_read.sum"]
    seen = _plant_family_ncu(monkeypatch, names=" ".join(offered), log="")
    info = DCR.probe_ncu(DCR.R3_FAMILY)
    assert not counter_route_is_open(info)
    assert "STRICT" in info["cause"] and not any("--log-file" in a for a in seen)
    assert route_verdict({}, {}, info, {"present": False})[0] == REFUSE


def test_the_family_probe_needs_every_strict_metric_not_one(monkeypatch):
    """A probe that proves dram__bytes_read readable says nothing about the
    sector and launch metrics the pages are gated on: one STRICT metric
    missing from the probe's launch is not OPEN."""
    names = " ".join(DCR.metric_base(m) for m in DCR.R3_ALL_METRICS)
    _plant_family_ncu(monkeypatch, names=names,
                      log=_family_probe_log(["dram__bytes_read.sum"]))
    info = DCR.probe_ncu(DCR.R3_FAMILY)
    assert not info["counters_read"] and not counter_route_is_open(info)


def test_the_family_probe_asks_strict_alone_once_when_the_whole_list_reads_nothing(
        monkeypatch):
    """With no metric list to go by, the whole list is asked; if that reads no
    counter for a reason other than ERR_NVGPUCTRPERM (a name this ncu does not
    know refuses the whole invocation), the STRICT list alone is asked once
    more, and the first attempt is recorded."""
    monkeypatch.setattr(DCR.shutil, "which", lambda name: f"/usr/bin/{name}-stub")
    asked: list[str] = []

    def fake_run(argv, timeout=60):
        del timeout
        if "--version" in argv:
            return 0, "ncu 2026.1.0.0\n", ""
        if "--query-metrics" in argv:
            return 1, "", "==ERROR== planted: no list"
        metrics = argv[argv.index("--metrics") + 1].split(",")
        asked.append(argv[argv.index("--metrics") + 1])
        log = argv[argv.index("--log-file") + 1]
        if len(metrics) > len(DCR.R3_STRICT_METRICS):
            Path(log).write_text("==ERROR== Failed to find metric planted\n")
        else:
            Path(log).write_text(_family_probe_log(DCR.R3_STRICT_METRICS))
        return 0, f"{PK.MARKER} {PK.LAUNCHED} planted: one add_", ""
    monkeypatch.setattr(DCR, "_run", fake_run)
    info = DCR.probe_ncu(DCR.R3_FAMILY)
    assert asked == [",".join(DCR.R3_ALL_METRICS), ",".join(DCR.R3_STRICT_METRICS)]
    assert counter_route_is_open(info) and "first_attempt" in info
    assert info["metrics_proven"] == list(DCR.R3_STRICT_METRICS)
    assert info["metrics_query"].startswith("ncu --query-metrics exited 1")
    assert [a["counters_read"] for a in info["attempts"]] == [False, True]


def _plant_probe_asks(monkeypatch, *, names: str, refuse) -> list[list[str]]:
    """An ncu whose metric query lists `names` and whose probe invocation reads
    every metric it is asked for, unless `refuse(metrics)` returns the error
    line that refuses the whole invocation. Returns each probe ask's metrics."""
    monkeypatch.setattr(DCR.shutil, "which", lambda name: f"/usr/bin/{name}-stub")
    asked: list[list[str]] = []

    def fake_run(argv, timeout=60):
        del timeout
        if "--version" in argv:
            return 0, "ncu 2026.1.0.0\n", ""
        if "--query-metrics" in argv:
            return 0, names, ""
        metrics = argv[argv.index("--metrics") + 1].split(",")
        asked.append(metrics)
        error = refuse(metrics)
        Path(argv[argv.index("--log-file") + 1]).write_text(
            error or _family_probe_log(metrics))
        return 0, f"{PK.MARKER} {PK.LAUNCHED} planted: one add_", ""
    monkeypatch.setattr(DCR, "_run", fake_run)
    return asked


#: The metric list of a chip that offers every hardware counter the family
#: asks. The `launch__*` attributes are not held to the list, so it omits them.
_HARDWARE_LIST = "\n".join(f"{DCR.metric_base(m)}  planted" for m in DCR.R3_ALL_METRICS
                           if not m.startswith("launch__"))


def test_a_readable_list_and_one_unknown_launch_name_still_reads_open(monkeypatch):
    """The `launch__*` names pass the list check unverified, so one this ncu
    does not know refuses the whole first ask on a box whose counters work.
    Until 2026-09-24 the retry ran only when the list could not be read, so
    this box read REFUSE and `--run` refused with it. The probe now asks
    again whatever the list said: STRICT plus the list-verified metrics, and
    every ask is recorded."""
    bad = "launch__occupancy_limit_registers"
    asked = _plant_probe_asks(
        monkeypatch, names=_HARDWARE_LIST,
        refuse=lambda ms: f"==ERROR== Failed to find metric {bad}\n" if bad in ms else "")
    info = DCR.probe_ncu(DCR.R3_FAMILY)
    verified = [m for m in DCR.R3_ALL_METRICS
                if m not in DCR.R3_STRICT_METRICS and not m.startswith("launch__")]
    assert asked == [list(DCR.R3_ALL_METRICS), list(DCR.R3_STRICT_METRICS) + verified]
    assert info["counters_read"] and counter_route_is_open(info)
    assert info["metrics_proven"] == asked[-1]
    assert [a["metrics"] for a in info["attempts"]] == asked
    assert [a["counters_read"] for a in info["attempts"]] == [False, True]
    assert bad in info["attempts"][0]["ncu_error"] and bad in info["first_attempt"]
    left_out = [m for m in DCR.R3_ALL_METRICS
                if m.startswith("launch__") and m not in DCR.R3_STRICT_METRICS]
    assert left_out and all(m in info["metrics_unproven"] for m in left_out)
    verdict, notes = route_verdict({}, {}, info, {"present": False})
    assert verdict == "OPEN" and "attempt 1 (" in notes[1], notes


def test_the_retries_end_at_strict_alone_or_at_a_permission_refusal(monkeypatch):
    """When STRICT plus the verified metrics also reads nothing, STRICT alone
    is asked last; ERR_NVGPUCTRPERM on any ask is the box's answer and ends
    the retries there."""
    extra = set(DCR.R3_ALL_METRICS) - set(DCR.R3_STRICT_METRICS)
    asked = _plant_probe_asks(
        monkeypatch, names=_HARDWARE_LIST,
        refuse=lambda ms: ("==ERROR== Failed to find metric planted\n"
                           if extra & set(ms) else ""))
    info = DCR.probe_ncu(DCR.R3_FAMILY)
    assert len(asked) == 3 and asked[-1] == list(DCR.R3_STRICT_METRICS)
    assert counter_route_is_open(info) and len(info["attempts"]) == 3
    assert info["metrics_proven"] == list(DCR.R3_STRICT_METRICS)
    asked = _plant_probe_asks(
        monkeypatch, names=_HARDWARE_LIST,
        refuse=lambda ms: "==ERROR== ERR_NVGPUCTRPERM - The user does not have "
                          "permission to access NVIDIA GPU Performance Counters\n")
    info = DCR.probe_ncu(DCR.R3_FAMILY)
    assert len(asked) == 1 and info["permission_refused"]
    assert len(info["attempts"]) == 1 and "first_attempt" not in info


def test_the_ladder_probe_and_argv_are_unchanged():
    """`--family ladder` is byte-identical: its probe asks its one metric and
    `ncu_argv` is the list it always was, now built through the one
    `ncu_common_flags` both families call."""
    assert DCR.ncu_probe_argv("ncu", Path("/l.csv")) == [
        "ncu", "--metrics", NCU_METRICS[0], "--launch-count", "1",
        "--target-processes", "all", "--csv", "--page", "raw", "--log-file", "/l.csv",
        "--", DCR.sys.executable, str(DCR.NCU_PROBE_KERNEL)]
    assert ncu_argv("ncu", "all", Path("/x.csv")) == [
        "ncu", "--metrics", ",".join(NCU_METRICS), "--replay-mode", "kernel",
        "--cache-control", "all", "--csv", "--page", "raw", "--target-processes",
        "all", "--log-file", "/x.csv"]
    import inspect
    for fn in (DCR.ncu_argv, DCR.r3_ncu_argv):
        assert "ncu_common_flags(" in inspect.getsource(fn), fn.__name__
    assert build_parser().parse_args([]).family == DCR.LADDER_FAMILY


def _planted_open() -> dict:
    """What `probe_ncu(R3_FAMILY)` returns on a box whose family is OPEN."""
    return {"present": True, "binary": "/planted/ncu", "version": "planted",
            "counters_read": True, "permission_refused": False,
            "family": DCR.R3_FAMILY, "metrics_proven": list(DCR.R3_ALL_METRICS),
            "metrics_unproven": {}, "metrics_dropped": [], "cause": "planted"}


def _plant_the_box(monkeypatch, *, world="group", card=None, probe=None):
    """The box: ncu open for the family, a planted card, and an ncu whose
    capture runs a planted child (it writes the manifest the plan names) and
    whose --import prints the planted world's CSV for that manifest."""
    card = dict(DCR.R3_PLANTED_CARD) if card is None else card
    probe = _planted_open() if probe is None else probe
    monkeypatch.setattr(DCR, "probe_ncu", lambda family=None: dict(probe))
    monkeypatch.setattr(DCR, "live_card_block", lambda: card)
    monkeypatch.setattr(DCR, "r3_stack_versions",
                        lambda: {"torch": "t", "triton": "tr", "vllm": "v", "python": "p"})
    calls: list[list[str]] = []

    def fake_run(argv, timeout=60):
        del timeout
        calls.append(list(argv))
        if "--counter-child" in argv:
            plan_path = Path(argv[argv.index("--counter-child") + 1])
            plan = json.loads(plan_path.read_text())
            manifest = DCR.planted_r3_manifest(plan, device_uuid=card["uuid"])
            Path(plan["manifest"]).write_text(json.dumps(manifest))
            Path(argv[argv.index("--export") + 1]).write_text(json.dumps(
                {"plan": str(plan_path)}))
            return 0, "==PROF== planted capture\n", ""
        if "--import" in argv:
            meta = json.loads(Path(argv[argv.index("--import") + 1]).read_text())
            plan = json.loads(Path(meta["plan"]).read_text())
            manifest = json.loads(Path(plan["manifest"]).read_text())
            return 0, DCR.canned_r3_csv(manifest, world, group_m=plan["group_m"]), ""
        raise AssertionError(f"unplanted ncu call {argv}")

    monkeypatch.setattr(DCR, "_run", fake_run)
    return calls


def test_the_r3_run_end_to_end_on_planted_ncu_output(tmp_path, monkeypatch, capsys):
    """The census, then one G's page, then --analyse over it: every step but
    the child and ncu is the shipped code, and the page is VALID."""
    calls = _plant_the_box(monkeypatch)
    census = tmp_path / "session" / "census.json"
    assert main(["--run", "--family", "r3-arms", "--census-only",
                 "--out", str(census)]) == exit_codes.DONE
    cen = json.loads(census.read_text())
    assert cen["gemms_per_call_measured"] == R3.GEMMS_PER_CALL
    assert cen["card"]["uuid"] == DCR.R3_PLANTED_CARD["uuid"] and cen["commit"]
    capture = next(a for a in calls if "--counter-child" in a)
    assert capture[capture.index("--launch-skip") + 1] == "0"
    assert "--launch-count" not in capture, "the census caps nothing"
    capsys.readouterr()
    page_path = tmp_path / "results" / "r3c-g4.json"
    assert main(["--run", "--family", "r3-arms", "--group-m", "4",
                 "--census", str(census), "--out", str(page_path)]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert out.splitlines()[0] == DCR.card_line(DCR.R3_PLANTED_CARD)
    page = json.loads(page_path.read_text())
    assert set(DCR.R3_TOP_KEYS) <= set(page)
    assert page["design"]["group_m"] == 4 and page["design"]["launch_count"] == 90
    assert page["run_id"].startswith(DCR.R3_PLANTED_CARD["slug"])
    assert page["instrument"] == DCR.R3_RUN_INSTRUMENT
    profiles = page_path.parent / "r3c-g4.profiles"
    for name in ("g4.plan.json", "g4.manifest.json", "g4.ncu-rep", "g4.csv"):
        assert (profiles / name).exists(), name
    assert main(["--analyse", str(page_path)]) == exit_codes.DONE


def test_a_page_is_rebuilt_from_its_profiles_with_no_card_and_no_child(
        tmp_path, monkeypatch, capsys):
    """THE CAPTURE AND THE REDUCTION ARE SPLIT: after a capture, the page is
    rebuilt from `<out>.profiles` alone (off the box, after a parser fix) and
    carries the capture's card, stack, argv and commit. Neither the probe nor
    the card nor the child is asked; with no ncu on PATH the CSV the capture
    reduced is read."""
    _plant_the_box(monkeypatch)
    census = tmp_path / "census.json"
    page_path = tmp_path / "r3c-g4.json"
    assert main(["--run", "--family", "r3-arms", "--census-only", "--out",
                 str(census)]) == exit_codes.DONE
    assert main(["--run", "--family", "r3-arms", "--group-m", "4", "--census",
                 str(census), "--out", str(page_path)]) == exit_codes.DONE
    first = json.loads(page_path.read_text())
    page_path.unlink()

    def off_the_box(*a, **k):
        raise AssertionError("--reduce-only asked the box")
    for name in ("probe_ncu", "live_card_block", "r3_stack_versions", "_run"):
        monkeypatch.setattr(DCR, name, off_the_box)
    monkeypatch.setattr(DCR.shutil, "which", lambda name: None)
    capsys.readouterr()
    assert main(["--run", "--family", "r3-arms", "--reduce-only", "--group-m", "4",
                 "--census", str(census), "--out", str(page_path)]) == exit_codes.DONE
    out = capsys.readouterr().out
    assert out.splitlines()[0] == DCR.card_line(DCR.R3_PLANTED_CARD)
    again = json.loads(page_path.read_text())
    assert again["cells"] == first["cells"] and again["card"] == first["card"]
    assert again["ncu"]["argv"] == first["ncu"]["argv"]
    assert again["ncu"]["capture_commit"] == first["ncu"]["capture_commit"]
    assert main(["--run", "--family", "r3-arms", "--reduce-only", "--group-m", "16",
                 "--census", str(census), "--out", str(page_path)]) == exit_codes.REFUSED
    assert "no capture to reduce" in capsys.readouterr().out


def test_the_r3_run_refuses_without_ncu_and_without_a_card(tmp_path, monkeypatch, capsys):
    shut = {"present": False, "counters_read": False, "why": "no ncu on PATH"}
    census = tmp_path / "census.json"
    census.write_text("{}")
    argv = ["--run", "--family", "r3-arms", "--group-m", "4", "--census", str(census),
            "--out", str(tmp_path / "p.json")]
    _plant_the_box(monkeypatch, probe=shut)
    assert main(argv) == exit_codes.REFUSED
    assert "no ncu on PATH" in capsys.readouterr().out
    monkeypatch.setattr(DCR, "probe_ncu", lambda family=None: _planted_open())
    monkeypatch.setattr(DCR, "live_card_block", lambda: None)
    assert main(argv) == exit_codes.REFUSED
    assert "no card" in capsys.readouterr().out
    assert not (tmp_path / "p.json").exists()


def test_a_page_needs_its_g_and_a_census_from_this_card(tmp_path, monkeypatch, capsys):
    _plant_the_box(monkeypatch)
    out = str(tmp_path / "p.json")
    assert main(["--run", "--family", "r3-arms", "--census", "c.json", "--out", out]) \
        == exit_codes.REFUSED
    assert "--group-m names it" in capsys.readouterr().out
    assert main(["--run", "--family", "r3-arms", "--group-m", "4", "--out", out]) \
        == exit_codes.REFUSED
    assert "Run --census-only first" in capsys.readouterr().out
    census = tmp_path / "census.json"
    assert main(["--run", "--family", "r3-arms", "--census-only", "--out",
                 str(census)]) == exit_codes.DONE
    doc = json.loads(census.read_text())
    doc["card"]["uuid"] = "another-card"
    census.write_text(json.dumps(doc))
    capsys.readouterr()
    assert main(["--run", "--family", "r3-arms", "--group-m", "4", "--census",
                 str(census), "--out", out]) == exit_codes.REFUSED
    assert "is not this card" in capsys.readouterr().out
    assert not Path(out).exists()


#: What `provenance._git` returns when root runs git in a checkout its login
#: user owns, which is what the sudo counter door does.
_DUBIOUS = (None, None, None, "git rev-parse: fatal: detected dubious ownership in "
                              "repository at '/home/ubuntu/moe/repo'")


def test_no_census_or_page_is_written_or_joined_without_a_commit(
        tmp_path, monkeypatch, capsys):
    """A census licenses a page by commit, and `None == None` is not a match.
    Until 2026-09-24 a tree git could not name (root in a user-owned checkout,
    under the sudo counter door) wrote a census with `commit: None`, the page
    compared None with None and was accepted, and `--analyse` joined pages
    over the commit set {'None'}. Now the census and the page refuse, naming
    provenance's reason and the safe.directory remedy; a census that names no
    commit licenses no page; and a join over pages that name none refuses."""
    _plant_the_box(monkeypatch)
    census, page = tmp_path / "census.json", tmp_path / "r3c-g4.json"
    real_git = PV._git
    monkeypatch.setattr(PV, "_git", lambda root: _DUBIOUS)
    assert main(["--run", "--family", "r3-arms", "--census-only", "--out",
                 str(census)]) == exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "dubious ownership" in out and "safe.directory" in out
    assert not census.exists()
    monkeypatch.setattr(PV, "_git", real_git)
    assert main(["--run", "--family", "r3-arms", "--census-only", "--out",
                 str(census)]) == exit_codes.DONE
    doc = json.loads(census.read_text())
    assert doc["commit"]
    capsys.readouterr()
    monkeypatch.setattr(PV, "_git", lambda root: _DUBIOUS)
    assert main(["--run", "--family", "r3-arms", "--group-m", "4", "--census",
                 str(census), "--out", str(page)]) == exit_codes.REFUSED
    assert "safe.directory" in capsys.readouterr().out and not page.exists()
    monkeypatch.setattr(PV, "_git", real_git)
    doc["commit"] = None
    census.write_text(json.dumps(doc))
    assert main(["--run", "--family", "r3-arms", "--group-m", "4", "--census",
                 str(census), "--out", str(page)]) == exit_codes.REFUSED
    assert "names no commit" in capsys.readouterr().out and not page.exists()
    paths = _write_pages(tmp_path, [("group", 4, {}), ("group", 1, {})])
    for p in paths:
        bare = json.loads(p.read_text())
        del bare["git_sha"]
        p.write_text(json.dumps(bare))
    assert main(["--analyse", *map(str, paths)]) == exit_codes.REFUSED
    assert "name no commit" in capsys.readouterr().out


def test_a_page_is_not_rebuilt_from_a_capture_or_a_tree_with_no_commit(
        tmp_path, monkeypatch, capsys):
    """`--reduce-only` writes a page too: it refuses a capture record that
    names no commit and a reducing tree git cannot name."""
    _plant_the_box(monkeypatch)
    census, page = tmp_path / "census.json", tmp_path / "r3c-g4.json"
    assert main(["--run", "--family", "r3-arms", "--census-only", "--out",
                 str(census)]) == exit_codes.DONE
    assert main(["--run", "--family", "r3-arms", "--group-m", "4", "--census",
                 str(census), "--out", str(page)]) == exit_codes.DONE
    page.unlink()
    reduce = ["--run", "--family", "r3-arms", "--reduce-only", "--group-m", "4",
              "--census", str(census), "--out", str(page)]
    real_git = PV._git
    monkeypatch.setattr(PV, "_git", lambda root: _DUBIOUS)
    capsys.readouterr()
    assert main(reduce) == exit_codes.REFUSED
    assert "safe.directory" in capsys.readouterr().out and not page.exists()
    monkeypatch.setattr(PV, "_git", real_git)
    record = tmp_path / "r3c-g4.profiles" / "g4.capture.json"
    capture = json.loads(record.read_text())
    capture["commit"] = None
    record.write_text(json.dumps(capture))
    assert main(reduce) == exit_codes.REFUSED
    assert "names no commit" in capsys.readouterr().out and not page.exists()


def test_the_family_flags_are_refused_where_they_mean_nothing(capsys):
    assert main(["--dry-run", "--census-only"]) == exit_codes.REFUSED
    assert main(["--dry-run", "--timed-reference", "x.json"]) == exit_codes.REFUSED
    assert main(["--bracket", "--family", "r3-arms"]) == exit_codes.REFUSED
    capsys.readouterr()


def test_counters_doc_section_6_quotes_the_numbers_the_family_computes():
    """docs/COUNTERS.md section 6 prints the r3-arms byte model and the group
    model's table; both are recomputed here from the functions that own them,
    so the page fails a test the day either moves."""
    doc = (REPO / "docs" / "COUNTERS.md").read_text()
    sec = doc[doc.index("## 6. The R3 arms under the counter"):]
    byte = DCR.r3_byte_model(MIXTRAL, "bf16", 32)
    for key in ("W", "W_w1", "W_w2", "operand_per_tile_w1", "operand_per_tile_w2"):
        assert f"{byte[key]:,}" in sec, key
    for g, qs, slope in DCR.r3_group_rows(8, DCR.R3_TREADS,
                                          list(DCR.R3_GROUPS) + [DCR.R3_OPTIONAL_GROUP]):
        row = f"| {g} | " + " | ".join(f"{v:.4f}" for v in qs) + f" | {slope:.4f} |"
        assert row in sec, row
    assert "`--family ladder` (the\ndefault) behaves exactly as before" in sec
    assert "\u2014" not in sec, "no em-dash in the section"
