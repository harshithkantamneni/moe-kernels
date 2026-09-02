"""The one measurement in this study that goes through no fit, and its refusals.

`scripts/bm128_roofline.py` forces BLOCK_SIZE_M=128, sweeps the batch across the
multi-tile onset, and divides achieved TFLOP/s by the attached card's own
measured dense bf16 rate. Nothing in its verdict comes from alpha, an anchor, an
estimator or a ladder, so most of this file is about the two things that CAN
still go wrong: the denominator belonging to another machine, and a derivative
read off noise.

SIX GROUPS.

  - THE GEOMETRY. Multi-tile onset is `T > BLOCK_M E / k`, computed per model and
    never hardcoded; the grid is a chain of doublings or it is refused, because
    "gain per doubling" is not a quantity on a grid that does not double.
  - THE DERIVATIVE. A plateau is a claim about one, so a flat curve, a rising
    curve, a curve too short to have a slope and a curve whose noise swamps the
    threshold must produce four different answers, and exactly one of them may
    be PASS.
  - THE EXCLUSIONS. A throttled repeat leaves the median; a point whose every
    repeat was throttled leaves the gated set and stays on the plot; and a
    throttle check over cells that never carried a clock reports UNKNOWN rather
    than "no failures", which is the non-vacuity rule this project keeps
    relearning.
  - THE DENOMINATOR. A measured run with no calibration for its own device
    REFUSES, a hypothesis roof fails V0, and a report standing on one can never
    reach a verdict however clean its claim gates look. Seven published A100
    reports were scored against an H200 ridge because a missing calibration was
    allowed to fall back.
  - THE IDENTITY. The run id carries every swept knob AND the card, and this repo
    has lost an arm to each of those omissions once already. Resume is keyed on
    ROWS and not on tiles, because the three pre-onset batches all have one tile
    per expert.
  - THE SELF TEST, which is the claim that these gates DISCRIMINATE. Gates that
    answer the same in every planted world cannot settle this experiment.

The script is loaded by path, because `scripts/` is not a package.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.spec import MODEL_CONFIGS  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module that is not there yet makes the
    # decorator fail with an AttributeError that names nothing useful.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rf():
    return _load("bm128_roofline", "bm128_roofline.py")


@pytest.fixture(scope="module")
def cfg():
    return MODEL_CONFIGS["mixtral-8x7b"]


@pytest.fixture(scope="module")
def roof(rf):
    """The committed H200 calibration, which is what --dry-run is allowed."""
    return rf._hypothesis_roof("test")


def _args(rf, **over):
    args = rf.build_parser().parse_args([])
    for key, value in over.items():
        setattr(args, key, value)
    return args


def _point(rf, *, block_m=128, rows=256, tflops=500.0, roof_tflops=712.0,
           spread=0.001, reps=3, retained=True, tokens=None, tiles=None):
    tiles = tiles if tiles is not None else max(1, -(-rows // block_m))
    return rf.Point(
        block_m=block_m, rows_per_expert=rows, tiles=tiles,
        tokens=tokens if tokens is not None else rows * 4,
        regime=rf.regime_of(rows, block_m), tile_eff=1.0, reps=reps,
        ms_p50=1.0, spread=spread, useful_tflops=tflops,
        roof_fraction=tflops / roof_tflops, sm_clock_mhz=1500,
        throttled_reps=0, retained=retained, excluded_why="")


# --------------------------------------------------------------------------
# The geometry.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model,expected", [("mixtral-8x7b", 512),
                                            ("qwen2-57b-a14b", 1024)])
def test_multi_tile_onset_is_computed_per_model_not_hardcoded(rf, model, expected):
    """`T = BLOCK_M E / k`. A 512 written down straddles nothing on qwen2."""
    model_cfg = MODEL_CONFIGS[model]
    assert rf.onset_tokens(model_cfg, rf.SUBJECT_BLOCK_M) == expected
    assert (expected * model_cfg.top_k // model_cfg.num_experts
            == rf.SUBJECT_BLOCK_M)


def test_the_grid_is_a_chain_of_doublings_spanning_the_onset(rf, cfg):
    rows = rf.doubling_rows(cfg, 32, 4096, 128)
    assert rows == [32, 64, 128, 256, 512, 1024, 2048, 4096]
    assert all(b == 2 * a for a, b in zip(rows, rows[1:], strict=False))
    assert [r for r in rows if r < 128] and [r for r in rows if r > 128], \
        "the sweep must spend cells on both sides of the onset"
    assert max(rows) // 128 == 32, "the observed arm reaches 32 M-tiles"


def test_a_grid_that_does_not_double_is_refused(rf, cfg):
    """"Gain per doubling" is not a quantity on a grid that does not double."""
    with pytest.raises(SystemExit, match="not a power of two"):
        rf.doubling_rows(cfg, 48, 4096, 128)


def test_a_model_whose_routing_cannot_express_the_grid_is_refused(rf):
    """deepseek-v2-lite needs rows to be a multiple of 3; 32 is not.

    REFUSED rather than nudged. A nudged row is not a full tile stack and its
    throughput is divided by padding nobody recorded.
    """
    lite = MODEL_CONFIGS["deepseek-v2-lite"]
    with pytest.raises(SystemExit, match="multiple of 3"):
        rf.doubling_rows(lite, 32, 4096, 128)


def test_regime_names_the_three_places_a_point_can_be(rf):
    assert rf.regime_of(64, 128) == "pre-onset"
    assert rf.regime_of(128, 128) == "onset"
    assert rf.regime_of(256, 128) == "multi-tile"


def test_only_full_stack_multi_tile_points_are_gated(rf):
    points = [_point(rf, rows=64), _point(rf, rows=128), _point(rf, rows=256),
              _point(rf, rows=512, retained=False)]
    gated = rf.multi_tile(points)
    assert [p.rows_per_expert for p in gated] == [256], \
        "pre-onset, onset and excluded points must all stay out of the gates"


# --------------------------------------------------------------------------
# The derivative. Four curves, four answers, one PASS.
# --------------------------------------------------------------------------

def test_a_flat_curve_plateaus(rf):
    points = [_point(rf, rows=r, tflops=500.0 + i * 0.5)
              for i, r in enumerate((256, 512, 1024, 2048, 4096))]
    p = rf.plateau_of(points, doublings=2)
    assert p.plateaued is True
    assert p.span_doublings == pytest.approx(2.0)
    assert abs(p.gain_per_doubling) < rf.PLATEAU_GAIN_PER_DOUBLING


def test_a_rising_curve_is_still_rising_and_the_word_plateau_is_refused(rf):
    points = [_point(rf, rows=r, tflops=100.0 * 2 ** i)
              for i, r in enumerate((256, 512, 1024, 2048, 4096))]
    p = rf.plateau_of(points, doublings=2)
    assert p.plateaued is False
    assert p.gain_per_doubling == pytest.approx(1.0, abs=1e-9)
    gate = rf.gate_c2_plateau(p, points)
    assert gate.passed is False
    text = " ".join(gate.lines)
    assert "still rising at the largest batch measured" in text
    assert "may say 'plateaued'" in text


def test_the_gain_is_per_doubling_and_not_per_grid_point(rf):
    """The same total gain over twice the span is half the gain per doubling."""
    short = [_point(rf, rows=1024, tflops=100.0), _point(rf, rows=4096, tflops=400.0)]
    long = [_point(rf, rows=256, tflops=100.0), _point(rf, rows=4096, tflops=400.0)]
    a = rf.plateau_of(short, doublings=2)
    b = rf.plateau_of(long, doublings=2)
    assert a.total_gain == pytest.approx(b.total_gain)
    assert a.span_doublings == 2 and b.span_doublings == 4
    assert a.gain_per_doubling == pytest.approx(1.0)
    assert b.gain_per_doubling == pytest.approx(2 ** 0.5 - 1.0)


def test_too_few_points_is_unknown_and_never_a_plateau(rf):
    p = rf.plateau_of([_point(rf, rows=256)], doublings=2)
    assert p.plateaued is None
    assert "at least two" in p.reason
    assert rf.gate_c2_plateau(p, []).passed is None


def test_a_span_shorter_than_the_gate_asks_for_is_unknown(rf):
    """A plateau over one doubling is not a plateau over two, and says so."""
    points = [_point(rf, rows=2048, tflops=500.0), _point(rf, rows=4096, tflops=501.0)]
    p = rf.plateau_of(points, doublings=2)
    assert p.span_doublings == pytest.approx(1.0)
    assert p.plateaued is None, "a short span may not report a plateau"
    assert "span only" in p.reason


def test_a_spread_that_cannot_resolve_the_threshold_refuses(rf):
    """A plateau read off noise is not a plateau.

    The ratio of two per-tread medians carries about `s sqrt(2) / sqrt(reps)`, so
    a 10% spread over three repeats puts 8.2% on a gate that has to resolve 2%.
    """
    points = [_point(rf, rows=r, tflops=500.0, spread=0.10, reps=3)
              for r in (1024, 2048, 4096)]
    p = rf.plateau_of(points, doublings=2)
    assert p.resolvable is False
    assert p.plateaued is None
    gate = rf.gate_c2_plateau(p, points)
    assert gate.passed is None
    assert "REFUSED on resolution" in " ".join(gate.lines)


def test_a_single_repeat_has_no_spread_and_so_cannot_resolve_anything(rf):
    points = [_point(rf, rows=r, tflops=500.0, spread=None, reps=1)
              for r in (1024, 2048, 4096)]
    p = rf.plateau_of(points, doublings=2)
    assert p.resolution is None and p.resolvable is None
    assert p.plateaued is None


# --------------------------------------------------------------------------
# The exclusions.
# --------------------------------------------------------------------------

def _timing(rf, *, block_m=128, rows=256, rep=1, ms=1.0, start=1500, end=1500):
    drift = (start - end) / start * 100.0 if start else 0.0
    return rf.Timing(block_m, rows, max(1, -(-rows // block_m)), rows * 4, rep,
                     ms, ms, 0.0, 10, start, end, 50, 50, drift,
                     drift > rf.THROTTLE_DRIFT_PCT)


def test_a_throttled_repeat_leaves_the_median_but_the_point_survives(rf, cfg, roof):
    # The three medians differ on purpose: over all three repeats it is 1.2, over
    # the two clean ones 1.1. A throttled repeat that happened not to move the
    # median would let a broken filter pass this test.
    timings = [_timing(rf, rep=1, ms=1.0), _timing(rf, rep=2, ms=1.2),
               _timing(rf, rep=3, ms=5.0, start=1500, end=1000)]
    points = rf.build_points(timings, cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=1500)
    assert len(points) == 1
    assert points[0].retained is True
    assert points[0].reps == 2, "the throttled repeat must not be averaged in"
    assert points[0].throttled_reps == 1
    assert points[0].ms_p50 == pytest.approx(1.1), \
        "1.2 is the median WITH the throttled repeat; 1.1 is without it"


def test_a_point_whose_every_repeat_throttled_leaves_the_gated_set(rf, cfg, roof):
    timings = [_timing(rf, rep=r, ms=2.0, start=1500, end=1000) for r in (1, 2)]
    points = rf.build_points(timings, cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=1500)
    assert points[0].retained is False
    assert "throttled" in points[0].excluded_why
    assert rf.multi_tile(points) == []


def test_a_cell_below_the_session_clock_floor_is_excluded(rf, cfg, roof):
    """Off-clock is not the same failure as throttled, and both must exclude."""
    cold = _timing(rf, rep=1, ms=2.0, start=1200, end=1200)
    assert cold.throttled is False, "a steady low clock does not drift"
    points = rf.build_points([cold], cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=1500)
    assert points[0].retained is False


def test_the_throttle_gate_refuses_when_no_clock_was_ever_read(rf, cfg, roof):
    """NON-VACUITY: a check that examined nothing also reports zero failures."""
    timings = [_timing(rf, rep=r, start=0, end=0) for r in (1, 2, 3)]
    points = rf.build_points(timings, cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=rf.modal_clock(timings))
    gate = rf.gate_v3_clocks(timings, points, roof, rf.modal_clock(timings))
    assert gate.passed is None, "zero throttled cells over zero clocks is not a PASS"
    assert "no clock was read" in gate.observed


def test_the_throttle_gate_refuses_when_the_roof_has_no_clock(rf, cfg, roof):
    """Without the roof's own clock the direction of the bias is unknowable."""
    from dataclasses import replace
    clockless = replace(roof, clock_mhz=0)
    timings = [_timing(rf, rep=r) for r in (1, 2, 3)]
    points = rf.build_points(timings, cfg, 128, clockless, sm_count=132,
                             block_n=64, clock_ref=1500)
    gate = rf.gate_v3_clocks(timings, points, clockless, 1500)
    assert gate.passed is None
    assert "records no GEMM clock" in gate.observed


def test_one_sagging_cell_does_not_kill_a_run_but_a_tenth_of_them_does(rf, cfg,
                                                                      roof):
    """The exclusion machinery exists so a gate does not have to fail on weather.

    Two cells in twenty is the boundary and is allowed; three is a box whose
    surviving medians are not trustworthy either.
    """
    def session(bad: int, total: int = 20):
        rows = [_timing(rf, rows=256, rep=r) for r in range(bad, total)]
        rows += [_timing(rf, rows=256, rep=r, ms=2.0, start=1500, end=1000)
                 for r in range(bad)]
        points = rf.build_points(rows, cfg, 128, roof, sm_count=132,
                                 block_n=64, clock_ref=1500)
        return rf.gate_v3_clocks(rows, points, roof, 1500)

    assert session(1).passed is True
    assert session(2).passed is True, "the threshold is inclusive at 10%"
    assert session(3).passed is False
    assert "10%" in session(3).rule
    assert "15.0%" in session(3).observed


def test_the_modal_clock_is_a_median_and_not_a_maximum(rf):
    """One high sample taken during a ramp must not exclude the whole run."""
    timings = ([_timing(rf, rep=r, start=1500, end=1500) for r in range(1, 6)]
               + [_timing(rf, rep=6, start=1980, end=1980)])
    assert rf.modal_clock(timings) == 1500


# --------------------------------------------------------------------------
# The denominator.
# --------------------------------------------------------------------------

def test_a_measured_run_with_no_calibration_for_its_device_refuses(rf):
    """The failure that put a stale H200 ridge into seven A100 reports."""
    import torch
    if torch.cuda.is_available():                       # pragma: no cover
        pytest.skip("this box has a device; the refusal is the off-GPU path")
    with pytest.raises(rf.RoofUnavailable, match="no calibration for this device"):
        rf.resolve_roof("bf16", synthetic=False)


def test_the_hypothesis_roof_is_reachable_only_for_a_synthetic_run(rf):
    import torch
    if torch.cuda.is_available():                       # pragma: no cover
        pytest.skip("this box has a device")
    got = rf.resolve_roof("bf16", synthetic=True)
    assert got.attached is False
    assert "HYPOTHESIS" in got.source
    assert got.tflops > 0, "REFUSE rather than default; never 0.0 for a ceiling"


def test_a_roof_this_card_did_not_measure_fails_v0(rf, roof):
    gate = rf.gate_v0_roof(roof)
    assert gate.passed is False
    assert "every fraction in this report" in gate.invalidates


def test_a_hypothesis_roof_can_never_reach_a_verdict(rf, roof):
    """However clean the claim gates look, an unquotable denominator stops there."""
    gates = [rf.gate_v0_roof(roof),
             rf.Gate(rf.CLAIM, "C1 roof", "", "", True, ""),
             rf.Gate(rf.CLAIM, "C2 plateau", "", "", True, ""),
             rf.Gate(rf.CLAIM, "C3 tile attribution", "", "", True, "")]
    call, why = rf.verdict(gates)
    assert call == rf.UNSETTLED
    assert "V0 roof provenance" in " ".join(why)


def test_the_roof_is_the_measured_rate_and_not_the_datasheet(rf, roof):
    """712 TFLOP/s measured against a 989 marketing figure: a 28% difference."""
    assert 600 < roof.tflops < 800
    assert roof.ridge == pytest.approx(roof.tflops * 1e12
                                       / (roof.bandwidth_gbps * 1e9), rel=1e-9)


# --------------------------------------------------------------------------
# The control.
# --------------------------------------------------------------------------

def test_the_subject_may_not_be_its_own_control(rf):
    why = rf.check_control(rf.SUBJECT_BLOCK_M, 0.558, 162.8, 2)
    assert "is the SUBJECT" in why


def test_a_control_with_less_headroom_than_the_subject_is_refused(rf):
    assert "below the subject" in rf.check_control(64, 0.558, 162.8, 2)


def test_a_control_whose_own_cap_binds_is_refused(rf):
    """A control whose ceiling is part of the argument settles nothing."""
    why = rf.check_control(256, 3.0, 162.8, 2)
    assert "under the 1.30x a control needs" in why


def test_the_default_control_clears_the_margin_on_both_cards(rf):
    for ridge in (145.8, 162.8):
        assert rf.check_control(rf.DEFAULT_CONTROL_BLOCK_M, 0.558, ridge, 2) == ""


def test_a_control_that_cannot_run_pinned_is_refused_before_any_gpu_time(rf, capsys):
    """The BN=256 accumulator: 256 registers per thread against a ceiling of 255.

    A spilled kernel still returns a time, that time still fits a line, and that
    line still qualified as this study's compute reference at 0.2% error.
    """
    code = rf.main(["--dry-run", "--block-n", "256", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert code == 2
    assert "REFUSED before any GPU time" in out
    assert "registers per thread" in out


# --------------------------------------------------------------------------
# The identity: the run id, and what resume is keyed on.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("knob,value", [
    ("model", "qwen2-57b-a14b"), ("dtype", "fp16"), ("control", 512),
    ("r_min", 64), ("r_max", 8192), ("reps", 5), ("iters", 200),
    ("warmup", 40), ("cell_budget_ms", 800.0), ("seed", 7), ("group_m", 16),
    ("block_n", 128), ("block_k", 32), ("num_stages", 3), ("num_warps", 4),
])
def test_every_swept_knob_is_in_the_run_id(rf, knob, value):
    """Twice-learned: a key that omits a swept knob is a silent overwrite."""
    base = rf.default_run_id(_args(rf), "nvidia_h200")
    other = rf.default_run_id(_args(rf, **{knob: value}), "nvidia_h200")
    assert base != other, f"--{knob.replace('_', '-')} does not reach the run id"


def test_the_card_is_in_the_run_id(rf):
    """The omission with the published proof: two cards, one report filename."""
    a = rf.default_run_id(_args(rf), "nvidia_h200")
    b = rf.default_run_id(_args(rf), "nvidia_a100_sxm4_80gb")
    assert a != b
    assert a.startswith("nvidia_h200") and b.startswith("nvidia_a100")


def test_alpha_is_not_in_the_run_id(rf):
    """It selects the control and prints a prediction; it re-reads one sweep."""
    assert (rf.default_run_id(_args(rf), "nvidia_h200")
            == rf.default_run_id(_args(rf, alpha=0.9), "nvidia_h200"))


def test_resume_is_keyed_on_rows_because_three_batches_share_one_tile(rf, tmp_path):
    """32, 64 and 128 rows per expert are all `tiles == 1` at BLOCK_M=128.

    A manifest keyed on the tile count would find the first present and skip the
    other two, reporting a quarter-full tile's throughput under a full one's
    label.
    """
    path = tmp_path / "cells.csv"
    for rows in (32, 64, 128):
        rf.append_timing(path, _timing(rf, rows=rows, rep=1))
    done, read = rf.read_timings(path)
    assert len({t.tiles for t in read}) == 1, "all three are one tile per expert"
    assert done == {(128, 32, 1), (128, 64, 1), (128, 128, 1)}


def test_a_failed_timing_is_not_recorded_as_done(rf, tmp_path):
    path = tmp_path / "cells.csv"
    bad = rf.Timing(128, 256, 2, 1024, 1, 0.0, 0.0, 0.0, 0, status="failed",
                    detail="boom")
    rf.append_timing(path, bad)
    done, read = rf.read_timings(path)
    assert read and done == set(), "a failed cell must be retried, not skipped"


# --------------------------------------------------------------------------
# Non-vacuity: the gates must be able to notice that nothing happened.
# --------------------------------------------------------------------------

def test_a_sweep_that_never_crossed_the_onset_fails_v2(rf, cfg):
    """Below the onset there is one tile per expert and no re-read term at all."""
    points = [_point(rf, rows=r) for r in (32, 64, 128)]
    gate = rf.gate_v2_non_vacuity(points, [], 5, onset_tokens_value=512)
    assert gate.passed is False
    assert "every claim gate" in gate.invalidates


def test_a_shallow_sweep_fails_v2_on_depth(rf):
    points = [_point(rf, rows=r) for r in (256, 512)]
    gate = rf.gate_v2_non_vacuity(points, points, 5, onset_tokens_value=512)
    assert gate.passed is False


def test_a_run_with_no_control_fails_v4(rf):
    subject = [_point(rf, rows=r) for r in (256, 512, 1024, 2048)]
    gate = rf.gate_v4_control_ran([], subject, 256)
    assert gate.passed is False
    assert "gate C3" in gate.invalidates


def test_c1_refuses_when_nothing_multi_tile_was_measured(rf, roof):
    plateau = rf.plateau_of([], doublings=2)
    gate = rf.gate_c1_roof([_point(rf, rows=64)], plateau, roof, (0.77, 1.0))
    assert gate.passed is None


def test_c3_refuses_when_the_two_tiles_share_no_batch(rf, roof):
    subject = [_point(rf, rows=r) for r in (256, 512, 1024)]
    control = [_point(rf, block_m=256, rows=r, tokens=r * 4 + 1)
               for r in (2048, 4096)]
    gate = rf.gate_c3_attribution(subject, control, rf.plateau_of(control,
                                                                 doublings=2),
                                  256, roof)
    assert gate.passed is None
    assert "share no multi-tile token count" in gate.observed


def test_c1_fails_when_the_subject_reaches_the_roof(rf, roof):
    """The other fork. A FAIL here is a result, not an error."""
    points = [_point(rf, rows=r, tflops=0.99 * roof.tflops,
                     roof_tflops=roof.tflops)
              for r in (256, 512, 1024, 2048, 4096)]
    gate = rf.gate_c1_roof(points, rf.plateau_of(points, doublings=2), roof,
                           (0.77, 1.0))
    assert gate.passed is False
    assert "REACHED the roof" in " ".join(gate.lines)


def test_both_tiles_plateauing_together_is_named_as_its_own_result(rf, roof):
    """If the control plateaus with the subject, the shortfall is the layer's."""
    subject = [_point(rf, rows=r, tflops=500.0, roof_tflops=roof.tflops)
               for r in (256, 512, 1024, 2048, 4096)]
    control = [_point(rf, block_m=256, rows=r, tflops=505.0,
                      roof_tflops=roof.tflops)
               for r in (256, 512, 1024, 2048, 4096)]
    gate = rf.gate_c3_attribution(subject, control,
                                  rf.plateau_of(control, doublings=2), 256, roof)
    assert gate.passed is False
    assert "BOTH TILES PLATEAU TOGETHER" in " ".join(gate.lines)
    call, _ = rf.verdict([rf.Gate(rf.CLAIM, "C1 roof", "", "", True, ""),
                          rf.Gate(rf.CLAIM, "C2 plateau", "", "", True, ""),
                          gate])
    assert call == rf.NOT_TILE


# --------------------------------------------------------------------------
# The picture and the figure data.
# --------------------------------------------------------------------------

def _series(rf, roof):
    subject = [_point(rf, rows=r, tflops=500.0, roof_tflops=roof.tflops)
               for r in (256, 1024, 4096)]
    subject.append(_point(rf, rows=2048, tflops=300.0, roof_tflops=roof.tflops,
                          retained=False))
    control = [_point(rf, block_m=256, rows=r, tflops=0.99 * roof.tflops,
                      roof_tflops=roof.tflops) for r in (256, 1024, 4096)]
    return [rf.Series("control", "o", control), rf.Series("subject", "#", subject)]


def test_the_roof_is_drawn_on_the_line_labelled_one(rf, roof):
    """An axis whose top row reads 1.02 and is dashed as the roof is read twice."""
    lines = rf.ascii_plot(_series(rf, roof), roof)
    marked = [ln for ln in lines if "<- the roof" in ln]
    assert len(marked) == 1
    assert marked[0].strip().startswith("1.00 |")
    assert "-" * 10 in marked[0]


def test_both_tiles_and_the_exclusions_are_visible_on_the_plot(rf, roof):
    body = "\n".join(rf.ascii_plot(_series(rf, roof), roof))
    assert "#" in body and "o" in body
    assert "x" in body, "an excluded point stays on the plot and out of the gates"
    assert "BLOCK_M=128" in body and "BLOCK_M=256" in body


def test_the_plot_never_ships_trailing_whitespace_or_runaway_width(rf, roof):
    lines = rf.ascii_plot(_series(rf, roof), roof)
    assert all(ln == ln.rstrip() for ln in lines[:-1])
    assert max(len(ln) for ln in lines) < 120


def test_a_point_above_the_roof_gets_headroom_rather_than_being_clipped(rf, roof):
    """Clocks can put a fraction over 1, and a clipped point hides that."""
    hot = [rf.Series("subject", "#",
                     [_point(rf, rows=r, tflops=1.15 * roof.tflops,
                             roof_tflops=roof.tflops) for r in (256, 1024)])]
    lines = rf.ascii_plot(hot, roof)
    top = lines[1].strip().split()[0]
    assert float(top) > 1.0
    assert any("<- the roof" in ln for ln in lines)


def test_the_figure_csv_carries_its_own_denominator_on_every_row(rf, roof, tmp_path):
    """A figure gets redrawn months later from a committed CSV.

    A fraction whose denominator is not in the file is a fraction that will be
    redrawn against whatever roof the plotting script happens to load.
    """
    rows = rf.figure_rows(_series(rf, roof), roof, "nvidia_h200")
    path = tmp_path / "figure.csv"
    rf.write_figure_csv(path, rows)
    back = list(csv.DictReader(path.open()))
    assert len(back) == len(rows)
    assert set(back[0]) == set(rf.FIGURE_FIELDS)
    for row in back:
        assert row["card"] == "nvidia_h200"
        assert float(row["roof_tflops"]) == pytest.approx(roof.tflops)
        assert row["roof_source"]
    assert any(row["retained"] == "False" for row in back)
    assert {row["regime"] for row in back} <= {"pre-onset", "onset", "multi-tile"}


def test_the_point_table_prints_every_point_including_the_excluded(rf, roof):
    table = rf.point_table(_series(rf, roof))
    assert len(table) == 1 + 7
    assert any("excluded" in line for line in table)


# --------------------------------------------------------------------------
# Paths.
# --------------------------------------------------------------------------

def test_git_visibility_knows_results_is_dropped_and_published_is_kept(rf):
    """This project has already lost every published plot to that pattern."""
    assert rf.git_visibility(ROOT / "results" / "bm128_roofline" / "x"
                             ).startswith("IGNORED")
    assert "keep" in rf.git_visibility(ROOT / "results" / "published" / "x.csv")


# --------------------------------------------------------------------------
# The CLI, off GPU.
# --------------------------------------------------------------------------

def test_there_is_no_block_m_flag(rf):
    """A --block-m would let a run answer a different question under this name."""
    with pytest.raises(SystemExit):
        rf.build_parser().parse_args(["--block-m", "64"])


def test_dry_run_prints_the_predictions_with_numbers_before_the_plan(rf, capsys):
    code = rf.main(["--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.index("Registered predictions") < out.index("## The plan")
    for tag in ("P1", "P2", "P3", "P4"):
        assert f"  {tag}  " in out
    assert "multi-tile onset    T > 512" in out
    assert "TFLOP/s" in out and "HYPOTHESIS" in out
    assert "Invocation for a session script" in out


def test_dry_run_names_the_path_it_would_write_and_whether_git_keeps_it(rf, capsys):
    rf.main(["--dry-run"])
    out = capsys.readouterr().out
    assert "WRITES TO" in out
    assert "IGNORED by git" in out or "git will keep" in out
    assert "figure.csv" in out


def test_the_self_test_passes_every_one_of_its_own_gates(rf, capsys):
    """The claim that these gates DISCRIMINATE, which is what --self-test is for."""
    code = rf.main(["--self-test", "--fail-on-gate"])
    out = capsys.readouterr().out
    assert code == 0, out[-3000:]
    assert "0 FAIL, 0 UNKNOWN" in out


def test_the_three_planted_worlds_reach_three_different_verdicts(rf, cfg, roof):
    lines, gates = rf.self_test(cfg, roof, 2, r_min=32, r_max=4096,
                                control_block_m=256, doublings=2)
    body = "\n".join(lines)
    for expected in (rf.BINDING, rf.NOT_BINDING, rf.STILL_RISING):
        assert expected in body
    named = {g.name for g in gates}
    assert "S discrimination" in named and "S throttle exclusion" in named
    assert all(g.passed is True for g in gates), \
        [g.name for g in gates if g.passed is not True]


def test_a_card_flag_may_not_contradict_an_attached_device(rf, capsys):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("the contradiction is only detectable with a device attached")
    code = rf.main(["--dry-run", "--card", "not_this_card"])  # pragma: no cover
    assert code == 2
    assert "may never contradict" in capsys.readouterr().out


def test_off_gpu_the_run_path_stops_before_measuring_and_says_where_to_look(
        rf, capsys):
    import torch
    if torch.cuda.is_available():                       # pragma: no cover
        pytest.skip("this box has a device")
    code = rf.main([])
    out = capsys.readouterr().out
    assert code == 2
    assert "--self-test" in out and "--dry-run" in out
