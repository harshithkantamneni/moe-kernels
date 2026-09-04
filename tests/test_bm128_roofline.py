"""The one measurement in this study that goes through no fit, and its refusals.

`scripts/bm128_roofline.py` forces BLOCK_SIZE_M=128, sweeps the batch across the
multi-tile onset, and divides achieved TFLOP/s by the attached card's own
measured dense bf16 rate. Nothing in its verdict comes from alpha, an anchor, an
estimator or a ladder, so most of this file is about the two things that CAN
still go wrong: the denominator belonging to another machine, and a derivative
read off noise.

TWELVE GROUPS. Four were added on 2026-09-02, one per audit finding, R5 on
2026-09-03 when the production arms turned out to be unrunnable, and R6 later
the same day when a re-run of every arm found that R5's own page argued with
itself and that its verdict never reached a line a driver reads.

  - THE GEOMETRY. Multi-tile onset is `T > BLOCK_M E / k`, computed per model and
    never hardcoded; the grid is a chain of doublings or it is refused, because
    "gain per doubling" is not a quantity on a grid that does not double.
  - THE DERIVATIVE. A plateau is a claim about one, so a flat curve, a rising
    curve, a curve too short to have a slope and a curve whose noise swamps the
    threshold must produce four different answers, and exactly one of them may
    be PASS.
  - THE EXCLUSIONS. A drifting repeat leaves the median; a point whose every
    repeat was excluded leaves the gated set and stays on the plot; a cell that
    ran steadily below the roof's clock is excluded although it never drifted;
    and a clock check over cells that never carried one reports UNKNOWN rather
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
  - R1, A CEILING NEEDS A CONTROL THAT REACHED IT. The headline used to be
    issued on a gap of 0.10 alone, with no requirement that the positive control
    ever go positive, and no tile in the published corpus reaches 0.95 of the
    roof. C4 gates it, TILE-DEPENDENT GAP / CEILING UNLOCATED is the third
    outcome, and the residency confound is named under every verdict rather than
    under the one it embarrasses.
  - R2, THE PRODUCTION CONFIGURATION. BLOCK_SIZE_N=256 with GROUP_SIZE_M 16 or
    32 is what vLLM ships for mixtral on the H200; it plans end to end, it is
    visible in the run id, and the plan states the outcome the already-published
    arms imply for whatever configuration was asked for, or says it cannot.
  - R3, ONE INSTRUMENT AND CLOCKS UNDER LOAD. The card idles at 1980 MHz and
    runs a dense GEMM at 1515, so a post-synchronise sample compared with the
    roof's clock was comparing two operating points through a number measured at
    neither.
  - R5, THE CONTROL THAT DOES NOT EXIST. At BLOCK_SIZE_N=256 no tile above the
    subject can be pinned: a 256x256 fp32 accumulator is the whole per-block
    register file at every warp count, which the sweep's PER-THREAD bill stops
    seeing above 8 warps, so the file that recommended --num-warps 16 was
    recommending a kernel no card can run. Each escape is refused on its own
    arithmetic -- stages are not the accumulator, warps divide it rather than
    shrink it, and a control at its own BLOCK_SIZE_N has EXACTLY the subject's
    cap because `ai_model.cap` is symmetric in the two tile dimensions. What
    survives is `--control none`, which can refute this study's headline and can
    never confirm it, and both directions of that are planted.
  - R6, WHAT R5 GOT WRONG, found by re-running every arm the way the session
    driver invokes it. The UNCONTROLLED verdict reached no `RESULT: ` line, so
    the arm the driver calls the claim exited 0 DONE with six PASSes; the
    control search priced its candidates with the scalar cap the same printed
    page called the trap, overstating the production tile by 3.7x; the finding
    naming BLOCK_SIZE_N=256 was printed at every BLOCK_SIZE_N, including one
    where a control fits; the escape it recommended is pre-registered to fail
    the gate that would confirm the claim; and the controlled arms' run id moved
    with nothing about them changing. The honest statement is stronger than the
    one R5 wrote: no arm reaches CEILING BINDING at ANY BLOCK_SIZE_N on sm_90,
    and that is computed here over 56 tiles rather than argued.
  - R4, THE APPARATUS. Every KernelTiming column reaches the cells and the
    figure, the provenance block reaches the report, the run id comes from the
    shared builder, the exit code comes from the shared table, every gate prints
    exactly one RESULT line, and the plan states a minimum detectable effect
    from a stated noise assumption.

The script is loaded by path, because `scripts/` is not a package.
"""
from __future__ import annotations

import csv
import importlib.util
import json
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
    assert max(rows) // 128 == 32, \
        "the observed arm reaches 32 M-tiles on mean rows (34 at the busiest expert)"


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

def _timing(rf, *, block_m=128, rows=256, rep=1, ms=1.0, load=1500.0,
            level=True, drift=True, clocks=True):
    """One row in the shape `time_kernel` writes.

    `load`, `level` and `drift` are the three things the instrument reports
    about the clock, and they are planted independently because they fail
    independently: a cell can drift without being cold and be cold without
    drifting. `clocks=False` is the container with no NVML, where all three are
    None and nothing may be excluded on any of them.
    """
    return rf.Timing(
        block_m, rows, max(1, -(-rows // block_m)), rows * 4, rep, ms, ms, 0.0,
        10, sm_clock_load_mhz=load if clocks else None,
        sm_clock_start_mhz=load if clocks else None,
        sm_clock_end_mhz=load if clocks else None,
        clock_level_ok=level if clocks else None,
        clock_drift_ok=drift if clocks else None,
        instrument="test/fake", warmup_ms=300.0, trials=3, l2_flush=True)


def test_a_drifting_repeat_leaves_the_median_but_the_point_survives(rf, cfg, roof):
    # The three medians differ on purpose: over all three repeats it is 1.2, over
    # the two clean ones 1.1. A throttled repeat that happened not to move the
    # median would let a broken filter pass this test.
    timings = [_timing(rf, rep=1, ms=1.0), _timing(rf, rep=2, ms=1.2),
               _timing(rf, rep=3, ms=5.0, drift=False)]
    points = rf.build_points(timings, cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=1500)
    assert len(points) == 1
    assert points[0].retained is True
    assert points[0].reps == 2, "the drifting repeat must not be averaged in"
    assert points[0].throttled_reps == 1
    assert points[0].ms_p50 == pytest.approx(1.1), \
        "1.2 is the median WITH the drifting repeat; 1.1 is without it"


def test_a_point_whose_every_repeat_was_excluded_leaves_the_gated_set(rf, cfg,
                                                                      roof):
    timings = [_timing(rf, rep=r, ms=2.0, drift=False) for r in (1, 2)]
    points = rf.build_points(timings, cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=1500)
    assert points[0].retained is False
    assert "excluded" in points[0].excluded_why
    assert rf.multi_tile(points) == []


def test_a_cell_below_the_roofs_clock_is_excluded_though_it_never_drifted(
        rf, cfg, roof):
    """R3. Off-clock is not the same failure as drifting, and both must exclude.

    The pre-2026-09-02 flag scored a DROP between two samples and nothing else,
    so a cell that sat steadily at 1200 MHz while the roof was measured at 1515
    passed a check named for the exact fault it had.
    """
    cold = _timing(rf, rep=1, ms=2.0, load=1200.0, level=False)
    assert cold.throttled is False, "a steady low clock does not drift"
    assert cold.cold is True and cold.excluded is True
    points = rf.build_points([cold], cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=1500)
    assert points[0].retained is False
    assert "below" in points[0].excluded_why


def test_a_clock_verdict_of_none_excludes_nothing(rf, cfg, roof):
    """None means NOT DETERMINED, and an exclusion has to be established."""
    unknown = _timing(rf, rep=1, clocks=False)
    assert unknown.throttled is False and unknown.cold is False
    points = rf.build_points([unknown], cfg, 128, roof, sm_count=132,
                             block_n=64, clock_ref=0)
    assert points[0].retained is True, \
        "a row whose clock was never read is unknown, not bad"


def test_the_clock_gate_refuses_when_no_clock_was_ever_read(rf, cfg, roof):
    """NON-VACUITY: a check that examined nothing also reports zero failures."""
    timings = [_timing(rf, rep=r, clocks=False) for r in (1, 2, 3)]
    points = rf.build_points(timings, cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=rf.modal_clock(timings))
    gate = rf.gate_v3_clocks(timings, points, roof, rf.modal_clock(timings))
    assert gate.passed is None, "zero throttled cells over zero clocks is not a PASS"
    assert "no under-load clock sample landed" in gate.observed


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
        rows += [_timing(rf, rows=256, rep=r, ms=2.0, drift=False)
                 for r in range(bad)]
        points = rf.build_points(rows, cfg, 128, roof, sm_count=132,
                                 block_n=64, clock_ref=1500)
        return rf.gate_v3_clocks(rows, points, roof, 1500)

    assert session(1).passed is True
    assert session(2).passed is True, "the threshold is inclusive at 10%"
    assert session(3).passed is False
    assert "10%" in session(3).rule
    assert "15.0%" in session(3).observed


def test_the_session_clock_gate_counts_cold_cells_too(rf, cfg, roof):
    """Both faults spend the same budget: either at a tenth of the cells fails."""
    rows = [_timing(rf, rows=256, rep=r) for r in range(3, 20)]
    rows += [_timing(rf, rows=256, rep=r, load=1200.0, level=False)
             for r in range(3)]
    points = rf.build_points(rows, cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=1500)
    gate = rf.gate_v3_clocks(rows, points, roof, 1500)
    assert gate.passed is False
    assert "15.0%" in gate.observed


def test_the_modal_clock_is_a_median_of_under_load_samples(rf):
    """R3. A median, so one ramp sample cannot exclude the run; and under load,
    because the idle instant reads the card's other operating point."""
    timings = ([_timing(rf, rep=r, load=1500.0) for r in range(1, 6)]
               + [_timing(rf, rep=6, load=1980.0)])
    assert rf.modal_clock(timings) == 1500
    assert all(t.sm_clock_load_mhz is not None for t in timings)


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

def _check(rf, cfg, control, *, block_n=64, alpha=0.558, ridge=162.8, fits=()):
    return rf.check_control(cfg, control, alpha, ridge, 2, block_n=block_n,
                            fits=list(fits))


def test_the_subject_may_not_be_its_own_control(rf, cfg):
    assert "is the SUBJECT" in _check(rf, cfg, rf.SUBJECT_BLOCK_M)


def test_a_control_with_less_headroom_than_the_subject_is_refused(rf, cfg):
    assert "below the subject" in _check(rf, cfg, 64)


def test_a_control_with_no_headroom_on_the_subject_is_refused(rf, cfg):
    """The refusal alternative 4 names, run through the function that makes it.

    `ai_model.cap` is symmetric in BLOCK_M and BLOCK_SIZE_N, so a tile that buys
    its M back out of its N buys nothing at all. The subject at BLOCK_SIZE_N=256
    and a 256x128 control have the same cap to the last decimal, and a control
    with the subject's own cap makes C3's difference zero by construction. The
    scalar form the search used to price with has no BLOCK_SIZE_N in it and
    would have called that control twice the headroom.
    """
    from moe.bench import ai_model
    subject = ai_model.cap(cfg.intermediate_size, cfg.hidden_size, block_m=128,
                           block_n=256, alpha_b=0.558, alpha_a=0.558, b=2)
    control = ai_model.cap(cfg.intermediate_size, cfg.hidden_size, block_m=256,
                           block_n=128, alpha_b=0.558, alpha_a=0.558, b=2)
    assert control == pytest.approx(subject, rel=1e-12)
    # And the function refuses a control whose cap does not clear the subject's,
    # exercised through a model where it can happen at a shared BLOCK_SIZE_N.
    assert rf.symmetric_cap(cfg, 256, 64, 0.558, 2) > \
        rf.symmetric_cap(cfg, 128, 64, 0.558, 2), \
        "at a shared BLOCK_SIZE_N a larger M always buys some headroom"


def test_the_cap_margin_no_longer_refuses_because_it_could_only_ever_refuse(
        rf, cfg):
    """R6. The gate that ran on the model the same page called wrong.

    `check_control` used to refuse a control whose cap was under 1.30x the
    ridge, computed with `SWEEP.ai_cap` = 2 BM / (alpha b). Priced with
    `symmetric_cap` the default control is 0.55x the ridge at BLOCK_SIZE_N=64,
    not 2.82x, and NO tile a block can hold clears the ridge at all -- so a
    refusal on that margin would refuse every geometry this study can run. A
    check that can only fail decides as little as one that can only pass, so the
    margin became a disclosure and the refusals are the register file and
    headroom. The arm is not cancelled for the answer C4 is expected to give.
    """
    scalar = rf.SWEEP.ai_cap(rf.DEFAULT_CONTROL_BLOCK_M, 0.558, 2)
    symmetric = rf.symmetric_cap(cfg, rf.DEFAULT_CONTROL_BLOCK_M, 64, 0.558, 2)
    assert scalar / 162.8 > 2.8 and symmetric / 162.8 < 0.6, \
        "the two models must still disagree, or this test proves nothing"
    assert symmetric < 162.8 * rf.CONTROL_CAP_MARGIN
    assert _check(rf, cfg, rf.DEFAULT_CONTROL_BLOCK_M) == "", \
        "a control under the margin is admitted, and C4 is registered to fail"


def test_no_arm_reaches_the_headline_at_any_block_n_on_this_card(rf, cfg):
    """THE FINDING, computed rather than written down.

    C4 requires the control to reach the roof, and a tile reaches a compute roof
    only if its cap clears the ridge. Over 56 power-of-two tiles up to
    2048x1024, on both models this study measures and at every alpha it has
    measured, not one tile whose cap clears the H200 ridge fits the per-block
    register file: the smallest accumulator among them is exactly the whole
    file. So CEILING BINDING is unreachable by any arm at any BLOCK_SIZE_N, not
    only at 256, and the finding that named 256 was true and too weak.
    """
    for name in ("mixtral-8x7b", "qwen2-57b-a14b"):
        model = MODEL_CONFIGS[name]
        for alpha in (0.558, 0.625, 1.0):
            reach, buildable = rf.ridge_reaching_tiles(model, 162.8, alpha, 2)
            assert reach, "no tile clears the ridge at all: the sweep is wrong"
            assert buildable == [], f"{name} at alpha {alpha}"
            assert min(t[3] for t in reach) >= rf.REGISTERS_PER_BLOCK
    lines = rf.binding_reachability(cfg, 162.8, 0.558, 2)
    assert any("AT ANY BLOCK_SIZE_N" in line for line in lines)
    assert any(rf.GAP_UNLOCATED in line for line in lines)


def test_the_predicted_separation_is_clamped_at_the_roof(rf, cfg):
    """A cap of twice the ridge does not predict twice the roof.

    The same clamp `predicted_plateau_band` applies. Unclamped, the 1024x256
    candidate printed a predicted separation of +1.163 of the roof -- more
    throughput than the card has -- in the line an operator reads to decide
    whether the control is worth its minutes.
    """
    huge = rf.symmetric_cap(cfg, 1024, 256, 0.558, 2)
    subject = rf.symmetric_cap(cfg, 128, 256, 0.558, 2)
    assert huge / 162.8 > 2.0, "the unclamped ratio must exceed 1 or nothing is proved"
    assert rf.predicted_separation(huge, subject, 162.8) <= 1.0
    assert rf.predicted_separation(huge, subject, 162.8) == pytest.approx(
        1.0 - min(subject / 162.8, 1.0))


def test_an_alpha_outside_zero_to_one_is_refused_and_not_priced(rf, capsys):
    """A miss fraction above 1 is more traffic than a full re-read.

    `ai_model` refuses it; the scalar form accepted `--alpha 3.0` and returned a
    cap of 17 Op/B without a word, and that number then decided which tile could
    be a control.
    """
    code = rf.main(["--dry-run", "--alpha", "3.0"])
    out = capsys.readouterr().out
    assert code == 2
    assert "is outside [0, 1]" in out
    assert "MISS FRACTION" in out


def test_no_control_is_asked_of_a_run_that_declared_it_has_none(rf, cfg):
    """`--control none` has no control to check, and says so by returning empty.

    The PASS half of the gate above: a function that refused everything would
    also refuse the subject's own tile, and this one has to let exactly one
    thing through. At BLOCK_SIZE_N=256 the search finds nothing, which is the
    condition the mode exists for.
    """
    assert _check(rf, cfg, None, block_n=256, fits=()) == ""


def test_dropping_the_control_where_one_fits_is_refused(rf, cfg, capsys):
    """R6. REFUSE rather than default, at the geometry the caveat lies about.

    `--control none` says, under its verdict and in every plan, that the
    hardware left this arm no control. That is true at BLOCK_SIZE_N=256 and
    false at 64, where the search names eighteen pins -- so an operator could
    drop V4, C3 and C4 by choice and receive a page telling them the register
    file forced it. The search's own answer now decides.
    """
    why = _check(rf, cfg, None, block_n=64, fits=[(256, 8, 5)])
    assert "where a control DOES fit" in why
    assert "Pass --control 256" in why
    code = rf.main(["--dry-run", "--block-n", "64", "--group-m", "1",
                    "--control", "none", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert code == 2, out[-2000:]
    assert "--control none at BLOCK_SIZE_N=64, where a control DOES fit" in out
    assert "A CONTROL DOES FIT AT BLOCK_SIZE_N=64" in out
    assert "at BLOCK_SIZE_N=256 there is no positive control" not in out


def test_the_finding_is_not_printed_at_a_block_n_it_is_false_at(rf, cfg):
    """It was a module constant and it named 256 at every BLOCK_SIZE_N.

    One page printed twelve `FITS at --num-warps 8` lines for a BLOCK_M=256
    control at BLOCK_SIZE_N=64 and then, sixty lines later, "at BLOCK_SIZE_N=256
    there is no positive control". The search said a control exists and the
    paragraph under it said none does.
    """
    fits, _ = rf.control_feasibility(cfg, block_n=64, block_k=64, dtype_bytes=2,
                                     capability=(9, 0), alpha=0.558, ridge=162.8)
    assert fits, "a control does fit at BLOCK_SIZE_N=64"
    present = rf.no_control_finding(cfg, 162.8, 0.558, 2, block_n=64, fits=fits,
                                    capability=(9, 0))
    assert "A CONTROL DOES FIT AT BLOCK_SIZE_N=64" in present[0]
    assert "THE STUDY CANNOT CONFIRM ITS HEADLINE" not in " ".join(present)

    absent = rf.no_control_finding(cfg, 162.8, 0.558, 2, block_n=256, fits=[],
                                   capability=(9, 0))
    assert "THE STUDY CANNOT CONFIRM ITS HEADLINE" in absent[0]
    assert "at BLOCK_SIZE_N=256 there is no positive control" in absent[0]

    # And off a device neither sentence may be spoken: the shared-memory half of
    # the bill has no limit to check against, so an empty `fits` is not evidence.
    undecided = rf.no_control_finding(cfg, 162.8, 0.558, 2, block_n=64, fits=[],
                                      capability=None)
    assert "UNDECIDABLE FROM HERE" in undecided[0]
    assert "THE STUDY CANNOT CONFIRM ITS HEADLINE" not in " ".join(undecided)

    # But the BLOCK_SIZE_N=256 refusal is architecture-wide and needs no flag,
    # and the driver's own dry run of that arm passes none. A head that went
    # undecidable there would make this file's strongest sentence conditional
    # on something nothing supplies.
    laptop = rf.no_control_finding(cfg, 162.8, 0.558, 2, block_n=256, fits=[],
                                   capability=None)
    assert "THE STUDY CANNOT CONFIRM ITS HEADLINE" in laptop[0]
    assert "needs no --capability" in laptop[0]


def test_the_escape_the_finding_recommends_is_priced_and_not_oversold(rf, cfg):
    """Alternative 3 promised a confirmation it cannot deliver.

    "The subject at BLOCK_SIZE_N=128, where a BLOCK_M=256 control fits" is
    buildable and it does buy headroom -- 147.4 Op/B against 111.6, unlike
    alternative 4's transposed tile which buys exactly none. What it cannot buy
    is C4: 147.4 is 0.91x the ridge, so that control is memory bound by
    construction and the arm lands GAP_UNLOCATED. The handoff called it "the one
    to schedule if the owner wants an arm that CAN reach BINDING", and it is not
    that arm; no arm is.
    """
    text = " ".join(rf.no_control_finding(cfg, 162.8, 0.558, 2, block_n=256,
                                          fits=[], capability=(9, 0)))
    control = rf.symmetric_cap(cfg, 256, 128, 0.558, 2)
    subject = rf.symmetric_cap(cfg, 128, 128, 0.558, 2)
    assert control > subject, "alternative 3 does buy headroom"
    assert control < 162.8, "and it cannot reach the ridge"
    assert "IT STILL DOES NOT CONFIRM THE HEADLINE" in text
    assert rf.GAP_UNLOCATED in text
    assert "NONE OF THE FIVE CONFIRMS THE HEADLINE ON THIS CARD" in text


def test_a_control_that_cannot_run_pinned_is_refused_before_any_gpu_time(rf, capsys):
    """The BN=256 accumulator, billed per BLOCK and not only per thread.

    A spilled kernel still returns a time, that time still fits a line, and that
    line still qualified as this study's compute reference at 0.2% error.
    """
    code = rf.main(["--dry-run", "--block-n", "256", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert code == 2
    assert "REFUSED:" in out
    assert "65536 32-bit registers per thread block" in out


# --------------------------------------------------------------------------
# The identity: the run id, and what resume is keyed on.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("knob,value", [
    ("model", "qwen2-57b-a14b"), ("dtype", "fp16"), ("control", 512),
    ("r_min", 64), ("r_max", 8192), ("reps", 5), ("trials", 5),
    ("warmup", 40.0), ("no_l2_flush", True), ("cell_budget_ms", 800.0),
    ("seed", 7), ("group_m", 16), ("block_n", 128), ("block_k", 32),
    ("num_stages", 3), ("num_warps", 4),
])
def test_every_swept_knob_is_in_the_run_id(rf, knob, value):
    """Twice-learned: a key that omits a swept knob is a silent overwrite."""
    base = rf.default_run_id(_args(rf), "nvidia_h200")
    other = rf.default_run_id(_args(rf, **{knob: value}), "nvidia_h200")
    assert base != other, f"--{knob.replace('_', '-')} does not reach the run id"


@pytest.mark.parametrize("knob,value", [("block_n", 256), ("group_m", 16),
                                        ("control", 512), ("num_stages", 3)])
def test_the_production_knobs_are_visible_in_the_run_id_not_only_hashed(
        rf, knob, value):
    """R2. The driver's second arm varies these, and `ls` has to show which.

    The hash separates every run; the VISIBLE prefix is what an operator reads
    off a network volume six directories deep, and a knob that only reaches the
    hash is a knob nobody can see was set.
    """
    got = rf.default_run_id(_args(rf, **{knob: value}), "nvidia_h200")
    letter = {"block_n": "n", "group_m": "g", "control": "c",
              "num_stages": "s"}[knob]
    assert f"-{letter}{value}-" in got, got


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


def test_every_planted_world_reaches_a_different_verdict(rf, cfg, roof):
    lines, gates = rf.self_test(cfg, roof, 2, r_min=32, r_max=4096,
                                control_block_m=256, doublings=2)
    body = "\n".join(lines)
    for expected in (rf.BINDING, rf.NOT_BINDING, rf.STILL_RISING, rf.NOT_TILE,
                     rf.GAP_UNLOCATED):
        assert expected in body
    named = {g.name for g in gates}
    assert {"S discrimination", "S drift exclusion",
            "S level exclusion"} <= named
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


# --------------------------------------------------------------------------
# R1. A ceiling needs a control that reached it.
#
# The audit's finding: `gate_c3_attribution` passed on a gap of 0.10 alone and
# `verdict` issued the study's headline on C1+C2+C3, so "the ceiling is real and
# it binds" could be printed with the subject at 0.58 and the control at 0.70,
# both stopped far below the roof for reasons this study's model does not
# describe. No tile in the published corpus reaches 0.95 of ridge x bandwidth,
# so that is not a hypothetical world; it is the one the pod arm will find.
# --------------------------------------------------------------------------

def _pair(rf, roof, *, subject_frac, control_frac, rows=(256, 512, 1024, 2048,
                                                         4096)):
    """A subject and a control curve at stated fractions of the roof, flat."""
    subject = [_point(rf, rows=r, tflops=subject_frac * roof.tflops,
                      roof_tflops=roof.tflops) for r in rows]
    control = [_point(rf, block_m=256, rows=r,
                      tflops=control_frac * roof.tflops,
                      roof_tflops=roof.tflops) for r in rows]
    return subject, control


def test_c4_fails_when_the_control_never_reached_the_roof(rf, roof):
    """The world the published corpus describes: 0.42 against 0.55."""
    _, control = _pair(rf, roof, subject_frac=0.42, control_frac=0.55)
    gate = rf.gate_c4_ceiling_located(control, 256, roof)
    assert gate.passed is False
    assert "0.55" in gate.observed
    assert rf.GAP_UNLOCATED in " ".join(gate.lines)


def test_c4_passes_when_the_control_reaches_the_roof(rf, roof):
    """The FAIL branch above means nothing without this one."""
    _, control = _pair(rf, roof, subject_frac=0.42, control_frac=0.99)
    gate = rf.gate_c4_ceiling_located(control, 256, roof)
    assert gate.passed is True


def test_c4_refuses_rather_than_passing_when_the_control_did_not_run(rf, roof):
    gate = rf.gate_c4_ceiling_located([], 256, roof)
    assert gate.passed is None, "no control is not a control that reached it"


def test_a_gap_without_a_reached_roof_is_not_the_headline(rf, roof):
    """R1. The verdict the audit says the arm should reach, named as its own."""
    subject, control = _pair(rf, roof, subject_frac=0.42, control_frac=0.55)
    gates = [rf.gate_c1_roof(subject, rf.plateau_of(subject, doublings=2), roof,
                             (0.77, 1.0)),
             rf.gate_c2_plateau(rf.plateau_of(subject, doublings=2), subject),
             rf.gate_c3_attribution(subject, control,
                                    rf.plateau_of(control, doublings=2), 256,
                                    roof),
             rf.gate_c4_ceiling_located(control, 256, roof)]
    assert [g.passed for g in gates] == [True, True, True, False]
    call, why = rf.verdict(gates)
    assert call == rf.GAP_UNLOCATED
    assert "may quote the gap and may not quote a ceiling" in " ".join(why)


def test_the_same_gap_with_a_reached_roof_is_the_headline(rf, roof):
    """And the PASS branch, or the test above only proves nothing passes."""
    subject, control = _pair(rf, roof, subject_frac=0.80, control_frac=0.99)
    gates = [rf.gate_c1_roof(subject, rf.plateau_of(subject, doublings=2), roof,
                             (0.77, 1.0)),
             rf.gate_c2_plateau(rf.plateau_of(subject, doublings=2), subject),
             rf.gate_c3_attribution(subject, control,
                                    rf.plateau_of(control, doublings=2), 256,
                                    roof),
             rf.gate_c4_ceiling_located(control, 256, roof)]
    assert all(g.passed for g in gates)
    call, _ = rf.verdict(gates)
    assert call == rf.BINDING


def test_the_headline_may_not_be_issued_when_c4_was_never_scored(rf):
    """An unscored positive control is not a passed one."""
    call, why = rf.verdict([rf.Gate(rf.CLAIM, "C1 roof", "", "", True, ""),
                            rf.Gate(rf.CLAIM, "C2 plateau", "", "", True, ""),
                            rf.Gate(rf.CLAIM, "C3 tile attribution", "", "",
                                    True, "")])
    assert call == rf.UNSETTLED
    assert "was not scored" in " ".join(why)


@pytest.mark.parametrize("verdict_name", ["BINDING", "NOT_BINDING",
                                          "STILL_RISING", "NOT_TILE",
                                          "GAP_UNLOCATED", "UNSETTLED"])
def test_the_residency_confound_is_named_under_every_verdict(rf, roof,
                                                             verdict_name):
    """R1. It was named under NOT_TILE alone, which is the branch it embarrasses.

    Each world below is built to land on one verdict, and every one of them has
    to carry the sentence: a gap can be residency rather than tile, and an
    absent gap can be residency cancelling a tile effect.
    """
    def gates_for(name):
        if name == "UNSETTLED":
            return [rf.gate_v0_roof(roof)]
        table = {"BINDING": (0.80, 0.99), "NOT_BINDING": (0.99, 0.99),
                 "STILL_RISING": None, "NOT_TILE": (0.55, 0.56),
                 "GAP_UNLOCATED": (0.42, 0.55)}
        if name == "STILL_RISING":
            # Climbing steeply and still under the roof at the deepest point,
            # so C1 PASSES and it is the DERIVATIVE that refuses the word.
            subject = [_point(rf, rows=r, tflops=0.10 * roof.tflops * 1.5 ** i,
                              roof_tflops=roof.tflops)
                       for i, r in enumerate((256, 512, 1024, 2048, 4096))]
            control = [_point(rf, block_m=256, rows=r, tflops=0.9 * roof.tflops,
                              roof_tflops=roof.tflops)
                       for r in (256, 512, 1024, 2048, 4096)]
        else:
            subject, control = _pair(rf, roof, subject_frac=table[name][0],
                                     control_frac=table[name][1])
        sub_p = rf.plateau_of(subject, doublings=2)
        ctl_p = rf.plateau_of(control, doublings=2)
        return [rf.gate_c1_roof(subject, sub_p, roof, (0.77, 1.0)),
                rf.gate_c2_plateau(sub_p, subject),
                rf.gate_c3_attribution(subject, control, ctl_p, 256, roof),
                rf.gate_c4_ceiling_located(control, 256, roof)]

    call, why = rf.verdict(gates_for(verdict_name))
    assert call == getattr(rf, verdict_name), f"world landed on {call}"
    assert "OCCUPANCY" in " ".join(why)
    assert "occupancy_vs_swizzle" in " ".join(why)


def test_the_confound_is_beside_the_gap_on_a_c3_pass_as_well_as_a_fail(rf, roof):
    """It is on the PASS that the gap becomes a number somebody quotes."""
    for subject_frac, control_frac in ((0.42, 0.55), (0.55, 0.56)):
        subject, control = _pair(rf, roof, subject_frac=subject_frac,
                                 control_frac=control_frac)
        gate = rf.gate_c3_attribution(subject, control,
                                      rf.plateau_of(control, doublings=2), 256,
                                      roof)
        assert "OCCUPANCY" in " ".join(gate.lines)


def test_the_self_test_plants_the_world_where_the_gap_is_not_the_tiles(rf):
    """The audit's acceptance check, run as it is written.

    `--self-test | grep -c NOT_TILE` was zero: the (C1 PASS, C2 PASS, C3 FAIL)
    world was never planted, so the branch that reports "the shortfall is the
    layer's" had never been executed by anything.
    """
    quads = {(w[4], w[5]) for w in rf.SELF_TEST_WORLDS}
    assert ("C3", False) in quads, "the (P, P, F) world is not planted"
    assert ("C4", False) in quads, "the unreached-roof world is not planted"
    assert {w[6] for w in rf.SELF_TEST_WORLDS} == {
        rf.BINDING, rf.NOT_BINDING, rf.STILL_RISING, rf.NOT_TILE,
        rf.GAP_UNLOCATED}, "every verdict branch must be reached by some world"


def test_the_self_test_output_names_both_new_outcomes_by_their_identifier(
        rf, capsys):
    """`--self-test | grep -c NOT_TILE` >= 1, literally, plus the third outcome."""
    rf.main(["--self-test"])
    out = capsys.readouterr().out
    assert out.count("NOT_TILE") >= 1
    assert out.count("GAP_UNLOCATED") >= 1


# --------------------------------------------------------------------------
# R2. The production configuration, and an arm whose result is already known.
# --------------------------------------------------------------------------

def test_the_production_swizzle_plans_end_to_end_without_a_control(rf, capsys):
    """R2/R5. BLOCK_SIZE_N=256 with GROUP_SIZE_M 16 and 32 is what vLLM ships.

    It planned end to end at `--num-warps 16 --num-stages 3` until 2026-09-03,
    and that plan was for a kernel no card can run: the 256x256 accumulator is
    the whole per-block register file at every warp count, and the per-thread
    bill stops seeing it above 8 warps. The production geometry now plans only
    with the control it can actually have, which is none.
    """
    for group_m in ("16", "32"):
        code = rf.main(["--dry-run", "--block-n", "256", "--group-m", group_m,
                        "--control", "none", "--capability", "9.0"])
        out = capsys.readouterr().out
        assert code == 0, out[-2000:]
        assert f"'GROUP_SIZE_M': {group_m}" in out
        assert "'BLOCK_SIZE_N': 256" in out
        assert "BLOCK_M= 128" in out and "ok" in out
        assert "REFUTE the claim and can never confirm it" in out


def test_the_production_geometry_refuses_a_control_and_prints_the_arithmetic(
        rf, capsys):
    """REFUSE rather than default, and then show the bill rather than a guess.

    A refusal that ends in "change --num-stages, --block-n or --control" sends
    an operator to guess at a bill this file can compute for nothing, and the
    bill says none of those three changes anything.
    """
    code = rf.main(["--dry-run", "--block-n", "256", "--group-m", "16",
                    "--control", "256", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert code == 2
    assert "65536 32-bit registers per thread block" in out
    assert "REFUSED AT EVERY WARP AND STAGE COUNT" in out
    assert "8 warps = 256 reg/thread" in out
    assert "16 warps = 128 reg/thread" in out
    assert "NO TILE ABOVE BLOCK_M=128 CAN BE PINNED AT BLOCK_SIZE_N=256" in out


def test_the_refusal_never_names_a_pin_that_does_not_exist(rf, capsys):
    """The bug this replaced, pinned so it cannot come back.

    `control_resource_hint` printed "THE REQUESTED CONFIGURATION DOES FIT, at
    --num-warps 16 --num-stages 3" for 256x256 on sm_90, in the one sentence an
    operator was meant to act on. 16 warps do fit the PER-THREAD ceiling: 128
    registers each against 255. The thread block still asks for all 65536
    registers in the file and has none left for a pointer, and no session that
    followed that advice would have produced a comparable timing.
    """
    for extra in ([], ["--capability", "9.0"]):
        rf.main(["--dry-run", "--block-n", "256", "--group-m", "16",
                 "--control", "256"] + extra)
        out = capsys.readouterr().out
        assert "DOES FIT" not in out
        assert "FITS at --num-warps" not in out


def test_the_finding_names_what_would_confirm_the_claim(rf, capsys):
    """A refusal whose consequence is unstated gets read as a warning.

    The consequence here is that the study's headline is unreachable at the
    configuration production runs, so the refusal has to say that in those words
    and then say what would settle it.
    """
    rf.main(["--dry-run", "--block-n", "256", "--group-m", "16",
             "--control", "256", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert ("THE STUDY CANNOT CONFIRM ITS HEADLINE AT vLLM'S SHIPPED "
            "CONFIGURATION ON sm_90.") in out
    assert "WHAT WOULD CONFIRM IT" in out
    for n in range(1, 6):
        assert f"\n  {n}. " in out, f"alternative {n} is not listed"


def test_an_uncontrolled_plan_carries_the_arithmetic_that_says_why(rf, capsys):
    """The plan an operator buys must contain the reason it has no control.

    Not only the refusal: the run that GOES AHEAD is the one whose report gets
    quoted, so the search and the finding ride in its plan too.
    """
    code = rf.main(["--dry-run", "--block-n", "256", "--group-m", "16",
                    "--control", "none", "--capability", "9.0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "CONTROL SEARCH at BLOCK_SIZE_N=256" in out
    assert "65536" in out
    assert "227 KiB" in out
    assert "THE STUDY CANNOT CONFIRM ITS HEADLINE" in out


def test_the_plan_predicts_the_outcome_the_published_arms_already_imply(rf, cfg,
                                                                       roof):
    """R2 / A8. The scheduled arm at BLOCK_SIZE_N=64, GROUP_SIZE_M=1 is not an
    experiment: its subject peaks near 0.47 and its control near 0.53 in arms
    already committed, a gap under CONTROL_SEPARATION, so the verdict is
    predictable and the operator is entitled to read that before renting a pod.
    """
    arm, why = rf.published_prediction(cfg, roof, block_n=64, group_m=1,
                                       control_block_m=256)
    assert arm is not None, why
    assert 0.44 < arm.subject_peak < 0.50
    assert 0.49 < arm.control_peak < 0.55
    assert arm.gap < rf.CONTROL_SEPARATION
    assert arm.outcome() == rf.NOT_TILE
    assert "report.json" in why


def test_the_prediction_says_how_many_arms_matched_and_over_what_span(rf, cfg,
                                                                     roof):
    """TWO arms match the headline configuration and only one is quoted.

    alpha-surface-s4 reads 0.468 / 0.526, a gap of 0.058, and cross-card-s3
    reads 0.482 / 0.502, a gap of 0.019. The line an operator uses to decide the
    arm is not worth renting used to call the winner "the published arm", which
    is a uniqueness claim, and the selection rule maximises the subject peak,
    which is the SMALLEST gap when the controls sit close. Both predict NOT_TILE
    today, so the bias costs nothing yet; the count and the span are printed so
    a later corpus that disagrees cannot disagree silently.
    """
    arm, why = rf.published_prediction(cfg, roof, block_n=64, group_m=1,
                                       control_block_m=256)
    assert "2 matching" in why, why
    assert "+0.019" in why and "+0.058" in why, why
    assert "HIGHEST subject peak" in why and "SMALLEST gap" in why, why
    assert "the published arm at" not in why, \
        "one of two matching arms was described as the only one"
    lines = " ".join(rf.prior_arm_lines(cfg, roof, block_n=64, group_m=1,
                                        control_block_m=256))
    assert "2 matching" in lines, "the plan must carry the disclosure too"


def test_a_single_matching_arm_is_named_as_the_only_one(rf, cfg, roof, tmp_path):
    """The other branch of the same sentence. A corpus with ONE match must not
    be described with a count and a span it does not have."""
    only = tmp_path / "2026-09-02-nvidia_h200-one-arm"
    only.mkdir()
    (only / "x.report.json").write_text(json.dumps({
        "model": cfg.name, "fixed": {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1},
        "ladder": {"128": {"points": [[8, 8.66]]},
                   "256": {"points": [[4, 8.0]]}}}))
    arm, why = rf.published_prediction(cfg, roof, block_n=64, group_m=1,
                                       control_block_m=256,
                                       published_dir=tmp_path)
    assert arm is not None, why
    assert "the ONE published arm at" in why, why
    assert "matching model=" not in why and "HIGHEST" not in why, why


def test_the_plan_says_it_cannot_predict_when_the_corpus_has_no_control_ladder(
        rf, cfg, roof):
    """R2. The honest branch, and the one the production swizzle actually hits.

    No arm in the corpus carries a BLOCK_M=256 ladder at BLOCK_SIZE_N=256,
    because that tile cannot be pinned there. Half a table is not a prediction.
    """
    arm, why = rf.published_prediction(cfg, roof, block_n=256, group_m=1,
                                       control_block_m=256)
    assert arm is None or arm.control_peak is None
    assert "cannot be predicted" in why
    lines = " ".join(rf.prior_arm_lines(cfg, roof, block_n=256, group_m=1,
                                        control_block_m=256))
    assert "NOT MEASURED" in lines or "not predictable" in lines.lower()


def test_a_configuration_the_corpus_never_ran_is_named_as_unpredictable(rf, cfg,
                                                                        roof):
    arm, why = rf.published_prediction(cfg, roof, block_n=64, group_m=7,
                                       control_block_m=256)
    assert arm is None
    assert "no published arm matches" in why


def test_the_prediction_never_divides_one_cards_ladder_by_anothers_roof(
        rf, cfg, roof, tmp_path):
    """The defect that put a stale H200 ridge into seven A100 reports."""
    other = tmp_path / "2026-09-02-nvidia_a100_sxm4_80gb-arm"
    other.mkdir()
    (other / "x.report.json").write_text(json.dumps({
        "model": cfg.name, "fixed": {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1},
        "ladder": {"128": {"points": [[8, 8.66]]},
                   "256": {"points": [[4, 8.0]]}}}))
    arm, why = rf.published_prediction(cfg, roof, block_n=64, group_m=1,
                                       control_block_m=256,
                                       published_dir=tmp_path)
    assert arm is None, "an A100 arm may not be scored against the H200 roof"
    assert "nvidia_h200" in why


def test_the_prediction_reads_the_card_off_the_roof_and_not_off_a_flag(rf, roof):
    """The condition is that ladders and denominator share a card, not that a
    flag says so."""
    assert rf.roof_card_slug(roof) == "nvidia_h200"


# --------------------------------------------------------------------------
# R3. One instrument, and clocks sampled under the load they describe.
# --------------------------------------------------------------------------

def test_the_mirrored_clock_constants_are_the_instruments_own(rf):
    """`moe.bench.timing` imports torch at module scope, so the two thresholds
    are mirrored rather than imported; this asserts them wherever it imports."""
    timing = pytest.importorskip("moe.bench.timing")
    assert rf.CLOCK_FLOOR_FRACTION == timing.LEVEL_FRACTION
    assert rf.THROTTLE_DRIFT_PCT == timing.DRIFT_FRACTION * 100.0


def test_the_stale_cap_sentence_is_gone_and_the_estimator_is_named(rf):
    """R3. "cap near 150 against a ridge near 163" came from alpha 0.85, which
    no ladder at this swizzle reads, through an identity no estimator here
    computes."""
    doc = " ".join(rf.__doc__.split())
    # The sentence survives ONLY as a quoted retraction, never as a claim: a
    # number deleted without a reason comes back, and this one is cited in
    # docs/FINDINGS.md.
    assert "used to read" in doc and "Both halves are retracted" in doc
    assert doc.count("near 150") == 1, "quoted once, asserted never"
    assert "(EXA)" in doc and "alpha_fitted" in doc
    # The G=1 alpha is a BRACKET across models (0.62 to 1.02), not the
    # mixtral-only 0.92-1.02 once quoted for every model; the old figure
    # survives only inside its own retraction sentence.
    assert "0.62 to 1.02" in doc, "the alphas the ladders actually read"
    assert "BRACKET" in doc
    assert doc.count("0.92-1.02") == 1 and "used to say" in doc
    assert doc.count("163.7") == 1, "the old ridge figure survives only in the retraction"


def test_the_script_no_longer_asks_the_sweep_for_the_retired_instrument(rf):
    """`time_call` raises `RetiredInstrument`; a caller that still required the
    symbol would abort a laptop dry run for a name it does not use."""
    src = (ROOT / "scripts" / "bm128_roofline.py").read_text()
    assert "SWEEP.time_call(" not in src, "the retired loop is still called"
    assert "time_kernel" in src
    # And it is not REQUIRED either: `_load_sweep` aborts at import on a missing
    # symbol, so a name this file no longer uses must not be in its list.
    src_needed = src[src.index("needed = ("):src.index("missing = [n for n")]
    assert "time_call" not in src_needed
    assert "reference_clock_mhz" in src_needed


def test_a_session_at_the_roofs_clock_passes_where_an_idle_sample_would_fail(
        rf, cfg, roof):
    """R3. The H200 idles at 1980 and runs a dense GEMM at 1515.

    Under load at 1515 this session is at the roof's operating point and PASSES.
    The same session sampled at the idle instant reads 1980, which is +31%
    against the roof and FAILS a 10% gate for no reason connected to the run.
    """
    assert roof.clock_mhz == 1515
    rows = [_timing(rf, rows=256, rep=r, load=1515.0) for r in range(1, 6)]
    points = rf.build_points(rows, cfg, 128, roof, sm_count=132, block_n=64,
                             clock_ref=1515)
    assert rf.gate_v3_clocks(rows, points, roof, 1515).passed is True
    idle = [_timing(rf, rows=256, rep=r, load=1980.0) for r in range(1, 6)]
    idle_points = rf.build_points(idle, cfg, 128, roof, sm_count=132,
                                  block_n=64, clock_ref=1980)
    assert rf.gate_v3_clocks(idle, idle_points, roof, 1980).passed is False


# --------------------------------------------------------------------------
# R4. The instrument's columns, the provenance block, the exit table and the
# one greppable line.
# --------------------------------------------------------------------------

KERNEL_TIMING_COLUMNS = ("instrument", "warmup_ms", "iters", "trials",
                         "sm_clock_load_mhz", "clock_level_ok",
                         "clock_drift_ok", "l2_flush")


def test_the_cells_csv_carries_every_kernel_timing_column(rf, tmp_path):
    path = tmp_path / "cells.csv"
    rf.append_timing(path, _timing(rf))
    header = path.read_text().splitlines()[0].split(",")
    for column in KERNEL_TIMING_COLUMNS:
        assert column in header, column


def test_the_cells_csv_carries_the_provenance_columns_when_given_one(rf,
                                                                     tmp_path):
    from moe.bench import provenance
    prov = provenance.provenance_block(repo_root=ROOT, instrument="test/fake")
    path = tmp_path / "cells.csv"
    rf.append_timing(path, _timing(rf), prov)
    header = path.read_text().splitlines()[0].split(",")
    assert "prov_git_sha" in header and "prov_instrument" in header


def test_a_clock_verdict_of_none_round_trips_as_none_and_not_as_false(rf,
                                                                      tmp_path):
    """A replayed row whose verdict was "not determined" must not become
    "determined, and fine"."""
    path = tmp_path / "cells.csv"
    rf.append_timing(path, _timing(rf, clocks=False))
    _, back = rf.read_timings(path)
    assert back[0].clock_level_ok is None and back[0].clock_drift_ok is None
    assert back[0].cold is False and back[0].throttled is False


def test_the_figure_csv_carries_the_instrument_on_every_row(rf, roof, tmp_path):
    rows = rf.figure_rows(_series(rf, roof), roof, "nvidia_h200")
    path = tmp_path / "figure.csv"
    rf.write_figure_csv(path, rows)
    header = path.read_text().splitlines()[0].split(",")
    for column in ("instrument", "warmup_ms", "trials", "l2_flush",
                   "clock_level_ok"):
        assert column in header, column


def test_every_gate_prints_exactly_one_result_line(rf, cfg, roof):
    """R4. The driver's summary keys on `RESULT: ` at column zero and on nothing
    else; a gate that printed two would be counted twice and one that printed
    none would vanish."""
    from moe.bench import exit_codes
    _, gates = rf.self_test(cfg, roof, 2, r_min=32, r_max=4096,
                            control_block_m=256, doublings=2)
    text = "\n".join(rf.render_gates(gates))
    parsed = exit_codes.parse_result_lines(text)
    assert len(parsed) == len(gates)
    assert [p.name for p in parsed] == [g.token for g in gates]
    assert [p.verdict for p in parsed] == [g.verdict for g in gates]


def test_the_exit_code_a_report_implies_is_the_one_it_returns(rf, cfg, roof):
    """`classify_text` closes the loop: what was printed and what was returned
    must be the same verdict."""
    from moe.bench import exit_codes
    _, gates = rf.self_test(cfg, roof, 2, r_min=32, r_max=4096,
                            control_block_m=256, doublings=2)
    text = "\n".join(rf.render_gates(gates))
    assert exit_codes.classify_text(text) == exit_codes.classify(gates)


def test_a_gate_that_did_not_pass_is_not_classified_as_done(rf, roof):
    """The FAIL branch of the table, planted."""
    from moe.bench import exit_codes
    assert exit_codes.classify([rf.gate_v0_roof(roof)]) == exit_codes.INVALID
    ok = rf.Gate(rf.CLAIM, "C1 roof", "", "", False, "")
    assert exit_codes.classify([ok]) == exit_codes.CLAIM_FAIL
    good = rf.Gate(rf.CLAIM, "C1 roof", "", "", True, "")
    assert exit_codes.classify([good]) == exit_codes.DONE


def test_a_broken_self_test_exits_invalid_rather_than_zero(rf, monkeypatch,
                                                           capsys):
    """R4. A self test that cannot fail is not a test.

    A wrong expectation is planted in the world table, and the run has to come
    back INVALID: its gates are all VALIDITY, so a failure means the instrument
    was not shown to discriminate and nothing it says may be quoted.
    """
    from moe.bench import exit_codes
    broken = tuple((name, alpha, overhead, share, moves, not expect, want)
                   for name, alpha, overhead, share, moves, expect, want
                   in rf.SELF_TEST_WORLDS)
    monkeypatch.setattr(rf, "SELF_TEST_WORLDS", broken)
    code = rf.main(["--self-test"])
    out = capsys.readouterr().out
    assert code == exit_codes.INVALID, out[-2000:]
    assert "FAIL" in out


def test_a_working_self_test_exits_done(rf, capsys):
    from moe.bench import exit_codes
    assert rf.main(["--self-test"]) == exit_codes.DONE
    assert "0 FAIL, 0 UNKNOWN" in capsys.readouterr().out


def test_an_unplanned_crash_exits_error_and_never_claim_fail(rf, monkeypatch,
                                                             capsys):
    """R4. The apparatus breaking must not be filed as one of the outcomes.

    An exception left to propagate exits the interpreter ONE, and ONE is
    CLAIM_FAIL, which `moe/bench/exit_codes.py` puts in FINISHED_CODES: the
    driver records it as a RESULT and never retries it. A torch OOM would then
    be published as "the claim did not hold". ERROR is outside FINISHED_CODES
    so that the two can be told apart, and the traceback is printed rather than
    swallowed because a bare code names nothing to fix.
    """
    from moe.bench import exit_codes

    def explode(argv=None):
        raise RuntimeError("the allocator gave up halfway through the ladder")

    monkeypatch.setattr(rf, "_main", explode)
    code = rf.main(["--dry-run"])
    err = capsys.readouterr().err
    assert code == exit_codes.ERROR
    assert code != exit_codes.CLAIM_FAIL
    assert code not in exit_codes.FINISHED_CODES, \
        "an apparatus failure must stay retryable"
    assert "the allocator gave up halfway" in err, "the traceback was swallowed"
    assert "RuntimeError" in err


def test_a_string_refusal_still_exits_refused_and_not_error(rf, monkeypatch,
                                                            capsys):
    """The other branch of the same handler. A `raise SystemExit("sentence")`
    is a PRECONDITION not met, which is free and distinct from a crash, so it
    must not be caught by the new `except Exception` and relabelled."""
    from moe.bench import exit_codes

    def refuse(argv=None):
        raise SystemExit("no calibration for the attached device")

    monkeypatch.setattr(rf, "_main", refuse)
    code = rf.main(["--dry-run"])
    err = capsys.readouterr().err
    assert code == exit_codes.REFUSED
    assert err.startswith("REFUSED: no calibration")
    assert "Traceback" not in err


def test_the_report_payload_takes_a_provenance_stamp_without_colliding(rf, cfg,
                                                                       roof):
    """R4. `stamp` raises rather than layering one block over another, so the
    payload must not already carry the five keys under different meanings."""
    from moe.bench import provenance
    timings = rf.planted_timings(cfg, roof, 2,
                                 {128: [256, 512, 1024, 2048, 4096],
                                  256: [256, 512, 1024, 2048, 4096]},
                                 alpha=1.0, overhead_ms=0.05, reps=2,
                                 noise=0.002, seed=0)
    _, gates, payload, _ = rf.analyse(
        timings, cfg, roof, control_block_m=256, b=2, sm_count=132, block_n=64,
        doublings=2, compiles={128: 1, 256: 1}, executed={128: 8, 256: 5},
        planned_multi_tile=5)
    prov = provenance.provenance_block(repo_root=ROOT,
                                       instrument=rf.instrument_name(),
                                       ridge=roof.ridge, ridge_source=roof.source)
    stamped = prov.stamp(payload)
    assert set(provenance.TOP_LEVEL_KEYS) <= set(stamped)
    assert stamped["provenance"]["instrument"] == rf.instrument_name()
    assert stamped["exit_code"] == 3, "a hypothesis roof fails V0"


def test_the_planted_cells_never_claim_the_real_instrument(rf, cfg, roof):
    """A self test that stamped TIMING_BASIS on its own fabrications would make
    a pod row and a generated one indistinguishable in a cells.csv."""
    rows = rf.planted_timings(cfg, roof, 2, {128: [256]}, alpha=1.0,
                              overhead_ms=0.0, reps=1, noise=0.0, seed=0)
    assert rows[0].instrument == rf.SWEEP.SYNTHETIC_INSTRUMENT
    assert "synthetic" in rows[0].instrument


def test_the_plan_prints_an_mde_from_a_stated_noise_assumption(rf, capsys):
    """R4 / B14. No arm in this study stated one, so every threshold was a prior
    with nothing behind it."""
    rf.main(["--dry-run"])
    out = capsys.readouterr().out
    assert "MDE" in out
    assert "noise assumption" in out
    assert "RESOLVABLE" in out
    assert f"{rf.DEFAULT_PLAN_NOISE_REL:.2%}" in out


def test_the_mde_says_not_resolvable_when_the_noise_swamps_the_gate(rf, capsys):
    """The FAIL branch. A line that only ever says RESOLVABLE says nothing."""
    rf.main(["--dry-run", "--plan-noise", "0.5", "--reps", "2"])
    out = capsys.readouterr().out
    assert "NOT RESOLVABLE" in out


def test_the_mde_refuses_outright_below_two_repeats(rf):
    lines = " ".join(rf.mde_lines(reps=1, noise_rel=0.014,
                                  noise_source="test"))
    assert "NOTHING here is resolvable" in lines


def test_the_mde_is_the_studys_own_two_sample_t_and_not_a_normal(rf):
    """Adopted from `scripts/replicate_noise_floor.py`, which owns it: at 4
    degrees of freedom a z is 35% too generous, and this line exists to say what
    the run cannot resolve."""
    power = rf._load_power()
    sd = 0.014 * 0.5
    expected = power.mde_two_sample(sd, 3, level=rf.MDE_LEVEL,
                                    power=rf.MDE_POWER)
    lines = " ".join(rf.mde_lines(reps=3, noise_rel=0.014, noise_source="test"))
    assert f"{expected:.4f}" in lines


def _plant_power(rf, monkeypatch, tmp_path, body: str):
    """A `scripts/replicate_noise_floor.py` that is broken in a stated way.

    `_load_power` loads that file BY PATH off `rf.ROOT`, so moving ROOT is how
    its two refusals are reached without touching the real file, which another
    slice owns. The module name is cleared either side because `_load_power`
    registers it with `setdefault` and a planted module left behind would be
    the next test's import.
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "replicate_noise_floor.py").write_text(body)
    monkeypatch.setattr(rf, "ROOT", tmp_path)
    monkeypatch.delitem(sys.modules, "replicate_noise_floor", raising=False)


def test_the_mde_refuses_when_the_file_that_owns_the_power_will_not_import(
        rf, monkeypatch, tmp_path):
    """The FAIL branch of the adoption. `_load_power` refuses rather than
    substituting a normal quantile, which at 4 degrees of freedom is 35% too
    generous and would advertise a resolution this run does not have. The
    refusal is what stops another slice's edit from silently downgrading it."""
    _plant_power(rf, monkeypatch, tmp_path,
                 "raise ValueError('half a rewrite')\n")
    with pytest.raises(SystemExit) as caught:
        rf._load_power()
    assert str(caught.value).startswith("REFUSED: ")
    assert "did not import" in str(caught.value)
    assert "ValueError: half a rewrite" in str(caught.value)
    assert "35% too generous" in str(caught.value)


def test_the_mde_refuses_when_the_power_symbols_have_been_renamed(
        rf, monkeypatch, tmp_path):
    """The second FAIL branch, and the one the handover leans on: the file
    imports cleanly but no longer exports what is called on it. Without this the
    rename would surface as an AttributeError inside a plan, not as a refusal
    naming the two symbols to re-point."""
    _plant_power(rf, monkeypatch, tmp_path,
                 "def mde_paired(sd, n):\n    return sd\n")
    with pytest.raises(SystemExit) as caught:
        rf._load_power()
    assert str(caught.value).startswith("REFUSED: ")
    assert "no longer exports" in str(caught.value)
    assert "mde_two_sample" in str(caught.value)
    assert "replicates_for" in str(caught.value)


def test_the_real_power_file_satisfies_both_of_those_refusals(rf):
    """The PASS branch, asserted against the file the study actually ships, so
    the two tests above cannot pass by planting worlds nothing resembles."""
    power = rf._load_power()
    assert callable(power.mde_two_sample) and callable(power.replicates_for)


def test_the_run_id_comes_from_the_shared_builder_and_needs_a_card(rf):
    """R4. `provenance.run_id` refuses an empty card; a private builder would
    have produced a directory name with a hole where the card goes."""
    from moe.bench import provenance
    with pytest.raises(provenance.NoCard):
        rf.default_run_id(_args(rf), "")
    assert rf.default_run_id(_args(rf), "NVIDIA H200").startswith("nvidia_h200-")


# --------------------------------------------------------------------------
# R5. THE CONTROL THAT DOES NOT EXIST.
#
# The session driver scheduled three arms around vLLM's shipped configuration
# for mixtral on the H200 -- BLOCK_SIZE_M=128 at BLOCK_SIZE_N=256, GROUP_SIZE_M
# 16 and 32 -- and all three refused, because the arm needs a positive control
# and no tile above 128 can be pinned at BLOCK_SIZE_N=256. The refusal was
# correct. What was wrong was the next sentence, which named --num-warps 16 as
# the remedy: 16 warps clear the PER-THREAD ceiling and the thread block still
# asks for the entire per-block register file.
#
# These tests pin the arithmetic that settles it, each candidate escape, and the
# mode that survives: `--control none`, which can refute the claim and can never
# confirm it.
# --------------------------------------------------------------------------

def test_the_accumulator_is_billed_per_block_and_not_only_per_thread(rf):
    """The limit the sweep's per-thread check stops seeing above 8 warps."""
    assert rf.accumulator_registers(256, 256) == rf.REGISTERS_PER_BLOCK
    assert rf.register_file_refusal(256, 256)
    # And the per-thread bill, at the warp count the retired hint recommended,
    # passes: that is why a second bill was needed and not a tighter one.
    assert 256 * 256 / (32 * 16) <= 255
    assert rf.register_file_refusal(128, 256) == "", \
        "the subject's own tile must still be runnable, or nothing is"


def test_more_warps_never_rescue_a_block_that_is_the_whole_register_file(rf):
    """Warps divide the accumulator across threads; they do not shrink it."""
    for warps in rf.CONTROL_WARP_COUNTS:
        per_thread = 256 * 256 / (32 * warps)
        assert per_thread * 32 * warps == rf.accumulator_registers(256, 256)
    assert rf.register_file_refusal(512, 256)
    assert rf.register_file_refusal(1024, 256)


def test_no_pin_at_all_yields_a_control_at_the_production_block_n(rf, cfg):
    """Every warp and stage count this file will try, on the real card."""
    fits, lines = rf.control_feasibility(
        cfg, block_n=256, block_k=64, dtype_bytes=2, capability=(9, 0),
        alpha=0.558, ridge=162.8)
    assert fits == []
    assert any("NO TILE ABOVE BLOCK_M=128" in line for line in lines)


def test_a_control_does_exist_one_block_n_lower(rf, cfg):
    """The FAIL branch above means nothing without this one.

    At BLOCK_SIZE_N=128 the 256x128 accumulator is half the file and 128
    registers per thread at 8 warps, and 4 stages of (256x64 + 64x128) is 192
    KiB against sm_90's 227. So the search finds a control, and the sentence
    "no control exists" is about the geometry and not about the search.
    """
    fits, _ = rf.control_feasibility(
        cfg, block_n=128, block_k=64, dtype_bytes=2, capability=(9, 0),
        alpha=0.558, ridge=162.8)
    assert (256, 8, 4) in fits


def test_the_search_names_nothing_a_fit_without_a_capability(rf, cfg):
    """Off a device the shared-memory limit is unknown, which is not "fits".

    The register half is decidable everywhere, so the production refusal is
    still reachable from a laptop with no flag; the stage half is not, and a
    triple returned on the register half alone would be the pod's rejection
    stated as advice.
    """
    fits, lines = rf.control_feasibility(
        cfg, block_n=128, block_k=64, dtype_bytes=2, capability=None,
        alpha=0.558, ridge=162.8)
    assert fits == []
    assert any("NO --capability GIVEN" in line for line in lines)


def test_vllm_ships_no_block_m_128_entry_at_block_n_128_on_the_h200():
    """The evidence behind alternative 3, read from the shipped files.

    "Run the subject at BLOCK_SIZE_N=128, where a control fits" is buildable and
    is a DIFFERENT question, and the reason is a fact about vLLM's tuned tables
    rather than an opinion: every BLOCK_SIZE_M=128 entry vLLM ships for the H200
    carries BLOCK_SIZE_N=256, on both models this study measures and on both
    dtypes. Checked here rather than asserted in prose, because the prose is
    what an operator acts on.
    """
    root = ROOT / "moe" / "bench" / "hardware" / "vllm_configs"
    files = sorted(root.glob("*device_name=NVIDIA_H200*.json"))
    assert files, "the shipped H200 tunings are not in the tree"
    seen = set()
    for path in files:
        for entry in json.loads(path.read_text()).values():
            seen.add((entry["BLOCK_SIZE_M"], entry["BLOCK_SIZE_N"]))
    assert (128, 256) in seen, "the production entry is not in these files"
    assert not [bn for bm, bn in seen if bm == 128 and bn != 256], \
        "vLLM ships a BLOCK_SIZE_M=128 entry at some other BLOCK_SIZE_N"


def test_a_control_at_its_own_block_n_has_exactly_the_subjects_cap():
    """Alternative 4, and the trap it is: the cap is symmetric in BM and BN.

    `moe.bench.ai_model.cap` is 2 / (b (alpha_b/BM + alpha_a/BN + 1/K)), so
    256x128 has the same ceiling as 128x256 to the last decimal. The scalar form
    the sweep quotes, 2 BM / (alpha b), has no BLOCK_SIZE_N in it and would have
    called that control twice the headroom.
    """
    from moe.bench import ai_model
    subject = ai_model.cap(14336, 4096, block_m=128, block_n=256,
                           alpha_b=1.0, alpha_a=1.0, b=2)
    transposed = ai_model.cap(14336, 4096, block_m=256, block_n=128,
                              alpha_b=1.0, alpha_a=1.0, b=2)
    assert transposed == pytest.approx(subject, rel=1e-12)
    bigger = ai_model.cap(14336, 4096, block_m=256, block_n=256,
                          alpha_b=1.0, alpha_a=1.0, b=2)
    assert bigger > subject, "a real control does have more headroom"


def _solo(rf, roof, cfg, alpha):
    """One uncontrolled run's gates, from the study's own planted model."""
    rows = rf.doubling_rows(cfg, 32, 4096, rf.SUBJECT_BLOCK_M)
    timings = rf.planted_timings(cfg, roof, 2, {rf.SUBJECT_BLOCK_M: rows},
                                 alpha=alpha, overhead_ms=0.05, reps=3,
                                 noise=0.002, seed=0)
    _, gates, payload, series = rf.analyse(
        timings, cfg, roof, control_block_m=None, b=2,
        sm_count=rf.SWEEP.DEFAULT_SM_COUNT,
        block_n=rf.SWEEP.FIXED["BLOCK_SIZE_N"], doublings=2,
        compiles={rf.SUBJECT_BLOCK_M: 1}, executed={rf.SUBJECT_BLOCK_M: len(rows)},
        planned_multi_tile=len([r for r in rows if r > rf.SUBJECT_BLOCK_M]))
    return gates, payload, series


def test_the_control_gates_are_absent_and_not_unknown(rf, cfg, roof):
    """A gate that examined nothing reporting no failure is the shape refused.

    V4 is a VALIDITY gate, so scoring it UNKNOWN would void C1 and C2 -- the two
    readings an uncontrolled run exists to produce -- and file the whole arm as
    an instrument failure. Omission is the honest state and it is what `verdict`
    reads to reach UNCONTROLLED.
    """
    gates, _, series = _solo(rf, roof, cfg, 1.0)
    names = {g.name.split()[0] for g in gates}
    assert {"V0", "V1", "V2", "V3", "C1", "C2"} <= names
    assert not ({"V4", "C3", "C4"} & names)
    assert len(series) == 1, "an uncontrolled run plots one curve"


def test_a_subject_below_the_roof_with_no_control_is_not_a_ceiling(rf, cfg,
                                                                   roof):
    """The verdict the production arm will reach, named as its own outcome."""
    gates, _, _ = _solo(rf, roof, cfg, 1.0)
    call, why = rf.verdict([g for g in gates if g.kind == rf.CLAIM])
    assert call == rf.UNCONTROLLED
    assert "CAN REFUTE THE HEADLINE AND CANNOT CONFIRM IT" in " ".join(why)


def test_an_uncontrolled_run_can_still_refute_the_claim(rf, cfg, roof):
    """The asymmetry, in its other direction, or the mode buys nothing.

    A refutation needs no positive control: the subject reaching the roof IS an
    instrument shown able to see a tile arrive. So C1 can fail with no control
    anywhere in the run, and that failure kills the study's central claim.
    """
    gates, _, _ = _solo(rf, roof, cfg, 0.10)
    call, _ = rf.verdict([g for g in gates if g.kind == rf.CLAIM])
    assert call == rf.NOT_BINDING


def test_the_residency_confound_is_replaced_where_there_was_no_control(rf, cfg,
                                                                      roof):
    """It describes how the subject and the control differ; there is no control.

    Printing it here would describe a comparison that did not happen. What
    replaces it is a larger caveat and not a smaller one.
    """
    gates, _, _ = _solo(rf, roof, cfg, 1.0)
    _, why = rf.verdict([g for g in gates if g.kind == rf.CLAIM])
    joined = " ".join(why)
    assert "NO CONTROL RAN IN THIS ARM" in joined
    assert "OCCUPANCY" not in joined
    # And a validity-only refusal still carries the residency paragraph: that
    # arm HAD a control and did not get far enough to compare it.
    _, validity_why = rf.verdict([rf.gate_v0_roof(roof)])
    assert "OCCUPANCY" in " ".join(validity_why)


def test_both_directions_of_the_uncontrolled_mode_are_planted(rf, capsys):
    """A mode whose two directions are not both planted is one nobody has run."""
    code = rf.main(["--self-test"])
    out = capsys.readouterr().out
    assert code == 0
    assert "RESULT: VALIDITY S_uncontrolled_capped PASS" in out
    assert "RESULT: VALIDITY S_uncontrolled_uncapped PASS" in out
    assert "UNCONTROLLED" in out


def test_the_self_test_still_plants_the_controlled_worlds_without_a_control(rf):
    """`--control none` asks one ARM to run without a control; it does not ask
    the self test to stop checking whether the controlled gates discriminate."""
    _, gates = rf.self_test(MODEL_CONFIGS["mixtral-8x7b"],
                            rf._hypothesis_roof("test"), 2, r_min=32,
                            r_max=4096, control_block_m=None, doublings=2)
    assert all(g.passed for g in gates)
    assert any(g.name == "S discrimination" for g in gates)


def test_an_uncontrolled_arm_cannot_resume_into_a_controlled_ones_directory(rf):
    """The absence has a spelling, because a knob missing from a run id is a
    knob two runs can silently share a directory across."""
    solo = rf.default_run_id(_args(rf, control=None), "NVIDIA H200")
    paired = rf.default_run_id(_args(rf, control=256), "NVIDIA H200")
    assert "cnone" in solo
    assert solo != paired


def test_spelling_the_control_did_not_move_the_controlled_arms_directory(rf):
    """R6. The digest moved and nothing about the arm did.

    `moe.bench.provenance._canonical` keeps an int an int and a string a string,
    so `c=256` and `c="256"` hash differently while rendering the same visible
    prefix. `default_run_id` was passing `control_key()` -- a spelling built for
    the CLI echo -- so the session's roofline-n64-g1 arm silently changed
    directory from ...-965a1e55 to ...-9d41e41f and could no longer resume
    anything a previous run of that arm had written. The visible half was
    byte-identical, which is the collision shape `provenance.py` exists to
    prevent, and the third instance of a family this repo has now fixed twice.

    BOTH HALVES ARE PINNED. The hazard, so a reader can see it is real, and the
    id itself, so a future edit cannot move it again in silence. The uncontrolled
    id is pinned too: the absence has to keep its spelling.
    """
    from moe.bench import provenance as PV

    assert PV.run_id(card="nocard", c=256, x=3) \
        != PV.run_id(card="nocard", c="256", x=3), \
        "provenance no longer distinguishes an int from its spelling"
    assert rf.default_run_id(_args(rf, control=256), rf.UNKNOWN_CARD_SLUG) == (
        "nocard-b400.0-c256-dbf16-e0-ftrue-g1-hi4096-k64-lo32-mmixtral_8x7b"
        "-n64-s4-t3-u300.0-w8-x3-965a1e55")
    assert rf.default_run_id(_args(rf, control=None), rf.UNKNOWN_CARD_SLUG) == (
        "nocard-b400.0-cnone-dbf16-e0-ftrue-g1-hi4096-k64-lo32-mmixtral_8x7b"
        "-n64-s4-t3-u300.0-w8-x3-f0bc11f5")


# --------------------------------------------------------------------------
# R6. The uncontrolled verdict, on a line the session driver can read.
# --------------------------------------------------------------------------

def test_the_uncontrolled_verdict_reaches_a_scored_result_line(rf, cfg, roof):
    """The blocking defect: UNCONTROLLED was in the prose and nowhere else.

    `scripts/h200_gaps_session.sh` summarises an arm by grepping `^RESULT: ` and
    nothing else -- it was rewritten that way because the old summary grepped
    prose. Rendered, the uncontrolled mode's own capped world produced six
    RESULT lines, every one PASS, and exit 0 DONE; the word UNCONTROLLED
    appeared only under `## Verdict`, which nothing machine-readable emits. A
    BINDING run and an UNCONTROLLED run differed in the summary by two MISSING
    lines, and an absence is the same shape as a check that examined nothing
    reporting zero failures.

    ON AN ATTACHED ROOF, because that is the only run whose exit code is about
    its claim gates: the hypothesis roof fails V0 and lands INVALID, which is
    what stops a laptop report being quotable and is asserted elsewhere.
    """
    from dataclasses import replace

    from moe.bench import exit_codes

    gates, payload, _ = _solo(rf, replace(roof, attached=True), cfg, 1.0)
    assert payload["verdict"] == rf.UNCONTROLLED
    lines = [g.result_line() for g in gates]
    carrying = [line for line in lines if "UNCONTROLLED" in line]
    assert len(carrying) == 1, lines
    assert carrying[0].startswith("RESULT: CLAIM CU_UNCONTROLLED_attribution ")
    assert payload["exit_code"] == exit_codes.CLAIM_FAIL
    assert exit_codes.ledger_state(payload["exit_code"]) == "CLAIM_FAIL"
    # And the code the log IMPLIES agrees with the code the payload carries,
    # which is the loop `exit_codes.classify_text` exists to close.
    assert exit_codes.classify_text("\n".join(lines)) == payload["exit_code"]


def test_the_uncontrolled_gate_can_never_pass_in_either_direction(rf, cfg, roof):
    """It is the mirror of a rubber stamp, and only the mirror is safe.

    A gate that cannot FAIL launders an unexamined claim into an exit code of 0.
    This one cannot PASS, so it can never turn CLAIM_FAIL into DONE. Its two
    reachable verdicts are both readings of the world: FAIL when the subject
    REACHED the roof, because a refutation needs no positive control, and
    UNKNOWN when it stopped below it, because nothing ran that could attribute
    the shortfall. Both are planted.
    """
    from moe.bench import exit_codes

    capped, _, _ = _solo(rf, roof, cfg, 1.0)
    uncapped, _, _ = _solo(rf, roof, cfg, 0.10)
    verdicts = {}
    for label, gates in (("capped", capped), ("uncapped", uncapped)):
        gate = [g for g in gates if g.name == rf.UNCONTROLLED_GATE]
        assert len(gate) == 1, label
        verdicts[label] = gate[0].verdict
        assert gate[0].verdict != exit_codes.PASS
    assert verdicts == {"capped": exit_codes.UNKNOWN, "uncapped": exit_codes.FAIL}


def test_a_controlled_arm_scores_no_uncontrolled_gate(rf, cfg, roof):
    """C3 and C4 do that scoring where they exist, and two gates saying the same
    thing is how a summary comes to read one of them."""
    timings, grid = _planted_pair(rf, cfg, roof)
    _, gates, payload, _ = rf.analyse(
        timings, cfg, roof, control_block_m=256, b=2,
        sm_count=rf.SWEEP.DEFAULT_SM_COUNT,
        block_n=rf.SWEEP.FIXED["BLOCK_SIZE_N"], doublings=2,
        compiles={128: 1, 256: 1},
        executed={128: len(grid[128]), 256: len(grid[256])},
        planned_multi_tile=len([r for r in grid[128] if r > 128]))
    names = {g.name for g in gates}
    assert rf.UNCONTROLLED_GATE not in names
    assert "C3 tile attribution" in names and "C4 ceiling located" in names
    assert not [line for line in (g.result_line() for g in gates)
                if "UNCONTROLLED" in line]
    assert "OCCUPANCY" in " ".join(payload["verdict_why"]), \
        "a controlled arm keeps the residency paragraph"


def _planted_pair(rf, cfg, roof):
    """One controlled run's timings, both tiles, from the study's own model."""
    rows = rf.doubling_rows(cfg, 32, 4096, rf.SUBJECT_BLOCK_M)
    grid = {rf.SUBJECT_BLOCK_M: rows, 256: [r for r in rows if r % 256 == 0]}
    return rf.planted_timings(cfg, roof, 2, grid, alpha=1.0, overhead_ms=0.05,
                              reps=3, noise=0.002, seed=0), grid


def test_control_takes_a_block_m_or_the_word_none_and_nothing_else(rf):
    """REFUSE rather than default: dropping the control drops three gates."""
    assert rf.parse_control("none") is None
    assert rf.parse_control("NONE") is None
    assert rf.parse_control("256") == 256
    with pytest.raises(Exception) as caught:
        rf.parse_control("maybe")
    assert "'none'" in str(caught.value)
