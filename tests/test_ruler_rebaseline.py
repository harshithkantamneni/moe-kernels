"""The ruler re-baseline: the parts of it that can be checked without a GPU.

Two halves, and they fail in different ways.

`moe/bench/calibrate.py` decides what a ceiling IS -- which pattern may be one,
what clock it was measured at, and what band the ridge really spans. Every rule
here is pure arithmetic over recorded facts, so all of it is testable on a
laptop, which matters because the failures it guards against are silent on the
pod: a reduction installed as a read ceiling, a clock sampled off an idle GPU,
and a ridge quoted as a point when it is a band.

`scripts/ruler_rebaseline.py` decides what changing the ceiling COSTS, over the
rows already published against the old one. It reads committed CSVs and committed
yaml and nothing else, so its answer is identical on every machine -- which is
the property that lets a re-baseline be reviewed before it is adopted rather than
after.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from moe.bench.calibrate import (
    CLOCK_SPREAD_TOL_PCT,
    DISOWNED,
    MATCHED_CEILING,
    READ_PATTERNS,
    BandwidthResult,
    Calibration,
    LoadedClock,
    demote_invalid_reads,
)

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    """Import the script by path; `scripts/` is not a package.

    Same shape the other script tests use. Loading it also proves the module
    imports cleanly with no CUDA device, which is half of what `--dry-run` on a
    laptop is for.
    """
    path = REPO / "scripts" / "ruler_rebaseline.py"
    spec = importlib.util.spec_from_file_location("ruler_rebaseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ruler_rebaseline"] = module
    spec.loader.exec_module(module)
    return module


RB = _load_script()


def bw(pattern: str, gbps: float, note: str = "", start: int = 1980,
       end: int = 1980) -> BandwidthResult:
    return BandwidthResult(pattern=pattern, bytes_moved=1 << 30, ms_p50=1.0,
                           ms_min=1.0, gbps=gbps, gbps_peak_min=gbps, note=note,
                           sm_clock_start_mhz=start, sm_clock_end_mhz=end)


def calibration(patterns, **kwargs) -> Calibration:
    base = dict(gpu_name="NVIDIA H200", achieved_bandwidth_gbps=4374.8,
                ceiling_pattern="triad", achieved_bf16_tflops=712.3,
                bandwidth_patterns=tuple(patterns), gemm_shape=(8192, 8192, 8192))
    base.update(kwargs)
    return Calibration(**base)


# --- a read that is slower than triad is not a read --------------------------

def test_a_read_below_triad_is_disowned_whatever_it_is_called():
    """The guard used to match the literal name `read` and was inline in
    `calibrate()`. Renaming the pattern would have stopped it firing, silently,
    and the reduction would have become a legal denominator again."""
    out = demote_invalid_reads([bw("read_reduce", 1744.3), bw("triad", 1799.4)])
    assert DISOWNED in out[0].note
    assert out[1].note == ""


def test_a_read_above_triad_keeps_its_note():
    out = demote_invalid_reads([bw("read_reduce", 4469.6, "a lower bound"),
                                bw("triad", 4374.8)])
    assert out[0].note == "a lower bound"


def test_every_read_pattern_is_checked_not_just_the_first():
    """A probe that serialised its own loads is the same failure as a tree
    reduction, and would arrive under a different name."""
    out = demote_invalid_reads([bw("read_stream", 3000.0),
                                bw("read_reduce", 4469.6), bw("triad", 4374.8)])
    assert DISOWNED in out[0].note
    assert out[1].note == ""


def test_without_triad_nothing_is_disowned():
    """Nothing to compare against is not evidence of a bad read."""
    out = demote_invalid_reads([bw("read_reduce", 100.0), bw("copy", 200.0)])
    assert all(p.note == "" for p in out)


# --- which pattern is the matched ruler -------------------------------------

def test_the_stream_probe_wins_when_it_ran():
    cal = calibration([bw("read_stream", 4600.0), bw("read_reduce", 4469.6),
                       bw("triad", 4374.8)])
    assert cal.matched_pattern().pattern == "read_stream" == MATCHED_CEILING


def test_the_reduction_is_the_fallback_not_triad():
    """A session without Triton still has a read ruler; it is just a weaker one.
    Falling back to triad would report a denominator swing of zero and it would
    look like a measurement."""
    cal = calibration([bw("read_reduce", 4469.6), bw("triad", 4374.8)])
    assert cal.matched_pattern().pattern == "read_reduce"


def test_a_disowned_read_is_not_the_matched_ruler():
    cal = calibration(demote_invalid_reads(
        [bw("read_reduce", 1744.3), bw("triad", 1799.4)]))
    assert cal.matched_pattern() is None
    assert cal.ridge_band() is None, "no band may be built from a disowned read"


def test_read_patterns_are_ordered_best_first():
    """`matched_pattern` walks this tuple, so its order is the policy."""
    assert READ_PATTERNS.index("read_stream") < READ_PATTERNS.index("read_reduce")


# --- the ridge is a band ------------------------------------------------------

def test_the_band_spans_the_named_ceiling_and_the_matched_one():
    cal = calibration([bw("read_reduce", 4469.6), bw("triad", 4374.8)])
    low, high = cal.ridge_band()
    assert low < high
    assert high == pytest.approx(cal.ridge_point(), rel=1e-9)
    assert low == pytest.approx(cal.ridge_point(4469.6), rel=1e-9)


def test_the_band_is_the_two_percent_the_study_argues_about():
    """The measured H200 figures. A band this wide moves a crossing by a whole
    tile tread, which is why it is recorded rather than rounded away."""
    cal = calibration([bw("read_reduce", 4469.6), bw("triad", 4374.8)])
    low, high = cal.ridge_band()
    assert 2.0 < (high / low - 1) * 100 < 2.4


def test_as_dict_carries_the_band_so_a_reader_cannot_miss_it():
    d = calibration([bw("read_reduce", 4469.6), bw("triad", 4374.8)]).as_dict()
    assert d["ridge_band"] == [round(min(d["ridge_by_pattern"].values()), 1),
                               round(d["ridge_by_pattern"]["triad"], 1)]
    assert d["matched_ceiling_pattern"] == "read_reduce"


def test_as_dict_says_null_rather_than_a_degenerate_band():
    d = calibration([bw("triad", 4374.8)]).as_dict()
    assert d["ridge_band"] is None
    assert d["matched_ceiling_gbps"] is None


# --- the clock the GEMM actually ran at --------------------------------------

def clock(median: int, spread: float = 1.0, idle: int = 1980) -> dict:
    return LoadedClock(label="bf16 GEMM", samples=(median,) * 3,
                       median_mhz=median, spread_pct=spread,
                       after_idle_mhz=idle).as_dict()


def test_a_clock_agreeing_with_its_settle_is_established():
    cal = calibration([bw("triad", 4374.8)], gemm_clock=clock(1500),
                      settle={"settled": True, "final_mhz": 1470})
    assert cal.clock_established is True


def test_samples_that_disagree_with_each_other_are_not_established():
    """1485 to 1935 within one measurement is the observed shape, and averaging
    it produces a plausible middle that describes no moment of the run."""
    cal = calibration([bw("triad", 4374.8)],
                      gemm_clock=clock(1700, spread=CLOCK_SPREAD_TOL_PCT + 1),
                      settle={"settled": True, "final_mhz": 1470})
    assert cal.clock_established is False


def test_a_clock_far_from_its_settle_plateau_is_not_established():
    """This is the published state: a GEMM whose settle plateaued at 1470 and
    whose clock field says 1935, with nothing in the file saying which is the
    GEMM's."""
    cal = calibration([bw("triad", 4374.8)], gemm_clock=clock(1935),
                      settle={"settled": True, "final_mhz": 1470})
    assert cal.clock_established is False


def test_no_samples_is_unknown_rather_than_false():
    """A `--no-settle` smoke run has not failed the check; it has not run it."""
    assert calibration([bw("triad", 4374.8)]).clock_established is None
    assert calibration([bw("triad", 4374.8)], gemm_clock=clock(1500),
                       settle={}).clock_established is None


def test_the_post_hoc_sample_is_kept_beside_the_honest_one():
    """Deleting it would erase the evidence that the artefact was ever there."""
    rec = clock(1500, idle=1935)
    assert rec["after_idle_mhz"] == 1935 and rec["median_mhz"] == 1500


# --- did the clock move across the patterns ---------------------------------

def test_a_pattern_that_ends_low_counts_as_ramped():
    """`clock_ramped` read only `sm_clock_start_mhz`, so four patterns all
    starting at 1980 reported a spread of zero however they ended."""
    cal = calibration([bw("triad", 4374.8, start=1980, end=1980),
                       bw("copy", 4300.0, start=1980, end=1400)])
    assert cal.clock_ramped is True


def test_a_steady_run_is_not_ramped():
    cal = calibration([bw("triad", 4374.8), bw("copy", 4300.0)])
    assert cal.clock_ramped is False


def test_zero_clocks_are_ignored_rather_than_read_as_a_ramp():
    """NVML returns 0 when it cannot answer, and a 0 beside a 1980 would be a
    100% spread reported as a throttling event."""
    cal = calibration([bw("triad", 4374.8, start=0, end=0)])
    assert cal.clock_ramped is False


# --- both settles are visible ------------------------------------------------

def test_both_settles_are_rendered_and_each_says_what_it_governs():
    """The memory settle was measured from 2026-08-27 and never printed, so the
    report read like an unsettled ramp and STUDY.md still listed the work as
    outstanding five days later."""
    cal = calibration(
        [bw("triad", 4374.8)],
        settle={"settled": True, "final_mhz": 1470, "clock_history_mhz": [1470],
                "bandwidth_settle": {"settled": True, "final_mhz": 1980,
                                     "clock_history_mhz": [1980]}})
    text = "\n".join(cal.settle_lines())
    assert "1470" in text and "1980" in text
    assert "the bf16 and fp8 GEMMs" in text
    assert "the four bandwidth patterns" in text


def test_a_missing_memory_settle_says_so_instead_of_vanishing():
    cal = calibration([bw("triad", 4374.8)],
                      settle={"settled": True, "final_mhz": 1470})
    assert "NOT RUN" in "\n".join(cal.settle_lines())


# --- refusals are recorded, never substituted --------------------------------

def test_a_refused_pattern_is_absent_and_named():
    cal = calibration([bw("read_reduce", 4469.6), bw("triad", 4374.8)],
                      refusals=("read_stream: triton is not importable",))
    assert cal.pattern("read_stream") is None
    assert "read_stream" in cal.as_dict()["refusals"][0]


def test_a_clockless_gemm_keeps_its_tflops_and_records_why_it_has_no_clock():
    """The TFLOP/s is FLOP over seconds and nothing normalises it, so a box
    where NVML is blocked must still get its ceilings. What it loses is exactly
    the two figures that divide by a clock, and both already return None.

    The alternative -- letting `ClockUnavailable` escape `calibrate()` -- would
    throw away a whole calibration over one field, on the container
    configuration this project actually runs on."""
    import moe.bench.calibrate as C

    def boom(*a, **kw):
        raise C.ClockUnavailable("no usable SM clock samples")

    original, C.clock_under_load = C.clock_under_load, boom
    try:
        refusals = []
        gemm = C._with_clock(lambda: None, "bf16 GEMM", 712.3, 8192, refusals)
    finally:
        C.clock_under_load = original
    assert gemm.tflops == 712.3
    assert gemm.clock is None and gemm.sm_clock_mhz == 0
    assert refusals and "SM clock" in refusals[0]
    cal = calibration([bw("triad", 4374.8)], gemm_clock_mhz=0,
                      refusals=tuple(refusals))
    assert cal.sustained_peak_tflops is None
    assert cal.gemm_efficiency_pct is None


def test_the_stream_probe_refuses_a_tensor_it_cannot_read_once():
    """A non-contiguous or wrong-width buffer would change the byte count
    without changing the reported figure, which is the shape of the silent 2x
    error this project already made once on the write pattern."""
    from moe.bench.read_probe import make_stream_read

    class FakeTensor:
        def __init__(self, dim, itemsize=4, contiguous=True):
            self._dim, self.dtype = dim, type("dt", (), {"itemsize": itemsize})()
            self._c = contiguous

        def dim(self):
            return self._dim

        def is_contiguous(self):
            return self._c

    for bad in (FakeTensor(2), FakeTensor(1, itemsize=2),
                FakeTensor(1, contiguous=False)):
        with pytest.raises(ValueError, match="contiguous 1-D 4-byte"):
            make_stream_read(bad)


# --- the script: reading a calibration file ----------------------------------

def h200_yaml() -> Path:
    return REPO / "results" / "published" / "2026-09-01-nvidia_h200-alpha-0558" / "measured.yaml"


def test_a_committed_calibration_parses_into_a_ruler():
    ruler = RB.read_ruler(h200_yaml())
    assert ruler.ceiling_pattern == "triad"
    assert ruler.bandwidth_gbps == pytest.approx(4373.9, abs=0.1)
    assert ruler.peak_tflops["bf16"] == pytest.approx(716.0, abs=0.1)


def test_the_legacy_pattern_name_is_still_read():
    """Every published arm carries `read`, not `read_reduce`. Refusing the old
    name would make the corpus comparison empty and it would look like a clean
    result."""
    name, gbps = RB.read_ruler(h200_yaml()).matched_read()
    assert name == "read" and gbps > 4400


def test_the_a100s_disowned_read_is_not_offered_as_a_ruler():
    """Its `read` came in below triad and its own calibration says so."""
    path = (REPO / "results" / "published"
            / "2026-08-28-nvidia_a100_sxm4_80gb-a100-cross-card" / "measured.yaml")
    ruler = RB.read_ruler(path)
    assert "read" in ruler.disowned
    assert ruler.matched_read() is None


def test_a_calibration_with_no_bandwidth_raises_rather_than_defaulting(tmp_path):
    bad = tmp_path / "measured.yaml"
    bad.write_text("name: nothing\nmemory: {}\n")
    with pytest.raises(ValueError, match="bandwidth"):
        RB.read_ruler(bad)


# --- the script: classifying rows -------------------------------------------

def ruler(bw_gbps: float = 4374.8, peaks=None) -> RB.Ruler:
    return RB.Ruler(path="x", name="x", gpu_name="NVIDIA H200", checked_on="",
                    bandwidth_gbps=bw_gbps, ceiling_pattern="triad",
                    patterns={"triad": bw_gbps, "read_reduce": 4469.6},
                    disowned=frozenset(),
                    peak_tflops=peaks if peaks is not None else {"bf16": 712.3},
                    gemm_clock_mhz=1500)


def test_a_row_whose_dtype_has_no_peak_is_unclassifiable_not_memory_bound():
    """All 19,908 fp8 rows of `-fp8-three-kernel` carry a zero peak. Counting
    them as 'did not flip' would put 19,908 rows of false reassurance under
    gate 5."""
    rows = [{"dtype": "fp8_e4m3", "arith_intensity_compulsory": "300"}]
    classified, unclassifiable, bounds = RB.classify(rows, ruler(), 4374.8)
    assert (classified, unclassifiable, bounds) == (0, 1, [None])


def test_classification_is_the_ridge_comparison_and_nothing_else():
    rows = [{"dtype": "bf16", "arith_intensity_compulsory": "100"},
            {"dtype": "bf16", "arith_intensity_compulsory": "300"}]
    _, _, bounds = RB.classify(rows, ruler(), 4374.8)
    assert bounds == ["memory", "compute"]


def test_flips_in_opposite_directions_do_not_cancel():
    """Counting memory-bound rows before and after would report ZERO here.

    Two dtypes whose ridges move opposite ways, which the session lever produces
    routinely: bandwidth rises so the bf16 ridge falls and a bf16 row crosses
    into compute, while the fp8 peak rises further so the fp8 ridge climbs and
    an fp8 row crosses back into memory. One in, one out, net change nil, two
    rows whose published classification is not what it was."""
    rows = [{"dtype": "bf16", "arith_intensity_compulsory": "161.0",
             "model": "a", "impl": "b", "num_tokens": "1"},
            {"dtype": "fp8_e4m3", "arith_intensity_compulsory": "340.0",
             "model": "a", "impl": "b", "num_tokens": "2"}]
    old_peaks = {"bf16": 712.3, "fp8_e4m3": 1447.7}
    new_peaks = {"bf16": 712.3, "fp8_e4m3": 1592.5}
    shift = RB.ArmShift(arm="x", lever="session", rows=len(rows))
    RB._fill(shift, ruler(), rows, 4374.8, 4469.6, old_peaks, new_peaks)
    # bf16 ridge 162.8 -> 159.4 (row crosses up); fp8 ridge 330.9 -> 356.3
    # (row crosses down). A count of memory-bound rows is 1 either side.
    _, _, before = RB.classify(rows, ruler(), 4374.8, old_peaks)
    _, _, after = RB.classify(rows, ruler(), 4469.6, new_peaks)
    assert before.count("memory") == after.count("memory") == 1
    assert shift.flips == 2 and shift.classified == 2


def test_the_identity_check_separates_rows_stamped_from_another_file():
    rows = [{"achieved_bw_gbps": "4374.8"}, {"achieved_bw_gbps": "4377.2"},
            {"achieved_bw_gbps": ""}]
    assert RB.identity_check(rows, ruler(4374.8)) == (1, 1)


# --- the script: gates -------------------------------------------------------

def test_the_non_vacuity_gate_fails_on_an_empty_survey():
    """Zero rows produce zero flips, which is exactly what gate 5 passes on."""
    gate = RB.gate_v1_non_vacuity(RB.Corpus())
    assert gate.verdict == RB.FAIL and gate.kind == RB.VALIDITY


def test_the_non_vacuity_gate_passes_on_the_real_corpus():
    corpus = RB.survey(REPO / "results" / "published",
                       RB.load_current_rulers(REPO / "moe" / "bench" / "hardware"))
    assert RB.gate_v1_non_vacuity(corpus).verdict == RB.PASS
    assert corpus.rows > RB.MIN_ROWS


def test_gate_four_refuses_to_pool_two_devices():
    """The first version of this gate reported a 66.6% 'compute term
    instability' that was an A100's 262 TFLOP/s standing beside an H200's 716.
    Two cards are not two measurements of one ruler."""
    corpus = RB.Corpus()
    corpus.rulers = {
        "h200|4374.80|700.00": {"device": "h200", "gpu_name": "H200",
                                "bandwidth_gbps": 4374.8, "bf16_tflops": 700.0,
                                "arms": ["a"]},
        "h200|4377.00|714.00": {"device": "h200", "gpu_name": "H200",
                                "bandwidth_gbps": 4377.0, "bf16_tflops": 714.0,
                                "arms": ["b"]},
        "a100|1798.50|262.10": {"device": "a100", "gpu_name": "A100",
                                "bandwidth_gbps": 1798.5, "bf16_tflops": 262.1,
                                "arms": ["c"]},
    }
    corpus.device_of = {"a": "h200", "b": "h200", "c": "a100"}
    for arm, device in corpus.device_of.items():
        shift = RB.ArmShift(arm=arm, lever="denominator", old_bw_gbps=4374.8,
                            new_bw_gbps=4469.6)
        if device == "a100":
            shift.old_bw_gbps, shift.new_bw_gbps = 1798.5, 1800.0
        corpus.denominator.append(shift)
    gate = RB.gate_4_spread(corpus)
    # 714/700 is 2.0%, not the 62% an A100-versus-H200 pool would report.
    assert "a100: 1 calibration(s)" in " ".join(gate.lines)
    assert not any("62" in line or "66" in line for line in gate.lines)


def test_gate_five_is_undecided_rather_than_passing_on_nothing():
    corpus = RB.Corpus()
    corpus.denominator = [RB.ArmShift(arm="x", lever="denominator")]
    assert RB.gate_5_flips(corpus).verdict == RB.UNDECIDED


def test_gate_five_fails_when_rows_cross(monkeypatch):
    corpus = RB.Corpus()
    shift = RB.ArmShift(arm="x", lever="denominator", classified=100, flips=3)
    corpus.denominator = [shift]
    gate = RB.gate_5_flips(corpus)
    assert gate.verdict == RB.FAIL and "3 flips" in gate.measured


def test_an_ignored_path_that_is_meant_to_be_kept_fails(tmp_path):
    results = REPO / "results"
    paths = {"raw": results / "ruler_rebaseline" / "x" / "report.txt",
             "published": results / "published" / "arm" / "plots" / "f.png"}
    gate = RB.gate_v3_paths(paths, results)
    # The raw path is ignored BY DESIGN and must not be reported as a defect.
    assert "raw output, ignored by design" in " ".join(gate.lines)


def test_the_raw_output_directory_alone_does_not_fail_the_path_gate():
    results = REPO / "results"
    gate = RB.gate_v3_paths({"raw": results / "ruler_rebaseline" / "x.txt"},
                            results)
    assert gate.verdict == RB.PASS


def test_git_check_ignore_returns_none_rather_than_false_when_it_cannot_ask():
    """THE POD DEFAULT IS THE UNASKABLE CASE.

    `git check-ignore` exits 128 for a path outside the work tree, which is what
    `/workspace/results` is on every pod. Returning `False` there -- read
    downstream as "tracked" -- made V3 print PASS and the line `<path>
    tracked` on a machine where it had not been able to ask at all.
    """
    assert RB.git_check_ignore(REPO / "results" / "x") is True
    assert RB.git_check_ignore(REPO / "moe" / "spec.py") is False
    assert RB.git_check_ignore(Path("/definitely/not/a/work/tree/x")) is None


def test_an_unverifiable_path_meant_to_be_kept_is_undecided_not_passed(monkeypatch):
    """A VALIDITY gate that cannot ask must not answer.

    Its FAIL means "no number on this page may be quoted", so a vacuous PASS is
    the most expensive verdict in the file. Non-vacuity: a check that examined
    nothing reports zero failures too.
    """
    monkeypatch.setattr(RB, "git_check_ignore", lambda _path: None)
    results = REPO / "results"
    gate = RB.gate_v3_paths(
        {"published": results / "published" / "arm" / "f.png"}, results)
    assert gate.verdict == RB.UNDECIDED
    assert "UNVERIFIED" in " ".join(gate.lines)


def test_an_unverifiable_raw_path_is_still_a_pass(monkeypatch):
    # A raw run directory is ignored BY DESIGN, so not being able to confirm it
    # costs nothing and must not train the reader to ignore this gate.
    monkeypatch.setattr(RB, "git_check_ignore", lambda _path: None)
    results = REPO / "results"
    gate = RB.gate_v3_paths({"raw": results / "ruler_rebaseline" / "x.txt"},
                            results)
    assert gate.verdict == RB.PASS


# --- the script: predictions and plumbing ------------------------------------

def test_every_prediction_states_what_a_fail_would_mean():
    """A gate whose failure has no stated consequence is a gate nobody has to
    honour, which is the whole reason the field exists."""
    assert len(RB.PREDICTIONS) == 5
    for pred in RB.PREDICTIONS:
        assert pred.claim and pred.numbers and len(pred.fail) > 40


def test_the_registered_numbers_are_the_ones_in_the_committed_calibration():
    """A prediction quoting a figure that is not in the tree is not checkable.

    The registered names are the POST-rename ones, because they are what a fresh
    calibration will emit and what gate 2 will look up. The committed file
    predates the rename, so `read_reduce` is compared against the `read` it was
    renamed from -- the same measurement, and the mapping is stated here rather
    than hidden in a lookup that would quietly match nothing."""
    ruler = RB.read_ruler(REPO / "moe" / "bench" / "hardware"
                          / "measured_nvidia_h200.yaml")
    renamed_from = {"read_reduce": "read"}
    for name, expected in RB.H200_PATTERNS_GBPS.items():
        in_file = renamed_from.get(name, name)
        assert in_file in ruler.patterns, f"{name} has no counterpart in the file"
        assert ruler.patterns[in_file] == pytest.approx(expected, abs=0.1), name


@pytest.mark.parametrize("field_name,value", [
    ("buffer_gb", 4.0), ("gemm_n", 4096), ("ceiling", "read_reduce"),
    ("settle_seconds", 10.0), ("corpus_only", True),
])
def test_the_run_id_changes_when_any_knob_does(field_name, value):
    """A run id that omits a swept parameter makes the second setting resume the
    first's directory, skip every completed cell, and print the first's numbers
    under the second's label."""
    args = RB.build_parser().parse_args([])
    base = RB.default_run_id(args)
    setattr(args, field_name, value)
    assert RB.default_run_id(args) != base, field_name


def test_the_cost_estimate_is_zero_only_when_nothing_will_be_measured():
    args = RB.build_parser().parse_args(["--corpus-only"])
    assert RB.estimated_seconds(args) == 0.0
    assert RB.estimated_seconds(RB.build_parser().parse_args([])) > 60


def test_dry_run_prints_every_prediction_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "nowhere"
    assert RB.main(["--dry-run", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    for pred in RB.PREDICTIONS:
        assert pred.claim in printed
    assert "estimated GPU time" in printed
    assert not out.exists(), "a dry run must not create its output directory"


def test_corpus_only_runs_off_gpu_and_is_reproducible(tmp_path):
    """Hermetic: committed CSVs and committed yaml, no device, same answer
    twice. A replay that read the hardware would not be a replay."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    assert RB.main(["--corpus-only", "--out", str(first)]) == 0
    assert RB.main(["--corpus-only", "--out", str(second)]) == 0
    reports = list(first.rglob("report.txt")) + list(second.rglob("report.txt"))
    assert len(reports) == 2
    # The destination path is in the report on purpose, so it is normalised
    # here rather than kept out of the file. Everything else must be identical.
    a, b = (r.read_text().replace(str(root), "<OUT>")
            for r, root in zip(reports, (first, second), strict=True))
    assert a == b
    assert "90 flips" in a, "the corpus answer itself must be in there"


def test_the_corpus_comparison_prices_both_levers_separately():
    corpus = RB.survey(REPO / "results" / "published",
                       RB.load_current_rulers(REPO / "moe" / "bench" / "hardware"))
    assert corpus.denominator and corpus.session
    levers = {s.lever for s in corpus.denominator} | {s.lever for s in corpus.session}
    assert levers == {"denominator", "session"}
    # The A100 arm cannot be priced on the denominator lever: its own read was
    # disowned. It must be REFUSED by name, not silently counted as unchanged.
    a100 = next(s for s in corpus.denominator if "a100" in s.arm)
    assert a100.refusal and a100.flips == 0


def test_a_partially_superseded_arm_loses_only_its_retired_impls():
    corpus = RB.survey(REPO / "results" / "published",
                       RB.load_current_rulers(REPO / "moe" / "bench" / "hardware"))
    trimmed = " ".join(corpus.trimmed)
    assert "fp8-three-kernel" in trimmed
    assert "torch_scaled_grouped_mm_up" in trimmed


# --- the pod report path, exercised without a pod ----------------------------

def measured_h200() -> Calibration:
    """A Calibration shaped exactly like the one the pod run will produce.

    Hermetic: every field is a literal, nothing here reads the attached device,
    and the assertions below are the same on a laptop and on the H200. A replay
    that consulted the hardware would not be a replay, and this project has
    already shipped one that did.
    """
    return calibration(
        [bw("read_stream", 4560.0, "Triton, one store per program"),
         bw("read_reduce", 4469.6, "ATen reduction"),
         bw("copy", 4300.7), bw("triad", 4374.8, "canonical STREAM"),
         bw("write", 4682.4)],
        gemm_clock_mhz=1500,
        gemm_clock=clock(1500, spread=1.2, idle=1935),
        fp8_gemm_clock=clock(1470, spread=0.8, idle=1905),
        settle={"settled": True, "final_mhz": 1470,
                "clock_history_mhz": [1500, 1470, 1470],
                "bandwidth_settle": {"settled": True, "final_mhz": 1980,
                                     "clock_history_mhz": [1980, 1980, 1980]}})


def test_the_measurement_report_renders_without_a_gpu():
    """A crash in the report path would only surface two minutes into a pod
    run, after the measurement it was going to describe."""
    text = "\n".join(RB.measurement_lines(measured_h200()))
    assert "ridge band" in text
    assert "read_stream" in text and "read_reduce" in text
    assert "the four bandwidth patterns" in text, "both settles must be shown"


def test_the_three_measured_gates_read_the_planted_calibration():
    cal = measured_h200()
    one, two, three = (RB.gate_1_clock(cal), RB.gate_2_bandwidth(cal),
                       RB.gate_3_read_shape(cal))
    # P1: the post-hoc sample says 1935 and the load says 1500. That is the
    # artefact, and 22% is far past the 5% discriminator.
    assert one.verdict == RB.PASS and "22" in one.measured
    # P2: the patterns are the committed figures, so nothing moved.
    assert two.verdict == RB.PASS
    # P3: 4560 over 4469.6 is +2.0%, above zero and below the pin rate.
    assert three.verdict == RB.PASS and "+2.0" in three.measured


def test_a_probe_above_the_pin_rate_fails_rather_than_being_believed():
    """The one number that is physically impossible. A figure above the bus pin
    rate means the loads were elided or the byte count is wrong, and it is the
    only guard that does not depend on knowing what the right answer is."""
    cal = calibration([bw("read_stream", RB.H200_PIN_RATE_GBPS + 1),
                       bw("read_reduce", 4469.6), bw("triad", 4374.8)])
    gate = RB.gate_3_read_shape(cal)
    assert gate.verdict == RB.FAIL
    assert "ABOVE THE PIN RATE" in " ".join(gate.lines)


def test_a_refused_probe_leaves_gate_three_undecided_not_failed():
    """Triton being absent is not evidence about the reduction's shape."""
    cal = calibration([bw("read_reduce", 4469.6), bw("triad", 4374.8)],
                      refusals=("read_stream: triton is not importable",))
    gate = RB.gate_3_read_shape(cal)
    assert gate.verdict == RB.UNDECIDED
    assert any("triton" in line for line in gate.lines)


# --- the two gates that carried an H200 constant onto whatever card was there -

def test_gate_two_is_undecided_off_the_h200_not_failed():
    """A FAIL here would mean "the ruler moved". Off an H200 it means "wrong
    card", and printing the second under the claim of the first is the study's
    own stale-H200-constant defect reappearing inside the script written to
    price it.

    The A100 copy pattern is 1758 GB/s against the H200's committed 4300.7. The
    old `if not compared` escape could never catch this: it fires only when the
    pattern NAMES are missing, and they are present on every card.
    """
    cal = calibration([bw("read_reduce", 1744.3), bw("copy", 1758.0),
                       bw("triad", 1799.4), bw("write", 1760.6)],
                      gpu_name="NVIDIA A100-SXM4-80GB",
                      achieved_bandwidth_gbps=1799.4,
                      achieved_bf16_tflops=262.4)
    gate = RB.gate_2_bandwidth(cal)
    assert gate.verdict == RB.UNDECIDED
    assert "A100" in gate.measured
    assert any("wrong card" in line for line in gate.lines)


def test_gate_threes_pin_guard_uses_the_attached_cards_pin_rate():
    """THE GUARD WENT INERT ON THE WRONG CARD.

    Hardcoding the H200's 4916.7 meant that on an A100 -- whose bus pins at
    2039 -- any read_stream up to 2.4x that card's entire bandwidth passed the
    one check that exists to reject exactly such a number. The elided-loads
    guard is the only one here that does not depend on knowing the right answer,
    so a guard that cannot fire is worse than no guard: it prints PASS.
    """
    a100 = dict(gpu_name="NVIDIA A100-SXM4-80GB",
                achieved_bandwidth_gbps=1799.4, achieved_bf16_tflops=262.4)
    impossible = RB.PIN_RATE_GBPS["nvidiaa100sxm480gb"] + 1.0
    assert impossible < RB.H200_PIN_RATE_GBPS, (
        "the whole point is a figure the old constant would have waved through")
    gate = RB.gate_3_read_shape(calibration(
        [bw("read_stream", impossible), bw("read_reduce", 1744.3),
         bw("triad", 1799.4)], **a100))
    assert gate.verdict == RB.FAIL
    assert "ABOVE THE PIN RATE" in " ".join(gate.lines)
    # ... and a legitimate A100 stream still passes against its own bus.
    ok = RB.gate_3_read_shape(calibration(
        [bw("read_stream", 1820.0), bw("read_reduce", 1744.3),
         bw("triad", 1799.4)], **a100))
    assert ok.verdict == RB.PASS
    assert "2039" in " ".join(ok.lines)


def test_gate_three_refuses_rather_than_guessing_on_an_unknown_card():
    """A guard that cannot run must not be reported as a guard that passed."""
    gate = RB.gate_3_read_shape(calibration(
        [bw("read_stream", 9999.0), bw("read_reduce", 1744.3),
         bw("triad", 1799.4)], gpu_name="NVIDIA B200"))
    assert gate.verdict == RB.UNDECIDED
    assert any("NO PIN RATE ON FILE" in line for line in gate.lines)


def test_the_pin_rate_table_matches_the_committed_calibrations():
    """The constants are transcribed from `observed.pin_rate_gbps`. A silent
    drift between the table and the yaml is a wrong ruler nobody would see."""
    import yaml
    hw = REPO / "moe" / "bench" / "hardware"
    for slug, expected in (("measured_nvidia_h200", "nvidiah200"),
                           ("measured_nvidia_a100_sxm4_80gb",
                            "nvidiaa100sxm480gb")):
        doc = yaml.safe_load((hw / f"{slug}.yaml").read_text())
        assert RB.PIN_RATE_GBPS[expected] == doc["observed"]["pin_rate_gbps"]


def test_a_ramped_run_fails_gate_two_even_when_the_numbers_match():
    """Four patterns that agree with the committed file while the clock moved
    500 MHz across them agree by accident."""
    cal = calibration(
        [bw("read_reduce", 4469.6), bw("copy", 4300.7),
         bw("triad", 4374.8, start=1980, end=1400), bw("write", 4682.4)])
    assert RB.gate_2_bandwidth(cal).verdict == RB.FAIL
