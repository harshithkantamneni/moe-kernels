"""Whether a SAMPLED GPU metric can stand in for the DRAM counter ncu refuses.

THE QUESTION THESE TESTS PIN DOWN. Every byte figure in this study is
compulsory-traffic arithmetic, and `alpha` -- refit 2026-08-31 from 0.10 to
0.558, and the parameter the whole tile-corrected roofline rests on -- is fitted
through that same unvalidated model. `ncu` would count real bytes and cannot on
a rented pod (ERR_NVGPUCTRPERM). `nsys --gpu-metrics-device` SAMPLES the hardware
monitor through a different path, and docs/FINDINGS.md calls it "the open path,
not a closed door".

The half of that path that can be settled without a GPU is settled here, and the
answer it reaches is mostly ARITHMETIC rather than empirical:

  * A single 54 us launch, which is what a T=1 fused MoE cell costs on an H200,
    holds 0.54 samples at the nsys default and 10.8 at the documented ceiling.
    It is not measurable at any rate the tool offers, and no pod run can change
    that. `test_the_cell_this_study_cares_about_is_unmeasurable_as_a_single_launch`.
  * The quantisation is charged per WINDOW, so profiling a thousand separate
    launches buys samples and NO accuracy. The only route to a usable number is
    to make the launches contiguous so they merge into one window, which is why
    the workloads in `scripts/nsys_dram_probe.py` never synchronise inside their
    timed loop. `test_the_edge_term_is_charged_per_window_...`.
  * A sampled metric may be a PERCENTAGE OF PEAK rather than a byte count, in
    which case converting it needs the measured DRAM ceiling and the answer
    inherits that calibration's uncertainty. It stays independent of the
    compulsory byte model, which is the thing under test, and the route travels
    with the number so the two can never be confused.

WHAT IS NOT TESTED HERE, and it is the important half. Nothing below proves nsys
samples anything on any real device, or that a device-wide sampler can be
attributed to one kernel on a shared pod. Those need the pod, and
`scripts/nsys_dram_probe.py` is what asks. These tests fix the arithmetic, the
refusals and the parse so that when the pod answers, the answer is readable.

The sqlite fixtures below mirror the schema `nsys export --type sqlite` writes.
Any real report that differs produces a `ReportSchemaUnsupported` naming what it
found instead of an OperationalError from three frames down, which is what
`test_a_report_with_no_metrics_table_...` asserts.
"""
from __future__ import annotations

import sqlite3

import pytest

from moe.bench.nsys_metrics import (
    DEFAULT_SAMPLE_HZ,
    GPU_METRICS_DEVICE_FLAGS,
    MAX_SAMPLE_HZ,
    SHORT_KERNEL_US,
    CeilingRequired,
    GpuMetricsUnsupported,
    Metric,
    MetricNotFound,
    MetricUnit,
    NsysProbeRefused,
    NsysUnavailable,
    RateUnverifiable,
    ReportSchemaUnsupported,
    UnitUnknown,
    Window,
    WindowTooShort,
    alpha_uncertainty,
    calibration_verdict,
    classify_unit,
    compare_to_model,
    dram_traffic,
    duty_cycle,
    find_dram_metrics,
    kernel_windows,
    longest_window,
    merge_windows,
    metric_catalogue,
    observed_sample_hz,
    open_report,
    parse_gpu_metrics_support,
    parse_metric_sets,
    parse_nsys_version,
    resolve,
    select_metric_set,
    single_launch_verdict,
    sum_samples,
    to_bytes,
    unit_scale,
)

#: A convenient round rate: 10 us per sample. Not one nsys offers as a default,
#: chosen so the fixtures below have exact sample counts and a reader can check
#: the arithmetic in their head.
FIXTURE_HZ = 100_000.0
FIXTURE_PERIOD_NS = 10_000

BYTE_METRIC = (7, "DRAM Read Bytes", "bytes")
WRITE_METRIC = (8, "DRAM Write Bytes", "bytes")
PERCENT_METRIC = (9, "DRAM Read Bandwidth [Throughput %]", "%")


def write_report(path, kernels, metrics, samples, with_type_id=True):
    """Build a database shaped like `nsys export --type sqlite` output.

    `kernels` is [(start_ns, end_ns, name)], `metrics` is
    [(metric_id, name, unit)], `samples` is [(timestamp_ns, metric_id, value)].
    Names go through `StringIds` exactly as nsys interns them, so the parser's
    string indirection is exercised rather than bypassed.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                 "(start INTEGER, end INTEGER, demangledName INTEGER)")
    conn.execute("CREATE TABLE TARGET_INFO_GPU_METRICS "
                 "(typeId INTEGER, metricId INTEGER, metricName TEXT, unit TEXT)")
    cols = ("timestamp INTEGER, typeId INTEGER, metricId INTEGER, value INTEGER"
            if with_type_id else "timestamp INTEGER, metricId INTEGER, value INTEGER")
    conn.execute(f"CREATE TABLE GPU_METRICS ({cols})")
    names: dict[str, int] = {}
    for start, end, name in kernels:
        if name not in names:
            names[name] = len(names) + 1
            conn.execute("INSERT INTO StringIds VALUES (?, ?)", (names[name], name))
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
                     (start, end, names[name]))
    for metric_id, name, unit in metrics:
        conn.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?, ?)",
                     (1, metric_id, name, unit))
    for ts, metric_id, value in samples:
        if with_type_id:
            conn.execute("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)",
                         (ts, 1, metric_id, value))
        else:
            conn.execute("INSERT INTO GPU_METRICS VALUES (?, ?, ?)",
                         (ts, metric_id, value))
    conn.commit()
    conn.close()
    return path


def evenly_sampled(span_ns, metric_id, value_inside, windows, value_outside=0):
    """One sample every period across `span_ns`, valued by whether it is in a window."""
    out = []
    for ts in range(0, span_ns, FIXTURE_PERIOD_NS):
        inside = any(w[0] <= ts < w[1] for w in windows)
        out.append((ts, metric_id, value_inside if inside else value_outside))
    return out


# --------------------------------------------------------------------------
# The sampling arithmetic. This section is the actual answer to the task, and
# none of it needs a GPU, a pod or an nsys.
# --------------------------------------------------------------------------


def test_the_cell_this_study_cares_about_is_unmeasurable_as_a_single_launch():
    """A 54 us kernel is under one sample period at the default and 11 at the max.

    This is the finding, not a limitation of the parser. At the nsys default of
    10 kHz the period is 100 us, so the kernel can finish between two samples
    and contribute NOTHING. Raising to the documented 200 kHz ceiling buys about
    eleven samples, which carries an 18% edge term: enough to see that DRAM is
    busy, nowhere near enough to measure alpha.
    """
    at_default = single_launch_verdict(SHORT_KERNEL_US, DEFAULT_SAMPLE_HZ)
    assert at_default.n_samples == pytest.approx(0.54, abs=0.01)
    assert not at_default.ok

    at_ceiling = single_launch_verdict(SHORT_KERNEL_US, MAX_SAMPLE_HZ)
    assert at_ceiling.n_samples == pytest.approx(10.8, abs=0.05)
    assert not at_ceiling.ok
    assert "under the 20" in at_ceiling.reason


def test_the_edge_term_is_charged_per_window_so_unmerged_launches_never_improve():
    """Ten milliseconds of kernel time is usable as one window and useless as ten.

    The trap this catches is real and is what a first attempt would do: launch
    the cell a thousand times, note that thousands of samples landed inside a
    kernel, and quote the sum. Every launch brings its own two straddled sample
    periods, so the SYSTEMATIC error scales with the launch count and does not
    average down however many samples are collected.
    """
    merged = resolve(int(10e6), 1, DEFAULT_SAMPLE_HZ)
    split = resolve(int(10e6), 10, DEFAULT_SAMPLE_HZ)
    assert merged.n_samples == split.n_samples == pytest.approx(100.0)
    assert merged.ok and merged.edge_error == pytest.approx(0.02)
    assert not split.ok and split.edge_error == pytest.approx(0.20)
    assert "back to back" in split.reason


def test_back_to_back_launches_merge_when_the_gap_is_under_a_sample_period():
    """Merging is legitimate exactly when the sampler could not have seen the gap."""
    launches = [Window(0, 50_000), Window(51_000, 100_000), Window(101_000, 150_000)]
    merged = merge_windows(launches, FIXTURE_PERIOD_NS)
    assert len(merged) == 1
    assert merged[0].start_ns == 0 and merged[0].end_ns == 150_000
    assert merged[0].n_launches == 3


def test_merging_refuses_to_bridge_a_gap_longer_than_a_sample_period():
    """A gap the sampler COULD resolve is real idle time and folding it in would
    count duration that moved nothing, biasing traffic down."""
    launches = [Window(0, 50_000), Window(500_000, 550_000)]
    assert len(merge_windows(launches, FIXTURE_PERIOD_NS)) == 2
    assert duty_cycle(launches) == pytest.approx(100_000 / 550_000)


def test_no_matched_window_is_a_refusal_and_not_a_zero():
    got = resolve(0, 0, DEFAULT_SAMPLE_HZ)
    assert not got.ok
    assert "nothing to sum" in got.reason


def test_alpha_is_unidentifiable_at_a_single_m_tile():
    """`Q(1) = 1` for every alpha, which is why the refit needed the multi-tile
    regime and why `scripts/tile_sweep.py` could not answer this question."""
    with pytest.raises(ValueError, match="unidentifiable"):
        alpha_uncertainty(0.05, n_tiles=1)


def test_more_m_tiles_per_expert_sharpen_alpha_at_a_fixed_traffic_error():
    """The design advice this function exists to make quantitative.

    A probe that cannot resolve a two-tile cell may still resolve an eight-tile
    one, because alpha is `(Q - 1) / (n - 1)` and the divisor grows.
    """
    bands = [alpha_uncertainty(0.10, n_tiles=n) for n in (2, 4, 8, 16)]
    assert bands == sorted(bands, reverse=True)
    assert bands[0] == pytest.approx(0.1558, abs=1e-4)
    assert bands[-1] == pytest.approx(0.0625, abs=1e-3)


def test_ten_percent_traffic_error_cannot_pin_alpha_but_can_choose_between_worlds():
    """The two candidate alphas are 0.10 and 0.558, so they are 0.458 apart.

    Reported as a test because it is the sentence a reader wants: a sampler good
    to 10% puts alpha within +/-0.156 at two tiles, which is far too coarse to
    quote a value and comfortably enough to say which of the two worlds is the
    one we are in.
    """
    band = alpha_uncertainty(0.10, n_tiles=2)
    assert band < abs(0.558 - 0.10) / 2      # discriminates
    assert band > 0.05                       # but does not pin


# --------------------------------------------------------------------------
# Asking the installed nsys what it offers, rather than guessing a flag.
# --------------------------------------------------------------------------


def test_the_version_is_read_from_the_binarys_own_output():
    got = parse_nsys_version(
        "NVIDIA Nsight Systems version 2024.6.1.90-246134139542v0\n")
    assert got.as_tuple == (2024, 6, 1)
    assert "2024.6.1.90" in got.raw


def test_a_build_counter_is_not_mistaken_for_a_version():
    """`1.2.3` is not a version year, and a probe that cannot name its tool has
    nothing to report, so this refuses rather than guessing."""
    with pytest.raises(NsysUnavailable, match="could not find a version"):
        parse_nsys_version("nsys: build 1.2.3\n")


def test_the_newer_gpu_metrics_flag_spelling_wins_when_both_appear():
    """Renamed in 2024.5. Both are searched for in the help text of the binary
    that will actually be run, because a distro package can carry either."""
    support = parse_gpu_metrics_support(
        "  --gpu-metrics-devices=...\n  --gpu-metrics-device=... (deprecated)\n"
        "  --gpu-metrics-set=...\n  --gpu-metrics-frequency=...\n")
    assert support.device_flag == "--gpu-metrics-devices"
    assert support.set_flag == "--gpu-metrics-set"
    assert support.frequency_flag == "--gpu-metrics-frequency"
    assert set(support.offered) == set(GPU_METRICS_DEVICE_FLAGS) | {
        "--gpu-metrics-set", "--gpu-metrics-frequency"}


def test_the_older_spelling_is_accepted_on_its_own():
    support = parse_gpu_metrics_support("  --gpu-metrics-device=all\n")
    assert support.device_flag == "--gpu-metrics-device"
    assert support.set_flag is None


def test_a_build_with_no_gpu_metrics_flag_is_refused_and_says_what_it_offered():
    """A real state: some slim container packages ship the trace path only.

    The refusal names what WAS there, because "nsys has no sampler" and "the
    flag is spelled differently here" have opposite fixes and this repo has
    already lost a run to branching on presence instead of capability.
    """
    with pytest.raises(GpuMetricsUnsupported) as exc:
        parse_gpu_metrics_support("  --trace=cuda\n  --sample=none\n")
    assert "cannot sample GPU hardware metrics" in str(exc.value)
    assert "Tracing still works" in str(exc.value)


def test_the_metric_set_for_an_h200_is_the_gh100_one():
    sets = parse_metric_sets(
        "Possible --gpu-metrics-set values are:\n"
        "\t[0] [ga100] General Metrics for NVIDIA GA100\n"
        "\t[13] [gh100] General Metrics for NVIDIA GH100\n")
    assert len(sets) == 2
    assert select_metric_set(sets, "NVIDIA H200").chip == "gh100"
    assert select_metric_set(sets, "NVIDIA A100-SXM4-80GB").chip == "ga100"


def test_an_unknown_device_lets_nsys_choose_its_own_set():
    """None is not a failure: nsys picks a set unasked, and naming one exists
    only so the report can say which was used."""
    sets = parse_metric_sets("\t[13] [gh100] General Metrics for NVIDIA GH100\n")
    assert select_metric_set(sets, "NVIDIA GeForce RTX 4090") is None


# --------------------------------------------------------------------------
# What a sampled value means. Getting this wrong is wrong by the peak
# bandwidth, a factor of thousands, and would read as a finding.
# --------------------------------------------------------------------------


def test_a_percent_of_peak_metric_is_recognised_as_one():
    assert classify_unit("DRAM Read Bandwidth [Throughput %]", None) \
        is MetricUnit.PERCENT_OF_PEAK
    assert classify_unit("dram__read_throughput.avg.pct_of_peak", "") \
        is MetricUnit.PERCENT_OF_PEAK


def test_a_percent_of_peak_metric_cannot_become_bytes_without_a_ceiling():
    """And the refusal names the file to take the ceiling from, because a
    datasheet peak would silently import a number this study measured as wrong
    by up to 10%."""
    with pytest.raises(CeilingRequired) as exc:
        to_bytes(MetricUnit.PERCENT_OF_PEAK, 5000.0, 100, FIXTURE_PERIOD_NS)
    assert "measured_<device>.yaml" in str(exc.value)


def test_percent_of_peak_becomes_bytes_through_the_measured_ceiling():
    """50% of a 4377 GB/s ceiling over 1000 samples of 10 us is 21.885 GB."""
    got = to_bytes(MetricUnit.PERCENT_OF_PEAK, 50.0 * 1000, 1000,
                   FIXTURE_PERIOD_NS, peak_bytes_per_s=4377e9)
    assert got == pytest.approx(21.885e9, rel=1e-9)


def test_an_unnamed_unit_is_refused_rather_than_guessed():
    with pytest.raises(UnitUnknown, match="factor of thousands"):
        to_bytes(MetricUnit.UNKNOWN, 1234.0, 10, FIXTURE_PERIOD_NS)


def test_a_gigabyte_per_second_metric_is_not_treated_as_bytes_per_second():
    """The units bug that would be wrong by 1e9 and look like a discovery."""
    assert classify_unit("DRAM Read Bandwidth", "GB/s") is MetricUnit.BYTES_PER_SECOND
    assert unit_scale("GB/s") == 1e9
    plain = to_bytes(MetricUnit.BYTES_PER_SECOND, 1000.0, 1000, FIXTURE_PERIOD_NS)
    scaled = to_bytes(MetricUnit.BYTES_PER_SECOND, 1000.0, 1000, FIXTURE_PERIOD_NS,
                      scale=1e9)
    assert scaled == pytest.approx(plain * 1e9)


def test_a_combined_dram_metric_is_refused_because_reads_are_the_question():
    """Several nsys metric sets expose one DRAM figure with no direction split.

    That is a real outcome of this probe and not a parse failure: on such a chip
    nsys cannot answer "how many times were the WEIGHTS read" however well the
    sampler works, because writes are a different term of the byte model.
    """
    with pytest.raises(MetricNotFound) as exc:
        find_dram_metrics((Metric(1, 3, "DRAM Bandwidth [Throughput %]", "%"),))
    assert "no sampled metric names a DRAM READ" in str(exc.value)
    assert "DRAM Bandwidth [Throughput %]" in str(exc.value)


def test_a_byte_valued_read_metric_is_preferred_over_a_percentage_one():
    """A byte count needs nothing but itself; a percentage drags the
    calibration's uncertainty into the answer, so it is the last resort rather
    than whichever the catalogue happened to list first."""
    catalogue = (Metric(1, 9, *PERCENT_METRIC[1:]), Metric(1, 7, *BYTE_METRIC[1:]))
    assert find_dram_metrics(catalogue).read.name == "DRAM Read Bytes"


def test_a_read_only_metric_set_reports_no_write_rather_than_a_zero():
    found = find_dram_metrics((Metric(1, 7, *BYTE_METRIC[1:]),))
    assert found.read.name == "DRAM Read Bytes"
    assert found.write is None


# --------------------------------------------------------------------------
# Parsing what nsys wrote.
# --------------------------------------------------------------------------


def test_a_report_with_no_metrics_table_says_so_instead_of_raising_from_sqlite(tmp_path):
    """The single most likely real failure: the run collected a TRACE and no
    metrics, because the gpu-metrics flag was rejected or never passed."""
    db = write_report(tmp_path / "trace.sqlite",
                      kernels=[(0, 54_000, "cutlass_grouped")], metrics=[], samples=[])
    conn = sqlite3.connect(str(db))
    conn.execute("DROP TABLE GPU_METRICS")
    conn.execute("DROP TABLE TARGET_INFO_GPU_METRICS")
    conn.commit()
    conn.close()
    with pytest.raises(ReportSchemaUnsupported) as exc:
        dram_traffic(open_report(db), sample_hz=FIXTURE_HZ, allow_unresolved=True)
    assert "collected WITHOUT the gpu-metrics flag" in str(exc.value)


def test_a_missing_report_file_names_the_export_command(tmp_path):
    with pytest.raises(ReportSchemaUnsupported, match="nsys export --type sqlite"):
        open_report(tmp_path / "absent.sqlite")


def test_dram_bytes_are_summed_only_inside_the_kernel_window(tmp_path):
    """Ten milliseconds of kernel inside a twenty millisecond trace.

    1000 samples land inside at 1000 bytes each, so the measurement is exactly
    1.0 MB and the 1000 samples outside are excluded however large they are.
    """
    window = (0, 10_000_000)
    db = write_report(
        tmp_path / "r.sqlite",
        kernels=[(window[0], window[1], "reduce_kernel")],
        metrics=[BYTE_METRIC, WRITE_METRIC],
        samples=(evenly_sampled(20_000_000, BYTE_METRIC[0], 1000, [window], 0)
                 + evenly_sampled(20_000_000, WRITE_METRIC[0], 4, [window], 0)))
    got = dram_traffic(open_report(db), sample_hz=FIXTURE_HZ)
    assert got.read_bytes == pytest.approx(1_000_000)
    assert got.write_bytes == pytest.approx(4_000)
    assert got.n_samples == 1000
    assert got.resolution.ok
    assert got.resolution.edge_error == pytest.approx(0.002)
    assert got.idle_read_bytes_per_s == 0.0
    assert "bytes accumulated in one sample period" in got.route


def test_traffic_outside_every_kernel_becomes_the_idle_baseline(tmp_path):
    """The isolation check on a device-wide sampler.

    A sampler cannot tell our kernel from a neighbour's, but it can say what the
    device was reading when we had launched nothing. Here the trace is half
    kernel and half idle, and the idle half is reading 100 bytes per 10 us,
    which is 10 MB/s and has to appear as a baseline rather than vanishing.
    """
    window = (0, 10_000_000)
    db = write_report(
        tmp_path / "r.sqlite",
        kernels=[(window[0], window[1], "reduce_kernel")],
        metrics=[BYTE_METRIC],
        samples=evenly_sampled(20_000_000, BYTE_METRIC[0], 1000, [window], 100))
    got = dram_traffic(open_report(db), sample_hz=FIXTURE_HZ)
    assert got.read_bytes == pytest.approx(1_000_000)
    assert got.idle_read_bytes_per_s == pytest.approx(1e7)
    assert "GB/s read while nothing of ours was running" in got.text()


def test_a_single_short_launch_is_refused_rather_than_returned_unsupported(tmp_path):
    """The whole point. A 54 us window at 100 kHz holds five samples, and a
    number built on five samples with a 40% edge term is not a measurement."""
    db = write_report(
        tmp_path / "r.sqlite",
        kernels=[(1_000_000, 1_054_000, "fused_moe_kernel")],
        metrics=[BYTE_METRIC],
        samples=evenly_sampled(3_000_000, BYTE_METRIC[0], 1000,
                               [(1_000_000, 1_054_000)]))
    with pytest.raises(WindowTooShort) as exc:
        dram_traffic(open_report(db), sample_hz=FIXTURE_HZ)
    assert "not measurable at any rate nsys offers" in str(exc.value)
    assert "10.8 samples" in str(exc.value)


def test_the_refusal_can_be_overridden_but_the_verdict_travels_with_the_number(tmp_path):
    """`allow_unresolved` exists for characterising the sampler itself, and the
    result still carries `ok=False` so the figure cannot be quoted clean."""
    db = write_report(
        tmp_path / "r.sqlite",
        kernels=[(1_000_000, 1_054_000, "fused_moe_kernel")],
        metrics=[BYTE_METRIC],
        samples=evenly_sampled(3_000_000, BYTE_METRIC[0], 1000,
                               [(1_000_000, 1_054_000)]))
    got = dram_traffic(open_report(db), sample_hz=FIXTURE_HZ, allow_unresolved=True)
    assert not got.resolution.ok
    assert "NOT USABLE" in got.resolution.text()
    scored = compare_to_model(got, modelled_bytes_per_launch=5000.0, launches=1)
    assert scored.verdict.startswith("UNRESOLVED")


def test_a_kernel_pattern_that_matches_nothing_says_how_many_it_examined(tmp_path):
    """A pattern that silently matches nothing would report zero traffic, which
    is the failure mode this module was written to make impossible."""
    db = write_report(tmp_path / "r.sqlite",
                      kernels=[(0, 10_000_000, "reduce_kernel")],
                      metrics=[BYTE_METRIC], samples=[])
    with pytest.raises(MetricNotFound) as exc:
        kernel_windows(open_report(db), "cutlass")
    assert "1 launches were examined" in str(exc.value)


def test_the_longest_window_is_the_timed_loop_and_not_the_warmup(tmp_path):
    """The workloads warm up, go idle for a beat, then run the timed loop back
    to back, so selecting the longest contiguous window selects the measurement
    without depending on a kernel name that changes across torch versions."""
    warmup = [(0, 100_000, "k"), (110_000, 200_000, "k")]
    loop = [(5_000_000 + i * 100_000, 5_000_000 + i * 100_000 + 99_000, "k")
            for i in range(50)]
    db = write_report(tmp_path / "r.sqlite", kernels=warmup + loop,
                      metrics=[BYTE_METRIC], samples=[])
    merged = merge_windows(kernel_windows(open_report(db)), FIXTURE_PERIOD_NS)
    chosen = longest_window(merged)
    assert len(chosen) == 1
    assert chosen[0].n_launches == 50
    assert chosen[0].start_ns == 5_000_000


def test_a_report_without_a_type_id_column_is_still_readable(tmp_path):
    """nsys's export schema has moved across versions, so every column is
    introspected before it is read rather than assumed present."""
    window = (0, 10_000_000)
    db = write_report(
        tmp_path / "r.sqlite", kernels=[(*window, "k")], metrics=[BYTE_METRIC],
        samples=evenly_sampled(10_000_000, BYTE_METRIC[0], 1000, [window]),
        with_type_id=False)
    conn = open_report(db)
    assert metric_catalogue(conn)[0].name == "DRAM Read Bytes"
    got = sum_samples(conn, Metric(0, BYTE_METRIC[0], BYTE_METRIC[1], "bytes"),
                      [Window(*window)], FIXTURE_PERIOD_NS)
    assert got.total == pytest.approx(1_000_000)


def test_boundary_samples_are_counted_so_the_edge_term_can_be_checked(tmp_path):
    """The 2/n arithmetic is a bound; this is the population it describes,
    reported as a count so a reader does not have to trust the formula."""
    window = (0, 10_000_000)
    db = write_report(
        tmp_path / "r.sqlite", kernels=[(*window, "k")], metrics=[BYTE_METRIC],
        samples=evenly_sampled(10_000_000, BYTE_METRIC[0], 1000, [window]))
    got = dram_traffic(open_report(db), sample_hz=FIXTURE_HZ)
    assert got.n_boundary_samples == 2       # the first sample and the last
    assert got.n_samples == 1000


# --------------------------------------------------------------------------
# Against the byte model, which is the only form the output is allowed to take.
# --------------------------------------------------------------------------


def measured(tmp_path, per_sample_bytes, idle_bytes=0):
    window = (0, 10_000_000)
    db = write_report(
        tmp_path / f"r{per_sample_bytes}-{idle_bytes}.sqlite",
        kernels=[(*window, "k")], metrics=[BYTE_METRIC],
        samples=evenly_sampled(20_000_000, BYTE_METRIC[0], per_sample_bytes,
                               [window], idle_bytes))
    return dram_traffic(open_report(db), sample_hz=FIXTURE_HZ)


def test_measured_over_modelled_is_reported_with_a_named_error_budget(tmp_path):
    """The output is never a byte count on its own: the study's question is
    whether the compulsory model is right, so the modelled figure sits beside
    the measured one or it is not an answer."""
    got = compare_to_model(measured(tmp_path, 1000), 10_000.0, launches=100)
    assert got.modelled_bytes == pytest.approx(1_000_000)
    assert got.ratio == pytest.approx(1.0)
    assert dict(got.terms)["edge"] == pytest.approx(0.002)
    assert dict(got.terms)["idle"] == 0.0
    assert "consistent with the compulsory model" in got.verdict
    assert "modelled" in got.text() and "measured" in got.text()


def test_traffic_above_the_floor_is_named_as_traffic_the_model_does_not_account_for(
        tmp_path):
    """A ratio of 1.5 with a 0.2% band is re-read, which is exactly what alpha
    is, and is the shape of answer this whole exercise wants."""
    got = compare_to_model(measured(tmp_path, 1500), 10_000.0, launches=100)
    assert got.ratio == pytest.approx(1.5)
    assert got.ratio_low > 1.0
    assert "above the compulsory floor" in got.verdict


def test_a_ratio_below_one_blames_the_attribution_before_the_model(tmp_path):
    """Below the compulsory floor is impossible for a correct model AND correct
    attribution, and on a device-wide sampler over a partial window the
    attribution is the thing that undercounts, so the verdict says so rather
    than reporting a kernel beating its own byte floor."""
    got = compare_to_model(measured(tmp_path, 400), 10_000.0, launches=100)
    assert got.ratio == pytest.approx(0.4)
    assert "BELOW the compulsory floor" in got.verdict
    assert "suspect the attribution first" in got.verdict


def test_the_idle_baseline_enters_the_error_budget_rather_than_a_footnote(tmp_path):
    """Contamination on a device-wide counter is bounded by what the sampler saw
    while nothing of ours ran, so it belongs in the band and not in prose."""
    clean = compare_to_model(measured(tmp_path, 1000, idle_bytes=0), 10_000.0, 100)
    dirty = compare_to_model(measured(tmp_path, 1000, idle_bytes=200), 10_000.0, 100)
    assert dict(clean.terms)["idle"] == 0.0
    assert dict(dirty.terms)["idle"] == pytest.approx(0.2)
    assert dirty.ratio_high - dirty.ratio_low > clean.ratio_high - clean.ratio_low


def test_a_zero_model_is_rejected_rather_than_dividing(tmp_path):
    with pytest.raises(ValueError, match="infinite ratio"):
        compare_to_model(measured(tmp_path, 1000), 0.0, launches=100)


def test_the_known_traffic_calibration_passes_on_a_case_with_a_known_answer(tmp_path):
    """A streaming read of a buffer far larger than L2 moves almost exactly its
    own size, so measured-over-known reads the sampler, the unit interpretation
    and the window attribution at once, with nothing modelled."""
    verdict = calibration_verdict(
        compare_to_model(measured(tmp_path, 1000), 10_000.0, launches=100))
    assert verdict.startswith("PASS")


def test_the_known_traffic_calibration_fails_loudly_when_the_number_is_wrong(tmp_path):
    """And names the three causes in the order worth checking them."""
    verdict = calibration_verdict(
        compare_to_model(measured(tmp_path, 400), 10_000.0, launches=100))
    assert verdict.startswith("FAIL")
    assert "wrong unit interpretation" in verdict
    assert "served by L2" in verdict


def test_a_read_only_metric_set_makes_the_ratio_a_lower_bound(tmp_path):
    """The compulsory model counts writes and this metric set does not measure
    them, so the ratio understates by whatever the writes were. Said on the
    verdict rather than in a docstring, because the number is what gets copied
    into a write-up and the caveat has to travel with it."""
    window = (0, 10_000_000)
    db = write_report(
        tmp_path / "readonly.sqlite", kernels=[(*window, "k")],
        metrics=[BYTE_METRIC],
        samples=evenly_sampled(10_000_000, BYTE_METRIC[0], 1000, [window]))
    got = dram_traffic(open_report(db), sample_hz=FIXTURE_HZ)
    assert got.write_bytes is None
    assert "not split by this metric set" in got.text()
    assert "LOWER bound" in compare_to_model(got, 10_000.0, launches=100).verdict


# --------------------------------------------------------------------------
# The delivered sampling rate. `resolve()` computes its sample count from the
# rate that was REQUESTED, so a build that silently clamps the frequency flag
# would leave every resolution verdict overconfident by the clamp factor while
# the traffic total stayed correct. These pin the detector for that.
# --------------------------------------------------------------------------


def _rate_db(tmp_path, period_ns, n=200, metric=BYTE_METRIC):
    return write_report(
        tmp_path / "rate.sqlite",
        kernels=[(0, period_ns * n, "moe")],
        metrics=[metric],
        samples=[(i * period_ns, metric[0], 1000) for i in range(n)])


def _metric_of(db):
    conn = open_report(db)
    return conn, next(m for m in metric_catalogue(conn) if m.metric_id == BYTE_METRIC[0])


def test_observed_rate_confirms_an_honoured_request(tmp_path):
    # 100 kHz requested, 10 us period delivered: the flag was honoured.
    conn, metric = _metric_of(_rate_db(tmp_path, period_ns=10_000))
    got = observed_sample_hz(conn, metric, requested_hz=100_000.0)
    assert got.honoured
    assert got.observed_hz == pytest.approx(100_000.0)
    assert abs(got.rel_error) < 1e-9
    assert "stands" in got.detail


def test_observed_rate_catches_a_silent_clamp(tmp_path):
    # Asked for 200 kHz, got 50 kHz: a 4x clamp, exactly the failure that would
    # otherwise inflate every sample count fourfold and shrink every edge term.
    conn, metric = _metric_of(_rate_db(tmp_path, period_ns=20_000))
    got = observed_sample_hz(conn, metric, requested_hz=200_000.0)
    assert not got.honoured
    assert got.observed_hz == pytest.approx(50_000.0)
    assert got.rel_error == pytest.approx(-0.75)
    assert "1/4.0" in got.detail
    assert "overstated" in got.detail


def test_observed_rate_uses_the_median_so_a_dropped_sample_does_not_move_it(
        tmp_path):
    # One gap of three periods where a sample went missing. A mean would report
    # a rate about 2% low and read as honoured-but-drifting; the median is exact.
    period = 10_000
    stamps = [i * period for i in range(200)]
    del stamps[100:102]
    db = write_report(
        tmp_path / "gap.sqlite",
        kernels=[(0, stamps[-1], "moe")],
        metrics=[BYTE_METRIC],
        samples=[(t, BYTE_METRIC[0], 1000) for t in stamps])
    conn = open_report(db)
    metric = next(m for m in metric_catalogue(conn) if m.metric_id == BYTE_METRIC[0])
    got = observed_sample_hz(conn, metric, requested_hz=100_000.0)
    assert got.observed_hz == pytest.approx(100_000.0)
    assert got.honoured


def test_observed_rate_refuses_rather_than_guessing_from_too_few_samples(
        tmp_path):
    conn, metric = _metric_of(_rate_db(tmp_path, period_ns=10_000, n=4))
    with pytest.raises(RateUnverifiable) as exc:
        observed_sample_hz(conn, metric, requested_hz=100_000.0)
    assert "NOT verified" in str(exc.value)
    assert "3 usable sample interval" in str(exc.value)


def test_rate_unverifiable_is_a_refusal_not_a_coarseness_report():
    # WindowTooShort says "too coarse to report"; this says "we cannot tell how
    # coarse". Conflating them would let an unmeasured rate pass as a wide one.
    assert issubclass(RateUnverifiable, NsysProbeRefused)
    assert not issubclass(RateUnverifiable, WindowTooShort)
