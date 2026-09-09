"""The one timing instrument, exercised off-GPU against fakes.

`time_kernel` is the apparatus every published number is supposed to come
from, and its failure modes are silent: an event structure that lets host
time into the interval, a callable whose host side outruns its GPU side so
the queue drains whatever the events do, and a clock flag that passes a card
running at 1500 MHz against a roof measured at 1980. None of them leaves a
mark on the number. So every branch here is planted with a fake GPU whose
per-call durations, host cost, backlog and clock trace are scripted, and the
assertions are on what the instrument DID (order of records, count of
synchronises, calls made during warmup, wall clocks it compared), not only on
what it returned.

The one GPU test at the bottom is the audit's acceptance case: a ~55 us kernel
(8192x8192 bf16, 128 MiB in and out) measured by `time_kernel` agrees with
`time_eager` within 2%, because both run the same `_timed_trials` loop.
"""
import itertools
import statistics
import time

import pytest
import torch

from moe.bench import timing as T

# --- fakes -------------------------------------------------------------------

class _FakeEvent:
    def __init__(self, pairs, kind: str, i: int):
        self._pairs, self._kind, self._i = pairs, kind, i

    def record(self) -> None:
        self._pairs.gpu.on_record(self._pairs, self._kind, self._i)


class _FakePairs:
    """Stands in for `_EventPairs`: primed at construction, scripted `elapsed`."""

    def __init__(self, gpu, n: int):
        self.gpu, self.n = gpu, n
        self.starts = [_FakeEvent(self, "start", i) for i in range(n)]
        self.ends = [_FakeEvent(self, "end", i) for i in range(n)]
        self.ms = [0.0] * n
        self.syncs = 0
        gpu.log.append(("events", n))

    def synchronize(self) -> None:
        self.syncs += 1
        self.gpu.log.append(("sync", self.n))
        if self.n > 1 and self.gpu.sync_sleep_s > 0:
            # The GPU's backlog at the end of a trial: the host waits here.
            time.sleep(self.gpu.sync_sleep_s)

    def elapsed(self, n: int) -> list[float]:
        return list(self.ms[:n])


class FakeGPU:
    """A GPU whose kernel takes a scripted time per call.

    `fn` is the callable under test; each call consumes the next duration
    from `warm_call_ms` until the clock sampler enters (the warmup batches
    AND the flushed probe that sizes `iters` are both warmup), then from
    `timed_call_ms` while the trials run, and adds it to the open interval.
    `events` is the factory, `flush` the flusher. Keyed on the sampler's
    phase rather than on the pair count because the v3 probe is a multi-pair
    interval that is still warmup.

    Two real-time knobs plant the host-bound branches: `host_call_s` makes
    each `fn` call cost the host that much wall (a slow Python launcher), and
    `sync_sleep_s` makes each trial's synchronise wait that long (the GPU
    still working through a deep queue after the host finished).
    """

    def __init__(self, warm_call_ms=1.0, timed_call_ms=1.0,
                 host_call_s: float = 0.0, sync_sleep_s: float = 0.0):
        self._warm = self._iter(warm_call_ms)
        self._timed = self._iter(timed_call_ms)
        self.host_call_s = host_call_s
        self.sync_sleep_s = sync_sleep_s
        self.log: list = []
        self.calls = 0
        self.flushes = 0
        self.megabytes = 240
        self._open = None
        self.in_trials = False
        self.instances: list[_FakePairs] = []

    @staticmethod
    def _iter(v):
        return itertools.repeat(float(v)) if isinstance(v, (int, float)) else iter(v)

    def events(self, n: int) -> _FakePairs:
        inst = _FakePairs(self, n)
        self.instances.append(inst)
        return inst

    def on_record(self, pairs, kind, i) -> None:
        self.log.append((kind, pairs.n, i))
        if kind == "start":
            pairs.ms[i] = 0.0
            self._open = (pairs, i)
        else:
            assert self._open == (pairs, i), "end recorded without its start"
            self._open = None

    def fn(self) -> None:
        self.calls += 1
        self.log.append(("fn",))
        if self.host_call_s > 0:
            time.sleep(self.host_call_s)
        pairs, i = self._open if self._open else (None, None)
        if pairs is None:
            return
        pairs.ms[i] += next(self._timed if self.in_trials else self._warm)

    def flush(self) -> None:
        self.flushes += 1
        self.log.append(("flush",))

    @property
    def timed(self) -> _FakePairs:
        """The trials' pairs: the last multi-pair instance `time_kernel` made.

        The probe's pairs come before it and may be a single pair (a kernel
        slower than one warmup batch gets a one-call probe), so the trials'
        are identified by position, not by count.
        """
        multi = [p for p in self.instances if p.n > 1]
        assert multi, "expected the trials' pairs"
        return multi[-1]

    @property
    def probe(self) -> _FakePairs:
        """The pairs the flushed probe used: created just before the trials'."""
        k = self.instances.index(self.timed)
        assert k >= 1, "expected the probe's pairs before the trials' pairs"
        return self.instances[k - 1]


class ScriptedClocks:
    """Stands in for `BackgroundClockSampler`: a scripted trace, in MHz."""

    def __init__(self, gpu: FakeGPU, trace, note: str = "",
                 poll_cost_ms: float | None = None):
        self.gpu, self.trace = gpu, list(trace)
        self.samples: tuple = ()
        self.source = "scripted"
        self.note = note
        self.poll_cost_ms = poll_cost_ms

    def __enter__(self):
        self.gpu.log.append(("clock_on",))
        self.gpu.in_trials = True
        return self

    def __exit__(self, *exc):
        self.gpu.log.append(("clock_off",))
        self.gpu.in_trials = False
        self.samples = tuple(T.ClockState(m, 50) for m in self.trace)


def run(gpu: FakeGPU, trace=(1980, 1980, 1980), **kw) -> T.KernelTiming:
    kw.setdefault("warmup_ms", 50.0)
    kw.setdefault("target_ms", 20.0)
    kw.setdefault("trials", 2)
    return T.time_kernel(gpu.fn, events=gpu.events, flusher=gpu,
                         clock_sampler=ScriptedClocks(gpu, trace), **kw)


# --- (1) calibration honours target_ms ------------------------------------------

@pytest.mark.parametrize("per_call, target, expect", [
    (0.5, 200.0, 400),
    (1.0, 100.0, 100),
    (0.001, 200.0, 2000),       # capped at hi
    (50.0, 20.0, 10),           # floored at lo
])
def test_iters_are_sized_from_target_ms_and_the_warm_per_call_time(per_call, target, expect):
    gpu = FakeGPU(warm_call_ms=per_call, timed_call_ms=per_call)
    res = run(gpu, warmup_ms=per_call * 3, target_ms=target, trials=1)
    assert res.iters == expect
    assert gpu.timed.n == expect


# --- (2) warmup is a duration, not a count ------------------------------------

def test_warmup_runs_until_warmup_ms_of_delivered_time():
    fast = run(FakeGPU(warm_call_ms=1.0), warmup_ms=50.0)
    slow = run(FakeGPU(warm_call_ms=10.0), warmup_ms=50.0)
    for res in (fast, slow):
        assert res.warmup_ms >= 50.0
        # overshoot bounded by one batch plus the probe, which is at most one
        # batch's worth of calls
        assert res.warmup_ms < 50.0 + 2 * T.WARMUP_BATCH_MS
    # 1 ms kernel: batches of 1, 25, 25 -> 51 calls, then a probe of
    # min(WARMUP_PROBE_CALLS, 25) = 25. 10 ms kernel: 1, 2, 2 -> 5, probe 2.
    assert fast.warmup_calls == 51 + 25
    assert slow.warmup_calls == 5 + 2
    assert fast.warmup_calls != slow.warmup_calls, "a count would be the same"


def test_warmup_batches_keep_the_queue_deep():
    gpu = FakeGPU(warm_call_ms=0.01)
    report = T.warm_until(gpu.fn, 30.0, gpu.events)
    # first batch is one call, the rest are sized to ~WARMUP_BATCH_MS
    assert report.batches == 3
    # 1 + 2 * 2500 batch calls plus the probe, give or take the float
    # rounding of 25.0 / 0.01
    assert report.probe_calls == T.WARMUP_PROBE_CALLS
    assert abs(report.calls - report.probe_calls
               - (1 + 2 * T.WARMUP_BATCH_MS / 0.01)) <= 2
    assert report.per_call_ms == pytest.approx(0.01)


def test_warmup_batch_is_capped_for_a_microsecond_kernel():
    # A 1 us kernel would need 25,000 calls per batch to reach WARMUP_BATCH_MS;
    # the cap holds the batch at 10,000 (10 ms delivered), and the loop takes
    # more batches instead of a larger one.
    gpu = FakeGPU(warm_call_ms=0.001)
    report = T.warm_until(gpu.fn, 25.0, gpu.events)
    cap = T.WARMUP_BATCH_MAX_CALLS
    # batches: 1 call (0.001 ms), then cap, cap, cap -> 30.001 ms >= 25; then
    # the probe, min(WARMUP_PROBE_CALLS, cap) calls, counted as delivered load
    assert report.batches == 4
    assert report.probe_calls == T.WARMUP_PROBE_CALLS
    assert report.calls == 1 + 3 * cap + report.probe_calls
    assert report.delivered_ms == pytest.approx(
        0.001 * (1 + 3 * cap + report.probe_calls))
    assert report.delivered_ms >= 25.0


def test_zero_warmup_is_refused_not_defaulted():
    gpu = FakeGPU()
    with pytest.raises(T.TimingRefused, match="warmup_ms=0"):
        run(gpu, warmup_ms=0.0)


# --- (3) event structure: primed first, one sync per trial ---------------------

def test_events_are_primed_before_the_timed_region_and_synced_once_per_trial():
    gpu = FakeGPU()
    res = run(gpu, trials=3, target_ms=10.0)
    timed = gpu.timed
    assert timed.syncs == 3 == res.trials
    log = gpu.log
    created = log.index(("events", res.iters))
    first_start = log.index(("start", res.iters, 0))
    assert created < first_start, "pairs must be primed before the first start record"
    # Between two syncs of the trial instance there are exactly iters starts.
    trial_syncs = [k for k, e in enumerate(log) if e == ("sync", res.iters)]
    assert len(trial_syncs) == 3
    prev = created
    for k in trial_syncs:
        starts = [e for e in log[prev:k] if e[0] == "start" and e[1] == res.iters]
        assert len(starts) == res.iters
        prev = k
    assert res.samples == res.iters * 3


def test_flush_is_enqueued_before_every_start_record_and_outside_the_interval():
    gpu = FakeGPU()
    res = run(gpu, trials=2, target_ms=10.0)
    log = gpu.log
    n = res.iters
    for k, e in enumerate(log):
        if e[0] == "start" and e[1] == n:
            assert log[k - 1] == ("flush",)
    # every warmup call (batches and probe) was flushed too: the warmup runs
    # the loop the trials run, and the count is the whole story
    assert gpu.flushes == res.warmup_calls + n * 2
    assert res.flush_mb == 240 and res.l2_flush is True


def test_no_flush_when_l2_flush_is_off():
    gpu = FakeGPU()
    res = T.time_kernel(gpu.fn, warmup_ms=10.0, target_ms=10.0, trials=1,
                        l2_flush=False, events=gpu.events,
                        clock_sampler=ScriptedClocks(gpu, [1980]))
    assert gpu.flushes == 0
    assert res.flush_mb == 0 and res.l2_flush is False


def test_clock_is_sampled_during_the_trials_not_around_them():
    gpu = FakeGPU()
    res = run(gpu, trials=2, target_ms=10.0)
    log = gpu.log
    on, off = log.index(("clock_on",)), log.index(("clock_off",))
    first_start = log.index(("start", res.iters, 0))
    last_sync = max(k for k, e in enumerate(log) if e == ("sync", res.iters))
    assert on < first_start
    assert off > last_sync
    # and the warmup's own syncs all precede the sampler
    warm_syncs = [k for k, e in enumerate(log) if e == ("sync", 1)]
    assert max(warm_syncs) < on


# --- (3b) the host-bound verdict: both branches, on real wall clocks -----------

def test_host_bound_when_the_host_enqueue_outruns_the_gpu():
    # A 2 ms host cost per call against a "0.5 ms" kernel: the queue drains
    # on every call, the synchronise returns at once, and the backlog at the
    # end of the enqueue loop is nothing. This is the B0 shape (fused_experts
    # at T=1: 0.18 ms of Python per 0.04 ms of GPU).
    gpu = FakeGPU(warm_call_ms=10.0, timed_call_ms=0.5, host_call_s=0.002,
                  sync_sleep_s=0.0)
    res = run(gpu, warmup_ms=10.0, target_ms=5.0, trials=1)
    assert res.iters == 10
    assert res.host_bound is True
    assert res.host_enqueue_ms is not None and res.host_enqueue_ms >= 10 * 2.0
    assert "host-bound" in res.host_note and "upper bound" in res.host_note
    # The intervals themselves are the contaminated witness: they read the
    # scripted 0.5 ms and would never have flagged it.
    assert res.ms_p50 == 0.5


def test_not_host_bound_when_the_gpu_still_has_a_backlog_after_the_enqueue():
    # Enqueue costs the host nothing measurable; the synchronise then waits
    # 50 ms for the GPU to drain a queue that stayed deep. Backlog (50 ms) is
    # far above two iterations of the trial's per-iteration wall (~10 ms).
    gpu = FakeGPU(warm_call_ms=10.0, timed_call_ms=0.5, host_call_s=0.0,
                  sync_sleep_s=0.05)
    res = run(gpu, warmup_ms=10.0, target_ms=5.0, trials=1)
    assert res.iters == 10
    assert res.host_bound is False
    assert res.host_enqueue_ms is not None and res.host_enqueue_ms < 10.0
    assert res.host_note == ""


def test_host_bound_is_any_trial_not_the_median_trial():
    walls = [T.TrialWall(enqueue_s=0.001, wall_s=0.050),     # deep
             T.TrialWall(enqueue_s=0.001, wall_s=0.050),     # deep
             T.TrialWall(enqueue_s=0.020, wall_s=0.0205)]    # drained
    bound, ms, backlog, note = T.host_bound_verdict(walls, iters=10)
    assert bound is True
    assert "1 of 3 trials" in note
    assert ms == pytest.approx(1.0)                           # median enqueue, ms
    # the ratio is the SMALLEST trial's: 0.0005 s over 0.00205 s / 10
    assert backlog == pytest.approx(0.0005 / (0.0205 / 10))
    assert f"smallest backlog {backlog:.2f}" in note


def test_host_bound_verdict_boundary_is_two_iterations_of_backlog():
    # wall 10 ms over 10 iters -> 1 ms per iteration; backlog of exactly 2 ms
    # is NOT host-bound, 1.99 ms is.
    deep = T.host_bound_verdict([T.TrialWall(enqueue_s=0.008, wall_s=0.010)], 10)
    drained = T.host_bound_verdict([T.TrialWall(enqueue_s=0.00801, wall_s=0.010)], 10)
    assert deep[0] is False
    assert drained[0] is True
    assert T.HOST_BOUND_BACKLOG_ITERS == 2


def test_host_bound_verdict_refuses_rather_than_guessing():
    assert T.host_bound_verdict([], 10) == (
        None, None, None, "no trials; host-bound not determinable")
    bound, ms, backlog, note = T.host_bound_verdict([T.TrialWall(0.0, 0.0)], 10)
    assert bound is None and ms is None and backlog is None and "no wall time" in note
    bound, ms, backlog, note = T.host_bound_verdict([T.TrialWall(0.001, 0.010)], 0)
    assert bound is None and backlog is None


# --- (4) LEVEL: the case the old flag missed -----------------------------------

def test_level_fails_when_the_clock_sits_low_at_both_ends_with_zero_drift():
    trace = [1500, 1500, 1500, 1500]
    res = run(FakeGPU(), trace=trace, reference_clock_mhz=1980.0)
    assert res.sm_clock_load_mhz == 1500.0
    assert res.sm_clock_start_mhz == 1500.0 and res.sm_clock_end_mhz == 1500.0
    assert res.clock_drift_ok is True
    assert res.clock_level_ok is False
    # The legacy flag, on the same clocks, passes it. That is the defect.
    _, throttled = T.clock_drift(T.ClockState(1500, 50), T.ClockState(1500, 50))
    assert throttled is False


def test_level_passes_at_the_reference_and_just_inside_the_fraction():
    ok = run(FakeGPU(), trace=[1980] * 3, reference_clock_mhz=1980.0)
    assert ok.clock_level_ok is True
    edge = run(FakeGPU(), trace=[1881] * 3, reference_clock_mhz=1980.0)
    assert edge.clock_level_ok is True          # 0.95 * 1980 = 1881
    below = run(FakeGPU(), trace=[1880] * 3, reference_clock_mhz=1980.0)
    assert below.clock_level_ok is False


def test_level_is_none_without_a_reference_and_says_so():
    res = run(FakeGPU(), trace=[1500] * 3)
    assert res.clock_level_ok is None
    assert res.clock_drift_ok is True
    assert "no reference clock" in res.clock_note


# --- (5) DRIFT: drop fails, small rise passes, large rise fails too ------------

def test_drift_fails_on_a_drop_beyond_the_fraction():
    res = run(FakeGPU(), trace=[1980, 1900, 1800, 1700], reference_clock_mhz=1980.0)
    assert res.sm_clock_start_mhz == 1980.0 and res.sm_clock_end_mhz == 1700.0
    assert res.clock_drift_ok is False
    # median (1900+1800)/2 = 1850 < 0.95 * 1980, so the level fails too: a
    # cell that throttled mid-trial is not at the roof's clock either.
    assert res.sm_clock_load_mhz == 1850.0
    assert res.clock_level_ok is False


def test_drift_passes_on_a_rise_within_the_fraction():
    res = run(FakeGPU(), trace=[1900, 1940, 1980], reference_clock_mhz=1980.0)
    assert res.clock_drift_ok is True             # +4.2%
    assert res.clock_level_ok is True


def test_drift_fails_on_a_rise_beyond_the_fraction():
    # A rise is a warmup that did not reach the operating point. The old flag
    # ignored rises entirely; the audit counted 80 kept rows that rose >5%.
    res = run(FakeGPU(), trace=[1500, 1700, 1980], reference_clock_mhz=1980.0)
    assert res.clock_drift_ok is False


def test_clock_flags_is_pure_and_refuses_half_a_comparison():
    assert T.clock_flags(1500.0, 1500.0, 1500.0, 1980.0) == (False, True)
    assert T.clock_flags(1980.0, 1980.0, 1700.0, 1980.0) == (True, False)
    assert T.clock_flags(1980.0, 1980.0, 1980.0, None) == (None, True)
    assert T.clock_flags(None, None, None, 1980.0) == (None, None)
    assert T.clock_flags(1980.0, 0.0, 1980.0, 1980.0) == (True, None)


def test_no_usable_clock_sample_gives_none_with_a_reason_not_zero():
    res = run(FakeGPU(), trace=[0, 0, 0], reference_clock_mhz=1980.0)
    assert res.sm_clock_load_mhz is None
    assert res.sm_clock_start_mhz is None and res.sm_clock_end_mhz is None
    assert res.clock_level_ok is None and res.clock_drift_ok is None
    assert res.clock_samples == 0
    assert "3 polled" in res.clock_note
    empty = run(FakeGPU(), trace=[], reference_clock_mhz=1980.0)
    assert empty.clock_level_ok is None and "0 polled" in empty.clock_note


def test_zero_samples_are_dropped_before_the_median():
    res = run(FakeGPU(), trace=[0, 1980, 0, 1980, 0, 1980], reference_clock_mhz=1980.0)
    assert res.sm_clock_load_mhz == 1980.0
    assert res.clock_samples == 3
    assert res.clock_level_ok is True


def test_the_median_floor_is_the_same_three_samples_the_reference_clock_needs():
    # One sample: neither verdict. Two: drift (a first and a last) but no
    # median, so no level. Three: both. calibrate.clock_under_load refuses
    # the reference below three for the same reason, so LEVEL compares two
    # numbers of one kind.
    assert T.CLOCK_SAMPLE_FLOOR == 3
    one = run(FakeGPU(), trace=[1980], reference_clock_mhz=1980.0)
    assert one.sm_clock_load_mhz is None and one.clock_level_ok is None
    assert one.clock_drift_ok is None and one.sm_clock_end_mhz is None
    assert "below the floor of 3" in one.clock_note and "drift needs two" in one.clock_note
    two = run(FakeGPU(), trace=[1980, 1700], reference_clock_mhz=1980.0)
    assert two.sm_clock_load_mhz is None and two.clock_level_ok is None
    assert two.clock_drift_ok is False
    assert "below the floor of 3" in two.clock_note
    three = run(FakeGPU(), trace=[1980, 1980, 1980], reference_clock_mhz=1980.0)
    assert three.clock_level_ok is True and three.clock_drift_ok is True
    assert "floor" not in three.clock_note


# --- (6) statistics match a hand computation ----------------------------------

def test_percentiles_match_a_hand_computation():
    # warm per call 1 ms, target 10 ms -> iters 10; one trial -> the ten
    # scripted durations 1..10 are the sample set.
    gpu = FakeGPU(warm_call_ms=1.0, timed_call_ms=[float(k) for k in range(1, 11)])
    res = run(gpu, warmup_ms=5.0, target_ms=10.0, trials=1)
    assert res.iters == 10 and res.samples == 10
    assert res.ms_p50 == 5.5
    assert res.ms_p90 == 9.0                      # sorted[round(0.9 * 9)] = sorted[8]
    assert res.ms_min == 1.0
    assert res.ms_std == pytest.approx(statistics.stdev(range(1, 11)))


def test_statistics_pool_across_trials():
    gpu = FakeGPU(warm_call_ms=1.0, timed_call_ms=[2.0] * 10 + [4.0] * 10)
    res = run(gpu, warmup_ms=5.0, target_ms=10.0, trials=2)
    assert res.samples == 20
    assert res.ms_p50 == 3.0 and res.ms_min == 2.0 and res.ms_p90 == 4.0


def test_kernel_timing_and_time_eager_share_one_percentile_definition():
    samples = [3.0, 1.0, 2.0, 5.0, 4.0]
    p50, p90, lo, std = T._stats(samples)
    legacy = T._summarise(samples, warmup=0, iters=5, trials=1, l2_flush=False,
                          cuda_graph=False)
    assert (legacy.ms_p50, legacy.ms_p90, legacy.ms_min, legacy.ms_std) == (p50, p90, lo, std)


# --- (7) refusal off-GPU, planted everywhere by taking CUDA away ------------------

@pytest.fixture
def no_cuda(monkeypatch):
    """Make the refuse branch run on the GPU box too, not only where CUDA is absent."""
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: False)


def test_refuses_off_gpu_without_fakes(no_cuda):
    with pytest.raises(T.TimingRefused, match="events, clock_sampler, flusher"):
        T.time_kernel(lambda: None, warmup_ms=10.0)


def test_refuses_off_gpu_when_only_some_fakes_are_injected(no_cuda):
    gpu = FakeGPU()
    with pytest.raises(T.TimingRefused, match="clock_sampler, flusher"):
        T.time_kernel(gpu.fn, warmup_ms=10.0, events=gpu.events)
    with pytest.raises(T.TimingRefused, match="flusher"):
        T.time_kernel(gpu.fn, warmup_ms=10.0, events=gpu.events,
                      clock_sampler=ScriptedClocks(gpu, [1980]))
    # with l2_flush off the flusher is not needed and the run proceeds
    res = T.time_kernel(gpu.fn, warmup_ms=10.0, target_ms=10.0, l2_flush=False,
                        events=gpu.events, clock_sampler=ScriptedClocks(gpu, [1980]))
    assert res.samples > 0


def test_refuses_zero_trials():
    with pytest.raises(T.TimingRefused, match="trials=0"):
        run(FakeGPU(), trials=0)


# --- (8) every record names its instrument and its clock source --------------------

def test_every_record_carries_the_instrument_name():
    for trace in ([1980] * 3, [1500] * 3, [0, 0]):
        res = run(FakeGPU(), trace=trace, reference_clock_mhz=1980.0)
        assert res.instrument == T.TIMING_BASIS
        assert res.clock_source == "scripted"
    assert T.TIMING_BASIS == "queue-deep/l2-flush/clock-under-load/v3"
    assert T.KernelTiming.__dataclass_params__.frozen


def test_sampler_note_and_poll_cost_reach_the_record():
    gpu = FakeGPU()
    sampler = ScriptedClocks(gpu, [1980] * 3, note="reader said so",
                             poll_cost_ms=0.03)
    res = T.time_kernel(gpu.fn, warmup_ms=10.0, target_ms=10.0, trials=1,
                        events=gpu.events, flusher=gpu, clock_sampler=sampler,
                        reference_clock_mhz=1980.0)
    assert res.clock_note.startswith("reader said so")
    assert res.clock_poll_ms == 0.03


# --- the clock reader: NVML only, bound to the caller's device -------------------

def test_nvml_clock_reader_refuses_without_cuda(no_cuda):
    with pytest.raises(T.ClockSourceUnavailable, match="no CUDA device"):
        T.nvml_clock_reader()


def test_nvml_clock_reader_names_a_missing_pynvml_and_never_forks(monkeypatch):
    # The base pod venv has torch but no nvidia-ml-py: clock_rate raises
    # ModuleNotFoundError. The reader must surface that, with the remedy,
    # and must not reach for nvidia-smi.
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T.torch.cuda, "current_device", lambda: 0)

    def no_pynvml(device=None):
        raise ModuleNotFoundError("No module named 'pynvml'")
    monkeypatch.setattr(T.torch.cuda, "clock_rate", no_pynvml)
    forks = []
    monkeypatch.setattr(T, "_nvidia_smi", lambda q: forks.append(q) or [])
    with pytest.raises(T.ClockSourceUnavailable, match="pynvml.*nvidia-ml-py"):
        T.nvml_clock_reader(0)
    assert forks == []
    # and the sampler, resolving its source at entry, records the reason
    # instead of raising, starts no thread, and collects nothing
    with T.BackgroundClockSampler(device_index=0, poll_seconds=0.001) as s:
        time.sleep(0.01)
    assert s.source == "none" and s.samples == ()
    assert "nvidia-ml-py" in s.note and s.poll_cost_ms is None
    assert forks == []


def test_nvml_clock_reader_reads_the_device_it_was_given(monkeypatch):
    # torch's current device is per host thread; the poll thread would start
    # on device 0. The reader carries the index it was constructed with.
    seen = []
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(T.torch.cuda, "clock_rate", lambda d: seen.append(("clk", d)) or 1755)
    monkeypatch.setattr(T.torch.cuda, "temperature", lambda d: seen.append(("tmp", d)) or 61)
    read = T.nvml_clock_reader(3)
    assert read() == T.ClockState(1755, 61)
    assert all(d == 3 for _, d in seen)
    with T.BackgroundClockSampler(device_index=3, poll_seconds=0.002) as s:
        time.sleep(0.03)
    assert s.source == "nvml"
    assert len(s.samples) >= 1 and s.samples[0] == T.ClockState(1755, 61)
    assert s.note == "" and s.poll_cost_ms is not None and s.poll_cost_ms < 1.0
    # a bare reader with no index takes the CALLING thread's device
    seen.clear()
    T.nvml_clock_reader()()
    assert seen and all(d == 0 for _, d in seen)


# --- the background sampler, with a fake clock and a real thread -----------------

def test_background_sampler_waits_before_its_first_sample_and_collects_under_load():
    trace = iter([1980, 1900, 1850, 1800, 1800, 1800, 1800, 1800])
    sampler = T.BackgroundClockSampler(
        sample=lambda: T.ClockState(next(trace, 1800), 50), poll_seconds=0.005)
    with sampler as s:
        assert s is sampler
        assert s.samples == ()
        time.sleep(0.05)
    # The count over a 50 ms window is scheduler-dependent; the first sample
    # and the short-region property below are not.
    assert 1 <= len(sampler.samples) <= 12
    assert sampler.samples[0].sm_clock_mhz == 1980
    assert sampler.source == "injected"
    # and a region too short for one poll yields NO sample rather than an idle one
    quick = T.BackgroundClockSampler(sample=lambda: T.ClockState(1980, 50),
                                     poll_seconds=1.0)
    with quick:
        pass
    assert quick.samples == () and quick.note == ""


def test_background_sampler_reports_a_reader_that_raises_instead_of_dying_quietly():
    # Used to reach threading.excepthook; the record then said "one usable
    # sample, drift not determinable", which misstates the cause.
    def flaky_reader():
        reads = iter([T.ClockState(1980, 50)])

        def flaky():
            try:
                return next(reads)
            except StopIteration:
                raise OSError("NVML lost the device") from None
        return flaky

    sampler = T.BackgroundClockSampler(sample=flaky_reader(), poll_seconds=0.002)
    with sampler:
        time.sleep(0.05)
    assert sampler.samples == (T.ClockState(1980, 50),)
    assert "raised after 1 samples" in sampler.note
    assert "OSError: NVML lost the device" in sampler.note
    assert "polling stopped" in sampler.note
    # and time_kernel puts it in front of the floor note. The fake trial must
    # outlast two polls for the reader to be asked twice, hence the backlog.
    gpu = FakeGPU(sync_sleep_s=0.05)
    res = T.time_kernel(gpu.fn, warmup_ms=10.0, target_ms=10.0, trials=1,
                        events=gpu.events, flusher=gpu,
                        clock_sampler=T.BackgroundClockSampler(
                            sample=flaky_reader(), poll_seconds=0.002),
                        reference_clock_mhz=1980.0)
    assert res.clock_note.startswith("clock sampler raised after 1 samples")
    assert "below the floor of 3" in res.clock_note
    assert res.clock_samples == 1 and res.clock_source == "injected"


def test_background_sampler_notes_a_reader_slower_than_its_budget():
    # A 20 ms read against a 5 ms poll: the poller is awake far more than the
    # 10% budget allows, competing with the enqueue thread. The cost is what
    # the reader took, so the assertion is deterministic.
    def slow():
        time.sleep(0.02)
        return T.ClockState(1980, 50)
    sampler = T.BackgroundClockSampler(sample=slow, poll_seconds=0.005)
    with sampler:
        time.sleep(0.06)
    assert sampler.poll_cost_ms is not None and sampler.poll_cost_ms >= 20.0
    assert "clock poll cost" in sampler.note and "competed with the enqueue thread" in sampler.note
    assert f"over {T.CLOCK_POLL_BUDGET_FRACTION:.0%}" in sampler.note
    # a fast reader gets no such note
    fast = T.BackgroundClockSampler(sample=lambda: T.ClockState(1980, 50),
                                    poll_seconds=0.005)
    with fast:
        time.sleep(0.03)
    assert fast.poll_cost_ms is not None and fast.poll_cost_ms < 0.5
    assert fast.note == ""


# --- (4b) LEVEL is two-sided: the boosted cell the old flag passed ----------------

def test_level_fails_HIGH_on_a_boosted_clock_and_names_the_side():
    """THE MIRROR IMAGE OF THE THROTTLE, and the side the flag could not see.
    A memory-shaped H200 decode cell runs at 1980 MHz; the compute roof was
    measured power-limited at ~1515. Against the one-sided `load >= 0.95 *
    ref` that cell PASSED, with its fixed-roof fraction inflated by 1980/1515
    = 1.31x, toward the study's claim."""
    res = run(FakeGPU(), trace=[1980] * 3, reference_clock_mhz=1515.0)
    assert res.sm_clock_load_mhz == 1980.0
    assert res.clock_level_ok is False
    assert res.clock_level_side == T.LEVEL_HIGH
    assert res.reference_clock_mhz == 1515.0
    assert "LEVEL failed HIGH" in res.clock_note
    assert "1.31x" in res.clock_note and "roof_at_cell_clock_tflops" in res.clock_note
    # the old rule, restated, would have passed it
    assert 1980.0 >= T.LEVEL_FRACTION * 1515.0


def test_level_fails_LOW_on_a_throttled_clock_and_names_the_side():
    res = run(FakeGPU(), trace=[1400] * 3, reference_clock_mhz=1515.0)
    assert res.clock_level_ok is False
    assert res.clock_level_side == T.LEVEL_LOW
    assert "LEVEL failed LOW" in res.clock_note


def test_level_passes_inside_the_band_with_no_side():
    res = run(FakeGPU(), trace=[1500] * 3, reference_clock_mhz=1515.0)
    assert res.clock_level_ok is True
    assert res.clock_level_side == ""
    assert "LEVEL failed" not in res.clock_note


def test_level_side_is_the_one_definition_of_the_band():
    """`clock_flags` derives LEVEL from `level_side`, so the two edges live in
    one place. Both edges, both sides, and the None that is not a side."""
    ref = 1515.0
    assert T.level_side(ref * T.LEVEL_FRACTION, ref) == ""            # low edge, inside
    assert T.level_side(ref * T.LEVEL_FRACTION - 1, ref) == T.LEVEL_LOW
    assert T.level_side(ref * T.LEVEL_HIGH_FRACTION, ref) == ""       # high edge, inside
    assert T.level_side(ref * T.LEVEL_HIGH_FRACTION + 1, ref) == T.LEVEL_HIGH
    assert T.level_side(None, ref) is None
    assert T.level_side(1500.0, None) is None
    assert T.level_side(1500.0, 0.0) is None
    for load in (1400.0, 1500.0, 1980.0):
        level, _ = T.clock_flags(load, load, load, ref)
        assert level is (T.level_side(load, ref) == "")
    assert T.LEVEL_HIGH_FRACTION == 1.05 and T.LEVEL_FRACTION == 0.95


# --- (3c) the backlog ratio reaches the record ----------------------------------

def test_the_backlog_ratio_is_on_the_record_beside_the_verdict():
    gpu = FakeGPU(warm_call_ms=10.0, timed_call_ms=0.5, host_call_s=0.0,
                  sync_sleep_s=0.05)
    res = run(gpu, warmup_ms=10.0, target_ms=5.0, trials=1)
    assert res.host_bound is False
    # 50 ms of backlog over a ~50 ms trial of 10 iterations: about 10
    assert res.host_backlog_iters is not None
    assert res.host_backlog_iters > T.HOST_BOUND_BACKLOG_ITERS
    drained = run(FakeGPU(warm_call_ms=10.0, timed_call_ms=0.5, host_call_s=0.002),
                  warmup_ms=10.0, target_ms=5.0, trials=1)
    assert drained.host_bound is True
    assert drained.host_backlog_iters is not None
    assert drained.host_backlog_iters < T.HOST_BOUND_BACKLOG_ITERS


# --- (2b) the warmup runs the loop the trials run ---------------------------------

def test_the_warmup_is_flushed_when_the_trials_are():
    """Until v3 the warmup ran unflushed while the trials ran flushed, so the
    governor reached the operating point of a different workload. Now every
    warmup call, batch and probe alike, is preceded by a flush when the trials
    will be, and by nothing when they will not."""
    gpu = FakeGPU()
    res = run(gpu, warmup_ms=10.0, target_ms=10.0, trials=1)
    log = gpu.log
    on = log.index(("clock_on",))
    warm_fns = [k for k, e in enumerate(log[:on]) if e == ("fn",)]
    assert warm_fns, "no warmup calls were logged"
    for k in warm_fns:
        # a batch call is flush, fn; a probe call is flush, start, fn
        assert log[k - 1] == ("flush",) or (
            log[k - 1][0] == "start" and log[k - 2] == ("flush",)), log[k - 2:k + 1]
    assert res.warmup_calls == len(warm_fns)
    bare = FakeGPU()
    T.time_kernel(bare.fn, warmup_ms=10.0, target_ms=10.0, trials=1,
                  l2_flush=False, events=bare.events,
                  clock_sampler=ScriptedClocks(bare, [1980] * 3))
    assert bare.flushes == 0


def test_iters_are_sized_from_the_flushed_probe_not_the_batch_mean():
    """A kernel that reads 1 ms warm in L2 (the batch mean) and 0.5 ms in the
    probe's cold-L2 per-iteration intervals must be sized from the probe: 40
    iterations for a 20 ms target, not 20. The fake scripts the batch calls at
    1 ms and the probe's 25 calls at 0.5 ms."""
    warm = [1.0] * 51 + [0.5] * 25
    gpu = FakeGPU(warm_call_ms=warm, timed_call_ms=0.5)
    res = run(gpu, warmup_ms=50.0, target_ms=20.0, trials=1)
    assert res.warmup_calls == 76
    assert gpu.probe.n == 25
    assert res.iters == 40, "sized from the batch mean of 1 ms, not the probe"
    assert gpu.timed.n == 40


# --- (8b) the basis is bound to the loop's shape ------------------------------------

def test_timing_basis_is_bound_to_the_loop_shape():
    """A change to the flush position, the iteration cap, the warmup batch
    length, whether the warmup is flushed, the probe, or the flush buffer's
    L2 multiple moves numbers, and until 2026-09-03 nothing but convention
    tied the basis string to any of them. The live values are derived from
    the code, not read off `LOOP_SHAPE`, so editing the dict alone cannot
    satisfy this; and the string is pinned beside them, so bumping one
    without the other fails here rather than in a published arm."""
    import inspect

    sig = inspect.signature(T.iters_for)
    live = {
        "flush_position": "before start record",
        "iters_hi": sig.parameters["hi"].default,
        "iters_lo": sig.parameters["lo"].default,
        "warmup_batch_ms": T.WARMUP_BATCH_MS,
        "warmup_flushed": "flush" in inspect.signature(T.warm_until).parameters,
        "iters_sized_from": "flushed per-iteration probe",
        "flush_l2_multiple": (inspect.signature(T.flush_mb_for_device)
                              .parameters["multiple"].default),
    }
    src = inspect.getsource(T._timed_trials)
    body = src[src.index("for i in range(iters):"):]
    assert body.index("flush()") < body.index("starts[i].record()"), (
        "the flush must be enqueued before the start record")
    warm_src = inspect.getsource(T.warm_until)
    assert "_timed_trials(fn, n, 1, events(n), flush)" in warm_src, (
        "iters are sized from a flushed per-iteration probe")
    assert live == T.LOOP_SHAPE
    assert T.TIMING_BASIS == "queue-deep/l2-flush/clock-under-load/v3", (
        "the loop shape changed: bump TIMING_BASIS and this pin together")


# --- (8c) the retired timers name themselves, and not as the instrument -----------

def test_the_retired_timer_record_names_its_apparatus():
    legacy = T._summarise([1.0, 2.0, 3.0], warmup=5, iters=3, trials=1,
                          l2_flush=False, cuda_graph=False)
    assert legacy.instrument == T.RETIRED_TIMER_BASIS
    assert legacy.instrument != T.TIMING_BASIS
    assert "time_eager" in T.RETIRED_TIMER_BASIS


# --- the clock sample says which reader answered ----------------------------------

def test_clock_state_sample_names_its_source(monkeypatch):
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T.torch.cuda, "clock_rate", lambda *a: 1515)
    monkeypatch.setattr(T.torch.cuda, "temperature", lambda *a: 60)
    s = T.ClockState.sample()
    assert (s.sm_clock_mhz, s.source) == (1515, T.CLOCK_SOURCE_NVML)

    def no_pynvml(*a):
        raise ModuleNotFoundError("No module named 'pynvml'")
    monkeypatch.setattr(T.torch.cuda, "clock_rate", no_pynvml)
    monkeypatch.setattr(T, "_nvidia_smi", lambda q: ["1980, 44"])
    forked = T.ClockState.sample()
    assert (forked.sm_clock_mhz, forked.source) == (1980, T.CLOCK_SOURCE_NVIDIA_SMI)
    monkeypatch.setattr(T, "_nvidia_smi", lambda q: [])
    none = T.ClockState.sample()
    assert (none.sm_clock_mhz, none.source) == (0, T.CLOCK_SOURCE_NONE)
    # positional construction keeps meaning what it meant
    assert T.ClockState(1755, 61).source == T.CLOCK_SOURCE_NVML


# --- the audit's acceptance case, on the box only ---------------------------------

@pytest.mark.gpu
def test_time_kernel_matches_time_eager_on_a_short_kernel_within_two_percent():
    # 8192x8192 bf16: 128 MiB read, 128 MiB written, ~55 us on an H200. Large
    # enough that launch latency is not the number, and above the regime
    # where L2Flusher's docstring records the flush itself moving a kernel.
    a = torch.randn((8192, 8192), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)

    def k():
        torch.mul(a, 1.0001, out=out)

    eager = T.time_eager(k, warmup=200, iters=None, trials=3, l2_flush=True,
                         flush_mb=T.flush_mb_for_device())
    ours = T.time_kernel(k, warmup_ms=500.0, trials=3, l2_flush=True)
    assert ours.instrument == T.TIMING_BASIS
    assert ours.host_bound is False, ours.host_note
    if ours.clock_source == "nvml":
        assert ours.sm_clock_load_mhz is not None and ours.clock_samples >= 3
    else:
        assert ours.clock_source == "none" and "nvidia-ml-py" in ours.clock_note
    assert abs(ours.ms_p50 - eager.ms_p50) / eager.ms_p50 <= 0.02


# --- the header describes the tree it is in ---------------------------------

def test_the_module_docstring_no_longer_defers_the_ladder_migration():
    """Inverse hit, closed: the header said the ladder scripts "still carry
    their private `time_call`" and that moving them was "the next phase",
    after every arm script had moved. Asserted against the tree, not against
    the prose alone: the scripts that call `time_kernel` are counted."""
    import pathlib

    doc = T.__doc__ or ""
    assert "still carry their private" not in doc
    assert "next phase" not in doc
    assert "moved onto it" in doc
    scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    on_instrument = [p.name for p in scripts.glob("*.py")
                     if "time_kernel(" in p.read_text()]
    assert len(on_instrument) >= 10, on_instrument
    private = [p.name for p in scripts.glob("*.py")
               if "def time_call" in p.read_text()]
    assert private == ["block_m_crossing_sweep.py"], private



def test_the_clock_poll_is_sized_so_the_floor_of_samples_lands_inside_the_region():
    """2026-09-09: alias_ablation times 50 ms x 3 trials, ~150 ms of region;
    a fixed 50 ms poll landed two or three samples against a floor of three
    and LEVEL was undetermined on a sound rung. The poll aims for twice the
    floor, capped at CLOCK_POLL_SECONDS on long regions and floored at
    CLOCK_POLL_FLOOR_SECONDS on tiny ones."""
    alias = T.clock_poll_for(trials=3, iters=50, per_call_ms=1.0)   # 150 ms
    assert alias == pytest.approx(0.025)
    assert 0.150 / alias >= 2 * T.CLOCK_SAMPLE_FLOOR
    assert T.clock_poll_for(trials=3, iters=1000, per_call_ms=1.0) == T.CLOCK_POLL_SECONDS
    assert T.clock_poll_for(trials=1, iters=1, per_call_ms=0.01) == T.CLOCK_POLL_FLOOR_SECONDS
    for region_ms in (60, 150, 400, 3000):
        poll = T.clock_poll_for(trials=1, iters=1, per_call_ms=float(region_ms))
        assert region_ms / 1e3 / poll >= 2 * T.CLOCK_SAMPLE_FLOOR
