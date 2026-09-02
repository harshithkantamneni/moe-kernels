"""The one timing instrument, exercised off-GPU against fakes.

`time_kernel` is the apparatus every published number is supposed to come
from, and its two failure modes are silent: an event structure that lets host
time into the interval, and a clock flag that passes a card running at 1500 MHz
against a roof measured at 1980. Neither leaves a mark on the number. So every
branch here is planted with a fake GPU whose per-call durations and clock trace
are scripted, and the assertions are on what the instrument DID (order of
records, count of synchronises, calls made during warmup), not only on what it
returned.

The one GPU test at the bottom is the audit's acceptance case: a ~50 us kernel
measured by `time_kernel` agrees with `time_eager` within 2%, because both run
the same `_timed_trials` loop.
"""
import itertools
import statistics

import pytest
import torch

from moe.bench import timing as T

NO_CUDA = not torch.cuda.is_available()


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

    def elapsed(self, n: int) -> list[float]:
        return list(self.ms[:n])


class FakeGPU:
    """A GPU whose kernel takes a scripted time per call.

    `fn` is the callable under test; each call consumes the next duration
    from `warm_call_ms` while a one-pair (warmup) interval is open, or from
    `timed_call_ms` while a multi-pair (trial) interval is open, and adds it
    to the open interval. `events` is the factory, `flush` the flusher.
    """

    def __init__(self, warm_call_ms=1.0, timed_call_ms=1.0):
        self._warm = self._iter(warm_call_ms)
        self._timed = self._iter(timed_call_ms)
        self.log: list = []
        self.calls = 0
        self.flushes = 0
        self.megabytes = 240
        self._open = None
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
        pairs, i = self._open if self._open else (None, None)
        if pairs is None:
            return
        pairs.ms[i] += next(self._warm if pairs.n == 1 else self._timed)

    def flush(self) -> None:
        self.flushes += 1
        self.log.append(("flush",))

    @property
    def timed(self) -> _FakePairs:
        multi = [p for p in self.instances if p.n > 1]
        assert len(multi) == 1, "expected exactly one multi-pair (trial) instance"
        return multi[0]


class ScriptedClocks:
    """Stands in for `BackgroundClockSampler`: a scripted trace, in MHz."""

    def __init__(self, gpu: FakeGPU, trace):
        self.gpu, self.trace = gpu, list(trace)
        self.samples: tuple = ()

    def __enter__(self):
        self.gpu.log.append(("clock_on",))
        return self

    def __exit__(self, *exc):
        self.gpu.log.append(("clock_off",))
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
        assert res.warmup_ms < 50.0 + T.WARMUP_BATCH_MS     # overshoot bounded by one batch
    # 1 ms kernel: batches of 1, 25, 25 -> 51 calls. 10 ms kernel: 1, 2, 2 -> 5.
    assert fast.warmup_calls == 51
    assert slow.warmup_calls == 5
    assert fast.warmup_calls != slow.warmup_calls, "a count would be the same"


def test_warmup_batches_keep_the_queue_deep():
    gpu = FakeGPU(warm_call_ms=0.01)
    report = T.warm_until(gpu.fn, 30.0, gpu.events)
    # first batch is one call, the rest are sized to ~WARMUP_BATCH_MS
    assert report.batches == 3
    # 1 + 2 * 2500, give or take the float rounding of 25.0 / 0.01
    assert abs(report.calls - (1 + 2 * T.WARMUP_BATCH_MS / 0.01)) <= 2
    assert report.per_call_ms == pytest.approx(0.01)


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
    assert gpu.flushes == n * 2
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
    ok = run(FakeGPU(), trace=[1980, 1980], reference_clock_mhz=1980.0)
    assert ok.clock_level_ok is True
    edge = run(FakeGPU(), trace=[1881, 1881], reference_clock_mhz=1980.0)
    assert edge.clock_level_ok is True          # 0.95 * 1980 = 1881
    below = run(FakeGPU(), trace=[1880, 1880], reference_clock_mhz=1980.0)
    assert below.clock_level_ok is False


def test_level_is_none_without_a_reference_and_says_so():
    res = run(FakeGPU(), trace=[1500, 1500])
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
    res = run(FakeGPU(), trace=[0, 1980, 0, 1980, 0], reference_clock_mhz=1980.0)
    assert res.sm_clock_load_mhz == 1980.0
    assert res.clock_samples == 2
    assert res.clock_level_ok is True


def test_one_sample_gives_a_level_but_no_drift():
    res = run(FakeGPU(), trace=[1980], reference_clock_mhz=1980.0)
    assert res.clock_level_ok is True
    assert res.clock_drift_ok is None
    assert res.sm_clock_end_mhz is None
    assert "one usable clock sample" in res.clock_note


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


# --- (7) refusal off-GPU ----------------------------------------------------------

@pytest.mark.skipif(not NO_CUDA, reason="refusal is the off-GPU branch")
def test_refuses_off_gpu_without_fakes():
    with pytest.raises(T.TimingRefused, match="events, clock_sampler, flusher"):
        T.time_kernel(lambda: None, warmup_ms=10.0)


@pytest.mark.skipif(not NO_CUDA, reason="refusal is the off-GPU branch")
def test_refuses_off_gpu_when_only_some_fakes_are_injected():
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


# --- (8) every record names its instrument ---------------------------------------

def test_every_record_carries_the_instrument_name():
    for trace in ([1980, 1980], [1500, 1500], [0, 0]):
        res = run(FakeGPU(), trace=trace, reference_clock_mhz=1980.0)
        assert res.instrument == T.TIMING_BASIS
    assert T.TIMING_BASIS == "queue-deep/l2-flush/clock-under-load/v2"
    assert T.KernelTiming.__dataclass_params__.frozen


# --- the background sampler, with a fake clock and a real thread -----------------

def test_background_sampler_waits_before_its_first_sample_and_collects_under_load():
    import time

    trace = iter([1980, 1900, 1850, 1800, 1800, 1800, 1800, 1800])
    sampler = T.BackgroundClockSampler(
        sample=lambda: T.ClockState(next(trace, 1800), 50), poll_seconds=0.005)
    with sampler as s:
        assert s is sampler
        assert s.samples == ()
        time.sleep(0.05)
    assert 3 <= len(sampler.samples) <= 12
    assert sampler.samples[0].sm_clock_mhz == 1980
    # and a region too short for one poll yields NO sample rather than an idle one
    quick = T.BackgroundClockSampler(sample=lambda: T.ClockState(1980, 50),
                                     poll_seconds=1.0)
    with quick:
        pass
    assert quick.samples == ()


# --- the audit's acceptance case, on the box only ---------------------------------

@pytest.mark.gpu
def test_time_kernel_matches_time_eager_on_a_short_kernel_within_two_percent():
    a = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)

    def k():
        torch.mul(a, 1.0001, out=out)

    eager = T.time_eager(k, warmup=200, iters=None, trials=3, l2_flush=True,
                         flush_mb=T.flush_mb_for_device())
    ours = T.time_kernel(k, warmup_ms=500.0, trials=3, l2_flush=True)
    assert ours.instrument == T.TIMING_BASIS
    assert ours.sm_clock_load_mhz is not None and ours.clock_samples >= 3
    assert abs(ours.ms_p50 - eager.ms_p50) / eager.ms_p50 <= 0.02
