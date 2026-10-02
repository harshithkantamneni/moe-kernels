"""Rental 2's scaffolding off the GPU (docs/registered/2026-10-01-rental2-*): every new
flag, plan key and refusal of private_weight_reference.py, dram_counter_route.py and
launch_floor.py. No kernel is written anywhere: BLOCK_SIZE_K and num_stages are config
values for the existing fused_moe kernel, and the null kernel is torch.cuda._sleep."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import dram_counter_route as DCR  # noqa: E402
import launch_floor as LF  # noqa: E402
import private_weight_reference as PW  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

TP2 = MODEL_CONFIGS["mixtral-8x7b-tp2"]


# --------------------------------------------------------------------------
# private_weight_reference: config values, slot padding, the null kernel
# --------------------------------------------------------------------------

def test_pinned_config_takes_block_k_and_keeps_its_default():
    assert PW.pinned_config(64, 1, 4) == dict(PW.SWEEP.FIXED, GROUP_SIZE_M=1)
    assert PW.pinned_config(64, 1, 4, block_k=32)["BLOCK_SIZE_K"] == 32
    assert PW.block_k_kw(64) == {} and PW.block_k_kw(None) == {}
    assert PW.block_k_kw(128) == {"block_k": 128}


@pytest.mark.parametrize("bk, st", [(64, 4), (32, 4), (128, 2), (64, 8), (64, 6)])
def test_every_config_the_plan_asks_is_accepted(bk, st):
    assert PW.config_refusal(TP2, block_m=32, block_n=64, block_k=bk, num_stages=st) == ""


@pytest.mark.parametrize("bk, st, cfg, why", [
    (48, 4, TP2, "power of two"),
    (8, 4, TP2, "power of two"),
    (64, 0, TP2, "at least 1 stage"),
    (256, 8, TP2, "OutOfResources"),
    (64, 4, SimpleNamespace(hidden_size=4096, intermediate_size=1800), "not a multiple"),
])
def test_what_triton_or_the_kstep_count_would_not_take_is_refused(bk, st, cfg, why):
    assert why in PW.config_refusal(cfg, block_m=32, block_n=64, block_k=bk, num_stages=st)


def test_pipeline_smem_is_the_stage_buffers():
    assert PW.pipeline_smem_bytes(32, 64, 64, 4) == 4 * (32 * 64 + 64 * 64) * 2


TINY = SimpleNamespace(num_experts=2, hidden_size=8, intermediate_size=4, top_k=2)


def test_slot_padding_moves_stride_and_nothing_else():
    w1, w2, _ = PW.build_private_weights(TINY, "bf16", 3, 0, device="cpu", pad_rows=7)
    v1, v2, _ = PW.build_private_weights(TINY, "bf16", 3, 0, device="cpu")
    assert w1.shape == v1.shape == (6, 8, 8) and w2.shape == v2.shape == (6, 8, 4)
    assert w1.stride(0) == (8 + 7) * 8 and w2.stride(0) == (8 + 7) * 4
    assert v1.stride(0) == 8 * 8 and w1.stride(-1) == 1
    assert bool((w1 == v1).all()) and bool((w2 == v2).all())
    span = w1.shape[0] * (w1.stride(0) + w2.stride(0)) * w1.element_size()
    assert span == PW.padded_weight_bytes(TINY, "bf16", 3, 7)
    assert PW.padded_weight_bytes(TP2, "bf16", 3, 0) == PW.weight_bytes_total(TP2, "bf16", 3)
    with pytest.raises(PW.PrivateWeightRefusal):
        PW.build_private_weights(TINY, "bf16", 3, 0, device="cpu", pad_rows=-1)


def test_memory_plan_prices_the_padded_allocation():
    a = PW.memory_plan(TP2, "bf16", 2, 9, 1024, None, "x", flush=(0, ""))
    b = PW.memory_plan(TP2, "bf16", 2, 9, 1024, None, "x", flush=(0, ""), pad_rows=7)
    assert b.weight_bytes == PW.padded_weight_bytes(TP2, "bf16", 9, 7) > a.weight_bytes


def _plan(**over):
    treads = list(over.pop("treads", [1, 2, 3, 4, 6]))
    arms = list(over.pop("arms", PW.ARMS))
    cfg = MODEL_CONFIGS[over.get("model", "mixtral-8x7b-tp2")]
    copies, reason = PW.counter_declaration(cfg, 32)
    plan = {"family": "r3-arms", "kind": "measure", "model": "mixtral-8x7b-tp2", "dtype": "bf16",
            "block_m": 32, "block_n": 64, "num_stages": 4, "group_m": 4, "treads": treads,
            "arms": arms, "cells": [[a, n] for a in arms for n in treads],
            "copies_declared": copies, "declared_reason": reason, "calls_per_cell": 2,
            "warmup_calls": 2, "gemms_per_call": PW.GEMMS_PER_CALL, "seed": 0,
            "manifest": "unused", "triton_cache": "unused"}
    plan.update(over)
    return plan


@pytest.mark.parametrize("over, why", [
    ({"block_k": 48}, "power of two"),
    ({"num_stages": 0}, "at least 1 stage"),
    ({"slot_pad_rows": -1}, "slot_pad_rows"),
    ({"null_kernel": True}, "NATIVE-only floor plan"),
    ({"null_kernel": True, "arms": [PW.NATIVE], "kind": "census"}, "census"),
])
def test_validate_counter_plan_refuses_rental2s_bad_keys(over, why):
    with pytest.raises(PW.CounterPlanRefused, match=why):
        PW.validate_counter_plan(_plan(**over))


def test_the_null_kernel_rides_in_the_skip_and_the_count():
    base = _plan(arms=[PW.NATIVE], treads=[2, 4, 6])
    null = dict(base, null_kernel=True)
    PW.validate_counter_plan(null)
    a, b = PW.counter_schedule(base), PW.counter_schedule(null)
    cells = len(base["cells"])
    assert (a.launch_skip, a.launch_count) == (2 * 2 * cells, 2 * 2 * cells)
    assert (b.launch_skip, b.launch_count) == (3 * 2 * cells, 3 * 2 * cells)
    assert PW.launches_per_call(null) == 3 and PW.launches_per_call(base) == 2
    assert PW.NULL_KERNEL_NAME == DCR.NULL_KERNEL_NAME == "spin_kernel"


def test_the_counter_child_sleeps_after_every_call_and_records_it(monkeypatch):
    import torch
    plan = _plan(arms=[PW.NATIVE], treads=[2, 4], null_kernel=True, block_k=32)
    cfg = MODEL_CONFIGS[plan["model"]]
    events = []

    def tiny_inputs(cfg_, n, bm, c, seed, dtype, w_dtype, *, device, arms):
        tokens = PW.SWEEP.tokens_for_rows(cfg_, n * bm)
        ids = torch.arange(tokens * cfg_.top_k).reshape(tokens, cfg_.top_k) % cfg_.num_experts
        return tokens, torch.zeros(tokens, 2), {PW.NATIVE: ids}, None, {}
    monkeypatch.setattr(PW, "arm_inputs", tiny_inputs)
    monkeypatch.setattr(PW, "build_private_weights", lambda cfg_, dtype, c, seed, **k: (
        torch.zeros(cfg.num_experts * c, 2, 2), torch.zeros(cfg.num_experts * c, 2, 2), None))
    confs = []

    def fused(hidden_states, **k):
        events.append("call")
        return hidden_states

    @contextlib.contextmanager
    def override(conf):
        confs.append(conf)
        yield

    stack = PW.CounterStack(
        fused_experts=fused, override_config=override,
        align=lambda ids, bm, d, emap: (torch.empty(PW.predicted_sorted_ids(ids.numel(), d, bm)),),
        device="cpu", synchronize=lambda: None, nvtx_range=lambda name: contextlib.nullcontext(),
        device_free=lambda: (None, "planted"), versions={}, device_identity={"uuid": "planted"},
        null_kernel=lambda cycles: events.append(("sleep", cycles)))
    man = PW.counter_child(plan, stack)
    calls = 2 * (plan["warmup_calls"] + plan["calls_per_cell"])
    assert events == ["call", ("sleep", PW.NULL_KERNEL_CYCLES)] * calls
    assert man["null_kernel"] is True and man["null_kernel_name"] == "spin_kernel"
    assert all(c["BLOCK_SIZE_K"] == 32 for c in confs)
    no_sleep = dict(stack.__dict__, null_kernel=None)
    with pytest.raises(PW.CounterPlanRefused, match="_sleep"):
        PW.counter_child(plan, PW.CounterStack(**no_sleep))


def test_the_timed_cli_refuses_a_config_triton_would_not_take(capsys):
    for flags, why in ((["--block-k", "48"], "power of two"),
                       (["--num-stages", "0"], "at least 1 stage"),
                       (["--slot-pad-rows", "-1"], "slot")):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = PW.main(["--dry-run", "--device-memory-gb", "140", *flags])
        assert rc == exit_codes.REFUSED and why in out.getvalue(), (flags, out.getvalue()[-400:])


def test_the_timed_run_id_moves_only_off_the_defaults():
    p = PW.build_parser()
    base = PW.default_run_id(p.parse_args([]), "card")
    assert PW.default_run_id(p.parse_args(["--block-k", "64"]), "card") == base
    assert PW.default_run_id(p.parse_args(["--slot-pad-rows", "0"]), "card") == base
    assert PW.default_run_id(p.parse_args(["--block-k", "32"]), "card") != base
    assert PW.default_run_id(p.parse_args(["--slot-pad-rows", "7"]), "card") != base


# --------------------------------------------------------------------------
# dram_counter_route: flags, plans, attribution, run ids
# --------------------------------------------------------------------------

def _main(*a):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = DCR.main(list(a))
    return rc, out.getvalue()


@pytest.mark.parametrize("argv, why", [
    (["--dry-run", "--family", "r3-arms", "--floor", "--block-k", "32"], "belong to a page"),
    (["--dry-run", "--block-k", "32"], "belong to a page"),
    (["--dry-run", "--family", "r3-arms", "--partition-metrics"], "--partition-metrics belongs"),
    (["--dry-run", "--family", "r3-arms", "--floor-shape-metrics"], "belong to --floor"),
    (["--dry-run", "--family", "r3-arms", "--floor-null-kernel"], "belong to --floor"),
    (["--dry-run", "--family", "r3-arms", "--slot-pad-rows", "-1"], "shorter"),
])
def test_each_rental2_flag_belongs_where_it_says(argv, why):
    rc, out = _main(*argv)
    assert rc == exit_codes.REFUSED and why in out, out[-300:]


def test_the_r3_dry_run_prints_the_pages_config_values():
    rc, out = _main("--dry-run", "--family", "r3-arms", "--model", "mixtral-8x7b-tp2",
                    "--block-k", "128", "--num-stages", "2", "--slot-pad-rows", "7")
    assert rc == exit_codes.DONE, out[-500:]
    assert "BLOCK_SIZE_K=128" in out and "num_stages=2" in out and "slot padding    7 rows" in out
    rc, out = _main("--dry-run", "--family", "r3-arms", "--model", "mixtral-8x7b-tp2",
                    "--block-k", "48")
    assert rc == exit_codes.REFUSED and "power of two" in out


def _r3_plan(**kw):
    return DCR.r3_plan(model="mixtral-8x7b-tp2", dtype="bf16", block_m=32, block_n=64,
                       num_stages=kw.pop("num_stages", 4), group_m=4, treads=[1, 2, 3],
                       kind="measure", arms=kw.pop("arms", PW.ARMS), calls=3, warmups=2,
                       profile_dir=Path("p"), stem="g4", **kw)


def test_r3_plan_carries_rental2s_keys_only_when_set():
    plain = _r3_plan()
    assert not {"block_k", "slot_pad_rows", "null_kernel"} & set(plain)
    assert "block_k" not in _r3_plan(block_k=64)
    p = _r3_plan(block_k=32, slot_pad_rows=7)
    assert p["block_k"] == 32 and p["slot_pad_rows"] == 7
    d = DCR.r3_design(p)
    assert d["block_k"] == 32 and d["slot_pad_rows"] == 7
    assert DCR.r3_design(plain)["block_k"] == 64 and "slot_pad_rows" not in DCR.r3_design(plain)
    with pytest.raises(PW.CounterPlanRefused):
        _r3_plan(block_k=48)
    f = DCR.r3_floor_plan(64, profile_dir=Path("p"), model="mixtral-8x7b-tp4",
                          treads=[2, 4, 16], null_kernel=True)
    assert f["null_kernel"] is True and PW.counter_schedule(f).launch_count == 3 * 2 * 3


def _null_manifest():
    return {"gemms_per_call": 2, "order": [["native", 2]], "calls_per_cell": 2,
            "warmup_calls": 2, "launch_count": 6, "null_kernel": True,
            "null_kernel_cycles": 1000, "grids": {"native/2": {"w1": 10, "w2": 20}}}


def _ln(i, name, grid, **m):
    return DCR.Launch(str(i), name, {"launch__grid_size": grid, **m})


def test_the_null_kernel_is_attributed_at_exactly_its_slots():
    man = _null_manifest()
    seq = DCR.r3_launch_sequence(man)
    assert [g for _k, _c, g, _w in seq] == ["w1", "w2", "null"] * 2
    good = []
    for i, g in enumerate(["w1", "w2", "null"] * 2):
        good.append(_ln(i, "spin_kernel" if g == "null" else "fused_moe_kernel",
                        1 if g == "null" else man["grids"]["native/2"][g]))
    att = DCR.attribute_launches(good, man)
    assert [a["gemm"] for a in att] == ["w1", "w2", "null"] * 2
    order = [0, 2, 1, 3, 4, 5]
    swapped = [DCR.Launch(str(i), good[j].kernel, good[j].metrics) for i, j in enumerate(order)]
    with pytest.raises(DCR.CounterRunRefused):
        DCR.attribute_launches(swapped, man)
    wide = [_ln(i, ln.kernel, 2 if ln.kernel == "spin_kernel" else ln.metrics["launch__grid_size"])
            for i, ln in enumerate(good)]
    with pytest.raises(DCR.CounterRunRefused, match="one block"):
        DCR.attribute_launches(wide, man)
    with pytest.raises(DCR.CounterRunRefused, match="null kernel"):
        DCR.attribute_launches(good[:-1], man)


def test_floor_cells_keep_the_null_kernel_apart_and_shape_metrics_when_returned():
    man = _null_manifest()
    lns = []
    for i, g in enumerate(["w1", "w2", "null"] * 2):
        extra = {"sm__cycles_elapsed.avg": 2500.0 if g == "null" else 1e5,
                 "gpu__time_duration.sum": 1500.0 if g == "null" else 6e4}
        if g == "w1":
            extra["sm__cycles_active.max"] = 9e4
        lns.append(_ln(i, "spin_kernel" if g == "null" else "fused_moe_kernel",
                       1 if g == "null" else man["grids"]["native/2"][g], **extra))
    cells = DCR.r3_floor_cells(DCR.attribute_launches(lns, man), man)
    c = cells[0]
    assert set(c["per_gemm"]) == {"w1", "w2"}
    assert c["null"]["sm__cycles_elapsed.avg"] == 2500.0 and c["null"]["calls"] == 2
    assert c["per_gemm"]["w1"]["sm__cycles_active.max"] == 9e4
    assert "sm__cycles_active.max" not in c["per_gemm"]["w2"]


def test_the_null_kernel_filter_and_units():
    argv = DCR.r3_ncu_argv("ncu", Path("p.json"), Path("r"), ["m"], launch_skip=6, launch_count=6,
                           kernel_filter=DCR.R3_NULL_KERNEL_FILTER)
    assert argv[argv.index("-k") + 1] == "regex:^(fused_moe_kernel|spin_kernel)$"
    plain = DCR.r3_ncu_argv("ncu", Path("p.json"), Path("r"), ["m"], launch_skip=6, launch_count=6)
    assert plain[plain.index("-k") + 1] == DCR.R3_KERNEL_FILTER
    for m in DCR.R3_PARTITION_METRICS:
        assert DCR.unit_table(m)[0] == "sector"
    units = {m: DCR.unit_table(m)[0] for m in DCR.R3_FLOOR_SHAPE_METRICS}
    assert units["sm__ctas_launched.sum"] == "block" and units["sm__warps_launched.sum"] == "warp"
    assert units["gpc__cycles_elapsed.max"] == "cycle" and units["sm__ctas_active.sum"] == "block"


def test_rental2s_run_id_knobs_appear_only_when_set():
    def rid(*extra, mode="r3-run"):
        a = DCR.build_parser().parse_args(["--family", "r3-arms", "--run", *extra])
        DCR.resolve_r3_defaults(a, ["--family", "r3-arms", "--run", *extra])
        return DCR.run_id_for(mode, a, "card")
    base = rid()
    assert rid("--block-k", "64") == base and rid("--slot-pad-rows", "0") == base
    assert len({base, rid("--block-k", "32"), rid("--slot-pad-rows", "7"),
                rid("--partition-metrics")}) == 4
    fbase = rid("--floor", mode="r3-floor")
    assert rid("--floor", "--floor-shape-metrics", mode="r3-floor") != fbase
    assert rid("--floor", "--floor-null-kernel", mode="r3-floor") != fbase


def test_partition_metrics_are_recorded_on_a_page_cell():
    man = {"gemms_per_call": 2, "order": [["shared", 2]], "calls_per_cell": 2, "warmup_calls": 1,
           "launch_count": 4, "grids": {"shared/2": {"w1": 10, "w2": 20}},
           "tokens": {"2": 64}, "declared_by_arm": {"shared": 72}}
    lns = [_ln(i, "fused_moe_kernel", man["grids"]["shared/2"][g],
               **{"dram__bytes_read.sum": 1e6, DCR.R3_PARTITION_METRICS[1]: 123.0})
           for i, g in enumerate(["w1", "w2"] * 2)]
    cells = DCR.r3_reduce_cells(DCR.attribute_launches(lns, man), man, ["dram__bytes_read.sum"])
    assert cells[0]["recorded"]["w1"][DCR.R3_PARTITION_METRICS[1]] == 123.0


# --------------------------------------------------------------------------
# launch_floor: the two processes, the profiler guard, the host probe
# --------------------------------------------------------------------------

BASE = ["--model", "mixtral-8x7b-tp8", "--treads", "1,2,3", "--modes", "E240,E0,E480,GR",
        "--out", "o"]


def _plan_of(*extra):
    return LF.make_plan(LF.build_parser().parse_args([*BASE, *extra]))


def test_the_phases_split_cells_probes_and_traces():
    t = _plan_of("--phase", "timed")
    assert t["timed_cells"] == 3 * 3 * 4 * 3 and t["traces"] == [] and t["probes"] == 2 * 3 * 3 * 4
    r = _plan_of("--phase", "trace", "--trace-treads", "1,2")
    assert r["timed_cells"] == 0 and r["probes"] == 0 and len(r["traces"]) == 2 * 3 * 2 * 3
    b = _plan_of("--trace-treads", "1")
    assert b["phase"] == "both" and b["timed_cells"] == t["timed_cells"] and b["traces"]


@pytest.mark.parametrize("extra, why", [
    (["--phase", "timed", "--trace-treads", "1"], "takes no --trace-treads"),
    (["--phase", "trace"], "needs --trace-treads"),
])
def test_each_phase_refuses_the_others_work(extra, why, capsys):
    assert LF.main([*BASE, *extra, "--dry-run"]) == exit_codes.REFUSED
    assert why in capsys.readouterr().out


def test_the_profiler_guard_raises_on_every_entry_point():
    import torch
    saved = (torch.profiler.profile.__init__, torch.autograd.profiler.profile.__init__)
    try:
        got = LF.install_profiler_guard(torch)
        assert got == ["torch.profiler.profile", "torch.autograd.profiler.profile"]
        with pytest.raises(LF.ProfilerForbidden):
            torch.profiler.profile()
        with pytest.raises(LF.ProfilerForbidden):
            torch.autograd.profiler.profile()
    finally:
        torch.profiler.profile.__init__, torch.autograd.profiler.profile.__init__ = saved
    LF.assert_profiler_off(torch)
    on = SimpleNamespace(autograd=SimpleNamespace(_profiler_enabled=lambda: True))
    with pytest.raises(LF.ProfilerForbidden):
        LF.assert_profiler_off(on)


class FakeGPU:
    """A sleep that holds for `hold_ns` of host clock; the clock advances by each
    call's `cost_ns`."""

    def __init__(self, clock, hold_ns):
        self.clock, self.hold_ns, self.until = clock, hold_ns, 0

    def hold(self, cycles):
        self.until = self.clock.t + self.hold_ns(cycles)
        return None

    def still_busy(self, ev):
        return self.clock.t < self.until

    def sync(self):
        pass


class Clock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t


def _fn(clock, ns, spike_at=None, spike=0):
    state = {"i": 0}

    def f():
        state["i"] += 1
        clock.t += ns + (spike if state["i"] == spike_at else 0)
    return f


def test_the_host_probe_reads_each_part_per_iteration():
    clock = Clock()
    gpu = FakeGPU(clock, lambda cycles: cycles * 1_000)     # 1 cycle = 1 us of hold
    parts = {"body": _fn(clock, 300_000), "flush": _fn(clock, 5_000),
             "event": _fn(clock, 2_000), "call": _fn(clock, 250_000)}
    r = LF.host_probe(parts, gpu, iters=8, repeats=3, cycles_per_ms=1000.0, clock=clock)
    assert r["status"] == "ok" and r["held"] is True
    assert r["H_pre_ms"] == pytest.approx(0.300) and r["h_flush_ms"] == pytest.approx(0.005)
    assert r["h_event_ms"] == pytest.approx(0.002) and r["h_call_ms"] == pytest.approx(0.250)
    assert r["max_over_median"] == pytest.approx(1.0)
    no_flush = LF.host_probe(dict(parts, flush=None), gpu, iters=8, repeats=1,
                             cycles_per_ms=1000.0, clock=clock)
    assert no_flush["h_flush_ms"] == 0.0


def test_a_synchronising_body_is_refused():
    clock = Clock()
    gpu = FakeGPU(clock, lambda cycles: 10 ** 12)
    # every 3rd iteration of every loop waits 5 ms: max/median 17 > 10
    state = {"i": 0}

    def body():
        state["i"] += 1
        clock.t += 300_000 + (5_000_000 if state["i"] % 3 == 0 else 0)
    r = LF.host_probe({"body": body, "flush": None, "event": _fn(clock, 1_000),
                       "call": _fn(clock, 1_000)}, gpu, iters=8, repeats=2,
                      cycles_per_ms=1000.0, clock=clock)
    assert r["status"] == "refused" and "synchronised" in r["detail"] and r["H_pre_ms"] is None


def test_a_gpu_not_held_through_the_loop_is_refused():
    clock = Clock()
    gpu = FakeGPU(clock, lambda cycles: 0)       # the sleep is over before the loop starts
    r = LF.host_probe({"body": _fn(clock, 300_000), "flush": None, "event": _fn(clock, 1_000),
                       "call": _fn(clock, 1_000)}, gpu, iters=8, repeats=1,
                      cycles_per_ms=1000.0, clock=clock)
    assert r["status"] == "refused" and r["held"] is False and "not held" in r["detail"]


def test_probe_parts_are_the_timed_loops_body():
    log = []
    ev = lambda: SimpleNamespace(record=lambda: log.append("event"))  # noqa: E731
    fl = SimpleNamespace(flush=lambda: log.append("flush"))
    parts = LF.probe_parts(lambda: log.append("call"), fl, ev)
    parts["body"]()
    assert log == ["flush", "event", "call", "event"]
    assert LF.probe_parts(lambda: None, None, ev)["flush"] is None
    assert set(LF.PROBE_COLUMNS) >= {"H_pre_ms", "h_flush_ms", "status", "held", "max_over_median"}
