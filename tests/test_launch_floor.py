"""scripts/launch_floor.py off the GPU: the driver's CLI contract, the plan and
its refusals, the mirror of R3's time_cell (it must size every burst exactly
as time_cell does), the trace parser on a planted Kineto trace, and the
refusal without a device."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import launch_floor as LF  # noqa: E402
import private_weight_reference as PW  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

CONTRACT = ["--model", "granite-3.0-3b-a800m", "--treads", "1,2,3,4,5,6,7,8,9",
            "--modes", "E240,E0,E480,GR", "--trace-treads", "1,2,4,6,9",
            "--arms", "native,shared,private", "--group-m", "4", "--duty", "0.25",
            "--repeats", "3", "--seed", "0", "--out", "OUT"]


def test_the_driver_contract_parses_to_the_registered_design():
    a = LF.build_parser().parse_args(CONTRACT)
    assert a.model == "granite-3.0-3b-a800m" and a.treads == list(range(1, 10))
    assert a.modes == ["E240", "E0", "E480", "GR"] and a.trace_treads == [1, 2, 4, 6, 9]
    assert a.arms == ["native", "shared", "private"] and a.lever_trace_treads == [1, 2]
    assert (a.group_m, a.duty, a.repeats, a.seed) == (4, 0.25, 3, 0)
    assert str(a.out) == "OUT" and a.dry_run is False
    with pytest.raises(SystemExit):
        LF.build_parser().parse_args([*CONTRACT[:6], "E240,EX"])


def test_the_plan_is_r3s_design_and_counts_its_cells_and_minutes():
    plan = LF.make_plan(LF.build_parser().parse_args(CONTRACT))
    assert plan["copies_declared"] == 9
    assert plan["declared_by_arm"] == {"native": 40, "shared": 360, "private": 360}
    assert plan["pinned"] == {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4,
                              "num_warps": 8, "num_stages": 4, "BLOCK_SIZE_M": 32}
    assert plan["timed_cells"] == 9 * 3 * 4 * 3 == 324
    # TR-E and TR-G at E240 at every trace tread, and at E0 and E480 at the
    # lever treads 1 and 2 (the review: E0 also moves D_tail)
    assert len(plan["traces"]) == 5 * 3 * 2 + 2 * 3 * 2 * 2
    lever = {(t["n"], t["flush"]) for t in plan["traces"]}
    assert (1, "E0") in lever and (2, "E480") in lever and (4, "E0") not in lever
    assert plan["minutes"] == pytest.approx(324 * 3.0 / 60 + 54 * 4.0 / 60 + 3.0, abs=0.1)
    # P1's GR read comes from R3's own timed defaults
    assert (plan["warmup_ms"], plan["cell_budget_ms"], plan["trials"]) == (300.0, 200.0, 3)


@pytest.mark.parametrize("swap,why", [
    (("--treads", "1,2,10"), "not on R3's counter ladder"),
    (("--duty", "1.0"), "duty-cycled"),
    (("--trace-treads", "11"), "not timed treads"),
])
def test_what_r3_would_not_time_is_refused_before_anything(swap, why, capsys):
    argv = list(CONTRACT)
    argv[argv.index(swap[0]) + 1] = swap[1]
    assert LF.main([*argv, "--dry-run"]) == exit_codes.REFUSED
    assert why in capsys.readouterr().out


def test_the_dry_run_prints_the_plan_and_exits_refused(capsys):
    assert LF.main([*CONTRACT, "--dry-run"]) == exit_codes.REFUSED
    out = capsys.readouterr().out
    assert "LAUNCH FLOOR PLAN granite-3.0-3b-a800m" in out and "324 timed cells" in out
    assert "estimated 22.8 min" in out and "DRY RUN" in out


def test_the_1b_runs_r3s_9_deep_ladder():
    a = LF.build_parser().parse_args(["--model", "granite-3.0-1b-a400m", "--treads",
                                      "1,2,3,4,5,6,7,8,9", "--modes", "E240,GR",
                                      "--trace-treads", "1,6,9", "--out", "o"])
    plan = LF.make_plan(a)
    assert plan["copies_declared"] == 9 and plan["timed_cells"] == 162
    assert plan["declared_by_arm"]["private"] == 32 * 9


def test_no_device_is_a_refusal_not_a_number(monkeypatch, capsys, tmp_path):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    argv = list(CONTRACT)
    argv[argv.index("--out") + 1] = str(tmp_path / "o")
    assert LF.main(argv) == exit_codes.REFUSED
    assert "REFUSED: no CUDA device" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def _duty_result(**kw):
    base = dict(ms_p50=0.252, ms_p90=0.26, ms_min=0.25, ms_std=0.001, samples=395,
                bursts=5, calls_per_burst=80, gap_ms=120.0, duty_achieved=0.25,
                host_bound=False, host_enqueue_ms=0.21, host_backlog_iters=37.0,
                sm_clock_load_mhz=1710.0, clock_samples_mhz=(1710.0, 1710.0),
                clock_level_ok=True, clock_drift_ok=True, power_w=300.0,
                mem_clock_mhz=2619.0, clock_note="", trials=3, instrument="fake/time_duty",
                warmup_ms=300.0, clock_level_side="", head_ms=0.25, tail_ms=0.25,
                within_burst_ok=True, gap_basis="burst", l2_flush=True)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("sizing_ms", [0.5, 0.0952, 0.31])
def test_the_mirror_sizes_every_burst_exactly_as_time_cell(sizing_ms):
    """Same fake timer, same fake duty timer: the duty timer's kwargs from
    the mirror are time_cell's plus the injected flusher, nothing else."""
    seen = {}

    def timer(fn, **kw):
        seen.setdefault("timer", []).append(kw)
        return SimpleNamespace(ms_p50=sizing_ms)

    def duty_timer(fn, **kw):
        seen.setdefault("duty", []).append(kw)
        return _duty_result()
    common = dict(duty=0.25, warmup_ms=300.0, cell_budget_ms=200.0, trials=3,
                  l2_flush=True, reference_clock_mhz=1710.0, timer=timer,
                  duty_timer=duty_timer)
    PW.time_cell(lambda: None, **common)
    flusher = SimpleNamespace(megabytes=240, flush=lambda: None)
    t, per_call = LF.mirror_time_cell(lambda: None, flusher=flusher, **common)
    (s_r3, s_lf), (d_r3, d_lf) = seen["timer"], seen["duty"]
    assert s_lf.pop("flusher") is flusher and d_lf.pop("flusher") is flusher
    assert s_r3 == s_lf and d_r3 == d_lf
    assert per_call == pytest.approx(max(sizing_ms, 1e-4))
    assert t.host_enqueue_ms == 0.21


def test_e0_injects_no_flusher_and_flushes_nothing():
    seen = []
    LF.mirror_time_cell(lambda: None, flusher=None, l2_flush=False, duty=0.25,
                        timer=lambda fn, **kw: seen.append(kw) or SimpleNamespace(ms_p50=0.3),
                        duty_timer=lambda fn, **kw: seen.append(kw) or _duty_result())
    assert all("flusher" not in kw and kw["l2_flush"] is False for kw in seen)


def test_every_row_keeps_the_host_enqueue_even_when_gpu_bound():
    r = LF.row_from(_duty_result(host_bound=False), 0.25, model="m", mode="E240", arm="shared",
                    tiles=7, repeat=0, flush_mb=240)
    assert r["status"] == "ok" and r["host_enqueue_ms"] == 0.21 and r["backlog"] == 37.0
    assert r["host_bound"] is False and r["clock_samples_mhz"] == "1710 1710"
    assert set(r) == set(LF.CSV_COLUMNS)
    bad = LF.row_from(None, None, model="m", mode="GR", status="failed", detail="GR NOT CAPTURED")
    assert bad["status"] == "failed" and bad["ms_p50"] == ""


def _x(name, cat, ts, dur, corr=None):
    e = {"ph": "X", "name": name, "cat": cat, "ts": ts, "dur": dur}
    if corr is not None:
        e["args"] = {"correlation": corr}
    return e


def planted_trace():
    """Two calls of the timed loop body, as Kineto writes them (us). Call 0:
    flush annotation at 0 (host 10 us) launching the flush kernel (68 us on
    the GPU); start event record at 12; r3call 15..265 launching align,
    w1 GEMM, silu, w2 GEMM and moe_sum, whose kernels run on the GPU after
    the flush; end record at 266. Call 1 the same, 300 us later."""
    ev = []
    for i in range(2):
        o = 300.0 * i
        c = 100 * (i + 1)
        ev += [_x("r3flush", "user_annotation", o + 0, 10),
               _x("cudaLaunchKernel", "cuda_runtime", o + 2, 5, c),
               _x("reduce_kernel", "kernel", o + 20, 68, c),
               _x("cudaEventRecord", "cuda_runtime", o + 12, 2),
               _x("r3call", "user_annotation", o + 15, 250)]
        names = ["moe_align_block_size_kernel", "fused_moe_kernel", "act_and_mul_kernel",
                 "fused_moe_kernel", "moe_sum_kernel"]
        launch_at = [40, 90, 140, 190, 240]
        k_start = [95, 120, 170, 200, 252]
        k_dur = [6, 20, 4, 18, 3]
        for j, (nm, la, ks, kd) in enumerate(zip(names, launch_at, k_start, k_dur, strict=True)):
            ev.append(_x("cudaLaunchKernel", "cuda_runtime", o + la, 6, c + j + 1))
            ev.append(_x(nm, "kernel", o + ks, kd, c + j + 1))
        ev.append(_x("cudaEventRecord", "cuda_runtime", o + 266, 2))
    return {"traceEvents": ev + [{"ph": "M", "name": "process_name"}]}


def test_parse_trace_reads_the_planted_burst():
    got = LF.parse_trace(planted_trace())
    assert len(got["calls"]) == 2
    c = got["calls"][0]
    assert c["host_span_us"] == 250 and c["h_flush_us"] == 10 and c["flush_us"] == 68
    assert c["event_record_us"] == [2]
    assert c["launch_offsets_us"] == [40 - 12, 90 - 12, 140 - 12, 190 - 12, 240 - 12]
    assert [k["dur_us"] for k in c["kernels"]] == [6, 20, 4, 18, 3]
    assert c["gemm_us"] == 38 and c["non_gemm_us"] == 13 and c["kernel_sum_us"] == 51
    assert c["gaps_us"] == [120 - 101, 170 - 140, 200 - 174, 252 - 218]
    # lambda: kernel start minus its launch-API return, least over the call
    assert c["lambda_us"] == min(95 - 46, 120 - 96, 170 - 146, 200 - 196, 252 - 246)
    # D_tail: the last kernel's end after the last launch-API return
    assert c["d_tail_us"] == 255 - 246
    assert got["median"]["flush_us"] == 68 and got["median"]["h_flush_us"] == 10
    assert got["launch_api_share"] == pytest.approx(30 / 250)


# --------------------------------------------------------------------------
# rental 3 (2026-10-05): the flush ladder's modes and the NATIVE wall-time check
# --------------------------------------------------------------------------

class _FakeFlusher:
    def __init__(self, megabytes):
        self.megabytes = megabytes


_FAKE_T = SimpleNamespace(L2Flusher=_FakeFlusher, flush_mb_for_device=lambda: 240)


def test_every_mode_has_a_flusher_of_its_registered_size():
    """The run loop indexes `flushers[mode]` (probe_all, time_cells) and
    `flushers[tr["flush"]]` (trace_cell): a mode in MODES without an entry raises
    KeyError on the GPU, past the dry run, which never builds them."""
    fl = LF.make_flushers(_FAKE_T)
    assert set(fl) == set(LF.MODES) == set(LF.FLUSH_MB)
    assert {"E120", "E360"} <= set(LF.MODES)
    for mode, mb in LF.FLUSH_MB.items():
        if mb == 0:
            assert fl[mode] is None
        else:
            assert fl[mode].megabytes == (240 if mb is None else mb), mode
    assert fl["GR"] is fl["E240"]
    assert (fl["E120"].megabytes, fl["E360"].megabytes) == (120, 360)
    # every flush a planned trace names is a key too
    a = LF.build_parser().parse_args([*CONTRACT[:4], "--modes", "E0,E240,E360,E480",
                                      "--trace-treads", "1", "--out", "OUT"])
    for tr in LF.make_plan(a)["traces"]:
        assert tr["flush"] in fl


def test_run_builds_its_flushers_from_make_flushers_only():
    import inspect
    src = inspect.getsource(LF.run)
    assert "flushers = make_flushers(T)" in src
    assert src.count("flushers = {") == 0 and "L2Flusher(" not in src


@pytest.mark.parametrize("modes", ["E0,E240,E360,E480", "E120", "E240,E120,E360,GR"])
def test_the_flush_ladder_modes_plan(modes):
    a = LF.build_parser().parse_args(["--model", "mixtral-8x7b-tp8", "--treads", "1,2,3",
                                      "--modes", modes, "--out", "OUT", "--phase", "timed"])
    plan = LF.make_plan(a)
    assert plan["modes"] == modes.split(",")
    assert plan["timed_cells"] == 3 * 3 * len(modes.split(",")) * 3


WALL = ["--model", "mixtral-8x7b-tp4", "--treads", "2,4,6,8,10,11,12,13,14,15,16",
        "--modes", "E240,GR", "--group-m", "64", "--out", "OUT", "--phase", "timed"]


def test_a_native_only_plan_takes_treads_past_the_counter_ladder():
    plan = LF.make_plan(LF.build_parser().parse_args([*WALL, "--arms", "native"]))
    assert plan["treads"][-1] == 16 == PW.NATIVE_COUNTER_MAX_TREADS
    assert plan["arms"] == ["native"] and plan["group_m"] == 64
    assert plan["timed_cells"] == 11 * 2 * 3
    with pytest.raises(LF.Refused, match="counter ladder"):
        LF.make_plan(LF.build_parser().parse_args([*WALL, "--arms", "native,shared"]))
    with pytest.raises(LF.Refused, match="NATIVE's ladder"):
        LF.make_plan(LF.build_parser().parse_args(
            ["--model", "mixtral-8x7b-tp4", "--treads", "17", "--arms", "native", "--out", "OUT"]))


def test_the_ladder_is_natives_only_for_a_native_only_plan():
    cfg = LF.MODEL_CONFIGS["mixtral-8x7b-tp4"]
    lad, bound = LF.plan_ladder(cfg, ["native"])
    assert not bound and lad == PW.ladder_treads(cfg, LF.BLOCK_M, PW.NATIVE_COUNTER_MAX_TREADS)
    lad, bound = LF.plan_ladder(cfg, ["native", "private"])
    assert bound and lad == PW.counter_ladder(cfg, LF.BLOCK_M)


def test_the_inputs_build_only_the_plans_arms():
    """private_topk_ids refuses a tread past the copies declared by design, so a
    native-only plan must not build it: calls_for hands its arms to arm_inputs."""
    import inspect
    src = inspect.getsource(LF.run)
    assert 'PW.arm_inputs(\n            cfg, n, BLOCK_M, copies, args.seed, "bf16", w1.dtype, arms=tuple(args.arms))' in src
    assert "arms" in inspect.signature(PW.arm_inputs).parameters
