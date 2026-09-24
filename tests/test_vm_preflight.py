"""scripts/vm_preflight.py: the VM's preflight, as pure functions of what the box said.

The preflight is the one stage of setup_vm.sh that decides whether a rented
box is READY, and its whole job is to refuse that word unless a DRAM counter
was actually read. So its checks are tested here as functions of planted
readings: PF5 through dram_counter_route.py's OWN probe chain (planted ncu and
child bytes through probe_reading, route_verdict, probe_gates, probe_exit),
the others through planted nvidia-smi and torch views.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import dram_counter_route as D  # noqa: E402
import vm_preflight as VP  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

H100 = "NVIDIA H100 80GB HBM3"
UUID = "GPU-0ffa33b8-1111-2222-3333-444455556666"


def _probe_world(**planted) -> tuple[int, dict, str]:
    """`--probe` over planted ncu and child bytes, through the probe's OWN
    chain (probe_reading -> route_verdict -> probe_gates -> probe_exit), as
    `(exit code, payload, log text)`: what PF5 reads off a real probe."""
    ncu = D.probe_reading("/planted/ncu", "Version 2025.3.1.0", **planted)
    caps = {"available": True, "cap_eff_field": "000001ffffffffff", "cap_eff_bits": 64,
            "sys_admin": True, "perfmon": True}
    verdict, notes = D.route_verdict(caps, {"available": True, "restrict": 1}, ncu,
                                     {"present": False})
    gates = D.probe_gates(verdict, ncu)
    log = "\n".join(line for g in gates for line in g.render())
    return D.probe_exit(verdict, gates), {"verdict": verdict, "notes": notes, "ncu": ncu}, log


LAUNCHED = f"{D.PK.MARKER} {D.PK.LAUNCHED} NVIDIA H100 80GB HBM3: one add_"
HEADER = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
OPEN_WORLD = dict(returncode=0, stdout=LAUNCHED, stderr="",
                  log_text=HEADER + '"0","probe","dram__bytes_read.sum","byte","4194304"\n')
BLOCKED_WORLD = dict(returncode=1, stdout=LAUNCHED,
                     stderr="==ERROR== ERR_NVGPUCTRPERM - The user does not have permission",
                     log_text="")
NOTHING_WORLD = dict(returncode=0, stdout="", stderr="",
                     log_text="==WARNING== No kernels were profiled.\n")


#: A planted STRICT class, shaped like dram_counter_route's R3_STRICT_METRICS.
STRICT = ("dram__bytes_read.sum", "launch__grid_size")


def _proven(payload: dict, metrics) -> dict:
    """`payload` with the probe's per-metric record planted: the r3-arms probe
    writes `ncu.metrics_proven`, the metrics that came back as numbers on one
    profiled launch. This checkout's probe chain (the ladder family) writes
    none, which is itself a payload PF5 must fail."""
    return {**payload, "ncu": {**payload["ncu"], "metrics_proven": list(metrics)}}


def test_pf5_passes_only_on_a_counter_that_came_back():
    """THE GATE the preflight exists for. A box on which ncu profiled a kernel
    and handed back a number for every STRICT metric is the only box PF5
    passes; a refused read, a probe that profiled nothing, a payload that
    disagrees with its own log, and a probe that could not run all fail, even
    with a per-metric record planted beside them."""
    rc, payload, log = _probe_world(**OPEN_WORLD)
    assert (rc, payload["verdict"]) == (exit_codes.DONE, "OPEN")
    assert VP.pf5_probe(rc, _proven(payload, STRICT), log, strict=STRICT).verdict == VP.PASS
    for world in (BLOCKED_WORLD, NOTHING_WORLD):
        rc, payload, log = _probe_world(**world)
        check = VP.pf5_probe(rc, _proven(payload, STRICT), log, strict=STRICT)
        assert check.verdict == VP.FAIL, payload["verdict"]
    rc, payload, log = _probe_world(**OPEN_WORLD)
    payload = _proven(payload, STRICT)
    liar = VP.pf5_probe(exit_codes.DONE, {**payload, "ncu": {**payload["ncu"],
                                                             "counters_read": False}}, log,
                        strict=STRICT)
    assert liar.verdict == VP.FAIL, "an OPEN word with no counter read is not a counter"
    mismatch = VP.pf5_probe(exit_codes.CLAIM_FAIL, payload, log, strict=STRICT)
    assert mismatch.verdict == VP.FAIL and "DEFECT" in mismatch.detail
    assert VP.pf5_probe(exit_codes.DONE, None, "", strict=STRICT).verdict == VP.FAIL
    assert VP.pf5_probe(None, None, "", strict=STRICT,
                        why_not_run="no ncu on PATH").verdict == VP.FAIL


def test_pf5_fails_an_open_probe_that_did_not_read_every_strict_metric():
    """PF5 passed on ONE counter (`counters_read`), and the run refuses a page
    whose STRICT metric is not numeric (its V2), so a box that proved only
    dram__bytes_read.sum was READY and then refused the run on paid time. An
    OPEN probe whose per-metric record lacks a STRICT metric fails, naming
    it; so does an OPEN probe that recorded no per-metric reading at all
    (the ladder family's payload), and a checkout with no STRICT class."""
    rc, payload, log = _probe_world(**OPEN_WORLD)
    assert (rc, payload["verdict"]) == (exit_codes.DONE, "OPEN")
    short = VP.pf5_probe(rc, _proven(payload, ["dram__bytes_read.sum"]), log, strict=STRICT)
    assert short.verdict == VP.FAIL and "launch__grid_size" in short.detail, short.detail
    assert short.data["strict_unread"] == ["launch__grid_size"]
    unrecorded = VP.pf5_probe(rc, payload, log, strict=STRICT)
    assert "metrics_proven" not in payload["ncu"]
    assert unrecorded.verdict == VP.FAIL and "no per-metric reading" in unrecorded.detail
    classless = VP.pf5_probe(rc, _proven(payload, STRICT), log, strict=None)
    assert classless.verdict == VP.FAIL and "STRICT" in classless.detail


def test_pf4_needs_every_strict_metric_and_drops_absent_recorded_ones():
    class Family:
        R3_STRICT_METRICS = ("dram__bytes_read.sum", "launch__grid_size")
        R3_CROSSCHECK_METRICS = ("lts__d_sectors_fill_device.sum",)
        R3_RECORDED_METRICS = ("launch__occupancy_limit_warps",)
    classes = VP.family_metric_classes(Family)
    good = VP.pf4_metrics(classes, "dram__bytes_read Counter byte", "launch__grid_size",
                          ncu="/x/ncu")
    assert good.verdict == VP.PASS, good.detail
    assert good.data["recorded_dropped"] == ["launch__occupancy_limit_warps"]
    assert good.data["crosscheck_absent"] == ["lts__d_sectors_fill_device.sum"]
    # launch__* is checked against the LAUNCH collection, not the profiling one
    wrong_place = VP.pf4_metrics(classes, "dram__bytes_read launch__grid_size", "", ncu="/x/ncu")
    assert wrong_place.verdict == VP.FAIL and "launch__grid_size" in wrong_place.detail
    assert VP.pf4_metrics(None, "dram__bytes_read", "", ncu="/x/ncu").verdict == VP.FAIL
    assert VP.pf4_metrics(classes, "", "", ncu=None).verdict == VP.FAIL
    assert VP.family_metric_classes(object()) is None


def _torch(**kw) -> dict:
    base = {"name": H100, "uuid": UUID[4:], "capability": "9.0", "sm_count": 132,
            "l2_bytes": 50 * 2**20, "memory_bytes": 80 * 10**9, "torch": "2.13.0+cu130",
            "cuda": "13.0", "triton": "3.7.1", "kernel_ok": True}
    return {**base, **kw}


SMI = VP.parse_smi(f"{H100}, {UUID}, 580.95.05, 81559, 81000",
                   "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0 |")


def test_pf1_names_one_card_and_says_it_is_not_the_studys():
    ok = VP.pf1_card(SMI, _torch(), _torch())
    assert ok.verdict == VP.PASS, ok.detail
    assert ok.data["slug"] == "nvidia_h100_80gb_hbm3" and ok.data["same_card_as_study"] is False
    assert ok.data["card_line"].startswith(
        "CARD NVIDIA H100 80GB HBM3 (nvidia_h100_80gb_hbm3, UUID 0ffa33b8-")
    assert "132 SMs, 50 MiB L2" in ok.data["card_line"]
    assert "NOT the study's nvidia_h200" in ok.detail
    other = VP.pf1_card(SMI, _torch(), _torch(uuid="aaaa"))
    assert other.verdict == VP.FAIL and "disagree" in other.detail
    blind = VP.pf1_card(SMI, _torch(), {"torch_error": "ModuleNotFoundError: torch"})
    assert blind.verdict == VP.FAIL and "vllm" in blind.detail
    assert VP.pf1_card(None, _torch(), _torch()).verdict == VP.FAIL


def test_pf2_catches_a_cu13_wheel_on_a_pre_r580_driver():
    assert VP.pf2_wheel(SMI, _torch(), _torch()).verdict == VP.PASS
    old = VP.parse_smi(f"{H100}, {UUID}, 575.57.08, 81559, 81000", "CUDA Version: 12.9")
    bad = VP.pf2_wheel(old, _torch(cuda="12.8"), _torch(cuda="13.0"))
    assert bad.verdict == VP.FAIL and "vllm: torch built for CUDA 13.0" in bad.detail
    dead = VP.pf2_wheel(SMI, _torch(), _torch(kernel_ok=False, kernel_error="no kernel image"))
    assert dead.verdict == VP.FAIL and "no kernel image" in dead.detail


def test_pf3_reads_its_pins_off_the_resolved_sets(tmp_path):
    (tmp_path / "resolved-base.txt").write_text("torch==2.13.0\ntriton==3.7.1\nnumpy==2.1\n")
    (tmp_path / "resolved-vllm.txt").write_text("vllm==0.27.1\ntorch==2.13.0\ntriton==3.7.1\n")
    pins = VP.resolved_pins(tmp_path)
    assert pins == {"base": {"torch": "2.13.0", "triton": "3.7.1"},
                    "vllm": {"vllm": "0.27.1", "torch": "2.13.0", "triton": "3.7.1"}}
    vllm = _torch(vllm="0.27.1", fused_experts=True, torch="2.13.0")
    assert VP.pf3_stack(_torch(), vllm, pins).verdict == VP.PASS
    drift = VP.pf3_stack(_torch(triton="3.6.0"), vllm, pins)
    assert drift.verdict == VP.FAIL and "base triton 3.6.0 (pinned 3.7.1)" in drift.detail
    no_moe = VP.pf3_stack(_torch(), {**vllm, "fused_experts": False}, pins)
    assert no_moe.verdict == VP.FAIL and "fused_experts" in no_moe.detail
    # the repository's own sets are what the VM is held to
    assert VP.resolved_pins(REPO / "requirements")["vllm"]["vllm"] == "0.27.1"


def test_pf7_asks_host_ram_only_when_the_card_cannot_hold_ncus_save():
    fp = 26 * 10**9
    assert VP.pf7_host_ram(80 * 10**9, fp, None).verdict == VP.PASS
    assert VP.pf7_host_ram(40 * 10**9, fp, 200 * 10**9).verdict == VP.PASS
    tight = VP.pf7_host_ram(40 * 10**9, fp, 30 * 10**9)
    assert tight.verdict == VP.FAIL and "1.5" in tight.detail


def test_pf7s_footprint_is_r3s_memory_plan_with_no_flush_buffer(monkeypatch):
    import private_weight_reference as PWR
    seen = {}
    real = PWR.memory_plan

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)
    monkeypatch.setattr(PWR, "memory_plan", spy)
    footprint, source = VP.r3_footprint()
    assert seen["flush"][0] == 0
    from moe.spec import MODEL_CONFIGS
    defaults = PWR.build_parser().parse_args([])
    per_copy = PWR.WEIGHTS.routed_expert_weight_bytes(MODEL_CONFIGS[defaults.model],
                                                      defaults.dtype)
    copies = int(re.search(r"(\d+) copies", source).group(1))
    assert footprint > copies * per_copy, "the footprint holds every declared copy"


def test_the_preflight_exits_zero_only_when_every_check_passed():
    passing = [VP.Check(f"PF{i}", "w", VP.PASS, "d") for i in range(1, 8)]
    for c in passing:
        parsed = exit_codes.parse_result_lines(c.result())
        assert parsed and parsed[0].verdict == exit_codes.PASS and parsed[0].name == c.id
    assert VP.exit_for(passing) == exit_codes.DONE
    for verdict in (VP.FAIL, VP.SKIP):
        one_off = passing[:4] + [VP.Check("PF5", "w", verdict, "d")] + passing[5:]
        assert VP.exit_for(one_off) == exit_codes.CLAIM_FAIL, verdict


def test_the_runbooks_footprint_is_r3s_own_plan():
    """docs/LAMBDA.md quotes R3's allocation for the A100's PF7 note; the
    number is recomputed from R3's memory_plan, so the page cannot drift."""
    footprint, _ = VP.r3_footprint()
    text = (REPO / "docs" / "LAMBDA.md").read_text()
    assert f"about {footprint / 1e9:.1f} GB, R3's own" in " ".join(text.split())


def test_pf4_names_ncus_lock_file_when_ncu_reports_it():
    """ncu's lock under /tmp belongs to the first user that ran ncu; a login
    user's and root's refuse each other. PF4 used to report that box as "ncu
    --query-metrics listed no metric at all", which sends the operator after
    the wrong ncu. The lock-file line is the cause, quoted."""
    class Family:
        R3_STRICT_METRICS = ("dram__bytes_read.sum",)
        R3_CROSSCHECK_METRICS = ()
        R3_RECORDED_METRICS = ()
    said = "==ERROR== Failed to create lock file /tmp/nsight-compute-lock: Permission denied"
    check = VP.pf4_metrics(VP.family_metric_classes(Family), said, "", ncu="/x/ncu")
    assert check.verdict == VP.FAIL
    assert "lock file" in check.detail and "/tmp/nsight-compute-lock" in check.detail
    assert "listed no metric" not in check.detail
    assert check.data["lock_file"] == said
    clean = VP.pf4_metrics(VP.family_metric_classes(Family), "dram__bytes_read", "",
                           ncu="/x/ncu")
    assert clean.verdict == VP.PASS and "lock_file" not in clean.data
