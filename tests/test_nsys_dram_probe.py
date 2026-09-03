"""The nsys probe's report has to be attributable, and its clock has to be named.

WHAT WENT WRONG. `probe.json` was `asdict(Report)` and nothing else: no commit,
no dirty flag, no utc beyond a local `started` string, no instrument, no
bandwidth source, no `missing` map. Its directory was
`results_root()/nsys_dram_probe/<timestamp>` with NO card slug, on the network
volume the runbook uses BECAUSE it outlives the pod, so two pods' probes
interleaved in one tree and only opening a file said which machine wrote it.
That is collision 2 of `moe.bench.provenance`'s docstring wearing a new hat.

WORSE, AND THE REASON THIS IS NOT COSMETIC. `--peak-gbps` is the measured DRAM
ceiling that CONVERTS a percent-of-peak sample into bytes, so it sets the
reported traffic and any alpha derived from it, and it appeared in neither the
returned dict nor the report. It is in the run id and in the provenance block
now, as `bandwidth` with `bandwidth_source` listed in `missing`, because this
script cannot check where the operator got the number and pretending otherwise
is what `provenance_block` exists to stop.

AND THE CLOCK. The per-launch milliseconds and `achieved_gbps` in every workload
dict come from a wall clock around a queue of launches: no L2 flush, no clock
read under load, no trial spread. That is defensible for sizing an nsys window
and it is NOT comparable with anything `moe.bench.timing.time_kernel` produced,
so the two are never allowed to wear the same instrument name.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import provenance as PV  # noqa: E402
from moe.bench import timing  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "nsys_dram_probe", ROOT / "scripts" / "nsys_dram_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NP = _load()
H200 = "NVIDIA H200"
A100 = "NVIDIA A100-SXM4-80GB"


def _args(argv: list[str] | None = None):
    """The probe's own parser, so a knob added to it reaches these tests."""
    return NP.build_parser().parse_args(argv or [])


# --------------------------------------------------------------------------
# The run id.
# --------------------------------------------------------------------------

def test_the_run_id_names_the_card_first():
    assert NP.run_id_for(_args(), H200).startswith("nvidia_h200-")
    assert NP.run_id_for(_args(), H200) != NP.run_id_for(_args(), A100)


def test_the_run_id_is_stable_when_nothing_moves():
    """The other half of every id test in this repo: an id that changed on every
    call would hide the same failure the other way round."""
    assert NP.run_id_for(_args(), H200) == NP.run_id_for(_args(), H200)


@pytest.mark.parametrize("argv", [
    ["--calibrate"],
    ["--measure"],
    ["--sample-hz", "10000"],
    ["--seconds", "9.0"],
    ["--buffer-mb", "512"],
    ["--model", "mixtral-8x7b"],
    ["--tokens", "512"],
    # THE ONE THAT MATTERS MOST: it converts percent-of-peak into bytes, so two
    # sessions that differ only in it report different traffic from one trace.
    ["--peak-gbps", "4377.2"],
])
def test_every_knob_that_moves_a_number_moves_the_run_id(argv):
    assert NP.run_id_for(_args(argv), H200) != NP.run_id_for(_args(), H200)


def test_a_probe_with_no_card_is_labelled_as_having_none():
    """`--report` parses a trace on a laptop and `provenance.run_id` refuses an
    id without a card, so the no-card case gets a name `nvidia-smi` cannot
    produce rather than an empty string or a guess."""
    assert NP.run_id_for(_args(), NP.NO_CARD).startswith("no_card_nothing_measured-")
    with pytest.raises(PV.NoCard):
        NP.run_id_for(_args(), "")


# --------------------------------------------------------------------------
# The instrument.
# --------------------------------------------------------------------------

def test_the_wall_clock_is_never_called_the_one_instrument():
    assert NP.WALL_CLOCK_INSTRUMENT != timing.TIMING_BASIS
    assert "not-time_kernel" in NP.WALL_CLOCK_INSTRUMENT


def test_both_workloads_stamp_the_clock_that_timed_them():
    """Structural, because running either workload needs CUDA. Each workload
    returns a dict of milliseconds and GB/s that travels into probe.json, and a
    bandwidth figure whose own docstring calls it "the calibration case and the
    load-bearing one" has to say what measured it."""
    text = (ROOT / "scripts" / "nsys_dram_probe.py").read_text()
    assert text.count('"instrument": WALL_CLOCK_INSTRUMENT,') == 2


def test_the_peak_gbps_is_recorded_as_a_bandwidth_without_a_source():
    """The state `provenance_block` was written to make visible.

    The operator is told to take `--peak-gbps` from `measured_<device>.yaml`;
    this script cannot check that they did, so the number is recorded and its
    source is NAMED as absent. A publish gate is then supposed to fail on it.
    """
    prov = NP.probe_provenance(_args(["--peak-gbps", "4377.2"]))
    assert prov.bandwidth == pytest.approx(4377.2e9)
    assert prov.bandwidth_source is None
    assert prov.missing["bandwidth_source"] == "supplied without a source"
    assert prov.instrument == NP.WALL_CLOCK_INSTRUMENT

    # And with no ceiling supplied at all, the field is None with its own reason
    # rather than a silent default. `resolve_bandwidth`'s 4374.5 GB/s fallback is
    # exactly the shape this refuses to have.
    bare = NP.probe_provenance(_args())
    assert bare.bandwidth is None
    # "supplied as None" and "not supplied by caller" are different states and
    # this is the first: the script always names the field, and None means the
    # operator gave no ceiling.
    assert bare.missing["bandwidth"] == "supplied as None"


# --------------------------------------------------------------------------
# The report on disk.
# --------------------------------------------------------------------------

def _fake_discovery() -> NP.Discovery:
    return NP.Discovery(
        binary="/usr/local/bin/nsys", version="2025.3.1", version_raw="planted",
        device_flag="--gpu-metrics-device", set_flag="--gpu-metrics-set",
        frequency_flag="--gpu-metrics-frequency", offered=("all",),
        metric_sets=("ga100",), chosen_set=None, device_name=H200,
        compute_apps="none")


def test_probe_json_carries_the_provenance_the_audit_checks(tmp_path, monkeypatch):
    """The whole path, run: no invocation samples DRAM, the probe says NO, and
    the file it leaves behind names its commit, its machine and its knobs.

    Before 2026-09-02 this file had none of them and cost pod minutes to make.
    """
    monkeypatch.setattr(NP, "nsys_binary", lambda explicit: "/usr/local/bin/nsys")
    monkeypatch.setattr(NP, "discover", lambda binary: _fake_discovery())
    monkeypatch.setattr(NP, "run_ladder", lambda *a, **k: [])
    monkeypatch.setattr(NP, "compute_apps", lambda: "none")
    out = tmp_path / "probe"
    NP.main(["--out", str(out), "--peak-gbps", "4377.2"])

    doc = json.loads((out / "probe.json").read_text())
    for key in PV.TOP_LEVEL_KEYS:
        assert key in doc, key
    assert doc["provenance"]["git_sha"], "a probe names the commit that made it"
    assert doc["provenance"]["utc"]
    assert doc["instrument"] == NP.WALL_CLOCK_INSTRUMENT
    assert doc["provenance"]["bandwidth"] == pytest.approx(4377.2e9)
    assert doc["run_id"] and doc["settings"]["peak_gbps"] == pytest.approx(4377.2)
    assert doc["verdict"].startswith("NO.")


def test_the_output_directory_names_the_card_and_not_only_the_clock(
        tmp_path, monkeypatch):
    """Two pods, one network volume, one tree. The directory was a bare
    `%Y-%m-%d-%H%M%S`, so which card wrote a probe was legible only by opening
    it, and the results root is a network volume the runbook uses BECAUSE it
    outlives the pod.

    THE CARD COMES FROM `nvidia-smi`. The provenance block is planted with the
    OTHER card here on purpose: if the directory were named from
    `prov.gpu_name`, as it was between the provenance commit and this one, this
    test would read `nvidia_a100_...` and fail. See `probe_card` for why torch
    may not be asked.
    """
    monkeypatch.setattr(NP, "results_root", lambda: tmp_path)
    monkeypatch.setattr(NP, "gpu_name", lambda: H200)
    # The real function, captured BEFORE the patch: `NP.PV` is this module's
    # `PV`, so a lambda that called `PV.provenance_block` would call itself.
    real = PV.provenance_block
    monkeypatch.setattr(NP.PV, "provenance_block",
                        lambda **k: dataclasses.replace(real(**k), gpu_name=A100))
    monkeypatch.setattr(NP, "nsys_binary", lambda explicit: "/usr/local/bin/nsys")
    monkeypatch.setattr(NP, "discover", lambda binary: _fake_discovery())
    monkeypatch.setattr(NP, "run_ladder", lambda *a, **k: [])
    monkeypatch.setattr(NP, "compute_apps", lambda: "none")
    NP.main([])
    made = list((tmp_path / "nsys_dram_probe").iterdir())
    assert len(made) == 1 and made[0].name.startswith("nvidia_h200-")
    # ...and the timestamp is still there, after the id: two probes of one
    # machine at one setting are two different sessions.
    assert made[0].name.endswith(tuple("0123456789"))


# --------------------------------------------------------------------------
# The driver process may not touch CUDA.
# --------------------------------------------------------------------------

def test_the_card_for_the_run_id_is_asked_of_nvidia_smi_and_not_of_torch(
        monkeypatch):
    """`probe_card` is `nvidia-smi`, and an unavailable one is `NO_CARD`.

    `gpu_name` answers "unknown" when the query fails, which is a fine metric
    set selector and a terrible card: `provenance.run_id` would put
    `unknown-...` at the front of a directory that names no machine, which is
    the collision this whole slice closes wearing a friendlier word.
    """
    monkeypatch.setattr(NP, "gpu_name", lambda: H200)
    assert NP.probe_card() == H200
    monkeypatch.setattr(NP, "gpu_name", lambda: "unknown")
    assert NP.probe_card() == NP.NO_CARD


def test_no_cuda_context_is_created_before_either_neighbour_snapshot(
        tmp_path, monkeypatch):
    """THE FAIL BRANCH IS THE PREVIOUS COMMIT. Naming the card from
    `provenance_block().gpu_name` called `torch.cuda.current_device()`, whose
    `_lazy_init()` creates a primary context in the DRIVER process, before
    `discover()` ran `nvidia-smi --query-compute-apps`. The probe would then
    have listed itself as the neighbour in the one field that records whether
    it had one, and `compute_apps`' own docstring says that snapshot is the
    only honest thing a device-wide sampler can offer.

    Order, not absence, is what is asserted: the block is still built, and it
    is built last, from `write_report`.
    """
    log: list[str] = []
    monkeypatch.setattr(NP, "gpu_name", lambda: H200)
    monkeypatch.setattr(NP, "nsys_binary", lambda explicit: "/usr/local/bin/nsys")
    monkeypatch.setattr(NP, "discover",
                        lambda binary: (log.append("before-snapshot"),
                                        _fake_discovery())[1])
    monkeypatch.setattr(NP, "run_ladder", lambda *a, **k: (log.append("ladder"), [])[1])
    monkeypatch.setattr(NP, "compute_apps",
                        lambda: (log.append("after-snapshot"), "none")[1])
    real = PV.provenance_block
    monkeypatch.setattr(NP.PV, "provenance_block",
                        lambda **k: (log.append("provenance"), real(**k))[1])
    NP.main(["--out", str(tmp_path / "probe")])
    assert log == ["before-snapshot", "ladder", "after-snapshot", "provenance"]
    # ...and it really was built, so this is not passing by doing less.
    assert json.loads((tmp_path / "probe" / "probe.json").read_text())["git_sha"]
