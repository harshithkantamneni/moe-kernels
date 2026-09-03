"""The swizzle sweep must survive its own apparatus breaking.

`tests/test_group_m_sweep.py` owns the science: the three stated laws, the fit,
the gates. This file owns the two ways the run ends without one, both of which
were exiting the interpreter's ONE, and ONE is CLAIM_FAIL:

  the ERROR handler   `raise SystemExit(main())` had nothing above it, so an
                      unplanned exception -- a torch OOM, an import that
                      drifted -- exited ONE. ONE is in FINISHED_CODES: the
                      session driver records it and never retries the arm. This
                      is the longer of the two alphas `pod_session.sh`
                      reconciles as the last thing on the screen, so the arm
                      that would be lost is not a cheap one.

  the second door     `timing.TimingRefused` subclasses RuntimeError, so the
                      per-cell `except Exception` in `measure` caught every
                      refusal the INSTRUMENT itself raises and wrote it as one
                      cell's `ms_p50=nan` record. The refusal is the same fact
                      for every cell, so the sweep filled cells.jsonl with them
                      and then reported "nothing was timed" over a stated
                      reason that named the last cell's exception rather than
                      the instrument that had refused all of them.

`measure` is run against a faked pod rather than described, because a handler
can only be shown to be at the call site by executing the call site. `torch` is
swapped in `sys.modules` for a proxy that drops `device="cuda"` and answers
`is_available()`, vLLM's entry points are stubbed, and the instrument is a
callable that records what it was handed -- which is also how the LEVEL
reference resolved for this card is checked to reach the cells it scores.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch as real_torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.baselines import _framework_config as FC  # noqa: E402
from moe.bench import exit_codes, timing  # noqa: E402
from moe.reference import torch_ref as TORCH_REF  # noqa: E402
from moe.routing import distributions as DIST  # noqa: E402

H200 = "NVIDIA H200"
#: What `moe/bench/hardware/measured_nvidia_h200.yaml` publishes for its dense
#: GEMM, and therefore what LEVEL on this card is scored against.
H200_REFERENCE_MHZ = 1515.0


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GM = _load("group_m_alpha_sweep")


# --------------------------------------------------------------------------
# a pod, faked at the seams `measure` actually reaches through
# --------------------------------------------------------------------------

class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_properties(index):
        return types.SimpleNamespace(name=H200)

    @staticmethod
    def synchronize() -> None:
        pass


class _CudaLessTorch(types.ModuleType):
    """Real torch with `cuda` answered and `device=` dropped.

    A `ModuleType` subclass because `measure` does `import torch` inside the
    function, so `sys.modules` is the only seam: a module attribute cannot be
    patched onto a name the function re-imports on every call. A proxy rather
    than a stub, so `equal`, `bincount` and `zeros` stay the real ones and a
    shape the sweep gets wrong still fails here.
    """

    cuda = _Cuda

    def __getattr__(self, name):
        return getattr(real_torch, name)

    def full(self, *args, **kwargs):
        kwargs.pop("device", None)
        return real_torch.full(*args, **kwargs)


def timing_at(load_mhz: float, reference_mhz: float | None,
              trials: int = 3) -> timing.KernelTiming:
    """A `KernelTiming` whose clock verdicts come from the REAL `clock_flags`.

    Hand-set flags would answer the question these tests ask -- whether the
    sweep hands the instrument a reference at all -- whichever way it went.
    """
    level, drift = timing.clock_flags(load_mhz, load_mhz, load_mhz, reference_mhz)
    return timing.KernelTiming(
        ms_p50=1.0, ms_p90=1.1, ms_min=0.9, ms_std=0.01, iters=100,
        trials=trials, warmup_ms=300.0, l2_flush=True,
        sm_clock_load_mhz=load_mhz, sm_clock_start_mhz=load_mhz,
        sm_clock_end_mhz=load_mhz, clock_level_ok=level, clock_drift_ok=drift,
        samples=100 * trials, warmup_calls=10, flush_mb=64, clock_samples=9,
        clock_source="injected", clock_poll_ms=1.0, host_bound=False,
        host_enqueue_ms=0.01, clock_note="scripted clock")


@pytest.fixture
def pod(monkeypatch, tmp_path):
    state = types.SimpleNamespace(seen=[], out=tmp_path, timing_result=None)

    monkeypatch.setitem(sys.modules, "torch", _CudaLessTorch("torch"))
    fused = types.ModuleType("vllm.model_executor.layers.fused_moe")
    fused.fused_experts = lambda **kw: real_torch.zeros(2, 2)
    activation = types.ModuleType("vllm.model_executor.layers.fused_moe.activation")
    activation.MoEActivation = lambda value: value
    for name, module in (
            ("vllm", types.ModuleType("vllm")),
            ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
            ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
            ("vllm.model_executor.layers.fused_moe", fused),
            ("vllm.model_executor.layers.fused_moe.activation", activation)):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(GM, "find_override_config",
                        lambda: (lambda conf: contextlib.nullcontext(), "fake.module"))
    monkeypatch.setattr(FC, "recording_tile_config",
                        lambda capture, *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(FC, "vllm_override_active", lambda *a, **k: True)
    monkeypatch.setattr(TORCH_REF, "make_inputs",
                        lambda spec, device=None: (real_torch.zeros(2, 2),
                                                   types.SimpleNamespace(w1=None, w2=None)))
    monkeypatch.setattr(DIST, "sample_topk_ids",
                        lambda routing, tokens, experts, k, seed=0, device=None:
                        real_torch.zeros((tokens, k), dtype=real_torch.long))

    def timer(fn, **kwargs):
        state.seen.append(kwargs)
        result = state.timing_result
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result(**kwargs)
        return result

    monkeypatch.setattr(timing, "time_kernel", timer)
    return state


def measure(pod):
    """`measure` over the smallest plan that still sweeps two settings."""
    args = GM.parse_args(["--run", "--model", "mixtral-8x7b", "--tokens", "8",
                          "--group-m", "1,8", "--routings", "uniform",
                          "--seeds", "1"])
    return GM.measure(GM.build_plan(args), args, pod.out, set())


def written_records(pod) -> list[dict]:
    path = pod.out / "cells.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# --------------------------------------------------------------------------
# H1: an unplanned exception is ERROR, and a refusal is REFUSED
# --------------------------------------------------------------------------

def test_an_unplanned_exception_is_error_and_not_a_claim_that_failed(
        monkeypatch, capsys):
    """ONE is CLAIM_FAIL: a RESULT, latched by the session driver, skipped on
    every resume. A crash in the last cell of the long arm was filed as this
    experiment's registered answer about alpha. ERROR (4) is outside
    FINISHED_CODES precisely so the apparatus breaking can be told from the
    claim not holding, and it is the only retryable code."""
    def boom(argv=None):
        raise RuntimeError("torch OOM on the pod")

    monkeypatch.setattr(GM, "_run", boom)
    assert GM.main([]) == exit_codes.ERROR
    err = capsys.readouterr().err
    assert "torch OOM on the pod" in err, "the traceback was swallowed"
    assert "RuntimeError" in err
    assert exit_codes.ledger_state(exit_codes.ERROR) == "RETRY"
    assert exit_codes.ledger_state(exit_codes.CLAIM_FAIL) != "RETRY"

    # ...and the PASS branch: a run that reached a verdict keeps its own code.
    monkeypatch.setattr(GM, "_run", lambda argv=None: exit_codes.CLAIM_FAIL)
    assert GM.main([]) == exit_codes.CLAIM_FAIL
    monkeypatch.setattr(GM, "_run", lambda argv=None: exit_codes.DONE)
    assert GM.main([]) == exit_codes.DONE


def test_the_planning_refusal_still_returns_refused(monkeypatch, capsys):
    """The handler that was already there must survive being wrapped: a
    `CannotRunHere` out of planning is a precondition, not a crash."""
    def cannot(argv=None):
        raise GM.CannotRunHere("vLLM is not importable in this interpreter")

    monkeypatch.setattr(GM, "_run", cannot)
    assert GM.main([]) == exit_codes.REFUSED
    assert "REFUSED" in capsys.readouterr().err


def test_a_refusal_sentence_is_refused_and_not_a_claim_that_failed(
        monkeypatch, capsys):
    """`raise SystemExit(<str>)` sets `code` to the STRING and exits ONE. This
    file raises none today and says so; the branch exists so that one added
    later, or one out of a library it imports, cannot land as a refuted claim."""
    def refuse(argv=None):
        raise SystemExit("the calibration for this card records no clock")

    monkeypatch.setattr(GM, "_run", refuse)
    assert GM.main([]) == exit_codes.REFUSED
    assert "records no clock" in capsys.readouterr().err


def test_an_integer_systemexit_still_means_what_it_says():
    """The FAIL branch of the string test. argparse exits `SystemExit(2)` after
    printing its own message; reclassifying it would report a refusal twice and
    hide a usage error behind it."""
    with pytest.raises(SystemExit) as caught:
        GM.main(["--model", "not-a-model"])
    assert caught.value.code == 2


# --------------------------------------------------------------------------
# H3: the instrument's own refusal is not one cell's error
# --------------------------------------------------------------------------

def test_an_instrument_refusal_leaves_the_sweep_instead_of_being_recorded(pod):
    """THE SECOND DOOR, executed at the call site. `TimingRefused` subclasses
    RuntimeError, so `measure`'s per-cell handler recorded every refusal the
    instrument raises -- trials=0, a meaningless warmup, no CUDA and no
    injected fakes -- as this one cell's failure. It is the same fact for every
    cell, so cells.jsonl filled with nan records and the report's stated reason
    named the last cell's exception instead of the instrument."""
    pod.timing_result = timing.TimingRefused(
        "trials=0: a measurement needs at least one trial")
    with pytest.raises(timing.TimingRefused, match="at least one trial"):
        measure(pod)
    assert written_records(pod) == [], (
        "a record was written for a cell that was never measured")


def test_an_escaped_instrument_refusal_exits_refused_and_not_error(
        monkeypatch, capsys):
    """Where it lands. Nothing was measured and nothing was spent, which is
    REFUSED (2) and not ERROR (4); `cli._main` catches the same base class
    around `driver.run_sweep` and exits the same code, so the two entry points
    into this repository's instrument agree."""
    def refuse(argv=None):
        raise timing.TimingRefused("warmup_ms=0 makes the measurement meaningless")

    monkeypatch.setattr(GM, "_run", refuse)
    assert GM.main([]) == exit_codes.REFUSED
    assert "meaningless" in capsys.readouterr().err


def test_a_kernels_own_runtime_error_is_still_one_cells_error(pod):
    """The PASS branch of the same door, and why it is a subclass check rather
    than a blanket re-raise: a kernel that launched badly IS a per-cell fact,
    the record keeps it, and the sweep goes on to the next setting."""
    pod.timing_result = RuntimeError("CUDA error: an illegal memory access")
    fresh, _ = measure(pod)
    assert len(fresh) == 2
    assert all("illegal memory access" in r["error"] for r in fresh)
    assert len(written_records(pod)) == 2


# --------------------------------------------------------------------------
# H2: the reference this arm resolves reaches the cells it scores
# --------------------------------------------------------------------------

def test_a_sagging_card_reads_clock_level_ok_false_on_every_row(pod):
    """The LEVEL flag with a left-hand side. 1000 MHz against this card's own
    1515 MHz GEMM clock is under `LEVEL_FRACTION`, so the real `clock_flags`
    says False; the record carries the verdict AND the number it was scored
    against, because the tri-state alone cannot tell a row that had no
    reference from one that passed."""
    pod.timing_result = lambda **kw: timing_at(1000.0, kw["reference_clock_mhz"])
    fresh, meta = measure(pod)

    assert [kw["reference_clock_mhz"] for kw in pod.seen] == \
        [H200_REFERENCE_MHZ] * 2, "the instrument was handed no reference"
    assert meta["reference_clock_mhz"] == H200_REFERENCE_MHZ
    assert "gemm_clock" in meta["reference_clock_source"]
    assert [r["clock_level_ok"] for r in fresh] == [False, False]
    assert [r["reference_clock_mhz"] for r in fresh] == [H200_REFERENCE_MHZ] * 2


def test_a_card_at_its_reference_clock_reads_true(pod):
    """The PASS branch of the same flag: a column that can only say False is as
    useless as one that can only say nothing."""
    pod.timing_result = lambda **kw: timing_at(H200_REFERENCE_MHZ,
                                               kw["reference_clock_mhz"])
    fresh, _ = measure(pod)
    assert [r["clock_level_ok"] for r in fresh] == [True, True]
