"""The tile sweep must be able to reach a verdict, and to fail to reach one.

Three defects, all of them silent, all of them found by a reviewer who did not
own the file, and each one planted here in both directions:

  the ERROR handler   an unplanned exception left `main` as the interpreter's
                      exit ONE, and ONE is CLAIM_FAIL, which the session driver
                      LATCHES: a torch OOM three tiles in was filed as this
                      experiment's registered answer and the arm was never
                      re-run. ERROR (4) is the only retryable code.

  the dead reference  `time_kernel` was called with no `reference_clock_mhz`
                      while the row carried a `clock_level_ok` column and the
                      loop branched on `t.clock_level_ok is False`. Without a
                      reference `clock_flags` leaves that field None, so the
                      column was empty on every row ever written and the branch
                      could not run. A FLAT curve is what C1 reads as
                      confirmation, and a card that sagged at one setting can
                      manufacture flatness; the apparatus could not tell.

  the second door     `timing.TimingRefused` subclasses RuntimeError, so the
                      per-cell `except Exception` caught every refusal the
                      INSTRUMENT itself raises and filed it as one cell's
                      failure. The refusal is the same fact for every cell, so
                      the sweep wrote a page of FAILED rows and still scored
                      gates over them.

The pod is faked rather than described. `torch` is proxied so `device="cuda"`
is dropped and `is_available()` is True, vLLM's two entry points are stubbed
into `sys.modules`, and the instrument is replaced by a callable that records
what it was handed. That is enough to run `main` end to end off a GPU, which is
the only way these three can be shown at the call site rather than asserted
about it.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch as real_torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes, timing  # noqa: E402

H200 = "NVIDIA H200"
#: The clock `moe/bench/hardware/measured_nvidia_h200.yaml` publishes for its
#: dense GEMM. Named here so a test that asserts the instrument was handed a
#: reference asserts it was handed THE reference, not merely something.
H200_REFERENCE_MHZ = 1515.0
#: No calibration names this, and none can: `measured_slug` would look for
#: `measured_nvidia_not_a_card.yaml`.
UNCALIBRATED = "NVIDIA NOT-A-CARD"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TILE = _load("tile_sweep")


# --------------------------------------------------------------------------
# a pod, faked at the seams the script actually reaches through
# --------------------------------------------------------------------------

class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(index=0) -> str:
        return H200

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def synchronize() -> None:
        pass


class _CudaLessTorch:
    """Real torch with the two things a laptop cannot provide overridden.

    A proxy and not a stub: every call the sweep makes on the way to the
    instrument -- `bincount`, `zeros`, `long` -- is the real one, so a shape
    the script gets wrong still fails here. Only `cuda` and the `device=`
    keyword are faked, because those are exactly what a laptop lacks.
    """

    cuda = _Cuda

    def __getattr__(self, name):
        return getattr(real_torch, name)

    def full(self, *args, **kwargs):
        kwargs.pop("device", None)
        return real_torch.full(*args, **kwargs)


def _fake_vllm(monkeypatch) -> None:
    """vLLM's two entry points, stubbed into `sys.modules`.

    `main` imports them inline, after the CUDA check, so they cannot be
    monkeypatched as module attributes; `setitem` is the seam, and it is undone
    with the rest of the patches.
    """
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


def timing_at(load_mhz: float, reference_mhz: float | None,
              trials: int = 3, warmup_ms: float = 300.0,
              l2_flush: bool = True) -> timing.KernelTiming:
    """A `KernelTiming` whose clock verdicts come from the REAL `clock_flags`.

    Not hand-set booleans: the question these tests ask is whether the sweep
    hands the instrument a reference, and a hand-set flag would answer it
    whatever the sweep passed.
    """
    level, drift = timing.clock_flags(load_mhz, load_mhz, load_mhz, reference_mhz)
    return timing.KernelTiming(
        ms_p50=1.0, ms_p90=1.1, ms_min=0.9, ms_std=0.01, iters=100,
        trials=trials, warmup_ms=warmup_ms, l2_flush=l2_flush,
        sm_clock_load_mhz=load_mhz, sm_clock_start_mhz=load_mhz,
        sm_clock_end_mhz=load_mhz, clock_level_ok=level, clock_drift_ok=drift,
        samples=100 * trials, warmup_calls=10, flush_mb=64, clock_samples=9,
        clock_source="injected", clock_poll_ms=1.0, host_bound=False,
        host_enqueue_ms=0.01, clock_note="scripted clock",
        # The side and the reference, as `time_kernel` stamps them: LEVEL is
        # two-sided and a consumer reads which way it went.
        clock_level_side=timing.level_side(load_mhz, reference_mhz) or "",
        reference_clock_mhz=reference_mhz)


@pytest.fixture
def pod(monkeypatch, tmp_path):
    """Everything `main` reaches for between the CUDA check and the CSV.

    `seen` collects the keyword arguments each `time_kernel` call was handed,
    which is where the reference clock either arrives or does not.
    """
    state = types.SimpleNamespace(seen=[], out=tmp_path, timing_result=None)

    monkeypatch.setattr(TILE, "torch", _CudaLessTorch())
    monkeypatch.setattr(TILE, "find_override",
                        lambda: (lambda conf: contextlib.nullcontext(), "fake.module"))
    monkeypatch.setattr(TILE, "make_inputs",
                        lambda spec, device=None: (real_torch.zeros(2, 2),
                                                   types.SimpleNamespace(w1=None, w2=None)))
    monkeypatch.setattr(TILE, "sample_topk_ids",
                        lambda routing, tokens, experts, k, seed=0, device=None:
                        real_torch.zeros((tokens, k), dtype=real_torch.long))
    _fake_vllm(monkeypatch)

    def timer(fn, **kwargs):
        state.seen.append(kwargs)
        result = state.timing_result
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result(**kwargs)
        return result

    monkeypatch.setattr(TILE.timing, "time_kernel", timer)
    return state


def run(pod, *extra, card: str = H200) -> int:
    return TILE.main(["--card", card, "--tokens", "8", "--tiles", "16,64",
                      "--out-dir", str(pod.out), *extra])


def cells(pod) -> list[dict]:
    return list(csv.DictReader(next(pod.out.rglob("cells.csv")).open()))


def report(pod) -> dict:
    return json.loads(next(pod.out.rglob("report.json")).read_text())


# --------------------------------------------------------------------------
# H1: an unplanned exception is ERROR, and a refusal sentence is REFUSED
# --------------------------------------------------------------------------

def test_an_unplanned_exception_is_error_and_not_a_claim_that_failed(
        monkeypatch, capsys):
    """`sys.exit(main())` had nothing above it, so a crash exited ONE.

    ONE is CLAIM_FAIL: a RESULT, in FINISHED_CODES, recorded by the session
    driver and never retried. An OOM partway through the sweep was therefore
    filed as this experiment's registered answer to "does a bigger tile buy
    anything". ERROR (4) sits outside FINISHED_CODES precisely so "the
    apparatus broke" can be told from "the claim did not hold".
    """
    def boom(argv=None):
        raise RuntimeError("torch OOM on the pod")

    monkeypatch.setattr(TILE, "_main", boom)
    assert TILE.main([]) == exit_codes.ERROR
    err = capsys.readouterr().err
    assert "torch OOM on the pod" in err, "the traceback was swallowed"
    assert "RuntimeError" in err
    assert exit_codes.ledger_state(exit_codes.ERROR) == "RETRY"
    assert exit_codes.ledger_state(exit_codes.CLAIM_FAIL) != "RETRY"

    # ...and the PASS branch: a run that reached a verdict keeps its own code.
    monkeypatch.setattr(TILE, "_main", lambda argv=None: exit_codes.CLAIM_FAIL)
    assert TILE.main([]) == exit_codes.CLAIM_FAIL
    monkeypatch.setattr(TILE, "_main", lambda argv=None: exit_codes.DONE)
    assert TILE.main([]) == exit_codes.DONE


def test_the_version_skew_refusal_is_refused_and_not_a_claim_that_failed(
        monkeypatch, capsys):
    """`find_override` raises `SystemExit(<str>)`, which exits ONE.

    The REAL refusal is executed here, not a stand-in: vLLM is not installed in
    this interpreter, so `find_override` walks its three candidate module names
    and raises. It fires before a single cell is timed, so it is REFUSED (2):
    free, nothing measured, retryable on a box that has the thing missing.
    """
    monkeypatch.setattr(TILE, "_main", lambda argv=None: TILE.find_override())
    assert TILE.main([]) == exit_codes.REFUSED
    err = capsys.readouterr().err
    assert "REFUSED" in err and "override_config" in err, (
        "a code with no sentence tells an operator nothing to fix")


def test_an_integer_systemexit_still_means_what_it_says(monkeypatch):
    """The FAIL branch of the string test: argparse exits `SystemExit(2)` and a
    handler that reclassified it would turn `--tiles hello` into a refusal it
    had already reported. Ints are re-raised untouched."""
    with pytest.raises(SystemExit) as caught:
        TILE.main(["--model", "not-a-model"])
    assert caught.value.code == 2


# --------------------------------------------------------------------------
# H2: the clock reference, and the column that was null without it
# --------------------------------------------------------------------------

def test_the_reference_clock_is_the_cards_own_and_says_where_it_came_from():
    """Resolved through `roofline.reference_clock`, not a fourth copy of the
    three-field rule, and the reason travels with the number."""
    ref = TILE.reference_clock_for(H200)
    assert ref.mhz == H200_REFERENCE_MHZ
    assert "gemm_clock" in ref.source and ref.card == H200

    missing = TILE.reference_clock_for(UNCALIBRATED)
    assert missing.mhz is None
    assert missing.card == UNCALIBRATED, (
        "the card must survive a failed lookup: a card with no calibration is a "
        "pod misconfiguration and no card at all is a laptop")


def test_a_sagging_card_now_reads_clock_level_ok_false_on_every_row(pod):
    """THE DEAD COLUMN, brought to life, and this is the FAIL branch of it.

    1000 MHz against a 1515 MHz reference is below `LEVEL_FRACTION`, so the
    real `clock_flags` returns False. Before the fix `reference_clock_mhz` was
    never passed, `clock_flags` returned None, and this cell was EMPTY on every
    row the sweep has ever written while the branch below it could not run.
    """
    pod.timing_result = lambda **kw: timing_at(1000.0, kw["reference_clock_mhz"])
    assert run(pod) == exit_codes.DONE

    assert [kw["reference_clock_mhz"] for kw in pod.seen] == \
        [H200_REFERENCE_MHZ] * 2, "the instrument was handed no reference"
    rows = cells(pod)
    assert len(rows) == 2
    assert [r["clock_level_ok"] for r in rows] == ["0", "0"]
    assert [r["reference_clock_mhz"] for r in rows] == ["1515", "1515"], (
        "the number LEVEL was scored AGAINST belongs on the row it scored; the "
        "tri-state alone cannot tell a row that passed from one with no "
        "reference at all, since both read empty")
    assert report(pod)["reference_clock_mhz"] == H200_REFERENCE_MHZ


def test_a_card_at_its_reference_clock_reads_true(pod, capsys):
    """The PASS branch of the same flag. A column that can only say 0 is as
    useless as one that can only say nothing, and the note printed under a
    flagged row must not appear under a level one."""
    pod.timing_result = lambda **kw: timing_at(H200_REFERENCE_MHZ,
                                               kw["reference_clock_mhz"])
    assert run(pod) == exit_codes.DONE
    assert [r["clock_level_ok"] for r in cells(pod)] == ["1", "1"]
    assert "scripted clock" not in capsys.readouterr().out, (
        "the clock note is printed only when the cell was flagged")


def test_the_flagged_row_prints_the_clock_note_that_could_not_fire_before(pod, capsys):
    """`if t.clock_level_ok is False` was unreachable code: with no reference
    the field is None, and None is not False. The branch is the operator's only
    live warning that the card sagged mid-sweep."""
    pod.timing_result = lambda **kw: timing_at(1000.0, kw["reference_clock_mhz"])
    run(pod)
    assert "scripted clock" in capsys.readouterr().out


def test_an_attached_card_with_no_calibration_refuses_before_it_compiles(
        pod, capsys):
    """REFUSE rather than pass None, the way `driver.refuse_unreferenced_clock`
    does. Past the CUDA check a card is attached by definition, and a sweep of
    rows that CANNOT report a clock problem costs the same rental as one that
    can. Nothing may be spent and nothing may be written."""
    assert run(pod, card=UNCALIBRATED) == exit_codes.REFUSED
    assert pod.seen == [], "a cell was timed after the refusal"
    assert list(pod.out.rglob("cells.csv")) == []
    out = capsys.readouterr().out
    assert "REFUSED" in out and "calibrate_hardware.py --publish" in out
    assert exit_codes.parse_result_lines(out) == [], (
        "a REFUSED log carrying RESULT lines lets the driver recompute DONE "
        "from them")
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(out)


# --------------------------------------------------------------------------
# H3: the instrument's own refusal is not one cell's error
# --------------------------------------------------------------------------

def test_an_instrument_refusal_leaves_the_sweep_instead_of_being_recorded(
        pod, capsys):
    """THE SECOND DOOR. `TimingRefused` subclasses RuntimeError, so the bare
    per-cell handler caught every refusal the instrument raises -- trials=0, a
    meaningless warmup, no CUDA and no injected fakes -- and filed it as this
    one cell's failure. Each is the same fact for every cell, so the sweep
    wrote a FAILED row per setting and then scored V1 and C1 over a page of
    nothing. It must leave the loop and exit REFUSED: nothing was measured."""
    pod.timing_result = timing.TimingRefused(
        "trials=0: a measurement needs at least one trial")
    assert run(pod) == exit_codes.REFUSED
    out, err = capsys.readouterr()
    assert "at least one trial" in err
    assert list(pod.out.rglob("cells.csv")) == [], (
        "a page of zeroed rows was written for a run that measured nothing")
    assert exit_codes.parse_result_lines(out) == []


def test_a_kernels_own_runtime_error_is_still_one_cells_error(pod):
    """The PASS branch of the same door, and the reason it is a subclass check
    and not a blanket re-raise: a kernel that launched badly IS a per-cell fact,
    the row records it, the sweep carries on, and V1 then fails for the right
    reason -- one setting cannot report a flat curve."""
    pod.timing_result = RuntimeError("CUDA error: an illegal memory access")
    assert run(pod) == exit_codes.INVALID
    rows = cells(pod)
    assert len(rows) == 2
    assert all("illegal memory access" in r["error"] for r in rows)
    verdicts = {g["name"].split()[0]: g["verdict"] for g in report(pod)["gates"]}
    assert verdicts["V1"] == exit_codes.FAIL
    assert verdicts["C1"] == exit_codes.UNKNOWN


# --------------------------------------------------------------------------
# LEVEL is two-sided since 03df2d4, and this consumer reads the side
# --------------------------------------------------------------------------

#: The H200 shape the fifteenth instance of the recurring defect was found on:
#: the bf16-GEMM reference the roof was measured at, and the clock the
#: committed calibration holds under memory load for 30 s.
H200_GEMM_REFERENCE_MHZ = 1515.0
H200_MEMORY_LOAD_MHZ = 1980.0
SAGGED_MHZ = 1400.0


def _kernel_timing_at(load_mhz, reference_mhz, *, drift_to=None):
    """A `KernelTiming` scored the way `time_kernel` scores one: verdicts from
    the real `clock_flags`, the side from the real `level_side`, and the
    reference on the record. Not hand-set booleans: a hand-set side would
    pass whatever the consumer did with it."""
    from moe.bench import timing
    end = load_mhz if drift_to is None else drift_to
    level, drift = timing.clock_flags(load_mhz, load_mhz, end, reference_mhz)
    return timing.KernelTiming(
        ms_p50=1.0, ms_p90=1.1, ms_min=0.9, ms_std=0.01, iters=100, trials=3,
        warmup_ms=300.0, l2_flush=True, sm_clock_load_mhz=load_mhz,
        sm_clock_start_mhz=load_mhz, sm_clock_end_mhz=end,
        clock_level_ok=level, clock_drift_ok=drift, samples=300,
        warmup_calls=10, flush_mb=256, clock_samples=9, clock_source="injected",
        clock_poll_ms=1.0, host_bound=False, host_enqueue_ms=0.01,
        clock_note="scripted clock",
        clock_level_side=timing.level_side(load_mhz, reference_mhz) or "",
        reference_clock_mhz=reference_mhz)


def test_the_exclusion_rule_is_the_drivers_low_or_drift_and_high_is_kept():
    """THE RULE, PINNED TO THE INSTRUMENT'S OWN CONSTANTS AND TO THE DRIVER'S.

    `moe.bench.driver` writes `throttled = drift failed or (level failed and
    side != HIGH)` on its rows (driver.py, the `throttled` assignment). This
    consumer has no such column and restates the rule; the two must agree on
    every cell of the truth table or a boosted tread is kept by one reader and
    dropped by the next, which is the shape of the defect.
    """
    from moe.bench import timing
    ex = TILE.clock_excluded
    # HIGH is not an exclusion, in any combination with a good drift.
    assert ex(False, timing.LEVEL_HIGH, True) is False
    assert ex(False, timing.LEVEL_HIGH, None) is False
    # LOW is, and so is a False that recorded no side (the one-sided era).
    assert ex(False, timing.LEVEL_LOW, True) is True
    assert ex(False, "", True) is True
    # DRIFT excludes whatever LEVEL said, HIGH included.
    assert ex(True, "", False) is True
    assert ex(False, timing.LEVEL_HIGH, False) is True
    assert ex(None, "", False) is True
    # Not determined is not an exclusion: one has to be positively established.
    assert ex(None, "", None) is False
    assert ex(True, "", True) is False
    assert ex(True, "", None) is False
    # The driver's rule, evaluated over the same table.
    for level in (True, False, None):
        for side in ("", timing.LEVEL_LOW, timing.LEVEL_HIGH):
            for drift in (True, False, None):
                driver_rule = (drift is False
                               or (level is False and side != timing.LEVEL_HIGH))
                assert ex(level, side, drift) is driver_rule, (level, side, drift)


def test_a_boosted_record_reads_high_and_a_sagged_one_reads_low():
    """1980 against 1515 is 1.31x, above `LEVEL_HIGH_FRACTION`; 1400 against
    1515 is 0.92x, below `LEVEL_FRACTION`. Both fail LEVEL, and the side is
    the only thing that tells them apart."""
    from moe.bench import timing
    high = _kernel_timing_at(H200_MEMORY_LOAD_MHZ, H200_GEMM_REFERENCE_MHZ)
    low = _kernel_timing_at(SAGGED_MHZ, H200_GEMM_REFERENCE_MHZ)
    assert high.clock_level_ok is False and low.clock_level_ok is False
    assert TILE.clock_side_of(high) == timing.LEVEL_HIGH
    assert TILE.clock_side_of(low) == timing.LEVEL_LOW
    assert TILE.clock_excluded(high.clock_level_ok, TILE.clock_side_of(high),
                                 high.clock_drift_ok) is False, "HIGH is kept"
    assert TILE.clock_excluded(low.clock_level_ok, TILE.clock_side_of(low),
                                 low.clock_drift_ok) is True, "LOW is excluded"
    # A record without the field (every fake before 2026-09-03) gets its side
    # derived from its own numbers, the way driver.py derives it.
    import dataclasses
    bare = dataclasses.replace(high, clock_level_side="")
    assert TILE.clock_side_of(bare) == timing.LEVEL_HIGH
    # And one with neither answers "", which the rule reads as below.
    blind = dataclasses.replace(bare, reference_clock_mhz=None)
    assert TILE.clock_side_of(blind) == ""


def test_only_the_rule_and_the_summary_compare_the_level_verdict_bare():
    """THE SECOND CALL SITE, GUARDED. A `clock_level_ok is False` outside the
    rule and the counting block is a reader that has not learned the side, and
    that is how the fifteenth instance happened: one producer fixed, thirteen
    consumers left on the old meaning."""
    import ast
    tree = ast.parse((ROOT / "scripts" / "tile_sweep.py").read_text())
    readers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Code, not prose: a docstring that names the old test is history.
            body = "\n".join(ast.unparse(s) for s in node.body
                             if not (isinstance(s, ast.Expr)
                                     and isinstance(s.value, ast.Constant)))
            if "clock_level_ok is False" in body:
                readers.add(node.name)
    allowed = {"clock_excluded"}
    assert readers <= allowed, (
        f"{sorted(readers - allowed)} test the LEVEL verdict "
        "without its side; route them through clock_excluded")


def test_a_boosted_card_is_kept_and_named_so_on_every_row(pod, capsys):
    """THE PLANTED HIGH RUN: the H200's memory-load clock against its GEMM
    reference, which is every decode cell this sweep times. The rows carry
    side "high", the report's clock-state block counts them as HIGH and as
    zero excluded-shaped, and the operator's line says kept."""
    pod.timing_result = lambda **kw: timing_at(H200_MEMORY_LOAD_MHZ,
                                               kw["reference_clock_mhz"])
    assert run(pod) == exit_codes.DONE
    rows = cells(pod)
    assert [r["clock_level_ok"] for r in rows] == ["0", "0"]
    assert [r["clock_level_side"] for r in rows] == ["high", "high"]
    state = report(pod)["clock_state"]
    assert state["high"] == 2 and state["low"] == 0
    assert state["excluded_shaped"] == 0
    out = capsys.readouterr().out
    assert out.count("kept (LEVEL high is not an exclusion): scripted clock") == 2
    assert "2 HIGH (boosted above the band, kept" in out


def test_a_sagging_card_is_excluded_shaped_and_says_low(pod, capsys):
    """THE PLANTED LOW RUN, the FAIL branch of the same rule: side "low", both
    rows excluded-shaped, and the note printed without the word kept."""
    pod.timing_result = lambda **kw: timing_at(SAGGED_MHZ, kw["reference_clock_mhz"])
    assert run(pod) == exit_codes.DONE
    rows = cells(pod)
    assert [r["clock_level_side"] for r in rows] == ["low", "low"]
    state = report(pod)["clock_state"]
    assert state["low"] == 2 and state["high"] == 0
    assert state["excluded_shaped"] == 2
    out = capsys.readouterr().out
    assert "kept (LEVEL high" not in out
    assert out.count("^ scripted clock") == 2
    assert "2 LOW (below the band, excluded-shaped)" in out


def test_a_level_card_carries_an_empty_side(pod):
    pod.timing_result = lambda **kw: timing_at(H200_REFERENCE_MHZ,
                                               kw["reference_clock_mhz"])
    assert run(pod) == exit_codes.DONE
    assert [r["clock_level_side"] for r in cells(pod)] == ["", ""]
    state = report(pod)["clock_state"]
    assert state["level"] == 2 and state["excluded_shaped"] == 0


def test_clock_state_reads_the_csv_cells_it_wrote():
    """The block reads the string cells `timing_columns` wrote, so a replay of
    cells.csv and the live run count the same rows the same way, and an empty
    flag is NOT DETERMINED rather than a pass or a fail."""
    rows = [
        {"clock_level_ok": "0", "clock_level_side": "high", "clock_drift_ok": "1"},
        {"clock_level_ok": "0", "clock_level_side": "low", "clock_drift_ok": "1"},
        {"clock_level_ok": "0", "clock_level_side": "", "clock_drift_ok": "1"},
        {"clock_level_ok": "1", "clock_level_side": "", "clock_drift_ok": "0"},
        {"clock_level_ok": "", "clock_level_side": "", "clock_drift_ok": ""},
        {"clock_level_ok": "0", "clock_level_side": "high", "clock_drift_ok": "1",
         "error": "OOM"},
    ]
    state = TILE.clock_state(rows)
    assert state["timed"] == 5
    assert state["high"] == 1
    assert state["low"] == 2, "a False with no side is the one-sided era's below"
    assert state["level"] == 1 and state["drift"] == 1 and state["unknown"] == 1
    assert state["excluded_shaped"] == 3
