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
#: reference asserts it was handed THE reference, not merely something. It was
#: 1515.0 until the 2026-09-09 recalibration (ab61e55) measured 1485 under a
#: 700 W cap and left this literal behind, red in two tests.
H200_REFERENCE_MHZ = 1485.0
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

    1000 MHz against the card's 1485 MHz reference is below `LEVEL_FRACTION`,
    so the real `clock_flags` returns False. Before the fix
    `reference_clock_mhz` was never passed, `clock_flags` returned None, and
    this cell was EMPTY on every row the sweep has ever written while the
    branch below it could not run.
    """
    pod.timing_result = lambda **kw: timing_at(1000.0, kw["reference_clock_mhz"])
    assert run(pod) == exit_codes.DONE

    assert [kw["reference_clock_mhz"] for kw in pod.seen] == \
        [H200_REFERENCE_MHZ] * 2, "the instrument was handed no reference"
    rows = cells(pod)
    assert len(rows) == 2
    assert [r["clock_level_ok"] for r in rows] == ["0", "0"]
    assert [r["reference_clock_mhz"] for r in rows] == ["1485", "1485"], (
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

#: A PLANTED WORLD, not a card. These three numbers are the shape the
#: fifteenth instance of the recurring defect was found on (a memory-shaped
#: cell boosting above the roof's clock, a hungry one sagging below it), and
#: the tests below pass all three sides of every ratio, so what they test is
#: the consumer's arithmetic and not any card's figures.
#:
#: THE REFERENCE IS DELIBERATELY A ROUND NUMBER NO CARD PUBLISHES. It read
#: 1515.0 until 2026-09-09, under a comment calling it "the bf16-GEMM
#: reference the roof was measured at": that was the H200's committed
#: calibration until ab61e55 remeasured the card at 1485 MHz under the 700 W
#: cap on 2026-09-09, so the constant was the superseded live number wearing
#: the name of the current one, and the file that carried it also defines
#: `H200_REFERENCE_MHZ = 1485.0` for the card's real figure. A planted world
#: gets a planted number; a test that needs the card's own clock reads the
#: committed calibration.
PLANTED_REFERENCE_MHZ = 1500.0
#: Well above `PLANTED_REFERENCE_MHZ * timing.LEVEL_HIGH_FRACTION`, the state
#: of every memory-shaped tread the H200 gaps session measured (1950-1980).
H200_MEMORY_LOAD_MHZ = 1980.0
#: Well below `PLANTED_REFERENCE_MHZ * timing.LEVEL_FRACTION`.
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


def test_the_exclusion_rule_is_drift_alone_and_both_level_sides_are_kept():
    """THE RULE AS IT STANDS SINCE 2026-09-09: DRIFT excludes, LEVEL records.

    Until then this asserted "LOW or DRIFT excludes, HIGH is kept". The
    750-cell census of the H200 gaps session showed the LOW side is the steady
    operating point of a hungry tile under the 700 W cap (BLOCK_M=128 at
    BLOCK_N=64 held 1380-1410 MHz in every rep of every arm, BLOCK_M=64 at
    GROUP_SIZE_M=1 1358), so excluding it excluded a tile rather than a
    defect, and removed both of this study's primary tiles from measurability
    on the card.

    `moe.bench.driver`'s `throttled` assignment is the twin of this rule and
    moves in the same commit; the truth table below is written out here rather
    than imported so this file states the rule instead of quoting whatever the
    driver currently does.
    """
    from moe.bench import timing
    ex = TILE.clock_excluded
    # Neither side is an exclusion when the clock held still.
    assert ex(False, timing.LEVEL_HIGH, True) is False
    assert ex(False, timing.LEVEL_HIGH, None) is False
    assert ex(False, timing.LEVEL_LOW, True) is False
    assert ex(False, timing.LEVEL_LOW, None) is False
    # A False that recorded no side is the one-sided era's row: still kept.
    assert ex(False, "", True) is False
    # DRIFT excludes whatever LEVEL said, on either side.
    assert ex(True, "", False) is True
    assert ex(False, timing.LEVEL_HIGH, False) is True
    assert ex(False, timing.LEVEL_LOW, False) is True
    assert ex(None, "", False) is True
    # Not determined is not an exclusion: one has to be positively established.
    assert ex(None, "", None) is False
    assert ex(True, "", True) is False
    assert ex(True, "", None) is False
    # The whole table: DRIFT and nothing else.
    for level in (True, False, None):
        for side in ("", timing.LEVEL_LOW, timing.LEVEL_HIGH):
            for drift in (True, False, None):
                assert ex(level, side, drift) is (drift is False), (
                    level, side, drift)


def test_a_boosted_record_reads_high_and_a_sagged_one_reads_low():
    """1980 against 1515 is 1.31x, above `LEVEL_HIGH_FRACTION`; 1400 against
    1515 is 0.92x, below `LEVEL_FRACTION`. Both fail LEVEL, and the side is
    the only thing that tells them apart."""
    from moe.bench import timing
    high = _kernel_timing_at(H200_MEMORY_LOAD_MHZ, PLANTED_REFERENCE_MHZ)
    low = _kernel_timing_at(SAGGED_MHZ, PLANTED_REFERENCE_MHZ)
    assert high.clock_level_ok is False and low.clock_level_ok is False
    assert TILE.clock_side_of(high) == timing.LEVEL_HIGH
    assert TILE.clock_side_of(low) == timing.LEVEL_LOW
    assert TILE.clock_excluded(high.clock_level_ok, TILE.clock_side_of(high),
                                 high.clock_drift_ok) is False, "HIGH is kept"
    assert TILE.clock_excluded(low.clock_level_ok, TILE.clock_side_of(low),
                                 low.clock_drift_ok) is False, (
        "since 2026-09-09 a steady LOW is kept and its side recorded")
    # A record without the field (every fake before 2026-09-03) gets its side
    # derived from its own numbers, the way driver.py derives it.
    import dataclasses
    bare = dataclasses.replace(high, clock_level_side="")
    assert TILE.clock_side_of(bare) == timing.LEVEL_HIGH
    # And one with neither answers "", which is "no side recorded".
    blind = dataclasses.replace(bare, reference_clock_mhz=None)
    assert TILE.clock_side_of(blind) == ""


def test_only_the_rule_and_the_summary_compare_the_level_verdict_bare():
    """THE SECOND CALL SITE, GUARDED. A `clock_level_ok is False` outside the
    rule and the counting block is a reader that has not learned the side, and
    that is how the fifteenth instance happened: one producer fixed, thirteen
    consumers left on the old meaning.

    A NESTED HELPER IS ITS OUTERMOST FUNCTION, since 2026-09-09. The walk
    attributed a def to its own name whatever it was nested in, so a reader
    written as a closure inside a disallowed function passed under a name that
    was not on the list, and a helper factored out of an allowed one failed
    while doing exactly what the allowed one did. Attributing by ancestor
    makes the list about the block the code lives in, which is what the rule
    is about."""
    import ast
    tree = ast.parse((ROOT / "scripts" / "tile_sweep.py").read_text())
    owner = {}
    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            for node in ast.walk(top):
                if isinstance(node, ast.FunctionDef):
                    owner[node] = top.name
    readers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Code, not prose: a docstring that names the old test is history.
            body = "\n".join(ast.unparse(s) for s in node.body
                             if not (isinstance(s, ast.Expr)
                                     and isinstance(s.value, ast.Constant)))
            if "clock_level_ok is False" in body:
                readers.add(owner.get(node, node.name))
    allowed = {"clock_excluded"}
    assert readers <= allowed, (
        f"{sorted(readers - allowed)} test the LEVEL verdict "
        "without its side; route them through clock_excluded")


def test_a_boosted_card_is_kept_and_named_so_on_every_row(pod, capsys):
    """THE PLANTED HIGH RUN: the H200's memory-load clock against its GEMM
    reference, which is every decode cell this sweep times. The rows carry
    side "high", the report's clock-state block counts them as HIGH and as
    zero excluded-shaped, and the operator's line says kept, recorded."""
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
    assert out.count("kept (LEVEL high is recorded, not excluded): "
                     "scripted clock") == 2
    assert "2 steady HIGH (kept, side recorded)" in out


def test_a_sagging_card_is_kept_and_says_low(pod, capsys):
    """THE PLANTED LOW RUN: side "low" on both rows, both KEPT since
    2026-09-09, and the operator's line naming the side as recorded. Until
    then this run counted two excluded-shaped rows, which on the H200 is what
    a BLOCK_M=128 tile at its own operating point under the power cap looks
    like."""
    pod.timing_result = lambda **kw: timing_at(SAGGED_MHZ, kw["reference_clock_mhz"])
    assert run(pod) == exit_codes.DONE
    rows = cells(pod)
    assert [r["clock_level_side"] for r in rows] == ["low", "low"]
    state = report(pod)["clock_state"]
    assert state["low"] == 2 and state["high"] == 0
    assert state["excluded_shaped"] == 0
    out = capsys.readouterr().out
    assert out.count("kept (LEVEL low is recorded, not excluded): "
                     "scripted clock") == 2
    assert "2 steady LOW (kept, side recorded)" in out


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
    flag is NOT DETERMINED rather than a pass or a fail.

    THE SIXTH ROW IS THE OVERLAPPING ONE: LEVEL false on the HIGH side AND
    DRIFT false. No planted world here had one until 2026-09-09, so the
    counters could filter on the LEVEL verdict alone, count that row as a kept
    HIGH and again as a DRIFT, and stay green while the printed line named it
    kept and excluded at once."""
    rows = [
        {"clock_level_ok": "0", "clock_level_side": "high", "clock_drift_ok": "1"},
        {"clock_level_ok": "0", "clock_level_side": "low", "clock_drift_ok": "1"},
        {"clock_level_ok": "0", "clock_level_side": "", "clock_drift_ok": "1"},
        {"clock_level_ok": "1", "clock_level_side": "", "clock_drift_ok": "0"},
        {"clock_level_ok": "", "clock_level_side": "", "clock_drift_ok": ""},
        {"clock_level_ok": "0", "clock_level_side": "high", "clock_drift_ok": "0"},
        {"clock_level_ok": "0", "clock_level_side": "high", "clock_drift_ok": "1",
         "error": "OOM"},
    ]
    state = TILE.clock_state(rows)
    assert state["timed"] == 6
    assert state["high"] == 1
    assert state["low"] == 2, "a False with no side is the one-sided era's below"
    # The level row of this world is the DRIFTING one, so the steady level
    # count is ZERO and that row appears once, in `drift_level`. It was
    # counted as both "1 level" and "1 DRIFT" until 2026-09-09.
    assert state["level"] == 0 and state["drift"] == 2 and state["unknown"] == 1
    assert (state["level"] + state["low"] + state["high"] + state["unknown"]
            + state["drift"]) == state["timed"]
    assert state["drift_high"] == 1 and state["drift_level"] == 1
    assert state["drift_low"] == 0 and state["drift_unknown"] == 0
    # Only the drifted rows: since 2026-09-09 neither LEVEL side excludes.
    assert state["excluded_shaped"] == 2
    said = "\n".join(TILE.clock_state_lines(state))
    assert "1 steady HIGH (kept, side recorded)" in said
    assert "2 DRIFT failed (excluded, and not in the three counts before it)" in said
    assert "the 2 drifted rows by side: 1 level, 0 LOW, 1 HIGH" in said
    # THE PAGE MUST NOT NAME A COLUMN THIS SCRIPT DOES NOT WRITE.
    assert "roof_at_cell_clock" not in said


def test_the_clock_state_docstring_carries_the_rule_it_applies():
    """THE SECOND CALL SITE IN ONE FUNCTION. `clock_state`'s docstring said
    "`low` and `drift` are the excluded-shaped states `clock_excluded` names;
    `high` is kept and counted apart" twenty lines above its own `rule` string,
    which since 2026-09-09 says "DRIFT excludes; BOTH LEVEL sides are kept".
    The four sibling scripts had this paragraph rewritten that day and
    tile_sweep was the one that was missed."""
    doc = TILE.clock_state.__doc__
    assert "`drift` IS THE ONE EXCLUDED STATE" in doc
    assert TILE.clock_state([])["rule"].startswith(
        "DRIFT excludes; BOTH LEVEL sides are kept")
    # The retired wording survives only as history, and history is dated. A
    # sentence that states it in the present tense is the defect returning.
    retired = "`high` is kept and counted apart"
    if retired in doc:
        assert "Until 2026-09-09 this docstring said" in doc[:doc.index(retired)]


def test_the_new_clock_columns_are_in_the_header_and_the_row():
    """R3's evidence columns. `time_kernel` computed the first and last
    under-load sample and this writer dropped them, so a row that failed DRIFT
    could not say which way its clock went."""
    t = timing_at(H200_MEMORY_LOAD_MHZ, H200_REFERENCE_MHZ)
    row = TILE.timing_columns(t)
    for column in ("sm_clock_start_mhz", "sm_clock_end_mhz",
                   "clock_samples_mhz", "power_w"):
        assert column in TILE.TIMING_CSV_COLUMNS, column
        assert column in TILE.CSV_COLUMNS, column
        assert column in row, column
    assert row["sm_clock_start_mhz"] == f"{H200_MEMORY_LOAD_MHZ:.0f}"
    assert row["sm_clock_end_mhz"] == f"{H200_MEMORY_LOAD_MHZ:.0f}"
    # ALL FOUR CLOCK COLUMNS GO THROUGH ONE FORMATTER. `sm_clock_load_mhz`
    # carried an inline copy of `_mhz`'s body until 2026-09-09, which is one
    # column formatted by hand beside three through the helper.
    assert row["sm_clock_load_mhz"] == TILE._mhz(t.sm_clock_load_mhz)
    assert TILE._mhz(None) == "", "empty is NOT DETERMINED, never 0 MHz"
    # And the docstring lists what it writes: it named `clock_level_side` and
    # not the four evidence columns until 2026-09-09.
    for column in ("sm_clock_start_mhz", "sm_clock_end_mhz",
                   "clock_samples_mhz", "power_w"):
        assert column in TILE.timing_columns.__doc__, column
    # An instrument without the sample list or the draw writes them EMPTY,
    # which is NOT DETERMINED and never zero.
    assert row["clock_samples_mhz"] == "" and row["power_w"] == ""
    assert set(row) == set(TILE.TIMING_CSV_COLUMNS)


def test_a_row_with_no_clock_at_all_writes_the_new_columns_empty():
    class Blind:
        instrument, warmup_ms, iters, trials = "queue-deep/test", 300.0, 10, 3
        sm_clock_load_mhz = None
        clock_level_ok = clock_drift_ok = host_bound = None
        l2_flush = True

    row = TILE.timing_columns(Blind())
    assert row["sm_clock_load_mhz"] == ""
    assert row["sm_clock_start_mhz"] == "" and row["sm_clock_end_mhz"] == ""
    assert row["power_w"] == ""
