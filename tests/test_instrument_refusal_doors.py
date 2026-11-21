"""The instrument's own refusal must leave a measuring loop, not be filed as a cell.

`timing.TimingRefused` subclasses RuntimeError. Every per-cell timing loop in
scripts/ wraps `timing.time_kernel` in an `except Exception` so a kernel that
launches badly is ONE cell's fact and the sweep carries on. That handler also
caught the instrument refusing outright (no CUDA and no fakes, trials=0, a
warmup too short to settle), which is the same fact for every cell: the arm
then walked its whole grid writing zeroed rows and exited DONE. Reproduced on
the driver path on 2026-09-03 (3 cells, 3 error rows, returned normally), then
found open at the call site in three more scripts and at the top-level guard of
two others.

The door is `except timing.TimingRefused: raise`, placed BEFORE the swallow.
Placing it after would be a no-op, and moving the call outside the try would
lose the per-cell handling that is correct for a kernel's own error. So what
this file pins is the ORDER at the call site, in the source, for every script
that times a cell, plus the behaviour of the two guards that sit above a
sibling's loop rather than owning one.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moe.bench import exit_codes, timing  # noqa: E402

PER_CELL = ["block_m_crossing_sweep", "bm128_roofline", "occupancy_vs_swizzle",
            "bn_decomposition", "bm128_depth"]


def _calls_time_kernel(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "time_kernel" for n in ast.walk(node))


def _innermost_try_around_time_kernel(tree: ast.AST) -> list[ast.Try]:
    """The Try whose OWN body (not a nested Try's) holds the time_kernel call."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = ast.Module(body=node.body, type_ignores=[])
        if not _calls_time_kernel(body):
            continue
        nested = [n for n in ast.walk(body) if isinstance(n, ast.Try) and n is not node
                  and _calls_time_kernel(ast.Module(body=n.body, type_ignores=[]))]
        if not nested:
            found.append(node)
    return found


def _handler_catches(h: ast.ExceptHandler, dotted: str) -> bool:
    t = h.type
    if t is None:
        return False
    if isinstance(t, ast.Attribute):
        return f"{getattr(t.value, 'id', '?')}.{t.attr}" == dotted
    if isinstance(t, ast.Name):
        return t.id == dotted
    return False


@pytest.mark.parametrize("script", PER_CELL)
def test_the_refusal_door_is_at_the_call_site_and_before_the_swallow(script):
    """For every script that times a cell: the innermost try around
    `timing.time_kernel` re-raises `timing.TimingRefused` FIRST, and only then
    catches `Exception`. Checked in the source so that reordering the handlers,
    or adding a broad handler above the door, fails here without a GPU."""
    src = (ROOT / "scripts" / f"{script}.py").read_text()
    tries = _innermost_try_around_time_kernel(ast.parse(src))
    assert tries, f"{script}: no try wraps a timing.time_kernel call"
    for node in tries:
        kinds = [h for h in node.handlers]
        door = [i for i, h in enumerate(kinds) if _handler_catches(h, "timing.TimingRefused")]
        swallow = [i for i, h in enumerate(kinds) if _handler_catches(h, "Exception")]
        assert door, (f"{script}:{node.lineno}: the try around time_kernel has no "
                      "`except timing.TimingRefused` door")
        assert swallow, (f"{script}:{node.lineno}: no per-cell `except Exception`; a "
                         "kernel's own error must still be one cell's fact")
        assert door[0] < swallow[0], (
            f"{script}:{node.lineno}: the TimingRefused door is AFTER the Exception "
            "handler, which makes it unreachable; the instrument's refusal is "
            "swallowed as one cell's error and the arm exits DONE over zeroes")
        body = kinds[door[0]].body
        assert len(body) == 1 and isinstance(body[0], ast.Raise) and body[0].exc is None, (
            f"{script}:{node.lineno}: the door must be a bare `raise`, so the "
            "refusal reaches the top-level guard with its remedy intact")


# --------------------------------------------------------------------------
# The two scripts that do not own a timing loop but sit above a sibling's.
# --------------------------------------------------------------------------

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


def _raising(exc):
    def _main(argv=None):
        raise exc
    return _main


def test_tile_cap_test_reports_an_instrument_refusal_as_refused_not_as_a_crash(monkeypatch):
    """tile_cap_test times through SWEEP.run_sweep, which now re-raises the
    refusal. Its top-level guard caught it under `except Exception` and printed
    a traceback with exit ERROR (4), a retryable crash, when it is REFUSED (2),
    a precondition with the remedy in its message. Both branches, both codes."""
    m = _load("tile_cap_test")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        monkeypatch.setattr(m, "_main", _raising(timing.TimingRefused("trials=0: needs one trial")))
        assert m.main([]) == exit_codes.REFUSED
        monkeypatch.setattr(m, "_main", _raising(RuntimeError("a real crash")))
        assert m.main([]) == exit_codes.ERROR
    text = err.getvalue()
    assert text.startswith("REFUSED:"), "the refusal must print as one, ahead of any traceback"
    assert "Traceback" in text, "the crash must still print its traceback"


def test_tuned_vs_fallback_has_the_guard_every_sibling_grew(monkeypatch):
    """It was the one measuring script with `raise SystemExit(main())` bare.
    An unplanned exception exited the interpreter's 1 = CLAIM_FAIL, a RESULT
    the driver latches and never retries; a `SystemExit("sentence")` refusal
    exited 1 too. Now: crash -> ERROR, string refusal -> REFUSED, argparse's
    own int passes through untouched."""
    m = _load("tuned_vs_fallback")
    with contextlib.redirect_stderr(io.StringIO()):
        monkeypatch.setattr(m, "_main", _raising(RuntimeError("torch OOM on the pod")))
        assert m.main([]) == exit_codes.ERROR
        monkeypatch.setattr(m, "_main", _raising(SystemExit("no calibration for this device")))
        assert m.main([]) == exit_codes.REFUSED
        monkeypatch.setattr(m, "_main", _raising(SystemExit(2)))
        with pytest.raises(SystemExit) as caught:
            m.main([])
        assert caught.value.code == 2
