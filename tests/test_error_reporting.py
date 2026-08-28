"""An exception with no message must still say what went wrong.

MEASURED, H200 2026-08-27. The first fp8 sweep failed 147 cells and printed
`[warn] mixtral-8x7b/T128/fp8_e4m3/uniform/s0 vllm_fused_experts: ` with nothing
after the colon, 147 times. Formatting with `{e}` renders the empty string for
any exception carrying no args, and vLLM's fused_moe validates its arguments
with bare `assert` statements, which are exactly that.

The console is the only surface a person watches while a GPU is billing. It said
a thing failed 147 times and refused to say what, which turned a two-minute fix
into a re-run.
"""
from __future__ import annotations

import pytest

from moe.bench.driver import describe_exception


def test_a_bare_assert_still_identifies_itself():
    """The exact shape that produced 147 blank lines."""
    try:
        assert False  # noqa: B011  the bare form IS the case under test
    except AssertionError as e:
        got = describe_exception(e)
    assert got, "a message-less exception rendered as the empty string"
    assert "AssertionError" in got


def test_a_message_is_kept_and_typed():
    got = describe_exception(ValueError("w1_scale has shape (8,) expected (8,1)"))
    assert "ValueError" in got
    assert "w1_scale has shape (8,) expected (8,1)" in got


def test_it_is_one_line_so_it_cannot_flood_the_console():
    """A sweep prints one of these per failing cell. A multi-line render would
    bury the sweep's own progress under thousands of lines."""
    got = describe_exception(ValueError("line one\nline two\nline three"))
    assert "\n" not in got
    assert "line one" in got and "line two" in got


def test_a_very_long_message_is_bounded():
    got = describe_exception(RuntimeError("x" * 5000))
    assert len(got) <= 300


@pytest.mark.parametrize("exc", [
    AssertionError(),
    RuntimeError(),
    KeyError(),
    ValueError(),
])
def test_no_exception_type_renders_blank(exc):
    """Whatever the framework raises, the operator learns something."""
    assert describe_exception(exc).strip()
