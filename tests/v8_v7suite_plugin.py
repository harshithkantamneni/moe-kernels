"""A pytest plugin for tests/test_timing_model_v8.py's V7-SUITE check (review 1, MAJOR 9): v7's own test file
(tests/test_timing_model_v7.py) run in a process where v8 is imported and EVERY test runs inside `v8.installed()`
(every binding of every swapped function is v8's copy), so each of v7's checks is a check of v8's swaps: a copy that
did not dispatch to the frozen original on v7's integer treads would move a v7 number and fail v7's own test.

    python -m pytest -q -p v8_v7suite_plugin tests/test_timing_model_v7.py     # tests/ and scripts/ on PYTHONPATH

At the end of the session it writes $V8_V7SUITE_REPORT (JSON): the tests wrapped, the swaps' Tread-path calls (none
expected: v7 never hands a Tread) and whether every binding was restored."""
from __future__ import annotations

import json
import os

import pytest

_SEEN = {"wrapped": 0, "depth_ok": 0}


@pytest.fixture(autouse=True)
def _v8_swaps_installed():
    import r3_timing_model_v8 as V8
    with V8.installed():
        _SEEN["wrapped"] += 1
        _SEEN["depth_ok"] += V8._DEPTH[0] == 1
        yield


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("V8_V7SUITE_REPORT")
    if not out:
        return
    import sys
    V8 = sys.modules.get("r3_timing_model_v8")
    restored = None
    if V8 is not None and V8._PLAN:
        restored = all((h.get(a) if isinstance(h, dict) else vars(h).get(a)) is f for h, a, f, _c in V8._PLAN)
    with open(out, "w") as fh:
        json.dump({**_SEEN, "exitstatus": int(exitstatus), "plan_entries": len(V8._PLAN) if V8 else 0,
                   "tread_calls": dict(V8.STATS) if V8 else {}, "restored": restored}, fh)
