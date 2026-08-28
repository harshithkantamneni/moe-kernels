"""The report must separate implementations, not rely on the caller to.

`--impl` reads as a convenience filter, but without it the grouping key was
(model, dtype) and rows from different implementations were averaged into one
median. Measured on the published H200 sweep at mixtral/T=512, the six impls in
a merged CSV span 0.439 ms (torch_grouped_mm_down, one stage) to 7.337 ms
(__pipeline__, the all-reference whole layer): a 16.7x spread. A median across
those describes nothing.

Same rule the rest of the project already follows: `impl` names exactly one
measured scope, so anything grouping rows has to key on it.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _csv(path: Path, rows: list[dict]):
    from moe.bench.schema import COLUMNS
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            base = {c: "" for c in COLUMNS}
            base.update(r)
            w.writerow(base)


def _row(impl, t, ms, model="mixtral-8x7b", dtype="bf16"):
    return {"impl": impl, "num_tokens": t, "ms_p50": f"{ms}", "model": model,
            "dtype": dtype, "routing_kind": "uniform",
            "correctness_passed": "True", "throttled": "False"}


def test_two_implementations_are_reported_separately(tmp_path):
    """A fast one-stage span and a slow whole-layer one, same model and dtype.
    Without impl in the key their medians merge into a curve neither of them
    has."""
    rows = []
    for t in (128, 256, 512, 1024, 2048, 4096):
        rows.append(_row("torch_grouped_mm_up", t, max(0.4, 0.4 * t / 512)))
        rows.append(_row("__pipeline__", t, max(7.0, 7.0 * t / 512)))
    p = tmp_path / "mixed.csv"
    _csv(p, rows)

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "crossing_report.py"), str(p),
         "--ridge", "160.3"],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert "torch_grouped_mm_up" in out.stdout
    assert "__pipeline__" in out.stdout
    # Two sections, not one blended one.
    assert out.stdout.count("=== ") >= 2, out.stdout


def test_the_impl_flag_still_narrows_the_output(tmp_path):
    rows = []
    for t in (128, 256, 512, 1024, 2048, 4096):
        rows.append(_row("torch_grouped_mm_up", t, max(0.4, 0.4 * t / 512)))
        rows.append(_row("__pipeline__", t, max(7.0, 7.0 * t / 512)))
    p = tmp_path / "mixed.csv"
    _csv(p, rows)

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "crossing_report.py"), str(p),
         "--ridge", "160.3", "--impl", "torch_grouped_mm_up"],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert "torch_grouped_mm_up" in out.stdout
    assert "__pipeline__" not in out.stdout
