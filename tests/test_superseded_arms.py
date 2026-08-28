"""A superseded result arm must not be counted twice.

MEASURED, 2026-08-28. `results/published/` holds both
`2026-08-26-nvidia_h200-full-three-way` and its `-recalibrated` twin. All 17,640
ms_p50 values are IDENTICAL between them: the recalibration recomputed ceiling
columns (12,034 rows have a different achieved_bw_gbps) and touched no timing.

So any analysis pointed at `published/*/merged.csv` weights those rows double.
It changed a published figure: mixtral's torch_grouped_mm_up crossing came out
0.63x of prediction from one file set and 1.46x from the other, and nothing in
either run said which was right.

A marker file in the superseded directory makes it self-describing, rather than
a rule every caller has to remember. Skipping is ANNOUNCED, never silent: a
dropped input that nobody sees is how the double count happened.
"""
from __future__ import annotations

from pathlib import Path

from moe.bench.published import SUPERSEDED_MARKER, filter_superseded, is_superseded


def test_a_marked_directory_is_superseded(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "merged.csv").write_text("x")
    assert not is_superseded(d / "merged.csv")
    (d / SUPERSEDED_MARKER).write_text("replaced by the recalibrated arm")
    assert is_superseded(d / "merged.csv")


def test_filtering_returns_the_kept_paths_and_the_dropped_ones(tmp_path):
    keep, drop = tmp_path / "keep", tmp_path / "drop"
    for d in (keep, drop):
        d.mkdir()
        (d / "merged.csv").write_text("x")
    (drop / SUPERSEDED_MARKER).write_text("superseded")

    kept, dropped = filter_superseded([keep / "merged.csv", drop / "merged.csv"])
    assert kept == [keep / "merged.csv"]
    assert dropped == [drop / "merged.csv"]


def test_the_reason_travels_with_the_marker(tmp_path):
    """So a report can say WHY it dropped something, not just that it did."""
    d = tmp_path / "arm"
    d.mkdir()
    (d / "merged.csv").write_text("x")
    (d / SUPERSEDED_MARKER).write_text("superseded by: the-recalibrated-arm\n")
    from moe.bench.published import superseded_reason
    assert "recalibrated" in superseded_reason(d / "merged.csv")


def test_a_path_that_does_not_exist_is_not_superseded(tmp_path):
    assert not is_superseded(tmp_path / "nope" / "merged.csv")


def test_the_real_repo_marks_exactly_one_arm():
    """The state this exists for. If a second arm is ever superseded, this test
    is the place to say so deliberately."""
    pub = Path(__file__).resolve().parent.parent / "results" / "published"
    if not pub.exists():
        return
    marked = sorted(p.parent.name for p in pub.glob(f"*/{SUPERSEDED_MARKER}"))
    assert marked == ["2026-08-26-nvidia_h200-full-three-way"], marked
