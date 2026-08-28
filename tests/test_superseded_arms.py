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


def test_the_real_repo_marks_exactly_two_arms():
    """The state this exists for, and the place a new superseding gets declared
    on purpose rather than noticed later.

    Two now, and they are different KINDS:

      full-three-way        WHOLE arm, replaced by its recalibrated twin
      fp8-three-kernel      PARTIAL, only its two torch spans, replaced by
                            fp8-refixed
    """
    from moe.bench.published import superseded_impls
    pub = Path(__file__).resolve().parent.parent / "results" / "published"
    if not pub.exists():
        return
    marked = sorted(p.parent.name for p in pub.glob(f"*/{SUPERSEDED_MARKER}"))
    assert marked == ["2026-08-26-nvidia_h200-full-three-way",
                      "2026-08-28-nvidia_h200-h200-fp8-three-kernel"], marked
    assert superseded_impls(pub / marked[0] / "merged.csv") is None
    assert superseded_impls(pub / marked[1] / "merged.csv") == {
        "torch_scaled_grouped_mm_up", "torch_scaled_grouped_mm_down"}


# --- an arm can be superseded only in PART -----------------------------------

def test_an_arm_can_supersede_only_some_implementations(tmp_path):
    """MEASURED, 2026-08-28. `2026-08-28-nvidia_h200-h200-fp8-three-kernel` holds
    valid vLLM and SGLang fp8 rows alongside two torch spans that measured a
    quantisation pass inside the timed region. `-fp8-refixed` replaces only the
    torch ones: deepseek-v2-lite at T=8192 went 1.9855 ms to 0.7659.

    A directory-level marker would throw away 9,576 good rows to retract 9,408
    bad ones. A whole-arm rule is the wrong granularity for a partial retraction.
    """
    d = tmp_path / "arm"
    d.mkdir()
    (d / "merged.csv").write_text("x")
    (d / SUPERSEDED_MARKER).write_text(
        "superseded by: the-refixed-arm\n"
        "impls: torch_scaled_grouped_mm_up, torch_scaled_grouped_mm_down\n")
    from moe.bench.published import superseded_impls
    assert superseded_impls(d / "merged.csv") == {
        "torch_scaled_grouped_mm_up", "torch_scaled_grouped_mm_down"}


def test_no_impls_line_still_means_the_WHOLE_arm(tmp_path):
    """The existing full-three-way marker has no impls line and must keep
    meaning what it meant."""
    d = tmp_path / "arm"
    d.mkdir()
    (d / "merged.csv").write_text("x")
    (d / SUPERSEDED_MARKER).write_text("superseded by: the recalibrated arm\n")
    from moe.bench.published import superseded_impls
    assert superseded_impls(d / "merged.csv") is None      # None = all of it


def test_an_unmarked_arm_supersedes_nothing(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "merged.csv").write_text("x")
    from moe.bench.published import superseded_impls
    assert superseded_impls(d / "merged.csv") == set()     # empty = drop none


def test_a_partially_superseded_arm_is_not_dropped_wholesale(tmp_path):
    """filter_superseded must keep it, because most of its rows are good."""
    d = tmp_path / "arm"
    d.mkdir()
    (d / "merged.csv").write_text("x")
    (d / SUPERSEDED_MARKER).write_text("superseded by: x\nimpls: a_span\n")
    kept, dropped = filter_superseded([d / "merged.csv"])
    assert kept == [d / "merged.csv"] and dropped == []


def test_the_real_repo_marks_the_fp8_torch_spans():
    """The state this exists for."""
    from moe.bench.published import superseded_impls
    pub = Path(__file__).resolve().parent.parent / "results" / "published"
    arm = pub / "2026-08-28-nvidia_h200-h200-fp8-three-kernel" / "merged.csv"
    if not arm.exists():
        return
    assert superseded_impls(arm) == {
        "torch_scaled_grouped_mm_up", "torch_scaled_grouped_mm_down"}
