"""Which published result arms an analysis should read.

`results/published/` accumulates arms, and some of them are REPLACEMENTS rather
than additions. The 2026-08-26 three-way sweep has a `-recalibrated` twin whose
17,640 ms_p50 values are identical to the original's; only ceiling columns
differ. Reading both weights every one of those rows twice.

That is not hypothetical. Pointed at both, mixtral's torch_grouped_mm_up crossing
reported 1.46x of prediction; pointed at one, 0.63x. Neither run announced which
file set it had used.

A marker file inside the superseded directory makes the arm describe its own
status, so the rule lives with the data instead of in every caller's head.
"""
from __future__ import annotations

from pathlib import Path

#: Dropped in a published arm that a later one replaces. Its contents are the
#: human-readable reason, which reports quote when they skip it.
SUPERSEDED_MARKER = "SUPERSEDED"


def is_superseded(csv_path: Path | str) -> bool:
    """Does this file live in an arm that a later one replaced?"""
    return (Path(csv_path).parent / SUPERSEDED_MARKER).exists()


def superseded_reason(csv_path: Path | str) -> str:
    marker = Path(csv_path).parent / SUPERSEDED_MARKER
    try:
        return marker.read_text().strip()
    except OSError:
        return ""


def filter_superseded(paths) -> tuple[list[Path], list[Path]]:
    """`(kept, dropped)`.

    Both halves are returned so a caller can ANNOUNCE what it dropped. A silently
    discarded input is the same class of error as a silently double-counted one.
    """
    kept, dropped = [], []
    for p in paths:
        (dropped if is_superseded(p) else kept).append(Path(p))
    return kept, dropped
