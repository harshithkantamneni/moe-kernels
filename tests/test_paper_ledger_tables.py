"""scripts/paper/ledger_tables.py: Appendix A and the summary counts regenerate byte
for byte from committed files, every checked row agrees with its scorer output, and
the counts are the outline's."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))

import ledger_tables as LT  # noqa: E402

DOC = ROOT / "docs/paper"
FILES = ("appendix_a.md", "appendix_a.csv", "data/appendix_a_cells.csv", "summary_counts.md",
         "summary_counts.csv")


def test_committed_tables_regenerate(tmp_path):
    LT.write(LT.build(), tmp_path)
    for f in FILES:
        assert (tmp_path / f).read_bytes() == (DOC / f).read_bytes(), f


def test_every_row_and_every_check_agrees():
    rows = LT.build()["rows"]
    assert [r["row"] for r in rows] == list(range(1, 77))
    assert not [r["row"] for r in rows if r["agree"] == "NO"]
    hand = [r["row"] for r in rows if r["read_verdict"].startswith("hand-scored")]
    assert hand == [*range(1, 10), 13, 18]


def test_summary_counts():
    s = {(g, v): k for g, v, k, _w in LT.summary()}
    t = "primary time tests"
    assert (s[(t, "HELD")], s[(t, "FALSIFIED")], s[(t, "NOT SCORABLE")], s[(t, "NOT RUN")]) == (3, 3, 1, 2)
    f = "per-GEMM floor tests (base capture, each registration once)"
    assert (s[(f, "HELD")], s[(f, "FALSIFIED")], s[(f, "NOT SCORABLE")]) == (14, 2, 1)
    p = "PRIVATE bytes per GEMM, five 2026-09-29 GH200 registrations"
    assert (s[(p, "HELD")], s[(p, "FALSIFIED")]) == (9, 1)
    assert list(csv.DictReader((DOC / "summary_counts.csv").open()))
