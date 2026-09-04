"""Every number the documentation quotes is recomputed from the tree.

Three documents once gave three different test counts (549, 874 and 605
against a tree at 3,401), README said ten arms and 96,448 rows against 14 and
100,144, and the rebuild's four foundation modules were documented nowhere
outside their own docstrings. A reader cannot tell a stale number from a
current one by looking at it, so the numbers on the page are pinned to the
tree here and this test is what fails when the page goes stale.

The counts are computed the way the study defines them, through the same
module the publishing path uses, not re-typed: a "current" row is one whose
implementation is not marked superseded in its own arm's directory
(`moe.bench.published.superseded_impls`), which is how 100,144 rows in eleven
CSVs become 72,760 current ones.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moe.bench import published as P  # noqa: E402

README = (ROOT / "README.md").read_text()
PUBLISHED = ROOT / "results" / "published"


def _readme_int(pattern: str) -> int:
    m = re.search(pattern, README)
    assert m, f"README no longer states {pattern!r}"
    return int(m.group(1).replace(",", ""))


def _arms() -> list[Path]:
    return sorted(p for p in PUBLISHED.glob("2026-*") if p.is_dir())


def test_readme_test_count_is_the_trees():
    """`pytest --collect-only -q` on this tree, versus the README's sentence."""
    claimed = _readme_int(r"(\d[\d,]*) tests collected off-GPU")
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         str(ROOT / "tests")],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    m = re.search(r"(\d+) tests? collected", out.stdout)
    assert m, out.stdout[-500:]
    assert int(m.group(1)) == claimed, (
        f"README says {claimed} tests collected, the tree collects {m.group(1)}: "
        "update the Status paragraph")


def test_readme_arm_counts_are_the_trees():
    arms = _arms()
    with_csv = [a for a in arms if (a / "merged.csv").exists()]
    reports = sum(1 for a in arms for _ in a.glob("*.report.json"))
    ladder_arms = [a for a in arms if not (a / "merged.csv").exists()
                   and any(a.glob("*.report.json"))]
    assert _readme_int(r"(\d+) published arms") == len(arms), len(arms)
    assert _readme_int(r"(\d+) carry a `merged.csv`") == len(with_csv), len(with_csv)
    assert _readme_int(r"(\d+) are ladder\s+arms") == len(ladder_arms), len(ladder_arms)
    assert _readme_int(r"carrying (\d+) `\*\.report\.json`") == reports, reports


def test_readme_row_counts_are_the_trees_by_the_studys_own_definition():
    total = current = 0
    for a in _arms():
        f = a / "merged.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open()))
        sup = P.superseded_impls(f)
        total += len(rows)
        if sup is None:            # the whole arm is superseded
            continue
        current += sum(1 for r in rows if r.get("impl") not in sup)
    assert _readme_int(r"(\d[\d,]*) rows in all") == total, total
    assert _readme_int(r"(\d[\d,]*) of\s+them current") == current, current


def test_the_four_foundation_modules_are_documented_outside_their_docstrings():
    """The audit's reproducibility check: `grep -rc 'EXA|TIMING_BASIS|time_kernel'
    docs/*.md README.md` returned 0 for every file."""
    text = "\n".join(p.read_text() for p in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
    for needle in ("moe/bench/timing.py", "moe/bench/exit_codes.py",
                   "moe/bench/provenance.py", "moe/bench/ai_model.py",
                   "TIMING_BASIS", "time_kernel", "(1 + phi + delta)", "RESULT: "):
        assert needle in text, f"{needle!r} is documented nowhere under docs/ or README"
    assert (ROOT / "docs" / "APPARATUS.md").exists()


def test_the_session_driver_is_documented():
    runbook = (ROOT / "docs" / "POD_RUNBOOK.md").read_text()
    assert "scripts/h200_gaps_session.sh" in runbook
    assert "SESSION=" in runbook or "--resume-latest" in runbook
