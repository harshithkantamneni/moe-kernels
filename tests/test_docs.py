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


def test_the_runbook_arm_table_names_only_arms_the_driver_lists():
    """The runbook says its arm names "are asserted against `--list` by
    `tests/test_docs.py`"; until 2026-09-08 nothing did. Every `| \`arm\` | min |`
    row in the runbook has to be an arm `h200_gaps_session.sh --list` prints,
    off-GPU, so a renamed or retired arm cannot keep a row."""
    runbook = (ROOT / "docs" / "POD_RUNBOOK.md").read_text()
    documented = set(re.findall(r"^\| `([a-z0-9_-]+)` \| *\d+ \|", runbook, re.M))
    assert documented, "the runbook's arm table has no rows"
    out = subprocess.run(
        ["bash", str(ROOT / "scripts" / "h200_gaps_session.sh"), "--list"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    listed = set(re.findall(r"^  ([a-z0-9_-]+)\s+~\d+ min", out.stdout, re.M))
    assert listed, out.stdout[-500:]
    assert documented <= listed, documented - listed


# --------------------------------------------------------------------------
# LEVEL is a band with a side, and the docs must say so in the same words
# --------------------------------------------------------------------------

#: Sentences that describe the RETIRED one-sided LEVEL ("at least 0.95 of the
#: reference"), or call the n256 roofline arm confirmable. Each was standing
#: in a doc on 2026-09-08 after the instrument went two-sided (03df2d4), the
#: docs slice and the timing slice having merged side by side. A doc that
#: carries one describes a flag the tree no longer computes.
STALE_LEVEL_PHRASES = (
    "at least `LEVEL_FRACTION",
    "at least LEVEL_FRACTION",  # the retired one-sided wording, hunted here
    "one fix unblocks",
    "only arm that could confirm",
    "only arm that can confirm",
)

#: What each doc that describes LEVEL has to say: the high edge exists, the
#: side is a column, the rule is stated in these words, and the corrected
#: column has a name a reader can grep the CSV for.
REQUIRED_LEVEL_PHRASES = {
    "docs/APPARATUS.md": ("LEVEL_HIGH_FRACTION", "clock_level_side",
                          "pct_of_roof_at_cell_clock", "LOW or DRIFT excludes",
                          "NOT an exclusion"),
    "docs/POD_RUNBOOK.md": ("clock_level_side", "pct_of_roof_at_cell_clock",
                            "NOT an exclusion"),
    "docs/RUNPOD.md": ("clock_level_side", "pct_of_roof_at_cell_clock",
                       "Only LOW or DRIFT excludes"),
    "README.md": ("pct_of_roof_at_cell_clock", "LOW or DRIFT excludes",
                  "not an exclusion"),
}


def _level_prose_defects(text: str) -> list[str]:
    """The stale phrases a text carries. Empty means clean."""
    return [phrase for phrase in STALE_LEVEL_PHRASES if phrase in text]


def test_the_level_prose_checker_fires_on_a_planted_stale_row_and_not_on_a_clean_one():
    """Both sides planted: the exact table row APPARATUS.md carried on
    2026-09-08 must be caught, and the corrected wording must pass, or the
    test below proves nothing either way."""
    planted = ("| LEVEL | `clock_level_ok` | was the SM clock under load at least "
               "`LEVEL_FRACTION = 0.95` of the reference clock the ROOF was "
               "measured at |")
    assert _level_prose_defects(planted) == ["at least `LEVEL_FRACTION"]
    assert _level_prose_defects("THE CLAIM ... the only arm that could confirm "
                                "the ceiling") == ["only arm that could confirm"]
    clean = ("inside the band [`LEVEL_FRACTION = 0.95`, `LEVEL_HIGH_FRACTION = "
             "1.05`]; LOW or DRIFT excludes; HIGH is NOT an exclusion, read "
             "`pct_of_roof_at_cell_clock`")
    assert _level_prose_defects(clean) == []


def test_every_doc_describes_LEVEL_as_a_two_sided_band_and_names_the_column_to_read():
    """No doc or the README may describe the retired one-sided LEVEL or call
    the n256 arm confirmable, and each doc that describes LEVEL states the
    band, the side, the rule (LOW or DRIFT excludes; HIGH is not an exclusion)
    and `pct_of_roof_at_cell_clock` as the column to read."""
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        defects = _level_prose_defects(path.read_text())
        assert defects == [], f"{path.relative_to(ROOT)} still says {defects}"
    for rel, needles in REQUIRED_LEVEL_PHRASES.items():
        text = (ROOT / rel).read_text()
        for needle in needles:
            assert needle in text, f"{rel} does not say {needle!r}"
