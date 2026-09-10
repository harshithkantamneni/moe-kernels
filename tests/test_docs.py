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

import pytest

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

#: Sentences that describe a RETIRED clock rule. The first group is the
#: one-sided LEVEL ("at least 0.95 of the reference"), retired when the
#: instrument went two-sided on 03df2d4; the second is the two-sided rule that
#: EXCLUDED the LOW side, retired on 2026-09-09 when the 750-cell H200 census
#: showed the under-load clock is set per tile by the kernel's own power draw,
#: so a band around the calibration GEMM's clock excluded the study's two
#: primary tiles and nothing else. A doc that carries one describes a rule the
#: tree no longer applies.
STALE_LEVEL_PHRASES = (
    "at least `LEVEL_FRACTION",
    "at least LEVEL_FRACTION",  # the retired one-sided wording, hunted here
    "one fix unblocks",
    "only arm that could confirm",
    "only arm that can confirm",
)

#: The LOW-excludes wording, retired 2026-09-09 and landed in README.md with
#: the test count in the 2026-09-10 integration, so these are checked over
#: README.md as well as `docs/` and are folded in below.
RETIRED_LOW_EXCLUDES_PHRASES = (
    "LOW or DRIFT excludes",
    "Only LOW or DRIFT excludes",
    "must test LEVEL AND",
)

#: What each doc that describes the clock rule has to say: the high edge
#: exists, the side is a column, DRIFT is the exclusion, the side excludes
#: nothing, and the corrected column has a name a reader can grep the CSV for.
REQUIRED_LEVEL_PHRASES = {
    "docs/APPARATUS.md": ("LEVEL_HIGH_FRACTION", "clock_level_side",
                          "pct_of_roof_at_cell_clock",
                          "EXCLUDED IF AND ONLY IF `clock_drift_ok` IS FALSE",
                          "excludes nothing"),
    "docs/POD_RUNBOOK.md": ("clock_level_side", "pct_of_roof_at_cell_clock",
                            "NOT an exclusion",
                            "`clock_drift_ok is False`"),
    "docs/RUNPOD.md": ("clock_level_side", "pct_of_roof_at_cell_clock",
                       "DRIFT alone excludes a row"),
    "README.md": ("pct_of_roof_at_cell_clock",),
}


#: A line that dates its own retraction may quote the retired sentence: that is
#: how this repository keeps the history visible without git, and forbidding it
#: would push the correction out of the document. The marker is the date.
RETRACTION_MARKER = "Until 2026-09-"


def _level_prose_defects(text: str, extra: tuple = ()) -> list[str]:
    """The stale phrases a text carries, ignoring lines that retract them.

    Read paragraph by paragraph rather than line by line, because these
    documents wrap at 78 columns and a retracted sentence and its date
    routinely sit on different lines of one paragraph."""
    found = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())   # these documents wrap at 78 columns
        if RETRACTION_MARKER in para:
            continue
        found += [phrase for phrase in STALE_LEVEL_PHRASES + extra
                  if phrase in para and phrase not in found]
    return found


def test_the_level_prose_checker_fires_on_a_planted_stale_row_and_not_on_a_clean_one():
    """Both sides planted: the exact table row APPARATUS.md carried on
    2026-09-08 must be caught, the LOW-excludes sentence it carried on
    2026-09-09 must be caught, and the corrected wording must pass, or the test
    below proves nothing either way."""
    planted = ("| LEVEL | `clock_level_ok` | was the SM clock under load at least "
               "`LEVEL_FRACTION = 0.95` of the reference clock the ROOF was "
               "measured at |")
    assert _level_prose_defects(planted) == ["at least `LEVEL_FRACTION"]
    assert _level_prose_defects("THE CLAIM ... the only arm that could confirm "
                                "the ceiling") == ["only arm that could confirm"]
    two_sided = ("inside the band [`LEVEL_FRACTION = 0.95`, `LEVEL_HIGH_FRACTION = "
                 "1.05`]; LOW or DRIFT excludes; HIGH is NOT an exclusion, read "
                 "`pct_of_roof_at_cell_clock`")
    assert _level_prose_defects(two_sided) == []
    assert _level_prose_defects(two_sided, RETIRED_LOW_EXCLUDES_PHRASES) == [
        "LOW or DRIFT excludes"]
    clean = ("inside the band [`LEVEL_FRACTION = 0.95`, `LEVEL_HIGH_FRACTION = "
             "1.05`], and the side is recorded: DRIFT alone excludes a row, "
             "read `pct_of_roof_at_cell_clock` beside the fixed-roof fraction")
    assert _level_prose_defects(clean, RETIRED_LOW_EXCLUDES_PHRASES) == []


def test_every_doc_describes_the_clock_rule_as_drift_only_with_the_side_recorded():
    """No doc or the README may describe a retired one-sided LEVEL, say the
    LOW side excludes, or call the n256 arm confirmable; and each doc that
    describes the rule states the band, the side, the exclusion (DRIFT alone)
    and `pct_of_roof_at_cell_clock` as the column to read.

    README.md was exempt from the LOW-excludes phrases while its sentence was
    the integrator's to land. It landed on 2026-09-10 in the same pass that
    set the test count, so the exemption is gone and the README is checked
    against the same list as every doc."""
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        defects = _level_prose_defects(path.read_text(),
                                       RETIRED_LOW_EXCLUDES_PHRASES)
        assert defects == [], f"{path.relative_to(ROOT)} still says {defects}"
    for rel, needles in REQUIRED_LEVEL_PHRASES.items():
        # Flattened: a required sentence that straddles a wrap is still there.
        text = " ".join((ROOT / rel).read_text().split())
        for needle in needles:
            assert needle in text, f"{rel} does not say {needle!r}"


# --------------------------------------------------------------------------
# the 2026-09-09 H200 session, in the two documents that carry results
# --------------------------------------------------------------------------

#: What FINDINGS and STUDY have to carry from that session, as (needle, why)
#: pairs. Every number here is off a `RESULT:` line in
#: `results/published/2026-09-09-nvidia_h200-gaps-session/session/logs/`, or
#: off the committed calibration, so a number that drifts here has drifted from
#: the page that produced it.
SESSION_2026_09_09 = (
    ("4814.3", "the derived pin rate every anchor result is quoted against"),
    ("668.5", "the measured dense bf16 rate"),
    ("1485", "the GEMM's own clock under load, the LEVEL reference"),
    ("1395", "the BLOCK_M=128/BLOCK_N=64 tile's own clock"),
    ("1650", "the BLOCK_M=256 tile's own clock"),
    ("76.2-77.9%", "anchor_measure P2, the anchor rate against the pin rate"),
    ("2.28%", "anchor_measure P1, swizzle invariance"),
    ("0.31%", "anchor_measure P3, slope independence"),
    ("BLOCK_M % 64 == 0", "mma_switch G4, the wgmma condition"),
    ("2025.1.1", "the ncu build that attached, so the counter route is open"),
    ("0.094", "the median re-anchoring shift against a quoted 0.05"),
    ("0.678", "the worst cap/ridge the BLOCK_M <= 64 cap survives at"),
    ("53,188", "the ruler's classified rows"),
    ("90", "the rows the 2.2% denominator swing flips"),
    ("144.9-152.8", "the ridge band a crossing inside it must be quoted as"),
)


@pytest.mark.parametrize("needle,why", SESSION_2026_09_09,
                         ids=[n for n, _ in SESSION_2026_09_09])
def test_findings_carries_the_2026_09_09_session_numbers(needle, why):
    text = (ROOT / "docs" / "FINDINGS.md").read_text()
    assert "## The 2026-09-09 H200 session" in text, "the dated section is gone"
    assert needle in text, f"FINDINGS no longer states {needle!r}: {why}"


def test_findings_files_the_six_invalid_arms_as_apparatus_and_not_as_results():
    """The INVALID arms are in the session section with a CAUSE each, and their
    numbers are not quoted as results. `cap_test`'s counterfactual alpha is the
    one most likely to be lifted out of context, so the page has to say in
    words that it is not a result until the arm is re-run."""
    text = (ROOT / "docs" / "FINDINGS.md").read_text()
    section = text.split("## The 2026-09-09 H200 session")[1].split("\n## ")[0]
    for arm in ("roofline-n64-g1", "bm128_depth", "bn_g16", "alias_ablation",
                "cap_test", "dtype"):
        assert f"`{arm}`" in section, arm
    assert "as apparatus findings" in section
    assert "Nothing on their pages\nis a result" in section
    assert "NOT quoted as a result until the arm is re-run" in section
    # And the defect of each is named, not just the arm.
    for cause in ("override_config", "688 % 32 = 16", "0.838 of the roof",
                  "smallest SWEPT block size", "5500 GB/s"):
        assert cause in section, cause


def test_study_records_what_the_session_settled_and_what_it_left():
    text = (ROOT / "docs" / "STUDY.md").read_text()
    assert "## What the 2026-09-09 H200 session settled" in text
    # Item 3's loose end is struck through where it lives, not only claimed.
    order = text.split("## Order of work")[1]
    assert "~~Loose end: confirm the instruction actually switched" in order
    assert "CLOSED 2026-09-09" in order
    # The three things that changed the study's working state.
    for needle in ("BLOCK_M % 64 == 0", "counter route is OPEN",
                   "may be quoted as a point", "0.678 of the ridge",
                   "144.9-152.8", "--new"):
        assert needle in text, needle


def test_apparatus_states_the_clock_rule_with_its_date_and_its_reason():
    """R1-R3 on one page: the rule, the date it changed, and the per-tile
    numbers that changed it. A rule with no reason on the page is a setting,
    and the next reader will widen it."""
    # Flattened: this page wraps at 78 columns and a sentence that straddles
    # two lines is still the sentence.
    text = " ".join((ROOT / "docs" / "APPARATUS.md").read_text().split())
    assert "SINCE 2026-09-09 A CELL IS EXCLUDED IF AND ONLY IF" in text
    for needle in ("1395 MHz", "1650 MHz", "1485 MHz", "700 W",
                   "15 MHz step", "settle_ms", "power_w", "1469.9",
                   "15 of the roofline arm's 39 cells",
                   "110 of the depth arm's 168 treads"):
        assert needle in text, needle
    # THE SWIZZLE SPLIT AT BLOCK_M=32. This table read "BLOCK_M=32 (any
    # GROUP_SIZE_M) | 1736 MHz | 68" until 2026-09-09, which is a median over a
    # bimodal set and names no operating point of either family: the corpus has
    # 18 cells at GROUP_SIZE_M=1 with a median of 1474 MHz and 50 from
    # GROUP_SIZE_M=8 up with a median of 1740, 266 MHz apart, and the row
    # asserted a swizzle invariance one row above the row that shows the swizzle
    # moving BLOCK_M=64 by 100 MHz. The pooled figure survives only where it is
    # retracted.
    assert "| BLOCK_M=32, GROUP_SIZE_M=1 | 1474 MHz | 18 |" in text
    assert "| BLOCK_M=32, GROUP_SIZE_M>=8 | 1740 MHz | 50 |" in text
    assert "(any GROUP_SIZE_M)" not in text
    # The pooled figure survives only inside the paragraph that retracts it.
    before, _, after = text.partition("Two of those rows were wrong")
    assert "1736" not in before and "1736 MHz" in after
    # AND THE BLOCK_N SPLIT IS COUNTED OVER ITS OWN POOL. The three medians at
    # fixed BLOCK_M=256 are over 68 cells each, from bn_decomposition alone; the
    # row was labelled 311, which is every BLOCK_M=256 cell in the session, 107
    # of which carry no BLOCK_N at all.
    assert "1725 / 1620 / 1560 MHz | 68 each" in text
    # The scoring half of the rule, which is the half a gate reads.
    assert "compute-bound CLAIM gates read the FIXED roof fraction" in text
    assert "issue efficiency" in text


# --------------------------------------------------------------------------
# the 2026-09-10 H200 session, and the three retractions that are its product
# --------------------------------------------------------------------------

#: What FINDINGS has to carry from that session, as (needle, why) pairs. Every
#: number is either off a `RESULT:` line in
#: `results/published/2026-09-10-nvidia_h200-gaps-session/session/logs/`, off
#: the committed calibration, or recomputed from that directory's cells by the
#: synthesis scripts named beside it. INTERVALS, NOT POINTS: where the quantity
#: has a spread the pinned string is the spread, because the session's own
#: lesson is that this study's headline numbers were points printed by an
#: estimator whose spread nobody had propagated.
SESSION_2026_09_10 = (
    ("682.1", "the measured dense bf16 rate, up from 668.5 on 2026-09-09"),
    ("1470 MHz", "the GEMM's own clock under load on this rental"),
    ("155.9", "this card's ridge; the 2026-09-09 one was 152.8"),
    ("147.9-155.9", "the ridge band a crossing inside it must be quoted as"),
    ("2.8186 GB", "mixtral bf16's whole expert weight set"),
    ("0.6443 ms", "one full stream of it at the card's measured triad rate"),
    ("0.68 to 1.37", "the per-M-tile cost in weight streams, over 23 ladders"),
    ("89.158 ms", "cap_test's measured time at n = 132 M-tiles"),
    ("85.054 ms", "132 full weight streams at triad, the model-free comparison"),
    ("95.4%", "the fraction of BLOCK_M=16's wall clock that is one re-read"),
    ("1.115 +/- 0.003", "the BN-scaling ratio where the model requires 2.000"),
    ("z = -303", "how far that sits from the model"),
    ("0.0218", "the 1/BLOCK_N rival's rms"),
    ("0.0675", "the published form's rms, 3.1x worse at equal parameters"),
    ("-23.7%", "what GROUP_SIZE_M 1 -> 16 does to the per-M-tile cost"),
    ("0.099", "BLOCK_M=16's peak fraction of the dense roof"),
    ("0.537", "the BLOCK_M=256 control's, on the same layer"),
    ("65536", "the accumulator and the per-block register file, coinciding"),
    ("1275 to 1935 MHz", "the clock range at a power held at the cap"),
    ("2328 cells", "the census the clock result is over"),
    ("0.013380", "the pooled same-session replicate floor"),
    ("[0.0100, 0.0204]", "its 95% interval, which excludes the assumed 0.0229"),
    ("+/- 0.048", "alpha_b's honest interval, not the bootstrap's +/- 0.0113"),
    ("30.5%", "the D-propagated draws that put alpha_b outside [0, 1]"),
    ("0.807", "BLOCK_M=128 cap/ridge at alpha_b = 0.9794, capped"),
    ("1.293", "the same at LIN's 0.5977, NOT capped: the verdict flips"),
    ("0.784", "the alpha_b at which BLOCK_M=128 stops being capped"),
    ("3.85 GB", "the traffic reading the counter would see at BLOCK_N=32"),
    ("2.06 GB", "and at BLOCK_N=128; the same bytes at both means time"),
)


@pytest.mark.parametrize("needle,why", SESSION_2026_09_10,
                         ids=[n for n, _ in SESSION_2026_09_10])
def test_findings_carries_the_2026_09_10_session_numbers(needle, why):
    text = (ROOT / "docs" / "FINDINGS.md").read_text()
    assert "## The 2026-09-10 H200 session" in text, "the dated section is gone"
    assert needle in text, f"FINDINGS no longer states {needle!r}: {why}"


def test_findings_retracts_the_three_numbers_the_session_retired():
    """THE RETRACTIONS ARE THE POINT OF THAT SECTION, so they are pinned in
    words and not only in numbers.

    (1) alpha_b = 0.9794 +/- 0.0113 is a fit whose partner ai_model refuses;
    (2) the 207% TEMPO contradiction compares a bound with a number;
    (3) cap/ridge = 0.080 is computed from an alpha ai_model refuses to invert,
        and at BLOCK_M=128, the tile vLLM ships, the verdict flips across
        the candidate range rather than binding.
    A reader who finds any of the three quoted as a measurement elsewhere has
    to find the retraction here."""
    text = (ROOT / "docs" / "FINDINGS.md").read_text()
    section = text.split("## The 2026-09-10 H200 session")[1].split("\n## ")[0]
    assert "THE RETRACTIONS" in section
    for needle in ("0.9794", "-0.8143", "207%", "0.080", "1.28982"):
        assert needle in section, needle
    # Flattened from here: this file wraps at 78 columns and a sentence that
    # straddles two lines is still the sentence.
    flat = " ".join(section.split())
    # Each one says, in words, what it is being retracted FOR.
    assert "must NOT be quoted as a measurement" in flat
    assert "compares a BOUND with a number" in flat
    assert "refuses to invert" in flat
    # And the half that survives is stated as plainly as the half that does not.
    assert "the cap binds under every reading this study has ever held" in flat
    assert "the verdict FLIPS across the candidate range" in flat
    assert "NOT ESTABLISHED" in flat
    # The reachability caveat, which is what stops the surviving half from
    # being read as a statement about production.
    assert "exactly ONE M-tile per expert" in flat


def test_study_records_what_the_2026_09_10_session_settled_and_what_it_left():
    text = (ROOT / "docs" / "STUDY.md").read_text()
    assert "## What the 2026-09-10 H200 session settled" in text
    section = text.split("## What the 2026-09-10 H200 session settled")[1]
    section = section.split("\n## ")[0]
    for needle in ("0.6443 ms", "0.68 to 1.37", "95.4%", "z = -303",
                   "65536", "1275-1935 MHz", "+/- 0.048", "30.5%",
                   "1.293", "0.784", "one M-tile per expert",
                   "Never quote the 207% TEMPO", "--new"):
        assert needle in section, needle
    # The next session is a command a reader can run, with the arms named.
    assert "--only calibrate,pin_probe-n64-g1,bn_g16,dtype,counter_plan,counter" in section


def test_apparatus_states_the_estimator_change_and_prints_both():
    """R1-R3 for the estimator, on the page that is the vocabulary: what
    B/(A+B) does wrong, what replaces it, and that both are printed. A page
    that names only the replacement leaves every published alpha unreadable
    against its own history; a page that names only the fit teaches the next
    reader to quote it."""
    text = " ".join((ROOT / "docs" / "APPARATUS.md").read_text().split())
    assert "The estimator changed on 2026-09-10, and both are printed" in text
    # What is wrong with it, with the arithmetic that makes it wrong.
    assert "value of the ladder's straight line at `n = 0`" in text
    assert "exceeds 1 exactly when `D > A`" in text
    assert "-0.8143" in text and "REFUSES the pair" in text
    # What replaces it, with the measurement in its denominator.
    assert "2.8186 GB" in text and "0.6443 ms" in text
    assert "0.68 to 1.37" in text
    # And that it is not alpha_b, which is the mistake the change invites.
    assert "`w` is not `alpha_b`" in text
    for rate in ("0.6087", "0.5930", "0.5143"):
        assert rate in text, rate
