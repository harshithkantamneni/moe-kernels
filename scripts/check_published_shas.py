#!/usr/bin/env python
"""Does every commit a published row names still exist in this repository?

A row's `git_sha` is the whole of its attribution: it is the only thing that
says which code produced the number. Nothing checked it until now, and one
value in `results/published` does not resolve. 2,100 ridge-resolution rows (all
deepseek-v3 bf16) cite `7eecff427626c40795b9543a10122a6d86595ab2`, a commit made
on the pod and rewritten by the very `git pull --rebase` that
`scripts/publish_results.sh` recommends when a push is rejected. Those rows say
"clean at 7eecff4" and no one can check the claim, because the tree that sha
named was never pushed and no longer exists anywhere.

The related shape is a value that is not a sha at all.
`publish_results.sh` writes its commit message as "Results: <label> on
$(hostname)", and a RunPod hostname is twelve hex characters
(`0b23ff0a8486`), which reads exactly like an abbreviated commit and sorts into
the same column of a reader's attention. So a value that is not 40 lowercase hex
is reported as MALFORMED with that history named, rather than being handed to
git and coming back as an ordinary miss.

WHAT EACH VERDICT MEANS
-----------------------
    PRESENT     40 hex, and `git cat-file -e <sha>^{commit}` resolves it here.
    MISSING     40 hex, and it does not resolve. The rows cite code nobody has.
    MALFORMED   not 40 lowercase hex. Not a commit id; see the hostname above.
    UNRECORDED  empty, or the schema's UNRECORDED sentinel. The harness did not
                stamp a sha at all. Reported and never blocking: it is the state
                of the oldest arms and of every synthetic fixture, and it is a
                different defect from a sha that is present and wrong.
    UNPUSHED    resolves here, but no remote branch contains it (--require-remote).
                A stranger cloning the repo cannot reach it, so for them it is
                MISSING; for you it is one `git push` away.

WHY UNRECORDED DOES NOT FAIL THE GATE. A gate has to be able to pass, and a gate
that fails on every legacy arm is one an operator learns to skip, which is the
failure mode `docs/POD_RUNBOOK.md` already warns about for test counts. The
blocking question is narrower and answerable: of the shas that ARE recorded,
does every one resolve.

EXIT CODES are `moe.bench.exit_codes`: DONE when every recorded sha is present
and well formed, CLAIM_FAIL when one is not, REFUSED when there were no rows to
check, INVALID when git itself could not be asked (an answer of "nothing is
missing" from a git that never ran is the shape this file exists to refuse).
Every scored gate prints exactly one `RESULT:` line and nothing else does.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import exit_codes as EX  # noqa: E402
from moe.bench.provenance import provenance_block  # noqa: E402

#: What this script IS, recorded in its report so a stranger reading the JSON
#: knows no timing happened here.
INSTRUMENT = "git cat-file over the git_sha column; no measurement"

#: A full commit id and nothing else. An abbreviation is refused on purpose:
#: it cannot be checked for uniqueness against a repository that has grown, and
#: the twelve-hex pod hostname is indistinguishable from one.
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

#: The schema's "the harness did not record this" spelling, plus the empties a
#: hand-written CSV produces.
_UNRECORDED = {"", "unrecorded", "UNRECORDED", "none", "None", "n/a"}

PRESENT, MISSING, MALFORMED, UNRECORDED, UNPUSHED = (
    "PRESENT", "MISSING", "MALFORMED", "UNRECORDED", "UNPUSHED")


class GitUnavailable(RuntimeError):
    """git could not be asked, so no answer about resolvability is available.

    Distinct from "nothing was missing": a check that could not run reports
    zero failures, and that is this project's documented failure shape.
    """


@dataclass
class ShaRecord:
    """One distinct git_sha value, its verdict, and where it came from."""

    value: str
    verdict: str
    rows: int = 0
    arms: set[str] = field(default_factory=set)
    note: str = ""

    def as_dict(self) -> dict:
        return {"git_sha": self.value, "verdict": self.verdict, "rows": self.rows,
                "arms": sorted(self.arms), "note": self.note}


def collect_shas(paths: list[Path]) -> dict[str, ShaRecord]:
    """Every distinct `git_sha` in the given `run_*.csv` files, with its row count.

    A CSV with no `git_sha` column contributes nothing and is not an error:
    `merged.csv` and the report sidecars live in the same directories.
    """
    out: dict[str, ShaRecord] = {}
    for path in paths:
        arm = path.parent.name
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if "git_sha" not in row:
                    break
                value = (row.get("git_sha") or "").strip()
                rec = out.get(value)
                if rec is None:
                    rec = out[value] = ShaRecord(value=value, verdict=UNRECORDED)
                rec.rows += 1
                rec.arms.add(arm)
    return out


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=30)
    except FileNotFoundError as exc:
        raise GitUnavailable("git is not on PATH, so no sha can be resolved") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitUnavailable(f"git failed: {type(exc).__name__}") from exc


def classify_shas(records: dict[str, ShaRecord], repo: Path,
                  require_remote: bool = False) -> dict[str, ShaRecord]:
    """Give every record its verdict. Mutates and returns `records`.

    `git cat-file -e <sha>^{commit}` rather than `-t`: the peel refuses a blob
    or a tree that happens to share the abbreviation, and the rows claim a
    COMMIT.
    """
    probe = _git(repo, "rev-parse", "--git-dir")
    if probe.returncode != 0:
        raise GitUnavailable(f"{repo} is not a git repository, so no sha can be "
                             "resolved from here")
    for value, rec in records.items():
        if value in _UNRECORDED:
            rec.verdict = UNRECORDED
            rec.note = "the harness stamped no commit on these rows"
            continue
        if not _FULL_SHA.match(value):
            rec.verdict = MALFORMED
            rec.note = ("not 40 lowercase hex, so it is not a commit id. A "
                        "RunPod hostname is twelve hex characters and reads "
                        "exactly like an abbreviated sha; publish_results.sh "
                        "puts one in its commit message")
            continue
        if _git(repo, "cat-file", "-e", f"{value}^{{commit}}").returncode != 0:
            rec.verdict = MISSING
            rec.note = ("no such commit in this repository; a pod commit "
                        "rewritten by git pull --rebase leaves rows exactly "
                        "like this")
            continue
        if require_remote:
            out = _git(repo, "branch", "-r", "--contains", value)
            if out.returncode != 0 or not out.stdout.strip():
                rec.verdict = UNPUSHED
                rec.note = ("resolves locally, but no remote branch contains "
                            "it: a stranger who clones cannot reach it")
                continue
        rec.verdict = PRESENT
        rec.note = ""
    return records


def render(records: dict[str, ShaRecord]) -> list[str]:
    """One line per distinct sha, worst verdict first, then by row count."""
    order = {MISSING: 0, MALFORMED: 1, UNPUSHED: 2, UNRECORDED: 3, PRESENT: 4}
    lines = []
    for rec in sorted(records.values(),
                      key=lambda r: (order.get(r.verdict, 9), -r.rows, r.value)):
        shown = rec.value or "(empty)"
        lines.append(f"{rec.verdict:<10} {shown:<42} {rec.rows:>7} row(s)  "
                     f"{', '.join(sorted(rec.arms))}")
        if rec.note:
            lines.append(f"{'':<10} {rec.note}")
    return lines


def gates(records: dict[str, ShaRecord], require_remote: bool) -> list[tuple[str, str, str, str]]:
    """`(kind, name, verdict, detail)` for every gate this run scores.

    Both claims can PASS and both can FAIL; `--self-test` plants each failure.
    """
    missing = [r for r in records.values() if r.verdict == MISSING]
    malformed = [r for r in records.values() if r.verdict == MALFORMED]
    unpushed = [r for r in records.values() if r.verdict == UNPUSHED]
    unrecorded = sum(r.rows for r in records.values() if r.verdict == UNRECORDED)

    out = [
        (EX.CLAIM, "shas_resolvable",
         EX.PASS if not missing else EX.FAIL,
         (f"every recorded sha resolves; {unrecorded} row(s) carry none"
          if not missing else
          f"{len(missing)} sha(s) resolve nowhere, over "
          f"{sum(r.rows for r in missing)} row(s): "
          + " ".join(r.value[:12] for r in missing))),
        (EX.CLAIM, "shas_well_formed",
         EX.PASS if not malformed else EX.FAIL,
         ("every recorded sha is 40 hex" if not malformed else
          f"{len(malformed)} value(s) are not commit ids: "
          + " ".join(r.value or "(empty)" for r in malformed))),
    ]
    if require_remote:
        out.append((EX.CLAIM, "shas_on_a_remote",
                    EX.PASS if not unpushed else EX.FAIL,
                    ("every resolvable sha is on a remote branch" if not unpushed
                     else f"{len(unpushed)} sha(s) are local only: "
                          + " ".join(r.value[:12] for r in unpushed))))
    return out


def run(published: Path, arms: list[Path] | None, repo: Path,
        require_remote: bool, json_out: Path | None,
        stream=sys.stdout) -> int:
    """Check, print, and return the exit code the table gives.

    REFUSED when there is nothing to check: an empty published tree is not a
    clean bill of health, and returning DONE for it is the "a check that
    examined nothing reports zero failures" shape.
    """
    if arms:
        files = sorted(p for arm in arms for p in Path(arm).glob("run_*.csv"))
        where = ", ".join(str(a) for a in arms)
    else:
        files = sorted(published.glob("*/run_*.csv"))
        where = str(published)
    if not files:
        print(f"REFUSED: no run_*.csv under {where}; there are no rows whose "
              "commit could be checked", file=sys.stderr)
        return EX.REFUSED

    records = collect_shas(files)
    if not records:
        print(f"REFUSED: {len(files)} CSV(s) under {where} carry no git_sha "
              "column at all", file=sys.stderr)
        return EX.REFUSED

    try:
        classify_shas(records, repo, require_remote=require_remote)
    except GitUnavailable as exc:
        # INVALID, not REFUSED: the rows were read, and the reason nothing can
        # be said about them is the instrument, not the world.
        print(f"INVALID: {exc}", file=sys.stderr)
        print(EX.result_line(EX.VALIDITY, "git_available", EX.FAIL, str(exc)),
              file=stream)
        return EX.INVALID

    print(f"# git_sha resolvability over {len(files)} CSV(s) under {where}",
          file=stream)
    print(f"# repository {repo}", file=stream)
    for line in render(records):
        print(line, file=stream)
    print(file=stream)

    scored = gates(records, require_remote)
    print(EX.result_line(EX.VALIDITY, "git_available", EX.PASS,
                         f"{len(records)} distinct value(s) read"), file=stream)
    for kind, name, verdict, detail in scored:
        print(EX.result_line(kind, name, verdict, detail), file=stream)

    if json_out is not None:
        payload = {
            "shas": [r.as_dict() for r in records.values()],
            "gates": [{"kind": k, "name": n, "verdict": v, "detail": d}
                      for k, n, v, d in scored],
            "files": [str(f) for f in files],
        }
        prov = provenance_block(repo_root=repo, instrument=INSTRUMENT)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(prov.stamp(payload), indent=2, default=str))
        print(f"wrote {json_out}", file=stream)

    return EX.classify([(EX.VALIDITY, EX.PASS)] + [(k, v) for k, _n, v, _d in scored])


# --------------------------------------------------------------------------
# self-test: both branches of both gates, against a real throwaway repository
# --------------------------------------------------------------------------

def _fixture(root: Path, shas: list[str]) -> tuple[Path, Path]:
    """A one-commit git repo and a published tree whose rows cite `shas`.

    Returns `(repo, published)`. The first element of `shas` may be the literal
    "HEAD", which is replaced by the repo's real commit; that is how the PASS
    branch gets a sha that genuinely resolves.
    """
    repo = root / "repo"
    (repo / "results" / "published" / "arm").mkdir(parents=True)
    for args in (["init", "-q", "-b", "main"],
                 ["config", "user.email", "t@example.com"],
                 ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True,
                       capture_output=True)
    (repo / "README").write_text("fixture\n")
    subprocess.run(["git", "-C", str(repo), "add", "README"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"],
                   check=True, capture_output=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    published = repo / "results" / "published"
    with (published / "arm" / "run_test_base.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["git_sha", "ms_p50"])
        w.writeheader()
        for sha in shas:
            w.writerow({"git_sha": head if sha == "HEAD" else sha, "ms_p50": "1.0"})
    return repo, published


def self_test() -> int:
    """Exercise every branch off-GPU. Returns 0 when all four cases agree.

    The cases, and why each is here:
      clean       one resolvable sha            -> DONE
      missing     a 40-hex commit nobody has    -> CLAIM_FAIL on shas_resolvable
      malformed   the twelve-hex pod hostname   -> CLAIM_FAIL on shas_well_formed
      unrecorded  an empty column               -> DONE, reported and not blocking
    The proof that this function can return 1 lives in
    `tests/test_shell_gates.py::test_sha_self_test_fails_when_a_case_is_broken`,
    which plants a wrong expectation and asserts the 1.
    """
    import io
    cases = [
        ("clean", ["HEAD"], EX.DONE, ()),
        ("missing", ["HEAD", "7eecff427626c40795b9543a10122a6d86595ab2"],
         EX.CLAIM_FAIL, ("shas_resolvable",)),
        ("malformed", ["HEAD", "0b23ff0a8486"], EX.CLAIM_FAIL,
         ("shas_well_formed",)),
        ("unrecorded", ["HEAD", ""], EX.DONE, ()),
    ]
    bad = 0
    for name, shas, want, failing in cases:
        with tempfile.TemporaryDirectory() as tmp:
            repo, published = _fixture(Path(tmp), shas)
            buf = io.StringIO()
            got = run(published, None, repo, False, None, stream=buf)
            lines = EX.parse_result_lines(buf.getvalue())
            failed = tuple(sorted(r.name for r in lines if r.verdict != EX.PASS))
            ok = got == want and failed == tuple(sorted(failing))
            bad += not ok
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: exit {got} "
                  f"{EX.CODE_NAMES.get(got, '?')} (want {want} "
                  f"{EX.CODE_NAMES[want]}), failing gates {failed} "
                  f"(want {tuple(sorted(failing))})")
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "published"
        empty.mkdir()
        got = run(empty, None, Path(tmp), False, None)
        ok = got == EX.REFUSED
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] empty tree refuses: exit {got} "
              f"(want {EX.REFUSED} REFUSED)")
    return 0 if bad == 0 else 1


def main(argv: list[str] | None = None) -> int:
    repo_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published", type=Path,
                    default=repo_default / "results" / "published",
                    help="the published tree to scan (default: this repo's)")
    ap.add_argument("--arm", type=Path, action="append", default=None,
                    help="check only these arm directories; repeatable")
    ap.add_argument("--repo", type=Path, default=repo_default,
                    help="the repository the shas must resolve in")
    ap.add_argument("--require-remote", action="store_true",
                    help="also require that a remote branch contains each sha, "
                         "which is what a stranger cloning the repo needs. Used "
                         "by publish_results.sh AFTER its push")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the verdicts and a provenance block here")
    ap.add_argument("--self-test", action="store_true",
                    help="run the four planted cases off-GPU and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    return run(args.published, args.arm, args.repo, args.require_remote, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
