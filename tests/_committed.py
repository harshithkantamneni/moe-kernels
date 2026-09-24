"""What git tracks under a directory, for the tests that mean the committed tree.

`results/published/` is the one place under `results/` that .gitignore does NOT
exclude, so a directory there that nobody committed is simply untracked: `ls`
sees it, `git status` lists it with `??`, and a test that counts
`PUBLISHED.iterdir()` counts it. Session 4's pod suite and session 5's both
ran on a checkout carrying one (`2026-09-15-nvidia_h200-session3`, left on the
network volume by session 3's publish and missing the `KIND` file the
committed copy carries), and the census and README-count tests failed on it:
15 arms against a README's 14, and an `unknown` verdict nobody declared.

Those tests say what they mean in their own words: "the committed report",
"every published arm has a declared verdict", "the numbers on the page are
pinned to the tree". A published arm is one git tracks. So they read git's
index, where a newly published arm counts from the moment it is `git add`ed,
and an untracked directory is not an arm until it is.

The same holds for the RULERS. A pod's `calibrate` rewrites the TRACKED
`moe/bench/hardware/measured_<card>.yaml` before the end suite runs (session
5's read ridge 152.9 against the committed 151.4), and a test that regenerates
a committed artefact, or checks that committed reports cite "their card's
committed calibration", has to read the committed ruler rather than the
working copy. `committed_copy` materialises the index's version of a tracked
directory for exactly that.

Nothing here falls back to reading the directory when git cannot answer: a
fallback would restore the defect exactly on the box that has it. A checkout
without git fails these tests loudly, which is what the provenance tests in
this suite already require of it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def tracked_files(directory: Path) -> list[Path]:
    """Every file under `directory` that git's index holds, staged or committed."""
    rel = Path(directory).resolve().relative_to(REPO.resolve())
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--", rel.as_posix()],
        capture_output=True, check=True)
    return sorted(REPO / p for p in out.stdout.decode().split("\0") if p)


def tracked_children(directory: Path) -> list[Path]:
    """The directories directly under `directory` that hold a tracked file.

    Sorted, and as paths under `directory` exactly as the caller spelled it, so
    a test that used to iterate `directory.iterdir()` gets the same objects for
    every directory git knows about and none for one it does not.
    """
    directory = Path(directory)
    rel = directory.resolve().relative_to(REPO.resolve())
    names = {path.relative_to(REPO).relative_to(rel).parts[0]
             for path in tracked_files(directory)
             if len(path.relative_to(REPO).relative_to(rel).parts) > 1}
    return sorted(directory / name for name in names)


def committed_copy(directory: Path, dest: Path) -> Path:
    """Write the INDEX's version of every tracked file under `directory` into
    `dest`, same relative paths, and return `dest`.

    The index and not HEAD, so a ruler staged for commit is the one compared
    against, exactly as `tracked_files` counts a staged arm. A working copy
    that differs (a pod's fresh calibration) is never read.
    """
    directory = Path(directory)
    rel = directory.resolve().relative_to(REPO.resolve())
    for path in tracked_files(directory):
        name = path.relative_to(REPO).as_posix()
        blob = subprocess.run(["git", "-C", str(REPO), "show", f":{name}"],
                              capture_output=True, check=True).stdout
        target = Path(dest) / path.relative_to(REPO).relative_to(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    return Path(dest)


def ncu_refusals() -> list[tuple[str, str, str]]:
    """Every ERR_NVGPUCTRPERM refusal a tracked profile log under `profiles/`
    commits, as (the log's repo path, the date its run's own rows carry, the
    card those rows name), read off the log and the rows file it names rather
    than typed anywhere.

    RunPod's ncu record is stated in the chain's help, the runbook, the gaps
    driver's arm text and the counter docs, and it read "one refusal on one
    pod" (2026-09-15) in several of them while `profiles/q2_kernel_names.txt`
    held an earlier one: ncu over the harness's own CLI on an H200, refused,
    with the rows that run wrote dated 2026-08-25. The tests that hold those
    texts to the record take it from here. The 2026-09-15 refusal is committed
    only as prose (docs/FINDINGS.md's retraction), so it is not in this list.
    """
    import csv
    import re

    found = []
    for log in tracked_files(REPO / "profiles"):
        if log.suffix != ".txt":
            continue
        text = log.read_text(errors="replace")
        if "ERR_NVGPUCTRPERM" not in text:
            continue
        rows = re.search(r"wrote \d+ rows -> (\S+\.csv)", text)
        if rows is None:
            raise AssertionError(f"{log.name} names no rows file to date it by")
        with open(REPO / rows.group(1), newline="") as fh:
            row = next(csv.DictReader(fh))
        found.append((log.relative_to(REPO).as_posix(), row["timestamp"][:10],
                      row["gpu_name"]))
    return found
