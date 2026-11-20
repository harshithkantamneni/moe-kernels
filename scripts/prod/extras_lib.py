"""extras.sh's own checks and the pure pieces the scorer reads (owner 2026-11-19: "this is the final run; get as much
data and information as possible"). Pure Python + numpy: importable on the laptop.

    python scripts/prod/extras_lib.py main-finished --results R            # main.sh is done with R (hard, first)
    python scripts/prod/extras_lib.py commit-switch --repo REPO --results R --expect SHA
    python scripts/prod/extras_lib.py budget --block X1                    # $EXTRAS_T0, $EXTRAS_BUDGET_S
    python scripts/prod/extras_lib.py inputs --block X2 --results R        # what the block reads from main's run
    python scripts/prod/extras_lib.py expected-outputs --results R --out F # offline.sh: what the extras measured

THE COMMIT SWITCH (a documented operator step between main.sh and extras.sh, on the VM): main.sh ran at the registered
commit; extras.sh is NEWER. After main.sh's SUMMARY.txt is complete and before extras.sh:

    cd "$REPO" && git fetch origin && git checkout <extras sha>
    PROD_EXTRAS_COMMIT=<extras sha> bash scripts/prod/extras.sh --results <main's results dir>

`commit-switch` holds that: a clean tree, HEAD the expected extras commit, and main's own commit (the one main's
env-preflight P1 recorded in <results>/env-preflight.json) an ancestor of HEAD, so main's code is in this tree's history.

THE BUDGET: one total for every extras block (default 9000 s, extras.sh --budget-s / $PROD_EXTRAS_BUDGET_S), counted
from extras.sh's start ($EXTRAS_T0; a resume starts its own count). Before each block `budget` passes only if the time
spent plus the block's expected seconds (plan_time.extras_plan) fits; else the block's budget step FAILS (soft) and
every step of the block is SKIPPED (kitlib --needs).

THE NOISE FLOOR (X1, `noise_floor`): main's A1710 and the extras' identical repeat A1710r, calls matched on (layer,
T_pad, counts digest), the k-th occurrence of a key in one run paired with the k-th in the other; per tile the
run-to-run variation of kernel-sum on the matched calls (|a - b| / mean(a, b): median, p95; signed median of (b - a) /
a) and the two runs' per-tile medians over all their calls. DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prodlib as PL  # noqa: E402

BLOCKS = ("X1", "X2", "X3", "X4")
BLOCK_WHAT = {"X1": "noise floor: a repeat of A1710, identical", "X2": "ncu DRAM bytes on a replay subsample",
              "X3": "second blind model Qwen3-30B-A3B", "X4": "long-context prefill (ERNIE, 1410 lock)"}
DEFAULT_BUDGET_S = 9000
QWEN3 = "qwen3-30b-a3b"
SHAREGPT = "data/ShareGPT_V3_unfiltered_cleaned_split.json"
#: what each block reads from main's results tree (relative paths; `a|b`: either)
INPUTS = {"X1": ("prompts.jsonl", "SUMMARY.txt"),
          "X2": ("sample-decode/sample.json", "sample-prefill/sample.json"),
          "X3": ("sag/par.json", "replay-probe-decode/replay_wall.json", "replay-probe-prefill/replay_wall.json"),
          "X4": ()}


def sh(cmd: list[str], timeout: float = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "").strip() + (("\n" + p.stderr.strip()) if p.stderr.strip() else "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"


# ------------------------------------------------------------------------------------------------------------
# the hard checks before anything runs
# ------------------------------------------------------------------------------------------------------------

def main_finished(results, *, running: bool | None = None) -> tuple[bool, str]:
    """main.sh is done with `results`: its SUMMARY.txt exists and no main.sh process is alive (`running` overrides
    the process check in the tests). Reports main's last recorded step and whether it reached its end."""
    summ = Path(results) / "SUMMARY.txt"
    if not summ.is_file():
        return False, f"no {summ}: main.sh has not run in {results}"
    if running is None:
        rc, out = sh(["pgrep", "-f", "[s]cripts/prod/main[.]sh"])
        running = rc == 0 and bool(out.strip())
    if running:
        return False, "a main.sh process is still running: extras.sh starts after main.sh has finished"
    lines = [ln for ln in summ.read_text().splitlines() if ln.strip()]
    last = lines[-1] if lines else "(empty)"
    end = any(ln.split()[1:2] == ["held-back-check"] for ln in lines if ln.startswith(("PASS ", "FAIL ")))
    return True, (f"main finished ({'reached held-back-check' if end else 'stopped before its end'}); last line: "
                  f"{last[:160]}")


def main_commit(results) -> str | None:
    """main's HEAD as its env-preflight P1 recorded it (<results>/env-preflight.json: 'clean at <sha> ...')."""
    p = Path(results) / "env-preflight.json"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except ValueError:
        return None
    for c in doc.get("checks", []):
        if c.get("id") == "P1":
            m = re.search(r"\b([0-9a-f]{40})\b", str(c.get("detail", "")))
            return m.group(1) if m else None
    return None


def commit_switch(repo, results, expect: str) -> tuple[bool, str]:
    """The tree is clean, HEAD is the expected extras commit, and main's commit is HEAD or an ancestor of it."""
    if not expect:
        return False, ("no expected extras commit: pass --expect-commit or $PROD_EXTRAS_COMMIT (the documented switch: "
                       "git fetch && git checkout <extras sha>, then extras.sh)")
    rc, st = sh(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"])
    if rc != 0:
        return False, f"git status failed: {st[:200]}"
    if st.strip():
        return False, f"the tree is dirty: {st.splitlines()[:3]}"
    rc, head = sh(["git", "-C", str(repo), "rev-parse", "HEAD"])
    rc2, want = sh(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", expect + "^{commit}"])
    if rc != 0 or rc2 != 0:
        return False, f"cannot resolve HEAD or the expected commit {expect} ({head[:80]}; {want[:80]})"
    if head != want:
        return False, f"HEAD {head} is not the expected extras commit {want}: git checkout {expect} first"
    mc = main_commit(results)
    if mc is None:
        return False, f"{Path(results) / 'env-preflight.json'} records no main commit (P1)"
    if mc == head:
        return True, f"clean at {head}, the same commit main ran at"
    rc, _ = sh(["git", "-C", str(repo), "merge-base", "--is-ancestor", mc, head])
    if rc != 0:
        return False, f"main ran at {mc}, which is not an ancestor of HEAD {head}"
    return True, f"clean at {head}, newer than main's {mc} (an ancestor)"


def budget_check(block: str, *, t0: float, budget_s: float, now: float, expected: dict) -> tuple[bool, str]:
    """PASS when the time spent since t0 plus the block's expected seconds is within the budget."""
    spent = max(0.0, float(now) - float(t0))
    need = float(expected[block])
    if spent + need <= float(budget_s):
        return True, f"{block}: spent {spent:.0f} s + expected {need:.0f} s <= budget {budget_s:.0f} s"
    return False, (f"BUDGET {block} ({BLOCK_WHAT[block]}) skipped: spent {spent:.0f} s + expected {need:.0f} s > "
                   f"budget {budget_s:.0f} s")


def inputs_check(block: str, results) -> tuple[bool, str]:
    R = Path(results)
    miss = [rel for rel in INPUTS[block] if not any((R / a).exists() for a in rel.split("|"))]
    return (not miss), (f"{block} reads {list(INPUTS[block]) or 'nothing'} from main's run" if not miss else
                        f"{block}: main's run lacks {miss}")


# ------------------------------------------------------------------------------------------------------------
# offline: what the extras measured (their own SUMMARY.txt)
# ------------------------------------------------------------------------------------------------------------

#: per block: the step whose PASS means the output the laptop scores, and that output (relative to <results>/extras)
EXTRAS_OUTPUTS = (("X1", "X1-extract", "A1710r/calls.jsonl|A1710r/calls.shards.json"),
                  ("X2", "X2-ncu-decode", "ncu-decode/ncu_bytes.json"),
                  ("X2", "X2-ncu-prefill", "ncu-prefill/ncu_bytes.json"),
                  ("X2", "X2-subsample", "ncu-sample/sample.json"),
                  ("X3", "X3-A1710-extract", "qwen3/A1710/calls.jsonl|qwen3/A1710/calls.shards.json"),
                  ("X3", "X3-A1410-extract", "qwen3/A1410/calls.jsonl|qwen3/A1410/calls.shards.json"),
                  ("X3", "X3-frontier-predict", "qwen3/frontier/predictions/index.json"),
                  ("X4", "X4-extract", "LC1410/calls.jsonl|LC1410/calls.shards.json"))


def expected_outputs(results) -> dict:
    """{"passed": [steps], "have": {step: rel}, "missing": [why]} from <results>/extras/SUMMARY.txt: every step that
    PASSED must have left its output (or its .xz)."""
    EX = Path(results) / "extras"
    summ = EX / "SUMMARY.txt"
    if not summ.is_file():
        return {"passed": [], "have": {}, "missing": [], "extras": False}
    passed = {ln.split()[1] for ln in summ.read_text().splitlines() if ln.startswith("PASS ") and len(ln.split()) > 1}
    have, missing = {}, []
    for _blk, st, rel in EXTRAS_OUTPUTS:
        if st not in passed:
            continue
        if any((EX / a).exists() or (EX / (a + ".xz")).exists() for a in rel.split("|")):
            have[st] = rel
        else:
            missing.append(f"extras/{rel}: {st} PASSED on the rental, the file is missing")
    for b in ("decode", "prefill"):
        if f"X3-replay-{b}" in passed and not any((EX / "qwen3" / f"replay-{b}").rglob("cells.csv*")):
            missing.append(f"extras/qwen3/replay-{b}: X3-replay-{b} PASSED, no cells.csv")
    return {"passed": sorted(passed), "have": have, "missing": missing, "extras": True}


# ------------------------------------------------------------------------------------------------------------
# the noise floor (pure; score_prod reads it)
# ------------------------------------------------------------------------------------------------------------

def match_key(r: dict) -> tuple:
    return (int(r["layer"]), int(r["T_pad"]), str(r["counts_digest"]))


def match_calls(rows_a: list[dict], rows_b: list[dict], truth: str = "kernel_sum_ms") -> list[tuple[dict, dict]]:
    """Pairs (a, b) with equal (layer, T_pad, counts digest): the k-th occurrence of a key in a (in g order) with the
    k-th in b; calls without the truth are left out."""
    def occ(rows):
        d: dict = defaultdict(list)
        for r in sorted(rows, key=lambda x: int(x["g"])):
            if r.get(truth) is not None:
                d[match_key(r)].append(r)
        return d
    A, B = occ(rows_a), occ(rows_b)
    out = []
    for k, ra in A.items():
        for a, b in zip(ra, B.get(k, []), strict=False):
            out.append((a, b))
    return out


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def noise_floor(rows_a: list[dict], rows_b: list[dict], truth: str = "kernel_sum_ms") -> dict:
    """Per tile (and all): the matched calls' run-to-run variation, and each run's per-tile median over all calls."""
    pairs = match_calls(rows_a, rows_b, truth)
    by: dict = defaultdict(list)
    for a, b in pairs:
        by[a["tile"]].append((float(a[truth]), float(b[truth])))
        by["all"].append((float(a[truth]), float(b[truth])))
    matched = {}
    for t, v in by.items():
        x = np.asarray(v, dtype=float)
        ok = (x[:, 0] > 0) & (x[:, 1] > 0)
        x = x[ok]
        if not len(x):
            continue
        rel = np.abs(x[:, 0] - x[:, 1]) / x.mean(axis=1)
        sgn = (x[:, 1] - x[:, 0]) / x[:, 0]
        matched[t] = {"n": int(len(x)), "abs_rel_median_pct": 100 * _pct(rel, 50), "abs_rel_p95_pct": 100 * _pct(rel, 95),
                      "signed_median_pct": 100 * _pct(sgn, 50), "a_median_ms": float(np.median(x[:, 0])),
                      "b_median_ms": float(np.median(x[:, 1]))}
    med = {}
    for name, rows in (("a", rows_a), ("b", rows_b)):
        acc: dict = defaultdict(list)
        for r in rows:
            if r.get(truth) is not None:
                acc[r["tile"]].append(float(r[truth]))
        med[name] = {t: {"n": len(v), "median_ms": float(np.median(v))} for t, v in acc.items()}
    per_tile = {t: {"a_ms": med["a"][t]["median_ms"], "b_ms": med["b"][t]["median_ms"], "n_a": med["a"][t]["n"],
                    "n_b": med["b"][t]["n"],
                    "delta_pct": 100 * (med["b"][t]["median_ms"] - med["a"][t]["median_ms"]) / med["a"][t]["median_ms"]}
                for t in med["a"] if t in med["b"] and med["a"][t]["median_ms"] > 0}
    return {"truth": truth, "calls_a": len(rows_a), "calls_b": len(rows_b), "matched_calls": len(pairs),
            "matched": matched, "per_tile_medians": per_tile,
            "key": "(layer, T_pad, counts digest), k-th occurrence with k-th occurrence"}


# ------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------

def _expected_blocks(long_context: bool = True) -> dict:
    import plan_time as PT
    return PT.extras_plan(long_context=long_context)["blocks"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("main-finished")
    m.add_argument("--results", required=True)
    c = sub.add_parser("commit-switch")
    c.add_argument("--repo", required=True)
    c.add_argument("--results", required=True)
    c.add_argument("--expect", default="")
    b = sub.add_parser("budget")
    b.add_argument("--block", required=True, choices=BLOCKS)
    b.add_argument("--t0", type=float, default=None, help="default $EXTRAS_T0")
    b.add_argument("--budget-s", type=float, default=None, help="default $EXTRAS_BUDGET_S, else 9000")
    b.add_argument("--now", type=float, default=None, help="tests")
    i = sub.add_parser("inputs")
    i.add_argument("--block", required=True, choices=BLOCKS)
    i.add_argument("--results", required=True)
    e = sub.add_parser("expected-outputs")
    e.add_argument("--results", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "main-finished":
        ok, why = main_finished(a.results)
    elif a.cmd == "commit-switch":
        ok, why = commit_switch(a.repo, a.results, a.expect)
    elif a.cmd == "budget":
        import time
        t0 = a.t0 if a.t0 is not None else float(os.environ.get("EXTRAS_T0") or 0)
        if not t0:
            print("FAIL budget: no start time ($EXTRAS_T0)")
            return 3
        bud = a.budget_s if a.budget_s is not None else float(os.environ.get("EXTRAS_BUDGET_S") or DEFAULT_BUDGET_S)
        ok, why = budget_check(a.block, t0=t0, budget_s=bud, now=a.now if a.now is not None else time.time(),
                               expected=_expected_blocks())
    elif a.cmd == "inputs":
        ok, why = inputs_check(a.block, a.results)
    else:
        got = expected_outputs(a.results)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(got, indent=1) + "\n")
        ok = not got["missing"]
        why = ("no extras run (no extras/SUMMARY.txt)" if not got["extras"] else
               f"{len(got['have'])} extras outputs present" if ok else "; ".join(got["missing"]))
    print(f"{'PASS' if ok else 'FAIL'} {a.cmd}: {why}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
