"""extras.sh's own checks and the pure pieces the scorer reads (owner 2026-11-19: "this is the final run; get as much
data and information as possible"). Pure Python + numpy: importable on the laptop.

    python scripts/prod/extras_lib.py main-finished --results R [--after-main-fail]   # main.sh is done (hard, first)
    python scripts/prod/extras_lib.py commit-switch --repo REPO --results R --expect SHA
    python scripts/prod/extras_lib.py budget --block X1 [--levels L --n N --qwen3-levels Q]  # $EXTRAS_T0, $EXTRAS_BUDGET_S
    python scripts/prod/extras_lib.py inputs --block X2 --results R        # what the block reads from main's run
    python scripts/prod/extras_lib.py disk-need --results R                # P10's need from main's trace sizes
    python scripts/prod/extras_lib.py x1-identical --base R/A1710 --repeat R/extras/A1710r
    python scripts/prod/extras_lib.py push-size --dir D [--dir D2]         # one push against GitHub's limits
    python scripts/prod/extras_lib.py expected-outputs --results R --out F # offline.sh: what the extras measured

MAIN FINISHED (review of the extras M1): main's SUMMARY.txt records 'PASS held-back-check' (main reached its end;
--after-main-fail, the operator's override, lets a main that stopped early through and says so), and no main.sh,
serve_capture.py, locked_r3.py or nsys process is running.

THE COMMIT SWITCH (a documented operator step between main.sh and extras.sh, on the VM): main.sh ran at the registered
commit; extras.sh is NEWER. After main.sh's SUMMARY.txt is complete and before extras.sh:

    cd "$REPO" && git fetch origin && git checkout --detach <extras sha>        # cloned with --repo (origin)
    cd "$REPO" && git fetch <bundle> '+refs/heads/*:refs/remotes/origin/*' && git checkout --detach <extras sha>
                                                                                # cloned from a bundle (no origin)
    PROD_EXTRAS_COMMIT=<extras sha> bash scripts/prod/extras.sh --results <main's results dir>

`commit-switch` holds that: a clean tree (env-preflight P1's rule: `git status --porcelain` empty, untracked files
included), HEAD the expected extras commit, and main's own commit (the one main's env-preflight P1 recorded in
<results>/env-preflight.json) an ancestor of HEAD, so main's code is in this tree's history.

THE BUDGET: one total for every extras block (default 9000 s, extras.sh --budget-s / $PROD_EXTRAS_BUDGET_S), counted
from extras.sh's start ($EXTRAS_T0; a resume starts its own count). Before each block `budget` passes only if the time
spent plus the block's expected seconds (plan_time.extras_plan at extras.sh's own levels and N) fits; else the block's
budget step FAILS (soft) and every step of the block is SKIPPED (kitlib --needs). A step that would start past
$EXTRAS_DEADLINE (start + budget + grace) is SKIPPED by kitlib, and each step's timeout is capped at the time left.

THE NOISE FLOOR (X1, `noise_floor`; review of the extras M8): `x1-identical` holds A1710r's manifest to main's A1710
(levels, N, seed, mode, stream, the prompt body's sha256). When the two runs' (step, layer, T_pad) sequences are
identical, call k of one is call k of the other (`pair_by_step`, asserted) and per tile the run-to-run variation of
kernel-sum on the pairs is reported (|a - b| / mean(a, b): median, p95; signed median of (b - a) / a); when they
differ, no call has a twin and per tile the medians of measured / v8 and measured / roofline are compared across the
runs. Both: the two runs' per-tile medians over all their calls. DESCRIPTIVE.
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

#: what must not be running when the extras start (review of the extras M1): main.sh itself, or anything it leaves
#: on the card (a serving run, a locked replay, an nsys session). pgrep -f extended regexes; the bracket keeps the
#: pattern from matching the pgrep command line itself
BUSY_PATTERNS = (("main.sh", "[m]ain[.]sh"), ("serve_capture", "[s]erve_capture[.]py"),
                 ("locked_r3", "[l]ocked_r3[.]py"), ("nsys", "(^|/)[n]sys( |$)"))


def busy_processes() -> list[str]:
    out = []
    for name, pat in BUSY_PATTERNS:
        rc, txt = sh(["pgrep", "-f", pat])
        if rc == 0 and txt.strip():
            out.append(f"{name} (pid {' '.join(txt.split()[:4])})")
    return out


def main_finished(results, *, after_main_fail: bool = False, running: list[str] | None = None) -> tuple[bool, str]:
    """main.sh is done with `results`: its SUMMARY.txt records 'PASS held-back-check' (main reached its end; the
    operator's --after-main-fail lets a main that stopped early through, said in the reason) and none of
    BUSY_PATTERNS is running (`running`: the busy list, the tests)."""
    summ = Path(results) / "SUMMARY.txt"
    if not summ.is_file():
        return False, f"no {summ}: main.sh has not run in {results}"
    busy = busy_processes() if running is None else list(running)
    if busy:
        return False, f"still running: {', '.join(busy)} (extras.sh starts after main.sh and its children end)"
    lines = [ln for ln in summ.read_text().splitlines() if ln.strip()]
    last = lines[-1] if lines else "(empty)"
    end = any(ln.split()[:2] == ["PASS", "held-back-check"] for ln in lines)
    if end:
        return True, f"main finished (PASS held-back-check); last line: {last[:160]}"
    if after_main_fail:
        return True, (f"OVERRIDE --after-main-fail: main did NOT pass held-back-check (it stopped early); the extras "
                      f"run anyway; main's last line: {last[:160]}")
    return False, (f"main's SUMMARY.txt has no 'PASS held-back-check' (main stopped before its end; last line: "
                   f"{last[:160]}): rerun with --after-main-fail to run the extras anyway")


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
                       "git fetch, git checkout --detach <extras sha>, then extras.sh)")
    # the same rule as env-preflight P1 (git status --porcelain: an untracked file is dirty too)
    rc, st = sh(["git", "-C", str(repo), "status", "--porcelain"])
    if rc != 0:
        return False, f"git status failed: {st[:200]}"
    if st.strip():
        return False, f"the tree is dirty: {st.splitlines()[:3]}"
    rc, head = sh(["git", "-C", str(repo), "rev-parse", "HEAD"])
    rc2, want = sh(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", expect + "^{commit}"])
    if rc != 0 or rc2 != 0:
        return False, f"cannot resolve HEAD or the expected commit {expect} ({head[:80]}; {want[:80]})"
    if head != want:
        return False, f"HEAD {head} is not the expected extras commit {want}: git checkout --detach {expect} first"
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


#: X3's own disk: Qwen3's bf16 weights (GB; plan_time.QWEN3_WEIGHTS_GB) and a margin on every estimate below
QWEN3_WEIGHTS_GB = 61.1
DISK_MARGIN = 1.25
#: the trace dirs (under main's $MOE_HOME/traces/<run>/) each block's captures are sized by
DISK_FROM = {"X1": ("A1710",), "X3": ("gate60", "A1710", "A1410"), "X4": ("A1410",)}


def disk_need_gb(results, *, long_context: bool = True, floor_gb: float = 50.0) -> tuple[float, str]:
    """P10's need for the extras (review of the extras, MINOR), from main's ACTUAL trace sizes (<results>/
    traces.sha256.json, kit_push's manifest): X1 repeats A1710's traces, X3 writes Qwen3's weights plus traces of
    the size of main's gate60, A1710 and A1410 (Qwen3: 48 x 8 vs 27 x 6 ids per token, a smaller step count: taken
    1:1), X4 A1410's; x DISK_MARGIN, at least floor_gb. No manifest: (None, why)."""
    man = Path(results) / "traces.sha256.json"
    if not man.is_file():
        return None, f"no {man}: main's trace sizes unknown"
    files = json.loads(man.read_text()).get("files", {})
    size = defaultdict(float)
    for rel, v in files.items():
        size[rel.split("/", 1)[0]] += float(v.get("bytes", 0))
    blocks = ("X1", "X3", "X4") if long_context else ("X1", "X3")
    gb = {b: sum(size[d] for d in DISK_FROM[b]) / 1e9 for b in blocks}
    gb["X3"] += QWEN3_WEIGHTS_GB
    need = max(float(floor_gb), DISK_MARGIN * sum(gb.values()))
    return need, (f"{need:.0f} GB = {DISK_MARGIN} x (" + " + ".join(f"{b} {v:.1f}" for b, v in gb.items())
                  + f" GB) from main's traces, floor {floor_gb:.0f}")


#: GitHub's limits on one push (the push script xz-compresses or holds back a file over 95 MB; GitHub refuses a
#: push over 2 GB): every X3 capture is pushed on its own, and this check sizes each push first
PUSH_FILE_MAX_MB = 95
PUSH_TOTAL_MAX_MB = 1900


def push_size(dirs, *, file_max_mb: float = PUSH_FILE_MAX_MB, total_max_mb: float = PUSH_TOTAL_MAX_MB) -> tuple[bool, str]:
    """The pushed bytes of `dirs` (nothing PUSH_EXPENDABLE: traces live outside the results tree) against GitHub's
    limits: FAIL when their total passes total_max_mb or a needed file passes file_max_mb (the push would hold it
    back: copy it off the VM by hand)."""
    import fnmatch
    tot, big = 0, []
    for d in dirs:
        for f in sorted(x for x in Path(d).rglob("*") if x.is_file()):
            rel = str(f.relative_to(d))
            if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f.name, pat) for pat in PL.PUSH_EXPENDABLE):
                continue
            n = f.stat().st_size
            tot += n
            if n > file_max_mb * 1e6:
                big.append(f"{f} {n / 1e6:.0f} MB")
    why = f"{tot / 1e6:.0f} MB under {[str(d) for d in dirs]} (limit {total_max_mb:.0f} MB per push)"
    if tot > total_max_mb * 1e6 or big:
        return False, why + (f"; files over {file_max_mb:.0f} MB: {big[:4]}" if big else "") + \
            ": the push may fail or hold back; copy by hand before the terminate"
    return True, why


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

#: what X1 must share with main's A1710 to be its identical repeat (manifest.json keys; review of the extras M8)
IDENTICAL_KEYS = ("levels", "requests_per_level", "seed", "mode", "shared_stream", "prompts.body_sha256")


def _get(doc: dict, dotted: str):
    for k in dotted.split("."):
        doc = (doc or {}).get(k) if isinstance(doc, dict) else None
    return doc


def identical_check(base_manifest: dict, repeat_manifest: dict) -> tuple[bool, str]:
    """A1710r's serve_capture manifest holds main's A1710 levels, N, seed, mode, stream and prompt body sha256."""
    bad = [f"{k}: A1710 {_get(base_manifest, k)!r}, A1710r {_get(repeat_manifest, k)!r}" for k in IDENTICAL_KEYS
           if _get(base_manifest, k) is None or _get(base_manifest, k) != _get(repeat_manifest, k)]
    if bad:
        return False, "the repeat is not identical: " + "; ".join(bad)
    return True, "identical " + ", ".join(f"{k}={_get(base_manifest, k)!r}" for k in IDENTICAL_KEYS[:5]) + \
        f", prompts {str(_get(base_manifest, 'prompts.body_sha256'))[:12]}"


def identical_files(base_dir, repeat_dir) -> tuple[bool, str]:
    docs = []
    for d in (base_dir, repeat_dir):
        f = Path(d) / "manifest.json"
        if not f.is_file():
            return False, f"no {f}"
        docs.append(json.loads(f.read_text()))
    return identical_check(*docs)


def step_sequence(rows: list[dict]) -> list[tuple]:
    """(step, layer, T_pad) of every call in g order: what the engine did, call by call."""
    return [(r.get("step"), int(r["layer"]), int(r["T_pad"])) for r in sorted(rows, key=lambda x: int(x["g"]))]


def pair_by_step(rows_a: list[dict], rows_b: list[dict]) -> list[tuple[dict, dict]] | None:
    """The k-th call of a with the k-th call of b (g order) when the two runs' step sequences are identical, else None.
    Every pair is asserted to share (step, layer, T_pad)."""
    if step_sequence(rows_a) != step_sequence(rows_b):
        return None
    A = sorted(rows_a, key=lambda x: int(x["g"]))
    B = sorted(rows_b, key=lambda x: int(x["g"]))
    pairs = list(zip(A, B, strict=True))
    for a, b in pairs:
        assert (a.get("step"), int(a["layer"]), int(a["T_pad"])) == (b.get("step"), int(b["layer"]), int(b["T_pad"]))
    return pairs


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def _ratio_medians(rows: list[dict], truth: str, pred: str) -> dict:
    acc: dict = defaultdict(list)
    for r in rows:
        t, p = r.get(truth), r.get(pred)
        if t is not None and p:
            acc[r["tile"]].append(float(t) / float(p))
            acc["all"].append(float(t) / float(p))
    return {t: {"n": len(v), "median": float(np.median(v))} for t, v in acc.items()}


def noise_floor(rows_a: list[dict], rows_b: list[dict], truth: str = "kernel_sum_ms", ok=None) -> dict:
    """Run-to-run variation of A1710 and its repeat. PAIRING "step": the step sequences are identical, so call k of a
    is call k of b (pair_by_step) and per tile the paired calls' |a - b| / mean (median, p95) and signed median of
    (b - a) / a. PAIRING "ratio" (the sequences differ, no call has a twin): per tile the median of each call's
    measured / v8 and measured / roofline in each run (rows carry v8_ms and roof_ms), compared across the runs.
    Both: each run's per-tile median of the truth. `ok(row)` keeps a call (score_prod: status ok, no warm-up, on
    lock)."""
    ok = ok or (lambda r: True)
    ka, kb = [r for r in rows_a if ok(r)], [r for r in rows_b if ok(r)]
    pairs = pair_by_step(rows_a, rows_b)
    matched, ratio = {}, {}
    if pairs is not None:
        by: dict = defaultdict(list)
        for a, b in pairs:
            if ok(a) and ok(b) and a.get(truth) is not None and b.get(truth) is not None:
                by[a["tile"]].append((float(a[truth]), float(b[truth])))
                by["all"].append((float(a[truth]), float(b[truth])))
        for t, v in by.items():
            x = np.asarray(v, dtype=float)
            x = x[(x[:, 0] > 0) & (x[:, 1] > 0)]
            if not len(x):
                continue
            rel = np.abs(x[:, 0] - x[:, 1]) / x.mean(axis=1)
            sgn = (x[:, 1] - x[:, 0]) / x[:, 0]
            matched[t] = {"n": int(len(x)), "abs_rel_median_pct": 100 * _pct(rel, 50),
                          "abs_rel_p95_pct": 100 * _pct(rel, 95), "signed_median_pct": 100 * _pct(sgn, 50),
                          "a_median_ms": float(np.median(x[:, 0])), "b_median_ms": float(np.median(x[:, 1]))}
    else:
        for pred in ("v8_ms", "roof_ms"):
            ra, rb = _ratio_medians(ka, truth, pred), _ratio_medians(kb, truth, pred)
            ratio[pred] = {t: {"n_a": ra[t]["n"], "n_b": rb[t]["n"], "a": ra[t]["median"], "b": rb[t]["median"],
                               "delta_pct": 100 * (rb[t]["median"] - ra[t]["median"]) / ra[t]["median"]}
                           for t in ra if t in rb and ra[t]["median"] > 0}
    med = {}
    for name, rows in (("a", ka), ("b", kb)):
        acc: dict = defaultdict(list)
        for r in rows:
            if r.get(truth) is not None:
                acc[r["tile"]].append(float(r[truth]))
        med[name] = {t: {"n": len(v), "median_ms": float(np.median(v))} for t, v in acc.items()}
    per_tile = {t: {"a_ms": med["a"][t]["median_ms"], "b_ms": med["b"][t]["median_ms"], "n_a": med["a"][t]["n"],
                    "n_b": med["b"][t]["n"],
                    "delta_pct": 100 * (med["b"][t]["median_ms"] - med["a"][t]["median_ms"]) / med["a"][t]["median_ms"]}
                for t in med["a"] if t in med["b"] and med["a"][t]["median_ms"] > 0}
    return {"truth": truth, "calls_a": len(rows_a), "calls_b": len(rows_b),
            "pairing": "step" if pairs is not None else "ratio",
            "matched_calls": sum(1 for a, b in (pairs or []) if ok(a) and ok(b)), "matched": matched,
            "ratio_medians": ratio, "per_tile_medians": per_tile,
            "key": ("call k with call k (identical (step, layer, T_pad) sequences)" if pairs is not None else
                    "no call twins (the step sequences differ): per-tile medians of measured / v8 and measured / "
                    "roofline per run")}


# ------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------

def _expected_blocks(long_context: bool = True, levels: str | None = None, n: int | None = None,
                     qwen3_levels: str | None = None) -> dict:
    """The blocks' expected seconds at extras.sh's own serving workload ($PROD_LEVELS, $PROD_REQUESTS_PER_LEVEL,
    $PROD_QWEN3_LEVELS: the budget prices what will run, not the defaults)."""
    import plan_time as PT
    kw = {k: v for k, v in (("levels", levels), ("n", n), ("qwen3_levels", qwen3_levels)) if v is not None}
    return PT.extras_plan(long_context=long_context, **kw)["blocks"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("main-finished")
    m.add_argument("--results", required=True)
    m.add_argument("--after-main-fail", action="store_true",
                   help="the operator's override: run the extras after a main that did not pass held-back-check")
    c = sub.add_parser("commit-switch")
    c.add_argument("--repo", required=True)
    c.add_argument("--results", required=True)
    c.add_argument("--expect", default="")
    b = sub.add_parser("budget")
    b.add_argument("--block", required=True, choices=BLOCKS)
    b.add_argument("--t0", type=float, default=None, help="default $EXTRAS_T0")
    b.add_argument("--budget-s", type=float, default=None, help="default $EXTRAS_BUDGET_S, else 9000")
    b.add_argument("--now", type=float, default=None, help="tests")
    b.add_argument("--levels", default=None, help="ERNIE's levels ($PROD_LEVELS; default plan_time's)")
    b.add_argument("--n", type=int, default=None, help="requests per level ($PROD_REQUESTS_PER_LEVEL)")
    b.add_argument("--qwen3-levels", default=None, help="Qwen3's levels ($PROD_QWEN3_LEVELS)")
    b.add_argument("--no-long-context", action="store_true")
    dn = sub.add_parser("disk-need", help="prints P10's --min-free-gb for the extras from main's trace sizes")
    dn.add_argument("--results", required=True)
    dn.add_argument("--no-long-context", action="store_true")
    dn.add_argument("--default-gb", type=float, default=200.0, help="printed when main's trace manifest is absent")
    ps = sub.add_parser("push-size", help="FAIL when a push of these dirs would pass GitHub's limits")
    ps.add_argument("--dir", action="append", required=True)
    ps.add_argument("--total-max-mb", type=float, default=PUSH_TOTAL_MAX_MB)
    x1 = sub.add_parser("x1-identical", help="A1710r's manifest holds A1710's levels, N, seed, mode, prompts")
    x1.add_argument("--base", required=True, help="main's A1710 run dir")
    x1.add_argument("--repeat", required=True, help="the extras' A1710r run dir")
    i = sub.add_parser("inputs")
    i.add_argument("--block", required=True, choices=BLOCKS)
    i.add_argument("--results", required=True)
    e = sub.add_parser("expected-outputs")
    e.add_argument("--results", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "disk-need":
        need, why = disk_need_gb(a.results, long_context=not a.no_long_context)
        print(f"{need if need is not None else a.default_gb:.0f}")
        print(f"disk-need: {why}" + ("" if need is not None else f" (default {a.default_gb:.0f} GB)"), file=sys.stderr)
        return 0
    if a.cmd == "main-finished":
        ok, why = main_finished(a.results, after_main_fail=a.after_main_fail)
    elif a.cmd == "push-size":
        ok, why = push_size(a.dir, total_max_mb=a.total_max_mb)
    elif a.cmd == "x1-identical":
        ok, why = identical_files(a.base, a.repeat)
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
                               expected=_expected_blocks(not a.no_long_context, a.levels, a.n, a.qwen3_levels))
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
