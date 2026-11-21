"""scripts/prod/{smoke,main,frontier_setup}.sh: shell syntax, and the --dry-run plans (every step printed, nothing
created, the ERNIE 60 s gate before any long run, the locks per branch)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROD = REPO / "scripts" / "prod"


@pytest.mark.parametrize("name", ["smoke.sh", "main.sh", "kitlib.sh", "frontier_setup.sh", "sag1410.sh", "offline.sh",
                                  "pull_traces.sh", "push_chunks.sh"])
def test_shell_syntax(name):
    assert subprocess.run(["bash", "-n", str(PROD / name)]).returncode == 0


def _dry(name, tmp_path, *extra):
    env = dict(os.environ, PY_VLLM="python3", HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
    p = subprocess.run(["bash", str(PROD / name), "--dry-run", "--results", str(tmp_path / "res"), *extra],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=60)
    assert p.returncode == 0, p.stderr
    assert not (tmp_path / "res").exists() and not (tmp_path / "moe").exists()
    return [line for line in p.stdout.splitlines() if line.startswith("STEP ")]


def _soft(step: str) -> bool:
    return step.split()[2] == "[soft]"


def _cap(step: str) -> int:
    m = re.search(r" cap=(\d+) ", step)
    assert m, step
    return int(m.group(1))


def test_smoke_dry_run_lists_the_smoke_sequence(tmp_path):
    steps = _dry("smoke.sh", tmp_path)
    names = [s.split()[1] for s in steps]
    for want in ("env", "price-check", "capture-graphs-nsys", "nsys-export", "extract-gate-A", "sequences-A",
                 "capture-eager-torchprof", "extract-gate-B", "sag1410", "replay-decode", "replay-prefill",
                 "replay-wall", "replay-check-decode", "replay-check-prefill", "frontier-setup",
                 "frontier-timer-prefill", "frontier-timer-export-decode", "frontier-train", "frontier-predict", "score"):
        assert want in names, want
    by = dict(zip(names, steps, strict=True))
    # SMOKE-FIXES 7: the nsys replay passes run under locked_r3 at the branch clock, with cell marks
    for b, lock in (("decode", 1710), ("prefill", 1410)):
        assert "locked_r3.py" in by[f"replay-nsys-{b}"] and f"--locks {lock} " in by[f"replay-nsys-{b}"]
        assert "--cell-marks" in by[f"replay-nsys-{b}"] and f"--branch {b}" in by[f"replay-check-{b}"]
    assert "--replay-check decode=" in by["score"] and not any("fidelity" in n for n in names)
    # review 1: the prefill replay, the sag par, the sequences and the replay-vs-serving check are HARD in the smoke
    for hard in ("replay-prefill", "sag1410", "sequences-A", "replay-check-prefill", "price-check"):
        assert not _soft(by[hard]), hard
    assert "--min-big-T 8256" in by["replay-sample-prefill"]
    assert "--sag" in by["score"] and "sag/par.json" in by["score"]
    cap = next(s for s in steps if s.split()[1] == "capture-graphs-nsys")
    assert "--cuda-graph-trace=node" in cap and "olmoe-1b-7b" in cap and "--max-seconds 60" in cap
    assert "--allow-unpinned" in next(s for s in steps if s.split()[1] == "prompts-fetch")
    assert all(_cap(s) > 0 for s in steps)                      # review 2, MAJOR C: every step has its cap
    assert "--model olmoe-1b-7b" in by["replay-wall"]           # MAJOR B: the wall names its model
    # review 2026-11-19: the early probe (hard) right after the gate; every locked_r3 call names its run cap
    for n in ("frontier-setup", "frontier-grid-check", "frontier-timer-probe"):
        assert not _soft(by[n]), n
    assert names.index("sequences-A") < names.index("frontier-timer-probe") < names.index("sag1410")
    assert 'PROBE_POINTS="${PROD_FRONTIER_PROBE_POINTS:-64,192,256}"' in (PROD / "smoke.sh").read_text()
    for n, st in by.items():
        if "locked_r3.py" in st:
            m = re.search(r"--run-cap-s (\d+) ", st)
            assert m and int(m.group(1)) < _cap(st), n
    assert "--truth with_fills " in by["frontier-csv-fills"] and "--models cli-fills=" in by["frontier-predict"]
    assert _cap(by["env-preflight"]) == 600


def test_main_dry_run_gates_first_and_locks_each_branch(tmp_path):
    steps = _dry("main.sh", tmp_path)
    names = [s.split()[1] for s in steps]
    by = dict(zip(names, steps, strict=True))
    assert names.index("gate60-extract") < names.index("A1710-capture") < names.index("A1410-capture")
    # review 1 MAJOR 5: gate60, (the sag par), A1710, A1410, replay C, Frontier, then the secondary runs, all soft
    order = ["gate60-extract", "sag1410", "A1710-capture", "A1410-capture", "replay-decode", "replay-prefill",
             "frontier-timer-decode", "frontier-train", "Adefault-capture", "Aon-capture", "Beager-capture",
             "replay-offset-nsys-decode", "frontier-predict"]
    assert [names.index(n) for n in order] == sorted(names.index(n) for n in order)
    for n in ("Adefault-capture", "Adefault-extract", "Aon-lock", "Aon-capture", "Aon-extract", "Beager-capture",
              "Beager-extract", "replay-offset-nsys-decode", "replay-offset-check-prefill", "frontier-timer-prefill",
              "frontier-train", "frontier-predict"):
        assert _soft(by[n]), n
    for n in ("A1710-capture", "A1410-extract", "replay-prefill", "sag1410", "price-check", "gate60-sequences",
              "revision-pinned", "replay-wall-present"):
        assert not _soft(by[n]), n
    # review 1 MAJOR 7: no CPU scoring on the rental (owner 2026-11-19: Frontier's batch predict runs on the VM, last,
    # after every run it predicts); the replay budget reads the smoke's walls
    assert "score" not in names and names.index("frontier-predict") > names.index("Aon-extract")
    for lab in ("A1710", "A1410", "Adefault", "Aon", "Beager"):
        assert f"--calls {lab}=" in by["frontier-predict"], lab
    assert "--models cli=" in by["frontier-predict"] and "--models simpath=" in by["frontier-predict"]
    assert "--wall " in by["replay-probe-sample-decode"] and "--wall " in by["replay-probe-sample-prefill"]
    assert all("--require-pinned" in by[n] for n in ("gate60-capture", "A1710-capture", "A1410-capture"))
    assert all("--allow-unpinned" not in s for s in steps)
    assert "kit_lock 1710" in steps[names.index("A1710-lock")] and "kit_lock 1410" in steps[names.index("A1410-lock")]
    rd = steps[names.index("replay-decode")]
    rp = steps[names.index("replay-prefill")]
    assert "--locks 1710" in rd and "--branch decode" in rd and "--locks 1410" in rp and "--branch prefill" in rp
    aon = steps[names.index("Aon-capture")]
    assert "--shared-stream on" in aon
    assert re.search(r"--shared-stream off", steps[names.index("A1710-capture")])
    assert "--torch-profiler" in steps[names.index("Beager-capture")]
    # owner decision 2026-11-19: Frontier's profiler cannot run on the GH200; no fidelity / variant arm, no Frontier
    # profiler; its FULL grid timed by the lightweight timer (no pages, no budget), one pass per branch at its clock
    assert not any("variant" in n or "fidelity" in n or "profile-" in n or n.startswith("graphmode") for n in names)
    assert not any(n.startswith("frontier-page") for n in names) and "PROD_FRONTIER_BUDGET_S" not in \
        (PROD / "main.sh").read_text()
    assert "--arms primary " in by["frontier-setup"]
    for b, lock in (("decode", 1710), ("prefill", 1410)):
        assert by[f"frontier-timer-lock-{b}"].endswith(f"kit_lock {lock} ") and _soft(by[f"frontier-timer-lock-{b}"])
        assert f"fr_timer {b} {lock}" in by[f"frontier-timer-{b}"] and _soft(by[f"frontier-timer-{b}"])
        assert f"needs=frontier-timer-lock-{b} " in by[f"frontier-timer-{b}"]
        assert names.index(f"frontier-timer-lock-{b}") < names.index(f"frontier-timer-{b}") < \
            names.index(f"frontier-timer-export-{b}") < names.index("frontier-points")
    fr = (PROD / "main.sh").read_text().split("fr_timer() {", 1)[1].split("\n}", 1)[0]
    assert "frontier_op.py\" timer" in fr and "--cuda-graph-trace" not in fr and "--max-tokens 16384" in fr
    tr = f"{tmp_path}/moe/traces/res"                    # review m6: raw traces outside the pushed results tree
    assert f"fpoints --model ernie-4.5-21b-a3b --timer {tmp_path}/res/frontier/timer-decode --timer-sqlite " \
        f"{tr}/frontier/timer-decode/timer.sqlite --timer {tmp_path}/res/frontier/timer-prefill --timer-sqlite " \
        f"{tr}/frontier/timer-prefill/timer.sqlite --max-tokens 16384 " in by["frontier-points"]
    for n in ("frontier-csv", "frontier-train", "frontier-holdout", "frontier-simpath", "frontier-csv-fills",
              "frontier-train-fills", "frontier-csv-partial", "frontier-train-partial"):
        assert f"--src {tmp_path}/moe/frontier/src" in by[n] and _soft(by[n]), n
    assert "fpoints" in by["frontier-points"] and "csv" in by["frontier-csv"].split("frontier_op.py")[1]
    opt = _dry("main.sh", tmp_path / "x", "--graphmode-overhead")
    onames = [x.split()[1] for x in opt]
    oby = dict(zip(onames, opt, strict=True))
    assert "--cuda-graph-trace=graph" in oby["graphmode-capture"] and _soft(oby["graphmode-capture"])
    for n in names:
        if not n.startswith("push-"):
            assert oby[n] == by[n].replace(str(tmp_path), str(tmp_path / "x")), n
    for gone in ("--frontier-fidelity", "--frontier-all-dists"):
        env = dict(os.environ, PY_VLLM="python3", HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
        q = subprocess.run(["bash", str(PROD / "main.sh"), "--dry-run", gone], capture_output=True, text=True, env=env,
                           cwd=str(REPO), timeout=60)
        assert q.returncode == 2 and "unknown option" in q.stderr, gone
    # review 2, MAJOR C: every step carries its registered cap; the long ones above their own budgets
    caps = {n: _cap(by[n]) for n in names}
    assert all(c > 0 for c in caps.values())
    assert caps["sag1410"] > 75 * 60 and caps["replay-decode"] > 2400 and caps["replay-prefill"] > 2400
    assert caps["A1710-capture"] == caps["A1410-capture"] == 10800 and caps["held-back-check"] == 120
    # MINOR 6: each capture / export / extract needs the step before it (a failed lock SKIPS the capture)
    for lab, pre in (("A1710", "A1710-lock"), ("Aon", "Aon-lock"), ("Adefault", "Adefault-unlock")):
        assert f"needs={pre} " in by[f"{lab}-capture"]
        assert f"needs={lab}-capture " in by[f"{lab}-export"] and f"needs={lab}-export " in by[f"{lab}-extract"]
    # SMOKE-FIXES 7: the replay under nsys per branch, at its branch clock under locked_r3 (no bare kit_lock)
    for b, lock, src in (("decode", 1710, "A1710"), ("prefill", 1410, "A1410")):
        nb = by[f"replay-offset-nsys-{b}"]
        assert "locked_r3.py" in nb and f"--locks {lock} " in nb and f"--branch {b} " in nb and "--cell-marks" in nb
        assert f"needs=replay-offset-nsys-{b} " in by[f"replay-offset-export-{b}"]
        ck = by[f"replay-offset-check-{b}"]
        assert f"needs=replay-offset-export-{b} " in ck and f"--serving {tmp_path}/res/{src} " in ck
    assert "replay-offset-lock" not in names
    # MAJOR B: per branch a probe, its own ERNIE walls, then the rest sized by them inside the fixed budget
    for b in ("decode", "prefill"):
        seq = [f"replay-probe-sample-{b}", f"replay-probe-{b}", f"replay-probe-wall-{b}", f"replay-sample-{b}",
               f"replay-{b}"]
        assert [names.index(n) for n in seq] == sorted(names.index(n) for n in seq)
        assert "--per-tile 2" in by[f"replay-probe-sample-{b}"] and "--wall " in by[f"replay-probe-sample-{b}"]
        assert "--model ernie-4.5-21b-a3b" in by[f"replay-probe-wall-{b}"]
        assert f"--probe-wall {tmp_path}/res/replay-probe-{b}/replay_wall.json" in by[f"replay-sample-{b}"]
        assert "--wall " not in by[f"replay-sample-{b}"]
    # review 2026-11-19 M1: Frontier's default routing; m4: the two truths; M3: the partial arm (no needs on the
    # failed points step), trained only when csv --partial wrote one; every arm predicted when its trainer passed
    assert "--truth kernel_sum " in by["frontier-csv"] and "--truth with_fills " in by["frontier-csv-fills"]
    assert "--partial " in by["frontier-csv-partial"] and "needs=" not in by["frontier-csv-partial"]
    assert f"--only-if-partial {tmp_path}/res/frontier/profile-partial/csv_meta.json" in by["frontier-train-partial"]
    assert "--arm-label cli-partial " in by["frontier-train-partial"] and "--arm-label cli-fills " in by["frontier-train-fills"]
    for arm in ("cli", "simpath", "cli-fills", "cli-partial"):
        assert f"--models {arm}=" in by["frontier-predict"], arm
    assert "needs=" not in by["frontier-predict"] and "cli-balanced" not in by["frontier-predict"]
    assert 'FR_ROUTING="${PROD_FRONTIER_ROUTING:-uniform}"' in (PROD / "main.sh").read_text()
    assert not any("balanced" in n for n in names)
    # the EARLY PROBE (hard) right after the gate, before every expensive block: Frontier's venv, its grid-check
    for n in ("frontier-setup", "frontier-grid-check", "frontier-timer-probe"):
        assert not _soft(by[n]), n
    assert names.index("gate60-sequences") < names.index("frontier-setup") < names.index("frontier-grid-check") \
        < names.index("frontier-timer-probe") < names.index("sag1410") < names.index("A1710-capture")
    assert by["frontier-timer-probe"].rstrip().endswith("fr_probe") and _cap(by["frontier-timer-probe"]) <= 900
    pr = (PROD / "main.sh").read_text().split("fr_probe() {", 1)[1].split("\n}", 1)[0]
    for want in ('--grid-points "$PROBE_POINTS"', "fpoints", " csv ", "train --narrow", "predict", "--points",
                 "kit_lock", "$PT/timer-$b/timer.sqlite"):
        assert want in pr, want
    # the probe points come from the model (prodlib.py model-info; ERNIE's are 64,1024,8192), an env override first
    assert 'PROBE_POINTS="${PROD_FRONTIER_PROBE_POINTS:-$SPEC_PROBE_POINTS}"' in (PROD / "main.sh").read_text()
    sys.path.insert(0, str(PROD))
    import prodlib as PL
    assert "SPEC_PROBE_POINTS=64,1024,8192" in PL.shell_model_info("ernie-4.5-21b-a3b")
    # review M2: every locked_r3 call names its run cap, below its step's cap
    runcaps = {}
    for n, st in by.items():
        if "locked_r3.py" in st:
            m = re.search(r"--run-cap-s (\d+) ", st)
            assert m, n
            runcaps[n] = int(m.group(1))
            assert runcaps[n] < _cap(st), n
    assert runcaps == {"replay-probe-decode": 1700, "replay-decode": 5000, "replay-probe-prefill": 1700,
                       "replay-prefill": 5000, "replay-offset-nsys-decode": 1700, "replay-offset-nsys-prefill": 1700}
    # review m6: no step writes a raw trace into the results tree
    for n, st in by.items():
        for tok in st.split():
            for path in tok.removeprefix("out=").replace("|", ",").split(","):
                if path.endswith((".nsys-rep", ".sqlite")):
                    assert path.startswith(f"{tr}/"), (n, path)
    for n in ("gate60-export", "A1710-export", "replay-offset-export-decode", "frontier-timer-export-prefill"):
        assert f"out={tr}/" in by[n], n
    # the opt-in balanced arm: its own passes, CSV and trainer, predicted as cli-balanced
    bal = _dry("main.sh", tmp_path / "bal", "--frontier-balanced-arm")
    bnames = [x.split()[1] for x in bal]
    bby = dict(zip(bnames, bal, strict=True))
    for n in ("frontier-balanced-lock-decode", "frontier-balanced-decode", "frontier-balanced-export-prefill",
              "frontier-balanced-points", "frontier-balanced-csv", "frontier-balanced-train"):
        assert n in bnames and (_soft(bby[n]) or n.endswith("unlock")), n
    assert "fr_timer prefill 1410 balanced balanced-prefill" in bby["frontier-balanced-prefill"]
    assert "--models cli-balanced=" in bby["frontier-predict"] and "--arm-label cli-balanced " in bby["frontier-balanced-train"]
    # MAJOR A: the last step is the hard held-back check, after the final push
    assert names[-1] == "held-back-check" and not _soft(by["held-back-check"])
    assert names.index("push-final") < names.index("held-back-check")


def test_offline_dry_run_scores_the_vms_predictions_on_the_laptop(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    p = subprocess.run(["bash", str(PROD / "offline.sh"), "--dry-run", "--results", str(tmp_path / "res")],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=60)
    assert p.returncode == 0, p.stderr
    steps = [x for x in p.stdout.splitlines() if x.startswith("STEP ")]
    names = [x.split()[1] for x in steps]
    assert names[-1] == "score" and not any(n.startswith("frontier-") for n in names)
    assert not (tmp_path / "res").exists()
    sc = steps[-1]
    assert "--sag" in sc and "--workers 4" in sc and "ernie-4.5-21b-a3b" in sc
    assert f"--frontier {tmp_path}/res/frontier/predictions " in sc
    assert f"--replay-check decode={tmp_path}/res/replay-offset-decode/replay_check.json" in sc
    # review 2, MAJOR A: the fetched tree is decompressed and checked for every expected output before anything else
    assert names[:2] == ["decompress", "expected-outputs"] and not _soft(steps[0]) and not _soft(steps[1])
    assert all(_cap(x) > 0 for x in steps)


def _kit(tmp_path, body: str, **env_extra):
    """Run a bash script that sources kitlib.sh for real (KIT_DRY 0) in tmp_path; returns (proc, SUMMARY lines)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "seq.sh"
    script.write_text("set -euo pipefail\n. " + str(PROD / "kitlib.sh") + "\nkit_init " + str(tmp_path / "r")
                      + "\n" + body + "\nkit_summary\n")
    env = dict(os.environ, KIT_KILL_AFTER="2", **env_extra)
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    summ = tmp_path / "r" / "SUMMARY.txt"
    return p, (summ.read_text().splitlines() if summ.exists() else [])


def test_a_step_past_its_cap_is_killed_soft_goes_on_and_hard_stops(tmp_path):
    """review 2, MAJOR C: a hung step is killed at its cap; soft: dropped and the next step runs; hard: FAIL, stop."""
    p, summ = _kit(tmp_path, "step hang --soft --cap 1 -- sleep 30\nstep after --cap 10 -- true\n"
                             "step hang2 --cap 1 -- sleep 30\nstep never --cap 10 -- true")
    assert summ[0].startswith("FAIL (soft, dropped) hang: exit 124: TIMEOUT")
    assert summ[1].startswith("PASS after") and summ[2].startswith("FAIL hang2: exit 124: TIMEOUT")
    assert len(summ) == 3 and p.returncode == 124
    # a step with no cap is refused, dry run included
    p, _ = _kit(tmp_path / "b", "step nocap -- true")
    assert p.returncode == 2 and "no registered --cap" in p.stderr
    p = subprocess.run(["bash", "-c", f". {PROD / 'kitlib.sh'}; KIT_DRY=1; step nocap -- true"], capture_output=True,
                       text=True, timeout=60)
    assert p.returncode == 2


def test_a_step_whose_prerequisite_failed_is_skipped_not_run(tmp_path):
    """review 2, MINOR 6: a soft lock that fails SKIPS the capture that needs it (and what needs the capture). The
    failing step prints no FAIL line: its reason is the log's tail, and the sequence goes on (set -e, pipefail)."""
    flag = tmp_path / "captured"
    p, summ = _kit(tmp_path, f"step lk --soft --cap 10 -- false\nstep cap --soft --cap 10 --needs lk -- touch {flag}\n"
                             "step ex --soft --cap 10 --needs cap -- true\nstep other --soft --cap 10 --needs nope -- true")
    assert p.returncode == 0 and not flag.exists()
    assert summ[1] == "SKIPPED cap: needs lk (failed or skipped)" and summ[2] == "SKIPPED ex: needs cap (failed or skipped)"
    assert summ[3].startswith("PASS other")


def test_shell_function_steps_run_under_the_timeout(tmp_path):
    p, summ = _kit(tmp_path, "myfn() { echo \"PASS fn $1 $NSYS\"; }\nNSYS=/x/nsys\nstep f --cap 10 -- myfn arg")
    assert p.returncode == 0 and summ[0].startswith("PASS f")
    assert "PASS fn arg /x/nsys" in (tmp_path / "r" / "logs" / "f.log").read_text()


def test_the_exit_trap_pushes_with_the_exit_code_after_a_hard_fail(tmp_path):
    """review 2, MAJOR A: whatever stops main.sh, the EXIT trap pushes once more (--final RC), soft."""
    stub = tmp_path / "push_stub.sh"
    got = tmp_path / "pushed.txt"
    stub.write_text(f'printf "%s\\n" "$@" > {got}\n')
    p, summ = _kit(tmp_path, "kit_trap_unlock --push\nstep ok --cap 10 -- true\nstep boom --cap 10 -- bash -c 'exit 5'\n"
                             "step never --cap 10 -- true", KIT_PUSH_SCRIPT=str(stub))
    assert p.returncode == 5 and summ[-1].startswith("FAIL boom: exit 5")
    args = got.read_text().split("\n")
    assert args[0] == "push" and "--final" in args and args[args.index("--final") + 1] == "5"
    # a failing push never changes the exit code
    stub.write_text("exit 9\n")
    p, _ = _kit(tmp_path / "c", "kit_trap_unlock --push\nstep boom --cap 10 -- bash -c 'exit 6'",
                KIT_PUSH_SCRIPT=str(stub))
    assert p.returncode == 6


def test_offline_refuses_a_fetched_tree_missing_an_expected_run_and_logs_without_a_logs_dir(tmp_path):
    """review 2, MAJOR A + MINOR 7: on a fetched tree with no logs/, offline.sh creates its own log dir, writes .xz
    files out, and stops (hard) when a run the rental's SUMMARY says it measured is missing."""
    import lzma
    res = tmp_path / "run"
    (res / "A1710").mkdir(parents=True)
    (res / "A1710" / "calls.jsonl.xz").write_bytes(lzma.compress(b'{"g": 0}\n'))
    (res / "SUMMARY.txt").write_text("PASS A1710-extract (5s)\nPASS A1410-extract (5s)\n")
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    p = subprocess.run(["bash", str(PROD / "offline.sh"), "--results", str(res)], capture_output=True, text=True,
                       env=env, cwd=str(REPO), timeout=120)
    assert p.returncode == 3, p.stdout + p.stderr
    assert (res / "A1710" / "calls.jsonl").read_bytes() == b'{"g": 0}\n'          # decompressed
    log = res / "offline-logs" / "logs" / "expected-outputs.log"
    assert log.exists() and "run A1410: no calls" in log.read_text()
    assert "FAIL expected-outputs" in p.stdout and "No such file" not in p.stdout + p.stderr
    assert not (res / "logs").exists() and not (res / "score").exists()


def test_sag1410_fails_early_without_the_preflight_census(tmp_path):
    """review 2, MINOR 5: no $SESSION_ROOT/census.json -> FAIL before the driver runs (dry run and real)."""
    ses = tmp_path / "session"
    ses.mkdir()
    env = dict(os.environ, MOE_HOME=str(tmp_path / "moe"), HOME=str(tmp_path / "home"), SESSION_ROOT=str(ses),
               RESULTS_ROOT=str(tmp_path / "results"))
    for extra in (["--dry-run"], []):
        p = subprocess.run(["bash", str(PROD / "sag1410.sh"), "--out", str(tmp_path / "sag"), *extra],
                           capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
        assert p.returncode == 3 and "no census at" in p.stdout, p.stdout + p.stderr
        assert not (ses / "gh200-driver").exists() and not (tmp_path / "sag").exists()


def test_the_sag_plan_is_the_v7_prefill_blinds_own_units_and_the_driver_accepts_it(tmp_path):
    plan = [x for x in (PROD / "sag1410.plan").read_text().splitlines() if x and not x.startswith("#")]
    reg = (REPO / "scripts/plans/rentalV7P-prefill-blind-2026-11.plan").read_text().splitlines()
    assert plan[0] == "- prelude lock-mhz=1410 power-limit-w=700 est=2 cap=15" and plan[0] in reg
    assert all(x in reg for x in plan) and len(plan) == 4
    assert [x.split()[1] for x in plan] == ["prelude", "bytes", "bytes", "gate"] and plan[-1].endswith("cap=5")
    env = dict(os.environ, MOE_HOME=str(tmp_path / "moe"), HOME=str(tmp_path / "home"))
    p = subprocess.run(["bash", str(PROD / "sag1410.sh"), "--out", str(tmp_path / "sag"), "--dry-run"],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "PASS sag1410 plan" in p.stdout and "gate b3sag" in p.stdout and not (tmp_path / "sag").exists()
    assert "precondition: $SESSION_ROOT/census.json" in p.stdout


def test_frontier_setup_dry_run(tmp_path):
    env = dict(os.environ, PY_VLLM="python3")
    p = subprocess.run(["bash", str(PROD / "frontier_setup.sh"), "--dry-run", "--dest", str(tmp_path / "fr")],
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "checkout --quiet --detach db5a85b" in p.stdout and "vllm==0.10.2" in p.stdout
    assert "--no-deps -e" in p.stdout and not (tmp_path / "fr").exists()


def _git_repo(d: Path) -> str:
    d.mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    for cmd in (["git", "init", "-q"], ["git", "commit", "-q", "--allow-empty", "-m", "x"]):
        subprocess.run(cmd, cwd=d, check=True, env=env, capture_output=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, check=True, capture_output=True, text=True).stdout.strip()


def test_frontier_setup_reuses_a_checkout_at_the_pin_and_refuses_another(tmp_path):
    """SMOKE-FIXES 9: a second run found the source dir and `git clone` failed (128). A checkout already at the pinned
    commit is reused (no clone, no fetch); one at another commit, or a non-git directory, is refused (exit 4)."""
    src = tmp_path / "fr" / "src"
    head = _git_repo(src)
    kit = tmp_path / "kit" / "scripts" / "prod"
    kit.mkdir(parents=True)
    text = (PROD / "frontier_setup.sh").read_text()
    assert 'COMMIT="db5a85b"' in text
    (kit / "frontier_setup.sh").write_text(text.replace('COMMIT="db5a85b"', f'COMMIT="{head[:7]}"')
                                           .replace('URL="https://github.com/NetX-lab/Frontier"', 'URL="file:///nonexistent"'))
    run = lambda: subprocess.run(["bash", str(kit / "frontier_setup.sh"), "--dest", str(tmp_path / "fr"), "--arms", "none"],
                                 capture_output=True, text=True, timeout=120)
    p = run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "reusing" in p.stdout and "git clone" not in p.stdout and (tmp_path / "fr" / "frontier.env").exists()
    p = run()                                                   # idempotent: a second run is the same
    assert p.returncode == 0 and "reusing" in p.stdout
    (kit / "frontier_setup.sh").write_text(text.replace('COMMIT="db5a85b"', 'COMMIT="0000000"'))
    p = run()
    assert p.returncode == 4 and "refused" in p.stdout and head[:7] in p.stdout
    import shutil
    shutil.rmtree(src)
    src.mkdir()
    (src / "junk").write_text("x")
    p = run()
    assert p.returncode == 4 and "not a git checkout" in p.stdout


def test_every_main_step_has_a_cap_at_least_twice_its_expected_time_and_the_plan_fits_4h(tmp_path):
    """README "main.sh: every step": the expected seconds come from the smoke's measured step times scaled to ERNIE
    (plan_time.py); every step the plan expects to run has its cap >= 2x that; the default plan is <= 4.0 h with
    >= 30 steps per tile (the estimator) and the dry run prints the expected total."""
    sys.path.insert(0, str(PROD))
    import plan_time as PT
    steps = _dry("main.sh", tmp_path)
    caps = {s.split()[1]: _cap(s) for s in steps}
    plan = PT.plan()
    names = [st for _b, st, _s, _x in plan["steps"]]
    assert set(names) == set(caps), (set(names) ^ set(caps))
    for _blk, st, sec, _basis in plan["steps"]:
        assert caps[st] >= 2 * sec, (st, caps[st], sec)
    assert plan["total_h"] <= 4.0 and min(plan["serving"]["per_tile"].values()) >= 30
    assert plan["frontier"]["points"] == plan["frontier"]["grid_points"] == 387          # the full grid
    assert plan["frontier"]["samples"] == 3 * 387                                       # uniform x 3 seeds
    assert plan["worst_download_total_s"] > plan["total_s"] and caps["env-preflight"] == 600
    env = dict(os.environ, PY_VLLM=sys.executable, HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
    p = subprocess.run(["bash", str(PROD / "main.sh"), "--dry-run", "--results", str(tmp_path / "res2")],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    tot = [x for x in p.stdout.splitlines() if x.startswith("PLAN EXPECTED TOTAL")]
    assert tot and f"{plan['total_s']:.0f} s" in tot[0], p.stdout[-2000:]
    assert any(x.startswith("PLAN WORST CASE at the 50 MB/s download floor") for x in p.stdout.splitlines())


def test_held_back_check_fails_on_a_needed_file_and_passes_on_the_rentals_intermediates(tmp_path):
    """review 2, MAJOR A: vm_results_push.sh (pinned) lists in HELD-BACK.txt what it compressed or left on the VM;
    main.sh's last hard step fails if any of it is a file the laptop needs."""
    import sys
    sys.path.insert(0, str(PROD))
    import prodlib as PL
    run = "prod-main-20261120T000000Z"
    ok = (f"vm/results/{run}/A1710/serve.nsys-rep\t9e9 bytes\tsha256 aa\tLEFT ON THE VM\n"
          f"vm/results/{run}/A1710/serve.sqlite\t5e9 bytes\tsha256 bb\tCOMPRESSED as x.xz\n"
          f"vm/results/{run}/data/ShareGPT_V3_unfiltered_cleaned_split.json\t7e8 bytes\tsha256 cc\tLEFT ON THE VM\n"
          f"vm/results/other-run/A1710/calls.jsonl\t2e8 bytes\tsha256 dd\tLEFT ON THE VM\n")
    assert PL.held_back_needed(ok, run) == []
    bad = ok + f"vm/results/{run}/A1410/calls.jsonl\t2e8 bytes\tsha256 ee\tCOMPRESSED as calls.jsonl.xz\n"
    assert len(PL.held_back_needed(bad, run)) == 1
    held = tmp_path / "HELD-BACK.txt"
    held.write_text(ok)
    (tmp_path / "SHA256SUMS").write_text(f"ab  vm/results/{run}/SUMMARY.txt\n")
    r = subprocess.run([sys.executable, str(PROD / "prodlib.py"), "held-back-check", "--held", str(held), "--results",
                        str(tmp_path / run)], capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert r.returncode == 0, r.stdout
    # a run the push never carried (not under the pushed results root) is not a vacuous pass
    r = subprocess.run([sys.executable, str(PROD / "prodlib.py"), "held-back-check", "--held", str(held), "--results",
                        str(tmp_path / "prod-main-elsewhere")], capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert r.returncode == 3 and "holds no file" in r.stdout
    held.write_text(bad)
    r = subprocess.run([sys.executable, str(PROD / "prodlib.py"), "held-back-check", "--held", str(held), "--results",
                        str(tmp_path / run)], capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert r.returncode == 3 and "A1410/calls.jsonl" in r.stdout
    r = subprocess.run([sys.executable, str(PROD / "prodlib.py"), "held-back-check", "--held", str(tmp_path / "none"),
                        "--results", str(tmp_path / run)], capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert r.returncode == 3 and "never pushed" in r.stdout
    # the capture ring's shards stay below the cap by construction (drain at half of a 64 Mi int16 ring)
    import moe_capture as MC
    assert MC.IDS_CAP * MC.DRAIN_FILL * 2 + MC.REC_CAP * MC.DRAIN_FILL * 16 < 95_000_000


def test_expected_outputs_follow_the_rentals_own_summary(tmp_path):
    import sys
    sys.path.insert(0, str(PROD))
    import prodlib as PL
    R = tmp_path
    assert PL.expected_outputs(R)["missing"]                                      # no SUMMARY.txt: refused
    (R / "SUMMARY.txt").write_text("PASS A1710-extract (1s)\nPASS A1410-extract (1s)\nPASS Aon-extract (1s)\n"
                                   "FAIL (soft, dropped) Beager-extract: exit 3\nPASS frontier-predict (1s)\n"
                                   "PASS replay-offset-check-decode (1s)\n")
    for lab in ("A1710", "A1410"):
        (R / lab).mkdir()
        (R / lab / "calls.jsonl").write_text("{}\n")
    (R / "sag").mkdir()
    (R / "sag" / "par.json").write_text("{}")
    for b in ("decode", "prefill"):
        (R / f"replay-{b}" / "x").mkdir(parents=True)
        (R / f"replay-{b}" / "x" / "cells.csv").write_text("")
    got = PL.expected_outputs(R)
    assert got["runs"] == ["A1710", "A1410", "Aon"]
    assert any("run Aon" in m for m in got["missing"]) and any("predictions/index.json" in m for m in got["missing"])
    assert any("replay-offset-decode/replay_check.json" in m for m in got["missing"])
    (R / "Aon").mkdir()
    (R / "Aon" / "calls.shards.json").write_text("{}")
    (R / "frontier" / "predictions").mkdir(parents=True)
    (R / "frontier" / "predictions" / "index.json").write_text("{}")
    (R / "replay-offset-decode").mkdir()
    (R / "replay-offset-decode" / "replay_check.json.xz").write_bytes(b"x")
    assert PL.expected_outputs(R)["missing"] == []
    # review m8: frontier-csv is tied to the CSV it really writes (not points.json)
    with open(R / "SUMMARY.txt", "a") as fh:
        fh.write("PASS frontier-points (1s)\nPASS frontier-csv (1s)\n")
    (R / "frontier" / "points.json").write_text("{}")
    miss = PL.expected_outputs(R)["missing"]
    assert len(miss) == 1 and "frontier/profile/compute/gh200/ernie45-21b-a3b-moe/moe_kernel_only.csv" in miss[0]
    import frontier_op as FO
    assert PL.FRONTIER_CSV_REL == str(FO.csv_relpath("ernie-4.5-21b-a3b"))
    csv = R / "frontier" / "profile" / PL.FRONTIER_CSV_REL
    csv.parent.mkdir(parents=True)
    csv.write_text("x\n")
    assert PL.expected_outputs(R)["missing"] == []


# ---- smoke rental 2026-11-19 (SMOKE-FIXES.md 1-5): env, resume, preflight -------------------------------------

def _dry_raw(name, tmp_path, *extra, py="python3"):
    env = dict(os.environ, PY_VLLM=py, HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
    return subprocess.run(["bash", str(PROD / name), "--dry-run", "--results", str(tmp_path / "res"), *extra],
                          capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)


@pytest.mark.parametrize("name", ["smoke.sh", "main.sh"])
def test_the_first_step_is_the_hard_env_preflight_and_every_step_sees_tmpdir_and_the_venv_bin(name, tmp_path):
    p = _dry_raw(name, tmp_path, py="/opt/venvs/vllm/bin/python")
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    # items 2 and 3: TMPDIR under $MOE_HOME (nsys: the VM's /tmp/nvidia is root-owned) and the venv's bin first on
    # PATH (FlashInfer's JIT runs `ninja`), exported once for every step; the dry run creates neither
    assert f"ENV TMPDIR={tmp_path}/moe/tmp" in lines and "ENV PATH first=/opt/venvs/vllm/bin" in lines
    assert not (tmp_path / "moe").exists()
    steps = [x for x in lines if x.startswith("STEP ")]
    first = steps[0]
    assert first.split()[1] == "env-preflight" and not _soft(first) and _cap(first) == 600 and " always " in first
    for want in ("--model ", f"--tmpdir {tmp_path}/moe/tmp", "--locks 1710\\,1410", "--power-w 700",
                 "--min-free-gb 200", "env_preflight.py"):
        assert want in first, want
    # the lock / unlock / env steps are re-run on a resume; the captures are not
    by = {x.split()[1]: x for x in steps}
    for n in ("env", "power-readback", "unlock-start", "unlock-after-sag", "unlock-after-replay"):
        assert " always " in by[n], n
    assert " always " not in by["sag1410"] and "out=" in by["sag1410"]


def test_power_check_reads_the_enforced_limit_not_the_default_setting(tmp_path):
    """item 1: the GH200 reports power.limit 900 (the default setting) and enforced.power.limit 700."""
    b = tmp_path / "bin"
    b.mkdir()
    (b / "nvidia-smi").write_text('#!/bin/bash\ncase "$*" in *enforced.power.limit*) echo "700.00";; '
                                  '*power.limit*) echo "900.00";; esac\n')
    (b / "nvidia-smi").chmod(0o755)
    path = f"{b}:{os.environ['PATH']}"
    p, summ = _kit(tmp_path / "a", "step pw --cap 10 -- kit_power_check 700", PATH=path)
    assert p.returncode == 0 and summ[0].startswith("PASS pw"), p.stdout + p.stderr
    p, summ = _kit(tmp_path / "b", "step pw --cap 10 -- kit_power_check 900", PATH=path)
    assert summ[0].startswith("FAIL pw") and "enforced limit 700 W, want 900 W" in summ[0]


def test_kit_env_exports_tmpdir_and_puts_the_venv_bin_first_for_every_step(tmp_path):
    """items 2 and 3: a python step finds the venv's ninja; an nsys step sees the kit's TMPDIR (created)."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "ninja").write_text("#!/bin/bash\necho ninja-of-the-venv\n")
    (venv / "ninja").chmod(0o755)
    py = venv / "python"
    py.symlink_to(sys.executable)
    p, summ = _kit(tmp_path, f"kit_env {py}\n"
                             f"step pyjit --cap 30 -- {py} -c 'import subprocess; print(subprocess.check_output([\"ninja\"]).decode())'\n"
                             "step tmp --cap 10 -- bash -c 'echo TMP=$TMPDIR; test -d \"$TMPDIR\" && test -w \"$TMPDIR\"'",
                   MOE_HOME=str(tmp_path / "moe"))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "ninja-of-the-venv" in (tmp_path / "r" / "logs" / "pyjit.log").read_text()
    assert f"TMP={tmp_path}/moe/tmp" in (tmp_path / "r" / "logs" / "tmp.log").read_text()


def _make(paths):
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


def test_smoke_resume_skips_the_done_steps_and_refuses_a_missing_output(tmp_path):
    """item 5: --from STEP --results DIR: the steps before STEP are SKIPPED-RESUME (never re-run) when their
    declared outputs are present; a missing one refuses the resume; the --always steps run again."""
    res = tmp_path / "res"
    p = _dry_raw("smoke.sh", tmp_path, "--from", "sag1410")
    assert p.returncode == 2 and "RESUME REFUSED" in p.stderr and "prompts-fetch" in p.stderr
    _make([res / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json", res / "prompts.jsonl",
           res / "A" / "serve.nsys-rep", res / "A" / "manifest.json", res / "A" / "serve.sqlite",
           res / "A" / "calls.shards.json", res / "B" / "torch_trace", res / "B" / "calls.jsonl",
           tmp_path / "moe" / "frontier" / "frontier.env", res / "frontier-probe" / "predictions" / "index.json"])
    p = _dry_raw("smoke.sh", tmp_path, "--from", "sag1410")
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    skipped = [x.split()[1] for x in lines if x.startswith("SKIPPED-RESUME ")]
    run = [x.split()[1] for x in lines if x.startswith("STEP ")]
    assert {"price-check", "prompts-fetch", "capture-graphs-nsys", "extract-gate-A", "capture-eager-torchprof",
            "extract-gate-B"} <= set(skipped)
    # the soft graph-mode overhead run had no output: it stays dropped, not re-run
    assert any(x.startswith("SKIPPED-RESUME capture-graphmode-overhead: output absent") for x in lines)
    assert run[:4] == ["env-preflight", "env", "power-readback", "unlock-start"] and run[4] == "sag1410"
    assert "capture-graphs-nsys" not in run and "score" in run
    assert any(x.startswith("RESUME from sag1410 ") for x in lines)
    # --from without --results, and a step the plan does not have, are refused
    env = dict(os.environ, PY_VLLM="python3", HOME=str(tmp_path / "home"), MOE_HOME=str(tmp_path / "moe"))
    q = subprocess.run(["bash", str(PROD / "smoke.sh"), "--dry-run", "--from", "sag1410"], capture_output=True,
                       text=True, env=env, cwd=str(REPO), timeout=60)
    assert q.returncode == 2 and "needs --results" in q.stderr
    q = _dry_raw("smoke.sh", tmp_path, "--from", "no-such-step")
    assert q.returncode == 2 and "no such step" in q.stderr


def test_main_resume_relocks_before_the_capture_it_resumes(tmp_path):
    """item 5: resuming main at A1410-capture re-runs the locks and unlocks (always) and skips A1710's work."""
    res = tmp_path / "res"
    p = _dry_raw("main.sh", tmp_path, "--from", "A1410-capture")
    assert p.returncode == 2 and "RESUME REFUSED" in p.stderr
    tr = tmp_path / "moe" / "traces" / "res"            # the raw traces live outside the results tree (m6)
    _make([res / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json", res / "prompts.jsonl", res / "sag" / "par.json",
           tmp_path / "moe" / "frontier" / "frontier.env", res / "frontier-probe" / "predictions" / "index.json"]
          + [res / lab / f for lab in ("gate60", "A1710") for f in ("manifest.json", "calls.jsonl")]
          + [tr / lab / f for lab in ("gate60", "A1710") for f in ("serve.nsys-rep", "serve.sqlite")])
    p = _dry_raw("main.sh", tmp_path, "--from", "A1410-capture")
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    run = [x.split()[1] for x in lines if x.startswith("STEP ")]
    skipped = [x.split()[1] for x in lines if x.startswith("SKIPPED-RESUME ")]
    assert {"A1710-capture", "A1710-export", "A1710-extract", "sag1410", "gate60-capture", "price-check"} <= set(skipped)
    assert run.index("A1710-lock") < run.index("A1410-lock") < run.index("A1410-capture")
    assert "A1710-capture" not in run and run[-1] == "held-back-check"


def test_a_real_resume_never_reruns_a_done_step_and_records_it(tmp_path):
    """item 5, the step runner itself (KIT_DRY 0): a done step is recorded SKIPPED-RESUME and not run; an --always
    step runs; a hard step whose output is gone refuses before anything after it runs."""
    a, c = tmp_path / "a.out", tmp_path / "c.ran"
    body = (f"step a --cap 10 --out {a} -- bash -c 'echo run >> {tmp_path}/a.count; touch {a}'\n"
            f"step lk --always --cap 10 -- bash -c 'echo lk >> {tmp_path}/lk.count'\n"
            f"step c --cap 10 -- touch {c}")
    p, summ = _kit(tmp_path, body)
    assert p.returncode == 0 and (tmp_path / "a.count").read_text() == "run\n"
    p, summ = _kit(tmp_path, "KIT_FROM=c; KIT_RESUMING=1\n" + body)
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "a.count").read_text() == "run\n" and (tmp_path / "lk.count").read_text() == "lk\nlk\n"
    assert "SKIPPED-RESUME a" in summ and summ[-1].startswith("PASS c")
    a.unlink()
    c.unlink()
    p, summ = _kit(tmp_path, "KIT_FROM=c; KIT_RESUMING=1\n" + body)
    assert p.returncode == 2 and "RESUME REFUSED" in p.stderr and not c.exists()


# ---- review 2026-11-19 m6: the raw traces outside the pushed tree ----------------------------------------------

def test_kit_push_records_the_traces_sha256_in_the_results_tree_and_rehashes_only_new_files(tmp_path):
    import hashlib
    sys.path.insert(0, str(PROD))
    import prodlib as PL
    tr, res = tmp_path / "moe" / "traces" / "run1", tmp_path / "moe" / "results" / "run1"
    (tr / "A1710").mkdir(parents=True)
    res.mkdir(parents=True)
    (tr / "A1710" / "serve.nsys-rep").write_bytes(b"rep" * 1000)
    stub = tmp_path / "push_stub.sh"
    stub.write_text(f'printf "%s\\n" "$@" > {tmp_path / "pushed.txt"}\n')
    p, summ = _kit(tmp_path / "k", "step push-x --soft --cap 30 -- kit_push 'msg'", KIT_PUSH_SCRIPT=str(stub),
                   KIT_TRACES=str(tr), KIT_RESULTS=str(res), REPO=str(REPO), PY_VLLM=sys.executable)
    assert p.returncode == 0 and summ[0].startswith("PASS push-x"), (summ, p.stderr)
    man = json.loads((res / PL.TRACES_MANIFEST).read_text())
    assert man["files"]["A1710/serve.nsys-rep"]["sha256"] == hashlib.sha256(b"rep" * 1000).hexdigest()
    assert (tmp_path / "pushed.txt").read_text().startswith("push")
    # a second manifest keeps an unchanged file's hash (no re-read) and hashes a new one
    man["files"]["A1710/serve.nsys-rep"]["sha256"] = "kept"
    (res / PL.TRACES_MANIFEST).write_text(json.dumps(man))
    (tr / "A1710" / "serve.sqlite").write_bytes(b"sq")
    doc = PL.trace_manifest(tr, res / PL.TRACES_MANIFEST)
    assert doc["files"]["A1710/serve.nsys-rep"]["sha256"] == "kept" and len(doc["files"]) == 2
    assert doc["total_bytes"] == 3002


def test_pull_traces_plans_an_rsync_with_the_runs_known_hosts_and_refuses_an_unknown_host(tmp_path):
    env = dict(os.environ, MOE_VM_STATE_ROOT=str(tmp_path / "runs"), HOME=str(tmp_path / "home"))
    p = subprocess.run(["bash", str(PROD / "pull_traces.sh"), "10.0.0.7", "prod-main-x", "--dest", str(tmp_path / "d"),
                        "--dry-run"], capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "ubuntu@10.0.0.7:moe/traces/prod-main-x/" in p.stdout and "StrictHostKeyChecking=yes" in p.stdout
    assert f"UserKnownHostsFile={tmp_path}/runs/known_hosts" in p.stdout and "moe/results/prod-main-x/traces.sha256.json" in p.stdout
    assert not (tmp_path / "d").exists()
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "known_hosts").write_text("")
    p = subprocess.run(["bash", str(PROD / "pull_traces.sh"), "10.0.0.7", "prod-main-x", "--dest", str(tmp_path / "d")],
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 2 and "not in" in p.stderr and not (tmp_path / "d").exists()
    p = subprocess.run(["bash", str(PROD / "pull_traces.sh"), "10.0.0.7"], capture_output=True, text=True, env=env,
                       timeout=60)
    assert p.returncode == 2


# ---- v10 blind run (2026-11-21): the three kit faults it hit -----------------------------------------------------

def _git(cwd, *args, **kw):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True, env=env, **kw)


def _push_home(tmp_path, branch="run-x"):
    """A bare 'GitHub', and a push home as vm_results_push.sh init leaves it: push.env and the clone at push/."""
    bare = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(bare))
    home = tmp_path / "pushhome"
    work = home / "push"
    work.mkdir(parents=True)
    _git(work, "init", "-q")
    _git(work, "checkout", "-q", "--orphan", branch)
    _git(work, "remote", "add", "origin", str(bare))
    (work / "README.md").write_text("run\n")
    (work / "old.txt").write_text("to be deleted\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "first")
    _git(work, "push", "-q", "origin", f"HEAD:refs/heads/{branch}")
    (home / "push.env").write_text(f"BRANCH={branch}\nRUN_ID=blind-x\nREMOTE={bare}\n")
    return bare, home, work


def test_push_chunks_pushes_the_same_tree_in_parts_under_the_limit_on_the_runs_branch(tmp_path):
    """GitHub refused the v10 run's 2.4 GiB pack. push_chunks.sh pushes HEAD's tree as a chain of fast-forward commits
    on the run's own branch, each adding at most LIMIT bytes, deletions and HEAD's message in the last; the local
    branch then points at the last part (same tree)."""
    bare, home, work = _push_home(tmp_path)
    for i in range(5):
        (work / "vm" / f"page{i}.bin").parent.mkdir(exist_ok=True)
        (work / "vm" / f"page{i}.bin").write_bytes(bytes([i]) * 1000)
    (work / "old.txt").unlink()
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "blind-x: the big push")
    tree = _git(work, "rev-parse", "HEAD^{tree}").stdout.strip()
    env = dict(os.environ, MOE_PUSH_HOME=str(home))
    p = subprocess.run(["bash", str(PROD / "push_chunks.sh"), "--limit", "2500"], capture_output=True, text=True,
                       env=env, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    tip = _git(bare, "rev-parse", "refs/heads/run-x").stdout.strip()
    assert _git(bare, "rev-parse", f"{tip}^{{tree}}").stdout.strip() == tree
    assert _git(work, "rev-parse", "HEAD").stdout.strip() == tip            # the local branch follows (soft)
    assert _git(work, "status", "--porcelain").stdout == ""
    log = _git(bare, "log", "--format=%s", "refs/heads/run-x").stdout.splitlines()
    assert log[0] == "blind-x: the big push" and log[-1] == "first" and len(log) == 4   # 2 + 2 + 1 pages
    for c in _git(bare, "rev-list", "refs/heads/run-x~3..refs/heads/run-x").stdout.split():
        added = _git(bare, "diff-tree", "-r", "--no-commit-id", "--diff-filter=AM", "-z", "--name-only", c).stdout
        assert sum(1000 for f in added.split("\0") if f) <= 2500
    assert "old.txt" not in _git(bare, "ls-tree", "-r", "--name-only", tip).stdout
    # nothing left: a second run pushes nothing
    p = subprocess.run(["bash", str(PROD / "push_chunks.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0 and "nothing to push" in p.stdout
    # no config: refused
    p = subprocess.run(["bash", str(PROD / "push_chunks.sh")], capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, MOE_PUSH_HOME=str(tmp_path / "none")))
    assert p.returncode == 2


def test_kit_push_falls_back_to_parts_only_on_githubs_pack_limit(tmp_path):
    home = tmp_path / "pushhome"
    (home / "push" / ".git").mkdir(parents=True)
    err = home / "push" / ".git" / "push-error.txt"
    stub = tmp_path / "push_stub.sh"
    stub.write_text("exit 1\n")
    chunks = tmp_path / "chunks_stub.sh"
    chunks.write_text(f'echo called > {tmp_path / "chunked"}\n')
    env = dict(KIT_PUSH_SCRIPT=str(stub), KIT_CHUNK_SCRIPT=str(chunks), MOE_PUSH_HOME=str(home), REPO=str(REPO))
    err.write_text("remote: fatal: pack exceeds maximum allowed size (2.00 GiB)\n")
    p, summ = _kit(tmp_path / "a", "step push-x --soft --cap 30 -- kit_push 'msg'", **env)
    assert summ[0].startswith("PASS push-x") and (tmp_path / "chunked").exists(), (summ, p.stdout, p.stderr)
    (tmp_path / "chunked").unlink()
    err.write_text("ssh: connect to host github.com port 22: Connection timed out\n")   # any other failure: no parts
    p, summ = _kit(tmp_path / "b", "step push-x --soft --cap 30 -- kit_push 'msg'", **env)
    assert summ[0].startswith("FAIL (soft, dropped) push-x") and not (tmp_path / "chunked").exists(), summ


def test_frontier_setup_reuses_a_primary_venv_that_imports_the_study_stack_and_never_blanks_the_interpreter(tmp_path):
    """v10 blind run: a second model on the board found venv-primary, `uv venv` refused it, and frontier.env was
    written with FR_PY_PRIMARY="". A venv that imports vLLM 0.27.1 at the study's torch and Frontier's kernel module
    is reused (no uv call); any other is rebuilt with --clear."""
    src = tmp_path / "fr" / "src"
    head = _git_repo(src)
    kit = tmp_path / "kit" / "scripts" / "prod"
    kit.mkdir(parents=True)
    text = (PROD / "frontier_setup.sh").read_text()
    (kit / "frontier_setup.sh").write_text(text.replace('COMMIT="db5a85b"', f'COMMIT="{head[:7]}"'))
    (tmp_path / "kit" / "requirements").mkdir()
    (tmp_path / "kit" / "requirements" / "resolved-vllm.txt").write_text("vllm==0.27.1\n")
    fake = ("#!/usr/bin/env bash\n"
            'case "$*" in\n'
            '  *sys.version_info*) echo 3.12 ;;\n'
            '  *"vllm, torch"*) echo "0.27.1 2.13.0+cu130" ;;\n'
            '  *moe_vllm_kernel*) echo v1 ;;\n'
            '  *torch.__version__*) echo 2.13.0+cu130 ;;\n'
            "esac\n")
    vpy = tmp_path / "fr" / "venv-primary" / "bin" / "python"
    vpy.parent.mkdir(parents=True)
    vpy.write_text(fake)
    vpy.chmod(0o755)
    study = tmp_path / "study_python"
    study.write_text(fake)
    study.chmod(0o755)
    uvlog = tmp_path / "uv.log"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "uv").write_text(f'#!/usr/bin/env bash\necho "uv $*" >> {uvlog}\nexit 1\n')
    (bin_ / "uv").chmod(0o755)
    env = dict(os.environ, PY_VLLM=str(study), PATH=f"{bin_}:{os.environ['PATH']}")
    p = subprocess.run(["bash", str(kit / "frontier_setup.sh"), "--dest", str(tmp_path / "fr"), "--arms", "primary"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "reusing" in p.stdout and not uvlog.exists()
    assert f'FR_PY_PRIMARY="{vpy}"' in (tmp_path / "fr" / "frontier.env").read_text()
    # a venv at another stack is rebuilt with --clear (the stub uv fails here, so the arm FAILs and says so)
    vpy.write_text(fake.replace("0.27.1", "0.10.2"))
    p = subprocess.run(["bash", str(kit / "frontier_setup.sh"), "--dest", str(tmp_path / "fr"), "--arms", "primary"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 4 and "rebuilding" in p.stdout
    assert "uv venv --clear --python 3.12" in uvlog.read_text()


def test_sag1410_moves_an_earlier_sags_pages_pointers_and_gate_record_aside(tmp_path):
    """v10 blind run: LFM2's sag ran after granite's on one board and its pages landed in granite's page directory.
    Before the driver runs, an earlier sag's pages, pointer files and gate record are renamed .aside-<UTC>."""
    kit = tmp_path / "kit"
    (kit / "scripts" / "prod").mkdir(parents=True)
    (kit / "scripts" / "prod" / "sag1410.sh").write_text((PROD / "sag1410.sh").read_text())
    (kit / "scripts" / "gh200_v7p_session.sh").write_text(
        'ls "$RESULTS_ROOT" > "$RESULTS_ROOT/../seen-by-driver.txt"\n')
    (kit / "scripts" / "prod" / "sag_par.py").write_text("import sys\n")
    S, R = tmp_path / "session", tmp_path / "results"
    (S / "gh200-driver").mkdir(parents=True)
    (S / "census.json").write_text("{}")
    old = R / "2026-11-21-gh200-olmoe-1b-7b-f128-r3-counters"
    (old / "lock1410").mkdir(parents=True)
    (S / "counters-dir-olmoe-1b-7b-f128.txt").write_text(str(old) + "\n")
    (R / "2026-11-20-gh200-mixtral-8x7b-f128-r3-counters").mkdir()
    (S / "gh200-driver" / "gate-b3sag-1410.json").write_text("{}")
    (R / "prod-other-run").mkdir()
    env = dict(os.environ, SESSION_ROOT=str(S), RESULTS_ROOT=str(R), PY_BASE=sys.executable)
    p = subprocess.run(["bash", str(kit / "scripts" / "prod" / "sag1410.sh"), "--out", str(tmp_path / "sag")],
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    seen = (tmp_path / "seen-by-driver.txt").read_text().split()
    assert not [d for d in seen if d.endswith("-f128-r3-counters")] and "prod-other-run" in seen
    assert len([d for d in seen if ".aside-" in d]) == 2
    assert not (S / "counters-dir-olmoe-1b-7b-f128.txt").exists()
    assert not (S / "gh200-driver" / "gate-b3sag-1410.json").exists()
    assert len(list((S / "gh200-driver").glob("gate-b3sag-1410.json.aside-*"))) == 1
