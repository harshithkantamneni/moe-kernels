"""scripts/prod/{smoke,main,frontier_setup}.sh: shell syntax, and the --dry-run plans (every step printed, nothing
created, the ERNIE 60 s gate before any long run, the locks per branch)."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROD = REPO / "scripts" / "prod"


@pytest.mark.parametrize("name", ["smoke.sh", "main.sh", "kitlib.sh", "frontier_setup.sh", "sag1410.sh", "offline.sh"])
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
                 "frontier-profile-fidelity", "score"):
        assert want in names, want
    by = dict(zip(names, steps, strict=True))
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


def test_main_dry_run_gates_first_and_locks_each_branch(tmp_path):
    steps = _dry("main.sh", tmp_path)
    names = [s.split()[1] for s in steps]
    by = dict(zip(names, steps, strict=True))
    assert names.index("gate60-extract") < names.index("A1710-capture") < names.index("A1410-capture")
    # review 1 MAJOR 5: gate60, (the sag par), A1710, A1410, replay C, Frontier, then the secondary runs, all soft
    order = ["gate60-extract", "sag1410", "A1710-capture", "A1410-capture", "replay-decode", "replay-prefill",
             "frontier-profile-primary-uniform", "Adefault-capture", "Aon-capture", "Beager-capture"]
    assert [names.index(n) for n in order] == sorted(names.index(n) for n in order)
    for n in ("Adefault-capture", "Adefault-extract", "Aon-lock", "Aon-capture", "Aon-extract", "Beager-capture",
              "Beager-extract", "replay-offset-nsys"):
        assert _soft(by[n]), n
    for n in ("A1710-capture", "A1410-extract", "replay-prefill", "sag1410", "price-check", "gate60-sequences",
              "revision-pinned", "replay-wall-present"):
        assert not _soft(by[n]), n
    # review 1 MAJOR 7: no CPU scoring or Frontier predict on the rental; the replay budget reads the smoke's walls
    assert not any(n == "score" or n.startswith("frontier-predict") for n in names)
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
    assert not any("variant-all4" in n for n in _dry("main.sh", tmp_path / "x", "--skip-frontier-variant"))
    # review 2, MAJOR C: every step carries its registered cap; the long ones above their own budgets
    caps = {n: _cap(by[n]) for n in names}
    assert all(c > 0 for c in caps.values())
    assert caps["sag1410"] > 75 * 60 and caps["replay-decode"] > 2400 and caps["replay-prefill"] > 2400
    assert caps["A1710-capture"] == caps["A1410-capture"] == 10800 and caps["held-back-check"] == 120
    # MINOR 6: each capture / export / extract needs the step before it (a failed lock SKIPS the capture)
    for lab, pre in (("A1710", "A1710-lock"), ("Aon", "Aon-lock"), ("Adefault", "Adefault-unlock")):
        assert f"needs={pre} " in by[f"{lab}-capture"]
        assert f"needs={lab}-capture " in by[f"{lab}-export"] and f"needs={lab}-export " in by[f"{lab}-extract"]
    assert "needs=replay-offset-lock " in by["replay-offset-nsys"]
    # MAJOR B: per branch a probe, its own ERNIE walls, then the rest sized by them inside the fixed budget
    for b in ("decode", "prefill"):
        seq = [f"replay-probe-sample-{b}", f"replay-probe-{b}", f"replay-probe-wall-{b}", f"replay-sample-{b}",
               f"replay-{b}"]
        assert [names.index(n) for n in seq] == sorted(names.index(n) for n in seq)
        assert "--per-tile 2" in by[f"replay-probe-sample-{b}"] and "--wall " in by[f"replay-probe-sample-{b}"]
        assert "--model ernie-4.5-21b-a3b" in by[f"replay-probe-wall-{b}"]
        assert f"--probe-wall {tmp_path}/res/replay-probe-{b}/replay_wall.json" in by[f"replay-sample-{b}"]
        assert "--wall " not in by[f"replay-sample-{b}"]
    # MAJOR A: the last step is the hard held-back check, after the final push
    assert names[-1] == "held-back-check" and not _soft(by["held-back-check"])
    assert names.index("push-final") < names.index("held-back-check")


def test_offline_dry_run_predicts_and_scores_on_the_laptop(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    p = subprocess.run(["bash", str(PROD / "offline.sh"), "--dry-run", "--results", str(tmp_path / "res")],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=60)
    assert p.returncode == 0, p.stderr
    steps = [x for x in p.stdout.splitlines() if x.startswith("STEP ")]
    names = [x.split()[1] for x in steps]
    assert names[-1] == "score" and "frontier-predict-primary-uniform" in names and not (tmp_path / "res").exists()
    sc = steps[-1]
    assert "--sag" in sc and "--workers 4" in sc and "ernie-4.5-21b-a3b" in sc
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
                                   "FAIL (soft, dropped) Beager-extract: exit 3\nPASS frontier-train-primary-uniform (1s)\n")
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
    assert any("run Aon" in m for m in got["missing"]) and any("primary-uniform/models" in m for m in got["missing"])
    (R / "Aon").mkdir()
    (R / "Aon" / "calls.shards.json").write_text("{}")
    (R / "frontier" / "primary-uniform" / "models").mkdir(parents=True)
    assert PL.expected_outputs(R)["missing"] == []
