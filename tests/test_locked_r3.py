"""scripts/locked_r3.py: R3 under an nvidia-smi SM clock lock, with no GPU.

Everything here runs against a PLANTED nvidia-smi on PATH (and a planted sudo
that passes `-n` through to it) and a PLANTED R3: a short script that prints
R3's two plan lines (`session`, `WRITES TO`), writes `cells.csv` rows at the
clock the planted card delivers under the lock the planted nvidia-smi holds,
and a `report.json` carrying its session tag. The regression this file exists
for is the Lambda H100's 1890 MHz attempt (2026-09-25): the driver found its
run directory by modification time, read the 1980 MHz attempt's two cells at
1830 MHz, and called the 1890 lock a slip it never tested.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import locked_r3 as LR  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

SCRIPT = REPO / "scripts" / "locked_r3.py"
TAG = "alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z"
H100 = (REPO / "results" / "published" / "2026-09-25-nvidia_h100_80gb_hbm3-session"
        / "session" / "locked-r3")

#: nvidia-smi as the H100 answered it, every call logged as a JSON argv line.
#: `-lgc` writes the lock to a file the planted R3 reads; `-rgc` removes it.
#: PLANT_LGC_SET makes `-lgc` set (and say it set) a clock other than the one
#: asked; PLANT_BUSY_FROM=N lists a compute process (pid 4242) from the N-th
#: `--query-compute-apps` call on, a card another process still holds.
PLANTED_SMI = r'''
import json, os, sys
d = os.environ["PLANT_DIR"]
args = sys.argv[1:]
with open(os.path.join(d, "smi.log"), "a") as fh:
    fh.write(json.dumps(args) + "\n")
lockf = os.path.join(d, "lock")
if args[:1] == ["-lgc"]:
    lo, hi = args[1].split(",")
    code = int(os.environ.get("PLANT_LGC_RC", "0"))
    if code:
        print("Setting locked GPU clocks is not supported: Insufficient Permissions")
        sys.exit(code)
    lo = hi = os.environ.get("PLANT_LGC_SET") or lo
    open(lockf, "w").write(lo)
    print(f'GPU clocks set to "(gpuClkMin {lo}, gpuClkMax {hi})" for GPU 00000000:06:00.0')
elif args[:1] == ["-rgc"]:
    code = int(os.environ.get("PLANT_RGC_RC", "0"))
    if code:
        print("Unable to reset the locked clocks")
        sys.exit(code)
    if os.path.exists(lockf):
        os.remove(lockf)
    print("All done.")
elif args[:1] == ["-pm"]:
    print("Enabled persistence mode for GPU 00000000:06:00.0.")
elif args[:3] == ["-q", "-d", "SUPPORTED_CLOCKS"]:
    for f in json.loads(os.environ.get("PLANT_SUPPORTED", "[1980, 1965, 1890, 1800, 1710]")):
        print(f"            Graphics                          : {f} MHz")
elif any(a.startswith("--query-compute-apps") for a in args):
    with open(os.path.join(d, "apps.log"), "a") as fh:
        fh.write("x\n")
    n = len(open(os.path.join(d, "apps.log")).read().splitlines())
    busy_from = int(os.environ.get("PLANT_BUSY_FROM", "0"))
    if busy_from and n >= busy_from:
        print("4242")
elif "-lms" in args:
    print("timestamp, clocks.current.sm [MHz]")
elif any(a.startswith("--query-gpu=name") for a in args):
    pm = os.environ.get("PLANT_PM", "Enabled")
    print(f"NVIDIA H100 80GB HBM3, GPU-planted, {pm}, 700.00 W, 1980 MHz")
elif any(a.startswith("--query-gpu") for a in args):
    print("1830 MHz, 1980 MHz, 0x0000000000000000, 72.38 W, 700.00 W")
'''

PLANTED_SUDO = r'''
import os, sys
args = sys.argv[1:]
with open(os.path.join(os.environ["PLANT_DIR"], "sudo.log"), "a") as fh:
    fh.write(" ".join(args) + "\n")
if args[:1] != ["-n"]:
    sys.exit("planted sudo: -n expected")
os.execvp(args[1], args[1:])
'''

#: R3 as the driver sees it: the plan's two lines, the directory (named like
#: R3's, the tag only in the hash, as on the H100), an alignment probe's worth
#: of silence, then one cell per PLANT_CELL_S at the clock the planted card
#: reads under the lock, then report.json with the tag. Keyed "<lock>:<G>":
#: PLANT_NOPAGE exits with that code before the first cell, PLANT_SLOW waits
#: that long per cell instead, PLANT_EXIT exits with that code AFTER writing
#: the page.
PLANTED_R3 = r'''
import hashlib, json, os, sys, time
args = sys.argv[1:]
G = int(args[args.index("--group-m") + 1])
tag = args[args.index("--session-tag") + 1]
d = os.environ["PLANT_DIR"]
with open(os.path.join(d, f"pid-{tag}-g{G}"), "w") as fh:
    fh.write(str(os.getpid()))
with open(os.path.join(d, f"args-{tag}-g{G}"), "w") as fh:
    fh.write(json.dumps(args))
h = hashlib.sha1(f"{tag}/{G}".encode()).hexdigest()[:8]
run = os.path.join(os.environ["MOE_RESULTS_DIR"], "private_weight_reference",
                   "nvidia_h100_80gb_hbm3-bm32-budget200.0-declauto-dtypebf16-duty0.25"
                   f"-g{G}-l2flushtrue-modelmixtral_8-{h}")
print("experiment  private_weight_reference / " + os.path.basename(run))
print("session     " + tag)
print("card        nvidia_h100_80gb_hbm3")
print("WRITES TO   " + run)
os.makedirs(run, exist_ok=True)
open(os.path.join(run, "DEVICE"), "w").write("GPU-planted\n")
time.sleep(float(os.environ.get("PLANT_PROBE_S", "0.3")))
lockf = os.path.join(d, "lock")
lock = int(open(lockf).read()) if os.path.exists(lockf) else None
reads = json.loads(os.environ.get("PLANT_READS", "{}"))
clock = reads.get(f"{lock}:{G}", reads.get(str(lock), lock or 1980))
nopage = json.loads(os.environ.get("PLANT_NOPAGE", "{}"))
if f"{lock}:{G}" in nopage:
    print("Traceback: planted crash before the first cell")
    sys.exit(nopage[f"{lock}:{G}"])
cells = os.path.join(run, "cells.csv")
with open(cells, "w") as fh:
    fh.write("arm,repeat,sm_clock_load_mhz,status\n")
cell_s = json.loads(os.environ.get("PLANT_SLOW", "{}")).get(
    f"{lock}:{G}", float(os.environ.get("PLANT_CELL_S", "0.05")))
for i in range(int(os.environ.get("PLANT_CELLS", "4"))):
    time.sleep(cell_s)
    with open(cells, "a") as fh:
        fh.write(f"shared,{i},{float(clock)},ok\n")
rc = json.loads(os.environ.get("PLANT_EXIT", "{}")).get(f"{lock}:{G}", 1 if G == 1 else 0)
with open(os.path.join(run, "report.json"), "w") as fh:
    json.dump({"session_tag": tag, "exit": rc}, fh)
sys.exit(rc)
'''


def _executable(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)


class World:
    """A planted box: nvidia-smi, sudo and R3, a results root, an out dir."""

    def __init__(self, tmp: Path, monkeypatch) -> None:
        self.tmp = tmp
        self.bin = tmp / "bin"
        self.plant = tmp / "plant"
        self.results = tmp / "results"
        self.out = tmp / "out"
        self.bin.mkdir()
        self.plant.mkdir()
        _executable(self.bin / "nvidia-smi", PLANTED_SMI)
        _executable(self.bin / "sudo", PLANTED_SUDO)
        self.r3 = tmp / "planted_r3.py"
        self.r3.write_text(PLANTED_R3)
        self.mp = monkeypatch
        monkeypatch.setenv("PATH", f"{self.bin}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("PLANT_DIR", str(self.plant))
        for var in ("MOE_RESULTS_DIR", "PLANT_READS", "PLANT_NOPAGE", "PLANT_LGC_RC",
                    "PLANT_RGC_RC", "PLANT_PM", "PLANT_CELL_S", "PLANT_CELLS",
                    "PLANT_LGC_SET", "PLANT_BUSY_FROM", "PLANT_SLOW", "PLANT_EXIT"):
            monkeypatch.delenv(var, raising=False)

    def set(self, **env) -> World:
        for k, v in env.items():
            self.mp.setenv(k, v if isinstance(v, str) else json.dumps(v))
        return self

    @property
    def pwr(self) -> Path:
        return self.results.resolve() / "private_weight_reference"

    def argv(self, *extra, locks=(1980, 1890, 1800, 1710), groups=(1, 2), sample_ms=0):
        return ["--session-tag", TAG, "--locks", *map(str, locks),
                "--groups", *map(str, groups), "--results-dir", str(self.results),
                "--out-dir", str(self.out), "--python", sys.executable,
                "--r3-script", str(self.r3), "--poll-s", "0.05", "--stop-grace-s", "10",
                "--idle-cap-s", "2", "--sample-ms", str(sample_ms), *extra]

    def calls(self) -> list[list[str]]:
        log = self.plant / "smi.log"
        return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []

    def clock_verbs(self) -> list[str]:
        """The calls that move the card, in order: `-lgc F`, `-rgc`, `-pm 1`."""
        out = []
        for c in self.calls():
            if c[:1] == ["-lgc"]:
                out.append(f"-lgc {c[1].split(',')[0]}")
            elif c[:1] in (["-rgc"], ["-pm"]):
                out.append(" ".join(c))
        return out

    def summary(self) -> dict:
        return json.loads((self.out / "summary.json").read_text())

    def status(self) -> str:
        return (self.out / "status").read_text()

    def pid(self, lock: int, G: int) -> int:
        return int((self.plant / f"pid-{TAG}-lock{lock}-g{G}").read_text())


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    return World(tmp_path, monkeypatch)


def _gone(pid: int, wait_s: float = 5.0) -> bool:
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _attempts(summary: dict) -> list[tuple[int, int, str]]:
    return [(a["lock"], a["G"], a["state"]) for a in summary["attempts"]]


# --------------------------------------------------------------------------
# 1. The run directory: named by the attempt's own R3, never found by time
# --------------------------------------------------------------------------

@pytest.mark.parametrize("old_has_report", [False, True],
                         ids=["stopped-mid-run-as-on-the-h100", "finished-with-its-own-tag"])
def test_the_attempt_ignores_a_previous_attempts_directory_with_a_close_mtime(
        world, old_has_report):
    """THE REGRESSION TEST. A previous attempt's directory, the same visible
    name up to the hash, its cells.csv just written (1830 MHz, under the 1890
    lock by more than a step) and a different tag: the 2026-09-25 logic took
    it as the 1890 attempt's, because its mtime fell inside the 5 s slack,
    and stopped the 1890 run as a slip. The attempt must read only the
    directory its own R3 named, find its own cells at 1890 and hold."""
    old = world.pwr / ("nvidia_h100_80gb_hbm3-bm32-budget200.0-declauto-dtypebf16-duty0.25"
                       "-g1-l2flushtrue-modelmixtral_8-bf2b7e51")
    old.mkdir(parents=True)
    (old / "cells.csv").write_text("arm,repeat,sm_clock_load_mhz,status\n"
                                   "shared,0,1830.0,ok\nprivate,0,1830.0,ok\n")
    if old_has_report:
        (old / "report.json").write_text(json.dumps({"session_tag": f"{TAG}-lock1980"}))
    (world.plant / "lock").write_text("1890")        # as if -lgc 1890,1890 had run
    run = LR.prepare(world.argv(locks=(1890,), groups=(1,)))
    run.out_dir.mkdir(parents=True)
    os.utime(old / "cells.csv")                       # modified NOW, inside any slack
    att = run.attempt(1890, 1)
    assert att.state == LR.HELD, att.line()
    assert att.dir and Path(att.dir).name != old.name, att.line()
    assert Path(att.dir).parent == world.pwr
    assert (att.cells, att.worst_mhz, att.slip_mhz) == (4, 1890.0, None), att.line()
    assert json.loads((Path(att.dir) / "report.json").read_text())["session_tag"] == \
        f"{TAG}-lock1890"


def test_the_h100_ladder_replayed_tests_the_1890_lock_it_skipped(world):
    """The whole 2026-09-25 sequence on a planted card that reads 1830 MHz
    under the 1980 lock and holds 1890: 1980 slips at G=1, then 1890 is tried
    on its OWN cells and holds at every G. The old driver's summary named the
    1980 directory for the 1890 attempt and stepped down to 1800."""
    world.set(PLANT_READS={"1980": 1830})
    rc = LR.main(world.argv(sample_ms=500))
    s = world.summary()
    assert rc == exit_codes.DONE, world.status()
    assert _attempts(s) == [(1980, 1, "slipped"), (1890, 1, "held"), (1890, 2, "held")]
    first, second = s["attempts"][0], s["attempts"][1]
    assert first["slip_mhz"] == 1830.0 and second["worst_mhz"] == 1890.0
    assert Path(first["dir"]).name != Path(second["dir"]).name, "the 1890 attempt read 1980's"
    assert s["lock"] == 1890 and [p["G"] for p in s["pages"]] == [1, 2]
    assert world.clock_verbs() == ["-lgc 1980", "-lgc 1890", "-rgc"]
    assert (world.out / "smi.csv").exists(), "the sampler wrote nothing"
    args = json.loads((world.plant / f"args-{TAG}-lock1890-g2").read_text())
    assert args[-4:] == ["--group-m", "2", "--session-tag", f"{TAG}-lock1890"]
    assert args[:len(LR.DEFAULT_R3_ARGS)] == list(LR.DEFAULT_R3_ARGS)


def test_own_run_dir_takes_only_this_attempts_new_directory_under_its_own_tag(tmp_path):
    """Each refusal of `own_run_dir`, and the wait before the plan is printed."""
    pwr = tmp_path / LR.PWR_DIR
    new = pwr / "card-g1-aaaaaaaa"
    tag = f"{TAG}-lock1890"
    plan = f"experiment  x\nsession     {tag}\nWRITES TO   {new}\n"
    assert LR.own_run_dir(tag, "", pwr, set()) == LR.Named(), "no plan yet: wait"
    assert LR.own_run_dir(tag, f"session     {tag}\n", pwr, set()) == LR.Named()
    assert LR.own_run_dir(tag, plan, pwr, set()).path == new
    other = LR.own_run_dir(f"{TAG}-lock1980", plan, pwr, set())
    assert other.path is None and "not this attempt's" in other.error
    resumed = LR.own_run_dir(tag, plan, pwr, {new.name})
    assert resumed.path is None and "was on disk before this attempt started" in resumed.error
    outside = LR.own_run_dir(tag, plan, tmp_path / "elsewhere", set())
    assert outside.path is None and "outside" in outside.error
    new.mkdir(parents=True)
    (new / "report.json").write_text(json.dumps({"session_tag": f"{TAG}-lock1980"}))
    wrong = LR.own_run_dir(tag, plan, pwr, set())
    assert wrong.path is None and "records session tag" in wrong.error
    (new / "report.json").write_text(json.dumps({"session_tag": tag}))
    assert LR.own_run_dir(tag, plan, pwr, set()).path == new


def test_the_plan_parser_reads_r3s_own_dry_run(tmp_path):
    """The two plan lines the driver keys on, as the real R3 prints them: its
    --dry-run off-GPU plans a `nocard` run, writes nothing, and names the
    directory under MOE_RESULTS_DIR with the tag on its session line. A
    reworded plan fails here, not on a rented card."""
    tag = f"{TAG}-lock1710"
    env = {**os.environ, "MOE_RESULTS_DIR": str(tmp_path), "PYTHONUNBUFFERED": "1"}
    done = subprocess.run([sys.executable, str(LR.R3_SCRIPT), *LR.DEFAULT_R3_ARGS,
                           "--group-m", "1", "--session-tag", tag, "--dry-run"],
                          env=env, capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert done.returncode == exit_codes.REFUSED, done.stdout[-1500:] + done.stderr[-1500:]
    named = LR.own_run_dir(tag, done.stdout, tmp_path / LR.PWR_DIR, set())
    assert named.error == "" and named.path is not None, done.stdout[:3000]
    assert named.path.parent == tmp_path / LR.PWR_DIR and "-g1-" in named.path.name
    assert not list(tmp_path.iterdir()), "R3's dry run wrote something"


def test_the_published_h100_logs_each_name_their_own_directory():
    """The record: every 2026-09-25 attempt's log names its own directory
    under its own tag, the 1890 attempt's included (`e0a0087e`), while that
    driver's summary.json named the 1980 attempt's `bf2b7e51` for it."""
    summary = json.loads((H100 / "summary.json").read_text())
    by_attempt = {(a["lock"], a["G"]): Path(a["dir"]).name for a in summary["attempts"]
                  if a.get("dir")}
    assert by_attempt[(1890, 1)].endswith("bf2b7e51"), "the defect as published"
    for log in sorted(H100.glob("r3-g*-lock*.log")):
        lock = int(log.stem.rsplit("lock", 1)[1])
        session, writes = LR.plan_names(log.read_text(errors="replace"))
        assert session == f"{TAG}-lock{lock}", log.name
        assert writes and "/private_weight_reference/" in writes, log.name
    _, writes = LR.plan_names((H100 / "r3-g1-lock1890.log").read_text(errors="replace"))
    assert writes.endswith("-e0a0087e")


def test_a_row_still_being_written_is_not_read_as_a_cell(tmp_path):
    """R3 appends while the driver reads: '1710.0' cut to '17' would be a
    17 MHz slip that never happened. Only rows ending in a newline count,
    and a row with no clock carries none."""
    cells = tmp_path / "cells.csv"
    cells.write_text("arm,repeat,sm_clock_load_mhz,status\nshared,0,1710.0,ok\n"
                     "private,0,,error\nshared,1,17")
    assert LR.cell_clocks(cells) == [1710.0]
    assert LR.off_lock(LR.cell_clocks(cells), 1710) == (None, None)
    assert LR.off_lock([1710.0, 1695.0, 1680.0], 1710) == (1680.0, None), "one step is not a slip"
    assert LR.off_lock([1710.0, 1740.0], 1710) == (None, 1740.0)
    assert LR.cell_clocks(tmp_path / "absent.csv") == []


def test_a_plan_line_still_being_written_is_not_read(tmp_path):
    """The same rule for R3's log (2026-09-26 review): a poll that lands
    inside R3's plan write sees a cut last line. Cut in the directory's
    parent it names a path outside the results root, cut in its name a
    directory that will never exist, cut in the tag another session; each
    would end the attempt NOT_OWN_DIR (or read the wrong directory) for
    good. An unterminated line is not read: the attempt waits for it."""
    pwr = tmp_path / LR.PWR_DIR
    tag = f"{TAG}-lock1890"
    new = pwr / "card-g1-aaaaaaaa"
    head = f"experiment  x\nsession     {tag}\n"
    for cut in (f"{head}WRITES TO   {str(pwr)[:-6]}", f"{head}WRITES TO   {str(new)[:-3]}"):
        assert LR.plan_names(cut) == (tag, None), cut
        assert LR.own_run_dir(tag, cut, pwr, set()) == LR.Named(), cut
    assert LR.plan_names(f"session     {tag[:-2]}") == (None, None)
    assert LR.own_run_dir(tag, f"session     {tag[:-2]}", pwr, set()) == LR.Named()
    assert LR.own_run_dir(tag, f"{head}WRITES TO   {new}\n", pwr, set()).path == new


# --------------------------------------------------------------------------
# 2. The ladder: a slip steps down and restarts at the first G
# --------------------------------------------------------------------------

def test_a_slip_at_a_later_g_steps_down_and_restarts_from_the_first_g(world, monkeypatch):
    """1890 holds at G=1 and slips at G=2: the G=2 run is stopped before it
    finishes, 1800 is locked, and the ladder starts again at G=1, so every
    kept page shares one clock. The 1890 G=1 page is recorded, not kept. And
    summary.json is written as each attempt ends, not only at the end, so a
    SIGKILL leaves the finished attempts on disk."""
    world.set(PLANT_READS={"1890:2": 1860}, PLANT_CELL_S="0.2")
    saved, real_save = [], LR.Run.save

    def spy(self, payload):
        saved.append(json.loads(json.dumps(payload)))
        real_save(self, payload)
    monkeypatch.setattr(LR.Run, "save", spy)
    rc = LR.main(world.argv(locks=(1890, 1800), groups=(1, 2, 4)))
    assert [len(p["attempts"]) for p in saved if p["exit"] is None] == [1, 2, 3, 4, 5]
    s = world.summary()
    assert rc == exit_codes.DONE, world.status()
    assert _attempts(s) == [(1890, 1, "held"), (1890, 2, "slipped"),
                            (1800, 1, "held"), (1800, 2, "held"), (1800, 4, "held")]
    slipped = s["attempts"][1]
    assert slipped["slip_mhz"] == 1860.0 and slipped["cells"] < 4
    assert not (Path(slipped["dir"]) / "report.json").exists(), "the slipped run was not stopped"
    assert slipped["rc"] == -signal.SIGINT and "R3 exit on SIGINT" in world.status()
    assert _gone(world.pid(1890, 2))
    assert s["lock"] == 1800 and [(p["lock"], p["G"]) for p in s["pages"]] == [
        (1800, 1), (1800, 2), (1800, 4)]
    assert world.clock_verbs() == ["-lgc 1890", "-lgc 1800", "-rgc"]
    assert "stepping down to 1800 MHz and restarting at G=1" in world.status()


def test_every_lock_slipping_is_claim_fail_with_the_clock_reset(world, capsys):
    """No lock held: a RESULT about this card (CLAIM_FAIL), not a retry."""
    world.set(PLANT_READS={"1890": 1830, "1800": 1770})
    rc = LR.main(world.argv(locks=(1890, 1800), groups=(1, 2)))
    out = capsys.readouterr().out
    assert rc == exit_codes.CLAIM_FAIL, out[-2000:]
    assert _attempts(world.summary()) == [(1890, 1, "slipped"), (1800, 1, "slipped")]
    assert "RESULT: CLAIM C1 FAIL" in out and "RESULT: VALIDITY V2 PASS" in out
    assert "no lock on the ladder held: every one slipped" in out
    assert world.clock_verbs() == ["-lgc 1890", "-lgc 1800", "-rgc"]


@pytest.mark.parametrize("planted, state, why", [
    ({"PLANT_NOPAGE": {"1890:2": 4}}, "no-page", "R3 exited 4 and wrote no report.json"),
    ({"PLANT_NOPAGE": {"1890:2": 2}}, "no-page", "R3 exited 2 and wrote no report.json"),
    ({"PLANT_EXIT": {"1890:2": 4}}, "no-page", "R3 exited 4 ERROR"),
    ({"PLANT_READS": {"1890:2": 1980}}, "lock-not-in-force", "the lock is not in force"),
    ({"PLANT_SLOW": {"1890:2": 60}}, "timeout", "past --run-cap-s 5 s"),
    ({"PLANT_BUSY_FROM": 2}, "gpu-busy", "pid 4242"),
], ids=["r3-crashed", "r3-refused-after-a-page", "r3-wrote-its-page-then-exited-error",
        "cell-over-the-lock", "r3-ran-past-its-cap", "gpu-busy-between-g"])
def test_a_failure_a_lower_lock_cannot_fix_stops_the_ladder_invalid(world, capsys,
                                                                   planted, state, why):
    """R3 ending without a page (a crash, a refusal, a page under an ERROR
    exit), a cell OVER the lock (the lock is not in force), a run past
    --run-cap-s, or a card another process held between two G: none is a
    slip, so no step down. The G=1 page was measured first, so each is
    INVALID, with the clock reset."""
    world.set(**planted)
    rc = LR.main(world.argv("--run-cap-s", "5", locks=(1890, 1800), groups=(1, 2)))
    out = capsys.readouterr().out
    assert rc == exit_codes.INVALID, out[-2000:]
    s = world.summary()
    assert _attempts(s) == [(1890, 1, "held"), (1890, 2, state)]
    assert why in s["attempts"][1]["why"], s["attempts"][1]
    assert "RESULT: VALIDITY V2 FAIL" in out and "RESULT: CLAIM C1 UNKNOWN" in out
    assert world.clock_verbs() == ["-lgc 1890", "-rgc"], "stepped down past a non-slip"
    if state == "timeout":
        assert s["attempts"][1]["rc"] == -signal.SIGINT and _gone(world.pid(1890, 2))


def test_a_busy_card_at_a_step_down_is_waited_for_before_the_lock_never_under_it(world,
                                                                               capsys):
    """1890 slips at G=1, then another process holds the card: the 1800 lock
    is NOT taken while it runs (the 2026-09-26 review: a busy check made only
    after -lgc timed that process under a lock nobody set for it for up to
    --idle-cap-s). Something was measured, so the stop is INVALID."""
    world.set(PLANT_READS={"1890": 1830}, PLANT_BUSY_FROM=2)
    rc = LR.main(world.argv(locks=(1890, 1800), groups=(1, 2)))
    out = capsys.readouterr().out
    assert rc == exit_codes.INVALID, out[-2000:]
    assert _attempts(world.summary()) == [(1890, 1, "slipped")]
    assert world.clock_verbs() == ["-lgc 1890", "-rgc"], "locked a card another process held"
    assert "lock 1800 MHz not taken: a compute process (pid 4242)" in world.status()
    assert "RESULT: VALIDITY V2 FAIL" in out


def test_persistence_mode_off_is_turned_on_before_the_first_lock(world):
    world.set(PLANT_PM="Disabled")
    assert LR.main(world.argv(locks=(1710,), groups=(2,))) == exit_codes.DONE
    assert world.clock_verbs() == ["-pm 1", "-lgc 1710", "-rgc"]


# --------------------------------------------------------------------------
# 3. The reset: every exit a process can catch
# --------------------------------------------------------------------------

def test_the_clock_is_reset_and_r3_stopped_on_an_exception(world, monkeypatch):
    """A crash mid-attempt: R3 is stopped, -rgc runs, the summary says so,
    and the exception still propagates (the __main__ wrapper exits 4)."""
    before = signal.getsignal(signal.SIGTERM)
    world.set(PLANT_CELL_S="5")

    def crash(_path):
        raise RuntimeError("planted")
    monkeypatch.setattr(LR, "cell_clocks", crash)
    with pytest.raises(RuntimeError, match="planted"):
        LR.main(world.argv(locks=(1890, 1800), groups=(1,)))
    assert world.clock_verbs() == ["-lgc 1890", "-rgc"]
    assert _gone(world.pid(1890, 1)), "R3 outlived the driver"
    s = world.summary()
    assert s["exit"] == exit_codes.ERROR and "RuntimeError: planted" in s["stopped_by"]
    assert s["reset"]["rc"] == 0 and "SM clock reset" in world.status()
    assert signal.getsignal(signal.SIGTERM) == before


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP", "SIGINT"])
def test_the_clock_is_reset_and_r3_stopped_on_a_signal(world, monkeypatch, capsys, signame):
    """The kill lands while R3 runs: R3 is stopped, the clock reset, the
    handler put back, and the exit is 128 + the signal (Ctrl-C still ends as
    the KeyboardInterrupt it always raised)."""
    sig = getattr(signal, signame)
    before = signal.getsignal(sig)
    world.set(PLANT_CELL_S="5")
    real = LR.cell_clocks

    def killed(path):
        assert signal.getsignal(sig) not in (signal.SIG_DFL, before), "no handler installed"
        signal.raise_signal(sig)
        return real(path)
    monkeypatch.setattr(LR, "cell_clocks", killed)
    expect = KeyboardInterrupt if sig == signal.SIGINT else SystemExit
    with pytest.raises(expect) as got:
        LR.main(world.argv(locks=(1890,), groups=(1,)))
    if expect is SystemExit:
        assert got.value.code == 128 + sig
    assert world.clock_verbs() == ["-lgc 1890", "-rgc"]
    assert _gone(world.pid(1890, 1)), "R3 outlived the driver"
    assert signal.getsignal(sig) == before
    err = capsys.readouterr().err
    assert f"{signame} caught: stopping R3, resetting the SM clock, then exiting " \
           f"{128 + sig}" in err
    assert world.summary()["exit"] == 128 + sig
    assert f"stopped by {signame}" in world.status()


def test_sigterm_to_the_real_process_resets_the_clock_and_exits_143(world):
    """The same, end to end: the driver as its own process, R3 as its child,
    SIGTERM from outside, as `kill` or a VM's shutdown sends it."""
    world.set(PLANT_CELL_S="30")
    proc = subprocess.Popen([sys.executable, str(SCRIPT), *world.argv(locks=(1890,),
                                                                      groups=(1,))],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=dict(os.environ), cwd=str(REPO))
    pid_file = world.plant / f"pid-{TAG}-lock1890-g1"
    end = time.monotonic() + 60
    while not pid_file.exists() and time.monotonic() < end and proc.poll() is None:
        time.sleep(0.05)
    assert pid_file.exists(), proc.communicate(timeout=30)
    time.sleep(0.3)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 128 + signal.SIGTERM, out[-1500:] + err[-1500:]
    assert "SIGTERM caught: stopping R3, resetting the SM clock, then exiting 143" in err
    assert world.clock_verbs() == ["-lgc 1890", "-rgc"]
    assert _gone(int(pid_file.read_text())), "R3 outlived the driver"


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGINT"])
def test_a_kill_as_r3_is_launched_still_stops_r3_and_resets_the_clock(world, monkeypatch,
                                                                      capsys, signame):
    """The kill lands after Popen has started R3 but before the driver holds
    the child (the 2026-09-26 review): the handler raised inside the launch,
    the Popen object was lost, the clock was reset and R3 went on timing a
    page tagged -lock1890 with no lock. Here the signal is raised the moment
    the planted R3 exists, before Popen even returns."""
    sig = getattr(signal, signame)
    world.set(PLANT_CELL_S="30")
    real, launched = subprocess.Popen, []

    def racing(argv, *a, **kw):
        child = real(argv, *a, **kw)
        if str(world.r3) in map(str, argv) and not launched:
            launched.append(child)
            signal.raise_signal(sig)
        return child
    monkeypatch.setattr(LR.subprocess, "Popen", racing)
    try:
        expect = KeyboardInterrupt if sig == signal.SIGINT else SystemExit
        with pytest.raises(expect) as got:
            LR.main(world.argv(locks=(1890, 1800), groups=(1,)))
        if expect is SystemExit:
            assert got.value.code == 128 + sig
        assert launched, "R3 was never launched"
        assert _gone(launched[0].pid), "R3 outlived the driver, timing with no lock"
        assert world.clock_verbs() == ["-lgc 1890", "-rgc"]
        assert f"{signame} caught" in capsys.readouterr().err
        assert world.summary()["exit"] == 128 + sig
    finally:
        for child in launched:
            if child.poll() is None:
                child.kill()
                child.wait()


def test_a_sighup_nohup_ignored_stays_ignored(world, monkeypatch):
    """`nohup` starts the driver with SIGHUP ignored; a dropped SSH session
    must not end a lock run that nohup was asked to keep alive."""
    old = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        real, sent = LR.cell_clocks, []

        def hangup(path):
            if not sent:
                sent.append(1)
                signal.raise_signal(signal.SIGHUP)
            return real(path)
        monkeypatch.setattr(LR, "cell_clocks", hangup)
        assert LR.main(world.argv(locks=(1890,), groups=(1,))) == exit_codes.DONE
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, old)


def test_a_reset_that_fails_after_a_lock_is_error_and_says_still_locked(world, capsys):
    world.set(PLANT_RGC_RC="1")
    rc = LR.main(world.argv(locks=(1890,), groups=(1,)))
    out = capsys.readouterr().out
    assert rc == exit_codes.ERROR, out[-2000:]
    assert "the card is STILL LOCKED at 1890 MHz" in out
    assert "SM CLOCK NOT RESET" in world.status()


# --------------------------------------------------------------------------
# 4. The dry run, and what is refused before anything is measured
# --------------------------------------------------------------------------

def test_the_dry_run_prints_the_plan_and_touches_nothing(world, capsys):
    rc = LR.main(world.argv("--dry-run", locks=(1980, 1710), groups=(1, 64)))
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED
    assert not world.calls() and not (world.plant / "sudo.log").exists(), "nvidia-smi was called"
    assert not world.out.exists() and not world.results.exists(), "the dry run wrote something"
    assert not list(world.plant.glob("pid-*")), "R3 ran"
    for lock in (1980, 1710):
        for G in (1, 64):
            assert f"--group-m {G} --session-tag {TAG}-lock{lock}" in out
    assert "nvidia-smi -lgc F,F" in out and "nvidia-smi -rgc on every exit" in out
    assert "REFUSED. Nothing was measured and nothing was written." in out
    assert "RESULT:" not in out


@pytest.mark.parametrize("extra, locks, needle", [
    ((), (1710, 1800), "strictly descending"),
    (("--", "--session-tag", "x"), (1710,), "--session-tag (the driver sets"),
    (("--", "--grou", "4"), (1710,), "--grou (the driver sets G"),
    (("--", "--run-id=r"), (1710,), "every attempt would share one directory"),
    (("--", "--out", "/x"), (1710,), "--out (the driver hands R3"),
    (("--", "--o", "/x"), (1710,), "--o (the driver hands R3"),
    (("--", "--o=/x"), (1710,), "--o=/x (the driver hands R3"),
    (("--", "--g", "4"), (1710,), "--g (the driver sets G"),
], ids=["ascending", "tag", "abbreviated-g", "run-id", "out", "out-as-o", "out-as-o-equals",
        "g-as-g"])
def test_arguments_the_ladder_cannot_run_are_refused_before_anything(world, capsys,
                                                                    extra, locks, needle):
    argv = world.argv(locks=locks, groups=(1,))
    rc = LR.main([*argv, *extra])
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED and needle in out, out
    assert not world.calls() and not world.out.exists()


def test_every_token_r3_reads_as_an_owned_flag_is_refused_and_no_other():
    """Held to R3's OWN parser (2026-09-26 review: `--o /x` passed the old
    three-character floor and R3 read it as `--out /x`). argparse takes an
    option spelled out, else the one option the token begins (allow_abbrev).
    Every such token for an owned flag must be refused, `=value` or not, and
    none for another flag, so a legitimate pass-through like `--rep 9` is
    not. A prefix shared by several options is R3's own error; the driver
    may refuse it too."""
    import private_weight_reference as R3
    ap = R3.build_parser()
    assert ap.allow_abbrev and ap.parse_args(["--o", "/x"]).out == Path("/x"), "no escape to test"
    options = sorted(s for s in ap._option_string_actions if s.startswith("--"))
    assert set(LR.OWNED_FLAGS) <= set(options), "an owned flag R3 no longer has"
    checked = 0
    for opt in options:
        for k in range(3, len(opt) + 1):
            token = opt[:k]
            hits = [token] if token in options else [o for o in options if o.startswith(token)]
            if len(hits) != 1:
                continue
            want = hits[0] if hits[0] in LR.OWNED_FLAGS else None
            assert LR.owned_flag(token) == want, (token, hits[0])
            assert LR.owned_flag(f"{token}=v") == want, (token, hits[0])
            checked += 1
    assert checked > 100 and LR.owned_flag("--rep") is None and LR.owned_flag("--") is None


def test_a_busy_card_before_the_first_lock_is_refused_with_no_lock_taken(world, capsys):
    """Another process on the card before anything ran (a chain arm still
    exiting): no -lgc at all, so that process never runs under a lock nobody
    set for it, and REFUSED, because nothing was measured (the 2026-09-26
    review: the old order locked first, polled the busy card under the lock
    for --idle-cap-s, then exited INVALID with zero cells)."""
    world.set(PLANT_BUSY_FROM=1)
    rc = LR.main(world.argv(locks=(1890, 1800), groups=(1,)))
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED, out[-1500:]
    assert world.clock_verbs() == ["-rgc"], "locked a card another process held"
    assert not list(world.plant.glob("pid-*")), "R3 ran on a busy card"
    assert "REFUSED, nothing measured: a compute process (pid 4242)" in out
    assert "RESULT:" not in out


def test_r3_refusing_at_the_first_attempt_is_refused_not_invalid(world, capsys):
    """R3 prints its plan and refuses (exit 2) before its first cell, at the
    ladder's first attempt: nothing was measured, so the driver REFUSES too,
    with no RESULT line and no step down. After a page (see the INVALID
    cases above) the same refusal is INVALID."""
    world.set(PLANT_NOPAGE={"1890:1": 2})
    rc = LR.main(world.argv(locks=(1890, 1800), groups=(1, 2)))
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED, out[-1500:]
    assert _attempts(world.summary()) == [(1890, 1, "no-page")]
    assert world.clock_verbs() == ["-lgc 1890", "-rgc"], "stepped down past a refusal"
    assert "REFUSED, nothing measured: R3 refused before its first timed cell" in out
    assert "RESULT:" not in out


@pytest.mark.parametrize("rgc_rc, code", [("0", exit_codes.REFUSED), ("1", exit_codes.ERROR)],
                         ids=["reset", "reset-failed"])
def test_an_lgc_that_sets_another_clock_is_not_taken(world, capsys, rgc_rc, code):
    """-lgc 1890,1890 exits 0 but says it set 1875: that is not the lock, and
    R3 never runs under it. Nothing measured is REFUSED, unless the reset
    then fails: a card left locked is ERROR whatever else happened."""
    world.set(PLANT_LGC_SET="1875", PLANT_RGC_RC=rgc_rc)
    rc = LR.main(world.argv(locks=(1890, 1800), groups=(1,)))
    out = capsys.readouterr().out
    assert rc == code, out[-1500:]
    assert "set gpuClkMin 1875, gpuClkMax 1875: a clock other than the lock" in out
    assert world.clock_verbs() == ["-lgc 1890", "-rgc"]
    assert not list(world.plant.glob("pid-*")), "R3 ran under a clock other than the lock"
    assert ("the card is STILL LOCKED" in out) == (code == exit_codes.ERROR)


def test_a_lock_the_card_does_not_support_is_refused_before_any_lgc(world, capsys):
    rc = LR.main(world.argv(locks=(1890, 1760), groups=(1,)))
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED, out[-1500:]
    assert "--locks [1760] are not supported graphics clocks here" in out
    assert world.clock_verbs() == ["-rgc"], "an unsupported lock must never reach -lgc"


def test_lgc_refused_at_the_first_lock_is_refused_in_the_drivers_own_words(world, capsys):
    world.set(PLANT_LGC_RC="4", PLANT_RGC_RC="4")
    rc = LR.main(world.argv(locks=(1890, 1800), groups=(1,)))
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED, out[-1500:]
    assert "exited 4: Setting locked GPU clocks is not supported" in out
    assert world.clock_verbs() == ["-lgc 1890", "-rgc"]
    assert not list(world.plant.glob("pid-*")), "R3 ran with no lock"


def test_a_tag_already_on_disk_is_refused_so_r3_resumes_nothing(world, capsys):
    """A finished run records one of this ladder's tags, or the out dir holds
    this tag's attempts: R3 would resume those directories and score their
    cells as new ones."""
    done = world.pwr / "card-g1-done"
    done.mkdir(parents=True)
    (done / "report.json").write_text(json.dumps({"session_tag": f"{TAG}-lock1800"}))
    assert LR.main(world.argv(locks=(1890, 1800), groups=(1,))) == exit_codes.REFUSED
    assert "already record this ladder's tags" in capsys.readouterr().out
    (done / "report.json").unlink()
    world.out.mkdir()
    (world.out / "r3-g1-lock1890.log").write_text("")
    assert LR.main(world.argv(locks=(1890, 1800), groups=(1,))) == exit_codes.REFUSED
    assert "already holds attempts under this tag" in capsys.readouterr().out
    assert not world.calls()
