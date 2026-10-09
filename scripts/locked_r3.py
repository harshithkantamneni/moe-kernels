#!/usr/bin/env python3
"""R3's timed pages under an nvidia-smi SM clock LOCK, at the highest lock the card holds.

    . ~/moe/env.sh && cd "$REPO" && export MOE_RESULTS_DIR=$RESULTS_ROOT/gaps-$MOE_CARD
    python3 scripts/locked_r3.py --session-tag <chain session> --locks 1980 1890 1800 1710 \\
        --dry-run                                         # the plan, touching nothing
    python3 scripts/locked_r3.py --session-tag <chain session> --locks 1980 1890 1800 1710 \\
        --groups 1 2 4 16 64 -- --model mixtral-8x7b --block-m 32 --treads 6 --repeats 9 \\
        --duty 0.25 --seed 0

Runs on a VM where we are root or have passwordless sudo (Lambda), under the
VM's own `python3`: it imports nothing but `moe.bench.exit_codes`, and R3 runs
in its own interpreter (`--python`, default `$PY_VLLM`).

WHY A LOCK, AND WHY THE LOCK IS FOUND ON THE RUN ITSELF. On the Lambda GH200
(2026-09-25) R3's duty-0.25 pages failed V7: the shared and private arms ran at
different clocks. A 1965 MHz lock was pulled to 1785-1830 MHz during R3's
bursts in the first cells, while 1710 held exactly. So no lock is chosen from
the card's maximum. For each lock F, top down: `nvidia-smi -lgc F,F`, then R3
at every G in turn under the session tag `<tag>-lock<F>`, reading each timed
cell's under-load clock (`cells.csv` column `sm_clock_load_mhz`) as it lands.
A cell more than one 15 MHz step under F is a SLIP: that R3 run is stopped,
the next lower lock is taken, and the ladder starts again at the first G, so
every kept page shares one clock. A cell more than one step OVER F means the
lock is not in force at all (a reset from elsewhere, a lost setting), and that
stops the ladder rather than stepping down, because a lower lock would not fix
it.

THE RUN DIRECTORY IS NAMED BY THE ATTEMPT'S OWN R3, NEVER FOUND BY TIME
(2026-09-26). The driver the H100 ran on 2026-09-25 picked the newest
`*-duty0.25-g<G>-*` directory whose `cells.csv` was modified at or after the
attempt's start less 5 s, and never checked the session tag. The 1980 MHz
attempt's last cell landed 4 s before the 1890 attempt started, so the 1890
attempt's first check read the 1980 attempt's two cells at 1830 MHz, called
them a slip and stopped its own run 12 s in, still in the alignment probe:
the 1890 lock was never tested (the published H100 session's README, "The
1890 attempt measured nothing"; its `summary.json` names `bf2b7e51`, the 1980
directory, for the 1890 attempt, whose own directory `e0a0087e` holds no
cells).

HOW R3 RECORDS ITS SESSION TAG, which is what decides how the directory is
found. Three places, and only one of them exists while the run is going:

  - the run id (`default_run_id`) hashes `--session-tag`, but the visible part
    of the id is cut at 96 characters and on these runs the tag never shows,
    so two attempts' directories differ only in the 8-character hash;
  - `report.json`'s `session_tag`, written once, when the run ENDS;
  - the plan R3 prints before its first cell: a `session` line with the tag
    and a `WRITES TO` line with the directory.

So each attempt reads its OWN log (opened fresh when the attempt starts) for
the plan's two lines, and takes the directory only when the `session` line is
exactly this attempt's tag, the directory sits under this run's results root,
and its name was NOT on disk when this attempt started (a directory that was
would be R3 resuming an earlier attempt's cells under this tag). When the run
ends, the directory's own `report.json` must record the same tag. No other
directory is ever read, whatever its modification time. The child runs with
PYTHONUNBUFFERED=1: redirected to a file, R3's stdout is block-buffered and
reaches the log a buffer at a time, so the `WRITES TO` line (5215 bytes into
the H100's plan) could wait there until R3 printed more or exited.

THE CLOCK IS RESET ON EVERY EXIT A PROCESS CAN CATCH. `ClockGuard` follows
`scripts/clock_elasticity.py`'s `ClockLock`, rule for rule: SIGTERM and SIGHUP
become SystemExit(128 + the signal) and SIGINT the KeyboardInterrupt Ctrl-C
raises, so the `with` block's exit, which ALWAYS runs `nvidia-smi -rgc`, runs
on a kill as well as on a return or an exception; a signal ALREADY IGNORED
when it is entered stays ignored (`nohup`'s SIGHUP); the first kill wins, the
handler ignoring all three before it unwinds, so a second Ctrl-C cannot abort
the reset. The running R3 child is sent SIGINT before the reset and waited
for after it; a kill that lands while R3 is being launched is held until the
guard has the child, so no R3 outlives the driver (ClockGuard's docstring,
2026-09-26). It is a copy and not an import because `clock_elasticity`
imports `moe.bench.timing`, which imports torch, and this driver runs under
the VM's system `python3`, as the H100's did. What no handler catches leaves
the card locked: SIGKILL. Reset by hand after one: `sudo -n nvidia-smi -rgc`.

EVERY LOCK MUST BE A SUPPORTED GRAPHICS CLOCK (`nvidia-smi -q -d
SUPPORTED_CLOCKS`), and the ladder must run top down. The driver would round
an unsupported lock and every page would name a clock that never ran. The
2026-09-25 driver snapped each lock down to a supported clock; this one
refuses instead, as `clock_elasticity --lock-clocks` does, so the tag names
the lock that ran.

WHAT IT WRITES, in `--out-dir` (default `$SESSION_ROOT/locked-r3/<tag>`):

    status                  one UTC line per event, appended
    summary.json            every attempt, the kept pages, the gates, every nvidia-smi call
    r3-g<G>-lock<F>.log     each attempt's R3 output, its plan first
    smi.csv                 nvidia-smi sampled every --sample-ms (0: none)

R3 writes its pages under `--results-dir` (default `$MOE_RESULTS_DIR`, else
`$RESULTS_ROOT/gaps-$MOE_CARD`, docs/LAMBDA.md section 3b's place for an arm
run by hand), which the driver hands the child as MOE_RESULTS_DIR.

EXIT CODES are `moe.bench.exit_codes`, through three gates:

  V1  every attempt read only the run directory its own R3 named under its own
      tag, created after the attempt started (FAIL: INVALID, 3)
  V2  the ladder reached a verdict: a lock held at every G, or every lock
      slipped. An R3 run that ended without a page, a run or ladder time cap,
      a GPU that stayed busy, a lock not in force or refused mid-ladder, once
      something was measured: FAIL, INVALID (3). Stepping down would not fix
      any of them.
  C1  a lock on the ladder held at every G: DONE (0). Every lock slipped:
      CLAIM_FAIL (1), a result about this card, not a retry.

A lock held means the clocks held; each page's own verdict is R3's, printed
beside it (a G=1 page reading CLAIM_FAIL under a held lock is R3's C1). Nothing
measured (a bad argument, a lock the card does not support, a card another
process still held or an -lgc refused before the first lock took, R3 refusing
before the first timed cell, a tag already on disk, `--dry-run`): REFUSED (2),
no RESULT line. The busy check runs BEFORE each -lgc, never under the lock,
so a process still on the card is never timed at a clock nobody set for it;
between two G at one lock it runs again, under that lock, for the R3 just
ended. A reset that FAILED: ERROR (4) whatever the gates said, so whatever runs
next stops before it measures on a card left locked (clock_elasticity's rule).
A caught kill: 128 + the signal (143 SIGTERM, 129 SIGHUP, 130 Ctrl-C), after
the reset and the summary, outside the table as in clock_elasticity's lock
mode. A crash: ERROR (4), after the reset.

SLIP POLICY (rental 6, 2026-10-09). `--slip-policy page`, the default, is
everything above, unchanged: the first slipped cell stops the page. Rental 5
lost two Mixtral pages that way (one repeat at 1635 and 1680 MHz ended each,
and R3 writes report.json only at the end of a page, so nothing was kept).
`--slip-policy cell` (histogram pages only) hands R3 the lock (`--lock-mhz`),
the policy and `--min-clean-repeats`; R3 flags each repeat under the lock by
more than one step (`lock_slip`, the same predicate as `off_lock` on the same
`sm_clock_load_mhz`), keeps it out of the cell median, drops a cell that can
no longer reach the minimum and stops the page once no cell can. This driver
then does not stop R3 on a slip: it records every slipped row in the attempt's
`slipped_cells` and lets the page finish. A page that ends with slips is
HELD_WITH_SLIPS, a held page (exit 0, kept in summary.json's pages, the ladder
goes on); the scorer decides what its slips void. A cell more than one step
OVER the lock still stops the ladder under either policy.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes  # noqa: E402  (torch-free)

#: One step of the NVML clock grid. A timed cell more than one step under the
#: lock is a slip; one step is the reading's own resolution and is not.
STEP_MHZ = 15
#: The study's G ladder, in the order the 2026-09-25 runs took it.
DEFAULT_GROUPS = (1, 2, 4, 16, 64)
#: R3's arguments when nothing follows `--`: the GH200 and H100 locked runs'
#: own (2026-09-25), the chain's R3 at duty 0.25.
DEFAULT_R3_ARGS = ("--model", "mixtral-8x7b", "--block-m", "32", "--treads", "6",
                   "--repeats", "9", "--duty", "0.25", "--seed", "0")
R3_SCRIPT = ROOT / "scripts" / "private_weight_reference.py"
#: R3's experiment directory under the results root (`out_dir` in its main).
PWR_DIR = "private_weight_reference"
#: R3 flags this driver owns, each with why it may not be passed through.
#: Matched as prefixes too, however short, because argparse accepts an
#: abbreviation (`owned_flag` says why no floor).
OWNED_FLAGS = {
    "--group-m": "the driver sets G per attempt (--groups)",
    "--session-tag": "the driver sets <tag>-lock<F> per lock (--session-tag)",
    "--run-id": "it bypasses the tag-keyed run id, so every attempt would share one directory",
    "--out": "the driver hands R3 its results root (--results-dir)",
    "--dry-run": "the driver's own --dry-run goes before the --",
    "--self-test": "a planted world measures nothing under a lock",
    "--read": "a re-read measures nothing",
    "--rescore": "a re-score measures nothing",
    "--probe-check": "the probe check writes no page",
    "--counter-child": "the counter child is dram_counter_route's, not a timed page",
    "--slip-policy": "the driver's own --slip-policy hands it to R3 with the lock",
    "--lock-mhz": "the driver sets the lock R3's slip predicate reads, per attempt",
    "--min-clean-repeats": "the driver's own --min-clean-repeats rides with --slip-policy cell",
}
#: The two lines of R3's plan (`plan_lines`) that name the attempt's session
#: and directory, anchored at column zero. tests/test_locked_r3.py holds this
#: parser to R3's own --dry-run output, so a reworded plan fails there.
PLAN_SESSION = re.compile(r"^session {2,}(\S.*?)\s*$", re.M)
PLAN_WRITES = re.compile(r"^WRITES TO {2,}(\S.*?)\s*$", re.M)
#: What `nvidia-smi -lgc F,F` prints when it takes: 'GPU clocks set to
#: "(gpuClkMin 1980, gpuClkMax 1980)" for GPU 00000000:06:00.0' (H100, driver
#: 580.105.08, `lgc-1980.txt`).
LGC_SET = re.compile(r"gpuClkMin\s+(\d+)\s*,\s*gpuClkMax\s+(\d+)")
READBACK_QUERY = ("--query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active,"
                  "power.draw,power.limit", "--format=csv,noheader")
CARD_QUERY = ("--query-gpu=name,uuid,persistence_mode,power.limit,clocks.max.sm",
              "--format=csv,noheader")
SAMPLER_QUERY = ("--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,"
                 "temperature.gpu,utilization.gpu,clocks_event_reasons.active")

HELD, SLIPPED = "held", "slipped"
#: Rental 6: a page that held under --slip-policy cell with at least one slipped repeat.
#: A held page everywhere HELD is (`kept`).
HELD_WITH_SLIPS = "held-with-slips"
KEPT = (HELD, HELD_WITH_SLIPS)
#: Attempt states that stop the ladder without a verdict: V2 FAILS on each.
NOT_OWN_DIR, NO_PAGE, OFF_LOCK_HIGH = "not-own-dir", "no-page", "lock-not-in-force"
TIMEOUT, LADDER_CAP, GPU_BUSY = "timeout", "ladder-cap", "gpu-busy"


class Refusal(Exception):
    """A precondition not met, before anything was measured: REFUSED (2)."""


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _one(text: str) -> str:
    return " ".join(str(text).split())


# --------------------------------------------------------------------------
# Reading what R3 wrote: pure functions of the files
# --------------------------------------------------------------------------

def cell_clocks(csv_path: Path) -> list[float]:
    """Every complete row's `sm_clock_load_mhz`, in file order.

    A ROW STILL BEING WRITTEN IS NOT A ROW. R3 appends one row per cell while
    this reads, so the text after the last newline is dropped: read mid-write,
    "1710.0" can arrive as "17", and a 17 MHz cell is a slip that never
    happened. A row whose clock is empty or not a number carries no clock and
    is not scored.
    """
    try:
        text = csv_path.read_text(errors="replace")
    except OSError:
        return []
    text = text[:text.rfind("\n") + 1]
    out: list[float] = []
    rows = csv.DictReader(io.StringIO(text))
    try:
        for row in rows:
            try:
                v = float(row.get("sm_clock_load_mhz") or "nan")
            except ValueError:
                continue
            if v == v and v > 0:
                out.append(v)
    except csv.Error:
        pass
    return out


def off_lock(clocks: list[float], lock: int) -> tuple[float | None, float | None]:
    """(the first cell more than one step UNDER the lock, the first more than
    one step OVER it), None where there is none. Under is a slip; over means
    the lock is not in force."""
    under = next((v for v in clocks if v < lock - STEP_MHZ), None)
    over = next((v for v in clocks if v > lock + STEP_MHZ), None)
    return under, over


def slipped_rows(csv_path: Path, lock: int) -> list[dict]:
    """Every complete row of `cells.csv` more than one step under the lock (`off_lock`'s
    predicate), as {row, arm, tiles, repeat, label, mhz}: what --slip-policy cell records
    in place of stopping. The label is the histogram page's (`detail` begins
    `histogram=<label>`), empty elsewhere."""
    try:
        text = csv_path.read_text(errors="replace")
    except OSError:
        return []
    text = text[:text.rfind("\n") + 1]
    out: list[dict] = []
    try:
        for i, row in enumerate(csv.DictReader(io.StringIO(text))):
            try:
                v = float(row.get("sm_clock_load_mhz") or "nan")
            except ValueError:
                continue
            if v == v and v > 0 and v < lock - STEP_MHZ:
                m = re.match(r"histogram=([^;]+)", row.get("detail") or "")
                out.append({"row": i, "arm": row.get("arm") or "", "tiles": row.get("tiles") or "",
                            "repeat": row.get("repeat") or "", "label": m.group(1) if m else "",
                            "mhz": v})
    except csv.Error:
        pass
    return out


def recorded_tag(run_dir: Path) -> str | None:
    """The session tag a run directory's own `report.json` records, None when
    it has no readable report yet (R3 writes it once, when the run ends)."""
    try:
        payload = json.loads((run_dir / "report.json").read_text())
    except (OSError, ValueError):
        return None
    tag = payload.get("session_tag") if isinstance(payload, dict) else None
    return tag if isinstance(tag, str) else None


def tags_on_disk(pwr_dir: Path) -> dict[str, list[str]]:
    """Every session tag a finished run under `pwr_dir` records, with its
    directories."""
    out: dict[str, list[str]] = {}
    if pwr_dir.is_dir():
        for d in sorted(pwr_dir.iterdir()):
            tag = recorded_tag(d) if d.is_dir() else None
            if tag:
                out.setdefault(tag, []).append(d.name)
    return out


def plan_names(log_text: str) -> tuple[str | None, str | None]:
    """(the session tag, the directory) R3's plan printed, each None until it
    has. The first of each: a report printed later may repeat a session line.

    A LINE STILL BEING WRITTEN IS NOT A LINE (2026-09-26), `cell_clocks`'s
    rule for the log: the patterns' `$` also matches at the end of the text,
    so a poll landing inside R3's plan write read "WRITES TO /home/ubuntu/moe/res"
    as a directory outside the results root and "session T-lock18" as another
    tag, and `own_run_dir` ended the attempt NOT_OWN_DIR for good. The text
    after the last newline is dropped, and the attempt waits for it."""
    log_text = log_text[:log_text.rfind("\n") + 1]
    s, w = PLAN_SESSION.search(log_text), PLAN_WRITES.search(log_text)
    return (s.group(1) if s else None, w.group(1) if w else None)


@dataclass
class Named:
    """The run directory an attempt's own R3 named: `path` once it is this
    attempt's, `error` when what it named cannot be, both empty while R3 has
    not printed its plan yet."""
    path: Path | None = None
    error: str = ""


def own_run_dir(tag: str, log_text: str, pwr_dir: Path, existed_before: set[str]) -> Named:
    """The attempt's run directory, found ONLY through its own R3's plan.

    `log_text` is this attempt's log and nothing else's. The directory is
    taken when the plan's `session` line is exactly `tag`, its `WRITES TO`
    directory sits in `pwr_dir`, its name was not in `pwr_dir` when the attempt
    started (`existed_before`, listed before R3 was launched), and its
    `report.json`, once written, records `tag`. No other directory is looked
    at, so one from a previous attempt is never read however close its
    modification time (the 2026-09-25 H100 defect; see the module docstring).
    """
    session, writes = plan_names(log_text)
    if session is None or writes is None:
        return Named()
    if session != tag:
        return Named(error=f"R3's plan names session {session!r}, not this attempt's {tag!r}")
    path = Path(writes)
    if path.resolve().parent != pwr_dir.resolve():
        return Named(error=f"R3 writes to {path}, outside this run's {pwr_dir}")
    if path.name in existed_before:
        return Named(error=f"{path.name} was on disk before this attempt started: R3 would "
                           f"resume an earlier run's cells under {tag!r}. Use a fresh "
                           "--session-tag")
    recorded = recorded_tag(path)
    if recorded is not None and recorded != tag:
        return Named(error=f"{path.name}'s report.json records session tag {recorded!r}, "
                           f"not {tag!r}")
    return Named(path=path)


def lgc_confirms(text: str, mhz: int) -> str:
    """"" unless `nvidia-smi -lgc`'s own output names a clock other than the
    one asked; an output that names none is recorded, not refused, because
    the cells are the evidence a lock took (an idle query after it is not:
    the H100 read 1830 MHz idle right after its 1800 lock)."""
    m = LGC_SET.search(text or "")
    if m and (int(m.group(1)), int(m.group(2))) != (mhz, mhz):
        return (f"-lgc {mhz},{mhz} set gpuClkMin {m.group(1)}, gpuClkMax {m.group(2)}: a "
                "clock other than the lock")
    return ""


def supported_graphics(text: str) -> list[int]:
    return sorted({int(m) for m in re.findall(r"Graphics\s*:\s*(\d+)\s*MHz", text or "")})


# --------------------------------------------------------------------------
# nvidia-smi, and the clock guard
# --------------------------------------------------------------------------

class Smi:
    """Every nvidia-smi call, logged. `sudo -n` in front of the ones that
    need root, only when not root (clock_elasticity's `ClockLock` rule: a root
    container gets the driver's own answer, not sudo's)."""

    def __init__(self, euid: int | None = None) -> None:
        self.prefix = [] if (os.geteuid() if euid is None else euid) == 0 else ["sudo", "-n"]
        self.log: list[dict] = []

    def call(self, argv, *, root: bool = False, timeout: float = 60) -> tuple[int, str]:
        cmd = [*(self.prefix if root else []), "nvidia-smi", *argv]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            rc, text = done.returncode, ((done.stdout or "") + (done.stderr or "")).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            rc, text = -1, f"{type(exc).__name__}: {exc}"
        self.log.append({"utc": utc(), "command": " ".join(cmd), "rc": rc,
                         "output": text[-400:]})
        return rc, text

    def command(self) -> str:
        return self.log[-1]["command"] if self.log else ""


def interrupt(child: subprocess.Popen | None) -> None:
    """SIGINT to the child's process group, not waited for. R3 ends on a
    KeyboardInterrupt, the way the 2026-09-25 runs' stopped attempts did."""
    if child is None or child.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, signal.SIGINT)


def stop(child: subprocess.Popen | None, grace_s: float) -> None:
    """SIGINT, `grace_s` to end, then SIGKILL to the whole group."""
    if child is None:
        return
    interrupt(child)
    try:
        child.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


class ClockGuard:
    """`nvidia-smi -lgc F,F` per lock and `nvidia-smi -rgc` at exit: the ONE
    place either is run here, `clock_elasticity.ClockLock`'s rules copied
    (the module docstring says why a copy).

    ENTERED, it turns SIGTERM and SIGHUP into SystemExit(128 + the signal) and
    SIGINT into KeyboardInterrupt, so its exit, which ALWAYS runs -rgc, runs
    on a kill as well as on a return or an exception. A signal already ignored
    when it is entered stays ignored. THE FIRST KILL WINS: the handler's first
    act ignores all three, so a second kill cannot interrupt the unwinding or
    the reset, and the -rgc child inherits the ignore. `child` is the running
    R3: sent SIGINT before the reset, waited for (then killed) after it, so
    the clock is released first and no R3 outlives the driver. `held` is the
    lock still on the card, None once a reset succeeded; `reset_record` says
    what the last reset did.

    A KILL DURING R3'S LAUNCH WAITS FOR THE CHILD TO BE HELD (2026-09-26).
    `child` was set only after Popen returned and the log file closed, so a
    kill in between raised with the Popen object unheld: the clock was reset,
    the driver exited, and R3 went on timing a whole page under the
    `<tag>-lock<F>` tag with no lock. Setting `child` on the very next line
    does not close it: CPython runs a signal handler as the Popen call
    returns, before the assignment. So inside `launching()` the handler only
    notes the first signal, and the block's exit, after `child` is set,
    raises it. Nothing is ignored meanwhile: an ignore set before the fork
    would be inherited by R3, and R3 is stopped with SIGINT.
    """

    SIGNALS = ("SIGTERM", "SIGHUP", "SIGINT")

    def __init__(self, smi: Smi, say, grace_s: float) -> None:
        self.smi = smi
        self.say = say
        self.grace_s = grace_s
        self.held: int | None = None
        self.child: subprocess.Popen | None = None
        self.reset_record: dict | None = None
        self.caught: str = ""
        self._saved: dict = {}
        self._quieted: dict = {}
        self._launching = False
        self._pending: int | None = None

    def lock(self, mhz: int) -> tuple[int, str]:
        rc, text = self.smi.call(["-lgc", f"{mhz},{mhz}"], root=True)
        if rc == 0:
            self.held = mhz
        return rc, text

    def _signals(self):
        return [s for s in (getattr(signal, n, None) for n in self.SIGNALS) if s]

    def __enter__(self) -> ClockGuard:
        for sig in self._signals():
            old = signal.getsignal(sig)
            if old in (None, signal.SIG_IGN):
                continue
            with contextlib.suppress(ValueError):   # not the main thread
                signal.signal(sig, self._on_signal)
                self._saved[sig] = old
        return self

    def _quiet(self) -> None:
        for sig in self._signals():
            if sig in self._quieted:
                continue
            old = signal.getsignal(sig)
            if old in (None, signal.SIG_IGN):
                continue
            with contextlib.suppress(ValueError):   # not the main thread
                signal.signal(sig, signal.SIG_IGN)
                self._quieted.setdefault(sig, old)

    @contextlib.contextmanager
    def launching(self):
        """The window from R3's launch to `child` being set: a kill caught in
        it is raised when the window closes (the class docstring says why).
        The first kill still wins; a crash in the window gives way to it."""
        self._launching = True
        try:
            yield
        finally:
            self._launching = False
            pending, self._pending = self._pending, None
            if pending is not None:
                self._raise(pending)

    def _on_signal(self, signum, _frame):
        if self._launching:
            if self._pending is None:
                self._pending = signum
            return
        self._raise(signum)

    def _raise(self, signum: int):
        self._quiet()
        self.caught = signal.Signals(signum).name
        # SAID BEFORE THE EXIT: a shell reports 143 for an uncaught SIGTERM too,
        # so only this line tells a reader the reset was reached.
        print(f"{self.caught} caught: stopping R3, resetting the SM clock, then exiting "
              f"{128 + signum}", file=sys.stderr, flush=True)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    def _reset(self) -> None:
        held = self.held
        rc, text = self.smi.call(["-rgc"], root=True)
        self.reset_record = {"command": self.smi.command(), "rc": rc, "output": text[-400:],
                             "held": held}
        if rc == 0:
            self.held = None
            self.say(f"SM clock reset: `{self.smi.command()}` exited 0"
                     + (f", the {held} MHz lock is released" if held is not None else ""))
            _, back = self.smi.call(list(READBACK_QUERY))
            self.say(f"read back after the reset: {_one(back)}")
        else:
            self.say(f"SM CLOCK NOT RESET: `{self.smi.command()}` exited {rc}: "
                     f"{_one(text) or 'no output'}"
                     + (f"; the card is STILL LOCKED at {held} MHz" if held is not None else "")
                     + ". Reset it by hand: sudo -n nvidia-smi -rgc")

    def __exit__(self, *exc) -> bool:
        # NOTHING MAY ABORT THE RESET, and nothing may SKIP it: the prologue
        # that ignores the three signals is a `try` whose `finally` is the
        # reset (ClockLock's structure).
        try:
            try:
                self._quiet()
                interrupt(self.child)
            finally:
                try:
                    self._reset()
                finally:
                    stop(self.child, self.grace_s)
                    self.child = None
        finally:
            for sig, old in {**self._quieted, **self._saved}.items():
                with contextlib.suppress(ValueError):   # not the main thread
                    signal.signal(sig, old)
            self._saved.clear()
            self._quieted.clear()
        return False


# --------------------------------------------------------------------------
# Gates: the shared table's shape (clock_elasticity's Gate)
# --------------------------------------------------------------------------

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
VALIDITY = "VALIDITY"
CLAIM = "CLAIM"


@dataclass
class Gate:
    kind: str
    number: str
    claim: str
    verdict: str
    measured: str
    threshold: str
    #: What a non-PASS here voids, printed only when it is not a PASS.
    invalidates: str = ""

    @property
    def token(self) -> str:
        return f"{self.kind[0]}{self.number}"

    def result_line(self) -> str:
        return exit_codes.result_line(
            exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
            self.token, self.verdict,
            _one(f"{self.claim} | measured {self.measured} | gate {self.threshold}"))

    def scored(self) -> tuple[str, str, str]:
        return (exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
                self.token, self.verdict)

    def render(self) -> list[str]:
        out = [self.result_line(),
               f"{self.kind} {self.number}  {self.verdict:8s} {self.claim}",
               f"{'':>11}measured {self.measured}",
               f"{'':>11}gate     {self.threshold}"]
        if self.verdict != PASS and self.invalidates:
            out.append(f"{'':>11}a non-PASS here voids {self.invalidates}")
        return out


# --------------------------------------------------------------------------
# One attempt, and the ladder
# --------------------------------------------------------------------------

@dataclass
class Attempt:
    lock: int
    G: int
    tag: str
    state: str = "running"
    rc: int | None = None
    cells: int = 0
    worst_mhz: float | None = None
    slip_mhz: float | None = None
    over_mhz: float | None = None
    dir: str | None = None
    why: str = ""
    utc_start: str = ""
    seconds: float = 0.0
    log: str = ""
    argv: list[str] = field(default_factory=list)
    #: rental 6, --slip-policy cell only: every row more than one step under the lock
    #: (`slipped_rows`), recorded in place of a stop; dropped from summary.json under page
    slipped_cells: list[dict] = field(default_factory=list)

    def r3_exit(self) -> str:
        """R3's exit through the shared table; a run this driver stopped ends
        on the signal it was sent, which Popen reports as minus its number."""
        if self.rc is None:
            return "-"
        if self.rc < 0:
            with contextlib.suppress(ValueError):
                return f"on {signal.Signals(-self.rc).name}"
            return f"on signal {-self.rc}"
        return f"{self.rc} {exit_codes.CODE_NAMES.get(self.rc, 'RETRY')}"

    def line(self) -> str:
        worst = "-" if self.worst_mhz is None else f"{self.worst_mhz:.1f}"
        name = Path(self.dir).name if self.dir else "(none named)"
        return (f"G={self.G} {self.state} at lock {self.lock}: R3 exit {self.r3_exit()}, "
                f"cells={self.cells} worst={worst} MHz, {self.seconds:.0f} s, dir {name}"
                + (f"; {len(self.slipped_cells)} slipped rows" if self.slipped_cells else "")
                + (f"; {self.why}" if self.why else ""))


@dataclass
class Outcome:
    """How the ladder ended: `held` the lock that held at every G, or
    `all_slipped`, or `stopped` saying why it stopped short, or `refused`
    when nothing was measured."""
    held: int | None = None
    all_slipped: bool = False
    stopped: str = ""
    refused: str = ""


class Run:
    """One invocation: the plan, and (unless --dry-run) the ladder."""

    def __init__(self, args, r3_args: list[str], results_dir: Path, out_dir: Path,
                 python: str, r3_script: Path) -> None:
        self.args = args
        self.base = args.session_tag
        self.locks = list(args.locks)
        self.groups = list(args.groups)
        self.r3_args = r3_args
        self.results_dir = results_dir
        self.pwr_dir = results_dir / PWR_DIR
        self.out_dir = out_dir
        self.python = python
        self.r3_script = r3_script
        self.attempts: list[Attempt] = []
        self.smi = Smi(euid=getattr(args, "euid", None))
        self.guard = ClockGuard(self.smi, self.say, args.stop_grace_s)
        self.t0 = time.monotonic()
        self.utc_start = ""
        self.card = ""

    # ---- the plan ---------------------------------------------------------

    def tag(self, lock: int) -> str:
        return f"{self.base}-lock{lock}"

    @property
    def cell_policy(self) -> bool:
        return getattr(self.args, "slip_policy", "page") == "cell"

    def slip_argv(self, lock: int) -> list[str]:
        """R3's slip arguments under --slip-policy cell (none under page: unchanged argv)."""
        if not self.cell_policy:
            return []
        return ["--slip-policy", "cell", "--lock-mhz", str(lock),
                "--min-clean-repeats", str(self.args.min_clean_repeats)]

    def r3_argv(self, lock: int, G: int) -> list[str]:
        return [self.python, str(self.r3_script), *self.r3_args, *self.slip_argv(lock),
                "--group-m", str(G), "--session-tag", self.tag(lock)]

    def plan_lines(self) -> list[str]:
        sudo = "sudo -n " if self.smi.prefix else ""
        lines = [
            "LOCKED R3: the timed R3 arm under an nvidia-smi SM clock lock, top lock first",
            f"session tag {self.base}; each lock's pages carry {self.base}-lock<F>",
            f"locks       {' '.join(map(str, self.locks))} MHz, top down; a slip (a timed "
            f"cell more than {STEP_MHZ} MHz under the lock) stops that lock, steps down one "
            f"and restarts at G={self.groups[0]}; a cell more than {STEP_MHZ} MHz over it "
            "means the lock is not in force and stops the ladder",
            f"groups      G = {' '.join(map(str, self.groups))}, in this order at every lock",
            f"R3          {shlex.join([self.python, str(self.r3_script), *self.r3_args])} "
            "--group-m G --session-tag <tag>-lock<F>",
            f"results     {self.pwr_dir}/<run id>: each attempt's directory is the one its own "
            "R3 names (the plan's `session` and `WRITES TO` lines) under its own tag, "
            "created after the attempt started; never found by time",
            f"lock        {sudo}nvidia-smi -lgc F,F, after `nvidia-smi -q -d "
            "SUPPORTED_CLOCKS` (every lock must be a supported graphics clock) and only "
            "once `nvidia-smi --query-compute-apps` lists no process on the card",
            f"reset       {sudo}nvidia-smi -rgc on every exit: a return, an exception, "
            "SIGTERM, SIGHUP, Ctrl-C. SIGKILL cannot be caught: after one, reset by hand "
            f"({sudo}nvidia-smi -rgc)",
            f"ledger      {self.out_dir}/status (one UTC line per event), summary.json, "
            "r3-g<G>-lock<F>.log per attempt"
            + (f", smi.csv every {self.args.sample_ms} ms" if self.args.sample_ms > 0 else ""),
            f"caps        one R3 run {self.args.run_cap_s:.0f} s, the whole ladder "
            f"{self.args.cap_s:.0f} s, a busy GPU {self.args.idle_cap_s:.0f} s; polled every "
            f"{self.args.poll_s:g} s",
            f"attempts    at most {len(self.locks) * len(self.groups)}, in this order until "
            "one lock holds at every G:",
        ]
        if self.cell_policy:
            lines.insert(-1, "slips       --slip-policy cell: a slipped repeat does NOT stop R3; "
                             "R3 flags it (lock_slip), keeps it out of the cell median, drops a "
                             f"cell under {self.args.min_clean_repeats} clean repeats and stops "
                             "once none can reach it; the page ends HELD_WITH_SLIPS")
        n = 0
        for lock in self.locks:
            for G in self.groups:
                n += 1
                lines.append(f"  {n:>3}  lock {lock}  G={G:<3} {shlex.join(self.r3_argv(lock, G))}")
        return lines

    # ---- the ledger -------------------------------------------------------

    def say(self, msg: str) -> None:
        line = f"{utc()} {msg}"
        print(line, flush=True)
        try:
            with (self.out_dir / "status").open("a") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            print(f"(status ledger not written: {exc})", file=sys.stderr, flush=True)

    def summary_payload(self, outcome: Outcome | None, gates: list[Gate], code: int | None,
                        stopped_by: str) -> dict:
        held = outcome.held if outcome else None
        pages = [self._dict(a) for a in self.attempts if a.lock == held and a.state in KEPT]
        return {"schema": 1, "utc_start": self.utc_start, "utc_end": utc(),
                "session_tag": self.base, "locks_asked": self.locks, "groups": self.groups,
                "r3_args": self.r3_args, "python": self.python,
                "results_dir": str(self.results_dir), "card": self.card,
                "attempts": [self._dict(a) for a in self.attempts], "lock": held,
                "pages": pages, "outcome": asdict(outcome) if outcome else None,
                "stopped_by": stopped_by, "gates": [asdict(g) for g in gates],
                "exit": code, "reset": self.guard.reset_record, "smi_log": self.smi.log}

    def _dict(self, a: Attempt) -> dict:
        """An attempt as summary.json holds it; under --slip-policy page without the
        rental-6 `slipped_cells` key, so a page-policy summary is the one written before."""
        d = asdict(a)
        if not self.cell_policy:
            d.pop("slipped_cells", None)
        return d

    def save(self, payload: dict) -> None:
        with contextlib.suppress(OSError):
            (self.out_dir / "summary.json").write_text(json.dumps(payload, indent=1) + "\n")

    def record(self, att: Attempt) -> None:
        """One attempt into the ledger and summary.json AS IT ENDS, so a
        SIGKILL, which no handler sees, still leaves every finished attempt
        on disk (`exit` null: the run had not ended)."""
        self.attempts.append(att)
        self.say(att.line())
        self.save(self.summary_payload(None, [], None, ""))

    # ---- the box ----------------------------------------------------------

    def over_cap(self) -> bool:
        return time.monotonic() - self.t0 > self.args.cap_s

    def card_busy(self) -> str:
        """"" once no compute process is on the card, waited for up to
        --idle-cap-s (a stopped R3 can hold the card for a few seconds after
        it exits), else what still held it."""
        t = time.monotonic()
        while True:
            rc, text = self.smi.call(["--query-compute-apps=pid", "--format=csv,noheader"])
            if rc == 0 and not text.strip():
                return ""
            if time.monotonic() - t > self.args.idle_cap_s:
                if rc != 0:
                    return (f"`{self.smi.command()}` exited {rc} for "
                            f"{self.args.idle_cap_s:.0f} s: {_one(text) or 'no output'}")
                return (f"a compute process (pid {', '.join(text.split())}) still held the "
                        f"card after {self.args.idle_cap_s:.0f} s")
            time.sleep(min(5.0, max(self.args.poll_s, 0.05)))

    def measured(self) -> bool:
        """True once any attempt timed a cell or R3 kept a page. Until then a
        stop is REFUSED (2), whatever stopped the ladder: nothing was measured,
        which is the table's REFUSED, not its INVALID ("measured; a VALIDITY
        gate failed after measuring")."""
        return any(a.cells or a.state in KEPT for a in self.attempts)

    def preflight(self) -> str:
        """"" when every lock is a supported graphics clock here, else why not.
        Records the card and, when persistence mode is off, turns it on (the
        2026-09-25 driver's `-pm 1`): without it the driver may unload between
        two R3 runs and take the lock with it."""
        rc, text = self.smi.call(list(CARD_QUERY))
        self.card = _one(text)
        self.say(f"card: {self.card or '(unread)'} (nvidia-smi exited {rc})")
        if rc == 0 and "Disabled" in text:
            prc, ptext = self.smi.call(["-pm", "1"], root=True)
            self.say(f"persistence mode was Disabled: `{self.smi.command()}` exited {prc}: "
                     f"{_one(ptext)}")
        rc, text = self.smi.call(["-q", "-d", "SUPPORTED_CLOCKS"])
        supported = supported_graphics(text)
        if rc != 0 or not supported:
            return (f"`nvidia-smi -q -d SUPPORTED_CLOCKS` exited {rc} and listed "
                    f"{len(supported)} graphics clocks: {_one(text)[-300:] or 'no output'}")
        self.say(f"supported graphics clocks: {len(supported)}, {supported[0]} to "
                 f"{supported[-1]} MHz")
        off = [f for f in self.locks if f not in supported]
        if off:
            near = {f: [s for s in supported if abs(s - f) <= 2 * STEP_MHZ] for f in off}
            return (f"--locks {off} are not supported graphics clocks here (nearest: {near}); "
                    "the driver would round the lock and the pages would name a clock that "
                    "never ran")
        return ""

    # ---- one attempt ------------------------------------------------------

    def attempt(self, lock: int, G: int) -> Attempt:
        """R3 at G under the lock, its cells scored as they land, stopped at
        the first slip. The run directory is the one `own_run_dir` takes from
        THIS attempt's log, never another."""
        tag = self.tag(lock)
        existed = {p.name for p in self.pwr_dir.iterdir()} if self.pwr_dir.is_dir() else set()
        log_path = self.out_dir / f"r3-g{G}-lock{lock}.log"
        att = Attempt(lock=lock, G=G, tag=tag, utc_start=utc(), log=str(log_path),
                      argv=self.r3_argv(lock, G))
        env = {**os.environ, "MOE_RESULTS_DIR": str(self.results_dir),
               "PYTHONUNBUFFERED": "1"}
        t = time.monotonic()
        # HELD BEFORE ANY KILL CAN UNWIND (2026-09-26): a signal caught inside
        # `launching()` is raised only after `child` is the guard's, so the
        # guard's exit always stops this R3 (ClockGuard's docstring).
        with self.guard.launching(), log_path.open("w") as log:
            child = subprocess.Popen(att.argv, cwd=str(ROOT), stdout=log,
                                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
            self.guard.child = child
        self.say(f"G={G} start at lock {lock}: pid {child.pid}, tag {tag}, log {log_path.name}")
        try:
            while True:
                exited = child.poll() is not None
                named = own_run_dir(tag, log_path.read_text(errors="replace"), self.pwr_dir,
                                    existed)
                if named.error:
                    att.state, att.why = NOT_OWN_DIR, named.error
                    break
                if named.path is not None:
                    if att.dir is None:
                        att.dir = str(named.path)
                        self.say(f"G={G} at lock {lock}: R3 named its run directory "
                                 f"{named.path.name}")
                    clocks = cell_clocks(named.path / "cells.csv")
                    att.cells = len(clocks)
                    att.worst_mhz = min(clocks) if clocks else None
                    att.slip_mhz, att.over_mhz = off_lock(clocks, lock)
                    if self.cell_policy and att.slip_mhz is not None:
                        # rental 6: recorded, not stopped; R3 excludes the repeat itself
                        rows = slipped_rows(named.path / "cells.csv", lock)
                        for r in rows[len(att.slipped_cells):]:
                            self.say(f"G={G} at lock {lock}: slipped row {r['row']} "
                                     f"{r['label'] or '-'} {r['arm']} n={r['tiles']} rep "
                                     f"{r['repeat']} at {r['mhz']:.0f} MHz (recorded, R3 runs on)")
                        att.slipped_cells = rows
                        att.slip_mhz = None
                    if att.over_mhz is not None:
                        att.state = OFF_LOCK_HIGH
                        att.why = (f"a timed cell read {att.over_mhz:.0f} MHz, more than "
                                   f"{STEP_MHZ} MHz over the {lock} MHz lock: the lock is not "
                                   "in force")
                        break
                    if att.slip_mhz is not None:
                        att.state = SLIPPED
                        att.why = (f"a timed cell read {att.slip_mhz:.0f} MHz, more than "
                                   f"{STEP_MHZ} MHz under the {lock} MHz lock")
                        break
                if exited:
                    break
                if time.monotonic() - t > self.args.run_cap_s:
                    att.state, att.why = TIMEOUT, f"past --run-cap-s {self.args.run_cap_s:.0f} s"
                    break
                if self.over_cap():
                    att.state, att.why = LADDER_CAP, f"past --cap-s {self.args.cap_s:.0f} s"
                    break
                time.sleep(self.args.poll_s)
        except BaseException:
            # A signal or a crash: SIGINT to R3 now, and the guard's exit waits
            # for it AFTER the reset, so the clock is released first.
            interrupt(child)
            raise
        stop(child, self.args.stop_grace_s)
        self.guard.child = None
        att.rc = child.returncode
        att.seconds = round(time.monotonic() - t, 1)
        if att.state == "running":
            att.state, att.why = self._ended(att, log_path)
        return att

    def _ended(self, att: Attempt, log_path: Path) -> tuple[str, str]:
        """R3 exited on its own with no cell off the lock: HELD when it wrote
        its page, under this attempt's tag, with a measured exit."""
        if att.dir is None:
            tail = next((ln for ln in reversed(log_path.read_text(errors="replace")
                                                .splitlines()) if ln.strip()), "an empty log")
            return NO_PAGE, (f"R3 exited {att.rc} before naming its run directory; its log "
                             f"ends: {_one(tail)[:200]}")
        recorded = recorded_tag(Path(att.dir))
        if recorded is None:
            return NO_PAGE, f"R3 exited {att.rc} and wrote no report.json"
        if recorded != att.tag:
            return NOT_OWN_DIR, f"report.json records session tag {recorded!r}, not {att.tag!r}"
        if att.rc not in exit_codes.MEASURED_CODES:
            return NO_PAGE, f"R3 exited {exit_codes.describe(att.rc)}"
        if att.slipped_cells:
            low = min(r["mhz"] for r in att.slipped_cells)
            return HELD_WITH_SLIPS, (f"{len(att.slipped_cells)} slipped rows, lowest {low:.0f} "
                                     "MHz, recorded and kept out of their cell medians by R3")
        return HELD, ""

    # ---- the ladder -------------------------------------------------------

    def ladder(self) -> Outcome:
        refused = self.preflight()
        if refused:
            return Outcome(refused=refused)
        for i, lock in enumerate(self.locks):
            # THE CARD IS IDLE BEFORE THE LOCK, NOT UNDER IT (2026-09-26). The
            # first version locked, then waited for a busy card inside the G
            # loop, so a process still on the card (a chain arm still exiting,
            # as the GH200's was when its locked R3 began) ran for up to
            # --idle-cap-s under a lock nobody set for it, and nothing notices
            # a lock (docs/LAMBDA.md section 3b). The 2026-09-25 driver checked
            # first; so does this one, before every lock, step-downs included.
            bad, text = self.card_busy(), ""
            if not bad:
                rc, text = self.guard.lock(lock)
                bad = (f"`{self.smi.command()}` exited {rc}: {_one(text) or 'no output'}"
                       if rc else lgc_confirms(text, lock))
            if bad:
                self.say(f"lock {lock} MHz not taken: {bad}")
                if not self.measured():
                    return Outcome(refused=bad)
                return Outcome(stopped=f"lock {lock} MHz not taken: {bad}")
            self.say(f"locked at {lock} MHz: {_one(text)}")
            _, back = self.smi.call(list(READBACK_QUERY))
            self.say(f"read back after the lock (an idle reading, not the lock's evidence): "
                     f"{_one(back)}")
            for j, G in enumerate(self.groups):
                if self.over_cap():
                    return Outcome(stopped=f"past --cap-s {self.args.cap_s:.0f} s before G={G} "
                                           f"at lock {lock}")
                # Between two G the lock stays on: the last R3 may still hold
                # the card for a moment. The first G was checked before -lgc.
                busy = self.card_busy() if j else ""
                if busy:
                    att = Attempt(lock=lock, G=G, tag=self.tag(lock), state=GPU_BUSY,
                                  utc_start=utc(), why=busy)
                    self.record(att)
                    return Outcome(stopped=att.line())
                att = self.attempt(lock, G)
                self.record(att)
                if att.state in KEPT:
                    continue
                if att.state == SLIPPED:
                    nxt = self.locks[i + 1] if i + 1 < len(self.locks) else None
                    self.say(f"lock {lock} MHz did not hold at G={G}; "
                             + (f"stepping down to {nxt} MHz and restarting at "
                                f"G={self.groups[0]}" if nxt else "no lower lock on the ladder"))
                    break
                # R3 REFUSING BEFORE ANYTHING WAS MEASURED IS REFUSED (2026-09-26):
                # a $PY_VLLM without the GPU stack, device_guard, a plan that
                # does not fit. Exiting INVALID said "measured" of zero cells.
                # The same refusal after a page is INVALID: the ladder was
                # measuring and stopped short.
                if (att.state == NO_PAGE and att.rc == exit_codes.REFUSED
                        and not self.measured()):
                    return Outcome(refused=f"R3 refused before its first timed cell: "
                                           f"{att.line()}")
                return Outcome(stopped=att.line())
            else:
                self.say(f"all G held at lock {lock} MHz")
                return Outcome(held=lock)
        return Outcome(all_slipped=True)

    # ---- the verdict ------------------------------------------------------

    def gates(self, outcome: Outcome | None, stopped_by: str) -> list[Gate]:
        foreign = [a for a in self.attempts if a.state == NOT_OWN_DIR]
        v1 = Gate(VALIDITY, "1", "every attempt read only the run directory its own R3 named",
                  FAIL if foreign else PASS,
                  f"{len(self.attempts) - len(foreign)} of {len(self.attempts)} attempts"
                  + (": " + "; ".join(a.line() for a in foreign) if foreign else ""),
                  "the plan's session line is the attempt's tag, the directory is new since "
                  "the attempt started, and its report.json records the same tag",
                  "the lock search: an attempt scored another run's cells, or R3 would have "
                  "resumed one")
        if stopped_by:
            why = f"stopped by {stopped_by}"
        elif outcome is None:
            why = "no outcome"
        else:
            why = outcome.stopped
        reached = outcome is not None and not stopped_by and (
            outcome.held is not None or outcome.all_slipped)
        v2 = Gate(VALIDITY, "2", "the lock search reached a verdict", PASS if reached else FAIL,
                  (f"lock {outcome.held} MHz held at every G" if reached and outcome.held
                   else "every lock slipped" if reached else why),
                  "a lock held at every G, or every lock on the ladder slipped",
                  "the verdict: the ladder stopped for a reason a lower lock would not fix")
        if reached:
            held = outcome.held is not None
            slips = sum(len(a.slipped_cells) for a in self.attempts if a.lock == outcome.held)
            c1 = Gate(CLAIM, "1", "a lock on the ladder held at every G",
                      PASS if held else FAIL,
                      (f"{outcome.held} MHz held at G = {' '.join(map(str, self.groups))}"
                       + (f", {slips} slipped rows recorded (--slip-policy cell)" if slips else "")
                       if held else f"every lock slipped: {' '.join(map(str, self.locks))} MHz"),
                      f"no timed cell more than {STEP_MHZ} MHz under the lock at any G"
                      + (" (under --slip-policy cell a slipped repeat is R3's to exclude, "
                         "not a stop)" if self.cell_policy else ""))
        else:
            c1 = Gate(CLAIM, "1", "a lock on the ladder held at every G", UNKNOWN,
                      "not established: " + why,
                      f"no timed cell more than {STEP_MHZ} MHz under the lock at any G")
        return [v1, v2, c1]

    def summary_lines(self, outcome: Outcome | None, gates: list[Gate], code: int,
                      stopped_by: str) -> list[str]:
        if stopped_by:
            head = f"stopped by {stopped_by} before a verdict"
        elif outcome and outcome.refused:
            head = f"REFUSED, nothing measured: {outcome.refused}"
        elif outcome and outcome.held is not None:
            head = (f"{outcome.held} MHz held at every G "
                    f"({' '.join(map(str, self.groups))})")
        elif outcome and outcome.all_slipped:
            head = "no lock on the ladder held: every one slipped"
        else:
            head = "stopped before a verdict: " + (outcome.stopped if outcome else "no outcome")
        out = ["=" * 78, f"LOCKED R3, session tag {self.base}: {head}",
               f"  {'lock':>5} {'G':>3}  {'state':<18} {'cells':>5} {'worst MHz':>9}  "
               f"{'R3 exit':<13} {'s':>6}  run directory"]
        for a in self.attempts:
            worst = "-" if a.worst_mhz is None else f"{a.worst_mhz:.1f}"
            name = Path(a.dir).name if a.dir else "-"
            out.append(f"  {a.lock:>5} {a.G:>3}  {a.state:<18} {a.cells:>5} {worst:>9}  "
                       f"{a.r3_exit():<13} {a.seconds:>6.0f}  {name}")
        held = outcome.held if outcome else None
        if held is not None:
            out.append(f"pages kept, all at {held} MHz (each verdict is R3's own):")
            out += [f"  G={a.G:<3} {a.r3_exit():<13} {a.dir}"
                    + (f" ({len(a.slipped_cells)} slipped rows)" if a.slipped_cells else "")
                    for a in self.attempts if a.lock == held and a.state in KEPT]
        for g in gates:
            out += g.render()
        rec = self.guard.reset_record or {}
        out.append(f"SM clock reset: `{rec.get('command', 'nvidia-smi -rgc')}` exited "
                   f"{rec.get('rc', 'never run')}"
                   + (f"; the card is STILL LOCKED at {self.guard.held} MHz, reset it by "
                      "hand: sudo -n nvidia-smi -rgc" if self.guard.held is not None else ""))
        if code > 100:
            out.append(f"exit {code}: stopped by {stopped_by}, after the reset (128 + the "
                       "signal, outside the exit-code table)")
        else:
            out.append(f"exit {exit_codes.describe(code)}")
        out += [f"ledger {self.out_dir / 'status'}; summary {self.out_dir / 'summary.json'}",
                "=" * 78]
        return out

    def finish(self, outcome: Outcome | None, exc: BaseException | None) -> int:
        """The summary, on every path: the gates, the exit, the ledger's last
        lines and summary.json. Runs after the guard's exit, so the reset has
        already happened and is reported, not attempted, here."""
        stopped_by = ""
        if isinstance(exc, KeyboardInterrupt):
            stopped_by, code = "SIGINT", 128 + signal.SIGINT
        elif isinstance(exc, SystemExit):
            stopped_by = self.guard.caught or f"SystemExit({exc.code})"
            code = exc.code if isinstance(exc.code, int) else exit_codes.ERROR
        elif exc is not None:
            stopped_by, code = f"a crash ({type(exc).__name__}: {exc})", exit_codes.ERROR
        else:
            code = exit_codes.DONE
        # A CARD LEFT LOCKED IS ERROR whatever the gates say (clock_elasticity's
        # rule): whatever runs next on it would measure under a lock nobody set
        # for it. A failed -rgc with no lock ever taken leaves nothing behind.
        left_locked = self.guard.held is not None
        if outcome is not None and outcome.refused and not stopped_by:
            gates: list[Gate] = []
            code = exit_codes.ERROR if left_locked else exit_codes.REFUSED
            self.say(f"REFUSED, nothing measured: {outcome.refused}")
        else:
            gates = self.gates(outcome, stopped_by)
            if not stopped_by:
                code = exit_codes.classify(g.scored() for g in gates)
                if left_locked:
                    code = exit_codes.ERROR
        for line in self.summary_lines(outcome, gates, code, stopped_by):
            print(line, flush=True)
        self.save(self.summary_payload(outcome, gates, code, stopped_by))
        self.say(f"locked_r3 end: exit {code}, " + (f"stopped by {stopped_by}" if stopped_by
                                                     else exit_codes.describe(code)))
        return code

    def execute(self) -> int:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.utc_start = utc()
        self.t0 = time.monotonic()
        self.say(f"locked_r3 start: tag {self.base}, locks {self.locks}, G {self.groups}, "
                 f"R3 args {shlex.join(self.r3_args)}, results {self.pwr_dir}")
        outcome: Outcome | None = None
        sampler = None
        try:
            with self.guard, contextlib.ExitStack() as stack:
                if self.args.sample_ms > 0:
                    sink = stack.enter_context((self.out_dir / "smi.csv").open("w"))
                    with contextlib.suppress(OSError):
                        sampler = subprocess.Popen(
                            ["nvidia-smi", SAMPLER_QUERY, "--format=csv", "-lms",
                             str(self.args.sample_ms)], stdout=sink, stderr=subprocess.STDOUT)
                try:
                    outcome = self.ladder()
                finally:
                    if sampler is not None and sampler.poll() is None:
                        sampler.terminate()
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            sampler.wait(timeout=10)
        except BaseException as exc:
            self.finish(outcome, exc)
            raise
        return self.finish(outcome, None)


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        usage="%(prog)s --session-tag TAG --locks F1 [F2 ...] [options] [-- R3 ARGS]")
    ap.add_argument("--session-tag", required=True,
                    help="the base tag; lock F's pages carry <tag>-lock<F> (the chain "
                         "session's name, as on 2026-09-25)")
    ap.add_argument("--locks", type=int, nargs="+", required=True, metavar="F",
                    help="SM clock locks in MHz, top down (or one lock); each a supported "
                         "graphics clock")
    ap.add_argument("--groups", type=int, nargs="+", default=list(DEFAULT_GROUPS),
                    metavar="G", help="GROUP_SIZE_M values, run in this order at every lock "
                                      "(default: 1 2 4 16 64)")
    ap.add_argument("--results-dir", default=None,
                    help="R3's results root, handed to it as MOE_RESULTS_DIR (default "
                         "$MOE_RESULTS_DIR, else $RESULTS_ROOT/gaps-$MOE_CARD)")
    ap.add_argument("--out-dir", default=None,
                    help="the driver's ledger, logs and summary (default "
                         "$SESSION_ROOT/locked-r3/<tag>)")
    ap.add_argument("--python", default=None,
                    help="R3's interpreter (default $PY_VLLM)")
    ap.add_argument("--r3-script", default=str(R3_SCRIPT), help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and REFUSE: no nvidia-smi, no R3, nothing written")
    ap.add_argument("--poll-s", type=float, default=10.0,
                    help="seconds between reads of the running attempt's cells")
    ap.add_argument("--run-cap-s", type=float, default=1800.0,
                    help="one R3 run's cap; past it the run is stopped and the ladder with it")
    ap.add_argument("--cap-s", type=float, default=4 * 3600.0,
                    help="the whole ladder's cap")
    ap.add_argument("--idle-cap-s", type=float, default=300.0,
                    help="how long a busy GPU is waited for before each lock and between "
                         "two G")
    ap.add_argument("--stop-grace-s", type=float, default=90.0,
                    help="how long a stopped R3 gets to end on SIGINT before SIGKILL")
    ap.add_argument("--slip-policy", choices=("page", "cell"), default="page",
                    help="page (default): the first slipped cell stops the page and the lock "
                         "steps down; cell (rental 6, histogram pages): R3 gets --lock-mhz, "
                         "flags and excludes each slipped repeat and runs on, and the page "
                         "ends HELD_WITH_SLIPS")
    ap.add_argument("--min-clean-repeats", type=int, default=6,
                    help="with --slip-policy cell: R3 drops a cell that cannot reach this many "
                         "clean repeats and stops the page once none can (default 6, of 9)")
    ap.add_argument("--sample-ms", type=int, default=500,
                    help="nvidia-smi's sampling period into smi.csv; 0 for none")
    return ap


def owned_flag(token: str) -> str | None:
    """The owned R3 flag `token` names or abbreviates, else None.

    ANY PREFIX, HOWEVER SHORT (2026-09-26). R3's parser has allow_abbrev, so
    it reads a token as the one option it begins: `--o /x` is `--out /x`,
    because `--out` is R3's only option starting `--o`. The first version
    matched only names longer than three characters, so `--o /x` went through,
    the card was locked, and R3 wrote outside the results root (INVALID after
    the lock instead of REFUSED before it). A prefix that is also the start
    of another R3 option is ambiguous, which R3 rejects on its own, so
    refusing it here costs nothing; and no option R3 has outside this table
    is itself a prefix of an owned flag, so no legitimate pass-through is
    refused. tests/test_locked_r3.py holds both to R3's own parser."""
    name = token.split("=", 1)[0]
    if not name.startswith("--") or name == "--":
        return None
    return next((f for f in OWNED_FLAGS if f.startswith(name)), None)


def _results_dir(args) -> Path:
    if args.results_dir:
        return Path(args.results_dir)
    if os.environ.get("MOE_RESULTS_DIR"):
        return Path(os.environ["MOE_RESULTS_DIR"])
    if os.environ.get("RESULTS_ROOT") and os.environ.get("MOE_CARD"):
        return Path(os.environ["RESULTS_ROOT"]) / f"gaps-{os.environ['MOE_CARD']}"
    raise Refusal("no results root: pass --results-dir, or export MOE_RESULTS_DIR="
                  "$RESULTS_ROOT/gaps-$MOE_CARD (docs/LAMBDA.md section 3b), so the pages "
                  "land beside the chain's and the driver knows where to find them")


def prepare(argv: list[str]) -> Run:
    """Parse and check everything that needs no card and writes nothing."""
    if "--" in argv:
        i = argv.index("--")
        mine, r3_args = argv[:i], list(argv[i + 1:])
    else:
        mine, r3_args = argv, list(DEFAULT_R3_ARGS)
    args = build_parser().parse_args(mine)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.session_tag):
        raise Refusal(f"--session-tag {args.session_tag!r}: letters, digits, '.', '_' and '-' "
                      "only (it names a directory)")
    if any(f <= 0 for f in args.locks) or any(
            a <= b for a, b in zip(args.locks, args.locks[1:], strict=False)):
        raise Refusal(f"--locks {args.locks} must be positive and strictly descending: a slip "
                      "steps DOWN the ladder")
    if any(g <= 0 for g in args.groups) or len(set(args.groups)) != len(args.groups):
        raise Refusal(f"--groups {args.groups} must be distinct positive integers")
    if args.slip_policy == "cell":
        if "--histogram" not in r3_args:
            raise Refusal("--slip-policy cell runs a histogram page only (R3 refuses --lock-mhz "
                          "on its balanced ladder)")
        if args.min_clean_repeats < 0:
            raise Refusal(f"--min-clean-repeats {args.min_clean_repeats}: 0 or more")
    owned = [(t, owned_flag(t)) for t in r3_args if owned_flag(t)]
    if owned:
        raise Refusal("R3 arguments this driver owns: " + "; ".join(
            f"{t} ({OWNED_FLAGS[f]})" for t, f in owned))
    for knob in ("poll_s", "run_cap_s", "cap_s", "idle_cap_s", "stop_grace_s"):
        if getattr(args, knob) <= 0:
            raise Refusal(f"--{knob.replace('_', '-')} must be positive")
    results_dir = _results_dir(args).expanduser().resolve()
    python = args.python or os.environ.get("PY_VLLM", "")
    if not python:
        raise Refusal("no interpreter for R3: pass --python, or source ~/moe/env.sh "
                      "($PY_VLLM, the vLLM venv)")
    r3_script = Path(args.r3_script)
    if not r3_script.is_file():
        raise Refusal(f"no R3 script at {r3_script}")
    session = Path(os.environ.get("SESSION_ROOT") or Path.home() / "moe" / "session")
    out_dir = (Path(args.out_dir) if args.out_dir
               else session / "locked-r3" / args.session_tag).expanduser().resolve()
    used = sorted(p.name for p in out_dir.glob("r3-*.log")) if out_dir.is_dir() else []
    if used:
        raise Refusal(f"{out_dir} already holds attempts under this tag ({', '.join(used[:3])}"
                      f"{', ...' if len(used) > 3 else ''}): their run directories would be "
                      "resumed. Use a fresh --session-tag")
    run = Run(args, r3_args, results_dir, out_dir, python, r3_script)
    planned = {run.tag(f) for f in run.locks}
    on_disk = {t: d for t, d in tags_on_disk(run.pwr_dir).items() if t in planned}
    if on_disk:
        raise Refusal(f"finished R3 runs under {run.pwr_dir} already record this ladder's "
                      f"tags: {on_disk}. Use a fresh --session-tag")
    return run


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        run = prepare(argv)
    except Refusal as exc:
        print(f"REFUSED. Nothing was measured: {exc}")
        return exit_codes.REFUSED
    if run.args.dry_run:
        print("\n".join(run.plan_lines()))
        print("\n".join(["", "=" * 78,
                         "REFUSED. Nothing was measured and nothing was written.",
                         "  reason: --dry-run was given. No nvidia-smi was called, no lock "
                         "was set, no R3 ran,",
                         "  and no gate was scored, so no RESULT line was printed.",
                         "=" * 78]))
        return exit_codes.REFUSED
    print("\n".join(run.plan_lines()), flush=True)
    return run.execute()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        raise SystemExit(128 + signal.SIGINT) from None
    except BaseException as exc:                                      # noqa: BLE001
        traceback.print_exc()
        print(f"locked_r3 crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(exit_codes.ERROR) from exc
