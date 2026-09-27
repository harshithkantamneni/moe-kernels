"""The unattended GH200 run, checked without a VM: the driver
(scripts/gh200_model_session.sh), the VM's results push
(scripts/vm_results_push.sh) and the laptop's side (scripts/vm_run.sh).

docs/LAMBDA.md section 3c was drafted on 2026-09-26 as blocks a person pastes,
each with a "Stops" paragraph. The driver runs those blocks unattended, so
every Stop became a rule, and these tests hold the rules. The driver runs for
real here, on a FAKE BOX: stub binaries on PATH (`nvidia-smi` keeping the lock
state in a file, `sudo`, `ncu`, `timeout`, `sleep`) and a stub interpreter that
plays each tool (`dram_counter_route.py`, `private_weight_reference.py`,
`locked_r3.py`, `calibrate_hardware.py`, `clock_elasticity.py`) from a scenario
file, logging every argv. So the tests read what the driver actually ran, in
order, and every command line is held to the tool's own argparse parser: a
flag the tool does not have fails here, on the laptop, not on a rented card.
One scenario hands R3's and R1's dry runs to the real scripts, so the checks
the driver reads off a dry run are checked against the real plan text.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from moe.bench import exit_codes  # noqa: E402

DRIVER = REPO / "scripts" / "gh200_model_session.sh"
PUSH = REPO / "scripts" / "vm_results_push.sh"
VMRUN = REPO / "scripts" / "vm_run.sh"
PY = sys.executable
CARD = "nvidia_gh200_480gb"
FAR = int(time.time()) + 10 * 86400
B_RE = r"gh200-\d{8}T\d{6}Z"

# --------------------------------------------------------------------------
# the fake box
# --------------------------------------------------------------------------

FAKEPY = r'''#!__PY__ -S
import json, os, re, sys, time
from pathlib import Path
REAL = "__PY__"
argv = sys.argv[1:]
if not argv or argv[0] in ("-", "-c", "-m"):
    os.execv(REAL, [REAL] + argv)
m = re.search(r"scripts/(\w+)\.py$", argv[0])
if not m:
    os.execv(REAL, [REAL] + argv)
tool, args = m.group(1), argv[1:]
SC = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"exe": "python", "tool": tool, "argv": args,
                        "dry": "--dry-run" in args}) + "\n")

def val(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default

def vals(flag):
    if flag not in args:
        return []
    out = []
    for a in args[args.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out

def real(script):
    os.execv(REAL, [REAL, os.path.join(os.environ["REAL_REPO"], "scripts", script)] + args)

if tool == "dram_counter_route":
    if "--analyse" in args:
        print("analysed", len(vals("--analyse")), "pages")
        sys.exit(0)
    G, out = val("--group-m"), val("--out")
    if "--floor" in args:
        key = "floor:" + Path(out).stem
    elif val("--page-lock-mhz"):
        key = f"bytes:{G}"
    else:
        key = f"base:{G}"
    spec = SC.get(key, {})
    fail = spec.get("fail", [])
    for g in ["V0", "V1", *fail]:
        print(f"RESULT: VALIDITY {g} {'FAIL' if g in fail else 'PASS'} the gate")
    if spec.get("passes"):
        print(f'==PROF== Profiling "fused_moe_kernel": 0%....50%....100% - {spec["passes"]} passes')
    if not spec.get("nopage"):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text("{}")
    sys.exit(spec.get("rc", 0))

if tool == "private_weight_reference":
    if "--dry-run" not in args:
        sys.exit(99)   # the driver never runs R3 itself: locked_r3.py does
    if os.environ.get("FAKE_REAL_DRY"):
        real("private_weight_reference.py")
    T, G = int(val("--treads")), val("--group-m")
    if f"{G}:{T}" in SC.get("dry_refuse", []):
        print("REFUSED: this design cannot separate the worlds it registers.")
    print(f"            n_decl = 9 against n_max = {T} read: planted")
    n = SC.get("retracted", 11)
    print(f"            alpha=0.100 retracted  tread   {n}   against the {T} planned")
    if "--ridge" in args:
        print("  ridge       177.93 Op/B, given on the command line")
    elif SC.get("hypothesis"):
        print("  ridge       160.30 Op/B, HYPOTHESIS: the withdrawn H200 band")
    else:
        print("  ridge       180.00 Op/B, measured_nvidia_gh200_480gb.yaml")
    sys.exit(2)

if tool == "locked_r3":
    tag, locks, groups = val("--session-tag"), vals("--locks"), vals("--groups")
    base = Path(os.environ["SESSION_ROOT"], "base-tag.txt").read_text().strip()
    suffix = tag[len(base) + 1:]
    d = Path(os.environ["SESSION_ROOT"], "locked-r3", tag)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "status", "a") as f:
        f.write("2026-09-27T00:00:00Z card: NVIDIA GH200 480GB "
                "GPU-deadbeef-1111-2222-3333-444455556666\n")
        for g in groups:
            f.write(f"2026-09-27T00:00:01Z G={g} held at lock {locks[0]}: R3 exit 0 DONE, "
                    f"cells=54 worst={locks[0]}.0 MHz, 600 s, dir run-{tag}-g{g}\n")
    if SC.get("hang") == suffix:
        Path(os.environ["FAKE_LOG"]).with_suffix(".hanging").write_text(str(os.getpid()))
        time.sleep(120)
    sys.exit(SC.get("locked", {}).get(suffix, 0))

if tool == "calibrate_hardware":
    c = SC.get("calibrate", {})
    for line in c.get("results", ["RESULT: CLAIM not_throttled PASS from gemm_clock",
                                  "RESULT: VALIDITY clock_established PASS agree"]):
        print(line)
    if "--publish" in args and c.get("write", True):
        y = Path.cwd() / "moe" / "bench" / "hardware" / f"measured_{os.environ['MOE_CARD']}.yaml"
        y.parent.mkdir(parents=True, exist_ok=True)
        y.write_text("ridge: 180.0\n")
    sys.exit(c.get("rc", 0))

if tool == "clock_elasticity":
    if "--dry-run" in args:
        if os.environ.get("FAKE_REAL_DRY"):
            real("clock_elasticity.py")
        print("the plan")
        sys.exit(2)
    sys.exit(SC.get("r1", {}).get(val("--group-m"), 1))
sys.exit(0)
'''

SMI = r'''#!__PY__ -S
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
SC = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
st_path = Path(os.environ["FAKE_STATE"])
st = (json.loads(st_path.read_text()) if st_path.exists()
      else {"locked": None, "pm": "Disabled", "pl": 900.0})
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"exe": "nvidia-smi", "argv": args}) + "\n")
def save():
    st_path.write_text(json.dumps(st))
joined = " ".join(args)
if "--query-compute-apps" in joined:
    if os.environ.get("FAKE_BUSY") or SC.get("busy"):
        print("4242, python")
    sys.exit(0)
if args[:3] == ["-q", "-d", "SUPPORTED_CLOCKS"]:
    print("Attached GPUs : 1\n    Supported Clocks\n        Memory : 2619 MHz")
    for c in SC.get("clocks", list(range(1980, 344, -15))):
        print(f"            Graphics                        : {c} MHz")
    sys.exit(0)
if args[:2] == ["-q", "-d"]:
    print("==== fake nvidia-smi -q", " ".join(args[2:]))
    sys.exit(0)
if "-pm" in args:
    st["pm"] = "Enabled"; save(); sys.exit(0)
if "-pl" in args:
    if SC.get("pl_fail"):
        sys.exit(1)
    st["pl"] = float(SC.get("pl_reads", args[args.index("-pl") + 1])); save(); sys.exit(0)
if "-lgc" in args:
    st["locked"] = args[args.index("-lgc") + 1]; save(); print("GPU clocks set"); sys.exit(0)
if "-rgc" in args:
    if SC.get("rgc_fail"):
        sys.exit(1)
    st["locked"] = None; save(); print("All done."); sys.exit(0)
q = next((a for a in args if a.startswith("--query-gpu=")), None)
if q:
    fields = q.split("=", 1)[1].split(",")
    units = "nounits" not in joined
    vals = []
    for fld in fields:
        if fld == "persistence_mode":
            vals.append(st["pm"])
        elif fld in ("power.limit", "power.default_limit"):
            w = st["pl"] if fld == "power.limit" else 900.0
            vals.append(f"{w:.2f}" + (" W" if units else ""))
        elif fld == "clocks.max.sm":
            vals.append("1980" + (" MHz" if units else ""))
        elif fld == "clocks.sm":
            vals.append(str(st["locked"] or "345").split(",")[0] + (" MHz" if units else ""))
        else:
            vals.append("0")
    if "noheader" not in joined:
        print(", ".join(fields))
    print(", ".join(vals))
    sys.exit(0)
sys.exit(0)
'''

SUDO = '#!/bin/bash\n[ "$1" = -n ] && shift\nexec "$@"\n'
NCU = ('#!/bin/bash\nprintf \'{"exe": "ncu", "argv": "%s"}\\n\' "$*" >> "$FAKE_LOG"\n'
       '[ "$FAKE_NCU_FAIL" = 1 ] && exit 1\nexit 0\n')
#: A timeout(1) that only runs the command: the fake box has no clock to wait on.
TIMEOUT = ('#!/bin/bash\nwhile [ $# -gt 0 ]; do case "$1" in -k|-s) shift 2 ;; -*) shift ;; '
           '*) break ;; esac; done\nshift\nexec "$@"\n')


def _exe(path: Path, body: str) -> Path:
    path.write_text(body.replace("__PY__", PY))
    path.chmod(0o755)
    return path


class Box:
    """A VM as the driver sees it: $MOE_HOME with env.sh, a session holding
    the preflight's census, and a clean checkout with the published results
    linked in (the timed references the byte analysis reads)."""

    def __init__(self, root: Path, scenario: dict, real_timeout: bool = False):
        self.root = root
        self.home = root / "home"
        self.moe = self.home / "moe"
        self.repo = self.moe / "repo"
        self.session = self.moe / "session"
        self.results = self.moe / "results"
        self.bin = root / "bin"
        self.log = root / "calls.jsonl"
        self.state = root / "smi-state.json"
        self.scenario_path = root / "scenario.json"
        for d in (self.repo, self.session, self.results, self.bin):
            d.mkdir(parents=True, exist_ok=True)
        self.scenario_path.write_text(json.dumps(scenario))
        _exe(self.bin / "fakepy", FAKEPY)
        os.symlink(self.bin / "fakepy", self.bin / "python3")
        _exe(self.bin / "nvidia-smi", SMI)
        _exe(self.bin / "sudo", SUDO)
        _exe(self.bin / "ncu", NCU)
        _exe(self.bin / "sleep", "#!/bin/bash\nexit 0\n")
        if real_timeout:
            real = shutil.which("timeout") or shutil.which("gtimeout")
            os.symlink(real, self.bin / "timeout")
        else:
            _exe(self.bin / "timeout", TIMEOUT)
        (self.session / "census.json").write_text("{}")
        git = ["git", "-C", str(self.repo)]
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / "README").write_text("the checkout\n")
        os.symlink(REPO / "results", self.repo / "results")
        (self.repo / ".git" / "info" / "exclude").write_text("results\n")
        subprocess.run([*git, "add", "README"], check=True)
        subprocess.run([*git, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"],
                       check=True)
        (self.moe / "env.sh").write_text(f"""
export MOE_HOME="{self.moe}"
export REPO="{self.repo}"
export WORKSPACE="{self.moe}"
export PY_BASE="{self.bin / 'fakepy'}"
export PY_VLLM="{self.bin / 'fakepy'}"
export RESULTS_ROOT="{self.results}"
unset MOE_RESULTS_DIR
export SESSION_ROOT="{self.session}"
export MOE_CARD={CARD}
export MOE_COUNTER_LAUNCHER=""
moe_counter() {{ "$@"; }}
""")

    def env(self, **extra) -> dict:
        env = {**os.environ, "HOME": str(self.home), "MOE_HOME": str(self.moe),
               "PATH": f"{self.bin}:{os.environ['PATH']}", "FAKE_LOG": str(self.log),
               "FAKE_STATE": str(self.state), "FAKE_SCENARIO": str(self.scenario_path),
               "REAL_REPO": str(REPO), "CUDA_VISIBLE_DEVICES": "",
               "MOE_DRIVER_BUSY_WAIT_S": "0"}
        env.pop("MOE_RESULTS_DIR", None)
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def run(self, *args, push: bool = False, timeout: int = 300, **extra):
        argv = ["bash", str(DRIVER), "--deadline", str(extra.pop("deadline", FAR)), *args]
        if not push:
            argv.append("--no-push")
        return subprocess.run(argv, env=self.env(**extra), capture_output=True, text=True,
                              timeout=timeout)

    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def tool(self, name: str, dry: bool | None = None) -> list[list[str]]:
        return [c["argv"] for c in self.calls()
                if c.get("tool") == name and (dry is None or c["dry"] == dry)]

    def smi(self) -> list[list[str]]:
        return [c["argv"] for c in self.calls() if c["exe"] == "nvidia-smi"]

    def ledger(self) -> str:
        p = self.session / "gh200-driver" / "status"
        return p.read_text() if p.exists() else ""

    def base(self) -> str:
        return (self.session / "base-tag.txt").read_text().strip()


def run_box(tmp_path, scenario=None, *args, **kw) -> tuple[Box, subprocess.CompletedProcess]:
    box = Box(tmp_path, scenario or {}, real_timeout=kw.pop("real_timeout", False))
    return box, box.run(*args, **kw)


def val(argv, flag):
    return argv[argv.index(flag) + 1]


def groups_of(argv):
    i = argv.index("--groups") + 1
    out = []
    while i < len(argv) and not argv[i].startswith("--"):
        out.append(argv[i])
        i += 1
    return out


def r3_side(argv):
    return argv[argv.index("--") + 1:]


# --------------------------------------------------------------------------
# the whole session, once, as the owner's plan runs it
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    return run_box(tmp_path_factory.mktemp("whole"))


def test_the_whole_session_runs_every_step_in_the_owners_order(whole):
    box, got = whole
    assert got.returncode == exit_codes.DONE, got.stdout + got.stderr
    ledger = box.ledger()
    starts = re.findall(r"^\S+ (\w+) START", ledger, re.M)
    assert starts == ["prelude", "bytes", "calibrate", "timed", "eta", "floor", "deep", "r1lock"]
    summary = (box.session / "gh200-driver" / "summary.txt").read_text()
    assert len(summary.splitlines()) == 8 and "SKIPPED" not in summary
    assert re.fullmatch(B_RE, box.base())


def test_the_byte_pages_are_the_owners_ladder_at_the_lock_then_one_base_control(whole):
    box, _ = whole
    runs = [a for a in box.tool("dram_counter_route") if "--run" in a and "--floor" not in a]
    locked = [a for a in runs if "--page-lock-mhz" in a]
    assert [val(a, "--group-m") for a in locked] == ["1", "2", "4", "16", "8", "32", "3", "64"]
    for a in locked:
        assert val(a, "--tiles") == "1,2,3,4,5,6,7,8,9"
        assert val(a, "--page-clock") == "none" and val(a, "--page-lock-mhz") == "1710"
        assert val(a, "--census") == str(box.session / "census.json")
        g = val(a, "--group-m")
        assert val(a, "--out").endswith(f"-{CARD}-r3-counters/lock1710/r3c-g{g}.json")
    base = [a for a in runs if "--page-lock-mhz" not in a]
    assert len(base) == 1 and val(base[0], "--group-m") == "2" and "--page-clock" not in base[0]
    assert val(base[0], "--out").endswith("/base/r3c-g2.json")
    assert val(base[0], "--tiles") == "1,2,3,4,5,6,7,8,9"
    analyse = [a for a in box.tool("dram_counter_route") if "--analyse" in a]
    assert len(analyse) == 1
    refs = analyse[0][analyse[0].index("--timed-reference") + 1:analyse[0].index("--out")]
    assert len(refs) == 5 and all(r.endswith("/report.json") for r in refs)
    pages = analyse[0][analyse[0].index("--analyse") + 1:analyse[0].index("--timed-reference")]
    assert sorted(Path(p).name for p in pages) == sorted(
        f"r3c-g{g}.json" for g in (1, 2, 4, 16, 8, 32, 3, 64))


def test_calibrate_runs_after_the_byte_pages_with_no_lock_in_force(whole):
    box, _ = whole
    calls = box.calls()
    cal = next(i for i, c in enumerate(calls) if c.get("tool") == "calibrate_hardware")
    last_page = max(i for i, c in enumerate(calls) if c.get("tool") == "dram_counter_route"
                    and "--page-lock-mhz" in c["argv"])
    assert last_page < cal
    assert calls[cal]["argv"] == ["--publish", "--results-root", str(box.results)]
    locked = None
    for c in calls[:cal]:
        if c["exe"] == "nvidia-smi" and "-lgc" in c["argv"]:
            locked = c["argv"][c["argv"].index("-lgc") + 1]
        if c["exe"] == "nvidia-smi" and "-rgc" in c["argv"]:
            locked = None
    assert locked is None, "calibrate took the clock a lock left"
    ruler = (box.session / "gh200-driver" / "ruler.env").read_text()
    assert ruler.startswith("RULER_KIND=measured")


def test_every_timed_run_is_the_registered_design_at_its_lock(whole):
    box, _ = whole
    b = box.base()
    runs = box.tool("locked_r3")
    by_tag = {val(a, "--session-tag"): a for a in runs}
    assert list(by_tag) == [f"{b}-p2", f"{b}-p5", f"{b}-eta1410", f"{b}-eta1500",
                            f"{b}-eta1605", f"{b}-deep"]
    want = {"p2": ("1710", ["8", "32"], "6"), "p5": ("1710", ["3"], "8"),
            "eta1410": ("1410", ["4", "2"], "6"), "eta1500": ("1500", ["4"], "6"),
            "eta1605": ("1605", ["4"], "6"), "deep": ("1710", ["4", "2"], "9")}
    for suffix, (lock, groups, treads) in want.items():
        a = by_tag[f"{b}-{suffix}"]
        assert val(a, "--locks") == lock and groups_of(a) == groups, suffix
        r3 = r3_side(a)
        assert val(r3, "--treads") == treads and "--ridge" not in r3, suffix
        for flag, v in (("--model", "mixtral-8x7b"), ("--block-m", "32"), ("--repeats", "9"),
                        ("--duty", "0.25"), ("--seed", "0")):
            assert val(r3, flag) == v, (suffix, flag)


def test_each_timed_design_passes_its_own_dry_run_before_its_lock(whole):
    box, _ = whole
    calls = box.calls()
    for i, c in enumerate(calls):
        if c.get("tool") != "locked_r3":
            continue
        tag = val(c["argv"], "--session-tag")
        treads = val(r3_side(c["argv"]), "--treads")
        for g in groups_of(c["argv"]):
            dry = [d for d in calls[:i] if d.get("tool") == "private_weight_reference"
                   and d["dry"] and val(d["argv"], "--group-m") == g
                   and val(d["argv"], "--treads") == treads]
            assert dry, f"{tag} G={g}: no dry run before the lock"


def test_r1_runs_last_at_the_locks_eta_held(whole):
    box, _ = whole
    runs = box.tool("clock_elasticity", dry=False)
    assert [val(a, "--group-m") for a in runs] == ["4", "1"]
    for a in runs:
        i = a.index("--lock-clocks")
        assert a[i + 1:i + 4] == ["1710", "1500", "1410"]
        assert val(a, "--session-tag") == f"{box.base()}-r1lock"
    assert [val(a, "--group-m") for a in box.tool("clock_elasticity", dry=True)] == ["4", "1"]


def test_the_floor_captures_are_the_drafts_four(whole):
    box, _ = whole
    floors = [a for a in box.tool("dram_counter_route") if "--floor" in a]
    assert [Path(val(a, "--out")).name for a in floors] == [
        "r3f-g64.json", "r3f-g2.json", "r3f-g64-unlocked.json", "r3f-g64-lock1710.json"]
    assert "--floor-clock" not in floors[0] and "--floor-clock" not in floors[1]
    assert val(floors[2], "--floor-clock") == "none" and "--floor-lock-mhz" not in floors[2]
    assert val(floors[3], "--floor-clock") == "none"
    assert val(floors[3], "--floor-lock-mhz") == "1710"


def test_the_prelude_sets_persistence_and_700_w_before_any_lock(whole):
    box, _ = whole
    smi = box.smi()
    i_pm = smi.index(["-pm", "1"])
    i_pl = smi.index(["-pl", "700", "-sc", "0"])
    first_lock = next(i for i, a in enumerate(smi) if "-lgc" in a)
    assert i_pm < i_pl < first_lock
    assert (box.session / "power-at-start.txt").exists()
    assert (box.session / "power-set.txt").exists()


def test_every_lock_is_reset_and_the_card_ends_unlocked(whole):
    box, _ = whole
    locked = None
    for a in box.smi():
        if "-lgc" in a:
            assert locked is None, "a lock set over a lock nothing reset"
            locked = a[a.index("-lgc") + 1]
        if "-rgc" in a:
            locked = None
    assert locked is None
    assert json.loads(box.state.read_text())["locked"] is None


def test_the_driver_ledger_names_no_card(whole):
    """locked_r3.py's ledger opens with a `card:` line (the UUID); the driver
    copies only the lines that name none."""
    box, _ = whole
    ledger = box.ledger()
    assert "card:" not in ledger and "GPU-deadbeef" not in ledger
    assert re.search(r"G=8 held at lock 1710: R3 exit 0", ledger)


# --------------------------------------------------------------------------
# guard: every command line the driver ran, held to the tool's own parser
# --------------------------------------------------------------------------

def test_every_command_line_parses_under_its_tools_own_parser(whole):
    import clock_elasticity as CE
    import dram_counter_route as DCR
    import locked_r3 as LR
    import private_weight_reference as R3
    box, _ = whole
    seen = set()
    for c in box.calls():
        tool, argv = c.get("tool"), c["argv"]
        if tool == "dram_counter_route":
            DCR.build_parser().parse_args(argv)
        elif tool == "private_weight_reference":
            R3.build_parser().parse_args(argv)
        elif tool == "locked_r3":
            mine, r3 = argv[:argv.index("--")], r3_side(argv)
            LR.build_parser().parse_args(mine)
            assert not [t for t in r3 if LR.owned_flag(t)], r3
            R3.build_parser().parse_args(r3 + ["--group-m", "4", "--session-tag", "t"])
        elif tool == "clock_elasticity":
            CE.build_parser().parse_args(argv)
        elif tool == "calibrate_hardware":
            cal = REPO / "scripts" / "calibrate_hardware.py"
            help_text = subprocess.run([PY, str(cal), "--help"], capture_output=True, text=True,
                                       timeout=120).stdout
            assert all(f in help_text for f in argv if f.startswith("--"))
        else:
            continue
        seen.add(tool)
    assert seen == {"dram_counter_route", "private_weight_reference", "locked_r3",
                    "clock_elasticity", "calibrate_hardware"}


def _driver_array(name: str) -> list[str]:
    m = re.search(rf"^{name}=\(([^)]*)\)", DRIVER.read_text(), re.M)
    return m.group(1).split()


def test_the_drivers_locks_and_treads_are_the_registered_ones():
    import private_weight_reference as R3
    import r3_timing_model as TM
    assert [int(x) for x in _driver_array("P1_LOCKS")] == [int(f) for f in TM.P1_CLOCKS]
    src = DRIVER.read_text()
    assert re.search(r"^P1_FALLBACK_1605=1590$", src, re.M)
    assert "(1590: 0.5909" in (REPO / "scripts" / "r3_timing_model.py").read_text()
    treads = re.search(r"^TREADS=(\S+)", src, re.M).group(1)
    assert treads == ",".join(str(n) for n in range(1, R3.COUNTER_MAX_TREADS + 1))
    assert re.search(r"^LOCK_TIMED=1710$", src, re.M)
    assert re.search(r"^POWER_LIMIT_W=700$", src, re.M)
    assert _driver_array("BYTE_GS") == ["1", "2", "4", "16", "8", "32", "3", "64"]


def test_the_fallback_ruler_is_the_2026_09_25_gh200_ruler():
    import yaml
    y = yaml.safe_load((REPO / "results" / "published" / "2026-09-25-nvidia_gh200_480gb-session"
                        / "calibration" / "measured_nvidia_gh200_480gb.yaml").read_text())
    bw, tf = y["detail"]["achieved_bandwidth_gbps"], y["detail"]["achieved_bf16_tflops"]
    assert _driver_array("RULER_FALLBACK") == [
        "--ridge", f"{tf * 1e3 / bw:.2f}", "--bandwidth-gbps", f"{bw:.1f}"]


def test_the_timed_references_are_the_gh200s_five_lock1710_pages():
    src = DRIVER.read_text()
    d = REPO / re.search(r'^TIMED_REF_DIR="([^"]+)"', src, re.M).group(1)
    for rid in _driver_array("TIMED_REF_IDS"):
        hits = list(d.glob(f"*{rid}/report.json"))
        assert len(hits) == 1, rid
        j = json.loads(hits[0].read_text())
        assert j["session_tag"].endswith("-lock1710") and j["gpu_name"].startswith("NVIDIA GH200")


def test_the_scripts_parse_under_bash_n():
    for f in (DRIVER, PUSH, VMRUN):
        got = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True, timeout=60)
        assert got.returncode == 0, (f.name, got.stderr)


def test_the_plan_prints_and_refuses_as_every_dry_run_here():
    got = subprocess.run(["bash", str(DRIVER), "--dry-run"], capture_output=True, text=True,
                         timeout=60, env={**os.environ, "HOME": "/nonexistent"})
    assert got.returncode == exit_codes.REFUSED
    assert got.stdout.startswith("THE GH200 MODEL-TEST SESSION")
    for s in ("prelude", "bytes", "calibrate", "timed", "eta", "floor", "deep", "r1lock"):
        assert re.search(rf"^  {s}\s+\d+\s+\d+", got.stdout, re.M), s


# --------------------------------------------------------------------------
# the draft's Stops, as rules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("G,fail,rc,nopage,goes_on", [
    (32, ["V7"], 3, False, True),          # a written INVALID page goes on
    (32, ["V6", "V7"], 3, False, True),    # whatever its gates: they re-score on the laptop
    (16, ["V7"], 3, False, True),
    (4, ["V10"], 3, False, True),
    (32, ["V7"], 3, True, False),          # no page written: stop
    (2, [], 2, False, False),              # refused: stop
    (8, [], 4, False, False),              # a crash: stop
])
def test_the_byte_loop_goes_on_past_every_written_page(tmp_path, G, fail, rc, nopage, goes_on):
    box, got = run_box(tmp_path, {f"bytes:{G}": {"rc": rc, "fail": fail, "nopage": nopage}})
    order = [1, 2, 4, 16, 8, 32, 3, 64]
    ran = [int(val(a, "--group-m")) for a in box.tool("dram_counter_route")
           if "--page-lock-mhz" in a]
    assert ran == (order if goes_on else order[:order.index(G) + 1])
    # the session goes on either way: the stop is the byte block's
    assert box.tool("calibrate_hardware") and box.tool("clock_elasticity", dry=False)
    assert got.returncode == (exit_codes.DONE if goes_on else
                              exit_codes.ERROR if rc == 4 else exit_codes.INVALID)
    assert [a for a in box.tool("dram_counter_route") if "--analyse" in a]


def test_a_slow_first_byte_page_drops_g64_and_the_base_control(tmp_path):
    box, got = run_box(tmp_path, {}, MOE_DRIVER_REPRICE_S=-1)
    runs = [a for a in box.tool("dram_counter_route") if "--run" in a and "--floor" not in a]
    assert [val(a, "--group-m") for a in runs] == ["1", "2", "4", "16", "8", "32", "3"]
    assert "the G=64 page and the base-clock control dropped" in box.ledger()


@pytest.mark.parametrize("rc,results,kept", [
    (1, ["RESULT: CLAIM not_throttled FAIL under the floor"], False),
    (1, ["RESULT: CLAIM not_throttled PASS", "RESULT: CLAIM C2 FAIL steady"], True),
    (2, ["REFUSED: no card"], False),
    (4, [], False),
    (3, ["RESULT: VALIDITY clock_established FAIL"], False),
])
def test_calibrate_either_stands_or_every_r3_takes_the_0925_ruler(tmp_path, rc, results, kept):
    box, got = run_box(tmp_path, {"calibrate": {"rc": rc, "results": results}})
    ruler = box.repo / "moe" / "bench" / "hardware" / f"measured_{CARD}.yaml"
    assert ruler.exists() is kept
    r3_lines = [r3_side(a) for a in box.tool("locked_r3")]
    r3_lines += [a for a in box.tool("private_weight_reference", dry=True)]
    assert r3_lines
    for a in r3_lines:
        if kept:
            assert "--ridge" not in a
        else:
            assert val(a, "--ridge") == "177.93" and val(a, "--bandwidth-gbps") == "3725.1"
    assert ("FALLBACK" in box.ledger()) is not kept


def test_1605_unsupported_takes_the_registered_1590(tmp_path):
    clocks = [c for c in range(1980, 344, -15) if c != 1605]
    box, got = run_box(tmp_path, {"clocks": clocks})
    tags = [val(a, "--session-tag") for a in box.tool("locked_r3")]
    assert f"{box.base()}-eta1590" in tags and f"{box.base()}-eta1605" not in tags
    assert "the registered fallback for 1605" in box.ledger()


def test_an_unsupported_1410_is_substituted_and_labelled(tmp_path):
    clocks = [c for c in range(1980, 344, -15) if c != 1410] + [1417]
    box, got = run_box(tmp_path, {"clocks": sorted(clocks, reverse=True)})
    a = next(a for a in box.tool("locked_r3") if "-eta14" in val(a, "--session-tag"))
    assert val(a, "--locks") == "1417"
    assert "SUBSTITUTED for 1410: P1 at it is computed after registration" in box.ledger()


@pytest.mark.parametrize("scenario,extra,why", [
    ({"pl_fail": True}, {}, "power limit was not set"),
    ({"pl_reads": 650}, {}, "power.limit reads"),
    ({"busy": True}, {}, "the GPU is in use"),
    ({"clocks": [1980, 1965, 1500, 1410, 1605]}, {}, "1710 MHz is not a supported"),
])
def test_the_prelude_refuses_and_nothing_is_locked(tmp_path, scenario, extra, why):
    box, got = run_box(tmp_path, scenario, **extra)
    assert got.returncode == exit_codes.REFUSED
    assert why in box.ledger(), box.ledger()
    assert not [a for a in box.smi() if "-lgc" in a]
    assert not [c for c in box.calls() if c.get("tool")]
    assert re.findall(r"^\S+ (\w+) START", box.ledger(), re.M) == ["prelude"]


def test_a_dirty_checkout_is_refused(tmp_path):
    box = Box(tmp_path, {})
    (box.repo / "stray.txt").write_text("x")
    got = box.run()
    assert got.returncode == exit_codes.REFUSED
    assert "the checkout is not clean" in box.ledger()


def test_a_failed_reset_stops_the_session(tmp_path):
    box, got = run_box(tmp_path, {"rgc_fail": True})
    assert got.returncode == exit_codes.ERROR
    assert "HYGIENE: nvidia-smi -rgc failed" in box.ledger()
    assert re.findall(r"^\S+ (\w+) START", box.ledger(), re.M) == ["prelude"]


def test_a_slipped_eta_lock_is_reset_and_tried_once_more_under_a_new_tag(tmp_path):
    box, got = run_box(tmp_path, {"locked": {"eta1500": 1, "eta1500b": 1}})
    tags = [val(a, "--session-tag") for a in box.tool("locked_r3")]
    b = box.base()
    assert tags.count(f"{b}-eta1500") == 1 and tags.count(f"{b}-eta1500b") == 1
    assert f"{b}-eta1605" in tags, "the next lock still runs"
    # R1 takes only the locks that held: 1500 slipped twice
    r1 = box.tool("clock_elasticity", dry=False)[0]
    i = r1.index("--lock-clocks")
    assert r1[i + 1:i + 3] == ["1710", "1410"] and r1[i + 3].startswith("--")


def test_a_design_whose_dry_run_refuses_is_never_locked(tmp_path):
    box, got = run_box(tmp_path, {"dry_refuse": ["2:9", "32:6"]})
    b = box.base()
    by_tag = {val(a, "--session-tag"): a for a in box.tool("locked_r3")}
    assert groups_of(by_tag[f"{b}-p2"]) == ["8"]
    assert groups_of(by_tag[f"{b}-deep"]) == ["4"]
    assert "may not lock: REFUSED" in box.ledger()


def test_no_ruler_means_no_lock(tmp_path):
    """R3 without a ruler plans on the H200 HYPOTHESIS and refuses the timed
    run on the card: the dry-run check catches it before any lock."""
    box, got = run_box(tmp_path, {"hypothesis": True})
    assert not box.tool("locked_r3")
    assert "no ruler: the ridge is R3's H200 HYPOTHESIS" in box.ledger()


def _left(box, minutes):
    now = 2_000_000_000
    return {"MOE_DRIVER_NOW": now, "deadline": now + (minutes + 8) * 60}


def test_the_deadline_drops_r1_at_g1_first(tmp_path):
    box = Box(tmp_path, {})
    got = box.run(**_left(box, 240))
    assert "DROPPED R1 at G=1" in box.ledger() and "DROPPED R1 at G=4" not in box.ledger()
    assert [val(a, "--group-m") for a in box.tool("clock_elasticity", dry=False)] == ["4"]
    assert got.returncode == exit_codes.DONE


def test_the_deadline_drops_in_the_drafts_order_and_skips_what_is_left(tmp_path):
    box = Box(tmp_path, {})
    got = box.run(**_left(box, 150))
    drops = re.findall(r"DROPPED (.+?):", box.ledger())
    assert drops == ["R1 at G=1", "R1 at G=4", "the 1605 lock",
                     "the G=64 byte page and the base-clock control"]
    assert "SKIPPED r1lock: every part of it is dropped" in box.ledger()
    assert not box.tool("clock_elasticity")
    assert f"{box.base()}-eta1605" not in [val(a, "--session-tag") for a in box.tool("locked_r3")]
    assert got.returncode == exit_codes.INVALID


def test_a_prelude_with_no_time_left_stops_everything(tmp_path):
    box = Box(tmp_path, {})
    got = box.run(**_left(box, 1))
    assert "SKIPPED prelude" in box.ledger() and not box.smi()
    assert got.returncode == exit_codes.INVALID


def test_a_step_refuses_without_the_prelude(tmp_path):
    box = Box(tmp_path, {})
    got = box.run("--from", "bytes")
    assert "the prelude has not passed on this VM" in (
        box.session / "logs" / "driver-bytes.log").read_text()
    assert not box.tool("dram_counter_route")
    assert got.returncode == exit_codes.INVALID


def test_the_real_dry_runs_pass_the_drivers_checks_with_the_fallback_ruler(tmp_path):
    """R3's and R1's own dry runs, on the laptop, with the arguments the driver
    builds: the retracted-tread and declaration lines the driver reads are
    there, and every design locks (with the 2026-09-25 ruler's numbers, as the
    fallback hands them; without a ruler, R3 plans on the H200 hypothesis,
    which test_no_ruler_means_no_lock holds)."""
    box, got = run_box(tmp_path, {"calibrate": {"rc": 4, "results": []}}, FAKE_REAL_DRY=1,
                       timeout=900)
    assert "may not lock" not in box.ledger(), box.ledger()
    tags = [val(a, "--session-tag").rsplit("-", 1)[-1] for a in box.tool("locked_r3")]
    assert tags == ["p2", "p5", "eta1410", "eta1500", "eta1605", "deep"]
    for log in (box.session / "logs").glob("r3-*-dry.log"):
        text = log.read_text()
        assert "retracted  tread   11" in text and "n_decl = 9 against" in text, log.name
    assert got.returncode == exit_codes.ERROR   # calibrate itself crashed (exit 4)


@pytest.mark.skipif(not (shutil.which("timeout") or shutil.which("gtimeout")),
                    reason="no GNU timeout on this machine")
def test_sigterm_stops_the_step_resets_the_card_and_pushes(tmp_path):
    box = Box(tmp_path, {"hang": "p2"}, real_timeout=True)
    proc = subprocess.Popen(["bash", str(DRIVER), "--deadline", str(FAR), "--no-push"],
                            env=box.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    marker = box.log.with_suffix(".hanging")
    t = time.time()
    while not marker.exists():
        assert time.time() - t < 120 and proc.poll() is None, "the timed step never started"
        time.sleep(0.2)
    lock_state = json.loads(box.state.read_text())["locked"]
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 143, out
    assert "STOPPED by SIGTERM during timed" in box.ledger()
    assert json.loads(box.state.read_text())["locked"] is None
    hung = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(hung, 0)
    del lock_state


# --------------------------------------------------------------------------
# the results push
# --------------------------------------------------------------------------

def push_env(home: Path, remote: Path, **extra) -> dict:
    env = {**os.environ, "HOME": str(home), "MOE_PUSH_REMOTE": str(remote),
           "MOE_PUSH_BACKOFF_S": "0"}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def push(home, remote, *args, **extra):
    return subprocess.run(["bash", str(PUSH), *args], env=push_env(home, remote, **extra),
                          capture_output=True, text=True, timeout=120)


@pytest.fixture
def pushed(tmp_path):
    home, remote = tmp_path / "home", tmp_path / "remote.git"
    (home / "moe" / "results" / "gaps").mkdir(parents=True)
    (home / "moe" / "session").mkdir(parents=True)
    (home / "moe" / "session" / "commit.txt").write_text("0123abc\n")
    (home / "moe" / "results" / "gaps" / "page.json").write_text("{}\n")
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    got = push(home, remote, "init", "--branch", "run-gh200-2026-09-27", "--run-id", "r1",
               "--name", "x", "--email", "x@y.z")
    assert got.returncode == 0, got.stderr
    return home, remote, got


def tree(remote, branch="run-gh200-2026-09-27"):
    return subprocess.run(["git", "-C", str(remote), "ls-tree", "-r", "--name-only", branch],
                          capture_output=True, text=True).stdout.split()


def show(remote, path, branch="run-gh200-2026-09-27"):
    return subprocess.run(["git", "-C", str(remote), "show", f"{branch}:{path}"],
                          capture_output=True, text=True).stdout


def test_init_prints_only_the_public_half_of_a_key_made_here(pushed):
    home, remote, got = pushed
    key = home / ".ssh" / "moe_results_ed25519"
    assert key.exists() and (key.parent / "moe_results_ed25519.pub").exists()
    assert oct(key.parent.stat().st_mode)[-3:] == "700"
    pub = [ln for ln in got.stdout.splitlines() if ln.startswith("PUBLIC-KEY ")]
    assert len(pub) == 1 and pub[0].split()[1] == "ssh-ed25519"
    assert "PRIVATE" not in got.stdout and key.read_text() not in got.stdout
    known = (home / "moe" / "push-known_hosts").read_text()
    assert known.startswith("github.com ssh-ed25519 "
                            "AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl")


@pytest.mark.parametrize("branch", ["main", "model-2026-09-26", "integrate-2026-09-26",
                                    "run-", "Run-X", "run-a b"])
def test_results_never_go_to_a_branch_that_is_not_the_runs_own(tmp_path, branch):
    got = push(tmp_path, tmp_path / "r.git", "init", "--branch", branch, "--run-id", "r",
               "--name", "x", "--email", "x@y.z")
    assert got.returncode == exit_codes.REFUSED


def test_a_push_is_an_orphan_snapshot_with_its_manifest(pushed):
    home, remote, _ = pushed
    got = push(home, remote, "push", "--message", "3c.1 prelude: exit 0")
    assert got.returncode == 0, got.stdout + got.stderr
    names = tree(remote)
    assert {"README.md", "SHA256SUMS", "HELD-BACK.txt", "PUSHES.txt",
            "vm/results/gaps/page.json", "vm/session/commit.txt"} <= set(names)
    assert not [n for n in names if n.startswith(("vm/push", ".ssh", "vm/.ssh")) or "push.env" in n]
    log = subprocess.run(["git", "-C", str(remote), "log", "--format=%P %an <%ae> %s",
                          "run-gh200-2026-09-27"], capture_output=True, text=True).stdout
    assert log.splitlines() == [" x <x@y.z> r1: 3c.1 prelude: exit 0"], "one parentless commit"
    sums = show(remote, "SHA256SUMS").splitlines()
    assert {ln.split("  ", 1)[1] for ln in sums} == {n for n in names if n.startswith("vm/")}
    assert "0123abc" in show(remote, "README.md")


def test_every_push_carries_the_whole_snapshot_and_a_final_one_marks_done(pushed):
    home, remote, _ = pushed
    push(home, remote, "push", "--message", "one")
    (home / "moe" / "results" / "gaps" / "page.json").unlink()
    (home / "moe" / "results" / "gaps" / "page2.json").write_text("{}\n")
    got = push(home, remote, "push", "--message", "two", "--final", "3")
    assert got.returncode == 0
    names = tree(remote)
    assert "vm/results/gaps/page2.json" in names and "vm/results/gaps/page.json" not in names
    assert show(remote, "DRIVER-DONE").startswith("driver exit 3 at ")
    assert [ln.split(" ", 1)[1] for ln in show(remote, "PUSHES.txt").splitlines()] == ["one", "two"]


def test_a_file_over_the_limit_is_compressed_or_left_on_the_vm(pushed):
    home, remote, _ = pushed
    (home / "moe" / "results" / "gaps" / "zeros.bin").write_bytes(b"\0" * 5000)
    (home / "moe" / "results" / "gaps" / "noise.bin").write_bytes(os.urandom(5000))
    got = push(home, remote, "push", "--message", "big", MOE_PUSH_MAX_BYTES=3000)
    assert got.returncode == 0
    names = tree(remote)
    held = show(remote, "HELD-BACK.txt")
    assert "vm/results/gaps/noise.bin" not in names and "LEFT ON THE VM" in held
    if shutil.which("xz"):
        assert "vm/results/gaps/zeros.bin.xz" in names and "COMPRESSED" in held
    assert "vm/results/gaps/zeros.bin" not in names


def test_a_failed_push_is_committed_and_carried_by_the_next(pushed, tmp_path):
    home, remote, _ = pushed
    moved = tmp_path / "gone.git"
    remote.rename(moved)
    got = push(home, remote, "push", "--message", "lost", MOE_PUSH_TRIES=2)
    assert got.returncode == exit_codes.CLAIM_FAIL and "NOT PUSHED" in got.stdout
    moved.rename(remote)
    assert push(home, remote, "push", "--message", "found").returncode == 0
    lines = show(remote, "PUSHES.txt").splitlines()
    assert [ln.split(" ", 1)[1] for ln in lines] == ["lost", "found"]


def test_the_push_never_touches_the_measured_checkout(pushed):
    home, remote, _ = pushed
    repo = home / "moe" / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "f").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qm", "c"], check=True)
    push(home, remote, "push", "--message", "m")
    st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
                        text=True).stdout
    assert st == ""


def test_a_push_before_init_is_refused(tmp_path):
    got = push(tmp_path, tmp_path / "r.git", "push", "--message", "m")
    assert got.returncode == exit_codes.REFUSED and "not initialised" in got.stderr


def test_the_driver_pushes_after_every_step(tmp_path):
    box = Box(tmp_path / "box", {})
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    env = {"MOE_PUSH_REMOTE": str(remote), "MOE_PUSH_BACKOFF_S": "0"}
    got = subprocess.run(["bash", str(PUSH), "init", "--branch", "run-gh200-t", "--run-id", "t",
                          "--name", "x", "--email", "x@y.z"], env=box.env(**env),
                         capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    got = box.run(push=True, **env)
    assert got.returncode == exit_codes.DONE, got.stdout
    pushes = [ln.split(" ", 1)[1] for ln in show(remote, "PUSHES.txt", "run-gh200-t").splitlines()]
    assert [p.split(":")[0] for p in pushes] == [
        "prelude", "bytes", "calibrate", "timed", "eta", "floor", "deep", "r1lock",
        "the session ended"]
    assert show(remote, "DRIVER-DONE", "run-gh200-t").startswith("driver exit 0")
    status = show(remote, "vm/session/gh200-driver/status", "run-gh200-t")
    assert "r1lock END" in status and "card:" not in status


# --------------------------------------------------------------------------
# the laptop's side
# --------------------------------------------------------------------------

FAKESSH = r'''#!__PY__ -S
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
while args and args[0] == "-o":
    args = args[2:]
host, cmd = args[0], " ".join(args[1:])
st_path = Path(os.environ["FAKE_VM"])
st = json.loads(st_path.read_text())
st.setdefault("cmds", []).append(cmd)
out, rc = "", 0
if cmd == "true":
    pass
elif "driver_version" in cmd:
    out = st["driver"]
elif "boot_id" in cmd:
    out = str(st["boot"])
elif "systemctl reboot" in cmd:
    st["boot"] += 1; st["driver"] = st.get("after", st["driver"])
elif "apt-get" in cmd and "install" in cmd:
    rc = st.get("apt_rc", 0) if "server-open=" in cmd else 0
elif "vm_results_push.sh init" in cmd:
    out = "[push] ready\nPUBLIC-KEY ssh-ed25519 AAAAC3testkey moe-results r"
elif "vm_results_push.sh check" in cmd:
    rc = st.get("check_rc", 0)
elif "--dry-run" in cmd:
    out = "THE GH200 MODEL-TEST SESSION (plan)"
elif "nohup setsid" in cmd:
    out = "STARTED"
st_path.write_text(json.dumps(st))
if out:
    print(out)
sys.exit(rc)
'''

FAKEGH = r'''#!__PY__ -S
import json, os, sys
from pathlib import Path
p = Path(os.environ["FAKE_GH"])
st = json.loads(p.read_text()) if p.exists() else {"keys": {}, "calls": []}
a = sys.argv[1:]
st["calls"].append(a)
if a[:3] == ["api", "-X", "POST"]:
    kid = str(1000 + len(st["keys"]))
    st["keys"][kid] = next(x for x in a if x.startswith("key="))[4:]
    print(kid)
elif a[:3] == ["api", "-X", "DELETE"]:
    st["keys"].pop(a[3].rsplit("/", 1)[1], None)
elif a[:1] == ["api"]:
    print("\n".join(st["keys"]))
p.write_text(json.dumps(st))
'''


class Laptop:
    """A clone of a repo whose origin is a local bare repository, holding a
    copy of vm_run.sh (its ROOT is the clone), with ssh, scp and gh stubbed."""

    def __init__(self, root: Path, driver="570.148.08", after="580.105.08", **vm):
        self.root = root
        self.remote = root / "origin.git"
        self.clone = root / "clone"
        self.bin = root / "bin"
        self.state = root / "runs"
        self.vm = root / "vm.json"
        self.gh = root / "gh.json"
        self.bin.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", "--bare", str(self.remote)], check=True)
        subprocess.run(["git", "clone", "-q", str(self.remote), str(self.clone)], check=True,
                       capture_output=True)
        (self.clone / "scripts").mkdir()
        for f in (VMRUN, PUSH, DRIVER, REPO / "scripts" / "setup_vm.sh"):
            shutil.copy(f, self.clone / "scripts" / f.name)
        g = ["git", "-C", str(self.clone)]
        subprocess.run([*g, "config", "user.name", "x"], check=True)
        subprocess.run([*g, "config", "user.email", "x@y.z"], check=True)
        subprocess.run([*g, "add", "-A"], check=True)
        subprocess.run([*g, "commit", "-qm", "c"], check=True)
        subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-t"], check=True,
                       capture_output=True)
        _exe(self.bin / "ssh", FAKESSH)
        _exe(self.bin / "scp", "#!/bin/bash\nexit 0\n")
        _exe(self.bin / "gh", FAKEGH)
        self.vm.write_text(json.dumps({"driver": driver, "after": after, "boot": 1, **vm}))

    def run(self, *args):
        env = {**os.environ, "MOE_VM_SSH": str(self.bin / "ssh"),
               "MOE_VM_SCP": str(self.bin / "scp"),
               "MOE_VM_GH": str(self.bin / "gh"), "MOE_VM_STATE_ROOT": str(self.state),
               "MOE_VM_POLL_S": "0", "MOE_VM_WAIT_S": "5", "FAKE_VM": str(self.vm),
               "FAKE_GH": str(self.gh)}
        return subprocess.run(["bash", str(self.clone / "scripts" / "vm_run.sh"), *args], env=env,
                              capture_output=True, text=True, timeout=120)

    def make_branch(self, branch="run-gh200-t", final=None):
        """The VM's pushes, played by the real push script into origin."""
        home = self.root / "vmhome"
        (home / "moe" / "session" / "gh200-driver").mkdir(parents=True, exist_ok=True)
        (home / "moe" / "session" / "gh200-driver" / "status").write_text("t0 prelude START\n")
        env = push_env(home, self.remote)
        if not (home / "moe" / "push.env").exists():
            subprocess.run(["bash", str(PUSH), "init", "--branch", branch, "--run-id", "r",
                            "--name", "x", "--email", "x@y.z"], env=env, check=True,
                           capture_output=True)
        args = ["push", "--message", "a step"]
        if final is not None:
            args += ["--final", str(final)]
        subprocess.run(["bash", str(PUSH), *args], env=env, check=True, capture_output=True)

    def vm_cmds(self):
        return json.loads(self.vm.read_text())["cmds"]


def test_prepare_upgrades_r570_reboots_adds_only_the_public_key_and_checks(tmp_path):
    lap = Laptop(tmp_path)
    lap.make_branch()   # the VM's first push lands on origin, as `check` does
    got = lap.run("prepare", "--ip", "10.0.0.1", "--run-id", "r", "--branch", "run-gh200-t")
    assert got.returncode == 0, got.stdout + got.stderr
    cmds = lap.vm_cmds()
    i_install = next(i for i, c in enumerate(cmds) if "apt-get" in c and "install" in c)
    assert "nvidia-driver-580-server-open=580.105.08-0lambda0.22.04.1" in cmds[i_install]
    i_reboot = next(i for i, c in enumerate(cmds) if "systemctl reboot" in c)
    i_init = next(i for i, c in enumerate(cmds) if "vm_results_push.sh init" in c)
    i_check = next(i for i, c in enumerate(cmds) if "vm_results_push.sh check" in c)
    assert i_install < i_reboot < i_init < i_check
    assert any("apt-daily" in c and "disable" in c for c in cmds[:i_install])
    gh = json.loads(lap.gh.read_text())
    post = next(c for c in gh["calls"] if c[:3] == ["api", "-X", "POST"])
    assert "read_only=false" in post and "repos/harshithkantamneni/moe-kernels/keys" in post
    assert list(gh["keys"].values()) == ["ssh-ed25519 AAAAC3testkey moe-results r"]
    run = lap.state / "r"
    assert (run / "deploy-key.id").read_text().strip() == "1000"
    env = (run / "run.env").read_text()
    assert "driver=580.105.08" in env and "branch=run-gh200-t" in env


def test_prepare_on_r580_moves_no_driver(tmp_path):
    lap = Laptop(tmp_path, driver="580.105.08")
    lap.make_branch()
    got = lap.run("prepare", "--ip", "10.0.0.1", "--run-id", "r", "--branch", "run-gh200-t")
    assert got.returncode == 0, got.stdout
    assert not [c for c in lap.vm_cmds() if "apt-get" in c or "reboot" in c]


def test_prepare_takes_the_newest_580_when_the_exact_build_is_gone(tmp_path):
    lap = Laptop(tmp_path, apt_rc=100, after="580.126.09")
    lap.make_branch()
    got = lap.run("prepare", "--ip", "10.0.0.1", "--run-id", "r", "--branch", "run-gh200-t")
    assert got.returncode == 0, got.stdout
    installs = [c for c in lap.vm_cmds() if "apt-get" in c and "install" in c]
    assert "=580.105.08" in installs[0] and installs[1].rstrip().split(">>")[0].endswith(
        "nvidia-driver-580-server-open ")


def test_prepare_refuses_a_dirty_checkout_or_a_commit_github_lacks(tmp_path):
    lap = Laptop(tmp_path / "a")
    (lap.clone / "stray").write_text("x")
    got = lap.run("prepare", "--ip", "10.0.0.1", "--run-id", "r", "--branch", "run-gh200-t")
    assert got.returncode == exit_codes.REFUSED and "local changes" in got.stdout
    lap = Laptop(tmp_path / "b")
    subprocess.run(["git", "-C", str(lap.clone), "commit", "-q", "--allow-empty", "-m", "local"],
                   check=True)
    got = lap.run("prepare", "--ip", "10.0.0.1", "--run-id", "r", "--branch", "run-gh200-t")
    assert got.returncode == exit_codes.REFUSED and "push it first" in got.stdout
    assert not Path(lap.vm).read_text().count("cmds")


@pytest.mark.parametrize("branch", ["main", "model-2026-09-26", "run-"])
def test_prepare_refuses_a_branch_that_is_not_a_runs(tmp_path, branch):
    lap = Laptop(tmp_path)
    got = lap.run("prepare", "--ip", "10.0.0.1", "--run-id", "r", "--branch", branch)
    assert got.returncode == exit_codes.REFUSED


def test_watch_waits_for_driver_done_and_verify_checks_every_file(tmp_path):
    lap = Laptop(tmp_path)
    lap.make_branch()
    run = lap.state / "r"
    run.mkdir(parents=True)
    (run / "run.env").write_text("ip=1\nbranch=run-gh200-t\ncommit=x\n")
    got = lap.run("watch", "--run-id", "r")
    assert got.returncode == exit_codes.INVALID and "prelude START" in got.stdout
    lap.make_branch(final=0)
    got = lap.run("watch", "--run-id", "r")
    assert got.returncode == 0 and "DONE: driver exit 0" in got.stdout
    got = lap.run("verify", "--run-id", "r")
    assert got.returncode == 0 and "VERIFIED" in got.stdout, got.stdout
    # a file that does not match its manifest line fails the check
    work = tmp_path / "vmhome" / "moe" / "push"
    sums = work / "SHA256SUMS"
    sums.write_text(sums.read_text().replace(sums.read_text()[:8], "00000000", 1))
    g = ["git", "-C", str(work)]
    subprocess.run([*g, "commit", "-qam", "tamper"], check=True)
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/run-gh200-t"], check=True,
                   capture_output=True)
    got = lap.run("verify", "--run-id", "r")
    assert got.returncode == exit_codes.INVALID and "VERIFY FAILED" in got.stdout


def test_forget_deletes_the_runs_deploy_key(tmp_path):
    lap = Laptop(tmp_path)
    lap.make_branch()
    assert lap.run("prepare", "--ip", "1.2.3.4", "--run-id", "r",
                   "--branch", "run-gh200-t").returncode == 0
    got = lap.run("forget", "--run-id", "r")
    assert got.returncode == 0 and "deploy key 1000 deleted" in got.stdout
    assert json.loads(lap.gh.read_text())["keys"] == {}
    assert lap.run("forget", "--run-id", "r").returncode == exit_codes.REFUSED


def test_start_refuses_a_checkout_that_moved_since_prepare(tmp_path):
    lap = Laptop(tmp_path)
    lap.make_branch()
    assert lap.run("prepare", "--ip", "1.2.3.4", "--run-id", "r",
                   "--branch", "run-gh200-t").returncode == 0
    got = lap.run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000")
    assert got.returncode == 0 and "STARTED" in got.stdout, got.stdout
    start = next(c for c in lap.vm_cmds() if "nohup setsid" in c)
    sha = subprocess.run(["git", "-C", str(lap.clone), "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    url = "https://github.com/harshithkantamneni/moe-kernels"
    assert f"--setup --commit {sha} --repo {url}" in start
    assert "--deadline 2000000000" in start
    subprocess.run(["git", "-C", str(lap.clone), "commit", "-q", "--allow-empty", "-m", "moved"],
                   check=True)
    got = lap.run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000")
    assert got.returncode == exit_codes.REFUSED and "moved since prepare" in got.stdout


def test_another_model_runs_its_own_census_and_no_8x7b_references(tmp_path):
    """--model mixtral-8x22b (the cross-model test, docs/registered): every R3,
    locked_r3 and counter command names the model; the byte and floor pages
    read a census taken for it (never the preflight's 8x7B census); the
    analysis gets no 2026-09-25 8x7B timed references; eta and R1 are left
    out; locked_r3 takes an hour per R3 run; every argv parses."""
    import dram_counter_route as DCR
    import locked_r3 as LR
    import private_weight_reference as R3
    box, got = run_box(tmp_path, {}, "--model", "mixtral-8x22b")
    assert got.returncode == exit_codes.DONE, got.stdout + got.stderr
    starts = re.findall(r"^\S+ (\w+) START", box.ledger(), re.M)
    assert starts == ["prelude", "bytes", "calibrate", "timed", "floor", "deep"]
    dcr = box.tool("dram_counter_route")
    census = [a for a in dcr if "--census-only" in a]
    assert len(census) == 1 and val(census[0], "--model") == "mixtral-8x22b"
    want = str(box.session / "census-mixtral-8x22b.json")
    assert val(census[0], "--out") == want
    runs = [a for a in dcr if "--run" in a and "--census-only" not in a]
    assert runs and all(val(a, "--model") == "mixtral-8x22b" and val(a, "--census") == want
                        for a in runs)
    assert all("--timed-reference" not in a for a in dcr if "--analyse" in a)
    for a in box.tool("locked_r3"):
        assert val(r3_side(a), "--model") == "mixtral-8x22b" and val(a, "--run-cap-s") == "3600"
        LR.build_parser().parse_args(a[:a.index("--")])
    for a in box.tool("private_weight_reference", dry=True):
        assert val(a, "--model") == "mixtral-8x22b"
        R3.build_parser().parse_args(a)
    for a in dcr:
        DCR.build_parser().parse_args(a)
    assert not box.tool("clock_elasticity")
    assert not [a for a in runs if "/base/" in val(a, "--out")], "no base-clock control"
    floors = [Path(val(a, "--out")).name for a in runs if "--floor" in a]
    assert floors == ["r3f-g64.json", "r3f-g64-lock1710.json"]


def test_start_passes_the_model_to_the_driver(tmp_path):
    lap = Laptop(tmp_path)
    lap.make_branch()
    assert lap.run("prepare", "--ip", "1.2.3.4", "--run-id", "r",
                   "--branch", "run-gh200-t").returncode == 0
    got = lap.run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000",
                  "--model", "mixtral-8x22b")
    assert got.returncode == 0, got.stdout
    start = next(c for c in lap.vm_cmds() if "nohup setsid" in c)
    plan = next(c for c in lap.vm_cmds() if "--dry-run" in c)
    assert "--model mixtral-8x22b" in start and "--model mixtral-8x22b" in plan
    assert "model=mixtral-8x22b" in (lap.state / "r" / "run.env").read_text()
