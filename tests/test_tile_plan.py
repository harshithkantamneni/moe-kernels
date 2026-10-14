"""Rental A's plan and the tile stage's driver off the GPU (scripts/plans/rentalA-tile-2026-10.plan,
scripts/gh200_tile_session.sh): the parse-only dry run, its minutes, its nodrop units and its drop
order (design-tile-review 4, 5 and 8), the tile keys and where they are refused, the driver against
the pinned one (every changed hunk marked TILE), the units reaching their tools on the fake box (R3
through r3_tile.py by locked_r3's --r3-script, the counter route through its wrapper, the census), and
`vm_run.sh start --driver gh200_tile_session.sh` with the real plan against a stub VM whose ssh drains
stdin, every delivered file compared by sha256."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_gh200_model_session as TG  # noqa: E402
import test_rental5_driver as T5  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

PLAN = REPO / "scripts" / "plans" / "rentalA-tile-2026-10.plan"
TILE_DRIVER = REPO / "scripts" / "gh200_tile_session.sh"
PINNED_DRIVER = TG.DRIVER
HA = "docs/registered/2026-10-14-tileA-hist"
NODROP = {("mixtral-8x7b", "calibrate", "ta"), ("-", "tilecensus", "ta"),
          ("olmoe-1b-7b", "bytes", "f128"), ("mixtral-8x7b", "bytes", "f128"),
          ("olmoe-1b-7b", "bytes", "f64w8"), ("mixtral-8x7b", "bytes", "f64w8"),
          ("olmoe-1b-7b", "bytes", "b128g1"), ("mixtral-8x7b", "bytes", "b64g1"), ("mixtral-8x7b", "bytes", "b128g16"),
          ("qwen2-57b-a14b", "bytes", "b128g1"),
          ("olmoe-1b-7b", "timed", "e128g1"), ("mixtral-8x7b", "timed", "e128g16"), ("mixtral-8x7b", "timed", "e64g1"),
          ("qwen2-57b-a14b", "timed", "e128g1"), ("mixtral-8x7b", "timed", "e128g1")}


def dry(*args, driver=TILE_DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


def _label(u):
    m = re.match(r"label=(\S+)", u[5])
    return m.group(1) if m else None


def _group(u):
    m = re.search(r"\[drop-group ([\w-]+)\]", u[5])
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# the plan, parse only
# --------------------------------------------------------------------------

def test_rental_a_parses_in_147_minutes_and_drops_only_dk_ms_w4_rep_last_first():
    got = dry("--plan", str(PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    assert f"a PLAN of {len(units)} units" in got.stdout and len(units) == 21
    total = sum(int(u[3]) for u in units)
    assert total == 147 and "147 min of units" in got.stdout
    assert sum(int(u[3]) for u in units if "[nodrop]" in u[5] or u[2] == "prelude") == 117
    assert [u[2] for u in units[:3]] == ["prelude", "calibrate", "tilecensus"]
    nodrop = {(u[1], u[2], _label(u)) for u in units if "[nodrop]" in u[5]}
    assert nodrop == NODROP
    seen = []
    for u in reversed(units):
        g = _group(u)
        if g and g not in seen:
            seen.append(g)
    assert seen == ["dk", "ms", "w4", "rep"]
    order = re.search(r"drop order when time runs short: .*\n\s+([\d ]+)", got.stdout).group(1).split()
    assert order == ["21", "20", "19", "18", "17"]
    dirs = [TG._unit_dir(u[5].replace(" [nodrop]", "")) for u in units if u[2] not in ("prelude", "calibrate")]
    assert len(set(dirs)) == len(dirs)
    for u in units[3:]:
        assert all(k in u[5] for k in ("block-m=", "block-n=", "block-k=64", "num-warps=", "num-stages=3")), u
        if u[2] == "timed":
            assert "slip-policy=cell" in u[5] and "g1-rule=r7-median-run3" in u[5] and "declared-copies=" in u[5]
    assert "the stage's" not in got.stderr


def test_every_file_the_plan_names_is_registered_and_the_e1cal_pages_are_the_stop_rules():
    text = PLAN.read_text()
    body = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    files = set(re.findall(r"histogram=(\S+)", body))
    assert files and all(f.startswith(HA) for f in files) and len(files) == 6
    reg = {}
    for part in ("bytes", "e1cal", "dead"):
        d = json.loads((REPO / f"docs/registered/2026-10-14-tileA-{part}-gh200.json").read_text())
        reg.update({v["path"]: v["sha256"] for v in d["pages"]["files"].values()})
    for f in files:
        assert hashlib.sha256((REPO / f).read_bytes()).hexdigest() == reg[f], f
    timed = re.findall(r"^(\S+) timed label=(e\S+) .*histogram=(\S+)", body, re.M)
    assert {(m, lab) for m, lab, _f in timed} == {("olmoe-1b-7b", "e128g1"), ("mixtral-8x7b", "e128g16"),
                                                 ("mixtral-8x7b", "e64g1"), ("qwen2-57b-a14b", "e128g1"),
                                                 ("mixtral-8x7b", "e128g1"), ("olmoe-1b-7b", "e128g1r")}
    # a byte page and the timed page of one (model, tile, G) read one file (design-tile-review 9.14)
    by = re.findall(r"^(\S+) bytes label=b(\S+) .*histogram=(\S+)", body, re.M)
    for m, lab, f in by:
        twin = [tf for tm, tl, tf in timed if tm == m and tl[1:] == lab]
        assert not twin or twin == [f], (m, lab)


@pytest.mark.parametrize("body, why", [
    ("- prelude\nolmoe-1b-7b calibrate label=a block-m=128 block-n=128 block-k=64 num-warps=8 num-stages=3\n",
     "block-m belongs to a timed or bytes unit, not calibrate"),
    ("- prelude\nolmoe-1b-7b bytes label=a block-m=128 byte-groups=64\n", "a tile is whole"),
    ("- prelude\nolmoe-1b-7b bytes label=a block-m=96 block-n=128 block-k=64 num-warps=8 num-stages=3\n",
     "block-m 96: 16, 32, 64 or 128"),
    ("- prelude\nolmoe-1b-7b bytes label=a block-m=128 block-n=128 block-k=64 num-warps=6 num-stages=3\n", "num-warps 6: 4 or 8"),
    ("- prelude\nolmoe-1b-7b bytes label=a tiles=128x128x64w8s3\n", "tiles belongs to a tilecensus unit, not bytes"),
    ("- prelude\n- tilecensus label=c tiles=128x128x64w8s3\n", "a tilecensus unit needs tiles= and models="),
    ("- prelude\n- tilecensus tiles=128x128x64w8s3 models=olmoe-1b-7b\n", "a tilecensus unit names its file: label=..."),
    ("- prelude\n- tilecensus label=c tiles=128x128 models=olmoe-1b-7b\n", "tiles 128x128: BMxBNxBKw<warps>s<stages>"),
])
def test_the_tile_keys_are_refused_where_they_cannot_run(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


def test_the_pinned_driver_refuses_the_tile_keys():
    got = dry("--plan", str(PLAN), driver=PINNED_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "no step 'tilecensus'" in got.stderr


def test_the_tile_driver_is_the_pinned_driver_plus_lines_marked_tile():
    """build-tile-review 2a: EVERY changed or added line past the header block carries TILE; the pinned
    driver's lines are never deleted."""
    a = PINNED_DRIVER.read_text().splitlines()
    b = TILE_DRIVER.read_text().splitlines()
    header = b.index("set -uo pipefail")
    hunks = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert hunks
    for tag, i1, i2, j1, j2 in hunks:
        assert tag in ("replace", "insert") or j1 < header, (tag, a[i1:i2])
        for j in range(j1, j2):
            assert j < header or "TILE" in b[j], (j + 1, b[j])
    assert b[0] == a[0] and len(b) - len(a) < 120


def test_the_lambda_stop_instruction_stops_either_driver():
    text = (REPO / "docs" / "LAMBDA.md").read_text()
    assert "[g]h200_(model|tile)_session.sh" in text
    import re as _re
    pat = _re.search(r"pkill -TERM -f '(\[g\]h200[^']+)'", text).group(1)
    for name in ("gh200_model_session.sh --setup", "bash gh200_tile_session.sh --plan x"):
        assert _re.search(pat, name)


# --------------------------------------------------------------------------
# the units on the fake box
# --------------------------------------------------------------------------

FAKE_CENSUS = r'''
if tool == "tile_census":
    if "--dry-run" in args:
        print("TILE CENSUS: compile only (planted)")
        sys.exit(2)
    out = Path(val("--out")); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tiles": val("--tiles"), "models": val("--models"), "entries": [], "path_mismatch": []}))
    print("RESULT: VALIDITY census_path PASS 0 configs, 0 path mismatches, 0 errors")
    sys.exit(0)
'''


def fakepy_tile() -> str:
    body = T5.fakepy5()
    for old, new in (('if tool == "dram_counter_route":', 'if tool in ("dram_counter_route", "dram_counter_route_tile"):'),
                     ('if tool == "private_weight_reference":', 'if tool in ("private_weight_reference", "r3_tile"):')):
        assert old in body
        body = body.replace(old, new)
    head, tail = body.rsplit("sys.exit(0)", 1)
    return head + FAKE_CENSUS + "sys.exit(0)" + tail


PLAN_BOX = f"""- prelude
mixtral-8x7b calibrate label=ta nodrop=1 est=2 cap=5
- tilecensus label=ta tiles=128x128x64w8s3,64x128x64w8s3 models=olmoe-1b-7b,phi-3.5-moe nodrop=1 est=2 cap=5
olmoe-1b-7b bytes label=f64w4 block-m=64 block-n=128 block-k=64 num-warps=4 num-stages=3 byte-groups=64 est=2 cap=5
olmoe-1b-7b timed label=e128g1 block-m=128 block-n=128 block-k=64 num-warps=8 num-stages=3 histogram={HA}/olmoe-1b-7b-128g1.json arms=native,shared declared-copies=9 timed-groups=1 slip-policy=cell g1-rule=r7-median-run3 est=2 cap=5
olmoe-1b-7b timed label=plain histogram=docs/registered/2026-10-09-rental6-skew-hist/olmoe-1b-7b-A.json arms=native,shared declared-copies=9 timed-groups=8 slip-policy=cell est=2 cap=5
"""


@pytest.fixture(scope="module")
def tilebox(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tilebox")
    box = TG.Box(tmp, {"retracted": 40})
    TG._exe(box.bin / "fakepy", fakepy_tile())
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp / "tile.plan"
    f.write_text(PLAN_BOX)
    got = subprocess.run(["bash", str(TILE_DRIVER), "--deadline", str(TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(), capture_output=True, text=True, timeout=600)
    return box, got


def test_the_tile_units_run_and_reach_their_tools_with_their_tiles(tilebox):
    box, got = tilebox
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    starts = re.findall(r"^\S+ (\w+) START", led, re.M)
    assert starts == ["prelude", "calibrate", "tilecensus", "bytes", "timed", "timed"]
    cen = box.tool("tile_census", dry=False)
    assert len(cen) == 1 and TG.val(cen[0], "--tiles") == "128x128x64w8s3,64x128x64w8s3"
    assert TG.val(cen[0], "--models") == "olmoe-1b-7b,phi-3.5-moe" and "tile-census-" in TG.val(cen[0], "--out")
    assert "tilecensus: RESULT: VALIDITY census_path PASS" in led
    dcr = [a for a in box.tool("dram_counter_route_tile") if "--run" in a]
    # the model's occupancy census (--census-only) stays the pinned route's: it does not key on the tile
    assert len(dcr) == 1 and not [a for a in box.tool("dram_counter_route") if "--run" in a and "--census-only" not in a]
    a = dcr[0]
    assert (TG.val(a, "--num-warps"), TG.val(a, "--block-m"), TG.val(a, "--block-n")) == ("4", "64", "128")
    assert (TG.val(a, "--block-k"), TG.val(a, "--num-stages"), TG.val(a, "--group-m")) == ("64", "3", "64")


def test_a_tile_timed_unit_runs_r3_through_the_wrapper_and_a_plain_unit_does_not(tilebox):
    box, _got = tilebox
    lr = box.tool("locked_r3")
    tile = [x for x in lr if f"{HA}/olmoe-1b-7b-128g1.json" in x]
    plain = [x for x in lr if "docs/registered/2026-10-09-rental6-skew-hist/olmoe-1b-7b-A.json" in x]
    assert len(tile) == 1 and len(plain) == 1
    i = tile[0].index("--")
    mine, r3 = tile[0][:i], tile[0][i + 1:]
    assert TG.val(mine, "--r3-script") == "scripts/r3_tile.py"
    assert (TG.val(r3, "--block-m"), TG.val(r3, "--block-n"), TG.val(r3, "--num-warps")) == ("128", "128", "8")
    assert (TG.val(r3, "--block-k"), TG.val(r3, "--num-stages")) == ("64", "3")
    j = plain[0].index("--")
    assert "--r3-script" not in plain[0][:j] and TG.val(plain[0][j + 1:], "--block-m") == "32"
    assert "--num-warps" not in plain[0][j + 1:]
    import locked_r3 as LR
    assert LR.build_parser().parse_args(mine).r3_script == "scripts/r3_tile.py"
    dry_t = box.tool("r3_tile", dry=True)
    dry_p = box.tool("private_weight_reference", dry=True)
    assert len(dry_t) == 1 and TG.val(dry_t[0], "--num-warps") == "8" and TG.val(dry_t[0], "--block-m") == "128"
    assert len(dry_p) == 1 and "--num-warps" not in dry_p[0]


def test_the_real_r3_dry_run_of_a_tile_unit_passes_the_drivers_checks(tmp_path):
    body = (f"- prelude\nmixtral-8x7b calibrate label=ta nodrop=1 est=2 cap=5\n"
            f"olmoe-1b-7b timed label=e128g1 block-m=128 block-n=128 block-k=64 num-warps=8 num-stages=3 "
            f"histogram={HA}/olmoe-1b-7b-128g1.json arms=native,shared declared-copies=9 timed-groups=1 "
            "slip-policy=cell g1-rule=r7-median-run3 nodrop=1 est=2 cap=5\n")
    box = TG.Box(tmp_path, {"calibrate": {"rc": 4, "results": []}})
    body_py = fakepy_tile().replace('if os.environ.get("FAKE_REAL_DRY"):\n        real("private_weight_reference.py")',
                                    'if os.environ.get("FAKE_REAL_DRY"):\n        real(tool + ".py")')
    TG._exe(box.bin / "fakepy", body_py)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "t.plan"
    f.write_text(body)
    subprocess.run(["bash", str(TILE_DRIVER), "--deadline", str(TG.FAR), "--plan", str(f), "--no-push"],
                   env=box.env(FAKE_REAL_DRY=1), capture_output=True, text=True, timeout=900)
    led = box.ledger()
    assert "may not lock" not in led, led
    logs = sorted((box.session / "logs").glob("r3-olmoe-1b-7b-e128g1-g1-t4-dry.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "'num_warps': 8" in text and "'BLOCK_SIZE_N': 128" in text and "BLOCK_M=128" in text, text[-3000:]
    assert "reason: --dry-run was given" in text and "REFUSED:" not in text


# --------------------------------------------------------------------------
# vm_run.sh start --driver gh200_tile_session.sh
# --------------------------------------------------------------------------

def test_start_with_rental_a_delivers_the_tile_driver_the_plan_and_every_page_byte_for_byte(tmp_path):
    plan_rel = f"scripts/plans/{PLAN.name}"
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN.read_text())))
    assert len(pages) == 6
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    shutil.copy(PLAN, lap.clone / plan_rel)
    shutil.copy(TILE_DRIVER, lap.clone / "scripts" / TILE_DRIVER.name)
    for p in pages:
        (lap.clone / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, lap.clone / p)
    g = ["git", "-C", str(lap.clone)]
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "plan"], check=True)
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-t"], check=True, capture_output=True)
    lap.make_branch()
    assert lap.run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
    real = lap.bin / "ssh.real"
    (lap.bin / "ssh").rename(real)
    TG._exe(lap.bin / "ssh", f'#!/bin/bash\ncat > /dev/null\nexec "{real}" "$@"\n')
    log = tmp_path / "scp.log"
    vm = tmp_path / "vm-home"
    vm.mkdir()
    TG._exe(lap.bin / "scp", (
        '#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"\n'
        f'printf "%s %s\\n" "$src" "$dst" >> "{log}"\n'
        '[[ "$dst" == *:* ]] || exit 0\n'
        f'rel="${{dst#*:}}"; [[ -n "$rel" ]] || rel="$(basename "$src")"\n'
        f'mkdir -p "$(dirname "{vm}/$rel")" && cp "$src" "{vm}/$rel"\nexit 0\n'))
    lap.run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", plan_rel,
            "--driver", TILE_DRIVER.name)
    sent = [ln.split(" ", 1) for ln in log.read_text().splitlines()] if log.exists() else []
    got = {(Path(src).resolve(), dst) for src, dst in sent}
    assert (lap.clone.resolve() / plan_rel, "ubuntu@1.2.3.4:") in got, sent
    assert (lap.clone.resolve() / "scripts" / TILE_DRIVER.name, "ubuntu@1.2.3.4:") in got, sent
    for p in pages:
        assert (lap.clone.resolve() / p, f"ubuntu@1.2.3.4:{p}") in got, (p, sent)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(vm / PLAN.name) == sha(PLAN) and sha(vm / TILE_DRIVER.name) == sha(TILE_DRIVER)
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "gh200_tile_session.sh --dry-run" in cmds and "gh200_tile_session.sh --setup" in cmds
    run_env = (lap.state / "r" / "run.env").read_text()
    assert "driver_script=gh200_tile_session.sh" in run_env


def test_start_refuses_a_driver_that_is_not_a_session_driver(tmp_path):
    lap = TG.Laptop(tmp_path)
    lap.make_branch()
    assert lap.run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
    got = lap.run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--driver", "evil.sh")
    assert got.returncode == exit_codes.REFUSED and "--driver evil.sh" in got.stdout


def test_the_pre_setup_dry_run_prints_rental_a_from_the_files_vm_run_copies(tmp_path):
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN.read_text())))
    home = tmp_path / "home"
    home.mkdir()
    shutil.copy(TILE_DRIVER, home / TILE_DRIVER.name)
    shutil.copy(PLAN, home / PLAN.name)
    for p in pages:
        (home / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, home / p)
    env = {**os.environ, "HOME": str(home), "MOE_HOME": str(home / "moe")}
    got = subprocess.run(["bash", str(home / TILE_DRIVER.name), "--dry-run", "--plan", f"scripts/plans/{PLAN.name}"],
                         capture_output=True, text=True, timeout=60, cwd=home, env=env)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 21 units" in got.stdout, got.stdout[-2000:] + got.stderr[-2000:]
    assert got.stdout.startswith("THE GH200 MODEL-TEST SESSION")
