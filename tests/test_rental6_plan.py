"""Rental 6's plan and driver keys off the GPU (scripts/plans/rental6-2026-10.plan): the
parse-only dry run and its budget, slip-policy= and nodrop= (design-r6-review (d) and (g)),
the no-drop guard on the fake box, and `vm_run.sh start` with the real plan against a stub
VM whose ssh drains stdin, so every file the plan names must still arrive (rental 5 lost two
starts to that class of bug)."""
from __future__ import annotations

import hashlib
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

PLAN6 = REPO / "scripts" / "plans" / "rental6-2026-10.plan"
DRIVER = TG.DRIVER
H6 = "docs/registered/2026-10-09-rental6-skew-hist"
SKEW = [(m, lab) for m in ("mixtral-8x7b", "olmoe-1b-7b", "phi-3.5-moe") for lab in ("ska", "skb", "skc")]


def dry(*args):
    return T5.dry(*args)


def _label(u):
    m = re.match(r"label=(\S+)", u[5])
    return m.group(1) if m else None


def _group(u):
    m = re.search(r"\[drop-group ([\w-]+)\]", u[5])
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# the plan, parse only
# --------------------------------------------------------------------------

def test_the_rental6_plan_parses_fits_three_hours_and_never_drops_a_skew_page():
    got = dry("--plan", str(PLAN6))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    assert f"a PLAN of {len(units)} units" in got.stdout
    total = sum(int(e) for _, _m, _s, e, *_r in units)
    assert f"{total} min of units" in got.stdout and total <= 180
    assert [u[2] for u in units[:2]] == ["prelude", "calibrate"] and "[nodrop]" in units[1][5]
    skew = units[2:11]
    assert [(u[1], _label(u)) for u in skew] == SKEW
    for u in skew:
        # never dropped, and in no drop-group, so one page running short drops no block
        assert u[2] == "timed" and "[nodrop]" in u[5] and _group(u) is None, u
        assert "slip-policy=cell" in u[5] and "declared-copies=9" in u[5] and "arms=native,shared" in u[5]
    rest = units[11:]
    assert all("[nodrop]" not in u[5] and _group(u) for u in rest), rest
    # the drop order, bottom up: k2, q1j, byt, q1p, q1q (the byte group no longer shares a page label)
    seen = []
    for u in reversed(rest):
        if _group(u) not in seen:
            seen.append(_group(u))
    assert seen == ["k2", "q1j", "byt", "q1p", "q1q"]
    assert "byt" not in {_label(u) for u in units}
    for u in units:
        if u[2] == "timed":
            assert "slip-policy=cell" in u[5], u
        if "histogram=" in u[5]:
            f = re.search(r"histogram=(\S+)", u[5]).group(1)
            assert f.startswith(H6) and (REPO / f).is_file()
    # Mixtral's byte page is dropped (review (c)); OLMoE's is kept
    bytes_units = [(u[1], _label(u)) for u in units if u[2] == "bytes"]
    assert ("olmoe-1b-7b", "skbytes") in bytes_units and ("mixtral-8x7b", "skbytes") not in bytes_units
    order = re.search(r"drop order when time runs short: .*\n\s+([\d ]+)", got.stdout).group(1).split()
    assert not {str(i) for i in range(1, 12)} & set(order), "a nodrop unit is in the drop order"
    dirs = [TG._unit_dir(u[5].replace(" [nodrop]", "")) for u in units if u[2] not in ("prelude", "calibrate")]
    assert len(set(dirs)) == len(dirs)


def test_the_plan_names_no_file_but_its_histogram_pages():
    """vm_run.sh copies the plan and each histogram= page and nothing else before setup: a
    plan key naming another file would arrive missing on the VM."""
    text = PLAN6.read_text()
    body = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    files = set(re.findall(r"=(\S+\.(?:json|yaml|txt|csv|plan))", body))
    assert files and files == set(re.findall(r"histogram=(\S+)", body))


@pytest.mark.parametrize("body, why", [
    ("- prelude\nolmoe-1b-7b bytes label=b byte-groups=8 nodrop=2\n", "nodrop 2: 1 or leave it out"),
    ("- prelude nodrop=1\nolmoe-1b-7b bytes label=b byte-groups=8\n", "nodrop on the prelude"),
    ("- prelude\nolmoe-1b-7b bytes label=b byte-groups=8 nodrop=1 drop-group=x\n", "nodrop with drop-group= or depends="),
    ("- prelude\nolmoe-1b-7b bytes label=a byte-groups=8 drop-group=x\nolmoe-1b-7b bytes label=b byte-groups=8 nodrop=1 depends=x\n",
     "nodrop with drop-group= or depends="),
    (f"- prelude\nolmoe-1b-7b bytes label=b byte-groups=8 histogram={H6}/olmoe-1b-7b-bytes.json slip-policy=cell\n",
     "slip-policy belongs to a timed unit"),
    ("- prelude\nolmoe-1b-7b timed label=a slip-policy=cell\n", "slip-policy=cell needs histogram="),
    (f"- prelude\nolmoe-1b-7b timed label=a histogram={H6}/olmoe-1b-7b-A.json declared-copies=9 slip-policy=row\n",
     "slip-policy row: page or cell"),
])
def test_rental6_keys_are_refused_where_they_cannot_run(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


# --------------------------------------------------------------------------
# the no-drop guard and the slip policy on the fake box
# --------------------------------------------------------------------------

GUARD = """- prelude
olmoe-1b-7b bytes label=s1 byte-groups=1 nodrop=1 est=6 cap=20
olmoe-1b-7b bytes label=s2 byte-groups=1 nodrop=1 est=6 cap=20
olmoe-1b-7b bytes label=s3 byte-groups=1 nodrop=1 est=6 cap=20
mixtral-8x7b bytes label=q1 byte-groups=1 drop-group=q est=3 cap=20
mixtral-8x7b bytes label=q2 byte-groups=1 drop-group=q est=3 cap=20
mixtral-8x7b bytes label=k byte-groups=1 est=3 cap=20
"""
NOW = 1_900_000_000


def test_a_short_deadline_drops_the_droppable_tail_and_never_a_nodrop_unit(tmp_path):
    # 33 min less the 8-minute reserve: 25 against 30 planned. Unit 7 goes (27 left), then
    # unit 6 with its group q (unit 5): 21 fits; the three nodrop units all run
    box, _got = TG._plan_box(tmp_path, GUARD, deadline=NOW + 33 * 60, MOE_DRIVER_NOW=NOW)
    led = box.ledger()
    assert re.findall(r"DROPPED unit (\d+)/", led) == ["7", "6", "5"], led
    assert "NODROP" not in led
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "bytes", "bytes", "bytes"]


def test_with_only_nodrop_units_left_over_the_deadline_none_is_dropped_and_the_ledger_says_so(tmp_path):
    # 23 min less the reserve: 15 against 30. Everything droppable goes and 21 still overruns;
    # the nodrop units are not dropped, and each runs while its own 6 min fits the 15 left
    box, _got = TG._plan_box(tmp_path, GUARD, deadline=NOW + 23 * 60, MOE_DRIVER_NOW=NOW)
    led = box.ledger()
    assert sorted(re.findall(r"DROPPED unit (\d+)/", led)) == ["5", "6", "7"], led
    assert "NODROP: " in led and "only nodrop=1 units" in led
    assert not re.search(r"DROPPED unit [234]/", led)
    assert re.findall(r"^\S+ (\w+) START", led, re.M) == ["prelude", "bytes", "bytes", "bytes"]


def test_one_skew_page_running_short_drops_no_other_skew_page(tmp_path):
    """Rental 5's plan put a model's pages in one drop-group, so a shortfall of one page dropped
    the block. Rental 6's skew pages carry no group: with nothing droppable after them they all
    stay planned whatever the deadline."""
    body = "- prelude\n" + "".join(f"olmoe-1b-7b bytes label=s{i} byte-groups=1 nodrop=1 est=6 cap=20\n" for i in range(4))
    box, _got = TG._plan_box(tmp_path, body, deadline=NOW + 20 * 60, MOE_DRIVER_NOW=NOW)
    led = box.ledger()
    assert "DROPPED" not in led and "NODROP: " in led


PLAN_SLIP = f"""- prelude
mixtral-8x7b calibrate label=r6 nodrop=1 est=2 cap=5
olmoe-1b-7b timed label=ska histogram={H6}/olmoe-1b-7b-A.json arms=native,shared declared-copies=9 timed-groups=8 slip-policy=cell nodrop=1 est=2 cap=5
olmoe-1b-7b timed label=q9 histogram={H6}/olmoe-1b-7b-B.json arms=native,shared declared-copies=9 timed-groups=8 drop-group=q est=2 cap=5
"""


def test_slip_policy_cell_reaches_locked_r3_and_page_policy_adds_nothing(tmp_path):
    box, got = T5.box5(tmp_path, PLAN_SLIP, {"retracted": 40})
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout + got.stderr + led
    lr = box.tool("locked_r3")
    ska = [a for a in lr if f"{H6}/olmoe-1b-7b-A.json" in a]
    q9 = [a for a in lr if f"{H6}/olmoe-1b-7b-B.json" in a]
    assert len(ska) == 1 and len(q9) == 1
    mine = ska[0][:ska[0].index("--")]
    assert mine[mine.index("--slip-policy") + 1] == "cell"
    assert "--slip-policy" not in q9[0] and "--lock-mhz" not in q9[0]
    import locked_r3 as LR
    for a in (ska[0], q9[0]):
        LR.build_parser().parse_args(a[:a.index("--")])
    assert "slip-policy cell" in led


# --------------------------------------------------------------------------
# vm_run.sh start with the real plan, ssh draining stdin
# --------------------------------------------------------------------------

def test_start_with_the_rental6_plan_copies_the_plan_and_every_page_it_names(tmp_path):
    plan_rel = f"scripts/plans/{PLAN6.name}"
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN6.read_text())))
    assert len(pages) == 14
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    shutil.copy(PLAN6, lap.clone / plan_rel)
    for p in pages:
        (lap.clone / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, lap.clone / p)
    g = ["git", "-C", str(lap.clone)]
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "plan"], check=True)
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-t"], check=True, capture_output=True)
    lap.make_branch()
    assert lap.run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
    # the stub ssh reads all of stdin before it runs, as the real one does
    real = lap.bin / "ssh.real"
    (lap.bin / "ssh").rename(real)
    TG._exe(lap.bin / "ssh", f'#!/bin/bash\ncat > /dev/null\nexec "{real}" "$@"\n')
    log = tmp_path / "scp.log"
    vm = tmp_path / "vm-home"
    vm.mkdir()
    # the stub scp DELIVERS each file into the stub VM's home (ubuntu@ip:REL, or the home with
    # the source's name for ubuntu@ip:), so what arrives can be compared byte for byte
    TG._exe(lap.bin / "scp", (
        '#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"\n'
        f'printf "%s %s\\n" "$src" "$dst" >> "{log}"\n'
        '[[ "$dst" == *:* ]] || exit 0\n'
        f'rel="${{dst#*:}}"; [[ -n "$rel" ]] || rel="$(basename "$src")"\n'
        f'mkdir -p "$(dirname "{vm}/$rel")" && cp "$src" "{vm}/$rel"\nexit 0\n'))
    lap.run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", plan_rel)
    sent = [ln.split(" ", 1) for ln in log.read_text().splitlines()] if log.exists() else []
    got = {(Path(src).resolve(), dst) for src, dst in sent}
    assert (lap.clone.resolve() / plan_rel, "ubuntu@1.2.3.4:") in got, sent
    for p in pages:
        assert (lap.clone.resolve() / p, f"ubuntu@1.2.3.4:{p}") in got, (p, sent)
    mk = [c for c in lap.vm_cmds() if "mkdir -p" in c]
    assert any(H6 in c for c in mk), mk

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    # every delivered file is the repo's, byte for byte (not only the copy command)
    assert sha(vm / PLAN6.name) == sha(PLAN6)
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p


def test_the_pre_setup_dry_run_prints_the_rental6_plan_from_the_files_vm_run_copies(tmp_path):
    """In the VM's home before setup there are only the driver, the plan and the pages
    vm_run.sh copied: with exactly those the plan prints."""
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN6.read_text())))
    home = tmp_path / "home"
    home.mkdir()
    shutil.copy(DRIVER, home / DRIVER.name)
    shutil.copy(PLAN6, home / PLAN6.name)
    for p in pages:
        (home / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, home / p)
    env = {**os.environ, "HOME": str(home), "MOE_HOME": str(home / "moe")}
    got = subprocess.run(["bash", str(home / DRIVER.name), "--dry-run", "--plan", f"scripts/plans/{PLAN6.name}"],
                         capture_output=True, text=True, timeout=60, cwd=home, env=env)
    assert got.returncode == exit_codes.REFUSED and "THE GH200 MODEL-TEST SESSION" in got.stdout, (
        got.stdout[-2000:] + got.stderr[-2000:])
