"""Rental 7's plan and driver keys off the GPU (scripts/plans/rental7-2026-10.plan): the parse-only
dry run, its 99 minutes and its drop order (design-r7-review (g)), the g1-rule key (rental7-g1rule)
and where it is refused, the per-unit log names on the fake box (design-r7 S9, review (f)), and
`vm_run.sh start` with the real plan against a stub VM whose ssh drains stdin, every delivered file
compared by sha256."""
from __future__ import annotations

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

PLAN7 = REPO / "scripts" / "plans" / "rental7-2026-10.plan"
DRIVER = TG.DRIVER
H7 = "docs/registered/2026-10-13-rental7-skew-hist"
H6 = "docs/registered/2026-10-09-rental6-skew-hist"
SKEW = [(m, lab) for m in ("mixtral-8x7b", "phi-3.5-moe") for lab in ("ska", "skb", "skc")]
Q1 = [(m, lab) for m in ("qwen1.5-moe-a2.7b", "phi-3.5-moe", "jetmoe-8b") for lab in ("q9", "q15")]


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

def test_the_rental7_plan_parses_in_99_minutes_and_never_drops_a_skew_page():
    got = dry("--plan", str(PLAN7))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    units = TG._units(got.stdout)
    assert f"a PLAN of {len(units)} units" in got.stdout and len(units) == 15
    total = sum(int(e) for _, _m, _s, e, *_r in units)
    assert total == 99 and "99 min of units" in got.stdout and total <= 120
    assert [u[2] for u in units[:2]] == ["prelude", "calibrate"] and "[nodrop]" in units[1][5]
    skew = units[2:8]
    assert [(u[1], _label(u)) for u in skew] == SKEW
    for u in skew:
        assert u[2] == "timed" and "[nodrop]" in u[5] and _group(u) is None, u
        assert "slip-policy=cell" in u[5] and "g1-rule=r7-median-run3" in u[5] and "declared-copies=9" in u[5]
        assert f"histogram={H7}/" in u[5]
    q1 = units[8:14]
    assert [(u[1], _label(u)) for u in q1] == Q1
    for u in q1:
        assert u[2] == "timed" and "g1-rule=r7-median-run3" in u[5] and f"histogram={H6}/q1-" in u[5]
        assert ("declared-copies=15" in u[5]) == (_label(u) == "q15")
    assert (units[14][1], units[14][2], _label(units[14])) == ("olmoe-1b-7b", "bytes", "l2open")
    assert f"histogram={H7}/olmoe-1b-7b-l2open.json" in units[14][5] and "g1-rule" not in units[14][5]
    # the drop order, last first: byt, q1j, q1p, q1q (review (g)); no nodrop unit in it
    seen = []
    for u in reversed(units[8:]):
        if _group(u) not in seen:
            seen.append(_group(u))
    assert seen == ["byt", "q1j", "q1p", "q1q"]
    assert not {_group(u) for u in units if _group(u)} & {_label(u) for u in units}
    order = re.search(r"drop order when time runs short: .*\n\s+([\d ]+)", got.stdout).group(1).split()
    assert order == [str(i) for i in range(15, 8, -1)]
    assert sum(int(u[3]) for u in units if _group(u) != "byt") == 95
    dirs = [TG._unit_dir(u[5].replace(" [nodrop]", "")) for u in units if u[2] not in ("prelude", "calibrate")]
    assert len(set(dirs)) == len(dirs)


def test_every_histogram_the_plan_names_is_registered_and_the_plan_names_no_other_file():
    text = PLAN7.read_text()
    body = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    files = set(re.findall(r"=(\S+\.(?:json|yaml|txt|csv|plan))", body))
    assert files and files == set(re.findall(r"histogram=(\S+)", body)) and len(files) == 10
    skew = json.loads((REPO / "docs/registered/2026-10-13-rental7-skew-gh200.json").read_text())["histograms"]["files"]
    q1 = json.loads((REPO / "docs/registered/2026-10-13-rental7-q1-gh200.json").read_text())["pages"]["files"]
    l2 = json.loads((REPO / "docs/registered/2026-10-13-rental7-l2open-gh200.json").read_text())["page"]["file"]
    reg = {v["path"]: v["sha256"] for d in (skew, q1, l2) for v in d.values()}
    for f in files:
        assert f in reg, f
        assert hashlib.sha256((REPO / f).read_bytes()).hexdigest() == reg[f], f


@pytest.mark.parametrize("body, why", [
    (f"- prelude\nolmoe-1b-7b bytes label=b byte-groups=8 histogram={H6}/olmoe-1b-7b-bytes.json g1-rule=r7-median-run3\n",
     "g1-rule belongs to a timed unit, not bytes"),
    ("- prelude\nolmoe-1b-7b timed label=a g1-rule=r7-median-run3\n", "g1-rule needs histogram="),
    (f"- prelude\nolmoe-1b-7b timed label=a histogram={H6}/olmoe-1b-7b-A.json declared-copies=9 g1-rule=r7-median-run3\n",
     "g1-rule=r7-median-run3 needs slip-policy=cell"),
    (f"- prelude\nolmoe-1b-7b timed label=a histogram={H6}/olmoe-1b-7b-A.json declared-copies=9 slip-policy=page "
     "g1-rule=r7-median-run3\n", "g1-rule=r7-median-run3 needs slip-policy=cell"),
    (f"- prelude\nolmoe-1b-7b timed label=a histogram={H6}/olmoe-1b-7b-A.json declared-copies=9 slip-policy=cell "
     "g1-rule=r7-share20\n", "g1-rule r7-share20: r5-any or r7-median-run3"),
])
def test_g1_rule_is_refused_where_it_cannot_run(tmp_path, body, why):
    f = tmp_path / "p.plan"
    f.write_text(body)
    got = dry("--plan", str(f))
    assert got.returncode == exit_codes.REFUSED and why in got.stderr, got.stderr


# --------------------------------------------------------------------------
# g1-rule on the fake box
# --------------------------------------------------------------------------

PLAN_G1 = f"""- prelude
mixtral-8x7b calibrate label=r7 nodrop=1 est=2 cap=5
olmoe-1b-7b timed label=ska histogram={H6}/olmoe-1b-7b-A.json arms=native,shared declared-copies=9 timed-groups=8 slip-policy=cell g1-rule=r7-median-run3 nodrop=1 est=2 cap=5
olmoe-1b-7b timed label=skb histogram={H6}/olmoe-1b-7b-B.json arms=native,shared declared-copies=9 timed-groups=8 slip-policy=cell nodrop=1 est=2 cap=5
"""


def test_g1_rule_hands_locked_r3_min_clean_zero_and_r3_gets_no_new_flag(tmp_path):
    """The build review's fix: R3 and locked_r3 stay at their pins; g1-rule=r7-median-run3 hands
    locked_r3 --min-clean-repeats 0 (so R3 times every repeat) and the dry run the slip arguments
    locked_r3 then adds; a unit without the key is rental 6's, min-clean 6 by default."""
    box, got = T5.box5(tmp_path, PLAN_G1, {"retracted": 40})
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    lr = box.tool("locked_r3")
    ska = [a for a in lr if f"{H6}/olmoe-1b-7b-A.json" in a]
    skb = [a for a in lr if f"{H6}/olmoe-1b-7b-B.json" in a]
    assert len(ska) == 1 and len(skb) == 1
    i = ska[0].index("--")
    mine, r3 = ska[0][:i], ska[0][i + 1:]
    assert TG.val(mine, "--min-clean-repeats") == "0" and TG.val(mine, "--slip-policy") == "cell"
    assert not [t for t in r3 if t.startswith("--g1") or t == "--min-clean-repeats"]
    assert "--min-clean-repeats" not in skb[0][:skb[0].index("--")]
    import locked_r3 as LR
    import private_weight_reference as PW
    for a in (ska[0], skb[0]):
        j = a.index("--")
        assert LR.build_parser().parse_args(a[:j]).min_clean_repeats == (0 if a is ska[0] else 6)
        assert not [t for t in a[j + 1:] if LR.owned_flag(t)]
    dr = box.tool("private_weight_reference", dry=True)
    d_ska = [a for a in dr if f"{H6}/olmoe-1b-7b-A.json" in a]
    d_skb = [a for a in dr if f"{H6}/olmoe-1b-7b-B.json" in a]
    assert len(d_ska) == 1 and len(d_skb) == 1
    assert TG.val(d_ska[0], "--slip-policy") == "cell" and TG.val(d_ska[0], "--lock-mhz") == "1710"
    assert TG.val(d_ska[0], "--min-clean-repeats") == "0"
    assert "--slip-policy" not in d_skb[0] and "--min-clean-repeats" not in d_skb[0]
    PW.build_parser().parse_args(d_ska[0])
    assert "scorer G1 rule r7-median-run3 (locked_r3 --min-clean-repeats 0)" in led


def test_the_rental7_plans_real_r3_dry_runs_pass_at_min_clean_zero(tmp_path):
    body = (f"- prelude\nmixtral-8x7b calibrate label=r7 nodrop=1 est=2 cap=5\n"
            f"phi-3.5-moe timed label=ska histogram={H7}/phi-3.5-moe-A.json arms=native,shared declared-copies=9 "
            "timed-groups=8 slip-policy=cell g1-rule=r7-median-run3 nodrop=1 est=2 cap=5\n")
    box, got = T5.box5(tmp_path, body, {"calibrate": {"rc": 4, "results": []}}, FAKE_REAL_DRY=1, timeout=900)
    led = box.ledger()
    assert "may not lock" not in led, led
    logs = sorted((box.session / "logs").glob("r3-phi-3.5-moe-ska-g8-t12-dry.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "slips       --slip-policy cell at the 1710 MHz lock" in text and "no minimum clean count" in text, text[-3000:]
    assert "reason: --dry-run was given" in text and "REFUSED:" not in text


# --------------------------------------------------------------------------
# S9: every per-unit log named by model and label (the fake box)
# --------------------------------------------------------------------------

PLAN_LOGS = f"""- prelude
mixtral-8x7b calibrate label=r7 nodrop=1 est=2 cap=5
- rulers label=ra est=3 cap=10
- rulers label=rb est=3 cap=10
- gpubench label=g7 est=3 cap=10
olmoe-1b-7b timed label=ska histogram={H6}/olmoe-1b-7b-A.json arms=native,shared declared-copies=9 timed-groups=8 slip-policy=cell nodrop=1 est=2 cap=5
phi-3.5-moe timed label=ska histogram={H6}/phi-3.5-moe-A.json arms=native,shared declared-copies=9 timed-groups=8 slip-policy=cell nodrop=1 est=2 cap=5
olmoe-1b-7b bytes label=skbytes histogram={H6}/olmoe-1b-7b-bytes.json arms=native,shared byte-groups=8 est=2 cap=5
phi-3.5-moe bytes label=skbytes histogram={H6}/phi-3.5-moe-bytes.json arms=native,shared byte-groups=8 est=2 cap=5
"""


def test_units_sharing_a_tread_a_label_or_a_g_write_distinct_logs_and_none_is_overwritten(tmp_path):
    """Rental 6's collisions, replayed: OLMoE and Phi timed units with deepest tread 12 and label
    ska (one dry-run log on rental 6), two byte units at G 8 (one r3c log, Phi's alone survived),
    and two rulers units (one dry-run log). Each now has its own file, named by model and label,
    and each holds its own unit's lines only."""
    box, got = T5.box5(tmp_path, PLAN_LOGS, {"retracted": 40})
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    logs = box.session / "logs"
    names = sorted(p.name for p in logs.iterdir())
    for want in ("r3-olmoe-1b-7b-ska-g8-t12-dry.log", "r3-phi-3.5-moe-ska-g8-t12-dry.log",
                 "r3c-olmoe-1b-7b-skbytes-g8-lock1710.log", "r3c-phi-3.5-moe-skbytes-g8-lock1710.log",
                 "analyse-olmoe-1b-7b-skbytes-lock1710.log", "analyse-phi-3.5-moe-skbytes-lock1710.log",
                 "lock1710-pages-olmoe-1b-7b-skbytes.status", "lock1710-pages-phi-3.5-moe-skbytes.status",
                 "rulers-mixtral-8x7b-ra-dry.log", "rulers-mixtral-8x7b-rb-dry.log",
                 "rulers-mixtral-8x7b-ra.log", "rulers-mixtral-8x7b-rb.log",
                 "gpubench-mixtral-8x7b-g7-dry.log", "gpubench-mixtral-8x7b-g7.log",
                 "calibrate-mixtral-8x7b-r7.log",
                 "driver-timed-olmoe-1b-7b-ska.log", "driver-timed-phi-3.5-moe-ska.log",
                 "driver-bytes-olmoe-1b-7b-skbytes.log", "driver-bytes-phi-3.5-moe-skbytes.log",
                 "driver-rulers-mixtral-8x7b-ra.log", "driver-rulers-mixtral-8x7b-rb.log"):
        assert want in names, (want, names)
    # the old shared names are gone
    for old in ("r3-g8-t12-dry-ska.log", "r3c-g8-lock1710.log", "rulers-dry.log", "gpubench-dry.log",
                "analyse-lock1710.log", "lock1710-pages.status", "calibrate.log", "driver-timed.log", "driver-bytes.log"):
        assert old not in names, old
    # each byte unit's status holds its own page's one line, and its driver log names its own model only
    for m in ("olmoe-1b-7b", "phi-3.5-moe"):
        assert len((logs / f"lock1710-pages-{m}-skbytes.status").read_text().splitlines()) == 1
        other = "phi-3.5-moe" if m == "olmoe-1b-7b" else "olmoe-1b-7b"
        text = (logs / f"driver-bytes-{m}-skbytes.log").read_text()
        assert m in text and other not in text, m
        text = (logs / f"driver-timed-{m}-ska.log").read_text()
        assert m in text and other not in text, m
    # every log under logs/ carries a model name (the prelude's included)
    models = ("mixtral-8x7b", "olmoe-1b-7b", "phi-3.5-moe")
    assert all(any(m in n for m in models) for n in names if n.endswith((".log", ".status"))), names
    # the dry runs the driver checked are the files it wrote, one per unit and G
    dry = box.tool("private_weight_reference", dry=True)
    assert len(dry) == 2 and len([n for n in names if n.startswith("r3-") and n.endswith("-dry.log")]) == 2


def test_the_floor_reads_its_replay_passes_from_its_own_log(tmp_path):
    """r3f-g64.log (driver line 1048, review (f)) is read back for the replay passes: the floor
    capture's log now carries the unit's model and label, and the read is the same file."""
    body = "- prelude\nmixtral-8x7b floor label=fl est=5 cap=20\n"
    box, got = TG._plan_box(tmp_path, body, {"floor:r3f-g64": {"passes": 70}})
    led = box.ledger()
    names = sorted(p.name for p in (box.session / "logs").iterdir())
    assert "r3f-g64-mixtral-8x7b-fl.log" in names and "r3f-g64.log" not in names, names
    res = (box.session / "gh200-driver" / "floor-results")
    text = res.read_text() if res.exists() else led
    assert "r3f-g2 dropped: the G=64 capture took 70 replay passes" in text + led, text + led


# --------------------------------------------------------------------------
# vm_run.sh start with the real plan, ssh draining stdin
# --------------------------------------------------------------------------

def test_start_with_the_rental7_plan_delivers_the_plan_and_every_page_byte_for_byte(tmp_path):
    plan_rel = f"scripts/plans/{PLAN7.name}"
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN7.read_text())))
    assert len(pages) == 10
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    shutil.copy(PLAN7, lap.clone / plan_rel)
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
    lap.run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", plan_rel)
    sent = [ln.split(" ", 1) for ln in log.read_text().splitlines()] if log.exists() else []
    got = {(Path(src).resolve(), dst) for src, dst in sent}
    assert (lap.clone.resolve() / plan_rel, "ubuntu@1.2.3.4:") in got, sent
    for p in pages:
        assert (lap.clone.resolve() / p, f"ubuntu@1.2.3.4:{p}") in got, (p, sent)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(vm / PLAN7.name) == sha(PLAN7)
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p


def test_the_pre_setup_dry_run_prints_the_rental7_plan_from_the_files_vm_run_copies(tmp_path):
    pages = sorted(set(re.findall(r"histogram=(\S+)", PLAN7.read_text())))
    home = tmp_path / "home"
    home.mkdir()
    shutil.copy(DRIVER, home / DRIVER.name)
    shutil.copy(PLAN7, home / PLAN7.name)
    for p in pages:
        (home / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, home / p)
    env = {**os.environ, "HOME": str(home), "MOE_HOME": str(home / "moe")}
    got = subprocess.run(["bash", str(home / DRIVER.name), "--dry-run", "--plan", f"scripts/plans/{PLAN7.name}"],
                         capture_output=True, text=True, timeout=60, cwd=home, env=env)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 15 units" in got.stdout, got.stdout[-2000:] + got.stderr[-2000:]
