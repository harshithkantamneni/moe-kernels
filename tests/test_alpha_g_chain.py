"""scripts/alpha_g_chain.sh, checked without a pod.

The chain sequences the alpha(G) matrix session: preflight, the driver's
preconditions, the clock elasticity at every G, the ratio at every G x seed at
--duty 0.5 with later seeds scored with the earlier ones. What this file pins:
the three shell habits this project has been burned by, the ledger's second
opinion (the driver's rule, lifted), pairing only with reports that formed a
ratio, the elasticity band read through the arm's own `band_of`, and a laptop
dry run that prices every step off the arms' own plans and writes nothing into
the tree.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scripts" / "alpha_g_chain.sh"
HELPERS = ROOT / "scripts" / "alpha_g_chain_helpers.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import alpha_g_chain_helpers as H  # noqa: E402
from _hermetic import laptop_env  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

CODE = CHAIN.read_text()


def lift(script: str, **variables) -> subprocess.CompletedProcess:
    """Evaluate the chain's LIFTABLE block, then `script`, the driver's way."""
    setup = "\n".join(f"{k}={v!r}" for k, v in variables.items())
    body = (f"set -uo pipefail\n{setup}\n"
            f'eval "$(sed -n \'/^# >>> LIFTABLE/,/^# <<< LIFTABLE/p\' "{CHAIN}")"\n'
            f"{script}\n")
    return subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          timeout=300, env=laptop_env(REPO=str(ROOT)))


# --------------------------------------------------------------------------
# the shell itself
# --------------------------------------------------------------------------

def test_the_chain_parses_and_avoids_the_three_habits():
    assert subprocess.run(["bash", "-n", str(CHAIN)], capture_output=True).returncode == 0
    assert re.search(r"^set -uo pipefail$", CODE, re.M)
    assert not re.search(r"^set -[a-z]*e", CODE, re.M), "set -e would abort a rented session"
    assert "errexit" not in CODE
    assert "$!" not in CODE, "a PID variable in a shell that starts no background job"
    body = CODE.split("\nrun_step() {", 1)[1].split("\n}", 1)[0]
    assert '"$@" > "$log" 2>&1 || rc=$?' in body, "the measured command's rc is captured directly"
    assert "| tee" not in body


def test_the_state_word_mirrors_the_exit_code_table():
    for rc, word in exit_codes.CODE_NAMES.items():
        got = lift(f"state_word {rc}")
        assert got.stdout.strip() == word, (rc, got.stdout)
    assert lift("state_word 9").stdout.strip() == "UNKNOWN"


@pytest.mark.parametrize("rc,implied,state,needle", [
    (0, "0", "DONE", "log agrees"),
    (1, "1", "CLAIM_FAIL", "log agrees"),
    (3, "3", "INVALID", "log agrees"),
    (0, "1", "UNKNOWN", "DEFECT: page implies 1"),
    (0, "NONE", "UNKNOWN", "UNEARNED DONE"),
    (2, "NONE", "REFUSED", ""),
    (3, "NONE", "UNKNOWN", "UNEARNED INVALID"),
    (4, "NONE", "ERROR", "crash"),
    (0, "UNREADABLE", "UNKNOWN", "SECOND OPINION UNAVAILABLE"),
])
def test_the_second_opinion_is_the_drivers_rule(rc, implied, state, needle):
    got = lift(f"state_for {rc} {implied}")
    word, _, note = got.stdout.rstrip("\n").partition("\t")
    assert word == state, got.stdout
    assert needle in note


def test_a_step_is_latched_only_on_a_result_state(tmp_path):
    ledger = tmp_path / "CHAIN.tsv"
    ledger.write_text("step\tstate\trc\tseconds\tdirty\tlog\tnote\n"
                      "r1-g1\tREFUSED\t2\t3\t0\tx\t\n"
                      "r1-g1\tDONE\t0\t900\t0\tx\tlog agrees\n"
                      "r3-g1-s0\tUNKNOWN\t0\t200\t0\tx\tUNEARNED DONE\n"
                      "r3-g4-s0\tCLAIM_FAIL\t1\t200\t0\tx\t\n")
    assert lift(f"latched r1-g1 {ledger!s} && echo yes").stdout.strip() == "yes"
    assert lift(f"latched r3-g4-s0 {ledger!s} && echo yes").stdout.strip() == "yes"
    assert lift(f"latched r3-g1-s0 {ledger!s} || echo no").stdout.strip() == "no"
    assert lift(f"latched never-ran {ledger!s} || echo no").stdout.strip() == "no"


def test_run_step_writes_the_row_with_the_second_opinion_taken(tmp_path):
    ledger = tmp_path / "CHAIN.tsv"
    ledger.write_text("step\tstate\trc\tseconds\tdirty\tlog\tnote\n")
    page = tmp_path / "page.py"
    page.write_text("import sys\n"
                    "print('RESULT: CLAIM C1 FAIL [CLAIM] x | measured 0.9 | gate y')\n"
                    "sys.exit(1)\n")
    got = lift(f"run_step r3-g1-s0 {tmp_path / 'a.log'!s} {sys.executable} {page!s}; echo rc=$?",
               LEDGER=str(ledger))
    assert "rc=1" in got.stdout, got.stdout + got.stderr
    rows = ledger.read_text().splitlines()[1:]
    assert len(rows) == 1
    name, state, rc, _secs, _dirty, _log, note = rows[0].split("\t")
    assert (name, state, rc) == ("r3-g1-s0", "CLAIM_FAIL", "1")
    assert "log agrees" in note
    # exit 0 with no RESULT line is not a DONE anybody earned
    silent = tmp_path / "silent.py"
    silent.write_text("print('nothing scored')\n")
    lift(f"run_step preflight-r1 {tmp_path / 'b.log'!s} {sys.executable} {silent!s}",
         LEDGER=str(ledger))
    last = ledger.read_text().splitlines()[-1].split("\t")
    assert last[1] == "UNKNOWN" and "UNEARNED DONE" in last[6]


# --------------------------------------------------------------------------
# the helpers: every read of a report, a log or a ledger
# --------------------------------------------------------------------------

def _report(tmp_path, name, *, G=1, seed=0, ratio=0.9551, lo=0.9544, hi=0.9695,
            synthetic=False, experiment="private_weight_reference", duty=0.5):
    payload = {"experiment": experiment, "synthetic": synthetic,
               "pinned": {"GROUP_SIZE_M": G}, "seed": seed, "duty": duty,
               "run_id": name, "ratio": ratio,
               "ratio_interval": [lo, hi],
               "gates": [{"tag": "V7", "kind": "VALIDITY", "verdict": "PASS"},
                         {"tag": "C1", "kind": "CLAIM", "verdict": "FAIL"}]}
    p = tmp_path / name / "report.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(payload))
    return p


def test_pairs_admits_only_measured_reports_that_formed_a_ratio(tmp_path):
    a = _report(tmp_path, "run-a")
    b = _report(tmp_path, "run-b", seed=1, ratio=None, lo=None, hi=None)
    c = _report(tmp_path, "run-c", seed=2, synthetic=True)
    d = _report(tmp_path, "run-d", seed=2, experiment="clock_elasticity")
    assert H.pairs([str(a), str(b), str(c), str(d), str(tmp_path / "nope.json")]) == [str(a)]


def test_reading_prints_the_pair_row_and_the_exit_word(tmp_path):
    a = _report(tmp_path, "run-a", G=16, seed=2)
    row = H.reading(str(a))
    assert row == ["16", "2", "0.9551", "0.9544", "0.9695", "CLAIM_FAIL", "0.5", "run-a"]
    assert H.reading(str(tmp_path / "missing.json")) == ["unreadable"]


def test_eta_reads_the_band_through_the_arms_own_edges(tmp_path):
    import clock_elasticity as CE
    p = tmp_path / "r1" / "report.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"elasticity": {"value": 0.20, "lo": 0.15, "hi": 0.24},
                             "gates": [{"kind": "CLAIM", "number": "2", "verdict": "PASS"}]}))
    got = H.eta(str(p))
    assert got[:3] == ["0.2000", "0.1500", "0.2400"]
    assert got[3] == CE.band_of(0.15, 0.24)[0]
    assert got[4] == "DONE"
    p.write_text(json.dumps({"elasticity": {"value": 0.30, "lo": 0.20, "hi": 0.45}, "gates": []}))
    assert H.eta(str(p))[3] == "STRADDLES"


def test_run_id_and_estimate_come_off_the_plan_page(tmp_path):
    log = tmp_path / "r3.log"
    log.write_text("experiment  private_weight_reference / nvidia_h200-bm32-g4-abc123\n"
                   "estimated GPU time 154 s at the model's own timings\n"
                   "WALL CLOCK at duty 0.50: the ladder's 145 s of kernel time takes about 291 s,"
                   " the idle gaps\n")
    assert H.run_id(str(log), "private_weight_reference") == "nvidia_h200-bm32-g4-abc123"
    assert H.run_id(str(log), "clock_elasticity") == ""
    assert H.estimate(str(log)) == "291"          # the wall figure, not the kernel one
    r1 = tmp_path / "r1.log"
    r1.write_text("experiment  clock_elasticity / x-1cb0bac3\n"
                  "estimated wall time 876 s (14.6 min), itemised:\n")
    assert H.estimate(str(r1)) == "876"
    assert H.verdict(str(log)) == "NONE"
    log.write_text("RESULT: VALIDITY V7 PASS [VALIDITY] x | measured y | gate z\n"
                   "RESULT: CLAIM C1 FAIL [CLAIM] x | measured y | gate z\n")
    assert H.verdict(str(log)) == str(exit_codes.CLAIM_FAIL)


# --------------------------------------------------------------------------
# the laptop dry run: priced off the arms' own plans, nothing written into the tree
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    root = tmp_path_factory.mktemp("chain")
    env = laptop_env(REPO=str(ROOT), PY_BASE=sys.executable, PY_VLLM=sys.executable,
                     SESSION_ROOT=str(root / "session"), RESULTS_ROOT=str(root / "results"),
                     MOE_RESULTS_DIR=str(root / "results" / "gaps-nocard"), WORKSPACE=str(root))
    before = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    got = subprocess.run(["bash", str(CHAIN), "--dry-run"], capture_output=True,
                         text=True, timeout=1500, env=env)
    after = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    return got, root, before, after


def test_the_dry_run_prices_every_step_off_the_arms_own_plans(dry):
    got, root, _b, _a = dry
    assert got.returncode == 0, got.stdout[-2500:] + got.stderr[-800:]
    out = got.stdout
    assert "DRY RUN: every arm's own --dry-run is run and priced" in out
    for g in (1, 4, 16, 64):
        assert re.search(rf"^r1-g{g}\s", out, re.M), out
        for seed in (0, 1, 2):
            assert re.search(rf"^r3-g{g}-s{seed}\s", out, re.M), out
    assert "r1-g32" not in out
    assert out.count("off its own plan") == 4 + 12, "one priced line per arm, none for preflights"
    assert "PRICE, off the arms' own plans: 4 elasticity runs + 12 ratio runs" in out
    assert re.search(r"= \d+ min; at \$4\.59/h about \$\d+\.\d\d\. Book \d h\.", out), out
    assert "exfil-alpha_g-nocard.tar.gz" in out
    session = next((root / "session").glob("alpha_g-nocard-*"))
    ledger = (session / "CHAIN-dryrun.tsv").read_text().splitlines()
    names = [ln.split("\t")[0] for ln in ledger[1:]]
    assert names[:3] == ["preflight-r1", "preflight-r3", "preconditions"]
    rows = {ln.split("\t")[0]: ln.split("\t") for ln in ledger[1:]}
    assert rows["preflight-r1"][1] == "DONE" and "log agrees" in rows["preflight-r1"][6]
    assert rows["preconditions"][1] == "DONE" and "ARMS.tsv" in rows["preconditions"][6]
    assert rows["r3-g1-s0"][1] == "REFUSED", "a dry-run plan scores no gate: REFUSED"
    assert names[3:7] == ["r1-g1", "r1-g4", "r1-g16", "r1-g64"]
    assert names[7:] == [f"r3-g{g}-s{s}" for s in (0, 1, 2) for g in (1, 4, 16, 64)]
    assert not (session / "CHAIN.tsv").exists()


def test_the_dry_run_lines_carry_the_duty_the_swizzle_and_the_seed(dry):
    got, root, _b, _a = dry
    session = next((root / "session").glob("alpha_g-nocard-*"))
    logs = session / "chain-logs"
    r3 = (logs / "r3-g16-s2.log").read_text()
    assert "duty        0.50: every cell timed as bursts" in r3
    assert "'GROUP_SIZE_M': 16" in r3        # the ratio arm prints its pinned dict
    r1 = (logs / "r1-g1.log").read_text()
    assert re.search(r"GROUP_SIZE_M=1\b", r1) and "GROUP_SIZE_M=16" not in r1   # R1 prints k=v
    assert "duty 1, 0.7, 0.5" in r1
    # both preflights ran the scorer and said so
    assert "SELF-TEST OK" in (logs / "preflight-r3.log").read_text()
    assert re.search(r"^preflight-r1\s+DONE", got.stdout, re.M), got.stdout


def test_a_results_dir_without_the_card_is_refused_before_anything_is_made(tmp_path):
    """The driver's own rule, applied by the chain first: a MOE_RESULTS_DIR that
    does not carry the card is refused, exit 2, and no session directory is
    opened. The suite's results sandbox is exactly such a directory, so without
    this the chain would open a session and then watch the driver refuse."""
    env = laptop_env(REPO=str(ROOT), PY_BASE=sys.executable, PY_VLLM=sys.executable,
                     SESSION_ROOT=str(tmp_path / "session"), RESULTS_ROOT=str(tmp_path / "results"),
                     MOE_RESULTS_DIR=str(tmp_path / "plain"), WORKSPACE=str(tmp_path))
    got = subprocess.run(["bash", str(CHAIN), "--dry-run"], capture_output=True, text=True,
                         timeout=600, env=env)
    assert got.returncode == 2, got.stdout[-1500:]
    assert "REFUSED: MOE_RESULTS_DIR=" in got.stdout
    assert "does not contain the card 'nocard'" in got.stdout
    assert not (tmp_path / "session").exists(), "refused, yet a session directory was opened"
    assert "alpha(G) chain" not in got.stdout


def test_the_dry_run_writes_nothing_into_the_tree(dry):
    _got, _root, before, after = dry
    assert after == before, "the chain's dry run dirtied the checkout"
