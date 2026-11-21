"""Model v7's decode blind (docs/registered/2026-11-17-v7-decode-blind-gh200, final since the v7 gate read DECODE: GO),
off the GPU and on SYNTHETIC trees only: the registration recomputes to its committed files; its predictions recompute
from the frozen v7 at the CAL2 refit (no blind page exists); the freeze holds and its scorer refuses a broken one; the
bars are the ones registered for these cells (owner, D4-1), verbatim; the plan covers exactly the registered cells in the old blind's structure and
the pinned decode driver's dry run lists every unit; the launcher is RR's plus lines marked V7B, delivers the driver, the
plan and every page byte for byte on a stub VM, and refuses a DRAFT plan; the scorer certifies a model v7 prices and
not one it misprices. The registration's unseen scan is computed over the live tree, so since the blind's own session
was published (2026-11-18) it is recomputed over the sessions published before it (_build), and the live scan is held
to the registered one plus exactly that session's line and, since the v7 domain map's session was published
(2026-11-18), that session's line, which names no reserved model in a file name, and, since v7's prefill blind's
session was published (2026-11-18), that session's line, whose page names carry Granite-3.0-1B-A400M and
Qwen2-57B-A14B-tp8 (prefill pages, not decode pages)."""
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
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "decode",
          REPO / "scripts" / "scoring" / "v7blind"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decode_fakes as F  # noqa: E402
import register_v7_blind as RG  # noqa: E402
import score_blind_v7 as SV  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402
import test_rr_driver as TRR  # noqa: E402
import v7bcommon as V  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

REGF = REPO / V.REG_JSON
PLAN = REPO / V.PLAN
DRIVER = REPO / "scripts" / V.DRIVER
LAUNCHER = REPO / V.LAUNCHER
PARENT = REPO / V.PARENT_LAUNCHER
OLD_PAGES = REPO / "docs" / "registered" / "2026-10-14-decode-pages"


def _reg() -> dict:
    return json.loads(REGF.read_text())


@pytest.fixture(scope="module")
def pred():
    return V.predictions()


# ---------------------------------------------------------------- the registration and its predictions

def test_the_registration_recomputes_to_its_committed_files(pred):
    reg = _reg()
    doc = json.loads(json.dumps(_build(reg["draft"], pred), default=str))
    assert RG.close(doc, reg) == []
    assert (REPO / V.REG_TXT).read_text() == "\n".join(RG.text(doc)) + "\n"
    assert PLAN.read_text() == V.plan_text(doc["predictions"]["frozen"], doc["draft"])
    assert reg["rental"]["plan_sha256"] == hashlib.sha256(PLAN.read_bytes()).hexdigest()


def test_the_predictions_recompute_from_the_frozen_v7_before_any_page(pred):
    reg = _reg()["predictions"]
    got = json.loads(json.dumps(pred, default=str))
    assert RG.close(got["cells"], reg["cells"]) == [] and RG.close(got["frozen"], reg["frozen"]) == []
    assert len(got["cells"]) == 143 and got["not_priced"] == 0 and len(got["descriptive_cells"]) == 18
    keys = {(r["model"], r["label"]) for r in got["cells"]}
    for m in V.MODELS:
        page = json.loads((REPO / f"{V.PAGES}/{m}-main.json").read_text())
        assert {(m, c["label"]) for c in page["cells"]} == {k for k in keys if k[0] == m}
    fz = got["frozen"]
    assert fz["model"]["switches"] == {"RR": True, "D6FIX": True} and fz["model"]["name"] == "v7"
    assert fz["model"]["constants_sha256"] == V.V7.constants_sha256()
    # the CAL2 refit under v7: v6d's fit (fixed rho), GO; D6FIX's fit scaled at s_naive +9.55 us (the gate's disclosure)
    assert fz["rho_form"] == "fixed" and round(fz["constants"]["T0_GR_ms"] * 1e3, 2) == -1.87
    assert round(fz["constants"]["s_naive_GR_ms"] * 1e3, 2) == -2.14
    fix = fz["model"]["D6FIX"]
    assert fix["rho_form"] == "scaled" and round(fix["s_naive_GR_ms"] * 1e3, 2) == 9.55
    assert pred["CAL2 refit under v7"]["verdict"] == "GO"
    assert pred["D6FIX rung (DESCRIPTIVE)"]["cells moved by D6FIX"] == []
    # the gate registration's freeze is the one v7 is priced under
    assert V.verify_freeze(_reg()) == []


def test_the_bars_are_the_owners_d4_1_registered_for_these_cells_verbatim():
    reg = _reg()
    want = {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015}
    assert {k: reg["bars"][k] for k in want} == want == V.BARS
    assert {k: reg["predictions"]["frozen"]["bars"][k] for k in want} == want
    a5 = (REPO / "docs/registered/2026-11-04-decode-addendum5-gh200.txt").read_text().splitlines()
    a6 = (REPO / "docs/registered/2026-11-10-decode-addendum6-gh200.txt").read_text().splitlines()
    assert a5[8] == "new blind bars (owner, D4-1): rms 3% per model, worst cell 8%, |mean| 1.5%"
    assert a5[102].strip() == a6[158].strip() == \
        "bars: the owner's (D4-1): rms 3%, worst 8%, |mean| 1.5%, per model and pooled, not tightened"
    assert a6[157].strip().startswith("rules (frozen otherwise): addendum 5's: the power check first")
    assert a5[101].strip().startswith("rules (frozen otherwise): decodeB-blind's and addenda 2 and 3's")
    decb = (REPO / "docs/registered/2026-10-14-decodeB-blind-gh200.txt").read_text()
    assert "a model is CERTIFIED when its own bar and the pooled bar hold" in decb
    for q in (a5[8], "not tightened", "a model is CERTIFIED when its own bar and the pooled bar hold"):
        assert q in reg["bars"]["source"] + reg["certification"], q
    assert "tightened bars" not in json.dumps(reg["disclosures"])


#: the blind's own published session: the unseen scan (V.unseen_scan, every published session dated 2026-11-04 or
#: later) reads it since its publication, so the registration's scan is the live one without this line
BLIND_SESSION = "2026-11-18-nvidia_gh200_480gb-v7b-session"
#: the v7 domain map's published session (2026-11-18-v7-domain-map-gh200; synthetic shapes only), dated after the
#: 2026-11-04 cut, so the scan reads it too
MAP_SESSION = "2026-11-18-nvidia_gh200_480gb-v7m-session"
#: v7's prefill blind's published session (2026-11-18-v7-prefill-blind-gh200), dated after the cut too
PREFILL_SESSION = "2026-11-18-nvidia_gh200_480gb-v7p-session"
#: the sessions published after the registration, which its scan therefore does not hold
LATER = (BLIND_SESSION, MAP_SESSION, PREFILL_SESSION)


def _before(lines: list[str]) -> list[str]:
    return [ln for ln in lines if not ln.startswith(tuple(s + ":" for s in LATER))]


def _registered_scan() -> list[str]:
    return _before(V.unseen_scan())


def _build(draft: bool, pred) -> dict:
    """RG.build on the tree the registration was computed on: the unseen scan without the sessions published after it
    (the pinned builder and v7bcommon are not edited)."""
    live = V.unseen_scan
    V.unseen_scan = lambda: _before(live())
    try:
        return RG.build(draft=draft, pred=pred)
    finally:
        V.unseen_scan = live


@pytest.fixture(scope="module")
def draft_doc(pred):
    """The DRAFT the builder writes (--write without --final), generated here, never committed."""
    return json.loads(json.dumps(_build(True, pred), default=str))


def test_the_live_unseen_scan_is_the_registered_one_plus_the_blinds_own_session():
    """The pages the blind timed are the reserved models' first decode pages; once published they are SEEN, and the
    scan's only new line is their session, naming all three. register_v7_blind.py --check reads DIFFERS on this field
    alone (docs/FINDINGS.md, Model v7 decode blind)."""
    key = "unseen scan (COMPUTED: published sessions dated 2026-11-04 or later, check_unseen's pattern)"
    registered = _reg()["models"][key]
    live = V.unseen_scan()
    assert _registered_scan() == registered
    extra = [ln for ln in live if ln not in registered and ln.startswith(BLIND_SESSION + ":")]
    assert len(extra) == 1 and extra[0].startswith(BLIND_SESSION + ": ")
    assert int(re.search(r"file names naming a reserved model: (\d+);", extra[0]).group(1)) > 0
    pages = sorted(p.name for p in (REPO / "results/published" / BLIND_SESSION / "results").glob("gaps-*"))
    assert pages == [f"gaps-nvidia_gh200_480gb-{m}-{u}" for m, u in (
        ("granite-3.0-1b-a400m", "dc"), ("granite-3.0-1b-a400m", "dm"), ("mixtral-8x22b", "dm"),
        ("qwen2-57b-a14b", "dc"), ("qwen2-57b-a14b", "dm"))]


def test_the_live_unseen_scan_adds_only_the_blinds_and_the_maps_sessions_and_the_map_names_no_reserved_model():
    """Since the v7 domain map's session was published the live scan is the registered one plus exactly two lines,
    the blind's and the map's, in that order; the map's pages time synthetic shapes only, so no file name in its
    session names a reserved model (the text hits are setup logs and READMEs, as in every earlier line).
    register_v7_blind.py --check reads DIFFERS on this field alone (length 8 against 6; docs/FINDINGS.md, Model v7
    decode domain map)."""
    key = "unseen scan (COMPUTED: published sessions dated 2026-11-04 or later, check_unseen's pattern)"
    registered = _reg()["models"][key]
    live = V.unseen_scan()
    extra = [ln for ln in live if ln not in registered and not ln.startswith(PREFILL_SESSION + ":")]
    assert [ln.split(":")[0] for ln in extra] == [BLIND_SESSION, MAP_SESSION]
    assert int(re.search(r"file names naming a reserved model: (\d+);", extra[1]).group(1)) == 0
    pages = sorted(p.name for p in (REPO / "results/published" / MAP_SESSION / "results").glob("gaps-*"))
    assert len(pages) == 14 and all(p.startswith("gaps-nvidia_gh200_480gb-map-") and p.endswith("-dm") for p in pages)


def test_the_live_unseen_scan_adds_the_prefill_blinds_session_last_and_its_pages_are_prefill_pages():
    """Since v7's prefill blind's session was published the live scan is the registered one plus exactly three lines,
    the decode blind's, the map's and the prefill blind's, in that order. The prefill session's file names carry
    Granite-3.0-1B-A400M (a reserved model here) and Qwen2-57B-A14B-tp8 (the pattern's qwen2-57b): its timed pages are
    B3's prefill pages at 128x128x64 w8 s3, G 1 and G 16, no decode page. register_v7_blind.py --check reads DIFFERS
    on this field alone (length 9 against 6)."""
    key = "unseen scan (COMPUTED: published sessions dated 2026-11-04 or later, check_unseen's pattern)"
    registered = _reg()["models"][key]
    live = V.unseen_scan()
    extra = [ln for ln in live if ln not in registered]
    assert len(live) == len(registered) + 3 and [ln.split(":")[0] for ln in extra] == list(LATER)
    assert int(re.search(r"file names naming a reserved model: (\d+);", extra[2]).group(1)) > 0
    pages = sorted(p.name for p in (REPO / "results/published" / PREFILL_SESSION / "results").glob("gaps-*"))
    assert pages == sorted(f"gaps-nvidia_gh200_480gb-{m}-t128g{g}" for m in (
        "granite-3.0-1b-a400m", "granite-3.0-3b-a800m", "qwen2-57b-a14b-tp8") for g in (1, 16))


def test_the_committed_registration_and_plan_are_final_and_the_committed_gate_reads_decode_go():
    reg = _reg()
    assert reg["draft"] is False and reg["status"].startswith("REGISTERED BEFORE ANY PAGE")
    assert (REPO / V.REG_TXT).read_text().startswith("REGISTERED ")
    assert "# DRAFT" not in PLAN.read_text() and V.DRAFT_LINE not in PLAN.read_text().splitlines()
    assert V.gate_reads_decode_go()                         # --final's precondition holds on the committed gate output
    assert any(ln.startswith("DECODE: GO") for ln in (REPO / V.GATE_OUT).read_text().splitlines())
    assert SV.refusals(reg) == []                           # the committed registration is scorable as is


def test_a_draft_differs_only_in_its_marks_and_final_needs_the_gates_decode_go(draft_doc, tmp_path, monkeypatch, capsys):
    reg = _reg()
    assert draft_doc["draft"] is True and draft_doc["status"].startswith("DRAFT")
    assert "\n".join(RG.text(draft_doc)).startswith("DRAFT (not registered)")
    assert V.DRAFT_LINE in V.plan_text(draft_doc["predictions"]["frozen"], True).splitlines()
    same = {k: v for k, v in draft_doc.items() if k not in ("draft", "status", "rental")}
    assert RG.close(same, {k: v for k, v in reg.items() if k not in ("draft", "status", "rental")}) == []
    # --final refuses without a 'DECODE: GO' line in the gate output, and writes nothing
    monkeypatch.setattr(V, "GATE_OUT", "no/such/gate_v7.out.txt")
    assert RG.main(["--write", "--final", "--root", str(tmp_path)]) == 2
    assert "REFUSED" in capsys.readouterr().out and not (tmp_path / V.REG_JSON).exists()
    monkeypatch.setattr(V, "GATE_OUT", "gate.txt")
    for text, go in (("R3: GO\nPREFILL: NO-GO\nDECODE: GO: this regime may go to ONE blind\n", True),
                     ("DECODE: NO-GO: no blind for this regime\n", False), ("", False)):
        (tmp_path / "gate.txt").write_text(text)
        assert V.gate_reads_decode_go(tmp_path) is go


def test_the_registration_discloses_the_reopening_the_known_gate_and_d6_24():
    t = " ".join(_reg()["disclosures"])
    for s in ("reopening of 2026-11-17", "gate outcome is known before this registration", "D6-24 was found on SEEN",
              "owner, D4-1"):
        assert s in t, s


# ---------------------------------------------------------------- the models and cells

def test_the_models_have_no_decode_page_and_every_other_candidate_is_named():
    reg = _reg()["models"]
    cen = V.census()
    assert set(reg["every model with no decode page"]) == {m for m, x in cen.items() if x["verdict"] == "NO DECODE PAGE"}
    assert set(V.MODELS) <= set(reg["every model with no decode page"])
    for m in ("mixtral-8x7b-tp2", "mixtral-8x7b-tp4", "mixtral-8x7b-tp8"):
        assert cen[m]["verdict"] == "HAS A DECODE PAGE"          # PC's A cells at the 64x128x64 w4 s3 decode tile
    for m in V.MODELS:
        assert reg["footprint (COMPUTED, d5common.footprint)"][m]["weights_GB"] < 5
    q = reg["labels"]["per_model"]["qwen2-57b-a14b"]
    assert "A100 (2026-08-28 a100-cross-card" in q and "H200" in q and "no GH200 decode page" in q


def test_the_cells_keep_the_old_decode_blinds_structure():
    reg = _reg()["cells"]
    old_T = set()
    old_roles = set()
    for f in OLD_PAGES.glob("*-main.json"):
        if f.name.startswith(("mixtral-8x7b", "olmoe")):
            continue
        for c in json.loads(f.read_text())["cells"]:
            old_T.add(c["T"])
            old_roles.add(c["role"])
    for m in V.MODELS:
        doc = json.loads((REPO / f"{V.PAGES}/{m}-main.json").read_text())
        assert doc["shuffle_seed"] == 20261014 and all(c["arms"] == ["native"] for c in doc["cells"])
        assert {c["T"] for c in doc["cells"]} == old_T - {3}
        assert {c["role"] for c in doc["cells"]} == old_roles - {"TRACE-DW EXTRAPOLATED"}
    for m in V.CTL:
        main = {c["label"] for c in json.loads((REPO / f"{V.PAGES}/{m}-main.json").read_text())["cells"]}
        ctl = json.loads((REPO / f"{V.PAGES}/{m}-ctl.json").read_text())["cells"]
        assert {c["label"] for c in ctl} <= main and {c["role"] for c in ctl} == {"BALANCED", "UNIFORM", "LADDER-BOTTOM"}
    assert reg["units"] == V.add6_blind()["units"] and reg["pages"] == V.add6_blind()["pages"]
    assert [(u["label"], u["inst"], u["drop"]) for u in reg["units"]] == \
        [("dm", "GRR", "nodrop")] * 3 + [("dc", "GR1", "gr1")] * 2


def test_the_plan_covers_exactly_the_registered_cells_and_the_pinned_driver_lists_every_unit():
    reg = _reg()
    units = [ln for ln in PLAN.read_text().splitlines() if ln and not ln.startswith("#")]
    flat = [y for x in reg["cells"]["plan_units"] for y in ([x] if isinstance(x, str) else x)]
    assert units == flat == V.D5.blind_plan_units()
    pages = re.findall(r"histogram=(\S+)", PLAN.read_text())
    assert sorted(pages) == sorted(v["path"] for v in reg["cells"]["pages"]["files"].values())
    for p in pages:
        assert hashlib.sha256((REPO / p).read_bytes()).hexdigest() == \
            reg["cells"]["pages"]["files"][Path(p).name]["sha256"] == reg["freeze"]["own"][p]
    got = subprocess.run(["bash", str(DRIVER), "--dry-run", "--plan", str(PLAN), "--deadline", "2000000000"],
                         capture_output=True, text=True, timeout=60, env={**os.environ, "HOME": "/nonexistent"},
                         cwd=REPO)
    assert got.returncode == 2, got.stdout + got.stderr
    assert "a PLAN of 7 units" in got.stdout and "72 min of units" in got.stdout
    assert got.stdout.count(" r3decode ") == 5 and "drop order when time runs short" in got.stdout
    r = reg["rental"]
    assert (r["units_min"], r["wall_min"], r["deadline_after_launch_min"], r["hard_cap_after_launch_min"]) == \
        (72, 111, 125, 140) and r["cost_usd"] == 4.24 and r["cost_at_hard_cap_usd"] == 5.34


# ---------------------------------------------------------------- the launcher

def test_the_launcher_is_rrs_plus_lines_marked_v7b():
    changed = TRR._held_to_parent(PARENT, LAUNCHER, None, mark="V7B")
    drivers = re.search(r"^SESSION_DRIVERS=\(([^)]*)\)", LAUNCHER.read_text(), re.M).group(1).split()
    assert drivers == ["gh200_decode_session.sh"]
    assert sum('local drv="${DRIVER_ARG:-gh200_decode_session.sh}"' in x for x in changed) == 1
    assert sum("^# DRAFT" in x and "refuse" in x for x in changed) == 1
    assert "MOE_VM_APT_BUDGET_S" in LAUNCHER.read_text() and (LAUNCHER.stat().st_mode & 0o777) == \
        (PARENT.stat().st_mode & 0o777)
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == _reg()["freeze"]["own"][V.PARENT_LAUNCHER]


def _laptop(tmp_path, plan_text: str):
    pages = sorted(set(re.findall(r"histogram=(\S+)", plan_text)))
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    (lap.clone / V.PLAN).write_text(plan_text)
    for f in (DRIVER, LAUNCHER):
        shutil.copy(f, lap.clone / "scripts" / f.name)
    for p in pages:
        (lap.clone / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, lap.clone / p)
    g = ["git", "-C", str(lap.clone)]
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "plan"], check=True)
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-t"], check=True, capture_output=True)
    lap.make_branch()
    env = {**os.environ, "MOE_VM_SSH": str(lap.bin / "ssh"), "MOE_VM_SCP": str(lap.bin / "scp"),
           "MOE_VM_GH": str(lap.bin / "gh"), "MOE_VM_STATE_ROOT": str(lap.state), "MOE_VM_POLL_S": "0",
           "MOE_VM_WAIT_S": "5", "FAKE_VM": str(lap.vm), "FAKE_GH": str(lap.gh)}

    def run(*args):
        return subprocess.run(["bash", str(lap.clone / V.LAUNCHER), *args], env=env, capture_output=True, text=True,
                              timeout=120)
    assert run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
    return lap, run, pages


def test_start_delivers_the_driver_the_final_plan_and_every_page_byte_for_byte(tmp_path):
    final = V.plan_text(_reg()["predictions"]["frozen"], draft=False)
    assert "# DRAFT" not in final
    lap, run, pages = _laptop(tmp_path, final)
    real = lap.bin / "ssh.real"
    (lap.bin / "ssh").rename(real)
    TG._exe(lap.bin / "ssh", f'#!/bin/bash\ncat > /dev/null\nexec "{real}" "$@"\n')
    vm = tmp_path / "vm-home"
    vm.mkdir()
    TG._exe(lap.bin / "scp", (
        '#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"\n'
        '[[ "$dst" == *:* ]] || exit 0\n'
        f'rel="${{dst#*:}}"; [[ -n "$rel" ]] || rel="$(basename "$src")"\n'
        f'mkdir -p "$(dirname "{vm}/$rel")" && cp "$src" "{vm}/$rel"\nexit 0\n'))
    run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", V.PLAN)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(vm / PLAN.name) == hashlib.sha256(final.encode()).hexdigest() and sha(vm / DRIVER.name) == sha(DRIVER)
    assert len(pages) == 5
    for p in pages:
        assert (vm / p).is_file() and sha(vm / p) == sha(REPO / p), p
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "gh200_decode_session.sh --dry-run" in cmds and "gh200_decode_session.sh --setup" in cmds


def test_start_refuses_a_draft_plan_and_starts_nothing(tmp_path, draft_doc):
    draft = V.plan_text(draft_doc["predictions"]["frozen"], True)
    lap, run, _pages = _laptop(tmp_path, draft)
    got = run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", V.PLAN)
    assert got.returncode == exit_codes.REFUSED and "carries a DRAFT line" in got.stdout + got.stderr
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "--dry-run" not in cmds and "--setup" not in cmds


# ---------------------------------------------------------------- the scorer, on synthetic trees only

def _tree(root: Path, reg: dict, slow: dict) -> Path:
    preds = {(r["model"], r["label"]): r["ms"]["v4d"] for r in reg["predictions"]["cells"]}
    (root / "session").mkdir(parents=True)
    shutil.copy(REPO / "results/published/2026-11-03-nvidia_gh200_480gb-decodeCAL2-session/session/power-set.txt",
                root / "session" / "power-set.txt")
    tree = root / "results"
    for u in reg["cells"]["units"]:
        m = u["model"]
        k = slow.get(m, 1.0)
        plus = 0.002 if u["inst"] == "GR1" else 0.0
        F.write_page(tree, m, u["label"], REPO / reg["cells"]["pages"]["files"][u["page"]]["path"], u["inst"],
                     lambda c, a, m=m, k=k, plus=plus: preds[(m, c.label)] * k + plus, noise=0.0005, seed=7)
    return tree


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    reg = _reg()
    good = SV.score(_tree(tmp_path_factory.mktemp("good"), reg, {}), reg)
    slow = SV.score(_tree(tmp_path_factory.mktemp("slow"), reg, {"mixtral-8x22b": 1.035}), reg)
    bad = SV.score(_tree(tmp_path_factory.mktemp("bad"), reg, {"granite-3.0-1b-a400m": 1.06}), reg)
    return good, slow, bad


def test_the_scorer_certifies_every_model_v7_prices_on_a_synthetic_tree(scored):
    good, _slow, _bad = scored
    assert good["POWER"]["verdict"] == "READ" and good["bars"] == V.BARS
    assert good["E1_DEC"]["pooled"]["verdict"] == "HOLDS"
    assert set(good["certified"].values()) == {"CERTIFIED"}
    assert good["pooled_over"] == sorted(V.MODELS)
    assert len(good["rows"]) == 143 - 18
    assert {m: good["GD2"][m]["verdict"] for m in V.CTL} == {m: "PASS" for m in V.CTL}


def test_the_scorer_refuses_to_certify_a_model_v7_misprices(scored):
    _good, s, bad = scored
    # Mixtral-8x22B's calls (hundreds of us) 3.5% slower than v7 prices: its own rms bar (3%) fails, the pooled bars
    # hold, so the other two certify
    assert s["E1_DEC"]["per_model"]["mixtral-8x22b"]["verdict"] == "FAILS"
    assert s["certified"]["mixtral-8x22b"] == "NOT CERTIFIED" and s["E1_DEC"]["pooled"]["verdict"] == "HOLDS"
    assert s["certified"]["qwen2-57b-a14b"] == s["certified"]["granite-3.0-1b-a400m"] == "CERTIFIED"
    # Granite-1B 6% slow: its own bars and the pooled bars fail, so no model certifies (the pooled clause)
    assert bad["E1_DEC"]["per_model"]["granite-3.0-1b-a400m"]["verdict"] == "FAILS"
    assert bad["E1_DEC"]["pooled"]["verdict"] == "FAILS"
    assert set(bad["certified"].values()) == {"NOT CERTIFIED"}


def test_the_scorer_reads_only_the_committed_registration_and_refuses_a_draft_and_a_broken_freeze(tmp_path, draft_doc):
    with pytest.raises(SystemExit):
        SV.main([str(tmp_path), str(tmp_path / "o"), "--registration", str(REGF)])   # no such option
    assert SV.refusals(draft_doc)[0].startswith("the registration is a DRAFT")
    assert SV.refusals(draft_doc, allow_draft=True) == []
    reg = _reg()
    assert SV.refusals(reg) == []
    p = next(iter(reg["freeze"]["own"]))
    reg["freeze"]["own"][p] = "0" * 64
    assert SV.refusals(reg)[0].startswith("the freeze does not hold")
    assert SV.refusals(dict(_reg(), name="another"))[0].startswith("the registration is not")
    pv = SV.provenance(REGF)
    assert pv["registration_sha256"] == hashlib.sha256(REGF.read_bytes()).hexdigest()
    assert pv["registration_path"] == V.REG_JSON


def test_no_em_dash_and_no_trailer_in_the_blind_files():
    files = [REPO / V.REG_JSON, REPO / V.REG_TXT, PLAN, LAUNCHER, Path(__file__),
             *(REPO / "scripts/scoring/v7blind").glob("*.py")]
    for f in files:
        t = f.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, f
        assert ("Co-" + "Authored") not in t and ("Clau" + "de") not in t, f
