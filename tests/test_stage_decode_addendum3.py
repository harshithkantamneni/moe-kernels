"""The decode stage's third addendum (docs/registered/2026-10-29-decode-addendum3-gh200), off the GPU: v4d'
(scripts/r3_timing_model_v4dp.py) reduces to v4d exactly with its fixes off (every page cell, and the published CAL
score byte for byte); each fix acts where it is registered and nowhere else; its measured constants recompute from the
published pages; the stop rule reads G3-D label DESCRIPTIVE and nothing else; the CAL2 plan is addendum 2's plus the
settling units and parses under the D3 driver, whose every body change is marked D3 against the pinned decode driver;
the settling pages and the two GPU-side wrappers; and the wrappers on SYNTHETIC trees (decode_fakes) where v4d' is the
truth: CAL2 reads GO with a planted Mixtral label effect, recovers the constants, finalises, certifies the blind models,
and reads the settling units."""
from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "decode"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import d2common as D2  # noqa: E402
import d3common as D3  # noqa: E402
import dcommon as DC  # noqa: E402
import decode_fakes as F  # noqa: E402
import finalise_decode_d2 as FN2  # noqa: E402
import finalise_decode_d3 as FN3  # noqa: E402
import register_decode_d3 as RG3  # noqa: E402
import score_blind_d3 as SB3  # noqa: E402
import score_cal_d2 as SC2  # noqa: E402
import score_cal_d3 as SC3  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

V4, V4D, RD, V4DP = DC.V4, DC.V4D, DC.RD, D3.V4DP
REG = REPO / "docs" / "registered"
NOM = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0})
PLANT = dict(T0_GR_ms=0.028, s_naive_GR_ms=-0.003, s_block_GR_ms=0.004)
POWER_TXT = """GPU 00000000:DD:00.0
    GPU Power Readings
        Current Power Limit               : {gpu} W
    Module Power Readings
        Current Power Limit               : 1000.00 W
"""
DRIVER = REPO / "scripts" / D3.DRIVER3
PINNED_DRIVER = REPO / "scripts" / "gh200_decode_session.sh"


def _sha(p) -> str:
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def _reg() -> dict:
    return json.loads((REG / f"{D3.NAME}.json").read_text())


# ---------------------------------------------------------------- registration

def test_the_addendum_recomputes_to_its_committed_files(capsys):
    assert RG3.main(["--check"]) == 0, capsys.readouterr().out


def test_the_addendum_pins_every_earlier_decode_file_and_itself_at_their_bytes():
    pins = _reg()["code_pins"]
    for p, s in pins.items():
        assert _sha(p) == s, p
    for name in (DC.NAMES["cal"], DC.NAMES["blind"], RG3.RG2.TILESOURCE, D2.NAME):
        for x in ("json", "txt"):
            assert f"docs/registered/{name}.{x}" in pins
        for p, s in json.loads((REG / f"{name}.json").read_text())["code_pins"].items():
            if p != f"docs/registered/{D3.NAME}.json":
                assert pins.get(p) == s, p
    for p in RG3.D3_CODE + ("scripts/r3_timing_model_v4d.py", D3.PUBLISHED_SCORE, "scripts/gh200_decode_session.sh"):
        assert p in pins, p


def test_the_rentals_state_minutes_wall_cost_deadline_and_cap():
    r = _reg()["rentals"]
    assert (r["cal2"]["units_min"], r["cal2"]["wall_min"], r["cal2"]["cost_usd"]) == (136, 187, 7.14)
    assert (r["cal2"]["deadline_after_launch_min"], r["cal2"]["hard_cap_after_launch_min"]) == (200, 215)
    assert (r["blind"]["units_min"], r["blind"]["wall_min"], r["blind"]["cost_usd"]) == (101, 145, 5.53)
    assert (r["blind"]["deadline_after_launch_min"], r["blind"]["hard_cap_after_launch_min"]) == (160, 175)
    for x in (r["cal2"], r["blind"]):
        assert x["cost_usd"] == round(x["wall_min"] / 60 * 2.29, 2) and x["boot_setup_min"] == 25
    assert sum(u["est"] for u in D3.settle_units()) == 30


def test_the_registration_states_the_decisions_the_odds_and_the_open_questions():
    d = _reg()
    m = d["model v4d' (owner, 2026-10-29, D3-1)"]
    assert m["fitted constants added"] == 0 and m["reduction (fixes off)"]["verdict"] == "IDENTICAL"
    assert m["reduction (fixes off)"]["predictions"] == 92 and m["reduction (fixes off)"]["predictions_identical"]
    fx = m["fixes"]
    assert fx["R (w1 rate, 16x64x128w4s4)"]["recomputed"]["agrees"].startswith("READ")
    assert len(fx["R (w1 rate, 16x64x128w4s4)"]["source_pairs"]) == 7
    assert set(fx) == {"R (w1 rate, 16x64x128w4s4)", "A (align loop): DROPPED, not in v4d' (owner, 2026-11-03, D3-10)",
                       "W (naive order)"}
    assert fx["R (w1 rate, 16x64x128w4s4)"]["source"].startswith("MEASURED on SEEN pages:")
    odds = d["bars and certification (owner, 2026-10-29, D3-3)"]["expected odds (ESTIMATES, the diagnosis's, not measured)"]
    assert [odds[k] for k in list(odds)[:4]] == ["about 0.75", "about 0.7", "about 0.05", "about 0.15"]
    assert "DESCRIPTIVE" in d["G3-D label (owner, 2026-10-29, D3-2)"]["rule"]
    assert "-1.658" in d["G3-D label (owner, 2026-10-29, D3-2)"]["the published effect (Mixtral, quoted)"]
    dec = d["owner decisions of 2026-11-03 (on the build's open list)"]
    assert [x[:5] for x in dec] == ["D3-6 ", "D3-7 ", "D3-8 ", "D3-10", "D3-11", "D3-9 "] and "own registration" in dec[5]
    assert "own registration" in dec[3] and "52 rows" in dec[4]
    assert "OWNER DECISIONS LEFT" not in d
    g = d["G3-D label (owner, 2026-10-29, D3-2)"]
    assert "never voids" in g["blind (owner, 2026-11-03, D3-6)"] and g["scope line (blind and finalisation)"] == D3.SCOPE
    assert D3.SCOPE.startswith(D2.SCOPE) and "G3-D label, the test of that dependence, is DESCRIPTIVE" in D3.SCOPE
    assert "does not cover a dependence on which experts are active" in D3.SCOPE              # review m7
    st_ = d["CAL2 plan (owner, 2026-10-29, D3-4)"]["settling readings (DESCRIPTIVE, registered before pages)"]
    assert "exactly 60 MiB" in st_["M4 gaps (units m4: dg GRR and dgn counters on OLMoE)"]         # review m3
    assert "P >= 51.2" in st_["M3 pad (owner, 2026-11-03, D3-11)"]
    s = d["seen_reading"]
    assert s["selected"] == "scaled" and s["STOP_D3"]["verdict"] == "GO" and s["E1_CAL_IN_SAMPLE"]["verdict"] == "HOLDS"
    txt = (REG / f"{D3.NAME}.txt").read_text()
    assert "reduction: IDENTICAL" in txt and "cal2 rental: 136 min of units" in txt and "deadline launch + 200 min" in txt


# ---------------------------------------------------------------- v4d' reduces to v4d, and each fix

def _all_cells():
    for m in DC.CAL_MODELS + DC.BLIND_MODELS:
        for kind in ("main", "ctl", "decl"):
            if (REPO / DC.page_path(m, kind)).exists():
                for c in DC.page_doc(m, kind)["cells"]:
                    yield m, kind, c


def test_with_every_fix_off_v4dp_is_v4d_on_every_registered_cell():
    prm = V4D.DecodeParams(T0_GR_ms=0.025, s_naive_GR_ms=-0.008, s_block_GR_ms=-0.01, rho_form="scaled", tile=NOM)
    n = 0
    for m, _kind, c in _all_cells():
        try:
            a = V4D.price(m, c["T"], c["counts"], prm)
        except V4.TM.Refused:
            with pytest.raises(V4.TM.Refused):
                V4DP.price(m, c["T"], c["counts"], prm, fixes=V4DP.NONE)
            continue
        b = V4DP.price(m, c["T"], c["counts"], prm, fixes=V4DP.NONE, ids=V4DP.ids_of(m, c))
        assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str), (m, c["label"])
        n += 1
    assert n > 400


def test_the_published_cal_score_is_reproduced_byte_for_byte_with_the_fixes_off():
    got = RG3.reduction()
    assert got["verdict"] == "IDENTICAL" and got["predictions"] == 92


def test_fix_r_scales_the_w1_gemm_on_the_16x64x128_tile_only_and_there_is_no_align_term():
    prm = V4D.DecodeParams(T0_GR_ms=0.025, rho_form="scaled", tile=NOM)
    only_r = V4DP.Fixes(False, True)
    seen = set()
    assert set(V4DP.Fixes().as_dict()) == {"naive_order", "w1_rate"} and set(V4DP.describe()) == {"W", "R"}
    assert not hasattr(V4DP, "ALIGN_US")
    for m, _kind, c in _all_cells():
        a = DC.price_cell(m, c, prm)
        if a is None:
            continue
        r = D3.price_cell_v4dp(m, c, prm, fixes=only_r)
        if a["tile"] == V4DP.W1_TILE:
            want = a["gemm"]["w1"]["gemm_ms"] * (V4DP.W1_SCALE - 1)
            assert r["ms"] - a["ms"] == pytest.approx(want, rel=1e-9, abs=1e-12)
            assert r["gemm"]["w2"]["gemm_ms"] == a["gemm"]["w2"]["gemm_ms"]
            seen.add("R")
        else:
            assert r["ms"] == a["ms"]
        if a["path"] != RD.NAIVE:            # off the naive path v4d' is v4d plus R and nothing else (A dropped)
            assert D3.price_cell_v4dp(m, c, prm)["ms"] == r["ms"]
            seen.add("no A")
        assert r["reads"] == a["reads"]
    assert seen == {"R", "no A"}


def test_fix_w_reproduces_the_diagnosis_bytes_and_leaves_no_repeat_naive_cells_alone():
    prm = V4D.DecodeParams(rho_form="scaled", tile=NOM)
    only_w = V4DP.Fixes(True, False)
    cells = {c["label"]: c for c in DC.page_doc("olmoe-1b-7b", "main")["cells"]}
    w1 = {lab: D3.price_cell_v4dp("olmoe-1b-7b", cells[lab], prm, fixes=only_w)["reads"]["w1"] / 1e6
          for lab in ("L-T2-A8", "U-T2", "L-T1-A8", "L-T2-A16")}
    assert w1["L-T2-A8"] == pytest.approx(97.6, abs=0.1) and w1["U-T2"] == pytest.approx(134.3, abs=0.1)
    for lab in ("L-T1-A8", "L-T2-A16"):          # no repeated expert: W is v4d
        a, b = DC.price_cell("olmoe-1b-7b", cells[lab], prm), D3.price_cell_v4dp("olmoe-1b-7b", cells[lab], prm, fixes=only_w)
        assert a["reads"] == b["reads"] and a["body_ms"] == pytest.approx(b["body_ms"], rel=1e-12)
    with pytest.raises(V4.TM.Refused, match="ids"):
        V4DP.price("olmoe-1b-7b", 2, cells["L-T2-A8"]["counts"], prm, fixes=only_w)
    nod = dataclasses.replace(prm, naive_dedup=False)       # NAIVE-NODEDUP stays v4d's
    assert D3.price_cell_v4dp("olmoe-1b-7b", cells["L-T2-A8"], nod, fixes=only_w)["reads"] == \
        DC.price_cell("olmoe-1b-7b", cells["L-T2-A8"], nod)["reads"]


def test_the_measured_constant_recomputes_and_the_dropped_a_is_recorded():
    c = RG3.constants()
    r = c["R (w1 rate, 16x64x128w4s4)"]
    assert r["recomputed"]["scale"] == pytest.approx(V4DP.W1_SCALE, rel=1e-12)
    assert r["recomputed"]["counter_GBps"] == pytest.approx(3895.49, abs=0.01)
    a = c["A (align loop): DROPPED, not in v4d' (owner, 2026-11-03, D3-10)"]["why (review MAJOR-1, recomputed here, SEEN pages)"]
    assert a["olmoe-1b-7b"]["slope_us"] == pytest.approx(0.995, abs=5e-4)
    assert a["mixtral-8x7b"]["slope_us"] == pytest.approx(0.922, abs=5e-4)
    assert a["mixtral-8x7b"]["slope_with_held_out_cells_us"] == pytest.approx(1.33, abs=5e-3)
    assert a["mean_slope_with_held_out_us"] == pytest.approx(1.16, abs=5e-3)
    assert a["mixtral-8x7b"]["B_other_slope_us"] == pytest.approx(0.95, abs=5e-3)
    assert a["olmoe-1b-7b"]["B_other_slope_us"] == pytest.approx(0.20, abs=5e-3)


def test_the_seen_reading_is_the_diagnosis_w_r_fit():
    s = _reg()["seen_reading"]
    f = s["forms"]["scaled"]
    assert f["T0_GR_ms"] * 1e3 == pytest.approx(28.8958, abs=1e-3)
    assert f["s_naive_GR_ms"] * 1e3 == pytest.approx(-13.1267, abs=1e-3)
    assert f["s_block_GR_ms"] * 1e3 == pytest.approx(-14.4718, abs=1e-3)
    st = s["E1_CAL_IN_SAMPLE"]["stats"]
    assert (st["rms"], st["worst"], st["mean"]) == pytest.approx((0.0213, -0.0566, 0.0015), abs=2e-4)
    assert s["STOP_D3"]["verdict"] == "GO" and s["selected"] == "scaled"


def test_the_m3_pad_is_the_smallest_that_crosses_a_2_mib_page():
    P = D3.M3_PADS[1]
    row = MODEL_CONFIGS["mixtral-8x7b"].hidden_size * 2
    assert D3.M3_PADS == (0, 52) and 5 * P * row >= 2 * 2 ** 20 > 5 * (P - 1) * row
    labs = [u["label"] for u in D3.settle_units() if u["unit"] == "m3"]
    assert labs == ["dp0", "dp52"] and "pad-rows=52" in (REPO / D3.CAL2_PLAN).read_text()


def test_the_hook_restores_the_frozen_price_cell():
    orig = DC.price_cell
    with D3.v4dp():
        assert DC.price_cell is not orig
    assert DC.price_cell is orig


# ---------------------------------------------------------------- the stop rule

def _score(**over):
    s = {"units": {}, "FIT": {"selected": "scaled"}, "E1_CAL_IN_SAMPLE": {"verdict": "HOLDS"},
         "G3D": {"shuffle": {"verdict": "EQUIVALENT"}, "label": {"verdict": "EQUIVALENT"}}, "GD2": {"verdict": "PASS"}}
    s.update(over)
    return s


def test_the_stop_rule_moves_only_the_label_line_to_descriptive():
    assert D3.stop_rule_d3(_score())["verdict"] == "GO"
    lab = _score(G3D={"shuffle": {"verdict": "EQUIVALENT"}, "label": {"verdict": "NOT EQUIVALENT"}})
    got = D3.stop_rule_d3(lab)
    assert got["verdict"] == "GO" and got["frozen_rule"]["verdict"] == "STOP" and got["descriptive"][0].startswith("G3-D label")
    for bad in (dict(E1_CAL_IN_SAMPLE={"verdict": "FAILS"}), dict(GD2={"verdict": "FAIL (a STOP): x"}),
                dict(G3D={"shuffle": {"verdict": "NOT EQUIVALENT"}, "label": {"verdict": "NOT EQUIVALENT"}}),
                dict(units={"x": {"nodrop": True, "state": "NOT SCORED"}}), dict(NOT_PRICED=["a"]),
                dict(V_ALIGN_refused=["a"]), dict(FIT={"selected": None})):
        assert D3.stop_rule_d3(_score(**bad))["verdict"] == "STOP", bad


# ---------------------------------------------------------------- the plan and the D3 driver

def dry(*args, driver=DRIVER):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


def test_the_cal2_plan_is_addendum_2s_plan_then_the_settling_units():
    new = (REPO / D3.CAL2_PLAN).read_text()
    assert new == D3.cal2_plan()
    a = [ln for ln in (REPO / D2.CAL_PLAN_D2).read_text().splitlines() if ln and not ln.startswith("#")]
    b = [ln for ln in new.splitlines() if ln and not ln.startswith("#")]
    assert b[:len(a)] == a and len(b) == len(a) + 8
    assert [re.search(r"drop-group=(\w+)", ln).group(1) for ln in b[len(a):]] == ["m1", "m1", "m4", "m4", "m2", "m2", "m3", "m3"]
    assert _reg()["CAL2 plan (owner, 2026-10-29, D3-4)"]["plan_sha256"] == _sha(D3.CAL2_PLAN)


def test_the_d3_driver_parses_the_cal2_plan_in_136_minutes():
    got = dry("--plan", str(REPO / D3.CAL2_PLAN))
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    assert "a PLAN of 21 units" in got.stdout and "136 min of units" in got.stdout
    order = re.search(r"drop order when time runs short: .*\n\s+([\d ]+)", got.stdout).group(1).split()
    assert order == ["21", "20", "19", "18", "17", "16", "15", "14", "13", "12", "11", "8"]
    assert got.stdout.count("--r3-script scripts/r3_decode3.py") == 2 and "pad rows 52" in got.stdout
    assert got.stdout.count("dram_counter_route_decode3.py --kernels wide") == 2


def test_the_pinned_driver_still_parses_addendum_2s_plan_and_refuses_the_d3_keys():
    got = dry("--plan", str(REPO / D2.CAL_PLAN_D2), driver=PINNED_DRIVER)
    assert got.returncode == exit_codes.REFUSED and "a PLAN of 13 units" in got.stdout
    got = dry("--plan", str(REPO / D3.CAL2_PLAN), driver=PINNED_DRIVER)
    assert "no key 'pad-rows'" in got.stdout + got.stderr or "no key 'kernels'" in got.stdout + got.stderr


@pytest.mark.parametrize("line, why", [
    ("olmoe-1b-7b decodebytes label=x histogram={p} pad-rows=7", "pad-rows belongs to an r3decode unit"),
    ("olmoe-1b-7b r3decode label=x histogram={p} inst=GRR slip-policy=cell kernels=wide", "kernels belongs to a decodebytes unit"),
    ("olmoe-1b-7b decodebytes label=x histogram={p} kernels=all", "kernels all: wide"),
    ("olmoe-1b-7b r3decode label=x histogram={p} inst=GRR slip-policy=cell pad-rows=-1", "pad-rows -1")])
def test_the_d3_keys_are_refused_off_their_step(tmp_path, line, why):
    plan = tmp_path / "p.plan"
    plan.write_text("- prelude\n" + line.format(p=D3.page3_path("olmoe-1b-7b", "wide")) + "\n")
    got = dry("--plan", str(plan))
    assert got.returncode != 0 and why in got.stdout + got.stderr, got.stdout + got.stderr


def test_the_d3_driver_is_the_pinned_decode_driver_with_marked_hunks_only():
    a = PINNED_DRIVER.read_text().splitlines()
    b = DRIVER.read_text().splitlines()
    ha, hb = a.index("set -uo pipefail"), b.index("set -uo pipefail")
    assert b[hb - (ha - 1):hb] == a[1:ha]                     # the pinned header, whole, under the D3 header
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a[ha:], b=b[hb:], autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        assert tag in ("insert", "replace"), (tag, a[ha + i1:ha + i2])
        for ln in b[hb + j1:hb + j2]:
            assert ln.rstrip().endswith("D3"), ln
        if tag == "replace":
            for x, y in zip(a[ha + i1:ha + i2], b[hb + j1:hb + j2]):
                assert "DECODE" in x and "D3" in y
    assert _sha("scripts/gh200_decode_session.sh") == DC.registration(REPO, "cal")["code_pins"]["scripts/gh200_decode_session.sh"]
    assert sum(1 for ln in b[hb:] if ln.rstrip().endswith("D3")) < 40


PLAN_BOX3 = """- prelude
mixtral-8x7b calibrate label=dcal nodrop=1 est=2 cap=5
olmoe-1b-7b r3decode label=dm histogram=docs/registered/2026-10-14-decode-pages/olmoe-1b-7b-main.json inst=GRR slip-policy=cell g1-rule=r7-median-run3 est=2 cap=5
mixtral-8x7b r3decode label=dp7 histogram={label} inst=GRR pad-rows=7 slip-policy=cell g1-rule=r7-median-run3 est=2 cap=5
olmoe-1b-7b decodebytes label=dn histogram=docs/registered/2026-10-14-decode-pages/olmoe-1b-7b-ctl.json est=2 cap=5
olmoe-1b-7b decodebytes label=dw histogram={wide} kernels=wide est=2 cap=5
"""


def test_the_d3_units_reach_their_wrappers_on_the_fake_box(tmp_path):
    import test_gh200_model_session as TG
    import test_stage_decode_plan as TD
    py = TD.fakepy_decode().replace('if tool == "r3_decode":', 'if tool in ("r3_decode", "r3_decode3"):').replace(
        'if tool == "dram_counter_route_decode":', 'if tool in ("dram_counter_route_decode", "dram_counter_route_decode3"):')
    box = TG.Box(tmp_path, {"retracted": 40})
    TG._exe(box.bin / "fakepy", py)
    os.symlink(REPO / "docs", box.repo / "docs")
    (box.repo / ".git" / "info" / "exclude").write_text("results\ndocs\n")
    f = tmp_path / "d3.plan"
    f.write_text(PLAN_BOX3.format(label=D3.page3_path("mixtral-8x7b", "label"), wide=D3.page3_path("olmoe-1b-7b", "wide")))
    got = subprocess.run(["bash", str(DRIVER), "--deadline", str(TG.FAR), "--plan", str(f), "--no-push"],
                         env=box.env(), capture_output=True, text=True, timeout=600)
    led = box.ledger()
    assert got.returncode == exit_codes.DONE, got.stdout[-2000:] + got.stderr[-2000:] + led
    lr = box.tool("locked_r3")
    assert len(lr) == 2
    scripts = [TG.val(a[:a.index("--")], "--r3-script") for a in lr]
    assert scripts == ["scripts/r3_decode.py", "scripts/r3_decode3.py"]
    assert "--pad-rows" not in lr[0] and TG.val(lr[1][lr[1].index("--") + 1:], "--pad-rows") == "7"
    assert len(box.tool("r3_decode", dry=True)) == 1 and len(box.tool("r3_decode3", dry=True)) == 1
    assert TG.val(box.tool("r3_decode3", dry=True)[0], "--pad-rows") == "7"
    plain_c = [a for a in box.tool("dram_counter_route_decode") if "--run" in a]
    wide_c = [a for a in box.tool("dram_counter_route_decode3") if "--run" in a]
    assert len(plain_c) == 1 and "--kernels" not in plain_c[0]
    assert len(wide_c) == 1 and TG.val(wide_c[0], "--kernels") == "wide"
    assert TG.val(wide_c[0], "--out").endswith("-olmoe-1b-7b-dw-r3-counters/lock1710/r3d.json")
    assert "pad rows 7 (scripts/r3_decode3.py)" in led


def test_vm_run_and_lambda_name_the_d3_driver():
    vm = (REPO / "scripts" / "vm_run.sh").read_text()
    assert D3.DRIVER3 in re.search(r"^SESSION_DRIVERS=\((.*)\)$", vm, re.M).group(1)
    assert f"|{D3.DRIVER3}]" in vm.splitlines()[5]
    assert f"pkill -TERM -f '[g]h200_decode3_session.sh'" in (REPO / "docs" / "LAMBDA.md").read_text()


# ---------------------------------------------------------------- the settling pages and the GPU-side wrappers

def test_the_settling_pages_parse_and_hold_the_registered_cells():
    for (m, kind) in D3.PAGE_SPECS:
        page = RD.load_page(REPO / D3.page3_path(m, kind), MODEL_CONFIGS[m], m)
        assert page.shuffle_seed == V4DP.PAGE_SEED and page.cells
    for m, labs in D3.M1_CELLS.items():
        src = {c["label"]: c for c in DC.page_doc(m, "main")["cells"]}
        got = D3.page_doc_any(m, "tiles")["cells"]
        assert [c["label"] for c in got] == list(labs)
        for c in got:
            assert c["counts"] == src[c["label"]]["counts"] and D2.second_tile(m, c) is None
    g = {c["label"]: c for c in D3.page_doc_any("olmoe-1b-7b", "gaps")["cells"]}
    for s in D3.M4_GAPS3:
        assert [r[1] - r[0] for r in g[f"G{8 + s:02d}-R3"]["d3_rows"]] == [8 + s] * 3
    assert g["G15-R1"]["d3_rows"] == [[0, 15]] and g["BASE-A16"]["d3_rows"] == []
    assert all(RD.geometry(MODEL_CONFIGS["olmoe-1b-7b"], 2, c["counts"]).path == RD.NAIVE for c in g.values())
    lab = D3.page_doc_any("mixtral-8x7b", "label")["cells"]
    assert [c["experts"] for c in lab] == [list(e) for _s, e in D3.M3_SETS]
    assert all(sorted(c["counts"], reverse=True)[:2] == [4, 4] and sum(c["counts"]) == 8 for c in lab)


def test_fix_w_splits_the_gap_page_at_the_wave_boundary():
    prm = V4D.DecodeParams(rho_form="scaled", tile=NOM)
    g = {c["label"]: c for c in D3.page_doc_any("olmoe-1b-7b", "gaps")["cells"]}
    fetch = {lab: D3.price_cell_v4dp("olmoe-1b-7b", c, prm)["v4dp"]["naive_order"] for lab, c in g.items() if lab != "BASE-A16"}
    assert fetch["G08-R3"]["w1"]["fetch"] == 0 and fetch["G09-R3"]["w1"]["fetch"] == 0
    assert fetch["G13-R3"]["w1"]["fetch"] == fetch["G13-R3"]["w1"]["followers"] == 96
    assert 0 < fetch["G10-R3"]["w1"]["fetch"] < fetch["G11-R3"]["w1"]["fetch"] < fetch["G12-R3"]["w1"]["fetch"] < 96


def test_r3_decode3_takes_out_its_flag_and_dry_runs_with_the_padding(tmp_path):
    import r3_decode3 as R3D
    assert R3D.split_argv(["--model", "m", "--pad-rows", "7", "--dry-run"]) == (7, ["--model", "m", "--dry-run"])
    assert R3D.split_argv(["--pad-rows=0", "x"]) == (0, ["x"]) and R3D.split_argv(["x"]) == (0, ["x"])
    with pytest.raises(ValueError):
        R3D.split_argv(["--pad-rows", "-1"])
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "r3_decode3.py"), "--pad-rows", "7", "--model",
                          "mixtral-8x7b", "--histogram", str(REPO / D3.page3_path("mixtral-8x7b", "label")), "--inst",
                          "GRR", "--dry-run"], capture_output=True, text=True, timeout=180, cwd=tmp_path)
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    assert "DECODE PAGE" in got.stdout and "pad rows    7 per expert slot" in got.stdout


def test_the_wide_route_dry_runs_and_attributes_the_other_launches_by_place():
    import dram_counter_route as DCR
    got = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "dram_counter_route_decode3.py"), "--dry-run",
                          "--family", "r3-decode", "--model", "olmoe-1b-7b", "--histogram",
                          str(REPO / D3.page3_path("olmoe-1b-7b", "wide")), "--kernels", "wide"],
                         capture_output=True, text=True, timeout=180)
    assert got.returncode == exit_codes.REFUSED and "moe_sum" in got.stdout and "(wide)" in got.stdout
    seq = ["moe_align_block_size_small_batch_expert_kernel", "fused_moe_kernel", "act_and_mul_kernel", "fused_moe_kernel",
           "moe_sum_vec_kernel"] * 2
    launches = [DCR.Launch(str(i), k, {"gpu__time_duration.sum": 1000.0 * (i + 1)}) for i, k in enumerate(seq)]
    gem = [ln for ln in launches if "fused_moe_kernel" in ln.kernel]
    att = [{"key": "native/4/X", "call": c, "gemm": g, "warmup": False, "launch": ln}
           for (c, g), ln in zip(((0, "w1"), (0, "w2"), (1, "w1"), (1, "w2")), gem)]
    got = D3.split_wide(launches, att)
    assert [(o["kernel"][:8], o["call"]) for o in got] == [("moe_alig", 0), ("act_and_", 0), ("moe_sum_", 0),
                                                           ("moe_alig", 1), ("act_and_", 1), ("moe_sum_", 1)]
    import dram_counter_route_decode3 as W
    red = W.reduce_wide(got, {"order": [["native", 4, "X"]]})["native/4/X"]["per_class"]
    assert set(red) == {"align", "silu_and_mul", "moe_sum"} and red["align"]["calls"] == 2


# ---------------------------------------------------------------- the wrappers on synthetic trees

def _truth(m, prm, *, label_us=0.0, label_cells=frozenset(), fixes=V4DP.ALL, extra=None):
    E = MODEL_CONFIGS[m].num_experts

    def f(cell, arm):
        d = {"T": cell.T, "counts": list(cell.counts), "shuffle": cell.shuffle}
        p = D3.price_cell_v4dp(m, d, prm, arm, 9, fixes=fixes)
        ms = p["ms"]
        if arm == RD.NATIVE and cell.label in label_cells:
            ms += label_us * 1e-3
        if extra is not None:
            ms += extra(cell, arm)
        return ms
    return f


def _record_pad(out: Path, pad):
    """As r3_decode3.py does: the page's report.json records its pad rows."""
    rj = out / "report.json"
    doc = json.loads(rj.read_text())
    doc["d3"] = {"pad_rows": pad}
    rj.write_text(json.dumps(doc))


def _power(run: Path, gpu="700.00"):
    (run / "session").mkdir(parents=True, exist_ok=True)
    (run / "session" / "power-set.txt").write_text(POWER_TXT.format(gpu=gpu))


def _mixtral_label_cells() -> set:
    doc = DC.page_doc("mixtral-8x7b", "main")
    return {c["pair"] for c in doc["cells"] if c["role"] == "MINCOVER" and c.get("pair")}


def _counter_page(root: Path, m: str, label: str, kind: str, prm, *, rate_gbps=3700.0, wide=False):
    """A synthetic counter page: counted bytes v4d''s reads, ncu time reads / rate; wide: the align class at
    0.96 us x iterations + 9.7 us, silu_and_mul and moe_sum at 2 us each."""
    rows = []
    for c in D3.page_doc_any(m, kind)["cells"]:
        p = D3.price_cell_v4dp(m, c, prm)
        per = {g: {"dram__bytes_read.sum": p["reads"][g], "gpu__time_duration.sum": p["reads"][g] / rate_gbps,
                   "lts__t_sector_op_read_hit_rate.pct": 1.0} for g in RD.GEMMS}
        row = {"key": f"native/{c['T']}/{c['label']}", "arm": "native", "T": c["T"], "label": c["label"], "per_gemm": per}
        if wide:
            it = D3.align_iterations(c["T"] * MODEL_CONFIGS[m].top_k, MODEL_CONFIGS[m].num_experts)
            row["per_class"] = {"align": {"gpu__time_duration.sum": (0.96 * it + 9.7) * 1e3},
                                "silu_and_mul": {"gpu__time_duration.sum": 2e3}, "moe_sum": {"gpu__time_duration.sum": 2e3}}
            row["kernels"] = ["act_and_mul_kernel", "moe_align_block_size_small_batch_expert_kernel", "moe_sum_vec_kernel"]
        rows.append(row)
    d = root / f"2026-10-29-{F.CARD}-{m}-{label}-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    (d / "r3d.json").write_text(json.dumps({"cells": rows}))


@pytest.fixture(scope="module")
def cal2(tmp_path_factory):
    run = tmp_path_factory.mktemp("d3cal")
    root = run / "results"
    prm = V4D.DecodeParams(**PLANT, rho_form="scaled", tile=NOM)
    reg = DC.registration(REPO, "cal")
    lab = _mixtral_label_cells()
    for u in reg["units"]:
        if u["step"] != "r3decode":
            continue
        m, kind = u["model"], u["page"].rsplit("-", 1)[1][:-5]
        plus = {"GR1": 0.002, "E240": 0.007}.get(u["inst"], 0.0)
        base = _truth(m, prm, label_us=1.8 if (m == "mixtral-8x7b" and kind == "main") else 0.0, label_cells=lab)
        F.write_page(root, m, u["label"], REPO / DC.page_path(m, kind), u["inst"],
                     (lambda c, a, b=base, p=plus: b(c, a) + p), noise=0.001, seed=hash((m, u["label"])) % 1000,
                     copies=9 if kind == "decl" else 1)
    # the settling pages: M3 with the expert-5 sets 2 us slow at both pads, M4 at v4d', the counter pages
    slow = {f"{s}-T4" for s, e in D3.M3_SETS if 5 in e}
    for pad in D3.M3_PADS:
        out = F.write_page(root, "mixtral-8x7b", f"dp{pad}", REPO / D3.page3_path("mixtral-8x7b", "label"), "GRR",
                           _truth("mixtral-8x7b", prm, label_us=2.0, label_cells=slow), noise=0.0005, seed=7)
        _record_pad(out, pad)
    F.write_page(root, "olmoe-1b-7b", "dg", REPO / D3.page3_path("olmoe-1b-7b", "gaps"), "GRR",
                 _truth("olmoe-1b-7b", prm), noise=0.0005, seed=8)
    bare = V4D.DecodeParams(tile=NOM, rho_form="scaled")
    for m in DC.CAL_MODELS:
        _counter_page(root, m, "dt", "tiles", bare)
        _counter_page(root, m, "dw", "wide", bare, wide=True)
    _counter_page(root, "olmoe-1b-7b", "dgn", "gaps", bare)
    for m in DC.CAL_MODELS:                     # the registered dn pages (BYTES-D, BYTES-D2, the constants' replicate)
        _counter_page(root, m, "dn", "ctl", bare)
    _power(run)
    return root, SC3.score(root, reg, NOM, "planted NOM"), SC2.score(root, reg, NOM, "planted NOM")


def test_cal2_reads_go_with_the_label_effect_descriptive_and_recovers_the_constants(cal2):
    _root, s, d2 = cal2
    assert d2["STOP"]["verdict"] == "STOP" and any(w.startswith("G3-D label") for w in d2["STOP"]["why"])
    assert s["STOP"]["verdict"] == "GO", s["STOP"]
    assert s["STOP"]["descriptive"] and s["STOP"]["descriptive"][0].startswith("G3-D label")
    assert s["STOP_D2"]["verdict"] == "STOP" and s["G3D_LABEL"]["verdict"] == "NOT EQUIVALENT"
    assert s["G3D_LABEL"]["per_model"]["mixtral-8x7b"]["mean_us"] == pytest.approx(-1.8, abs=0.3)
    f = s["FIT"]["forms"][s["FIT"]["selected"]]
    assert s["FIT"]["selected"] == "scaled" and f["T0_GR_ms"] == pytest.approx(PLANT["T0_GR_ms"], abs=3e-4)
    assert s["E1_CAL_IN_SAMPLE"]["verdict"] == "HOLDS" and s["E1_CAL_IN_SAMPLE"]["stats"]["cells"] == 92
    assert s["model"]["fixes"] == V4DP.ALL.as_dict() and s["addendum"] == D3.NAME
    assert s["scope"] == D3.SCOPE and D3.SCOPE in "\n".join(SC3.lines(s))                   # review m4
    assert s["V4D_READING"]["E1_CAL_IN_SAMPLE"]["stats"]["rms"] > s["E1_CAL_IN_SAMPLE"]["stats"]["rms"]
    text = "\n".join(SC3.lines(s))
    assert "addendum 3" in text and "STOP (addendum 3, G3-D label DESCRIPTIVE): GO" in text and "G3-D label (DESCRIPTIVE)" in text


def test_cal2_reads_the_settling_units(cal2):
    _root, s, _d2 = cal2
    st = s["SETTLE"]
    m1 = st["M1"]
    assert m1["verdict"] == "DESCRIPTIVE" and set(m1["pages"]) == set(DC.CAL_MODELS) and m1["pairs"]
    for x in m1["pairs"]:
        assert x["w1"]["counter_GBps"] == pytest.approx(3700.0, rel=1e-9)
    m2 = st["M2"]["models"]
    for m in DC.CAL_MODELS:
        assert m2[m]["align_fit"]["slope_us"] == pytest.approx(0.96, abs=1e-6)
    assert st["M3"]["verdict"] == "FOLLOWS-EXPERT" and st["M3"]["d0_us"] == pytest.approx(2.0, abs=0.3)
    assert st["M3"]["pads"] == [0, 52]
    m4 = st["M4"]
    assert m4["agree"].startswith("14 of 14") and len(m4["time"]) == 8
    assert all(abs(r["e_v4dp"]) < 0.01 for r in m4["time"])
    assert any(abs(r["e_dedup"]) > 0.005 for r in m4["time"] if r["label"] in ("G12-R3", "G13-R3", "G15-R1"))
    rep = s["CONSTANTS_REPLICATE"]
    assert rep["R"]["verdict"] == "READ" and rep["R"]["counter_GBps"] == pytest.approx(3700.0, rel=1e-9)
    assert all(rep["A"][m]["verdict"] == "READ" for m in DC.CAL_MODELS)


def test_finalise_d3_refuses_other_scores_and_freezes_v4dp(cal2, tmp_path):
    _root, s, d2 = cal2
    (tmp_path / "cal.score.json").write_text(json.dumps(d2, default=str))
    with pytest.raises(FN3.Refused, match="score_cal_d3"):
        FN3.finalise(tmp_path)
    with pytest.raises(FN2.Refused):
        FN2.finalise(tmp_path)          # addendum 2's finalise refuses it too: its stop is a STOP
    for bad, why in ((dict(s, model=dict(s["model"], fixes=V4DP.NONE.as_dict())), "fixes"),
                     (dict(s, POWER={"verdict": "NOT SCORED: x"}), "power"),
                     (dict(s, STOP={"verdict": "STOP", "why": ["x"]}), "STOP")):
        (tmp_path / "cal.score.json").write_text(json.dumps(bad, default=str))
        with pytest.raises(FN3.Refused, match=why):
            FN3.finalise(tmp_path)
    (tmp_path / "cal.score.json").write_text(json.dumps(s, default=str))
    doc = FN3.finalise(tmp_path)
    assert doc["addendum"] == D3.NAME and doc["frozen"]["model"]["fixes"] == V4DP.ALL.as_dict() and doc["not_priced"] == 0
    assert doc["frozen"]["bars"] == s["REP"]["bars_blind"] and len(doc["descriptive_cells"]) == 27
    assert doc["scope"] == D3.SCOPE and "never voids" in doc["frozen"]["g3d_label"]
    prm = FN3.FN.params_of(doc["frozen"])
    cells = {c["label"]: c for c in DC.page_doc("qwen1.5-moe-a2.7b", "main")["cells"]}
    for r in doc["cells"]:
        if r["model"] == "qwen1.5-moe-a2.7b" and r["path"] == RD.NAIVE:
            assert r["ms"]["v4d"] == D3.price_cell_v4dp(r["model"], cells[r["label"]], prm)["ms"]


def test_the_blind_wrapper_certifies_v4dp_truth_and_refuses_a_d2_finalisation(cal2, tmp_path):
    _root, s, _d2 = cal2
    (tmp_path / "cal.score.json").write_text(json.dumps(s, default=str))
    fin = json.loads(json.dumps(FN3.finalise(tmp_path), default=str))
    prm = V4D.DecodeParams(**PLANT, rho_form="scaled", tile=NOM)
    root = tmp_path / "blind" / "results"
    reg = DC.registration(REPO, "blind")
    for u in reg["units"]:
        m, kind = u["model"], u["page"].rsplit("-", 1)[1][:-5]
        plus = {"GR1": 0.002}.get(u["inst"], 0.0)
        base = _truth(m, prm)
        F.write_page(root, m, u["label"], REPO / DC.page_path(m, kind), u["inst"], (lambda c, a, b=base, p=plus: b(c, a) + p),
                     noise=0.001, seed=hash((m, u["label"])) % 1000)
    _power(tmp_path / "blind")
    got = SB3.score(root, reg, fin)
    assert got["E1_DEC"]["pooled"]["verdict"] == "HOLDS", got["E1_DEC"]
    assert set(got["certified"].values()) == {"CERTIFIED"} and got["addendum"] == D3.NAME
    assert "addendum 3" in SB3.lines(got)[0]
    (tmp_path / "fin.json").write_text(json.dumps(dict(fin, addendum=D2.NAME)))
    assert SB3.main([str(root), str(tmp_path / "fin.json"), str(tmp_path / "o")]) == 2


def test_the_blind_label_effect_is_reported_and_never_voids(cal2, tmp_path):
    """Owner, 2026-11-03: a planted Phi-3.5 expert-set effect (2 us on its G3-D label partner cells) voids Phi under
    addendum 2's blind wrapper; under addendum 3 the label test reads NOT EQUIVALENT, is reported, and Phi is CERTIFIED."""
    import score_blind_d2 as SB2
    _root, s, _d2 = cal2
    (tmp_path / "cal.score.json").write_text(json.dumps(s, default=str))
    fin = json.loads(json.dumps(FN3.finalise(tmp_path), default=str))
    prm = V4D.DecodeParams(**PLANT, rho_form="scaled", tile=NOM)
    root = tmp_path / "blind" / "results"
    reg = DC.registration(REPO, "blind")
    doc = DC.page_doc("phi-3.5-moe", "main")
    partners = {c["pair"] for c in doc["cells"] if c["role"] == "MINCOVER" and c.get("pair")}
    assert partners
    for u in reg["units"]:
        m, kind = u["model"], u["page"].rsplit("-", 1)[1][:-5]
        plus = {"GR1": 0.002}.get(u["inst"], 0.0)
        base = _truth(m, prm, label_us=2.0 if (m == "phi-3.5-moe" and kind == "main") else 0.0, label_cells=partners)
        F.write_page(root, m, u["label"], REPO / DC.page_path(m, kind), u["inst"], (lambda c, a, b=base, p=plus: b(c, a) + p),
                     noise=0.001, seed=hash((m, u["label"])) % 1000)
    _power(tmp_path / "blind")
    with D3.v4dp(D3.fixes_from(fin["frozen"]["model"]["fixes"])):
        old = SB2.score(root, reg, fin)
    assert old["certified"]["phi-3.5-moe"] == "VOID: G3-D"
    got = SB3.score(root, reg, fin)
    assert got["G3D_LABEL"]["phi-3.5-moe"]["verdict"] == "NOT EQUIVALENT"
    assert got["G3D_LABEL"]["phi-3.5-moe"]["mean_us"] == pytest.approx(-2.0, abs=0.4)
    assert got["G3D"]["phi-3.5-moe"]["label"]["verdict"] == "NOT EQUIVALENT"
    assert "phi-3.5-moe" not in got["voided"] and got["certified"]["phi-3.5-moe"] == "CERTIFIED", got["E1_DEC"]
    assert set(got["certified"].values()) == {"CERTIFIED"} and got["scope"] == D3.SCOPE
    text = "\n".join(SB3.lines(got))
    assert "G3-D label phi-3.5-moe (DESCRIPTIVE, never a void): NOT EQUIVALENT" in text and D3.SCOPE in text
    assert DC.g3d.__name__ == "g3d" and DC.g3d.__module__ == "dcommon"          # restored


# ---------------------------------------------------------------- house rules

def test_no_em_dash_and_no_trailer_in_the_addendum_files():
    paths = [REPO / p for p in RG3.D3_CODE] + [REG / f"{D3.NAME}.{x}" for x in ("json", "txt")] + [Path(__file__)]
    paths += [REPO / D3.page3_path(m, k) for m, k in D3.PAGE_SPECS]
    for p in paths:
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
        assert not re.search("|".join(("Co-" + "Authored-By", "Claude " + "Code", "Generated " + "with", "Anth" + "ropic")), t), p


@pytest.mark.parametrize("recorded, why", [(None, "records pad rows [None]"), (7, "records pad rows [7]")])
def test_m3_refuses_a_shifted_page_that_does_not_record_its_registered_pad(cal2, tmp_path, recorded, why):
    """Review m1: a lost --pad-rows would run dp52 unpadded and read a false FOLLOWS-EXPERT."""
    import shutil
    root, _s, _d2 = cal2
    for lab in ("dp0", "dp52"):
        src = root / f"gaps-{F.CARD}-mixtral-8x7b-{lab}"
        shutil.copytree(src, tmp_path / src.name)
    rj = next((tmp_path / f"gaps-{F.CARD}-mixtral-8x7b-dp52").rglob("report.json"))
    doc = json.loads(rj.read_text())
    if recorded is None:
        doc.pop("d3", None)
    else:
        doc["d3"] = {"pad_rows": recorded}
    rj.write_text(json.dumps(doc))
    got = D3.read_m3(tmp_path)
    assert got["verdict"].startswith("NOT READ: dp52") and why in got["verdict"]
    assert D3.read_m3(root)["verdict"] == "FOLLOWS-EXPERT"
