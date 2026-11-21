"""The decode stage's second addendum (docs/registered/2026-10-28-decode-addendum2-gh200), off the GPU: the
registration recomputes and pins every frozen decode file; the DESCRIPTIVE cells are the second-tile cells by
the registered rule; the plan is the frozen plan with the two dn units nodrop; the L2-served estimate; the
power read-back; and the wrappers on SYNTHETIC results trees (decode_fakes) where the second-tile cells run
at the served estimate: the frozen scorer STOPs there, the addendum reads GO, recovers the planted
constants, certifies the blind models and reads SIGMA-D at the served end. With no cell left out the CAL
wrapper reads what the frozen scorer reads."""
from __future__ import annotations

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
import dcommon as DC  # noqa: E402
import decode_fakes as F  # noqa: E402
import finalise_decode as FN  # noqa: E402
import finalise_decode_d2 as FN2  # noqa: E402
import register_decode_d2 as RG2  # noqa: E402
import score_blind as SB  # noqa: E402
import score_blind_d2 as SB2  # noqa: E402
import score_cal as SC  # noqa: E402
import score_cal_d2 as SC2  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

V4, V4D, RD = DC.V4, DC.V4D, DC.RD
REG = REPO / "docs" / "registered"
NOM = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0})
PLANT = dict(T0_GR_ms=0.028, s_naive_GR_ms=-0.003, s_block_GR_ms=0.004)
CAL_MAIN = {
    "olmoe-1b-7b": ["L-T32-A8", "L-T64-A8", "MIN-T64", "L-T96-A8", "L-T128-A8"],
    "mixtral-8x7b": ["L-T32-A2", "L-T64-A2", "MIN-T64", "L-T96-A2", "L-T96-A4", "L-T128-A2"],
}
POWER_TXT = """GPU 00000000:DD:00.0
    GPU Power Readings
        Average Power Draw                : 1.00 W
        Current Power Limit               : {gpu} W
        Default Power Limit               : 900.00 W
    Module Power Readings
        Current Power Limit               : 1000.00 W
"""


def _sha(p) -> str:
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- registration

def test_the_addendum_recomputes_to_its_committed_files(capsys):
    assert RG2.main(["--check"]) == 0, capsys.readouterr().out


def test_the_addendum_pins_every_frozen_decode_file_and_itself_at_their_bytes():
    pins = json.loads((REG / f"{D2.NAME}.json").read_text())["code_pins"]
    for p, s in pins.items():
        assert _sha(p) == s, p
    for name in (DC.NAMES["cal"], DC.NAMES["blind"], RG2.TILESOURCE):
        for x in ("json", "txt"):
            assert f"docs/registered/{name}.{x}" in pins
        for p, s in json.loads((REG / f"{name}.json").read_text())["code_pins"].items():
            assert pins.get(p) == s, p
    for p in RG2.D2_CODE + ("scripts/scoring/decode/score_cal.py", "scripts/scoring/decode/score_blind.py",
                            "scripts/scoring/tileA2/floor.score.json", FN.CAL_PLAN):
        assert p in pins, p


def test_the_descriptive_cells_are_the_second_tile_cells_by_identity():
    d = json.loads((REG / f"{D2.NAME}.json").read_text())
    got = {m: [r["label"] for r in d["excluded_cells"]["cal main"] if r["model"] == m] for m in DC.CAL_MODELS}
    assert got == CAL_MAIN
    assert d["descriptive cells (owner, 2026-10-28, D2-2)"]["cal_main"] == 11
    assert d["descriptive cells (owner, 2026-10-28, D2-2)"]["cal_main_of"] == 103
    assert len(d["excluded_cells"]["blind main"]) == 27
    from moe.spec import MODEL_CONFIGS
    for m in DC.CAL_MODELS + DC.BLIND_MODELS:
        bm = lambda T, m=m: int(RD.default_config(T, MODEL_CONFIGS[m].num_experts)["BLOCK_SIZE_M"])  # noqa: E731
        for c in DC.page_doc(m, "main")["cells"]:
            g = RD.geometry(MODEL_CONFIGS[m], c["T"], c["counts"])
            tiles = sum(-(-v // bm(c["T"])) for v in c["counts"] if v)
            assert (D2.second_tile(m, c) is not None) == (g.path != RD.NAIVE and tiles > g.A), (m, c["label"])
    txt = (REG / f"{D2.NAME}.txt").read_text()
    assert "MIN-T64 (T 64, 32x64x128w4s3" in txt and "deadline launch + 165 min" in txt
    assert "decodeC-cal-gh200" in txt and "700 W" in txt


def test_the_rentals_state_minutes_wall_cost_deadline_and_cap():
    r = json.loads((REG / f"{D2.NAME}.json").read_text())["rentals (owner, 2026-10-28, D2-5)"]
    assert (r["cal"]["units_min"], r["cal"]["wall_min"], r["cal"]["cost_usd"]) == (106, 151, 5.76)
    assert (r["cal"]["deadline_after_launch_min"], r["cal"]["hard_cap_after_launch_min"]) == (165, 180)
    assert (r["blind"]["units_min"], r["blind"]["wall_min"], r["blind"]["cost_usd"]) == (101, 145, 5.53)
    assert (r["blind"]["deadline_after_launch_min"], r["blind"]["hard_cap_after_launch_min"]) == (160, 175)
    for x in (r["cal"], r["blind"]):
        assert x["cost_usd"] == round(x["wall_min"] / 60 * 2.29, 2) and x["boot_setup_min"] == 25


# ---------------------------------------------------------------- the plan

def test_the_plan_is_the_frozen_plan_with_the_two_dn_units_nodrop():
    new = (REPO / D2.CAL_PLAN_D2).read_text()
    assert new == D2.cal_plan_d2()
    a = [ln for ln in (REPO / FN.CAL_PLAN).read_text().splitlines() if not ln.startswith("#")]
    b = [ln for ln in new.splitlines() if not ln.startswith("#")]
    assert len(a) == len(b)
    moved = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(moved) == 2
    for x, y in moved:
        assert " decodebytes label=dn " in x and x.replace("drop-group=ncu", "nodrop=1") == y
    assert "DD6 superseded" in new and "rental B" in new


def test_the_plan_parses_in_106_minutes_with_the_dn_units_nodrop():
    got = subprocess.run(["bash", str(REPO / "scripts" / "gh200_decode_session.sh"), "--dry-run", "--plan",
                          str(REPO / D2.CAL_PLAN_D2)], capture_output=True, text=True, timeout=60,
                         env={**os.environ, "HOME": "/nonexistent"})
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    assert "a PLAN of 13 units" in got.stdout and "106 min of units" in got.stdout
    dn = [ln for ln in got.stdout.splitlines() if " decodebytes " in ln and "label=dn" in ln]
    assert len(dn) == 2 and all("[nodrop]" in ln for ln in dn)
    order = re.search(r"drop order when time runs short: .*\n\s+([\d ]+)", got.stdout).group(1).split()
    assert order == ["13", "12", "11", "8"]


# ---------------------------------------------------------------- the served estimate and the power check

def test_the_served_estimate_removes_only_the_second_tile_slabs():
    prm = V4D.DecodeParams(T0_GR_ms=0.025, tile=NOM)
    for m in DC.CAL_MODELS:
        for c in DC.page_doc(m, "main")["cells"]:
            a, b = DC.price_cell(m, c, prm), D2.served_price(m, c, prm)
            if D2.second_tile(m, c) is None:
                assert b["ms"] == a["ms"] and b["reads"] == a["reads"]
                continue
            assert b["ms"] < a["ms"] and b["live_ctas"] == a["live_ctas"]
            ra, rb = sum(a["reads"].values()), sum(b["reads"].values())
            assert rb < ra and 1.0 <= rb / a["weight_bytes"] < 1.1 and ra / a["weight_bytes"] >= 2.0
    assert DC.V4.TM.window.__name__ == "window" and hasattr(DC.V4.TM.window, "cache_info")    # restored


@pytest.mark.parametrize("gpu, ok", [("700.00", True), ("700.40", True), ("900.00", False), ("650.00", False)])
def test_the_power_check_reads_the_gpu_scope_limit(tmp_path, gpu, ok):
    tree = tmp_path / "run" / "results"
    tree.mkdir(parents=True)
    (tmp_path / "run" / "session").mkdir()
    (tmp_path / "run" / "session" / "power-set.txt").write_text(POWER_TXT.format(gpu=gpu))
    got = D2.power_check(tree)
    assert (got["verdict"] == "READ") == ok, got
    assert got["gpu_current_power_limit_w"] == float(gpu)


def test_no_power_file_is_not_scored_and_a_stop(tmp_path):
    s = SC2.score(tmp_path, DC.registration(REPO, "cal"), NOM, "planted")
    assert s["verdict"] == "NOT SCORED" and s["STOP"]["verdict"] == "STOP" and "power" in s["STOP"]["why"][0]
    assert "NOT SCORED" in SC2.lines(s)[0]
    (tmp_path / "cal.score.json").write_text(json.dumps(s))
    with pytest.raises(FN2.Refused, match="STOP"):
        FN2.finalise(tmp_path)


# ---------------------------------------------------------------- the wrappers on synthetic trees

def _truth(m, prm, served_keys, plus_ms=0.0):
    """v4d at prm (DECL at 9 copies), the served estimate on the (model, label) cells in served_keys."""
    from moe.spec import MODEL_CONFIGS

    def f(cell, arm):
        if arm == RD.NATIVE and (m, cell.label) in served_keys:
            with D2._served_windows():
                p = V4D.price(m, cell.T, cell.counts, prm)
        else:
            p = V4D.price(m, cell.T, cell.counts, prm,
                          declared=None if arm == RD.NATIVE else MODEL_CONFIGS[m].num_experts * 9)
        return p["ms"] + plus_ms
    return f


def _power(run: Path, gpu="700.00"):
    (run / "session").mkdir(parents=True, exist_ok=True)
    (run / "session" / "power-set.txt").write_text(POWER_TXT.format(gpu=gpu))


def _write(run: Path, part: str, prm, served_models=(), bias=None):
    """Every r3decode page of a registration; on served_models the second-tile cells (main and ctl) run at the
    served estimate, the rest at v4d (sigma 1); bias(cell) multiplies the truth of the other cells."""
    reg = DC.registration(REPO, part)
    root = run / "results"
    for u in reg["units"]:
        if u["step"] != "r3decode":
            continue
        m, kind = u["model"], u["page"].rsplit("-", 1)[1][:-5]
        keys = (D2.excluded_keys([m]) | D2.excluded_keys([m], "ctl")) if m in served_models else set()
        plus = {"GR1": 0.002, "E240": 0.007}.get(u["inst"], 0.0)
        base = _truth(m, prm, keys)

        def truth(c, a, base=base, plus=plus, keys=keys):
            k = 1.0 if bias is None or (m, c.label) in keys else bias(c)
            return base(c, a) * k + plus
        F.write_page(root, m, u["label"], REPO / DC.page_path(m, kind), u["inst"], truth,
                     noise=0.001, seed=hash((m, u["label"])) % 1000, copies=9 if kind == "decl" else 1)
    _power(run)
    return root


@pytest.fixture(scope="module")
def cal_served(tmp_path_factory):
    run = tmp_path_factory.mktemp("d2cal")
    prm = V4D.DecodeParams(**PLANT, rho_form="scaled", tile=NOM)
    root = _write(run, "cal", prm, DC.CAL_MODELS)
    reg = DC.registration(REPO, "cal")
    return root, SC.score(root, reg, NOM, "planted NOM"), SC2.score(root, reg, NOM, "planted NOM")


def test_the_frozen_scorer_stops_on_the_served_cells_and_the_addendum_reads_go(cal_served):
    _root, full, s = cal_served
    assert full["STOP"]["verdict"] == "STOP" and full["E1_CAL_IN_SAMPLE"]["verdict"] == "FAILS"
    assert s["STOP"] == {"verdict": "GO", "why": []}, s["STOP"]
    assert s["REGISTERED_READING"]["STOP"]["verdict"] == "STOP"
    got = s["FIT"]["forms"][s["FIT"]["selected"]]
    assert s["FIT"]["selected"] == "scaled"
    assert got["T0_GR_ms"] == pytest.approx(PLANT["T0_GR_ms"], abs=2e-4)
    assert got["s_block_GR_ms"] == pytest.approx(PLANT["s_block_GR_ms"], abs=5e-4)
    assert s["E1_CAL_IN_SAMPLE"]["verdict"] == "HOLDS" and s["E1_CAL_IN_SAMPLE"]["stats"]["cells"] == 92
    assert not {tuple(r["key"].split("/")[0:3:2]) for r in s["rows"]} & D2.excluded_keys(DC.CAL_MODELS)
    assert {t: s["G3D"][t]["verdict"] for t in ("shuffle", "label")} == {"shuffle": "EQUIVALENT", "label": "EQUIVALENT"}
    assert any(r["a"] == "MIN-T64" for r in s["G3D"]["label"]["rows"])
    assert s["GD2"] == full["GD2"] and s["POWER"]["verdict"] == "READ" and s["scope"] == D2.SCOPE
    assert {k: v for k, v in s["REP_ALL_CELLS"].items() if k != "label"} == full["REP"]
    assert s["REP"]["pairs"] == full["REP"]["pairs"] - 5           # the replicate's five DESCRIPTIVE cells leave
    assert all(r["band"] == "L2-SERVED" for r in s["SIGMA_D"]["rows"])
    assert {m: x["band"] for m, x in s["SIGMA_D"]["per_model"].items()} == {m: "L2-SERVED" for m in DC.CAL_MODELS}


def test_sigma_d_reads_the_served_end_and_the_knee_rows_sit_beside(cal_served):
    _root, _full, s = cal_served
    rows = s["SIGMA_D"]["rows"]
    assert {(r["model"], r["label"]) for r in rows} == D2.excluded_keys(DC.CAL_MODELS)
    for r in rows:
        assert r["position"] == pytest.approx(1.0, abs=0.05) and abs(r["e_served"]) < 0.01 and r["e_sigma1"] > 0.2
    for m, x in s["SIGMA_D"]["per_model"].items():
        assert x["cells"] == len(CAL_MAIN[m])
    assert set(s["KNEE_D"]) == {"p 5.75", "p 3.75"}
    assert all(x["label"].startswith("DESCRIPTIVE") and x["stats"]["cells"] == 92 for x in s["KNEE_D"].values())
    assert "tile rental A" not in s["tile_params"]
    text = "\n".join(SC2.lines(s))
    assert "addendum 2" in text and "SIGMA-D olmoe-1b-7b: L2-SERVED (median position" in text and "5 of 5 cells read" in text and "knee p 5.75" in text


def test_with_no_cell_left_out_the_wrapper_reads_what_the_frozen_scorer_reads(cal_served):
    root, full, _s = cal_served
    s = SC2.score(root, DC.registration(REPO, "cal"), NOM, "planted NOM", excluded=set())
    s["REP"] = {k: v for k, v in s["REP"].items() if k != "label"}
    for k in ("FIT", "E1_CAL_IN_SAMPLE", "GD2", "GD3", "REP", "rows", "RIVAL_FITS", "NOT_PRICED", "DECL", "STOP"):
        assert json.dumps(s[k], sort_keys=True, default=str) == json.dumps(full[k], sort_keys=True, default=str), k
    for t in ("shuffle", "label", "fill"):
        assert s["G3D"][t]["verdict"] == full["G3D"][t]["verdict"] and s["G3D"][t]["rows"] == full["G3D"][t]["rows"]


def test_finalise_d2_refuses_a_frozen_score_and_freezes_the_amended_one(cal_served, tmp_path):
    _root, full, s = cal_served
    (tmp_path / "cal.score.json").write_text(json.dumps(full, default=str))
    with pytest.raises(FN2.Refused, match="score_cal_d2"):
        FN2.finalise(tmp_path)
    (tmp_path / "cal.score.json").write_text(json.dumps(s, default=str))
    doc = FN2.finalise(tmp_path)
    assert doc["addendum"] == D2.NAME and doc["frozen"]["rho_form"] == "scaled" and doc["not_priced"] == 0
    assert doc["frozen"]["bars"] == s["REP"]["bars_blind"] and doc["frozen"]["sigma_page"] == s["REP"]["sigma_page"]
    assert doc["scope"] == D2.SCOPE
    for bad, why in ((dict(s, POWER={"verdict": "NOT SCORED: x"}), "power check"),
                     (dict(s, excluded_cells=s["excluded_cells"][1:]), "excluded cells")):
        (tmp_path / "cal.score.json").write_text(json.dumps(bad, default=str))
        with pytest.raises(FN2.Refused, match=why):
            FN2.finalise(tmp_path)
    assert len(doc["descriptive_cells"]) == 27
    assert {(r["model"], r["label"]) for r in doc["cells"]} >= D2.excluded_keys(DC.BLIND_MODELS)


def test_the_blind_wrapper_certifies_the_models_the_frozen_scorer_fails_on_the_served_cells(cal_served, tmp_path):
    _root, _full, s = cal_served
    (tmp_path / "cal.score.json").write_text(json.dumps(s, default=str))
    fin = json.loads(json.dumps(FN2.finalise(tmp_path), default=str))
    prm = V4D.DecodeParams(**PLANT, rho_form="scaled", tile=NOM)
    root = _write(tmp_path / "blind", "blind", prm, DC.BLIND_MODELS)
    reg = DC.registration(REPO, "blind")
    full = SB.score(root, reg, fin)
    assert full["E1_DEC"]["pooled"]["verdict"] == "FAILS"
    got = SB2.score(root, reg, fin)
    assert got["E1_DEC"]["pooled"]["verdict"] == "HOLDS", got["E1_DEC"]
    assert set(got["certified"].values()) == {"CERTIFIED"}
    assert got["REGISTERED_READING"]["E1_DEC"]["pooled"]["verdict"] == "FAILS"
    assert {(r["model"], r["label"]) for r in got["SIGMA_D"]["rows"]} == D2.excluded_keys(DC.BLIND_MODELS)
    assert all(r["position"] == pytest.approx(1.0, abs=0.05) for r in got["SIGMA_D"]["rows"])
    assert not {(r["model"], r["label"]) for r in got["rows"]} & D2.excluded_keys(DC.BLIND_MODELS)
    text = "\n".join(SB2.lines(got))
    assert got["scope"] == D2.SCOPE and text.count(D2.SCOPE) == 5 and "CERTIFIED\n    Certified domain" in text
    assert set(got["RIVALS_ALL_CELLS"]) == set(SB.RIVALS) | {"label"}
    assert "27 descriptive cells out of E1-DEC" in text and "SIGMA-D granite-3.0-3b-a800m: L2-SERVED" in text and "6 of 6 cells read" in text
    (tmp_path / "fin.json").write_text(json.dumps(fin))
    assert SB2.main([str(root), str(tmp_path / "cal.score.json"), str(tmp_path / "o")]) == 2       # not a finalised file
    _power(tmp_path / "blind", gpu="900.00")
    assert SB2.score(root, reg, fin)["verdict"] == "NOT SCORED"


def test_bytes_d2_reads_the_counted_second_tile_bytes_against_both_read_totals(tmp_path):
    m = "olmoe-1b-7b"
    cells = {c["label"]: c for c in DC.page_doc(m, "ctl")["cells"]}
    bare = V4D.DecodeParams(tile=NOM)
    rows = []
    for lab, c in cells.items():
        a, b = V4D.price(m, c["T"], c["counts"], bare, with_constants=False), D2.served_price(m, c, bare)
        vm = {"error": "Refused: v4 has no F_wg yet"} if c["T"] > 96 else a["reads"]
        rows.append({"key": f"native/{c['T']}/{lab}", "arm": "native", "T": c["T"], "label": lab, "v4d_reads": vm,
                     "per_gemm": {g: {"dram__bytes_read.sum": b["reads"][g]} for g in RD.GEMMS}})
    d = tmp_path / f"2026-10-29-{F.CARD}-{m}-dn-r3-counters" / "lock1710"
    d.mkdir(parents=True)
    (d / "r3d.json").write_text(json.dumps({"cells": rows}))
    got = SC2.bytes_d2(tmp_path, NOM)["rows"]
    assert {r["cell"].split("/")[1] for r in got} == {lab for (_m, lab) in D2.excluded_keys([m], "ctl")}
    assert len(got) == 8 and all(r["share_not_seen"] == pytest.approx(1.0) and r["counted_over_v4d"] < 0.6 for r in got)
    t128 = [r for r in got if "/L-T128-" in r["cell"]]
    assert len(t128) == 2 and all(str(r["vm_v4d_reads"]).startswith("error") and r["v4d_reads"] > 0 for r in t128)


def test_sigma_d_separates_served_from_full_rereads_on_a_mixed_truth(tmp_path):
    """The readiness review's expectation (build review m4): OLMoE's second tiles served, Mixtral's at sigma 1."""
    prm = V4D.DecodeParams(**PLANT, rho_form="scaled", tile=NOM)
    root = _write(tmp_path, "cal", prm, ("olmoe-1b-7b",))
    s = SC2.score(root, DC.registration(REPO, "cal"), NOM, "planted NOM")
    assert s["STOP"]["verdict"] == "GO", s["STOP"]
    pm = s["SIGMA_D"]["per_model"]
    assert pm["olmoe-1b-7b"]["band"] == "L2-SERVED" and pm["olmoe-1b-7b"]["position_median"] == pytest.approx(1.0, abs=0.05)
    assert pm["mixtral-8x7b"]["band"] == "FULL-REREAD" and pm["mixtral-8x7b"]["position_median"] == pytest.approx(0.0, abs=0.05)


def test_the_addendum_still_stops_on_a_kept_cell_defect(tmp_path):
    """Served second tiles and 15% on the kept T 4 and T 8 cells (build review m4): E1 FAILS, a STOP."""
    prm = V4D.DecodeParams(**PLANT, rho_form="scaled", tile=NOM)
    root = _write(tmp_path, "cal", prm, DC.CAL_MODELS, bias=lambda c: 1.15 if c.T in (4, 8) else 1.0)
    s = SC2.score(root, DC.registration(REPO, "cal"), NOM, "planted NOM")
    assert s["E1_CAL_IN_SAMPLE"]["verdict"] == "FAILS" and s["STOP"]["verdict"] == "STOP"
    assert any("E1-DEC" in w for w in s["STOP"]["why"])


@pytest.mark.parametrize("pos, gap_rel, gap_us, want", [
    (0.1, 0.4, 30.0, "FULL-REREAD"), (0.5, 0.4, 30.0, "MIXED"), (0.9, 0.4, 30.0, "L2-SERVED"),
    (-0.2, 0.4, 30.0, "FULL-REREAD"), (1.3, 0.4, 30.0, "L2-SERVED"), (0.9, 0.05, 30.0, "NOT READ"), (0.9, 0.4, 2.0, "NOT READ")])
def test_the_sigma_d_bands_and_the_not_read_rule(pos, gap_rel, gap_us, want):
    assert D2.band(pos, gap_rel, gap_us).startswith(want)


def test_the_operator_power_step_reads_a_published_session(tmp_path, capsys):
    run = tmp_path / "run"
    _power(run)
    (run / "results").mkdir()
    assert D2.main(["--power", str(run / "results")]) == 0 and capsys.readouterr().out.strip().endswith("READ")
    _power(run, gpu="900.00")
    assert D2.main(["--power", str(run / "results")]) == 1
    d = json.loads((REG / f"{D2.NAME}.json").read_text())
    assert "d2common.py --power" in d["rentals (owner, 2026-10-28, D2-5)"]["operator power step (review m2)"]
    for p in ("moe/spec.py", "moe/bench/tile_resolve.py"):
        assert d["code_pins"][p] == _sha(p)
    assert d["scope (owner, 2026-10-28, review M1)"] == D2.SCOPE and D2.SCOPE in (REG / f"{D2.NAME}.txt").read_text()


# ---------------------------------------------------------------- house rules

def test_no_em_dash_and_no_trailer_in_the_addendum_files():
    paths = [REPO / p for p in RG2.D2_CODE] + [REG / f"{D2.NAME}.{x}" for x in ("json", "txt")] + [Path(__file__)]
    for p in paths:
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
        assert not re.search("|".join(("Co-" + "Authored-By", "Claude " + "Code", "Generated " + "with")), t), p
