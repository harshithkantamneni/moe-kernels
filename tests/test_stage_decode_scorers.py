"""The decode scorers on SYNTHETIC pages only (no decode page exists): results trees written by the real
page writer (r3_decode.run_page and write_report, through tests/decode_fakes.FakeBackend) with times
planted from v4d at known constants. The CAL scorer must recover the constants and the rho form, read
G3-D, GD2 and the replicate, and say GO; the finalisation must freeze them and write the blind plan; the
blind scorer must certify a model v4d prices and not one it misprices; every planted defect is a STOP,
and the finalisation refuses a STOP."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "decode"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dcommon as DC  # noqa: E402
import decode_fakes as F  # noqa: E402
import finalise_decode as FN  # noqa: E402
import score_blind as SB  # noqa: E402
import score_cal as SC  # noqa: E402

V4, V4D, RD = DC.V4, DC.V4D, DC.RD
NOM = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0})
PLANT = dict(T0_GR_ms=0.028, s_naive_GR_ms=-0.003, s_block_GR_ms=0.004)
PAGES = REPO / DC.PAGES


def prm(form="scaled", **over):
    return V4D.DecodeParams(**{**PLANT, **over}, rho_form=form, tile=NOM)


def write_cal(root: Path, *, form="scaled", label_bias=0.0, refuse=(), skip=(), gd2_us=2.0, noise=0.001):
    regs = DC.registration(REPO, "cal")
    for u in regs["units"]:
        if u["step"] != "r3decode" or (u["model"], u["label"]) in skip:
            continue
        m, kind = u["model"], u["page"].rsplit("-", 1)[1][:-5]
        base = F.truth_v4d(m, prm(form), decl_copies=9 if kind == "decl" else None)
        plus = {"GR1": gd2_us * 1e-3, "E240": gd2_us * 1e-3 + 0.005}.get(u["inst"], 0.0)

        def truth(c, a, base=base, plus=plus):
            t = base(c, a) + plus
            return t * (1 + label_bias) if c.role == "MINCOVER" else t
        F.write_page(root, m, u["label"], REPO / DC.page_path(m, kind), u["inst"], truth, noise=noise,
                     seed=hash((m, u["label"])) % 1000, copies=9 if kind == "decl" else 1,
                     bad_readback=refuse if u["label"] == "dm" and m == "olmoe-1b-7b" else ())
    return root


@pytest.fixture(scope="module")
def cal_good(tmp_path_factory):
    root = write_cal(tmp_path_factory.mktemp("cal"))
    s = SC.score(root, DC.registration(REPO, "cal"), NOM, "planted NOM")
    return root, s


def test_the_cal_scorer_recovers_the_planted_constants_and_the_rho_form(cal_good):
    _root, s = cal_good
    f = s["FIT"]
    assert f["selected"] == "scaled"
    got = f["forms"]["scaled"]
    assert got["T0_GR_ms"] == pytest.approx(PLANT["T0_GR_ms"], abs=2e-4)
    assert got["s_naive_GR_ms"] == pytest.approx(PLANT["s_naive_GR_ms"], abs=5e-4)
    assert got["s_block_GR_ms"] == pytest.approx(PLANT["s_block_GR_ms"], abs=5e-4)
    assert s["E1_CAL_IN_SAMPLE"]["verdict"] == "HOLDS" and s["E1_CAL_IN_SAMPLE"]["stats"]["rms"] < 0.005


def test_the_cal_scorer_reads_g3d_gd2_gd3_the_replicate_and_says_go(cal_good):
    _root, s = cal_good
    assert {t: s["G3D"][t]["verdict"] for t in ("shuffle", "label")} == {"shuffle": "EQUIVALENT", "label": "EQUIVALENT"}
    assert s["G3D"]["fill"]["rows"]
    assert s["GD2"]["verdict"] == "PASS" and s["GD2"]["median_us"] == pytest.approx(2.0, abs=0.3)
    lo, hi = s["GD2"]["band_us"]
    assert lo < 2.0 < hi
    assert s["GD3"]["cells"] > 0 and s["GD3"]["median_us"] == pytest.approx(5.0, abs=1.0)
    assert s["REP"]["pairs"] > 30 and s["REP"]["sigma_page"] is not None
    assert s["REP"]["bars_blind"]["rms_max"] == DC.BARS_FLOOR["rms_max"]       # 5 sigma under the floor
    assert s["DECL"]["rows"] and all(r["path_decl"] in ("block-scan", "naive") for r in s["DECL"]["rows"])
    assert all(v["state"] == "counted" for k, v in s["units"].items() if v["nodrop"])
    assert s["STOP"] == {"verdict": "GO", "why": []}


def test_finalise_freezes_the_cal_and_writes_the_blind_plan(cal_good, tmp_path):
    _root, s = cal_good
    (tmp_path / "cal.score.json").write_text(json.dumps(s, default=str))
    doc = FN.finalise(tmp_path)
    fz = doc["frozen"]
    assert fz["rho_form"] == "scaled" and fz["constants"]["T0_GR_ms"] == pytest.approx(PLANT["T0_GR_ms"], abs=2e-4)
    assert fz["gd2_band_us"] == s["GD2"]["band_us"] and fz["bars"] == s["REP"]["bars_blind"]
    models = {r["model"] for r in doc["cells"]}
    assert models == set(DC.BLIND_MODELS) and doc["not_priced"] == 0
    row = next(r for r in doc["cells"] if r["model"] == "granite-3.0-3b-a800m" and r["T"] == 1)
    assert set(row["ms"]) == {"v4d", "rho other", "NAIVE-NODEDUP", "eager", "roofline", "line", "OLS"}
    plan = FN.blind_plan(fz)
    assert plan.count(" r3decode ") == len(DC.BLIND_UNITS) and "Finalised: T0_GR" in plan


@pytest.mark.parametrize("defect, why", [
    (dict(label_bias=0.05), "G3-D label"),
    (dict(refuse=("L-T8-A64",)), "refused by V-ALIGN"),
    (dict(skip=(("mixtral-8x7b", "dc"),)), "nodrop unit mixtral-8x7b-dc"),
    (dict(gd2_us=15.0), "GD2"),
])
def test_every_planted_defect_is_a_stop_and_finalise_refuses_it(tmp_path, defect, why):
    root = write_cal(tmp_path / "tree", **defect)
    s = SC.score(root, DC.registration(REPO, "cal"), NOM, "planted NOM")
    assert s["STOP"]["verdict"] == "STOP" and any(why in w for w in s["STOP"]["why"]), s["STOP"]
    (tmp_path / "cal.score.json").write_text(json.dumps(s, default=str))
    with pytest.raises(FN.Refused, match="STOP"):
        FN.finalise(tmp_path)


def test_a_wrong_model_is_a_stop_on_the_in_sample_bar(tmp_path):
    """Truth carries a per-row cost v4d does not price (the FILL-r rival at 5 ns a padded row)."""
    regs = DC.registration(REPO, "cal")
    for u in regs["units"]:
        if u["step"] != "r3decode" or u["label"] not in ("dm", "dc"):
            continue
        m, kind = u["model"], u["page"].rsplit("-", 1)[1][:-5]
        base = F.truth_v4d(m, V4D.DecodeParams(**PLANT, rho_form="fixed", tile=NOM, fill_r_us=0.08))
        plus = 0.002 if u["inst"] == "GR1" else 0.0
        F.write_page(tmp_path, m, u["label"], REPO / DC.page_path(m, kind), u["inst"],
                     lambda c, a, base=base, plus=plus: base(c, a) + plus)
    s = SC.score(tmp_path, regs, NOM, "planted")
    assert s["E1_CAL_IN_SAMPLE"]["verdict"] == "FAILS" and s["STOP"]["verdict"] == "STOP"


def _blind_tree(fin, root, slow=None):
    reg = DC.registration(REPO, "blind")
    for u in reg["units"]:
        m, kind = u["model"], u["page"].rsplit("-", 1)[1][:-5]
        base = F.truth_v4d(m, prm("scaled"))
        k = 1.06 if m == slow else 1.0
        plus = 0.002 if u["inst"] == "GR1" else 0.0
        F.write_page(root, m, u["label"], REPO / DC.page_path(m, kind), u["inst"],
                     lambda c, a, base=base, k=k, plus=plus: base(c, a) * k + plus, noise=0.001, seed=7)
    return SB.score(root, reg, fin)


def test_the_blind_scorer_certifies_every_model_v4d_prices(cal_good, tmp_path):
    _root, s = cal_good
    (tmp_path / "cal.score.json").write_text(json.dumps(s, default=str))
    got = _blind_tree(FN.finalise(tmp_path), tmp_path / "tree")
    assert got["E1_DEC"]["pooled"]["verdict"] == "HOLDS"
    assert set(got["certified"].values()) == {"CERTIFIED"}


@pytest.fixture(scope="module")
def blind_case(cal_good, tmp_path_factory):
    _root, s = cal_good
    d = tmp_path_factory.mktemp("fin")
    (d / "cal.score.json").write_text(json.dumps(s, default=str))
    fin = FN.finalise(d)
    tree = tmp_path_factory.mktemp("blind")
    reg = DC.registration(REPO, "blind")
    for u in reg["units"]:
        m, kind = u["model"], u["page"].rsplit("-", 1)[1][:-5]
        base = F.truth_v4d(m, prm("scaled"))
        # Granite's truth is 6% slow everywhere: v4d must not certify it
        k = 1.06 if m == "granite-3.0-3b-a800m" else 1.0
        plus = 0.002 if u["inst"] == "GR1" else 0.0
        F.write_page(tree, m, u["label"], REPO / DC.page_path(m, kind), u["inst"],
                     lambda c, a, base=base, k=k, plus=plus: base(c, a) * k + plus, noise=0.001, seed=7)
    return SB.score(tree, reg, fin)


def test_the_blind_scorer_certifies_what_v4d_prices_and_not_what_it_misprices(blind_case):
    s = blind_case
    cert = s["certified"]
    assert cert["granite-3.0-3b-a800m"] == "NOT CERTIFIED"
    assert s["E1_DEC"]["per_model"]["phi-3.5-moe"]["verdict"] == "HOLDS"
    assert s["E1_DEC"]["pooled"]["verdict"] == "FAILS"          # Granite's 6% fails the pooled worst bar
    assert all(v == "NOT CERTIFIED" for v in cert.values())     # so no model certifies: the pooled clause
    assert s["GD2"]["qwen1.5-moe-a2.7b"]["verdict"] == "PASS"
    assert {t: s["G3D"]["phi-3.5-moe"][t]["verdict"] for t in ("shuffle", "label")} == {"shuffle": "EQUIVALENT",
                                                                                            "label": "EQUIVALENT"}
    assert s["rivals"]["roofline"]["verdict"] in ("EXCLUDED", "NOT EXCLUDED") or "NOT SEPARATED" in s["rivals"]["roofline"]["verdict"]


def test_five_refusals_on_a_48_cell_page_leave_the_page_standing(tmp_path):
    """build-decode review 8: refused cells write a non-clean row on every repeat; counted in r7's 10% share,
    5 of 48 (10.4%) would FAIL G1 and lose the page. They are taken out of the share first."""
    m = "granite-3.0-3b-a800m"
    reg = DC.registration(REPO, "blind")
    page = REPO / DC.page_path(m, "main")
    doc = json.loads(page.read_text())
    assert len(doc["cells"]) == 48
    bad = tuple(c["label"] for c in doc["cells"][10:15])
    F.write_page(tmp_path, m, "dm", page, "GRR", F.truth_v4d(m, prm("scaled")), bad_readback=bad)
    import r7common as C7
    pdir = next(tmp_path.rglob("report.json")).parent
    raw_rep, raw = C7.read_page(pdir)
    assert C7.rebuild(raw_rep, raw)[0]["g1_r7"]["verdict"] == "FAIL"        # the defect, as the review found it
    rep, why = DC.decode_page(tmp_path, m, "dm", reg["pages"]["files"][f"{m}-main.json"], "GRR")
    assert why == "counted", why
    assert sorted(rep["g1_r7"]["refused_or_failed_cells_excluded"]) == sorted(f"{b}/native/T{c['T']}" for b, c in
                                                                              ((c["label"], c) for c in doc["cells"][10:15]))
    v, dropped = DC.measured(rep, m, "main")
    assert len(v) == 43 and sum("refused" in d[1] for d in dropped) == 5


def test_score_cal_refuses_without_the_tile_stages_selections(capsys, tmp_path):
    assert SC.main([str(tmp_path), str(tmp_path / "o")]) == 2
    assert "--tile-scores" in capsys.readouterr().out


def test_missing_tile_selections_or_an_unpriced_cell_is_a_stop(cal_good):
    root, _s = cal_good
    s = SC.score(root, DC.registration(REPO, "cal"), V4.Params(), "none")
    assert s["tile_selections_missing"] and s["NOT_PRICED"]
    assert s["STOP"]["verdict"] == "STOP"
    assert any("tile stage's selections" in w for w in s["STOP"]["why"]) and any("NOT PRICED" in w for w in s["STOP"]["why"])


def test_an_unfitted_offset_is_not_priced_never_zero():
    fit = {"T0_GR_ms": 0.03, "s_naive_GR_ms": None, "s_block_GR_ms": 0.004}
    p = V4D.params_from(fit, rho_form="fixed", tile=NOM)
    naive = {"T": 1, "counts": [1] * 8 + [0] * 56}
    small = {"T": 8, "counts": [1] * 64}
    assert DC.price_cell("olmoe-1b-7b", naive, p) is None
    assert DC.price_cell("olmoe-1b-7b", small, p)["ms"] > 0.03
    raw = V4D.price("olmoe-1b-7b", 1, naive["counts"], p)
    assert raw["ms"] is None and raw["not_priced"].startswith("NOT PRICED")


def test_the_gd2_band_has_a_width_cap():
    assert DC.gd2_band([1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0])["verdict"] == "PASS"
    wide = DC.gd2_band([0.0, 0.5, 1.0, 2.0, 3.0, 9.0, 9.5])
    assert wide["verdict"].startswith("FAIL") and "cap" in wide["verdict"]
    assert DC.gd2_band([12.0] * 7)["verdict"].startswith("FAIL")
