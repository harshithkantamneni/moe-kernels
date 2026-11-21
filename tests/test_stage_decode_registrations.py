"""The decode stage's two registrations and its CAL plan: recomputed from committed files they are the
committed bytes (register_decode.py --check); every page they name is the committed page at its
sha256; the CAL plan is the generator's output from the registered unit list; the blind registration
is frozen and names its finalisation, which refuses on a STOP."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "decode"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dcommon as DC  # noqa: E402
import finalise_decode as FN  # noqa: E402
import register_decode as RGD  # noqa: E402


def test_the_registrations_recompute_to_their_committed_bytes(capsys):
    assert RGD.main(["--check"]) == 0, capsys.readouterr().out


def test_every_registered_page_is_the_committed_file_and_every_cell_its_counts():
    import r3_decode as RD
    for part in DC.PARTS:
        reg = DC.registration(REPO, part)
        for name, f in reg["pages"]["files"].items():
            data = (REPO / f["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == f["sha256"], name
            doc = json.loads(data)
            assert [c["counts_sha256"] for c in f["cells"]] == [RD.counts_sha256(c["counts"]) for c in doc["cells"]]
        assert {u["page"] for u in reg["units"]} == set(reg["pages"]["files"])


def test_the_cal_plan_is_generated_from_the_registered_units():
    text = (REPO / FN.CAL_PLAN).read_text()
    assert text == FN.cal_plan()
    reg = DC.registration(REPO, "cal")
    body = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    units = body[3:]
    assert len(units) == len(reg["units"])
    for ln, u in zip(units, reg["units"]):
        assert ln.startswith(f"{u['model']} {u['step']} label={u['label']} histogram={DC.PAGES}/{u['page']}")
        assert ("nodrop=1" in ln) == (u["drop"] == "nodrop")
        if u["step"] == "r3decode":
            assert f"inst={u['inst']}" in ln and "slip-policy=cell" in ln and "g1-rule=r7-median-run3" in ln
    assert "tiles=" + ",".join(DC.DECODE_TILES) in body[2]


def test_the_blind_registration_is_frozen_and_its_plan_generates():
    reg = DC.registration(REPO, "blind")
    assert reg["status"].startswith("FROZEN") and "refuses" in reg["status"]
    assert {u["model"] for u in reg["units"]} == set(DC.BLIND_MODELS)
    plan = FN.blind_plan()
    assert plan.count(" r3decode ") == len(DC.BLIND_UNITS)
    assert all(f"histogram={DC.PAGES}/{u['page']}" in plan for u in reg["units"])


def test_the_registrations_state_the_rentals_the_risk_and_the_fallback():
    cal, blind = DC.registration(REPO, "cal"), DC.registration(REPO, "blind")
    for d in (cal, blind):
        r = d["rental"]
        assert r["units_min"] > 0 and r["cost_usd"] == round(r["wall_min"] / 60 * 2.29, 2)
    assert "EXTRAPOLATED" in cal["trace_statistics"]["fallback"] and "NOT RUN" in cal["trace_statistics"]["primary"]
    assert "transfers across models" in blind["riskiest_assumption"]
    assert cal["upstream"]["sha256"] == hashlib.sha256((REPO / "moe/instrumented/UPSTREAM_DECODE.json").read_bytes()).hexdigest()
    assert cal["power"]["p_holds"]["the model is the truth"] >= 0.95


def test_no_em_dash_and_no_trailer_in_the_decode_files():
    paths = [REPO / p for p in RGD.DECODE_CODE] + [REPO / FN.CAL_PLAN] + sorted((REPO / "docs/registered").glob("2026-10-14-decode*-gh200.*"))
    paths += sorted((REPO / "tests").glob("test_stage_decode_*.py")) + [REPO / "tests/decode_fakes.py",
                                                                   REPO / "scripts/scoring/decode/score_cal.py",
                                                                   REPO / "scripts/scoring/decode/score_blind.py",
                                                                   REPO / "scripts/scoring/decode/register_decode.py"]
    for p in paths:
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
        assert not re.search("|".join(("Co-" + "Authored-By", "Claude " + "Code", "Generated " + "with")), t), p
