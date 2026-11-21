"""Tile rental A2's registrations (docs/registered/2026-10-20-tileA2-*, 2026-10-20-tileB2-*, 2026-10-20-decode-tilesource):
`register_a2.py --check` recomputes all eleven from committed files; each tileB2 file equals its frozen tileB twin
on every key outside the registered whitelist (REVIEW-A2 7b: the power block is whitelisted, never held equal);
the review's fixes are in them as registered text and numbers (the owner's lock and power limit, the SEEN block,
the score directory, the gates, CONV-L's ratio and its wording, the census rules, the decode addendum); their
code pins hold every file at its bytes while the two stage BASELINEs stay unchanged (REVIEW-A2 7a); the pinned
tests' globs do not sweep them; no em-dash or trailer in any A2 file; the copied pages and the offline check
recompute byte for byte."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
A2DIR = REPO / "scripts" / "scoring" / "tileA2"
for p in (REPO, REPO / "scripts", A2DIR, REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import a2common as A2  # noqa: E402

RT = A2.TC._load("regtile_a2", A2DIR / "regtile_a2.py")
REG = REPO / "docs" / "registered"
NAMES = {**A2.NAMES_A2, "tilesource": A2.TILESOURCE}
DOCS = {p: json.loads((REG / f"{n}.json").read_text()) for p, n in NAMES.items()}
A2_FILES = sorted({*(REG.glob("2026-10-20-tileA2-*-gh200.*")), *(REG.glob("2026-10-20-tileB2-*-gh200.*")),
                   *(REG.glob("2026-10-20-decode-tilesource-gh200.*")), *A2DIR.glob("*.py"),
                   REPO / "scripts" / "r3_timing_model_v4L.py", REPO / "scripts" / "tile_census_a2.py",
                   REPO / "scripts" / "tileA2_gate.py", REPO / "scripts" / "gh200_tileA2_session.sh",
                   REPO / "scripts" / "plans" / "rentalA2-tile-2026-10.plan",
                   *(REPO / "tests").glob("test_tileA2_*.py")})


def _sha(p) -> str:
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def test_register_a2_check_recomputes_all_eleven_files():
    got = subprocess.run([sys.executable, str(A2DIR / "register_a2.py"), "--check"], capture_output=True, text=True,
                         cwd=REPO, timeout=1800)
    assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-3000:]
    assert got.stdout.count(": same") == 22 and "DIFFERS" not in got.stdout


@pytest.mark.parametrize("tool", ["hist_a2.py", "conv_offline.py"])
def test_the_copied_pages_and_the_offline_check_recompute_byte_for_byte(tool):
    got = subprocess.run([sys.executable, str(A2DIR / tool), "--check"], capture_output=True, text=True, cwd=REPO, timeout=900)
    assert got.returncode == 0 and "DIFFERS" not in got.stdout, got.stdout + got.stderr[-2000:]


def _diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                out.append(p)
            else:
                out += _diff(a[k], b[k], p)
    elif a != b:
        out.append(path)
    return out


@pytest.mark.parametrize("part", ["e1tile", "stop"])
def test_each_tileb2_file_is_its_frozen_tileb_twin_outside_the_whitelist(part):
    frozen = json.loads((REG / f"2026-10-14-tileB-{part}-gh200.json").read_text())
    mine = dict(DOCS[part])
    frozen.pop("code_pins")
    mine.pop("code_pins")
    wl = RT.WHITELIST_B2[part]
    outside = [p for p in _diff(frozen, mine) if not any(p == w or p.startswith(w + ".") for w in wl)]
    assert outside == [], outside
    if part == "e1tile":
        assert mine["power"] != frozen["power"]          # REVIEW-A2 7b: regenerated at L, printed beside, never held equal
        assert mine["pages"]["files"] == frozen["pages"]["files"] and mine["tests"]["E1-TILE (primary, D2)"] == frozen["tests"]["E1-TILE (primary, D2)"]
        assert "1.71, 1.41 or 1.305 GHz" in mine["tests"]["rivals"]["roofline"] and mine["gates"]["G4"].startswith("cell clock >= L - 5 MHz")
        assert mine["power limit and lock (owner, 2026-10-20)"].startswith("rental B2 runs at the GPU power limit and the lock rung")
        assert "tileB-e1tile's own" in mine["at rung 1710"]
    else:
        risk = mine["a2"]["conversion risk at 1410 and 1305 (build review MAJOR 1; owner, 2026-10-20)"]
        assert "1.36%" in risk and "1.4x" in risk and "0.75%" in risk


def test_every_a2_registration_names_the_owners_lock_and_power_limit_and_carries_the_seen_block():
    for p, d in DOCS.items():
        assert d["registered"] == "2026-10-20, before any tile rental A2 page" and d["name"] == NAMES[p]
        assert "A 900 W GPU POWER LIMIT AND THE LOCK LADDER 1710, 1410, 1305" in d["design"]
        assert "700 W decision for tile rental A2" in d["design"]
        lock = d["lock_block"] if p == "tilesource" else d["lock"]
        assert lock["decision (owner, 2026-10-20)"].startswith("a 900 W GPU power limit (GPU scope, -sc 0) and the lock ladder 1710")
        seen = d["seen (REVIEW-A2 5)"]
        for item in ("(i)", "(ii)", "(iii)", "(iv)", "(v)", "(vi)", "(vii)", "(viii)"):
            assert any(k.startswith(item + " ") for k in seen), (p, item)
        assert "698.67 +- 10.63" in json.dumps(seen) and "UNREAD" in seen["(viii) rental A's times"]
        assert "OD13, a rule of this registration" in seen["(viii) rental A's times"]


def test_the_gates_are_registered_mechanically_with_the_power_rule_and_rental_as_board():
    g = DOCS["gates"]
    pw = g["power (rule P-W; the owner's 2026-10-20 decision)"]
    assert "power.max_limit" in pw["rule"] and "no fallback to 700 W" in pw["rule"]
    assert pw["rental A's board"]["Max Power Limit (GPU)"] == 900.0
    gcal = next(v for k, v in g.items() if k.startswith("G-CAL"))
    assert "LOWEST rung, 1305 MHz" in gcal
    probe = g["G-PROBE (the lock-hold probe; DESIGN-A2 6 risk 2, REVIEW-A2 1a, 1c, 6)"]
    assert "1710, then 1410, then 1305 MHz" in probe["ladder"] and "L - 5 MHz" in probe["rung"] and "10%" in probe["rung"]
    assert "one LOST cell" in probe["strict (owner, 2026-10-20)"]
    pr = probe["pass rates (build review fix 6, MINOR 3)"]
    assert pr["rung"]["P(both pass) at p 0.08"] < pr["Mixtral page (16 cells, 144 rows)"]["P(pass) at p 0.08"] < 0.82
    assert "REMOVED" in next(k for k in g if k.startswith("1710 at 900 W"))
    assert g["plan"]["deadline"]["deadline_min_after_launch"] == 220 and g["plan"]["deadline"]["hard_cap_min_after_launch"] == 235
    assert "182 <= 187" in g["plan"]["deadline"]["arithmetic"] and "rep" in g["plan"]["deadline"]["drop"]
    p5 = probe["counter clause P5 (owner, 2026-10-20: the re-review's MAJOR 1, option a)"]
    assert "at least 6 floor-bound cells" in p5 and "steps down" in p5 and "P5" in probe["rung"]
    assert "not modelled" in pr["model"]
    assert "lock 1710: about 173 min" in g["plan"]["minutes"] and "drops dk, ms, w4, rep" in g["plan"]["minutes"]
    assert "scripts/scoring/tileA2/" in g["score directory (build review fix 4)"]
    assert "held or held-with-slips" in probe["rung"] and "no person reads" in probe["ms unread"]
    assert set(probe["pages"]) == {"mixtral-8x7b-128g16-probe.json", "qwen2-57b-a14b-128g1-probe.json"}
    assert "joint_cF over the f128 pages" in g["G-BAND (after f128 Mixtral; REVIEW-A2 2b)"]
    assert "joint_cF over the f64w8 pages" in g["G-64 (after f64w8 Mixtral; REVIEW-A2 2b)"]
    assert "ANY exit but 0" in g["abort mechanics"] and "SKIPPED for time" in g["abort mechanics"]
    assert g["plan"]["path"] == A2.PLAN_A2 and g["plan"]["est_total_min"] == 168


def test_the_score_directory_and_the_census_rules_are_registered():
    for p in ("census", "e1cal", "gates"):
        sd = DOCS[p]["a2_score_directory"]
        assert "NAMED MERGE STEP" in sd["census.score.json"] and "never copied in" in sd["rep.score.json"]
        assert "CLAMP" in sd["bytes.score.json"]
    c = DOCS["census"]
    assert c["rental_A_census"]["sha256"] == _sha(c["rental_A_census"]["path"]) and c["rental_A_census"]["entries"] == 72
    assert c["a2_rules"]["treads"]["deepseek-v2-lite"] == {t: 3 for t in c["a2_rules"]["treads"]["deepseek-v2-lite"]}
    assert "NOT SCORED" in c["a2_rules"]["OLMoE n 1 / n 3 (REVIEW-A2 8a)"]
    assert "STACK DRIFT" in c["a2_rules"]["equality gate (DESIGN-A2 3.1; tileA2-gates G-CEN)"]


def test_the_floor_bytes_and_e1cal_carry_the_band_g4_and_the_conversion_at_l():
    f = DOCS["floor"]
    assert "[1374.75, 1410]" in f["counter band at L (A2)"]["primary"] and "NEW rule" in f["counter band at L (A2)"]["primary"]
    assert "RATIO_REF holds 1710 only" in f["counter band at L (A2)"]["lock_gate"]
    assert "1410 MHz" in f["parameters"]["hardware (no fit)"] and "FALSIFIED by construction" in f["expected outcomes (SEEN, DESIGN-A2 7d)"]
    assert "lever_counts_on_B at the fallback rung 1305" in f["selection_for_rental_B (D3)"]
    assert "band in force" in DOCS["bytes"]["gates"]["clock"]
    e = DOCS["e1cal"]
    assert e["gates"]["G4"].startswith("cell clock >= L - 5 MHz (1705 at 1710, 1405 at 1410")
    assert "1.36%" in e["conversion risk at 1410 and 1305 (build review MAJOR 1; owner, 2026-10-20)"]
    assert "MVA2" in e["conversion (UNTESTED here)"] and e["conversion (UNTESTED here)"].startswith("at rung 1710 none")
    assert "UNTESTED" not in e["predictions"] and "T0, bw" in e["conversion (UNTESTED here)"]
    old = json.loads((REG / "2026-10-14-tileA-e1cal-gh200.json").read_text())
    assert len(e["nominal_cells"]) == len(old["nominal_cells"])
    assert all(a["ms_nominal"] >= b["ms_nominal"] for a, b in zip(e["nominal_cells"], old["nominal_cells"], strict=True))
    assert DOCS["dead"]["status"].endswith("NOT A TEST (REVIEW-A2 5 iv)")


def test_conv_l_is_a_per_cell_ratio_at_two_locks_worded_indicates_and_its_1710_prices_are_rental_7s():
    c = DOCS["convl"]
    assert c["rule"]["classes"] == ["G1", "G8"] and "|mean| <= 1%" in c["rule"]["bar"]
    assert "NOT a stop clause" in c["status"] and c["pages"]["unless_lock"] == 1710
    assert "NOT SCORED" in c["at rung 1710 (owner, 2026-10-20)"]
    for k, v in c["reading (DESCRIPTIVE, REVIEW-A2 3a)"].items():
        if "FAILS" in k:
            assert "indicates" in v and " reads " not in v, k
    assert {u["label"] for u in c["pages"]["units"]["at_L"]} == {"cvm1", "cvo1", "cvm8"}
    assert {u["label"] for u in c["pages"]["units"]["at_1710"]} == {"cvm1t", "cvo1t", "cvm8t"}
    r7 = json.loads((REG / "2026-10-13-rental7-skew-gh200.json").read_text())
    reg = {(p["n"], p["label"]): p["ms"]["v3"] for p in r7["predictions"]
           if p["model"] == "mixtral-8x7b" and p["page"] == "A" and p["arm"] == "native"}
    g8 = [p for p in c["predictions"] if p["page"] == "convl-mixtral-8x7b-g8"]
    assert len(g8) == 8 and all(p["ms"]["1710"] == reg[(p["n"], p["label"])] for p in g8)
    assert all(p["ms"]["1410"] > p["ms"]["1710"] and p["ratio"]["1410"] < 1710 / 1410 for p in c["predictions"])
    off = c["offline_check (SEEN, OD12; REVIEW-A2 3b and 3c)"]
    assert off["sha256"] == _sha(off["file"]) and "SEEN-SELECTED" in off["labels (REVIEW-A2 3c)"]


def test_the_decode_addendum_binds_a2s_score_directory_and_says_what_decode_does_on_an_a2_stop():
    t = DOCS["tilesource"]
    assert "stays at 1710" in t["lock"] and "byte-identical" in t["status"]
    assert "does not stop decode" in t["on an A2 STOP (REVIEW-A2 7d; owner, 2026-10-20)"]
    gate = t["the decode gate (owner, 2026-10-20)"]
    assert "ONLY IF A2's floor calibration is scored" in gate and "WAIT" in gate and "byte-identical" in gate
    assert "scripts/scoring/tileA2/" in t["binds"] and "decode_gate_a2.py scripts/scoring/tileA2" in t["tool"]
    assert "A2's merged census" in t["census (REVIEW-A2 7d)"] and "six token-count errors" in t["decode's own census unit"]
    assert "698.67" in t["cross-clock readout (REVIEW-A2 7c, DESCRIPTIVE)"] and "score_floor_a2.py" in t["cross-clock readout (REVIEW-A2 7c, DESCRIPTIVE)"]


# --------------------------------------------------------------------------
# pins, globs, text
# --------------------------------------------------------------------------

def test_the_a2_code_pins_hold_every_file_and_the_stage_baselines_are_unchanged():
    import test_pinned_files as TP
    import test_stage_decode_pins as TD
    for base in (TP.BASELINE, TD.BASELINE):
        for p, s in base.items():
            assert _sha(p) == s, p
    pins = {}
    for d in DOCS.values():
        for p, s in d["code_pins"].items():
            pins.setdefault(p, set()).add(s)
    tile = {}
    for f in REG.glob("2026-10-14-tile[AB]-*.json"):
        tile.update(json.loads(f.read_text())["code_pins"])
    for p, shas in pins.items():
        assert len(shas) == 1 and _sha(p) == next(iter(shas)), p
        if p in tile:
            assert tile[p] == next(iter(shas)), p
    assert set(tile) <= set(pins)
    for p in ("scripts/gh200_tileA2_session.sh", "scripts/r3_timing_model_v4L.py", "scripts/tile_census_a2.py",
              "scripts/scoring/tileA/score_floor.py", "scripts/scoring/tileA/score_bytes.py", "scripts/tileA2_gate.py"):
        assert p in pins, p


def test_the_pinned_tests_globs_do_not_sweep_the_a2_files():
    import fnmatch
    for f in REG.glob("2026-10-20-*.json"):
        for g in ("*-tileA-*.json", "*-tileB-*.json", "*-tile[AB]-*.json", "*-rental6-*.json", "*-rental7-*.json",
                  "2026-10-14-decode*-gh200.*"):
            assert not fnmatch.fnmatch(f.name, g), (f.name, g)


@pytest.mark.parametrize("path", A2_FILES, ids=lambda p: p.name)
def test_no_em_dash_and_no_trailer_in_any_a2_file(path):
    t = path.read_text()
    assert chr(0x2014) not in t and chr(0x2013) not in t, path
    assert not re.search("|".join(("Co-" + "Authored-By", "Claude " + "Code", "Generated " + "with", "Anthro" + "pic")), t), path
