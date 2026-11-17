"""Reproduction within stated tolerances, labels exact: the one comparison every reproduction test that re-runs a
fit uses to hold a regenerated output to its committed file.

Why not byte identity. The committed outputs were written on macOS 26.6.2, where numpy 2.5.3 (uv cpython 3.12.13)
links Apple's Accelerate for BLAS and LAPACK. The update to macOS 27.2.1 (2026-11-16) replaced Accelerate, and every
output that goes through a least-squares fit or a matrix product now differs from its committed file in the last
digits of some floats (the diagnosis: ~/code/moe-kernels-work/2026-11-17/numdrift/REPORT.md, the coordinator's work
folder, not in this repo). No verdict, band, READ / NOT READ, status, key or list length moved. So a reproduction test
holds every string, bool, null, integer, key and list length exactly, and every float within the tolerance below.

JSON (compare_json): the same structure (dict keys equal as sets and in order, list lengths equal), every str, bool
and None equal, every int equal (an int is a count, an id or a size), every float within the profile's tolerance:
|got - want| <= max(rel * max(|got|, |want|), abs). A float never matches an int or a bool.

Text (compare_text): the same number of lines; each line split into numeric and non-numeric tokens; every non-numeric
token exact (whitespace included); an integer token exact; a decimal token within the profile's tolerance, or one unit
in its last printed place when both sides print the same number of decimals and that unit is at most 1e-3 of the
value (a rounded value whose rounding flipped; a shorter rounded value, where one unit is a real change, is held to
the tolerance). One narrow rule for the truncated dumps read_pc.py and read_pc2.py write (a line "  <first 20000
characters of a json.dumps> ... [TRUNCATED: N more characters; the whole reading is in X]"): the cut point and N move
with float print length (pc2.read.txt line 10 cut 2 characters later on 27.2.1), so on two such lines the wording
after the cut is exact, the two bodies are compared token by token up to the last token before the shorter cut, the
token count may differ by at most 2, and N is not compared. The whole reading is in the .json beside it, which
compare_json holds in full.

Profiles (tolerances; magnitudes from numdrift/REPORT.md, section 2):
- DEFAULT, REL_TOL 1e-12: every drifted output but two moved at most 1.4e-13 relative (rr.read), most near 1e-15.
- DS, DS_REL_TOL 1e-6: decode DS's W1RATE fit (scripts/scoring/decode/d4common.py:838-845) puts 0/1 ladder columns
  beside an unscaled ~1e8 byte count, condition about 1e8; ds.readings moved 1.4e-7 relative (offsets_us
  phi-3.5-moe/T96 34.524621 to 34.524616), below every printed digit.
- CAL3, numbers not compared: read_c3.wave_theta (scripts/scoring/cal3/read_c3.py:161-163) fits np.linalg.lstsq on
  raw globaltimer stamps (~1.793e18 ns, float64 spacing 256 ns), an ill-conditioned fit that keeps about three good
  digits; cal3.read thetas moved up to 2.6e-2 relative. Its C3-R1 reading is a defect of record (docs/FINDINGS.md):
  a well-conditioned recomputation reads NEAR or OUT (PERTURBED) where the committed file reads OUT. So for cal3 the
  structure, every label and every int (counts) are held exactly, and no float is compared.

ABS_TOL is the floor for values near zero (differences of nearly equal numbers): 1e-15 for DEFAULT, 1e-12 for DS.

Run as a script, `python tests/_reproduce.py register_decode_d5` is register_decode_d5.py --check with these rules
(the registered generator compares its recomputed JSON with ==, text and gate output byte for byte, and it is pinned,
so it is not edited); it prints the same "<name>: same | DIFFERS" lines and exits 0 only when all read same."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: default relative tolerance (numdrift/REPORT.md: at most 1.4e-13 relative outside DS and cal3)
REL_TOL = 1e-12
#: absolute floor for values near zero under REL_TOL
ABS_TOL = 1e-15
#: decode DS (W1RATE fit, condition about 1e8, d4common.py:838-845; ds.readings moved 1.4e-7 relative)
DS_REL_TOL = 1e-6
DS_ABS_TOL = 1e-12
#: one unit in the last printed place is allowed only where that unit is at most this fraction of the value
ROUNDED_UNIT_MAX = 1e-3
#: how far the token counts of two truncated dumps may differ (the cut moves with float print length)
TRUNC_TOKEN_SLACK = 2


@dataclass(frozen=True)
class Profile:
    name: str
    rel: float
    abs: float
    numbers: bool = True          # False: floats are not compared (cal3); ints, strings and structure still are


DEFAULT = Profile("default", REL_TOL, ABS_TOL)
DS = Profile("decode DS", DS_REL_TOL, DS_ABS_TOL)
CAL3 = Profile("cal3", math.nan, math.nan, numbers=False)

_NUM = re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_TRUNC = re.compile(r"^(?P<body>.*) \.\.\. \[TRUNCATED: (?P<n>\d+) more characters; (?P<tail>[^\]]*)\]$")


def _close(a: float, b: float, p: Profile) -> bool:
    if not p.numbers:
        return True
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= max(p.rel * max(abs(a), abs(b)), p.abs)


def compare_json(got, want, profile: Profile = DEFAULT, skip=(), _path=()) -> list[str]:
    """Every difference between two parsed JSON values under `profile`, as 'path: why'. `skip` holds key paths
    (tuples) whose values the caller checks itself; each skip must name a documented byte-identity check."""
    if _path in skip:
        return []
    where = "/".join(map(str, _path)) or "<root>"
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return [f"{where}: {type(got).__name__} where the committed file has an object"]
        if list(got) != list(want):
            return [f"{where}: keys differ: missing {sorted(set(want) - set(got))}, extra {sorted(set(got) - set(want))}"
                    if set(got) != set(want) else f"{where}: key order differs"]
        out = []
        for k in want:
            out += compare_json(got[k], want[k], profile, skip, _path + (k,))
        return out
    if isinstance(want, list):
        if not isinstance(got, list):
            return [f"{where}: {type(got).__name__} where the committed file has a list"]
        if len(got) != len(want):
            return [f"{where}: list length {len(got)}, committed {len(want)}"]
        out = []
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            out += compare_json(g, w, profile, skip, _path + (i,))
        return out
    if type(got) is not type(want):
        return [f"{where}: {got!r} ({type(got).__name__}), committed {want!r} ({type(want).__name__})"]
    if isinstance(want, float):
        return [] if _close(got, want, profile) else [f"{where}: {got!r}, committed {want!r} (beyond {profile.name})"]
    return [] if got == want else [f"{where}: {got!r}, committed {want!r}"]


def _tokens(line: str) -> list[tuple[str, str]]:
    out, i = [], 0
    for m in _NUM.finditer(line):
        if m.start() > i:
            out.append(("s", line[i:m.start()]))
        out.append(("n", m.group()))
        i = m.end()
    if i < len(line):
        out.append(("s", line[i:]))
    return out


def _decimals(tok: str) -> int | None:
    if "e" in tok.lower() or "." not in tok:
        return None
    return len(tok.split(".", 1)[1])


def _num_ok(g: str, w: str, p: Profile) -> bool:
    gi, wi = "." not in g and "e" not in g.lower(), "." not in w and "e" not in w.lower()
    if gi or wi:
        return g == w                                    # an integer token is a count, an id or a size: exact
    a, b = float(g), float(w)
    if _close(a, b, p):
        return True
    d = _decimals(g)
    if d is not None and d == _decimals(w):              # a rounded value whose last printed digit flipped
        unit = 10.0 ** -d
        return abs(a - b) <= unit * (1 + 1e-9) and unit <= ROUNDED_UNIT_MAX * max(abs(a), abs(b))
    return False


def _line_diffs(g: str, w: str, p: Profile) -> str | None:
    tg, tw = _tokens(g), _tokens(w)
    if len(tg) != len(tw):
        return f"token count {len(tg)}, committed {len(tw)}"
    for (kg, sg), (kw, sw) in zip(tg, tw, strict=True):
        if kg != kw or (kg == "s" and sg != sw) or (kg == "n" and not _num_ok(sg, sw, p)):
            return f"{sg!r}, committed {sw!r}"
    return None


def _trunc_diffs(mg: re.Match, mw: re.Match, p: Profile) -> str | None:
    if mg["tail"] != mw["tail"]:
        return f"truncation wording {mg['tail']!r}, committed {mw['tail']!r}"
    if len(mg["body"]) != len(mw["body"]):
        return f"truncated body {len(mg['body'])} characters, committed {len(mw['body'])}"
    tg, tw = _tokens(mg["body"]), _tokens(mw["body"])
    if abs(len(tg) - len(tw)) > TRUNC_TOKEN_SLACK:
        return f"truncated body {len(tg)} tokens, committed {len(tw)}"
    n = min(len(tg), len(tw)) - 1                        # the last token before the shorter cut may itself be cut
    for (kg, sg), (kw, sw) in zip(tg[:n], tw[:n], strict=True):
        if kg != kw or (kg == "s" and sg != sw) or (kg == "n" and not _num_ok(sg, sw, p)):
            return f"{sg!r}, committed {sw!r}"
    return None


def compare_text(got: str, want: str, profile: Profile = DEFAULT) -> list[str]:
    """Every differing line between two texts under `profile`, as 'line N: why'."""
    gl, wl = got.split("\n"), want.split("\n")
    if len(gl) != len(wl):
        return [f"line count {len(gl)}, committed {len(wl)}"]
    out = []
    for i, (g, w) in enumerate(zip(gl, wl, strict=True), 1):
        if g == w:
            continue
        mg, mw = _TRUNC.match(g), _TRUNC.match(w)
        why = _trunc_diffs(mg, mw, profile) if mg and mw else _line_diffs(g, w, profile)
        if why:
            out.append(f"line {i}: {why}")
    return out


def compare(got: str, want: str, kind: str, profile: Profile = DEFAULT, skip=()) -> list[str]:
    """compare_json on parsed texts for kind 'json', compare_text for kind 'txt'."""
    if kind == "json":
        return compare_json(json.loads(got), json.loads(want), profile, skip)
    if kind == "txt":
        return compare_text(got, want, profile)
    raise ValueError(kind)


def assert_reproduces(got: str, want: str, kind: str, profile: Profile = DEFAULT, skip=(), what: str = "") -> None:
    diffs = compare(got, want, kind, profile, skip)
    assert not diffs, f"{what} does not reproduce ({profile.name}): " + "; ".join(diffs[:10])


def profile_for(name: str) -> Profile:
    """The profile for a committed output by its file name."""
    if name.startswith("cal3."):
        return CAL3
    if name.startswith("ds.readings."):
        return DS
    return DEFAULT


# ---------------------------------------------------------------------------------------------------------------
# the decode registrations' --check, re-expressed with these rules (the generators are pinned and not edited)
# ---------------------------------------------------------------------------------------------------------------

def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "code_pins"}


#: addendum 3's "reduction (fixes off)" block: verdict and score_sha256 state byte identity with the committed
#: cal.score.json, so they read DIFFERS and a new sha wherever the numerics moved. Re-expressed: the regenerated score
#: reproduces the committed file under DEFAULT, and the registered sha256 is the committed file's own bytes' sha256.
D3_REDUCTION = ("model v4d' (owner, 2026-10-29, D3-1)", "reduction (fixes off)")


def d3_reduction_text(RG3) -> str:
    """register_decode_d3.reduction()'s regenerated score text (the same call, kept here because reduction() returns
    only the verdict and the sha256)."""
    D3, DC, SC2 = RG3.D3, RG3.DC, __import__("score_cal_d2")
    _s, tp, src, _prm = RG3._published()
    with D3.v4dp(D3.V4DP.NONE), contextlib.chdir(REPO):
        got = SC2.score(Path(D3.PUBLISHED) / "results", DC.registration(REPO, "cal"), tp, src)
    return json.dumps(got, indent=1, default=str) + "\n"


def d3_reduction(RG3) -> dict:
    """The reduction read with these rules: 'REPRODUCES' when the regenerated score is the committed one within
    DEFAULT and the committed registration's sha256 is the committed score file's."""
    D3 = RG3.D3
    want = (REPO / D3.PUBLISHED_SCORE).read_text()
    diffs = compare(d3_reduction_text(RG3), want, "json")
    reg = json.loads((REPO / "docs/registered" / f"{D3.NAME}.json").read_text())
    r = reg[D3_REDUCTION[0]][D3_REDUCTION[1]]
    sha_ok = r["score_sha256"] == hashlib.sha256(want.encode()).hexdigest() and r["verdict"] == "IDENTICAL"
    return {"verdict": "REPRODUCES" if not diffs and sha_ok else "DIFFERS", "diffs": diffs, "registered_sha_ok": sha_ok,
            "registered": r}


def check_d3(RG3) -> list[tuple[str, bool, list]]:
    """register_decode_d3.py --check under these rules."""
    D3 = RG3.D3
    red = d3_reduction(RG3)
    doc = RG3.build()
    doc["code_pins"] = RG3.pins()
    if red["verdict"] == "REPRODUCES":       # restore the registered byte-identity facts, then compare the rest
        blk = doc[D3_REDUCTION[0]][D3_REDUCTION[1]]
        blk["verdict"], blk["score_sha256"] = red["registered"]["verdict"], red["registered"]["score_sha256"]
    jpath, tpath = REPO / "docs/registered" / f"{D3.NAME}.json", REPO / "docs/registered" / f"{D3.NAME}.txt"
    want = json.loads(jpath.read_text())
    got = json.loads(json.dumps(doc, indent=1, default=str))
    pdir = REPO / D3.PAGES3
    files = D3.page3_files()
    return [("reduction (fixes off)", red["verdict"] == "REPRODUCES", red["diffs"]),
            (jpath.name, not (d := compare_json(_strip(got), _strip(want))), d),
            (tpath.name, not (d := compare_text("\n".join(RG3.text(doc)) + "\n", tpath.read_text())), d),
            ("settling pages", all((pdir / n).exists() and (pdir / n).read_text() == t for n, t in files.items()), []),
            ("CAL2 plan", (REPO / D3.CAL2_PLAN).exists() and (REPO / D3.CAL2_PLAN).read_text() == D3.cal2_plan(), [])]


def check_d5(RG5) -> list[tuple[str, bool, list]]:
    """register_decode_d5.py --check under these rules."""
    D5 = RG5.D5
    files = D5.page5_files()
    pdir = REPO / D5.PAGES5
    doc = RG5.build()
    doc["code_pins"] = RG5.pins()
    jpath, tpath = REPO / "docs/registered" / f"{D5.NAME}.json", REPO / "docs/registered" / f"{D5.NAME}.txt"
    got = json.loads(json.dumps(doc, indent=1, default=str))
    extra = sorted(p.name for p in pdir.glob("*.json") if p.name not in files) if pdir.is_dir() else []
    gate = REPO / RG5.GATE_OUT
    return [(jpath.name, not (d := compare_json(_strip(got), _strip(json.loads(jpath.read_text())))), d),
            (tpath.name, not (d := compare_text("\n".join(RG5.text(doc)) + "\n", tpath.read_text())), d),
            ("pages", all((pdir / n).exists() and (pdir / n).read_text() == t for n, t in files.items()) and not extra,
             extra),
            ("no blind plan (NOT RUN)", not (REPO / D5.BLIND5_PLAN).exists(), []),
            (gate.name, gate.exists() and not (d := compare_text("\n".join(RG5.GATE_LINES) + "\n", gate.read_text())),
             d)]


def check_d6(RG6) -> list[tuple[str, bool, list]]:
    """register_decode_d6.py --check under these rules."""
    D5, D6 = RG6.D5, RG6.D6
    doc = RG6.build()
    gtext = "\n".join(RG6.build.GATE_LINES) + "\n"
    doc["code_pins"] = RG6.pins()
    jpath, tpath = REPO / "docs/registered" / f"{D6.NAME}.json", REPO / "docs/registered" / f"{D6.NAME}.txt"
    got = json.loads(json.dumps(doc, indent=1, default=str))
    gate = REPO / RG6.GATE_OUT
    pages = all((REPO / D5.page5_path(m, k)).read_text() == t
                for (m, k), t in zip(D5.PAGE_KEYS, D5.page5_files().values(), strict=True))
    return [(jpath.name, not (d := compare_json(_strip(got), _strip(json.loads(jpath.read_text())))), d),
            (tpath.name, not (d := compare_text("\n".join(RG6.text(doc)) + "\n", tpath.read_text())), d),
            ("no blind plan (NOT RUN)", not (REPO / D6.BLIND6_PLAN).exists() and not (REPO / D5.BLIND5_PLAN).exists(), []),
            (gate.name, gate.exists() and not (d := compare_text(gtext, gate.read_text())), d),
            ("addendum 5's pages", pages, [])]


def report(rows) -> tuple[int, str]:
    lines = [f"{name}: {'same' if ok else 'DIFFERS'}" + ("" if ok or not d else f" ({'; '.join(map(str, d[:5]))})")
             for name, ok, d in rows]
    return (0 if all(ok for _n, ok, _d in rows) else 1), "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    which = argv[0]
    for p in (REPO / "scripts" / "scoring" / "decode", REPO / "scripts", REPO):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    mod = __import__(which)
    check = {"register_decode_d3": check_d3, "register_decode_d5": check_d5, "register_decode_d6": check_d6}[which]
    rc, text = report(check(mod))
    print(text)
    return rc


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
