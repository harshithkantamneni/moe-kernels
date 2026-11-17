"""tests/_reproduce.py still catches real changes: a flipped verdict or label fails, a number moved beyond the
tolerance fails, a changed key or list length fails, a cal3 label or count change fails though cal3's floats are not
compared; a drift inside the tolerance passes; the truncated-dump rule is narrow. Run on the committed files
themselves (pc2.read, cal3.read, ds.readings, decode cal.score), edited in memory."""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
import _reproduce as RT  # noqa: E402

CAL = REPO / "scripts/scoring/decode/cal.score"
CAL3 = REPO / "scripts/scoring/cal3/cal3.read"
DS = REPO / "scripts/scoring/decode/ds.readings"
PC2 = REPO / "scripts/scoring/pc2/pc2.read"


def _f(p: Path, ext: str) -> Path:
    return p.parent / (p.name + ext)


def _load(p: Path) -> dict:
    return json.loads(_f(p, ".json").read_text())


def _first(d, pred, path=()):
    """The first (path, value) in document order whose value satisfies pred."""
    if isinstance(d, dict):
        it = d.items()
    elif isinstance(d, list):
        it = enumerate(d)
    else:
        return (path, d) if pred(d) else None
    for k, v in it:
        r = _first(v, pred, path + (k,))
        if r:
            return r
    return None


def _set(d, path, v):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = v


def _is_float(x):
    return type(x) is float and x != 0.0


def test_every_committed_file_reproduces_itself():
    for p, prof in ((CAL, RT.DEFAULT), (CAL3, RT.CAL3), (DS, RT.DS), (PC2, RT.DEFAULT)):
        for ext in ("json", "txt"):
            t = _f(p, f".{ext}").read_text()
            assert RT.compare(t, t, ext, prof) == [], (p, ext)
    assert RT.profile_for("cal3.read.json") is RT.CAL3 and RT.profile_for("ds.readings.txt") is RT.DS
    assert RT.profile_for("pc2.read.json") is RT.DEFAULT and RT.REL_TOL == 1e-12 and RT.DS_REL_TOL == 1e-6


def test_a_flipped_verdict_or_label_string_fails():
    d = _load(CAL)
    path, v = _first(d, lambda x: x in ("STOP", "GO", "READ", "NOT READ", "PASS", "FAIL"))
    e = copy.deepcopy(d)
    _set(e, path, {"STOP": "GO", "GO": "STOP", "READ": "NOT READ"}.get(v, "READ"))
    assert RT.compare_json(e, d), path
    t = _f(CAL, ".txt").read_text()
    assert "STOP: STOP" in t and RT.compare_text(t.replace("STOP: STOP", "STOP: GO"), t)


def test_a_number_moved_beyond_the_tolerance_fails_and_a_drift_within_it_passes():
    d = _load(CAL)
    path, v = _first(d, _is_float)
    for scale, fails in ((1 + 1e-13, False), (1 + 1e-10, True), (1.01, True)):
        e = copy.deepcopy(d)
        _set(e, path, v * scale)
        assert bool(RT.compare_json(e, d)) is fails, (scale, path)
    ds = _load(DS)
    path, v = _first(ds, lambda x: _is_float(x) and abs(x) > 1)
    for scale, fails in ((1 + 1e-7, False), (1 + 1e-5, True)):
        e = copy.deepcopy(ds)
        _set(e, path, v * scale)
        assert bool(RT.compare_json(e, ds, RT.DS)) is fails, (scale, path)
        assert RT.compare_json(e, ds, RT.DEFAULT), "DS's drift fails the default tolerance"


def test_text_numbers_hold_to_the_tolerance_and_to_one_rounded_unit_only():
    assert RT.compare_text("x 0.30000000000000004 y", "x 0.3 y") == []
    assert RT.compare_text("x 0.3001 y", "x 0.3 y")                       # beyond tolerance, decimals differ
    assert RT.compare_text("theta 0.3515", "theta 0.3514") == []          # one unit in the last place, 4 digits
    assert RT.compare_text("theta 0.3516", "theta 0.3514")                # two units
    assert RT.compare_text("x 0.2", "x 0.1")                              # one unit, but 50% of the value
    assert RT.compare_text("cells 7720", "cells 7721")                    # an integer is exact
    assert RT.compare_text("a\nb", "a\nb\nc")                              # line count


def test_a_changed_key_or_list_length_or_type_fails():
    d = _load(PC2)
    e = copy.deepcopy(d)
    k = next(iter(e))
    e[k + " "] = e.pop(k)
    assert RT.compare_json(e, d)
    e = copy.deepcopy(d)
    path, lst = _first_list(e)
    lst.append(lst[-1])
    assert RT.compare_json(e, d), path
    e = copy.deepcopy(d)
    path, v = _first(e, lambda x: type(x) is int and not isinstance(x, bool))
    _set(e, path, float(v))
    assert RT.compare_json(e, d), path                                    # an int never matches a float
    e = copy.deepcopy(d)
    path, v = _first(e, lambda x: isinstance(x, bool))
    _set(e, path, not v)
    assert RT.compare_json(e, d), path


def _first_list(d, path=()):
    if isinstance(d, list) and d:
        return path, d
    if isinstance(d, dict):
        for k, v in d.items():
            r = _first_list(v, path + (k,))
            if r:
                return r
    return None


def test_cal3_floats_are_not_compared_but_its_labels_and_counts_are():
    d = _load(CAL3)
    assert d["C3-R1"]["band"] == "OUT"
    e = copy.deepcopy(d)
    path, v = _first(e, _is_float)
    _set(e, path, v * 1.5)
    assert RT.compare_json(e, d, RT.CAL3) == [] and RT.compare_json(e, d, RT.DEFAULT)
    e = copy.deepcopy(d)
    e["C3-R1"]["band"] = "NEAR or OUT (PERTURBED)"
    assert RT.compare_json(e, d, RT.CAL3)
    e = copy.deepcopy(d)
    path, v = _first(e, lambda x: type(x) is int and not isinstance(x, bool))
    _set(e, path, v + 1)
    assert RT.compare_json(e, d, RT.CAL3), path
    t = _f(CAL3, ".txt").read_text()
    assert RT.compare_text(t.replace(" OUT", " NEAR", 1), t, RT.CAL3)


def test_the_truncated_dump_rule_is_narrow():
    t = _f(PC2, ".txt").read_text()
    line = next(ln for ln in t.splitlines() if "[TRUNCATED: " in ln)
    m = RT._TRUNC.match(line)
    body, n = m["body"], int(m["n"])
    i = body.rindex(", ")
    # a full-precision float printed two digits shorter (a 1e-15 move) shifts the cut by two characters and the count
    # by two: passes, as pc2.read.txt line 10 did on macOS 27.2.1
    f = re.search(r"\d\.\d{15,}", body)
    shorter = body[:f.end() - 2] + body[f.end():] + "XX"
    moved = line.replace(body, shorter).replace(f"[TRUNCATED: {n} ", f"[TRUNCATED: {n - 2} ")
    assert len(shorter) == len(body) and RT.compare_text(moved, line) == []
    # the same float moved beyond the tolerance fails
    wrong = body[:f.start()] + str(float(f.group()) * 1.001)[:f.end() - f.start()] + body[f.end():]
    assert len(wrong) == len(body) and RT.compare_text(line.replace(body, wrong), line)
    # a changed label inside the body fails; a changed wording after the cut fails
    lab = body.index('"status": "') + len('"status": "')
    assert RT.compare_text(line[:2 + lab] + "X" + line[3 + lab:], line)
    assert RT.compare_text(line.replace("the whole reading is in", "the reading is in"), line)
    # a body cut at a different length fails
    assert RT.compare_text(line.replace(body, body[:i]), line)
    # the rule applies only when both lines are truncated dumps
    assert RT.compare_text(line.replace(" ... [TRUNCATED", " ... [CUT"), line)

