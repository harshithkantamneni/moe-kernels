#!/usr/bin/env python
# ruff: noqa: E501  (published page paths and scorer keys are quoted whole)
"""Appendix A (the full ledger) and the summary counts table (OUTLINE section 3),
assembled from committed files.

    python scripts/paper/ledger_tables.py [--out DIR]
        writes DIR/appendix_a.md, DIR/appendix_a.csv, DIR/data/appendix_a_cells.csv,
               DIR/summary_counts.md, DIR/summary_counts.csv

Inputs: scripts/paper/ledger_rows.json (the ledger as the outline types it) and the
committed scorer outputs (scripts/scoring/crossmodel/*.json and *.txt,
scripts/scoring/rental1/*.score.json, scripts/scoring/rental2/*.score.json,
scripts/scoring/rental3/*.score.json, scripts/scoring/session0927/score.json). Nothing is
fitted, measured or rescored here.

APPENDIX A. One line per ledger row: the typed verdict and, where a committed
scorer output carries a verdict or the numbers that decide one, the verdict read
from that file (`CHECKS` below says which file and key) and whether they agree. A
row agrees when every verdict word the scorer output carries (HELD, FALSIFIED,
INCONCLUSIVE, NEITHER, NOT SCORED, NOT SCORABLE, NOT RUN, NOT HELD, SELECTED, NL,
H1, FAILED, UNDECIDED, NOT ANSWERED) appears in the typed verdict. Rows 13 and 18
have no scorer: they print "hand-scored, no scorer" and the session README that holds
the count. Rows 1 to 9 carry the registered procedure's verdicts, read from
scripts/scoring/session0927/score.json (a scorer written 2026-10-05, after the pages); rows 2
and 6 note that the 2026-09-27 hand scoring read HELD.
Rows whose scorer output carries no single verdict print "typed from" and the file.
The per-cell tables are data/appendix_a_cells.csv: every cell of the cross-model
time tests (rows 12, 17, 20, 25, 29, 33, 77) and byte tests (rows 14, 19, 23, 28, 32,
36, 40, 79), as the committed outputs hold them (row 77's resid is put as predicted /
measured - 1, the other rows' convention; row 79's are rental 3's ALL rows).

SUMMARY COUNTS. Derived, not typed: primary time tests (rows 12, 17, 20, 24, 25, 29,
33, 37, 41, 77), per-GEMM floor tests counted once per registration on the base
capture (rows 10, 15, 22, 26, 30, 34, 38, 39, 42, 43, 47), PRIVATE byte tests per
GEMM on the five 2026-09-29 GH200 registrations (rows 23, 28, 32, 36, 40), each
from the scorer output's own verdict or stats against the registered bar, and the
code-docstring tests of the 2026-09-27 board (rows 1 to 8) from session0927/score.json,
and the floor-law verdicts of rental 2 part 4 (row 60) and rental 3 part B (row 83).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
XM = ROOT / "scripts/scoring/crossmodel"
SC1 = ROOT / "scripts/scoring/rental1"
SC2 = ROOT / "scripts/scoring/rental2"
SC3 = ROOT / "scripts/scoring/rental3"
PUB = ROOT / "results/published"
ROWS = ROOT / "scripts/paper/ledger_rows.json"
BAR_RMS, LIMIT = 0.02, 0.05

SESS = ROOT / "scripts/scoring/session0927/score.json"
HAND = {}
HAND[13] = "results/published/2026-09-28-nvidia_gh200_480gb-8x22b-session/session/README.md"
HAND[18] = "results/published/2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/session/README.md"


def _j(p: Path) -> dict:
    return json.loads(p.read_text())


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


# ------------------------------------------------------------------- readers
def floor(case: str, gemms=("w1",)) -> tuple[str, str, dict]:
    f = XM / f"{case}.registered.json"
    fl = _j(f)["floor"]["r3f-g64.json"]
    v = {g: fl[g]["verdict"] for g in gemms}
    return _join(v), f"{_rel(f)} floor r3f-g64.json (base) " + ", ".join(gemms), v


def _join(v: dict) -> str:
    vals = set(v.values())
    return vals.pop() if len(vals) == 1 else ", ".join(f"{g} {x}" for g, x in v.items())


def slope(case: str) -> tuple[str, str, dict]:
    f = XM / f"{case}.registered.json"
    s = _j(f)["slope_g8"]
    reg, band, meas = s["registered"], s["band"], s["measured"]
    ok = []
    for i, (_k, m) in enumerate(meas.items()):
        if band:
            ok.append(band[0] <= m <= band[1])
        else:
            r = reg[i] if isinstance(reg, list) else reg
            ok.append(abs(m / r - 1) <= 0.02)
    return ("HELD" if all(ok) else "FALSIFIED"), f"{_rel(f)} slope_g8 against band or 2%", {}


def _time(st: dict) -> str:
    return "HELD" if st["rms"] is not None and st["rms"] <= BAR_RMS and st["beyond_5pct"] == 0 \
        else "FALSIFIED"


def time_reg(case: str) -> tuple[str, str, dict]:
    f = XM / f"{case}.registered.json"
    st = _j(f)["time_predicted_bytes"]["shared+private"]
    return _time(st), f"{_rel(f)} time_predicted_bytes shared+private", st


def time_own(name: str) -> tuple[str, str, dict]:
    f = XM / f"{name}.score.json"
    st = _j(f)["sets"]["shared+private"]
    return _time(st), f"{_rel(f)} sets shared+private", st


def time_refused() -> tuple[str, str, dict]:
    f = XM / "granite-3.0-3b-a800m.score.txt"
    return ("NOT SCORABLE" if "REFUSED" in f.read_text() else "?"), f"{_rel(f)} REFUSED", {}


def not_run(model_dir: str) -> tuple[str, str, dict]:
    hits = [p for p in PUB.glob("2026-09-29*-nvidia_h100*") if model_dir in p.name] + \
        [p for p in PUB.glob("2026-09-3*-nvidia_h100*")] + [p for p in PUB.glob("2026-10-*-nvidia_h100*")]
    return ("NOT RUN" if not hits else "RUN"), "results/published: no H100 session after 2026-09-29", {}


def bytes_named(case: str) -> tuple[str, str, dict]:
    f = XM / f"{case}.registered.json"
    b = _j(f)["bytes"]
    key = next(k for k in b if k.startswith("named set"))
    return ("HELD" if b[key]["beyond_5pct"] == 0 else "FALSIFIED"), f"{_rel(f)} bytes {key}", b[key]


def private(case: str) -> tuple[str, str, dict]:
    f = XM / f"{case}.registered.json"
    b = _j(f)["bytes"]
    v = {g: ("HELD" if b[f"private {g}"]["beyond_5pct"] == 0 else "FALSIFIED") for g in ("w1", "w2")}
    return _join(v), f"{_rel(f)} bytes private w1, private w2", v


def key(path: Path, *keys) -> tuple[str, str, dict]:
    d = _j(path)
    for k in keys:
        d = d[k]
    return str(d), f"{_rel(path)} " + ".".join(keys), {}


def many(path: Path, keysets) -> tuple[str, str, dict]:
    vals = []
    for ks in keysets:
        d = _j(path)
        for k in ks:
            d = d[k]
        vals.append(str(d))
    return "; ".join(vals), f"{_rel(path)} " + ", ".join(".".join(ks) for ks in keysets), {}


def models_P(p: str) -> tuple[str, str, dict]:
    f = SC2 / "launch.score.json"
    m = _j(f)["models"]
    return "; ".join(f"{k} {v['P'][p]['verdict']}" for k, v in m.items()), \
        f"{_rel(f)} models.*.P.{p}.verdict", {}


def controls() -> tuple[str, str, dict]:
    f = SC1 / "l2.score.json"
    c = _j(f)["rules"]["controls"]
    held = sum(bool(v["ok"]) for v in c.values())
    return f"{held} of {len(c)} HELD", f"{_rel(f)} rules.controls ok", {}


def t3() -> tuple[str, str, dict]:
    f = SC1 / "atile.score.json"
    t = _j(f)["T3"]
    held = [k.split("-")[-1] for k, v in t.items()
            if isinstance(v, dict) and v.get("verdict") == "PASS"]
    return "HELD " + ", ".join(held), f"{_rel(f)} T3.*.verdict (PASS)", {}


FL = SC1 / "floor.score.json"
LA = SC1 / "launch.score.json"
AT = SC1 / "atile.score.json"
L2 = SC1 / "l2.score.json"
W1 = SC2 / "w1floor.score.json"
CO = SC2 / "const.score.json"
KN = SC2 / "knobs.score.json"
E2E = SC3 / "e2e.score.json"
FLW = SC3 / "floorlaw.score.json"
ZF = SC3 / "zform.score.json"
STG = SC3 / "stages.score.json"
FLU = SC3 / "flush.score.json"

def session0927(row: int) -> tuple[str, str, dict]:
    v = _j(SESS)["rows"][str(row)]["verdict"]
    return v, f"{_rel(SESS)} rows.{row}.verdict (written after the pages)", {}


CHECKS = {
    **{r: (lambda r=r: session0927(r)) for r in range(1, 10)},
    10: lambda: floor("8x22b"), 11: lambda: slope("8x22b"), 12: lambda: time_reg("8x22b"),
    14: lambda: bytes_named("8x22b"),
    15: lambda: floor("qwen2"), 16: lambda: slope("qwen2"), 17: lambda: time_reg("qwen2"),
    19: lambda: bytes_named("qwen2"),
    20: lambda: time_own("olmoe-1b-7b"), 21: lambda: slope("olmoe"), 22: lambda: floor("olmoe"),
    23: lambda: private("olmoe"), 24: lambda: not_run("olmoe"),
    25: lambda: time_own("qwen1.5-moe-a2.7b"), 26: lambda: floor("qwen1.5", ("w1", "w2")),
    27: lambda: slope("qwen1.5"), 28: lambda: private("qwen1.5"),
    29: lambda: time_own("phi-3.5-moe"), 30: lambda: floor("phi3.5", ("w1", "w2")),
    31: lambda: slope("phi3.5"), 32: lambda: private("phi3.5"),
    33: lambda: time_own("jetmoe-8b"), 34: lambda: floor("jetmoe", ("w1", "w2")),
    35: lambda: slope("jetmoe"), 36: lambda: private("jetmoe"),
    37: time_refused, 38: lambda: floor("granite", ("w2",)), 39: lambda: floor("granite", ("w1",)),
    40: lambda: private("granite"), 41: lambda: not_run("granite"),
    42: lambda: floor("jetmoe-floor", ("w1", "w2")), 43: lambda: floor("mixtral-floor", ("w1", "w2")),
    44: lambda: key(FL, "captures", "r3f-g64", "F1", "verdict"),
    45: lambda: key(FL, "captures", "r3f-g64", "F2", "verdict"),
    46: lambda: key(FL, "captures", "r3f-g64", "F3", "verdict"),
    47: lambda: (lambda d: (_join({g: d[g]["verdict"] for g in ("w1", "w2")}),
                            f"{_rel(FL)} captures.r3f-g64.F4.w1, w2", {}))(
        _j(FL)["captures"]["r3f-g64"]["F4"]),
    48: lambda: key(LA, "P", "P1", "verdict"),
    49: lambda: many(LA, [("P", p, "verdict") for p in ("P2", "P3", "P4", "P6", "P8")]),
    50: lambda: key(LA, "P", "P5", "verdict"),
    51: lambda: key(L2, "rules", "H1_vs_H0_tp2_w2", "verdict"),
    52: lambda: key(L2, "rules", "H1_vs_H0_tp4_w1", "verdict"),
    53: lambda: key(L2, "rules", "x_only_rival", "verdict"),
    54: controls,
    56: lambda: many(AT, [("T1", m, "verdict") for m in ("olmoe-1b-7b", "qwen2-57b-a14b")]),
    57: lambda: many(AT, [("T2", "cands", c, "verdict") for c in _j(AT)["T2"]["cands"]]),
    58: t3,
    59: lambda: many(AT, [("T4", c, "verdict") for c in _j(AT)["T4"] if isinstance(_j(AT)["T4"][c], dict) and "verdict" in _j(AT)["T4"][c]]),
    60: lambda: key(W1, "base (PRIMARY)", "families", "verdict"),
    61: lambda: key(W1, "base (PRIMARY)", "families", "co_primary"),
    63: lambda: many(CO, [("K", k, "verdict") for k in ("K1", "K2", "K4")]),
    64: lambda: key(CO, "K", "K3", "verdict"),
    65: lambda: key(CO, "identification"),
    66: lambda: models_P("P0"), 67: lambda: models_P("P1"),
    68: lambda: "; ".join(models_P(p)[0] for p in ("P2", "P4", "P5")),
    69: lambda: key(SC2 / "launch.score.json", "P3_pooled", "verdict"),
    70: lambda: models_P("P6"), 71: lambda: models_P("P8"),
    72: lambda: many(KN, [("tests", "tp2 w2 s8 (PRIMARY)", "verdict"),
                        ("tests", "tp4 w1 s8 (PRIMARY, NL vs lag)", "verdict")]),
    73: lambda: key(KN, "tests", "tp2 w2 s6 (secondary, the curve's shape)", "verdict"),
    74: lambda: many(KN, [("tests", "8x7B w2 s8 (x-invariance, tail)", "verdict")]),
    75: lambda: many(KN, [("tests", "H2c BLOCK_K at G=1 (tail)", "l2bk128 w1", "verdict"),
                        ("tests", "H2c BLOCK_K at G=1 (tail)", "l2bk32 w1", "verdict"),
                        ("tests", "H3 slot pad 7 (tail)", "verdict")]),
    77: lambda: key(E2E, "E1", "verdict"), 78: lambda: key(E2E, "E2", "verdict"),
    79: lambda: many(E2E, [("E3", "verdict"), ("E3s", "verdict")]),
    80: lambda: key(E2E, "E4", "verdict"), 81: lambda: key(E2E, "E6", "verdict"),
    83: lambda: many(FLW, [("verdict",), ("hypotheses", "CEIL"), ("hypotheses", "FLUID")]),
    84: lambda: key(FLW, "tests", "B6", "verdict"), 85: lambda: key(FLW, "tests", "B7", "verdict"),
    86: lambda: many(ZF, [("verdict",), ("forms", "PROP", "status")]),
    87: lambda: many(STG, [("verdict",)] + [("hypotheses", h, "status")
                                         for h in ("DEPTH", "WIDTH", "U-OCC", "U-DEPTH")]),
    88: lambda: many(FLU, [("hypotheses", "OVERLAP"), ("hypotheses", "SAT")]),
}
# 68 returns a bare string; wrap it with its source.
_c68 = CHECKS[68]
CHECKS[68] = lambda: (_c68(), f"{_rel(SC2 / 'launch.score.json')} models.*.P.{{P2,P4,P5}}.verdict", {})
#: Rows whose scorer output carries no single verdict: the file the typed verdict is read from.
TYPED = {55: "scripts/scoring/rental1/l2.score.json rules.secondary_G2 and README",
         62: "scripts/scoring/rental2/SCORES.md part 4", 76: "docs/registered/README.md \"Scored 2026-10-02\"",
         82: "scripts/scoring/rental3/replicate.score.json RK and SCORES.md part R"}

WORDS = re.compile(r"NOT SCORABLE|NOT SCORED|NOT RUN|NOT HELD|INCONCLUSIVE|FALSIFIED|NEITHER|"
                   r"SELECTED|FAILED|HOLDS|HOLD|HELD|PASS|H1|NL|refuted|RECORD|UNDECIDED|"
                   r"NOT ANSWERED|not answered")
NORM = {"HOLDS": "HELD", "HOLD": "HELD", "PASS": "HELD", "refuted": "FALSIFIED",
        "not answered": "NOT ANSWERED"}


def words(s: str) -> set[str]:
    return {NORM.get(w, w) for w in WORDS.findall(s)}


def build() -> dict:
    rows = _j(ROWS)["rows"]
    out, last_reg = [], ""
    for r in rows:
        reg = r["registration"]
        if reg == "same" or reg.startswith("same +"):
            reg = last_reg + (reg[4:] if reg.startswith("same +") else "")
        else:
            last_reg = reg
        n = r["row"]
        if n in HAND:
            read, src, agree = "hand-scored, no scorer", HAND[n], "n/a"
        elif n in CHECKS:
            read, src, _ = CHECKS[n]()
            agree = "yes" if words(read) <= words(r["verdict"]) and words(read) else "NO"
        else:
            read, src, agree = "typed from", TYPED.get(n, ""), "n/a"
        out.append({**r, "registration": reg, "read_verdict": read, "read_from": src,
                    "agree": agree})
    return {"rows": out, "summary": summary(), "cells": cells()}


def summary() -> list[tuple]:
    t = {12: time_reg("8x22b"), 17: time_reg("qwen2"), 20: time_own("olmoe-1b-7b"),
         25: time_own("qwen1.5-moe-a2.7b"), 29: time_own("phi-3.5-moe"),
         33: time_own("jetmoe-8b"), 37: time_refused(), 24: not_run("olmoe"),
         41: not_run("granite"), 77: key(E2E, "E1", "verdict")}
    fl = {10: floor("8x22b"), 15: floor("qwen2"), 22: floor("olmoe"),
          26: floor("qwen1.5", ("w1", "w2")), 30: floor("phi3.5", ("w1", "w2")),
          34: floor("jetmoe", ("w1", "w2")), 38: floor("granite", ("w2",)),
          39: floor("granite", ("w1",)), 42: floor("jetmoe-floor", ("w1", "w2")),
          43: floor("mixtral-floor", ("w1", "w2"))}
    f4 = _j(FL)["captures"]["r3f-g64"]["F4"]
    per_gemm = [(row, g, v) for row, (_s, _src, d) in fl.items() for g, v in d.items()]
    per_gemm += [(47, g, f4[g]["verdict"]) for g in ("w1", "w2")]
    pv = {row: private(c) for row, c in ((23, "olmoe"), (28, "qwen1.5"), (32, "phi3.5"),
                                         (36, "jetmoe"), (40, "granite"))}
    pg = [(row, g, v) for row, (_s, _src, d) in pv.items() for g, v in d.items()]
    miss = {row: private(c)[2] for row, c in ((14, "8x22b"), (19, "qwen2"))}

    def count(items, verdict):
        hits = [x for x in items if x[-1] == verdict]
        return len(hits), ", ".join(f"{x[0]}" + (f" {x[1]}" if len(x) == 3 else "") for x in hits)

    tv = [(row, v[0]) for row, v in sorted(t.items())]
    out = []
    for verdict in ("HELD", "FALSIFIED", "NOT SCORABLE", "NOT RUN"):
        k, which = count(tv, verdict)
        out.append(("primary time tests", verdict, k, which))
    for verdict in ("HELD", "FALSIFIED", "NOT SCORABLE"):
        k, which = count(sorted(per_gemm), verdict)
        out.append(("per-GEMM floor tests (base capture, each registration once)", verdict, k, which))
    for verdict in ("HELD", "FALSIFIED"):
        k, which = count(sorted(pg), verdict)
        out.append(("PRIVATE bytes per GEMM, five 2026-09-29 GH200 registrations", verdict, k, which))
    for row, d in miss.items():
        out.append((f"PRIVATE bytes per GEMM, row {row} (inside its byte test)", "FALSIFIED",
                    sum(v == "FALSIFIED" for v in d.values()),
                    ", ".join(g for g, v in d.items() if v == "FALSIFIED")))
    sess = [(row, session0927(row)[0]) for row in range(1, 9)]
    for verdict in ("HELD", "FALSIFIED", "UNDECIDED"):
        k, which = count(sess, verdict)
        out.append(("code-docstring tests, 2026-09-27 board (rows 1 to 8, the procedure's verdicts)",
                    verdict, k, which))
    out.append(("floor law, rental 2 part 4 (row 60)", key(W1, "base (PRIMARY)", "families",
                                                          "verdict")[0], 1, "60"))
    out.append(("floor law, rental 3 part B (row 83)", key(FLW, "verdict")[0], 1, "83"))
    return out


def cells() -> list[tuple]:
    out = []
    for row, case in ((12, "8x22b"), (17, "qwen2")):
        for c in _j(XM / f"{case}.registered.json")["time_predicted_bytes"]["cells"]:
            out.append((row, "time", case, c["arm"], c["G"], c["n"], "", c["measured_ms"],
                        c["registered_ms"], c["resid"]))
    for row, name in ((20, "olmoe-1b-7b"), (25, "qwen1.5-moe-a2.7b"), (29, "phi-3.5-moe"),
                      (33, "jetmoe-8b")):
        for c in _j(XM / f"{name}.score.json")["cells"]:
            out.append((row, "time", name, c["arm"], c["G"], c["n"], "", c["measured_ms"],
                        c["predicted_ms"], c["resid"]))
    for row, case in ((14, "8x22b"), (19, "qwen2"), (23, "olmoe"), (28, "qwen1.5"),
                      (32, "phi3.5"), (36, "jetmoe"), (40, "granite")):
        for c in _j(XM / f"{case}.registered.json")["bytes"]["cells"]:
            out.append((row, "bytes q", case, c["arm"], c["G"], c["n"], c["gemm"], c["q_meas"],
                        c["q_reg"], c["rel"]))
    e2e = _j(E2E)
    for c in e2e["E1"]["cells"]:
        out.append((77, "time", "qwen2-57b-a14b-tp8", c["arm"], c["G"], c["n"], "", c["T_meas"],
                    c["T_M"], c["T_M"] / c["T_meas"] - 1))
    for key3 in ("E3", "E3s"):
        for c in e2e[key3]["rows"]:
            arm = "private" if key3 == "E3" else "shared"
            out.append((79, "bytes q", "qwen2-57b-a14b-tp8", arm, c["G"], c["n"], c["gemm"], c["q"],
                        c["q_pred"], c["rel"]))
    return out


def _f(x):
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def _md_cell(s: str) -> str:
    return str(s).replace("|", "/").replace("\n", " ")


def write(b: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    cols = ["row", "test", "registration", "verdict", "number", "label", "scored_in",
            "read_verdict", "read_from", "agree"]
    with (out / "appendix_a.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in b["rows"]:
            w.writerow([r[c] for c in cols])
    n_check = sum(r["agree"] in ("yes", "NO") for r in b["rows"])
    n_yes = sum(r["agree"] == "yes" for r in b["rows"])
    md = ["# Appendix A: the full ledger", "",
          "Generated by `python scripts/paper/ledger_tables.py` from `scripts/paper/ledger_rows.json` "
          "(the outline's ledger, transcribed) and the committed scorer outputs. "
          f"{n_check} rows are checked against a scorer output and {n_yes} agree; rows 13 and 18 "
          "are hand-scored with no scorer (gap 6.13); rows 1 to 9 carry the procedure's verdicts "
          "from `scripts/scoring/session0927/score.json`, written 2026-10-05 after the pages (rows "
          "2 and 6 differ from the 2026-09-27 hand scoring: see `docs/FINDINGS.md`, the 2026-09-27 "
          "section); the rest are typed from the file named. Per-cell tables: "
          "`docs/paper/data/appendix_a_cells.csv`.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(_md_cell(r[c]) for c in cols) + " |" for r in b["rows"]]
    (out / "appendix_a.md").write_text("\n".join(md) + "\n")
    with (out / "data/appendix_a_cells.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["row", "quantity", "case", "arm", "G", "n", "gemm", "measured", "predicted",
                    "resid"])
        for c in b["cells"]:
            w.writerow([_f(x) for x in c])
    scols = ["group", "verdict", "count", "rows"]
    with (out / "summary_counts.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(scols)
        w.writerows(b["summary"])
    smd = ["# Summary counts", "",
           "Generated by `python scripts/paper/ledger_tables.py` from the committed scorer outputs "
           "(each row's verdict or stats against its registered bar; see the script's docstring).",
           "", "| " + " | ".join(scols) + " |", "|" + "---|" * len(scols)]
    smd += ["| " + " | ".join(str(x) for x in r) + " |" for r in b["summary"]]
    (out / "summary_counts.md").write_text("\n".join(smd) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs/paper")
    a = ap.parse_args(argv)
    write(build(), a.out)
    print(f"ledger tables: written under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
