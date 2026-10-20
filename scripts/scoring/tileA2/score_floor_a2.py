#!/usr/bin/env python
"""Score tile rental A2's floor (docs/registered/2026-10-20-tileA2-floor-gh200.json): rental A's pinned scorer
scripts/scoring/tileA/score_floor.py, run unchanged inside a2common.scoring at the lock A2 ran.

    python scripts/scoring/tileA2/score_floor_a2.py <repo> <tree> <out> [--lock L] [--rules tileA2|tileA]

Reads under <tree>: <date>-<card>-<model>-<label>-r3-counters/lock<L>/r3c-g64.json (f128, f64w8, f64w4,
f32w4) and <out>/census.score.json. THE BAND IN FORCE (tileA2-floor, REVIEW-A2 2c): the primary counter
band [0.975 L, L]; when it leaves tcommon.joint_cF None on the f128 pages, the registered fallback
[0.959 L, L], labelled, and that band is then every counter scorer's for the rental (score_bytes_a2
reads it from floor.score.json). --rules tileA reproduces rental A's own score (1710, [1640, 1710]).
--lock defaults to the session's gh200-driver/a2-lock.env. Writes <out>/floor.score.{json,txt}.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402
#: rental A's scorer by path: rental 1 and rental 4 also hold a score_floor.py and a score_dead.py
SF = A2.TC._load("tile_score_floor", A2.TILEA / "score_floor.py")

TC = A2.TC
PART = SF.PART


def calibration(tree: Path, rules: A2.Rules) -> tuple[dict | None, dict]:
    """tcommon.joint_cF over the f128 pages in `rules`' band, and each page's state."""
    with A2.scoring(rules):
        pages, why = SF.pages_of(tree, "f128")
        return TC.joint_cF(pages), why


def fit_64(tree: Path, rules: A2.Rules) -> tuple[dict | None, dict]:
    """tcommon.joint_cF over the f64w8 pages in `rules`' band (finalise refuses without it when no
    accounting is selected)."""
    with A2.scoring(rules):
        pages, why = SF.pages_of(tree, "f64w8")
        return TC.joint_cF(pages), why


def choose_band(tree: Path, lock: float) -> tuple[A2.Rules, dict]:
    """The registered band rule: the primary band, else the fallback when the primary leaves no fit."""
    prim = A2.Rules.tileA2(lock, "primary")
    fit, why = calibration(tree, prim)
    rec = {"primary": [A2.r6(x) for x in prim.band], "primary_fit": "fit" if fit is not None else "None",
           "primary_pages": why}
    if fit is not None:
        return prim, dict(rec, which="primary")
    fb = A2.Rules.tileA2(lock, "fallback")
    fit2, why2 = calibration(tree, fb)
    rec.update(fallback=[A2.r6(x) for x in fb.band], fallback_fit="fit" if fit2 is not None else "None",
               fallback_pages=why2)
    if fit2 is not None:
        return fb, dict(rec, which="fallback (registered: the primary band left joint_cF None)")
    return prim, dict(rec, which="primary (neither band gives a calibration)")


#: rental A's SEEN out-of-band fit (DESIGN-A2 7d, design-A2 work/fit.py): c_128 and its se, cycles, at 1601 to 1640 MHz
SEEN_C128, SEEN_C128_SE, SEEN_CLOCKS = 698.67, 10.63, "1601 to 1640 MHz (rental A, out of band, under the 1710 lock)"


def cross_clock(cal: dict, lock: float) -> dict:
    """REVIEW-A2 7c (build review fix 5), DESCRIPTIVE: A2's c_128 at L against rental A's SEEN out-of-band fit."""
    c, se = cal.get("c"), (cal.get("se") or {}).get("c")
    if c is None or se is None:
        return {"verdict": "NOT READ: no calibration"}
    d = float(c) - SEEN_C128
    z = d / (float(se) ** 2 + SEEN_C128_SE ** 2) ** 0.5
    return {"label": "DESCRIPTIVE (the only cross-clock wgmma reading; decode-tilesource)", "a2_c128": A2.r6(c),
            "a2_se": A2.r6(se), "a2_lock_mhz": lock, "rental_A_c128": SEEN_C128, "rental_A_se": SEEN_C128_SE,
            "rental_A_clocks": SEEN_CLOCKS, "difference": A2.r6(d), "relative": A2.r6(d / SEEN_C128), "z": A2.r6(z)}


def score(repo: Path, tree: Path, out: Path, rules: A2.Rules) -> tuple[dict, list[str]]:
    if not rules.label.startswith("tileA2"):
        with A2.scoring(rules):
            res = SF.score(repo, tree, out)
        return res, SF.lines(res)
    use, rec = choose_band(tree, rules.lock)
    with A2.scoring(use):
        res = SF.score(repo, tree, out)
    res["lock_mhz"] = rules.lock
    res["band_in_force"] = rec
    res["lock_gate_note"] = ("lock_gate.RATIO_REF holds only 1710 (lock_gate.py:127): at L the counter gate is the "
                             "nvidia-smi bracket and the null / spread readings, no base-twin level reading")
    xc = cross_clock(res.get("calibration") or {}, rules.lock)
    res["cross-clock c_128 (DESCRIPTIVE, REVIEW-A2 7c)"] = xc
    out_lines = SF.lines(res) + [f"LOCK {rules.lock:g} MHz; BAND IN FORCE {rec['which']} {list(use.band)}"]
    if "z" in xc:
        out_lines.append(f"CROSS-CLOCK c_128 (DESCRIPTIVE): A2 {xc['a2_c128']} +- {xc['a2_se']} at {rules.lock:g} MHz "
                         f"against rental A's {SEEN_C128} +- {SEEN_C128_SE} at {SEEN_CLOCKS}: difference {xc['difference']} "
                         f"({100 * xc['relative']:+.2f}%), z {xc['z']}")
    else:
        out_lines.append(f"CROSS-CLOCK c_128 (DESCRIPTIVE): {xc['verdict']}")
    return res, out_lines


def main(argv=None) -> int:
    a, rules = A2.parse_cli(argv if argv is not None else sys.argv[1:], __doc__)
    if a is None:
        return A2.usage(__doc__)
    repo, tree, out = (Path(x) for x in a)
    res, lines = score(repo, tree, out, rules)
    print(TC.C3.write_score(out, PART, res, lines), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
