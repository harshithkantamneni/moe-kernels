"""Tile rental A2's shared rules (docs/registered/2026-10-20-tileA2-* and 2026-10-20-tileB2-*): the lock as data.

One module for A2's registration generator (regtile_a2.py, register_a2.py), its scorer wrappers
(score_*_a2.py), its VM gate helper (scripts/tileA2_gate.py) and finalise_a2.py, so a rule is computed
one way on every side. It imports the tile stage's pinned tcommon and changes nothing on disk: the
rental-A scorers (scripts/scoring/tileA/score_*.py) run unchanged INSIDE `scoring(rules)`, which sets,
in memory and for that block only:

  r6common.LOCK_MHZ, G4_MIN_MHZ, BAND_MHZ   on the ONE r6common module object tcommon loaded (TC.C6; the
                                            context refuses if another copy is loaded), read by
                                            clock_rule (the counter band) and row_verdict (G4);
  tcommon.NAMES                             the registration names TC.registration reads (tileA2 / tileB2);
  tcommon.counter_page                      a copy whose glob is lock<L>/r3c-g<G>.json instead of the
                                            literal lock1710 (tcommon.py:460);
  r3_timing_model_v4L.at_lock(L)            v4 priced at L by P1's rule (c x 1710 / L, every tile).

The timed pages' dips need nothing here: r7common.rebuild reads the page's own lock_mhz (r7-median-run3:
a read is dipped under lock - 15 MHz), and R3 records each repeat's slip against its own --lock-mhz.

THE RULES AT LOCK L (tileA2-gates; DESIGN-A2 2.2, REVIEW-A2 2):
  at 1710        rental A's own rules exactly (band [1640, 1710], the band derived at that lock; G4 1705):
                 no conversion, v4L is v4, and the fallback band [0.959 x 1710, 1710] is the same band;
  counter band   below 1710, [L x (1 - 0.025), L]: 2.5% = 2 x the 1.02% largest held-lock shortfall of rental 3's
                 tp4 floor own-clock cells at 1410, rounded up to 0.5% (a NEW rule in lock_gate's
                 convention: the 1710 band [1640, 1710] is 4.1% wide and does not follow it).
  fallback band  [L x (1 - 0.041), L] (the 1710 band's width scaled), applied ONLY when the primary
                 band leaves tcommon.joint_cF None on the f128 pages; then it is the counter band of
                 every counter scorer of the rental (floor and bytes), labelled FALLBACK.
  G4             cell clock >= L - 5 MHz (1405 at 1410, as 1705 at 1710).
  dips           r7-median-run3 at the page's lock (1395 is not a dip at 1410, 1380 is).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TILEA = HERE.parent / "tileA"
for p in (str(HERE), str(TILEA), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tcommon as TC  # noqa: E402  (first: it puts the repo and scripts/ on sys.path)
import r3_timing_model_v4L as V4L  # noqa: E402

C6 = TC.C6
CARD = TC.CARD
DATE = "2026-10-20"
#: the owner's decisions of 2026-10-20: a 900 W GPU power limit (A2, and rental B2 at the limit A2 held; the
#: 2026-09-26 decision's 700 W stands for every other rental) and the lock ladder 1710, 1410, 1305: the first
#: rung whose probe passes is the lock of every later unit
LOCK_LADDER = (1710.0, 1410.0, 1305.0)
LOCK_LOWEST = min(LOCK_LADDER)
POWER_LIMIT_W = 900
FIT_CLOCK_MHZ = V4L.FIT_CLOCK_MHZ
#: the tile stage's registration names and rental A's rules, captured at import (before any swap)
NAMES_TILEA = dict(TC.NAMES)
RULES_TILEA = (float(C6.LOCK_MHZ), float(C6.G4_MIN_MHZ), tuple(float(x) for x in C6.BAND_MHZ))
BAND_FRAC = 0.025
BAND_FALLBACK_FRAC = 0.041
G4_BELOW_MHZ = 5.0
DIP_BELOW_MHZ = 15.0
PARTS_A2 = ("census", "floor", "bytes", "e1cal", "dead", "rep", "convl", "gates")
PARTS_B2 = ("e1tile", "stop")
NAMES_A2 = {**{p: f"{DATE}-tileA2-{p}-gh200" for p in PARTS_A2}, **{p: f"{DATE}-tileB2-{p}-gh200" for p in PARTS_B2}}
TILESOURCE = f"{DATE}-decode-tilesource-gh200"
HIST = f"docs/registered/{DATE}-tileA2-hist"
PREDICTIONS_B2 = f"docs/registered/{DATE}-tileB2-predictions-gh200.json"
PLAN_A2 = "scripts/plans/rentalA2-tile-2026-10.plan"
PLAN_B2 = "scripts/plans/rentalB2-tile-2026-10.plan"
#: A2's score directory (REVIEW-A2 build fix 4): the score files beside A2's scorers, as rental A's beside its own
SCORE_DIR_A2 = "scripts/scoring/tileA2"
DRIVER_A2 = "scripts/gh200_tileA2_session.sh"


def r6(x):
    return TC.r6(x)


def band_of(lock: float, which: str = "primary") -> tuple[float, float]:
    """The counter band at L: rental A's registered [1640, 1710] at 1710 (both bands), else the A2 rule."""
    f = {"primary": BAND_FRAC, "fallback": BAND_FALLBACK_FRAC}[which]
    if float(lock) == FIT_CLOCK_MHZ:
        return tuple(RULES_TILEA[2])
    return (float(lock) * (1.0 - f), float(lock))


@dataclasses.dataclass(frozen=True)
class Rules:
    """The lock-dependent rules a scorer runs under, and which registrations it reads."""
    lock: float
    band: tuple
    g4: float
    names: dict
    label: str

    @classmethod
    def tileA(cls) -> Rules:
        """Rental A's own rules (1710, [1640, 1710], G4 1705, the 2026-10-14 names): the wrappers at
        these rules must reproduce rental A's scores (the L = 1710 equality check)."""
        lock, g4, band = RULES_TILEA
        return cls(lock, band, g4, dict(NAMES_TILEA), "tileA (rental A's registrations at 1710)")

    @classmethod
    def tileA2(cls, lock: float, band: str = "primary") -> Rules:
        lock = float(lock)
        return cls(lock, band_of(lock, band), lock - G4_BELOW_MHZ, dict(NAMES_A2),
                   f"tileA2 at {lock:g} MHz ({band} counter band)")


def counter_page_at(tree: Path, model: str, label: str, G: int, reg_file: dict | None = None,
                    tile: list | None = None, *, lock: float):
    """tcommon.counter_page with the lock as data: the unit's one G counter page under
    lock<L>/r3c-g<G>.json, then the same tile and G5 checks, line for line."""
    hits = sorted(Path(tree).rglob(f"*-{CARD}-{model}-{label}-r3-counters/lock{int(round(lock))}/r3c-g{G}.json"))
    if len(hits) != 1:
        return None, None, f"NOT SCORED: {len(hits)} counter pages for {model} {label} G {G}"
    p = json.loads(hits[0].read_text())
    de = p.get("design") or {}
    if tile is not None:
        got = [int(de.get(k) or 0) for k in ("block_m", "block_n", "block_k", "num_warps", "num_stages")]
        if got != list(tile):
            return None, None, f"NOT SCORED: the page ran tile {got}, the registration's is {list(tile)}"
    if reg_file is not None:
        h = de.get("histogram") or {}
        if h.get("file_sha256") != reg_file["sha256"] or h.get("shuffle_seed") != reg_file.get("shuffle_seed"):
            return None, None, "NOT SCORED (G5): the page's histogram sha256 or shuffle seed is not the registered one"
        want = {(c["label"], c["n"]): c["counts_sha256"] for c in reg_file["cells"]}
        bad = [c for c in p.get("cells") or [] if want.get((str(c.get("histogram")), int(c["n"]))) != c.get("counts_sha256")]
        if bad:
            return None, None, "NOT SCORED (G5): a cell carries another counts sha256"
    return p, hits[0], "counted"


@contextlib.contextmanager
def scoring(rules: Rules):
    """The rental-A scorers' module state at `rules`, restored on exit (see the module docstring)."""
    if sys.modules.get("r6common") is not C6:
        raise RuntimeError("two r6common module objects are loaded: the swap would miss the one the scorers read")
    keep = (C6.LOCK_MHZ, C6.G4_MIN_MHZ, C6.BAND_MHZ, dict(TC.NAMES), TC.counter_page)

    def cp(tree, model, label, G, reg_file=None, tile=None, _lock=rules.lock):
        return counter_page_at(tree, model, label, G, reg_file, tile, lock=_lock)
    try:
        C6.LOCK_MHZ, C6.G4_MIN_MHZ, C6.BAND_MHZ = float(rules.lock), float(rules.g4), tuple(rules.band)
        TC.NAMES.clear()
        TC.NAMES.update(rules.names)
        TC.counter_page = cp
        with V4L.at_lock(rules.lock):
            yield rules
    finally:
        C6.LOCK_MHZ, C6.G4_MIN_MHZ, C6.BAND_MHZ = keep[0], keep[1], keep[2]
        TC.NAMES.clear()
        TC.NAMES.update(keep[3])
        TC.counter_page = keep[4]


def registration(repo: Path, part: str) -> dict:
    name = TILESOURCE if part == "tilesource" else NAMES_A2[part]
    return json.loads((Path(repo) / "docs" / "registered" / f"{name}.json").read_text())


def lock_of_session(tree: Path) -> float | None:
    """The lock A2's driver selected, from <tree>/../session/gh200-driver/a2-lock.env (A2_LOCK=...)."""
    for env in (Path(tree).parent / "session" / "gh200-driver" / "a2-lock.env",
                Path(tree) / "session" / "gh200-driver" / "a2-lock.env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("A2_LOCK="):
                    return float(line.split("=", 1)[1])
    return None


def parse_cli(argv, doc: str, *, nargs: int = 3):
    """<repo> <tree> <out> [--lock L] [--rules tileA2|tileA]: (args, rules). --rules tileA (rental A's
    registrations at 1710) is the equality check; tileA2 takes --lock or the session's a2-lock.env."""
    a = list(argv)
    opts = {}
    for flag in ("--lock", "--rules"):
        if flag in a:
            i = a.index(flag)
            if i + 1 >= len(a):
                return None, None
            opts[flag] = a[i + 1]
            del a[i:i + 2]
    if len(a) != nargs or any(x.startswith("--") for x in a):
        return None, None
    which = opts.get("--rules", "tileA2")
    if which == "tileA":
        if "--lock" in opts and float(opts["--lock"]) != RULES_TILEA[0]:
            return None, None
        return a, Rules.tileA()
    if which != "tileA2":
        return None, None
    lock = float(opts["--lock"]) if "--lock" in opts else lock_of_session(Path(a[1]))
    if lock is None:
        print("no --lock and no session a2-lock.env beside the tree: give the lock A2's driver selected", file=sys.stderr)
        return None, None
    return a, Rules.tileA2(lock)


def usage(doc: str) -> int:
    print(doc)
    return 2


def read_score(out: Path, part: str) -> dict | None:
    p = Path(out) / f"{part}.score.json"
    return json.loads(p.read_text()) if p.exists() else None


def band_in_force(out: Path, rules: Rules) -> Rules:
    """The counter band a counter scorer reads after the floor: the one floor.score.json recorded
    (primary, or the registered fallback when the primary left joint_cF None)."""
    if not rules.label.startswith("tileA2"):
        return rules
    fl = read_score(out, "floor") or {}
    which = ((fl.get("band_in_force") or {}).get("which")) or "primary"
    return Rules.tileA2(rules.lock, "fallback" if which.startswith("fallback") else "primary")
