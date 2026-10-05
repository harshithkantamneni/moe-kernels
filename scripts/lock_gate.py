#!/usr/bin/env python3
"""The lock check: was the nvidia-smi lock in force while ncu counted?

    python scripts/lock_gate.py                     # re-gate every published lock page
    python scripts/lock_gate.py results/published   # the same, over another tree
    python scripts/lock_gate.py --derive            # the held-lock statistics behind each tolerance

FL1 (a floor capture under `--floor-lock-mhz F`) and V10 (a byte page under
`--page-lock-mhz F`) ask one question: did the GEMMs run at F. Until
2026-10-05 both answered it with `lock_fit`: each GEMM's ncu duration fitted
as t0 + sm__cycles_elapsed.avg / f over its cells, f held to one 15 MHz step
of F, t0 held to [-1, 50] us, and every cell held to 3 steps of the lock's
line. THAT FIT IS DEGENERATE (scratchpad gate-audit/AUDIT.md, read against
every published page): the shortfall of sm__cycles_elapsed.avg against the
duration differs cell to cell, so t0 is not a fixed overhead and f absorbs
whatever part of it grows with cycles. Whenever t0 comes out small, f comes
out low. Every fit with t0 <= 4.4 us reads f <= 1676 MHz at a 1710 lock, and
the same t0 / f co-movement shows at ncu's pinned base clock, where no lock is
involved (Granite 1337 MHz - 0.5 us, Mixtral 1389 to 1411 MHz + 22 to 28 us).
Its single-cell check also fails n = 1 cells, whose duration is mostly the
part that does not scale with work. So FL1 / V10 failed captures whose lock
held, on small-shard models (JetMoE, OLMoE, Qwen1.5, Granite, rental 1's
tp4 / tp8) and on n = 1 cells (Phi-3.5 G=64, rental 1's tp2 atile). The one
real off-lock capture in the published data, the 2026-09-28 8x22B floor (the
lock not in force: nvidia-smi read 1980 MHz before and after), is a gross
error of about +5.5%, which the readings below see without any fit.

THE CHECK NOW READS INDEPENDENT CLOCKS ALREADY ON THE PAGE, and rejects only
what a held lock cannot produce. Every tolerance is two times the largest
deviation the published held-lock captures show, rounded up to 0.5% (or, for
nvidia-smi, the reading's own resolution), and `derive()` recomputes each
statistic from results/published (tests/test_lock_gate.py pins them). "Held"
for that derivation means nvidia-smi read the lock before and after the
capture, which every published lock page does except the 8x22B floor; the
statistics are therefore not taken from this gate's own verdicts.

  smi     nvidia-smi `clocks.sm` before and after the capture (`smi_before`,
          `smi_after`), each within SMI_TOL_MHZ of F. On the 117 published
          lock pages, 116 read F exactly in all 232 readings (1710, or 1005 on
          the rental-2 floor1005) and the 8x22B floor reads 1980 both sides.
          SMI_TOL_MHZ is one supported-clock step, the reading's resolution.
          It sees a lock that was not in force. It is an idle bracket and
          cannot see a sag under load, which is why the two readings below
          are added where they exist.
  null    the null kernel's own `sm_clock_mhz` (rental 2 floors, one per cell):
          the median over cells within NULL_LEVEL_TOL of F. Held captures read
          -1.29% to -1.50% (1684.4, 1687.5, 1687.9 MHz at 1710; 991.3 at 1005),
          a uniform bias of that 1000-cycle probe, so 2 x 1.50% = 3.0%, the
          rule rental 2's gate addendum (docs/registered/2026-10-02-rental2-
          addendum-gates.txt, rule 5) fixed and applied.
  ratio   for a capture with a base twin (the same cells captured at ncu's
          base clock, `twin_path`): per cell and GEMM R = (cycles / duration)
          at the lock over the same at base. The cell's own counter shortfall
          divides out. The median R over readable cells is compared with
          RATIO_REF[F], the median over the held captures (1.2198 at 1710):
          |median / RATIO_REF - 1| <= RATIO_LEVEL_TOL. Held captures span
          -1.39% (JetMoE floor 2026-09-30) to +0.97% (Granite), so 3.0%; the
          8x22B floor reads +5.5%. R is not a clock ratio by itself (the null
          kernel reads the base clock near 1508 MHz, the GEMMs' R implies
          1402), so no reference exists for a lock the published data never
          held, and there the level part is not read.

DRIFT WITHIN A CAPTURE is read cell by cell, where a reading exists per cell:

  ratio   every readable cell's R within RATIO_SPREAD_TOL of the capture's
          median R. Held captures: largest |R / median - 1| over readable
          cells 2.24% (JetMoE floor 2026-09-30, n=2 w2), so 4.5%.
  null    every cell's null-kernel clock within NULL_SPREAD_TOL of the
          capture's null median. Held captures: largest 1.05% (rental-2 tp4
          floor, n=13), so 2.5%. The null kernel is one fixed probe, so its
          offset is the same in every cell and the short-cell rule below does
          not apply to it.

CELLS TOO SHORT TO READ are left out of every per-cell GEMM check, by one
clock-free rule over the capture's own durations: per (arm, GEMM), b is the
median of the per-tread slopes (t[n_i+1] - t[n_i]) / (n_i+1 - n_i) between
consecutive treads, and a cell's offset share is (t - n b) / t, the part of
its duration that does not scale with its treads. A cell whose share exceeds
SHORT_SHARE_MAX is too short to read. On the published held floors every
n >= 2 cell reads 0.21 or less and every n = 1 cell 0.36 to 0.50, the cells
whose R strays to +3.5% (rental-2 tp4 n=1 w1) at a held lock; 0.30 sits in
that gap. A GEMM with fewer than two distinct treads has no slope, and its
cells are left out.

THE VERDICT. FAIL when any reading present is out of its tolerance; PASS when
none is and at least one level reading (smi, null, or the ratio's level) was
present; REFUSE (UNKNOWN to the exit-code table) when no level reading exists.
FALLBACK, stated: a page with no null kernel and no base twin (every published
byte page) is judged on the nvidia-smi bracket alone, and its measured text
says that drift within the capture was not tested. The old fit is computed
and printed beside every verdict as a DIAGNOSTIC (`lock_fit`, "old fit"), and
decides nothing.

Published page JSONs are never rewritten: `regate_path` recomputes a stored
page's verdict on read, and `stored_vs_recomputed` gives the line the readers
(r3_timing_model.py, wave_split_bytes.py) print, the stored verdict labelled
stale.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PUBLISHED = REPO / "results" / "published"

PASS, FAIL, REFUSE = "PASS", "FAIL", "REFUSE"
LOCK_GATES = ("FL1", "V10")

#: One supported-clock step on the GH200's NVML grid (the band the old fit held f to).
LOCK_STEP_MHZ = 15.0

# Tolerances: each is 2 x the largest deviation on the published held-lock pages,
# rounded up to 0.5% (module docstring; `derive` recomputes the statistics).
SMI_TOL_MHZ = LOCK_STEP_MHZ
NULL_LEVEL_TOL = 0.030
NULL_SPREAD_TOL = 0.025
RATIO_LEVEL_TOL = 0.030
RATIO_SPREAD_TOL = 0.045
#: The median over the published held captures of each capture's median R, by lock.
RATIO_REF: dict[float, float] = {1710.0: 1.2198}
#: The offset share above which a cell is too short to read.
SHORT_SHARE_MAX = 0.30
#: The margin and rounding every tolerance above is set by.
TOL_MARGIN, TOL_ROUND = 2.0, 0.005

# The old fit's bounds, kept for the diagnostic (`lock_fit`).
FIT_OFFSET_MAX_NS = 50_000.0
FIT_OFFSET_MIN_NS = -1_000.0
FIT_CELL_STEPS = 3.0


def tolerance_from(max_dev: float) -> float:
    """The tolerance a held-lock maximum deviation sets: TOL_MARGIN x it, rounded up."""
    return math.ceil(round(TOL_MARGIN * max_dev / TOL_ROUND, 9)) * TOL_ROUND


# --------------------------------------------------------------------------
# The old fit, now a printed diagnostic.
# --------------------------------------------------------------------------

def lock_fit(points: list[tuple[str, float | None, float | None]], gemm: str,
             lock: float) -> tuple[list[str], str]:
    """The pre-2026-10-05 FL1 / V10 test, kept as a DIAGNOSTIC. `points` are
    (label, cycles, ns) for one GEMM's cells. Fits duration = t0 + cycles / f
    and returns what the old gate would have named off the lock (empty when
    it would have passed) and the fit, as printed. It decides no verdict: the
    fit is degenerate (module docstring), so a low f beside a small t0 is the
    estimator, not the clock."""
    step = LOCK_STEP_MHZ
    off = [f"{label} no clock" for label, cyc, ns in points if not (cyc and ns)]
    pts = [(label, float(cyc), float(ns)) for label, cyc, ns in points if cyc and ns]
    if len({c for _, c, _ in pts}) < 3:
        off += [f"{label} {1e3 * c / t:.0f} MHz" for label, c, t in pts
                if abs(1e3 * c / t - lock) > step]
        return off, f"{gemm} per cell (too few cells to fit)"
    mx = statistics.fmean(c for _, c, _ in pts)
    my = statistics.fmean(t for _, _, t in pts)
    b = (sum((c - mx) * (t - my) for _, c, t in pts)
         / sum((c - mx) ** 2 for _, c, _ in pts))
    t0 = my - b * mx
    f = 1e3 / b if b > 0 else float("nan")
    fit = f"{gemm} {f:.1f} MHz + {t0 / 1e3:.1f} us over {len(pts)} cells"
    if not abs(f - lock) <= step:
        off.append(f"{gemm} fitted {f:.0f} MHz")
    if not FIT_OFFSET_MIN_NS <= t0 <= FIT_OFFSET_MAX_NS:
        off.append(f"{gemm} offset {t0 / 1e3:.1f} us")
    for label, c, t in pts:
        cell_step = 1e3 * c / (lock - step) - 1e3 * c / lock
        if abs(t - (t0 + 1e3 * c / lock)) > FIT_CELL_STEPS * cell_step:
            mhz = 1e3 * c / (t - t0) if t > t0 else float("nan")
            off.append(f"{label} {mhz:.0f} MHz")
    return off, fit


# --------------------------------------------------------------------------
# Reading a page.
# --------------------------------------------------------------------------

Key = tuple[str, int, str]          # (arm, tread n, GEMM)


def smi_clocks(record: dict | None) -> list[float]:
    """Every card's `clocks.sm` in one nvidia-smi reading, MHz (empty when none)."""
    out = []
    for row in (record or {}).get("rows") or []:
        try:
            out.append(float(row.get("clocks.sm")))
        except (TypeError, ValueError):
            continue
    return out


def page_kind(page: dict) -> str | None:
    """'floor' for a floor capture, 'page' for a byte page, None otherwise."""
    if not isinstance(page, dict) or not isinstance(page.get("cells"), list):
        return None
    if page.get("kind") == "floor":
        return "floor"
    if isinstance(page.get("ncu"), dict) and "design" in page:
        return "page"
    return None


def page_lock(page: dict) -> float | None:
    """The nvidia-smi lock a page was captured under, or None."""
    kind = page_kind(page)
    lock = ((page.get("clock") or {}).get("lock_mhz") if kind == "floor"
            else (page.get("ncu") or {}).get("lock_mhz") if kind == "page" else None)
    return None if lock is None else float(lock)


def page_smi(page: dict) -> tuple[dict | None, dict | None]:
    holder = page.get("clock") if page_kind(page) == "floor" else page.get("ncu")
    holder = holder or {}
    return holder.get("smi_before"), holder.get("smi_after")


def page_points(page: dict) -> dict[Key, tuple[float | None, float | None]]:
    """(cycles, ns) per (arm, n, GEMM): a floor's cell means, or a byte page's
    recorded `sm__cycles_elapsed.avg` over its mean `gpu_time_ns`."""
    out: dict[Key, tuple[float | None, float | None]] = {}
    kind = page_kind(page)
    for c in page.get("cells") or []:
        for g, v in (c.get("per_gemm") or {}).items():
            v = v or {}
            if kind == "floor":
                cyc, ns = v.get("sm__cycles_elapsed.avg"), v.get("gpu__time_duration.sum")
            else:
                cyc = ((c.get("recorded") or {}).get(g) or {}).get("sm__cycles_elapsed.avg")
                ns = v.get("gpu_time_ns")
            out[(str(c.get("arm")), int(c.get("n")), str(g))] = (cyc, ns)
    return out


def null_clocks(page: dict) -> dict[int, float]:
    """The null kernel's `sm_clock_mhz` per cell tread, where the capture ran one."""
    return {int(c["n"]): float(c["null"]["sm_clock_mhz"]) for c in page.get("cells") or []
            if (c.get("null") or {}).get("sm_clock_mhz")}


def twin_path(path: Path) -> Path | None:
    """Where a lock capture's base-clock twin is published: a floor's
    `r3f-g<G>-lock<F>.json` beside `r3f-g<G>.json`, a byte page's
    `lock<F>/r3c-g<G>.json` beside `base/r3c-g<G>.json`."""
    path = Path(path)
    if re.search(r"-lock\d+(\.\d+)?\.json$", path.name):
        return path.with_name(re.sub(r"-lock\d+(\.\d+)?\.json$", ".json", path.name))
    if re.fullmatch(r"lock\d+(\.\d+)?", path.parent.name):
        return path.parent.parent / "base" / path.name
    return None


def load_twin(path: Path | None, page: dict) -> dict | None:
    """The base twin's page when it is published, is the same kind, ran at
    ncu's base clock with no lock, and records cycles; else None."""
    t = twin_path(path) if path else None
    if t is None or not t.exists():
        return None
    try:
        twin = json.loads(t.read_text())
    except (OSError, ValueError):
        return None
    if page_kind(twin) != page_kind(page) or page_lock(twin) is not None:
        return None
    control = ((twin.get("clock") or {}).get("control") if page_kind(twin) == "floor"
               else (twin.get("ncu") or {}).get("clock_control"))
    if control != "base":
        return None
    return twin


# --------------------------------------------------------------------------
# The check.
# --------------------------------------------------------------------------

def offset_shares(points: dict[Key, tuple[float | None, float | None]]) -> dict[Key, float | None]:
    """Each cell's offset share (t - n b) / t, b the median per-tread slope of
    its (arm, GEMM)'s consecutive treads; None where no slope exists."""
    groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for (arm, n, g), (_c, t) in points.items():
        if t:
            groups.setdefault((arm, g), []).append((n, float(t)))
    out: dict[Key, float | None] = {k: None for k in points}
    for (arm, g), cs in groups.items():
        cs.sort()
        slopes = [(t2 - t1) / (n2 - n1) for (n1, t1), (n2, t2) in zip(cs, cs[1:], strict=False)
                  if n2 > n1]
        if not slopes:
            continue
        b = statistics.median(slopes)
        for n, t in cs:
            out[(arm, n, g)] = (t - n * b) / t
    return out


def readable(points: dict[Key, tuple[float | None, float | None]]) -> tuple[set[Key], list[str]]:
    """The cells long enough to read, and a label per cell left out."""
    shares = offset_shares(points)
    keep, short = set(), []
    for k in sorted(points):
        s = shares.get(k)
        cyc, t = points[k]
        if not (cyc and t):
            short.append(f"{k[0]}/{k[1]} {k[2]} no clock")
        elif s is None:
            short.append(f"{k[0]}/{k[1]} {k[2]} no slope")
        elif s > SHORT_SHARE_MAX:
            short.append(f"{k[0]}/{k[1]} {k[2]} share {s:.2f}")
        else:
            keep.add(k)
    return keep, short


@dataclass
class Reading:
    name: str                 # smi | null | ratio level | ratio spread | null spread
    level: bool               # a level reading (says the lock was in force)
    ok: bool
    text: str


@dataclass
class LockCheck:
    verdict: str
    lock: float
    readings: list[Reading] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    fit: list[str] = field(default_factory=list)
    fit_off: list[str] = field(default_factory=list)
    fallback: str = ""

    @property
    def off(self) -> list[str]:
        return [r.text for r in self.readings if not r.ok]

    def measured(self) -> str:
        head = ("lock in force" if self.verdict == PASS else
                "off the lock" if self.verdict == FAIL else "no independent clock reading")
        parts = [f"{r.name} {r.text}{'' if r.ok else ' OUT'}" for r in self.readings]
        tail = []
        if self.fallback:
            tail.append(self.fallback)
        if self.excluded:
            tail.append(f"too short to read, left out: {', '.join(self.excluded[:6])}"
                        + (f" and {len(self.excluded) - 6} more" if len(self.excluded) > 6 else ""))
        old = ("FAIL (" + "; ".join(self.fit_off[:3])
               + (f"; and {len(self.fit_off) - 3} more" if len(self.fit_off) > 3 else "")
               + ")") if self.fit_off else "PASS"
        tail.append(f"old fit, diagnostic only: {'; '.join(self.fit)} -> would read {old}")
        return f"{head}: " + "; ".join(parts) + " | " + " | ".join(tail)

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "lock_mhz": self.lock,
                "readings": [{"name": r.name, "level": r.level, "ok": r.ok, "text": r.text}
                             for r in self.readings],
                "excluded": self.excluded, "fallback": self.fallback,
                "fit": self.fit, "fit_off": self.fit_off}


THRESHOLD = (f"nvidia-smi clocks.sm before and after within {SMI_TOL_MHZ:g} MHz of the lock; "
             f"null-kernel median within {100 * NULL_LEVEL_TOL:g}% and every cell's within "
             f"{100 * NULL_SPREAD_TOL:g}% of that median; base-twin ratio median within "
             f"{100 * RATIO_LEVEL_TOL:g}% of the held reference and every readable cell's within "
             f"{100 * RATIO_SPREAD_TOL:g}% of the capture's median; cells whose offset share "
             f"exceeds {SHORT_SHARE_MAX:g} left out; tolerances 2 x the published held-lock "
             "maxima (scripts/lock_gate.py); the old t0 + cycles / f fit printed, not gated")


def check_lock(lock: float, *, points: dict[Key, tuple[float | None, float | None]],
               smi_before: dict | None = None, smi_after: dict | None = None,
               nulls: dict[int, float] | None = None,
               twin_points: dict[Key, tuple[float | None, float | None]] | None = None,
               label=None) -> LockCheck:
    """The lock check over one capture's readings (module docstring)."""
    lock = float(lock)
    chk = LockCheck(verdict=REFUSE, lock=lock)
    label = label or (lambda k: f"{k[0]}/{k[1]} {k[2]}")
    # the old fit, per GEMM, printed only
    for g in sorted({k[2] for k in points}):
        pts = [(label(k), *points[k]) for k in sorted(points) if k[2] == g]
        o, fit = lock_fit(pts, g, lock)
        chk.fit_off += o
        chk.fit.append(fit)
    # smi bracket
    before, after = smi_clocks(smi_before), smi_clocks(smi_after)
    if before or after:
        vals = before + after
        ok = all(abs(v - lock) <= SMI_TOL_MHZ for v in vals)
        chk.readings.append(Reading("smi", True, ok,
                                    f"before {'/'.join(f'{v:g}' for v in before) or 'none'}, "
                                    f"after {'/'.join(f'{v:g}' for v in after) or 'none'} MHz"))
    # null kernel
    if nulls:
        med = statistics.median(nulls.values())
        dev = med / lock - 1
        chk.readings.append(Reading(
            "null", True, abs(dev) <= NULL_LEVEL_TOL,
            f"median {med:.1f} MHz ({100 * dev:+.2f}%) over {len(nulls)} cells"))
        if len(nulls) >= 2:
            worst = max(nulls, key=lambda n: abs(nulls[n] / med - 1))
            wd = nulls[worst] / med - 1
            chk.readings.append(Reading(
                "null spread", False, abs(wd) <= NULL_SPREAD_TOL,
                f"worst n={worst} {nulls[worst]:.1f} MHz ({100 * wd:+.2f}%)"))
    # base twin
    keep, chk.excluded = readable(points)
    if twin_points is not None:
        ratios = {}
        for k in keep:
            c, t = points[k]
            bc, bt = twin_points.get(k, (None, None))
            if c and t and bc and bt:
                ratios[k] = (c / t) / (bc / bt)
        if ratios:
            med = statistics.median(ratios.values())
            ref = RATIO_REF.get(lock)
            if ref is not None:
                dev = med / ref - 1
                chk.readings.append(Reading(
                    "ratio", True, abs(dev) <= RATIO_LEVEL_TOL,
                    f"median R {med:.4f} vs held {ref:.4f} ({100 * dev:+.2f}%, implied "
                    f"{lock * med / ref:.0f} MHz) over {len(ratios)} cells"))
            if len(ratios) >= 2:
                worst = max(ratios, key=lambda k: abs(ratios[k] / med - 1))
                wd = ratios[worst] / med - 1
                chk.readings.append(Reading("ratio spread", False, abs(wd) <= RATIO_SPREAD_TOL,
                                            f"worst {label(worst)} R {ratios[worst]:.4f} "
                                            f"({100 * wd:+.2f}% of the median)"))
    if not nulls and twin_points is None:
        chk.fallback = ("fallback: no null kernel and no base twin, judged on the nvidia-smi "
                        "bracket alone; drift within the capture not tested")
    if any(not r.ok for r in chk.readings):
        chk.verdict = FAIL
    elif any(r.level for r in chk.readings):
        chk.verdict = PASS
    return chk


def check_page(page: dict, path: Path | None = None) -> LockCheck | None:
    """The lock check of one stored page (None for a page under no lock)."""
    lock = page_lock(page)
    if lock is None:
        return None
    twin = load_twin(path, page) if path else None
    tp = page_points(twin) if twin else None
    if tp is not None and not any(c for c, _ in tp.values()):
        tp = None                   # a base byte page records no cycles
    before, after = page_smi(page)
    lab = ((lambda k: f"n={k[1]} {k[2]}") if page_kind(page) == "floor" else None)
    return check_lock(lock, points=page_points(page), smi_before=before, smi_after=after,
                      nulls=null_clocks(page), twin_points=tp, label=lab)


def stored_lock_gate(page: dict) -> tuple[str, str] | None:
    """(gate number, stored verdict) of the page's FL1 or V10, or None."""
    for g in page.get("gates") or []:
        num = g.get("number") or g.get("tag") or g.get("name")
        if num in LOCK_GATES:
            return num, g.get("verdict")
    return None


def regate_path(path: Path) -> dict | None:
    """One published file re-gated on read: its stored FL1 / V10 and the
    recomputed verdict. None for a file that is not a lock page."""
    path = Path(path)
    try:
        page = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if page_lock(page) is None:
        return None
    chk = check_page(page, path)
    stored = stored_lock_gate(page)
    return {"path": path, "gate": stored[0] if stored else ("FL1" if page_kind(page) == "floor"
                                                           else "V10"),
            "stored": stored[1] if stored else None, "new": chk.verdict, "check": chk}


def stored_vs_recomputed(page: dict, path: Path | None) -> str | None:
    """The line a reader prints for a page's lock gate: the recomputed verdict,
    the stored one labelled stale. None when the page carries no lock gate."""
    stored = stored_lock_gate(page)
    lock = page_lock(page)
    if stored is None or lock is None:
        return None
    chk = check_page(page, Path(path) if path else None)
    return f"{stored[0]} {chk.verdict} (recomputed lock check; stored {stored[1]}, stale)"


def reader_gate_entries(page: dict, path: Path | None, *, kind: str | None = None) -> list[str]:
    """The gate entries a reader prints for a stored page: every gate whose
    verdict is not PASS (only `kind` gates when given), as stored, except
    FL1 / V10, which is recomputed on read (`stored_vs_recomputed`) and listed
    whenever the recomputed or the stored verdict is not PASS."""
    out = []
    for g in page.get("gates") or []:
        num = g.get("number") or g.get("tag") or g.get("name")
        if kind is not None and g.get("kind") != kind:
            continue
        if num in LOCK_GATES:
            line = stored_vs_recomputed(page, path)
            if line and (g.get("verdict") != PASS or not line.startswith(f"{num} {PASS} ")):
                out.append(line)
            continue
        if g.get("verdict") != PASS:
            out.append(f"{num} {g.get('verdict')}")
    return out


def published_lock_pages(root: Path = PUBLISHED) -> list[Path]:
    """Every published page or floor capture taken under a lock, sorted. A
    run's summary.json repeats its pages' stored gates and is not a page."""
    out = []
    for p in sorted(Path(root).rglob("*.json")):
        if ".profiles" in str(p) or p.name == "summary.json":
            continue
        if not re.match(r"r3[cf]-", p.name):
            continue
        try:
            page = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if page_lock(page) is not None:
            out.append(p)
    return out


def regate_table(root: Path = PUBLISHED) -> list[dict]:
    return [r for r in (regate_path(p) for p in published_lock_pages(root)) if r]


def table_lines(rows: list[dict], root: Path = PUBLISHED) -> list[str]:
    out = [f"{'page':<104} {'gate':<4} {'stored':<6} {'new':<6}"]
    for r in rows:
        rel = str(Path(r["path"]).relative_to(root))
        rel = re.sub(r"/results/\d{4}-\d\d-\d\d-nvidia_gh200_480gb-", "/", rel)
        out.append(f"{rel:<104} {r['gate']:<4} {str(r['stored']):<6} {r['new']:<6}")
    return out


# --------------------------------------------------------------------------
# The statistics behind each tolerance, from the published held-lock pages.
# --------------------------------------------------------------------------

def held_by_smi(page: dict) -> bool:
    """Held for the derivation: nvidia-smi read the lock both sides (exactly)."""
    lock = page_lock(page)
    b, a = (smi_clocks(x) for x in page_smi(page))
    return lock is not None and bool(b) and bool(a) and all(v == lock for v in b + a)


def derive(root: Path = PUBLISHED) -> dict:
    """Recompute, from the published held-lock captures, the maxima each
    tolerance is set by, and the shares the short-cell rule separates."""
    null_level, null_spread, ratio_meds, ratio_spread = [], [], [], []
    short_n1, long_max = [], 0.0
    smi_reads, smi_off = 0, 0.0
    for p in published_lock_pages(root):
        page = json.loads(p.read_text())
        b, a = (smi_clocks(x) for x in page_smi(page))
        lock = page_lock(page)
        if not held_by_smi(page):
            continue
        smi_reads += len(b + a)
        smi_off = max([smi_off] + [abs(v - lock) for v in b + a])
        nulls = null_clocks(page)
        if nulls:
            med = statistics.median(nulls.values())
            null_level.append(abs(med / lock - 1))
            null_spread.append(max(abs(v / med - 1) for v in nulls.values()))
        twin = load_twin(p, page)
        if twin is None:
            continue
        pts, tp = page_points(page), page_points(twin)
        if not any(c for c, _ in tp.values()):
            continue
        shares = offset_shares(pts)
        for k, s in shares.items():
            if s is None:
                continue
            if k[1] == 1:
                short_n1.append(s)
            else:
                long_max = max(long_max, s)
        keep, _ = readable(pts)
        rs = [(pts[k][0] / pts[k][1]) / (tp[k][0] / tp[k][1]) for k in keep
              if k in tp and tp[k][0] and tp[k][1]]
        if rs:
            med = statistics.median(rs)
            ratio_meds.append((lock, med))
            ratio_spread.append(max(abs(r / med - 1) for r in rs))
    ref = statistics.median(m for lk, m in ratio_meds if lk == 1710.0)
    ratio_level = [abs(m / ref - 1) for lk, m in ratio_meds if lk == 1710.0]
    return {"smi_readings": smi_reads, "smi_max_off_mhz": smi_off,
            "null_level_max": max(null_level), "null_spread_max": max(null_spread),
            "ratio_ref_1710": ref, "ratio_level_max": max(ratio_level),
            "ratio_spread_max": max(ratio_spread), "ratio_captures": len(ratio_meds),
            "null_captures": len(null_level),
            "share_n1_min": min(short_n1), "share_n1_max": max(short_n1),
            "share_long_max": long_max}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=str(PUBLISHED))
    ap.add_argument("--derive", action="store_true",
                    help="print the held-lock statistics each tolerance is set from")
    ap.add_argument("--detail", action="store_true", help="print each page's readings")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    if args.derive:
        d = derive(root)
        for k, v in d.items():
            print(f"{k:<20} {v:.5f}" if isinstance(v, float) else f"{k:<20} {v}")
        print(f"tolerances: null level {tolerance_from(d['null_level_max']):.3f}, null spread "
              f"{tolerance_from(d['null_spread_max']):.3f}, ratio level "
              f"{tolerance_from(d['ratio_level_max']):.3f}, ratio spread "
              f"{tolerance_from(d['ratio_spread_max']):.3f}")
        return 0
    rows = regate_table(root)
    for line in table_lines(rows, root):
        print(line)
    if args.detail:
        for r in rows:
            print(f"{Path(r['path']).relative_to(root)}\n    {r['check'].measured()}")
    changed = sum(1 for r in rows if r["stored"] != r["new"])
    fails = [r for r in rows if r["new"] != PASS]
    print(f"{len(rows)} lock pages; {changed} verdicts change; new verdict not PASS on "
          f"{len(fails)}: " + (", ".join(str(Path(r['path']).relative_to(root)) for r in fails)
                                or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
