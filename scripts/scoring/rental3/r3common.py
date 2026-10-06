"""Rental 3's shared rules (docs/registered/2026-10-05-rental3-*).

One module for the registration generator (register.py) and the six scorers, so
a rule is computed one way on both sides. Pure functions over pages and the
registered JSON; nothing here fits a model. The two-view machinery (ALL and
CLEAN, rule 3 of the 2026-10-02 addendum) is rental 2's, imported from
scripts/scoring/rental2/common.py and never copied; what rental 3 adds is its own
lock-in-force check for floor captures (the null kernel AND, at 1710 only, the L2
clock AND nvidia-smi before and after) and a view of R3 timed pages, whose gates
are keyed `tag`, not `number`.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_rental2_common():
    """rental 2's common.py under its own module name (both directories hold a
    `common.py`, so neither may shadow the other on sys.path)."""
    name = "rental2_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "rental2" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CM2 = _load_rental2_common()
CARD = CM2.CARD
DATE = "2026-10-05"
PARTS = ("e2e", "floorlaw", "zform", "stages", "flush", "replicate")
NAMES = {p: f"{DATE}-rental3-{p}-gh200" for p in PARTS}
TARGET = "qwen2-57b-a14b-tp8"
SMS = 132
#: the per-CTA floor's c and F on the CAL counters (0f77622; r3_timing_model's CTA_FIXED_KSTEPS)
C_KSTEP = 344.1
F_CTA = {"w1": 520.0, "w2": 979.0}

median = CM2.median
no_data = CM2.no_data
combine = CM2.combine
combine_verdict = CM2.combine_verdict


def registration(repo: Path, part: str) -> dict:
    return json.loads((Path(repo) / "docs" / "registered" / f"{NAMES[part]}.json").read_text())


def r6(x):
    return None if x is None else round(float(x), 6)


def u_cycles(ksteps: int, gemm: str) -> float:
    return ksteps * C_KSTEP + F_CTA[gemm]


# --------------------------------------------------------------------------
# rulers and the joint fit (part B; floor_estimator holds the arithmetic)
# --------------------------------------------------------------------------

def frac_of(q: float) -> float:
    """The last wave's idle-SM share, ceil(q) - q."""
    return math.ceil(q) - q


def corr(xs, ys) -> float | None:
    import numpy as np
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(x) < 3 or float(np.std(x)) == 0 or float(np.std(y)) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def identifiable(q, frac, *, max_corr: float, min_cells: int) -> tuple[bool, str]:
    """Whether y/u = a + b q + theta frac can be fitted on these cells: at least
    `min_cells` cells and |corr(q, frac)| <= max_corr (a frac proportional to q,
    as on qwen2-tp8 w1 and granite-1B w2, makes the design singular)."""
    if len(q) < min_cells:
        return False, f"{len(q)} cells, under {min_cells}"
    r = corr(q, frac)
    if r is None:
        return False, "frac does not vary over the cells"
    if abs(r) > max_corr:
        return False, f"frac proportional to q (corr {r:+.3f}): theta not identifiable"
    return True, f"corr(q, frac) {r:+.3f}"


# --------------------------------------------------------------------------
# floor captures: rental 3's lock-in-force check and its view
# --------------------------------------------------------------------------

def l2_clock_mhz(page: dict, min_duration_ns: float, min_cells: int = 3) -> float | None:
    """The median over a capture's (cell, GEMM) of duration >= `min_duration_ns` of
    lts__cycles_elapsed.avg over gpu__time_duration.sum, in MHz; None when fewer than
    `min_cells` reach that duration. Short cells read low through a fixed offset of
    about 0.75 us (SEEN rental 2: 1675 to 1677 MHz under 50 us, 1700.5 to 1706 at
    0.5 ms or more), so they are never read for the gate; lock_in_force then prints
    the L2 leg instead of gating it (decided 2026-10-05, before any rental-3 page)."""
    xs = []
    for c in page.get("cells") or []:
        for _g, pg in (c.get("per_gemm") or {}).items():
            d, l2 = pg.get("gpu__time_duration.sum"), pg.get("lts__cycles_elapsed.avg")
            if d and l2 and d >= min_duration_ns:
                xs.append(l2 / d * 1e3)
    return median(xs) if len(xs) >= min_cells else None


def smi_clocks(page: dict) -> tuple[float | None, float | None]:
    """nvidia-smi's clocks.sm before and after the capture (its `clock` block)."""
    clk = page.get("clock") or {}

    def read(key):
        rows = (clk.get(key) or {}).get("rows") or []
        try:
            return float(rows[0]["clocks.sm"]) if rows else None
        except (KeyError, TypeError, ValueError):
            return None
    return read("smi_before"), read("smi_after")


def lock_in_force(page: dict, rules: dict) -> dict | None:
    """The registered lock-in-force check (floorlaw registration, gates.floor):
    None for a base-clock capture. ok is False when any available leg fails, None
    when the null-kernel leg is missing (CLEAN then falls back to FL1)."""
    lock = CM2.page_lock(page)
    if lock is None:
        return None
    why, ok = [], True
    null = CM2.null_clock_mhz(page)
    if null is None:
        return {"ok": None, "why": "no null kernel: CLEAN falls back to FL1"}
    rel = null / lock - 1
    leg = abs(rel) <= rules["null_tol"]
    ok &= leg
    why.append(f"null kernel {null:.1f} MHz ({100 * rel:+.2f}% of {lock:g}: {leg})")
    l2 = l2_clock_mhz(page, rules["l2_min_duration_ns"])
    if abs(lock - rules["l2_lock_mhz"]) < 0.5 and l2 is None:
        why.append("L2 clock not read: fewer than 3 cells of duration >= 0.5 ms (printed, not "
                   "gated; the null kernel and nvidia-smi gate)")
    elif abs(lock - rules["l2_lock_mhz"]) < 0.5:
        lo, hi = rules["l2_band_mhz"]
        leg = l2 is not None and lo <= l2 <= hi
        ok &= leg
        why.append(f"L2 clock {l2 if l2 is None else round(l2, 1)} MHz in [{lo}, {hi}]: {leg}")
    else:
        why.append(f"L2 clock {l2 if l2 is None else round(l2, 1)} MHz (printed: the L2 follows a "
                   "low SM lock, so it gates only at 1710)")
    b, a = smi_clocks(page)
    tol = rules["smi_tol_mhz"]
    leg = b is not None and a is not None and abs(b - lock) <= tol and abs(a - lock) <= tol
    ok &= leg
    why.append(f"nvidia-smi before {b}, after {a} (within {tol:g} MHz of the lock: {leg})")
    return {"ok": bool(ok), "null_median_mhz": null, "l2_mhz": l2, "smi": [b, a],
            "why": "; ".join(why)}


class FloorView(CM2.View):
    """Rental 2's View with rental 3's lock-in-force check in place of the
    null-kernel-only check. Everything else (V1 unusable in both, rule 2's
    above-lock drop, the CLEAN exclusion) is rental 2's."""

    def __init__(self, name: str, add: dict, rules: dict, ignore=()):
        super().__init__(name, add, ignore)
        self.rules = rules

    def admit(self, key, page):
        if page is None:
            return None
        rules = self.rules
        saved = CM2.lock_check
        CM2.lock_check = lambda p, tol: lock_in_force(p, rules)
        try:
            return super().admit(key, page)
        finally:
            CM2.lock_check = saved


def find_floor(tree: Path, model: str, label: str, stem: str) -> Path | None:
    hits = sorted(Path(tree).rglob(f"*-{CARD}-{model}-{label}-r3-counters/{stem}.json"))
    return hits[0] if len(hits) == 1 else None


def find_bytes(tree: Path, model: str, label: str, G: int) -> Path | None:
    hits = sorted(Path(tree).rglob(f"*-{CARD}-{model}-{label}-r3-counters/lock1710/r3c-g{G}.json"))
    return hits[0] if len(hits) == 1 else None


def floor_rows(page: dict, cells_reg: list, gemm: str, f_ruler: float | None) -> list[dict]:
    """Per registered floor-bound cell of `gemm` present on the page: q, frac, u and
    the rulers (floor_estimator.rulers) at f_ruler (the lock; None: no DUR)."""
    import floor_estimator as FE
    by_n = {int(c["n"]): c for c in page.get("cells") or []}
    out = []
    for rc in cells_reg:
        if rc["gemm"] != gemm or not rc["floor_bound"] or int(rc["n"]) not in by_n:
            continue
        pg = (by_n[int(rc["n"])].get("per_gemm") or {}).get(gemm)
        if pg is None:
            continue
        r = FE.rulers(pg, f_ruler)
        r.update(n=int(rc["n"]), q=rc["q"], frac=frac_of(rc["q"]), ceil=rc["ceil"], u=rc["u"],
                 mhz=pg.get("sm_clock_mhz"))
        out.append(r)
    return out


# --------------------------------------------------------------------------
# R3 timed pages and their view
# --------------------------------------------------------------------------

def timed_failed(report: dict, ignore=()) -> list[str]:
    """The VALIDITY gates an R3 page records as anything but PASS, less `ignore`."""
    return [str(g.get("tag") or g.get("number")) for g in report.get("gates") or []
            if g.get("kind", "VALIDITY") == "VALIDITY" and g.get("verdict") != "PASS"
            and str(g.get("tag") or g.get("number")) not in ignore]


def timed_pages(root: Path | None) -> list[tuple[Path, dict]]:
    """Every R3 report.json under `root` (the unit's gaps directory)."""
    if root is None or not Path(root).exists():
        return []
    return [(p, json.loads(p.read_text()))
            for p in sorted(Path(root).rglob("private_weight_reference/*/report.json"))]


def find_gaps(tree: Path, model: str, label: str) -> Path | None:
    hits = sorted(p for p in Path(tree).rglob(f"gaps-{CARD}-{model}-{label}") if p.is_dir())
    return hits[0] if len(hits) == 1 else None


class TimedView:
    """ALL counts every R3 page of the unit except one whose V1 failed (unusable in
    both views); CLEAN also drops every page that failed a VALIDITY gate not in
    `ignore` (V5 for part E, as registered)."""

    def __init__(self, name: str, ignore=(), unusable=("V1",)):
        if name not in ("ALL", "CLEAN"):
            raise ValueError(name)
        self.name, self.ignore, self.unusable = name, tuple(ignore), tuple(unusable)
        self.pages: dict = {}

    def admit(self, key: str, report: dict) -> dict | None:
        every = timed_failed(report)
        failed = [g for g in every if g not in self.ignore]
        rec = {"failed_gates": every, "failed_but_not_gating": [g for g in every if g in self.ignore],
               "session_tag": report.get("session_tag")}
        if any(g in self.unusable for g in every):
            rec["use"] = "unusable in both views (V1 failed)"
            report = None
        elif failed and self.name == "CLEAN":
            rec["use"] = "excluded"
            report = None
        else:
            rec["use"] = "counted"
        self.pages[key] = rec
        return report


def treads_ms(report: dict) -> dict:
    """{(arm, n): ms_p50} off an R3 page's treads_table."""
    return {(str(r["arm"]), int(r["tiles"])): float(r["ms_p50"])
            for r in report.get("treads_table") or [] if r.get("ms_p50") is not None}


def group_of(report: dict) -> int:
    return int((report.get("pinned") or {})["GROUP_SIZE_M"])


def two_views(fn, *, floor_rules=None, timed_ignore=(), floor_ignore=(), repo=None) -> dict:
    """Run fn(floor_view, timed_view) for ALL and for CLEAN and combine them by
    rental 2's rule 3 (CM2.combine). The registered tree carries the page record
    of both views and the CLEAN tree, as rental 2's two_views does."""
    add = CM2.addendum(repo or REPO)
    out = {}
    views = {}
    for name in ("ALL", "CLEAN"):
        fv = FloorView(name, add, floor_rules or {}, floor_ignore) if floor_rules else None
        tv = TimedView(name, timed_ignore)
        out[name] = fn(fv, tv)
        views[name] = (fv, tv)
    reg, table = combine(out["ALL"], out["CLEAN"])
    pages = {}
    for name, (fv, tv) in views.items():
        for k, v in ((fv.pages if fv else {}) | tv.pages).items():
            pages.setdefault(k, {})[name] = v
    reg["addendum"] = {"rule": "rental 2's addendum rule 3, the rental-3 gate definitions",
                       "pages": pages, "verdicts": table, "CLEAN": out["CLEAN"]}
    return reg


def view_lines(res: dict) -> list[str]:
    ad = res.get("addendum")
    if not ad:
        return []
    out = ["  two views (ALL, CLEAN):"]
    for k, p in sorted(ad["pages"].items()):
        a, c = p.get("ALL", {}), p.get("CLEAN", {})
        if a.get("failed_gates") or a.get("dropped_above_lock") or a.get("clean_failed"):
            out.append(f"    page {k}: failed {a.get('failed_gates')}; ALL {a.get('use')}, CLEAN {c.get('use')}")
            if a.get("lock_check"):
                out.append(f"      lock check: {a['lock_check'].get('why')}")
    differ = [r for r in ad["verdicts"] if r["ALL"] != r["CLEAN"]
              and not (no_data(r["ALL"]) and no_data(r["CLEAN"]))]
    out.append(f"    {len(differ)} of {len(ad['verdicts'])} verdicts differ between ALL and CLEAN")
    for r in differ:
        out.append(f"    {r['path']}: ALL {r['ALL']} | CLEAN {r['CLEAN']} -> {r['registered']}")
    return out


def write_score(out: Path, stem: str, res: dict, text_lines: list[str]) -> str:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.score.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    text = "\n".join(text_lines) + "\n"
    (out / f"{stem}.score.txt").write_text(text)
    return text
