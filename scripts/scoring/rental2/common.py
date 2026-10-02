"""Rental 2's shared law and band arithmetic (docs/registered/2026-10-01-rental2-*).

One module for the registration generator (register.py) and the scorers, so a
band is computed by one rule on both sides. Pure functions; nothing here reads
a page. Every constant a band uses is passed in from the registered JSON.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

REG_DIR = REPO / "docs" / "registered"
DATE = "2026-10-01"
NAMES = {"knobs": f"{DATE}-rental2-knobs-gh200", "launch2": f"{DATE}-rental2-launch2-gh200",
         "const": f"{DATE}-rental2-const-gh200", "w1floor": f"{DATE}-rental2-w1floor-gh200"}
CARD = "nvidia_gh200_480gb"


def registration(repo: Path, part: str) -> dict:
    """The registered JSON of one part, read from `repo`'s docs/registered."""
    return json.loads((Path(repo) / "docs" / "registered" / f"{NAMES[part]}.json").read_text())


# --------------------------------------------------------------------------
# part 1: the lag law L2r, its rivals and their bands
# --------------------------------------------------------------------------

def ratio(dd: float, lam: float) -> float:
    """exp(-dd / lam), 0 when lam <= 0 (an infinitely steep law)."""
    if lam <= 0:
        return 0.0 if dd > 0 else 1.0
    return math.exp(-dd / lam)


def s_cap(x: float, knots: list) -> float:
    """Piecewise-linear s_cap through the registered knots [[x, s], ...]
    (ascending x), flat outside them."""
    if x <= knots[0][0]:
        return float(knots[0][1])
    if x >= knots[-1][0]:
        return float(knots[-1][1])
    for (x0, s0), (x1, s1) in zip(knots, knots[1:], strict=False):
        if x0 <= x <= x1:
            return float(s0 + (s1 - s0) * (x - x0) / (x1 - x0))
    raise AssertionError("unreachable")


def law_value(s_base: float, cap: float, dd: float, lam: float, past_edge: bool) -> float:
    """The L2r knob prediction as a ratio to the same-board base: past the edge
    s_base r, below it s_cap + (s_base - s_cap) r."""
    r = ratio(dd, lam)
    if past_edge:
        return s_base * r
    return cap + (s_base - cap) * r


def bands(s_base: float, cap: float, dd: float, *, lam: float, lam_sigma: float,
          lam_steep: float, past_edge: bool, s_noise: float, nl_half: float) -> dict:
    """{"L2r": [lo, hi], "NL": [lo, hi], "ST": [lo, hi]} for one cell and n."""
    lo_l = max(lam - lam_sigma, 0.0)
    hi_l = lam + lam_sigma
    a = law_value(s_base, cap, dd, lo_l, past_edge)
    b = law_value(s_base, cap, dd, hi_l, past_edge)
    st_top = law_value(s_base, cap, dd, lam_steep, past_edge)
    return {"L2r": [min(a, b) - s_noise, max(a, b) + s_noise],
            "NL": [s_base - nl_half, s_base + nl_half],
            "ST": [0.0, st_top + s_noise]}


def inside(v: float, band) -> bool:
    return band[0] <= v <= band[1]


def select(meas: dict, band_by_n: dict, hyps, ns, *, need: int) -> dict:
    """The 4-of-6 rule. For each hypothesis: the n inside its band and inside no
    other's (exclusive), the n outside. SELECTED: exclusive >= need; FALSIFIED:
    outside >= need. The verdict names the one SELECTED hypothesis, else
    INCONCLUSIVE."""
    per = {}
    for h in hyps:
        excl = [n for n in ns if inside(meas[n], band_by_n[n][h])
                and not any(inside(meas[n], band_by_n[n][o]) for o in hyps if o != h)]
        out = [n for n in ns if not inside(meas[n], band_by_n[n][h])]
        per[h] = {"exclusive": excl, "outside": out,
                  "status": ("SELECTED" if len(excl) >= need else
                             "FALSIFIED" if len(out) >= need else "neither")}
    sel = [h for h in hyps if per[h]["status"] == "SELECTED"]
    return {"per_hypothesis": per, "verdict": sel[0] if len(sel) == 1 else "INCONCLUSIVE"}


def lag_union(b: dict) -> list:
    """tp4 w1 s8's LAG hypothesis: the envelope of L2r and ST (the two cannot be
    told apart there; the review, section 1)."""
    return [min(b["L2r"][0], b["ST"][0]), max(b["L2r"][1], b["ST"][1])]


# --------------------------------------------------------------------------
# part 1, T5: the BLOCK_K separator
# --------------------------------------------------------------------------

def f_of_q(q: float, n: int) -> float:
    """PRIVATE w2's A-tile excess per tread, f = (q - n) 128 / (63 n): P_w2 = 64
    N-tiles, ATILE / SLAB = BLOCK_M / BLOCK_N = 1/2 on every Mixtral shard, at
    any BLOCK_K."""
    return (q - n) * 128.0 / (63.0 * n)


def sigma_rho(rho: float, f_num: float, f_den: float, sigma_q: float, n: int = 9) -> float:
    """sigma of rho = f_num / f_den, each f carrying sigma_q x 128 / (63 n) in
    quadrature (rental 1's T2 noise model)."""
    sf = sigma_q * 128.0 / (63.0 * n)
    return rho * math.sqrt((sf / f_num) ** 2 + (sf / f_den) ** 2)


def t5_candidates(reg_t5: dict):
    """{name: (wave_split_bytes.Params, content_a)} from the registered params."""
    import wave_split_bytes as W
    return {k: (W.Params.from_json(v["params"]), bool(v["content_a"]))
            for k, v in reg_t5["candidates"].items()}


def price_f(geom, params, content_a: bool, n: int = 9, G: int = 64) -> float:
    """One candidate's f at PRIVATE w2 (G, n) on a page's own geometry (its
    block_k and its W_c, from `wave_split_bytes.page_geometry`)."""
    import wave_split_bytes as W
    q = W.shown(W.Model(geom, content_a=content_a).evaluate(params, [("private", G, n, "w2")])[0])
    return f_of_q(q, n)


# --------------------------------------------------------------------------
# part 3 and 4 helpers
# --------------------------------------------------------------------------

def u_cycles(ksteps: int, gemm: str, c: float, F: dict) -> float:
    """The per-CTA floor u = S c + F_g (cycles)."""
    return ksteps * c + F[gemm]


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


# --------------------------------------------------------------------------
# the gate addendum (docs/registered/2026-10-02-rental2-addendum-gates.json)
# --------------------------------------------------------------------------

ADDENDUM = "2026-10-02-rental2-addendum-gates"
NAMES["addendum"] = ADDENDUM

#: a verdict string that says the test had nothing to read
NO_DATA = ("NOT SCORED", "NOT RUN", "NOT TESTED", "missing", "RECORD: a page is missing")
#: the keys whose string values are verdicts (numbers under the same key are not)
VERDICT_KEYS = ("verdict", "identification", "winner", "co_primary",
                "H_EST", "FLUID", "FLUID_LOW", "H_NPN (secondary)")


def addendum(repo: Path) -> dict:
    """The addendum's JSON, read from `repo`'s docs/registered."""
    return json.loads((Path(repo) / "docs" / "registered" / f"{ADDENDUM}.json").read_text())


def failed_gates(page: dict, ignore=()) -> list[str]:
    """The VALIDITY gates a page records as anything but PASS (FAIL, REFUSE), less
    `ignore`. A page that records no gate has failed none."""
    return [str(g.get("number")) for g in page.get("gates") or []
            if g.get("kind", "VALIDITY") == "VALIDITY" and g.get("verdict") != "PASS"
            and str(g.get("number")) not in ignore]


def page_lock(page: dict) -> float | None:
    """The nvidia-smi lock a floor capture was taken under (its `clock.lock_mhz`),
    None for a base-clock capture and for a byte page (whose lock lives in its ncu
    block and whose scores read no clock)."""
    lock = (page.get("clock") or {}).get("lock_mhz")
    return None if lock is None else float(lock)


def drop_above_lock(page: dict, step: float) -> tuple[dict, list[str]]:
    """Rule 2: a copy of `page` without every (cell, GEMM) whose own measured clock
    (`sm_clock_mhz`, its counters' cycles over its duration) sits more than `step`
    above the capture's lock, and the list of what was dropped. A clock above the
    lock cannot be run; it is the fixed ncu duration offset on a short GEMM. The
    reading's known bias is downward (about -1.4%), so this can leave such a cell
    in but never drops one that ran at the lock."""
    import copy
    lock = page_lock(page)
    if lock is None:
        return page, []
    out = copy.deepcopy(page)
    dropped = []
    for c in out.get("cells") or []:
        for g in sorted(c.get("per_gemm") or {}):
            mhz = c["per_gemm"][g].get("sm_clock_mhz")
            if mhz is not None and mhz > lock + step:
                dropped.append(f"n={c.get('n')} {g} {mhz:.1f} MHz, over the {lock:g} MHz lock "
                               f"by {mhz - lock:.1f} (> {step:g})")
                del c["per_gemm"][g]
    return out, dropped


def null_clock_mhz(page: dict) -> float | None:
    """The median over a floor capture's cells of its null kernel's sm_clock_mhz
    (`cell["null"]`, written with --floor-null-kernel), None when it has none."""
    xs = [c["null"]["sm_clock_mhz"] for c in page.get("cells") or []
          if (c.get("null") or {}).get("sm_clock_mhz") is not None]
    return median(xs) if xs else None


def lock_check(page: dict, tol: float) -> dict | None:
    """The addendum's lock-in-force check for a floor capture taken under a lock:
    the null kernel's median clock within `tol` of the lock. None when the page has
    no lock; {"ok": None} when it has no null kernel (CLEAN then falls back to FL1)."""
    lock = page_lock(page)
    if lock is None:
        return None
    med = null_clock_mhz(page)
    if med is None:
        return {"ok": None, "why": "no null kernel: CLEAN falls back to FL1"}
    rel = med / lock - 1
    return {"ok": abs(rel) <= tol, "null_median_mhz": med, "rel": rel,
            "why": f"null kernel median {med:.1f} MHz, {100 * rel:+.2f}% of the {lock:g} MHz lock "
                   f"(within {100 * tol:g}%: {abs(rel) <= tol})"}


class View:
    """One of the addendum's two views of a part's pages. ALL counts every page
    (rules 1 and 2 applied); CLEAN also drops every page that failed a VALIDITY
    gate not in `ignore`, except that a floor capture under a lock is judged by the
    null kernel's lock check in place of FL1 (FL1 again where it has no null
    kernel). A page whose V1 (launch attribution) failed is unusable in both. `pages` records what happened to each page this view was shown."""

    def __init__(self, name: str, add: dict, ignore=()):
        if name not in ("ALL", "CLEAN"):
            raise ValueError(name)
        self.name, self.ignore = name, tuple(ignore)
        self.step = float(add["constants"]["step_mhz"])
        self.unusable = tuple(add["constants"]["unusable_gates"])
        self.lock_tol = float(add["constants"]["lock_check_tol"])
        self.lock_gate = str(add["constants"]["lock_gate_set_aside"])
        self.pages: dict = {}

    def admit(self, key, page: dict | None) -> dict | None:
        if page is None:
            return None
        every = failed_gates(page)
        failed = [g for g in every if g not in self.ignore]
        rec = {"failed_gates": failed, "failed_but_not_gating": [g for g in every if g in self.ignore],
               "dropped_above_lock": []}
        # a floor capture under a lock: CLEAN reads the null kernel's lock check, not FL1
        chk = lock_check(page, self.lock_tol)
        clean_failed = list(failed)
        if chk is not None:
            rec["lock_check"] = chk
            if chk["ok"] is not None:
                clean_failed = [g for g in clean_failed if g != self.lock_gate]
                if not chk["ok"]:
                    clean_failed.append("lock not in force (null kernel)")
        rec["clean_failed"] = clean_failed
        if any(g in self.unusable for g in every):
            rec["use"] = "unusable in both views (" + ", ".join(g for g in every if g in self.unusable) + " failed)"
            page = None
        elif clean_failed and self.name == "CLEAN":
            rec["use"] = "excluded"
            page = None
        else:
            page, rec["dropped_above_lock"] = drop_above_lock(page, self.step)
            rec["use"] = "counted"
        self.pages[str(key)] = rec
        return page

    def load(self, path) -> dict | None:
        """The page at `path` as this view counts it (None when missing or out)."""
        if path is None:
            return None
        path = Path(path)
        # the page's run directory, its lock sub-directory where it has one, and its name
        return self.admit("/".join(path.parts[-3:] if path.parent.name.startswith("lock")
                                   else path.parts[-2:]), json.loads(path.read_text()))


def no_data(v) -> bool:
    return v is None or (isinstance(v, str) and v.startswith(NO_DATA))


def combine_verdict(v_all, v_clean):
    """Rule 3: the common verdict when ALL and CLEAN agree, INCONCLUSIVE when they
    disagree, NOT SCORED when CLEAN has no data and ALL does (ALL's verdict printed,
    labelled as read only off gate-failed pages)."""
    if no_data(v_all) and no_data(v_clean):
        return v_all if v_all is not None else v_clean
    if no_data(v_clean):
        return f"NOT SCORED (CLEAN has no data); ALL, reading only gate-failed pages: {v_all}"
    if no_data(v_all):          # cannot happen (ALL holds every CLEAN page); kept honest
        return f"INCONCLUSIVE (ALL has no data; CLEAN {v_clean})"
    if v_all == v_clean:
        return v_all
    return f"INCONCLUSIVE (ALL {v_all}; CLEAN {v_clean})"


def _verdict_leaves(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            if k in VERDICT_KEYS and isinstance(v, str):
                yield path + (str(k),), v
            elif isinstance(v, dict):
                yield from _verdict_leaves(v, path + (str(k),))


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def combine(res_all: dict, res_clean: dict) -> tuple[dict, list[dict]]:
    """The registered tree (ALL's numbers, every verdict replaced by rule 3's) and
    the table of every verdict path with its ALL, CLEAN and registered reading."""
    import copy
    reg = copy.deepcopy(res_all)
    table = []
    seen = set()
    for path, _ in list(_verdict_leaves(res_all)) + list(_verdict_leaves(res_clean)):
        if path in seen:
            continue
        seen.add(path)
        a, c = _get(res_all, path), _get(res_clean, path)
        a = a if isinstance(a, str) else None
        c = c if isinstance(c, str) else None
        v = combine_verdict(a, c)
        parent = _get(reg, path[:-1])
        if isinstance(parent, dict) and path[-1] in parent:
            parent[path[-1]] = v
        table.append({"path": " / ".join(path), "ALL": a, "CLEAN": c, "registered": v})
    return reg, table


def two_views(fn, repo: Path, part: str, ignore=None) -> dict:
    """Run `fn(view)` once per view and combine them (the addendum's rule 3). The
    result is the registered tree, with `addendum` holding the page record, the
    verdict table and the CLEAN tree."""
    add = addendum(repo)
    if ignore is None:
        ignore = add["per_part"][part]["gates_not_gating"]
    va, vc = View("ALL", add, ignore), View("CLEAN", add, ignore)
    res_all, res_clean = fn(va), fn(vc)
    reg, table = combine(res_all, res_clean)
    pages = {}
    for k in sorted(set(va.pages) | set(vc.pages)):
        a, c = va.pages.get(k, {}), vc.pages.get(k, {})
        pages[k] = {"failed_gates": a.get("failed_gates", c.get("failed_gates")),
                    "failed_but_not_gating": a.get("failed_but_not_gating", []),
                    "dropped_above_lock": a.get("dropped_above_lock", []),
                    "lock_check": a.get("lock_check", c.get("lock_check")),
                    "clean_failed": a.get("clean_failed", c.get("clean_failed")),
                    "ALL": a.get("use"), "CLEAN": c.get("use")}
    reg["addendum"] = {"registration": ADDENDUM, "gates_not_gating": list(ignore),
                       "pages": pages, "verdicts": table, "CLEAN": res_clean}
    return reg


def addendum_lines(res: dict) -> list[str]:
    """The addendum's printed part: every page a gate or rule 2 touched, and every
    verdict whose ALL and CLEAN readings differ."""
    ad = res.get("addendum")
    if not ad:
        return []
    out = [f"  addendum {ad['registration']}: verdicts computed on ALL and CLEAN pages"
           + (f"; not gating here (as registered): {ad['gates_not_gating']}" if ad["gates_not_gating"] else "")]
    for k, p in ad["pages"].items():
        if p["failed_gates"] or p["dropped_above_lock"] or p["failed_but_not_gating"] or p.get("clean_failed"):
            out.append(f"    page {k}: failed {p['failed_gates']}"
                       + (f" (not gating {p['failed_but_not_gating']})" if p["failed_but_not_gating"] else "")
                       + f"; ALL {p['ALL']}, CLEAN {p['CLEAN']}")
            if p.get("lock_check"):
                out.append(f"      lock check: {p['lock_check']['why']}; CLEAN reads {p['clean_failed'] or 'no failure'}")
            for d in p["dropped_above_lock"]:
                out.append(f"      dropped {d}")
    differ = [r for r in ad["verdicts"] if r["ALL"] != r["CLEAN"]
              and not (no_data(r["ALL"]) and no_data(r["CLEAN"]))]
    out.append(f"    {len(differ)} of {len(ad['verdicts'])} verdicts differ between ALL and CLEAN")
    for r in differ:
        out.append(f"    {r['path']}: ALL {r['ALL']} | CLEAN {r['CLEAN']} -> {r['registered']}")
    return out
