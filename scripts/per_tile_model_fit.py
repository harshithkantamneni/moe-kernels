#!/usr/bin/env python
"""Per-M-tile cost models, ADDITIVE against OVERLAP, fitted to measured cells.

    python scripts/per_tile_model_fit.py <session dir or run dir> [...]
    python scripts/per_tile_model_fit.py DIR --include-invalid --r3 --out fit.json

WHAT IT COMPUTES. A cell is one per-state, per-tread median: ms per call, the
under-load SM clock f, GROUP_SIZE_M G and the tread n (M-tiles per expert). R1
cells are `clock_elasticity.collapse` over that arm's own kept rows; with --r3
the R3 arms' per-tread medians (`private_weight_reference.collapse`) join them.
Two families are fitted by bounded least squares on relative residuals:

    ADDITIVE  T = T0 + n [alpha(G) W + a] / BW x Phi(f)      the paper form,
              and the variants: Phi on the non-traffic term only, a compulsory
              first read, the group-distinct traffic count D(G, n)
    OVERLAP   T = T0 + max(D(G, n) tau, n c0 Phi(f))

with Phi(f) = (f_ref / f)^e and tau = W / BW, the time to stream the routed
weight set W once; every structure is fitted with tau free and with tau at or
above W / pin rate. f_ref is the fastest fitted cell's clock (--f-ref moves it
where Phi multiplies only the non-traffic term, which rescales c0 and no rms);
where Phi multiplies tau too it stays there, so tau >= W / pin holds the rate
under the pin rate at every cell. It prints the parameters, rms and worst relative error, the
rms per G, the rms in both families at a FIXED alpha(G>1) over a grid (alpha(1)
= 1 and free), a leave-one-G-out check and, with --predict, other pages scored
with the fitted parameters. An INVISIBLE DIRECTION is one the parameters can
move along, within their bounds, that changes no fitted cell (first order, from
the Jacobian at the fit); a parameter or a bandwidth W / tau that one moves
prints "not identified", not a number. The pin rate and the read ceilings are
the ruler the session published (`calibration/measured_*.yaml` above the page)
or --ruler; W is `moe.bench.weights.routed_expert_weight_bytes` of the rows'
model and dtype. A page whose own report.json fails a VALIDITY gate is listed
and not fitted unless --include-invalid; a page with no report.json unless
--include-unscored; a gate spelled outside `moe.bench.exit_codes`' table
refuses the run.

WHAT IT CANNOT DISTINGUISH. A hard max from a smooth transition near the kink.
What sets the floor c0: issue rate, L2-to-SM delivery, latency and occupancy all
scale with the SM clock. alpha(G) wherever the traffic branch sits under the
floor at every fitted cell (the scan prints that window). tau from alpha(1),
which trade against each other. Bytes from rate, since no counter is read. The
parameters are fitted, not measured, and the traffic counts D are assumptions.
The identification check is local: a direction flat only past first order, or a
second minimum elsewhere, is not an invisible direction.
It writes nothing unless --out is given.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import clock_elasticity as CE  # noqa: E402
import private_weight_reference as PWR  # noqa: E402
import ruler_rebaseline as RB  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.bench.weights import routed_expert_weight_bytes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

# --------------------------------------------------------------------------
# Design constants. None is a calibration: each is a choice about the fit.
# --------------------------------------------------------------------------

#: Upper bound on the clock exponent e. A fit that runs to it prints
#: "[at upper bound]" beside e.
E_MAX = 5.0

#: A scan value is inside the flat window when its sum of squared relative
#: residuals is within this fraction of the scan's minimum.
FLAT_TOL = 0.01

#: Grid step of the alpha(G>1) scan.
SCAN_STEP = 0.02

#: Starts per fit: the given start plus seeded draws inside a data-scaled box.
STARTS = 12

#: Levenberg-Marquardt iteration cap and forward-difference step.
MAX_ITER = 400
FD_STEP = 1e-7

#: Perturbed restarts from the best point, per round, and the relative sizes of
#: the kicks. The max() has kinks, and a forward-difference Jacobian taken on
#: one side of a kink can stall there: on session 5's cells one scan point sat
#: 0.19% of SSR above the minimum a 1% kick reaches.
POLISH_KICKS = (0.01, 0.01, 0.03, 0.03, 0.1, 0.1)
POLISH_ROUNDS = 3

ADDITIVE = "ADDITIVE"
OVERLAP = "OVERLAP"

VALID = "VALID"
INVALID = "INVALID"
UNSCORED = "UNSCORED"

R3_ARMS = ("native", "shared", "private")


class Refused(Exception):
    """A precondition failed; nothing was fitted. Exits REFUSED."""


# --------------------------------------------------------------------------
# The ruler and the weight set.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Ruler:
    """What this fit reads from a calibration file, and nothing else."""

    path: str
    pin_gbps: float
    #: Every bandwidth pattern the file records, GB/s, by name.
    patterns: dict
    ceiling_pattern: str

    def key(self) -> tuple:
        return (round(self.pin_gbps, 6),
                tuple(sorted((k, round(v, 6)) for k, v in self.patterns.items())))


def read_ruler(path: Path) -> Ruler:
    """The pin rate and read ceilings of one calibration yaml, or a refusal."""
    import yaml

    try:
        base = RB.read_ruler(path)
    except (OSError, ValueError) as exc:
        raise Refused(f"ruler {path}: {exc}") from exc
    doc = yaml.safe_load(path.read_text()) or {}
    observed = doc.get("observed") or {}
    pin = float(observed.get("pin_rate_gbps") or 0.0)
    if pin <= 0:
        raise Refused(f"ruler {path} records no observed.pin_rate_gbps; the tau "
                      "floor W / pin cannot be formed. Pass --ruler with one that does")
    return Ruler(path=str(path), pin_gbps=pin, patterns=dict(base.patterns),
                 ceiling_pattern=base.ceiling_pattern)


def find_ruler(page_dir: Path) -> Path | None:
    """The `calibration/measured_*.yaml` nearest above a page, the published
    session layout. None when there is none; two in one directory refuse."""
    for anc in [page_dir, *page_dir.parents]:
        found = sorted((anc / "calibration").glob("measured_*.yaml"))
        if len(found) > 1:
            raise Refused(f"{anc / 'calibration'} holds {len(found)} rulers; "
                          "pass --ruler to say which")
        if found:
            return found[0]
    return None


@dataclass(frozen=True)
class Context:
    """The constants every model evaluation needs."""

    weight_bytes: int
    experts: int
    #: W / pin rate, ms: the fastest the whole weight set can stream.
    tau_pin_ms: float
    #: Phi's reference clock where Phi multiplies only the non-traffic term.
    f_ref: float
    #: Phi's reference clock where Phi multiplies tau as well: the fastest
    #: fitted cell, so Phi >= 1 at every cell and tau >= W / pin bounds the
    #: rate at each of them. None means f_ref.
    f_top: float | None = None

    @property
    def f_traffic(self) -> float:
        return self.f_ref if self.f_top is None else self.f_top

    def bandwidth_gbps(self, tau_ms: float) -> float:
        return math.inf if tau_ms <= 0 else self.weight_bytes / (tau_ms * 1e6)


def tau_ms_at(weight_bytes: int, gbps: float) -> float:
    return 1e3 * weight_bytes / (gbps * 1e9)


# --------------------------------------------------------------------------
# Cells and pages.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    page: str
    #: "R1", or "R3 native" / "R3 shared" / "R3 private".
    arm: str
    #: The duty state (R1) or the R3 run's duty.
    state: str
    group_m: int
    tread: int
    mhz: float
    ms: float
    #: Every M-tile reads its own weight copy (the R3 private arm), so the
    #: traffic is n full sets whatever G is.
    private: bool = False


@dataclass
class Page:
    path: Path
    kind: str
    run: str
    label: str
    failed: tuple = ()
    group_m: int | None = None
    states: str = ""
    tile: tuple | None = None
    card: str = ""
    cells: list = field(default_factory=list)
    note: str = ""

    def describe(self) -> str:
        why = self.label if not self.failed else f"{self.label} ({', '.join(self.failed)} not PASS)"
        g = "?" if self.group_m is None else str(self.group_m)
        return (f"{self.kind} {self.run:<28} G={g:<3} {self.states:<22} "
                f"{why:<28} {len(self.cells):>3} cells")


_RUN_RE = re.compile(r"([0-9a-f]{8}(?:\.[^/]*)?)$")


def short_run(path: Path) -> str:
    m = _RUN_RE.search(path.name)
    return m.group(1) if m else path.name


def _header(path: Path) -> set[str]:
    with path.open(newline="") as fh:
        return set(fh.readline().strip().split(","))


R1_COLUMNS = {"duty_requested", "tiles", "ms_p50", "sm_clock_load_mhz", "group_m"}
R3_COLUMNS = {"arm", "copies", "tiles", "ms_p50", "sm_clock_load_mhz"}


def discover(root: Path) -> list[tuple[str, Path]]:
    """Every R1 and R3 page under `root` (or `root` itself), by cells.csv header."""
    if not root.exists():
        raise Refused(f"{root}: no such directory")
    csvs = [root / "cells.csv"] if (root / "cells.csv").exists() else sorted(
        p for p in root.rglob("cells.csv") if "triton-cache" not in p.parts)
    out = []
    for p in csvs:
        head = _header(p)
        if R1_COLUMNS <= head:
            out.append(("R1", p.parent))
        elif R3_COLUMNS <= head:
            out.append(("R3", p.parent))
    return out


def _label(report: dict | None, where: Path) -> tuple[str, tuple]:
    """UNSCORED with no report.json or no gates; otherwise INVALID or VALID
    by `exit_codes.classify` over every gate, with the VALIDITY gates that are
    not PASS. A gate spelled outside exit_codes' table refuses, as classify
    does: skipping it would label a page with a failing VALIDITY gate VALID."""
    if report is None:
        return UNSCORED, ()
    gates = report.get("gates") or []
    if not gates:
        return UNSCORED, ()
    try:
        if not all(isinstance(g, dict) for g in gates):
            raise exit_codes.MalformedGate("a gate is not a JSON object")
        code = exit_codes.classify((g.get("kind"), g.get("verdict")) for g in gates)
    except exit_codes.MalformedGate as exc:
        raise Refused(f"{where / 'report.json'}: {exc}; a gate outside exit_codes' "
                      "table cannot be scored, so the page cannot be labelled") from exc
    if code == exit_codes.INVALID:
        return INVALID, tuple(g.get("tag") or f"V{g.get('number', '?')}" for g in gates
                              if g["kind"] == exit_codes.VALIDITY
                              and g["verdict"] != exit_codes.PASS)
    return VALID, ()


def _report(path: Path) -> dict | None:
    f = path / "report.json"
    return json.loads(f.read_text()) if f.exists() else None


def _card(path: Path) -> str:
    f = path / "DEVICE"
    return f.read_text().strip() if f.exists() else ""


def load_r1(path: Path) -> Page:
    """One clock_elasticity page: the arm's own kept rows, its own collapse."""
    label, failed = _label(_report(path), path)
    page = Page(path=path, kind="R1", run=short_run(path), label=label,
                failed=failed, card=_card(path))
    rows = CE.read_rows(path / "cells.csv")
    kept = CE.kept_rows(rows)
    if not kept:
        page.note = "no kept rows"
        return page
    groups = {r.group_m for r in kept}
    tiles = {(r.model, r.dtype, r.block_m, r.block_n, r.block_k, r.num_warps,
              r.num_stages) for r in kept}
    if len(groups) != 1 or len(tiles) != 1:
        raise Refused(f"{path}: kept rows carry {len(groups)} GROUP_SIZE_M and "
                      f"{len(tiles)} tiles; one page is one kernel")
    page.group_m = groups.pop()
    page.tile = tiles.pop()
    duties = sorted({CE._duty_key(r.duty_requested) for r in kept}, key=float, reverse=True)
    page.states = "duty " + "/".join(f"{float(d):g}" for d in duties)
    for (tread, duty), (ms, mhz, _count) in sorted(CE.collapse(rows).items()):
        page.cells.append(Cell(page=page.run, arm="R1", state=duty,
                               group_m=page.group_m, tread=int(tread),
                               mhz=float(mhz), ms=float(ms)))
    return page


def load_r3(path: Path, arms) -> Page:
    """One private_weight_reference page: the arms' own per-tread medians and
    the median under-load clock over the same usable cells."""
    report = _report(path)
    label, failed = _label(report, path)
    page = Page(path=path, kind="R3", run=short_run(path), label=label,
                failed=failed, card=_card(path))
    samples = PWR.read_samples(path / "cells.csv")
    if report is None or not samples:
        page.note = "no report.json" if report is None else "no cells"
        return page
    pinned = report.get("pinned") or {}
    page.group_m = int(pinned.get("GROUP_SIZE_M"))
    page.tile = (report.get("model"), report.get("dtype"), int(report.get("block_m")),
                 int(pinned.get("BLOCK_SIZE_N")), int(pinned.get("BLOCK_SIZE_K")),
                 int(pinned.get("num_warps")), int(pinned.get("num_stages")))
    duty = report.get("duty")
    page.states = f"{'/'.join(arms)} @{duty}"
    for arm in arms:
        points, _spread, _dropped = PWR.collapse(samples, arm)
        for tread, ms in points:
            clocks = [s.sm_clock_load_mhz for s in samples
                      if s.arm == arm and s.tiles == tread and s.usable
                      and s.sm_clock_load_mhz]
            if not clocks:
                continue
            page.cells.append(Cell(page=page.run, arm=f"R3 {arm}", state=str(duty),
                                   group_m=page.group_m, tread=int(tread),
                                   mhz=float(statistics.median(clocks)), ms=float(ms),
                                   private=(arm == "private")))
    return page


def load_pages(roots, *, r3: bool, arms) -> list[Page]:
    pages, seen = [], set()
    for root in roots:
        for kind, path in discover(Path(root)):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            if kind == "R1":
                pages.append(load_r1(path))
            elif r3:
                pages.append(load_r3(path, arms))
    return pages


def admitted(page: Page, *, include_invalid: bool, include_unscored: bool) -> bool:
    if not page.cells:
        return False
    if page.label == INVALID:
        return include_invalid
    if page.label == UNSCORED:
        return include_unscored
    return True


# --------------------------------------------------------------------------
# The traffic counts and the models.
# --------------------------------------------------------------------------

def group_traffic(group_m: int, tread: int, experts: int) -> float:
    """D(G, n): full weight sets streamed when every GROUP_SIZE_M group reads
    each distinct expert in it once and nothing is shared across groups.

    The M-tiles run expert-major, `tread` per expert, and the kernel walks them
    in groups of `group_m` consecutive M-tiles. G = 1 gives n; one group that
    holds every expert gives 1.
    """
    tiles = [e for e in range(experts) for _ in range(tread)]
    total = sum(len(set(tiles[i:i + group_m])) for i in range(0, len(tiles), group_m))
    return total / experts


@dataclass(frozen=True)
class Structure:
    """One model structure. `combine` sum is ADDITIVE, max is OVERLAP."""

    key: str
    combine: str             # "sum" | "max"
    phi: str                 # "all" | "non_traffic"
    traffic: str             # "per_tile" | "first_read" | "group"
    alpha_one: str = "one"   # "one" (alpha(1) = 1) | "free"
    alpha_rest: str = "per_g"  # "per_g" | "tied"
    tau: str = "free"        # "free" | "pin"

    @property
    def family(self) -> str:
        return OVERLAP if self.combine == "max" else ADDITIVE

    @property
    def has_alpha(self) -> bool:
        return self.traffic in ("per_tile", "first_read")

    def formula(self) -> str:
        d = {"per_tile": "n alpha(G)", "first_read": "(1 + alpha(G)(n-1))",
             "group": "D(G,n)"}[self.traffic]
        if self.combine == "max":
            body = f"T0 + max({d} tau, n c0 Phi)"
        elif self.phi == "all":
            body = f"T0 + ({d} tau + n c0) Phi"
        else:
            body = f"T0 + {d} tau + n c0 Phi"
        alphas = ""
        if self.has_alpha:
            one = "alpha(1) = 1" if self.alpha_one == "one" else "alpha(1) free"
            rest = "per G" if self.alpha_rest == "per_g" else "tied"
            alphas = f"; {one}, alpha(G>1) {rest}"
        floor = "tau >= W/pin" if self.tau == "pin" else "tau free"
        return f"{body}{alphas}; {floor}"


def _pair(key, **kw) -> tuple[Structure, Structure]:
    return (Structure(key, **kw), Structure(key + ".pin", tau="pin", **kw))


#: The table, in print order. `paper` is Phi on everything; `split` is Phi on
#: the non-traffic term only; `first` adds the compulsory first read with
#: alpha(1) free; `group` is the group-distinct count with no alpha.
STRUCTURES = (
    *_pair("add.paper", combine="sum", phi="all", traffic="per_tile"),
    *_pair("add.split", combine="sum", phi="non_traffic", traffic="per_tile"),
    *_pair("add.first", combine="sum", phi="non_traffic", traffic="first_read",
           alpha_one="free"),
    *_pair("add.group", combine="sum", phi="non_traffic", traffic="group"),
    *_pair("ovl.group", combine="max", phi="non_traffic", traffic="group"),
    *_pair("ovl.first", combine="max", phi="non_traffic", traffic="first_read"),
    *_pair("ovl.first.a1", combine="max", phi="non_traffic", traffic="first_read",
           alpha_one="free"),
)

#: The scan's structures: the same traffic count and Phi placement in both
#: families, so the only difference between a pair of columns is sum or max.
SCAN_STRUCTURES = (
    Structure("ovl a1=1", combine="max", phi="non_traffic", traffic="first_read",
              alpha_one="one", alpha_rest="tied", tau="pin"),
    Structure("ovl a1 free", combine="max", phi="non_traffic", traffic="first_read",
              alpha_one="free", alpha_rest="tied", tau="pin"),
    Structure("add a1=1", combine="sum", phi="non_traffic", traffic="first_read",
              alpha_one="one", alpha_rest="tied", tau="pin"),
    Structure("add a1 free", combine="sum", phi="non_traffic", traffic="first_read",
              alpha_one="free", alpha_rest="tied", tau="pin"),
)


@dataclass(frozen=True)
class Data:
    n: np.ndarray
    g: np.ndarray
    f: np.ndarray
    ms: np.ndarray
    private: np.ndarray
    dgroup: np.ndarray
    groups: tuple

    @classmethod
    def of(cls, cells, ctx: Context) -> Data:
        n = np.array([c.tread for c in cells], dtype=float)
        g = np.array([c.group_m for c in cells], dtype=int)
        return cls(n=n, g=g,
                   f=np.array([c.mhz for c in cells], dtype=float),
                   ms=np.array([c.ms for c in cells], dtype=float),
                   private=np.array([c.private for c in cells], dtype=bool),
                   dgroup=np.array([group_traffic(int(gg), int(nn), ctx.experts)
                                    for gg, nn in zip(g, n, strict=True)]),
                   groups=tuple(sorted({int(x) for x in g})))


@dataclass(frozen=True)
class Layout:
    """Parameter names, bounds and how alpha maps onto cells."""

    names: tuple
    lb: np.ndarray
    ub: np.ndarray
    alpha_slots: tuple  # per name index: None, or ("one",) / ("g", G) / ("rest",)
    fixed_rest: float | None


def layout(s: Structure, data: Data, ctx: Context, fixed_rest=None) -> Layout:
    lo_ms, hi_ms = float(data.ms.min()), float(data.ms.max())
    names = ["T0", "tau", "c0", "e"]
    lb = [0.0, ctx.tau_pin_ms if s.tau == "pin" else 0.0, 0.0, 0.0]
    ub = [lo_ms, max(hi_ms, ctx.tau_pin_ms * 1.5), hi_ms, E_MAX]
    slots = [None, None, None, None]
    if s.has_alpha:
        rest = [g for g in data.groups if g != 1]
        if s.alpha_one == "free" and 1 in data.groups:
            names.append("alpha(1)")
            slots.append(("one",))
        if fixed_rest is None and rest:
            if s.alpha_rest == "per_g":
                for g in rest:
                    names.append(f"alpha({g})")
                    slots.append(("g", g))
            else:
                names.append("alpha(G>1)")
                slots.append(("rest",))
        lb += [0.0] * (len(names) - len(lb))
        ub += [1.0] * (len(names) - len(ub))
    return Layout(tuple(names), np.array(lb), np.array(ub), tuple(slots), fixed_rest)


def alpha_per_cell(lay: Layout, p: np.ndarray, data: Data) -> np.ndarray:
    alpha = np.ones_like(data.n)
    if lay.fixed_rest is not None:
        alpha[data.g != 1] = lay.fixed_rest
    for i, slot in enumerate(lay.alpha_slots):
        if slot is None:
            continue
        if slot[0] == "one":
            alpha[data.g == 1] = p[i]
        elif slot[0] == "g":
            alpha[data.g == slot[1]] = p[i]
        else:
            alpha[data.g != 1] = p[i]
    return alpha


def predict(s: Structure, lay: Layout, p: np.ndarray, data: Data, ctx: Context) -> np.ndarray:
    t0, tau, c0, e = p[0], p[1], p[2], p[3]
    if s.traffic == "group":
        d = data.dgroup
    else:
        alpha = alpha_per_cell(lay, p, data)
        d = alpha * data.n if s.traffic == "per_tile" else 1.0 + alpha * (data.n - 1.0)
    d = np.where(data.private, data.n, d)
    if s.phi == "all":
        return t0 + (d * tau + data.n * c0) * (ctx.f_traffic / data.f) ** e
    phi = (ctx.f_ref / data.f) ** e
    if s.combine == "max":
        return t0 + np.maximum(d * tau, data.n * c0 * phi)
    return t0 + d * tau + data.n * c0 * phi


# --------------------------------------------------------------------------
# The fitter: projected Levenberg-Marquardt, multi-start. numpy only; the
# repository carries no scipy.
# --------------------------------------------------------------------------

def _jacobian(fun, p, r, lb, ub):
    jac = np.empty((r.size, p.size))
    for i in range(p.size):
        h = FD_STEP * max(1.0, abs(p[i]))
        if p[i] + h > ub[i]:
            h = -h
        q = p.copy()
        q[i] += h
        jac[:, i] = (fun(q) - r) / h
    return jac


def bounded_lsq(fun, p0, lb, ub, *, max_iter: int = MAX_ITER):
    """Minimise |fun(p)|^2 over lb <= p <= ub. Returns (p, residual, cost)."""
    p = np.clip(np.asarray(p0, dtype=float), lb, ub)
    r = fun(p)
    cost = float(r @ r)
    lam = 1e-3
    span = np.maximum(ub - lb, 1e-12)
    for _ in range(max_iter):
        jac = _jacobian(fun, p, r, lb, ub)
        grad = jac.T @ r
        at_lb = p <= lb + 1e-12 * span
        at_ub = p >= ub - 1e-12 * span
        free = ~((at_lb & (grad > 0)) | (at_ub & (grad < 0)))
        if not free.any():
            break
        jf = jac[:, free]
        a = jf.T @ jf
        diag = np.diag(a).copy()
        diag = np.maximum(diag, 1e-12 * max(float(diag.max()), 1e-300))
        accepted = False
        while lam <= 1e12:
            try:
                step = np.linalg.solve(a + lam * np.diag(diag), -grad[free])
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            q = p.copy()
            q[free] += step
            q = np.clip(q, lb, ub)
            rq = fun(q)
            cq = float(rq @ rq)
            if cq < cost:
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            break
        gain = (cost - cq) / max(cost, 1e-300)
        moved = float(np.max(np.abs(q - p) / (np.abs(p) + 1e-8)))
        p, r, cost = q, rq, cq
        lam = max(lam / 10.0, 1e-12)
        if gain < 1e-13 or moved < 1e-12:
            break
    return p, r, cost


@dataclass
class Fit:
    structure: Structure
    names: tuple
    values: tuple
    lower: tuple
    upper: tuple
    at_lower: tuple
    at_upper: tuple
    flat: tuple
    #: Numerical rank of the column-normalised Jacobian over every parameter
    #: that can move within its bounds (a parameter on a bound counts: it can
    #: leave it), against how many there are.
    rank: int
    movable: int
    #: `invisible_directions` of the fit: rows are unit directions, in
    #: parameters divided by `scale`, that the parameters can move within
    #: their bounds and that change no fitted cell to first order.
    scale: np.ndarray
    invisible: np.ndarray
    #: Per parameter: False when some invisible direction moves it.
    identified: tuple
    rel: np.ndarray
    cells: int
    rms: float
    worst: float
    ssr: float
    #: W / tau at the fitted point. Not a measurement of anything when
    #: `bandwidth_identified` is False: tau then slides along a direction the
    #: cells cannot see, and this is where the fitter stopped on it.
    bandwidth_gbps: float
    per_g: dict

    def value(self, name: str) -> float | None:
        return self.values[self.names.index(name)] if name in self.names else None

    @property
    def blind(self) -> bool:
        """Some direction inside the bounds moves the parameters and no cell."""
        return self.invisible.shape[0] > 0

    @property
    def bandwidth_identified(self) -> bool:
        return self.identified[self.names.index("tau")]

    def is_identified(self, name: str) -> bool | None:
        return self.identified[self.names.index(name)] if name in self.names else None


#: A singular value of the column-normalised Jacobian below this fraction of
#: the largest counts as zero: forward differences carry about 1e-7 of noise.
RANK_TOL = 1e-6

#: A component of a unit invisible direction below this is SVD residue: it
#: counts as zero when deciding whether the direction leaves the bounds.
SIGN_TOL = 1e-6

#: An invisible direction changes a quantity when the quantity's first-order
#: change along it, per unit step in column-normalised parameters, exceeds
#: this fraction of the quantity's largest change along any unit step. On
#: session 5's cells (the findings' four pages, the default and
#: --include-invalid sets, --r3) and session 4's G=16 page, every parameter
#: and every held-out G fell below 1.3e-7 of it or above 2.4e-3: forward-
#: difference residue on one side, an alpha near 0 scaled along the T0-tau
#: valley on the other. This sits about 80 times above the first and 250 times
#: below the second.
IDENT_TOL = 1e-5


def _null(a: np.ndarray, dim: int, tol: float = SIGN_TOL) -> np.ndarray:
    """An orthonormal basis (columns) of {x in R^dim : a x = 0}."""
    if a.shape[0] == 0 or dim == 0:
        return np.eye(dim)
    _u, sv, vt = np.linalg.svd(a)
    rank = int(np.sum(sv > tol))
    return vt[rank:].T


def invisible_directions(jac: np.ndarray, at_lower, at_upper, movable
                         ) -> tuple[np.ndarray, np.ndarray, int]:
    """The directions the parameters can move, within their bounds, that
    change no fitted cell to first order.

    `jac` is the Jacobian of the fitted residuals; `at_lower`, `at_upper` and
    `movable` (lower < upper) are per parameter. Returns (scale, generators,
    rank). `scale` is each parameter's Jacobian column norm (1 for a column
    that is exactly zero), the unit the generators are in. A parameter on a
    bound may move off it and not past it, so the directions form a cone:
    the generators (rows, unit length) are a basis of the directions usable
    with either sign, then the cone's extreme rays; an empty array means the
    fit is determined. `rank` is the rank of the column-normalised Jacobian
    over the movable parameters.
    """
    at_lower, at_upper, movable = (np.asarray(x, dtype=bool)
                                   for x in (at_lower, at_upper, movable))
    k = jac.shape[1]
    norms = np.linalg.norm(jac, axis=0)
    top = float(norms[movable].max()) if movable.any() else 0.0
    live = movable & (norms > 1e-9 * max(top, 1e-300))
    scale = np.where(live, norms, 1.0)
    basis, rank = [], 0
    idx = np.flatnonzero(live)
    if idx.size:
        # Every right-singular vector, also with fewer cells than parameters.
        _u, sv, vt = np.linalg.svd(jac[:, idx] / norms[idx],
                                   full_matrices=jac.shape[0] < idx.size)
        rank = int(np.sum(sv > RANK_TOL * sv[0]))
        for row in vt[rank:]:
            v = np.zeros(k)
            v[idx] = row
            basis.append(v)
    for i in np.flatnonzero(movable & ~live):
        v = np.zeros(k)
        v[i] = 1.0
        basis.append(v)
    if not basis:
        return scale, np.zeros((0, k)), rank
    null = np.array(basis).T
    dim = null.shape[1]
    bounded = np.flatnonzero(movable & (at_lower | at_upper))
    # A generator c of the null space is feasible when every bounded
    # parameter moves off its bound or not at all: sign_i (null c)_i >= 0.
    cons = np.where(at_lower[bounded], 1.0, -1.0)[:, None] * null[bounded]
    cons[np.abs(cons) < SIGN_TOL] = 0.0
    both = _null(cons, dim)
    gens = [null @ both[:, j] for j in range(both.shape[1])]
    rest = _null(both.T, dim) if both.shape[1] else np.eye(dim)
    reduced = cons @ rest
    free = rest.shape[1]
    if free:
        # A pointed cone's extreme rays each lie on free - 1 independent
        # active constraints.
        for rows in itertools.combinations(range(len(bounded)), free - 1):
            w = _null(reduced[list(rows)], free)
            if w.shape[1] != 1:
                continue
            for sgn in (1.0, -1.0):
                if np.all(reduced @ (sgn * w[:, 0]) >= -SIGN_TOL):
                    gens.append(null @ (rest @ (sgn * w[:, 0])))
    return scale, (np.array(gens) if gens else np.zeros((0, k))), rank


def moved_by(grads: np.ndarray, scale: np.ndarray, gens: np.ndarray) -> bool:
    """Whether any generator changes the quantities whose gradients (rows, in
    parameter units) are `grads`, by more than IDENT_TOL of their largest
    change along any unit step."""
    if gens.shape[0] == 0:
        return False
    g = np.atleast_2d(np.asarray(grads, dtype=float)) / scale
    ref = float(np.linalg.norm(g, 2))
    if ref == 0.0:
        return False
    return bool(np.max(np.linalg.norm(g @ gens.T, axis=0)) > IDENT_TOL * ref)


def _start(lay: Layout, data: Data, ctx: Context, rng) -> np.ndarray:
    """A start inside a box scaled by the data: the median per-tread slope b
    sets tau and c0, the smallest call sets T0."""
    slopes = []
    for g in data.groups:
        sel = data.g == g
        if len(set(data.n[sel])) >= 2:
            slopes.append(float(np.polyfit(data.n[sel], data.ms[sel], 1)[0]))
    b = statistics.median(slopes) if slopes else float(data.ms.mean())
    b = max(b, 1e-6)
    lo_ms = float(data.ms.min())
    if rng is None:
        p = [0.1 * lo_ms, max(b, lay.lb[1]), 0.8 * b, 1.0] + [0.5] * (len(lay.names) - 4)
    else:
        p = [rng.uniform(0, 0.3 * lo_ms), rng.uniform(max(0.3 * b, lay.lb[1]),
                                                      max(1.5 * b, lay.lb[1] * 1.01)),
             rng.uniform(0, 1.5 * b), rng.uniform(0, 2.0)]
        p += list(rng.uniform(0, 1, len(lay.names) - 4))
    return np.clip(np.array(p, dtype=float), lay.lb, lay.ub)


def polish(fun, p, r, cost, lay: Layout):
    """Kick the best point by `POLISH_KICKS` (relative, seeded) and rerun from
    each kick; keep any lower cost and go again, up to `POLISH_ROUNDS` rounds."""
    rng = np.random.default_rng(0)
    for _ in range(POLISH_ROUNDS):
        improved = False
        for kick in POLISH_KICKS:
            q = p + kick * np.maximum(np.abs(p), 1e-3) * rng.standard_normal(p.size)
            p2, r2, c2 = bounded_lsq(fun, np.clip(q, lay.lb, lay.ub), lay.lb, lay.ub)
            if c2 < cost * (1 - 1e-12):
                p, r, cost, improved = p2, r2, c2, True
        if not improved:
            break
    return p, r, cost


def transfer(source: Fit, lay: Layout) -> np.ndarray | None:
    """`source`'s solution as a point of `lay`, when it is one: every name
    `lay` fits is in `source` (alpha(1) may be absent, meaning 1), every value
    sits inside `lay`'s bounds, and nothing `source` fitted is lost except an
    alpha(1) of exactly 1. None otherwise."""
    have = dict(zip(source.names, source.values, strict=True))
    if "alpha(1)" in have and "alpha(1)" not in lay.names and have["alpha(1)"] != 1.0:
        return None
    vec = []
    for name in lay.names:
        if name in have:
            vec.append(have[name])
        elif name == "alpha(1)":
            vec.append(1.0)
        else:
            return None
    if any(n not in lay.names and n != "alpha(1)" for n in have):
        return None
    v = np.array(vec)
    tol = 1e-12 * np.maximum(np.abs(lay.ub - lay.lb), 1.0)
    if np.any(v < lay.lb - tol) or np.any(v > lay.ub + tol):
        return None
    return np.clip(v, lay.lb, lay.ub)


def fit(s: Structure, cells, ctx: Context, *, fixed_rest=None, starts: int = STARTS,
        warm=()) -> Fit:
    """Bounded least squares of `s` on `cells`, relative residuals, from the
    data-scaled start, every `warm` point (a Fit or a vector in this layout)
    and `starts - 1` seeded draws; the lowest cost wins, and its Jacobian
    gives the rank and the invisible directions (`invisible_directions`)."""
    data = Data.of(cells, ctx)
    lay = layout(s, data, ctx, fixed_rest)

    def fun(p):
        return (predict(s, lay, p, data, ctx) - data.ms) / data.ms

    candidates = [_start(lay, data, ctx, None)]
    for w in warm:
        v = transfer(w, lay) if isinstance(w, Fit) else np.asarray(w, dtype=float)
        if v is not None and len(v) == len(lay.names):
            candidates.append(np.clip(v, lay.lb, lay.ub))
    candidates += [_start(lay, data, ctx, np.random.default_rng(k)) for k in range(1, starts)]
    best = None
    for p0 in candidates:
        p, r, cost = bounded_lsq(fun, p0, lay.lb, lay.ub)
        if best is None or cost < best[2] - 1e-15:
            best = (p, r, cost)
    p, r, cost = polish(fun, *best, lay)
    jac = _jacobian(fun, p, r, lay.lb, lay.ub)
    norms = np.linalg.norm(jac, axis=0)
    flat = tuple(bool(x <= 1e-9 * max(float(norms.max()), 1e-300)) for x in norms)
    span = np.maximum(lay.ub - lay.lb, 1e-12)
    at_lower = p <= lay.lb + 1e-7 * span
    at_upper = p >= lay.ub - 1e-7 * span
    movable = lay.lb < lay.ub
    scale, gens, rank = invisible_directions(jac, at_lower, at_upper, movable)
    identified = tuple(not moved_by(np.eye(len(p))[i], scale, gens) for i in range(len(p)))
    per_g = {int(g): float(np.sqrt(np.mean(r[data.g == g] ** 2))) for g in data.groups}
    return Fit(structure=s, names=lay.names, values=tuple(float(x) for x in p),
               lower=tuple(lay.lb), upper=tuple(lay.ub),
               at_lower=tuple(bool(x) for x in at_lower),
               at_upper=tuple(bool(x) for x in at_upper),
               flat=flat, rank=rank, movable=int(movable.sum()), scale=scale,
               invisible=gens, identified=identified, rel=r, cells=int(r.size),
               rms=float(np.sqrt(np.mean(r ** 2))), worst=float(np.max(np.abs(r))),
               ssr=float(cost), bandwidth_gbps=ctx.bandwidth_gbps(float(p[1])),
               per_g=per_g)


def score(f: Fit, cells, ctx: Context) -> np.ndarray:
    """Relative residuals of a fitted model on other cells, never refitted.
    A G the fit never saw takes the fitted alpha(G>1) when the structure ties
    it across G; otherwise it has no alpha and every residual is NaN."""
    data = Data.of(cells, ctx)
    s = f.structure
    names = list(f.names)
    p = np.array(f.values)
    if s.has_alpha:
        extra = []
        for g in data.groups:
            if g == 1 and s.alpha_one == "free" and "alpha(1)" not in names:
                return np.full(data.n.shape, np.nan)
            if g != 1 and s.alpha_rest == "per_g" and f"alpha({g})" not in names:
                extra.append(g)
        if extra:
            return np.full(data.n.shape, np.nan)
    lay = layout(s, data, ctx)
    q = np.array([p[names.index(n)] if n in names else np.nan for n in lay.names])
    if np.isnan(q).any():
        return np.full(data.n.shape, np.nan)
    return (predict(s, lay, q, data, ctx) - data.ms) / data.ms


# --------------------------------------------------------------------------
# The scan and the leave-one-G-out check.
# --------------------------------------------------------------------------

def scan_grid(step: float) -> list[float]:
    count = int(round(1.0 / step))
    grid = [round(i * step, 6) for i in range(count + 1)]
    return sorted(set(grid + [1.0]))


def registered_alphas() -> dict[str, float]:
    """The refit alpha and its band, the values C1 reads, marked in the scan."""
    lo, hi = CE.SWEEP.ALPHA_BAND
    return {"refit": float(CE.SWEEP.ALPHA), "band lo": float(lo), "band hi": float(hi)}


@dataclass
class ScanColumn:
    structure: Structure
    #: (alpha, rms, ssr, alpha(1) or None, registered: bool, alpha(1)
    #: identified: bool or None). An unidentified alpha(1) is where the fitter
    #: stopped on a direction the cells cannot see.
    points: list
    window: tuple | None
    contiguous: bool
    ssr_min: float
    fits: dict = field(default_factory=dict)


def scan(s: Structure, cells, ctx: Context, grid, extras, tol: float,
         nested: ScanColumn | None = None) -> ScanColumn:
    """rms and SSR with alpha(G>1) HELD at each grid value (and each registered
    value in `extras`), every other parameter refitted. The FLAT WINDOW is the
    grid values whose SSR is within `tol` of the column's minimum. `nested` is
    the column this one contains (alpha(1) = 1 inside alpha(1) free): its fit at
    the same alpha is a start here, so the larger model never reports a worse
    minimum than the one it contains."""
    if len({c.group_m for c in cells if c.group_m != 1}) == 0:
        return ScanColumn(s, [], None, True, math.nan)
    alphas = sorted(set(grid) | set(extras))
    fits, prev = {}, None
    for a in alphas:
        warm = [w for w in (prev, nested.fits.get(a) if nested else None) if w is not None]
        fits[a] = prev = fit(s, cells, ctx, fixed_rest=a, starts=4, warm=warm)
    # Back down the grid, each point restarted from its upper neighbour: a
    # solution family that only opens above some alpha reaches below it too.
    for lower, upper in zip(alphas[-2::-1], alphas[::-1], strict=False):
        again = fit(s, cells, ctx, fixed_rest=lower, starts=0, warm=[fits[upper]])
        if again.ssr < fits[lower].ssr:
            fits[lower] = again
    pts = [(a, fits[a].rms, fits[a].ssr, fits[a].value("alpha(1)"), a not in grid,
            fits[a].is_identified("alpha(1)")) for a in alphas]
    on_grid = [p for p in pts if not p[4]]
    ssr_min = min(p[2] for p in on_grid)
    inside = [p[0] for p in on_grid if p[2] <= ssr_min * (1.0 + tol)]
    idx = [i for i, p in enumerate(on_grid) if p[0] in inside]
    contiguous = idx == list(range(idx[0], idx[-1] + 1)) if idx else True
    window = (min(inside), max(inside)) if inside else None
    return ScanColumn(s, pts, window, contiguous, ssr_min, fits)


def scan_all(cells, ctx: Context, grid, extras, tol: float) -> list[ScanColumn]:
    """Every column of `SCAN_STRUCTURES`, each alpha(1)-free column seeded
    from its alpha(1) = 1 partner."""
    cols: list[ScanColumn] = []
    for s in SCAN_STRUCTURES:
        partner = next((c for c in cols if c.structure.combine == s.combine
                        and c.structure.alpha_one == "one"), None)
        cols.append(scan(s, cells, ctx, grid, extras, tol,
                         nested=partner if s.alpha_one == "free" else None))
    return cols


def fit_all(cells, ctx: Context, structures=STRUCTURES, starts: int = STARTS) -> list[Fit]:
    """Every structure, then a second pass that starts each one from every
    other solution that is a point of it (`transfer`): tau free contains
    tau >= W/pin and alpha(1) free contains alpha(1) = 1, so a containing model
    can never end above the model it contains."""
    fits = [fit(s, cells, ctx, starts=starts) for s in structures]
    out = []
    for i, s in enumerate(structures):
        others = [g for j, g in enumerate(fits) if j != i]
        again = fit(s, cells, ctx, starts=0, warm=others)
        out.append(again if again.ssr < fits[i].ssr else fits[i])
    return out


def leave_one_g_out(s: Structure, cells, ctx: Context) -> dict[int, tuple | None]:
    """Fit on every other G, score the held-out G. alpha(G>1) is TIED across G
    here, so a held-out G has an alpha to be predicted with."""
    tied = dataclasses.replace(s, alpha_rest="tied")
    groups = sorted({c.group_m for c in cells})
    out = {}
    for g in groups:
        train = [c for c in cells if c.group_m != g]
        test = [c for c in cells if c.group_m == g]
        if not train:
            out[g] = None
            continue
        f = fit(tied, train, ctx, starts=4)
        r = score(f, test, ctx)
        out[g] = None if np.isnan(r).any() else (float(np.sqrt(np.mean(r ** 2))),
                                                 float(np.max(np.abs(r))))
    return out


LOGO_KEYS = ("add.paper.pin", "add.split.pin", "add.group.pin",
             "ovl.group.pin", "ovl.first.pin")


# --------------------------------------------------------------------------
# Planted worlds, for the tests and for anyone checking the fitter by hand.
# --------------------------------------------------------------------------

def planted_cells(s: Structure, params: dict, ctx: Context, *, groups, treads, states,
                  noise: float = 0.0, seed: int = 0) -> list[Cell]:
    """Cells generated by structure `s` at `params` (by name).

    `states` is a list of clocks, one per duty state; each may be a number or a
    callable of the tread, so a state's clock can move along its treads the way
    a power-capped card's does.
    """
    rng = np.random.default_rng(seed)
    skeleton = [Cell(page=f"planted-g{g}", arm="R1", state=str(i), group_m=g, tread=n,
                     mhz=float(st(n) if callable(st) else st), ms=1.0)
                for g in groups for i, st in enumerate(states) for n in treads]
    data = Data.of(skeleton, ctx)
    lay = layout(s, data, ctx)
    p = np.array([params[name] for name in lay.names], dtype=float)
    outside = [n for n, v, lo, hi in zip(lay.names, p, lay.lb, lay.ub, strict=True)
               if not lo <= v <= hi]
    if outside:
        raise ValueError(f"planted {outside} lie outside {s.key}'s bounds, so no fit "
                         "of it could recover them")
    ms = predict(s, lay, p, data, ctx)
    if noise:
        ms = ms * (1.0 + noise * rng.standard_normal(ms.shape))
    return [dataclasses.replace(c, ms=float(m)) for c, m in zip(skeleton, ms, strict=True)]


# --------------------------------------------------------------------------
# The page.
# --------------------------------------------------------------------------

def _pct(x: float) -> str:
    return "   n/a" if x is None or not math.isfinite(x) else f"{100 * x:6.3f}%"


def _param_text(f: Fit) -> str:
    parts = []
    for i, name in enumerate(f.names):
        tags = []
        if f.at_lower[i]:
            tags.append("at lower bound")
        elif f.at_upper[i]:
            tags.append("at upper bound")
        if f.flat[i]:
            tags.append("flat")
        elif not f.identified[i]:
            tags.append("not identified")
        tag = f" [{', '.join(tags)}]" if tags else ""
        parts.append(f"{name} {f.values[i]:.4f}{tag}")
    return ", ".join(parts)


def rms_by_arm(f: Fit, cells) -> dict[str, float]:
    arms = np.array([c.arm for c in cells])
    return {a: float(np.sqrt(np.mean(f.rel[arms == a] ** 2))) for a in sorted(set(arms))}


def fit_lines(fits, ctx: Context, ruler: Ruler, groups, cells) -> list[str]:
    read = ruler.patterns.get("read_stream")
    out = ["FITS: rms and worst of (model - measured) / measured over the fitted cells",
           f"  {'family':<9}{'model':<17}{'cells':>5}  {'rms':>8} {'worst':>8}  "
           f"{'rank':>5}   {'BW = W/tau GB/s':<30}"]
    for f in fits:
        bw = f.bandwidth_gbps
        if not f.bandwidth_identified:
            rel = "     n/a  not identified: tau is on an invisible direction"
        elif math.isfinite(bw):
            rel = f"{bw:8.0f} = {bw / ruler.pin_gbps:.3f}x pin"
            if read:
                rel += f", {bw / read:.3f}x read_stream"
        else:
            rel = "     inf (tau = 0)"
        rank = f"{f.rank}/{f.movable}" + ("!" if f.blind else " ")
        out.append(f"  {f.structure.family:<9}{f.structure.key:<17}{f.cells:>5}  "
                   f"{_pct(f.rms):>8} {_pct(f.worst):>8}  {rank:>6}  {rel}")
    out += ["  rank: of the column-normalised Jacobian over every parameter that can "
            "move within its bounds (one on a bound can leave it), of how many there are",
            "  ! marks an INVISIBLE DIRECTION: the parameters can move along it, within "
            "their bounds, and no fitted cell changes. A rank short of the count with "
            "no ! is a direction that would leave the bounds",
            "", "PARAMETERS (tau, c0, T0 in ms; c0 is per M-tile at F_REF, and in "
            "add.paper tau and c0 are at F_TOP; [not identified]: moves along an "
            "invisible direction, so the value is where the fitter stopped; [flat]: "
            "zero gradient at the optimum, a direction of its own)"]
    for f in fits:
        out.append(f"  {f.structure.key:<17}{_param_text(f)}")
    out += ["", "RMS BY G",
            "  " + f"{'model':<17}" + "".join(f"{'G=' + str(g):>10}" for g in groups)]
    for f in fits:
        out.append("  " + f"{f.structure.key:<17}"
                   + "".join(f"{_pct(f.per_g.get(g)):>10}" for g in groups))
    arms = sorted({c.arm for c in cells})
    if len(arms) > 1:
        out += ["", "RMS BY ARM", "  " + f"{'model':<17}" + "".join(f"{a:>12}" for a in arms)]
        for f in fits:
            per = rms_by_arm(f, cells)
            out.append("  " + f"{f.structure.key:<17}"
                       + "".join(f"{_pct(per[a]):>12}" for a in arms))
    return out


def legend_lines() -> list[str]:
    out = ["MODELS (Phi = (F_REF/f)^e, in add.paper (F_TOP/f)^e; D(G,n) = "
           "group-distinct weight sets; a private-arm cell reads n sets)"]
    for s in STRUCTURES:
        if s.tau == "free":
            out.append(f"  {s.key:<13} {s.formula().replace('; tau free', '')}")
    out.append("  <model>.pin   the same with tau >= W/pin (BW at most the pin rate)")
    return out


def scan_lines(cols, tol: float, extras: dict) -> list[str]:
    if not cols or not cols[0].points:
        return ["SCAN: no G other than 1 among the fitted cells; nothing to scan"]
    width = 27
    out = ["SCAN: alpha(G>1) HELD at each value, every other parameter refitted; "
           "traffic (1 + alpha(n-1)) tau, tau >= W/pin",
           "  each column: rms, SSR over the column's minimum SSR, and the fitted "
           "alpha(1) where it is free (? where it is not identified)",
           "  " + f"{'alpha(G>1)':<16}" + "".join(f"{c.structure.key:>{width}}" for c in cols)]
    names = {v: k for k, v in extras.items()}
    for i, pt in enumerate(cols[0].points):
        a = pt[0]
        label = f"{a:.3f}" + (f" {names[a]}" if pt[4] and a in names else "")
        row = []
        for c in cols:
            q = c.points[i]
            ratio = q[2] / c.ssr_min if c.ssr_min > 0 else math.nan
            a1 = "" if q[3] is None else f"  a1 {q[3]:.3f}" + ("" if q[5] else "?")
            row.append(f"{_pct(q[1]).strip():>7} x{ratio:<7.2f}{a1}")
        out.append("  " + f"{label:<16}" + "".join(f"{x:>{width}}" for x in row))
    out.append(f"  FLAT WINDOW: the grid values whose SSR is within {100 * tol:g}% "
               "of the column's minimum")
    for c in cols:
        w = "none" if c.window is None else f"[{c.window[0]:.2f}, {c.window[1]:.2f}]"
        if c.window is not None and not c.contiguous:
            w += " (not contiguous)"
        out.append(f"    {c.structure.key:<14} {w}")
    return out


def logo_lines(results: dict, groups) -> list[str]:
    out = ["LEAVE ONE G OUT: fit on the other Gs (alpha(G>1) tied), score the held-out G "
           "(rms / worst)",
           "  " + f"{'model':<17}" + "".join(f"{'G=' + str(g):>20}" for g in groups)]
    for key, per in results.items():
        row = []
        for g in groups:
            v = per.get(g)
            row.append("n/a" if v is None else f"{_pct(v[0]).strip()} / {_pct(v[1]).strip()}")
        out.append("  " + f"{key:<17}" + "".join(f"{x:>20}" for x in row))
    return out


def predict_lines(fits, cells, ctx: Context) -> list[str]:
    groups = sorted({c.group_m for c in cells})
    out = [f"PREDICTED: the fits above scored on {len(cells)} --predict cells, never refitted",
           "  " + f"{'model':<17}{'rms':>9}{'worst':>9}"
           + "".join(f"{'G=' + str(g):>10}" for g in groups)]
    g_arr = np.array([c.group_m for c in cells])
    for f in fits:
        r = score(f, cells, ctx)
        if np.isnan(r).any():
            out.append(f"  {f.structure.key:<17}  n/a (a G the fit never saw has no alpha)")
            continue
        out.append(f"  {f.structure.key:<17}{_pct(float(np.sqrt(np.mean(r ** 2)))):>9}"
                   f"{_pct(float(np.max(np.abs(r)))):>9}"
                   + "".join(f"{_pct(float(np.sqrt(np.mean(r[g_arr == g] ** 2)))):>10}"
                             for g in groups))
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit ADDITIVE and OVERLAP per-M-tile cost models to R1 "
                    "(clock_elasticity) cells, and optionally R3 ladders, side by side.")
    p.add_argument("inputs", nargs="+", type=Path,
                   help="published session dirs or run dirs; every clock_elasticity "
                        "page under each is read (private_weight_reference pages with --r3)")
    p.add_argument("--ruler", type=Path, default=None,
                   help="calibration yaml for the pin rate and read ceilings; default: "
                        "the calibration/measured_*.yaml nearest above each page")
    p.add_argument("--include-invalid", action="store_true",
                   help="fit pages whose own report.json fails a VALIDITY gate "
                        "(listed and excluded by default)")
    p.add_argument("--include-unscored", action="store_true",
                   help="fit pages with no report.json (listed and excluded by default)")
    p.add_argument("--r3", action="store_true",
                   help="add the R3 per-arm ladders (private_weight_reference pages)")
    p.add_argument("--r3-arms", default="native,shared",
                   help="R3 arms to add with --r3 (default native,shared; private cells "
                        "read n weight sets at every G)")
    p.add_argument("--f-ref", type=float, default=None,
                   help="reference clock for Phi where it multiplies only the "
                        "non-traffic term, MHz; default: the fastest fitted cell. "
                        "Rescales c0 there and changes no rms; the paper form's Phi "
                        "stays at the fastest fitted cell")
    p.add_argument("--scan-step", type=float, default=SCAN_STEP,
                   help=f"alpha(G>1) grid step for the scan (default {SCAN_STEP})")
    p.add_argument("--flat-tol", type=float, default=FLAT_TOL,
                   help=f"SSR tolerance of the scan's flat window (default {FLAT_TOL})")
    p.add_argument("--no-scan", action="store_true", help="skip the scan")
    p.add_argument("--no-logo", action="store_true", help="skip leave-one-G-out")
    p.add_argument("--predict", type=Path, nargs="+", default=None,
                   help="dirs whose R1 pages, admitted by the same rules as the "
                        "inputs, are scored with the fitted parameters and never fitted")
    p.add_argument("--out", type=Path, default=None,
                   help="write the pages, cells, fits, scan, leave-one-G-out and "
                        "predictions as JSON to this file")
    return p


def _context(pages, ruler: Ruler, f_ref: float | None) -> Context:
    """W, E and tau_pin from the pages' model and the ruler; F_TOP is the
    fastest fitted cell's clock, and F_REF is it unless --f-ref says otherwise."""
    tiles = {p.tile for p in pages}
    if len(tiles) != 1:
        raise Refused(f"the fitted pages ran {len(tiles)} different tiles or models: "
                      f"{sorted(map(str, tiles))}; one fit is one kernel")
    model, dtype = next(iter(tiles))[:2]
    if model not in MODEL_CONFIGS:
        raise Refused(f"model {model!r} is not in moe.spec.MODEL_CONFIGS")
    w = routed_expert_weight_bytes(model, dtype)
    top = max(c.mhz for p in pages for c in p.cells)
    return Context(weight_bytes=w, experts=MODEL_CONFIGS[model].num_experts,
                   tau_pin_ms=tau_ms_at(w, ruler.pin_gbps),
                   f_ref=float(f_ref) if f_ref else top, f_top=top)


def _resolve_ruler(pages, explicit: Path | None) -> Ruler:
    if explicit is not None:
        return read_ruler(explicit)
    found = {}
    for p in pages:
        path = find_ruler(p.path)
        if path is None:
            raise Refused(f"no calibration/measured_*.yaml above {p.path}; pass --ruler")
        r = read_ruler(path)
        found.setdefault(r.key(), r)
    if len(found) != 1:
        raise Refused(f"the fitted pages resolve to {len(found)} rulers with different "
                      "numbers; pass --ruler to fit them against one")
    return next(iter(found.values()))


def _fit_json(f: Fit) -> dict:
    """One fit. `bandwidth_gbps` is null when tau is not identified; the
    value the fitter stopped at is then `parameters.tau.value`."""
    return {"model": f.structure.key, "family": f.structure.family,
            "formula": f.structure.formula(), "cells": f.cells, "rms": f.rms,
            "worst": f.worst, "ssr": f.ssr,
            "bandwidth_gbps": f.bandwidth_gbps if f.bandwidth_identified else None,
            "bandwidth_identified": f.bandwidth_identified,
            "rank": f.rank, "movable_parameters": f.movable,
            "invisible_directions": int(f.invisible.shape[0]),
            "parameters": {n: {"value": v, "lower": lo, "upper": hi, "at_lower": al,
                               "at_upper": au, "flat": fl, "identified": ok}
                           for n, v, lo, hi, al, au, fl, ok in zip(
                               f.names, f.values, f.lower, f.upper, f.at_lower,
                               f.at_upper, f.flat, f.identified, strict=True)},
            "rms_by_g": {str(g): v for g, v in f.per_g.items()}}


def run(args) -> tuple[list[str], dict]:
    arms = tuple(a.strip() for a in args.r3_arms.split(",") if a.strip())
    bad = [a for a in arms if a not in R3_ARMS]
    if bad:
        raise Refused(f"--r3-arms {bad}: the arms are {R3_ARMS}")
    pages = load_pages(args.inputs, r3=args.r3, arms=arms)
    if not pages:
        raise Refused(f"no clock_elasticity pages under {[str(p) for p in args.inputs]}")
    use = [p for p in pages if admitted(p, include_invalid=args.include_invalid,
                                        include_unscored=args.include_unscored)]
    lines = ["PAGES (label from each page's own report.json gates)"]
    for p in pages:
        state = "fitted" if p in use else "EXCLUDED"
        extra = f"  {p.note}" if p.note else ""
        if p not in use and p.label == INVALID:
            extra += "  (--include-invalid to fit)"
        if p not in use and p.label == UNSCORED:
            extra += "  (--include-unscored to fit)"
        lines.append(f"  {p.describe()}  {state}{extra}")
    if not use:
        raise Refused("\n".join(lines + ["no page is admitted; nothing to fit"]))
    ruler = _resolve_ruler(use, args.ruler)
    ctx = _context(use, ruler, args.f_ref)
    cells = [c for p in use for c in p.cells]
    groups = sorted({c.group_m for c in cells})
    cards = sorted({p.card for p in use if p.card})
    read = ruler.patterns.get("read_stream")
    ceiling = ruler.patterns.get(ruler.ceiling_pattern)
    lines = [f"per_tile_model_fit: {len(cells)} cells from {len(use)} pages, G in {groups}",
             "", *lines, "",
             f"RULER {ruler.path}",
             f"  pin {ruler.pin_gbps:.1f} GB/s"
             + (f", read_stream {read:.1f}" if read else "")
             + (f", {ruler.ceiling_pattern} {ceiling:.1f}" if ceiling else ""),
             f"WEIGHT SET {ctx.weight_bytes / 1e9:.4f} GB over {ctx.experts} experts; "
             f"tau at the pin rate W/pin = {ctx.tau_pin_ms:.4f} ms",
             f"F_REF {ctx.f_ref:.0f} MHz ("
             + ("--f-ref" if args.f_ref else "the fastest fitted cell") + "); "
             f"F_TOP {ctx.f_traffic:.0f} MHz (the fastest fitted cell, add.paper's Phi)"]
    if len(cards) > 1:
        lines.append(f"CARDS {len(cards)} devices pooled: {', '.join(cards)}")
    unfit = [f"{p.run} ({p.label}{': ' + ', '.join(p.failed) if p.failed else ''})"
             for p in use if p.label != VALID]
    if unfit:
        lines.append(f"FITTED PAGES THAT ARE NOT VALID: {'; '.join(unfit)}")
    lines += ["", *legend_lines(), ""]
    fits = fit_all(cells, ctx)
    lines += fit_lines(fits, ctx, ruler, groups, cells)
    doc = {"inputs": [str(p) for p in args.inputs],
           "pages": [{"path": str(p.path), "kind": p.kind, "run": p.run,
                      "label": p.label, "failed": list(p.failed), "group_m": p.group_m,
                      "states": p.states, "card": p.card, "cells": len(p.cells),
                      "fitted": p in use, "note": p.note} for p in pages],
           "ruler": {"path": ruler.path, "pin_gbps": ruler.pin_gbps,
                     "patterns": ruler.patterns, "ceiling_pattern": ruler.ceiling_pattern},
           "weight_bytes": ctx.weight_bytes, "experts": ctx.experts,
           "tau_pin_ms": ctx.tau_pin_ms, "f_ref_mhz": ctx.f_ref,
           "f_top_mhz": ctx.f_traffic,
           "cells": [dataclasses.asdict(c) for c in cells],
           "fits": [dict(_fit_json(f), rms_by_arm=rms_by_arm(f, cells)) for f in fits]}
    if not args.no_scan:
        extras = registered_alphas()
        cols = scan_all(cells, ctx, scan_grid(args.scan_step), list(extras.values()),
                        args.flat_tol)
        lines += ["", *scan_lines(cols, args.flat_tol, extras)]
        doc["scan"] = [{"model": c.structure.key, "formula": c.structure.formula(),
                        "window": c.window, "contiguous": c.contiguous,
                        "points": [{"alpha": a, "rms": r, "ssr": s, "alpha1": a1,
                                    "alpha1_identified": ok, "registered": reg}
                                   for a, r, s, a1, reg, ok in c.points]}
                       for c in cols]
    if not args.no_logo:
        if len(groups) < 2:
            lines += ["", "LEAVE ONE G OUT: one G among the fitted cells; nothing to hold out"]
            doc["leave_one_g_out"] = {}
        else:
            by_key = {s.key: s for s in STRUCTURES}
            logo = {k: leave_one_g_out(by_key[k], cells, ctx) for k in LOGO_KEYS}
            lines += ["", *logo_lines(logo, groups)]
            doc["leave_one_g_out"] = {k: {str(g): v for g, v in per.items()}
                                      for k, per in logo.items()}
    if args.predict:
        ppages = load_pages(args.predict, r3=False, arms=())
        puse = [p for p in ppages if admitted(p, include_invalid=args.include_invalid,
                                              include_unscored=args.include_unscored)]
        lines += ["", "PREDICT PAGES"] + [
            f"  {p.describe()}  {'scored' if p in puse else 'EXCLUDED'}" for p in ppages]
        if puse:
            if {p.tile for p in puse} != {p.tile for p in use}:
                raise Refused("--predict pages ran a different tile or model than the fit")
            pcells = [c for p in puse for c in p.cells]
            lines += [*predict_lines(fits, pcells, ctx)]
            rms = {}
            for f in fits:
                r = score(f, pcells, ctx)
                rms[f.structure.key] = (None if np.isnan(r).any()
                                        else float(np.sqrt(np.mean(r ** 2))))
            doc["predict"] = {"pages": [p.run for p in puse], "rms": rms}
    return lines, doc


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lines, doc = run(args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED
    print("\n".join(lines))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(f"\nwrote {args.out}")
    return exit_codes.DONE


if __name__ == "__main__":
    sys.exit(main())
