"""Rental 6's pricing: model v3 and its rivals, from the repo's flags (no prototype code).

    v2       the registered model (rental 5): M-tile w1 A keys, timing v2
    v3       MODEL M4 bytes (`wave_split_bytes_v3.ModelV3`) and timing v3
             (`r3_timing_model_v3.call_ms`: v2 plus term (b), the page's A plus V3_H_MS on
             SHARED, only where NATIVE takes block-scan). The pinned rental-5 modules
             (r3_timing_model, wave_split_bytes, skewmodel) are imported, never edited
    v3_nob   v3 without (b): M4 bytes, timing v2
    v3_noca  v3 without (a): v2 bytes, timing v3
    LT3      v3 with wave_q = 1 (no last-wave quantisation)
    v3_all   v3 with (b) on every non-NATIVE cell (the path condition dropped)
    H8       M4 bytes, (T0, c, bw, s_small, s_block, h) from the 8x7B CAL 6-parameter refit
             (A priced from those pages' own probes), (b) on every non-NATIVE cell

Nothing here is fitted on a rental-6 page. The 8x7B source fit is rental 5's (skewmodel.Engine,
v2). A, when a page has no `align_probe`, is the SEEN proxy (`align_proxy`): the median over
every published GH200 BLOCK_M 32 report's probe at that (model, SHARED declaration), A at n > 9
extrapolated along the n 8 -> 9 slope, and a declaration with no table (c15) taking the c9
table. The scorers re-price with the page's own A: (b) is additive, so
T(page A) = T(registered) - A_proxy + A_page on every block-scan non-NATIVE cell.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "rental5")):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v3 as TV3  # noqa: E402
import skewmodel as SM  # noqa: E402
import wave_split_bytes as W  # noqa: E402
import wave_split_bytes_v3 as WV3  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

VARIANTS = ("v2", "v3", "v3_nob", "v3_noca", "LT3", "v3_all", "H8")
H_V3_US = round(TV3.V3_H_MS * 1e3, 3)
PUBLISHED = REPO / "results" / "published"


def align_proxy(root: Path = PUBLISHED) -> dict:
    """{"model|D_shared": {arm: {n: A_us}, "native_abs_us": {...}, "reports": k}}: the SEEN proxy
    table, per (model, SHARED declaration), from every published GH200 BLOCK_M 32 report's
    `align_probe` (each report's per-(arm, n) median, then the median over reports)."""
    tab: dict = {}
    nrep: dict = {}
    for rp in sorted(glob.glob(str(root / "*gh200*" / "**" / "private_weight_reference" / "*" / "report.json"),
                               recursive=True)):
        r = json.loads(Path(rp).read_text())
        ap = r.get("align_probe")
        if not ap or int(r.get("block_m", 32)) != 32:
            continue
        D = {t["arm"]: int(t["experts_declared"]) for t in r.get("treads_table") or []}
        by: dict = {}
        for c in ap["cells"]:
            by.setdefault((c["label"], int(c["tread"])), []).append(float(c["ms"]))
        key = f"{r['model']}|{D.get('shared')}"
        nrep[key] = nrep.get(key, 0) + 1
        for (lab, n), v in by.items():
            tab.setdefault((key, lab, n), []).append(st.median(v))
    out: dict = {}
    for (key, lab, n), v in sorted(tab.items()):
        out.setdefault(key, {}).setdefault(lab, {})[n] = st.median(v)
    res = {}
    for key, d in out.items():
        if "native" not in d:
            continue
        res[key] = {arm: {str(n): round(1e3 * (d[arm][n] - d["native"][n]), 2) for n in sorted(d[arm]) if n in d["native"]}
                    for arm in d if arm != "native"}
        res[key]["native_abs_us"] = {str(n): round(1e3 * v, 2) for n, v in sorted(d["native"].items())}
        res[key]["reports"] = nrep[key]
    return res


def proxy_A_us(proxy: dict, model: str, arm: str, n: int, copies: int) -> tuple[float | None, str]:
    """(A_us, how) from the proxy table; None when the model has no table."""
    if arm == "native":
        return 0.0, "native"
    E = MODEL_CONFIGS[model].num_experts
    how = "own declaration"
    tab = proxy.get(f"{model}|{E * copies}")
    if tab is None:
        tab, how = proxy.get(f"{model}|{E * 9}"), "c9 table (no table at this declaration)"
    if tab is None or arm not in tab:
        return None, "no table"
    t = tab[arm]
    if str(n) in t:
        return float(t[str(n)]), how
    top = max(int(k) for k in t)
    v = float(t[str(top)]) + (n - top) * (float(t[str(top)]) - float(t[str(top - 1)]))
    return round(v, 4), how + f", extrapolated from n {top - 1} -> {top}"


class Pricer:
    """One engine (rental 5's v2 source fits) plus the M4 byte models and the H8 refit."""

    def __init__(self, proxy: dict | None = None, *, h8: bool = True):
        self.eng = SM.Engine()
        self.ctx, self.f2 = self.eng.ctx, self.eng.fit2
        self.proxy = align_proxy() if proxy is None else proxy
        self._m4: dict = {}
        self.h8 = self.fit_h8() if h8 else None

    # ---- the H8 rival: 8x7B CAL refit with h free --------------------------------------
    def cal_align(self) -> dict:
        al = {}
        for p in self.eng.src2["use"]:
            rp = Path(p.path)
            rp = rp / "report.json" if rp.is_dir() else rp
            by: dict = {}
            for c in json.loads(rp.read_text())["align_probe"]["cells"]:
                by.setdefault((c["label"], int(c["tread"])), []).append(float(c["ms"]))
            al[p.run] = {k: st.median(v) for k, v in by.items()}
        return al

    def fit_h8(self) -> dict:
        """The 6-parameter 8x7B CAL refit: (T0, c, bw, s_small, s_block, h), A on every
        non-NATIVE cell from its own pages' probes (the median over the cell's pages)."""
        al = self.cal_align()
        cells = [c for c in self.eng.src2["cells"] if c.fit]

        def A(c):
            if c.arm == "native":
                return 0.0
            return st.median(al[r][(c.arm, c.n)] - al[r][("native", c.n)] for r in c.runs)
        Ac = {c.key: A(c) for c in cells}

        def fun(x):
            return np.array([(TM.call_ms(x[:5], c, self.ctx, self.f2.k_w) + Ac[c.key]
                              + (x[5] if c.arm == "shared" else 0.0)) / c.ms - 1 for c in cells])
        old = TM.set_model("mixtral-8x7b")
        try:
            with TM.dead_model("v2"):
                xb, _rb, _cost = TM.PTF.bounded_lsq(fun, np.append(self.f2.x, -0.005),
                                                   np.append(TM.LB, -0.05), np.append(TM.UB, 0.05))
                rf = fun(xb)
        finally:
            TM.set_model(old)
        return {"x": [float(v) for v in xb[:5]], "h_ms": float(xb[5]),
                "rms": float(np.sqrt(np.mean(rf ** 2))), "cells": len(cells)}

    # ---- bytes ---------------------------------------------------------------------------
    def m4(self, model: str, copies: int) -> WV3.ModelV3:
        key = (model, copies)
        if key not in self._m4:
            self._m4[key] = WV3.ModelV3(self.eng.geom(model, copies))
        return self._m4[key]

    def reads(self, model: str, n: int, counts, copies: int, seed, version: str) -> dict | None:
        """Per-GEMM predicted DRAM read bytes (SHARED's walk for both arms, as rental 5)."""
        self.eng.geom(model, copies)
        m = self.eng._models[(model, copies)] if version == "v2" else self.m4(model, copies)
        tail = () if counts is None else ((tuple(counts), seed) if version == "v3" else (tuple(counts),))
        ev = m.evaluate(self.eng.prm, [("shared", SM.G_SKEW, n, g) + tail for g in W.GEMMS])
        if any(r["ill"] for r in ev):
            return None
        q = {g: float(r["q"]) for g, r in zip(W.GEMMS, ev, strict=True)}
        return {g: q[g] * TM.BYTE_MODEL[f"W_{g}"] + n * TM.BYTE_MODEL[f"operand_per_tile_{g}"] for g in TM.GEMMS}

    # ---- time ----------------------------------------------------------------------------
    def price(self, model: str, arm: str, n: int, counts=None, copies: int = 9, seed=None,
              A_us: float | None = None) -> dict | None:
        """Every variant's ms for one cell; A_us the page's A (None: the proxy)."""
        ct = None if counts is None else tuple(int(v) for v in counts)
        how = "page"
        if A_us is None:
            A_us, how = proxy_A_us(self.proxy, model, arm, n, copies)
        old = TM.set_model(model)
        try:
            r2 = self.reads(model, n, ct, copies, seed, "v2")
            r3 = self.reads(model, n, ct, copies, seed, "v3")
            if r2 is None or r3 is None:
                return None
            A = None if A_us is None else A_us * 1e-3
            c2 = self.eng.cell(model, arm, n, ct, copies, reads=r2)
            c3 = self.eng.cell(model, arm, n, ct, copies, reads=r3)
            npath = self.eng.path(model, "native", n, copies)
            x, k = self.f2.x, self.f2.k_w
            v3 = dict(native_path=npath, align_ms=A)
            out = {}
            with TM.dead_model("v2"):
                out["v2"] = TM.call_ms(x, c2, self.ctx, k)
                out["v3_nob"] = TM.call_ms(x, c3, self.ctx, k)
                out["v3"] = TV3.call_ms(x, c3, self.ctx, k, **v3)
                out["v3_noca"] = TV3.call_ms(x, c2, self.ctx, k, **v3)
                with SM.no_wave_q():
                    out["LT3"] = TV3.call_ms(x, c3, self.ctx, k, **v3)
                b_all = 0.0 if arm == "native" else (A or 0.0) + (TV3.V3_H_MS if arm == "shared" else 0.0)
                out["v3_all"] = out["v3_nob"] + b_all
                if self.h8 is not None:
                    hJ = self.h8["h_ms"] if arm == "shared" else 0.0
                    out["H8"] = TM.call_ms(np.array(self.h8["x"]), c3, self.ctx, k) + (0.0 if arm == "native" else (A or 0.0)) + hJ
            out.update(bytes_v2=r2, bytes_v3=r3, A_us=A_us, A_how=how, path=c3.path, native_path=npath,
                       dead={g: TM.dead_ctas(c3.declared, n, g, ct) for g in TM.GEMMS}, declared=c3.declared)
            return out
        finally:
            TM.set_model(old)

    def price_counted(self, model: str, arm: str, n: int, counts, reads: dict, copies: int = 9,
                      A_us: float | None = None) -> float:
        """E2-SKEW: v3's time of one cell from a counter page's read bytes per GEMM."""
        if A_us is None:
            A_us, _ = proxy_A_us(self.proxy, model, arm, n, copies)
        old = TM.set_model(model)
        try:
            c = self.eng.cell(model, arm, n, counts, copies, reads=reads)
            with TM.dead_model("v2"):
                return TV3.call_ms(self.f2.x, c, self.ctx, self.f2.k_w,
                                   native_path=self.eng.path(model, "native", n, copies),
                                   align_ms=None if A_us is None else A_us * 1e-3)
        finally:
            TM.set_model(old)


def reprice(registered_ms: float, A_proxy_us: float | None, A_page_us: float | None, arm: str,
            native_path: str, variant: str = "v3") -> float:
    """A registered prediction carried to the page's own A: (b) is additive in A, so only
    block-scan non-NATIVE cells move (v3_all and H8 move on every non-NATIVE cell)."""
    if arm == "native" or A_page_us is None or A_proxy_us is None:
        return registered_ms
    moves = variant in ("v3_all", "H8") or (variant in ("v3", "v3_noca", "LT3") and native_path == TM.BLOCK_SCAN)
    return registered_ms + (A_page_us - A_proxy_us) * 1e-3 if moves else registered_ms


def as_json(d):
    if dataclasses.is_dataclass(d):
        return dataclasses.asdict(d)
    return d
