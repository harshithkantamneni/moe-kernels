"""Rental 5's skew pricing: model v2 with the per-expert schedule (MODEL M2, r3_timing_model)
and the per-expert byte walk (MODEL M3, wave_split_bytes), and the rivals U, LT, PW and M1.

Used by register.py to write every skew and c15 prediction BEFORE any page, and by
score_skew.py only for E2-SKEW (time from a page's counted bytes, at the registered fit and
code). Nothing here is fitted on a skew page: the timing fit is 8x7B's CAL 2026-09-27 pages
under v2 (and under M for the printed M1), the byte model is the registered WSC MIX view on
8x7B's CAL counter pages, carried to OLMoE and Qwen1.5 by their shapes and R3's design
(cross_model_predict.target_geometry), as every cross-model registration did.

THE RIVALS (design-r5 2.4, design-r5-review d):
  S   v2 + M2 + M3: a partly filled M-tile is a whole live CTA, live = sum_e t_e, an expert
      with no row owns nothing, the bytes are the per-expert walk's
  U   skew ignored: the cell priced at the uniform histogram of the same tokens (r_U = 1)
  LT  S without the last wave's quantisation (wave_q = 1): the close rival
  PW  S plus each expert's CTAs quantised to their own waves (the grouped-GEMM view)
  M1  S under model M's dead term (DEAD_CTA_NS 1.333, k_w), printed only
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cores_heldout_predict as CP  # noqa: E402
import cross_model_predict as CMP  # noqa: E402
import private_weight_reference as PWR  # noqa: E402
import r3_timing_model as TM  # noqa: E402
import wave_split_bytes as W  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

G_SKEW = 8
COPIES = 9
BLOCK_M = 32


@contextlib.contextmanager
def no_wave_q():
    """LT: the floor without the last wave's idle-SM stretch."""
    saved = TM.wave_q
    TM.wave_q = lambda live, sms: 1.0
    try:
        yield
    finally:
        TM.wave_q = saved


class Engine:
    """The source fits (v2 and M) and the byte model, built once."""

    def __init__(self):
        argv = [*map(str, CP.source_pages()), "--counters", str(CP.C27)]
        with TM.dead_model("v2"):
            self.src2 = TM.build(TM.build_parser().parse_args(argv))
        with TM.dead_model("m"):
            self.srcM = TM.build(TM.build_parser().parse_args(argv))
        self.ctx = self.src2["ctx"]
        self.fit2, self.fitM = self.src2["main"], self.srcM["main"]
        self.card = W.load_card(CP.C27)
        self.wsc = W.analyse(self.card)
        self.prm = self.wsc.params(W.REGISTERED_VIEW)
        self._geom: dict = {}
        self._models: dict = {}

    # ---- geometry -----------------------------------------------------------
    def geom(self, model: str, copies: int = COPIES) -> W.Geometry:
        key = (model, copies)
        if key not in self._geom:
            src, tgt = MODEL_CONFIGS[self.card.geom.model], MODEL_CONFIGS[model]
            same = (src.num_experts, src.top_k) == (tgt.num_experts, tgt.top_k)
            design = None if same else CMP.target_design(model, BLOCK_M)
            g = CMP.target_geometry(self.card.geom, model, design)
            E = tgt.num_experts
            decl = dict(g.declared)
            decl["shared"] = E * copies
            decl["private"] = E * copies
            g = dataclasses.replace(g, copies_declared=copies,
                                    declared=tuple(sorted(decl.items())))
            self._geom[key] = g
            self._models[key] = W.Model(g)
        return self._geom[key]

    def wsc_q(self, model: str, arm: str, n: int, counts, copies: int = COPIES) -> dict | None:
        """Per-GEMM q (weight sets) of the registered view; NATIVE's are SHARED's walk (the
        byte model has no dead-CTA term). None when either GEMM is ILL-POSED."""
        self.geom(model, copies)
        m = self._models[(model, copies)]
        qa = "shared"
        cells = [(qa, G_SKEW, n, g) + ((tuple(counts),) if counts is not None else ()) for g in W.GEMMS]
        ev = m.evaluate(self.prm, cells)
        if any(r["ill"] for r in ev):
            return None
        return {g: float(r["q"]) for g, r in zip(W.GEMMS, ev, strict=True)}

    # ---- one cell -----------------------------------------------------------
    def declared(self, model: str, arm: str, copies: int = COPIES) -> int:
        E = MODEL_CONFIGS[model].num_experts
        return E if arm == "native" else E * copies

    def path(self, model: str, arm: str, n: int, copies: int = COPIES) -> str:
        cfg = MODEL_CONFIGS[model]
        numel = PWR.ids_for_tread(cfg, n, BLOCK_M)
        return PWR.align_path(numel, self.declared(model, arm, copies))

    def cell(self, model: str, arm: str, n: int, counts, copies: int = COPIES,
             reads: dict | None = None) -> TM.Cell | None:
        """A TM.Cell with predicted (or given counted) bytes; TM must be set to `model`."""
        D = self.declared(model, arm, copies)
        ct = None if counts is None else tuple(int(v) for v in counts)
        if reads is None:
            q = self.wsc_q(model, arm, n, ct, copies)
            if q is None:
                return None
            reads = {g: q[g] * TM.BYTE_MODEL[f"W_{g}"] + n * TM.BYTE_MODEL[f"operand_per_tile_{g}"]
                     for g in TM.GEMMS}
        sigma = {}
        for g in TM.GEMMS:
            w = TM.window_width(self.fit2.k_w, self.ctx.sms, self.ctx.occupancy[g])
            sigma[g] = reads[g] / TM.window(arm, D, G_SKEW, n, g, w, ct).reads
        return TM.Cell(arm=arm, G=G_SKEW, n=n, ms=float("nan"), mhz=None,
                       path=self.path(model, arm, n, copies), declared=D, runs=(),
                       source="PREDICTED", reads=reads, sigma=sigma, fit=False, counts=ct)

    def pw_extra_ms(self, cell: TM.Cell, c_ns: float) -> float:
        """PW's add-on: each expert's CTAs quantised to their own waves, past S's floor."""
        counts = cell.counts if cell.counts is not None else TM.uniform_counts(cell.n)
        out = 0.0
        t = TM.expert_tiles(counts)
        for g in TM.GEMMS:
            occ = self.ctx.occupancy[g]
            npn = TM.GEOMETRY[g].npn
            life = TM.floor_ksteps(g) * c_ns * 1e-6 * occ
            fpw = sum(math.ceil(int(te) * npn / (self.ctx.sms * occ)) for te in t if te) * life
            live = int(t.sum()) * npn
            fs = live * TM.wave_q(live, self.ctx.sms) * TM.floor_ksteps(g) * c_ns * 1e-6 / self.ctx.sms
            out += max(0.0, fpw - fs)
        return out

    def price(self, model: str, arm: str, n: int, counts, copies: int = COPIES) -> dict:
        """{S, LT, PW, M1} ms of one cell (None where the byte model is ILL-POSED), with the
        live and dead CTA counts and the predicted q."""
        old = TM.set_model(model)
        try:
            cell = self.cell(model, arm, n, counts, copies)
            if cell is None:
                return {"S": None, "LT": None, "PW": None, "M1": None, "q": None}
            with TM.dead_model("v2"):
                S = TM.call_ms(self.fit2.x, cell, self.ctx, self.fit2.k_w)
                with no_wave_q():
                    LT = TM.call_ms(self.fit2.x, cell, self.ctx, self.fit2.k_w)
                PW = S + self.pw_extra_ms(cell, float(self.fit2.x[1]))
            with TM.dead_model("m"):
                M1 = TM.call_ms(self.fitM.x, cell, self.ctx, self.fitM.k_w)
            ct = cell.counts
            q = {g: (cell.reads[g] - n * TM.BYTE_MODEL[f"operand_per_tile_{g}"]) / TM.BYTE_MODEL[f"W_{g}"]
                 for g in TM.GEMMS}
            return {"S": S, "LT": LT, "PW": PW, "M1": M1, "q": q,
                    "live_rows": TM.live_rows(n, ct),
                    "dead_ctas": {g: TM.dead_ctas(cell.declared, n, g, ct) for g in TM.GEMMS},
                    "path": cell.path, "declared": cell.declared}
        finally:
            TM.set_model(old)

    def price_counted(self, model: str, arm: str, n: int, counts, reads: dict,
                      copies: int = COPIES) -> float:
        """E2-SKEW: S's time of one cell from given read bytes per GEMM (a counter page's)."""
        old = TM.set_model(model)
        try:
            cell = self.cell(model, arm, n, counts, copies, reads=reads)
            with TM.dead_model("v2"):
                return TM.call_ms(self.fit2.x, cell, self.ctx, self.fit2.k_w)
        finally:
            TM.set_model(old)

    def source_doc(self) -> dict:
        return {"timing_fit_v2": self.fit2.params, "timing_fit_m": self.fitM.params,
                "k_w": self.fit2.k_w, "p_knee": TM.P_KNEE, "sms": self.ctx.sms,
                "occupancy": dict(self.ctx.occupancy), "clock_mhz": self.ctx.clock_mhz,
                "timing_pages": [p.run for p in self.src2["use"]],
                "dead_v2": {"d_ns": TM.V2_DEAD_NS, "kappa": TM.V2_KAPPA},
                "byte_view": W.REGISTERED_VIEW, "byte_params": self.prm.as_json(),
                "byte_card": self.card.geom.card,
                "byte_pages": sorted(Path(p).name for p in self.card.paths.values())}
