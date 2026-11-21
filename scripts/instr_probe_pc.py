#!/usr/bin/env python
"""THE INSTRUMENTED KERNEL ON FREE CELLS AND FREE TILES: rental PC's probe (docs/registered/2026-11-06-pc-gh200).

    python scripts/instr_probe_pc.py --mode regcheck --variants FILE --out DIR [--dry-run]
    python scripts/instr_probe_pc.py --mode perturb  --variants FILE --only ID[,ID] --out DIR [--dry-run]
    python scripts/instr_probe_pc.py --mode stamps   --variants FILE --variant ID --gate FILE --regcheck FILE --out DIR

SCAFFOLDING, NOT A KERNEL. The kernel is moe/instrumented/fused_moe_instr.py (vLLM v0.27.1's fused_moe_kernel with
rental 4's owner-approved timestamps), unchanged, at the "ends" level only (STAMPS 3: a CTA's start before the dead
check, a dead CTA's exit, the end after the final store; nothing near the k-loop, no eviction hint). This file builds
inputs (routing, token counts, synthetic or SEEN weights, configs) and times and stamps the existing kernel through
vLLM's own fused_experts and its override hook. scripts/instr_probe_c3.py and scripts/instr_probe.py (both pinned)
are imported and stay byte-identical; what neither can do is here:

  free cells   a PC page (moe-kernels/pc-page/1, scripts/scoring/pc/pccommon.check_page_doc): per cell its shape, T,
               counts, config (BLOCK_SIZE_M, N, K, num_warps, num_stages; G 1), flush and read GEMM. The ids are
               r3_decode.cell_ids (realize_counts, then the page's token-row shuffle), NATIVE, one weight copy (the
               flush makes every call cold). V-BIN and V-ALIGN (r3_decode.check_readback with the cell's config)
               run on an untimed eager call before any timed or stamped call of the cell; a mismatch fails the cell
  free tiles   the registered tile list (pccommon.TILES) in place of C3's single tile; regcheck's path leg reads the
               census predicate (wgmma iff BLOCK_M % 64 == 0 and num_warps % 4 == 0) instead of requiring wgmma
  shapes       synthetic shapes from the registered file (docs/registered/2026-11-06-pc-shapes.json) built here as
               moe.spec.MoEConfig, never added to moe/spec.py; a shape equal to any model's (E, H, F, k) is refused
  tread cells  a histogram variant (Phi-3.5's 64g1 page) through C3's own Card.call path at a registered tile
  the guard    Mixtral-8x22B, Qwen2-57B (and its tp8 shard) and Granite-3.0-1B are reserved (owner decision 3): a
               variant, cell or weight build naming one, or a shape equal to one, is refused before anything loads
Everything else is C3's: Keep, gate_c3, overhead, stamp_span, sass_checks_c3, ptx_path, the burst, the flusher,
sleep_ahead, BufferPool, read_env, the exit codes, the PERTURBED rule.

REGCHECK (compile only, plus V-ALIGN). Per variant and per distinct config among its cells: the plain kernel, the
all-off copy and the stamped copy's registers, shared and CTAs per SM (instr_probe.regcheck_rows), PASS when the copy
keeps the plain kernel's CTAs per SM and both sit on the census predicate's MMA path; a cell failing its V-BIN or
V-ALIGN readback is excluded and listed (no cell read back FAILs the variant). Each variant's line is written as soon
as it is decided; an uncaught error exits 4 (never 0 or 1). Each config's PTX loads in flight (pccommon.inflight_stages) are printed. DIR/regcheck.json,
DIR/regcheck.env (REGCHECK_<id>=PASS|FAIL ...).

PERTURB (C3's gate, owner decision 5). The variants named by --only, every cell: the plain kernel against the stamped
copy, instr_probe's burst (REPS bursts of CALLS calls each way, alternating, an L2 flush before each call outside the
event pairs, the GPU held ahead). DIR/perturb.json (per (cell, GEMM) plain_ms, copy_ms, overhead_median and
overhead_worst: the uninstrumented reference), DIR/gate.env (GATE_<id>, HARD_<id>: C3's split).

STAMPS. One variant whose HARD line and regcheck line read PASS (else REFUSED, exit 2, nothing launched); a timing FAIL
alone labels the page PERTURBED. Per cell: the readback, STAMP_WARMUP warm-up calls, then pccommon.stamp_calls stamped
calls (3, or 6 where the read GEMM has under 4 live CTAs), each with its buffers reserved, the GPU held ahead and an L2
flush (none on a flush-false cell: the L2 control). Per launch DIR/stamps/<label>-call<c>-<w1|w2>.npy (the int64 rows)
and ...-experts.npy, indexed in DIR/stamps.json with the launch record, num_tokens_post_padded, USE_TD, the event-pair
ms and the stamps' own span; each compiled config's SASS (C3's bracket checks) and PTX (DIR/ptx/, its loads in flight).

EXIT CODES (C3's): 0 done; 1 a gate or regcheck FAILED; 2 refused (and --dry-run); 3 a cell failed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE), str(HERE / "scoring" / "pc")):
    if p not in sys.path:
        sys.path.insert(0, p)

import instr_probe as IP  # noqa: E402
import instr_probe_c3 as C3  # noqa: E402
import pccommon as PCC  # noqa: E402
import private_weight_reference as PW  # noqa: E402

from moe import instrumented as I  # noqa: E402
from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

LEVEL = I.STAMP_LEVELS["ends"]
STAMP_WARMUP = IP.STAMP_WARMUP
TOL = IP.PERTURB_TOL_R5
DECLARED_COPIES = C3.DECLARED_COPIES
NEED = ("id", "kind", "probe", "spec")
NEED_HIST = ("model", "groups", "arms", "block_m", "block_n", "block_k", "num_warps", "num_stages", "histogram")
REFUSALS = (PCC.Refused, C3.Refused, PW.PrivateWeightRefusal)


class Refused(PCC.Refused):
    pass


# --------------------------------------------------------------------------
# variants and cells (pure)
# --------------------------------------------------------------------------

def _spec(v: dict):
    try:
        spec = I.parse_spec(v["spec"])
    except I.SpecError as exc:
        raise Refused(f"variant {v['id']}: {exc}") from None
    if spec.stamps != LEVEL or spec.hints:
        raise Refused(f"variant {v['id']}: PC stamps the CTA ends only (stamps=ends, no hint), not {spec.text()}")
    return spec


def _rel(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise Refused(f"{path}: a repo-relative path")
    return REPO / p


def decode_cells(v: dict, shapes: dict) -> tuple[list[dict], str, dict]:
    """A page variant's cells, the page's sha256 and the page document."""
    path = _rel(v["page"])
    try:
        data = path.read_bytes()
        doc = json.loads(data)
    except (OSError, ValueError) as exc:
        raise Refused(f"variant {v['id']}: cannot read the page {v['page']}: {exc}") from None
    cells = PCC.check_page_doc(doc, shapes)
    for c in cells:
        bm, bn, bk, _w, st = c["tile"]
        why = PW.config_refusal(c["cfg"], block_m=bm, block_n=bn, block_k=bk, num_stages=st)
        if why:
            raise Refused(f"variant {v['id']} cell {c['label']}: the config cannot run: {why}")
        c.update(kind="decode", stem=c["label"], model=c["shape"], arm="native")
    return cells, hashlib.sha256(data).hexdigest(), doc


def hist_cells(v: dict) -> tuple[list[dict], object]:
    """A histogram variant's NATIVE cells at a registered tile (C3's check_variant with the tile list)."""
    miss = [k for k in NEED_HIST if k not in v]
    if miss:
        raise Refused(f"variant {v['id']} has no {miss}")
    cfg = PCC.shape_cfg(str(v["model"]))
    if v["model"] not in MODEL_CONFIGS:
        raise Refused(f"variant {v['id']}: {v['model']} is not a model (a histogram variant runs a model's weights)")
    tile = tuple(int(v[k]) for k in ("block_m", "block_n", "block_k", "num_warps", "num_stages"))
    if PCC.tile_key(tile) not in PCC.TILES:
        raise Refused(f"variant {v['id']}: tile {PCC.tile_key(tile)} is not one of the registered {sorted(PCC.TILES)}")
    if [int(g) for g in v["groups"]] != [1]:
        raise Refused(f"variant {v['id']}: groups {v['groups']}; PC's cells are G 1")
    if [str(a) for a in v["arms"]] != [PW.NATIVE]:
        raise Refused(f"variant {v['id']}: arms {v['arms']}; PC stamps NATIVE only")
    bm, bn, bk, _nw, st = tile
    why = PW.config_refusal(cfg, block_m=bm, block_n=bn, block_k=bk, num_stages=st)
    if why:
        raise Refused(f"variant {v['id']}: the config cannot run: {why}")
    try:
        page = PW.load_histogram_page(_rel(v["histogram"]), cfg, bm, v["model"])
    except PW.HistogramRefused as exc:
        raise Refused(f"variant {v['id']}: {exc}") from None
    cells = []
    for c in page.ordered():
        if PW.NATIVE not in (c.arms or (PW.NATIVE,)):
            continue
        cells.append({"label": c.label, "n": int(c.n), "arm": PW.NATIVE, "kind": "hist", "model": v["model"],
                      "shape": v["model"], "cfg": cfg, "tile": tile, "flush": True, "read": "w1",
                      "stem": C3.stem(c.label, int(c.n), PW.NATIVE), "config": PCC.config_of(tile)})
    if not cells:
        raise Refused(f"variant {v['id']}: no NATIVE cell in {v['histogram']}")
    return cells, page


def check_variant(v: dict, shapes: dict | None = None) -> dict:
    """One PC variant validated before anything runs: a page variant (page=) or a histogram variant (histogram=)."""
    miss = [k for k in NEED if k not in v]
    if miss:
        raise Refused(f"variant {v.get('id')!r} has no {miss}")
    if v["kind"] != "stamps" or v["probe"] != "pc":
        raise Refused(f"variant {v['id']}: kind {v['kind']!r}, probe {v['probe']!r}; this probe runs PC stamps units")
    spec = _spec(v)
    gate_spec = dataclasses.replace(spec, evict_a="none", evict_b="none")
    if ("page" in v) == ("histogram" in v):
        raise Refused(f"variant {v['id']}: page= or histogram=, one of them")
    if "page" in v:
        shapes = PCC.load_shapes() if shapes is None else shapes
        cells, sha, doc = decode_cells(v, shapes)
        return {**v, "spec_obj": spec, "gate_spec": gate_spec, "cells": cells, "source": v["page"],
                "source_sha256": sha, "shuffle_seed": int(doc["shuffle_seed"]), "page_doc": doc, "kind_": "page"}
    cells, page = hist_cells(v)
    tile = cells[0]["tile"]
    return {**v, "spec_obj": spec, "gate_spec": gate_spec, "cells": cells, "source": v["histogram"],
            "source_sha256": page.file_sha256, "page": page, "tile": tile, "arms": [PW.NATIVE], "kind_": "hist"}


def load_variants(path: Path) -> list[dict]:
    try:
        vs = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"{path}: {exc}") from None
    if not isinstance(vs, list) or not vs:
        raise Refused(f"{path}: a JSON list of variants")
    ids = [v.get("id") for v in vs]
    if len(set(ids)) != len(ids):
        raise Refused(f"{path}: duplicate variant ids {ids}")
    shapes = PCC.load_shapes()
    return [check_variant(v, shapes) for v in vs]


def order(cells: list[dict]) -> list[dict]:
    """Cells grouped by shape in the order each shape first appears (one weight build per shape), the page's order
    within a shape."""
    first = {}
    for i, c in enumerate(cells):
        first.setdefault(c["shape"], i)
    return sorted(cells, key=lambda c: (first[c["shape"]], cells.index(c)))


def conf_of(tile) -> dict:
    """vLLM's config for a cell: R3's pinned_config at the tile, BLOCK_M and num_warps put over it (C3's conf)."""
    bm, bn, bk, nw, st = (int(x) for x in tile)
    return dict(PW.pinned_config(bn, 1, st, **PW.block_k_kw(bk)), BLOCK_SIZE_M=bm, num_warps=nw)


def regcheck_pc(rows: list[dict], own: str, paths: dict) -> tuple[bool, str]:
    """instr_probe.regcheck_ok (CTAs per SM of the all-off copy and the own spec equal the plain kernel's) and PC's
    path leg: every config of the plain kernel and the stamped copy on the census predicate's MMA path."""
    ok, why = IP.regcheck_ok(rows, own)
    bad = []
    for who, d in paths.items():
        for k, p in d.items():
            want = PCC.mma_path(k[0], k[5])
            if p["mma"] != want:
                bad.append(f"{who} {IP.key_name(tuple(k))} on the {p['mma']} path, the predicate's is {want}")
    if not any(paths.values()):
        bad.append("no PTX read")
    why = "; ".join([w for w in [why] if w] + bad)
    return ok and not bad, why


def plan_lines(variants: list[dict], mode: str, only=None) -> list[str]:
    out = [f"INSTRUMENTED KERNEL PC ({mode}): {len(variants)} variant(s); upstream vLLM {I.UPSTREAM['tag']} fused_moe.py "
           f"sha256 {I.UPSTREAM['file_sha256'][:16]}"]
    for v in variants:
        if only is not None and v["id"] not in only:
            continue
        tiles = sorted({PCC.tile_key(c["tile"]) for c in v["cells"]})
        shapes = sorted({c["shape"] for c in v["cells"]})
        out.append(f"  {v['id']:12s} {v['kind_']:5s} {v['source']} cells {len(v['cells'])} tiles {' '.join(tiles)} "
                   f"shapes {' '.join(shapes)} spec {v['spec_obj'].text()}")
    if mode == "perturb":
        out.append(f"  gate (C3's): median |plain/copy - 1| <= {100 * TOL['median_max']:.0f}%, worst <= "
                   f"{100 * TOL['worst_max']:.0f}%, CTAs per SM identical, all-off SASS equal, the upstream file, C3's SASS "
                   f"legs; {IP.REPS} bursts of {IP.CALLS} calls each way on every cell of {sorted(only or [])}")
    if mode == "stamps":
        out.append(f"  {STAMP_WARMUP} warm-up calls, then {PCC.STAMP_CALLS} stamped calls per cell ({PCC.STAMP_CALLS_SMALL} "
                   f"where the read GEMM has under {PCC.SMALL_G} live CTAs), each held ahead and flushed (not on a "
                   "flush-false cell); on a PASS regcheck and gate hard legs (PERTURBED on a timing FAIL alone)")
    return out


# --------------------------------------------------------------------------
# the card
# --------------------------------------------------------------------------

class Card(C3.Card):
    """C3's box (vLLM, the override hook, the burst timer, the flush, its histogram cells) with PC's decode cells."""

    #: the device the decode cells' weights and inputs live on (a test seam: the suite runs a cell on the CPU)
    device = "cuda"

    def __init__(self, args):
        super().__init__(args)
        self.shapes = PCC.load_shapes()

    def load(self, model: str):
        PCC.shape_cfg(model, self.shapes)          # the guard, before any weight build
        return super().load(model)

    def load_shape(self, name: str):
        """One copy of a shape's weights (synthetic or SEEN), NATIVE."""
        key = f"shape:{name}"
        if self.model == key:
            return
        cfg = PCC.shape_cfg(name, self.shapes)
        torch = self.torch
        self.w1 = self.w2 = self.nw1 = self.nw2 = None        # Phi's 9 copies (22 GB) go too (review MINOR-15)
        torch.cuda.empty_cache()
        self.copies = 1
        self.w1, self.w2, _ = PW.build_private_weights(cfg, "bf16", 1, self.args.seed, device=self.device)
        g = torch.Generator(device=self.device).manual_seed(int(self.args.seed) + 1)
        self.x_all = torch.randn((4096, cfg.hidden_size), device=self.device, dtype=self.w1.dtype, generator=g)
        self.cfg, self.model, self.inputs = cfg, key, {}

    def cell_fn(self, v: dict, c: dict):
        if c["kind"] == "hist":
            self.load(c["model"])
            return self.call(v, c["label"], c["n"], PW.NATIVE), None
        import r3_decode as RD
        self.load_shape(c["shape"])
        torch, cfg = self.torch, self.cfg
        if c["label"] not in self.inputs:
            cell = RD.DecodeCell(label=c["label"], T=int(c["T"]), counts=tuple(c["counts"]), arms=(RD.NATIVE,))
            ids = RD.cell_ids(cfg, cell, v["shuffle_seed"], device=self.device).to(torch.int32)
            self.inputs[c["label"]] = ids
        ids = self.inputs[c["label"]]
        x = self.x_all[:int(c["T"])].contiguous()
        w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32, device=self.device)
        fe, act, w1, w2, E = self.fused_experts, self.activation(), self.w1, self.w2, cfg.num_experts

        def call():
            return fe(hidden_states=x, w1=w1, w2=w2, topk_weights=w, topk_ids=ids, activation=act,
                      apply_router_weight_on_input=False, global_num_experts=E, expert_map=None)
        return call, ids

    def activation(self):
        """vLLM's silu activation for fused_experts (r3_decode's call)."""
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        return MoEActivation("silu")

    def readback(self, c: dict, fn, ids, conf: dict) -> list[str]:
        """V-BIN and V-ALIGN on one untimed eager call of a decode cell: r3_decode.check_readback against the
        expected readback at the cell's config (GD5's config clause holds the kernel to the cell's config; the
        override is PC's, so the tuned-file and override clauses are not asked)."""
        if c["kind"] == "hist":
            return []
        import r3_decode as RD
        with self.override_config(conf):
            with RD.Spy(self.FM) as spy:
                fn()
                self.torch.cuda.synchronize()
        exp = RD.expected_readback(self.cfg, int(c["T"]), list(c["counts"]), ids.reshape(-1).tolist(),
                                   declared=self.cfg.num_experts, copies=1, conf=conf)
        return RD.check_readback({"gemms": spy.records}, exp)


def _facts(launches, into):
    return IP.facts_by_config(launches, into)


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a temporary file beside `path`, then os.replace it: a reader never sees a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_save(path: Path, array) -> None:
    """numpy.save through a temporary file and os.replace (the name keeps its .npy suffix)."""
    import numpy as np
    tmp = path.with_name(path.name[:-4] + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def distinct_compiles(launches: list[dict], into: dict) -> dict:
    """Every distinct compile of the unit, keyed (config_key, the CompiledKernel's hash): a config Triton compiled
    twice (another specialisation) gets its SASS legs and its loads in flight checked too (review MINOR-12)."""
    from moe.instrumented import cubin as CB
    for lr in launches:
        ck = lr.get("compiled")
        h = str(getattr(ck, "hash", None) or id(ck))[:16]
        key = (I.config_key(lr), h)
        if key not in into:
            f = CB.compiled_facts(ck)
            if f is not None:
                into[key] = f
    return into


def regcheck_env_line(vid: str, ok: bool, why: str, excluded: dict) -> str:
    """instr_probe.regcheck_line with the excluded cells appended (readers parse the first word only)."""
    line = IP.regcheck_line(vid, ok, why)
    if excluded:
        line += f" excluded {len(excluded)}: {','.join(sorted(excluded))}"
    return line


def _flush(out: Path, name_json: str, report: dict, name_env: str, env: list[str]) -> None:
    """The unit's files as they stand, written after every variant: a crash or a cap kill later in the unit keeps
    every line already decided (review MAJOR-2)."""
    atomic_write_text(out / name_json, json.dumps(report, indent=1, sort_keys=True, default=str))
    atomic_write_text(out / name_env, "\n".join(env) + ("\n" if env else ""))


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def regcheck_variant(card, args, v: dict) -> tuple[bool, str, dict]:
    """One variant's regcheck. A cell whose readback (V-BIN, V-ALIGN) fails, or whose call raises, is EXCLUDED and
    listed (its stamps cell then refuses on its own readback); the variant FAILs only on the occupancy or path legs,
    or when no cell reads back (review MAJOR-2: one cell no longer sinks a block)."""
    own = v["spec_obj"].text()
    plain_f, off_f, own_f, readbacks = {}, {}, {}, {}
    done_tiles = set()
    for c in order(v["cells"]):
        cf = conf_of(c["tile"])
        try:
            fn, ids = card.cell_fn(v, c)
            bad = card.readback(c, fn, ids, cf)
            readbacks[c["stem"]] = bad or "PASS"
            if bad or c["tile"] in done_tiles:
                continue
            _facts(card.compile_once(fn, cf, None), plain_f)
            _facts(card.compile_once(fn, cf, I.InstrSpec()), off_f)
            _facts(card.compile_once(fn, cf, v["spec_obj"]), own_f)
            done_tiles.add(c["tile"])
        except REFUSALS + (Exception,) as exc:                    # noqa: BLE001
            readbacks[c["stem"]] = [f"the cell raised {_err(exc)}"]
    rows = IP.regcheck_rows(plain_f, off_f, {own: own_f})
    paths = {"plain": {k: C3.ptx_path(f.get("ptx") or "") for k, f in plain_f.items()},
             "copy": {k: C3.ptx_path(f.get("ptx") or "") for k, f in own_f.items()}}
    ok, why = regcheck_pc(rows, own, paths)
    excluded = {s: b for s, b in readbacks.items() if b != "PASS"}
    missing = sorted({PCC.tile_key(c["tile"]) for c in v["cells"]} - {PCC.tile_key(t) for t in done_tiles})
    if len(excluded) == len(v["cells"]):
        ok, why = False, "; ".join([w for w in [why] if w] + ["no cell read back"])
    inflight = {IP.key_name(k): PCC.inflight_stages(f.get("ptx")) for k, f in own_f.items()}
    brackets = {}
    for k, f in own_f.items():
        t = IP.sass_of(f.get("cubin"), Path(args.out) / "sass" / IP.gate_key(v["id"]) / f"copy-{IP.key_name(k)}.sass")
        brackets[IP.key_name(k)] = C3.sass_checks_c3(t) if t else None
    rec = {"source": v["source"], "source_sha256": v["source_sha256"], "own_spec": own, "rows": rows,
           "own_spec_ok": ok, "why": why, "readbacks": readbacks, "excluded_cells": excluded,
           "tiles_not_compiled": missing,
           "paths": {w: {IP.key_name(k): p for k, p in d.items()} for w, d in paths.items()},
           "inflight_printed": inflight, "sass_checks_c3_printed": brackets,
           "rule": ("CTAs per SM identical to the plain kernel's and plain and copy on the census predicate's MMA path, "
                    "on every config compiled; a cell failing V-BIN or V-ALIGN is excluded and listed; no cell read "
                    "back FAILs the variant")}
    for r in rows:
        x = r["specs"][own]["copy"] or {}
        print(f"  {v['id']} {IP.key_name(tuple(r['config']))}: plain {r['plain']['regs']}r/{r['plain']['ctas_per_sm']} "
              f"off {'=' if r['off_same'] else '!='} | {own}: {x.get('regs')}r/{x.get('ctas_per_sm')}", flush=True)
    return ok, why, rec


def regcheck(args, variants: list[dict]) -> int:
    card = Card(args)
    report = {"tool": "scripts/instr_probe_pc.py --mode regcheck", "versions": PW._package_versions(), "variants": {}}
    worst, env, out = exit_codes.DONE, [], Path(args.out)
    for v in variants:
        try:
            ok, why, rec = regcheck_variant(card, args, v)
        except Exception as exc:                                  # noqa: BLE001
            ok, why, rec = False, f"the variant raised {_err(exc)}", {"error": _err(exc)}
        report["variants"][v["id"]] = rec
        env.append(regcheck_env_line(v["id"], ok, why, rec.get("excluded_cells") or {}))
        if not ok:
            worst = exit_codes.CLAIM_FAIL
        _flush(out, "regcheck.json", report, "regcheck.env", env)
    return worst


def perturb_variant(card, v: dict, upstream_ok: bool, out: Path) -> dict:
    """One variant's gate (C3's legs). A cell whose readback fails, or whose timing raises, is EXCLUDED and listed;
    the gate reads the cells that ran (review MAJOR-2)."""
    rows, plain_f, copy_f, off_f, excluded = [], {}, {}, {}, {}
    for c in order(v["cells"]):
        cf = conf_of(c["tile"])
        try:
            fn, ids = card.cell_fn(v, c)
            bad = card.readback(c, fn, ids, cf)
            if bad:
                excluded[c["stem"]] = bad
                continue
            per = {"plain": {"w1": [], "w2": []}, "copy": {"w1": [], "w2": []}}
            for rep in range(IP.REPS):
                for which in (("plain", "copy") if rep % 2 == 0 else ("copy", "plain")):
                    gspec = v["gate_spec"]
                    make = ((lambda ev, pool: I.timed_plain(module=card.FM, events=ev)) if which == "plain"
                            else (lambda ev, pool, gs=gspec: I.install(gs, module=card.FM, events=ev, alloc=pool)))
                    ms, launches = card.burst(fn, cf, make)
                    _facts(launches, plain_f if which == "plain" else copy_f)
                    ms = IP.split_gemms(ms)
                    for gemm in ("w1", "w2"):
                        per[which][gemm].append(statistics.median(ms[gemm]))
            for gemm in ("w1", "w2"):
                p, cc = statistics.median(per["plain"][gemm]), statistics.median(per["copy"][gemm])
                rows.append({"label": c["label"], "n": c.get("n"), "arm": PW.NATIVE, "stem": c["stem"], "gemm": gemm,
                             "tile": PCC.tile_key(c["tile"]), "plain_ms": p, "copy_ms": cc,
                             "ratio": p / cc if cc else None, "plain_reps": per["plain"][gemm],
                             "copy_reps": per["copy"][gemm],
                             **C3.overhead(p, cc, per["plain"][gemm], per["copy"][gemm])})
            _facts(card.compile_once(fn, cf, I.InstrSpec()), off_f)
        except REFUSALS + (Exception,) as exc:                    # noqa: BLE001
            excluded[c["stem"]] = [f"the cell raised {_err(exc)}"]
    configs = IP.compare_configs(plain_f, copy_f, IP.OCC_LEGS_R5)
    out_dir = out / "sass" / IP.gate_key(v["id"])
    sass_ok, sass_rows = IP.sass_equal_leg(plain_f, off_f, out_dir)
    brackets, loops = {}, {}
    for key, f in copy_f.items():
        t = IP.sass_of(f.get("cubin"), out_dir / f"copy-{IP.key_name(key)}.sass")
        if t:
            brackets[IP.key_name(key)] = C3.sass_checks_c3(t)
            loops[IP.key_name(key)] = C3.loop_compare(IP.sass_of((plain_f.get(key) or {}).get("cubin")), t)
    verdict = C3.gate_c3([r["ratio"] for r in rows], configs, upstream_ok=upstream_ok, sass_equal=sass_ok,
                         brackets=brackets)
    if not rows:
        verdict.update(hard_verdict="FAIL", verdict="FAIL", hard_why=verdict["hard_why"] + ["no cell ran"],
                       why=verdict["why"] + ["no cell ran"])
    return {"source": v["source"], "source_sha256": v["source_sha256"], "gate_spec": v["gate_spec"].as_dict(),
            "cells": rows, "excluded_cells": excluded, "configs": configs, "sass_equal": sass_ok,
            "sass_rows": sass_rows, "sass_checks_c3": brackets, "loop_sass_vs_plain_printed": loops, **verdict}


def perturb(args, variants: list[dict]) -> int:
    card = Card(args)
    torch = card.torch
    report = {"tool": "scripts/instr_probe_pc.py --mode perturb", "tolerance": TOL, "reps": IP.REPS,
              "calls": IP.CALLS, "upstream": card.upstream, "override_hook": card.where,
              "triton_cache": os.environ.get("TRITON_CACHE_DIR"), "versions": PW._package_versions(),
              "device": {"name": torch.cuda.get_device_name(0), "uuid": PW.device_identity()},
              "smi_before": IP._smi(), "variants": {}, "started": time.time()}
    upstream_ok = bool(card.upstream["file_matches"] and card.upstream["excerpt_matches"])
    env, worst, out = [], exit_codes.DONE, Path(args.out)
    for v in variants:
        try:
            rec = perturb_variant(card, v, upstream_ok, out)
        except Exception as exc:                                  # noqa: BLE001
            rec = {"error": _err(exc), "verdict": "FAIL", "hard_verdict": "FAIL", "median_dev": None, "worst_dev": None,
                   "why": [f"the variant raised {_err(exc)}"]}
        report["variants"][v["id"]] = rec
        env.append(f"GATE_{IP.gate_key(v['id'])}={rec['verdict']} median={rec['median_dev']} worst={rec['worst_dev']}")
        env.append(f"HARD_{IP.gate_key(v['id'])}={rec['hard_verdict']}")
        if rec["verdict"] != "PASS":
            worst = exit_codes.CLAIM_FAIL
        print(f"  {v['id']}: {rec['verdict']} {'; '.join(rec['why'])}", flush=True)
        report["smi_after"], report["ended"] = IP._smi(), time.time()
        _flush(out, "perturb.json", report, "gate.env", env)
    return worst


def _stamps_manifest(out: Path, args, v: dict, card, fl, cells_rec: dict, index: list, sass: list) -> None:
    torch = card.torch
    man = {"tool": "scripts/instr_probe_pc.py --mode stamps", "variant": v["id"], "kind": v["kind_"],
           "page_label": getattr(args, "page_label", None), "source": v["source"],
           "source_sha256": v["source_sha256"], "shapes_file": PCC.SHAPES_FILE,
           "shapes_sha256": PCC.sha256(REPO / PCC.SHAPES_FILE), "group_m": 1,
           "copies_declared": DECLARED_COPIES if v["kind_"] == "hist" else 1, "spec_text": v["spec_obj"].text(),
           "spec": v["spec_obj"].as_dict(), "columns": list(I.COLUMNS), "hdr": I.HDR, "upstream": card.upstream,
           "warmup": STAMP_WARMUP, "calls_rule": {"calls": PCC.STAMP_CALLS, "small": PCC.STAMP_CALLS_SMALL,
                                                   "small_below_live": PCC.SMALL_G},
           "triton_cache": os.environ.get("TRITON_CACHE_DIR"), "l2_flush_mb": fl.megabytes,
           "versions": PW._package_versions(), "device": {"name": torch.cuda.get_device_name(0),
                                                           "uuid": PW.device_identity()},
           "smi": IP._smi(), "sass": sass, "cells": cells_rec, "order": [c["stem"] for c in order(v["cells"])], "complete": bool(sass) or not index,
           "launches": index}
    atomic_write_text(out / "stamps.json", json.dumps(man, indent=1, sort_keys=True, default=str))


def stamps(args, v: dict) -> int:
    import numpy as np

    from moe.instrumented import fused_moe_instr
    card = Card(args)
    torch = card.torch
    out = Path(args.out)
    (out / "stamps").mkdir(parents=True, exist_ok=True)
    (out / "ptx").mkdir(parents=True, exist_ok=True)
    fl = card.flusher()
    index, cells_rec, bad, compiled = [], {}, 0, {}
    for c in order(v["cells"]):
        cf = conf_of(c["tile"])
        rec_c = {"label": c["label"], "shape": c["shape"], "kind": c["kind"], "tile": PCC.tile_key(c["tile"]),
                 "flush": c["flush"], "read": c["read"]}
        if c["kind"] == "decode":
            rec_c.update(T=c["T"], counts=c["counts"],
                         live={g: PCC.live_ctas(c["cfg"], c["counts"], c["tile"], g) for g in ("w1", "w2")})
        else:
            rec_c.update(n=c["n"])
        try:
            fn, ids = card.cell_fn(v, c)
            why = card.readback(c, fn, ids, cf)
            rec_c["readback"] = why or "PASS"
            if why:
                raise Refused(f"V-ALIGN: {why[:3]}")
            calls = PCC.stamp_calls(rec_c["live"][c["read"]]) if c["kind"] == "decode" else PCC.STAMP_CALLS
            rec_c["calls"] = calls
            pool = I.BufferPool()
            like = torch.empty(0, device="cuda")
            with card.override_config(cf):
                with I.install(v["spec_obj"], module=card.FM) as warm:
                    for _ in range(STAMP_WARMUP):
                        fn()
                torch.cuda.synchronize()
                shapes = I.BufferPool.shapes(warm.launches[-PW.GEMMS_PER_CALL:])
                for k in range(calls):
                    pool.reserve(shapes, like)      # the buffers before the flush (rental 5 item 4)
                    card.sleep_ahead()              # the GPU held ahead (C3 review MAJOR-2)
                    if c["flush"]:
                        fl.flush()                  # outside the stamped launches; none on the L2 control
                    sink: list = []
                    with I.install(v["spec_obj"], module=card.FM, kernel=C3.Keep(fused_moe_instr.fused_moe_kernel, sink),
                                   alloc=pool, events=IP.launch_event_factory(torch)) as rec:
                        fn()
                        torch.cuda.synchronize()
                    distinct_compiles(rec.launches, compiled)
                    ms_each = IP.launch_ms(rec.launches)
                    for lr, kept, gemm, lms in zip(rec.launches, sink, ("w1", "w2"), ms_each, strict=True):
                        if kept["use_td"]:
                            raise Refused("the launch took the tensor-descriptor path (USE_TD True): not the copy's "
                                          "pointer path")
                        name = f"{c['stem']}-call{k}-{gemm}"
                        rows = lr["buffer"].cpu().numpy()
                        atomic_save(out / "stamps" / f"{name}.npy", rows)
                        atomic_save(out / "stamps" / f"{name}-experts.npy", kept["expert_ids"].cpu().numpy())
                        ntpp = int(kept["num_tokens_post_padded"].reshape(-1)[0].item())
                        index.append({k2: lr[k2] for k2 in lr if k2 not in ("buffer", "events", "compiled")}
                                     | {"label": c["label"], "stem": c["stem"], "n": c.get("n"), "arm": PW.NATIVE,
                                        "call": k, "gemm": gemm, "file": f"{name}.npy",
                                        "experts_file": f"{name}-experts.npy", "num_tokens_post_padded": ntpp,
                                        "use_td": kept["use_td"], "stamp_span_ns": C3.stamp_span(rows),
                                        "launch_ms": lms, "late_buffers": pool.late, "flush": c["flush"],
                                        "tile": PCC.tile_key(c["tile"])})
        except Exception as exc:                                  # noqa: BLE001
            index.append({"label": c["label"], "stem": c["stem"], "error": f"{type(exc).__name__}: {exc}"})
            rec_c["error"] = f"{type(exc).__name__}: {exc}"
            bad += 1
        cells_rec[c["stem"]] = rec_c
        _stamps_manifest(out, args, v, card, fl, cells_rec, index, [])   # the page as it stands (a cap kill keeps it)
    sass = []
    for (key, h), f in compiled.items():
        tag = f"{IP.key_name(key)}-{h}"
        name = f"sass/{tag}.sass"
        text = IP.sass_of(f.get("cubin"), out / name)
        ptx = f.get("ptx") or ""
        pname = f"ptx/{tag}.ptx"
        atomic_write_text(out / pname, ptx)
        sass.append({"config": list(key), "compile": h, "file": name if text else None, "ptx_file": pname,
                     "checks_c3": C3.sass_checks_c3(text) if text else None, "ptx": C3.ptx_path(ptx),
                     "inflight": PCC.inflight_stages(ptx), "regs": f["regs"], "shared": f["shared"],
                     "ctas_per_sm": (f.get("occupancy") or {}).get("ctas_per_sm")})
    _stamps_manifest(out, args, v, card, fl, cells_rec, index, sass)
    print(f"STAMPS {v['id']}: {len(index)} launches, {bad} cells failed; {out}")
    return exit_codes.INVALID if bad else exit_codes.DONE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("perturb", "stamps", "regcheck"), required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--variant", default=None, help="stamps: the variant id to run")
    p.add_argument("--only", default=None, help="perturb: the variant ids to gate, comma-separated")
    p.add_argument("--gate", type=Path, default=None, help="stamps: the perturb units' gate env")
    p.add_argument("--regcheck", type=Path, default=None, help="stamps: the regcheck unit's regcheck env")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    only = None
    try:
        variants = load_variants(args.variants)
        ids = [v["id"] for v in variants]
        if args.mode == "stamps":
            mine = [v for v in variants if v["id"] == args.variant]
            if len(mine) != 1:
                raise Refused(f"--variant {args.variant!r} is not one of {ids}")
            variants = mine
        elif args.mode == "perturb":
            only = [x for x in (args.only or "").split(",") if x]
            if not only or set(only) - set(ids):
                raise Refused(f"--only {args.only!r}: the perturb unit names variants of {ids}")
            variants = [v for v in variants if v["id"] in only]
        elif args.only:
            raise Refused("--only belongs to perturb (regcheck checks every variant)")
    except REFUSALS as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    print("\n".join(plan_lines(variants, args.mode, only)))
    if args.mode == "stamps":
        g = C3.read_env(args.gate, "GATE", args.variant)
        h = C3.read_env(args.gate, "HARD", args.variant)
        if h != "PASS" or not g:
            print(f"REFUSED: the perturbation gate's hard legs for {args.variant} read {h or 'nothing'} ({args.gate}): "
                  "a PC stamps unit runs only when occupancy, the all-off SASS, the upstream file, C3's SASS legs and "
                  "every cell's readback pass")
            return exit_codes.REFUSED
        args.page_label = "CLEAN" if g == "PASS" else "PERTURBED"
        if g != "PASS":
            print(f"PERTURBED: the overhead gate for {args.variant} reads {g}; the stamps run and the pages are labelled "
                  "PERTURBED, the measured overhead beside every reading (C3's OD-1, owner decision 5)")
        r = IP.read_regcheck(args.regcheck, args.variant) if args.regcheck else ""
        if r != "PASS":
            print(f"REFUSED: the regcheck line for {args.variant} reads {r or 'nothing'} ({args.regcheck}): a PC stamps "
                  "unit runs only on its regcheck PASS")
            return exit_codes.REFUSED
    if args.dry_run:
        print("DRY RUN: nothing launched (exit 2)")
        return exit_codes.REFUSED
    why = IP.run_refusal()
    if why:
        print(f"REFUSED: {why}")
        return exit_codes.REFUSED
    Path(args.out).mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(Path(args.out) / "triton-cache")
    try:
        if args.mode == "regcheck":
            return regcheck(args, variants)
        return perturb(args, variants) if args.mode == "perturb" else stamps(args, variants[0])
    except REFUSALS as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    except Exception as exc:                                      # noqa: BLE001  (review MAJOR-1: never exit 1)
        print(f"CRASHED: {_err(exc)}")
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
