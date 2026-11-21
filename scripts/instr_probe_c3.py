#!/usr/bin/env python
"""THE INSTRUMENTED KERNEL AT vLLM'S 128x128 TILE: the calibration rental C3's probe (docs/registered/2026-10-26-cal3-gh200).

    python scripts/instr_probe_c3.py --mode regcheck --variants FILE --out DIR [--dry-run]
    python scripts/instr_probe_c3.py --mode perturb  --variants FILE --out DIR [--dry-run]
    python scripts/instr_probe_c3.py --mode stamps   --variants FILE --variant ID --gate FILE --regcheck FILE --out DIR

SCAFFOLDING, NOT A KERNEL. The kernel is moe/instrumented/fused_moe_instr.py, vLLM v0.27.1's fused_moe_kernel with the
owner-approved timestamps (rental 4, decision 5), unchanged; C3 runs it at the "ends" level only (STAMPS 3: the CTA's
start before the dead check, a dead CTA's exit, the end after the final store; nothing near the wgmma k-loop:
moe-kernels-work 2026-10-26 build-C3/CHECK.md). scripts/instr_probe.py (rental 4 and 5's probe, pinned) is
imported for its pure helpers and its burst timer and stays byte-identical; what it cannot do at C3's cells is here:

  the tile     vLLM's default 128x128x64 w8 s3 GROUP_SIZE_M 1 (instr_probe pins R3's BLOCK_M 32 and BLOCK_N 64 and
               takes num_warps from the sweep's FIXED); the config is handed to vLLM's override hook as every R3 page's
  the cells    A2's E1-CAL histogram cells (a registered histogram file), built through R3's own arm_inputs(cell=...)
               with the file's shuffle seed and R3's 9 declared copies, so a stamped cell is a cell of A2's timed page;
               a cell is (label, n, arm): A2's four Mixtral cells share n 2, so files are named by label
  the launch   each stamped launch also keeps the expert_ids and num_tokens_post_padded vLLM handed it (references held
               during the call, copied to the host after its synchronize, nothing added between the event pair) and
               its USE_TD flag, which must be False (the pointer path the copy instruments)
  the SASS     a wgmma-aware bracket check (sass_checks_c3): the k-loop is the back-branch spanning the most HGMMA (or
               HMMA), and the gate requires a clock read before the first LDG and none inside that loop
               (moe.instrumented.cubin.sass_checks finds the loop by HMMA only, blind at wgmma)

REGCHECK (compile only). Per variant, on its first cell: the plain kernel, the all-off copy and the variant's own spec,
per config the registers, shared and CTAs per SM (instr_probe.regcheck_rows); PASS when the all-off copy and the spec
keep the plain kernel's CTAs per SM (instr_probe.regcheck_ok, rental 5's rule) AND both the plain kernel and the copy
are on the wgmma path (C3). Writes DIR/regcheck.json and DIR/regcheck.env.

PERTURB (decision 2, rental 5's legs). Per variant, on EVERY cell, the plain kernel (instrumented.timed_plain) and the
stamped copy (instrumented.install) timed the same way, instr_probe's burst (an L2 flush before each call outside the
launch event pairs, the GPU held ahead by a sleep, REPS bursts of CALLS calls in alternating order); r = median plain
ms / median copy ms per (cell, GEMM); instr_probe.gate_verdict at PERTURB_TOL_R5 (median |r - 1| <= 1%, worst <= 2%,
CTAs per SM identical, the all-off copy's SASS equal to the plain kernel's, the upstream file), plus C3's SASS legs on
the stamped copy. The plain medians are the UNINSTRUMENTED REFERENCE the reader prices against. Writes
DIR/perturb.json (per (cell, GEMM) overhead_median = copy / plain - 1 and overhead_worst, the worst burst pair),
DIR/gate.env (GATE_<id>=PASS|FAIL median= worst=, and HARD_<id>=PASS|FAIL on every leg but timing) and the SASS.

STAMPS (owner OD-1, 2026-10-26; review MAJOR-2: each stamped call is held ahead by instr_probe's sleep, and each
launch records its stamps' own span, first live start to last live end). One variant whose HARD line (every gate leg but timing) AND regcheck line read PASS
(else REFUSED, exit 2, nothing launched); a FAIL on the timing legs alone runs the unit and labels its page PERTURBED
(stamps.json page_label; the per-cell overhead is in perturb.json, overhead_median and overhead_worst): per cell
STAMP_WARMUP calls, then STAMP_CALLS stamped calls, an L2 flush before each, every stamp buffer allocated before the
flush (moe.instrumented.BufferPool). Per launch: DIR/stamps/<label>-n<n>-<arm>-call<c>-<w1|w2>.npy (the int64 rows,
moe.instrumented.COLUMNS) and ...-experts.npy (expert_ids as vLLM passed them), indexed in DIR/stamps.json with the
launch record (grid, N, K, EM, config), num_tokens_post_padded, USE_TD and the launch's event-pair ms. Each compiled
config's SASS and its C3 bracket checks are kept under DIR/sass/.

EXIT CODES (instr_probe's): 0 done (every gate PASS for perturb); 1 a gate or regcheck FAILED; 2 refused (bad
variants, no CUDA / vLLM, the gate or regcheck not PASS; and --dry-run); 3 a cell failed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import instr_probe as IP  # noqa: E402
import private_weight_reference as PW  # noqa: E402

from moe import instrumented as I  # noqa: E402
from moe.bench import exit_codes  # noqa: E402
from moe.instrumented import cubin as CB  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

#: vLLM's default config at C3's cells (A2's E1-CAL 128 G 1): BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages
TILES = {(128, 128, 64, 8, 3)}
GROUP_M = 1
#: the copies R3's E1-CAL histogram pages declared (the C2 and A2 plans' declared-copies=9)
DECLARED_COPIES = 9
#: the one stamp level C3 runs: the CTA ends, nothing near the k-loop
LEVEL = I.STAMP_LEVELS["ends"]
STAMP_WARMUP = IP.STAMP_WARMUP
STAMP_CALLS = 3
TOL = IP.PERTURB_TOL_R5
ARMS_OK = (PW.NATIVE, PW.SHARED)
NEED = ("id", "kind", "model", "groups", "arms", "spec", "block_m", "block_n", "block_k", "num_warps",
        "num_stages", "histogram")


class Refused(Exception):
    pass


# --------------------------------------------------------------------------
# variants and cells (pure)
# --------------------------------------------------------------------------

def check_variant(v: dict) -> dict:
    """One C3 variant validated before anything runs: the spec is stamps=ends and nothing else, the tile is vLLM's
    128x128x64 w8 s3, G is 1, the arms are NATIVE and/or SHARED, and the histogram file parses under R3 for the
    model at that BLOCK_M. Returns the variant with its config, spec, page and cells [(label, n, arm)]."""
    miss = [k for k in NEED if k not in v]
    if miss:
        raise Refused(f"variant {v.get('id')!r} has no {miss}")
    if v["kind"] != "stamps":
        raise Refused(f"variant {v['id']}: kind {v['kind']!r}; C3 runs stamps units only")
    if v["model"] not in MODEL_CONFIGS:
        raise Refused(f"variant {v['id']}: unknown model {v['model']!r}")
    cfg = MODEL_CONFIGS[v["model"]]
    try:
        spec = I.parse_spec(v["spec"])
    except I.SpecError as exc:
        raise Refused(f"variant {v['id']}: {exc}") from None
    if spec.stamps != LEVEL or spec.hints:
        raise Refused(f"variant {v['id']}: C3 stamps the CTA ends only (stamps=ends, no hint), not {spec.text()}")
    tile = tuple(int(v[k]) for k in ("block_m", "block_n", "block_k", "num_warps", "num_stages"))
    if tile not in TILES:
        raise Refused(f"variant {v['id']}: tile {tile} is not one of C3's {sorted(TILES)}")
    groups = [int(g) for g in v["groups"]]
    if groups != [GROUP_M]:
        raise Refused(f"variant {v['id']}: groups {groups}; C3's cells are G {GROUP_M}")
    arms = [str(a) for a in v["arms"]]
    if not arms or set(arms) - set(ARMS_OK) or len(set(arms)) != len(arms):
        raise Refused(f"variant {v['id']}: arms {arms} are not distinct arms from {list(ARMS_OK)}")
    bm, bn, bk, nw, st = tile
    why = PW.config_refusal(cfg, block_m=bm, block_n=bn, block_k=bk, num_stages=st)
    if why:
        raise Refused(f"variant {v['id']}: the config cannot run: {why}")
    path = Path(v["histogram"])
    path = path if path.is_absolute() else REPO / path
    try:
        page = PW.load_histogram_page(path, cfg, bm, v["model"])
    except PW.HistogramRefused as exc:
        raise Refused(f"variant {v['id']}: {exc}") from None
    # each cell on the variant's arms among its own (a histogram cell names native and shared; C3 stamps NATIVE)
    cells = [(c.label, int(c.n), a) for c in page.ordered() for a in ARMS_OK if a in (c.arms or ARMS_OK) and a in arms]
    if not cells:
        raise Refused(f"variant {v['id']}: no cell of {v['histogram']} runs on {arms}")
    gate_spec = dataclasses.replace(spec, evict_a="none", evict_b="none")
    return {**v, "cfg": cfg, "spec_obj": spec, "gate_spec": gate_spec, "tile": tile, "groups": groups,
            "arms": arms, "page": page, "cells": cells, "gate_cells": cells}


def load_variants(path: Path) -> list[dict]:
    vs = json.loads(Path(path).read_text())
    if not isinstance(vs, list) or not vs:
        raise Refused(f"{path}: a JSON list of variants")
    ids = [v.get("id") for v in vs]
    if len(set(ids)) != len(ids):
        raise Refused(f"{path}: duplicate variant ids {ids}")
    return [check_variant(v) for v in vs]


def conf(v: dict) -> dict:
    """vLLM's config for the variant's launches: R3's pinned_config at the tile, BLOCK_M and num_warps put over it."""
    bm, bn, bk, nw, st = v["tile"]
    return dict(PW.pinned_config(bn, GROUP_M, st, **PW.block_k_kw(bk)), BLOCK_SIZE_M=bm, num_warps=nw)


def stem(label: str, n: int, arm: str) -> str:
    return f"{label}-n{n}-{arm}"


def plan_lines(variants: list[dict], mode: str) -> list[str]:
    out = [f"INSTRUMENTED KERNEL C3 ({mode}): {len(variants)} variant(s); upstream vLLM {I.UPSTREAM['tag']} fused_moe.py "
           f"sha256 {I.UPSTREAM['file_sha256'][:16]}"]
    for v in variants:
        bm, bn, bk, nw, st = v["tile"]
        out.append(f"  {v['id']:16s} {v['model']:22s} {bm}x{bn}x{bk} w{nw} s{st} G{GROUP_M} {v['page'].page or v['histogram']} "
                   f"cells {len(v['cells'])} ({', '.join(stem(*c) for c in v['cells'])}) spec {v['spec_obj'].text()}")
    if mode == "perturb":
        out.append(f"  gate (decision 2, rental 5's legs): median |plain/copy - 1| <= {100 * TOL['median_max']:.0f}%, "
                   f"worst <= {100 * TOL['worst_max']:.0f}%, CTAs per SM identical, all-off SASS equal, the upstream file; "
                   f"C3: a clock read before the first LDG and none inside the HGMMA loop; {IP.REPS} bursts of {IP.CALLS} "
                   "calls each way on every cell")
    if mode == "stamps":
        out.append(f"  {STAMP_WARMUP} warm-up and {STAMP_CALLS} stamped calls per cell, each held ahead; on a PASS regcheck "
                   "and gate hard legs (PERTURBED on a timing FAIL alone)")
    return out


# --------------------------------------------------------------------------
# SASS (pure)
# --------------------------------------------------------------------------

_BRA = re.compile(r"0x([0-9a-f]+)")


def sass_checks_c3(text: str | None) -> dict:
    """Where the stamps' clock reads sit in a compiled kernel's SASS, wgmma-aware. The k-loop is the back-branch
    (a BRA to a lower address) spanning the most HGMMA, or HMMA where there is no HGMMA; `loop_clock_reads` counts
    clock reads (CS2R / S2R / S2UR of SR_CLOCK* or SR_GLOBALTIMER*) inside it. `start_before_first_ldg`: a clock
    read precedes the first LDG (the num_tokens_post_padded load). `bars_after_loop`: BAR instructions past the
    loop (printed: whether a barrier can precede the end stamp). `ok`: start_before_first_ldg and a loop found
    with no clock read inside it."""
    out = {"instructions": 0, "mma": None, "clock_reads": 0, "start_before_first_ldg": None, "loop": None,
           "loop_mma": 0, "loop_clock_reads": None, "bars_after_loop": None, "ok": False}
    ins = CB.parse_sass(text or "")
    if not ins:
        return out
    out["instructions"] = len(ins)
    hg = [i["addr"] for i in ins if i["op"].startswith("HGMMA")]
    hm = [i["addr"] for i in ins if i["op"].startswith("HMMA")]
    mma = hg or hm
    out["mma"] = "HGMMA" if hg else ("HMMA" if hm else None)
    clk = [i["addr"] for i in ins if CB._is_clock(i)]
    ldg = [i["addr"] for i in ins if i["op"].startswith("LDG")]
    out["clock_reads"] = len(clk)
    out["start_before_first_ldg"] = bool(clk and ldg and min(clk) < min(ldg))
    loops = []
    for i in ins:
        m = _BRA.match(i["args"]) if i["op"].startswith("BRA") else None
        if m and int(m.group(1), 16) < i["addr"]:
            lo, hi = int(m.group(1), 16), i["addr"]
            loops.append((sum(lo <= a <= hi for a in mma), lo, hi))
    if loops:
        n, lo, hi = max(loops)
        if n > 0:
            out.update(loop=[lo, hi], loop_mma=n, loop_clock_reads=sum(lo <= a <= hi for a in clk),
                       bars_after_loop=sum(1 for i in ins if i["addr"] > hi and i["op"].startswith("BAR")))
    out["ok"] = bool(out["start_before_first_ldg"] and out["loop"] is not None and out["loop_clock_reads"] == 0)
    return out


def ptx_path(ptx: str) -> dict:
    """The MMA path of a compiled kernel's PTX (the tile census's reading): wgmma counts, else mma.sync."""
    return {"wgmma": ptx.count("wgmma.mma_async"), "wgmma_wait_group": ptx.count("wgmma.wait_group"),
            "mma_sync": ptx.count("mma.sync"),
            "mma": "wgmma" if "wgmma.mma_async" in ptx else ("mma.sync" if "mma.sync" in ptx else "?")}


def stamp_span(rows) -> int | None:
    """The stamps' own GPU-side span of a launch: the last live end minus the first live start (globaltimer ns)."""
    import numpy as np
    a = np.asarray(rows, dtype=np.int64)
    live = a[:, 9] == 1
    return int(a[live, 7].max() - a[live, 1].min()) if live.any() else None


def _op(x: str) -> str:
    w = x.split()
    return w[1] if w and w[0].startswith("@") and len(w) > 1 else (w[0] if w else "")


def loop_sass(text: str | None) -> list[str]:
    """The normalized instructions inside the k-loop sass_checks_c3 finds (empty when none)."""
    c = sass_checks_c3(text)
    if not c["loop"]:
        return []
    lo, hi = c["loop"]
    return [f"{i['pred']} {i['op']} {i['args']}".strip() for i in CB.parse_sass(text or "") if lo <= i["addr"] <= hi]


def loop_compare(plain_text: str | None, copy_text: str | None) -> dict:
    """Review MINOR-5 (printed, not a gate): the stamped copy's k-loop against the plain kernel's. `identical`: the
    normalized loop instructions equal; `same_ops`: the opcode sequence equal (registers may differ)."""
    a, b = loop_sass(plain_text), loop_sass(copy_text)
    if not a or not b:
        return {"identical": None, "same_ops": None, "plain_len": len(a), "copy_len": len(b)}
    return {"identical": a == b, "same_ops": [_op(x) for x in a] == [_op(x) for x in b],
            "plain_len": len(a), "copy_len": len(b)}


def regcheck_c3(rows: list[dict], own: str, paths: dict) -> tuple[bool, str]:
    """instr_probe.regcheck_ok (CTAs per SM of the all-off copy and the own spec equal the plain kernel's) and C3's
    path leg: every config of the plain kernel and of the stamped copy on the wgmma path."""
    ok, why = IP.regcheck_ok(rows, own)
    bad = [f"{who} {IP.key_name(tuple(k))} on the {p['mma']} path" for who, d in paths.items()
           for k, p in d.items() if p["mma"] != "wgmma"]
    if not any(paths.values()):
        bad.append("no PTX read")
    why = "; ".join([w for w in [why] if w] + bad)
    return ok and not bad, why


def gate_c3(ratios: list, configs: list, *, upstream_ok: bool, sass_equal, brackets: dict) -> dict:
    """instr_probe.gate_verdict at rental 5's tolerance and occupancy legs, with C3's SASS legs added: every stamped
    config's sass_checks_c3 ok (None, unread, FAILS). Split for the owner's OD-1 (2026-10-26): `hard_why` is every
    leg but timing (occupancy, the all-off SASS, the upstream file, C3's SASS legs), `timing_why` the median and worst
    legs. `verdict` FAILs on either; `hard_verdict` on the hard legs alone. A C3 stamps unit runs on a hard PASS and
    its page is PERTURBED when the timing legs fail."""
    v = IP.gate_verdict(ratios, configs, upstream_ok=upstream_ok, sass_equal=sass_equal, hint_ok=None, tol=TOL)
    hard = list(IP.gate_verdict([1.0], configs, upstream_ok=upstream_ok, sass_equal=sass_equal, hint_ok=None,
                                tol=TOL)["why"])
    if not brackets:
        hard.append("C3 SASS legs unread (no stamped cubin, or no cuobjdump)")
    for k, b in sorted(brackets.items()):
        if not b.get("ok"):
            hard.append(f"{k}: start_before_first_ldg {b.get('start_before_first_ldg')}, loop {b.get('loop')}, "
                        f"loop_clock_reads {b.get('loop_clock_reads')}")
    timing = [w for w in v["why"] if w not in hard]
    why = timing + hard
    return {**v, "why": why, "verdict": "FAIL" if why else "PASS", "timing_why": timing, "hard_why": hard,
            "hard_verdict": "FAIL" if hard else "PASS"}


def overhead(plain_ms: float, copy_ms: float, plain_reps: list, copy_reps: list) -> dict:
    """The stamped copy's cost on one (cell, GEMM): median copy / median plain - 1, and the worst burst pair (the
    largest in absolute value)."""
    pairs = [c / p - 1 for p, c in zip(plain_reps, copy_reps) if p]
    # review MINOR-1: two-sided; the worst pair is the largest |copy / plain - 1|, sign kept
    return {"overhead_median": (copy_ms / plain_ms - 1) if plain_ms else None,
            "overhead_worst": max(pairs, key=abs) if pairs else None}


def read_env(path, prefix: str, vid: str) -> str:
    """The verdict word of `<prefix>_<vid>=WORD ...` in an env file (ids as instr_probe.gate_key), or ''."""
    if path is None or not Path(path).exists():
        return ""
    for line in Path(path).read_text().splitlines():
        k, _, val = line.partition("=")
        if k == f"{prefix}_{IP.gate_key(vid)}":
            return val.split()[0] if val.split() else ""
    return ""


# --------------------------------------------------------------------------
# the card
# --------------------------------------------------------------------------

class Keep:
    """Stands in for the copy inside instrumented.install: `keep[grid](*args, **kw)` launches the copy and keeps
    references to the expert_ids and num_tokens_post_padded tensors vLLM passed (positional 8 and 9 of
    fused_moe_kernel) and its USE_TD keyword. Nothing runs on the GPU between the launch's event pair but the launch."""

    def __init__(self, kernel, sink: list):
        self.kernel, self.sink = kernel, sink

    def __getitem__(self, grid):
        def launch(*args, **kw):
            out = self.kernel[grid](*args, **kw)
            self.sink.append({"expert_ids": args[8], "num_tokens_post_padded": args[9],
                              "use_td": bool(kw.get("USE_TD", False))})
            return out
        return launch


class Card(IP.Card):
    """instr_probe's box (vLLM, the override hook, the burst timer, the flush) with C3's weights and cells."""

    def load(self, model: str):
        if self.model == model:
            return
        torch = self.torch
        self.w1 = self.w2 = None
        torch.cuda.empty_cache()
        cfg = MODEL_CONFIGS[model]
        self.copies = DECLARED_COPIES
        self.w1, self.w2, _ = PW.build_private_weights(cfg, "bf16", self.copies, self.args.seed)
        self.nw1, self.nw2 = self.w1[::self.copies], self.w2[::self.copies]
        self.declared = {a: PW.declared_experts(a, cfg.num_experts, self.copies) for a in PW.ARMS}
        self.cfg, self.model, self.inputs = cfg, model, {}

    def call(self, v: dict, label: str, n: int, arm: str):
        key = (v["id"], label, n)
        if key not in self.inputs:
            page = v["page"]
            cell = next(c for c in page.cells if c.label == label and int(c.n) == n)
            self.inputs[key] = PW.arm_inputs(self.cfg, n, v["tile"][0], self.copies, self.args.seed, "bf16",
                                             self.w1.dtype, arms=tuple(v["arms"]), cell=cell,
                                             shuffle_seed=page.shuffle_seed)
        _tokens, x, ids_by_arm, weights, kw = self.inputs[key]
        return PW.arm_call(self.fused_experts, arm, self.w1, self.w2, self.nw1, self.nw2, self.declared, x,
                           ids_by_arm, weights, kw)


def _facts(launches, into):
    return IP.facts_by_config(launches, into)


def regcheck(args, variants: list[dict]) -> int:
    card = Card(args)
    report = {"tool": "scripts/instr_probe_c3.py --mode regcheck", "versions": PW._package_versions(), "variants": {}}
    worst, env = exit_codes.DONE, []
    for v in variants:
        card.load(v["model"])
        label, n, arm = v["cells"][0]
        fn, cf = card.call(v, label, n, arm), conf(v)
        plain_f = _facts(card.compile_once(fn, cf, None), {})
        off_f = _facts(card.compile_once(fn, cf, I.InstrSpec()), {})
        own = v["spec_obj"].text()
        own_f = _facts(card.compile_once(fn, cf, v["spec_obj"]), {})
        rows = IP.regcheck_rows(plain_f, off_f, {own: own_f})
        paths = {"plain": {k: ptx_path(f.get("ptx") or "") for k, f in plain_f.items()},
                 "copy": {k: ptx_path(f.get("ptx") or "") for k, f in own_f.items()}}
        ok, why = regcheck_c3(rows, own, paths)
        brackets = {}
        for k, f in own_f.items():
            t = IP.sass_of(f.get("cubin"), Path(args.out) / "sass" / IP.gate_key(v["id"]) / f"copy-{IP.key_name(k)}.sass")
            brackets[IP.key_name(k)] = sass_checks_c3(t) if t else None
        report["variants"][v["id"]] = {"cell": [label, n, arm], "own_spec": own, "rows": rows, "own_spec_ok": ok,
                                       "why": why, "paths": {w: {IP.key_name(k): p for k, p in d.items()}
                                                             for w, d in paths.items()},
                                       "sass_checks_c3_printed": brackets,
                                       "rule": "CTAs per SM identical to the plain kernel's; plain and copy on wgmma"}
        env.append(IP.regcheck_line(v["id"], ok, why))
        for r in rows:
            x = r["specs"][own]["copy"] or {}
            print(f"  {v['id']} {IP.key_name(tuple(r['config']))}: plain {r['plain']['regs']}r/{r['plain']['ctas_per_sm']} "
                  f"off {'=' if r['off_same'] else '!='} | {own}: {x.get('regs')}r/{x.get('ctas_per_sm')}", flush=True)
        if not ok:
            worst = exit_codes.CLAIM_FAIL
    out = Path(args.out)
    (out / "regcheck.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    (out / "regcheck.env").write_text("\n".join(env) + "\n")
    return worst


def perturb(args, variants: list[dict]) -> int:
    card = Card(args)
    torch = card.torch
    report = {"tool": "scripts/instr_probe_c3.py --mode perturb", "tolerance": TOL, "reps": IP.REPS,
              "calls": IP.CALLS, "upstream": card.upstream, "override_hook": card.where,
              "triton_cache": os.environ.get("TRITON_CACHE_DIR"), "versions": PW._package_versions(),
              "device": {"name": torch.cuda.get_device_name(0), "uuid": PW.device_identity()},
              "smi_before": IP._smi(), "variants": {}, "started": time.time()}
    upstream_ok = bool(card.upstream["file_matches"] and card.upstream["excerpt_matches"])
    env, worst = [], exit_codes.DONE
    for v in variants:
        card.load(v["model"])
        cf = conf(v)
        rows, plain_f, copy_f, off_f = [], {}, {}, {}
        for label, n, arm in v["gate_cells"]:
            fn = card.call(v, label, n, arm)
            per = {"plain": {"w1": [], "w2": []}, "copy": {"w1": [], "w2": []}}
            for rep in range(IP.REPS):
                order = ("plain", "copy") if rep % 2 == 0 else ("copy", "plain")
                for which in order:
                    gspec = v["gate_spec"]
                    make = ((lambda ev, pool: I.timed_plain(module=card.FM, events=ev)) if which == "plain"
                            else (lambda ev, pool, gs=gspec: I.install(gs, module=card.FM, events=ev, alloc=pool)))
                    ms, launches = card.burst(fn, cf, make)
                    _facts(launches, plain_f if which == "plain" else copy_f)
                    ms = IP.split_gemms(ms)
                    for gemm in ("w1", "w2"):
                        per[which][gemm].append(statistics.median(ms[gemm]))
            for gemm in ("w1", "w2"):
                p, c = statistics.median(per["plain"][gemm]), statistics.median(per["copy"][gemm])
                rows.append({"label": label, "n": n, "arm": arm, "gemm": gemm, "plain_ms": p, "copy_ms": c,
                             "ratio": p / c if c else None, "plain_reps": per["plain"][gemm],
                             "copy_reps": per["copy"][gemm],
                             **overhead(p, c, per["plain"][gemm], per["copy"][gemm])})
            _facts(card.compile_once(fn, cf, I.InstrSpec()), off_f)
        configs = IP.compare_configs(plain_f, copy_f, IP.OCC_LEGS_R5)
        out_dir = Path(args.out) / "sass" / IP.gate_key(v["id"])
        sass_ok, sass_rows = IP.sass_equal_leg(plain_f, off_f, out_dir)
        brackets, loops = {}, {}
        for key, f in copy_f.items():
            t = IP.sass_of(f.get("cubin"), out_dir / f"copy-{IP.key_name(key)}.sass")
            if t:
                brackets[IP.key_name(key)] = sass_checks_c3(t)
                loops[IP.key_name(key)] = loop_compare(IP.sass_of((plain_f.get(key) or {}).get("cubin")), t)
        verdict = gate_c3([r["ratio"] for r in rows], configs, upstream_ok=upstream_ok, sass_equal=sass_ok,
                          brackets=brackets)
        report["variants"][v["id"]] = {"kind": v["kind"], "model": v["model"], "tile": list(v["tile"]),
                                       "histogram": v["histogram"], "gate_spec": v["gate_spec"].as_dict(),
                                       "cells": rows, "configs": configs, "sass_equal": sass_ok,
                                       "sass_rows": sass_rows, "sass_checks_c3": brackets,
                                       "loop_sass_vs_plain_printed": loops, **verdict}
        env.append(f"GATE_{IP.gate_key(v['id'])}={verdict['verdict']} median={verdict['median_dev']} "
                   f"worst={verdict['worst_dev']}")
        env.append(f"HARD_{IP.gate_key(v['id'])}={verdict['hard_verdict']}")
        if verdict["verdict"] != "PASS":
            worst = exit_codes.CLAIM_FAIL
        print(f"  {v['id']}: {verdict['verdict']} {'; '.join(verdict['why'])}", flush=True)
    report["smi_after"] = IP._smi()
    report["ended"] = time.time()
    out = Path(args.out)
    (out / "perturb.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    (out / "gate.env").write_text("\n".join(env) + "\n")
    return worst


def stamps(args, v: dict) -> int:
    import numpy as np

    from moe.instrumented import fused_moe_instr
    card = Card(args)
    torch = card.torch
    out = Path(args.out)
    (out / "stamps").mkdir(parents=True, exist_ok=True)
    card.load(v["model"])
    fl = card.flusher()
    cf = conf(v)
    index, bad, compiled = [], 0, {}
    for label, n, arm in v["cells"]:
        fn = card.call(v, label, n, arm)
        try:
            pool = I.BufferPool()
            like = torch.empty(0, device="cuda")
            with card.override_config(cf):
                with I.install(v["spec_obj"], module=card.FM) as warm:
                    for _ in range(STAMP_WARMUP):
                        fn()
                torch.cuda.synchronize()
                shapes = I.BufferPool.shapes(warm.launches[-PW.GEMMS_PER_CALL:])
                for c in range(STAMP_CALLS):
                    pool.reserve(shapes, like)      # the buffers before the flush (rental 5 item 4)
                    card.sleep_ahead()              # review MAJOR-2: the GPU held ahead, as instr_probe's burst does
                    fl.flush()                      # outside the stamped launches, as R3 flushes
                    sink: list = []
                    with I.install(v["spec_obj"], module=card.FM, kernel=Keep(fused_moe_instr.fused_moe_kernel, sink),
                                   alloc=pool, events=IP.launch_event_factory(torch)) as rec:
                        fn()
                        torch.cuda.synchronize()
                    _facts(rec.launches, compiled)
                    ms_each = IP.launch_ms(rec.launches)
                    for lr, kept, gemm, lms in zip(rec.launches, sink, ("w1", "w2"), ms_each, strict=True):
                        if kept["use_td"]:
                            raise Refused("the launch took the tensor-descriptor path (USE_TD True): not the copy's "
                                          "pointer path")
                        name = f"{stem(label, n, arm)}-call{c}-{gemm}"
                        rows = lr["buffer"].cpu().numpy()
                        np.save(out / "stamps" / f"{name}.npy", rows)
                        np.save(out / "stamps" / f"{name}-experts.npy", kept["expert_ids"].cpu().numpy())
                        ntpp = int(kept["num_tokens_post_padded"].reshape(-1)[0].item())
                        index.append({k: lr[k] for k in lr if k not in ("buffer", "events", "compiled")}
                                     | {"label": label, "n": n, "arm": arm, "call": c, "gemm": gemm,
                                        "file": f"{name}.npy", "experts_file": f"{name}-experts.npy",
                                        "num_tokens_post_padded": ntpp, "use_td": kept["use_td"],
                                        "stamp_span_ns": stamp_span(rows),
                                        "launch_ms": lms, "late_buffers": pool.late})
        except Exception as exc:                                  # noqa: BLE001
            index.append({"label": label, "n": n, "arm": arm, "error": f"{type(exc).__name__}: {exc}"})
            bad += 1
    sass = []
    for key, f in compiled.items():
        name = f"sass/{IP.key_name(key)}.sass"
        text = IP.sass_of(f.get("cubin"), out / name)
        sass.append({"config": list(key), "file": name if text else None,
                     "checks_c3": sass_checks_c3(text) if text else None, "ptx": ptx_path(f.get("ptx") or ""),
                     "regs": f["regs"], "shared": f["shared"],
                     "ctas_per_sm": (f.get("occupancy") or {}).get("ctas_per_sm")})
    man = {"tool": "scripts/instr_probe_c3.py --mode stamps", "variant": v["id"], "model": v["model"],
           "page_label": getattr(args, "page_label", None),
           "histogram": v["histogram"], "histogram_sha256": v["page"].file_sha256, "tile": list(v["tile"]),
           "group_m": GROUP_M, "copies_declared": card.copies, "spec_text": v["spec_obj"].text(),
           "spec": v["spec_obj"].as_dict(), "columns": list(I.COLUMNS), "hdr": I.HDR, "upstream": card.upstream,
           "warmup": STAMP_WARMUP, "calls": STAMP_CALLS, "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
           "l2_flush_mb": fl.megabytes, "versions": PW._package_versions(),
           "device": {"name": torch.cuda.get_device_name(0), "uuid": PW.device_identity()},
           "smi": IP._smi(), "sass": sass, "launches": index}
    (out / "stamps.json").write_text(json.dumps(man, indent=1, sort_keys=True, default=str))
    print(f"STAMPS {v['id']}: {len(index)} launches, {bad} failed; {out}")
    return exit_codes.INVALID if bad else exit_codes.DONE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("perturb", "stamps", "regcheck"), required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--variant", default=None, help="stamps: the variant id to run")
    p.add_argument("--gate", type=Path, default=None, help="stamps: the perturb unit's gate.env")
    p.add_argument("--regcheck", type=Path, default=None, help="stamps: the regcheck unit's regcheck.env")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        variants = load_variants(args.variants)
        if args.mode == "stamps":
            mine = [v for v in variants if v["id"] == args.variant]
            if len(mine) != 1:
                raise Refused(f"--variant {args.variant!r} is not one of {[v['id'] for v in variants]}")
            variants = mine
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    print("\n".join(plan_lines(variants, args.mode)))
    if args.mode == "stamps":
        g = read_env(args.gate, "GATE", args.variant)
        h = read_env(args.gate, "HARD", args.variant)
        if h != "PASS" or not g:
            print(f"REFUSED: the perturbation gate's hard legs for {args.variant} read {h or 'nothing'} ({args.gate}): "
                  "a C3 stamps unit runs only when occupancy, the all-off SASS, the upstream file and C3's SASS legs pass")
            return exit_codes.REFUSED
        args.page_label = "CLEAN" if g == "PASS" else "PERTURBED"
        if g != "PASS":
            print(f"PERTURBED: the overhead gate for {args.variant} reads {g}; the stamps run and the pages are labelled "
                  "PERTURBED, the measured overhead beside every reading (owner OD-1, 2026-10-26)")
        r = IP.read_regcheck(args.regcheck, args.variant) if args.regcheck else ""
        if r != "PASS":
            print(f"REFUSED: the regcheck line for {args.variant} reads {r or 'nothing'} ({args.regcheck}): a C3 "
                  "stamps unit runs only on its regcheck PASS")
            return exit_codes.REFUSED
    if args.dry_run:
        print("DRY RUN: nothing launched (exit 2)")
        return exit_codes.REFUSED
    why = IP.run_refusal()
    if why:
        print(f"REFUSED: {why}")
        return exit_codes.REFUSED
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # a cache of this unit's own (build-r4-review G1); the driver sets the same directory
    os.environ["TRITON_CACHE_DIR"] = str(Path(args.out) / "triton-cache")
    try:
        if args.mode == "regcheck":
            return regcheck(args, variants)
        return perturb(args, variants) if args.mode == "perturb" else stamps(args, variants[0])
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED


if __name__ == "__main__":
    sys.exit(main())
