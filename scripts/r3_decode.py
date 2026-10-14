#!/usr/bin/env python
"""THE DECODE STAGE'S PAGE: vLLM's own decode call at T = 1..128 tokens, timed inside a CUDA graph
(design-decode DESIGN.md; owner decisions DD1 to DD6 of 2026-10-14). R3 stays byte-identical: this
is a new module that imports R3's pieces and writes a page R3's readers can read.

    python scripts/r3_decode.py --model olmoe-1b-7b --histogram PAGE.json --inst GRR --dry-run
    python scripts/locked_r3.py --session-tag T --groups 1 --locks 1710 --slip-policy cell \\
        --min-clean-repeats 0 --r3-script scripts/r3_decode.py -- --model olmoe-1b-7b \\
        --histogram PAGE.json --inst GRR --repeats 9 --duty 0.25 --seed 0
    python scripts/r3_decode.py --counter-child PLAN.json      # under ncu (dram_counter_route_decode.py)

WHAT IS TIMED. vLLM's `fused_experts` exactly as vLLM calls it at decode: NATIVE (E experts declared,
copy j of the weights through the strided view `w1[j::C]` of R3's `build_private_weights`), vLLM's
own tile (no `override_config`: `get_default_config` picks it, and GD5 reads it back off the
kernel), and vLLM's own assignment path, the naive one included (T k 4 <= E, which R3 refuses).
Claude wrote no kernel, tiling or scheduling logic: this file builds inputs (routing, token counts,
weights, arms) and times and counts the existing kernel.

THE PAGE (moe-kernels/decode-page/1, written by scripts/decode_synth.py): cells (label, T, counts)
with their arms (native, and the optional DECL lever: NATIVE's ids on copy j, declaring E x C) and
a shuffle flag. A cell's ids are `realize_counts(counts, T, k)`, then the page's token-row shuffle
unless the cell says shuffle false (G3-D's unshuffled twin). V-BIN refuses unless bincount(ids) is
the counts exactly, k distinct experts per token.

THE INSTRUMENTS (--inst, one per page; DD4):
  GRR   primary. One CUDA graph holding `calls` back-to-back calls; call j reads weight copy
        j mod R, with R chosen so R x A_min x W_e >= 240 MiB (A_min the page's least coverage), so
        every call is cold with no flush kernel, as vLLM's decode graph runs layer after layer.
        Captured by launch_floor.graph_call (imported), timed by launch_floor.mirror_time_cell
        (R3's time_cell at duty < 1, imported) with no flush; per-call time = replay / calls.
  GR1   control. Rentals 1 and 2's GR mode: a one-call graph, the 240 MiB flush outside it.
  E240  control. The eager call with the 240 MiB flush (R3's own instrument), for GD3.

BEFORE ANY CELL IS TIMED (untimed, per cell and arm):
  V-BIN    bincount(ids) == counts, k distinct per token (`feasible`, `realize_counts`, imported).
  V-ALIGN  a spy on vLLM's `dispatch_fused_moe_kernel` and `fused_moe_kernel` records one eager
           call's real kernel inputs: sorted_token_ids is None iff the path is naive;
           num_tokens_post_padded == sum ceil(c_e / BM) BM (naive numel BM); the sentinel count
           (== numel) in sorted_ids[:post] is P; expert_ids[:L] are the active experts ascending,
           each ceil(c_e / BM) times (naive: topk_ids flat); len(sorted_ids) is the capped
           max_num_tokens_padded; the launch grid from EM is `fused_moe_grid`'s; USE_TD False.
  GD5      the config the kernel received is `tile_resolve.default_config(T, E)` and vLLM finds
           no tuned file (`get_moe_configs` empty) and no override (`get_config()` None).
  GD4      the graph's output equals one eager call's output bit for bit (the capture took the
           eager path), for every call of a GRR graph. It covers the CHECK-PHASE capture only: each
           timed repeat recaptures the same calls on the same tensors and is not compared again
           (registered).
  COPIES   (GRR) calls 0..R-1, each spied, read R distinct w1 copies (check_copies); at R = 1 the
           call is cold by volume, stated in the registration.
  GD1      GPU time per replay >= 3 x the host enqueue per replay (launch_floor's host probe on
           the replay body, before the first timed cell and after the last); a cell failing it
           is excluded (G2), not voided.
A V-ALIGN, V-BIN or GD5 mismatch REFUSES the cell (status refused, never timed); the readback is in
cells.csv (post_padded, padded_rows, live_tiles, dead_ctas_w1/w2, path).

THE PAGE IS R3'S SHAPE where R3's readers look: the plan prints R3's `session` and `WRITES TO`
lines, the run directory is <results>/private_weight_reference/<run id>, cells.csv carries
`sm_clock_load_mhz` per repeat and `detail` "histogram=<label>", and the repeat loop is R3's
`histogram_schedule` (slips, lost cells, the early stop). report.json's `treads_table` rows are
(histogram label, tiles = T, arm), so rental 7's r7-median-run3 rebuild reads a decode page as it
reads a histogram page. --group-m is locked_r3's and must be 1 (vLLM's GROUP_SIZE_M at decode).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import private_weight_reference as PWR  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

SWEEP = PWR.SWEEP
SCHEMA = "moe-kernels/decode-page/1"
NATIVE, DECL = "native", "decl"
ARMS = (NATIVE, DECL)
INSTRUMENTS = ("GRR", "GR1", "E240")
NAIVE = "naive"
SMALL_BATCH, BLOCK_SCAN = PWR.SMALL_BATCH, PWR.BLOCK_SCAN
#: vLLM's naive-assignment factor (UPSTREAM_DECODE rule NAIVE), R3's own constant
NAIVE_FACTOR = PWR.NAIVE_ASSIGNMENT_SPARSITY
#: GRR: every call cold, R x A_min x W_e >= this many MiB of weights between two reads of one copy
GRR_COLD_MIB = 240
#: GRR: at least this many calls per graph (the graph's head amortised); a multiple of R
GRR_MIN_CALLS = 8
#: GD1: GPU time per replay against the host enqueue per replay
GD1_FACTOR = 3.0
DUTY = 0.25
WARMUP_MS = 100.0
CELL_BUDGET_MS = 100.0
TRIALS = 3
#: seconds per (cell, arm, repeat) at the defaults, and per cell-arm of untimed checks
SEC_PER_CELL = 1.6
SEC_PER_CHECK = 1.0
SETUP_MIN = 2.0
UPSTREAM = REPO / "moe" / "instrumented" / "UPSTREAM_DECODE.json"
LABEL_RE = PWR.HIST_LABEL_RE
GEMMS = ("w1", "w2")


class DecodeRefused(PWR.PrivateWeightRefusal):
    """A decode page or cell this tool will not run."""


# --------------------------------------------------------------------------
# vLLM's decode geometry, from the cited v0.27.1 rules (UPSTREAM_DECODE.json). Pure.
# --------------------------------------------------------------------------

def default_config(T: int, E: int) -> dict:
    """vLLM's general-branch default config at M = T (moe.bench.tile_resolve, verbatim v0.27.1)."""
    from moe.bench.tile_resolve import default_config as dc
    return dc(int(T), int(E))


def tile_key(conf: dict) -> str:
    return (f"{conf['BLOCK_SIZE_M']}x{conf['BLOCK_SIZE_N']}x{conf['BLOCK_SIZE_K']}"
            f"w{conf['num_warps']}s{conf['num_stages']}")


def decode_path(T: int, top_k: int, declared: int) -> str:
    """vLLM's assignment path: naive when T k 4 <= E declared (rule NAIVE), else the align kernel
    R3 cites (small-batch at numel < 1024 and E <= 64, rule SMALL_BATCH)."""
    numel = int(T) * int(top_k)
    if numel * NAIVE_FACTOR <= int(declared):
        return NAIVE
    return PWR.align_path(numel, int(declared))


def max_padded(numel: int, declared: int, bm: int) -> int:
    """moe_align_block_size's sorted-id buffer: numel + E (BM - 1), capped at numel BM when
    numel < E (rules MAXPAD, MAXPAD_CAP)."""
    m = numel + declared * (bm - 1)
    if numel < declared:
        m = min(numel * bm, m)
    return m


@dataclass(frozen=True)
class Geometry:
    """One (T, counts, declaration)'s decode call as vLLM forms it. Per-GEMM tuples are (w1, w2)."""
    path: str
    T: int
    numel: int
    declared: int
    tile: str
    bm: int
    bn: int
    A: int
    L: int
    P: int
    multi_tile_experts: int
    sorted_len: int | None
    post_padded: int
    em: tuple
    npn: tuple
    grid: tuple
    live_ctas: tuple
    dead_ctas: tuple

    def as_dict(self) -> dict:
        return asdict(self)


def geometry(cfg, T: int, counts, declared: int | None = None, conf: dict | None = None) -> Geometry:
    """The decode call's geometry. `declared` is E (NATIVE) or E x C (DECL); `conf` vLLM's config
    (default_config(T, E) unless given). Live tiles L = sum ceil(c_e / BM) on the aligned path;
    the grid from EM by R3's `fused_moe_grid` (the EM cut is in it); dead CTAs = (grid M-tiles - L)
    x N-tiles. Naive: one M-tile per (token, expert) row, every CTA live."""
    E, k = cfg.num_experts, cfg.top_k
    D = E if declared is None else int(declared)
    conf = conf or default_config(T, E)
    bm, bn = int(conf["BLOCK_SIZE_M"]), int(conf["BLOCK_SIZE_N"])
    c = [int(v) for v in counts]
    numel = int(T) * k
    if sum(c) != numel:
        raise DecodeRefused(f"counts sum to {sum(c)}, T k is {numel}")
    path = decode_path(T, k, D)
    A = sum(1 for v in c if v > 0)
    multi = sum(1 for v in c if v > bm)
    N = {"w1": 2 * cfg.intermediate_size, "w2": cfg.hidden_size}
    npn = tuple(-(-N[g] // bn) for g in GEMMS)
    if path == NAIVE:
        L, post, sorted_len = numel, numel * bm, None
        em = (numel * bm, numel * bm)
        mt = (numel, numel)
    else:
        L = sum(-(-v // bm) for v in c if v)
        post, sorted_len = L * bm, max_padded(numel, D, bm)
        grids = PWR.expected_grids(cfg, em=sorted_len, tokens=int(T), block_m=bm, block_n=bn)
        mt = tuple(grids[g] // npn[i] for i, g in enumerate(GEMMS))
        em = (min(sorted_len, T * k * bm) if T < bm else sorted_len,
              min(sorted_len, numel * bm) if numel < bm else sorted_len)
    grid = tuple(mt[i] * npn[i] for i in range(2))
    live = tuple(L * npn[i] for i in range(2))
    dead = tuple((mt[i] - L) * npn[i] for i in range(2))
    return Geometry(path=path, T=int(T), numel=numel, declared=D, tile=tile_key(conf), bm=bm, bn=bn,
                    A=A, L=L, P=post - numel, multi_tile_experts=multi, sorted_len=sorted_len,
                    post_padded=post, em=em, npn=npn, grid=grid, live_ctas=live, dead_ctas=dead)


def expert_order(counts, bm: int, copies: int = 1) -> list[int]:
    """expert_ids[:L] on the aligned path: the active experts ascending, each ceil(c_e / BM) times,
    in the declared slot space (copy 0 of expert e is slot e x copies, R3's expert-first layout)."""
    return [e * copies for e, v in enumerate(counts) for _ in range(-(-int(v) // bm)) if v]


def expected_readback(cfg, T: int, counts, ids_flat, *, declared: int, copies: int, conf: dict | None = None) -> dict:
    """What the spy must read back for this cell and arm (V-ALIGN, GD5)."""
    g = geometry(cfg, T, counts, declared, conf)
    conf = conf or default_config(T, cfg.num_experts)
    if g.path == NAIVE:
        order = [int(x) for x in ids_flat]
    else:
        order = expert_order(counts, g.bm, copies)
    return {"path": g.path, "post_padded": g.post_padded, "padded_rows": g.P, "live_tiles": g.L,
            "sorted_len": g.sorted_len, "expert_order": order, "grid": list(g.grid),
            "dead_ctas": list(g.dead_ctas), "config": {k: int(v) for k, v in conf.items()},
            "numel": g.numel}


#: the config keys GD5 compares (SPLIT_K is vLLM's own constant 1, set after the lookup)
CONFIG_KEYS = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages")


def check_copies(rec: dict, R: int) -> list[str]:
    """GRR's coldness by rotation: the w1 data pointers vLLM received on calls 0..R-1 are R distinct
    copies (a wrong stride would read one copy R times, warm, and GD4 could not see it: the copies are
    bitwise identical). At R = 1 the call is cold BY VOLUME alone (one copy of A W_e >= 240 MiB between
    two reads of it), which the registration states with each model's reuse distance."""
    ptrs = list(rec.get("copy_ptrs") or [])
    if len(ptrs) != R or None in ptrs or len(set(ptrs)) != R:
        return [f"GRR copies: calls 0..{R - 1} read w1 at {len(set(p for p in ptrs if p is not None))} distinct "
                f"addresses, not {R}"]
    return []


def check_readback(rec: dict, exp: dict) -> list[str]:
    """V-ALIGN and GD5 on one eager call's spy record (`Spy.records` of one call: two GEMMs).
    Returns every mismatch; empty is a pass."""
    bad = []
    gem = rec.get("gemms") or []
    if len(gem) != 2:
        return [f"the spy saw {len(gem)} fused_moe_kernel dispatches, not 2"]
    numel = exp["numel"]
    for i, (g, r) in enumerate(zip(GEMMS, gem)):
        naive = r.get("sorted") is None
        if naive != (exp["path"] == NAIVE):
            bad.append(f"{g}: sorted_token_ids is {'None' if naive else 'a tensor'} on a {exp['path']} cell")
            continue
        if int(r.get("post_padded", -1)) != exp["post_padded"]:
            bad.append(f"{g}: num_tokens_post_padded {r.get('post_padded')} != {exp['post_padded']}")
        eids = list(r.get("expert_ids") or [])
        L = exp["live_tiles"]
        if eids[:L] != exp["expert_order"]:
            bad.append(f"{g}: expert_ids[:{L}] {eids[:L][:12]}... != {exp['expert_order'][:12]}...")
        if not naive:
            srt = list(r["sorted"])
            if len(srt) != exp["sorted_len"]:
                bad.append(f"{g}: len(sorted_ids) {len(srt)} != {exp['sorted_len']}")
            sent = sum(1 for x in srt[:exp["post_padded"]] if int(x) == numel)
            if sent != exp["padded_rows"]:
                bad.append(f"{g}: {sent} sentinel ids in sorted_ids[:post], P is {exp['padded_rows']}")
        if int(r.get("grid", -1)) != exp["grid"][i]:
            bad.append(f"{g}: launch grid {r.get('grid')} != fused_moe_grid {exp['grid'][i]}")
        if r.get("use_td"):
            bad.append(f"{g}: the kernel took the tensor-descriptor path (USE_TD True)")
        got = {k: int(r["config"][k]) for k in CONFIG_KEYS if k in (r.get("config") or {})}
        want = {k: exp["config"][k] for k in CONFIG_KEYS}
        if got != want:
            bad.append(f"{g}: GD5 the kernel received {got}, default_config is {want}")
    if rec.get("tuned") not in (None, False, "none"):
        bad.append(f"GD5: vLLM found a tuned config ({rec.get('tuned')})")
    if rec.get("override") not in (None, False, "none"):
        bad.append(f"GD5: an override config is in force ({rec.get('override')})")
    return bad


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DecodeCell:
    label: str
    T: int
    counts: tuple
    arms: tuple
    shuffle: bool = True
    role: str = ""

    @property
    def key(self) -> tuple:
        return (self.label, self.T)


@dataclass(frozen=True)
class DecodePage:
    model: str
    page: str
    shuffle_seed: int
    decl_copies: int | None
    cells: tuple
    file_sha256: str
    label: str = ""
    schema: str = SCHEMA

    @property
    def Ts(self) -> list[int]:
        return sorted({c.T for c in self.cells})

    def ordered(self) -> list:
        """T ascending, then the file's order (R3's `HistogramPage.ordered`)."""
        return sorted(self.cells, key=lambda c: (c.T, self.cells.index(c)))


def counts_sha256(counts) -> str:
    return PWR.counts_sha256(counts)


def page_from(doc: dict, cfg, model: str, file_sha256: str = "") -> DecodePage:
    """A decode page checked against vLLM's decode call; every refusal is one the card would meet."""
    from moe.routing.distributions import feasible
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise DecodeRefused(f"the page's schema is {doc.get('schema') if isinstance(doc, dict) else '?'}, not {SCHEMA}")
    if doc.get("model") != model:
        raise DecodeRefused(f"the page is {doc.get('model')!r}'s, this run is {model!r}")
    seed = doc.get("shuffle_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise DecodeRefused(f"shuffle_seed {seed!r}: a whole number")
    dc = doc.get("decl_copies")
    if dc is not None and (not isinstance(dc, int) or dc < 2):
        raise DecodeRefused(f"decl_copies {dc!r}: null or a whole number of at least 2")
    raw = doc.get("cells")
    if not isinstance(raw, list) or not raw:
        raise DecodeRefused("the page holds no cells")
    E, k = cfg.num_experts, cfg.top_k
    out, seen = [], set()
    for c in raw:
        label, T = c.get("label"), c.get("T")
        if not isinstance(label, str) or not LABEL_RE.match(label):
            raise DecodeRefused(f"cell label {label!r}: letters, digits and _ . - only")
        if not isinstance(T, int) or isinstance(T, bool) or not 1 <= T <= 128:
            raise DecodeRefused(f"cell {label}: T {T!r} is not a decode batch (1..128)")
        if label in seen:
            raise DecodeRefused(f"cell label {label} twice (a label names one cell on a page)")
        seen.add(label)
        counts = c.get("counts")
        if (not isinstance(counts, list) or len(counts) != E
                or any(not isinstance(v, int) or isinstance(v, bool) for v in counts)):
            raise DecodeRefused(f"cell {label}: counts must be {E} whole numbers")
        ok, why = feasible(counts, T, k)
        if not ok:
            raise DecodeRefused(f"cell {label} (V-BIN): {why}")
        arms = tuple(c.get("arms") or (NATIVE,))
        if any(a not in ARMS for a in arms) or len(set(arms)) != len(arms) or NATIVE not in arms:
            raise DecodeRefused(f"cell {label}: arms {list(arms)}: native, and optionally decl")
        if DECL in arms and dc is None:
            raise DecodeRefused(f"cell {label}: a decl arm needs the page's decl_copies")
        sh = c.get("shuffle", True)
        if not isinstance(sh, bool):
            raise DecodeRefused(f"cell {label}: shuffle {sh!r}: true or false")
        out.append(DecodeCell(label=label, T=int(T), counts=tuple(int(v) for v in counts),
                              arms=tuple(a for a in ARMS if a in arms), shuffle=sh, role=str(c.get("role", ""))))
    return DecodePage(model=model, page=str(doc.get("page", "")), shuffle_seed=int(seed),
                      decl_copies=dc, cells=tuple(out), file_sha256=file_sha256, label=str(doc.get("label", "")))


def load_page(path, cfg, model: str) -> DecodePage:
    try:
        data = Path(path).read_bytes()
        doc = json.loads(data)
    except (OSError, ValueError) as exc:
        raise DecodeRefused(f"cannot read the decode page {path}: {exc}") from None
    return page_from(doc, cfg, model, hashlib.sha256(data).hexdigest())


def cell_ids(cfg, cell: DecodeCell, shuffle_seed: int, device: str = "cpu"):
    """NATIVE's [T, k] ids: realize_counts, then the page's token-row shuffle (unless the cell's
    shuffle is false). V-BIN: refused unless the bincount is the counts, k distinct per token."""
    import torch

    from moe.routing.distributions import realize_counts
    ids = realize_counts(list(cell.counts), cell.T, cfg.top_k, device="cpu")
    if cell.shuffle:
        g = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
        ids = ids[torch.randperm(cell.T, generator=g)]
    got = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=cfg.num_experts).tolist()
    if got != list(cell.counts):
        raise DecodeRefused(f"cell {cell.label} (V-BIN): bincount {got} is not {list(cell.counts)}")
    if any(len(set(r.tolist())) != cfg.top_k for r in ids):
        raise DecodeRefused(f"cell {cell.label} (V-BIN): a token repeats an expert")
    return ids.to(device)


def expert_bytes(cfg, dtype: str = "bf16") -> int:
    """W_e: one expert's three matrices (w1's 2I x H, w2's H x I)."""
    return 3 * cfg.hidden_size * cfg.intermediate_size * dtype_bytes(dtype)


def rotation(cfg, page: DecodePage, dtype: str = "bf16", override: int | None = None) -> tuple[int, int, str]:
    """(R copies, calls per graph, why): R x A_min x W_e >= GRR_COLD_MIB MiB, A_min the page's least
    coverage; calls the least multiple of R at or over GRR_MIN_CALLS."""
    a_min = min(sum(1 for v in c.counts if v) for c in page.cells)
    need = GRR_COLD_MIB * 2 ** 20
    R = max(1, -(-need // (a_min * expert_bytes(cfg, dtype))))
    why = f"R x A_min {a_min} x W_e {expert_bytes(cfg, dtype) / 2 ** 20:.1f} MiB >= {GRR_COLD_MIB} MiB"
    if override:
        if override < R:
            raise DecodeRefused(f"--rot {override} is under the {R} copies the cold rule needs ({why})")
        R, why = int(override), f"--rot {override} (the rule needs {R}: {why})"
    calls = R * max(1, -(-GRR_MIN_CALLS // R))
    return int(R), int(calls), why


def copies_for(page: DecodePage, inst: str, R: int) -> int:
    """Copies allocated: GRR rotates through R; a DECL page declares decl_copies; the larger."""
    base = R if inst == "GRR" else 1
    return max(base, page.decl_copies or 1)


# --------------------------------------------------------------------------
# the spy: vLLM's real kernel inputs of one eager call
# --------------------------------------------------------------------------

class Spy:
    """Patches vLLM v0.27.1's `dispatch_fused_moe_kernel` and `fused_moe_kernel` (module globals of
    vllm.model_executor.layers.fused_moe.fused_moe, looked up at call time by fused_experts_impl and
    invoke_fused_moe_triton_kernel) for the duration of a `with`, recording per GEMM the sorted ids,
    expert ids, num_tokens_post_padded, the config, the launch grid (the grid lambda evaluated on
    the launch's own META) and USE_TD; then restores both. Synchronising reads: an untimed eager
    call only, never inside a capture."""

    def __init__(self, module=None):
        self.module = module
        self.records: list[dict] = []
        self._saved = None

    def __enter__(self):
        if self.module is None:
            import vllm.model_executor.layers.fused_moe.fused_moe as FM
            self.module = FM
        FM = self.module
        self._saved = (FM.dispatch_fused_moe_kernel, FM.fused_moe_kernel)
        orig_dispatch, orig_kernel = self._saved
        spy = self

        def dispatch(*a, **kw):
            names = ("A", "B", "C", "A_scale", "B_scale", "B_zp", "topk_weights", "sorted_token_ids",
                     "expert_ids", "num_tokens_post_padded", "mul_routed_weight", "top_k", "config")
            got = dict(zip(names, a))
            got.update({k: v for k, v in kw.items() if k in names})
            srt = got.get("sorted_token_ids")
            spy.records.append({
                "a_rows": int(got["A"].size(0)), "b_shape": list(got["B"].shape), "b_ptr": int(got["B"].data_ptr()),
                "sorted": None if srt is None else [int(x) for x in srt.to("cpu").tolist()],
                "expert_ids": [int(x) for x in got["expert_ids"].to("cpu").tolist()],
                "post_padded": int(got["num_tokens_post_padded"].to("cpu").reshape(-1)[0].item()),
                "top_k": int(got["top_k"]), "config": dict(got["config"])})
            return orig_dispatch(*a, **kw)

        class _Kernel:
            def __getitem__(self, grid):
                launcher = orig_kernel[grid]

                def launch(*a, **kw):
                    meta = dict(kw)
                    g = grid(meta) if callable(grid) else grid
                    n = 1
                    for x in (g if isinstance(g, (tuple, list)) else (g,)):
                        n *= int(x)
                    if spy.records:
                        spy.records[-1].update(grid=n, use_td=bool(kw.get("USE_TD", False)),
                                               naive_flag=bool(kw.get("naive_block_assignment", False)),
                                               launched={k: kw[k] for k in CONFIG_KEYS + ("BLOCK_SIZE_K",) if k in kw})
                    return launcher(*a, **kw)
                return launch

            def __getattr__(self, k):
                return getattr(orig_kernel, k)

        FM.dispatch_fused_moe_kernel = dispatch
        FM.fused_moe_kernel = _Kernel()
        return self

    def __exit__(self, *exc):
        FM = self.module
        FM.dispatch_fused_moe_kernel, FM.fused_moe_kernel = self._saved
        return False


def tuned_and_override(cfg, w2_experts: int) -> tuple:
    """GD5's two lookups: vLLM's tuned file for (E, N, bf16) and the override hook."""
    try:
        import vllm.model_executor.layers.fused_moe.fused_moe as FM
        from vllm.model_executor.layers.fused_moe import get_config
        tuned = FM.get_moe_configs(int(w2_experts), int(cfg.intermediate_size), None, 0, 0)
        override = get_config()
        return ("a tuned file" if tuned else None), ("set" if override else None)
    except Exception as exc:                                           # noqa: BLE001
        return f"UNREAD: {type(exc).__name__}: {exc}", None


# --------------------------------------------------------------------------
# rows, store and the schedule
# --------------------------------------------------------------------------

@dataclass
class DecodeSample:
    """One (cell, arm, repeat): what cells.csv holds. The attributes R3's `histogram_schedule`,
    `lock_slip` and `histogram_gates` read (arm, sm_clock_load_mhz, usable, excluded) are here."""
    arm: str
    repeat: int
    tiles: int
    label: str
    inst: str
    ms_p50: float
    status: str = "ok"
    ms_min: float | None = None
    ms_stdev: float | None = None
    calls_per_replay: int = 1
    sm_clock_load_mhz: float | None = None
    clock_samples_mhz: str | None = None
    clock_drift_ok: bool | None = None
    clock_level_ok: bool | None = None
    #: the LEVEL flag's side (HIGH or LOW): recorded, never an exclusion (DRIFT alone excludes)
    clock_level_side: str = ""
    host_bound: bool | None = None
    host_enqueue_ms: float | None = None
    duty: float = DUTY
    duty_achieved: float | None = None
    power_w: float | None = None
    iters: int | None = None
    path: str = ""
    tile: str = ""
    post_padded: int | None = None
    padded_rows: int | None = None
    live_tiles: int | None = None
    dead_ctas_w1: int | None = None
    dead_ctas_w2: int | None = None
    experts_declared: int | None = None
    counts_sha256: str = ""
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status == "ok" and self.ms_p50 > 0

    @property
    def excluded(self) -> bool:
        return False


CSV_FIELDS = [f.name for f in fields(DecodeSample)]


class Store:
    """Append-only cells.csv. A file already on disk must carry exactly this header: DictWriter never
    reads the file, so wider rows under a narrower header would shift every field (the tree's rule)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        new = not self.path.exists()
        if not new:
            with self.path.open(newline="") as fh:
                on_disk = next(csv.reader(fh), [])
            if on_disk and on_disk != CSV_FIELDS:
                raise DecodeRefused(f"{self.path} holds another header ({len(on_disk)} columns, not "
                                    f"{len(CSV_FIELDS)}): a fresh run directory is needed")
        self.fh = self.path.open("a", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=CSV_FIELDS)
        if new:
            self.w.writeheader()
            self.fh.flush()

    def append(self, s: DecodeSample) -> None:
        self.w.writerow({k: getattr(s, k) for k in CSV_FIELDS})
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def schedule(page: DecodePage, repeats: int, run_cell, *, lock_mhz: int | None, min_clean: int, say=print):
    """R3's histogram_schedule over the decode page's cells (T ascending, reversed on odd repeats,
    arms rotated): slips, lost cells and the early stop exactly as on a histogram page."""
    class _C:  # histogram_schedule reads .label, .n
        def __init__(self, c):
            self.c, self.label, self.n = c, c.label, c.T
    order = [_C(c) for c in page.ordered()]
    return PWR.histogram_schedule(order, lambda x: x.c.arms, repeats,
                                  lambda x, rep, rot: run_cell(x.c, rep, rot),
                                  lock_mhz=lock_mhz, min_clean=min_clean, say=say)


# --------------------------------------------------------------------------
# the GPU backend (the off-GPU tests inject their own)
# --------------------------------------------------------------------------

class GPUBackend:
    """Everything that touches torch, vLLM and the card. One weight build per page; per (cell, arm)
    the eager call, the instrument's replay, the readback, GD4 and the host probe."""

    def __init__(self, cfg, page: DecodePage, *, inst: str, R: int, calls: int, copies: int, seed: int,
                 dtype: str = "bf16", duty: float = DUTY, warmup_ms: float = WARMUP_MS,
                 cell_budget_ms: float = CELL_BUDGET_MS, trials: int = TRIALS):
        import torch

        import launch_floor as LF
        from moe.bench import timing as T
        from vllm.model_executor.layers.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        self.torch, self.LF, self.T = torch, LF, T
        self.cfg, self.page, self.inst, self.R, self.calls, self.copies = cfg, page, inst, R, calls, copies
        self.duty, self.warmup_ms, self.budget, self.trials = duty, warmup_ms, cell_budget_ms, trials
        self.fused_experts, self.act = fused_experts, MoEActivation("silu")
        self.w1, self.w2, self.delta = PWR.build_private_weights(cfg, dtype, copies, seed)
        self.flusher = None if inst == "GRR" else T.L2Flusher(T.flush_mb_for_device())
        self.ref_clock, _src = SWEEP.reference_clock_mhz()
        self.dtype = self.w1.dtype
        g = torch.Generator(device="cuda").manual_seed(int(seed) + 1)
        T_max = max(page.Ts)
        self.x_all = torch.randn((T_max, cfg.hidden_size), device="cuda", dtype=self.dtype, generator=g)
        self.gpu = LF.ProbeGPU(torch)
        self.cpm = self.gpu.cycles_per_ms()

    # one call j of (cell, arm)
    def _call(self, cell: DecodeCell, arm: str, ids, j: int):
        torch, cfg, C = self.torch, self.cfg, self.copies
        x = self.x_all[:cell.T].contiguous()
        w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32, device="cuda")
        if arm == NATIVE:
            w1, w2, use, E = self.w1[j::C], self.w2[j::C], ids, cfg.num_experts
        else:
            w1, w2, use, E = self.w1, self.w2, (ids.to(torch.int64) * C + j).to(ids.dtype), cfg.num_experts * C
        fe, act = self.fused_experts, self.act

        def call():
            return fe(hidden_states=x, w1=w1, w2=w2, topk_weights=w, topk_ids=use, activation=act,
                      apply_router_weight_on_input=False, global_num_experts=E, expert_map=None)
        return call

    def build(self, cell: DecodeCell, arm: str):
        ids = cell_ids(self.cfg, cell, self.page.shuffle_seed, device="cuda").to(self.torch.int32)
        calls = [self._call(cell, arm, ids, j % (self.R if self.inst == "GRR" else 1))
                 for j in range(self.calls if self.inst == "GRR" else 1)]
        return ids, calls

    def readback(self, cell: DecodeCell, arm: str, ids, calls) -> dict:
        with Spy() as spy:
            calls[0]()
            self.torch.cuda.synchronize()
        declared = self.cfg.num_experts * (self.copies if arm == DECL else 1)
        tuned, override = tuned_and_override(self.cfg, declared)
        # every one of GRR's R calls spied too: call j must read its own weight copy (j mod R)
        ptrs = [spy.records[0]["b_ptr"]] if spy.records else []
        for c in calls[1:self.R if self.inst == "GRR" else 1]:
            with Spy() as s2:
                c()
                self.torch.cuda.synchronize()
            ptrs.append(s2.records[0]["b_ptr"] if s2.records else None)
        return {"gemms": spy.records, "tuned": tuned, "override": override, "copy_ptrs": ptrs}

    def replay_fn(self, calls):
        """(fn, calls per replay, outputs holder): GRR's graph of every call, GR1's of one, E240's eager."""
        if self.inst == "E240":
            return calls[0], 1, None
        holder = [None] * len(calls)

        def body():
            for i, c in enumerate(calls):
                holder[i] = c()
        replay = self.LF.graph_call(body, override_config=_no_override, conf=None)
        return replay, len(calls), holder

    def fidelity(self, calls, replay, holder) -> tuple[bool, str]:
        """GD4: every captured call's output equals the eager output bit for bit."""
        if holder is None:
            return True, "eager (no graph)"
        torch = self.torch
        want = calls[0]().clone()
        replay()
        torch.cuda.synchronize()
        bad = [i for i, h in enumerate(holder) if h is None or not torch.equal(h, want)]
        return (not bad), (f"{len(holder)} captured calls equal the eager output bit for bit" if not bad
                           else f"captured calls {bad[:8]} differ from the eager output")

    def host_probe(self, replay) -> float | None:
        r = self.LF.host_probe(self.LF.probe_parts(replay, self.flusher, self.gpu.body_event),
                               self.gpu, cycles_per_ms=self.cpm)
        return r.get("H_pre_ms") if r.get("status") == "ok" else None

    def time(self, replay, per_replay: int):
        t, _per = self.LF.mirror_time_cell(replay, flusher=self.flusher, l2_flush=self.flusher is not None,
                                           duty=self.duty, warmup_ms=self.warmup_ms, cell_budget_ms=self.budget,
                                           trials=self.trials, reference_clock_mhz=self.ref_clock)
        return t


@contextlib.contextmanager
def _no_override(_conf):
    """launch_floor.graph_call's override_config, here a no-op: vLLM picks its own tile (GD5)."""
    yield


def timing_fields(t, per_replay: int) -> dict:
    """A DutyTiming as one DecodeSample's columns, times per CALL (a replay holds per_replay)."""
    clocks = getattr(t, "clock_samples_mhz", ()) or ()
    return {"ms_p50": float(t.ms_p50) / per_replay, "ms_min": float(t.ms_min) / per_replay,
            "ms_stdev": float(t.ms_std) / per_replay, "calls_per_replay": per_replay,
            "sm_clock_load_mhz": t.sm_clock_load_mhz, "clock_samples_mhz": " ".join(f"{c:.0f}" for c in clocks) or None,
            "clock_drift_ok": t.clock_drift_ok, "clock_level_ok": t.clock_level_ok,
            "clock_level_side": getattr(t, "clock_level_side", "") or "", "host_bound": t.host_bound,
            "host_enqueue_ms": t.host_enqueue_ms, "duty_achieved": t.duty_achieved, "power_w": t.power_w,
            "iters": t.samples // max(1, t.trials)}


# --------------------------------------------------------------------------
# the page run (backend injected)
# --------------------------------------------------------------------------

def run_page(cfg, page: DecodePage, backend, *, inst: str, repeats: int, store: Store, lock_mhz, min_clean: int,
             say=print) -> dict:
    """Checks every (cell, arm) untimed (V-BIN, V-ALIGN, GD5, GD4, the GD1 host probe), then R3's
    repeat loop over the cells that passed. Returns the per-cell checks, the samples by cell, the
    lost cells and the early stop."""
    checks: dict = {}
    refused: set = set()
    for c in page.ordered():
        for a in c.arms:
            key = (c.label, c.T, a)
            declared = cfg.num_experts * (backend.copies if a == DECL else 1)
            try:
                ids, calls = backend.build(c, a)
                exp = expected_readback(cfg, c.T, c.counts, ids.reshape(-1).tolist() if a == NATIVE else
                                        [int(v) * backend.copies for v in ids.reshape(-1).tolist()],
                                        declared=declared, copies=backend.copies if a == DECL else 1)
                rec = backend.readback(c, a, ids, calls)
                bad = check_readback(rec, exp)
                if inst == "GRR":
                    bad += check_copies(rec, backend.R)
                replay, per, holder = backend.replay_fn(calls)
                ok4, why4 = backend.fidelity(calls, replay, holder)
                h_pre = backend.host_probe(replay)
            except Exception as exc:                                  # noqa: BLE001
                bad, exp, rec = [f"{type(exc).__name__}: {exc}"], {}, {}
                ok4, why4, h_pre, per = False, "not reached", None, 1
            checks[key] = {"readback_bad": bad, "expected": {k: v for k, v in exp.items() if k != "expert_order"},
                           "gd4": ok4, "gd4_detail": why4, "h_pre_ms": h_pre, "h_post_ms": None,
                           "calls_per_replay": per}
            if bad:
                refused.add(key)
                say(f"  REFUSED {c.label} {a} T={c.T}: " + "; ".join(bad)[:300])

    def run_cell(c, rep, rot):
        got = []
        for a in rot:
            key = (c.label, c.T, a)
            ck = checks[key]
            exp = ck["expected"]
            base = dict(arm=a, repeat=rep, tiles=c.T, label=c.label, inst=inst, path=exp.get("path", ""),
                        tile=tile_key(exp["config"]) if exp.get("config") else "",
                        post_padded=exp.get("post_padded"), padded_rows=exp.get("padded_rows"),
                        live_tiles=exp.get("live_tiles"),
                        dead_ctas_w1=(exp.get("dead_ctas") or [None, None])[0],
                        dead_ctas_w2=(exp.get("dead_ctas") or [None, None])[1],
                        experts_declared=cfg.num_experts * (backend.copies if a == DECL else 1),
                        counts_sha256=counts_sha256(c.counts))
            tag = f"histogram={c.label}"
            if key in refused:
                s = DecodeSample(ms_p50=0.0, status="refused", detail=f"{tag}; " + "; ".join(ck["readback_bad"])[:400], **base)
            else:
                try:
                    ids, calls = backend.build(c, a)
                    replay, per, _h = backend.replay_fn(calls)
                    t = backend.time(replay, per)
                    s = DecodeSample(**base, **timing_fields(t, per), detail=tag + (f"; {t.clock_note}" if getattr(t, "clock_note", "") else ""))
                except Exception as exc:                              # noqa: BLE001
                    s = DecodeSample(ms_p50=0.0, status="failed", detail=f"{tag}; {type(exc).__name__}: {exc}", **base)
            store.append(s)
            got.append(s)
            say(f"  rep{rep:2d} {c.label:<16s} {a:6s} T={c.T:3d} {s.ms_p50 * 1e3:9.2f} us {s.status}")
        return got

    samples, lost, early = schedule(page, repeats, run_cell, lock_mhz=lock_mhz, min_clean=min_clean, say=say)
    for c in page.ordered():
        for a in c.arms:
            key = (c.label, c.T, a)
            if key in refused:
                continue
            try:
                ids, calls = backend.build(c, a)
                replay, _per, _h = backend.replay_fn(calls)
                checks[key]["h_post_ms"] = backend.host_probe(replay)
            except Exception as exc:                                  # noqa: BLE001
                checks[key]["h_post_ms"] = None
                checks[key]["h_post_error"] = f"{type(exc).__name__}: {exc}"
    return {"checks": checks, "samples": samples, "lost": lost, "early": early, "refused": refused}


def gd1(check: dict, ms_call: float | None, per: int) -> dict:
    """GD1 on one (cell, arm): GPU per replay >= GD1_FACTOR x the host enqueue per replay, the larger
    of the pre and post probes. An unread probe fails (excluded, not voided)."""
    hs = [h for h in (check.get("h_pre_ms"), check.get("h_post_ms")) if h is not None]
    if ms_call is None or not hs:
        return {"verdict": "FAIL", "why": "no time or no host probe", "gpu_ms": None, "host_ms": None}
    gpu, host = ms_call * per, max(hs)
    ok = gpu >= GD1_FACTOR * host
    return {"verdict": "PASS" if ok else "FAIL", "gpu_ms": gpu, "host_ms": host, "factor": GD1_FACTOR,
            "why": "" if ok else f"GPU {gpu * 1e3:.1f} us per replay under {GD1_FACTOR:g} x host {host * 1e3:.1f} us"}


def rows_of(cfg, page: DecodePage, result: dict, lock_mhz, min_clean: int) -> list[dict]:
    """report.json's treads_table: one row per (cell, arm), R3's histogram-page fields plus the
    decode readback, GD1 and GD4."""
    rows = []
    for c in page.ordered():
        for a in c.arms:
            key = (c.label, c.T, a)
            mine = result["samples"].get(key, [])
            got = [s for s in mine if PWR.clean_repeat(s, lock_mhz)]
            clk = [s.sm_clock_load_mhz for s in got if s.sm_clock_load_mhz is not None]
            ck = result["checks"][key]
            ms = statistics.median(s.ms_p50 for s in got) if got else None
            every = [s.sm_clock_load_mhz for s in mine if s.sm_clock_load_mhz is not None]
            g = geometry(cfg, c.T, c.counts, cfg.num_experts * ((page.decl_copies or 1) if a == DECL else 1))
            row = {"arm": a, "tiles": c.T, "histogram": c.label, "role": c.role, "counts_sha256": counts_sha256(c.counts),
                   "shuffle_seed": page.shuffle_seed if c.shuffle else None, "shuffled": c.shuffle,
                   "bincount": list(c.counts),
                   "bincount_ok": not any("V-BIN" in b for b in ck["readback_bad"]),
                   "ms_p50": ms, "sm_clock_load_mhz": statistics.median(clk) if clk else None,
                   "repeats": len(got), "host_bound": any(bool(s.host_bound) for s in got),
                   "host_enqueue_ms": statistics.median(h) if (h := [s.host_enqueue_ms for s in got
                                                                     if s.host_enqueue_ms is not None]) else None,
                   "repeat_rows": len(mine), "lock_slip_repeats": sum(PWR.lock_slip(s, lock_mhz) for s in mine),
                   "clean_repeats": len(got), "worst_repeat_mhz": min(every) if every else None,
                   "lost": key in result["lost"] or len(got) < min_clean or key in result["refused"],
                   "refused": key in result["refused"], "readback_bad": ck["readback_bad"],
                   "readback": ck["expected"], "geometry": g.as_dict(), "A": g.A, "path": g.path, "tile": g.tile,
                   "gd4": ck["gd4"], "gd4_detail": ck["gd4_detail"],
                   "gd1": gd1(ck, ms, ck.get("calls_per_replay") or 1),
                   "experts_declared": g.declared, "align_path": g.path}
            rows.append(row)
    return rows


def decode_gates(rows: list[dict], samples: dict) -> dict:
    """R3's histogram-page gates (G1, G2, G4, G5) plus the decode ones: V-ALIGN (refused cells,
    listed), GD1 (excluded cells, listed), GD4 (a failure voids the page), GD5 (inside V-ALIGN)."""
    g = PWR.histogram_gates([r for r in rows if not r["refused"]] or rows, samples)
    refused = [f"{r['histogram']}/{r['arm']}/T{r['tiles']}" for r in rows if r["refused"]]
    g1x = [f"{r['histogram']}/{r['arm']}/T{r['tiles']}" for r in rows if not r["refused"] and r["gd1"]["verdict"] != "PASS"]
    g4x = [f"{r['histogram']}/{r['arm']}/T{r['tiles']}" for r in rows if not r["refused"] and not r["gd4"]]
    g.update({
        "V_ALIGN": {"verdict": "PASS" if not refused else "FAIL", "refused_cells": refused,
                    "rule": "a readback mismatch (V-ALIGN or GD5) refuses the cell; the page stands"},
        "GD1_gpu_bound": {"verdict": "PASS", "excluded": g1x,
                          "rule": f"GPU per replay >= {GD1_FACTOR:g} x host per replay; a cell failing it is excluded"},
        "GD4_graph_fidelity": {"verdict": "PASS" if not g4x else "FAIL", "failed": g4x,
                               "rule": "every captured call equals the eager output bit for bit; a failure voids the page"}})
    return g


# --------------------------------------------------------------------------
# the counter child (under ncu, launched by scripts/dram_counter_route_decode.py)
# --------------------------------------------------------------------------

COUNTER_PLAN_KEYS = ("family", "model", "dtype", "page", "page_sha256", "cells", "calls_per_cell",
                     "warmup_calls", "gemms_per_call", "seed", "manifest", "decl_copies")


def validate_counter_plan(plan: dict, page: DecodePage) -> None:
    missing = [k for k in COUNTER_PLAN_KEYS if k not in plan]
    if missing:
        raise DecodeRefused(f"the counter plan lacks {missing}")
    if plan["family"] != "r3-decode":
        raise DecodeRefused(f"family {plan['family']!r}: this child runs r3-decode plans only")
    if plan["page_sha256"] != page.file_sha256:
        raise DecodeRefused("the plan's page sha256 is not the page's")
    if int(plan["gemms_per_call"]) != PWR.GEMMS_PER_CALL:
        raise DecodeRefused(f"gemms_per_call {plan['gemms_per_call']}: vLLM makes {PWR.GEMMS_PER_CALL}")
    by = {c.label: c for c in page.cells}
    for lab, arm in plan["cells"]:
        if lab not in by or arm not in by[lab].arms:
            raise DecodeRefused(f"plan cell ({lab}, {arm}) is not on the page")


def counter_child(plan_path: Path) -> int:
    """The planned calls, nothing timed: per cell (in plan order) `warmup_calls` then `calls_per_cell`
    eager calls on copy 0, and a manifest with each cell's grids READ BACK by the spy (so ncu's
    launch__grid_size is held to vLLM's own launch, the existing counter route's attribution)."""
    import torch
    plan = json.loads(Path(plan_path).read_text())
    cfg = MODEL_CONFIGS[plan["model"]]
    page = load_page(REPO / plan["page"] if not Path(plan["page"]).is_absolute() else plan["page"], cfg, plan["model"])
    validate_counter_plan(plan, page)
    copies = max(1, int(plan.get("decl_copies") or 1))
    be = GPUBackend.__new__(GPUBackend)
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    be.torch, be.cfg, be.page, be.inst, be.R, be.calls, be.copies = torch, cfg, page, "E240", 1, 1, copies
    be.fused_experts, be.act = fused_experts, MoEActivation("silu")
    be.w1, be.w2, _d = PWR.build_private_weights(cfg, plan["dtype"], copies, int(plan["seed"]))
    g = torch.Generator(device="cuda").manual_seed(int(plan["seed"]) + 1)
    be.x_all = torch.randn((max(page.Ts), cfg.hidden_size), device="cuda", dtype=be.w1.dtype, generator=g)
    by = {c.label: c for c in page.cells}
    if int(plan["warmup_calls"]) < 1:
        raise DecodeRefused("warmup_calls >= 1: the spy's readback call is the first warmup")
    order, grids, refused, built = [], {}, [], {}
    # THE LAUNCH ORDER the pinned route's attribution expects with no launch count
    # (dram_counter_route.r3_launch_sequence): every cell's warmups first, in plan order (the
    # readback call is the first of them), then every cell's measured calls.
    for lab, arm in plan["cells"]:
        c = by[lab]
        ids, calls = be.build(c, arm)
        rec = be.readback(c, arm, ids, calls)
        declared = cfg.num_experts * (copies if arm == DECL else 1)
        flat = ids.reshape(-1).tolist()
        exp = expected_readback(cfg, c.T, c.counts, flat if arm == NATIVE else [int(v) * copies for v in flat],
                                declared=declared, copies=copies if arm == DECL else 1)
        bad = check_readback(rec, exp)
        key = f"{arm}/{c.T}/{lab}"
        if bad:
            refused.append({"cell": key, "why": bad})
            break
        order.append([arm, c.T, lab])
        grids[key] = {g_: int(r["grid"]) for g_, r in zip(GEMMS, rec["gemms"])}
        for _ in range(int(plan["warmup_calls"]) - 1):
            calls[0]()
        torch.cuda.synchronize()
        built[key] = calls[0]
    if not refused:
        for arm, T, lab in order:
            key = f"{arm}/{T}/{lab}"
            for _ in range(int(plan["calls_per_cell"])):
                torch.cuda.nvtx.range_push(f"decode/{key}")
                built[key]()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
    manifest = {"family": "r3-decode", "order": order, "grids": grids, "refused": refused,
                "calls_per_cell": int(plan["calls_per_cell"]), "warmup_calls": int(plan["warmup_calls"]),
                "gemms_per_call": PWR.GEMMS_PER_CALL, "launch_count": None, "versions": PWR._package_versions(),
                "page_sha256": page.file_sha256}
    Path(plan["manifest"]).write_text(json.dumps(manifest, indent=1))
    print(f"counter child: {len(order)} cells, {len(refused)} refused, manifest {plan['manifest']}")
    return exit_codes.DONE if not refused else exit_codes.INVALID


# --------------------------------------------------------------------------
# provenance of the installed vLLM against the fetched v0.27.1 (V-SRC)
# --------------------------------------------------------------------------

def upstream_record() -> dict:
    return json.loads(UPSTREAM.read_text())


def installed_source_check() -> list[str]:
    """The VM's installed fused_moe.py and moe_align_block_size.py against UPSTREAM_DECODE.json's
    sha256: every rule here was read off those bytes. A mismatch refuses the page."""
    rec = upstream_record()["files"]
    bad = []
    try:
        import vllm.model_executor.layers.fused_moe.fused_moe as FM
        import vllm.model_executor.layers.fused_moe.moe_align_block_size as MA
    except Exception as exc:                                           # noqa: BLE001
        return [f"vLLM does not import: {type(exc).__name__}: {exc}"]
    for name, mod in (("fused_moe.py", FM), ("moe_align_block_size.py", MA)):
        got = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
        if got != rec[name]["sha256"]:
            bad.append(f"{name}: installed sha256 {got[:16]} is not v0.27.1's {rec[name]['sha256'][:16]}")
    return bad


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="olmoe-1b-7b", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--dtype", default="bf16", choices=("bf16",))
    ap.add_argument("--histogram", type=Path, default=None, help="the decode page (decode-page/1)")
    ap.add_argument("--inst", default="GRR", choices=INSTRUMENTS)
    ap.add_argument("--rot", default="auto", help="GRR's copies R: auto (the cold rule) or a whole number at or over it")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--duty", type=float, default=DUTY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=WARMUP_MS)
    ap.add_argument("--cell-budget-ms", type=float, default=CELL_BUDGET_MS)
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--group-m", type=int, default=1)
    ap.add_argument("--session-tag", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--slip-policy", choices=("page", "cell"), default="page")
    ap.add_argument("--lock-mhz", type=int, default=None)
    ap.add_argument("--min-clean-repeats", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--counter-child", default=None, metavar="PLAN")
    return ap


def run_id_for(args, card: str, page: DecodePage) -> str:
    key = json.dumps({"model": args.model, "inst": args.inst, "rot": args.rot, "page": page.file_sha256,
                      "session": args.session_tag, "repeats": args.repeats, "duty": args.duty, "seed": args.seed,
                      "group_m": args.group_m}, sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"{card}-decode-{args.model}-{args.inst}-{(page.page or 'page')[:40]}-{h}"


def estimated_seconds(page: DecodePage, repeats: int) -> float:
    arms = sum(len(c.arms) for c in page.cells)
    return arms * repeats * SEC_PER_CELL + 2 * arms * SEC_PER_CHECK + 60 * SETUP_MIN


def plan_lines(args, cfg, page: DecodePage, *, R: int, calls: int, why_R: str, copies: int, out_dir: Path,
               slip_lock) -> list[str]:
    lines = [
        f"experiment  r3_decode / {out_dir.name}",
        f"model       {args.model} E={cfg.num_experts} k={cfg.top_k} {args.dtype}; W_e {expert_bytes(cfg) / 2 ** 20:.1f} MiB",
        f"page        {args.histogram} (sha256 {page.file_sha256[:16]}...), {page.page or '-'}, {len(page.cells)} cells, "
        f"{sum(len(c.arms) for c in page.cells)} (cell, arm), token-row shuffle seed {page.shuffle_seed}",
        f"instrument  {args.inst}: " + {
            "GRR": f"one graph of {calls} calls, call j on copy j mod {R} ({why_R}); per call = replay / {calls}; no flush",
            "GR1": "a one-call graph, the 240 MiB flush outside it",
            "E240": "the eager call, the 240 MiB flush"}[args.inst],
        f"copies      {copies} allocated (expert-first, R3's build_private_weights); NATIVE reads w1[j::{copies}]"
        + (f"; DECL declares E x {page.decl_copies} = {cfg.num_experts * page.decl_copies}" if page.decl_copies else ""),
        "tile        vLLM's own: get_default_config(T, E), read back off the kernel (GD5); no override_config",
        f"repeats     {args.repeats}, T ascending, reversed on odd repeats, arms rotated (R3's histogram_schedule)",
        f"duty        {args.duty:.2f}; warmup {args.warmup:g} ms, {args.trials} trials of {args.cell_budget_ms:g} ms",
        f"estimate    {estimated_seconds(page, args.repeats) / 60:.1f} min",
        "session     " + (args.session_tag or "(none: a bare run, keyed on its arguments and card alone)"),
        f"WRITES TO   {out_dir}",
    ]
    if slip_lock is not None:
        lines.append(f"slips       --slip-policy cell at the {slip_lock} MHz lock (R3's lock_slip); "
                     f"min clean repeats {args.min_clean_repeats}")
    lines.append("cells (T, label, arms, path, tile, A, L, P, dead w1/w2, counts sha256):")
    for c in page.ordered():
        g = geometry(cfg, c.T, c.counts)
        lines.append(f"  T={c.T:<4d}{c.label:<22s}{','.join(c.arms):<12s}{g.path:<12s}{g.tile:<15s}A={g.A:<4d}"
                     f"L={g.L:<4d}P={g.P:<5d}dead={g.dead_ctas[0]}/{g.dead_ctas[1]:<6d}{counts_sha256(c.counts)[:12]}")
    return lines


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.counter_child:
        return counter_child(Path(args.counter_child))
    cfg = MODEL_CONFIGS[args.model]
    try:
        if args.histogram is None:
            raise DecodeRefused("a decode page is --histogram PAGE (decode-page/1)")
        page = load_page(args.histogram, cfg, args.model)
        if args.group_m != 1:
            raise DecodeRefused(f"--group-m {args.group_m}: vLLM's decode config has GROUP_SIZE_M 1 (locked_r3 --groups 1)")
        rot = None if args.rot == "auto" else int(args.rot)
        R, calls, why_R = rotation(cfg, page, args.dtype, rot)
        if args.inst != "GRR" and rot:
            raise DecodeRefused("--rot is GRR's")
        if any(DECL in c.arms for c in page.cells) and args.inst == "GRR" and (page.decl_copies or 0) < R:
            raise DecodeRefused(f"a DECL page under GRR needs decl_copies >= R = {R}")
        copies = copies_for(page, args.inst, R)
        slip_lock = None
        if args.slip_policy == "cell":
            if not args.lock_mhz:
                raise DecodeRefused("--slip-policy cell needs --lock-mhz (locked_r3 passes it)")
            if not 0 <= args.min_clean_repeats <= args.repeats:
                raise DecodeRefused(f"--min-clean-repeats {args.min_clean_repeats}: 0 to --repeats")
            slip_lock = int(args.lock_mhz)
        elif args.lock_mhz is not None or args.min_clean_repeats:
            raise DecodeRefused("--lock-mhz and --min-clean-repeats ride with --slip-policy cell")
        if not 0 < args.duty < 1:
            raise DecodeRefused(f"--duty {args.duty}: the duty timer's range (0, 1); the registered 0.25")
    except (DecodeRefused, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    card = PWR.detect_card_slug()
    run_id = args.run_id or run_id_for(args, card, page)
    out_dir = (args.out or SWEEP.results_root()) / "private_weight_reference" / run_id
    header = plan_lines(args, cfg, page, R=R, calls=calls, why_R=why_R, copies=copies, out_dir=out_dir,
                        slip_lock=slip_lock)
    print("DECODE PAGE")
    print("\n".join(header))
    b = dtype_bytes(args.dtype)
    free, src = PWR._device_memory(argparse.Namespace(device_memory_gb=0.0, dry_run=args.dry_run))
    mem = PWR.memory_plan(cfg, args.dtype, b, copies, max(page.Ts), free, src, copies_read=1)
    if mem.fits is False:
        print(f"\nREFUSED: the weight copies do not fit: {mem.predicted_peak_bytes / 1e9:.2f} GB predicted")
        return exit_codes.REFUSED
    if args.dry_run:
        print("\n".join(["", "=" * 72, "REFUSED. Nothing was measured and nothing was written.",
                         "  reason: --dry-run was given", "=" * 72]))
        return exit_codes.REFUSED
    missing = SWEEP.missing_gpu_stack()
    if missing:
        print("\n" + missing)
        return exit_codes.REFUSED
    src_bad = installed_source_check()
    if src_bad:
        print("\nREFUSED (V-SRC): " + "; ".join(src_bad))
        return exit_codes.REFUSED
    unreadable = PWR.clock_sampler_refusal()
    if unreadable:
        print(f"\nREFUSED: {unreadable}")
        return exit_codes.REFUSED
    out_dir.mkdir(parents=True, exist_ok=True)
    wrong = PWR.device_guard(out_dir, PWR.device_identity())
    if wrong:
        print(f"\nREFUSED: {wrong}")
        return exit_codes.REFUSED
    os.environ["TRITON_CACHE_DIR"] = str(out_dir / "triton-cache")
    try:
        store = Store(out_dir / "cells.csv")
    except DecodeRefused as exc:
        print(f"\nREFUSED: {exc}")
        return exit_codes.REFUSED
    t0 = time.time()
    backend = GPUBackend(cfg, page, inst=args.inst, R=R, calls=calls, copies=copies, seed=args.seed,
                         dtype=args.dtype, duty=args.duty, warmup_ms=args.warmup,
                         cell_budget_ms=args.cell_budget_ms, trials=args.trials)
    result = run_page(cfg, page, backend, inst=args.inst, repeats=args.repeats, store=store, lock_mhz=slip_lock,
                      min_clean=args.min_clean_repeats if slip_lock is not None else 0,
                      say=lambda m: print(m, flush=True))
    store.close()
    return write_report(args, cfg, page, result, out_dir=out_dir, run_id=run_id, card=card, R=R, calls=calls,
                        copies=copies, slip_lock=slip_lock, header=header, seconds=time.time() - t0)


def write_report(args, cfg, page: DecodePage, result: dict, *, out_dir: Path, run_id: str, card: str, R: int,
                 calls: int, copies: int, slip_lock, header: list[str], seconds: float) -> int:
    rows = rows_of(cfg, page, result, slip_lock, args.min_clean_repeats if slip_lock is not None else 0)
    gates = decode_gates(rows, result["samples"])
    payload = {
        "experiment": "r3_decode", "kind": "decode-page", "run_id": run_id, "session_tag": args.session_tag,
        "card": card, "model": args.model, "dtype": args.dtype, "inst": args.inst, "duty": args.duty,
        "repeats": args.repeats, "group_m": args.group_m, "rotation": R, "calls_per_graph": calls if args.inst == "GRR" else 1,
        "copies_allocated": copies, "decl_copies": page.decl_copies, "seed": args.seed,
        "pinned": {"tile": "vLLM default_config per T (GD5)", "num_warps": None},
        "histogram": {"schema": page.schema, "file_sha256": page.file_sha256, "page": page.page,
                      "shuffle_seed": page.shuffle_seed,
                      "cells": [{"label": c.label, "n": c.T, "counts_sha256": counts_sha256(c.counts),
                                 "arms": list(c.arms), "shuffle": c.shuffle, "role": c.role} for c in page.ordered()]},
        "histogram_gates": gates, "treads_table": rows,
        "gates": [{"kind": "VALIDITY", "tag": k, "verdict": v["verdict"]} for k, v in gates.items()
                  if v["verdict"] in (PWR.PASS, PWR.FAIL)],
        "upstream": {k: v["sha256"] for k, v in upstream_record()["files"].items()},
        "versions": PWR._package_versions(), "seconds": round(seconds, 1),
    }
    if slip_lock is not None:
        payload.update({"slip_policy": "cell", "lock_mhz": slip_lock, "min_clean_repeats": args.min_clean_repeats,
                        "repeat_rows": sum(len(v) for v in result["samples"].values()),
                        "lock_slip_rows": sum(PWR.lock_slip(x, slip_lock) for v in result["samples"].values() for x in v),
                        "lost_cells": {f"{k[0]}/{k[2]}/T{k[1]}": why for k, why in result["lost"].items()},
                        "early_stop": result["early"]})
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str))
    lines = [f"  {k}: {v['verdict']}" for k, v in gates.items()]
    (out_dir / "report.txt").write_text("\n".join(header + ["", "DECODE GATES:", *lines]) + "\n")
    print("\n".join(["", "DECODE GATES:", *lines]))
    print(f"json     {out_dir / 'report.json'}")
    bad = [k for k, v in gates.items() if v["verdict"] == PWR.FAIL and k not in ("V_ALIGN",)]
    rc = exit_codes.INVALID if bad else exit_codes.DONE
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
