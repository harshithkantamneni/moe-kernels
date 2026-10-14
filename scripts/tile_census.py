#!/usr/bin/env python
"""The tile census: compile only, no timing (registration 2026-10-14-tileA-census; design-tile 3.1
step 1 and 5.3, design-tile-review 6).

    python scripts/tile_census.py --tiles 128x128x64w8s3,64x128x64w8s3 --models olmoe-1b-7b,phi-3.5-moe \\
        --out $D/tile-census-tA.json                     # on the VM (vLLM, Triton, a GPU)
    python scripts/tile_census.py --dry-run --tiles ... --models ...   # the plan; exit 2
    python scripts/tile_census.py --read-ptx FILE [--warps N]          # the PTX counts of one file

For every (tile, model, GEMM) it compiles vLLM's fused_moe_kernel at the tile's config values (one
untimed NATIVE call at the tile's first vLLM-consistent tread, n 1) in a Triton cache of its own,
and records what v4 reads instead of fitting it: registers (the cubin's EIATTR_REGCOUNT), shared
bytes, warps, stages, the CTAs per SM and the binding limit (moe.instrumented.cubin.occupancy, the
sm_90 occupancy calculator), the MMA path, and the main loop's per-thread PTX counts:
  mma.sync  ldmatrix and cp.async (r3_timing_model.ptx_main_loop, unchanged), and the LSU
            wavefronts per CTA k-step (r3_timing_model_v4.wavefronts_from_counts);
  wgmma     wgmma.mma_async count and shape, cp.async count and bytes, bar.sync,
            fence.proxy.async, wgmma.wait_group and cp.async.wait_group in the loop that issues
            wgmma (`wgmma_main_loop`, new here beside ptx_main_loop, which raises on a wgmma loop).

THE UNIT REFUSES ITSELF (exit 3, VALIDITY) when any config's compiled MMA path is not the one
Triton's predicate names (wgmma iff BLOCK_M % 64 == 0 and num_warps % 4 == 0): every v4 law is
per path, so a path mismatch voids the census. It compiles the BLIND models' configs too: a
compile is not a timing, and their occupancy is then read, not fitted.

The warps reach vLLM's config through `r3_tile.set_warps` (FIXED, in memory), as R3's pages do.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_timing_model_v4 as V4  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

_LABEL_RE = re.compile(r"^(\$L__BB\d+_\d+):")
_BRA_RE = re.compile(r"bra(?:\.uni)?\s+(\$L__BB\d+_\d+);")
_WGMMA_RE = re.compile(r"wgmma\.mma_async\.sync\.aligned\.(m64n\d+k16)\.f32\.(\w+)\.(\w+)")
_CP_RE = re.compile(r"cp\.async\.cg\.shared\.global\s*\[[^]]*\],\s*\[[^]]*\],\s*(0x[0-9a-fA-F]+|\d+)")
_REQNTID_RE = re.compile(r"\.reqntid\s+(\d+)")


def wgmma_main_loop(text: str) -> dict:
    """Per-thread counts in the loop that issues wgmma.mma_async (between its label and its
    back-branch): the wgmma instructions and their one shape, cp.async (and bytes per thread),
    bar.sync, fence.proxy.async, wgmma.wait_group and cp.async.wait_group, and the CTA's threads."""
    lines = [ln.strip() for ln in text.splitlines()]
    labels = {m.group(1): i for i, ln in enumerate(lines) for m in [_LABEL_RE.match(ln)] if m}
    for i, ln in enumerate(lines):
        m = _BRA_RE.search(ln)
        if not (m and m.group(1) in labels and labels[m.group(1)] < i):
            continue
        body = lines[labels[m.group(1)]:i + 1]
        wg = [x for b in body for x in [_WGMMA_RE.search(b)] if x]
        if not wg:
            continue
        shapes = sorted({x.group(1) for x in wg})
        if len(shapes) != 1:
            raise ValueError(f"wgmma loop issues {shapes}: one shape expected")
        cp = [int(x.group(1), 0) for b in body for x in [_CP_RE.search(b)] if x]
        nt = _REQNTID_RE.search(text)
        return {"wgmma": len(wg), "instruction": shapes[0], "dtype": wg[0].group(2),
                "cp_async_cg": len(cp), "cp_async_bytes_per_thread": sum(cp),
                "bar_sync": sum(b.startswith("bar.sync") for b in body),
                "fence_proxy_async": sum(b.startswith("fence.proxy.async") for b in body),
                "wgmma_wait_group": sum(b.startswith("wgmma.wait_group") for b in body),
                "cp_async_wait_group": sum(b.startswith("cp.async.wait_group") for b in body),
                "threads": int(nt.group(1)) if nt else None}
    raise ValueError("no loop issuing wgmma.mma_async in this PTX")


def ptx_counts(ptx: str, tile: V4.Tile) -> dict:
    """The loop counts v4 reads, by the tile's path: mma.sync gives W (wavefronts), wgmma the
    instruction, its count per warpgroup and the warpgroups (the input of wgmma_bytes)."""
    if tile.path == "mma.sync":
        import r3_timing_model as TM
        c = TM.ptx_main_loop(ptx)
        return {"path": "mma.sync", **c, "wavefronts": V4.wavefronts_from_counts(c, tile.warps)}
    c = wgmma_main_loop(ptx)
    return {"path": "wgmma", **c, "per_warpgroup": c["wgmma"], "warpgroups": tile.warps // 4}


def path_check(entries: list[dict]) -> list[str]:
    """Every config whose compiled MMA path is not the predicate's."""
    bad = []
    for e in entries:
        t = V4.Tile.parse(e["tile"])
        if e.get("mma") != t.path:
            bad.append(f"{e['model']} {e['tile']} {e['gemm']}: compiled {e.get('mma')}, the predicate says {t.path}")
    return bad


def summary(entries: list[dict]) -> dict:
    """Per tile: the CTAs per SM each (model, GEMM) compiled to, and whether one value holds."""
    out: dict = {}
    for e in entries:
        o = (e.get("occupancy") or {}).get("ctas_per_sm")
        d = out.setdefault(e["tile"], {"by": {}, "values": set()})
        d["by"][f"{e['model']}|{e['gemm']}"] = o
        d["values"].add(o)
    return {k: {"by": v["by"], "values": sorted(x for x in v["values"] if x is not None),
                "one_value": len(v["values"]) == 1} for k, v in out.items()}


def _gemm_of(entry: dict) -> str:
    """w2 multiplies the routed weights (MUL_ROUTED_WEIGHT True); w1 does not."""
    return "w2" if entry.get("MUL_ROUTED_WEIGHT") else "w1"


def compile_tile(model: str, tile: V4.Tile, seed: int = 0) -> list[dict]:
    """ON THE VM: one untimed NATIVE call of the plain kernel at the tile's config, in a fresh
    Triton cache; the compiled facts of each GEMM (moe.instrumented.cubin.compiled_facts)."""
    import r3_tile as RT
    RT.set_warps(tile.warps)
    import block_m_crossing_sweep as SWEEP
    import private_weight_reference as PW
    import torch
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe import fused_moe as FM

    from moe import instrumented as I
    from moe.instrumented import cubin as CB
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    override, _where = SWEEP.find_override()
    w1, w2, _ = PW.build_private_weights(cfg, "bf16", 1, seed)
    declared = {a: PW.declared_experts(a, cfg.num_experts, 1) for a in PW.ARMS}
    _t, x, ids_by_arm, weights, kw = PW.arm_inputs(cfg, 1, tile.bm, 1, seed, "bf16", w1.dtype, arms=("native",))
    fn = PW.arm_call(fused_experts, "native", w1, w2, w1, w2, declared, x, ids_by_arm, weights, kw)
    conf = dict(PW.pinned_config(tile.bn, 1, tile.stages, **PW.block_k_kw(tile.bk)), BLOCK_SIZE_M=tile.bm)
    with override(conf):
        with I.timed_plain(module=FM) as rec:
            fn()
        torch.cuda.synchronize()
    out = []
    for lr in rec.launches:
        f = CB.compiled_facts(lr.get("compiled"))
        if f is None:
            continue
        e = {"model": model, "tile": tile.key, "gemm": _gemm_of(lr), "config": list(I.config_key(lr)),
             "regs": f["regs"], "shared": f["shared"], "num_warps": f["num_warps"],
             "mma": ("wgmma" if "wgmma.mma_async" in f["ptx"] else "mma.sync" if "mma.sync" in f["ptx"] else "?"),
             "occupancy": f["occupancy"]}
        try:
            e["ptx_counts"] = ptx_counts(f["ptx"], tile)
        except ValueError as exc:
            e["ptx_counts"] = {"error": str(exc)}
        if not any((o["gemm"], o["config"]) == (e["gemm"], e["config"]) for o in out):
            out.append(e)
    del w1, w2
    torch.cuda.empty_cache()
    return out


def plan_lines(tiles, models) -> list[str]:
    out = ["TILE CENSUS: compile only, no lock, nothing timed",
           f"tiles   {' '.join(t.key for t in tiles)}", f"models  {' '.join(models)}",
           "per (tile, model, GEMM): registers, shared, warps, stages, CTAs per SM and the binding limit, the "
           "MMA path, the main loop's PTX counts; a path other than the predicate's refuses the unit (exit 3)"]
    for t in tiles:
        out.append(f"  {t.key}: path {t.path}, shared {t.shared_bytes()} B, design occupancy {t.occ_shared()}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--read-ptx", type=Path, default=None)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.read_ptx:
        text = a.read_ptx.read_text()
        kind = "wgmma" if "wgmma.mma_async" in text else "mma.sync"
        if kind == "wgmma":
            c = wgmma_main_loop(text)
        else:
            import r3_timing_model as TM
            c = TM.ptx_main_loop(text)
            c["wavefronts"] = V4.wavefronts_from_counts(c, a.warps)
        print(json.dumps({"path": kind, **c}, indent=1))
        return 0
    try:
        tiles = [V4.Tile.parse(s) for s in a.tiles.split(",") if s]
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED
    models = [m for m in a.models.split(",") if m]
    from moe.spec import MODEL_CONFIGS
    bad = [m for m in models if m not in MODEL_CONFIGS]
    if not tiles or not models or bad:
        print(f"REFUSED: --tiles and --models are required (unknown models: {bad})", file=sys.stderr)
        return exit_codes.REFUSED
    for line in plan_lines(tiles, models):
        print(line)
    if a.dry_run:
        print("DRY RUN: nothing compiled (exit 2)")
        return exit_codes.REFUSED
    if a.out is None:
        print("REFUSED: --out is required", file=sys.stderr)
        return exit_codes.REFUSED
    cache = Path(tempfile.mkdtemp(prefix="tile-census-triton-"))
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    entries, errors = [], []
    for t in tiles:
        for m in models:
            try:
                entries += compile_tile(m, t, a.seed)
            except Exception as exc:                          # noqa: BLE001
                errors.append(f"{m} {t.key}: {type(exc).__name__}: {exc}"[:300])
            print(f"  {t.key} {m}: {len([e for e in entries if e['tile'] == t.key and e['model'] == m])} configs", flush=True)
    bad_path = path_check(entries)
    doc = {"tool": "scripts/tile_census.py", "tiles": [t.key for t in tiles], "models": models,
           "triton_cache": str(cache), "entries": entries, "summary": summary(entries), "errors": errors,
           "path_mismatch": bad_path,
           "verdict": "REFUSED (path mismatch)" if bad_path else ("INCOMPLETE" if errors else "COMPLETE")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"RESULT: VALIDITY census_path {'FAIL' if bad_path else 'PASS'} {len(entries)} configs, "
          f"{len(bad_path)} path mismatches, {len(errors)} errors")
    if bad_path:
        return exit_codes.INVALID
    return exit_codes.DONE if not errors else exit_codes.INVALID


if __name__ == "__main__":
    sys.exit(main())
