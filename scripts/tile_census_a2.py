#!/usr/bin/env python
"""The tile census for tile rental A2 (registration 2026-10-20-tileA2-census): compile only, no timing.

    python scripts/tile_census_a2.py --tiles 128x128x64w8s3,64x128x64w8s3 --models olmoe-1b-7b,deepseek-v2-lite \\
        --out $R/tile-census.json                     # on the VM (vLLM, Triton, a GPU)
    python scripts/tile_census_a2.py --dry-run --tiles ... --models ...   # the plan; exit 2

scripts/tile_census.py (pinned: tileA code_pins, the decode BASELINE) compiles every model at tread 1
(tile_census.py:140), and R3's arm_inputs refuses n x BLOCK_M x E / k when it is not whole, so on
rental A deepseek-v2-lite (E 64, k 6) read 0 configs at every tile (six errors). This module imports
tile_census's functions unchanged and calls its compile_tile with ONE thing different: the tread, the
smallest n with n x BLOCK_M x E divisible by k (`tread_of`; 1 for every model of rentals A and B but
deepseek-v2-lite, whose n is 3 at every tile, 3 x BM x 64 / 6 = 32 BM tokens: the tread rental B's
deepseek page times). The tread enters only through private_weight_reference.arm_inputs, which
compile_tile calls by module attribute; `at_tread(n)` hands it n for one call and restores it.

THE OLMoE n 1 / n 3 RULE (REVIEW-A2 8a, registered): Triton specialises integer arguments divisible
by 16, so tread 3 could change the specialisation key against tread 1. The unit therefore also
compiles OLMoE at n 3 on every tile and compares it with OLMoE's n 1 entry field by field (regs,
shared, num_warps, mma, occupancy, ptx_counts, config). Any difference: deepseek-v2-lite's entries
are NOT SCORED (the merge step of score_census_a2.py leaves them out, and rental B2's deepseek pages
stay NOT SCORED, as today). Every entry records its tread.

Exit codes as tile_census.py: 3 on an MMA path mismatch (VALIDITY census_path FAIL) or any compile
error, 0 complete, 2 a dry run or a refusal.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_timing_model_v4 as V4  # noqa: E402
import tile_census as TCEN  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

#: the fields the OLMoE n 1 / n 3 rule and A2's equality gate compare (every compiled fact v4 reads)
FIELDS = ("regs", "shared", "num_warps", "mma", "occupancy", "ptx_counts", "config")
CHECK_MODEL, CHECK_TREAD = "olmoe-1b-7b", 3
GATED_MODEL = "deepseek-v2-lite"


def tread_of(model: str, bm: int) -> int:
    """The smallest tread n with n x BLOCK_M x E divisible by k (R3's whole-token rule)."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    for n in range(1, cfg.top_k + 1):
        if (n * bm * cfg.num_experts) % cfg.top_k == 0:
            return n
    raise ValueError(f"{model} at BLOCK_M {bm}: no tread up to k makes a whole token count")


@contextlib.contextmanager
def at_tread(n: int):
    """private_weight_reference.arm_inputs called at tread n (compile_tile passes 1), restored on exit."""
    import private_weight_reference as PW
    orig = PW.arm_inputs

    def arm_inputs(cfg, _n, *a, **k):
        return orig(cfg, n, *a, **k)
    PW.arm_inputs = arm_inputs
    try:
        yield
    finally:
        PW.arm_inputs = orig


def compile_at(model: str, tile: V4.Tile, n: int, seed: int = 0) -> list[dict]:
    """ON THE VM: tile_census.compile_tile at tread n, each entry labelled with its tread."""
    with at_tread(n):
        got = TCEN.compile_tile(model, tile, seed)
    return [dict(e, tread=n) for e in got]


def facts(e: dict) -> dict:
    return {k: e.get(k) for k in FIELDS}


def compare(a: list[dict], b: list[dict]) -> dict:
    """Per (tile, GEMM) of `a`: SAME when `b` holds that (tile, GEMM) with every FIELDS value equal,
    else what differs. Missing on either side is a difference."""
    ka = {(e["tile"], e["gemm"]): e for e in a}
    kb = {(e["tile"], e["gemm"]): e for e in b}
    out = {}
    for k in sorted(set(ka) | set(kb)):
        if k not in ka or k not in kb:
            out[f"{k[0]}|{k[1]}"] = "DIFFERS: missing " + ("n 1" if k not in ka else "n 3")
            continue
        d = [f for f in FIELDS if facts(ka[k])[f] != facts(kb[k])[f]]
        out[f"{k[0]}|{k[1]}"] = "SAME" if not d else "DIFFERS: " + ", ".join(d)
    return out


def check_rule(entries: list[dict], check: list[dict]) -> dict:
    """The OLMoE n 1 / n 3 rule over a census's entries and its OLMoE n-3 entries."""
    n1 = [e for e in entries if e["model"] == CHECK_MODEL]
    if not n1 or not check:
        return {"verdict": "NOT RUN (no OLMoE entries at n 1 and n 3): deepseek-v2-lite's entries are NOT SCORED",
                "admit": False, "rows": {}}
    rows = compare(n1, check)
    ok = all(v == "SAME" for v in rows.values())
    return {"verdict": ("SAME on every (tile, GEMM): deepseek-v2-lite's tread-3 entries are admitted" if ok else
                        "DIFFERS: deepseek-v2-lite's entries are NOT SCORED (rental B2's deepseek pages stay NOT SCORED)"),
            "admit": ok, "rows": rows}


def plan_lines(tiles, models) -> list[str]:
    out = TCEN.plan_lines(tiles, models)
    out[0] = "TILE CENSUS (A2): compile only, no lock, nothing timed; each model at its smallest whole tread"
    for m in models:
        out.append(f"  {m}: tread " + ", ".join(f"{t.key} n {tread_of(m, t.bm)}" for t in tiles))
    if CHECK_MODEL in models:
        out.append(f"  and {CHECK_MODEL} again at n {CHECK_TREAD} on every tile (the n 1 / n 3 rule for {GATED_MODEL})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
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
    cache = Path(tempfile.mkdtemp(prefix="tile-census-a2-triton-"))
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    entries, errors, check = [], [], []
    for t in tiles:
        for m in models:
            n = tread_of(m, t.bm)
            try:
                entries += compile_at(m, t, n, a.seed)
            except Exception as exc:                          # noqa: BLE001
                errors.append(f"{m} {t.key} n {n}: {type(exc).__name__}: {exc}"[:300])
            print(f"  {t.key} {m} n {n}: {len([e for e in entries if e['tile'] == t.key and e['model'] == m])} configs", flush=True)
        if CHECK_MODEL in models:
            try:
                check += compile_at(CHECK_MODEL, t, CHECK_TREAD, a.seed)
            except Exception as exc:                          # noqa: BLE001
                errors.append(f"{CHECK_MODEL} {t.key} n {CHECK_TREAD} (the n 1 / n 3 rule): {type(exc).__name__}: {exc}"[:300])
    bad_path = TCEN.path_check(entries + check)
    rule = check_rule(entries, check)
    doc = {"tool": "scripts/tile_census_a2.py (tile_census.compile_tile at each model's smallest whole tread)",
           "tiles": [t.key for t in tiles], "models": models, "triton_cache": str(cache), "entries": entries,
           "olmoe_n3_entries": check, "olmoe_n1_n3_rule": rule, "summary": TCEN.summary(entries), "errors": errors,
           "path_mismatch": bad_path,
           "verdict": "REFUSED (path mismatch)" if bad_path else ("INCOMPLETE" if errors else "COMPLETE")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"RESULT: VALIDITY census_path {'FAIL' if bad_path else 'PASS'} {len(entries)} configs, "
          f"{len(bad_path)} path mismatches, {len(errors)} errors")
    print(f"RESULT: DESCRIPTIVE olmoe_n1_n3 {'SAME' if rule['admit'] else 'DIFFERS'} {rule['verdict']}")
    if bad_path:
        return exit_codes.INVALID
    return exit_codes.DONE if not errors else exit_codes.INVALID


if __name__ == "__main__":
    sys.exit(main())
