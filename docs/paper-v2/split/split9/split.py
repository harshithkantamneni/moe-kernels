"""Per-kernel split of v8 / v9 on served prefill calls, both prefill tiles (2026-11-20, after the blind run).

For a seeded sample of distinct calls on TILE, prices each call through prodlib.price_v8 (v8, at the board's par) and
records the LAST price_parts output of that pricing (the one whose ms_sum is v8's ms: on 128x128 the sag-faced pass),
by wrapping every module binding of price_parts. Also prices v9. Measured parts come from the call's own kernels.
DESCRIPTIVE ONLY: nothing here changes v8, v9 or any registered value. Run with the frozen code (dc8b3f73) first on
sys.path:

    python split.py REPO RUN_DIR SAG_PAR_JSON MODEL_KEY TILE OUT_JSONL [N_SAMPLE]
"""
import importlib
import json
import random
import sys
from pathlib import Path

repo, run_dir, par_path, model_key, TILE, out_path = sys.argv[1:7]
n_sample = int(sys.argv[7]) if len(sys.argv) > 7 else 150
REPO = Path(repo)
for p in (REPO, REPO / "scripts", REPO / "scripts/prod"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import prodlib as PL  # noqa: E402

model = PL.MODELS[model_key]
rows = PL.read_calls(run_dir)
counts = PL.load_counts(run_dir)
keep = [i for i, r in enumerate(rows)
        if r.get("tile") == TILE and not r.get("warmup") and r.get("status") == "ok" and r.get("sm_mhz") == 1410]
print(f"{len(rows)} calls, {len(keep)} on {TILE} at 1410, not warmup", flush=True)

by_key = {}
for i in keep:
    r = rows[i]
    by_key.setdefault((r["T_pad"], r["counts_digest"], r["ids_digest"]), []).append(i)
keys = sorted(by_key)
random.Random(20261120).shuffle(keys)
keys = keys[:n_sample]
need_g = {rows[by_key[k][0]]["g"] for k in keys}
recs = PL.decode_shards(sorted((Path(run_dir) / "ring").glob("shard_*.npz")), model.cfg.top_k)
ids = {r["g"]: r["ids"] for r in recs if r["g"] in need_g}
del recs

sag128 = PL.load_board_par(par_path)
V8 = importlib.import_module(PL.PRICER_MODULE)
importlib.import_module("r3_timing_model_v9")

captured = []


def wrap(orig):
    def f(*a, **k):
        out = orig(*a, **k)
        captured.append(out)
        return out
    f._split_orig = orig
    return f


def num(x):
    return None if x is None else float(x)


with open(out_path, "w") as fo, V8.installed():
    # warm: the frozen chain loads its modules on first use, so price one call before patching
    k0 = keys[0]
    i0 = by_key[k0][0]
    PL.price_v8(model.cfg, rows[i0]["T_pad"], counts[i0], ids.get(rows[i0]["g"]), sag128=sag128)
    import types
    R5 = sys.modules["round5_chain"]
    seen, todo = set(), list(sys.modules.values()) + [v for v in R5.chain().values() if isinstance(v, types.ModuleType)]
    patched = []
    while todo:
        mod = todo.pop()
        if not isinstance(mod, types.ModuleType) or id(mod) in seen:
            continue
        seen.add(id(mod))
        todo.extend(v for v in vars(mod).values() if isinstance(v, types.ModuleType))
        fn = vars(mod).get("price_parts")
        if callable(fn) and not hasattr(fn, "_split_orig"):
            setattr(mod, "price_parts", wrap(fn))
            patched.append(mod)
    print("patched", [m.__name__ for m in patched], flush=True)
    assert patched, "no price_parts binding found"
    try:
        for n, k in enumerate(keys):
            idx = by_key[k]
            r0 = rows[idx[0]]
            captured.clear()
            a8 = PL.price_v8(model.cfg, r0["T_pad"], counts[idx[0]], ids.get(r0["g"]), sag128=sag128)
            last = captured[-1] if captured else None
            a9 = PL.price_v8(model.cfg, r0["T_pad"], counts[idx[0]], ids.get(r0["g"]), sag128=sag128,
                             module="r3_timing_model_v9")
            meas = [rows[i] for i in idx]
            med = lambda f: float(np.median([m[f] for m in meas])) / 1e6  # noqa: E731
            rec = {"T_pad": r0["T_pad"], "T": r0["T"], "n_calls": len(idx), "G": a8.get("G"), "tile": TILE,
                   "v8_ms": num(a8.get("ms")), "v9_ms": num(a9.get("ms")), "why": a8.get("why"),
                   "n_captured": len(captured),
                   "ms_sum": num(last and last.get("ms_sum")), "T0": num(last and last.get("T0")),
                   "s": num(last and last.get("s")), "b": num(last and last.get("b")),
                   "other": num(last and last.get("other")), "path": last and last.get("path"),
                   "w1_gemm": num(last and (last["w1"]["gemm"] + last["w1"]["dead"] + last["w1"]["const"])),
                   "w2_gemm": num(last and (last["w2"]["gemm"] + last["w2"]["dead"] + last["w2"]["const"])),
                   "w1_ms": med("w1_ns"), "w2_ms": med("w2_ns"), "align_ms": med("align_ns"),
                   "ks_ms": med("kernel_sum_ns")}
            fo.write(json.dumps(rec) + "\n")
            if n % 25 == 0:
                print(n, rec["T_pad"], rec["v8_ms"], rec["ms_sum"], rec["ks_ms"], flush=True)
    finally:
        for mod in patched:
            setattr(mod, "price_parts", getattr(mod, "price_parts")._split_orig)
print("done", flush=True)
