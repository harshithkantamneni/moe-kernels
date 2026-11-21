"""Root cause of v8's 64x128x64w8s3 prefill bias: v8's per-GEMM price (floor, sag factor) vs the measured w1 / w2.

Reads one served run (A1410), keeps the calls on the 64x128 prefill tile, prices a sample of distinct calls through
prodlib.price_v8 exactly as the scorer does, and records v8's own per-GEMM parts by wrapping the installed v5_parts.
DESCRIPTIVE ONLY: nothing here changes v8 or any registered value.

    python parts.py RESULTS_RUN_DIR SAG_PAR_JSON MODEL_KEY OUT_JSONL [N_SAMPLE]
"""
import json
import random
import sys
from pathlib import Path

REPO = Path.home() / "code/moe-kernels"
for p in (REPO, REPO / "scripts", REPO / "scripts/prod"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import prodlib as PL  # noqa: E402
import importlib  # noqa: E402

run_dir, par_path, model_key, out_path = sys.argv[1:5]
n_sample = int(sys.argv[5]) if len(sys.argv) > 5 else 400
TILE = sys.argv[6] if len(sys.argv) > 6 else "64x128x64w8s3"

model = PL.MODELS[model_key]
rows = PL.read_calls(run_dir)
counts = PL.load_counts(run_dir)
keep = [i for i, r in enumerate(rows)
        if r.get("tile") == TILE and not r.get("warmup") and r.get("status") == "ok" and r.get("sm_mhz") == 1410]
print(f"{len(rows)} calls, {len(keep)} on {TILE} at 1410, not warmup", flush=True)

# one representative call per distinct (T_pad, counts, ids) key, then a seeded sample of keys
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

captured = []


def _safe(x, d=0):
    if d > 4:
        return repr(x)[:80]
    if isinstance(x, dict):
        return {str(k): _safe(v, d + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe(v, d + 1) for v in x][:16]
    if isinstance(x, (int, float, str, type(None), bool)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 16 else f"array{x.shape}"
    if isinstance(x, np.generic):
        return x.item()
    return repr(x)[:80]


def wrap(copy):
    def f(*a, **k):
        out = copy(*a, **k)
        if k.get("sag") != "none":
            captured.append(_safe(out))
        return out
    return f


with open(out_path, "w") as fo, V8.installed():
    plan = [(h, a, f, c) for h, a, f, c in V8._plan() if a == "v5_parts"]
    assert plan, "no v5_parts binding in v8's plan"
    for h, a, f, c in plan:
        V8._set(h, a, wrap(c))
    try:
        for n, k in enumerate(keys):
            idx = by_key[k]
            r0 = rows[idx[0]]
            captured.clear()
            ans = PL.price_v8(model.cfg, r0["T_pad"], counts[idx[0]], ids.get(r0["g"]), sag128=sag128)
            meas = [rows[i] for i in idx]
            rec = {"T_pad": r0["T_pad"], "T": r0["T"], "n_calls": len(idx), "layer_set": sorted({m["layer"] for m in meas})[:4],
                   "gridX_w1": r0["gridX_w1"], "gridX_w2": r0["gridX_w2"],
                   "v8_ms": ans.get("ms"), "why": ans.get("why"), "G": ans.get("G"), "reads": ans.get("reads"),
                   "parts": captured[-1] if captured else None, "n_captured": len(captured),
                   "w1_ms": float(np.median([m["w1_ns"] for m in meas])) / 1e6,
                   "w2_ms": float(np.median([m["w2_ns"] for m in meas])) / 1e6,
                   "align_ms": float(np.median([m["align_ns"] for m in meas])) / 1e6,
                   "ks_ms": float(np.median([m["kernel_sum_ns"] for m in meas])) / 1e6,
                   "nonzero_experts": int((np.asarray(counts[idx[0]]) > 0).sum())}
            fo.write(json.dumps(rec, default=float) + "\n")
            if n % 50 == 0:
                print(n, rec["T_pad"], rec["v8_ms"], rec["ks_ms"], flush=True)
    finally:
        for h, a, f, c in plan:
            V8._set(h, a, c)
print("done", flush=True)
