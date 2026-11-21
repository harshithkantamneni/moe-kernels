"""Summarise split.py output: where v8's (and v9's) error on a prefill tile sits, GEMMs vs the rest.

    python split_summary.py FILE.jsonl [...]

Per file (distinct sampled calls, unweighted, as paper-v2 7.1): mean bias of the call, then each part's mean
predicted - measured in us and as % of the measured call. Measured non-GEMM = kernel sum - w1 - w2 (alignment,
act_and_mul, moe_sum). Predicted non-GEMM = price - predicted GEMMs (T0 + s + b + other on v8).
"""
import json
import sys

import numpy as np

for f in sys.argv[1:]:
    R = [json.loads(x) for x in open(f)]
    R = [r for r in R if r["v8_ms"] is not None and r["ms_sum"] is not None]
    a = lambda k: np.array([r[k] for r in R], dtype=float)  # noqa: E731
    ks, w1, w2 = a("ks_ms"), a("w1_ms"), a("w2_ms")
    p1, p2, v8, v9, v10 = a("w1_gemm"), a("w2_gemm"), a("v8_ms"), a("v9_ms"), a("v10_ms")
    assert np.allclose(a("ms_sum"), v8), "captured parts do not sum to v8's price"
    ng_m = ks - w1 - w2
    ng8, ng9, ng10 = v8 - p1 - p2, v9 - p1 - p2, v10 - p1 - p2
    us = lambda x: 1e3 * float(np.mean(x))  # noqa: E731
    pc = lambda x: 100 * float(np.mean(x / ks))  # noqa: E731
    print(f"== {f}: {len(R)} distinct calls, T {int(a('T').min())}..{int(a('T').max())}, "
          f"call {us(ks):.1f} us mean")
    print(f"   bias of the call      v8 {pc(v8 - ks):+.2f}%   v9 {pc(v9 - ks):+.2f}%   v10 {pc(v10 - ks):+.2f}%")
    print(f"   w1 GEMM   pred - meas {us(p1 - w1):+7.1f} us  {pc(p1 - w1):+.2f}%   (meas {us(w1):.1f} us, "
          f"pred/meas {float(np.mean(p1 / w1)):.3f})")
    print(f"   w2 GEMM   pred - meas {us(p2 - w2):+7.1f} us  {pc(p2 - w2):+.2f}%   (meas {us(w2):.1f} us, "
          f"pred/meas {float(np.mean(p2 / w2)):.3f})")
    print(f"   non-GEMM  v8 charges {us(ng8):.1f} us, v9 {us(ng9):.1f} us, v10 {us(ng10):.1f} us, measured {us(ng_m):.1f} us: "
          f"v8 {pc(ng8 - ng_m):+.2f}%, v9 {pc(ng9 - ng_m):+.2f}%, v10 {pc(ng10 - ng_m):+.2f}% of the call")
    print(f"   of v8's non-GEMM: T0 {us(a('T0')):.2f}  s {us(a('s')):+.2f}  b {us(a('b')):.2f}  other {us(a('other')):.2f} us;"
          f" paths {sorted(set(r['path'] for r in R))}")
