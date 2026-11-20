#!/usr/bin/env python
"""v9's non-GEMM constants, from the published GH200 launch-floor cell pages only (rentals 1 to 3, 2026-10-01..06,
vLLM 0.27.1): results/published/*/results/*launch-floor*/*/traces/parsed.json. No serving data is read.

Per traced call the kernels are classed by name: alignment (moe_align_block_size_kernel + count_and_sort_expert_tokens
_kernel, the block-scan path's two launches), activation (act_and_mul_kernel) and reduction (moe_sum_vec_kernel).

  K0_ACT, K0_SUM  the intercept of dur = k0 + bytes / rate, least squares per kernel pooled over every model and size
                  (activation bytes ids x 3F x 2, reduction bytes ids x H x 2 + ids / k x H x 2; Mixtral's tp8 slice
                  F / 8). v8 already prices the bytes part (`other`, B_OTHER / bw); v9 adds only the intercepts.
  AS_US, AS_US_PER_1K_IDS  alignment on vLLM's small-batch path (one launch, moe_align_block_size_small_batch_expert
                  _kernel): a straight line in ids over the pages' one-launch calls. Its growth is the kernel's own
                  (one CTA walks every id), so it is carried.
  A0              block-scan alignment: the median over models of the block-scan calls' summed alignment time at the smallest
                  page size (256 ids). Its growth with ids on these pages is not carried: the pages flush L2 before
                  every call, so alignment reads its ids cold, and its slope there rises with E (0.9 us per 1k ids at
                  E 8, 2.9 to 3.4 at E 32 to 40). See r3_timing_model_v9.py, KNOWN ASSUMPTIONS.

    python scripts/v9_nongemm_fit.py            # prints the three constants and writes nothing
    python scripts/v9_nongemm_fit.py --json OUT # also writes them with their inputs' sha256
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from moe.spec import MODEL_CONFIGS  # noqa: E402

PAGES = "results/published/*/results/*launch-floor*/*/traces/parsed.json"
TP = {"mixtral-8x7b-tp8": ("mixtral-8x7b", 8)}


def kernel_class(name: str) -> str | None:
    if "moe_align_block_size" in name or "count_and_sort_expert_tokens" in name:
        return "align"
    if "act_and_mul" in name:
        return "act"
    if "moe_sum" in name:
        return "sum"
    return None


def points(repo: Path = REPO) -> tuple[dict, list[str]]:
    """{"act"|"sum": [(model, ids, bytes, us)], "align": [(model, ids, n_align_kernels, us)]}, and the files read."""
    out: dict = {"act": [], "sum": [], "align": []}
    files = sorted(glob.glob(str(repo / PAGES)))
    for f in files:
        label = Path(f).parents[1].name
        cfg_key, tp = TP.get(label, (label, 1))
        cfg = MODEL_CONFIGS[cfg_key]
        H, F, k = cfg.hidden_size, cfg.intermediate_size // tp, cfg.top_k
        for key, v in json.loads(Path(f).read_text()).items():
            ids = int(re.search(r"-n(\d+)", key).group(1)) * 256
            nbytes = {"act": ids * 3 * F * 2, "sum": ids * H * 2 + ids // k * H * 2}
            for c in v["calls"]:
                per = {"act": 0.0, "sum": 0.0, "align": 0.0}
                n_al = 0
                for kk in c["kernels"]:
                    cls = kernel_class(kk["name"])
                    if cls:
                        per[cls] += float(kk["dur_us"])
                        n_al += cls == "align"
                for cls in ("act", "sum"):
                    out[cls].append((label, ids, nbytes[cls], per[cls]))
                out["align"].append((label, ids, n_al, per["align"]))
    return out, files


def fit(repo: Path = REPO) -> dict:
    P, files = points(repo)
    if not files:
        raise SystemExit(f"no launch-floor pages under {repo / PAGES}")
    res = {}
    for cls in ("act", "sum"):
        X = np.array([[1.0, p[2] / 1e9] for p in P[cls]])
        y = np.array([p[3] for p in P[cls]])
        (k0, us_per_gb), *_ = np.linalg.lstsq(X, y, rcond=None)
        res[f"K0_{cls.upper()}_US"] = round(float(k0), 3)
        res[f"{cls}_rate_GBps"] = round(1e6 / float(us_per_gb), 1)
        res[f"{cls}_rms_us"] = round(float(np.sqrt(np.mean((y - X @ np.array([k0, us_per_gb])) ** 2))), 3)
        res[f"{cls}_n"] = len(y)
    small = {}
    for label, ids, n_al, us in P["align"]:
        if n_al == 2 and ids == 256:
            small.setdefault(label, []).append(us)
    res["A0_US"] = round(st.median(st.median(v) for v in small.values()), 3)
    one = [(ids, us) for _label, ids, n_al, us in P["align"] if n_al == 1]
    X = np.array([[1.0, ids / 1e3] for ids, _ in one]); y = np.array([us for _, us in one])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    res["AS_US"], res["AS_US_PER_1K_IDS"] = round(float(a), 3), round(float(b), 3)
    res["align_small_rms_us"], res["align_small_n"] = round(float(np.sqrt(np.mean((y - X @ np.array([a, b])) ** 2))), 3), len(y)
    res["align_per_model_256ids_us"] = {m: round(st.median(v), 3) for m, v in sorted(small.items())}
    res["sources"] = {str(Path(f).relative_to(repo)): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    r = fit()
    print(json.dumps({k: v for k, v in r.items() if k != "sources"}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
