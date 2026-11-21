# Per-kernel splits of served prefill calls (paper-v2 Tables 12 and 18)

DESCRIPTIVE. Each call in a seeded sample of 150 distinct served calls per tile is priced through the frozen model
code, and its price is split into the w1 GEMM, the w2 GEMM and the non-GEMM rest, beside the call's own measured
kernels. Nothing here changes a model, a constant or a registered value.

- `split9/` (2026-11-20, Table 12, and the 7.1 analysis): `split.py` at the v9 code (`dc8b3f73`) on the v8 serving
  run (ERNIE, Qwen3) and the v9 blind runs (Trinity-Mini, Ling-mini-2.0); `parts.py` and `t0fit.py` are the 7.1 root
  cause of the 64x128x64 intercept (`t0fit.py` reads the Mixtral-8x7B fit cells from a checkout at `~/code/moe-kernels`).
- `split10/` (2026-11-21, Table 18): the same split with a v10 price added, at the v10 code (`f1a0c0c4`), on the v10
  blind run (LFM2-8B-A1B, Phi-mini-MoE, granite-4.0-h-small). `run_all.sh` is the exact invocation (laptop paths:
  the frozen checkout, and a sparse checkout of the run branch's `A1410/` and `sag/` directories); `run_all.log` is
  its output. v10 is priced in a second pass, after the split's own `price_parts` wrappers are removed, because v10
  installs its own capture of the same binding.

Summarise any output with `python split_summary.py FILE.jsonl` (`split10/` prints v8, v9 and v10).
