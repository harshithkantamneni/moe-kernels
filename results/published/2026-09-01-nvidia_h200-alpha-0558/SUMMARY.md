# Results: 2026-09-01-nvidia_h200-alpha-0558

- rows: 3696
- correctness passed: 3696 / 3696
- arms: 1
    - `eeea4eaef73e` base+sglang+vllm: 3696 rows
- implementations: ['sglang_fused_experts', 'torch_grouped_mm_down', 'torch_grouped_mm_up', 'vllm_fused_experts']
- gpu: ['NVIDIA H200']
- commit: ['d9190b276b8c']
- **WARNING: some rows were measured from a dirty working tree**
- **1797 rows throttled (clocks dropped >5% mid-cell)**
    - NOTE 2026-09-03: the line above counts rows carrying `throttled`, and its parenthetical "clocks dropped >5% mid-cell" is withdrawn: no detector ever measured a mid-cell drop. The count is unchanged and nothing in this arm was re-measured. Every row in this arm predates schema v5 (schema_version 4; no `clock_level_ok`, `clock_drift_ok` or `instrument` column), so the flag is the retired two-sample drift detector: the SM clock read at an idle instant before the cell and again after it, set when the second read was more than 5% below the first. That detected whether the FIRST read had caught the idle boost clock, not throttling under load: on the 2026-09-01 alpha-0558 arm it flagged 91% of vLLM rows above T=4096 while flagged and unflagged replicates of one cell timed at ratio 0.998 (moe/bench/timing.py, "CLOCKS ARE READ UNDER LOAD"; moe/bench/roofline.py `reference_clock`). Rows written after 00f3324 carry the LEVEL and DRIFT verdicts taken while the trials ran (`clock_level_ok`, `clock_drift_ok`, instrument `queue-deep/l2-flush/clock-under-load/v2`), and `throttled` on those rows means one of the two failed.

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |
|---|---|---|---:|---:|---:|---:|
