# Results: 2026-08-26-nvidia_h200-full-three-way

- rows: 17640
- correctness passed: 17640 / 17640
- implementations: ['__pipeline__', 'sglang_fused_experts', 'torch_grouped_mm_down', 'torch_grouped_mm_up', 'vllm_fused_experts']
- gpu: ['NVIDIA H200']
- **1663 rows throttled**
    - NOTE 2026-09-03: the line above counts rows carrying `throttled`; what that flag detected was never stated on this page and is stated now. The count is unchanged and nothing in this arm was re-measured. Every row in this arm predates schema v5 (schema_version 3; no `clock_level_ok`, `clock_drift_ok` or `instrument` column), so the flag is the retired two-sample drift detector: the SM clock read at an idle instant before the cell and again after it, set when the second read was more than 5% below the first. That detected whether the FIRST read had caught the idle boost clock, not throttling under load: on the 2026-09-01 alpha-0558 arm it flagged 91% of vLLM rows above T=4096 while flagged and unflagged replicates of one cell timed at ratio 0.998 (moe/bench/timing.py, "CLOCKS ARE READ UNDER LOAD"; moe/bench/roofline.py `reference_clock`). Rows written after 00f3324 carry the LEVEL and DRIFT verdicts taken while the trials ran (`clock_level_ok`, `clock_drift_ok`, instrument `queue-deep/l2-flush/clock-under-load/v2`), and `throttled` on those rows means one of the two failed.
