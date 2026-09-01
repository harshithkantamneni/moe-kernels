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

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |
|---|---|---|---:|---:|---:|---:|
