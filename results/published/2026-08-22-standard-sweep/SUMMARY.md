# Results: 2026-08-22-standard-sweep

- rows: 840
- correctness passed: 840 / 840
- gpu: ['NVIDIA H200']
- commit: ['65ebea9562b3']
- **79 rows throttled (clocks dropped >5% mid-cell)**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | model | tokens | ms p50 | TFLOP/s | AI | tile eff @128 |
|---|---|---:|---:|---:|---:|---:|
| torch_grouped_mm_down | deepseek-v3 | 1 | 0.1011 | 2.3 | 1.0 | 0.008 |
| torch_grouped_mm_down | deepseek-v3 | 4 | 0.2481 | 3.8 | 1.6 | 0.013 |
| torch_grouped_mm_down | deepseek-v3 | 16 | 0.5175 | 7.3 | 2.6 | 0.020 |
| torch_grouped_mm_down | deepseek-v3 | 64 | 1.1611 | 12.9 | 4.3 | 0.034 |
| torch_grouped_mm_down | deepseek-v3 | 256 | 1.6134 | 37.3 | 11.9 | 0.094 |
| torch_grouped_mm_down | deepseek-v3 | 1024 | 2.0895 | 115.1 | 38.1 | 0.281 |
| torch_grouped_mm_down | deepseek-v3 | 4096 | 2.7102 | 355.0 | 118.5 | 0.853 |
| torch_grouped_mm_down | mixtral-8x7b | 1 | 0.1046 | 2.2 | 1.0 | 0.008 |
| torch_grouped_mm_down | mixtral-8x7b | 4 | 0.1492 | 6.3 | 2.7 | 0.021 |
| torch_grouped_mm_down | mixtral-8x7b | 16 | 0.2595 | 14.5 | 5.3 | 0.042 |
| torch_grouped_mm_down | mixtral-8x7b | 64 | 0.3142 | 47.8 | 15.9 | 0.125 |
| torch_grouped_mm_down | mixtral-8x7b | 256 | 0.3154 | 190.7 | 62.7 | 0.500 |
| torch_grouped_mm_down | mixtral-8x7b | 1024 | 0.4954 | 485.5 | 237.0 | 0.842 |
| torch_grouped_mm_down | mixtral-8x7b | 4096 | 1.5583 | 617.4 | 774.9 | 0.941 |
| torch_grouped_mm_down | qwen2-57b-a14b | 1 | 0.0753 | 1.9 | 1.0 | 0.008 |
| torch_grouped_mm_down | qwen2-57b-a14b | 4 | 0.1358 | 4.3 | 1.8 | 0.014 |
| torch_grouped_mm_down | qwen2-57b-a14b | 16 | 0.2353 | 10.0 | 3.9 | 0.030 |
| torch_grouped_mm_down | qwen2-57b-a14b | 64 | 0.2988 | 31.4 | 11.5 | 0.091 |
| torch_grouped_mm_down | qwen2-57b-a14b | 256 | 0.3367 | 111.6 | 39.9 | 0.296 |
| torch_grouped_mm_down | qwen2-57b-a14b | 1024 | 0.4190 | 358.7 | 117.9 | 0.865 |
| torch_grouped_mm_down | qwen2-57b-a14b | 4096 | 1.0502 | 572.5 | 381.3 | 0.901 |
| torch_grouped_mm_up | deepseek-v3 | 1 | 0.1814 | 2.6 | 1.0 | 0.008 |
| torch_grouped_mm_up | deepseek-v3 | 4 | 0.4200 | 4.5 | 1.6 | 0.013 |
| torch_grouped_mm_up | deepseek-v3 | 16 | 0.9408 | 8.0 | 2.6 | 0.020 |
| torch_grouped_mm_up | deepseek-v3 | 64 | 2.2445 | 13.4 | 4.3 | 0.034 |
| torch_grouped_mm_up | deepseek-v3 | 256 | 3.1313 | 38.4 | 11.9 | 0.094 |
| torch_grouped_mm_up | deepseek-v3 | 1024 | 4.0643 | 118.4 | 38.4 | 0.281 |
| torch_grouped_mm_up | deepseek-v3 | 4096 | 5.3235 | 361.4 | 122.0 | 0.853 |
| torch_grouped_mm_up | mixtral-8x7b | 1 | 0.2021 | 2.3 | 1.0 | 0.008 |
| torch_grouped_mm_up | mixtral-8x7b | 4 | 0.2751 | 6.8 | 2.7 | 0.021 |
| torch_grouped_mm_up | mixtral-8x7b | 16 | 0.5892 | 12.8 | 5.3 | 0.042 |
| torch_grouped_mm_up | mixtral-8x7b | 64 | 0.6724 | 44.7 | 15.9 | 0.125 |
| torch_grouped_mm_up | mixtral-8x7b | 256 | 0.6145 | 195.7 | 62.9 | 0.500 |
| torch_grouped_mm_up | mixtral-8x7b | 1024 | 0.9260 | 519.5 | 238.9 | 0.842 |
| torch_grouped_mm_up | mixtral-8x7b | 4096 | 3.0515 | 630.6 | 796.4 | 0.955 |
| torch_grouped_mm_up | qwen2-57b-a14b | 1 | 0.1291 | 2.3 | 1.0 | 0.008 |
| torch_grouped_mm_up | qwen2-57b-a14b | 4 | 0.2630 | 4.5 | 1.8 | 0.014 |
| torch_grouped_mm_up | qwen2-57b-a14b | 16 | 0.4356 | 10.8 | 3.9 | 0.030 |
| torch_grouped_mm_up | qwen2-57b-a14b | 64 | 0.5689 | 33.0 | 11.6 | 0.091 |
| torch_grouped_mm_up | qwen2-57b-a14b | 256 | 0.6478 | 116.0 | 40.2 | 0.296 |
| torch_grouped_mm_up | qwen2-57b-a14b | 1024 | 0.8152 | 368.8 | 120.7 | 0.865 |
| torch_grouped_mm_up | qwen2-57b-a14b | 4096 | 2.0821 | 577.6 | 412.0 | 0.901 |
