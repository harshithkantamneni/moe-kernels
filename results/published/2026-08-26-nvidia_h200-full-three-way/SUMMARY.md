# Results: 2026-08-26-nvidia_h200-full-three-way

- rows: 3528
- correctness passed: 3528 / 3528
- gpu: ['NVIDIA H200']
- commit: ['11555d997e7a']
- **WARNING: some rows were measured from a dirty working tree**
- **360 rows throttled (clocks dropped >5% mid-cell)**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | model | tokens | ms p50 | TFLOP/s | AI | tile eff @128 |
|---|---|---:|---:|---:|---:|---:|
| sglang_fused_experts | deepseek-v3 | 1 | 0.2384 | 3.0 | 1.0 | 0.008 |
| sglang_fused_experts | deepseek-v3 | 2 | 0.2480 | 5.7 | 1.6 | 0.013 |
| sglang_fused_experts | deepseek-v3 | 4 | 0.3366 | 8.4 | 2.5 | 0.019 |
| sglang_fused_experts | deepseek-v3 | 8 | 0.4587 | 12.3 | 3.2 | 0.025 |
| sglang_fused_experts | deepseek-v3 | 16 | 0.7314 | 15.4 | 3.9 | 0.030 |
| sglang_fused_experts | deepseek-v3 | 32 | 1.0226 | 22.1 | 6.0 | 0.047 |
| sglang_fused_experts | deepseek-v3 | 64 | 1.5164 | 29.7 | 9.0 | 0.070 |
| sglang_fused_experts | deepseek-v3 | 128 | 2.3146 | 39.0 | 13.6 | 0.107 |
| sglang_fused_experts | deepseek-v3 | 256 | 3.8291 | 47.1 | 20.3 | 0.151 |
| sglang_fused_experts | deepseek-v3 | 512 | 5.5155 | 65.4 | 29.0 | 0.201 |
| sglang_fused_experts | deepseek-v3 | 1024 | 7.7592 | 93.0 | 43.5 | 0.311 |
| sglang_fused_experts | deepseek-v3 | 2048 | 9.8162 | 147.0 | 82.1 | 0.485 |
| sglang_fused_experts | deepseek-v3 | 4096 | 14.8694 | 194.1 | 127.4 | 0.845 |
| sglang_fused_experts | deepseek-v3 | 8192 | 25.2707 | 228.5 | 295.7 | 0.784 |
| sglang_fused_experts | mixtral-8x7b | 1 | 0.1877 | 3.8 | 1.0 | 0.008 |
| sglang_fused_experts | mixtral-8x7b | 2 | 0.1958 | 7.2 | 2.0 | 0.016 |
| sglang_fused_experts | mixtral-8x7b | 4 | 0.2732 | 10.3 | 2.7 | 0.021 |
| sglang_fused_experts | mixtral-8x7b | 8 | 0.3451 | 16.3 | 4.0 | 0.031 |
| sglang_fused_experts | mixtral-8x7b | 16 | 0.3541 | 31.8 | 8.0 | 0.062 |
| sglang_fused_experts | mixtral-8x7b | 32 | 0.4764 | 47.3 | 12.8 | 0.100 |
| sglang_fused_experts | mixtral-8x7b | 64 | 0.4754 | 94.9 | 25.6 | 0.200 |
| sglang_fused_experts | mixtral-8x7b | 128 | 0.6827 | 132.1 | 42.6 | 0.333 |
| sglang_fused_experts | mixtral-8x7b | 256 | 0.7855 | 229.7 | 63.9 | 0.500 |
| sglang_fused_experts | mixtral-8x7b | 512 | 1.0722 | 336.5 | 127.6 | 0.800 |
| sglang_fused_experts | mixtral-8x7b | 1024 | 1.8281 | 394.8 | 254.5 | 0.842 |
| sglang_fused_experts | mixtral-8x7b | 2048 | 3.1067 | 464.6 | 506.0 | 0.941 |
| sglang_fused_experts | mixtral-8x7b | 4096 | 5.6866 | 507.6 | 1000.3 | 0.970 |
| sglang_fused_experts | mixtral-8x7b | 8192 | 11.1642 | 517.1 | 1955.1 | 0.977 |
| sglang_fused_experts | qwen2-57b-a14b | 1 | 0.2351 | 1.9 | 1.0 | 0.008 |
| sglang_fused_experts | qwen2-57b-a14b | 2 | 0.2499 | 3.5 | 1.5 | 0.011 |
| sglang_fused_experts | qwen2-57b-a14b | 4 | 0.2551 | 6.9 | 2.1 | 0.017 |
| sglang_fused_experts | qwen2-57b-a14b | 8 | 0.2966 | 11.9 | 3.0 | 0.024 |
| sglang_fused_experts | qwen2-57b-a14b | 16 | 0.3631 | 19.4 | 4.9 | 0.038 |
| sglang_fused_experts | qwen2-57b-a14b | 32 | 0.5101 | 27.6 | 7.3 | 0.057 |
| sglang_fused_experts | qwen2-57b-a14b | 64 | 0.5972 | 47.2 | 12.8 | 0.100 |
| sglang_fused_experts | qwen2-57b-a14b | 128 | 0.7006 | 80.5 | 23.3 | 0.182 |
| sglang_fused_experts | qwen2-57b-a14b | 256 | 0.8779 | 128.5 | 40.9 | 0.296 |
| sglang_fused_experts | qwen2-57b-a14b | 512 | 1.0095 | 223.4 | 63.9 | 0.500 |
| sglang_fused_experts | qwen2-57b-a14b | 1024 | 1.2538 | 359.8 | 127.5 | 0.865 |
| sglang_fused_experts | qwen2-57b-a14b | 2048 | 2.1744 | 414.9 | 253.9 | 0.895 |
| sglang_fused_experts | qwen2-57b-a14b | 4096 | 4.0421 | 446.4 | 503.7 | 0.911 |
| sglang_fused_experts | qwen2-57b-a14b | 8192 | 7.5292 | 479.3 | 991.1 | 0.948 |
