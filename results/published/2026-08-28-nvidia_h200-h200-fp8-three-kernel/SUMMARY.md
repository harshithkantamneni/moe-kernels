# Results: 2026-08-28-nvidia_h200-h200-fp8-three-kernel

- rows: 19908
- correctness passed: 19908 / 19908
- arms: 5
    - `h200fp8b` vllm: 588 rows
    - `h200fp8full` vllm: 4704 rows
    - `h200fp8rest` base+sglang: 14112 rows
    - `h200fp8smoke2` base: 336 rows
    - `h200sglfp8smoke` sglang: 168 rows
- implementations: ['sglang_fused_experts', 'torch_scaled_grouped_mm_down', 'torch_scaled_grouped_mm_up', 'vllm_fused_experts']
- gpu: ['NVIDIA H200']
- commit: ['443945d85307', '52f5ddc77800', 'a3fa81c221d7', 'ab4b26b61832']
- **1442 rows throttled (clocks dropped >5% mid-cell)**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |
|---|---|---|---:|---:|---:|---:|
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 1 | 0.3298 | 0.3 | 2.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 2 | 0.3285 | 0.6 | 2.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 4 | 0.3260 | 1.3 | 2.2 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 8 | 0.3274 | 2.5 | 6.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 16 | 0.3106 | 5.4 | 8.3 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 32 | 0.3113 | 10.7 | 10.4 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 64 | 0.2184 | 30.4 | 12.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 128 | 0.2189 | 60.7 | 24.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 256 | 0.2171 | 122.5 | 47.8 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 512 | 0.2171 | 245.0 | 119.4 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 1024 | 0.2596 | 409.6 | 189.2 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 2048 | 0.3950 | 538.5 | 372.8 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 4096 | 0.6919 | 614.9 | 724.2 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 8192 | 1.1766 | 723.1 | 1369.7 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 1 | 0.3272 | 2.2 | 2.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 2 | 0.3273 | 4.3 | 2.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 4 | 0.3303 | 8.5 | 4.9 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 8 | 0.3399 | 16.6 | 5.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 16 | 0.3892 | 29.0 | 7.8 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 32 | 0.5157 | 43.7 | 11.9 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 64 | 0.6514 | 69.3 | 18.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 128 | 0.8754 | 103.1 | 27.3 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 256 | 1.2442 | 145.0 | 40.5 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 512 | 1.8773 | 192.2 | 58.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 1024 | 2.5163 | 286.8 | 86.9 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 2048 | 3.2279 | 447.2 | 163.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 4096 | 4.4015 | 655.9 | 253.4 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 8192 | 7.2792 | 793.2 | 501.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 1 | 0.2264 | 3.1 | 2.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 2 | 0.2327 | 6.1 | 4.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 4 | 0.2359 | 11.9 | 4.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 8 | 0.2343 | 24.1 | 8.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 16 | 0.2384 | 47.3 | 16.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 32 | 0.2599 | 86.8 | 25.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 64 | 0.2667 | 169.1 | 51.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 128 | 0.3725 | 242.2 | 85.2 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 256 | 0.4522 | 399.0 | 127.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 512 | 0.5917 | 609.8 | 254.5 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 1024 | 0.9541 | 756.3 | 506.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 2048 | 1.6418 | 879.1 | 1000.3 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 4096 | 3.0141 | 957.7 | 1955.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 8192 | 5.8764 | 982.4 | 3740.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 1 | 0.3306 | 1.3 | 2.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 2 | 0.3376 | 2.6 | 2.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 4 | 0.3360 | 5.2 | 3.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 8 | 0.3403 | 10.4 | 5.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 16 | 0.3227 | 21.8 | 9.8 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 32 | 0.3304 | 42.7 | 13.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 64 | 0.3122 | 90.3 | 25.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 128 | 0.3560 | 158.4 | 46.5 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 256 | 0.4503 | 250.5 | 81.7 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 512 | 0.5751 | 392.2 | 145.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 1024 | 0.7355 | 613.3 | 253.9 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 2048 | 1.2153 | 742.4 | 503.7 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 4096 | 2.2545 | 800.4 | 991.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 8192 | 4.1792 | 863.5 | 1920.0 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 1 | 0.0713 | 0.5 | 2.0 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 2 | 0.0722 | 1.0 | 2.2 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 4 | 0.0726 | 1.9 | 2.7 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 8 | 0.0735 | 3.8 | 5.3 |
