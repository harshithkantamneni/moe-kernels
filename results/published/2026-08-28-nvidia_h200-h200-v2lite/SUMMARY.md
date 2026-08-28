# Results: 2026-08-28-nvidia_h200-h200-v2lite

- rows: 5880
- correctness passed: 5880 / 5880
- arms: 1
    - `h200v2lite` base+sglang+vllm: 5880 rows
- implementations: ['__pipeline__', 'sglang_fused_experts', 'torch_grouped_mm_down', 'torch_grouped_mm_up', 'vllm_fused_experts']
- gpu: ['NVIDIA H200']
- commit: ['84ad70905b48']
- **WARNING: some rows were measured from a dirty working tree**
- **451 rows throttled (clocks dropped >5% mid-cell)**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |
|---|---|---|---:|---:|---:|---:|
| __pipeline__ | all | deepseek-v2-lite | 1 | 0.9465 | 0.1 | 1.0 |
| __pipeline__ | all | deepseek-v2-lite | 2 | 1.1454 | 0.2 | 1.5 |
| __pipeline__ | all | deepseek-v2-lite | 4 | 1.4261 | 0.3 | 2.2 |
| __pipeline__ | all | deepseek-v2-lite | 8 | 1.7805 | 0.5 | 3.2 |
| __pipeline__ | all | deepseek-v2-lite | 16 | 2.3100 | 0.7 | 4.8 |
| __pipeline__ | all | deepseek-v2-lite | 32 | 3.0372 | 1.1 | 6.8 |
| __pipeline__ | all | deepseek-v2-lite | 64 | 4.1652 | 1.6 | 9.4 |
| __pipeline__ | all | deepseek-v2-lite | 128 | 4.6000 | 2.9 | 17.3 |
| __pipeline__ | all | deepseek-v2-lite | 256 | 5.4163 | 4.9 | 29.5 |
| __pipeline__ | all | deepseek-v2-lite | 512 | 6.0871 | 8.8 | 54.7 |
| __pipeline__ | all | deepseek-v2-lite | 1024 | 7.5725 | 14.1 | 92.5 |
| __pipeline__ | all | deepseek-v2-lite | 2048 | 10.4913 | 20.3 | 153.7 |
| __pipeline__ | all | deepseek-v2-lite | 4096 | 16.0603 | 26.6 | 229.7 |
| __pipeline__ | all | deepseek-v2-lite | 8192 | 27.6508 | 30.8 | 308.9 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 1 | 0.2156 | 0.5 | 1.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 2 | 0.2276 | 0.9 | 1.2 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 4 | 0.2233 | 1.9 | 1.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 8 | 0.2263 | 3.7 | 2.7 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 16 | 0.2109 | 7.9 | 4.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 32 | 0.2090 | 15.9 | 6.9 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 64 | 0.2024 | 32.8 | 9.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 128 | 0.3177 | 41.8 | 17.8 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 256 | 0.3785 | 70.2 | 31.3 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 512 | 0.4329 | 122.8 | 61.2 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 1024 | 0.6043 | 176.0 | 108.8 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 2048 | 0.9697 | 219.4 | 215.8 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 4096 | 1.7560 | 242.3 | 410.1 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 8192 | 3.2258 | 263.8 | 781.8 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 1 | 0.0288 | 1.2 | 1.0 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 2 | 0.0301 | 2.3 | 1.5 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 4 | 0.0380 | 3.6 | 2.2 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 8 | 0.0451 | 6.1 | 3.2 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 16 | 0.0585 | 9.5 | 4.8 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 32 | 0.0733 | 15.1 | 6.8 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 64 | 0.0975 | 22.7 | 9.5 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 128 | 0.1017 | 43.6 | 17.5 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 256 | 0.1119 | 79.2 | 29.6 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 512 | 0.1258 | 140.8 | 57.2 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 1024 | 0.1305 | 271.5 | 86.1 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 2048 | 0.1651 | 429.4 | 156.1 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 4096 | 0.2931 | 483.6 | 263.0 |
| torch_grouped_mm_down | down_gemm | deepseek-v2-lite | 8192 | 0.5227 | 542.3 | 399.9 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 1 | 0.0402 | 1.7 | 1.0 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 2 | 0.0497 | 2.8 | 1.5 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 4 | 0.0594 | 4.7 | 2.2 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 8 | 0.0772 | 7.2 | 3.2 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 16 | 0.1001 | 11.1 | 4.8 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 32 | 0.1283 | 17.3 | 6.8 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 64 | 0.1725 | 25.7 | 9.5 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 128 | 0.1785 | 49.6 | 16.8 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 256 | 0.2038 | 86.9 | 29.9 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 512 | 0.2264 | 156.5 | 58.4 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 1024 | 0.2413 | 293.6 | 88.8 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 2048 | 0.3202 | 442.7 | 165.2 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 4096 | 0.5806 | 488.3 | 290.1 |
| torch_grouped_mm_up | up_gemm | deepseek-v2-lite | 8192 | 1.0139 | 559.2 | 466.1 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 1 | 0.1221 | 0.9 | 1.0 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 2 | 0.1231 | 1.7 | 1.1 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 4 | 0.1424 | 2.9 | 2.0 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 8 | 0.1486 | 5.6 | 3.0 |
