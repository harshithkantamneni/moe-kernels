# Results: 2026-08-28-nvidia_a100_sxm4_80gb-a100-cross-card

- rows: 9408
- correctness passed: 9408 / 9408
- arms: 1
    - `a100xcard` vllm: 9408 rows
- implementations: ['__pipeline__:vllm_fused_experts', 'vllm_fused_experts']
- gpu: ['NVIDIA A100-SXM4-80GB']
- commit: ['11a32dc2e88f']
- **296 rows throttled (clocks dropped >5% mid-cell)**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |
|---|---|---|---:|---:|---:|---:|
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 1 | 0.3427 | 0.3 | 1.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 2 | 0.3324 | 0.6 | 1.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 4 | 0.3660 | 1.1 | 2.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 8 | 0.3753 | 2.2 | 2.7 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 16 | 0.3807 | 4.4 | 3.3 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 32 | 0.3891 | 8.6 | 6.9 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 64 | 0.5104 | 13.1 | 9.6 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 128 | 0.5837 | 22.8 | 17.9 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 256 | 0.6920 | 38.5 | 31.3 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 512 | 0.8257 | 64.6 | 61.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 1024 | 1.1207 | 95.1 | 95.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 2048 | 1.8332 | 116.3 | 188.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 4096 | 3.1454 | 135.6 | 367.8 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 8192 | 5.8023 | 147.0 | 704.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 1 | 0.4959 | 1.4 | 1.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 2 | 0.8645 | 1.6 | 1.3 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 4 | 1.6184 | 1.8 | 1.8 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 8 | 3.1324 | 1.8 | 2.6 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 16 | 1.7311 | 6.5 | 3.9 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 32 | 2.4156 | 9.4 | 6.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 64 | 3.1271 | 14.5 | 9.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 128 | 4.1990 | 21.6 | 13.7 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 256 | 6.0565 | 29.9 | 20.3 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 512 | 9.2575 | 39.2 | 29.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 1024 | 13.2080 | 54.9 | 44.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 2048 | 17.1653 | 84.5 | 83.2 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 4096 | 21.6362 | 134.1 | 127.6 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 8192 | 39.4709 | 147.0 | 253.2 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 1 | 0.4888 | 1.4 | 1.0 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 2 | 0.4983 | 2.8 | 2.0 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 4 | 0.6846 | 4.1 | 2.7 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 8 | 0.8969 | 6.3 | 4.0 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 16 | 0.9123 | 12.4 | 8.0 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 32 | 1.3941 | 16.2 | 12.8 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 64 | 1.4658 | 30.8 | 25.6 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 128 | 1.8458 | 48.9 | 42.6 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 256 | 2.2130 | 81.5 | 63.9 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 512 | 3.8801 | 93.0 | 127.5 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 1024 | 5.0828 | 142.0 | 253.8 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 2048 | 8.6017 | 167.8 | 503.1 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 4096 | 16.2749 | 177.4 | 988.8 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 8192 | 32.0476 | 180.2 | 1911.5 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 1 | 0.3478 | 1.3 | 1.0 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 2 | 0.5848 | 1.5 | 1.2 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 4 | 0.5755 | 3.1 | 2.1 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 8 | 0.7910 | 4.5 | 3.0 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 16 | 0.8913 | 7.9 | 4.9 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 32 | 1.2475 | 11.3 | 7.3 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 64 | 1.4035 | 20.1 | 12.8 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 128 | 1.6553 | 34.1 | 23.3 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 256 | 2.0078 | 56.2 | 40.9 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 512 | 2.6159 | 86.3 | 63.9 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 1024 | 3.4329 | 131.5 | 127.3 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 2048 | 6.0400 | 149.5 | 253.1 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 4096 | 11.6660 | 154.8 | 499.8 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 8192 | 22.1347 | 163.2 | 975.4 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 1 | 0.1462 | 0.7 | 1.0 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 2 | 0.1459 | 1.4 | 1.5 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 4 | 0.1774 | 2.3 | 2.0 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 8 | 0.1920 | 4.3 | 3.2 |
