# Results: 2026-08-28-nvidia_h200-h200-whole-layer

- rows: 9408
- correctness passed: 9408 / 9408
- arms: 1
    - `h200xcard` vllm: 9408 rows
- implementations: ['__pipeline__:vllm_fused_experts', 'vllm_fused_experts']
- gpu: ['NVIDIA H200']
- commit: ['873183a931d2']
- **986 rows throttled (clocks dropped >5% mid-cell)**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |
|---|---|---|---:|---:|---:|---:|
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 1 | 0.2490 | 0.4 | 1.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 2 | 0.2516 | 0.8 | 1.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 4 | 0.2795 | 1.5 | 1.7 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 8 | 0.2817 | 3.0 | 3.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 16 | 0.2825 | 5.9 | 1.9 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 32 | 0.2836 | 11.7 | 3.8 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 64 | 0.2897 | 23.0 | 9.4 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 128 | 0.2904 | 45.9 | 17.9 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 256 | 0.3272 | 81.5 | 31.3 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 512 | 0.3796 | 140.4 | 61.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 1024 | 0.4668 | 228.4 | 95.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 2048 | 0.7388 | 288.6 | 188.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 4096 | 1.2774 | 333.9 | 367.8 |
| __pipeline__:vllm_fused_experts | all | deepseek-v2-lite | 8192 | 2.2519 | 378.8 | 704.1 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 1 | 0.3392 | 2.1 | 1.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 2 | 0.4066 | 3.5 | 1.3 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 4 | 0.7029 | 4.0 | 1.3 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 8 | 1.3271 | 4.3 | 1.6 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 16 | 0.7715 | 14.7 | 3.9 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 32 | 1.0203 | 22.2 | 6.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 64 | 1.3387 | 33.9 | 9.0 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 128 | 1.7863 | 50.8 | 13.7 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 256 | 2.5291 | 71.7 | 20.3 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 512 | 3.8159 | 95.1 | 28.5 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 1024 | 5.3843 | 134.7 | 43.7 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 2048 | 6.9804 | 207.9 | 82.3 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 4096 | 8.6834 | 334.2 | 127.6 |
| __pipeline__:vllm_fused_experts | all | deepseek-v3 | 8192 | 15.4528 | 375.6 | 253.2 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 1 | 0.3149 | 2.2 | 1.0 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 2 | 0.3452 | 4.1 | 1.3 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 4 | 0.3487 | 8.1 | 2.7 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 8 | 0.4043 | 13.9 | 4.0 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 16 | 0.4095 | 27.5 | 8.0 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 32 | 0.5292 | 42.6 | 12.8 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 64 | 0.5283 | 85.4 | 25.6 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 128 | 0.7373 | 122.4 | 42.6 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 256 | 0.8608 | 209.6 | 63.9 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 512 | 1.1482 | 314.3 | 127.5 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 1024 | 1.9083 | 378.2 | 253.8 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 2048 | 3.2035 | 450.6 | 503.1 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 4096 | 5.8515 | 493.4 | 988.8 |
| __pipeline__:vllm_fused_experts | all | mixtral-8x7b | 8192 | 11.2695 | 512.3 | 1911.5 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 1 | 0.2561 | 1.7 | 1.0 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 2 | 0.2587 | 3.4 | 1.2 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 4 | 0.2885 | 6.1 | 2.0 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 8 | 0.3506 | 10.1 | 3.0 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 16 | 0.4130 | 17.1 | 4.9 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 32 | 0.5550 | 25.4 | 7.3 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 64 | 0.6572 | 42.9 | 12.8 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 128 | 0.7542 | 74.8 | 23.3 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 256 | 0.9427 | 119.8 | 40.9 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 512 | 1.1010 | 205.1 | 63.9 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 1024 | 1.3452 | 335.7 | 127.3 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 2048 | 2.2702 | 397.8 | 253.1 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 4096 | 4.2099 | 429.1 | 499.8 |
| __pipeline__:vllm_fused_experts | all | qwen2-57b-a14b | 8192 | 7.6855 | 470.0 | 975.4 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 1 | 0.1242 | 0.8 | 1.0 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 2 | 0.1247 | 1.7 | 1.0 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 4 | 0.1433 | 2.9 | 2.0 |
| vllm_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v2-lite | 8 | 0.1486 | 5.6 | 3.0 |
