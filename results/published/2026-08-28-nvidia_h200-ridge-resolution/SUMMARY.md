# Results: 2026-08-28-nvidia_h200-ridge-resolution

- rows: 22800
- correctness passed: 22800 / 22800
- arms: 14
    - `0b346aa1ea3e` base: 20 rows
    - `1300be94c335` sglang: 3528 rows
    - `2cdc4e12b905` base: 16 rows
    - `371c110e7dca` base: 16 rows
    - `3c48bab0f877` sglang: 8 rows
    - `3ec1d822ea9b` base+sglang+vllm: 2100 rows
    - `80a83dd6c2bc` base+sglang+vllm: 2100 rows
    - `915cc2fe28eb` base: 10584 rows
    - `92572c5216fb` base: 840 rows
    - `973ec3b9681a` vllm: 8 rows
    - `c8728dd73700` vllm: 3528 rows
    - `dba265af3ee1` base: 16 rows
    - `e61862e7cad7` base: 16 rows
    - `f2a140af4e34` base: 20 rows
- implementations: ['__pipeline__', 'sglang_fused_experts', 'torch_grouped_mm_down', 'torch_grouped_mm_up', 'vllm_fused_experts']
- gpu: ['NVIDIA H200']
- commit: ['11555d997e7a', '17efc6b10818', '4580223e3646', '65ebea9562b3', '66d4d530ea74', '839eb5e77540', 'c07c41ecdbba']
- **WARNING: some rows were measured from a dirty working tree**
- **2312 rows throttled (clocks dropped >5% mid-cell)**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |
|---|---|---|---:|---:|---:|---:|
| __pipeline__ | all | deepseek-v3 | 1 | 1.8765 | 0.4 | 1.0 |
| __pipeline__ | all | deepseek-v3 | 2 | 2.4690 | 0.6 | 1.6 |
| __pipeline__ | all | deepseek-v3 | 4 | 3.1328 | 0.9 | 2.5 |
| __pipeline__ | all | deepseek-v3 | 8 | 4.8555 | 1.2 | 3.2 |
| __pipeline__ | all | deepseek-v3 | 16 | 7.8225 | 1.4 | 3.9 |
| __pipeline__ | all | deepseek-v3 | 32 | 10.2045 | 2.2 | 5.9 |
| __pipeline__ | all | deepseek-v3 | 64 | 13.7662 | 3.3 | 8.9 |
| __pipeline__ | all | deepseek-v3 | 128 | 19.0669 | 4.8 | 13.5 |
| __pipeline__ | all | deepseek-v3 | 256 | 27.1295 | 6.7 | 20.0 |
| __pipeline__ | all | deepseek-v3 | 512 | 39.7217 | 9.1 | 28.4 |
| __pipeline__ | all | deepseek-v3 | 1024 | 56.7308 | 12.8 | 44.5 |
| __pipeline__ | all | deepseek-v3 | 2048 | 78.9673 | 18.4 | 76.5 |
| __pipeline__ | all | deepseek-v3 | 4096 | 115.1900 | 25.2 | 135.8 |
| __pipeline__ | all | deepseek-v3 | 8192 | 182.7678 | 31.8 | 233.5 |
| __pipeline__ | all | mixtral-8x7b | 1 | 1.6697 | 0.4 | 1.0 |
| __pipeline__ | all | mixtral-8x7b | 2 | 1.7957 | 0.8 | 2.0 |
| __pipeline__ | all | mixtral-8x7b | 4 | 2.5800 | 1.1 | 2.7 |
| __pipeline__ | all | mixtral-8x7b | 8 | 3.4635 | 1.6 | 4.0 |
| __pipeline__ | all | mixtral-8x7b | 16 | 3.7677 | 3.0 | 8.0 |
| __pipeline__ | all | mixtral-8x7b | 32 | 4.6772 | 4.8 | 12.7 |
| __pipeline__ | all | mixtral-8x7b | 64 | 5.1395 | 8.8 | 25.2 |
| __pipeline__ | all | mixtral-8x7b | 128 | 6.8993 | 13.1 | 41.6 |
| __pipeline__ | all | mixtral-8x7b | 256 | 9.6937 | 18.6 | 70.0 |
| __pipeline__ | all | mixtral-8x7b | 512 | 14.0475 | 25.7 | 118.7 |
| __pipeline__ | all | mixtral-8x7b | 576 | 14.8394 | 27.4 | 132.3 |
| __pipeline__ | all | mixtral-8x7b | 640 | 15.4851 | 29.1 | 164.4 |
| __pipeline__ | all | mixtral-8x7b | 704 | 16.3918 | 30.3 | 158.8 |
| __pipeline__ | all | mixtral-8x7b | 768 | 17.8414 | 30.3 | 193.3 |
| __pipeline__ | all | mixtral-8x7b | 1024 | 21.9366 | 32.9 | 221.2 |
| __pipeline__ | all | mixtral-8x7b | 2048 | 36.3885 | 39.7 | 430.1 |
| __pipeline__ | all | mixtral-8x7b | 4096 | 66.7773 | 43.2 | 627.9 |
| __pipeline__ | all | mixtral-8x7b | 8192 | 125.5799 | 46.0 | 958.5 |
| __pipeline__ | all | qwen2-57b-a14b | 1 | 1.3906 | 0.3 | 1.0 |
| __pipeline__ | all | qwen2-57b-a14b | 2 | 1.7109 | 0.5 | 1.6 |
| __pipeline__ | all | qwen2-57b-a14b | 4 | 2.5095 | 0.7 | 2.1 |
| __pipeline__ | all | qwen2-57b-a14b | 8 | 3.5056 | 1.0 | 3.0 |
| __pipeline__ | all | qwen2-57b-a14b | 16 | 4.4348 | 1.6 | 4.9 |
| __pipeline__ | all | qwen2-57b-a14b | 32 | 5.9116 | 2.4 | 7.3 |
| __pipeline__ | all | qwen2-57b-a14b | 64 | 7.3330 | 3.8 | 12.6 |
| __pipeline__ | all | qwen2-57b-a14b | 128 | 8.7003 | 6.5 | 22.7 |
| __pipeline__ | all | qwen2-57b-a14b | 256 | 10.9354 | 10.3 | 39.2 |
| __pipeline__ | all | qwen2-57b-a14b | 512 | 14.2367 | 15.9 | 71.2 |
| __pipeline__ | all | qwen2-57b-a14b | 1024 | 19.7863 | 22.8 | 127.6 |
| __pipeline__ | all | qwen2-57b-a14b | 1152 | 21.4421 | 23.7 | 141.1 |
| __pipeline__ | all | qwen2-57b-a14b | 1280 | 22.8819 | 24.7 | 151.7 |
| __pipeline__ | all | qwen2-57b-a14b | 1408 | 23.8878 | 26.0 | 166.5 |
| __pipeline__ | all | qwen2-57b-a14b | 1536 | 25.1546 | 26.9 | 178.6 |
| __pipeline__ | all | qwen2-57b-a14b | 2048 | 30.5188 | 29.6 | 220.2 |
| __pipeline__ | all | qwen2-57b-a14b | 4096 | 52.1312 | 34.6 | 324.9 |
| __pipeline__ | all | qwen2-57b-a14b | 8192 | 93.1566 | 38.8 | 475.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 1 | 0.2384 | 3.0 | 1.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 2 | 0.2480 | 5.7 | 1.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 4 | 0.3366 | 8.4 | 2.5 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 8 | 0.4587 | 12.3 | 3.2 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 16 | 0.7314 | 15.4 | 3.9 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 32 | 1.0226 | 22.1 | 6.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 64 | 1.5164 | 29.7 | 9.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 128 | 2.3146 | 39.0 | 13.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 256 | 3.8291 | 47.1 | 20.3 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 512 | 5.5155 | 65.4 | 29.0 |
