# Results: 2026-08-28-nvidia_h200-ridge-resolution

- rows: 6300
- correctness passed: 6300 / 6300
- arms: 3
    - `3ec1d822ea9b` base+sglang+vllm: 2100 rows
    - `80a83dd6c2bc` base+sglang+vllm: 2100 rows
    - `ridgedeepseek` base+sglang+vllm: 2100 rows
- implementations: ['__pipeline__', 'sglang_fused_experts', 'torch_grouped_mm_down', 'torch_grouped_mm_up', 'vllm_fused_experts']
- gpu: ['NVIDIA H200']
- commit: ['17efc6b10818', '7eecff427626', 'c07c41ecdbba']
- **1011 rows throttled (clocks dropped >5% mid-cell)**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |
|---|---|---|---:|---:|---:|---:|
| __pipeline__ | all | deepseek-v3 | 4096 | 115.3751 | 25.2 | 135.8 |
| __pipeline__ | all | deepseek-v3 | 4608 | 123.6631 | 26.4 | 150.3 |
| __pipeline__ | all | deepseek-v3 | 5120 | 131.5959 | 27.6 | 164.2 |
| __pipeline__ | all | deepseek-v3 | 5632 | 140.1170 | 28.5 | 177.1 |
| __pipeline__ | all | deepseek-v3 | 6144 | 147.9665 | 29.4 | 188.7 |
| __pipeline__ | all | mixtral-8x7b | 512 | 14.0654 | 25.7 | 118.7 |
| __pipeline__ | all | mixtral-8x7b | 576 | 14.8394 | 27.4 | 132.3 |
| __pipeline__ | all | mixtral-8x7b | 640 | 15.4851 | 29.1 | 164.4 |
| __pipeline__ | all | mixtral-8x7b | 704 | 16.3918 | 30.3 | 158.8 |
| __pipeline__ | all | mixtral-8x7b | 768 | 17.8414 | 30.3 | 193.3 |
| __pipeline__ | all | qwen2-57b-a14b | 1024 | 19.8177 | 22.8 | 127.6 |
| __pipeline__ | all | qwen2-57b-a14b | 1152 | 21.4421 | 23.7 | 141.1 |
| __pipeline__ | all | qwen2-57b-a14b | 1280 | 22.8819 | 24.7 | 151.7 |
| __pipeline__ | all | qwen2-57b-a14b | 1408 | 23.8878 | 26.0 | 166.5 |
| __pipeline__ | all | qwen2-57b-a14b | 1536 | 25.1546 | 26.9 | 178.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 4096 | 14.4549 | 199.7 | 127.4 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 4608 | 15.5378 | 209.0 | 174.3 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 5120 | 17.2288 | 209.5 | 193.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 5632 | 18.0794 | 219.6 | 174.8 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | deepseek-v3 | 6144 | 19.5301 | 221.7 | 228.7 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 512 | 1.0457 | 345.1 | 127.6 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 576 | 1.1480 | 353.6 | 143.5 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 640 | 1.1590 | 389.1 | 159.4 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 704 | 1.2831 | 386.7 | 175.3 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | mixtral-8x7b | 768 | 1.2919 | 418.9 | 191.2 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 1024 | 1.2401 | 363.7 | 127.5 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 1152 | 1.4696 | 345.3 | 143.4 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 1280 | 1.5720 | 358.7 | 159.2 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 1408 | 1.6693 | 371.6 | 175.0 |
| sglang_fused_experts | permute+up_gemm+act+down_gemm+unpermute | qwen2-57b-a14b | 1536 | 1.8386 | 368.0 | 190.9 |
| torch_grouped_mm_down | down_gemm | deepseek-v3 | 4096 | 2.7488 | 350.0 | 118.5 |
| torch_grouped_mm_down | down_gemm | deepseek-v3 | 4608 | 3.0793 | 351.5 | 132.1 |
| torch_grouped_mm_down | down_gemm | deepseek-v3 | 5120 | 3.1051 | 387.3 | 145.4 |
| torch_grouped_mm_down | down_gemm | deepseek-v3 | 5632 | 3.1545 | 419.3 | 158.5 |
| torch_grouped_mm_down | down_gemm | deepseek-v3 | 6144 | 3.1848 | 453.1 | 171.3 |
| torch_grouped_mm_down | down_gemm | mixtral-8x7b | 512 | 0.3350 | 359.0 | 123.1 |
| torch_grouped_mm_down | down_gemm | mixtral-8x7b | 576 | 0.3594 | 376.4 | 137.8 |
| torch_grouped_mm_down | down_gemm | mixtral-8x7b | 640 | 0.3636 | 413.5 | 152.3 |
| torch_grouped_mm_down | down_gemm | mixtral-8x7b | 704 | 0.3694 | 447.6 | 166.8 |
| torch_grouped_mm_down | down_gemm | mixtral-8x7b | 768 | 0.3723 | 484.5 | 181.1 |
| torch_grouped_mm_down | down_gemm | qwen2-57b-a14b | 1024 | 0.4228 | 355.5 | 117.9 |
| torch_grouped_mm_down | down_gemm | qwen2-57b-a14b | 1152 | 0.4768 | 354.7 | 131.3 |
| torch_grouped_mm_down | down_gemm | qwen2-57b-a14b | 1280 | 0.4829 | 389.1 | 144.5 |
| torch_grouped_mm_down | down_gemm | qwen2-57b-a14b | 1408 | 0.4916 | 420.4 | 157.4 |
| torch_grouped_mm_down | down_gemm | qwen2-57b-a14b | 1536 | 0.4971 | 453.6 | 170.1 |
| torch_grouped_mm_up | up_gemm | deepseek-v3 | 4096 | 5.3106 | 362.3 | 122.0 |
| torch_grouped_mm_up | up_gemm | deepseek-v3 | 4608 | 5.8432 | 370.5 | 136.5 |
| torch_grouped_mm_up | up_gemm | deepseek-v3 | 5120 | 5.8439 | 411.6 | 150.7 |
| torch_grouped_mm_up | up_gemm | deepseek-v3 | 5632 | 5.8807 | 449.9 | 164.9 |
| torch_grouped_mm_up | up_gemm | deepseek-v3 | 6144 | 5.9492 | 485.1 | 178.8 |
| torch_grouped_mm_up | up_gemm | mixtral-8x7b | 512 | 0.6853 | 351.0 | 123.6 |
| torch_grouped_mm_up | up_gemm | mixtral-8x7b | 576 | 0.7548 | 358.5 | 138.4 |
| torch_grouped_mm_up | up_gemm | mixtral-8x7b | 640 | 0.7562 | 397.6 | 153.2 |
| torch_grouped_mm_up | up_gemm | mixtral-8x7b | 704 | 0.7633 | 433.3 | 167.8 |
| torch_grouped_mm_up | up_gemm | mixtral-8x7b | 768 | 0.7701 | 468.5 | 182.2 |
| torch_grouped_mm_up | up_gemm | qwen2-57b-a14b | 1024 | 0.8250 | 364.4 | 120.7 |
| torch_grouped_mm_up | up_gemm | qwen2-57b-a14b | 1152 | 0.9466 | 357.3 | 134.8 |
| torch_grouped_mm_up | up_gemm | qwen2-57b-a14b | 1280 | 0.9567 | 392.8 | 148.7 |
| torch_grouped_mm_up | up_gemm | qwen2-57b-a14b | 1408 | 0.9632 | 429.2 | 162.4 |
| torch_grouped_mm_up | up_gemm | qwen2-57b-a14b | 1536 | 0.9646 | 467.5 | 176.0 |
