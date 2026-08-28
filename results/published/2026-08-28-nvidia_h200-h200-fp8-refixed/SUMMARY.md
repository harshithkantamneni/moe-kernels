# Results: 2026-08-28-nvidia_h200-h200-fp8-refixed

- rows: 9408
- correctness passed: 9408 / 9408
- arms: 1
    - `h200fp8redo` base: 9408 rows
- implementations: ['torch_scaled_grouped_mm_down', 'torch_scaled_grouped_mm_up']
- gpu: ['NVIDIA H200']
- commit: ['5687de86f530']
- **962 rows throttled (clocks dropped >5% mid-cell)**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | covers | model | tokens | ms p50 | TFLOP/s | AI |
|---|---|---|---:|---:|---:|---:|
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 1 | 0.0220 | 1.6 | 2.0 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 2 | 0.0250 | 2.8 | 3.0 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 4 | 0.0288 | 4.8 | 4.0 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 8 | 0.0321 | 8.6 | 6.4 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 16 | 0.0386 | 14.3 | 9.5 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 32 | 0.0468 | 23.7 | 13.5 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 64 | 0.0595 | 37.2 | 18.8 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 128 | 0.0627 | 70.7 | 34.3 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 256 | 0.0707 | 125.3 | 57.2 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 512 | 0.0812 | 218.2 | 107.1 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 1024 | 0.0876 | 404.6 | 156.1 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 2048 | 0.3184 | 222.6 | 263.0 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 4096 | 0.5544 | 255.7 | 399.9 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v2-lite | 8192 | 0.9646 | 293.9 | 540.7 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 1 | 0.0627 | 3.7 | 2.0 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 2 | 0.0733 | 6.4 | 3.2 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 4 | 0.0905 | 10.4 | 4.9 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 8 | 0.1268 | 14.8 | 6.4 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 16 | 0.1965 | 19.1 | 7.7 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 32 | 0.2505 | 30.0 | 11.8 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 64 | 0.3255 | 46.2 | 17.8 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 128 | 0.4227 | 71.1 | 26.8 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 256 | 0.5907 | 101.8 | 39.5 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 512 | 0.8811 | 136.5 | 56.1 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 1024 | 1.1548 | 208.3 | 82.6 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 2048 | 1.4746 | 326.2 | 149.2 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 4096 | 1.7880 | 538.1 | 220.6 |
| torch_scaled_grouped_mm_down | down_gemm | deepseek-v3 | 8192 | 2.6912 | 715.0 | 387.5 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 1 | 0.0827 | 2.8 | 2.0 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 2 | 0.0841 | 5.6 | 2.7 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 4 | 0.0842 | 11.2 | 5.3 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 8 | 0.1187 | 15.8 | 8.0 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 16 | 0.1191 | 31.6 | 15.9 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 32 | 0.1204 | 62.5 | 25.4 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 64 | 0.1206 | 124.7 | 50.4 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 128 | 0.1553 | 193.6 | 83.1 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 256 | 0.1911 | 314.7 | 123.1 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 512 | 0.2338 | 514.5 | 237.0 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 1024 | 0.3175 | 757.5 | 441.1 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 2048 | 0.5699 | 844.0 | 774.9 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 4096 | 1.0076 | 954.8 | 1246.6 |
| torch_scaled_grouped_mm_down | down_gemm | mixtral-8x7b | 8192 | 1.9552 | 984.1 | 1792.0 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 1 | 0.0426 | 3.4 | 2.0 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 2 | 0.0495 | 5.9 | 3.2 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 4 | 0.0696 | 8.4 | 4.0 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 8 | 0.0876 | 13.4 | 6.1 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 16 | 0.1025 | 22.9 | 9.8 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 32 | 0.1329 | 35.3 | 14.5 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 64 | 0.1475 | 63.7 | 25.2 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 128 | 0.1597 | 117.7 | 45.1 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 256 | 0.1922 | 195.5 | 77.7 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 512 | 0.2330 | 322.6 | 117.9 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 1024 | 0.2743 | 548.0 | 218.5 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 2048 | 0.4037 | 744.6 | 381.3 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 4096 | 0.7492 | 802.5 | 607.5 |
| torch_scaled_grouped_mm_down | down_gemm | qwen2-57b-a14b | 8192 | 1.4452 | 832.1 | 863.6 |
| torch_scaled_grouped_mm_up | up_gemm | deepseek-v2-lite | 1 | 0.0295 | 2.3 | 2.0 |
| torch_scaled_grouped_mm_up | up_gemm | deepseek-v2-lite | 2 | 0.0341 | 4.1 | 3.0 |
| torch_scaled_grouped_mm_up | up_gemm | deepseek-v2-lite | 4 | 0.0423 | 6.5 | 4.0 |
| torch_scaled_grouped_mm_up | up_gemm | deepseek-v2-lite | 8 | 0.0479 | 11.5 | 6.4 |
