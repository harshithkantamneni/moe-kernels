# Results: 2026-08-22-first-smoke

- rows: 16
- correctness passed: 16 / 16
- gpu: ['NVIDIA H200']
- commit: ['66d4d530ea74']
- **WARNING: some rows were measured from a dirty working tree**

## Fastest per (impl, model, tokens), L2-flushed eager rows

| impl | model | tokens | ms p50 | TFLOP/s | AI | tile eff @128 |
|---|---|---:|---:|---:|---:|---:|
| torch_grouped_mm_down | mixtral-8x7b | 1 | 0.1052 | 2.2 | 1.0 | 0.008 |
| torch_grouped_mm_down | mixtral-8x7b | 128 | 0.3278 | 91.7 | 31.7 | 0.250 |
| torch_grouped_mm_down | toy | 1 | 0.0132 | 0.0 | 1.0 | 0.008 |
| torch_grouped_mm_down | toy | 128 | 0.0132 | 0.3 | 25.6 | 0.500 |
| torch_grouped_mm_up | mixtral-8x7b | 1 | 0.2088 | 2.3 | 1.0 | 0.008 |
| torch_grouped_mm_up | mixtral-8x7b | 128 | 0.6445 | 93.3 | 31.7 | 0.250 |
| torch_grouped_mm_up | toy | 1 | 0.0134 | 0.0 | 1.0 | 0.008 |
| torch_grouped_mm_up | toy | 128 | 0.0135 | 0.6 | 28.4 | 0.500 |
