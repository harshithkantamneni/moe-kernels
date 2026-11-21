# v10 blind serving check: registration (2026-11-20)

Written and committed with v10 (`scripts/r3_timing_model_v10.py`) before any board for this check is rented. Nothing
below changes after the first launch. A repair to the run (a rerun after a kit fault) is reported with the result.

## What is tested

v10 changes one term: on the 128x128x64 prefill tile, v8's fitted intercepts T0 + s + b are replaced by v9's measured
non-GEMM constants (alignment, act_and_mul, moe_sum), with no new fitted value. Its decode is v8's and its 64x128x64
prefill is v9's, bit for bit (tests/test_timing_model_v10.py).

The form was chosen after seeing ERNIE-4.5-21B-A3B, Qwen3-30B-A3B, GLM-4.7-Flash, Trinity-Mini and Ling-mini-2.0
(v10's development data). The primary result is v10's prefill on the three models below, none of which any predictor
has seen (no cell page, no earlier serving run, not in Frontier's paper):

| key | repo @ pinned commit | E / top-k | I | H | MoE layers |
|---|---|---|---|---|---|
| lfm2-8b-a1b | LiquidAI/LFM2-8B-A1B @ 7ed1404f | 32 / 4 | 1792 | 2048 | 22 |
| granite-4.0-h-small | ibm-granite/granite-4.0-h-small @ b8c0982b | 72 / 10 | 768 | 4096 | 40 |
| phi-mini-moe | microsoft/Phi-mini-MoE-instruct @ f620b32c | 16 / 2 | 960 | 4096 | 32 |

## How it is run and scored

- One model per 1x GH200, `main.sh --profile blind` unchanged from the v9 check apart from the v10 price-check and the
  v10 column: sag1410, the 60 s gate, A1710 (decode) and A1410 (prefill) under graphs, levels 1,8,24,48,80,112,256,1024x3.
- Truth: each call's kernel sum from nsys node tracing. Scored on the VM by `offline.sh`.
- Predictors, all frozen before the run: v8 (`v8_ms`), v9 (`v9_ms`), v10 (`v10_ms`), the roofline, and Frontier's
  three arms (cli, simpath, cli-fills), trained per model on the same board's profiles.
- Metrics per model and regime (decode at 1710 MHz, prefill at 1410 MHz): MAPE and p95 exactly as `score_prod.py`
  reports them for paper-v2 sections 6 and 7, with 95% step
  bootstrap intervals (`score_prod.bootstrap_by_step`, 400 reps, seed 20261119).
- Also reported, descriptive only: the per-tile split on 128x128x64 and 64x128x64 (bias, w1, w2, non-GEMM).
- Decode for v10 is v8's decode; on these models it is a further blind decode check of the same predictions.
