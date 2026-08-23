# What the first standard sweep says

840 rows, run `92572c5216fb`, commit `65ebea9`, one H200 SXM (700 W, 132 SMs, 60 MiB
L2), all 840 correctness-passing against the fp32 oracle. Baseline under test is
`torch.nn.functional.grouped_mm`, which on this build dispatches to CUTLASS
`bf16bf16_grouped_gemm_impl_sm90_sm100` — a Hopper-native, WGMMA grouped GEMM, not a
per-expert loop. Ceilings are the ones measured on this device the same afternoon
(`measured.yaml`), never datasheet numbers.

Sweep axes: 3 models x 7 token counts (1..4096) x 5 routing distributions x
{L2-cold, L2-warm} x {eager, CUDA-graph replay} x 2 stages.

---

## 1. The starting thesis was wrong, and the data says so plainly

Going in, the expected story was *"a fixed `BLOCK_M` degrades under skewed routing."*
It does not. Within a `(model, stage, token count)` cell, where the routing
distribution is the **only** thing that varies, skew makes the incumbent kernel
**faster**, not slower — up to 2x faster:

| model | T | routing | active experts | tile eff @128 | ms p50 | vs uniform |
|---|---:|---|---:|---:|---:|---:|
| deepseek-v3 | 16 | uniform | 100 | 0.010 | 1.9118 | 1.00x |
| deepseek-v3 | 16 | zipf:0.6 | 88 | 0.011 | 1.7031 | 0.89x |
| deepseek-v3 | 16 | dirichlet:0.3 | 68 | 0.015 | 1.3123 | 0.69x |
| deepseek-v3 | 16 | zipf:1.2 | 49 | 0.020 | **0.9408** | **0.49x** |
| deepseek-v3 | 16 | hot:0.5 | 94 | 0.011 | 1.7916 | 0.94x |

Note `hot:0.5` — the most *imbalanced* distribution by `max/mean` (32.0, the maximum
possible) — is nearly the *slowest*, while `zipf:1.2` at the same imbalance is the
fastest. Imbalance does not predict time. One number does.

## 2. The one number is active expert count, and it is a bandwidth law

Divide each cell's time by its active expert count and the routing distribution stops
mattering almost entirely:

| model | stage | T | active experts | us per active expert | spread across 5 routings |
|---|---|---:|---|---:|---:|
| deepseek-v3 | up | 16 | 49–100 | 19.21 | **1.5%** |
| deepseek-v3 | up | 64 | 118–225 | 18.46 | 5.1% |
| deepseek-v3 | up | 256 | 171–256 | 18.26 | 2.6% |
| qwen2-57b-a14b | up | 16 | 33–56 | 13.12 | 4.8% |
| qwen2-57b-a14b | up | 64 | 44–64 | 12.49 | 6.8% |
| qwen2-57b-a14b | up | 256 | 50–64 | 12.55 | 5.5% |

Runtime varies 2x across those cells. Runtime *per active expert* varies 1.5–7%.

The constant is not a scheduling cost — it is HBM. One DeepSeek expert's `w1` is
7168 x 4096 x 2 B = 58.7 MB; 58.7 MB / 19.21 us = 3055 GB/s, against a measured read
ceiling of 4390 GB/s. The same arithmetic lands in the same place for all three models
and both stages. Independently, the harness's own compulsory-bytes model agrees with
this hand calculation to within 2% on every row.

**At decode-shaped batch sizes an MoE grouped GEMM is not a GEMM. It is a weight
streaming problem.** Time is `(active experts x expert weight bytes) / achieved
bandwidth`, and routing skew matters only through the first factor. Skew helps because
concentrating tokens touches fewer experts, so fewer weight matrices cross the bus.

## 3. Padding MACs are free; padding *bytes* are not

Tile efficiency at `BLOCK_M=128` is **0.008 to 0.06** for T <= 256 — 94% to 99.2% of
the rows a tile computes are padding. That sounds like the bug to fix. It is not,
because it costs nothing measurable: across the cells in section 2, tile efficiency
varies 2–4x while time per active expert holds to within 7%. The tensor cores doing
127 rows of arithmetic on zeros are idle capacity that was going to be idle anyway,
waiting on weights.

Tile efficiency has a clean closed form here. When mean rows per expert is below
`BLOCK_M`, uniform routing gives exactly `mean_rows / BLOCK_M`: DeepSeek at T=1024 has
32 rows per expert on average and scores exactly 0.250 at `BLOCK_M=128` and 0.500 at 64.

The sign of skew's effect on tiling flips at the tile boundary:

| model | T | mean rows/expert | uniform eff@128 | hot:0.5 eff@128 |
|---|---:|---:|---:|---:|
| deepseek-v3 | 1024 | 32 (< BLOCK_M) | 0.250 | 0.243 (worse) |
| deepseek-v3 | 4096 | 128 (= BLOCK_M) | 0.692 | **0.853** (better) |
| qwen2-57b-a14b | 1024 | 128 (= BLOCK_M) | 0.696 | **0.865** (better) |

Below the boundary, skew strands more experts on 1–2 row tiles. At or above it,
uniform routing is the *worst* case: every expert lands just over a tile boundary and
pays for a second, nearly empty tile. That is the cliff to attack — but only once
you are past the point where bandwidth sets the time.

## 4. Where the headroom actually is

Against the measured read ceiling (4390.3 GB/s, the pattern explicitly calibrated as
the analogue of streaming expert weights), the incumbent CUTLASS kernel in the decode
regime runs at:

| model | stage | T | measured ms | bandwidth floor ms | x floor | % of read BW |
|---|---|---:|---:|---:|---:|---:|
| deepseek-v3 | up | 1 | 0.1814 | 0.1070 | 1.70x | 59% |
| deepseek-v3 | down | 1 | 0.1026 | 0.0535 | 1.92x | 52% |
| qwen2-57b-a14b | down | 1 | 0.0770 | 0.0335 | **2.30x** | **43%** |
| mixtral-8x7b | up | 16 | 0.7719 | 0.4285 | 1.80x | 56% |
| deepseek-v3 | up | 64 | 4.0687 | 3.0120 | 1.35x | 74% |

Across every T <= 64 cell the range is **1.35x to 2.30x the bandwidth floor**, and it
is worst at T=1 — exactly the shape decode inference runs at. That gap is the target.

Caveat stated up front: `ncu` is unavailable on a rented pod (`ERR_NVGPUCTRPERM`), so
this cannot separate *"the kernel moves more bytes than compulsory"* from *"the kernel
moves compulsory bytes at below-ceiling efficiency."* The harness reports this as
`implied_traffic_ratio`, a conditional bound, and it is labelled as such rather than
claimed as measured traffic. Either reading leaves the same headroom.

## 5. Two things that are confirmed dead ends

**CUDA graphs are not the lever.** 280 rows captured cleanly and passed replay
verification against the oracle. Median eager-minus-replay was **0.5 us** (p10 -3.8,
p90 +3.0) — noise. One grouped-GEMM launch against a 130 us to 7 ms kernel is not
where the time goes. The harness declined to even time the remaining 140 graph cells,
on the grounds that the launch overhead it could recover was under 1% of the roofline
minimum, which is the same conclusion reached before spending the measurement.

**L2 absorbs nothing.** Cold/warm ratio is 1.00 at every point measured. The working
set is 294 MB to 15 GB against a 60 MiB L2 — 5x to 250x oversubscribed. This is a
useful negative: it means the compulsory-traffic model has no cache reuse hiding
inside it, so the bandwidth arithmetic above is sound.

## 6. Measurement hygiene, recorded not hidden

79 of 840 rows tripped the throttle detector (SM clock dropping >5% within a cell).
They cluster entirely on token count — 0 rows at T <= 64, 14 at T=256, 31 at T=1024,
34 at T=4096 — and are spread evenly across models and stages. That is sustained power
draw on a 700 W board, not a model-specific artefact, and it affects only the
compute-bound region, not the decode-regime conclusions above. Those rows carry
`throttled=True` and stay in the CSV rather than being quietly dropped.

One caveat on the ridge point: the compute ceiling was measured at its settled
1500 MHz while the bandwidth patterns ran at 1980 MHz, so the recorded 166 FLOP/byte
ridge is conservative (185 if compute is clock-normalised to match). Everything in the
decode regime sits at AI 1–32, two orders of magnitude below either figure, so the
ambiguity does not touch any conclusion here.

---

## What this means for the kernel

The problem statement changes shape. It is not *"schedule ragged tiles better."* It is:

> Move `active_experts x expert_weight_bytes` from HBM at closer to 4390 GB/s than the
> 43–74% the incumbent achieves, while the M dimension is 1–64 rows per group.

Tiling still matters, but as a means to that end — the tile shape's job at T=1 is to
keep the memory pipe saturated, not to keep the tensor cores fed. The `BLOCK_M`
question only becomes a first-order question above the tile boundary (section 3),
which is prefill, not decode.
