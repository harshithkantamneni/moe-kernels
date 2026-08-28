# The study: what this project is now, and what each piece is for

Written 2026-08-27. Read this first if you have been away from the project.

## What changed

This started as a kernel project: build a grouped GEMM that beats the incumbent
by attacking the padding tax under skewed routing. **That premise is dead**, and
it was killed by this repo's own measurements. Padding is either zero or free:

- above batch 256, vLLM's autotuner sizes `BLOCK_SIZE_M` to exactly
  rows-per-expert, so padding waste is 0%
- below it, waste hits 50-100% and costs nothing, because 2 us of wasted
  arithmetic hides inside a 20 us weight read

So the project is now a **measurement study**. `moe/kernels/` stays empty unless
something in the results argues for filling it.

## The hypothesis

> The standard mental model of what limits an MoE grouped GEMM is wrong in four
> specific, measurable ways, and every error points the same direction: cost is
> attributed to arithmetic when it belongs to memory.

Each claim below is one probe at that.

## The four claims

**C1. The CUTLASS tile is 64, not 128, and it was never a choice.**
`torch.nn.functional.grouped_mm` on Hopper reports `TileShape M,N = 64,128`,
MMA atom `MMA_64x128x16_F32BF16BF16_SS`, schedule `Pingpong`, identical at
T = 1, 16, 256, 1024, 4096. Hopper's `wgmma.mma_async.m64nNk16` has **M fixed at
64** by the instruction set, so no selection was ever happening.
STATUS: **established.** Refutes a figure in three published write-ups,
including two of ours.

**C2. Arithmetic intensity is `2R/b`, independent of expert architecture.**
Every weight element is used once per row (2 FLOPs) and read once (`b` bytes),
so for `N` weight elements across any number of layers, `AI = 2NR / Nb = 2R/b`.
`N` is a SUM over layers and cancels, so shapes and layer counts are irrelevant.
For bf16 that is `AI = rows per expert`.

STATUS: **ESTABLISHED, on a prediction that could have refuted it. Measured
2026-08-28, 19,908 fp8 rows across three kernels, published at
`results/published/2026-08-28-nvidia_h200-h200-fp8-three-kernel/`.**

THE PREDICTION THAT WAS WRONG WAS MINE, NOT C2'S. The fp8 sweep was built to
test a "2x crossing shift": halve the weight bytes, halve the batch at which a
model crosses its ridge. But the ridge is `peak_FLOPS / bandwidth`, and bf16 ->
fp8 halves `b` AND doubles `peak_FLOPS`, because the same silicon runs fp8
tensor cores at twice the bf16 rate:

    fp8:   2R/1 = 2 * ridge_bf16   ->   R = ridge_bf16
    bf16:  2R/2 =     ridge_bf16   ->   R = ridge_bf16

The same rows per expert, so the same crossing. Both sides of the roofline scale
together and their intersection does not move. The 2x figure came from holding
the ridge at its bf16 value while changing the format.

MEASURED. Crossings are recovered from TIME (`moe/bench/crossing.py`: the slope
of `log ms` against `log T` passing 0.5, floored at the saturation batch), never
from the byte model, so the prediction is refutable.

fp8/bf16 crossing ratio -- corrected theory says 1.00, the retracted 2x says 0.50:

| model            | vLLM | SGLang |
|------------------|-----:|-------:|
| mixtral-8x7b     | 1.25 |   1.16 |
| qwen2-57b-a14b   | 1.11 |   1.15 |
| deepseek-v2-lite | 1.06 |   1.16 |
| deepseek-v3      | 1.07 |   1.23 |

**1.15 +/- 0.07** over eight measurements from two unrelated kernels, which also
agree with EACH OTHER on absolute bf16 crossings to within a few percent (454 vs
464, 810 vs 819, 3240 vs 3048). The traffic reduction is real and appears in the
TIME rather than the crossing: mixtral at T=512 goes 1.1567 -> 0.6383 ms, 0.55x.

THE THIRD KERNEL IS RETRACTED. `torch_scaled_grouped_mm_*` gives 0.44 +/- 0.13,
which looks like a confirmation of the 2x prediction and is not. It is an
artefact of this harness: the span quantises activations INSIDE the timed region,
because `_scaled_grouped_mm` needs both operands in fp8 while the harness hands
out bf16 activations for vLLM's sake. The giveaway is direct -- deepseek-v2-lite
`torch_scaled_grouped_mm_up` at T=8192 is 1.9855 ms against bf16's 1.0503, so
fp8 is 1.89x SLOWER on the same GEMM. That cannot be a dtype effect. The added
work scales with tokens, which biases the crossing early. Fixable by hoisting
the quantisation into the prologue; until then this span cannot test C2.

WHERE THE REMAINING OFFSET LIVES, and it is not the hardware. bf16
measured/predicted, split by how much of the layer the span covers:

| model            |  F/H | vLLM | SGLang | torch_up | torch_down |
|------------------|-----:|-----:|-------:|---------:|-----------:|
| mixtral-8x7b     | 3.50 | 0.71 |   0.72 |     1.46 |       0.64 |
| qwen2-57b-a14b   | 0.71 | 0.63 |   0.64 |     1.00 |       1.18 |
| deepseek-v2-lite | 0.69 | 0.54 |   0.60 |     1.05 |       1.19 |
| deepseek-v3      | 0.29 | 0.63 |   0.59 |     1.26 |       1.27 |
| **mean**         |      | **0.63 +/- 0.06** || **1.13 +/- 0.24** ||

The FIVE-STAGE kernels sit at 0.63 with a tight spread. The ONE-STAGE grouped
GEMM sits at 1.13: `2R/b` predicts it about right. Same hardware, same ridge, so
the offset is not the kernel falling short of datasheet peak -- it belongs to the
extra stages, whose permute, activation and unpermute traffic the weights-only
model never counted.

READ THOSE ABSOLUTES WITH THE RIDGE BAND, NOT AS FIXED. Two calibrations of the
same H200 give:

    bandwidth   4377.2 -> 4374.5 GB/s       0.06% apart
    bf16 GEMM    701.6 ->  770.9 TFLOP/s    9.9% apart
    ridge        160.3 ->  176.2 FLOP/byte  9.9% apart

The bandwidth reproduces; the compute term does not, because the GEMM runs at
whatever clock the thermal state allows -- 1530 MHz on the second run against a
1830 MHz datasheet boost. So the ridge is a RANGE, wider than the +/-1.5% this
document previously quoted, and the whole table moves with it: at ridge 176.2 the
means are 0.58 and 1.03 rather than 0.63 and 1.13.

WHAT SURVIVES THE BAND is the comparison between span extents. Both sides divide
by the same predicted crossing, so the ridge cancels ALGEBRAICALLY: five-stage
over one-stage is 0.561 at ridge 160.3, at 176.2, and at any other value
(`tests/test_ridge_band.py`). A five-stage span crosses at 56% of the batch a
one-stage span does, and no calibration uncertainty touches that.

So the claim is the SEPARATION, and the absolutes are quoted with their band.

THE fp8 RIDGE RATIO IS MEASURED NOW, AND IT IS NOT 2. The `ridge_fp8 =
2 x ridge_bf16` above is the datasheet relationship. Measured on the H200:

    bf16    770.9 TFLOP/s at 1530 MHz   93.2% of that clock's peak
    fp8    1409.2 TFLOP/s at 1740 MHz   74.9% of that clock's peak

The two GEMMs ran at different clocks, so the 1.828 the calibration records
conflates format with clock. Per clock it is 1.607: fp8 reaches materially less
of its own peak than bf16 does of its.

That makes C2's prediction WORSE, not better, and the honest statement is that
this calibration cannot discriminate:

    ridge ratio 2.000 (datasheet)          predicts crossing_fp8/bf16 = 1.000
    ridge ratio 1.828 (this calibration)   predicts 0.914
    ridge ratio 2.008 (vs the older bf16)  predicts 1.004
    MEASURED, two production kernels                1.150 +/- 0.07

The prediction spans 0.914 to 1.004 across two measurements of one machine,
because the bf16 denominator moves 9.9%. Measured is 1.150. Dtype-invariance
holds against the naive 0.50 by a wide margin either way; the ARITHMETIC is not
pinned down to better than about 10% until the GEMM clock is controlled.

NOT CLAIMED. An earlier reading had the one-stage deviation ordered by expert
shape (`F/H`), matching `ridge.py`'s prediction that mixtral would deviate most.
That ordering came from an input set that double-counted a superseded arm. On
the canonical set it is 1.05, 1.09, 1.12, 1.27 against F/H of 3.50, 0.71, 0.69,
0.29 -- monotonic in the means, but mixtral's internal disagreement makes its
mean unreliable and the effect is too weak to assert. See
`moe/bench/published.py` for why the input set now defends itself.

**C3. vLLM's decode path may not use Hopper's tensor core at all.**
vLLM's tuned H200 config sets `BLOCK_SIZE_M = 16` for every batch size from 1 to
256, the whole decode range, and tracks rows-per-expert exactly above it. 16 is
below WGMMA's fixed M of 64. If Triton therefore falls back to `mma.sync`, then
the decode path runs on Ampere-era tensor cores on Hopper silicon, and correctly
so: the tensor core idles waiting on weights either way, while a small tile buys
occupancy and more memory requests in flight.
STATUS: **ESTABLISHED, measured 2026-08-27.** `scripts/check_mma_path.sh` on a
real deepseek-v3 T=16 cell: both compiled `fused_moe_kernel` variants show
`wgmma=0`, `mma.sync=16`, and every one of the 32 tensor-core instructions is
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. No warpgroup MMA anywhere.
The strongest single piece of evidence for the hypothesis: an autotuner with no
theory gave away Hopper's headline feature because, memory-bound, it does not
matter.

**C4. STREAM-style calibration may understate achievable read bandwidth.**
A production kernel sustained 4483.4 GB/s where `calibrate_hardware.py` reaches
4389.4 GB/s on a pure read. If the calibration is measuring its own achieved rate
rather than a ceiling, every percent-of-ceiling figure computed that way is
pessimistic, ours and everyone else's.
STATUS: **CONFIRMED and closed, 2026-08-28.** The cause was the SHAPE of the
read, not the clock. `calibrate.py` measured it as `torch.sum(a, dim=0)` on a 1-D
buffer into a scalar: a full tree reduction, which bounds on ATen's reduction
rather than on DRAM. Reducing a 2-D view along the contiguous axis instead gives
thousands of independent reductions and no global combine.

Measured on the same card before and after the fix:

    read ceiling   4389.3  ->  4470.7 GB/s     +1.85%

That closes the anomaly. The 83 rows implied 4483.4 GB/s, which is 102.4% of the
named `triad` ceiling and looked impossible, but is **100.28% of the corrected
read ceiling**: at the ceiling within three parts in a thousand, not above it.
Those kernels were running at essentially 100% of achievable read bandwidth on
pure weight streaming, which is a strong result rather than a broken one.

A clock hypothesis was tested first and REFUTED. Settling under a memory load
instead of a matmul is correct in itself, and the memory settle converges at
1980 MHz as designed, but it moved triad by +0.05% and read by -0.00%: the
existing two-pass warmup had already handled the clock.

Confirmed independently on an A100, where the flaw is unmissable: the same
benchmark reports read at 1770 GB/s against triad's 1798, and triad moves three
times the bytes. `calibrate.py` detects that case and refuses to name read as a
ceiling. On the H200 it hid, because read landed just above triad.

**Caveat on the ridge.** Bandwidth is stable to 0.005% across three
calibrations (4377.0 / 4377.0 / 4377.2). The GEMM is not: it lands at 1560 or
1845 MHz depending on the run, giving 701.6 or 712.4 TFLOP/s and a ridge of
160.3 or 162.8. The clock normalisation works, but the ridge is a +/-1.5%
quantity and should be quoted as approximate. Mixtral's predicted crossing spans
641 to 651 across that range, all inside the same measured bracket, so C2 is
unaffected.

## What a whole MoE layer costs, and how much of it is routing

MEASURED 2026-08-28, `results/published/2026-08-28-nvidia_h200-h200-whole-layer/`,
9,408 rows. The first complete-layer measurement in this project: every framework
span covers five of six stages and leaves the router out, so until now the study
could not say what a full layer costs. `__pipeline__:vllm_fused_experts` times
`ref_router` plus the fused kernel as one cell.

THE ROUTER IS A FIXED COST, and at decode it is a third of the layer:

| model            | T=1 layer | router | share | T=4096 share |
|------------------|----------:|-------:|------:|-------------:|
| mixtral-8x7b     |    0.2707 | 0.0814 | 30.1% |         1.7% |
| qwen2-57b-a14b   |    0.2019 | 0.0773 | 38.3% |         3.4% |
| deepseek-v2-lite |    0.1591 | 0.0762 | 47.9% |         7.1% |
| deepseek-v3      |    0.2867 | 0.0974 | 34.0% |         4.9% |

The absolute cost barely moves with batch -- roughly 0.05 to 0.10 ms everywhere,
which is launch and dispatch overhead for a matmul and a top-k, not work. So its
SHARE collapses as the batch grows while the number itself does not. At T=1,
between 30% and 48% of an MoE layer is deciding which experts to use.

That is the same story the rest of the study tells from the other end: at decode
nothing is FLOP-bound, and what dominates is whatever does not scale.

CAVEAT, and it bounds the claim. This is `ref_router`: a PyTorch matmul plus a
top-k, the harness's reference. A production router is fused and faster, so 30
to 48% is an UPPER bound on the share, not a measurement of what vLLM spends. It
does establish that a whole-layer number is not the fused span's number, and how
much is missing.

THE CROSSING IS UNMOVED, which is the confirmation. Same run, ridge 176.2:

| model            | predicted | span | whole layer |
|------------------|----------:|-----:|------------:|
| mixtral-8x7b     |       705 |  543 |         549 |
| qwen2-57b-a14b   |      1410 |  914 |         960 |
| deepseek-v2-lite |      1879 |  897 |        1006 |
| deepseek-v3      |      5638 | 3474 |        3375 |

One to twelve percent apart, mostly under five. A fixed cost added to a
bandwidth-driven turning point should not move it, and it does not. The
five-stage ratio here (0.63 mean) also reproduces the 0.58 to 0.63 seen in the
earlier sweeps, which is the run-to-run spread the ridge band predicts.

## Two analysis bugs that had to be fixed before any of this was readable

Both produced confident, wrong numbers rather than errors, which is the failure
mode this project is most exposed to.

**A row that was never timed is not a measurement of zero.** A skipped or
uncapturable graph mode still writes a row, with `ms_p50` left at its 0.0
default; `run_all.sh --dry-run` says so in as many words. Feeding those to a
median dragged it toward zero and the first fp8 report concluded deepseek-v3
crossed at 2 tokens. 2,356 of 11,264 rows were untimed. `crossing.timed_rows`
now drops them and the report states how many, rather than silently using fewer.

**Below `E/k`, a batch does not touch every expert.** mixtral at T=1 reaches 2
of 8, so active experts and weight traffic grow WITH the batch and time rises
nearly linearly. That slope crosses 0.5 for a reason unrelated to the ridge, and
the detector stopped there: mixtral reported 5 tokens, deepseek-v3 reported 25.
`2R/b` assumes all E experts are active, so those points are outside the claim's
domain rather than evidence against it. `crossing_from_points` now floors at the
saturation batch, which `ridge.saturation_batch` already computed.

## Supporting results, already measured

- **Span extent is a trap.** `grouped_mm` covers 1 of 6 canonical stages;
  `fused_experts` covers 5. Comparing their milliseconds compares a GEMM to a
  fused block. Recorded per row in `covers`, enforced by `scripts/compare.py`.
- **Distance from the compulsory byte floor**, L2-cold eager unthrottled,
  n=3225: vLLM 1.16x, SGLang 1.17x, torch `grouped_mm` 1.62x, reference 12.43x.
- **Bimodality is real but cheap.** At deepseek T=4096 zipf:2.0, 24 of 248 active
  experts hold 89% of the rows. One global tile costs only 1.00x-1.18x of ideal
  weight traffic, so per-expert tiling is a 5-15% target, not a 2x one.
- **83 rows report the impossible.** All vLLM, all deepseek-v3, all T in
  {16,32,64}, 56 unthrottled. Peak 4483.4 GB/s = 102.1% of measured read, and
  **zero rows anywhere exceed the pin rate**. Explanation C4.
- **Routing is not reproducible off the GPU.** `cli.build_routing_source` passes
  `device=args.device`, so a GPU run samples with a CUDA generator. Same seed on
  CPU gives a different expert assignment. Row totals match, distributions do
  not. NOT fixed; fixing it changes future routing relative to published rows.

## The two additions that make it a study rather than a report

The sweep is 3 models x 14 token counts x 7 routings x 3 seeds, and **1 dtype,
1 device**. Both degenerate axes are where the value is.

**A. fp8 — a prediction test.** C2 says `AI = 2R/b`. Halving `b` must double
arithmetic intensity and halve the ridge crossing: deepseek-v3 from ~5,500 tokens
to ~2,750. That is falsifiable, quantitative, and derived before being measured.
Needs the baselines to declare and support fp8; `spec.py` already knows
`fp8_e4m3` and `fp8_e5m2` at 1 byte and the byte model is already
dtype-parametric.

**B. A second device — a generalisation test.** The crossing scales with the
ridge. An A100's ridge is 153.02 Op/B against the H200's 206.15 (Yun et al.
Table I). Running the same sweep on an A100 and checking the crossings move as
predicted is the cheapest possible falsification test, and the device-agnostic
work is already done and unused.

## What runs where

| piece | where | what it does |
|---|---|---|
| `scripts/run_all.sh` | pod | the whole session: tests, calibration, smoke, sweep, plots |
| `scripts/calibrate_hardware.py` | pod | measures this card's real bandwidth and bf16 rate |
| `scripts/calibrate_read_variants.py` | pod | **C4**: naive vs vectorised read, is our ceiling too low |
| `scripts/check_mma_path.sh` | pod | **C3**: dumps PTX, greps wgmma against mma.sync |
| `scripts/kernel_name.py` | pod | **C1**: reads the CUTLASS tile out of the profiler |
| `scripts/compare.py` | anywhere | span-aware comparison, refuses to hide the extent |
| `scripts/publish_results.sh` | pod | commits a result set back to the repo |
| `moe/bench/ridge.py` | anywhere | predicts the crossing per model from a calibration |
| `tests/` | anywhere | 358 tests, all green off-GPU |

## Order of work

1. ~~`check_mma_path.sh`~~ DONE 2026-08-27. C3 established.
2. ~~`calibrate_read_variants.py`~~ DONE 2026-08-27. C4 resolved: the anomaly was
   the ruler. Follow-up, not yet done: settle clocks before the bandwidth
   patterns in `calibrate.py`, and stop naming a tree reduction as the read
   ceiling. Both change the ruler every published number used, so they are a
   deliberate re-baseline rather than a patch.
3. ~~Tile sweep~~ DONE 2026-08-27. Forcing BLOCK_SIZE_M never helps: 16->32 is
   flat (padded MACs are free, MEASURED not modelled), 64 is 1.7-9% slower even
   though it reaches WGMMA, 128 is 27-30% slower. Padded arithmetic hides while
   it is ~20% of the memory time and costs above 40%. Occupancy confound not
   separated, and it does not need to be: the hypothesis was that bigger would
   help. Loose end: confirm the instruction actually switched by re-running
   check_mma_path.sh under the override.
4. The ablation, using the GPU MODE method: alias B by taking the tile offset
   modulo so every iteration reloads the same tile (loads execute, L2 hits, no
   HBM traffic, nothing folds since values are runtime); `acc += tl.sum(b) +
   tl.sum(a)` to keep loads live on the compute side. Settles critical path
   without relying on the byte model, which C4 puts under suspicion.
4. fp8 baselines, then re-sweep. Tests C2's prediction.
5. A100 hour, same sweep. Tests C2's generality.
6. Rewrite FINDINGS around C1-C4, scoped to "single GPU, all experts resident".

## Standing scope limit

Every claim is about **one GPU holding every expert**. It is not a claim about
how frontier MoE is served: DeepSeek runs decode on DP144+EP144 precisely to
scale the aggregate batch past this ridge, and at that scale all-to-all
communication dominates, not the GEMM. Half the corrections in this project came
from stating single-node results as universal. Do not do it again.
