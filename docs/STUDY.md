# The study: what this project is now, and what each piece is for

Written 2026-08-27. Read this first if you have been away from the project.

This file is the working state: how each claim got where it is, what was
retracted, what runs where, what to do next. The RESULTS are in
`docs/FINDINGS.md`, organised around C1-C5 and regenerated from the published
rows.

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

## The claims

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

A CONFOUND ON THE 1.15, found 2026-08-28 while rescoping C3. vLLM's tuned
configs pick a DIFFERENT TILE for the two dtypes on the same shape. The mixtral
sweep loaded `E=8,N=14336,dtype=fp8_w8a8`, which sets `BLOCK_SIZE_M = 64` from
M=1, while its bf16 twin sets 16 to 32 at low M. So the fp8 arm ran on taller
tiles than the bf16 arm throughout, and the dtype comparison silently varied the
tile as well.

The direction of the bias is known and matches. A taller tile is 3.1 to 3.6x
faster above the ridge (fewer weight re-reads, `alpha ~ 0.21` per extra tile),
which speeds the compute-bound side and pushes the crossing LATER. Measured
fp8/bf16 is 1.15 against a predicted 1.00, later, so some unknown part of that
0.15 is tile rather than dtype.

Dtype-invariance survives it: the retracted alternative needs 0.50, and no tile
effect of this size closes a gap that wide. But 1.15 is not a pure dtype
measurement, and separating them needs a run with `BLOCK_SIZE_M` pinned equal
across both dtypes, which `override_config` can do and this study has not done.

THE THIRD KERNEL: ITS TIMINGS ARE FIXED, ITS CROSSINGS ARE UNUSABLE.
`torch_scaled_grouped_mm_*` first reported 0.44 +/- 0.13, which looks like
support for the retracted 2x prediction and was an artefact: the span quantised
activations INSIDE the timed region, because `_scaled_grouped_mm` needs both
operands in fp8 while the harness hands out bf16 activations for vLLM's sake.
The giveaway was direct -- deepseek-v2-lite at T=8192 measured 1.9855 ms in fp8
against bf16's 1.0503, and fp8 moves half the weight bytes so it cannot be
slower.

FIXED AND RE-MEASURED (`6652c66`, arm `-fp8-refixed`, 9,408 rows). The same cell
is now 0.7659 ms, 0.73x of bf16, reproducing the smoke's 0.7666 to 0.3%. A 2.59x
change, and it scaled with tokens. The timings from this span are now sound.

BUT THE CROSSINGS FROM IT STILL ARE NOT:

| model            |   up | down |
|------------------|-----:|-----:|
| mixtral-8x7b     | 0.87 | 1.99 |
| qwen2-57b-a14b   | 0.88 | 0.61 |
| deepseek-v2-lite | 0.63 | 0.41 |
| deepseek-v3      |   -- | 0.52 |
| **mean of 7**    | **0.84 +/- 0.53** ||

CORRECTED 2026-08-31. This table previously read `0.52 | 1.23` on the last row
and `0.89 +/- 0.51` for the mean. deepseek-v3's fp8 `up` span has NO crossing:
its slope peaks at 0.497 at T=8192 and never reaches the threshold, under every
filter including `--include-throttled`. The 0.52 is the `down` value, and the
1.23 does not reproduce from either the refixed arm or the superseded one.

The mean moved from 0.44 to 0.84, toward the corrected theory's 1.00, which is
what removing a token-scaling bias should do. But 0.41 to 1.99 is a five-fold
range, against 1.15 +/- 0.07 from the two production kernels. Nothing can be
concluded from a spread that wide.

The likely cause is the method, not the span. A single GEMM's time-against-T
curve is flatter and less structured than a fused layer's, so the slope has less
to cross and the 0.5 threshold lands wherever local noise puts it. `up` and
`down` are the same arithmetic on the same cells and disagree by 2.3x on mixtral,
which is not a property either dtype has.

SO: dtype-invariance rests on vLLM and SGLang. The one-stage span contributes its
TIMES, which is what the five-stage/one-stage separation below uses, and does not
contribute a third crossing measurement. Getting one needs a token grid dense
enough to locate a shallow slope change, which this profile's powers of two are
not.

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

The bandwidth reproduces; the compute term does not. THE CLOCK IS NOT THE
EXPLANATION, corrected 2026-08-31: across the three H200 calibrations the GEMM
clock moves 20.6% and the achieved rate moves 9.9% the OTHER WAY. 1845 MHz
reached 71.4% of its own clock's peak, 1560 MHz reached 83.2%, 1530 MHz reached
93.2%. Clock normalisation does not collapse the band; the spread lives in
achieved efficiency, and what drives that is not established. So the ridge is a
RANGE, wider than the +/-1.5% this document previously quoted, and the whole
table moves with it: at ridge 176.2 the means are 0.58 and 1.03 rather than 0.63
and 1.13.

WHAT SURVIVES THE BAND is the comparison between span extents. Both sides divide
by the same predicted crossing, so the ridge cancels ALGEBRAICALLY: five-stage
over one-stage is 0.561 at ridge 160.3, at 176.2, and at any other value
(`tests/test_ridge_band.py`). A five-stage span crosses at 56% of the batch a
one-stage span does, and no calibration uncertainty touches that.

AND IT SURVIVES A BETTER BYTE MODEL TOO. The predictions above solve `2R/b`,
which is the weight-dominated LIMIT of the general GEMM intensity

    AI = 2MNK / ((MK + KN + MN) b)   ->   2M/b   when KN dominates

while every measured row is scored with the full model, activations included.
Two models on the two sides of one comparison. `2R/b` overstates AI by ~4% for
mixtral at its crossing and ~7% for deepseek-v3, and overstating AI understates
the batch needed, so every prediction here is systematically low by 5 to 18%.
`crossing_batch_full` solves the same byte model the rows use:

    |                     | five-stage | one-stage | separation |
    |---------------------|-----------:|----------:|-----------:|
    | 2R/b,  ridge 160.3  |       0.63 |      1.13 |      0.561 |
    | full,  ridge 160.3  |       0.58 |      1.03 |      0.563 |
    | full,  ridge 176.2  |       0.52 |      0.92 |      0.563 |

The one-stage span lands within about 10% of prediction under every combination,
0.92 to 1.13, which is the agreement `2R/b` was reaching for. The five-stage span
sits at 0.52 to 0.63 whatever is done to the model. And the separation is 0.563
throughout.

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

**C3. Below roughly 100 rows per expert, and in bf16, vLLM emits no warpgroup MMA.**
Hopper's `wgmma.mma_async.m64nNk16` has M fixed at 64, so any tile shorter than
that runs on `mma.sync`, the Ampere-era instruction. The question is when vLLM's
chosen tile is shorter than 64, and what it costs when it is not.

STATUS: **ESTABLISHED, measured 2026-08-27/28, and RESCOPED twice since.**

THE MEASUREMENT. `scripts/check_mma_path.sh` on a real deepseek-v3 T=16 cell:
both compiled `fused_moe_kernel` variants show `wgmma=0`, `mma.sync=16`, and
every one of the 32 tensor-core instructions is
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. Forcing `BLOCK_M >= 64`
does reach `wgmma.mma_async.sync.aligned.m64n32k16` and `m64n64k16`, and is
1.7 to 9% SLOWER: the capability is there and declining it is correct, because
the tensor core idles waiting on weights either way while a short tile buys
occupancy.

FIRST RESCOPE: IT IS NOT ABOUT "DECODE". The kernel sees M, the rows entering
the layer, and cannot tell whether they came from one prefill or from a thousand
concurrent decodes. A serving system with enough concurrency is in decode AND at
large M simultaneously, and there the tuned config picks a taller tile and does
emit wgmma. The claim is about a REGIME, in rows per expert, which batching can
leave. Reading the tuned H200 configs: `E=1,N=3072` steps 16 -> 32 -> 64 -> 128
by M=128, while `E=128,N=1024` stays on 16 until M=1536. That spread is the
`E/k` dilution appearing in someone else's grid search, since a constant
rows-per-expert threshold means `M_switch` scales with `E/k`.

SECOND RESCOPE: IT IS bf16-SPECIFIC. The same shapes tuned for fp8 pick a
warpgroup tile at M=1:

    E=8,N=14336   fp8   1:64, 2:64, 4:64, 8:64, 16:64 ...
    E=8,N=14336   bf16  1:16, 2:32, 4:16, 8:16, 16:16 ...
    E=64,N=2560   fp8   1:64, 2:64, 4:64 ...
    E=64,N=2560   bf16  1:16, 2:16, 4:16 ...

So at fp8 decode vLLM reaches wgmma immediately, and C3 describes the bf16 path.

AND THE MEASURED CELL WAS RUNNING A FALLBACK, NOT A TUNED CONFIG. deepseek-v3 is
`E=256,N=2048`, and no tuned H200 config exists for it: the run log prints
`Using default MoE config. Performance might be sub-optimal!`. So the 16 in that
PTX comes from `get_default_config`'s hardcoded small-M branch rather than from
a grid search. A GPU MODE reader raised exactly this, and was right. It does not
weaken the measurement, since a fallback is what deepseek-v3 actually runs, but
it changes what the 16 is EVIDENCE of: a default, not a tuned optimum.

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

**C5. Does the crossing scale with the ridge across architectures?**
`AI = 2R/b` says the crossing is at a fixed rows-per-expert set by the ridge, so
two cards with different ridges should cross at rows-per-expert in the same
proportion. H200 ridge 176.2, A100 ridge 145.7.

STATUS: **PARTIAL, measured 2026-08-28, RESCORED 2026-08-31.** Same profile, same
kernel, one run per card, `vllm_fused_experts` bf16.

THE TARGET IS 0.83, NOT 1.00, AND THIS SECTION PREVIOUSLY USED 1.00. For bf16
`b = 2`, so `2R/b = ridge` puts the crossing at `R = ridge` rows per expert, a
DIFFERENT R on each card. The ratio the two cards should show is
145.7 / 176.2 = 0.83, or 0.91 if the H200 ridge is taken at the low end of its
band. A measured ratio of 1.00 means the two cards crossed at the same rows per
expert, which is what NO ridge scaling looks like.

| model            |   E | A100 | H200 | ratio | vs target 0.83 |
|------------------|----:|-----:|-----:|------:|---------------:|
| mixtral-8x7b     |   8 |   58 |  136 |  0.43 |           0.52 |
| qwen2-57b-a14b   |  64 |   81 |  114 |  0.71 |           0.86 |
| deepseek-v2-lite |  64 |   74 |   84 |  0.88 |       **1.06** |
| deepseek-v3      | 256 |  110 |  109 |  1.01 |           1.22 |

So deepseek-v2-lite is the model that scales with the ridge, within 6% and inside
the band. deepseek-v3 overshoots by 22%. This section previously reported
deepseek-v3 as agreeing "to 1% across two architectures, three years apart",
which was agreement with the null. No model confirms cleanly and none refutes by
an order of magnitude.

WHAT SURVIVES: the deviation is monotonic in EXPERT COUNT either way, 0.52, 0.86,
1.06, 1.22 against E of 8, 64, 64, 256, and expert count is not a term in the
model. Correcting the target moves the deviation from "approaches agreement from
below" to "crosses agreement between 64 and 256 experts" without changing its
ordering, and the ordering is what needs explaining.

THE HYPOTHESIS, STATED AS UNTESTED. Expert count sets how many thread blocks a
launch has. With 8 experts there may not be enough to fill 108 SMs (A100) or 132
(H200) until T is large, so the time-against-T curve is shaped by an occupancy
ramp rather than by the roofline, and what the detector finds is not the ridge.
With 256 experts there are always enough blocks. The A100 has fewer SMs, so any
block-count effect should hit it harder, which is the direction observed. Four
points and a plausible story are not evidence, and this is recorded as a
hypothesis with an experiment attached: sweep expert count at FIXED
rows-per-expert and see whether the deviation follows blocks or follows E.

WHAT DOES TRANSFER. The five-stage offset is a property of the fused span on
both cards: measured/predicted is 0.55 +/- 0.15 on the A100 against 0.63 +/- 0.12
on the H200. The means differ by 13% and the spreads overlap heavily, so read it
as consistent across two machines rather than as the same number. The 13% is the
C5 discrepancy appearing in another coordinate.

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

**A. fp8 -- a prediction test.** C2 says `AI = 2R/b`. Halving `b` must double
arithmetic intensity and halve the ridge crossing: deepseek-v3 from ~5,500 tokens
to ~2,750. That is falsifiable, quantitative, and derived before being measured.
Needs the baselines to declare and support fp8; `spec.py` already knows
`fp8_e4m3` and `fp8_e5m2` at 1 byte and the byte model is already
dtype-parametric.

**B. A second device -- a generalisation test.** The crossing scales with the
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
| `scripts/tile_sweep.py` | pod | **C3**: forces BLOCK_M past 64 and dumps the PTX it emits |
| `scripts/compare.py` | anywhere | span-aware comparison, refuses to hide the extent |
| `scripts/crossing_report.py` | anywhere | **C2/C5**: reads the crossing off measured TIME |
| `scripts/efficiency_report.py` | anywhere | is the crossing offset achieved-versus-peak? |
| `scripts/recompute_ceilings.py` | anywhere | re-derives a published arm's ceiling columns |
| `scripts/plot.py` | anywhere | figures, one set per dtype present in the rows |
| `scripts/publish_results.sh` | pod | commits a result set back to the repo |
| `scripts/setup_runpod.sh` | pod | builds the venvs and reports what the card is |
| `scripts/sweep_progress.py` | pod | how far a running sweep has got, and its real rate |
| `scripts/capture_traces.py` | pod | real routing distributions from a real MoE model |
| `scripts/probe_baseline_api.py` | pod | what MoE entry points this venv's framework exposes |
| `scripts/probe_baseline_types.py` | pod | and the argument types of the one it exposes |
| `scripts/profile_open_questions.sh` | pod | the two questions the first sweep could not answer |
| `scripts/preflight_cutile.py` | pod | whether cuTile is worth more of this pod's time |
| `moe/bench/ridge.py` | anywhere | predicts the crossing per model from a calibration |
| `moe/bench/published.py` | anywhere | which published arms an analysis should read |
| `tests/` | anywhere | 549 tests, all green off-GPU |

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
6. ~~Rewrite FINDINGS around C1-C4~~ DONE 2026-08-31, around C1-C5, at
   `docs/FINDINGS.md`. Study-level rather than arm-level: the arm-local
   FINDINGS in the superseded three-way directory is now banner-marked as
   historical. Every table in the new file names the command that regenerates
   it, and every number in it was recomputed from the rows rather than carried
   forward, which is how the two corrections above were found.

## Standing scope limit

Every claim is about **one GPU holding every expert**. It is not a claim about
how frontier MoE is served: DeepSeek runs decode on DP144+EP144 precisely to
scale the aggregate batch past this ridge, and at that scale all-to-all
communication dominates, not the GEMM. Half the corrections in this project came
from stating single-node results as universal. Do not do it again.
