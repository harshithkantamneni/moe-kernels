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
STATUS: **established.** Verified numerically against deliberately lopsided
synthetic architectures, and matches Yun et al. arXiv:2507.15465, whose
`B_MoE = RP_acc * (n_e/n_k)` lands inside all three of our measured brackets.

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
