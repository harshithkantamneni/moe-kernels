# What the three-way sweep says

17,640 rows, one H200 SXM (700 W, 132 SMs, 60 MiB L2, driver 580.159.04, CUDA 13.0).
All 17,640 passed correctness against an fp32 oracle, bf16 throughout. Three arms,
one per venv, all from commit `11555d9`:

| run id | env | version | rows |
|---|---|---|---:|
| `915cc2fe28eb` | base | torch 2.13.0+cu130 / triton 3.7.1 | 10,584 |
| `c8728dd73700` | vllm | vLLM 0.27.1 | 3,528 |
| `1300be94c335` | sglang | SGLang 0.5.18 | 3,528 |

882 cells = 3 model geometries x 14 token counts (1..8192, powers of two) x 7 routings
(uniform, zipf:0.6/1.2/2.0, dirichlet:0.3, hot:0.5/0.8) x 3 seeds. Each cell is run by
5 implementations in 4 timing modes ({L2-cold, L2-warm} x {eager, graph replay}):
882 x 5 x 4 = 17,640. Ceilings are measured on this device (`measured.yaml`), never
datasheet numbers: 4374.69 GB/s triad, 4389.39 GB/s read, 701.65 TFLOP/s bf16, ridge
160.4 FLOP/byte. The pin rate implied by the 6144-bit bus at 3201 MHz is 4916.7 GB/s.

**Column trap, stated up front.** `achieved_bw_gbps` is NOT this row's achieved
bandwidth. It is the calibration ceiling, identical on all 12,034 rows that carry it.
The row's own apparent bandwidth is `compulsory_gbps`: compulsory bytes divided by
measured time. Anyone recomputing from the shipped CSV will get a flat line from the
first column and the real distribution from the second.

**Every row is `git_dirty`.** The tree carried uncommitted harness changes for the whole
session. The commit is recorded and identical across all three arms, so the arms are
comparable to each other; they are not exactly reproducible from `11555d9` alone.

**DeepSeek-V3 caveat, carried forward:** its 1369 GB of bf16 weights do not fit on one
H200, so its routing is parametric rather than replayed. Its geometry is real; its token
distribution is synthetic. Mixtral and Qwen2 are the same way.

---

## 1. The comparison that looks obvious and is wrong

The five implementations do not cover the same work. `covers` is a column for exactly
this reason, and putting their millisecond figures side by side compares a single GEMM
against a five-stage fused block:

| implementation | covers | stages |
|---|---|---:|
| `torch_grouped_mm_up` / `_down` | one of `up_gemm`, `down_gemm` | 1 of 6 |
| `vllm_fused_experts` | `permute+up_gemm+act+down_gemm+unpermute` | 5 of 6 |
| `sglang_fused_experts` | same five | 5 of 6 |
| `__pipeline__` | the whole reference layer | 6 of 6 |

Nothing in the CSV stops a chart from ignoring this. `scripts/compare.py` prints the
extent under every table and refuses to stay quiet when the columns disagree, which is
the only reason the tables below are on a normalised metric rather than on time.

## 2. Where each implementation sits against the compulsory floor

`implied_traffic_ratio` is bytes the timing implies were moved, divided by bytes the
arithmetic actually requires. 1.00 means the implementation moved exactly the weights
and activations the math demands and nothing else. Basis is L2-cold, eager, unthrottled:
3,225 of 17,640 rows, the cleanest measurement condition in the sweep.

| implementation | span | n | min | median | max | compulsory share at the median |
|---|---|---:|---:|---:|---:|---:|
| PyTorch `grouped_mm` | 1 of 6 | 1053 | 1.35 | **1.62** | 2.31 | 62% |
| vLLM `fused_experts` | 5 of 6 | 546 | 0.98 | **1.16** | 3.12 | 86% |
| SGLang `fused_experts` | 5 of 6 | 548 | 1.02 | **1.17** | 3.19 | 85% |
| reference pipeline | 6 of 6 | 714 | 9.50 | **12.43** | 24.64 | 8% |

The two production kernels move about 1.16x the bytes their arithmetic requires while
covering five stages. `grouped_mm` moves 1.62x while covering one. That gap is not
straightforwardly a kernel-quality gap: a fused span amortises permute and combine
traffic that the single-stage span pays separately and that the compulsory model counts
separately too. What the row does support is narrower and still useful — the fused
implementations are close enough to the floor that the remaining headroom on this axis
is roughly 15%, and the reference pipeline, at 12x, is a correctness oracle and not a
performance baseline in any sense.

## 3. Eighty-three rows report the impossible, and they are not scattered

83 rows have `implied_traffic_ratio` below 1.00, meaning the timing implies fewer bytes
moved than the arithmetic compels. The same 83 rows are exactly the rows whose apparent
bandwidth exceeds the STREAM triad ceiling. They are not spread across the sweep:

- all 83 are `vllm_fused_experts`
- all 83 are deepseek-v3
- all 83 are at T of 16, 32 or 64
- 27 of the 83 are `throttled`; 56 are not, so throttling does not explain them
- peak is 4483.4 GB/s: 102.5% of triad, 91.2% of the pin rate
- **zero rows anywhere in the sweep exceed the pin rate**

That last line is the check that matters. A measurement artifact large enough to break
the model would not politely stop at the bus limit. Exceeding an achievable STREAM
figure by 2.5% while staying 9% under the physical pin rate is what a slightly wrong
byte model looks like, not what a broken timer looks like.

The regime narrows it further. deepseek-v3 at T=16..64 activates 54..225 of 256 experts
with 1.2..2.3 rows each.

**The obvious explanation has been ruled out.** This section previously argued that
vLLM's kernel might skip or truncate weight reads for experts holding almost no rows,
which would make the compulsory model over-charge exactly here and nowhere else. Asked
directly in GPU MODE, the answer was that `fused_experts` reads the entire w1 and w2 for
an expert holding two rows. The byte model is therefore correct and the anomaly is not
the kernel doing less work than charged.

**The surviving explanation is that the ceiling is mis-set, not the numerator.** The same
reply pointed out that decode is effectively pure weight streaming, so the `read` pattern
is the right denominator rather than `triad`. Agreed, and the calibration already
annotates `read` that way, but it only moves the figure from 102.5% to 102.1%: this
card's MEASURED read is 4389.39 GB/s. The 4.8 TB/s that makes the anomaly vanish (93.4%)
is the datasheet number, not a measurement of this device.

That points somewhere more interesting than a kernel quirk. If a production kernel
sustains 4483.4 GB/s on pure weight streaming while `scripts/calibrate_hardware.py`
reaches only 4389.4 GB/s on a STREAM read, then the calibration is not measuring a
ceiling, it is measuring its own achieved rate, and every "percent of ceiling" figure in
this file is pessimistic by that margin. A TMA-driven kernel plausibly beats a naive
read loop. **Open action: harden the calibration read kernel and see whether it climbs
past 4483.4.** That needs no counters and is the cheapest test available.

## 4. Arithmetic intensity is rows per active expert, and it is measured

Not a derivation carried in from the model — the sweep's own columns. Uniform routing,
seed 0, up stage, `AI` is `arith_intensity_compulsory`:

| model | T | active experts | rows/expert | AI | regime |
|---|---:|---:|---:|---:|---|
| mixtral-8x7b | 64 | 8 | 16.0 | 15.9 | memory |
| mixtral-8x7b | 256 | 8 | 64.0 | 62.9 | memory |
| mixtral-8x7b | 1024 | 8 | 256.0 | 238.9 | **compute** |
| qwen2-57b-a14b | 1024 | 64 | 128.0 | 120.7 | memory |
| qwen2-57b-a14b | 4096 | 64 | 512.0 | 412.0 | **compute** |
| deepseek-v3 | 4096 | 256 | 128.0 | 122.0 | memory |
| deepseek-v3 | 8192 | 256 | 256.0 | 233.1 | **compute** |

AI tracks rows-per-expert to within a few percent at every point, and falls away from it
only as the ratio saturates. The consequence is the whole shape of the problem: you
cannot raise arithmetic intensity in an MoE layer without putting more rows on each
expert, and the router decides that, not the kernel.

Where each geometry crosses the 160.4 FLOP/byte ridge, from the measured points:

- **mixtral** (E/k = 4) between T=256 and T=1024
- **qwen2** (8) between T=1024 and T=4096
- **deepseek** (32) between T=4096 and T=8192, interpolating to roughly T = 5500

The dilution is `E/k`: the more experts a model has per token routed, the further right
its crossing moves, so the models with the most experts are the ones least able to reach
their own compute ceiling.

**Why the identity holds, and why no expert architecture escapes it.** Every weight
element is used exactly once per row, contributing 2 FLOPs, and costs `b` bytes read
once. So for an expert holding any weight tensors at all, with `N` elements in total
across however many layers:

```
    FLOPs (R rows) = 2 N R          bytes (weights) = N b
    AI = 2NR / Nb  = 2R / b
```

`N` is a SUM over layers and appears identically top and bottom, so it cancels for any
layer count and any shapes: square, rectangular, mismatched, five layers or one. For bf16
that is `AI = R` exactly. Verified numerically against deliberately lopsided synthetic
architectures (layers of 7168x2048, 2048x999, 999x31, 31x4096, 4096x123 gives the same AI
as two equal 4096x4096 layers). Shapes enter only the second-order activation term, which
is why mixtral at `F/H = 3.50` deviates most (ratio 0.64 at R=2048) and deepseek-v3 at
`F/H = 0.29` least (0.91 at R=256).

The consequence is worth stating plainly: **arithmetic intensity is weight reuse times
`2/b`, and nothing else.** Restructuring the expert cannot move it. The only levers are
`R` (batch and routing) and `b` (fp8 doubles AI and halves the crossing).

**This agrees with published analysis, and the sweep is an independent check on it.**
Yun et al. derive expert-layer intensity as `Γ_imb · B · n_k/n_e` and the compute-bound
threshold as `B_MoE = RP_acc · (n_e/n_k)`, which is the same relation reached here from
measurement rather than from a model. Their Table I gives H200 SXM5 a ridge of 206.15
Op/B in BF16. Their thresholds land inside every bracket measured above:

| model | E/k | `B_MoE` at 206.15 | `B_MoE` at 160.4 | measured bracket |
|---|---:|---:|---:|---|
| mixtral-8x7b | 4 | 825 | 642 | 256 - 1024 |
| qwen2-57b-a14b | 8 | 1649 | 1283 | 1024 - 4096 |
| deepseek-v3 | 32 | 6597 | 5133 | 4096 - 8192 |

The 206.15 is datasheet-derived (989.4 TFLOP/s over 4.8 TB/s). This card measures 160.4,
because achieved bandwidth reaches 91.1% of spec while achieved bf16 reaches only 70.9%.
Compute falls further short than bandwidth does, so the real ridge is LOWER and the
crossing arrives EARLIER than the published figure implies.

**Scope, stated because an earlier draft of this section overreached.** The claim is
about a single GPU holding every expert. It is not a claim about how frontier MoE is
served. DeepSeek's own decode deployment is DP144+EP144 across 18 nodes, and expert
parallelism exists precisely to scale the aggregate batch: rows-per-expert is
`T_aggregate * k / E` regardless of sharding, so 144 GPUs' worth of KV cache can push
`T_aggregate` past 5,133 and over this ridge. Yun et al. describe what replaces the
bottleneck there: "as the batch size increases, the interconnect bandwidth and `Γ_imb`
become the most critical factors." The defensible statement is therefore narrow:
**single-GPU MoE decode with all experts resident is memory-bound at any batch that
fits**, and escaping it costs a network.

Sources: Yun et al., *Rethinking LLM Inference Bottlenecks: Insights from Latent
Attention and Mixture-of-Experts*, arXiv:2507.15465v3 [cs.AR], 2026-01-29.
DeepSeek-V3/R1 Inference System Overview, deepseek-ai/open-infra-index, 2025-02.

## 5. Throttling

1,663 of 17,640 rows carry `throttled` (9.4%). The detector is unchanged and still two
point samples, not a time series: SM clock read once before a cell's timing call and
once after, flag set when the second is more than 5% below the first. It is directional,
so a ramping clock can never trip it, and blind to a dip that recovers in between. A
throttled row is not a failed row; all 1,663 passed the fp32 oracle.

Section 2 is computed with them excluded, which is why its n is 3,225 rather than 4,410.
Excluding them moves the PyTorch median from 1.59 to 1.62 and leaves vLLM and SGLang
within 0.01. Section 3's finding survives the exclusion outright: 56 of the 83 impossible
rows are unthrottled.

## 6. What this sweep does not establish

- **DRAM traffic is still modelled, not counted.** `ncu` needs a host module flag a
  container tenant cannot set (`ERR_NVGPUCTRPERM`), so every byte figure here is
  compulsory-traffic arithmetic. `nsys` does run, and its `--gpu-metrics-device` route is
  untested; that is the open path, not a closed door.
- **SGLang was configured by a default publish, not by a server.** `fused_experts`
  reaches process-wide config that only a running server normally publishes, so the
  harness publishes default `ServerArgs(model_path="dummy")`. The MoE path reads four
  leaves and all four are correct at single-GPU defaults, which makes the risk narrow.
  It does not make it zero: if some other default steers kernel selection, these rows
  measure a path production would not take.
- **Nothing here separates a kernel-quality gap from a span-extent gap.** Section 2 is
  reported per span for that reason. Settling it needs the fused implementations run at
  a single-stage extent, or the harness's own spans fused, and neither exists yet.
- **The MACs-vs-weight-reads separation has a cheaper route than a tile sweep.** Because
  the Triton `fused_moe` kernel is editable Python, replacing the weight `tl.load` with
  `tl.zeros` removes weight traffic while keeping the MACs, and no-oping the dot removes
  the MACs while keeping the traffic. Two ablations isolate the terms directly, with no
  occupancy confound. The hazard is dead-code elimination: zeroing the weights may let
  the compiler fold the dot away, and no-oping the dot may let it delete the loads, so
  the data dependency to the store has to be kept alive and the result checked rather
  than assumed. Suggested in GPU MODE and not yet run.
- **The tile sweep described below needs a precondition this file did not state.** It
  assumes `M_tiles = active_experts` independent of `BLOCK_M`, which holds only while
  EVERY expert fits in one tile. `load_max_rows` says that fails under skew: deepseek-v3
  at T=64 `hot:0.5` puts 64 rows on the top expert, exactly one tile, so any smaller
  `BLOCK_M` spills it and traffic stops being flat. Uniform routing at T <= 64 keeps the
  maximum at 3 to 7 rows, and is where the sweep is clean. Separately, `BLOCK_M` sizes
  the register accumulator and therefore occupancy, so a change in time is ambiguous
  between padded MACs and latency hiding: only a FLAT result is clean evidence.
- **The 1.16x floor is a ratio against a model, not against a counter.** If the
  compulsory model is wrong in the direction section 3 suggests, it is wrong for these
  numbers too, and the true figure is closer to 1.00 than reported.

---

## What this means for the kernel

Three things are now measured rather than assumed, and they point the same way.

`BLOCK_M` is **64**, read from the kernel name rather than inferred from timing.
`scripts/kernel_name.py` was run at T = 1, 16, 256, 1024 and 4096 and the name is
identical at all five: `TileShape M,N = 64,128`, MMA atom
`MMA_64x128x16_F32BF16BF16_SS`, schedule `Pingpong` and never `Cooperative`, so no two
warpgroups share a tile and the effective M never doubles. That is a 4096x range in
token count with no shape-dependent selection. It was NOT re-run at the 8192 this sweep
added, so the constancy is established over 1..4096 and assumed above it. The published
claim of 128 is refuted. Refitting the re-read cost against the observed tile over the
151 unthrottled memory-bound rows gives alpha = 0.10 (mean ratio 1.65x, CV 12.8%),
against 1.67x / 13.1% at alpha = 0 and 1.60x / 17.5% at alpha = 1: an extra M-tile on
the same expert costs about a tenth of a fresh weight read, not a whole one.

Arithmetic intensity is rows per active expert, section 4 measures it, and every
realistic serving point is on the memory side of the ridge. A kernel that wins here wins
by moving fewer weight bytes, not by doing arithmetic faster.

And the incumbents are already close to that floor while covering five stages. 1.16x is
not a lot of room. The honest read is that the remaining win is not in the grouped GEMM
as a GEMM; it is in not streaming a weight matrix per active expert when the expert holds
two rows. That is a routing-shaped and scheduling-shaped problem.

The tile-height sweep that would separate wasted MACs from wasted weight reads is
**runnable today, and needs no new kernel**: vLLM and SGLang both ship tuned
`BLOCK_SIZE_M` per batch size in their fused_moe config JSONs, so it is a config edit and
a rerun. It has not been run here, and saying otherwise would be wrong. A DRAM counter
would settle the same question directly, which is the other route this pod blocks.

What no shipped implementation offers is a tile that varies **per expert within a
launch**. `moe_align_block_size` emits one padded, expert-sorted token array with a
single block size, and both GEMMs consume it, so every expert in a launch is served by
the same tile height. Under skew that puts experts on both sides of the ridge under one
tile. `torch.nn.functional.grouped_mm` is stricter still: its tile is fixed in the
CUTLASS template and the API exposes no parameter at all.

---

*Corrections and provenance: `merged.csv` in this directory was contaminated when the
arm was first published. `CsvWriter` is append-only by design so a killed pod loses at
most one cell, `merge_csvs` used it, and `run_all.sh` merges into a persistent
`results/merged.csv` once per sweep, so every sweep silently inherited every sweep before
it on that pod. This arm's merge carried 872 foreign rows from three other runs,
including all 840 rows of the 2026-08-22 standard sweep, which were measured against a
different calibration and therefore quote efficiency columns against the wrong ceiling
with nothing in the row to say so. `merge_csvs` now rebuilds its output; all three
published arms have been rebuilt; the run_*.csv arm files were never affected, so the
2026-08-22 FINDINGS, which were computed from the arm and not the merge, stand unchanged.*

*Every number in this file was recomputed from the repaired merge on a stated basis. An
earlier draft of the companion write-up reported the section 2 triplets without naming
the L2-cold/eager/unthrottled basis they came from, gave medians that do not reproduce
under any filter, and transposed the reference pipeline's maximum as 14.64 where the data
says 24.64. The counts of impossible rows and the peak bandwidth in that draft (83 rows,
4483.4 GB/s, 102.5% of triad, 91.2% of pin, none above the pin rate) do reproduce exactly
and are unchanged.*
