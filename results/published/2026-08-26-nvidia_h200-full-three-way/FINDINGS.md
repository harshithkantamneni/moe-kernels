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
with 1.2..2.3 rows each. The compulsory model charges a full weight matrix per active
expert. If vLLM's kernel skips or truncates weight reads for experts holding almost no
rows, the model over-charges exactly there and nowhere else. That is a testable claim
and this sweep does not test it: it needs a counter, and see section 6.

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

Every serving batch size anyone actually runs sits on the memory side of all three. The
dilution is `E/k`: the more experts a model has per token routed, the further right its
crossing moves, so the models with the most experts are the ones least able to reach
their own compute ceiling.

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
two rows. That is a routing-shaped and scheduling-shaped problem, and it is the one
experiment the incumbent cannot run, because its `BLOCK_M` is not a knob.

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
