# Findings

What this project has measured, claim by claim. Written 2026-08-31, against the
ten published arms in `results/published/`.

`docs/STUDY.md` is the working state: how each claim got where it is, what was
retracted and why, what runs where, what to do next. This file is the result.
Where the two disagree, this one was regenerated from the rows and STUDY was not.

Every number that can be recomputed from `results/published/` was recomputed for
this file rather than carried forward, and every table names the command that
produces it. Observations that need a GPU (PTX dumps, CUTLASS kernel names, the
tile sweep, the vLLM config ladders) are quoted from run logs and named as such.

---

## The evidence base

96,448 measured rows on two cards. 69,064 of them are current; the rest are
superseded and are kept for provenance, not for analysis.

| arm | rows | current | what it is for |
|---|---:|---:|---|
| `2026-08-22-first-smoke` | 16 | 16 | first working cell |
| `2026-08-22-standard-sweep` | 840 | 840 | torch spans, coarse grid, adds points to the bf16 pool |
| `2026-08-26-...-full-three-way` | 17,640 | 0 | **superseded whole** by its recalibrated twin |
| `2026-08-26-...-full-three-way-recalibrated` | 17,640 | 17,640 | the main bf16 sweep, three kernels |
| `2026-08-28-...-a100-cross-card` | 9,408 | 9,408 | **C5**: the second device |
| `2026-08-28-...-h200-fp8-three-kernel` | 19,908 | 10,164 | **C2**: fp8, vLLM and SGLang rows only |
| `2026-08-28-...-h200-fp8-refixed` | 9,408 | 9,408 | **C2**: the re-measured torch fp8 spans |
| `2026-08-28-...-h200-v2lite` | 5,880 | 5,880 | adds deepseek-v2-lite to the bf16 pool |
| `2026-08-28-...-h200-whole-layer` | 9,408 | 9,408 | router included, six stages of six |
| `2026-08-28-...-ridge-resolution` | 6,300 | 6,300 | bf16 re-run at a second calibration |

Two arms are partially or wholly retired and both say so in a `SUPERSEDED` file
that `moe/bench/published.py` reads. The fp8 arm's retirement is per
implementation: its vLLM and SGLang rows are current and carry C2, while its two
`torch_scaled_grouped_mm_*` spans quantised activations inside the timed region
and are replaced by `-fp8-refixed`.

**The canonical bf16 H200 pool**, which every bf16 crossing in this file comes
from, is four arms: `standard-sweep`, `full-three-way-recalibrated`,
`ridge-resolution`, `h200-v2lite`. It is a pool rather than a single run because
no single arm carries all four model geometries. There is no machine-readable
definition of it anywhere in the repo, which is a gap: `published.py` can say
which arms are retired but not which are comparable.

```
python scripts/crossing_report.py \
  results/published/2026-08-22-standard-sweep/run_*.csv \
  results/published/2026-08-26-nvidia_h200-full-three-way-recalibrated/run_*.csv \
  results/published/2026-08-28-nvidia_h200-ridge-resolution/run_*.csv \
  results/published/2026-08-28-nvidia_h200-h200-v2lite/run_*.csv \
  --ridge 160.3
```

### The ruler is a band, and it moves on one side only

Three calibrations of the same H200:

| | bandwidth (triad) | dense bf16 | GEMM clock | % of that clock's peak | ridge |
|---|---:|---:|---:|---:|---:|
| `full-three-way-recalibrated`, `ridge-resolution` | 4377.0 GB/s | 712.4 TFLOP/s | 1845 MHz | 71.4% | 162.8 |
| `fp8-three-kernel`, `v2lite` | 4377.2 GB/s | 701.6 TFLOP/s | 1560 MHz | 83.2% | 160.3 |
| `fp8-refixed`, `whole-layer` | 4374.5 GB/s | 770.9 TFLOP/s | 1530 MHz | 93.2% | 176.2 |

Bandwidth reproduces to 0.06%. The compute term does not: 9.9% between the
extremes. So the ridge is **160.3 to 176.2 FLOP/byte**, and every absolute
measured-over-predicted figure in this file carries that band. The A100 measures
145.7 (262.0 TFLOP/s over 1798.5 GB/s).

**The clock is not the explanation, which is worth stating because STUDY.md says
it is.** Across the three calibrations the clock moves 20.6% and the achieved
rate moves 9.9% in the OPPOSITE direction: the run at 1845 MHz reached 71.4% of
its own clock's peak, and the run at 1530 MHz reached 93.2% of its. Clock
normalisation therefore does not collapse the band, and the spread lives in
achieved efficiency. A one-line "the GEMM runs at whatever clock the thermal
state allows" covers the direction of the clock but not the sign of the result.
What causes the efficiency spread is not established here; an 8192-cubed cuBLAS
GEMM measured for a few seconds on a rented pod has at least thermal state,
neighbour load and measurement duration confounded. Treat the band as an
empirical range, not as a clock artefact with a known correction.

### Three defaults that are not measurements

Each one produced a confident wrong number before it was caught. All three are
still live properties of the CSV and any new analysis has to filter them.

1. **`ms_p50 = 0.0` means the cell never ran.** A skipped or uncapturable graph
   mode still writes a row. 8,848 of the canonical pool's 30,660 rows are like
   this. Feeding them to a median made the first fp8 report conclude deepseek-v3
   crossed at 2 tokens. `crossing.timed_rows` drops them.
2. **`implied_traffic_ratio = 0.0` means the column does not apply.**
   `driver.py` writes it only when the cell is memory bound, so every
   compute-bound row keeps the default. 2,060 timed rows in the recalibrated arm
   carry it, all at T >= 1024. Including them moves vLLM's median from 1.16 to
   1.13 and turns 82 sub-floor rows into 2,142. Nothing in `timed_rows` guards
   this one; the filter has to be written per analysis.
3. **A slope below `E/k` tokens is not the ridge.** Below saturation a batch does
   not touch every expert, so active experts and weight traffic grow with the
   batch and time rises nearly linearly. That slope crosses 0.5 for a reason
   unrelated to the roofline. `crossing_from_points` floors at
   `ridge.saturation_batch`.

### Standing scope limit

Every claim here is about **one GPU holding every expert**. It is not a claim
about how frontier MoE is served: DeepSeek runs decode on DP144+EP144 precisely
to scale the aggregate batch past this ridge, and at that scale all-to-all
communication dominates rather than the GEMM. Half the corrections in this
project came from stating a single-node result as universal.

### And a second scope limit, added 2026-09-01: MoE decode is NOT universally memory-bound

This project has repeatedly said "decode is memory-bound" as though it were a
property of MoE. It is a property of a REGIME, and the study's own measured
crossings say where that regime ends. Tokens entering the layer, H200, uniform
routing, `vllm_fused_experts`:

| model | crossing | | 1 | 256 | 512 | 1024 | 4096 | 8192 |
|---|---:|---|---|---|---|---|---|---|
| mixtral-8x7b | 316 | | . | . | X | X | X | X |
| qwen2-57b-a14b | 787 | | . | . | . | X | X | X |
| deepseek-v2-lite | 931 | | . | . | . | X | X | X |
| deepseek-v3 | 3010 | | . | . | . | . | X | X |

`.` memory-bound, `X` compute-bound. **At a few thousand tokens per forward,
three of four models are compute-bound.** Chunked prefill routinely puts that many
tokens in one pass, prefill is compute-bound on its own, and with expert
parallelism rows-per-expert is `T_aggregate k / E` regardless of sharding, so a
DP144+EP144 decode deployment is deliberately engineered to be on the compute side.

In PURE decode each sequence contributes one token, so crossing over requires 316
concurrent sequences for mixtral and 3,010 for deepseek-v3. Hyperscale serving
reaches that; most deployments do not.

SO THE REGIME THIS STUDY CHARACTERISES IS: pure decode at modest concurrency,
single-user and on-prem deployment, and latency-bound serving where batching is
deliberately capped. It is NOT high-throughput production serving, and every
finding here should be read against that.

Two things this sharpens rather than weakens. The measured crossings arrive at
0.5-0.6x of what `2R/b` predicts, because the weights-only model omits the
permute, activation and unpermute traffic, so real systems cross into
compute-bound EARLIER than the standard analysis says. And the `E/k` dilution law
means the crossing moves right with every generation: mixtral crosses at 316
tokens and deepseek-v3 at 3,010, a 10x spread driven purely by expert count at
fixed `top_k`. Architectures keep adding experts, so each generation stays
memory-bound at batch sizes where the previous one was already compute-bound.

NOT COVERED AT ALL: the offload regime, where experts do not fit in HBM and stream
over PCIe. The roofline extends to it in principle -- PCIe at ~64 GB/s puts the
ridge near 12,000 FLOP/byte, so the crossing would sit around 385,000 tokens for
deepseek-v3 and that regime is never compute-bound -- but this project has never
measured a host-to-device transfer, has no offload path in the byte model, and has
no rows there. Any statement about offload is an extrapolation from a model, not a
measurement.

---

## C1. The CUTLASS tile is 64, and it was never a choice

**ESTABLISHED.**

`torch.nn.functional.grouped_mm` on Hopper reports `TileShape M,N = 64,128`, MMA
atom `MMA_64x128x16_F32BF16BF16_SS`, schedule `Pingpong` and never
`Cooperative`, identical at T = 1, 16, 256, 1024 and 4096. Read out of the
profiler by `scripts/kernel_name.py`, not inferred from timing.

It could not have been otherwise. Hopper's `wgmma.mma_async.m64nNk16` fixes
**M at 64** in the instruction set: N is any multiple of 8 from 8 to 256, K is 16
for 16-bit operands, and the instruction is issued collectively by a warpgroup of
four warps. There is no shape at which CUTLASS could have selected a different M,
which is why the name is constant over a 4096x range in token count.

This refutes a `BLOCK_M = 128` figure in three published write-ups, two of them
ours. `Pingpong` also matters: two warpgroups never share a tile, so the
effective M never doubles to 128 that way either.

Constancy is established over T = 1 to 4096 and assumed above it. The 8192 point
this study later added was never re-run through `kernel_name.py`.

RETRACTED 2026-09-01, alpha is 0.558 not 0.10. See the refit below; the 0.10 is
an artefact of the estimator, not a measurement. The original text follows.

Refitting the re-read cost against the observed tile over the 151 unthrottled
memory-bound rows gives **alpha = 0.10** (mean ratio 1.65x, CV 12.8%), against
1.67x / 13.1% at alpha = 0 and 1.60x / 17.5% at alpha = 1. An extra M-tile on the
same expert costs about a tenth of a fresh weight read, not a whole one. This
figure is carried forward from the 2026-08-26 write-up: the fit has no script and
was not regenerated here.

---

## C2. Arithmetic intensity is `2R/b`, and expert architecture does not enter it

**ESTABLISHED, on a prediction that could have refuted it.**

Every weight element is used once per row (2 FLOPs) and read once (`b` bytes), so
for `N` weight elements across any number of layers holding `R` rows:

```
FLOPs = 2NR     weight bytes = Nb     AI = 2NR / Nb = 2R / b
```

`N` is a sum over layers and cancels. Layer count and matrix shapes are
irrelevant: square, rectangular, mismatched, one layer or five. For bf16,
`AI = rows per expert`. Verified numerically against deliberately lopsided
synthetic architectures (7168x2048, 2048x999, 999x31, 31x4096, 4096x123 gives the
same AI as two equal 4096x4096 layers).

### The prediction that was wrong was ours, not C2's

The fp8 sweep was built to test a 2x crossing shift: halve the weight bytes,
halve the batch at which a model crosses. That is wrong. The ridge is
`peak_FLOPS / bandwidth`, and bf16 to fp8 halves `b` **and** doubles `peak_FLOPS`,
because the same silicon runs fp8 tensor cores at twice the bf16 rate:

```
fp8:   2R/1 = 2 * ridge_bf16   ->   R = ridge_bf16
bf16:  2R/2 =     ridge_bf16   ->   R = ridge_bf16
```

Same rows per expert, so the same crossing. Both sides of the roofline scale
together and their intersection does not move. The 2x figure came from holding
the ridge at its bf16 value while changing the format. `ridge.ridge_for_dtype`
exists to make that mistake impossible to repeat.

### Measured

Crossings are recovered from **time**, never from the byte model: the slope of
`log ms` against `log T` passing 0.5, floored at the saturation batch. The
prediction is therefore refutable.

fp8/bf16 crossing ratio. Corrected theory says 1.00; the retracted 2x says 0.50.

| model | vLLM bf16 | vLLM fp8 | ratio | SGLang bf16 | SGLang fp8 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| mixtral-8x7b | 454 | 568 | 1.25 | 464 | 536 | 1.16 |
| qwen2-57b-a14b | 810 | 900 | 1.11 | 819 | 945 | 1.15 |
| deepseek-v2-lite | 922 | 976 | 1.06 | 1025 | 1193 | 1.16 |
| deepseek-v3 | 3240 | 3459 | 1.07 | 3048 | 3741 | 1.23 |

**1.149 +/- 0.069** over eight measurements from two unrelated kernels, which
also agree with each other on absolute bf16 crossings to within a few percent
(454 vs 464, 810 vs 819, 3240 vs 3048). The two columns come from arms measured
against different calibrations, which does not affect the ratio: both sides are
crossings read off measured time, and no prediction enters it. The traffic reduction is real and shows
up in the time rather than the crossing: mixtral at T=512 goes 1.1568 to 0.6383
ms, 0.55x.

```
python scripts/crossing_report.py \
  results/published/2026-08-28-nvidia_h200-h200-fp8-three-kernel/run_*.csv \
  results/published/2026-08-28-nvidia_h200-h200-fp8-refixed/run_*.csv \
  --ridge 160.3
```

### A confound on the 1.15, and it is not closed

vLLM's tuned configs pick a **different tile for the two dtypes on the same
shape**. The mixtral sweep loaded `E=8,N=14336,dtype=fp8_w8a8`, which sets
`BLOCK_SIZE_M = 64` from M=1, while its bf16 twin sets 16 to 32 at low M. So the
fp8 arm ran on taller tiles than the bf16 arm throughout, and the dtype
comparison silently varied the tile as well.

The direction is known and matches. A taller tile is 3.1 to 3.6x faster above the
ridge, which speeds the compute-bound side and pushes the crossing later.
Measured fp8/bf16 is 1.15 against a predicted 1.00, later, so some unknown part
of that 0.15 is tile rather than dtype.

Dtype-invariance survives it: the retracted alternative needs 0.50, and no tile
effect of this size closes a gap that wide. But 1.15 is not a pure dtype
measurement. Separating them needs a run with `BLOCK_SIZE_M` pinned equal across
both dtypes, which `override_config` can do and this study has not done.

### The arithmetic is not pinned to better than about 10% either

The `ridge_fp8 = 2 x ridge_bf16` above is the datasheet relationship. Measured on
this H200:

```
bf16    770.9 TFLOP/s at 1530 MHz   93.2% of that clock's peak
fp8    1409.2 TFLOP/s at 1740 MHz   74.9% of that clock's peak
```

The two GEMMs ran at different clocks, so the 1.828 the calibration records
conflates format with clock. Per clock it is 1.607: fp8 reaches materially less of
its own peak than bf16 does of its. That makes the prediction worse, not better:

| ridge ratio | predicts fp8/bf16 crossing |
|---|---:|
| 2.000 (datasheet) | 1.000 |
| 1.828 (this calibration) | 0.914 |
| 2.008 (against the older bf16 figure) | 1.004 |
| **measured, two production kernels** | **1.149 +/- 0.069** |

The prediction spans 0.914 to 1.004 across two measurements of one machine,
because the bf16 denominator moves 9.9%. Dtype-invariance holds against the naive
0.50 by a wide margin either way; the arithmetic is not settled to better than
about 10% until the bf16 calibration is pinned down, and the ridge-band section
above says why that is harder than normalising a clock.
(`tests/test_ridge_band.py`)

### The third kernel: timings fixed, crossings unusable

`torch_scaled_grouped_mm_*` first reported 0.44 +/- 0.13, which looks like support
for the retracted 2x prediction and was an artefact: the span quantised
activations inside the timed region, because `_scaled_grouped_mm` needs both
operands in fp8 while the harness hands out bf16 activations for vLLM's sake. The
giveaway was direct. deepseek-v2-lite at T=8192 measured 1.9855 ms in fp8 against
bf16's 1.0503, and fp8 moves half the weight bytes so it cannot be slower.

Fixed in `6652c66` and re-measured as `-fp8-refixed`, 9,408 rows. The same cell is
now 0.7659 ms, 0.73x of bf16, reproducing the smoke's 0.7666 to 0.3%. A 2.59x
change, and it scaled with tokens, so it biased every crossing the old arm
reported. **The timings from this span are now sound.**

The crossings from it still are not:

| model | up bf16 | up fp8 | ratio | down bf16 | down fp8 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| mixtral-8x7b | 938 | 819 | 0.87 | 409 | 814 | 1.99 |
| qwen2-57b-a14b | 1277 | 1119 | 0.88 | 1508 | 920 | 0.61 |
| deepseek-v2-lite | 1794 | 1131 | 0.63 | 2027 | 823 | 0.41 |
| deepseek-v3 | 6446 | none | -- | 6525 | 3393 | 0.52 |
| **mean of 7** | | | **0.844 +/- 0.534** | | | |

deepseek-v3's fp8 up span has **no crossing at all**: its slope peaks at 0.497 at
T=8192 and never reaches the 0.5 threshold, under every filter tried including
`--include-throttled`. That is a real answer, not a gap: the grid does not bracket
the transition.

The mean moved from 0.44 to 0.84, toward the corrected theory's 1.00, which is
what removing a token-scaling bias should do. But 0.41 to 1.99 is a five-fold
range against 1.15 +/- 0.07 from the two production kernels. Nothing can be
concluded from a spread that wide.

The likely cause is the method, not the span. A single GEMM's time-against-T
curve is flatter and less structured than a fused layer's, so the slope has less
to cross and the 0.5 threshold lands wherever local noise puts it. `up` and
`down` are the same arithmetic on the same cells and disagree by 2.3x on mixtral,
which is not a property either dtype has.

**So dtype-invariance rests on vLLM and SGLang.** The one-stage span contributes
its times, which is what the span-extent separation below uses, and does not
contribute a third crossing measurement.

---

## C3. Below roughly 100 rows per expert, and in bf16, vLLM emits no warpgroup MMA

**ESTABLISHED, and rescoped three times. Read the rescopes; the raw claim
overstates it in three separate directions.**

`scripts/check_mma_path.sh` dumps the PTX Triton generates for a real
`fused_experts` cell (deepseek-v3, T=16) and counts instructions. Both compiled
`fused_moe_kernel` variants show `wgmma = 0`, `mma.sync = 16`, and every one of
the 32 tensor-core instructions is
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. `m16n8k16` is the
Ampere-era instruction, running on Hopper silicon.

Forcing `BLOCK_M >= 64` does reach `wgmma.mma_async.sync.aligned.m64n32k16` and
`m64n64k16`, verified with a fresh `TRITON_CACHE_DIR` so every specialisation
recompiles, and it is **1.7 to 9% slower**. 128 costs 27 to 30%. The capability
is there and declining it is correct: the tensor core idles waiting on weights
either way, so its throughput is irrelevant, while a short tile buys occupancy
and occupancy buys memory requests in flight.

`scripts/tile_sweep.py`, deepseek-v3, 50 timed iterations, uniform routing at
small T so every expert stays inside one tile:

| T | active | BLOCK_M=16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|
| 16 | 100 | 2.2194 ms | 0.996x | 1.017x | 1.270x |
| 64 | 225 | 4.8127 ms | 1.002x | 1.054x | 1.290x |
| 256 | 256 | 5.6687 ms | 1.000x | 1.090x | 1.298x |

16 to 32 doubles padded arithmetic and the time is flat to within 0.4%. That is a
direct measurement of "wasted MACs are free", with no byte model involved. It
stops being true above about 40% of the memory time, which is what the 128 column
locates.

### First rescope: it is not about "decode"

The kernel sees M, the rows entering the layer, and cannot tell whether they came
from one prefill or a thousand concurrent decodes. A serving system with enough
concurrency is in decode **and** at large M simultaneously, and there the tuned
config picks a taller tile and does emit wgmma.

The claim is about a regime in rows per expert, which batching can leave. Reading
the tuned H200 configs: `E=1,N=3072` steps 16 to 32 to 64 to 128 by M=128, while
`E=128,N=1024` stays on 16 until M=1536. That spread is `E/k` dilution appearing
in someone else's grid search, since a constant rows-per-expert threshold means
`M_switch` scales with `E/k`.

### Second rescope: it is bf16-specific

The same shapes tuned for fp8 pick a warpgroup tile at M=1:

```
E=8,N=14336    fp8   1:64, 2:64, 4:64, 8:64, 16:64 ...
E=8,N=14336    bf16  1:16, 2:32, 4:16, 8:16, 16:16 ...
E=64,N=2560    fp8   1:64, 2:64, 4:64 ...
E=64,N=2560    bf16  1:16, 2:16, 4:16 ...
```

At fp8 decode vLLM reaches wgmma immediately. C3 describes the bf16 path.

### Third rescope: the measured cell was running a fallback, not a tuned config

deepseek-v3 is `E=256,N=2048`, and **no tuned H200 config exists for it**. The run
log prints `Using default MoE config. Performance might be sub-optimal!`. So the
16 in that PTX comes from `get_default_config`'s hardcoded small-M branch (16 for
M<=32, 32 for M<=96, 64 for M<=512, 128 above) rather than from a grid search.

A GPU MODE reader raised exactly this and was right. It does not weaken the
measurement, since the fallback is what deepseek-v3 actually runs on this card,
but it changes what the 16 is evidence of: a default, not a tuned optimum. The
"an autotuner searched the space and gave away Hopper's headline feature"
reading, which an earlier write-up made, does not hold for this cell. It does
hold for the `E=8` and `E=64` shapes above, where a tuned config exists and picks
16 in bf16.

### One caveat that is not resolved

`BLOCK_SIZE_M` sizes the register accumulator, so a larger tile loses resident
blocks at the same time as it gains padded work and switches instruction. The
27 to 30% at 128 cannot be attributed among the three. It does not need to be:
the hypothesis under test was that bigger would help.

---

## C4. STREAM-style calibration understated achievable read bandwidth

**CONFIRMED and closed.**

A production kernel sustained 4483.4 GB/s where `calibrate_hardware.py` reached
4389.4 GB/s on a pure read. The cause was the **shape** of the read, not the
clock. `calibrate.py` measured it as `torch.sum(a, dim=0)` on a 1-D buffer into a
scalar: a full tree reduction, which bounds on ATen's reduction rather than on
DRAM. Reducing a 2-D view along the contiguous axis gives thousands of
independent reductions and no global combine.

```
read ceiling   4389.3  ->  4470.7 GB/s     +1.85%
```

That closes the anomaly. **82 rows** in the current arm report an implied traffic
ratio below the compulsory floor, and they are not scattered:

- all 82 are `vllm_fused_experts`
- all 82 are deepseek-v3
- all 82 are at T of 16, 32 or 64
- 27 are throttled and 55 are not, so throttling does not explain them
- peak is 4483.4 GB/s, and **zero rows anywhere exceed the 4916.7 GB/s pin rate**

4483.4 GB/s is **100.28% of the corrected read ceiling**: at the ceiling within
three parts in a thousand, not above it. Those kernels were running at
essentially 100% of achievable read bandwidth on pure weight streaming, which is
a strong result rather than a broken one.

The count is 82 on the current ruler and was **83** on the ruler of 2026-08-26.
Raising the ceiling by 2.3 GB/s moved one row from just under 1.00 to just over.
Both numbers are correct against their own calibration; the pair is the cleanest
demonstration in the study that the ruler moved.

A clock hypothesis was tested first and refuted. Settling under a memory load
rather than a matmul is correct in itself, and the memory settle converges at
1980 MHz as designed, but it moved triad by +0.05% and read by -0.00%: the
existing two-pass warmup had already handled the clock.

Confirmed independently on the A100, where the flaw is unmissable. The same
benchmark reports read at 1752.9 GB/s against triad's 1798.5, and triad moves
three times the bytes. `calibrate.py` now detects that case and refuses to name
read as a ceiling (`note: reduction-limited, not DRAM-limited`). On the H200 it
hid, because read landed just above triad.

---

## C5. Does the crossing scale with the ridge across cards?

**NOT ESTABLISHED. The measurement cannot answer the question, and the reason is
three uncontrolled variables rather than a subtle effect.** Rewritten 2026-08-31
after the original reading was found to be scored against the wrong target,
computed over routing regimes outside the model's domain, and taken across two
cards that were running different kernels.

### The prediction, and the target the original table used

For bf16 `b = 2`, so `2R/b = ridge` puts the crossing at `R = ridge` rows per
expert -- a DIFFERENT R on each card. Two cards should therefore show a
rows-per-expert ratio equal to their RIDGE ratio:

    A100 ridge 145.7,  H200 ridge 176.2  ->  target 0.827
    with the H200 ridge at the low end of its band (160.3) ->  target 0.909

so the target is a band, **0.81 to 0.91**, and it never reaches 1.00. Reaching
1.00 would need the A100 above its datasheet dense peak, or the H200 nine percent
below the worst of its six measured calibrations.

The original table scored the measured ratio against **1.00**. A ratio of 1.00
means both cards crossed at the same rows per expert, which is what NO ridge
scaling looks like. So the model reported as agreeing "to 1% across two
architectures" was agreeing with the null.

### Defect 1: the numbers pooled seven routing regimes

`2R/b` describes uniform routing. Under skew the busy experts are compute-bound
while the quiet ones are still memory-bound AT THE SAME BATCH, so the layer
straddles the ridge and there is no single crossing to find. Uniform is 14% of
the published cells; the other 86% are outside the claim's domain.

Pooling is not merely noisy, it is invalid. Two demonstrations:

- The cross-card ratio is not stable across routings even though both cards ran
  the IDENTICAL seven distributions. mixtral: uniform 0.72, zipf 0.44, hot 0.45,
  dirichlet 1.92. A 4.3x spread on a quantity whose two candidate values are
  0.83 and 0.82.
- Pooled deepseek-v3 crosses at 3474 with the saturation floor and at **14.6**
  without it, a 238x swing, because the pooled curve is still steep where the
  floor cuts. Uniform gives 3010 either way.

Restricted to uniform, `vllm_fused_experts` bf16:

| model | E | A100 | H200 | ratio | vs target 0.83 |
|---|---:|---:|---:|---:|---:|
| mixtral-8x7b | 8 | 229 | 316 | 0.725 | 0.88 |
| qwen2-57b-a14b | 64 | 742 | 787 | 0.943 | 1.14 |
| deepseek-v2-lite | 64 | 906 | 931 | 0.973 | 1.18 |
| deepseek-v3 | 256 | 2848 | 3010 | 0.946 | 1.14 |

RETRACTED WITH IT: the "deviation is monotonic in expert count" pattern, which
both this file and STUDY.md previously called the finding that survives and
attached the next experiment to. It is an artifact of pooling. Under uniform the
scores are 0.88 / 1.14 / 1.18 / 1.14 against E of 8 / 64 / 64 / 256 -- not
monotonic, and mixtral moves from worst to best.

### Defect 2: the crossings have no error bars, and they are wide

Times reproduce to 0.2%. Crossings do not. The crossing is interpolated between
two slopes with leverage `1/(s1 - s0)`, which is small on a flat curve, so the
detector amplifies timing noise about 10x. Measured directly: at A100 qwen2
T=512 throttling dropped one of two replicate rows, moving that single point 6%,
and the crossing moved from 593 to 824.

Propagating each cell's own replicate spread (`crossing.crossing_interval`,
4000 draws):

| model | ratio | 5th-95th | discriminates? |
|---|---:|---|---|
| mixtral-8x7b | 0.73 | 0.64 - 0.80 | rejects the target AND the null |
| qwen2-57b-a14b | 0.94 | 0.73 - 1.23 | no |
| deepseek-v2-lite | 0.97 | 0.89 - 1.08 | no |
| deepseek-v3 | 0.95 | 0.88 - 1.02 | no |

Three of four models cannot tell ridge scaling from the null. Every crossing
quoted anywhere in this study before 2026-08-31 was a point estimate with no band.

### Defect 3: the two cards never ran the same kernel

Verified against vLLM 0.27.1 by direct fetch of the shipped config tree. Configs
are keyed `(E, N, device_name)` with N the third dim of `w2`, and the tuned
lookup takes the NEAREST key, not the floor.

**Only 2 of the 8 model-by-card cells ran a tuned config.** Nothing ships for
`NVIDIA_A100-SXM4-80GB` at any of the four shapes, so all four A100 cells took
the hardcoded fallback: M<=32 -> 16, M<=96 -> 32, M<=512 -> 64, else 128.

What each card actually compiled for mixtral:

| T | A100 BM / BN / ISA | H200 BM / BN / ISA |
|---:|---|---|
| 64 | 32 / 64 / mma.sync | 32 / 128 / mma.sync |
| 128 | 64 / 128 / mma.sync | 64 / 128 / **wgmma** |
| 256 | 64 / 128 / mma.sync | **128** / 256 / wgmma |
| 512 | 64 / 128 / mma.sync | **128** / 256 / wgmma |

Three consequences, in increasing order of how much they matter.

**There is no grid point where the two cards ran the same kernel.** At T=128 the
tile heights finally match and the instruction still differs, because the A100 is
sm80: `getMMAVersionSafe` returns only `{2}` below compute capability 9.0, so it
is on `mma.sync.aligned.m16n8k16` at every tile it could ever run. The Hopper gate
is `BLOCK_M % 64 == 0 && num_warps % 4 == 0` (triton release/3.7.x,
`lib/Analysis/Utility.cpp` `supportMMA`), not the loose `>= 64` this study
previously quoted -- 80 or 96 would fall back.

**Every cross-card ratio is partly a warpgroup-MMA kernel measured against a
synchronous per-warp one**, not just mixtral's.

**And mixtral has a mechanism for its deviation.** Its H200 tile doubles from 64
to 128 at exactly T=256, which holds the M-tile count flat across the interval
where its crossing is interpolated:

    A100 (BM=64 throughout)      T=128:  8 tiles   T=256: 11 tiles   slope 0.363
    H200 (BM 64 -> 128 at 256)   T=128:  8 tiles   T=256:  8 tiles   slope 0.182

The tile doubling cancels the batch doubling, so the H200 streams the weights the
same number of times at twice the tokens, its time barely rises, and the
suppressed slope pushes its interpolated crossing later. The A100, on the
fallback ladder, keeps BM=64 and shows the real growth. Routing is not the cause:
the histograms are BYTE-IDENTICAL across the two cards at every shared T (same
active count, same max rows, same `tile_eff`).

DERIVED, NOT OBSERVED. No run log is committed anywhere, so the above is what
vLLM 0.27.1 WOULD resolve given the shipped configs and the `gpu_name` the CSVs
record. `get_moe_configs` logs `Using configuration from %s` on a hit and
`Using default MoE config. Performance might be sub-optimal!` on a miss; capturing
that line is two lines of code and turns this section from derived into measured.

### What was tested and did not explain mixtral

Recorded so the next reader does not re-run them: an expert-count trend (a pooling
artifact), grid or seed noise (the deviation reproduces across all three seeds,
A100 208-232 against H200 291-317), and throttle-exclusion bias (including the
throttled rows moves every ratio by 2% or less, despite the exclusion dropping
33% of H200 rows near the crossing and ~0% of A100 rows).

### The occupancy hypothesis is refuted

STUDY.md proposed that with 8 experts there may not be enough thread blocks to
fill 108 or 132 SMs. The vLLM grid is `cdiv(EM, BLOCK_M) * cdiv(N, BLOCK_N)`, and
the second term does not depend on expert count. At T=16, the smallest batch:

    mixtral        E=8    F=14336   3584 blocks    27 waves of 132 SMs
    qwen2          E=64   F=2560    5120 blocks    39 waves
    deepseek-v2-lite E=64 F=1408    2816 blocks    21 waves
    deepseek-v3    E=256  F=2048    8192 blocks    62 waves

Minimum over all models, both cards and every tile setting in any tuned config is
7 waves. Nothing is ever SM-starved. The ordering is inverted as well: mixtral has
the FEWEST experts and among the MOST blocks, because its experts are ten times
wider. Block count is dominated by expert WIDTH, not expert count.

### What would settle it

In order, cheapest first. Steps 1 and 2 are prerequisites, not options.

1. **Record the resolved tile config.** Schema v4 adds `tile_block_m/n/k`,
   `tile_num_warps`, `tile_config_source`, `tile_config_key` and `sm_capability`.
   Until a row states which kernel ran, no cross-device comparison is
   interpretable.
2. **Same-session calibration.** This arm's `measured.yaml` is byte-identical to
   one recorded 28 minutes after the sweep finished, at a different commit, and
   its own rows carry a third value. There is no principled point target, which
   is why the band above is 0.81 to 0.91 rather than a number.
3. **Pin one identical tile config on both cards and re-measure.** Bounds how much
   of the 0.73 was tile. It cannot make this a same-kernel comparison -- the A100
   has no wgmma at any tile -- but it separates tile from everything else.
4. **A same-architecture pair.** H100 SXM and H200 SXM are the same die: 132 SMs,
   identical compute, identical instruction set, 3.35 against 4.8 TB/s. SM ratio
   1.000, ridge ratio 1.68. That is the only pair in reach that isolates the
   roofline, and A100-vs-H200 never could: its SM ratio 0.818 sits within 1% of
   its ridge ratio 0.827, so the two hypotheses make the same prediction.

## The five-stage over one-stage separation: 0.563, and it is probably an artefact

**DOWNGRADED 2026-09-01, from "the most robust number in the study".** Read the
staircase section below before quoting any figure here.

THE DETECTOR RETURNS THE FIRST UPCROSSING OF 0.5, AND 8 OF 16 CANONICAL UNIFORM
CELLS CROSS TWICE. Taking the last instead moves the separation from 0.5602 to
0.8889, and mixtral and qwen2 go from 0.56 and 0.46 to 1.01 and 1.00, meaning the
two spans cross at the SAME batch.

    mixtral vLLM      313 then 800        qwen2 vLLM      730 then 1573
    deepseek-v3 vLLM 2925 then 6391       mixtral SGLang  313 then 778

AND THE A100 CROSSES TWICE WHERE THE H200 CROSSES ONCE. A100 mixtral uniform gives
229 AND 776 on the same octave grid where the H200 gives a single 313. So the
cross-card mixtral ratio in C5 compares the A100's FIRST step against the H200's
ONLY crossing, and taking the last on both reverses the sign entirely (776 / 313
is 2.5, against 0.73 for the first). There is no matched quantity to compare, which
is a cleaner reason the mixtral cross-card number is void than "different kernels".

WHY THE CURVE CROSSES TWICE: M-TILE QUANTISATION. M-tiles per expert is
`ceil(rows_per_expert / BLOCK_M)` and each extra tile is another pass over that
expert's weight matrix. mixtral with `BLOCK_M = 128` held CONSTANT across the
whole band (checked against the shipped tuned JSON, the config does not change
here):

    T=512   128 rows/expert   12 M-tiles   1.2224 ms
    T=576   144              15           1.3323   slope 0.731   tiles JUMP
    T=640   160              16           1.3991   slope 0.464
    T=704   176              16           1.4292   slope 0.223
    T=768   192              16           1.4437   slope 0.116   tiles FLAT
    T=1024  256              21           1.9088   slope 0.971   tiles JUMP

THOSE COUNTS ARE REPLICATE MEDIANS, corrected 2026-09-01. Uniform routing is
SAMPLED per replicate, so the tile count varies within a cell: T=576 draws
14/14/15/15/16/16 over its six rows and T=1024 draws 19/19/21/21/21/21. An earlier
version quoted 12/16/16/16/16/19, which is ONE draw and does not line up with a
median time. The STEP POSITIONS are unaffected, and they are what the mechanism
rests on.

AND 15 OF 16 CROSSINGS ARE TILE STEPS, NOT ALL 16. The exception is
deepseek-v3 / SGLang's second crossing at T 5120 to 5632, where tiles are already
saturated (511 to 512, every expert on two) and the slope merely grazes the
threshold, 0.473 then 0.634. The same model on vLLM puts its second crossing 1450
tokens later. State the exception rather than the round number.

Time JUMPS at a tile step and FLATLINES between, so the slope spikes above 0.5 at
every step. A first-passage detector reads a tile step, not a roofline transition.

AND THE TWO SPANS USE DIFFERENT TILES. The one-stage span is CUTLASS with
`BLOCK_M` fixed at 64 by the instruction set; the five-stage span is Triton with
`BLOCK_M` varying 16 to 128. Different tile heights put their steps in different
places, so 0.563 compared where two staircases have their first step rather than
anything about span extent. The CUTLASS-versus-Triton confound named below is not
a nuisance variable here, it is the whole effect.

WHICH READING IS RIGHT IS NOT SETTLED. Rows-per-expert at the LAST crossing is
mean 175.8 with CV 21%, against a measured ridge band of 160.3 to 176.2, which is
exactly what `2R/b` says R should equal. At the FIRST it is 123.4 with CV 40%.
That favours the last, but the dip is only visible because one arm added
T=576/640/704/768: on powers of two alone the slopes read 0.175, 0.587, 0.643,
0.791, perfectly monotone, staircase invisible. Four points revealed structure the
coarse grid hid, and whether more steps exist needs the dense sweep.

THE EXPERIMENT THAT DECIDES IT is not a denser grid alone. Pin `BLOCK_M` and sweep
it: if the measured crossing MOVES with the tile it is the ladder, and every
empirical MoE crossing including this one is measuring kernel configuration; if it
STAYS it is the roofline and the staircase is a wobble on top of a real transition.
`override_config` already does this in `scripts/tile_sweep.py`.

Everything below is the ORIGINAL analysis, retained because its arithmetic is
correct given the first-crossing reading, and because the contrast is the point.

### The original reading, first-crossing basis

A five-stage fused span crosses the ridge at **56% of the batch a one-stage
grouped GEMM does**, and no calibration uncertainty touches that figure.

bf16 measured/predicted, canonical pool, split by how much of the layer the span
covers. `F/H` is the expert's intermediate-to-hidden ratio.

| model | F/H | vLLM | SGLang | torch up | torch down |
|---|---:|---:|---:|---:|---:|
| mixtral-8x7b | 3.50 | 0.71 | 0.72 | 1.46 | 0.64 |
| qwen2-57b-a14b | 0.71 | 0.63 | 0.64 | 1.00 | 1.18 |
| deepseek-v2-lite | 0.69 | 0.54 | 0.60 | 1.05 | 1.19 |
| deepseek-v3 | 0.29 | 0.63 | 0.59 | 1.26 | 1.27 |
| **mean** | | **0.633 +/- 0.06** || **1.129 +/- 0.24** ||

The five-stage kernels sit at 0.63 with a tight spread. The one-stage grouped
GEMM sits at 1.13, which is `2R/b` predicting it about right. Same hardware, same
ridge, so the offset is **not** the kernel falling short of datasheet peak. It
belongs to the extra stages, whose permute, activation and unpermute traffic a
weights-only model never counted.

Those absolutes move with the ridge band. The comparison between span extents
does not, because both sides divide by the same predicted crossing and the ridge
cancels algebraically:

| | five-stage | one-stage | separation |
|---|---:|---:|---:|
| `2R/b`, ridge 160.3 | 0.633 | 1.129 | 0.561 |
| `2R/b`, ridge 176.2 | 0.576 | 1.028 | 0.561 |
| full byte model, ridge 160.3 | 0.578 | 1.027 | 0.563 |
| full byte model, ridge 176.2 | 0.521 | 0.925 | 0.563 |

AND IT SURVIVES THE RESTRICTION THAT KILLED C5. The table above is computed on
crossings pooled over seven routing regimes, which C5 shows is invalid for a
cross-CARD comparison. Recomputed on uniform routing only:

| | five-stage | one-stage | separation |
|---|---:|---:|---:|
| pooled, `2R/b`, ridge 160.3 | 0.633 | 1.129 | 0.5607 |
| **uniform only**, `2R/b`, ridge 160.3 | 0.553 | 0.987 | **0.5602** |
| **uniform only**, full model, ridge 176.2 | 0.452 | 0.807 | **0.5600** |

The absolutes move by 13%; the separation moves by 0.0005. It holds because both
spans run in the SAME session on the SAME card under the SAME routing, so any
routing distortion applies to both sides and cancels -- exactly the property the
cross-card comparison lacked.

THE CONFOUND THIS CLAIM STILL CARRIES, and it is not small. The one-stage span is
`torch.nn.functional.grouped_mm`: CUTLASS, tile fixed at 64 by the instruction
set. The five-stage span is Triton `fused_moe` with a tile that varies with batch
and device. So 0.56 is span extent CONVOLVED with CUTLASS-versus-Triton and
fixed-versus-variable tile. Separating them needs the same kernel measured at two
span extents, which means either fusing the reference spans or running vLLM's
kernel restricted to one stage. Neither exists, and this is the largest single
piece of work standing between this result and a defensible claim.

A second, smaller caveat: the one-stage crossings are internally inconsistent for
two models. Under uniform routing mixtral reports 332 (up) against 780 (down) and
deepseek-v3 reports 2751 against 6315, both about 2.3x apart, on what is the same
arithmetic over the same cells. The one-stage spread widens from +/-0.24 pooled to
+/-0.31 uniform for that reason.

It also survives a better byte model. The predictions above solve `2R/b`, the
weight-dominated limit of the general GEMM intensity

```
AI = 2MNK / ((MK + KN + MN) b)   ->   2M/b   when KN dominates
```

while every measured row is scored with the full model, activations included. Two
models on the two sides of one comparison. `2R/b` overstates AI by about 4% for
mixtral at its crossing and 7% for deepseek-v3, and overstating AI understates the
batch needed, so the `2R/b` predictions are systematically low by 5 to 18%.
`ridge.crossing_batch_full` bisects the same byte model the rows use.

The one-stage span lands within about 10% of prediction under every combination,
0.92 to 1.13, which is the agreement `2R/b` was reaching for. The five-stage span
sits at 0.52 to 0.63 whatever is done to the model. And the separation is 0.561
to 0.563 throughout, across an absurd range of ridges as well as the real one
(`tests/test_ridge_band.py`).

**Not claimed.** An earlier reading had the one-stage deviation ordered by expert
shape (`F/H`), matching `ridge.py`'s prediction that mixtral would deviate most.
That ordering came from an input set that double-counted a superseded arm. On the
canonical set the per-model one-stage means are 1.05, 1.09, 1.12, 1.27 against
`F/H` of 3.50, 0.71, 0.69, 0.29: monotonic in the means, but mixtral's internal
disagreement (1.46 up against 0.64 down) makes its mean unreliable and the effect
is too weak to assert.

---

## What a whole MoE layer costs, and how much of it is routing

`2026-08-28-...-h200-whole-layer`, 9,408 rows. The first complete-layer
measurement in this project: every framework span covers five of six stages and
leaves the router out, so until this arm the study could not say what a full layer
costs. `__pipeline__:vllm_fused_experts` times `ref_router` plus the fused kernel
as one cell, and `vllm_fused_experts` times the fused kernel alone in the same
run, so the router is the difference.

| model | T=1 layer | router | share | T=4096 share |
|---|---:|---:|---:|---:|
| mixtral-8x7b | 0.2707 ms | 0.0814 | 30.1% | 1.7% |
| qwen2-57b-a14b | 0.2019 | 0.0773 | 38.3% | 3.4% |
| deepseek-v2-lite | 0.1591 | 0.0762 | 47.9% | 7.1% |
| deepseek-v3 | 0.2867 | 0.0974 | 34.0% | 4.9% |

The absolute cost barely moves with batch, roughly 0.08 to 0.10 ms at T=1 and 0.10
to 0.50 ms at T=4096, which is launch and dispatch overhead for a matmul and a
top-k rather than work. So its share collapses as the batch grows while the number
itself does not. **At T=1, between 30% and 48% of an MoE layer is deciding which
experts to use.**

That is the same story the rest of the study tells from the other end: at decode
nothing is FLOP-bound, and what dominates is whatever does not scale.

**Caveat, and it bounds the claim.** This is `ref_router`, a PyTorch matmul plus a
top-k, the harness's reference. A production router is fused and faster, so 30 to
48% is an **upper** bound on the share, not a measurement of what vLLM spends. It
does establish that a whole-layer number is not the fused span's number, and how
much is missing.

**The crossing is unmoved, which is the confirmation.** RE-SCORED 2026-09-01 on
two counts. The ridge is 160.3, not 176.2: `entitled_ridge` refuses this arm
because its shipped `measured.yaml` is byte-identical to a calibration recorded
28 minutes AFTER the sweep ended, and the arm's own rows carry 701.6 TFLOP/s over
4377.2 GB/s. And the routing is uniform only, since pooling is invalid for a
crossing.

| model | predicted (160.3) | five-stage span | whole layer |
|---|---:|---:|---:|
| mixtral-8x7b | 641 | 316 | 327 |
| qwen2-57b-a14b | 1282 | 787 | 828 |
| deepseek-v2-lite | 1710 | 931 | 1020 |
| deepseek-v3 | 5130 | 3010 | 2888 |

Three to ten percent apart. A fixed cost added to a bandwidth-driven turning point
should not move it, and it does not. (The previous table read 705/1410/1879/5638
predicted and 543/914/897/3474 measured: the wrong ridge and the pooled-routing
crossings, both now retracted. The conclusion is unchanged, which is why it is
worth stating that the conclusion survived both corrections.)

THE ROUTER SHARE ABOVE IS UNAFFECTED BY EITHER CORRECTION. It is
`(pipeline - fused) / pipeline` within one run, so no calibration ceiling enters
it, and it is routing-robust: 30.1 / 38.3 / 47.9 / 34.0 pooled against
30.3 / 38.3 / 47.9 / 34.2 uniform. It is the one result this arm contributes that
does not depend on the ridge it is not entitled to quote.

---

## Supporting results

**Span extent is a trap.** `grouped_mm` covers 1 of 6 canonical stages;
`fused_experts` covers 5; `__pipeline__` covers 6. Comparing their milliseconds
compares a GEMM to a fused block, 16.7x apart on the published sweep. Recorded
per row in `covers`, enforced by `scripts/compare.py`, and the reason
`crossing_report` keys its medians on `impl`.

**Distance from the compulsory byte floor.** `implied_traffic_ratio` is bytes the
timing implies were moved over bytes the arithmetic requires. Basis is L2-cold,
eager, unthrottled: 3,225 of the recalibrated arm's 17,640 rows, of which 2,861
are memory-bound and therefore carry the column.

| implementation | span | n | min | median | max |
|---|---|---:|---:|---:|---:|
| vLLM `fused_experts` | 5 of 6 | 546 | 0.98 | **1.16** | 3.12 |
| SGLang `fused_experts` | 5 of 6 | 548 | 1.02 | **1.17** | 3.19 |
| torch `grouped_mm` | 1 of 6 | 1053 | 1.35 | **1.62** | 2.31 |
| reference pipeline | 6 of 6 | 714 | 9.50 | **12.43** | 24.66 |

THE 1.16 IS NOT THE NUMBER THAT MATTERS, corrected 2026-09-01. That median is
taken over the whole token grid, so it is pulled up by the compute-bound
transition where the ratio climbs (mixtral reaches 1.910 at T=512). Restricted to
uniform routing and to the MEMORY-BOUND regime a kernel would actually target:

| T | deepseek-v3 | mixtral | qwen2 |
|---:|---:|---:|---:|
| 1 | 1.176 | 1.172 | 1.254 |
| 8 | 1.124 | 1.080 | 1.046 |
| **16** | **0.981** | 1.035 | 1.039 |
| **32** | **0.977** | 1.083 | 1.029 |
| **64** | **0.984** | 1.057 | 1.055 |
| 512 | 1.083 | 1.910 | 1.293 |

median over T <= 64, uniform: **1.106**, and deepseek-v3 sits at 0.98, at or
below the compulsory floor. The 82 sub-floor rows are exactly these cells. So in
the regime this project cares about the incumbent has roughly ZERO headroom, not
15%. "Roughly 15%" was an artefact of averaging the compute-bound side in.

The two production kernels move about 1.16x the bytes their arithmetic requires
while covering five stages, ACROSS THE WHOLE GRID. That gap against `grouped_mm`'s 1.62x is not
straightforwardly a kernel-quality gap: a fused span amortises permute and combine
traffic that the single-stage span pays separately and the compulsory model counts
separately too. What it does support is narrower and still useful. The fused
implementations are close enough to the floor that the remaining headroom on this
axis is roughly 15%, and the reference pipeline at 12x is a correctness oracle,
not a performance baseline in any sense.

**Bimodality is real but cheap, and this one is not re-derivable here.** At
deepseek T=4096 zipf:2.0, 24 of 248 active experts hold 89% of the rows, and one
global tile costs only 1.00x to 1.18x of ideal weight traffic, so per-expert
tiling is a 5-15% target rather than a 2x one. This is the result that killed the
project's original premise. It survives only as prose: there is no test, no
script and no published table behind it, and the routing realisation it was
computed from cannot be regenerated off the GPU (see below). The published rows do
carry `load_gini = 0.91` and `load_entropy_norm = 0.58` for that cell, which is
consistent with it, but is not the same statistic.

**Padding is either zero or free.** Above batch 256 vLLM's autotuner sizes
`BLOCK_SIZE_M` to exactly rows-per-expert, so padding waste is 0%. Below it, waste
hits 50-100% and costs nothing, because 2 us of wasted arithmetic hides inside a
20 us weight read. Measured directly by the C3 tile sweep, not modelled.

**Routing is not reproducible off the GPU.** `cli.build_routing_source` passes
`device=args.device` into `routing_source`, so a GPU run draws its Gumbel keys
from a CUDA generator, and CUDA and CPU RNG produce different streams from the
same seed. Row totals match, since `T x k` is fixed; the distribution across
experts does not. Observed directly: mixtral T=2 uniform seed 0 records 4 active
experts on the GPU and 2 on the CPU. **Not fixed**, because fixing it changes the
routing of any future run relative to the published rows, and that is a decision
to take deliberately rather than as a side effect.

---

## The headroom is dtype-gated

Added 2026-09-01, and it is currently the strongest positive result in this study.

Production fused MoE kernels, uniform routing, T <= 64, which is the memory-bound
regime a kernel would target. Traffic ratio is achievable bandwidth over the row's
own `compulsory_gbps`:

| dtype | mode | n | p10 | median | p90 |
|---|---|---:|---:|---:|---:|
| bf16 | eager | 279 | 1.030 | **1.144** | 2.833 |
| bf16 | graph | 182 | 1.090 | **1.162** | 1.438 |
| fp8 | eager | 275 | 1.158 | **1.959** | 7.982 |
| fp8 | graph | 284 | 1.157 | **1.361** | 2.062 |

CORRECTED 2026-09-01, twice, and the surviving claim is narrower.

FIRST: bf16 IS NOT ON THE FLOOR. 1.144 eager / 1.162 graph is numerically the same
"roughly 15% headroom" the supporting-results section already reports for these
kernels. The defensible statement is that fp8 has MORE headroom than bf16, not
that bf16 has none.

SECOND: QUOTE THE GRAPHED ROW, NOT THE EAGER ONE. Decomposed over the 170 cells
measured in all four of (bf16, fp8) x (eager, graph), the fp8 eager figure is 78%
per-call HOST DISPATCH, and the stated mechanism cannot produce it:

    eager minus graph, in TIME, where a fixed cost is fixed
      bf16   median graph 203.7 us   eager - graph    4.9 us
      fp8    median graph 139.0 us   eager - graph  131.3 us

A cost that is merely fixed in time, divided by a floor that fp8 halves, can
contribute at most 2x more to an fp8 ratio than a bf16 one. The measured ratio is
27x. So the fp8 code path does genuinely more host work per call, CUDA graph
capture replays it away, and production serving uses graphs.

WHAT SURVIVES is the GRAPHED RESIDUAL, +0.337, positive in 96% of matched cells
and in every model: in graph mode fp8 takes 0.68x the time for exactly 0.500x the
bytes, and that excess IS a fixed cost surviving a halved floor. Headline figures
are therefore bf16 1.162 against fp8 1.361 pooled, 1.475 matched.

So the bandwidth headroom that routing-aware dispatch work reports is DTYPE-GATED:
it appears once you quantise, and in bf16 it is not there. That positions with
RaMP (arXiv:2604.26039) rather than against it, and it is not a claim any of the
2026 MoE papers surveyed makes.

CAVEAT THAT MUST TRAVEL WITH THIS NUMBER. The fp8 arm carries
`achieved_peak_tflops = 0.0` because its calibration measured no fp8 ceiling, so
its `implied_traffic_ratio` column is EMPTY and the figures above were
reconstructed as `achieved_bw_gbps / compulsory_gbps`. That is arithmetically the
same quantity, but it means the headline number is not a published column, and it
comes from an arm `entitled_ridge` already refuses. A same-session fp8 calibration
is one line on a pod and it is a precondition for publishing this.

---

## The tile-corrected roofline, and why arithmetic intensity is bounded

Proposed 2026-09-01. NOT YET VALIDATED; the experiment that tests it is named at
the end. This is a closed form for the staircase, and it makes predictions the
uncorrected model does not.

### What of this is ours, checked against the literature 2026-09-01

arXiv:2608.13057 (TEMPO, 13 Aug 2026) independently measures the same staircase in
tokens-per-expert at BLOCK_M=128, and fits the same extra-tile L2 discount, which
is this section's `alpha`. So THE BYTE ACCOUNTING BELOW IS NOT NEW. What is not
found in TEMPO, RaMP, Yun or Sieve:

 - the CEILING `2 BM / (alpha b)` stated as a bound on arithmetic intensity, and
   its consequence that a tile height can put the compute roof permanently out of
   reach. TEMPO models TIME as a max-affine with two branches, which cannot
   express a bounded AI.
   AND MEASURED 2026-09-01: max-affine was implemented and run against these rows.
   It gives one stable answer on all 8 ambiguous cells, its advertised property,
   but it does NOT describe the stepped curves: p95 relative error 61-263% on the
   variable-tile Triton spans against 14-47% on the fixed-tile CUTLASS ones, and
   its single answer lands 3-9x BELOW the ridge band. "One crossing by
   construction" is also false along a measured grid: 14 of 16 cells show a
   reversal, because two planes cross once but the path a sweep walks through them
   need not. So the detector is not the problem; an estimator that structurally
   cannot see a staircase does not fit these curves.
 - the crossing as a FIXED POINT on a staircase, and therefore the possibility of
   several crossings or none.
 - the step-position law in GLOBAL batch, `T = n BM E/k`, validated across an 8x
   spread in `E/k`. TEMPO and RaMP both stay in tokens-per-expert.

AND THE TWO MEASUREMENTS OF `alpha` DISAGREE BY 3.3x. This repo refit 0.10 with a
CV of 12.8%; TEMPO fits `b2/b` about 0.33. That is not a detail, because `alpha`
sets which tile heights can ever reach the roof at all:

REFIT 2026-09-01 AND THE ANSWER IS 0.558, not 0.10 and not 0.33. Group-intercept
fit over 10,813 admitted rows, 3,124 of them able to move alpha at all, 90%
cluster-bootstrap band 0.529 to 0.588, placebo -0.002. Stable across models
(0.46-0.72), cards (0.53 / 0.57), timing modes (0.48-0.59) and routing
(0.44-0.63). The original 0.10 reproduces exactly on its own 151 rows, and
changing ONLY the estimator on those same rows gives 0.484: the 0.10 came from
minimising the CV of a POOLED ratio, an objective that falls 0.7% across its whole
range and lets alpha absorb a between-cell level trend running the wrong way.

    BLOCK_M    cap @0.10   cap @0.33   cap @0.558   ridge band 160.3-176.2
         16          160          48           29   NEVER at any alpha
         32          320          97           57   NEVER at 0.33 and 0.558
         64          640         194          115   NEVER at 0.558
        128         1280         388          229   crosses
        256         2560         776          459   crosses

AT THE REFITTED ALPHA, BLOCK_M OF 16, 32 AND 64 ALL CAP BELOW THE RIDGE. vLLM's
tuned configs run BLOCK_M = 16 through the entire decode range. So on this
hardware a decode-configured MoE kernel is structurally incapable of reaching its
compute roof, at any batch size.

AND ALPHA IS NOT A SCALAR, which the fit also shows: it drifts with BLOCK_M
(0.466 at 64, 0.625 at 128) and falls with GROUP_SIZE_M (0.570 at 1, 0.488 at 16).
The GROUP_SIZE_M direction is exactly what a swizzle-for-L2-reuse mechanism
predicts, which is mechanistic support rather than a nuisance. But GROUP_SIZE_M 32
and 64 have ZERO discriminating rows in the published pool, so "alpha varies with
the swizzle" is UNTESTED rather than established, and needs override_config
varying it at fixed batch.

### The formula

For one expert holding `r` rows, with `N_w` weight elements at `b` bytes each and
tile height `BM`:

    useful FLOPs  = 2 N_w r
    M-tiles       = ceil(r / BM)
    weight bytes  = N_w b (1 + alpha (M-tiles - 1))

the first tile reads the weights in full and each additional M-tile costs `alpha`
of a fresh read, since it re-reads the same B operand across its N-tiles and L2
absorbs part of it. So

    AI(r) = (2r/b) / Q(r),      Q(r) = 1 + alpha (ceil(r/BM) - 1)

`2R/b` is the `alpha = 0` or single-tile LIMIT of this, not the general case. `Q`
is a tile-quantisation penalty: exactly 1 while `r <= BM`, then stepping up by
`alpha` at every multiple of `BM`, flat in between. That is the staircase, in
closed form.

Scope: this handles the M direction only. The grid is two-dimensional,
`cdiv(EM, BLOCK_M) x cdiv(N, BLOCK_N)`, and the N direction re-reads the
ACTIVATIONS rather than the weights. In the weight-dominated regime that term is
second order, which is the same assumption `2R/b` already makes.

### Consequence 1: arithmetic intensity is BOUNDED

As `r` grows, `ceil(r/BM)` tends to `r/BM`, so

    AI  ->  2 BM / (alpha b)        INDEPENDENT OF r

AI does not grow without limit with batch. It saturates at a value set by the TILE
HEIGHT. And if that ceiling sits below the hardware ridge, the kernel can never
become compute bound at any batch size at all.

    ridge 160.3, bf16, alpha = 0.10
      BLOCK_M =  16  ->  AI cap  160   NEVER CROSSES (needs alpha < 0.0998)
      BLOCK_M =  32  ->          320   crosses
      BLOCK_M =  64  ->          640   crosses
      BLOCK_M = 128  ->         1280   crosses

vLLM's tuned configs run `BLOCK_M = 16` through the whole decode range, where the
cap is 160 against a ridge of 160.3 to 176.2. Whether a decode-configured MoE
kernel can EVER reach compute bound therefore turns on whether `alpha` is above or
below 0.0998, and this repo's own refit put `alpha` at about 0.10. That is a knife
edge, and it is measurable.

### Consequence 2: the crossing is a fixed point on a staircase

    AI(R) = ridge   =>   R = ridge b Q(R) / 2,   with Q a STEP function of R

A step function on both sides can have several solutions, or none inside a step.
So the multiple crossings recorded above are not a detector artefact; they are a
property of the equation. Solving it:

    | BLOCK_M | uncorrected 2R/b | tile-corrected | shift |
    |--------:|-----------------:|---------------:|------:|
    |      32 |            160.3 |          304.6 | 1.90x |
    |      64 |            160.3 |          208.4 | 1.30x |
    |     128 |            160.3 |          176.3 | 1.10x |
    |     256 |            160.3 |          160.3 | 1.00x |

### A DEGENERACY THAT MUST BE STATED

    tile-corrected, ridge 160.3, BM=128, alpha=0.10   ->  176.3
    UNcorrected, at the high end of the ridge band    ->  176.2
    measured mean rows/expert at the last crossing    ->  175.8

The first two are numerically identical for entirely different reasons, so the
measured 175.8 does NOT confirm this formula. An earlier draft of this section
claimed it did. It cannot: the two hypotheses predict the same number at
`BLOCK_M = 128`.

### The experiment that validates or kills it

Sweep `BLOCK_M` and measure the crossing at each. The uncorrected prediction does
NOT move with `BLOCK_M`; the tile-corrected one moves by the shift column, a 1.90x
spread between 32 and 256 which is not subtle. Three readouts from one sweep:

1. WHERE the steps land. Both the traffic and the occupancy mechanism predict
   `R = n BM`, so this confirms the steps are about tiles without saying which
   tile effect causes them.
2. WHICH WAY time moves in the MULTI-TILE regime. Bigger `BLOCK_M` means fewer
   re-reads (faster) but fewer blocks (slower). C3 measured slower at small T,
   but there every expert is one tile at any `BLOCK_M`, so there were no re-reads
   to save and only occupancy could move. That regime is not this one.
   Run it where wave count exceeds about 10 on BOTH sides of a step, so occupancy
   is saturated and any remaining movement is traffic.
3. WHETHER THE CROSSING SHIFTS BY Q, which is the direct test of the formula, and
   whether `alpha` fitted this way matches the 0.10 from the re-read refit.
4. AND THE CAP: force `BLOCK_M = 16` and sweep T as far as the grid allows. The
   formula says no crossing exists. If one appears, `alpha < 0.0998` and the cap
   is real but higher than assumed. If none appears, a decode-tuned MoE kernel is
   structurally incapable of reaching its compute roof.

WAYS THE FORMULA MAY NEED MODIFYING, to look for in the fit: `alpha` may itself
depend on `BLOCK_M` (a taller tile holds more of B resident, so L2 reuse changes),
on expert width (a wider expert evicts more), or on `BLOCK_N`. A single scalar
`alpha` is the simplest thing that could work and should be tested as such before
anything more elaborate.

---

## Three results the study measured and never reported

Added 2026-09-01. `cuda_graph` and `l2_flush` are FULLY SWEPT axes with measured
columns, and before today neither appeared once in this file or in STUDY.md. The
first is the largest single effect in the dataset.

### CUDA-graph replay is worth up to 2.87x at decode, and nothing at all to a single kernel

14,050 matched pairs, same cell, same L2 mode, both timed, neither throttled.
Median `ms_p50(graph) / ms_p50(eager)`:

| implementation | T=1 | 2 | 8 | 32 | 256 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| `sglang_fused_experts` | **0.349** | 0.565 | 0.749 | 0.963 | 0.983 | 0.996 |
| `__pipeline__:vllm_fused_experts` | **0.608** | 0.851 | 0.948 | 0.935 | 0.961 | 0.994 |
| `vllm_fused_experts` | **0.678** | 0.929 | 0.980 | 0.983 | 0.984 | 0.996 |
| `torch_grouped_mm_up` | 0.997 | 0.997 | 0.999 | 1.000 | 1.000 | 1.005 |
| `torch_grouped_mm_down` | 0.997 | 0.998 | 0.997 | 0.999 | 1.001 | 1.002 |

Per-call launch overhead, `eager - graph`, spans a HUNDREDFOLD across
implementations that compute the same thing:

    __pipeline__:vllm_fused_experts   36.2 us median   208.8 p90
    sglang_fused_experts              18.1            243.9
    vllm_fused_experts                 6.7            120.4
    torch_grouped_mm_down              0.3              2.7
    torch_grouped_mm_up                0.2              2.8

This is the study's own thesis measured on a different axis. At decode what
dominates is whatever does not scale, and here it is CPU dispatch rather than
weight traffic. The single-kernel CUTLASS span has nothing to remove, which is
the control that makes the fused numbers mean something.

The crossings are unaffected: graph rows survive the cost policy only at small T,
below the crossing, so they carry no weight where the slope turns.

### L2 residency helps only where the working set fits, and that is a cliff

CUDA-GRAPH ROWS ONLY. Eager rows are unusable for this question: the 256 MB flush
kernel is itself sustained work that keeps the launch queue busy, so flushing
makes an eager cell FASTER, and that artefact swamps the cache effect. Bucketed by
weight footprint (`active_experts x 3FH x b`) over `l2_bytes`:

| footprint / L2 | n | median cold/warm | max |
|---|---:|---:|---:|
| **under 1x, FITS** | 84 | **1.0871** | 1.1710 |
| 1-2x | 213 | 0.9993 | 1.1419 |
| 2-4x | 403 | 0.9979 | 1.0721 |
| 4-8x | 894 | 0.9980 | 1.0408 |
| over 16x | 4122 | 0.9974 | 2.2420 |

Above 1.0 means warm L2 helped. It is a CLIFF at exactly 1x capacity, not a
gradient: a cyclic stream through a too-small LRU cache has near-zero hit rate,
because LRU evicts precisely what is needed next. Effective capacity is closer to
33 MiB than the nominal 60, so halve any bound computed from the datasheet.

### AND THE BENEFIT IS UNREACHABLE, which is the actual finding

Residency benefit and bandwidth utilisation are exactly anticorrelated. The cells
whose weights fit run at about 25% of achievable bandwidth; the 83 rows at 100.3%
of the read ceiling have footprints 20 to 1000x L2.

**The only cells where the weights fit are the cells that do not need the
bandwidth.** That is structural, not a sampling accident: saturating HBM needs
many waves, many waves needs many active experts, and many active experts is
hundreds of MiB. The two regimes cannot be occupied at once, which retires
hot-expert L2 caching with a mechanism rather than a null result.

### One expert can never hold most of the rows

    top1_share = max_rows / (T k),  and max_rows <= T,  so  top1_share <= 1/k

A token routes to k DISTINCT experts, so it contributes at most one row to any
one of them. Verified exactly on 64,669 published rows, zero exceedances, every
model hitting its bound:

    mixtral      k=2  ->  max observed 0.5000   bound 0.5000
    v2-lite      k=6  ->  0.1667                0.1667
    deepseek-v3  k=8  ->  0.1250                0.1250
    qwen2        k=8  ->  0.1250                0.1250

So for a high-`k` model there IS no dominant expert, however skewed the router.
Any argument of the form "cache the hot expert, it carries most of the traffic"
is arithmetically impossible above `k = 2`. And it is doubly wrong, because cost
tracks ACTIVE EXPERTS rather than rows: a cold expert holding one row still costs
a full weight read, so row share is the wrong ranking regardless.

This bounds the whole family of skew-exploiting designs, including several
proposed and discarded in this project.

---

## What this does not establish

- **DRAM traffic is modelled, not counted.** `ncu` needs a host module flag a
  container tenant cannot set (`ERR_NVGPUCTRPERM`), so every byte figure here is
  compulsory-traffic arithmetic. `nsys` does run and its `--gpu-metrics-device`
  route is untested. That is the open path, not a closed door.
- **C1 and C3 rest on transient pod output.** The PTX dumps, the CUTLASS kernel
  names and the `Using default MoE config` warning are quoted from run logs that
  were never committed. `scripts/check_mma_path.sh`, `scripts/kernel_name.py` and
  `scripts/tile_sweep.py` regenerate them on a GPU, but nothing in the repository
  lets a reader check them without one. Every other claim here can be recomputed
  from `results/published/` on a laptop.
- **The MACs-versus-weight-reads separation has not been run.** The cheap route is
  the GPU MODE method: alias B by taking the tile offset modulo so every iteration
  reloads the same tile (loads execute, L2 hits, no HBM traffic, nothing folds
  because the values are runtime), with `acc += tl.sum(b) + tl.sum(a)` to keep the
  loads live. It settles the critical path without relying on the byte model.
- **Nothing separates a kernel-quality gap from a span-extent gap.** The traffic
  table is reported per span for that reason. Settling it needs the fused
  implementations run at a single-stage extent, or the harness's own spans fused,
  and neither exists.
- **SGLang was configured by a default publish, not by a server.**
  `fused_experts` reaches process-wide config that only a running server normally
  publishes, so the harness publishes default `ServerArgs(model_path="dummy")`.
  The MoE path reads four leaves and all four are correct at single-GPU defaults,
  which makes the risk narrow but not zero.
- **DeepSeek-V3 routing is synthetic.** Its 1369 GB of bf16 weights do not fit on
  one H200, so its geometry is real and its token distribution is parametric.
  Mixtral and Qwen2 are the same way.
- **C5 is not established and cannot be settled from these rows.** See its section
  above: wrong target, invalid routing pool, and two cards running different
  kernels. `tests/test_c5_cross_card.py` pins the target so the scoring cannot
  drift back. The block-count hypothesis it proposed is separately REFUTED, on
  block-count arithmetic that needs no GPU.
- **Nothing in the published rows records which kernel ran.** Schema v4 adds the
  columns; the ten published v3 arms carry an explicit unrecorded sentinel that
  raises rather than returning a plausible default. Every tile-related statement
  about those arms, in this file and in STUDY.md, is derived from vLLM's source
  plus the recorded `gpu_name`, not observed.
- **Every routing distribution in this study is parametric.**
  `scripts/capture_traces.py` captures real per-layer expert histograms and works
  out that mixtral, qwen2 and deepseek-v2-lite all fit on one H200. It has never
  been run; `traces/` holds a single `.gitkeep`. Any claim about realistic skew
  rests on zipf, hot and dirichlet standing in for measurements never taken.
- **The sweep is unsharded and unquantized.** Every cell is TP=1 bf16. Real MoE
  serving shards, which changes `N` and therefore the config lookup, the tile and
  the block count, and quantizes, usually fp8 with `block_shape=[128,128]`.
  deepseek-v3 at `E=256,N=2048` is the UNSHARDED shape, needs 1369 GB, and is why
  no tuned config exists for it -- vLLM ships configs at N=256 and N=512, the
  TP=8 and TP=4 widths that are actually served. That is a limitation of this
  study, not a coverage gap in vLLM.
- **There is no machine-readable canonical arm set.** `published.py` prevents
  double-counting a superseded arm, which is the error that produced a wrong C2
  ordering. It does not say which arms belong in a pool. That is still carried in
  prose, here and in commit messages.

---

## Regenerating this file

```bash
# tests, all green off-GPU
.venv/bin/python -m pytest tests/ -q            # 605 passed, 34 skipped

# EVERY CROSSING BELOW IS UNIFORM-ONLY. Pooling the seven routing regimes is
# INVALID for a crossing, not merely noisy: 2R/b describes uniform routing, and
# under skew the layer straddles the ridge so there is no single crossing. Drop
# --routing uniform from any command here and the numbers change by up to 4.3x.

# every bf16 crossing, canonical four-arm pool
python scripts/crossing_report.py \
  results/published/2026-08-22-standard-sweep/run_*.csv \
  results/published/2026-08-26-nvidia_h200-full-three-way-recalibrated/run_*.csv \
  results/published/2026-08-28-nvidia_h200-ridge-resolution/run_*.csv \
  results/published/2026-08-28-nvidia_h200-h200-v2lite/run_*.csv \
  --ridge 160.3 --routing uniform --uncertainty

# fp8 crossings; the partial supersession is announced, not silent
python scripts/crossing_report.py \
  results/published/2026-08-28-nvidia_h200-h200-fp8-three-kernel/run_*.csv \
  results/published/2026-08-28-nvidia_h200-h200-fp8-refixed/run_*.csv \
  --ridge 160.3 --routing uniform --uncertainty

# C5, one card each. Quote the band, never the point estimate.
python scripts/crossing_report.py \
  results/published/2026-08-28-nvidia_a100_sxm4_80gb-a100-cross-card/run_*.csv \
  --ridge 145.7 --impl vllm_fused_experts --routing uniform --uncertainty
python scripts/crossing_report.py \
  results/published/2026-08-28-nvidia_h200-h200-whole-layer/run_*.csv \
  --ridge 176.2 --impl vllm_fused_experts --routing uniform --uncertainty

# the ridge band, the span-extent separation, C5's scoring, tile provenance
.venv/bin/python -m pytest tests/test_ridge_band.py tests/test_c5_cross_card.py \
  tests/test_crossing_uncertainty.py tests/test_tile_provenance.py -q
```

On a GPU, `scripts/kernel_name.py` (C1), `scripts/check_mma_path.sh` and
`scripts/tile_sweep.py` (C3), and `scripts/calibrate_read_variants.py` (C4).

WHAT CANNOT BE REGENERATED FROM THESE ROWS, and needs a pod:

- which tile config each published cell actually ran. Schema v4 records it going
  forward; the ten v3 arms carry an unrecorded sentinel and every tile statement
  about them is derived from vLLM's source, not observed
- vLLM's `Using configuration from` / `Using default MoE config` line, which is
  what would turn that derivation into a measurement
- PTX at the tuned specialisation (BM=128, BN=256), never compiled to disk, and
  any PTX at all from the A100
- a same-session calibration for the whole-layer arm, without which C5's target
  is the band 0.81-0.91 rather than a number

---

## What changed while this file was written

Regenerating instead of transcribing found four errors in `docs/STUDY.md`, which
had been the canonical document, and a further five in this file's own first
draft. All are corrected in place above; they are listed here so the pattern is
visible rather than buried.

**Found by recomputing STUDY's tables from the rows**

1. **C5 was scored against the wrong target.** `2R/b` predicts a cross-card
   rows-per-expert ratio equal to the RIDGE ratio, 0.83, and the table used 1.00.
   That inverted the result: deepseek-v3's 1.01 was reported as agreement "to 1%"
   when 1.00 is the no-scaling null.
2. **The ridge band was attributed to clock.** Across six H200 calibrations the
   GEMM clock moves 20.6% and the achieved rate moves 9.9% the OTHER way. The
   spread is in achieved efficiency, 71.4% to 93.2% of each run's own clock peak.
3. **A row in the fp8 one-stage table did not exist.** deepseek-v3's fp8 `up` span
   has no crossing at all, its slope peaking at 0.497. The mean over the 7 real
   values is 0.84 +/- 0.53, not 0.89 +/- 0.51.
4. **The A100 ridge is 145.7, not 146.6.**

**Found by adversarially re-checking this file's own first draft**

5. **The "deviation is monotonic in expert count" pattern is a pooling artifact.**
   Both documents called it the finding that survives and attached the next
   experiment to it. Under uniform routing it is gone.
6. **The tile-bin explanation for mixtral was wrong three ways** -- wrong
   `BLOCK_SIZE_M` (assumed 64 on both cards; the H200 runs 128 from a tuned file),
   wrong statistic (the mean, when `tile_eff_bm64/128` were already in every row),
   and backwards in direction.
7. **The occupancy hypothesis is refuted**, on block-count arithmetic that needs
   no GPU. Minimum 7 waves of 132 SMs at the smallest batch and largest tile.
8. **No crossing in this study had an error bar**, and the detector amplifies
   timing noise about 10x.
9. **The two cards never ran the same kernel**, and nothing in 94 columns recorded
   that.

**Two smaller ones, in code rather than prose.** `published.py` quoted the wrong
retract/keep row counts for the partially superseded fp8 arm (10,164 and 9,744,
not 9,576 and 9,408), and its illustrative anecdote about double-counting no
longer reproduces, because the saturation floor and the untimed-row filter both
landed after it was written.

**The pattern worth naming.** Four separate explanations for mixtral's deviation
were proposed and tested to destruction before the real one surfaced. Each was a
hypothesis about a variable the experiment never recorded. The fix was not a
better hypothesis; it was the column.
