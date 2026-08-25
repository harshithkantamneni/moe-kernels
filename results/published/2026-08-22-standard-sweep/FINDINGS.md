# What the first standard sweep says

840 rows, run `92572c5216fb`, commit `65ebea9`, one H200 SXM (700 W, 132 SMs, 60 MiB L2,
torch 2.13.0+cu130, CUDA 13.0, driver 580.159.04, Triton 3.7.1). All 840 passed
correctness against an fp32 oracle, bf16 throughout, seed 0, fan-in initialised.

Sweep is six axes: 3 model geometries x 7 token counts (1..4096) x 5 routing
distributions x {L2-cold, L2-warm} x {eager, CUDA-graph replay} x 2 stages (up, down).
3 x 7 x 5 x 2 x 2 x 2 = 840.

Baseline under test is `torch.nn.functional.grouped_mm`. On this build it dispatches to
a CUTLASS Hopper grouped GEMM rather than a per-expert loop, per the torch source; that
is a source-reading inference, not a profiled fact, because `ncu` is blocked on a rented
pod (`ERR_NVGPUCTRPERM`). Ceilings are measured on this device the same afternoon
(`measured.yaml`), never datasheet numbers.

**Denominator note, stated up front:** the CSV's own `achieved_bw_gbps` column uses the
`triad` STREAM pattern (4375.57 GB/s). Sections below re-denominate against the `read`
pattern (4390.29 GB/s), which the calibration explicitly labels as the analogue of
streaming expert weights. The two differ by 0.34% and no conclusion here turns on the
choice, but a reader recomputing from the shipped columns will get triad numbers.

**What `throttled` means, since two sections below lean on rows that carry it:** the
detector is two point samples, not an average and not a time series. The SM clock is read
once before a cell's timing call and once after, and `throttled` is set when the second is
more than 5% below the first. It is directional, so a clock RAMPING UP can never trip it,
and it is blind to a dip that recovers before the second sample. A throttled row is not a
failed row: all 79 passed the fp32 oracle and none had its timing zeroed, which only
happens on a correctness failure. It means the measurement was taken across a moving
clock, so an absolute time from it is softer than one from a pinned-clock row. Ratios
taken WITHIN a cell are much less affected, because every routing in that cell was
measured under the same thermal conditions. Section 6 has the counts.

**DeepSeek-V3 caveat:** its 1369 GB of bf16 weights do not fit on one H200, so its
routing is parametric rather than replayed from captured traces. Its geometry is real;
its token distribution is synthetic. Mixtral and Qwen2 are the same way in this run.

---

## 1. The starting thesis was wrong

Going in, the expected story was *"a fixed `BLOCK_M` degrades under skewed routing."*
It does not. Within one cell, where routing is the only thing that varies
(deepseek-v3, up-stage, T=16, L2-cold, eager):

| routing | max/mean | active experts | ms p50 | vs uniform |
|---|---:|---:|---:|---:|
| uniform | 6.00 | 100 | 1.9118 | 1.000x |
| zipf:0.6 | 12.00 | 88 | 1.7031 | 0.891x |
| dirichlet:0.3 | 16.00 | 68 | 1.3123 | 0.686x |
| zipf:1.2 | 30.00 | 49 | **0.9408** | **0.492x** |
| hot:0.5 | **32.00** | 94 | 1.7916 | 0.937x |

`hot:0.5` carries the maximum attainable imbalance at this geometry (32.00) and is the
second *slowest*. `zipf:1.2` at nearly the same imbalance is the fastest. Imbalance does
not order the times. Active expert count orders them perfectly: 49 < 68 < 88 < 94 < 100
maps onto 0.94 < 1.31 < 1.70 < 1.79 < 1.91 ms. That is a rank correlation on one cell of
five points (null probability 1/120), not a law.

**This is a small-batch result and it reverses.** What orders the times is active-expert
count, and a skewed routing moves that count in either direction. At qwen2 T=16,
`dirichlet:0.3` activates 33 of 64 and runs 0.650x, while `hot:0.5` activates 56 and runs
**1.074x**, slower than uniform, in the same cell. Concentration on one expert is not the
same thing as fewer experts.

Once uniform routing itself activates every expert, no routing can reduce the count and
the lever is gone. That happens at `T` of order `E/k`: mixtral (E/k=4) at T=16, qwen2 (8)
at T=64, deepseek (32) at T=256. Above it, the routings that tie uniform on active count
split both ways:

| cell | routings tied with uniform on active experts | spread vs uniform |
|---|---|---|
| mixtral T=256 | all four, 8 of 8 | 0.995x to **1.211x** |
| qwen2 T=256 | three of four, 64 of 64 | 1.003x to **1.052x** |
| deepseek T=1024 | three of four, 256 of 256 | 1.032x to **1.096x** |
| deepseek T=4096 | three of four, 256 of 256 | **0.796x** to 1.076x |

What is left once the count is tied is tile alignment. deepseek at T=4096 is the clearest
case: uniform lands on exactly 128 rows per expert, one whole `BLOCK_M`, so `hot:0.5`
beats it at 0.796x while `zipf:1.2` loses at 1.076x on the same 256 active experts. Read
"skew is fast" as "skew is fast while it keeps experts idle".

**Two of those four cells are entirely throttled, and it has to be said before the table is
used.** qwen2 T=256 has no throttled rows and mixtral T=256 has three of five, but deepseek
T=1024 and T=4096 are 5 of 5, the latter drifting 26.9% to 33.6% with clocks falling from
1965 MHz to as low as 1305. Two things keep the rows usable. The claim is a ratio taken
WITHIN a cell against that cell's own uniform baseline, which was throttled to the same
degree, so it is like-for-like in a way an absolute time would not be. And at T=1024 three
of the five routings were measured at bit-identical clocks, 1965 MHz to 1710 MHz at both
ends, and still spread 0.891x, 1.000x and 1.032x: a spread that appears at a fixed clock is
not a clock artefact.

The weak cell is T=4096, where the ordering partly tracks the drift. `hot:0.5` is fastest
and drifts least (26.9%); `dirichlet:0.3` is nearly slowest and drifts most (33.6%), which
is the direction a clock artefact would produce. Taking the midpoint clock of each, those
two differ by about 3%, against a 33% spread in time, so the clock accounts for a few
points of it and not the effect. The tile-alignment reading survives, but this particular
cell should be re-measured on a settled clock before it carries weight on its own.

## 2. What orders them is weight traffic; which unit it counts in is not resolved

The first version of this document claimed the explanatory variable was *active expert
count*. The second replaced it with the **M-tile**, on the reasoning that a tiled GEMM
reads its weight matrix once per M-tile: for output tile `(m, n)` it reads `B[:, n]`, so
summing over the `M/BLOCK_M` by `N/BLOCK_N` tile grid, `B` is streamed `M/BLOCK_M` times.
When every expert holds fewer rows than `BLOCK_M`, M-tiles and active experts are the
same number, which is why the simpler model appeared to work.

**That replacement was decided on a contaminated comparison.** Scoring each model as
`measured_ms / (predicted_bytes / 4390.29 GB/s)` (a correct model gives a *constant*
ratio, so spread is the score) over the 210 L2-cold eager rows pools two regimes that
answer different questions. Splitting them by `rows / active_experts` against the
measured 166 FLOP/byte ridge:

| traffic model | mean, all 210 | CV, all 210 | CV, 180 memory-bound | CV, 30 compute-bound |
|---|---:|---:|---:|---:|
| bytes = active_experts x W | 2.11x | 63.1% | **14.3%** | 44.2% |
| bytes = M_tiles x W, BLOCK_M=64 | 1.35x | 35.6% | 24.0% | 10.7% |
| bytes = M_tiles x W, **BLOCK_M=128** | 1.49x | 21.1% | **14.5%** | 7.2% |

**On the memory-bound rows the two models tie**: 1.67x at 14.3% CV counting active
experts, 1.59x at 14.5% counting M-tiles. The whole of the published 63% -> 21% gap came
from the 30 compute-bound rows, and on those an M-tile traffic model reads **0.93x**,
below 1, which would have the kernel moving bytes faster than the measured read ceiling.
A traffic model that beats the ceiling is not describing traffic. Above the ridge the
time is set by arithmetic, and M-tiles track arithmetic there because padded MACs are
exactly what M-tiles count, so those rows fit well for a reason that has nothing to do
with bytes. With them removed, this sweep does not separate the two units.

**`BLOCK_M=128` survives the split.** 54 of the 180 memory-bound rows have an expert
crossing a 64-row boundary, so the two tile counts differ and the row discriminates. Half
of those 54 are throttled, and the throttling inflates the argument, so the honest version
is the 27 that are not:

| rows | `BLOCK_M=64` | `BLOCK_M=128` |
|---|---|---|
| all 54 discriminating | 1.07x, 23.6% CV, **20 below the ceiling** (min 0.67x) | 1.38x, 5.6% CV, none below (min 1.25x) |
| 27 unthrottled | 1.21x, 17.0% CV, **4 below the ceiling** (min 0.73x) | 1.42x, **5.2%** CV, none below (min 1.28x) |

An earlier version of this section quoted the 20 without the split. On clean rows it is 4,
and the argument is unchanged in kind rather than in degree: a single row below the
measured read ceiling is already unphysical, `BLOCK_M=64` produces some on either subset,
and `BLOCK_M=128` produces none while fitting tighter. So the timing alone still bounds the
incumbent's effective M-tile at 128 rows, which is a fact `ncu` would normally be needed to
establish. Per model, over the 151 unthrottled memory-bound rows, the `BLOCK_M=128` fit is
1.55x at 12.0% CV (deepseek), 1.66x at 15.9% (qwen2), 1.70x at 9.5% (mixtral).

Absolute scale: one DeepSeek expert's `w1` is 7168 x 4096 x 2 B = 58.72 MB, and the
measured cost is 19.21 us per active expert at T=16, so 58,720,256 / 19.21 us =
3057 GB/s against a 4390 GB/s read ceiling.

**Honest scoping of the per-expert constant.** Across all 18 `(model, stage, T)` cells at
T in {16, 64, 256}, the spread of per-active-expert time across the five routings runs
**1.5% to 26.4%**, not the 1.5-7% originally printed here. The tight end is deepseek and
qwen2 up-stage (1.5-6.8%); mixtral is the loose end and supplies every large value
(mixtral down T=256 at 26.4%, up T=256 at 19.7%). Runtime *within* a cell still varies
about 2x while per-expert time varies far less, so the qualitative claim survives, but
the original band was a six-cell subset presented as a general result.

**Byte-model reconciliation.** The hand calculation above agrees with the harness's own
`compulsory_bytes` column to within 0.8% at T <= 64 and 2.7% at T <= 256. Above that they
diverge by up to 27.4%, because `bytes_model.py` counts activation traffic that the
weights-only calculation omits and that traffic scales with T while weights do not. The
deviation is one-signed on all 700 timed rows, which is the signature of exactly that
missing term rather than an error.

## 3. Padding is not free, and nothing in this sweep can say what it costs

Tile efficiency at `BLOCK_M=128` runs 0.0078 to 0.125 at T <= 64, 0.0078 to 0.500 at
T <= 256, and 0.0078 to 0.955 over the whole sweep. (An earlier version of this document
said "0.008 to 0.06 for T <= 256", which was wrong by 8x on the upper bound.)

The earlier version then argued that padding is free because tile efficiency varies while
time-per-active-expert does not. **That argument was circular.** Whenever every expert
holds fewer than `BLOCK_M` rows, tile efficiency is not an independent variable at all:

```
tile_eff = total_rows / (M_tiles * BLOCK_M) = (T * top_k) / (active * BLOCK_M)
```

Verified exactly, zero violations on all 134 rows with `max_rows <= 128`. At fixed T,
tile efficiency is literally `1/active` rescaled, so it is perfectly anti-correlated with
time by construction. Nothing about the cost of padding can be read off a sweep in which
the two are the same variable.

The cells that *can* answer it are the ones where active saturates and tile efficiency
moves independently. Mixtral pins all 8 experts active from T=64 up:

| T | routing | eff@128 | ms p50 | eff ratio | speed ratio |
|---:|---|---:|---:|---:|---:|
| 256 | dirichlet:0.3 | 0.400 | 0.7480 | 0.800 | 0.825 |
| 256 | zipf:1.2 | 0.444 | 0.7225 | 0.889 | 0.855 |
| 256 | hot:0.5 | 0.444 | 0.6853 | 0.889 | 0.901 |
| 256 | uniform | 0.500 | 0.6174 | 1.000 | 1.000 |
| 4096 | uniform | 0.928 | 3.2303 | 1.000 | 1.000 |
| 4096 | zipf:1.2 | 0.941 | 3.1801 | 1.015 | 1.016 |
| 4096 | hot:0.5 | 0.955 | 3.0515 | 1.030 | 1.059 |

Speed tracks tile efficiency close to proportionally. So padding is **not** free, and
this is the same effect section 1 called tile alignment: with active saturated, what is
left to move the time is where the group boundaries fall relative to `BLOCK_M`.

**But an earlier version of this section then named the mechanism, and it had no right
to.** It said the cost is the M-tile weight re-read rather than wasted MACs. Those two
make the *same prediction here*, and not approximately:

```
wasted MACs      -> padded rows = M_tiles * BLOCK_M
wasted weight reads -> traffic  = M_tiles * W
```

Both are linear in `M_tiles`, so at fixed model and stage they are one variable scaled by
the constant `W / BLOCK_M`. Predicting the table from either gives an identical column:

| T | routing | eff@128 | padded rows vs uniform | traffic vs uniform | ms vs uniform |
|---:|---|---:|---:|---:|---:|
| 256 | dirichlet:0.3 | 0.400 | 1.250x | 1.250x | 1.211x |
| 256 | zipf:1.2 | 0.444 | 1.125x | 1.125x | 1.170x |
| 256 | hot:0.5 | 0.444 | 1.125x | 1.125x | 1.110x |
| 4096 | zipf:1.2 | 0.941 | 0.986x | 0.986x | 0.984x |
| 4096 | hot:0.5 | 0.955 | 0.971x | 0.971x | 0.945x |

This is the same shape of error as the circular argument in the first version of this
document: two quantities that are the same variable, presented as a test between them.

The regimes make it worse rather than better. The T=256 block sits at 64 rows per active
expert and is memory bound at the mean, though its largest expert runs 77 to 209 rows and
so straddles the 166 ridge; the T=4096 block sits at 1024 rows and is entirely compute
bound. The two halves agreeing is not corroboration, because in one of them `M_tiles`
tracks arithmetic and in the other it tracks bytes.

What survives: padding costs something, and its size is roughly the tile-efficiency
ratio, moving a little less than predicted at T=256 and a little more at T=4096.

**Only a `BLOCK_M` sweep separates the two, and that is exactly why it works.** Changing
`BLOCK_M` breaks the proportionality that makes them indistinguishable here: padded rows
go as `M_tiles * BLOCK_M` while traffic goes as `M_tiles * W`, so their ratio `W/BLOCK_M`
stops being a constant and the two models finally predict different columns. The incumbent
cannot run that experiment because its `BLOCK_M` is not a knob. That is the first
experiment worth running, not an afterthought.

One more thing the sweep does settle, which is not about the mechanism: padded arithmetic
intensity is exactly `BLOCK_M` = 128 FLOP/byte regardless of model geometry, below the
measured 166 ridge. So even a kernel that genuinely computed every padding row would still
be memory bound while it did so.

## 4. Where the headroom is, after accounting for tile re-reads

Decomposing the harness's `implied_traffic_ratio` (a conditional bound: it says what
traffic *would* be implied if the kernel ran at the ceiling) into the M-tile factor and
what remains, deepseek-v3 up-stage, uniform, L2-cold:

| T | active | M-tiles | tiles/active | implied ratio | residual |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 1.00 | 1.69 | **1.69x** |
| 16 | 100 | 100 | 1.00 | 1.42 | 1.42x |
| 64 | 225 | 225 | 1.00 | 1.35 | 1.35x |
| 256 | 256 | 256 | 1.00 | 1.35 | 1.35x |
| 1024 | 256 | 256 | 1.00 | 1.31 | 1.31x |
| 4096 | 256 | 370 | 1.45 | 1.86 | 1.28x |

The residual is stable at 1.28-1.42x from T=16 up, and worst at T=1. That residual is
the part tiling does not explain, and at T=1 tiles and experts are identical so tiling
explains none of it.

Against the read ceiling, over all 120 rows at T <= 64 with no routing filter, the ratio
of measured time to the compulsory-bytes floor runs **1.351x** (deepseek up, T=64,
uniform, 74% of read BW) to **2.312x** (qwen2 down, T=1, zipf, 43%). The four worst
ratios are all T=1 rows. That gap is the target, and it is worst at exactly the shape
decode inference runs at.

What this cannot say, because `ncu` is blocked: whether that residual is extra bytes on
the wire, or compulsory bytes moved at below-ceiling efficiency (wave quantisation,
descriptor and TMA setup, memory-level parallelism at 8 concurrent groups). Both readings
leave the same headroom; they imply different kernels.

## 5. Two things that are confirmed dead ends

**CUDA graphs are not the lever.** 280 rows captured cleanly, zero capture failures, all
replay-verified against the oracle. Over the 140 matched L2-cold pairs the eager-minus-replay
median is **+0.54 us** (p10 -3.80, p90 +2.56) against kernels spanning 74.8 us to 1.614 ms.
The harness declined to time the other 140 graph cells, each with an explicit
`graph_skip_reason` recording that launch overhead was 0.14-0.93% of the roofline minimum,
below its 1% threshold.

**L2 absorbs nothing.** Over all 210 matched cold/warm pairs the ratio runs 0.9595 to
1.0081, median 0.9993, with 28 pairs outside +/-1% **in both directions** (the extreme is a
flushed run 4.05% *faster* than its warm twin, a sign impossible for a cache effect). That
two-sided spread is the result: it is noise, not absorption. The working set runs 146.8 MB
to 15.03 GB against a 60 MiB L2, 2.3x to 239x oversubscribed, so this is what theory
predicts. It matters because it means the compulsory-traffic model has no cache reuse
hiding inside it.

## 6. Measurement hygiene, recorded not hidden

79 of 840 rows tripped the throttle detector (SM clock dropping >5% between the sample
before a cell's timing call and the sample after it; see the note at the top for what that
does and does not mean). They
cluster on token count: 0 at T <= 64, 14 at T=256, 31 at T=1024, 34 at T=4096. Across
stages they are even (up 42, down 37); across models they are **not** (mixtral 34, qwen2
25, deepseek 20), and 13 of the 14 rows at T=256 are mixtral. Sustained draw on a 700 W
board, affecting only the compute-bound region. Every decode-regime conclusion above comes
from rows at T <= 64, where zero rows throttled. Throttled rows carry `throttled=True` and
stay in the CSV rather than being dropped.

The compute ceiling was measured at its settled 1500 MHz while the bandwidth patterns ran
at 1980 MHz, so the recorded 166 FLOP/byte ridge is conservative (185 if compute is
clock-normalised to match). Everything in the decode regime sits at AI 1-32, far below
either figure.

The detector's blind spot is worth stating: two samples cannot see a dip that recovers
between them, so the 79 is a lower bound on rows measured across a moving clock. 75 rows
went the other way and ended FASTER than they started, 65 of them at T >= 256, which is
recovery from a previous heavy cell rather than a warm-up. At T <= 64 there is almost no
headroom to ramp into: 412 of those 436 rows read 1980 MHz at both samples, the end clock
never falls below 1905, and not one row is throttled. The 24 exceptions start as low as
1500 MHz and finish at 1980, which is the ramp itself, and drift NEGATIVE so they cannot
trip a detector that only looks for a drop.

Correctness gating is structural: the driver zeroes timing fields on a row that fails the
oracle. It never fired here, since all 840 rows passed, so this run does not demonstrate
the gate, only that nothing needed it.

---

## What this means for the kernel

The problem statement:

> Move `active_experts x expert_weight_bytes` from HBM at closer to 4390 GB/s than the
> 43-74% the incumbent achieves, while M per group is 1-64 rows.

`active_experts`, not `M_tiles`, and in this regime the choice is free: across all 120
rows at T <= 64 no expert holds more than 64 rows, so every active expert is exactly one
M-tile at any `BLOCK_M >= 64` and the two counts are the same number, zero violations. `active_experts` is the one that stays physical when the regime changes,
since it counts weight matrices that must reach the SMs, while `M_tiles` counts a
scheduling decision the kernel is free to make differently. Section 2 is why the
distinction is stated rather than assumed: outside this regime the sweep does not
resolve which of the two the traffic actually scales with.

And the first experiment the incumbent cannot run, because its `BLOCK_M` is not a knob:
sweep `BLOCK_M` at fixed routing and separate wasted MACs from wasted weight reads. Every
number above is consistent with both, and they call for different kernels.

---

*Corrections: an earlier version of this file overstated the byte-model agreement (2% vs
0.8% at T<=64 and 27% overall), understated tile efficiency's upper bound by 8x, presented
a six-cell spread as an eighteen-cell result, called the cold/warm ratio "1.00 at every
point" when it is 0.96-1.01, and argued from a circular decorrelation in section 3.*

*This revision fixes three more. The section 2 traffic-model comparison pooled memory-bound
and compute-bound rows; split by regime the two models tie on the 180 memory-bound rows and
the published 63% -> 21% gap turns out to have come entirely from 30 compute-bound rows
where a traffic model has no business being fitted. Section 1 read as a general result when
it is a small-batch one: skew reverses sign once uniform routing saturates the experts, at
`T` of order `E/k`. The closing problem statement counted `M_tiles`, which the sweep
only pins down in the decode regime where it is identical to `active_experts`; it now counts
the latter. And section 3 named a mechanism it could not test: wasted MACs and wasted
weight reads are both linear in `M_tiles`, so at fixed model and stage they are one
variable, and the table offered as evidence between them predicts an identical column for
each. That is the same circularity the previous revision fixed elsewhere in this file. The numbers were re-derived from the raw CSV. Two of the three load-bearing
conclusions survive unchanged (graphs and L2 are dead ends, the decode gap is 1.35-2.31x);
"skew helps" is now scoped to below saturation.*

*A later pass added the throttling audit. The `BLOCK_M=64` count of rows below the read
ceiling was 20 of 54 with throttled rows included and is 4 of 27 without them, so the
figure now carries both. The traffic-model tie is unaffected: dropping throttled rows moves
it from 1.67x/14.3% versus 1.59x/14.5% to 1.67x/13.1% versus 1.64x/13.4%. Section 1's
sign-reversal table draws two of its four cells entirely from throttled rows and now says
so.*
