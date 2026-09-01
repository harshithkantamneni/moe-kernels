# The instrument, and every way it distorted the measurement

What this document is for. `docs/FINDINGS.md` says what was measured.
`docs/STUDY.md` says how each claim got where it is. This file says what the
MEASURING APPARATUS did to the numbers on its way to producing them, one entry
per defect, each with the amount it moved a published figure and the check that
now stops it recurring.

It exists because this project's largest errors were not wrong hypotheses. They
were the harness, the detector, the calibration and the CSV schema quietly
answering a different question than the one asked, in a way that produced a
plausible number rather than a failure. A wrong hypothesis announces itself when
the data disagrees. An instrument defect agrees with everything.

Nine entries, ordered by how much each one moved a published number, biggest
first. Every figure below was recomputed from `results/published/` while writing
this file rather than transcribed from `docs/FINDINGS.md`, and the three that did
not reproduce are named as such in "What did not reproduce" at the end. That
section is the point of the exercise as much as the catalogue is.

Environment for every command here:

```bash
cd /Users/harshithkantamneni/Desktop/moe-kernels
.venv/bin/python -m pytest tests/ -q          # 874 passed, 34 skipped
```

---

## 1. Pooling routing regimes, which is INVALID and not merely noisy

**What the instrument did.** The sweep runs seven routing distributions per cell
(uniform, zipf at three exponents, hot, dirichlet, and so on) and writes them all
into one CSV with a `routing_kind` column. Every crossing report before
2026-08-31 took a median at each token count ACROSS those regimes and read a
slope off the result. `AI = 2R/b` is a uniform-routing statement: `R = T*k/E` is a
MEAN, and under skew no expert experiences the mean. The busy experts are
compute-bound while the quiet ones are still memory-bound at the same batch, so
the layer straddles the ridge and there is no single crossing for a slope
detector to find. What it finds is where a blend passes the threshold, which
moves with the mix rather than with the hardware.

Uniform is 1,344 of the 9,408 cells in each cross-card arm, 14.3%. The other 86%
were outside the model's domain and were being averaged in.

**How much it moved.** The largest instrument effect in the study.

Pooled deepseek-v3 on the whole-layer arm crosses at **3474** tokens with the
saturation floor and at **14.6** without it, a **238x** swing, because the pooled
curve is still steep where the floor cuts. The same cell restricted to uniform
gives 3010 either way, to the last bit. A number that moves 238x under a filter
the in-domain answer does not notice at all is not reading the ridge.

The cross-card ratio, mixtral-8x7b / `vllm_fused_experts` / bf16, one card each,
both cards having run the IDENTICAL seven distributions:

| routing | A100 | H200 | ratio |
|---|---:|---:|---:|
| uniform | 229.3 | 315.8 | 0.726 |
| zipf | 247.9 | 559.2 | 0.443 |
| hot | 243.4 | 535.2 | 0.455 |
| dirichlet | 254.0 | 132.1 | 1.922 |

A **4.34x** spread on a quantity whose two candidate values are 0.83 (ridge
scaling, which is what C5 claims) and 0.82 (the SM-count rival). Those differ by
0.01. The choice of routing flag moves the same number by 1.48, a hundred and
forty times the gap it was asked to resolve, so the measurement resolves the
analyst's flag and not the hardware.

Across the whole-layer arm, every model, switching from pooled to uniform:

| model | fused span pooled | uniform | whole layer pooled | uniform |
|---|---:|---:|---:|---:|
| mixtral-8x7b | 543 | 316 | 549 | 327 |
| qwen2-57b-a14b | 914 | 787 | 960 | 828 |
| deepseek-v2-lite | 897 | 931 | 1006 | 1020 |
| deepseek-v3 | 3474 | 3010 | 3375 | 2888 |

mixtral moves 1.72x. deepseek-v2-lite moves the other way. There is no correction
factor; the pooled number is not a biased estimate of the uniform one.

**How it was found.** By the C5 rewrite of 2026-08-31, recomputing STUDY.md's
cross-card table from the rows instead of carrying it forward. The tell was not
the ratio being wrong, it was the ratio being UNSTABLE: four routings, four
answers, on two cards that had run byte-identical histograms. Nothing in the
report had ever printed which routings fed a number.

**What now prevents it.** `crossing.routing_domain` counts the regimes behind any
selection and returns a `RoutingDomain` whose `inside` property is true only when
every row is uniform. `crossing_report.py` and `compare.py` print
`warning_lines()` as a banner AND `crossing_note()` beside each number, because a
banner forty lines above a figure does not stop the figure being quoted alone,
and every crossing this study retracted was quoted alone. Neither tool silently
switches to the uniform rows; it warns and names `--routing uniform`. Landed in
`023dd8e`, pinned by `tests/test_routing_domain.py`, 22 tests, including both
demonstrations above reproduced from the published rows.

```bash
.venv/bin/python -m pytest tests/test_routing_domain.py -q     # 22 passed
```

---

## 2. Span extent unstated, which compares a GEMM to a fused block

**What the instrument did.** The harness times four different SCOPES and writes
them into one `impl` column with milliseconds beside each. `covers` says what each
one is, and nothing was reading it:

| impl | covers | canonical stages |
|---|---|---:|
| `torch_grouped_mm_up` / `_down` | `up_gemm` / `down_gemm` | 1 of 6 |
| `vllm_fused_experts`, `sglang_fused_experts` | `permute+up_gemm+act+down_gemm+unpermute` | 5 of 6 |
| `__pipeline__`, `__pipeline__:vllm_fused_experts` | `all` | 6 of 6 |

The original grouping key in `crossing_report.py` was `(model, dtype)`, so a
median at one token count was taken across all of them.

**How much it moved.** mixtral-8x7b at T=512 on the recalibrated arm, medians per
implementation:

```
torch_grouped_mm_down    0.4116 ms      1 of 6 stages
torch_grouped_mm_up      0.8218
sglang_fused_experts     1.1549         5 of 6
vllm_fused_experts       1.1584
__pipeline__            14.6952         6 of 6
                                        spread 35.70x
```

A median across those describes nothing that ran. The number `docs/FINDINGS.md`
and `tests/test_crossing_report_grouping.py` both quote for this is **16.7x**;
see "What did not reproduce" below, because the measured spread is larger, not
smaller.

**How it was found.** By a merged CSV producing a crossing that matched no single
implementation. The scope was always in the row; the report was not keyed on it.

**What now prevents it.** `crossing_report.py` groups by `impl` rather than
trusting the caller to pass `--impl` (`0bb128b`), and `scripts/compare.py`
enforces equal `covers` before comparing two rows. `covers` is a schema column, so
the scope travels with every row rather than with the analyst.
`tests/test_crossing_report_grouping.py` pins that two implementations in one file
come back as two sections.

This one is a hazard the tooling now blocks rather than a published number that
was wrong, which is why it sits here rather than lower: the size of the mistake it
would make is 35x, and it was one grouping key away.

---

## 3. The crossing detector reads a TILE STEP, not a roofline transition

**What the instrument did.** `crossing_from_points` returns the FIRST adjacent
slope pair bracketing `d(log ms)/d(log T) = 0.5`. That is a first-passage
detector, and it is the right reduction only if the measured curve makes one
transition from flat to linear.

It does not. M-tiles per expert is `ceil(rows_per_expert / BLOCK_M)`, and each
extra M-tile is another pass over that expert's weight matrix. Time JUMPS when a
tile is added and FLATLINES while the tile count holds, so the slope spikes above
0.5 at every step and sags below it on every tread. The curve is a staircase, and
a first-passage detector reads its first step.

mixtral, `vllm_fused_experts`, uniform, tiles counted at BLOCK_M 128 from the
rows' own load columns (replicate medians, because uniform routing is SAMPLED per
replicate so the tile count genuinely varies within a cell):

```
T=512    128 rows/expert   12 tiles   1.2224 ms
T=576    144               15         1.3323   slope 0.731   tiles JUMP
T=640    160               16         1.3991   slope 0.464
T=704    176               16         1.4292   slope 0.223
T=768    192               16         1.4437   slope 0.116   tiles FLAT
T=1024   256               21         1.9088   slope 0.971   tiles JUMP
```

**How much it moved.** Three published numbers.

**8 of the 16 canonical uniform cells cross 0.5 upward twice.** The gap between
the two crossings is 2.0x to 2.6x, far outside any band the replicate spread puts
on either:

```
mixtral vLLM      313 then  800      qwen2 vLLM       730 then 1573
mixtral SGLang    313 then  778      qwen2 SGLang     730 then 1574
mixtral torch_up  332 then  789      deepseek-v3 vLLM 2925 then 6391
deepseek-v3 SGLang 3104 then 4941    deepseek-v3 torch_up 2751 then 6243
```

**The headline separation moves 59%.** Taking the last crossing rather than the
first moves the five-stage over one-stage separation from **0.5602 to 0.8889**;
the two arms move from 0.553 / 0.987 to 1.032 / 1.161. Per model, mixtral and
qwen2 go from 0.56 and 0.46 to 1.01 and 1.00, which is the difference between
"a fused span crosses at half the batch a GEMM does" and "the two spans cross at
the same batch". One input, two findings, separated only by which step was read.

**The cross-card mixtral ratio reverses sign.** A100 mixtral uniform crosses at
**229.3 AND 775.8**; the H200 on the same octave grid crosses **once, at 315.8**.
So C5's mixtral ratio compared the A100's first step against the H200's only
crossing. First-against-first is 0.726; last-against-last is **2.457**. There is
no matched quantity to compare, which is a cleaner reason that number is void than
"the two cards ran different kernels".

Two slopes recorded here are impossible for a roofline, whose asymptote is 1.0:
qwen2 / vLLM reaches **1.720** and deepseek-v3 / vLLM reaches **1.782**.

**How it was found.** By an ACCIDENT of grid design, and this is the part worth
keeping. `2026-08-28-ridge-resolution` was swept to pin the ridge, and it is the
only arm carrying T = 576, 640, 704, 768. On powers of two alone mixtral's slopes
read 0.083, 0.175, 0.587, 0.643, 0.791, 0.866, 0.907: perfectly monotone, and the
detector finds exactly ONE crossing at 312.7. Add the four dense points and the
same cell crosses at 312.7 and 799.9. Four token counts revealed structure the
coarse grid hid, and nothing about the coarse grid looked wrong.

```
crossings on the FULL grid : [312.7, 799.9]
crossings on POWERS OF TWO : [312.7]
```

**What now prevents it.** `crossing.all_crossings_from_points` returns EVERY
upcrossing and its docstring carries the mechanism; `crossing_from_points` is
retained for the published figures and its docstring says it is not what a new
analysis should call. `upcrossings` returns an `Upcrossing` carrying the grid
interval whose slope rose through the threshold, which is deliberately NOT the
interval the crossing's own value falls in: qwen2's first crossing is 730 tokens
and the step that caused it is 1024-to-1152, so an annotation read off the
crossing would name the wrong step. `m_tiles_for_row` puts the tile count beside
the token count so the staircase is visible instead of arguable. Landed in
`f0c4adf`, pinned by `tests/test_multiple_crossings.py`, 39 tests.

```bash
.venv/bin/python -m pytest tests/test_multiple_crossings.py -q   # 39 passed
```

WHICH CROSSING IS THE RIDGE IS STILL NOT SETTLED. Rows per expert at the last
crossing is mean 175.8 with CV 21.2%, inside the measured ridge band of
160.3-176.2, which is what `2R/b` says it should equal; at the first it is 123.4
with CV 40.0%, below the band. That favours the last and does not establish it,
since it is one prediction scoring itself. The experiment that decides it is
pinning `BLOCK_M` and sweeping it, which `override_config` already does in
`scripts/tile_sweep.py`.

---

## 4. Two fully swept axes that no finding read, one holding the largest effect
in the dataset

**What the instrument did.** `cuda_graph` and `l2_flush` are FULLY SWEPT
factorial axes with their own columns. 52,186 current timed rows carry them:
17,646 graph against 34,540 eager, and 26,101 L2-cold against 26,085 L2-warm.
Neither column appeared ONCE in `docs/FINDINGS.md` or `docs/STUDY.md` before
2026-09-01. Checked directly: `git show f0c4adf~1:docs/FINDINGS.md | grep -ci
'cuda.graph\|l2_flush'` returns 0, and the same on STUDY.md returns nothing.

This is a different failure mode from the rest of the catalogue. Nothing was
distorted. The sweep paid for two axes, and every analysis marginalised over them
without saying so.

**How much it moved.** Nothing, and that is the finding: a **2.87x** effect sat in
the CSV unread. 13,565 matched pairs (same cell, same L2 mode, both timed,
neither throttled), median `ms_p50(graph) / ms_p50(eager)`:

| implementation | T=1 | 2 | 8 | 32 | 256 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| `sglang_fused_experts` | **0.349** | 0.565 | 0.749 | 0.963 | 0.983 | 0.996 |
| `__pipeline__:vllm_fused_experts` | **0.608** | 0.851 | 0.948 | 0.935 | 0.961 | 0.994 |
| `vllm_fused_experts` | **0.678** | 0.929 | 0.980 | 0.983 | 0.984 | 0.996 |
| `torch_grouped_mm_up` | 0.997 | 0.997 | 0.999 | 1.000 | 1.000 | 1.005 |
| `torch_grouped_mm_down` | 0.997 | 0.998 | 0.997 | 0.999 | 1.001 | 1.002 |

Per-call launch overhead, `eager - graph`, spans a hundredfold across
implementations computing the same thing:

```
__pipeline__:vllm_fused_experts   36.2 us median   208.8 p90
sglang_fused_experts              18.1            243.9
vllm_fused_experts                 6.7            120.4
torch_grouped_mm_down              0.3              2.7
torch_grouped_mm_up                0.2              2.8
```

The single-kernel CUTLASS span is the control that makes the fused numbers mean
something: it has nothing to remove and shows nothing.

The L2 axis carries a second, structural result. Bucketed by weight footprint
(`active_experts x 3FH x b`) over `l2_bytes`, CUDA-graph rows only, it is a CLIFF
at exactly 1x capacity and not a gradient: the only bucket where warm L2 helps is
the one where the working set FITS (n=84, max 1.171), and every bucket above 1x
sits at 0.998 to 1.000. A cyclic stream through a too-small LRU cache has near-zero
hit rate, because LRU evicts precisely what is needed next. Eager rows are
UNUSABLE for this question: the 256 MB flush kernel is itself sustained work that
keeps the launch queue busy, so flushing makes an eager cell FASTER, and that
artefact swamps the cache effect. That is the same instrument defect as the rest of
this file, in the axis that measures the instrument.

**How it was found.** By asking, on 2026-09-01, which swept columns no finding had
ever cited. Not by any result looking wrong.

**What now prevents it.** Nothing structural, and that is honest to state. The
result is reported in `docs/FINDINGS.md` under "Three results the study measured
and never reported" (`f0c4adf`). There is no test and no script asserting that
every swept axis is read, and no mechanism would have caught this: the axes were
correctly recorded, correctly varied, and correctly ignored.

---

## 5. A column that silently defaults to zero, meaning NOT APPLICABLE

**What the instrument did.** `driver.py` writes `implied_traffic_ratio` only when
a cell is memory bound, so every compute-bound row keeps the dataclass default of
`0.0`. In a CSV, `0.0` is a number: it survives a `float()`, it plots, it averages,
and it reads as "this kernel moved zero times the compulsory traffic".

**How much it moved.** In the recalibrated arm, **2,060 of 12,034 timed rows**
carry it, and every one is at T >= 1024 (token counts 1024, 2048, 4096, 8192)
because that is where the layer goes compute bound.

On the published traffic table's basis (L2-cold, eager, unthrottled, passed:
3,225 rows, of which 2,861 are memory bound):

| implementation | median including the zeros | median over rows the column applies to |
|---|---:|---:|
| `vllm_fused_experts` | 1.133 | **1.163** |
| `sglang_fused_experts` | 1.139 | **1.171** |
| `torch_grouped_mm_up` | 1.609 | 1.632 |
| `__pipeline__` | 11.776 | 12.433 |

And on a COUNT rather than a median it is not 2.6%, it is 26x. Rows below the
compulsory byte floor in that arm:

```
real sub-floor rows (0 < ratio < 1) :    82
naive count of ratio < 1.0          : 2,142
```

The 82 are not scattered. All 82 are `vllm_fused_experts`, all 82 are
deepseek-v3, all 82 are at T of 16, 32 or 64, and 27 are throttled against 55
not, so throttling does not explain them. The other 2,060 are the default.

This is the same shape as `ms_p50 = 0.0`, which means the cell never ran: a
skipped or uncapturable graph mode still writes a row, 8,848 of the canonical
pool's 30,660 rows are like that, and feeding them to a median made the first fp8
report conclude deepseek-v3 crossed at 2 tokens.

**How it was found.** The `ms_p50` version was found by a crossing of 2 tokens
against a prediction of thousands, which was absurd enough to chase. The
`implied_traffic_ratio` version was found by auditing for the same shape after the
first one, which is the only reason it was found at all: a median moving from 1.16
to 1.13 is not absurd, it is plausible.

**What now prevents it.** Partially. `crossing.timed_rows` drops `ms_p50 == 0.0`
and says why (`c999213`). `schema.tile_field` raises `TileConfigUnrecorded` rather
than returning a plausible default for the v4 columns, and `schema.UNRECORDED` is
a STRING that no numeric path accepts, precisely so a hole cannot be read as a
value. But **`implied_traffic_ratio` is a v3 float column and no guard covers it**:
`timed_rows` does not filter it, and the filter has to be written per analysis.
`moe/bench/efficiency.py` documents the two meanings and the ambiguity its
signature exists to resolve, and that documentation is the whole mitigation.

---

## 6. A ridge that moves 9.9% run to run on ONE card, and not because of clock

**What the instrument did.** Every efficiency column in the study divides by a
measured ceiling, and the H200 was recalibrated six times. Six DISTINCT
`measured.yaml` files ship inside the ten published arms:

| md5 | bf16 TFLOP/s | bandwidth GB/s | GEMM clock | % of that clock's peak | ridge | arms |
|---|---:|---:|---:|---:|---:|---|
| `686caaea` | 785.5 | 4375.2 | 1905 | 76.3 | 179.5 | first-smoke |
| `f262cb20` | 730.0 | 4375.6 | 1500 | 90.0 | 166.8 | standard-sweep |
| `e774ed1e` | 701.6 | 4374.7 | 1485 | 87.4 | 160.4 | full-three-way (superseded) |
| `17a5bca2` | 712.4 | 4377.0 | 1845 | 71.4 | 162.8 | full-three-way-recalibrated, ridge-resolution |
| `db981ff9` | 770.9 | 4374.5 | 1530 | 93.2 | 176.2 | fp8-refixed, whole-layer |
| `4d84542b` | 701.6 | 4377.2 | 1560 | 83.2 | 160.3 | fp8-three-kernel, v2lite |

**How much it moved.** Bandwidth reproduces to **0.06%** across all six. The
compute term does not: **9.9%** across the three that current arms quote, and
**12.0%** across all six. Every absolute measured-over-predicted figure in
`docs/FINDINGS.md` therefore carries the band **160.3 to 176.2 FLOP/byte**, and
across all six calibrations that band widens to 160.3 to 179.5.

**THE CLOCK IS NOT THE EXPLANATION**, which is worth stating because STUDY.md said
it was. Across the three the GEMM clock moves **20.6%** (1530 to 1845) and the
achieved rate moves 9.9% in the OPPOSITE direction: the run at 1845 MHz reached
71.4% of its own clock's peak and the run at 1530 MHz reached 93.2% of its. Across
all six the clock moves 28.3% (1485 to 1905) with the same inversion. Clock
normalisation does not collapse the band; the spread lives in achieved
EFFICIENCY, and what causes that is not established here. An 8192-cubed cuBLAS
GEMM measured for a few seconds on a rented pod has thermal state, neighbour load
and measurement duration confounded.

**How it was found.** Adversarially, by recomputing STUDY.md's claim from the
yaml files rather than repeating it. The clock explanation was in the document and
was internally coherent; the numbers point the other way.

**What now prevents it.** `tests/test_ridge_band.py` (`ce958e9`) pins that the
span-extent separation survives the whole band, because both sides divide by the
same predicted crossing and the ridge cancels algebraically. That protects the
COMPARISON, not the absolutes, and the absolutes are still quoted with a band by
hand. Nothing forces a new figure to carry one.

---

## 7. An arm shipping a calibration recorded 28 minutes AFTER its own sweep

**What the instrument did.** `moe/bench/hardware/measured_<device>.yaml` is one
file per device, every recalibration overwrites it, and `publish_results.sh`
copies whatever is in it at publish time. So an arm can ship a ruler it never
used, and nothing checked.

`2026-08-28-nvidia_h200-h200-whole-layer` swept 18:08:21 to 19:21:43 UTC at commit
`873183a9`. Commit `89f9f7a` overwrote the calibration file at 19:29:01, seven
minutes later. The arm published carrying it. Its `measured.yaml` is byte-identical
(md5 `db981ff9`) to the one beside `2026-08-28-nvidia_h200-h200-fp8-refixed`,
whose first row is stamped 19:49:38 -- **27.9 minutes after the whole-layer rows
stopped** -- at a different commit, `5687de86`. Nobody noticed for three days.

```
whole-layer : 2026-08-28T18:08:21 .. 2026-08-28T19:21:43   sha 873183a9
89f9f7a     : 2026-08-28 19:29:01 +0000  "H200 recalibration with the measured fp8 compute roof"
fp8-refixed : 2026-08-28T19:49:38 .. 2026-08-28T21:41:41   sha 5687de86
gap between whole-layer's last row and fp8-refixed's first row: 27.9 minutes
both arms' measured.yaml: md5 db981ff9  (identical)
```

**How much it moved.** The arm has TWO candidate ridges: its own rows carry
701.61 TFLOP/s over 4377.21 GB/s, giving **160.3**, and the file beside them
reports 770.92 over 4374.49, giving **176.2**. That is the 9.9% of entry 6, now
attached to a single arm that cannot say which is its own. The bill is claim C5:
`2R/b` scales with the ridge, so a cross-card prediction needs a ridge, and C5 is
scored against the band **0.81 to 0.91** rather than against 0.83.

**9 of 10 arms pass** the gate. **2 are entitled to no ridge at all**: the
whole-layer arm for the disagreement above, and `-fp8-three-kernel` because its
calibration measured no fp8 ceiling, which is why every one of its 19,908 rows
carries `achieved_peak_tflops = 0.0`.

**How it was found.** Not by the date and not by the commit. `checked_on` has DAY
resolution and the swap happened inside one day. The commit differing is the
NORMAL state, since the workflow is calibrate, commit the yaml, then sweep, so the
sweep runs one commit later by construction and six of ten arms differ this way
with nothing wrong. The only decisive signal is that `driver.py` stamps
`achieved_peak_tflops` and `achieved_bw_gbps` onto EVERY timed row, so the rows
themselves say which ruler they were computed against, and they disagreed with the
file next to them.

**What now prevents it.** `published.calibration_provenance` returns a verdict
plus its evidence, never a bare bool, with the ceilings decisive, the commit weak
and the date weaker. `published.entitled_ridge` returns `(ridge, why)` and the
reason comes back whether or not the number does, so an analysis that skips an arm
ANNOUNCES it. The committed report is rendered by the code that decides:

```bash
.venv/bin/python -m moe.bench.published results/published/*/ \
  > results/published/CALIBRATION_PROVENANCE.md
```

Verified while writing this file: the report regenerates BYTE-IDENTICAL to the one
committed. `tests/test_calibration_provenance.py` (`4d4e977`) regenerates and
compares byte for byte, so the rule and the document that states it cannot drift
apart.

---

## 8. A calibration ceiling that moves 1.85% with the SHAPE of the read

**What the instrument did.** `calibrate.py` measured the `read` bandwidth pattern
as `torch.sum(a, dim=0, out=scalar_sink)` on a 1-D buffer: a full tree reduction to
ONE value, which bounds on ATen's reduction tree rather than on DRAM. It is the
most reduction-limited shape available.

**How much it moved.** 1.85% on the ceiling, and 83 rows from "impossible" to
"excellent".

```
read ceiling   4389.3  ->  4470.7 GB/s     +1.85%
```

The anomaly it created: on the 2026-08-26 ruler, **83** rows implied a bandwidth
above the ceiling. On the recalibrated ruler it is **82**. Both counts are correct
against their own calibration, and the pair is the cleanest demonstration in the
study that the ruler moved -- raising the ceiling by 2.3 GB/s moved exactly one row
from just under 1.00 to just over.

Verified here on both arms:

```
2026-08-26-full-three-way              sub-floor rows (0 < ratio < 1): 83
2026-08-26-full-three-way-recalibrated sub-floor rows (0 < ratio < 1): 82
peak implied bandwidth, both arms: 4483.4 GB/s
```

4483.4 is **100.28% of the corrected read ceiling**: at the ceiling within three
parts in a thousand, not above it. Zero rows anywhere exceed the 4916.7 GB/s pin
rate, so nothing impossible ever happened. Those kernels were running at
essentially 100% of achievable read bandwidth on pure weight streaming, which is a
strong result that the instrument had been reporting as a violation.

**ON THE A100 THE SAME FLAW IS UNMISSABLE, AND ON THE H200 IT HID.** The A100
calibration reports read at **1752.9 GB/s** against triad's **1798.5**, and triad
moves THREE times the bytes. A pure read cannot be slower than 2R+1W at the DRAM
level. On the H200 the broken read landed just ABOVE triad, so nothing looked
wrong.

**How it was found.** By the DATA contradicting the INSTRUMENT: 83 rows implying
more bandwidth than the calibration said existed. A clock hypothesis was tested
first and refuted -- settling under a memory load rather than a matmul is correct
in itself and converges at 1980 MHz as designed, but it moved triad by +0.05% and
read by -0.00%, because the existing two-pass warmup had already handled the clock.
`scripts/calibrate_read_variants.py` exists to decide it and states the decision
rule in its own docstring: if some formulation beats 4483.4, C4 is confirmed and
the anomaly dissolves into a calibration artifact; if none does, the anomaly
survives and needs another explanation.

**What now prevents it.** Two things, and the second is the general one.
`calibrate.py` now reads a 2-D view along the CONTIGUOUS axis
(`torch.sum(a2d, dim=1, out=sink_col)`), giving thousands of independent reductions
with no global combine, and the traffic is still 1N. And it now DETECTS the
pathology structurally: if `read < triad` it stamps every pattern with
`note: reduction-limited, not DRAM-limited; not a valid ceiling` and raises
outright if `--ceiling read` was requested. That note is live in the committed A100
yaml today. Landed in `a6ee65d`, guard pinned by `tests/test_calibrate_settle.py`.

---

## 9. 76.7% of the canonical pool was measured from a dirty working tree

**What the instrument did.** `schema.git_provenance` records `git_sha` and
`git_dirty` per row, and 23,520 of the canonical pool's 30,660 rows carry
`git_dirty = True`. Those rows are not exactly reproducible from the commit they
name.

**How much it moved.** No measured number, and the shape of the risk is worth
stating precisely: it is not scattered noise, it is TWO WHOLE ARMS.

| arm | rows | dirty | |
|---|---:|---:|---:|
| `2026-08-22-standard-sweep` | 840 | 0 | 0.0% |
| `2026-08-26-...-full-three-way-recalibrated` | 17,640 | 17,640 | **100.0%** |
| `2026-08-28-...-ridge-resolution` | 6,300 | 0 | 0.0% |
| `2026-08-28-...-h200-v2lite` | 5,880 | 5,880 | **100.0%** |
| **pool total** | **30,660** | **23,520** | **76.7%** |

The main bf16 sweep, which carries almost every crossing in `docs/FINDINGS.md`, is
one of the two. Two further arms carry MULTIPLE commits in one arm
(`-ridge-resolution` has three, `-fp8-three-kernel` has four), which is a separate
smaller version of the same hole.

**How it was found.** By auditing the provenance columns for this file. No result
pointed at it.

**What now prevents it.** Detection but not prevention, and only partly.
`publish_results.sh` emits `**WARNING: some rows were measured from a dirty working
tree**` into the arm's summary, and `2026-08-28-...-h200-v2lite/SUMMARY.md` carries
that line today. But `-full-three-way-recalibrated` has a hand-written `README.md`
from `recompute_ceilings.py` instead of a generated `SUMMARY.md`, and **it does not
carry the warning**, so the largest dirty arm in the study advertises nothing.
Nothing refuses to publish a dirty arm, and nothing in the analysis path reads the
column.

---

## Where these numbers came from

Every figure above was recomputed against `results/published/` while writing this
file. The ones with a committed command:

```bash
# entries 1, 3: the routing domain and the staircase
.venv/bin/python -m pytest tests/test_routing_domain.py tests/test_multiple_crossings.py -q

# entry 6: the ridge band, and that the separation survives it
.venv/bin/python -m pytest tests/test_ridge_band.py -q

# entry 7: the calibration gate, regenerating its own report
.venv/bin/python -m moe.bench.published results/published/*/ | diff - results/published/CALIBRATION_PROVENANCE.md

# the crossings every entry above is scored against, uniform only, with bands
python scripts/crossing_report.py \
  results/published/2026-08-22-standard-sweep/run_*.csv \
  results/published/2026-08-26-nvidia_h200-full-three-way-recalibrated/run_*.csv \
  results/published/2026-08-28-nvidia_h200-ridge-resolution/run_*.csv \
  results/published/2026-08-28-nvidia_h200-h200-v2lite/run_*.csv \
  --ridge 160.3 --routing uniform --uncertainty
```

Entries 2, 4, 5, 8 and 9 were recomputed by direct reduction over the published
CSVs and have no committed script. That is itself a gap: five of the nine entries
in a document about instrument provenance are not regenerable by a named command.

---

## What did not reproduce

Three published figures did not come back at the value they are quoted at when
recomputed for this file. Reporting them is the point of recomputing rather than
transcribing.

**The 16.7x span-extent spread is real but mislabelled, and it understates.**
`tests/test_crossing_report_grouping.py` and `docs/FINDINGS.md` both quote
"0.439 ms (torch_grouped_mm_down, one stage) to 7.337 ms (__pipeline__), at
mixtral/T=512: a 16.7x spread". Neither millisecond is at T=512 and neither is
that implementation. Searching the published rows, `7.337` occurs on `__pipeline__`
at mixtral **T=32, hot routing**, and `0.439` on `torch_grouped_mm_up` at mixtral
**T=32, dirichlet routing**. So the ratio pairs two different routing regimes at
T=32, which is entry 1's defect appearing inside entry 2's evidence. The measured
spread at mixtral/T=512 is **35.70x** (0.4116 to 14.6952) and at T=32 uniform it is
**22.27x**. The claim survives with a bigger number; the receipt behind it does
not. The docstring is prose and nothing asserts it, so the test still passes.

**The L2 residency table's headline median does not reproduce.** Under every
reconstruction tried (per-row pairing, per-cell median-then-ratio, with and without
`run_id` in the pairing key, with and without the throttle filter), the
`under 1x, FITS` bucket comes back at **1.0628** against a published **1.0871**,
and the `over 16x` bucket's max at **1.0858** against a published **2.2420**. The
bucket COUNTS match exactly (84, 213, 403) and so does the FITS bucket's max
(1.1710), so the row sets are the same or nearly so, and the aggregation must
differ. The finding is unaffected: the cliff at 1x capacity reproduces in every
variant, with the FITS bucket the only one above 1.0 and every larger bucket at
0.998 to 1.000. The table names no command, which is why this could not be settled.

**The CUDA-graph pair count is 13,565, not 14,050.** Every per-cell median in the
published table reproduces to the digit (0.349, 0.608, 0.678, 0.929, 0.997) and so
does every launch-overhead figure (36.2/208.8, 18.1/243.9, 6.7/120.4, 0.3/2.7,
0.2/2.8), once the pairing key excludes `run_id`. With `run_id` IN the key the
count rises to 14,736 and `vllm_fused_experts` at T=1 reads 0.803 instead of
0.678. So the numbers are right and the stated `n` is off by 3.5%, and the
sensitivity of that cell to the pairing key is worth knowing before the figure is
quoted again.

**Two smaller ones.** `docs/FINDINGS.md` quotes the mixtral cross-card routing
spread as "uniform 0.72, zipf 0.44, hot 0.45, dirichlet 1.92" and
`tests/test_routing_domain.py` asserts 0.73 / 0.44 / 0.46 / 1.92; the exact values
are 0.7260, 0.4434, 0.4547, 1.9223, so the two documents round in opposite
directions on two of four cells. And the log-log slope of **1.720** attributed to
qwen2 is correct but is NOT the maximum in the pool: deepseek-v3 / `vllm_fused_experts`
reaches **1.782** on the same filter. Both are impossible for a roofline, which is
what the number is for, so the argument is unharmed and strengthened.

---

## Mitigated, versus merely recorded

**Structurally prevented.** A new analysis cannot repeat these without
deliberately defeating a guard.

- **Routing pooling** (1). `routing_domain` counts the regimes, both report tools
  warn in a banner and again beside the number, and neither silently substitutes
  the uniform rows.
- **Span extent** (2). `crossing_report.py` keys on `impl`; `compare.py` enforces
  equal `covers`; the scope is a column on every row.
- **A borrowed calibration** (7). `entitled_ridge` refuses and names both
  candidates; the committed provenance report is rendered by the deciding code and
  regenerates byte-identical.
- **The read-shape ceiling** (8). Fixed at the source, AND `calibrate.py` now
  detects the pathology from `read < triad` and refuses to name read as a ceiling.
  The A100 yaml carries that refusal today.
- **Untimed rows read as zero** (5, the `ms_p50` half). `timed_rows` drops them.
- **An unrecorded tile read as a number** (the v4 columns). `UNRECORDED` is a
  string no numeric path accepts and `tile_field` RAISES; `READABLE_VERSIONS` keeps
  the ten v3 arms readable while `read_csv` marks the hole rather than filling it.

**Reported, with the mechanism named, but nothing enforces it.**

- **The staircase** (3). `all_crossings_from_points` exists and its docstring
  carries the mechanism, but `crossing_from_points` still returns the first
  crossing and the published figures are still read off it. Which crossing is the
  ridge is unsettled, so nothing can be enforced yet. The 0.563 separation is
  DOWNGRADED in `docs/FINDINGS.md` rather than retracted or replaced.
- **The ridge band** (6). `tests/test_ridge_band.py` protects the comparison that
  survives the band; the absolutes still carry a hand-written band, and nothing
  forces a new figure to carry one.
- **`implied_traffic_ratio = 0.0`** (5, the other half). No guard covers it. It is
  documented in `moe/bench/efficiency.py` and has to be filtered per analysis.
- **The unread axes** (4). Now reported. No mechanism would have caught them and
  none exists now.
- **`git_dirty`** (9). `publish_results.sh` warns and one of the two dirty arms
  carries the warning. Nothing refuses to publish, nothing in the analysis path
  reads the column, and the largest dirty arm has a hand-written README with no
  warning in it.

---

## Who found what, which is itself the finding

Worth separating, because the pattern predicts where the next one is.

**Found by the author, from a result that looked wrong.** Two, and both were
absurd rather than merely surprising. The `ms_p50 = 0.0` default was found because
it produced a crossing at 2 tokens against a prediction of thousands. The read-shape
ceiling (8) was found because 83 rows implied more bandwidth than the calibration
said existed.

**Found by an adversarial recheck of documents that already read as finished.**
Six of the nine. The C5 rewrite of 2026-08-31 recomputed STUDY.md's tables from
the rows and found the wrong scoring target, the clock misattribution of the ridge
band (6), and the routing pooling (1). Re-checking FINDINGS' own first draft found
that the surviving "monotonic in expert count" pattern was a pooling artifact, that
no crossing in the study had an error bar, and that the two cards never ran the same
kernel. Auditing for the shape of a known bug found the `implied_traffic_ratio`
default (5). Asking which swept columns no finding had cited found the two unread
axes (4). None of these was prompted by a result looking wrong. Each was prompted
by re-deriving a number that had already been believed.

**Found by an outside reader.** One. A GPU MODE reader pointed out that the cell
behind C3's PTX dump was running vLLM's hardcoded fallback rather than a tuned
config, and was right. It did not weaken the measurement, since the fallback is what
deepseek-v3 actually runs on that card, but it changed what the 16 is evidence OF:
a default, not a grid-searched optimum. The "an autotuner searched the space and gave
away Hopper's headline feature" reading does not hold for that cell.

**Found by accident of grid design.** One, and the most uncomfortable. The
staircase (3) is visible only because `-ridge-resolution` happened to add T = 576,
640, 704 and 768 for an unrelated reason. On powers of two the slopes are perfectly
monotone and the detector returns one crossing. Four token counts stood between the
study's headline separation being 0.560 and being 0.889, and nothing about the
coarse grid looked wrong from the inside.

THE PATTERN. Every entry in this catalogue is a variable the experiment did not
record, or recorded and did not read. The fix was never a better hypothesis: four
separate explanations for mixtral's cross-card deviation were proposed and tested to
destruction before the real one surfaced, and each was a hypothesis about something
no column held. The fix was the column. Where a column already existed and no
finding read it (entry 4), the effect sitting unreported was 2.87x, larger than
anything the study had published.
