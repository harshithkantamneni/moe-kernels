# alpha by ablation, without the byte model   (c0644be)

card: NVIDIA H200
output directory: /workspace/results/gaps-nvidia_h200/alias_ablation/nvidia_h200-1modesum-2bm16-3design598d40ee59-4seed0-5flushtrue-6warmup300.0-7budget50.0-8trials3-5d0bc0eb
Everything below is written there as report.md, beside plan.json and cells.jsonl.
candidate values: agrees with alpha_refit.py on both rival values
ceilings: measured_nvidia_h200.yaml, read_stream roof 4612 GB/s, L2 60.0 MiB

## the prediction, before anything runs

  P1  alpha = 0.558, 90% band 0.529-0.588  (today's refit, 10,813 rows)
      TEMPO arXiv:2608.13057 fits 0.33; this repo's retracted value is 0.10.
      This ablation runs at GROUP_SIZE_M=1, where the refit's own split reads 0.570,
      so that is the closer comparison and both are printed.

  P2  alpha is not a scalar: at GROUP_SIZE_M=1 the reuse distance is one pass over
      one expert's weight block, so alpha should track PER-EXPERT BYTES against L2.

  PASS/FAIL below is against P1's band, and the report names which of the three
  candidate values the measured band actually supports.

## the design

compute mode sum   BLOCK_M 16   tile {'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 1, 'num_warps': 8, 'num_stages': 3}   replicates 9

  model                   E      K      N   MiB/expert  weight GiB  rows/expert   act frac
  mixtral-8x7b            8   4096  28672       224.0        1.75           16     0.0006
  mixtral-8x7b            8   4096  28672       224.0        1.75           32     0.0011
  mixtral-8x7b            8   4096  28672       224.0        1.75           64     0.0022
  mixtral-8x7b            8   4096  28672       224.0        1.75          128     0.0045
  qwen2-57b-a14b         64   3584   5120        35.0        2.19           16     0.0031
  qwen2-57b-a14b         64   3584   5120        35.0        2.19           32     0.0063
  qwen2-57b-a14b         64   3584   5120        35.0        2.19           64     0.0125
  qwen2-57b-a14b         64   3584   5120        35.0        2.19          128     0.0250
  deepseek-v2-lite       64   2048   2816        11.0        0.69           16     0.0057
  deepseek-v2-lite       64   2048   2816        11.0        0.69           32     0.0114
  deepseek-v2-lite       64   2048   2816        11.0        0.69           64     0.0227
  deepseek-v2-lite       64   2048   2816        11.0        0.69          128     0.0455
  deepseek-v3           256   7168   4096        56.0       14.00           16     0.0039
  deepseek-v3           256   7168   4096        56.0       14.00           32     0.0078
  deepseek-v3           256   7168   4096        56.0       14.00           64     0.0156
  deepseek-v3           256   7168   4096        56.0       14.00          128     0.0312
  control-l2-resident    32   2048   4096        16.0        0.50           16     0.0039
  control-l2-resident    32   2048   4096        16.0        0.50           32     0.0078
  control-l2-resident    32   2048   4096        16.0        0.50           64     0.0156
  control-l2-resident    32   2048   4096        16.0        0.50          128     0.0312

  L2 on the attached card: 60.0 MiB. P2 says alpha is near 1 above it and near 0 below.

  [PASS] every rung divides exactly, so no mask and no padding
          N % BLOCK_N, K % BLOCK_K and M-tiles % GROUP_M are all zero
  [PASS] the ladder has at least three rungs, so D(n) can be shown affine
          tile ladder [1, 2, 4, 8]; two rungs fit a line through two points and can never contradict the form
  [PASS] the ladder starts at one tile, which is the only source of W
          smallest rung is 1 tiles. D(1) is the denominator of every alpha here; without it W has to come from a byte model, which is the thing this experiment exists to avoid
  [PASS] the activation stream is small against the weight stream
          worst rung streams 4.55% as many activation bytes as weight bytes (limit 5%). This bounds the L2-capacity confound: the aliased variant frees L2, and what it could free it for is the activation re-stream
  [PASS] the aliased arm is L2-resident, so it removes traffic and not work
          extent 'block': the widest aliased footprint is 1.75 MiB against 25% of an L2 of 60.0 MiB (15.00 MiB). An aliased arm that misses is not an ablation of the weight read, it is a second copy of it, and D would be noise around zero
  [PASS] P2 is testable: the models straddle L2
          per-expert weight blocks run 11.0 to 224.0 MiB against an L2 of 60.0 MiB

## what this costs

  BOOKING   `alias_ablation.py --run` on one card. These are that RUN's figures,
            whether or not this invocation is one: a plan is read before there is a card.

  KERNEL    5.0 min   20 rungs x 3 passes x 9 replicates, priced from the byte model at 0.61
                        of this card's read roof, with time_kernel's ten-iteration floor applied
                        per rung. The probe adds 0.1 min of that.
  EXCLUDES              Triton specialisations (one per model per pinning, plus the
                        constexpr variant), tensor allocation and fill, and the
                        closed-form correctness reductions. THIS IS NOT A WALL FIGURE.
  WALL     13.7 min   the KERNEL figure at the one measured wall-over-kernel ratio in
                        this repository, 2.35x (replicate_noise_floor.py, mixtral_g1:
                        127 s of wall against a 54 s model), PLUS 2.0 min charged
                        outright for the probe: 10 distinct specialisations at 12 s each,
                        with their tensor sets, which that ratio was not measured over
                        (mixtral_g1 compiled one pinning). BOOK THIS ONE.
  OF WHICH               16% is the probe, and a probe that clears no pinning
                        stops the arm having spent only that. It is the cheapest
                        thing here that can refuse the expensive one.

  The 0.61 is MEASURED, on 2026-09-01, on five geometries; a probe that finds a
  faster pinning makes both figures over-estimates, which is the right direction.

  PROBE     2.2 min   `--dot-fallback refuse`: the widened sum grid, 7 sum
                        pinning(s) of 10, stops the arm at the probe when none of them clears
                        2.111 x the roof (9737 GB/s here). That is INVALID and it is the
                        cheapest question this arm can ask. The sum half widened DOWN in
                        warps on 2026-09-09 because 8 -> 4 warps was the only knob that
                        moved the rate (2869 -> 5500 GB/s); it was 1.3 min at three sum
                        pinnings before those four were added:
                           8 warps  3 stages  BLOCK_K 64
                           8 warps  4 stages  BLOCK_K 128
                           4 warps  5 stages  BLOCK_K 128
                           4 warps  4 stages  BLOCK_K 64
                           4 warps  6 stages  BLOCK_K 128
                           2 warps  4 stages  BLOCK_K 128
                           2 warps  5 stages  BLOCK_K 64

  FALLBACK  32.1 min   WALL for the dot-mode LOWER BOUND, if one is wanted anyway.
                        `alias_ablation.py --run --cell-budget-ms 200 --replicates 18 --models mixtral-8x7b,qwen2-57b-a14b,deepseek-v3 --no-probe --compute dot --num-warps 8 --num-stages 4 --block-k 128`
                        (13.7 kernel min). dot mode cannot answer P1 at all, so this
                        buys a bound and not alpha; the longer trial and the doubled
                        replicates are aimed at the governor oscillation that produced
                        the 28% placebo, and the sub-L2 model that failed placebo, form
                        and bracket is dropped. It read 33.4 min on 2026-09-09, when it
                        ran under the probe and reached dot mode by falling down the
                        fallback; at 2.111 x the roof nothing in that grid clears, so it now
                        names the dot pinning and skips the probe rather than booking
                        an outcome no probe can reach.

## what this design can see (MDE)

Effect under test: the three candidates span 0.1 to 0.558, so an interval wider than 0.11 cannot pick one.
Noise assumption: per-pass timing spread MEASURED over the 26 published reports; median 0.77%, worst 1.82%. Within-cell, so a FLOOR.
Design: 9 interleaved replicates per rung, ladder topping out at 8 tiles, D(1) held above 25% of a pass by the signal gate.

  at the median spread 0.77%: MDE on alpha 0.040 against the 0.11 limit  resolves the candidates
  at the worst  spread 1.82%: MDE on alpha 0.095 against the 0.11 limit  resolves the candidates

An MDE above the limit does not make a PASS wrong; it makes a FAIL uninformative, and
it is the number to raise --replicates against before the box is rented.

## the probe: can any pinning see DRAM at all

  --dot-fallback refuse: if no sum-mode pinning clears, the arm STOPS at the probe
  rather than measure a lower bound. That is INVALID and it costs only the probe.

  Every pinning below is timed on the SMALLEST model at (1, 2) tiles. The number that
  decides is the ALIASED ladder's achieved request bandwidth over the card's DRAM roof,
  h. Below 1 the shared non-DRAM path is the slower one, DRAM has slack in the normal
  arm, and no ablation of DRAM can move the clock however large alpha is. Above 1 DRAM
  binds, but binding is not enough to fit: r = 1/(h-1), so the bar is the one printed
  under the table and a pinning below it runs a ladder `bracket` is bound to void.

  warps  stages  BLOCK_K  compute      aliased GB/s   of roof   note
      8       3       64  sum               2637     0.572   
      8       4      128  sum               2873     0.623   
      4       5      128  sum               5512     1.195   
      4       4       64  sum               4235     0.918   
      4       6      128  sum               5502     1.193   
      2       4      128  sum               8934     1.937   
      2       5       64  sum              10655     2.310   
      8       4      128  dot               9011     1.954   
      8       3       64  dot               8761     1.900   
     16       4      128  dot               6171     1.338   

  measured read roof: 4612 GB/s; a pinning clears at 2.111 of it (9737 GB/s), which is the larger of 1.333 from `signal` and 2.111 from `bracket`: r = 1/(h-1), so a pinning below it runs a ladder `bracket` is bound to void.
  ADOPTED: {'num_warps': 2, 'num_stages': 5, 'block_k': 64, 'compute': 'sum'} at 10655 GB/s, 2.310 of the roof: the fastest sum-mode pinning that clears

## re-pinned by the probe; output directory is now /workspace/results/gaps-nvidia_h200/alias_ablation/nvidia_h200-1modesum-2bm16-3designcf3015e427-4seed0-5flushtrue-6warmup300.0-7budget50.0-8trials3-c02cb6d3

  [PASS] every rung divides exactly, so no mask and no padding
          N % BLOCK_N, K % BLOCK_K and M-tiles % GROUP_M are all zero
  [PASS] the ladder has at least three rungs, so D(n) can be shown affine
          tile ladder [1, 2, 4, 8]; two rungs fit a line through two points and can never contradict the form
  [PASS] the ladder starts at one tile, which is the only source of W
          smallest rung is 1 tiles. D(1) is the denominator of every alpha here; without it W has to come from a byte model, which is the thing this experiment exists to avoid
  [PASS] the activation stream is small against the weight stream
          worst rung streams 4.55% as many activation bytes as weight bytes (limit 5%). This bounds the L2-capacity confound: the aliased variant frees L2, and what it could free it for is the activation re-stream
  [PASS] the aliased arm is L2-resident, so it removes traffic and not work
          extent 'block': the widest aliased footprint is 1.75 MiB against 25% of an L2 of 60.0 MiB (15.00 MiB). An aliased arm that misses is not an ablation of the weight read, it is a second copy of it, and D would be noise around zero
  [PASS] P2 is testable: the models straddle L2
          per-expert weight blocks run 11.0 to 224.0 MiB against an L2 of 60.0 MiB

## measuring: 20 rungs to do, 0 already on disk

  2 of 18 rungs moved more than 5% between the two idle-instant SM clock samples of the RETIRED check. That check
  detects whether the first sample caught the idle boost, not throttling; the under-load
  DRIFT verdict is clock_drift_ok, and the interleaved order plus the placebo gate are what
  protect a paired difference from either. On the row they are sm_clock_idle_before_mhz
  and sm_clock_idle_after_mhz, renamed on 2026-09-09 so they cannot be read as the
  under-load pair sm_clock_start_mhz / sm_clock_end_mhz beside them.
  mixtral-8x7b|t2|bm16@1980MHz, qwen2-57b-a14b|t8|bm16@1950MHz

  DRIFT: 105 passes EXCLUDED across 20 rungs that carry per-pass records. A cell is excluded if and
  only if its clock drifted; a steady clock on either side of the roof's own operating
  point is the kernel's power state under this card's cap, is recorded as the LEVEL side
  below, and excludes nothing.
  deepseek-v3|t1|bm16/normal#0, deepseek-v3|t1|bm16/placebo#5, mixtral-8x7b|t2|bm16/normal#1, mixtral-8x7b|t2|bm16/placebo#2
  mixtral-8x7b|t2|bm16/normal#3, mixtral-8x7b|t2|bm16/placebo#4, mixtral-8x7b|t2|bm16/normal#5, mixtral-8x7b|t2|bm16/normal#6
  qwen2-57b-a14b|t2|bm16/placebo#0, qwen2-57b-a14b|t2|bm16/placebo#1, qwen2-57b-a14b|t2|bm16/normal#2, qwen2-57b-a14b|t2|bm16/normal#4
  qwen2-57b-a14b|t2|bm16/placebo#5, qwen2-57b-a14b|t2|bm16/placebo#6, qwen2-57b-a14b|t2|bm16/placebo#8, deepseek-v2-lite|t2|bm16/normal#0
  deepseek-v2-lite|t2|bm16/normal#1, deepseek-v2-lite|t2|bm16/normal#2, deepseek-v2-lite|t2|bm16/placebo#4, deepseek-v2-lite|t2|bm16/normal#5
  deepseek-v2-lite|t2|bm16/normal#7, deepseek-v3|t2|bm16/placebo#1, deepseek-v3|t2|bm16/normal#2, deepseek-v3|t2|bm16/normal#3
  deepseek-v3|t2|bm16/normal#4, deepseek-v3|t2|bm16/placebo#5, deepseek-v3|t2|bm16/normal#5, deepseek-v3|t2|bm16/placebo#6
  deepseek-v3|t2|bm16/placebo#7, deepseek-v3|t2|bm16/normal#8, control-l2-resident|t2|bm16/placebo#4, mixtral-8x7b|t4|bm16/placebo#1
  mixtral-8x7b|t4|bm16/placebo#2, mixtral-8x7b|t4|bm16/normal#4, mixtral-8x7b|t4|bm16/placebo#5, qwen2-57b-a14b|t4|bm16/normal#0
  qwen2-57b-a14b|t4|bm16/normal#1, qwen2-57b-a14b|t4|bm16/placebo#3, qwen2-57b-a14b|t4|bm16/placebo#5, qwen2-57b-a14b|t4|bm16/normal#6
  deepseek-v2-lite|t4|bm16/placebo#0, deepseek-v2-lite|t4|bm16/placebo#2, deepseek-v2-lite|t4|bm16/placebo#3, deepseek-v2-lite|t4|bm16/normal#4
  deepseek-v2-lite|t4|bm16/placebo#6, deepseek-v2-lite|t4|bm16/placebo#7, deepseek-v2-lite|t4|bm16/placebo#8, deepseek-v3|t4|bm16/normal#0
  deepseek-v3|t4|bm16/normal#1, deepseek-v3|t4|bm16/placebo#1, deepseek-v3|t4|bm16/normal#2, deepseek-v3|t4|bm16/placebo#2
  deepseek-v3|t4|bm16/normal#3, deepseek-v3|t4|bm16/placebo#3, deepseek-v3|t4|bm16/placebo#4, deepseek-v3|t4|bm16/normal#4
  deepseek-v3|t4|bm16/placebo#5, deepseek-v3|t4|bm16/normal#5, deepseek-v3|t4|bm16/normal#6, deepseek-v3|t4|bm16/placebo#6
  deepseek-v3|t4|bm16/placebo#7, deepseek-v3|t4|bm16/placebo#8, deepseek-v3|t4|bm16/normal#8, control-l2-resident|t4|bm16/normal#0
  control-l2-resident|t4|bm16/placebo#1, control-l2-resident|t4|bm16/normal#2, control-l2-resident|t4|bm16/normal#4, control-l2-resident|t4|bm16/placebo#6
  control-l2-resident|t4|bm16/normal#7, control-l2-resident|t4|bm16/placebo#8, mixtral-8x7b|t8|bm16/placebo#1, mixtral-8x7b|t8|bm16/normal#2
  qwen2-57b-a14b|t8|bm16/normal#0, qwen2-57b-a14b|t8|bm16/normal#2, qwen2-57b-a14b|t8|bm16/placebo#4, qwen2-57b-a14b|t8|bm16/normal#5
  qwen2-57b-a14b|t8|bm16/normal#7, deepseek-v2-lite|t8|bm16/normal#1, deepseek-v2-lite|t8|bm16/normal#2, deepseek-v2-lite|t8|bm16/normal#4
  deepseek-v2-lite|t8|bm16/normal#6, deepseek-v2-lite|t8|bm16/normal#7, deepseek-v3|t8|bm16/placebo#0, deepseek-v3|t8|bm16/normal#0
  deepseek-v3|t8|bm16/placebo#1, deepseek-v3|t8|bm16/placebo#2, deepseek-v3|t8|bm16/normal#2, deepseek-v3|t8|bm16/placebo#3
  deepseek-v3|t8|bm16/normal#3, deepseek-v3|t8|bm16/normal#4, deepseek-v3|t8|bm16/placebo#4, deepseek-v3|t8|bm16/placebo#5
  deepseek-v3|t8|bm16/normal#5, deepseek-v3|t8|bm16/normal#6, deepseek-v3|t8|bm16/placebo#7, deepseek-v3|t8|bm16/normal#7
  deepseek-v3|t8|bm16/placebo#8, deepseek-v3|t8|bm16/normal#8, control-l2-resident|t8|bm16/placebo#0, control-l2-resident|t8|bm16/placebo#1
  control-l2-resident|t8|bm16/placebo#2, control-l2-resident|t8|bm16/placebo#3, control-l2-resident|t8|bm16/placebo#5, control-l2-resident|t8|bm16/placebo#7
  control-l2-resident|t8|bm16/normal#8
  SKIPPED deepseek-v3|t4|bm16: fewer than 3 undrifted replicates in normal, placebo. Its ladder is not fitted.
  SKIPPED deepseek-v3|t8|bm16: fewer than 3 undrifted replicates in normal, placebo. Its ladder is not fitted.

  LEVEL: 18 of 18 rungs ran ABOVE 105% of the clock this card's roof was measured at (LEVEL high).
  That is a boosted memory-bound rung, not a sag: the time is at one clock and alpha is a
  ratio of such times. Kept. What is not comparable is a fixed-roof fraction, which this
  arm never forms. This arm writes no fixed-roof column at all: a reader who needs
  one takes the rung's own load clock, printed beside each id below, against the
  reference clock on the row. (It said 'reads roof_at_cell_clock' until 2026-09-09,
  naming a field these cells.jsonl rows have never had.)
  mixtral-8x7b|t1|bm16@1980MHz, qwen2-57b-a14b|t1|bm16@1980MHz, deepseek-v2-lite|t1|bm16@1980MHz, deepseek-v3|t1|bm16@1972MHz
  control-l2-resident|t1|bm16@1980MHz, mixtral-8x7b|t2|bm16@1980MHz, qwen2-57b-a14b|t2|bm16@1958MHz, deepseek-v2-lite|t2|bm16@1980MHz
  deepseek-v3|t2|bm16@1928MHz, control-l2-resident|t2|bm16@1980MHz, mixtral-8x7b|t4|bm16@1980MHz, qwen2-57b-a14b|t4|bm16@1905MHz
  deepseek-v2-lite|t4|bm16@1980MHz, control-l2-resident|t4|bm16@1980MHz, mixtral-8x7b|t8|bm16@1965MHz, qwen2-57b-a14b|t8|bm16@1950MHz
  deepseek-v2-lite|t8|bm16@1980MHz, control-l2-resident|t8|bm16@1980MHz

## the measurements

  model                  tiles   rows/e     normal    aliased          D    placebo
  mixtral-8x7b               1       16     0.4882     0.2749     0.2133     0.0001
  mixtral-8x7b               2       32     0.7296     0.4911     0.2386     0.0007
  mixtral-8x7b               4       64     1.4946     0.8548     0.6398    -0.0003
  mixtral-8x7b               8      128     3.0036     1.6193     1.3843    -0.0112
  qwen2-57b-a14b             1       16     0.6252     0.3372     0.2880    -0.0000
  qwen2-57b-a14b             2       32     0.8002     0.5352     0.2651     0.0010
  qwen2-57b-a14b             4       64     1.3366     1.0198     0.3168     0.0024
  qwen2-57b-a14b             8      128     2.3884     1.9853     0.4031     0.0271
  deepseek-v2-lite           1       16     0.2165     0.1262     0.0903     0.0000
  deepseek-v2-lite           2       32     0.2640     0.1957     0.0683    -0.0003
  deepseek-v2-lite           4       64     0.4367     0.3594     0.0773    -0.0000
  deepseek-v2-lite           8      128     0.7601     0.6496     0.1106    -0.0000
  deepseek-v3                1       16     3.7630     1.6710     2.0919     0.0407
  deepseek-v3                2       32     5.1950     3.2085     1.9865     0.0001
  control-l2-resident        1       16     0.1424     0.0795     0.0629     0.0000
  control-l2-resident        2       32     0.1881     0.1401     0.0480    -0.0001
  control-l2-resident        4       64     0.3048     0.2544     0.0505    -0.0001
  control-l2-resident        8      128     0.5588     0.4814     0.0775     0.0001

  D is the ablation difference and is the HBM cost of that rung's weight reads.
  placebo is a SECOND normal launch minus the first: two identical configurations,
  so it is the noise floor D has to beat and nothing else.

## the ISA check: did the aliased kernel really issue the loads?

  rung                          variant           PTX      ld.global   cp.async   global loads   mma.sync
  mixtral-8x7b|t1|bm16          normal           0040a8883f98         19          0             19          0
  mixtral-8x7b|t1|bm16          aliased          0040a8883f98         19          0             19          0
  mixtral-8x7b|t1|bm16          constexpr-alias  2d8c7f399500         18          0             18          0

  ld.global ALONE would read zero here: Triton pipelines its global-to-shared copies
  as cp.async at num_stages > 1, so the gate is on the SUM. normal and aliased are
  the same compiled kernel driven by three runtime scalars, so equal counts are a
  property of the design; constexpr-alias is the naive form and is shown folding or not.

## alpha, from the ablation alone, as a bracket

  Two estimators, because exactly one of them is right and which one depends on how
  L2 service and HBM service compose. DIFFERENCE subtracts the aliased ladder and is
  exact if the two costs ADD; DIRECT fits the normal ladder with the fixed cost taken
  from the aliased ladder's own n=0 intercept and is exact if the kernel runs at
  max(L2, HBM). For any alpha <= 1 they BRACKET the truth, and the bracket is narrow
  exactly when r, the aliased ladder's per-tile cost over one weight read, is small.

  model                  MiB/expert    W (ms)   fixed  difference  direct       r      R^2   resid/W   form
  mixtral-8x7b               224.0    0.1370  0.0952       1.280   1.117   1.392   0.9864    40.3%   affine
  qwen2-57b-a14b              35.0    0.2669  0.0787       0.070   0.508   0.890   0.9147     5.7%   affine
  deepseek-v2-lite            11.0    0.0750  0.0509       0.056   0.529   1.002   0.5065    14.9%   NEITHER
  deepseek-v3                 56.0    2.0919  0.1336      -0.050   0.395   0.735   1.0000     0.0%   affine
  control-l2-resident         16.0    0.0516  0.0242       0.057   0.553   1.109   0.4554    16.8%   not scored  (L2-RESIDENT CONTROL)

  form PASSES on R^2 >= 0.97 OR resid/W <= 10%. The second is there because a FLAT D(n) is alpha
  near zero, which P2 predicts below L2, and R^2 has no variance to explain there.

  model                  bracket           90% interval        per-rung alphas (difference)
  mixtral-8x7b         1.117 to 1.280       1.090 to 1.425   n=2:0.119  n=4:0.667  n=8:0.784
  qwen2-57b-a14b       0.070 to 0.508       0.066 to 0.548   n=2:-0.080  n=4:0.033  n=8:0.057
  deepseek-v2-lite     0.056 to 0.529       0.056 to 0.529   n=2:-0.244  n=4:-0.048  n=8:0.032
  deepseek-v3          -0.050 to 0.395      -0.051 to 0.399   n=2:-0.050
  control-l2-resident  0.057 to 0.553       0.056 to 0.554   n=2:-0.238  n=4:-0.066  n=8:0.033

  W is the measured time of ONE full pass over every expert's weight block, read off
  the n=1 rung. Both alphas are a fitted slope over that intercept, so every unit of
  time cancels and no bandwidth, byte count or ridge enters either number.

  POOLED (median over the non-control models): alpha is in 0.063 to 0.518, 90% interval 0.061 to 0.538
  against the refit's 0.558 (0.529-0.588) pooled and 0.570 at GROUP_SIZE_M=1.

## P2, reported and deliberately NOT gated

  A step in four points is a pattern, not a test. This study has already retracted one
  monotone-in-expert-count reading that was an artefact of pooling, so the ordering is
  printed and left for a design that varies the footprint continuously.

  deepseek-v2-lite         11.0 MiB/expert  below L2   alpha 0.056 to 0.529
  qwen2-57b-a14b           35.0 MiB/expert  below L2   alpha 0.070 to 0.508
  deepseek-v3              56.0 MiB/expert  below L2   alpha -0.050 to 0.395
  mixtral-8x7b            224.0 MiB/expert  above L2   alpha 1.117 to 1.280

## what else differs between aliased and normal

  BOUNDED. L2 capacity. The aliased variant leaves L2 to the activations, whose stream
  is 4.55% of the weight stream at the worst rung, so the most the freed
  capacity can be worth is that fraction of D. The output write is identical in both.

  BOUNDED BY MEASUREMENT. TLB, page behaviour and code path. The L2-resident control
  ran the same ladder on a geometry whose PER-EXPERT block fits in L2, so NORMAL
  has no HBM re-read to save. Its W is 0.0516 ms and its alpha brackets 0.057 to 0.553.
  Whatever survives there is not weight traffic.

  BOUNDED BY THE BRACKET, AND GATED. L2 service, and the cache-set distribution that
  makes it worse. Both variants push n W bytes through L2. At --alias-extent 'block' the aliased arm touches
  one BLOCK_N x K column block, 1.75 MiB, walked with the normal arm's own
  sequential K stride over thousands of lines rather than pinned to a handful of
  slices. Whatever it still costs shows up in the ALIASED ladder's own slope, r, and
  `bracket` FAILS the page if r exceeds 0.9, above which the two estimators
  land on the same side of alpha and the interval between them is not a bracket.

  ABSORBED BY CONSTRUCTION. Any cost that scales with the extra-tile count and is not a
  weight re-read lands in alpha, because (n-1) is the regressor. That is a property of
  the estimator and no control can remove it.

## what this run could see (MDE, on its own spread)

Effect under test: the three candidates span 0.1 to 0.558, so an interval wider than 0.11 cannot pick one.
Noise MEASURED IN THIS RUN: pass-to-pass pstdev over p50 across the interleaved replicates.
The plan's line above used the corpus's WITHIN-cell spread, which is a floor and not this.
Signal MEASURED IN THIS RUN: D(1) is 41.7% of the weakest one-tile pass, the number
signal_gate scores, against the 25% floor the plan had to assume.
Design: 9 interleaved replicates per rung, ladder topping out at 8 tiles.

  at the median spread 0.10%: MDE on alpha 0.003 against the 0.11 limit  resolves the candidates
  at the worst  spread 2.62%: MDE on alpha 0.082 against the 0.11 limit  resolves the candidates

An MDE above the limit does not make a PASS wrong; it makes a FAIL uninformative. This
ladder is already spent, so a CANNOT here means the interval below cannot pick a candidate,
and the lever is --replicates: the signal cleared its floor, so the scatter is what binds.

## gates

RESULT: VALIDITY every-rung-divides-exactly-so-no-mask-and-no-padding PASS N % BLOCK_N, K % BLOCK_K and M-tiles % GROUP_M are all zero
  [PASS] every rung divides exactly, so no mask and no padding
          N % BLOCK_N, K % BLOCK_K and M-tiles % GROUP_M are all zero
RESULT: VALIDITY the-ladder-has-at-least-three-rungs-so-D-n-can-be-shown- PASS tile ladder [1, 2, 4, 8]; two rungs fit a line through two points and can never contradict the form
  [PASS] the ladder has at least three rungs, so D(n) can be shown affine
          tile ladder [1, 2, 4, 8]; two rungs fit a line through two points and can never contradict the form
RESULT: VALIDITY the-ladder-starts-at-one-tile-which-is-the-only-source-o PASS smallest rung is 1 tiles. D(1) is the denominator of every alpha here; without it W has to come from a byte model, which is the thing this experiment exists to
  [PASS] the ladder starts at one tile, which is the only source of W
          smallest rung is 1 tiles. D(1) is the denominator of every alpha here; without it W has to come from a byte model, which is the thing this experiment exists to avoid
RESULT: VALIDITY the-activation-stream-is-small-against-the-weight-stream PASS worst rung streams 4.55% as many activation bytes as weight bytes (limit 5%). This bounds the L2-capacity confound: the aliased variant frees L2, and what it co
  [PASS] the activation stream is small against the weight stream
          worst rung streams 4.55% as many activation bytes as weight bytes (limit 5%). This bounds the L2-capacity confound: the aliased variant frees L2, and what it could free it for is the activation re-stream
RESULT: VALIDITY the-aliased-arm-is-L2-resident-so-it-removes-traffic-and PASS extent 'block': the widest aliased footprint is 1.75 MiB against 25% of an L2 of 60.0 MiB (15.00 MiB). An aliased arm that misses is not an ablation of the weig
  [PASS] the aliased arm is L2-resident, so it removes traffic and not work
          extent 'block': the widest aliased footprint is 1.75 MiB against 25% of an L2 of 60.0 MiB (15.00 MiB). An aliased arm that misses is not an ablation of the weight read, it is a second copy of it, and D would be noise around zero
RESULT: VALIDITY P2-is-testable-the-models-straddle-L2 PASS per-expert weight blocks run 11.0 to 224.0 MiB against an L2 of 60.0 MiB
  [PASS] P2 is testable: the models straddle L2
          per-expert weight blocks run 11.0 to 224.0 MiB against an L2 of 60.0 MiB
RESULT: VALIDITY level-the-clock-each-rung-ran-at-is-recorded PASS 0 of 18 rungs ran BELOW 95% of the clock this card's roof was measured at (LEVEL low) and 18 ran ABOVE 105% (LEVEL high, first mixtral-8x7b|t1|bm16). Both sides
  [PASS] level: the clock each rung ran at is recorded
          0 of 18 rungs ran BELOW 95% of the clock this card's roof was measured at (LEVEL low) and 18 ran ABOVE 105% (LEVEL high, first mixtral-8x7b|t1|bm16). Both sides KEPT and the side recorded on the row: the under-load clock is set per tile by the kernel's own power draw under this card's cap, alpha is a ratio of two times at that clock, and this arm forms no fixed-roof fraction. Only DRIFT excludes, and it excludes the pass
RESULT: VALIDITY ISA-the-aliased-kernel-issued-the-same-global-loads PASS 18 rungs compared. First, mixtral-8x7b|t1|bm16: normal 19 global-load instructions (ld.global 19, cp.async 0); aliased 19 (ld.global 19, cp.async 0). Every rung
  [PASS] ISA: the aliased kernel issued the same global loads
          18 rungs compared. First, mixtral-8x7b|t1|bm16: normal 19 global-load instructions (ld.global 19, cp.async 0); aliased 19 (ld.global 19, cp.async 0). Every rung's two launches are ONE compiled kernel. The constexpr-aliased kernel, the naive way to write this, issues 18 and DID fold, which is the hazard happening in front of you
RESULT: VALIDITY correctness-each-variant-reproduced-its-closed-form PASS 36 checks at relative RMS <= 1e-03; worst 1.34e-07
  [PASS] correctness: each variant reproduced its closed form
          36 checks at relative RMS <= 1e-03; worst 1.34e-07
RESULT: VALIDITY headroom-DRAM-binds-by-enough-for-bracket-to-hold PASS weakest model deepseek-v3: the aliased ladder delivers 9778 GB/s of weight requests against a measured read roof of 4612 GB/s, a ratio of 2.120 (limit 2.111, th
  [PASS] headroom: DRAM binds by enough for `bracket` to hold
          weakest model deepseek-v3: the aliased ladder delivers 9778 GB/s of weight requests against a measured read roof of 4612 GB/s, a ratio of 2.120 (limit 2.111, the larger of 1/(1-0.25) = 1.333 from `signal` and 1+1/0.9 = 2.111 from `bracket`, which scores the same h as r = 1/(h-1)). Above 1 the shared path IS faster than DRAM and DRAM did bind, but the bar is not 1: r = 1/(h-1) = 0.893 here, above MAX_BRACKET_R 0.9, so the two estimators do not bracket alpha and `bracket` voids the page this ladder paid for
RESULT: VALIDITY attribution-D-1-reaches-the-card-s-floor-for-the-bytes-i PASS weakest one-tile rung mixtral-8x7b|t1|bm16: D(1) is 0.2133 ms against a floor of 0.4074 ms for the same bytes at the card's measured read roof, a ratio of 0.524
  [PASS] attribution: D(1) reaches the card's floor for the bytes it removes
          weakest one-tile rung mixtral-8x7b|t1|bm16: D(1) is 0.2133 ms against a floor of 0.4074 ms for the same bytes at the card's measured read roof, a ratio of 0.524 (limit 0.25). Below 1 the difference is smaller than the fastest this card can move those bytes, so it is not the cost of moving them
RESULT: VALIDITY placebo-two-identical-launches-differ-by-far-less-than-D PASS worst rung qwen2-57b-a14b|t8|bm16: two identical configurations differ by 0.0271 ms against a D of 0.4031 ms, 6.7% (limit 10%)
  [PASS] placebo: two identical launches differ by far less than D
          worst rung qwen2-57b-a14b|t8|bm16: two identical configurations differ by 0.0271 ms against a D of 0.4031 ms, 6.7% (limit 10%)
RESULT: VALIDITY signal-the-weight-read-is-most-of-what-the-kernel-does PASS weakest one-tile rung deepseek-v2-lite|t1|bm16 has D(1) at 41.7% of its own time (limit 25%). Below that the difference is measuring something the weight-read l
  [PASS] signal: the weight read is most of what the kernel does
          weakest one-tile rung deepseek-v2-lite|t1|bm16 has D(1) at 41.7% of its own time (limit 25%). Below that the difference is measuring something the weight-read label does not cover
RESULT: VALIDITY form-D-n-is-affine-in-n-1-as-W-1-alpha-n-1-requires FAIL worst R^2 0.5065 on deepseek-v2-lite; scored on deepseek-v2-lite at R^2 0.5065 (limit 0.97) OR residual RMS 14.9% of W (limit 10%). A FLAT D(n) is alpha near ze
  [FAIL] form: D(n) is affine in (n-1), as W(1+alpha(n-1)) requires
          worst R^2 0.5065 on deepseek-v2-lite; scored on deepseek-v2-lite at R^2 0.5065 (limit 0.97) OR residual RMS 14.9% of W (limit 10%). A FLAT D(n) is alpha near zero, which P2 predicts below L2, and R^2 has no variance to explain there; a large residual is a wrong form and slope-over-intercept is not alpha. 1 of 4 model(s) clear neither
RESULT: VALIDITY control-an-L2-resident-expert-shows-no-extra-tile-cost PASS the control's bracket is 0.057 to 0.553 on a W of 0.0516 ms, against a top end of 0.395 for the weakest real model. It must be consistent with zero to within 0.
  [PASS] control: an L2-resident expert shows no extra-tile cost
          the control's bracket is 0.057 to 0.553 on a W of 0.0516 ms, against a top end of 0.395 for the weakest real model. It must be consistent with zero to within 0.15: its re-reads hit L2, so anything it does show is the size of an extra-tile cost that is not weight traffic
RESULT: VALIDITY bracket-r-1-so-the-two-estimators-sit-on-opposite-sides FAIL worst r is 1.392 on mixtral-8x7b (limit 0.9). r is the aliased ladder's per-tile cost over one weight read, and the difference estimator is (alpha - r)/(1 - r):
  [FAIL] bracket: r < 1, so the two estimators sit on opposite sides
          worst r is 1.392 on mixtral-8x7b (limit 0.9). r is the aliased ladder's per-tile cost over one weight read, and the difference estimator is (alpha - r)/(1 - r): above 1 that denominator changes sign, both ends land on the same side of alpha, and the printed interval is not a bracket
RESULT: VALIDITY resolution-the-interval-can-separate-the-three-candidate UNKNOWN 90% interval is 0.477 wide (limit 0.11). The candidates span 0.1 to 0.558, and an interval wider than half the largest gap between adjacent candidates cannot pi
  [NOT TESTABLE] resolution: the interval can separate the three candidates
          90% interval is 0.477 wide (limit 0.11). The candidates span 0.1 to 0.558, and an interval wider than half the largest gap between adjacent candidates cannot pick one. r, the aliased ladder's per-tile cost over one weight read, is 0.735 to 1.392, and r is what sets the bracket's width: the two estimators differ by roughly 2r/(1-r^2), and `bracket` voids the page above r = 0.9. Narrowing it means a cheaper aliased ladder: --alias-extent block spreads the alias over a whole BLOCK_N x K column block instead of one tile, and --probe finds the pinning that delivers requests fastest
RESULT: CLAIM P1-the-ablation-agrees-with-the-refit-alpha-0-558 PASS measured alpha is in 0.063 to 0.518 before noise and 0.061 to 0.538 after it, against the refit's 0.529-0.588: they overlap. the interval 0.061-0.538 contains t
  [PASS] P1: the ablation agrees with the refit, alpha = 0.558
          measured alpha is in 0.063 to 0.518 before noise and 0.061 to 0.538 after it, against the refit's 0.529-0.588: they overlap. the interval 0.061-0.538 contains this repo, retracted and TEMPO arXiv:2608.13057

VERDICT: REFUTED or VOID. 2 gate(s) failed: form: D(n) is affine in (n-1), as W(1+alpha(n-1)) requires; bracket: r < 1, so the two estimators sit on opposite sides
EXIT: 3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
