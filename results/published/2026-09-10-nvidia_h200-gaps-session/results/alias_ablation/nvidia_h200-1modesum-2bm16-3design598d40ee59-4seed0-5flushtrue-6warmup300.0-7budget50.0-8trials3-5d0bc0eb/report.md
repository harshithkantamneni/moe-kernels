# alpha by ablation, without the byte model   (c0644be)

card: NVIDIA H200
output directory: /workspace/results/gaps-nvidia_h200/alias_ablation/nvidia_h200-1modesum-2bm16-3design598d40ee59-4seed0-5flushtrue-6warmup300.0-7budget50.0-8trials3-5d0bc0eb
Everything below is written there as report.md, beside plan.json and cells.jsonl.
candidate values: agrees with alpha_refit.py on both rival values
ceilings: measured_nvidia_h200.yaml, read_stream roof 4613 GB/s, L2 60.0 MiB

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
                        2.111 x the roof (9739 GB/s here). That is INVALID and it is the
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
                        (13.6 kernel min). dot mode cannot answer P1 at all, so this
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

Nothing was measured. Add --run on the pod, --synthetic to exercise the gates,
or --replay <dir> to re-report a finished run.
