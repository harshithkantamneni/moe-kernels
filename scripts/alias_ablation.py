#!/usr/bin/env python3
"""Measure `alpha` by ABLATION, without the byte model. STUDY.md item 4.

    python scripts/alias_ablation.py                    # plan + cost + MDE, no GPU
    python scripts/alias_ablation.py --run              # the pod run
    python scripts/alias_ablation.py --replay <dir>     # re-report, no GPU
    python scripts/alias_ablation.py --synthetic refit  # exercise the gates, no GPU
    python scripts/alias_ablation.py --run --no-probe --num-warps 8
    python scripts/alias_ablation.py --synthetic alias-blind   # the 2026-09-01 world

WHY THIS EXISTS. `alpha` is the cost of an extra M-tile as a fraction of a fresh
weight read. One expert holding `r` rows is scheduled as `ceil(r/BM)` M-tiles;
the first reads that expert's weight block in full and each later one costs
`alpha` of a fresh read, because L2 absorbs part of the re-read:

    Q(r) = 1 + alpha (ceil(r/BM) - 1),   AI(r) = (2r/b) / Q(r) -> 2 BM / (alpha b)

That bound is why alpha is not a nuisance parameter: at 0.558 it puts BLOCK_M of
16, 32 and 64 BELOW the measured ridge band of 160.3-176.2, and vLLM runs
BLOCK_M=16 through the whole decode range, so a decode-configured MoE kernel can
never reach its compute roof. The whole tile-corrected roofline rests on it.

AND IT RESTS ON ONE REGRESSION AGAINST A BYTE MODEL THAT HAS NO TILE TERM.
`scripts/alpha_refit.py` fits alpha from `implied_traffic_ratio`, which is
time x bandwidth / COMPULSORY BYTES, and C4 (`docs/FINDINGS.md`) is a confirmed
finding that the compulsory-byte ruler was wrong by 1.85% until it was fixed.
A number that decides a paper's headline should not depend on one estimator over
one derived column. This script measures the same physical quantity by a route
that touches neither: no compulsory bytes, no calibrated bandwidth, no ridge, no
fitted intercept, no `implied_traffic_ratio`.

THE METHOD, quoted from `docs/STUDY.md` order-of-work item 4, never done since
August: "alias B by taking the tile offset modulo so every iteration reloads the
same tile (loads execute, L2 hits, no HBM traffic, nothing folds since values
are runtime); acc += tl.sum(b) + tl.sum(a) to keep loads live on the compute
side."

WHY THAT MEASURES ALPHA. Run the same grouped-GEMM access pattern twice at n
M-tiles per expert. NORMAL reads every expert's weight block, once per M-tile,
so its HBM weight traffic is `W (1 + alpha (n-1))` in units of one full pass.
ALIASED points every weight load at ONE resident tile, so its weight loads all
hit L2 and its HBM weight traffic is essentially zero. Everything else --
activation reads, output writes, arithmetic, launch, the grid, the instruction
stream -- is IDENTICAL, so it cancels in the difference:

    D(n) = T_normal(n) - T_aliased(n) = W (1 + alpha (n-1))
    D(1) = W                                       <- one full weight read
    alpha = (D(n)/D(1) - 1) / (n-1)

The n=1 rung supplies the denominator directly, which is what makes this
self-calibrating: W never has to be predicted from bytes and a bandwidth, it is
measured in the same units, in the same session, by the same clock.

AND THAT SUBTRACTION IS ONLY VALID IF L2 AND HBM COSTS ADD, WHICH THEY MAY NOT.
The aliased variant issues exactly the same loads; what it does not do is MISS.
So both variants push `n W` bytes through L2, and whether that common cost
cancels depends on how the two units compose. If they add, it cancels and the
line above is exact. If the kernel instead runs at `max(L2, HBM)`, which is what
a streaming kernel with enough parallelism does, then

    D(n) = W(1 + alpha(n-1)) - r W n,   fitting to   (alpha - r)/(1 - r)

where `r` is the aliased ladder's per-tile cost as a fraction of one weight
read. On an H200, whose L2 bandwidth is only about twice HBM's, `r` is plausibly
0.4 to 0.6 -- and at r = 0.55 a true alpha of 0.558 fits to 0.018. THAT IS THE
SHAPE OF A CONFIDENT WRONG NUMBER, and it lands on top of the retracted 0.10.
This trap is not hypothetical arithmetic; it is what the naive version of this
experiment would have reported.

SO THE ANSWER IS A BRACKET, FROM TWO ESTIMATORS THAT FAIL IN OPPOSITE
DIRECTIONS. The second one fits `T_normal(n)` directly, taking the fixed launch
cost from the ALIASED ladder's own n=0 intercept rather than as a free
parameter, and is exact under max() while biased UP to `(alpha + r)/(1 + r)`
under addition. For any alpha <= 1:

    (alpha - r)/(1 - r)  <=  alpha  <=  (alpha + r)/(1 + r)
      DIFFERENCE estimator          DIRECT estimator

Both are reported, the interval between them is what the report quotes, and `r`
is measured and printed because it is the only thing that sets the width. A run
whose interval is too wide to separate 0.10 from 0.33 from 0.558 says NOT
TESTABLE and names the reason, rather than picking one.

HOW THE ALIASING IS DONE, and why nothing folds. NOT with a constexpr flag and
NOT with a compile-time modulo. The two variants are THE SAME COMPILED KERNEL,
launched with three different runtime integers, each of which is a STRIDE:

    stride_be_eff      N*K  ->  0    every M-tile reads expert 0's block
    stride_bn_blk_eff  BLOCK_N*K -> 0   every N-tile reads column block 0
    b_k_advance        BLOCK_K -> 0    ONLY at --alias-extent tile

In NORMAL those three are the real strides. At `--alias-extent block`, the
default, ALIASED zeroes the first two and keeps the K advance, so every weight
load lands in the SAME `BLOCK_N x K` column block: `BLOCK_N * K * 2` bytes,
resident in any L2 this study has met, spread over thousands of lines, walked
with the normal arm's own sequential stride. At `--alias-extent tile`, which
reproduces the 2026-09-01 run, the third is zeroed too and every load lands on
one 16 KiB tile; that is the pinning whose L2 slice contention drove `r` to 12
and is kept only so the two can be compared in one session. The kernel emits
one add per loop iteration
either way, so the two variants are byte-identical machine code and differ only
in the VALUES of three scalars the compiler cannot see. Dead-code elimination
cannot apply to one and not the other, because there is only one code path. A
modulo would have cost an integer remainder per iteration inside the K loop,
which is real work charged to both sides and is avoidable; the pointer-advance
form costs nothing.

THEY ARE STRIDES AND NOT 1/0 FLAGS FOR A REASON, and it is a trap that would
have silently destroyed the experiment. Triton SPECIALISES integer kernel
arguments: an argument whose value is exactly 1 is compiled in as a constant,
and an argument divisible by 16 gets a divisibility hint. Written the obvious
way -- `b_expert_scale` of 1 for normal and 0 for aliased -- the two variants
would land in DIFFERENT specialisations, the normal one would fold `off_e * 1`
at compile time, and the "same compiled kernel" guarantee this whole design
rests on would be false while every table still printed numbers. Every one of
the three scalars is therefore a large stride in the normal case and 0 in the
aliased case, so both take the divisible-by-16 path and neither takes the
equal-to-1 path. The ISA gate CHECKS that they compiled to one kernel rather
than trusting this paragraph.

THE HAZARD IS NAMED IN THE REPO AND IS NOT NEGOTIABLE. Aliasing a load is
exactly the shape a compiler folds: hoist the invariant load out of the loop and
the measurement reports the optimiser rather than the cache, and it reports it
as a beautiful clean alpha of nearly zero. Three independent checks, all of
which must pass before a number is quoted:

  1. ONE KERNEL. The PTX of the launch used for NORMAL and the PTX of the launch
     used for ALIASED are compared hash for hash, and the global-memory
     instruction counts are printed side by side. They are equal by
     construction, which is a stronger guarantee than two compilations that
     happen to agree, and the gate still checks it rather than asserting it.
  2. THE FOLD IS DEMONSTRATED, not assumed away. A THIRD kernel is compiled with
     the aliasing as `tl.constexpr` -- the naive way to write this experiment --
     and its instruction counts are printed in the same table. If the constexpr
     variant issues fewer global loads than the runtime one, that is the fold
     happening in front of the reader, and it is the evidence that the runtime
     design was necessary. If it does NOT fold, that is reported too.
  3. THE OUTPUT IS CHECKED, both ways. NORMAL must reproduce a torch reference,
     which proves it really read every expert's whole weight block. ALIASED must
     reproduce a DIFFERENT closed-form reference -- `(K/BLOCK_K)` copies of the
     one aliased tile -- which proves the aliasing did exactly what was intended
     and is not a silent no-op that would make D(n) pure noise.

Counting `ld.global` alone would have read ZERO on this kernel. Triton pipelines
global-to-shared copies as `cp.async.cg.shared.global` at num_stages > 1, and on
Hopper can use `cp.async.bulk.tensor`. Every global-memory mnemonic is counted
and reported separately, and the gate is on the total.

THE PREDICTION, stated here so it can fail, and printed before anything runs:

  P1  alpha = 0.558, 90% band 0.529-0.588 (today's refit, 10,813 rows).
      The retracted repo value is 0.10 and TEMPO (arXiv:2608.13057) fits 0.33.
      The report says PASS or FAIL against P1 and names which of the three
      candidate values this ablation supports.

  P2  THE MECHANISM, and it is the more interesting half. At GROUP_SIZE_M=1 the
      reuse distance between M-tile i and M-tile i+1 of one expert is exactly
      one pass over that expert's weight block, so alpha should be a function of
      PER-EXPERT BYTES against L2 and not a universal constant:

          mixtral-8x7b      235.0 MB/expert   >> L2  ->  alpha near 1
          deepseek-v3        58.7 MB/expert   ~= L2  ->  alpha intermediate
          qwen2-57b-a14b     36.7 MB/expert    < L2  ->  alpha small
          deepseek-v2-lite   11.5 MB/expert   << L2  ->  alpha near 0

      All four are run, because those four models ARE the pool the 0.558 was
      fitted over. If P2 holds then 0.558 is a pool average of a step function
      and 0.10, 0.33 and 0.558 can all be right about different pools, which
      would explain a threefold disagreement that has stood since August.

WHAT ELSE DIFFERS BETWEEN ALIASED AND NORMAL, being adversarial about it, since
D(n) charges the whole difference to weight traffic:

  * L2 SERVICE, and the cache-set distribution that makes it worse. Both
    variants push `n W` bytes through L2. MEASURED AND GATED, not argued: `r`
    is the aliased ladder's per-tile cost over one weight read, the bracket
    derivation is only a bracket while `r < 1`, and `bracket_gate` refuses to
    quote an interval built on an `r` above that. The 2026-09-01 run measured
    `r` between 6.4 and 19.1 because its alias pinned every load in the K loop
    to one 16 KiB tile and a handful of L2 slices; `--alias-extent block`
    spreads it over a whole `BLOCK_N x K` column block, which is the same
    sequential walk the normal arm makes and has the closed form that pinning
    was said to lack.
  * L2 CAPACITY. The aliased variant leaves L2 almost entirely to activations
    and outputs. Bounded by design rather than hoped away: at BLOCK_M=16 the
    activation re-stream is `n BLOCK_M / N` of the weight stream, which the plan
    prints per rung and which is under 5% at every shipped rung. The output is
    write traffic and is identical in both variants.
  * TLB AND PAGE BEHAVIOUR. The aliased variant touches one page of B where the
    normal one touches the whole tensor. What bounds it is the L2-RESIDENT
    CONTROL (`--control`): the same ladder on a geometry whose PER-EXPERT block
    fits in L2, so NORMAL has no HBM re-read to save and the tile-count slope
    has nothing legitimate to be. Its alpha must come out consistent with zero,
    and whatever it does come out at is the size of every extra-tile cost that
    is not weight traffic.
  * ANY COST THAT SCALES WITH THE EXTRA-TILE COUNT AND IS NOT A WEIGHT RE-READ
    is absorbed into alpha by construction, because that is the regressor. Said
    plainly because `scripts/group_m_alpha_sweep.py` had to say it too.

WHAT THIS IS NOT. It is not vLLM's `fused_moe_kernel`. It is a kernel with
vLLM's B-pointer arithmetic, vLLM's GROUP_SIZE_M swizzle, vLLM's [E, N, K]
weight layout and vLLM's tile constants, whose reduction is replaced by
`acc += tl.sum(a) + tl.sum(b)` on the study's own instruction. That replacement
is deliberate and it is what makes the estimator unbiased under ADDITION: with
a real `tl.dot` the aliased variant can become compute-bound while the normal
one stays memory-bound, so D(n) loses one copy of the per-tile compute cost and
the fitted alpha is biased DOWN. `--compute dot` runs it that way anyway,
prints the bound and refuses to quote the result as P1's answer. The
accumulator keeps its full BLOCK_M x BLOCK_N float32 shape in both modes, so
register pressure, occupancy and therefore the L2 working set are unchanged
between them.

AND THE BIAS IS BOUNDED WHILE THE 2026-09-01 CEILING IS NOT. That reduction is
a full 64x128 tree per K iteration on the CUDA cores, and the run it produced
sat at 0.61 of the card's read roof on every geometry. `dot` moves the same
work onto the tensor cores at an arithmetic intensity of `BLOCK_M` FLOP/byte,
16 against a ridge of 162.8, so its compute cost is at most a tenth of its
memory cost and is IDENTICAL in both arms. That is why `--probe` is allowed to
consider `dot`: a bounded tenth-of-memory bias in a design that can see DRAM
beats an unbiased design that cannot. The probe prints which mode it chose and
`prediction_gate` still refuses to answer P1 from `dot`.

WHY THE FIRST ATTEMPT WAS VOID, AND WHY THAT WAS THE APPARATUS AND NOT THE
WORLD. This arm ran once, on 2026-09-01, and returned VOID: gates `signal` and
`form` FAILED and the report recorded that the HBM read was 5 to 8% of one
tile's time. Read as a null result that would say the per-tile cost is NOT
DRAM, and the study's whole mechanism sentence would have to drop the word.
It is not a null result. It is a design that could not have seen DRAM whatever
alpha was, and the card's own calibration proves it in one line per model.

D(1) is claimed to be the cost of reading one full pass over every expert's
weight block. `moe/bench/hardware/measured_nvidia_h200.yaml` records this card
moving 4469.6 GB/s on the READ pattern, which its own note calls "the closest
analogue to streaming expert weights". Divide the bytes by that ceiling and
compare with the D(1) the run reported:

    model                  weight bytes   DRAM floor   measured D(1)   ratio
    mixtral-8x7b            1879048192     0.4204 ms      0.0586 ms     7.2x
    qwen2-57b-a14b          2348810240     0.5255 ms      0.0600 ms     8.8x
    deepseek-v2-lite         738197504     0.1652 ms      0.0528 ms     3.1x
    deepseek-v3            15032385536     3.3632 ms      0.3129 ms    10.7x
    control-l2-resident      536870912     0.1201 ms      0.0258 ms     4.7x

Every rung's "one weight read" came out three to eleven times FASTER than the
fastest this card can move those bytes. A weight read cannot beat the card's
own read ceiling, so D(1) was never a weight read, and `alpha = slope/D(1)`
was never a fraction of one. The `signal` gate was right to void the page; its
detail named the symptom and not the cause.

THE CAUSE IS A SHARED CEILING AT 61% OF THE DRAM ROOF. Divide each ladder's
per-tile slope into the weight bytes one extra tile requests:

    model                  normal GB/s   aliased GB/s   aliased / read roof
    mixtral-8x7b               2619.7        2755.4            0.616
    qwen2-57b-a14b             2660.0        2739.7            0.613
    deepseek-v2-lite           2703.4        2735.3            0.612
    deepseek-v3                2666.0        2753.7            0.616
    control-l2-resident        2682.2        2711.1            0.607

The aliased arm, whose loads all hit cache, ran at 0.607 to 0.616 of the DRAM
roof on five geometries spanning 20x in footprint. That is a hard, shape
independent, NON-DRAM ceiling in the request path, and it sits BELOW the DRAM
roof. So in the normal arm DRAM was never the binding resource: it had 39%
slack, ablating it could save only the un-overlapped residue, and the residue
is the 5 to 8% the run reported. The same fact reads off `r`, the aliased
ladder's per-tile cost over one weight read, which came out 6.4 to 19.1 where
the bracket derivation below needs r < 1; at r > 1 the difference estimator's
denominator changes sign and the reported "bracket" is not one.

WHAT THAT CHANGES HERE, and all three are checkable before the box is rented:

  1. THE ALIAS IS SPREAD, NOT PINNED (`--alias-extent`, default `block`). The
     2026-09-01 alias zeroed the K advance too, so every load in the K loop
     landed on ONE BLOCK_K x BLOCK_N tile: 16 KiB, a hundred and twenty eight
     cache lines, a handful of L2 slices. The report named that as the confound
     setting `r` and said a spread alias "has no closed form to check the output
     against". It has one. Keep the K advance REAL and zero only the expert and
     N-block strides, and the aliased arm streams `BLOCK_N * K * 2` bytes, half
     a MiB to under two, L2 resident on any card this study runs on, over
     thousands of lines, with the same sequential K stride the normal arm walks.
     Its closed form is the normal reference evaluated at expert 0, N-block 0.
  2. THE HEADROOM IS MEASURED AND GATED. `headroom` asks whether the aliased
     arm's achieved request bandwidth EXCEEDS the card's own read roof, which is
     the condition under which DRAM binds in the normal arm and the ablation can
     see it at all. `attribution` asks whether D(1) reached the card's floor for
     the bytes the alias removed. Either failing means the page is VOID, and it
     is void for a stated arithmetic reason rather than for a threshold.
  3. THE PROBE IS A TENTH OF THE ARM AND CAN STOP THE OTHER NINE. `--probe`
     times the two cheapest rungs of the smallest model across a short pinning
     grid, prints each pinning's achieved bandwidth against the card's roofs,
     and runs the ladder at the pinning that clears the roof by the widest
     margin. If none clears it, the arm stops there: nothing measured afterwards
     could have answered, and the INVALID costs the probe's own wall minutes
     rather than the whole booking. NO ABSOLUTE MINUTES ARE QUOTED HERE ON
     PURPOSE. `report_cost` is the only place in this file that names a
     duration. Until 2026-09-03 this paragraph asserted one of its own, "three
     minutes before it spends sixty", while that function was printing 5.0
     KERNEL and 11.8 WALL for the shipped design: the fix landed at the print
     and not in the prose, which is the same one-of-two-sites shape as the rest
     of this rebuild, and a driver owner who books from a header reserves an
     hour for a quarter of one. AND THE PRINT DENIED THE PROBE UNTIL THE SAME
     DAY: `probing` was `args.probe and args.run`, and an operator books a pod
     before they have one, so the only page they could read priced the ladder
     WITHOUT the probe the default invocation runs, under-booking the arm by
     the tenth this paragraph is named for. Both halves are now scored against
     the table. The header may QUOTE a retired figure, as the
     sentence you are reading does, and may not assert one;
     `test_the_header_quotes_no_duration_of_its_own` enforces exactly that and
     scores the ratio above against the table `report_cost` prints.
  4. THE MDE IS STATED TWICE, AND THE SECOND ONE IS THE RUN'S. `report_mde` in
     the plan sizes the booking against the corpus's spread, which
     `rescore_published_reports` defines as a WITHIN-cell pstdev and therefore a
     floor, and against `MIN_SIGNAL_FRACTION`, the share `signal_gate` will
     admit. Both are the best available before the rental and neither is what
     the run delivered. `_analyse` prints a second block on this run's own
     pass-to-pass scatter and its own worst signal share -- the number
     `signal_gate` scores, read from the one function `one_tile_signal_shares`
     so the two cannot drift. It is the line that reads 2026-09-01: at D(1) =
     5.3% of a pass that design could resolve alpha only to 0.19, against a 0.11
     limit and a 0.46 gap between the candidates. Its "REFUTED or VOID" was
     therefore not a null result about DRAM, and nothing on the page said so,
     because the MDE was stated once and before the fact.

EXIT CODES ARE `moe/bench/exit_codes`'s TABLE AND NOT A SECOND ONE. This
paragraph carried a different table until 2026-09-03 -- 2 for a usage error,
3 for "cannot run here", 4 for "not testable" -- while the code below already
returned REFUSED(2) on every path that measures nothing and let
`exit_codes.classify` decide the rest. Two tables in one file is the defect
that module exists to prevent, so there is now one:

    0  DONE        measured; every VALIDITY and CLAIM gate PASSED
    1  CLAIM_FAIL  measured; VALIDITY passed; P1 did not. A RESULT, never a retry
    2  REFUSED     nothing measured: no GPU, no triton, a refused preflight, or
                   a bare invocation that printed the plan. Free
    3  INVALID     measured, then a VALIDITY gate failed. Nothing may be quoted
    4  ERROR       crashed. The handler at the foot of this file is what keeps a
                   traceback from leaving 1 behind, which the session ledger
                   LATCHES as a refuted claim and never attempts again
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes, timing  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

# --------------------------------------------------------------------------
# the design
# --------------------------------------------------------------------------

#: M-tiles per expert. Powers of two so rows per expert is an exact multiple of
#: BLOCK_M at every rung and the padding term is EXACTLY zero, which is what
#: leaves the tile count as the only thing that changes between rungs. The
#: ladder is the one STUDY.md asks for; four rungs is also the minimum that can
#: show D(n) is affine in (n-1) rather than merely fitted through two points.
TILE_LADDER = (1, 2, 4, 8)

#: The four models the published alpha was fitted over, so a pooled number here
#: is comparable with a pooled number there. They also span 20x in per-expert
#: bytes, which is what makes P2 testable at all.
DEFAULT_MODELS = ("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v2-lite",
                  "deepseek-v3")

#: What vLLM actually runs through the decode range, and the only tile at which
#: the multi-tile and memory-bound windows overlap generously: multi-tile needs
#: rows per expert above BLOCK_M, memory bound needs rows per expert below the
#: ridge, and at BLOCK_M=16 the whole ladder (16, 32, 64, 128 rows per expert)
#: sits inside 160.3.
DEFAULT_BLOCK_M = 16

#: Everything else about the tile, held fixed so the ONLY thing that varies
#: between the two variants is three runtime integers. GROUP_M=1 is the
#: load-bearing entry and is chosen, not inherited: at 1 the reuse distance
#: between consecutive M-tiles of one expert is exactly one pass over that
#: expert's weight block, which is what makes P2's per-expert-bytes prediction a
#: prediction rather than a hope. It is also the setting the published pool
#: mostly sits at -- `alpha_refit` split by GROUP_SIZE_M reads 0.570 at 1 and
#: 0.488 at 16 -- so 0.570 is the closer comparison and the report says so.
FIXED_TILE = {"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1,
              "num_warps": 8, "num_stages": 3}

#: How the loads are kept live. "sum" is what STUDY.md item 4 prescribes and is
#: the unbiased estimator; "dot" is the real GEMM reduction and is biased low.
COMPUTE_MODES = ("sum", "dot")

#: Timed launches per (rung, variant). Nine because the estimator divides two
#: differences of medians, so the noise on alpha is roughly four times the noise
#: on one timing, and nine samples put a median's standard error at about a
#: third of a single sample's.
DEFAULT_REPLICATES = 9

#: `time_kernel`'s knobs, defaulted here and swept by the operator. All three
#: set the measured milliseconds of every pass, so all three are in the run id:
#: a re-run at a different warmup landing in the same directory would print the
#: old numbers under the new label. The warmup is a DURATION because the tile
#: ladder spans an order of magnitude in per-call time and a call count would
#: warm the top rung a hundred times harder than the bottom one.
DEFAULT_WARMUP_MS = 300.0
DEFAULT_CELL_BUDGET_MS = 50.0
DEFAULT_TRIALS = 3

#: The L2-resident control geometry, and every number in it is chosen.
#:
#: 16.0 MiB PER EXPERT, comfortably inside any L2 this study has run on (H200
#: 60 MiB, A100 40 MiB, both read off `observed.l2_bytes` in
#: `moe/bench/hardware/measured_*.yaml` and checked by a test, not remembered:
#: this comment said "H200 50 MiB" until 2026-09-03 and the H200 has 60), so at
#: GROUP_SIZE_M=1 the re-read between one expert's
#: consecutive M-tiles HITS and NORMAL has no HBM re-read to save. Its alpha
#: must therefore come out near zero, and whatever it does come out at is the
#: size of everything that scales with the tile count and is not weight traffic:
#: TLB and page behaviour, cache-set distribution, and any code-path difference.
#:
#: 512 MiB IN TOTAL, not L2-resident, because the control has to keep a real
#: first-pass W. A control whose whole tensor fitted in L2 would have no W to
#: divide by and no alpha at all.
#:
#: N = 4096 so the activation stream stays at 3% of the weight stream at the top
#: rung, which is the same bound the real models get. A narrower control fails
#: the activation-fraction preflight, which is the gate catching a control that
#: could not have controlled for anything.
CONTROL_MODEL = "control-l2-resident"
CONTROL_EXPERTS = 32
CONTROL_K = 2048
CONTROL_N = 4096


# --------------------------------------------------------------------------
# the card's own ceilings, read from its calibration and never remembered
# --------------------------------------------------------------------------

#: Which STREAM pattern is the roof a weight read is held to. `read`, because
#: `measured_nvidia_h200.yaml` says of it in its own file: "closest analogue to
#: streaming expert weights; reduced along the contiguous axis so the tree does
#: not bound it". It is also the LARGEST of the four patterns on both cards, so
#: every headroom and attribution verdict below is scored against the ceiling
#: most generous to this design. A design that fails against the most generous
#: roof has not failed on a threshold choice.
ROOF_PATTERN = "read"

#: The card this file's PLANTED laws and its off-GPU arithmetic are stated for.
#: `--synthetic` must not read the attached hardware (see `main`), so the plant
#: needs a number, and a number this repository can check.
PLANT_CARD = "NVIDIA H200"


def measured_card(card: str) -> dict:
    """`observed.l2_bytes` and the `read` roof for one card, from its yaml.

    THE FILE, NEVER A LITERAL, and the defect is in this file's own history:
    `SYNTHETIC_L2_BYTES` was `50 * 2**20` and the control's docstring said the
    H200 has 50 MiB of L2. It has 60 (`observed.l2_bytes: 62914560`), the A100
    has 40 (`41943040`), and 50 is neither. A rehearsal planted at 50 MiB
    straddles a cache no card in this study owns: deepseek-v3 holds 56 MiB per
    expert, which is ABOVE a 50 MiB line and BELOW the real one, so the planted
    world classified one of the four models onto the wrong side of the
    threshold the plant exists to test.

    Returns `{}` rather than raising when there is no calibration for the card:
    a missing ruler makes the headroom gates NOT TESTABLE, which is a VALIDITY
    UNKNOWN and therefore INVALID, and that is the right answer. It is not a
    reason to invent a roof.
    """
    import yaml

    from moe.bench import roofline as RF
    path = RF.HARDWARE_DIR / f"{RF.measured_slug(card)}.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    detail = data.get("detail") or {}
    roof = None
    for entry in detail.get("bandwidth_patterns") or []:
        if entry.get("pattern") == ROOF_PATTERN and entry.get("gbps"):
            roof = float(entry["gbps"]) * 1e9
    return {"card": data.get("name", card),
            "l2_bytes": int((data.get("observed") or {}).get("l2_bytes") or 0),
            "roof_bytes_s": roof,
            "roof_pattern": ROOF_PATTERN,
            "source": path.name}


# --------------------------------------------------------------------------
# what the alias actually removes
# --------------------------------------------------------------------------

#: How far the alias reaches, and it is the single lever that decides whether
#: `r` is 12 or is small enough for the bracket to be a bracket.
#:
#:   block  zero the expert and N-block strides, KEEP the K advance. Every
#:          weight load lands in `B[0, 0:BLOCK_N, :]`: `BLOCK_N * K * 2` bytes,
#:          0.5 to 1.75 MiB on the shipped models, resident in any L2 here, over
#:          thousands of lines, walked with the normal arm's own sequential K
#:          stride. THE DEFAULT.
#:   tile   zero the K advance as well, so every load in the K loop lands on one
#:          16 KiB BLOCK_K x BLOCK_N tile. This is the 2026-09-01 design and is
#:          kept only so a session can measure both and show what the pinning
#:          cost. Its L2 slice contention is what drove `r` to 6.4-19.1.
ALIAS_EXTENTS = ("block", "tile")
DEFAULT_ALIAS_EXTENT = "block"

#: The aliased arm's footprint must be at most this fraction of L2, or it is not
#: resident and the arm is not an ablation of DRAM traffic at all. A quarter
#: rather than a whole because the activations and the output share the cache.
ALIAS_RESIDENT_FRACTION = 0.25


# --------------------------------------------------------------------------
# what the answer is being scored against
# --------------------------------------------------------------------------

#: Today's refit and its 90% band, from `docs/FINDINGS.md` "The tile-corrected
#: roofline" and the 2026-09-01 rewrite of C1: 0.558 over 10,813 rows.
REFIT_ALPHA = 0.558
REFIT_BAND = (0.529, 0.588)

#: The two rivals. Cross-checked against `scripts/alpha_refit.py`'s own
#: constants at import time, and by a test, so the three numbers this report
#: scores against cannot drift away from the estimator's.
REPO_RETRACTED_ALPHA = 0.10
TEMPO_ALPHA = 0.33

#: The GROUP_SIZE_M=1 split of the published refit. Printed beside the pooled
#: 0.558 because this ablation runs at GROUP_M=1 and the pooled figure does not.
REFIT_ALPHA_AT_GROUP_M_1 = 0.570

CANDIDATES = (
    ("this repo, retracted", REPO_RETRACTED_ALPHA),
    ("TEMPO arXiv:2608.13057", TEMPO_ALPHA),
    ("today's refit", REFIT_ALPHA),
)


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

#: The placebo. Two launches of the IDENTICAL configuration must differ by less
#: than this fraction of D(1), or the timing noise floor is a large enough part
#: of the signal that no alpha read off it means anything.
PLACEBO_MAX_FRACTION = 0.10

#: D(1) must be at least this fraction of T_normal(1). Below it the weight read
#: is not what the kernel is doing, and the difference is measuring something
#: else with a weight-read label on it.
MIN_SIGNAL_FRACTION = 0.25

# --------------------------------------------------------------------------
# the headroom, which is what the 2026-09-01 run did not have
# --------------------------------------------------------------------------

#: The aliased ladder's achieved request bandwidth, over the card's read roof.
#: Below 1.0 the shared non-DRAM path is SLOWER than DRAM, DRAM has slack in
#: the normal arm, and no ablation of DRAM can move the clock however large
#: alpha is. The 2026-09-01 run measured 0.607 to 0.616 on five geometries.
#:
#: IT IS DERIVED FROM `MIN_SIGNAL_FRACTION` AND NOT CHOSEN BESIDE IT, because
#: two thresholds over the same physical quantity is this repository's recurring
#: defect written as arithmetic. Let `h` be this ratio. When DRAM binds in the
#: normal arm and the shared path binds in the aliased one, the normal arm's
#: time is the DRAM time and the aliased arm's is that time over `h`, so
#:
#:     D(1) / T_normal(1) = 1 - 1/h
#:
#: `signal` will not admit a run whose left side is below `MIN_SIGNAL_FRACTION`.
#: Inverting gives the smallest `h` that can produce an admissible run, and a
#: headroom limit ANY LOWER would pass designs `signal` then voids while a
#: reader hunts for the reason in alpha. At 0.25 it is 1.333. Written as the
#: inversion so the two move together for ever.
MIN_HEADROOM_RATIO = 1.0 / (1.0 - MIN_SIGNAL_FRACTION)

#: D(1) over the card's own floor for the bytes the alias removes. The ablation
#: claims D(1) IS that read; a D(1) below the floor is a read that beat the
#: card, which is impossible, so the label is wrong.
#:
#: THE SAME NUMBER AS `MIN_SIGNAL_FRACTION`, and for the same reason as above.
#: `T_normal(1)` is at least the floor, always, so
#:
#:     D(1)/floor  >=  D(1)/T_normal(1)  =  1 - 1/h
#:
#: and this gate is the WEAKER of the pair whenever the normal arm runs below
#: the roof, which every real kernel does. It is kept beside `signal` and not
#: folded into it because it is stated in the CARD's units rather than the
#: kernel's own time: `signal` says a difference is a small part of a pass and
#: cannot say why, this says a difference is smaller than physics allows for the
#: bytes it names. The 2026-09-01 run read 0.053 on the first and 0.093 on the
#: second, so both failed and only the second one said what was wrong.
MIN_ATTRIBUTION_RATIO = MIN_SIGNAL_FRACTION

#: `r` above which the two estimators stop bracketing anything. The DIFFERENCE
#: estimator is `(alpha - r)/(1 - r)`; at `r > 1` that denominator is negative,
#: the estimator exceeds the truth instead of falling short of it, and the
#: interval printed between the two is not an interval containing alpha. 0.9
#: rather than 1.0 because at `r` just under one the bracket is arbitrarily wide
#: and `band_gate` would call it NOT TESTABLE anyway.
MAX_BRACKET_R = 0.9

#: D(n) must be affine in (n-1) to at least this R^2, or `W(1 + alpha(n-1))` is
#: the wrong functional form and its slope-over-intercept is not alpha.
MIN_LINEARITY_R2 = 0.97

#: A reported interval wider than this cannot separate the three candidate
#: values, which span 0.10 to 0.558. Half the widest gap between adjacent
#: candidates (0.228) is the loosest interval that could still put exactly one
#: of them inside. It is a NOT-TESTABLE threshold and not a failure: an
#: unresolved measurement is not a refutation.
MAX_BAND_WIDTH = 0.11

#: How far the L2-resident control's BRACKET may sit from zero. Its weight
#: re-reads hit L2, so there is no HBM re-read for the aliasing to remove and
#: its alpha must be consistent with zero. Anything beyond this is the size of
#: every extra-tile-scaling difference that is NOT weight traffic, and it is
#: charged against the real alphas.
CONTROL_MAX_ALPHA = 0.15

#: Relative RMS error allowed between a variant's output and its closed-form
#: reference. Both sides sum the same bf16 values in a different order, so the
#: floor is summation order, not correctness; 1e-3 is three orders above it.
CORRECTNESS_RTOL = 1e-3

#: Activation re-stream as a fraction of the weight stream, above which the L2
#: capacity confound stops being negligible. n*BLOCK_M/N at the shipped rungs is
#: under 0.01 everywhere, so this is a tripwire for someone changing the shape.
MAX_ACTIVATION_FRACTION = 0.05

#: Clock drift across a rung, above which the rung is flagged. Matches
#: `moe/bench/timing.clock_drift`'s own threshold.
CLOCK_DRIFT_LIMIT = 0.05


# --------------------------------------------------------------------------
# the rungs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Rung:
    """One (model, tile count) cell, with everything the report needs.

    `rows_per_expert` is an EXACT multiple of BLOCK_M by construction: the
    M-tile-to-expert map is built directly rather than sampled from a routing
    distribution, so padding is exactly zero and the tile count is the only
    thing that moves across the ladder. `moe/bench/published.py`'s rows cannot
    do that -- uniform routing is SAMPLED per replicate and its tile count
    varies within a cell, which is the correction FINDINGS.md had to make to the
    staircase table on 2026-09-01.
    """

    model: str
    tiles: int
    block_m: int
    experts: int
    k: int
    n: int
    block_n: int
    block_k: int
    group_m: int
    control: bool = False

    @property
    def key(self) -> str:
        return f"{self.model}|t{self.tiles}|bm{self.block_m}"

    @property
    def rows_per_expert(self) -> int:
        return self.tiles * self.block_m

    @property
    def total_rows(self) -> int:
        return self.experts * self.rows_per_expert

    @property
    def per_expert_bytes(self) -> int:
        """One expert's weight block, in bytes. The reuse distance at GROUP_M=1
        and therefore the quantity P2 says alpha is a function of."""
        return self.n * self.k * 2

    @property
    def weight_bytes(self) -> int:
        return self.experts * self.per_expert_bytes

    @property
    def activation_bytes(self) -> int:
        return self.total_rows * self.k * 2

    @property
    def output_bytes(self) -> int:
        """float32, so the correctness check is not fighting bf16 rounding."""
        return self.total_rows * self.n * 4

    @property
    def footprint_bytes(self) -> int:
        return self.weight_bytes + self.activation_bytes + self.output_bytes

    @property
    def activation_fraction(self) -> float:
        """The L2-capacity confound's bound: activation stream over weight
        stream. Equals n*BLOCK_M/N once the E and K terms cancel."""
        return self.activation_bytes / self.weight_bytes

    @property
    def arith_intensity(self) -> float:
        """Compulsory FLOP/byte in `dot` mode, which is `2 r / b` = rows per
        expert in bf16. Meaningless in `sum` mode, where there is no matmul and
        the kernel is memory bound by construction; reported anyway because the
        `dot` mode gate reads it."""
        return float(self.rows_per_expert)

    @property
    def num_pid_m(self) -> int:
        return self.experts * self.tiles

    @property
    def num_pid_n(self) -> int:
        return self.n // self.block_n

    @property
    def programs(self) -> int:
        return self.num_pid_m * self.num_pid_n

    @property
    def k_iters(self) -> int:
        return self.k // self.block_k

    def alias_bytes(self, extent: str) -> int:
        """Bytes of B the ALIASED arm touches. This is what must fit in L2.

        `block` keeps the K advance, so every program walks ONE `BLOCK_N x K`
        column block of expert 0: `BLOCK_N * K * 2` bytes, and the walk is the
        same sequential stride the normal arm makes. `tile` zeroes the advance
        as well, so the whole K loop re-reads one `BLOCK_K x BLOCK_N` tile: 16
        KiB at the shipped constants. Both are resident; only the second is a
        hot spot, and the hot spot is what the 2026-09-01 report named as the
        thing setting `r` and then declined to fix.
        """
        if extent not in ALIAS_EXTENTS:
            raise ValueError(f"unknown alias extent {extent!r}; "
                             f"expected one of {ALIAS_EXTENTS}")
        if extent == "tile":
            return self.block_k * self.block_n * 2
        return self.block_n * self.k * 2


def rung_for(model: str, tiles: int, block_m: int, tile: dict) -> Rung:
    """A rung from a real model geometry, shaped as vLLM's up-projection.

    N is `2 * intermediate_size` and K is `hidden_size`, which is w1's shape in
    vLLM's `fused_experts`: the gate and up halves are one tensor. Using the
    real w1 rather than a square stand-in matters because P2 is a claim about
    PER-EXPERT BYTES against L2, and a stand-in would be a claim about a shape
    no model has.
    """
    if model == CONTROL_MODEL:
        experts, k, n = CONTROL_EXPERTS, CONTROL_K, CONTROL_N
    else:
        cfg = MODEL_CONFIGS[model]
        experts, k, n = cfg.num_experts, cfg.hidden_size, 2 * cfg.intermediate_size
    return Rung(model=model, tiles=tiles, block_m=block_m, experts=experts,
                k=k, n=n, block_n=tile["BLOCK_N"], block_k=tile["BLOCK_K"],
                group_m=tile["GROUP_M"], control=(model == CONTROL_MODEL))


@dataclass(frozen=True)
class Design:
    models: tuple[str, ...]
    tiles: tuple[int, ...]
    block_m: int
    tile: dict
    compute: str
    replicates: int
    rungs: tuple[Rung, ...]
    alias_extent: str = DEFAULT_ALIAS_EXTENT

    @property
    def fingerprint(self) -> str:
        payload = json.dumps({
            "models": list(self.models), "tiles": list(self.tiles),
            "block_m": self.block_m, "tile": self.tile,
            "compute": self.compute, "replicates": self.replicates,
            # THE EXTENT IS THE EXPERIMENT. `block` and `tile` remove different
            # bytes and produce different `r`; sharing a directory would let a
            # resumed run mix the two ladders under one alpha.
            "alias_extent": self.alias_extent,
            # THE GEOMETRIES, not just the model names. `--replay` and the
            # resume path key on `rung.key`, which is model|tiles|block_m, so a
            # change to the control constants would silently reuse records
            # measured on a different shape under the same key. The fingerprint
            # is what sends a changed design to a different directory.
            "shapes": [[r.model, r.experts, r.k, r.n] for r in self.rungs],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:10]

    def rungs_for(self, model: str) -> list[Rung]:
        return [r for r in self.rungs if r.model == model]


def build_design(args) -> Design:
    models = tuple(args.models)
    if args.control:
        models = models + (CONTROL_MODEL,)
    tile = dict(FIXED_TILE)
    tile["GROUP_M"] = args.group_m
    tile["BLOCK_N"] = args.block_n
    tile["BLOCK_K"] = args.block_k
    # THE PINNING IS PART OF THE DESIGN AND NOT A LAUNCH DETAIL. It sets the
    # achieved request bandwidth, which is the one thing deciding whether this
    # arm can see DRAM at all, so it belongs in the fingerprint and therefore in
    # the run id: two pinnings are two experiments and must not share a
    # directory. `--probe` changes both of these and rebuilds the design.
    tile["num_warps"] = args.num_warps
    tile["num_stages"] = args.num_stages
    rungs = tuple(rung_for(m, t, args.block_m, tile)
                  for m in models for t in args.tiles)
    return Design(models=models, tiles=tuple(args.tiles), block_m=args.block_m,
                  tile=tile, compute=args.compute, replicates=args.replicates,
                  rungs=rungs, alias_extent=args.alias_extent)


#: The design knobs EVERY CELL carries, and the one place that decides which
#: they are.
#:
#: THERE ARE TWO WAYS INTO A REPORT AND ONLY ONE OF THEM IS A FILE. `--replay`
#: adopts a knob out of plan.json, which is wall one; `_analyse` asks the CELLS,
#: which is wall two and the one that still stands when the plan is old,
#: hand-edited, absent, or the directory was named with `--out` and replayed
#: from somewhere else. `compute` got both walls on 2026-09-03. `alias_extent`
#: got only the first, because NEITHER cell writer recorded it, so the second
#: wall had nothing to read: two extents pooled into one directory would have
#: been scored as one ladder with no line printed anywhere saying so. That is
#: this rebuild's recurring defect for the eighth time, and it is why the list
#: is a constant read by both writers and by the wall rather than three literals
#: that a third writer could disagree with.
CELL_KNOBS = ("compute", "alias_extent")

#: Per knob: the heading a scored-from-the-cells page prints, the noun a
#: refusal names, why one page cannot hold two values, and what the knob
#: decides. The prose lives BESIDE the knob and not at the wall, so a knob added
#: to `CELL_KNOBS` cannot reach the wall without a stated reason to refuse.
CELL_KNOB_STAKES = {
    "compute": {
        "heading": "MODE",
        "noun": "compute mode",
        "mixed": "one page cannot score an unbiased estimator and a lower "
                 "bound as one ladder",
        "decides": "prediction_gate answers P1 from sum and refuses to answer "
                   "it from dot",
    },
    "alias_extent": {
        "heading": "ALIAS EXTENT",
        "noun": "alias extent",
        "mixed": "one page cannot pool a ladder whose aliased arm streamed a "
                 "whole BLOCK_N x K column block with one that hammered a "
                 "single BLOCK_K x BLOCK_N tile",
        "decides": "the extent sets how much of the shared path the aliased "
                   "arm actually exercises, and the pinned 16 KiB variant is "
                   "what cost the 2026-09-01 run its experiment",
    },
}


def cell_knobs(design: Design) -> dict[str, str]:
    """`CELL_KNOBS` as a record fragment, for every writer of a cell.

    Both cell writers -- `measure_rung` on the card and `synthesise` off it --
    splat this rather than listing keys, so a knob added to `CELL_KNOBS` reaches
    the measured path, the planted path and the wall in `_analyse` together or
    not at all. A writer that listed its own keys is exactly how the synthetic
    path came to omit `compute` while the measured one carried it, and every
    planted world replayed as sum mode however it was planted.
    """
    return {knob: getattr(design, knob) for knob in CELL_KNOBS}


# --------------------------------------------------------------------------
# gates as a value, not as an exception
# --------------------------------------------------------------------------

@dataclass
class Gate:
    """One numeric verdict. `ok=None` means the data cannot answer it.

    The bool coercion is not cosmetic and the bug it fixes was live in
    `scripts/group_m_alpha_sweep.py` for one run: a numpy bool is not `False`,
    so `g.ok is False` skipped a gate that had printed FAIL, and the verdict
    disagreed with the table above it. Anything derived from a fitted number can
    arrive here as a numpy scalar.
    """

    name: str
    ok: bool | None
    detail: str
    kind: str = ""

    def __post_init__(self) -> None:
        self.ok = None if self.ok is None else bool(self.ok)
        if not self.kind:
            self.kind = self._kind_from_name()
        if self.kind not in exit_codes.KINDS:
            raise ValueError(
                f"gate kind {self.kind!r} is not one of {exit_codes.KINDS}")

    @property
    def label(self) -> str:
        return {True: "PASS", False: "FAIL", None: "NOT TESTABLE"}[self.ok]

    @property
    def token(self) -> str:
        """The gate's name as ONE whitespace-free token, for `RESULT:`.

        THE WHOLE NAME IS SLUGGED, not just its first word, and the reason is in
        THIS script's own gate list. Three of the six preflight names begin with
        the word `the` -- `the ladder has at least three rungs ...`, `the ladder
        starts at one tile ...`, `the activation stream is small ...` -- so a
        first-word token would emit three gates called `the` and a driver keying
        on the name field would see one of them and silently lose the other two.
        That is the shape of failure `exit_codes` exists to stop rather than to
        reproduce. (This paragraph cited `regime: every cell is memory bound`
        until 2026-09-02; those are `group_m_alpha_sweep`'s gates, copied in with
        the code. The slug is still right here; the example was not.)
        """
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.name).strip("-")
        return slug[:56] or "gate"

    def _kind_from_name(self) -> str:
        """CLAIM for the pre-registered predictions, VALIDITY for the rest.

        `P1: the ablation agrees with the refit` is the only statement about the
        WORLD this script makes, and a FAIL on it is a RESULT: the ablation
        would have refuted the refit, which is the most valuable outcome the
        experiment has. Everything else -- correctness, placebo, signal, form,
        control, resolution, the ISA census and the preflights -- says whether
        the apparatus was sound, and a FAIL there means nothing on the page may
        be quoted at all.

        The rule is `P<digit>` on the first word, the same one
        `group_m_alpha_sweep` uses, so the two scripts cannot classify a gate
        differently. The trailing colon is stripped because this file writes
        `P1:` and that one writes `P1 `.

        A FALLBACK, NOT THE ANSWER, and the preflight is why. `P2 is testable:
        the models straddle L2` asks whether this DESIGN can address prediction
        2 at all; it mentions P2 because that is what it is a precondition for,
        and the rule read the mention and emitted `RESULT: CLAIM
        P2-is-testable-...`. A design that cannot ask the question was therefore
        reported to the driver as a REFUTED CLAIM -- a statement about the world
        -- when what happened is that the apparatus was unsound, which is
        INVALID. Every gate `preflight` builds now passes `kind` explicitly, and
        `__post_init__` only falls back here for the result gates, whose names
        this file controls one line above where they are scored.
        """
        first = self.name.split()[0].rstrip(":")
        return (exit_codes.CLAIM
                if first[:1] == "P" and first[1:].isdigit()
                else exit_codes.VALIDITY)

    @property
    def verdict(self) -> str:
        return {True: exit_codes.PASS, False: exit_codes.FAIL,
                None: exit_codes.UNKNOWN}[self.ok]

    def result_line(self) -> str:
        """The ONE line the session driver may grep for this gate.

        Rendered by `moe.bench.exit_codes.result_line`, so the prefix, the field
        order and the refusals are identical in every script. The `[PASS] name`
        block beside it is prose: a line that merely contains "PASS" is not a
        result, which is what the driver's old free-text grep was reading
        pre-registered expectations out of.
        """
        return exit_codes.result_line(self.kind, self.token, self.verdict,
                                      self.detail.replace("\n", " ")[:160])

    def scored(self) -> tuple[str, str, str]:
        return (self.kind, self.token, self.verdict)


def preflight(design: Design, l2_bytes: int) -> list[Gate]:
    """Refuse a design that cannot answer the question, before it is paid for.

    EVERY GATE HERE IS VALIDITY, PASSED EXPLICITLY, and the one that made it
    necessary is `P2 is testable: the models straddle L2`. Its name mentions the
    prediction it is a precondition FOR, `Gate._kind_from_name` read the mention,
    and the gate came out CLAIM: a design that cannot ask the question announced
    itself to the driver as a refuted claim about the world. None of these six
    is a claim about the world. They say whether the apparatus can answer one.
    """
    def _validity(name: str, ok, detail: str) -> Gate:
        return Gate(name, ok, detail, kind=exit_codes.VALIDITY)

    gates: list[Gate] = []

    bad_shape = [r.key for r in design.rungs
                 if r.n % r.block_n or r.k % r.block_k
                 or r.num_pid_m % r.group_m]
    gates.append(_validity(
        "every rung divides exactly, so no mask and no padding",
        not bad_shape,
        "N % BLOCK_N, K % BLOCK_K and M-tiles % GROUP_M are all zero"
        if not bad_shape else
        f"{len(bad_shape)} rungs do not divide: {bad_shape[:3]}. A masked tile "
        "loads a partial block and the two variants would stop being the same "
        "amount of work."))

    gates.append(_validity(
        "the ladder has at least three rungs, so D(n) can be shown affine",
        len(design.tiles) >= 3,
        f"tile ladder {list(design.tiles)}; two rungs fit a line through two "
        "points and can never contradict the form"))

    gates.append(_validity(
        "the ladder starts at one tile, which is the only source of W",
        min(design.tiles) == 1,
        f"smallest rung is {min(design.tiles)} tiles. D(1) is the denominator "
        "of every alpha here; without it W has to come from a byte model, "
        "which is the thing this experiment exists to avoid"))

    worst = max((r.activation_fraction for r in design.rungs), default=0.0)
    gates.append(_validity(
        "the activation stream is small against the weight stream",
        worst <= MAX_ACTIVATION_FRACTION,
        f"worst rung streams {worst * 100:.2f}% as many activation bytes as "
        f"weight bytes (limit {MAX_ACTIVATION_FRACTION * 100:.0f}%). This "
        "bounds the L2-capacity confound: the aliased variant frees L2, and "
        "what it could free it for is the activation re-stream"))

    if design.compute == "dot":
        over = [r.key for r in design.rungs
                if not r.control and r.arith_intensity >= 160.3]
        gates.append(_validity(
            "in dot mode every rung stays below the ridge band",
            not over,
            "all rungs below 160.3 FLOP/byte" if not over else
            f"{len(over)} rungs are compute bound: {over[:3]}. A compute-bound "
            "rung pays for extra tiles in padded arithmetic, not in traffic"))

    if l2_bytes:
        widest = max((r.alias_bytes(design.alias_extent) for r in design.rungs),
                     default=0)
        limit = ALIAS_RESIDENT_FRACTION * l2_bytes
        gates.append(_validity(
            "the aliased arm is L2-resident, so it removes traffic and not work",
            widest <= limit,
            f"extent {design.alias_extent!r}: the widest aliased footprint is "
            f"{widest / 2**20:.2f} MiB against "
            f"{ALIAS_RESIDENT_FRACTION:.0%} of an L2 of "
            f"{l2_bytes / 2**20:.1f} MiB ({limit / 2**20:.2f} MiB). An aliased "
            "arm that misses is not an ablation of the weight read, it is a "
            "second copy of it, and D would be noise around zero"))

    if l2_bytes:
        spread = sorted({r.per_expert_bytes for r in design.rungs
                         if not r.control})
        both = spread and spread[0] < l2_bytes < spread[-1]
        gates.append(_validity(
            "P2 is testable: the models straddle L2",
            bool(both),
            f"per-expert weight blocks run {spread[0] / 2**20:.1f} to "
            f"{spread[-1] / 2**20:.1f} MiB against an L2 of "
            f"{l2_bytes / 2**20:.1f} MiB"
            if spread else "no non-control rungs"))
    return gates


# --------------------------------------------------------------------------
# the estimator. Deliberately NOT alpha_refit's.
# --------------------------------------------------------------------------

@dataclass
class AlphaFit:
    """alpha from the ablation, as a BRACKET, and everything that qualifies it.

    TWO ESTIMATORS, BECAUSE ONE OF THEM IS ALWAYS WRONG. The aliased variant
    issues exactly the same loads as the normal one; what it does not do is miss
    L2. So both variants pay the L2 service cost of `n W` bytes, and whether
    that common cost CANCELS in the difference depends on how the two units
    compose:

        if L2 and HBM service ADD          D(n) = W(1 + alpha(n-1)) exactly,
                                           and the DIFFERENCE estimator is right
        if the kernel runs at max(L2, HBM) D(n) = W(1+alpha(n-1)) - r W n,
                                           and the difference estimator returns
                                           (alpha - r)/(1 - r), biased DOWN

    where `r` is the aliased ladder's per-tile cost as a fraction of one weight
    read. A streaming kernel is closer to max() than to a sum, and r on an H200
    is plausibly 0.4 to 0.6, at which the difference estimator returns about
    0.02 for a true alpha of 0.558. THAT IS THE SHAPE OF A CONFIDENT WRONG
    NUMBER, and it would land on top of the retracted 0.10.

    The DIRECT estimator is the other extreme: fit `T_normal(n)` itself, with
    the fixed launch cost taken from the aliased ladder's own n=0 intercept. It
    is exact under max() and biased UP to (alpha + r)/(1 + r) under addition.

    For any alpha <= 1 the two bracket the truth:

        (alpha - r)/(1 - r)  <=  alpha  <=  (alpha + r)/(1 + r)

    so this dataclass reports both and the report quotes the interval. The
    bracket is narrow exactly when the aliased ladder is cheap, which is
    MEASURED rather than assumed, and a run whose bracket is too wide to
    separate the candidates says NOT TESTABLE rather than picking one.

    `alpha` remains the difference estimator, so it is always the LOW end.
    """

    alpha: float | None
    w_ms: float | None
    alpha_direct: float | None = None
    fixed_ms: float | None = None
    aliased_slope: float | None = None
    per_rung: dict[int, float] = field(default_factory=dict)
    r2: float | None = None
    diffs: dict[int, float] = field(default_factory=dict)
    why: str = ""

    @property
    def ok(self) -> bool:
        return self.alpha is not None

    @property
    def bracket(self) -> tuple[float, float] | None:
        if self.alpha is None:
            return None
        if self.alpha_direct is None:
            return (self.alpha, self.alpha)
        return (min(self.alpha, self.alpha_direct),
                max(self.alpha, self.alpha_direct))

    @property
    def width(self) -> float:
        span = self.bracket
        return 0.0 if span is None else span[1] - span[0]

    @property
    def l2_share(self) -> float | None:
        """The aliased ladder's per-tile cost over one weight read. This is `r`,
        and it is what sets the bracket's width."""
        if self.aliased_slope is None or not self.w_ms:
            return None
        return self.aliased_slope / self.w_ms


def _fit_line(points: list[tuple[int, float]]) -> tuple[float, float, float] | None:
    """Least squares of y on x = tiles - 1. Returns (intercept, slope, R^2).

    x is `tiles - 1` and not `tiles` so that the intercept IS the one-tile
    value, which is the quantity every estimator here divides by.
    """
    if len(points) < 2:
        return None
    xs = [float(n - 1) for n, _ in points]
    ys = [float(v) for _, v in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2
                 for x, y in zip(xs, ys, strict=True))
    return intercept, slope, (1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)


def fit_alpha(diffs: dict[int, float]) -> AlphaFit:
    """The DIFFERENCE estimator: alpha from `D(n) = W (1 + alpha (n-1))`.

    Slope over intercept, and both come out of the same fit, so the units of
    time cancel: no bandwidth, no byte count and no calibration enters. That is
    the entire point of the ablation route.

    Returns an unfitted result rather than raising when the intercept is not
    positive. A non-positive W means the aliased variant was not faster at one
    tile, so there is no weight read to be a fraction OF, and a ratio computed
    through it would be a large confident number with no meaning.
    """
    points = sorted((n, d) for n, d in diffs.items() if d == d)
    line = _fit_line(points)
    if line is None:
        return AlphaFit(None, None,
                        why="fewer than two rungs, or one tile count only")
    intercept, slope, r2 = line
    if intercept <= 0:
        return AlphaFit(
            None, None, diffs=dict(points),
            why=f"fitted W is {intercept:.4f} ms, not positive: the aliased "
                "variant was not faster, so there is no weight read to divide by")
    base = diffs.get(1)
    per_rung = {}
    if base and base > 0:
        per_rung = {n: (d / base - 1.0) / (n - 1) for n, d in points if n > 1}
    return AlphaFit(alpha=slope / intercept, w_ms=intercept, per_rung=per_rung,
                    r2=r2, diffs=dict(points))


def fit_bracket(samples: dict[int, dict[str, list[float]]]) -> AlphaFit:
    """Both estimators, so the answer is an interval that contains the truth.

    The fixed cost is the ALIASED ladder's n=0 intercept, `I_a - S_a`, which is
    launch and dispatch and nothing else: the aliased variant's own per-tile
    cost is its slope, and extrapolating it out leaves what a zero-tile launch
    would have cost. Taking it from the aliased ladder rather than fitting it as
    a free parameter of the normal one is what keeps the direct estimator from
    being "a fitted intercept", which is the thing this whole experiment exists
    to avoid.
    """
    fit = fit_alpha(differences(samples))
    if not fit.ok:
        return fit
    aliased = sorted((n, statistics.median(v["aliased"]))
                     for n, v in samples.items() if v.get("aliased"))
    normal = sorted((n, statistics.median(v["normal"]))
                    for n, v in samples.items() if v.get("normal"))
    line_a, line_n = _fit_line(aliased), _fit_line(normal)
    if line_a is None or line_n is None:
        return fit
    fixed = line_a[0] - line_a[1]
    w_direct = line_n[0] - fixed
    fit.fixed_ms = fixed
    fit.aliased_slope = line_a[1]
    if w_direct > 0:
        fit.alpha_direct = line_n[1] / w_direct
    return fit


def differences(samples: dict[int, dict[str, list[float]]]) -> dict[int, float]:
    """D(n) = median(normal) - median(aliased), per rung."""
    out: dict[int, float] = {}
    for tiles, by_variant in samples.items():
        normal = by_variant.get("normal") or []
        aliased = by_variant.get("aliased") or []
        if normal and aliased:
            out[tiles] = statistics.median(normal) - statistics.median(aliased)
    return out


def bootstrap_alpha(samples: dict[int, dict[str, list[float]]], draws: int,
                    seed: int) -> tuple[float, float] | None:
    """90% interval on the alpha BRACKET, by resampling the timings.

    Resampling replicates WITHIN each (rung, variant) is the right unit: the
    rungs are a fixed designed ladder and are not a sample of anything, while
    the replicates are repeated draws of the same quantity and are exactly what
    the band is supposed to describe.
    """
    rng = random.Random(seed)
    lows: list[float] = []
    highs: list[float] = []
    for _ in range(draws):
        fit = fit_bracket(_resample(samples, rng))
        span = fit.bracket
        if span is not None:
            lows.append(span[0])
            highs.append(span[1])
    if len(lows) < max(20, draws // 10):
        return None
    lows.sort()
    highs.sort()
    # The 5th of the LOW end and the 95th of the HIGH end, so the reported
    # interval carries both the model ambiguity and the timing noise. Taking
    # percentiles of a single number would have quietly dropped the first.
    return (_percentile(lows, 5.0), _percentile(highs, 95.0))


def _resample(samples: dict[int, dict[str, list[float]]], rng):
    return {tiles: {name: [rng.choice(v) for _ in v] if v else []
                    for name, v in by_variant.items()}
            for tiles, by_variant in samples.items()}


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = (len(sorted_values) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def pooled_band(per_model: dict[str, dict[int, dict[str, list[float]]]],
                draws: int, seed: int) -> tuple[float, float] | None:
    """An interval on the MEDIAN of the per-model alpha brackets.

    The median rather than the mean because P2 says alpha is a step function of
    per-expert bytes against L2, and a mean of a step function over four models
    is a number about this particular four models. The median is quoted as the
    pool statistic and every per-model alpha is printed beside it, so a reader
    who cares about a specific geometry is never made to read the pool.
    """
    rng = random.Random(seed)
    lows: list[float] = []
    highs: list[float] = []
    for _ in range(draws):
        low, high = [], []
        for samples in per_model.values():
            span = fit_bracket(_resample(samples, rng)).bracket
            if span is not None:
                low.append(span[0])
                high.append(span[1])
        if low:
            lows.append(statistics.median(low))
            highs.append(statistics.median(high))
    if len(lows) < max(20, draws // 10):
        return None
    lows.sort()
    highs.sort()
    return (_percentile(lows, 5.0), _percentile(highs, 95.0))


def supported_candidate(band: tuple[float, float]) -> tuple[list[str], str]:
    """Which of the three published values this band contains, and the verdict.

    Returns the names inside the band and a sentence naming the nearest one when
    the band contains none, because "supports none of them" is the answer that
    matters most and is the easiest to leave unsaid.
    """
    inside = [name for name, value in CANDIDATES if band[0] <= value <= band[1]]
    if inside:
        return inside, (
            f"the interval {band[0]:.3f}-{band[1]:.3f} contains "
            + " and ".join(inside))
    mid = 0.5 * (band[0] + band[1])
    name, value = min(CANDIDATES, key=lambda c: abs(c[1] - mid))
    width = max(band[1] - band[0], 1e-9)
    return [], (
        f"the interval {band[0]:.3f}-{band[1]:.3f} contains NONE of the "
        "three. "
        f"Nearest is {name} at {value:.3f}, "
        f"{abs(value - mid) / width:.1f} interval-widths away")


# --------------------------------------------------------------------------
# the kernel, built lazily so this file imports on a laptop
# --------------------------------------------------------------------------

class CannotRunHere(RuntimeError):
    """No GPU, no triton, or not enough memory. Named so `main` can exit 3."""


def build_kernel():
    """The ablation kernel. Imports triton, so it is never called off-GPU.

    THE SHAPE IS vLLM's, not a convenience. `b_ptrs` is
    `b + e*stride_be + offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn`
    over a [E, N, K] weight tensor with K contiguous, advanced by
    `BLOCK_K * stride_bk` per iteration, under the same
    `GROUP_SIZE_M` swizzle -- which is `fused_moe_kernel` line for line on the
    B side. The A side and the reduction are the parts that differ, and the
    docstring at the top of this file says why.

    THE THREE RUNTIME SCALARS ARE THE ABLATION. `stride_be_eff`,
    `stride_bn_blk_eff` and `b_k_advance` are ordinary int arguments, so the
    compiler sees three unknown values and cannot fold, hoist or specialise on
    any of them. NORMAL passes the real strides; ALIASED passes (0, 0, 0), which
    pins every weight load to `B[0, 0:BLOCK_N, 0:BLOCK_K]`. Same kernel, same
    instructions, same counts. They are strides rather than 1/0 flags because
    Triton compiles an integer argument of exactly 1 in as a constant, which
    would have put the two variants in different specialisations; the top of
    this file says what that would have cost.

    `CONSTEXPR_ALIAS` exists ONLY to demonstrate the hazard. It is the naive way
    to write this experiment and it is compiled, counted and reported so that a
    reader can see whether the fold the repo warned about actually happens.
    """
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - needs the box
        raise CannotRunHere(
            "triton is not importable in this interpreter. Run inside the vllm "
            "venv on the pod: /workspace/venvs/vllm/bin/python "
            "scripts/alias_ablation.py --run") from exc

    @triton.jit
    def _ablation_kernel(
        a_ptr, b_ptr, c_ptr, expert_of_tile_ptr,
        EM, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_be_eff, stride_bn_blk_eff, b_k_advance,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr, COMPUTE_DOT: tl.constexpr,
        CONSTEXPR_ALIAS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

        off_e = tl.load(expert_of_tile_ptr + pid_m).to(tl.int64)
        if CONSTEXPR_ALIAS:
            # The naive experiment: the aliasing is a compile-time constant, so
            # the compiler can see the whole loop reads one address. Compiled
            # and counted, never timed as the answer.
            b_base = off_e * 0
            k_advance = 0
        else:
            b_base = (off_e * stride_be_eff
                      + pid_n.to(tl.int64) * stride_bn_blk_eff)
            k_advance = b_k_advance
        offs_bn = tl.arange(0, BLOCK_N).to(tl.int64)
        b_ptrs = (b_ptr + b_base
                  + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            if COMPUTE_DOT:
                acc += tl.dot(a, b)
            else:
                # STUDY.md item 4's own prescription. Both loads are consumed,
                # so neither can be eliminated, and the accumulator keeps its
                # full BLOCK_M x BLOCK_N float32 shape so register pressure and
                # therefore occupancy match the dot-mode kernel.
                acc += (tl.sum(tl.sum(a.to(tl.float32), axis=1), axis=0)
                        + tl.sum(tl.sum(b.to(tl.float32), axis=1), axis=0))
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += k_advance

        offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        tl.store(c_ptrs, acc)

    return _ablation_kernel


# --------------------------------------------------------------------------
# the ISA check
# --------------------------------------------------------------------------

#: Every PTX mnemonic that moves data between global memory and the SM. Counting
#: `ld.global` ALONE would read zero on this kernel: Triton pipelines its
#: global-to-shared copies as `cp.async.cg.shared.global` whenever num_stages is
#: above one, and on Hopper can issue `cp.async.bulk.tensor` instead. A fold
#: check that counts the wrong mnemonic passes silently and is worse than none.
GLOBAL_OPS = {
    "ld.global": r"\bld\.global[a-z0-9_.]*",
    "cp.async": r"\bcp\.async[a-z0-9_.]*",
    "st.global": r"\bst\.global[a-z0-9_.]*",
}

#: Reported beside the load counts, because a change in the reduction would show
#: up here first and would mean the two variants are not the same work.
COMPUTE_OPS = {
    "mma.sync": r"\bmma\.sync[a-z0-9_.]*",
    "wgmma": r"\bwgmma[a-z0-9_.]*",
    "ld.shared": r"\bld\.shared[a-z0-9_.]*",
}


def count_ops(ptx: str) -> dict[str, int]:
    counts = {name: len(re.findall(pattern, ptx))
              for name, pattern in {**GLOBAL_OPS, **COMPUTE_OPS}.items()}
    counts["global_loads"] = counts["ld.global"] + counts["cp.async"]
    return counts


def extract_ptx(launch_result, jit_fn, seen: set) -> tuple[str | None, str]:
    """The PTX of the kernel that was just launched, however this Triton keeps it.

    Probed rather than assumed. `JITFunction.run` returns the CompiledKernel in
    Triton 3.x, but the attribute has moved before and a wrong guess would make
    the fold check silently unavailable at exactly the moment it matters. Every
    route is tried and the one that worked is reported, so a report never says
    "counts equal" when it means "counts unavailable".

    THE CACHE FALLBACK TAKES THE NEWLY ADDED ENTRY, not the first one it finds.
    `JITFunction.cache` holds one entry per specialisation and accumulates
    across rungs and across the constexpr variant, so "the first entry with
    PTX" would hand back whichever kernel happened to compile first and the fold
    check would compare a kernel against itself. `seen` carries the keys already
    accounted for; an ambiguous fallback -- more than one new entry, or none --
    reports unavailable rather than guessing, because a wrong PTX here is worse
    than no PTX.
    """
    asm = getattr(launch_result, "asm", None)
    if isinstance(asm, dict) and asm.get("ptx"):
        return asm["ptx"], "launch return value"
    cache = getattr(jit_fn, "cache", None)
    fresh = []
    if isinstance(cache, dict):
        for device, device_cache in cache.items():
            if not isinstance(device_cache, dict):
                continue
            for key, compiled in device_cache.items():
                if (device, key) in seen:
                    continue
                seen.add((device, key))
                asm = getattr(compiled, "asm", None)
                if isinstance(asm, dict) and asm.get("ptx"):
                    fresh.append(asm["ptx"])
    if len(fresh) == 1:
        return fresh[0], "JITFunction.cache, newly compiled entry"
    if fresh:
        return None, (f"{len(fresh)} kernels compiled during that launch, so "
                      "which PTX belongs to it is ambiguous")
    return None, ("no PTX reachable from the launch result, and nothing new in "
                  "JITFunction.cache; the fold check cannot run on this Triton")


@dataclass
class IsaReading:
    variant: str
    source: str
    digest: str
    counts: dict[str, int]


def isa_gate(records: list[dict]) -> Gate:
    """Did the aliased launch issue the same global loads as the normal one?

    A silent fold is the failure this whole design is arranged around, and it
    would produce a clean alpha of nearly zero that looked like a triumph.

    COMPARED WITHIN A RUNG, NEVER POOLED, and that is not fussiness. Triton
    specialises on each rung's own K, N and strides, so mixtral's kernel and
    deepseek-v3's kernel are legitimately different compilations with different
    digests. A gate that pooled every reading and compared digests would fail on
    every multi-model run for a reason that has nothing to do with folding, and
    a gate that fails for the wrong reason gets disabled.

    The constexpr reading is reported beside the pair and is NOT gated: it is
    allowed -- expected, even -- to fold, and its folding is the evidence that
    the runtime form was necessary.
    """
    name = "ISA: the aliased kernel issued the same global loads"
    pairs = []
    for row in records:
        by_variant = {r["variant"]: r for r in row.get("isa", [])}
        if "normal" in by_variant and "aliased" in by_variant:
            pairs.append((row["id"], by_variant))
    if not pairs:
        return Gate(name, None,
                    "PTX was not reachable for both runtime variants on any "
                    "rung, so the fold check did not run. No alpha from this "
                    "run should be quoted until it does")
    bad_loads, bad_digest = [], []
    for rid, by_variant in pairs:
        normal, aliased = by_variant["normal"], by_variant["aliased"]
        if normal["counts"]["global_loads"] != aliased["counts"]["global_loads"]:
            bad_loads.append((rid, normal, aliased))
        elif normal["digest"] != aliased["digest"]:
            bad_digest.append(rid)
    first = pairs[0][1]
    detail = (f"{len(pairs)} rungs compared. First, {pairs[0][0]}: normal "
              f"{first['normal']['counts']['global_loads']} global-load "
              f"instructions (ld.global {first['normal']['counts']['ld.global']}, "
              f"cp.async {first['normal']['counts']['cp.async']}); aliased "
              f"{first['aliased']['counts']['global_loads']} "
              f"(ld.global {first['aliased']['counts']['ld.global']}, "
              f"cp.async {first['aliased']['counts']['cp.async']})")
    if bad_loads:
        rid, normal, aliased = bad_loads[0]
        detail += (f". {len(bad_loads)} rungs DIFFER, first {rid}: "
                   f"{normal['counts']['global_loads']} against "
                   f"{aliased['counts']['global_loads']}. The aliased launch "
                   "did not issue the same loads, so its time is the optimiser "
                   "and not the cache, and every number below is void")
    elif bad_digest:
        detail += (f". {len(bad_digest)} rungs match on counts but not on PTX "
                   f"digest, first {bad_digest[0]}. Same counts from DIFFERENT "
                   "code is weaker than this design intends: the two launches "
                   "were supposed to be one compiled kernel driven by three "
                   "runtime scalars, and Triton's equal-to-1 argument "
                   "specialisation is the usual reason they are not")
    else:
        detail += ". Every rung's two launches are ONE compiled kernel"
    constexpr = next((v["constexpr-alias"] for _, v in pairs
                      if "constexpr-alias" in v), None)
    if constexpr is not None:
        folded = (constexpr["counts"]["global_loads"]
                  < first["normal"]["counts"]["global_loads"])
        detail += (f". The constexpr-aliased kernel, the naive way to write "
                   f"this, issues {constexpr['counts']['global_loads']} and "
                   + ("DID fold, which is the hazard happening in front of you"
                      if folded else
                      "did NOT fold on this Triton, so the hazard is real in "
                      "principle and did not bite here"))
    return Gate(name, not bad_loads and not bad_digest, detail)


# --------------------------------------------------------------------------
# measuring, which is the only part that needs the box
# --------------------------------------------------------------------------

def ablation_scalars(rung: Rung, extent: str = DEFAULT_ALIAS_EXTENT
                     ) -> dict[str, dict[str, int]]:
    """The three runtime integers that ARE the ablation, per variant.

    THE EXTENT HAS EXACTLY TWO CALL SITES AND THEY MUST AGREE: this function,
    which tells the kernel what to alias, and `check_output`, which says what
    the aliased output must then equal. A change made here alone leaves the
    correctness gate comparing the new kernel against the old closed form,
    which is a FAIL that reads as a broken alias; a change made THERE alone is
    worse, because it is a reference that agrees with an alias that did not
    happen. `test_the_alias_extent_reaches_both_call_sites` pins the pair, and
    until 2026-09-03 it did not: it called this function and never called
    `check_output` at all, so the test named for the recurring defect was an
    instance of it and the dangerous direction was the untested one. It now
    emulates this kernel's address arithmetic on the CPU in float64 from these
    very scalars and scores the result through `check_output` at BOTH extents,
    so neither call site can move alone.

    Pure arithmetic over the rung's shape, so the values the pod will pass can
    be checked on a laptop. `B` is a contiguous [E, N, K] tensor, matching
    vLLM's w1 layout, so its strides are (N*K, K, 1); `measure_rung` asserts the
    allocated tensor really has those strides rather than assuming it, because a
    non-contiguous B would change the address arithmetic under both variants
    while every table still printed numbers.

    NO ENTRY MAY EVER BE 1. Triton compiles an integer argument of exactly 1 in
    as a constant, which would put NORMAL and ALIASED in different
    specialisations and destroy the one-compiled-kernel guarantee the fold check
    rests on. Every entry is a large stride or 0, so both take the
    divisible-by-16 path. A test pins this for every shipped rung.
    """
    if extent not in ALIAS_EXTENTS:
        raise ValueError(f"unknown alias extent {extent!r}; "
                         f"expected one of {ALIAS_EXTENTS}")
    stride_be, stride_bn, stride_bk = rung.n * rung.k, rung.k, 1
    advance = rung.block_k * stride_bk
    return {
        "normal": {"stride_be_eff": stride_be,
                   "stride_bn_blk_eff": rung.block_n * stride_bn,
                   "b_k_advance": advance},
        # `block` KEEPS the advance: the aliased arm walks one BLOCK_N x K
        # column block, sequentially, exactly as the normal arm walks its own.
        # `tile` zeroes it and re-reads 16 KiB, which is the 2026-09-01 pinning.
        "aliased": {"stride_be_eff": 0, "stride_bn_blk_eff": 0,
                    "b_k_advance": 0 if extent == "tile" else advance},
    }


def check_output(rung: Rung, a, b, c, compute: str, aliased: bool,
                 torch, extent: str = DEFAULT_ALIAS_EXTENT) -> float | None:
    """Relative RMS error of a variant's output against its own closed form.

    BOTH DIRECTIONS ARE CHECKED AND BOTH ARE LOAD BEARING. NORMAL reproducing
    the reference proves it really read every expert's whole weight block, which
    is what W is supposed to be the cost of. ALIASED reproducing a DIFFERENT
    closed form -- one BLOCK_N x K column block of expert 0 at `--alias-extent
    block`, or `K/BLOCK_K` copies of one tile at `tile` -- proves the aliasing
    took effect, which is the failure mode where the three runtime scalars do
    nothing, D(n) is pure noise, and alpha comes out near zero for the second
    time in this project's history. The 2026-09-01 report asserted that a
    spread alias "has no closed form to check the output against" and used that
    to justify the 16 KiB pinning that cost it the experiment. It has one, and
    it is the branch below: the normal reference read at expert 0, N-block 0.

    THE EXTENT ARGUMENT IS THE SECOND OF TWO CALL SITES, `ablation_scalars`
    being the first, and the two are held together by a CPU emulation rather
    than by a comment: `test_the_alias_extent_reaches_both_call_sites` walks
    this kernel's pointer arithmetic in float64 from the scalars that function
    returns and scores the result here at both extents, matching extent within
    float32 rounding and mismatched extent off by a factor no tolerance
    absorbs. Reverting the `block` branch below to the tile closed form used to
    leave the suite green.

    NOTHING IS MATERIALISED AT FULL SIZE. deepseek-v3's weight tensor is 15 GiB
    in bf16, so a float64 copy of it is 60 GiB and an obvious `b.to(float64)`
    would turn a correctness check into an out-of-memory crash on the one model
    that matters most. Every reduction passes `dtype=` so torch accumulates in
    float64 without a cast, and the comparison is against the ONE distinct value
    each (M-tile, N-block) cell holds rather than against an expanded copy of
    the output.

    In `dot` mode there is no cheap closed form, so three experts -- first,
    middle and last -- are checked against a real matmul. Three rather than one
    because the failure this catches is "normal did not read every expert".
    """
    if compute == "dot":
        if aliased:
            return None
        picks = sorted({0, rung.experts // 2, rung.experts - 1})
        worst = 0.0
        for e in picks:
            lo = e * rung.rows_per_expert
            hi = lo + rung.rows_per_expert
            want = (a[lo:hi].to(torch.float32) @ b[e].to(torch.float32).T)
            got = c[lo:hi]
            scale = want.abs().mean().item()
            err = (got - want).abs().mean().item()
            worst = max(worst, err / scale if scale else err)
        return float(worst)

    # sum mode: every (M-tile, N-block) cell of the output holds ONE value.
    a_tile = a.view(rung.num_pid_m, rung.block_m, rung.k).sum(
        dim=(1, 2), dtype=torch.float64)
    if aliased:
        if extent == "tile":
            one = b[0, :rung.block_n, :rung.block_k].sum(dtype=torch.float64)
            one = one * rung.k_iters
        else:
            one = b[0, :rung.block_n, :].sum(dtype=torch.float64)
        cell = a_tile[:, None] + one
        cell = cell.expand(rung.num_pid_m, rung.num_pid_n)
    else:
        b_block = b.view(rung.experts, rung.num_pid_n, rung.block_n,
                         rung.k).sum(dim=(2, 3), dtype=torch.float64)
        tile_expert = (torch.arange(rung.num_pid_m, device=a.device)
                       // rung.tiles)
        cell = a_tile[:, None] + b_block[tile_expert]

    got = c.view(rung.num_pid_m, rung.block_m, rung.num_pid_n, rung.block_n)
    # Within a cell every element is the same computation, so any spread is a
    # kernel bug and is folded into the error rather than averaged away.
    spread = (got.amax(dim=(1, 3)) - got.amin(dim=(1, 3))).to(torch.float64)
    err = (got[:, 0, :, 0].to(torch.float64) - cell).abs() + spread
    scale = cell.abs().mean().item()
    return float(err.mean().item() / scale) if scale else float(err.mean().item())


#: The `timing.KernelTiming` columns every measured rung carries into
#: `cells.jsonl`. Named as a group so the record builder and any reader of the
#: file cannot drift apart, and listed here because the question a reader asks
#: of a published alpha is "was this rung at the roof's clock, and was its L2
#: cold" -- which a prose note cannot answer and cannot be filtered on.
TIMING_COLUMNS = ("instrument", "warmup_ms", "iters", "trials",
                  "sm_clock_load_mhz", "clock_level_ok", "clock_drift_ok",
                  "l2_flush", "host_bound")


def _fold_flag(values: list, bad: bool = False) -> bool | None:
    """Fold a rung's tri-state flags so a filter cannot be too permissive.

    `bad` is the value that must dominate: False for the two clock flags (one
    bad pass makes the rung bad), True for `host_bound` (one host-bound pass
    makes the rung host-bound). None absorbs everything the dominant value did
    not decide, because NOT DETERMINED is not the same as OK and reading it as
    OK is how a throttled rung keeps its alpha.
    """
    if any(v is bad for v in values):
        return bad
    if any(v is None for v in values):
        return None
    return not bad


def fold_timings(timings: list) -> dict:
    """One rung's `time_kernel` calls reduced to one set of columns.

    A rung is `3 * replicates` calls -- normal, aliased and placebo, shuffled --
    so it has that many `KernelTiming` records and the row has one of each
    column. The reductions are chosen so that a filter over the row cannot be
    more permissive than a filter over the calls: the flags fold with the bad
    value dominating, `sm_clock_load_mhz` is the median of the calls that
    reported one, and `iters`, `trials` and `warmup_ms` are per-call medians
    rather than totals so `iters * trials * calls = samples` still checks out.
    `instrument` and `l2_flush` are constant across one rung by construction.
    """
    if not timings:
        return {}
    clocks = [t.sm_clock_load_mhz for t in timings if t.sm_clock_load_mhz]
    return {
        "instrument": timings[0].instrument,
        "warmup_ms": float(statistics.median([t.warmup_ms for t in timings])),
        "iters": int(statistics.median([t.iters for t in timings])),
        "trials": int(statistics.median([t.trials for t in timings])),
        "sm_clock_load_mhz": (float(statistics.median(clocks)) if clocks
                              else None),
        "clock_level_ok": _fold_flag([t.clock_level_ok for t in timings]),
        "clock_drift_ok": _fold_flag([t.clock_drift_ok for t in timings]),
        "l2_flush": bool(timings[0].l2_flush),
        "host_bound": _fold_flag([t.host_bound for t in timings], bad=True),
    }


def reference_clock_for(gpu_name: str):
    """The clock this card's roof was measured at, which LEVEL is scored against.

    Resolved through `roofline.reference_clock` rather than as another copy of
    the three-field rule. `block_m_crossing_sweep`, `dtype_tile_confound` and
    `memory_branch_anchor` each read those fields their own way, and copies of
    one rule are how the driver came to believe no committed calibration
    recorded a clock while the sweep read 1515 MHz out of the same yaml.

    Returns the `ReferenceClock`, whose one `source` string says where the
    number came from when there is one and why there is none when there is not,
    so a provenance and a reason cannot drift apart.

    A CLOCK IS NOT A BANDWIDTH, which is the only reason this arm is allowed to
    touch `roofline` at all. `test_this_script_never_reaches_the_byte_model_or_a
    _calibrated_bandwidth` pins the surface: this file reads `HARDWARE_DIR`,
    `measured_slug` and `reference_clock` out of that module, all three of which
    are readers of a hardware FILE, and none of the roof, ridge, attainable or
    efficiency arithmetic that would make this alpha a function of the refit's
    ruler. Same import shape as `measured_card`, deliberately, so there is one
    door.
    """
    from moe.bench import roofline as RF
    return RF.reference_clock(gpu_name or None)


def require_reference_clock(torch):
    """Resolve the LEVEL reference once for this run, or REFUSE the run.

    THE FLAG HAD NO LEFT-HAND SIDE HERE, AT BOTH CALL SITES.
    `timing.clock_flags` will not invent one: handed no reference it leaves
    `clock_level_ok` None on every pass, `_fold_flag` folds three Nones to None
    because NOT DETERMINED is not OK, and a card pegged at 1400 MHz for a whole
    ladder produces a page with no level failure anywhere on it. Both this
    arm's `time_kernel` calls -- the probe's and the ladder's -- passed no
    reference until 2026-09-03, so every rung it has ever published carries the
    column and no verdict in it.

    THIS IS THE ARM THAT CANNOT ABSORB A SAG. Everywhere else in this tree a
    clock excursion inflates one cell and the cell is excluded. Here the answer
    is a ratio of two DIFFERENCES between two ladders, D(n)/D(1), and a sag part
    way up a ladder moves the numerator and the denominator by different
    amounts: the slope-over-intercept that cancels the time unit does not cancel
    that. The difference between two ladders IS the result.

    REFUSED RATHER THAN RESOLVE-AND-SAY, which is the opposite of the choice
    `scripts/group_m_alpha_sweep.py:reference_clock_for` documents, and the
    difference is what the column is load-bearing for. That arm's correctness
    gates and paired ratios stand without it, so an absent yaml there is worth
    printing and running past rather than turning into a lost hour of card.
    Here the column is the only evidence for the only number the arm produces,
    and the remedy is one minute of `calibrate_hardware.py --publish` on a box
    that is already rented and already running. A minute against an hour.

    A card is attached by the time either caller reaches this: both check
    `torch.cuda.is_available()` first, so `ReferenceClock.card` is never the
    empty laptop case here and a None `mhz` always means a pod whose ruler was
    never measured.
    """
    ref = reference_clock_for(torch.cuda.get_device_properties(0).name)
    if ref.mhz is None:
        raise CannotRunHere(
            f"no LEVEL reference for this card: {ref.source}. Every cell this "
            "arm would write records clock_level_ok undetermined, and alpha "
            "here is a ratio of two ladders' DIFFERENCE, which a clock sag "
            "moves and which no gate on the page could then see. Run `python "
            "scripts/calibrate_hardware.py --publish` on this box first: it is "
            "a minute against this arm's hour.")
    print(f"[alias] LEVEL reference: {ref.mhz:.0f} MHz, {ref.source}")
    return ref


def measure_rung(kernel, rung: Rung, design: Design, args,
                 flusher, order_seed: int, torch, seen_kernels: set,
                 reference_clock_mhz: float) -> dict:
    """Time both variants of one rung, interleaved, on the same tensors.

    THE INTERLEAVING IS THE POINT. Normal, aliased and a SECOND normal are timed
    in a shuffled order inside every replicate, on tensors allocated once, so a
    clock that drifts during the rung hits all three roughly equally and the
    second normal measures how much drift is left. That second pass is the
    placebo: two launches of an identical configuration, differing in nothing,
    whose difference is the noise floor D(n) has to beat.

    ONE INSTRUMENT (A7), AND THIS LOOP WAS THE FOURTH COPY. Until 2026-09-02 the
    replicate loop below created a fresh `torch.cuda.Event` pair INSIDE itself,
    recorded `start` on a stream the previous iteration's `torch.cuda.synchronize`
    had just drained, and synchronised after every single launch: the exact shape
    the audit condemns, differing from the three deleted `time_call` copies only
    in that it flushed. It exposed a full host launch prefix in every sample, it
    read no clock UNDER LOAD -- the two `ClockState.sample()` calls beside it are
    idle instants, and the audit showed those detect whether the START sample
    caught the idle boost rather than whether the kernel throttled -- and it
    reached `cells.jsonl` carrying none of the `KernelTiming` columns, so no
    reader could tell a cell at the roof's clock from a cell at two thirds of it.
    It survived the first pass of this slice because it had no function name and
    the acceptance check greps for the RETIRED LOOP'S NAME. A check keyed on a
    name cannot see a loop that was never given one, which is why
    `tests/test_p7_instrument_and_gates.py` now parses this function and looks
    for the shape instead: a `cuda.Event` construction or an `elapsed_time` call
    anywhere inside it.

    Every pass is now one `timing.time_kernel` call, still inside the shuffled
    replicate loop, so the interleaving that protects the paired difference is
    unchanged and each pass carries its own instrument, warmup, iteration count
    and three verdicts. `fold_timings` reduces the rung's calls to one set of
    columns with the BAD value dominating.

    `reference_clock_mhz` is REQUIRED and has no default, because a default is
    how the third verdict came to be undetermined on every published rung: the
    call below carried the `clock_level_ok` column and passed nothing for
    `clock_flags` to score it against. `require_reference_clock` resolves it
    once per run and refuses the run when this card has none, so by here it is a
    number.
    """
    from moe.bench import timing as T
    from moe.bench.timing import ClockState, clock_drift

    device = "cuda"
    torch.manual_seed(0)
    # `empty().uniform_()` rather than `torch.rand(float32).to(bfloat16)`: the
    # float32 intermediate for deepseek-v3's weights is 30 GiB and would run the
    # card out of memory before a single kernel launched.
    a = torch.empty((rung.total_rows, rung.k), device=device,
                    dtype=torch.bfloat16).uniform_(-0.5, 0.5)
    b = torch.empty((rung.experts, rung.n, rung.k), device=device,
                    dtype=torch.bfloat16).uniform_(-0.5, 0.5)
    c = torch.zeros((rung.total_rows, rung.n), device=device, dtype=torch.float32)
    expert_of_tile = (torch.arange(rung.num_pid_m, device=device,
                                   dtype=torch.int32) // rung.tiles)

    grid = (rung.programs,)
    common = dict(
        EM=rung.total_rows, N=rung.n, K=rung.k,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bn=b.stride(1), stride_bk=b.stride(2),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_M=rung.block_m, BLOCK_N=rung.block_n, BLOCK_K=rung.block_k,
        GROUP_M=rung.group_m, COMPUTE_DOT=(design.compute == "dot"),
    )
    if tuple(b.stride()) != (rung.n * rung.k, rung.k, 1):
        raise CannotRunHere(
            f"the weight tensor is not contiguous [E, N, K]: strides "
            f"{tuple(b.stride())}. `ablation_scalars` computes the three "
            "runtime integers from the shape, and a different layout would "
            "alias something other than one BLOCK_K x BLOCK_N tile while every "
            "table still printed numbers.")
    scalars = ablation_scalars(rung, design.alias_extent)

    def launch(variant: str, constexpr_alias: bool = False):
        return kernel[grid](
            a, b, c, expert_of_tile,
            CONSTEXPR_ALIAS=constexpr_alias, num_warps=design.tile["num_warps"],
            num_stages=design.tile["num_stages"],
            **scalars[variant], **common)

    readings: list[IsaReading] = []
    correctness: dict[str, float | None] = {}
    for variant in ("normal", "aliased"):
        c.zero_()
        handle = launch(variant)
        torch.cuda.synchronize()
        ptx, source = extract_ptx(handle, kernel, seen_kernels)
        if ptx is not None:
            readings.append(IsaReading(
                variant=variant, source=source,
                digest=hashlib.sha256(ptx.encode()).hexdigest()[:12],
                counts=count_ops(ptx)))
        correctness[variant] = check_output(
            rung, a, b, c, design.compute, variant == "aliased", torch,
            design.alias_extent)
    with contextlib.suppress(Exception):
        c.zero_()
        handle = launch("normal", constexpr_alias=True)
        torch.cuda.synchronize()
        ptx, source = extract_ptx(handle, kernel, seen_kernels)
        if ptx is not None:
            readings.append(IsaReading(
                variant="constexpr-alias", source=source,
                digest=hashlib.sha256(ptx.encode()).hexdigest()[:12],
                counts=count_ops(ptx)))

    for _ in range(3):
        launch("normal")
        launch("aliased")
    torch.cuda.synchronize()

    rng = random.Random(order_seed)
    samples: dict[str, list[float]] = {"normal": [], "aliased": [], "placebo": []}
    timings: list = []
    clock_start = ClockState.sample()
    for _ in range(args.replicates):
        passes = ["normal", "aliased", "placebo"]
        rng.shuffle(passes)
        for name in passes:
            variant = "normal" if name == "placebo" else name
            measured = T.time_kernel(
                lambda v=variant: launch(v), warmup_ms=args.warmup,
                target_ms=args.cell_budget_ms, trials=args.trials,
                l2_flush=bool(args.l2_flush),
                reference_clock_mhz=reference_clock_mhz,
                flusher=flusher if args.l2_flush else None)
            samples[name].append(measured.ms_p50)
            timings.append(measured)
    clock_end = ClockState.sample()
    drift, throttled = clock_drift(clock_start, clock_end)

    # NOT `del a, b, c` here: `launch` closes over them, and deleting a name a
    # nested function reads is a live bug the moment anyone adds a call below.
    # The tensors die with the frame and the caller drops the cache.
    return {
        "kind": "rung", "id": rung.key, "model": rung.model,
        "tiles": rung.tiles, "block_m": rung.block_m,
        "rows_per_expert": rung.rows_per_expert,
        "experts": rung.experts, "k": rung.k, "n": rung.n,
        "per_expert_bytes": rung.per_expert_bytes,
        "weight_bytes": rung.weight_bytes,
        "activation_fraction": rung.activation_fraction,
        "programs": rung.programs, "control": rung.control,
        **cell_knobs(design),
        "ms": {name: values for name, values in samples.items()},
        "correctness": correctness,
        "isa": [{"variant": r.variant, "source": r.source, "digest": r.digest,
                 "counts": r.counts} for r in readings],
        # The two idle-instant samples are KEPT beside the queue-deep ones, not
        # instead of them: they are what every published rung was scored on, so
        # a resumed jsonl still parses and the next pod can compare the two.
        "sm_clock_start": clock_start.sm_clock_mhz,
        "sm_clock_end": clock_end.sm_clock_mhz,
        "clock_drift": drift, "throttled": bool(throttled),
        # WHAT LEVEL WAS SCORED AGAINST, on the row LEVEL was scored on. A
        # `clock_level_ok` False with no reference beside it is an exclusion a
        # reader cannot trace back to a field in a file, and this arm's rows
        # outlive the pod they were measured on.
        "reference_clock_mhz": reference_clock_mhz,
        **fold_timings(timings),
        "provenance": "measured",
    }


# --------------------------------------------------------------------------
# the probe: a tenth of the arm's wall, spent to protect the other nine
# --------------------------------------------------------------------------

#: The pinnings the probe walks, and every entry earns its compile.
#:
#: The quantity being maximised is the ALIASED ladder's achieved request
#: bandwidth, because the whole design can only see DRAM while that number is
#: above the card's DRAM roof. The 2026-09-01 run measured 0.61 of the roof at
#: the first row of this table, so the shipped pinning is known to fail and the
#: rest of the table is the search for one that does not.
#:
#:   num_stages    bytes in flight per program. 3 was shipped; 4 and 5 are the
#:                 cheapest lever on a loop that is doing nothing but loading.
#:   BLOCK_K       128 halves the number of reduction trees per byte and doubles
#:                 the transaction length. K is 2048 to 7168 on every model here
#:                 so 128 still divides exactly, which the preflight re-checks.
#:   num_warps     4 raises the programs resident per SM, 16 raises the width of
#:                 each one. Both are ways past an issue-rate ceiling and which
#:                 one works is not predictable from here.
#:   compute       `dot` moves the reduction onto the tensor cores. It is BIASED
#:                 (see the header) and `prediction_gate` still refuses to
#:                 answer P1 from it, but the bias is bounded at BLOCK_M/ridge =
#:                 16/162.8 and is identical in both arms, while a shared
#:                 ceiling below the DRAM roof is not a bias at all: it is the
#:                 absence of a measurement. `choose_pinning` prefers `sum`
#:                 whenever a `sum` pinning clears, and says so.
PROBE_PINNINGS = (
    {"num_warps": 8, "num_stages": 3, "block_k": 64, "compute": "sum"},
    {"num_warps": 8, "num_stages": 4, "block_k": 128, "compute": "sum"},
    {"num_warps": 4, "num_stages": 5, "block_k": 128, "compute": "sum"},
    {"num_warps": 8, "num_stages": 4, "block_k": 128, "compute": "dot"},
    {"num_warps": 8, "num_stages": 3, "block_k": 64, "compute": "dot"},
    {"num_warps": 16, "num_stages": 4, "block_k": 128, "compute": "dot"},
)

#: The probe's own timing knobs. Short on purpose: it is measuring a slope
#: between two rungs of one small model to three significant figures, not
#: publishing anything, and every millisecond here is taken off the ladder.
PROBE_TILES = (1, 2)
PROBE_WARMUP_MS = 100.0
PROBE_CELL_BUDGET_MS = 20.0
PROBE_TRIALS = 2


def probe_model(design: Design) -> str | None:
    """The smallest non-control weight tensor in the design.

    Smallest because the probe measures a RATE, which is geometry independent
    (the 2026-09-01 run read 0.607 to 0.616 across a 20x spread in footprint),
    so the cheapest geometry answers the same question as the dearest one and
    leaves its allocation time on the ladder.
    """
    real = [r for r in design.rungs if not r.control]
    if not real:
        return None
    return min(real, key=lambda r: r.weight_bytes).model


def choose_pinning(readings: list[dict], roof_bytes_s: float | None,
                   dot_fallback: bool = True) -> tuple[dict | None, str]:
    """The pinning to run the ladder at, or None and the reason there is none.

    A pinning CLEARS when its aliased ladder delivers requests faster than the
    card's DRAM roof by `MIN_HEADROOM_RATIO`, which is the condition under which
    DRAM binds in the normal arm and the ablation has something to remove.

    AMONG THE CLEARING PINNINGS, `sum` WINS OVER `dot` EVEN WHEN IT IS SLOWER.
    Speed is not the objective; headroom is, and once a pinning has headroom the
    remaining question is which estimator is less biased. `sum` is the unbiased
    one and `prediction_gate` will not answer P1 from `dot` at all, so a `dot`
    pinning that is 30% faster and cannot answer the question loses to a `sum`
    pinning that can. Only when no `sum` pinning clears does `dot` get the
    ladder, and then the report says the answer is a bound.

    THE FALL TO `dot` IS THE OPERATOR'S AND NOT THIS FUNCTION'S. Half the probe
    grid is `dot`, and the fall is the LIKELY case rather than the corner: the
    0.61-of-roof ceiling this arm exists to escape has the signature of the
    cross-lane `tl.sum` tree, which is exactly what `dot` removes. A ladder run
    in `dot` mode leaves P1 UNKNOWN, `exit_codes.classify` maps an UNKNOWN CLAIM
    to CLAIM_FAIL, and the driver latches a CLAIM_FAIL and never retries, so a
    silent fall spends the whole arm and files "alpha is not 0.558" when what
    happened was "alpha was not asked". `--dot-fallback` makes the choice
    explicit at the command line, the report prints which was in force, and
    `dot_mode_reading` prints the disambiguation beside the verdict. Default
    `allow`, because refusing turns a lower bound on alpha into the 2026-09-01
    void for the sake of a code the log already explains.
    """
    if not roof_bytes_s:
        return None, ("no committed calibration for this card, so no pinning "
                      "can be shown to clear a roof that has not been measured")
    scored = [(r["aliased_bytes_s"] / roof_bytes_s, r) for r in readings
              if r.get("aliased_bytes_s")]
    if not scored:
        return None, "no pinning produced an aliased ladder slope"
    clearing = [(ratio, r) for ratio, r in scored if ratio >= MIN_HEADROOM_RATIO]
    if not clearing:
        best_ratio, best = max(scored, key=lambda pair: pair[0])
        return None, (
            f"no pinning cleared the roof. The best of {len(scored)} was "
            f"{best['pinning']} at {best['aliased_bytes_s'] / 1e9:.0f} GB/s, "
            f"{best_ratio:.3f} of the measured read roof of "
            f"{roof_bytes_s / 1e9:.0f} GB/s, against a limit of "
            f"{MIN_HEADROOM_RATIO}. Every one of them is limited by a shared "
            "non-DRAM path SLOWER than DRAM, so in the normal arm DRAM has "
            "slack and removing it cannot move the clock. No ladder run at any "
            "of these pinnings could have measured alpha, so none was run")
    sums = [pair for pair in clearing if pair[1]["pinning"]["compute"] == "sum"]
    if not sums and not dot_fallback:
        best_ratio, best = max(clearing, key=lambda pair: pair[0])
        return None, (
            f"no SUM-mode pinning cleared the roof, and --dot-fallback refuse "
            f"was in force. {len(clearing)} dot-mode pinning(s) did clear, the "
            f"best {best['pinning']} at {best_ratio:.3f} of the roof, and a "
            "ladder run at one of them would have measured a LOWER BOUND on "
            "alpha rather than alpha. Re-run with --dot-fallback allow to buy "
            "that bound, or widen the sum half of PROBE_PINNINGS")
    pool = sums or clearing
    ratio, best = max(pool, key=lambda pair: pair[0])
    why = ("the fastest sum-mode pinning that clears" if sums else
           "no sum-mode pinning cleared, so --dot-fallback allow took the "
           "fastest dot-mode one and the answer it produces is a LOWER BOUND, "
           "not P1's answer")
    return best["pinning"], (
        f"{best['pinning']} at {best['aliased_bytes_s'] / 1e9:.0f} GB/s, "
        f"{ratio:.3f} of the roof: {why}")


def measure_probe(design: Design, args, roof_bytes_s: float | None,
                  torch) -> list[dict]:
    """Time the probe grid. GPU only; the DECIDING is in `choose_pinning`.

    Two rungs and two variants per pinning. The reading kept is the aliased
    ladder's slope between the two rungs, which is the achieved request
    bandwidth of the shared path and the only thing that decides whether any
    ladder here can see DRAM.

    THE TENSORS ARE REBUILT PER RUNG AND DROPPED, because BLOCK_K moves across
    the grid and a pinning that runs out of memory must not take the rest of the
    grid with it. `launch` takes them as DEFAULT ARGUMENTS rather than closing
    over the names, which is what makes the `del` below safe; `measure_rung`
    carries the opposite warning for the opposite reason.

    NO CORRECTNESS CHECK RUNS HERE, deliberately. The probe measures a rate and
    quotes no alpha; the ladder that follows checks both closed forms on every
    rung and `correctness` is a VALIDITY gate, so an alias that did nothing
    cannot reach a published number through this door.

    THE PROBE IS THE SECOND `time_kernel` CALL SITE AND IT NEEDS THE REFERENCE
    AS MUCH AS THE FIRST. It does not publish a number, but it CHOOSES the
    pinning the ladder is then measured at, off achieved request bandwidths a
    sagging clock deflates. A pinning rejected for running at 0.61 of the read
    roof, when the card was at 0.7 of its clock while it was tried, is the arm
    stopping itself for the wrong reason and the operator re-renting to find
    out. Fixing the ladder alone would have been this defect's ninth instance
    inside its own repair.
    """
    from moe.bench import timing as T
    from moe.bench.timing import L2Flusher, flush_mb_for_device

    reference_clock_mhz = require_reference_clock(torch).mhz
    model = probe_model(design)
    if model is None:
        return []
    flusher = L2Flusher(flush_mb_for_device() if args.l2_flush else 0)
    readings: list[dict] = []
    for pinning in PROBE_PINNINGS:
        tile = dict(design.tile, BLOCK_K=pinning["block_k"],
                    num_warps=pinning["num_warps"],
                    num_stages=pinning["num_stages"])
        rungs = [rung_for(model, t, design.block_m, tile) for t in PROBE_TILES]
        if any(r.k % r.block_k for r in rungs):
            readings.append({"pinning": pinning, "aliased_bytes_s": None,
                             "note": f"K {rungs[0].k} is not a multiple of "
                                     f"BLOCK_K {pinning['block_k']}"})
            continue
        try:
            kernel = build_kernel()
            points = []
            normal_ms = None
            for rung in rungs:
                a = torch.empty((rung.total_rows, rung.k), device="cuda",
                                dtype=torch.bfloat16).uniform_(-0.5, 0.5)
                b = torch.empty((rung.experts, rung.n, rung.k), device="cuda",
                                dtype=torch.bfloat16).uniform_(-0.5, 0.5)
                c = torch.zeros((rung.total_rows, rung.n), device="cuda",
                                dtype=torch.float32)
                eot = (torch.arange(rung.num_pid_m, device="cuda",
                                    dtype=torch.int32) // rung.tiles)
                scalars = ablation_scalars(rung, design.alias_extent)
                common = dict(
                    EM=rung.total_rows, N=rung.n, K=rung.k,
                    stride_am=a.stride(0), stride_ak=a.stride(1),
                    stride_bn=b.stride(1), stride_bk=b.stride(2),
                    stride_cm=c.stride(0), stride_cn=c.stride(1),
                    BLOCK_M=rung.block_m, BLOCK_N=rung.block_n,
                    BLOCK_K=rung.block_k, GROUP_M=rung.group_m,
                    COMPUTE_DOT=(pinning["compute"] == "dot"))

                def launch(variant, k=kernel, r=rung, aa=a, bb=b, cc=c, ee=eot,
                           sc=scalars, cm=common, pn=pinning):
                    return k[(r.programs,)](
                        aa, bb, cc, ee, CONSTEXPR_ALIAS=False,
                        num_warps=pn["num_warps"], num_stages=pn["num_stages"],
                        **sc[variant], **cm)

                for variant in ("normal", "aliased"):
                    measured = T.time_kernel(
                        lambda v=variant: launch(v),
                        warmup_ms=PROBE_WARMUP_MS,
                        target_ms=PROBE_CELL_BUDGET_MS, trials=PROBE_TRIALS,
                        l2_flush=bool(args.l2_flush),
                        reference_clock_mhz=reference_clock_mhz,
                        flusher=flusher if args.l2_flush else None)
                    if variant == "aliased":
                        points.append((rung.tiles, measured.ms_p50))
                    elif rung.tiles == 1:
                        normal_ms = measured.ms_p50
                del a, b, c, eot
                torch.cuda.empty_cache()
            line = _fit_line(sorted(points))
            slope = line[1] if line else 0.0
            readings.append({
                "pinning": pinning, "model": model,
                "aliased_bytes_s": (rungs[0].weight_bytes / (slope * 1e-3)
                                    if slope > 0 else None),
                "aliased_ms": dict(points), "normal_ms_t1": normal_ms,
                "note": "" if slope > 0 else "aliased ladder slope was not positive",
            })
        except CannotRunHere:
            # NOT DATA. A missing triton or a card that cannot hold the tensors
            # is a precondition, and swallowing it here would turn a REFUSED
            # arm into "no pinning cleared the roof", which is INVALID and says
            # something false about the kernel.
            raise
        except Exception as exc:  # noqa: BLE001 - one pinning failing IS data
            readings.append({"pinning": pinning, "aliased_bytes_s": None,
                             "note": f"{type(exc).__name__}: {exc}"})
    return readings


def report_probe(say, readings: list[dict], roof_bytes_s: float | None,
                 chosen: dict | None, why: str,
                 dot_fallback: bool = True) -> None:
    say()
    say("## the probe: can any pinning see DRAM at all")
    say()
    if dot_fallback:
        say("  --dot-fallback allow: if no sum-mode pinning clears, the "
            "fastest dot-mode one runs")
        say("  the ladder, P1 comes back UNKNOWN and the process exits 1. That "
            "1 is 'not asked',")
        say("  not 'refuted', and the verdict block below says so in words.")
    else:
        say("  --dot-fallback refuse: if no sum-mode pinning clears, the arm "
            "STOPS at the probe")
        say("  rather than measure a lower bound. That is INVALID and it costs "
            "only the probe.")
    say()
    say("  Every pinning below is timed on the SMALLEST model at "
        f"{PROBE_TILES} tiles. The number that")
    say("  decides is the ALIASED ladder's achieved request bandwidth: while it "
        "is BELOW the")
    say("  card's DRAM roof, the shared non-DRAM path is the slower one, DRAM "
        "has slack in the")
    say("  normal arm, and no ablation of DRAM can move the clock however "
        "large alpha is.")
    say()
    say("  warps  stages  BLOCK_K  compute      aliased GB/s   of roof   note")
    for reading in readings:
        pin = reading["pinning"]
        rate = reading.get("aliased_bytes_s")
        rate_s = f"{rate / 1e9:14.0f}" if rate else f"{'none':>14s}"
        ratio = (f"{rate / roof_bytes_s:9.3f}" if rate and roof_bytes_s
                 else f"{'-':>9s}")
        say(f"  {pin['num_warps']:5d}  {pin['num_stages']:6d}  "
            f"{pin['block_k']:7d}  {pin['compute']:7s} {rate_s} {ratio}   "
            f"{reading.get('note', '')}")
    say()
    if roof_bytes_s:
        say(f"  measured read roof: {roof_bytes_s / 1e9:.0f} GB/s; a pinning "
            f"clears at {MIN_HEADROOM_RATIO} of it.")
    say(f"  ADOPTED: {why}" if chosen else f"  NONE ADOPTED: {why}")
    if not chosen:
        say()
        say("  This is the 2026-09-01 result restated in the units that name "
            "its cause, and it")
        say("  is rehearsable off-GPU: --synthetic "
            f"{SYNTHETIC_ALIAS_LAWS[1]} plants exactly this world and "
            f"--synthetic {SYNTHETIC_ALIAS_LAWS[0]} plants its opposite.")


# --------------------------------------------------------------------------
# what this costs, in the units the session driver reads
# --------------------------------------------------------------------------

#: Fraction of the card's read roof this kernel is assumed to deliver when the
#: cost is priced. 0.61 because that is what it MEASURED on 2026-09-01, on five
#: geometries spanning 20x in footprint, to three digits. It is the only
#: measured number available for this kernel and it is stated rather than
#: rounded to a convenient one; a probe that finds a faster pinning makes every
#: figure below an OVER-estimate, which is the right direction for a booking.
KERNEL_EFFICIENCY_PRIOR = 0.61

#: The one measured wall-over-model ratio in this repository:
#: `scripts/replicate_noise_floor.py` sets it from its own ARMS.tsv, where the
#: mixtral_g1 arm logged 127 s of wall against a cost model's 54 s.
#: `scripts/h200_gaps_session.sh:wall_over_model_pct` reads the same 235.
WALL_OVER_KERNEL = 127.0 / 54.0

#: Seconds allowed per PROBE PINNING for the work `WALL_OVER_KERNEL` cannot
#: cover: one Triton specialisation (BLOCK_K, num_warps and num_stages all move
#: across `PROBE_PINNINGS`, so no two entries share a compile) and the two rung
#: tensor sets that pinning allocates, fills and frees.
#:
#: AN ALLOWANCE AND NOT A MEASUREMENT, and it is charged separately because the
#: 2.35x was measured on `replicate_noise_floor`'s mixtral_g1, an arm that
#: compiled ONE pinning. Scaling the probe's 0.1 kernel minutes by it yields
#: 0.24 wall minutes for a grid that compiles six kernels, which is the one
#: place in this table where the estimate was under the truth rather than over
#: it, and a booking that is under is the booking that runs out of pod. 12.0 is
#: `scripts/memory_branch_anchor.py:estimated_seconds`'s own per-setting compile
#: allowance, which is the largest figure this repository states for a Triton
#: compile; `scripts/ruler_rebaseline.py` uses 10.0 for the same job.
PROBE_FIXED_S_PER_PINNING = 12.0


def estimated_kernel_ms(design: Design, args, roof_bytes_s: float | None
                        ) -> float | None:
    """GPU milliseconds the ladder will deliver, from the byte model.

    Per pass, `time_kernel` delivers `warmup_ms` of sustained load and then
    `trials` trials of `iters` calls, where `iters` is sized from the warmup's
    own per-call time against `target_ms` and has a floor of ten. The floor is
    what dominates the big rungs: deepseek-v3 at eight tiles is tens of
    milliseconds per call, so ten calls is a 400 ms trial and not the 50 ms
    trial the budget asks for. Pricing without the floor under-counts the top
    of the ladder by an order of magnitude, which is how an arm gets booked at
    eleven minutes and spends thirty-six.
    """
    if not roof_bytes_s:
        return None
    total = 0.0
    for rung in design.rungs:
        per_call = (rung.weight_bytes * rung.tiles
                    / (roof_bytes_s * KERNEL_EFFICIENCY_PRIOR) * 1e3)
        iters = max(10, math.ceil(args.cell_budget_ms / max(per_call, 1e-9)))
        one_pass = args.warmup + iters * per_call * args.trials
        total += 3 * design.replicates * one_pass
    return total


def report_cost(say, design: Design, args, roof_bytes_s: float | None,
                probing: bool) -> None:
    """The booking, labelled WALL or KERNEL the way the session driver labels it.

    `scripts/h200_gaps_session.sh:arm_clock` sorts every arm's figure into WALL
    (a plan that charges its own compiles and allocation) or KERNEL (one that
    does not) and prints a second total with the KERNEL rows put on the wall
    clock. An arm that prints one number and does not say which it is gets
    booked as whichever the driver guesses.

    THE ONLY DURATION THIS FILE NAMES IS THE ONE THIS FUNCTION PRINTS. The
    header carried its own copy until 2026-09-03, "three minutes before it
    spends sixty", against the figures printed below, and a header is what a
    driver owner books from. The header now states a ratio and no minutes, and
    `test_the_header_quotes_no_duration_of_its_own` scores that ratio against
    this table rather than against the prose.

    THE PROBE'S COMPILES ARE CHARGED HERE AND NOT LEFT TO THE RATIO. Everywhere
    else in this table an error is an over-estimate, which is the safe
    direction; the probe was the one place it ran the other way, because 2.35x
    was measured on an arm that compiled one pinning and the probe compiles six.
    `PROBE_FIXED_S_PER_PINNING` is added to the WALL figure outright and the
    line says so, so the WALL figure is bookable as printed and the driver's
    `arm_unpriced` entry for this arm can be empty.

    AND THE FIGURE IS THE ARM'S, NOT THIS INVOCATION'S, WHICH IS WHY `probing`
    IS NOT `args.probe and args.run`. It was, until 2026-09-03, and the page a
    rental is sized from is the BARE invocation: there is no card yet when an
    operator books one, so `--run` is false there, the probe's cost was denied,
    and the only page anybody could read under-booked the arm by the tenth of
    it the probe spends. Every other error in this table is an over-estimate.
    `--no-probe` still prints the smaller figure, because that is an operator
    saying the probe will not be spent, and the BOOKING line below names the
    command each figure belongs to so the two cannot be confused.
    """
    kernel_ms = estimated_kernel_ms(design, args, roof_bytes_s)
    say()
    say("## what this costs")
    say()
    if kernel_ms is None:
        say("  NOT PRICED. There is no committed calibration for this card, so "
            "there is no")
        say("  bandwidth to turn the byte model into milliseconds. Run "
            "scripts/calibrate_hardware.py")
        say("  --publish first; this arm needs that file for its gates as well "
            "as for this line.")
        return
    say(f"  BOOKING   `alias_ablation.py --run"
        f"{'' if probing else ' --no-probe'}` on one card. These are that "
        "RUN's figures,")
    say("            whether or not this invocation is one: a plan is read "
        "before there is a card.")
    say()
    probe_ms = (len(PROBE_PINNINGS) * len(PROBE_TILES) * 2
                * (PROBE_WARMUP_MS + PROBE_TRIALS * PROBE_CELL_BUDGET_MS)
                if probing else 0.0)
    probe_fixed_min = (len(PROBE_PINNINGS) * PROBE_FIXED_S_PER_PINNING / 60.0
                       if probing else 0.0)
    kernel_min = (kernel_ms + probe_ms) / 60000.0
    wall_min = kernel_min * WALL_OVER_KERNEL + probe_fixed_min
    say(f"  KERNEL  {kernel_min:5.1f} min   {len(design.rungs)} rungs x 3 "
        f"passes x {design.replicates} replicates, priced from the byte model "
        f"at {KERNEL_EFFICIENCY_PRIOR:.2f}")
    say("                        of this card's read roof, with time_kernel's "
        "ten-iteration floor applied")
    say(f"                        per rung. The probe adds "
        f"{probe_ms / 60000.0:.1f} min of that.")
    say("  EXCLUDES              Triton specialisations (one per model per "
        "pinning, plus the")
    say("                        constexpr variant), tensor allocation and "
        "fill, and the")
    say("                        closed-form correctness reductions. THIS IS "
        "NOT A WALL FIGURE.")
    say(f"  WALL    {wall_min:5.1f} min   the KERNEL figure at the one measured "
        f"wall-over-kernel ratio in")
    say(f"                        this repository, {WALL_OVER_KERNEL:.2f}x "
        "(replicate_noise_floor.py, mixtral_g1:")
    say(f"                        127 s of wall against a 54 s model), PLUS "
        f"{probe_fixed_min:.1f} min charged")
    say(f"                        outright for the probe: "
        f"{len(PROBE_PINNINGS)} distinct specialisations at "
        f"{PROBE_FIXED_S_PER_PINNING:.0f} s each,")
    say("                        with their tensor sets, which that ratio was "
        "not measured over")
    say("                        (mixtral_g1 compiled one pinning). BOOK THIS "
        "ONE.")
    if probe_fixed_min:
        probe_share = ((probe_ms / 60000.0 * WALL_OVER_KERNEL + probe_fixed_min)
                       / wall_min)
        say(f"  OF WHICH               {probe_share * 100:.0f}% is the probe, "
            "and a probe that clears no pinning")
        say("                        stops the arm having spent only that. It "
            "is the cheapest")
        say("                        thing here that can refuse the expensive "
            "one.")
    say()
    say("  The 0.61 is MEASURED, on 2026-09-01, on five geometries; a probe "
        "that finds a")
    say("  faster pinning makes both figures over-estimates, which is the "
        "right direction.")


def measurement_order(design: Design) -> tuple[Rung, ...]:
    """Every model at one tile count before any model's next. Cell by cell.

    THE CONTRAST IS PAIRED IN TIME, AND IT WAS NOT. The rungs used to be walked
    model-major -- all four of mixtral, then all four of qwen -- so the
    L2-resident control's ladder was the LAST four cells of the run and the real
    ladders it is supposed to bound were up to forty minutes earlier, on another
    thermal state of the card. `control_gate` compares them anyway. Tile-major
    puts each model's rung and the control's matching rung within one rung of
    each other, and the normal/aliased/placebo triple inside a rung was already
    interleaved within seconds.

    IT ALSO CHANGES WHAT AN INTERRUPTED HOUR LEAVES BEHIND. Model-major, a run
    killed at the two-thirds mark leaves three complete ladders and one empty
    one, and a model with no ladder contributes nothing. Tile-major leaves every
    model's ladder complete up to some tile count, which still fits a line, so
    the arm degrades into a shorter ladder rather than into a smaller pool.

    The allocation count is unchanged: one tensor set per rung either way.
    """
    order = {model: index for index, model in enumerate(design.models)}
    return tuple(sorted(design.rungs,
                        key=lambda r: (r.tiles, order.get(r.model, 0))))


def measure(design: Design, args, out_dir: Path, done: set[str]) -> tuple[list[dict], dict]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - needs the box
        raise CannotRunHere("torch is not importable in this interpreter") from exc
    if not torch.cuda.is_available():
        raise CannotRunHere(
            "no CUDA device. The plan, the prediction and the estimator's "
            "self-test above all ran; only the timings need the box.")

    from moe.bench.timing import L2Flusher, flush_mb_for_device, runtime_info

    # RESOLVED ONCE PER RUN, not per rung: the answer is a property of this box
    # and the calibration on it, and a per-rung lookup would read a yaml off
    # disk for every cell of a metered ladder. It refuses here, before the
    # kernel is built and before a byte is allocated, so a pod with no
    # calibration costs a compile rather than an hour.
    reference_clock_mhz = require_reference_clock(torch).mhz

    kernel = build_kernel()
    seen_kernels: set = set()
    meta = runtime_info()
    free_bytes = torch.cuda.mem_get_info()[0]
    flusher = L2Flusher(flush_mb_for_device() if args.l2_flush else 0)

    records: list[dict] = []
    path = out_dir / "cells.jsonl"
    for index, rung in enumerate(measurement_order(design)):
        if rung.key in done:
            continue
        need = rung.footprint_bytes + 512 * 2 ** 20
        if need > free_bytes:
            record = {"kind": "rung", "id": rung.key, "model": rung.model,
                      "tiles": rung.tiles, "ms": {}, "skipped": True,
                      "provenance": "measured",
                      "why": f"needs {need / 2**30:.1f} GiB, "
                             f"{free_bytes / 2**30:.1f} GiB free"}
            print(f"[alias] SKIP {rung.key}: {record['why']}")
            _append(path, record)
            records.append(record)
            continue
        record = measure_rung(kernel, rung, design, args, flusher,
                              args.seed + index, torch, seen_kernels,
                              reference_clock_mhz)
        # deepseek-v3's rung holds 16 GiB; the next model cannot be allocated
        # until the caching allocator gives it back.
        torch.cuda.empty_cache()
        _append(path, record)
        records.append(record)
        med = {k: (statistics.median(v) if v else float("nan"))
               for k, v in record["ms"].items()}
        print(f"[alias] {rung.key:44s} normal {med['normal']:9.4f} ms  "
              f"aliased {med['aliased']:9.4f} ms  "
              f"D {med['normal'] - med['aliased']:9.4f} ms")
    return records, meta


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A run killed mid-write leaves a partial last line. Dropping it is
            # right; failing the whole replay because of it is not.
            continue
        if row.get("kind") == "rung":
            out.append(row)
    return out


# --------------------------------------------------------------------------
# synthetic measurements, so the gates can be exercised without a GPU
# --------------------------------------------------------------------------

#: Each law exists to show one gate can bite. A pod script that always prints
#: PASS is worth nothing.
SYNTHETIC_LAWS = {
    "refit": "alpha = 0.558, L2 and HBM costs ADD; every gate should pass",
    "retracted": "alpha = 0.10, the value this repo retracted; P1 must FAIL",
    "tempo": "alpha = 0.33, TEMPO's value; P1 must FAIL",
    "folded": "alpha = 0.558 but the aliased kernel issued half the global "
              "loads; the ISA gate must FAIL and nothing may be quoted",
    "l2-step": "alpha is a step function of per-expert bytes against the "
               "MEASURED L2, which is P2's mechanism; the bands must separate",
    "noise": "alpha = 0.558 with a placebo drift as large as the signal; the "
             "placebo gate must FAIL",
    "max-model": "alpha = 0.558 but the kernel runs at max(L2, HBM), so the "
                 "DIFFERENCE estimator is biased down and only the bracket "
                 "still contains the truth",
    "l2-heavy": "alpha = 0.558 at max(L2, HBM) with r = 0.45, which is the H200 "
                "end of plausible; the bracket is too wide to pick a candidate "
                "and the run must say NOT TESTABLE rather than choose",
    "alias-free": "alpha = 0.558 and the ALIASED LADDER'S SLOPE IS ZERO: the "
                  "alias costs nothing per extra tile, so D(n) is the whole "
                  "weight cost and r is 0. The design working perfectly",
    "alias-blind": "alpha = 0.558 and the ALIASED LADDER'S SLOPE IS THE NORMAL "
                   "ONE'S: a shared non-DRAM ceiling at 0.61 of the read roof, "
                   "which is what 2026-09-01 measured. headroom, attribution "
                   "and bracket must all FAIL and no alpha may be quoted",
}

#: The pair the brief asks for, named together because their whole purpose is
#: to be each other's opposite: in one the aliased slope is ZERO and in the
#: other it is UNCHANGED from the normal ladder's, and a gate that cannot
#: separate those two cannot separate the hypothesis from its negation.
#: `tests/test_alias_ablation.py` runs both and asserts opposite exit codes.
SYNTHETIC_ALIAS_LAWS = ("alias-free", "alias-blind")

#: The shared non-DRAM ceiling `alias-blind` plants, as a fraction of the read
#: roof. 0.61 is MEASURED: the 2026-09-01 aliased ladders delivered 0.607,
#: 0.612, 0.613, 0.616 and 0.616 of this card's read roof on five geometries
#: spanning 20x in footprint.
BLIND_CEILING_FRACTION = 0.61

#: How much of the DRAM cost still leaks into the normal arm's clock when the
#: shared path is the slower one. 0.14 reproduces the 5 to 8% of a pass that
#: the run reported for D(1), which is what makes `alias-blind` the world that
#: actually happened rather than a caricature of it.
BLIND_OVERLAP_LEAK = 0.14

#: What fraction of the card's read roof the planted kernel delivers. Below 1
#: because no kernel reaches a STREAM ceiling, and stated because
#: `attribution_gate` divides a planted D(1) by that ceiling: at exactly 1.0 the
#: `l2-heavy` law would sit 0.05 above its own limit and the rehearsal would
#: turn on rounding.
SYNTHETIC_KERNEL_EFFICIENCY = 0.90

#: Laws that compose L2 and HBM service as `max` rather than as a sum. A
#: streaming kernel is closer to this end, which is why two of the eight are
#: here and why the report brackets rather than reporting one number.
SYNTHETIC_MAX_LAWS = ("max-model", "l2-heavy")

#: `r`: the aliased ladder's per-tile cost as a fraction of one weight read.
#: 0.12 for most laws so the shipped gates have something to resolve, and 0.45
#: for `l2-heavy`, which is what an L2 only about twice HBM's bandwidth would
#: give and is the reason the bracket exists at all.
SYNTHETIC_L2_RATIO = 0.12
SYNTHETIC_L2_RATIO_HEAVY = 0.45

#: The L2 the planted laws are stated against: THE H200's, read from its own
#: calibration rather than typed. It was `50 * 2**20` until 2026-09-03, which is
#: neither card's L2 (H200 60 MiB, A100 40 MiB) and put deepseek-v3's 56 MiB
#: expert above a line the real card puts it below. `or` a literal only so this
#: module imports in a tree with no calibration committed; a test pins the two
#: together so the fallback can never quietly become the value in use.
SYNTHETIC_L2_BYTES = measured_card(PLANT_CARD).get("l2_bytes") or 60 * 2 ** 20

#: The read roof the planted laws are stated against, same file, same reason.
SYNTHETIC_ROOF_BYTES_S = (measured_card(PLANT_CARD).get("roof_bytes_s")
                          or 4469.60368208941e9)


def synthesise(design: Design, law: str, seed: int,
               noise: float = 0.004) -> list[dict]:
    """Timings generated from a STATED law, so a verdict can be checked.

    The generator is the ablation's own physics and nothing more: a fixed cost,
    an activation and output stream that scales with the tile count, and a
    weight stream of `W (1 + alpha (n-1))` that the aliased variant does not
    pay. If the estimator cannot recover a planted alpha from that, it cannot
    recover a real one either.
    """
    if law not in SYNTHETIC_LAWS:
        raise ValueError(f"unknown law {law!r}; expected one of {sorted(SYNTHETIC_LAWS)}")
    rng = random.Random(seed)
    # THE ROOF, NOT A ROUND NUMBER. `attribution_gate` divides a measured D(1)
    # by `weight_bytes / roof`, so a generator that invents its own bandwidth
    # plants a world whose gates turn on the ratio between two literals.
    bandwidth = SYNTHETIC_ROOF_BYTES_S * SYNTHETIC_KERNEL_EFFICIENCY
    records = []
    for rung in design.rungs:
        if law in ("l2-heavy", "max-model"):
            alpha = REFIT_ALPHA
        elif law == "l2-step":
            alpha = 0.95 if rung.per_expert_bytes > SYNTHETIC_L2_BYTES else 0.05
        elif law == "retracted":
            alpha = REPO_RETRACTED_ALPHA
        elif law == "tempo":
            alpha = TEMPO_ALPHA
        else:
            alpha = REFIT_ALPHA
        if rung.control:
            alpha = 0.0
        w_ms = rung.weight_bytes / bandwidth * 1e3
        other = (rung.activation_bytes + rung.output_bytes) / bandwidth * 1e3
        fixed = 0.01
        # THE ALIASED LADDER IS NOT FREE, and pretending it was is what made an
        # earlier version of this generator unable to exercise the bracket at
        # all. Both variants issue n W bytes of loads; the aliased one hits L2
        # for all of them, at `r` of a weight read per tile.
        ratio = (SYNTHETIC_L2_RATIO_HEAVY if law == "l2-heavy"
                 else SYNTHETIC_L2_RATIO)
        if law == "alias-free":
            ratio = 0.0
        l2_cost = ratio * w_ms * rung.tiles
        hbm_cost = w_ms * (1.0 + alpha * (rung.tiles - 1))
        if law == "alias-blind":
            # THE WORLD OF 2026-09-01. A shared path SLOWER than DRAM sets the
            # clock in both arms, so the aliased ladder's slope is the normal
            # one's and the only thing the ablation removes is the residue that
            # failed to overlap. r comes out near 12, headroom near 0.61 and
            # D(1) at a seventh of the card's floor, which is what was measured.
            shared = (rung.weight_bytes * rung.tiles
                      / (SYNTHETIC_ROOF_BYTES_S * BLIND_CEILING_FRACTION) * 1e3)
            base_alias = fixed + other + shared
            base_normal = base_alias + BLIND_OVERLAP_LEAK * hbm_cost
        elif law in SYNTHETIC_MAX_LAWS:
            base_alias = fixed + other + l2_cost
            base_normal = fixed + other + max(l2_cost, hbm_cost)
        else:
            base_alias = fixed + other + l2_cost
            base_normal = base_alias + hbm_cost
        drift = 0.0
        if law == "noise":
            drift = 0.30 * w_ms

        def draw(mean, spread=noise, rng=rng):
            return [mean * (1.0 + rng.gauss(0.0, spread))
                    for _ in range(design.replicates)]

        counts_normal = {"ld.global": 0, "cp.async": 64, "st.global": 1,
                         "mma.sync": 0, "wgmma": 0, "ld.shared": 128,
                         "global_loads": 64}
        counts_alias = dict(counts_normal)
        if law == "folded":
            counts_alias["cp.async"] = 32
            counts_alias["global_loads"] = 32
        isa = [{"variant": "normal", "source": "synthetic", "digest": "aaaaaaaaaaaa",
                "counts": counts_normal},
               {"variant": "aliased", "source": "synthetic",
                "digest": "aaaaaaaaaaaa" if law != "folded" else "bbbbbbbbbbbb",
                "counts": counts_alias},
               {"variant": "constexpr-alias", "source": "synthetic",
                "digest": "cccccccccccc",
                "counts": dict(counts_normal, cp_async=1, global_loads=1)}]
        records.append({
            "kind": "rung", "id": rung.key, "model": rung.model,
            "tiles": rung.tiles, "block_m": rung.block_m,
            "rows_per_expert": rung.rows_per_expert, "experts": rung.experts,
            "k": rung.k, "n": rung.n,
            "per_expert_bytes": rung.per_expert_bytes,
            "weight_bytes": rung.weight_bytes,
            "activation_fraction": rung.activation_fraction,
            "programs": rung.programs, "control": rung.control,
            **cell_knobs(design),
            "ms": {"normal": draw(base_normal), "aliased": draw(base_alias),
                   "placebo": draw(base_normal + drift)},
            "correctness": {"normal": 1e-7, "aliased": 1e-7},
            "isa": isa, "sm_clock_start": 1500, "sm_clock_end": 1500,
            "clock_drift": 0.0, "throttled": False,
            "provenance": "synthetic", "law": law,
        })
    return records


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

class Report:
    """Everything printed, kept so it can also be written beside the data.

    A report that exists only in a terminal scrollback does not survive a pod
    teardown, and the whole point of the output directory is that the analysis
    leaves with the numbers.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def report_prediction(say) -> None:
    say("## the prediction, before anything runs")
    say()
    say(f"  P1  alpha = {REFIT_ALPHA:.3f}, 90% band "
        f"{REFIT_BAND[0]:.3f}-{REFIT_BAND[1]:.3f}  (today's refit, 10,813 rows)")
    say(f"      TEMPO arXiv:2608.13057 fits {TEMPO_ALPHA:.2f}; this repo's "
        f"retracted value is {REPO_RETRACTED_ALPHA:.2f}.")
    say(f"      This ablation runs at GROUP_SIZE_M=1, where the refit's own "
        f"split reads {REFIT_ALPHA_AT_GROUP_M_1:.3f},")
    say("      so that is the closer comparison and both are printed.")
    say()
    say("  P2  alpha is not a scalar: at GROUP_SIZE_M=1 the reuse distance is "
        "one pass over")
    say("      one expert's weight block, so alpha should track PER-EXPERT "
        "BYTES against L2.")
    say()
    say("  PASS/FAIL below is against P1's band, and the report names which of "
        "the three")
    say("  candidate values the measured band actually supports.")


#: Two-sided 5% at 80% power, the convention every MDE in this study is quoted
#: at, named rather than inlined so a reader can see nothing was chosen to make
#: a gate pass.
MDE_LEVEL = 0.05
MDE_POWER = 0.80

#: The run-to-run timing spread this design is sized against, as a fraction of
#: one pass. MEASURED, not assumed: the median and worst `timing_spread_median`
#: over the 26 published `*.report.json` files in `results/published`, which is
#: every arm in this repository that records one (min 0.0039, median 0.0077,
#: max 0.0182 on 2026-09-02). The convention this repo used to reach for was
#: 0.5%, which sits below the whole measured range.
MEASURED_SPREAD_MEDIAN = 0.0077
MEASURED_SPREAD_MAX = 0.0182


def mde_of_alpha(spread: float, replicates: int, top_tile: int,
                 alpha: float = REFIT_ALPHA,
                 signal_share: float = MIN_SIGNAL_FRACTION) -> float:
    """Smallest alpha this design could resolve, from a stated timing spread.

    B14: no arm in this study stated a minimum detectable effect, so a gate
    could pass or fail without anyone knowing whether the design could have
    resolved the difference either way. This one is derivable, so it is derived
    rather than asserted, in four steps a reader can check:

      1. ONE PASS AGAINST ANOTHER. Normal, aliased and placebo are interleaved
         inside every replicate on the same tensors, so the quantity is a paired
         fractional difference of two medians over `replicates` passes and each
         inherits the spread: `z * spread * sqrt(2 / replicates)`. Sigma is
         imported from the corpus rather than estimated inside the run, so this
         is a known-variance z test, the stricter of the two forms available at
         nine replicates.
      2. INTO D. D(n) is that difference, so the relative error on D is step 1
         divided by D's share of the pass. BEFORE THE RENTAL that share is
         `MIN_SIGNAL_FRACTION`, the floor `signal_gate` will admit and therefore
         the worst case the design accepts. AFTER IT the run has measured its
         own share and `signal_share` carries it, which is not a refinement but
         the difference between a readable report and the 2026-09-01 one: that
         run's D(1) was 5.3% of a pass, a fifth of the floor, so every quantity
         downstream of this division was five times worse than the plan's line
         said and no number in the report showed it.
      3. INTO THE RATIO. alpha comes from D(n)/D(1), and both are measured, so
         the ratio inherits both: another sqrt(2).
      4. INTO ALPHA. `alpha = (D(n)/D(1) - 1) / (n - 1)`, so a relative error on
         the ratio becomes `error * (1 + alpha (n-1)) / (n - 1)` on alpha. The
         top rung of the ladder is what sets it, which is why the ladder has one.

    The answer is compared with `MAX_BAND_WIDTH`, the width above which the
    interval cannot pick one of the three candidates apart, in `report_mde`.
    """
    if replicates < 1:
        raise ValueError(f"an MDE needs at least one replicate, got {replicates}")
    if spread <= 0:
        raise ValueError(f"an MDE needs a positive spread, got {spread}")
    if signal_share <= 0:
        raise ValueError(
            f"an MDE needs a positive signal share, got {signal_share}: D is "
            "divided by it, and a run whose ablation removed nothing has no "
            "resolution to quote rather than an infinite one")
    if top_tile < 2:
        raise ValueError(
            f"an MDE needs a rung above the first, got {top_tile}: alpha is the "
            "slope in (n-1) and one rung has no slope")
    normal = statistics.NormalDist()
    z = normal.inv_cdf(1.0 - MDE_LEVEL / 2.0) + normal.inv_cdf(MDE_POWER)
    per_pass = z * spread * math.sqrt(2.0 / replicates)
    rel_ratio = per_pass / signal_share * math.sqrt(2.0)
    return rel_ratio * (1.0 + alpha * (top_tile - 1)) / (top_tile - 1)


def observed_spreads(records: list[dict]) -> tuple[tuple[str, float], ...]:
    """The pass-to-pass spread THIS RUN delivered, median and worst.

    WHY THE PLAN'S MDE IS NOT THE WHOLE ANSWER, and this is the MDE's second
    call site. `MEASURED_SPREAD_MEDIAN` and `MEASURED_SPREAD_MAX` are
    `timing_spread_median` over the 26 published reports, and
    `rescore_published_reports.mde_report` states what that quantity is: each
    sweep's WITHIN-CELL pstdev over p50, "a WITHIN-process spread and therefore
    a floor". The MDE derivation needs the PASS-TO-PASS spread instead -- the
    scatter of the `replicates` p50s that `measure_rung` interleaves -- and that
    is a different number and never a smaller one. So the plan's line is a
    floor: honest before the box is rented, and useless after it. A run whose
    passes scattered at 4% prints a plan MDE of 0.04 and could not in fact have
    separated 0.33 from 0.558, and until this function existed nothing in the
    report said so. The plan says what the design hoped to see; this says what
    it saw.

    The reduction is the corpus's own, so the two numbers are comparable: a
    relative pstdev per pass group, then the median and the worst over every
    group. A group needs two passes to have a spread and a positive median to
    have a relative one; a run with neither returns `()` and the caller says
    NOT STATED rather than inventing a number.
    """
    rel: list[float] = []
    for row in records:
        if row.get("skipped"):
            continue
        for values in (row.get("ms") or {}).values():
            if len(values) < 2:
                continue
            mid = statistics.median(values)
            if mid > 0:
                rel.append(statistics.pstdev(values) / mid)
    if not rel:
        return ()
    return (("median", statistics.median(rel)), ("worst", max(rel)))


def report_mde(say, design: Design, spreads=None, measured: bool = False,
               signal_share: float | None = None) -> None:
    """The MDE line: what this design can resolve, at a stated timing spread.

    Printed TWICE by a run that measures anything, and the two are different
    statements. In the plan, `spreads` is None and the corpus floor is used,
    because the only noise this repository has before the rental is other arms'.
    In the report, `measured=True` and `spreads` comes from
    `observed_spreads(records)`, which is this run's own scatter. A design whose
    plan cleared `MAX_BAND_WIDTH` and whose run did not has not measured alpha,
    it has spent an hour, and the reader has to be able to see which happened
    without holding the two numbers in their head.

    `signal_share` is the other half of that and the half that decides it: the
    MDE divides by D's share of a pass, the plan can only use the floor
    `signal_gate` will admit, and the 2026-09-01 run's real share was a fifth of
    it. Passing the run's own worst share, the number `signal_gate` scores on,
    is what makes the second block a measurement rather than the first one
    repeated.

    `spreads` is also how the CANNOT-RESOLVE branch is planted off-GPU: on
    today's corpus this ladder clears `MAX_BAND_WIDTH` at both ends, and a
    branch nothing can reach is a branch nobody has read.
    """
    if spreads is None:
        spreads = (("median", MEASURED_SPREAD_MEDIAN),
                   ("worst", MEASURED_SPREAD_MAX))
    share = MIN_SIGNAL_FRACTION if signal_share is None else signal_share
    top = max(design.tiles)
    say()
    say("## what this run could see (MDE, on its own spread)" if measured
        else "## what this design can see (MDE)")
    say()
    say(f"Effect under test: the three candidates span "
        f"{REPO_RETRACTED_ALPHA} to {REFIT_ALPHA}, so an interval wider than "
        f"{MAX_BAND_WIDTH} cannot pick one.")
    if measured:
        say("Noise MEASURED IN THIS RUN: pass-to-pass pstdev over p50 across "
            "the interleaved replicates.")
        say("The plan's line above used the corpus's WITHIN-cell spread, which "
            "is a floor and not this.")
        say(f"Signal MEASURED IN THIS RUN: D(1) is {share:.1%} of the weakest "
            f"one-tile pass, the number")
        say(f"signal_gate scores, against the {MIN_SIGNAL_FRACTION:.0%} floor "
            f"the plan had to assume.")
    else:
        say(f"Noise assumption: per-pass timing spread MEASURED over the 26 "
            f"published reports; median {MEASURED_SPREAD_MEDIAN:.2%}, worst "
            f"{MEASURED_SPREAD_MAX:.2%}. Within-cell, so a FLOOR.")
    say(f"Design: {design.replicates} interleaved "
        f"{'replicate' if design.replicates == 1 else 'replicates'} per rung, "
        f"ladder topping out at {top} tiles"
        + ("." if measured else
           f", D(1) held above {MIN_SIGNAL_FRACTION:.0%} of a pass by the "
           "signal gate."))
    say()
    if not spreads:
        say("  NOT STATED. No pass group carried two replicates and a positive "
            "median, so this run")
        say("  recorded no spread of its own and its resolution cannot be "
            "judged. Raise --replicates.")
        return
    if share <= 0:
        say(f"  NOT STATED. D(1) is {share:.1%} of a pass, so the ablation "
            f"removed nothing measurable")
        say("  and there is no resolution to quote. `signal` below is the gate "
            "that says so.")
        return
    for label, spread in spreads:
        if spread <= 0:
            say(f"  at the {label:<6} spread 0.00%: NOT STATED. A zero spread "
                f"is an instrument that")
            say("          did not resolve its own passes, not a design that "
                "can see anything.")
            continue
        mde = mde_of_alpha(spread, design.replicates, top, signal_share=share)
        verdict = ("resolves the candidates" if mde <= MAX_BAND_WIDTH
                   else "CANNOT resolve them")
        say(f"  at the {label:<6} spread {spread:.2%}: MDE on alpha "
            f"{mde:.3f} against the {MAX_BAND_WIDTH} limit  {verdict}")
    say()
    if measured:
        say("An MDE above the limit does not make a PASS wrong; it makes a "
            "FAIL uninformative. This")
        say("ladder is already spent, so a CANNOT here means the interval "
            "below cannot pick a candidate,")
        # WHICH REMEDY, BY WHICH LEVER MOVED. Replicates buy resolution as
        # sqrt(n) and cost pod minutes; a thin signal is not noise and no
        # number of replicates fixes it. Printing "raise --replicates" over a
        # 5.3% share would send the next rental to buy nine more hours of the
        # same answer.
        if share < MIN_SIGNAL_FRACTION:
            say(f"and replicates are NOT the remedy: at a signal share of "
                f"{share:.1%} against the {MIN_SIGNAL_FRACTION:.0%} floor the "
                f"ablation")
            say("removed too little to resolve, and the lever is the alias and "
                "the pinning. See `headroom`.")
        else:
            say("and the lever is --replicates: the signal cleared its floor, "
                "so the scatter is what binds.")
    else:
        say("An MDE above the limit does not make a PASS wrong; it makes a "
            "FAIL uninformative, and")
        say("it is the number to raise --replicates against before the box is "
            "rented.")


def report_design(say, design: Design, gates: list[Gate], l2_bytes: int,
                  synthetic: bool = False) -> None:
    """The plan, and the preflight verdicts AS PROSE.

    NO `RESULT:` LINE IS PRINTED HERE, and that is the whole of the 2026-09-02
    fix to this function. A bare `alias_ablation.py` measures nothing and used
    to print four `RESULT: VALIDITY ... PASS` lines and exit 0, so
    `exit_codes.classify_text` recomputed DONE for a run that spent nothing --
    the shape a REFUSED log must never have. The preflight gates print their
    RESULT lines in `verdict`, which runs only on a page that has measurements
    on it; a run that stops before that prints none, `classify_text` raises
    `NoGatesScored`, and the process returns REFUSED to agree with it.
    """
    say()
    say("## the design")
    say()
    say(f"compute mode {design.compute}   BLOCK_M {design.block_m}   "
        f"tile {design.tile}   replicates {design.replicates}")
    if design.compute == "dot":
        say("*** dot mode is BIASED LOW and is not P1's answer. The aliased "
            "variant becomes")
        say("*** compute bound while the normal one stays memory bound, so "
            "D(n) loses one copy")
        say("*** of the per-tile compute cost and the fitted alpha falls "
            "toward zero.")
    say()
    say("  model                   E      K      N   MiB/expert  weight GiB  "
        "rows/expert   act frac")
    for model in design.models:
        for rung in design.rungs_for(model):
            say(f"  {model:20s} {rung.experts:4d} {rung.k:6d} {rung.n:6d} "
                f"{rung.per_expert_bytes / 2**20:11.1f} "
                f"{rung.weight_bytes / 2**30:11.2f} "
                f"{rung.rows_per_expert:12d} {rung.activation_fraction:10.4f}")
    if l2_bytes:
        say()
        where = ("the PLANTED threshold, not the attached card" if synthetic
                 else "the attached card")
        say(f"  L2 on {where}: {l2_bytes / 2**20:.1f} MiB. P2 says "
            "alpha is near 1 above it and near 0 below.")
    say()
    for gate in gates:
        say(f"  [{gate.label}] {gate.name}")
        say(f"          {gate.detail}")


def samples_from(records: list[dict]) -> dict[str, dict[int, dict[str, list[float]]]]:
    out: dict[str, dict[int, dict[str, list[float]]]] = {}
    for row in records:
        if row.get("skipped") or not row.get("ms"):
            continue
        out.setdefault(row["model"], {})[row["tiles"]] = {
            name: list(values) for name, values in row["ms"].items()}
    return out


def report_measurements(say, records: list[dict], design: Design) -> None:
    say()
    say("## the measurements")
    say()
    say("  model                  tiles   rows/e     normal    aliased          D"
        "    placebo")
    for model in design.models:
        for row in sorted((r for r in records if r.get("model") == model),
                          key=lambda r: r.get("tiles", 0)):
            if row.get("skipped"):
                say(f"  {model:20s} {row.get('tiles', 0):7d}   SKIPPED: "
                    f"{row.get('why', '')}")
                continue
            ms = row["ms"]
            normal = statistics.median(ms["normal"])
            aliased = statistics.median(ms["aliased"])
            placebo = statistics.median(ms["placebo"]) if ms.get("placebo") else float("nan")
            say(f"  {model:20s} {row['tiles']:7d} {row['rows_per_expert']:8d} "
                f"{normal:10.4f} {aliased:10.4f} {normal - aliased:10.4f} "
                f"{placebo - normal:10.4f}")
    say()
    say("  D is the ablation difference and is the HBM cost of that rung's "
        "weight reads.")
    say("  placebo is a SECOND normal launch minus the first: two identical "
        "configurations,")
    say("  so it is the noise floor D has to beat and nothing else.")


def report_isa(say, records: list[dict]) -> None:
    say()
    say("## the ISA check: did the aliased kernel really issue the loads?")
    say()
    say("  rung                          variant           PTX      "
        "ld.global   cp.async   global loads   mma.sync")
    seen = set()
    for row in records:
        for reading in row.get("isa", []):
            key = (reading["variant"], reading["digest"])
            if key in seen:
                continue
            seen.add(key)
            counts = reading["counts"]
            say(f"  {row['id']:28s}  {reading['variant']:16s} "
                f"{reading['digest']:>12s} {counts.get('ld.global', 0):10d} "
                f"{counts.get('cp.async', 0):10d} "
                f"{counts.get('global_loads', 0):14d} "
                f"{counts.get('mma.sync', 0):10d}")
    say()
    say("  ld.global ALONE would read zero here: Triton pipelines its "
        "global-to-shared copies")
    say("  as cp.async at num_stages > 1, so the gate is on the SUM. normal and "
        "aliased are")
    say("  the same compiled kernel driven by three runtime scalars, so equal "
        "counts are a")
    say("  property of the design; constexpr-alias is the naive form and is "
        "shown folding or not.")


@dataclass
class ModelResult:
    model: str
    fit: AlphaFit
    band: tuple[float, float] | None
    per_expert_mib: float
    control: bool


def analyse_models(design: Design, records: list[dict], draws: int,
                   seed: int) -> list[ModelResult]:
    per_model = samples_from(records)
    sizes = {r["model"]: r.get("per_expert_bytes", 0) for r in records}
    controls = {r["model"]: bool(r.get("control")) for r in records}
    out = []
    for model in design.models:
        samples = per_model.get(model)
        if not samples:
            continue
        fit = fit_bracket(samples)
        band = bootstrap_alpha(samples, draws, seed) if fit.ok else None
        out.append(ModelResult(model=model, fit=fit, band=band,
                               per_expert_mib=sizes.get(model, 0) / 2 ** 20,
                               control=controls.get(model, False)))
    return out


def report_alphas(say, results: list[ModelResult], pooled: tuple[float, float] | None,
                  pooled_bracket: tuple[float, float] | None) -> None:
    say()
    say("## alpha, from the ablation alone, as a bracket")
    say()
    say("  Two estimators, because exactly one of them is right and which one "
        "depends on how")
    say("  L2 service and HBM service compose. DIFFERENCE subtracts the aliased "
        "ladder and is")
    say("  exact if the two costs ADD; DIRECT fits the normal ladder with the "
        "fixed cost taken")
    say("  from the aliased ladder's own n=0 intercept and is exact if the "
        "kernel runs at")
    say("  max(L2, HBM). For any alpha <= 1 they BRACKET the truth, and the "
        "bracket is narrow")
    say("  exactly when r, the aliased ladder's per-tile cost over one weight "
        "read, is small.")
    say()
    say("  model                  MiB/expert    W (ms)   fixed  difference  "
        "direct       r      R^2")
    for res in results:
        fit = res.fit
        if not fit.ok:
            say(f"  {res.model:20s} {res.per_expert_mib:11.1f}   NOT FITTED: "
                f"{fit.why}")
            continue
        direct = "none" if fit.alpha_direct is None else f"{fit.alpha_direct:.3f}"
        share = "none" if fit.l2_share is None else f"{fit.l2_share:.3f}"
        tag = "  (L2-RESIDENT CONTROL)" if res.control else ""
        say(f"  {res.model:20s} {res.per_expert_mib:11.1f} {fit.w_ms:9.4f} "
            f"{(fit.fixed_ms or 0.0):7.4f} {fit.alpha:11.3f} {direct:>7s} "
            f"{share:>7s} {fit.r2:8.4f}{tag}")
    say()
    say("  model                  bracket           90% interval        "
        "per-rung alphas (difference)")
    for res in results:
        fit = res.fit
        if not fit.ok:
            continue
        span = fit.bracket
        band = (f"{res.band[0]:.3f} to {res.band[1]:.3f}" if res.band else "none")
        rungs = "  ".join(f"n={n}:{a:.3f}" for n, a in sorted(fit.per_rung.items()))
        say(f"  {res.model:20s} {span[0]:.3f} to {span[1]:.3f}   "
            f"{band:>18s}   {rungs}")
    say()
    say("  W is the measured time of ONE full pass over every expert's weight "
        "block, read off")
    say("  the n=1 rung. Both alphas are a fitted slope over that intercept, so "
        "every unit of")
    say("  time cancels and no bandwidth, byte count or ridge enters either "
        "number.")
    if pooled_bracket is not None:
        say()
        say(f"  POOLED (median over the non-control models): alpha is in "
            f"{pooled_bracket[0]:.3f} to {pooled_bracket[1]:.3f}"
            + (f", 90% interval {pooled[0]:.3f} to {pooled[1]:.3f}"
               if pooled else ", no interval"))
        say(f"  against the refit's {REFIT_ALPHA:.3f} "
            f"({REFIT_BAND[0]:.3f}-{REFIT_BAND[1]:.3f}) pooled and "
            f"{REFIT_ALPHA_AT_GROUP_M_1:.3f} at GROUP_SIZE_M=1.")


def report_confounds(say, design: Design, results: list[ModelResult]) -> None:
    say()
    say("## what else differs between aliased and normal")
    say()
    worst = max((r.activation_fraction for r in design.rungs), default=0.0)
    say("  BOUNDED. L2 capacity. The aliased variant leaves L2 to the "
        "activations, whose stream")
    say(f"  is {worst * 100:.2f}% of the weight stream at the worst rung, so "
        "the most the freed")
    say("  capacity can be worth is that fraction of D. The output write is "
        "identical in both.")
    say()
    control = next((r for r in results if r.control), None)
    if control is not None and control.fit.ok:
        say("  BOUNDED BY MEASUREMENT. TLB, page behaviour and code path. The "
            "L2-resident control")
        say("  ran the same ladder on a geometry whose PER-EXPERT block fits "
            "in L2, so NORMAL")
        say(f"  has no HBM re-read to save. Its W is {control.fit.w_ms:.4f} ms "
            f"and its alpha brackets "
            f"{control.fit.bracket[0]:.3f} to {control.fit.bracket[1]:.3f}.")
        say("  Whatever survives there is not weight traffic.")
    else:
        say("  NOT BOUNDED. TLB, page behaviour and code path. The aliased "
            "variant touches one")
        say("  page of B where the normal one touches the whole tensor, and "
            "this run has no")
        say("  L2-resident control to bound it. Re-run with --control.")
    say()
    widest = max((r.alias_bytes(design.alias_extent) for r in design.rungs),
                 default=0)
    say("  BOUNDED BY THE BRACKET, AND GATED. L2 service, and the cache-set "
        "distribution that")
    say("  makes it worse. Both variants push n W bytes through L2. At "
        f"--alias-extent {design.alias_extent!r} the aliased arm touches")
    if design.alias_extent == "tile":
        say(f"  ONE {design.tile['BLOCK_K']} x {design.tile['BLOCK_N']} tile, "
            f"{widest / 2**10:.0f} KiB, re-read every K iteration. That is the "
            "2026-09-01 pinning and its")
        say("  slice contention is what drove r to 6.4-19.1; the numbers below "
            "are here to be")
        say("  compared with an --alias-extent block run, not to be quoted "
            "alone.")
    else:
        say(f"  one BLOCK_N x K column block, {widest / 2**20:.2f} MiB, walked "
            "with the normal arm's own")
        say("  sequential K stride over thousands of lines rather than pinned "
            "to a handful of")
        say("  slices. Whatever it still costs shows up in the ALIASED ladder's "
            "own slope, r, and")
        say(f"  `bracket` FAILS the page if r exceeds {MAX_BRACKET_R}, above "
            "which the two estimators")
        say("  land on the same side of alpha and the interval between them is "
            "not a bracket.")
    say()
    say("  ABSORBED BY CONSTRUCTION. Any cost that scales with the extra-tile "
        "count and is not a")
    say("  weight re-read lands in alpha, because (n-1) is the regressor. That "
        "is a property of")
    say("  the estimator and no control can remove it.")


# --------------------------------------------------------------------------
# gates on the result
# --------------------------------------------------------------------------

def _aliased_slope_bytes_s(row_group: list[dict]) -> float | None:
    """One model's aliased ladder as an achieved REQUEST bandwidth, bytes/s.

    The regressor is `tiles - 1` and the response is the aliased median, so the
    slope is the cost of one extra M-tile per expert. One extra M-tile requests
    exactly one more pass over every expert's weight block, `weight_bytes`,
    whatever the alias then does with the addresses. Bytes over seconds is
    therefore the rate at which the SHARED, non-DRAM path in this kernel can
    deliver requests, and it is the number the whole headroom question is about.
    """
    pts = []
    weight_bytes = 0
    for row in row_group:
        ms = row.get("ms") or {}
        if not ms.get("aliased"):
            continue
        pts.append((row["tiles"], statistics.median(ms["aliased"])))
        weight_bytes = max(weight_bytes, int(row.get("weight_bytes") or 0))
    line = _fit_line(sorted(pts))
    if line is None or line[1] <= 0 or not weight_bytes:
        return None
    return weight_bytes / (line[1] * 1e-3)


def headroom_gate(records: list[dict], roof_bytes_s: float | None) -> Gate:
    """Was DRAM the binding resource at all? The 2026-09-01 run's real defect.

    THE ABLATION CAN ONLY SEE A RESOURCE THAT BINDS. Both arms issue the same
    loads and both pay the same non-DRAM cost of getting them from L2 into the
    SM; only the normal arm additionally pays DRAM. If the shared path is
    SLOWER than DRAM, the normal arm is limited by the shared path too, DRAM has
    slack, and removing every byte of it moves the clock by the un-overlapped
    residue and by nothing else. The measured alpha is then a fact about the
    residue, and it is small for reasons that have nothing to do with alpha.

    That is not hypothetical. The 2026-09-01 run's aliased ladder delivered
    0.607 to 0.616 of this card's read roof on five geometries spanning 20x in
    footprint, so the ceiling was shape independent, non-DRAM, and 39% below
    the roof. `signal` reported the consequence (D(1) at 5.3% of a pass) and
    `form` reported the consequence of the consequence (R^2 0.42 on the model
    with the smallest D). Neither named the cause, so the page read as evidence
    that the per-tile cost is not DRAM. It was evidence that the apparatus
    could not have seen DRAM.

    UNKNOWN, NOT PASS, WITHOUT A ROOF. This is a VALIDITY gate, so UNKNOWN is
    INVALID and nothing is quotable: a card with no committed calibration has
    no ceiling to compare against and this gate will not invent one.
    """
    name = "headroom: the shared path is faster than DRAM, so DRAM could bind"
    if not roof_bytes_s:
        return Gate(name, None,
                    "no committed calibration for this card, so there is no "
                    "measured DRAM roof to compare the aliased ladder against. "
                    "Run scripts/calibrate_hardware.py --publish first; without "
                    "it nothing on this page says whether DRAM ever bound")
    by_model: dict[str, list[dict]] = {}
    for row in records:
        if row.get("control") or not (row.get("ms") or {}).get("aliased"):
            continue
        by_model.setdefault(row["model"], []).append(row)
    ratios = []
    for model, rows in by_model.items():
        achieved = _aliased_slope_bytes_s(rows)
        if achieved:
            ratios.append((achieved / roof_bytes_s, model, achieved))
    if not ratios:
        return Gate(name, None, "no model produced an aliased ladder slope")
    ratio, model, achieved = min(ratios)
    return Gate(name, ratio >= MIN_HEADROOM_RATIO,
                f"weakest model {model}: the aliased ladder delivers "
                f"{achieved / 1e9:.0f} GB/s of weight requests against a "
                f"measured read roof of {roof_bytes_s / 1e9:.0f} GB/s, a ratio "
                f"of {ratio:.3f} (limit {MIN_HEADROOM_RATIO}). Below 1 the "
                "shared non-DRAM path is slower than DRAM, so DRAM had slack "
                "in the normal arm and no ablation of it can move the clock "
                "however large alpha is")


def attribution_gate(records: list[dict], roof_bytes_s: float | None) -> Gate:
    """Is D(1) big enough to BE the read it is named after?

    D(1) is the whole denominator of this experiment: every alpha is a slope
    over it, and the design's claim is that it is the time cost of one full
    pass over every expert's weight block. That pass has a floor, and the floor
    is the card's own measured read ceiling. A D(1) below it is a read that beat
    the card, which is not a small effect or a noisy one, it is impossible, and
    it means the label on D is wrong.

    On 2026-09-01 every one of the five geometries came in between 3.1x and
    10.7x BELOW its own floor. This gate states that in the units the claim is
    made in, which `signal` (a fraction of the pass's own time) could not.
    """
    name = "attribution: D(1) reaches the card's floor for the bytes it removes"
    if not roof_bytes_s:
        return Gate(name, None,
                    "no committed calibration for this card, so there is no "
                    "floor for the bytes the alias removes. "
                    "scripts/calibrate_hardware.py --publish writes one")
    worst = None
    for row in records:
        if row.get("control") or row.get("tiles") != 1:
            continue
        ms = row.get("ms") or {}
        weight_bytes = int(row.get("weight_bytes") or 0)
        if not ms.get("normal") or not ms.get("aliased") or not weight_bytes:
            continue
        d1 = statistics.median(ms["normal"]) - statistics.median(ms["aliased"])
        floor_ms = weight_bytes / roof_bytes_s * 1e3
        ratio = d1 / floor_ms
        if worst is None or ratio < worst[0]:
            worst = (ratio, row["id"], d1, floor_ms)
    if worst is None:
        return Gate(name, None, "no one-tile rung carried both variants")
    ratio, rid, d1, floor_ms = worst
    return Gate(name, ratio >= MIN_ATTRIBUTION_RATIO,
                f"weakest one-tile rung {rid}: D(1) is {d1:.4f} ms against a "
                f"floor of {floor_ms:.4f} ms for the same bytes at the card's "
                f"measured read roof, a ratio of {ratio:.3f} (limit "
                f"{MIN_ATTRIBUTION_RATIO}). Below 1 the difference is smaller "
                "than the fastest this card can move those bytes, so it is not "
                "the cost of moving them")


def bracket_gate(results: list[ModelResult]) -> Gate:
    """Is the interval between the two estimators a BRACKET at all?

    The DIFFERENCE estimator returns `(alpha - r)/(1 - r)` under max
    composition. That is below alpha only while `r < 1`. At `r > 1` the
    denominator is negative, the estimator sits ABOVE alpha, both ends of the
    printed interval are on the same side of the truth, and the interval
    contains nothing in particular while still looking like a measurement with
    error bars. The 2026-09-01 report printed exactly that: brackets built on
    an `r` of 6.4 to 19.1, one of which (0.535 to 0.969 pooled) was then read as
    containing the refit.

    This is a VALIDITY gate and not a resolution one. `band_gate` asks whether a
    valid interval is narrow enough to choose between candidates; this asks
    whether it is an interval at all, and a wide answer to the first question is
    NOT TESTABLE while a failure of the second is INVALID.
    """
    name = "bracket: r < 1, so the two estimators sit on opposite sides"
    shares = [(r.fit.l2_share, r.model) for r in results
              if r.fit.ok and not r.control and r.fit.l2_share is not None]
    if not shares:
        return Gate(name, None,
                    "no model produced an r; without it nothing says which "
                    "side of alpha either estimator is on")
    worst, model = max(shares)
    return Gate(name, worst <= MAX_BRACKET_R,
                f"worst r is {worst:.3f} on {model} (limit {MAX_BRACKET_R}). r "
                "is the aliased ladder's per-tile cost over one weight read, "
                "and the difference estimator is (alpha - r)/(1 - r): above 1 "
                "that denominator changes sign, both ends land on the same "
                "side of alpha, and the printed interval is not a bracket")


def probe_gate(readings: list[dict], chosen: dict | None, why: str) -> Gate:
    """The probe's verdict, as the one RESULT line a stopped run prints.

    A probe that finds no pinning STOPS the arm, and it stops it after
    measuring, which is INVALID and not REFUSED: the probe's own wall minutes
    were spent (`report_cost` prices them and is the only place that names a
    duration) and there is a directory of probe cells on disk that must not be
    scored as an answer. `exit_codes.classify_text` has to be able to
    recompute that 3 from the log, so the probe prints this gate's RESULT line
    through `verdict` exactly as the ladder's gates do. Without it the log would
    carry no RESULT line at all, which is the REFUSED shape, and the driver
    would read a spent arm as a free one.
    """
    return Gate("probe: some pinning delivers requests faster than DRAM",
                chosen is not None,
                f"{len(readings)} pinning(s) timed. {why}",
                kind=exit_codes.VALIDITY)


def correctness_gate(records: list[dict], compute: str) -> Gate:
    checked = [(r["id"], name, value)
               for r in records
               for name, value in (r.get("correctness") or {}).items()
               if value is not None]
    if not checked:
        return Gate("correctness: each variant reproduced its closed form", None,
                    "no rung recorded a correctness check")
    bad = [(rid, name, value) for rid, name, value in checked
           if not (value == value) or value > CORRECTNESS_RTOL]
    detail = (f"{len(checked)} checks at relative RMS <= {CORRECTNESS_RTOL:.0e}; "
              f"worst {max(v for _, _, v in checked):.2e}")
    if compute == "dot":
        detail += (". In dot mode the ALIASED side has no cheap closed form and "
                   "is UNCHECKED, so nothing here says the aliasing took "
                   "effect; the normal side is checked against a real matmul on "
                   "the first, middle and last expert")
    if bad:
        detail += (f". {len(bad)} failed, first {bad[0][0]} {bad[0][1]} at "
                   f"{bad[0][2]:.2e}. A failing NORMAL means it did not read "
                   "every expert's block; a failing ALIASED means the three "
                   "runtime scalars did nothing and D is noise")
    return Gate("correctness: each variant reproduced its closed form",
                not bad, detail)


#: The LEVEL gate's name, in one place because the tests, the RESULT token and
#: the paragraph above the table all have to name the same gate. Short on
#: purpose: `Gate.token` slugs the WHOLE name and then truncates at 56
#: characters, so a longer sentence here ships a RESULT token that stops mid
#: word with a trailing hyphen, and a driver keying on the token would be
#: keying on where the sentence happened to fall.
LEVEL_GATE = "level: every rung ran at the roof's measured clock"


def level_split(records: list[dict]) -> tuple[list[str], list[str]]:
    """The rungs that sagged and the rungs that carry no reference, in that
    order, from ONE place.

    TWO READERS NEED THE SAME TWO COUNTS, which is the shape
    `one_tile_signal_shares` exists in for the same reason: `_analyse` prints
    them as prose and `level_gate` scores them, and a page that narrates three
    sagged rungs while the gate scores a different three is the defect this
    rebuild keeps finding one call site at a time. `None` here means the column
    is ABSENT or null: a row written before there was a reference to sag
    against is unknown, never bad, and an exclusion has to be positively
    established (`bm128_roofline.Timing.throttled` says the same thing).
    """
    sagged = [r["id"] for r in records if r.get("clock_level_ok") is False]
    blind = [r["id"] for r in records if r.get("clock_level_ok") is None]
    return sagged, blind


def level_gate(records: list[dict]) -> Gate:
    """Did this card sit at the clock its roof was measured at, on every rung.

    THIS ARM SCORES LEVEL WHERE ITS SIBLINGS NARRATE OR EXCLUDE IT, and the
    estimator is the reason. `group_m_alpha_sweep` prints the same two counts as
    prose; `bm128_roofline` drops the row (`Timing.cold` feeds `excluded`, and
    the fit skips it). Neither answer is available here. The whole result is
    D(n) = T_normal(n) - T_aliased(n), a difference of two ladders, and alpha is
    the slope of that difference over its intercept: a sag part way up one
    ladder moves D(n) and D(1) by different amounts, so it does not cancel, and
    dropping the sagged rungs silently re-shapes the very ladder alpha is fitted
    from. The only honest move left is to refuse the run and name the rungs to
    re-measure.

    IT WAS A PARAGRAPH UNTIL 2026-09-03, AND A PARAGRAPH MOVES NOTHING THE
    SESSION GRADES. `pod_session.sh` scores this arm on exactly two things: the
    exit code (SBa) and the count of `[PASS]`/`[FAIL]` lines in the log (SBb,
    via `gate_from_log`). The LEVEL block moved neither, so a directory with
    three sagged rungs printed its warning, fed all twenty rungs to the fit, and
    exited `0 DONE: every VALIDITY and CLAIM gate PASSED`. The session graded a
    sagged hour green and only a human reading the prose could have caught it.

    NOT TESTABLE, NOT PASS, WHEN A RUNG CARRIES NO REFERENCE. `classify` turns
    an unknown VALIDITY gate into INVALID, which is the right reading: a run
    that could not be scored against a clock has not shown the card was level,
    and "a check that examined nothing reports zero failures" is this project's
    first named failure mode. That is also why a partly-blind ladder cannot
    PASS on the rungs that happen to carry the key.

    Planted rows carry no clock, so `_analyse` scores this gate only on a
    measured run; a synthetic pass would otherwise be INVALID for want of
    hardware it never touched.
    """
    sagged, blind = level_split(records)
    limit = f"{timing.LEVEL_FRACTION:.0%}"
    if sagged:
        detail = (f"{len(sagged)} of {len(records)} rungs ran below {limit} of "
                  f"the clock this card's roof was measured at, first "
                  f"{sagged[0]}. alpha is a slope over an intercept of the same "
                  "difference, so a sag part way up one ladder does not cancel")
        if blind:
            detail += (f", and a further {len(blind)} carry no reference at all")
        return Gate(LEVEL_GATE, False, detail)
    if blind:
        return Gate(LEVEL_GATE, None,
                    f"{len(blind)} of {len(records)} rungs carry no reference "
                    "clock, so they could not be scored against the one this "
                    "card's roof was measured at. Publish a calibration for "
                    "this card and re-measure them; nothing here says they were "
                    "level")
    return Gate(LEVEL_GATE, True,
                f"all {len(records)} rungs ran within {limit} of the clock this "
                "card's roof was measured at")


def placebo_gate(records: list[dict]) -> Gate:
    """Two launches of an identical configuration, against the signal.

    THE CONTROL RUNG IS EXCLUDED and that is not a convenience. Its D is near
    zero by construction, so the ratio drift/D is a ratio of two noise floors
    and reads as a catastrophic failure on a perfectly clean run -- observed at
    542% on the first synthetic pass of this gate. The question the gate asks is
    whether the noise floor is small against the signal that alpha is FITTED
    from, and no alpha is fitted from the control.
    """
    worst = None
    for row in records:
        ms = row.get("ms") or {}
        if row.get("control"):
            continue
        if not ms.get("normal") or not ms.get("placebo") or not ms.get("aliased"):
            continue
        normal = statistics.median(ms["normal"])
        diff = normal - statistics.median(ms["aliased"])
        drift = abs(statistics.median(ms["placebo"]) - normal)
        if diff <= 0:
            continue
        ratio = drift / diff
        if worst is None or ratio > worst[0]:
            worst = (ratio, row["id"], drift, diff)
    if worst is None:
        return Gate("placebo: two identical launches differ by far less than D",
                    None, "no rung has both a placebo pass and a positive D")
    ratio, rid, drift, diff = worst
    return Gate("placebo: two identical launches differ by far less than D",
                ratio <= PLACEBO_MAX_FRACTION,
                f"worst rung {rid}: two identical configurations differ by "
                f"{drift:.4f} ms against a D of {diff:.4f} ms, "
                f"{ratio * 100:.1f}% (limit {PLACEBO_MAX_FRACTION * 100:.0f}%)")


def one_tile_signal_shares(records: list[dict]) -> list[tuple[float, str]]:
    """D(1) as a fraction of the one-tile pass, per real model, worst first.

    ONE FUNCTION BECAUSE TWO READERS NEED THE SAME NUMBER. `signal_gate` scores
    the worst of these against `MIN_SIGNAL_FRACTION`, and `report_mde` divides
    by it to say what the run could resolve. Computed twice they drift, and a
    report that gates on 5.3% while pricing its resolution at 25% is the exact
    shape of the defect this rebuild keeps finding: the fix applied at one of
    two call sites.

    The control is excluded because its D is zero BY DESIGN -- its expert fits
    in L2, so there is no HBM re-read to remove -- and scoring the signal on it
    would fail every honest run.
    """
    fractions = []
    for row in records:
        if row.get("control") or row.get("tiles") != 1:
            continue
        ms = row.get("ms") or {}
        if not ms.get("normal") or not ms.get("aliased"):
            continue
        normal = statistics.median(ms["normal"])
        if normal <= 0:
            continue
        fractions.append(((normal - statistics.median(ms["aliased"])) / normal,
                          row["id"]))
    return sorted(fractions)


def signal_gate(records: list[dict]) -> Gate:
    fractions = one_tile_signal_shares(records)
    if not fractions:
        return Gate("signal: the weight read is most of what the kernel does",
                    None, "no one-tile rung was timed")
    worst, rid = fractions[0]
    return Gate("signal: the weight read is most of what the kernel does",
                worst >= MIN_SIGNAL_FRACTION,
                f"weakest one-tile rung {rid} has D(1) at {worst * 100:.1f}% of "
                f"its own time (limit {MIN_SIGNAL_FRACTION * 100:.0f}%). Below "
                "that the difference is measuring something the weight-read "
                "label does not cover")


def linearity_gate(results: list[ModelResult]) -> Gate:
    fitted = [r for r in results if r.fit.ok and not r.control]
    if not fitted:
        return Gate("form: D(n) is affine in (n-1), as W(1+alpha(n-1)) requires",
                    None, "no model produced a fit")
    worst = min(fitted, key=lambda r: r.fit.r2)
    return Gate("form: D(n) is affine in (n-1), as W(1+alpha(n-1)) requires",
                worst.fit.r2 >= MIN_LINEARITY_R2,
                f"worst R^2 {worst.fit.r2:.4f} on {worst.model} "
                f"(limit {MIN_LINEARITY_R2}). Below it the functional form is "
                "wrong and slope-over-intercept is not alpha")


def control_gate(results: list[ModelResult]) -> Gate:
    """The control bounds everything that scales with the tile count and is not
    weight traffic.

    IT IS THE CONTROL'S ALPHA THAT MATTERS, NOT ITS W. An earlier version of
    this gate compared the control's W against the real models' and would have
    failed on a correct run: the control still pays a full first pass over its
    512 MiB tensor, so its W is legitimately the same order as theirs. What the
    control does NOT have is an HBM re-read, because 16 MiB per expert fits in
    L2, so the tile-count SLOPE is where its emptiness has to show up. If TLB
    pressure, page behaviour or a code-path difference were driving D(n), they
    would drive it here too and the control's alpha would not be near zero.
    """
    name = "control: an L2-resident expert shows no extra-tile cost"
    control = next((r for r in results if r.control and r.fit.ok), None)
    real = [r for r in results if not r.control and r.fit.ok]
    if control is None or not real:
        return Gate(name, None,
                    "no L2-resident control in this run; re-run with --control. "
                    "Without it, TLB and page behaviour are named but unbounded")
    smallest = min(r.fit.bracket[1] for r in real)
    span = control.fit.bracket
    # CONSISTENT WITH ZERO, not "small in the difference estimator". The
    # control carries the same L2-versus-HBM ambiguity as everything else, so
    # its difference estimator reads -r/(1-r) and its direct one r/(1+r) for a
    # true alpha of zero. Gating either end alone would fail a clean control.
    ok = span[0] <= CONTROL_MAX_ALPHA and span[1] >= -CONTROL_MAX_ALPHA
    return Gate(name, ok,
                f"the control's bracket is {span[0]:.3f} to {span[1]:.3f} on a "
                f"W of {control.fit.w_ms:.4f} ms, against a top end of "
                f"{smallest:.3f} for the weakest real model. It must be "
                f"consistent with zero to within {CONTROL_MAX_ALPHA}: its "
                "re-reads hit L2, so anything it does show is the size of an "
                "extra-tile cost that is not weight traffic")


def band_gate(pooled: tuple[float, float] | None,
              l2_shares: list[float]) -> Gate:
    """Can the reported interval pick one of the three candidates at all?

    NOT TESTABLE rather than FAIL when it cannot. A wide interval is not a
    refutation of anything, it is a measurement that did not resolve, and the
    difference matters: this project has a standing habit of reading a wide
    number as a finding. The detail names WHY it is wide, because r is
    measured and is the only thing that sets it.
    """
    name = "resolution: the interval can separate the three candidates"
    if pooled is None:
        return Gate(name, None, "no pooled interval was produced")
    width = pooled[1] - pooled[0]
    detail = (f"90% interval is {width:.3f} wide (limit {MAX_BAND_WIDTH}). The "
              f"candidates span {REPO_RETRACTED_ALPHA} to {REFIT_ALPHA}, and an "
              "interval wider than half the largest gap between adjacent "
              "candidates cannot pick one")
    if l2_shares:
        detail += (f". r, the aliased ladder's per-tile cost over one weight "
                   f"read, is {min(l2_shares):.3f} to {max(l2_shares):.3f}, and "
                   "r is what sets the bracket's width: the two estimators "
                   f"differ by roughly 2r/(1-r^2), and `bracket` voids the page "
                   f"above r = {MAX_BRACKET_R}. Narrowing it means a cheaper "
                   "aliased ladder: --alias-extent block spreads the alias over "
                   "a whole BLOCK_N x K column block instead of one tile, and "
                   "--probe finds the pinning that delivers requests fastest")
    # NOT TESTABLE, not FAIL. A wide interval refutes nothing; it says the run
    # did not resolve, and this project has a standing habit of reading a wide
    # number as a finding.
    return Gate(name, True if width <= MAX_BAND_WIDTH else None, detail)


def prediction_gate(pooled: tuple[float, float] | None,
                    pooled_bracket: tuple[float, float] | None,
                    compute: str) -> Gate:
    name = f"P1: the ablation agrees with the refit, alpha = {REFIT_ALPHA}"
    if pooled is None or pooled_bracket is None:
        return Gate(name, None, "no pooled interval; nothing to score")
    if compute == "dot":
        # THE FIRST WORDS ARE THE ONES THAT SURVIVE. `Gate.result_line` cuts the
        # detail at 160 characters, and the RESULT line is the ONE line the
        # driver may grep. This detail used to open with the interval and reach
        # "it is a lower bound, not P1's answer" at character 190, so the only
        # sentence that separates "the world disagreed" from "the question was
        # not asked" was the one the cut removed, on the gate whose UNKNOWN is
        # what makes the process exit 1.
        return Gate(name, None,
                    "NOT A REFUTATION: dot mode cannot answer P1 at all, so "
                    f"this exit 1 means the question was not asked. It puts "
                    f"alpha at {pooled[0]:.3f} or above, a LOWER BOUND biased "
                    "LOW by one copy of the per-tile compute cost on top of "
                    "everything the bracket already carries "
                    f"(upper end {pooled[1]:.3f}). Re-run in sum mode")
    overlap = pooled[0] <= REFIT_BAND[1] and REFIT_BAND[0] <= pooled[1]
    _, sentence = supported_candidate(pooled)
    return Gate(name, overlap,
                f"measured alpha is in {pooled_bracket[0]:.3f} to "
                f"{pooled_bracket[1]:.3f} before noise and {pooled[0]:.3f} to "
                f"{pooled[1]:.3f} after it, against the refit's "
                f"{REFIT_BAND[0]:.3f}-{REFIT_BAND[1]:.3f}: "
                f"{'they overlap' if overlap else 'they are DISJOINT'}. "
                f"{sentence}")


# The sentence that keeps exit 1 from being read as a refutation, in the two
# places a reader meets it: beside the probe's re-pin and beside the verdict.
# ONE list and not two copies, because two copies of a disambiguation is the
# shape that leaves the second one saying the old thing.
def dot_mode_reading(compute: str) -> list[str]:
    """How to read this run's exit code when the ladder ran in `dot` mode.

    `exit_codes` has five states and none of them is "measured, sound, and the
    claim was not askable". `dot` mode is that state: every VALIDITY gate can
    pass, the numbers are real, and P1 is UNKNOWN because the estimator is
    biased low by construction. `classify` maps an UNKNOWN CLAIM to CLAIM_FAIL,
    which is right by the table -- the claim was not established -- and the
    ledger will latch it and never retry. What it is NOT is evidence that alpha
    differs from the refit, and the difference between those two readings is a
    retraction. So the run says which one it is, in words, next to the number.

    Returns an empty list in `sum` mode, where exit 1 means exactly what the
    table says it means and an extra paragraph would only dilute it.
    """
    if compute != "dot":
        return []
    return [
        "READ THIS RUN AS A LOWER BOUND, NOT AS A REFUTATION.",
        "",
        "P1 is UNKNOWN, not FAIL. dot mode moves the reduction onto the tensor",
        "cores, which is what buys the headroom the sum kernel could not reach,",
        "and it costs the estimator one copy of the per-tile compute cost. The",
        "interval it produces is a bound BELOW alpha and it is not alpha, so no",
        "candidate can be refuted from this page and none is.",
        "",
        "The process exits 1 CLAIM_FAIL because exit_codes.classify counts an",
        "UNKNOWN CLAIM against the gate, which is the right rule: DONE requires",
        "every gate to say PASS in so many words. But the ledger will latch that",
        "1 and not retry, so the arm is FINISHED WITHOUT AN ANSWER TO P1 and the",
        "next attempt has to be booked by a human. What it takes is a sum-mode",
        "pinning that clears the roof: widen the sum half of PROBE_PINNINGS, or",
        "run with --compute sum --no-probe at a pinning found by hand.",
    ]


def mechanism_note(say, results: list[ModelResult], l2_bytes: int) -> None:
    """P2, reported and never gated.

    It is not gated because a step in four points is a pattern and not a test,
    and this project has already had to retract one monotone-in-expert-count
    pattern that was an artefact of pooling. Printed with the sizes beside it so
    a reader can see the ordering rather than be told about it.
    """
    fitted = [r for r in results if r.fit.ok and not r.control]
    if len(fitted) < 2:
        return
    say()
    say("## P2, reported and deliberately NOT gated")
    say()
    say("  A step in four points is a pattern, not a test. This study has "
        "already retracted one")
    say("  monotone-in-expert-count reading that was an artefact of pooling, so "
        "the ordering is")
    say("  printed and left for a design that varies the footprint "
        "continuously.")
    say()
    for res in sorted(fitted, key=lambda r: r.per_expert_mib):
        side = "?" if not l2_bytes else (
            "above L2" if res.per_expert_mib * 2 ** 20 > l2_bytes else "below L2")
        span = res.fit.bracket
        say(f"  {res.model:20s} {res.per_expert_mib:8.1f} MiB/expert  "
            f"{side:9s}  alpha {span[0]:.3f} to {span[1]:.3f}")


def verdict(say, gates: list[Gate], reading: list[str] | None = None) -> int:
    """Print every gate's RESULT line and the human table, and return the code.

    THE CODE COMES FROM `exit_codes.classify` OVER THE SAME GATE OBJECTS THAT
    PRINTED THE LINES, so `classify_text` over this function's output recomputes
    the integer the process returns and a disagreement between the two is
    itself a defect. Until 2026-09-02 it did not: the rule here was "any failed
    gate -> 1, any undecided gate -> 4", and the table's rule over the very same
    RESULT lines is "any failed or undecided VALIDITY gate -> 3 INVALID, else
    any failed or undecided CLAIM gate -> 1 CLAIM_FAIL". They parted on every
    VALIDITY failure, which is nearly every gate this script has: `ISA`,
    `correctness`, `placebo`, `signal`, `form`, `control`, `resolution` and all
    six preflights are VALIDITY, and only `P1` is a CLAIM. So
    `--synthetic folded` (a planted fold that fails ISA) and
    `--synthetic noise` (a placebo as large as the signal) both printed
    `RESULT: VALIDITY ... FAIL` and exited 1, and 1 is the code for "the world
    disagreed with a pre-registered prediction". A failed apparatus gate is not
    a finding about the world. It means nothing on the page may be quoted, which
    is INVALID, and the ledger must not file it as a result.

    THE THREE `VERDICT:` LINES STAY, and they are prose. They are what a human
    reads, they are not what the driver greps, and the exit code no longer comes
    from the branch that prints them: the same three sentences are chosen the
    same way, and `classify` answers separately over the gates. One `EXIT:` line
    beside them names the code in words, so the transcript says which of the
    five states this run ended in without anyone counting brackets.

    `reading` is prose printed between the VERDICT lines and the EXIT line, for
    the one case where the code is right and its plain reading is wrong:
    `dot_mode_reading` supplies it when the ladder ran on the tensor cores,
    where P1 is UNKNOWN, `classify` correctly returns CLAIM_FAIL, and CLAIM_FAIL
    read plainly says the world disagreed with the refit when what happened is
    that the estimator could not ask. Empty on every other path, because a
    caveat printed under every verdict is read under none.

    AN EMPTY GATE LIST RAISES `NoGatesScored` rather than returning DONE, which
    is `classify`'s own rule and the right one here: `_analyse` only reaches
    this call once records exist, so no gates at all means the gate builders
    silently produced nothing, and "a check that examined nothing reports zero
    failures" is this project's documented failure shape.
    """
    say()
    say("## gates")
    say()
    for gate in gates:
        say(gate.result_line())
        say(f"  [{gate.label}] {gate.name}")
        say(f"          {gate.detail}")
    say()
    failed = [g for g in gates if g.ok is False]
    untested = [g for g in gates if g.ok is None]
    code = exit_codes.classify(g.scored() for g in gates)
    if failed:
        say(f"VERDICT: REFUTED or VOID. {len(failed)} gate(s) failed: "
            + "; ".join(g.name for g in failed))
    elif untested:
        say(f"VERDICT: NOT TESTABLE. {len(untested)} gate(s) had no evidence: "
            + "; ".join(g.name for g in untested))
    else:
        say("VERDICT: the ablation agrees with the refit. alpha measured "
            "without the byte model,")
        say("without a calibrated bandwidth and without the ridge lands inside "
            "the refit's band,")
        say("so the number the tile-corrected roofline rests on has "
            "independent support.")
    for line in (reading or []):
        say(line)
    if reading:
        say()
    say(f"EXIT: {exit_codes.describe(code)}")
    return code


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

#: The card label a run that touches no GPU carries. `--synthetic` generates its
#: rows from a stated law and `--replay` re-reports a finished directory, so
#: neither has a card, and `provenance.run_id` refuses an id without one. A name
#: no `nvidia-smi` can produce, so it can never be read as a real card.
NO_CARD = "no-card-nothing-measured"


def resolve_card(args) -> str:
    """The card this run is about, or `NO_CARD` when there is not one.

    `--card`, else the live device, else `NO_CARD`. `CannotRunHere` already
    refuses the measuring path without a GPU, so `NO_CARD` reaches the id only
    on the two paths that measure nothing.
    """
    if getattr(args, "card", None):
        return str(args.card)
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            if name:
                return str(name)
    return NO_CARD


def default_run_id(args, card: str, design: Design) -> str:
    """The output directory's name: card first, then every knob that moves a row.

    THE THREE OMISSIONS (A5/P5), all live before 2026-09-02:

      * THE CARD. It is not swept by this script, it is swept by the operator
        moving to another pod, and `results_root()` prefers `$MOE_RESULTS_DIR`
        then `/workspace/results`, a network volume the runbook uses BECAUSE it
        outlives the pod. The resume key is `rung.key`, which is
        model|tiles|block_m and carries no device, so a second card would find
        every rung present, spend no GPU time, and report the first card's
        timings. This experiment's whole subject is L2 behaviour and the two
        cards' L2 differ by 20 MiB, so that is not a small error.
      * `--seed`. It seeds the bootstrap AND, through `measure_rung`, the tensor
        contents; two seeds are two datasets and they shared a directory.
      * `--no-l2-flush`. This is an ABLATION OF CACHE BEHAVIOUR. A warm-L2 run
        and a flushed run are different experiments by construction, and the
        design fingerprint did not distinguish them.
      * `--warmup`, `--cell-budget-ms` and `--trials`, added with `time_kernel`
        on 2026-09-02. Each one sets the measured milliseconds of every pass, so
        a re-run at a different warmup landing in the same directory would find
        every rung already present and report the old numbers under the new
        label. They are in the id for the same reason the card is.

    `design.fingerprint` still carries the models, the tile ladder, BLOCK_M, the
    fixed tile, the compute mode, the replicate count and the control geometry,
    so a change to any of those still lands elsewhere; it is passed through
    rather than re-listed here so the two cannot drift apart. `--bootstrap`
    stays OUT: it re-analyses a set of measurements rather than change one.
    """
    return PV.run_id(card=card, **{
        "1mode": design.compute, "2bm": design.block_m,
        "3design": design.fingerprint, "4seed": args.seed,
        "5flush": bool(args.l2_flush), "6warmup": args.warmup,
        "7budget": args.cell_budget_ms, "8trials": args.trials})


def replay_plan(out_dir: Path) -> dict:
    """`plan.json` of a finished run, for `--replay`. `{}` when there is none.

    A REPLAY MUST BE SCORED AGAINST THE RULER THE RUN WAS MEASURED WITH.
    `--replay` carries neither `--card` nor `--synthetic`, so without this the
    card resolves to `NO_CARD`, `measured_card` finds no calibration, and
    `headroom` and `attribution` -- both VALIDITY -- read UNKNOWN. A finished,
    passing run would come back INVALID purely because it was re-reported on a
    laptop, which is the shape of a false retraction. The plan carries the card,
    the roof and the L2 for exactly this reason, and the synthetic path writes
    one too so a planted world replays as itself.
    """
    with contextlib.suppress(Exception):
        return json.loads((out_dir / "plan.json").read_text())
    return {}


def results_root() -> Path:
    """Where output goes so that it survives the pod being terminated.

    Same rule as `scripts/run_all.sh` and `scripts/group_m_alpha_sweep.py`:
    `$MOE_RESULTS_DIR`, else the network volume at `/workspace/results` when
    there is one, else the repo's own `results/`. The pod's container disk dies
    with the pod and the volume does not.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return ROOT / "results"


def git_head() -> str:
    with contextlib.suppress(Exception):
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    return ""


def l2_bytes_here() -> int:
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(getattr(props, "L2_cache_size", 0))
    return 0


def cross_check_candidates() -> str:
    """The two published rivals, checked against `scripts/alpha_refit.py`.

    Loaded by path because `scripts/` is not a package. This does NOT import the
    estimator -- the whole value of this experiment is that it fits alpha by a
    route `alpha_refit` does not touch -- but the three numbers being scored
    against must not drift away from the ones the estimator prints.
    """
    import importlib.util
    path = ROOT / "scripts" / "alpha_refit.py"
    try:
        spec = importlib.util.spec_from_file_location("alpha_refit_constants", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - a missing rival is not fatal
        return f"could not read {path.name} ({type(exc).__name__}); using literals"
    mismatches = []
    if getattr(module, "REPO_PUBLISHED_ALPHA", None) != REPO_RETRACTED_ALPHA:
        mismatches.append("REPO_PUBLISHED_ALPHA")
    if getattr(module, "TEMPO_ALPHA", None) != TEMPO_ALPHA:
        mismatches.append("TEMPO_ALPHA")
    if mismatches:
        return (f"DISAGREES with alpha_refit.py on {', '.join(mismatches)}. "
                "One of the two files has drifted and the scoring below is "
                "against this file's literals")
    return "agrees with alpha_refit.py on both rival values"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--tiles", default=",".join(str(t) for t in TILE_LADDER))
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=FIXED_TILE["BLOCK_N"])
    parser.add_argument("--block-k", type=int, default=FIXED_TILE["BLOCK_K"])
    parser.add_argument("--group-m", type=int, default=FIXED_TILE["GROUP_M"])
    parser.add_argument("--num-warps", type=int,
                        default=FIXED_TILE["num_warps"])
    parser.add_argument("--num-stages", type=int,
                        default=FIXED_TILE["num_stages"])
    parser.add_argument("--alias-extent", choices=ALIAS_EXTENTS,
                        default=DEFAULT_ALIAS_EXTENT,
                        help="how far the alias reaches. block (default) zeroes "
                             "the expert and N-block strides and KEEPS the K "
                             "advance, so the aliased arm walks one BLOCK_N x K "
                             "column block, L2-resident and spread. tile zeroes "
                             "the advance too and re-reads one 16 KiB tile, "
                             "which is the 2026-09-01 design and the pinning "
                             "whose L2 slice contention drove r to 12")
    parser.add_argument("--probe", action="store_true", default=True,
                        help="before the ladder, time a short pinning grid and "
                             "adopt the one whose aliased arm delivers requests "
                             "faster than the card's DRAM roof. Without "
                             "headroom no ladder can measure alpha, so a probe "
                             "that finds none stops the arm (default on)")
    parser.add_argument("--no-probe", dest="probe", action="store_false")
    parser.add_argument("--compute", choices=COMPUTE_MODES, default="sum",
                        help="how the loads are kept live. sum is STUDY.md's "
                             "prescription and is unbiased; dot is the real "
                             "GEMM reduction and is biased low")
    # A STRING AND NOT A BOOL, because argparse applies `type` BEFORE it checks
    # `choices`, so a converting type here would score True against
    # ("allow", "refuse") and reject every value including the default.
    parser.add_argument("--dot-fallback", choices=("allow", "refuse"),
                        default="allow", dest="dot_fallback",
                        help="what the probe may do when no sum-mode pinning "
                             "clears the DRAM roof. allow (default) takes the "
                             "fastest dot-mode one, which measures a LOWER "
                             "BOUND on alpha, leaves P1 UNKNOWN and therefore "
                             "exits 1 CLAIM_FAIL: a code the ledger latches, "
                             "so the run prints in words that the claim was "
                             "not asked rather than refuted. refuse stops the "
                             "arm at the probe instead, which is INVALID and "
                             "costs only the probe")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_MS,
                        help="milliseconds of GPU time delivered under "
                             "sustained load before a pass is timed. A "
                             "DURATION, not a call count: the rungs differ by "
                             "an order of magnitude in per-call time, so a "
                             "count would warm them by an order of magnitude "
                             "of different load and then compare their clocks")
    parser.add_argument("--cell-budget-ms", type=float,
                        default=DEFAULT_CELL_BUDGET_MS,
                        help="target duration of ONE trial; time_kernel sizes "
                             "the iteration count from the warmup's own "
                             "queue-deep per-call time to hit it")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help="queue-deep trials per time_kernel call; the p50 "
                             "over all of them is the pass's number")
    parser.add_argument("--control", action="store_true", default=True,
                        help="include the L2-resident control rung (default on)")
    parser.add_argument("--no-control", dest="control", action="store_false")
    parser.add_argument("--run", action="store_true",
                        help="measure on the GPU; without it this plans only")
    parser.add_argument("--replay", type=Path, default=None,
                        help="re-report an existing output directory, no GPU")
    parser.add_argument("--synthetic", choices=sorted(SYNTHETIC_LAWS), default=None,
                        help="generate timings from a stated law and run the "
                             "gates on them, so the gates are testable off-GPU")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any rungs already on disk and start over")
    parser.add_argument("--no-l2-flush", dest="l2_flush", action="store_false",
                        help="time with L2 warm; the default flushes, because "
                             "an ablation of cache behaviour must control the "
                             "cache state it starts from")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--card", default=None,
                        help="the card this run measures, as nvidia-smi names "
                             "it. Defaults to the live device; --synthetic and "
                             "--replay touch no GPU and are labelled "
                             f"{NO_CARD!r}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.models = tuple(v for v in str(args.models).split(",") if v)
    args.tiles = tuple(sorted(int(v) for v in str(args.tiles).split(",") if v))
    unknown = [m for m in args.models
               if m != CONTROL_MODEL and m not in MODEL_CONFIGS]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from "
                     f"{sorted(MODEL_CONFIGS)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    design = build_design(args)
    suffix = f"-synthetic-{args.synthetic}" if args.synthetic else ""
    card = resolve_card(args)

    def resolve_out(d: Design) -> Path:
        return args.replay or args.out or (
            results_root() / "alias_ablation"
            / f"{default_run_id(args, card, d)}{suffix}")

    out_dir = resolve_out(design)

    # THE CEILINGS ARE RESOLVED BEFORE ANYTHING IS PRINTED, because every number
    # on the page divides by one of them: the cost, the headroom, the
    # attribution and the probe's own verdict. A reader must see which file they
    # came from at the top, beside the card, and not thirty lines later.
    planned = replay_plan(out_dir) if args.replay else {}
    # THE ESTIMATOR IS PART OF THE RULER. `replay_plan` restored the card, the
    # roof and the L2 so that a finished run is not retracted by being
    # re-reported on a laptop, and it left out the two knobs that decide what
    # the numbers MEAN. `--replay` carries no `--compute`, so a dot-mode ladder
    # replayed with a bare `--replay` was rebuilt as a sum-mode design and
    # scored by `prediction_gate`'s UNBIASED branch: the run that exited 1 with
    # P1 UNKNOWN and "this is a LOWER BOUND" came back 0 DONE, "alpha measured",
    # off the same cells. `scripts/pod_session.sh:2467` copies report.md into
    # the published arm directory, so that replay is what would have been
    # published. `--alias-extent` is restored with it, for the same reason and
    # one line earlier than it would have been needed.
    for knob in ("compute", "alias_extent"):
        want = planned.get(knob)
        if want and want != getattr(args, knob, None):
            setattr(args, knob, want)
            design = build_design(args)
    card = planned.get("card") or card
    facts = {} if args.synthetic else measured_card(card)
    if planned.get("roof_bytes_s") and not facts.get("roof_bytes_s"):
        facts = dict(facts, source=f"{out_dir.name}/plan.json",
                     roof_pattern=planned.get("roof_pattern", ROOF_PATTERN))
    l2 = (SYNTHETIC_L2_BYTES if args.synthetic
          else (planned.get("l2_bytes") or facts.get("l2_bytes")
                or l2_bytes_here()))
    # NO FALLBACK ON THE ROOF. Every headroom, attribution and cost number
    # divides by it, and a datasheet pin rate in its place would move each of
    # them by the ratio of two machines. None means those gates read UNKNOWN,
    # which is a VALIDITY UNKNOWN and therefore INVALID, and that is correct.
    roof = (SYNTHETIC_ROOF_BYTES_S if args.synthetic
            else (planned.get("roof_bytes_s") or facts.get("roof_bytes_s")))

    # PROVENANCE, built once and written into every artefact this run leaves.
    # None of the ten gaps-session scripts wrote a commit, a card or an
    # instrument, so not one of the 26 published reports can be attributed to a
    # code version (A5).
    prov = PV.provenance_block(instrument=timing.TIMING_BASIS)

    say = Report()
    say(f"# alpha by ablation, without the byte model   ({git_head() or 'no git'})")
    say()
    say(f"card: {card}")
    say(f"output directory: {out_dir}")
    say("Everything below is written there as report.md, beside plan.json and "
        "cells.jsonl.")
    say(f"candidate values: {cross_check_candidates()}")
    if facts.get("source"):
        say(f"ceilings: {facts['source']}, {facts['roof_pattern']} roof "
            f"{(roof or 0) / 1e9:.0f} GB/s, L2 {l2 / 2**20:.1f} MiB")
    elif args.synthetic:
        say(f"ceilings: PLANTED at {PLANT_CARD}'s measured values, read roof "
            f"{SYNTHETIC_ROOF_BYTES_S / 1e9:.0f} GB/s, L2 "
            f"{SYNTHETIC_L2_BYTES / 2**20:.1f} MiB")
    else:
        say("ceilings: NO CALIBRATION for this card. The headroom and "
            "attribution gates will read UNKNOWN, which is INVALID; run "
            "scripts/calibrate_hardware.py --publish first")
    if args.synthetic:
        say()
        say(f"*** SYNTHETIC ({args.synthetic}): {SYNTHETIC_LAWS[args.synthetic]}.")
        say("*** Nothing here was measured. These rows come from a stated law "
            "and exist to show")
        say("*** the gates can see an effect and can miss its absence.")
    say()
    report_prediction(say)

    # A SYNTHETIC REPLAY MUST NOT READ THE HARDWARE. `--synthetic l2-step`
    # plants its law at SYNTHETIC_L2_BYTES, so classifying the recovered models
    # against whatever card happens to be attached makes the planted answer
    # depend on the box: this read `l2_bytes_here() or SYNTHETIC_L2_BYTES`, which
    # took the real L2 whenever a GPU was present and only fell back to the
    # planted value on a laptop. It passed for weeks purely because the laptop
    # fallback and the planted threshold are both 50 MiB, and failed the moment
    # it met an H200, where L2 is 60 MiB and deepseek-v2's 56 MiB/expert crosses
    # from above the line to below it. The whole point of a planted law is that
    # recovering it is machine-independent.
    pre = preflight(design, l2)
    report_design(say, design, pre, l2, synthetic=bool(args.synthetic))
    # `args.probe` ALONE. `and args.run` denied the probe's cost on the only
    # page an operator can read before they have a card, which is the page a
    # rental is sized from. See `report_cost`.
    report_cost(say, design, args, roof, probing=bool(args.probe))
    report_mde(say, design)
    if any(g.ok is False for g in pre):
        say()
        say("REFUSED: the design is refused before spending anything. Fix the "
            "failed preflight gate above.")
        _save(out_dir, say, prov)
        # REFUSED (2), NOT CLAIM_FAIL (1). It returned 1 until 2026-09-02, and 1
        # means "measured; VALIDITY passed; a pre-registered claim did not" --
        # a statement about the world, made by a run that had not started. The
        # preflight gates print no RESULT line here on purpose (they are scored
        # in `_analyse`, on a page that HAS timings), so the log carries none at
        # all and `exit_codes.classify_text` raises `NoGatesScored`, which is
        # the REFUSED shape. Nothing was spent and nothing was measured.
        return exit_codes.REFUSED

    records: list[dict] = []
    if args.replay:
        records = read_records(out_dir / "cells.jsonl")
        say()
        say(f"## replay: {len(records)} rungs read from disk, nothing measured")
    elif args.synthetic:
        records = synthesise(design, args.synthetic, args.seed)
        with contextlib.suppress(OSError):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "cells.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records))
            # A PLANTED WORLD NEEDS ITS PLAN AS MUCH AS A MEASURED ONE. Without
            # it `--replay` of this directory has no roof to score headroom and
            # attribution against and the planted world comes back INVALID for
            # a reason that has nothing to do with the law it plants.
            (out_dir / "plan.json").write_text(json.dumps(prov.stamp(
                {"fingerprint": design.fingerprint, "argv": sys.argv[1:],
                 "synthetic": args.synthetic, "card": card,
                 # BOTH WRITERS OR NEITHER. The measured writer below has
                 # carried `compute` since it was written; this one did not, so
                 # every planted world replayed as sum mode however it was
                 # planted, and the rehearsal of the dot path could not
                 # rehearse its own replay.
                 "compute": design.compute,
                 "alias_extent": design.alias_extent,
                 "roof_bytes_s": SYNTHETIC_ROOF_BYTES_S,
                 "roof_pattern": ROOF_PATTERN,
                 "l2_bytes": SYNTHETIC_L2_BYTES,
                 "run_id": out_dir.name, "seed": args.seed}), indent=2))
    elif args.run:
        if args.probe:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise CannotRunHere("no CUDA device; the probe measures one")
                readings = measure_probe(design, args, roof, torch)
            except (ImportError, CannotRunHere) as exc:
                say()
                say(f"REFUSED. CANNOT RUN HERE: {exc}")
                say("The plan, the prediction and the preflight above are still "
                    "valid and cost nothing;")
                say("re-run with --run on the pod, or --synthetic to exercise "
                    "the gates.")
                _save(out_dir, say, prov)
                # REFUSED (2), the same code and the same words the measuring
                # path uses. The probe is the first thing --run reaches, so on a
                # box with no card this branch is the one that answers, and it
                # must not answer differently from the branch below it.
                return exit_codes.REFUSED
            chosen, why = choose_pinning(readings, roof,
                                         args.dot_fallback == "allow")
            report_probe(say, readings, roof, chosen, why,
                         args.dot_fallback == "allow")
            if chosen is None:
                # INVALID (3), NOT REFUSED (2). The probe SPENT card time and
                # left cells on disk; REFUSED means free. The one RESULT line
                # `verdict` prints below is what lets `classify_text` recompute
                # this 3 from the log, and the arm must not be re-queued: at
                # this pinning grid the design cannot see DRAM and re-running it
                # reproduces that exactly.
                code = verdict(say, [probe_gate(readings, chosen, why)])
                say()
                say("The ladder was NOT run. Nothing after the probe could have "
                    "measured alpha, so")
                say("the remaining rungs were not paid for. Widen "
                    "PROBE_PINNINGS or accept that the")
                say("ablation route is closed on this kernel and take the "
                    "counter route instead.")
                _save(out_dir, say, prov)
                with contextlib.suppress(OSError):
                    (out_dir / "probe.json").write_text(json.dumps(
                        prov.stamp({"readings": readings, "chosen": None,
                                    "why": why, "card": card,
                                    "roof_bytes_s": roof}), indent=2))
                return code
            if chosen["compute"] != args.compute:
                say()
                say(f"## the probe changed the COMPUTE MODE: "
                    f"{args.compute} -> {chosen['compute']}")
                say()
                for line in dot_mode_reading(chosen["compute"]):
                    say(f"  {line}")
            args.num_warps = chosen["num_warps"]
            args.num_stages = chosen["num_stages"]
            args.block_k = chosen["block_k"]
            args.compute = chosen["compute"]
            design = build_design(args)
            out_dir = resolve_out(design)
            pre = preflight(design, l2)
            say()
            say(f"## re-pinned by the probe; output directory is now {out_dir}")
            say()
            for gate in pre:
                say(f"  [{gate.label}] {gate.name}")
                say(f"          {gate.detail}")
            if any(g.ok is False for g in pre):
                say()
                say("REFUSED: the adopted pinning does not pass the preflight.")
                _save(out_dir, say, prov)
                return exit_codes.REFUSED
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.fresh:
            (out_dir / "cells.jsonl").unlink(missing_ok=True)
        # THE RUN ID IS ONE WAY IN AND `--out` IS THE OTHER. Every knob that
        # moves a row is in `default_run_id`, so the default path cannot resume
        # into another design's directory. `--out` names a directory outright
        # and walks past all of it, and `scripts/pod_session.sh:1676` passes
        # `--out`. The resume key is `rung.key`, which carries no pinning and no
        # extent, so a second design landing in the first one's directory finds
        # every rung present, spends no GPU time, and reports the other design's
        # timings under its own label. The probe makes that live rather than
        # hypothetical: it can adopt a different pinning on two runs of one
        # command. The fingerprint is checked here, on the way in.
        prior = replay_plan(out_dir)
        if prior.get("fingerprint") and prior["fingerprint"] != design.fingerprint:
            say()
            say(f"REFUSED: {out_dir} holds a run of design "
                f"{prior['fingerprint']} and this is {design.fingerprint}.")
            say(f"  its argv: {prior.get('argv')}")
            say("  The resume key is the rung, which carries neither the "
                "pinning nor the alias extent, so")
            say("  resuming here would report that design's timings under this "
                "one's label. Use a")
            say("  different --out, or --fresh, or drop --out and let the run "
                "id separate them.")
            _save(out_dir, say, prov)
            # REFUSED (2): nothing was measured and nothing is owed. The fix is
            # a flag, and the log's first REFUSED line says which.
            return exit_codes.REFUSED
        existing = read_records(out_dir / "cells.jsonl")
        done = {r["id"] for r in existing}
        (out_dir / "plan.json").write_text(json.dumps(prov.stamp(
            {"fingerprint": design.fingerprint, "argv": sys.argv[1:],
             "git": git_head(), "models": list(design.models),
             "tiles": list(design.tiles), "block_m": design.block_m,
             "tile": design.tile, "compute": design.compute,
             "alias_extent": design.alias_extent,
             "roof_bytes_s": roof, "roof_pattern": ROOF_PATTERN,
             "l2_bytes": l2,
             "card": card, "run_id": out_dir.name, "seed": args.seed,
             "l2_flush": bool(args.l2_flush),
             "warmup_ms": args.warmup, "cell_budget_ms": args.cell_budget_ms,
             "trials": args.trials,
             "replicates": design.replicates}), indent=2))
        say()
        say(f"## measuring: {len(design.rungs) - len(done)} rungs to do, "
            f"{len(done)} already on disk")
        try:
            fresh, _ = measure(design, args, out_dir, done)
        except CannotRunHere as exc:
            say()
            say(f"REFUSED. CANNOT RUN HERE: {exc}")
            say("The plan, the prediction and the preflight above are still "
                "valid and cost nothing;")
            say("re-run with --run on the pod, or --synthetic to exercise the "
                "gates.")
            _save(out_dir, say, prov)
            # REFUSED (2), NOT INVALID (3). It returned 3 until 2026-09-02, and
            # 3 means "measured, then a VALIDITY gate failed: nothing quotable,
            # and do NOT retry it". Nothing was measured here and there is no
            # gate in the log to have failed; the missing thing is a GPU, which
            # is a precondition, and the arm is free to retry on a box that has
            # one. The driver treats the two oppositely, which is why they are
            # two codes.
            return exit_codes.REFUSED
        except KeyboardInterrupt:
            say()
            say("## aborted; reporting on what reached disk")
            fresh = []
        records = existing + fresh
    else:
        say()
        say("Nothing was measured. Add --run on the pod, --synthetic to "
            "exercise the gates,")
        say("or --replay <dir> to re-report a finished run.")
        _save(out_dir, say, prov)
        # REFUSED, NOT DONE. This path prints a plan, a prediction, a preflight
        # and an MDE, and times nothing; it used to return 0, so a driver that
        # asked for the arm and got a bare invocation logged it DONE and never
        # ran it. There are no RESULT lines above either, so
        # `exit_codes.classify_text` raises `NoGatesScored` on this log, which
        # is what a REFUSED log looks like from there: the two agree.
        return exit_codes.REFUSED

    return _analyse(say, design, records, args, out_dir, l2, prov, pre, roof)


def _analyse(say, design: Design, records: list[dict], args, out_dir: Path,
             l2: int, prov=None, pre: list[Gate] | None = None,
             roof: float | None = None) -> int:
    known = {r.key for r in design.rungs}
    stray = [r for r in records if r.get("id") not in known]
    synthetic = bool(args.synthetic) or any(
        r.get("provenance") == "synthetic" for r in records)
    if synthetic and not args.synthetic:
        say()
        say("*** SYNTHETIC. These records were generated from a stated law "
            f"({records[0].get('law', 'unknown')}) and nothing here was "
            "measured on any hardware.")
    if stray:
        say()
        say(f"  {len(stray)} records name a rung this design does not contain "
            f"and are IGNORED. First: {stray[0].get('id')}. That means the "
            "flags differ from the ones that")
        say("  produced the file; re-run --replay with the argv in plan.json.")
    records = [r for r in records if r.get("id") in known]

    # THE SECOND WAY IN, AND THE ONE THAT DOES NOT DEPEND ON A FILE BEING
    # RIGHT. Above, `--replay` adopts the knob out of plan.json; here the CELLS
    # are asked, and they are asked on every path. A plan.json written before
    # 2026-09-03, a hand-edited one, a directory named with `--out` and replayed
    # from somewhere else: all of them reach this line, and every record on both
    # the measured and the synthetic paths carries `CELL_KNOBS`. Two walls
    # because there are two ways in, which is the defect this rebuild has hit
    # eight times -- and this loop is the eighth: it read `compute` alone while
    # `--replay` restored BOTH knobs, so a directory holding two alias extents
    # was pooled into one alpha with nothing on the page saying so.
    for knob in CELL_KNOBS:
        stakes = CELL_KNOB_STAKES[knob]
        seen = {r.get(knob) for r in records if r.get(knob)}
        if len(seen) > 1:
            say()
            say(f"REFUSED: these cells were measured in more than one "
                f"{stakes['noun']} ({', '.join(sorted(seen))}), and")
            say(f"  {stakes['mixed']}. Split the")
            say("  directory or re-run with --fresh.")
            _save(out_dir, say, prov)
            return exit_codes.REFUSED
        if seen and seen != {getattr(design, knob)}:
            found = seen.pop()
            design = replace(design, **{knob: found})
            say()
            say(f"## SCORED AS {found.upper()} {stakes['heading']}: the cells "
                f"say so and this invocation said {getattr(args, knob)}")
            say()
            say(f"  Every record in cells.jsonl carries {knob}={found}. The "
                "design this process built")
            say(f"  said {getattr(args, knob)}, and the two decide different "
                f"things: {stakes['decides']}.")
            say("  The CELLS win, because they are the measurement and the "
                "flags are only how it was")
            say("  asked for. The verdict below is scored on the value named "
                "on this line.")
            setattr(args, knob, found)

    timed = [r for r in records if r.get("ms")]
    if not timed:
        say()
        say("REFUSED. NOT TESTABLE: nothing was timed, so no gate below could "
            "have examined anything.")
        _save(out_dir, say, prov)
        # REFUSED (2), NOT ERROR (4). It returned 4 until 2026-09-02, and 4 is
        # "crashed; an exception the script did not plan for", which the ledger
        # reads as RETRY with a traceback to go and find. This path is planned
        # and there is no traceback: every record that reached disk carried no
        # `ms`, so nothing was measured. No gate is scored below it either, so
        # the log carries no RESULT line and `exit_codes.classify_text` raises
        # `NoGatesScored`, which is the REFUSED shape.
        return exit_codes.REFUSED

    throttled = [r["id"] for r in timed if r.get("throttled")]
    if throttled:
        say()
        say(f"  {len(throttled)} rungs drifted more than "
            f"{CLOCK_DRIFT_LIMIT * 100:.0f}% in SM clock: {throttled[:3]}. The "
            "interleaved order is what")
        say("  protects a paired difference from that, and the placebo gate is "
            "what measures it.")

    # A COLUMN NOTHING READS IS A COLUMN NOTHING PROTECTS. `clock_level_ok`
    # reached `cells.jsonl` on every rung and was consulted by no line of this
    # report, so a card held below the roof's clock for a whole ladder produced
    # a published alpha with nothing on the page saying so. Planted rows carry
    # no clock and are not scored against one.
    #
    # THE TWO FAULTS ARE INDEPENDENT, AND THIS WAS `if sagged / elif blind`, so
    # a directory holding both reported only the first. That directory is not
    # hypothetical: it is a resume onto a re-rented pod, where the rungs already
    # on disk are skipped and keep whatever they were written with, so rows that
    # sagged and rows written before there was a reference to sag against travel
    # together. Twenty rungs with three sagged and seventeen carrying no key at
    # all printed the three and said nothing about the seventeen, and a reader
    # took those seventeen for level when none of them carried a verdict.
    #
    # BLIND IS PRINTED FIRST, for the reason `group_m_alpha_sweep` states at its
    # own copy of this block: whether the column could have said anything is a
    # different question from what it said, and a count of flagged rungs read
    # without it is silence read as evidence.
    if not synthetic:
        sagged, blind = level_split(timed)
        say()
        if blind:
            say(f"  LEVEL: UNDETERMINED on {len(blind)} of {len(timed)} rungs. "
                "They were written with no")
            say("  reference clock, which is every row this arm wrote before "
                "2026-09-03, so none of")
            say("  them can be excluded on the clock it ran at.")
        if sagged:
            say(f"  LEVEL: {len(sagged)} of {len(timed)} rungs ran below "
                f"{timing.LEVEL_FRACTION:.0%} of the clock this card's roof "
                f"was measured at: {sagged[:3]}.")
            say("  D(n) and D(1) are differences between two ladders, and a sag "
                "part way up one moves")
            say("  them by different amounts, so the slope over intercept does "
                "not cancel it. Those")
            say("  rungs have to be re-measured before their alpha means "
                "anything.")
        if not blind and not sagged:
            say(f"  LEVEL: all {len(timed)} rungs ran within "
                f"{timing.LEVEL_FRACTION:.0%} of the clock this card's roof "
                "was measured at.")

    report_measurements(say, timed, design)
    report_isa(say, timed)

    results = analyse_models(design, timed, args.bootstrap, args.seed)
    by_model = samples_from(timed)
    real = {r.model: by_model[r.model]
            for r in results if not r.control and r.model in by_model}
    brackets = [r.fit.bracket for r in results if r.fit.ok and not r.control]
    pooled_bracket = (
        (statistics.median(b[0] for b in brackets),
         statistics.median(b[1] for b in brackets)) if brackets else None)
    pooled = pooled_band(real, args.bootstrap, args.seed) if real else None
    l2_shares = [r.fit.l2_share for r in results
                 if r.fit.ok and not r.control and r.fit.l2_share is not None]
    report_alphas(say, results, pooled, pooled_bracket)
    mechanism_note(say, results, l2)
    report_confounds(say, design, results)
    # THE MDE'S SECOND CALL SITE. `main` printed one before a rung was timed,
    # against the corpus's within-cell floor; this one is against the scatter
    # this run's own interleaved passes delivered. Both are needed and neither
    # substitutes: the first sizes the booking, the second says whether the
    # gates below are informative when they FAIL. Stating only the first is how
    # an arm returns a FAIL nobody can read.
    shares = one_tile_signal_shares(timed)
    report_mde(say, design, observed_spreads(timed), measured=True,
               signal_share=shares[0][0] if shares else None)

    # ORDER IS THE READING ORDER. headroom and attribution come before signal
    # and form because they are the CAUSE those two report the consequence of:
    # on 2026-09-01 signal said D(1) was 5.3% of a pass and form said R^2 0.42,
    # and a reader who stopped there concluded the per-tile cost is not DRAM.
    # These two say, in the card's own units, that DRAM was never on the
    # critical path, which is a statement about the apparatus.
    #
    # LEVEL IS THE FIRST GATE AND IT IS SCORED, NOT NARRATED. It was a
    # paragraph, and `pod_session.sh` grades this arm on the exit code and on
    # the count of `[PASS]`/`[FAIL]` lines, so a paragraph let a sagged hour
    # exit 0 DONE. It runs first for the same reason headroom precedes signal:
    # whether the card was at the clock its roof was measured at is a statement
    # about the apparatus, and it is upstream of every number below it. Scored
    # only on a measured run, because planted rows carry no clock.
    gates = [
        *([level_gate(timed)] if not synthetic else []),
        isa_gate(timed),
        correctness_gate(timed, design.compute),
        headroom_gate(timed, roof),
        attribution_gate(timed, roof),
        placebo_gate(timed),
        signal_gate(timed),
        linearity_gate(results),
        control_gate(results),
        bracket_gate(results),
        band_gate(pooled, l2_shares),
        prediction_gate(pooled, pooled_bracket, design.compute),
    ]
    # THE PREFLIGHT GATES ARE SCORED HERE AND NOWHERE ELSE. They are decided
    # before a rung is timed, but their RESULT lines belong to a page that HAS
    # timings on it: printed at plan time they let a run that measured nothing
    # classify as DONE. `main` only reaches this call after `records` exist.
    code = verdict(say, list(pre or []) + gates,
                   dot_mode_reading(design.compute))
    _save(out_dir, say, prov)
    return code


def _save(out_dir: Path, say: Report, prov=None) -> None:
    with contextlib.suppress(OSError):
        out_dir.mkdir(parents=True, exist_ok=True)
        say.save(out_dir / "report.md")
        if prov is not None:
            # Beside the markdown, not inside it: the report is prose a human
            # reads and the block is fields a script reads, and a run whose
            # report.md was hand-edited must still carry an unedited record of
            # the commit, the card and the instrument.
            (out_dir / "provenance.json").write_text(
                json.dumps(prov.stamp({"run_id": out_dir.name}), indent=2))
        print()
        print(f"[alias] report written to {out_dir / 'report.md'}")
        print(f"[alias] measurements at  {out_dir / 'cells.jsonl'}")


def _guarded(argv: list[str] | None = None) -> int:
    """`main`, with the ERROR(4) handler the session ledger needs.

    THE HOLE THIS CLOSES IS WORTH A RENTAL. Python spends 1 on any exception
    that escapes, `moe/bench/exit_codes` reads 1 as CLAIM_FAIL, and
    `scripts/h200_gaps_session.sh` LATCHES CLAIM_FAIL: the arm is marked
    finished, skipped on every resume, and its silence is printed in the closing
    summary as a refuted prediction. A crash is the apparatus failing, not the
    world disagreeing, so it exits 4 and the driver may attempt it again.

    A `SystemExit` carrying a string is argparse or an explicit refusal, which
    is REFUSED(2) and free, not a crash.
    """
    import traceback
    try:
        return main(argv)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            message = (exc.code if exc.code.startswith("REFUSED")
                       else f"REFUSED: {exc.code}")
            print(message, file=sys.stderr)
            return exit_codes.REFUSED
        raise
    except KeyboardInterrupt:
        print("REFUSED: interrupted before a verdict", file=sys.stderr)
        return exit_codes.REFUSED
    except Exception:  # noqa: BLE001 - the whole point is to catch everything
        traceback.print_exc()
        print("ERROR: alias_ablation crashed before it could reach a verdict. "
              "This is the apparatus failing, not a claim failing, so it exits "
              f"{exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: the "
              "traceback above is the thing to fix, and the arm may be re-run.",
              file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    raise SystemExit(_guarded())
