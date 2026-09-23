#!/usr/bin/env python
"""alpha as a measured ratio of two slopes, with no assumed bandwidth in it.

    python scripts/private_weight_reference.py --dry-run --device-memory-gb 140
    python scripts/private_weight_reference.py --self-test refit
    python scripts/private_weight_reference.py --self-test issue-bound
    python scripts/private_weight_reference.py                # the pod run
    python scripts/private_weight_reference.py --duty 0.25    # the pod setting (V7 checks it)
    python scripts/private_weight_reference.py --seed 1 --replicate-of RUN0/report.json
                                       # a second run; C1 scored WITH the first
    python scripts/private_weight_reference.py --read RUN1/report.json \
                                               --replicate-of RUN0/report.json
                                       # off GPU: a stored pair re-read, nothing measured

WHY THIS ARM EXISTS. `alpha` is defined in this study as the fraction of the
routed expert weight set that is re-read per extra M-tile. Every estimate of it
so far divides a measured TIME by something that was not measured in the same
breath: either an ASSUMED bandwidth (`w = slope_ms / weight_stream_ms(rate)`)
or a FITTED intercept (`LadderFit.alpha = B / (A + B)`, an extrapolation of the
ladder back to n = 0). The 13-agent reading of the 2026-09-10 session concluded
that alpha, so defined, is NOT IDENTIFIED BY THIS APPARATUS on three
independent grounds: the same six cells give 0.9794 under one parameterisation
and 0.5977 under another, a factor of 1.6 apart, against that fit's own
bootstrap sd of 0.0113 -- and against the 0.0037 the corpus published, which
the same reading calls an understatement (FORM; the pair, the cell count and
both sds are `bn_decomposition`'s own committed report, not this file's);
the card is power-capped so the SM clock is an endogenous response to the
tile, and sweeping the admissible clock elasticity moves pooled alpha from
0.974 to 0.897, seven times that bootstrap sd and 21 times the published one
(CLOCK);
and by bus arithmetic 4 of the 40 published alphas, re-scored against their
own measured n=1 anchors, imply more than their card's pin rate -- all four
on the A100, none on the H200 (PHYSICS; ANCHOR_RESCORE.json gate C1).

This arm removes the denominator. It measures a SECOND ladder in which the
re-read fraction is ONE BY CONSTRUCTION, and divides one measured slope by the
other:

    alpha_ratio = slope(SHARED) / slope(PRIVATE)

No bandwidth, no intercept, no assumed rate anywhere in it. If the PRIVATE
ladder really re-reads the whole weight set per M-tile, its slope IS the
alpha = 1 reference in the same units, on the same card, in the same kernel,
measured minutes apart -- and V7 checks, rather than assumes, that it was at
the same clock.

THE LAYOUT, and why it is expert-first. The allocation holds `n_decl`
complete copies, and copy `c` of expert `e` is expert slot `e x n_decl + c`
(`copy_slot`). SHARED and PRIVATE both declare all `E x n_decl` slots at
EVERY tread and both pass the whole `w1`/`w2`. `n_decl` is the DEEPEST
TREAD's copy count `n_max`, or more when `declared_copies_for` pads it past
vLLM's small-batch expert bound to keep both ratio arms on one alignment
kernel; the extra copies are filled and never read. That is what makes the
two one call with one difference:

    SHARED    `E x n_decl` experts declared, every M-tile of expert `e` routed
              to slot `e x n_decl` (copy 0). One copy is READ. This is the
              numerator of the ratio.
    PRIVATE   the same declaration, the same tensors, M-tile `j` of expert `e`
              routed to slot `e x n_decl + j`. Reuse across M-tiles is
              impossible: copy j is read by tile j and by nothing else. This
              is the denominator.
    NATIVE    the study's own call: `E` experts declared, `w1[::n_decl]` -- a
              strided view whose rows ARE copy 0 of each expert, the same bytes
              SHARED reads at the same addresses -- and the unrelabelled ids.
              The control that ties the ratio to the path the study fits.

WHAT AN EARLIER VERSION GOT WRONG, and it is why the layout is what it is.
Copies were laid out COPY-FIRST (`c x E + e`) and PRIVATE and its control
declared `E x n` experts at tread `n` while SHARED declared `E`. Two defects
followed, both invisible to every planted world: (1) `moe_align_block_size`
sizes its sorted-id buffer as `numel + num_experts x (BLOCK_M - 1)` and the
kernel's launch grid is `cdiv(buffer, BLOCK_M) x cdiv(N, BLOCK_N)`, so the
private arm launched ~8 dead M-rows per extra tread that SHARED did not --
a per-tread term in the DENOMINATOR slope, biasing the ratio low in exactly
the issue-and-latency world; and (2) the alignment sorts by expert id, so
copy-first ids ran the private arm's tiles in a DIFFERENT ORDER (copy 0 of
every expert, then copy 1 ...) from SHARED's (every tile of expert 0, then
expert 1 ...), and this study's cleanest result is that ORDER moves the
per-M-tile cost by 30-48%. Expert-first slots at a fixed `E x n_decl`
declaration make the buffer, the launch grid, the dead launches (a constant
per tread, so an INTERCEPT and not a slope), Triton's integer specialisation
and the tile order identical in SHARED and PRIVATE at every tread.

NO KERNEL WAS WRITTEN FOR THIS AND NONE IS NEEDED. vLLM's fused_moe Triton
kernel reads its expert index PER M-TILE (`off_experts = tl.load(expert_ids_ptr
+ pid_m)`, filled by `moe_align_block_size` from `topk_ids`), and addresses
the weights through `stride(0)`, so giving each M-tile a private weight copy is
a RELABELLING of `topk_ids` plus a wider `w1` and `w2`. Between SHARED and
PRIVATE at a tread the kernel binary, the tile, the launch grid, the M-tile
count, the padded rows, the FLOPs, the tile order and the activation traffic
are identical; the only thing that moves is which address each tile reads its
weights from. NATIVE differs from SHARED in the declaration only, which V5
measures. `private_topk_ids` and `shared_topk_ids` are the whole of it and
both are pure.

THE CAVEAT, REGISTERED HERE AND PRINTED ON THE PLAN PAGE BEFORE THE RUN.
Private copies change the address stream the kernel reads its weights from,
which is also its TLB footprint and its DRAM page locality. So slope(PRIVATE)
is a BOUNDED PROXY for the no-reuse case, not the no-reuse case itself, and
this arm measures PART of the bound rather than asserting it. WHICH PART,
exactly, because the pieces are not equally covered:

  * NATIVE bounds the DECLARATION. It reads SHARED's bytes at SHARED's
    addresses in SHARED's order and declares `E` instead of `E x n_decl`.
    `slope(NATIVE) - slope(SHARED)` is therefore the per-M-tile cost of the
    wider declaration, and V5 requires it to be small against slope(PRIVATE):
    that is what licenses reading the machinery-matched ratio as a statement
    about the path the study fits. A large difference does not refute the
    ratio; it says the ratio is about a call the study does not make, which is
    a VALIDITY failure and not a claim failure.
  * The n = 1 TREAD bounds the INSTRUMENT. At one M-tile per expert SHARED and
    PRIVATE route every tile to copy 0 -- the relabelling is the identity --
    so they are the SAME CALL, and the gap between their timings is this arm's
    own floor for a slope comparison, measured rather than imported. V6
    requires it to be small. NATIVE is not in that comparison: it differs
    from SHARED by a constant declaration at every tread including n = 1, and
    its n = 1 offset is printed as a record. IT IS A PLAN-TIME REQUIREMENT,
    not a post-hoc one: a model whose routing cannot form `r = 1 x BLOCK_M` as
    an integer token count is REFUSED before a pod is rented, and
    `deepseek-v2-lite` (rows_quantum 3) is exactly such a model. See
    `identity_tread_refusal`.
  * THE CLOCK. The card is power capped and its clock is an outcome of the
    work, and PRIVATE moves more bytes per call than SHARED. V7 requires the
    two arms' under-load clocks to agree at every fitted tread, so a ratio
    that carries a clock difference is refused rather than published.
  * THE ALIGNMENT KERNEL. vLLM builds the sorted-id table with one of two
    kernels, chosen by the id count and the declared expert count (the cited
    `ALIGN_*` constants), and for this model at this tile the switch falls
    INSIDE the ladder. A step in the per-call time that a straight-line fit
    reads as slope -- in both ratio arms, or in ONE of them, which is why
    the bound V8 scores assumes neither -- pulling the ratio toward 1.
    Three things are done about it, kept apart: the RATIO ARMS ARE MOVED OFF
    THE SWITCH by construction (`declared_copies_for` declares enough copies
    that `E x n_decl` clears the expert bound, so both take the scan path at
    every tread, and the census REFUSES a plan where they do not); the switch
    is MEASURED on the attached build by `probe_alignment`, timing the
    alignment op alone along the ladder once per ARM (NATIVE's declaration,
    and SHARED's and PRIVATE's id sets at the ratio arms') UNDER A CUDA
    GRAPH (`PROBE_CALLS_PER_REPLAY` calls per replay, so each cell is GPU
    time and no host enqueue sits in it; session 4's eager probe on the H200
    was host-bound 36 of 36 and read UNKNOWN), and V8 refuses
    the design if the ratio arms' own series carries a step worth more than
    `ALIGN_STEP_RATIO_BUDGET` of the ratio; and NATIVE, which keeps the
    study's declaration and so keeps the switch, has its ladder difference
    from SHARED fitted WITH a step term (`declaration_fit`), so V5 scores the
    per-tile cost of the declaration with the step taken out, and the step
    itself is a reported number with an interval.
  * NOTHING BOUNDS THE FOOTPRINT ITSELF, and the page says so rather than
    implying otherwise. A cost that (a) vanishes at one copy and (b) is paid by
    READING the copies rather than by declaring them -- a TLB or DRAM-page
    locality cost is exactly that shape -- is invisible to NATIVE, which does
    not read them, and to n = 1, where there is only one. A tenth of an
    inflation on the deepest private cell moves the headline from
    REFIT-CONFIRMED to BELOW-THE-REFIT-BAND with V5 reporting 0.00% and every
    validity gate PASS. The one trace such a cost leaves on this page is the
    PRIVATE ladder's own `mean_rel_err`: a footprint cost that grows with the
    resident copies is the one thing that makes that ladder non-affine, and it
    is printed per arm in the ladder table. NO GATE READS IT, and gating it, or
    adding a fourth arm that spreads the addresses at fixed traffic, are the
    two ways to close this and are the owner's to choose.

THE ASSUMPTION THE RATIO KEEPS, REGISTERED. `slope(SHARED)/slope(PRIVATE)`
removes the assumed RATE -- no bandwidth and no intercept enter the number --
and it is alpha as a traffic fraction only if the two arms deliver the same
bytes per second. The shared arm's non-re-read bytes come from L2, which is not
free, so time is proportional to DRAM traffic in both arms only if that
proportionality holds, and the corpus rescore's PHYSICS finding -- 4 of 40
published alphas implying more bandwidth than their own card's pin rate,
worst 2335 GB/s against the A100's 2039 -- is the case where it does not. This
arm removes the assumed rate and keeps the assumption that the two arms SHARE
one. That is weaker than assuming a number and it is not nothing.

WHAT A RATIO MEANS, PARTITIONED BEFORE THE RUN. `OUTCOMES` below is one
ordered table covering [0, inf) with no gap and no overlap, read by the
prediction page and by the report, so the world the measurement lands in is
NAMED by this script and not left to a reader:

    ratio < 0.35              ISSUE-AND-LATENCY. Most of the per-M-tile cost
                              is not weight traffic at all. The traffic model
                              is the wrong KIND of model and the study's
                              negative result becomes a positive one.
    0.35 .. ALPHA_BAND[0]     BELOW THE REFIT BAND.
    ALPHA_BAND                REFIT CONFIRMED: the fitted 0.558 is a traffic
                              fraction after all, measured without a rate.
    ALPHA_BAND[1] .. 0.85     ABOVE THE REFIT BAND.
    ratio >= 0.85             NO REUSE. Essentially the whole weight set is
                              re-read per M-tile; the re-read is real and the
                              model's SHAPE was right even where its
                              coefficient was not identifiable from timing.

THE BY-PRODUCT, and it is not small. `slope(PRIVATE)` divided by the card's
calibrated weight-stream time is `w(PRIVATE)`, which SHOULD be 1.0 exactly if
the private arm re-reads the whole set once per tile and achieves the
calibrated rate. Read the other way round, `weight_bytes / slope(PRIVATE)` is
the DELIVERED bandwidth of the grouped GEMM's weight read, measured inside the
kernel under test with no triad benchmark in it. C2 scores the relation
between that and the card's own calibrated ceiling. A kernel cannot beat its
card's measured ceiling, so a violation says the copies were not all read (and
V2 says they were) or the calibration is not a ceiling.

WHY THE CLOCK SPLIT EXISTS AND HOW IT IS REMOVED (DESIGN DECISION 15). Session
4's pages were INVALID on V7 because a power-capped card boosts whichever arm
reads fewer bytes: both arms ran flat out against the 700 W cap and the shared
arm sat 2% (G=1) to 18% (G=16) above the private one. `--duty D` times every
cell as bursts of about 40 ms of kernel time, each followed by an idle gap of
`burst x (1/D - 1)`, the mechanism `clock_elasticity.time_duty` already has
(imported, not copied), so the same kernel moves the same bytes at a lower
AVERAGE board power. Whether a duty is low enough that the clock stops
following power is a measured fact about the card, and session 4's clock arm
measured it (2026-09-21, the G=16 mixtral arm, run 9f91fa91 under
results/published/2026-09-21-nvidia_h200-session4/ on the pod-h200-session4
branch): at duty 0.5 the clock still tracked board power, -1.09 MHz/W over
1882-1965 MHz, and 13 of its 104 cells failed DRIFT (16.7% of those at treads
1-6); at duty 0.25 every tread's median clock read 1965 MHz, the slope was
-0.07 MHz/W and no cell drifted. SO THE POD SETTING IS `--duty 0.25`. There V7
is expected to hold if both arms' average power stays where session 4's clock
was flat (the clock arm's kernel, the study's own call, drew 277-317 W at
0.25; the private arm reads more bytes and draws more), and V7 checks it: a
FAIL below full duty is a duty not yet low enough, and its remedy names a
lower one. When it holds the raw ratio is quotable and no elasticity model
enters the number. The cost is wall clock, about 1/duty x the kernel time.
THE DEFAULT STAYS 1.0: `duty` is one of `DESIGN_KEYS`, read as 1.0 from a
report that predates it, so a different default would make every bare
command a different design from session 4's full-duty runs, which
`--replicate-of` then refuses, and would move their run ids. Board power is
recorded per cell (`power_w`, NVML's ~1 s average, so below full duty an
average over the bursts and the gaps together) and V7 prints each arm's
median per tread beside its clocks: the traffic signal that does not go
through the clock. Beside it goes each arm's median in-burst sag, the change
inside a burst that one clock read per burst cannot see. Both are scored by
nothing.

WHAT ONE RUN'S INTERVAL IS NOT. The bootstrap is over repeats WITHIN a run. Two
runs of this arm on one card at two seeds, 77 minutes apart (session 4), sat
further apart than either interval was wide, so a single run's C1 PASS is a
within-run statement and the page says so until a second run is read beside
it with `--replicate-of`, when C1 is scored on the ENVELOPE of the runs'
intervals (DESIGN DECISION 14).

WHAT THIS ARM DOES NOT DO. It does not fit `alpha_a`, it does not separate the
activation term from the weight term, and it does not decide anything about
BLOCK_M other than the one it is run at. The ratio is scored RAW -- both slopes
carry the same activation and compute terms, and dividing them subtracts
nothing -- because being model-free is the entire reason this ladder was worth
renting a card for. Two corrected ratios are PRINTED beside it, each under the
model it assumes, and both are scored by nothing: the activation-corrected
ratio, and (when `--clock-elasticity` names a measured PER-M-TILE
elasticity and its source) the CLOCK-corrected ratio, every ratio-arm cell
carried to one reference clock by `(f_cell / f_ref) ** eta` before the fit
(exact when each arm holds one clock at every fitted tread, first-order when
a clock moves across treads; CLOCK_PARITY's comment has why). V7 still
refuses a clock split of more than CLOCK_PARITY. At FULL duty on a pod that
cannot lock its clock (RunPod refuses `nvidia-smi -lgc`) the arm reading less
draws less power and clocks higher, so V7 fails by construction there and the
page says so while it prints what the correction would read; below the duty
at which the clock stops following power (DESIGN DECISION 15) it is expected
to hold, and it is scored either way.

EXIT CODES are `moe/bench/exit_codes.py`'s table and nothing is folded into
DONE. There is deliberately no gate-softening flag: a CLAIM that did not pass
returns 1, which the ledger already reads as a finished result, and the
clock correction is not one either -- it changes no verdict.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# `scripts/` is not a package. The sibling sweep is imported by putting its own
# directory on the path and registering it under its module name, which is what
# every other runner in this directory does and what `@dataclass` needs when it
# resolves annotations through `sys.modules[cls.__module__]`.
sys.path.insert(0, str(HERE))

import block_m_crossing_sweep as SWEEP  # noqa: E402

from moe.bench import exit_codes, roofline  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench import weights as WEIGHTS  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

VALIDITY, CLAIM = exit_codes.VALIDITY, exit_codes.CLAIM
PASS, FAIL, UNKNOWN = exit_codes.PASS, exit_codes.FAIL, exit_codes.UNKNOWN

# --------------------------------------------------------------------------
# THE ARMS. Three names, used as a CSV column, as a dict key and in every
# printed table, declared once so a typo is an import error and not a silently
# empty ladder.
# --------------------------------------------------------------------------

NATIVE = "native"
SHARED = "shared"
PRIVATE = "private"

#: In the order they are printed and fitted. The rotation the sweep runs them
#: in is derived from this tuple, never listed a second time.
ARMS: tuple[str, ...] = (NATIVE, SHARED, PRIVATE)

#: The pair the ratio is formed from. They share the declaration, the tensors,
#: the launch grid and the tile order; only the addresses differ.
RATIO_ARMS: tuple[str, str] = (SHARED, PRIVATE)

ARM_MEANING = {
    NATIVE: "E experts declared over w1[::n_decl], copy 0 of each expert: the "
            "call this study fits, reading SHARED's bytes at SHARED's addresses",
    SHARED: "E x n_decl experts declared, every M-tile of expert e routed to "
            "copy 0: PRIVATE's call with one copy read (the numerator)",
    PRIVATE: "E x n_decl experts declared, M-tile j of expert e routed to copy "
             "j: reuse across M-tiles impossible by construction (the "
             "denominator)",
}

# --------------------------------------------------------------------------
# Everything this script argues about, before any of it is used. The study's
# constants are IMPORTED, never restated: a local copy of alpha that drifted
# from the sweep's would put the two files in different worlds while both
# printed a confident table.
# --------------------------------------------------------------------------

ALPHA = SWEEP.ALPHA                       # 0.558, refit 2026-08-31
ALPHA_BAND = SWEEP.ALPHA_BAND             # (0.529, 0.588), the 90% band
RETRACTED_ALPHA = SWEEP.RETRACTED_ALPHA   # 0.10, the world the study retracted

#: DESIGN DECISION 1. The default model is mixtral-8x7b and not
#: deepseek-v2-lite, and the reason is the n = 1 control, not the footprint.
#: deepseek-v2-lite has E=64, k=6, so `rows_quantum` is 3 and rows per expert
#: must be a multiple of 3; `r = n x BLOCK_M` at any power-of-two BLOCK_M is
#: never a multiple of 3 at n = 1, so THE ONE TREAD WHERE THE THREE ARMS ARE
#: ONE PHYSICAL SITUATION CANNOT BE FORMED ON THAT MODEL. mixtral has
#: rows_quantum 1, so n = 1, 2, 3, ... all exist; its routed weight set is
#: 2.8186 GB against deepseek-v2-lite's 1.1073 GB, which is the figure the
#: memory plan prints and gates rather than a figure this file asserts. It is
#: also the model every `w` in the 2026-09-10 synthesis is quoted on, and its
#: declared expert space stays small (8 x 9 = 72 at the default declaration,
#: which is `n_max` = 6 padded to 9 by `declared_copies_for`) where
#: deepseek-v2-lite's would reach 768, far outside anything
#: `moe_align_block_size` is exercised at in this repository.
#: `--model deepseek-v2-lite` still runs; it refuses at plan time with the
#: quantum arithmetic printed, and `identity_tread_refusal` is that refusal.
DEFAULT_MODEL = "mixtral-8x7b"

#: DESIGN DECISION 2. One BLOCK_M per run, 32 by default, and it is in the run
#: id. NOT 64, and the reason is `discrimination_floor`, not taste: the compute
#: per M-tile scales with the tile height while the weight traffic per M-tile
#: does not, so a taller tile runs into its compute ceiling at a shallower
#: tread in exactly the world this arm exists to be able to find. Computed off
#: GPU at this repo's H200 ridge and triad, the deepest tread whose SHARED
#: ladder is still memory bound is:
#:
#:     BLOCK_M     alpha=1.0   alpha=0.558   alpha=0.10   alpha=0.05
#:         16        deeper      deeper        deeper          18
#:         32        deeper      deeper             8           6
#:         64        deeper      deeper             2           2
#:
#: "deeper" is deeper than the search cap, not a depth. THE TABLE IS A FROZEN
#: COPY OF A RIDGE-AND-BANDWIDTH COMPUTATION and can go stale silently here;
#: the code RECOMPUTES it at the run's own ridge and triad and prints it on the
#: plan page, and `depth_lines` is the line to read. This comment is why the
#: default is what it is, not a number anything scores against.
#:
#: At 64 the ISSUE-AND-LATENCY world is unmeasurable past tread 2 -- V4 would
#: void the page in the one outcome that would turn the study's negative result
#: into a positive one -- and a design that cannot resolve one of its own
#: registered outcomes is not a design. 32 keeps the whole planned ladder
#: memory bound down to alpha ~ 0.05, is in the study's forced tile set, and is
#: a tile alpha is quoted at. `--block-m 16` is safer still and `--block-m 64`
#: is available; the plan page prints the floor for whichever is chosen and
#: REFUSES when the registered alternative world falls below it.
#: A second tile is a second run and a second directory, not a second column.
DEFAULT_BLOCK_M = 32

#: DESIGN DECISION 3. Six treads, n = 1 .. 6. The deepest tread sets the memory
#: bill (`n` complete copies) and the ladder's lever arm at once. Six rather
#: than the eight the depth table above allows at BLOCK_M=32: eight is the
#: H200's own number and the A100's is seven, so eight is a design that passes
#: its own depth check by one tread on one card. Six copies of mixtral bf16 is
#: 16.9 GB, which the plan checks against the attached card rather than against
#: this comment.
DEFAULT_TREADS = 6

#: DESIGN DECISION 4. Nine repeats of the whole ladder, repeats OUTER, arms
#: rotated within a tread. Nine because the interval on the ratio is a
#: bootstrap over repeats and the arm is cheap: the whole sweep is minutes of
#: kernel time. Repeats outer and arms rotated is what keeps a thermal or
#: governor drift from landing on one arm: with arms innermost and a fixed
#: order, SHARED would be first in every triple for the whole run.
DEFAULT_REPEATS = 9

#: DESIGN DECISION 5. The interval is a percentile bootstrap over repeats at
#: 90%, matching `ALPHA_BAND`'s own convention, so C1 compares two intervals
#: of the same kind (it is no longer an overlap test: see `c1_verdict`).
DEFAULT_DRAWS = 2000
INTERVAL_PCT = 90.0

#: DESIGN DECISION 14. THE INTERVAL ABOVE IS WITHIN ONE RUN AND IS NOT THE
#: RUN-TO-RUN SPREAD. Two G=1 runs of this arm on one H200, one tree, 77
#: minutes apart, differing only in --seed (the session-4 pair under
#: results/published/2026-09-21-nvidia_h200-session4/, on its own branch),
#: read ratios whose within-run intervals do not overlap and whose points sit
#: further apart than either interval is wide; the seed-0 point also sat at
#: the 5th-percentile edge of its own interval, which is that bootstrap
#: flagging itself as skewed. So `--replicate-of` reads one or more earlier
#: report.json files of the SAME design (`DESIGN_KEYS`), prints the cross-run
#: spread and the ENVELOPE of the intervals beside this run's own, and C1 is
#: then scored on the envelope (`CrossRun.verdict`, built on `c1_verdict`):
#: PASS needs every run's point in ALPHA_BAND and the whole envelope inside
#: it. OPT-IN: a run with no replicate named is scored alone, as before, and
#: the page says so. `--read` re-scores a stored report the same way with
#: nothing measured and nothing written, so a pair's spread has a committed,
#: recomputable source. The spread itself is NOT typed here: `--read` over the
#: committed pair prints it. A replicate is the same DESIGN, not the same
#: bytes: `--seed` draws the weights and the inputs as well as the bootstrap.
DESIGN_KEYS: tuple[str, ...] = ("experiment", "card", "model", "dtype",
                                "block_m", "pinned", "treads", "repeats",
                                "copies_declared", "alpha_band", "duty")
#: What a report written before a design key existed is read as carrying.
DESIGN_KEY_DEFAULTS: dict[str, object] = {"duty": 1.0}

#: DESIGN DECISION 6. V5's bound on the declaration. `|slope(NATIVE) -
#: slope(SHARED)|` must be under this fraction of `slope(PRIVATE)`, which is
#: the denominator the ratio is formed against, so the number bounds how far
#: the machinery-matched ratio can sit from the one the study's own call would
#: give, in the ratio's own units. SCORED ON THE FAR EDGE of `b`'s bootstrap
#: band, not on the point: the bound is a claim about how far the ratio CAN
#: sit, so the band's worst edge is what it has to hold for.
#:
#: 0.03, CHOSEN, and it was 0.10. A tenth admitted a machinery error in the
#: ratio larger than ALPHA_BAND's entire 0.059 width, i.e. a V5 PASS could
#: coexist with a declaration cost big enough to carry the ratio from one
#: registered world to the next. Three hundredths is half the band's width.
MACHINERY_BOUND = 0.03

#: What V5 asks for, written once: three call sites printed it and step 5
#: moved only one of them to the far-edge rule.
MACHINERY_WANT = (f"PASS: the FAR edge of b's {INTERVAL_PCT:.0f}% band < "
                  f"{MACHINERY_BOUND:.0%} of the private slope. FAIL: its "
                  "NEAR edge is over that bound, i.e. the whole band is. "
                  "Otherwise UNKNOWN")

#: DESIGN DECISION 7. V6's bound on the instrument. At n = 1 SHARED and
#: PRIVATE are the same call; the relative gap between their medians must be
#: under this. 2% is above the spread of a MEASURED TREAD TIME across cold
#: replicates in the one committed arm that has any -- 0.16% median and 1.18%
#: worst over the 188 treads of `results/published/2026-09-10-nvidia_h200-
#: gaps-session/results/gaps-nvidia_h200/replicate_noise_floor/nvidia_h200-
#: fresh-n3`, the same 0.2-1.0% band `bm128_roofline.py` quotes -- and well
#: under the smallest effect the ratio has to resolve (the 0.35 / 0.529
#: boundary is 0.18 wide). The "0.115% cold-replicate" and "0.37% cross-
#: session" figures this line used to name are the session-3 analysis's, hold
#: in no committed file, and are spreads of a FITTED SLOPE, not of the
#: per-call median V6 compares. The run-to-run spread of the RATIO itself is
#: DESIGN DECISION 14's and is read from committed reports by --read; it is a
#: spread of a different quantity and the two are never compared.
IDENTITY_SPREAD = 0.02

#: DESIGN DECISION 11. V7's bound on the clock. At every fitted tread the
#: median under-load SM clock of PRIVATE must sit within this fraction of
#: SHARED's. CHOSEN, a tolerance and not a quantity derived from any card, and
#: the reason is first-order arithmetic: time that is SM cycles over the clock
#: has an elasticity of 1 to it, so a 1% clock gap moves such a cell's time by
#: about 1%, and a slope that moved in proportion would move the ratio by
#: about 0.01, a sixth of ALPHA_BAND's width. THE SLOPE NEED NOT MOVE IN
#: PROPORTION, and session 4 says it did not: its clock arm's cells (run
#: 9f91fa91, 2026-09-21, G=16), re-scored by `clock_elasticity.fit` at
#: 12ec932, read the per-M-tile elasticity at 1.21 over treads 1-8 and 1.11
#: over treads 2-8 against a pooled per-call 0.74, because the intercept
#: follows the clock less than the tiles do. Both per-M-tile readings are
#: above 1, so ~0.01 is a first-order figure and not a bound, and 1% stays
#: the tolerance.
#:
#: WHICH ELASTICITY `--clock-elasticity` TAKES: THE PER-M-TILE ONE.
#: `clock_corrected` scales each cell's whole per-call ms by
#: `(f / f_ref) ** eta`, and the ratio reads nothing but the two SLOPES. When
#: each arm holds one clock at every fitted tread, the factor is one number
#: per arm: the intercept's share of it stays in the intercept, the slope is
#: carried by exactly that factor, and the corrected ratio is exact when eta
#: is the elasticity of the per-M-tile cost. That is `clock_elasticity`'s
#: gated claim, `elasticity.value` in a report written since 8d4eb78
#: (2026-09-22), over treads 2 and deeper by the owner's decision D2 of
#: 2026-09-22 (that file's `Elasticity` docstring says which treads `value`
#: spans). NOT its pooled per-call reading, `elasticity.fixed_tread`, which
#: is also what `elasticity.value` holds in a report written before 8d4eb78,
#: session 4's included (0.7436 [0.7277, 0.7559] at G=16): that blends the
#: one-tile call's 0.17 with the deeper treads' 1.04-1.11 on the same cells,
#: and it is the elasticity of no slope.
#: When a clock moves ACROSS treads inside one arm, as at full duty where
#: power follows the tread, the intercept's clock term leaks into the slope
#: and no single eta is exact (two elasticities would be; this correction
#: takes one), so there the corrected ratio is first-order only. The page
#: PRINTS it beside the raw one. It never scores it, and it never moves this
#: bound: at FULL duty on a card that cannot lock its clock the two arms draw
#: different power and V7 refuses by construction, which is a fact about the
#: platform the page should state, not soften; below full duty DESIGN
#: DECISION 15 picks a duty at which session 4's clock no longer followed
#: power.
CLOCK_PARITY = 0.01

#: DESIGN DECISION 8. V3's two tolerances. The weight allocation is an exact
#: arithmetic prediction -- `n_decl x E x 3FH x bytes` -- so it is gated tight;
#: the high-water mark includes the framework's own intermediates, which this
#: file predicts by a stated ALLOWANCE and therefore gates one-sided.
WEIGHT_ALLOC_TOLERANCE = 0.01
#: The multiplier on the modelled activation working set, as an allowance for
#: vLLM's intermediate caches, the sorted-id tables and the allocator's own
#: rounding. Stated as a number here because it is a DESIGN choice, and the
#: arithmetic it multiplies is printed on the plan page in full.
ACTIVATION_ALLOWANCE = 3.0

#: DESIGN DECISION 9. C2's tolerance. The delivered weight-read rate implied
#: by slope(PRIVATE) is compared with the card's own calibrated ceiling as a
#: RELATION, never against a literal: a kernel may not exceed its card's
#: measured ceiling by more than this. The number is a tolerance on a
#: comparison, not a quantity derived from a calibration.
ACHIEVED_RATE_TOLERANCE = 0.10

#: DESIGN DECISION 10. The fraction of the card's free memory the predicted
#: peak may occupy before the plan REFUSES. Two thirds leaves the allocator
#: room to fragment and leaves the pod usable for the arm that follows.
MEMORY_HEADROOM = 0.66

#: A tread needs this many M-tiles-per-expert points before a slope may be
#: quoted. The sweep's own reason applies unchanged: two points make a line
#: with no residual, so a two-tread fit cannot notice that one of its points
#: was wrong.
MIN_TREADS = SWEEP.MIN_MEMORY_TREADS      # 3

#: A tread whose achieved throughput reaches this fraction of the fixed roof is
#: compute bound, and V4 refuses the whole page if any fitted tread of SHARED
#: or PRIVATE is. Imported: a second copy of the sweep's threshold would
#: disagree with it one day.
COMPUTE_BOUND_FRACTION = SWEEP.COMPUTE_BOUND_FRACTION   # 0.95

#: Repeats a cell needs before its median is a median.
MIN_REPEATS = 3

#: The share of TIMED cells a DRIFTING clock may exclude before V0 calls the
#: kept set a subsample the card chose rather than the grid that was planned.
#:
#: WHY THERE IS A CEILING AT ALL RATHER THAN "ZERO DRIFTED CELLS". V0 used to
#: require `len(usable) == planned`, so ONE drifted cell in 162 -- on a card
#: this apparatus documents as power-capped with an endogenous clock, where
#: DRIFT exclusion is the study's NORMAL membership rule -- latched INVALID
#: over a run that met every floor it states. `alias_ablation` is written
#: against exactly this failure: 16 of that H200 run's 20 rungs drifted on the
#: same arithmetic, and excluding them left four rungs, six UNKNOWN gates and
#: exit 3. Every other non-vacuity gate in this tree is a counts-above-zero
#: gate. The floors below (usable treads and repeats per arm) are what V0
#: scores; the drifted COUNT is printed on its own line and fails only past
#: here. A fifth is the figure `clock_elasticity` uses for the same question at
#: a quarter of it, and it is chosen rather than derived: overrule it.
MAX_DRIFT_FRACTION = 0.20

#: The card slug a run id carries when no device is attached.
NO_CARD_SLUG = SWEEP.NO_CARD_SLUG

#: What `--self-test` writes into the provenance block's `instrument`, so a
#: planted report cannot satisfy a presence check while naming an instrument it
#: never touched.
SYNTHETIC_INSTRUMENT = SWEEP.SYNTHETIC_INSTRUMENT

#: The instrument this arm times with at full duty. Imported from the sweep so
#: a rename lands here as a refusal rather than as a row with an empty column.
def timing_basis() -> str | None:
    return SWEEP.timing_basis()


def ladder_instrument(duty: float, *, synthetic: bool = False) -> str | None:
    """The instrument the LADDER CELLS are timed with at `duty`, as the
    provenance block and report.json's top-level `instrument` name it.

    TWO INSTRUMENTS, ONE KNOB (DESIGN DECISION 15). At full duty it is
    `timing_basis()`, the queue-deep `time_kernel` loop. Below it the cells
    are timed by `clock_elasticity.time_duty`, whose own `INSTRUMENT` string
    says it is NOT `timing.TIMING_BASIS`, and naming the queue-deep loop there
    would put a duty-cycled page beside a full-duty one as one instrument.
    Below full duty this is that string, the one every row's `instrument`
    column carries, followed by the duty every row's `duty` column carries.
    None off-torch, as `timing_basis` is: `clock_elasticity` imports the
    timing module, which imports torch.
    """
    if synthetic:
        return SYNTHETIC_INSTRUMENT
    if duty >= 1.0:
        return timing_basis()
    try:
        import clock_elasticity as CE  # scripts/ is on sys.path, as SWEEP is
    except Exception:                                     # noqa: BLE001
        # Broad for `timing_basis`'s reason: an installed, broken torch raises
        # OSError, and naming the instrument is never worth the report.
        return None
    return f"{CE.INSTRUMENT} | duty {duty:.2f}"


# --------------------------------------------------------------------------
# Refusals. Typed, so "this cannot be measured" and "this measured nothing" are
# distinguishable by a caller and by the exit code.
# --------------------------------------------------------------------------

class PrivateWeightRefusal(RuntimeError):
    """The arm declines to produce a number rather than produce one."""


class SchemaCollision(PrivateWeightRefusal):
    """A cells.csv on disk whose header is not the one this build writes.

    `csv.DictWriter` writes the fieldnames it was given and never looks at the
    file, so a wider row appended under a narrower header shifts every field
    past the first difference and nothing downstream can tell. Raised at the
    Store, converted to REFUSED at the call site.
    """


class Unmeasurable(PrivateWeightRefusal):
    """A quantity the samples cannot support. Never substituted with a default."""


# --------------------------------------------------------------------------
# THE OUTCOME PARTITION. One ordered table, covering [0, inf) with no gap and
# no overlap, registered before the run and read by BOTH the prediction page
# and the report. Two copies of this partition is how a prediction page and a
# report come to name two different worlds for one number.
# --------------------------------------------------------------------------

#: The boundary below which the per-M-tile cost is not weight traffic. 0.35 is
#: the "near 0.3" world this arm was commissioned to be able to find, with room
#: for the activation term the ratio does not subtract. That term is NOT a
#: number in this comment because it moves with the tile: it is
#: `E x BLOCK_M x (2H + 3F) x bytes` over one weight stream, which the plan
#: page prints per run (0.93% of a stream per M-tile at the default tile,
#: 1.86% at twice it).
ISSUE_BOUND_MAX = 0.35

#: The boundary above which the weight set is, to within the arm's own floor,
#: re-read whole per M-tile. 0.85 is the brief's registered "near 1.0" world
#: with room for the same activation term and for a machinery bound at its V5
#: limit.
NO_REUSE_MIN = 0.85

#: `(name, lo, hi, meaning)`, half-open `[lo, hi)`, in order. The last row is
#: closed at infinity.
OUTCOMES: tuple[tuple[str, float, float, str], ...] = (
    ("ISSUE-AND-LATENCY", 0.0, ISSUE_BOUND_MAX,
     "most of the per-M-tile cost is NOT weight traffic: giving every M-tile "
     "its own copy barely moved the time. The traffic model is the wrong KIND "
     "of model, and the study's negative result about alpha becomes a positive "
     "result about the mechanism."),
    ("BELOW-THE-REFIT-BAND", ISSUE_BOUND_MAX, ALPHA_BAND[0],
     "the re-read is real but smaller than the refit says: between the two "
     "registered worlds, and the page says so rather than rounding to one."),
    ("REFIT-CONFIRMED", ALPHA_BAND[0], ALPHA_BAND[1],
     "the refit 0.558 is a traffic fraction after all, measured here without "
     "an assumed rate and without a fitted intercept."),
    ("ABOVE-THE-REFIT-BAND", ALPHA_BAND[1], NO_REUSE_MIN,
     "the re-read is larger than the refit says: between the two registered "
     "worlds, and the page says so rather than rounding to one."),
    ("NO-REUSE", NO_REUSE_MIN, math.inf,
     "essentially the whole weight set is re-read per extra M-tile. The "
     "re-read is real and the traffic model's SHAPE was right even though its "
     "coefficient was not identifiable from timing."),
)


def outcome_for(ratio: float) -> tuple[str, str]:
    """`(name, meaning)` for a ratio. Total over the reals, refusing nothing.

    A negative ratio is a real state -- one of the two ladders got FASTER with
    another M-tile, which says its branch membership is wrong and not that the
    ratio is small -- so it is named rather than clamped into the first band.
    """
    if not math.isfinite(ratio):
        return ("NOT-A-RATIO",
                "one of the two slopes is not finite, so no ratio was formed")
    if ratio < 0.0:
        return ("DESCENDING",
                "the ratio is NEGATIVE: one of the two ladders got faster with "
                "another M-tile, so it is not a fraction of anything and its "
                "branch membership is what to look at")
    for name, lo, hi, meaning in OUTCOMES:
        if lo <= ratio < hi:
            return name, meaning
    # `OUTCOMES` ends at infinity, so this is unreachable for a finite,
    # non-negative ratio; it is here so a future edit that leaves a gap fails
    # loudly instead of returning the last row by accident.
    raise Unmeasurable(f"ratio {ratio} fell through the OUTCOMES partition; "
                       "the table has a gap in it")


def partition_is_total() -> str:
    """"" when `OUTCOMES` tiles [0, inf) with no gap and no overlap, else why not.

    Checked at import by the test suite and printed by `--dry-run`, because the
    whole content of the table is that it has no gap: a ratio that fell into
    one would be named by whichever row happened to be tested last.
    """
    edge = 0.0
    for name, lo, hi, _meaning in OUTCOMES:
        if lo != edge:
            return f"{name} starts at {lo} where the previous row ended at {edge}"
        if not hi > lo:
            return f"{name} is empty: [{lo}, {hi})"
        edge = hi
    if edge != math.inf:
        return f"the table ends at {edge} and not at infinity"
    return ""


# --------------------------------------------------------------------------
# Geometry. Pure arithmetic over `moe.spec`, so every number on the plan page
# is checkable off GPU and by the test suite.
# --------------------------------------------------------------------------

def ladder_treads(cfg, block_m: int, max_treads: int) -> list[int]:
    """`n = 1 .. max_treads`, refusing a model whose routing cannot form them.

    Exactly-full tile stacks only: `r = n x BLOCK_M`, zero padding, one tread
    per `n`. A nudged row is not a full stack and a fit over partly-filled
    treads is a fit over padding.
    """
    quantum = SWEEP.rows_quantum(cfg)
    bad = [n for n in range(1, max_treads + 1) if (n * block_m) % quantum]
    if bad:
        raise PrivateWeightRefusal(
            f"{cfg.name} routes k={cfg.top_k} over E={cfg.num_experts}, so "
            f"rows per expert must be a multiple of {quantum}, and treads "
            f"{bad} at BLOCK_M={block_m} are not. See "
            "identity_tread_refusal for what that costs this design.")
    return list(range(1, max_treads + 1))


def identity_tread_refusal(cfg, block_m: int) -> str:
    """"" when `n = 1` is formable on this model at this tile, else why not.

    THE CONTROL IS PART OF THE DESIGN AND SO IS ITS FEASIBILITY. V6 reads the
    n = 1 tread, where shared and private are the same call, and a model
    that cannot form `r = 1 x BLOCK_M` as an integer token count has no such
    tread at any depth. Finding that out after renting a pod would cost the
    whole arm; finding it out here costs nothing and the message names the
    model that does have one.
    """
    quantum = SWEEP.rows_quantum(cfg)
    if block_m % quantum == 0:
        return ""
    return (
        f"{cfg.name} needs rows per expert to be a multiple of "
        f"{quantum} (k={cfg.top_k} over E={cfg.num_experts}), and one M-tile "
        f"is {block_m} rows, which is not. So the n = 1 tread -- the ONE tread "
        "where SHARED and PRIVATE are the same call, and "
        "the control that bounds this arm's own instrument floor (NATIVE is "
        "not in that comparison: it declares E where they declare E x n_decl) "
        "-- cannot be "
        "formed on this model at this tile at any depth. Run --model "
        f"{DEFAULT_MODEL} (rows_quantum "
        f"{SWEEP.rows_quantum(MODEL_CONFIGS[DEFAULT_MODEL])}), or choose a "
        f"--block-m that is a multiple of {quantum}, which no power of two is "
        f"when {quantum} is odd and greater than one.")


def modelled_roof_fraction(cfg, *, block_m: int, tiles: int, alpha: float,
                           ridge: float, bandwidth_gbps: float, b: int) -> float:
    """What fraction of the fixed roof a tread would reach in a world of `alpha`.

    The study's own `model_ms` -- `overhead + max(traffic, padded compute)` --
    divided into the useful FLOPs of the same tread. It is the MODEL and
    nothing that reads it is evidence for it; its job here is to say, before a
    pod is rented, whether the design can still see traffic at this depth in
    each world it registers.
    """
    rows = tiles * block_m
    ms = SWEEP.model_ms(cfg, rows, block_m, alpha=alpha, ridge=ridge,
                        bandwidth_gbps=bandwidth_gbps, b=b)
    if ms <= 0:
        return math.inf
    tflops = SWEEP.useful_flops(cfg, cfg.num_experts * rows) / (ms * 1e-3) / 1e12
    roof = ridge * bandwidth_gbps / 1e3
    return tflops / roof if roof > 0 else math.inf


def deepest_memory_bound_tread(cfg, *, block_m: int, alpha: float, ridge: float,
                               bandwidth_gbps: float, b: int,
                               limit: int = 256) -> int:
    """The deepest `n` whose SHARED ladder is still under the compute ceiling.

    THE DESIGN QUESTION THIS ANSWERS. The compute an M-tile does scales with
    the tile height and the weight traffic an extra M-tile costs does not, so a
    ladder in a LOW-alpha world climbs toward its roof as it deepens: at
    BLOCK_M=64 on mixtral at this repo's H200 ridge, a world of alpha=0.10 is
    compute bound from tread 3. A ladder fitted through such treads has a slope
    set by arithmetic, and the ratio it feeds moves toward 1.0 -- toward the
    NO-REUSE reading -- for a reason that has nothing to do with reuse. V4
    catches that AFTER the pod is paid for; this catches it before.
    """
    last = 0
    for n in range(1, limit + 1):
        frac = modelled_roof_fraction(cfg, block_m=block_m, tiles=n, alpha=alpha,
                                      ridge=ridge, bandwidth_gbps=bandwidth_gbps,
                                      b=b)
        if frac >= COMPUTE_BOUND_FRACTION:
            return last
        last = n
    return last


def discrimination_floor(cfg, *, block_m: int, treads: int, ridge: float,
                         bandwidth_gbps: float, b: int) -> float:
    """The LOWEST alpha at which this design's own shared ladder still sees traffic.

    Bisected on a monotone predicate: more re-read is more traffic is more
    memory bound, so "the whole planned ladder is under the compute ceiling" is
    monotone increasing in alpha. Below the number this returns the design
    cannot report an ISSUE-AND-LATENCY world at all, because its own shared
    ladder would be compute bound before the deepest tread and V4 would void
    the page. It is a property of the DESIGN and belongs on the plan page
    beside the depth, not in a post mortem.
    """
    def ok(alpha: float) -> bool:
        return deepest_memory_bound_tread(
            cfg, block_m=block_m, alpha=alpha, ridge=ridge,
            bandwidth_gbps=bandwidth_gbps, b=b, limit=treads) >= treads

    if ok(0.0):
        return 0.0
    if not ok(1.0):
        return math.inf
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def depth_refusal(cfg, *, block_m: int, treads: int, ridge: float,
                  bandwidth_gbps: float, b: int) -> str:
    """"" when the design can still discriminate its registered worlds, else why not.

    ASKED BEFORE THE POD IS RENTED. The registered alternative this arm exists
    to be able to FIND is the ISSUE-AND-LATENCY world, whose registered
    representative is the study's own retracted alpha. If this design's shared
    ladder would be compute bound at that alpha before the deepest planned
    tread, then a run in that world returns INVALID on V4 rather than the
    finding, and the pod minutes buy nothing.
    """
    floor = discrimination_floor(cfg, block_m=block_m, treads=treads,
                                 ridge=ridge, bandwidth_gbps=bandwidth_gbps,
                                 b=b)
    if floor <= RETRACTED_ALPHA:
        return ""
    deepest = deepest_memory_bound_tread(
        cfg, block_m=block_m, alpha=RETRACTED_ALPHA, ridge=ridge,
        bandwidth_gbps=bandwidth_gbps, b=b, limit=treads + 64)
    return (
        f"at BLOCK_M={block_m} on {cfg.name}, a world of alpha="
        f"{RETRACTED_ALPHA} puts the SHARED ladder over "
        f"{COMPUTE_BOUND_FRACTION:.0%} of the fixed roof from tread "
        f"{deepest + 1}, and this run plans {treads}. The design's "
        f"discrimination floor is alpha={floor:.4f}: below it the shared "
        "ladder is compute bound before the deepest tread, V4 voids the page, "
        "and the ISSUE-AND-LATENCY world -- the one outcome that would turn "
        "this study's negative result into a positive one -- cannot be "
        f"reported at all. Lower --treads to {deepest}, or lower --block-m: "
        "the compute an M-tile does scales with the tile height and the weight "
        "traffic an extra M-tile costs does not.")


def depth_lines(cfg, *, block_m: int, treads: list[int], ridge: float,
                bandwidth_gbps: float, b: int, alpha: float) -> list[str]:
    """The depth table, printed on the plan page in every registered world."""
    floor = discrimination_floor(cfg, block_m=block_m, treads=len(treads),
                                 ridge=ridge, bandwidth_gbps=bandwidth_gbps, b=b)
    out = ["depth       the deepest tread whose SHARED ladder is still under "
           f"{COMPUTE_BOUND_FRACTION:.0%} of the fixed roof, per world, at this "
           "run's own ridge:"]
    for label, a in (("no-reuse ", 1.0), ("refit    ", alpha),
                     ("retracted", RETRACTED_ALPHA)):
        deepest = deepest_memory_bound_tread(
            cfg, block_m=block_m, alpha=a, ridge=ridge,
            bandwidth_gbps=bandwidth_gbps, b=b, limit=len(treads) + 64)
        # THE SEARCH CAP IS NOT A MEASUREMENT. `deepest_memory_bound_tread`
        # returns its `limit` when the ladder never reaches the ceiling, and
        # the cap here is len(treads) + 64, so a world that is memory bound
        # forever printed "tread 70 against the 6 planned" and read as a
        # measured depth. Past the cap the honest word is "deeper than".
        cap = len(treads) + 64
        reached = f"{deepest:>3d}" if deepest < cap else f">{cap - 1}"
        out.append(f"            alpha={a:<5.3f} {label}  tread "
                   f"{reached:>4s}   against the {len(treads)} planned")
    out.append(f"            DISCRIMINATION FLOOR: alpha={floor:.4f}. Below "
               "it this design's own shared ladder is compute bound before the "
               "deepest tread, V4 voids the page, and no ISSUE-AND-LATENCY "
               "result can be reported. The arm can therefore separate the "
               f"registered worlds down to alpha={floor:.4f} and no further, "
               "and that is a property of the design and not of the card.")
    return out


def weight_bytes_total(cfg, dtype: str, copies: int) -> int:
    """Bytes of routed expert weights resident when `copies` copies are held.

    `copies x E x 3FH x bytes(dtype)`, with the per-copy term taken from
    `moe.bench.weights.routed_expert_weight_bytes` -- the module that owns the
    one multiplication -- rather than recomputed here. A second copy of a byte
    count is how two halves of a study come to divide by different
    denominators, and this file's whole argument is a division.
    """
    return copies * WEIGHTS.routed_expert_weight_bytes(cfg, dtype)


def activation_working_set(cfg, tokens: int, b: int) -> int:
    """A MODELLED activation working set for one cell, in bytes.

    `x` and `y` at `[T, H]`, plus the permuted stack at `[T k, 2F + F + H]`:
    the gate+up output, the activated half and the down output, which is what
    a fused grouped GEMM has live at once. It is an ALLOWANCE and is labelled
    one everywhere it is printed: vLLM's own intermediate caches, the sorted-id
    tables and the allocator's rounding are not modelled here, which is what
    `ACTIVATION_ALLOWANCE` is for.
    """
    dense = 2 * tokens * cfg.hidden_size * b
    permuted = (tokens * cfg.top_k
                * (2 * cfg.intermediate_size + cfg.intermediate_size
                   + cfg.hidden_size) * b)
    return dense + permuted


def flush_buffer_bytes() -> tuple[int, str]:
    """The L2 flush buffer `time_kernel` allocates, and where the size came from.

    A NAMED TERM AND NOT PART OF AN ALLOWANCE. It is a multiple of THIS
    device's own L2, asked of the instrument -- so it is a different number on
    a different card, and off a GPU box it is the fallback size, which is why
    the figure that stood here (a worked H200 example) did not match the page
    the docstring describes -- which is the same order as every activation
    buffer in the cell put together, and leaving it inside a fudge factor is
    how a memory prediction comes to be a memory prediction of nothing. ASKED
    of the instrument, never restated: `timing.flush_mb_for_device` owns the
    rule and a change there moves this line with it.
    """
    try:
        from moe.bench import timing
    except Exception as exc:                              # noqa: BLE001
        # Broad for the reason `timing_basis` is: an installed-and-broken torch
        # raises OSError on a missing libcudart, and a plan is never worth
        # taking down for it.
        return 0, (f"not asked: moe.bench.timing did not import "
                   f"({type(exc).__name__}); the flush buffer is NOT in the "
                   "prediction below")
    megabytes = timing.flush_mb_for_device()
    fallback = megabytes == timing.DEFAULT_FLUSH_MB
    return megabytes * 2 ** 20, (
        "timing.flush_mb_for_device(), "
        + ("this device's own L2 x 4" if not fallback else
           f"DEFAULT_FLUSH_MB, the fallback for a device that could not be "
           f"queried ({megabytes} MB)"))


@dataclass(frozen=True)
class MemoryPlan:
    """The device memory this arm commits to, and whether it fits.

    Built before any allocation and printed on the plan page. `fits` is None
    when there is no card and no `--device-memory-gb` to check against, which
    is NOT the same answer as True.
    """

    #: Copies ALLOCATED AND FILLED: what V3 prices. At or above `copies_read`.
    copies: int
    per_copy_bytes: int
    weight_bytes: int
    activation_bytes: int
    allowance: float
    flush_bytes: int
    flush_source: str
    predicted_peak_bytes: int
    device_free_bytes: int | None
    device_source: str
    headroom: float
    #: Copies the deepest tread READS. The difference is the padding that
    #: keeps the ratio arms on one alignment kernel (`declared_copies_for`).
    copies_read: int = 0

    @property
    def fits(self) -> bool | None:
        if self.device_free_bytes is None:
            return None
        return self.predicted_peak_bytes <= self.headroom * self.device_free_bytes

    def lines(self) -> list[str]:
        gb = 1e9
        out = [
            "memory      the arithmetic this arm commits to, before a byte is "
            "allocated:",
            f"            per copy of the routed expert weight set   "
            f"{self.per_copy_bytes / gb:9.4f} GB",
            f"            x {self.copies} copies declared and filled"
            + (f" ({self.copies_read} read at the deepest tread, "
               f"{self.copies - self.copies_read} never read)"
               if self.copies_read and self.copies_read != self.copies
               else " (all read at the deepest tread)")
            + f"  {self.weight_bytes / gb:9.4f} GB",
            f"            + modelled activation working set x {self.allowance:.1f} "
            f"allowance  {self.activation_bytes * self.allowance / gb:9.4f} GB",
            f"            + the instrument's L2 flush buffer          "
            f"{self.flush_bytes / gb:9.4f} GB  ({self.flush_source})",
            f"            = predicted peak                            "
            f"{self.predicted_peak_bytes / gb:9.4f} GB",
        ]
        if self.device_free_bytes is None:
            out.append("            against NO DEVICE and no --device-memory-gb: "
                       "the fit is NOT CHECKED, which is not the same answer as "
                       "fits")
        else:
            out.append(
                f"            against {self.device_free_bytes / gb:.1f} GB "
                f"({self.device_source}), of which this arm may take "
                f"{self.headroom:.0%} = "
                f"{self.headroom * self.device_free_bytes / gb:.1f} GB: "
                + ("FITS" if self.fits else "DOES NOT FIT"))
        return out


def memory_plan(cfg, dtype: str, b: int, copies: int, tokens_max: int,
                device_free_bytes: int | None, device_source: str,
                allowance: float = ACTIVATION_ALLOWANCE,
                headroom: float = MEMORY_HEADROOM,
                flush: tuple[int, str] | None = None,
                copies_read: int | None = None) -> MemoryPlan:
    per_copy = WEIGHTS.routed_expert_weight_bytes(cfg, dtype)
    weights = weight_bytes_total(cfg, dtype, copies)
    act = activation_working_set(cfg, tokens_max, b)
    flush_bytes, flush_source = flush if flush is not None else flush_buffer_bytes()
    return MemoryPlan(
        copies=copies, per_copy_bytes=per_copy, weight_bytes=weights,
        activation_bytes=act, allowance=allowance, flush_bytes=flush_bytes,
        flush_source=flush_source,
        predicted_peak_bytes=int(weights + allowance * act + flush_bytes),
        device_free_bytes=device_free_bytes, device_source=device_source,
        headroom=headroom,
        copies_read=copies if copies_read is None else copies_read)


# --------------------------------------------------------------------------
# THE RELABELLING. Pure, and the whole of the private arm's mechanism.
# --------------------------------------------------------------------------

def copy_slot(expert, copy, copies_declared: int):
    """The expert slot holding copy `copy` of expert `expert`: EXPERT-FIRST.

    `expert x n_decl + copy`. Named and single because the layout is read in
    four places -- the weight build, both relabellings and the buffer proof --
    and the defect this layout replaces was a copy-first layout (`copy x E +
    expert`), under which `moe_align_block_size`'s sort by expert id ran the
    private arm's tiles in a different order from the shared arm's. Works on
    ints and on torch tensors alike.
    """
    return expert * copies_declared + copy


def private_copy_index(ranks, block_m: int):
    """Which copy a routing slot reads, from its rank within its own expert.

    Split out and named because it is the one line the whole arm rests on:
    the `j`-th M-tile of an expert holds ranks `[j BM, (j+1) BM)`, so the copy
    a slot reads is its rank floor-divided by the tile height. Takes and
    returns anything supporting `//`, which is both a torch tensor and an int,
    so the test suite can check it without a device.
    """
    return ranks // block_m


def shared_topk_ids(ids, copies_declared: int):
    """Route every slot naming expert `e` to copy 0 of `e`, at slot `e x n_decl`.

    The shared arm's ids under the expanded declaration. Every M-tile of
    expert `e` lands in one expert slot, so the tiles are formed and ordered
    exactly as the native call forms them; only the slot NUMBER moved.
    """
    return copy_slot(ids, 0, copies_declared)


def private_topk_ids(ids, num_experts: int, block_m: int, rows_per_expert: int,
                     copies_declared: int):
    """Relabel `[T, k]` expert ids so every M-tile reads its own weight copy.

    Slot `(t, j)` currently naming expert `e` is renamed to
    `copy_slot(e, copy, n_decl)`, where `copy` is the slot's rank within expert
    `e` under flattened index order, divided by `BLOCK_M`. Expert `e` then
    splits into exactly `rows_per_expert / BLOCK_M` slots of exactly `BLOCK_M`
    rows each, CONTIGUOUS in slot number, so `moe_align_block_size`'s sort by
    slot runs expert 0's tiles, then expert 1's -- the shared arm's order --
    and builds the same number of M-tiles, with zero padding.

    THE ROW ORDER INSIDE AN EXPERT DOES NOT HAVE TO MATCH THE KERNEL'S SORT.
    If the alignment groups an expert's rows in a different order from this
    one (its parallel path need not be stable), the CONTENTS of a given M-tile
    differ between the two arms while the count, the height, the padding, the
    tile order and the arithmetic do not -- and every copy holds the same
    weights, so the layer's output does not differ either. `V2`'s
    output-equality part is bitwise for that reason, not because the rows are
    guaranteed to coincide.

    REFUSES an imbalanced histogram rather than rounding one, and a ladder
    deeper than the declaration: this arm is run on `balanced_ids`, whose
    per-expert count is exact by construction, and a histogram that is off by
    one row would put `rows_per_expert / BLOCK_M + 1` copies under one expert
    and read a slot belonging to the NEXT expert -- which, expert-first, is
    allocated memory holding the wrong weights and would not even crash.
    """
    import torch

    if rows_per_expert % block_m:
        raise PrivateWeightRefusal(
            f"{rows_per_expert} rows per expert is not a whole number of "
            f"BLOCK_M={block_m} tiles; this arm fits exactly-full stacks only")
    if rows_per_expert // block_m > copies_declared:
        raise PrivateWeightRefusal(
            f"{rows_per_expert // block_m} tiles per expert need that many "
            f"copies and only {copies_declared} are declared; expert-first, "
            "the extra tiles would read the next expert's weights")
    flat = ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts)
    if int(counts.min()) != rows_per_expert or int(counts.max()) != rows_per_expert:
        raise PrivateWeightRefusal(
            f"the routing histogram is not exactly {rows_per_expert} rows for "
            f"every one of {num_experts} experts (min {int(counts.min())}, max "
            f"{int(counts.max())}); the private relabelling needs an exact "
            "histogram, which balanced_ids produces by construction")
    order = torch.argsort(flat, stable=True)
    sorted_e = flat[order]
    starts = torch.cumsum(counts, 0) - counts
    ranks = (torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
             - starts[sorted_e])
    copies = private_copy_index(ranks, block_m)
    renamed = copy_slot(sorted_e, copies, copies_declared)
    out = torch.empty_like(flat)
    out[order] = renamed
    return out.reshape(ids.shape).to(ids.dtype)


def expert_space(num_experts: int, copies_declared: int) -> int:
    """`E x n_decl`: the expert count SHARED and PRIVATE declare at EVERY tread.

    Fixed, never `E x n` at tread `n`: the sorted-id buffer and the launch
    grid scale with the declaration, and a declaration that grew with the
    tread put a per-tread dead-launch term into one slope and not the other.
    `n_decl` is the deepest tread's copy count, or more when
    `declared_copies_for` pads it past vLLM's small-batch expert bound.
    """
    return num_experts * copies_declared


def declared_experts(arm: str, num_experts: int, copies_declared: int) -> int:
    """What `global_num_experts` an arm passes. One place, read by the sweep,
    the planted cells and the sample rows."""
    return num_experts if arm == NATIVE else expert_space(num_experts,
                                                          copies_declared)


def copies_read(arm: str, tiles: int) -> int:
    """How many distinct copies an arm's call READS at tread `tiles`."""
    return tiles if arm == PRIVATE else 1


# --------------------------------------------------------------------------
# THE ALIGNMENT PATH. vLLM builds the sorted-id table with one of two kernels,
# chosen by the id count and the declared expert count, and the switch sits
# INSIDE this ladder. What is cited, what is derived and what is measured are
# kept apart below.
# --------------------------------------------------------------------------

#: A HYPOTHESIS ABOUT THE POD'S BUILD, cited and not measured. vLLM v0.27.1,
#: `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`, host function
#: `moe_align_block_size`:
#:
#:     bool small_batch_expert_mode =
#:         (topk_ids.numel() < 1024) && (num_experts <= 64);
#:
#: Under both bounds ONE block does the whole alignment; at or past either, a
#: two-block scan plus a sort kernel whose grid is `numel / 256` blocks. For
#: mixtral at BLOCK_M=32 the id count is 256 n, so an E=8 declaration crosses
#: the id bound between treads 3 and 4: a step in the per-call time that a
#: straight-line fit reads as slope. These numbers shape the PLAN and name the
#: split the NATIVE step is fitted at; the probe MEASURES where the step is on
#: the attached build, and V8 is scored on the probe and never on these.
ALIGN_SMALL_BATCH_MAX_IDS = 1024
ALIGN_SMALL_BATCH_MAX_EXPERTS = 64
#: `STD_TORCH_CHECK(padded_num_experts < 1024)`: the scan path assigns one
#: thread per expert padded to a warp and refuses past this.
ALIGN_MAX_PADDED_EXPERTS = 1024
ALIGN_WARP = 32
#: `fused_experts_impl` skips alignment entirely when
#: `num_tokens x top_k x 4 <= global_num_experts` (its "naive" assignment), a
#: THIRD path this arm must never be on: a different kernel launch.
NAIVE_ASSIGNMENT_SPARSITY = 4

SMALL_BATCH = "small-batch"
BLOCK_SCAN = "block-scan"


def align_path(numel: int, declared: int) -> str:
    """Which alignment kernel the cited hypothesis says a call takes."""
    if (numel < ALIGN_SMALL_BATCH_MAX_IDS
            and declared <= ALIGN_SMALL_BATCH_MAX_EXPERTS):
        return SMALL_BATCH
    return BLOCK_SCAN


def ids_for_tread(cfg, tread: int, block_m: int) -> int:
    """`topk_ids.numel()` at a tread: tokens x k, which is E x rows per expert."""
    return SWEEP.tokens_for_rows(cfg, tread * block_m) * cfg.top_k


def declared_copies_for(cfg, treads: list[int], block_m: int,
                        requested: int = 0) -> tuple[int, str]:
    """How many copies SHARED and PRIVATE declare, and why.

    At least the deepest tread's count, so every read copy has a slot. And
    when the ladder's id counts straddle the small-batch id bound while
    `E x n_max` is under the expert bound, ENOUGH MORE that `E x n_decl`
    clears the expert bound: both ratio arms then take the scan kernel at
    every tread and no step can enter either slope. The extra copies are
    allocated and filled like the others and never read; their price is
    memory and dead launches, and dead launches are a per-tread constant.

    `requested` overrides the rule and is refused below `n_max`. A request
    that leaves the ratio arms on the crossing is refused by the CENSUS, with
    the crossing named, not here.
    """
    n_max = treads[-1]
    e = cfg.num_experts
    if requested:
        if requested < n_max:
            raise PrivateWeightRefusal(
                f"--declared-copies {requested} is below the {n_max} copies "
                "the deepest tread reads; every read copy needs a slot")
        return requested, f"--declared-copies {requested}, the operator's choice"
    counts = [ids_for_tread(cfg, n, block_m) for n in treads]
    straddles = min(counts) < ALIGN_SMALL_BATCH_MAX_IDS <= max(counts)
    if straddles and e * n_max <= ALIGN_SMALL_BATCH_MAX_EXPERTS:
        n_decl = max(n_max, -(-(ALIGN_SMALL_BATCH_MAX_EXPERTS + 1) // e))
        return n_decl, (
            f"the ladder's id count runs {min(counts)}..{max(counts)}, across "
            f"the small-batch id bound {ALIGN_SMALL_BATCH_MAX_IDS}, and E x "
            f"n_max = {e * n_max} is under the expert bound "
            f"{ALIGN_SMALL_BATCH_MAX_EXPERTS}; declaring {n_decl} copies "
            f"(E x {n_decl} = {e * n_decl}) keeps both ratio arms on the scan "
            "kernel at every tread")
    if straddles:
        return n_max, (f"E x n_max = {e * n_max} already clears the expert "
                       f"bound {ALIGN_SMALL_BATCH_MAX_EXPERTS}: one kernel "
                       "throughout")
    return n_max, (f"the ladder's id count {min(counts)}..{max(counts)} does "
                   f"not cross {ALIGN_SMALL_BATCH_MAX_IDS}: one kernel "
                   "throughout")


@dataclass(frozen=True)
class PathCensus:
    """Every (arm, tread)'s alignment kernel under the hypothesis, and what the
    plan refuses on. Pure arithmetic, printed on the plan page."""

    #: `(arm, tread, tokens, numel, declared, path)`
    rows: tuple[tuple[str, int, int, int, int, str], ...]
    refusals: tuple[str, ...]

    def switch_tread(self, arm: str) -> int | None:
        """The first tread at which `arm`'s kernel differs from its first."""
        mine = [(n, path) for a, n, _t, _i, _d, path in self.rows if a == arm]
        for n, path in mine[1:]:
            if path != mine[0][1]:
                return n
        return None

    def lines(self) -> list[str]:
        out = ["alignment   which moe_align_block_size kernel each call takes "
               f"under the cited HYPOTHESIS (ids < {ALIGN_SMALL_BATCH_MAX_IDS} "
               f"and experts <= {ALIGN_SMALL_BATCH_MAX_EXPERTS} -> "
               f"{SMALL_BATCH}, else {BLOCK_SCAN}); the probe MEASURES it:"]
        for a, n, t, i, d, path in self.rows:
            out.append(f"            {a:8s} n={n:<2d} T={t:<6d} ids={i:<6d} "
                       f"declared={d:<4d} {path}")
        for arm in ARMS:
            sw = self.switch_tread(arm)
            out.append(f"            {arm:8s} "
                       + (f"SWITCHES at tread {sw}" if sw
                          else "one kernel throughout"))
        return out

    def as_dict(self) -> dict:
        return {"rows": [list(r) for r in self.rows],
                "refusals": list(self.refusals),
                "switch_tread": {arm: self.switch_tread(arm) for arm in ARMS},
                "hypothesis": {"max_ids": ALIGN_SMALL_BATCH_MAX_IDS,
                               "max_experts": ALIGN_SMALL_BATCH_MAX_EXPERTS}}


def path_census(cfg, treads: list[int], block_m: int,
                declared_by_arm: dict[str, int]) -> PathCensus:
    rows = []
    refusals = []
    for arm in ARMS:
        d = declared_by_arm[arm]
        padded = -(-d // ALIGN_WARP) * ALIGN_WARP
        if padded >= ALIGN_MAX_PADDED_EXPERTS:
            refusals.append(
                f"{arm} declares {d} experts, padded to {padded}, and vLLM's "
                f"scan kernel refuses {ALIGN_MAX_PADDED_EXPERTS} or more")
        for n in treads:
            tokens = SWEEP.tokens_for_rows(cfg, n * block_m)
            numel = tokens * cfg.top_k
            if numel * NAIVE_ASSIGNMENT_SPARSITY <= d:
                refusals.append(
                    f"{arm} at n={n}: {tokens} x {cfg.top_k} x "
                    f"{NAIVE_ASSIGNMENT_SPARSITY} <= {d} declared puts vLLM on "
                    "its naive assignment, a different kernel launch with no "
                    "alignment at all")
            if numel < d:
                refusals.append(
                    f"{arm} at n={n}: {numel} ids < {d} declared clamps the "
                    "sorted-id buffer, a different launch grid")
            rows.append((arm, n, tokens, numel, d, align_path(numel, d)))
    census = PathCensus(tuple(rows), ())
    for arm in RATIO_ARMS:
        sw = census.switch_tread(arm)
        if sw:
            refusals.append(
                f"{arm} changes alignment kernel at tread {sw}: a step inside "
                "a ratio arm's ladder, which a straight-line fit reads as "
                "slope. Declare more copies (--declared-copies) so that E x "
                f"n_decl > {ALIGN_SMALL_BATCH_MAX_EXPERTS}, or keep the ladder "
                "on one side of the id bound")
    return PathCensus(tuple(rows), tuple(refusals))


# --------------------------------------------------------------------------
# THE PROBE: the step, measured on the attached build, and the fit that finds
# a step in a series.
# --------------------------------------------------------------------------

#: The probe's cells are cheap, so it repeats them: the step it reports is a
#: median over these, and its own across-repeat spread is printed beside the
#: standard error the verdict is taken on.
PROBE_REPEATS = 3
#: Below this the probe cannot form an across-repeat spread at all, and a
#: gate whose noise is undefined cannot fail. Refused rather than run.
MIN_PROBE_REPEATS = 2
PROBE_WARMUP_MS = 50.0
PROBE_TARGET_MS = 30.0
PROBE_TRIALS = 3
#: DESIGN DECISION 13. The probe times the op UNDER A CUDA GRAPH: this many
#: calls are captured into one graph and one replay is the timed callable, so
#: the interval is GPU time and a cell's `ms` is the replay's p50 over this
#: count. One cudaGraphLaunch per replay against N calls of GPU work keeps the
#: queue deep (session 4's EAGER probe on the H200 was host-bound 36 of 36:
#: 32-36 us of host per call against a kernel of a few us, and read UNKNOWN),
#: and the replay's own launch gap lands in every cell of every series at the
#: same 1/N, an intercept and never a step. What the interval DOES carry is
#: the per-node launch latency inside the graph, which scales with the kernel
#: count of the path taken: that is GPU time, the same the queue-deep sweep
#: pays, and it is the quantity the ratio arms' slopes are about. A CONSTANT
#: and not an adaptive count, because it is recorded on every cell and
#: identical across arms and treads, which is what makes the residual an
#: intercept; if a pod shows host_bound True at this count, it is doubled
#: once, not fitted. Recorded on every cell as `graph_calls`; 0 there means
#: the cell was timed eagerly (capture refused, or a pre-2026-09-22 row).
PROBE_CALLS_PER_REPLAY = 16
#: `torch.cuda.graph` synchronises, collects and empties the cache before each
#: capture; the per-cell allowance `probe_seconds` books for that. A guess
#: until a pod measures it; the estimate prints it and the session driver's
#: booking row quotes the printed estimate, so both move together.
PROBE_CAPTURE_MS = 20.0
#: A step is REAL, for the record and for choosing the split the V5 fit uses,
#: when it exceeds this many STANDARD ERRORS of its own fitted coefficient,
#: widened for the fact that the split was CHOSEN by minimum residual over
#: every candidate (`selection_penalty`).
#:
#: WHAT THIS REPLACED, and why, measured rather than argued. The rule was
#: `|step| > 3 x the median across-repeat spread of the series' cells`. Two
#: defects, both simulated at this design's own geometry (6 treads, the probe's
#: own affine cost model, 400 worlds per cell):
#:   * it fired on PURE NOISE 28.2% of the time at 3 repeats, because the step
#:     was chosen as the best of five candidate splits and then tested as if
#:     the split had been named in advance;
#:   * and it got STRICTER as the probe was repeated more (4.2% at 5 repeats,
#:     0.0% at 9), because the series is a MEDIAN over repeats -- whose noise
#:     falls as the repeats grow -- while the threshold stayed at the spread of
#:     a SINGLE cell. More measurement made the instrument blinder.
#: The standard error of the fitted coefficient is on the median series, so it
#: falls with the repeats the way the estimate does, and the penalty prices the
#: search. AND THE SIGMA IS A TAIL PROBABILITY, NOT A MULTIPLIER: `s2` is
#: estimated on `n - 3` degrees of freedom, three at the default six treads,
#: so the threshold is the Student-t quantile with the two-sided tail this
#: sigma leaves under a normal (`step_quantile`: 9.22 at three dof). A bare
#: 3 sigma fired on pure noise 0.42 of the time at 4 treads and 0.15 at 5
#: (0.4202 +/- 0.0011 and 0.1534 +/- 0.0008, 200,000 worlds over 8 seeds at
#: this file's own probe geometry; the owner's 2,000-trial run of the same
#: rule read 0.428 and 0.155, which is within one standard error of it).
#: Measured at 3 repeats, 600 worlds: 0.3% on pure noise at six treads, flat
#: across 3, 5 and 9 repeats, under 1% at 4 and 5 treads; a step worth the
#: whole 0.01 budget is found 96% of the time and one worth 1.5 budgets
#: 99.8%. Sub-budget steps are found less often (a fifth of the budget: 25%)
#: and PASS V8 either way. All properties of the RULE, not of any card.
PROBE_STEP_SIGMA = 3.0

#: DESIGN DECISION 12. How much of the ratio an alignment step in the ratio
#: arms may be worth before V8 refuses the design: a hundredth, a sixth of
#: ALPHA_BAND's width. CHOSEN. The bias is arithmetic (`pair_step_bias` over
#: both ratio series, or `step_bias` when PRIVATE's was not probed) at the
#: run's own calibrated weight-stream time; nothing here is a card's number.
ALIGN_STEP_RATIO_BUDGET = 0.01


@dataclass(frozen=True)
class ProbeCell:
    label: str       # the arm whose declaration was probed
    tread: int
    numel: int
    declared: int
    repeat: int
    ms: float
    #: `time_kernel`'s own host-bound verdict for this cell. An alignment call
    #: is a few microseconds of GPU against tens of host, so an EAGER cell that
    #: is host-bound timed the HOST; a GRAPH-timed cell (`graph_calls` > 0)
    #: that is host-bound means cudaGraphLaunch outran `graph_calls` calls of
    #: GPU work, an anomaly the page names rather than a host time. None is
    #: "not determinable", which is not the same as False.
    host_bound: bool | None = None
    host_note: str = ""
    #: Calls captured per graph replay when this cell was graph-timed; `ms` is
    #: then the replay's p50 divided by it. 0 means timed eagerly.
    graph_calls: int = 0
    #: The undivided replay p50 in ms when graph-timed, None when eager.
    replay_ms: float | None = None


@dataclass(frozen=True)
class AlignProbe:
    cells: tuple[ProbeCell, ...]
    synthetic: bool
    note: str = ""

    def labels(self) -> list[str]:
        return [a for a in ARMS if any(c.label == a for c in self.cells)]

    def series(self, label: str) -> list[tuple[int, int, float]]:
        """`(tread, numel, median ms)` per tread for one declaration."""
        by: dict[tuple[int, int], list[float]] = {}
        for c in self.cells:
            if c.label == label:
                by.setdefault((c.tread, c.numel), []).append(c.ms)
        return [(n, i, statistics.median(v)) for (n, i), v in sorted(by.items())]

    def spread_ms(self, label: str) -> float | None:
        """Median across-repeat spread of this declaration's cells, in ms."""
        by: dict[int, list[float]] = {}
        for c in self.cells:
            if c.label == label:
                by.setdefault(c.tread, []).append(c.ms)
        s = [statistics.pstdev(v) for v in by.values() if len(v) > 1]
        return statistics.median(s) if s else None

    def host_bound(self, label: str) -> tuple[int, int, str]:
        """`(cells the instrument called host-bound, cells with a verdict, a
        note from one of them)` for one declaration.

        An EAGER host-bound probe cell is a measurement of the HOST's enqueue
        cost, and a step in that is not a step in the alignment kernel; a
        GRAPH-timed cell the instrument called host-bound is an anomaly
        (`ProbeCell.graph_calls`). V8 reads this rather than scoring a number
        about the wrong machine.
        """
        mine = [c for c in self.cells if c.label == label]
        verdicts = [c for c in mine if c.host_bound is not None]
        hot = [c for c in verdicts if c.host_bound]
        note = hot[0].host_note if hot else ""
        return len(hot), len(verdicts), note

    def graph_calls(self, label: str) -> int:
        """The ONE `graph_calls` every cell of this declaration carries; 0 is
        eager. A series that mixes eager and graph-timed cells, or two counts,
        has two intercepts and would hand `step_fit` a step that belongs to the
        instrument, so it is refused rather than fitted."""
        got = {c.graph_calls for c in self.cells if c.label == label}
        if len(got) > 1:
            raise Unmeasurable(
                f"{label}'s probe series was timed on more than one instrument "
                f"(graph_calls {sorted(got)}); one series, one instrument")
        return got.pop() if got else 0

    def as_dict(self) -> dict:
        return {"synthetic": self.synthetic, "note": self.note,
                "cells": [asdict(c) for c in self.cells]}


@dataclass(frozen=True)
class StepFit:
    """`ms = a + c numel + s 1{tread >= split}`, at the split that fits best,
    beside the plain affine fit it is judged against.

    `step_se` is the standard error of `s` from the fit's own residual, and
    `splits_tried` is how many candidate splits the minimum was taken over:
    the two numbers a verdict on `s` needs, because the split was chosen by
    looking at the data.
    """

    split_tread: int | None
    step_ms: float
    slope_ms_per_id: float
    intercept_ms: float
    rss_with: float
    rss_without: float
    step_se: float = 0.0
    splits_tried: int = 0

    #: `n - 3`: points left over after the intercept, the per-id slope and
    #: the step. `step_fit` refuses a series that leaves none.
    dof: int = 0

    def threshold_ms(self, sigma: float = PROBE_STEP_SIGMA) -> float:
        """How big `|s|` has to be before it is a step and not the best of
        `splits_tried` noise draws: the Student-t quantile at this fit's own
        `dof` for the two-sided tail `sigma` standard errors leave under a
        normal (`step_quantile`), times `step_se`, times the search penalty.

        A bare `sigma` here judged `s2 = RSS/(n-3)` with one to three degrees
        of freedom as if it were the true variance, which fired on pure noise
        far more often than its sigma claimed at four and five treads;
        `PROBE_STEP_SIGMA`'s note carries that measurement and this docstring
        does not restate it. The t quantile prices the variance estimate's
        own noise."""
        return (step_quantile(sigma, self.dof) * self.step_se
                * selection_penalty(self.splits_tried))

    def resolved(self, sigma: float = PROBE_STEP_SIGMA) -> bool:
        return (self.split_tread is not None and self.step_se > 0.0
                and abs(self.step_ms) > self.threshold_ms(sigma))


def selection_penalty(splits_tried: int) -> float:
    """The widening a threshold needs when the split was CHOSEN as the best of
    `splits_tried`: `sqrt(2 ln k)`, the growth of the maximum of k standard
    normals. 1.0 for a single candidate, 1.79 for the five a six-tread ladder
    offers. It is a correction for a search, not a quantity about a card."""
    return math.sqrt(2.0 * math.log(splits_tried)) if splits_tried > 1 else 1.0


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the regularised incomplete beta function, by
    the modified Lentz method (Numerical Recipes 6.4). No scipy on the pod
    image this runs on, and a table would cover only the dof it listed."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((a - 1.0 + m2) * (a + m2)),
                   -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            return h
    raise Unmeasurable("the incomplete beta continued fraction did not converge")


def _betai(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function `I_x(a, b)`."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_tail(t: float, dof: int) -> float:
    """`P(|T| > t)` for Student's t on `dof` degrees of freedom."""
    return _betai(dof / 2.0, 0.5, dof / (dof + t * t))


def step_quantile(sigma: float, dof: int) -> float:
    """The t quantile on `dof` degrees of freedom whose two-sided tail equals
    the two-sided NORMAL tail `sigma` leaves, `erfc(sigma / sqrt 2)`: 0.0027
    for 3. So `PROBE_STEP_SIGMA` keeps its meaning as a false-alarm rate and
    the threshold widens for a variance estimated from few points: 235.8 at
    one dof, 19.21 at two, 9.22 at three (the six-tread default), 3.01 at a
    thousand. Refuses `dof <= 0`, where there is no variance to be uncertain
    about and the old code returned a threshold of zero."""
    if dof <= 0:
        raise Unmeasurable(f"{dof} degrees of freedom: no residual to estimate "
                           "a step's standard error from")
    p = math.erfc(sigma / math.sqrt(2.0))
    lo, hi = 0.0, 1.0
    while t_two_sided_tail(hi, dof) > p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_tail(mid, dof) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _ols(rows: list[list[float]], ys: list[float]) -> list[float]:
    """Least squares by the normal equations, for two or three columns.
    REFUSES a singular design rather than returning a number for it."""
    return _ols_se(rows, ys)[0]


def _ols_se(rows: list[list[float]], ys: list[float]
            ) -> tuple[list[float], list[float]]:
    """`(coefficients, their standard errors)`, by Gauss-Jordan on
    `[X'X | I | X'y]`, so the inverse that the errors need comes out of the
    same elimination as the fit. `s^2 = RSS / (n - k)`; the errors are NaN
    when there is no degree of freedom left to estimate one with, and a caller
    that reads them has to say what it does with a zero. REFUSES a singular
    design rather than returning a number for it."""
    k = len(rows[0])
    n = len(rows)
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys, strict=True)) for i in range(k)]
    m = [ata[i][:] + [1.0 if j == i else 0.0 for j in range(k)] + [aty[i]]
         for i in range(k)]
    scale = max(abs(v) for row in ata for v in row) or 1.0
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12 * scale:
            raise Unmeasurable("the design is singular: a column is constant "
                               "or two columns coincide")
        m[col], m[piv] = m[piv], m[col]
        pivot = m[col][col]
        m[col] = [v / pivot for v in m[col]]
        for r in range(k):
            if r != col:
                f = m[r][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col], strict=True)]
    coefs = [m[i][2 * k] for i in range(k)]
    resid = [sum(c * x for c, x in zip(coefs, row, strict=True)) - y
             for row, y in zip(rows, ys, strict=True)]
    dof = n - k
    # NaN, NOT ZERO, with nothing left over: a zero error made every step
    # "exact" and `resolved` had to special-case it. Coefficients stand.
    s2 = (sum(r * r for r in resid) / dof) if dof > 0 else math.nan
    ses = [math.sqrt(max(0.0, s2 * m[i][k + i])) if dof > 0 else math.nan
           for i in range(k)]
    return coefs, ses


#: `a + c numel + s 1{tread >= split}`: the step fit's column count.
STEP_FIT_COLUMNS = 3


def step_fit(points: list[tuple[int, int, float]]) -> StepFit:
    """The best single step in an otherwise affine series.

    `points` are `(tread, numel, ms)`. For every split between two treads the
    series is fitted as `a + c numel + s 1{tread >= split}`; the split with
    the smallest residual sum of squares is reported with its step, beside
    the plain affine fit's residual. Whether the step is REAL is the caller's
    call against the series' own noise; this only finds it.
    """
    pts = sorted(points)
    if len(pts) <= STEP_FIT_COLUMNS:
        raise Unmeasurable(
            f"{len(pts)} treads cannot resolve a step: the fit has "
            f"{STEP_FIT_COLUMNS} columns (intercept, per-id slope, step), so "
            f"{len(pts)} treads leave {len(pts) - STEP_FIT_COLUMNS} degrees of "
            "freedom and no standard error to judge the step against; run at "
            f"least {STEP_FIT_COLUMNS + 1} treads")
    ys = [ms for _n, _i, ms in pts]
    plain = _ols([[1.0, float(i)] for _n, i, _ms in pts], ys)
    rss0 = sum((plain[0] + plain[1] * i - ms) ** 2 for _n, i, ms in pts)
    best = None
    tried = 0
    for j in range(1, len(pts)):
        split = pts[j][0]
        rows = [[1.0, float(i), 1.0 if n >= split else 0.0] for n, i, _ in pts]
        try:
            (a, c, s_), ses = _ols_se(rows, ys)
        except Unmeasurable:
            continue
        tried += 1
        rss = sum((a + c * i + s_ * (1.0 if n >= split else 0.0) - ms) ** 2
                  for n, i, ms in pts)
        if best is None or rss < best[0]:
            best = (rss, split, s_, c, a, ses[2])
    dof = len(pts) - STEP_FIT_COLUMNS
    if best is None:
        return StepFit(None, 0.0, plain[1], plain[0], rss0, rss0, 0.0, 0, dof)
    rss, split, s_, c, a, se = best
    return StepFit(split, s_, c, a, rss, rss0, se, tried, dof)


def leverage(treads: list[int], split_tread: int) -> float:
    """How much of a unit step at `split_tread` a straight-line fit over
    `treads` reads as SLOPE: `Sxy / Sxx` of the indicator on `n`. Arithmetic
    over the tread set alone; 0.257 for n = 1..6 split at 4."""
    xs = [float(n) for n in treads]
    ind = [1.0 if n >= split_tread else 0.0 for n in treads]
    mx, mi = statistics.fmean(xs), statistics.fmean(ind)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        raise Unmeasurable("one tread has no lever arm")
    return sum((x - mx) * (d - mi) for x, d in zip(xs, ind, strict=True)) / sxx


def step_bias(step_ms: float, treads: list[int], split_tread: int,
              weight_stream_ms: float) -> float:
    """An upper bound on what a step common to BOTH ratio arms does to the
    ratio. THE PREMISE IS CHECKED ONLY WHEN PRIVATE'S IDS WERE PROBED, and
    then V8 scores `pair_step_bias`, which does not need it; this bound is
    what a probe without PRIVATE's series falls back on.

    A step `s` at `split_tread` enters a straight-line fit as `L s` of slope in
    BOTH arms, `L` the leverage. The ratio `(B_s + L s)/(B_p + L s)` moves most
    when the denominator is smallest, so the bound is `L|s| / (B_p - L|s|)` for
    a NEGATIVE step and `L|s| / B_p` for a positive one. The earlier form used
    the positive bound for both and was therefore not a bound at all on the
    side where the denominator shrinks: at `L|s|` equal to half the stream it
    understated the move by a factor of two. `B_p` is the denominator's own
    scale, for which the caller passes the card's calibrated weight-stream time
    (what a no-reuse slope is, to within the arm's own residual).

    `inf` when the step is big enough to consume the denominator: not a bias,
    a design with nothing left to measure.
    """
    if weight_stream_ms <= 0:
        return math.inf
    moved = leverage(treads, split_tread) * abs(step_ms)
    if step_ms >= 0:
        return moved / weight_stream_ms
    if moved >= weight_stream_ms:
        return math.inf
    return moved / (weight_stream_ms - moved)


def pair_step_bias(shared: tuple[float, int] | None,
                   private: tuple[float, int] | None, treads: list[int],
                   weight_stream_ms: float) -> float:
    """An upper bound on what the ratio arms' steps do to the ratio, with NO
    premise that the step is common to both: each arm's `(step_ms, split)`
    is its own fitted step, `None` for none.

    A step enters a straight-line fit as `a = L s` of slope. With `a` in the
    numerator's slope and `c` in the denominator's, the ratio
    `(B_s + a) / (B_p + c)` moves from `R = B_s / B_p` by exactly
    `(a - R c) / (B_p + c)`. That is linear in `R`, so over `R` in `[0, 1]`
    its size is at most `max(|a|, |a - c|) / (B_p + c)`. For a common step
    (`a == c`) this is `|a| / (B_p + a)`, no looser than `step_bias`; for a
    step in ONE arm it is what `step_bias` never covered.

    ASSUMED: `R <= 1`, i.e. the shared arm does not re-read more than the
    whole weight set per M-tile. The NO-REUSE world sits at 1.0.
    """
    if weight_stream_ms <= 0:
        return math.inf

    def moved(step):
        return 0.0 if step is None else leverage(treads, step[1]) * step[0]
    a, c = moved(shared), moved(private)
    denom = weight_stream_ms + c
    if denom <= 0:
        return math.inf
    return max(abs(a), abs(a - c)) / denom


def time_probe_cell(call, *, arm: str, tread: int, numel: int, declared: int,
                    repeat: int, reference_clock: float | None,
                    calls_per_replay: int, graph_timer, eager_timer) -> ProbeCell:
    """ONE probe cell on the instrument. `calls_per_replay` > 0: capture that
    many calls in one graph (`graph_timer` is `driver.time_kernel_graph`, the
    same capture the sweep's graph rows use), time the replay, divide. 0: the
    eager path (`eager_timer` is `timing.time_kernel`), kept for the capture
    refusal and for reproducing pre-2026-09-22 rows. Pure plumbing: the
    off-GPU tests drive it with fake timers. `NotCapturable` propagates."""
    kw = dict(warmup_ms=PROBE_WARMUP_MS, target_ms=PROBE_TARGET_MS,
              trials=PROBE_TRIALS, l2_flush=False,
              reference_clock_mhz=reference_clock)
    if calls_per_replay <= 0:
        t = eager_timer(call, **kw)
        return ProbeCell(arm, tread, numel, declared, repeat, t.ms_p50,
                         host_bound=getattr(t, "host_bound", None),
                         host_note=getattr(t, "host_note", ""))

    def batch():
        for _ in range(calls_per_replay):
            call()
    t = graph_timer(batch, **kw)
    return ProbeCell(arm, tread, numel, declared, repeat,
                     t.ms_p50 / calls_per_replay,
                     host_bound=getattr(t, "host_bound", None),
                     host_note=getattr(t, "host_note", ""),
                     graph_calls=calls_per_replay, replay_ms=t.ms_p50)


def probe_cells(cfg, *, block_m: int, treads: list[int],
                declared_by_arm: dict[str, int], copies_declared: int,
                reference_clock: float | None, repeats: int,
                calls_per_replay: int, op, sync, graph_timer, eager_timer,
                device: str = "cuda") -> AlignProbe:
    """The probe's loop with its op and instrument injected, so the plumbing
    runs off-GPU with fakes: `op(ids, block_m, declared)` is the alignment
    call, `sync` the device synchronise.

    ONE SERIES, ONE INSTRUMENT. Every cell is tried under the graph; a
    `NotCapturable` on ANY cell, first or later, discards what was collected
    and re-runs the WHOLE probe eagerly with the reason on the probe's note.
    A refusal on a later cell used to be the alternative worth stating: it
    would have escaped to main's catch-all as ERROR with the graph cells
    lost, and a mix that was merely recorded is what `AlignProbe.graph_calls`
    refuses at fit time. The eager instrument plus NATIVE's control is still
    a probe, which is why the fallback is a fallback and not a refusal."""
    from moe.bench.timing import NotCapturable

    def collect(mode: int) -> list[ProbeCell]:
        cells = []
        for rep_ in range(repeats):
            for n in treads:
                tokens = SWEEP.tokens_for_rows(cfg, n * block_m)
                ids = SWEEP.balanced_ids(cfg, tokens, device)
                # ONE LABEL PER ARM, at that arm's declaration and on that
                # arm's own ids. SHARED and PRIVATE share a declaration and
                # differ in the id SET: PRIVATE spreads each expert's rows
                # over n copies, so the alignment's per-expert counters see
                # different contention. Probing both is what measures that
                # asymmetry and what checks the "common to both arms" premise
                # of `step_bias`.
                use_by_arm = {
                    NATIVE: ids,
                    SHARED: shared_topk_ids(ids, copies_declared),
                    PRIVATE: private_topk_ids(ids, cfg.num_experts, block_m,
                                              n * block_m, copies_declared),
                }
                for arm in ARMS:
                    d = declared_by_arm[arm]
                    use = use_by_arm[arm]

                    def call(use=use, d=d):
                        op(use, block_m, d)
                    call()
                    sync()
                    cells.append(time_probe_cell(
                        call, arm=arm, tread=n, numel=tokens * cfg.top_k,
                        declared=d, repeat=rep_,
                        reference_clock=reference_clock,
                        calls_per_replay=mode, graph_timer=graph_timer,
                        eager_timer=eager_timer))
        return cells

    note = ""
    try:
        cells = collect(calls_per_replay)
    except NotCapturable as exc:
        note = (f"capture refused ({str(exc)[:160]}); the WHOLE probe was "
                "re-run EAGERLY, so a host-bound cell timed the host")
        cells = collect(0)
    return AlignProbe(tuple(cells), synthetic=False, note=note)


def probe_alignment(cfg, *, block_m: int, treads: list[int],
                    declared_by_arm: dict[str, int], copies_declared: int,
                    reference_clock: float | None,
                    repeats: int = PROBE_REPEATS,
                    calls_per_replay: int = PROBE_CALLS_PER_REPLAY,
                    graph_timer=None, eager_timer=None) -> AlignProbe:
    """Time vLLM's alignment op ALONE, once per ARM, along the ladder, on the
    attached build, UNDER A CUDA GRAPH (`PROBE_CALLS_PER_REPLAY` calls per
    replay, so each cell is GPU time; eager is the fallback when the capture
    is refused, and the probe's note says so).

    Three series: NATIVE's own declaration, and SHARED's and PRIVATE's ID
    SETS at the ratio arms' shared declaration. The two id sets are timed
    apart because they are what differs between the arms whose slopes the
    ratio divides, and V8 bounds the step over both of them.

    The op is `moe_align_block_size(topk_ids, BLOCK_M, declared)`, called as
    `fused_experts_impl` calls it, on the ids each arm passes. It is the one
    place in the call whose kernel choice depends on the id count and the
    declaration, so timing it alone isolates the switch from the GEMM. Cheap:
    tens of milliseconds a cell and no Triton compile.
    """
    import torch
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    from moe.bench import timing
    if graph_timer is None:
        # IMPORTED, not copied: the capture the sweep's graph rows use
        # (`timing.time_graph` is the RETIRED timer and is not this).
        from moe.bench.driver import time_kernel_graph as graph_timer
    if eager_timer is None:
        eager_timer = timing.time_kernel
    return probe_cells(cfg, block_m=block_m, treads=treads,
                       declared_by_arm=declared_by_arm,
                       copies_declared=copies_declared,
                       reference_clock=reference_clock, repeats=repeats,
                       calls_per_replay=calls_per_replay,
                       op=moe_align_block_size, sync=torch.cuda.synchronize,
                       graph_timer=graph_timer, eager_timer=eager_timer)


#: What the probe times per tread and repeat: one series per ARM (NATIVE's
#: declaration, and SHARED's and PRIVATE's id sets at the shared declaration).
PROBE_LABELS = len(ARMS)


def probe_seconds(treads: list[int], declarations: int,
                  repeats: int = PROBE_REPEATS) -> float:
    return (declarations * len(treads) * repeats
            * (PROBE_CAPTURE_MS + PROBE_WARMUP_MS + PROBE_TRIALS * PROBE_TARGET_MS)
            * 1e-3)


@dataclass(frozen=True)
class ProbeReading:
    """One declaration's series read: its best step, whether that step clears
    the series' own noise, and what the hypothesis said."""

    label: str
    fit: StepFit
    spread_ms: float | None
    real: bool
    hypothesis_split: int | None
    #: The instrument every cell of the series was timed on: calls per graph
    #: replay, 0 for eager. `AlignProbe.graph_calls` refuses a mix.
    graph_calls: int = 0

    def lines(self) -> list[str]:
        f = self.fit
        out = [f"{self.label:8s} best split at tread "
               f"{f.split_tread if f.split_tread else '-'}: step "
               f"{f.step_ms * 1e3:+.2f} us +/- {f.step_se * 1e3:.2f} us "
               f"(1 se), affine part {f.intercept_ms * 1e3:.2f} "
               f"us + {f.slope_ms_per_id * 1e6:.4f} ns/id; RSS with the step "
               f"{f.rss_with:.3e}, without {f.rss_without:.3e}; across-repeat "
               f"spread "
               + (f"{self.spread_ms * 1e3:.2f} us" if self.spread_ms is not None
                  else "NOT DETERMINED")
               + f"; the step is {'REAL' if self.real else 'NOT resolved'} "
               f"against {PROBE_STEP_SIGMA:.0f} se widened by "
               f"{selection_penalty(f.splits_tried):.2f} for the "
               f"{f.splits_tried} split(s) the minimum was taken over, i.e. "
               f"{f.threshold_ms() * 1e3:.2f} us"
               + (f" [GPU time: {self.graph_calls} calls per graph replay]"
                  if self.graph_calls else " [eager]")]
        if self.hypothesis_split:
            out.append(f"         the hypothesis puts this declaration's switch at "
                       f"tread {self.hypothesis_split}: "
                       + ("FOUND THERE" if self.real
                          and f.split_tread == self.hypothesis_split
                          else "found elsewhere" if self.real
                          else "no step resolved, so the hypothesis is not "
                               "confirmed on this build"))
        else:
            out.append("         the hypothesis puts this declaration on one "
                       "kernel throughout: "
                       + ("a step was resolved anyway" if self.real
                          else "no step resolved, consistent"))
        return out


def read_probe(probe: AlignProbe, label: str, census: PathCensus
               ) -> ProbeReading:
    graph_calls = probe.graph_calls(label)   # refuses a two-instrument series
    fit = step_fit(probe.series(label))
    return ProbeReading(label, fit, probe.spread_ms(label), fit.resolved(),
                        census.switch_tread(label), graph_calls)


# --------------------------------------------------------------------------
# One measured sample, and the store that appends them.
# --------------------------------------------------------------------------

@dataclass
class Sample:
    """One (arm, tread, repeat) timing and everything derivable from it.

    THE STATE THE CELL WAS TIMED IN IS A COLUMN. `instrument`, `warmup_ms`,
    `iters`, `trials`, the three clock fields and `l2_flush` are what the
    instrument reported about the measurement it just made
    (`moe.bench.timing.time_kernel` at full duty, `clock_elasticity.time_duty`
    below it: `time_cell`), written per row because they are what makes a
    row comparable with the roof or not. The duty timer's own diagnostics
    follow `power_w` and are records.

    AND A FAILED LEVEL CARRIES ITS SIDE. `clock_level_side` is LOW, HIGH or ""
    and is a RECORD, never an exclusion: since 2026-09-09 `clock_drift_ok` alone
    decides membership, because on a power-capped card the under-load clock is
    an outcome of the tile and a band around the calibration GEMM's operating
    point excludes a TILE rather than a defect. `excluded` is the one predicate
    any fit here reads.
    """

    arm: str
    repeat: int
    block_m: int
    tiles: int
    rows_per_expert: int
    tokens: int
    #: Copies of the weight set this call READ (`copies_read`): `n` for
    #: private, 1 for shared and native. Every arm could ADDRESS the whole
    #: `n_decl` allocation except native, which sees copy 0 through a view;
    #: `experts_declared` is the column that says how much was in reach.
    copies: int
    #: `global_num_experts` as the call declared it (`declared_experts`): `E`
    #: for native, `E x n_decl` for shared and private at EVERY tread. Written
    #: per row because it sizes `moe_align_block_size`'s sorted-id buffer and
    #: the launch grid, which is the machinery V5 measures the cost of.
    experts_declared: int
    ms_p50: float
    ms_min: float = 0.0
    ms_stdev: float = 0.0
    iters: int = 0
    trials: int = 0
    warmup_ms: float = 0.0
    instrument: str = ""
    sm_clock_load_mhz: float | None = None
    clock_level_ok: bool | None = None
    clock_level_side: str = ""
    clock_drift_ok: bool | None = None
    l2_flush: bool = False
    status: str = "ok"
    detail: str = ""
    #: The duty cycle this cell was timed at (DESIGN DECISION 15): 1.0 is the
    #: driver's instrument with the queue kept full; below 1.0 the cell was
    #: bursts of kernel time with idle gaps, at a lower average board power.
    #: A row written before the column reads back as 1.0, which is what it
    #: was. The REQUESTED duty; `duty_achieved` is the measured one.
    duty: float = 1.0
    #: Median board power over the cell's clock samples, W, None when no read
    #: carried one. BOTH instruments read it, at the same NVML call as the
    #: clock: `time_kernel` through `KernelTiming.power_w` at full duty,
    #: `time_duty` once per burst below it. It is NVML's ~1 s average board
    #: power, so below full duty it averages the bursts WITH the idle gaps
    #: and is not the in-burst draw. A RECORD, the traffic signal that does
    #: not go through the clock: V7 prints each arm's median per tread beside
    #: its clocks and no gate scores it.
    power_w: float | None = None
    # THE DUTY TIMER'S OWN DIAGNOSTICS (review findings 17 and 24), so a V7 or
    # V0 failure at a duty below 1 can be read off cells.csv rather than
    # bought again: `clock_elasticity.time_duty` measures every one of them
    # and this arm used to keep none. RECORDS: no gate reads any of them. None
    # on every row written before the columns existed, and at full duty for
    # the burst quantities `time_kernel` does not have.
    #: The GPU-busy fraction the duty timer measured over the trials' wall
    #: clock, a LOWER BOUND (`clock_elasticity.DutyTiming.duty_achieved`).
    duty_achieved: float | None = None
    #: Calls per burst, the first of which is discarded, and the idle gap
    #: after each burst, ms, sized from `DUTY_SIZING_MS` of full-duty reading.
    calls_per_burst: int | None = None
    gap_ms: float | None = None
    #: Medians of the first and last quarter of a burst's kept per-call
    #: times, and whether they agree within `timing.DRIFT_FRACTION`: R1 gates
    #: this pair as its V5; here V7 prints each arm's median `burst_sag` per
    #: tread beside its clocks and scores nothing on it.
    head_ms: float | None = None
    tail_ms: float | None = None
    within_burst_ok: bool | None = None
    #: Every usable under-load clock read, MHz, space-joined as
    #: `clock_elasticity` writes them: one per burst below full duty, the
    #: poller's reads at full duty. The samples and not only their median,
    #: for `timing.KernelTiming.clock_samples_mhz`'s reason: a median and a
    #: DRIFT flag cannot tell a settling ramp from a card hunting.
    clock_samples_mhz: str | None = None
    #: `timing.host_bound_verdict` on the cell: True when the queue drained
    #: while the host was still enqueueing, so `ms_*` bound the kernel from
    #: above. Both instruments compute it.
    host_bound: bool | None = None

    def __post_init__(self) -> None:
        # The sweep owns the rule that a failed LEVEL without a side is not a
        # state the instrument produces; borrowed rather than restated.
        SWEEP.check_level_side(self.clock_level_ok, self.clock_level_side)

    @property
    def excluded(self) -> bool:
        """Did the clock MOVE inside the timed region. The ONE exclusion.

        False for None on purpose: an exclusion has to be positively
        established, and a row with no clock is a row whose comparability is
        unknown, which the report counts rather than throws away.
        """
        return self.clock_drift_ok is False

    @property
    def usable(self) -> bool:
        return self.status == "ok" and self.ms_p50 > 0 and not self.excluded

    @property
    def burst_sag(self) -> float | None:
        """`(tail - head) / head` of the per-call time inside a burst: how
        much slower the end of a burst ran than its start. None when either
        quarter is unread, which is every full-duty cell."""
        if self.head_ms is None or self.tail_ms is None or self.head_ms <= 0:
            return None
        return (self.tail_ms - self.head_ms) / self.head_ms


CSV_FIELDS = list(Sample.__dataclass_fields__)
PROVENANCE_COLUMNS = sorted(PV.Provenance().as_columns())


class Store:
    """Append-mode `cells.csv`, refusing a header that is not this one.

    `csv.DictWriter` writes the fieldnames it was given and never looks at the
    file, so a wider row appended under a narrower header shifts every field
    past the first difference and nothing downstream can tell: `clock_drift_ok`
    would read the LEVEL side, and a FAILED drift -- the one rule in this tree
    that excludes a cell -- would come back None, which every gate keeps. The
    header already on disk is READ and compared before the first append, and a
    disagreement is a typed refusal naming the added and missing columns rather
    than a wider row under a narrower head.
    """

    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = list(fields)
        if path.exists() and path.stat().st_size:
            with path.open(newline="") as fh:
                on_disk = next(csv.reader(fh), [])
            if on_disk and on_disk != self.fields:
                added = [c for c in self.fields if c not in on_disk]
                gone = [c for c in on_disk if c not in self.fields]
                raise SchemaCollision(
                    f"{path} was written with a different header.\n"
                    f"  columns this build adds:   {added or 'none'}\n"
                    f"  columns on disk and gone:  {gone or 'none'}\n"
                    "Appending wider rows under a narrower header shifts every "
                    "field past the first difference and nothing downstream can "
                    "tell. Point --out at a fresh directory, or delete this "
                    "file and re-measure it.")
        self._new = not (path.exists() and path.stat().st_size)

    def append(self, sample: Sample, prov=None) -> None:
        row = asdict(sample)
        if prov is not None:
            row.update(prov.as_columns())
        with self.path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fields,
                                    extrasaction="ignore")
            if self._new:
                writer.writeheader()
                self._new = False
            writer.writerow(row)
            fh.flush()


def _opt_float(text: str):
    return float(text) if text not in ("", None) else None


def _opt_bool(text: str):
    if text in ("", None):
        return None
    return text == "True"


def _opt_int(text: str):
    return int(text) if text not in ("", None) else None


#: Bursts of about this much KERNEL time at a duty cycle below 1, the size the
#: clock arm settled on: long enough that one NVML read per burst is a clock
#: under load, short enough that the governor cannot ramp inside one.
DUTY_BURST_MS = 40.0
#: The short full-duty reading that sizes the bursts (calls per burst) before
#: a duty-cycled cell is timed; not a measurement, never written.
DUTY_SIZING_MS = 20.0


@dataclass(frozen=True)
class CellTiming:
    """What one ladder cell's timing contributes to its `Sample`, whichever
    instrument produced it. `iters` is ITERATIONS PER TRIAL on both: at full
    duty `time_kernel`'s, below it the duty timer's KEPT calls per trial,
    `bursts x (calls_per_burst - 1)` (its `samples` summed over every trial,
    divided by the trials), so the page's "iterations per trial" line and
    provenance's `iters` mean one thing at either duty; `note` is the duty
    timer's clock note (empty at full duty)."""
    ms_p50: float
    ms_min: float
    ms_stdev: float
    iters: int
    trials: int
    warmup_ms: float
    instrument: str
    sm_clock_load_mhz: float | None
    clock_level_ok: bool | None
    clock_level_side: str
    clock_drift_ok: bool | None
    l2_flush: bool
    duty: float
    power_w: float | None
    host_bound: bool | None
    note: str
    #: The duty timer's own diagnostics, `Sample`'s columns of the same
    #: names; None where the instrument has no such quantity.
    clock_samples_mhz: str | None = None
    duty_achieved: float | None = None
    calls_per_burst: int | None = None
    gap_ms: float | None = None
    head_ms: float | None = None
    tail_ms: float | None = None
    within_burst_ok: bool | None = None


def _joined_clocks(reads) -> str | None:
    """Clock reads, MHz, space-joined as `clock_elasticity` writes its
    `clock_samples_mhz` column; None when there were none."""
    reads = tuple(reads or ())
    return " ".join(f"{c:.0f}" for c in reads) if reads else None


def time_cell(call, *, duty: float, warmup_ms: float, cell_budget_ms: float,
              trials: int, l2_flush: bool, reference_clock_mhz: float | None,
              timer=None, duty_timer=None) -> CellTiming:
    """ONE ladder cell on the instrument the duty selects.

    `duty >= 1`: `timing.time_kernel`, the driver's instrument, the queue kept
    full; what every page before 2026-09-22 was timed with. `duty < 1`:
    `clock_elasticity.time_duty` (IMPORTED, not copied), `DUTY_BURST_MS` of
    kernel time per burst with an idle gap of `burst x (1/duty - 1)` after
    each, the burst sized off a `DUTY_SIZING_MS` full-duty reading of the same
    call, the same warmup and trials and L2 flush. The kernel, its bytes and
    its launch shape do not change between the two; what changes is board
    power, and with it the clock the cap allows. Pure plumbing: the off-GPU
    tests drive it with fake timers.
    """
    if timer is None:
        from moe.bench import timing
        timer = timing.time_kernel
    if duty >= 1.0:
        t = timer(call, warmup_ms=warmup_ms, target_ms=cell_budget_ms,
                  trials=trials, l2_flush=l2_flush,
                  reference_clock_mhz=reference_clock_mhz)
        # `power_w`, `host_bound` and the clock list are `KernelTiming`'s
        # own fields; read with a default because a timer written before any
        # of them existed does not carry it. The burst diagnostics stay None:
        # a queue-deep loop has no bursts.
        return CellTiming(
            ms_p50=t.ms_p50, ms_min=t.ms_min, ms_stdev=t.ms_std,
            iters=t.iters, trials=t.trials, warmup_ms=t.warmup_ms,
            instrument=t.instrument, sm_clock_load_mhz=t.sm_clock_load_mhz,
            clock_level_ok=t.clock_level_ok,
            clock_level_side=t.clock_level_side,
            clock_drift_ok=t.clock_drift_ok, l2_flush=t.l2_flush, duty=1.0,
            power_w=getattr(t, "power_w", None),
            host_bound=getattr(t, "host_bound", None), note="",
            clock_samples_mhz=_joined_clocks(getattr(t, "clock_samples_mhz", ())))
    if duty_timer is None:
        import clock_elasticity as CE  # scripts/ is on sys.path, as SWEEP is
        duty_timer = CE.time_duty
    sizing = timer(call, warmup_ms=min(warmup_ms, DUTY_SIZING_MS),
                   target_ms=DUTY_SIZING_MS, trials=1, l2_flush=l2_flush,
                   reference_clock_mhz=reference_clock_mhz)
    per_call = max(float(sizing.ms_p50), 1e-4)
    calls_per_burst = max(2, round(DUTY_BURST_MS / per_call))
    bursts = max(1, round(cell_budget_ms / DUTY_BURST_MS))
    t = duty_timer(call, duty=duty, calls_per_burst=calls_per_burst,
                   bursts=bursts, trials=trials, warm_ms=warmup_ms,
                   l2_flush=l2_flush, per_call_ms=per_call,
                   reference_clock_mhz=reference_clock_mhz)
    return CellTiming(
        ms_p50=t.ms_p50, ms_min=t.ms_min, ms_stdev=t.ms_std,
        # PER TRIAL, as `time_kernel`'s `iters` is: `samples` is the kept
        # calls summed over every burst of every trial.
        iters=t.samples // max(1, t.trials), trials=t.trials,
        warmup_ms=t.warmup_ms, instrument=t.instrument,
        sm_clock_load_mhz=t.sm_clock_load_mhz,
        clock_level_ok=t.clock_level_ok, clock_level_side=t.clock_level_side,
        clock_drift_ok=t.clock_drift_ok, l2_flush=t.l2_flush, duty=duty,
        power_w=t.power_w, host_bound=t.host_bound, note=t.clock_note or "",
        # EVERYTHING THE DUTY TIMER MEASURED, kept: what V7 and V0 need to
        # tell a split between arms from a card jittering at this duty.
        clock_samples_mhz=_joined_clocks(t.clock_samples_mhz),
        duty_achieved=t.duty_achieved, calls_per_burst=t.calls_per_burst,
        gap_ms=t.gap_ms, head_ms=t.head_ms, tail_ms=t.tail_ms,
        within_burst_ok=t.within_burst_ok)


def sample_from_timing(t: CellTiming, **cell) -> Sample:
    """The `Sample` one timed cell becomes: `cell` names it (arm, repeat,
    tread and the rest of its identity), `t` fills EVERY column the
    instrument measured, by name, and its note becomes the row's `detail`.

    ONE MAPPING, NOT A LIST AT THE CALL SITE. `run_sweep` used to copy the
    fields one by one, and `host_bound` was on `CellTiming` and on no row: a
    column the instrument measured, dropped at the second call site without
    a sound. A `CellTiming` field with no `Sample` column is now a test
    failure, not a silent loss."""
    measured = {name: getattr(t, name) for name in CellTiming.__dataclass_fields__
                if name in Sample.__dataclass_fields__}
    return Sample(**cell, **measured, detail=t.note)


def duty_of(samples) -> float:
    """The duty the ladder was timed at, off the rows (the smallest, so a
    resumed ladder that mixed two is named by the one that mattered)."""
    return min((s.duty for s in samples if s.status == "ok"), default=1.0)


def read_samples(path: Path) -> list[Sample]:
    """Every row back, BY HEADER NAME, with optional fields absent rather than
    defaulted. A pre-column file reads back as None, never as 0 or False."""
    if not path.exists():
        return []
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Sample(
                arm=row["arm"], repeat=int(row["repeat"]),
                block_m=int(row["block_m"]), tiles=int(row["tiles"]),
                rows_per_expert=int(row["rows_per_expert"]),
                tokens=int(row["tokens"]), copies=int(row["copies"]),
                experts_declared=int(row["experts_declared"]),
                ms_p50=float(row["ms_p50"]),
                ms_min=float(row.get("ms_min") or 0.0),
                ms_stdev=float(row.get("ms_stdev") or 0.0),
                iters=int(row.get("iters") or 0),
                trials=int(row.get("trials") or 0),
                warmup_ms=float(row.get("warmup_ms") or 0.0),
                instrument=row.get("instrument", ""),
                sm_clock_load_mhz=_opt_float(row.get("sm_clock_load_mhz", "")),
                clock_level_ok=_opt_bool(row.get("clock_level_ok", "")),
                clock_level_side=row.get("clock_level_side", "") or "",
                clock_drift_ok=_opt_bool(row.get("clock_drift_ok", "")),
                l2_flush=(row.get("l2_flush", "") == "True"),
                status=row.get("status", "ok"), detail=row.get("detail", ""),
                duty=float(row.get("duty") or 1.0),
                power_w=_opt_float(row.get("power_w", "")),
                duty_achieved=_opt_float(row.get("duty_achieved", "")),
                calls_per_burst=_opt_int(row.get("calls_per_burst", "")),
                gap_ms=_opt_float(row.get("gap_ms", "")),
                head_ms=_opt_float(row.get("head_ms", "")),
                tail_ms=_opt_float(row.get("tail_ms", "")),
                within_burst_ok=_opt_bool(row.get("within_burst_ok", "")),
                clock_samples_mhz=row.get("clock_samples_mhz") or None,
                host_bound=_opt_bool(row.get("host_bound", ""))))
    return out


# --------------------------------------------------------------------------
# The fit. Ordinary least squares in `n`, and a percentile bootstrap over
# repeats. Pure: samples in, numbers out, no GPU and no I/O.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Ladder:
    """One arm's ladder: the per-tread medians and the line through them."""

    arm: str
    points: tuple[tuple[int, float], ...]
    intercept_ms: float
    slope_ms: float
    mean_rel_err: float
    spread: float | None
    excluded: int

    @property
    def treads(self) -> int:
        return len(self.points)


def fit_line(points) -> tuple[float, float, float]:
    """`(intercept, slope, mean relative residual)` by ordinary least squares.

    Refuses fewer than two points rather than returning a slope of zero: a
    one-point ladder has no slope, and zero is a value a gate would read.
    """
    pts = list(points)
    if len(pts) < 2:
        raise Unmeasurable(f"a ladder of {len(pts)} point(s) has no slope")
    xs = [float(n) for n, _ in pts]
    ys = [float(t) for _, t in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        raise Unmeasurable("every tread of this ladder is at the same n; "
                           "there is no lever arm to fit a slope over")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = my - slope * mx
    errs = [abs(intercept + slope * x - y) / y for x, y in zip(xs, ys, strict=True)
            if y > 0]
    return intercept, slope, (statistics.fmean(errs) if errs else 0.0)


def repeat_indices(samples) -> list[int]:
    """Every repeat index that produced a usable cell, sorted. The bootstrap's
    population, shared by all three arms so a draw can be PAIRED."""
    return sorted({s.repeat for s in samples if s.usable})


def collapse(samples, arm: str, repeats: list[int] | None = None
             ) -> tuple[list[tuple[int, float]], float | None, int]:
    """Per-tread median across repeats, the median across-repeat spread, and the
    count of treads-by-repeat a DRIFTING clock excluded.

    `repeats` IS A LIST OF REPEAT INDICES, possibly with duplicates, and that
    is what makes this a bootstrap OVER REPEATS. What stood here resampled the
    cells within each tread INDEPENDENTLY, and `ratio_interval` then called it
    once for SHARED and once for PRIVATE, so the two arms were resampled
    independently too. In the world the arm rotation exists to produce -- a
    governor or thermal walk over the sweep's 145 s, common to both arms in a
    repeat -- that draws an interval out of noise the ratio does not have: a
    ladder with ZERO disagreement between repeats came back 0.4652 to 0.6614,
    3.3x the width of ALPHA_BAND, and the widening runs in the direction that
    makes C1 EASIER to pass, by widening the interval its verdict reads.

    A repeat drawn twice contributes its cells twice, which is the resample.
    """
    wanted = None if repeats is None else collections.Counter(repeats)
    by: dict[int, list[float]] = {}
    dropped = 0
    for s in samples:
        if s.arm != arm or s.status != "ok" or s.ms_p50 <= 0:
            continue
        if s.excluded:
            dropped += 1
            continue
        times = 1 if wanted is None else wanted.get(s.repeat, 0)
        for _ in range(times):
            by.setdefault(s.tiles, []).append(s.ms_p50)
    points = []
    for n, vals in sorted(by.items()):
        points.append((n, statistics.median(vals)))
    spreads = [statistics.pstdev(v) / statistics.median(v)
               for v in by.values() if len(v) > 1 and statistics.median(v) > 0]
    return points, (statistics.median(spreads) if spreads else None), dropped


def ladder_for(samples, arm: str, repeats: list[int] | None = None) -> Ladder:
    points, spread, dropped = collapse(samples, arm, repeats)
    intercept, slope, err = fit_line(points)
    return Ladder(arm=arm, points=tuple(points), intercept_ms=intercept,
                  slope_ms=slope, mean_rel_err=err, spread=spread,
                  excluded=dropped)


@dataclass(frozen=True)
class ClockElasticity:
    """A PER-M-TILE clock elasticity, `eta = -d log b / d log f` of the
    per-M-tile cost `b = d ms / d n`, with the interval it was measured with
    and where it came from. PER-M-TILE because the ratio reads nothing but
    slopes: `clock_corrected` scales each cell's whole per-call ms by one
    factor, and when each arm holds one clock at every fitted tread that
    factor reaches the ratio through the slope alone, which only the
    per-M-tile elasticity carries exactly (CLOCK_PARITY's comment has the
    arithmetic, and the case where no single eta is exact).
    `scripts/clock_elasticity.py` is the arm that measures it and nothing here
    does: its gated claim, `elasticity.value` in a report written since
    8d4eb78 (2026-09-22). Its pooled per-CALL reading, `elasticity.fixed_tread`
    and the `elasticity.value` of a report written before 8d4eb78, session 4's
    included, is a different quantity and [0, 1.5] admits both, so the SOURCE
    is what says which one was given (DESIGN DECISION 11)."""
    eta: float
    lo: float
    hi: float
    source: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.lo <= self.eta <= self.hi <= 1.5):
            raise Unmeasurable(
                f"an elasticity of {self.eta} [{self.lo}, {self.hi}] is not "
                "admissible: it must sit in [0, 1.5] with lo <= eta <= hi")
        if not self.source.strip():
            raise Unmeasurable("a clock elasticity needs a SOURCE, the report "
                               "or file it was read from")


def clock_corrected(samples, eta: float, f_ref: float) -> list:
    """Every RATIO-ARM cell carried to `f_ref` under `eta`: `ms x (f / f_ref)
    ** eta`. NATIVE is untouched (the ratio never reads it), an excluded cell
    is untouched (nothing reads it), and a usable ratio-arm cell WITHOUT a
    clock refuses the whole correction rather than leaving that cell raw.

    `eta` IS THE PER-M-TILE ELASTICITY (`ClockElasticity`), applied as one
    factor to the whole per-call time, intercept and tiles alike. Where an arm
    holds one clock at every fitted tread that factor is one number per arm,
    the intercept's share of it stays in the intercept, and the slope, which
    is all the ratio reads, is carried exactly. Where a clock moves across
    treads inside an arm the intercept's share leaks into the slope and the
    correction is first-order only.

    THE RATIO DOES NOT DEPEND ON f_ref: both arms carry the same factor
    `f_ref ** -eta`, which cancels. It is here so the corrected CELLS are at a
    named clock, and it is the calibration's reference clock when one was
    resolved, else the private arm's own median.
    """
    if f_ref <= 0:
        raise Unmeasurable(f"f_ref = {f_ref} MHz is not a clock")
    out = []
    missing = 0
    for s in samples:
        if s.arm not in RATIO_ARMS or s.status != "ok" or s.ms_p50 <= 0 \
                or s.excluded:
            out.append(s)
            continue
        if not s.sm_clock_load_mhz:
            missing += 1
            out.append(s)
            continue
        k = (s.sm_clock_load_mhz / f_ref) ** eta
        out.append(replace(s, ms_p50=s.ms_p50 * k, ms_min=s.ms_min * k))
    if missing:
        raise Unmeasurable(f"{missing} usable ratio-arm cell(s) carry no "
                           "under-load clock, so they cannot be carried to "
                           f"{f_ref:.0f} MHz and no corrected ratio is formed")
    return out


@dataclass(frozen=True)
class ClockCorrection:
    """The clock-corrected ratio, PRINTED beside the raw one and scored by
    nothing. `interval` is the same paired bootstrap over the corrected cells
    at `eta`; `envelope` is the union of the corrected intervals at `lo` and
    `hi`, i.e. ONE estimator's bootstrap carried across eta's own interval,
    not a between-estimator spread; `at_unit` is the ratio at eta = 1, the
    first-order figure DD11 argues from, printed as a reference point and NOT
    a bound: a per-M-tile eta above 1, which is what session 4's clock-arm
    cells read when re-scored (CLOCK_PARITY's comment), carries the
    correction past it."""
    elasticity: ClockElasticity
    f_ref: float
    f_ref_source: str
    ratio: float
    interval: tuple[float, float] | None
    envelope: tuple[float, float] | None
    at_unit: float
    raw: float

    def as_dict(self) -> dict:
        return {"eta": self.elasticity.eta, "eta_lo": self.elasticity.lo,
                "eta_hi": self.elasticity.hi,
                "eta_source": self.elasticity.source,
                "f_ref_mhz": self.f_ref, "f_ref_source": self.f_ref_source,
                "ratio": self.ratio,
                "interval": list(self.interval) if self.interval else None,
                "envelope": list(self.envelope) if self.envelope else None,
                "ratio_at_eta_1": self.at_unit, "ratio_raw": self.raw,
                "scored": False}

    def lines(self) -> list[str]:
        e = self.elasticity
        return [
            f"clock-corrected ratio {self.ratio:.4f} at eta = {e.eta:.4f} "
            f"[{e.lo:.4f}, {e.hi:.4f}] ({e.source}), every ratio-arm cell "
            f"carried to {self.f_ref:.0f} MHz ({self.f_ref_source}) by "
            "(f / f_ref) ** eta before the fit; PRINTED ONLY, the raw ratio "
            f"above is the one this arm was built to produce (moved "
            f"{self.ratio - self.raw:+.4f})",
            "  " + (f"its own {INTERVAL_PCT:.0f}% paired bootstrap "
                    f"[{self.interval[0]:.4f}, {self.interval[1]:.4f}]"
                    if self.interval else "its bootstrap was NOT FORMED")
            + (f"; over eta's interval the envelope is [{self.envelope[0]:.4f}, "
               f"{self.envelope[1]:.4f}] -- ONE estimator's bootstrap carried "
               "across eta, not a between-estimator spread"
               if self.envelope else ""),
            f"  at eta = 1, the first-order figure DD11 argues from, it would "
            f"read {self.at_unit:.4f}; not a bound: an eta above 1 carries the "
            "correction past it",
        ]


def clock_corrected_ratio(samples, elasticity: ClockElasticity, *,
                          f_ref: float, f_ref_source: str, draws: int,
                          seed: int) -> ClockCorrection:
    """Form the corrected ratio at `eta`, its interval, the envelope over
    [lo, hi], and the eta = 1 reference point, from the same samples and the
    same paired bootstrap the raw ratio uses."""
    def ratio_at(eta: float) -> float:
        cells = clock_corrected(samples, eta, f_ref)
        return ladder_for(cells, SHARED).slope_ms / ladder_for(cells, PRIVATE).slope_ms

    def interval_at(eta: float) -> tuple[float, float] | None:
        try:
            lo, hi, _n = ratio_interval(clock_corrected(samples, eta, f_ref),
                                        draws, seed)
        except Unmeasurable:
            return None
        return (lo, hi)

    raw = ladder_for(samples, SHARED).slope_ms / ladder_for(samples, PRIVATE).slope_ms
    point = ratio_at(elasticity.eta)
    interval = interval_at(elasticity.eta)
    ends = [interval_at(elasticity.lo), interval_at(elasticity.hi)]
    envelope = ((min(i[0] for i in ends), max(i[1] for i in ends))
                if all(ends) else None)
    return ClockCorrection(elasticity, f_ref, f_ref_source, point, interval,
                           envelope, ratio_at(1.0), raw)


def ratio_interval(samples, draws: int, seed: int, pct: float = INTERVAL_PCT
                   ) -> tuple[float, float, int]:
    """`(lo, hi, draws that produced a ratio)` by percentile bootstrap.

    OVER REPEATS AND PAIRED. One list of repeat indices is drawn per draw and
    BOTH arms are collapsed over that same list, so a repeat in which the whole
    card ran slow moves the numerator and the denominator together and cancels
    out of the ratio, which is what the ratio is for. Drawing the two arms
    independently -- which is what calling this with two separate resamples
    did -- puts that common factor back in as noise the ratio does not have.

    A draw that cannot be fitted -- too few treads survived the resample, or a
    zero denominator -- is DROPPED and counted, never replaced by a value. The
    count is printed beside the interval so an interval standing on a third of
    its draws cannot pass for one standing on all of them.
    """
    rng = random.Random(seed)
    population = repeat_indices(samples)
    got: list[float] = []
    for _ in range(draws):
        if len(population) < 2:
            break
        drawn = [rng.choice(population) for _ in population]
        try:
            shared = ladder_for(samples, SHARED, drawn)
            private = ladder_for(samples, PRIVATE, drawn)
        except Unmeasurable:
            continue
        if private.slope_ms == 0:
            continue
        got.append(shared.slope_ms / private.slope_ms)
    if len(got) < 2:
        raise Unmeasurable(
            f"only {len(got)} of {draws} bootstrap draws produced a ratio; "
            "there is no interval to quote")
    got.sort()
    tail = (100.0 - pct) / 2.0

    def at(percentile: float) -> float:
        idx = min(len(got) - 1, max(0, int(round(percentile / 100.0 * (len(got) - 1)))))
        return got[idx]

    return at(tail), at(100.0 - tail), len(got)


# --------------------------------------------------------------------------
# The proof that the private arm really read distinct buffers.
# --------------------------------------------------------------------------

#: The five parts of the buffer proof, in the order they are REPORTED, which is
#: not the order they run in: `prove_distinct_buffers` takes `same_layer` third,
#: because it is the only part that has to be measured before anything is
#: zeroed. The tuple's order is the reading order and the function's inline
#: comments name the running order at each step. COUNTED here rather than
#: described in prose: a proof that claims four parts and runs three is the
#: shape of defect this repository keeps producing, and `BufferProof.verdict`
#: reads this tuple rather than a literal.
PROOF_PARTS: tuple[tuple[str, str], ...] = (
    ("addresses", "the n copies occupy disjoint contiguous address ranges "
                  "spanning exactly n x the weight set, in both w1 and w2"),
    ("sentinels", "a distinct value written into each copy reads back from "
                  "that copy and from no other, and the original bytes are "
                  "restored before anything else reads them"),
    ("kernel_read", "zeroing ONE copy c at a time, for every c >= 1, changes "
                    "EXACTLY the private output rows of the tokens routed to "
                    "copy c and no others, so every tile reads its own copy "
                    "and no tile reads another's"),
    ("shared_blind", "each of those zeroings leaves the shared arm's output "
                     "bitwise unchanged, and restoring the copy restores the "
                     "private output bitwise, so the changes above are the "
                     "private mapping and not a corruption"),
    ("same_layer", "with nothing zeroed, the private arm's output is bitwise "
                   "equal to the shared arm's, so the relabelling computes the "
                   "same layer"),
)


@dataclass(frozen=True)
class BufferProof:
    """Which parts of the five-part proof held, and whether it was measured.

    `synthetic` is True for a planted proof under `--self-test`, where no
    device exists to prove anything on. The report says so on the gate's own
    line: a planted proof satisfying a presence check while describing a device
    the run never touched is exactly the defect the provenance block's
    `instrument` field exists against.
    """

    parts: dict[str, bool]
    detail: dict[str, str]
    synthetic: bool = False

    @property
    def verdict(self) -> str:
        missing = [name for name, _ in PROOF_PARTS if name not in self.parts]
        if missing:
            return UNKNOWN
        return PASS if all(self.parts[name] for name, _ in PROOF_PARTS) else FAIL

    def lines(self) -> list[str]:
        out = []
        for name, what in PROOF_PARTS:
            got = self.parts.get(name)
            mark = "PASS" if got else ("FAIL" if got is False else "NOT RUN")
            out.append(f"{mark:8s} {name}: {what}")
            if self.detail.get(name):
                out.append(f"         {self.detail[name]}")
        if self.synthetic:
            out.append("this proof was PLANTED by --self-test. No device was "
                       "read and nothing here is evidence about hardware.")
        return out


def planted_proof(ok: bool) -> BufferProof:
    return BufferProof(
        parts={name: ok for name, _ in PROOF_PARTS},
        detail={name: "planted" for name, _ in PROOF_PARTS},
        synthetic=True)


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

@dataclass
class Gate:
    """One scored gate: the claim, the verdict, and what a FAIL costs.

    `verdict` is PASS, FAIL or UNKNOWN in `moe.bench.exit_codes`'s own
    vocabulary, so `classify` refuses a spelling it does not recognise rather
    than letting it fall through a comparison. UNKNOWN counts AGAINST the gate
    on both kinds.
    """

    tag: str
    kind: str
    claim: str
    verdict: str
    measured: str
    threshold: str
    #: Read as "if this FAILS, <consequence>". Printed on a PASS too, so a
    #: reader knows what was at stake without re-deriving it.
    consequence: str
    lines: list[str] = field(default_factory=list)

    def scored(self) -> tuple[str, str, str]:
        return (self.kind, self.tag, self.verdict)

    def result_line(self) -> str:
        detail = (f"[{self.kind}] {self.claim} | measured {self.measured} "
                  f"| gate {self.threshold}")
        return exit_codes.result_line(*self.scored(), " ".join(detail.split()))

    def render(self) -> list[str]:
        out = [self.result_line(),
               f"{self.tag:3s} {self.kind:8s} {self.verdict:8s} {self.claim}",
               f"             measured {self.measured}   gate {self.threshold}",
               f"             if this FAILS: {self.consequence}"]
        out += [f"             {line}" for line in self.lines]
        return out

    def as_dict(self) -> dict:
        return {"tag": self.tag, "kind": self.kind, "claim": self.claim,
                "verdict": self.verdict, "measured": self.measured,
                "gate": self.threshold, "consequence": self.consequence,
                "detail": list(self.lines)}

    @classmethod
    def from_dict(cls, d: dict) -> Gate:
        """The inverse of `as_dict`, for `--read`: a stored gate re-rendered
        as it was scored, from the one document the writer serialised."""
        return cls(d["tag"], d["kind"], d["claim"], d["verdict"], d["measured"],
                   d["gate"], d["consequence"], list(d.get("detail") or []))


def gate_v0_non_vacuity(samples, *, planned: int, treads: list[int],
                        repeats: int) -> Gate:
    """Did the run measure the grid it planned.

    A CHECK THAT EXAMINED NOTHING REPORTS NO FAILURES. Every gate below reads
    ladders, and a ladder assembled from three surviving cells out of a hundred
    still fits, still reports a slope and still passes. So the counts are a
    gate: cells measured against cells planned, usable treads per arm, repeats
    per cell, and the cells a DRIFTING clock excluded named on their own line
    rather than silently dropped.
    """
    ok = {(s.arm, s.tiles, s.repeat) for s in samples if s.usable}
    failed = [s for s in samples
              if s.status != "ok" and (s.arm, s.tiles, s.repeat) not in ok]
    per_arm = {arm: len({s.tiles for s in samples if s.usable and s.arm == arm})
               for arm in ARMS}
    dropped = {arm: sum(1 for s in samples
                        if s.arm == arm and s.status == "ok" and s.excluded)
               for arm in ARMS}
    reps = {arm: min([sum(1 for s in samples
                          if s.usable and s.arm == arm and s.tiles == n)
                      for n in treads] or [0])
            for arm in ARMS}
    short = [arm for arm in ARMS if per_arm[arm] < max(MIN_TREADS, 2)]
    thin = [arm for arm in ARMS if reps[arm] < MIN_REPEATS]
    timed = sum(1 for s in samples if s.status == "ok")
    drift_share = (sum(dropped.values()) / timed) if timed else 1.0
    detail = [
        f"{len(ok)} of {planned} planned cells measured and usable",
        "usable treads per arm: "
        + ", ".join(f"{arm}:{per_arm[arm]}" for arm in ARMS)
        + f" (of {len(treads)} planned)",
        "fewest repeats behind any tread, per arm: "
        + ", ".join(f"{arm}:{reps[arm]}" for arm in ARMS)
        + f" (of {repeats} planned, floor {MIN_REPEATS})",
        "cells excluded because the clock DRIFTED across their own trials, and "
        "so absent from every ladder above: "
        + ", ".join(f"{arm}:{dropped[arm]}" for arm in ARMS)
        + (" (none)" if not sum(dropped.values()) else "")
        + f"; {100 * drift_share:.1f}% of the {timed} timed cells against a "
          f"ceiling of {100 * MAX_DRIFT_FRACTION:.0f}%",
        f"{len(failed)} cell(s) failed and were not recovered",
    ]
    for s in failed[:5]:
        detail.append(f"  {s.arm} n={s.tiles} rep={s.repeat}: {s.detail}")
    # DRIFT IS A CEILING AND NOT A ZERO. The cells this run RE-MEASURED after a
    # drift are already back in `ok`: `run_arm` keys its resume on `usable` and
    # not on `status == "ok"`, so a drifted cell is re-timed rather than skipped.
    verdict = PASS if (not failed and not short and not thin
                       and drift_share <= MAX_DRIFT_FRACTION
                       and len(ok) >= max(MIN_TREADS, 2) * MIN_REPEATS
                       * len(ARMS)) else FAIL
    return Gate("V0", VALIDITY, "the run measured the grid it planned",
                verdict,
                f"{len(ok)}/{planned} cells, treads "
                + "/".join(str(per_arm[a]) for a in ARMS)
                + f", drift {100 * drift_share:.1f}%",
                f">= {max(MIN_TREADS, 2)} usable treads and >= {MIN_REPEATS} "
                f"repeats per arm, no unrecovered failure, and DRIFT under "
                f"{100 * MAX_DRIFT_FRACTION:.0f}% of the timed cells",
                "every ladder below was fitted on a grid with holes in it, and "
                "no slope, ratio or interval on this page may be quoted",
                detail)


def gate_v1_matched_geometry(samples, *, block_m: int, treads: list[int]) -> Gate:
    """Did the three arms run the SAME geometry over the SAME treads.

    The ratio is a difference of two slopes, so the two ladders have to be
    measured over one tread set at one tile with one token count per tread. If
    they are not, the ratio is a comparison of two different experiments and the
    number it produces looks exactly as plausible as the one it should have
    been. Asked of the ROWS rather than of the plan: the plan is what was
    intended and the rows are what ran.
    """
    tiles = {arm: {s.tiles for s in samples if s.usable and s.arm == arm}
             for arm in ARMS}
    common = set.intersection(*tiles.values()) if all(tiles.values()) else set()
    mismatched = [arm for arm in ARMS if tiles[arm] != common]
    bms = sorted({s.block_m for s in samples if s.usable})
    tokens = {}
    for s in samples:
        if s.usable:
            tokens.setdefault(s.tiles, set()).add(s.tokens)
    split = sorted(n for n, seen in tokens.items() if len(seen) > 1)
    rows_ok = all(s.rows_per_expert == s.tiles * s.block_m
                  for s in samples if s.usable)
    detail = [
        "treads per arm: "
        + "; ".join(f"{arm}:{sorted(tiles[arm])}" for arm in ARMS),
        f"tread set common to all three arms: {sorted(common)}",
        f"BLOCK_M seen in the rows: {bms}",
        "token count per tread is single-valued: "
        + ("yes" if not split else f"NO, split at treads {split}"),
        "every usable row has rows_per_expert == tiles x BLOCK_M: "
        + ("yes" if rows_ok else "NO"),
    ]
    verdict = PASS if (not mismatched and bms == [block_m] and not split
                       and rows_ok and common) else FAIL
    return Gate("V1", VALIDITY,
                "the three arms ran one geometry over one tread set",
                verdict,
                f"{len(common)} common treads at BLOCK_M={bms}",
                f"all three arms over the same treads at BLOCK_M={block_m}, "
                "one token count per tread",
                "the ratio compares two ladders measured over different grids, "
                "and its value is a fact about the difference between the "
                "grids",
                detail)


def gate_v2_distinct_buffers(proof: BufferProof) -> Gate:
    """Did the private arm allocate AND READ distinct weight buffers.

    PROVEN, NOT ASSERTED, in five parts counted from `PROOF_PARTS`. The part
    that matters most is `kernel_read`: zeroing ONE copy at a time must change
    exactly the tokens routed to that copy. Without it, a relabelling bug that
    sent every M-tile back to copy 0 would produce a private ladder identical
    to the shared one, a ratio of exactly 1.0, and the headline "the whole
    weight set is re-read" -- from an arm that measured no private read at all.
    ONE AT A TIME because zeroing copies 1..n-1 together, which an earlier
    version did, also passes a bug that sends every tile with c >= 1 to copy
    1. `shared_blind` is the control on that control.
    """
    passed = sum(1 for name, _ in PROOF_PARTS if proof.parts.get(name))
    return Gate("V2", VALIDITY,
                "the private arm allocated and READ distinct weight copies",
                proof.verdict,
                f"{passed} of {len(PROOF_PARTS)} parts",
                f"all {len(PROOF_PARTS)} parts of the proof",
                "the private ladder is not a no-reuse reference at all, and the "
                "ratio is a number about the apparatus",
                proof.lines())


def gate_v3_memory(plan: MemoryPlan, *, weight_delta_bytes: int | None,
                   high_water_bytes: int | None) -> Gate:
    """Was the memory the plan predicted the memory the run took.

    TWO PARTS, and they are gated differently because they are known
    differently. The weight allocation is exact arithmetic -- `n x E x 3FH x
    bytes` -- and is gated tight and two-sided: too little means the copies
    were never materialised, too much means something else was allocated under
    their name. The high-water mark includes the framework's own intermediates,
    which this file predicts by a stated ALLOWANCE, so it is gated one-sided
    against that ceiling.
    """
    detail = []
    parts: list[bool | None] = []
    if weight_delta_bytes is None:
        detail.append("the weight allocation delta was not recorded")
        parts.append(None)
    else:
        rel = abs(weight_delta_bytes - plan.weight_bytes) / plan.weight_bytes
        parts.append(rel <= WEIGHT_ALLOC_TOLERANCE)
        detail.append(
            f"weight allocation: {weight_delta_bytes / 1e9:.4f} GB measured "
            f"against {plan.weight_bytes / 1e9:.4f} GB predicted, "
            f"{rel:.2%} apart (gate {WEIGHT_ALLOC_TOLERANCE:.0%})")
    if high_water_bytes is None:
        detail.append("the high-water mark was not recorded")
        parts.append(None)
    else:
        parts.append(high_water_bytes <= plan.predicted_peak_bytes)
        detail.append(
            f"high-water mark: {high_water_bytes / 1e9:.4f} GB against the "
            f"plan's ceiling {plan.predicted_peak_bytes / 1e9:.4f} GB "
            f"(weights {plan.weight_bytes / 1e9:.4f} + {plan.allowance:.1f}x "
            f"the modelled activation set "
            f"{plan.allowance * plan.activation_bytes / 1e9:.4f} + the "
            f"instrument's L2 flush buffer {plan.flush_bytes / 1e9:.4f}, all "
            "in GB)")
    if any(p is None for p in parts):
        verdict = UNKNOWN
    else:
        verdict = PASS if all(parts) else FAIL
    return Gate("V3", VALIDITY,
                "the device memory taken is the memory the plan predicted",
                verdict,
                f"{sum(1 for p in parts if p)} of {len(parts)} parts",
                f"weight allocation within {WEIGHT_ALLOC_TOLERANCE:.0%} of "
                "prediction, high-water mark at or under the plan's ceiling",
                "the copies are not the copies the plan priced, so neither the "
                "no-reuse reference nor the cost of reaching it is what this "
                "page says",
                detail)


def gate_v4_memory_bound(rows, *, roof_tflops: float, roof_source: str) -> Gate:
    """Is every fitted tread of SHARED and PRIVATE on the memory branch.

    A ratio of two slopes is a ratio of two TRAFFIC costs only where traffic is
    what the time is made of. A tread that has run into its compute ceiling has
    a slope set by the tile's arithmetic and not by its reads, and including one
    in either ladder drags that ladder's slope toward the other's, which moves
    the ratio toward 1.0 -- toward the NO-REUSE reading -- for a reason that has
    nothing to do with reuse.

    SCORED AGAINST THE FIXED ROOF, which is what `roofline.ROOF_NOTE_SCORED`
    says a compute-bound gate reads: the calibration's GEMM ran against the
    board power cap, and at full duty so does every cell here, so the fixed
    roof compares delivered throughput under one budget. Below full duty
    (DESIGN DECISION 15) the cells run under the cap at a higher clock than
    the GEMM had, so their fraction of the fixed roof reads HIGH, which errs
    toward calling a tread compute bound: conservative for this gate. The
    own-clock fraction is printed beside each tread as issue efficiency and is
    scored by nothing.
    """
    fitted = [r for r in rows if r["arm"] in RATIO_ARMS]
    hot = [r for r in fitted if r["pct_of_roof"] >= COMPUTE_BOUND_FRACTION]
    worst = max((r["pct_of_roof"] for r in fitted), default=0.0)
    detail = [f"roof {roof_tflops:.1f} TFLOP/s, {roof_source}",
              f"worst fitted tread of shared/private reaches {worst:.1%} of it"]
    if not fitted:
        return Gate("V4", VALIDITY,
                    "every fitted tread of both ladders is memory bound",
                    UNKNOWN, "no shared or private tread was measured",
                    f"< {COMPUTE_BOUND_FRACTION:.0%} of the fixed roof on every "
                    "fitted tread of shared and private",
                    "at least one ladder's slope is set by arithmetic and not "
                    "by reads, which pulls the ratio toward 1.0 for a reason "
                    "that is not reuse",
                    detail + ["nothing to examine: a check that examined "
                              "nothing reports no failures, so this is UNKNOWN "
                              "and not PASS"])
    for r in hot[:5]:
        detail.append(f"  COMPUTE BOUND: {r['arm']} n={r['tiles']} at "
                      f"{r['pct_of_roof']:.1%} of the fixed roof")
    detail.append("own-clock issue efficiency is printed per tread in the "
                  "ladder table and is scored by nothing here; "
                  + roofline.ROOF_NOTE_SCORED)
    return Gate("V4", VALIDITY,
                "every fitted tread of both ladders is memory bound",
                PASS if not hot else FAIL,
                f"worst {worst:.1%} of the fixed roof",
                f"< {COMPUTE_BOUND_FRACTION:.0%} of the fixed roof on every "
                "fitted tread of shared and private",
                "at least one ladder's slope is set by arithmetic and not by "
                "reads, which pulls the ratio toward 1.0 for a reason that is "
                "not reuse",
                detail)


@dataclass(frozen=True)
class DeclarationFit:
    """`ms(NATIVE) - ms(SHARED)` per tread, fitted as
    `a + s 1{n >= switch} + b n`: the declaration's per-tile cost `b` with
    NATIVE's alignment step `s` taken out.

    The two arms read the same bytes at the same addresses in the same order,
    so their difference carries no traffic: a constant (the dead launches of
    the wider declaration, at NATIVE's smaller count), the step NATIVE alone
    crosses (it keeps the study's `E` declaration, so it keeps the switch the
    ratio arms were moved off), and whatever the declaration costs per tile,
    which is what V5 bounds. `step_ms` is None when no switch falls inside the
    treads present, and the fit is then the two-term line.
    """

    switch_tread: int | None
    step_ms: float | None
    per_tile_ms: float
    intercept_ms: float
    points: tuple[tuple[int, float], ...]
    dof: int


def declaration_fit(samples, treads: list[int], switch_tread: int | None,
                    repeats: list[int] | None = None) -> DeclarationFit:
    """Fit the NATIVE - SHARED difference, paired per tread over the same
    repeats, with the step at `switch_tread` when it falls inside the ladder."""
    native, _sp, _d = collapse(samples, NATIVE, repeats)
    shared, _sp, _d = collapse(samples, SHARED, repeats)
    sh = dict(shared)
    pts = [(n, ms - sh[n]) for n, ms in native if n in sh]
    if len(pts) < 3:
        raise Unmeasurable(f"{len(pts)} treads common to native and shared "
                           "cannot carry a step and a slope")
    ys = [d for _n, d in pts]
    ind = [1.0 if (switch_tread is not None and n >= switch_tread) else 0.0
           for n, _d in pts]
    if switch_tread is not None and 0.0 < statistics.fmean(ind) < 1.0:
        a, s_, b = _ols([[1.0, i, float(n)] for (n, _d), i in
                         zip(pts, ind, strict=True)], ys)
        return DeclarationFit(switch_tread, s_, b, a, tuple(pts), len(pts) - 3)
    a, b = _ols([[1.0, float(n)] for n, _d in pts], ys)
    return DeclarationFit(None, None, b, a, tuple(pts), len(pts) - 2)


def declaration_interval(samples, treads: list[int], switch_tread: int | None,
                         draws: int, seed: int, pct: float = INTERVAL_PCT
                         ) -> tuple[tuple[float, float], tuple[float, float] | None, int]:
    """Percentile bootstrap over repeats, paired, for `b` and for `s`."""
    rng = random.Random(seed + 1)
    population = repeat_indices(samples)
    bs: list[float] = []
    ss: list[float] = []
    for _ in range(draws):
        if len(population) < 2:
            break
        drawn = [rng.choice(population) for _ in population]
        try:
            fit = declaration_fit(samples, treads, switch_tread, drawn)
        except Unmeasurable:
            continue
        bs.append(fit.per_tile_ms)
        if fit.step_ms is not None:
            ss.append(fit.step_ms)
    if len(bs) < 2:
        raise Unmeasurable(f"only {len(bs)} of {draws} draws fitted the "
                           "declaration difference")
    tail = (100.0 - pct) / 2.0

    def band(vals: list[float]) -> tuple[float, float]:
        vals = sorted(vals)

        def at(q: float) -> float:
            idx = min(len(vals) - 1, max(0, int(round(q / 100.0 * (len(vals) - 1)))))
            return vals[idx]
        return at(tail), at(100.0 - tail)

    return band(bs), (band(ss) if len(ss) >= 2 else None), len(bs)


def gate_v5_machinery(native: Ladder, shared: Ladder, private: Ladder,
                      fit: DeclarationFit | None = None,
                      per_tile_band: tuple[float, float] | None = None,
                      step_band: tuple[float, float] | None = None,
                      switch_source: str = "",
                      band_absence: str = "") -> Gate:
    """Does the machinery-matched ratio speak for the study's own call.

    THE RATIO IS FORMED BETWEEN TWO CALLS WITH ONE DIFFERENCE. SHARED and
    PRIVATE declare the same `E x n_decl` experts over the same tensors, so
    they share the sorted-id buffer, the launch grid, the dead launches and
    the tile order, and differ only in the addresses tiles read. What that
    pair does NOT share with the study is the declaration: the study's call
    declares `E`. NATIVE is that call, reading SHARED's bytes at SHARED's
    addresses in SHARED's order. `slope(NATIVE) - slope(SHARED)` is therefore
    the per-M-tile cost of the wider declaration with the traffic, the
    addresses and the order held fixed; the dead launches it adds are a
    constant per tread, so in a healthy run this is an intercept shift and the
    slope gap is noise. Divided by `slope(PRIVATE)` -- the ratio's denominator
    -- it is how far the matched ratio can sit from the native one, in the
    ratio's own units.

    WHAT IS NOT BOUNDED HERE, and the module docstring says where the only
    trace of it shows up: a footprint cost paid by READING the copies (the
    private ladder's own residual, printed and scored by nothing).

    A FAILURE IS A VALIDITY FAILURE AND NOT A FINDING. It does not say the
    ratio is wrong; it says the ratio describes a call the study does not make,
    so nothing on the page is quotable as a statement about the study's alpha.
    """
    if private.slope_ms == 0:
        return Gate("V5", VALIDITY,
                    "the declaration's own per-M-tile cost is bounded",
                    UNKNOWN, "no private slope", MACHINERY_WANT,
                    "the ratio's distance from the study's own call is "
                    "unbounded", [])
    raw_gap = native.slope_ms - shared.slope_ms
    gap = fit.per_tile_ms if fit is not None else raw_gap
    rel = abs(gap) / abs(private.slope_ms)
    # THE TWO EDGES OF b, IN THE RATIO'S OWN UNITS. A PASS is a claim about
    # how far the ratio CAN sit from the study's call, so it has to hold at
    # the band's WORST edge. A FAIL is a claim that the declaration's cost IS
    # over the bound, which the measurement makes only when the band's NEAREST
    # edge is over it too -- every other gate here is built the same way (V8
    # fails on the bound over the RESOLVED steps and answers UNKNOWN when only
    # the wide one is over; C1 fails only when the whole interval misses
    # ALPHA_BAND). Scoring the FAIL on the far edge alone printed "the wider
    # declaration changes the per-M-tile cost itself" off a measurement whose
    # point sat well inside the bound.
    edges = [abs(v) for v in per_tile_band] if per_tile_band else []
    far = max([abs(gap)] + edges)
    near = min(edges) if edges else abs(gap)
    rel_far = far / abs(private.slope_ms)
    rel_near = near / abs(private.slope_ms)
    detail = [
        f"slope(native)  {native.slope_ms:.6f} ms per M-tile  "
        f"(the study's call: E declared)",
        f"slope(shared)  {shared.slope_ms:.6f} ms per M-tile  "
        f"(same bytes, addresses and order; E x n_decl declared)",
        f"slope(private) {private.slope_ms:.6f} ms per M-tile",
    ]
    if fit is not None:
        detail.append(
            f"native - shared per tread, fitted as a + s 1{{n >= "
            f"{fit.switch_tread if fit.switch_tread else '-'}}} + b n over "
            f"{len(fit.points)} treads ({fit.dof} dof): b = {fit.per_tile_ms:+.6f} "
            "ms per M-tile"
            + (f" [{per_tile_band[0]:+.6f}, {per_tile_band[1]:+.6f}]"
               if per_tile_band else "")
            + (f", s = {fit.step_ms * 1e3:+.2f} us"
               + (f" [{step_band[0] * 1e3:+.2f}, {step_band[1] * 1e3:+.2f}]"
                  if step_band else "")
               + " -- NATIVE's alignment step, taken OUT of b"
               if fit.step_ms is not None else
               ", no switch inside the ladder, so no step term")
            + f", a = {fit.intercept_ms:+.4f} ms")
        if switch_source:
            detail.append(f"the switch tread came from {switch_source}")
        detail.append(
            f"the raw slope gap native - shared is {raw_gap:+.6f} ms per M-tile "
            f"({abs(raw_gap) / abs(private.slope_ms):.2%} of the private "
            "slope); it carries the step at the design's leverage and is "
            "printed, not scored")
    detail += [
        f"declaration = b = {gap:+.6f} ms per M-tile, which is {rel:.2%} of "
        "the private slope",
        f"intercepts: native {native.intercept_ms:.4f} ms, shared "
        f"{shared.intercept_ms:.4f} ms -- the dead launches of the wider "
        "declaration belong HERE, as a constant, and are not scored",
        "read b as the additive error bar on reading the ratio as the native "
        + (f"call's: about +/-{rel_far:.3f} at the band's far edge, over and "
           "above the ratio's own bootstrap interval" if per_tile_band else
           f"call's: about +/-{rel:.3f} at the POINT, which is all there is "
           "without a band, and over and above the ratio's own interval"),
    ]
    if rel_near >= MACHINERY_BOUND:
        verdict = FAIL
    elif per_tile_band is None:
        # The point is under the bound but nothing says how far b can sit
        # from it, and a PASS is a claim about the far edge.
        verdict = UNKNOWN
        detail.append("NO BAND on b, so only the point was scored: the point "
                      "is under the bound but its far edge is unknown"
                      + (f" -- {band_absence}" if band_absence else ""))
    elif rel_far >= MACHINERY_BOUND:
        verdict = UNKNOWN
        detail.append(
            f"b's band STRADDLES the bound: its near edge is {rel_near:.2%} "
            f"of the private slope and its far edge {rel_far:.2%}, against "
            f"{MACHINERY_BOUND:.0%}. The measurement neither bounded the "
            "declaration's per-M-tile cost nor showed it over: not a finding "
            "about the declaration, a statement about this run's precision")
    else:
        verdict = PASS
    return Gate("V5", VALIDITY,
                "the declaration's own per-M-tile cost is bounded",
                verdict,
                f"{rel_far:.2%} of the private slope at b's far edge, "
                f"{rel_near:.2%} at its near edge (point {rel:.2%})",
                MACHINERY_WANT,
                "the wider declaration changes the per-M-tile cost itself, so "
                "the matched ratio is about a call the study does not make, "
                "and nothing on this page may be quoted as the study's alpha",
                detail)


def _median_by_arm(samples, tread: int, field_name: str) -> dict[str, float]:
    out = {}
    for arm in ARMS:
        vals = [getattr(s, field_name) for s in samples
                if s.usable and s.arm == arm and s.tiles == tread
                and getattr(s, field_name)]
        if vals:
            out[arm] = statistics.median(vals)
    return out


def _arm_medians(samples, tread: int, value) -> dict[str, float]:
    """Each arm's median of `value(sample)` over its usable cells at `tread`,
    an arm with no value left out. Not `_median_by_arm`, which drops a FALSY
    value: right for a time or a clock, wrong for a sag of exactly zero."""
    out = {}
    for arm in ARMS:
        vals = [v for s in samples
                if s.usable and s.arm == arm and s.tiles == tread
                for v in (value(s),) if v is not None]
        if vals:
            out[arm] = statistics.median(vals)
    return out


def gate_v6_identity(samples, *, identity_tread: int = 1) -> Gate:
    """At one M-tile per expert SHARED and PRIVATE are the SAME CALL.

    One tile per expert, so the private relabelling routes every tile to copy
    0 and the two calls pass identical ids, tensors and declarations. The gap
    between their medians is not a difference between arms, it is THIS ARM'S
    OWN FLOOR for a slope comparison, measured on the card that will carry
    the ratio rather than imported from another session.

    NATIVE IS NOT IN THE COMPARISON. It differs from SHARED by the declaration
    at every tread, n = 1 included -- a constant, which belongs in an
    intercept -- and an earlier version that required all three to agree here
    would have refused a correct instrument over it. Its offset is printed.

    WHAT A DISAGREEMENT WOULD MEAN, registered before the run: a governor that
    moves between calls seconds apart, an allocator state that does not reset,
    or an ordering effect the rotation did not remove. The ratio would then be
    unreadable whatever value it took, which is why this is VALIDITY.
    """
    med = _median_by_arm(samples, identity_tread, "ms_p50")
    pair = [a for a in RATIO_ARMS if a in med]
    if len(pair) < len(RATIO_ARMS):
        return Gate("V6", VALIDITY,
                    f"shared and private agree at n={identity_tread}, where "
                    "they are the same call",
                    UNKNOWN,
                    f"{len(pair)} of {len(RATIO_ARMS)} arms reached "
                    f"n={identity_tread}",
                    f"both ratio arms measured at n={identity_tread}",
                    "this arm's own instrument floor is unknown, so the ratio "
                    "cannot be read against it",
                    [f"medians at n={identity_tread}: "
                     + ", ".join(f"{a}:{med[a]:.4f} ms" for a in sorted(med))])
    lo, hi = min(med[a] for a in pair), max(med[a] for a in pair)
    rel = (hi - lo) / lo if lo > 0 else math.inf
    detail = [f"medians at n={identity_tread}: "
              + ", ".join(f"{a}:{med[a]:.4f} ms" for a in ARMS if a in med),
              f"shared/private gap {rel:.3%} of the faster",
              "this is the floor the ratio's own difference has to stand "
              "above, measured on this card in this session"]
    if NATIVE in med and med[SHARED] > 0:
        detail.append(f"native sits {med[NATIVE] / med[SHARED] - 1:+.3%} from "
                      "shared here: the declaration's constant, RECORDED and "
                      "not scored")
    return Gate("V6", VALIDITY,
                f"shared and private agree at n={identity_tread}, where they "
                "are the same call",
                PASS if rel <= IDENTITY_SPREAD else FAIL,
                f"{rel:.3%} gap",
                f"<= {IDENTITY_SPREAD:.1%}",
                "one call timed twice does not agree with itself, so the "
                "difference between the two arms at depth is not attributable "
                "to traffic and the ratio is unreadable at any value",
                detail)


#: The duty the pod runs this arm at (DESIGN DECISION 15), and the evidence
#: for it, QUOTED from session 4's clock arm (the 2026-09-21 G=16 mixtral
#: arm, run 9f91fa91, results/published/2026-09-21-nvidia_h200-session4/ on
#: the pod-h200-session4 branch) for the remedy V7 prints; nothing scores it.
FLAT_DUTY = 0.25
FLAT_DUTY_EVIDENCE = (
    "session 4's clock arm, 2026-09-21, G=16 mixtral: at duty 0.5 the clock "
    "still tracked board power, -1.09 MHz/W over 1882-1965 MHz; at duty 0.25 "
    "every tread's median read 1965 MHz and no cell drifted; at duty 0.1, "
    "1965-1980 MHz")


def v7_remedy(duty: float) -> str:
    """What a V7 FAIL tells the operator to do, at the duty the ladder ran.
    BELOW FULL DUTY TOO: a split there is a duty not yet low enough, and a
    FAIL page with no remedy on it leaves the operator to re-run at the same
    duty rather than lower it."""
    if duty >= 1.0:
        return (f"the remedy is --duty {FLAT_DUTY}: bursts of kernel time with "
                "idle gaps lower the arms' average board power until the clock "
                f"stops following it ({FLAT_DUTY_EVIDENCE})")
    return (f"the remedy is a lower --duty than {duty:.2f}: the arms' clocks "
            "still split at this duty (each arm's board power, where it was "
            "read, is printed above)"
            + (f"; --duty {FLAT_DUTY} is the pod setting" if duty > FLAT_DUTY
               else "")
            + f" ({FLAT_DUTY_EVIDENCE})")


def gate_v7_clock_parity(samples, *, treads: list[int]) -> Gate:
    """Did PRIVATE and SHARED run at the same clock, tread by tread.

    THE RATIO SAYS "AT THE SAME CLOCK" AND THIS IS WHERE THAT IS CHECKED. The
    card is power capped and its SM clock is an outcome of the work; PRIVATE
    moves up to `n` times the weight bytes of SHARED per call, so it can
    settle at a different clock, and a slope measured at a different clock is
    a slope with the clock problem in it -- the one this arm exists to avoid.
    Scored at every tread both arms reached, on the per-tread MEDIAN of the
    under-load clock the cell's instrument records: `time_kernel`'s poller at
    full duty, and below it `clock_elasticity.time_duty`'s one NVML read per
    burst, taken with the burst still in flight.

    AT FULL DUTY on a card that cannot lock its clock this fails by
    construction whenever the arms draw different power. Below it (DESIGN
    DECISION 15) it holds when the duty is low enough that the clock no
    longer follows power, and a FAIL there says this duty was not; the
    remedy printed on a FAIL names a lower duty in both cases.

    UNKNOWN, NOT PASS, when a tread has no clock in either arm: an unread
    clock is not a matching one.

    PRINTED BESIDE THE CLOCKS AND SCORED BY NOTHING: each arm's median board
    power per tread (`power_w`, NVML's ~1 s average, so duty-averaged below
    full duty) and each arm's median in-burst sag (`burst_sag`, the duty
    timer's first and last quarter of a burst). A FAIL is then readable off
    the page: a systematic power difference between the arms is a split, the
    same power in both is a card jittering.
    """
    detail = []
    worst = 0.0
    unread = []
    over = []
    records: list[str] = []
    averaged = (", duty-averaged over bursts and idle gaps"
                if duty_of(samples) < 1.0 else "")
    order = (*RATIO_ARMS, NATIVE)
    for n in treads:
        # A tread one ratio arm never reached is V0's and V1's to score; a
        # tread both reached with no clock in one of them is unread HERE.
        timed = _median_by_arm(samples, n, "ms_p50")
        if not all(a in timed for a in RATIO_ARMS):
            continue
        med = _median_by_arm(samples, n, "sm_clock_load_mhz")
        if not all(a in med for a in RATIO_ARMS):
            unread.append(n)
            continue
        rel = abs(med[PRIVATE] - med[SHARED]) / med[SHARED]
        worst = max(worst, rel)
        if rel > CLOCK_PARITY:
            over.append(n)
        detail.append(f"n={n}: shared {med[SHARED]:.0f} MHz, private "
                      f"{med[PRIVATE]:.0f} MHz, {rel:.2%} apart"
                      + (f"; native {med[NATIVE]:.0f} MHz" if NATIVE in med
                         else ""))
        power = _arm_medians(samples, n, lambda s: s.power_w)
        if power:
            detail.append(f"      power (NVML's ~1 s average{averaged}): "
                          + ", ".join(f"{a} {power[a]:.0f} W"
                                      for a in order if a in power))
            records.append("power")
        sag = _arm_medians(samples, n, lambda s: s.burst_sag)
        if sag:
            detail.append("      in-burst sag (tail - head) / head: "
                          + ", ".join(f"{a} {sag[a]:+.2%}"
                                      for a in order if a in sag))
            records.append("in-burst sag")
    if unread:
        detail.append(f"no clock in one or both ratio arms at treads {unread}")
    if not detail:
        # No tread reached by both ratio arms: nothing was compared, and a
        # check that examined nothing reports no failures. V0 and V1 own that
        # state; this gate says UNKNOWN rather than PASS over it.
        detail.append("no tread was reached by both ratio arms, so no clocks "
                      "were compared")
        verdict = UNKNOWN
    elif unread:
        verdict = UNKNOWN
    else:
        verdict = PASS if not over else FAIL
    if records:
        named = list(dict.fromkeys(records))
        detail.append(" and ".join(named)
                      + (" are" if len(named) > 1 else " is")
                      + " each arm's median over its cells at the tread, "
                      "RECORDS: V7 scores the clocks alone")
    if over:
        detail.append(v7_remedy(duty_of(samples)))
    return Gate("V7", VALIDITY,
                "shared and private ran at the same clock at every tread",
                verdict,
                f"worst {worst:.2%}"
                + (f", over at treads {over}" if over else "")
                + (f", unread at treads {unread}" if unread else ""),
                f"<= {CLOCK_PARITY:.0%} at every tread, clock read in both arms",
                "the two slopes were taken at different clocks on a card whose "
                "time moves with its clock, so the raw ratio carries a clock "
                "difference and may not be quoted; the clock-corrected ratio "
                "this page may print beside it is a model's number, scored by "
                "nothing",
                detail)


def gate_v8_alignment(probe: AlignProbe | None, *, treads: list[int],
                      census: PathCensus, weight_stream_ms: float,
                      ratio_label: str = SHARED) -> Gate:
    """Are the two ratio arms on ONE alignment kernel along the whole ladder,
    as MEASURED, and is what is left worth less than the budget.

    The probe timed vLLM's alignment op alone at every tread, once per ARM.
    `step_fit` finds the best single step in each series, and
    `pair_step_bias` says what SHARED's step and PRIVATE's TOGETHER could do
    to the ratio through a straight-line fit -- with no premise that the two
    are one step, which is what probing PRIVATE's id set bought.
    (`step_bias`, the common-step bound, is the fallback when PRIVATE's
    series was not probed.) PASS when that bias is under
    `ALIGN_STEP_RATIO_BUDGET`; FAIL when the bound over the RESOLVED steps
    is over it, because a design is not refused on a step this gate called
    noise; UNKNOWN when only the wide bound is over (a noisy probe has shown
    neither), or when no probe ran. The host-bound census covers every series
    that enters the bias, not SHARED's alone.

    NATIVE's declaration is read beside it. It is never scored as a design
    fault -- NATIVE is allowed its switch, and the V5 fit takes it out. The
    probe times the op UNDER A CUDA GRAPH (`PROBE_CALLS_PER_REPLAY` calls per
    replay), so its cells are GPU time by construction and NATIVE's switch
    beside them CONFIRMS the cited source on this build or BOUNDS its cost
    under the fit's own threshold (`gpu_time_control`); it never withholds
    the verdict, because the reason a control ever gated was host blindness,
    which the graph removes, and the threshold prices the noise. On the EAGER
    fallback (capture refused, named on the page), or on cells the instrument
    called host-bound anyway, NATIVE IS the probe's POSITIVE CONTROL
    (`native_control`): an eager host-bound cell times the host's enqueue
    cost, which on an H200 is several times the alignment kernel's own
    (session 4: 32-36 us against a few), so a flat ratio series from such a
    probe is not by itself evidence that nothing stepped. NATIVE declares E,
    under the expert bound, and the census puts its switch at
    `census.switch_tread(NATIVE)`: if the probe resolves NATIVE's step AT that
    tread, it has shown it can see a kernel switch of this op at this size
    through whatever host cost is present, and the ratio series is scored
    exactly as a GPU-bound one would be. If it does not, UNKNOWN, and the
    page says it means "the probe could not see the switch it was shown".
    """
    if probe is None:
        return Gate("V8", VALIDITY,
                    "the ratio arms take one alignment kernel along the ladder",
                    UNKNOWN, "no probe ran",
                    f"step bias on the ratio <= {ALIGN_STEP_RATIO_BUDGET}",
                    "an alignment step of unknown size may sit inside both "
                    "ratio slopes", [])
    detail = []
    readings = {}
    for label in probe.labels():
        try:
            readings[label] = read_probe(probe, label, census)
        except Unmeasurable as exc:
            detail.append(f"{label}: series not fitted: {exc}")
    if probe.synthetic:
        detail.append("this probe was PLANTED by --self-test; nothing here was "
                      "read off a device")
    if ratio_label not in readings:
        return Gate("V8", VALIDITY,
                    "the ratio arms take one alignment kernel along the ladder",
                    UNKNOWN, "the ratio declaration was not probed",
                    f"step bias on the ratio <= {ALIGN_STEP_RATIO_BUDGET}",
                    "an alignment step of unknown size may sit inside both "
                    "ratio slopes", detail)
    r = readings[ratio_label]
    rp = readings.get(PRIVATE) if ratio_label == SHARED else None
    graph_calls = r.graph_calls
    detail.append(instrument_line(graph_calls, probe))

    def step_of(reading, resolved_only: bool = False):
        """`(step_ms, split)` for the bound, or None for "no step here".

        `resolved_only` drops a step the series' own standard error did not
        resolve, and TWO BOUNDS COME OUT OF THIS. `bias` is the widest
        defensible bound, over every fitted step, and is what the page
        reports; `bias_real` is the bound over the steps the rule RESOLVED,
        and is what may refuse a design. Scoring the FAIL on the wide one let
        an unresolved noise step in one arm carry the verdict while the other
        arm's resolved step supplied `real` -- a FAIL earned by a step this
        gate itself called noise, which on a pod skips the whole sweep.
        """
        if reading is None:
            return None
        f = reading.fit
        if f.split_tread is None or (resolved_only and not reading.real):
            return None
        return (f.step_ms, f.split_tread)

    if rp is not None:
        bias = pair_step_bias(step_of(r), step_of(rp), treads, weight_stream_ms)
        bias_real = pair_step_bias(step_of(r, True), step_of(rp, True),
                                   treads, weight_stream_ms)
        real = r.real or rp.real
    else:
        bias = (0.0 if r.fit.split_tread is None else
                step_bias(r.fit.step_ms, treads, r.fit.split_tread,
                          weight_stream_ms))
        bias_real = bias if r.real else 0.0
        real = r.real
    for reading in readings.values():
        detail += reading.lines()
    if rp is not None:
        detail += private_ids_lines(probe, r, rp, weight_stream_ms)
    else:
        detail.append("PRIVATE's id set was NOT probed: the bias below takes "
                      "the ratio arms' step as COMMON to both, a premise "
                      "nothing here checked")
    def step_said(reading):
        f = reading.fit
        return ("no step resolved a split" if f.split_tread is None else
                f"{f.step_ms * 1e3:+.2f} us at tread {f.split_tread}, at "
                f"leverage {leverage(treads, f.split_tread):.3f}")
    detail.append(
        f"{ratio_label}'s step: {step_said(r)}"
        + (f"; {PRIVATE}'s step: {step_said(rp)}" if rp is not None else "")
        + f"; against a weight stream of {weight_stream_ms:.4f} ms (the "
        f"denominator's own scale) they could move the ratio by at most "
        f"{bias:.4f}"
        + (f", and by {bias_real:.4f} over the RESOLVED steps alone, which is "
           "the number a FAIL is taken on" if bias_real != bias else ""))
    # EVERY SERIES THAT ENTERS THE BIAS, not SHARED's alone. PRIVATE's id set
    # has been scored since step 5, and the two are not host-bound together:
    # the host's enqueue cost is the same for both (one op, one declaration,
    # one call shape, ids built outside the timed region) while their GPU
    # time differs by exactly the counter asymmetry this gate prints below,
    # so the instrument's verdict is the sign test g > h and each series
    # crosses at its own tread. Reading SHARED alone let a host-timed PRIVATE
    # series be scored with the control never consulted.
    scored_labels = [ratio_label] + ([PRIVATE] if rp is not None else [])
    counts = {lab: probe.host_bound(lab) for lab in scored_labels}
    hot = sum(c[0] for c in counts.values())
    judged = sum(c[1] for c in counts.values())
    note = next((c[2] for c in counts.values() if c[2]), "")
    if graph_calls and hot:
        # The instrument's own remedy is "time the callable as a graph
        # replay", which is what these cells already were: name the anomaly.
        note = (f"the replay's launch outran {graph_calls} calls of GPU work, "
                "an anomaly on this op; raise PROBE_CALLS_PER_REPLAY")
    if not judged:
        detail.append("the instrument returned no host-bound verdict for any "
                      "probed cell")
    elif len(scored_labels) == 1:
        detail.append(
            f"the instrument called {hot} of {judged} probed cells HOST-BOUND "
            "at this declaration" + (f": {note}" if note else ""))
    else:
        # PER ARM AND NOT POOLED: an asymmetry here is itself a reading. It
        # says the op's GPU time straddles the host's enqueue cost between
        # the two id sets, which is the asymmetry private_ids_lines measures.
        detail.append(
            "the instrument called "
            + ", ".join(f"{counts[lab][0]} of {counts[lab][1]} of {lab}'s"
                        for lab in scored_labels)
            + " probed cells HOST-BOUND at the ratio arms' declaration"
            + (f": {note}" if note else ""))
    control = None
    if judged == 0 or hot:
        # AN EAGER HOST-BOUND PROBE TIMES THE HOST; a graph-timed one the
        # instrument still called host-bound is an anomaly (the replay's
        # launch outran N calls). Either way it is not evidence on its own,
        # and NATIVE's switch is the positive control that decides whether it
        # became evidence anyway.
        control, control_lines = native_control(readings, census, hot=hot,
                                                judged=judged,
                                                native=probe.host_bound(NATIVE),
                                                graph_calls=graph_calls)
        detail += control_lines
    elif graph_calls:
        # GPU TIME BY CONSTRUCTION: the control confirms or bounds NATIVE's
        # switch and never withholds the verdict; the threshold prices noise.
        detail += gpu_time_control(readings, census, treads=treads,
                                   weight_stream_ms=weight_stream_ms)
    if control is False:
        verdict = UNKNOWN
    elif bias <= ALIGN_STEP_RATIO_BUDGET:
        verdict = PASS
    elif real and bias_real > ALIGN_STEP_RATIO_BUDGET:
        verdict = FAIL
    elif real:
        verdict = UNKNOWN
        detail.append(
            f"over budget at {bias:.4f} only through a step the rule did NOT "
            f"resolve: the RESOLVED steps alone are worth {bias_real:.4f}, "
            "inside the budget. A design is not refused on a step this gate "
            "called noise, and it is not certified while an unresolved step "
            "could be that big; more --probe-repeats is what closes it")
    else:
        verdict = UNKNOWN
        detail.append("the step is over budget but NOT resolved against its own "
                      "standard error: the design is not shown sound, and not "
                      "shown unsound; more --probe-repeats is what closes it")
    return Gate("V8", VALIDITY,
                "the ratio arms take one alignment kernel along the ladder",
                verdict,
                f"bias <= {bias:.4f}"
                + (f", {bias_real:.4f} over the resolved steps"
                   if bias_real != bias else "")
                + (", a REAL step" if real else ", no step resolved"),
                f"step bias on the ratio <= {ALIGN_STEP_RATIO_BUDGET}, from "
                "the probed series at the ratio arms' own declaration",
                "vLLM changes alignment kernel inside the ratio arms' ladder "
                "on this build, and the step it leaves -- in one of the two "
                "slopes or in both, which the page names -- is worth more "
                "than the budget; the ratio is not quotable at this "
                "declaration",
                detail)


def private_ids_lines(probe: AlignProbe, shared: ProbeReading,
                      private: ProbeReading, weight_stream_ms: float
                      ) -> list[str]:
    """What PRIVATE's id set costs the alignment op beyond SHARED's, per
    tread, and whether the two arms' steps are one step.

    The plan page registers the counter asymmetry -- SHARED's E counters take
    BLOCK_M n increments each, PRIVATE's E x n take BLOCK_M each -- as a
    per-tread term in the numerator alone. This is that term, MEASURED:
    `slope(private ids) - slope(shared ids)` in us per tread, and in the
    ratio's units against the weight stream. Printed, not scored."""
    series = probe.series(SHARED)
    if len(series) >= 2 and series[-1][0] != series[0][0]:
        ids_per_tread = ((series[-1][1] - series[0][1])
                         / (series[-1][0] - series[0][0]))
    else:
        ids_per_tread = 0.0
    diff_ms = (private.fit.slope_ms_per_id - shared.fit.slope_ms_per_id) \
        * ids_per_tread
    neither = not shared.real and not private.real
    same = (shared.real and private.real
            and shared.fit.split_tread == private.fit.split_tread)
    return [
        f"slope(private ids) - slope(shared ids) = {diff_ms * 1e3:+.3f} us per "
        f"tread ({ids_per_tread:.0f} ids a tread), "
        + (f"{diff_ms / weight_stream_ms:+.5f} of the weight stream per tread"
           if weight_stream_ms > 0 else "no weight stream to scale it by")
        + ": the alignment's counter asymmetry the plan page registers, "
        "MEASURED; printed, not scored",
        "the ratio arms' steps: "
        + ("NEITHER arm resolved one" if neither else
           "ONE STEP (both resolved, same split)" if same else
           "NOT the same step: the bias is taken on both, with no premise "
           "that they are common")
        + f"; shared {shared.fit.step_ms * 1e3:+.2f} us at "
        f"{shared.fit.split_tread or '-'}, private "
        f"{private.fit.step_ms * 1e3:+.2f} us at "
        f"{private.fit.split_tread or '-'}",
        "the bias bound is pair_step_bias, ASSUMED ratio <= 1: the shared arm "
        "re-reads no more than the whole set per M-tile"]


def native_step_is_admissible(reading, probe: AlignProbe | None
                              ) -> tuple[bool, str]:
    """May NATIVE's probed step stand in for the census hypothesis:
    `(admissible, why)`.

    ADMISSIBLE MEANS TIMED ON THE MACHINE THE STEP IS ABOUT. NATIVE's cells
    are what the positive control is read from, and a cell the instrument
    called host-bound timed the host's enqueue cost, not the kernel: a step
    in it is a step in the wrong machine's time. So the probe's split
    replaces the hypothesis only when NATIVE's own cells were JUDGED and
    none came back host-bound. "No verdict at all" is not GPU time and is
    not admissible -- V8's own prose already says such cells establish
    nothing.

    WHY NOT "the positive control confirmed it": the control confirms when
    the probe's split EQUALS the census tread, so taking the probe's split
    only then is an override that can never change anything. The question
    the override answers is the opposite one -- whether to believe a probe
    that DISAGREES with the hypothesis -- and only the machine it was timed
    on can answer it.

    ONE HOME. `native_control` scores the control for V8 and `analyse`
    chooses the tread `declaration_fit` is given; both read this, so the
    rule is not written twice.
    """
    if reading is None or not reading.real:
        return False, "the probe resolved no step for native"
    if probe is None:
        return False, "no probe"
    hot, judged, _note = probe.host_bound(NATIVE)
    if not judged:
        return False, ("the instrument returned no host-bound verdict for "
                       "native's own cells, so what they timed is not "
                       "established")
    if hot:
        g = probe.graph_calls(NATIVE)
        if g:
            return False, (f"{hot} of {judged} of native's own cells came back "
                           f"host-bound UNDER A CUDA GRAPH of {g} calls per "
                           "replay: the replay's launch outran that much GPU "
                           "work, an anomaly on this op, so what they timed "
                           "is not established; raise PROBE_CALLS_PER_REPLAY")
        return False, (f"{hot} of {judged} of native's own cells were "
                       "HOST-BOUND, so the step in them is a step in the "
                       "host's enqueue cost and not in the kernel")
    return True, (f"native's own {judged} cells were judged and none was "
                  "host-bound, so its step was read in GPU time"
                  + (f" under a CUDA graph of {probe.graph_calls(NATIVE)} calls "
                     "per replay, GPU time by construction"
                     if probe.graph_calls(NATIVE) else ""))


def native_control(readings: dict, census: PathCensus, *, hot: int,
                   judged: int, native: tuple[int, int, str],
                   graph_calls: int = 0) -> tuple[bool, list[str]]:
    """Did the probe resolve NATIVE's kernel switch at the tread the census
    puts it: `(confirmed, the lines that say so)`. Reached on the EAGER
    fallback and on graph-timed cells the instrument still called host-bound;
    `graph_calls` says which, so the state names the instrument and the
    remedy does not prescribe what was already done.

    `hot`/`judged` are the ratio series' own host-bound counts and `native`
    is NATIVE's, so the lines say which case they are in instead of asserting
    one. This branch is reached on two of them -- cells the instrument called
    host-bound, and cells it could not judge at all -- and the confirmed line
    used to open "the probe was host-bound" on both.

    THE ASSUMPTION THIS RESTS ON, registered here and printed on the page:
    the host's enqueue cost does not itself step at that tread. The dispatch
    is the same call at every tread and at both declarations; only `numel`
    and `num_experts` move. If the host cost stepped there, a resolved NATIVE
    step could be the host's, and the control would certify nothing.
    """
    want = census.switch_tread(NATIVE)
    assumption = ("ASSUMED: the host's enqueue cost does not itself step at "
                  "that tread -- the dispatch is the same call at every tread "
                  "and only numel and num_experts move")
    if want is None:
        return False, [
            "POSITIVE CONTROL UNAVAILABLE: the census puts NATIVE on one "
            "alignment kernel throughout this ladder, so there is no switch "
            "to show the host-bound probe; UNKNOWN means the instrument was "
            "not demonstrated, not that the design is doubted"]
    graph = (f" under a CUDA graph with {graph_calls} calls per replay -- the "
             f"replay's launch outran {graph_calls} calls of GPU work, an "
             "anomaly on this op" if graph_calls else ", timed eagerly")
    state = (f"the probe's ratio cells were host-bound ({hot} of {judged}){graph}"
             if hot else
             "the instrument returned no host-bound verdict for the ratio "
             "cells, so what they timed is not established")
    native_hot, native_judged, _note = native
    seen = (f"NATIVE's own cells were host-bound in {native_hot} of "
            f"{native_judged}" if native_hot else
            f"NATIVE's own {native_judged} judged cells were NOT called "
            "host-bound, so the switch below was seen in GPU time and this "
            "control does not demonstrate the same sensitivity under host "
            "cost" if native_judged else
            "NATIVE's own cells carried no host-bound verdict")
    r = readings.get(NATIVE)
    if r is not None and r.real and r.fit.split_tread == want:
        # The control is the PREDICTION met, which is a different question
        # from `native_step_is_admissible` (was the step timed on the GPU):
        # a host-bound probe that finds the switch where the census puts it
        # has shown its sensitivity, which is the whole point of the control.
        return True, [
            f"POSITIVE CONTROL CONFIRMED: {state}, but the probe resolved "
            f"NATIVE's step at tread {want}, where the census puts its kernel "
            f"switch ({r.fit.step_ms * 1e3:+.2f} us against a threshold of "
            f"{r.fit.threshold_ms() * 1e3:.2f} us); it has shown it can see a "
            "kernel switch of this op at this size, so the ratio series is "
            "scored as evidence",
            f"  {seen}",
            f"  {assumption}"]
    got = ("was not probed" if r is None
           else "resolved no step" if not r.real
           else f"resolved its step at tread {r.fit.split_tread}, not {want}")
    return False, [
        f"POSITIVE CONTROL NOT CONFIRMED: {state}; NATIVE's switch is due at "
        f"tread {want} and the probe {got}. UNKNOWN therefore means one "
        "specific thing: this probe could not see a kernel switch it was "
        "shown, so a flat ratio series from it certifies nothing, and a step "
        "in it is not evidence of a kernel switch either. What would close "
        "it is a timing that excludes the host: read the step from a "
        + ("profiler, or raise PROBE_CALLS_PER_REPLAY so the replay outlasts "
           "its own launch" if graph_calls else
           "profiler, or time the op under a CUDA graph (the probe's default; "
           "this probe ran eagerly, see the instrument line above)"),
        f"  {assumption}"]


def instrument_line(graph_calls: int, probe: AlignProbe) -> str:
    """What timed the probe's cells, on the page: nothing printed
    `AlignProbe.note` before this, so a refused capture reached report.json
    and never the pod log."""
    if graph_calls:
        return (f"the probe timed the op under a CUDA graph, {graph_calls} "
                "calls per replay: each cell is GPU time (the replay's p50 "
                "over that count) and no host enqueue sits in it")
    return ("the probe timed the op EAGERLY, so a host-bound cell timed the "
            "host" + (f" ({probe.note})" if probe.note and not probe.synthetic
                      else ""))


def gpu_time_control(readings: dict, census: PathCensus, *, treads: list[int],
                     weight_stream_ms: float) -> list[str]:
    """NATIVE's switch read beside a GRAPH-TIMED ratio series: informational,
    never a verdict.

    The ratio series are GPU time by construction, so V8 rests on them and
    NATIVE's reading is a statement about the BUILD and the INSTRUMENT's
    resolution, not a licence: resolved at the census tread, the cited
    source's switch is where it says on this build and the instrument
    resolves a step of that size at this geometry; resolved elsewhere, a
    finding about the build (`analyse` fits V5 at the probe's tread, since
    `native_step_is_admissible` is True); NOT resolved, a BOUND: whatever
    switch NATIVE makes at the census tread costs under the fit's threshold
    per call, which is a fact about the kernel and not a blind spot, because
    no host cost sits in a graph replay. The lever on a loose bound is a
    denser probe ladder (its treads need not be the arms'), not more
    --probe-repeats: at six treads the threshold is 16.5 standard errors and
    session 4's eager cells spread 0.04-0.12 us across repeats, so at that
    depth a 2 us switch resolves or not on the noise, which the pod decides.
    """
    want = census.switch_tread(NATIVE)
    r = readings.get(NATIVE)
    if want is None:
        return ["POSITIVE CONTROL UNAVAILABLE, and not needed: the census puts "
                "NATIVE on one alignment kernel throughout this ladder, so "
                "there is no switch to read; the ratio series are GPU time by "
                "construction and V8 rests on them"]
    if r is None:
        return ["NATIVE's declaration was not probed; the ratio series are GPU "
                "time by construction and V8 rests on them"]
    f = r.fit
    thr_us = f.threshold_ms() * 1e3
    step = f"{f.step_ms * 1e3:+.2f} us +/- {f.step_se * 1e3:.2f} us"
    if r.real and f.split_tread == want:
        return [f"POSITIVE CONTROL CONFIRMED IN GPU TIME: the probe resolved "
                f"NATIVE's step at tread {want}, where the census puts its "
                f"kernel switch ({step} against a threshold of {thr_us:.2f} "
                "us): the cited source's switch is where it says on this "
                "build, and the instrument resolves a step of that size at "
                "this geometry"]
    if r.real:
        return [f"NATIVE's switch resolved at tread {f.split_tread} in GPU "
                f"time, not the census tread {want} ({step}): a finding about "
                "the build; V5 is fitted at the probe's tread"]
    worth = step_bias(f.threshold_ms(), treads, want, weight_stream_ms)
    return [f"NATIVE's switch NOT RESOLVED IN GPU TIME: best split at tread "
            f"{f.split_tread if f.split_tread else '-'}, step {step}, "
            f"threshold {thr_us:.2f} us at {f.dof} dof over {f.splits_tried} "
            f"split(s), so whatever switch NATIVE makes at tread {want} costs "
            f"under {thr_us:.2f} us per call -- a bound on the KERNEL, not a "
            "blind spot: no host cost sits in a graph replay. A step that size "
            f"in a ratio arm at that tread would be worth {worth:.5f} of the "
            "ratio; a denser probe ladder is what tightens it, not more "
            "--probe-repeats"]


def c1_verdict(ratio: float, interval: tuple[float, float]) -> str:
    """PASS, FAIL or UNKNOWN for C1, from the point AND the interval together.

    PASS only when the point lands in ALPHA_BAND and the WHOLE interval sits
    inside it. FAIL only when the interval misses ALPHA_BAND entirely, with the
    band's own half-open membership: `[lo, hi)`, so an interval starting at
    the upper edge misses it. Everything else -- the point outside the band,
    or an interval that reaches past either edge -- is UNKNOWN: the
    measurement did not resolve the claim.

    WHY INSIDE AND NOT MERELY RESOLVED, which is what stood here: PASS used to
    need the interval only to exclude the two alternative worlds, so an
    interval reaching into ABOVE-THE-REFIT-BAND passed as "refit confirmed".

    WHY NOT PLAIN OVERLAP, which is what stood here: a ratio of 0.600 with an
    interval [0.585, 0.615] overlapped the band, PASSED, and exited DONE while
    the same page named the world ABOVE-THE-REFIT-BAND; and an interval wide
    enough to reach 0.3 passed as "refit confirmed".
    """
    lo, hi = interval
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return UNKNOWN
    if hi < ALPHA_BAND[0] or lo >= ALPHA_BAND[1]:
        return FAIL
    in_band = ALPHA_BAND[0] <= ratio < ALPHA_BAND[1]
    inside = ALPHA_BAND[0] <= lo and hi < ALPHA_BAND[1]
    return PASS if (in_band and inside) else UNKNOWN


@dataclass(frozen=True)
class RunReading:
    """One run's C1 inputs as a report.json stores them, with what the page
    prints beside them. `run_id`, `seed`, `utc` are None for a report written
    before DESIGN DECISION 14 (the session-4 pair)."""
    ratio: float
    interval: tuple[float, float]
    path: str | None = None
    run_id: str | None = None
    seed: int | None = None
    utc: str | None = None
    hostname: str | None = None
    exit_code: int | None = None
    slopes: dict = field(default_factory=dict)
    excluded: dict = field(default_factory=dict)
    git_dirty: bool | None = None

    @property
    def name(self) -> str:
        if self.run_id:
            return self.run_id
        if self.path:
            return Path(self.path).parent.name or self.path
        return "this run"

    @property
    def stamp(self) -> str | None:
        return f"{self.utc}@{self.hostname}" if self.utc and self.hostname else None

    @property
    def position(self) -> float:
        """Where the point sits in its own interval, 0 at the low end."""
        lo, hi = self.interval
        return (self.ratio - lo) / (hi - lo) if hi > lo else math.nan


@dataclass(frozen=True)
class CrossRun:
    """Several runs of one design read together. Built on `c1_verdict` so the
    band rule is written once: PASS iff every run's point PASSes against the
    ENVELOPE of all intervals; FAIL iff the envelope misses the band (which
    every point then agrees on); otherwise UNKNOWN. With one reading it is
    exactly `c1_verdict(ratio, interval)`."""
    readings: tuple[RunReading, ...]

    @property
    def points(self) -> list[float]:
        return [r.ratio for r in self.readings]

    @property
    def envelope(self) -> tuple[float, float]:
        return (min(r.interval[0] for r in self.readings),
                max(r.interval[1] for r in self.readings))

    @property
    def spread(self) -> float:
        return max(self.points) - min(self.points)

    @property
    def relative_spread(self) -> float:
        med = statistics.median(self.points)
        return self.spread / med if med else math.nan

    @property
    def sd(self) -> float | None:
        return statistics.stdev(self.points) if len(self.points) >= 3 else None

    def disjoint_pairs(self) -> list[tuple[int, int]]:
        out = []
        for i, a in enumerate(self.readings):
            for j in range(i + 1, len(self.readings)):
                b = self.readings[j]
                if a.interval[1] < b.interval[0] or b.interval[1] < a.interval[0]:
                    out.append((i, j))
        return out

    @property
    def verdict(self) -> str:
        got = {c1_verdict(p, self.envelope) for p in self.points}
        if got == {PASS}:
            return PASS
        if FAIL in got:
            return FAIL
        return UNKNOWN

    def lines(self) -> list[str]:
        n = len(self.readings)
        lo, hi = self.envelope
        out = [f"REPLICATES: {n} run(s) of this design read together; C1 is "
               "scored on the ENVELOPE of their intervals"]
        for k, r in enumerate(self.readings):
            word = (exit_codes.CODE_NAMES[r.exit_code]
                    if r.exit_code is not None else "unscored")
            out.append(
                f"  run {k}: {r.name}; seed "
                + (str(r.seed) if r.seed is not None
                   else "unrecorded (pre-DD14 report)")
                + f"; {r.utc or 'utc unrecorded'}; ratio {r.ratio:.4f} "
                f"[{r.interval[0]:.4f}, {r.interval[1]:.4f}], the point at "
                f"{r.position:.2f} of its own interval; own exit {word}")
            if r.slopes:
                out.append(
                    "         slopes ms per M-tile: "
                    + ", ".join(f"{arm} {v:.4f}" for arm, v in r.slopes.items())
                    + ("; drift-excluded cells: "
                       + ", ".join(f"{arm} {c}" for arm, c in r.excluded.items())
                       if r.excluded else "")
                    + (f"; git_dirty {r.git_dirty}" if r.git_dirty is not None
                       else ""))
        widths = [r.interval[1] - r.interval[0] for r in self.readings]
        med_w = statistics.median(widths)
        out.append(
            f"  spread of the points {self.spread:.4f} "
            f"({self.relative_spread:.2%} of the median), envelope "
            f"[{lo:.4f}, {hi:.4f}]"
            + (f", sd {self.sd:.4f} over {n} points" if self.sd is not None
               else " (an sd needs three points)"))
        out.append(
            f"  the spread is {self.spread / med_w:.1f}x the median within-run "
            f"interval width ({med_w:.4f})" if med_w > 0 else
            "  the within-run intervals have no width")
        pairs = self.disjoint_pairs()
        out.append("  within-run intervals that do not overlap: "
                   + (", ".join(f"runs {i} and {j}" for i, j in pairs)
                      if pairs else "none"))
        out.append("  the within-run interval is a bootstrap over repeats and "
                   "does not cover the run-to-run spread; a point at an edge of "
                   "its own interval is that bootstrap flagging itself as skewed")
        out.append(f"  joint C1 over the envelope: {self.verdict} (PASS needs "
                   "every point in ALPHA_BAND and the whole envelope inside it; "
                   "FAIL needs the envelope to miss the band)")
        return out

    def as_dict(self) -> dict:
        return {"n": len(self.readings), "points": self.points,
                "spread": self.spread, "relative_spread": self.relative_spread,
                "sd": self.sd, "envelope": list(self.envelope),
                "disjoint_pairs": [list(p) for p in self.disjoint_pairs()],
                "verdict": self.verdict,
                "runs": [asdict(r) for r in self.readings]}


def cross_run(readings) -> CrossRun:
    return CrossRun(tuple(readings))


def run_reading(payload: dict, path: Path | None) -> RunReading:
    """A stored report's C1 inputs. Refuses a payload that formed no ratio or
    no interval: it is not a replicate READING, whatever else it recorded."""
    ratio = payload.get("ratio")
    iv = payload.get("ratio_interval") or [None, None]
    if ratio is None or iv[0] is None or iv[1] is None:
        raise PrivateWeightRefusal(
            f"{path}: formed no ratio or no interval, so it is not a replicate "
            "reading")
    prov = payload.get("provenance") or {}
    ladders = payload.get("ladders") or {}
    gates = payload.get("gates") or []
    code = (exit_codes.classify((g["kind"], g["tag"], g["verdict"]) for g in gates)
            if gates else None)
    return RunReading(
        float(ratio), (float(iv[0]), float(iv[1])),
        path=(str(path) if path is not None else None),
        run_id=payload.get("run_id"), seed=payload.get("seed"),
        utc=prov.get("utc"), hostname=prov.get("hostname"), exit_code=code,
        slopes={arm: lad["slope_ms"] for arm, lad in ladders.items()
                if lad.get("slope_ms") is not None},
        excluded={arm: lad.get("excluded_drifted") for arm, lad in ladders.items()},
        git_dirty=prov.get("git_dirty"))


def load_replicates(paths, *, design: dict, card_known: bool,
                    this: RunReading | None = None, this_run_id: str = ""
                    ) -> list[RunReading]:
    """The replicates named on the command line, or a refusal that names the
    path and what differs. Refused BEFORE the card is touched: a file that is
    missing or not JSON, another experiment's report, a planted (--self-test)
    report, a design that differs in any of `DESIGN_KEYS` (card only when this
    run has one), a report that formed no ratio, and a duplicate of another
    replicate or of this run (same file, same run id, or same provenance
    stamp). A replicate whose own gates classify to INVALID or CLAIM_FAIL is
    ADMITTED with its exit word printed: its spread is the information, its
    ratio is not quotable on its own."""
    seen_paths = {Path(this.path).resolve()} if this and this.path else set()
    seen_ids = {x for x in (this.run_id if this else None, this_run_id) if x}
    seen_stamps = {this.stamp} if this and this.stamp else set()
    out: list[RunReading] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            p = p / "report.json"
        if not p.is_file():
            raise PrivateWeightRefusal(f"--replicate-of {p}: no such report")
        try:
            payload = json.loads(p.read_text())
        except ValueError as exc:
            raise PrivateWeightRefusal(f"--replicate-of {p}: not JSON ({exc})") from None
        if not isinstance(payload, dict):
            raise PrivateWeightRefusal(f"--replicate-of {p}: not a report")
        if payload.get("experiment") != "private_weight_reference":
            raise PrivateWeightRefusal(
                f"--replicate-of {p}: a {payload.get('experiment')!r} report, "
                "not this arm's")
        if payload.get("synthetic"):
            raise PrivateWeightRefusal(
                f"--replicate-of {p}: a planted (--self-test) report; a "
                "replicate is a measured run")
        differ = [k for k in DESIGN_KEYS
                  if (k != "card" or card_known)
                  and payload.get(k, DESIGN_KEY_DEFAULTS.get(k)) != design.get(k)]
        if differ:
            raise PrivateWeightRefusal(
                f"--replicate-of {p}: not a replicate of this design; it "
                f"differs in {', '.join(differ)}: "
                + "; ".join(f"{k} {payload.get(k)!r} against {design.get(k)!r}"
                            for k in differ))
        r = run_reading(payload, p)
        resolved = p.resolve()
        if resolved in seen_paths:
            raise PrivateWeightRefusal(
                f"--replicate-of {p} is named twice, or is this run's own report")
        if r.run_id and r.run_id in seen_ids:
            raise PrivateWeightRefusal(
                f"--replicate-of {p}: run id {r.run_id} is already read (twice, "
                "or it is this run)")
        if r.stamp and r.stamp in seen_stamps:
            raise PrivateWeightRefusal(
                f"--replicate-of {p}: the same provenance stamp ({r.stamp}) as "
                "a report already read; one run, one reading")
        seen_paths.add(resolved)
        if r.run_id:
            seen_ids.add(r.run_id)
        if r.stamp:
            seen_stamps.add(r.stamp)
        out.append(r)
    return out


def replicate_plan_lines(replicates) -> list[str]:
    """The plan page's replicate rows: the second call site beside the report."""
    if not replicates:
        return ["replicates  (none: this run is scored ALONE; its interval is "
                "over repeats within one run and is not the run-to-run spread; "
                "a second --seed run read back through --replicate-of is what "
                "puts one on this page)"]
    out = [f"replicates  {len(replicates)} earlier run(s) of this design; C1 "
           "will be scored on the envelope of every interval"]
    for r in replicates:
        word = (exit_codes.CODE_NAMES[r.exit_code]
                if r.exit_code is not None else "unscored")
        out.append(f"            {r.path}: {r.name}, seed "
                   + (str(r.seed) if r.seed is not None else "unrecorded")
                   + f", {r.utc or 'utc unrecorded'}, ratio {r.ratio:.4f} "
                   f"[{r.interval[0]:.4f}, {r.interval[1]:.4f}], own exit {word}")
    return out


def gate_c1_ratio(ratio: float, interval: tuple[float, float], draws: int,
                  *, corrected: float | None,
                  clock: ClockCorrection | None = None,
                  cross: CrossRun | None = None) -> Gate:
    """The measurement: `slope(SHARED) / slope(PRIVATE)`, against the refit.

    THE PRE-REGISTERED CLAIM is the study's own refit band, `ALPHA_BAND`
    (0.529-0.588, 90%). `c1_verdict` scores it from the point and the interval
    together; with replicates (`cross`) the same rule is applied to every
    run's point against the ENVELOPE of the runs' intervals, and the page
    keeps what this run alone said beside it (DESIGN DECISION 14). Both
    registered alternatives -- a ratio near 1.0, meaning the
    whole set is re-read, and a ratio near 0.3, meaning the per-M-tile cost is
    not traffic -- FAIL it, and that is the point: a CLAIM that does not pass
    is a result, and this is the arm where either of those results is worth
    more than a pass.

    SCORED RAW. Both slopes carry the same activation and compute terms and
    dividing them subtracts nothing, so the raw ratio owes nothing to a model.
    Two corrected values are printed beside it, each under the model it
    assumes, and both are scored by nothing, because being model-free is the
    entire reason this ladder exists: the activation-corrected ratio, and the
    CLOCK-corrected ratio when an elasticity was given (`clock`).
    """
    lo, hi = interval
    name, meaning = outcome_for(ratio)
    alone = c1_verdict(ratio, interval)
    verdict = cross.verdict if cross is not None else alone
    detail = [
        f"ratio = slope(shared) / slope(private) = {ratio:.4f}",
        f"{INTERVAL_PCT:.0f}% percentile bootstrap interval "
        f"[{lo:.4f}, {hi:.4f}] over {draws} draws that produced a ratio",
        f"THE WORLD THIS LANDS IN: {name}",
        f"  {meaning}",
        "the registered partition, in full: "
        + "; ".join(f"{n} [{a:.3f}, {b:.3f})" for n, a, b, _ in OUTCOMES),
    ]
    if corrected is not None:
        detail.append(
            f"activation-corrected ratio {corrected:.4f}, printed only: it "
            "subtracts a MODELLED activation slope from both terms, and the "
            "raw number above is the one this arm was built to produce")
    if clock is not None:
        detail += clock.lines()
    detail.append(
        "NO BANDWIDTH AND NO INTERCEPT ENTER THIS NUMBER. It is one measured "
        "slope over another, taken minutes apart on one card at one tile in "
        "one kernel.")
    if cross is not None:
        detail.append(f"this run alone would read {alone}; the verdict above "
                      "is the JOINT one over the runs below")
        detail += cross.lines()
    else:
        detail.append(
            "scored on this run's within-run interval alone: no replicate was "
            "named (--replicate-of), and a bootstrap over repeats is not the "
            "run-to-run spread (DESIGN DECISION 14)")
    if verdict == UNKNOWN:
        detail.append(
            "UNKNOWN, NOT A RESULT: the interval touches ALPHA_BAND but does "
            "not sit inside it, or the point is outside it, so the claim is "
            "unresolved at this precision"
            + (" (over the ENVELOPE of every run's interval)"
               if cross is not None else ""))
    return Gate("C1", CLAIM,
                "the re-read fraction, measured against a no-reuse reference, "
                "is the study's refit alpha",
                verdict,
                f"{ratio:.4f} [{lo:.4f}, {hi:.4f}], {name}"
                + (f"; over {len(cross.readings)} runs: spread "
                   f"{cross.spread:.4f}, envelope [{cross.envelope[0]:.4f}, "
                   f"{cross.envelope[1]:.4f}]" if cross is not None else ""),
                f"PASS: point and whole interval inside ALPHA_BAND "
                f"[{ALPHA_BAND[0]}, {ALPHA_BAND[1]}); FAIL: interval misses "
                "ALPHA_BAND; otherwise UNKNOWN"
                + ("; with replicates: every run's point in the band and the "
                   "ENVELOPE of their intervals inside it"
                   if cross is not None else ""),
                "the refit alpha is not the traffic fraction it is quoted as, "
                "and the page names which of the registered worlds it is "
                "instead",
                detail)


def gate_c2_achieved_rate(private: Ladder, *, weight_bytes: int,
                          bandwidth_gbps: float, bandwidth_source: str,
                          stream_ms: float) -> Gate:
    """The private slope, read as a delivered bandwidth, against the ceiling.

    `weight_bytes / slope(PRIVATE)` is the rate at which the grouped GEMM
    delivered its weight read, measured INSIDE the kernel under test with no
    triad benchmark in it -- provided V2 holds and the private arm really did
    read a whole fresh copy per M-tile. A kernel cannot deliver more than its
    card's own measured ceiling, so this is a RELATION and not a comparison
    with a literal: a violation says either that the copies were not all read,
    which V2 answers, or that the calibration is not a ceiling.

    The same number inverted is `w(PRIVATE) = slope / stream_ms`, which SHOULD
    be 1.0 when the arm re-reads the set once per tile at the calibrated rate.
    Both are printed; the gate is on the relation.

    THE GATE IS ONE-SIDED, AND THE OPEN SIDE IS THE ONE THAT HAPPENS. `achieved`
    also carries the activation term, so it UNDERSTATES the true delivered rate,
    and the only world that FAILS this gate is a kernel beating the card's triad
    by more than the tolerance -- which no grouped GEMM does. A private arm
    delivering 80%, 60% or 40% of the calibrated rate scores the same word as
    one hitting it exactly, with w(private) at 1.25, 1.67 and 2.50 printed
    beside it and read by nothing. So a PASS here is "the denominator is not
    impossible" and NOT "the denominator is 1.0". Making it two-sided -- w
    within a registered band of 1.0, with the activation share (which the plan
    page prices at under 1% of a stream at this tile) taken out first -- would
    make it a claim about the study's denominator rather than a sanity check on
    it, and that is a change to a REGISTERED claim and so the owner's to make.
    """
    if private.slope_ms <= 0:
        return Gate("C2", CLAIM,
                    "the private arm's delivered weight-read rate is at or "
                    "under the card's own ceiling",
                    UNKNOWN, "no positive private slope",
                    f"<= the calibrated rate x {1 + ACHIEVED_RATE_TOLERANCE:.2f}",
                    "the denominator of the ratio is not a weight stream", [])
    achieved = weight_bytes / (private.slope_ms * 1e-3) / 1e9
    w_private = private.slope_ms / stream_ms
    ceiling = bandwidth_gbps * (1.0 + ACHIEVED_RATE_TOLERANCE)
    detail = [
        f"slope(private) {private.slope_ms:.6f} ms per M-tile moves "
        f"{weight_bytes / 1e9:.4f} GB, so it delivered "
        f"{achieved:.1f} GB/s",
        f"the card's calibrated rate is {bandwidth_gbps:.1f} GB/s "
        f"({bandwidth_source or 'source NOT STATED'}); the gate allows "
        f"{ACHIEVED_RATE_TOLERANCE:.0%} over it, i.e. {ceiling:.1f} GB/s",
        f"w(private) = {w_private:.4f} weight-streams per M-tile at that rate; "
        "1.0000 is what a full fresh read per tile at the calibrated rate "
        "would give, and the excess over 1 is the shortfall of the achieved "
        "rate against the calibrated one",
        "THIS IS THE STUDY'S OWN DENOMINATOR, MEASURED. Every published w "
        "divides a slope by a stream time computed at an ASSUMED rate; this "
        "row says what that assumption was worth on this card in this kernel.",
    ]
    return Gate("C2", CLAIM,
                "the private arm's delivered weight-read rate is at or under "
                "the card's own ceiling",
                PASS if achieved <= ceiling else FAIL,
                f"{achieved:.1f} GB/s, w={w_private:.4f}",
                f"<= {ceiling:.1f} GB/s",
                "either the private arm did not read every copy (V2 answers "
                "that) or the calibrated bandwidth this study divides every "
                "slope by is not a ceiling",
                detail)


# --------------------------------------------------------------------------
# The plan and the predictions, printed BEFORE anything runs.
# --------------------------------------------------------------------------

def prediction_lines(cfg, *, block_m: int, treads: list[int], alpha: float,
                     bandwidth_gbps: float, bw_source: str, dtype: str,
                     stream_ms: float, ridge: float, ridge_source: str,
                     copies_declared: int,
                     census: PathCensus | None = None) -> list[str]:
    weight = WEIGHTS.routed_expert_weight_bytes(cfg, dtype)
    act_per_tile = (cfg.num_experts * block_m
                    * SWEEP.activation_bytes_per_row(cfg))
    act_share = act_per_tile / weight
    out = ["", "PREDICTIONS, registered before the run and printed before any "
                "measurement",
           "  the quantity: ratio = slope(shared) / slope(private), one "
           "measured slope over another",
           f"  the study's refit alpha {alpha:.3f}, band "
           f"{ALPHA_BAND[0]}-{ALPHA_BAND[1]} (90%), against the retracted "
           f"{RETRACTED_ALPHA}",
           f"  ridge       {ridge:.2f} Op/B, {ridge_source or 'source not stated'}",
           f"  bandwidth   {bandwidth_gbps:.1f} GB/s, "
           f"{bw_source or 'source not stated'}",
           f"  one complete stream of the routed expert weight set: "
           f"{weight / 1e9:.4f} GB in {stream_ms:.4f} ms at that rate",
           "",
           "  THE REGISTERED PARTITION OF THE RATIO, covering [0, inf) with no "
           "gap and no overlap:"]
    for name, lo, hi, meaning in OUTCOMES:
        top = "inf" if hi == math.inf else f"{hi:.3f}"
        out.append(f"    [{lo:.3f}, {top:>5s})  {name}")
        out.append(f"                     {meaning}")
    gap = partition_is_total()
    out.append("  the partition is total and disjoint: "
               + ("checked" if not gap else f"BROKEN -- {gap}"))
    out.append("")
    out.append("  WHAT EACH ARM PREDICTS, under the study's own traffic model, "
               "so the two worlds are on the page before the run:")
    out.append("    native   the study's call over the same bytes: the SAME "
               "slope as shared, a different intercept, and the slope "
               "difference is V5")
    out.append(f"    shared   slope ~ alpha x one stream + activations "
               f"= {alpha:.3f} + {act_share:.4f} streams per M-tile")
    out.append(f"    private  slope ~ one stream + activations = 1.0000 + "
               f"{act_share:.4f} streams per M-tile, BY CONSTRUCTION")
    out.append(f"    so the ratio is predicted at "
               f"{(alpha + act_share) / (1.0 + act_share):.4f} in the refit "
               f"world and {(1.0 + act_share) / (1.0 + act_share):.4f} in the "
               "no-reuse world")
    out.append(f"    and at {(RETRACTED_ALPHA + act_share) / (1.0 + act_share):.4f} "
               f"in the retracted alpha={RETRACTED_ALPHA} world, which the "
               "partition names ISSUE-AND-LATENCY")
    out.append("")
    out.append("  THE GATES, and the thresholds they are scored at:")
    out.append(f"    V0 all cells, >= {max(MIN_TREADS, 2)} usable treads and "
               f">= {MIN_REPEATS} repeats per arm")
    out.append("    V1 one tread set, one BLOCK_M, one token count per tread")
    out.append(f"    V2 all {len(PROOF_PARTS)} parts of the buffer proof: "
               + ", ".join(name for name, _ in PROOF_PARTS))
    out.append(f"    V3 weight allocation within {WEIGHT_ALLOC_TOLERANCE:.0%} "
               "of prediction; high-water mark under the plan's ceiling")
    out.append(f"    V4 every fitted tread of shared and private under "
               f"{COMPUTE_BOUND_FRACTION:.0%} of the fixed roof")
    out.append(f"    V5 the far edge of b's {INTERVAL_PCT:.0f}% band, b the "
               "step-aware native - shared per-tile cost, < "
               f"{MACHINERY_BOUND:.0%} of slope(private): the wider "
               "declaration's own per-M-tile cost, which ties the matched "
               "ratio to the study's call")
    out.append(f"    V6 shared and private within {IDENTITY_SPREAD:.1%} at n=1, "
               "where they are the same call")
    out.append(f"    V7 shared and private under-load clocks within "
               f"{CLOCK_PARITY:.0%} at every tread; at FULL duty a card that "
               "cannot lock its clock fails this by construction whenever the "
               "arms draw different power, and below it this holds only at a "
               "duty low enough that the clock stops following power (session "
               f"4's clock arm: flat at {FLAT_DUTY}, still tracking power at "
               "0.5); --clock-elasticity PRINTS a corrected ratio beside the "
               "raw one, scored by nothing")
    out.append(f"    V8 the probed alignment steps in SHARED's and PRIVATE's id "
               f"sets are worth <= {ALIGN_STEP_RATIO_BUDGET} of the ratio "
               "together; FAIL needs it over budget AND resolved, and a FAIL "
               "alone skips the sweep; the probe is timed under a CUDA graph "
               f"({PROBE_CALLS_PER_REPLAY} calls per replay) so its cells are "
               "GPU time and NATIVE's switch beside them confirms or bounds, "
               "never withholds; an EAGER fallback that comes back host-bound "
               "is scored only if it resolves NATIVE's switch at the census "
               "tread")
    out.append("    C1 PASS: point and whole interval inside ALPHA_BAND; "
               "FAIL: interval misses ALPHA_BAND; otherwise UNKNOWN")
    out.append(f"    C2 the delivered weight-read rate is at or under the "
               f"card's own, +{ACHIEVED_RATE_TOLERANCE:.0%}")
    out.append("")
    out.append(f"  THE LAYOUT, REGISTERED: copy c of expert e is slot "
               f"e x {copies_declared} + c, and shared and private declare all "
               f"E x {copies_declared} = "
               f"{expert_space(cfg.num_experts, copies_declared)} slots at "
               "every tread, so the two share the sorted-id buffer, the launch "
               "grid, the dead launches and the tile order, and differ ONLY in "
               "the addresses their tiles read. Of those "
               f"{copies_declared} copies the deepest tread READS "
               f"{treads[-1]}; the rest are padding, filled and never read, "
               "and the buffer proof zeroes each of them and requires neither "
               "arm's output to move.")
    out.append("  THE CAVEAT, REGISTERED: those addresses are also the TLB "
               "footprint and the DRAM page locality, so slope(private) is a "
               "BOUNDED PROXY for the no-reuse case and not the no-reuse case "
               "itself. V5 measures the declaration at depth, V6 this arm's "
               "own floor at n=1, V7 the clock. None is an argument; all are "
               "numbers on this page after the run.")
    out.append("  AND WHAT NONE BOUNDS: a cost paid by READING the copies "
               "that vanishes at one copy, which is the shape of a TLB or "
               "DRAM-page locality cost. NATIVE does not read them and n=1 has "
               "one. The only trace such a cost leaves here is the private "
               "ladder's own mean relative residual, printed in the ladder "
               "table and SCORED BY NOTHING.")
    native_switch = census.switch_tread(NATIVE) if census is not None else None
    out.append("  THE ALIGNMENT KERNEL, SEPARATED THREE WAYS: vLLM switches "
               "moe_align_block_size kernel at the cited id and expert bounds, "
               + (f"and for this ladder the id bound falls inside it (native "
                  f"switches at tread {native_switch}). The ratio "
                  if native_switch else
                  "and for THIS ladder it does not: the id count stays on one "
                  "side of the bound at every tread, so NATIVE switches "
                  "nowhere in it. The ratio ")
               + "arms are declared past the expert bound so both take one "
               "kernel throughout (the census above, refused otherwise); the "
               "probe times the alignment op alone along the ladder once per "
               "ARM -- NATIVE's declaration, and SHARED's and PRIVATE's id "
               "sets at the ratio arms' -- and V8 scores both ratio series; "
               "and NATIVE, "
               "which keeps the study's declaration and its switch, has its "
               "difference from SHARED fitted with a step term so V5 reads the "
               "declaration's per-tile cost with the step out and the step is "
               "printed with an interval.")
    if census is not None and census.switch_tread(NATIVE) is None:
        # REGISTERED BEFORE A POD IS RENTED, because it decides what V8 can
        # say. The positive control is NATIVE's own kernel switch, and with
        # none in this ladder it is UNAVAILABLE. Under the graph that is
        # informational: the ratio series are GPU time by construction and
        # V8 rests on them. It binds only on the EAGER fallback, whose cells
        # an H200 makes host-bound (the alignment call is a few microseconds
        # against tens of host enqueue): V8 can then reach PASS only on a
        # probe the instrument judged and cleared; otherwise UNKNOWN, which
        # latches the page INVALID after the ladder has been paid for -- the
        # skip is FAIL-only.
        out.append("  AND THIS LADDER GIVES V8 NO POSITIVE CONTROL: its id "
                   f"count runs {ids_for_tread(cfg, treads[0], block_m)}.."
                   f"{ids_for_tread(cfg, treads[-1], block_m)} and stays on "
                   f"one side of the {ALIGN_SMALL_BATCH_MAX_IDS}-id bound, so "
                   "NATIVE switches kernel nowhere in it and there is no "
                   "switch to read beside the ratio series. Under the CUDA "
                   "graph that is informational and V8 rests on the GPU-timed "
                   "ratio series; on the EAGER fallback, which an H200 makes "
                   "host-bound, V8 can only read UNKNOWN -- the page latches "
                   "INVALID and the ladder is still paid for, because the "
                   "sweep is skipped on a FAIL alone. A tile whose ladder "
                   "CROSSES that bound supplies the control; the booked tile "
                   "does.")
    slot_elems = max(2 * cfg.intermediate_size * cfg.hidden_size,
                     cfg.hidden_size * cfg.intermediate_size)
    top_slot = expert_space(cfg.num_experts, copies_declared) - 1
    out.append("  AND TWO SMALL ASYMMETRIES, REGISTERED AND NOT REMOVED: on "
               "the scan kernel, `count_and_sort_expert_tokens` does one "
               "atomic add per id into a per-expert counter, so SHARED's "
               f"{cfg.num_experts} counters take {block_m} n increments each "
               f"while PRIVATE's {cfg.num_experts} x n take {block_m} each; a "
               "per-tread term in the numerator alone. The probe times the "
               "alignment op on PRIVATE's id set as well as SHARED's, and V8 "
               "prints slope(private ids) - slope(shared ids) in us per tread "
               "and in the ratio's units: MEASURED, printed, not scored. And "
               "the wider "
               f"declaration puts the top expert slot {top_slot} at "
               f"{top_slot * slot_elems / 2**30:.2f} Gi ELEMENTS from the "
               "weight base in w1, past the 2^31 an int32 offset holds, which "
               "the kernel addresses through an int64 cast of `off_experts`; "
               "NATIVE's strided view multiplies its own index by the same "
               "stride and depends on the cast too. A build without it reads "
               "the wrong bytes, and V2's same_layer and kernel_read parts are "
               "what catch that.")
    out.append("  AND THE ASSUMPTION THE RATIO KEEPS: no rate enters the "
               "number, but the ratio is alpha as a traffic fraction only if "
               "both arms deliver the same bytes per second. The shared arm's "
               "non-re-read bytes come from L2, which is not free.")
    out.append("  NOT A READOUT: the LEVEL of any ladder. Only the slopes "
               "are compared; native's intercept differs from shared's by the "
               "declaration's dead launches, which are a constant per tread.")
    out.append("  NOT MEASURED HERE: alpha_a, the activation-side miss "
               "fraction. Both ladders carry the same activation term and the "
               "ratio subtracts nothing.")
    return out


def plan_lines(cfg, args, *, block_m: int, treads: list[int], b: int,
               bandwidth_gbps: float, bw_source: str, out_dir: Path,
               pinned: dict, run_id: str, resources, card: str, git_note: str,
               mem: MemoryPlan, tokens: dict[int, int], ridge: float,
               alpha: float, copies_declared: int, declared_reason: str,
               census: PathCensus, replicates: tuple = ()) -> list[str]:
    deepest = treads[-1]
    return [
        f"experiment  private_weight_reference / {run_id}",
        f"model       {args.model} E={cfg.num_experts} k={cfg.top_k}  "
        f"{args.dtype} ({b} bytes)",
        f"tile        BLOCK_M={block_m}, one tile per run and in the run id",
        f"pinned      {pinned}",
        "arms        " + "; ".join(f"{a}: {ARM_MEANING[a]}" for a in ARMS),
        f"ladder      {len(treads)} treads n={treads[0]}..{treads[-1]}, "
        f"exactly-full tile stacks only (r = n x {block_m})",
        "            r per tread: "
        + ", ".join(f"n={n}:r={n * block_m}:T={tokens[n]}" for n in treads),
        f"            {SWEEP.rows_step(cfg)} token step, rows quantum "
        f"{SWEEP.rows_quantum(cfg)}",
        f"repeats     {args.repeats} of the whole ladder, repeats OUTER, arms "
        f"rotated within a tread so no arm is first in every triple, and the "
        f"tread order REVERSED on odd repeats so no tread is always first",
        f"cells       {len(treads)} treads x {len(ARMS)} arms x "
        f"{args.repeats} repeats = {len(treads) * len(ARMS) * args.repeats}",
        f"duty        {args.duty:.2f}"
        + (f": every cell timed as bursts of ~{DUTY_BURST_MS:.0f} ms of kernel "
           f"time with idle gaps of {DUTY_BURST_MS * (1 / args.duty - 1):.0f} ms, "
           "so the arms' AVERAGE board power drops, and with it the clock "
           "split the cap forces at full duty (session 4 at full duty: the "
           "arm reading less boosted 2-18%). Whether this duty is low enough "
           f"is measured, not assumed ({FLAT_DUTY_EVIDENCE}); V7 checks it "
           "here and a FAIL names a lower duty; "
           f"wall clock over the ladder ~{1 / args.duty:.1f}x the kernel time"
           if args.duty < 1.0 else
           ": the queue kept full, the driver's instrument; on a power-capped "
           "card the two ratio arms then draw different power and V7 decides "
           f"whether their clocks agreed (--duty {FLAT_DUTY} is the setting "
           "session 4's clock arm read flat: DESIGN DECISION 15)"),
        f"experts     native declares E={cfg.num_experts}; shared and private "
        f"declare E x n_decl = {expert_space(cfg.num_experts, copies_declared)} "
        "at EVERY tread, copy c of expert e at slot e x n_decl + c",
        f"            n_decl = {copies_declared} against n_max = {deepest} "
        f"read: {declared_reason}",
        *census.lines(),
        "session     " + (args.session_tag or "(none: a bare run, keyed on its "
                                              "arguments and card alone)"),
        *replicate_plan_lines(replicates),
        f"bandwidth   {bandwidth_gbps:.1f} GB/s, {bw_source}",
        f"card        {card}"
        + ("   (no CUDA device: this is a plan or a replay, not a measurement)"
           if card == NO_CARD_SLUG else ""),
        f"WRITES TO   {out_dir}",
        "            cells.csv (appended per cell), report.txt, report.json, "
        "triton-cache/",
        f"git         {git_note}",
        "resources   one CTA's bill under the pinned constants, so a setting "
        "that cannot physically run is refused here and not diagnosed from its "
        "timing afterwards:",
        *[resources[block_m].render()],
        *mem.lines(),
        *depth_lines(cfg, block_m=block_m, treads=treads, ridge=ridge,
                     bandwidth_gbps=bandwidth_gbps, b=b, alpha=alpha),
    ]


def estimated_seconds(cfg, *, treads: list[int], block_m: int, repeats: int,
                      alpha: float, ridge: float, bandwidth_gbps: float, b: int,
                      warmup_ms: float, trials: int, cell_budget_ms: float,
                      probe_repeats: int = PROBE_REPEATS) -> float:
    """Kernel seconds at the model's own timings. EXCLUDING compiles and
    allocation, which the plan says in those words and the driver's cost table
    reads the clock off.

    The private arm is priced at `alpha = 1`, which is what it is by
    construction, and native at the shared arm's traffic, which is what it
    reads. Pricing all three at the study's alpha would under-book the one
    arm whose whole design is to move more bytes.
    """
    total = 0.0
    for arm in ARMS:
        a = 1.0 if arm == PRIVATE else alpha
        for n in treads:
            rows = n * block_m
            ms = SWEEP.model_ms(cfg, rows, block_m, alpha=a, ridge=ridge,
                                bandwidth_gbps=bandwidth_gbps, b=b)
            iters = SWEEP.planned_iters(ms, cell_budget_ms)
            total += repeats * (warmup_ms + trials * iters * ms) * 1e-3
    # The alignment probe: one series per arm (native's declaration, and the
    # shared and private id sets at the ratio arms'), every tread,
    # PROBE_REPEATS times, at its own budget.
    total += probe_seconds(treads, PROBE_LABELS, probe_repeats)
    return total


# --------------------------------------------------------------------------
# The analysis. Pure: samples in, report out. No GPU, no I/O, so `--self-test`
# and the test suite exercise exactly what the pod run prints.
# --------------------------------------------------------------------------

@dataclass
class Report:
    lines: list[str]
    gates: list[Gate]
    payload: dict

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def tread_rows(samples, cfg, *, block_m: int, roof_tflops: float,
               reference_mhz: float | None, reference_grade: str) -> list[dict]:
    """One row per (arm, tread): the median time, the achieved rate, and BOTH
    roof fractions, the fixed one and the one at the cell's own clock."""
    rows = []
    for arm in ARMS:
        by: dict[int, list] = {}
        for s in samples:
            if s.usable and s.arm == arm:
                by.setdefault(s.tiles, []).append(s)
        for n, group in sorted(by.items()):
            ms = statistics.median([s.ms_p50 for s in group])
            rows_total = cfg.num_experts * n * block_m
            tflops = (SWEEP.useful_flops(cfg, rows_total) / (ms * 1e-3)) / 1e12
            clocks = [s.sm_clock_load_mhz for s in group
                      if s.sm_clock_load_mhz]
            load = statistics.median(clocks) if clocks else None
            own_roof, note = roofline.cell_clock_roof(
                roof_tflops, load, reference_mhz, reference_grade)
            rows.append({
                "arm": arm, "tiles": n, "rows_per_expert": n * block_m,
                "tokens": group[0].tokens, "copies": group[0].copies,
                "experts_declared": group[0].experts_declared,
                "repeats": len(group), "ms_p50": ms,
                "achieved_tflops": tflops,
                "pct_of_roof": tflops / roof_tflops if roof_tflops else 0.0,
                "sm_clock_load_mhz": load,
                "roof_at_cell_clock_tflops": own_roof,
                "pct_of_roof_at_cell_clock": (tflops / own_roof if own_roof
                                              else 0.0),
                "roof_note": note,
                "level_sides": sorted({s.clock_level_side for s in group
                                       if s.clock_level_side}),
            })
    return rows


def native_switch_source(probe: AlignProbe | None, census: PathCensus
                         ) -> tuple[int | None, str]:
    """`(the tread V5's declaration fit puts NATIVE's step at, why)`: the
    probe when it resolved a step there in GPU time, else the cited
    hypothesis. ONE HOME, read by `analyse` and testable without a page.

    The tread logic is `native_step_is_admissible`'s; what this adds is the
    sentence for a GRAPH-timed probe that resolved no step: that is a BOUND
    on the switch (under the fit's threshold per call), not an unconfirmed
    hypothesis, because no host cost sits in a graph replay.
    """
    native_switch = census.switch_tread(NATIVE)
    switch_source = (f"the cited hypothesis (tread {native_switch})"
                     if native_switch else "the cited hypothesis (no switch)")
    if probe is not None and NATIVE in probe.labels():
        try:
            reading = read_probe(probe, NATIVE, census)
            ok, why = native_step_is_admissible(reading, probe)
            if ok:
                native_switch = reading.fit.split_tread
                switch_source = (f"the probe, which resolved native's step at "
                                 f"tread {native_switch} ({why})")
            elif reading.real and reading.fit.split_tread != native_switch:
                # A STEP THE PROBE COULD NOT TIME ON THE GPU DOES NOT GET TO
                # CHOOSE THE TREAD V5 IS FITTED AT. b is what V5 scores at
                # MACHINERY_BOUND, and the tread the step term sits at moves
                # it; a host-timed split would put a step in the host's time
                # into the declaration's own per-M-tile cost.
                switch_source += (
                    f"; the probe put native's step at tread "
                    f"{reading.fit.split_tread} instead, and that reading is "
                    f"REFUSED as the fit's tread because {why}")
            elif reading.real:
                # Resolved AT the census tread but not on the GPU: the tread
                # is the same either way, and the sentence used to say "no
                # step" here.
                switch_source += (
                    f"; the probe resolved native's step at that tread too, "
                    f"but not as the fit's reading, because {why}")
            elif reading.graph_calls:
                switch_source += (
                    "; the probe resolved no step for native IN GPU TIME "
                    f"(threshold {reading.fit.threshold_ms() * 1e3:.2f} us at "
                    f"{reading.fit.dof} dof), so the hypothesis stands as the "
                    "fit's tread and the step it names is under that per call")
            else:
                switch_source += ("; the probe resolved no step for native, so "
                                  "the hypothesis stands unconfirmed")
        except Unmeasurable as exc:
            switch_source += f"; the probe's native series was not fitted ({exc})"
    return native_switch, switch_source


def analyse(samples, cfg, *, block_m: int, treads: list[int], repeats: int,
            alpha: float, dtype: str, b: int, bandwidth_gbps: float,
            bandwidth_source: str, ridge: float, ridge_source: str,
            roof_tflops: float, roof_source: str,
            reference_mhz: float | None, reference_grade: str,
            reference_source: str,
            mem: MemoryPlan, proof: BufferProof,
            weight_delta_bytes: int | None, high_water_bytes: int | None,
            draws: int, seed: int, header: list[str], card: str,
            synthetic: bool, model_name: str, pinned: dict, prov=None,
            probe: AlignProbe | None = None, census: PathCensus | None = None,
            copies_declared: int | None = None,
            clock_elasticity: ClockElasticity | None = None,
            run_id: str = "", session_tag: str = "",
            replicates: tuple = ()) -> Report:
    planned = len(treads) * len(ARMS) * repeats
    if census is None:
        census = path_census(cfg, treads, block_m, {
            arm: declared_experts(arm, cfg.num_experts,
                                  copies_declared or treads[-1])
            for arm in ARMS})
    lines = list(header)
    lines += ["", "=" * 72, "LADDERS", "=" * 72,
              f"{'arm':8s} {'n':>3s} {'r':>6s} {'T':>7s} {'copies':>6s} "
              f"{'E*':>6s} {'reps':>4s} {'ms':>10s} {'TFLOP/s':>9s} "
              f"{'%roof':>7s} {'%own':>7s} {'MHz':>6s}"]
    rows = tread_rows(samples, cfg, block_m=block_m, roof_tflops=roof_tflops,
                      reference_mhz=reference_mhz,
                      reference_grade=reference_grade)
    for r in rows:
        own = (f"{r['pct_of_roof_at_cell_clock']:6.1%}"
               if r["roof_at_cell_clock_tflops"] else "     --")
        mhz = f"{r['sm_clock_load_mhz']:6.0f}" if r["sm_clock_load_mhz"] else "    --"
        lines.append(
            f"{r['arm']:8s} {r['tiles']:3d} {r['rows_per_expert']:6d} "
            f"{r['tokens']:7d} {r['copies']:6d} {r['experts_declared']:6d} "
            f"{r['repeats']:4d} {r['ms_p50']:10.4f} "
            f"{r['achieved_tflops']:9.1f} {r['pct_of_roof']:6.1%} {own} {mhz}")
    if rows and not any(r["roof_at_cell_clock_tflops"] for r in rows):
        lines.append(f"  %own is not scored on any row: {rows[0]['roof_note']}")

    ladders: dict[str, Ladder] = {}
    unmeasurable = ""
    try:
        for arm in ARMS:
            ladders[arm] = ladder_for(samples, arm)
    except Unmeasurable as exc:
        unmeasurable = str(exc)

    lines += ["", "FITS, ms = A + B n over the treads above"]
    for arm in ARMS:
        lad = ladders.get(arm)
        if lad is None:
            lines.append(f"  {arm:8s} NOT FITTED: {unmeasurable}")
            continue
        stream = WEIGHTS.weight_streams_per_tile(
            lad.slope_ms, cfg, dtype, bandwidth_gbps,
            bandwidth_source=bandwidth_source)
        lines.append(
            f"  {arm:8s} A={lad.intercept_ms:9.4f} ms  B={lad.slope_ms:9.6f} "
            f"ms/M-tile  treads={lad.treads}  mean rel err {lad.mean_rel_err:.3%}"
            + (f"  across-repeat spread {lad.spread:.3%}" if lad.spread is not None
               else "  across-repeat spread NOT DETERMINED")
            + (f"  drifting cells excluded {lad.excluded}" if lad.excluded else ""))
        lines.append(f"           w = {stream.render()}")

    ratio = corrected = None
    interval = (math.nan, math.nan)
    got_draws = 0
    if ladders.get(SHARED) and ladders.get(PRIVATE) and ladders[PRIVATE].slope_ms:
        ratio = ladders[SHARED].slope_ms / ladders[PRIVATE].slope_ms
        act_ms = (cfg.num_experts * block_m
                  * SWEEP.activation_bytes_per_row(cfg)
                  / (bandwidth_gbps * 1e9) * 1e3)
        denom = ladders[PRIVATE].slope_ms - act_ms
        if denom > 0:
            corrected = (ladders[SHARED].slope_ms - act_ms) / denom
        try:
            lo, hi, got_draws = ratio_interval(samples, draws, seed)
            interval = (lo, hi)
        except Unmeasurable as exc:
            lines.append(f"  interval NOT FORMED: {exc}")
    clock_correction = None
    if ratio is not None and clock_elasticity is not None:
        # f_ref is cosmetic for the RATIO (it cancels) and named for the
        # CELLS: the calibration's reference clock when one was resolved,
        # else the private arm's own median under load.
        if reference_mhz:
            f_ref, f_ref_source = float(reference_mhz), "the calibration's reference clock"
        else:
            clocks = [s.sm_clock_load_mhz for s in samples
                      if s.arm == PRIVATE and s.sm_clock_load_mhz]
            f_ref = statistics.median(clocks) if clocks else 0.0
            f_ref_source = "the private arm's median under-load clock"
        try:
            clock_correction = clock_corrected_ratio(
                samples, clock_elasticity, f_ref=f_ref,
                f_ref_source=f_ref_source, draws=draws, seed=seed)
        except Unmeasurable as exc:
            lines.append(f"  clock-corrected ratio NOT FORMED: {exc}")

    stream_ms = WEIGHTS.weight_stream_ms(cfg, dtype, bandwidth_gbps)

    # WHERE NATIVE'S SWITCH IS TAKEN FROM: the probe when it resolved a step
    # there, else the cited hypothesis. Said on the page either way.
    native_switch, switch_source = native_switch_source(probe, census)
    decl_fit = None
    decl_bands: tuple = (None, None)
    band_absence = ""
    if ladders.get(NATIVE) and ladders.get(SHARED):
        # TWO REFUSALS, TWO SENTENCES. The fit and its bootstrap shared one
        # try, so a failed BOOTSTRAP was reported as a failed FIT -- "the
        # declaration difference NOT FITTED" printed two lines above the
        # fitted declaration. And since V5 reads UNKNOWN when b has no band,
        # the bootstrap's failure now decides a verdict, which makes saying
        # which one failed the difference between a page a reader can follow
        # and one that contradicts itself. The ratio's own interval has said
        # it this way since it was written: "interval NOT FORMED: ...".
        try:
            decl_fit = declaration_fit(samples, treads, native_switch)
        except Unmeasurable as exc:
            band_absence = f"the declaration difference was not fitted ({exc})"
            lines.append(f"  declaration difference NOT FITTED: {exc}")
        else:
            try:
                b_band, s_band, _n = declaration_interval(
                    samples, treads, native_switch, draws, seed)
                decl_bands = (b_band, s_band)
            except Unmeasurable as exc:
                band_absence = f"b's band was NOT FORMED ({exc})"
                lines.append(f"  b's band NOT FORMED over {draws} draws: {exc}"
                             " -- the declaration IS fitted below; what is "
                             "missing is the band V5 scores its far edge on")
    if decl_fit is not None:
        lines += ["", "DECLARATION, native - shared per tread (no traffic in it): "
                  + ", ".join(f"n={n}:{d:+.4f}" for n, d in decl_fit.points)]

    gates: list[Gate] = [
        gate_v0_non_vacuity(samples, planned=planned, treads=treads,
                            repeats=repeats),
        gate_v1_matched_geometry(samples, block_m=block_m, treads=treads),
        gate_v2_distinct_buffers(proof),
        gate_v3_memory(mem, weight_delta_bytes=weight_delta_bytes,
                       high_water_bytes=high_water_bytes),
        gate_v4_memory_bound(rows, roof_tflops=roof_tflops,
                             roof_source=roof_source),
    ]
    if ladders.get(NATIVE) and ladders.get(SHARED) and ladders.get(PRIVATE):
        gates.append(gate_v5_machinery(ladders[NATIVE], ladders[SHARED],
                                       ladders[PRIVATE], decl_fit,
                                       decl_bands[0], decl_bands[1],
                                       switch_source, band_absence))
    else:
        gates.append(Gate("V5", VALIDITY,
                          "the declaration's own per-M-tile cost is bounded",
                          UNKNOWN, "a ladder was not fitted",
                          MACHINERY_WANT,
                          "the ratio's distance from the study's own call is "
                          "unbounded",
                          [unmeasurable] if unmeasurable else []))
    gates.append(gate_v6_identity(samples, identity_tread=treads[0]))
    gates.append(gate_v7_clock_parity(samples, treads=treads))
    gates.append(gate_v8_alignment(probe, treads=treads, census=census,
                                   weight_stream_ms=stream_ms))
    # THIS RUN IS THE FIRST READING when replicates were named. A run whose
    # interval was not formed contributes no reading; the replicates are still
    # printed together so the page carries their spread, and C1 stays UNKNOWN.
    this_reading = (RunReading(ratio, tuple(interval), run_id=run_id or None,
                               seed=seed,
                               slopes={arm: lad.slope_ms
                                       for arm, lad in ladders.items()},
                               excluded={arm: lad.excluded
                                         for arm, lad in ladders.items()})
                    if ratio is not None and all(math.isfinite(v) for v in interval)
                    else None)
    cross = (cross_run(([this_reading] if this_reading else []) + list(replicates))
             if replicates else None)
    if ratio is None or this_reading is None:
        gates.append(Gate("C1", CLAIM,
                          "the re-read fraction, measured against a no-reuse "
                          "reference, is the study's refit alpha",
                          UNKNOWN,
                          "no ratio was formed" if ratio is None
                          else f"{ratio:.4f}, no interval was formed",
                          "point and whole interval inside ALPHA_BAND "
                          f"[{ALPHA_BAND[0]}, {ALPHA_BAND[1]})",
                          "the refit alpha is not the traffic fraction it is "
                          "quoted as",
                          ([unmeasurable] if unmeasurable else [])
                          + (["this run enters no reading; the replicates "
                              "named are read together below and C1 stays "
                              "UNKNOWN for this run"] + cross.lines()
                             if cross is not None else [])))
    else:
        gates.append(gate_c1_ratio(ratio, interval, got_draws,
                                   corrected=corrected, clock=clock_correction,
                                   cross=cross))
    if ladders.get(PRIVATE):
        gates.append(gate_c2_achieved_rate(
            ladders[PRIVATE],
            weight_bytes=WEIGHTS.routed_expert_weight_bytes(cfg, dtype),
            bandwidth_gbps=bandwidth_gbps, bandwidth_source=bandwidth_source,
            stream_ms=stream_ms))
    else:
        gates.append(Gate("C2", CLAIM,
                          "the private arm's delivered weight-read rate is at "
                          "or under the card's own ceiling",
                          UNKNOWN, "no private ladder", "a relation",
                          "the denominator of the ratio is not a weight stream",
                          []))

    lines += ["", "=" * 72, "GATES", "=" * 72]
    for g in gates:
        lines += g.render()
        lines.append("")
    rc = exit_codes.classify(g.scored() for g in gates)
    lines.append(f"the gates imply {exit_codes.describe(rc)}")

    payload = {
        "experiment": "private_weight_reference",
        "synthetic": synthetic,
        "card": card,
        "model": model_name,
        "dtype": dtype,
        "block_m": block_m,
        "pinned": dict(pinned),
        "treads": list(treads),
        "repeats": repeats,
        "duty": duty_of(samples),
        "alpha_refit": alpha,
        "alpha_band": list(ALPHA_BAND),
        "ridge": ridge,
        "ridge_source": ridge_source,
        "bandwidth_gbps": bandwidth_gbps,
        "bandwidth_source": bandwidth_source,
        "roof_tflops": roof_tflops,
        "roof_source": roof_source,
        "reference_clock_mhz": reference_mhz,
        "reference_clock_grade": reference_grade,
        "reference_clock_source": reference_source,
        "weight_stream_ms": stream_ms,
        "memory_plan": asdict(mem),
        "weight_delta_bytes": weight_delta_bytes,
        "high_water_bytes": high_water_bytes,
        "buffer_proof": {"parts": dict(proof.parts), "detail": dict(proof.detail),
                         "synthetic": proof.synthetic,
                         "verdict": proof.verdict},
        "ladders": {arm: {"points": [list(p) for p in lad.points],
                          "intercept_ms": lad.intercept_ms,
                          "slope_ms": lad.slope_ms,
                          "mean_rel_err": lad.mean_rel_err,
                          "across_repeat_spread": lad.spread,
                          "excluded_drifted": lad.excluded}
                    for arm, lad in ladders.items()},
        "treads_table": rows,
        "copies_declared": copies_declared,
        "path_census": census.as_dict(),
        "align_probe": probe.as_dict() if probe is not None else None,
        "declaration_fit": ({"switch_tread": decl_fit.switch_tread,
                             "switch_source": switch_source,
                             "step_ms": decl_fit.step_ms,
                             "per_tile_ms": decl_fit.per_tile_ms,
                             "intercept_ms": decl_fit.intercept_ms,
                             "per_tile_band": (list(decl_bands[0])
                                               if decl_bands[0] else None),
                             "step_band": (list(decl_bands[1])
                                           if decl_bands[1] else None),
                             "points": [list(p) for p in decl_fit.points],
                             "dof": decl_fit.dof}
                            if decl_fit is not None else None),
        "run_id": run_id or None,
        "seed": seed,
        "session_tag": session_tag or None,
        "ratio": ratio,
        # NaN IS NOT JSON. `json.dumps` writes a bare `NaN` token, which is
        # valid for Python's own loader and invalid for every strict parser
        # that reads these reports downstream. An interval that was not formed
        # is `null`, which is what "not formed" means.
        "ratio_interval": [None if not math.isfinite(v) else v
                           for v in interval],
        "ratio_interval_pct": INTERVAL_PCT,
        "ratio_draws": got_draws,
        "ratio_corrected": corrected,
        # WHAT THIS RUN ALONE SAID, beside the joint verdict in gates[C1]: a
        # reader of the document can tell which rule produced the verdict.
        "c1_verdict_alone": (c1_verdict(ratio, tuple(interval))
                             if ratio is not None else None),
        "replicates": (cross.as_dict() if cross is not None else None),
        "clock_correction": (clock_correction.as_dict()
                             if clock_correction is not None else None),
        "outcome": (outcome_for(ratio)[0] if ratio is not None else None),
        "outcomes_partition": [[n, lo, (None if hi == math.inf else hi), m]
                               for n, lo, hi, m in OUTCOMES],
        "gates": [g.as_dict() for g in gates],
    }
    if prov is not None:
        payload = prov.stamp(payload)
    return Report(lines, gates, payload)


# --------------------------------------------------------------------------
# Planted worlds: the scorer proven off GPU.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class World:
    """A planted world, what it is planted to demonstrate, and the verdicts.

    A SELF-TEST THAT ASSERTS NOTHING IS A SMOKE TEST. `expect` is the
    registration: the verdict every named gate must return in this world. A
    gate absent from `expect` is deliberately unregistered; a gate NAMED in
    `expect` that the report does not contain is itself a mismatch, because a
    registration that silently matches nothing is the
    check-that-examined-nothing shape one level up.
    """

    name: str
    why: str
    expect: dict[str, str]
    #: The re-read fraction the SHARED arm is generated at. The private arm is
    #: always generated at 1.0, which is what it is by construction.
    alpha: float = ALPHA
    #: Milliseconds per M-tile the native arm costs OVER the shared arm: the
    #: declaration, planted, which is what V5 measures. Charged on `n - n0`
    #: so the planted difference is a SLOPE and not the constant a healthy
    #: declaration costs; planted on native because native is not in the
    #: ratio, so the world moves V5 and nothing else.
    machinery_ms_per_tile: float = 0.0
    #: Relative perturbation applied to the private arm at the identity tread
    #: alone, which is what V6 measures. It also tilts the private slope, so
    #: the world that plants it registers V6 and nothing downstream of it.
    identity_skew: float = 0.0
    #: The under-load clock every planted cell reports, in MHz. A PLANTED
    #: value for a synthetic world, not a card's reading and scored against
    #: nothing but the other planted arms.
    clock_mhz: float = 1500.0
    #: Relative offset of the private arm's planted clock, which is what V7
    #: measures.
    private_clock_skew: float = 0.0
    #: The elasticity the planted private cells OBEY: their time is inflated
    #: by `(1 + private_clock_skew) ** -planted_eta`, the law the clock
    #: correction inverts, so a self-test run with `--clock-elasticity` at
    #: this eta must recover the refit world's raw ratio EXACTLY (to 1e-12 at
    #: zero noise) and its corrected/raw ratio must be
    #: `(1 + private_clock_skew) ** -planted_eta`. None means the split is
    #: planted with no time effect, which is what `clock-split` plants.
    planted_eta: float | None = None
    #: NATIVE's alignment step in the SWEEP, planted from the tread the census
    #: hypothesis says its kernel switches: the thing `declaration_fit` takes
    #: out of V5's number.
    alignment_step_ms: float = 0.0
    #: The PROBE's planted step at native's declaration (a real build has
    #: one) and at the ratio arms' declaration (a sound design has none).
    #: `None` on the native side means "the same step the SWEEP plants", which
    #: is what one op measured two ways has to mean: a world that plants 400 us
    #: in the ladder and 20 us in the probe is two different builds.
    native_probe_step_ms: float | None = None
    ratio_probe_step_ms: float = 0.0
    #: The ratio arms' common probe step sized IN BUDGETS rather than in ms:
    #: a multiple of ALIGN_STEP_RATIO_BUDGET, converted at the run's own
    #: weight-stream time and leverage by `planted_probe`, so no calibrated
    #: quantity is a literal here. Overrides `ratio_probe_step_ms`.
    ratio_probe_budgets: float | None = None
    #: The same, planted in PRIVATE's series ALONE. `pair_step_bias` exists
    #: because the two ratio arms' steps need not be one step, and every
    #: other world plants them equal, which makes the pair bound numerically
    #: the common-step bound it replaced. A world that steps one arm is the
    #: only registration that tells the two apart.
    private_probe_budgets: float | None = None
    #: What the instrument said about the planted probe's cells. A world that
    #: plants no verdict would leave V8 UNKNOWN on every page.
    probe_host_bound: bool = False
    #: The instrument the planted probe was timed on, mirroring the real one:
    #: `PROBE_CALLS_PER_REPLAY` (graph, GPU time) unless a world plants the
    #: eager fallback, which is what the two host-bound worlds are.
    probe_graph_calls: int = PROBE_CALLS_PER_REPLAY
    #: The buffer proof's planted verdict.
    proof_ok: bool = True
    #: Planted allocation observations, as a multiple of the prediction.
    weight_alloc_factor: float = 1.0
    high_water_factor: float = 1.0
    #: Multiplier on the roof, so a world can put the ladder over its ceiling.
    roof_factor: float = 1.0
    #: Multiplier on EVERY arm's milliseconds. A card that delivers faster than
    #: its own calibrated rate leaves the ratio untouched -- both terms move
    #: together -- and refutes the ceiling, which is what C2 is about and what
    #: no other field in this table can reach.
    speedup: float = 1.0
    #: Treads to delete from EVERY arm, so a world can thin the grid without
    #: unmatching it: V0 counts cells and fails, V1 compares tread SETS and
    #: does not, which is the distinction between the two gates.
    drop_treads: tuple[int, ...] = ()
    #: Treads to delete from ONE arm, which unmatches the tread sets and is the
    #: only thing V1 is about. Named separately because a world that drops from
    #: every arm registered V1 FAIL once and got PASS, correctly: the gates
    #: measure different failures and a world has to produce the one it names.
    drop_treads_from: tuple[str, tuple[int, ...]] | None = None

    def check(self, report: Report) -> list[str]:
        got = {g.tag: g.verdict for g in report.gates}
        bad = []
        for tag, want in sorted(self.expect.items()):
            if tag not in got:
                bad.append(f"{tag}: registered {want}, but the report has no "
                           f"gate {tag}")
            elif got[tag] != want:
                bad.append(f"{tag}: registered {want}, got {got[tag]}")
        if self.planted_eta is not None:
            # THE EXACT IDENTITY, not a tolerance on alpha: the correction
            # inverts the planted law cell by cell, so corrected / raw is
            # (1 + skew) ** -eta to rounding at ANY noise.
            block = report.payload.get("clock_correction")
            if not block:
                bad.append("clock_correction: this world plants an elasticity "
                           "and the page carries no clock-corrected ratio; "
                           f"run it with --clock-elasticity {self.planted_eta}")
            else:
                want_factor = (1.0 + self.private_clock_skew) ** -self.planted_eta
                got_factor = block["ratio"] / block["ratio_raw"]
                if abs(got_factor - want_factor) > 1e-9:
                    bad.append(f"clock_correction: corrected / raw = "
                               f"{got_factor:.12f}, the planted law says "
                               f"{want_factor:.12f}")
        return bad


ALL_PASS = {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
            "V5": PASS, "V6": PASS, "V7": PASS, "V8": PASS, "C1": PASS,
            "C2": PASS}

#: Every world this file knows, and every gate's FAIL branch is reachable from
#: one of them. A gate that cannot fail is as useless as one that cannot pass,
#: and until a world reaches its FAIL branch nothing has shown which it is.
WORLDS: dict[str, World] = {
    "refit": World(
        "refit",
        "the world the study says it is in: the shared ladder re-reads "
        f"alpha={ALPHA} of the weight set per M-tile, the private ladder "
        "re-reads all of it, and the ratio lands in ALPHA_BAND",
        dict(ALL_PASS)),
    "no-reuse": World(
        "no-reuse",
        "the first registered alternative: the shared ladder re-reads the "
        "WHOLE set per M-tile, so the ratio is 1.0 and the refit band is "
        "refuted from above",
        dict(ALL_PASS, C1=FAIL), alpha=1.0),
    "issue-bound": World(
        "issue-bound",
        "the second registered alternative: the shared ladder re-reads almost "
        f"nothing (alpha={RETRACTED_ALPHA}), so the per-M-tile cost is issue "
        "and latency and the traffic model is the wrong kind of model",
        dict(ALL_PASS, C1=FAIL), alpha=RETRACTED_ALPHA),
    "aliased": World(
        "aliased",
        "the relabelling silently sent every M-tile back to copy 0: the "
        "buffer proof fails, and without V2 this world would have printed a "
        "ratio of 1.0 and the NO-REUSE headline off an arm that measured no "
        "private read at all",
        {"V2": FAIL}, proof_ok=False),
    "machinery": World(
        "machinery",
        "the wider declaration costs as much per M-tile as the effect: V5 "
        "fails and the page is unquotable as the study's alpha, which is not "
        "the same statement as the matched ratio being wrong",
        dict(ALL_PASS, V5=FAIL), machinery_ms_per_tile=0.35),
    "compute-bound": World(
        "compute-bound",
        "the roof is low enough that the ladders run into it: V4 fails, "
        "because a slope set by arithmetic is not a slope about traffic",
        {"V4": FAIL}, roof_factor=0.02),
    "noisy-identity": World(
        "noisy-identity",
        "shared and private disagree at n=1, where they are the same call: "
        "this arm's own floor is above the effect it has to resolve, so V6 "
        "fails and the ratio is unreadable at any value. Registered on V6 "
        "ALONE: the skew sits on a ratio arm and tilts its slope, so what C1 "
        "reads in this world is not what the world is about",
        {"V6": FAIL}, identity_skew=0.08),
    "clock-split-elastic": World(
        "clock-split-elastic",
        "the private arm settled 2% below the shared arm's clock, the split "
        "session 4 read at G=1, AND its time obeys a planted elasticity of "
        "0.75: PRIVATE's slope is inflated by 0.98 ** -0.75, the raw ratio "
        "deflated by the same factor, and V7 fails while V6 still passes at "
        "n=1 (a 1.5% gap is under IDENTITY_SPREAD; clock-split's 3% at this "
        "elasticity would not be, and would fail V6 too); run with "
        "--clock-elasticity 0.75 the page prints a clock-corrected ratio "
        "that recovers the refit world's raw ratio exactly, and still scores "
        "the raw one: the correction is printed, V7 is unmoved, and the page "
        "is INVALID as it should be",
        dict(ALL_PASS, V7=FAIL), private_clock_skew=-0.02, planted_eta=0.75),
    "clock-split": World(
        "clock-split",
        "the private arm settled 3% below the shared arm's clock on a power "
        "capped card: V7 fails, because a ratio of slopes taken at two clocks "
        "carries a clock difference of unknown elasticity. The planted "
        "timings do not move, so every other gate passes and V7 is the only "
        "thing standing between this world and a published ratio",
        dict(ALL_PASS, V7=FAIL), private_clock_skew=-0.03),
    "alignment-step": World(
        "alignment-step",
        "NATIVE crosses vLLM's alignment-kernel switch inside the ladder and "
        "pays a step of 0.4 ms from there: the raw native - shared slope gap "
        "reads that step at the design's leverage and would FAIL V5, and the "
        "step-aware fit takes it out, so V5 PASSES and the step is printed "
        "with an interval. Registered ALL PASS: this is the world V5's fit "
        "exists for",
        dict(ALL_PASS), alignment_step_ms=0.4),
    "host-bound-probe": World(
        "host-bound-probe",
        "the probe's own cells came back HOST-BOUND and it did NOT resolve "
        "NATIVE's kernel switch where the census puts it: the positive "
        "control failed, so the probe has not shown it can see a switch "
        "through the host's enqueue cost, and V8 reads UNKNOWN rather than "
        "certifying a flat series it may have been blind to. On a pod the "
        "sweep still runs: the page latches INVALID on V8 and carries every "
        "other gate's number",
        dict(ALL_PASS, V8=UNKNOWN), probe_host_bound=True,
        probe_graph_calls=0, native_probe_step_ms=0.0),
    "graph-probe-unresolved": World(
        "graph-probe-unresolved",
        "the probe timed the op under a CUDA graph and resolved no NATIVE step "
        "in GPU time: the switch the census names costs less than the printed "
        "threshold per call, a bound on the kernel and not a blind instrument, "
        "so V8 PASSES on the GPU-timed ratio series. The same planted cells "
        "timed EAGERLY and host-bound are the host-bound-probe world, which "
        "reads UNKNOWN: the two differ in the instrument and in the verdict "
        "the instrument gave, nothing else",
        dict(ALL_PASS), native_probe_step_ms=0.0),
    "host-bound-controlled": World(
        "host-bound-controlled",
        "the probe's cells came back HOST-BOUND, as an EAGER probe's do on an "
        "H200 (the pre-2026-09-22 probe, and the fallback when the capture is "
        "refused), but it resolved NATIVE's kernel switch at the tread the "
        "census puts it: the positive control shows the probe sees a switch "
        "of this op at this size through the host cost, so the ratio arms' "
        "flat series is evidence and V8 PASSES. Without the control this "
        "world, which is the one a rented card WAS in, produced no ladder",
        dict(ALL_PASS), probe_host_bound=True, probe_graph_calls=0),
    "ratio-path-split": World(
        "ratio-path-split",
        "the probe finds a step at the RATIO arms' own declaration: on this "
        "build the alignment kernel switches inside their ladder after all "
        "(the expert bound is not where the source says), so V8 fails before "
        "a slope is trusted. The planted sweep is healthy, so every other "
        "gate passes and V8 is the only thing between this world and a "
        "published ratio",
        dict(ALL_PASS, V8=FAIL), ratio_probe_step_ms=0.5),
    "ratio-step-over-budget": World(
        "ratio-step-over-budget",
        "the probe finds a step at the ratio arms' own declaration worth 1.5 "
        "budgets on the ratio, the nearest a world sits to V8's FAIL edge: "
        "resolved and over, so V8 fails. Sized in budgets at the run's own "
        "weight stream, not in microseconds",
        dict(ALL_PASS, V8=FAIL), ratio_probe_budgets=1.5),
    "ratio-step-under-budget": World(
        "ratio-step-under-budget",
        "the same step at 0.7 budgets: real, resolved, and worth less than "
        "the budget on the ratio, so V8 PASSES. A step is not a defect; a "
        "step worth more than a hundredth of the ratio is",
        dict(ALL_PASS), ratio_probe_budgets=0.7),
    "private-step-alone": World(
        "private-step-alone",
        "the step is in PRIVATE's id set and NOT in SHARED's: the alignment "
        "op costs the two arms differently at the same declaration, which is "
        "the case pair_step_bias exists for and the only one the common-step "
        "bound it replaced could not express. Sized at the same 1.5 budgets "
        "ratio-step-over-budget plants in both arms, so the two worlds "
        "differ in exactly one variable: which series carries the step",
        dict(ALL_PASS, V8=FAIL), private_probe_budgets=1.5),
    "over-allocated": World(
        "over-allocated",
        "the weight allocation is not the one the plan priced: V3 fails, "
        "because copies that are not the copies the plan priced are not the "
        "no-reuse reference the plan registered",
        dict(ALL_PASS, V3=FAIL), weight_alloc_factor=1.5, high_water_factor=1.5),
    "holes": World(
        "holes",
        "two treads never landed, in every arm: V0 fails on the counts, "
        "because a ladder assembled from what survived still fits and still "
        "reports a slope. V1 PASSES here and that is the registration: the "
        "tread sets still match, so the two gates are about different failures",
        {"V0": FAIL, "V1": PASS}, drop_treads=(2, 3)),
    "faster-than-its-ruler": World(
        "faster-than-its-ruler",
        "every arm delivers 25% more bandwidth than the card's calibration "
        "claims is possible: C2 fails and C1 does not, because both terms of "
        "the ratio moved together. This is the world where the denominator "
        "every published w is divided by is refuted, and it is the only one "
        "this table can reach it from",
        dict(ALL_PASS, C2=FAIL), speedup=0.8),
    "ragged": World(
        "ragged",
        "two treads never landed IN ONE ARM: the ratio would be a comparison "
        "of two ladders measured over different grids, which is what V1 is "
        "for and what dropping from every arm does not produce",
        {"V0": FAIL, "V1": FAIL}, drop_treads_from=(PRIVATE, (2, 3))),
}


def planted_samples(world: World, cfg, *, block_m: int, treads: list[int],
                    repeats: int, alpha_shared: float, ridge: float,
                    bandwidth_gbps: float, b: int, noise: float,
                    seed: int, copies_declared: int | None = None,
                    native_switch: int | None = None) -> list[Sample]:
    """Cells GENERATED from the study's own traffic model at a stated alpha.

    Every row carries `SYNTHETIC_INSTRUMENT`, so "not measured" is a VALUE on
    the row and not an absence a reader has to notice. Nothing here was
    measured and the report says so on its own line.
    """
    rng = random.Random(seed)
    out: list[Sample] = []
    n_decl = copies_declared or treads[-1]
    for rep in range(repeats):
        for n in treads:
            if n in world.drop_treads:
                continue
            rows = n * block_m
            tokens = SWEEP.tokens_for_rows(cfg, rows)
            for arm in ARMS:
                if (world.drop_treads_from
                        and arm == world.drop_treads_from[0]
                        and n in world.drop_treads_from[1]):
                    continue
                a = 1.0 if arm == PRIVATE else alpha_shared
                ms = SWEEP.model_ms(cfg, rows, block_m, alpha=a, ridge=ridge,
                                    bandwidth_gbps=bandwidth_gbps, b=b)
                if arm == NATIVE:
                    # A SLOPE, charged on `n - n0`: what V5 is about.
                    ms += world.machinery_ms_per_tile * (n - treads[0])
                    # A STEP, from the tread native's kernel switches: what
                    # V5's fit takes out.
                    if native_switch is not None and n >= native_switch:
                        ms += world.alignment_step_ms
                if arm == PRIVATE and n == treads[0]:
                    ms *= (1.0 + world.identity_skew)
                if arm == PRIVATE and world.planted_eta is not None:
                    # THE LAW THE CORRECTION INVERTS: a slower clock makes a
                    # longer call, by the planted elasticity.
                    ms *= (1.0 + world.private_clock_skew) ** -world.planted_eta
                ms *= world.speedup
                if noise:
                    ms *= (1.0 + rng.gauss(0.0, noise))
                clock = world.clock_mhz * (1.0 + (world.private_clock_skew
                                                  if arm == PRIVATE else 0.0))
                out.append(Sample(
                    arm=arm, repeat=rep, block_m=block_m, tiles=n,
                    rows_per_expert=rows, tokens=tokens,
                    copies=copies_read(arm, n),
                    experts_declared=declared_experts(arm, cfg.num_experts,
                                                      n_decl),
                    ms_p50=ms, ms_min=ms,
                    ms_stdev=0.0, iters=0, trials=0, warmup_ms=0.0,
                    instrument=SYNTHETIC_INSTRUMENT,
                    sm_clock_load_mhz=clock, clock_level_ok=None,
                    clock_level_side="", clock_drift_ok=None, l2_flush=False))
    return out


#: What a world plants at native's declaration when it plants no sweep step:
#: a real build has a step there, and a planted probe with none would leave
#: `switch_source` reading "the hypothesis, unconfirmed" on every page.
DEFAULT_PLANTED_PROBE_STEP_MS = 0.02


def planted_step_in_budgets(budgets: float | None, world_name: str,
                            treads: list[int], split: int,
                            weight_stream_ms: float | None) -> float:
    """A planted probe step, in ms, sized so `pair_step_bias` reads exactly
    `budgets` ALIGN_STEP_RATIO_BUDGETs.

    `|a| / (B + a) = f` gives `a = f B / (1 - f)`, and `s = a / L`. The same
    inversion serves a step common to both ratio arms and a step in one of
    them alone: with `a = c` the bound is `|a| / (B + a)`, and with the
    numerator flat it is `|c| / (B + c)` -- the same number, which is why
    the two worlds sized at 1.5 budgets read the same bias and differ only
    in WHICH series carries the step. 0.0 for a world that plants none.
    """
    if budgets is None:
        return 0.0
    if not weight_stream_ms:
        raise Unmeasurable(f"the {world_name!r} world sizes its step in "
                           "budgets and needs the run's weight stream")
    f = budgets * ALIGN_STEP_RATIO_BUDGET
    return f * weight_stream_ms / (1.0 - f) / leverage(treads, split)


def planted_ratio_step_ms(world: World, treads: list[int], split: int,
                          weight_stream_ms: float | None) -> float:
    """The step planted in BOTH ratio series, in ms."""
    if world.ratio_probe_budgets is None:
        return world.ratio_probe_step_ms
    return planted_step_in_budgets(world.ratio_probe_budgets, world.name,
                                   treads, split, weight_stream_ms)


def planted_probe(world: World, cfg, *, block_m: int, treads: list[int],
                  declared_by_arm: dict[str, int], census: PathCensus,
                  noise: float, seed: int,
                  weight_stream_ms: float | None = None) -> AlignProbe:
    """A probe GENERATED for a planted world: an affine alignment cost per
    arm, native's step where the census hypothesis puts it, and the ratio
    arms' step -- the same in SHARED's and PRIVATE's series -- only in the
    world that plants one."""
    rng = random.Random(seed + 7)
    ratio_split = census.switch_tread(NATIVE) or treads[len(treads) // 2]
    ratio_step = planted_ratio_step_ms(world, treads, ratio_split,
                                       weight_stream_ms)
    private_step = planted_step_in_budgets(world.private_probe_budgets,
                                           world.name, treads, ratio_split,
                                           weight_stream_ms)
    cells = []
    for rep_ in range(PROBE_REPEATS):
        for n in treads:
            numel = ids_for_tread(cfg, n, block_m)
            for arm in ARMS:
                d = declared_by_arm[arm]
                ms = 0.012 + 8e-6 * numel
                sw = census.switch_tread(arm)
                native_step = (world.native_probe_step_ms
                               if world.native_probe_step_ms is not None
                               else max(world.alignment_step_ms,
                                        DEFAULT_PLANTED_PROBE_STEP_MS))
                if arm == NATIVE and sw is not None and n >= sw:
                    ms += native_step
                if arm != NATIVE and n >= ratio_split:
                    ms += ratio_step
                if arm == PRIVATE and n >= ratio_split:
                    ms += private_step
                if noise:
                    ms *= (1.0 + rng.gauss(0.0, noise))
                g = world.probe_graph_calls
                cells.append(ProbeCell(arm, n, numel, d, rep_, ms,
                                       host_bound=world.probe_host_bound,
                                       host_note=("planted host-bound"
                                                  if world.probe_host_bound
                                                  else ""),
                                       graph_calls=g,
                                       replay_ms=(ms * g if g else None)))
    return AlignProbe(tuple(cells), synthetic=True,
                      note=f"planted for the {world.name!r} world")


# --------------------------------------------------------------------------
# The GPU half.
# --------------------------------------------------------------------------

def build_private_weights(cfg, dtype: str, copies: int, seed: int,
                          device: str = "cuda"):
    """`(w1, w2, allocated_bytes)` with `copies` BITWISE IDENTICAL copies resident.

    ONE ALLOCATION AT THE DEEPEST TREAD, passed WHOLE to every call at every
    tread. That is not a saving, it is the design: the memory environment --
    the resident footprint, the allocator's state, the pressure on the TLB --
    is then IDENTICAL at every tread and in every arm, and so is the declared
    expert space, so nothing in the ladder moves because an allocation or a
    declaration moved. Copy `c` of expert `e` is slot `copy_slot(e, c, n_decl)`,
    EXPERT-FIRST, so the native arm's `w1[::n_decl]` is copy 0 of every expert
    as a strided view of the same storage (the kernel addresses experts
    through `stride(0)`, and vLLM asserts only `stride(-1) == 1`). `copies` is
    the DECLARED count, which `declared_copies_for` may set above the deepest
    tread's: the extra copies are filled like the others and never read.

    Every copy is FILLED, not merely reserved: an untouched copy is a page
    table entry, not a byte on the bus, and the arm's whole claim is about
    bytes on the bus.

    AND THE COPIES LEAVE HERE BITWISE IDENTICAL. Nothing distinguishing is
    written into them, which is what makes V2's `same_layer` part a bitwise
    comparison: the private arm reads different ADDRESSES holding the same
    BYTES, so if it computes anything other than the shared arm's output the
    relabelling is wrong. The sentinel write that tells one copy's memory from
    another's lives in `prove_distinct_buffers`, which runs after the last
    timed cell and restores what it wrote before the comparison; an earlier
    version of this function wrote the sentinels here and left them in, which
    made every copy different from every other and would have failed
    `same_layer` on a correct relabelling.
    """
    import torch

    e, h, f = cfg.num_experts, cfg.hidden_size, cfg.intermediate_size
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    total = expert_space(e, copies)
    cuda = device.startswith("cuda")
    before = torch.cuda.memory_allocated() if cuda else None
    w1 = torch.empty((total, 2 * f, h), dtype=dt, device=device)
    w2 = torch.empty((total, h, f), dtype=dt, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    # Fan-in scaling, the same shape `moe.reference.torch_ref.make_inputs`
    # uses, so the numerics sit where every other arm's do. Drawn into copy 0
    # (every n_max-th slot) and copied out to the rest.
    w1[::copies].normal_(0.0, h ** -0.5, generator=g)
    w2[::copies].normal_(0.0, f ** -0.5, generator=g)
    for c in range(1, copies):
        w1[c::copies].copy_(w1[::copies])
        w2[c::copies].copy_(w2[::copies])
    if not cuda:
        # None means NOT MEASURED, never zero: V3 then reads UNKNOWN, which
        # counts against it, where a zero would read as an allocation that did
        # not happen and FAIL it for the wrong reason.
        return w1, w2, None
    torch.cuda.synchronize()
    return w1, w2, torch.cuda.memory_allocated() - before


#: The value written into one element of each copy to tell its memory from
#: every other copy's. Far outside the fan-in-scaled range the weights are
#: drawn in (order 1e-2), exactly representable in bf16 and fp16, and NEGATIVE,
#: so a read-back that came from the wrong copy or from an uninitialised page
#: cannot be mistaken for a weight.
SENTINEL_BASE = -1024.0


def proof_calls(copies_read: int, copies_declared: int | None = None) -> int:
    """fused_experts calls the buffer proof makes: three before anything is
    zeroed, three per READ copy c >= 1 (private zeroed, shared zeroed, private
    restored), and two per never-read copy (private and shared under its
    zeroing). Counted here so the plan page and the session's cost note
    cannot drift from the proof."""
    declared = copies_read if copies_declared is None else copies_declared
    return (3 + 3 * max(0, copies_read - 1)
            + 2 * max(0, declared - copies_read))


def sentinel_roundtrip(w1, copies: int) -> tuple[bool, str]:
    """Write a distinct value into each copy, read them all back, PUT THE
    ORIGINAL BYTES BACK, and say whether all three held.

    Split out of the prover and made device-agnostic for one reason: the write
    and the restore have to be one operation, and the first version of this
    wrote sentinels at BUILD time and never restored them. Every copy was then
    different from every other, which is exactly the state part 5 exists to
    rule out, so a CORRECT relabelling would have failed `same_layer` and the
    arm would have exited INVALID on a working instrument. It is testable
    without a device so that pairing cannot come apart again.
    """
    import torch

    try:
        original = w1[:, 0, 0].clone()
        want = {c: SENTINEL_BASE * (c + 1) for c in range(copies)}
        for c in range(copies):
            w1[copy_slot(0, c, copies), 0, 0] = want[c]
        seen = {c: float(w1[copy_slot(0, c, copies), 0, 0].item())
                for c in range(copies)}
        w1[:, 0, 0] = original
        restored = bool(torch.equal(w1[:, 0, 0], original))
    except RuntimeError as exc:
        # A FAILED PART, NOT A CRASH. torch refuses to write a tensor whose
        # elements share a memory location, which is exactly the world this
        # part exists to detect; raising here would exit ERROR (the apparatus
        # broke) where the truth is INVALID (the instrument is not what the
        # page says it is). Nothing is timed after the proof, so the sentinels
        # left behind by a half-finished round trip cost nothing.
        return False, (f"torch refused the sentinel round trip ({exc}); that "
                       "refusal is itself the finding: the copies do not hold "
                       "distinct memory")
    ok = seen == want and len(set(seen.values())) == copies and restored
    return ok, (f"wrote {copies} distinct values, read back "
                f"{len(set(seen.values()))} distinct, and restored the "
                f"original bytes: {restored}")


def _sync(t) -> None:
    """Synchronise when the tensor lives on a CUDA device, and only then, so
    the proof runs unchanged against a CPU reference in the test suite."""
    if getattr(t, "is_cuda", False):
        import torch
        torch.cuda.synchronize()


def prove_distinct_buffers(call_for, w1, w2, *, cfg, copies_read: int,
                           dtype: str, private_ids,
                           copies_declared: int | None = None) -> BufferProof:
    """Run the five-part proof. Returns the parts and their evidence.

    `call_for(arm)` returns a zero-argument callable producing that arm's
    `[T, H]` output at the deepest tread; `private_ids` are the `[T, k]` slot
    ids that call passes, from which the tokens routed to each copy are read.
    `copies_read` is the deepest tread's count; `copies_declared`, at or above
    it, is the allocation. A declared-but-never-read copy is zeroed too, and
    NEITHER arm's output may move: that is the check that the padding
    `declared_copies_for` adds is inert.

    ONE COPY AT A TIME, AND RESTORED. For every copy `c >= 1` the proof zeroes
    that copy alone, checks that the private output changed on EXACTLY the
    tokens with a slot routed to copy `c` (and on no other token, bitwise),
    that the shared output did not change at all, then puts the copy back and
    checks the private output is bitwise what it was. An earlier version zeroed
    copies 1..n-1 together and never restored them: that passes a relabelling
    that sends every tile with `c >= 1` to copy 1, and it left the weights
    corrupted. Copy 0 is not zeroed -- the shared arm reads it -- and is
    covered by `same_layer`.

    STILL RUN AFTER THE LAST TIMED CELL, because a restore that failed would
    otherwise time a ladder against a corrupted weight set; the restore check
    is part of `shared_blind` and says so if it did.

    THE RESTORE COPIES FROM COPY 0 AND KEEPS NO BACKUP. Every copy is bitwise
    copy 0 by construction, copy 0 is never zeroed, and its sentinels are put
    back before this loop. A version that cloned the copy before zeroing it
    held one whole weight set (two, at the tuple rebind) above the sweep's
    peak, and V3 scores the peak against a ceiling with under a gigabyte of
    slack: a perfect run came back INVALID on its own proof. The restore is
    still CHECKED, bitwise, against the output from before the zeroing.

    THE SENTINELS ARE WRITTEN AND THEN PUT BACK, inside part 2, BEFORE part 5
    compares the two arms' outputs. A distinct value per copy is what tells one
    copy's memory from another's; leaving it there would make the copies
    different from each other, which is precisely the state part 5 exists to
    rule out, and `same_layer` would then FAIL on a correct relabelling.
    """
    import torch

    e = cfg.num_experts
    copies = copies_read if copies_declared is None else copies_declared
    if copies < copies_read:
        raise PrivateWeightRefusal(f"{copies} copies declared, {copies_read} read")
    parts: dict[str, bool] = {}
    detail: dict[str, str] = {}

    # 1. addresses, expert-first, and the native view over copy 0.
    item = w1.element_size()
    ok = True
    for ex in range(e):
        for c in range(copies):
            slot = copy_slot(ex, c, copies)
            if (w1[slot].data_ptr() != w1.data_ptr() + slot * w1.stride(0) * item
                    or w2[slot].data_ptr()
                    != w2.data_ptr() + slot * w2.stride(0) * item):
                ok = False
    native1, native2 = w1[::copies], w2[::copies]
    view_ok = (native1.data_ptr() == w1.data_ptr()
               and native1.stride(0) == copies * w1.stride(0)
               and native2.stride(0) == copies * w2.stride(0)
               and native1.shape[0] == e and native1.stride(-1) == 1)
    span = w1.shape[0] * (w1.stride(0) + w2.stride(0)) * item
    want = weight_bytes_total(cfg, dtype, copies)
    parts["addresses"] = ok and view_ok and span == want
    detail["addresses"] = (
        f"{copies} copies x {e} experts span {span} bytes against {want} "
        f"predicted; slot = expert x {copies} + copy at every slot: {ok}; "
        f"native view is every {copies}th slot of the same storage: {view_ok}")

    # 2. sentinels
    parts["sentinels"], detail["sentinels"] = sentinel_roundtrip(w1, copies)

    # 5. same layer, measured BEFORE anything is zeroed
    shared_before = call_for(SHARED)().clone()
    private_before = call_for(PRIVATE)().clone()
    native_before = call_for(NATIVE)().clone()
    parts["same_layer"] = bool(torch.equal(shared_before, private_before))
    detail["same_layer"] = (
        ("private output is bitwise equal to shared"
         if parts["same_layer"] else
         "private and shared outputs DIFFER with nothing zeroed; the "
         "relabelling is not computing the same layer")
        + "; native output is "
        + ("bitwise equal to shared" if torch.equal(native_before, shared_before)
           else "NOT bitwise equal to shared")
        + " (recorded, not scored: native runs a different launch grid)")

    # 3 and 4. one copy at a time: the read copies, then the never-read ones.
    copy_of_slot = (private_ids.to(torch.int64) % copies)
    read_ok = True
    blind_ok = True
    notes = []
    for c in range(copies_read, copies):
        w1[c::copies].zero_()
        w2[c::copies].zero_()
        _sync(w1)
        inert = (bool(torch.equal(call_for(PRIVATE)(), private_before))
                 and bool(torch.equal(call_for(SHARED)(), shared_before)))
        w1[c::copies].copy_(w1[::copies])
        w2[c::copies].copy_(w2[::copies])
        _sync(w1)
        blind_ok = blind_ok and inert
        notes.append(f"c={c} (never read): zeroing it left both outputs "
                     f"unchanged={inert}")
    for c in range(1, copies_read):
        expect = (copy_of_slot == c).any(dim=1)
        w1[c::copies].zero_()
        w2[c::copies].zero_()
        _sync(w1)
        private_zeroed = call_for(PRIVATE)()
        shared_zeroed = call_for(SHARED)()
        changed = (private_zeroed != private_before).any(dim=1)
        exact = bool(torch.equal(changed, expect))
        unchanged_shared = bool(torch.equal(shared_zeroed, shared_before))
        del private_zeroed, shared_zeroed
        w1[c::copies].copy_(w1[::copies])
        w2[c::copies].copy_(w2[::copies])
        _sync(w1)
        restored = bool(torch.equal(call_for(PRIVATE)(), private_before))
        read_ok = read_ok and exact and int(expect.sum()) > 0
        blind_ok = blind_ok and unchanged_shared and restored
        notes.append(f"c={c}: {int(changed.sum())} rows changed, "
                     f"{int(expect.sum())} routed, exact={exact}, "
                     f"shared unchanged={unchanged_shared}, "
                     f"restored={restored}")
    if copies_read < 2:
        read_ok = blind_ok = False
        notes.append("fewer than two copies read: nothing to prove a private "
                     "read of")
    parts["kernel_read"] = read_ok
    detail["kernel_read"] = (
        ("every copy c >= 1, zeroed alone, changed exactly the tokens routed "
         "to it" if read_ok else
         "at least one copy, zeroed alone, did NOT change exactly the tokens "
         "routed to it: some tile reads a copy that is not its own, and the "
         "private ladder is not a no-reuse reference")
        + " | " + "; ".join(notes))
    parts["shared_blind"] = blind_ok
    detail["shared_blind"] = (
        "the shared output was bitwise unchanged by every zeroing, every "
        "restore put the private output back bitwise, and zeroing a never-read "
        "copy moved neither output"
        if blind_ok else
        "the shared output CHANGED under a zeroing, a restore did not put the "
        "private output back, or a never-read copy was read: the zeroing "
        "corrupted what it should not have, and part 3 proves nothing")
    return BufferProof(parts=parts, detail=detail, synthetic=False)


def run_sweep(args, cfg, *, block_m: int, treads: list[int], pinned: dict,
              csv_path: Path, cache_root: Path, store: Store, prov,
              dtype: str, copies_declared: int, census: PathCensus,
              stream_ms: float
              ) -> tuple[list[Sample], BufferProof, int | None, int | None,
                         AlignProbe]:
    """The metered part. Appends every cell as it lands, so aborting keeps it.

    THE PROBE RUNS FIRST AND CAN END IT. Before a weight is allocated the
    alignment op is timed along the ladder once per arm, and V8 is
    scored on it: a FAIL means this build switches kernel inside the ratio
    arms' ladder, the sweep would measure a slope with a step in it, and the
    sweep is SKIPPED. Seconds, against minutes of card time for an INVALID.

    REPEATS OUTER, ARMS ROTATED, TREADS ALTERNATED. A repeat walks the whole
    ladder and the three arms are rotated inside each tread, so no arm is first
    in every triple and a drift over the session lands on all three rather
    than on the one that always ran last. The tread order is REVERSED on odd
    repeats, so a drift within a repeat does not always climb with n -- which
    would add the drift times each arm's own intercept to its slope, and the
    intercepts differ between arms. Within a repeat the inputs for a tread are
    built once and the three arms share them, which is what makes the
    comparison a comparison.
    """
    import torch

    from moe.bench import timing
    from moe.spec import BenchSpec, RoutingSpec

    reference_clock, clock_source = SWEEP.reference_clock_mhz()
    print("reference clock: "
          + (f"{reference_clock:.0f} MHz, {clock_source}" if reference_clock
             else f"NOT RESOLVED ({clock_source}); every cell's clock LEVEL "
                  "verdict will be None and no cell can be excluded for it"))

    # BEFORE vLLM is imported: Triton may snapshot this at import, and a warm
    # cache compiles and dumps nothing.
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)

    override_config, where = SWEEP.find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.baselines._framework_config import vllm_call_kwargs

    print(f"override hook: {where}.override_config")
    print(f"triton cache: {cache_root} (fresh for this run)")

    e = cfg.num_experts
    deepest = treads[-1]
    declared_by_arm = {arm: declared_experts(arm, e, copies_declared)
                       for arm in ARMS}
    probe = probe_alignment(cfg, block_m=block_m, treads=treads,
                            declared_by_arm=declared_by_arm,
                            copies_declared=copies_declared,
                            reference_clock=reference_clock,
                            repeats=args.probe_repeats)
    early = gate_v8_alignment(probe, treads=treads, census=census,
                              weight_stream_ms=stream_ms)
    print("\n".join(["", "ALIGNMENT PROBE, before any weight is allocated:",
                     *[f"  {ln}" for ln in early.lines]]))
    if early.verdict == FAIL:
        # ON FAIL ONLY. A FAIL is a statement about the BUILD: it switches
        # alignment kernel inside the ratio arms' ladder, and no slope
        # measured here would be free of it. An UNKNOWN is a statement about
        # the INSTRUMENT -- on the eager fallback, with NATIVE's switch as the
        # positive control, it means the probe could not see a switch it was
        # shown; under the graph, that a step is over budget and unresolved
        # or that the replay's launch outran its calls -- and the
        # page still latches INVALID on it, but throwing away the ladder and
        # every other gate's number over an inconclusive probe is the wrong
        # trade on a rented card. So the sweep runs and V8 stays on the page.
        # NAMING WHAT WAS MEASURED, because this line is the only thing the
        # pod prints when the arm ends here: the gate's own `measured` string
        # carries the bound, which of the two ratio series resolved a step,
        # and the bound over the resolved ones.
        print("SWEEP SKIPPED: V8 came back FAIL, so this build switches "
              "alignment kernel inside the ratio arms' ladder and no slope "
              f"measured here would be free of it ({early.measured}). "
              "Nothing was allocated and nothing was timed.")
        return ([], BufferProof(parts={}, detail={"skipped": (
                    f"V8 came back {early.verdict} on the probe; the proof "
                    "did not run")},
                                synthetic=False), None, None, probe)
    if early.verdict != PASS:
        print(f"V8 came back {early.verdict} on the probe: the sweep RUNS, "
              "because an inconclusive probe is a statement about the "
              "instrument and not the design; the page will latch INVALID on "
              "V8 and carry every other gate's number beside it.")

    torch.cuda.reset_peak_memory_stats()
    w1, w2, weight_delta = build_private_weights(cfg, dtype, copies_declared,
                                                 args.seed)
    native_w1, native_w2 = w1[::copies_declared], w2[::copies_declared]
    print(f"private weights: {copies_declared} copies declared ({deepest} read "
          f"at the deepest tread), {weight_delta / 1e9:.4f} GB allocated and "
          "filled")

    # A DRIFTED CELL IS NOT DONE. Keyed on `status == "ok"`, a resume of the
    # same command skipped every cell the clock had drifted through -- the one
    # class of cell a second pass could actually fix -- and the only recovery
    # left was deleting cells.csv and re-paying the whole sweep. `usable` is
    # the same predicate every ladder fits on, so what a resume re-measures is
    # exactly what the gates are missing.
    done = {(s.arm, s.tiles, s.repeat) for s in read_samples(csv_path)
            if s.usable}
    samples = read_samples(csv_path)
    conf = dict(pinned, BLOCK_SIZE_M=block_m)

    def inputs_for(n: int):
        rows = n * block_m
        tokens = SWEEP.tokens_for_rows(cfg, rows)
        spec = BenchSpec(cfg, num_tokens=tokens, dtype=dtype,
                         routing=RoutingSpec("uniform", 0.0), seed=args.seed)
        x = torch.randn((tokens, cfg.hidden_size), device="cuda",
                        dtype=w1.dtype)
        ids = SWEEP.balanced_ids(cfg, tokens, "cuda")
        weights = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                             device="cuda")
        private_ids = private_topk_ids(ids, e, block_m, rows, copies_declared)
        shared_ids = shared_topk_ids(ids, copies_declared)
        kw = vllm_call_kwargs(spec)
        kw["activation"] = MoEActivation(kw["activation"])
        return tokens, x, {NATIVE: ids, SHARED: shared_ids,
                           PRIVATE: private_ids}, weights, kw

    def call_for(arm: str, x, ids_by_arm, weights, kw):
        # ONE DECLARATION FOR BOTH RATIO ARMS AT EVERY TREAD, over the whole
        # allocation; native is the study's call over copy 0 through a view.
        experts = declared_by_arm[arm]
        a1, a2 = (native_w1, native_w2) if arm == NATIVE else (w1, w2)
        args_kw = dict(kw, global_num_experts=experts)
        use_ids = ids_by_arm[arm]

        def call():
            return fused_experts(hidden_states=x, w1=a1, w2=a2,
                                 topk_weights=weights, topk_ids=use_ids,
                                 **args_kw)
        return call

    started = time.time()
    for rep in range(args.repeats):
        for n in (treads if rep % 2 == 0 else list(reversed(treads))):
            # A RESUME BUILDS NOTHING IT IS NOT GOING TO TIME. The inputs are
            # per (tread, repeat) and the three arms share them, so a tread
            # whose whole triple is already on disk is skipped before the
            # allocation rather than after it.
            if all((a, n, rep) in done for a in ARMS):
                continue
            tokens, x, ids_by_arm, weights, kw = inputs_for(n)
            # THE ROTATION IS DERIVED FROM `ARMS`, never listed a second time.
            order = [ARMS[(i + rep) % len(ARMS)] for i in range(len(ARMS))]
            for arm in order:
                if (arm, n, rep) in done:
                    continue
                call = call_for(arm, x, ids_by_arm, weights, kw)
                copies = copies_read(arm, n)
                experts = declared_by_arm[arm]
                try:
                    with override_config(conf):
                        call()
                        torch.cuda.synchronize()
                        t = time_cell(call, duty=args.duty,
                                      warmup_ms=args.warmup,
                                      cell_budget_ms=args.cell_budget_ms,
                                      trials=args.trials,
                                      l2_flush=not args.no_l2_flush,
                                      reference_clock_mhz=reference_clock)
                    # THE SIDE TRAVELS WITH THE VERDICT, with every other
                    # column the instrument measured: `sample_from_timing`
                    # copies them all. A failed LEVEL without its side is
                    # refused at construction, because read as LOW it would
                    # drop every boosted tread; NEITHER side excludes here,
                    # DRIFT alone does.
                    sample = sample_from_timing(
                        t, arm=arm, repeat=rep, block_m=block_m, tiles=n,
                        rows_per_expert=n * block_m, tokens=tokens,
                        copies=copies, experts_declared=experts)
                    if t.clock_level_side:
                        print(f"  ^ LEVEL {t.clock_level_side.upper()}: kept in "
                              "every fit, side recorded; its fraction of the "
                              "fixed roof is read beside the own-clock one")
                except timing.TimingRefused:
                    # THE INSTRUMENT'S OWN REFUSAL IS NOT ONE CELL'S ERROR.
                    # Filed per cell, the arm walks its whole grid writing
                    # zeroed rows and exits DONE.
                    raise
                except Exception as exc:                  # noqa: BLE001
                    sample = Sample(
                        arm=arm, repeat=rep, block_m=block_m, tiles=n,
                        rows_per_expert=n * block_m, tokens=tokens,
                        copies=copies, experts_declared=experts, ms_p50=0.0,
                        status="failed",
                        detail=f"{type(exc).__name__}: {exc}")
                    print(f"  {arm} n={n} rep={rep} FAILED  {sample.detail}")
                samples.append(sample)
                store.append(sample, prov)
                print(f"  rep{rep:2d} {arm:8s} n={n:3d} T={tokens:7d} "
                      f"copies={copies:3d} E*={experts:5d} "
                      f"{sample.ms_p50:9.4f} ms")
    print(f"\nswept in {time.time() - started:.0f} s")

    # THE PROOF IS LAST, and it zeroes buffers: nothing is timed after it.
    #
    # AND IT RUNS UNDER THE SAME PIN AS EVERY TIMED CELL. Outside it, vLLM
    # re-resolves the tile per call from `E, _, N = w2.shape`, so the proof's
    # NATIVE call (E=8) would take this card's tuned entry and its SHARED and
    # PRIVATE calls (E = E*n_decl) would fall to `get_default_config`, which is a
    # different BLOCK_SIZE_M, BLOCK_SIZE_N and GROUP_SIZE_M. Two consequences,
    # and both are defects: `same_layer` is a BITWISE comparison between two
    # calls that would not be guaranteed to run one kernel configuration, and
    # every part of V2 would be evidence about a launch at a tile the ratio is
    # not made of. Every other arm in this tree wraps its kernel calls; this
    # one did not.
    tokens, x, ids_by_arm, weights, kw = inputs_for(deepest)
    with override_config(conf):
        proof = prove_distinct_buffers(
            lambda arm: call_for(arm, x, ids_by_arm, weights, kw),
            w1, w2, cfg=cfg, copies_read=deepest, dtype=dtype,
            private_ids=ids_by_arm[PRIVATE], copies_declared=copies_declared)
    return (samples, proof, weight_delta, torch.cuda.max_memory_allocated(),
            probe)


# --------------------------------------------------------------------------
# Persistence and CLI.
# --------------------------------------------------------------------------

def detect_card_slug() -> str:
    """Slug for the ATTACHED device, or `NO_CARD_SLUG`.

    THE CARD IS A SWEPT PARAMETER, swept by the operator moving pods, and the
    results root defaults to a network volume that outlives the pod. Without
    the card in the run id the same command on two cards shares one directory,
    the second finds every cell present and prints the first card's timings
    under the second's heading.
    """
    try:
        import torch
    except ImportError:
        return NO_CARD_SLUG
    try:
        if not torch.cuda.is_available():
            return NO_CARD_SLUG
        name = torch.cuda.get_device_name(0)
    except Exception:                                     # noqa: BLE001
        return NO_CARD_SLUG
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or NO_CARD_SLUG


NO_UUID_PREFIX = "no-uuid:"


def device_identity() -> str:
    """The attached GPU's UUID; `no-uuid:<slug>` when a device is attached but
    its UUID cannot be read; "" with no device at all.

    The card SLUG names a model of card; the UUID names the card. Two H200
    pods share a slug, and this study has measured two rentals of one card
    type 7.1% apart. A torch that exposes no UUID gets the weaker identity
    RECORDED, so a resume on the same slug is still possible and a resume
    from a UUID-bearing record onto it is still refused; the guard's
    docstring says what that costs.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
    except Exception:                                     # noqa: BLE001
        return ""
    try:
        return str(torch.cuda.get_device_properties(0).uuid)
    except Exception:                                     # noqa: BLE001
        return NO_UUID_PREFIX + detect_card_slug()


DEVICE_FILE = "DEVICE"


def device_guard(out_dir: Path, identity: str) -> str:
    """"" when cells on disk were measured on this card (or none exist), else
    why a resume here would mix two cards.

    Writes `DEVICE` on first use. A directory with cells in it and no DEVICE
    file, or a different identity, is REFUSED rather than resumed: the resume
    is keyed on (arm, tread, repeat), so the new card would fill the holes in
    the old card's ladder and the report would print one ratio from two
    cards. A `no-uuid:<slug>` identity matches only itself, so on a torch
    with no UUID two pods of one card type CAN resume each other: the guard
    is then only as strong as the slug, and the page's DEVICE line says so.
    """
    marker = out_dir / DEVICE_FILE
    has_cells = (out_dir / "cells.csv").exists()
    recorded = marker.read_text().strip() if marker.exists() else None
    if recorded is None:
        if has_cells:
            return (f"{out_dir} holds cells.csv but no {DEVICE_FILE} file, so "
                    "the card those cells came from is unknown; resuming would "
                    "put this card's cells into that ladder. The driver passes "
                    "--session-tag itself, from the session directory's name, "
                    "so the way to a fresh directory is a fresh session "
                    "(`--new`); running this script by hand takes --run-id.")
        if identity:
            out_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(identity + "\n")
        return ""
    if not identity:
        return (f"this card's UUID could not be read, and {out_dir} holds cells "
                f"from {recorded}; a resume cannot be shown to be on the same "
                "card")
    if identity != recorded:
        return (f"{out_dir} holds cells measured on {recorded} and this card is "
                f"{identity}; resuming would print one ratio from two cards. "
                "The driver passes --session-tag itself, from the session "
                "directory's name, so the way to a fresh directory is a fresh "
                "session (`--new`); running this script by hand takes "
                "--run-id.")
    return ""


def git_visibility(path: Path) -> str:
    """ASK GIT whether it would keep this path. Never assert it from memory.

    rc 0 ignored, rc 1 kept, anything else UNVERIFIED and said so: rc 128 is
    what `git check-ignore` returns for a path outside the work tree, which is
    the pod default `/workspace/results/...`, and reporting that as tracked is
    the failure this function exists to prevent.
    """
    try:
        proc = subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=str(HERE.parent), capture_output=True,
                              timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git check-ignore could not run ({exc}); path UNVERIFIED"
    if proc.returncode == 0:
        return ("IGNORED by git: nothing written here enters the repo, which is "
                "the intended deal for a pod run.")
    if proc.returncode == 1:
        return "git WILL KEEP this path: anything written here is committable."
    return (f"git check-ignore exited {proc.returncode}; path UNVERIFIED "
            f"({proc.stderr.decode(errors='replace').strip()}). Common cause: "
            "the path is outside this work tree, e.g. /workspace/results on a "
            "pod, which git has no opinion about at all.")


def default_run_id(args, card: str) -> str:
    """Derived from every argument that changes a measured cell.

    IN THE KEY: the card, the model, the dtype, the tile, the pinned tile
    knobs, the ladder depth, the repeat count, and every TIMING knob
    (`--warmup`, `--trials`, `--cell-budget-ms`, the L2 flush), because those
    change the measured milliseconds and a resumed run keyed on
    `(arm, tread, repeat)` would otherwise print one timing regime's numbers
    under another's label.

    AND `--seed`, which is in the key because it also draws the weights and
    the inputs, not only the bootstrap.

    AND `--session-tag`, which is what stops a SECOND SESSION resuming the
    first. The results root is scoped to the card and not to the session, so
    without it a `--new` session on the same card found every usable cell of
    the last one on disk, timed nothing, and scored those cells as its own --
    and a resume on a DIFFERENT pod with the same card slug mixed two cards'
    timings in one ladder (see `device_guard`).

    OUT OF THE KEY: `--ridge`, `--bandwidth-gbps`, `--draws`, `--replicate-of`
    (and `--read`, which forms no id). They re-analyse one set of cells, and
    two analyses of one sweep belong in one directory; a replicate read beside
    this run moves C1's verdict and not one measured millisecond.
    `--device-memory-gb` is out for the same reason: it gates a plan, it does
    not move a millisecond. AND `--clock-elasticity`, which is admissible out
    of the key ONLY because it changes no verdict: it prints a corrected ratio
    beside the raw one. Were it ever scored, a re-run at another eta would
    re-score the cells on disk into the same directory and overwrite a page
    that read INVALID, which is the reason it never will be.

    AND `--probe-repeats`, WHICH IS OUT ON PURPOSE AND IS THE ONE THAT LOOKS
    LIKE IT SHOULD BE IN. It changes the probe's cells and so can change V8's
    verdict. But the probe is re-timed on every invocation and is never
    resumed -- nothing in `cells.csv` comes from it, and the resume key is
    `(arm, tread, repeat)` over the LADDER -- so two runs differing only in
    this knob hold the same measured ladder, and keying on it would put them
    in different directories and stop the second resuming the first's card
    minutes. The value is not lost: `report.json` carries `align_probe` with
    every probed cell and its repeat index, so the count is recoverable from
    the artefact this key exists to protect.

    A SELF-TEST IS PREFIXED AND ITS WORLD IS IN THE KEY. A planted report
    written into a metered run's directory would overwrite the only
    machine-readable artefact of an arm that cost pod minutes, with a synthetic
    one; the prefix and the world keep them apart, and `report.json` carries
    `synthetic: true` besides.
    """
    swept = {
        "model": args.model, "dtype": args.dtype, "bm": args.block_m,
        "n": args.block_n, "g": args.group_m, "stages": args.num_stages,
        "treads": args.treads, "reps": args.repeats,
        "warmup": args.warmup, "budget": args.cell_budget_ms,
        "trials": args.trials, "l2flush": not args.no_l2_flush,
        "seed": args.seed,
        "session": args.session_tag or "bare",
        # The declaration sizes the launch grid, so it moves a millisecond;
        # "auto" is a function of knobs already in the key.
        "decl": args.declared_copies or "auto",
        # NEVER None: `provenance.run_id` refuses an unresolved knob, and it is
        # right to. "measured" is a resolved value that says a real card was
        # asked; a planted world is a different resolved value.
        "planted": "measured" if args.self_test is None else args.self_test,
        "plantnoise": args.plant_noise,
        # The duty moves board power and so the clock every cell is timed at;
        # in the key WHEN IT IS NOT 1.0, so every run id written before the
        # knob existed (all of them at full duty) is the id the same command
        # still produces: the same command names the same directory. It does
        # NOT resume there. That directory's cells.csv predates the duty and
        # diagnostic columns, `Store` refuses its header (SchemaCollision)
        # rather than append wider rows under it, and --read still re-reads
        # its report.json.
        **({"duty": args.duty} if args.duty != 1.0 else {}),
    }
    prefix = "synthetic-" if args.self_test is not None else ""
    return prefix + PV.run_id(card=card, **swept)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_CONFIGS),
                    help="mixtral by default; see DESIGN DECISION 1 for why "
                         "and not deepseek-v2-lite")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"),
                    help="float only: this arm allocates and copies whole "
                         "weight sets and an fp8 path would quantise each copy "
                         "separately")
    ap.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M,
                    help="the one tile height this run measures; in the run id")
    ap.add_argument("--treads", type=int, default=DEFAULT_TREADS,
                    help="ladder depth in M-tiles per expert; the deepest "
                         "tread sets the memory bill at one copy per tile")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help="repeats of the whole ladder; the bootstrap resamples "
                         "these")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                    help="bootstrap draws for the interval on the ratio")
    ap.add_argument("--seed", type=int, default=0,
                    help="draws the weights, the inputs and the bootstrap; in "
                         "the run id. A second run at another seed is a "
                         "REPLICATE: read it back with --replicate-of "
                         "(DESIGN DECISION 14)")
    ap.add_argument("--block-n", type=int, default=SWEEP.FIXED["BLOCK_SIZE_N"])
    ap.add_argument("--group-m", type=int, default=SWEEP.FIXED["GROUP_SIZE_M"])
    ap.add_argument("--num-stages", type=int, default=SWEEP.FIXED["num_stages"])
    ap.add_argument("--warmup", type=float, default=300.0,
                    help="MILLISECONDS of delivered GPU load, not a call count")
    ap.add_argument("--cell-budget-ms", type=float, default=200.0)
    ap.add_argument("--duty", type=float, default=1.0,
                    help="duty cycle every ladder cell is timed at (DESIGN "
                         "DECISION 15). 1.0, the default, keeps the queue "
                         "full, the driver's instrument, and stays the "
                         "default so --replicate-of still reads session 4's "
                         "full-duty runs as this design. Below 1 each cell is "
                         "bursts of ~40 ms of kernel time with idle gaps of "
                         "burst x (1/duty - 1), lowering average board power. "
                         "The pod setting is 0.25: session 4's clock arm "
                         "(2026-09-21) read the clock flat at 1965 MHz there, "
                         "and still tracking board power at 0.5 (-1.09 "
                         "MHz/W). V7 "
                         "checks it either way. Wall clock ~1/duty x. In the "
                         "run id")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--no-l2-flush", action="store_true")
    ap.add_argument("--capability", default="",
                    help="compute capability for the off-GPU resource check, "
                         "e.g. 9.0; read from the device when it is attached")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="operator's assertion; otherwise the attached card's "
                         "own calibration")
    ap.add_argument("--ridge-band", default="")
    ap.add_argument("--clock-elasticity", type=float, nargs="+", default=None,
                    metavar="ETA",
                    help="a MEASURED per-M-TILE clock elasticity, eta or "
                         "eta lo hi, from scripts/clock_elasticity.py: its "
                         "gated claim (elasticity.value in a report written "
                         "since 8d4eb78, 2026-09-22), NOT its pooled per-call "
                         "'eta, fixed tread, pooled' reading, which is "
                         "elasticity.fixed_tread and also the elasticity.value "
                         "of an older report, session 4's included. The ratio "
                         "reads only slopes, and when each arm holds one clock "
                         "the per-M-tile elasticity is the one that carries a "
                         "slope exactly. The page then PRINTS a clock-corrected "
                         "ratio beside the raw one. Scores nothing and moves "
                         "no gate; needs --clock-elasticity-source")
    ap.add_argument("--clock-elasticity-source", default="",
                    help="where --clock-elasticity was read from (a report "
                         "path, the key read, e.g. elasticity.value, and the "
                         "commit), recorded on the page and in report.json")
    ap.add_argument("--bandwidth-gbps", type=float, default=0.0)
    ap.add_argument("--device-memory-gb", type=float, default=0.0,
                    help="a HYPOTHETICAL card's memory, for checking the plan "
                         "off a GPU box. Ignored when a device is attached, "
                         "which is asked instead")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="the refit the predictions are printed against; it "
                         "scores nothing")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--replicate-of", type=Path, nargs="+", default=(),
                    metavar="REPORT",
                    help="earlier report.json files (or run directories) of "
                         "the SAME design; C1 is then scored on the envelope "
                         "of every run's interval and the page prints the "
                         "cross-run spread. Out of the run id. Refused under "
                         "--self-test")
    ap.add_argument("--read", type=Path, default=None, metavar="REPORT",
                    help="score THIS stored report.json together with "
                         "--replicate-of, off GPU: nothing is measured, "
                         "nothing is written")
    ap.add_argument("--session-tag", default="",
                    help="the driving session's name; in the run id, so a new "
                         "session measures fresh and a resumed one resumes")
    ap.add_argument("--declared-copies", type=int, default=0,
                    help="copies SHARED and PRIVATE declare and the build "
                         "allocates; 0 = the rule in declared_copies_for, "
                         "which pads past vLLM's small-batch expert bound "
                         "when the ladder crosses its id bound")
    ap.add_argument("--probe-repeats", type=int, default=PROBE_REPEATS,
                    help="repeats of the alignment probe's cells; NOT in the "
                         "run id, because the probe is never resumed (see "
                         "default_run_id) -- report.json's align_probe carries "
                         "the count")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the predictions and the cost, and "
                         "REFUSE: nothing is measured and no gate is scored")
    ap.add_argument("--self-test", default=None, choices=sorted(WORLDS),
                    help="score the gates against a planted world, off GPU")
    ap.add_argument("--plant-noise", type=float, default=0.004,
                    help="relative spread planted on every synthetic cell")
    return ap


def _device_memory(args) -> tuple[int | None, str]:
    """`(free bytes, source)` for the memory check. The ATTACHED card first.

    `--device-memory-gb` is a hypothetical and is labelled one; it is ignored
    whenever a device answers, because a plan checked against a number the
    operator typed while a real card sat underneath it is a plan checked
    against nothing.
    """
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free), "free on the attached device, asked of the driver"
    except Exception:                                     # noqa: BLE001
        pass
    if args.device_memory_gb:
        return (int(args.device_memory_gb * 1e9),
                f"HYPOTHETICAL: --device-memory-gb {args.device_memory_gb} "
                "names a card that is not attached")
    return None, "no device and no --device-memory-gb"


def _main(argv=None) -> int:
    """The body, and the one place an exit code is chosen. `main` wraps it.

    Every return is a member of `moe.bench.exit_codes`'s table: REFUSED (2)
    before anything is measured, `--dry-run` included, because a plan scores no
    gate; ERROR (4) for a planted world that came out other than registered;
    and otherwise `classify` over the scored gates with NOTHING FOLDED. There
    is deliberately no gate-softening flag (--clock-elasticity prints and
    scores nothing): a CLAIM_FAIL is returned as 1,
    which the ledger already reads as a finished result.
    """
    args = build_parser().parse_args(argv)
    if args.read is not None:
        # BEFORE the partition check, the ridge and the card: a stored pair is
        # re-read on a laptop with none of them.
        return _read_mode(args)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    block_m = args.block_m
    synthetic = args.self_test is not None
    if not 0.0 < args.duty <= 1.0:
        print(f"REFUSED: --duty {args.duty} is not in (0, 1]: a duty cycle is "
              "the fraction of wall clock the kernel is busy, 1.0 is the "
              "queue kept full and 0 is no measurement at all")
        return exit_codes.REFUSED

    gap = partition_is_total()
    if gap:
        print(f"REFUSED: the OUTCOMES partition is broken: {gap}")
        return exit_codes.REFUSED

    # THE CONTROL IS PART OF THE DESIGN, SO ITS FEASIBILITY IS A PLAN-TIME
    # REFUSAL. V6 reads the n=1 tread and a model that cannot form one has no
    # such tread at any depth; finding that out after renting a pod costs the
    # whole arm.
    why = identity_tread_refusal(cfg, block_m)
    if why:
        print(f"REFUSED: {why}")
        return exit_codes.REFUSED
    try:
        treads = ladder_treads(cfg, block_m, args.treads)
    except PrivateWeightRefusal as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    if len(treads) < MIN_TREADS:
        print(f"REFUSED: --treads {args.treads} gives {len(treads)} tread(s) "
              f"and a slope may not be quoted below {MIN_TREADS}: two points "
              "make a line with no residual, so a two-tread fit cannot notice "
              "that one of its points was wrong.")
        return exit_codes.REFUSED
    # AND V8'S OWN FLOOR, which is one tread higher. The probe's step fit has
    # STEP_FIT_COLUMNS columns, so a ladder of that many treads leaves no
    # degree of freedom: the step's standard error used to come back exactly
    # 0.0 and V8 could never FAIL (a planted 501 us step, 39x the budget, read
    # UNKNOWN). Refused here, before a card is touched, with the count named.
    if len(treads) <= STEP_FIT_COLUMNS:
        print(f"REFUSED: --treads {args.treads} gives {len(treads)} tread(s), "
              f"and V8's step fit has {STEP_FIT_COLUMNS} columns (intercept, "
              f"per-id slope, step): {len(treads)} treads leave "
              f"{len(treads) - STEP_FIT_COLUMNS} degrees of freedom, so no "
              "alignment step could be judged against its own error and V8 "
              f"could not fail. Run at least {STEP_FIT_COLUMNS + 1} treads.")
        return exit_codes.REFUSED
    # AND THE SAME FLOOR ON THE REPEATS, WHICH WAS NOT REFUSED AND IS NOW.
    # `--treads 2` cost nothing and said why; `--repeats 2` measured all 36
    # cells, then failed V0 on a floor this file already carries and latched
    # INVALID over something a flag would have fixed. The arm rotation is the
    # second reason: it is `ARMS[(i + rep) % 3]`, so below three repeats one
    # arm is never first in a triple while the plan page goes on advertising
    # that none of them is.
    if args.probe_repeats < MIN_PROBE_REPEATS:
        print(f"REFUSED: --probe-repeats {args.probe_repeats} is below "
              f"{MIN_PROBE_REPEATS}. A single pass forms no across-repeat "
              "spread, so V8 -- the gate that decides whether the ratio arms "
              "share one alignment kernel -- would print its step with no "
              "replicate beside it, and one bad pass would enter the fit "
              "unchallenged.")
        return exit_codes.REFUSED
    if args.repeats < MIN_REPEATS:
        print(f"REFUSED: --repeats {args.repeats} is below the {MIN_REPEATS} "
              "every ladder here needs. V0 scores that floor after the whole "
              "grid has been measured, so this would be an INVALID bought with "
              "the card's own minutes; and the arm rotation ARMS[(i + rep) % "
              f"{len(ARMS)}] does not reach every arm in first position below "
              f"{len(ARMS)} repeats, which the plan page claims it does.")
        return exit_codes.REFUSED

    try:
        rr = SWEEP.resolve_ridge(args, synthetic=synthetic or args.dry_run)
    except SWEEP.RidgeUnavailable as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    ridge, ridge_source, ridge_device = rr.ridge, rr.source, rr.device
    ridge_kind = getattr(rr, "source_kind", "")
    bw = SWEEP.resolve_bandwidth(args, synthetic=synthetic or args.dry_run)
    bandwidth, bw_source = bw.gbps, bw.detail

    # THE SECOND RULER COMES OUT OF THE SAME FILE AS THE FIRST OR NOT AT ALL.
    # The own-clock roof is `peak x load / reference`, so `reference` has to be
    # the clock the peak in `ridge x bandwidth` was measured at, and that holds
    # for exactly one of the three ways this run can get a ridge: the attached
    # device's own calibration.
    reference_mhz: float | None = None
    reference_grade = ""
    if synthetic or args.dry_run:
        reference_source = ("no hardware is read in this mode, so there is no "
                            "clock to scale a roof by")
    elif ridge_kind != "calibration":
        reference_source = (f"the ridge came from elsewhere "
                            f"({ridge_kind or 'unstated'}), so this card's "
                            "calibration clock does not describe the roof it "
                            "states")
    else:
        rc = roofline.reference_clock(
            ridge_device or None, family=roofline.reference_family(args.dtype))
        reference_source, reference_grade = rc.source, rc.grade
        if rc.usable_for_roof:
            reference_mhz = rc.mhz
        elif rc.mhz:
            reference_source += (
                f" [grade {rc.grade!r}, not the under-load median, so it is "
                "recorded and NOT used to scale a roof]")

    roof_tflops = ridge * bandwidth / 1e3
    roof_source = (f"ridge {ridge:.2f} Op/B x bandwidth {bandwidth:.1f} GB/s; "
                   f"ridge: {ridge_source}; bandwidth: {bw_source}")

    tokens = {n: SWEEP.tokens_for_rows(cfg, n * block_m) for n in treads}
    try:
        copies_declared, declared_reason = declared_copies_for(
            cfg, treads, block_m, args.declared_copies)
    except PrivateWeightRefusal as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    declared_by_arm = {arm: declared_experts(arm, cfg.num_experts,
                                             copies_declared) for arm in ARMS}
    census = path_census(cfg, treads, block_m, declared_by_arm)
    free_bytes, mem_source = _device_memory(args)
    mem = memory_plan(cfg, args.dtype, b, copies_declared, tokens[treads[-1]],
                      free_bytes, mem_source, copies_read=treads[-1])

    card = detect_card_slug()
    pinned = dict(SWEEP.FIXED, num_stages=args.num_stages,
                  GROUP_SIZE_M=args.group_m, BLOCK_SIZE_N=args.block_n)
    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.out or SWEEP.results_root()) / "private_weight_reference" / run_id
    csv_path = out_dir / "cells.csv"
    cache_root = out_dir / "triton-cache"

    # THE REPLICATES, refused before the card is touched. The design dict is
    # built from the SAME variables that fill the payload, so a payload key
    # and a comparison key cannot drift apart.
    if args.replicate_of and synthetic:
        print("REFUSED: --replicate-of names a measured run and --self-test "
              "plants one; a planted world has no measured replicate")
        return exit_codes.REFUSED
    try:
        replicates = load_replicates(
            args.replicate_of, card_known=(card != NO_CARD_SLUG),
            this_run_id=run_id,
            design={"experiment": "private_weight_reference", "card": card,
                    "model": args.model, "dtype": args.dtype,
                    "block_m": block_m, "pinned": pinned,
                    "treads": list(treads), "repeats": args.repeats,
                    "copies_declared": copies_declared,
                    "alpha_band": list(ALPHA_BAND), "duty": args.duty})
    except PrivateWeightRefusal as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED

    capability = SWEEP.resolve_capability(args, synthetic=synthetic or args.dry_run)
    resources, refused = SWEEP.tile_resource_plan(pinned, (block_m,), b,
                                                 capability)
    stream_ms = WEIGHTS.weight_stream_ms(cfg, args.dtype, bandwidth)

    header = plan_lines(cfg, args, block_m=block_m, treads=treads, b=b,
                        bandwidth_gbps=bandwidth, bw_source=bw_source,
                        out_dir=out_dir, pinned=pinned, run_id=run_id,
                        resources=resources, card=card,
                        git_note=git_visibility(out_dir), mem=mem,
                        tokens=tokens, ridge=ridge, alpha=args.alpha,
                        copies_declared=copies_declared,
                        declared_reason=declared_reason, census=census,
                        replicates=tuple(replicates))
    header += prediction_lines(cfg, block_m=block_m, treads=treads,
                               alpha=args.alpha, bandwidth_gbps=bandwidth,
                               bw_source=bw_source, dtype=args.dtype,
                               stream_ms=stream_ms, ridge=ridge,
                               ridge_source=ridge_source,
                               copies_declared=copies_declared,
                               census=census)
    print("\n".join(header))

    if refused:
        print("\nREFUSED: the pinned setting cannot physically run.")
        for tile, reason in sorted(refused.items()):
            print(f"  BLOCK_M={tile}: {reason}")
        return exit_codes.REFUSED

    # A LADDER WITH A KERNEL SWITCH INSIDE A RATIO ARM IS REFUSED HERE, from
    # the cited hypothesis, before a pod is rented; the probe then checks the
    # hypothesis on the pod, and V8 refuses on the measurement.
    if census.refusals:
        print("\nREFUSED: the alignment census finds a path this design "
              "cannot carry.")
        for why in census.refusals:
            print(f"  {why}")
        return exit_codes.REFUSED

    # A DESIGN THAT CANNOT REPORT ITS OWN REGISTERED ALTERNATIVE IS REFUSED
    # HERE, not measured and then voided on V4. The compute an M-tile does
    # scales with the tile height and the weight traffic an extra M-tile costs
    # does not, so the deepest tread that still sees traffic depends on the
    # world, and the shallowest world this arm registers is the one that
    # decides the depth.
    short = depth_refusal(cfg, block_m=block_m, treads=len(treads), ridge=ridge,
                          bandwidth_gbps=bandwidth, b=b)
    if short:
        print("\nREFUSED: this design cannot separate the worlds it registers.")
        print("  " + short)
        return exit_codes.REFUSED

    # A PLAN THAT WILL NOT FIT IS A REFUSAL AND NOT A WARNING. The deepest
    # tread holds `n` complete copies of the weight set, and an allocation
    # that fails halfway through a metered run costs the arm.
    if mem.fits is False:
        print("\nREFUSED: the private weight copies do not fit this card.")
        print(f"  predicted peak {mem.predicted_peak_bytes / 1e9:.2f} GB "
              f"against {mem.headroom:.0%} of "
              f"{mem.device_free_bytes / 1e9:.2f} GB "
              f"({mem.device_source})")
        print(f"  the bill is {mem.copies} DECLARED copies at "
              f"{mem.per_copy_bytes / 1e9:.4f} GB each, of which "
              f"{mem.copies_read} are read: lowering --treads only helps while "
              "it lowers the DECLARATION, which `declared_copies_for` may hold "
              "at its own floor to keep the ratio arms on one alignment "
              "kernel. Run a model with a smaller routed expert weight set, or "
              "a ladder that does not straddle the id bound (see the alignment "
              "census above).")
        return exit_codes.REFUSED

    if args.dry_run:
        secs = estimated_seconds(cfg, treads=treads, block_m=block_m,
                                 repeats=args.repeats, alpha=args.alpha,
                                 ridge=ridge, bandwidth_gbps=bandwidth, b=b,
                                 warmup_ms=args.warmup, trials=args.trials,
                                 cell_budget_ms=args.cell_budget_ms,
                                 probe_repeats=args.probe_repeats)
        print(f"\nestimated GPU time {secs:.0f} s at the model's own timings, "
              "excluding compiles and allocation; that includes the alignment "
              f"probe's {probe_seconds(treads, PROBE_LABELS, args.probe_repeats):.0f} s")
        if args.duty < 1.0:
            ladder_secs = secs - probe_seconds(treads, PROBE_LABELS, args.probe_repeats)
            print(f"WALL CLOCK at duty {args.duty:.2f}: the ladder's {ladder_secs:.0f} s "
                  f"of kernel time takes about {ladder_secs / args.duty:.0f} s, "
                  "the idle gaps between bursts being the point; the probe is "
                  "timed at full duty")
        print("NOT IN THAT FIGURE: the Triton compiles, and the private weight "
              f"build, which copies {mem.weight_bytes / 1e9:.2f} GB "
              "device-to-device once, and the five-part buffer proof's "
              f"{proof_calls(treads[-1], copies_declared)} extra fused_experts "
              "calls at the deepest tread.")
        # REFUSED (2) AND NOT DONE (0). A dry run scores no gate, prints no
        # RESULT line, and `exit_codes.classify_text` over this log raises
        # `NoGatesScored`, which that module documents as what a REFUSED log
        # looks like from there. DONE says "measured; every gate PASSED", and
        # this run measured nothing.
        print("\n".join(["", "=" * 72,
                         "REFUSED. Nothing was measured and nothing was "
                         "written.",
                         "  reason: --dry-run was given",
                         "  Everything above is arithmetic over this repo's "
                         "calibration, moe/spec.py's",
                         "  geometry and vLLM's resource model. No gate was "
                         "scored, so no RESULT line",
                         "  was printed and none of it is a result. Run "
                         "--self-test <world> for the",
                         "  planted worlds, or the bare command on the pod.",
                         "=" * 72]))
        return exit_codes.REFUSED

    if not synthetic:
        missing = SWEEP.missing_gpu_stack()
        if missing:
            print("\n" + missing)
            return exit_codes.REFUSED
        # V7 READS THE UNDER-LOAD CLOCK OF EVERY CELL, so a host whose NVML
        # bindings are absent would spend the whole sweep and then read
        # UNKNOWN at every tread, which is INVALID. Asked once, here, the way
        # the sampler itself asks, before a byte is allocated.
        unreadable = clock_sampler_refusal()
        if unreadable:
            print(f"\nREFUSED: {unreadable}")
            return exit_codes.REFUSED

    prov = PV.provenance_block(
        instrument=ladder_instrument(args.duty, synthetic=synthetic),
        ridge=ridge, ridge_source=ridge_source,
        bandwidth=bandwidth, bandwidth_source=bw_source,
        warmup_ms=args.warmup, iters=None, target_ms=args.cell_budget_ms)

    if synthetic:
        world = WORLDS[args.self_test]
        samples = planted_samples(
            world, cfg, block_m=block_m, treads=treads, repeats=args.repeats,
            alpha_shared=world.alpha, ridge=ridge, bandwidth_gbps=bandwidth,
            b=b, noise=args.plant_noise, seed=args.seed,
            copies_declared=copies_declared,
            native_switch=census.switch_tread(NATIVE))
        probe = planted_probe(world, cfg, block_m=block_m, treads=treads,
                              declared_by_arm=declared_by_arm, census=census,
                              noise=args.plant_noise, seed=args.seed,
                              weight_stream_ms=stream_ms)
        proof = planted_proof(world.proof_ok)
        weight_delta = int(mem.weight_bytes * world.weight_alloc_factor)
        high_water = int(mem.predicted_peak_bytes * world.high_water_factor)
        roof_tflops *= world.roof_factor
        roof_source += f"; PLANTED x {world.roof_factor} by --self-test"
        print(f"\nSELF TEST: cells GENERATED from the study's traffic model in "
              f"the {world.name!r} world at a planted spread of "
              f"{args.plant_noise:.2%}. Nothing here was measured.")
        print("The gates below are being run against a world we constructed, "
              "which tests the gates and not the hardware.")
        print(f"  registered world: {world.why}")
    else:
        world = None
        out_dir.mkdir(parents=True, exist_ok=True)
        wrong_card = device_guard(out_dir, device_identity())
        if wrong_card:
            print(f"\nREFUSED: {wrong_card}")
            return exit_codes.REFUSED
        try:
            store = Store(csv_path, CSV_FIELDS + PROVENANCE_COLUMNS)
        except SchemaCollision as exc:
            print(f"\nREFUSED: {exc}")
            return exit_codes.REFUSED
        samples, proof, weight_delta, high_water, probe = run_sweep(
            args, cfg, block_m=block_m, treads=treads, pinned=pinned,
            csv_path=csv_path, cache_root=cache_root, store=store, prov=prov,
            dtype=args.dtype, copies_declared=copies_declared, census=census,
            stream_ms=stream_ms)

    clock_elasticity = None
    if args.clock_elasticity is not None:
        vals = args.clock_elasticity
        if len(vals) not in (1, 3):
            print("REFUSED: --clock-elasticity takes ETA or ETA LO HI, got "
                  f"{len(vals)} number(s)")
            return exit_codes.REFUSED
        eta, lo, hi = (vals[0], vals[0], vals[0]) if len(vals) == 1 else vals
        try:
            clock_elasticity = ClockElasticity(eta, lo, hi,
                                               args.clock_elasticity_source)
        except Unmeasurable as exc:
            print(f"REFUSED: {exc}")
            return exit_codes.REFUSED
    if synthetic:
        world_eta = WORLDS[args.self_test].planted_eta
        if world_eta is not None and clock_elasticity is None:
            clock_elasticity = ClockElasticity(world_eta, world_eta, world_eta,
                                               "PLANTED by --self-test")
        elif world_eta is None and clock_elasticity is not None:
            print("REFUSED: a planted world's registration is about its own "
                  f"planted eta, and {args.self_test!r} plants none; the "
                  "clock-split-elastic world is the one that takes one")
            return exit_codes.REFUSED
    report = analyse(
        samples, cfg, block_m=block_m, treads=treads, repeats=args.repeats,
        alpha=args.alpha, dtype=args.dtype, b=b, bandwidth_gbps=bandwidth,
        bandwidth_source=bw_source, ridge=ridge, ridge_source=ridge_source,
        roof_tflops=roof_tflops, roof_source=roof_source,
        reference_mhz=reference_mhz, reference_grade=reference_grade,
        reference_source=reference_source, mem=mem, proof=proof,
        weight_delta_bytes=weight_delta, high_water_bytes=high_water,
        draws=args.draws, seed=args.seed, header=header, card=card,
        synthetic=synthetic, model_name=args.model, pinned=pinned,
        prov=_observed_iters(prov, samples), probe=probe, census=census,
        copies_declared=copies_declared, clock_elasticity=clock_elasticity,
        run_id=run_id, session_tag=args.session_tag,
        replicates=tuple(replicates))

    print("\n".join(report.lines[len(header):]))
    print(_iters_line(samples))
    if not synthetic:
        (out_dir / "report.txt").write_text(report.text())
        (out_dir / "report.json").write_text(json.dumps(report.payload, indent=2))
        print(f"cells    {csv_path}")
        print(f"report   {out_dir / 'report.txt'}")
        print(f"json     {out_dir / 'report.json'}")

    # THE PLANTED WORLD IS CHECKED AGAINST ITS REGISTRATION, and a mismatch is
    # ERROR rather than any code in the gate table. A self-test that came out
    # differently from the world it planted has not produced a result about
    # anything -- the apparatus is broken -- and INVALID or CLAIM_FAIL would
    # both invite a reader to interpret it.
    if world is not None:
        bad = world.check(report)
        for line in bad:
            print(f"SELF-TEST MISMATCH  {line}")
        if bad:
            print(f"the planted world {world.name!r} did not return its "
                  f"registered verdicts ({len(bad)} of {len(world.expect)} "
                  "gates). This is a defect in the gates or in the model that "
                  "generates the cells, not a finding about hardware; nothing "
                  "here may be read as a result.")
            return exit_codes.ERROR
        print(f"SELF-TEST OK  all {len(world.expect)} registered verdicts in "
              f"the {world.name!r} world came back as registered")

    rc = exit_codes.classify(g.scored() for g in report.gates)
    print(f"exit     {exit_codes.describe(rc)}")
    if rc == exit_codes.CLAIM_FAIL:
        print("         a claim that did not pass is a RESULT and the arm is "
              "FINISHED, not broken.")
    return rc


def _read_mode(args) -> int:
    """READ MODE: this run IS the stored report at --read; --replicate-of
    names the others; nothing is measured and nothing is written. The plan
    header's identity lines are rendered from the payload (the one document
    the writer serialised), every stored gate is re-rendered as it was scored,
    and C1 alone is rebuilt through `gate_c1_ratio` with the CrossRun, so the
    joint rule has one home. The exit is `classify` over that set, which is
    what the pair's page would have exited with."""
    path = Path(args.read)
    if path.is_dir():
        path = path / "report.json"
    if not args.replicate_of:
        print(f"REFUSED: --read {path} names one stored report and nothing to "
              "read it against; give --replicate-of")
        return exit_codes.REFUSED
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"REFUSED: --read {path}: {exc}")
        return exit_codes.REFUSED
    if not isinstance(payload, dict) or payload.get("experiment") != "private_weight_reference":
        print(f"REFUSED: --read {path} is not a private_weight_reference report")
        return exit_codes.REFUSED
    if payload.get("synthetic"):
        print(f"REFUSED: --read {path} is a planted (--self-test) report")
        return exit_codes.REFUSED
    try:
        this = run_reading(payload, path)
        replicates = load_replicates(
            args.replicate_of, card_known=True, this=this,
            design={k: payload.get(k, DESIGN_KEY_DEFAULTS.get(k))
                    for k in DESIGN_KEYS})
    except PrivateWeightRefusal as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    cross = cross_run([this, *replicates])
    prov = payload.get("provenance") or {}
    print(f"READ MODE: nothing measured, nothing written; gates re-rendered "
          f"from {path}")
    print(f"experiment  private_weight_reference / {this.name}")
    for key in ("card", "model", "dtype", "block_m", "pinned", "treads",
                "repeats", "copies_declared", "duty"):
        print(f"{key:<12}{payload.get(key, DESIGN_KEY_DEFAULTS.get(key))}")
    print(f"session     {payload.get('session_tag') or '(unrecorded)'}")
    print(f"measured    {prov.get('utc') or 'utc unrecorded'} on "
          f"{prov.get('hostname') or 'an unrecorded host'}, tree "
          f"{prov.get('git_sha') or 'unrecorded'}")
    print("\n".join(cross.lines()))
    print()
    gates = []
    for d in payload.get("gates") or []:
        g = Gate.from_dict(d)
        if g.tag == "C1":
            g = gate_c1_ratio(this.ratio, this.interval,
                              payload.get("ratio_draws") or 0,
                              corrected=payload.get("ratio_corrected"),
                              clock=None, cross=cross)
        gates.append(g)
        print("\n".join(g.render()))
        print()
    if not gates:
        print("REFUSED: the stored report carries no gates")
        return exit_codes.REFUSED
    rc = exit_codes.classify(g.scored() for g in gates)
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


def clock_sampler_refusal() -> str:
    """"" when the under-load clock sampler can read this device, else why not.

    The same probe `BackgroundClockSampler` makes at the start of every timed
    region; made once here so its failure is a plan-time refusal and not an
    INVALID after the sweep.
    """
    from moe.bench import timing

    try:
        timing.nvml_clock_reader(None)
    except timing.ClockSourceUnavailable as exc:
        return (f"the under-load clock cannot be read on this host ({exc}); "
                "V7 compares the two ratio arms' clocks at every tread and "
                "would read UNKNOWN throughout, which is INVALID after the "
                "whole sweep is paid for")
    return ""


def _observed_iters(prov, samples):
    """The provenance block with the iteration count the cells were actually
    timed at: the median ITERATIONS PER TRIAL, which below full duty is the
    duty timer's kept calls per trial (`time_cell`). Nothing timed means
    nothing recorded: a planted world carries iters=0 on every row, so there
    is no median to take and the block keeps its None and its reason."""
    counts = sorted(s.iters for s in samples if s.usable and s.iters > 0)
    if not counts:
        return prov
    missing = {k: v for k, v in prov.missing.items() if k != "iters"}
    return replace(prov, iters=int(statistics.median(counts)), missing=missing)


def _iters_line(samples) -> str:
    counts = sorted(s.iters for s in samples if s.usable and s.iters > 0)
    if not counts:
        return ("iterations per trial: none recorded (nothing was timed; a "
                "planted world's cells carry iters=0)")
    head = (f"iterations per trial: median {int(statistics.median(counts))} "
            f"over {len(counts)} timed cells, range {counts[0]}-{counts[-1]}. ")
    duty = duty_of(samples)
    if duty >= 1.0:
        return head + "Sized per cell by the instrument from --cell-budget-ms."
    return head + (
        f"At duty {duty:.2f} an iteration is a KEPT call of the duty timer: "
        "bursts x (calls per burst - 1) per trial, the first call of every "
        f"burst discarded, the burst count from --cell-budget-ms / "
        f"{DUTY_BURST_MS:.0f} ms and the calls per burst from a "
        f"{DUTY_SIZING_MS:.0f} ms full-duty reading of the same call.")


def main(argv=None) -> int:
    """AN UNPLANNED CRASH IS ERROR (4), which is the only retryable code.

    Left to propagate, an unexpected exception exits the interpreter ONE, and
    ONE is CLAIM_FAIL, which the shared table defines as a RESULT: the driver
    would file the arm as finished, skip it on every resume and exit the
    session 0 over an arm that never measured. A torch OOM, a truncated report
    or a drifted import would be published as one of this experiment's
    registered outcomes.

    A REFUSAL FROM THE INSTRUMENT IS NOT A CRASH. `timing.TimingRefused` is
    re-raised by the sweep rather than filed per cell, and it is REFUSED (2)
    with the remedy in its message, not ERROR (4).
    """
    try:
        return _main(argv)
    except PrivateWeightRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED
    except SystemExit as exc:
        # `SWEEP.resolve_bandwidth` refuses through a `SystemExit` carrying
        # `code = REFUSED`, and `ladder_rows`-style refusals in the sibling do
        # the same with a string. A string payload is a refusal with its remedy
        # in the message, not an exit code.
        if isinstance(exc.code, str):
            print(f"REFUSED: {exc.code}", file=sys.stderr)
            return exit_codes.REFUSED
        return int(exc.code or 0)
    except Exception as exc:                              # noqa: BLE001
        try:
            from moe.bench import timing
            refused = isinstance(exc, timing.TimingRefused)
        except ImportError:
            refused = False
        if refused:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return exit_codes.REFUSED
        traceback.print_exc()
        print("ERROR: private_weight_reference crashed before it could reach a "
              "verdict. This is the apparatus failing, not a claim failing, so "
              f"it exits {exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: "
              "the traceback above is the thing to fix, and the arm may be "
              "re-run.", file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
