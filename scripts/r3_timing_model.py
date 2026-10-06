#!/usr/bin/env python
"""R3's call time, per card, from the kernel's own launch order: LOWM-1/2.

    python scripts/r3_timing_model.py <timed session dir or run dir> [...]
        [--counters DIR] [--bytes counted|groupmodel|borrowed:DIR] [--k-w 0.5]
        [--out fit.json]

STATUS (2026-09-26): NOT FINAL. This is the best current per-card model of how
R3's call time moves with the tread n and GROUP_SIZE_M G at ONE locked clock, on
cards that have their own counter pages. It is what a judge picked on
2026-09-26 after refitting three candidate models (mechanistic, phenomenological
and constraints) on one footing; this docstring carries that report's spec and
falsifiers, and tests/test_r3_timing_model.py carries its pinned numbers. It
fails on the H100 at G=1 (the SHARED slope 2-6 there comes out 0.8207 against
0.9071 measured, about 10% low, in this and every other form the judge ran), it
carries one fitted study constant (k_w, the window scale), and it has no clock
term that any fit identifies. Every number it prints is labelled with its card.
Any one of these would falsify it:

  - P3's odd-n excess of G=2 over G=4 collapsing (below 0.10 ms) or trending
    with n;
  - P2's floor moving with G beyond its band;
  - P1's two eta readings (from the G=4 slope and from the G=2 low steps)
    disagreeing;
  - P6 showing achieved occupancy near the occupancy limit AND cycles per CTA
    k-step far from 354;
  - P8 (a timed-regime counter capture on the H100 at G=1) reading ncu's
    cold-L2 bytes, not the bytes the fit needs;
  - a new card where the fitted bw and PRIVATE's per-byte line R_P disagree
    by more than 2% (gate T2).

WHAT IT FITS. One GEMM of one fused_experts call is a grid of CTAs that vLLM
v0.27.1's `fused_moe_kernel` launches in pid order. Each CTA either waits on
the SM (a floor: S_g k-steps of c ns each, shared by the SMs) or on DRAM (the
new bytes the CTAs in flight around it ask for, at bw). Per card, per cell
(arm, G, n), per GEMM g in {w1, w2}, with BLOCK 32x64x64 in bf16 on mixtral's
shapes (w1: K 4096, N 28672, 448 N-tiles, S 64 k-steps; w2: K 14336, N 4096,
64 N-tiles, S 224):

  1. Grid rows R = ceil((256 n + D x 31) / 32), D the page's
     `experts_declared` (8 NATIVE, 72 SHARED and PRIVATE); 8 n rows are live.
  2. Launch order, pid in [0, R x npn): gid = pid // (G npn), first = G gid,
     size = min(R - first, G), pid_m = first + (pid mod G npn) mod size,
     pid_n = (pid mod G npn) // size. Rows pid_m >= 8 n are dead and dropped.
  3. Slab key: (pid_m // n, pid_n) for SHARED and NATIVE (expert e owns rows
     [e n, e n + n)); (pid_m, pid_n) for PRIVATE (every M-tile has its own
     copy). L_i = 1 on the first live CTA of each (gid, key): that CTA pulls a
     weight slab from DRAM, the rest of its group finds it in L2.
  4. r_i = L_i s_g + a_g / npn_g, with slab s_g = 64 K 2 B and A-row
     a_g = 32 K 2 B read once per M-tile and shared by its npn CTAs. The byte
     source sets sigma_g = (the source's DRAM read bytes of g) / sum_i r_i, and
     b_i = sigma_g r_i + o, o = 4096 B the CTA's output tile.
  5. b_bar_i = the mean of b over the box [i - floor(w/2), i + w - floor(w/2))
     clipped to the grid, w = floor(k_w x SMs x occ_g), occ_g the recorded
     CTAs per SM (the least of the page's `launch__occupancy_limit_*`).
  6. t_i = smax_p(q_g S'_g c / SMs, b_bar_i / bw), q_g = ceil(N/SMs) / (N/SMs)
     for N live CTAs (the last wave's idle SMs), S'_g = S_g + PHI_g the CTA's
     k-steps plus its fixed cost (PHI_w1 1.512, PHI_w2 2.846 k-steps, MEASURED on
     the GH200's counter pages, see CTA_FIXED_KSTEPS; the partial-wave lifetime
     and the dead CTAs' hidden lifetime use S'_g too; `--no-cta-fixed` is S'_g =
     S_g, every pin before 2026-09-29), smax_p(f, m) = (f^p + m^p)^(1/p)
     with p = P_KNEE = 14, a fitted study constant (2026-09-27: until then a
     hard max, which priced a CTA whose DRAM time sits near its floor at the
     larger of the two; see P_KNEE). A grid under two waves prices its last
     partial wave of k CTAs at no less than its co-residency lifetime (CORES,
     2026-09-30): max(ceil(k / SMs) S'_g c, the wave's bytes / min(bw, rho k)),
     rho = CORES_RHO_GBPS; `--no-cores` is the lifetime S'_g c occ_g before it.
  7. T = T0 + sum_g sum_i t_i + n B_other / bw, B_other = 25,165,824 B per
     tread (silu_and_mul reads [256, 2F] and writes [256, F], moe_sum reads
     [256, H] and writes [128, H], bf16), plus for NATIVE s_small when the
     page's path census says small-batch and s_block when it says block-scan.
  8. Plus, per GEMM, the dead CTAs' dispatch past one effective lifetime,
     max(0, (R - 8 n) npn_g x DEAD_CTA_NS - k_w occ_g S'_g c), DEAD_CTA_NS = 1.333
     ns MEASURED on the 2026-09-27 GH200's 8x7B counter pages (see DEAD). Fixed
     per arm on one model's pages, so no single-card fit sees it; at 64 experts
     x 9 copies it is 56 us a SHARED or PRIVATE call (2026-09-29).
  9. Under `--gemm-const` only (rental 3, 2026-10-05; off by default): plus, per
     GEMM, (g / C_CYC + b S'_g) k-steps of c, the per-GEMM constant's cycle form
     Z = g + b u measured on the CAL floor captures (see GEMM_CONST).

Five parameters are fitted per card by this tool, never pooled across cards:
T0 (ms), c (ns per CTA k-step per SM), bw (GB/s), s_small and s_block (ms).
They are not the model's only fitted numbers. k_w = 1/2 is a study-level FITTED
constant (the rms optimum sits at 0.40 to 0.50 on all three cards, the judge's
scan); the k_w = 1 fit, the window the occupancy limit derives, is always
printed beside it and is worse on every card so far. P_KNEE, DEAD_CTA_NS,
CTA_FIXED_KSTEPS (from c 344.1, F_w1 and F_w2) and CORES_RHO_GBPS are fitted on
counter or timed pages too (each one's comment below): seven study-level
fitted numbers beside the five. The byte model that supplies predicted bytes
fits ten more per card (scripts/wave_split_bytes.py, MIX view). Every one, its
value and its pages: docs/paper/T2_parameters.md
(`python scripts/paper/t2_parameters.py`).

WHAT IS DERIVED AND WHAT IS NOT. The schedule (steps 1 to 4) is derived from
the kernel's pid mapping and is card-free: its lead count equals
`dram_counter_route.group_reads(8, n, G) x 8 x npn` at every G and n (the
tests hold it there), and R x npn equals every counter cell's `grid_size` on
the GH200, H100 and A100 pages. sigma_g is the ONE place measured bytes enter:
counted bytes are ncu's, at the ncu base clock with a cold L2, not the timed
regime's. q_g and B_other are arithmetic. c, bw, T0 and the two offsets are
fitted, and so is k_w.

INPUTS.
  timed pages   every R3 page (`private_weight_reference`, found by
                `per_tile_model_fit.discover`) under the positional dirs: from
                its report.json the `treads_table` (arm, tiles, ms_p50,
                sm_clock_load_mhz, experts_declared), the `path_census` rows
                (arm, n, path), `pinned` (GROUP_SIZE_M; BLOCK_N and BLOCK_K
                must be 64, or the run refuses) and the page's label
                (`per_tile_model_fit._label`). Only VALID pages are fitted;
                the rest are listed. A cell is the MEDIAN over the VALID pages
                at its G (the H200's seeds).
  counter pages `r3c-g{G}.json` under --counters (default: found under the
                positional dirs): per GEMM `dram_bytes_read` and `grid_size`,
                the recorded occupancy limits, `card.sm_count`. The page's
                own stored gates are printed, not re-scored, except V10,
                the lock gate, which is recomputed on read (scripts/
                lock_gate.py; the stored verdict is printed beside it,
                labelled stale): this tool reads only bytes, grid and
                occupancy.
  --bytes       `counted` (default, the card's own counter pages), `groupmodel`
                (sigma = 1: the schedule's own derived bytes) or
                `borrowed:DIR` (another card's counter pages). The source is
                printed on every cell. With no counter page of its own a card's
                SM count is read from its own calibration ruler
                (`observed.sm_count`) and its occupancy is ASSUMED, labelled,
                by its architecture: the compute capability its timed pages'
                own `fused_moe_kernel.ptx` was compiled for (`.target sm_90a`
                is 9.0). A borrowed card of that capability lends its recorded
                occupancy; otherwise capability 9.0 takes 5 w1 / 4 w2, what
                both Hopper counter cards recorded on every cell. Occupancy is
                architecture-specific (the A100's pages record 4 / 4), so a
                borrow from another architecture lends bytes and nothing else,
                and a card of any other capability with no same-architecture
                borrow refuses. (Review, 2026-09-26: the H200 borrowing the
                A100's bytes had taken its 4 / 4 too, which shrank the H200's
                w1 window from 330 CTAs to 264.)
  n = 5         no counter cell. Its bytes are INTERPOLATED: at G=1 and for
                PRIVATE the mean of n=4 and n=6; otherwise the group model
                (`group_reads x W_g + n x operand`) times the median
                counted/model ratio at n in {2, 3, 4, 6}. n = 5 is scored,
                never fitted.

REFUSALS. Pages of more than one card (slug or device); locked pages (every
cell at one clock) whose clocks differ, or locked and unlocked pages together,
or more than one duty, since the model has no clock term; counted bytes from
another card; counter pages of more than one card (slug or uuid), whether
under --counters, under a borrowed DIR or gathered across the inputs, even at
Gs that do not overlap (review, 2026-09-26: GH200 pages at G=1, 2, 4 beside
H100 pages at G=16, 64 had fitted as one card's COUNTED bytes); a card with no
counter page of its own whose architecture its pages do not show, or whose
occupancy nothing gives (above); a tile, model or dtype other than the
constants above; a G with no counter page under counted or borrowed bytes; a
cell whose pages disagree on its path or declaration.

THE FIT. `per_tile_model_fit.bounded_lsq` on relative residuals over the
fitted treads (n in 1, 2, 3, 4, 6), bounds T0 [-0.3, 0.5] ms, c [50, 500] ns,
bw [500, 9000] GB/s, s [-0.1, 0.1] ms, from (0.05, 206, 0.95 x the pages'
`bandwidth_gbps`, 0, 0) and three perturbed starts; identification by
`per_tile_model_fit.invisible_directions`, the repository's own rule. Then:
leave-one-G-out (LOGO), the F1 to F5 table, R_P (the OLS slope of PRIVATE's ms
against its read bytes), rho*_g = S_g c bw / (SMs s_g), the k_w = 1 fit, and
the registered predictions P1 to P8, computed from the fitted parameters.

THE REGISTERED PREDICTIONS, registered for the next GH200 run and printed for
every card from its own fit. An unmeasured cell's sigma is imputed from the
byte source's counted cells: SHARED and NATIVE take the median sigma at the
measured Gs either side (n=1 apart from n >= 2), PRIVATE the nearest measured
G's (by log2; a tie takes the smaller G, as the judge's reference does, so
G=8 reads G=4's); a prediction no measured G brackets prints n/a and the rest
still print. c at another clock f is c (clock / f)^eta.
  P1  a held lock at 1410, 1500 and 1605 MHz: the G=4 SHARED slope 2-6, and
      the G=2 increments at 1410, at eta 1 (a fixed cycle count) and at eta
      0.35 (what the GH200's unlocked pages read). The eta the G=4 slope gives
      and the eta the G=2 low steps give must agree.
  P2  G=8 and G=32 at the fit clock. The band holds the per-tread increments
      from n=2 to n=6 and their slope 2-6. The n=1 to 2 step is a separate
      point prediction and is not held to the band: on the GH200 it is 0.315
      ms against a band of [0.538, 0.556]. The judge stated that band and
      its reason (the measured floor falls about 1% with G) but gave no rule
      for it. [0.98, 1.01] x the model's slope, rounded outward to 3
      decimals (1% of noise either side, 1% more below for the fall), is
      THIS TOOL'S READING of the judge's band: it reproduces [0.538, 0.556],
      and it is not the judge's stated rule. Beside G=8 it prints PRIVATE's
      ladder at G=8 (n=1 to 6), which the judge's reference printed too. It
      is a point prediction with no band of its own, and it is the one
      output that uses PRIVATE's imputation rule.
  P3  G=2 at n=7 to 10, and its odd-n excess over G=4: at least 0.10 ms and
      flat in n.
  P4  G=4 n=8 to 9, where q_w2 drops (1.031 to 1.003 at 132 SMs).
  P5  G=3: flat when rho*_w1 exceeds the largest slab-fetching share a full
      G=3 group has (2/3), a period-3 ripple when it is below. FALSIFIED on
      the 2026-09-27 GH200 (rho* 0.690, registered flat; its steps n=2..8 read
      0.491 0.576 0.550 | 0.516 0.580 0.560). The hard max hid the fetch whenever
      the floor exceeded it; the knee (P_KNEE) prices the near-balance cells,
      so the rule is soft and the ladder the model prints replaces it.
  P6  the floor in cycles per CTA k-step, beside the census PTX's
      shared-memory cycles when the session carries its PTX.
  P7  on a LOCKED fit only, every unlocked page of the same card among the
      inputs (INVALID ones too, labelled, NEVER fitted): c refitted on its
      floor-bound cells (SHARED and NATIVE at G >= 2, n >= 2), the rest held,
      and read as an in-kernel clock if the cycle count is fixed (the GH200's
      two such pages: 1782 and 1787 MHz against NVML's 1935). That reading
      is an INTERPRETATION, the judge's: it holds only if the floor is a
      fixed cycle count, and every P7 line says so.
  P8  where T3 trips: the w2 bytes the fit without that G needs at each SHARED
      tread, against the byte source's (the H100 at G=1: 1.32 to 4.84 W_w2
      against the counted 1.34 to 3.57). When no multiple of the source's w2
      bytes in [0.2, 3.0] meets the measured time (a cell the floor alone
      already overshoots, or one beyond 3x), the needed bytes are n/a (None
      in the JSON), with the reason, never the range's endpoint.

THE GATES.
  T1  every parameter identified and none at a bound.
  T2  |bw / R_P - 1| <= 2%: DRAM's rate in the fit is PRIVATE's own per-byte
      rate. It failed at +7.47% on the H200 with group-model bytes, which is
      how the model flags bytes it is missing.
  T3  no G whose rms exceeds 3 x the median of the other Gs' rms. A trip
      prints FORM FAILS AT G and adds a labelled fit without that G.
  T4  the worst fitted cell within 2%.

WHAT IT CANNOT TELL. Which unit sets the floor: 353.8 cycles per CTA k-step
at the 1710 MHz lock on both the GH200 and the H100 matches the 352 cycles the
PTX's shared-memory traffic needs at 128 B per clock (45056 B per CTA k-step:
8 ldmatrix.x4 and 3 cp.async.cg per thread, 256 threads), and that shared
memory is the limiter is an INTERPRETATION. With the per-CTA fixed cost taken
out (CTA_FIXED_KSTEPS) the main loop reads 344 cycles per k-step on the
counters and 346 in the 8x7B fit, 2% under those 352 cycles: either the pipe
moves more than 128 B a clock or it is not the whole limiter. Why k_w = 1/2 (fitted; P6 asks
whether achieved occupancy is half the limit). The floor's clock exponent:
unlocked pages read 383 to 387 cycles at their NVML clocks against 353.8
locked, and an NVML over-read and partial clock scaling are indistinguishable
on them. The H100 at G=1. The floor's roughly -1% trend with G on the GH200
and H100 (the model is flat). The G=4 n=3 to 4 dip. The G-dependence of
NATIVE against SHARED. PRIVATE at G=64. Anything on the H200's bytes: it has
no counter page of its own. Single pages per G and no seeds on the GH200 and
the H100.

It writes nothing unless --out is given. No line it prints carries a device
uuid: a device is printed as `board` and six hex digits of the uuid's sha256
(`board()`), which still tells two boards apart; the JSON keeps the uuid.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import dram_counter_route as DCR  # noqa: E402
import lock_gate as LG  # noqa: E402
import per_tile_model_fit as PTF  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

# --------------------------------------------------------------------------
# The kernel R3 times, and what one tread adds around it. Every number here is
# derived from the model config and the pinned tile; the tests hold each to
# the value the judge's spec states.
# --------------------------------------------------------------------------

MODEL, DTYPE = "mixtral-8x7b", "bf16"
CFG = MODEL_CONFIGS[MODEL]
BYTES = dtype_bytes(DTYPE)
BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 64
E = CFG.num_experts
ARMS = ("native", "shared", "private")
GEMMS = ("w1", "w2")

#: DRAM read bytes of the whole routed weight set and of each GEMM's, and the
#: A-operand bytes one more M-tile per expert makes each GEMM read once
#: (`dram_counter_route.r3_byte_model`, which owns them).
BYTE_MODEL = DCR.r3_byte_model(CFG, DTYPE, BLOCK_M)
W = float(BYTE_MODEL["W"])


@dataclass(frozen=True)
class Gemm:
    name: str
    K: int
    N: int
    #: N-tiles per M-row (`num_pid_n`): 448 on w1, 64 on w2.
    npn: int
    #: k-steps per CTA, K / BLOCK_K: 64 on w1, 224 on w2.
    ksteps: int
    #: One CTA's weight slab, BLOCK_N x K: the bytes a slab-fetching CTA pulls.
    slab: int
    #: One M-tile's A rows, BLOCK_M x K, read once and shared by its npn CTAs.
    arow: int


def _gemm(name: str) -> Gemm:
    K, N = DCR.r3_gemm_geometry(CFG, name)
    return Gemm(name=name, K=K, N=N, npn=DCR.pid_n_count(N, BLOCK_N),
                ksteps=-(-K // BLOCK_K), slab=BLOCK_N * K * BYTES,
                arow=BLOCK_M * K * BYTES)


GEOMETRY = {g: _gemm(g) for g in GEMMS}

#: One CTA's output tile, BLOCK_M x BLOCK_N, written once. Unscaled by sigma:
#: the counter pages' bytes are reads.
OUT_TILE = BLOCK_M * BLOCK_N * BYTES

#: Rows of the sorted-id buffer per tread: E x BLOCK_M (256), which is R3's
#: 128 tokens x top_k 2.
IDS_PER_TREAD = E * BLOCK_M

#: The two small kernels one tread adds to a call besides the GEMMs, read and
#: written once at DRAM speed: silu_and_mul reads [ids, 2F] and writes
#: [ids, F]; moe_sum reads [ids, H] and writes [tokens, H]. 25,165,824 B.
B_OTHER = (IDS_PER_TREAD * 2 * CFG.intermediate_size + IDS_PER_TREAD * CFG.intermediate_size
           + IDS_PER_TREAD * CFG.hidden_size
           + IDS_PER_TREAD // CFG.top_k * CFG.hidden_size) * BYTES



def set_model(name: str) -> str:
    """Rebuild every shape above for `name` (a key of MODEL_CONFIGS) and clear
    the launch-order caches, returning the model it replaced. The schedule,
    the byte model and the extra kernels' bytes all follow the config; the
    fitted parameters (c per CTA k-step, bw) are the card's and carry over,
    which is what a cross-model prediction tests (2026-09-27: Mixtral 8x22B
    from 8x7B's fit). `build` sets the model its pages ran and restores the
    one it found."""
    global MODEL, CFG, E, BYTE_MODEL, W, GEOMETRY, IDS_PER_TREAD, B_OTHER
    if name not in MODEL_CONFIGS:
        raise Refused(f"no model {name!r}; the configs are {sorted(MODEL_CONFIGS)}")
    old = MODEL
    MODEL, CFG = name, MODEL_CONFIGS[name]
    E = CFG.num_experts
    BYTE_MODEL = DCR.r3_byte_model(CFG, DTYPE, BLOCK_M)
    W = float(BYTE_MODEL["W"])
    GEOMETRY = {g: _gemm(g) for g in GEMMS}
    IDS_PER_TREAD = E * BLOCK_M
    B_OTHER = (IDS_PER_TREAD * 2 * CFG.intermediate_size
               + IDS_PER_TREAD * CFG.intermediate_size + IDS_PER_TREAD * CFG.hidden_size
               + IDS_PER_TREAD // CFG.top_k * CFG.hidden_size) * BYTES
    for f in (schedule, window):
        f.cache_clear()
    return old


#: The counter pages' ladder: the treads with counted bytes, and the treads
#: every fit uses (n = 5 is scored, never fitted, whatever the byte source).
LADDER = (1, 2, 3, 4, 6)
INTERPOLATED_TREAD = 5

SMALL_BATCH, BLOCK_SCAN = "small-batch", "block-scan"

#: Occupancy a card with no counter page of its own (and no borrowed card of
#: its architecture) is ASSUMED to have, by the compute capability its timed
#: kernel's PTX targets. 9.0 is what both Hopper counter cards (GH200, H100)
#: recorded on every cell, w1 then w2. Only Hopper is here: the A100's pages
#: record 4 / 4, occupancy is the SM's resources against the kernel's, and
#: the study times no other architecture without that card's own pages.
ASSUMED_OCCUPANCY = {"9.0": {"w1": 5, "w2": 4}}

#: The window scale: the judge's scan put the rms optimum at 0.40 to 0.50 on
#: every card, and this is fitted, not derived (P6 asks why). The k_w = 1 fit
#: is always printed beside whichever --k-w runs.
K_W = 0.5
K_W_REFERENCE = 1.0

#: THE KNEE (2026-09-27): a CTA's floor and its DRAM stream overlap imperfectly
#: when the two are nearly equal, so its time is the soft max (f^p + m^p)^(1/p)
#: of them, not the larger. The hard max priced the cells where most CTAs sit
#: near balance fast: on the 2026-09-27 GH200 board G=3 n=2 (93% of w1's CTAs at
#: m/f = 0.97) by 2.3%, G=3 n=4 and G=2 n=3 by 0.6 to 1.2%; the 2026-09-25 GH200's
#: worst cell (G=64 n=2, m/f 0.93) and the H100's (G=16 n=2, 0.98) are the same
#: signature. Fitted free, p read 13.9, 13.9 and 12.6 on the three Hopper boards,
#: so it is ONE study constant, not a card parameter. Held at 14: rms 0.53% ->
#: 0.28% on the new board (G=3 0.95% -> 0.26%), 0.43% -> 0.41% on the old one,
#: 1.51% -> 1.33% on the H100; leave-one-G-out 0.51 -> 0.30, 0.48 -> 0.45,
#: 1.49 -> 1.27. c and bw do not move. It costs the old board's G=1 (0.44% ->
#: 0.51%), where sigma < 1 spreads cold-L2 bytes evenly. INTERPRETATION: the data
#: fix the knee's shape, not its cause (a shared-memory-bound floor and a slab
#: stream contending near balance). P_KNEE = inf is the hard max.
P_KNEE = 14.0


def smax(f: float | np.ndarray, m: np.ndarray, p: float | None = None) -> np.ndarray:
    """(f^p + m^p)^(1/p), written so no power overflows; p = inf is max(f, m).
    p defaults to the module's P_KNEE, read at call time (`--p-knee` sets it
    for one build)."""
    p = P_KNEE if p is None else p
    hi, lo = np.maximum(f, m), np.minimum(f, m)
    if np.isinf(p):
        return hi
    return hi * (1.0 + (lo / hi) ** p) ** (1.0 / p)

# --------------------------------------------------------------------------
# The fit's parameters. None of these numbers is a calibration.
# --------------------------------------------------------------------------

NAMES = ("T0", "c", "bw", "s_small", "s_block")
UNITS = {"T0": "ms", "c": "ns", "bw": "GB/s", "s_small": "ms", "s_block": "ms"}
LB = np.array([-0.3, 50.0, 500.0, -0.1, -0.1])
UB = np.array([0.5, 500.0, 9000.0, 0.1, 0.1])
#: The start: T0 0.05 ms, c 206 ns, bw 0.95 x the pages' triad rate, no offsets.
C_START_NS = 206.0
BW_START_FRACTION = 0.95
#: Three more starts, each X0 scaled by 1 + 0.1 N(0, 1) per parameter with
#: 0.005 N(0, 1) ms added to each offset, from this seed (the judge's).
EXTRA_STARTS, START_SEED = 3, 3

#: At-bound tolerance, as a fraction of the bound span (`per_tile_model_fit`'s).
BOUND_TOL = 1e-7

#: Gates.
T2_TOL = 0.02
T3_FACTOR = 3.0
T4_TOL = 0.02

#: P1: the held locks registered for the next GH200 run, and the two clock
#: exponents the floor might have: 1 (a fixed cycle count) and 0.35 (what the
#: GH200's unlocked pages read, the judge's clock check). The judge registered
#: 1600 MHz, which is off the 15 MHz grid of supported graphics clocks (the
#: H100's recorded list runs 345 to 1980 in 15 MHz steps: 1590 and 1605, no
#: 1600), and nvidia-smi and locked_r3.py refuse an unsupported lock. So 1605,
#: re-registered 2026-09-26 before any page at it exists: G=4 slope 0.5854 at
#: eta 1 and 0.5621 at eta 0.35 (1600 read 0.5872 and 0.5627). The GH200's own
#: list is read on the box before the first lock; if it lacks 1605, the nearest
#: supported clock is used and its numbers printed before the run (1590: 0.5909
#: and 0.5639).
P1_CLOCKS = (1410.0, 1500.0, 1605.0)
P1_ETAS = (1.0, 0.35)
#: P2: the unmeasured Gs, and the band around the model's flat slope: 1% of
#: noise either side and 1% more below, the fall of the measured floor with G
#: on the GH200 and the H100, rounded outward to 3 decimals. The judge stated
#: the band ([0.538, 0.556]) and its reason, not this rule: the rule is this
#: tool's reading, kept because it reproduces the judge's band.
P2_GS = (8, 32)
P2_NOISE, P2_TREND = 0.01, 0.01
#: P2's PRIVATE ladder: the G and treads the judge's reference printed it at.
P2_PRIVATE_G, P2_PRIVATE_NS = 8, range(1, 7)
#: P3: the odd-n excess of G=2 over G=4 must stay at or above this, flat in n.
P3_MIN_EXCESS_MS = 0.10
#: P6: bytes the shared-memory pipe moves per SM clock on Hopper (INTERPRETATION
#: that it is the floor's limiter; the PTX counts are arithmetic).
SMEM_BYTES_PER_CLK = 128
#: P7: the judge's mark on the in-kernel clock reading (timing judge, 2026-09-26:
#: "On P6 and P7, the INTERPRETATION is the one that holds if the floor is a
#: fixed cycle count"). Printed on every P7 line.
P7_INTERPRETATION = "INTERPRETATION: holds only if the floor is a fixed cycle count"
#: P8: the w2 byte multiplier the bisection searches, and its steps.
P8_RANGE, P8_STEPS = (0.2, 3.0), 60

FALSIFIERS = (
    "P3: the odd-n excess of G=2 over G=4 collapsing (below 0.10 ms) or trending with n",
    "P2: the floor moving with G beyond the band",
    "P1: the eta read from the G=4 slope and the eta read from the G=2 low steps disagreeing",
    "P6: achieved occupancy near the limit and cycles per CTA k-step far from 354",
    "P8: a timed-regime counter capture on the H100 at G=1 reading ncu's cold-L2 bytes",
    "T2 on a new card: bw and R_P disagreeing by more than 2%",
)

Refused = PTF.Refused


# --------------------------------------------------------------------------
# The schedule: card-free, from vLLM's pid mapping.
# --------------------------------------------------------------------------

def grid_rows(declared: int, n: int) -> int:
    """M-rows of the launch grid: the sorted ids padded per declared expert,
    ceil((256 n + D (BLOCK_M - 1)) / BLOCK_M)."""
    return -(-(IDS_PER_TREAD * n + declared * (BLOCK_M - 1)) // BLOCK_M)


def live_rows(n: int) -> int:
    return E * n


@cache
def schedule(arm: str, declared: int, G: int, n: int, gemm: str) -> tuple[np.ndarray, int]:
    """r_i (bytes) for every live CTA of one GEMM in launch order, and how
    many of them lead (fetch a weight slab). Read-only: it is cached."""
    if arm not in ARMS:
        raise ValueError(f"no arm {arm!r}")
    geo = GEOMETRY[gemm]
    R = grid_rows(declared, n)
    P = geo.npn
    pid = np.arange(R * P, dtype=np.int64)
    in_group = G * P
    gid = pid // in_group
    first = gid * G
    size = np.minimum(R - first, G)
    pid_m = first + (pid % in_group) % size
    pid_n = (pid % in_group) // size
    live = pid_m < live_rows(n)
    pid_m, pid_n, gid = pid_m[live], pid_n[live], gid[live]
    key = pid_m * P + pid_n if arm == "private" else (pid_m // n) * P + pid_n
    _, first_seen = np.unique(gid * (R * P) + key, return_index=True)
    lead = np.zeros(pid_m.size, dtype=bool)
    lead[first_seen] = True
    r = lead * float(geo.slab) + geo.arow / P
    r.setflags(write=False)
    return r, int(lead.sum())


def box_mean(x: np.ndarray, w: int) -> np.ndarray:
    """The mean of x over [i - floor(w/2), i + w - floor(w/2)), clipped."""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(x.size)
    lo = np.clip(i - w // 2, 0, x.size)
    hi = np.clip(i + (w - w // 2), 0, x.size)
    return (cs[hi] - cs[lo]) / (hi - lo)


def window_width(k_w: float, sms: int, occupancy: int) -> int:
    return max(1, int(k_w * sms * occupancy))


def wave_q(live: int, sms: int) -> float:
    """ceil(N/SMs) / (N/SMs): the floor's stretch for a last wave's idle SMs."""
    return math.ceil(live / sms) / (live / sms)


@dataclass(frozen=True)
class Window:
    """One GEMM's schedule seen through a box of w CTAs, sorted, so that the
    sum of per-CTA maxima is one search (exact: the same sum, reordered)."""
    live: int
    leads: int
    #: sum_i r_i, the schedule's own derived read bytes (sigma's denominator).
    reads: float
    sorted_mean: np.ndarray
    cumsum: np.ndarray
    #: The same box means in launch order (the tail's CTAs are the last ones).
    mean: np.ndarray
    #: The prefix sums of r_i itself in launch order (not box means): what the
    #: last k CTAs read, for the partial wave's DRAM term (CORES). None on a
    #: hand-built Window, which then reads the box means' sum instead.
    rcum: np.ndarray | None = None


@cache
def window(arm: str, declared: int, G: int, n: int, gemm: str, w: int) -> Window:
    r, leads = schedule(arm, declared, G, n, gemm)
    m = box_mean(r, w)
    s = np.sort(m)
    m.setflags(write=False)
    return Window(live=int(r.size), leads=leads, reads=float(r.sum()), sorted_mean=s,
                  cumsum=np.concatenate([[0.0], np.cumsum(s)]), mean=m,
                  rcum=np.concatenate([[0.0], np.cumsum(r)]))


#: THE PARTIAL LAST WAVE (2026-09-28). A grid of N live CTAs under 2 x SMs x
#: occ runs one full wave and then a partial one that starts together (every
#: first-wave CTA did the same work), and a CTA in that partial wave takes its
#: whole lifetime, S_g k-steps x c x occ_g, however few CTAs share its SM: the
#: floor is a per-CTA latency, not only a throughput. So those k CTAs cost
#: max(their summed time, S_g c occ_g); a single partial wave (N <= SMs x occ)
#: is that tail entire. The throughput floor gave 8x22B's w2 at n=1 (768 CTAs,
#: 1.45 waves of 528) 60 us too few: every 8x22B n=1 cell -3.8% from 8x7B's fit
#: (8x7B's n=1 is 0.97 of a wave, so no 8x7B cell carries a partial second
#: wave). With it, fitted on 8x7B only: 8x22B's registered cells 1.78 -> 1.18%
#: rms, its n=1 cells -3.8 -> +1.25% mean; 8x7B rms 0.275 -> 0.256%, LOGO 0.303
#: -> 0.284%; the 2026-09-25 GH200 0.41 -> 0.38%, the H100 1.33 -> 1.32%. Applied
#: to every tail (many-wave grids) it breaks every fit (8x7B 0.94%), as ncu shows
#: no tail cost in w1's many-wave tails. No parameter added. `--no-tail` (the
#: judge's pins) prices every CTA at throughput.
TAIL = True

#: THE PARTIAL WAVE'S CO-RESIDENCY LAW (2026-09-30), which replaces the lifetime
#: above (`--no-cores` restores it). The last k CTAs of a grid under two waves
#: start together and are dealt round-robin, CTA j of the partial wave to SM
#: j mod SMs, so an SM holds k_SM = ceil(k / SMs) of them, not occ_g. Their
#: lifetime is the larger of two terms:
#:   floor  S'_g c g(k_SM), g(k) = k: a CTA sharing its SM with k - 1 others runs
#:          a k-step in k c, the throughput share, at every k. MEASURED on the four
#:          calibration models' counter pages (8x7B incl. its floor session,
#:          8x22B, Qwen2-57B, OLMoE; 620 floor-bound NATIVE and SHARED GEMM cells,
#:          G >= 8 and the floor captures, n >= 2): with g free and one intercept
#:          per series, g_w1(1..4) = 0.76, 2.10, 3.10, 4.10 and g_w2(1..3) = 1.15,
#:          1.94, 3.21 (rms 0.349%; g(k) = k 0.383%; every CTA at occ_g, the
#:          lifetime above, 1.86%). No latency floor is seen: a CTA alone on its
#:          SM is not held near its full-occupancy lifetime. g_w1(1) under 1 is
#:          below the shared-memory limit (352 cycles), an intercept artefact, so
#:          g(k) = k is adopted and adds no constant.
#:   DRAM   the partial wave's bytes (sigma r_i + o over its k CTAs) at
#:          min(bw, CORES_RHO_GBPS x k): the wave streams at most rho per
#:          resident CTA. MEASURED, not fitted on timing: rho 11.48 GB/s from the
#:          72 calibration GEMM cells in this rule's domain (8x7B w2 n = 1, 2;
#:          8x22B w2 n = 1; every arm and G), per-GEMM model against in-kernel
#:          cycles at the 2026-09-27 fit's c and bw; leave-one-G-out 11.34 to
#:          11.63. Only 8x22B's n = 1 tail (240 CTAs, every one fetching its slab)
#:          binds it; there the old lifetime was +3.2% (NATIVE, SHARED) and +5.8%
#:          (PRIVATE) against the counters and this rule is -1.2% and +1.1%.
#: WHICH FORM IS A CHOICE THE CALIBRATION CANNOT MAKE: rho per resident CTA
#: pooled over the wave, or rho per slab-fetching CTA (`--cores-per-lead`),
#: read the same on 8x22B, where every tail CTA fetches. JetMoE-8B's SHARED
#: n = 3 w2 tail (240 CTAs, 90 fetching) separates them: pooled -4.3%, per
#: lead +18.9%, the old lifetime +27.0% against its counters. The pooled form
#: is chosen on that page, which is published and not calibration: it is
#: DIAGNOSIS, and the held-out registration (docs/registered, 2026-09-30) is
#: what tests it.
CORES = True
CORES_RHO_GBPS = 11.48
CORES_PER_LEAD = False


def cores_lifetime(win: Window, k: int, gemm: str, sigma: float, c_ns: float, bw: float,
                   sms: int) -> float:
    """The partial wave's lifetime in ms under the co-residency law (CORES): the
    last k CTAs of `win`, dealt round-robin to `sms` SMs."""
    floor = math.ceil(k / sms) * floor_ksteps(gemm) * c_ns * 1e-6
    lo = win.live - k
    if win.rcum is not None:
        raw = win.rcum[win.live] - win.rcum[lo]
    else:
        raw = float(win.mean[lo:].sum())
    if CORES_PER_LEAD:
        geo = GEOMETRY[gemm]
        dram = ((sigma * (geo.slab + geo.arow / geo.npn) + OUT_TILE) / (CORES_RHO_GBPS * 1e6)
                if _tail_has_lead(win, k, gemm) else 0.0)
    else:
        dram = (sigma * raw + OUT_TILE * k) / (min(bw, CORES_RHO_GBPS * k) * 1e6)
    return max(floor, dram)


def _tail_has_lead(win: Window, k: int, gemm: str) -> bool:
    """Whether any of the last k CTAs fetches a slab (r_i above the A-row share)."""
    if win.rcum is None:
        return False
    geo = GEOMETRY[gemm]
    r = np.diff(win.rcum[win.live - k:])
    return bool((r > geo.arow / geo.npn * 1.5).any())


@contextlib.contextmanager
def cores(on: bool):
    """Hold CORES at `on` for a block (a caller pricing cells outside `build`)."""
    global CORES
    saved, CORES = CORES, bool(on)
    try:
        yield
    finally:
        CORES = saved

#: THE DEAD-CTA TAIL (2026-09-29). The grid is sized for the declaration, ceil((ids
#: + D (BLOCK_M - 1)) / BLOCK_M) M-rows, and every CTA past the live rows loads
#: num_tokens_post_padded and exits (fused_moe.py:405-406). Those CTAs come last in
#: launch order, and dispatching them costs DEAD_CTA_NS each (GPU-wide, a dispatch
#: interval, not a per-SM cost), hidden while the last live CTAs drain: one
#: effective lifetime, k_w x occ_g x S_g x c (the window's own co-residency scale).
#: A GEMM's call pays max(0, dead_g x DEAD_CTA_NS - k_w occ_g S'_g c). MEASURED, not
#: fitted on timing: the 2026-09-27 GH200's 8x7B counter pages, in-kernel cycles,
#: SHARED minus NATIVE (identical bytes, 27,776 more dead CTAs on w1): w1 8.74 us
#: (se 0.58, 72 cells), w2 0.02 us (se 0.14; its 3,968 are hidden), so 1.333 ns =
#: (8.74 us + k_w 5 x 64 x c) / 31,360. On any one model's pages the dead count is
#: fixed per arm, so a single-card fit moves only T0, s_small and s_block (rms,
#: LOGO, c and bw exact). It is not small at 64 experts: Qwen2-57B's 576 declared
#: slots make 44,640 + 31,248 dead CTAs a call (8x7B: 31,360 + 4,480), counted
#: SHARED minus NATIVE 23.9 + 21.5 us against 30.6 + 25.1 printed here. `--no-dead`
#: (the judge's pins) drops the term.
DEAD_CTA_NS = 1.333
DEAD = True

#: THE PER-CTA FIXED COST (2026-09-29). A CTA on the floor costs S_g + PHI_g
#: k-steps of c, not S_g: the prologue (offsets, the sorted ids, num_stages - 1 = 3
#: stages of cp.async issued before the first MMA and 72 predicated-off ldgsts past
#: the last), the pipeline's fill and drain, and the epilogue (the 4 KB bf16 tile;
#: w2 also loads and multiplies the routed weights). Invisible on 8x7B's long CTAs
#: (c absorbs PHI / S: 2% of w1's 64 k-steps, 1.3% of w2's 224), 15% of OLMoE's w2
#: (16 k-steps). MEASURED, not fitted on timing: the four GH200 models' lock-1710
#: byte counter pages (8x7B 2026-09-27, 8x22B and Qwen2-57B 2026-09-28, OLMoE
#: 2026-09-29), every SHARED and NATIVE cell at G >= 8 and n = 2 to 9 (512 GEMM
#: cells), sm__cycles_elapsed.avg = a_series + ceil(N_live / SMs) (S_g c + F_g), the
#: model's own quantised floor, one c for both GEMMs as the model has: c 344.1
#: cycles, F_w1 520 (se 13), F_w2 979 (se 8), rel rms 0.33%, so PHI = F / c. The
#: floor pages (NATIVE G=64, n = 2, 3, 4, 6, lock and base clocks) read c 345.2, F
#: 455 and 937 (PHI 1.32, 2.71). Cycles, not ns: the base-clock and lock captures
#: agree to 0.3%. Mixtral's two depths alone cannot fix it (F_w2 1524 +- 236): it
#: needs the small-K models, which are therefore calibration data. `--no-cta-fixed`
#: (every pin before 2026-09-29) drops it.
CTA_FIXED_KSTEPS = {"w1": 1.512, "w2": 2.846}
CTA_FIXED = True


def floor_ksteps(gemm: str) -> float:
    """k-steps of c one CTA costs on the floor: S_g, plus PHI_g unless --no-cta-fixed."""
    return GEOMETRY[gemm].ksteps + (CTA_FIXED_KSTEPS[gemm] if CTA_FIXED else 0.0)


@contextlib.contextmanager
def cta_fixed(on: bool):
    """Hold CTA_FIXED at `on` for a block (a caller pricing cells outside `build`)."""
    global CTA_FIXED
    saved, CTA_FIXED = CTA_FIXED, bool(on)
    try:
        yield
    finally:
        CTA_FIXED = saved


#: THE PER-GEMM CONSTANT (rental 3, 2026-10-05; docs/registered/2026-10-05-rental3-zform-gh200
#: part A and -e2e-gh200 E5). The intercept Z of sm__cycles_active.avg on q over a floor
#: capture's floor-bound cells (rental 2's estimator, floor_estimator.intercept) is a
#: per-GEMM-call cost the CTA lifetime does not hold: an intercept in q, the slope free,
#: so a per-call constant and not a per-CTA one. Its CYCLE form (AFF), Z = g + b u in SM
#: cycles with u = S'_g c the floor-bound CTA's unit, is MEASURED, not fitted on timing:
#: weighted least squares over the CAL models' G = 64 base and 1710-lock floor captures
#: (8x7B 09-27 and 09-30, 8x22B, Qwen2-57B, OLMoE; 20 series, weight 1 / sigma_Z^2),
#: computed and cross-checked by scripts/scoring/rental3/register.py. The call pays
#: (g / C_CYC + b S'_g) k-steps of c per GEMM, beside dead_ms, outside smax. In cycles,
#: so it moves with the clock through c. On one model at one clock it is collinear with
#: T0: an 8x7B refit with it on gives the same residuals with T0 lower by 8x7B's own Z
#: (the tests hold both); only cross-model and cross-clock predictions move. OFF by
#: default (`--gemm-const` turns it on) until part A scores it. It is not the per-kernel
#: excess X of the rental-3 design (outside SM-active time, in T0), and not the withdrawn
#: "ncu duration offset".
GEMM_CONST_CYCLES = 2850.0
GEMM_CONST_PER_U = 0.2081
#: cycles per CTA k-step on the counter pages, CTA_FIXED_KSTEPS' c (the CAL counters)
C_CYC_PER_KSTEP = 344.1
GEMM_CONST = False


def gemm_const_ksteps(gemm: str) -> float:
    """The per-GEMM constant in k-steps of c: g / C_CYC + b S'_g (S'_g = floor_ksteps)."""
    return GEMM_CONST_CYCLES / C_CYC_PER_KSTEP + GEMM_CONST_PER_U * floor_ksteps(gemm)


def gemm_const_ms(gemm: str, c_ns: float) -> float:
    """The per-GEMM constant's time per call, in ms (0 unless --gemm-const)."""
    if not GEMM_CONST:
        return 0.0
    return gemm_const_ksteps(gemm) * c_ns * 1e-6


@contextlib.contextmanager
def gemm_const(on: bool):
    """Hold GEMM_CONST at `on` for a block (a caller pricing cells outside `build`)."""
    global GEMM_CONST
    saved, GEMM_CONST = GEMM_CONST, bool(on)
    try:
        yield
    finally:
        GEMM_CONST = saved


def dead_ctas(declared: int, n: int, gemm: str) -> int:
    """CTAs of one GEMM's grid past the live rows: they exit after one load."""
    return (grid_rows(declared, n) - live_rows(n)) * GEOMETRY[gemm].npn


def dead_ms(declared: int, n: int, gemm: str, c_ns: float, k_w: float, occ: int) -> float:
    """The dead CTAs' exposed dispatch time, in ms (0 under --no-dead)."""
    if not DEAD:
        return 0.0
    hidden = k_w * occ * floor_ksteps(gemm) * c_ns
    return max(0.0, dead_ctas(declared, n, gemm) * DEAD_CTA_NS - hidden) * 1e-6


def gemm_ms(win: Window, gemm: str, sigma: float, c_ns: float, bw: float, sms: int,
            p: float | None = None, occ: int | None = None) -> float:
    """sum_i smax_p(q S c / SMs, (sigma b_bar_i + o) / bw), in ms. At p = inf
    (the hard max) the CTAs whose window mean sits under the floor's byte
    threshold cost the floor and the rest stream, split in one search."""
    floor = wave_q(win.live, sms) * floor_ksteps(gemm) * c_ns * 1e-6 / sms
    rate = bw * 1e6
    p = P_KNEE if p is None else p
    if TAIL and occ and win.live < 2 * sms * occ:
        slots = sms * occ
        k = win.live - slots if win.live > slots else win.live
        t = smax(floor, (sigma * win.mean + OUT_TILE) / rate, p)
        if CORES:
            lifetime = cores_lifetime(win, k, gemm, sigma, c_ns, bw, sms)
        else:
            lifetime = floor_ksteps(gemm) * c_ns * 1e-6 * occ
        return float(t[:win.live - k].sum() + max(float(t[win.live - k:].sum()), lifetime))
    if np.isinf(p):
        k = int(np.searchsorted(win.sorted_mean, (floor * rate - OUT_TILE) / sigma,
                                side="left"))
        return floor * k + (sigma * (win.cumsum[-1] - win.cumsum[k])
                            + OUT_TILE * (win.live - k)) / rate
    return float(smax(floor, (sigma * win.sorted_mean + OUT_TILE) / rate, p).sum())


def cta_ms(arm: str, declared: int, G: int, n: int, gemm: str, sigma: float,
           c_ns: float, bw: float, sms: int, w: int, p: float | None = None) -> np.ndarray:
    """t_i for every live CTA, one by one: the spec's formula as written. The
    fit uses `gemm_ms`; the tests hold the two to each other."""
    r, _ = schedule(arm, declared, G, n, gemm)
    b_bar = sigma * box_mean(r, w) + OUT_TILE
    floor = wave_q(r.size, sms) * floor_ksteps(gemm) * c_ns * 1e-6 / sms
    return smax(floor, b_bar / (bw * 1e6), p)


def group_model_bytes(arm: str, G: int, n: int, gemm: str) -> float:
    """The group model's DRAM reads of one GEMM: every slab once per group its
    expert's tiles fall in (n per tile for PRIVATE), and A once per M-tile."""
    q = float(n) if arm == "private" else DCR.group_reads(E, n, G)
    return q * BYTE_MODEL[f"W_{gemm}"] + n * BYTE_MODEL[f"operand_per_tile_{gemm}"]


# --------------------------------------------------------------------------
# The PTX: the main loop's shared-memory traffic per CTA k-step.
# --------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^(\$L__BB\d+_\d+):")
_BRA_RE = re.compile(r"bra(?:\.uni)?\s+(\$L__BB\d+_\d+);")
_LDMATRIX_RE = re.compile(r"ldmatrix\.sync\.aligned\.m8n8\.x(\d)")
_CP_ASYNC_RE = re.compile(r"cp\.async\.cg\.shared\.global\s*\[[^]]*\],\s*\[[^]]*\],"
                          r"\s*(0x[0-9a-fA-F]+|\d+)")
_REQNTID_RE = re.compile(r"\.reqntid\s+(\d+)")


def ptx_main_loop(text: str) -> dict:
    """Per-thread counts in the loop that issues mma.sync (between its label
    and its back-branch): mma.sync, ldmatrix (and bytes per thread: 4 per
    8x8 b16 matrix), cp.async.cg (and their cp-size), and the CTA's threads."""
    lines = [ln.strip() for ln in text.splitlines()]
    labels = {m.group(1): i for i, ln in enumerate(lines) for m in [_LABEL_RE.match(ln)] if m}
    for i, ln in enumerate(lines):
        m = _BRA_RE.search(ln)
        if not (m and m.group(1) in labels and labels[m.group(1)] < i):
            continue
        body = lines[labels[m.group(1)]:i + 1]
        mma = sum("mma.sync" in b for b in body)
        if not mma:
            continue
        ldm = [int(x.group(1)) for b in body for x in [_LDMATRIX_RE.search(b)] if x]
        cp = [int(x.group(1), 0) for b in body for x in [_CP_ASYNC_RE.search(b)] if x]
        nt = _REQNTID_RE.search(text)
        return {"mma_sync": mma, "ldmatrix": len(ldm), "cp_async_cg": len(cp),
                "ldmatrix_bytes_per_thread": 4 * sum(ldm), "cp_async_bytes_per_thread": sum(cp),
                "threads": int(nt.group(1)) if nt else None}
    raise ValueError("no loop issuing mma.sync in this PTX")


def smem_bytes_per_kstep(counts: dict) -> int:
    """Shared-memory bytes one CTA k-step moves: every thread's ldmatrix reads
    and cp.async writes."""
    return counts["threads"] * (counts["ldmatrix_bytes_per_thread"]
                                + counts["cp_async_bytes_per_thread"])


_TARGET_RE = re.compile(r"^\s*\.target\s+sm_(\d+)(\d)a?\b", re.MULTILINE)
#: `.target` sits in the PTX's header, in its first few hundred bytes.
_PTX_HEAD_BYTES = 4096


def timed_capability(pages) -> str | None:
    """The compute capability the timed kernel was compiled for, as "9.0":
    the `.target sm_XY` of every `fused_moe_kernel.ptx` in the pages' own run
    dirs (each R3 run keeps its triton-cache). None when no page carries one.
    This is the ONE place a card with no counter page shows its architecture:
    neither report.json nor the calibration ruler records a capability."""
    caps = set()
    for p in pages:
        for ptx in sorted(p.path.rglob("fused_moe_kernel.ptx")):
            with ptx.open() as fh:
                m = _TARGET_RE.search(fh.read(_PTX_HEAD_BYTES))
            if m:
                caps.add(f"{m.group(1)}.{m.group(2)}")
    if len(caps) > 1:
        raise Refused(f"the timed pages' kernels target {len(caps)} architectures "
                      f"{sorted(caps)}; one card is one architecture")
    return next(iter(caps), None)


def find_census_ptx(pages) -> list[Path]:
    """The census's fused_moe_kernel PTX in the session above the pages."""
    out: list[Path] = []
    for p in pages:
        for anc in p.path.parents:
            root = anc / "session" / "census.profiles"
            if root.is_dir():
                out += sorted(root.rglob("fused_moe_kernel.ptx"))
                break
    return sorted(set(out))


# --------------------------------------------------------------------------
# The timed pages.
# --------------------------------------------------------------------------

@dataclass
class TimedPage:
    path: Path
    run: str
    label: str
    failed: tuple = ()
    card: str = ""
    device: str = ""
    G: int | None = None
    duty: float | None = None
    #: (arm, n) -> (ms_p50, sm_clock_load_mhz, experts_declared)
    rows: dict = field(default_factory=dict)
    #: (arm, n) -> path census's word
    paths: dict = field(default_factory=dict)
    bandwidth_gbps: float | None = None
    tile: tuple = ()
    note: str = ""

    @property
    def clocks(self) -> set:
        return {v[1] for v in self.rows.values()}

    @property
    def locked(self) -> bool:
        """Every cell at one clock: the page's own evidence of a lock."""
        return len(self.clocks) == 1 and None not in self.clocks

    def describe(self) -> str:
        why = self.label if not self.failed else f"{self.label} ({', '.join(self.failed)} not PASS)"
        ck = sorted(c for c in self.clocks if c is not None)
        clk = (f"{ck[0]:.0f} MHz locked" if self.locked else
               f"{ck[0]:.0f}-{ck[-1]:.0f} MHz" if ck else "no clock")
        return (f"{self.run:<10} G={self.G if self.G is not None else '?':<3} {why:<26} "
                f"duty {self.duty} {clk}{'  ' + self.note if self.note else ''}")


def load_timed_page(path: Path) -> TimedPage:
    report = PTF._report(path)
    label, failed = PTF._label(report, path)
    page = TimedPage(path=path, run=PTF.short_run(path), label=label, failed=failed,
                     device=PTF._card(path))
    if report is None:
        page.note = "no report.json"
        return page
    pinned = report.get("pinned") or {}
    page.card = str(report.get("card") or "")
    page.G = int(pinned["GROUP_SIZE_M"]) if pinned.get("GROUP_SIZE_M") is not None else None
    page.duty = report.get("duty")
    page.bandwidth_gbps = report.get("bandwidth_gbps")
    page.tile = (report.get("model"), report.get("dtype"), report.get("block_m"),
                 pinned.get("BLOCK_SIZE_N"), pinned.get("BLOCK_SIZE_K"))
    for r in report.get("treads_table") or []:
        if r.get("ms_p50") is None:
            continue
        n = int(r["tiles"])
        if r.get("tokens") is not None and int(r["tokens"]) * CFG.top_k != IDS_PER_TREAD * n:
            page.note = (f"{r['arm']} n={n} ran {r['tokens']} tokens, not "
                         f"{IDS_PER_TREAD * n // CFG.top_k}: another ladder")
        clk = r.get("sm_clock_load_mhz")
        page.rows[(r["arm"], n)] = (float(r["ms_p50"]), None if clk is None else float(clk),
                                    int(r["experts_declared"]))
    for row in (report.get("path_census") or {}).get("rows") or []:
        page.paths[(row[0], int(row[1]))] = row[5]
    return page


def discover_timed(inputs) -> list[TimedPage]:
    pages, seen = [], set()
    for root in inputs:
        for kind, path in PTF.discover(Path(root)):
            if kind != "R3" or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            pages.append(load_timed_page(path))
    return pages


#: How printed text names a device: the counter pages' own tag (`DCR.board`,
#: six hex digits of the uuid's sha256), so one board reads alike in both
#: tools' output. The JSON's `device` keeps the uuid, for provenance.
board = DCR.board


def admit(pages: list[TimedPage]) -> list[TimedPage]:
    """The VALID pages, after every refusal the model's footing needs."""
    use = [p for p in pages if p.label == PTF.VALID and p.rows]
    if not use:
        raise Refused("no VALID R3 page among the inputs; nothing to fit")
    cards = {(p.card, p.device) for p in use}
    if len({c for c, _d in cards}) > 1 or len({d for _c, d in cards if d}) > 1:
        raise Refused("the VALID pages come from more than one card ("
                      + "; ".join(f"{p.run}: {p.card} {board(p.device)}" for p in use)
                      + "); cards are never pooled: pass one card's pages")
    models = {p.tile[0] for p in use if p.tile}
    if len(models) > 1:
        raise Refused(f"the VALID pages ran {len(models)} models ({sorted(models)}); one fit is "
                      "one model's shapes")
    if models and next(iter(models)) != MODEL:
        set_model(next(iter(models)))
    want = (MODEL, DTYPE, BLOCK_M, BLOCK_N, BLOCK_K)
    for p in use:
        if p.tile != want:
            raise Refused(f"page {p.run} ran (model, dtype, BLOCK_M, BLOCK_N, BLOCK_K) = "
                          f"{p.tile}; this model's constants are {want}")
        if p.note:
            raise Refused(f"page {p.run}: {p.note}")
        if p.G is None:
            raise Refused(f"page {p.run} records no GROUP_SIZE_M")
    duties = {p.duty for p in use}
    if len(duties) > 1:
        raise Refused(f"the VALID pages ran at {len(duties)} duties {sorted(map(str, duties))}; "
                      "the model has no clock or duty term, so one duty per fit")
    locked = [p for p in use if p.locked]
    if locked:
        clocks = {next(iter(p.clocks)) for p in locked}
        if len(clocks) > 1 or len(locked) != len(use):
            raise Refused("the VALID pages' cell clocks differ: "
                          + "; ".join(f"{p.run} G={p.G} "
                                      + (f"locked {next(iter(p.clocks)):.0f}" if p.locked
                                         else "unlocked") for p in use)
                          + ". The model has no clock term, so a fit reads one lock: pass "
                          "the run dirs of the pages at the clock you mean")
    return use


def page_ruler_sms(pages) -> tuple[int | None, str]:
    """observed.sm_count of the calibration ruler above the pages, if any."""
    import yaml
    found = {}
    for p in pages:
        rp = PTF.find_ruler(p.path)
        if rp is None:
            continue
        doc = yaml.safe_load(rp.read_text()) or {}
        sms = (doc.get("observed") or {}).get("sm_count")
        if sms is not None:
            found[str(rp)] = int(sms)
    if not found:
        return None, "no ruler above the pages records observed.sm_count"
    if len(set(found.values())) > 1:
        raise Refused(f"the pages' rulers disagree on the SM count: {found}")
    return next(iter(found.values())), ", ".join(sorted(found))


# --------------------------------------------------------------------------
# The counter pages and the byte sources.
# --------------------------------------------------------------------------

_COUNTER_RE = re.compile(r"^r3c-g(\d+)\.json$")


def load_counter_pages(root: Path) -> dict[int, dict]:
    """{G: page} for every `r3c-g{G}.json` under root."""
    if not root.exists():
        raise Refused(f"{root}: no such directory")
    out: dict[int, dict] = {}
    files = [root] if root.is_file() else sorted(p for p in root.rglob("r3c-g*.json")
                                                  if _COUNTER_RE.match(p.name))
    for f in files:
        named = _COUNTER_RE.match(f.name)
        if named is None:
            raise Refused(f"{f}: not an r3c-g{{G}}.json counter page")
        page = json.loads(f.read_text())
        G = int(page["design"]["group_m"])
        if G != int(named.group(1)):
            raise Refused(f"{f}: design.group_m {G} is not the G its name says")
        if G in out:
            raise Refused(f"two counter pages at G={G} under {root}")
        d = page["design"]
        got = (d.get("model"), d.get("dtype"), d.get("block_m"), d.get("block_n"), d.get("block_k"))
        if got != (MODEL, DTYPE, BLOCK_M, BLOCK_N, BLOCK_K):
            raise Refused(f"{f} ran {got}; this model's constants are "
                          f"{(MODEL, DTYPE, BLOCK_M, BLOCK_N, BLOCK_K)}")
        page["_path"] = str(f)
        out[G] = page
    counter_card(out, f"under {root}")
    return out


def counter_card(pages: dict[int, dict], where: str) -> tuple[str, str | None] | None:
    """The one card, (slug, uuid), a set of counter pages was captured on, or
    a refusal. Every later reader (the own-card check, the byte source's
    label, the SM count, the occupancy) looks at one page and speaks for
    all, so the set must be one card at every G. Review, 2026-09-26: only a
    duplicate G used to refuse, so GH200 pages at G=1, 2, 4 beside H100 pages
    at G=16, 64 fitted as one card's COUNTED bytes."""
    cards = {(p["card"]["slug"], p["card"].get("uuid")) for p in pages.values()}
    if len(cards) > 1:
        raise Refused(f"the counter pages {where} come from more than one card ("
                      + "; ".join(f"G={G}: {p['card']['slug']} "
                                  f"{board(p['card'].get('uuid'))}"
                                  for G, p in sorted(pages.items()))
                      + "); cards are never pooled: pass one card's counter pages")
    return next(iter(cards), None)


def recorded_occupancy(pages: dict[int, dict]) -> dict[str, int]:
    """CTAs per SM per GEMM: the least `launch__occupancy_limit_*` on every
    cell. One value per GEMM over all the pages, or a refusal."""
    occ: dict[str, set] = {g: set() for g in GEMMS}
    for page in pages.values():
        for cell in page["cells"]:
            for g in GEMMS:
                rec = (cell.get("recorded") or {}).get(g) or {}
                lim = [v for k, v in rec.items() if k.startswith("launch__occupancy_limit_")]
                if lim:
                    occ[g].add(int(min(lim)))
    for g, v in occ.items():
        if len(v) != 1:
            raise Refused(f"the counter pages record {sorted(v) or 'no'} occupancy limits on {g}")
    return {g: next(iter(v)) for g, v in occ.items()}


def stored_validity(page: dict) -> list[str]:
    """The page's own VALIDITY gates that are not PASS, as stored, except the
    lock gate (V10, FL1): its stored verdict came from the degenerate
    t0 + cycles / f fit, so it is recomputed on read (`lock_gate`) and printed
    with the stored one labelled stale."""
    path = page.get("_path")
    return LG.reader_gate_entries(page, Path(path) if path else None, kind="VALIDITY")


@dataclass
class ByteSource:
    kind: str                 # counted | groupmodel | borrowed
    label: str
    pages: dict | None = None     # {G: counter page}
    card: str = ""
    #: The source card's `card.capability` ("9.0"), None under GROUPMODEL.
    capability: str | None = None

    def cell_bytes(self, arm: str, G: int, n: int) -> tuple[dict | None, str] | None:
        """(read bytes per GEMM, the cell's source word), or None when this
        source has no bytes for the cell. GROUPMODEL returns (None, word):
        sigma = 1."""
        if self.kind == "groupmodel":
            return None, "GROUPMODEL"
        prefix = "" if self.kind == "counted" else f"BORROWED-{self.card}/"
        page = self.pages.get(G)
        if page is None:
            return None
        cells = {(c["arm"], int(c["n"])): c for c in page["cells"]}
        if (arm, n) in cells:
            return ({g: float(cells[(arm, n)]["per_gemm"][g]["dram_bytes_read"]) for g in GEMMS},
                    prefix + "COUNTED")
        if n != INTERPOLATED_TREAD or (arm, 4) not in cells or (arm, 6) not in cells:
            return None
        out = {}
        for g in GEMMS:
            d4 = float(cells[(arm, 4)]["per_gemm"][g]["dram_bytes_read"])
            d6 = float(cells[(arm, 6)]["per_gemm"][g]["dram_bytes_read"])
            if arm == "private" or G == 1:
                out[g] = 0.5 * (d4 + d6)
            else:
                ratio = statistics.median(
                    float(cells[(arm, m)]["per_gemm"][g]["dram_bytes_read"])
                    / group_model_bytes(arm, G, m, g)
                    for m in (2, 3, 4, 6) if (arm, m) in cells)
                out[g] = ratio * group_model_bytes(arm, G, n, g)
        return out, prefix + "INTERPOLATED"


# --------------------------------------------------------------------------
# The cells and the card.
# --------------------------------------------------------------------------

@dataclass
class Cell:
    arm: str
    G: int
    n: int
    ms: float
    mhz: float | None
    path: str
    declared: int
    runs: tuple
    source: str
    #: The source's DRAM read bytes per GEMM (None under GROUPMODEL).
    reads: dict | None
    sigma: dict
    fit: bool

    @property
    def key(self) -> str:
        return f"{self.arm}/G{self.G}/n{self.n}"


@dataclass(frozen=True)
class Context:
    card: str
    device: str
    sms: int
    sms_source: str
    occupancy: dict
    occupancy_source: str
    clock_mhz: float
    locked: bool
    bandwidth_gbps: float
    byte_label: str


def make_cells(pages: list[TimedPage], source: ByteSource, ctx_sms: int,
               occupancy: dict, k_w: float) -> tuple[list[Cell], list[str]]:
    """One cell per (arm, G, n): the median over the VALID pages at its G."""
    cells, notes = [], []
    for G in sorted({p.G for p in pages}):
        gp = [p for p in pages if p.G == G]
        keys = sorted({k for p in gp for k in p.rows}, key=lambda k: (ARMS.index(k[0]), k[1]))
        for arm, n in keys:
            have = [p for p in gp if (arm, n) in p.rows]
            paths = {p.paths.get((arm, n)) for p in have}
            declared = {p.rows[(arm, n)][2] for p in have}
            if len(paths) != 1 or len(declared) != 1:
                raise Refused(f"{arm} G={G} n={n}: the pages disagree on its path {paths} "
                              f"or declaration {declared}")
            path, D = paths.pop(), declared.pop()
            if arm == "native" and path not in (SMALL_BATCH, BLOCK_SCAN):
                raise Refused(f"{arm} G={G} n={n}: path census word {path!r} is neither "
                              f"{SMALL_BATCH} nor {BLOCK_SCAN}")
            got = source.cell_bytes(arm, G, n)
            if got is None:
                notes.append(f"{arm} G={G} n={n}: no bytes in {source.label}; not scored")
                continue
            reads, word = got
            sigma = {}
            for g in GEMMS:
                win = window(arm, D, G, n, g, window_width(k_w, ctx_sms, occupancy[g]))
                sigma[g] = 1.0 if reads is None else reads[g] / win.reads
            clocks = [p.rows[(arm, n)][1] for p in have if p.rows[(arm, n)][1] is not None]
            cells.append(Cell(arm=arm, G=G, n=n,
                              ms=statistics.median(p.rows[(arm, n)][0] for p in have),
                              mhz=statistics.median(clocks) if clocks else None,
                              path=path, declared=D, runs=tuple(p.run for p in have),
                              source=word, reads=reads, sigma=sigma,
                              fit=(n in LADDER and "INTERPOLATED" not in word)))
    return cells, notes


# --------------------------------------------------------------------------
# The model and the fit.
# --------------------------------------------------------------------------

def call_ms(x, cell: Cell, ctx: Context, k_w: float, *, c_scale: float = 1.0,
            w2_scale: float = 1.0) -> float:
    """T for one cell at parameters x = (T0, c, bw, s_small, s_block)."""
    T0, c_ns, bw, s_small, s_block = (float(v) for v in x)
    t = T0 + cell.n * B_OTHER / (bw * 1e6)
    for g in GEMMS:
        win = window(cell.arm, cell.declared, cell.G, cell.n, g,
                     window_width(k_w, ctx.sms, ctx.occupancy[g]))
        sigma = cell.sigma[g] * (w2_scale if g == "w2" else 1.0)
        t += gemm_ms(win, g, sigma, c_ns * c_scale, bw, ctx.sms, occ=ctx.occupancy[g])
        t += dead_ms(cell.declared, cell.n, g, c_ns * c_scale, k_w, ctx.occupancy[g])
        t += gemm_const_ms(g, c_ns * c_scale)
    if cell.arm == "native":
        t += s_small if cell.path == SMALL_BATCH else s_block
    return t


def residuals(x, cells, ctx: Context, k_w: float) -> np.ndarray:
    return np.array([call_ms(x, c, ctx, k_w) / c.ms - 1.0 for c in cells])


def starts(ctx: Context) -> list[np.ndarray]:
    x0 = np.array([0.05, C_START_NS, BW_START_FRACTION * ctx.bandwidth_gbps, 0.0, 0.0])
    rng = np.random.default_rng(START_SEED)
    out = [np.clip(x0, LB, UB)]
    for _ in range(EXTRA_STARTS):
        out.append(np.clip(x0 * (1 + 0.1 * rng.standard_normal(5))
                           + np.array([0, 0, 0, 0.005, 0.005]) * rng.standard_normal(5), LB, UB))
    return out


@dataclass
class Fit:
    label: str
    k_w: float
    x: np.ndarray
    fitted_gs: tuple
    identified: tuple
    at_bound: tuple
    scale: np.ndarray
    invisible: np.ndarray
    cost: float

    @property
    def params(self) -> dict:
        return dict(zip(NAMES, (float(v) for v in self.x), strict=True))


def fit(cells, ctx: Context, k_w: float, label: str, *, drop_g=()) -> Fit:
    """Bounded least squares on the fitted cells (less any G in drop_g)."""
    fc = [c for c in cells if c.fit and c.G not in drop_g]
    if not fc:
        raise Refused(f"{label}: no fitted cells")

    def fun(x):
        return residuals(x, fc, ctx, k_w)

    best = None
    for x0 in starts(ctx):
        x, r, cost = PTF.bounded_lsq(fun, x0, LB, UB)
        if best is None or cost < best[2] - 1e-15:
            best = (x, r, cost)
    x, r, cost = best
    jac = PTF._jacobian(fun, x, r, LB, UB)
    span = UB - LB
    at_lower, at_upper = x <= LB + BOUND_TOL * span, x >= UB - BOUND_TOL * span
    scale, gens, _rank = PTF.invisible_directions(jac, at_lower, at_upper, LB < UB)
    identified = tuple(not PTF.moved_by(np.eye(len(x))[i], scale, gens) for i in range(len(x)))
    return Fit(label=label, k_w=k_w, x=x, fitted_gs=tuple(sorted({c.G for c in fc})),
               identified=identified, at_bound=tuple(bool(v) for v in at_lower | at_upper),
               scale=scale, invisible=gens, cost=float(cost))


def determines(f: Fit, cells, ctx: Context) -> bool:
    """False when a direction the fitted cells cannot see moves these cells."""
    if f.invisible.shape[0] == 0:
        return True

    def fun(x):
        return residuals(x, cells, ctx, f.k_w)
    jac = PTF._jacobian(fun, f.x, fun(f.x), LB, UB)
    return not PTF.moved_by(jac, f.scale, f.invisible)


def rms(v) -> float:
    v = [float(a) for a in v]
    return math.sqrt(statistics.fmean(a * a for a in v)) if v else float("nan")


def score(f: Fit, cells, ctx: Context) -> dict:
    """rms overall, by G and by arm (fitted cells), the worst fitted cell and
    the scored-only (n=5) rms, and every cell's prediction and residual."""
    res = {c.key: call_ms(f.x, c, ctx, f.k_w) / c.ms - 1.0 for c in cells}
    fitted = [c for c in cells if c.fit]
    fitted_in = [c for c in fitted if c.G in f.fitted_gs]
    worst = max(fitted_in, key=lambda c: abs(res[c.key]))
    return {"rms": rms(res[c.key] for c in fitted_in),
            "worst": res[worst.key], "worst_cell": worst.key,
            "per_G": {G: rms(res[c.key] for c in fitted if c.G == G)
                      for G in sorted({c.G for c in cells})},
            "per_arm": {a: rms(res[c.key] for c in fitted_in if c.arm == a)
                        for a in ARMS if any(c.arm == a for c in fitted_in)},
            "scored_only_rms": rms(res[c.key] for c in cells if not c.fit),
            "resid": res,
            "pred": {c.key: call_ms(f.x, c, ctx, f.k_w) for c in cells}}


NOT_IDENTIFIED = PTF.NOT_IDENTIFIED


def logo(cells, ctx: Context, k_w: float) -> dict:
    """Fit every other G, score the held-out G's fitted cells."""
    out = {}
    gs = sorted({c.G for c in cells if c.fit})
    if len(gs) < 2:
        return out
    for G in gs:
        f = fit(cells, ctx, k_w, f"LOGO without G={G}", drop_g=(G,))
        test = [c for c in cells if c.fit and c.G == G]
        if not determines(f, test, ctx):
            out[G] = NOT_IDENTIFIED
            continue
        r = residuals(f.x, test, ctx, k_w)
        i = int(np.argmax(np.abs(r)))
        out[G] = {"rms": rms(r), "worst": float(r[i]), "worst_cell": test[i].key}
    return out


def logo_mean(lg: dict) -> float | None:
    v = [d["rms"] for d in lg.values() if isinstance(d, dict)]
    return statistics.fmean(v) if v else None


# --------------------------------------------------------------------------
# PRIVATE's per-byte line, rho*, and the F1 to F5 table.
# --------------------------------------------------------------------------

def read_bytes(cell: Cell, ctx: Context, k_w: float) -> float:
    """The cell's DRAM read bytes as its byte source has them, both GEMMs."""
    return sum(cell.sigma[g] * window(cell.arm, cell.declared, cell.G, cell.n, g,
                                      window_width(k_w, ctx.sms, ctx.occupancy[g])).reads
               for g in GEMMS)


def private_line(cells, ctx: Context, k_w: float) -> dict:
    """R_P: the OLS slope of PRIVATE's ms against its read bytes over the fitted
    PRIVATE cells at every G, as a rate."""
    pc = [c for c in cells if c.arm == "private" and c.fit]
    if len({c.n for c in pc}) < 2:
        return {"R_P_gbps": None, "cells": len(pc)}
    a, b = DCR.ols([read_bytes(c, ctx, k_w) for c in pc], [c.ms for c in pc])
    return {"R_P_gbps": 1.0 / (b * 1e6), "intercept_ms": a, "cells": len(pc)}


def rho_star(x, ctx: Context) -> dict:
    """The fraction of slab-fetching CTAs in a window above which it streams:
    S_g c bw / (SMs s_g). Equal on w1 and w2 (S_g / s_g is the same)."""
    c_ns, bw = float(x[1]), float(x[2])
    return {g: floor_ksteps(g) * c_ns * bw / (ctx.sms * GEOMETRY[g].slab) for g in GEMMS}


def _slope(ys) -> float:
    return DCR.ols_slope(range(2, 2 + len(ys)), ys)


def f_table(cells, pred: dict) -> dict:
    """The five facts, measured beside model, on the cells that exist."""
    ms = {c.key: c.ms for c in cells}

    def series(src, arm, G, ns):
        keys = [f"{arm}/G{G}/n{n}" for n in ns]
        return [src[k] for k in keys] if all(k in src for k in keys) else None

    def slope(src, arm, G):
        s = series(src, arm, G, range(2, 7))
        return None if s is None else _slope(s)

    def pair(fn, *a):
        m, p = fn(ms, *a), fn(pred, *a)
        return None if m is None or p is None else {"measured": m, "model": p}

    gs = sorted({c.G for c in cells})
    F: dict = {}
    F["F1_G1_shared_slope_2_6"] = pair(slope, "shared", 1)

    def incs(src, G):
        s = series(src, "shared", G, range(1, 7))
        return None if s is None else [s[i + 1] - s[i] for i in range(5)]
    f2 = pair(incs, 2)
    if f2:
        for side in ("measured", "model"):
            i = f2[side]
            f2[f"amplitude_{side}"] = (i[1] + i[3]) / 2 - (i[2] + i[4]) / 2
    F["F2_G2_shared_increments"] = f2
    F["F3_shared_slope_2_6"] = {G: pair(slope, "shared", G) for G in gs if G >= 4}
    f3 = [G for G in gs if G >= 4 and F["F3_shared_slope_2_6"][G]]
    if len(f3) >= 2:
        a, b = F["F3_shared_slope_2_6"][f3[0]], F["F3_shared_slope_2_6"][f3[-1]]
        F["F3_trend"] = {"from_G": f3[0], "to_G": f3[-1],
                         "measured": b["measured"] / a["measured"] - 1,
                         "model": b["model"] / a["model"] - 1}

    def dip(src, G):
        s = series(src, "shared", G, range(2, 6))
        return None if s is None else (s[2] - s[1]) - 0.5 * ((s[1] - s[0]) + (s[3] - s[2]))
    F["F3_G4_dip_3_to_4"] = pair(dip, 4)
    F["F4_private_slope_2_6"] = {G: pair(slope, "private", G) for G in gs}

    def nat_vs_shared(src, G):
        a, b = slope(src, "native", G), slope(src, "shared", G)
        return None if a is None or b is None else 100 * (a / b - 1)
    F["F5_native_vs_shared_slope_pct"] = {G: pair(nat_vs_shared, G) for G in gs}
    return F


# --------------------------------------------------------------------------
# The registered predictions: unmeasured cells, sigma imputed.
# --------------------------------------------------------------------------

def sigma_table(source: ByteSource, ctx: Context, k_w: float, declared: dict) -> dict | None:
    """{(arm, g, G): {n: sigma_g}} over the source's counted cells, or None
    under GROUPMODEL (sigma = 1 everywhere)."""
    if source.kind == "groupmodel":
        return None
    tab = {}
    for G, page in source.pages.items():
        for c in page["cells"]:
            arm, n = c["arm"], int(c["n"])
            for g in GEMMS:
                win = window(arm, declared[arm], G, n, g,
                             window_width(k_w, ctx.sms, ctx.occupancy[g]))
                tab.setdefault((arm, g, G), {})[n] = float(
                    c["per_gemm"][g]["dram_bytes_read"]) / win.reads
    return tab


def imputed_sigma(tab: dict | None, arm: str, g: str, G: int, n: int) -> float:
    """SHARED and NATIVE: at n=1 the median of n=1's sigma at the measured Gs
    either side; at n >= 2 the median of every n >= 2 sigma there. PRIVATE:
    the nearest measured G (log2; on a tie the smaller G, which is what min
    over the sorted Gs gives and what the judge's reference does), its own n,
    the mean either side between ladder points, the last one beyond. P2's
    PRIVATE G=8 ladder is where the PRIVATE rule is used and tested."""
    if tab is None:
        return 1.0
    gs = sorted({k[2] for k in tab if k[0] == arm and k[1] == g})
    if not gs or not gs[0] <= G <= gs[-1]:
        raise Refused(f"no measured G brackets G={G} for {arm} {g}")
    lo, hi = max(x for x in gs if x <= G), min(x for x in gs if x >= G)
    if arm == "private":
        v = tab[(arm, g, min(gs, key=lambda x: abs(math.log2(x) - math.log2(G))))]
        if n in v:
            return v[n]
        if n > max(v):
            return v[max(v)]
        below, above = max(m for m in v if m < n), min(m for m in v if m > n)
        return 0.5 * (v[below] + v[above])
    if n == 1:
        return statistics.median([tab[(arm, g, lo)][1], tab[(arm, g, hi)][1]])
    return statistics.median([s for m, s in tab[(arm, g, lo)].items() if m >= 2]
                             + [s for m, s in tab[(arm, g, hi)].items() if m >= 2])


def predict_unmeasured(x, ctx: Context, k_w: float, tab, arm: str, G: int, n: int,
                       declared: int, *, mhz: float | None = None, eta: float = 1.0) -> float:
    """A cell nobody measured: sigma imputed, c scaled (clock / mhz)^eta."""
    cell = Cell(arm=arm, G=G, n=n, ms=float("nan"), mhz=mhz, path=BLOCK_SCAN,
                declared=declared, runs=(), source="IMPUTED", reads=None,
                sigma={g: imputed_sigma(tab, arm, g, G, n) for g in GEMMS}, fit=False)
    scale = 1.0 if mhz is None else (ctx.clock_mhz / mhz) ** eta
    return call_ms(x, cell, ctx, k_w, c_scale=scale)


def p2_band(slope: float) -> tuple[float, float]:
    lo = math.floor(slope * (1 - P2_NOISE - P2_TREND) * 1000) / 1000
    hi = math.ceil(slope * (1 + P2_NOISE) * 1000) / 1000
    return lo, hi


def slab_fraction_max(G: int, ns=range(2, 9)) -> float:
    """The largest share of SHARED's CTAs that fetch a slab inside one FULL
    GROUP_SIZE_M group (G live rows), over n >= 2: distinct experts over G.
    2/3 at G=3 (two experts in a three-row group). A window (330 w1 CTAs) sits
    inside one group's G x 448 CTAs, so this is the share a window sees; the
    grid's last, partial group is a tail and is left out."""
    return max(len({r // n for r in range(first, first + G)}) / G
               for n in ns for first in range(0, live_rows(n) - G + 1, G))


def c_only_fit(x, cells, ctx: Context, k_w: float) -> tuple[float, float, int]:
    """c refitted on the floor-bound cells (SHARED and NATIVE, G >= 2, n >= 2),
    every other parameter held: P7's reading of an unlocked page."""
    fc = [c for c in cells if c.arm in ("shared", "native") and c.G >= 2 and c.n >= 2]
    t = np.array([c.ms for c in fc])

    def fun(p):
        xx = np.array(x, dtype=float)
        xx[1] = p[0]
        return np.array([call_ms(xx, c, ctx, k_w) for c in fc]) / t - 1.0
    p, r, _cost = PTF.bounded_lsq(fun, np.array([x[1]]), LB[1:2], UB[1:2])
    return float(p[0]), rms(r), len(fc)


def w2_needed(x, cell: Cell, ctx: Context, k_w: float) -> tuple[float | None, str]:
    """P8: the multiple of the cell's w2 bytes that makes the model meet its
    measured time, by bisection. (None, the reason) when no multiple in
    P8_RANGE brackets it. Review, 2026-09-26: the range's endpoint used to
    come back, and have x 0.2 printed as if it were a solved byte count."""
    lo, hi = P8_RANGE
    if call_ms(x, cell, ctx, k_w, w2_scale=lo) >= cell.ms:
        return None, f"floor-bound: even {lo}x the w2 bytes overshoots the measured time"
    if call_ms(x, cell, ctx, k_w, w2_scale=hi) < cell.ms:
        return None, f"beyond {hi}x the w2 bytes"
    for _ in range(P8_STEPS):
        mid = 0.5 * (lo + hi)
        if call_ms(x, cell, ctx, k_w, w2_scale=mid) < cell.ms:
            lo = mid
        else:
            hi = mid
    return lo, ""


def predictions(f: Fit, cells, ctx: Context, source: ByteSource, extra_pages, t3_refits,
                ptx_cycles) -> dict:
    """P1 to P8 from the fitted parameters of this card."""
    x, k_w = f.x, f.k_w
    # Each arm's declaration off its cells; the fallbacks are the model's own
    # (NATIVE declares E, the ratio arms E x 9), not Mixtral's 8 and 72.
    native_decl = {c.declared for c in cells if c.arm == "native"}
    shared_decl = {c.declared for c in cells if c.arm == "shared"}
    private_decl = {c.declared for c in cells if c.arm == "private"}
    decl = {"native": min(native_decl) if native_decl else E,
            "shared": min(shared_decl) if shared_decl else 9 * E,
            "private": min(private_decl) if private_decl else 9 * E}
    tab = sigma_table(source, ctx, k_w, decl)
    measured_gs = sorted({c.G for c in cells})
    out: dict = {"sigma": "imputed from " + source.label if tab else "GROUPMODEL (sigma = 1)",
                 "clock_mhz": ctx.clock_mhz}

    def ladder(arm, G, ns, **kw):
        v = [predict_unmeasured(x, ctx, k_w, tab, arm, G, n, decl[arm], **kw) for n in ns]
        return {"n": list(ns), "T": v, "inc": [v[i + 1] - v[i] for i in range(len(v) - 1)]}

    by = {c.key: c for c in cells}
    check = {}
    for G in measured_gs:
        keys = [f"shared/G{G}/n{n}" for n in range(1, 7)]
        if all(k in by for k in keys):
            check[G] = [predict_unmeasured(x, ctx, k_w, tab, "shared", G, n, decl["shared"])
                        / by[k].ms - 1 for n, k in zip(range(1, 7), keys, strict=True)]
    out["imputation_check"] = check

    # Each prediction needs measured Gs either side of the G it predicts (sigma
    # is imputed from them). One that has none is recorded as unavailable, with
    # the reason, and the rest are still made.
    out["unavailable"] = {}

    def attempt(tag, make):
        try:
            out[tag] = make()
        except Refused as exc:
            out[tag] = None
            out["unavailable"][tag] = str(exc)

    def p1():
        d = {}
        for eta in P1_ETAS:
            for mhz in P1_CLOCKS:
                s4 = ladder("shared", 4, range(1, 7), mhz=mhz, eta=eta)
                d[f"G4/f{mhz:.0f}/eta{eta}"] = {"slope_2_6": _slope(s4["T"][1:6]), **s4}
            d[f"G2/f{P1_CLOCKS[0]:.0f}/eta{eta}"] = ladder("shared", 2, range(1, 7),
                                                            mhz=P1_CLOCKS[0], eta=eta)
        return d
    attempt("P1", p1)

    def p2():
        d = {}
        for G in P2_GS:
            s = ladder("shared", G, range(1, 9))
            slope = _slope(s["T"][1:6])
            d[G] = {**s, "step_1_2": s["inc"][0], "slope_2_6": slope, "band": p2_band(slope)}
        # PRIVATE at G=8, as the judge's reference printed it: the one ladder
        # PRIVATE's imputation rule feeds. No band: a point prediction.
        pv = ladder("private", P2_PRIVATE_G, P2_PRIVATE_NS)
        d[P2_PRIVATE_G]["private"] = {**pv, "slope_2_6": _slope(pv["T"][1:6])}
        return d
    attempt("P2", p2)

    def p3():
        g2, g4 = ladder("shared", 2, range(1, 11)), ladder("shared", 4, range(1, 11))
        return {"G2": g2, "G4": g4,
                "odd_n_excess": {n: g2["T"][n - 1] - g4["T"][n - 1] for n in (3, 5, 7, 9)}}
    attempt("P3", p3)

    def p4():
        g4 = ladder("shared", 4, range(1, 11))
        return {"G4": g4, "inc_8_9": g4["inc"][7],
                "inc_others_median": statistics.median(g4["inc"][1:7] + g4["inc"][8:]),
                "q_w2": {n: wave_q(live_rows(n) * GEOMETRY["w2"].npn, ctx.sms) for n in (8, 9)}}
    attempt("P4", p4)

    def p5():
        rho, frac = rho_star(x, ctx), slab_fraction_max(3)
        return {"G3": ladder("shared", 3, range(1, 9)), "rho_star_w1": rho["w1"],
                "slab_fraction_max_G3": frac, "flat": rho["w1"] > frac}
    attempt("P5", p5)
    cyc = float(x[1]) * ctx.clock_mhz * 1e-3
    out["P6"] = {"cycles_per_cta_kstep": cyc, "ptx_smem_cycles": ptx_cycles,
                 # what a floor capture's slope over CTA k-steps reads: c (1 + PHI_g / S_g)
                 "capture_slope": {g: cyc * floor_ksteps(g) / GEOMETRY[g].ksteps for g in GEMMS},
                 "cta_fixed": CTA_FIXED}
    p7 = {"locked_fit": ctx.locked, "cycles": cyc, "pages": []}
    if ctx.locked:
        for page, pcells in extra_pages:
            c_u, r_u, k = c_only_fit(x, pcells, ctx, k_w)
            if not k:
                continue
            nvml = statistics.median(c.mhz for c in pcells
                                     if c.arm == "shared" and c.n >= 2 and c.mhz)
            p7["pages"].append({"run": page.run, "label": page.label, "G": page.G,
                                "c_ns": c_u, "rms": r_u, "cells": k, "nvml_mhz": nvml,
                                "cycles_at_nvml": c_u * nvml * 1e-3,
                                "in_kernel_mhz": cyc / (c_u * 1e-3)})
    out["P7"] = p7
    p8 = {}
    for G, rf in t3_refits.items():
        rows = []
        for c in cells:
            if c.arm == "shared" and c.G == G and c.n >= 2:
                m, word = w2_needed(rf.x, c, ctx, k_w)
                have = c.sigma["w2"] * window(c.arm, c.declared, c.G, c.n, "w2", window_width(
                    k_w, ctx.sms, ctx.occupancy["w2"])).reads / BYTE_MODEL["W_w2"]
                rows.append({"n": c.n, "source": c.source, "w2_have_Ww2": have,
                             "w2_needed_Ww2": None if m is None else have * m, "note": word})
        p8[G] = rows
    out["P8"] = p8
    return out


# --------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------

def gates(f: Fit, sc: dict, pline: dict) -> dict:
    t1 = all(f.identified) and not any(f.at_bound)
    rp = pline.get("R_P_gbps")
    t2_val = None if rp is None else float(f.x[2]) / rp - 1
    per_g = {G: v for G, v in sc["per_G"].items() if G in f.fitted_gs}
    trips = []
    if len(per_g) >= 2:
        for G, v in per_g.items():
            others = [u for H, u in per_g.items() if H != G]
            if v > T3_FACTOR * statistics.median(others):
                trips.append(G)
    return {"T1": {"pass": t1, "identified": dict(zip(NAMES, f.identified, strict=True)),
                   "at_bound": dict(zip(NAMES, f.at_bound, strict=True))},
            "T2": {"pass": t2_val is not None and abs(t2_val) <= T2_TOL,
                   "bw_over_R_P_minus_1": t2_val},
            "T3": {"pass": not trips, "fails_at_G": trips,
                   "applicable": len(per_g) >= 2},
            "T4": {"pass": abs(sc["worst"]) <= T4_TOL, "worst": sc["worst"],
                   "worst_cell": sc["worst_cell"]}}


def byte_source(args, pages, found_counters: dict | None) -> ByteSource:
    spec = args.bytes
    if spec == "counted":
        if not found_counters:
            raise Refused("no counter pages for this card (none under --counters or the "
                          "inputs); pass --counters, or --bytes groupmodel or "
                          "--bytes borrowed:DIR, which are labelled on every cell")
        # One card at every G: load_counter_pages and find_counters refused
        # anything else, so the first page speaks for the set.
        first = next(iter(found_counters.values()))["card"]
        return ByteSource("counted", "COUNTED (the card's own counter pages)", found_counters,
                          card=first["slug"], capability=first.get("capability"))
    if spec == "groupmodel":
        return ByteSource("groupmodel", "GROUPMODEL (sigma = 1, the schedule's derived bytes)")
    if spec.startswith("borrowed:"):
        pages_b = load_counter_pages(Path(spec.split(":", 1)[1]))
        if not pages_b:
            raise Refused(f"no r3c-g*.json under {spec.split(':', 1)[1]}")
        first = next(iter(pages_b.values()))["card"]
        return ByteSource("borrowed", f"BORROWED-{first['slug']} (another card's counted bytes)",
                          pages_b, card=first["slug"], capability=first.get("capability"))
    raise Refused(f"--bytes {spec!r}: counted, groupmodel or borrowed:DIR")


def assumed_occupancy(capability: str | None, source: ByteSource,
                      card: str) -> tuple[dict, str]:
    """CTAs per SM for a card with no counter page of its own, and the label.
    A borrowed card of the same architecture lends its recorded occupancy;
    otherwise the ASSUMED table by capability; otherwise a refusal. Review,
    2026-09-26: this used to take any borrowed card's occupancy, so the H200
    borrowing the A100's bytes also took the A100's 4 / 4 and its w1 window
    shrank from 330 CTAs to 264, a second input changed by a byte flag."""
    if capability is None:
        raise Refused(f"{card} has no counter page of its own and no fused_moe_kernel.ptx "
                      "under its timed pages to read its architecture from; occupancy is "
                      "architecture-specific (Hopper 5/4, the A100 4/4), so the window "
                      "cannot be sized: pass the card's own counter pages")
    where = f"capability {capability}, the timed kernel's PTX target; no counter page of its own"
    if source.kind == "borrowed" and source.capability == capability:
        return (recorded_occupancy(source.pages),
                f"ASSUMED: {source.card}'s recorded occupancy, a borrowed card of the same "
                f"architecture ({where})")
    if capability not in ASSUMED_OCCUPANCY:
        raise Refused(f"{card} (capability {capability}) has no counter page of its own and "
                      "no borrowed card of its architecture, and this tool assumes occupancy "
                      f"only for {sorted(ASSUMED_OCCUPANCY)}: pass the card's own counter "
                      "pages, or borrow from a card of the same capability")
    occ = dict(ASSUMED_OCCUPANCY[capability])
    label = (f"ASSUMED: the {occ['w1']}/{occ['w2']} both Hopper counter cards recorded "
             f"({where})")
    if source.kind == "borrowed":
        label += (f"; the borrowed {source.card} is capability {source.capability}, another "
                  "architecture, so it lends bytes only, not its occupancy")
    return occ, label


def find_counters(args, inputs) -> dict | None:
    if args.counters is not None:
        return load_counter_pages(args.counters)
    found: dict[int, dict] = {}
    for root in inputs:
        root = Path(root)
        if root.is_dir() and any(root.rglob("r3c-g*.json")):
            for G, page in load_counter_pages(root).items():
                if G in found and found[G]["_path"] != page["_path"]:
                    raise Refused(f"two counter pages at G={G} under the inputs; pass --counters")
                found[G] = page
    # Each root is one card (load_counter_pages); two roots can still be two.
    counter_card(found, "under the inputs")
    return found or None


def build(args) -> dict:
    """Everything the page prints, as one dict (the --out JSON), under the
    knee exponent `--p-knee` (restored after, so one build never leaks it)."""
    global P_KNEE, TAIL, DEAD, CTA_FIXED, CORES, CORES_PER_LEAD, GEMM_CONST
    saved, P_KNEE = P_KNEE, float(getattr(args, "p_knee", P_KNEE))
    saved_const, GEMM_CONST = GEMM_CONST, bool(getattr(args, "gemm_const", False))
    saved_tail, TAIL = TAIL, not getattr(args, "no_tail", False)
    saved_cores, CORES = CORES, not getattr(args, "no_cores", False)
    saved_lead, CORES_PER_LEAD = CORES_PER_LEAD, bool(getattr(args, "cores_per_lead", False))
    saved_dead, DEAD = DEAD, not getattr(args, "no_dead", False)
    saved_fixed, CTA_FIXED = CTA_FIXED, not getattr(args, "no_cta_fixed", False)
    saved_model = MODEL
    if not P_KNEE >= 1:
        P_KNEE, TAIL, DEAD, CTA_FIXED = saved, saved_tail, saved_dead, saved_fixed
        CORES, CORES_PER_LEAD, GEMM_CONST = saved_cores, saved_lead, saved_const
        raise Refused(f"--p-knee {args.p_knee}: the knee exponent must be at least 1 (inf is "
                      "the hard max)")
    try:
        out = _build(args)
    finally:
        P_KNEE = saved
        TAIL = saved_tail
        CORES, CORES_PER_LEAD = saved_cores, saved_lead
        DEAD = saved_dead
        CTA_FIXED = saved_fixed
        GEMM_CONST = saved_const
        if MODEL != saved_model:
            set_model(saved_model)
    return out


def _build(args) -> dict:
    k_w = float(args.k_w)
    if not k_w > 0:
        raise Refused(f"--k-w {k_w}: the window scale must be positive")
    all_pages = discover_timed(args.inputs)
    if not all_pages:
        raise Refused("no R3 (private_weight_reference) page under the inputs")
    use = admit(all_pages)
    card, device = use[0].card, next((p.device for p in use if p.device), "")
    if args.bytes.startswith("borrowed:") and args.counters is not None:
        raise Refused("--counters with --bytes borrowed:DIR: say which counter pages once")
    own = find_counters(args, args.inputs)
    if own:
        oc = next(iter(own.values()))["card"]
        if oc["slug"] != card or (device and oc.get("uuid") and oc["uuid"] != device):
            if args.bytes == "counted":
                raise Refused(f"the counter pages are {oc['slug']} {board(oc.get('uuid'))}, "
                              f"the timed pages {card} {board(device)}: counted bytes come from "
                              "the card's own pages; --bytes borrowed:DIR labels a borrow")
            own = None
    source = byte_source(args, use, own)
    ruler_sms, ruler_src = page_ruler_sms(use)
    if own:
        sms = int(next(iter(own.values()))["card"]["sm_count"])
        if ruler_sms is not None and ruler_sms != sms:
            raise Refused(f"the counter pages say {sms} SMs, the ruler {ruler_sms}")
        sms_src = "the card's counter pages (card.sm_count)"
        occ, occ_src = recorded_occupancy(own), "RECORDED on the card's counter pages"
    else:
        if ruler_sms is None:
            raise Refused(f"no SM count for {card}: no counter page of its own and "
                          f"{ruler_src}")
        sms, sms_src = ruler_sms, f"the card's ruler ({ruler_src})"
        occ, occ_src = assumed_occupancy(timed_capability(use), source, card)
    gs = sorted({p.G for p in use})
    if source.pages is not None:
        missing = [G for G in gs if G not in source.pages]
        if missing:
            raise Refused(f"no counter page at G={missing} in {source.label}; pass the timed "
                          "pages at the counter pages' Gs, or --bytes groupmodel")
    bws = sorted({float(p.bandwidth_gbps) for p in use if p.bandwidth_gbps})
    if not bws:
        raise Refused("no page records bandwidth_gbps; the fit's start needs it")
    cells, notes = make_cells(use, source, sms, occ, k_w)
    fitted = [c for c in cells if c.fit]
    if not fitted:
        raise Refused("no cell has bytes at a fitted tread")
    # The clock c is quoted in cycles at: the lock, or on unlocked pages the
    # median over the fitted SHARED cells at G >= 2, the floor-bound ladders
    # the cycle count describes (all fitted cells when there are none).
    clocks = ([c.mhz for c in fitted if c.mhz and c.arm == "shared" and c.G >= 2]
              or [c.mhz for c in fitted if c.mhz])
    if not clocks:
        raise Refused("no fitted cell records its clock; c cannot be put in cycles")
    ctx = Context(card=card, device=device, sms=sms, sms_source=sms_src, occupancy=occ,
                  occupancy_source=occ_src, clock_mhz=statistics.median(clocks),
                  locked=use[0].locked, bandwidth_gbps=statistics.median(bws),
                  byte_label=source.label)

    main = fit(cells, ctx, k_w, f"{card} k_w={k_w}")
    sc = score(main, cells, ctx)
    lg = logo(cells, ctx, k_w)
    pline = private_line(cells, ctx, k_w)
    gt = gates(main, sc, pline)
    t3_refits, t3_scores = {}, {}
    for G in gt["T3"]["fails_at_G"]:
        rf = fit(cells, ctx, k_w, f"{card} k_w={k_w} WITHOUT G={G} (T3 tripped)", drop_g=(G,))
        t3_refits[G] = rf
        rsc = score(rf, cells, ctx)
        t3_scores[G] = {"fit": rf, "score": rsc, "F": f_table(cells, rsc["pred"])}
    ref = fit(cells, ctx, K_W_REFERENCE, f"{card} k_w={K_W_REFERENCE} (reference)")
    ref_sc = score(ref, cells, ctx)

    ptx = []
    for path in find_census_ptx(use):
        try:
            counts = ptx_main_loop(path.read_text())
        except ValueError:
            continue
        b = smem_bytes_per_kstep(counts) if counts["threads"] else None
        ptx.append({"path": str(path), **counts, "smem_bytes_per_kstep": b,
                    "smem_cycles": None if b is None else b / SMEM_BYTES_PER_CLK})
    ptx_cycles = sorted({p["smem_cycles"] for p in ptx if p["smem_cycles"] is not None})

    extra = []
    if ctx.locked:
        for p in all_pages:
            if p in use or not p.rows or p.card != card or p.locked or p.G is None:
                continue
            if device and p.device and p.device != device:
                continue
            if p.tile != (MODEL, DTYPE, BLOCK_M, BLOCK_N, BLOCK_K) or p.note:
                continue
            if source.pages is not None and p.G not in source.pages:
                continue
            pc, _n = make_cells([p], source, sms, occ, k_w)
            extra.append((p, pc))
    preds = predictions(main, cells, ctx, source, extra, t3_refits, ptx_cycles)

    return {"all_pages": all_pages, "use": use, "ctx": ctx, "source": source, "cells": cells,
            "notes": notes, "k_w": k_w, "main": main, "score": sc, "logo": lg,
            "private_line": pline, "gates": gt, "F": f_table(cells, sc["pred"]),
            "rho_star": rho_star(main.x, ctx), "t3": t3_scores, "reference": ref,
            "reference_score": ref_sc, "ptx": ptx, "predictions": preds}


# --------------------------------------------------------------------------
# The page.
# --------------------------------------------------------------------------

def _pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.2f}%"


def _params_line(f: Fit, ctx: Context) -> str:
    p = f.params
    flags = []
    for name, ok, bd in zip(NAMES, f.identified, f.at_bound, strict=True):
        if not ok:
            flags.append(f"{name} NOT IDENTIFIED")
        if bd:
            flags.append(f"{name} AT A BOUND")
    return (f"T0 {p['T0']:.5f} ms, c {p['c']:.2f} ns = {p['c'] * ctx.clock_mhz * 1e-3:.1f} "
            f"cycles per CTA k-step at {ctx.clock_mhz:.0f} MHz, bw {p['bw']:.1f} GB/s, "
            f"s_small {p['s_small']:+.5f} ms, s_block {p['s_block']:+.5f} ms"
            + (f"  [{'; '.join(flags)}]" if flags
               else "  [every parameter identified, none at a bound]"))


def _fit_lines(tag: str, f: Fit, sc: dict, ctx: Context) -> list[str]:
    return [f"{tag}: {_params_line(f, ctx)}",
            f"  rms {_pct(sc['rms'])} over the fitted cells, worst {100 * sc['worst']:+.2f}% "
            f"({sc['worst_cell']}); n=5 scored only: {_pct(sc['scored_only_rms'])}",
            "  rms per G: " + " ".join(
                f"G{G} {_pct(v)}{'' if G in f.fitted_gs else ' (NOT FITTED: a prediction)'}"
                for G, v in sc["per_G"].items())
            + " | per arm: " + " ".join(f"{a} {_pct(v)}" for a, v in sc["per_arm"].items())]


def _f_lines(F: dict) -> list[str]:
    def mm(d, fmt="{:.4f}"):
        return "n/a" if not d else f"{fmt.format(d['measured'])}/{fmt.format(d['model'])}"
    out = ["F1-F5 (measured/model; slopes over n = 2 to 6 in ms per tread):"]
    out.append(f"  F1 G=1 SHARED slope {mm(F['F1_G1_shared_slope_2_6'])}")
    f2 = F["F2_G2_shared_increments"]
    if f2:
        out.append("  F2 G=2 SHARED increments n1->6 measured "
                   + " ".join(f"{v:.3f}" for v in f2["measured"]) + " | model "
                   + " ".join(f"{v:.3f}" for v in f2["model"])
                   + f" | amplitude measured {f2['amplitude_measured']:.3f} model "
                   f"{f2['amplitude_model']:.3f}")
    else:
        out.append("  F2 n/a (no G=2 SHARED ladder)")
    out.append("  F3 SHARED slope at G >= 4: " + " ".join(
        f"G{G} {mm(d)}" for G, d in F["F3_shared_slope_2_6"].items())
        + (f" | trend G{F['F3_trend']['from_G']}->G{F['F3_trend']['to_G']} measured "
           f"{100 * F['F3_trend']['measured']:+.2f}% model {100 * F['F3_trend']['model']:+.2f}%"
           if F.get("F3_trend") else "")
        + f" | G=4 dip at 3->4 {mm(F['F3_G4_dip_3_to_4'], '{:+.4f}')} ms")
    out.append("  F4 PRIVATE slope: " + " ".join(
        f"G{G} {mm(d)}" for G, d in F["F4_private_slope_2_6"].items()))
    out.append("  F5 NATIVE vs SHARED slope %: " + " ".join(
        f"G{G} {mm(d, '{:+.2f}')}" for G, d in F["F5_native_vs_shared_slope_pct"].items()))
    return out


def _ladder(d: dict) -> str:
    return "T " + " ".join(f"{v:.4f}" for v in d["T"]) + " | inc " + " ".join(
        f"{v:.3f}" for v in d["inc"])


def _prediction_lines(P: dict, card: str) -> list[str]:
    out = [f"REGISTERED PREDICTIONS, {card} (this card's fit; {P['sigma']}; ms per call; "
           "c scaled (clock / f)^eta off the fit clock)"]
    for G, v in P["imputation_check"].items():
        out.append(f"  imputation check, measured G={G} SHARED with sigma imputed: residual % "
                   + " ".join(f"{100 * r:+.2f}" for r in v))
    for tag, why in P["unavailable"].items():
        out.append(f"  {tag} n/a: {why}")
    if P["P1"] is not None:
        out += _p1_lines(P)
    if P["P2"] is not None:
        out += _p2_lines(P)
    if P["P3"] is not None:
        out += _p3_lines(P)
    if P["P4"] is not None:
        out += _p4_lines(P)
    if P["P5"] is not None:
        out += _p5_lines(P)
    out += _p678_lines(P)
    return out


def _p1_lines(P: dict) -> list[str]:
    out = []
    out.append("  P1 held lock, G=4 SHARED slope 2-6: " + " | ".join(
        f"eta {eta}: " + " ".join(f"{f:.0f} {P['P1'][f'G4/f{f:.0f}/eta{eta}']['slope_2_6']:.4f}"
                                  for f in P1_CLOCKS) for eta in P1_ETAS))
    for eta in P1_ETAS:
        d = P["P1"][f"G2/f{P1_CLOCKS[0]:.0f}/eta{eta}"]
        out.append(f"  P1 G=2 increments at {P1_CLOCKS[0]:.0f}, eta {eta}: "
                   + " ".join(f"{v:.3f}" for v in d["inc"]))
    out.append("     falsified if: a G=4 slope that neither eta gives (between them, near "
               "0.62-0.64 at 1410 on the GH200), or the eta from the G=4 slope and the eta "
               "from the G=2 low steps disagreeing")
    return out


def _p2_lines(P: dict) -> list[str]:
    out = []
    for G, d in P["P2"].items():
        out.append(f"  P2 G={G} SHARED at {P['clock_mhz']:.0f} MHz: {_ladder(d)} | n1->2 "
                   f"{d['step_1_2']:.3f} (a point prediction, not held to the band), slope "
                   f"2-6 {d['slope_2_6']:.4f}, band [{d['band'][0]:.3f}, {d['band'][1]:.3f}]")
        if "private" in d:
            pv = d["private"]
            out.append(f"  P2 G={G} PRIVATE at {P['clock_mhz']:.0f} MHz: {_ladder(pv)} | slope "
                       f"2-6 {pv['slope_2_6']:.4f} (a point prediction, no band)")
    out.append("     falsified if: a SHARED per-tread increment from n=2 to n=6, or the slope "
               "2-6, falls outside the band. The band is this tool's reading of the judge's "
               "(0.98 to 1.01 x the slope, rounded outward: 1% of noise either side, 1% more "
               "below for the measured floor's fall with G)")
    return out


def _p3_lines(P: dict) -> list[str]:
    out = []
    g2 = P["P3"]["G2"]
    out.append("  P3 G=2 n=7..10: " + " ".join(f"{v:.4f}" for v in g2["T"][6:10])
               + " | odd-n excess over G=4 at n=3,5,7,9: "
               + " ".join(f"{v:.3f}" for v in P["P3"]["odd_n_excess"].values()) + " ms")
    out.append(f"     falsified if: the odd-n excess falls below {P3_MIN_EXCESS_MS:.2f} ms or "
               "trends with n")
    return out


def _p4_lines(P: dict) -> list[str]:
    out = []
    p4 = P["P4"]
    out.append(f"  P4 G=4 n=8->9: {p4['inc_8_9']:.3f} against {p4['inc_others_median']:.3f} "
               f"elsewhere, because q_w2 drops from {p4['q_w2'][8]:.3f} to {p4['q_w2'][9]:.3f}")
    out.append("     falsified if: n=8->9 reads like the others (then the q term is wrong)")
    return out


def _p5_lines(P: dict) -> list[str]:
    out = []
    p5 = P["P5"]
    out.append(f"  P5 G=3: {_ladder(p5['G3'])} | rho*_w1 {p5['rho_star_w1']:.3f} against the "
               f"largest slab-fetching share at G=3, {p5['slab_fraction_max_G3']:.3f}: "
               + ("flat (rho* above it)" if p5["flat"] else "a period-3 ripple (rho* below it)"))
    out.append("     falsified if: the ladder does the other")
    return out


def _p678_lines(P: dict) -> list[str]:
    out = []
    p6 = P["P6"]
    out.append(f"  P6 --floor capture: {p6['cycles_per_cta_kstep']:.1f} cycles per CTA k-step at "
               f"any held clock"
               + (" (a capture's slope over CTA k-steps reads "
                  + ", ".join(f"{g} {v:.1f}" for g, v in p6["capture_slope"].items())
                  + " with the per-CTA fixed cost)" if p6.get("cta_fixed") else "")
               + ("; the census PTX's shared-memory traffic needs "
                  + ", ".join(f"{v:.0f}" for v in p6["ptx_smem_cycles"])
                  + f" at {SMEM_BYTES_PER_CLK} B per clock" if p6["ptx_smem_cycles"] else "")
               + ". INTERPRETATION: the shared-memory pipe near saturation and above the tensor "
               "pipe; achieved occupancy about half the limit would derive k_w = 1/2")
    out.append("     falsified if: cycles far from this, or achieved occupancy near the limit "
               "(k_w = 1/2 then stays unexplained)")
    p7 = P["P7"]
    if not p7["locked_fit"]:
        out.append("  P7 needs a LOCKED fit on this card to read an unlocked page against; "
                   "this fit is unlocked")
    elif not p7["pages"]:
        out.append(f"  P7 no unlocked page of this card among the inputs. The rule: a fixed "
                   f"{p7['cycles']:.1f} cycles per CTA k-step reads an unlocked page's c-only "
                   f"refit c_u as an in-kernel clock of cycles / c_u. {P7_INTERPRETATION}")
    else:
        for d in p7["pages"]:
            out.append(f"  P7 unlocked page {d['run']} G={d['G']} ({d['label']}, never fitted): "
                       f"c {d['c_ns']:.2f} ns over {d['cells']} cells (rms {_pct(d['rms'])}) = "
                       f"{d['cycles_at_nvml']:.1f} cycles at NVML {d['nvml_mhz']:.0f} MHz; a "
                       f"fixed {p7['cycles']:.1f} cycles reads it as an in-kernel clock of "
                       f"{d['in_kernel_mhz']:.0f} MHz. {P7_INTERPRETATION}")
    out.append("     falsified if: the in-kernel clock (a capture's own cycles over its "
               "duration) reads near NVML; then the floor is not a fixed cycle count")
    if not P["P8"]:
        out.append("  P8 T3 did not trip on this card; no byte inversion")
    for G, rows in P["P8"].items():
        out.append(f"  P8 SHARED at G={G}, w2 reads the fit without G={G} needs (units of W_w2), "
                   "against the byte source: " + " ".join(
                       f"n{r['n']} {r['w2_have_Ww2']:.2f} -> "
                       + ("n/a" if r["w2_needed_Ww2"] is None else f"{r['w2_needed_Ww2']:.2f}")
                       + (" (n=5 interpolated)" if "INTERPOLATED" in r["source"] else "")
                       + (f" ({r['note']})" if r["note"] else "") for r in rows))
        out.append("     falsified if: a timed-regime counter capture at this G reads the "
                   "source's bytes, not these (then the per-GEMM window is wrong here)")
    return out


def lines_of(R: dict) -> list[str]:
    ctx, main, sc = R["ctx"], R["main"], R["score"]
    card = ctx.card
    L = ["R3 TIMING MODEL LOWM-1/2 (scripts/r3_timing_model.py). NOT FINAL: the best current "
         "per-card model at one locked clock; it fails on the H100 at G=1, carries one fitted "
         "study constant (k_w) and has no identified clock term. Falsified by any of:"]
    L += [f"  - {f}" for f in FALSIFIERS]
    L += ["", f"CARD {card} {board(ctx.device)} (never pooled with another card)",
          f"  SMs {ctx.sms} from {ctx.sms_source}; occupancy w1 {ctx.occupancy['w1']} / w2 "
          f"{ctx.occupancy['w2']} CTAs per SM, {ctx.occupancy_source}",
          f"  bytes: {ctx.byte_label}",
          "  clock: " + ("LOCKED" if ctx.locked
                         else "UNLOCKED, median of the fitted SHARED cells at G >= 2")
          + f" {ctx.clock_mhz:.0f} MHz; k_w {R['k_w']} (FITTED study constant); window "
          + ", ".join(f"{g} {window_width(R['k_w'], ctx.sms, ctx.occupancy[g])} CTAs"
                      for g in GEMMS),
          "PAGES"]
    for p in R["all_pages"]:
        L.append(f"  {p.describe()}  {'FITTED' if p in R['use'] else 'EXCLUDED'}")
    if R["source"].pages:
        L.append("COUNTER PAGES (bytes, grid and occupancy read; gates as stored, not re-scored, "
                 "except the lock gate, recomputed on read)")
        for G, page in sorted(R["source"].pages.items()):
            bad = stored_validity(page)
            L.append(f"  G={G} {page['_path']}" + (f"  stored VALIDITY not PASS: {', '.join(bad)}"
                                                   if bad else ""))
    L += [f"  {n}" for n in R["notes"]]
    L += ["", *_fit_lines(f"FIT {card}", main, sc, ctx)]
    lg = R["logo"]
    if lg:
        L.append("  LOGO held-out rms/worst: " + " ".join(
            f"G{G} " + (d if isinstance(d, str) else f"{_pct(d['rms'])}/{100 * d['worst']:+.2f}%")
            for G, d in lg.items()) + f"  mean {_pct(logo_mean(lg))}")
    pl, gt = R["private_line"], R["gates"]
    L.append(f"  R_P (PRIVATE's ms against its read bytes, {pl['cells']} cells): "
             + ("n/a" if pl["R_P_gbps"] is None else f"{pl['R_P_gbps']:.1f} GB/s; bw/R_P - 1 = "
                f"{100 * gt['T2']['bw_over_R_P_minus_1']:+.2f}%"))
    L.append("  rho* = S c bw / (SMs s_g): " + " ".join(
        f"{g} {v:.3f}" for g, v in R["rho_star"].items())
        + " (INTERPRETATION: the slab-fetching share of a window above which it streams)")
    L.append("GATES " + " | ".join(
        [f"T1 {'PASS' if gt['T1']['pass'] else 'FAIL'}",
         f"T2 {'PASS' if gt['T2']['pass'] else 'FAIL'} ({_pct(gt['T2']['bw_over_R_P_minus_1'])})",
         f"T3 {'PASS' if gt['T3']['pass'] else 'FAIL'}"
         + (f" FORM FAILS AT G={','.join(map(str, gt['T3']['fails_at_G']))}"
            if gt["T3"]["fails_at_G"] else ""),
         f"T4 {'PASS' if gt['T4']['pass'] else 'FAIL'} ({100 * gt['T4']['worst']:+.2f}%)"]))
    L += ["", *_f_lines(R["F"])]
    for G, d in R["t3"].items():
        L += ["", *_fit_lines(f"FORM FAILS AT G={G}: FIT {card} WITHOUT G={G} (labelled; G={G} "
                              "scored as a prediction)", d["fit"], d["score"], ctx)]
        f1 = d["F"]["F1_G1_shared_slope_2_6"]
        if f1:
            L.append(f"  F1 G=1 SHARED slope measured {f1['measured']:.4f} model {f1['model']:.4f}")
    rf, rsc = R["reference"], R["reference_score"]
    L += ["", *_fit_lines(f"REFERENCE k_w = {K_W_REFERENCE} (the window the occupancy limit "
                          "derives)", rf, rsc, ctx)]
    if R["ptx"]:
        L.append("PTX (census fused_moe_kernel, the loop issuing mma.sync; per thread): "
                 + "; ".join(
            f"{Path(p['path']).parent.name[:10]} mma.sync {p['mma_sync']} ldmatrix "
            f"{p['ldmatrix']} cp.async.cg {p['cp_async_cg']} x {p['threads']} threads = "
            f"{p['smem_bytes_per_kstep']} B per CTA k-step = {p['smem_cycles']:.0f} cycles at "
            f"{SMEM_BYTES_PER_CLK} B/clk" for p in R["ptx"]))
    L += ["", *_prediction_lines(R["predictions"], card)]
    L += ["", "CELLS (arm/G/n, measured ms, model ms, residual, byte source, fitted or scored)"]
    for c in R["cells"]:
        L.append(f"  {c.key:<18} {c.ms:.4f} {sc['pred'][c.key]:.4f} "
                 f"{100 * sc['resid'][c.key]:+.2f}% {c.source:<24} "
                 f"{'fitted' if c.fit else 'scored'}")
    return L


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(u) for k, u in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(u) for u in v]
    if isinstance(v, np.ndarray):
        return [_jsonable(u) for u in v.tolist()]
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


def _fit_json(f: Fit, sc: dict, ctx: Context) -> dict:
    return {"label": f.label, "k_w": f.k_w, "params": f.params,
            "units": UNITS, "c_cycles": f.params["c"] * ctx.clock_mhz * 1e-3,
            "clock_mhz": ctx.clock_mhz, "fitted_G": list(f.fitted_gs),
            "identified": dict(zip(NAMES, f.identified, strict=True)),
            "at_bound": dict(zip(NAMES, f.at_bound, strict=True)),
            **{k: v for k, v in sc.items() if k not in ("resid", "pred")}}


def doc_of(R: dict) -> dict:
    ctx, sc = R["ctx"], R["score"]
    return _jsonable({
        "tool": "scripts/r3_timing_model.py", "model": "LOWM-1/2",
        "status": "NOT FINAL", "falsified_by": list(FALSIFIERS),
        "card": ctx.card, "device": ctx.device, "sms": ctx.sms, "sms_source": ctx.sms_source,
        "occupancy": ctx.occupancy, "occupancy_source": ctx.occupancy_source,
        "clock_mhz": ctx.clock_mhz, "locked": ctx.locked, "bytes": ctx.byte_label,
        "k_w": R["k_w"],
        "pages": [{"path": str(p.path), "run": p.run, "label": p.label, "failed": list(p.failed),
                   "G": p.G, "duty": p.duty, "locked": p.locked, "fitted": p in R["use"]}
                  for p in R["all_pages"]],
        "fit": _fit_json(R["main"], sc, ctx),
        "logo": R["logo"], "logo_mean": logo_mean(R["logo"]),
        "private_line": R["private_line"], "gates": R["gates"], "F": R["F"],
        "rho_star": R["rho_star"],
        "t3_refits": {G: {**_fit_json(d["fit"], d["score"], ctx), "F": d["F"]}
                      for G, d in R["t3"].items()},
        "reference_k_w_1": _fit_json(R["reference"], R["reference_score"], ctx),
        "ptx": R["ptx"],
        "cells": [{"key": c.key, "arm": c.arm, "G": c.G, "n": c.n, "ms": c.ms, "mhz": c.mhz,
                   "path": c.path, "declared": c.declared, "runs": list(c.runs),
                   "bytes_source": c.source, "reads": c.reads, "sigma": c.sigma,
                   "fitted": c.fit, "model_ms": sc["pred"][c.key], "residual": sc["resid"][c.key]}
                  for c in R["cells"]],
        "notes": R["notes"], "predictions": R["predictions"]})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit LOWM-1/2 (the launch-order window max, half window) to one card's R3 "
                    "timed pages. NOT FINAL; every output is labelled per card.")
    p.add_argument("inputs", nargs="+", type=Path,
                   help="published session dirs or run dirs; every private_weight_reference "
                        "page under each is read, VALID ones fitted")
    p.add_argument("--counters", type=Path, default=None,
                   help="the card's r3c-g{G}.json dir (default: found under the inputs)")
    p.add_argument("--bytes", default="counted",
                   help="counted (default), groupmodel (sigma = 1) or borrowed:DIR "
                        "(another card's counter pages); printed on every cell")
    p.add_argument("--k-w", type=float, default=K_W,
                   help=f"window scale (default {K_W}, FITTED); the k_w = {K_W_REFERENCE} "
                        "fit is always printed beside it")
    p.add_argument("--p-knee", type=float, default=P_KNEE,
                   help=f"the knee's soft-max exponent (default {P_KNEE:g}, a FITTED study "
                        "constant; inf is the hard max the judge's pins were fitted with)")
    p.add_argument("--no-tail", action="store_true",
                   help="price a partial last wave at throughput (the judge's pins); by "
                        "default its CTAs cost their whole lifetime (TAIL)")
    p.add_argument("--no-cores", action="store_true",
                   help="price a partial last wave's CTAs at their full-occupancy lifetime "
                        "S'_g c occ_g (every pin before 2026-09-30); by default the "
                        "co-residency law (CORES): round-robin k_SM, g(k) = k, and the "
                        f"wave's bytes at min(bw, {CORES_RHO_GBPS} GB/s x its CTAs)")
    p.add_argument("--cores-per-lead", action="store_true",
                   help="the co-residency law's DRAM term per slab-fetching CTA (rho each) "
                        "instead of pooled over the wave; the form the calibration cannot "
                        "tell apart from the default")
    p.add_argument("--no-dead", action="store_true",
                   help="price the grid's dead CTAs at zero (the judge's pins); by default "
                        f"each costs {DEAD_CTA_NS} ns of dispatch past one effective lifetime "
                        "(DEAD_CTA_NS)")
    p.add_argument("--no-cta-fixed", action="store_true",
                   help="price a floor-bound CTA at S_g k-steps of c (every pin before "
                        "2026-09-29); by default it costs S_g + CTA_FIXED_KSTEPS "
                        f"{CTA_FIXED_KSTEPS}, measured on the GH200's counter pages")
    p.add_argument("--gemm-const", action="store_true",
                   help="add the per-GEMM constant in its cycle form (GEMM_CONST: "
                        f"{GEMM_CONST_CYCLES:g} + {GEMM_CONST_PER_U:g} u SM cycles a GEMM call, "
                        "MEASURED on the CAL floor captures; rental 3, 2026-10-05); off by "
                        "default until part A scores it")
    p.add_argument("--out", type=Path, default=None, help="write everything as JSON here")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        R = build(args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED
    print("\n".join(lines_of(R)))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc_of(R), indent=1, default=str) + "\n")
        print(f"\nwrote {args.out}")
    return exit_codes.DONE


if __name__ == "__main__":
    sys.exit(main())
