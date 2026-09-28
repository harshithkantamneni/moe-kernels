#!/usr/bin/env python
"""THE WAVE-SPLIT CAPACITY MODEL (WSC) of fused_moe's per-GEMM DRAM weight reads,
fitted per card to the r3-arms counter pages. NOT FINAL (see the end of this
docstring for why and for what would falsify it).

    python scripts/wave_split_bytes.py <dir of one card's r3c-g*.json> [<dir> ...]
    python scripts/wave_split_bytes.py DIR --params-out OUTDIR --out cells.json
    python scripts/wave_split_bytes.py DIR --params PINNED.json      # no fit

WHAT IT IS FOR, 2026-09-26. The r3-arms counter pages (COUNTERS.md 6) read
DRAM bytes per GEMM for three arms (SHARED, NATIVE, PRIVATE) at GROUP_SIZE_M G
in {1, 2, 4, 16, 64} and treads n in {1, 2, 3, 4, 6}. `DCR.group_reads` (6.7)
says what they would read if L2 served every re-read inside a GROUP_SIZE_M
group and none across groups. On the 120 cells with n >= 2 it misses by
18.04% rms (A100), 17.96% (GH200) and 18.73% (H100), worst +70.5%: w2 re-reads
ACROSS groups are partly served (the co-resident window launches CTAs of
several groups together) and activation (A) tiles are partly re-read from
DRAM. Two research models were built for those misses (the w2 track, a
jittered-window LRU simulation and this capacity model) and a judge re-ran
both on identical cells (the w2 judge's report, 2026-09-26). This file is the
one the judge picked, built to the judge's implementation spec (section 6).

WHAT IT COMPUTES. For one (arm, G, n, GEMM g) cell, q_g in units of the GEMM's
weight set W_g (`DCR.r3_q`'s unit):

    q_g = [ S_c + sum_x (1 - sigma_B(theta_e D_e fbar)) + eps1_g N_in,win1 ] / (E P_g)
          + (ATILE_g / W_g) sum_A (1 - sigma_A(thetaA_e X_e))

  S_c        the compulsory slab reads: distinct (owner, pid_n) keys, owner the
             expert (SHARED, NATIVE) or the M-tile (PRIVATE);
  x events   slab RE-READS whose previous reader sat in ANOTHER GROUP_SIZE_M
             group, each with its live-rank distance D (the walk below);
  in events  re-reads by a partner in the SAME group. Inside the first
             co-residency window (live rank < W_c) a fraction eps1_g of them
             misses (partners launched in the same instant duplicate their
             misses); after it they never miss;
  A events   an M-tile's A tile read again at its next column, X_e its reuse
             distance: X = D_A fbar in the FILL view, or X = WS_col, the
             group's column-pass working set (live tiles x ATILE + distinct
             owners x SLAB, `DCR.r3_exposure`'s quantity), in the WS view; the
             MIX view, registered since 2026-09-27 (REGISTERED_VIEW), reads
             w1's in fill and w2's in ws, and the other two print beside it;
  sigma(F)   = exp(-(F / C)^beta), a capacity survival law in the bytes F the
             L2 takes in between: (C_B, beta_B) for slabs, (C_A, beta_A) for A;
  theta_e    theta1 inside the first window, 1 after it (the first window
             launches together, so its partners sit closer than their pid
             distance says); thetaA_e the same with theta1_A;
  fbar       = q_g W_g / (E n P_g), the DRAM bytes per live CTA. q appears on
             both sides, so q_g is a FIXED POINT q = F(q); F is monotone in q,
             and the model's q is its LEAST root on [S_c / (E P), (S_c + #x +
             #in) / (E P) + ATILE #A / W_g], found by a 97-point scan and then
             bisection (as the research code's `wavecap.predict`).

Q_total = (q_w1 W_w1 + q_w2 W_w2) / W, what a timed model consumes
(`per_tile_model_fit`'s structure `ovl.gemm.bytes`).

WHERE EACH INPUT COMES FROM. Nothing about a card is typed here: the SM count,
the L2 size and every cell's grid are the page's own (`page["card"]`,
`per_gemm[g].grid_size`), and the co-residency window is W_c,g = sm_count x the
least of the four occupancy limits the page recorded for GEMM g
(`DCR.coresident_window`, `DCR.R3_OCCUPANCY_LIMITS`): A100 432 / 432, GH200
660 / 528, H100 660 / 528 (w1 / w2) on the published pages. The research code
hard-coded both (the judge's defect (d)); a card whose pages record a
different occupancy gets its own window here. P_g = num_pid_n
(`DCR.pid_n_count`: 448 for w1, 64 for w2 on mixtral at BLOCK_N 64); SLAB_g =
BLOCK_N x K_g x b and ATILE_g = BLOCK_M x K_g x b (w1 524,288 and 262,144 B,
w2 1,835,008 and 917,504 B); W_g from `DCR.r3_byte_model` (W_w1 1,879,048,192
B, W_w2 939,524,096 B). A page whose grid / P_g is not ceil((E n BLOCK_M + D
(BLOCK_M - 1)) / BLOCK_M), D the arm's declared expert slots (72 for SHARED and
PRIVATE, 8 for NATIVE on these pages; `design.declared_by_arm`), is REFUSED:
the walk would not be the launch the page counted.

THE EVENTS: AN EXACT PID WALK AND A CLOSED FORM, WHICH MUST AGREE. `walk`
maps every pid of the grid exactly as vLLM 0.27.1's fused_moe_kernel does
(group_id = pid // (G P); first = G group_id; gs = min(num_pid_m - first, G);
pid_m = first + (pid mod G P) mod gs; pid_n = (pid mod G P) // gs; the same
arithmetic as `DCR.pid_mapping_reads`), skips dead CTAs (pid_m >= E n; they
exit at once, so the live rank r counts live CTAs only) and lists every
re-read with its distance. `closed_form` counts the same events from the
group boundaries alone. Every page cell is walked and checked against the
closed form before a number is printed; a disagreement refuses the run.

THE FIXED-POINT GUARD. On both Hoppers the fitted w2 G=1 map develops a
spurious low branch at n >= 6 (the judge's defect (c)): at n=6 the root is
ill-conditioned (kappa = 1 / (1 - F'(q*)) is 11.2 on GH200, 17.1 on H100), at
n=7 two roots sit 1.612 and 4.551 apart, and at n=8 the only root, 1.433,
would mean 94% of re-reads avoided while the measured trend is falling. A
cell is ILL-POSED, and prints NO number, when kappa > 10, when its least and
greatest roots differ by more than 0.01, or when G = 1 and its q is below the
q the same arm and GEMM gives at ANY smaller n (the monotonicity rule; the
spec names n - 1, which this contains, and its own example, n=8's 1.433 <
3.746, is n=6's q, the largest below it). Hopper (compute capability 9.x) w2
SHARED and NATIVE at G = 1 and n >= 6 is ILL-POSED by rule as well, so an n
the guard happens to pass there still prints nothing.

THE DOMAIN. w2 SHARED and NATIVE at G >= 32 is OUT-OF-DOMAIN on every card:
printed, never scored. At G=64 the ws view breaks w1 and the fill view runs
low on w2 (GH200 NATIVE -6.4% to -34.2%), and one activation law per card
cannot cover both GEMMs there (the judge's graft tests, section 4).

FITTING, PER CARD, NEVER POOLED ACROSS CARDS. Stage 1 fits the activation law
(C_A, beta_A, theta1_A) on PRIVATE alone, every G, every tread, both GEMMs
(50 cells): PRIVATE reads each slab exactly once, so its excess over n is A
re-reads alone. Stage 2 holds stage 1 and fits the slab law and the first
window (C_B, beta_B, theta1, eps1_w1, eps1_w2) on SHARED and NATIVE at G in {1,
2, 4} (60 cells). Loss: sum of (q_pred / q_meas - 1)^2, over every cell of the
stage, the two Hopper G=1 n=6 w2 cells the guard marks ILL-POSED included
(their status is read from the fitted parameters, after the fit; the
reference fit the tests pin included them). C and beta are fitted as logs,
theta and eps as logits, by the research fit's simplex: three fixed-seed
starts, each polished by a second run at a third of the step. It lands on the
reference parameters to four decimals on all three cards, in about five
seconds a card. `per_tile_model_fit.bounded_lsq` was tried first and is NOT
used: its forward-difference Jacobian stalls at the fold the guard exists for
(GH200 stage 2 stopped 1.2% above the simplex's SSE after 47,000 model
evaluations, 2026-09-26). A theta or eps is flagged at a bound when it is
within 1e-4 of 0 or 1, when its logit is past +-6, or when setting it to the
bound moves its stage's SSE by less than 1e-6 relative (eps1_w1 on both
Hoppers: the logit ran toward minus infinity and stopped at 7e-8 on GH200;
the GH200 ws view's theta1_A at 0.00125, whose stage-1 SSE is the same to
nine decimals at 0, review F8); one that neither bound moves is flagged
UNCONSTRAINED. G=16 and G=64 SHARED/NATIVE are HELD OUT and scored.
"slab law NOT IDENTIFIED" prints when beta_B > 8 or C_B < 0.25 x L2 (it fires
on the A100: C_B 6.75 MiB, beta_B 9.88, effectively a step), and every
parameter at a bound is flagged. All three views are fitted; mix is the
registered one (REGISTERED_VIEW), fill and ws print beside it.

WHAT IT PRINTS. Per card: the pages and their stored gate verdicts; the
geometry read off them; the parameters per view with stage SSEs and flags; a
per-cell table (arm, G, n, GEMM, measured q, q_fill, q_ws, q_group, kappa and
a status: FIT, HELD-OUT, PREDICTED, ILL-POSED or OUT-OF-DOMAIN); the w2
decomposition (group model, first-window partner loss, cross-group recovery
inside and after the first window, activation); the scores; the registered
predictions for cells no page measured; and the FIRST-WAVE STATISTIC,
s1 = max(0, W_c,w1 - P_w1) / (E P_w1), the w1 slab re-reads at G=1 that fall
inside the first window per weight set (212 / 3584 on both Hoppers), with the
measured recovery phi = (n - q_S,w1(G=1, n)) / s1 per tread: 0.26 on GH200,
about 0.95 on H100, constant in n; nothing here explains the difference.
With two cards of one geometry that differ in L2 it prints the CROSS-CARD L2
CHECK (a comparison of two per-card models, nothing pooled): modeled minus
measured of (H100 - GH200) w2 SHARED at G in {2, 4, 16}, which the judge
adopts as an acceptance test at 0.05 (worst gap 0.036).

THE SCORES are of two kinds, both printed. The MODEL'S OWN score is over cells
that are in domain and well posed. The JUDGE'S COMPARISON SETS (the 120 cells
with n >= 2, w2 S+N at G <= 16, w2 PRIVATE) are the sets the judge scored
every model on, reproduced so the numbers can be checked against the report:
they INCLUDE the OUT-OF-DOMAIN G=64 w2 cells and the least roots of the
ILL-POSED cells, and say so.

WHAT IT CANNOT TELL. The parameters are fitted, not measured. An effective
slab capacity of about 70 MiB on both Hoppers (1.16 x L2 on GH200, 1.40 x on
H100) is larger than either L2 and independent of it; nothing here explains
that. The activation reuse distance is not pinned: PRIVATE alone cannot tell
the fill view from the ws view (both fit it), and at G=64 neither covers both
GEMMs. The model has no term for dead CTAs: at G=64 SHARED reads more than
NATIVE on both Hoppers (+0.118 to +0.043 of W_w2 at n = 2 to 6, about 3e-3
per extra dead CTA per column pass; the A100 runs the other way), and the
model predicts 0 for that difference (INTERPRETATION: registered as a
hypothesis, not modelled). Cross-group recovery at G=2 and n >= 3 is
over-stated 1.5 to 2.6 times on the Hoppers, where the plain group model is
better. The A100's d=1 pair loss (rho 0.778 against at most 0.44 from any
model) and its slab law are unexplained. The H200 has no counter pages; its
q comes from another card's parameters and occupancy, two columns, never
averaged.

THE MODEL IS NOT FINAL. It is the best byte model the study has for G <= 16
and for PRIVATE, and through the per-GEMM overlap time map it predicts the
unmeasured n=5 times to 1.69% (GH200) and 0.39% (H100) rms; its worst n=5
cell, GH200 G=2 SHARED, is -3.14% (G=1 SHARED -2.99%). A fixed-point
defect, an unidentified activation distance and the missing dead-CTA term
are known structural gaps, not noise. WHAT WOULD FALSIFY IT (any one):
  - GH200 w2 SHARED at G=8, n = 2 or 4, above 1.12 (it predicts 1.082 and
    1.118; the LRU rival predicts 1.139 and 1.159);
  - n=5 or n=8 at G = 2, 4 or 16 outside +-5% of its registered predictions;
  - a w1 G=1 saving (n - q) that changes with n;
  - a cross-card test (H100 against GH200) with a gap larger than 0.05 at
    G >= 2;
  - a GH200 byte page taken with the co-residency window halved (fewer CTAs
    per SM) where the n=2 over-read at G=2 does not shrink;
  - on any timed page with the per-GEMM overlap map, n=5 errors above 3% at
    G=1 or G=2 SHARED.
OPEN (review F6, 2026-09-26): the last test, as the judge worded it (section
7), is already past its threshold on the published GH200 lock pages the judge
set it on: -3.14% at G=2 SHARED, from the judge's own section 3 and from
`per_tile_model_fit --gemm-bytes` on those pages. The report does not say
whether it means pages not yet taken or a threshold nearer 3.5%. This tool
reads it as a test of new pages and prints the GH200 cell as an open
question for the judge; it is not counted as a pass.

It writes nothing unless --out or --params-out is given. In --out's JSON,
q_fill and q_ws are null where the cell is ILL-POSED; the raw roots the guard
judged are kept as least_root_fill, least_root_ws and greatest_root_fill,
diagnostics only. A params JSON names its counter pages relative to its own
directory (absolute when written without one), and a q source REFUSES when
they are gone rather than drop to the model's q.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import scripts.dram_counter_route as DCR  # noqa: E402
from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

MIB = 2 ** 20
GEMMS = ("w1", "w2")
ARMS = ("shared", "native", "private")

# --------------------------------------------------------------------------
# Design constants. None is a calibration: each is a choice the judge's spec
# made, with its reason.
# --------------------------------------------------------------------------

#: The fixed-point scan and bisection of `wavecap.predict`, kept as they were
#: so the pinned numbers are the research code's.
SCAN_POINTS = 97
BISECT_STEPS = 60

#: The central-difference step for F'(q*) in kappa, as the judge's j07.
KAPPA_STEP = 1e-4

#: ILL-POSED past these: a kappa over 10 amplifies a 1% change in F into a
#: 10% change in q; two roots 0.01 apart are two answers.
KAPPA_MAX = 10.0
ROOT_GAP_MAX = 0.01

#: w2 SHARED and NATIVE at G >= this are OUT-OF-DOMAIN on every card.
OUT_OF_DOMAIN_G = 32

#: Hopper (compute capability major 9) w2 SHARED/NATIVE at G=1 and n >= this
#: is ILL-POSED by rule (the spurious branch of the judge's defect (c)).
HOPPER_MAJOR = 9
HOPPER_G1_N = 6

#: "slab law NOT IDENTIFIED" past these (beta_B large is a step, C_B small
#: against L2 is a capacity no cache has).
SLAB_BETA_MAX = 8.0
SLAB_C_MIN_L2 = 0.25

#: The acceptance threshold of the cross-card L2 check, and its cells.
CROSS_CARD_TOL = 0.05
CROSS_CARD_G = (2, 4, 16)

#: The range outside which a fitted C or beta is flagged (the simplex runs
#: unbounded in log units, so a law that runs away says so): C in MiB from
#: L2 / 1000 to 1000 x L2, beta from 0.1 to 50.
C_SPAN = 1000.0
BETA_LO, BETA_HI = 0.1, 50.0

#: Starts per stage (the research fit's three) and their spread, in log /
#: logit units, around the research fit's starting point.
FIT_STARTS = 3
START_SPREAD = 0.5

#: The stages. Stage 2's G set; G=16 and G=64 SHARED/NATIVE are held out.
STAGE2_G = (1, 2, 4)

#: Registered predictions for cells no page measured: these G beside the
#: page's, these treads beside the page's.
PREDICT_G = (8, 32)
PREDICT_N = (5, 7, 8)

FILL, WS, MIX = "fill", "ws", "mix"
VIEWS = (FILL, WS, MIX)
#: THE REGISTERED VIEW IS MIX (2026-09-27): each GEMM reads its activation
#: reuse distance in the view its own data follow. w2's A misses at fixed G
#: stay flat as n grows (the 2026-09-27 GH200: SHARED G=16 12.2, 10.5, 9.8% at
#: n = 2, 4, 8; G=32 22 to 24%), which the column-pass working set gives and
#: the fill distance does not (fbar falls about 4x from n=2 to n=8); w1 breaks
#: under the working set (SHARED G=64 n=8 1.009 measured, 1.186 in ws) and
#: follows the fill. No parameter is added. Fitted on one GH200 board and
#: predicting the other: w1 1.04 -> 0.62% rms, PRIVATE w2 1.23 -> 0.95%, w2
#: SHARED+NATIVE G <= 16 n >= 5 7.45 -> 4.89%, G >= 32 29.1 -> 12.3%; the
#: other direction w1 1.12 -> 0.67%, n >= 5 6.25 -> 5.24%, G >= 32 17.5 ->
#: 7.95%, its shallow w2 cells 5.41 -> 5.62%. The G=8 test still separates
#: WSC from the LRU rival (n=2, 4: 1.084, 1.138 against 1.078, 1.100 measured
#: and LRU's 1.139, 1.159). INTERPRETATION, untested: w1's CTAs are short
#: (K 4096) and its followers catch their leader; w2's are long (K 14336) and
#: do not. Not closed: G=2's over-recovery at even n, the SHARED-only dead-CTA
#: excess at G >= 32, and a C_A above the L2. The fill and ws views are fitted
#: and printed beside it.
REGISTERED_VIEW = MIX
#: At G >= 2 a cross-group slab re-read past the first co-residency window is
#: a certain miss (2026-09-28; `cell_events`). False restores the survival law
#: there, the law the judge's 2026-09-26 pins were fitted with (`later_law`).
LATER_MISS = True


@contextlib.contextmanager
def later_law(later_miss: bool):
    """Evaluate under LATER_MISS = `later_miss`, restored after."""
    global LATER_MISS
    saved, LATER_MISS = LATER_MISS, later_miss
    try:
        yield
    finally:
        LATER_MISS = saved

FIT, HELD_OUT, PREDICTED = "FIT", "HELD-OUT", "PREDICTED"
ILL_POSED, OUT_OF_DOMAIN = "ILL-POSED", "OUT-OF-DOMAIN"

STAGE1_NAMES = ("C_A", "beta_A", "theta1_A")
STAGE2_NAMES = ("C_B", "beta_B", "theta1", "eps1_w1", "eps1_w2")


class Refused(Exception):
    """A precondition failed; nothing was fitted. Exits REFUSED."""


# --------------------------------------------------------------------------
# The geometry, read off the pages.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    """Everything the walk and the model need about one card and one kernel,
    each value read off the card's r3c pages."""

    card: str             # the card's slug, the label everything is printed under
    name: str
    uuid: str
    capability: str
    sm_count: int
    l2_bytes: int
    model: str
    dtype: str
    experts: int
    block_m: int
    block_n: int
    #: The rest of the kernel (2026-09-26, review F3): a timed page is given
    #: this card's q only when it ran this same kernel and declaration, and
    #: these are what `per_tile_model_fit`'s page tile carries beside the four
    #: above.
    block_k: int
    num_warps: int
    num_stages: int
    copies_declared: int
    declared: tuple       # ((arm, declared expert slots), ...)
    P: tuple              # ((gemm, num_pid_n), ...)
    K: tuple              # ((gemm, K), ...)
    W: tuple              # ((gemm, W_g bytes), ...)
    W_c: tuple            # ((gemm, co-residency window), ...)
    ctas_per_sm: tuple    # ((gemm, least occupancy limit), ...)

    def get(self, what: str, key: str) -> int:
        return dict(getattr(self, what))[key]

    @property
    def kernel(self) -> tuple:
        """(model, dtype, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages): the
        shape of `per_tile_model_fit.Page.tile`, so the two compare directly."""
        return (self.model, self.dtype, self.block_m, self.block_n, self.block_k,
                self.num_warps, self.num_stages)

    @property
    def hopper(self) -> bool:
        try:
            return int(str(self.capability).split(".")[0]) == HOPPER_MAJOR
        except ValueError:
            return False

    @property
    def b(self) -> int:
        return dtype_bytes(self.dtype)

    def slab(self, gemm: str) -> int:
        return self.block_n * self.get("K", gemm) * self.b

    def atile(self, gemm: str) -> int:
        return self.block_m * self.get("K", gemm) * self.b

    def per_set(self, gemm: str) -> int:
        """Slabs per weight set, E x P_g: the unit the slab counts divide by."""
        return self.experts * self.get("P", gemm)

    def num_pid_m(self, arm: str, n: int) -> int:
        """ceil((E n BLOCK_M + D (BLOCK_M - 1)) / BLOCK_M), D the arm's
        declared expert slots: moe_align_block_size pads each declared
        expert's rows up to a multiple of BLOCK_M, and the grid is its rows."""
        d = dict(self.declared)[arm]
        rows = self.experts * n * self.block_m + d * (self.block_m - 1)
        return -(-rows // self.block_m)

    def as_json(self) -> dict:
        out = dataclasses.asdict(self)
        for k in ("declared", "P", "K", "W", "W_c", "ctas_per_sm"):
            out[k] = dict(getattr(self, k))
        return out

    @classmethod
    def from_json(cls, d: dict) -> Geometry:
        d = dict(d)
        missing = [f.name for f in dataclasses.fields(cls) if f.name not in d]
        if missing:
            raise Refused(f"a params geometry lacks {missing}; it was written before the "
                          "kernel was recorded, so no timed page can be matched to it: refit")
        for k in ("declared", "P", "K", "W", "W_c", "ctas_per_sm"):
            d[k] = tuple(sorted(d[k].items()))
        return cls(**d)


def page_window(page: dict, gemm: str) -> tuple[int, int]:
    """(W_c,g, CTAs per SM) from the page's own recorded occupancy limits:
    the least of the four, for GEMM g, the same in every cell (one kernel
    binary per GEMM), times the page's SM count. Refused otherwise."""
    card = page.get("card") or {}
    sm = card.get("sm_count")
    if sm is None:
        raise Refused(f"page {page.get('run_id')} records no sm_count; the co-residency "
                      "window cannot be formed")
    seen = set()
    for c in page["cells"]:
        rec = ((c.get("recorded") or {}).get(gemm)) or {}
        limits = [rec.get(m) for m in DCR.R3_OCCUPANCY_LIMITS]
        if any(v is None for v in limits):
            raise Refused(f"page {page.get('run_id')} cell {c['arm']} n={c['n']} {gemm}: an "
                          "occupancy limit is missing, so W_c is unknown")
        seen.add(int(min(limits)))
    if len(seen) != 1:
        raise Refused(f"page {page.get('run_id')} {gemm}: cells record {sorted(seen)} CTAs "
                      "per SM; one kernel binary per GEMM should give one")
    ctas = seen.pop()
    return DCR.coresident_window(int(sm), ctas), ctas


def page_geometry(page: dict) -> Geometry:
    design, card = page["design"], page.get("card") or {}
    model = design["model"]
    if model not in MODEL_CONFIGS:
        raise Refused(f"model {model!r} is not in moe.spec.MODEL_CONFIGS")
    cfg = MODEL_CONFIGS[model]
    bm = DCR.r3_byte_model(cfg, design["dtype"], int(design["block_m"]))
    P, K, Wc, ctas = {}, {}, {}, {}
    for g in GEMMS:
        k, n_cols = DCR.r3_gemm_geometry(cfg, g)
        P[g] = DCR.pid_n_count(n_cols, int(design["block_n"]))
        K[g] = int(k)
        Wc[g], ctas[g] = page_window(page, g)
    declared = design.get("declared_by_arm")
    if not declared:
        raise Refused(f"page {page.get('run_id')} records no declared_by_arm; the grid "
                      "cannot be checked against the declaration")
    if card.get("l2_bytes") is None:
        raise Refused(f"page {page.get('run_id')} records no l2_bytes")
    rest = [k for k in ("block_k", "num_warps", "num_stages", "copies_declared")
            if design.get(k) is None]
    if rest:
        raise Refused(f"page {page.get('run_id')} records no {rest}; the kernel a timed page "
                      "must match cannot be stated")
    return Geometry(
        card=str(card.get("slug") or card.get("name")), name=str(card.get("name")),
        uuid=str(card.get("uuid") or ""), capability=str(card.get("capability")),
        sm_count=int(card["sm_count"]), l2_bytes=int(card["l2_bytes"]),
        model=model, dtype=design["dtype"], experts=int(cfg.num_experts),
        block_m=int(design["block_m"]), block_n=int(design["block_n"]),
        block_k=int(design["block_k"]), num_warps=int(design["num_warps"]),
        num_stages=int(design["num_stages"]), copies_declared=int(design["copies_declared"]),
        declared=tuple(sorted((a, int(v)) for a, v in declared.items())),
        P=tuple(sorted(P.items())), K=tuple(sorted(K.items())),
        W=(("w1", int(bm["W_w1"])), ("w2", int(bm["W_w2"]))),
        W_c=tuple(sorted(Wc.items())), ctas_per_sm=tuple(sorted(ctas.items())))


# --------------------------------------------------------------------------
# The events: the exact pid walk, and the closed form it must agree with.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Events:
    """Every re-read of one (arm, G, n, GEMM) launch, grouped by value.

    Units: slab counts (divide by E P_g for q) and A-tile counts."""

    arm: str
    G: int
    n: int
    gemm: str
    P: int
    W_c: int
    num_pid_m: int
    live: int                  # live CTAs, E n P
    slabs: int                 # compulsory slab reads S_c
    x_D: np.ndarray            # cross-group slab re-reads: distance,
    x_win1: np.ndarray         #   inside the first window,
    x_k: np.ndarray            #   count
    in_win1: int               # within-group re-reads inside the first window
    in_later: int              # and after it
    a_D: np.ndarray            # A re-reads: distance,
    a_win1: np.ndarray         #   inside the first window,
    a_ws: np.ndarray           #   the re-reader group's column-pass working set (B),
    a_k: np.ndarray            #   count
    per_set: int = 0           # E P_g
    weight_bytes: int = 0      # W_g
    atile: int = 0             # ATILE_g

    def counts(self) -> dict:
        """The closed form's shape: {"x": {(D, win1): k}, "in1", "inL",
        "A": {(D, win1): k}, "slabs"}."""
        x, a = Counter(), Counter()
        for d, w, k in zip(self.x_D, self.x_win1, self.x_k, strict=True):
            x[(int(d), bool(w))] += int(k)
        for d, w, k in zip(self.a_D, self.a_win1, self.a_k, strict=True):
            a[(int(d), bool(w))] += int(k)
        return {"x": dict(x), "in1": self.in_win1, "inL": self.in_later, "A": dict(a),
                "slabs": self.slabs}


def pid_map(pid: np.ndarray, num_pid_m: int, num_pid_n: int, G: int):
    """vLLM 0.27.1 fused_moe_kernel's pid mapping, on an array of pids:
    (group_id, pid_m, pid_n). The arithmetic of `DCR.pid_mapping_reads`."""
    in_group = G * num_pid_n
    group_id = pid // in_group
    first = group_id * G
    size = np.minimum(num_pid_m - first, G)
    local = pid % in_group
    return group_id, first + local % size, local // size


def _previous(key: np.ndarray):
    """For every element whose key occurred before (in index order), its index
    and the index of the key's previous occurrence."""
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    return order[1:][same], order[:-1][same]


def walk(arm: str, G: int, n: int, P: int, W_c: int, num_pid_m: int, experts: int, *,
         slab_bytes: int = 0, atile_bytes: int = 0) -> Events:
    """Walk every pid of the launch grid through the pid mapping, dead CTAs
    skipped (they exit at once and take no slot time), and list every re-read.

    The live rank r of a CTA counts live CTAs launched before it. A slab key is
    (owner, pid_n), owner = pid_m for PRIVATE and pid_m // n otherwise; the
    first read of a key is compulsory, and each later one carries D = r -
    r_prev, 'in' when the previous reader sat in the same GROUP_SIZE_M group
    and 'x' otherwise, and win1 = r < W_c. An A re-read is M-tile m read again
    (its next column), with D_A = r - r_prev, win1, and WS_col of the
    re-reader's group = live tiles x ATILE + distinct owners x SLAB."""
    if min(G, n, P, num_pid_m, experts) < 1:
        raise ValueError(f"walk needs G, n, P, num_pid_m and E >= 1, got {G}, {n}, {P}, "
                         f"{num_pid_m}, {experts}")
    live_m = experts * n
    if num_pid_m < live_m:
        raise ValueError(f"num_pid_m {num_pid_m} is below the {live_m} live M-tiles")
    pid = np.arange(num_pid_m * P, dtype=np.int64)
    gid, pid_m, pid_n = pid_map(pid, num_pid_m, P, G)
    keep = pid_m < live_m
    gid, pid_m, pid_n = gid[keep], pid_m[keep], pid_n[keep]
    live = int(pid_m.size)
    owner = pid_m if arm == "private" else pid_m // n
    key = owner * P + pid_n
    cur, prev = _previous(key)
    slabs = live - int(cur.size)
    D = cur - prev
    inside = gid[cur] == gid[prev]
    win1 = cur < W_c
    x = Counter(zip(D[~inside].tolist(), win1[~inside].tolist(), strict=True))
    in_win1 = int(np.sum(inside & win1))
    in_later = int(np.sum(inside & ~win1))
    # Column-pass working set per group: live tiles and distinct owners.
    groups = -(-num_pid_m // G)
    ws = np.zeros(groups)
    for g in range(groups):
        first = g * G
        tiles = range(first, min(first + G, live_m)) if first < live_m else range(0)
        owners = len(tiles) if arm == "private" else len({m // n for m in tiles})
        ws[g] = len(tiles) * atile_bytes + owners * slab_bytes
    cur_a, prev_a = _previous(pid_m)
    da = cur_a - prev_a
    wa = cur_a < W_c
    a = Counter(zip(da.tolist(), wa.tolist(), ws[gid[cur_a]].tolist(), strict=True))

    def arrays(counter, width):
        if not counter:
            return [np.zeros(0) for _ in range(width + 1)]
        keys = sorted(counter)
        cols = [np.array([k[i] for k in keys], dtype=float) for i in range(width)]
        return cols + [np.array([counter[k] for k in keys], dtype=float)]

    x_D, x_w, x_k = arrays(x, 2)
    a_D, a_w, a_ws, a_k = arrays(a, 3)
    return Events(arm=arm, G=G, n=n, gemm="", P=P, W_c=W_c, num_pid_m=num_pid_m,
                  live=live, slabs=slabs, x_D=x_D, x_win1=x_w.astype(bool), x_k=x_k,
                  in_win1=in_win1, in_later=in_later, a_D=a_D,
                  a_win1=a_w.astype(bool), a_ws=a_ws, a_k=a_k)


def closed_form(arm: str, G: int, n: int, P: int, W_c: int, experts: int) -> dict:
    """The walk's event counts from the group boundaries alone.

    Live tiles 0..E n - 1 lead the grid and dead ones follow, so group g holds
    live tiles [g G, g G + L_g), L_g = min(G, E n - g G), and the live rank of
    (m, j) in group g is R_g + j L_g + (m - g G), R_g = P sum_{g' < g} L_g'.
    CROSS-GROUP (SHARED, NATIVE): at every boundary b = k G with n not
    dividing b, the expert holding tiles b - 1 and b re-reads slab j at live
    distance D_j = P L_{k-1} + j (L_k - L_{k-1}) - (G - 1), inside the first
    window when R_k + j L_k < W_c. WITHIN-GROUP: a tile whose predecessor in
    the column belongs to the same expert re-reads that slab at ranks R + j L +
    idx. A: tile idx of group g is re-read at every column j >= 1 at distance
    L_g. The same shape as `Events.counts`."""
    live = experts * n
    gs, R, g = [], 0, 0
    while g * G < live:
        L = min(G, live - g * G)
        gs.append((g, L, R))
        R += P * L
        g += 1

    def first_window(start: int, L: int) -> int:
        return max(0, min(P, math.ceil((W_c - start) / L))) if W_c > start else 0

    x: Counter = Counter()
    in1 = inl = 0
    if arm != "private":
        for k in range(1, len(gs)):
            if (k * G) % n == 0:
                continue
            _, lp, _rp = gs[k - 1]
            _, lk, rk = gs[k]
            for j in range(P):
                x[(P * lp + j * (lk - lp) - (G - 1), (rk + j * lk) < W_c)] += 1
        for (g, L, R) in gs:
            tiles = range(g * G, g * G + L)
            for idx in range(1, L):
                if tiles[idx] // n == tiles[idx - 1] // n:
                    w = first_window(R + idx, L)
                    in1 += w
                    inl += P - w
    a: Counter = Counter()
    for (_g, L, R) in gs:
        for idx in range(L):
            w = max(0, first_window(R + idx, L) - 1)
            a[(L, True)] += w
            a[(L, False)] += P - 1 - w
    return {"x": {k: v for k, v in x.items() if v}, "in1": in1, "inL": inl,
            "A": {k: v for k, v in a.items() if v},
            "slabs": live * P if arm == "private" else experts * P}


def cell_events(geom: Geometry, arm: str, G: int, n: int, gemm: str) -> Events:
    """The walk of one cell on `geom`'s grid, checked against the closed form,
    and its largest WS_col against `DCR.r3_exposure`'s fullest-group working
    set (the ws view's distance must be the repo's quantity); a disagreement
    refuses."""
    P = geom.get("P", gemm)
    Wc = geom.get("W_c", gemm)
    npm = geom.num_pid_m(arm, n)
    ev = walk(arm, G, n, P, Wc, npm, geom.experts, slab_bytes=geom.slab(gemm),
              atile_bytes=geom.atile(gemm))
    cf = closed_form(arm, G, n, P, Wc, geom.experts)
    if ev.counts() != cf:
        raise Refused(f"{geom.card} {arm} G={G} n={n} {gemm}: the pid walk and the closed "
                      "form disagree; no number from this walk can be trusted")
    if ev.a_ws.size:
        ex = DCR.r3_exposure(MODEL_CONFIGS[geom.model], geom.dtype, geom.block_m,
                             geom.block_n, G, n)[gemm]
        want = ex["private" if arm == "private" else "shared"]["working_set"]
        if float(ev.a_ws.max()) != float(want):
            raise Refused(f"{geom.card} {arm} G={G} n={n} {gemm}: the walk's largest "
                          f"column-pass working set {ev.a_ws.max():.0f} B is not "
                          f"DCR.r3_exposure's {want} B")
    if LATER_MISS and G >= 2 and ev.x_D.size:
        # THE LATER CROSS-GROUP RE-READ MISSES AT G >= 2 (2026-09-28). Past the
        # first co-residency window the re-read sits D = 2P-1 CTAs on (8x7B w2:
        # 105 to 141 MiB of fill, 8x22B: 189 to 252 MiB, 1.7 to 4.2 x the 60 MiB
        # L2), where the slab law, fitted mostly at G=1 near 1.1 x L2, still gave
        # 3 to 12% survival. Measured, survival there is about 0; the tail
        # over-stated G=2's recovery 8.5% rms on 8x7B (n=8 -14.8%). Counted as
        # certain misses: 8x7B G=2 w2 SHARED 8.5 -> 2.2% rms, 8x22B 2.3 -> 1.0%,
        # cross-model worst 3.2 to 4.3% (the fill-view law gave -72% from 8x22B's
        # parameters); w1 unchanged. G=1 keeps the law. No parameter added.
        later = ~ev.x_win1
        ev = dataclasses.replace(ev, slabs=ev.slabs + int(ev.x_k[later].sum()),
                                 x_D=ev.x_D[~later], x_win1=ev.x_win1[~later],
                                 x_k=ev.x_k[~later])
    return dataclasses.replace(ev, gemm=gemm, per_set=geom.per_set(gemm),
                               weight_bytes=geom.get("W", gemm), atile=geom.atile(gemm))


# --------------------------------------------------------------------------
# The model: many cells at once, so one fit is seconds and not minutes.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    """One card's parameters in one view. C in MiB."""

    C_A: float
    beta_A: float
    theta1_A: float
    C_B: float
    beta_B: float
    theta1: float
    eps1_w1: float
    eps1_w2: float
    view: str = FILL

    def replace(self, **kw) -> Params:
        return dataclasses.replace(self, **kw)

    def as_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> Params:
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d})


def sigma(F, C, beta):
    """Survival exp(-(F / C)^beta) of a line that F bytes have passed."""
    return np.exp(-np.power(np.maximum(F, 0.0) / C, beta))


def _segsum(vals: np.ndarray, cell: np.ndarray, count: int) -> np.ndarray:
    """Per-cell sums of event values, 1-D (events) or 2-D (events x points)."""
    if vals.ndim == 1:
        return np.bincount(cell, weights=vals, minlength=count)
    k = vals.shape[1]
    flat = (cell[:, None] * k + np.arange(k)[None, :]).ravel()
    return np.bincount(flat, weights=vals.ravel(), minlength=count * k).reshape(count, k)


class Batch:
    """The events of many cells packed into flat arrays, so F(q) for every
    cell (or every cell at many q) is one numpy pass."""

    def __init__(self, events):
        self.events = list(events)
        ev = self.events
        self.count = len(ev)
        self.S = np.array([e.per_set for e in ev], dtype=float)
        self.Wg = np.array([e.weight_bytes for e in ev], dtype=float)
        self.at = np.array([e.atile for e in ev], dtype=float)
        self.live = np.array([e.live for e in ev], dtype=float)
        self.slabs = np.array([e.slabs for e in ev], dtype=float)
        self.in1 = np.array([e.in_win1 for e in ev], dtype=float)
        self.inl = np.array([e.in_later for e in ev], dtype=float)
        self.is_w1 = np.array([e.gemm == "w1" for e in ev])

        def cat(name, dtype=float):
            parts = [np.asarray(getattr(e, name), dtype=dtype) for e in ev]
            return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)

        self.x_cell = np.concatenate([np.full(e.x_D.size, i) for i, e in enumerate(ev)]
                                     or [np.zeros(0, int)]).astype(int)
        self.x_D, self.x_k, self.x_w1 = cat("x_D"), cat("x_k"), cat("x_win1", bool)
        self.a_cell = np.concatenate([np.full(e.a_D.size, i) for i, e in enumerate(ev)]
                                     or [np.zeros(0, int)]).astype(int)
        self.a_D, self.a_k, self.a_ws = cat("a_D"), cat("a_k"), cat("a_ws")
        self.a_w1 = cat("a_win1", bool)
        self.q_min = self.slabs / self.S
        self.q_max = ((self.slabs + _segsum(self.x_k, self.x_cell, self.count)
                       + self.in1 + self.inl) / self.S
                      + self.at * _segsum(self.a_k, self.a_cell, self.count) / self.Wg + 1e-9)

    def F(self, prm: Params, q) -> np.ndarray:
        """The model map: the q every cell returns when its fill per live CTA
        is fbar = q W_g / N_live. Monotone non-decreasing in q. `q` is one
        value per cell, or a (cells x points) array."""
        q = np.asarray(q, dtype=float)
        two = q.ndim == 2
        per = self.Wg / self.live
        fbar = q * (per[:, None] if two else per)
        eps = np.where(self.is_w1, prm.eps1_w1, prm.eps1_w2)
        base = (self.slabs + eps * self.in1) / self.S
        out = np.broadcast_to(base[:, None], q.shape).copy() if two else base.copy()
        if self.x_D.size:
            th = np.where(self.x_w1, prm.theta1, 1.0) * self.x_D
            f = fbar[self.x_cell]
            F = th[:, None] * f if two else th * f
            miss = (self.x_k[:, None] if two else self.x_k) * (1.0 - sigma(F, prm.C_B * MIB,
                                                                            prm.beta_B))
            out += _segsum(miss, self.x_cell, self.count) / (self.S[:, None] if two else self.S)
        if self.a_D.size:
            tha = np.where(self.a_w1, prm.theta1_A, 1.0)
            if prm.view == WS:
                miss = self.a_k * (1.0 - sigma(tha * self.a_ws, prm.C_A * MIB, prm.beta_A))
                add = self.at * _segsum(miss, self.a_cell, self.count) / self.Wg
                out += add[:, None] if two else add
            elif prm.view in (FILL, MIX):
                f = fbar[self.a_cell]
                X = (tha * self.a_D)[:, None] * f if two else tha * self.a_D * f
                if prm.view == MIX:   # w2 reads the working set, w1 the fill
                    w2 = ~self.is_w1[self.a_cell]
                    Xw = tha * self.a_ws
                    X = np.where(w2[:, None], Xw[:, None], X) if two else np.where(w2, Xw, X)
                miss = (self.a_k[:, None] if two else self.a_k) * (
                    1.0 - sigma(X, prm.C_A * MIB, prm.beta_A))
                s = _segsum(miss, self.a_cell, self.count)
                out += s * (self.at / self.Wg)[:, None] if two else s * self.at / self.Wg
            else:
                raise ValueError(f"no view {prm.view!r}; the views are {VIEWS}")
        return out

    def solve(self, prm: Params, *, high: bool = False) -> np.ndarray:
        """The least fixed point of every cell (`high`: the greatest), by a
        97-point scan of g(q) = F(q) - q over [q_min, q_max] and 60 bisections
        of the bracketing interval: `wavecap.predict`, one cell per row."""
        c = self.count
        lo_q, hi_q = self.q_min, self.q_max
        step = (hi_q - lo_q) / (SCAN_POINTS - 1)
        grid = lo_q[:, None] + np.arange(SCAN_POINTS)[None, :] * step[:, None]
        grid[:, -1] = hi_q
        g = self.F(prm, grid) - grid
        rows = np.arange(c)
        out = np.full(c, np.nan)
        lo, hi = np.zeros(c), np.zeros(c)
        if not high:
            hit = g <= 0
            any_hit = hit.any(axis=1)
            i = np.argmax(hit, axis=1)
            out[~any_hit] = hi_q[~any_hit]
            zero = any_hit & (i == 0)
            out[zero] = grid[zero, 0]
            need = any_hit & (i > 0)
            lo[need] = grid[rows[need], i[need] - 1]
            hi[need] = grid[rows[need], i[need]]
        else:
            hit = g >= 0
            any_hit = hit.any(axis=1)
            i = SCAN_POINTS - 1 - np.argmax(hit[:, ::-1], axis=1)
            last = any_hit & (i == SCAN_POINTS - 1)
            out[last] = grid[last, -1]
            need = any_hit & (i < SCAN_POINTS - 1)
            lo[need] = grid[rows[need], i[need]]
            hi[need] = grid[rows[need], np.minimum(i[need] + 1, SCAN_POINTS - 1)]
        if need.any():
            for _ in range(BISECT_STEPS):
                mid = 0.5 * (lo + hi)
                up = self.F(prm, np.where(need, mid, lo_q)) - mid > 0
                lo = np.where(need & up, mid, lo)
                hi = np.where(need & ~up, mid, hi)
            out[need] = 0.5 * (lo + hi)[need]
        return out

    def kappa(self, prm: Params, q: np.ndarray) -> np.ndarray:
        """kappa = 1 / (1 - F'(q*)), F' by central difference (the judge's
        j07); infinite where F' >= 1."""
        h = KAPPA_STEP
        fp = (self.F(prm, q + h) - self.F(prm, np.maximum(q - h, 1e-6))) / (2 * h)
        with np.errstate(divide="ignore"):
            return np.where(fp < 1.0, 1.0 / (1.0 - fp), np.inf)


# --------------------------------------------------------------------------
# One card: pages, measured q, the model at any cell, the guard, the domain.
# --------------------------------------------------------------------------

@dataclass
class Card:
    """One card's pages, read and checked, and its measured q."""

    geom: Geometry
    pages: dict            # G -> page dict
    paths: dict            # G -> path
    #: (arm, G, n, gemm) -> measured q (`DCR.r3_q`)
    measured: dict
    #: (arm, G, n, gemm) -> per-call q (`DCR.r3_call_values`), may be empty
    calls: dict
    treads: tuple

    @property
    def label(self) -> str:
        return self.geom.card

    def run_ids(self) -> dict:
        return {int(G): p.get("run_id") for G, p in sorted(self.pages.items())}


def _pages_in(root: Path) -> list[Path]:
    found = sorted(root.glob("r3c-g*.json")) if root.is_dir() else [root]
    if not found and root.is_dir():
        found = sorted(root.rglob("r3c-g*.json"))
        dirs = {p.parent for p in found}
        if len(dirs) > 1:
            raise Refused(f"{root} holds r3c pages in {len(dirs)} directories; name one "
                          "card's directory (cards are never pooled)")
    if not found:
        raise Refused(f"no r3c-g*.json page under {root}")
    return found


#: The design keys every page of one card must share (G aside).
DESIGN_KEYS = ("model", "dtype", "block_m", "block_n", "block_k", "num_warps", "num_stages",
               "declared_by_arm")


def load_card(root: Path) -> Card:
    """One card's r3c pages: same card, same kernel (G aside), one page per G,
    every grid the walk's, W_c the same on every page."""
    paths = _pages_in(Path(root))
    pages, where = {}, {}
    geom = None
    for path in paths:
        page = json.loads(path.read_text())
        if page.get("family") not in (None, "r3-arms", "r3_arms"):
            raise Refused(f"{path}: family {page.get('family')!r} is not an r3-arms page")
        G = int(page["design"]["group_m"])
        if G in pages:
            raise Refused(f"{root}: two pages at G={G} ({where[G]}, {path}); name one")
        g = page_geometry(page)
        if geom is None:
            geom = g
            design0 = {k: page["design"].get(k) for k in DESIGN_KEYS}
        elif g != geom:
            raise Refused(f"{path}: card or window differs from {paths[0]} "
                          f"({g.card} W_c {dict(g.W_c)} against {geom.card} "
                          f"{dict(geom.W_c)}); one directory is one card")
        elif {k: page["design"].get(k) for k in DESIGN_KEYS} != design0:
            raise Refused(f"{path}: the kernel differs from {paths[0]}; one fit is one kernel")
        pages[G], where[G] = page, path
    measured, calls, treads = {}, {}, set()
    for G, page in pages.items():
        bm = DCR.r3_byte_model(MODEL_CONFIGS[geom.model], geom.dtype, geom.block_m)
        q = DCR.r3_q(page, bm)
        op = {"w1": bm["operand_per_tile_w1"], "w2": bm["operand_per_tile_w2"]}
        for c in page["cells"]:
            arm, n = str(c["arm"]), int(c["n"])
            treads.add(n)
            for gemm in GEMMS:
                grid = (c.get("per_gemm") or {}).get(gemm, {}).get("grid_size")
                want = geom.num_pid_m(arm, n) * geom.get("P", gemm)
                if grid is None or int(grid) != want:
                    raise Refused(f"{where[G]} {arm} n={n} {gemm}: grid {grid}, the walk's "
                                  f"{want} (num_pid_m {geom.num_pid_m(arm, n)} x P "
                                  f"{geom.get('P', gemm)}); the walk would not be this launch")
                measured[(arm, G, n, gemm)] = float(q[arm][gemm][n])
                calls[(arm, G, n, gemm)] = [(v - n * op[gemm]) / geom.get("W", gemm)
                                            for v in DCR.r3_call_values(c, gemm)]
    return Card(geom=geom, pages=pages, paths=where, measured=measured, calls=calls,
                treads=tuple(sorted(treads)))


class Model:
    """The WSC model on one card's geometry: events cached per cell."""

    def __init__(self, geom: Geometry):
        self.geom = geom
        self._events: dict = {}

    def events(self, arm: str, G: int, n: int, gemm: str) -> Events:
        key = (arm, G, n, gemm)
        if key not in self._events:
            self._events[key] = cell_events(self.geom, arm, G, n, gemm)
        return self._events[key]

    def batch(self, cells) -> Batch:
        return Batch(self.events(*c) for c in cells)

    def q(self, prm: Params, cells, *, high: bool = False) -> np.ndarray:
        """The least root (`high`: the greatest) at each (arm, G, n, gemm), with
        no guard: the raw number the guard judges."""
        return self.batch(cells).solve(prm, high=high)

    def domain(self, arm: str, G: int, n: int, gemm: str) -> str | None:
        """OUT_OF_DOMAIN or ILL_POSED by rule, with no evaluation; None when
        neither rule applies."""
        if gemm == "w2" and arm != "private" and G >= OUT_OF_DOMAIN_G:
            return OUT_OF_DOMAIN
        if (self.geom.hopper and gemm == "w2" and arm != "private" and G == 1
                and n >= HOPPER_G1_N):
            return ILL_POSED
        return None

    def evaluate(self, prm: Params, cells) -> list[dict]:
        """Every cell with its guard: q (least root), q_high, kappa, whether the
        guard fires and why, and the domain rule. `q` stays the raw least root
        (the scores of the judge's sets use it); a printed number must go
        through `shown`."""
        cells = list(cells)
        extra = []
        for arm, G, n, gemm in cells:
            if G == 1:
                extra += [(arm, 1, m, gemm) for m in range(1, n)]
        allc = list(dict.fromkeys(cells + extra))
        b = self.batch(allc)
        q = b.solve(prm)
        qh = b.solve(prm, high=True)
        kap = b.kappa(prm, q)
        at = {c: i for i, c in enumerate(allc)}
        out = []
        for c in cells:
            i = at[c]
            arm, G, n, gemm = c
            why = []
            if kap[i] > KAPPA_MAX:
                why.append(f"kappa {kap[i]:.1f} > {KAPPA_MAX:g}")
            if abs(qh[i] - q[i]) > ROOT_GAP_MAX:
                why.append(f"roots {q[i]:.3f} and {qh[i]:.3f}")
            if G == 1 and n > 1:
                below = [(q[at[(arm, 1, m, gemm)]], m) for m in range(1, n)]
                top, m_top = max(below)
                if q[i] < top:
                    why.append(f"monotonicity: {q[i]:.3f} < {top:.3f} at n={m_top}")
            rule = self.domain(arm, G, n, gemm)
            out.append({"cell": c, "q": float(q[i]), "q_high": float(qh[i]),
                        "kappa": float(kap[i]), "guard": why, "rule": rule,
                        "ill": bool(why) or rule == ILL_POSED})
        return out


def shown(r: dict | None) -> float | None:
    """The number a cell may print: None when it is ILL-POSED."""
    return None if r is None or r["ill"] else r["q"]


# --------------------------------------------------------------------------
# Fitting, per card, per view, never pooled.
# --------------------------------------------------------------------------

def nelder_mead(f, x0, step, iters: int = 500, tol: float = 1e-10):
    """The research fit's simplex (reflection 1, expansion 2, inside
    contraction 1/2, shrink 1/2), kept as it was so a refit lands where the
    reference fit did. Returns (x, f(x))."""
    x0 = np.asarray(x0, dtype=float)
    k = len(x0)
    pts = [x0] + [x0 + np.eye(k)[i] * step[i] for i in range(k)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol * (1 + abs(vals[0])):
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


#: A theta or eps closer than this to 0 or 1 sits at the edge of its range
#: ("at a bound"): the logit ran off toward infinity and the value is where
#: the simplex stopped. s07's GH200 eps1_w1 stopped at 7e-8.
EDGE = 1e-4
#: A theta or eps whose logit is past this sits at a bound too: the simplex
#: stops wherever the loss goes flat, which can be well inside EDGE. The GH200
#: ws view's theta1_A stopped at 0.00125 (logit -6.7) with its stage-1 SSE the
#: same to nine decimals at 0 (review F8, 2026-09-26).
LOGIT_EDGE = 6.0
#: And, where the card's cells are at hand, a theta or eps at a bound in effect
#: is one whose stage SSE moves by less than this, relative, when it is set to
#: that bound: the fit could not tell it from the bound.
FLAT_REL = 1e-6


@dataclass
class StageFit:
    names: tuple
    values: tuple
    sse: float
    cells: int
    evaluations: int


@dataclass
class ViewFit:
    """One card, one view: both stages and the flags."""

    view: str
    params: Params
    stage1: StageFit
    stage2: StageFit
    flags: list = field(default_factory=list)


def _run_stage(model: Model, cells, meas, pack, x0, step, starts: int = FIT_STARTS,
               iters: int = 500) -> tuple[np.ndarray, float, int]:
    """s07's `fit`: `starts` simplex runs from x0 and seeded draws around it,
    each polished by a second run at a third of the step; the lowest SSE of
    (q_pred / q_meas - 1) wins."""
    b = model.batch(cells)
    y = np.array(meas, dtype=float)
    count = [0]

    def sse(v):
        count[0] += 1
        return float(np.sum((b.solve(pack(v)) / y - 1.0) ** 2))

    rng = np.random.default_rng(0)
    best = None
    for s in range(starts):
        xs = np.array(x0, dtype=float) + (0 if s == 0 else rng.normal(0, START_SPREAD, len(x0)))
        v, val = nelder_mead(sse, xs, step, iters=iters)
        v, val = nelder_mead(sse, v, [st / 3 for st in step], iters=iters)
        if best is None or val < best[1]:
            best = (v, val)
    return best[0], best[1], count[0]


def stage1_cells(card: Card) -> list:
    """PRIVATE, every G, every tread, both GEMMs."""
    return [("private", G, n, g) for g in GEMMS for G in sorted(card.pages)
            for n in card.treads if ("private", G, n, g) in card.measured]


def stage2_cells(card: Card) -> list:
    """SHARED and NATIVE at G in STAGE2_G, every tread, both GEMMs."""
    return [(a, G, n, g) for g in GEMMS for a in ("shared", "native") for G in STAGE2_G
            for n in card.treads if (a, G, n, g) in card.measured]


def fit_view(card: Card, view: str, model: Model | None = None) -> ViewFit:
    """Stage 1 then stage 2 on one card, in one view."""
    model = model or Model(card.geom)
    l2 = card.geom.l2_bytes / MIB
    c1 = stage1_cells(card)
    if not c1:
        raise Refused(f"{card.label}: no PRIVATE cells; stage 1 has nothing to fit")

    def pack1(v):
        return Params(C_A=math.exp(v[0]), beta_A=math.exp(v[1]), theta1_A=_sig(v[2]),
                      C_B=1e9, beta_B=1.0, theta1=1.0, eps1_w1=0.0, eps1_w2=0.0, view=view)

    v1, s1, e1 = _run_stage(model, c1, [card.measured[c] for c in c1], pack1,
                            [math.log(2 * l2), math.log(2.0), _logit(0.3)], [0.4, 0.3, 0.8])
    A = pack1(v1)
    c2 = stage2_cells(card)
    if not c2:
        raise Refused(f"{card.label}: no SHARED/NATIVE cells at G in {STAGE2_G}")

    def pack2(v):
        return A.replace(C_B=math.exp(v[0]), beta_B=math.exp(v[1]), theta1=_sig(v[2]),
                         eps1_w1=_sig(v[3]), eps1_w2=_sig(v[4]))

    v2, s2, e2 = _run_stage(model, c2, [card.measured[c] for c in c2], pack2,
                            [math.log(l2), math.log(2.0), _logit(0.4), _logit(0.05),
                             _logit(0.1)], [0.4, 0.3, 0.8, 0.8, 0.8])
    prm = pack2(v2)
    fit = ViewFit(view=view, params=prm,
                  stage1=StageFit(STAGE1_NAMES, tuple(getattr(A, k) for k in STAGE1_NAMES),
                                  s1, len(c1), e1),
                  stage2=StageFit(STAGE2_NAMES, tuple(getattr(prm, k) for k in STAGE2_NAMES),
                                  s2, len(c2), e2))
    fit.flags = param_flags(prm, card.geom, card, model)
    return fit


#: The stage whose SSE judges each theta and eps (the stage that fits it).
UNIT_PARAMS = (("theta1_A", 1), ("theta1", 2), ("eps1_w1", 2), ("eps1_w2", 2))


def stage_sse(model: Model, card: Card, prm: Params, stage: int) -> float:
    """The loss a stage minimised, at `prm`: sum of (q / q_meas - 1)^2 over
    the stage's cells. Stage 1's cells are PRIVATE, which re-reads no slab, so
    the slab parameters in `prm` do not move it."""
    cells = stage1_cells(card) if stage == 1 else stage2_cells(card)
    y = np.array([card.measured[c] for c in cells], dtype=float)
    return float(np.sum((model.q(prm, cells) / y - 1.0) ** 2))


def param_flags(prm: Params, geom: Geometry, card: Card | None = None,
                model: Model | None = None) -> list[str]:
    """"slab law NOT IDENTIFIED" (beta_B > 8 or C_B < 0.25 x L2), and every
    parameter at the edge of its range. A theta or eps is at a bound when it
    is within EDGE of it, when its logit is past LOGIT_EDGE, or (with the
    card's cells) when setting it to the bound moves its stage SSE by less
    than FLAT_REL relative; it is UNCONSTRAINED when both bounds leave the SSE
    where it is (then no bound is named)."""
    l2 = geom.l2_bytes / MIB
    out = []
    if prm.beta_B > SLAB_BETA_MAX or prm.C_B < SLAB_C_MIN_L2 * l2:
        out.append(f"slab law NOT IDENTIFIED (C_B {prm.C_B:.2f} MiB = {prm.C_B / l2:.3f} x L2, "
                   f"beta_B {prm.beta_B:.2f}; flagged at beta_B > {SLAB_BETA_MAX:g} or C_B < "
                   f"{SLAB_C_MIN_L2:g} x L2)")
    if card is not None and model is None:
        model = Model(geom)
    base = {}
    for name, stage in UNIT_PARAMS:
        v = getattr(prm, name)
        flat = {}
        if card is not None:
            if stage not in base:
                base[stage] = stage_sse(model, card, prm, stage)
            s = base[stage]
            for end in (0.0, 1.0):
                moved = abs(stage_sse(model, card, prm.replace(**{name: end}), stage) - s)
                flat[end] = (moved <= FLAT_REL * max(s, 1e-300), moved / max(s, 1e-300))
        z = _logit(min(max(v, 1e-300), 1 - 1e-16))
        if flat and flat[0.0][0] and flat[1.0][0]:
            out.append(f"{name} UNCONSTRAINED ({v:.3g}): its stage-{stage} SSE moves by "
                       f"{flat[0.0][1]:.1e} / {flat[1.0][1]:.1e} relative at 0 / 1")
            continue
        for end, near, past in ((0.0, v < EDGE, z < -LOGIT_EDGE),
                                (1.0, v > 1 - EDGE, z > LOGIT_EDGE)):
            level = bool(flat) and flat[end][0]
            if not (near or past or level):
                continue
            why = [f"within {EDGE:g} of {end:g}"] if near else (
                [f"logit {z:+.1f} past {'-' if end == 0 else '+'}{LOGIT_EDGE:g}"] if past else [])
            if flat:
                why.append(f"its stage-{stage} SSE moves by {flat[end][1]:.1e} relative at "
                           f"{end:g}")
            out.append(f"{name} at bound {end:g} ({v:.2g}; {'; '.join(why)})")
            break
    for name in ("C_A", "C_B"):
        v = getattr(prm, name)
        if not l2 / C_SPAN <= v <= l2 * C_SPAN:
            out.append(f"{name} ran outside [L2/{C_SPAN:g}, {C_SPAN:g} x L2] ({v:.3g} MiB)")
    for name in ("beta_A", "beta_B"):
        v = getattr(prm, name)
        if not BETA_LO <= v <= BETA_HI:
            out.append(f"{name} ran outside [{BETA_LO:g}, {BETA_HI:g}] ({v:.3g})")
    return out


# --------------------------------------------------------------------------
# What the fit says about the card's cells.
# --------------------------------------------------------------------------

def group_q(geom: Geometry, arm: str, G: int, n: int) -> float:
    """The plain group model: `DCR.group_reads` (n for PRIVATE)."""
    return float(n) if arm == "private" else DCR.group_reads(geom.experts, n, G)


def role(card: Card | None, arm: str, G: int, n: int, gemm: str) -> str:
    """FIT (a stage's cell), HELD-OUT (a page cell no stage saw) or PREDICTED
    (no page measured it)."""
    if card is None or (arm, G, n, gemm) not in card.measured:
        return PREDICTED
    if arm == "private" or G in STAGE2_G:
        return FIT
    return HELD_OUT


def status_of(base: str, r: dict) -> str:
    """ILL-POSED over OUT-OF-DOMAIN over the cell's role."""
    if r["ill"]:
        return ILL_POSED
    if r["rule"] == OUT_OF_DOMAIN:
        return OUT_OF_DOMAIN
    return base


@dataclass
class CardResult:
    """One card's fit and every number printed from it."""

    card: Card
    model: Model
    fits: dict                          # view -> ViewFit (or a pinned ViewFit)
    rows: list = field(default_factory=list)       # page cells
    predicted: list = field(default_factory=list)  # cells no page measured

    @property
    def label(self) -> str:
        return self.card.label

    def params(self, view: str) -> Params:
        return self.fits[view].params


def page_cells(card: Card) -> list:
    return sorted(card.measured, key=lambda c: (GEMMS.index(c[3]), ARMS.index(c[0]), c[1], c[2]))


def _view_numbers(rf: dict, rw: dict | None, rm: dict | None = None) -> dict:
    """A row's model numbers. q_fill and q_ws are the numbers a cell may
    print, None (null in --out's JSON) where that view is ILL-POSED or was not
    given; the raw roots the guard judged live under their own names,
    least_root_* and greatest_root_fill, and are diagnostics: an ILL-POSED
    cell's least root is the spurious branch the guard exists for (GH200 w2
    G=1 n=8's 1.433, 94% of re-reads avoided), never a prediction. The judge's
    comparison sets are scored on least_root_*, as the judge scored them (review
    F1, 2026-09-26: the JSON had carried that root under q_fill)."""
    return {"q_fill": shown(rf), "q_ws": shown(rw), "q_mix": shown(rm),
            "least_root_fill": rf["q"], "least_root_ws": rw["q"] if rw else None,
            "least_root_mix": rm["q"] if rm else None,
            "greatest_root_fill": rf["q_high"], "kappa": rf["kappa"], "guard": rf["guard"],
            "ill_fill": rf["ill"], "ill_ws": bool(rw and rw["ill"]),
            "ill_mix": bool(rm and rm["ill"]), "rule": rf["rule"]}


def evaluate_card(res: CardResult) -> None:
    """Fill `res.rows` (every page cell) and `res.predicted` (the registered
    predictions), both views, guard and status per cell."""
    card, model = res.card, res.model
    cells = page_cells(card)
    ev = {v: dict(zip(cells, model.evaluate(res.params(v), cells), strict=True))
          for v in VIEWS if v in res.fits}
    res.rows = []
    for c in cells:
        arm, G, n, gemm = c
        rf = ev[FILL][c]
        rw = ev.get(WS, {}).get(c)
        rm = ev.get(MIX, {}).get(c)
        res.rows.append({"arm": arm, "G": G, "n": n, "gemm": gemm,
                         "measured": card.measured[c], **_view_numbers(rf, rw, rm),
                         "q_group": group_q(card.geom, arm, G, n), "role": role(card, *c),
                         "status": status_of(role(card, *c), rf),
                         "calls": card.calls.get(c) or []})
    gs = sorted(set(card.pages) | set(PREDICT_G))
    ns = sorted(set(card.treads) | set(PREDICT_N))
    want = [(a, G, n, g) for g in GEMMS for a in ("shared", "private") for G in gs for n in ns
            if (a, G, n, g) not in card.measured]
    ev = {v: dict(zip(want, model.evaluate(res.params(v), want), strict=True))
          for v in VIEWS if v in res.fits}
    res.predicted = []
    for c in want:
        rf = ev[FILL][c]
        rw = ev.get(WS, {}).get(c)
        rm = ev.get(MIX, {}).get(c)
        res.predicted.append({"arm": c[0], "G": c[1], "n": c[2], "gemm": c[3],
                              **_view_numbers(rf, rw, rm),
                              "q_group": group_q(card.geom, c[0], c[1], c[2]),
                              "status": status_of(PREDICTED, rf)})


def _rms_worst(rel) -> tuple[float, float]:
    a = np.asarray(rel, dtype=float)
    if not a.size:
        return math.nan, math.nan
    i = int(np.argmax(np.abs(a)))
    return float(np.sqrt(np.mean(a ** 2))), float(a[i])


def _total(res: CardResult, key: str, arm: str, G: int, n: int) -> float | None:
    geom = res.card.geom
    by = {r["gemm"]: r for r in res.rows if (r["arm"], r["G"], r["n"]) == (arm, G, n)}
    if set(by) != set(GEMMS):
        return None
    w = {g: geom.get("W", g) for g in GEMMS}
    return sum(by[g][key] * w[g] for g in GEMMS) / sum(w.values())


def scores(res: CardResult) -> dict:
    """The MODEL'S OWN scores (in domain, well posed) and the JUDGE'S
    COMPARISON SETS (which include OUT-OF-DOMAIN cells and ILL-POSED least
    roots), per view, beside the group model on the same cells. n >= 2, as
    the judge's; relative errors (model / measured - 1)."""
    rows = [r for r in res.rows if r["n"] >= 2]
    out = {"own": {}, "judge": {}}
    views = [v for v in VIEWS if v in res.fits]

    def block(sel, *, own: bool):
        entry = {}
        for v in views:
            ill = f"ill_{v}"
            rs = [r for r in rows if sel(r)
                  and (not own or (not r[ill] and r["status"] != OUT_OF_DOMAIN))]
            # The model's own score reads the printable q (never None here,
            # ILL-POSED cells being left out); the judge's sets read the raw
            # least root, ILL-POSED cells included, as the judge scored them.
            key = f"q_{v}" if own else f"least_root_{v}"
            rel = [r[key] / r["measured"] - 1 for r in rs]
            grel = [r["q_group"] / r["measured"] - 1 for r in rs]
            rms, worst = _rms_worst(rel)
            grms, gworst = _rms_worst(grel)
            entry[v] = {"cells": len(rs), "rms": rms, "worst": worst, "group_rms": grms,
                        "group_worst": gworst,
                        "out_of_domain": sum(r["status"] == OUT_OF_DOMAIN for r in rs),
                        "ill_posed": sum(bool(r[ill]) for r in rs)}
        return entry

    out["own"]["stage 1 fit: PRIVATE"] = block(lambda r: r["arm"] == "private", own=True)
    out["own"]["stage 2 fit: SHARED+NATIVE G<=4"] = block(
        lambda r: r["arm"] != "private" and r["G"] in STAGE2_G, own=True)
    out["own"]["HELD-OUT: SHARED+NATIVE G=16, 64"] = block(
        lambda r: r["arm"] != "private" and r["G"] not in STAGE2_G, own=True)
    out["own"]["every in-domain, well-posed cell"] = block(lambda r: True, own=True)
    out["judge"]["120 cells: S+N+P, w1+w2"] = block(lambda r: True, own=False)
    out["judge"]["w2 S+N, G<=16"] = block(
        lambda r: r["gemm"] == "w2" and r["arm"] != "private" and r["G"] <= 16, own=False)
    out["judge"]["w2 PRIVATE"] = block(lambda r: r["gemm"] == "w2" and r["arm"] == "private",
                                       own=False)
    tot = {}
    for v in views:
        rel, grel = [], []
        for arm in ARMS:
            for G in sorted(res.card.pages):
                for n in res.card.treads:
                    if n < 2:
                        continue
                    m = _total(res, "measured", arm, G, n)
                    if m is None:
                        continue
                    rel.append(_total(res, f"least_root_{v}", arm, G, n) / m - 1)
                    grel.append(_total(res, "q_group", arm, G, n) / m - 1)
        rms, worst = _rms_worst(rel)
        grms, gworst = _rms_worst(grel)
        tot[v] = {"cells": len(rel), "rms": rms, "worst": worst, "group_rms": grms,
                  "group_worst": gworst}
    out["judge"]["total q, S+N+P (what timing uses)"] = tot
    return out


def decomposition(res: CardResult) -> list[dict]:
    """w2 SHARED at G in {1, 2, 4}, n >= 2, fill view, in units of W_w2:
    q = group - x recovered in the first window - x recovered after it +
    first-window partner loss + activation. Every term from the fitted
    parameters (s10 section 3); ILL-POSED cells carry no numbers."""
    prm = res.params(FILL)
    out = []
    for r in res.rows:
        if r["gemm"] != "w2" or r["arm"] != "shared" or r["G"] not in STAGE2_G or r["n"] < 2:
            continue
        ev = res.model.events("shared", r["G"], r["n"], "w2")
        d = {"G": r["G"], "n": r["n"], "measured": r["measured"], "status": r["status"]}
        if r["ill_fill"]:
            out.append(d)
            continue
        q = r["q_fill"]
        S = ev.per_set
        fbar = q * ev.weight_bytes / ev.live
        CB = prm.C_B * MIB
        w1 = ev.x_win1
        rec1 = float(np.sum(ev.x_k[w1] * sigma(prm.theta1 * ev.x_D[w1] * fbar, CB, prm.beta_B)) / S)
        recl = float(np.sum(ev.x_k[~w1] * sigma(ev.x_D[~w1] * fbar, CB, prm.beta_B)) / S)
        tha = np.where(ev.a_win1, prm.theta1_A, 1.0)
        X = ev.a_D * fbar if prm.view == FILL else ev.a_ws
        act = float(ev.atile * np.sum(ev.a_k * (1 - sigma(tha * X, prm.C_A * MIB, prm.beta_A)))
                    / ev.weight_bytes)
        d.update({"q": q, "group": r["q_group"], "partner_loss_w1": prm.eps1_w2 * ev.in_win1 / S,
                  "x_w1": float(ev.x_k[w1].sum() / S), "recovered_w1": rec1,
                  "x_later": float(ev.x_k[~w1].sum() / S), "recovered_later": recl,
                  "activation": act})
        out.append(d)
    return out


def first_wave(res: CardResult) -> dict | None:
    """s1 = max(0, W_c,w1 - P_w1) / (E P_w1): at G=1 on w1, the slab re-reads
    (M-tile 1 re-reading M-tile 0's slabs, P_w1 CTAs later) that launch inside
    the first co-residency window, per weight set. The measured recovery
    phi = (n - q_S,w1(G=1, n)) / s1 per tread says what fraction of them L2
    served; None without a G=1 page."""
    card, geom = res.card, res.card.geom
    if 1 not in card.pages:
        return None
    P, Wc = geom.get("P", "w1"), geom.get("W_c", "w1")
    s1 = max(0, Wc - P) / (geom.experts * P)
    ev = res.model.events("shared", 1, 2, "w1")
    in_window = int(ev.x_k[ev.x_win1].sum())
    per_n = []
    for n in card.treads:
        if n < 2 or ("shared", 1, n, "w1") not in card.measured:
            continue
        m = card.measured[("shared", 1, n, "w1")]
        row = next(r for r in res.rows if (r["arm"], r["G"], r["n"], r["gemm"])
                   == ("shared", 1, n, "w1"))
        saving = n - m
        per_n.append({"n": n, "measured_q": m, "saving": saving,
                      "saved_slab_reads": saving * geom.experts * P,
                      "phi": saving / s1 if s1 > 0 else None,
                      "model_saving_fill": None if row["q_fill"] is None else n - row["q_fill"]})
    return {"W_c_w1": Wc, "P_w1": P, "s1": s1, "events_in_window": in_window,
            "per_n": per_n}


def cross_card(results: list[CardResult]) -> list[dict]:
    """THE CROSS-CARD L2 CHECK, for every pair of cards with one kernel, one SM
    count and one co-residency window but different L2: (smaller-L2 card
    minus larger-L2 card) of w2 SHARED q, measured beside modeled (each
    card's own fit, fill view), at G in {2, 4, 16}. A comparison of two
    per-card models; nothing is pooled."""
    out = []
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            ga, gb = a.card.geom, b.card.geom
            same = (ga.sm_count, ga.W_c, ga.model, ga.dtype, ga.block_m, ga.block_n) == \
                   (gb.sm_count, gb.W_c, gb.model, gb.dtype, gb.block_m, gb.block_n)
            if not same or ga.l2_bytes == gb.l2_bytes:
                continue
            lo, hi = (a, b) if ga.l2_bytes < gb.l2_bytes else (b, a)
            rows = []
            for G in CROSS_CARD_G:
                for n in sorted(set(lo.card.treads) & set(hi.card.treads)):
                    c = ("shared", G, n, "w2")
                    if n < 2 or c not in lo.card.measured or c not in hi.card.measured:
                        continue
                    rl = next(r for r in lo.rows if (r["arm"], r["G"], r["n"], r["gemm"]) == c)
                    rh = next(r for r in hi.rows if (r["arm"], r["G"], r["n"], r["gemm"]) == c)
                    if rl["ill_fill"] or rh["ill_fill"]:
                        continue
                    meas = rl["measured"] - rh["measured"]
                    mod = rl["q_fill"] - rh["q_fill"]
                    rows.append({"G": G, "n": n, "measured": meas, "model": mod,
                                 "gap": mod - meas})
            worst = max((abs(r["gap"]) for r in rows), default=math.nan)
            out.append({"minus": hi.label, "card": lo.label, "rows": rows, "worst_gap": worst,
                        "tolerance": CROSS_CARD_TOL,
                        "verdict": "PASS" if rows and worst <= CROSS_CARD_TOL else "FAIL"})
    return out


# --------------------------------------------------------------------------
# Parameters in and out, and the q source `per_tile_model_fit` consumes.
# --------------------------------------------------------------------------

def _page_ref(path: Path, relative_to: Path | None) -> str:
    """A counter page's path as a params JSON names it: relative to the
    directory the JSON is written into when that is known, absolute otherwise.
    Never relative to the cwd: `q_source` resolves a relative path against the
    JSON's own directory, so the same file gives the same q from any cwd
    (review F2, 2026-09-26: a cwd-relative path made a run from elsewhere drop
    silently to model-only q)."""
    p = Path(path).resolve()
    if relative_to is None:
        return str(p)
    return os.path.relpath(p, Path(relative_to).resolve())


def params_doc(res: CardResult, relative_to: Path | None = None) -> dict:
    """The params JSON of one card: run ids, W_c, L2, parameters per view,
    stage SSEs, flags, and the counter pages they came from (relative to
    `relative_to`, the directory the JSON goes into, when given; absolute
    otherwise)."""
    return {"tool": "scripts/wave_split_bytes.py", "status": "NOT FINAL",
            "card": res.label, "geometry": res.card.geom.as_json(),
            "counter_pages": {str(G): _page_ref(p, relative_to)
                              for G, p in sorted(res.card.paths.items())},
            "counter_pages_base": ("the directory of this JSON" if relative_to is not None
                                   else "absolute"),
            "run_ids": {str(G): r for G, r in res.card.run_ids().items()},
            "W_c": dict(res.card.geom.W_c), "l2_bytes": res.card.geom.l2_bytes,
            "registered_view": REGISTERED_VIEW,
            "views": {v: {"params": f.params.as_json(),
                          "stage1": dataclasses.asdict(f.stage1),
                          "stage2": dataclasses.asdict(f.stage2), "flags": f.flags}
                      for v, f in res.fits.items()}}


def pinned_fit(prm: Params, geom: Geometry, note: str = "pinned, not fitted",
               card: Card | None = None, model: Model | None = None) -> ViewFit:
    """A ViewFit around given parameters (no stage run), flags recomputed
    (against `card`'s cells when it is given)."""
    empty1 = StageFit(STAGE1_NAMES, tuple(getattr(prm, k) for k in STAGE1_NAMES), math.nan,
                      0, 0)
    empty2 = StageFit(STAGE2_NAMES, tuple(getattr(prm, k) for k in STAGE2_NAMES), math.nan,
                      0, 0)
    return ViewFit(view=prm.view, params=prm, stage1=empty1, stage2=empty2,
                   flags=[note] + param_flags(prm, geom, card, model))


def read_params(path: Path) -> dict:
    """A params JSON (`--params-out`), or a file of them under "cards":
    {card slug: {view: Params}} with the geometry each was fitted on."""
    doc = json.loads(Path(path).read_text())
    docs = list(doc["cards"].values()) if "cards" in doc else [doc]
    out = {}
    for d in docs:
        out[d["card"]] = {"geometry": Geometry.from_json(d["geometry"]),
                          "views": {v: Params.from_json(dict(x["params"], view=v))
                                    for v, x in d["views"].items()},
                          "counter_pages": d.get("counter_pages") or {}}
    return out


class QSource:
    """Per-GEMM q for a timed cell of one card: the counter page's measured q
    where it has the cell (and the timed page ran on the same physical card,
    by UUID), otherwise this model's q in the registered (fill) view. None
    where either GEMM is ILL-POSED: that cell has no byte number to time.

    It serves only a timed page that ran the kernel and declaration the byte
    model was fitted on (`refuses`): the walk's grid is num_pid_m x P_g, and
    both change with BLOCK_M, BLOCK_N and the declared expert slots, so q from
    one kernel says nothing about another's bytes (review F3, 2026-09-26: a
    timed page edited to BLOCK_M 64 took the BLOCK_M 32 counter q silently)."""

    #: Printed beside every column this source feeds (review F5).
    status = "NOT FINAL"

    def __init__(self, geom: Geometry, prm: Params, card: Card | None = None):
        self.geom, self.prm, self.card = geom, prm, card
        self.model = Model(geom)
        self._cache: dict = {}
        self.status_lines = tuple(STATUS_LINES)

    @property
    def label(self) -> str:
        return f"WSC-{self.prm.view}[{self.geom.card}]"

    def refuses(self, tile, declared_by_arm: dict, copies_declared) -> str | None:
        """Why a timed page may not take this source's q, or None. `tile` is
        `per_tile_model_fit.Page.tile` (model, dtype, BLOCK_M, BLOCK_N,
        BLOCK_K, num_warps, num_stages); `declared_by_arm` maps each loaded arm
        to the declared expert slots its rows record (cells.csv's
        experts_declared, a tuple of the distinct values); `copies_declared` is
        the page's report.json value."""
        want = self.geom.kernel
        if tile is None or tuple(tile) != want:
            return f"the timed page ran {tile}, the byte model's kernel is {want}"
        if copies_declared is None or int(copies_declared) != self.geom.copies_declared:
            return (f"the timed page declared {copies_declared} copies, the counter pages "
                    f"{self.geom.copies_declared}")
        slots = dict(self.geom.declared)
        for arm, got in sorted((declared_by_arm or {}).items()):
            if arm in slots and tuple(got) != (slots[arm],):
                return (f"the timed page's {arm} rows declare {list(got)} expert slots, the "
                        f"counter pages {slots[arm]}")
        return None

    def __call__(self, arm: str, G: int, n: int, device_uuid: str = ""):
        measured = (self.card is not None and device_uuid and device_uuid == self.geom.uuid
                    and all((arm, G, n, g) in self.card.measured for g in GEMMS))
        if measured:
            return (self.card.measured[(arm, G, n, "w1")], self.card.measured[(arm, G, n, "w2")],
                    "measured")
        key = (arm, G, n)
        if key not in self._cache:
            rs = self.model.evaluate(self.prm, [(arm, G, n, g) for g in GEMMS])
            if any(r["ill"] for r in rs):
                self._cache[key] = None
            else:
                ood = any(r["rule"] == OUT_OF_DOMAIN for r in rs)
                self._cache[key] = (rs[0]["q"], rs[1]["q"],
                                    self.label + (" OUT-OF-DOMAIN" if ood else ""))
        return self._cache[key]


def q_source(path: Path, *, view: str | None = None) -> QSource:
    """The q source of one params JSON: its card's parameters in `view`, and
    its counter pages (measured q). A relative page path is resolved against
    the JSON's own directory, never the cwd. A JSON that names counter pages
    which are not there, or whose directory now holds other pages, is REFUSED:
    dropping to the model's q would change what is fitted and print that the
    card has no counter page, which would be false (review F2)."""
    got = read_params(path)
    if len(got) != 1:
        raise Refused(f"{path} holds {len(got)} cards; one q source is one card")
    (slug, d), = got.items()
    base = Path(path).resolve().parent
    named = [p if p.is_absolute() else base / p for p in map(Path, d["counter_pages"].values())]
    card = None
    if named:
        missing = [str(p) for p in named if not p.exists()]
        if missing:
            raise Refused(f"{path} names counter pages that are not there: {missing}; move "
                          "them back or refit (a missing page is not a card without one)")
        dirs = {p.resolve().parent for p in named}
        if len(dirs) != 1:
            raise Refused(f"{path} names counter pages in {len(dirs)} directories; one card's "
                          "pages sit in one")
        card = load_card(dirs.pop())
        if {p.resolve() for p in card.paths.values()} != {p.resolve() for p in named}:
            raise Refused(f"{path}: the counter pages' directory now holds other pages than "
                          "the ones the parameters were fitted on")
        if card.geom != d["geometry"]:
            raise Refused(f"{path}: the counter pages it names now read a different "
                          "geometry than the one the parameters were fitted on")
    if view is None:   # the registered view, or fill in a JSON written before mix
        view = REGISTERED_VIEW if REGISTERED_VIEW in d["views"] else FILL
    return QSource(d["geometry"], d["views"][view], card)


# --------------------------------------------------------------------------
# The page.
# --------------------------------------------------------------------------

STATUS_LINES = [
    "STATUS: NOT FINAL. The best byte model the study has for G <= 16 and for PRIVATE; a",
    "  fixed-point defect, an unidentified activation distance and the missing dead-CTA",
    "  term are known structural gaps, not noise. WHAT WOULD FALSIFY IT (any one):",
    "  GH200 w2 SHARED at G=8, n = 2 or 4, above 1.12; n=5 or n=8 at G = 2, 4 or 16",
    "  outside +-5% of the registered predictions; a w1 G=1 saving (n - q) that changes",
    "  with n; an H100 against GH200 gap over 0.05 at G >= 2; a GH200 byte page with the",
    "  co-residency window halved where the n=2 over-read at G=2 does not shrink; n=5",
    "  time errors above 3% at G=1 or G=2 SHARED through the per-GEMM overlap map.",
    "  OPEN: the published GH200 1710 MHz lock pages, the ones the judge set that last",
    "  test on, already give -3.14% at G=2 SHARED (and -2.99% at G=1), past its 3% as",
    "  worded. The judge's report does not say whether it means pages not yet taken or",
    "  a looser threshold; it is read here as a test of new pages, the GH200 cell an",
    "  open question for the judge, not a pass.",
]


def _num(x, fmt="{:.3f}") -> str:
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def _pct(x) -> str:
    return "  n/a" if x is None or not math.isfinite(x) else f"{100 * x:+.2f}%"


def _rms(x) -> str:
    return " n/a" if x is None or not math.isfinite(x) else f"{100 * x:.2f}%"


def param_line(fit: ViewFit, geom: Geometry) -> str:
    p = fit.params
    l2 = geom.l2_bytes / MIB
    return (f"C_A {p.C_A:.2f} MiB ({p.C_A / l2:.3f} x L2), beta_A {p.beta_A:.3f}, "
            f"theta1_A {p.theta1_A:.3f}; C_B {p.C_B:.2f} MiB ({p.C_B / l2:.3f} x L2), "
            f"beta_B {p.beta_B:.3f}, theta1 {p.theta1:.3f}, eps1_w1 {p.eps1_w1:.4f}, "
            f"eps1_w2 {p.eps1_w2:.4f}")


def card_lines(res: CardResult, sc: dict) -> list[str]:
    card, geom = res.card, res.card.geom
    tag = f"[{res.label}]"
    out = [f"=== CARD {res.label} ({geom.name}, sm_{geom.capability}, {geom.sm_count} SMs, "
           f"L2 {geom.l2_bytes / MIB:.1f} MiB): its own fit, nothing pooled with another card",
           f"{tag} PAGES (stored gate verdicts that are not PASS; every page is fitted, as the "
           "judge's reproduction fitted them)"]
    for G, page in sorted(card.pages.items()):
        bad = [f"{g.get('number', g.get('name'))} {g.get('verdict')}"
               for g in page.get("gates", []) if g.get("verdict") != "PASS"]
        out.append(f"  G={G:<3} {page.get('run_id')}  {', '.join(bad) or 'all PASS'}")
    out.append(f"{tag} GEOMETRY (every value read off the pages)")
    for g in GEMMS:
        out.append(f"  {g}: num_pid_n P {geom.get('P', g)}, CTAs/SM {geom.get('ctas_per_sm', g)} "
                   f"-> W_c {geom.get('W_c', g)}, SLAB {geom.slab(g):,} B, ATILE "
                   f"{geom.atile(g):,} B, W_{g} {geom.get('W', g):,} B")
    out.append(f"  declared expert slots {dict(geom.declared)}; every grid is the walk's, and "
               "the pid walk agrees with the closed form on every cell walked")
    out.append(f"{tag} PARAMETERS (per view; the fill view is the registered default)")
    for v, f in res.fits.items():
        out.append(f"  {v:<4} {param_line(f, geom)}")
        s1, s2 = f.stage1, f.stage2
        if s1.cells:
            out.append(f"       stage 1 SSE {s1.sse:.4g} over {s1.cells} PRIVATE cells, stage 2 "
                       f"SSE {s2.sse:.4g} over {s2.cells} SHARED+NATIVE cells at G in "
                       f"{STAGE2_G} ({s1.evaluations + s2.evaluations} model evaluations)")
        for flag in f.flags:
            out.append(f"       FLAG {flag}")
    has_ws = WS in res.fits
    out.append(f"{tag} CELLS (q in units of W_g; '--' is ILL-POSED and prints no number, 'n/a' "
               "a view the parameters did not give; kappa and status of the fill view)")
    out.append(f"  {'arm':<8}{'G':>3}{'n':>3} {'g':<3}{'measured':>9}{'q_fill':>8}{'q_ws':>8}"
               f"{'q_group':>8}{'kappa':>7}  status")
    for r in res.rows:
        why = f"  ({'; '.join(r['guard'])})" if r["guard"] else ""
        if r["rule"] == ILL_POSED and not r["guard"]:
            why = "  (Hopper w2 G=1 n>=6 rule)"
        ws = _num(r["q_ws"]) if has_ws else "n/a"
        out.append(f"  {r['arm']:<8}{r['G']:>3}{r['n']:>3} {r['gemm']:<3}{r['measured']:>9.3f}"
                   f"{_num(r['q_fill']):>8}{ws:>8}{r['q_group']:>8.3f}"
                   f"{r['kappa']:>7.2f}  {r['status']}{why}")
    out.append(f"{tag} SCORES, n >= 2: rms / worst of (model / measured - 1), beside the group "
               "model on the same cells")
    out.append("  THE MODEL'S OWN (in domain and well posed only)")
    for name, entry in sc["own"].items():
        out.append(f"    {name:<40}" + "  ".join(
            f"{v} {_rms(e['rms'])} / {_pct(e['worst'])} ({e['cells']})" for v, e in entry.items())
            + f"  | group {_rms(entry[FILL]['group_rms'])}")
    out.append("  THE JUDGE'S COMPARISON SETS (reproduced for checking against the report; they "
               "INCLUDE OUT-OF-DOMAIN cells and the least roots of ILL-POSED cells)")
    for name, entry in sc["judge"].items():
        e0 = entry[FILL]
        extra = (f"; includes {e0.get('out_of_domain', 0)} OUT-OF-DOMAIN, "
                 f"{e0.get('ill_posed', 0)} ILL-POSED") if "out_of_domain" in e0 else ""
        out.append(f"    {name:<40}" + "  ".join(
            f"{v} {_rms(e['rms'])} / {_pct(e['worst'])} ({e['cells']})" for v, e in entry.items())
            + f"  | group {_rms(e0['group_rms'])}{extra}")
    out.append(f"{tag} W2 SHARED DECOMPOSITION, fill view, units of W_w2: q = group - "
               "recovered(first window) - recovered(later) + first-window partner loss + A "
               "[INTERPRETATION: each term is the model's reading, not a counter]")
    for d in decomposition(res):
        if "q" not in d:
            out.append(f"  G={d['G']} n={d['n']}: {d['status']}, no numbers  | measured "
                       f"{d['measured']:.3f}")
            continue
        out.append(f"  G={d['G']} n={d['n']}: {d['group']:.3f} - {d['recovered_w1']:.3f} "
                   f"(of {d['x_w1']:.3f}) - {d['recovered_later']:.3f} (of {d['x_later']:.3f})"
                   f" + {d['partner_loss_w1']:.3f} + {d['activation']:.3f} = {d['q']:.3f}"
                   f"  | measured {d['measured']:.3f}  (net vs group: model "
                   f"{d['q'] - d['group']:+.3f}, measured {d['measured'] - d['group']:+.3f})")
    fw = first_wave(res)
    if fw is not None:
        out.append(f"{tag} FIRST-WAVE STATISTIC: s1 = max(0, W_c,w1 - P_w1) / (E P_w1) = "
                   f"max(0, {fw['W_c_w1']} - {fw['P_w1']}) / {geom.experts * fw['P_w1']} = "
                   f"{fw['s1']:.5f} (w1 G=1 cross-group events inside the first window: "
                   f"{fw['events_in_window']})")
        for x in fw["per_n"]:
            phi = "n/a (s1 = 0: no re-read launches in the first window)" if x["phi"] is None \
                else f"{x['phi']:.3f}"
            ms = x["model_saving_fill"]
            out.append(f"  n={x['n']}: measured saving n - q {x['saving']:+.4f} "
                       f"({x['saved_slab_reads']:.0f} slab reads), recovery phi {phi}; model "
                       f"(fill) saving {'-- (ILL-POSED)' if ms is None else f'{ms:+.4f}'}")
    out.append(f"{tag} PREDICTED (cells no page measured): n: fill [ws] {{mix, the registered "
               "view}; ILL = ILL-POSED, no "
               "number; OOD = OUT-OF-DOMAIN, printed and never scored. NATIVE is not listed: "
               "its live CTAs run in SHARED's order at every G, so the model gives it SHARED's "
               "number (it has no dead-CTA term)")
    keyed: dict = {}
    for r in res.predicted:
        keyed.setdefault((r["gemm"], r["arm"], r["G"]), []).append(r)
    for (g, arm, G), rs in sorted(keyed.items(), key=lambda kv: (GEMMS.index(kv[0][0]),
                                                                  ARMS.index(kv[0][1]), kv[0][2])):
        parts = []
        for r in rs:
            if r["ill_fill"]:
                parts.append(f"n{r['n']} ILL")
                continue
            ws = "n/a" if not has_ws else "ILL" if r["ill_ws"] else _num(r["q_ws"])
            mx = "" if r.get("least_root_mix") is None else (
                " {ILL}" if r["ill_mix"] else f" {{{_num(r['q_mix'])}}}")
            ood = " OOD" if r["status"] == OUT_OF_DOMAIN else ""
            parts.append(f"n{r['n']} {r['q_fill']:.3f} [{ws}]{mx}{ood}")
        out.append(f"  {g} {arm:<7} G={G:<3} " + "  ".join(parts))
    return out


def cross_lines(checks: list[dict]) -> list[str]:
    out = []
    for c in checks:
        out.append(f"=== CROSS-CARD L2 CHECK: {c['card']} minus {c['minus']}, w2 SHARED "
                   "(two per-card fits compared; nothing pooled)")
        for r in c["rows"]:
            out.append(f"  G={r['G']:<3} n={r['n']}: measured {r['measured']:+.3f}, model "
                       f"{r['model']:+.3f}, gap {r['gap']:+.3f}")
        out.append(f"  worst |gap| {_num(c['worst_gap'])} against {c['tolerance']:g}: "
                   f"{c['verdict']}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit the wave-split capacity model of per-GEMM DRAM weight reads to "
                    "one card's r3-arms counter pages (NOT FINAL); cards are never pooled.")
    p.add_argument("inputs", nargs="+", type=Path,
                   help="one directory of r3c-g*.json pages per card")
    p.add_argument("--params", type=Path, default=None,
                   help="use these parameters (a --params-out JSON, or several under "
                        "\"cards\") instead of fitting; matched by card slug")
    p.add_argument("--params-out", type=Path, default=None,
                   help="write one <card>.wsc-params.json per card into this directory")
    p.add_argument("--out", type=Path, default=None,
                   help="write every card's cells, scores, decomposition, first-wave "
                        "statistic, predictions and the cross-card check as JSON")
    return p


def analyse(card: Card, *, params: dict | None = None) -> CardResult:
    """One card: fit both views (or take `params`, {view: Params}), then every
    printed number."""
    model = Model(card.geom)
    if params:
        fits = {v: pinned_fit(p, card.geom, card=card, model=model) for v, p in params.items()}
        if FILL not in fits:
            raise Refused(f"{card.label}: the given parameters have no fill view")
    else:
        fits = {v: fit_view(card, v, model) for v in VIEWS}
    res = CardResult(card=card, model=model, fits=fits)
    evaluate_card(res)
    return res


def run(args) -> tuple[list[str], dict]:
    cards = [load_card(p) for p in args.inputs]
    labels = [c.label for c in cards]
    if len(set(labels)) != len(labels):
        raise Refused(f"two inputs are one card ({labels}); a card is fitted once")
    pinned = read_params(args.params) if args.params else {}
    results = []
    for c in cards:
        if args.params and c.label not in pinned:
            raise Refused(f"{args.params} has no parameters for {c.label}")
        if c.label in pinned and pinned[c.label]["geometry"] != c.geom:
            raise Refused(f"{args.params}: {c.label}'s parameters were fitted on another "
                          "geometry than these pages read")
        results.append(analyse(c, params=pinned.get(c.label, {}).get("views")))
    lines = [f"wave_split_bytes: {len(results)} card(s), each fitted alone (cards are never "
             "pooled)", *STATUS_LINES, ""]
    doc = {"status": "NOT FINAL", "status_lines": list(STATUS_LINES), "cards": {},
           "keys": "q_fill / q_ws are null where that view is ILL-POSED (or not given); "
                   "least_root_* and greatest_root_fill are the raw roots the guard judged, "
                   "diagnostics and never predictions"}
    out_dir = args.out.parent if args.out is not None else None
    for res in results:
        sc = scores(res)
        lines += card_lines(res, sc) + [""]
        doc["cards"][res.label] = dict(params_doc(res, out_dir), cells=res.rows,
                                       predicted=res.predicted, scores=sc,
                                       decomposition=decomposition(res),
                                       first_wave=first_wave(res))
    checks = cross_card(results)
    lines += cross_lines(checks)
    doc["cross_card"] = checks
    if args.params_out:
        args.params_out.mkdir(parents=True, exist_ok=True)
        for res in results:
            path = args.params_out / f"{res.label}.wsc-params.json"
            path.write_text(json.dumps(params_doc(res, args.params_out), indent=1) + "\n")
            lines.append(f"wrote {path}")
    return lines, doc


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lines, doc = run(args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return exit_codes.REFUSED
    print("\n".join(lines))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(f"wrote {args.out}")
    return exit_codes.DONE


if __name__ == "__main__":
    sys.exit(main())
