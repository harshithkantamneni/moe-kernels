# Rental 2 scores (2026-10-02)

The four scorers in this directory, run unchanged at 38898b4 on
`results/published/2026-10-02-nvidia_gh200_480gb-rental2-session` (919 files, each matching
the run branch's SHA256SUMS):

    python scripts/scoring/rental2/score_<part>.py . results/published/2026-10-02-nvidia_gh200_480gb-rental2-session scripts/scoring/rental2

Outputs: `knobs`, `launch`, `const`, `w1floor` `.score.{json,txt}` beside the scorers. Each
JSON's `addendum.verdicts` lists every verdict path with its ALL, CLEAN and registered
reading (`docs/registered/2026-10-02-rental2-addendum-gates.txt`: the common verdict when the
two agree, INCONCLUSIVE when they differ, NOT SCORED with ALL's printed when CLEAN has no
data). No scorer crashed or refused. Two post-page scorer fixes changed printed verdicts to
what the registration's text says (end of this file).

## Pages each view drops

| page | failed validity gates | ALL | CLEAN |
|---|---|---|---|
| tp8 floor2, tp4, tp2 `r3f-g64-lock1710` | FL1 | counted | counted (null-kernel lock check 1684.4, 1687.5, 1687.9 MHz: -1.50, -1.32, -1.29%, within 3%) |
| tp8, tp4, tp2 `r3f-g64`, tp4 `r3f-g64-lock1005` | none | counted | counted |
| 8x7B l2s8, tp2 atbk32, tp2 atbk64, tp2 ats8 (`r3c`) | V7 (atbk64 also V10) | counted | excluded |
| tp2 l2base, l2s6, l2bk128, l2pad7; tp4 l2base, l2s8; tp8 l2s8 | V10, not gating in part 1 | counted | counted |
| tp2 l2bk32 | V6, flagged in both views (part 1's own rule) | counted | counted |
| tp2 l2s8, 8x7B l2base, tp2 atbk128 | none (atbk128: claim gates only) | counted | counted |

Above-lock rule: no floor cell's own `sm_clock_mhz` exceeds its lock by more than 15 MHz
(highest 1702.6 at 1710, 1004.9 at 1005); nothing is dropped. Part 2's directories carry no
gate: its two views agree by construction.

## Part 4, tp8's w1 floor (`2026-10-01-rental2-w1floor-gh200`), primary: base captures

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| theta call, tp8 w1 / tp4 w1 | H_EST / H_EST | same | H_EST side on both | theta +0.544 (sigma 0.276), +0.797 (sigma 0.200) |
| H_EST | FALSIFIED | FALSIFIED | **FALSIFIED** | delta_H tp8 +1.48% (2 sigma 2.62%), tp4 -1.29% (1.08%), tp2 -1.75% (0.62%): out on two of three |
| FLUID | FALSIFIED | FALSIFIED | **FALSIFIED** | tp4 theta 0.797 against 2 sigma 0.399; delta_F tp8 -1.06, tp4 -1.19, tp2 +0.20% |
| FLUID_LOW | FALSIFIED | FALSIFIED | **FALSIFIED** | tp4 theta; tp8 delta_F + 3.6% = +2.54% inside 2.62% |
| H_NPN (secondary) | FALSIFIED | FALSIFIED | **FALSIFIED** | tp8 delta_H +1.48% > -2.0% |
| families | NEITHER | NEITHER | **NEITHER** | slopes tp8 A/B/C 333.3 / 348.8 / 363.3 (H_EST 313.4 / 350.7 / 365.6); tp4 336.3 / 350.6 / 357.1 (339.5 / 352.6 / 365.6); tp2 D1/D2 354.4 / 351.4 |
| co-primary per-cell | INCONCLUSIVE | INCONCLUSIVE | **INCONCLUSIVE** | rms ceil / fluid: tp8 w1 4248 / 4104, tp4 w1 4028 / 4423 cycles (sigma_cell 2678) |
| w2 offset, H_EST | HOLDS x3 | HOLDS x3 | **HOLDS** (tp8, tp4, tp2) | mean r +1.81%, +2.61%, +3.89% against +3.79 to +3.81% |
| w2 offset, FLUID | HOLDS, HOLDS, FALSIFIED | same | **HOLDS tp8, tp4; FALSIFIED tp2** | tp2 +3.89%, sigma 0.87% |
| lock1710 view (printed only) | NEITHER | NEITHER | printed | tp8 slopes 352.7 / 350.0 / 350.8, theta -0.043, per-cell FLUID wins (5876 / 2017); cells within 0.6% of base |

## Part 3, the per-GEMM constant (`2026-10-01-rental2-const-gh200`), primary: 1710 captures

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| K1 L <= 5000 cycles on >= 80% | HOLDS | HOLDS | **HOLDS** | 7 of 60 cells over (11.7%) |
| K2 pooled w2 Z / u in [0.25, 0.69] | HOLDS | HOLDS | **HOLDS** | tp4 0.310, tp2 0.198, median 0.254 |
| K3 Z_1005 / Z_1710 | cycle form HOLDS, ns form FALSIFIED | same | **cycle form HOLDS, ns form FALSIFIED** | w1 6372 / 6810 = 0.936, w2 5702 / 6287 = 0.907, pooled 0.921 +- 0.075 (2 sigma 0.150); rho_f 0.597 |
| K4 D_imb within 0.2 u on >= 80% | HOLDS | HOLDS | **HOLDS** | 95% within, 3.3% at >= 0.3 u |
| K5 | RECORD | RECORD | RECORD | Z_f / u 1710: tp8 0.286 / 0.456, tp4 0.302 / 0.310, tp2 0.315 / 0.198 (w1 / w2); 1005: tp4 0.283 / 0.282 |
| identification | INCONCLUSIVE | INCONCLUSIVE | **INCONCLUSIVE** | K1, K2, K4 hold; H_ZT needs K3's ns form, K3 reads cycles |

K3's rho_f is the ratio of the captures' median GEMM-cell `sm_clock_mhz` (1004.6 / 1690.1 on
w1): the 1005 cells read on their lock while the 1710 cells read 1.2% under theirs, so the
ratio sits 1.2% above 1005 / 1710 = 0.588. K3 sits 1.05 sigma from the cycle form and 4.3
sigma from the ns form; a 1.2% shift of rho_f moves neither.

## Part 2, the launch-floor rerun (`2026-10-01-rental2-launch2-gh200`)

ALL and CLEAN agree on all 37 paths (no gates on these directories); the registered verdict
is the scorer's.

| prediction | mixtral-8x7b-tp8 | granite-3.0-3b-a800m | jetmoe-8b | granite-3.0-1b-a400m |
|---|---|---|---|---|
| P0 instrument | HELD (drift 1.4%; 3 of 108 GR cells out) | **FAILED**: 8 of 108 out, all GR n = 1..3 (drift 3.2%) | HELD (drift 1.1%) | **FAILED**: 15 of 54 out, all GR n = 1..6 (drift 1.1%) |
| P1 GR in [0.80, 1.10] C_reg | **HELD** 27 of 27, 0.877 to 1.015 | **HELD** 15 of 15, 0.807 to 0.948 | **NOT SCORED**: registered n = 5 not on the plan (post-page fix 2); 12 of 12 measured inside, 0.862 to 0.987 | **FALSIFIED**: SHARED n = 1, 2, PRIVATE n = 1 at 0.764 to 0.767; SHARED increment 0.089 < 0.10 ms |
| P2 E240 - GR | **HELD** 19 cells, 3.7 to 5.9 us | **HELD (host drift)** 12 cells, 3.7 to 5.1 us | **HELD** 8 cells, 3.6 to 6.7 us | **HELD (host drift)** 1 cell, 5.3 us |
| P3 (pooled) | 2 of 6 in | 3 of 6 in | 0 of 3 in | none qualify (no E0) |
| P4 E480 shift, verdict | **HELD** 4 shifts -66 to -70 us (band -67.6 +- 12) | **HELD (host drift)** 8 shifts | **HELD** 1 shift, 1 verdict wrong | **HELD (host drift)** no E480 cells |
| P5 C_reg + F < H_pre | **HELD** 77 / 77 | **HELD (host drift)** 72 / 72 | **HELD** 31 / 32 | **HELD (host drift)** 26 / 26 |
| P6 max-plus | **FALSIFIED** 5 of 24: n = 1 E240 / E480 measured 7 to 9% over MP | **HELD** 27 of 27 | **HELD** 21 of 21 | **HELD** 1 of 9 out |
| P7 | CLASSIFIES (n = 1): TR-G kernels 132.1 to 134.8 us, non-GEMM 9.5 to 12.3 us, launch API 9.8 to 11.2% | 78.1 to 85.2 us, 10.9 to 11.4, 10.4 to 11.3% | 197.1 to 199.2 us, 11.2 to 13.4, 10.5 to 11.9% | 50.7 to 54.4 us, 9.7 to 10.1, 10.4 to 11.4% |
| P8 plateau H - [0.08, 0.12] ms | **FALSIFIED** 4 of 7 (H - I 75 to 79 us at n = 2, 3) | **HELD (host drift)** 12 of 12 | **FALSIFIED** 2 of 3 (76 us) | **FALSIFIED (host drift)** 3 of 25 |

P3 pooled over the models: **FALSIFIED**, 10 of 15 qualifying cells outside (E0 - E240 33 to
61 us against bands of about 50 to 75 us). The registration's "both models" is tp8 and
Granite-3B; on their 12 cells alone 7 are outside, the same verdict.

## Part 1, B's knobs and C's T5 (`2026-10-01-rental2-knobs-gh200`)

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| tp2 w2 s8 (PRIMARY): L2r / NL / ST | each FALSIFIED | same | **INCONCLUSIVE** (none selected, all three falsified) | s n = 4..9: 0.855 0.758 0.753 0.759 0.758 0.755; base 0.575 to 0.531 (NL centre); L2r band tops 0.437 to 0.259; dd 0.121, W_c 528 -> 264 |
| tp4 w1 s8 (PRIMARY, NL vs LAG) | both FALSIFIED | same | **INCONCLUSIVE** | s 0.561 0.514 0.492 0.562 0.424 0.463; NL 0.424..0.524 down to 0.303..0.403; LAG tops 0.259 to 0.090 |
| tp2 w2 s6 (secondary) | NL SELECTED | same | **NL** | s 0.579 0.569 0.544 0.516 at n = 6..9, all four in NL only |
| 8x7B w2 s8 x-invariance | HELD | no data (page fails V7) | **NOT SCORED (CLEAN has no data); ALL, reading only gate-failed pages: HELD** | 8x7B minus tp2: -0.010 to -0.052, 5 of 6 within 0.05 |
| tp4 w2 s8 | RECORD | RECORD | RECORD | s 0.888 to 0.903 at n = 4, 9 against base 0.875 to 0.930 |
| H2c l2bk32 w1 | FALSIFIED | FALSIFIED | **INCONCLUSIVE (FLAGGED V6); post-page fix 1** | W_c same as base; deviation -0.018 to -0.028 |
| H2c l2bk32 w2 | SELECTED | SELECTED | **INCONCLUSIVE (FLAGGED V6); post-page fix 1** | W_c differs; -0.066 to -0.113 from the L2r centre |
| H2c l2bk128 w1 | SELECTED | SELECTED | **H2c SELECTED** | W_c differs; +0.38 to +0.57 from the L2r centre (L2r itself is falsified above) |
| H2c l2bk128 w2 | SELECTED | SELECTED | **H2c SELECTED** | W_c same as base; +0.13 to +0.27 |
| H3 slot pad 7 | FALSIFIED | FALSIFIED | **H3 FALSIFIED (no address effect)** | PRIVATE bytes within 0.02%; s within 0.022 (w1), 0.003 (w2) |
| control: tp8 w2 s8 >= 0.95 | ok | ok | **HELD** | none below 0.95 |
| control: tp2 w1 under L2r edge | s8 ok, s6 fails | same | s6's L2r selection would be demoted (there is none) | s6: 0.100 vs 0.075 at n = 4 down to 0.046 vs 0.040 |
| control: PRIVATE F / Mn in [0.50, 0.56] | ok | ok | **HELD** on every page | |
| partition cross-check | DISAGREE x2 | same | **DISAGREE**: far share from the direct count only | 26 of 32 (l2base), 24 of 32 (l2s8) outside 5%, ratio 0.14 to 0.94; far share by subtraction and by direct count differ by up to 0.04 (l2base) and 0.19 (l2s8) |
| T5 R0 / R1 / R2 | FALSIFIED | no data (BK32 page fails V7) | **NOT SCORED (CLEAN has no data); ALL, reading only gate-failed pages: FALSIFIED** | rho_BK 0.357 (n = 8: 0.357); f(BK128) 0.206, f(BK32) 0.577; centres 1.038, 1.038, 1.737 |
| T5 R2h | INCONCLUSIVE (board check failed; measured FALSIFIED) | no data | **NOT SCORED; ALL: INCONCLUSIVE (board check failed)** | f(BK64, n = 9) 0.480 against 0.5802 (-17.2%, tolerance 2.8%); centre 1.340 |

## The BLOCK_K 128 unit (14): analyse exit 1

`r3c-g64.json` of `*-tp2-atbk128-r3-counters/lock1710` passes V0 to V10 (V10 fits 1707.3 MHz +
15.0 us, 1704.0 + 12.4). Its exit 1 is `exit_codes.CLAIM_FAIL`: C1 REFUSE (SHARED's w1
weight-only bracket, its lowest call less PRIVATE's excess e to its highest, has edges that
disagree at n = 4..9, e 0.08 to 0.73: the counter cannot split weight from activation bytes
inside one GEMM), C2 FAIL, C6 FAIL (q_P / n - 1 up to +0.110 on w2). C1 REFUSE and C6 FAIL
read the same on every G = 64 tp2 page of rental 2 and on rental 1's; at G >= E n every live
M-tile sits in one group, so the pid-order claims are not decidable there. It is the
designed outcome, not a refusal of the page: claim gates gate nothing (addendum rule 4), and
T5 reads only PRIVATE w2's q, on which no gate fails.

## Two post-page scorer fixes

Each changed a printed verdict; neither touches a band, cell set or estimator. The ALL and CLEAN
columns above show the verdicts before fix 1.

1. **H2c on tp2 l2bk32 ignores the V6 flag.** The registration (knobs `.txt`, controls): "V6:
   a page whose V6 gate is not PASS is FLAGGED: its cells are excluded from every count and
   its verdicts read INCONCLUSIVE"; docs/registered/README.md: "V6-flagged cells are excluded
   from every count". `score_knobs.py` computes the flag (`controls.page_flags`:
   "mixtral-8x7b-tp2 l2bk32": ["V6 FAIL (FLAGGED)"]) and applies it in the hypothesis tests,
   but its H2c branch never read `flags`. FIXED: both l2bk32 verdicts now read INCONCLUSIVE
   (FLAGGED V6), with the earlier reading kept as `verdict_before_controls`.
2. **P1 on JetMoE counts cells that were never run.** P1's cell set is "tp8 every (arm, n <=
   9), 27 cells; Granite n <= 5, 15 cells", and JetMoE is a tail unit "scored by the same
   rules". The plan ran JetMoE at n = 1..4 (`lf-treads=1,2,3,4`), and `score_launch.py`
   counts a missing GR cell as outside, so its 3 "outside" cells are the three arms' n = 5.
   Every measured cell is inside (0.862 to 0.987 x C_reg); the n = 5 increment rule cannot be
   evaluated. FIXED: P1 reads NOT SCORED wherever a registered tread is absent from the plan
   (JetMoE: n = 5); the measured cells are still listed.
