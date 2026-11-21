# Rental 5 scores (2026-10-09)

The five page scorers in this directory, run on
`results/published/2026-10-09-nvidia_gh200_480gb-rental5-session` (1301 run files, each matching
the run branch's SHA256SUMS; the 18 compiled `cuda_utils*.so` files are left out):

    python scripts/scoring/rental5/score_<part>.py . results/published/2026-10-09-nvidia_gh200_480gb-rental5-session scripts/scoring/rental5

Outputs: `skew`, `c15`, `secondk`, `bk128`, `stamps2` `.score.{json,txt}` beside the scorers.
Each carries `addendum.verdicts`, every verdict path with its ALL, CLEAN and registered reading
(rental 2's addendum rule 3). The sixth registration, `2026-10-07-rental5-v2-gh200`, is the model
change and its SEEN diagnostic (`v2seen.score.*`, committed with the registration, unchanged). No
scorer crashed or refused. One post-page fix to reading code in `score_skew.py` follows the
registration's text and changes the G3 verdict from FAIL to INCONCLUSIVE (end of file).

**The descriptive numbers below** (per shape, per arm, the byte leg with its clock cut lifted, the
rivals over all ratios) are printed from the scorers' own functions on the ALL view and are not
verdicts. Each is labelled where it appears.

## Pages each view drops

| page | failed validity gates | ALL | CLEAN |
|---|---|---|---|
| timed mixtral-8x7b ska, skb | no report.json: `locked_r3` stopped each page at its first sub-lock cell (1635, 1680 MHz) | NOT SCORED (no page) | NOT SCORED |
| timed olmoe-1b-7b ska, skb, sk15 | none (worst cell 1710.0 MHz) | counted | counted |
| timed qwen1.5-moe-a2.7b ska | none | counted | counted |
| timed qwen1.5-moe-a2.7b skb | G1_lock_thermal (drift on three n = 2 cells; every cell 1710 MHz) | counted | excluded |
| timed olmoe-1b-7b bk64, bk128 | none (G2 excludes balanced / native / n 1, host-bound) | counted | counted |
| byte mixtral-8x7b skc, olmoe-1b-7b skc `r3c-g8` | the histogram byte page records no gate; the registered byte-leg gates are G5, G3's voiding and the 1705 MHz cut | counted | counted |
| byte qwen2-57b-a14b-tp8 k64s4, k32s4 `r3c-g64` | V5 (PRIVATE w1 q_P 0.957 n, under 0.97 n; as on rental 3's qwen2-tp8 pages) | counted | excluded (secondk gates on V5) |

G5 passes on every page that wrote one: the histogram file's sha256, every row's counts sha256, an
exact bincount and the shuffle seed 20261007. No trace-derived count is on any page.

## (a) Skew end to end: model v2 under skewed routing (`2026-10-07-rental5-skew-gh200`)

| prediction | ALL | CLEAN | registered verdict | key numbers (ALL) |
|---|---|---|---|---|
| E1-SKEW: v2 time from predicted bytes, rms <= 2.5%, worst <= 6%, \|mean\| <= 1.5% | HOLDS | NOT SCORED (G3 voids every histogram cell) | **NOT SCORED (CLEAN has no data); ALL HOLDS** | 72 cells: rms 1.26%, worst +4.38%, mean +0.40% |
| E2-SKEW: v2 time from counted bytes, rms <= 2%, worst <= 5%, \|mean\| <= 1.5% | NOT SCORED | NOT SCORED | **NOT SCORED** | no byte cell passes the registered 1705 MHz cut (part b) |
| SKEW-RATIO S: T(h) / T(uniform), rms <= 1.5%, worst <= 4% | HOLDS | NOT SCORED | **NOT SCORED (CLEAN has no data); ALL HOLDS** | 36 ratios: rms 1.05%, worst -2.09%, mean -0.85% |
| rival U (skew ignored, r = 1) | EXCLUDED | UNDECIDED (0 lever cells) | **INCONCLUSIVE** | rms 12.95% on 30 lever cells, S 1.13% there |
| rival PW (per-expert waves) | EXCLUDED | UNDECIDED | **INCONCLUSIVE** | rms 17.81% on 32 lever cells, S 1.11% there |
| rival LT (S without the last wave's quantisation, the close rival) | UNDECIDED (5 lever cells, under 6) | UNDECIDED | **INCONCLUSIVE** | on its 5 lever cells LT rms 0.70%, S 1.66% (printed) |
| M1 (S under model M's dead term) | printed | | printed | rms 1.14%, mean -0.93% |
| E1-UNI (balanced, n <= 6) | HOLDS | HOLDS | **HOLDS** | 16 cells: rms 1.80%, worst +3.79%, mean +1.33% |
| E1-UNI-EXT (balanced past every CAL tread) | HOLDS | HOLDS | **HOLDS** | 8 cells: rms 0.26%, worst +0.36% |
| calibration line (ln meas on ln pred, cluster = shape) | NOT RESOLVED (2 clusters) | NOT APPLICABLE | **INCONCLUSIVE, descriptive** | slope 1.0054, intercept -0.0048 over 60 cells |
| gamma (stratified) | CONSISTENT | NOT RESOLVED | **INCONCLUSIVE** | 0.978 (C 92, D 1), CI [0.957, 1.0] |

Mixtral's cells are NOT SCORED in both views (no page), so every number above is OLMoE (pages A and
B, BLIND-CALMODEL) and Qwen1.5 (A, and B in ALL only, BLIND).

E1-SKEW by shape, arm and tread (descriptive, ALL):

| set | cells | rms | worst | mean |
|---|---:|---:|---:|---:|
| olmoe-1b-7b | 36 | 1.12% | +3.21% | +0.56% |
| qwen1.5-moe-a2.7b | 36 | 1.40% | +4.38% | +0.23% |
| NATIVE | 42 | 0.77% | -1.95% | -0.13% |
| SHARED | 30 | 1.73% | +4.38% | +1.14% |
| SHARED at each shape's first tread (OLMoE n 2, Qwen1.5 n 1) | 10 | 2.84% | +4.38% | +2.64% |
| SHARED at the second, third tread | 10, 10 | 0.95%, 0.27% | | +0.57%, +0.21% |
| PT, PW, DW | 12 each | 0.91%, 0.93%, 1.53% | | |
| uniform | 24 | 1.60% | +3.87% | +1.11% |
| PW-hotfirst, PW-rand1 | 6 each | 0.66%, 0.65% | | -0.48%, -0.51% |

The largest residual is SHARED at the smallest tread, and it is not skew's: the balanced cells
there read +2.84 to +3.79% (E1-UNI), the uniform cells +3.1 to +3.9%, the skewed cells +1.1 to
+4.4%. The skew ratio itself is under-predicted by 0.85% on average (the measured penalty of a
skewed histogram over uniform is about 1 point larger than S's), and LT, which drops the last
wave's quantisation, reads closer than S on all 36 ratios (rms 0.85% against 1.05%, printed).

**Against the time bar (rms <= 2%, no cell beyond 5%)** model v2 passes in the ALL view from
predicted bytes (1.26% / 4.38%) and, printed with the byte leg's clock cut lifted, from counted
bytes on OLMoE (1.32% / 3.10%, part b). No registered verdict certifies it: G3 fails in CLEAN, the
byte leg is cut by a clock rule no GH200 counter page meets, and Mixtral has no page. The
registration also says this R3 tile (32x64x64 w8 s4) is not vLLM's, so nothing here is the
working bar on vLLM's own tile.

## (b) The byte legs (B-SKEW, E2-SKEW, the decomposition)

**Registered: NOT SCORED in both views.** Each byte cell's SM clock, sm__cycles_elapsed.avg /
gpu_time_ns over its GEMMs of 0.5 ms or longer, reads 1656 to 1698 MHz on both pages (OLMoE's
short-GEMM cells take the page median, 1691.8 MHz), so the registered 1705 MHz cut drops every
cell. The lock held: the 1710 lock was in force for both pages (`lock-plan.txt`, the
`clocks-after-*` reads) and every timed page of the session reads 1710.0 MHz. The ratio itself
reads low on every GH200 page: over the 113 published lock-1710 counter pages, 1 of 3646 GEMMs of
0.5 ms or longer reads 1705 MHz or more, page medians 1654 to 1695 MHz. The registered cut cannot
be met by this metric on this card. This is reported, not re-ruled (readings left to the owner).

Printed with the clock cut lifted (every other gate kept; descriptive, not a verdict):

| page, n | skewed PT / PW e_B w1 | w2 | uniform / balanced control w1 | w2 |
|---|---|---|---|---|
| mixtral-8x7b n 2 | +0.16 to +0.23% | +0.23 to +1.82% | +0.13 to +0.20% | +1.26 to +2.45% |
| mixtral-8x7b n 8 | +0.47 to +0.50% | -0.60 to +1.40% | +0.75 to +0.76% | +3.91 to +5.60% |
| olmoe-1b-7b n 2 | +0.22 to +1.37% | +3.43 to +3.97% | +2.95% | +1.41% |
| olmoe-1b-7b n 4 | +8.08 to +8.67% | +3.97 to +4.21% | +5.92% | +2.10 to +2.11% |
| olmoe-1b-7b n 16 (NATIVE) | +24.95 to +25.67% | +4.06 to +4.16% | +30.74% | +3.55 to +3.59% |

B-SKEW would read FALSIFIED (36 skewed (cell, GEMM): rms 6.94%, worst +25.67%); the uniform and
balanced controls miss as much (rms 7.87%, worst +30.74%). E2-SKEW would HOLD on its 20 OLMoE
cells (rms 1.32%, worst +3.10%, mean +0.86%; Mixtral has no timed page).

**Do the byte legs separate an L2-law failure from a timing failure?** On these pages, yes, and the
answer is that they are two different misses. The byte model misses OLMoE's w1 by 6 to 31% at n 4 and 16, and the
miss is the same on the uniform and balanced controls, so it is the L2 law's at OLMoE w1 and not
skew's; Mixtral's skewed bytes are within 1.8%. The decomposition, ln(T_pred / T_meas) =
ln(T_pred / T_counted) + ln(T_counted / T_meas), puts at most 0.0002 in the byte term on every
OLMoE cell: these G = 8 cells are floor-bound, so a 25% byte miss moves the predicted time by under
0.02%, and the whole time residual is the timing model's (E2's). The byte leg therefore cannot
convict or clear the L2 law through time here; it can only show that the time residual is not the
bytes'. That reading is descriptive: the registered verdicts are NOT SCORED.

## (c) G3: token order (the TOST with cluster-t, Cochran's Q)

d = T(uniform-shuffled) / T(balanced) - 1 - delta_M3, delta_M3 = 0 on every cell; clusters are
pages; the margin +-0.36% is fixed.

| view | cells | pages | mean d | se | t (df) | 90% CI | Q (df), p | verdict |
|---|---:|---:|---:|---:|---|---|---|---|
| ALL | 24 | 4 | -0.129% | 0.061% | 2.353 (3) | [-0.274, +0.015]% | 4.97 (23), 0.99997 | EQUIVALENT: PASS |
| CLEAN | 18 | 3 | -0.164% | 0.071% | 2.920 (2) | [-0.373, +0.044]% | 3.44 (17), 0.99982 | NOT EQUIVALENT: FAIL, every histogram cell voided |
| registered | | | | | | | | **INCONCLUSIVE** |

At sigma 0.10% the rule reads PASS in ALL and FAIL in CLEAN as well. Per shape (ALL): OLMoE
-0.235%, Qwen1.5 -0.024%; per arm: NATIVE -0.123%, SHARED -0.136%. Q is far from heterogeneous in
both views (p near 1), so no stratum is tested apart. Every per-cell difference:

| page | n | NATIVE d | SHARED d |
|---|---:|---:|---:|
| olmoe-1b-7b ska | 2 | -0.300% | -0.270% |
| olmoe-1b-7b ska | 4 | -0.270% | -0.240% |
| olmoe-1b-7b ska | 16 | -0.210% | -0.211% |
| olmoe-1b-7b skb | 2 | -0.209% | -0.223% |
| olmoe-1b-7b skb | 4 | -0.262% | -0.262% |
| olmoe-1b-7b skb | 16 | -0.179% | -0.184% |
| qwen1.5-moe-a2.7b ska | 1 | +0.056% | -0.071% |
| qwen1.5-moe-a2.7b ska | 2 | -0.124% | -0.057% |
| qwen1.5-moe-a2.7b ska | 8 | +0.047% | +0.014% |
| qwen1.5-moe-a2.7b skb (G1 FAIL: ALL only) | 1 | -0.075% | -0.081% |
| qwen1.5-moe-a2.7b skb (G1 FAIL: ALL only) | 2 | +0.010% | -0.013% |
| qwen1.5-moe-a2.7b skb (G1 FAIL: ALL only) | 8 | +0.043% | -0.032% |

Token order is not null on OLMoE: all 12 of its cells read the shuffled rows faster than the
balanced (unshuffled) rows, by 0.18 to 0.30%, on both pages and both arms: 1.6 to 2.7 times the 0.11%
noise of a two-cell ratio (the A / B same-call replicate reads a single-page sigma of 0.08%,
`docs/paper/noise_floors.md`), all 12 of one sign. Qwen1.5
reads zero (-0.12 to +0.06%). The CLEAN failure is a three-cluster interval (t 2.92 at df 2)
around a mean that sits inside the margin; the effect is OLMoE's and is inside +-0.36% on every
cell.

**The label-permutation control** (NATIVE, PW-hotfirst and PW-rand1 against PW, against S's
predicted dr): FAIL in both views, printed and voiding nothing, as registered. Its 12 cells read
-0.39 to +0.13% (mean -0.02%), and the interval is a two-cluster one (df 1, t 6.31: CI [-0.360,
+0.321]%), so the FAIL is the cluster count, not the cells. The scorer prints the G3 rule's own
text, "(voids every histogram cell of the board)", beside it; per the registration it voids
nothing, and the scorer's voiding reads G3 alone.

| permutation cell | d |
|---|---:|
| olmoe-1b-7b A n 2 PW-hotfirst | +0.133% |
| olmoe-1b-7b A n 2 PW-rand1 | +0.070% |
| olmoe-1b-7b A n 4 PW-hotfirst | +0.094% |
| olmoe-1b-7b A n 4 PW-rand1 | -0.018% |
| olmoe-1b-7b A n 16 PW-hotfirst | -0.026% |
| olmoe-1b-7b A n 16 PW-rand1 | -0.048% |
| qwen1.5-moe-a2.7b A n 1 PW-hotfirst | -0.392% |
| qwen1.5-moe-a2.7b A n 1 PW-rand1 | +0.070% |
| qwen1.5-moe-a2.7b A n 2 PW-hotfirst | +0.018% |
| qwen1.5-moe-a2.7b A n 2 PW-rand1 | -0.049% |
| qwen1.5-moe-a2.7b A n 8 PW-hotfirst | -0.049% |
| qwen1.5-moe-a2.7b A n 8 PW-rand1 | -0.040% |

## (d) The 15-copy SHARED page under skew (`2026-10-07-rental5-c15-gh200`)

Delta(h, n) = [S - N](c15) - [S - N](c9), N the NATIVE uniform cell of each page; raw (no
align_probe on the pages).

| h | n | Delta (us) | dead CTAs c9 / c15 | live rows |
|---|---:|---:|---|---:|
| PT | 2 | 27.07 | 33,856 / 57,664 | 157 |
| PW | 2 | 26.94 | 33,600 / 57,408 | 161 |
| DW | 2 | 27.55 | 33,728 / 57,536 | 159 |
| uniform | 2 | 27.49 | 35,712 / 59,520 | 128 |
| PT | 4 | 22.24 | 33,600 / 57,408 | 289 |
| PW | 4 | 23.23 | 33,472 / 57,280 | 291 |
| DW | 4 | 23.46 | 33,856 / 57,664 | 285 |
| uniform | 4 | 23.84 | 35,712 / 59,520 | 256 |

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| D (d 0.995 ns, kappa 0.325) | HOLDS | HOLDS | **HOLDS** | median Delta 25.39 us against 23.69, z +1.32 (at 0.10%: +2.38) |
| M (1.333 ns, k_w 0.5) | EXCLUDED | EXCLUDED | **EXCLUDED** | 31.74 us, z -4.93 |
| FIXED (no increment) | EXCLUDED | EXCLUDED | **EXCLUDED** | z +19.7 |
| SKEW-DEAD | not shown | not shown | **not shown** | median skewed minus uniform -0.46 us, inside 3 sigma (3.86 us) |
| two-regressor (printed) | | | | a 24.79 us, d 0.84 ns, b 0.148 us per live row, r(dead, live) -0.02, VIF 1.00 |

The dead-CTA cost rental 4 found on uniform routing carries to skewed routing unchanged: the
increment does not depend on the histogram. The n = 2 cells read 3 to 4 us above the n = 4 cells.

## (e) Second K and BK 128

**Second K** (`2026-10-07-rental5-secondk-gh200`), qwen2-57b-a14b-tp8 G = 64, c by r4common.page_c
over NATIVE and SHARED n = 4..9, F at CAL:

| GEMM | c64 | c32 | r_meas = u(k32) / u(k64) | H_ITER (pred) | H_CTA (pred) | reading |
|---|---:|---:|---:|---|---|---|
| w1 (K 3584) | 346.9 | 249.6 | 1.4274 | 1.4056, e +1.5% | 1.2318, e +15.9% | H_ITER |
| w2 (K 320) | 332.7 | 211.0 | 1.1690 | 1.2308, e -5.0% | 1.7386, e -32.8% | NEITHER |

Registered: **NOT SCORED (CLEAN has no data)**: both pages fail V5 (PRIVATE w1 q_P 0.957 n,
the shortfall rental 3's qwen2-tp8 pages showed), and the registration gates on every V-gate but
V6, V7 and V10. ALL reads UNDECIDED (w1 H_ITER, w2 NEITHER). w2's c64 is 332.7 against OLMoE's SEEN
344.0, and at K 320 a w2 CTA runs 5 k-steps, so F dominates u there.

**BK 128** (`2026-10-07-rental5-bk128-gh200`), OLMoE G = 64 NATIVE, q = T128 / T64:

| n | q_meas | H_C e | SYNC e | PS e |
|---:|---:|---:|---:|---:|
| 4 | 0.9958 | +0.18% | +71.44% | -0.09% |
| 5 | 0.9957 | +0.17% | +72.57% | -0.10% |
| 6 | 0.9951 | +0.13% | +73.39% | -0.15% |
| 7 | 0.9946 | +0.08% | +73.87% | -0.20% |
| 8 | 0.9949 | +0.11% | +74.42% | -0.17% |
| 9 | 0.9949 | +0.11% | +74.76% | -0.17% |

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| H_C (the occupancy-3 floor), rms <= 1% | HOLDS | HOLDS | **HOLDS** | rms 0.13% |
| SYNC (c held at BK 64: the floor halves) | EXCLUDED | EXCLUDED | **EXCLUDED** | 6 of 6 beyond 0.76% |
| PS (printed) | | | printed | rms 0.15% |
| calibration line | NOT RESOLVED (1 cluster) | same | descriptive | slope 6.75 over six q values spanning 0.12% |

BK 128 buys 0.4 to 0.5% at G = 64, as the counters' c128 = 1.98 c64 said it would: the per-k-step
cost doubles with the tile, so the floor is the occupancy-3 floor and not a synchronisation saving.

## (f) The stamps (`2026-10-07-rental5-stamps2-gh200`)

| unit | block | regcheck (CTAs/SM against the plain kernel's) | perturbation gate (median / worst) | verdict |
|---|---|---|---|---|
| stf (8x7B) | F2, K2 | v1, v2, v3 FAIL (5 against 4) | v1 2.35 / 3.03%, v2 2.19 / 3.00%, v3 2.22 / 2.75% | NOT SCORED: NOT RUN |
| stt8x22 (8x22B) | K2 | v1, v2 FAIL | v1 6.57 / 12.42%, v2 5.99 / 11.84% | NOT SCORED: NOT RUN |
| stt (8x7B) | K2 | v1, v2 FAIL | v1 2.26 / 14.66%, v2 2.18 / 14.34% | NOT SCORED: NOT RUN |
| std4 (OLMoE, s4) | D2 | v2, v3 FAIL | v2 2.36 / 3.61%, v3 2.27 / 3.33% | NOT SCORED: NOT RUN |
| std2 (OLMoE, s8) | D2 | v2, v3 PASS (2 = 2) | v2 3.16 / 6.55%, v3 2.85 / 5.95% | NOT SCORED: NOT RUN |
| F2, K2, D2 readouts | | | | NOT SCORED |

Registers (`regcheck.json`): the stamped copy compiles w2 at 48 registers against the plain
kernel's 55 and w1 at 44 against 48 in every variant, v3 included, so at BLOCK_K 64 s4 the copy
fits 5 CTAs/SM where the plain kernel fits 4; at s8 both are held to 2 by shared memory. The gate
numbers are the VM's `gate.env` lines (the scorer reads their PASS / FAIL); the tolerance is median
1%, worst 2%. std2 fails at equal occupancy, so its 2.8 to 3.2% is the stamps' own cost at the
sample level. The 1-in-17 variants (v2: iteration tops and dead CTAs; v3: everything) are no
cheaper than v1 on the median by more than 0.6 point.

## Readings left to the owner (no scorer change)

- **The byte-leg clock cut.** The registered G4 for the byte leg reads the SM clock as
  sm__cycles_elapsed.avg / gpu_time_ns; on this card that ratio sits 15 to 56 MHz under the lock (page medians) on
  every published lock-1710 counter page (1 GEMM of 3646 reaches 1705), so B-SKEW and E2-SKEW can
  never be scored as registered. The descriptive numbers are in part (b).
- **The Mixtral pages.** `locked_r3` stops a page at the first cell one step under the lock, which
  leaves no report for the registration's per-cell G4 to act on. Mixtral's six and eight completed
  repeats are in `cells.csv`; reading them would need new reading code and a rule for a partial
  page, and is not done.
- **The permutation control's printed text** says it voids every histogram cell; the registration
  says it voids nothing, and the scorer voids nothing on it.
- **"ALL, reading only gate-failed pages"** is rule 3's fixed wording for "CLEAN has no data". In
  skew, CLEAN has no data because G3 fails there, not because ALL reads only gate-failed pages:
  ALL reads four pages, one of them gate-failed.

## The post-page fix (one verdict changes)

`score_skew.py`, `page_report`. The registration's `views` rule rules on every VALIDITY gate by
rental 3's two views (ALL counts a page whose gates failed, unless V1; CLEAN excludes it), and its
G1 is "V7: the 1710 lock in force and the thermal gate, as every timed page". The scorer refused a
page whose `histogram_gates.G1_lock_thermal` was not PASS before the view saw it, so Qwen1.5 skb (G1
FAIL on a drift flag) was dropped from both views. It now leaves a page that records G1 in its
`gates` list to the views (ALL counts it, CLEAN excludes it), and still refuses a page that records
G1 nowhere in `gates`. Effect, flagged: G3 FAIL becomes INCONCLUSIVE (ALL PASS, CLEAN FAIL);
SKEW-RATIO S and E1-SKEW stay NOT SCORED, now with ALL's HOLDS beside them; the rivals U and PW go
from UNDECIDED to INCONCLUSIVE (ALL EXCLUDED). Before the fix: G3 FAIL (3 pages, CI [-0.373,
+0.044]%), every histogram cell voided, every skew test NOT SCORED, E1-UNI and E1-UNI-EXT HOLDS.
The c15, secondk, bk128 and stamps2 outputs are byte-identical before and after. Test:
`tests/test_scoring_rental5_skew.py::test_a_g1_failed_page_is_ruled_by_the_two_views`.
