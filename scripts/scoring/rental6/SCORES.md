# Rental 6 scores (2026-10-13)

The three page scorers in this directory, run on
`results/published/2026-10-13-nvidia_gh200_480gb-rental6-session` (511 run files, each matching
the run branch's SHA256SUMS; the 18 compiled `cuda_utils*.so` files are left out):

    python scripts/scoring/rental6/score_<part>.py . results/published/2026-10-13-nvidia_gh200_480gb-rental6-session scripts/scoring/rental6

Outputs: `skew`, `q1`, `secondk` `.score.{json,txt}` beside the scorers, each with
`addendum.verdicts` (every verdict path with its ALL, CLEAN and registered reading, rental 2's
addendum rule 3). The fourth registration, `2026-10-09-rental6-v3-gh200`, is the model and its
rivals; it has no scorer of its own. No scorer crashed or refused. **No post-page fix**: the
reading code was checked against the pages (section "Scorer check") and nothing in it changes a
verdict; one reading is left to the owner (Q1-H's rivals, below).

Numbers marked DESCRIPTIVE are printed from the scorers' own functions (`page_report`,
`valid_cells`, `carried`, `cluster_mean_se`) on the same valid cells and are not verdicts.

## The verdicts

| test | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| **(a) E1-SKEW**: v3 time from predicted bytes, rms <= 2%, worst <= 5%, \|mean\| <= 1% | HOLDS | HOLDS | **HOLDS** | ALL 122 cells (3 models): rms 0.55%, worst -1.55%, mean -0.00%; CLEAN 42 cells (OLMoE A, B, C): rms 0.54%, worst -1.47%, mean +0.16% |
| E1-UNI (balanced cells, the same bar) | HOLDS | HOLDS | **HOLDS** | ALL 54: rms 0.46%, worst -0.90%; CLEAN 18: rms 0.41%, worst +0.72% |
| SKEW-RATIO v3: rms <= 1.5%, worst <= 4% | HOLDS | HOLDS | **HOLDS** | ALL 68 ratios: rms 0.55%, worst -1.40%, mean -0.42%; CLEAN 24: rms 0.74%, worst -1.40% |
| **(b) E2-SKEW**: v3 time from counted bytes, rms <= 2%, worst <= 5%, \|mean\| <= 1.5% | HOLDS | HOLDS | **HOLDS** | ALL 39 cells (OLMoE, Phi page A): rms 0.62%, worst -1.56%; CLEAN 20 (OLMoE): rms 0.55%, worst -1.47% |
| **B-SKEW**: PT / PW byte rows, every \|e_B\| <= 5%, rms <= 3% (OLMoE w1 carved out) | HOLDS | HOLDS | **HOLDS** | 30 rows: rms 2.66%, worst +4.34%, mean +2.11% (OLMoE w2 rms 3.90%, worst +4.34%; Phi w1 0.47%; Phi w2 2.42%, worst +3.50%) |
| **OPEN-L2SKEW** v3 (OLMoE w1 PT / PW, scored apart) | FAILS | FAILS | **OPEN: v3 FAILS** | 10 rows: rms 11.76%, worst +20.56%, mean +10.40% (n 3 +4.8 to +5.0%, n 6 +10.5 to +11.9%, n 12 +18.9 to +20.6%) |
| OPEN-L2SKEW **row-key rival** at 39.5 MiB: BETTER iff every \|e\| <= 5% and rms <= 0.5 x v3's | NOT BETTER | NOT BETTER | **NOT BETTER** | rms 7.30%, worst +9.95% (n 3 +7.9 to +10.0%, n 6 +6.6 to +7.9%, n 12 -0.8 to -1.1%); printed at 60 MiB: rms 1.04%, worst -1.76% |
| B-CA v3: every w1 uniform / balanced \|e_B\| <= 3% | FAILS | FAILS | **FAILS** | 20 rows: rms 2.11%, worst +3.62% (OLMoE n 6 +3.60 to +3.62%, n 12 +3.08 to +3.09%; Phi within 0.24%) |
| B-CA v3_noca (v3 without content_a) | EXCLUDED | EXCLUDED | **EXCLUDED** | rms 9.87% against v3's 2.11% (OLMoE n 12 +23.4%) |
| **(c) G3**: token order, 90% cluster-t CI within +-0.36% | PASS | PASS | **PASS** | ALL 51 cells, 9 pages: mean -0.071%, CI [-0.140, -0.003]%, Q 9.97 (df 50), p 1.00; CLEAN 18 cells, 3 pages: mean -0.217%, CI [-0.227, -0.208]% |
| **(d) Q1-H**: \|mean g\| <= 1.5 us and \|g(c15) - g(c9)\| <= 1.5 us | UNDECIDED | UNDECIDED | **UNDECIDED** | the copies clause needs c15 pairs and every c15 page was lost; ALL 10 pairs mean g -0.98 us (se 1.07); CLEAN 7 pairs -1.67 us (se 0.80) |
| Q1-P (registered DESCRIPTIVE) | printed | no pairs | **DESCRIPTIVE** | ALL 6 Mixtral n 3 pairs: mean g -6.38 us (se 0.62, one cluster); Phi n 1 pairs lost |
| Q1-SK (registered OPEN) | printed | printed | **OPEN** | skewed minus control: OLMoE -1.57 us (4 / 6 pairs; SEEN prior -3.39), Phi -2.53 us (3 / 3, ALL only) |
| Q1-C: CALL against DEAD on D = x15 - x9 | NOT SCORED | NOT SCORED | **NOT SCORED** | the Qwen1.5 c15 page (unit 13) was lost |
| **(e) Second K**: H_ITER against H_CTA, both GEMMs within 3% of one | UNDECIDED | UNDECIDED | **UNDECIDED** | w1 r 1.4180: H_ITER +0.78%, H_CTA +17.8% (reads H_ITER); w2 r 1.3873: H_ITER +3.17%, H_CTA +26.3% (NEITHER, 0.17 point past the 3% tolerance) |
| **(f) E1 rival v2** | EXCLUDED | EXCLUDED | **EXCLUDED** | 7 lever cells (OLMoE n 3 SHARED): rms 0.96% against v3's 0.45% |
| E1 rival LT3 (S without last-wave quantisation) | EXCLUDED | NOT EXCLUDED | **INCONCLUSIVE** | ALL 82 lever cells 1.41% against 0.57%; CLEAN 22 cells 0.68% against 0.54% |
| E1 rival H8 | NOT EXCLUDED | EXCLUDED | **INCONCLUSIVE** | ALL 20 cells 1.06% against 0.67% (1.6x, under 2x); CLEAN 14 cells 1.24% against 0.54% |
| E1 rival v3_all (b on every non-NATIVE cell) | UNDECIDED (4 cells) | UNDECIDED (0) | **INCONCLUSIVE** | under 6 lever cells in both views |
| E1 rival v3_noca (v3 without content_a) | UNDECIDED (0) | UNDECIDED (0) | **UNDECIDED** | (a) moves no E1 cell by 0.54% (registered gap rms 0.0009%) |
| SKEW-RATIO rival LT3 | NOT EXCLUDED (7) | UNDECIDED (4) | **INCONCLUSIVE** | ALL 1.03% against v3's 0.79% on 7 lever ratios |
| SKEW-RATIO rival U (no skew) | EXCLUDED | EXCLUDED | **EXCLUDED** | ALL 40 lever ratios 11.90% against 0.65% |
| calibration line (secondary) | NOT RESOLVED (3 clusters) | NOT RESOLVED (1) | **INCONCLUSIVE** | ALL slope 0.998, intercept +0.002 (descriptive) |
| gamma (secondary) | CONSISTENT | CONSISTENT | **CONSISTENT** | ALL 0.955 (259 concordant, 6 discordant), CI [0.867, 1.0] |

**Does v3 meet the owner's working bar (rms <= 2%, no cell beyond 5%, here also \|mean\| <=
1%)?** Yes, as a registered verdict, on R3's tile: E1-SKEW HOLDS in both views. The CLEAN view
that certifies it is OLMoE's three pages alone (42 cells, rms 0.54%, worst -1.47%, mean +0.16%);
Mixtral and Phi meet the bar in ALL (Mixtral 42 cells rms 0.44%, worst -1.01%; Phi 38 cells rms
0.65%, worst -1.55%) and are out of CLEAN only because of the histogram G1 drift rule (below). E2-SKEW
(from counted bytes) also HOLDS. Four limits, stated with it: (1) the cells are R3's tile 32 x 64
x 64 w8 s4 at G = 8, and the registration says nothing here is the working bar on vLLM's own
tile; (2) on these floor-bound cells the bytes do not reach time: the decomposition puts at most
0.06% of ln T in the byte term on every cell, and the byte model misses OLMoE w1 by up to 20.6%
(OPEN-L2SKEW) and 3.6% on the controls (B-CA FAILS) without moving a time verdict, so the time
verdicts test the timing model; (3) OLMoE's E1 cells and its n 3 Q1-H pairs are IN-SAMPLE for
h = -5.16 us (SEEN-FITTED), and term (b) moves only SHARED cells whose NATIVE takes block-scan;
(4) Q1-H, the registered test of (b) on control pairs, is UNDECIDED because the c15 pages were
lost to the driver's `fresh_tag` defect.

## Pages each view counts

| page | failed validity gates | ALL | CLEAN |
|---|---|---|---|
| timed mixtral-8x7b ska, skb, skc | G1_lock_thermal (drift flag on 8, 4, 5 cells; every cell 1710.0 MHz); ska also 1 slipped repeat of 216 (PT / SHARED / n 3, repeat 7, 1620 MHz), kept out of its cell median by `--slip-policy cell` | counted | excluded |
| timed olmoe-1b-7b ska, skb, skc | none (no drift, no slip) | counted | counted |
| timed phi-3.5-moe ska, skb, skc | G1_lock_thermal (drift flag on 5 cells each, 4 of 5 at n 2; every cell 1710.0 MHz, no slip) | counted | excluded |
| timed qwen1.5-moe-a2.7b q9 | none | counted | counted |
| timed qwen1.5 q15, phi-3.5 q9 and q15, jetmoe-8b q9 and q15 | no page: each dry run was refused by argparse (`--session-tag -lock1710`, the empty `fresh_tag`); only `DECLARED_COPIES.txt` | NOT SCORED | NOT SCORED |
| byte olmoe-1b-7b, phi-3.5-moe skbytes `r3c-g8` | none recorded; lock_gate.check_page PASS, 20 of 20 cells each in [1640, 1710] MHz | counted | counted |
| byte mixtral-8x7b-tp4 k64s4, k32s4 `r3c-g64` | V7 (arm clock match) on both: read off NATIVE cells only, as registered; V5 PASS; check_page PASS, 27 of 27 cells in band (1651 to 1678, 1668 to 1693 MHz) | counted | counted |

**The Phi pages (checked, as asked).** All three are R3 exit 3 INVALID on G1_lock_thermal alone
(`gates`: G1 FAIL, G2 and G5 PASS), with no slipped repeat and every cell's median at 1710.0 MHz.
The drift flag is on 14, 11 and 17 repeats (6.5%, 6.8%, 10.6% of the page's repeat rows), 39 of
the 42 at n 2. Four NATIVE cells fall under 6 clean repeats of 9 and are lost by the cell rule
(uniform n 2 on A, B and C; PW2 n 2 on C); every other Phi cell is valid. The same holds for
Mixtral (drift on 10, 6, 9 repeats, 21 of 25 at n 3; no cell lost). The rental-6 G1 rule is "R3's
G1_lock_thermal PASS and slipped rows at most 10%", applied as written: the six pages count in ALL
and are dropped in CLEAN. The drift diagnosis delivered with this publication as a proposal (DRIFT.md, not committed) reads the flag: under the held lock
it fires when one of a repeat's 15 per-burst NVML reads, the first or the last, lands on a transient
dip, and the 72 flagged repeats' times match their cells' clean repeats to +0.004% (sd 0.058%).

G5 passes on every page that wrote one: the histogram files' sha256, every row's counts sha256,
an exact bincount, and the shuffle seed 20261010.

## (a) E1-SKEW, per shape, arm and tread (ALL; DESCRIPTIVE breakdown of the registered statistic)

| shape | cells | rms | worst | mean |
|---|---:|---:|---:|---:|
| mixtral-8x7b (ALL only) | 42 | 0.44% | -1.01% | -0.01% |
| olmoe-1b-7b (ALL and CLEAN) | 42 | 0.54% | -1.47% | +0.16% |
| phi-3.5-moe (ALL only) | 38 | 0.65% | -1.55% | -0.16% |
| NATIVE / SHARED | 59 / 63 | 0.57% / 0.52% | -1.55% / -1.49% | -0.05% / +0.04% |

| tread (rank) | Mixtral NATIVE / SHARED | OLMoE NATIVE / SHARED | Phi NATIVE / SHARED |
|---|---|---|---|
| first (n 3, 3, 2) | mean -0.66% / -0.34% | -0.36% / +0.13% | -1.17% / -0.94% (rms 1.23% / 1.00%) |
| second (n 6, 6, 4) | +0.00% / +0.09% | +0.36% / +0.50% | -0.19% / -0.15% |
| third (n 16, 12, 12) | +0.40% / +0.43% | +0.10% / +0.22% | +0.45% / +0.44% |

Per label: PT rms 0.66%, PW 0.55%, PW2 0.53%, DW 0.52%, uniform 0.51%. No cell is beyond 1.55%.
Rental 5's largest residual, SHARED at each shape's first tread (+2.64% mean under v2, at OLMoE
n 2 and Qwen1.5 n 1), does not recur at rental 6's first treads: OLMoE SHARED n 3 reads +0.13%
mean. The largest residual is now Phi n 2, where v3 predicts about 1% too little time on
both arms.

## (b) The byte legs

E2-SKEW HOLDS; B-SKEW HOLDS; OPEN-L2SKEW reads v3 FAILS and the row-key rival NOT BETTER; B-CA
reads v3 FAILS and v3_noca EXCLUDED (table above). The counter-page clock rule replaced rental 5's
unmeetable 1705 MHz cut and kept every byte cell (20 of 20 on each page).

**OLMoE w1 (OPEN-L2SKEW).** v3 over-predicts OLMoE's skewed w1 reads, and the miss grows with n:
+4.8 to +5.0% at n 3, +10.5 to +11.9% at n 6, +18.9 to +20.6% at n 12 (NATIVE only at n 12). The
registered row-key rival at C = 39.5 MiB cuts the n 12 rows to -0.8 to -1.1% but reads +6.6 to
+10.0% at n 3 and 6, so its rms (7.30%) is not half of v3's (11.76%) and two rows pass 5%: NOT
BETTER. The same rival priced at 60 MiB, the card's L2, printed and not registered as the rival,
reads every row within 1.8% (rms 1.04%). This is a SEEN reading on the pages it would be judged
on; it is a candidate for the next registration, not a result.

**The controls (B-CA).** content_a is what keeps OLMoE's uniform and balanced w1 rows near: without
it (v3_noca) they read +6.5% at n 3, +13.0% at n 6, +23.4% at n 12; with it +2.0%, +3.6%, +3.1%. v3
FAILS the 3% bar on the six OLMoE n 6 and n 12 rows (+3.08 to +3.62%); Phi's rows are within 0.24%.
OLMoE's w2 rows read +3.2 to +4.3% on every label, skewed (in B-SKEW) or not, and Phi's w2
uniform / balanced rows +3.4 to +5.1% at n 4 and 12 (outside every registered test).

**Do bytes reach time?** No, on these cells. ln(T_v3 / T_meas) = ln(T_v3 / T_counted) +
ln(T_counted / T_meas): the byte term is at most 0.06% (Phi n 2 PW SHARED) and under 0.002% on
every OLMoE cell, including the n 12 cells whose w1 reads v3 misses by 20%. A check outside the
scorer, scaling a cell's counted reads by 0.8 and 1.2 in `Pricer.price_counted`, moves OLMoE PT
n 12's time by under 1e-9 relative and n 3's by 0.03%: at G = 8 on this tile the calls are floor
bound. E1 and E2 therefore read the same residual (the timing model's), and the byte tests are
the only tests of the byte model.

## (c) G3: token order

d = T(uniform-shuffled) / T(balanced) - 1 - delta_v3 (0 on every pair); clusters are pages; the
margin +-0.36% is fixed.

| view | cells | pages | mean d | se | t (df) | 90% CI | Q (df), p | verdict |
|---|---:|---:|---:|---:|---|---|---|---|
| ALL | 51 | 9 | -0.071% | 0.037% | 1.86 (8) | [-0.140, -0.003]% | 9.97 (50), 1.00 | EQUIVALENT: PASS |
| CLEAN | 18 | 3 | -0.217% | 0.003% | 2.92 (2) | [-0.227, -0.208]% | 0.12 (17), 1.00 | EQUIVALENT: PASS |
| registered | | | | | | | | **PASS** |

Per shape: OLMoE -0.217% (all 18 cells -0.18 to -0.25%, both arms, all three pages: rental 5's
-0.18 to -0.30% again), Mixtral +0.016% (NATIVE n 3 +0.12 to +0.15% on all three pages, every
other cell within 0.03%), Phi -0.012%. Three Phi NATIVE n 2 uniform cells are lost (under 6
clean repeats), so ALL has 51 of 54. G3 passing voids nothing; the strict void-all branch and its
descriptive E1 never apply.

## (d) Q1: the S - N gap

Five of the six Q1 pages were lost (units 13 to 15, 18, 19: `fresh_tag`). What remains: the
Q1-H control pairs on the skew pages (OLMoE n 3 on A, B, C; Phi n 2 on A, B, C) and the Qwen1.5
c9 page (n 1 and 2).

| test | view | pairs (clusters) | mean g | se | reading |
|---|---|---|---:|---:|---|
| Q1-H | ALL | 10 (3) | -0.98 us | 1.07 us | UNDECIDED: no c15 pair for the copies clause |
| Q1-H | CLEAN | 7 (2) | -1.67 us | 0.80 us | UNDECIDED (same) |
| Q1-P | ALL | 6 (1) | -6.38 us | 0.62 us | DESCRIPTIVE |
| Q1-C | both | 0 | | | NOT SCORED: no Qwen1.5 c15 page |

Per pair (ALL): OLMoE n 3 -2.03 to -2.29 us (six pairs, mean -2.14; SEEN anchor -1.54); Phi n 2
balanced +0.45 to +0.77 us (three; the uniform pairs lost their NATIVE cell; anchor -0.74);
Qwen1.5 n 2 c9 +1.14 us (anchor +0.90). The level clause alone would read pass in ALL (0.98) and
fail in CLEAN (1.67, OLMoE and Qwen1.5 only).

**Q1-H's rivals (DESCRIPTIVE; reading left to the owner).** `r6common.q1_rule` returns UNDECIDED
before it reads the rivals when the copies clause has no c15 side, so the scorer prints no rival
verdict. The registration's rival rule ("EXCLUDED iff |mean g - g_R| > 3 se and > 1.5 us;
UNDECIDED below 6 pairs") is stated apart from the copies clause; whether a v3 UNDECIDED also
leaves the rivals undecided is a judgment, so the scorer is not changed. On the same pairs:

| rival | g_R | ALL: distance against 3 se (3.22 us) | CLEAN: distance against 3 se (2.41 us) |
|---|---:|---|---|
| v2 (no term b) | +4.55 us | 5.53 us: would read EXCLUDED | 6.15 us: would read EXCLUDED |
| H8 | +7.45 us | 8.42 us: would read EXCLUDED | 9.12 us: would read EXCLUDED |
| v3_all | 0.00 us | 0.98 us: NOT EXCLUDED (identical to v3 on these pairs by construction) | 1.67 us: NOT EXCLUDED |

**Q1-P (DESCRIPTIVE as registered):** the six Mixtral n 3 pairs read g -4.54 to -8.54 us (mean
-6.38; SEEN anchor -8.03). Against them v2 sits 6.38 us away, v3_all (g_R -10.43 us) 4.05 us, H8
(-3.81 us) 2.57 us, each beyond 3 se (1.87 us); v3 itself is outside the 1.5 us level. This is
the one population where the block-scan path condition (v3 against v3_all) shows, and it favours
the condition (v3 nearer by 4.05 us), but it is registered as descriptive.

## (e) Second K (`2026-10-09-rental6-secondk-gh200`)

mixtral-8x7b-tp4 G = 64, both pages taken in rental 6 on one board; c by r4common.page_c over
NATIVE n 4..9 (V7 failed on both pages, so NATIVE only, 6 floor-bound cells per GEMM), F at CAL
(w1 520, w2 979 cycles); the pages' median cell clocks 1670.6 and 1683.4 MHz (0.77% apart, inside
the 1% pair rule).

| GEMM | c64 (SEEN OLMoE) | c32 (SEEN OLMoE) | r_meas = u(k32) / u(k64) | H_ITER (pred) | H_CTA (pred) | reading |
|---|---:|---:|---:|---|---|---|
| w1 (K 4096) | 338.4 (346.2) | 241.6 (245.2) | 1.4180 | 1.4070, e +0.78% | 1.2035, e +17.8% | H_ITER |
| w2 (K 3584) | 346.8 (344.0) | 243.9 (234.3) | 1.3873 | 1.3447, e +3.17% | 1.0985, e +26.3% | NEITHER |

Registered: **UNDECIDED** in both views (w1 H_ITER, w2 NEITHER). H_CTA is excluded by 17.8 and
26.3 points on the two GEMMs; H_ITER is within 0.8% on w1 and misses the 3% tolerance on w2 by
0.17 point, so the registered rule cannot select it. Read over both arms (printed, against the
V7 rule) the ratios are 1.4185 and 1.3892: the same readings. The BK 32 excess is per k-step on
w1, and on w2 it is per k-step plus about 3% the hypothesis does not carry; it is not per CTA.
Each page's CLAIM C2 and C6 FAIL (page exit 3) are byte findings of R3's counter mode, not gates
of this registration.

## (f) v3 against its rivals

| rival | what it drops | registered test that separates it | reading |
|---|---|---|---|
| v2 | M4 bytes and term (b) | E1 rival rule, 7 lever cells | **EXCLUDED** (both views). Those 7 cells are OLMoE n 3 SHARED, IN-SAMPLE for h. Q1-H rival reading (descriptive): would read EXCLUDED |
| v3 without (b) (v3_nob) | term (b) | none registered as a rival (its lever list is empty) | DESCRIPTIVE: its time equals v2's on every E1 cell (E1 ALL rms 0.582% for both, identical to 3 decimals: bytes do not reach time), so v2's E1 exclusion is the exclusion of (b)'s absence |
| v3 without content_a (v3_noca) | content-keyed A tiles | B-CA (bytes) | **EXCLUDED on bytes** (rms 9.87% against 2.11%); E1 UNDECIDED (0 lever cells: no time cell moves) |
| LT3 (S without last-wave quantisation) | wave_q | E1 rival rule; SKEW-RATIO rival rule | **INCONCLUSIVE** on both (E1: ALL EXCLUDED, CLEAN NOT EXCLUDED; ratio: ALL NOT EXCLUDED on 7, CLEAN 4 cells). DESCRIPTIVE: E1 ALL rms 1.19%, mean -0.82% against v3's 0.55%, -0.00% |
| v3_all | the block-scan path condition | E1 rival rule (too few lever cells); Q1-H cannot separate (g_R 0) | **INCONCLUSIVE** (E1). DESCRIPTIVE Q1-P favours the condition by 4.05 us |
| H8 | the 4-model fit of h (8x7B refit, h -0.65 us) | E1 rival rule | **INCONCLUSIVE** (ALL NOT EXCLUDED at 1.6x, CLEAN EXCLUDED at 2.3x). Q1-H rival reading (descriptive): would read EXCLUDED |
| U (no skew) | the skew term | SKEW-RATIO rival rule | **EXCLUDED** (both views, 11.9% against 0.65%) |
| row-key at 39.5 MiB | (OLMoE w1 rival) | OPEN-L2SKEW | **NOT BETTER** |

## Scorer check (no post-page fix)

Checked against the pages before scoring: the slip fields (`repeat_rows`, `lock_slip_rows`,
`slipped_repeats`, per-row `clean_repeats`, which R3 writes net of slipped and drift-excluded
repeats: 9 minus each cell's flagged repeats on every row); G4's `sm_clock_load_mhz`; the timer's
`host_bound`; `align_probe`'s `label` / `tread` / `ms`; G5's file sha256, counts sha256,
`bincount_ok` and shuffle seed; the counter pages' `design.histogram`, `per_gemm.dram_bytes_read`
and `recorded.sm__cycles_elapsed.avg`; V7's `number`; the second-K design check. Each reads what
the page carries. The scorers are deterministic (two runs byte-identical).

## Readings left to the owner (no scorer change)

- **Q1-H's rivals** when the copies clause is undecided (part d): the scorer reads none; the
  descriptive numbers say v2 and H8 would be excluded.
- **The histogram G1 drift rule.** R3's histogram G1 fails a page for a single drift-excluded
  repeat; ladder pages tolerate 20% (V0). It removed every Mixtral and Phi page from CLEAN for 67
  flagged repeats of 1,078 whose times match their clean neighbours. A proposal with pass rates is
  delivered beside this publication (DRIFT.md, not committed); it is not applied, and under it CLEAN would equal ALL, changing only
  the rival verdicts that differ between the views today (E1 LT3 to EXCLUDED, H8 to NOT EXCLUDED).
- **B-CA's and OPEN-L2SKEW's v3 readings** are keyed `v3`, not `verdict`, so rule 3's table does
  not list them; the byte pages carry no G1 and both views read the same rows, so nothing hides.
- **Bytes do not reach time** on these floor-bound cells (part b): E2-SKEW holding says nothing for
  the byte model, and B-SKEW / B-CA / OPEN-L2SKEW are its only tests.
