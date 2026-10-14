# Rental 7 scores (2026-10-14)

The three page scorers in this directory, run on
`results/published/2026-10-14-nvidia_gh200_480gb-rental7-session` (449 run files, each matching
the run branch's SHA256SUMS; the 15 compiled `cuda_utils*.so` files are left out):

    python scripts/scoring/rental7/score_<part>.py . results/published/2026-10-14-nvidia_gh200_480gb-rental7-session scripts/scoring/rental7

Outputs: `skew`, `q1`, `l2open` `.score.{json,txt}` beside the scorers. The skew and q1 outputs
carry `addendum.verdicts` (every verdict path with its ALL, CLEAN and registered reading, rental
2's addendum rule 3) and `P20_view` (the owner's P20 proposal rebuilt from the same cells.csv,
DESCRIPTIVE, never a verdict). The fourth registration, `2026-10-13-rental7-g1rule-gh200`, is the
drift rule the other three apply; it has no scorer of its own. No scorer crashed or refused; two
runs are byte-identical. **No post-page fix**: the reading code was checked against the pages
(section "Scorer check") and nothing in it changes a verdict. One non-scorer fix, flagged below:
`register.py --check` now reads the leakage set as of the registration date.

Numbers marked DESCRIPTIVE are computed from the scorers' own functions (`page_report`,
`valid_cells`, `carried`, `q1h_rule`) on the same valid cells and are not verdicts.

## The verdicts

**ALL and CLEAN are identical on every rental-7 page**: the rebuilt G1 passes on all twelve timed
pages, so CLEAN drops nothing, and 0 of 14 verdicts differ between the views in either scorer.

| test | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| **(a) E1-SKEW**: v3 time from predicted bytes, pooled AND each shape rms <= 2%, worst <= 5%, \|mean\| <= 1% | HOLDS | HOLDS | **HOLDS** | 84 cells (Mixtral 42, Phi 42, 6 of 6 pages): rms 0.61%, worst -1.61%, mean -0.24% |
| E1-SKEW, Mixtral 8x7B (per shape) | HOLDS | HOLDS, **CERTIFIED** (3 of 3 pages) | **HOLDS, CERTIFIED** | 42 cells: rms 0.51%, worst -1.22%, mean -0.17% |
| E1-SKEW, Phi-3.5-MoE (per shape) | HOLDS | HOLDS, **CERTIFIED** (3 of 3 pages) | **HOLDS, CERTIFIED** | 42 cells: rms 0.69%, worst -1.61%, mean -0.31% |
| E1-UNI (balanced cells, the same bar) | HOLDS | HOLDS | **HOLDS** | 36 cells: rms 0.46%, worst -0.84% |
| SKEW-RATIO v3: rms <= 1.5%, worst <= 4% | HOLDS | HOLDS | **HOLDS** | 48 ratios: rms 0.45%, worst -1.10%, mean -0.39% |
| SKEW-RATIO rival U (no skew) | EXCLUDED | EXCLUDED | **EXCLUDED** | 32 lever ratios: 12.85% against v3's 0.50% |
| SKEW-RATIO rival LT3 | UNDECIDED (4) | UNDECIDED (4) | **UNDECIDED** | 4 lever ratios, under 6 |
| **(b) Q1-H** (REPLICATE): \|mean g\| <= 1.5 us AND the copies clause \|g(c15) - g(c9)\| <= 1.5 us | FAILS | FAILS | **FAILS** | 24 pairs, 5 clusters: mean g +0.42 us (se 0.52), level clause HOLDS; copies clause d +2.24 us FAILS (Phi n 2 +3.44, Qwen1.5 n 2 +3.05, JetMoE n 4 +0.22) |
| Q1-H rival v2 (no term b) | EXCLUDED | EXCLUDED | **EXCLUDED** | g_R +4.52 us: 4.10 us from mean g, against 3 se = 1.55 us and the 1.5 us band |
| Q1-H rival H8 | EXCLUDED | EXCLUDED | **EXCLUDED** | g_R +7.44 us: 7.02 us from mean g |
| Q1-H rival v3_all | NOT EXCLUDED | NOT EXCLUDED | **NOT EXCLUDED** | g_R 0.00 us on every block-scan pair: not separated from v3 by construction (registered) |
| Q1-C: CALL against DEAD on D = x15 - x9 (Qwen1.5 SHARED n 1) | SELECTED CALL | SELECTED CALL | **SELECTED CALL** | x9 -8.23 us, x15 -7.65 us, D +0.58 us; CALL at 0, DEAD at -5.47 us; DEAD 6.05 us away against 2 se_D = 1.27 us |
| Q1-P (registered DESCRIPTIVE) | printed | printed | **DESCRIPTIVE** | 8 pairs, 2 clusters: Mixtral n 3 mean g -10.56 us (six pairs), Phi n 1 c9 +2.53, c15 -8.97 us |
| Q1-SK (registered OPEN) | printed | printed | **OPEN** | Phi n 2 skewed minus control: -0.95 us (4 skewed, 6 control pairs) |
| **(c) OPEN-L2SKEW**, OLMoE w1 byte page: nearest of v3, ROWKEY39.5, ROWKEY60 NAMED iff next >= 2x and every skewed row of the nearest within 5% | NAMED ROWKEY60 | (one view) | **NAMED ROWKEY60** (OPEN: no model change follows) | 10 skewed w1 rows: rms ROWKEY60 1.99%, ROWKEY39.5 8.10% (4.06x), v3 15.33%; ROWKEY60 worst -3.21% |
| **(d) G3**: token order, 90% cluster-t CI within +-0.36% (6 page clusters, df 5) | PASS | PASS | **PASS** | 36 cells: mean +0.001%, CI [-0.013, +0.015]%, Q 0.84 (df 35), p 1.00 |
| **(e) E1 rival LT3** (S without last-wave quantisation) | EXCLUDED | EXCLUDED | **EXCLUDED** | 62 lever cells: 1.83% against v3's 0.68% |
| E1 rival v2 | UNDECIDED (0) | UNDECIDED (0) | **UNDECIDED** | 0 registered lever cells (by construction) |
| E1 rival H8 | UNDECIDED (5) | UNDECIDED (5) | **UNDECIDED** | 5 lever cells, under 6 (by construction) |
| E1 rival v3_all | UNDECIDED (3) | UNDECIDED (3) | **UNDECIDED** | 3 lever cells (by construction) |
| E1 rival v3_noca | UNDECIDED (0) | UNDECIDED (0) | **UNDECIDED** | 0 lever cells (by construction) |

### (a) Does CLEAN now certify more than OLMoE?

**Yes.** Rental 6's CLEAN view held OLMoE's three pages alone (42 cells, rms 0.54%, worst -1.47%,
mean +0.16%; board ff9942); Mixtral and Phi met the bar only in ALL because R3's histogram G1
removed their pages. Under the registered rental-7 rule (r7-median-run3, rebuilt from cells.csv),
every Mixtral and Phi page is in CLEAN, and each shape is **CERTIFIED** (its CLEAN cells meet the
bar with 3 of its 3 pages in CLEAN, and E1-SKEW's registered verdict holds):

| shape | source | CLEAN cells | rms | worst | mean | reading |
|---|---|---:|---:|---:|---:|---|
| Mixtral 8x7B (n 3, 6, 16) | rental 7, board 02e86e, fresh BLIND draws | 42 | 0.51% | -1.22% | -0.17% | CERTIFIED |
| Phi-3.5-MoE (n 2, 4, 12) | rental 7, board 02e86e, fresh BLIND draws | 42 | 0.69% | -1.61% | -0.31% | CERTIFIED |
| OLMoE-1B-7B (n 3, 6, 12) | rental 6, board ff9942 (as registered there) | 42 | 0.54% | -1.47% | +0.16% | rental 6's E1-SKEW HOLDS (its verdicts stand) |
| pooled, rental 7 | the registered pool | 84 | 0.61% | -1.61% | -0.24% | HOLDS |
| pooled, three models (DESCRIPTIVE, two boards, not a registered pool) | | 126 | 0.59% | -1.61% | -0.11% | printed |

The P20 view beside it (DESCRIPTIVE): E1-SKEW HOLDS on 83 cells (rms 0.59%, worst -1.61%), both
shapes CERTIFIED; it loses one cell (Phi skb DW SHARED n 2, 5 of 9 clean). Rental 5's tighter bands
(rms 1.2%, worst 3%, mean 0.6%) also hold.

**Does v3 now meet the working bar (rms <= 2%, no cell beyond 5%, here also \|mean\| <= 1%) on R3's
tile across several models in CLEAN?** Yes. With rental 7, the CLEAN view certifies it on Mixtral
8x7B and Phi-3.5-MoE (rental 7) and on OLMoE (rental 6): three models, two boards, every CLEAN
cell within 1.61%. Four limits go with it. (1) The cells are R3's tile 32 x 64 x 64 w8 s4 at G = 8.
The registration says nothing here is the working bar on vLLM's own tile, and **the tile stage
(vLLM's tiles) is still pending.** (2) These cells are floor-bound, so the bytes do not reach time
(rental 6's decomposition), and E1 tests the timing model, not the byte model. (3) The
uniform/balanced controls are SEEN replicates; only the 48 skewed cells are new draws. (4) Q1-H,
the registered test of term (b) on control pairs, FAILS on its copies clause (b, below). That is a
gap-model result on the SHARED arm at 15 declared copies, not an E1 cell.

### Pages each view counts: the rebuilt G1, clean counts and lost cells

R3 is byte-identical to rental 6's pin, so its own stored G1 (r5-any: one drift-flagged repeat
fails the page) reads FAIL on the six Mixtral and Phi skew pages and on Phi q9, as expected. The
scorers never read it. They rebuild every page from cells.csv under r7-median-run3 (a repeat is
non-clean only if lock-slipped, run-guarded by 3 or more consecutive reads 1694 MHz or lower, or not
ok; G1 fails past 10% non-clean rows or on a clock clause) and print the P20 view beside it.

| page | cells | repeat rows | R3 stored G1 (not read) | rebuilt G1 (r7) | r7 non-clean rows | r7 lost cells | min clean / cell | P20 G1 | P20 non-clean (drift-flagged) | P20 lost cells |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---|
| mixtral-8x7b ska | 24 | 216 | FAIL | **PASS** | 0 (0.00%) | 0 | 9 | PASS | 3 (1.39%) | none |
| mixtral-8x7b skb | 18 | 162 | FAIL | **PASS** | 0 | 0 | 9 | PASS | 4 (2.47%) | none |
| mixtral-8x7b skc | 18 | 162 | FAIL | **PASS** | 0 | 0 | 9 | PASS | 7 (4.32%) | none |
| phi-3.5-moe ska | 24 | 216 | FAIL | **PASS** | 0 | 0 | 9 | PASS | 13 (6.02%) | none |
| phi-3.5-moe skb | 18 | 162 | FAIL | **PASS** | 0 | 0 | 9 | PASS | 14 (8.64%) | DW / SHARED / n 2 (5 of 9) |
| phi-3.5-moe skc | 18 | 162 | FAIL | **PASS** | 0 | 0 | 9 | PASS | 10 (6.17%) | none |
| qwen1.5-moe-a2.7b q9 | 4 | 36 | PASS | **PASS** | 0 | 0 | 9 | PASS | 0 | none |
| qwen1.5-moe-a2.7b q15 | 4 | 36 | PASS | **PASS** | 0 | 0 | 9 | PASS | 0 | none |
| phi-3.5-moe q9 | 4 | 36 | FAIL | **PASS** | 0 | 0 | 9 | PASS | 1 (2.78%) | none |
| phi-3.5-moe q15 | 4 | 36 | PASS | **PASS** | 0 | 0 | 9 | PASS | 0 | none |
| jetmoe-8b q9 | 6 | 54 | PASS | **PASS** | 0 | 0 | 9 | PASS | 0 | none |
| jetmoe-8b q15 | 6 | 54 | PASS | **PASS** | 0 | 0 | 9 | PASS | 0 | none |
| byte olmoe-1b-7b l2open `r3c-g8` | 20 | | (no G1: counter page) | | | | | | | |

Totals: 1,332 timed repeat rows; under r7 every one is clean and no cell is lost (every cell 9 of 9).
Under P20, 52 rows are drift-flagged (3.9%) and one cell is lost. Every cell's median clock is
1710.0 MHz on every page; there is no lock slip and no run-guarded repeat in the session. The dips
are there: 668 of the 1,332 repeats carry at least one read of 1694 MHz or lower (1,307 of 19,980 reads, down to 1515 MHz),
and the longest run of consecutive dipped reads is 2 (3 repeats, Mixtral ska and Phi ska). The
r7 rule keeps them, as registered (an isolated dip, at an end or inside, stays in). The counter page
passes rental 6's clock rule: lock_gate.check_page PASS, 20 of 20 cells in [1640, 1710] MHz.

G5 passes on every page: the histogram files' sha256, every row's counts sha256, an exact
bincount, the shuffle seed (20261020 on the skew pages, rental 6's on the Q1 pages, 20261030 on
the byte page), and `--slip-policy cell --min-clean-repeats 0` on every timed page.

### (a) E1-SKEW by shape, tread and arm (DESCRIPTIVE breakdown of the registered statistic)

| tread (rank) | Mixtral NATIVE / SHARED | Phi NATIVE / SHARED |
|---|---|---|
| first (n 3, 2) | mean -0.88% / -0.51% (rms 0.90% / 0.69%) | -1.11% / -1.00% (rms 1.16% / 1.04%) |
| second (n 6, 4) | -0.11% / -0.06% | -0.19% / -0.19% |
| third (n 16, 12) | +0.26% / +0.26% | +0.30% / +0.32% |

Per label: PT rms 0.72%, PW 0.68%, PW2 0.59%, DW 0.78%, uniform 0.47%. Per arm: NATIVE 42 cells rms
0.65%, SHARED 42 rms 0.56%. The pattern of rental 6 repeats on fresh draws and a new board: v3
predicts about 1% too little time at each shape's first tread (Phi n 2 again the largest, worst
-1.61%), and about 0.3% too much at the third. The residual tracks the tread, not the skew label.
Skewed cells read 0.59 to 0.78% rms against uniform's 0.47%.

## (b) Q1: the S - N gap

All six Q1 pages ran (the driver's `fresh_tag` fix gave units 13 and 14 the tags `-p2-11` and
`-p2-12`). JetMoE n 1 and n 2 are host-bound by rental 3's rule on both pages and are out of every
Q1 test; the registered Q1-H population does not use them.

| cluster | pairs | mean g | g_R v2 | g_R H8 | g_R v3_all |
|---|---:|---:|---:|---:|---:|
| jetmoe-8b n 4 (c9, c15) | 2 | -0.81 us | +4.46 | +7.45 | 0.00 |
| mixtral-8x7b n 6 (A, B, C controls; SEEN-SELECTED) | 6 | -0.07 us | +4.28 | +7.44 | 0.00 |
| phi-3.5-moe n 2 (A, B, C controls, c9, c15) | 8 | -0.36 us | +4.79 | +7.44 | 0.00 |
| phi-3.5-moe n 4 (A, B, C controls; SEEN-SELECTED) | 6 | +1.24 us | +4.37 | +7.44 | 0.00 |
| qwen1.5-moe-a2.7b n 2 (c9, c15) | 2 | +3.80 us | +4.70 | +7.45 | 0.00 |
| all (pair-weighted) | 24 | **+0.42 us** (se 0.52) | +4.52 | +7.44 | 0.00 |

The level clause holds: v3's gap is within 0.42 us of the measured one on the 24 pairs, and the
rivals that drop term (b) (v2) or refit h (H8) sit 4.1 and 7.0 us away, past 3 se. **The copies
clause fails.** On the same board, going from 9 to 15 declared copies moves g by +3.44 us at Phi
n 2 (c9 -0.68, c15 +2.76) and +3.05 us at Qwen1.5 n 2 (c9 +2.27, c15 +5.32), and by +0.22 us at
JetMoE n 4. The pair-weighted d is +2.24 us against the 1.5 us band. The shift is in the SHARED arm.
NATIVE runs the same call on both pages and moves -0.19 us (Phi n 2) and -1.12 us (Qwen1.5 n 2)
between them (`docs/paper/noise_floors.md`, rental 7). The registration labels the copies clause
BLIND (the c15 SHARED cells are the only blind cells of Q1-H), so this is the one blind reading in
Q1-H, and it fails. The design's 12 pairs alone (DESCRIPTIVE) read the same: level +0.26 us, d
+2.24 us, FAILS.

**Q1-C** reads CALL: at Qwen1.5 SHARED n 1 the 15-copy residual equals the 9-copy one within 0.58
us (x9 -8.23, x15 -7.65), where DEAD would have moved it by -5.47 us. The n = 1 SHARED residual of
about -8 us is a per-call cost, not a dead-CTA cost.

**Q1-P (DESCRIPTIVE as registered).** The six Mixtral n 3 control pairs read g -3.90 and -6.97 us
on page A and -12.41 to -13.98 us on B and C (mean -10.56 us). On these pairs v3_all predicts
-10.61 us, H8 -3.98 and v2 0.00. Rental 6 read the same population at -6.38 us (v3_all then -10.43,
v3 nearer by 4.05 us). Rental 7 moves the descriptive reading to v3_all, against the block-scan
path condition. Phi n 1 reads +2.53 us at c9 and -8.97 us at c15 (v3_all -8.76 / -8.74). Q1-P is
registered descriptive and decides nothing. It is the one population where the path condition
(v3 against v3_all) shows, and on rental 7's board it does not favour the condition.

**Q1-SK (OPEN):** Phi n 2 skewed pairs minus control pairs -0.95 us (P20 view -1.72 us).

The P20 view (DESCRIPTIVE) reads every Q1 verdict the same: Q1-H FAILS (mean g +0.46 us, d +2.22
us), v2 and H8 EXCLUDED, Q1-C SELECTED CALL.

## (c) The OLMoE w1 byte page (OPEN-L2SKEW): the C = 60 MiB capacity-cliff candidate

The page (unit 15, `2026-10-14-nvidia_gh200_480gb-olmoe-1b-7b-l2open-r3-counters/lock1710/r3c-g8.json`)
carries exactly the draws the registration priced (seed base 20261030; G5 PASS on the file sha256,
the shuffle seed and every cell's counts sha256). So every prediction below was fixed before the
page ran (`docs/registered/2026-10-13-rental7-l2open-gh200.json`, `byte_predictions`). The registered
pre-page separations on these rows were: v3 against ROWKEY60 rms 17.6%, ROWKEY60 against ROWKEY39.5
rms 8.4% (mainly at n 4, -14.2 / -11.7%).

| n | label | arm | q_v3 | q_ROWKEY39.5 | q_ROWKEY60 | q counted | e v3 | e ROWKEY39.5 | e ROWKEY60 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | PT | NATIVE | 1.0504 | 1.1074 | 0.9506 | 0.9752 | +7.72% | +13.56% | -2.52% |
| 4 | PT | SHARED | 1.0504 | 1.1074 | 0.9506 | 0.9748 | +7.76% | +13.60% | -2.48% |
| 4 | PW | NATIVE | 1.0560 | 1.0891 | 0.9615 | 0.9751 | +8.30% | +11.70% | -1.40% |
| 4 | PW | SHARED | 1.0560 | 1.0891 | 0.9615 | 0.9753 | +8.28% | +11.67% | -1.42% |
| 8 | PT | NATIVE | 1.0449 | 0.9108 | 0.9022 | 0.9146 | +14.24% | -0.41% | -1.35% |
| 8 | PT | SHARED | 1.0449 | 0.9108 | 0.9022 | 0.9146 | +14.25% | -0.41% | -1.35% |
| 8 | PW | NATIVE | 1.0494 | 0.9350 | 0.9020 | 0.9131 | +14.92% | +2.39% | -1.22% |
| 8 | PW | SHARED | 1.0494 | 0.9350 | 0.9020 | 0.9131 | +14.93% | +2.40% | -1.21% |
| 16 | PT | NATIVE | 1.0402 | 0.8199 | 0.8062 | 0.8272 | +25.75% | -0.88% | -2.54% |
| 16 | PW | NATIVE | 1.0383 | 0.8256 | 0.8102 | 0.8370 | +24.05% | -1.36% | -3.21% |
| **rms** (10 skewed w1 rows) | | | | | | | **15.33%** | **8.10%** | **1.99%** |

q = (dram_bytes_read - n x operand_per_tile_w1) / W_w1; e = q_pred / q_counted - 1.

**NAMED ROWKEY60** (OPEN: no model change follows). ROWKEY60 is nearest (rms 1.99%), the next
(ROWKEY39.5, 8.10%) is 4.06x farther, and every skewed row of ROWKEY60 is within 3.21% (the 5%
clause). The row-granular w1 A key with one exact LRU at the card's 60 MiB reads every skewed row.
The 39.5 MiB capacity fails at n 4 (+11.7 to +13.6%), where 60 MiB separates from it, as the
registration said it would. v3 over-predicts skewed w1 reads by +7.7% at n 4, +14.2 to +14.9% at
n 8 and +24.0 to +25.8% at n 16, larger than rental 6's +4.8 to +20.6% at n 3 to 12. ROWKEY60 under-predicts
every skewed row (-1.2 to -3.2%, mean -1.9%), so it carries a small one-sided bias.
The controls (printed): uniform and balanced w1 rows read ROWKEY60 rms 0.03%, v3 0.99% (worst
+2.23% at n 16); w2 under v3 rms 3.48%, worst +4.34% (rental 6's OLMoE w2 level again). The C = 60
MiB choice was SEEN (swept over rentals 5 and 6). Its numbers on this page are BLIND, n 8 is a
new tread, and n 4 and n 16 are SEEN-SELECTED treads with new draws.

The page's own analyse step exits 3 on V1 ("cells missing ['shared/16']; 3 treads"). The registered
design has no SHARED cell at n 16 and three treads; R3's ladder analyse expects four. Rental 6's
byte pages exit 3 the same way. It is not a gate of this registration, and the counter clock rule
decides what counts.

## (d) G3: token order

d = T(uniform-shuffled) / T(balanced) - 1 - delta_v3; clusters are the six pages; margin +-0.36%.

| view | cells | pages | mean d | se | t (df) | 90% CI | Q (df), p | verdict |
|---|---:|---:|---:|---:|---|---|---|---|
| ALL = CLEAN | 36 | 6 | +0.001% | 0.007% | 2.02 (5) | [-0.013, +0.015]% | 0.84 (35), 1.00 | EQUIVALENT: PASS |
| P20 (DESCRIPTIVE) | 36 | 6 | +0.004% | 0.008% | 2.02 (5) | [-0.012, +0.019]% | 1.05 (35), 1.00 | PASS |
| registered | | | | | | | | **PASS** |

Page means: Mixtral A +0.026%, B +0.009%, C +0.008%; Phi A -0.004%, B -0.023%, C -0.007%. On these
two models token order moves no cell by more than a few hundredths of a percent. OLMoE's -0.18 to
-0.30% (rentals 5 and 6) is not timed here. G3 passing voids nothing, so the strict void-all branch
and its descriptive E1 do not apply.

## (e) Model v3 against its rivals

| rival | what it drops | registered test that separates it (rental 7) | reading |
|---|---|---|---|
| v2 | M4 bytes and term (b) | Q1-H rival rule; E1 (0 lever cells by construction) | **EXCLUDED by Q1-H** (g_R +4.52 us against measured +0.42 us, se 0.52); E1 UNDECIDED. DESCRIPTIVE: over all 84 E1 cells v2 reads rms 0.56% against v3's 0.61%, so on time these shapes do not separate them |
| H8 | the 4-model fit of h | Q1-H rival rule; E1 (5 lever cells) | **EXCLUDED by Q1-H** (g_R +7.44 us); E1 UNDECIDED. DESCRIPTIVE E1 rms 0.59% |
| LT3 | last-wave quantisation | E1 rival rule (62 lever cells); SKEW-RATIO (4) | **EXCLUDED on E1** (1.83% against 0.68%; rental 6 was INCONCLUSIVE); SKEW-RATIO UNDECIDED. DESCRIPTIVE E1 rms 1.57%, mean -1.16% |
| v3_all | the block-scan path condition | E1 (3 lever cells); Q1-H cannot separate (g_R 0) | **UNDECIDED / NOT EXCLUDED**. DESCRIPTIVE Q1-P: Mixtral n 3 reads -10.56 us, v3_all -10.61, v3 0: on rental 7's board the descriptive reading favours v3_all |
| v3_noca | content-keyed A tiles | E1 (0 lever cells) | **UNDECIDED** (no time cell moves; its byte exclusion is rental 6's B-CA) |
| U (no skew) | the skew term | SKEW-RATIO rival rule | **EXCLUDED** (12.85% against 0.50% on 32 lever ratios) |
| ROWKEY60 / ROWKEY39.5 | (OLMoE w1 byte rivals) | OPEN-L2SKEW | **ROWKEY60 NAMED** over v3 and ROWKEY39.5 on skewed w1 bytes (OPEN; bytes do not reach time at G = 8 on these cells) |

v3 holds the time bar (E1, SKEW-RATIO). Two of its rivals fall on Q1-H, LT3 falls on E1, and U falls
on the ratios. v3 itself FAILS Q1-H on the copies clause, and its w1 byte prediction under skew is
beaten by the row-key rival at 60 MiB. Both are outside the time cells E1 scores.

## Scorer check (no post-page fix)

Checked against the pages before the verdicts were read:

- **cells.csv**: one row per timed repeat with `status`, `ms_p50`, `sm_clock_load_mhz`,
  `clock_drift_ok`, `host_bound`, and `clock_samples_mhz` (15 space-separated integer reads on
  every one of the 1,332 rows). `repeat_rows` keys on `histogram=` in `detail`, `tiles` and `arm`.
  The rows per cell equal the report's `repeat_rows` on every cell (otherwise the page would be
  NOT SCORED). The run guard is exercised by real data (668 repeats with a dipped read, longest run
  2), so a run guard reading an empty list would not have shown up as zero dips.
- **R3 settings**: `slip_policy` "cell" and `min_clean_repeats` 0 on every timed report, as the
  page check requires.
- **G5**: the registered file sha256, each row's `counts_sha256` and `bincount_ok`, and the
  shuffle seed, read off `report.histogram` and the `treads_table` rows.
- **G2**: the registered-prediction host rule drops JetMoE n 1 and n 2 on both Q1 pages (8 cells)
  and nothing else. **G4**: every rebuilt cell clock is 1710.0 MHz.
- **align_probe**: present on every timed page; no Q1 pair skipped.
- **Counter page**: `design.histogram` (sha256, shuffle seed), each cell's `counts_sha256`,
  `per_gemm.{w1,w2}.dram_bytes_read`, and the clock rule's per-cell clocks, zipped with the cells
  in page order (strict).

Each reads what the page carries. Two runs of each scorer are byte-identical.

## Fixes and readings flagged for the owner

- **`register.py --check` (registration generator, not a scorer; no verdict moves).** With
  rental 7's pages published, `reg7.published_counts_sha` globbed them. Every rental-7 draw then
  read "SEEN (a published page carries this counts sha256)" and `--check` reported the skew and
  l2open registrations as DIFFERS (leakage counts 0 -> 48, 48 prediction labels, 2 l2open control
  labels). `register.py` now seeds that set from the sessions dated on or before 2026-10-13 without
  "rental7" in the name, as `reg7.seed_pricer` already does for the align proxy and rental 6's
  `register.py` did at its own publication. `reg7.py` (a pinned file) is unchanged, and `--check`
  reads `same` on all eight files.
- **The descriptive Q1-P reading moved to v3_all** (b). Registered descriptive, so no verdict. It
  is the first evidence on R3's tile that the block-scan path condition may be wrong at Mixtral
  n 3. A registered test of it would need lever cells (E1 has 3).
- **The copies clause fails on BLIND cells** (b). Q1-H FAILS as registered. Whatever term (b)
  should carry for SHARED at 15 copies, the registration adds no term and none is fitted here.
