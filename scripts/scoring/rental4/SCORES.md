# Rental 4 scores (2026-10-07)

The five scorers in this directory, run at 85ef38c on
`results/published/2026-10-07-nvidia_gh200_480gb-rental4-session` (798 run files, each matching
the run branch's SHA256SUMS; the 14 compiled `cuda_utils*.so` files are left out):

    python scripts/scoring/rental4/score_<part>.py . results/published/2026-10-07-nvidia_gh200_480gb-rental4-session scripts/scoring/rental4

Outputs: `dead`, `occlaw`, `perturb`, `stamps`, `hw` `.score.{json,txt}` beside the scorers.
`dead` and `occlaw` carry `addendum.verdicts`, every verdict path with its ALL, CLEAN and
registered reading (rental 2's addendum rule 3). The sixth registration,
`2026-10-06-rental4-nativegates-gh200`, is rental 5's gate definition: no rental-4 unit runs a
NATIVE-only histogram page and it has no scorer, so it scores nothing. No scorer crashed or
refused. One post-page fix, to reading code in `score_stamps.py`, changes no verdict (end of file).

## Pages each view drops

| page | failed validity gates | ALL | CLEAN |
|---|---|---|---|
| timed A1 qwen2-57b-a14b-tp8 c9 (`94e2fecb`, tag -p2-), c15 (`7f595452`, -p2b-) | none (C1 UNKNOWN, a claim gate) | counted | counted |
| timed A2 olmoe-1b-7b c9 (`183ddc17`, -p2c-), c15 (`a71cd9a0`, -p2d-) | none (C1 UNKNOWN) | counted | counted |
| byte pages `r3c-g64` k64s4, k32s4, k128s4, k64s6, k64s2, k64s3, k64s8 | none (V0 to V10 PASS; C1 FAIL, a claim gate) | counted | counted |

Every timed page's worst cell clock is 1710.0 MHz; V10 on every byte page reads the lock in force
on the nvidia-smi bracket. Each c15 page is labelled NOT JOINABLE TO COUNTER BYTES.

## Part A, the dead-CTA copies contrast (`2026-10-06-rental4-dead-gh200`)

The question: what one dead CTA costs, from the change in SHARED - NATIVE when the declared slots
go from 9 to 15 copies (576 to 960 slots), NATIVE's call being the same call on both pages.

| contrast | Delta measured (raw) | D (d 0.995 ns, CAL) | M (d 1.333 ns, current) | D2 | SLOT | K0 | FIXED | null control |
|---|---|---|---|---|---|---|---|---|
| A1 qwen2-57b-a14b-tp8 | **22.93 us** (22.70) | 20.73, z +1.79 | 27.77, z -5.38 | 21.46, +1.63 | 21.09, +2.04 | 24.43, -1.67 | 0, +25.5 | +0.77 us within 5.19: PASS |
| A2 olmoe-1b-7b | **24.45 us** (24.29) | 23.69, z +0.58 | 31.74, z -8.10 | 23.33, +1.24 | 21.70, +3.06 | 23.69, +0.84 | 0, +27.2 | 0.00 us within 8.11: PASS |

z at the registered sigma_noise 0.9 us (D's adds its fit se, sigma_D 0.83 / 0.95 us). Delta is the
median over n = 3..9 of Delta_n, less the alignment kernel's growth (median A_n -0.22 us on A1,
-0.16 on A2; raw in brackets).

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| D (candidate d 0.995 ns, kappa 0.325) | HOLDS | HOLDS | **HOLDS** on A1 and A2 | z +1.79, +0.58 (at 0.45 us, printed: +2.32, +0.72) |
| M (current DEAD_CTA_NS 1.333, k_w 0.5) | EXCLUDED | EXCLUDED | **EXCLUDED** on A1 and A2 | z -5.38, -8.10 (printed: -10.76, -16.2) |
| D2 (0.93 / 1.03 ns, SEEN-fitted) | HOLDS | HOLDS | **HOLDS**; D / D2 NOT SEPARATED (by design) | z +1.63, +1.24 |
| SLOT (t 4.05 ns / occ) | EXCLUDED on A2 | same | **EXCLUDED on A2**; D / SLOT NOT SEPARATED (by design) | z +2.04, +3.06 (see readings left to the owner) |
| K0 | NOT EXCLUDED | same | **NOT EXCLUDED** | z -1.67, +0.84 |
| FIXED | EXCLUDED | same | **EXCLUDED** on A1 and A2 | Delta 25 and 27 sigma above 0 |
| K5 | NOT TESTED | same | NOT TESTED | equals D on A1 and A2; A3 cut |
| per-GEMM d (printed, not a verdict) | | | | d_w2 = 22.93 us / 20,832 dead w2 CTAs = **1.10 ns** (A1); d_w1 = (24.45 - 11,904 x 1.10 ns) / 11,904 = **0.95 ns** (A2) |

The measured SHARED - NATIVE, us, per tread (G = 8, the treads_table's ms_p50):

| n | A1 c9 | A1 c15 | A1 Delta_n | A2 c9 | A2 c15 | A2 Delta_n |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -4.06 | 0.13 | (not scored) | 21.02 | 45.28 | (not scored) |
| 2 | 28.83 | 32.29 | (not scored) | 18.50 | 45.31 | (not scored) |
| 3 | 29.92 | 53.25 | 23.33 | 21.34 | 46.40 | 25.06 |
| 4 | 29.25 | 52.16 | 22.91 | 20.35 | 44.64 | 24.29 |
| 5 | 29.28 | 52.03 | 22.75 | 22.85 | 46.11 | 23.26 |
| 6 | 30.53 | 52.64 | 22.11 | 22.26 | 46.11 | 23.86 |
| 7 | 31.09 | 53.79 | 22.70 | 23.14 | 47.58 | 24.45 |
| 8 | 31.68 | 54.19 | 22.51 | 22.59 | 46.83 | 24.24 |
| 9 | 32.83 | 55.36 | 22.53 | 24.11 | 48.59 | 24.48 |

Delta_n is flat in n, as the dead count is n-free (A1 22.1 to 23.3 us, A2 23.3 to 25.1). On A1 the
w1 dead CTAs are hidden at both declarations (D, M and SLOT all give w1 0), so A1 reads w2's d
alone; on A2 both GEMMs carry dead CTAs. Both contrasts sit 2.2 and 0.8 us above D and 4.8 and
7.3 us below M. On qwen2-tp8, n = 1 and 2 at c15 are not in the dead regime (A1's c15 S - N reads
0.13 and 32.3 us there); the statistic reads n = 3..9, as registered.

## Part B', the k-step law at BLOCK_K / num_stages (`2026-10-06-rental4-occlaw-gh200`)

c by least squares of sm__cycles_elapsed.avg on the floor-bound NATIVE and SHARED cells n = 4..9
(12 per page and GEMM; every cell reads DRAM at 0.11 to 0.31 of 4022 GB/s), F held at CAL.

| page | w1 c (cycles per k-step) | w2 c | recorded CTAs/SM (w1 / w2) | registered | w1 r_meas | w2 r_meas |
|---|---:|---:|---|---|---:|---:|
| k64s4 (base, SEEN config) | 346.2 | 344.0 | 5 / 4 | 5 / 4 | 1 | 1 |
| k32s4 | 245.2 | 234.3 | 5 / 5 | 5 / 5 | 1.3979 | 1.3076 |
| k128s4 | 686.9 | 684.1 | 3 / 3 | 3 / 3 | 0.9924 | 0.9951 |
| k64s6 | 366.1 | 368.2 | 3 / 3 | 3 / 3 | 1.0550 | 1.0597 |
| k64s2 | 348.1 | 329.1 | 5 / 5 | 4 / 4: RE-KEYED | 1.0053 | 0.9631 |
| k64s3 | 342.6 | 336.1 | 5 / 4 | 5 / 4 | 0.9900 | 0.9804 |
| k64s8 | 409.6 | 420.4 | 2 / 2 | 2 / 2 | 1.1748 | 1.1885 |

e = r_meas / r_pred - 1, per law (2% per page and GEMM):

| page | MVA2 w1 / w2 | LK | PS | LITTLE | OCC |
|---|---|---|---|---|---|
| k32s4 | -0.6 / **-3.5** | **+2.5** / -1.2 | **+39.8** / **+30.8** | **+2.5** / -1.2 | - |
| k128s4 | -0.7 / +0.0 | -0.8 / -0.5 | -0.8 / -0.5 | -0.8 / -0.5 | - |
| k64s6 | **+2.3** / **+3.5** | **+5.5** / **+6.0** | **+5.5** / **+6.0** | **+5.5** / **+6.0** | -1.8 / +1.9 |
| k64s2 (re-keyed) | +0.5 / **-3.3** | +0.5 / **-3.7** | +0.5 / **-3.7** | **-50.3** / **-49.6** | +0.5 / -1.3 |
| k64s3 | -1.0 / -2.0 | -1.0 / -2.0 | -1.0 / -2.0 | **-4.2** / **-4.8** | -1.0 / -2.0 |
| k64s8 | **+2.1** / **+5.3** | **+17.5** / **+18.9** | **+17.5** / **+18.9** | **+17.5** / **+18.9** | +0.6 / **+6.0** |

| prediction | ALL | CLEAN | registered verdict | key numbers |
|---|---|---|---|---|
| MVA2 | FALSIFIED | FALSIFIED | **FALSIFIED** | 6 of 12 beyond 2%; rms 2.56% (in-sample at k32s4, k64s4, k64s8) |
| LK | FALSIFIED | FALSIFIED | **FALSIFIED** | 6 of 12; rms 7.93% |
| PS | FALSIFIED | FALSIFIED | **FALSIFIED** | 7 of 12; rms 16.5% |
| LITTLE | FALSIFIED | FALSIFIED | **FALSIFIED** | 9 of 12; rms 21.9% |
| OCC (BK 64 pages only) | FALSIFIED | FALSIFIED | **FALSIFIED** | 1 of 8 (k64s8 w2 +6.0%); rms 2.51% |
| part B' | UNDECIDED | UNDECIDED | **UNDECIDED, no survivor** | |

What the pages show without a law: c doubles from BK 64 to BK 128 (1.98x and 1.99x, PS-like),
falls only to 0.71x and 0.68x at BK 32 (floor-like), and at BK 64 rises as occupancy falls (345 at
4 to 5 CTAs/SM, 366 to 368 at 3, 410 to 420 at 2). The base reproduces PS's CAL c64 344.1 to 0.6%.
B' and the stamps K block are one test of the k-step law; K did not run.

## The perturbation gate (`2026-10-06-rental4-perturb-gh200`)

| variant | stamps | median | worst | configs whose registers differ (plain -> copy, CTAs/SM) | all-off SASS | VM gate.env |
|---|---|---:|---:|---|---|---|
| u8-stf | cta | 1.68% | 2.22% | w1 48 -> 44 (5, 5); w2 55 -> 48 (4 -> 5) | equal | FAIL (agrees) |
| u9-stk64s4 | iter, 64 marks | 7.27% | 9.06% | w2 55 -> 45 (4 -> 5); w1 48 = 48 | equal | FAIL |
| u10-stk32s4 | iter | 8.10% | 8.62% | w1 47 -> 40 (5 -> 6); w2 48 -> 43 (5, 5) | equal | FAIL |
| u11-stk128s4 | iter | 7.29% | 7.84% | w1 57 -> 56; w2 64 -> 58 (3, 3) | equal | FAIL |
| u12-stk64s8 | iter | 9.51% | 9.76% | w2 55 -> 48 (2, 2); w1 48 = 48 | equal | FAIL |
| u13-sttail (8x22B) | iter, 112 | 7.53% | 16.68% | w2 55 -> 45 (4 -> 5) | equal | FAIL |
| u14-sttail (8x7B) | iter, 112 | 5.90% | 14.65% | w2 55 -> 45 (4 -> 5) | equal | FAIL |
| u15-stdead4 | cta | 1.45% | 3.58% | w1 48 -> 44 (5, 5); w2 55 -> 48 (4 -> 5) | equal | FAIL |
| u16-stdead2 | cta | 2.14% | 5.64% | w1 48 -> 44; w2 55 -> 48 (2, 2) | equal | FAIL |

**FAIL on 9 of 9 variants**: every one fails the median, the worst and the occupancy leg; the
upstream check passes (vLLM 0.27.1 fused_moe.py and its excerpt match their sha256). The SASS leg
ran on all 18 configs and every all-off copy's SASS equals the plain kernel's (464 to 696
instructions), so the stamped copy differs from the plain kernel by the stamps' own code alone.
The diagnosis (ptxas's register allocation around the stamps; the cost of the stamps at equal
occupancy) is in `docs/FINDINGS.md`, rental 4; the proposed fix is outside this repository copy.

## The stamps (`2026-10-06-rental4-stamps-gh200`)

| unit | block | verdict |
|---|---|---|
| stf (8x7B) | F | NOT SCORED: NOT RUN (gate FAIL: median 1.68%, worst 2.22%) |
| stk64s4, stk32s4, stk128s4, stk64s8 (OLMoE G = 64) | K | NOT SCORED: NOT RUN (gate FAIL: median 7.27 / 8.10 / 7.29 / 9.51%, worst 9.06 / 8.62 / 7.84 / 9.76%) |
| sttail 8x22B, sttail 8x7B | T | NOT SCORED: NOT RUN (gate FAIL: median 7.53 / 5.90%, worst 16.68 / 14.65%) |
| stdead4, stdead2 (OLMoE G = 8) | D | NOT SCORED: NOT RUN (gate FAIL: median 1.45 / 2.14%, worst 3.58 / 5.64%) |
| K classes (PS, LK=LITTLE, MVA2~OCC) | K | NOT SCORED |

As registered: a unit whose gate is not PASS has no page, and the drop-group st goes together.

## The hardware constants (`2026-10-06-rental4-hw-gh200`)

| quantity | this rental (1710 lock) | RRZE GH200 (SEEN third-party, 1980 MHz) | check | verdict |
|---|---:|---:|---|---|
| far L2 latency (0.6 to 0.95 L, 12 cells) | 275.0 ns (470 cycles) | 253.4 ns (502 cycles) | +8.5%, band 5% | **DIFFERS** |
| DRAM latency (2.5 L and above) | no cell (gpu-latency timed out at 1.46 L) | 346.4 ns | band 10% | **NOT SCORED** |
| triad peak | 3781 GB/s | 3783 GB/s | -0.1%, band 3% | **CONSISTENT** |
| L2 size (the rulers' device query) | 62,914,560 B | 62,914,560 B | exact | **CONSISTENT** |
| near L2 latency (printed) | 152.4 ns (261 cycles) | 141.6 ns (280 cycles) | +7.6% | printed |
| stream read, init (printed) | 2651, 3947 GB/s | 2775, 3944 | -4.5%, +0.1% | printed |
| L2 bandwidth plateau, floor, half-way (printed) | 10,224 GB/s, 3891 GB/s, 0.81 L | | | printed |
| triad knee occupancy (printed) | 0.72 | | | printed |
| eta_mix = 3598 / ruler (printed) | read2d 0.943 (3815 GB/s), copy 0.980 (3673), add 0.965 (3728), gpubench triad 0.952 | | | printed |
| tau_mma (printed) | 3054 FLOP/clk/SM at 1710 from 689 TFLOP/s; the SM clock read 1395 MHz during the matmul, where it is 3744 | | | printed |

Both L2 latencies read longer in ns and shorter in cycles than RRZE's 1980 MHz run: neither the
fixed-ns nor the fixed-cycle reading of the registration's comparison holds, so the far-L2 DIFFERS
is a property of the clock, not evidence against the card. The stamps F block would have taken this
rental's constants only with all three latencies parsed; DRAM's is missing, so it would have fallen
back to RRZE's published values.

## Readings left to the owner (no scorer change)

- **SLOT on A2.** SLOT's rule (EXCLUDED beyond |z| 3) excludes it on A2 at z +3.06, 0.06 over the
  line at the registered sigma 0.9 us (6.11 at the printed 0.45). The registration's
  `not_separated` rule also says D / SLOT is "printed NOT SEPARATED whatever the pages read". The
  scorer prints the SLOT exclusion and does not print the NOT SEPARATED pairs (D / SLOT, D / D2).
  Both readings are reported here; which governs SLOT's standing is the owner's call.
- **tau_mma.** The registered print divides the matmul ruler by the 1710 lock; the ruler's own
  record puts the SM clock at 1395 MHz under that load (700 W). 3744 FLOP/clk/SM is the clock-read
  value, printed beside it.

## The post-page fix (changes no verdict)

`score_stamps.py`, `unit` (`git diff` of this directory). The registration's `missing` rule makes a
unit without a PASS gate or without a stamps.json NOT SCORED, and the session plan refuses an
ungated unit before it writes anything. The scorer looked for stamps.json first and printed "NOT
SCORED: 0 stamps.json for <model> <label>" for each refused unit, never reaching the gate. It now
reads the plan's variant id off `instr-variants.json`, and when no stamps.json exists and the
unit's gate.env line is not PASS it prints "NOT SCORED: NOT RUN (gate FAIL: median x%, worst y%;
variant <id>, refused by the driver)". The verdict (NOT SCORED) and every rule are unchanged; a
unit whose gate is PASS but whose page is missing keeps the old reading. The other four outputs
are byte-identical before and after. Test:
`tests/test_scoring_rental4.py::test_a_refused_stamps_unit_reads_not_run_with_its_gate_numbers`.
