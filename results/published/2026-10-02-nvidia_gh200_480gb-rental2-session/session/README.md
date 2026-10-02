# Rental 2: one NVIDIA GH200 480GB, 24 units of four registrations (docs/registered/README.md, 2026-10-01, and the 2026-10-02 gate addendum)

One rental ran `scripts/plans/rental2-2026-10.plan` unattended: the floor captures of parts 3 and 4 (tp8 floor2, tp4, tp2, tp4 at 1005 MHz), the knob byte pages of part 1 (B's lag law, C's BLOCK_K separator T5), the launch-floor rerun of part 2, then the tail. The four registrations, their scorers and the plan were committed at 2bf6fe3 before any page; the gate addendum (38898b4) was fixed after the gate results were seen and before any score. Every verdict below is the scorers' (`scripts/scoring/rental2/`, written and tested on synthetic pages before any page), run unchanged on this directory; the full tables are in `scripts/scoring/rental2/SCORES.md`, the scoring notes in `docs/registered/README.md` ("**Scored 2026-10-02**", one per part) and `docs/FINDINGS.md` (rental 2).

## Where it came from

Lambda Cloud instance 68226a2d781e4d3591438f30dbc80aa6, gpu_1x_gh200, us-east-3, launched
2026-10-02 05:22:58 UTC, terminated 07:41:41 UTC by the laptop's guard once every pushed file
had been verified (919 files match SHA256SUMS): 2 h 19 min, about $5.30 at $2.29/h. Board
`4da056` (`PREFLIGHT.txt`, a board no earlier session used), driver 570.148.08 upgraded to
580.105.08 before setup (`driver580.log`). `scripts/gh200_model_session.sh --plan
scripts/plans/rental2-2026-10.plan` ran at commit 2bf6fe3 (`commit.txt`); the plan's sha256
dbe9ae61... is in `gh200-driver/status`. This directory is the branch
`run-gh200-rental2-20261002t0522` at 7ea41cf; `SHA256SUMS.run-branch` and
`PUSHES.run-branch.txt` are its manifest and push log (43 pushes). Power limit 700 W,
persistence on (`power-set.txt`); the prelude locked 1005 MHz once, read it back and reset
(`lock plan: LOCK_PROBE_1005`, `lock probe: 1005 MHz held`); the 1710 lock was held for every
byte page and the launch floor (`lock-plan.txt`, `clocks-after-*.txt`).

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, design | start to end (UTC) | min | exit | pages (each page's gates not PASS) |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 1005 probe | 05:26:09 | 0 | 0 | |
| 2 | floor | mixtral-8x7b-tp8 floor2, G = 64, n = 1, 2, 6, 8, 10..16 | 05:26:14 to 05:28:16 | 2 | 3 | r3f-g64 (0); r3f-g64-lock1710 (3, FL1) |
| 3 | floor | mixtral-8x7b-tp4, G = 64, n = 1, 2, 4, 6, 8, 10..16 | 05:28:18 to 05:31:04 | 3 | 3 | r3f-g64 (0); r3f-g64-lock1710 (3, FL1) |
| 4 | floor | mixtral-8x7b-tp2, G = 64, n = 1, 2, 4, 6, 8, 10..16 | 05:31:06 to 05:35:06 | 4 | 3 | r3f-g64 (0); r3f-g64-lock1710 (3, FL1) |
| 5 | floor | mixtral-8x7b-tp4 at 1005 MHz, n = 2, 4, 6, 8, 12, 16 | 05:35:09 to 05:36:10 | 1 | 0 | r3f-g64-lock1005 (0) |
| 6 | bytes | tp2 l2base, G = 1, partition metrics | 05:36:14 to 05:39:51 | 4 | 0 | V10 |
| 7 | bytes | tp2 l2s8 (num_stages 8), partition metrics | 05:39:53 to 05:43:31 | 4 | 0 | none |
| 8 | bytes | tp4 l2base | 05:43:34 to 05:45:51 | 2 | 0 | V10 |
| 9 | bytes | tp4 l2s8 | 05:45:54 to 05:48:11 | 2 | 0 | V10 |
| 10 | bytes | tp8 l2s8 (negative control) | 05:48:14 to 05:49:52 | 2 | 0 | V10 |
| 11 | bytes | tp2 l2s6 | 05:49:55 to 05:53:32 | 4 | 0 | V10 |
| 12 | bytes | tp2 atbk64, G = 64 (BLOCK_K 64, 4 stages) | 05:53:35 to 05:57:12 | 4 | 0 | V7, V10; C1 REFUSE, C2, C6 |
| 13 | bytes | tp2 atbk32, G = 64 (BLOCK_K 32, 4 stages) | 05:57:16 to 06:00:53 | 4 | 0 | V7; C1 REFUSE, C2, C6 |
| 14 | bytes | tp2 atbk128, G = 64 (BLOCK_K 128, 2 stages) | 06:00:56 to 06:04:33 | 4 | 0 | C1 REFUSE, C2, C6 (page and analyse exit 1) |
| 15 | launchfloor | mixtral-8x7b-tp8, n = 1..9, E240 E0 E480 GR | 06:04:36 to 06:27:16 | 23 | 0 | |
| 16 | launchfloor | granite-3.0-3b-a800m, n = 1..9, E240 E0 E480 GR | 06:27:20 to 06:49:40 | 22 | 0 | |
| 17 | bytes | tp2 l2bk32, G = 1 | 06:49:44 to 06:53:22 | 4 | 0 | V6 |
| 18 | bytes | tp2 l2bk128, G = 1 | 06:53:25 to 06:57:03 | 4 | 0 | V10 |
| 19 | bytes | tp2 l2pad7, G = 1 (slot pad 7 rows) | 06:57:07 to 07:00:45 | 4 | 0 | V10 |
| 20 | bytes | tp2 ats8, G = 64 (8 stages) | 07:00:49 to 07:04:27 | 4 | 0 | V7; C1 REFUSE, C2 REFUSE, C6 |
| 21 | bytes | mixtral-8x7b l2base, G = 1 | 07:04:31 to 07:10:47 | 6 | 0 | none |
| 22 | bytes | mixtral-8x7b l2s8, G = 1 | 07:10:51 to 07:17:07 | 6 | 0 | V7 |
| 23 | launchfloor | jetmoe-8b, n = 1..4, E240 E0 E480 GR | 07:17:11 to 07:27:23 | 10 | 0 | |
| 24 | launchfloor | granite-3.0-1b-a400m, n = 1..9, E240 GR | 07:27:27 to 07:38:32 | 11 | 0 | |

132 minutes from the prelude to the last unit against the plan's 176-minute estimate; no unit
was dropped. The driver ends with exit 3 because the three 1710 floor captures fail FL1; the
recovery step found nothing to retake (`RECOVERY start: timed G to retake [], deep G to retake
[]`; `recovery/retake_auto.sh` is the script as it ran). A byte unit's exit is the lock block's
(0 on every unit); each page's own exit is in the ledger: 3 on a VALIDITY gate, 1 on a CLAIM
gate alone (tp2 atbk128 only), 0 on tp2 l2s8 and 8x7B l2base. The FL1 fails are the degenerate
f / t0 fit the gate audit describes (tp8 w1 1690.5 MHz + 3.9 us, w2 1672.6 + 0.6; tp4 and tp2
fits on the lock, 1695.6 to 1710.4 MHz, failed by one n = 1 w2 cell each, 1785 and 1773 MHz on
the t0-corrected per-cell clock); the null kernel reads 1684 to 1688 MHz on all three, and the
1005 capture fits 1005.0 MHz + 0.3 us on both GEMMs. No cell's own `sm_clock_mhz` is above its
lock (highest 1702.6 MHz at 1710), so the addendum's above-lock rule drops nothing.

## What it answered

Each verdict is the addendum's: ALL and CLEAN agree unless stated.

**Part 4, tp8's w1 floor** (base captures `results/*-mixtral-8x7b-{tp8-floor2,tp4-floor,tp2-floor}-r3-counters/r3f-g64.json`, primary): NEITHER. H_EST, FLUID, FLUID_LOW and H_NPN are all FALSIFIED. theta reads H_EST's side on tp8 (+0.54, sigma 0.28) and tp4 (+0.80, sigma 0.20), FLUID is out on tp4's theta, and H_EST is out on its level: delta_H -1.29% on tp4 w1 (2 sigma 1.08%) and -1.75% on tp2 w1 (0.62%). Per-set slopes: tp8 333.3 / 348.8 / 363.3 (H_EST 313.4 / 350.7 / 365.6), tp4 336.3 / 350.6 / 357.1, tp2 354.4 / 351.4. Co-primary per-cell test INCONCLUSIVE on both w1 GEMMs. w2 offset: H_EST HOLDS on all three shards, FLUID only on tp8 and tp4 (tp2 w2 +3.89% against +3.81%). The lock captures, printed: the same NEITHER, tp8 slopes flat (352.7 / 350.0 / 350.8, theta -0.04) on cells within 0.6% of the base cells.

**Part 3, the per-GEMM constant** (the three 1710 captures primary; the tp4 1005 capture): K1 HOLDS (L over 5000 cycles on 11.7% of 60 cells), K2 HOLDS (pooled w2 Z / u 0.254: tp4 0.310, tp2 0.198, band [0.25, 0.69]), K3 the cycle form HOLDS and the ns form is FALSIFIED (Z_1005 / Z_1710 0.936 w1, 0.907 w2, pooled 0.921; f_1005 / f_1710 0.597; sigma 0.075), K4 HOLDS (D_imb within 0.2 u on 95%). Identification INCONCLUSIVE: K1, K2 and K4 hold, but H_ZT needs K3's ns form and the constant reads in SM cycles.

**Part 2, the launch-floor rerun** (`results/2026-10-02-nvidia_gh200_480gb-launch-floor-r2/<model>/`): tp8 P0 to P5 HELD (P1 27 of 27, 0.877 to 1.015 x C_reg; P5 77 of 77), P6 FALSIFIED (5 of 24, the n = 1 E240 / E480 cells 7 to 9% over MP), P8 FALSIFIED (4 of 7, H - I 75 to 79 us at n = 2, 3). Granite-3B: P0 FAILED (H_cell / H_pre outside [0.93, 1.10] on 8 of 108 cells, all graph replay at n = 1 to 3), P1 and P6 HELD, P2, P4, P5, P8 HELD marked host drift. JetMoE: P1 FALSIFIED as the scorer counts it (see below), P2, P4, P5, P6 HELD, P8 FALSIFIED (n = 1 NATIVE and SHARED, H - I 76 us). Granite-1B: P0 FAILED (15 of 54, all graph replay), P1 FALSIFIED (SHARED n = 1, 2 and PRIVATE n = 1 at 0.764 to 0.767 x C_reg; SHARED's n = 1 to 5 increment 0.089 ms against 0.10), P6 HELD, P8 FALSIFIED (host drift). P3, pooled: FALSIFIED (10 of 15 outside; E0 - E240 33 to 61 us against bands of 50 to 75). The timed process never had the profiler on, and its post/pre host drift is 1.1 to 3.2%.

**Part 1, B's knobs and C's T5** (`results/*-mixtral-8x7b-*-r3-counters/lock1710/r3c-g{1,64}.json`): tp2 w2 s8 (primary) INCONCLUSIVE, L2r, NL and ST each FALSIFIED: s 0.86 0.76 0.75 0.76 0.76 0.76 at n = 4..9 against the base page's 0.58 to 0.53, above every band; the knob raised survival where L2r predicts a fall. tp4 w1 s8 (primary) INCONCLUSIVE, LAG and NL FALSIFIED (0.56 to 0.42, above NL at 5 of 6). tp2 w2 s6: NL SELECTED (0.58 to 0.52, the base page's level). 8x7B w2 s8: NOT SCORED (its knob page fails V7, so CLEAN has no data); ALL, reading only gate-failed pages: HELD. H2c: l2bk128 SELECTED on both GEMMs (w2 at the base page's W_c +0.13 to +0.27), l2bk32 w1 FALSIFIED and w2 SELECTED as the scorer prints them, on a page that fails V6 (see below). H3: FALSIFIED (no address effect: PRIVATE bytes within 0.02%). Controls: tp8 w2 s8 >= 0.95 HELD; tp2 w1 under its L2r edge on s8, over it on s6 (demotes only an L2r selection). Partition cross-check DISAGREE on both pages (81% and 75% of cells outside 5%; the direct-hit ratio reads 0.14 to 0.94): far share from the direct count only. T5: NOT SCORED (the BK32 page fails V7); ALL, reading only gate-failed pages: R0, R1, R2 FALSIFIED, R2h INCONCLUSIVE (board check failed): rho_BK 0.357 (f 0.206 at BLOCK_K 128, 0.577 at 32), every band at 0.97 or above; f(BK64, n = 9) 0.480, -17% from rental 1's 0.5802.

**Two scorer readings that need the owner** (neither scorer was changed; the registration's text and the scorer's output are both in SCORES.md): (1) part 1's registered V6 rule says a page failing V6 has its cells excluded and its verdicts read INCONCLUSIVE; the H2c branch of `score_knobs.py` does not apply it to tp2 l2bk32. (2) part 2's P1 counts a missing graph-replay cell as outside; JetMoE's plan unit ran n = 1..4, so the n = 5 cells P1 asks for (Granite's n <= 5 rule) do not exist and give all 3 of its "outside" cells; its 12 measured cells are all inside (0.862 to 0.987 x C_reg).

**BLOCK_K 128 (unit 14), analyse exit 1.** Every VALIDITY gate passes (V10 fits 1707.3 / 1704.0 MHz). The exit is the CLAIM code (`moe/bench/exit_codes.py`: 1 = CLAIM_FAIL): C1 REFUSE (SHARED's w1 weight-only bracket edges disagree at n = 4..9, where PRIVATE's excess, the activation re-read, is 0.08 to 0.73 and the counter cannot split weight from activation bytes), C2 FAIL and C6 FAIL (PRIVATE re-reads up to 11% over n on w2). C1 REFUSE and C6 FAIL read the same on every G = 64 tp2 page here and on rental 1's (C2 FAIL too, but REFUSE on ats8); on the other three pages here the exit 3 from V7 or V10 hides them. This is the designed outcome at G >= E n, not a refusal of the page: claim gates gate nothing (the addendum, rule 4), and T5 reads only PRIVATE w2's q.

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and consoles
- `results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-{tp8-floor2,tp4-floor,tp2-floor,tp4-floor1005}-r3-counters/`: the floor captures (units 2 to 5), base and lock, shape metrics and null kernel
- `results/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-*-r3-counters/lock1710/`: byte pages, one directory per unit (6 to 14, 17 to 22), each with `summary.json`
- `results/2026-10-02-nvidia_gh200_480gb-launch-floor-r2/<model>/`: cells.csv, hostprobe.csv, hostprobe-post.csv, manifest.json (timed process), manifest-trace.json and traces/ (trace process) (15, 16, 23, 24)
- `census-*.json`, `census-*.profiles/`: each model's census, written before its first page
- `recovery/retake_auto.sh`: the retake step as it ran (nothing to retake)

The `card:` lines and every page's CARD block carry the card's UUID, as every
published session does; prose here names the board by `r3_timing_model.board()`.
The Triton caches' compiled `cuda_utils*.so` files are left out by `.gitignore`, as in
rental 1.
