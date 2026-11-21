# Rental 3: one NVIDIA GH200 480GB, 24 units of six registrations (docs/registered/README.md, 2026-10-05)

One rental ran `scripts/plans/rental3-2026-10.plan` unattended: the end-to-end blind time test on
qwen2-57b-a14b-tp8 (part E: floor at base, 1710 and 1005 MHz, calibrate, timed G = 8, 32, 3, byte
pages G = 3, 8, 32), the floor-law captures of part B and RK (Granite-1B, tp4 at four clocks with a
same-clock replicate, the G = 64 wall-time check, tp8 and tp2 at 1005), the stage / BLOCK_K pages of
part C, the rep8 timed replicate of part R, then the tail (deep G = 4, 2; the tp8 flush ladder of
part D). The six registrations, their scorers and the plan were committed at 11244a6 before any
page; the run commit is 49f2198. Every verdict below is the scorers' (`scripts/scoring/rental3/`,
written and tested on synthetic pages before any page), run on this directory with one post-page
fix to reading code (E2, below); the full tables are in `scripts/scoring/rental3/SCORES.md`, the
scoring notes in `docs/registered/README.md` ("**Scored 2026-10-06**") and `docs/FINDINGS.md`
(rental 3).

## Where it came from

Lambda Cloud instance 7d3bb595, gpu_1x_gh200, launched 2026-10-06 06:55:50 UTC, terminated
09:20:25 UTC once every pushed file had been verified (942 files match SHA256SUMS): 2 h 25 min,
about $5.53 at $2.29/h. Board `bb7a34` (`PREFLIGHT.txt`, a board no earlier session used).
`scripts/gh200_model_session.sh --plan scripts/plans/rental3-2026-10.plan` ran at commit 49f2198
(`commit.txt`); the plan's sha256 3de1bab3... is in `gh200-driver/status`. This directory is the
branch `run-gh200-rental3-20261006t0655`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log. Power limit 700 W, persistence on (`power-set.txt`). The 1710 lock was
held for every byte page, every timed page and both launch-floor units (`lock-plan.txt`,
`clocks-after-*.txt`; every timed page's worst cell clock 1710.0 MHz).

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, design | start to end (UTC) | min | exit | pages (gates not PASS) |
|---|---|---|---|---:|---:|---|
| 1 | prelude | clocks, persistence, 700 W | 06:58:51 to 06:58:55 | 0 | 0 | |
| 2 | floor | qwen2-57b-a14b-tp8, G = 64, n = 2..6, 8, 10, 12..14, 16, base + 1710 | 06:58:57 to 07:01:20 | 2 | 0 | none |
| 3 | floor | qwen2-57b-a14b-tp8 at 1005 MHz | 07:01:23 to 07:02:46 | 1 | 0 | none |
| 4 | calibrate | the ruler (triad 3725.9 GB/s) | 07:02:49 to 07:03:13 | 0 | 0 | |
| 5 | timed | qwen2-57b-a14b-tp8 e2e: G = 8, 32 (treads 6), G = 3 (treads 8) | 07:03:15 to 07:38:59 | 36 | 0 | G = 32 V5 UNKNOWN (INVALID); G = 8, 3 claim gates only |
| 6 | bytes | qwen2-57b-a14b-tp8 e2e, G = 3, 8, 32 | 07:39:02 to 07:44:28 | 5 | 0 | V5 FAIL on all three (page exit 3) |
| 7 | floor | granite-3.0-1b-a400m, G = 64, n = 3..6, 8, 10, 12, 14, 16, base + 1710 | 07:44:31 to 07:46:00 | 1 | 0 | none |
| 8 | floor | granite-3.0-1b-a400m at 1005 | 07:46:03 to 07:46:53 | 1 | 0 | none |
| 9 | floor | mixtral-8x7b-tp4, n = 2, 4, 6, 8, 10..16, base + 1710 | 07:46:56 to 07:49:32 | 3 | 0 | none |
| 10 | floor | tp4 floorrep, 1710 only (the K1 / K4 replicate) | 07:49:35 to 07:50:50 | 1 | 0 | none |
| 11 | floor | tp4 at 1005 | 07:50:53 to 07:52:25 | 2 | 0 | none |
| 12 | floor | tp4 at 1410 | 07:52:29 to 07:53:46 | 1 | 0 | none |
| 13 | launchfloor | tp4 wall check: NATIVE, G = 64, E240 and GR, no profiler | 07:53:50 to 07:59:01 | 5 | 3 | 66 of 66 timed cells ok; 1 of 44 host probes refused |
| 14 | floor | tp8 at 1005, n = 6, 8, 10..16 | 07:59:04 to 08:00:21 | 1 | 0 | none |
| 15 | floor | tp2 at 1005 | 08:00:25 to 08:02:54 | 2 | 0 | none |
| 16 | bytes | tp2 l2base, G = 1 | 08:02:58 to 08:06:35 | 4 | 0 | none |
| 17 | bytes | tp2 k32s8 (BLOCK_K 32, 8 stages) | 08:06:38 to 08:10:15 | 4 | 0 | none |
| 18 | bytes | tp2 k128s4 | 08:10:19 to 08:13:55 | 4 | 0 | none |
| 19 | bytes | tp2 k128s3 | 08:13:59 to 08:17:35 | 4 | 0 | none |
| 20 | bytes | tp2 k64s7 | 08:17:39 to 08:21:16 | 4 | 0 | none |
| 21 | bytes | tp2 k64s3 (control) | 08:21:20 to 08:24:57 | 4 | 0 | none |
| 22 | timed | qwen2-57b-a14b-tp8 rep8: G = 8, treads 6 | 08:25:01 to 08:35:49 | 11 | 0 | claim gates only |
| 23 | deep | qwen2-57b-a14b-tp8 e2e: G = 4 then 2, treads 9 | 08:35:53 to 09:08:05 | 32 | 0 | claim gates only |
| 24 | launchfloor | mixtral-8x7b-tp8 flush ladder: n = 1..3, E0, E240, E360, E480, traces at 1 | 09:08:09 to 09:15:51 | 8 | 0 | |

137 minutes from the prelude to the last unit against the plan's 182-minute estimate; no unit was
dropped. 23 units exit 0. The driver ends with exit 3 because of unit 13. A byte unit's exit is
the lock block's; each page's own exit is in the ledger (3 on the three e2e byte pages, V5).

**Unit 13's exit 3.** All 66 timed cells (11 treads x E240, GR x 3 repeats) are `ok` in
`cells.csv`. The one failure the driver counts (`66 cells, 0 traces, 1 failed`) is a host probe:
`hostprobe.csv`, pre phase, GR NATIVE n = 13, `refused`: "call: an iteration took 11.7 x the median:
something in the body synchronised". The wall-time test B6 reads the timed GR cells only, so it
has all 11 treads; the probe's H_pre is not read by part B.

**The G = 32 timed page.** INVALID on V5 alone: V5 reads UNKNOWN (b = -0.004283 ms per M-tile,
3.09% of the private slope; its 90% band 2.42 to 3.64% straddles the 3% bound). The registration
(`docs/registered/2026-10-05-rental3-e2e-gh200.txt` l.200) excludes V5 from gating CLEAN, so the page
counts in both views.

**The recovery step: an operations defect, no data lost.** After the last unit the driver's
recovery asked for a G = 32 timed retake and passed `--model rental3`, the plan's name, to
`private_weight_reference.py`, which needs a model; R3 exited 2 REFUSED 10 s later with no cell
measured (`locked-r3/gh200-20261006T065851Z-t32b/`, `logs/locked_r3-gh200-20261006T065851Z-t32b.log`).
Nothing was measured and no page is affected; the V5 page counts as registered.

**The e2e byte pages' V5.** PRIVATE w1 reads q_P 0.957 n at every n (0.9566 to 0.9574 at n = 1),
under V5's 0.97 n floor: 4.3% of w1's weight bytes are not counted. w2 reads q within 0.2% of n.
Every byte V-gate gates CLEAN for E3, E3s and E6, so those three read NOT SCORED with ALL printed.

## What it answered

Each verdict is the registered reading (rental 2's rule 3 over ALL and CLEAN).

**Part E, end-to-end blind time** (`results/*-qwen2-57b-a14b-tp8-*`, `results/gaps-*-qwen2-57b-a14b-tp8-e2e/`):
E1, time from predicted bytes, FALSIFIED: rms 3.50% over the 31 core cells, worst -6.08% (PRIVATE
G = 3 n = 2); every cell is measured 9.7 to 29.3 us under the prediction; resolution 0.31%, so the
verdict is resolved. E2, time from counted bytes, FALSIFIED: rms 2.52%, worst +4.72%, none beyond
5% (post-page fix: the scorer had dropped the G = 32 page; `SCORES.md`). E3 and E3s NOT SCORED
(ALL: FALSIFIED; PRIVATE w1 -4.4%, SHARED w1 -22% median). E4 M HOLDS: w2's per-CTA unit at S 5
reads b 0.984 on the 1005 DUR, against F0's 0.637. E6 NOT SCORED (ALL: INCONCLUSIVE, 29.4 us
between DEAD's 29.6 to 44.4 and FIXED's 17.2 to 25.8). Printed: M+Z rms 1.89%; deep pages 2.83%.

**Part R and RK**: sigma_page 0.18% (rep8 against the e2e G = 8 page); K1 and K4 UNRESOLVED at the
floorrep noise (w1 sigma_L 3,888 cycles against K1's 5,000 threshold), on rental 3's tp4 pool and on
rental 2's; rental 2's HOLDS stand as registered.

**Part B, the floor law**: INCONCLUSIVE. CEIL and FLUID FALSIFIED, CLOCK neither held nor
falsified. B1 HOLDS (14 of 16 lock captures), B2 holds on 7 of 8 (qwen2-tp8 w2 reads 0.06 at
1005), B0 FALSIFIED on its one countable GEMM (tp4 w2 +0.21), B6 INCONCLUSIVE (theta_wall 0.58,
se 0.14), B7 H_EST HOLDS (theta 0.964).

**Part A, the constant's form**: INCONCLUSIVE. PROP FALSIFIED (A1 on both Granite-1B GEMMs); AFF
fails A2 on Granite-1B w2 only (r 0.842 +- 0.063); MIX fails nowhere; qwen2-tp8 w2 NOT SCORED
(curvature).

**Part C, what lifts G = 1 survival**: INCONCLUSIVE, survivors OCC and NULL. Every classified page
reads base: DEPTH, WIDTH, U-OCC and U-DEPTH FALSIFIED. No page recorded 2 or fewer CTAs per SM
(5, 3, 4, 3, 4 against the expected 4, 2, 3, 2, 4), so OCC was never put to a test page.

**Part D, the flush ladder**: OVERLAP SELECTED, SAT FALSIFIED, on the 3 qualifying n = 1 cells
(steps +27 to +42 us against the flush's 33.8 us, slopes 0.92 to 1.03).

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and consoles
- `results/2026-10-06-nvidia_gh200_480gb-<model>-<label>-r3-counters/`: floor captures
  (`r3f-g64[-lock<F>].json`, units 2, 3, 7 to 12, 14, 15) and byte pages (`lock1710/r3c-g<G>.json`,
  units 6, 16 to 21), each byte directory with `summary.json`
- `results/gaps-nvidia_gh200_480gb-qwen2-57b-a14b-tp8-{e2e,rep8}/private_weight_reference/`: the
  timed pages (units 5, 22, 23); `locked-r3/` their lock ledgers
- `results/2026-10-06-nvidia_gh200_480gb-launch-floor-{wall,r3}/<model>/`: cells.csv,
  hostprobe.csv, hostprobe-post.csv, manifest.json, and for the flush ladder manifest-trace.json
  and traces/ (units 13, 24)
- `census-*.json`, `census-*.profiles/`: each model's census, written before its first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console

The `card:` lines and every page's CARD block carry the card's UUID, as every published session
does; prose here names the board by `r3_timing_model.board()`. The Triton caches' compiled
`cuda_utils*.so` files (34) are left out, as in rentals 1 and 2; every other file of the run
branch is here, byte for byte (908 files against SHA256SUMS).
