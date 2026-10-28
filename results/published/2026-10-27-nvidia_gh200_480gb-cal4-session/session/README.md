# Calibration rental C4: one NVIDIA GH200 480GB, 8 units of one calibration registration (docs/registered/README.md, 2026-10-27)

One rental ran `scripts/plans/rentalC4-cal-2026-10.plan` unattended through the calibration rental's driver
(`scripts/gh200_cal4_session.sh`, C2's pinned driver with its own header and no other line changed). It set the GPU
power limit to 700 W and read it back, probed the 1410 MHz session lock, ran calibrate's ruler and a compile-only
census of Mixtral-8x7B-tp2 at 128x128x64 w8 s3, measured the sag estimator's two f128 pages on this board and gated
them with A2's G-BAND, then ran Mixtral-8x7B-tp2's 128x128 w8 s3 G 1 timed page (C4-T) and byte page (C4-B), the
fifth held-out CAL model, in blind Phi-3.5's w1 regime (K 4096, x near 1, P 112), under the round-4 model frozen at
the registration. The registration (`2026-10-27-cal4-gh200`), its reader, the tp2 histogram page, the driver and the
plan were committed before any page (aa5e1b7c, which is also the run commit).

C4 is a CALIBRATION rental: it certifies nothing and scores no verdict, and every C4 page is SEEN once read. Every
reading below is the registered reader's (`scripts/scoring/cal4/read_c4.py`, pinned by the registration), run on
this directory with no post-page fix; its output is `scripts/scoring/cal4/cal4.read.{json,txt}`, and the notes are
in `docs/registered/README.md` ("**Read 2026-10-27**" on this directory) and `docs/FINDINGS.md` (calibration rental
C4).

## Where it came from

Lambda Cloud instance e86d88b5 (one GH200, us-east-3), launched 22:38:07Z and terminated 23:04:21Z on 2026-10-27:
about 26 min, about $1.00 at the registration's $2.29/h (an estimate; the instance, the region and the two times
are the rental's records, not in the run files, and the bill is in neither). The VM's setup ran 22:39:55Z to
22:42:05Z and the driver 22:42:07Z to 23:02:25Z. Board `183300` (`PREFLIGHT.txt`), which no earlier published
session names. `scripts/gh200_cal4_session.sh --plan scripts/plans/rentalC4-cal-2026-10.plan` ran at commit
aa5e1b7c (`commit.txt`); the plan's sha256 ab5a7aed... is in `gh200-driver/status`. This directory is the branch
`run-gh200-tile4-20261027t2238`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and push log.
Driver 580.105.08, persistence on, power.max_limit 900 W read before the limit was set.

## The power limit and the lock

The registration asks for 700 W with no fallback and stops the session unless two read-backs agree:
`--query-gpu=enforced.power.limit` and the GPU scope's Current Power Limit in `nvidia-smi -q -d POWER`. Before the
limit was set (`power-at-start.txt`) the GPU scope read Current 700 W. After `-pl 700 -sc 0` the first poll (0 s)
read enforced.power.limit 700 W, the -q GPU scope Current 700 W (Requested 700 W) and power.limit 700 W
(`power-set.txt`, `gh200-driver/c2-power.env` `C2_POWER_W=700`); the module scope read 1000 W. No
`power-set-unsettled-700.txt` was written. Every unit ran at 700 W: the driver's `clocks:` line after every unit
reads power.limit 700 W, and the 6 power.limit records on the 3 counter pages read 700.00 W.

The prelude locked 1410 MHz, read clocks.sm back, reset, and recorded `LOCK_PROBE_1410=held`
(`gh200-driver/locks.env`). G-BAND (`gh200-driver/gate-band-1410.json`) then PASSED on the registered fallback
band [1352.19, 1410] MHz: in the primary band [1374.75, 1410] the joint fit was None (0 of 27 OLMoE cells and 5 of
27 Mixtral-8x7B cells in band); in the fallback 22 of 27 and 17 of 27. A2 and C2 also passed on the fallback.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | registered as | start to end (UTC) | min | exit | page exit |
|---|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W read back; lock 1410 held | C4-P | 22:42:07 to 22:42:09 | 0 | 0 | |
| 2 | calibrate | mixtral-8x7b tc4 (the ruler, no lock) | C4-P | 22:42:11 to 22:42:36 | 0 | 0 | triad 3725.4 GB/s |
| 3 | tilecensus | tc4: 128x128x64 w8 s3 on mixtral-8x7b-tp2, compile only | C4-P | 22:42:38 to 22:42:47 | 0 | 0 | 2 configs, 0 path mismatches |
| 4 | bytes | olmoe-1b-7b f128 (G 64, treads 1 to 9) | C4-F | 22:42:48 to 22:46:00 | 3 | 0 | 3 |
| 5 | bytes | mixtral-8x7b f128 (G 64, treads 1 to 9) | C4-F | 22:46:02 to 22:52:58 | 7 | 0 | 3 |
| 6 | gate | gband at 1410 | C4-F | 22:53:00 to 22:53:02 | 0 | 0 | PASS (fallback band) |
| 7 | timed | mixtral-8x7b-tp2 t128g1 (G 1, n 2, six labels, NATIVE and SHARED) | C4-T | 22:53:04 to 23:00:12 | 7 | 0 | R3 0 |
| 8 | bytes | mixtral-8x7b-tp2 b128g1 (G 1, the same cells) | C4-B | 23:00:14 to 23:02:22 | 2 | 0 | 0 |

8 of 8 units exit 0 in about 20 min of units against 25 estimated; none was dropped; the driver ends with exit 0. A
counter page exit 3 is the route's own page gates (the two G 64 f128 pages: OLMoE V4, V5, C1 and C6 FAIL, C2
REFUSE; Mixtral-8x7B V5, V7, C2 and C6 FAIL, C1 REFUSE), which C4 does not read: no C4 reading uses a page verdict.
Each byte unit's `analyse` step exits 3 as well (`summary.json`; on tp2's page, one tread and two arms). The tp2
timed page stores G1 lock-thermal PASS with 0 slipped repeats in 108 repeat rows (worst cell 1410.0 MHz) and is
admitted by the registered `tcommon.timed_page`.

## What the reader reads (`scripts/scoring/cal4/cal4.read.txt`)

| reading | outcome | numbers |
|---|---|---|
| board | valid | C2_POWER_W=700, LOCK_PROBE_1410=held, G-BAND PASS (fallback); the commit check passed on 74 files at aa5e1b7c; the fingerprint reproduces; the PREFLIGHT reproduces round 4's table and P(A) 0.3798 before any page |
| F1 on C4's board | par 54.25 us, tau 587.5 us | per page: OLMoE 40.95 us (525 us), Mixtral-8x7B 67.55 us (650 us); A2's par 42.02 us (425 us), C2's 41.59 us (412.5 us) |
| C4-R1 | X = -4.44%, LOW-FAR | six NATIVE n 2 cells kept and priced, none dropped: PT -5.69, PW -4.89, PT2 -5.70, PW2 -5.24, uniform -2.59, balanced -2.56; beside: X at A2's par -4.68, with M4's bytes -1.84, with the counted bytes (oracle) +0.12, at C4's triad -4.44, v5'' -1.61 |
| the table | spread 5.69 | OLMoE +0.48, Qwen2-57B -1.44, Mixtral-8x7B +1.25 (+1.49 on C2's board), Mixtral-8x22B +0.24 (round 4's), tp2 -4.44; max abs mean 4.44, rms of means 2.17 |
| pass estimates | | A (5 models) 0.0206, B (5 models) 0.0359, D (O, Q, 8x22B, X) 0.0118 |
| C4-R2 | Lambda 1.5505, BEYOND-M4 | W2 lever 0.1466; w1 reads M4 -7.47%, W2 -21.04% against the counters (6 cells); w2 beside: M4 -0.21%, W2 -5.23% |
| joint | LOW-FAR x BEYOND-M4 | neither byte law fits at P 112: Phi's stake at risk |
| C4-S1 | x per GEMM on the timed cells | w1 median 0.974 (M4), 0.833 (W2); w2 0.624 (M4), 0.613 (W2) |
| C4-S2 | K1 on counted bytes | wall: w1 -0.25%, w2 -2.53%, cell sum -0.98%; cycles: w1 +1.99%, w2 -2.88% (6 cells each) |
| C4-S3 | tp2 sag | own fit 53.82 us, tau 450 us; C4's par minus the measured off: mean -4.69 us, rms 5.19 us over 24 GEMMs |
| C4-S5 | census COMPLETE; triad just below | 2 CTAs per SM compiled and priced on w1 and w2; triad 3725.41 GB/s against A2's 3726.2 and the 15-rental range 3725.6 to 3727.6 |

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and the consoles;
  `gh200-driver/c2-power.env`: the power outcome; `gh200-driver/locks.env`: the lock probe;
  `gh200-driver/gate-band-1410.json`: G-BAND
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the -q -d POWER snapshots before `-pl`,
  before calibrate and after the 700 W read-back
- `results/2026-10-27-nvidia_gh200_480gb-<model>-<label>-r3-counters/lock1410/`: the three counter pages (C4-F f128
  on OLMoE and Mixtral-8x7B, C4-B b128g1 on Mixtral-8x7B-tp2), each `r3c-g<G>.json` with `summary.json`, profiles
  and Triton caches
- `results/gaps-nvidia_gh200_480gb-mixtral-8x7b-tp2-t128g1/`: the timed page (`report.json`, `report.txt`,
  `cells.csv`)
- `results/2026-10-27-nvidia_gh200_480gb-tile-census-mixtral-8x7b-tc4/`: the tile census; `results/calibration/`
  and `measured_nvidia_gh200_480gb.yaml`: calibrate's ruler
- `census*.json`, `census*.profiles/`: the per-model GEMM censuses written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console; `logs/` holds
  every per-unit log, named by model and label

**Histogram.** The tp2 pages read `docs/registered/2026-10-27-cal4-hist/mixtral-8x7b-tp2-128g1.json`, with its
sha256 on the page. No trace-derived count is on the VM or in this directory.

The `card:` lines and every page's CARD block carry the card's UUID, as every published session does (the files are
the run branch's, byte for byte); prose here names the board by its hash. The Triton caches' compiled
`cuda_utils*.so` files (7) are left out, as in every earlier session. Every other file of the run branch is here,
byte for byte (231 files against SHA256SUMS, which lists 238).
