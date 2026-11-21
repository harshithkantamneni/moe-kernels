# Tile rental A: one NVIDIA GH200 480GB, 21 units of six registrations (docs/registered/README.md, 2026-10-14)

One rental ran `scripts/plans/rentalA-tile-2026-10.plan` unattended through the tile stage's driver
(`scripts/gh200_tile_session.sh`). It covered the compile-only tile census, the G = 64 counter pages
that calibrate the wgmma floor (128x128x64 w8 s3 on OLMoE and Mixtral 8x7B) and test it (64x128x64
w8 s3 on both; w4 and the 32x128x64 w4 mma.sync tile on OLMoE), four G-1 / G-16 byte pages for the
A key, the five E1-CAL timed pages, OLMoE's 128x128 replicate, and the dead-CTA pair at 9 and 15
declared copies. The registrations, their scorers, the synthetic histograms and the plan were
committed before any page (c4914ba, which is also the run commit). Every verdict below is the
scorers' (`scripts/scoring/tileA/`, written and tested on synthetic pages before any page), run on
this directory with no post-page fix. The tables are in `scripts/scoring/tileA/SCORES.md`, and the
scoring notes in `docs/registered/README.md` ("**Scored 2026-10-20**") and `docs/FINDINGS.md`
(tile rental A).

## Where it came from

One Lambda GH200 (`gpu_1x_gh200`), 2026-10-20: the VM was prepared at 05:57Z and the driver ended at
07:55:39Z. The instance id and the bill are not in the run files and are not given here. Board
`03e9ec` (`PREFLIGHT.txt`), which no earlier published session names. `scripts/gh200_tile_session.sh
--plan scripts/plans/rentalA-tile-2026-10.plan` ran at commit c4914ba (`commit.txt`); the plan's
sha256 60726c3c... is in `gh200-driver/status`. This directory is the branch
`run-gh200-tile-20261020t0557`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its
manifest and push log. Driver 580.105.08, power limit 700 W (default 900 W), persistence on
(`power-set.txt`). The ruler (unit 2) reads triad 3725.9 GB/s.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | start to end (UTC) | min | exit | page result |
|---|---|---|---|---:|---:|---|
| 1 | prelude | clocks, persistence, 700 W | 06:00:59 | 0 | 0 | |
| 2 | calibrate | the ruler (triad 3725.9 GB/s) | 06:01:01 to 06:01:26 | 0 | 0 | |
| 3 | tilecensus | 6 tiles x 7 models, compile only | 06:01:27 to 06:01:45 | 0 | **3** | 72 configs, 0 MMA path mismatches, 6 errors (deepseek-v2-lite) |
| 4 | bytes | olmoe-1b-7b f128 (G 64) | 06:01:47 to 06:04:51 | 3 | 0 | page exit 3 (V4, V5 FAIL) |
| 5 | bytes | mixtral-8x7b f128 (G 64) | 06:04:53 to 06:11:34 | 7 | 0 | page exit 3 (V5, V7 FAIL) |
| 6 | bytes | olmoe-1b-7b f64w8 (G 64) | 06:11:36 to 06:14:13 | 3 | 0 | page exit 1 (claims only) |
| 7 | bytes | mixtral-8x7b f64w8 (G 64) | 06:14:15 to 06:20:37 | 6 | 0 | page exit 3 (V7 FAIL) |
| 8 | bytes | olmoe-1b-7b b128g1 | 06:20:39 to 06:22:54 | 2 | 0 | page exit 0 |
| 9 | bytes | mixtral-8x7b b64g1 | 06:22:56 to 06:26:41 | 4 | 0 | page exit 0 |
| 10 | bytes | mixtral-8x7b b128g16 | 06:26:43 to 06:32:20 | 6 | 0 | page exit 0 |
| 11 | bytes | qwen2-57b-a14b b128g1 | 06:32:22 to 06:37:13 | 5 | 0 | page exit 0 |
| 12 | timed | olmoe-1b-7b e128g1 | 06:37:15 to 06:51:24 | 14 | 0 | R3 exit 3, worst cell 1695 MHz |
| 13 | timed | mixtral-8x7b e128g16 | 06:51:27 to 07:05:26 | 14 | 0 | R3 exit 3, 56 slipped rows, lowest 1605 MHz |
| 14 | timed | mixtral-8x7b e64g1 | 07:05:29 to 07:15:08 | 10 | 0 | R3 exit 3, worst cell 1710 MHz |
| 15 | timed | qwen2-57b-a14b e128g1 | 07:15:11 to 07:25:19 | 10 | 0 | R3 exit 3, 33 slipped rows, lowest 1515 MHz |
| 16 | timed | mixtral-8x7b e128g1 | 07:25:23 to 07:30:31 | 5 | 0 | R3 exit 3, 4 slipped rows, lowest 1650 MHz |
| 17 | timed | olmoe-1b-7b e128g1r (replicate) | 07:30:34 to 07:44:53 | 14 | 0 | R3 exit 3, 1 slipped row, 1680 MHz |
| 18 | bytes | olmoe-1b-7b f64w4 (G 64) | 07:44:56 to 07:47:35 | 3 | 0 | page exit 1 (claims only) |
| 19 | bytes | olmoe-1b-7b f32w4 (G 64, mma.sync) | 07:47:39 to 07:50:12 | 3 | 0 | page exit 1 (claims only) |
| 20 | timed | olmoe-1b-7b d9 | 07:50:15 to 07:52:53 | 3 | 0 | R3 exit 0 |
| 21 | timed | olmoe-1b-7b d15 | 07:52:57 to 07:55:35 | 3 | 0 | R3 exit 0 |

115 minutes from the prelude to the last unit against the plan's 147 minutes of estimates. No unit
was dropped. The driver ends with exit 3 because the census unit exited 3 (the driver raises its
final exit to 3 on any unit exit 2 or 3); every other unit exited 0. The page results in the last
column are each page's own harness exit or R3's stored verdict; the scorers never read R3's stored
G1 and rebuild each timed page from `cells.csv` under r7-median-run3.

## What it answered

Each verdict is the registered reading of the scorer named.

**census** (`results/2026-10-20-nvidia_gh200_480gb-tile-census-mixtral-8x7b-ta/tile-census.json`):
**VALID (INCOMPLETE)**. No MMA path mismatch over 72 configs. Occupancy (CTAs per SM, both GEMMs):
128x128x64 w8 s3 2, 64x128x64 w8 s3 3, 64x128x64 w4 s3 3, 32x128x64 w4 s3 4 (the design's 5;
register-bound), 16x64x128 w4 s4 3, 32x64x128 w4 s3 4. deepseek-v2-lite raised "N rows per expert is
not an integer token count for E=64 k=6" on all six tiles: a census-tool defect (the tool derives a
token count from rows per expert, which E = 64, k = 6 does not divide), left as recorded.

**floor**: **CALIBRATION NOT SCORED** ("under 6 floor-bound cells or rank deficient"). The scorer
keeps only counter cells whose clock is inside rental 6's band [1640, 1710] MHz. On the two f128
pages that leaves 6 of 27 (OLMoE) and 7 of 27 (Mixtral) cells, every one of them PRIVATE; the
estimator reads NATIVE and SHARED cells only, so the calibration has no cell. No c_128, F_wg or Z_wg;
the 64x128 test, the selections and the w4 and mma.sync prints do not run.

**bytes**: **NOT NAMED (nearest CLAMP); v4 takes CLAMP**. rms over the 19 scored w1 rows that pass
the clock band: CLAMP 5.03%, v3 5.03%, ROWKEY30 12.29%, ROWKEY60 12.31%; worst CLAMP row 7.47%.
The registration scores 56 rows; the band drops OLMoE b128g1's 24 (0 of 24 cells in band) and 13
of Qwen2-57B's 16, so the reading rests on Mixtral b64g1 (16 rows) and Qwen2-57B (3 rows).

**e1cal**: **E1-CAL NOT SCORED**; **STOP: rental B does not run** (tA-floor has no calibration).
`scripts/scoring/tileA/finalise.py` refuses (exit 2, nothing written).

**rep**: **sigma_page 0.1275%** over 24 cells valid on both 128x128 pages (mean shift -0.154%).

**dead**: delta 20.01 us (se 0.61, 2 cells). D (10.02 us), D2 (10.10), SLOT (24.69) and d 1.06
(11.41) are each **EXCLUDED** (|z| 7.7 to 16.3); D against SLOT is **NOT SEPARATED by this page**
(neither holds). The kappa rows are **NOT SCORED** (no c_128). The D-family laws are read at the
nominal c_128 900; at the registered 600 and 1100 they are 12.13 and 7.57 us, also past 3 se.

## The clock

The 1710 MHz lock held as a ceiling: none of the 16,200 per-burst NVML reads on the eight timed
pages' `cells.csv` is above 1710 MHz, and 12,402 (76.6%) read exactly 1710. The counter pages pass
lock_gate.check_page (V10: the lock was in force while ncu counted), but on the wgmma tiles their
NATIVE and SHARED cells ran below rental 6's band:

| page (G 64 unless named) | NATIVE MHz (median) | SHARED MHz (median) | PRIVATE MHz (median) | cells in [1640, 1710] |
|---|---|---|---|---:|
| olmoe f128 (128x128 w8) | 1602.6 to 1636.5 (1626.7) | 1601.1 to 1639.4 (1636.5) | 1617.9 to 1666.4 (1646.8) | 6 of 27 |
| mixtral f128 (128x128 w8) | 1599.3 to 1639.9 (1626.0) | 1606.5 to 1637.2 (1621.8) | 1627.5 to 1694.8 (1677.5) | 7 of 27 |
| olmoe f64w8 | 1616.0 to 1663.3 (1663.3) | 1610.1 to 1663.3 (1663.3) | 1663.3 to 1679.8 (1668.5) | 21 of 27 |
| mixtral f64w8 | 1618.3 to 1654.2 (1637.8) | 1614.3 to 1657.2 (1636.5) | 1654.5 to 1698.9 (1685.2) | 13 of 27 |
| olmoe f64w4 | 1607.9 to 1667.5 (1667.5) | 1608.6 to 1667.5 (1667.5) | 1660.6 to 1678.6 (1667.5) | 21 of 27 |
| olmoe f32w4 (mma.sync) | 1636.6 to 1682.6 (1682.6) | 1636.6 to 1682.6 (1682.6) | 1678.2 to 1689.8 (1682.6) | 25 of 27 |
| olmoe b128g1 (G 1) | 1601.6 to 1620.5 (1614.0) | 1602.8 to 1620.7 (1614.0) | | 0 of 24 |
| mixtral b64g1 (G 1) | 1646.1 to 1673.7 (1664.8) | 1647.7 to 1673.4 (1664.4) | | 16 of 16 |
| mixtral b128g16 (G 16) | 1605.9 to 1649.4 (1630.7) | 1605.2 to 1649.8 (1630.8) | | 8 of 24 |
| qwen2-57b b128g1 (G 1) | 1607.7 to 1643.6 (1633.3) | 1608.1 to 1643.6 (1629.5) | | 3 of 16 |

Per-cell clocks are `r6common.clock_rule`'s, the function every tile scorer applies. The lock
ledgers of the timed pages (`locked-r3/*/smi.csv`, 8,578 NVML samples) set no throttle reason other
than 0x4 (SW power cap, 2,323 samples) and 0x1 (GPU idle, 47); peak temperature 60 C, peak draw
714.5 W against the 700 W limit. 2,072 samples read below 1710 MHz: 938 carry 0x4 and 1,134 carry
no reason flag at all. The counter pages have no smi ledger. That the sag on the counter pages is the
power cap is therefore an inference, not a measurement.

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `gh200-driver/tags-used`, `logs/driver-*.log`:
  the ledger, the consoles and the session tags the driver handed out
- `results/2026-10-20-nvidia_gh200_480gb-tile-census-mixtral-8x7b-ta/tile-census.json`: the census
  (unit 3)
- `results/2026-10-20-nvidia_gh200_480gb-<model>-<label>-r3-counters/lock1710/`: the ten counter
  pages (units 4 to 11, 18, 19), each `r3c-g<G>.json` with `summary.json`, profiles and Triton caches
- `results/gaps-nvidia_gh200_480gb-<model>-<label>/private_weight_reference/`: the eight timed
  histogram pages (units 12 to 17, 20, 21), each with `report.json`, `cells.csv` and
  `DECLARED_COPIES.txt`; `locked-r3/*-p2*/` are their lock ledgers
- `census*.json`, `census*.profiles/`: the per-model GEMM censuses written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console;
  `logs/` holds every per-unit log, named by model and label

**Histograms.** Every histogram page reads one of the registered SYNTHETIC files under
`docs/registered/2026-10-14-tileA-hist/`, with sha256 on every page; the scorers' G5 passed on
every page they counted. No trace-derived count is on the VM or in this directory.

The `card:` lines and every page's CARD block carry the card's UUID, as every published session
does (the files are the run branch's, byte for byte); prose here names the board by its hash. The
Triton caches' compiled `cuda_utils*.so` files (21) are left out, as in rentals 1 to 7. Every other
file of the run branch is here, byte for byte (625 files against SHA256SUMS).
