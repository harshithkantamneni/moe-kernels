# Tile rental A2: one NVIDIA GH200 480GB, 32 units of eight registrations (docs/registered/README.md, 2026-10-20)

One rental ran `scripts/plans/rentalA2-tile-2026-10.plan` unattended through tile rental A2's driver
(`scripts/gh200_tileA2_session.sh`, the pinned tile driver plus lines marked A2). It set a 900 W GPU power
limit, ran the lock-hold probe down the ladder 1710, 1410, 1305 MHz until a rung held, and then ran
rental A's pages again at that lock: the compile-only tile census, the G = 64 counter pages that
calibrate the wgmma floor (128x128x64 w8 s3 on OLMoE and Mixtral 8x7B) and test it (64x128x64 w8 s3 on
both; w4 and the 32x128x64 w4 mma.sync tile on OLMoE), the four byte pages for the A key, the five
E1-CAL timed pages, OLMoE's 128x128 replicate and the dead-CTA pair at 9 and 15 declared copies, plus
the CONV-L anchor pages (R3's tile, at the selected lock and at 1710). The registrations, their
scorers, the synthetic histograms and the plan were committed before any page (36291487, which is also
the run commit). Every verdict below is the scorers' (`scripts/scoring/tileA2/`, written and tested on
synthetic pages before any page), run on this directory with no post-page fix. The tables are in
`scripts/scoring/tileA2/SCORES.md`, and the scoring notes in `docs/registered/README.md` ("**Scored
2026-10-21**") and `docs/FINDINGS.md` (tile rental A2).

## Where it came from

One Lambda GH200 (`gpu_1x_gh200`), 2026-10-20 to 2026-10-21: the VM was prepared at 22:44:13Z and the
driver ended at 2026-10-21 01:29:36Z. The instance id and the bill are not in the run files and are not
given here. Board `fb8566` (`PREFLIGHT.txt`), which no earlier published session names.
`scripts/gh200_tileA2_session.sh --plan scripts/plans/rentalA2-tile-2026-10.plan` ran at commit
36291487 (`commit.txt`); the plan's sha256 0ab6c60c... is in `gh200-driver/status`. This directory is
the branch `run-gh200-tile2-20261020t2242`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log. Driver 580.105.08, persistence on, power.max_limit 900 W read before the
limit was set (`gh200-driver/status`). The ruler (unit 2) reads triad 3726.2 GB/s; calibrate's settled
GEMM clock 1455 MHz passed G-CAL against the lowest rung, 1305 MHz.

The power limit has two readings that disagree. `nvidia-smi --query-gpu=power.limit` reads 900 W
after `-pl 900 -sc 0` (`clocks-at-start.txt`, `clocks-after-timed.txt`, the prelude's own check,
the 44 power.limit records in the 11 counter pages' `r3c-g*.json` and `*.capture.json`, and every row
of the 17 `locked-r3/*/smi.csv` ledgers). `nvidia-smi -q -d POWER` reads the GPU scope's Current and Requested Power Limit 700 W every
time it was read: `power-at-start.txt` (before `-pl`), `power-set.txt` (just after it),
`power-before-calibrate.txt`, the four `power-lockprobe-*` files and all 1,612 per-second samples in
`logs/lockprobe-{1710,1410}-power.log` (22:47Z to 23:14Z). The module scope reads 1000 W throughout.
Which limit the board enforced is not established by these files (see "The clock").

## What ran (`gh200-driver/status`, `gh200-driver/timed-results`)

| unit | step | model, label | start to end (UTC) | min | exit | page result |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 900 W, persistence, rungs 1710, 1410, 1305 probed | 22:45:56 to 22:46:02 | 0 | 0 | each rung held its read-back |
| 2 | calibrate | the ruler (triad 3726.2 GB/s) | 22:46:04 to 22:46:29 | 0 | 0 | |
| 3 | gate | G-CAL | 22:46:31 to 22:46:32 | 0 | 0 | PASS (GEMM clock 1455 MHz) |
| 4 | tilecensus | 6 tiles x 7 models, compile only | 22:46:34 to 22:46:55 | 0 | 0 | |
| 5 | gate | G-CEN | 22:46:56 to 22:46:58 | 0 | 0 | PASS (EQUAL over 72, no path mismatch) |
| 6 | lockprobe | ladder 1710, 1410, 1305 | 22:46:59 to 23:14:26 | 27 | 0 | 1710 FAIL (Mixtral page; early stop), 1410 PASS: lock 1410 |
| 7 | bytes | olmoe-1b-7b f128 (G 64) | 23:14:28 to 23:17:25 | 3 | 0 | page exit 3 (V4, V5, C1, C6 FAIL) |
| 8 | bytes | mixtral-8x7b f128 (G 64) | 23:17:27 to 23:24:26 | 7 | 0 | page exit 3 (V5, V7, C2, C6 FAIL) |
| 9 | gate | G-BAND | 23:24:28 to 23:24:29 | 0 | 0 | PASS (the fallback band) |
| 10 | bytes | olmoe-1b-7b f64w8 (G 64) | 23:24:32 to 23:27:18 | 3 | 0 | page exit 1 (C1 only) |
| 11 | bytes | mixtral-8x7b f64w8 (G 64) | 23:27:20 to 23:34:01 | 7 | 0 | page exit 3 (V7, C2, C6 FAIL) |
| 12 | gate | G-64 | 23:34:04 to 23:34:05 | 0 | 0 | PASS |
| 13 | timed | mixtral-8x7b cvm1 (CONV-L, G 1, at 1410) | 23:34:08 to 23:36:46 | 3 | 0 | R3 exit 0 |
| 14 | timed | olmoe-1b-7b cvo1 (CONV-L, G 1, at 1410) | 23:36:49 to 23:38:57 | 2 | 0 | R3 exit 0 |
| 15 | timed | mixtral-8x7b cvm8 (CONV-L, G 8, at 1410) | 23:39:00 to 23:44:08 | 5 | 0 | R3 exit 0 |
| 16 | timed | mixtral-8x7b cvm1t (CONV-L, G 1, at 1710) | 23:44:11 to 23:46:49 | 3 | 0 | R3 exit 3, worst cell 1710 MHz |
| 17 | timed | olmoe-1b-7b cvo1t (CONV-L, G 1, at 1710) | 23:46:52 to 23:49:00 | 2 | 0 | R3 exit 0 |
| 18 | timed | mixtral-8x7b cvm8t (CONV-L, G 8, at 1710) | 23:49:04 to 23:53:52 | 5 | 0 | R3 exit 0 |
| 19 | timed | olmoe-1b-7b e128g1 | 23:53:55 to 00:08:14 | 14 | 0 | R3 exit 0 |
| 20 | timed | mixtral-8x7b e128g16 | 00:08:18 to 00:22:16 | 14 | 0 | R3 exit 0 |
| 21 | timed | mixtral-8x7b e64g1 | 00:22:20 to 00:31:48 | 9 | 0 | R3 exit 0 |
| 22 | timed | qwen2-57b-a14b e128g1 | 00:31:52 to 00:41:31 | 10 | 0 | R3 exit 0 |
| 23 | timed | mixtral-8x7b e128g1 | 00:41:35 to 00:46:33 | 5 | 0 | R3 exit 0 |
| 24 | bytes | olmoe-1b-7b b128g1 | 00:46:37 to 00:49:00 | 2 | 0 | page exit 0 (analyse exit 3: V1) |
| 25 | bytes | mixtral-8x7b b64g1 | 00:49:03 to 00:52:59 | 4 | 0 | page exit 0 (analyse exit 3: V1) |
| 26 | bytes | mixtral-8x7b b128g16 | 00:53:03 to 00:58:56 | 6 | 0 | page exit 0 (analyse exit 3: V1) |
| 27 | bytes | qwen2-57b-a14b b128g1 | 00:59:00 to 01:04:04 | 5 | 0 | page exit 0 (analyse exit 3: V1) |
| 28 | timed | olmoe-1b-7b e128g1r (replicate) | 01:04:08 to 01:18:27 | 14 | 0 | R3 exit 0 |
| 29 | bytes | olmoe-1b-7b f64w4 (G 64) | 01:18:32 to 01:21:19 | 3 | 0 | page exit 1 (C1, C6) |
| 30 | bytes | olmoe-1b-7b f32w4 (G 64, mma.sync) | 01:21:23 to 01:24:04 | 3 | 0 | page exit 1 (C1) |
| 31 | timed | olmoe-1b-7b d9 | 01:24:08 to 01:26:46 | 3 | 0 | R3 exit 0 |
| 32 | timed | olmoe-1b-7b d15 | 01:26:50 to 01:29:28 | 3 | 0 | R3 exit 0 |

164 minutes from the prelude to the last unit against the plan's 168 minutes of estimates. All 32 units
exited 0 and none was dropped (the registration's wall estimate at 1410 expected the dead-CTA pair to
be dropped for time; it ran). The driver ends with exit 0. Unit 6 ran, at 1710, the Mixtral
probe page and stopped the rung there (the registered early stop: the Qwen2-57B probe page and the OLMoE
counter page were not run at 1710); at 1410 it ran all three. The unit START lines of the bytes and
timed steps say "at the 1710 lock" and "at 1710": `step_what` prints `LOCK_TIMED` before the unit
reads `a2-lock.env` (`scripts/gh200_tileA2_session.sh` lines 459 to 466 and 664), so the ledger text is
stale; every page after unit 6 records lock 1410 except the three CONV-L pages whose plan line sets lock-mhz=1710 (units 16 to 18) and
the 1710 probe page. The V1 failures on the G-1 and G-16 byte pages are R3's attribution gate read on
a histogram page (two arms, three treads), as on rental A's; no tile scorer reads it.

## What it answered

Each verdict is the registered reading of the scorer named.

**session** (`session.score.txt`): **NOT READ: the two power readings disagree (900.0 and 700.0 W)**
(`clocks-at-start.txt` against `power-set.txt`); lock 1410 MHz; the probe's gate records read 1710
FAIL, 1410 PASS. `finalise_a2.py` refuses on it (exit 2), as it would on the STOP below.

**probe** (`probe.score.txt`, DESCRIPTIVE): **1710 MHz: a page FAILS**: the Mixtral 8x7B probe page
(G 16) lost 4 cells (4 or 5 clean repeats of 9) and 33 of 144 repeat rows are non-clean (22.9%, over
10%). **1410 MHz: every page PASSES**: Mixtral 0 of 144 and Qwen2-57B 0 of 72 rows non-clean, every cell
at 1410; the OLMoE f128 counter clause (P5) passes in the fallback band [1352.19, 1410] (24 of 27 cells in
the band, 22 floor-bound; the primary band [1374.75, 1410] holds 2 cells, 4 floor-bound, under the 6
needed; `gh200-driver/gate-probe-1410.json`).
A2's lock is 1410 MHz (`gh200-driver/a2-lock.env`).

**census** (`census.score.txt`): **VALID (COMPLETE)**; **MERGED**: the equality gate reads EQUAL over
rental A's 72 entries, and OLMoE at n 1 and n 3 is SAME on every (tile, GEMM), so deepseek-v2-lite's
tread-3 entries are admitted (84 entries, 0 errors). Occupancy (CTAs per SM, both GEMMs) as on rental A:
128x128x64 w8 s3 2, 64x128x64 w8 s3 3, 64x128x64 w4 s3 3, 32x128x64 w4 s3 4 (design 5), 16x64x128 w4 s4
3, 32x64x128 w4 s3 4.

**floor** (`floor.score.txt`): **CALIBRATION CAL-A** at 1410 MHz in the registered fallback band
[1352.19, 1410] (the primary band left `joint_cF` None): c_128 691.6 +- 9.9 cycles, F_wg,w1 2898 +- 582,
F_wg,w2 3346 +- 323 (34 cells, 6 series, rel rms 0.55%). Accounting: B unreachable, **W HOLDS**, T FAILS;
**SELECTED W**; F form **F-step**. Cross-clock readout (DESCRIPTIVE): -7.0 cycles (-1.01%, z -0.48)
against rental A's out-of-band 698.67 +- 10.63 at 1601 to 1640 MHz.

**bytes** (`bytes.score.txt`): **NOT NAMED (nearest v3); v4 takes CLAMP**. rms over 45 scored w1 rows:
v3 8.70%, ROWKEY60 9.78%, CLAMP 10.20%, ROWKEY30 14.38%; worst nearest (v3) 20.4%. The band dropped 15
cells (OLMoE b128g1 2, Qwen2-57B b128g1 9, Mixtral b128g16 4).

**rep** (`rep.score.txt`): **sigma_page 0.1199%** over 24 cells (mean shift -0.094%).

**e1cal** (`e1cal.score.txt`): **E1-CAL FAILS**: 42 NATIVE cells, rms 4.23%, worst -8.52%, mean -2.31%
(the bar: rms 2%, worst 5%, mean 1%). **STOP: rental B does not run**: class means 64g1 +3.574%, 128g1
-4.955%, 128g16 -1.376% (each beyond 0.75%), the 128 G 1 models' means differ by 2.581% (over 1%), and
the pooled bar FAILS.

**dead** (`dead.score.txt`, SEEN-REPLICATE, DESCRIPTIVE): delta 25.35 us (se 0.62); SLOT 24.69 us NOT
EXCLUDED (z 1.08); D 10.73 (z 23.7), D2 10.76 (23.7) and d 1.06 12.11 (21.5) EXCLUDED, at the measured
c_128; kappa rows NOT SCORED (c_128 under 900). The scorer's comparison line reads "SLOT HOLDS, D
EXCLUDED".

**convl** (`convl.score.txt`): **CONV-L HOLDS**: G1 rms 0.61% (7 cells, mean +0.38%), G8 rms 0.16%
(8 cells, mean -0.13%). Its registered reading (DESCRIPTIVE): CONV-L HOLDS and E1-CAL does not, which
indicates the wgmma calibration or v4's wgmma terms, not the conversion; a carried constant can still
weigh more on 128x128 G 1 cells than on R3's tile, so the conversion is not ruled out.

**decode gate** (`scripts/scoring/tileA2/decode_gate_a2.py scripts/scoring/tileA2`): **GO** (exit 0;
the floor calibration is scored and `finalise.selections` reads the directory), with A2's stop rule
printed beside it (STOP).

## The clock

At 1410 the lock held as a ceiling and almost always as a level: none of the 21,465 per-burst NVML reads
in the 13 timed pages' `cells.csv` is above 1410 MHz and 21,258 (99.0%) read exactly 1410; the lowest is
1335. At 1710 (the probe page and the three CONV-L anchors) 3,223 of 4,185 (77.0%) read 1710, none above.
The counter pages pass lock_gate.check_page (at 1410 that is the nvidia-smi bracket and the null and
spread readings; lock_gate holds a level reference for 1710 only), but their wgmma NATIVE and SHARED
cells again ran under the lock:

| page (G 64 unless named) | NATIVE MHz (median) | SHARED MHz (median) | PRIVATE MHz (median) | in [1352.19, 1410] | in [1374.75, 1410] |
|---|---|---|---|---:|---:|
| olmoe f128 (128x128 w8) | 1344.0 to 1375.4 (1365.8) | 1352.3 to 1373.7 (1365.8) | 1331.8 to 1376.1 (1365.8) | 24 of 27 | 2 |
| mixtral f128 (128x128 w8) | 1329.1 to 1376.1 (1368.5) | 1326.4 to 1376.4 (1365.8) | 1336.3 to 1397.6 (1382.5) | 19 of 27 | 9 |
| olmoe lp1410f (the probe's f128) | 1354.4 to 1376.4 (1363.2) | 1347.3 to 1376.7 (1363.2) | 1335.8 to 1369.3 (1363.2) | 24 of 27 | 2 |
| olmoe f64w8 | 1353.7 to 1373.9 (1368.5) | 1347.9 to 1368.5 (1368.5) | 1367.6 to 1379.5 (1368.5) | 26 of 27 | 3 |
| mixtral f64w8 | 1337.1 to 1372.0 (1361.8) | 1330.9 to 1375.4 (1359.9) | 1357.9 to 1400.7 (1390.4) | 22 of 27 | 8 |
| olmoe f64w4 | 1339.6 to 1364.7 (1364.5) | 1346.0 to 1364.5 (1364.5) | 1360.9 to 1379.5 (1368.0) | 23 of 27 | 3 |
| olmoe f32w4 (mma.sync) | 1373.7 to 1384.1 (1384.1) | 1361.9 to 1384.1 (1384.1) | 1379.9 to 1391.0 (1384.1) | 27 of 27 | 24 |
| olmoe b128g1 (G 1) | 1349.3 to 1358.2 (1354.2) | 1347.2 to 1366.0 (1354.2) | | 22 of 24 | 0 |
| mixtral b64g1 (G 1) | 1352.5 to 1381.2 (1369.4) | 1357.5 to 1373.6 (1368.6) | | 16 of 16 | 1 |
| mixtral b128g16 (G 16) | 1346.7 to 1388.3 (1370.3) | 1349.8 to 1387.4 (1370.1) | | 20 of 24 | 9 |
| qwen2-57b b128g1 (G 1) | 1340.6 to 1368.3 (1350.7) | 1341.7 to 1366.8 (1347.8) | | 7 of 16 | 0 |

Per-cell clocks are `r6common.clock_rule`'s under `a2common.scoring` at 1410, the function every A2
counter scorer applies. The timed pages' lock ledgers (`locked-r3/*/smi.csv`, 13,527 NVML samples on 17
pages) read power.limit 900 W in every row; peak temperature 57 C; peak power.draw 430.5 W. At 1410
(13 pages, 11,331 samples) 226 samples read under 1410 MHz: 93 carry 0x4 (SW power cap) and 133 no
reason flag; at 1710 (4 pages, 2,196 samples) 474 read under 1710: 211 with 0x4, 263 with none. During
the probe the per-second `nvidia-smi -q` samples read the GPU's 1 s average draw at most 633.4 W (1710
rung) and 680.0 W (1410 rung), with instantaneous readings up to 987.5 W and 735.5 W. None of this
says which of the two recorded limits was in force.

## Files

- `gh200-driver/status`, `gh200-driver/timed-results`, `gh200-driver/summary.txt`,
  `gh200-driver/tags-used`, `logs/driver-*.log`: the ledger, the consoles and the session tags
- `gh200-driver/gate-*.json`, `logs/gate-*.log`: the five gates' records; `gh200-driver/a2-lock.env`: the
  lock the probe selected
- `results/2026-10-20-nvidia_gh200_480gb-tile-census-mixtral-8x7b-ta2/tile-census.json`: the census
  (unit 4)
- `results/2026-10-2?-nvidia_gh200_480gb-<model>-<label>-r3-counters/lock1410/`: the eleven counter
  pages (the probe's lp1410f in unit 6; units 7, 8, 10, 11, 24 to 27, 29, 30), each `r3c-g<G>.json` with
  `summary.json`, profiles and Triton caches; the pages started after midnight carry 2026-10-21
- `results/2026-10-20-nvidia_gh200_480gb-rulers-lp{1710,1410}/`: the hardware rulers at each probed rung
- `results/gaps-nvidia_gh200_480gb-<model>-<label>/private_weight_reference/`: the 17 timed histogram
  pages (the three probe pages, the six CONV-L pages, the five E1-CAL pages, the replicate and the
  dead-CTA pair), each with `report.json`, `cells.csv` and `DECLARED_COPIES.txt`; `locked-r3/*/` are
  their lock ledgers
- `power-*.txt`, `logs/lockprobe-*-power.log`, `clocks-*.txt`: the power and clock read-backs above
- `census*.json`, `census*.profiles/`: the per-model GEMM censuses written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console;
  `logs/` holds every per-unit log, named by model and label

**Histograms.** Every histogram page reads one of the registered SYNTHETIC files under
`docs/registered/2026-10-14-tileA-hist/` or `docs/registered/2026-10-20-tileA2-hist/`, with sha256 on
every page. No trace-derived count is on the VM or in this directory.

The `card:` lines and every page's CARD block carry the card's UUID, as every published session does
(the files are the run branch's, byte for byte); prose here names the board by its hash. The Triton
caches' compiled `cuda_utils*.so` files (31) are left out, as in rentals 1 to 7 and tile rental A.
Every other file of the run branch is here, byte for byte (936 files against SHA256SUMS).
