# Calibration rental C2: one NVIDIA GH200 480GB, 10 units of one calibration registration (docs/registered/README.md, 2026-10-22)

One rental ran `scripts/plans/rentalC2-cal-2026-10.plan` unattended through the calibration rental's driver
(`scripts/gh200_cal2_session.sh`, settle rental S1's pinned driver plus lines marked C2). It set the GPU power
limit to 700 W and read it back, probed the 1410 MHz session lock, measured the averaged-fit sag estimator's two
f128 pages on this board and gated them with A2's G-BAND, then ran Mixtral-8x22B's 128x128 w8 s3 G 1 timed page
(C2-T) and byte page (C2-B), the fourth held-out CAL model of model v5'', and Mixtral-8x7B's 128 G 1 timed page
(C2-MT) and byte page (C2-M) on A2's E1-CAL cells. The registration (`2026-10-22-cal2-gh200`), its reader, the
8x22B histogram page and the plan were committed before any page (2366efcf), with one reader fix before the
rental (09b340ce, which is also the run commit).

C2 is a CALIBRATION rental: it certifies nothing and scores no verdict, and every C2 page is SEEN for v5'' and
B3 once read. Every reading below is the registered reader's (`scripts/scoring/cal2/read_c2.py`, pinned by the
registration), run on this directory with no post-page fix; its output is
`scripts/scoring/cal2/cal2.read.{json,txt}`, and the notes are in `docs/registered/README.md` ("**Read
2026-10-22**") and `docs/FINDINGS.md` (calibration rental C2).

## Where it came from

Lambda Cloud instance 70d96347 (one GH200, us-east-3), launched 18:42:47Z and terminated 19:19:24Z on
2026-10-21: about 37 min, about $1.40 at the registration's $2.29/h (an estimate; the instance, the region
and the two times are the rental's records, not in the run files, and the bill is in neither). The VM's setup
ran 18:44:56Z to 18:48:01Z and the driver 18:48:02Z to 19:18:45Z. Board `c3f52a` (`PREFLIGHT.txt`), which no
earlier published session names. `scripts/gh200_cal2_session.sh --plan scripts/plans/rentalC2-cal-2026-10.plan`
ran at commit 09b340ce (`commit.txt`); the plan's sha256 3ebcff1c... is in `gh200-driver/status`. This
directory is the branch `run-gh200-tile2-20261021t1842`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log. Driver 580.105.08, persistence on, power.max_limit 900 W read before the limit was
set.

## The power limit and the lock

The registration asks for 700 W with no fallback and stops the session unless two read-backs agree:
`--query-gpu=enforced.power.limit` and the GPU scope's Current Power Limit in `nvidia-smi -q -d POWER`. Before
the limit was set (`power-at-start.txt`) the GPU scope read Current and Requested 700 W. After `-pl 700 -sc 0`
the first poll (0 s) read enforced.power.limit 700 W, the -q GPU scope Current 700 W (Requested 700 W) and
power.limit 700 W (`power-set.txt`, `gh200-driver/c2-power.env` `C2_POWER_W=700`); the module scope read 1000 W.
No `power-set-unsettled-700.txt` was written. Every unit ran at 700 W: the driver's `clocks:` line after every
unit reads power.limit 700 W, and the 8 power.limit records on the 4 counter pages read 700.00 W.

The prelude locked 1410 MHz, read clocks.sm back, reset, and recorded `LOCK_PROBE_1410=held`
(`gh200-driver/locks.env`). G-BAND (`gh200-driver/gate-band-1410.json`) then PASSED on the registered fallback
band [1352.19, 1410] MHz: in the primary band [1374.75, 1410] the joint fit was None (1 of 27 OLMoE cells and
7 of 27 Mixtral-8x7B cells in band); in the fallback 24 of 27 and 21 of 27. A2 also passed on the fallback.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | registered as | start to end (UTC) | min | exit | page exit |
|---|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W read back; lock 1410 held | C2-P | 18:48:03 to 18:48:05 | 0 | 0 | |
| 2 | calibrate | mixtral-8x7b tc2 (the ruler, no lock) | C2-P | 18:48:07 to 18:48:32 | 0 | 0 | triad 3725.9 GB/s |
| 3 | tilecensus | tc2: 128x128x64 w8 s3 on mixtral-8x22b, compile only | C2-P | 18:48:34 to 18:48:43 | 0 | 0 | 2 configs, 0 path mismatches |
| 4 | bytes | olmoe-1b-7b f128 (G 64, treads 1 to 9) | C2-F | 18:48:45 to 18:51:57 | 3 | 0 | 3 |
| 5 | bytes | mixtral-8x7b f128 (G 64, treads 1 to 9) | C2-F | 18:51:59 to 18:58:57 | 7 | 0 | 3 |
| 6 | gate | gband at 1410 | C2-F | 18:58:59 to 18:59:00 | 0 | 0 | PASS (fallback band) |
| 7 | timed | mixtral-8x22b t128g1 (G 1, n 2, six labels, NATIVE and SHARED) | C2-T | 18:59:02 to 19:06:20 | 7 | 0 | R3 3 (stored G1 FAIL) |
| 8 | bytes | mixtral-8x22b b128g1 (G 1, the same cells) | C2-B | 19:06:23 to 19:11:27 | 5 | 0 | 0 |
| 9 | timed | mixtral-8x7b t128g1m (G 1, A2's four E1-CAL cells) | C2-MT | 19:11:29 to 19:16:28 | 5 | 0 | R3 0 |
| 10 | bytes | mixtral-8x7b b128g1 (G 1, the same cells) | C2-M | 19:16:30 to 19:18:43 | 2 | 0 | 0 |

10 of 10 units exit 0 in about 31 min of units against 48 estimated; none was dropped; the driver ends with exit
0. A counter page exit 3 is the route's own page gates (the two G 64 f128 pages: OLMoE V4, V5, C1 and C6 FAIL,
C2 REFUSE; Mixtral-8x7B V5, V7, C2 and C6 FAIL, C1 REFUSE), which C2 does not read: no C2 reading uses a page
verdict. Each byte unit's `analyse` step exits 3 as well (`summary.json`). The 8x22B timed page's R3 exit 3
(INVALID) is its stored G1 lock-thermal gate, which lists clock drift on the uniform cell (NATIVE and SHARED)
with 0 slipped repeats in 108; the registered admission (`tcommon.timed_page`) does not read the stored G1 and
rebuilds it under r7-median-run3 from `cells.csv`: PASS, 0 of 108 repeat rows non-clean. Mixtral-8x7B's page
stores G1 PASS and rebuilds to PASS (0 of 72).

## What the reader reads (`scripts/scoring/cal2/cal2.read.txt`)

| reading | outcome | numbers |
|---|---|---|
| board | valid | C2_POWER_W=700, LOCK_PROBE_1410=held, G-BAND PASS (fallback); the commit check passed on 73 files at 09b340ce; the fingerprint reproduces |
| F1 on C2's board | par 41.59 us, tau 412.5 us | per page: OLMoE 30.60 us (325 us), Mixtral-8x7B 52.58 us (500 us); A2's par 42.02 us (425 us) |
| C2-R1 | X = +4.57%, REGIME | six NATIVE n 2 cells kept, none dropped: PT +4.38, PW +4.08, PT2 +4.68, PW2 +5.26, uniform +4.49, balanced +4.53; X over A2's four labels +4.37; X at A2's par +4.57 |
| the table | spread 5.39 | OLMoE +0.13, Qwen2-57B -0.82, Mixtral-8x7B +3.13 (registered), Mixtral-8x22B +4.57 |
| pass estimates | | A (4 models) 0.0045, B (4 models) 0.0112, D (O, Q, X) 0.0049 |
| C2-S1 | x per GEMM on the timed cells | w1 median 1.188 (1.162 to 1.194), w2 median 1.071 (1.040 to 1.124) |
| C2-S2 | 8x22B b128g1 | M4 over counted reads: w1 +0.05% (worst +0.08), w2 +32.53% (worst +41.27); knee cycles error: w1 +1.99%, w2 +1.00% (6 cells each) |
| C2-S3 | 8x22B sag | own fit 51.06 us, tau 475 us; C2's par minus the measured off: mean -8.01 us, rms 9.14 us over 24 GEMMs |
| C2-S4 | SAME-AS-S1 | Mixtral-8x7B b128g1 knee cycles error w1 +4.20% (S1's board +5.42%), w2 -1.26% (-0.94%); M4 over counted reads w1 -0.21%, w2 +22.28% |
| C2-S5 | census COMPLETE; triad inside | 2 CTAs per SM compiled and priced on w1 and w2; triad 3725.87 GB/s against A2's 3726.2 and the 15-rental range 3725.6 to 3727.6 |
| C2-S6 | REPRODUCES | Mixtral-8x7B held-out 128 G 1 mean on C2's board +3.37% (A2's +3.13); split: GEMMs +3.20% of time, per-call +0.17% |

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and the consoles;
  `gh200-driver/c2-power.env`: the power outcome; `gh200-driver/locks.env`: the lock probe;
  `gh200-driver/gate-band-1410.json`: G-BAND
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the -q -d POWER snapshots before `-pl`,
  before calibrate and after the 700 W read-back
- `results/2026-10-21-nvidia_gh200_480gb-<model>-<label>-r3-counters/lock1410/`: the four counter pages (C2-F
  f128 on OLMoE and Mixtral-8x7B, C2-B b128g1 on Mixtral-8x22B, C2-M b128g1 on Mixtral-8x7B), each
  `r3c-g<G>.json` with `summary.json`, profiles and Triton caches
- `results/gaps-nvidia_gh200_480gb-mixtral-8x22b-t128g1/`, `results/gaps-nvidia_gh200_480gb-mixtral-8x7b-t128g1m/`:
  the two timed pages (`report.json`, `report.txt`, `cells.csv`)
- `results/2026-10-21-nvidia_gh200_480gb-tile-census-mixtral-8x7b-tc2/`: the tile census;
  `results/calibration/` and `measured_nvidia_gh200_480gb.yaml`: calibrate's ruler
- `census*.json`, `census*.profiles/`: the per-model GEMM censuses written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console; `logs/`
  holds every per-unit log, named by model and label

**Histograms.** The 8x22B pages read `docs/registered/2026-10-22-cal2-hist/mixtral-8x22b-128g1.json` and the
Mixtral-8x7B 128 G 1 pages `docs/registered/2026-10-14-tileA-hist/mixtral-8x7b-128g1.json`, each with its sha256
on the page. No trace-derived count is on the VM or in this directory.

The `card:` lines and every page's CARD block carry the card's UUID, as every published session does (the
files are the run branch's, byte for byte); prose here names the board by its hash. The Triton caches'
compiled `cuda_utils*.so` files (9) are left out, as in every earlier session. Every other file of the run
branch is here, byte for byte (288 files against SHA256SUMS, which lists 297).
