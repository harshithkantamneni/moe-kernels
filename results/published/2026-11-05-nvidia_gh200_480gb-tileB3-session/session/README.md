# Tile rental B3: one NVIDIA GH200 480GB, 23 units of the B3 plan (2026-11-04-tileB3-gh200, BLIND: E1-TILE FAILS)

One rental ran `scripts/plans/rentalB3-tile-2026-11.plan` unattended through B3's driver
(`scripts/gh200_tileB3_session.sh`, C2's pinned driver plus the three lines marked B3: the gates b3census, b3sag and
b3sag64). It set the GPU power limit to 700 W and the 1410 MHz lock, ran calibrate's ruler, took B3's compile-only
tile census of the four blind models at 128x128x64 w8 s3 and 64x128x64 w8 s3, took the four sag pages (OLMoE and
Mixtral-8x7B at f128 and f64w8, G 64) with their gates, then the eight blind timed pages of rental B2's 84 NATIVE
cells (Phi-3.5-MoE 64g1, 128g1 and 128g16; Qwen1.5-MoE-A2.7B 128g1 and 128g16; JetMoE-8B 128g16 and 128g1;
deepseek-v2-lite 128g1), the OLMoE f128 repeat and three byte pages. The registration
(`docs/registered/2026-11-04-tileB3-gh200`), its scorer and the plan were committed before any page at a5608079,
which is also the run commit. The predictions of record were finalised from the four sag pages and the gate records
only, with no timed page opened, and committed at 0b82fae3 (`scripts/scoring/tileB3/b3.predictions.json`) before any
blind page was scored.

Every verdict below is the registered scorer's, run as registered from the repository root with no post-page fix:
`PYTHONHASHSEED=0 .venv/bin/python scripts/scoring/tileB3/read_b3.py score --b3 results/published/2026-11-05-nvidia_gh200_480gb-tileB3-session --out-dir scripts/scoring/tileB3`.
Its output is `scripts/scoring/tileB3/tileB3.score.{json,txt}` (the scorer's default `--out-dir`). Scored on the
fetched tree and on this directory, the two runs differ only in the tree's path (the `b3` field and the `pages:`
line); the committed files are this directory's, and `scripts/scoring/rescore.py tileB3` re-scores it to them. The
notes are in `docs/registered/README.md` ("**Scored 2026-11-05**" under B3) and `docs/FINDINGS.md` (Tile rental B3).

## Where it came from

Lambda Cloud instance 58d3a671 (one GH200), launched 22:02:22Z on 2026-11-05 and terminated 00:25:36Z on
2026-11-06: about 143.2 min, about $5.47 at the registration's $2.29/h (an estimate; the instance and the two times
are the rental's records, not in the run files, and the bill is in neither). The registration budgeted about 168 min
of wall and $6.41. The VM's setup ran 22:04:09Z to 22:06:20Z (7 preflight checks passed) and the driver 22:06:21Z to
00:23:50Z, against a deadline of 01:08:22Z (launch + 186 min). Board `ebedab` (`PREFLIGHT.txt`), new to the study.
The plan's sha256 f5ed7386... is in `gh200-driver/status` and equals the registered plan's; the run commit a5608079
is in `commit.txt` and on the ledger's `driver start` line. This directory is the branch
`run-gh200-tileb3-20261105t2202`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and push log.
Driver 580.105.08, persistence on, power.max_limit 900 W read before the limit was set.

## The power limit and the lock

The prelude set 700 W and read it back at the first poll: enforced.power.limit 700.00 W and the `-q -d POWER` GPU
scope's Current Power Limit 700.00 W (Requested 700.00 W); `power-set.txt` reads the same, and the module scope reads
1000 W. `gh200-driver/c2-power.env` reads `C2_POWER_W=700`, and the driver's `clocks:` line after every unit reads
700.00 W. The prelude locked 1410 MHz and read it back (`gh200-driver/locks.env`: `LOCK_PROBE_1410=held`).

Every timed page held the 1410 lock: the eight `timed:` result lines (in `logs/driver-timed-*.log` and the ledger;
`summary.txt` gives each unit's exit) read "G=1 held at lock 1410" or "G=16 held at
lock 1410", R3 exit 0 DONE, worst cell 1410.0 MHz, and every repeat row of every page is clean (0 of 1,512 non-clean;
108 to 324 per page). The `clocks:` lines between units read 1980 MHz because the lock is set per unit, not held
across them.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`)

| unit | step | model, label | start to end (UTC) | min | exit | result |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, lock 1410 | 22:06:21 to 22:06:23 | 0 | 0 | lock held |
| 2 | calibrate | mixtral-8x7b tb (the ruler, no lock) | 22:06:25 to 22:06:50 | 0 | 0 | triad 3725.7 GB/s, GEMM clock 1455 MHz |
| 3 | tilecensus | tb: two tiles on four blind models (compile only) | 22:06:52 to 22:07:04 | 0 | 0 | 16 configs, 0 path mismatches |
| 4 | gate | b3census | 22:07:05 to 22:07:06 | 0 | 0 | PASS (16 compared, no drift; deepseek-v2-lite admitted) |
| 5 | bytes | olmoe-1b-7b f128 (G 64) | 22:07:08 to 22:10:20 | 3 | 0 | page exit 3 |
| 6 | bytes | mixtral-8x7b f128 (G 64) | 22:10:22 to 22:17:20 | 7 | 0 | page exit 3 |
| 7 | gate | G-BAND | 22:17:22 to 22:17:23 | 0 | 0 | PASS (the fallback band [1352.19, 1410]) |
| 8 | gate | b3sag | 22:17:26 to 22:17:26 | 0 | 0 | PASS, par 59.815 us / 612.5 us |
| 9 | bytes | olmoe-1b-7b f64w8 (G 64) | 22:17:28 to 22:20:12 | 3 | 0 | page exit 1 |
| 10 | bytes | mixtral-8x7b f64w8 (G 64) | 22:20:15 to 22:26:54 | 7 | 0 | page exit 3 |
| 11 | gate | b3sag64 | 22:26:56 to 22:26:58 | 0 | 0 | PASS, par64 39.465 us / 337.5 us |
| 12 | timed | phi-3.5-moe t64g1 | 22:27:00 to 22:34:08 | 7 | 0 | R3 exit 0, 108 rows |
| 13 | timed | phi-3.5-moe t128g1 | 22:34:11 to 22:48:10 | 14 | 0 | R3 exit 0, 216 rows |
| 14 | timed | phi-3.5-moe t128g16 | 22:48:12 to 23:02:21 | 14 | 0 | R3 exit 0, 216 rows |
| 15 | timed | qwen1.5-moe-a2.7b t128g1 | 23:02:24 to 23:23:43 | 21 | 0 | R3 exit 0, 324 rows |
| 16 | timed | jetmoe-8b t128g16 | 23:24:42 to 23:38:51 | 14 | 0 | R3 exit 0, 216 rows |
| 17 | timed | qwen1.5-moe-a2.7b t128g16 | 23:38:54 to 23:53:23 | 14 | 0 | R3 exit 0, 216 rows |
| 18 | timed | jetmoe-8b t128g1 | 23:53:26 to 00:00:45 | 7 | 0 | R3 exit 0, 108 rows |
| 19 | timed | deepseek-v2-lite t128g1 | 00:00:48 to 00:08:06 | 7 | 0 | R3 exit 0, 108 rows |
| 20 | bytes | olmoe-1b-7b f128r (the repeat, G 64) | 00:08:09 to 00:11:07 | 3 | 0 | page exit 3 |
| 21 | bytes | phi-3.5-moe b128g1 | 00:11:10 to 00:16:38 | 5 | 0 | page exit 0 |
| 22 | bytes | qwen1.5-moe-a2.7b b128g1 | 00:16:41 to 00:20:55 | 4 | 0 | page exit 0 |
| 23 | bytes | phi-3.5-moe b64g1 | 00:20:58 to 00:23:46 | 3 | 0 | page exit 0 |

23 units in about 137.5 min against 143 estimated; none was dropped or skipped and all 23 exit 0 (driver exit 0).
The page exits on the G 64 counter pages are R3's own analyse gates, the same ones A2's identical units failed: OLMoE
f128 and f128r V4, V5, C1, C6; Mixtral f128 V5, V7, C2, C6; OLMoE f64w8 C1, C6 (claims only, exit 1); Mixtral f64w8
V7, C2, C6. The three G 1 byte pages fail R3's attribution gate V1 in analyse, as A2's histogram byte pages did. No B3
gate or verdict reads those exits: the sag gates fit the pages' cells, and the byte pages are read under the clock
rule for the DESCRIPTIVE B-BYTES leg. The 59 s between units 15 and 16 is the results push.

## What the scorer reads (`scripts/scoring/tileB3/tileB3.score.txt`)

| reading | outcome | numbers |
|---|---|---|
| commit | checked | a5608079 on the ledger, the registration, plan and every code pin at their registered bytes |
| board | valid | `C2_POWER_W=700`, `LOCK_PROBE_1410=held`, G-BAND PASS (fallback band) |
| gates | census PASS, sag PASS, sag64 PASS | par 59.815 us / 612.5 us (OLMoE 48.36 / 575, rms 6.34; Mixtral 71.27 / 650, rms 13.2); par64 39.465 / 337.5 (OLMoE 31.93 / 300; Mixtral 47.0 / 375) |
| pages | all 8 counted in ALL and in CLEAN | 0 non-clean repeat rows on every page; the 84 SHARED cells carry no prediction and are not scored |
| G3 | PASS (EQUIVALENT, homogeneous) | uniform minus balanced +0.135%, 90% interval [+0.059, +0.211]% against the 0.36% margin, 14 cells in 8 clusters; no stratum voided |
| **E1-TILE** | **FAILS** | pooled 84 cells rms 2.41%, mean +0.29%, worst -8.22%; ALL and CLEAN agree on every verdict |
| certification | none | Phi-3.5-MoE, Qwen1.5-MoE-A2.7B and JetMoE-8B each NOT CERTIFIED |

Per model (the registered bar: rms <= 3%, every cell within 8%, |mean| <= 1.5%):

| model | cells | rms | mean | worst | own bar |
|---|---:|---:|---:|---:|---|
| pooled | 84 | 2.41% | +0.29% | -8.22% | FAILS (worst cell) |
| Phi-3.5-MoE | 30 | 3.50% | +1.30% | -8.22% | FAILS (rms, worst cell) |
| Qwen1.5-MoE-A2.7B | 30 | 1.18% | -0.74% | -3.68% | holds (descriptive) |
| JetMoE-8B | 18 | 2.05% | +0.57% | +3.82% | holds (descriptive) |
| deepseek-v2-lite | 6 | 0.65% | -0.40% | -0.95% | met (descriptive; in the pooled bar only, never certified) |

Phi-3.5-MoE by class: 128g1 mean +3.38% (12 cells, rms 4.17%, worst +6.48%); 64g1 mean -3.24% (6 cells, rms 4.66%,
worst -8.22%); 128g16 mean +1.50% (12 cells, rms 1.52%). Under the registered rule a model is certified only when
E1-TILE HOLDS, so the per-model "holds" above certifies nothing. The CLEAN view's own block in the score JSON
(`views.addendum.CLEAN.E1-TILE.per_model`) labels Qwen1.5 and JetMoE "CERTIFIED": that label is local to the view and
is overridden by the top-level `certification` (see `docs/registered/README.md`).

DESCRIPTIVE beside the verdict: B2's bar (2%, 5%, 1%) FAILS (JetMoE's 2.05% rms also misses it); the rivals on the
same cells read v5'' 1.99% rms (mean -0.31%), round 4's model 2.08% (83 cells, mean -0.47%) and v4 5.21% (mean
-4.29%); B-BYTES reads outside B2's byte bar (w1 17 rows, mean +3.60%, rms 5.06%; w2 mean +0.17%, rms 0.91%);
E2-TILE (the frozen model's time from the counted bytes, 17 128x128 cells) reads rms 1.49%, mean -1.05%, inside B2's
E2 bar; the sag drift: the end-of-rental OLMoE f128 repeat fits A 39.82 us, tau 400 us against the early page's
48.36 us and 575 us (36.55 against 39.86 us of sag at 1 ms). The board's 128x128 par (A 59.8 us) lies above the
SEEN boards' span (41.59 to 54.25 us) that the registration printed beside the gate (DESCRIPTIVE, no STOP).

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger, the per-unit exits and the
  consoles (the `timed:` result lines are in `logs/driver-timed-*.log` and the ledger; `gh200-driver/timed-results`
  is 0 bytes on this run); `gh200-driver/locks.env`, `a2-lock.env`, `c2-power.env`: the lock and power records the scorer
  reads; `gh200-driver/gate-*-1410.json`: the four gate records (b3census, band, b3sag, b3sag64); `lock-plan.txt`,
  `supported-clocks.txt`, `supported-locks.txt`: the clock table
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the `-q -d POWER` snapshots
- `results/gaps-nvidia_gh200_480gb-<model>-t<tile>g<G>/`: the eight blind timed pages; `locked-r3/<tag>/`: each
  timed page's lock ledger (`smi.csv`, `status`, `summary.json`, the R3 log)
- `results/2026-11-0{5,6}-nvidia_gh200_480gb-<model>-<label>-r3-counters/lock1410/`: the four sag pages, the OLMoE
  f128 repeat and the three byte pages (`r3c-g*.json`, `summary.json`, and `r3c-g*.profiles/` with the ncu report,
  CSV, manifest, plan, log and Triton cache); `counters-dir-<model>-<label>.txt` names each page's directory on the VM
- `results/2026-11-05-nvidia_gh200_480gb-tile-census-mixtral-8x7b-tb/tile-census.json`: B3's own tile census, the
  one the census gate reads
- `results/calibration/` and `measured_nvidia_gh200_480gb.yaml`: calibrate's ruler
- `census.json` and `census.profiles/`: the preflight's GEMM census; `census-<model>.json` and their `.profiles/`:
  the GEMM censuses the byte units wrote for OLMoE (unit 5), Phi-3.5-MoE (unit 21) and Qwen1.5-MoE-A2.7B (unit 22)
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session); `logs/` holds every per-unit log

The card's UUID is in `PREFLIGHT.json`, the census files and `setup_vm.log`, as in every published session (the files
are the run branch's, byte for byte); prose here names the board by its hash. The Triton caches' compiled
`cuda_utils*.so` files (20) are left out, as in every earlier session. Every other file of the run branch is here,
byte for byte (604 files against SHA256SUMS, which lists 624).
