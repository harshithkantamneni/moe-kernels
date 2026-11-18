# Model v7 decode domain map rental: one NVIDIA GH200 480GB, 16 units of the registered plan (2026-11-18-v7-domain-map-gh200)

One rental ran `scripts/plans/rentalV7M-domain-map-2026-11.plan` unattended through the pinned decode driver
(`scripts/gh200_decode_session.sh`, unchanged), started by `scripts/vm_run_v7m.sh`, which puts
`scripts/scoring/v7domain/vmsite` on the driver's PYTHONPATH so the 14 synthetic map shapes reach `moe.spec` on the VM
(they are not in `moe/spec.py`). It set the GPU power limit to 700 W, ran calibrate's ruler, then timed vLLM's own
decode MoE call at the 1710 MHz lock on one GRR main page per synthetic shape: family A (path twins, E 8 to 256 at a
6.29 MB slab), family B (slab per expert 2.36 to 37.75 MB at E 32, k 4) and family C (block-scan at E 256). The
registration (`docs/registered/2026-11-18-v7-domain-map-gh200.{txt,json}`, sha256 bf33f4cc..., REGISTERED,
DESCRIPTIVE) holds model v7's 130 predictions, frozen before any page, and the threshold-selection rule that decides
v7's decode validity domain D from these pages mechanically. It was public at the push of 556029e1, which is also
the run commit, pushed before the 09:40:04Z launch (the VM cloned it from GitHub at setup).

Every reading below is the registered reader's, run as registered from the repository root with no post-page fix:
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/scoring/v7domain/read_map.py results/published/2026-11-18-nvidia_gh200_480gb-v7m-session <out>`.
Its output is committed as `scripts/scoring/v7domain/map.domain.{json,txt}` (the `domain.json` and `domain.txt` it
writes), read from a clean checkout with this directory committed (provenance `git_dirty` false), which is the
readout path the domain blind's registration names. Read first on the fetched tree's `vm/` (this directory's layout)
at 6080bd76, the readout's rows, D and every table are the same; the JSON differs only in the power file's path and
in the recorded checkout. `scripts/scoring/rescore.py v7domain` re-reads this directory to the committed files. The
notes are in `docs/registered/README.md` ("RESULT 2026-11-18" under the v7 domain map) and `docs/FINDINGS.md` (Model
v7 decode domain map).

## Where it came from

Lambda Cloud instance 5b9c188e (one GH200), launched 09:40:04Z on 2026-11-18, termination confirmed 10:27:33Z: about
47.5 min, about $2.05 at Lambda's list price of $2.59/h (about $1.81 at the registration's d2common.RATE of $2.29/h;
an estimate: the instance and the two times are the rental's records, not in the run files, and the bill is in
neither). A first launch attempt at 09:07:07Z, in us-east-3, was refused for capacity (insufficient-capacity, no
instance created). The guardian's log names a region only for that refused attempt; it does not record the region of
the launch that ran (the launcher's own output, beside the log in the guardian's state, reads us-east-3 for it). The registration planned 76 unit-min and about 115 min of wall, $4.39
at $2.29/h ($4.96 at $2.59/h), with a deadline of launch + 130 min (11:50:04Z, the driver's own) and a hard cap of
launch + 145 min. The VM's setup ran 09:41:59Z to 09:44:09Z (7 preflight checks passed) and the driver 09:44:11Z to
10:26:12Z (driver end exit 0). The last push (DRIVER-DONE) was verified at 10:26:30Z, every one of the 1379 files
under the branch's `vm/` matching its SHA256SUMS, and termination was requested then; the run's deploy key was deleted
after the termination was confirmed. Board `1e520d` (`PREFLIGHT.txt`), new to the study. The plan's sha256
4d0f423d... is on the ledger's `plan:` line and equals the committed plan's; the run commit 556029e1 is in
`commit.txt` and on the ledger's `driver start` line. This directory is the branch `run-gh200-v7m-20261118t0940`;
`SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and push log. Driver 580.105.08 (the VM booted
570.148.08; `vm_run` upgraded it and rebooted before setup, the owner's decision of 2026-09-26), persistence on.

## The power limit and the lock

The reader reads nothing unless `power-set.txt` reads a GPU Current Power Limit of 700 W within 1 W. It does: GPU
Power Readings, Current Power Limit 700.00 W (Requested 700.00 W); the module scope reads 1000 W. The prelude recorded
persistence Enabled and power.limit 700.00 W, and the driver's `clocks:` line after every unit reads 700.00 W.

Every timed page held the 1710 MHz lock at G 1 (R3 exit 0 on all 14, state held, worst 1710.0 MHz on every page, no
slipped repeat). No cell was dropped: the readout reads all 130 registered cells valid and in scope.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | shape, label | start to end (UTC) | min | exit | page |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, base tag | 09:44:11 to 09:44:11 | 0 | 0 | |
| 2 | calibrate | mixtral-8x7b dcal (the ruler, no lock) | 09:44:13 to 09:44:38 | 0 | 0 | triad 3726.3 GB/s |
| 3 | r3decode | map-a-e8k2-h2048i512 dm (GRR) | 09:44:40 to 09:47:13 | 3 | 0 | 9 cells, 81 rows |
| 4 | r3decode | map-a-e16k4-h2048i512 dm (GRR) | 09:47:16 to 09:49:49 | 3 | 0 | 9 cells, 81 rows |
| 5 | r3decode | map-a-e32k8-h2048i512 dm (GRR) | 09:49:52 to 09:52:25 | 3 | 0 | 9 cells, 81 rows |
| 6 | r3decode | map-a-e64k8-h2048i512 dm (GRR) | 09:52:28 to 09:54:52 | 2 | 0 | 8 cells, 72 rows |
| 7 | r3decode | map-a-e128k8-h2048i512 dm (GRR) | 09:54:55 to 09:57:08 | 2 | 0 | 7 cells, 63 rows |
| 8 | r3decode | map-a-e256k8-h2048i512 dm (GRR) | 09:57:11 to 09:59:24 | 2 | 0 | 6 cells, 54 rows |
| 9 | r3decode | map-b-e32k4-h1024i384 dm (GRR) | 09:59:28 to 10:01:41 | 2 | 0 | 7 cells, 63 rows |
| 10 | r3decode | map-b-e32k4-h1536i512 dm (GRR) | 10:01:45 to 10:04:08 | 2 | 0 | 8 cells, 72 rows |
| 11 | r3decode | map-b-e32k4-h2048i768 dm (GRR) | 10:04:12 to 10:06:55 | 3 | 0 | 9 cells, 81 rows |
| 12 | r3decode | map-b-e32k4-h2560i1152 dm (GRR) | 10:07:00 to 10:09:53 | 3 | 0 | 10 cells, 90 rows |
| 13 | r3decode | map-b-e32k4-h3072i1536 dm (GRR) | 10:09:58 to 10:12:41 | 3 | 0 | 9 cells, 81 rows |
| 14 | r3decode | map-b-e32k4-h4096i1536 dm (GRR) | 10:12:46 to 10:15:29 | 3 | 0 | 9 cells, 81 rows |
| 15 | r3decode | map-c-e256k8-h7168i384 dm (GRR) | 10:15:35 to 10:20:28 | 5 | 0 | 15 cells, 135 rows |
| 16 | r3decode | map-c-e256k8-h4096i1024 dm (GRR) | 10:20:33 to 10:26:06 | 6 | 0 | 15 cells, 135 rows |

16 units in about 42 min against 76 estimated; none was dropped, and all 16 exit 0 (driver exit 0). Rows are the
native repeats (9 per cell) R3 timed.

## What the reader reads (`scripts/scoring/v7domain/map.domain.txt`)

| reading | outcome | numbers |
|---|---|---|
| power | READ | 700.0 W on the GPU scope of `power-set.txt` |
| cells | 130 valid of 130 | all in scope (every active expert in one M-tile) |
| **D** | **EMPTY: no path has an X_p** | the registered rule, mechanically |
| naive | OUTSIDE THE DOMAIN (no X_p) | 16 cells; the full set gives no X (one fold gives one, below) |
| small-batch | OUTSIDE THE DOMAIN (no X_p) | 75 cells; the full set gives no X (one fold gives one, below) |
| block-scan | OUTSIDE THE DOMAIN (no X_p) | 39 cells; neither the full set nor any fold gives an X |
| slab guard | NOT TRIGGERED | 30 family B cells under a wave: b -0.0094 us/MB, 95% [-0.0603, +0.0415], \|b\| x 35 MB 0.33 us |

Why each path has no X_p (the readout's rows, by the registered criteria (i) every e_ub <= 6%, (ii) rms <= 2.5%,
(iii) |mean| <= 1.0%, over every set {T_v7 >= X} of at least 10 cells). Rule (iv) as quoted verbatim from Phase 0
("X_p is the largest of the values this rule gives with each synthetic shape's cells removed in turn") does not say
what happens when the full set or a fold gives no value. The registration's points, part of the registration and
committed at 556029e1 before any page (`.txt` line 27, the JSON's rule points), settle it: "a path where any of them is
none has no X_p and is outside the domain", and `read_map.py` implements exactly that. Here every full set gives none,
so no path has an X_p. DESCRIPTIVE disclosure: one fold, the one without map-b-e32k4-h2560i1152, gives a value on naive
(27.6 us) and on small-batch (111.1 us), every other fold none; under the other reading of the verbatim rule (the
largest over the folds that give a value) the result would differ, naive 27.6 us and small-batch 111.1 us. That
reading is not the registered one, and nothing here says what a blind under it would have read.

- naive: the short calls run slower than priced (map-a-e8k2 L-T1-A2 15.8 us priced, 19.6 us measured, -19.39%;
  map-b-e32k4-h1024i384 L-T1-A4 13.6 against 16.8 us, -19.15%), and the smallest counted set (11 cells, X 27.6 us)
  still holds L-T1-A4 of map-b-e32k4-h2560i1152 (32.5 against 34.9 us, e_ub 6.81%) and reads rms 2.51%, so (i)
  fails at every candidate.
- small-batch: map-b-e32k4-h2560i1152 L-T96-A32 at T_v7 198.0 us (e_ub 7.38%) is 9th by T_v7, so it lies in every
  set of at least 10 cells and (i) fails at every candidate; its twin L-T64-A32 at 184.4 us (e_ub 6.37%) is 11th, in
  every counted set but the last (10 cells from 195.1 us); (ii) fails too (rms 2.64% at its lowest).
- block-scan: from X 70.5 us (36 cells) the sets pass (i) (max e_ub 3.33%) and (ii) (rms 1.83%) but not (iii): the mean
  e is +1.09% there and +2.12% at the 10-cell tail (+1.09% to +2.12% over the counted sets). Below 70.5 us, three short aligned cells (MIN-T8 and
  MIN-T16 at E 128, MIN-T16 at E 256; 37.6 to 39.8 us priced, +11.25% to +14.72%) fail all three.

The rest of the readout is DESCRIPTIVE (v7 out of sample on new synthetic shapes; no bar): pooled 130 cells rms
4.61%, worst -19.39%, mean +0.69%; naive 8.45% (mean -5.06%), small-batch 3.54% (+1.21%), block-scan 4.17% (+2.05%);
family A 5.67%, B 4.58%, C 2.07%. v7's naive-to-aligned step minus the measured one (the same GEMMs), median by E:
+3.43 us (E 8), +3.09 (E 16), +1.71 (E 32), +2.47 (E 64) on small-batch, +4.19 (E 128) and +3.85 (E 256) on
block-scan.

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt` (`summary.new` is its working copy, equal), `logs/driver-*.log`:
  the ledger, the per-unit exits and the consoles; `gh200-driver/locks.env`, `ruler.env`, `tags-used`,
  `lock-plan.txt`, `supported-clocks.txt`, `supported-locks.txt`: the lock records and the clock table
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the `-q -d POWER` snapshots
- `locked-r3/<tag>/` and `logs/locked_r3-*.log`: the lock ladder's record per page
- `results/gaps-nvidia_gh200_480gb-map-<family>-e<E>k<k>-h<H>i<I>-dm/`: the 14 timed pages
  (`private_weight_reference/<run>/`: `cells.csv`, `report.json`, `report.txt`, the Triton cache)
- `results/calibration/`: calibrate's ruler; `measured_nvidia_gh200_480gb.yaml` is its copy
- `census.json`, `census.profiles/`: the preflight's GEMM census; `PREFLIGHT.{txt,json}`, `probe.json`: the preflight
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session)

The pages time synthetic shapes only (no real model's name, config or checkpoint); no reserved or blind model is
timed here. The card's UUID is in `PREFLIGHT.json`, the census files and `setup_vm.log`, as in every published
session (the files are the run branch's, byte for byte); prose here names the board by its hash
(`dram_counter_route.board`, the first six hex digits of the sha256 of the bare UUID). `HELD-BACK.txt` on the branch
is empty: nothing was over the size limit. The Triton caches' compiled `cuda_utils*.so` files (15) are left out, as in
every earlier session. Every other file of the run branch is here, byte for byte (1364 files against SHA256SUMS, which
lists 1379).
