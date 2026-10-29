# Decode CAL rental: one NVIDIA GH200 480GB, 13 units of the decode stage's CAL plan under its second addendum (docs/registered/README.md, 2026-10-14, 2026-10-20 and 2026-10-28)

One rental ran `scripts/plans/rentalD-cal-d2-2026-10.plan` unattended through the decode stage's pinned driver
(`scripts/gh200_decode_session.sh`, the only driver with the `r3decode` and `decodebytes` steps the ledger shows).
It set the GPU power limit to 700 W, ran calibrate's ruler and a compile-only decode tile census, then timed vLLM's
own decode MoE call at T 1 to 128 on OLMoE-1B-7B and Mixtral-8x7B at the 1710 MHz lock: the GRR main pages (one CUDA
graph of back-to-back calls on rotating weight copies), the GR1 control pages, Mixtral's E240 eager page, two decode
counter pages, the DECL pages and OLMoE's main page again as the replicate. The three decode registrations
(`2026-10-14-decodeC-cal-gh200`, `2026-10-20-decode-tilesource-gh200`, `2026-10-28-decode-addendum2-gh200`), the
scorer `scripts/scoring/decode/score_cal_d2.py`, the pages and the plan were committed before any decode page
(a398c6a0, which is also the run commit).

Every reading below is the registered scorer's, run as registered from the repository root with no post-page fix:
`PYTHONHASHSEED=0 .venv/bin/python scripts/scoring/decode/score_cal_d2.py results/published/2026-10-29-nvidia_gh200_480gb-decodeCAL-session/results <out> --tile-scores scripts/scoring/tileA2`.
Its output is `scripts/scoring/decode/cal.score.{json,txt}`; the notes are in `docs/registered/README.md`
("**Scored 2026-10-29**" on this directory) and `docs/FINDINGS.md` (Decode CAL rental).

## Where it came from

Lambda Cloud instance 0e2c1113 (one GH200, us-east-3), launched 03:23:12Z and terminated 04:42:14Z on 2026-10-29:
about 79 min, about $3.02 at the registration's $2.29/h (an estimate; the instance, the region and the two times
are the rental's records, not in the run files, and the bill is in neither). The VM's setup ran 03:25:11Z to
03:27:06Z (7 preflight checks passed) and the driver 03:27:07Z to 04:40:33Z, against a deadline of 06:08:12Z
(launch + 165 min). Board `183300` (`PREFLIGHT.txt`), the board calibration rental C4 ran on two days earlier, on a
different instance. The plan's sha256 6bf17829... is in `gh200-driver/status` and equals the registered
`plan_sha256`; the run commit a398c6a0 is in `commit.txt`. This directory is the branch
`run-gh200-tile1-20261029t0323`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and push log.
Driver 580.105.08, persistence on.

## The power limit and the lock

Addendum 2 (D2-3) scores nothing unless the session's `power-set.txt` reads a GPU Current Power Limit of 700 W
within 1 W. It does: GPU Power Readings, Current Power Limit 700.00 W, Requested 700.00 W (Default and Max 900 W);
the module scope reads 1000 W. `power-at-start.txt` and `power-before-calibrate.txt` already read 700.00 W on the
GPU scope. The prelude recorded persistence Enabled and power.limit 700.00 W, and the driver's `clocks:` line after
every unit reads power.limit 700.00 W.

The decode pages run at the 1710 MHz lock (decodeC-cal, unchanged by both addenda). Every timed page held it at
G 1 (R3 exit 0 on all eight): worst cell 1710.0 MHz on every page except Mixtral's main page, 1695.0 MHz, inside the
registered 15 MHz slip rule (`--slip-policy cell`; no main-page cell dropped).

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | start to end (UTC) | min | exit | page |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, base tag | 03:27:07 to 03:27:07 | 0 | 0 | |
| 2 | calibrate | mixtral-8x7b dcal (the ruler, no lock) | 03:27:26 to 03:27:51 | 0 | 0 | triad 3724.9 GB/s |
| 3 | tilecensus | dec: 4 decode tiles on 7 models, compile only | 03:27:53 to 03:28:09 | 0 | 3 | 48 configs, 0 path mismatches, 4 errors (deepseek-v2-lite) |
| 4 | r3decode | olmoe-1b-7b dm (main, GRR) | 03:28:10 to 03:42:04 | 14 | 0 | 468 cells, worst 1710.0 MHz |
| 5 | r3decode | mixtral-8x7b dm (main, GRR) | 03:42:06 to 03:58:40 | 17 | 0 | 459 cells, worst 1695.0 MHz |
| 6 | r3decode | olmoe-1b-7b dc (control, GR1) | 03:58:43 to 04:05:26 | 7 | 0 | 234 cells, worst 1710.0 MHz |
| 7 | r3decode | mixtral-8x7b dc (control, GR1) | 04:05:29 to 04:12:12 | 7 | 0 | 234 cells, worst 1710.0 MHz |
| 8 | r3decode | mixtral-8x7b de (E240, eager) | 04:12:16 to 04:16:09 | 4 | 0 | 126 cells, worst 1710.0 MHz |
| 9 | decodebytes | olmoe-1b-7b dn (counter page on the control cells) | 04:16:13 to 04:17:28 | 1 | 0 | grid PASS, 26 cells, 260 launches |
| 10 | decodebytes | mixtral-8x7b dn | 04:17:32 to 04:19:27 | 2 | 0 | grid PASS, 26 cells, 260 launches |
| 11 | r3decode | olmoe-1b-7b dd (DECL) | 04:19:30 to 04:22:33 | 3 | 0 | 90 cells, worst 1710.0 MHz |
| 12 | r3decode | mixtral-8x7b dd (DECL) | 04:22:37 to 04:26:30 | 4 | 0 | 90 cells, worst 1710.0 MHz |
| 13 | r3decode | olmoe-1b-7b dr (replicate of dm) | 04:26:34 to 04:40:28 | 14 | 0 | 468 cells, worst 1710.0 MHz |

13 units in about 73 min against 106 estimated; none was dropped. 12 exit 0. The tilecensus unit exits 3 on
deepseek-v2-lite's known census error ("16 rows per expert is not an integer token count for E=64 k=6", and the
same at 32 and 64 rows), which the addendum expected (readiness I7) and no decode scorer reads; the driver therefore
ends with exit 3.

## What the scorer reads (`scripts/scoring/decode/cal.score.txt`)

| reading | outcome | numbers |
|---|---|---|
| power | READ | 700.0 W on the GPU scope of `power-set.txt` |
| descriptive cells | 11 | the registered second-tile cells, out of the fit, the rho selection and every E1 bar |
| fit | scaled selected | sub-wave rms 4.87% (41 cells) against fixed 19.56%; T0_GR 24.83 us, s_naive -8.37 us, s_block -10.41 us |
| E1-CAL-IN-SAMPLE | FAILS | 92 cells: rms 4.75%, worst -22.7% (OLMoE L-T2-A8, naive path, 61.2 us measured against 47.3 us), mean +1.67% |
| G3-D shuffle | EQUIVALENT | margin 1.87 us |
| G3-D label | NOT EQUIVALENT | margin 0.52 us; Mixtral -1.66, -2.02, -1.60 us at T 4, 16, 64; OLMoE within 0.12 us |
| G3-D fill | NOT EQUIVALENT | margin 2.83 us, mean +3.29 us |
| REP | sigma_page 0.19% | 47 kept pairs; 0.19% over all 52 (DESCRIPTIVE) |
| GD2 | PASS | median +3.66 us, band [-0.46, 5.91] us, 40 matched cells |
| GD3 | NOT SCORED | all 14 E240 cells fail GD1 (host-bound by the probe) |
| STOP rule | STOP | E1-CAL-IN-SAMPLE FAILS; G3-D label NOT EQUIVALENT. The blind decode rental does not run |
| SIGMA-D (DESCRIPTIVE) | Mixtral MIXED, OLMoE MIXED | median position 0.41 (6 of 6 read) and 0.62 (5 of 5) |
| knee rows (DESCRIPTIVE) | | p 5.75 rms 4.69%; p 3.75 rms 4.96% |
| as registered (DESCRIPTIVE) | E1 FAILS, STOP | 103 cells, rms 8.92% |

By path (descriptive, from the `path` field of the score's rows): block-scan 5 cells rms 1.02%, small-batch 78 cells
rms 4.11% mean +1.95%, naive 9 cells rms 9.14%.

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and the consoles;
  `gh200-driver/locks.env`: the registered lock plan; `lock-plan.txt`, `supported-clocks.txt`,
  `supported-locks.txt`: the clock table
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the -q -d POWER snapshots (the last is the one
  the scorer reads)
- `results/gaps-nvidia_gh200_480gb-<model>-<label>/private_weight_reference/nvidia_gh200_480gb-decode-...`: the eight
  timed decode pages (`report.json` and their Triton caches), labels dm, dc, de, dd and dr as in the table
- `results/2026-10-29-nvidia_gh200_480gb-<model>-dn-r3-counters/lock1710/r3d.json`: the two decode counter pages,
  with `r3d-profile/`
- `results/2026-10-29-nvidia_gh200_480gb-tile-census-mixtral-8x7b-dec/`: the decode tile census;
  `results/calibration/` and `measured_nvidia_gh200_480gb.yaml`: calibrate's ruler
- `locked-r3/gh200-20261029T032707Z-dec*/`: each locked run's record (`status`, `summary.json`, `smi.csv`, the R3 log)
- `census.json`, `census.profiles/`: the GEMM census written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console; `logs/` holds
  every per-unit log, named by step, model and label

**Pages.** Every decode page reads its registered page file in `docs/registered/2026-10-14-decode-pages/`, with its
sha256 on the page. The pages are SYNTHETIC (`scripts/decode_synth.py`); no trace-derived count is on the VM or in
this directory.

The `card:` lines in the logs and the locked runs' records (34 files) carry the card's UUID, as every published
session does (the files are the run branch's, byte for byte); prose here names the board by its hash. The Triton caches' compiled
`cuda_utils*.so` files (9) are left out, as in every earlier session. Every other file of the run branch is here,
byte for byte (890 files against SHA256SUMS, which lists 899).
