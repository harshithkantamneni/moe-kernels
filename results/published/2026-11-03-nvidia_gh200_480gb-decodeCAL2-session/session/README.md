# Decode CAL2 rental: one NVIDIA GH200 480GB, 21 units of the decode stage's CAL2 plan under its third addendum (docs/registered/2026-10-29-decode-addendum3-gh200, registered before any CAL2 page)

One rental ran `scripts/plans/rentalD-cal2-d3-2026-10.plan` unattended. The plan is addendum 2's CAL plan, every unit
line byte for byte and in its order, followed by the settling units M1 to M4 (addendum 3, D3-4). It set the GPU power
limit to 700 W, ran calibrate's ruler and a compile-only decode tile census, then timed vLLM's own decode MoE call at
T 1 to 128 on OLMoE-1B-7B and Mixtral-8x7B at the 1710 MHz lock: the GRR main pages, the GR1 control pages, Mixtral's
E240 eager page, two decode counter pages, the DECL pages and OLMoE's main page again as the replicate. Then the
settling units: two tile counter pages (M1), OLMoE's gap page and its counter page (M4), two wide counter pages (M2)
and Mixtral's label page at pad 0 and pad 52 (M3). Addendum 3 (model v4d', the G3-D label clause DESCRIPTIVE, the
settling units, the D3 driver `scripts/gh200_decode3_session.sh`, the scorer `scripts/scoring/decode/score_cal_d3.py`
and the plan) was committed before any CAL2 page at 88768aea, which is also the run commit.

Every reading below is the registered scorer's, run as registered from the repository root with no post-page fix:
`PYTHONHASHSEED=0 .venv/bin/python scripts/scoring/decode/score_cal_d3.py <tree>/results <out> --tile-scores scripts/scoring/tileA2`.
It was first run on the fetched tree; that output is committed as `scripts/scoring/decode/cal2.score.{json,txt}`
(82364213, together with the blind finalisation it fed). Run again on this directory, the `.txt` is byte-identical
and the JSON differs only in the power file's path; `scripts/scoring/rescore.py decodeCAL2` checks that. The notes are
in `docs/registered/README.md` ("**Scored 2026-11-03**" on this directory) and `docs/FINDINGS.md` (Decode CAL2 and
blind rentals).

## Where it came from

Lambda Cloud instance 4be9ec70 (one GH200, us-east-3), launched 22:03:20Z on 2026-11-03 and terminated 23:32:34Z:
about 89 min, about $3.41 at the registration's $2.29/h (an estimate; the instance, the region and the two times are
the rental's records, not in the run files, and the bill is in neither). The VM's setup ran 22:05:20Z to 22:07:20Z
(7 preflight checks passed) and the driver 22:07:21Z to 23:30:51Z, against a deadline of 01:23:20Z on 2026-11-04
(launch + 200 min). Board `ae5fc5` (`PREFLIGHT.txt`), new to the study. The plan's sha256 3d0f688a... is in
`gh200-driver/status` and equals the registered `plan_sha256`; the run commit 88768aea is in `commit.txt`. This
directory is the branch `run-gh200-tile2-20261103t2203`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its
manifest and push log. Driver 580.105.08, persistence on.

## The power limit and the lock

Addendum 2 (D2-3, kept by addendum 3) scores nothing unless the session's `power-set.txt` reads a GPU Current Power
Limit of 700 W within 1 W. It does: GPU Power Readings, Current Power Limit 700.00 W; the module scope reads 1000 W.
The prelude recorded persistence Enabled and power.limit 700.00 W, and the driver's `clocks:` line after every unit
reads 700.00 W.

Every timed page held the 1710 MHz lock at G 1 (R3 exit 0 on all eleven r3decode units, worst cell 1710.0 MHz on
every page). One Mixtral main-page cell (L-T64-A4) is dropped by the scorer as host-bound under GD1, so the fit reads
91 kept cells where the first CAL read 92.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | start to end (UTC) | min | exit | page |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, base tag | 22:07:21 to 22:07:21 | 0 | 0 | |
| 2 | calibrate | mixtral-8x7b dcal (the ruler, no lock) | 22:07:23 to 22:07:48 | 0 | 0 | triad 3725.6 GB/s |
| 3 | tilecensus | dec: 4 decode tiles on 7 models, compile only | 22:07:49 to 22:08:05 | 0 | 3 | 48 configs, 0 path mismatches, 4 errors (deepseek-v2-lite) |
| 4 | r3decode | olmoe-1b-7b dm (main, GRR) | 22:08:07 to 22:22:01 | 14 | 0 | 468 cells |
| 5 | r3decode | mixtral-8x7b dm (main, GRR) | 22:22:03 to 22:38:37 | 17 | 0 | 459 cells |
| 6 | r3decode | olmoe-1b-7b dc (control, GR1) | 22:38:40 to 22:45:24 | 7 | 0 | 234 cells |
| 7 | r3decode | mixtral-8x7b dc (control, GR1) | 22:45:27 to 22:52:10 | 7 | 0 | 234 cells |
| 8 | r3decode | mixtral-8x7b de (E240, eager) | 22:52:15 to 22:56:08 | 4 | 0 | 126 cells |
| 9 | decodebytes | olmoe-1b-7b dn (counter page on the control cells) | 22:56:12 to 22:57:27 | 1 | 0 | grid PASS, 26 cells, 260 launches |
| 10 | decodebytes | mixtral-8x7b dn | 22:57:31 to 22:59:25 | 2 | 0 | grid PASS, 26 cells, 260 launches |
| 11 | r3decode | olmoe-1b-7b dd (DECL) | 22:59:29 to 23:02:32 | 3 | 0 | 90 cells |
| 12 | r3decode | mixtral-8x7b dd (DECL) | 23:02:36 to 23:06:29 | 4 | 0 | 90 cells |
| 13 | r3decode | olmoe-1b-7b dr (replicate of dm) | 23:06:33 to 23:20:27 | 14 | 0 | 468 cells |
| 14 | decodebytes | olmoe-1b-7b dt (M1, tiles) | 23:20:31 to 23:21:01 | 1 | 0 | grid PASS, 6 cells, 60 launches |
| 15 | decodebytes | mixtral-8x7b dt (M1, tiles) | 23:21:05 to 23:21:45 | 1 | 0 | grid PASS, 6 cells, 60 launches |
| 16 | r3decode | olmoe-1b-7b dg (M4, gaps, GRR) | 23:21:49 to 23:24:02 | 2 | 0 | 72 cells |
| 17 | decodebytes | olmoe-1b-7b dgn (M4, gaps counters) | 23:24:07 to 23:24:41 | 1 | 0 | grid PASS, 8 cells, 80 launches |
| 18 | decodebytes | olmoe-1b-7b dw (M2, wide kernels) | 23:24:45 to 23:25:35 | 1 | 0 | grid PASS, 6 cells, 60 GEMM and 95 other launches |
| 19 | decodebytes | mixtral-8x7b dw (M2, wide kernels) | 23:25:40 to 23:26:51 | 1 | 0 | grid PASS, 6 cells, 60 GEMM and 90 other launches |
| 20 | r3decode | mixtral-8x7b dp0 (M3, label at pad 0) | 23:26:56 to 23:28:49 | 2 | 0 | 54 cells |
| 21 | r3decode | mixtral-8x7b dp52 (M3, label at pad 52) | 23:28:54 to 23:30:47 | 2 | 0 | 54 cells |

21 units in about 84 min against 136 estimated; none was dropped. 20 exit 0. The tilecensus unit exits 3 on
deepseek-v2-lite's known census error ("16 rows per expert is not an integer token count for E=64 k=6", and the same
at 32 and 64 rows), as in the first CAL rental (readiness I7); no decode scorer reads it, and the driver therefore
ends with exit 3.

## What the scorer reads (`scripts/scoring/decode/cal2.score.txt`)

| reading | outcome | numbers |
|---|---|---|
| power | READ | 700.0 W on the GPU scope of `power-set.txt` |
| descriptive cells | 11 | the registered second-tile cells, out of the fit, the rho selection and every E1 bar |
| fit | scaled selected | sub-wave rms 2.64% (40 cells) against fixed 18.29%; T0_GR 28.02 us, s_naive -12.71 us, s_block -14.43 us |
| E1-CAL-IN-SAMPLE (model v4d') | HOLDS | 91 cells: rms 2.29%, worst -6.35% (OLMoE L-T32-A27, 127.5 us measured against 119.4 us), mean +0.17% |
| G3-D shuffle | EQUIVALENT | margin 1.87 us |
| G3-D label (DESCRIPTIVE under addendum 3) | NOT EQUIVALENT | margin 0.51 us; Mixtral mean -2.29 us over 3 pairs (-2.04, -1.73, -3.09 us at T 4, 16, 64); OLMoE mean -0.04 us |
| G3-D fill | NOT EQUIVALENT (not a stop clause) | margin 2.83 us, mean +3.13 us |
| REP | sigma_page 0.159% | 47 kept pairs; blind bars at their floors (rms 1.5%, worst 4%, \|mean\| 0.75%); 0.162% over all 52 (DESCRIPTIVE) |
| GD2 | PASS | median +3.94 us, band [-0.14, 7.09] us, 40 matched cells |
| GD3 | NOT SCORED | no matched E240 cell |
| STOP rule (addendum 3) | GO | the G3-D label line DESCRIPTIVE; the frozen rule with the label as a clause reads STOP (DESCRIPTIVE) |
| SIGMA-D (DESCRIPTIVE) | Mixtral MIXED, OLMoE L2-SERVED | median position 0.35 (6 of 6 read) and 0.70 (5 of 5) |
| knee rows (DESCRIPTIVE) | | p 5.75 rms 2.32%; p 3.75 rms 2.67% |
| as registered, every cell (DESCRIPTIVE) | E1 FAILS, STOP | 102 cells, rms 8.65% |
| v4d as registered, same pages (DESCRIPTIVE) | E1 FAILS | 91 cells: rms 4.87%, worst -23.2%, mean +1.73% |
| M1 tiles (DESCRIPTIVE) | READ | fix R's ratio re-read on these dn pages: 0.920 (registered 0.9185); per-tile v4d' / counter w1 rates 0.90 to 1.01 |
| M2 wide (DESCRIPTIVE) | READ | align class slope 0.61 us (OLMoE) and 0.59 us (Mixtral) per iteration, against the diagnosis's 0.96 |
| M3 label (DESCRIPTIVE) | FOLLOWS-ADDRESS | expert-5 sets d0 +0.58 us at pad 0, -0.40 us at pad 52, q -0.69 |
| M4 gaps (DESCRIPTIVE) | 12 of 14 GEMM rows agree with fix W | the two that do not are w2 rows (G13-R3, G15-R1) read MIXED |

By path (descriptive, from the `path` field of the score's 91 rows): block-scan 5 cells rms 1.09%, small-batch 77
cells rms 2.42% mean +0.26%, naive 9 cells rms 1.92%. By model: OLMoE 47 cells rms 2.16% mean -0.72%, Mixtral 44
cells rms 2.47% mean +1.20%.

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and the consoles;
  `gh200-driver/locks.env`: the registered lock plan; `lock-plan.txt`, `supported-clocks.txt`,
  `supported-locks.txt`: the clock table
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the -q -d POWER snapshots (the last is the one
  the scorer reads)
- `results/gaps-nvidia_gh200_480gb-<model>-<label>/private_weight_reference/nvidia_gh200_480gb-decode-...`: the eleven
  timed decode pages (`report.json` and their Triton caches), labels dm, dc, de, dd, dr, dg, dp0 and dp52
- `results/2026-11-03-nvidia_gh200_480gb-<model>-<label>-r3-counters/lock1710/r3d.json`: the seven decode counter
  pages (dn on both models, dt on both, dgn, dw on both), with `r3d-profile/`
- `results/2026-11-03-nvidia_gh200_480gb-tile-census-mixtral-8x7b-dec/`: the decode tile census;
  `results/calibration/` and `measured_nvidia_gh200_480gb.yaml`: calibrate's ruler
- `locked-r3/gh200-20261103T220721Z-dec*/`: each locked run's record (`status`, `summary.json`, `smi.csv`, the R3 log)
- `census.json`, `census.profiles/`: the GEMM census written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session); `logs/` holds every per-unit log, named by step,
  model and label

**Pages.** Every decode page reads its registered page file in `docs/registered/2026-10-14-decode-pages/` or, for
the settling units, `docs/registered/2026-10-29-decode3-pages/`, with its sha256 on the page. The pages are SYNTHETIC;
no trace-derived count is on the VM or in this directory.

The card's UUID is in the logs and the locked runs' records, as in every published session (the files are the run
branch's, byte for byte); prose here names the board by its hash. The Triton caches' compiled `cuda_utils*.so` files
(12) are left out, as in every earlier session. Every other file of the run branch is here, byte for byte (1021
files against SHA256SUMS, which lists 1033).
