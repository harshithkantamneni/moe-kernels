# Decode blind rental: one NVIDIA GH200 480GB, 9 units of the frozen blind plan (decodeB-blind under addenda 2 and 3, finalised mechanically from the CAL2 score before any blind page)

One rental ran `scripts/plans/rentalD-blind-2026-10.plan` unattended: the plan `finalise_decode_d3.py` generated from
decodeB-blind's frozen unit list after the CAL2 rental read GO. It set the GPU power limit to 700 W, ran calibrate's
ruler, then timed vLLM's own decode MoE call at T 1 to 128 at the 1710 MHz lock on five models no decode page had
timed: GRR main pages on Phi-3.5-MoE, Qwen1.5-MoE-A2.7B, JetMoE-8B, deepseek-v2-lite and Granite-3.0-3B-A800M, and
GR1 control pages on Qwen1.5 and Granite. The predictions (`scripts/scoring/decode/decodeB.finalised.json`: 247
cells, 0 not priced, 27 DESCRIPTIVE, rho scaled, T0_GR 28.02 us, bars rms 1.5%, worst 4%, |mean| 0.75%), the plan
and the CAL2 score were committed before any blind page at 82364213, which is also the run commit.

Every reading below is the registered scorer's, run as registered from the repository root with no post-page fix:
`PYTHONHASHSEED=0 .venv/bin/python scripts/scoring/decode/score_blind_d3.py results/published/2026-11-04-nvidia_gh200_480gb-decodeBlind-session/results scripts/scoring/decode/decodeB.finalised.json <out>`.
Its output is `scripts/scoring/decode/blind.score.{json,txt}`; scored on the fetched tree and on this directory, the
`.txt` is byte-identical and the JSON differs only in the power file's path, and `scripts/scoring/rescore.py
decodeBlind` re-scores this directory to the committed files. The notes are in `docs/registered/README.md`
("**Scored 2026-11-04**" on this directory) and `docs/FINDINGS.md` (Decode CAL2 and blind rentals).

## Where it came from

Lambda Cloud instance 984549e6 (one GH200, us-east-3), launched 00:59:36Z and terminated 02:26:22Z on 2026-11-04:
about 87 min, about $3.31 at the registration's $2.29/h (an estimate; the instance, the region and the two times are
the rental's records, not in the run files, and the bill is in neither). The VM's setup ran 01:01:26Z to 01:03:33Z
(7 preflight checks passed) and the driver 01:03:34Z to 02:24:31Z, against a deadline of 03:39:36Z (launch + 160
min). Board `ae5fc5` (`PREFLIGHT.txt`), the board the CAL2 rental ran on the evening before, on a different instance.
The plan's sha256 989811c2... is in `gh200-driver/status` and equals the committed plan's; the run commit 82364213 is
in `commit.txt`. This directory is the branch `run-gh200-tile3-20261104t0059`; `SHA256SUMS.run-branch` and
`PUSHES.run-branch.txt` are its manifest and push log. Driver 580.105.08, persistence on.

## The power limit and the lock

The scorer scores nothing unless `power-set.txt` reads a GPU Current Power Limit of 700 W within 1 W. It does: GPU
Power Readings, Current Power Limit 700.00 W; the module scope reads 1000 W. The prelude recorded persistence Enabled
and power.limit 700.00 W, and the driver's `clocks:` line after every unit reads 700.00 W.

Every timed page held the 1710 MHz lock at G 1 (R3 exit 0 on all seven, worst cell 1710.0 MHz on every page). No
blind cell was dropped: the score reads all 220 kept cells and the 27 DESCRIPTIVE ones beside.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | start to end (UTC) | min | exit | page |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, base tag | 01:03:34 to 01:03:34 | 0 | 0 | |
| 2 | calibrate | mixtral-8x7b dcal (the ruler, no lock) | 01:03:36 to 01:04:01 | 0 | 0 | triad 3726.1 GB/s |
| 3 | r3decode | phi-3.5-moe dm (main, GRR) | 01:04:03 to 01:18:06 | 14 | 0 | 423 cells |
| 4 | r3decode | qwen1.5-moe-a2.7b dm (main, GRR) | 01:18:08 to 01:33:02 | 15 | 0 | 513 cells |
| 5 | r3decode | jetmoe-8b dm (main, GRR) | 01:33:05 to 01:45:19 | 12 | 0 | 423 cells |
| 6 | r3decode | deepseek-v2-lite dm (main, GRR) | 01:45:22 to 01:58:26 | 13 | 0 | 432 cells |
| 7 | r3decode | granite-3.0-3b-a800m dm (main, GRR) | 01:58:29 to 02:10:53 | 12 | 0 | 432 cells |
| 8 | r3decode | qwen1.5-moe-a2.7b dc (control, GR1) | 02:10:56 to 02:17:40 | 7 | 0 | 234 cells |
| 9 | r3decode | granite-3.0-3b-a800m dc (control, GR1) | 02:17:44 to 02:24:27 | 7 | 0 | 234 cells |

9 units in about 81 min against 101 estimated; none was dropped, and all 9 exit 0 (driver exit 0).

## What the scorer reads (`scripts/scoring/decode/blind.score.txt`)

| reading | outcome | numbers |
|---|---|---|
| power | READ | 700.0 W on the GPU scope of `power-set.txt` |
| descriptive cells | 27 | the second-tile cells as addendum 2 registers them, out of E1-DEC |
| bars | finalised | rms 1.5%, worst 4%, \|mean\| 0.75%: tightened from CAL2's sigma_page 0.159%, which puts every bar at its registered floor |
| **E1-DEC pooled** | **FAILS** | 220 cells: rms 11.40%, worst +39.24% (JetMoE L-T2-A2, 59.8 us measured against 83.3 us priced), mean +5.99% |
| Phi-3.5-MoE | FAILS, NOT CERTIFIED | 42 cells: rms 9.52%, worst +24.4%, mean +5.22% |
| Qwen1.5-MoE-A2.7B | FAILS, NOT CERTIFIED | 52 cells: rms 5.31%, worst +15.6%, mean +3.15% |
| JetMoE-8B | FAILS, NOT CERTIFIED | 41 cells: rms 19.46%, worst +39.2%, mean +15.11% |
| deepseek-v2-lite | FAILS, NOT CERTIFIED | 43 cells: rms 2.35%, worst +12.2% (L-T96-A25), mean +0.45% |
| Granite-3.0-3B-A800M | FAILS, NOT CERTIFIED | 42 cells: rms 13.41%, worst +33.8%, mean +7.05% |
| rivals | every one EXCLUDED | rho other, NAIVE-NODEDUP, eager, roofline, line, OLS; roofline closest (12.35% rms against v4d''s 11.41% on its 189 lever cells) |
| GD2 | PASS on both | Qwen1.5 13 matched cells, 0 outside the band; Granite 0 matched cells (all 26 GR1 cells host-bound under GD1) |
| G3-D shuffle | EQUIVALENT on all five | |
| G3-D label (DESCRIPTIVE, never a void) | EQUIVALENT on four, NOT EQUIVALENT on Phi-3.5 | Phi mean +0.44 us against margin 0.40 us; the other four within 0.11 us |
| SIGMA-D (DESCRIPTIVE) | L2-SERVED on all five | median positions 0.76 (deepseek), 0.94 (Granite), 0.88 (JetMoE), 0.81 (Phi), 0.83 (Qwen1.5) |
| rivals on every cell (DESCRIPTIVE) | roofline and OLS NOT EXCLUDED | the other four EXCLUDED |
| knee rows (DESCRIPTIVE) | | p 5.75 rms 11.51%; p 3.75 rms 11.94% |
| as registered, every cell (DESCRIPTIVE) | E1 FAILS, none certified | 247 cells, rms 16.70% |

Each model line in the score carries the certified-domain scope line beside its verdict. By batch (descriptive, from
the score's 220 rows): at T <= 16 the pooled rms is 13.6% (131 cells, mean +8.3%), at T >= 32 6.9% (89 cells, mean
+2.6%); per model at T <= 16 against T >= 32: deepseek-v2-lite 0.91% against 3.47%, Qwen1.5 6.48% against 2.10%,
Phi-3.5 12.37% against 2.78%, Granite 16.81% against 5.36%, JetMoE 22.55% against 13.99%. A positive error means
v4d' prices the call slower than it ran.

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and the consoles;
  `gh200-driver/locks.env`: the registered lock plan; `lock-plan.txt`, `supported-clocks.txt`,
  `supported-locks.txt`: the clock table
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the -q -d POWER snapshots (the last is the one
  the scorer reads)
- `results/gaps-nvidia_gh200_480gb-<model>-<label>/private_weight_reference/nvidia_gh200_480gb-decode-...`: the seven
  timed decode pages (`report.json` and their Triton caches), labels dm and dc
- `results/calibration/` and `measured_nvidia_gh200_480gb.yaml`: calibrate's ruler
- `locked-r3/gh200-20261104T010334Z-dec*/`: each locked run's record (`status`, `summary.json`, `smi.csv`, the R3 log)
- `census.json`, `census.profiles/`: the GEMM census written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session); `logs/` holds every per-unit log

**Pages.** Every decode page reads its registered page file in `docs/registered/2026-10-14-decode-pages/`, with its
sha256 on the page. The pages are SYNTHETIC; no trace-derived count is on the VM or in this directory. This session
has no counter page.

The card's UUID is in the logs and the locked runs' records, as in every published session (the files are the run
branch's, byte for byte); prose here names the board by its hash. The Triton caches' compiled `cuda_utils*.so` files
(8) are left out, as in every earlier session. Every other file of the run branch is here, byte for byte (807 files
against SHA256SUMS, which lists 815).
