# Model v7 decode blind rental: one NVIDIA GH200 480GB, 7 units of the registered plan (2026-11-17-v7-decode-blind-gh200)

One rental ran `scripts/plans/rentalV7B-decode-blind-2026-11.plan` unattended through the pinned decode driver
(`scripts/gh200_decode_session.sh`, unchanged), started by `scripts/vm_run_v7b.sh`. It set the GPU power limit to
700 W, ran calibrate's ruler, then timed vLLM's own decode MoE call at T 1 to 128 at the 1710 MHz lock on the decode
stage's three reserved models, which no decode page had timed: GRR main pages on Mixtral-8x22B, Qwen2-57B-A14B and
Granite-3.0-1B-A400M and GR1 control pages on Qwen2-57B and Granite-1B (addendum 5's pages and units, unchanged). The
registration (`docs/registered/2026-11-17-v7-decode-blind-gh200.{txt,json}`, sha256 ee67e173..., FINAL in dd1dff28)
holds model v7's 143 predictions, frozen before any page, and the bars (owner, D4-1: rms 3%, worst 8%, |mean| 1.5%,
per model and pooled, not tightened). It was public at the push of e58c1531 (2026-11-18T03:30:24Z), which is also the
run commit, before the launch.

Every reading below is the registered scorer's, run as registered from the repository root with no post-page fix:
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/scoring/v7blind/score_blind_v7.py results/published/2026-11-18-nvidia_gh200_480gb-v7b-session <out>`.
Its output is `scripts/scoring/v7blind/blind.score.{json,txt}`. The scorer was first run at e58c1531 on the fetched
tree's `vm/` (this directory's layout); scored there and on this directory, the `.txt` is byte-identical and the
JSON differs only in the power file's path and in `git_dirty` (this directory was untracked in the publication
checkout when it was scored, at the same e58c1531). `scripts/scoring/rescore.py v7blind` re-scores this directory to
the committed files. The notes are in `docs/registered/README.md` ("RESULT 2026-11-18" under the v7 decode blind) and
`docs/FINDINGS.md` (Model v7 decode blind).

## Where it came from

Lambda Cloud instance 28a23f07 (one GH200, us-east-3), launched 03:36:57Z on 2026-11-18, termination confirmed
04:44:46Z: about 67.8 min, about $2.93 at Lambda's list price of $2.59/h (an estimate; the instance, the region and
the two times are the rental's records, not in the run files, and the bill is in neither). The registration planned
72 unit-min and about 111 min of wall, $4.24 at $2.29/h (about $4.79 at $2.59/h), with a deadline of launch + 125 min
(05:41:57Z, the driver's own) and a hard cap of launch + 140 min. The VM's setup ran 03:38:42Z to 03:41:18Z (7
preflight checks passed) and the driver 03:41:19Z to 04:41:43Z (driver end exit 0). The last push (DRIVER-DONE) was
verified at 04:43:28Z, every one of the 567 files under the branch's `vm/` matching its SHA256SUMS, and termination
was requested then; the run's deploy key was deleted after the termination was confirmed. Board `e165c7`
(`PREFLIGHT.txt`), new to the study. The plan's sha256 74ded4bc... is on the ledger's `plan:` line and equals the
committed plan's; the run commit e58c1531 is in `commit.txt` and on the ledger's `driver start` line. This directory
is the branch `run-gh200-v7b-20261118t0336`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and
push log. Driver 580.105.08 (the VM booted 570.148.08; `vm_run` upgraded it and rebooted before setup, the owner's
decision of 2026-09-26), persistence on.

## The power limit and the lock

The scorer scores nothing unless `power-set.txt` reads a GPU Current Power Limit of 700 W within 1 W. It does: GPU
Power Readings, Current Power Limit 700.00 W (Requested 700.00 W); the module scope reads 1000 W. The prelude recorded
persistence Enabled and power.limit 700.00 W, and the driver's `clocks:` line after every unit reads 700.00 W.

Every timed page held the 1710 MHz lock at G 1 (R3 exit 0 on all five). Four pages read worst 1710.0 MHz. The
Mixtral-8x22B main page ended HELD_WITH_SLIPS: one slipped repeat, row 52 (L-T128-A2, repeat 1 of 9) at 1680 MHz
(`logs/locked_r3-mixtral-8x22b-dm-*.log`), recorded under `--slip-policy cell` and kept out of that cell's median by
R3. L-T128-A2 is one of the 18 second-tile cells the registration scores DESCRIPTIVE, so no E1-DEC cell carries a
slipped repeat. No cell was dropped: the score reads all 125 E1-DEC cells and the 18 DESCRIPTIVE ones beside (143, the
registration's count).

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | start to end (UTC) | min | exit | page |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, base tag | 03:41:19 to 03:41:19 | 0 | 0 | |
| 2 | calibrate | mixtral-8x7b dcal (the ruler, no lock) | 03:41:21 to 03:41:46 | 0 | 0 | triad 3726.3 GB/s |
| 3 | r3decode | mixtral-8x22b dm (main, GRR) | 03:41:48 to 04:00:13 | 18 | 0 | 423 cells, 1 slipped row |
| 4 | r3decode | qwen2-57b-a14b dm (main, GRR) | 04:00:16 to 04:15:40 | 15 | 0 | 432 cells |
| 5 | r3decode | granite-3.0-1b-a400m dm (main, GRR) | 04:15:42 to 04:27:46 | 12 | 0 | 432 cells |
| 6 | r3decode | qwen2-57b-a14b dc (control, GR1) | 04:27:49 to 04:34:42 | 7 | 0 | 234 cells |
| 7 | r3decode | granite-3.0-1b-a400m dc (control, GR1) | 04:34:45 to 04:41:39 | 7 | 0 | 234 cells |

7 units in about 60 min against 72 estimated; none was dropped, and all 7 exit 0 (driver exit 0).

## What the scorer reads (`scripts/scoring/v7blind/blind.score.txt`)

| reading | outcome | numbers |
|---|---|---|
| power | READ | 700.0 W on the GPU scope of `power-set.txt` |
| descriptive cells | 18 | the second-tile cells as addendum 5 registers them, out of E1-DEC |
| bars | as registered | rms 3%, worst 8%, \|mean\| 1.5% (owner, D4-1), per model and pooled, not tightened |
| **E1-DEC pooled** | **FAILS** | 125 cells: rms 4.17%, worst +17.20% (Granite-1B L-T2-A8, 25.7 us measured against 30.2 us priced), mean +1.61% |
| Mixtral-8x22B | HOLDS, NOT CERTIFIED | 41 cells: rms 1.79%, worst +2.98%, mean +1.49% |
| Qwen2-57B-A14B | HOLDS, NOT CERTIFIED | 43 cells: rms 2.05%, worst -4.28% (L-T96-A45), mean -0.34% |
| Granite-3.0-1B-A400M | FAILS, NOT CERTIFIED | 41 cells: rms 6.74%, worst +17.20%, mean +3.78% |
| rivals | EXCLUDED or NOT SEPARATED | eager, roofline, line, OLS EXCLUDED; rho other and NAIVE-NODEDUP NOT SEPARATED (0 lever cells, under 6) |
| GD2 | PASS on both | Qwen2-57B 26 matched cells, 0 outside the band; Granite-1B 0 matched cells |
| G3-D shuffle | EQUIVALENT on all three | |
| G3-D label (DESCRIPTIVE, never a void) | NOT EQUIVALENT on all three | means -1.12 us (Mixtral-8x22B, margin 1.27), -0.59 us (Qwen2-57B, margin 0.56), -0.30 us (Granite-1B, margin 0.30) |
| SIGMA-D (DESCRIPTIVE) | NOT READ on all three | no readable cell |
| knee rows (DESCRIPTIVE) | | p 5.75 and p 3.75 both rms 4.17% |
| as registered, every cell (DESCRIPTIVE) | E1 FAILS, none certified | 143 cells, rms 10.91% |

A model is CERTIFIED only when its own bar and the pooled bar hold (decodeB-blind's rule, inherited); the pooled bar
fails, so none is, although two models hold their own bars. Each model line in the score carries the certified-domain
scope line beside its verdict. By batch (descriptive, from the score's 125 rows): at T <= 16 the pooled rms is 4.98%
(74 cells, mean +2.03%), at T >= 32 2.57% (51 cells, mean +1.01%); per model at T <= 16 against T >= 32: Mixtral-8x22B
1.74% against 1.87%, Qwen2-57B 2.27% against 1.70%, Granite-1B 8.09% against 3.76%. A positive error means v7 prices
the call slower than it ran; a cell whose priced and measured times are within E1-DEC's 1 us floor scores 0 (12 cells,
8 of them Granite-1B's).

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt` (`summary.new` is its working copy, equal), `logs/driver-*.log`:
  the ledger, the per-unit exits and the consoles; `gh200-driver/locks.env`, `ruler.env`, `tags-used`,
  `lock-plan.txt`, `supported-clocks.txt`, `supported-locks.txt`: the lock records and the clock table
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the `-q -d POWER` snapshots
- `locked-r3/<tag>/` and `logs/locked_r3-*.log`: the lock ladder's record per page (the slip line is in the
  Mixtral-8x22B main page's log)
- `results/gaps-nvidia_gh200_480gb-<model>-dm/` and `-dc/`: the five timed pages (`private_weight_reference/<run>/`:
  `cells.csv`, `report.json`, `report.txt`, the Triton cache)
- `results/calibration/`: calibrate's ruler; `measured_nvidia_gh200_480gb.yaml` is its copy
- `census.json`, `census.profiles/`: the preflight's GEMM census; `PREFLIGHT.{txt,json}`, `probe.json`: the preflight
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session)

The pages are the first decode pages of Mixtral-8x22B, Qwen2-57B-A14B and Granite-3.0-1B-A400M; once published they
are SEEN. The card's UUID is in `PREFLIGHT.json`, the census files and `setup_vm.log`, as in every published session
(the files are the run branch's, byte for byte); prose here names the board by its hash (`dram_counter_route.board`,
the first six hex digits of the sha256 of the bare UUID). `HELD-BACK.txt` on the branch is empty: nothing was over
the size limit. The Triton caches' compiled `cuda_utils*.so` files (6) are left out, as in every earlier session.
Every other file of the run branch is here, byte for byte (561 files against SHA256SUMS, which lists 567).
