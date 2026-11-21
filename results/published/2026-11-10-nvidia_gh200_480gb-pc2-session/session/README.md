# Rental PC2: one NVIDIA GH200 480GB, 10 units of the PC2 plan (2026-11-10-pc2-gh200, MEASUREMENT: certifies nothing)

One rental ran `scripts/plans/rentalPC2-2026-11.plan` unattended through PC2's driver (`scripts/gh200_pc2_session.sh`,
PC's pinned driver plus the lines marked PC2), started by `scripts/vm_run_pc2.sh`. It set 700 W and the 1410 MHz
session lock, probed the 1710 MHz lock, took PC2's compile-only tile census over five tiles, ran the regcheck and one
perturbation gate over all six stamps units, then took per-CTA start and end stamps through the unchanged instrumented
copy (`moe/instrumented/fused_moe_instr.py`, the "ends" level) at the 1710 MHz lock on PC2's registered pages: pc2-d
(the 16x64x128 w4 s4 decode tile, G 4 to 1024 and F ladders at G 8 and 128), pc2-s (short slabs of 66 to 459 KB on
64x128x64 at G 8, 32, 128 and 256), pc2-n (naive-path cells of Mixtral-8x7B, JetMoE-8B, Phi-3.5-MoE,
Granite-3.0-3B-A800M, Qwen1.5-MoE-A2.7B, deepseek-v2-lite and a k 8 synthetic shape, each with its aligned twin or
naive pair), pc2-w (single waves of 165 to 528 CTAs on 32x64x128 and 32x128x64, and rungs past one wave), pc2-d3 (the
decode tile at s3) and pc2-r (a replicate of three rungs). The registration (`docs/registered/2026-11-10-pc2-gh200`),
its reader and the plan were committed before any page at e7786f4e, which is also the run commit.

PC2 is a MEASUREMENT rental. Every reading is DESCRIPTIVE; it registers no model form, no constant, no bar and no gate,
and it moves no registered verdict. Every reading below is the registered reader's, run as registered from the
repository root with no post-page fix:
`PYTHONHASHSEED=0 .venv/bin/python scripts/scoring/pc2/read_pc2.py --pc2 results/published/2026-11-10-nvidia_gh200_480gb-pc2-session --out-dir scripts/scoring/pc2`.
Its output is `scripts/scoring/pc2/pc2.read.{json,txt}`. Read on the fetched tree and on this directory, the two
outputs differ only in the tree's path (the `pc2` field and the `pages:` line); the committed files are this
directory's, and `scripts/scoring/rescore.py pc2` re-reads it to them. The text file caps each reading's line at 20,000
characters (the reader's `txt`, pinned), so PC2-R1, R2, R3 and R4 are cut there; the JSON holds every value. The notes
are in `docs/registered/README.md` ("**Read 2026-11-10**" under rental PC2) and `docs/FINDINGS.md` (Rental PC2).

## Where it came from

Lambda Cloud instance ce7894ed (one GH200), active 21:03:33Z and terminated 21:15:43Z on 2026-11-10: about 12.2 min,
about $0.46 at the registration's $2.29/h (an estimate; the instance and the two times are the rental's records, not
in the run files, and the bill is in neither). The registration budgeted about 60 min of wall and $2.29, with a deadline
of launch + 75 min (22:18:33Z, the driver's own) and a guardian hard cap of launch + 90 min. The VM's setup ran
21:05:23Z to 21:08:00Z (7 preflight checks passed) and the driver 21:08:01Z to 21:13:35Z. Board `588570`
(`PREFLIGHT.txt`), new to the study. The plan's sha256 4a83311b... is on the ledger's `plan:` line and equals the
registered plan's; the run commit e7786f4e is in `commit.txt` and on the ledger's `driver start` line. This directory
is the branch `run-gh200-pc2-20261110t2103`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and
push log. Driver 580.105.08, persistence on, power.max_limit 900 W read before the limit was set.

## The power limit and the locks

The prelude set 700 W and read it back at the first poll: enforced.power.limit 700.00 W and the `-q -d POWER` GPU
scope's Current Power Limit 700.00 W (Requested 700.00 W); `power-set.txt` reads the same.
`gh200-driver/c2-power.env` reads `C2_POWER_W=700`, and every `clocks:` line between units reads 700.00 W. The prelude
probed 1710 MHz (`LOCK_PROBE_1710=held`) and locked the session at 1410 MHz (`LOCK_PROBE_1410=held`,
`gh200-driver/locks.env`). The perturb unit and every stamps unit locked 1710 MHz and read it back before and after the
tool (`gh200-driver/pc-locks.env`: pt, sd, ss, sn, sw, sd3 and sr, every line before and after equal to 1710). The
`clocks:` lines between units read 1980 MHz because each unit sets its own lock and the reset on exit restores the
session.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`)

| unit | step | label | start to end (UTC) | min | exit | result |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, lock 1410, probe 1710 | 21:08:01 to 21:08:06 | 0 | 0 | both locks held |
| 2 | tilecensus | pc2: 5 tiles on Mixtral-8x7B and Phi-3.5-MoE (compile only) | 21:08:07 to 21:08:20 | 0 | 0 | 20 configs, 0 path mismatches |
| 3 | regcheck | rcp (probe=pc) | 21:08:22 to 21:08:54 | 1 | 0 | sd, ss, sn, sw, sd3, sr PASS |
| 4 | perturb | pt at 1710 (gates sd, ss, sn, sw, sd3, sr) | 21:08:58 to 21:10:39 | 2 | 1 | timing FAIL on sd, ss, sn, sw, sr; sd3 PASS; every hard leg PASS |
| 5 | stamps | sd (page pc2-d, 1710) | 21:10:46 to 21:11:03 | 0 | 0 | PERTURBED |
| 6 | stamps | ss (page pc2-s, 1710) | 21:11:12 to 21:11:29 | 0 | 0 | PERTURBED |
| 7 | stamps | sn (page pc2-n, 1710) | 21:11:38 to 21:12:01 | 0 | 0 | PERTURBED |
| 8 | stamps | sw (page pc2-w, 1710) | 21:12:10 to 21:12:28 | 0 | 0 | PERTURBED |
| 9 | stamps | sd3 (page pc2-d3, 1710) | 21:12:41 to 21:12:56 | 0 | 0 | CLEAN |
| 10 | stamps | sr (page pc2-r, 1710) | 21:13:08 to 21:13:23 | 0 | 0 | PERTURBED |

10 units in about 5.6 min against 35 estimated; none was dropped, skipped or refused, and the driver exits 0. The
regcheck read every stamped variant at its plain kernel's CTAs per SM (16x64x128 s4 3, s3 5; 64x128x64 s3 3; both
32x tiles 4). The perturb unit's exit 1 is a timing FAIL alone (`gh200-driver/instr-gate.env`, median and worst pair:
sd 1.27% and 7.18%, ss 2.74% and 6.76%, sn 0.92% and 4.20%, sw 0.60% and 5.67%, sr 1.79% and 6.78%; sd3 PASS at 0.38%
and 1.02%), so those pages run PERTURBED (C3's OD-1, owner decision 5) and the reader applies C3's per-cell NOT-READ
bound (overhead median 10%, worst pair 20%).

## What the reader reads (`scripts/scoring/pc2/pc2.read.json`)

| reading | outcome | numbers |
|---|---|---|
| commit | checked | e7786f4e holds the registration, pages, shapes, plan and every code pin (72 files) |
| board | valid | `C2_POWER_W=700`, the 1710 lock probe held |
| census | registered occupancy read | 16x64x128 s4 3, s3 5; 64x128x64 3; 32x64x128 4; 32x128x64 4 CTAs per SM |
| PC2-R0 | every unit valid; sd3 CLEAN, the rest PERTURBED | 174, 120, 156, 174, 18, 18 launches (sd, ss, sn, sw, sd3, sr); EXACT TIME APPLIED and a 32 ns timer on every unit; every stamped compile at its registered occupancy |
| cell GEMMs | 138 of 220 READ | 81 NOT READ on the per-call spread (3%, 5% at G <= 8), 1 on the overhead bound alone (D-G4 w2, worst pair -47.8%; D-G4 w1 fails both, counted under the spread) |
| PC2-R1 | 16x64x128 s4 | n_if 3 (Q_B 48 KB); X_sat 3782 GB/s (READ, 12 members); r 49.3 / 44.9 / 27.4 / 9.9 GB/s at G 8 / 32 / 128 / 384; kappa 1.003 to 1.196; c0 at G 8 2.72 us (READ), at G 128 NOT IDENTIFIED, so the L table is NOT IDENTIFIED |
| PC2-R2 | short slabs | c0 on 64x128x64 at G 8 2.40 us (READ); G 32, 128, 256 and every short-rung-only ladder NOT IDENTIFIED |
| PC2-R3 | both 4-per-SM tiles | X_sat 3726 (32x64x128) and 3650 GB/s (32x128x64); G 528 SATURATED (1.026, 1.022); kappa 528 / 396 MOVED (0.932, 0.923); v6d term 3 DOES NOT HOLD on both |
| PC2-R4 | naive path | twin t ratios READ on 10 of 22 GEMMs (0.93 to 1.02); pair t ratios READ on 2 of 8 (0.96, 0.98); PLAIN on all 52 |
| PC2-R5 | replicate | R-D-G8 r -1.3%, t +1.8%; R-D-G128 r 0.0%, t +1.3%; R-S-G128-F512 NOT READ |
| PC2-R6 | printed | clock at start, per-unit medians 0.98 to 1.03, 9 launches flagged under 0.97 (all sw) |

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt` (`summary.new` is its working copy, equal), `logs/driver-*.log`:
  the ledger, the per-unit exits and the consoles; `gh200-driver/locks.env`, `a2-lock.env`, `c2-power.env`,
  `pc-locks.env`: the lock and power records the reader reads; `instr-regcheck.env`, `instr-gate.env`: the regcheck
  and perturbation gate lines; `instr-variants.json`, `pc-variants.json`: the stamped variants; `lock-plan.txt`,
  `supported-clocks.txt`, `supported-locks.txt`: the clock table
- `power-at-start.txt`, `power-set.txt`: the `-q -d POWER` snapshots
- `results/2026-11-10-nvidia_gh200_480gb-instr-pc-<unit>/`: the stamp pages (`stamps.json`, the per-launch stamp arrays
  under `stamps/`, each config's PTX, SASS and Triton cache); `unit-dir-*.txt` names each unit's directory on the VM
- `results/2026-11-10-nvidia_gh200_480gb-perturb-pt/` and `-regcheck-rcp/`: the perturbation gate (plain medians and
  their 8 bursts, per-cell overheads, `gate.env`) and the regcheck (`regcheck.env`, `regcheck.json`)
- `results/2026-11-10-nvidia_gh200_480gb-tile-census-mixtral-8x7b-pc2/tile-census.json`: PC2's own tile census
- `census.json` and `census.profiles/`: the preflight's GEMM census
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session); `logs/` holds every per-unit log

No page names or loads Mixtral-8x22B, Qwen2-57B-A14B or Granite-3.0-1B-A400M, and no launch has either GEMM's (N, K) of
any of them: the models and shapes the pages name are Mixtral-8x7B, JetMoE-8B, Phi-3.5-MoE, Granite-3.0-3B-A800M,
Qwen1.5-MoE-A2.7B, deepseek-v2-lite and the registered synthetic shapes (`pcs-`, `pcs2-`). `setup_vm.log` lists old
run branch names from the clone (one of them names 8x22b). The card's UUID is in `PREFLIGHT.json`, the census files and
`setup_vm.log`, as in every published session (the files are the run branch's, byte for byte); prose here names the
board by its hash. The Triton caches' compiled `cuda_utils*.so` files (9) are left out, as in every earlier session.
Every other file of the run branch is here, byte for byte (3,294 files against SHA256SUMS, which lists 3,303).
