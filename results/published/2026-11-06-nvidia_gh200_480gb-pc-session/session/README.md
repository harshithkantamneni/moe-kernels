# Rental PC: one NVIDIA GH200 480GB, 20 units of the PC plan (2026-11-06-pc-gh200, MEASUREMENT: certifies nothing)

One rental ran `scripts/plans/rentalPC-2026-11.plan` unattended through PC's driver (`scripts/gh200_pc_session.sh`,
C3's pinned driver plus the lines marked PC: probe=pc, unit locks read back before and after, DS's decodebytes, the
regcheck drop rule, G-BAND scoped to the Phi group). It set 700 W and the 1410 MHz session lock, probed the 1710 MHz
lock, took PC's compile-only tile census, then per-CTA start and end stamps through the unchanged instrumented copy
(`moe/instrumented/fused_moe_instr.py`, the "ends" level) on PC's registered pages: block A (synthetic decode shapes,
Mixtral-8x7B, its tp2/tp4/tp8 shards and JetMoE-8B at 64x128x64 and its stage, BLOCK_K and warps variants, 1710
MHz), block C (the 1410 MHz twins), block R (a replicate of three A rungs) and Phi-3.5-MoE's w1 stamps at 128x128
(E1, C3's probe) and 64x128 (E2, PC's probe), each behind its regcheck and perturbation gate. Around them it took the
two f128 sag pages and G-BAND, three Phi-3.5-MoE histogram byte pages and one Mixtral-8x7B decode counter page. Block
D (16x64x128) was refused by its regcheck. The registration (`docs/registered/2026-11-06-pc-gh200`), its reader and
the plan were committed before any page at c9c92f1e, which is also the run commit.

PC is a MEASUREMENT rental. Every reading is DESCRIPTIVE; it registers no model form, no constant, no bar and no gate,
and it moves no registered verdict. Every reading below is the registered reader's, run as registered from the
repository root with no post-page fix:
`PYTHONHASHSEED=0 .venv/bin/python scripts/scoring/pc/read_pc.py --pc results/published/2026-11-06-nvidia_gh200_480gb-pc-session --out-dir scripts/scoring/pc`.
Its output is `scripts/scoring/pc/pc.read.{json,txt}`. Read on the fetched tree and on this directory, the two
outputs differ only in the tree's path (the `pc` field and the `pages:` line); the committed files are this
directory's, and `scripts/scoring/rescore.py pc` re-reads it to them. The notes are in `docs/registered/README.md`
("**Read 2026-11-06**" under rental PC) and `docs/FINDINGS.md` (Rental PC).

## Where it came from

Lambda Cloud instance d6459e41 (one GH200), active 18:04:50Z and terminated 18:45:12Z on 2026-11-06: about 40.4 min,
about $1.54 at the registration's $2.29/h (an estimate; the instance and the two times are the rental's records, not
in the run files, and the bill is in neither). The registration budgeted about 109 min of wall and $4.16, with a
deadline of launch + 125 min (20:09:50Z, the driver's own) and a guardian hard cap of launch + 140 min. The VM's
setup ran 18:06:41Z to 18:08:36Z (7 preflight checks passed) and the driver 18:08:37Z to 18:43:00Z. Board `c9f689`
(`PREFLIGHT.txt`), new to the study. The plan's sha256 5c19aa93... is on the ledger's `plan:` line and equals the
registered plan's; the run commit c9c92f1e is in `commit.txt` and on the ledger's `driver start` line. This directory
is the branch `run-gh200-pc-20261106t1804`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and
push log. Driver 580.105.08, persistence on, power.max_limit 900 W read before the limit was set.

## The power limit and the locks

The prelude set 700 W and read it back at the first poll: enforced.power.limit 700.00 W and the `-q -d POWER` GPU
scope's Current Power Limit 700.00 W (Requested 700.00 W); `power-set.txt` reads the same.
`gh200-driver/c2-power.env` reads `C2_POWER_W=700`, and every `clocks:` line between units reads 700.00 W. The
prelude probed 1710 MHz (`LOCK_PROBE_1710=held`) and locked the session at 1410 MHz (`LOCK_PROBE_1410=held`,
`gh200-driver/locks.env`). Every probe=pc unit and the counter unit locked their own clock and read it back before
and after the tool (`gh200-driver/pc-locks.env`): 1710 for ptp, sa, cnt and sr; 1410 for ptc, sc, pt3 and se2; every
line reads before and after equal to the lock. The `clocks:` lines between units read 1980 MHz because each unit
sets its own lock and the reset on exit restores the session.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`)

| unit | step | label | start to end (UTC) | min | exit | result |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, lock 1410, probe 1710 | 18:08:37 to 18:08:42 | 0 | 0 | both locks held |
| 2 | tilecensus | pc: 10 tiles on Mixtral-8x7B and Phi-3.5-MoE (compile only) | 18:08:44 to 18:09:02 | 0 | 0 | 40 configs, 0 path mismatches |
| 3 | regcheck | rcp (probe=pc) | 18:09:04 to 18:09:46 | 1 | 1 | sa, sc, se2, sr PASS; sd FAIL |
| 4 | perturb | ptp at 1710 (gates sa, sd, sr) | 18:09:51 to 18:11:20 | 1 | 1 | timing FAIL on all three; hard legs PASS but sd's |
| 5 | stamps | sa (page pc-a, 1710) | 18:11:28 to 18:12:04 | 1 | 0 | PERTURBED |
| 6 | perturb | ptc at 1410 (gates sc, se2) | 18:12:16 to 18:12:54 | 1 | 1 | sc timing FAIL; se2 PASS |
| 7 | stamps | sc (page pc-c, 1410) | 18:13:04 to 18:13:23 | 0 | 0 | PERTURBED |
| 8 | bytes | olmoe-1b-7b f128 (G 64) | 18:13:36 to 18:16:57 | 3 | 0 | page exit 3 |
| 9 | bytes | mixtral-8x7b f128 (G 64) | 18:17:08 to 18:24:14 | 7 | 0 | page exit 3 |
| 10 | gate | G-BAND | 18:24:26 to 18:24:27 | 0 | 0 | PASS (the fallback band [1352.19, 1410]) |
| 11 | regcheck | rc3 (C3's probe) | 18:24:38 to 18:24:49 | 0 | 0 | se1 PASS |
| 12 | perturb | pt3 at 1410 (gate se1) | 18:25:01 to 18:25:24 | 0 | 1 | timing FAIL; hard legs PASS |
| 13 | stamps | phi-3.5-moe se1 (128x128 G 1, E1) | 18:25:36 to 18:25:51 | 0 | 0 | PERTURBED |
| 14 | stamps | phi-3.5-moe se2 (64x128 G 1, E2) | 18:26:05 to 18:26:19 | 0 | 0 | CLEAN |
| 15 | bytes | phi-3.5-moe b128g1 | 18:26:32 to 18:32:08 | 6 | 0 | page exit 0, analyse exit 3 |
| 16 | bytes | phi-3.5-moe b64g1 | 18:32:21 to 18:35:18 | 3 | 0 | page exit 0, analyse exit 3 |
| 17 | stamps | sd (page pc-d, 1710) | 18:35:31 to 18:35:31 | 0 | 2 | REFUSED: gate hard legs FAIL |
| 18 | decodebytes | mixtral-8x7b cnt (1710, kernels wide) | 18:35:43 to 18:36:40 | 1 | 0 | V_ALIGN grid PASS, 4 cells, 40 GEMM launches |
| 19 | bytes | phi-3.5-moe b128g16 | 18:36:53 to 18:42:19 | 5 | 0 | page exit 0, analyse exit 3 |
| 20 | stamps | sr (page pc-r, 1710) | 18:42:32 to 18:42:47 | 0 | 0 | PERTURBED |

20 units in about 34.4 min against 84 estimated; none was dropped or skipped. The driver exits 3 because unit 17
exited 2 (refused); every other unit exits 0 or 1. Unit 3's exit 1 is the registered "a variant's occupancy FAIL"
case: `REGCHECK_u17_sd=FAIL 16-64-128-1-2-4-True-1: stamps=ends,evict_a=none,evict_b=none at 7 CTAs/SM against the
plain 8`, the stamped 16x64x128 w4 s2 variant (block D's s2 rungs) holds one CTA per SM fewer than the plain kernel.
That refuses the sd unit by its regcheck line and its gate's hard occupancy leg (`HARD_u17_sd=FAIL`); it is not the
drop rule (that fires only on an exit other than 0 or 1), so every other probe=pc unit ran. The perturb units' exit 1
is a timing FAIL alone (`gh200-driver/instr-gate.env`: sa worst 9.03%, sc 9.33%, sr 3.44%, se1 3.35%; se2 PASS at
0.86%), so those pages run PERTURBED (C3's OD-1, owner decision 5) and the reader applies C3's per-cell NOT-READ bound
(overhead median 10%, worst pair 20%). The f128 pages' exit 3 is R3's own analyse gates, as on C3's and B3's identical
units (OLMoE V4, V5, C1, C6; Mixtral V5, V7, C2, C6); G-BAND reads the pages' clocks, not those exits. The three Phi
byte pages wrote their pages (exit 0) and fail R3's attribution gate V1 in analyse (exit 3: "2 treads; the scorer
needs four for an OLS residual; arms ['native', 'shared'], not R3's ['native', 'shared', 'private']; 24 launches
attributed against the planned 144", and 1 tread and 12 of 72 on b64g1), as B3's and A2's histogram byte pages did;
PC reads those pages' counted w1 bytes descriptively, beside B3's.

## What the reader reads (`scripts/scoring/pc/pc.read.txt`)

| reading | outcome | numbers |
|---|---|---|
| commit | checked | c9c92f1e holds the registration, pages, shapes, priced file, plan and every code pin (68 files) |
| board | valid | `C2_POWER_W=700`, both lock probes held, G-BAND PASS (fallback band) |
| census | registered occupancy read | 64x128x64w4s3 3, 64x128x64w8s3 3, 16x64x128w4s4 3, 128x128x64w8s3 2 CTAs per SM |
| PC-R0 | sa, sc, sr, se1 PERTURBED; se2 CLEAN; sd REFUSED | 234, 54, 18, 72, 36 launches; sd none (regcheck FAIL, no lock line, no page) |
| **H1** | **NOT IDENTIFIED** | beta_f and the plateau (G 2 to 16) unread; beta_Q 0.911, omega 0.945, sigma 0.963, all in their support bands; the low-load clock twins that did read sit outside beta_f's support band (G 8 s2k64 0.48, G 32 0.45), so the unread input does not lean SUPPORTED; PC-R7's plateau READ (G 4 to 16) is a different reading |
| PC-R1 | read but 4 cells | r 36.2 / 35.8 / 33.9 GB/s at G 4 / 8 / 16; 23.0 at G 128; 8.6 to 8.7 at G 512 to 1024 |
| PC-R2 | G 128 READ; G 8 NOT IDENTIFIED | c0 4.70 us, r_inf 23.5 GB/s; G 8: 3 of 4 rungs read |
| PC-R3 | READ at G 8, 128, 384 | beta_Q 0.911 (G 8, BK 64), 0.466 and 0.400 (G 128), 0.350 (G 384); D16 unread |
| PC-R4 | READ but two twins | beta_f 0.447 (G 32), 0.236 (G 128), 0.211 (G 256), 0.018 (G 512), -0.018 (G 1024); C-G8 unread |
| PC-R5 | READ | omega 0.945 (G 8), 0.978 (G 128) |
| PC-R6 | READ at G 160 and 192 | sigma 0.960, 0.963; G 288 NOT IDENTIFIED (no CTA alone) |
| PC-R7 | plateau READ, saturation READ | r0 35.8 GB/s; X_sat 3612 GB/s (triad 3726); G_90 192; L_gpu 0.91 to 3.6 us |
| PC-R8 | CAL2 cell CARRIES | Mixtral T128 A4 against A-G128 t -0.8%; the curve prices it at -1.1% |
| PC-R9 | READ | flush off / on 1.008 at G 8 |
| PC-R10 | READ (E1, E2) | 128g1 miss fraction 0.69 to 0.76 (pooled 0.74); 64g1 0.22 to 0.52 (pooled 0.38) |
| PC-R11 | READ (a printed extra) | skewed means: life +5.3%, packing -12.3%, total -7.0% |
| PC-R12 | READ (T128 A4, A8) | counted DRAM over construction 1.001 to 1.016; D cells unread |
| PC-R13 | printed | per-unit medians 0.978 to 0.990; 41 launches flagged under 0.97 |
| PC-R14 | printed | G <= 64: one CTA per SM; G 128 on 120 SMs, at most 2 per SM |
| replicate | printed | +0.26%, -0.69%, -1.11% at G 8, 128, 512 |

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt` (`summary.new` is its working copy, equal), `logs/driver-*.log`:
  the ledger, the per-unit exits and the consoles; `gh200-driver/locks.env`, `a2-lock.env`, `c2-power.env`,
  `pc-locks.env`: the lock and power records the reader reads; `instr-regcheck.env`, `instr-gate.env`: the regcheck
  and perturbation gate lines; `instr-variants.json`, `pc-variants.json`: the stamped variants; `gate-band-1410.json`:
  G-BAND's record; `lock-plan.txt`, `supported-clocks.txt`, `supported-locks.txt`: the clock table
- `power-at-start.txt`, `power-set.txt`: the `-q -d POWER` snapshots
- `results/2026-11-06-nvidia_gh200_480gb-instr-<unit>/`: the stamp pages (`stamps.json`, the per-launch stamp arrays
  under `stamps/`, each config's PTX, SASS and Triton cache); `unit-dir-*.txt` names each unit's directory on the VM
- `results/2026-11-06-nvidia_gh200_480gb-perturb-<unit>/` and `-regcheck-<unit>/`: the perturbation gates (plain
  medians, per-cell overheads, `gate.env`) and the regchecks (`regcheck.env`, `regcheck.json`)
- `results/2026-11-06-nvidia_gh200_480gb-<model>-<label>-r3-counters/`: the two f128 sag pages and the three Phi byte
  pages at 1410 (`lock1410/`), and the Mixtral decode counter page at 1710 (`lock1710/r3d.json`, `r3d-profile/`);
  `counters-dir-<model>-<label>.txt` names each page's directory on the VM
- `results/2026-11-06-nvidia_gh200_480gb-tile-census-mixtral-8x7b-pc/tile-census.json`: PC's own tile census
- `census.json` and `census.profiles/`: the preflight's GEMM census; `census-olmoe-1b-7b.json`,
  `census-phi-3.5-moe.json` and their `.profiles/`: the GEMM censuses the byte units wrote
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session); `logs/` holds every per-unit log

No file names or loads Mixtral-8x22B, Qwen2-57B-A14B or Granite-3.0-1B-A400M: the models and shapes the pages name
are Mixtral-8x7B and its tp2, tp4 and tp8 shards, JetMoE-8B, Phi-3.5-MoE, OLMoE-1B-7B and the registered synthetic
shapes. The tile census log prints deepseek-v2-lite's name once, in a fixed NOT SCORED line, and `setup_vm.log` lists
old run branch names from the clone. The card's UUID is in `PREFLIGHT.json`, the census files and `setup_vm.log`, as
in every published session (the files are the run branch's, byte for byte); prose here names the board by its hash.
The Triton caches' compiled `cuda_utils*.so` files (18) are left out, as in every earlier session. Every other file of
the run branch is here, byte for byte (3,826 files against SHA256SUMS, which lists 3,844).
