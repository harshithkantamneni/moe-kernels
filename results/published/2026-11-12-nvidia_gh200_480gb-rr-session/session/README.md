# Rental RR: one NVIDIA GH200 480GB, 10 units of the RR plan (2026-11-11-rr-gh200, MEASUREMENT: certifies nothing)

One rental ran `scripts/plans/rentalRR-2026-11.plan` unattended through RR's driver (`scripts/gh200_rr_session.sh`,
PC2's pinned driver plus the lines marked RR), started by `scripts/vm_run_rr.sh`. It set 700 W and the 1410 MHz
session lock, took RR's compile-only tile census over 128x128x64 w8 s3 and 64x128x64 w8 s3, ran the regcheck and one
perturbation gate over the three stamps units, then, at the 1410 MHz lock, took per-CTA start and end stamps through
the unchanged instrumented copy (`moe/instrumented/fused_moe_instr.py`, the "ends" level, driven by
`scripts/instr_probe_rr.py`) and the plain kernel's counted w1 DRAM bytes (`scripts/dram_counter_route_rr.py`, family
rr-counter) on the same registered pages: rr-128 (40 cells at 128x128x64) and rr-64 (40 cells at 64x128x64), series A
(E 8, H 4096, P 20 to 224, m_e 2, 4, 8 and a skew; P 224 is Mixtral-8x7B's shape, run by that name), series B (K 8192)
and series C (E 64, k 8). The replicate took Phi-3.5-MoE's 128g1 stamps (sr); its counted bytes (br) refused. The
registration (`docs/registered/2026-11-11-rr-gh200`), its reader and the plan were committed before any page at
51d33d9b, which is also the run commit.

RR is a MEASUREMENT rental. Every reading is DESCRIPTIVE; it registers no model form, no constant, no bar and no gate,
and it moves no registered verdict. Every reading below is the registered reader's, run as registered from the
repository root with no post-page fix:
`PYTHONHASHSEED=0 .venv/bin/python scripts/scoring/rr/read_rr.py --rr results/published/2026-11-12-nvidia_gh200_480gb-rr-session --out-dir scripts/scoring/rr`.
Its output is `scripts/scoring/rr/rr.read.{json,txt}`. Read on the fetched tree and on this directory, the two outputs
differ only in the tree's path (the `rr` field and the `pages:` line); the committed files are this directory's, and
`scripts/scoring/rescore.py rr` re-reads it to them. The notes are in `docs/registered/README.md` ("**Read
2026-11-12**" under rental RR) and `docs/FINDINGS.md` (Rental RR).

## Where it came from

Lambda Cloud instance edf829f9 (one GH200), active 01:50:22Z and terminated 02:16:47Z on 2026-11-12: about 26.4 min,
about $1.01 at the registration's $2.29/h (an estimate; the instance and the two times are the rental's records, not
in the run files, and the bill is in neither). The registration budgeted about 68 min of wall and $2.60, with a deadline
of launch + 85 min (03:15:22Z, the driver's own) and a guardian hard cap of launch + 100 min. The VM's setup ran
01:52:23Z to 01:54:51Z (7 preflight checks passed) and the driver 01:54:53Z to 02:14:59Z. Board `73996e`
(`PREFLIGHT.txt`), new to the study. The plan's sha256 ce4d32e5... is on the ledger's `plan:` line and equals the
registered plan's; the run commit 51d33d9b is in `commit.txt` and on the ledger's `driver start` line. This directory
is the branch `run-gh200-rr-20261112t0150`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and
push log. Driver 580.105.08, persistence on, power.max_limit 900 W read before the limit was set.

## The power limit and the locks

The prelude set 700 W and read it back at the first poll: enforced.power.limit 700.00 W and the `-q -d POWER` GPU
scope's Current Power Limit 700.00 W (Requested 700.00 W); `power-set.txt` reads the same.
`gh200-driver/c2-power.env` reads `C2_POWER_W=700`, and every `clocks:` line between units reads 700.00 W. The prelude
locked the session at 1410 MHz (`LOCK_PROBE_1410=held`, `gh200-driver/locks.env`). The perturb unit, every stamps unit
and both counter units locked 1410 MHz and read it back before and after the tool (`gh200-driver/pc-locks.env`: pt,
s128, c128, s64, c64 and sr, every line before and after equal to 1410). The bytes unit br read the lock back before
its page only (the pinned bytes step has no read-back after its tool), and wrote no page. The `clocks:` lines between
units read 1980 MHz because each unit sets its own lock and the reset on exit restores the session.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`)

| unit | step | label | start to end (UTC) | min | exit | result |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, lock 1410 | 01:54:53 to 01:54:55 | 0 | 0 | the lock held |
| 2 | tilecensus | rr: 2 tiles on Mixtral-8x7B and Phi-3.5-MoE (compile only) | 01:54:57 to 01:55:07 | 0 | 0 | 8 configs, 0 path mismatches |
| 3 | regcheck | rcp (probe=pc) | 01:55:09 to 01:55:30 | 0 | 0 | s128, s64, sr PASS |
| 4 | perturb | pt at 1410 (gates s128, s64, sr) | 01:55:33 to 01:57:03 | 2 | 1 | timing FAIL on all three; every hard leg PASS |
| 5 | stamps | s128 (page rr-128) | 01:57:07 to 01:57:27 | 0 | 0 | PERTURBED |
| 6 | decodebytes | c128 (page rr-128, kernels=rr) | 01:57:35 to 02:05:30 | 8 | 0 | 40 cells counted |
| 7 | stamps | s64 (page rr-64) | 02:05:35 to 02:05:54 | 0 | 0 | PERTURBED |
| 8 | decodebytes | c64 (page rr-64, kernels=rr) | 02:06:04 to 02:13:50 | 8 | 0 | 40 cells counted |
| 9 | stamps | sr (Phi-3.5-MoE 128g1, NATIVE) | 02:13:57 to 02:14:12 | 0 | 0 | PERTURBED |
| 10 | bytes | br (Phi-3.5-MoE 128g1, arms native) | 02:14:21 to 02:14:51 | 1 | 2 | REFUSED before any launch, no page |

10 units in about 20.1 min against 43 estimated; none was dropped or skipped, and the driver exits 3 on br's exit 2.
The regcheck read every stamped variant at its plain kernel's CTAs per SM (128x128x64 2, 64x128x64 3). The perturb
unit's exit 1 is a timing FAIL alone (`gh200-driver/instr-gate.env`, median and worst pair against C3's 1% and 2%:
s128 2.11% and 6.15%, s64 0.998% and 2.90% (its worst pair only), sr 1.60% and 2.87%), so those pages run PERTURBED (C3's OD-1) and the
reader applies C3's per-cell NOT-READ bound.

**Why br refused.** Its plan line (`scripts/scoring/rr/rrcommon.py:502`, the registered PLAN's br row) runs the counter
route with `--arms native` on `docs/registered/2026-10-14-tileB-hist/phi-3.5-moe-128g1.json`, whose every cell names
its own arms `[native, shared]`. The route builds its plan from the page (`scripts/dram_counter_route.py:5595`,
`private_weight_reference.histogram_plan_cells`), and `cell_arms` (`scripts/private_weight_reference.py:8094-8100`)
refuses a cell that asks an arm the run does not run. The console (`logs/r3c-phi-3.5-moe-br-g1-lock1410.log`) holds
the one line `REFUSE: cell PT n=1 asks arms ['shared'] that the run's --arms native does not run`, returned as exit 2
(`dram_counter_route.py:8037-8039`, `moe/bench/exit_codes.py:98` REFUSED) after 5 s; the bytes step logs `lock block
exit 2` (`scripts/gh200_rr_session.sh:1104`) and `no page written`. PC's and B3's byte units on the same page ran
`arms=native,shared` (`rentalPC-2026-11.plan:34`, `rentalB3-tile-2026-11.plan:62`). This is a defect of RR's build,
in the registered plan line, not the environment (the census on the same unit passed CEN1 to CEN3 and the lock held)
and not a registered refusal (RR-R4 expected br's NATIVE counted bytes).

## What the reader reads (`scripts/scoring/rr/rr.read.json`)

| reading | outcome | numbers |
|---|---|---|
| commit | checked | 51d33d9b holds the registration, pages, shapes, plan and every code pin (79 files) |
| board | valid | `C2_POWER_W=700`, the 1410 lock probe held |
| census | registered occupancy read | 128x128x64 2, 64x128x64 3 CTAs per SM |
| RR-R0 | every stamps and counter unit valid; the stamps units PERTURBED | 240, 240, 72 launches (s128, s64, sr); EXACT TIME APPLIED and a 32 ns timer on every unit |
| RR-R1 | counted miss fraction m | 38 of 40 cells READ on 128x128x64, 32 of 40 on 64x128x64; every NOT READ on the 3% per-call spread |
| RR-R2 | the four forms on cells READ on both units | 69 cells (56 fresh): F2 the lowest rms, 0.024 on fresh cells, F1 0.030, F3 0.052, F4 0.060; refit 45.0 and 47.5 MiB |
| RR-R3 | wave-start spread, numbers only | pooled w_inf 0.586 (k_90 5) at 128x128x64, 0.897 (k_90 8) at 64x128x64 |
| RR-R4 | the Phi-3.5 replicate | NOT IDENTIFIED: no cell read on both (br wrote no counter page) |

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt` (`summary.new` is its working copy, equal), `logs/driver-*.log`:
  the ledger, the per-unit exits and the consoles; `gh200-driver/locks.env`, `a2-lock.env`, `c2-power.env`,
  `pc-locks.env`: the lock and power records the reader reads; `instr-regcheck.env`, `instr-gate.env`: the regcheck
  and perturbation gate lines; `instr-variants.json`, `pc-variants.json`: the stamped variants; `lock-plan.txt`,
  `supported-clocks.txt`, `supported-locks.txt`: the clock table
- `power-at-start.txt`, `power-set.txt`: the `-q -d POWER` snapshots
- `results/2026-11-12-nvidia_gh200_480gb-instr-pc-s128/`, `-instr-pc-s64/`, `-instr-phi-3.5-moe-sr/`: the stamp pages
  (`stamps.json`, the per-launch stamp arrays under `stamps/`, each config's PTX, SASS and Triton cache);
  `unit-dir-*.txt` names each unit's directory on the VM
- `results/2026-11-12-nvidia_gh200_480gb-mixtral-8x7b-c128-r3-counters/` and `-c64-r3-counters/`: the counter pages
  (`lock1410/r3d.json`) and their ncu profiles; `counters-dir-*.txt` names each counter unit's directory on the VM
  (br's names a directory that was never written)
- `results/2026-11-12-nvidia_gh200_480gb-perturb-pt/` and `-regcheck-rcp/`: the perturbation gate (plain medians and
  their bursts, per-cell overheads, `gate.env`) and the regcheck
- `results/2026-11-12-nvidia_gh200_480gb-tile-census-mixtral-8x7b-rr/tile-census.json`: RR's own tile census
- `census.json`, `census.profiles/`, `census-phi-3.5-moe.json`, `census-phi-3.5-moe.profiles/`: the preflight's GEMM
  census and br's own census (CEN1 to CEN3 PASS)
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session); `logs/` holds every per-unit log

No page names or loads Mixtral-8x22B, Qwen2-57B-A14B or Granite-3.0-1B-A400M: the shapes the pages name are the
registered synthetic shapes (`rrs-`), Mixtral-8x7B and Phi-3.5-MoE. Series C's skew is a count vector taken from a
seen Qwen2-57B counter page, rescaled (the registration's literal); no Qwen2-57B weight or shape is used.
`setup_vm.log` lists old run branch names from the clone (one of them names 8x22b). The card's UUID is in
`PREFLIGHT.json`, the census files and `setup_vm.log`, as in every published session (the files are the run branch's,
byte for byte); prose here names the board by its hash. The Triton caches' compiled `cuda_utils*.so` files (7) are left
out, as in every earlier session. Every other file of the run branch is here, byte for byte (1,816 files against
SHA256SUMS, which lists 1,823).
