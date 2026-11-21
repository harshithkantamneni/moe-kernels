# Decode settling rental DS: one NVIDIA GH200 480GB, 9 units of the DS plan (addendum 4, a measurement registration on SEEN models, DESCRIPTIVE)

One rental ran `scripts/plans/rentalDS-settle-d4-2026-11.plan` unattended through the D3 driver
(`scripts/gh200_decode3_session.sh`, its `kernels=wide` key). It set the GPU power limit to 700 W, ran calibrate's
ruler, took DS's own tile census (compile only) of vLLM's four decode tiles on the six DS models, then took one wide
decode counter page per model at the 1710 MHz lock: JetMoE-8B, Phi-3.5-MoE, Mixtral-8x7B, Granite-3.0-3B-A800M,
Qwen1.5-MoE-A2.7B and OLMoE-1B-7B, every one a model earlier decode pages had already timed (SEEN). The pages are the
registered `docs/registered/2026-11-04-decode4-pages/*-ds.json`. Addendum 4 (`docs/registered/2026-11-04-decode-addendum4-gh200`),
the plan and the reader were committed before any DS page at 5410ad00, which is also the run commit. The addendum
registers no model form, no constant and no gate: every reading is DESCRIPTIVE and changes no registered verdict.

Every reading below is the registered reader's, run as registered from the repository root with no post-page fix:
`PYTHONHASHSEED=0 .venv/bin/python scripts/scoring/decode/read_ds.py results/published/2026-11-04-nvidia_gh200_480gb-decodeDS-session/results <out>`.
Its output is `scripts/scoring/decode/ds.readings.{json,txt}`, read from this directory (the registration names no
location; the tool line says only `<out>`). Read on the fetched tree and on this directory, the `.txt` is
byte-identical and the JSON differs only in two paths (the power file's and DS's census file's), and
`scripts/scoring/rescore.py decodeDS` re-reads this directory to the committed files. The notes are in
`docs/registered/README.md` ("**Read 2026-11-04**" on this directory) and `docs/FINDINGS.md` (Decode settling rental
DS).

## Where it came from

Lambda Cloud instance 3565dbb9 (one GH200, us-east-3), launched 21:24:15Z and terminated 21:44:16Z on 2026-11-04:
about 20 min, about $0.76 at the registration's $2.29/h (an estimate; the instance, the region and the two times are
the rental's records, not in the run files, and the bill is in neither). The VM's setup ran 21:25:57Z to 21:27:55Z
(7 preflight checks passed) and the driver 21:27:56Z to 21:42:25Z, against a deadline of 23:04:15Z (launch + 100
min). Board `cce89f` (`PREFLIGHT.txt`), new to the study. The plan's sha256 fec3c67f... is in `gh200-driver/status`
and equals the registered plan's; the run commit 5410ad00 is in `commit.txt`. This directory is the branch
`run-gh200-decodeds-20261104t2124`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its manifest and push log.
Driver 580.105.08, persistence on.

## The power limit and the lock

The reader reads nothing unless `power-set.txt` reads a GPU Current Power Limit of 700 W within 1 W. It does: GPU
Power Readings, Current Power Limit 700.00 W; the module scope reads 1000 W. The prelude recorded persistence Enabled
and power.limit 700.00 W, and the driver's `clocks:` line after every unit reads 700.00 W.

Every counter page sits under `lock1710/`: the driver's `locked_run` sets the 1710 MHz lock and reads it back from
clocks.sm (two tries) before the counter route starts, and exits 2 with no page otherwise; all six units exit 0 and
no log holds a "not in force" line. DS has no timed page, so G4 does not apply. Every page's manifest carries its
registered page's sha256, its grid gate reads PASS and it refused no cell.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, label | start to end (UTC) | min | exit | page |
|---|---|---|---|---:|---:|---|
| 1 | prelude | 700 W, persistence, base tag | 21:27:56 to 21:27:56 | 0 | 0 | |
| 2 | calibrate | mixtral-8x7b dcal (the ruler, no lock) | 21:27:58 to 21:28:23 | 0 | 0 | triad 3726.1 GB/s |
| 3 | tilecensus | ds: four decode tiles on six models (compile only) | 21:28:25 to 21:28:40 | 0 | 0 | 48 configs, 0 path mismatches |
| 4 | decodebytes | jetmoe-8b ds (wide) | 21:28:42 to 21:31:03 | 2 | 0 | 15 cells |
| 5 | decodebytes | phi-3.5-moe ds (wide) | 21:31:05 to 21:34:53 | 4 | 0 | 16 cells |
| 6 | decodebytes | mixtral-8x7b ds (wide) | 21:34:56 to 21:38:32 | 4 | 0 | 14 cells |
| 7 | decodebytes | granite-3.0-3b-a800m ds (wide) | 21:38:35 to 21:39:43 | 1 | 0 | 12 cells |
| 8 | decodebytes | qwen1.5-moe-a2.7b ds (wide) | 21:39:45 to 21:41:17 | 2 | 0 | 13 cells |
| 9 | decodebytes | olmoe-1b-7b ds (wide) | 21:41:20 to 21:42:23 | 1 | 0 | 8 cells |

9 units in about 14.5 min against 49 estimated; every unit is nodrop, none was skipped, and all 9 exit 0 (driver exit
0). The 78 page cells are the registered pages' 69 NATIVE cells plus 9 DECL arms (JetMoE and Phi-3.5 declaring E 32,
Mixtral declaring E 16).

## What the reader reads (`scripts/scoring/decode/ds.readings.txt`)

| reading | outcome | numbers |
|---|---|---|
| power | READ | 700.0 W on the GPU scope of `power-set.txt` |
| PAGE | READ on all six | sha256, grid gate, wide filter and versions as CAL2's (python 3.12.16, torch 2.13.0, triton 3.7.1, vllm 0.27.1) |
| census gate (DS's own) | READ (COMPLETE) | every decode tile READ at the occupancy S uses |
| census cross-check (against CAL2's) | SAME | 48 entries compared, no difference; so no per-CTA STOP |
| refused rows | 0 | |
| LADDERS | READ on every tile | 16x64x128w4s4 30 ladders (13 mixed, 3 sub-wave); 32x64x128w4s3 12 (5, 2); 32x128x64w4s3 12 (3, 2); 64x128x64w4s3 14 (5, 3) |
| W1RATE | READ on every tile | 3897.9, 3883.2, 3873.2 and 3787.3 GB/s (ratio to v4d 0.918, 0.921, 0.924, 0.945) |
| SUBWAVE r_cta | READ on every tile | 53.8, 24.4, 37.7 and 46.9 GB/s |
| DRAIN r1 | READ on every tile | 90.6, 46.1, 67.2 and 103.4 GB/s |
| CALLS | READ on all six models | the summary lines over 69 NATIVE rows: silu_and_mul rms 0.46 us, moe_sum rms 0.35 us |
| ALIGN | READ; DECL TRANSFERS | small-batch E 8 to 64: 4.75 to 11.63 us; block-scan E 40 8.70 us, E 64 8.13 us |
| SLOPE | READ | 0.527 us per loop iteration over 74 rows, iterations 1 to 24; DIFFERS FROM 0 (t 71.0) |
| NOT IDENTIFIED | none | every reading READ |

The tiles are in the order 16x64x128w4s4, 32x64x128w4s3, 32x128x64w4s3, 64x128x64w4s3. Every reading is DESCRIPTIVE
on SEEN models: none is a model term, a bar or a STOP. The reader's last line restates the registration: the decode
model form for the new blind will be chosen after DS, from DS's and CAL2's pages, and frozen and registered before any
page of Mixtral-8x22B, Qwen2-57B-A14B or Granite-3.0-1B-A400M.

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and the consoles;
  `gh200-driver/locks.env`: the registered lock plan; `lock-plan.txt`, `supported-clocks.txt`,
  `supported-locks.txt`: the clock table
- `power-at-start.txt`, `power-before-calibrate.txt`, `power-set.txt`: the -q -d POWER snapshots (the last is the one
  the reader reads)
- `results/2026-11-04-nvidia_gh200_480gb-<model>-ds-r3-counters/lock1710/`: the six wide decode counter pages
  (`r3d.json`, and `r3d-profile/` with the ncu report, its CSV, manifest, plan and log); `counters-dir-<model>-ds.txt`
  names each page's directory on the VM
- `results/2026-11-04-nvidia_gh200_480gb-tile-census-mixtral-8x7b-ds/tile-census.json`: DS's own tile census, the one
  the census gate reads
- `results/calibration/` and `measured_nvidia_gh200_480gb.yaml`: calibrate's ruler
- `census.json`, `census.profiles/`: the preflight's GEMM census written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console (at the run
  branch's `vm/` top level, here beside the rest of the session); `logs/` holds every per-unit log

**Pages.** Every counter page reads its registered page file in `docs/registered/2026-11-04-decode4-pages/`, with its
sha256 on the page's manifest. The pages are SYNTHETIC; no trace-derived count is on the VM or in this directory.
This session has no timed page.

The card's UUID is in `PREFLIGHT.json`, the census files and `setup_vm.log`, as in every published session (the files
are the run branch's, byte for byte); prose here names the board by its hash. The Triton cache's compiled `cuda_utils*.so` file (1) is left out,
as in every earlier session. Every other file of the run branch is here, byte for byte (121 files against
SHA256SUMS, which lists 122).
