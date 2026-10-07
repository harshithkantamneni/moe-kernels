# Rental 4: one NVIDIA GH200 480GB, 25 units of five scored registrations (docs/registered/README.md, 2026-10-06)

One rental ran `scripts/plans/rental4-2026-10.plan` unattended: the hardware rulers and RRZE's
gpu-benches (hw), the dead-CTA copies contrast on qwen2-57b-a14b-tp8 (A1) and olmoe-1b-7b (A2) at 9
and 15 declared copies (dead), the perturbation gate of the instrumented kernel copy (perturb) and
the nine stamps units it gates (stamps), and the seven OLMoE G = 64 counter pages of part B' at
BLOCK_K 32 / 64 / 128 and num_stages 2 / 3 / 4 / 6 / 8 (occlaw). The six registrations (one,
nativegates, is rental 5's and scores nothing here), their scorers and the plan were committed
before any page; the run commit is 85ef38c. Every verdict below is the scorers'
(`scripts/scoring/rental4/`, written and tested on synthetic pages before any page), run on this
directory with one post-page fix to reading code that changes no verdict (the refused stamps units
now read NOT RUN with their gate numbers); the full tables are in
`scripts/scoring/rental4/SCORES.md`, the scoring notes in `docs/registered/README.md` ("**Scored
2026-10-07**") and `docs/FINDINGS.md` (rental 4).

## Where it came from

Lambda Cloud instance 11fcccbb, gpu_1x_gh200, 2026-10-06 22:45Z to 2026-10-07 00:26Z: 1 h 41 min,
about $3.85. Board `bb7a34` (`PREFLIGHT.txt`), the board rental 3 ran on.
`scripts/gh200_model_session.sh --plan scripts/plans/rental4-2026-10.plan` ran at commit 85ef38c
(`commit.txt`); the plan's sha256 01a4cc6f... is in `gh200-driver/status`. This directory is the
branch `run-gh200-rental4-20261006t2245`; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log. Power limit 700 W, persistence on (`power-set.txt`). The 1710 lock was
held for every byte page and every timed page (`lock-plan.txt`, `clocks-after-*.txt`; every timed
page's worst cell clock 1710.0 MHz). The ruler (unit 2) reads triad 3725.9 GB/s.

## What ran (`gh200-driver/status`, `gh200-driver/summary.txt`)

| unit | step | model, design | start to end (UTC) | min | exit | pages (gates not PASS) |
|---|---|---|---|---:|---:|---|
| 1 | prelude | clocks, persistence, 700 W | 22:49:00 to 22:49:01 | 0 | 0 | |
| 2 | calibrate | the ruler (triad 3725.9 GB/s) | 22:49:02 to 22:49:27 | 0 | 0 | |
| 3 | rulers | torch read2d, copy, add, matmul at the 1710 lock; the L2 size | 22:49:29 to 22:49:39 | 0 | 0 | |
| 4 | gpubench | RRZE-HPC/gpu-benches 23e586dd: latency, stream, l2-cache | 22:49:40 to 22:59:29 | 10 | 0 | gpu-latency stopped at its 420 s timeout (rc 124), 91.5 MB |
| 5 | timed | qwen2-57b-a14b-tp8 G = 8, n 1..9, 9 copies (A1) | 22:59:30 to 23:15:39 | 16 | 0 | claim gates only |
| 6 | timed | qwen2-57b-a14b-tp8 G = 8, n 1..9, 15 copies (A1) | 23:15:41 to 23:31:50 | 16 | 0 | claim gates only |
| 7 | perturb | the gate of units 8 to 16 | 23:31:52 to 23:32:44 | 1 | 1 | FAIL on 9 of 9 variants |
| 8 to 16 | stamps | stf, stk64s4, stk32s4, stk128s4, stk64s8, sttail (8x22B), sttail (8x7B), stdead4, stdead2 | 23:32:48 to 23:33:12 | 0 | 2 | REFUSED: the gate reads FAIL; nothing launched |
| 17 to 23 | bytes | olmoe-1b-7b G = 64 at k64s4, k32s4, k128s4, k64s6, k64s2, k64s3, k64s8 | 23:33:15 to 23:51:36 | 3 each | 0 | none (each page exits 1 on claim C1) |
| 24 | timed | olmoe-1b-7b G = 8, n 1..9, 9 copies (A2) | 23:51:40 to 00:07:49 | 16 | 0 | claim gates only |
| 25 | timed | olmoe-1b-7b G = 8, n 1..9, 15 copies (A2) | 00:07:58 to 00:24:06 | 16 | 0 | claim gates only |

95 minutes from the prelude to the last unit against the plan's 144 minutes of units. No unit was
dropped and there is no automatic retake (a plan rental). The driver ends with exit 3 (unit 7's
exit 1 and the nine refusals). Every timed page passes V0 to V8; C1 reads UNKNOWN (no same-card byte
page at these declarations) and C2 PASS. Every byte page passes every validity gate, V10 on the
nvidia-smi bracket; each page's exit 1 is claim C1 (SHARED w1 reads off the group model).

**The perturbation gate (unit 7).** Every instrumented variant FAILED: median |plain / copy - 1|
1.45 to 9.51%, worst 2.22 to 16.68%, and in every variant at least one config's registers differ
between the plain kernel and the stamped copy (w2 at BK 64 s4: 55 against 45 or 48, 4 against 5
CTAs per SM). The all-off copy's SASS equals the plain kernel's on all 18 configs (the SASS leg
ran and passed), so the copy's launch path, constexprs, warps, stages and Triton build are the
plain kernel's; the difference is the stamps' own code, which ptxas allocates around (diagnosis in
`docs/FINDINGS.md`, rental 4, and `scripts/scoring/rental4/SCORES.md`). Units 8 to 16 were refused
in 0 minutes, as designed: their pages were never written.

**gpubench's latency sweep.** `gpu-latency` ran into its 420 s timeout at a 91.5 MB buffer (1.46 x
the 60 MiB L2), so the DRAM window (2.5 L and above) holds no cell and the DRAM latency is not
measured; the near and far L2 windows are covered (44 and 12 cells). The stream and l2-cache benches
completed.

## What it answered

Each verdict is the registered reading (rental 2's rule 3 over ALL and CLEAN; no page is dropped
by either view).

**dead, the dead-CTA copies contrast** (`results/gaps-*-{qwen2-57b-a14b-tp8,olmoe-1b-7b}-c{9,15}/`):
D, the candidate d = 0.995 ns (kappa 0.325), HOLDS on both contrasts; M, the current 1.333 ns,
is EXCLUDED on both. A1 measures Delta 22.93 us (D 20.73, z +1.79; M 27.77, z -5.38), A2 24.45 us
(D 23.69, z +0.58; M 31.74, z -8.10). FIXED is EXCLUDED, SLOT EXCLUDED on A2 (z +3.06), K0 not
excluded; D / SLOT and D / D2 are not separated by design. Per-GEMM d, printed: d_w2 1.10 ns
(A1), d_w1 0.95 ns (A2). The NATIVE null control passes on both (+0.77 and 0.00 us).

**occlaw, part B'** (`results/*-olmoe-1b-7b-k*s*-r3-counters/lock1710/r3c-g64.json`): UNDECIDED
with no survivor: MVA2, LK, PS, LITTLE and OCC are each FALSIFIED by at least one of the 12 scored
(page, GEMM) cells (best rms MVA2 2.56%, OCC 2.51% on its 8 BK 64 cells). The base k64s4 reads
c 346.2 / 344.0 cycles per k-step (w1 / w2) against PS's CAL 344.1. k64s2 recorded 5 / 5 CTAs per
SM against the registered 4 / 4 and is re-keyed.

**perturb**: FAIL on all 9 variants (the scorer's recomputation agrees with the VM's gate.env).
**stamps**: every unit NOT SCORED: NOT RUN (gate FAIL), with each gate's median and worst.

**hw** (`results/*-gpubench-r4/`, `results/*-rulers-r4/`): far L2 275.0 ns DIFFERS from RRZE's
253.4 (+8.5%, band 5%); triad 3781 GB/s CONSISTENT (-0.1%); L2 62,914,560 bytes CONSISTENT; DRAM
NOT SCORED (no cell). Near L2 152.4 ns (RRZE 141.6). Rulers: read2d 3815, copy 3673, add 3728
GB/s; matmul 689 TFLOP/s, with the SM clock read at 1395 MHz during it (the lock does not hold
under bf16 matmul at 700 W).

## Files

- `gh200-driver/status`, `gh200-driver/summary.txt`, `logs/driver-*.log`: the ledger and consoles;
  `gh200-driver/instr-variants.json` and `instr-gate.env`: the gate's variants and verdicts
- `results/gaps-nvidia_gh200_480gb-<model>-c{9,15}/private_weight_reference/`: the timed pages
  (units 5, 6, 24, 25), each c15 directory with `DECLARED_COPIES.txt` (NOT JOINABLE TO COUNTER
  BYTES); `locked-r3/` their lock ledgers
- `results/2026-10-06-nvidia_gh200_480gb-olmoe-1b-7b-<knob>-r3-counters/lock1710/`: the byte pages
  (`r3c-g64.json`, units 17 to 23) with `summary.json` and their profiles and Triton caches
- `results/2026-10-06-nvidia_gh200_480gb-perturb-pt/`: `perturb.json`, `gate.env`, the compared
  SASS per variant and config (`sass/<unit>/{plain,off,copy}-<config>.sass`) and the unit's own
  Triton cache (TTIR, TTGIR, LLIR, PTX, cubin of every plain and copy compile)
- `results/2026-10-06-nvidia_gh200_480gb-rulers-r4/rulers.json`
- `results/2026-10-06-nvidia_gh200_480gb-gpubench-r4/`: gpu-benches' printed outputs and
  `constants.json`, `gpubench-log.json` (the fetch, build and run steps); see
  `THIRD-PARTY-NOTICE.txt` there
- `census-olmoe-1b-7b.json`, `census*.profiles/`: the censuses written before the first page
- `driver580.log`, `setup_vm.log`, `gh200-driver.out`: the VM's driver install, setup and console

**Third-party outputs.** `results/2026-10-06-nvidia_gh200_480gb-gpubench-r4/gpu-latency.txt`,
`gpu-stream.txt` and `gpu-l2-cache.txt` are the printed outputs of RRZE-HPC/gpu-benches
(https://github.com/RRZE-HPC/gpu-benches) at commit 23e586dd8d8a19e72abd9d385a088623cefabfae,
GPL-3.0, built and run on the VM outside the checkout. They are measurements, attributed to that
tool; none of its source is in this repository. `gpubench-log.json` quotes the build's console
tail (compiler command lines and one compiler warning that names one source line).

The `card:` lines and every page's CARD block carry the card's UUID, as every published session
does; prose here names the board by `r3_timing_model.board()`. The Triton caches' compiled
`cuda_utils*.so` files (14) are left out, as in rentals 1 to 3; every other file of the run
branch is here, byte for byte (798 files against SHA256SUMS).
