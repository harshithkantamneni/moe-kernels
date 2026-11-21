# One of the study's four cards, the primary one: R3 under DRAM counters, the R3 timing chain and a locked-clock R3 ladder on one NVIDIA GH200 480GB (Lambda Cloud); every measurement here is this card's, never averaged with another card's

Every measurement in this directory was taken on the GH200 480GB, and no H200
measurement is set beside one here. The tooling's own text does quote H200
figures, and none of them is a reading of this card:
`alpha_g-nvidia_gh200_480gb-20260925T071107Z/logs/counter_plan.log` is a plan
page for card nvidia_h200 (ridge 151.43 FLOP/byte, 4378.0 GB/s, from
`measured_nvidia_h200.yaml`); both chain sessions' `chain-logs/preflight-r3.log`
(a synthetic plan, run before anything was measured), the dry run's R3 step
logs and its `followup-g1.price.log` print the H200 hypothesis bandwidth 4374.5
GB/s and ridge 160.30; and the preconditions logs, the thermal logs and
reports, and the R3 pages' duty and V7 text cite earlier H200 sessions.

Lambda Cloud instance 2211a2da1d134a398981e18ebc917ceb, type gpu_1x_gh200,
name moe-gh200, region us-east-3, $2.29/h, launched on 2026-09-25 under the
owner's explicit permission for one GH200. First boot at about 06:38 UTC;
terminated at 08:45 UTC, right after the exfil was copied and verified (about
2.1 h; at $2.29/h, about $4.9). The instance id, type, name, region, the
permission and the two instance times are from the owner's session; no file
here records them. The rate is also the `RATE` default in
`drivers/vm_autopilot.sh`, which passes it to the chain as `RATE_USD_H`.

Card: NVIDIA GH200 480GB, GPU-33948cb3-46ff-1834-07b0-6d05730ad309, sm_90, 132
SMs, 60 MiB L2 (`PREFLIGHT.json`), 97871 MiB (`setup_vm.log` S0), 1980 MHz
maximum SM clock from NVML (the chain session's `logs/thermal.log` and
`logs/calibrate.log`). Host: Ubuntu 22.04 on aarch64, 525 GB RAM, Lambda
stack 0.1.17~22.04.1 (its system torch 2.7.0 is recorded and never used)
(`setup_vm.log` S0); kernel 6.8.0-1013-nvidia-64k (`driver580.log`).
Venvs torch 2.13.0+cu130 and triton 3.7.1 (base), plus vLLM 0.27.1 (vllm),
Python 3.12.14, CUDA 13.0 (`setup_vm.log`). Nsight Compute 2025.3.1.0 (build
36398880) at `/usr/local/cuda-13.0/bin/ncu`, installed by `setup_vm.sh` S5 as
`cuda-nsight-compute-13-0` from NVIDIA's sbsa repository. Counter door `sudo`:
uid 1000 had CapEff 0000000000000000 (both S0 blocks of `setup_vm.log`);
through `sudo`, PF5's probe read CapEff 0x1ffffffffff, with CAP_SYS_ADMIN and
CAP_PERFMON set (`logs/pf5-probe.log`).

Driver: the instance shipped 570.148.08. By owner decision it was upgraded to
`nvidia-driver-580-server-open` 580.105.08 and rebooted before `setup_vm.sh`
ran; the decision and the reboot are from the owner's session. The upgrade
itself is logged: `driver580.log` is the apt console, which removes the
570.148.08 packages, fetches `nvidia-driver-580-server-open`
580.105.08-0lambda0.22.04.1 and its libraries from
`http://archive.lambdalabs.com/ubuntu jammy/main arm64`, builds the DKMS
modules for 6.8.0-1013-nvidia-64k and ends `EXIT 0`. Every S0 DETECT block in
`setup_vm.log` reads `580.105.08 (CUDA 13.0 per nvidia-smi)`.

Code: `setup_vm.log` S1 fetched the repository from the git bundle
`/home/ubuntu/moe3.bundle` (branch `gates-v2`) and checked out f49a213
detached, clean. Every page, report and calibration here names f49a213.

## What is here

Paths inside the logs are the VM's: `/home/ubuntu/moe/session/<x>` is `./<x>`
here, `/home/ubuntu/moe/results/<x>` is `../results/<x>`, and
`/home/ubuntu/moe/repo` was the checkout (its `results/calibration/<dir>` is
`../calibration/<dir>`). `/home/ubuntu/<file>` for the consoles is `./<file>`.

- `../KIND`: `session`. See "Why KIND is session" below.
- `../results/2026-09-25-nvidia_gh200_480gb-r3-counters/`: the run directory
  the counter run wrote. `r3c-g{1,2,4,16,64}.json` is one page per G;
  `r3c-g<G>.profiles/` holds that G's `g<G>.ncu-rep`, the raw `g<G>.csv`,
  `g<G>.capture.json` (ncu's argv, version and return code), `g<G>.plan.json`,
  `g<G>.manifest.json`, `g<G>.ncu.log` and the Triton cache; `summary.json` is
  the `--analyse` output over the five pages.
- `../results/gaps-nvidia_gh200_480gb/`: every run directory the timing arms
  wrote, 11 in all. `thermal_acceptance/` holds the chain's thermal probe.
  `private_weight_reference/` holds 10 R3 directories: the chain's three
  pages at the unlocked clock, the partial directory of the 1965 MHz lock
  attempt (its cells ran at 1785 to 1830 MHz), the chain's partial G=16
  directory, and the five 1710 MHz pages (see "The readings").
- `../calibration/`: the ruler the chain's calibrate measured and its run
  directory; NOT adopted, see its README.
- The VM's session root, `/home/ubuntu/moe/session`, whole:
  - `PREFLIGHT.txt`, `PREFLIGHT.json`: `setup_vm.sh`'s preflight (06:53:16Z,
    READY).
  - `probe.json` and `logs/pf5-probe.log`: PF5's route probe.
  - `census.json`, `census.profiles/` and `logs/pf6-census.log`: PF6's launch
    census.
  - `ncu.txt` (the ncu binary, its version and `--list-chips`),
    `ncu-query-metrics.txt`.
  - `logs/setup_runpod.log` (the venv build `setup_vm.sh` S4 ran),
    `logs/r3c-g<G>.log` (one per G), `logs/analyse.log`.
  - `alpha_g-nocard-20260925T065808Z/`: a dry run of the chain, whole (below).
  - `alpha_g-nvidia_gh200_480gb-20260925T071107Z/`: the measuring chain
    session, whole: `CHAIN.tsv` (the chain's ledger), `ARMS.tsv` (the
    preconditions driver's), `DEVICE`, `R3_DUTY`, `COUNTERS`, `COUNTERS.json`,
    `chain.lock` (it holds `12641 192-222-57-200`, a PID and the VM's
    hostname), `chain-logs/` (one log per step, plus each R3 step's
    `.price.log`), `logs/` (thermal, calibrate, pin_probe-n64-g1, and
    `counter_plan.log`, a plan page printed for card nvidia_h200 that
    measures nothing on this card) and `pin-probe-n64-g1/`.
  - `locked-r3/`: the 1965 MHz attempt, stopped: `status` (its ledger),
    `lgc.txt` and `rgc.txt` (the lock and reset consoles), `clocks-locked.txt`
    and `clocks-after-reset.txt` (`nvidia-smi -q` at the lock and after the
    reset), `r3-g1.log`.
  - `locked-r3-1710/`: the 1710 MHz ladder, the same files, with one
    `r3-g<G>.log` per G.
- The tarball's top-level files and `moe/alpha_g_chain.out`:
  - `setup_vm.log`: the `setup_vm.sh` run at f49a213, ending `EXIT 0`.
  - `driver580.log`: the driver upgrade's apt console.
  - `run_counters.status`: the counter run's ledger, one start and one end
    line per G. `run_counters.out`: its console, 0 bytes.
  - `autopilot.log`, `autopilot.state`: `vm_autopilot.sh`'s log and state.
  - `alpha_g_chain.out`: the measuring chain's console.
- `drivers/`: `vm_autopilot.sh`, `switch_to_locked.sh` and `locked_r3.sh`, the
  VM-side driver scripts. These are the laptop's copies of the three scripts
  that were copied to the VM and run there. No file here records when each was
  copied there, or the bytes the VM ran. `switch_to_locked.sh` and
  `locked_r3.sh` name the chain session opened at 07:11:07, so neither existed
  at launch, and `autopilot.log` records no stage change before 07:11:06. The
  tarball holds no copy of any of them, and none is byte-identical to a file in
  the repository at f49a213 (git hash-object against every blob of `git
  ls-tree -r f49a213`).
- `EXFIL-NOT-PUBLISHED.txt`: every tarball file that is not published here,
  one per line with its reason, and the tarball's sha256 and entry count.

Not here: the counter run's driver `~/run_counters.sh`, which
`drivers/vm_autopilot.sh`'s bytes stage names, and the first-stage
`/home/ubuntu/setup_vm.sh` that `setup_vm.log` names are not in the tarball.
`run_counters.status` has the form the A100's committed driver writes
(`../../2026-09-25-nvidia_a100_sxm4_40gb-r3-counters/session/run_counters.sh`);
no file here says whether the GH200's copy was the same file.

## What ran, in order

Times are UTC on 2026-09-25, from the logs, the ledgers and the pages; where a
step writes no time, the file mtimes the tarball carries.

1. **Driver upgrade** (`driver580.log`, last written 06:49). See above.
2. **`setup_vm.sh` at f49a213** (`setup_vm.log`). S1 checked the repository
   out of the bundle; S4 built the base and vllm venvs from
   `requirements/resolved-*.txt`; S5 installed ncu; S6 chose the `sudo`
   counter door; S7 wrote `env.sh`. S8, the preflight, at 06:53:16Z: PF1 to
   PF7 PASS, verdict READY (exit 0).
   - PF5: all 5 STRICT metrics of the r3-arms family read back as numbers on
     a profiled launch of the probe kernel (dram__bytes_read.sum 4.19942e+06);
     11 of 11 cross-check and recorded metrics proven; the CSV was WIDE.
   - PF6: CEN1 to CEN3 PASS; one `fused_experts` call makes exactly 2
     `fused_moe_kernel` launches (8 over 4 calls), every grid is the grid the
     child derived, and the child ran on this card.
   - PF7: "the card keeps 75.6 GB free beside the 25.8 GB footprint, so the
     save stays on the device".
3. **The counter run** (`run_counters.status`): `scripts/dram_counter_route.py
   --run --family r3-arms` one G at a time in the order 64, 1, 4, 2, 16, then
   `--analyse` over the five pages. Every page and `summary.json` names
   f49a213 with `git_dirty: false`, every `capture.json` names f49a213, and
   every ncu return code is 0. No `g<G>.ncu.log` carries a warning.

   | G | start | end | wall | exit |
   |---|---|---|---|---|
   | 64 | 06:53:33 | 06:56:59 | 3 min 26 s | 3 |
   | 1 | 06:56:59 | 07:00:24 | 3 min 25 s | 3 |
   | 4 | 07:00:24 | 07:03:49 | 3 min 25 s | 3 |
   | 2 | 07:03:49 | 07:07:14 | 3 min 25 s | 3 |
   | 16 | 07:07:14 | 07:10:38 | 3 min 24 s | 3 |
   | `--analyse` | | 07:10:40 | | 3 |

   From `g<G>.capture.json` and `g<G>.plan.json`: ncu's argv includes
   `--replay-mode kernel --cache-control all --clock-control base --nvtx -k
   regex:^fused_moe_kernel$ --launch-skip 60 --launch-count 90` over 16
   metrics, the child being `scripts/private_weight_reference.py
   --counter-child g<G>.plan.json` in the vllm venv. mixtral-8x7b, bf16,
   BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=4; arms native,
   shared and private at treads 1, 2, 3, 4 and 6 (15 cells); 3 measured calls
   per cell after 2 warmups; 9 copies declared.
4. **A dry run of the chain, `alpha_g-nocard-20260925T065808Z`** (files written
   06:58, while the G=1 counter page ran). It is `scripts/alpha_g_chain.sh
   --dry-run`, not a measuring attempt: its ledgers are `CHAIN-dryrun.tsv` and
   `ARMS-dryrun.tsv`; `chain-logs/preconditions.log` gives the card as
   `nocard` and says `no CUDA`; every R3 and R1 step (G 1, 2, 4, 16, 64) is
   REFUSED with rc 2, and each of those logs, like `logs/thermal.log`, ends
   "REFUSED. Nothing was measured and nothing was written. reason: --dry-run
   was given";
   `gpu-tests` only collected (38 tests); the counter probe and the probe
   check were SKIPPED as a dry run does. Its results path is
   `/home/ubuntu/moe/results/gaps-nocard`, the chain's default when
   `MOE_RESULTS_DIR` is unset. No file here records who started it or why the
   chain found no CUDA device.
5. **The measuring chain, `alpha_g-nvidia_gh200_480gb-20260925T071107Z`**
   (`alpha_g_chain.out`, console header `NEW`). `autopilot.log` holds one line,
   `2026-09-25T07:11:06Z stage -> chain`; `vm_autopilot.sh` appends a line at
   every stage change, so no earlier stage change was logged to this file.
   That stage sources `env.sh`, runs `unset MOE_RESULTS_DIR`, resets the
   clocks (`nvidia-smi -rgc`) and runs the chain with `G_LADDER="1 2 4 16
   64" SEEDS=0`. At f49a213 `scripts/setup_vm.sh` S7 writes `export
   MOE_RESULTS_DIR="$RESULTS"` (the results root, without the card) into
   `env.sh`, and `scripts/alpha_g_chain.sh` refuses a `MOE_RESULTS_DIR` that
   does not contain the card; no log here shows that refusal happening. The
   chain's results path is `/home/ubuntu/moe/results/gaps-nvidia_gh200_480gb`.
   Its console reads `nvidia-smi NVIDIA GH200 480GB, 580.105.08, 900.00 W`
   and the ladder "G in {1 2 4 16 64}, seeds {0}, ratio arms at duty 0.25".
   From `CHAIN.tsv` and `ARMS.tsv`:
   - preflight-r1, preflight-r3: DONE.
   - preconditions: DONE (189 s). The work tree had 0 dirty files before
     anything ran.
     - thermal DONE (152 s): SM clock 1440 MHz (min, median and max) over 55
       samples, first and last third 1440 MHz, 0.0% apart; temperature 58 to
       62 C; "board power median 652 W of a 900 W limit"; "slowdown reasons
       0x4 SwPowerCap [RECORDED, NOT SCORED]" (`logs/thermal.log`).
     - calibrate DONE (23 s): triad 3725.1 GB/s, dense bf16 662.8 TFLOP/s at a
       1455 MHz GEMM clock, ridge 177.9. It printed `PUBLISHED to
       /home/ubuntu/moe/repo/moe/bench/hardware/measured_nvidia_gh200_480gb.yaml`,
       and the work tree went from 0 to 1 dirty file; the `dirty` column of
       every later `CHAIN.tsv` row reads 1. See `../calibration/README.md`.
     - pin_probe-n64-g1 DONE (11 s): F1 and F2 PASS under a forced
       BLOCK_SIZE_M=128 tile.
   - counter-probe: INFO (rc 1, 4 s; informational, it gates nothing).
     BLOCKED at 07:14:27Z: `ERR_NVGPUCTRPERM`; ncu 2025.3.1.0 at
     `/usr/local/cuda-13.0/bin/ncu` on PATH; CapEff 0000000000000000, so
     CAP_PERFMON and CAP_SYS_ADMIN clear; module flag unread (`COUNTERS`).
     The counter run's steps had gone through the `sudo` door
     (`PREFLIGHT.txt`: `counter door sudo`); this probe's process held no
     capability.
   - gpu-tests (`tests/test_gpu.py` on the card): 37 passed, 1 skipped, 2
     warnings in 35.01 s.
   - probe-check: P1 PASS, the alignment probe under a CUDA graph at 5.22 us
     per call against an eager p50 of 45.63 us.
   - r3-g1-s0 (`78c2414e`, from 07:15:15, 688 s): INVALID (rc 3).
   - r3-g2-s0 (`308f0841`, from 07:26:45, 685 s): INVALID (rc 3).
   - r3-g4-s0 (`da469da0`, from 07:38:13, 697 s): INVALID (rc 3).
   - r3-g16-s0 (`32bec6b2`): started, then stopped by `switch_to_locked.sh`
     (next step). Its directory holds `DEVICE` only, and
     `chain-logs/r3-g16-s0.log` ends inside the plan it prints before
     measuring. `CHAIN.tsv` has no row for it, and no R1 step ran.
6. **The 1965 MHz attempt** (`locked-r3/status`, `drivers/switch_to_locked.sh`).
   - 07:43:39Z "waiting for r3-g4-s0" in the chain session.
   - 07:50:00Z "r3-g4-s0 done; stopping the chain session 10102".
   - 07:50:35Z "locked at 1965 MHz" (`lgc.txt`: `GPU clocks set to
     "(gpuClkMin 1965, gpuClkMax 1965)"`), then "G=1 start", running, under
     `timeout --signal=INT --kill-after=60 1800` in the vllm venv,
     `scripts/private_weight_reference.py --model mixtral-8x7b --block-m 32
     --treads 6 --repeats 9 --group-m 1 --duty 0.25 --seed 0 --session-tag
     alpha_g-nvidia_gh200_480gb-20260925T071107Z-lock1965`, into
     `...-257313aa`. `clocks-locked.txt`, taken in the same second, reads SM
     1830 MHz, not 1965 MHz, with the Idle event reason Active. No part of
     this run is a 1965 MHz measurement (see "Clocks and power").
   - `r3-g1.log` ends in a `KeyboardInterrupt` traceback raised in
     `time_duty`'s `sleep(gap)`. `rgc.txt` and `clocks-after-reset.txt`
     (nvidia-smi timestamp 07:52:11) are the reset `switch_to_locked.sh`
     runs on exit. `...-257313aa/cells.csv` holds 19 cells (all 18 of repeat
     0 and one of repeat 1) and no report.
   - 07:52:17Z "stopped the 1965 run (clock capped at ~1815 by the power
     cap)". That line is not one `switch_to_locked.sh` writes (it writes
     `locked at`, `G=<G> start`, `G=<G> rc=<rc>` and `clock reset; DONE`),
     and the ledger has no `G=1 rc=` line. No file supports the line's cause
     on its own: over the 110 s from the 1965 MHz lock to the 1710 MHz lock,
     the SW Power Capping and SW Thermal Slowdown counters both grew, by 98.2
     s and 96.6 s (see "Clocks and power").
7. **The 1710 MHz ladder** (`locked-r3-1710/status`, `drivers/locked_r3.sh`
   with `LOCK=1710`): the same command at each G with `--session-tag
   alpha_g-nvidia_gh200_480gb-20260925T071107Z-lock1710`.

   | G | start | end | wall | exit | directory |
   |---|---|---|---|---|---|
   | 1 | 07:52:25 | 08:03:00 | 10 min 35 s | 1 | `d9f1f37c` |
   | 2 | 08:03:00 | 08:13:33 | 10 min 33 s | 1 | `df37ea07` |
   | 4 | 08:13:33 | 08:24:05 | 10 min 32 s | 1 | `01c08abd` |
   | 16 | 08:24:05 | 08:34:40 | 10 min 35 s | 1 | `1b285de2` |
   | 64 | 08:34:40 | 08:45:16 | 10 min 36 s | 0 | `6ff34777` |

   `lgc.txt`: `GPU clocks set to "(gpuClkMin 1710, gpuClkMax 1710)"`.
   08:45:16Z "clock reset; DONE"; `autopilot.state` reads `done`.

## The readings

### R3 under DRAM counters (the bytes pages)

The alpha(G) table as `logs/analyse.log` prints it (lines 1 to 10). These
alphas are THIS card's, measured by DRAM counters on the GH200 480GB; they are
not the study's H200 alphas, and every page is INVALID.

```
CARD NVIDIA GH200 480GB (nvidia_gh200_480gb, UUID 33948cb3-46ff-1834-07b0-6d05730ad309, sm_90, 132 SMs, 60 MiB L2): every number here is THIS card's; the study's timing pages are nvidia_h200.
ALPHA(G) OVER 5 PAGES, one card, one commit, one vLLM, one design
  alpha lo/hi: the least and greatest OLS slope over SHARED's weight-only q, lowest call of SHARED or NATIVE less e to the highest,
  e PRIVATE's excess over n at its highest call; slope: q_S's K-call means', every activation re-read counted as a weight re-read
     G  alpha lo  alpha hi    slope       w1       w2    resid    ratio     diff  exit
     1    0.8365    0.8680   0.8533   0.9977   0.5646    4.92%   0.8531   0.8524  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
     2    0.4031    0.4089   0.4072   0.4193   0.3831   27.53%   0.4082   0.4048  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
     4    0.1760    0.1855   0.1842   0.1820   0.1887   43.55%   0.1857   0.1774  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
    16    0.0089    0.0707   0.0655   0.0438   0.1090   10.79%   0.0653   0.0149  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
    64   -0.0946    0.0763   0.0664  -0.0019   0.2031    3.08%   0.0605  -0.0827  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
```

All five pages are INVALID, so `--analyse` exited 3. The gates, as
`summary.json` and `logs/analyse.log` print them. V0 to V5, V8 and V9 PASS on
every page.

| G | exit | VALIDITY gates that FAIL | CLAIM gates |
|---|---|---|---|
| 1 | 3 INVALID | V6: n=2 w2 0.5152%; n=3 w1 0.6433%; n=3 w2 1.2173%; n=4 w2 0.9855%; and 1 more, against 0.5% | C2 PASS, C3 PASS, C6 PASS |
| 2 | 3 INVALID | V6: n=2 w1 5.5592%; n=2 w2 6.0334%; n=3 w1 0.7274%; n=3 w2 4.4230%; and 4 more | C1 PASS, C2 PASS, C6 PASS |
| 4 | 3 INVALID | V6: n=2 w1 5.4707%; n=2 w2 6.2233%; n=3 w1 4.5063%; n=3 w2 4.5537%; and 4 more | C1 PASS, C2 UNKNOWN (its GATE line prints REFUSE), C6 PASS |
| 16 | 3 INVALID | V6: n=2 w1 5.6660%; n=2 w2 5.9360%; n=3 w1 4.6628%; n=3 w2 4.5230%; and 4 more | C1 UNKNOWN (REFUSE), C2 UNKNOWN (REFUSE), C6 FAIL |
| 64 | 3 INVALID | V6: n=1 w1 1.2119%; n=1 w2 0.7565%; n=2 w1 4.5711%; n=2 w2 5.1389%; and 6 more; V7: n=2 w2 -8.9860% against 4.2107%, and more | C1 UNKNOWN (REFUSE), C2 FAIL, C6 FAIL |

V6 reads "the three arms request the same L2 read sectors", gate "max/min - 1
<= 0.5% across arms". These verdicts are the scorer's at f49a213. A change to
V6 for Hopper cards is under review on another branch and is not applied here.

The per-(G, arm, GEMM) q tables and each gate's measured values and reasons
follow the alpha table in `logs/analyse.log`; the same are in each
`logs/r3c-g<G>.log`, which also prints the activation working set of one column
pass per tread against this card's 60 MiB L2.

### R3 timing pages (`private_weight_reference`)

mixtral-8x7b, bf16, BLOCK_M=32, treads 1 to 6, 9 repeats, duty 0.25, seed 0.
The ratio is slope(shared) / slope(private) over treads 2 to 6, with its 90%
percentile bootstrap interval; C1's band is ALPHA_BAND [0.529, 0.588). C2 is
the private arm's delivered weight-read rate against the calibrated 3725.1
GB/s plus 10% (4097.6 GB/s); it PASSES on every page. An INVALID page's exit
line reads "nothing quotable"; its ratio is printed here as its page prints
it. From each `report.json` and the `exit` line of each log.

| G | clock | directory | UTC start | exit | gates that FAIL | ratio [90%] | C1 world | private delivered |
|---|---|---|---|---|---|---|---|---|
| 1 | chain, not locked | `78c2414e` | 07:15:15 | 3 INVALID | V7 (worst 1.55%, treads 3 and 6), C1 | 0.8929 [0.8917, 0.8946] | NO-REUSE | 3517.0 GB/s |
| 2 | chain, not locked | `308f0841` | 07:26:45 | 3 INVALID | V0 (122/162 cells, drift 24.7%), V7 (worst 3.15%, treads 2, 3 and 6), C1 | 0.6588 [0.6584, 0.6598] | ABOVE-THE-REFIT-BAND | 3522.6 GB/s |
| 4 | chain, not locked | `da469da0` | 07:38:13 | 3 INVALID | V0 (134/162 cells, drift 17.3%), V7 (worst 1.96%, treads 2 and 3), C1 | 0.6474 [0.6459, 0.6482] | ABOVE-THE-REFIT-BAND | 3490.1 GB/s |
| 1 | locked 1710 MHz | `d9f1f37c` | 07:52:28 | 1 CLAIM_FAIL | C1 (V0 to V8 PASS) | 0.9021 [0.9017, 0.9025] | NO-REUSE | 3513.3 GB/s |
| 2 | locked 1710 MHz | `df37ea07` | 08:03:03 | 1 CLAIM_FAIL | C1 (V0 to V8 PASS) | 0.6836 [0.6832, 0.6840] | ABOVE-THE-REFIT-BAND | 3516.4 GB/s |
| 4 | locked 1710 MHz | `01c08abd` | 08:13:37 | 1 CLAIM_FAIL | C1 (V0 to V8 PASS) | 0.6737 [0.6735, 0.6740] | ABOVE-THE-REFIT-BAND | 3481.8 GB/s |
| 16 | locked 1710 MHz | `1b285de2` | 08:24:09 | 1 CLAIM_FAIL | C1 (V0 to V8 PASS) | 0.6509 [0.6507, 0.6511] | ABOVE-THE-REFIT-BAND | 3377.4 GB/s |
| 64 | locked 1710 MHz | `6ff34777` | 08:34:44 | 0 DONE | none | 0.5849 [0.5843, 0.5852] | REFIT-CONFIRMED | 3049.5 GB/s |

V0's gate includes at least 3 repeats behind every tread per arm and DRIFT
under 20% of the timed cells at every tread. At G=2 V0 fails on both: native
had 2 repeats behind a tread (floor 3), and 24.7% of the 162 timed cells
drifted, over the 20% ceiling. At G=4 it fails on repeats alone: shared had 2
behind a tread, and its 17.3% drift is under the ceiling. V7's gate is shared
and private under-load clocks within 1% at every tread. On the three chain pages
the per-tread clocks V7 prints for shared and private run 1905 to 1965 MHz.
The V0 figures of the 1710 MHz pages: 159, 151, 155, 158 and 159 of
162 cells, drift 1.9%, 6.8%, 4.3%, 2.5% and 1.9%; V7 reads "worst 0.00%" on
each. V6 (shared and private within 2.0% at n=1) passes on all eight, at most
0.033%.

Two R3 directories are partial and have no report: `...-257313aa` (the 1965
MHz attempt, 19 cells) and `...-32bec6b2` (the chain's G=16 page, `DEVICE`
only).

The chain's console summarises its three pages as `INVALID (C1 alone)` with
"reads as: unresolved" (`alpha_g_chain.out`); the pages' own gates are those
in the table.

### Clocks and power

- The power limit: `setup_vm.log` prints `power limit 900 W (SXM)`, the
  chain console at 07:11 prints `900.00 W` from `nvidia-smi
  --query-gpu=...,power.limit`, the thermal report reads "a 900 W limit", and
  the calibration yaml records `power_limit_w: 900.0`. The two `nvidia-smi -q
  -d CLOCK,PERFORMANCE,POWER` dumps, `locked-r3/clocks-locked.txt` (07:50:35)
  and `locked-r3-1710/clocks-locked.txt` (07:52:25), read Current Power Limit
  700.00 W, Requested 700.00 W, Default 900.00 W, Max 900.00 W, and under
  Module Power Readings a Current Power Limit of 1000.00 W. No file records a
  command that set 700 W, and no file says what the module limit covers.
- The 1965 MHz lock attempt: the 19 cells in `...-257313aa/cells.csv` read
  per-cell median clocks of 1785 to 1830 MHz and single samples of 1575 to
  1830 MHz, at duty-averaged board power of 282 to 320 W (NVML's about 1 s
  average). The two dumps' cumulative Clocks Event Reasons Counters grew over
  the 110 s between them: SW Power Capping from 2159571509 us to 2257802115 us (98.2 s),
  SW Thermal Slowdown from 15460893 us to 112091711 us (96.6 s). No file
  records the slowdown-reason mask during this run; the 0x4 SwPowerCap mask
  above is the thermal probe's, at 07:11 to 07:13 under a dense GEMM.
- 1710 MHz: every cell median on the five pages is 1710 MHz except one cell of
  1695 MHz on `df37ea07`; single samples inside bursts go down to 1500 MHz
  (`df37ea07`) and 1515 to 1560 MHz on the others; duty-averaged board power
  267 to 312 W. `clocks-after-reset.txt` (08:45:16) reads 1830 MHz.

## Why KIND is session

The repository's KIND vocabulary has one word, `session`
(`moe/bench/published.py`, `SESSION` and `KIND_MARKER`). The docstring of
`SESSION` describes a raw session directory that holds the driver's ledger,
its per-arm logs and the run directories the arms wrote, kept so each verdict
printed can be re-derived, with no `measured.yaml` and no `merged.csv` of its
own. This directory is that: `run_counters.status`, `CHAIN.tsv`, `ARMS.tsv`
and the two `locked-r3*/status` files are the ledgers, `logs/`,
`chain-logs/` and `locked-r3*/` the logs, `../results/` the run directories,
and there is no `measured.yaml` or `merged.csv` at its top. With any other
word, `calibration_provenance` reads the directory as an arm and
`tests/test_docs.py` counts it as a published arm. With `session`, the
regenerated `results/published/CALIBRATION_PROVENANCE.md` lists it as
`session | n/a | n/a | n/a | **refused**`, as it lists sessions 4 to 6 and the
A100 counter run, with the line "rows carry 0 dtypes [], so there is no
single ridge; pass one"; on this branch (r3-align at f49a213 plus this
directory) its count reads 17 of 21.

## Provenance of these files

The exfil tarball `exfil.tgz`, sha256
`d2eb10a708e14eda50970dfa77e0a58bcbe8d8c7a1685c36e19a0ba5a787b58c` (also at
the top of `EXFIL-NOT-PUBLISHED.txt`), holds 6148 entries: 5035 files and 1113
directories. It was extracted fresh for this
publication. Its layout maps as `moe/results/` to `../results/`,
`moe/session/` to `./`, `moe/alpha_g_chain.out` and the six top-level files
(`setup_vm.log`, `driver580.log`, `run_counters.status`, `run_counters.out`,
`autopilot.log`, `autopilot.state`) to `./`, and three files under
`moe/repo/` to `../calibration/`.

405 of the 5035 files are published here, each compared by sha256 against the
fresh extraction: 405 identical, 0 different, 0 missing. The other 4630 are
listed in `EXFIL-NOT-PUBLISHED.txt` with their reasons:

- 4615 files under `moe/repo/`, the VM checkout's `moe/bench/hardware/` and
  `results/`, each byte-identical to its blob at f49a213 (git hash-object
  against `git ls-tree -r f49a213`). Only three files there are not in git at
  f49a213, and they are the ones in `../calibration/`.
- 15 Triton `cuda_utils.cpython-312-aarch64-linux-gnu.so` files, one per
  Triton cache (the five counter pages, the census, and the nine R3 run
  directories that compiled), which the repository's `.gitignore` drops by its
  `*.so` rule (its `results/published/` block re-includes `.ptx`, `.cubin`,
  `.ncu-rep`, `.nsys-rep` and `.qdrep`, not `.so`).

The six `.ncu-rep` files (five pages and the census) and the `.cubin` and
`.ptx` files in the Triton caches are tracked through that block. Beside the
405, this directory adds `../KIND`, this README, `../calibration/README.md`,
`EXFIL-NOT-PUBLISHED.txt` and the three scripts in `drivers/`. Every published
text file was searched for credentials (tokens, API keys, secrets,
Authorization and Bearer headers, private key blocks); none was found and no
file was withheld.
